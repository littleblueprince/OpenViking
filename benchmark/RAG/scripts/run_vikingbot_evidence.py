#!/usr/bin/env python3
"""Run one dataset QA ID or all adapter QAs with independent VikingBot processes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import multiprocessing
import os
import re
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import unicodedata
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RAG_ROOT = REPO_ROOT / "benchmark/RAG"
LOCAL = REPO_ROOT / ".local-data/syllabusqa"
sys.path.insert(0, str(RAG_ROOT / "src"))

ADAPTERS = {
    "syllabusqa": ("syllabusqa_adapter", "SyllabusQAAdapter", "id", "SyllabusQA/test.csv"),
    "financebench": ("financebench_adapter", "FinanceBenchAdapter", "financebench_id",
                     "FinanceBench/financebench_open_source.jsonl"),
    "qasper": ("qasper_adapter", "QasperAdapter", "question_id", "Qasper/qasper-test-v0.3.json"),
    "locomo": ("locomo_adapter", "LocomoAdapter", "original_id", "Locomo/locomo10.json"),
    "musique": ("musique_adapter", "MuSiQueAdapter", "id",
                "MuSiQue/musique_full_v1.0_dev_answerable_subset_120_seed42.jsonl"),
}
DISABLED_TOOLS = [
    "web_search", "web_fetch", "write_file", "edit_file", "exec", "message",
    "spawn", "wait_subagents", "cron", "generate_image", "openviking_add_resource",
    "openviking_memory_commit", "openviking_invoke_mcp", "openviking_export", "compile",
]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def select_cases(samples, id_key, case_id=None, source_doc_id=None):
    cases = []
    for sample in samples:
        source = str(sample.sample_id)
        if source_doc_id is not None and source != source_doc_id:
            continue
        for index, qa in enumerate(sample.qa_pairs):
            native_id = qa.metadata.get(id_key)
            canonical = f"{source}:qa_{index}"
            identifier = str(native_id) if native_id not in (None, "") else canonical
            if case_id is None or case_id in (identifier, canonical):
                cases.append({"case_id": identifier, "source_doc_id": source,
                              "canonical_id": canonical, "qa": qa})
    if not cases:
        raise ValueError("No matching adapter QAs; use --list-cases to inspect available IDs")
    if case_id is not None and len(cases) != 1:
        raise ValueError("Ambiguous QA ID; also specify --source-doc-id or use canonical_id")
    return cases


def filename_label(text, max_bytes):
    label = re.sub(r"[^\w.-]+", "-", unicodedata.normalize("NFKC", text)).strip("._-")
    return label.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore").rstrip("._-") or "case"


def case_directory_name(case):
    # SyllabusQA adds the document name to every question; show the actual query
    # in the directory preview rather than repeating that common prefix.
    query = re.sub(r'^Based on the syllabus "[^"]*",\s*', "", case["qa"].question)
    identity = json.dumps(
        [case["source_doc_id"], case["canonical_id"], case["case_id"]],
        ensure_ascii=False, separators=(",", ":"),
    )
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return "__".join((
        filename_label(case["case_id"], 64), filename_label(query.lower(), 80), suffix,
    ))


def attach_case_paths(cases, output):
    planned, seen = [], set()
    for case in cases:
        name = case_directory_name(case)
        if name in seen:
            raise ValueError(f"Duplicate case identity: {case['canonical_id']}")
        seen.add(name)
        directory = Path(output).resolve() / "cases" / name
        planned.append({**case, "question": case["qa"].question, "case_name": name,
                        "case_dir": str(directory), "result_path": str(directory / "result.json")})
    return planned


def write_case_index(output, run_id, cases):
    """Write searchable IDs, full questions and planned paths before workers start."""
    entries = [{key: case[key] for key in (
        "case_id", "canonical_id", "source_doc_id", "question", "case_name", "case_dir", "result_path",
    )} for case in cases]
    write_json(Path(output) / "index.json", {"run_id": run_id, "cases": entries})
    temporary = Path(output) / "index.tsv.tmp"
    fields = ("case_id", "question", "source_doc_id", "canonical_id", "case_dir", "result_path")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t", lineterminator="\n")
        writer.writerow(fields)
        for entry in entries:
            # One physical line per QA so rg returns its paths beside the query.
            writer.writerow([" ".join(entry[key].split()) for key in fields])
    temporary.replace(Path(output) / "index.tsv")


def load_source_map(path):
    """Accept explicit document-ID maps or existing SyllabusQA import reports."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "resources" not in data:
        mapping = data
    else:
        entries = data.get("resources", []) if isinstance(data, dict) else data
        if not isinstance(entries, list):
            raise ValueError("Source map must be a document-ID object or resource list")
        mapping = {}
        for entry in entries:
            source = entry.get("source_doc_id") or entry.get("syllabus_name")
            if not source and entry.get("file"):
                source = Path(entry["file"]).stem
            uri = entry.get("resource_uri") or entry.get("uri") or (
                entry.get("import_result") or {}
            ).get("root_uri")
            if source and uri:
                if source in mapping and mapping[source] != uri:
                    raise ValueError(f"Conflicting resource URIs for {source}")
                mapping[source] = uri
    if not all(isinstance(k, str) and isinstance(v, str)
               and v.startswith("viking://resources/") for k, v in mapping.items()):
        raise ValueError("Source mapping values must be viking://resources/... URIs")
    return {key: value.rstrip("/") for key, value in mapping.items()}


def expand_config(value):
    if isinstance(value, dict):
        return {key: expand_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_config(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def server_settings(config):
    server = config.get("server", {})
    bot_server = config.get("bot", {}).get("ov_server", {})
    url = bot_server.get("server_url") or (
        f"http://{server.get('host', '127.0.0.1')}:{server.get('port', 1933)}"
    )
    headers = {
        "X-OpenViking-Account": bot_server.get("account_id") or "default",
        "X-OpenViking-User": bot_server.get("admin_user_id") or "default",
    }
    key = bot_server.get("api_key") or server.get("root_api_key")
    if key:
        headers["X-API-Key"] = key
    return url.rstrip("/"), headers


def snapshot_resource(config, uri, source_doc_id, chunk_source_map=None):
    url, headers = server_settings(config)
    by_uri = {}
    if chunk_source_map is not None:
        root = uri.rstrip("/") + "/"
        for source, target in chunk_source_map.items():
            if not target.startswith(root):
                raise ValueError(f"Chunk {source} is outside corpus root {uri}")
            if target in by_uri and by_uri[target] != source:
                raise ValueError(f"Multiple chunk IDs map to {target}")
            by_uri[target] = source

    def get(endpoint, **params):
        target = url + endpoint
        if params:
            target += "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(
            urllib.request.Request(target, headers=headers), timeout=30
        ) as response:
            data = json.load(response)
        if data.get("status") != "ok":
            raise RuntimeError(f"OpenViking read failed: {endpoint}")
        return data

    chunks, offset, seen = [], 0, set()
    while True:
        page = get("/api/v1/fs/ls", uri=uri, recursive="true", node_limit=1000, offset=offset)
        entries = page.get("result")
        if not isinstance(entries, list):
            raise ValueError("Expected a structured OpenViking resource listing")
        new_entries = 0
        for entry in entries:
            chunk_uri = entry.get("uri", "")
            if chunk_uri in seen:
                continue
            seen.add(chunk_uri)
            new_entries += 1
            if entry.get("isDir") or not chunk_uri.endswith(".md") or (
                chunk_uri.rsplit("/", 1)[-1].startswith(".")
            ):
                continue
            text = get("/api/v1/content/read", uri=chunk_uri).get("result")
            if not isinstance(text, str):
                raise ValueError(f"Expected full L2 text at {chunk_uri}")
            identity = source_doc_id
            if chunk_source_map is not None:
                ancestor = chunk_uri
                while ancestor not in by_uri and ancestor.startswith(uri.rstrip("/") + "/"):
                    ancestor = ancestor.rsplit("/", 1)[0]
                identity = by_uri.get(ancestor)
                if identity is None:
                    raise ValueError(f"Unmapped L2 chunk in shared corpus: {chunk_uri}")
            chunks.append({"uri": chunk_uri, "source_doc_id": identity, "text": text})
        if not page.get("has_more", False):
            break
        if not new_entries:
            raise ValueError("Resource pagination made no progress")
        offset += len(entries)
    if not chunks:
        raise ValueError(f"No loaded L2 chunks for document {source_doc_id} at {uri}")
    if chunk_source_map is not None:
        missing = set(chunk_source_map) - {chunk["source_doc_id"] for chunk in chunks}
        if missing:
            raise ValueError(f"Missing {len(missing)} mapped chunks in corpus snapshot")
    return chunks


def runtime_config(original, runtime):
    """Give every QA a separate local workspace and server-side user identity."""
    config = {**original, "storage": {
        **original.get("storage", {}), "workspace": str(Path(runtime) / "data"),
    }}
    bot = dict(original.get("bot", {}))
    bot.pop("rag_checkpoints", None)
    bot["mode"] = "readonly"
    bot["heartbeat"] = {**bot.get("heartbeat", {}), "enabled": False}
    bot["ov_server"] = {
        **bot.get("ov_server", {}), "server_url": server_settings(original)[0],
        "admin_user_id": f"rag-eval-{uuid.uuid4().hex}",
    }
    config["bot"] = bot
    return config


def judge_settings(config, args):
    vlm = config.get("vlm", {})
    provider = vlm.get("provider", "openai")
    if provider not in ("openai", "volcengine") and not args.judge_base_url:
        raise ValueError("This VLM provider requires --judge-base-url for an OpenAI-compatible judge")
    model = args.judge_model or vlm.get("model")
    if not model:
        raise ValueError("Missing judge model; set vlm.model or --judge-model")
    return {
        "model": model,
        "base_url": args.judge_base_url or vlm.get("api_base") or vlm.get("base_url") or (
            "https://ark.cn-beijing.volces.com/api/v3" if provider == "volcengine"
            else "https://api.openai.com/v1"
        ),
        "api_key": vlm.get("api_key") or os.environ.get("OPENAI_API_KEY"),
        "timeout": args.judge_timeout,
    }


def evaluate_answer(result, dataset, settings):
    """Use the existing dataset rubric; invocation/parse failures never become zero scores."""
    if settings is None:
        return {"status": "skipped", "reason": "disabled"}
    if result.get("status") != "completed":
        return {"status": "skipped", "reason": "agent_failed"}
    if not result.get("gold_answers"):
        return {"status": "skipped", "reason": "no_gold_answers"}
    evaluation = {"status": "failed", "model": settings["model"],
                  "score": None, "normalized_score": None}
    started = time.monotonic()

    class RecordingClient:
        response = None

        def invoke(self, messages):
            self.response = client.invoke(messages)
            return self.response

    recording = RecordingClient()
    try:
        from langchain_openai import ChatOpenAI
        from core.judge_util import llm_grader

        client = ChatOpenAI(
            model=settings["model"], temperature=0, base_url=settings["base_url"],
            api_key=settings["api_key"], timeout=settings["timeout"], max_retries=2,
        )
        grade = llm_grader(
            recording, settings["model"], result["question"], result["gold_answers"],
            result["answer"], dataset_name=dataset, strict=True,
        )
        evaluation.update(grade, status="completed", normalized_score=grade["score"] / 4)
    except Exception as exc:
        evaluation["error"] = str(exc)
    finally:
        evaluation["elapsed_seconds"] = time.monotonic() - started
        if recording.response is not None:
            evaluation["raw_output"] = recording.response.content
            evaluation["token_usage"] = getattr(recording.response, "usage_metadata", None)
    return evaluation


def aggregate_judges(results):
    evaluations = [result.get("llm_evaluation", {}) for result in results]
    completed = [item for item in evaluations if item.get("status") == "completed"]
    return {
        "completed": len(completed),
        "failed": sum(item.get("status") == "failed" for item in evaluations),
        "skipped": sum(item.get("status") == "skipped" for item in evaluations),
        "average_score": sum(item["score"] for item in completed) / len(completed) if completed else None,
        "average_normalized_score": (
            sum(item["normalized_score"] for item in completed) / len(completed) if completed else None
        ),
    }


def execute_case(task):
    from core.logger import setup_logging
    from core.readable_trace import format_step_log
    from core.trace_models import ChunkDocument, StepEvidenceRecord
    from vikingbot_runner import VikingBotRunner

    output = Path(task["output"])
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    try:
        chunks = [ChunkDocument(**chunk) for chunk in json.loads(
            Path(task["snapshot"]).read_text(encoding="utf-8")
        )]
        with tempfile.TemporaryDirectory(prefix="vikingbot-rag-") as runtime:
            config_path = Path(runtime) / "ov.conf"
            write_json(config_path, runtime_config(task["config"], runtime))
            config_path.chmod(0o600)
            # Keep any explicitly configured CLI authentication available.
            cli_path = Path(task["ov_conf"]).with_name("ovcli.conf")
            if cli_path.is_file():
                os.environ["OPENVIKING_CLI_CONFIG_FILE"] = str(cli_path)
            setup_logging(str(output / "benchmark.log"))
            runner = VikingBotRunner({"vikingbot": {
                "ov_conf": str(config_path), "manage_server": False,
                "max_iterations": task["max_iterations"], "timeout_seconds": task["timeout"],
                "evidence_preview_chars": task["preview_chars"],
                "prompt": (
                    "Use only the source corpus under {resource_uri}. "
                    "Read the relevant source text before answering. Answer briefly.\n\n"
                    "Question: {question}"
                ),
            }})
            result = runner.generate_qa(
                task["qa"], chunks, case_id=task["case_id"], run_id=task["run_id"],
                trace_dir=output, resource_uri=task["resource_uri"],
                disabled_tools=DISABLED_TOOLS,
            )
        trace = Path(result["evidence_trace_path"])
        records = [StepEvidenceRecord.from_dict(json.loads(line))
                   for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()]
        log_path = output / "steps.readable.log"
        log_path.write_text("\n\n".join(
            format_step_log(record, preview_chars=task["preview_chars"]) for record in records
        ) + "\n", encoding="utf-8")
        result["evidence_readable_log_path"] = str(log_path)
        if result.get("error") or result.get("answer", "").startswith(("Error calling LLM", "[ERROR]")):
            result["status"] = "failed"
            result["failure_reason"] = "agent_or_model_error"
        elif not records:
            result["status"] = "failed"
            result["failure_reason"] = "no_tool_evidence_trace"
        else:
            result["status"] = "completed"
    except Exception as exc:
        result.update(status="failed", error=str(exc), failure_reason="execution_error")
    result.update({
        key: task[key] for key in ("case_id", "canonical_id", "source_doc_id",
                                  "resource_uri", "run_id", "snapshot")
    })
    result.update(question=task["qa"].question, case_dir=str(output),
                  gold_answers=task["qa"].gold_answers,
                  worker_pid=os.getpid())
    # Persist the Agent output before the separate judge model call.
    write_json(output / "result.json", result)
    result["llm_evaluation"] = evaluate_answer(result, task.get("dataset", "unknown"), task.get("judge"))
    write_json(output / "result.json", result)
    return {key: result.get(key) for key in (
        "case_id", "canonical_id", "source_doc_id", "question", "case_dir",
        "status", "failure_reason", "error",
        "worker_pid", "evidence_summary", "iterations_used", "token_usage", "elapsed_seconds",
        "llm_evaluation",
    )} | {"result_path": str(output / "result.json")}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=str.lower, choices=ADAPTERS)
    parser.add_argument("--data", type=Path, help="Dataset file/directory; default: test split")
    parser.add_argument("--chunks", type=Path, help="MuSiQue shared chunks JSONL")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--case-id", "--id", help="Native QA ID or source_doc_id:qa_INDEX (zero-based)")
    mode.add_argument("--all", action="store_true", help="Run all QAs returned by the adapter")
    mode.add_argument("--list-cases", action="store_true", help="Print IDs without model/server calls")
    parser.add_argument("--source-doc-id", help="Restrict selection to one source document")
    parser.add_argument("--workers", type=int, default=1, help="Independent processes (default: 1)")
    parser.add_argument("--limit", type=int, help="Run only the first N selected adapter QAs")
    parser.add_argument("--no-judge", action="store_true", help="Skip answer grading (enabled by default)")
    parser.add_argument("--judge-model", help="Judge model; default: vlm.model")
    parser.add_argument("--judge-base-url", help="OpenAI-compatible judge endpoint; default: vlm.api_base")
    parser.add_argument("--judge-timeout", type=float, default=60, help="Seconds per judge request")
    parser.add_argument("--ov-conf", type=Path, default=(
        LOCAL / "ov.conf" if (LOCAL / "ov.conf").exists() else RAG_ROOT / "ov.conf"
    ))
    parser.add_argument("--ark-key-file", type=Path, help="Load ARK_API_KEY if unset")
    parser.add_argument("--source-map", type=Path,
                        help="JSON document-ID -> resource-URI map or SyllabusQA import report")
    parser.add_argument("--resource-uri", help="Explicit resource root for one selected document")
    parser.add_argument("--output-dir", type=Path, help="Parent directory for unique run directories")
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180, help="Seconds per agent turn")
    parser.add_argument("--preview-chars", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true", help="Save plan and full snapshots; no model calls")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.workers < 1 or args.max_iterations < 1 or args.timeout <= 0 or args.preview_chars < 1:
        parser.error("workers, max-iterations, timeout and preview-chars must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if args.judge_timeout <= 0:
        parser.error("judge-timeout must be positive")
    module, class_name, id_key, default_data = ADAPTERS[args.dataset]
    if args.chunks and args.dataset != "musique":
        parser.error("--chunks is only supported for musique")
    adapter_kwargs = {"chunks_file": str(args.chunks.resolve())} if args.chunks else {}
    adapter = getattr(importlib.import_module(f"adapters.{module}"), class_name)(
        str((args.data or RAG_ROOT / "datasets" / default_data).resolve()), **adapter_kwargs
    )
    cases = select_cases(adapter.load_and_transform(), id_key, args.case_id, args.source_doc_id)
    if args.limit is not None:
        cases = cases[:args.limit]
    if args.list_cases:
        for case in cases:
            print(json.dumps({key: value for key, value in case.items() if key != "qa"} |
                             {"question": case["qa"].question},
                             ensure_ascii=False))
        return 0
    source_map = args.source_map
    if source_map is None and args.dataset == "syllabusqa":
        candidate = LOCAL / "syllabusqa-test-import.json"
        if candidate.exists():
            source_map = candidate
    mapping = load_source_map(source_map) if source_map else {}
    sources = sorted({case["source_doc_id"] for case in cases})
    if args.resource_uri:
        if len(sources) != 1 or not args.resource_uri.startswith("viking://resources/"):
            raise ValueError("--resource-uri requires exactly one selected document and a resource URI")
        mapping[sources[0]] = args.resource_uri.rstrip("/")
    missing = [source for source in sources if source not in mapping]
    if missing:
        raise ValueError("Missing source mappings (no model calls made): " + json.dumps(
            missing, ensure_ascii=False
        ) + "; supply --source-map or --resource-uri")
    chunk_mapping = None
    if args.dataset == "musique":
        missing_chunks = set(adapter.chunks) - set(mapping)
        if missing_chunks:
            raise ValueError(f"Missing {len(missing_chunks)} MuSiQue chunk mappings; "
                             "use prepare_musique.py --ingest or --verify-only")
        chunk_mapping = {sid: mapping[sid] for sid in adapter.chunks}
    if args.ark_key_file and not os.environ.get("ARK_API_KEY"):
        os.environ["ARK_API_KEY"] = args.ark_key_file.read_text(encoding="utf-8").strip()
    config = expand_config(json.loads(args.ov_conf.read_text(encoding="utf-8")))
    key = config.get("vlm", {}).get("api_key", "")
    if not args.dry_run and isinstance(key, str) and "$" in key:
        raise ValueError("Model credential environment variable is unset; use --ark-key-file")
    judge = None if args.no_judge or args.dry_run else judge_settings(config, args)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output = (args.output_dir or RAG_ROOT / "Output" / "vikingbot" / args.dataset).resolve() / run_id
    output.mkdir(parents=True)
    snapshots = output / "snapshots"
    snapshots.mkdir()
    cases = attach_case_paths(cases, output)
    plan = {
        "run_id": run_id, "dataset": args.dataset, "data": str(args.data or RAG_ROOT / "datasets" / default_data),
        "workers": args.workers, "limit": args.limit, "total_cases": len(cases), "dry_run": args.dry_run,
        "max_iterations": args.max_iterations, "timeout": args.timeout,
        "preview_chars": args.preview_chars,
        "chunks": str(adapter.chunks_file.resolve()) if args.dataset == "musique" else None,
        "judge_enabled": not args.no_judge, "judge_model": (
            args.judge_model or config.get("vlm", {}).get("model")
        ), "judge_timeout": args.judge_timeout,
        "cases": [{key: value for key, value in case.items() if key != "qa"} |
                  {"resource_uri": mapping[case["source_doc_id"]]} for case in cases],
    }
    write_json(output / "plan.json", plan)
    write_case_index(output, run_id, cases)
    print(f"Run: {output}\nSelected {len(cases)} QAs, {len(sources)} documents, {args.workers} workers", flush=True)
    print(f"Case lookup: {output / 'index.tsv'}", flush=True)
    snapshot_paths = {}
    for source in sources:
        path = snapshots / (hashlib.sha256(source.encode()).hexdigest()[:20] + ".json")
        write_json(path, snapshot_resource(config, mapping[source], source, chunk_mapping)
                   if chunk_mapping is not None else snapshot_resource(config, mapping[source], source))
        snapshot_paths[source] = str(path)
    if args.dry_run:
        print("Dry run completed: plan.json and snapshots saved; no model calls.", flush=True)
        return 0
    tasks = [
        {**case, "config": config, "ov_conf": str(args.ov_conf.resolve()),
         "dataset": args.dataset, "judge": judge,
         "resource_uri": mapping[case["source_doc_id"]], "run_id": run_id,
         "snapshot": snapshot_paths[case["source_doc_id"]],
         "output": case["case_dir"],
         "max_iterations": args.max_iterations, "timeout": args.timeout,
         "preview_chars": args.preview_chars}
        for case in cases
    ]
    summary = {"run_id": run_id, "total_cases": len(tasks), "completed": 0, "failed": 0,
               "status": "running", "results": [], "llm_judge": aggregate_judges([])}
    write_json(output / "summary.json", summary)
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = {executor.submit(execute_case, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {key: task[key] for key in ("case_id", "canonical_id", "source_doc_id")}
                result.update(question=task["qa"].question, case_dir=task["output"])
                result.update(status="failed", error=str(exc), failure_reason="worker_error")
                case_output = Path(task["output"])
                case_output.mkdir(parents=True, exist_ok=True)
                result_path = case_output / "result.json"
                # A worker may have persisted its Agent output before failing in judging.
                if result_path.exists():
                    stored = json.loads(result_path.read_text(encoding="utf-8"))
                    stored["llm_evaluation"] = {
                        "status": "failed", "error": str(exc),
                        "score": None, "normalized_score": None,
                    }
                    write_json(result_path, stored)
                    result.update(status=stored["status"], llm_evaluation=stored["llm_evaluation"])
                else:
                    result["llm_evaluation"] = {"status": "skipped", "reason": "worker_error"}
                    write_json(result_path, result)
                result["result_path"] = str(case_output / "result.json")
            summary["results"].append(result)
            summary[result["status"]] += 1
            summary["llm_judge"] = aggregate_judges(summary["results"])
            write_json(output / "summary.json", summary)
            evaluation = result.get("llm_evaluation", {})
            print(
                f"[{len(summary['results'])}/{len(tasks)}] {result['case_id']}: {result['status']}; "
                f"judge={evaluation.get('status')}, score={evaluation.get('score')}", flush=True,
            )
    summary["status"] = "failed" if summary["failed"] or summary["llm_judge"]["failed"] else "completed"
    write_json(output / "summary.json", summary)
    print(f"Summary: {output / 'summary.json'}", flush=True)
    return 1 if summary["status"] == "failed" else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
