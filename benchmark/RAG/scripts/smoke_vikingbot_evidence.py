#!/usr/bin/env python3
"""Run one existing SyllabusQA case through VikingBotRunner and save its evidence trace."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "benchmark/RAG/src"))
sys.path.insert(0, str(REPO_ROOT / "bot"))

from adapters.syllabusqa_adapter import SyllabusQAAdapter
from core.evidence_matcher import EvidenceMatcher
from core.logger import setup_logging
from core.trace_models import ChunkDocument
from vikingbot_runner import VikingBotRunner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ov-conf", type=Path, required=True)
    parser.add_argument("--ark-key-file", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-doc-id", required=True)
    parser.add_argument("--resource-uri", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--preview-chars", type=int, default=500)
    args = parser.parse_args()
    if not os.environ.get("ARK_API_KEY") and args.ark_key_file is not None:
        os.environ["ARK_API_KEY"] = args.ark_key_file.read_text().strip()
    original = json.loads(os.path.expandvars(args.ov_conf.resolve().read_text()))
    key = original.get("vlm", {}).get("api_key")
    if isinstance(key, str) and key.startswith("$"):
        raise ValueError("Model credential environment variable is unset")
    server = original.get("server", {})
    bot_server = original.get("bot", {}).get("ov_server", {})
    server_url = bot_server.get("server_url") or (
        f"http://{server.get('host', '127.0.0.1')}:{server.get('port', 1933)}"
    )
    headers = {}
    api_key = bot_server.get("api_key") or server.get("root_api_key")
    if api_key:
        headers["X-API-Key"] = api_key
    headers["X-OpenViking-Account"] = bot_server.get("account_id", "default")
    headers["X-OpenViking-User"] = bot_server.get("admin_user_id", "default")

    def get(path: str, **params):
        url = server_url.rstrip("/") + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=30
        ) as response:
            data = json.load(response)
        if data.get("status") != "ok":
            raise RuntimeError(f"OpenViking read failed: {path}")
        return data

    get("/health")
    sample = next(
        sample for sample in SyllabusQAAdapter(str(args.dataset)).load_and_transform()
        if sample.sample_id == args.source_doc_id
    )
    qa = next(qa for qa in sample.qa_pairs if str(qa.metadata.get("id")) == args.case_id)
    entries = []
    offset = 0
    while True:
        page = get(
            "/api/v1/fs/ls", uri=args.resource_uri, recursive="true",
            node_limit=1000, offset=offset,
        )
        batch = page.get("result")
        if not isinstance(batch, list):
            raise ValueError("Expected a structured resource listing")
        entries.extend(batch)
        if not page.get("has_more", False):
            break
        if not batch:
            raise ValueError("Resource listing has_more without entries")
        offset += len(batch)
    chunks = [
        ChunkDocument(
            entry["uri"], sample.sample_id,
            get("/api/v1/content/read", uri=entry["uri"])["result"],
        )
        for entry in entries
        if not entry.get("isDir") and entry["uri"].endswith(".md")
        and not entry["uri"].rsplit("/", 1)[-1].startswith(".")
    ]
    if not chunks:
        raise ValueError("No loaded L2 text chunks under the supplied resource URI")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output = args.output_dir.resolve() / run_id
    output.mkdir(parents=True)
    (output / "snapshot.json").write_text(
        json.dumps([{
            "uri": chunk.uri, "source_doc_id": chunk.source_doc_id, "text": chunk.text,
        } for chunk in chunks], ensure_ascii=False, indent=2), encoding="utf-8",
    )
    matcher = EvidenceMatcher(qa.gold_evidence, chunks, snapshot_id=run_id)
    print(json.dumps({
        "stage": "snapshot_ready", "case_id": args.case_id, "question": qa.question,
        "chunks": len(chunks),
        "unresolved_evidence_ids": matcher.manifest.unresolved_evidence_ids,
        "output_dir": str(output),
    }, ensure_ascii=False), flush=True)

    with tempfile.TemporaryDirectory(prefix="vikingbot-evidence-smoke-") as runtime:
        runtime = Path(runtime)
        config = dict(original)
        config["storage"] = {**original.get("storage", {}), "workspace": str(runtime / "data")}
        bot = dict(original.get("bot", {}))
        # The existing service may have been started using another branch.
        removed_fields = []
        if "rag_checkpoints" in bot:
            bot.pop("rag_checkpoints")
            removed_fields.append("bot.rag_checkpoints")
        bot["mode"] = "readonly"
        bot["heartbeat"] = {**bot.get("heartbeat", {}), "enabled": False}
        bot["ov_server"] = {
            **bot_server, "server_url": server_url,
            "admin_user_id": f"evidence-smoke-{uuid.uuid4().hex[:12]}",
        }
        config["bot"] = bot
        config_path = runtime / "ov.conf"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        config_path.chmod(0o600)
        setup_logging(str(output / "benchmark.log"))
        runner = VikingBotRunner({"vikingbot": {
            "ov_conf": str(config_path), "manage_server": False,
            "max_iterations": args.max_iterations, "timeout_seconds": args.timeout,
            "evidence_preview_chars": args.preview_chars,
            "base_instruction": (
                "Answer briefly using only the specified syllabus resource. "
                "Read the relevant source file before answering."
            ),
        }})
        result = runner.generate_qa(
            qa, chunks, case_id=args.case_id, run_id=run_id, trace_dir=output,
            resource_uri=args.resource_uri,
            disabled_tools=[
                "web_search", "web_fetch", "write_file", "edit_file", "exec",
                "message", "spawn", "wait_subagents", "cron", "generate_image",
                "openviking_add_resource", "openviking_memory_commit",
                "openviking_invoke_mcp", "openviking_export", "compile",
            ],
        )
    result.update({
        "case_id": args.case_id, "run_id": run_id, "question": qa.question,
        "gold_answers": qa.gold_answers, "source_doc_id": sample.sample_id,
        "resource_uri": args.resource_uri, "snapshot_chunk_count": len(chunks),
        "runtime_config_removed_fields": removed_fields,
    })
    tool_results = [event for event in result.get("events", []) if event["event"] == "tool_result"]
    if result.get("error") or result["answer"].startswith(("Error calling LLM", "[ERROR]")):
        result["smoke_status"] = "failed"
        result["smoke_failure_reason"] = "agent_or_model_error"
    elif not tool_results or not Path(result["evidence_trace_path"]).stat().st_size:
        result["smoke_status"] = "failed"
        result["smoke_failure_reason"] = "no_tool_evidence_trace"
    else:
        result["smoke_status"] = "passed"
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({
        "stage": "completed", "answer": result["answer"], "error": result.get("error"),
        "smoke_status": result["smoke_status"],
        "iterations": result.get("iterations_used"),
        "evidence_summary": result.get("evidence_summary"),
        "result_path": str(output / "result.json"),
    }, ensure_ascii=False), flush=True)
    return 0 if result["smoke_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
