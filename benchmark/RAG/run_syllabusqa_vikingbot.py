#!/usr/bin/env python3
"""Run the SyllabusQA test split through VikingBot and score the recorded answers."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
from queue import Queue
import re
from statistics import fmean
import time
from uuid import uuid4

import httpx
from langchain_openai import ChatOpenAI

from src.core.judge_util import llm_grader
from src.core.metrics import MetricsCalculator


RAG_DIR = Path(__file__).resolve().parent
LOCAL_CONFIG = RAG_DIR.parents[1] / ".local-data/syllabusqa"
DISABLED_TOOLS = (
    "web_search", "web_fetch", "exec", "spawn", "generate_image", "read_file",
    "list_dir", "write_file", "edit_file", "openviking_export", "openviking_add_resource",
    "openviking_memory_commit", "openviking_find_mcp_tools", "openviking_invoke_mcp_tool",
    "openviking_rule_search",
)
SOURCE_LINK = re.compile(r"\[([^\]\n]*)\]\(viking://[^)\n]+\)")
SOURCE_URI = re.compile(r"viking://[^\s<>\]\[)\"'`]+")
NO_ANSWER = {"not mentioned", "noinsufficient information"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-csv", type=Path, default=RAG_DIR / "datasets/SyllabusQA/test.csv")
    parser.add_argument("--imports", type=Path, default=LOCAL_CONFIG / "syllabusqa-test-import.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:11933")
    parser.add_argument("--bot-url", action="append", help="A native VikingBot gateway URL; repeat for parallel workers.")
    parser.add_argument("--key-file", type=Path, default=LOCAL_CONFIG / "ark-api-key")
    parser.add_argument("--judge-base-url", default="https://ark.cn-beijing.volces.com/api/v3")
    parser.add_argument("--judge-model", default="doubao-seed-2-0-pro-260215")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--judge-workers", type=int, default=6)
    parser.add_argument("--phase", choices=("all", "generate", "score"), default="all")
    parser.add_argument("--limit", type=int, help="Process the first N cases. Reuse the output to resume.")
    args = parser.parse_args()
    if min(args.workers, args.judge_workers) < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Worker counts and --limit must be positive")
    return args


def atomic_json(path, data, *, lines=False):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        if lines:
            for row in data:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    tmp.replace(path)


def load_cases(args):
    imports = json.loads(args.imports.read_text(encoding="utf-8"))["resources"]
    uris = {
        Path(entry["file"]).stem: entry["uri"] for entry in imports if entry["status"] == "verified"
    }
    with args.test_csv.open(encoding="utf-8-sig", newline="") as handle:
        cases = [
            {"id": row["id"], "syllabus": row["syllabus_name"], "type": row["question_type"],
             "question": row["question"].strip(), "gold": row["answer"].strip()}
            for row in csv.DictReader(handle)
        ]
    if set(uris) != {case["syllabus"] for case in cases}:
        raise ValueError("The import manifest must cover exactly the syllabi in the test split")
    if len({case["id"] for case in cases}) != len(cases) or any(
        not re.fullmatch(r"[A-Za-z0-9_-]+", c["id"]) or not c["question"] or not c["gold"] for c in cases
    ):
        raise ValueError("Test cases must have unique file-safe IDs, questions and answers")
    for case in cases:
        case["resource_uri"] = uris[case["syllabus"]]
    return cases


def prompt_for(case):
    return (
        f"Answer the following question in concise English using only the course syllabus under "
        f"{case['resource_uri']}. Cite supporting viking:// sources separately. "
        'If the syllabus contains insufficient information, answer "Not mentioned".\n\n'
        + case["question"]
    )


def tool_call(data, elapsed):
    name, separator, arguments = str(data).partition("(")
    call = {"tool": name, "elapsed_seconds": elapsed}
    if separator and arguments.endswith(")"):
        try:
            params = json.loads(arguments[:-1])
        except json.JSONDecodeError:
            return call
        if isinstance(params, dict):
            call.update({key: params[key] for key in ("query", "pattern", "target_uri", "uri", "uris")
                         if key in params})
    return call


def ask_bot(client, args, run_id, case, bot_url):
    session = str(uuid4())
    trace = args.output / "traces_detailed" / f"{case['id']}.{session}.jsonl"
    payload = {"message": prompt_for(case), "session_id": session, "user_id": run_id,
               "disabled_tools": DISABLED_TOOLS}
    started = time.monotonic()
    calls = []
    counts = Counter()
    with trace.open("x", encoding="utf-8", buffering=1) as handle:
        def record(kind, data, elapsed=0):
            handle.write(json.dumps({"event": kind, "data": data, "elapsed_seconds": elapsed},
                                    ensure_ascii=False) + "\n")

        record("request", payload | {"bot_url": bot_url})
        try:
            with client.stream("POST", f"{bot_url}/bot/v1/chat/stream", json=payload,
                               headers={"Accept": "text/event-stream", "X-OpenViking-User": run_id}) as stream:
                stream.raise_for_status()
                for line in stream.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    event = json.loads(line[5:])
                    kind, data = event["event"], event.get("data")
                    elapsed = round(time.monotonic() - started, 3)
                    counts[kind] += 1
                    if kind not in {"reasoning", "reasoning_delta"}:
                        record(kind, {"timestamp": event.get("timestamp"), "value": data}, elapsed)
                    if kind == "tool_call":
                        calls.append(tool_call(data, elapsed))
                    elif kind in {"error", "interrupt"}:
                        raise RuntimeError(f"VikingBot stream ended with {kind}")
                    elif kind == "response":
                        answer = str((data or {}).get("content") or "").strip()
                        if not answer or answer.startswith("Sorry, I encountered an error:"):
                            raise RuntimeError("VikingBot returned an error or empty answer")
                        return {"id": case["id"], "answer": answer, "session_id": session,
                                "bot_url": bot_url,
                                "response_id": data.get("response_id"), "elapsed_seconds": elapsed,
                                "tools": calls, "event_counts": dict(counts),
                                "trace": str(trace.relative_to(args.output))}
            raise RuntimeError("VikingBot stream closed before the final response")
        except (httpx.HTTPError, ValueError, RuntimeError) as exc:
            record("request_failed", error_label(exc), round(time.monotonic() - started, 3))
            raise


def answer_text(answer):
    answer = SOURCE_LINK.sub(lambda link: "" if link[1].startswith("viking://") else link[1], answer)
    answer = SOURCE_URI.sub("", answer)
    return "\n".join(line for line in answer.splitlines()
                     if MetricsCalculator.normalize_answer(line) not in {"", "source", "sources"}).strip()


def token_scores(case, answer):
    raw_f1 = MetricsCalculator.calculate_f1(answer, case["gold"])
    abstained = (
        case["type"] == "no answer" and MetricsCalculator.normalize_answer(answer) in NO_ANSWER
    )
    return {"f1": 1.0 if abstained else raw_f1, "raw_token_f1": raw_f1,
            "no_answer_equivalence": abstained}


def score_answer(judge, case, generation, model):
    answer = answer_text(generation["answer"])
    raw = llm_grader(judge, model, case["question"], case["gold"], answer,
                     dataset_name="SyllabusQA", strict=True)
    return {"id": case["id"], **token_scores(case, answer), "judge": raw}


def error_label(exc):
    status = getattr(exc, "status_code", None)
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
    return f"{type(exc).__name__} (HTTP {status})" if status else type(exc).__name__


def retry(operation):
    try:
        return operation()
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
        if status in {400, 401, 403, 404, 422}:
            raise
        time.sleep(3)
        return operation()


def publish(output, cases, generations, scores, errors):
    brief = []
    results = []
    for case in cases:
        generation = generations.get(case["id"])
        if not generation:
            continue
        brief.append({key: value for key, value in generation.items() if key != "answer"}
                     | {"syllabus": case["syllabus"], "resource_uri": case["resource_uri"]})
        score = scores.get(case["id"]) or token_scores(case, answer_text(generation["answer"]))
        results.append(case | {"answer": generation["answer"], "scores": score})

    def summarize(rows):
        f1 = [row["scores"]["f1"] for row in rows]
        raw_f1 = [row["scores"]["raw_token_f1"] for row in rows]
        judge = [row["scores"]["judge"]["score"] for row in rows if "judge" in row["scores"]]
        return {"answered": len(rows), "scored": len(judge), "mean_f1": fmean(f1) if f1 else None,
                "mean_raw_token_f1": fmean(raw_f1) if raw_f1 else None,
                "mean_llm_score_0_to_4": fmean(judge) if judge else None,
                "llm_score_histogram": dict(sorted(Counter(judge).items()))}

    by_type = defaultdict(list)
    for row in results:
        by_type[row["type"]].append(row)
    report = {"total_cases": len(cases), **summarize(results),
              "by_question_type": {key: summarize(rows) for key, rows in sorted(by_type.items())},
              "errors_in_this_run": errors}
    atomic_json(output / "traces_brief.jsonl", brief, lines=True)
    atomic_json(output / "results.jsonl", results, lines=True)
    atomic_json(output / "report.json", report)
    print(f"Answered {len(generations)}/{len(cases)}, scored {len(scores)}/{len(cases)}", flush=True)


def execute(args, cases, run_id):
    output = args.output
    generations = {path.stem: json.loads(path.read_text(encoding="utf-8"))
                   for path in (output / "answers").glob("*.json")}
    scores = {path.stem: json.loads(path.read_text(encoding="utf-8"))
              for path in (output / "scores").glob("*.json")}
    selected = cases[:args.limit]
    judge = None
    if args.phase != "generate":
        key = args.key_file.read_text(encoding="utf-8").strip()
        if not key:
            raise ValueError("The judge API key file is empty")
        judge = ChatOpenAI(model=args.judge_model, base_url=args.judge_base_url, api_key=key,
                           temperature=0, timeout=60, max_retries=2, max_tokens=256,
                           extra_body={"thinking": {"type": "disabled"}},
                           model_kwargs={"response_format": {"type": "json_object"}})

    errors = []
    bot_urls = list(dict.fromkeys(url.rstrip("/") for url in (args.bot_url or [args.server])))
    gateways = Queue()
    bot_timeout = httpx.Timeout(330, connect=10)
    with (httpx.Client(timeout=bot_timeout, trust_env=False) as client,
          ThreadPoolExecutor(max_workers=min(args.workers, len(bot_urls))) as bot_pool,
          ThreadPoolExecutor(max_workers=args.judge_workers) as judge_pool):
        if args.phase != "score":
            for url in bot_urls:
                client.get(f"{url}/bot/v1/health").raise_for_status()
                gateways.put(url)
        jobs = {}

        def generate(case):
            url = gateways.get()
            try:
                return retry(lambda: ask_bot(client, args, run_id, case, url))
            finally:
                gateways.put(url)

        def enqueue_score(case):
            if judge is not None and case["id"] not in scores:
                generation = generations[case["id"]]
                future = judge_pool.submit(retry, lambda: score_answer(judge, case, generation, args.judge_model))
                jobs[future] = case

        def collect_score(future):
            case = jobs.pop(future)
            try:
                result = future.result()
                atomic_json(output / "scores" / f"{case['id']}.json", result)
                scores[case["id"]] = result
            except Exception as exc:
                errors.append({"id": case["id"], "phase": "score", "error": error_label(exc)})
                print(f"Score failed for {case['id']}: {error_label(exc)}", flush=True)

        tasks = {}
        for case in selected:
            if case["id"] in generations:
                enqueue_score(case)
            elif args.phase != "score":
                future = bot_pool.submit(generate, case)
                tasks[future] = case

        publish(output, cases, generations, scores, errors)
        for number, future in enumerate(as_completed(tasks), 1):
            case = tasks[future]
            try:
                result = future.result()
                atomic_json(output / "answers" / f"{case['id']}.json", result)
                generations[case["id"]] = result
                enqueue_score(case)
            except Exception as exc:
                errors.append({"id": case["id"], "phase": "generate", "error": error_label(exc)})
                print(f"VikingBot failed for {case['id']}: {error_label(exc)}", flush=True)
            for job in list(jobs):
                if job.done():
                    collect_score(job)
            if number % 10 == 0:
                publish(output, cases, generations, scores, errors)
        for number, future in enumerate(as_completed(jobs), 1):
            collect_score(future)
            if number % 10 == 0:
                publish(output, cases, generations, scores, errors)

    publish(output, cases, generations, scores, errors)
    required = generations if args.phase == "generate" else scores
    return int(any(case["id"] not in required for case in selected))


def main():
    os.umask(0o077)
    args = parse_args()
    args.server = args.server.rstrip("/")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args)
    metadata = {"dataset_hash": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
                "protocol": 1, "server": args.server,
                "judge_base_url": args.judge_base_url, "judge_model": args.judge_model}
    with (args.output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another process is already using this output directory") from None
        run = args.output / "run.json"
        if run.exists():
            saved = json.loads(run.read_text(encoding="utf-8"))
            if any(saved.get(key) != value for key, value in metadata.items()):
                raise SystemExit("This output was created with different data, server, or judge settings")
            run_id = saved["run_id"]
        else:
            run_id = "syllabusqa-" + uuid4().hex[:12]
            atomic_json(run, metadata | {"run_id": run_id})
        for directory in ("answers", "scores", "traces_detailed"):
            (args.output / directory).mkdir(exist_ok=True)
        return execute(args, cases, run_id)


if __name__ == "__main__":
    raise SystemExit(main())
