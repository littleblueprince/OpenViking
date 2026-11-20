#!/usr/bin/env python3
"""Compare live OV grep with literal matches in already imported source files."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import re
import shlex
import time
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class OVClient:
    def __init__(self, config, timeout=60):
        self.url = config["url"].rstrip("/")
        self.timeout = timeout
        self._local = threading.local()
        self._clients = []
        self.headers = {"Content-Type": "application/json"}
        for field, header in (
            ("api_key", "X-API-Key"), ("account_id", "X-OpenViking-Account"),
            ("user_id", "X-OpenViking-User"),
        ):
            if config.get(field):
                self.headers[header] = config[field]

    def request(self, endpoint, params=None, body=None):
        import httpx

        # Keep a connection pool per worker: opening one TCP connection per
        # query exhausts macOS ephemeral ports during large scans.
        if not hasattr(self._local, "client"):
            self._local.client = httpx.Client(
                base_url=self.url, headers=self.headers, timeout=self.timeout,
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
                trust_env=False,
            )
            self._clients.append(self._local.client)
        response = self._local.client.request(
            "POST" if body is not None else "GET", endpoint, params=params, json=body,
        )
        if response.is_error:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
        data = response.json()
        if data.get("status") != "ok":
            raise RuntimeError(f"OV request failed: {data.get('error')}")
        return data

    def close(self):
        for client in self._clients:
            client.close()

    def snapshot(self, roots):
        """Walk directory by directory so recursive-list depth defaults cannot omit files."""
        pending, seen, documents = deque(roots), set(), {}
        while pending:
            uri = pending.popleft().rstrip("/")
            if uri in seen:
                continue
            seen.add(uri)
            if uri.endswith(".md"):
                documents[uri] = self.request("/api/v1/content/read", {"uri": uri})["result"]
                continue
            page = self.request("/api/v1/fs/ls", {
                "uri": uri, "recursive": "false", "node_limit": 10000,
            })
            if page.get("has_more"):
                raise ValueError(f"Listing capped at {uri}; narrow --uri to avoid an incomplete snapshot")
            entries = page.get("result")
            if not isinstance(entries, list):
                raise ValueError(f"Unexpected resource listing at {uri}")
            for entry in entries:
                child = entry.get("uri", "")
                if not child or child.rsplit("/", 1)[-1].startswith("."):
                    continue
                if entry.get("isDir") or child.endswith(".md"):
                    pending.append(child)
        if not documents:
            raise ValueError("No imported .md source files found")
        if not all(isinstance(text, str) for text in documents.values()):
            raise ValueError("Expected complete string source contents")
        return documents

    def grep(self, case, node_limit):
        result = self.request("/api/v1/search/grep", body={
            "uri": case["scope_uri"], "pattern": case["pattern"],
            "case_insensitive": False, "node_limit": node_limit, "level_limit": 100,
        })["result"]
        if not isinstance(result, dict) or not isinstance(result.get("matches"), list):
            raise ValueError("Unexpected grep response format")
        return result


def inside(uri, scope):
    return uri == scope or uri.startswith(scope.rstrip("/") + "/")


def expected_matches(documents, scope, pattern):
    compiled = re.compile(pattern)
    return [
        {"uri": uri, "line": number, "content": line}
        for uri, text in documents.items() if inside(uri, scope)
        for number, line in enumerate(text.splitlines(), 1) if compiled.search(line)
    ]


def select_words(documents, max_words, seed):
    """Round-robin across files, with a reproducible shuffled vocabulary per file."""
    rng = random.Random(seed)
    queues = []
    for uri, text in sorted(documents.items()):
        words = sorted(set(re.findall(r"\b[A-Za-z]{7,24}\b", text)))
        rng.shuffle(words)
        if words:
            queues.append((uri, deque(words)))
    selected = []
    while queues and len(selected) < max_words:
        remaining = []
        for uri, words in queues:
            selected.append((uri, words.popleft()))
            if words:
                remaining.append((uri, words))
            if len(selected) == max_words:
                break
        queues = remaining
    return selected


def build_cases(documents, roots, max_words=200, seed=0, scopes=("file", "root")):
    cases, seen = [], set()
    for uri, word in select_words(documents, max_words, seed):
        root = max((root for root in roots if inside(uri, root)), key=len)
        for variant, literal in (("full", word), ("prefix_minus_1", word[:-1]),
                                 ("prefix_minus_2", word[:-2])):
            for scope_type in scopes:
                scope = uri if scope_type == "file" else root
                pattern = re.escape(literal)
                key = (scope, pattern)
                if key in seen:
                    continue
                seen.add(key)
                expected = expected_matches(documents, scope, pattern)
                if not expected:
                    raise AssertionError("Generated pattern must occur in source snapshot")
                identity = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]
                cases.append({
                    "case_id": identity, "word": word, "variant": variant,
                    "pattern": pattern, "scope_type": scope_type, "scope_uri": scope,
                    "source_uri": uri, "expected": expected,
                })
    return cases


def compare(case, response):
    actual = {(item.get("uri"), str(item.get("line"))) for item in response["matches"]}
    missing = [item for item in case["expected"] if (item["uri"], str(item["line"])) not in actual]
    capped = bool(response.get("limit_reached") or response.get("candidate_limit_reached"))
    # Missing lines under a reported cap are inconclusive, not proven omissions.
    status = "capped" if missing and capped else "miss" if missing else "pass"
    return {"status": status, "missing": missing, "response": response}


def run_case(client, case, node_limit):
    started = time.monotonic()
    try:
        result = compare(case, client.grep(case, node_limit))
        if result["status"] == "miss":
            result["repeat"] = compare(case, client.grep(case, node_limit))
            result["stable_miss"] = result["repeat"]["status"] == "miss"
            # Detect concurrent edits before calling a mismatch a grep omission.
            source_uris = {match["uri"] for match in result["missing"]}
            changed = []
            for uri in sorted(source_uris):
                text = client.request("/api/v1/content/read", {"uri": uri})["result"]
                current_lines = text.splitlines()
                for match in result["missing"]:
                    if match["uri"] == uri and (
                        match["line"] > len(current_lines)
                        or current_lines[match["line"] - 1] != match["content"]
                    ):
                        changed.append(match)
            result["source_changed"] = changed
            if changed:
                result["status"] = "source_changed"
    except Exception as exc:
        result = {"status": "error", "error": str(exc)}
    return {**case, **result, "elapsed_seconds": time.monotonic() - started}


def cli_command(case, config_path, output_json=False, node_limit=256):
    command = (
        f"OPENVIKING_CLI_CONFIG_FILE={shlex.quote(str(config_path))} "
        f"{shlex.quote(str(ROOT / '.venv/bin/ov'))} grep {shlex.quote(case['pattern'])} "
        f"--uri {shlex.quote(case['scope_uri'])} --node-limit {node_limit} --level-limit 100"
    )
    return command + (" -o json" if output_json else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", action="append", required=True, help="Loaded root/file URI; repeatable")
    parser.add_argument("--cli-config", type=Path, default=ROOT / ".local-data/syllabusqa/ovcli.conf")
    parser.add_argument("--max-words", type=int, default=200, help="Each word generates full and two prefixes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--scope", choices=("file", "root", "both"), default="both")
    parser.add_argument("--node-limit", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark/RAG/Output/grep-scan")
    args = parser.parse_args()
    if min(args.max_words, args.workers, args.node_limit) < 1:
        parser.error("max-words, workers and node-limit must be positive")
    config_path = args.cli_config.resolve()
    client = OVClient(json.loads(config_path.read_text()), timeout=60)
    health = client.request("/health")
    roots = sorted({uri.rstrip("/") for uri in args.uri})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir.resolve() / f"{stamp}_{time.time_ns() % 1000000:06d}"
    output.mkdir(parents=True)
    print(f"Output: {output}\nReading imported sources...", flush=True)
    documents = client.snapshot(roots)
    (output / "snapshot.json").write_text(json.dumps(documents, ensure_ascii=False, indent=2))
    scopes = ("file", "root") if args.scope == "both" else (args.scope,)
    cases = build_cases(documents, roots, args.max_words, args.seed, scopes)
    counts = {key: 0 for key in ("pass", "miss", "capped", "error", "source_changed")}
    print(f"{len(documents)} files, {len(cases)} grep cases, {args.workers} workers", flush=True)
    failures, results = [], []
    started = time.monotonic()
    with (output / "results.jsonl").open("w", encoding="utf-8") as log, \
         concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_case, client, case, args.node_limit) for case in cases]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            result["reproduce_command"] = cli_command(result, config_path, node_limit=args.node_limit)
            counts[result["status"]] += 1
            results.append(result)
            log.write(json.dumps(result, ensure_ascii=False) + "\n")
            log.flush()
            if result["status"] != "pass":
                failures.append(result)
            if len(results) % 100 == 0 or result["status"] == "miss":
                print(f"{len(results)}/{len(cases)}: {counts}", flush=True)
    summary = {
        "server_url": client.url, "health": health, "roots": roots,
        "source_files": len(documents), "total_cases": len(cases), "seed": args.seed,
        "node_limit": args.node_limit, "scope": args.scope, "counts": counts,
        "stable_misses": sum(item.get("stable_miss", False) and item["status"] == "miss" for item in failures),
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (output / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2))
    report = [
        "# OV grep 已导入文件对照扫描", "",
        f"服务：`{client.url}`；版本：`{health.get('version', 'unknown')}`。",
        f"读取 {len(documents)} 个正文文件，执行 {len(cases)} 个 grep 查询。",
        "查询来自原文中的英文单词，以及去掉末尾 1 / 2 个字符的前缀；匹配区分大小写。",
        f"结果：{json.dumps(counts, ensure_ascii=False)}；重复调用仍漏检：{summary['stable_misses']}。",
        "", "直接读取的完整正文作为对照，检查每个预期匹配行是否出现在 grep 返回中。",
        "候选/结果截断记为 capped；接口报错记为 error；均不当作已证实漏检。",
        "漏检后重复查询并重新读取相关行，排除临时失败和原文变化。",
        "", "本扫描仅覆盖选定文档、抽样单词与字面量前缀，不代表全部正则表达式均无漏检。",
        "results.jsonl 保存全部请求/返回和预期行；failures.json 保存异常及复现命令；snapshot.json 保存原文。",
    ]
    for failure in failures[:20]:
        report.extend(["", f"## {failure['status']}: `{failure['pattern']}`",
                       f"范围：`{failure['scope_uri']}`", "```bash",
                       failure["reproduce_command"], "```"])
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Report: {output / 'report.md'}", flush=True)
    client.close()
    return 1 if counts["miss"] or counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
