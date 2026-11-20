#!/usr/bin/env python3
"""Prepare the full MuSiQue corpus; optionally import and verify its source map."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from run_vikingbot_evidence import LOCAL, expand_config, server_settings, write_json
from adapters.musique_adapter import MuSiQueAdapter, chunk_filename


def validate_root(uri):
    if not re.fullmatch(r"viking://resources/[A-Za-z0-9_-][A-Za-z0-9_/-]*", uri) or any(
        part in ("", ".", "..") for part in uri.removeprefix("viking://resources/").split("/")
    ):
        raise ValueError("Use a named corpus URI under viking://resources/ (letters, digits, _ or -)")
    return uri


def verify_source_map(client, adapter, root):
    """Resolve actual leaf URIs, rejecting missing, split, duplicate or foreign files."""
    expected = {chunk_filename(sid): sid for sid in adapter.chunks}
    mapping, seen, offset = {adapter.corpus_id: root}, set(), 0
    while True:
        page = client.ls_page(root, recursive=True, node_limit=1000, offset=offset)
        entries = page.get("result")
        if not isinstance(entries, list):
            raise ValueError("Expected a structured resource listing")
        progress = 0
        for entry in entries:
            uri = entry.get("uri", "")
            if not uri.startswith(root + "/"):
                raise ValueError(f"Listing returned URI outside corpus: {uri}")
            if uri in seen:
                continue
            seen.add(uri)
            progress += 1
            name = uri.rsplit("/", 1)[-1]
            if entry.get("isDir") or name.startswith("."):
                continue
            if name not in expected:
                raise ValueError(f"Unexpected corpus file (requires no_split): {uri}")
            sid = expected[name]
            if sid in mapping:
                raise ValueError(f"Duplicate imported chunk: {sid}")
            mapping[sid] = uri
        if not page.get("has_more", False):
            break
        if not progress:
            raise ValueError("Resource pagination made no progress")
        offset += len(entries)
    missing = set(adapter.chunks) - set(mapping)
    if missing:
        raise ValueError(f"Missing {len(missing)} imported chunks; first IDs: {sorted(missing)[:5]}")
    return mapping


def import_corpus(client, directory, root, processing_mode):
    from openviking_sdk.errors import NotFoundError

    try:
        client.stat(root)
    except NotFoundError:
        pass
    else:
        raise ValueError(f"Target already exists: {root}; use --verify-only or a new URI")
    result = client.add_resource(
        str(directory), to=root, wait=True,
        options={
            "preserve_structure": True, "args": {"parse_mode": "no_split"},
            "processing_mode": processing_mode,
        },
    )
    if not isinstance(result, dict) or result.get("root_uri", "").rstrip("/") != root:
        raise ValueError(f"Import returned an unexpected root URI: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resource-uri", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--ingest", action="store_true", help="Upload corpus and wait for indexing")
    mode.add_argument("--verify-only", action="store_true", help="Map an already imported corpus")
    parser.add_argument("--ov-conf", type=Path, default=LOCAL / "ov.conf")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--processing-mode", choices=("vectors_only", "semantic_and_vectors"),
                        default="vectors_only")
    args = parser.parse_args()
    root = validate_root(args.resource_uri.rstrip("/"))
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    adapter = MuSiQueAdapter(str(args.questions.resolve()), str(args.chunks.resolve()))
    stats = adapter.stats()
    output = args.output_dir.resolve()
    directory = output / "chunks"
    # Validate before creating or overwriting files, including an old prepared corpus.
    expected = {chunk_filename(sid) for sid in adapter.chunks}
    if directory.exists() and any(path.name not in expected or not path.is_file()
                                  for path in directory.iterdir()):
        raise ValueError("Prepared chunks directory contains foreign files; use a new output directory")
    adapter.data_prepare(str(directory))
    manifest = {
        "schema_version": 1, "stats": stats, "resource_uri": root,
        "questions_file": str(args.questions.resolve()), "chunks_file": str(args.chunks.resolve()),
        "input_sha256": {
            "questions": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
            "chunks": hashlib.sha256(args.chunks.read_bytes()).hexdigest(),
        },
        "files": {sid: {"file": chunk_filename(sid), "title": chunk["title"]}
                  for sid, chunk in adapter.chunks.items()},
    }
    # Gold/questions/manifest never enter the uploaded chunks directory.
    write_json(output / "manifest.json", manifest)
    write_json(output / "stats.json", stats)
    # Do not leave a stale map usable after preparation or failed verification.
    (output / "source-map.json").unlink(missing_ok=True)
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    if not (args.ingest or args.verify_only):
        print(f"Prepared: {directory}\nNo server/model calls. Use --ingest to create source-map.json.")
        return 0
    from openviking_sdk import SyncHTTPClient

    config = expand_config(json.loads(args.ov_conf.read_text(encoding="utf-8")))
    url, headers = server_settings(config)
    client = SyncHTTPClient(
        url=url, api_key=headers.get("X-API-Key"),
        account=headers["X-OpenViking-Account"], user=headers["X-OpenViking-User"],
        timeout=args.timeout,
    )
    report = {"status": "running", "resource_uri": root, "processing_mode": args.processing_mode}
    write_json(output / "import-report.json", report)
    try:
        client.initialize()
        if args.ingest:
            report["import_result"] = import_corpus(client, directory, root, args.processing_mode)
            write_json(output / "import-report.json", report)
        mapping = verify_source_map(client, adapter, root)
        write_json(output / "source-map.json", mapping)
        report.update(status="verified", mapped_chunks=len(mapping) - 1)
        print(f"Verified source map: {output / 'source-map.json'}", flush=True)
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        write_json(output / "import-report.json", report)
        client.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
