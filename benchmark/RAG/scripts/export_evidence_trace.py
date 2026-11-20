#!/usr/bin/env python3
"""Convert a saved runner result into expanded JSON and readable step logs offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.evidence_matcher import EvidenceManifest, EvidenceMatcher
from core.readable_trace import (
    build_readable_trace, format_step_log, tool_results_from_events, write_readable_trace,
)
from core.trace_annotation import summarize_trace
from core.trace_models import StepEvidenceRecord


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preview-chars", type=int, default=500)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    trace = Path(result["evidence_trace_path"])
    records = [
        StepEvidenceRecord.from_dict(json.loads(line))
        for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    manifest = EvidenceManifest.from_dict(json.loads(
        Path(result["evidence_manifest_path"]).read_text(encoding="utf-8")
    ))
    matcher = EvidenceMatcher.from_manifest(manifest)
    view = build_readable_trace(
        records, summarize_trace(matcher, records).to_dict(),
        preview_chars=args.preview_chars,
        tool_results=tool_results_from_events(result.get("events", [])),
        context={
            **{key: result.get(key) for key in ("case_id", "run_id", "question", "answer", "error")},
            "model_iterations": result.get("iterations_used"),
            "pending_tool_call_ids": result.get("evidence_pending_tool_call_ids", []),
        },
    )
    output = args.output or trace.with_name(trace.stem + ".readable.json")
    log_path = output.with_suffix(".log")
    if log_path.exists():
        raise FileExistsError(log_path)
    write_readable_trace(output, view)
    with log_path.open("x", encoding="utf-8") as file:
        file.write("\n\n".join(
            format_step_log(record, preview_chars=args.preview_chars) for record in records
        ) + "\n")
    print(json.dumps({
        "total_steps": view["total_steps"], "hit_step_count": view["hit_step_count"],
        "json_path": str(output.resolve()), "log_path": str(log_path.resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
