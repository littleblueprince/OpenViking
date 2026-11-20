"""Human-readable views of already matched tool traces; no matching or inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .trace_models import StepEvidenceRecord


def validate_preview_chars(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("preview_chars must be a positive integer")


def tool_results_from_events(events: Iterable[dict[str, Any]]) -> dict[str, str]:
    results = {}
    for event in events:
        if event.get("event") != "tool_result":
            continue
        data = event.get("data") or {}
        call_id = (data.get("metadata") or {}).get("tool_call_id")
        value = data.get("value")
        if call_id and isinstance(value, str):
            results[call_id] = value
    return results


def readable_step(
    record: StepEvidenceRecord, *, preview_chars: int = 500, tool_result: str | None = None
) -> dict[str, Any]:
    validate_preview_chars(preview_chars)
    step = record.step
    chunks = []
    for observation, matched in zip(step.sources, record.sources, strict=True):
        # Separators are display-only: disjoint grep lines are never contiguous evidence.
        preview = "\n[separate fragment]\n".join(
            fragment.text for fragment in observation.returned_fragments
        )
        valid = step.success and matched.status == "ok"
        hit = bool(matched.returned_evidence_ids) if valid else None
        delivered_hit = (
            bool(matched.delivered_evidence_ids) if valid and matched.delivery_captured else None
        )
        if not valid:
            marker = matched.status.upper()
        elif hit:
            marker = "HIT"
        elif matched.located_evidence_ids:
            marker = "LOCATED_ONLY"
        elif matched.ancestor_evidence_ids:
            marker = "ANCESTOR_ONLY"
        else:
            marker = "MISS"
        chunks.append({
            "chunk_uri": matched.chunk_uri,
            "source_kind": "directory" if observation.is_directory else observation.content_kind,
            "content_preview": preview[:preview_chars],
            "hit": hit,
            "hit_marker": marker,
            "evidence_ids": list(matched.returned_evidence_ids),
            "located_hit": bool(matched.located_evidence_ids) if valid else None,
            "located_evidence_ids": list(matched.located_evidence_ids),
            "delivered_hit": delivered_hit,
            "delivered_evidence_ids": list(matched.delivered_evidence_ids),
            "status": matched.status,
            "revision_status": matched.revision_status,
            "preview_truncated": len(preview) > preview_chars,
            "returned_fragment_count": len(observation.returned_fragments),
            "tool_output_truncated": matched.truncated,
        })
    result = {
        "step": step.sequence,
        "iteration": step.iteration,
        "tool": step.tool_name,
        "tool_call_id": step.tool_call_id,
        "args": step.args,
        "success": step.success,
        "duration_sec": step.duration_sec,
        "chunk_count": len(chunks),
        "hit": any(chunk["hit"] is True for chunk in chunks),
        "hit_chunk_count": sum(chunk["hit"] is True for chunk in chunks),
        "chunks": chunks,
    }
    if tool_result is not None:
        result["result_preview"] = tool_result[:preview_chars]
        result["result_preview_truncated"] = len(tool_result) > preview_chars
    return result


def build_readable_trace(
    records: Iterable[StepEvidenceRecord],
    summary: dict[str, Any],
    *,
    preview_chars: int = 500,
    tool_results: dict[str, str] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_preview_chars(preview_chars)
    records = sorted(records, key=lambda record: record.step.sequence)
    tool_results = tool_results or {}
    steps = [
        readable_step(
            record, preview_chars=preview_chars,
            tool_result=tool_results.get(record.step.tool_call_id),
        )
        for record in records
    ]
    identity = {}
    if records:
        first = records[0].step
        identity = {
            "case_id": first.case_id, "run_id": first.run_id,
            "session_id": first.session_id, "response_id": first.response_id,
        }
    return {
        "view_schema_version": 1,
        **(context or {}),
        **identity,
        "total_steps": len(steps),
        "hit_step_count": sum(step["hit"] for step in steps),
        "preview_chars": preview_chars,
        "hit_definition": (
            "hit=true when full returned source text matches gold evidence; "
            "located_hit marks a known gold chunk URI; delivered_hit marks text "
            "inserted into agent context. null means failed/unknown or delivery uncaptured. "
            "Content previews are for display only and may omit the matching span."
        ),
        "evidence_summary": summary,
        "steps": steps,
    }


def write_readable_trace(path: str | Path, view: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(view, ensure_ascii=False, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as file:
        file.write(payload)
    return path


def format_step_log(record: StepEvidenceRecord, *, preview_chars: int = 200) -> str:
    view = readable_step(record, preview_chars=preview_chars)
    lines = [
        f"[RAG Evidence] case={record.step.case_id} step={view['step']} "
        f"iteration={view['iteration']} "
        f"tool={view['tool']} call={view['tool_call_id']} success={view['success']} "
        f"returned_hits={view['hit_chunk_count']}/{view['chunk_count']}"
    ]
    for chunk in view["chunks"]:
        lines.append(
            f"  [{chunk['hit_marker']}] chunk={chunk['chunk_uri']} "
            f"kind={chunk['source_kind']} returned={chunk['evidence_ids']} "
            f"located={chunk['located_evidence_ids']} "
            f"delivered={chunk['delivered_evidence_ids']} "
            f"delivered_hit={chunk['delivered_hit']} "
            f"status={chunk['status']} revision={chunk['revision_status']} "
            f"preview={json.dumps(chunk['content_preview'], ensure_ascii=False)}"
        )
    return "\n".join(lines)
