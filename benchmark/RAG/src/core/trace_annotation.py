"""Offline summaries and JSONL recording for one case/response.

Call record() in dispatch sequence order (buffer parallel completions upstream).
Each retry/response uses its own recorder. Gold data is owned by the matcher,
never passed to the Agent. This module performs no inference or retrieval.
"""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .evidence_matcher import EvidenceMatcher
from .readable_trace import build_readable_trace, format_step_log, write_readable_trace
from .trace_models import StepEvidenceRecord, ToolStep, TraceSummary


def summarize_trace(
    matcher: EvidenceMatcher, records: Iterable[StepEvidenceRecord]
) -> TraceSummary:
    records = sorted(records, key=lambda record: record.step.sequence)
    identities = {
        (r.step.run_id, r.step.case_id, r.step.session_id, r.step.response_id) for r in records
    }
    if len(identities) > 1:
        raise ValueError("A trace summary must contain only one case/response")
    sequences = [r.step.sequence for r in records]
    call_ids = [r.step.tool_call_id for r in records]
    if len(set(sequences)) != len(sequences) or len(set(call_ids)) != len(call_ids):
        raise ValueError("Trace contains duplicate sequences or tool_call_ids")

    manifest = matcher.manifest
    excluded = set(manifest.excluded_evidence_ids)
    applicable = tuple(e.evidence_id for e in manifest.evidences if e.evidence_id not in excluded)
    hits: dict[str, set[str]] = {"located": set(), "returned": set(), "delivered": set()}
    first_located = first_delivered = None
    uncaptured = 0
    for record in records:
        if (
            record.schema_version != manifest.schema_version
            or record.matcher_version != manifest.matcher_version
        ):
            raise ValueError("Trace and matcher versions do not match")
        for observation, source in zip(record.step.sources, record.sources, strict=True):
            if (
                record.step.success
                and observation.success
                and not observation.is_directory
                and observation.content_kind != "candidate"
                and not source.delivery_captured
            ):
                uncaptured += 1
            if source.status != "ok" or not record.step.success:
                continue
            for stage in hits:
                source_ids = set(getattr(source, f"{stage}_evidence_ids"))
                if not source_ids.issubset(applicable):
                    raise ValueError("Trace contains unknown or excluded evidence IDs")
                hits[stage].update(source_ids)
            if source.located_evidence_ids and first_located is None:
                first_located = record.step.sequence
            if source.delivered_evidence_ids and first_delivered is None:
                first_delivered = record.step.sequence

    count = len(applicable)
    groups: dict[str, set[str]] = {}
    for evidence in manifest.evidences:
        if evidence.evidence_id not in excluded and evidence.answer_group_id is not None:
            groups.setdefault(evidence.answer_group_id, set()).add(evidence.evidence_id)
    return TraceSummary(
        status=(
            "applicable"
            if count
            else "no_gold_evidence" if not manifest.evidences else "no_applicable_evidence"
        ),
        applicable_evidence_ids=applicable,
        unresolved_evidence_ids=manifest.unresolved_evidence_ids,
        excluded_evidence_ids=manifest.excluded_evidence_ids,
        located_evidence_ids=matcher.ordered_ids(hits["located"]),
        returned_evidence_ids=matcher.ordered_ids(hits["returned"]),
        delivered_evidence_ids=matcher.ordered_ids(hits["delivered"]),
        first_located_step=first_located,
        first_delivered_step=first_delivered,
        located_recall=len(hits["located"]) / count if count else None,
        returned_recall=len(hits["returned"]) / count if count else None,
        delivered_recall=len(hits["delivered"]) / count if count else None,
        answer_group_delivered_recall={
            group: len(ids & hits["delivered"]) / len(ids) for group, ids in groups.items()
        },
        delivery_uncaptured_source_count=uncaptured,
    )


class EvidenceTraceRecorder:
    """Create a new JSONL trace; protect each ordered record with a thread lock.

    Existing files are not overwritten. The first step fixes the case/run/session/
    response identity. State advances only after serialization and writing succeed.
    """

    def __init__(self, path: str | Path, matcher: EvidenceMatcher):
        self.path = Path(path)
        self.matcher = matcher
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("x", encoding="utf-8")
        self._lock = threading.Lock()
        self._records: list[StepEvidenceRecord] = []
        self._call_ids: set[str] = set()
        self._identity: tuple[str, ...] | None = None
        self._delivered: set[str] = set()

    def record(self, step: ToolStep) -> StepEvidenceRecord:
        with self._lock:
            if self._file.closed:
                raise ValueError("Recorder is closed")
            identity = (step.run_id, step.case_id, step.session_id, step.response_id)
            if self._identity is not None and self._identity != identity:
                raise ValueError("Recorder accepts only one case/response")
            if step.tool_call_id in self._call_ids:
                raise ValueError("Duplicate tool_call_id")
            if self._records and step.sequence <= self._records[-1].step.sequence:
                raise ValueError("Record steps in strictly increasing dispatch sequence order")
            record = self.matcher.annotate_step(step)
            delivered = self._delivered | {
                evidence_id
                for source in record.sources
                for evidence_id in source.delivered_evidence_ids
            }
            record = replace(
                record, cumulative_delivered_evidence_ids=self.matcher.ordered_ids(delivered)
            )
            self._file.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            self._file.flush()
            self._identity = identity
            self._delivered = delivered
            self._call_ids.add(step.tool_call_id)
            self._records.append(record)
            return record

    def summary(self) -> TraceSummary:
        with self._lock:
            return summarize_trace(self.matcher, self._records)

    def export_readable_json(
        self,
        path: str | Path,
        *,
        preview_chars: int = 500,
        tool_results: dict[str, str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> Path:
        with self._lock:
            view = build_readable_trace(
                self._records, summarize_trace(self.matcher, self._records).to_dict(),
                preview_chars=preview_chars, tool_results=tool_results, context=context,
            )
            return write_readable_trace(path, view)

    @staticmethod
    def format_log(record: StepEvidenceRecord, preview_chars: int = 200) -> str:
        return format_step_log(record, preview_chars=preview_chars)

    def close(self):
        with self._lock:
            self._file.close()

    def __enter__(self) -> EvidenceTraceRecorder:
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
