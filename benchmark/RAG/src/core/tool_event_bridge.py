"""Translate native structured tool events to benchmark-only evidence records."""

from __future__ import annotations

from typing import Any

from .trace_annotation import EvidenceTraceRecorder
from .trace_models import SourceObservation, ToolStep


class ToolEventEvidenceBridge:
    def __init__(
        self, recorder: EvidenceTraceRecorder, *, case_id: str, run_id: str, logger: Any,
        preview_chars: int = 500,
    ):
        if not case_id or not run_id:
            raise ValueError("Evidence capture requires case_id and run_id")
        self.recorder = recorder
        self.case_id = case_id
        self.run_id = run_id
        self.logger = logger
        self.preview_chars = preview_chars
        self._calls: dict[str, dict[str, Any]] = {}
        self._completed: set[str] = set()

    @property
    def pending_tool_call_ids(self) -> list[str]:
        return [call_id for call_id in self._calls if call_id not in self._completed]

    def consume(self, event: dict[str, Any]) -> None:
        kind = event["event"]
        if kind not in {"tool_call", "tool_result"}:
            return
        data = event["data"]
        metadata = data.get("metadata", {})
        call_id = metadata.get("tool_call_id")
        if not call_id:
            raise ValueError("Tool event lacks structured tool_call_id; upgrade VikingBot")
        identity = {
            key: data.get(key) for key in ("session_id", "response_id", "iteration")
        }
        if kind == "tool_call":
            if call_id in self._calls:
                raise ValueError("Duplicate tool_call event")
            self._calls[call_id] = {**identity, **metadata}
            return
        call = self._calls.get(call_id)
        if call is None:
            raise ValueError("Tool result has no preceding dispatch event")
        for key in ("session_id", "response_id", "iteration", "sequence", "tool_name"):
            actual = identity.get(key) if key in identity else metadata.get(key)
            if call.get(key) != actual:
                raise ValueError(f"Tool result differs from dispatch: {key}")
        record = self.recorder.record(ToolStep(
            case_id=self.case_id,
            run_id=self.run_id,
            session_id=data["session_id"],
            response_id=data["response_id"],
            tool_call_id=call_id,
            tool_name=metadata["tool_name"],
            iteration=data["iteration"],
            sequence=metadata["sequence"],
            args=metadata.get("resolved_args", call.get("args", {})),
            sources=tuple(
                SourceObservation.from_dict(source) for source in metadata.get("sources", [])
            ),
            success=metadata["success"],
            duration_sec=metadata.get("duration_sec", 0.0),
            timestamp=data["timestamp"],
        ))
        self.logger.info(self.recorder.format_log(record, self.preview_chars))
        self._completed.add(call_id)
