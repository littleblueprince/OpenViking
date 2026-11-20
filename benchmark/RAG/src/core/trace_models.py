"""Serializable inputs and records for benchmark-only evidence annotation.

Fragments are contiguous source text, not tool headers, arguments or error messages.
Keep disjoint grep lines and truncated preview pieces separate. Line ranges, when
available, are zero-based and end-exclusive. Matching offsets use normalized text.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StandardEvidence:
    evidence_id: str
    source_doc_id: str
    text: str
    kind: Literal["source_span", "dialogue_id", "reasoning_annotation"] = "source_span"
    annotation_id: str | None = None
    answer_group_id: str | None = None

    def __post_init__(self):
        if not self.evidence_id or not self.source_doc_id:
            raise ValueError("Evidence requires evidence_id and source_doc_id")
        if self.kind not in {"source_span", "dialogue_id", "reasoning_annotation"}:
            raise ValueError(f"Unsupported evidence kind: {self.kind}")


@dataclass(frozen=True)
class ChunkDocument:
    """Full L2 chunk from the evaluation snapshot, before range reads/previews."""

    uri: str
    source_doc_id: str
    text: str

    def __post_init__(self):
        if not self.uri or not self.source_doc_id:
            raise ValueError("Chunk requires uri and source_doc_id")


@dataclass(frozen=True)
class TextFragment:
    text: str
    start_line: int | None = None
    end_line: int | None = None

    def __post_init__(self):
        if self.start_line is not None and self.start_line < 0:
            raise ValueError("start_line must be non-negative")
        if self.end_line is not None and (
            self.start_line is None or self.end_line < self.start_line
        ):
            raise ValueError("end_line requires a start_line and must not precede it")


@dataclass(frozen=True)
class SourceObservation:
    """A returned source and the text actually returned/delivered for that source.

    None for delivered_fragments means delivery was not captured; () means no
    source text was delivered. content_hash refers to the full snapshot chunk,
    never the hash of a partial read. Missing hashes are explicitly unverified.
    """

    uri: str
    source_doc_id: str | None = None
    is_directory: bool = False
    content_kind: Literal["body", "abstract", "overview", "candidate"] = "body"
    returned_fragments: tuple[TextFragment, ...] = ()
    delivered_fragments: tuple[TextFragment, ...] | None = None
    content_hash: str | None = None
    success: bool = True
    truncated: bool = False
    limit_reached: bool = False

    def __post_init__(self):
        if not self.uri:
            raise ValueError("Source requires uri")
        if self.content_kind not in {"body", "abstract", "overview", "candidate"}:
            raise ValueError(f"Unsupported content kind: {self.content_kind}")
        if self.content_kind == "candidate" and (
            self.returned_fragments or self.delivered_fragments
        ):
            raise ValueError("Candidate-only observations cannot contain source text")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceObservation:
        values = dict(data)
        values["returned_fragments"] = tuple(
            TextFragment(**fragment) for fragment in values.get("returned_fragments", ())
        )
        delivered = values.get("delivered_fragments")
        values["delivered_fragments"] = (
            None if delivered is None else tuple(TextFragment(**fragment) for fragment in delivered)
        )
        return cls(**values)


@dataclass(frozen=True)
class ToolStep:
    """One invocation; sequence orders calls within one response, not completion.

    iteration may be zero for an automatic lookup preceding the first model turn.
    Multiple calls in one iteration still have different call IDs and sequences.
    success=False denotes whole-call failure. For partial multi-read failures,
    leave it True and set success=False only on the individual failed sources.
    """

    case_id: str
    run_id: str
    session_id: str
    response_id: str
    tool_call_id: str
    tool_name: str
    iteration: int
    sequence: int
    args: dict[str, Any] = field(default_factory=dict)
    sources: tuple[SourceObservation, ...] = ()
    success: bool = True
    duration_sec: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        identifiers = (
            self.case_id,
            self.run_id,
            self.session_id,
            self.response_id,
            self.tool_call_id,
            self.tool_name,
        )
        if not all(identifiers):
            raise ValueError("Step identifiers must be non-empty")
        if self.iteration < 0 or self.sequence < 0 or self.duration_sec < 0:
            raise ValueError("Iteration, sequence and duration must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolStep:
        values = dict(data)
        values["sources"] = tuple(
            SourceObservation.from_dict(source) for source in values.get("sources", ())
        )
        return cls(**values)


@dataclass(frozen=True)
class EvidenceTextMatch:
    evidence_id: str
    stage: Literal["returned", "delivered"]
    fragment_index: int
    normalized_start: int
    normalized_end: int
    match_method: Literal["normalized_exact", "dialogue_id"]


@dataclass(frozen=True)
class SourceEvidenceRecord:
    chunk_uri: str
    status: Literal["ok", "failed", "unknown_source", "source_mismatch", "stale_source"]
    revision_status: Literal["verified", "unverified", "mismatch", "not_applicable"]
    delivery_captured: bool
    located_evidence_ids: tuple[str, ...] = ()
    ancestor_evidence_ids: tuple[str, ...] = ()
    returned_evidence_ids: tuple[str, ...] = ()
    delivered_evidence_ids: tuple[str, ...] = ()
    matches: tuple[EvidenceTextMatch, ...] = ()
    truncated: bool = False
    limit_reached: bool = False


@dataclass(frozen=True)
class StepEvidenceRecord:
    step: ToolStep
    sources: tuple[SourceEvidenceRecord, ...]
    cumulative_delivered_evidence_ids: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION
    matcher_version: str = "1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepEvidenceRecord:
        values = dict(data)
        if values.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported evidence trace schema_version")
        values["step"] = ToolStep.from_dict(values["step"])
        sources = []
        for source in values["sources"]:
            source = dict(source)
            for key in (
                "located_evidence_ids",
                "ancestor_evidence_ids",
                "returned_evidence_ids",
                "delivered_evidence_ids",
            ):
                source[key] = tuple(source.get(key, ()))
            source["matches"] = tuple(
                EvidenceTextMatch(**match) for match in source.get("matches", ())
            )
            sources.append(SourceEvidenceRecord(**source))
        values["sources"] = tuple(sources)
        values["cumulative_delivered_evidence_ids"] = tuple(
            values.get("cumulative_delivered_evidence_ids", ())
        )
        return cls(**values)


@dataclass(frozen=True)
class TraceSummary:
    status: Literal["applicable", "no_gold_evidence", "no_applicable_evidence"]
    applicable_evidence_ids: tuple[str, ...]
    unresolved_evidence_ids: tuple[str, ...]
    excluded_evidence_ids: tuple[str, ...]
    located_evidence_ids: tuple[str, ...]
    returned_evidence_ids: tuple[str, ...]
    delivered_evidence_ids: tuple[str, ...]
    first_located_step: int | None
    first_delivered_step: int | None
    located_recall: float | None
    returned_recall: float | None
    delivered_recall: float | None
    answer_group_delivered_recall: dict[str, float]
    delivery_uncaptured_source_count: int
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
