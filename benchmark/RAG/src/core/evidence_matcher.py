"""Deterministic, document-scoped evidence matching against a chunk snapshot.

No LLM calls or store reads occur here. The caller supplies full L2 chunks once
and source observations for each step. Exact matches never join separate chunks
or fragments. Reasoning annotations are preserved but excluded from recall.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .trace_models import (
    ChunkDocument,
    EvidenceTextMatch,
    SourceEvidenceRecord,
    SourceObservation,
    StandardEvidence,
    StepEvidenceRecord,
    ToolStep,
)

MATCHER_VERSION = "1"


def normalize_evidence_text(text: str) -> str:
    """Preserve case, punctuation, numbers, units and negation; fold whitespace."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _uri_key(uri: str) -> str:
    # Do not lowercase or URL-decode paths: those can denote different resources.
    return uri.rstrip("/")


def _is_ascii_word(char: str) -> bool:
    return char.isascii() and (char.isalnum() or char == "_")


def _positions(text: str, evidence: StandardEvidence) -> Iterable[tuple[int, int]]:
    needle = normalize_evidence_text(evidence.text)
    if not needle:
        return
    if evidence.kind == "dialogue_id":
        # D7:5 must not match D7:50, D17:5 or part of another identifier.
        pattern = rf"(?<![\w:]){re.escape(needle)}(?![\w:])"
        for match in re.finditer(pattern, text):
            yield match.span()
        return
    start = 0
    while (start := text.find(needle, start)) != -1:
        end = start + len(needle)
        left_ok = not (
            start and _is_ascii_word(needle[0]) and _is_ascii_word(text[start - 1])
        )
        right_ok = not (
            end < len(text) and _is_ascii_word(needle[-1]) and _is_ascii_word(text[end])
        )
        # A short numeric annotation must not match part of 15.5 or 15,000.
        if needle[0].isdigit() and start >= 2:
            left_ok = left_ok and not (text[start - 1] in ".," and text[start - 2].isdigit())
        if needle[-1].isdigit() and end + 1 < len(text):
            right_ok = right_ok and not (text[end] in ".," and text[end + 1].isdigit())
        if left_ok and right_ok:
            yield start, end
        start += 1


@dataclass(frozen=True)
class ChunkSnapshot:
    uri: str
    source_doc_id: str
    content_hash: str


@dataclass(frozen=True)
class EvidenceLocation:
    evidence_id: str
    chunk_uri: str
    normalized_start: int
    normalized_end: int
    match_method: str


@dataclass(frozen=True)
class EvidenceManifest:
    evidences: tuple[StandardEvidence, ...]
    chunks: tuple[ChunkSnapshot, ...]
    locations: tuple[EvidenceLocation, ...]
    unresolved_evidence_ids: tuple[str, ...]
    excluded_evidence_ids: tuple[str, ...]
    snapshot_id: str = ""
    schema_version: int = 1
    matcher_version: str = MATCHER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceManifest:
        if data.get("schema_version") != 1 or data.get("matcher_version") != MATCHER_VERSION:
            raise ValueError("Unsupported evidence manifest version")
        values = dict(data)
        values["evidences"] = tuple(StandardEvidence(**item) for item in values["evidences"])
        values["chunks"] = tuple(ChunkSnapshot(**item) for item in values["chunks"])
        values["locations"] = tuple(EvidenceLocation(**item) for item in values["locations"])
        for key in ("unresolved_evidence_ids", "excluded_evidence_ids"):
            values[key] = tuple(values[key])
        return cls(**values)


class EvidenceMatcher:
    def __init__(
        self,
        evidences: Iterable[StandardEvidence],
        chunks: Iterable[ChunkDocument],
        *,
        snapshot_id: str = "",
    ):
        evidences, chunks = tuple(evidences), tuple(chunks)
        if len({e.evidence_id for e in evidences}) != len(evidences):
            raise ValueError("evidence_id must be unique within a case")
        if len({_uri_key(c.uri) for c in chunks}) != len(chunks):
            raise ValueError("chunk URI must be unique within a snapshot")

        excluded = tuple(e.evidence_id for e in evidences if e.kind == "reasoning_annotation")
        by_doc: dict[str, list[StandardEvidence]] = {}
        for evidence in evidences:
            if evidence.evidence_id not in excluded:
                by_doc.setdefault(evidence.source_doc_id, []).append(evidence)
        locations = []
        for chunk in chunks:
            normalized = normalize_evidence_text(chunk.text)
            for evidence in by_doc.get(chunk.source_doc_id, ()):
                for start, end in _positions(normalized, evidence):
                    locations.append(
                        EvidenceLocation(
                            evidence.evidence_id,
                            chunk.uri,
                            start,
                            end,
                            self._method(evidence),
                        )
                    )
        resolved = {location.evidence_id for location in locations}
        self.manifest = EvidenceManifest(
            evidences=evidences,
            chunks=tuple(
                ChunkSnapshot(c.uri, c.source_doc_id, content_hash(c.text)) for c in chunks
            ),
            locations=tuple(locations),
            unresolved_evidence_ids=tuple(
                e.evidence_id
                for e in evidences
                if e.evidence_id not in excluded and e.evidence_id not in resolved
            ),
            excluded_evidence_ids=excluded,
            snapshot_id=snapshot_id,
        )
        self._index_manifest()

    @classmethod
    def from_manifest(cls, manifest: EvidenceManifest) -> EvidenceMatcher:
        """Reannotate a trace without fetching the chunk bodies again."""
        if manifest.schema_version != 1 or manifest.matcher_version != MATCHER_VERSION:
            raise ValueError("Unsupported evidence manifest version")
        matcher = cls.__new__(cls)
        matcher.manifest = manifest
        matcher._index_manifest()
        return matcher

    def _index_manifest(self):
        self._evidences = {e.evidence_id: e for e in self.manifest.evidences}
        self._chunks = {_uri_key(c.uri): c for c in self.manifest.chunks}
        if len(self._evidences) != len(self.manifest.evidences) or len(self._chunks) != len(
            self.manifest.chunks
        ):
            raise ValueError("Duplicate evidence IDs or chunk URIs in manifest")
        self._by_uri: dict[str, set[str]] = {}
        for location in self.manifest.locations:
            evidence = self._evidences.get(location.evidence_id)
            chunk = self._chunks.get(_uri_key(location.chunk_uri))
            if (
                evidence is None
                or chunk is None
                or evidence.source_doc_id != chunk.source_doc_id
                or evidence.kind == "reasoning_annotation"
            ):
                raise ValueError("Manifest location must reference evidence in the same document")
            if (
                location.normalized_start < 0
                or location.normalized_end <= location.normalized_start
                or location.match_method != self._method(evidence)
            ):
                raise ValueError("Invalid manifest match offsets or method")
            self._by_uri.setdefault(_uri_key(location.chunk_uri), set()).add(location.evidence_id)
        excluded = tuple(
            e.evidence_id for e in self.manifest.evidences if e.kind == "reasoning_annotation"
        )
        resolved = {location.evidence_id for location in self.manifest.locations}
        unresolved = tuple(
            e.evidence_id
            for e in self.manifest.evidences
            if e.evidence_id not in excluded and e.evidence_id not in resolved
        )
        if (
            self.manifest.excluded_evidence_ids != excluded
            or self.manifest.unresolved_evidence_ids != unresolved
        ):
            raise ValueError("Manifest evidence statuses are inconsistent with locations")

    @staticmethod
    def _method(evidence: StandardEvidence) -> str:
        return "dialogue_id" if evidence.kind == "dialogue_id" else "normalized_exact"

    def ordered_ids(self, ids: Iterable[str]) -> tuple[str, ...]:
        ids = set(ids)
        return tuple(e.evidence_id for e in self.manifest.evidences if e.evidence_id in ids)

    def annotate_step(self, step: ToolStep) -> StepEvidenceRecord:
        return StepEvidenceRecord(
            step=step,
            sources=tuple(self._annotate_source(source, step.success) for source in step.sources),
            matcher_version=MATCHER_VERSION,
        )

    def _annotate_source(
        self, source: SourceObservation, step_success: bool
    ) -> SourceEvidenceRecord:
        base = {
            "chunk_uri": source.uri,
            "delivery_captured": source.delivered_fragments is not None,
            "truncated": source.truncated,
            "limit_reached": source.limit_reached,
        }
        if not step_success or not source.success:
            return SourceEvidenceRecord(**base, status="failed", revision_status="not_applicable")

        key = _uri_key(source.uri)
        if source.is_directory:
            ancestor_ids = set()
            for uri, ids in self._by_uri.items():
                chunk = self._chunks[uri]
                if uri.startswith(key + "/") and (
                    source.source_doc_id is None or source.source_doc_id == chunk.source_doc_id
                ):
                    ancestor_ids.update(ids)
            return SourceEvidenceRecord(
                **base,
                status="ok",
                revision_status="not_applicable",
                ancestor_evidence_ids=self.ordered_ids(ancestor_ids),
            )
        chunk = self._chunks.get(key)
        if chunk is None:
            return SourceEvidenceRecord(
                **base, status="unknown_source", revision_status="not_applicable"
            )
        if source.source_doc_id is not None and source.source_doc_id != chunk.source_doc_id:
            return SourceEvidenceRecord(
                **base, status="source_mismatch", revision_status="not_applicable"
            )
        if source.content_hash is not None and source.content_hash != chunk.content_hash:
            return SourceEvidenceRecord(**base, status="stale_source", revision_status="mismatch")

        ids = self.ordered_ids(self._by_uri.get(key, ()))
        matches = []
        for stage, fragments in (
            ("returned", source.returned_fragments),
            ("delivered", source.delivered_fragments or ()),
        ):
            for index, fragment in enumerate(fragments):
                normalized = normalize_evidence_text(fragment.text)
                for evidence_id in ids:
                    evidence = self._evidences[evidence_id]
                    for start, end in _positions(normalized, evidence):
                        matches.append(
                            EvidenceTextMatch(
                                evidence_id, stage, index, start, end, self._method(evidence)
                            )
                        )
        return SourceEvidenceRecord(
            **base,
            status="ok",
            revision_status="verified" if source.content_hash is not None else "unverified",
            located_evidence_ids=ids,
            returned_evidence_ids=self.ordered_ids(
                match.evidence_id for match in matches if match.stage == "returned"
            ),
            delivered_evidence_ids=self.ordered_ids(
                match.evidence_id for match in matches if match.stage == "delivered"
            ),
            matches=tuple(matches),
        )
