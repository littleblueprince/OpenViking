"""Evidence semantics without an OV server or LLM; also discoverable by pytest."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from benchmark.RAG.src.core.evidence_matcher import (
    EvidenceManifest,
    EvidenceMatcher,
    content_hash,
    normalize_evidence_text,
)
from benchmark.RAG.src.core.trace_annotation import EvidenceTraceRecorder, summarize_trace
from benchmark.RAG.src.core.trace_models import (
    ChunkDocument,
    SourceObservation,
    StandardEvidence,
    StepEvidenceRecord,
    TextFragment,
    ToolStep,
)

URI = "viking://resources/course/chunk_1.md"
TEXT = "Missing one exam decreases the grade by 15 points."


def make_step(*sources, sequence=1, call_id=None, iteration=1, **kwargs):
    return ToolStep(
        case_id="qa_1",
        run_id="run_1",
        session_id="session_1",
        response_id="response_1",
        tool_call_id=call_id or f"call_{sequence}",
        tool_name="openviking_multi_read",
        iteration=iteration,
        sequence=sequence,
        sources=tuple(sources),
        **kwargs,
    )


class EvidenceMatchingTests(unittest.TestCase):
    def setUp(self):
        self.evidence = StandardEvidence("ev_1", "course", TEXT, annotation_id="answer_span_1")
        self.matcher = EvidenceMatcher(
            [self.evidence], [ChunkDocument(URI, "course", TEXT)], snapshot_id="snapshot_1"
        )

    def annotate(self, **kwargs):
        return self.matcher.annotate_step(make_step(SourceObservation(URI, **kwargs))).sources[0]

    def test_normalization_preserves_semantics(self):
        self.assertEqual(
            normalize_evidence_text("Cafe\u0301\n costs\t15 USD."), "Café costs 15 USD."
        )
        evidence = StandardEvidence("e", "doc", "Do not pay 15 USD.")
        for text in ("Do pay 15 USD.", "Do not pay 150 USD.", "Do not pay 15 EUR."):
            with self.subTest(text=text):
                matcher = EvidenceMatcher([evidence], [ChunkDocument(URI, "doc", text)])
                self.assertEqual(matcher.manifest.unresolved_evidence_ids, ("e",))

    def test_short_number_does_not_match_larger_number(self):
        for text in ("150", "15.5", "15,000", "115", "1.15"):
            with self.subTest(text=text):
                matcher = EvidenceMatcher(
                    [StandardEvidence("e", "doc", "15")], [ChunkDocument(URI, "doc", text)]
                )
                self.assertFalse(matcher.manifest.locations)

    def test_whitespace_match_and_normalized_offsets(self):
        observed = "Preface.\nMissing one exam\n decreases the grade by 15 points."
        source = self.annotate(
            returned_fragments=(TextFragment(observed),),
            delivered_fragments=(TextFragment(observed),),
            content_hash=content_hash(TEXT),
        )
        self.assertEqual(source.revision_status, "verified")
        self.assertEqual(source.delivered_evidence_ids, ("ev_1",))
        hit = source.matches[0]
        self.assertEqual(
            normalize_evidence_text(observed)[hit.normalized_start : hit.normalized_end], TEXT
        )

    def test_candidate_and_directory_are_not_read_hits(self):
        candidate = self.annotate(content_kind="candidate", delivered_fragments=())
        self.assertEqual(candidate.located_evidence_ids, ("ev_1",))
        self.assertFalse(candidate.returned_evidence_ids)
        directory = SourceObservation("viking://resources/course", is_directory=True)
        result = self.matcher.annotate_step(make_step(directory)).sources[0]
        self.assertEqual(result.ancestor_evidence_ids, ("ev_1",))
        self.assertFalse(result.located_evidence_ids)
        sibling = replace(directory, uri="viking://resources/cour")
        self.assertFalse(
            self.matcher.annotate_step(make_step(sibling)).sources[0].ancestor_evidence_ids
        )
        with self.assertRaises(ValueError):
            SourceObservation(
                URI, content_kind="candidate", returned_fragments=(TextFragment(TEXT),)
            )

    def test_partial_read_locates_chunk_without_reading_evidence(self):
        source = self.annotate(
            returned_fragments=(TextFragment("Other section", 0, 1),),
            delivered_fragments=(TextFragment("Other section", 0, 1),),
        )
        self.assertEqual(source.located_evidence_ids, ("ev_1",))
        self.assertFalse(source.returned_evidence_ids)
        self.assertFalse(source.delivered_evidence_ids)

    def test_truncated_middle_is_not_delivered(self):
        source = self.annotate(
            returned_fragments=(TextFragment(TEXT),),
            delivered_fragments=(TextFragment("Introduction"), TextFragment("Conclusion")),
            truncated=True,
        )
        self.assertEqual(source.returned_evidence_ids, ("ev_1",))
        self.assertFalse(source.delivered_evidence_ids)
        self.assertTrue(source.delivery_captured)

    def test_disjoint_preview_and_grep_fragments_are_not_joined(self):
        pieces = (
            TextFragment("Missing one exam"),
            TextFragment("decreases the grade by 15 points."),
        )
        source = self.annotate(returned_fragments=pieces, delivered_fragments=pieces)
        self.assertFalse(source.returned_evidence_ids)
        self.assertFalse(source.delivered_evidence_ids)
        # An actual contiguous multiline read does match.
        source = self.annotate(
            returned_fragments=(TextFragment("\n".join(p.text for p in pieces)),)
        )
        self.assertEqual(source.returned_evidence_ids, ("ev_1",))

    def test_no_cross_chunk_matches(self):
        chunks = [
            ChunkDocument(URI, "course", "Missing one exam"),
            ChunkDocument(URI + ".other", "course", "decreases the grade by 15 points."),
        ]
        self.assertFalse(EvidenceMatcher([self.evidence], chunks).manifest.locations)

    def test_multiple_locations_do_not_duplicate_evidence_recall(self):
        matcher = EvidenceMatcher(
            [self.evidence],
            [
                ChunkDocument(URI, "course", TEXT + "\n" + TEXT),
                ChunkDocument(URI + ".other", "course", TEXT),
            ],
        )
        self.assertEqual(len(matcher.manifest.locations), 3)
        fragments = (TextFragment(TEXT),)
        source = SourceObservation(URI, returned_fragments=fragments, delivered_fragments=fragments)
        records = [matcher.annotate_step(make_step(source, sequence=i)) for i in (1, 2)]
        summary = summarize_trace(matcher, records)
        self.assertEqual(summary.delivered_recall, 1.0)
        self.assertEqual(summary.delivered_evidence_ids, ("ev_1",))

    def test_dialogue_ids_require_token_boundaries_and_sample_scope(self):
        evidence = StandardEvidence("e", "sample_A", "D7:5", kind="dialogue_id")
        chunks = [
            ChunkDocument(URI, "sample_A", "[D7:50] [D17:5] [D7:5]"),
            ChunkDocument(URI + ".other", "sample_B", "[D7:5]"),
        ]
        matcher = EvidenceMatcher([evidence], chunks)
        self.assertEqual(len(matcher.manifest.locations), 1)
        self.assertEqual(matcher.manifest.locations[0].chunk_uri, URI)
        source = SourceObservation(
            URI, returned_fragments=(TextFragment("[D7:50]"),), delivered_fragments=()
        )
        self.assertFalse(matcher.annotate_step(make_step(source)).sources[0].returned_evidence_ids)
        other = replace(source, uri=URI + ".other", returned_fragments=(TextFragment("[D7:5]"),))
        self.assertFalse(matcher.annotate_step(make_step(other)).sources[0].returned_evidence_ids)

    def test_revision_scope_and_failure_statuses(self):
        for kwargs, status in (
            ({"content_hash": content_hash("changed")}, "stale_source"),
            ({"source_doc_id": "another_course"}, "source_mismatch"),
            ({"success": False}, "failed"),
        ):
            with self.subTest(status=status):
                source = self.annotate(
                    returned_fragments=(TextFragment(TEXT),),
                    delivered_fragments=(TextFragment(TEXT),),
                    **kwargs,
                )
                self.assertEqual(source.status, status)
                self.assertFalse(source.delivered_evidence_ids)
        unknown = SourceObservation(
            URI + ".unknown", source_doc_id="course", returned_fragments=(TextFragment(TEXT),)
        )
        self.assertEqual(
            self.matcher.annotate_step(make_step(unknown)).sources[0].status, "unknown_source"
        )
        self.assertEqual(self.annotate().revision_status, "unverified")
        self.assertEqual(
            self.matcher.annotate_step(make_step(SourceObservation(URI), success=False))
            .sources[0]
            .status,
            "failed",
        )

    def test_missing_delivery_is_distinct_from_empty_delivery(self):
        unknown = self.annotate(returned_fragments=(TextFragment(TEXT),))
        empty = self.annotate(delivered_fragments=())
        self.assertFalse(unknown.delivery_captured)
        self.assertTrue(empty.delivery_captured)

    def test_manifest_round_trip_and_invalid_inputs(self):
        data = json.loads(json.dumps(self.matcher.manifest.to_dict()))
        restored = EvidenceMatcher.from_manifest(EvidenceManifest.from_dict(data))
        self.assertEqual(restored.manifest, self.matcher.manifest)
        step = make_step(SourceObservation(URI))
        self.assertEqual(restored.annotate_step(step), self.matcher.annotate_step(step))
        with self.assertRaises(ValueError):
            EvidenceMatcher([self.evidence, self.evidence], [])
        with self.assertRaises(ValueError):
            EvidenceMatcher(
                [],
                [
                    ChunkDocument(URI, "course", TEXT),
                    ChunkDocument(URI + "/", "course", TEXT),
                ],
            )
        with self.assertRaises(ValueError):
            EvidenceManifest.from_dict({**data, "matcher_version": "future"})
        bad = replace(self.matcher.manifest, unresolved_evidence_ids=("ev_1",))
        with self.assertRaises(ValueError):
            EvidenceMatcher.from_manifest(bad)


class TraceRecordingTests(unittest.TestCase):
    def setUp(self):
        self.matcher = EvidenceMatcher(
            [
                StandardEvidence("a", "course", TEXT, answer_group_id="annotator_1"),
                StandardEvidence("b", "course", "Not present", answer_group_id="annotator_2"),
                StandardEvidence("r", "course", "Reasoning", kind="reasoning_annotation"),
            ],
            [ChunkDocument(URI, "course", TEXT)],
        )
        self.source = SourceObservation(
            URI, returned_fragments=(TextFragment(TEXT),), delivered_fragments=(TextFragment(TEXT),)
        )

    def test_summary_keeps_unresolved_denominator_and_answer_groups(self):
        candidate = SourceObservation(URI, content_kind="candidate")
        records = [
            self.matcher.annotate_step(make_step(self.source, sequence=3)),
            self.matcher.annotate_step(make_step(candidate, sequence=2)),
        ]
        summary = summarize_trace(self.matcher, records)
        self.assertEqual(summary.unresolved_evidence_ids, ("b",))
        self.assertEqual(summary.excluded_evidence_ids, ("r",))
        self.assertEqual(summary.located_recall, 0.5)
        self.assertEqual(summary.delivered_recall, 0.5)
        self.assertEqual(summary.first_located_step, 2)
        self.assertEqual(summary.first_delivered_step, 3)
        self.assertEqual(
            summary.answer_group_delivered_recall,
            {"annotator_1": 1.0, "annotator_2": 0.0},
        )
        self.assertEqual(summary.delivery_uncaptured_source_count, 0)

    def test_not_applicable_is_not_zero(self):
        for evidences, status in (
            ([], "no_gold_evidence"),
            (
                [StandardEvidence("r", "course", "Reasoning", kind="reasoning_annotation")],
                "no_applicable_evidence",
            ),
        ):
            summary = summarize_trace(EvidenceMatcher(evidences, []), [])
            self.assertEqual(summary.status, status)
            self.assertIsNone(summary.delivered_recall)
        self.assertEqual(summarize_trace(self.matcher, []).delivered_recall, 0.0)

    def test_uncaptured_delivery_is_reported(self):
        source = replace(self.source, delivered_fragments=None)
        summary = summarize_trace(self.matcher, [self.matcher.annotate_step(make_step(source))])
        self.assertEqual(summary.delivery_uncaptured_source_count, 1)
        self.assertEqual(summary.delivered_recall, 0.0)

    def test_jsonl_round_trip_and_same_iteration_multiple_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "steps.jsonl"
            with EvidenceTraceRecorder(path, self.matcher) as recorder:
                first = recorder.record(make_step(self.source, sequence=1, iteration=2))
                second = recorder.record(make_step(self.source, sequence=2, iteration=2))
                self.assertEqual(second.cumulative_delivered_evidence_ids, ("a",))
                self.assertEqual(recorder.summary().delivered_recall, 0.5)
                log = recorder.format_log(first)
                self.assertIn("call=call_1", log)
                self.assertIn("delivered=['a']", log)
                with self.assertRaises(ValueError):
                    recorder.record(make_step(self.source, sequence=3, call_id="call_1"))
                with self.assertRaises(ValueError):
                    recorder.record(make_step(self.source, sequence=1, call_id="new_call"))
                with self.assertRaises(ValueError):
                    recorder.record(
                        replace(make_step(self.source, sequence=3), case_id="other_case")
                    )
            data = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(data), 2)
            self.assertEqual(StepEvidenceRecord.from_dict(data[0]), first)
            self.assertEqual(StepEvidenceRecord.from_dict(data[1]), second)
            with self.assertRaises(FileExistsError):
                EvidenceTraceRecorder(path, self.matcher)
            with self.assertRaises(ValueError):
                recorder.record(make_step(self.source, sequence=3))

    def test_trace_summary_rejects_mixed_responses_and_duplicate_calls(self):
        record = self.matcher.annotate_step(make_step(self.source))
        with self.assertRaises(ValueError):
            summarize_trace(self.matcher, [record, record])
        other = replace(record, step=replace(record.step, response_id="other"))
        with self.assertRaises(ValueError):
            summarize_trace(self.matcher, [record, other])

    def test_partial_tool_failure_preserves_successful_source(self):
        failed = replace(self.source, success=False)
        record = self.matcher.annotate_step(make_step(self.source, failed))
        self.assertEqual(record.sources[0].delivered_evidence_ids, ("a",))
        self.assertEqual(record.sources[1].status, "failed")
        self.assertEqual(summarize_trace(self.matcher, [record]).delivered_recall, 0.5)

    def test_serialization_failure_does_not_advance_cumulative_state(self):
        with tempfile.TemporaryDirectory() as directory:
            with EvidenceTraceRecorder(Path(directory) / "steps.jsonl", self.matcher) as recorder:
                with self.assertRaises(TypeError):
                    recorder.record(make_step(self.source, args={"not_json": object()}))
                self.assertEqual(recorder.summary().delivered_recall, 0.0)
                recorder.record(make_step(self.source))
                self.assertEqual(recorder.summary().delivered_recall, 0.5)

    def test_legacy_qa_still_constructs_without_structured_evidence(self):
        from benchmark.RAG.src.adapters.base import StandardQA

        qa = StandardQA("Question?", ["Answer"], ["D7:5"])
        self.assertEqual(qa.evidence, ["D7:5"])
        self.assertEqual(qa.gold_evidence, [])


if __name__ == "__main__":
    unittest.main()
