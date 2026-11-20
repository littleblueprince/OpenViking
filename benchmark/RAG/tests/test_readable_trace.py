"""Verify readable logs preserve evidence semantics while shortening only display text."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from benchmark.RAG.src.core.evidence_matcher import EvidenceMatcher
from benchmark.RAG.src.core.readable_trace import build_readable_trace, readable_step
from benchmark.RAG.src.core.trace_annotation import EvidenceTraceRecorder, summarize_trace
from benchmark.RAG.src.core.trace_models import (
    ChunkDocument, SourceObservation, StandardEvidence, TextFragment, ToolStep,
)

URI = "viking://resources/course/chunk.md"
BODY = "Preface. " + "x " * 100 + "gold span"


class ReadableTraceTests(unittest.TestCase):
    def setUp(self):
        self.matcher = EvidenceMatcher(
            [StandardEvidence("e", "course", "gold span")], [ChunkDocument(URI, "course", BODY)]
        )

    def record(self, *sources, sequence=1, success=True):
        return self.matcher.annotate_step(ToolStep(
            case_id="case", run_id="run", session_id="session", response_id="response",
            tool_call_id=f"call_{sequence}", tool_name="openviking_multi_read",
            iteration=2, sequence=sequence, sources=tuple(sources), success=success,
        ))

    def test_hit_is_based_on_full_text_not_short_preview(self):
        record = self.record(SourceObservation(
            URI, returned_fragments=(TextFragment(BODY),),
            delivered_fragments=(TextFragment("Preface."),), truncated=True,
        ))
        view = readable_step(record, preview_chars=8)
        chunk = view["chunks"][0]
        self.assertEqual(chunk["content_preview"], "Preface.")
        self.assertTrue(chunk["hit"])
        self.assertTrue(chunk["located_hit"])
        self.assertFalse(chunk["delivered_hit"])
        self.assertTrue(chunk["preview_truncated"])
        self.assertEqual(chunk["evidence_ids"], ["e"])
        self.assertEqual(chunk["hit_marker"], "HIT")

    def test_directory_and_candidate_are_not_text_hits(self):
        record = self.record(
            SourceObservation("viking://resources/course", is_directory=True),
            SourceObservation(URI, content_kind="candidate", delivered_fragments=()),
        )
        view = readable_step(record)
        directory, candidate = view["chunks"]
        self.assertEqual(directory["hit_marker"], "ANCESTOR_ONLY")
        self.assertEqual(directory["source_kind"], "directory")
        self.assertEqual(candidate["hit_marker"], "LOCATED_ONLY")
        self.assertEqual(candidate["content_preview"], "")
        self.assertFalse(candidate["hit"])
        self.assertTrue(candidate["located_hit"])

    def test_unknown_failed_and_uncaptured_are_not_hidden_as_misses(self):
        good = SourceObservation(URI, returned_fragments=(TextFragment(BODY),))
        view = readable_step(self.record(
            SourceObservation(URI + ".unknown"), replace(good, success=False), good,
        ))
        unknown, failed, uncaptured = view["chunks"]
        self.assertIsNone(unknown["hit"])
        self.assertEqual(unknown["hit_marker"], "UNKNOWN_SOURCE")
        self.assertIsNone(failed["hit"])
        self.assertEqual(failed["hit_marker"], "FAILED")
        self.assertTrue(uncaptured["hit"])
        self.assertIsNone(uncaptured["delivered_hit"])

    def test_fragment_boundaries_are_visible(self):
        record = self.record(SourceObservation(
            URI, returned_fragments=(TextFragment("gold"), TextFragment("span")),
            delivered_fragments=(),
        ))
        chunk = readable_step(record)["chunks"][0]
        self.assertIn("[separate fragment]", chunk["content_preview"])
        self.assertEqual(chunk["returned_fragment_count"], 2)
        self.assertFalse(chunk["hit"])

    def test_expanded_json_orders_calls_and_counts_tools_not_iterations(self):
        source = SourceObservation(
            URI, returned_fragments=(TextFragment(BODY),), delivered_fragments=(),
        )
        records = [self.record(source, sequence=2), self.record(source, sequence=1)]
        view = build_readable_trace(
            records, summarize_trace(self.matcher, records).to_dict(),
            preview_chars=10, tool_results={"call_1": "tool header and a long result"},
            context={"question": "Which?", "model_iterations": 3},
        )
        self.assertEqual(view["total_steps"], 2)
        self.assertEqual(view["model_iterations"], 3)
        self.assertEqual(view["hit_step_count"], 2)
        self.assertEqual([step["step"] for step in view["steps"]], [1, 2])
        self.assertEqual(view["steps"][0]["result_preview"], "tool heade")
        self.assertNotIn("sources", view["steps"][0])

    def test_recorder_exports_json_and_clear_per_chunk_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with EvidenceTraceRecorder(path / "raw.jsonl", self.matcher) as recorder:
                record = recorder.record(ToolStep(
                    case_id="case", run_id="run", session_id="session", response_id="response",
                    tool_call_id="call", tool_name="read", iteration=1, sequence=1,
                    sources=(SourceObservation(
                        URI, returned_fragments=(TextFragment(BODY),), delivered_fragments=(),
                    ),),
                ))
                recorder.export_readable_json(path / "readable.json", preview_chars=12)
                with self.assertRaises(FileExistsError):
                    recorder.export_readable_json(path / "readable.json")
                log = recorder.format_log(record)
                self.assertEqual(log.count("[RAG Evidence]"), 1)
                self.assertIn("[HIT] chunk=", log)
                self.assertIn('preview="Preface.', log)
            exported = json.loads((path / "readable.json").read_text())
            self.assertEqual(exported["total_steps"], 1)
            self.assertLessEqual(len(exported["steps"][0]["chunks"][0]["content_preview"]), 12)

    def test_zero_steps_and_invalid_preview_limit(self):
        view = build_readable_trace(
            [], summarize_trace(self.matcher, []).to_dict(), context={"case_id": "case"}
        )
        self.assertEqual(view["total_steps"], 0)
        self.assertEqual(view["case_id"], "case")
        for limit in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                readable_step(self.record(), preview_chars=limit)


if __name__ == "__main__":
    unittest.main()
