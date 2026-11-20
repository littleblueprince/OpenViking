"""Adapter and native tool-event integration, with no server or model calls."""

import asyncio
import csv
import hashlib
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bot"))
sys.path.insert(0, str(ROOT / "benchmark/RAG/src"))

from adapters.financebench_adapter import FinanceBenchAdapter
from adapters.locomo_adapter import LocomoAdapter
from adapters.qasper_adapter import QasperAdapter
from adapters.syllabusqa_adapter import SyllabusQAAdapter
from core.evidence_matcher import EvidenceMatcher
from core.trace_annotation import EvidenceTraceRecorder
from core.trace_models import ChunkDocument, StandardEvidence
from core.tool_event_bridge import ToolEventEvidenceBridge
from vikingbot.agent.loop.loop_compaction import LoopCompactionMixin
from vikingbot.agent.loop.loop_tools import LoopToolMixin, _LoopRequestContext
from vikingbot.agent.tools.base import ToolContext
from vikingbot.agent.tools.ov_file import (
    VikingGlobTool, VikingGrepTool, VikingListTool, VikingMultiReadTool, VikingSearchTool,
)
from vikingbot.agent.tools.registry import ToolRegistry
from vikingbot.agent.tools.source_capture import (
    capture_json_sources, delivered_sources, source_observation,
)
from vikingbot.config.schema import SessionKey
from vikingbot.utils.tracing import set_response_id
import vikingbot_runner

URI = "viking://resources/course/chunk.md"
SESSION = SessionKey(type="test", channel_id="channel", chat_id="case")


class AdapterTests(unittest.TestCase):
    def load(self, cls, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return cls(str(path)).load_and_transform()

    def test_locomo_document_scope_and_dialogue_ids(self):
        qa = self.load(LocomoAdapter, [{
            "sample_id": "sample-1",
            "qa": [{"question": "When?", "answer": "Tuesday", "evidence": ["D7:5"]}],
        }])[0].qa_pairs[0]
        self.assertEqual(qa.evidence, ["D7:5"])
        self.assertEqual(qa.gold_evidence[0].source_doc_id, "sample-1")
        self.assertEqual(qa.gold_evidence[0].kind, "dialogue_id")

    def test_syllabus_json_csv_preserve_annotation_numbers(self):
        row = {
            "syllabus_name": "course", "question": "When?", "answer": "Tuesday",
            "answer_span_2": "Tuesday", "reasoning_step_3": "Infer Tuesday",
        }
        qa = self.load(SyllabusQAAdapter, {"course": [row]})[0].qa_pairs[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            csv_qa = SyllabusQAAdapter(str(path)).load_and_transform()[0].qa_pairs[0]
        self.assertEqual(qa.gold_evidence, csv_qa.gold_evidence)
        self.assertEqual([e.evidence_id for e in qa.gold_evidence], [
            "answer_span_2", "reasoning_step_3",
        ])
        self.assertEqual(qa.gold_evidence[1].kind, "reasoning_annotation")
        self.assertEqual(qa.evidence, ["Tuesday", "Infer Tuesday"])

    def test_qasper_preserves_duplicate_evidence_in_distinct_answer_groups(self):
        qa = self.load(QasperAdapter, {"paper-1": {
            "qas": [{"question": "What?", "answers": [
                {"answer": {"free_form_answer": "a", "evidence": ["same span"]}},
                {"answer": {"free_form_answer": "b", "evidence": ["same span"]}},
            ]}],
        }})[0].qa_pairs[0]
        self.assertEqual(qa.evidence, ["same span"])
        self.assertEqual(len(qa.metadata["answer_evidence_pairs"]), 2)
        self.assertEqual([e.answer_group_id for e in qa.gold_evidence], [
            "answer_1", "answer_2",
        ])
        self.assertTrue(all(e.source_doc_id == "paper-1" for e in qa.gold_evidence))

    def test_finance_preserves_document_and_page_annotation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_text(json.dumps({
                "doc_name": "report", "question": "What?", "answer": "15 USD",
                "evidence": [{"evidence_text": "15 USD", "evidence_page_num": 8}],
            }) + "\n", encoding="utf-8")
            qa = FinanceBenchAdapter(str(path)).load_and_transform()[0].qa_pairs[0]
        self.assertEqual(qa.gold_evidence[0].source_doc_id, "report")
        self.assertEqual(qa.metadata["evidence_annotations"][0]["evidence_page_num"], 8)


class NativeSourceTests(unittest.IsolatedAsyncioTestCase):
    def tool(self, cls, client):
        tool = cls()
        tool._get_client = AsyncMock(return_value=client)
        tool._release_client = AsyncMock()
        return tool

    async def test_list_glob_candidates_and_search_abstract(self):
        client = SimpleNamespace(
            list_resources=AsyncMock(return_value=[{
                "name": "chunk.md", "size": 10, "uri": URI, "isDir": False,
            }]),
            glob=AsyncMock(return_value={"matches": [URI], "count": 1}),
            search=AsyncMock(return_value={"resources": [{
                "uri": URI, "abstract": 'Tuesday\n"lecture"', "is_leaf": True, "score": 1,
            }]}),
        )
        for cls, args in (
            (VikingListTool, {"uri": URI.rsplit("/", 1)[0]}),
            (VikingGlobTool, {"pattern": "*.md", "uri": URI.rsplit("/", 1)[0]}),
            (VikingSearchTool, {"query": "lecture", "target_uri": URI}),
        ):
            context = ToolContext(session_key=SESSION)
            result = await self.tool(cls, client).execute(context, **args)
            sources = delivered_sources(context.source_observations, result, result)
            self.assertEqual(sources[0]["uri"], URI)
            if cls == VikingSearchTool:
                self.assertEqual(sources[0]["delivered_fragments"], [
                    {"text": 'Tuesday\n"lecture"'},
                ])
            else:
                self.assertEqual(sources[0]["content_kind"], "candidate")
                self.assertEqual(sources[0]["delivered_fragments"], [])

    async def test_grep_disjoint_lines_remain_separate(self):
        client = SimpleNamespace(grep=AsyncMock(return_value={
            "matches": [
                {"uri": URI, "line": 2, "content": "first half"},
                {"uri": URI, "line": 8, "content": "second half"},
            ], "limit_reached": True,
        }))
        context = ToolContext(session_key=SESSION)
        result = await self.tool(VikingGrepTool, client).execute(context, pattern="half", uri=URI)
        sources = delivered_sources(context.source_observations, result, result)
        self.assertEqual(sources[0]["delivered_fragments"], [
            {"text": "first half"}, {"text": "second half"},
        ])
        self.assertTrue(sources[0]["limit_reached"])

    async def test_multi_read_revision_partial_failures_and_hash(self):
        raw = b"first\nTuesday\nlast\n"
        client = SimpleNamespace(
            download_bytes=AsyncMock(side_effect=[raw, ValueError("not found")]),
        )
        context = ToolContext(session_key=SESSION)
        result = await self.tool(VikingMultiReadTool, client).execute(
            context, uris=[URI, URI + ".missing"], with_revision=True, offset=1, limit=1,
        )
        sources = delivered_sources(context.source_observations, result, result)
        self.assertEqual(sources[0]["content_hash"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(sources[0]["delivered_fragments"], [{"text": "Tuesday\n"}])
        self.assertFalse(sources[1]["success"])
        self.assertEqual(sources[1]["returned_fragments"], [])
        self.assertTrue(sources[0]["truncated"])

    async def test_tool_loop_events_match_actual_preview_and_dispatch_order(self):
        class Loop(LoopToolMixin, LoopCompactionMixin):
            pass

        body = "A" * 4500 + " gold Tuesday " + "B" * 4500
        client = SimpleNamespace(stat=AsyncMock(return_value={"size": len(body)}))

        async def read(uri, **kwargs):
            if uri == URI:
                await asyncio.sleep(0.01)  # Complete the first dispatched call last.
            return body

        client.read_content = read
        registry = ToolRegistry()
        registry.register(self.tool(VikingMultiReadTool, client))
        loop = Loop()
        loop.bus = SimpleNamespace(publish_outbound=AsyncMock())
        loop.sandbox_manager = SimpleNamespace(
            get_sandbox=AsyncMock(return_value=SimpleNamespace(write_file_bytes=AsyncMock())),
            to_workspace_id=lambda session_key: "test-workspace",
        )
        model_results = []

        def add_result(messages, call_id, name, result):
            model_results.append(result)
            return messages + [{"role": "tool", "content": result}]

        loop.context = SimpleNamespace(
            add_assistant_message=lambda messages, *args, **kwargs: messages,
            add_tool_result=add_result,
        )
        calls = [
            SimpleNamespace(
                id=f"call-{index}", name="openviking_multi_read",
                arguments={"uris": [uri]}, tokens=1,
            )
            for index, uri in enumerate((URI, URI + ".second"), 1)
        ]
        used = []
        turns = []
        with set_response_id("response-1"):
            await loop._execute_tool_turn(
                SimpleNamespace(tool_calls=calls, content="", reasoning_content=""),
                [], _LoopRequestContext(
                    session_key=SESSION, sender_id=None, actor_peer_id=None,
                    memory_peer_ids=None, memory_owner_user_ids=None,
                    openviking_connection=None, channel_metadata=None, skill_runtime=None,
                ),
                active_tools=registry, visible_tool_names={"openviking_multi_read"},
                scoped_openviking_tools=None, tools_used=used, publish_events=True,
                captured_turns=turns, agent_id="agent", iteration=2,
            )
        messages = [call.args[0] for call in loop.bus.publish_outbound.await_args_list]
        self.assertEqual([msg.event_type.value for msg in messages], [
            "tool_call", "tool_call", "tool_result", "tool_result",
        ])
        self.assertEqual([msg.metadata["sequence"] for msg in messages], [1, 2, 1, 2])
        self.assertEqual([msg.metadata["tool_call_id"] for msg in messages], [
            "call-1", "call-2", "call-1", "call-2",
        ])
        self.assertEqual([msg.content for msg in messages[2:]], model_results)
        matcher = EvidenceMatcher(
            [StandardEvidence("span", "course", "gold Tuesday")],
            [ChunkDocument(URI, "course", body)],
        )
        with tempfile.TemporaryDirectory() as directory:
            with EvidenceTraceRecorder(Path(directory) / "steps.jsonl", matcher) as recorder:
                bridge = ToolEventEvidenceBridge(
                    recorder, case_id="case", run_id="run", logger=logging.getLogger("test")
                )
                for message in messages:
                    bridge.consume({
                        "event": message.event_type.value,
                        "data": {
                            "metadata": message.metadata, "response_id": message.response_id,
                            "session_id": SESSION.safe_name(), "iteration": message.iteration,
                            "timestamp": message.timestamp.isoformat()
                            if hasattr(message, "timestamp") else "2026-11-20T00:00:00",
                        },
                    })
                summary = recorder.summary()
                self.assertEqual(summary.returned_recall, 1)
                self.assertEqual(summary.delivered_recall, 0)
                self.assertEqual(summary.delivery_uncaptured_source_count, 0)
            records = [
                json.loads(line)
                for line in (Path(directory) / "steps.jsonl").read_text().splitlines()
            ]
        self.assertTrue(records[0]["sources"][0]["truncated"])
        self.assertEqual(used[0]["sources"], messages[2].metadata["sources"])
        self.assertEqual(turns[0]["tool_calls"][0]["iteration"], 2)

    async def test_hook_replacement_invalidates_captured_sources(self):
        context_client = SimpleNamespace(
            stat=AsyncMock(return_value={}),
            read_content=AsyncMock(return_value="Tuesday"),
        )
        registry = ToolRegistry()
        registry.register(self.tool(VikingMultiReadTool, context_client))
        with patch(
            "vikingbot.agent.tools.registry.hook_manager.execute_hooks",
            new=AsyncMock(return_value={"result": "redacted output"}),
        ):
            result = await registry.execute_detailed(
                "openviking_multi_read", {"uris": [URI]}, SESSION,
            )
        self.assertEqual(result.source_observations, ())


class ProjectionAndRunnerTests(unittest.TestCase):
    def test_preview_cut_inside_json_escape_drops_incomplete_character(self):
        empty = json.dumps([{"uri": URI, "content": ""}], ensure_ascii=False)
        start = empty.index('"content": "') + len('"content": "')
        value = "x" * (3999 - start) + "\nrest" + "y" * 9000
        text = json.dumps([{"uri": URI, "content": value}], ensure_ascii=False)
        self.assertEqual(text[3999:4001], "\\n")
        sources = capture_json_sources(
            text, [{"uri": URI, "content": value}], field="content", kind="body"
        )
        preview = text[:4000] + "\n\n[Output truncated. test]\n\n" + text[-2000:]
        head = delivered_sources(sources, text, preview)[0]["delivered_fragments"][0]["text"]
        self.assertEqual(head, "x" * (3999 - start))

    def test_json_escape_clipping_and_unknown_transformation(self):
        text = json.dumps([{"uri": URI, "content": 'a\n"b"' + "x" * 9000}], ensure_ascii=False)
        sources = capture_json_sources(
            text, [{"uri": URI, "content": 'a\n"b"' + "x" * 9000}],
            field="content", kind="body",
        )
        preview = text[:4000] + "\n\n[Output truncated. test]\n\n" + text[-2000:]
        captured = delivered_sources(sources, text, preview)[0]
        self.assertTrue(captured["delivered_fragments"][0]["text"].startswith('a\n"b"'))
        unknown = delivered_sources(sources, text, "redacted")[0]
        self.assertIsNone(unknown["delivered_fragments"])
        self.assertNotIn("_spans", unknown)

    def test_runner_qa_persists_manifest_summary_without_gold_in_prompt(self):
        qa = SimpleNamespace(
            question="When?", gold_evidence=[StandardEvidence("span", "course", "SECRET GOLD")]
        )
        runner = vikingbot_runner.VikingBotRunner({"vikingbot": {"manage_server": False}})

        def fake_run(**kwargs):
            self.assertNotIn("SECRET GOLD", kwargs["input_msg"])
            self.assertNotIn("gold_evidence", kwargs)
            source = source_observation(URI)
            source.pop("_spans")
            source["returned_fragments"] = [{"text": "SECRET GOLD"}]
            source["delivered_fragments"] = [{"text": "SECRET GOLD"}]
            for kind in ("tool_call", "tool_result"):
                event = {
                    "event": kind, "data": {
                        "session_id": "session", "response_id": "response", "iteration": 1,
                        "timestamp": "2026-11-20T00:00:00",
                        "metadata": {
                            "tool_call_id": "call", "tool_name": "openviking_multi_read",
                            "args": {"uris": [URI]}, "sequence": 1,
                            "success": True, "sources": [source],
                        },
                    },
                }
                kwargs["events"].append(event)
                kwargs["event_handler"](event)
            return {
                "content": "Tuesday", "response_id": "response", "time_cost": 1,
                "elapsed_seconds": 1, "token_usage": {}, "iteration": 1,
                "tools_used_names": ["openviking_multi_read"],
            }

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(vikingbot_runner, "_healthcheck", return_value=True), patch.object(
                vikingbot_runner, "_load_server_url", return_value="http://localhost:1933",
            ), patch.object(
                vikingbot_runner, "_run_vikingbot_in_process", side_effect=fake_run,
            ):
                result = runner.generate_qa(
                    qa, [ChunkDocument(URI, "course", "SECRET GOLD")],
                    case_id="case", run_id="run", trace_dir=directory,
                )
            self.assertEqual(result["answer"], "Tuesday")
            self.assertEqual(result["evidence_summary"]["delivered_recall"], 1)
            self.assertTrue(Path(result["evidence_manifest_path"]).is_file())
            readable = json.loads(Path(result["evidence_readable_path"]).read_text())
            self.assertEqual(readable["total_steps"], 1)
            self.assertEqual(readable["question"], "When?")
            self.assertEqual(readable["steps"][0]["chunks"][0]["hit"], True)
            trace = Path(result["evidence_trace_path"])
            self.assertTrue(trace.is_file())
            with trace.open("a") as file:  # Closed on completion.
                file.write("")

    def test_bridge_rejects_legacy_tool_event(self):
        bridge = ToolEventEvidenceBridge(
            None, case_id="case", run_id="run", logger=logging.getLogger("test")
        )
        with self.assertRaisesRegex(ValueError, "structured tool_call_id"):
            bridge.consume({"event": "tool_result", "data": {"value": "old result"}})

    def test_runner_timeout_preserves_partial_trace_and_pending_dispatch(self):
        matcher = EvidenceMatcher(
            [StandardEvidence("span", "course", "Tuesday")],
            [ChunkDocument(URI, "course", "Tuesday")],
        )
        runner = vikingbot_runner.VikingBotRunner({"vikingbot": {"manage_server": False}})

        def fail_run(**kwargs):
            event = {
                "event": "tool_call", "data": {
                    "session_id": "session", "response_id": "response", "iteration": 1,
                    "metadata": {
                        "tool_call_id": "pending", "tool_name": "openviking_multi_read",
                        "sequence": 1, "args": {"uris": [URI]},
                    },
                },
            }
            kwargs["events"].append(event)
            kwargs["event_handler"](event)
            raise TimeoutError("turn timed out")

        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "partial.jsonl"
            with patch.object(vikingbot_runner, "_healthcheck", return_value=True), patch.object(
                vikingbot_runner, "_load_server_url", return_value="http://localhost:1933",
            ), patch.object(
                vikingbot_runner, "_run_vikingbot_in_process", side_effect=fail_run,
            ):
                result = runner.generate_answer(
                    "When?", evidence_matcher=matcher, evidence_trace_path=trace,
                    case_id="case", run_id="run",
                )
            self.assertIn("timed out", result["error"])
            self.assertEqual(result["evidence_pending_tool_call_ids"], ["pending"])
            self.assertEqual(result["evidence_summary"]["delivered_recall"], 0)
            readable = json.loads(Path(result["evidence_readable_path"]).read_text())
            self.assertEqual(readable["total_steps"], 0)
            self.assertEqual(readable["pending_tool_call_ids"], ["pending"])
            self.assertEqual(trace.read_text(), "")
            with trace.open("a") as file:
                file.write("")


if __name__ == "__main__":
    unittest.main()
