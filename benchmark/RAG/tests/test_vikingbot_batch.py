import csv
import json
import multiprocessing
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_vikingbot_evidence as batch
from adapters.base import StandardQA, StandardSample


class BatchTests(unittest.TestCase):
    def test_case_directories_show_qid_query_and_are_order_independent(self):
        samples = [StandardSample("course", [
            StandardQA('Based on the syllabus "course", How can I get extra credit?', ["a"],
                       metadata={"id": "G4zZKP3yeU0rVsqy"}),
            StandardQA("What is the exam date?", ["b"], metadata={"id": "exam-1"}),
        ])]
        cases = batch.select_cases(samples, "id")
        planned = batch.attach_case_paths(cases, "/tmp/run")
        reversed_plan = batch.attach_case_paths(list(reversed(cases)), "/tmp/run")
        self.assertEqual(planned[0]["case_dir"], reversed_plan[1]["case_dir"])
        self.assertTrue(planned[0]["case_name"].startswith(
            "G4zZKP3yeU0rVsqy__how-can-i-get-extra-credit__"
        ))
        self.assertEqual(planned[0]["result_path"], str(
            Path(planned[0]["case_dir"]) / "result.json"
        ))

    def test_directory_labels_handle_unsafe_unicode_and_colliding_ids(self):
        samples = [StandardSample(source, [
            StandardQA("很长的问题？ / \n" * 100, ["a"], metadata={"id": "../../qid/../" * 100}),
        ]) for source in ("a", "b")]
        planned = batch.attach_case_paths(batch.select_cases(samples, "id"), "/tmp/run")
        self.assertNotEqual(planned[0]["case_name"], planned[1]["case_name"])
        for case in planned:
            self.assertLess(len(case["case_name"].encode("utf-8")), 255)
            self.assertEqual(Path(case["case_dir"]).parent, Path("/tmp/run").resolve() / "cases")
            self.assertNotIn("/", case["case_name"])
            self.assertNotIn("\n", case["case_name"])
        self.assertEqual(batch.filename_label("../", 64), "case")

    def test_duplicate_case_identity_is_rejected(self):
        case = batch.select_cases([StandardSample("doc", [StandardQA("q", ["a"])])], "id")[0]
        with self.assertRaisesRegex(ValueError, "Duplicate case identity"):
            batch.attach_case_paths([case, case], "/tmp/run")

    def test_case_index_preserves_full_query_and_searchable_path(self):
        with tempfile.TemporaryDirectory() as directory:
            cases = batch.select_cases([StandardSample("doc", [
                StandardQA("How can I get\nextra credit?", ["a"], metadata={"id": "qid"}),
            ])], "id")
            planned = batch.attach_case_paths(cases, directory)
            batch.write_case_index(directory, "run", planned)
            index = json.loads((Path(directory) / "index.json").read_text())
            self.assertEqual(index["cases"][0]["question"], "How can I get\nextra credit?")
            self.assertEqual(index["cases"][0]["result_path"], planned[0]["result_path"])
            tsv = (Path(directory) / "index.tsv").read_text()
            self.assertEqual(len(tsv.splitlines()), 2)
            self.assertIn("extra credit?", tsv.splitlines()[1])
            self.assertIn(planned[0]["result_path"], tsv.splitlines()[1])
            with (Path(directory) / "index.tsv").open(newline="") as file:
                row = list(csv.DictReader(file, delimiter="\t"))[0]
            self.assertEqual(row["case_id"], "qid")
            self.assertEqual(row["question"], "How can I get extra credit?")

    def test_selection_supports_native_and_fallback_ids(self):
        samples = [StandardSample("doc", [
            StandardQA("q1", ["a"], metadata={"id": "native"}),
            StandardQA("q2", ["b"]),
        ])]
        self.assertEqual(batch.select_cases(samples, "id", "native")[0]["qa"].question, "q1")
        self.assertEqual(batch.select_cases(samples, "id", "doc:qa_1")[0]["case_id"], "doc:qa_1")
        self.assertEqual(len(batch.select_cases(samples, "id")), 2)
        with self.assertRaisesRegex(ValueError, "No matching"):
            batch.select_cases(samples, "id", "unknown")

    def test_ambiguous_ids_require_document_filter(self):
        samples = [
            StandardSample(doc, [StandardQA("q", ["a"], metadata={"original_id": 3})])
            for doc in ("a", "b")
        ]
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            batch.select_cases(samples, "original_id", "3")
        self.assertEqual(batch.select_cases(samples, "original_id", "3", "b")[0]["source_doc_id"], "b")

    def test_source_mapping_import_report_and_explicit_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.json"
            batch.write_json(path, {"resources": [
                {"file": "course.pdf", "uri": "viking://resources/course/"},
                {"source_doc_id": "paper-id", "import_result": {"root_uri": "viking://resources/paper"}},
            ]})
            self.assertEqual(batch.load_source_map(path), {
                "course": "viking://resources/course", "paper-id": "viking://resources/paper",
            })
            batch.write_json(path, {"paper": "viking://resources/paper"})
            self.assertEqual(batch.load_source_map(path)["paper"], "viking://resources/paper")
            batch.write_json(path, {"paper": "http://bad"})
            with self.assertRaises(ValueError):
                batch.load_source_map(path)
            batch.write_json(path, [
                {"source_doc_id": "same", "uri": "viking://resources/a"},
                {"source_doc_id": "same", "uri": "viking://resources/b"},
            ])
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                batch.load_source_map(path)

    def test_runtime_isolation_preserves_original_configuration(self):
        config = {
            "storage": {"workspace": "original"},
            "bot": {"rag_checkpoints": {}, "ov_server": {
                "admin_user_id": "original", "server_url": "http://127.0.0.1:11933",
                "api_key": "test-key",
            }},
        }
        first, second = (batch.runtime_config(config, path) for path in ("/tmp/a", "/tmp/b"))
        self.assertNotEqual(first["storage"]["workspace"], second["storage"]["workspace"])
        self.assertNotEqual(first["bot"]["ov_server"]["admin_user_id"],
                            second["bot"]["ov_server"]["admin_user_id"])
        self.assertEqual(config["bot"]["ov_server"]["admin_user_id"], "original")
        self.assertIn("rag_checkpoints", config["bot"])
        self.assertNotIn("rag_checkpoints", first["bot"])
        self.assertEqual(first["bot"]["mode"], "readonly")
        self.assertEqual(first["bot"]["ov_server"]["api_key"], "test-key")

    def test_spawn_worker_failure_is_saved_without_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            case = batch.attach_case_paths(batch.select_cases([
                StandardSample("doc", [StandardQA("q", ["a"], metadata={"id": "q"})]),
            ], "id"), directory)[0]
            task = {
                "snapshot": str(Path(directory) / "missing.json"), "output": case["case_dir"],
                "case_id": "q", "canonical_id": "doc:qa_0", "source_doc_id": "doc",
                "resource_uri": "viking://resources/doc", "run_id": "test",
                "qa": StandardQA("q", ["a"]),
            }
            with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
                result = pool.submit(batch.execute_case, task).result(timeout=30)
            self.assertEqual(result["status"], "failed")
            self.assertNotEqual(result["worker_pid"], os.getpid())
            stored = json.loads(Path(case["result_path"]).read_text())
            self.assertEqual(stored["failure_reason"], "execution_error")
            self.assertEqual(stored["gold_answers"], ["a"])
            self.assertEqual(result["question"], "q")
            self.assertEqual(result["case_dir"], case["case_dir"])

    def test_missing_source_map_prevents_model_and_server_calls(self):
        samples = [StandardSample("unmapped", [StandardQA("q", ["a"], metadata={"id": "1"})])]
        adapter = unittest.mock.Mock()
        adapter.load_and_transform.return_value = samples
        module = unittest.mock.Mock(SyllabusQAAdapter=unittest.mock.Mock(return_value=adapter))
        with patch.object(batch.importlib, "import_module", return_value=module), \
             patch.object(batch, "load_source_map", return_value={}), \
             patch.object(batch, "snapshot_resource") as snapshot, \
             patch.object(sys, "argv", ["run", "--dataset", "syllabusqa", "--all"]):
            with self.assertRaisesRegex(ValueError, "Missing source mappings"):
                batch.main()
            snapshot.assert_not_called()

    def test_judge_reuses_dataset_rubric_and_records_score_and_usage(self):
        response = SimpleNamespace(
            content='{"score": 3, "reasoning": "A minor omission."}',
            usage_metadata={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
        )
        client = unittest.mock.Mock()
        client.invoke.return_value = response
        settings = {"model": "test-model", "base_url": "http://localhost/v1",
                    "api_key": "test-key", "timeout": 12}
        answer = {"status": "completed", "question": "question", "gold_answers": ["gold"],
                  "answer": "generated"}
        with patch("langchain_openai.ChatOpenAI", return_value=client) as factory:
            evaluation = batch.evaluate_answer(answer, "syllabusqa", settings)
        self.assertEqual(evaluation["status"], "completed")
        self.assertEqual(evaluation["score"], 3)
        self.assertEqual(evaluation["normalized_score"], .75)
        self.assertEqual(evaluation["prompt_type"], "Generic_0-4")
        self.assertEqual(evaluation["token_usage"]["total_tokens"], 30)
        factory.assert_called_once_with(
            model="test-model", temperature=0, base_url="http://localhost/v1",
            api_key="test-key", timeout=12, max_retries=2,
        )
        prompt = client.invoke.call_args.args[0][1].content
        self.assertIn("Gold Answers: gold", prompt)
        self.assertIn("Generated Answer: generated", prompt)

    def test_judge_failure_is_distinct_from_zero_score(self):
        settings = {"model": "test", "base_url": "http://localhost/v1",
                    "api_key": "test", "timeout": 1}
        answer = {"status": "completed", "question": "q", "gold_answers": ["a"], "answer": "b"}
        client = unittest.mock.Mock()
        for raw in ("invalid", '{"score": 8, "reasoning": "invalid"}', '{"reasoning": "no score"}'):
            client.invoke.return_value = SimpleNamespace(content=raw, usage_metadata=None)
            with patch("langchain_openai.ChatOpenAI", return_value=client):
                evaluation = batch.evaluate_answer(answer, "syllabusqa", settings)
            self.assertEqual(evaluation["status"], "failed")
            self.assertIsNone(evaluation["score"])
            self.assertEqual(evaluation["raw_output"], raw)
        client.invoke.side_effect = TimeoutError("timed out")
        with patch("langchain_openai.ChatOpenAI", return_value=client):
            evaluation = batch.evaluate_answer(answer, "syllabusqa", settings)
        self.assertEqual(evaluation["status"], "failed")
        self.assertIsNone(evaluation["score"])
        self.assertIn("timed out", evaluation["error"])

    def test_judge_skips_failed_agent_disabled_and_missing_gold(self):
        settings = {"model": "test"}
        with patch("langchain_openai.ChatOpenAI") as factory:
            self.assertEqual(batch.evaluate_answer({}, "syllabusqa", None)["reason"], "disabled")
            self.assertEqual(batch.evaluate_answer(
                {"status": "failed"}, "syllabusqa", settings
            )["reason"], "agent_failed")
            self.assertEqual(batch.evaluate_answer(
                {"status": "completed", "gold_answers": []}, "syllabusqa", settings
            )["reason"], "no_gold_answers")
            factory.assert_not_called()

    def test_judge_aggregation_excludes_failures_and_skips(self):
        results = [{"llm_evaluation": evaluation} for evaluation in (
            {"status": "completed", "score": 0, "normalized_score": 0},
            {"status": "completed", "score": 4, "normalized_score": 1},
            {"status": "failed", "score": None, "normalized_score": None},
            {"status": "skipped"},
        )]
        self.assertEqual(batch.aggregate_judges(results), {
            "completed": 2, "failed": 1, "skipped": 1,
            "average_score": 2, "average_normalized_score": .5,
        })
        self.assertIsNone(batch.aggregate_judges([])["average_score"])

    def test_judge_model_and_endpoint_defaults_and_overrides(self):
        args = SimpleNamespace(judge_model=None, judge_base_url=None, judge_timeout=60)
        config = {"vlm": {"model": "agent", "api_key": "test", "provider": "volcengine",
                          "api_base": "https://example.test/v3"}}
        settings = batch.judge_settings(config, args)
        self.assertEqual(settings["model"], "agent")
        self.assertEqual(settings["base_url"], "https://example.test/v3")
        args.judge_model, args.judge_base_url = "judge", "https://judge.test/v1"
        settings = batch.judge_settings(config, args)
        self.assertEqual(settings["model"], "judge")
        self.assertEqual(settings["base_url"], "https://judge.test/v1")

    def test_legacy_grader_fallback_and_locomo_strict_scoring(self):
        from core.judge_util import llm_grader
        client = unittest.mock.Mock()
        client.invoke.return_value = SimpleNamespace(content="Score: 3")
        self.assertEqual(llm_grader(client, "test", "q", ["a"], "b", "syllabusqa")["score"], 3)
        with self.assertRaises(ValueError):
            llm_grader(client, "test", "q", ["a"], "b", "syllabusqa", strict=True)
        client.invoke.return_value = SimpleNamespace(content='{"score": 3, "reasoning": "partial"}')
        with self.assertRaises(ValueError):
            llm_grader(client, "test", "q", ["a"], "b", "locomo", strict=True)
        client.invoke.return_value = SimpleNamespace(content='{"score": 4, "reasoning": "correct"}')
        self.assertEqual(llm_grader(client, "test", "q", ["a"], "b", "locomo", strict=True)["score"], 4)


if __name__ == "__main__":
    unittest.main()
