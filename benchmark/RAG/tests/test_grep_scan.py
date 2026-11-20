import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import scan_grep_recall as scan


class GrepScanTests(unittest.TestCase):
    def setUp(self):
        self.uri = "viking://resources/course/a.md"
        self.documents = {self.uri: "Section: Laboratory\nlaboratory equipment\n"}

    def test_generated_prefixes_exist_and_file_root_results_differ(self):
        documents = {**self.documents, "viking://resources/course/b.md": "Laboratory\n"}
        cases = scan.build_cases(documents, ["viking://resources/course"], 4)
        patterns = {case["pattern"] for case in cases}
        self.assertTrue({"Laboratory", "Laborator", "Laborato"} <= patterns)
        for case in cases:
            self.assertTrue(case["expected"])
        file_case = next(case for case in cases if case["pattern"] == "Laboratory"
                         and case["scope_uri"] == self.uri)
        root_case = next(case for case in cases if case["pattern"] == "Laboratory"
                         and case["scope_type"] == "root")
        self.assertEqual(len(file_case["expected"]), 1)
        self.assertEqual(len(root_case["expected"]), 2)

    def test_no_cross_scope_or_case_insensitive_ground_truth(self):
        documents = {**self.documents, "viking://resources/course-other/c.md": "Laboratory"}
        expected = scan.expected_matches(documents, "viking://resources/course", "Laboratory")
        self.assertEqual(len(expected), 1)
        self.assertEqual(expected[0]["line"], 1)

    def test_true_missing_lines_caps_and_errors_are_distinct(self):
        case = {"expected": scan.expected_matches(self.documents, self.uri, "Laborator")}
        self.assertEqual(scan.compare(case, {"matches": []})["status"], "miss")
        for flag in ("limit_reached", "candidate_limit_reached"):
            self.assertEqual(scan.compare(case, {"matches": [], flag: True})["status"], "capped")
        self.assertEqual(scan.compare(case, {"matches": [
            {"uri": self.uri, "line": "1", "content": "Section: Laboratory"},
        ]})["status"], "pass")
        client = Mock()
        client.grep.side_effect = TimeoutError("timeout")
        self.assertEqual(scan.run_case(client, case, 256)["status"], "error")

    def test_miss_is_retried_and_source_rechecked(self):
        case = {"expected": scan.expected_matches(self.documents, self.uri, "Laborator")}
        client = Mock()
        client.grep.return_value = {"matches": []}
        client.request.return_value = {"result": self.documents[self.uri]}
        result = scan.run_case(client, case, 256)
        self.assertTrue(result["stable_miss"])
        self.assertEqual(result["status"], "miss")
        self.assertEqual(client.grep.call_count, 2)
        client.request.return_value = {"result": "source replaced"}
        self.assertEqual(scan.run_case(client, case, 256)["status"], "source_changed")

    def test_snapshot_walks_nested_directories_and_ignores_abstracts(self):
        client = scan.OVClient({"url": "http://localhost"})
        client.request = Mock(side_effect=[
            {"status": "ok", "result": [
                {"uri": "viking://resources/course/sub", "isDir": True},
                {"uri": "viking://resources/course/.abstract.md", "isDir": False},
            ]},
            {"status": "ok", "result": [{"uri": self.uri, "isDir": False}]},
            {"status": "ok", "result": self.documents[self.uri]},
        ])
        self.assertEqual(client.snapshot(["viking://resources/course"]), self.documents)
        self.assertEqual(client.request.call_args_list[0].args[1]["recursive"], "false")

    def test_capped_listing_prevents_incomplete_baseline(self):
        client = scan.OVClient({"url": "http://localhost"})
        client.request = Mock(return_value={"result": [], "has_more": True})
        with self.assertRaisesRegex(ValueError, "Listing capped"):
            client.snapshot(["viking://resources/course"])

    def test_sampling_is_reproducible_and_covers_distinct_files_first(self):
        documents = {f"viking://resources/course/{i}.md": "Laboratory examination assignments"
                     for i in range(4)}
        first = scan.select_words(documents, 4, 0)
        self.assertEqual(first, scan.select_words(documents, 4, 0))
        self.assertEqual(len({uri for uri, word in first}), 4)

    def test_requests_reuse_connections_and_close_pool(self):
        client = scan.OVClient({"url": "http://localhost"})
        pooled = Mock()
        pooled.request.return_value.is_error = False
        pooled.request.return_value.json.return_value = {"status": "ok", "result": {}}
        with patch("httpx.Client", return_value=pooled) as factory:
            client.request("/health")
            client.request("/api/v1/search/grep", body={"pattern": "Laborator"})
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(pooled.request.call_count, 2)
            client.close()
            pooled.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
