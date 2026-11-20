import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_musique as prepare
import run_vikingbot_evidence as batch
from adapters.musique_adapter import MuSiQueAdapter, chunk_filename
from core.evidence_matcher import EvidenceMatcher
from core.trace_models import ChunkDocument


class MuSiQueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.chunks = self.root / "musique_dev_chunks.jsonl"
        self.questions = self.root / "musique_dev_answerable_subset_1_seed42.jsonl"
        self.chunk_records = [
            {"id": "a" * 64, "title": "Alpha", "text": "Alpha worked at the Beta site."},
            {"id": "b" * 64, "title": "Beta", "text": "The Beta director was Gamma."},
            {"id": "c" * 64, "title": "Distractor", "text": "An unrelated source."},
        ]
        self.question_records = [{
            "id": "2hop__test", "question": "Who directed the site where Alpha worked?",
            "answer": "Gamma", "answer_aliases": ["G", "Gamma"],
            "answerable": True, "supporting_chunk_ids": ["a" * 64, "b" * 64],
        }]
        self.save()

    def save(self):
        for path, records in ((self.chunks, self.chunk_records),
                              (self.questions, self.question_records)):
            path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    def adapter(self):
        return MuSiQueAdapter(str(self.questions))

    def test_shared_corpus_aliases_original_evidence_ids_and_no_prompt_leak(self):
        adapter = self.adapter()
        sample, = adapter.load_and_transform()
        self.assertEqual(sample.sample_id, "musique_dev")
        qa, = sample.qa_pairs
        self.assertEqual(qa.gold_answers, ["Gamma", "G"])
        self.assertEqual([e.source_doc_id for e in qa.gold_evidence], ["a" * 64, "b" * 64])
        prompt, _ = adapter.build_prompt(qa, ["retrieved context"])
        self.assertNotIn("Gamma", prompt)
        self.assertNotIn("a" * 64, prompt)
        self.assertEqual(batch.select_cases([sample], "id", "2hop__test")[0]["case_id"], "2hop__test")
        docs = adapter.data_prepare(str(self.root / "prepared"))
        self.assertEqual(len(docs), 3)  # Includes the distractor.
        for doc in docs:
            body = Path(doc.doc_path).read_text()
            self.assertNotIn(qa.question, body)
            self.assertNotIn("supporting_chunk_ids", body)
            self.assertLess(len(Path(doc.doc_path).stem), 50)
        self.assertEqual(adapter.stats()["unique_supporting_chunks"], 2)

    def test_duplicate_and_missing_references_and_schema_are_rejected(self):
        mutations = [
            ("duplicate id", lambda: self.chunk_records.append(self.chunk_records[0].copy())),
            ("missing supporting", lambda: self.question_records[0].update(supporting_chunk_ids=["missing"])),
            ("duplicate supporting", lambda: self.question_records[0].update(supporting_chunk_ids=["a" * 64] * 2)),
            ("string list", lambda: self.question_records[0].update(answer_aliases=None)),
            ("boolean", lambda: self.question_records[0].update(answerable="yes")),
            ("nonempty text", lambda: self.chunk_records[0].update(text=" ")),
        ]
        original_chunks = json.dumps(self.chunk_records)
        original_questions = json.dumps(self.question_records)
        for error, mutate in mutations:
            with self.subTest(error=error):
                self.chunk_records = json.loads(original_chunks)
                self.question_records = json.loads(original_questions)
                mutate()
                self.save()
                with self.assertRaisesRegex(ValueError, error):
                    self.adapter().load_and_transform()

    def listing(self, root):
        return [{"uri": root + "/nested/" + chunk_filename(r["id"]), "isDir": False}
                for r in self.chunk_records]

    def test_actual_uris_pagination_missing_and_extra_files(self):
        root = "viking://resources/musique"
        entries = self.listing(root)
        client = Mock()
        client.ls_page.side_effect = [
            {"result": entries[:2], "has_more": True},
            {"result": entries[2:], "has_more": False},
        ]
        mapping = prepare.verify_source_map(client, self.adapter(), root)
        self.assertEqual(mapping["a" * 64], entries[0]["uri"])
        self.assertEqual(client.ls_page.call_args.kwargs["offset"], 2)
        for bad, error in (
            (entries[:-1], "Missing"),
            (entries + [{"uri": root + "/other.md", "isDir": False}], "Unexpected"),
            (entries + [{"uri": root + "/duplicate/" + chunk_filename("a" * 64),
                         "isDir": False}], "Duplicate"),
        ):
            client.ls_page.side_effect = None
            client.ls_page.return_value = {"result": bad}
            with self.assertRaisesRegex(ValueError, error):
                prepare.verify_source_map(client, self.adapter(), root)

    def test_import_no_split_full_directory_and_existing_target_guard(self):
        from openviking_sdk.errors import NotFoundError
        client = Mock()
        root = "viking://resources/musique"
        client.stat.side_effect = NotFoundError("absent")
        client.add_resource.return_value = {"root_uri": root}
        prepare.import_corpus(client, self.root, root, "vectors_only")
        self.assertEqual(client.add_resource.call_args.kwargs, {
            "to": root, "wait": True, "options": {
                "preserve_structure": True, "args": {"parse_mode": "no_split"},
                "processing_mode": "vectors_only",
            },
        })
        client.stat.side_effect = None
        client.add_resource.reset_mock()
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare.import_corpus(client, self.root, root, "vectors_only")
        client.add_resource.assert_not_called()

    def test_snapshot_uses_original_sources_for_matching_and_rejects_unknown(self):
        root = "viking://resources/musique"
        adapter = self.adapter()
        mapping = {record["id"]: entry["uri"]
                   for record, entry in zip(self.chunk_records, self.listing(root))}
        texts = {mapping[r["id"]]: "# " + r["title"] + "\n\n" + r["text"]
                 for r in self.chunk_records}

        def urlopen(request, timeout):
            from urllib.parse import parse_qs, urlparse
            uri = parse_qs(urlparse(request.full_url).query)["uri"][0]
            result = list({"uri": u, "isDir": False} for u in texts) if request.full_url.startswith(
                "http://127.0.0.1:1933/api/v1/fs/ls"
            ) else texts[uri]
            import io
            return io.BytesIO(json.dumps({"status": "ok", "result": result}).encode())

        with patch.object(batch.urllib.request, "urlopen", side_effect=urlopen):
            snapshot = batch.snapshot_resource({}, root, adapter.corpus_id, mapping)
            qa = adapter.load_and_transform()[0].qa_pairs[0]
            matcher = EvidenceMatcher(qa.gold_evidence, [ChunkDocument(**c) for c in snapshot])
            self.assertEqual(matcher.manifest.unresolved_evidence_ids, ())
            self.assertEqual({l.evidence_id for l in matcher.manifest.locations}, set(qa.metadata["supporting_chunk_ids"]))
            with self.assertRaisesRegex(ValueError, "Unmapped"):
                batch.snapshot_resource({}, root, adapter.corpus_id,
                                        {k: v for k, v in mapping.items() if k != "c" * 64})
            texts.pop(mapping["c" * 64])
            with self.assertRaisesRegex(ValueError, "Missing"):
                batch.snapshot_resource({}, root, adapter.corpus_id, mapping)

    def test_local_prepare_creates_full_pool_without_calls_or_source_map(self):
        output = self.root / "output"
        with patch.object(sys, "argv", [
            "prepare", "--questions", str(self.questions), "--chunks", str(self.chunks),
            "--output-dir", str(output), "--resource-uri", "viking://resources/musique",
        ]), patch("openviking_sdk.SyncHTTPClient") as client:
            self.assertEqual(prepare.main(), 0)
            client.assert_not_called()
        self.assertEqual(len(list((output / "chunks").iterdir())), 3)
        self.assertFalse((output / "source-map.json").exists())
        self.assertTrue((output / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
