"""MuSiQue questions over a shared, unfiltered pool of source chunks."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from .base import BaseAdapter, StandardDoc, StandardEvidence, StandardQA, StandardSample


def read_jsonl(path):
    records, ids = [], set()
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"].strip():
                raise ValueError(f"{path}:{line_number}: expected an object with a nonempty string id")
            if item["id"] in ids:
                raise ValueError(f"{path}:{line_number}: duplicate id {item['id']}")
            ids.add(item["id"])
            records.append(item)
    if not records:
        raise ValueError(f"Empty dataset: {path}")
    return records


def chunk_filename(chunk_id):
    # MarkdownParser limits sanitized filename stems to 50 characters.
    return "c_" + hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()[:32] + ".md"


class MuSiQueAdapter(BaseAdapter):
    def __init__(self, raw_file_path, chunks_file=None):
        super().__init__(raw_file_path)
        questions = Path(raw_file_path)
        if chunks_file is None:
            prefix = questions.stem.split("_answerable_subset_", 1)[0]
            chunks_file = questions.with_name(prefix + "_chunks.jsonl")
        self.chunks_file = Path(chunks_file)
        self.corpus_id = self.chunks_file.stem.removesuffix("_chunks")
        self._chunks = None
        self._questions = None

    def _load(self):
        if self._chunks is not None:
            return
        chunks = read_jsonl(self.chunks_file)
        for chunk in chunks:
            if not isinstance(chunk.get("title"), str) or not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
                raise ValueError(f"Chunk {chunk['id']}: expected title and nonempty text")
        by_id = {chunk["id"]: chunk for chunk in chunks}
        if self.corpus_id in by_id:
            raise ValueError("Corpus id conflicts with a chunk id")
        names = [chunk_filename(chunk["id"]) for chunk in chunks]
        if len(set(names)) != len(names):
            raise ValueError("Prepared chunk filenames collide")
        questions = read_jsonl(self.raw_file_path)
        for question in questions:
            qid = question["id"]
            if not isinstance(question.get("question"), str) or not question["question"].strip():
                raise ValueError(f"Question {qid}: expected nonempty question")
            if not isinstance(question.get("answerable"), bool):
                raise ValueError(f"Question {qid}: expected boolean answerable")
            if not isinstance(question.get("answer"), str) or (
                question["answerable"] and not question["answer"].strip()
            ):
                raise ValueError(f"Question {qid}: expected answer")
            for key in ("answer_aliases", "supporting_chunk_ids"):
                values = question.get(key)
                if not isinstance(values, list) or not all(isinstance(v, str) and v.strip() for v in values):
                    raise ValueError(f"Question {qid}: expected string list {key}")
            supports = question["supporting_chunk_ids"]
            if len(set(supports)) != len(supports):
                raise ValueError(f"Question {qid}: duplicate supporting_chunk_ids")
            missing = [sid for sid in supports if sid not in by_id]
            if missing:
                raise ValueError(f"Question {qid}: missing supporting chunks {missing}")
            if question["answerable"] and not supports:
                raise ValueError(f"Question {qid}: answerable question has no supporting chunks")
        self._chunks, self._questions = by_id, questions

    @property
    def chunks(self):
        self._load()
        return self._chunks

    def stats(self):
        self._load()
        return {
            "corpus_id": self.corpus_id, "chunks": len(self._chunks),
            "questions": len(self._questions),
            "answerable": sum(q["answerable"] for q in self._questions),
            "support_counts": dict(Counter(len(q["supporting_chunk_ids"]) for q in self._questions)),
            "unique_supporting_chunks": len({
                sid for q in self._questions for sid in q["supporting_chunk_ids"]
            }),
        }

    def data_prepare(self, doc_dir):
        self._load()
        directory = Path(doc_dir)
        directory.mkdir(parents=True, exist_ok=True)
        docs = []
        for chunk in self._chunks.values():
            path = directory / chunk_filename(chunk["id"])
            path.write_text(f"# {chunk['title']}\n\n{chunk['text']}\n", encoding="utf-8")
            docs.append(StandardDoc(sample_id=chunk["id"], doc_path=str(path)))
        return docs

    def load_and_transform(self):
        self._load()
        qas = []
        for question in self._questions:
            supports = question["supporting_chunk_ids"]
            gold = list(dict.fromkeys(
                answer.strip() for answer in [question["answer"], *question["answer_aliases"]]
                if answer.strip()
            )) if question["answerable"] else ["Insufficient information"]
            qas.append(StandardQA(
                question=question["question"], gold_answers=gold,
                evidence=[self._chunks[sid]["text"] for sid in supports],
                gold_evidence=[
                    StandardEvidence(sid, sid, self._chunks[sid]["text"], annotation_id=sid)
                    for sid in supports
                ],
                category=f"{len(supports)}hop",
                metadata={
                    "id": question["id"], "answerable": question["answerable"],
                    "supporting_chunk_ids": list(supports), "hops": len(supports),
                },
            ))
        return [StandardSample(self.corpus_id, qas, metadata={"shared_corpus": True})]

    def build_prompt(self, qa, context_blocks):
        return (
            "Answer the question using the source context. Combine facts across sources "
            "when necessary. If the context is insufficient, answer 'Insufficient information'."
            "\n\nContext:\n" + "\n\n".join(context_blocks) + "\n\nQuestion: " + qa.question,
            {},
        )
