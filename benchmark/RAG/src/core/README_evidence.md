# Evidence matching and tool traces

For a shell entry point supporting one QA ID or a whole dataset with independent
worker processes, see [the batch launcher guide](../../scripts/README_vikingbot.md).

The matcher and recorder use the standard library. VikingBot native list, glob,
search, grep and multi-read tools now capture source observations, and tool events
carry call IDs, response IDs, iteration, dispatch sequence and effective arguments.
The runner translates these events into benchmark-only annotations.
`StandardQA.gold_evidence` is populated by the four dataset adapters; the existing
`evidence` field and legacy Recall remain available.

## VikingBot adapter entry point

```python
from adapters.syllabusqa_adapter import SyllabusQAAdapter
from core.trace_models import ChunkDocument
from vikingbot_runner import VikingBotRunner

sample = SyllabusQAAdapter("syllabusqa.json").load_and_transform()[0]
# Collect the full ingested L2 chunk snapshot before the agent turn. Map each
# chunk to the adapter's source document ID, not a guessed filename or URI.
chunks = [
    ChunkDocument(
        uri="viking://resources/course/chunk_1.md",
        source_doc_id=sample.sample_id,
        text=full_chunk_text,
    ),
]
runner = VikingBotRunner(config)
result = runner.generate_qa(
    sample.qa_pairs[0], chunks,
    case_id=f"{sample.sample_id}:qa_0",
    run_id="run_1",
    trace_dir="Output/run_1/evidence",
    resource_uri="viking://resources/course",
    disabled_tools=["web_search", "web_fetch"],
)
print(result["evidence_summary"])
```

`generate_qa` writes a manifest and a per-step JSONL under unique names and logs
`[RAG Evidence]` for each completed tool. It returns `evidence_manifest_path`,
`evidence_trace_path`, `evidence_summary`, and any pending call IDs. On turn errors,
the partial trace/summary remain available alongside `error`. The manifest and
gold evidence stay in benchmark objects; the agent receives only the question,
resource URI and tool restrictions. Load an existing manifest and use
`generate_answer(..., evidence_matcher=matcher, evidence_trace_path=path,
case_id=..., run_id=...)` to reuse a snapshot.

### Expanded readable JSON

The runner also returns `evidence_readable_path`, pointing to `*.steps.readable.json`.
This file contains `total_steps` (tool calls, not model iterations),
`model_iterations`, `hit_step_count`, and an expanded `steps` array. Each step
lists its tool/arguments, result preview, and `chunks`; every chunk has its URI,
`content_preview` next to `hit`, and the matching evidence IDs.

`hit` means **full returned source text** matches gold evidence, not that the
display preview happens to contain it. `located_hit` separately indicates a known
gold chunk URI; `delivered_hit` indicates matching text inserted into context.
`null` identifies unknown/failed sources or uncaptured delivery. Directories and
candidate-only listings have no returned source text and cannot be text hits.
`preview_truncated` reports display shortening; `tool_output_truncated` reports
actual tool preview truncation. Disjoint source fragments have display separators.

Set `vikingbot.evidence_preview_chars` (default 500), or use the smoke launcher's
`--preview-chars K`. Console logs now print one step header and one line per
chunk, with `[HIT]`, `[LOCATED_ONLY]`, `[ANCESTOR_ONLY]`, `[MISS]` or an error status
beside its URI and text preview.

To convert a saved result offline without rerunning the model:

```bash
.venv/bin/python benchmark/RAG/scripts/export_evidence_trace.py \
  --result /path/to/result.json --preview-chars 500
```

This writes expanded JSON plus a separate readable `.log`, preserving the original
result and full JSONL. Files are created exclusively; use `--output` for a new path.

This entry point requires an explicit full L2 snapshot with document mappings.
It does not derive the evaluation snapshot from retrieved text, and conventional
`run.py`/`BenchmarkPipeline` still use their existing retrieve-then-generate path.
Ordinary multi-read and search/grep captures lack full-file revision hashes and
are marked unverified; use a fixed store snapshot. Multi-read `with_revision=True`
captures the full-file SHA256 even when reading only a line range.

Adapter identities are case-local: LoCoMo uses dialogue tokens scoped to
`sample_id`; SyllabusQA retains `answer_span_i` and excludes `reasoning_step_i`
from source recall; Qasper keeps each annotator's `answer_i` evidence group;
FinanceBench scopes spans to `doc_name` and retains raw evidence/page annotations
in metadata. Duplicate text across Qasper annotators retains separate identities.

Delivery capture projects source offsets onto the exact normal tool preview,
keeping head and tail separate. JSON strings are decoded only at complete escape
boundaries. If post-call hooks replace the result, captures are invalidated.
Unknown delivery transformations remain uncaptured instead of assuming visibility.
Here, delivered means the tool text inserted into the agent message context after
preview/media policy. Later history compaction and provider transport are not
tracked by this capture; it does not prove the text reached a subsequent model call.
Exported sandbox files and subsequent generic `read_file` calls currently do not
carry OpenViking source provenance; use direct OpenViking reads for evidence
measurement. Media attachments themselves are not textual evidence.

## Minimal usage

```python
import logging

from benchmark.RAG.src.core.evidence_matcher import EvidenceMatcher, content_hash
from benchmark.RAG.src.core.trace_annotation import EvidenceTraceRecorder
from benchmark.RAG.src.core.trace_models import (
    ChunkDocument, SourceObservation, StandardEvidence, TextFragment, ToolStep,
)

uri = "viking://resources/course/chunk_1.md"
body = "Lectures take place on Tuesday."
matcher = EvidenceMatcher(
    [StandardEvidence("ev_1", "course", body, annotation_id="answer_span_1")],
    [ChunkDocument(uri, "course", body)],
    snapshot_id="run_1",
)
with EvidenceTraceRecorder("Output/run_1/qa_1.steps.jsonl", matcher) as recorder:
    record = recorder.record(ToolStep(
        case_id="qa_1",
        run_id="run_1",
        session_id="session_1",
        response_id="response_1",
        tool_call_id="call_1",
        tool_name="openviking_multi_read",
        iteration=1,
        sequence=1,
        sources=(SourceObservation(
            uri=uri,
            content_hash=content_hash(body),
            returned_fragments=(TextFragment(body),),
            delivered_fragments=(TextFragment(body),),
        ),),
    ))
    logging.getLogger("Benchmark").info(recorder.format_log(record))
    summary = recorder.summary().to_dict()
```

The manifest can be serialized with `matcher.manifest.to_dict()`. Reconstruct it
with `EvidenceManifest.from_dict(...)` and `EvidenceMatcher.from_manifest(...)`
to annotate traces without rereading chunks. Raw step dictionaries round-trip
through `ToolStep.from_dict(...)`; annotated records use
`StepEvidenceRecord.from_dict(...)`. `summarize_trace(...)` accepts restored
records, orders them by sequence and rejects mixed responses/duplicate calls.

## Matching semantics

- `located_evidence_ids`: the returned URI is a known L2 chunk containing evidence.
- `ancestor_evidence_ids`: a listed directory contains such a chunk. Not a hit.
- `returned_evidence_ids`: source text in this tool result contains evidence.
- `delivered_evidence_ids`: source text actually delivered to the model contains it.
- `matches`: stage, fragment index, normalized start/end offsets and matching method.

NFC Unicode and whitespace normalization preserve case, numbers, units,
punctuation and negation. Dialogue IDs use token boundaries. Evidence must belong
to the chunk's source document. There is no word-coverage or LLM approximation.
Offsets refer to normalized fragment/chunk text, not original character offsets.

Provide **full L2 snapshot chunks** to build the manifest. Provide **only source
text** in observations, excluding tool arguments, headers, notices and errors.
Keep disjoint grep hits and preview head/tail pieces in separate fragments; they
are never concatenated. Contiguous multiline text belongs in one fragment.
Line ranges are zero-based/end-exclusive and are descriptive, not instructions
for the matcher to slice the supplied text.

`content_hash` on an observation is the full chunk's snapshot hash, not a hash
of the partial returned text. A different hash produces `stale_source` with no
hits. A missing hash is `unverified`; callers must ensure a fixed snapshot.
Unknown URIs and conflicting source document IDs have explicit statuses.

`delivered_fragments=None` means delivery was not captured; `()` means no source
text was delivered. Missing captures are counted in summaries and are never
assumed equal to the returned text. Candidate listings cannot contain fragments.
Whole-call failures suppress hits; partial failures use per-source success flags.

Recall counts each evidence ID once, even across repeated reads and multiple
matching chunks. Unresolved source evidence stays in the denominator.
`reasoning_annotation` is excluded and reported separately. No applicable
evidence produces `None` recall rather than a misleading zero. Qasper-style
answer groups each get a separate delivered Recall; they are not collapsed into
a requirement to retrieve all annotators' evidence.

## Recording contract and tests

One recorder handles one case/run/session/response and creates a new file.
Record parallel tool results in **dispatch sequence** order, buffering completed
results upstream. Multiple calls in the same iteration need distinct IDs and
sequences. Each retry uses its own response/trace. JSONL writes are locked and
flushed; use the context manager or `close()` to release the file.

Run the suite from the repository root; no server or model calls are made.
Integration tests require the project's installed VikingBot dependencies, while
the matcher-only tests need no extra packages:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s benchmark/RAG/tests -v
```
