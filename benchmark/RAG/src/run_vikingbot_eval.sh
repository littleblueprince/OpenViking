#!/usr/bin/env bash
set -euo pipefail

runner_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$runner_dir/../../.." && pwd)"
python="$repo_dir/.venv/bin/python"
evaluator="$repo_dir/.local-data/syllabusqa/test-agentic-rag-tools/evaluate_vikingbot_syllabusqa.py"

if [[ ! -x "$python" ]]; then
    printf 'Missing Python environment at %s\n' "$python" >&2
    exit 1
fi
if [[ ! -f "$evaluator" ]]; then
    printf 'Missing local SyllabusQA evaluator at %s\n' "$evaluator" >&2
    exit 1
fi

exec "$python" "$evaluator" --output-root "$runner_dir/vikingbot-eval-output" "$@"
# ./benchmark/RAG/src/run_vikingbot_eval.sh --case G4zZKP3yeU0rVsqy
# ./benchmark/RAG/src/run_vikingbot_eval.sh --all --workers 4
# ./benchmark/RAG/src/run_vikingbot_eval.sh --all --workers 4
