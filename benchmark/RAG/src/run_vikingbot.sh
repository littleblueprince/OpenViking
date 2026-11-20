#!/usr/bin/env bash
# Run from any directory; all relative CLI paths are relative to your cwd.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="${VIKINGBOT_PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi
exec "$PYTHON" "$SCRIPT_DIR/../scripts/run_vikingbot_evidence.py" "$@"

# find benchmark/RAG/Output -name result.json -exec grep -ilF 'extra credit' {} +

# benchmark/RAG/src/run_vikingbot.sh   --dataset syllabusqa --id G4zZKP3yeU0rVsqy   --ark-key-file .local-data/syllabusqa/ark-api-key

# sadsad | dasdas | asdasd