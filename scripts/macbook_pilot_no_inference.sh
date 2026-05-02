#!/usr/bin/env bash
set -euo pipefail

uv run thesis-eval validate-config
uv run thesis-eval runtime-profiles
uv run thesis-eval prepare-uriel
if [ ! -f data/raw/strongreject/pilot_prompts.jsonl ]; then
  uv run thesis-eval import-strongreject --source github --pilot-size 2
fi
uv run thesis-eval run-pilot
uv run thesis-eval export-api-batches \
  --requests outputs/pilot/judge_requests.jsonl \
  --output-dir outputs/pilot/api_batches

echo "MacBook no-inference pilot files are ready under outputs/pilot/."
