#!/usr/bin/env bash
set -euo pipefail

uv run thesis-eval model-status --model sagui_7b
uv run thesis-eval generate-targets \
  --translations outputs/pilot/translations_por.blaser.jsonl \
  --model sagui_7b \
  --profile macbook \
  --backend transformers \
  --model-path assets/models/sagui-7b-instruct-v0.1 \
  --max-tokens 64 \
  --stream \
  --output outputs/pilot/target_generations_por.macbook_smoke.jsonl
uv run thesis-eval export-judge-batch \
  --generations outputs/pilot/target_generations_por.macbook_smoke.jsonl \
  --output outputs/pilot/judge_requests_por.macbook_smoke.jsonl
uv run thesis-eval export-api-batches \
  --requests outputs/pilot/judge_requests_por.macbook_smoke.jsonl \
  --output-dir outputs/pilot/api_batches_por_macbook_smoke
