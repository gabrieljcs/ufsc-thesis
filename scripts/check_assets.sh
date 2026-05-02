#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  set -- strongreject nllb_200 blaser_2_0_qe multijail belebele sabia_7b fauno gpt_sw3 bggpt
fi

for asset in "$@"; do
  echo "== ${asset} =="
  uv run thesis-eval asset-status --name "${asset}"
  uv run thesis-eval verify-assets --name "${asset}"
done
