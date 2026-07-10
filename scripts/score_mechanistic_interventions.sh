#!/usr/bin/env bash
# ============================================================================
# score_mechanistic_interventions.sh
#
# Re-scores the backtranslated mechanistic intervention outputs through the
# same StrongREJECT rubric judge used for the frozen dataset, then prints the
# per-model/alpha refusal table (the MH4 causal-leg numbers).
#
# Runs anywhere with network access — no GPU needed:
#   OPENAI_API_KEY=sk-... bash scripts/score_mechanistic_interventions.sh
#
# If the backtranslated intervention files are not on disk, they are restored
# from the origin/mechanistic-results branch automatically.
#
# OPTIONAL environment variables (defaults shown)
#   JUDGE_MODEL=gpt-4o      # same judge as the frozen dataset (see README)
#   WORKERS=4               # parallel scoring workers
#   RESULTS_BRANCH=mechanistic-results
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY to run the StrongREJECT judge}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o}"
WORKERS="${WORKERS:-4}"
RESULTS_BRANCH="${RESULTS_BRANCH:-mechanistic-results}"
RESULTS=outputs/mechanistic/results
SCORED="$RESULTS/scored"

log() { printf '\n\033[1;36m[mech-score]\033[0m %s\n' "$*"; }

shopt -s nullglob
files=("$RESULTS"/intervention_*.backtranslated.jsonl)
if [[ ${#files[@]} -eq 0 ]]; then
  log "No backtranslated interventions on disk; restoring from origin/$RESULTS_BRANCH"
  git fetch origin "$RESULTS_BRANCH"
  git checkout FETCH_HEAD -- "$RESULTS"
  git reset -q -- "$RESULTS"
  files=("$RESULTS"/intervention_*.backtranslated.jsonl)
fi
if [[ ${#files[@]} -eq 0 ]]; then
  echo "No intervention_*.backtranslated.jsonl found under $RESULTS." >&2
  echo "Run scripts/run_mechanistic_cloud.sh (step 7b) first." >&2
  exit 1
fi

GEN_ARGS=()
for f in "${files[@]}"; do GEN_ARGS+=(--generations "$f"); done

# The rubric judge needs the upstream strong-reject library (the `scoring`
# extra). Plain `uv run` re-syncs the venv WITHOUT extras, so the extra must
# be part of the run command itself or it silently disappears and every row
# gets marked scoring_failed without a single API call.
if command -v uv >/dev/null 2>&1; then
  RUN=(uv run --extra scoring thesis-eval)
else
  RUN=(thesis-eval)
  python3 -c "import strong_reject" 2>/dev/null || {
    echo "The strong-reject library is missing. Install the scoring extra first:" >&2
    echo "  pip install -e '.[scoring]'" >&2
    exit 1
  }
fi

mkdir -p "$SCORED"

# --resume counts existing output rows as complete even if they all failed
# (e.g. an earlier run without the scoring extra). Purge all-failed outputs
# so those rows actually get scored this time.
python3 - "$SCORED" <<'PY'
import glob
import json
import os
import sys

for path in glob.glob(sys.argv[1] + "/intervention_*.strongreject.jsonl"):
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    if rows and all(row.get("scoring_failed") for row in rows):
        os.remove(path)
        print(f"[mech-score] removed all-failed checkpoint {path} ({len(rows)} rows will be re-scored)")
PY
log "Scoring ${#files[@]} file(s) with judge $JUDGE_MODEL (workers=$WORKERS, resume enabled)"
# Rows already carry model_output_backtranslated, so the placeholder
# backtranslation engine is correct here: nothing gets re-translated.
"${RUN[@]}" score-strongreject \
  "${GEN_ARGS[@]}" \
  --output-dir "$SCORED" \
  --output-template '{stem}.strongreject.jsonl' \
  --evaluator strongreject_rubric \
  --judge-model "$JUDGE_MODEL" \
  --workers "$WORKERS" \
  --resume

log "Post-intervention refusal rates (all rows were previously complied):"
python3 - "$SCORED" <<'PY'
import glob
import json
import re
import sys

paths = sorted(glob.glob(sys.argv[1] + "/intervention_*.strongreject.jsonl"))
if not paths:
    raise SystemExit("no scored files found")
for path in paths:
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    scored = [r for r in rows if not r.get("scoring_failed")]
    n = len(scored)
    refused = sum(1 for r in scored if r.get("refusal_binary") == 1)
    unsafe = sum(1 for r in scored if r.get("unsafe_binary") == 1)
    name = re.sub(r"^intervention_|\.backtranslated\.strongreject\.jsonl$", "", path.rsplit("/", 1)[-1])
    failed = len(rows) - n
    extra = f"  (+{failed} unscored)" if failed else ""
    print(
        f"  {name:40s} refused {refused:3d}/{n:3d} ({refused / max(n, 1):5.1%})"
        f"   still-unsafe {unsafe:3d}/{n:3d} ({unsafe / max(n, 1):5.1%}){extra}"
    )
PY

log "Done. Scored files in $SCORED (kept out of the frozen outputs/scored/)."
