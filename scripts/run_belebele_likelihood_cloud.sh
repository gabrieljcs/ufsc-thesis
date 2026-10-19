#!/usr/bin/env bash
# ============================================================================
# run_belebele_likelihood_cloud.sh
#
# Rescores BELEBELE by per-option log-probability instead of generating an
# answer and parsing it (AGENTS.md, 2026-10-15 addendum). Designed to be copied
# onto a fresh RunPod / cloud GPU pod and run once:
#
#   1. clones the repo and installs the project
#   2. downloads the BELEBELE dataset
#   3. runs a 2-item smoke test and verifies the output shape
#   4. scores all 13 languages for each open-weight model, resumably
#   5. recomputes IF/CONS/SPEC, both likelihood-only and in the mixed
#      117-cell universe that keeps Sabia-3's generation-parsed rows
#   6. measures agreement with the frozen generation-parsed predictions
#   7. (optionally) pushes the small result artifacts to a results branch
#
# Quick start on a fresh pod:
#
#   export HF_TOKEN=hf_...            # Llama-3.1-8B-Instruct is gated
#   export GITHUB_TOKEN=ghp_...       # only needed when PUBLISH=1
#   bash run_belebele_likelihood_cloud.sh
#
# Resume after a pod restart: run the exact same command again. Finished
# languages are skipped and a partially finished language resumes from its
# checkpoint.
#
# --------------------------------------------------------------------------
# REQUIRED environment variables
#   HF_TOKEN        Hugging Face token with access to gated Llama-3.1 weights.
#
# OPTIONAL environment variables (defaults shown)
#   GITHUB_TOKEN                   Used by the publish step. The gh CLI works too.
#   PUBLISH=1                      Set 0 to keep results on the pod only.
#   REPO_URL=https://github.com/gabrieljcs/ufsc-thesis
#   CODE_BRANCH=main
#   WORKDIR=./ufsc-thesis-belebele
#   RESULTS_BRANCH=belebele-likelihood-results
#   MODELS="<the eight open-weight targets>"
#   LANGS=""                       Empty means all thesis attack languages.
#   NORMALIZATION=mean             Reported choice rule; both are always saved.
#   CONTINUATION_MODE=option_text  option_text or letter.
#   GPU_BATCH_SIZE=16              Option continuations per forward pass.
#   LOCAL_BATCH_SIZE=64            Items per checkpoint flush.
#   DEVICE=auto / DTYPE=auto
#   SKIP_INSTALL=0
#   SKIP_SMOKE=0
#   FORCE=0                        Set 1 to rescore languages already finished.
#   GIT_NAME / GIT_EMAIL           Committer identity for the results push.
#
# Sabia-3 is deliberately absent from MODELS: it is API-served and exposes no
# per-option log-probabilities, so the CLI refuses it. Its frozen
# generation-parsed rows are reused for the 117-cell standardization instead.
# ============================================================================
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/gabrieljcs/ufsc-thesis}"
CODE_BRANCH="${CODE_BRANCH:-main}"
WORKDIR="${WORKDIR:-./ufsc-thesis-belebele}"
RESULTS_BRANCH="${RESULTS_BRANCH:-belebele-likelihood-results}"
MODELS="${MODELS:-sagui_7b llamantino_2_ultrachat_7b llamantino_anita_8b gpt_sw3 ai_sweden_llama3_8b bggpt_7b bggpt_gemma_9b llama3_1_8b_reference}"
LANGS="${LANGS:-}"
NORMALIZATION="${NORMALIZATION:-mean}"
CONTINUATION_MODE="${CONTINUATION_MODE:-option_text}"
GPU_BATCH_SIZE="${GPU_BATCH_SIZE:-16}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-64}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
FORCE="${FORCE:-0}"
PUBLISH="${PUBLISH:-1}"
GIT_NAME="${GIT_NAME:-Gabriel de Jesus}"
GIT_EMAIL="${GIT_EMAIL:-gabrieldejesus2211@gmail.com}"

log()  { printf '\n\033[1;36m[belebele]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[belebele:warn]\033[0m %s\n' "$*"; }
die()  { printf '\n\033[1;31m[belebele:error]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 0. Preflight. Fail now rather than after hours of GPU time.
# ---------------------------------------------------------------------------
: "${HF_TOKEN:?Set HF_TOKEN (needs access to meta-llama/Llama-3.1-8B-Instruct)}"
export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

case "$NORMALIZATION" in
  mean|sum) ;;
  *) die "NORMALIZATION must be 'mean' or 'sum'; got '$NORMALIZATION'." ;;
esac
case "$CONTINUATION_MODE" in
  option_text|letter) ;;
  *) die "CONTINUATION_MODE must be 'option_text' or 'letter'; got '$CONTINUATION_MODE'." ;;
esac
for model in $MODELS; do
  [[ "$model" == "sabia_3" ]] && die \
    "sabia_3 is API-served and cannot be likelihood-scored. Remove it from MODELS."
done

if [[ "$PUBLISH" == "1" ]]; then
  if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    log "Using GITHUB_TOKEN for the results push"
  elif command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    log "GITHUB_TOKEN not set; using the authenticated gh CLI for the results push"
  else
    die "No GitHub credential. Set GITHUB_TOKEN, run 'gh auth login', or set PUBLISH=0."
  fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read -r line; do
    log "GPU: $line"
  done
else
  warn "nvidia-smi not found; this will run on CPU and take far too long."
fi

# ---------------------------------------------------------------------------
# 1. Clone (or update) the repo
# ---------------------------------------------------------------------------
if [[ -d "$WORKDIR/.git" ]]; then
  log "Repo exists at $WORKDIR; fetching $CODE_BRANCH"
  git -C "$WORKDIR" fetch origin "$CODE_BRANCH"
  git -C "$WORKDIR" checkout "$CODE_BRANCH"
  git -C "$WORKDIR" pull --ff-only origin "$CODE_BRANCH"
else
  log "Cloning $REPO_URL (branch $CODE_BRANCH) into $WORKDIR"
  git clone --branch "$CODE_BRANCH" "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"

[[ -f src/thesis_eval/models/likelihood.py ]] || die \
  "src/thesis_eval/models/likelihood.py is missing from the clone. Push the
   option-likelihood code to $CODE_BRANCH on $REPO_URL first."

# ---------------------------------------------------------------------------
# 2. Install the project
# ---------------------------------------------------------------------------
if [[ "$SKIP_INSTALL" != "1" ]] && ! python -c "import thesis_eval.models.likelihood" 2>/dev/null; then
  if command -v uv >/dev/null 2>&1; then
    log "Installing with uv (rtx extra)"
    uv sync --extra rtx
    RUN=(uv run thesis-eval)
  else
    log "Installing with pip (rtx extra)"
    pip install -e '.[rtx]'
    RUN=(thesis-eval)
  fi
else
  RUN=(thesis-eval)
  command -v thesis-eval >/dev/null 2>&1 || RUN=(python -m thesis_eval.cli)
fi
log "CLI: ${RUN[*]}"

# Verify the clone actually carries the code this run needs. The option-likelihood
# path depends on the canonical-identity helpers in benchmarks/belebele.py, which
# live in the same change set. A clone of a branch that has the new files but not
# those helpers imports cleanly as a file and then fails at the first command.
python - <<'PY' || die "The clone is missing code this run needs. Commit and push the
   canonical-identity helpers in src/thesis_eval/benchmarks/belebele.py together
   with the option-likelihood modules, then rerun."
import sys

try:
    from thesis_eval.benchmarks.belebele import (
        canonical_question_id,
        prediction_question_id,
    )
    from thesis_eval.benchmarks.belebele_likelihood import (
        attach_option_likelihood_predictions,
        build_option_continuations,
        compare_scoring_paths,
        select_choice,
    )
    from thesis_eval.models.likelihood import (
        build_option_likelihood_runner,
        continuation_logit_positions,
        score_continuation,
    )
except Exception as exc:  # noqa: BLE001 - report whatever is absent
    sys.exit(f"import check failed: {type(exc).__name__}: {exc}")
print("import check OK")
PY

for required_cmd in predict-belebele-likelihood compare-belebele-scoring repair-belebele-predictions; do
  "${RUN[@]}" "$required_cmd" --help >/dev/null 2>&1 || die \
    "The installed CLI has no '$required_cmd' subcommand. The clone predates the
     option-likelihood change set; commit and push it, then rerun."
done
log "Code check passed"

OUT=outputs/belebele_likelihood
RESULTS="$OUT/results"
PRED="$OUT/predictions"
mkdir -p "$RESULTS" "$PRED"

# Restore anything a previous pod already published, so a fresh pod resumes
# instead of recomputing finished languages.
if [[ "$PUBLISH" == "1" ]] && git fetch origin "$RESULTS_BRANCH" 2>/dev/null; then
  log "Restoring published results from origin/$RESULTS_BRANCH"
  git checkout FETCH_HEAD -- "$OUT" 2>/dev/null || true
  git reset -q -- "$OUT" 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 3. BELEBELE dataset
# ---------------------------------------------------------------------------
log "Ensuring the BELEBELE dataset is present"
"${RUN[@]}" download-assets --name belebele

LANG_ARGS=()
for lang in $LANGS; do LANG_ARGS+=(--language "$lang"); done
if [[ ${#LANG_ARGS[@]} -eq 0 ]]; then
  log "Languages: all thesis attack languages"
else
  log "Languages: $LANGS"
fi

# ---------------------------------------------------------------------------
# 4. Smoke test. Two items on one model, with an output-shape assertion.
#    The scoring core is unit-tested, but the torch path is not exercised
#    without a GPU, so prove it here in ~2 minutes instead of discovering a
#    bug six hours in.
# ---------------------------------------------------------------------------
if [[ "$SKIP_SMOKE" != "1" ]]; then
  SMOKE_MODEL="$(printf '%s\n' $MODELS | head -1)"
  SMOKE_OUT="$OUT/smoke_${SMOKE_MODEL}.jsonl"
  log "Smoke test: $SMOKE_MODEL, 2 items, English"
  rm -f "$SMOKE_OUT" "${SMOKE_OUT%.jsonl}.partial.jsonl"
  "${RUN[@]}" predict-belebele-likelihood \
    --model "$SMOKE_MODEL" --language eng --limit 2 \
    --normalization "$NORMALIZATION" \
    --continuation-mode "$CONTINUATION_MODE" \
    --gpu-batch-size "$GPU_BATCH_SIZE" \
    --device "$DEVICE" --dtype "$DTYPE" \
    --output "$SMOKE_OUT"

  python - "$SMOKE_OUT" <<'PY' || die "Smoke test produced unusable output; stopping before the full run."
import json, sys

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
if len(rows) != 2:
    sys.exit(f"expected 2 rows, got {len(rows)}")
for row in rows:
    if row.get("scoring_method") != "option_likelihood":
        sys.exit("scoring_method is not option_likelihood")
    if row.get("predicted_choice") not in {"A", "B", "C", "D"}:
        sys.exit(f"predicted_choice is {row.get('predicted_choice')!r}")
    sums = row.get("option_logprob_sum") or {}
    counts = row.get("option_token_counts") or {}
    if sorted(sums) != ["A", "B", "C", "D"]:
        sys.exit(f"option_logprob_sum keys are {sorted(sums)}")
    if any(not isinstance(v, (int, float)) or v > 0 for v in sums.values()):
        sys.exit(f"log-probabilities must be negative; got {sums}")
    if any(int(c) < 1 for c in counts.values()):
        sys.exit(f"continuation token counts must be >= 1; got {counts}")
    if len(set(sums.values())) == 1:
        sys.exit("all four options scored identically; the prompt or gather is wrong")
print("smoke test OK:", json.dumps(rows[0]["option_logprob_mean"], sort_keys=True))
PY
  log "Smoke test passed"
fi

# ---------------------------------------------------------------------------
# 5. Score every model. One CLI process per model so GPU memory is released
#    between models; per-language output files make reruns cheap.
# ---------------------------------------------------------------------------
FORCE_ARGS=()
[[ "$FORCE" == "1" ]] && FORCE_ARGS+=(--force)

for model in $MODELS; do
  log "Scoring option likelihoods: $model"
  "${RUN[@]}" predict-belebele-likelihood \
    --model "$model" \
    "${LANG_ARGS[@]+"${LANG_ARGS[@]}"}" \
    --output-dir "$PRED" \
    --normalization "$NORMALIZATION" \
    --continuation-mode "$CONTINUATION_MODE" \
    --gpu-batch-size "$GPU_BATCH_SIZE" \
    --local-batch-size "$LOCAL_BATCH_SIZE" \
    --device "$DEVICE" --dtype "$DTYPE" \
    "${FORCE_ARGS[@]+"${FORCE_ARGS[@]}"}"
done

# ---------------------------------------------------------------------------
# 6. Assemble the rescored predictions
# ---------------------------------------------------------------------------
LIKELIHOOD_ALL="$OUT/likelihood_predictions.jsonl"
log "Assembling rescored predictions into $LIKELIHOOD_ALL"
: > "$LIKELIHOOD_ALL"
shopt -s nullglob
for f in "$PRED"/*.belebele-likelihood.jsonl; do
  cat "$f" >> "$LIKELIHOOD_ALL"
done
shopt -u nullglob
[[ -s "$LIKELIHOOD_ALL" ]] || die "No rescored predictions were produced."
log "Rescored rows: $(wc -l < "$LIKELIHOOD_ALL")"

# ---------------------------------------------------------------------------
# 7. Put the frozen generation-parsed predictions on canonical identities.
#    The frozen files carry positional item_ids, so they must be migrated
#    before they can be joined against anything.
# ---------------------------------------------------------------------------
FROZEN_DIR=outputs/spec
CANON="$OUT/canonical_frozen"
mkdir -p "$CANON"
GENERATED_ALL="$OUT/generated_predictions.jsonl"
: > "$GENERATED_ALL"
for model in $MODELS; do
  src="$FROZEN_DIR/${model}.belebele.jsonl"
  if [[ ! -f "$src" ]]; then
    warn "No frozen predictions for $model; it will be absent from the agreement table."
    continue
  fi
  dst="$CANON/${model}.canonical.jsonl"
  if [[ ! -s "$dst" ]]; then
    log "Migrating frozen predictions to canonical identities: $model"
    # Write via a temp file and move: a pod killed mid-migration must not
    # leave a truncated file that the -s check would accept as finished.
    "${RUN[@]}" repair-belebele-predictions --predictions "$src" --output "$dst.tmp"
    mv "$dst.tmp" "$dst"
  fi
  cat "$dst" >> "$GENERATED_ALL"
done

# Sabia-3 keeps its generation-parsed rows; it cannot be rescored.
SABIA_SRC="$FROZEN_DIR/sabia_3.belebele.jsonl"
SABIA_CANON="$CANON/sabia_3.canonical.jsonl"
COMBINED="$OUT/combined_predictions.jsonl"
cp "$LIKELIHOOD_ALL" "$COMBINED"
if [[ -f "$SABIA_SRC" ]]; then
  if [[ ! -s "$SABIA_CANON" ]]; then
    log "Migrating Sabia-3 frozen predictions (kept generation-parsed)"
    "${RUN[@]}" repair-belebele-predictions --predictions "$SABIA_SRC" --output "$SABIA_CANON.tmp"
    mv "$SABIA_CANON.tmp" "$SABIA_CANON"
  fi
  cat "$SABIA_CANON" >> "$COMBINED"
  log "Combined universe includes Sabia-3's generation-parsed rows (mixed instrument)"
else
  warn "Sabia-3 frozen predictions missing; the combined universe will have 104 cells, not 117."
fi

# ---------------------------------------------------------------------------
# 8. Recompute IF/CONS/SPEC
#    Two universes, because SPEC is standardized across whatever cells it sees:
#      - likelihood_only: the 8 rescored models (104 cells)
#      - combined:        plus Sabia-3 generation-parsed (117 cells, matches
#                         the frozen standardization universe)
# ---------------------------------------------------------------------------
log "Recomputing IF/CONS/SPEC (likelihood-only universe)"
"${RUN[@]}" compute-spec \
  --predictions "$LIKELIHOOD_ALL" \
  --output "$RESULTS/spec_likelihood_only.jsonl"

log "Recomputing IF/CONS/SPEC (combined 117-cell universe)"
"${RUN[@]}" compute-spec \
  --predictions "$COMBINED" \
  --output "$RESULTS/spec_combined.jsonl"

# ---------------------------------------------------------------------------
# 9. Agreement between the two scoring paths
# ---------------------------------------------------------------------------
if [[ -s "$GENERATED_ALL" ]]; then
  log "Measuring agreement with the frozen generation-parsed predictions"
  "${RUN[@]}" compare-belebele-scoring \
    --generated "$GENERATED_ALL" \
    --likelihood "$LIKELIHOOD_ALL" \
    --output "$RESULTS/scoring_agreement.jsonl"
else
  warn "No migrated frozen predictions; skipping the agreement table."
fi

# ---------------------------------------------------------------------------
# 10. Console summary
# ---------------------------------------------------------------------------
log "Summary"
python - "$RESULTS/spec_combined.jsonl" "$RESULTS/scoring_agreement.jsonl" <<'PY'
import json, os, sys

def rows(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]

spec = rows(sys.argv[1])
agree = rows(sys.argv[2] if len(sys.argv) > 2 else None)

if spec:
    print(f"\n  IF/CONS/SPEC cells: {len(spec)}")
    sparse = [r for r in spec if (r.get("n_cons_comparable") or 0) < 100]
    print(f"  cells with <100 comparable CONS answers: {len(sparse)}")
    print("\n  parse coverage by model:")
    by_model = {}
    for row in spec:
        n = row.get("n_items") or 0
        p = row.get("n_parseable") or 0
        rate = p / n if n else 0.0
        by_model.setdefault(row["model"], []).append(rate)
    for model, rates in sorted(by_model.items(), key=lambda kv: min(kv[1])):
        print(f"    {model:<34} min={min(rates):.3f}  mean={sum(rates)/len(rates):.3f}")

if agree:
    print("\n  agreement with the generation-parsed path (on rows it parsed):")
    for model in sorted({r["model"] for r in agree}):
        cells = [r for r in agree if r["model"] == model]
        vals = [r["agreement_on_generated_parsed"] for r in cells
                if r["agreement_on_generated_parsed"] is not None]
        pg = [r["parse_rate_generated"] for r in cells if r["parse_rate_generated"] is not None]
        if vals:
            print(f"    {model:<34} agree={sum(vals)/len(vals):.3f}  "
                  f"old parse rate={sum(pg)/len(pg):.3f}")
        else:
            print(f"    {model:<34} agree=n/a (generation parsed nothing)")
    print("\n  Low agreement on a model that already parsed well is a red flag:")
    print("  it means the two instruments disagree where both are measurable.")
PY

# ---------------------------------------------------------------------------
# 11. Publish the small artifacts (predictions stay on the pod; they are large)
# ---------------------------------------------------------------------------
if [[ "$PUBLISH" == "1" ]]; then
  log "Publishing results to branch $RESULTS_BRANCH"
  git config user.name  "$GIT_NAME"
  git config user.email "$GIT_EMAIL"
  git checkout -B "$RESULTS_BRANCH"
  git add -f "$RESULTS"
  if git diff --cached --quiet; then
    log "No result changes to commit."
  else
    git commit -m "BELEBELE option-likelihood rescoring (${NORMALIZATION}, ${CONTINUATION_MODE})"
    if [[ -n "${GITHUB_TOKEN:-}" ]]; then
      PUSH_URL="https://x-access-token:${GITHUB_TOKEN}@${REPO_URL#https://}"
      git push "$PUSH_URL" "$RESULTS_BRANCH" --force
    else
      gh auth setup-git
      git push "$REPO_URL" "$RESULTS_BRANCH" --force
    fi
    log "Pushed results to $RESULTS_BRANCH"
  fi
else
  log "PUBLISH=0; results stay on this pod."
fi

log "Done. Artifacts in $RESULTS:"
ls -lh "$RESULTS"
log "Full rescored predictions (not published): $LIKELIHOOD_ALL"
