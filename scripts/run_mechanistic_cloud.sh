#!/usr/bin/env bash
# ============================================================================
# run_mechanistic_cloud.sh
#
# End-to-end runner for the cross-lingual refusal-direction audit (MECHANISTIC.md).
# Designed to be copied onto a fresh cloud GPU box (H100/A100), then:
#   1. clones the repo
#   2. installs the project
#   3. extracts residual activations for the triad across a layer band
#   4. sweeps layers, estimates directions, computes transfer + similarity
#   5. (optional) links separability to ASR if the frozen dataset is present
#   6. runs the addition-only refusal-strengthening intervention
#   7. pushes the small result artifacts to a results branch on GitHub
#
# It is standalone: run it from anywhere, it creates its own working copy.
#
# --------------------------------------------------------------------------
# REQUIRED environment variables
#   HF_TOKEN        Hugging Face token with access to gated Llama-3.1 weights.
#   GITHUB_TOKEN    GitHub token used to push results. Optional when the gh CLI
#                   is installed and authenticated (gh auth login); the push
#                   then goes through gh's stored credentials instead.
#
# OPTIONAL environment variables (defaults shown)
#   REPO_URL=https://github.com/gabrieljcs/ufsc-thesis
#   WORKDIR=./ufsc-thesis-mech
#   RESULTS_BRANCH=mechanistic-results
#   MODELS="llama3_1_8b_reference llamantino_anita_8b ai_sweden_llama3_8b"
#   LANGS="eng spa ita dan nor swe swa"   # matches the committed refusal_sample.jsonl
#   LAYERS="8 10 12 14 16 18 20 22 24"
#   ALPHAS="0.5 1 2"               # intervention strengths as MULTIPLES of the residual norm
#   FROZEN=outputs/dataset_frozen.jsonl   # if present, ASR linkage runs
#   SKIP_INSTALL=0                 # set 1 if the container already has deps
#   GIT_NAME / GIT_EMAIL           # committer identity for the results push
# ============================================================================
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/gabrieljcs/ufsc-thesis}"
CODE_BRANCH="${CODE_BRANCH:-main}"   # branch holding the mechanistic code + sample
WORKDIR="${WORKDIR:-./ufsc-thesis-mech}"
RESULTS_BRANCH="${RESULTS_BRANCH:-mechanistic-results}"
MODELS="${MODELS:-llama3_1_8b_reference llamantino_anita_8b ai_sweden_llama3_8b}"
LANGS="${LANGS:-eng spa ita dan nor swe swa}"
LAYERS="${LAYERS:-8 10 12 14 16 18 20 22 24}"
ALPHAS="${ALPHAS:-0.5 1 2}"   # multiples of the residual norm (used with --alpha-scale)
FROZEN="${FROZEN:-outputs/dataset_frozen.jsonl}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
GIT_NAME="${GIT_NAME:-Gabriel de Jesus}"
GIT_EMAIL="${GIT_EMAIL:-gabrieldejesus2211@gmail.com}"

log() { printf '\n\033[1;36m[mech]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[mech:error]\033[0m %s\n' "$*" >&2; exit 1; }

: "${HF_TOKEN:?Set HF_TOKEN (needs access to meta-llama/Llama-3.1-8B-Instruct)}"
export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# GitHub credentials are only used by the final publish step, but verify them NOW:
# failing after hours of GPU work would strand the results on an ephemeral pod.
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  log "Using GITHUB_TOKEN for the results push"
elif command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  log "GITHUB_TOKEN not set; using the authenticated gh CLI for the results push"
else
  die "No GitHub credential: set GITHUB_TOKEN, or install gh and run 'gh auth login'."
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

[[ -f src/thesis_eval/mechanistic/refusal.py ]] || \
  die "Mechanistic module missing from the clone — push the code to $REPO_URL first (see notes)."

# ---------------------------------------------------------------------------
# 2. Install the project
# ---------------------------------------------------------------------------
# Import the mechanistic module specifically: a pre-mechanistic thesis_eval baked
# into a Docker image must not satisfy this check, or the stale CLI would run.
if [[ "$SKIP_INSTALL" != "1" ]] && ! python -c "import thesis_eval.mechanistic.refusal" 2>/dev/null; then
  if command -v uv >/dev/null 2>&1; then
    log "Installing with uv (rtx extra: torch 2.8.0 CUDA build)"
    uv sync --extra rtx
    RUN=(uv run thesis-eval)
  else
    log "Installing with pip (rtx extra: torch 2.8.0 CUDA build)"
    pip install -e '.[rtx]'
    RUN=(thesis-eval)
  fi
else
  RUN=(thesis-eval)
  command -v thesis-eval >/dev/null 2>&1 || RUN=(python -m thesis_eval.cli)
fi
log "CLI: ${RUN[*]}"

MECH=outputs/mechanistic
RESULTS="$MECH/results"
mkdir -p "$RESULTS"

# Unpack the frozen dataset from its tracked archive if not already on disk.
FROZEN_ZIP="outputs/frozen_jsonl_artifacts.zip"
if [[ ! -f "$FROZEN" && -f "$FROZEN_ZIP" ]]; then
  log "Extracting frozen dataset from $FROZEN_ZIP"
  unzip -o "$FROZEN_ZIP" "outputs/dataset_frozen.jsonl" -d . >/dev/null
fi

# ---------------------------------------------------------------------------
# 3. Ensure the refusal sample exists (the extraction input)
# ---------------------------------------------------------------------------
SAMPLE="$MECH/refusal_sample.jsonl"
if [[ -f "$SAMPLE" ]]; then
  log "Using committed sample: $SAMPLE"
elif [[ -f "$FROZEN" ]]; then
  log "Sample absent; building it from $FROZEN"
  "${RUN[@]}" prepare-mechanistic-sample \
    --rows "$FROZEN" --output "$SAMPLE" \
    $(printf -- '--model %s ' $MODELS) \
    $(printf -- '--language %s ' $LANGS)
else
  die "No sample and no frozen dataset ($FROZEN). Commit $SAMPLE or upload the frozen dataset."
fi

# ---------------------------------------------------------------------------
# 4. Extract activations across the layer band, per model
# ---------------------------------------------------------------------------
ACT_ARGS=()
for model in $MODELS; do
  out="$MECH/${model}.activations.npz"
  log "Extracting activations: $model  (layers: $LAYERS)"
  "${RUN[@]}" extract-refusal-activations \
    --sample "$SAMPLE" --model "$model" --output "$out" \
    $(printf -- '--layers %s ' $LAYERS) \
    --device auto --dtype auto
  ACT_ARGS+=(--activations "$out")
done

# ---------------------------------------------------------------------------
# 5. Sweep layers, estimate directions, transfer + similarity
# ---------------------------------------------------------------------------
log "Analyzing directions (layer sweep + transfer + similarity)"
"${RUN[@]}" analyze-refusal-directions \
  "${ACT_ARGS[@]}" \
  --summary-output        "$RESULTS/refusal_summary.csv" \
  --projections-output    "$RESULTS/refusal_projections.jsonl" \
  --language-similarity-output "$RESULTS/language_similarity.csv" \
  --transfer-output       "$RESULTS/transfer.csv" \
  --layer-sweep-output    "$RESULTS/layer_sweep.csv" \
  --directions-output     "$RESULTS/directions.npz"

SEL_LAYER="$(python - "$RESULTS/layer_sweep.csv" <<'PY'
import csv, sys
with open(sys.argv[1], encoding="utf-8-sig") as fh:
    best = [r for r in csv.DictReader(fh) if str(r.get("selected")).lower() == "true"]
print(best[0]["layer"] if best else "")
PY
)"
[[ -n "$SEL_LAYER" ]] || die "Could not read the selected layer from the sweep."
log "Selected residual layer from sweep: $SEL_LAYER"
read -r BAND_MIN _ <<<"$(printf '%s\n' $LAYERS | sort -n | head -1)"
read -r BAND_MAX _ <<<"$(printf '%s\n' $LAYERS | sort -n | tail -1)"
if [[ "$SEL_LAYER" == "$BAND_MIN" || "$SEL_LAYER" == "$BAND_MAX" ]]; then
  log "WARNING: selected layer $SEL_LAYER sits at the edge of the sweep band ($BAND_MIN..$BAND_MAX);"
  log "         the true optimum may lie outside it. Consider re-running with a wider LAYERS band."
fi

# ---------------------------------------------------------------------------
# 6. ASR linkage (only if the frozen dataset is on the box)
# ---------------------------------------------------------------------------
if [[ -f "$FROZEN" ]]; then
  log "Linking separability to ASR"
  "${RUN[@]}" link-refusal-asr \
    --summary "$RESULTS/refusal_summary.csv" \
    --language-similarity "$RESULTS/language_similarity.csv" \
    --rows "$FROZEN" \
    --output "$RESULTS/separability_vs_asr.csv"
else
  log "SKIP ASR linkage: $FROZEN not present (upload it and run link-refusal-asr later)."
fi

# ---------------------------------------------------------------------------
# 7. Addition-only refusal-strengthening intervention, per model
# ---------------------------------------------------------------------------
# Intervene on previously-complied prompts: the 'unsafe' rows of the sample.
for model in $MODELS; do
  prompts="$MECH/${model}.complied_prompts.jsonl"
  python - "$SAMPLE" "$model" "$prompts" <<'PY'
import json, sys
src, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src, encoding="utf-8") as fh, open(out, "w", encoding="utf-8") as wh:
    for line in fh:
        row = json.loads(line)
        if row.get("model") == model and row.get("contrast_label") == "unsafe":
            wh.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
  if [[ ! -s "$prompts" ]]; then
    log "No complied prompts for $model; skipping intervention."
    continue
  fi
  for a in $ALPHAS; do
    log "Refusal-strengthening intervention: $model  (residual layer $SEL_LAYER, alpha ${a}x norm)"
    "${RUN[@]}" apply-refusal-addition \
      --prompts "$prompts" --model "$model" \
      --directions "$RESULTS/directions.npz" \
      --output "$RESULTS/intervention_${model}_alpha${a}.jsonl" \
      --layer "$SEL_LAYER" --alpha "$a" --alpha-scale \
      --device auto --dtype auto
  done
done

# ---------------------------------------------------------------------------
# 8. Push the small result artifacts to a results branch
#    (raw *.activations.npz are NOT pushed — too large; they stay on the box)
# ---------------------------------------------------------------------------
log "Publishing results to branch $RESULTS_BRANCH"
git config user.name  "$GIT_NAME"
git config user.email "$GIT_EMAIL"
git checkout -B "$RESULTS_BRANCH"
git add -f "$RESULTS"
if git diff --cached --quiet; then
  log "No result changes to commit."
else
  git commit -m "Mechanistic refusal-direction audit results (layer $SEL_LAYER, alphas ${ALPHAS})"
  if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    PUSH_URL="https://x-access-token:${GITHUB_TOKEN}@${REPO_URL#https://}"
    git push "$PUSH_URL" "$RESULTS_BRANCH" --force
  else
    gh auth setup-git   # route git's HTTPS auth through gh's stored credentials
    git push "$REPO_URL" "$RESULTS_BRANCH" --force
  fi
  log "Pushed results to $RESULTS_BRANCH"
fi

log "Done. Artifacts in $RESULTS:"
ls -lh "$RESULTS"
