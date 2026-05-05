# Laptop runbook — reconverge into the frozen dataset and Results-chapter tables

After [`TPU_RUN.md`](TPU_RUN.md) and [`RTX_RUN.md`](RTX_RUN.md) finish, all
artifacts should be on the laptop under `eval/outputs/`. This runbook is one
contiguous bash block plus a verification step.

> **Working directory** for everything below: `~/Developer/ufsc-thesis/eval`.

---

## §1 — Pre-flight checks

```bash
cd ~/Developer/ufsc-thesis/eval

# Counts before reconverging. Adjust the expected values down if you skipped
# the optional llama3_1_8b_reference baseline.
#
# llamantino_2_ultrachat_7b BELEBELE landed pre-pulled as a single combined file
# (outputs/spec/llamantino_2_ultrachat_7b.belebele.jsonl, 11700 rows = 900 × 13)
# instead of 13 per-language files, so it counts as 1 in the spec listing.
echo "translations/audit: $(ls outputs/translations/*.audit.jsonl 2>/dev/null | wc -l)   (expect 13)"
echo "generations:        $(ls outputs/generations/*.jsonl 2>/dev/null | wc -l)         (expect 104, or 117 with reference)"
echo "scored:             $(ls outputs/scored/*.strongreject.jsonl 2>/dev/null | wc -l) (must equal generations)"
echo "spec files:         $(ls outputs/spec/*.belebele*.jsonl 2>/dev/null | wc -l)      (mix of per-language + combined; sanity-check by row count below)"
echo "spec rows:          $(cat outputs/spec/*.belebele.jsonl outputs/spec/*.belebele.repaired.jsonl 2>/dev/null | wc -l)  (expect ~93600 with 8 paired models, ~105300 with reference)"
```

If any count is short, find the gap and rerun the corresponding TPU/RTX
runbook block before continuing.

---

## §2 — Repair, aggregate, compute-spec, build-dataset, freeze, export

### §2.1 Repair pass (idempotent; recovers answers emitted as choice text)

Some target models occasionally emit the answer text (e.g. "carry the herbs")
instead of the letter "A". `thesis-eval repair-belebele-predictions` re-parses
each row's `raw_output` against the actual BELEBELE choice text and updates
`prediction` / `predicted_choice` accordingly. The pass is idempotent — rows
that already had a clean letter are unchanged. Apply it to every per-model
BELEBELE output before aggregating.

```bash
# Build the input list. llamantino_2_ultrachat_7b already shipped a repaired
# file; everything else is per-language and needs the repair pass.
mkdir -p outputs/spec/repaired

for src in outputs/spec/*.belebele.jsonl; do
  name=$(basename "$src" .belebele.jsonl)
  case "$name" in
    llamantino_2_ultrachat_7b)
      # Pre-repaired file already at outputs/spec/llamantino_2_ultrachat_7b.belebele.repaired.jsonl
      cp outputs/spec/llamantino_2_ultrachat_7b.belebele.repaired.jsonl \
         outputs/spec/repaired/llamantino_2_ultrachat_7b.belebele.repaired.jsonl
      ;;
    *)
      uv run thesis-eval repair-belebele-predictions \
        --predictions "$src" \
        --output "outputs/spec/repaired/${name}.belebele.repaired.jsonl"
      ;;
  esac
done
```

### §2.2 Aggregate, compute-spec, build-dataset

```bash
# Aggregate per-model REPAIRED BELEBELE prediction files into one.
cat outputs/spec/repaired/*.belebele.repaired.jsonl > outputs/spec/belebele_predictions.jsonl

uv run thesis-eval compute-spec \
  --predictions outputs/spec/belebele_predictions.jsonl \
  --output outputs/spec/belebele_spec.jsonl

# Aggregate StrongREJECT-scored generations and audited translations.
cat outputs/scored/*.strongreject.jsonl    > outputs/scored/all_strongreject_scores.jsonl
cat outputs/translations/*.audit.jsonl     > outputs/translations/all_audited_translations.jsonl

# Build the frozen analysis dataset (one row per retained or excluded prompt-model-lang).
uv run thesis-eval build-dataset \
  --scored        outputs/scored/all_strongreject_scores.jsonl \
  --translations  outputs/translations/all_audited_translations.jsonl \
  --spec-scores   outputs/spec/belebele_spec.jsonl \
  --output        outputs/dataset_frozen.jsonl

# Parquet copy for archival + analysis notebooks.
uv run thesis-eval export-results \
  --rows   outputs/dataset_frozen.jsonl \
  --output outputs/dataset_frozen.parquet

# Results chapter CSVs (13 files, see eval/README.md §14).
mkdir -p outputs/tables
uv run thesis-eval export-report-tables \
  --rows       outputs/dataset_frozen.jsonl \
  --output-dir outputs/tables

# Mixed-effects logistic regression — main effects table.
uv run thesis-eval fit-glmm \
  --rows   outputs/dataset_frozen.jsonl \
  --output outputs/tables/glmm_main_effects.csv
```

---

## §3 — Verification

```bash
# Each entry in this list MUST exist after §2. The README §14 list is the
# source of truth; this is a copy for quick checking.
EXPECTED=(
  results_coverage.csv
  asr_by_model_language.csv
  crosslingual_asr_by_model_language.csv
  closest_farthest_languages.csv
  distance_asr_correlation.csv
  spec_asr_correlation.csv
  tokenizer_diagnostics.csv
  reference_distance_curve.csv
  counterfactual_safety_by_aligned_language.csv
  glmm_collinearity.csv
  prereg_distance_slope_retention.csv
  prereg_falsification_summary.csv
  glmm_main_effects.csv
)
for f in "${EXPECTED[@]}"; do
  if [ -f "outputs/tables/$f" ]; then
    printf "OK   %s (%d bytes)\n" "$f" "$(wc -c < "outputs/tables/$f")"
  else
    printf "MISS %s\n" "$f"
  fi
done

# SPEC sanity: no NaN-only rows, scores in [0,1] for if/cons.
uv run python - <<'PY'
import json
from pathlib import Path

bad = 0
total = 0
for line in Path("outputs/spec/belebele_spec.jsonl").read_text().splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    total += 1
    if row.get("if_score") is None or row.get("cons_score") is None:
        bad += 1
        continue
    for key in ("if_score", "cons_score"):
        v = row[key]
        if not (0 <= v <= 1):
            bad += 1
            break
print(f"belebele_spec rows: {total}; rows with bad if/cons: {bad}")
PY
```

If everything reads `OK` and `bad == 0`, the pipeline is done.

---

## §4 — Optional: BELEBELE repair-pass cross-check

The repair pass is idempotent. To prove no rows were lost between the raw
and repaired stages:

```bash
echo "raw rows:      $(cat outputs/spec/*.belebele.jsonl | wc -l)"
echo "repaired rows: $(cat outputs/spec/repaired/*.belebele.repaired.jsonl | wc -l)"
# These two should match exactly.
```

To see how many predictions changed during repair (i.e., raw_output was the
choice text rather than the letter — common for some LLaMAntino-style
checkpoints):

```bash
uv run python - <<'PY'
import json
from pathlib import Path
changed = total = 0
for repaired_path in Path("outputs/spec/repaired").glob("*.belebele.repaired.jsonl"):
    name = repaired_path.name.replace(".belebele.repaired.jsonl", "")
    raw_path = Path("outputs/spec") / f"{name}.belebele.jsonl"
    if not raw_path.exists():
        continue
    raw = {(r["language"], r["item_id"]): r for r in (json.loads(l) for l in raw_path.read_text().splitlines() if l.strip())}
    rep = [json.loads(l) for l in repaired_path.read_text().splitlines() if l.strip()]
    for r in rep:
        total += 1
        key = (r["language"], r["item_id"])
        if key in raw and raw[key].get("prediction") != r.get("prediction"):
            changed += 1
print(f"compared rows: {total}; predictions changed by repair: {changed}")
PY
```

## §5 — Optional: dataset row count cross-check

The frozen dataset should have exactly one row per (prompt, model, language)
that survived audit. Audit-excluded pairs stay as rows with
`excluded == true` and `exclusion_reason` set.

```bash
uv run python - <<'PY'
import json
from collections import Counter
from pathlib import Path

rows = [json.loads(l) for l in Path("outputs/dataset_frozen.jsonl").read_text().splitlines() if l.strip()]
print(f"total rows: {len(rows)}")
by_model = Counter(r["model"] for r in rows)
for m, c in sorted(by_model.items()):
    print(f"  {m}: {c}")

excluded = sum(1 for r in rows if r.get("excluded"))
print(f"excluded rows: {excluded}")
PY
```

Each model should appear (8 paired + optional reference), with row counts
proportional to (4,021 prompts × 13 languages) minus per-language audit
exclusions.

---

## §6 — Don't forget

- The thesis manuscript reads from `outputs/tables/*.csv` directly (see
  README §14). The table → LaTeX glue is checked in elsewhere; you don't
  need to copy values by hand.
- After this runbook, double-check no TPU VMs are still up:

  ```bash
  cd ~/Developer/ufsc-thesis/runbooks/tpu-terraform
  terraform destroy   # idempotent; safe to rerun
  gcloud compute tpus tpu-vm list --project=mestrado-495311
  ```
- Commit `outputs/dataset_frozen.jsonl` only if you've added it to the
  thesis-eval repo's frozen-dataset workflow. The default `.gitignore` keeps
  the large outputs out of git; that's intentional.
