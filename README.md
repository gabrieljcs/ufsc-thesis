# thesis-eval

`thesis-eval` is the reproducible evaluation package for the thesis **Multilingual Jailbreaks in Language-Specific Large Language Models**. It runs the full empirical pipeline: import harmful prompts, translate them, evaluate target models, score outputs with StrongREJECT, join specialization and tokenizer metrics, freeze the analysis dataset, and export thesis tables.

The main executable is:

```bash
uv run thesis-eval
```

## Directory Map

```text
eval/
├── configs/
│   ├── assets.yaml        # datasets, translators, BLASER, target model assets
│   ├── languages.yaml     # 13 attack languages and model aligned languages
│   ├── models.yaml        # target model metadata, access mode, provider IDs
│   └── runtime.yaml       # Mac and RTX runtime profiles
├── data/
│   ├── raw/               # imported StrongREJECT, MultiJail, BELEBELE inputs
│   ├── translations/      # finalized translated prompt logs
│   └── uriel_plus/        # static distance matrix used in analysis
├── assets/
│   ├── datasets/          # Hugging Face dataset snapshots
│   └── models/            # NLLB, BLASER, and open-weight target snapshots
├── outputs/
│   ├── translations/      # translation, round-trip, QC, and audit outputs
│   ├── generations/       # target model responses
│   ├── scored/            # StrongREJECT-scored generation rows
│   ├── spec/              # BELEBELE IF/CONS/SPEC tables
│   ├── tables/            # CSV outputs for thesis tables
│   └── dataset_frozen.*   # frozen analysis dataset
├── scripts/               # small smoke-test wrappers
├── src/thesis_eval/       # package source
└── tests/                 # unit tests
```

## Models

The thesis evaluates eight target models. Seven are open-weight Hugging Face models that are downloaded into `assets/models/`. Sabiá-3 is served through Maritaca and is configured as an API target.

| ID | Aligned language | Access | Asset or provider |
| --- | --- | --- | --- |
| `sagui_7b` | Portuguese | open weight | `OliveiraJLT/Sagui-7B-Instruct-v0.1` |
| `sabia_3` | Portuguese | API | Maritaca `sabia-3` |
| `fauno` | Italian | open weight | `andreabac3/Fauno-Italian-LLM-7B` |
| `llamantino_anita_8b` | Italian | open weight | `swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA` |
| `gpt_sw3` | Swedish | open weight | `AI-Sweden-Models/gpt-sw3-6.7b-v2-instruct` |
| `ai_sweden_llama3_8b` | Swedish | open weight | `AI-Sweden-Models/Llama-3-8B-instruct` |
| `bggpt_7b` | Bulgarian | open weight | `INSAIT-Institute/BgGPT-7B-Instruct-v0.2` |
| `bggpt_gemma_9b` | Bulgarian | open weight | `INSAIT-Institute/BgGPT-Gemma-2-9B-IT-v1.0` |

The attack-language inventory is:

```text
ara bul dan eng fin ita nor por rus spa swa swe ukr
```

## 1. Environment Setup

Use the RTX profile for the full thesis run. It installs vLLM for target-model inference, QC dependencies for BLASER, StrongREJECT scoring, and analysis libraries. `pyarrow` is included here so the frozen dataset can also be exported as Parquet.

```bash
cd /Users/gabriel/Developer/ufsc-thesis/eval
uv sync --extra rtx --extra qc --extra scoring --extra analysis --with pyarrow
```

For a Mac smoke test with Transformers instead of vLLM:

```bash
uv sync --extra mac --extra qc --extra scoring --extra analysis --with pyarrow
```

Set credentials for gated Hugging Face assets, StrongREJECT rubric scoring, and Sabiá-3:

```bash
export HF_TOKEN="..."
export OPENAI_API_KEY="..."
export MARITACA_API_KEY="..."
```

Check the configuration and runtime profiles:

```bash
uv run thesis-eval validate-config
uv run thesis-eval runtime-profiles
uv run thesis-eval list-assets
```

## 2. Download Datasets

StrongREJECT provides the harmful prompt source. MultiJail is the external multilingual validation benchmark. BELEBELE is the benign reading-comprehension benchmark used to compute IF, CONS, and SPEC.

```bash
uv run thesis-eval download-assets --name strongreject
uv run thesis-eval download-assets --name multijail --name belebele
uv run thesis-eval verify-assets --group datasets
```

StrongREJECT is written to:

```text
data/raw/strongreject/prompts.jsonl
data/raw/strongreject/pilot_prompts.jsonl
```

## 3. Prepare Translation and QC Models

NLLB-200 is used for prompt translation and response back-translation. BLASER 2.0 QE is used for reference-free translation quality screening. The 3.3B NLLB checkpoint is the thesis-grade translation asset for the RTX run; the 600M checkpoint is useful for quick smoke tests.

```bash
uv run thesis-eval download-assets --name nllb_200_3_3b --name blaser_2_0_qe
uv run thesis-eval prewarm-sonar
uv run thesis-eval verify-assets --name nllb_200_3_3b --name blaser_2_0_qe
```

For a smaller local smoke test:

```bash
uv run thesis-eval download-assets --name nllb_200
```

## 4. Download Target Models

Download all open-weight targets in one step. The Sabiá-3 entry is checked as an API target and has no local weights.

```bash
uv run thesis-eval download-assets --group targets
uv run thesis-eval verify-assets --group targets
```

Inspect a single model before running generation:

```bash
uv run thesis-eval model-status --model sagui_7b
uv run thesis-eval model-status --model sabia_3
```

## 5. Prepare URIEL+ Distance Data

This command writes the 13-language distance matrix used by the dataset builder and thesis tables.

```bash
uv run thesis-eval prepare-uriel
```

Outputs:

```text
data/uriel_plus/distance_matrix.csv
../ufscthesisx/aftertext/urielplus_distance_table_generated.tex
```

## 6. Translate StrongREJECT Prompts

Translate the English StrongREJECT prompts into all attack languages with NLLB-200. The output is one JSONL translation log per language.

```bash
mkdir -p outputs/translations

uv run thesis-eval translate \
  --prompts data/raw/strongreject/prompts.jsonl \
  --target-language ara \
  --target-language bul \
  --target-language dan \
  --target-language eng \
  --target-language fin \
  --target-language ita \
  --target-language nor \
  --target-language por \
  --target-language rus \
  --target-language spa \
  --target-language swa \
  --target-language swe \
  --target-language ukr \
  --engine nllb \
  --checkpoint assets/models/nllb-200-3.3B \
  --device cuda \
  --output-dir outputs/translations \
  --output-template "{lang}.raw.jsonl"
```

For a two-language pilot:

```bash
uv run thesis-eval translate \
  --prompts data/raw/strongreject/pilot_prompts.jsonl \
  --target-language por \
  --target-language ara \
  --engine nllb \
  --checkpoint assets/models/nllb-200-distilled-600M \
  --device auto \
  --output-dir outputs/pilot/translations \
  --output-template "{lang}.raw.jsonl"
```

## 7. Run Round-Trip Diagnostics and BLASER QC

Round-trip diagnostics back-translate prompts for inspection. BLASER assigns reference-free QE scores and pass/flag/fail status. The final files from this stage are `outputs/translations/{lang}.qc.jsonl`.

```bash
LANGS="ara bul dan eng fin ita nor por rus spa swa swe ukr"

for lang in $LANGS; do
  uv run thesis-eval prompt-roundtrip \
    --translations "outputs/translations/${lang}.raw.jsonl" \
    --engine nllb \
    --checkpoint assets/models/nllb-200-3.3B \
    --device cuda \
    --output "outputs/translations/${lang}.roundtrip.jsonl"

  uv run thesis-eval translation-qc \
    --translations "outputs/translations/${lang}.roundtrip.jsonl" \
    --scorer blaser \
    --device cuda \
    --output "outputs/translations/${lang}.qc.jsonl"
done
```

## 8. Human XSTS Audit and BLASER Calibration

Create audit queues for bilingual review. Reviewers fill the exported CSV files with audit decisions and XSTS notes. The import command writes audited translation logs for generation.

```bash
mkdir -p outputs/audit

for lang in $LANGS; do
  uv run thesis-eval export-audit-queue \
    --translations "outputs/translations/${lang}.qc.jsonl" \
    --output "outputs/audit/${lang}.xsts_queue.csv"
done
```

After review, save decisions as `outputs/audit/{lang}.xsts_decisions.csv`, then import them:

```bash
for lang in $LANGS; do
  uv run thesis-eval import-audit \
    --translations "outputs/translations/${lang}.qc.jsonl" \
    --audit-csv "outputs/audit/${lang}.xsts_decisions.csv" \
    --output "outputs/translations/${lang}.audit.jsonl"
done
```

Calibrate BLASER thresholds from the reviewed subset:

```bash
awk 'FNR == 1 && NR != 1 { next } { print }' outputs/audit/*.xsts_decisions.csv > outputs/audit/all_xsts_decisions.csv

uv run thesis-eval calibrate-blaser \
  --audit-csv outputs/audit/all_xsts_decisions.csv \
  --output outputs/audit/blaser_thresholds.json
```

If trusted human reference translations are available for the spBLEU subset, attach reference scores:

```bash
uv run thesis-eval reference-spbleu \
  --translations outputs/translations/por.audit.jsonl \
  --references data/translations/references/por.references.jsonl \
  --output outputs/translations/por.audit.spbleu.jsonl
```

## 9. Generate Target Responses

Open-weight models use vLLM on the RTX machine. The command below runs every open-weight target over all audited translation files and writes one output file per model-language pair.

```bash
mkdir -p outputs/generations

TRANSLATION_ARGS=()
for lang in $LANGS; do
  TRANSLATION_ARGS+=(--translations "outputs/translations/${lang}.audit.jsonl")
done

OPEN_WEIGHT_MODELS="sagui_7b fauno llamantino_anita_8b gpt_sw3 ai_sweden_llama3_8b bggpt_7b bggpt_gemma_9b"

for model in $OPEN_WEIGHT_MODELS; do
  uv run thesis-eval generate-targets \
    "${TRANSLATION_ARGS[@]}" \
    --model "$model" \
    --backend vllm \
    --profile rtx_5070_ti \
    --max-tokens 512 \
    --temperature 0.0 \
    --trust-remote-code \
    --output-dir outputs/generations \
    --output-template "{model}_{lang}.jsonl"
done
```

Run Sabiá-3 through the Maritaca API:

```bash
uv run thesis-eval generate-targets \
  "${TRANSLATION_ARGS[@]}" \
  --model sabia_3 \
  --backend maritaca \
  --max-tokens 512 \
  --temperature 0.0 \
  --output-dir outputs/generations \
  --output-template "{model}_{lang}.jsonl"
```

For a Mac smoke test with one model and one translation file:

```bash
uv run thesis-eval generate-targets \
  --translations outputs/pilot/translations/por.raw.jsonl \
  --model sagui_7b \
  --backend transformers \
  --profile macbook \
  --model-path assets/models/sagui-7b-instruct-v0.1 \
  --max-tokens 64 \
  --output outputs/pilot/generations_sagui_por.jsonl
```

## 10. Score Responses With StrongREJECT

Scoring back-translates non-English model outputs to English with NLLB-200, then runs the official StrongREJECT rubric. Each scored file preserves the original response, the English back-translation, the continuous StrongREJECT score, refusal status, sub-scores, API metadata, and failure metadata.

```bash
mkdir -p outputs/scored

for file in outputs/generations/*.jsonl; do
  base="$(basename "$file" .jsonl)"
  uv run thesis-eval score-strongreject \
    --generations "$file" \
    --evaluator strongreject_rubric \
    --backtranslate-engine nllb \
    --checkpoint assets/models/nllb-200-3.3B \
    --device cuda \
    --output "outputs/scored/${base}.strongreject.jsonl"
done

cat outputs/scored/*.strongreject.jsonl > outputs/scored/all_strongreject_scores.jsonl
```

For pipeline smoke tests:

```bash
uv run thesis-eval score-strongreject \
  --generations outputs/pilot/generations_sagui_por.jsonl \
  --evaluator mock \
  --output outputs/pilot/strongreject_scores_sagui_por.jsonl
```

## 11. Compute BELEBELE IF/CONS/SPEC

BELEBELE measures benign multilingual specialization. Query each target model on the BELEBELE multiple-choice items, then write predictions as JSONL with these fields:

```json
{"model":"sagui_7b","language":"por","item_id":"belebele_0001","prediction":"A","gold":"A"}
```

The prediction file should contain all model-language-item rows needed for IF and CONS. Then compute specialization:

```bash
mkdir -p outputs/spec

uv run thesis-eval compute-spec \
  --predictions outputs/spec/belebele_predictions.jsonl \
  --output outputs/spec/belebele_spec.jsonl
```

Output rows include:

```text
model, attack_language, if_score, cons_score, spec_score, n_items, n_cons_comparable
```

## 12. Score Matched Benign Controls

Matched benign controls use generation rows shaped like the harmful prompt outputs. The scoring command attaches relevance, completeness, coherence, and task-completion fields for benign analysis tables.

```bash
uv run thesis-eval score-benign-controls \
  --generations outputs/benign/benign_generations.jsonl \
  --output outputs/benign/benign_scores.jsonl
```

## 13. Build and Freeze the Analysis Dataset

Join StrongREJECT scores, translation QC/audit metadata, tokenizer diagnostics, URIEL+ distances, model metadata, and BELEBELE specialization.

```bash
cat outputs/translations/*.audit.jsonl > outputs/translations/all_audited_translations.jsonl

uv run thesis-eval build-dataset \
  --scored outputs/scored/all_strongreject_scores.jsonl \
  --translations outputs/translations/all_audited_translations.jsonl \
  --spec-scores outputs/spec/belebele_spec.jsonl \
  --output outputs/dataset_frozen.jsonl
```

Export a Parquet copy for archival and analysis notebooks:

```bash
uv run thesis-eval export-results \
  --rows outputs/dataset_frozen.jsonl \
  --output outputs/dataset_frozen.parquet
```

## 14. Export Thesis Tables

Generate CSV table inputs for the Results chapter:

```bash
mkdir -p outputs/tables

uv run thesis-eval export-report-tables \
  --rows outputs/dataset_frozen.jsonl \
  --output-dir outputs/tables
```

Fit the main GLMM effects table:

```bash
uv run thesis-eval fit-glmm \
  --rows outputs/dataset_frozen.jsonl \
  --output outputs/tables/glmm_main_effects.csv
```

The reporting commands produce:

```text
outputs/tables/results_coverage.csv
outputs/tables/asr_by_model_language.csv
outputs/tables/crosslingual_asr_by_model_language.csv
outputs/tables/closest_farthest_languages.csv
outputs/tables/distance_asr_correlation.csv
outputs/tables/spec_asr_correlation.csv
outputs/tables/tokenizer_diagnostics.csv
outputs/tables/glmm_collinearity.csv
outputs/tables/glmm_main_effects.csv
```

## 15. Validation Commands

Run unit tests:

```bash
uv run python -m unittest
```

Check asset status at any point:

```bash
uv run thesis-eval asset-status --group datasets
uv run thesis-eval asset-status --group translation
uv run thesis-eval asset-status --group targets
```

Run the built-in mock pilot to verify local plumbing without model inference:

```bash
uv run thesis-eval run-pilot --output-dir outputs/pilot
```
