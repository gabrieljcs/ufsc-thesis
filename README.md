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
| `fauno` | Italian | LoRA adapter | `andreabac3/Fauno-Italian-LLM-7B` |
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

## Checkpointed Execution

The workflow below is written one checkpoint at a time. Run one command, verify the output, then continue. This is especially useful for model downloads and generation runs that may take hours.

Downloads are resumable. Hugging Face assets are downloaded through `snapshot_download` with resume support into the fixed paths listed in `configs/assets.yaml`. If a network interruption or machine restart happens, rerun the same `download-assets --name ...` command. Completed files are reused, partial transfers continue, and verified assets are skipped. If a folder exists but required files are missing, the command resumes into that folder. `--force` starts a fresh download for that asset.

## 2. Download Datasets

StrongREJECT provides the harmful prompt source. MultiJail is the external multilingual validation benchmark. BELEBELE is the benign reading-comprehension benchmark used to compute IF, CONS, and SPEC.

```bash
uv run thesis-eval download-assets --name strongreject
uv run thesis-eval verify-assets --name strongreject
```

```bash
uv run thesis-eval download-assets --name multijail
uv run thesis-eval verify-assets --name multijail
```

```bash
uv run thesis-eval download-assets --name belebele
uv run thesis-eval verify-assets --name belebele
```

StrongREJECT is written to:

```text
data/raw/strongreject/prompts.jsonl
data/raw/strongreject/pilot_prompts.jsonl
```

## 3. Prepare Translation and QC Models

NLLB-200 is used for prompt translation and response back-translation. BLASER 2.0 QE is used for reference-free translation quality screening. The 3.3B NLLB checkpoint is the thesis-grade translation asset for the RTX run; the 600M checkpoint is useful for quick smoke tests.

```bash
uv run thesis-eval download-assets --name nllb_200_3_3b
uv run thesis-eval verify-assets --name nllb_200_3_3b
```

```bash
uv run thesis-eval download-assets --name blaser_2_0_qe
uv run thesis-eval prewarm-sonar
uv run thesis-eval verify-assets --name blaser_2_0_qe
```

For a smaller local smoke test:

```bash
uv run thesis-eval download-assets --name nllb_200
uv run thesis-eval verify-assets --name nllb_200
```

## 4. Download Target Models

Download each open-weight target separately. After each download, verify the asset and inspect model status. Sabiá-3 is an API target, so `model-status` checks the configured provider metadata instead of local weights.

```bash
uv run thesis-eval download-assets --name sagui_7b
uv run thesis-eval verify-assets --name sagui_7b
uv run thesis-eval model-status --model sagui_7b
```

```bash
uv run thesis-eval model-status --model sabia_3
```

```bash
uv run thesis-eval download-assets --name fauno
uv run thesis-eval verify-assets --name fauno
uv run thesis-eval model-status --model fauno
```

Fauno downloads as a PEFT/LoRA adapter (`adapter_config.json` and `adapter_model.bin`). Running it requires a compatible LLaMA/Baize base model plus LoRA-enabled inference. The current `generate-targets --backend vllm` command accepts a single model path, so Fauno needs a dedicated base-plus-adapter generation path before it can run in the same loop as the full checkpoints.

```bash
uv run thesis-eval download-assets --name llamantino_anita_8b
uv run thesis-eval verify-assets --name llamantino_anita_8b
uv run thesis-eval model-status --model llamantino_anita_8b
```

```bash
uv run thesis-eval download-assets --name gpt_sw3
uv run thesis-eval verify-assets --name gpt_sw3
uv run thesis-eval model-status --model gpt_sw3
```

```bash
uv run thesis-eval download-assets --name ai_sweden_llama3_8b
uv run thesis-eval verify-assets --name ai_sweden_llama3_8b
uv run thesis-eval model-status --model ai_sweden_llama3_8b
```

```bash
uv run thesis-eval download-assets --name bggpt_7b
uv run thesis-eval verify-assets --name bggpt_7b
uv run thesis-eval model-status --model bggpt_7b
```

```bash
uv run thesis-eval download-assets --name bggpt_gemma_9b
uv run thesis-eval verify-assets --name bggpt_gemma_9b
uv run thesis-eval model-status --model bggpt_gemma_9b
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

Translate the English StrongREJECT prompts into one attack language at a time with NLLB-200. Each command writes one JSONL translation log, so progress can be checked language by language.

```bash
mkdir -p outputs/translations
PROMPTS=data/raw/strongreject/prompts.jsonl
NLLB=assets/models/nllb-200-3.3B
DEVICE=cuda
```

Arabic:

```bash
uv run thesis-eval translate \
  --prompts "$PROMPTS" \
  --target-language ara \
  --engine nllb \
  --checkpoint "$NLLB" \
  --device "$DEVICE" \
  --output outputs/translations/ara.raw.jsonl
```

Bulgarian:

```bash
uv run thesis-eval translate \
  --prompts "$PROMPTS" \
  --target-language bul \
  --engine nllb \
  --checkpoint "$NLLB" \
  --device "$DEVICE" \
  --output outputs/translations/bul.raw.jsonl
```

Danish:

```bash
uv run thesis-eval translate \
  --prompts "$PROMPTS" \
  --target-language dan \
  --engine nllb \
  --checkpoint "$NLLB" \
  --device "$DEVICE" \
  --output outputs/translations/dan.raw.jsonl
```

English baseline:

```bash
uv run thesis-eval translate \
  --prompts "$PROMPTS" \
  --target-language eng \
  --engine nllb \
  --checkpoint "$NLLB" \
  --device "$DEVICE" \
  --output outputs/translations/eng.raw.jsonl
```

Finnish:

```bash
uv run thesis-eval translate \
  --prompts "$PROMPTS" \
  --target-language fin \
  --engine nllb \
  --checkpoint "$NLLB" \
  --device "$DEVICE" \
  --output outputs/translations/fin.raw.jsonl
```

Italian:

```bash
uv run thesis-eval translate \
  --prompts "$PROMPTS" \
  --target-language ita \
  --engine nllb \
  --checkpoint "$NLLB" \
  --device "$DEVICE" \
  --output outputs/translations/ita.raw.jsonl
```

Norwegian:

```bash
uv run thesis-eval translate \
  --prompts "$PROMPTS" \
  --target-language nor \
  --engine nllb \
  --checkpoint "$NLLB" \
  --device "$DEVICE" \
  --output outputs/translations/nor.raw.jsonl
```

Portuguese:

```bash
uv run thesis-eval translate \
  --prompts "$PROMPTS" \
  --target-language por \
  --engine nllb \
  --checkpoint "$NLLB" \
  --device "$DEVICE" \
  --output outputs/translations/por.raw.jsonl
```

Russian:

```bash
uv run thesis-eval translate \
  --prompts "$PROMPTS" \
  --target-language rus \
  --engine nllb \
  --checkpoint "$NLLB" \
  --device "$DEVICE" \
  --output outputs/translations/rus.raw.jsonl
```

Spanish:

```bash
uv run thesis-eval translate \
  --prompts "$PROMPTS" \
  --target-language spa \
  --engine nllb \
  --checkpoint "$NLLB" \
  --device "$DEVICE" \
  --output outputs/translations/spa.raw.jsonl
```

Swahili:

```bash
uv run thesis-eval translate \
  --prompts "$PROMPTS" \
  --target-language swa \
  --engine nllb \
  --checkpoint "$NLLB" \
  --device "$DEVICE" \
  --output outputs/translations/swa.raw.jsonl
```

Swedish:

```bash
uv run thesis-eval translate \
  --prompts "$PROMPTS" \
  --target-language swe \
  --engine nllb \
  --checkpoint "$NLLB" \
  --device "$DEVICE" \
  --output outputs/translations/swe.raw.jsonl
```

Ukrainian:

```bash
uv run thesis-eval translate \
  --prompts "$PROMPTS" \
  --target-language ukr \
  --engine nllb \
  --checkpoint "$NLLB" \
  --device "$DEVICE" \
  --output outputs/translations/ukr.raw.jsonl
```

For a two-language pilot:

```bash
uv run thesis-eval translate \
  --prompts data/raw/strongreject/pilot_prompts.jsonl \
  --target-language por \
  --engine nllb \
  --checkpoint assets/models/nllb-200-distilled-600M \
  --device auto \
  --output outputs/pilot/translations/por.raw.jsonl
```

```bash
uv run thesis-eval translate \
  --prompts data/raw/strongreject/pilot_prompts.jsonl \
  --target-language ara \
  --engine nllb \
  --checkpoint assets/models/nllb-200-distilled-600M \
  --device auto \
  --output outputs/pilot/translations/ara.raw.jsonl
```

## 7. Run Round-Trip Diagnostics and BLASER QC

Round-trip diagnostics back-translate prompts for inspection. BLASER assigns reference-free QE scores and pass/flag/fail status. Run these two commands for one language, check the resulting `.roundtrip.jsonl` and `.qc.jsonl`, then continue to the next language.

Arabic:

```bash
uv run thesis-eval prompt-roundtrip \
  --translations outputs/translations/ara.raw.jsonl \
  --engine nllb \
  --checkpoint "$NLLB" \
  --device "$DEVICE" \
  --output outputs/translations/ara.roundtrip.jsonl

uv run thesis-eval translation-qc \
  --translations outputs/translations/ara.roundtrip.jsonl \
  --scorer blaser \
  --device "$DEVICE" \
  --output outputs/translations/ara.qc.jsonl
```

Repeat for the remaining languages by replacing `ara` with:

```text
bul dan eng fin ita nor por rus spa swa swe ukr
```

## 8. Human XSTS Audit and BLASER Calibration

Create audit queues for bilingual review. Reviewers fill the exported CSV files with audit decisions and XSTS notes. The import command writes audited translation logs for generation.

```bash
mkdir -p outputs/audit

uv run thesis-eval export-audit-queue \
  --translations outputs/translations/ara.qc.jsonl \
  --output outputs/audit/ara.xsts_queue.csv
```

Repeat for the remaining languages by replacing `ara` with each language code.

After review, save decisions as `outputs/audit/{lang}.xsts_decisions.csv`, then import them:

```bash
uv run thesis-eval import-audit \
  --translations outputs/translations/ara.qc.jsonl \
  --audit-csv outputs/audit/ara.xsts_decisions.csv \
  --output outputs/translations/ara.audit.jsonl
```

Repeat for each reviewed language. The `.audit.jsonl` files are the prompt files used for model generation.

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

Open-weight models use vLLM on the RTX machine. Each command writes one output file for one model-language pair.

```bash
mkdir -p outputs/generations
```

Start with one model and one language:

```bash
uv run thesis-eval generate-targets \
  --translations outputs/translations/ara.audit.jsonl \
  --model sagui_7b \
  --backend vllm \
  --profile rtx_5070_ti \
  --max-tokens 512 \
  --temperature 0.0 \
  --trust-remote-code \
  --output outputs/generations/sagui_7b_ara.jsonl
```

Then continue one model-language cell at a time. Replace `ara` with each language code, and replace `sagui_7b` with each full-checkpoint model:

```text
sagui_7b
llamantino_anita_8b
gpt_sw3
ai_sweden_llama3_8b
bggpt_7b
bggpt_gemma_9b
```

Fauno is handled separately after the base-plus-adapter path is configured.

Sabiá-3 uses the Maritaca API:

```bash
uv run thesis-eval generate-targets \
  --translations outputs/translations/ara.audit.jsonl \
  --model sabia_3 \
  --backend maritaca \
  --max-tokens 512 \
  --temperature 0.0 \
  --output outputs/generations/sabia_3_ara.jsonl
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

uv run thesis-eval score-strongreject \
  --generations outputs/generations/sagui_7b_ara.jsonl \
  --evaluator strongreject_rubric \
  --backtranslate-engine nllb \
  --checkpoint "$NLLB" \
  --device "$DEVICE" \
  --output outputs/scored/sagui_7b_ara.strongreject.jsonl
```

Repeat for each file in `outputs/generations/`. After every generation file has a matching scored file, combine them:

```bash
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
