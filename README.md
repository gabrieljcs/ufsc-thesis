# thesis-eval

Evaluation code for the dissertation "Cross-Lingual Jailbreak Evaluation of Language-Specific LLMs." The authoritative research contract is `../AGENTS.md`.

## Current Status

This repository now contains the first pipeline foundation:

- configuration validation for models, languages, and judges;
- URIEL+ distance-matrix preparation and LaTeX table export;
- NLLB translation log records and QC hooks;
- batched API judge request/export/import flow;
- provider-shaped batch files for OpenAI and Anthropic plus a hosted Llama Guard template;
- XSTS audit queue export/import and spBLEU attachment hooks;
- majority-vote and metric helpers;
- a mockable tiny pilot path under `outputs/pilot/`;
- runtime profiles for MacBook and RTX 5070 Ti;
- a Docker GPU profile for RTX 5070 Ti / CUDA 12.8 / vLLM.

The current pilot is for pipeline validation only. It is not thesis evidence until real StrongREJECT prompts, real translations, real target-model generations, API judge results, and human audit decisions are imported.

## Setup

Local development:

```bash
cd eval
uv python install 3.12
uv sync
uv run thesis-eval validate-config
uv run thesis-eval runtime-profiles
```

MacBook Pro setup:

```bash
cd eval
uv python install 3.12
uv sync --python 3.12 --extra mac
uv run thesis-eval validate-config
uv run thesis-eval prepare-uriel
uv run thesis-eval run-pilot
```

Or use the convenience script:

```bash
scripts/macbook_pilot_no_inference.sh
```

The MacBook pilot above does not run model inference. It creates the full file layout with mock target responses so you can test translation logs, QC records, judge batch export, and majority-vote plumbing. Optional local Mac inference is available through the Transformers/MPS path, but use it only for tiny smoke tests unless the machine has enough unified memory for the selected model:

```bash
uv run thesis-eval generate-targets \
  --translations outputs/pilot/translations_qc.jsonl \
  --model sabia_7b \
  --profile macbook \
  --backend transformers \
  --model-path assets/models/sabia-7b \
  --stream \
  --output outputs/pilot/target_generations.macbook.jsonl
```

For the guarded two-prompt Sabiá smoke test after Portuguese translation and BLASER QC:

```bash
uv run thesis-eval model-status --model sabia_7b
uv run thesis-eval download-assets --name sabia_7b
uv run thesis-eval verify-assets --name sabia_7b
scripts/macbook_sabia_generation_smoke.sh
```

This uses `--max-tokens 64`, `--stream`, and only the Portuguese pilot translations. It may still be too heavy for some MacBooks; stop it if memory pressure becomes uncomfortable and move target generation to the RTX/vLLM path.

Generation prefers the downloaded local asset directory when present. You can force offline local loading with `--model-path assets/models/sabia-7b`; Transformers then uses `local_files_only=True`.

GPU environment for the RTX 5070 Ti:

```bash
cd eval
docker build -f docker/Dockerfile.gpu -t thesis-eval:gpu .
docker run --rm --gpus all -v "$PWD":/work/eval -w /work/eval thesis-eval:gpu thesis-eval validate-config
```

No-inference RTX validation:

```bash
scripts/rtx_validate_no_inference.sh
```

RTX target generation uses vLLM:

```bash
docker run --rm --gpus all \
  -v "$PWD":/work/eval \
  -w /work/eval \
  --env-file .env \
  thesis-eval:gpu thesis-eval generate-targets \
    --translations data/translations/ara.final.jsonl \
    --model sabia_7b \
    --profile rtx_5070_ti \
    --output outputs/pilot/target_generations.rtx.jsonl
```

Create `.env` from `.env.example` before running real API or Hugging Face workloads:

```bash
cp .env.example .env
```

## Core Workflow

Prepare the URIEL+ matrix and generated thesis table:

```bash
uv run thesis-eval prepare-uriel
```

The current reproducible path uses the static URIEL+ CSV already mirrored in the thesis. The upstream `urielplus` package currently depends on TensorFlow Addons wheels that do not support Python 3.12+, so it is kept out of the default/Mac install path.

## Modular Downloads

Downloads are intentionally modular. Inspect the manifest first:

```bash
uv run thesis-eval list-assets
uv run thesis-eval asset-status
```

Dry-run a specific asset or group:

```bash
uv run thesis-eval download-assets --name strongreject --dry-run
uv run thesis-eval download-assets --group translation --dry-run
```

Pull assets slowly, one stage at a time:

```bash
# Prompt datasets
uv run thesis-eval download-assets --name strongreject
uv run thesis-eval download-assets --name multijail

# Benign specialization probe
uv run thesis-eval download-assets --name belebele

# Translation/QC models
uv run thesis-eval download-assets --name nllb_200
uv run thesis-eval download-assets --name nllb_200_1_3b
uv run thesis-eval download-assets --name nllb_200_3_3b
uv run thesis-eval download-assets --name blaser_2_0_qe

# Target models, one at a time
uv run thesis-eval download-assets --name sabia_7b
uv run thesis-eval download-assets --name fauno
uv run thesis-eval download-assets --name gpt_sw3
uv run thesis-eval download-assets --name bggpt
```

After each download, verify it before continuing:

```bash
uv run thesis-eval verify-assets --name strongreject
uv run thesis-eval verify-assets --name nllb_200
uv run thesis-eval verify-assets --name sabia_7b
```

For a slower but more informative local inventory, add `--deep`. This computes a deterministic manifest hash from file names and sizes without reading every model weight into memory:

```bash
uv run thesis-eval verify-assets --name nllb_200 --deep
```

Convenience checker:

```bash
scripts/check_assets.sh strongreject nllb_200
```

Groups are available when you are ready:

```bash
uv run thesis-eval download-assets --group datasets
uv run thesis-eval download-assets --group jailbreaks
uv run thesis-eval download-assets --group translation
uv run thesis-eval download-assets --group specialization
uv run thesis-eval download-assets --group targets
```

Downloaded Hugging Face snapshots go under `assets/` and are ignored by git. StrongREJECT normalized JSONL is written under `data/raw/strongreject/`, also ignored except for `.gitkeep`.

The default NLLB asset is filtered to avoid duplicate weights. It downloads tokenizer/config files plus `pytorch_model.bin`, keeping the expected size near 2.5 GB. If you saw a roughly 20 MB download, that was only tokenizer/config files; rerun `download-assets --name nllb_200 --force` after this manifest update.

Larger translator options are available:

- `nllb_200`: `facebook/nllb-200-distilled-600M`
- `nllb_200_1_3b`: `facebook/nllb-200-1.3B`
- `nllb_200_3_3b`: `facebook/nllb-200-3.3B`

NLLB-200 does not have an official 6B/7B checkpoint. The 7B-class assets in this project are the target models, not translators.

Import StrongREJECT forbidden prompts. The default source is the public GitHub CSV from the StrongREJECT repository; Hugging Face is also supported but may require accepting dataset terms.

```bash
uv run thesis-eval import-strongreject --source github --pilot-size 2
```

This writes:

- `data/raw/strongreject/prompts.jsonl`
- `data/raw/strongreject/pilot_prompts.jsonl`

You can also import a local CSV with `category`, `source`, and `forbidden_prompt` columns:

```bash
uv run thesis-eval import-strongreject \
  --source local \
  --input /path/to/strongreject_dataset.csv
```

Translate a JSONL prompt file. Each row must include `prompt_id` and `text`; optional fields include `harmful_goal`, `expected_output_form`, `fixed_constraints`, and `allowed_adaptation`.

```bash
uv run thesis-eval translate \
  --prompts data/raw/strongreject/prompts.jsonl \
  --target-language ara \
  --engine nllb \
  --output data/translations/ara.jsonl
```

Run translation QC. The current code includes a deterministic development scorer so the flow is testable; final research runs must connect BLASER 2.0 and import human XSTS audit decisions.

```bash
uv run thesis-eval translation-qc \
  --translations data/translations/ara.jsonl \
  --output data/translations/ara.qc.jsonl
```

Thesis-valid QC uses BLASER 2.0 QE via SONAR. Use Python 3.12; fairseq2 does not currently support Python 3.13 in the supported wheels. On macOS, install libsndfile if needed:

```bash
brew install libsndfile
uv python install 3.12
uv sync --python 3.12 --extra qc
uv run thesis-eval download-assets --name blaser_2_0_qe
uv run thesis-eval verify-assets --name blaser_2_0_qe
uv run thesis-eval prewarm-sonar

uv run thesis-eval translation-qc \
  --translations outputs/pilot/translations_ara.nllb.jsonl \
  --scorer blaser \
  --output outputs/pilot/translations_ara.blaser.jsonl
```

BLASER 2.0 QE predicts XSTS-like semantic similarity scores on an approximately 1--5 scale. The default pilot thresholds are `pass >= 4.0`, `fail < 3.0`, and `flag` in between. These are starting thresholds only; AGENTS.md requires calibrating pass/flag/fail thresholds on the audited subset before final analysis.

`prewarm-sonar` intentionally triggers SONAR/fairseq2's own text encoder downloads before QC, so the later `translation-qc --scorer blaser` command is less surprising. The CLI prints stage-by-stage progress messages to stderr during downloads, translation, embedding, scoring, generation, and batch export.

Generate target-model responses locally. Use `--profile macbook --backend transformers` for Apple Silicon tests, `--profile rtx_5070_ti --backend vllm` for the RTX machine, or `--backend mock` for no-inference pipeline checks.

```bash
uv run thesis-eval generate-targets \
  --translations data/translations/ara.qc.jsonl \
  --model sabia_7b \
  --profile rtx_5070_ti \
  --output outputs/pilot/target_generations.jsonl
```

Export judge batch requests for API processing:

```bash
uv run thesis-eval export-judge-batch \
  --generations outputs/pilot/target_generations.jsonl \
  --output outputs/pilot/judge_requests.jsonl
```

Create provider-shaped batch files:

```bash
uv run thesis-eval export-api-batches \
  --requests outputs/pilot/judge_requests.jsonl \
  --output-dir outputs/pilot/api_batches
```

This writes:

- `openai_gpt_4o_batch.jsonl`: OpenAI Batch API input for `/v1/chat/completions`.
- `anthropic_claude_opus_4_batch.json`: Anthropic Message Batches request body.
- `hosted_llama_guard_4_batch.jsonl`: generic hosted Llama Guard request template.

Normalize downloaded provider results:

```bash
uv run thesis-eval normalize-judge-results \
  --provider openai \
  --input outputs/pilot/api_batches/openai_results.jsonl \
  --output outputs/pilot/api_batches/openai_normalized.jsonl

uv run thesis-eval normalize-judge-results \
  --provider anthropic \
  --input outputs/pilot/api_batches/anthropic_results.jsonl \
  --output outputs/pilot/api_batches/anthropic_normalized.jsonl

uv run thesis-eval normalize-judge-results \
  --provider hosted \
  --input outputs/pilot/api_batches/llama_guard_results.jsonl \
  --output outputs/pilot/api_batches/llama_guard_normalized.jsonl
```

Concatenate the three normalized files into one `judge_results.jsonl` before majority voting. Each normalized row contains `request_id`, `run_id`, `judge`, `label`, `raw_response`, and `failure`.

After querying GPT-4o, Claude/Opus, and hosted Llama Guard via API, import a JSONL result file:

```bash
uv run thesis-eval majority-vote \
  --generations outputs/pilot/target_generations.jsonl \
  --judge-results outputs/pilot/judge_results.jsonl \
  --output outputs/pilot/analysis_rows.jsonl
```

## Translation Review

Export the human XSTS audit queue:

```bash
uv run thesis-eval export-audit-queue \
  --translations data/translations/ara.qc.jsonl \
  --output data/translations/ara.audit.csv
```

Reviewers fill `audit_decision` with `pass` or `revise`; they may also fill `harm_preservation`, `translated_text`, `xsts_note`, and `revision_note`. Import the decisions:

```bash
uv run thesis-eval import-audit \
  --translations data/translations/ara.qc.jsonl \
  --audit-csv data/translations/ara.audit.csv \
  --output data/translations/ara.audited.jsonl
```

Attach reference-subset spBLEU scores when trusted references exist:

```bash
uv run thesis-eval reference-spbleu \
  --translations data/translations/ara.audited.jsonl \
  --references data/translations/ara.references.jsonl \
  --output data/translations/ara.final.jsonl
```

Reference rows must contain `prompt_id`, `target_language`, and `reference_text`.

## Tiny Pilot

The mock pilot creates the expected file layout without external services:

```bash
uv run thesis-eval prepare-uriel
uv run thesis-eval import-strongreject --source github --pilot-size 2
uv run thesis-eval run-pilot
```

This is the recommended MacBook Pro test path before moving to the RTX machine. It uses the imported two-prompt StrongREJECT slice when available, but it does not load target models, NLLB, or BLASER.

It writes:

- `outputs/pilot/translations.jsonl`
- `outputs/pilot/translations_qc.jsonl`
- `outputs/pilot/target_generations.jsonl`
- `outputs/pilot/judge_requests.jsonl`

Then export provider batches:

```bash
uv run thesis-eval export-api-batches \
  --requests outputs/pilot/judge_requests.jsonl \
  --output-dir outputs/pilot/api_batches
```

## Testing

```bash
PYTHONPATH=src python3 -m unittest
```

## Important Rules

- Do not overwrite `outputs/dataset_frozen.parquet` during pilots.
- Do not use StrongREJECT's evaluator.
- Do not short-circuit judges; every row needs all three labels.
- Do not treat mock pilot outputs as results.
- Download large models one at a time and check `asset-status` after each download.
- Run `verify-assets` after each download before starting the next one.
- Keep `TODO.md` and the LaTeX draft comments synchronized with implementation progress.
