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

## RTX Sabiá End-to-End Pilot

This is the copy-and-run pilot for the Linux RTX machine. It runs the real pipeline for Sabiá-7B on the limited StrongREJECT pilot list: Portuguese as the aligned-language baseline and Arabic as the first attack language. All outputs stay under `outputs/pilot/`; this does not write or overwrite `outputs/dataset_frozen.parquet`.

1. Enter the evaluation project.

This makes every relative path in the later commands resolve against the `eval/` repository.

```bash
cd /path/to/ufsc-thesis/eval
```

2. Check that the NVIDIA runtime is visible.

This confirms the RTX card and driver are available before installing or loading large models.

```bash
nvidia-smi
```

3. Create the environment file.

This prepares the place for `HF_TOKEN`, `OPENAI_API_KEY`, and `ANTHROPIC_API_KEY`. Fill the file before downloading gated Hugging Face assets or submitting API judge batches.

```bash
cp .env.example .env
```

4. Install the RTX and QC dependencies.

This installs the local CLI plus vLLM/Transformers for target generation and SONAR/BLASER dependencies for translation QC. Use Python 3.12 because the QC stack is not Python 3.13-compatible.

```bash
uv python install 3.12
uv sync --python 3.12 --extra rtx --extra qc
```

5. Validate the project configuration.

This checks that the model, language, and judge configuration can be loaded before any long-running work starts.

```bash
uv run thesis-eval validate-config
uv run thesis-eval runtime-profiles
```

6. Prepare the URIEL+ distance matrix.

This recreates the static 13-language distance CSV used to attach the Sabiá Portuguese-to-Arabic distance to generated rows.

```bash
uv run thesis-eval prepare-uriel
```

7. Import the limited StrongREJECT pilot list.

This writes the full normalized StrongREJECT prompt file and a 2-prompt pilot slice. Increase `--pilot-size` only after the two-prompt run succeeds end to end.

```bash
uv run thesis-eval import-strongreject \
  --source github \
  --pilot-size 2
```

8. Pull the required assets one at a time.

This downloads the larger NLLB translator, the BLASER QE checkpoint, and Sabiá-7B. The command is modular so a failed model download can be retried without repeating the others.

```bash
uv run thesis-eval download-assets --name nllb_200_3_3b
uv run thesis-eval download-assets --name blaser_2_0_qe
uv run thesis-eval download-assets --name sabia_7b
```

9. Verify the downloaded assets.

This checks that the expected files are present before the GPU run tries to load them.

```bash
uv run thesis-eval verify-assets --name nllb_200_3_3b
uv run thesis-eval verify-assets --name blaser_2_0_qe
uv run thesis-eval verify-assets --name sabia_7b
```

10. Prewarm SONAR.

This lets SONAR/fairseq2 fetch its own runtime encoder assets before BLASER QC, so the QC command is easier to monitor.

```bash
uv run thesis-eval prewarm-sonar
```

11. Translate the pilot prompts into Sabiá's baseline language.

This uses NLLB-200 3.3B on CUDA to create the Portuguese baseline translation log. The translator is loaded once and kept on GPU for the full command.

```bash
uv run thesis-eval translate \
  --prompts data/raw/strongreject/pilot_prompts.jsonl \
  --target-language por \
  --engine nllb \
  --checkpoint assets/models/nllb-200-3.3B \
  --device cuda \
  --dtype float16 \
  --output outputs/pilot/translations_por.nllb_3_3b.jsonl
```

12. Translate the same pilot prompts into Arabic.

This creates the Arabic attack-language translation log with the same NLLB checkpoint and GPU settings, keeping translator choice fixed across languages.

```bash
uv run thesis-eval translate \
  --prompts data/raw/strongreject/pilot_prompts.jsonl \
  --target-language ara \
  --engine nllb \
  --checkpoint assets/models/nllb-200-3.3B \
  --device cuda \
  --dtype float16 \
  --output outputs/pilot/translations_ara.nllb_3_3b.jsonl
```

13. Run BLASER QC for Portuguese.

This scores the source/translation pairs with BLASER 2.0 QE and writes pass/flag/fail fields into the translation log.

```bash
uv run thesis-eval translation-qc \
  --translations outputs/pilot/translations_por.nllb_3_3b.jsonl \
  --scorer blaser \
  --device cuda \
  --output outputs/pilot/translations_por.nllb_3_3b.blaser.jsonl
```

14. Run BLASER QC for Arabic.

This applies the same QC process to the Arabic attack prompts.

```bash
uv run thesis-eval translation-qc \
  --translations outputs/pilot/translations_ara.nllb_3_3b.jsonl \
  --scorer blaser \
  --device cuda \
  --output outputs/pilot/translations_ara.nllb_3_3b.blaser.jsonl
```

15. Export the human audit queues.

This prepares the XSTS-style review CSVs. For the tiny pilot, inspect these files before generation; for the final run, reviewer decisions should be imported and used to override BLASER where needed.

```bash
uv run thesis-eval export-audit-queue \
  --translations outputs/pilot/translations_por.nllb_3_3b.blaser.jsonl \
  --output outputs/pilot/audit_por.nllb_3_3b.csv \
  --include-passed

uv run thesis-eval export-audit-queue \
  --translations outputs/pilot/translations_ara.nllb_3_3b.blaser.jsonl \
  --output outputs/pilot/audit_ara.nllb_3_3b.csv \
  --include-passed
```

16. Generate Sabiá responses for Portuguese with vLLM.

This runs the target model locally on the RTX using vLLM. Portuguese is Sabiá's aligned-language baseline.

```bash
uv run thesis-eval generate-targets \
  --translations outputs/pilot/translations_por.nllb_3_3b.blaser.jsonl \
  --model sabia_7b \
  --profile rtx_5070_ti \
  --backend vllm \
  --model-path assets/models/sabia-7b \
  --max-tokens 64 \
  --output outputs/pilot/target_generations_por_sabia_rtx.jsonl
```

17. Generate Sabiá responses for Arabic with vLLM.

This runs the same target model and decoding settings on the Arabic attack-language prompts.

```bash
uv run thesis-eval generate-targets \
  --translations outputs/pilot/translations_ara.nllb_3_3b.blaser.jsonl \
  --model sabia_7b \
  --profile rtx_5070_ti \
  --backend vllm \
  --model-path assets/models/sabia-7b \
  --max-tokens 64 \
  --output outputs/pilot/target_generations_ara_sabia_rtx.jsonl
```

18. Combine the two language generation files.

This creates one Sabiá pilot generation file for judge batching.

```bash
cat \
  outputs/pilot/target_generations_por_sabia_rtx.jsonl \
  outputs/pilot/target_generations_ara_sabia_rtx.jsonl \
  > outputs/pilot/target_generations_sabia_rtx_pilot.jsonl
```

19. Export judge requests.

This creates one generic request row per target response per judge, so every response will be scored by all three judges.

```bash
uv run thesis-eval export-judge-batch \
  --generations outputs/pilot/target_generations_sabia_rtx_pilot.jsonl \
  --output outputs/pilot/judge_requests_sabia_rtx_pilot.jsonl
```

20. Export provider-shaped API batches.

This writes files for OpenAI GPT-4o, Anthropic Opus, and a hosted Llama Guard endpoint. Submit these through the provider APIs from your preferred batch client.

```bash
uv run thesis-eval export-api-batches \
  --requests outputs/pilot/judge_requests_sabia_rtx_pilot.jsonl \
  --output-dir outputs/pilot/api_batches_sabia_rtx_pilot
```

21. Save provider results in the expected locations.

After the API batch jobs finish, put the downloaded raw result files at these paths so the normalizers can read them.

```bash
ls outputs/pilot/api_batches_sabia_rtx_pilot
```

Expected input filenames for the next step:

- `outputs/pilot/api_batches_sabia_rtx_pilot/openai_results.jsonl`
- `outputs/pilot/api_batches_sabia_rtx_pilot/anthropic_results.jsonl`
- `outputs/pilot/api_batches_sabia_rtx_pilot/llama_guard_results.jsonl`

22. Normalize the OpenAI judge results.

This converts raw OpenAI batch output into the shared judge-result schema with `safe`, `unsafe`, or `invalid` labels.

```bash
uv run thesis-eval normalize-judge-results \
  --provider openai \
  --input outputs/pilot/api_batches_sabia_rtx_pilot/openai_results.jsonl \
  --output outputs/pilot/api_batches_sabia_rtx_pilot/openai_normalized.jsonl
```

23. Normalize the Anthropic judge results.

This converts raw Anthropic batch output into the same shared judge-result schema.

```bash
uv run thesis-eval normalize-judge-results \
  --provider anthropic \
  --input outputs/pilot/api_batches_sabia_rtx_pilot/anthropic_results.jsonl \
  --output outputs/pilot/api_batches_sabia_rtx_pilot/anthropic_normalized.jsonl
```

24. Normalize the hosted Llama Guard judge results.

This converts the hosted Llama Guard output into the same shared judge-result schema.

```bash
uv run thesis-eval normalize-judge-results \
  --provider hosted \
  --input outputs/pilot/api_batches_sabia_rtx_pilot/llama_guard_results.jsonl \
  --output outputs/pilot/api_batches_sabia_rtx_pilot/llama_guard_normalized.jsonl
```

25. Combine the normalized judge results.

This creates the single judge-result file required by majority vote. Majority vote will fail if any generation is missing one of the three judges.

```bash
cat \
  outputs/pilot/api_batches_sabia_rtx_pilot/openai_normalized.jsonl \
  outputs/pilot/api_batches_sabia_rtx_pilot/anthropic_normalized.jsonl \
  outputs/pilot/api_batches_sabia_rtx_pilot/llama_guard_normalized.jsonl \
  > outputs/pilot/judge_results_sabia_rtx_pilot.jsonl
```

26. Compute majority vote.

This validates the analysis-row schema, requires all three judge labels per response, and maps the final per-response label.

```bash
uv run thesis-eval majority-vote \
  --generations outputs/pilot/target_generations_sabia_rtx_pilot.jsonl \
  --judge-results outputs/pilot/judge_results_sabia_rtx_pilot.jsonl \
  --output outputs/pilot/analysis_rows_sabia_rtx_pilot.jsonl
```

27. Export the pilot result rows.

This writes a copy of the validated pilot rows in `outputs/pilot/`. It is deliberately not the frozen thesis dataset.

```bash
uv run thesis-eval export-results \
  --rows outputs/pilot/analysis_rows_sabia_rtx_pilot.jsonl \
  --output outputs/pilot/analysis_rows_sabia_rtx_pilot.export.jsonl
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

On the RTX machine, pull and verify a larger NLLB checkpoint before translating:

```bash
uv run thesis-eval download-assets --name nllb_200_3_3b
uv run thesis-eval verify-assets --name nllb_200_3_3b
```

Then translate on GPU by pointing `--checkpoint` at the local snapshot and forcing CUDA:

```bash
uv run thesis-eval translate \
  --prompts data/raw/strongreject/pilot_prompts.jsonl \
  --target-language ara \
  --engine nllb \
  --checkpoint assets/models/nllb-200-3.3B \
  --device cuda \
  --dtype float16 \
  --output outputs/pilot/translations_ara.nllb_3_3b.jsonl
```

Use `assets/models/nllb-200-1.3B` instead if you pulled `nllb_200_1_3b`. The translation command loads NLLB once per run and keeps it on the selected device for all prompts.

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
  --device cuda \
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
