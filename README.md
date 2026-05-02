# thesis-eval

Evaluation code for the thesis **Cross-Lingual Jailbreak Evaluation of Language-Specific LLMs**.

The authoritative research contract is [`../AGENTS.md`](../AGENTS.md). This README is the operator manual: it explains the project layout, the command interface, and the exact Windows + RTX Docker workflow for the Sabiá pilot and the full experiment.

## What Runs Where

The RTX machine is a Windows host with an RTX 5070 Ti. vLLM does not run natively on Windows, so the supported GPU path is:

- Windows PowerShell controls the run.
- Docker Desktop with WSL2 runs a Linux CUDA container.
- Model assets live on the Windows host under `Downloads\thesis-eval-assets`.
- The container mounts that host asset folder at `/work/eval/assets`.
- NLLB translation, BLASER QC, and target-model inference all run inside Docker on CUDA.
- API judges are not run locally. The CLI exports batch files; you submit them to provider APIs and import the returned results.

Each model-backed command is run in a fresh container. That frees the 16 GB GPU between stages, which matters because NLLB, BLASER, and 7B target inference should not share VRAM.

## Project Structure

- `configs/`: experiment configuration.
  - `assets.yaml`: download manifest for datasets, translators, BLASER, and target models.
  - `languages.yaml`: 13-language inventory and aligned-language map.
  - `models.yaml`: target model identifiers and metadata.
  - `judges.yaml`: API judge identifiers.
  - `runtime.yaml`: MacBook and RTX runtime profiles.
- `data/`: reproducible inputs and intermediate translation records.
  - `data/raw/strongreject/`: imported StrongREJECT prompts and the limited pilot slice.
  - `data/translations/`: final/full-run translation logs.
  - `data/uriel_plus/distance_matrix.csv`: static URIEL+ distance matrix.
- `assets/`: mounted model/dataset assets. On Windows this should be backed by `Downloads\thesis-eval-assets`.
- `outputs/`: generated experiment outputs.
  - `outputs/pilot/`: tiny Sabiá pilot outputs.
  - `outputs/final/`: recommended full-run working area.
  - `outputs/dataset_frozen.parquet`: final frozen dataset; do not overwrite during pilots.
- `src/thesis_eval/`: implementation.
  - `translation/`: NLLB translation, BLASER QC, XSTS audit, spBLEU hooks.
  - `models/`: target generation wrappers, including vLLM.
  - `evaluation/`: judge batch export/import and majority vote.
  - `benchmarks/`: StrongREJECT import and URIEL matrix handling.
  - `metrics/`: ASR and SPEC helpers.
- `docker/`: CUDA/vLLM image for the RTX workflow.
- `tests/`: deterministic unit tests.

## Command Interface

Common commands:

- `validate-config`: load model/language/judge configuration.
- `runtime-profiles`: show the MacBook and RTX runtime profiles.
- `list-assets`: list downloadable assets.
- `asset-status`: show whether assets exist locally.
- `download-assets`: download one asset or group.
- `verify-assets`: check that downloaded assets look complete.
- `prepare-uriel`: recreate the static URIEL+ CSV and LaTeX table.
- `import-strongreject`: import forbidden prompts and create a pilot slice.
- `translate`: run placeholder or NLLB translation.
- `translation-qc`: run heuristic or BLASER QC.
- `prewarm-sonar`: fetch/load SONAR runtime assets before BLASER QC.
- `export-audit-queue`: create XSTS human-review CSVs.
- `import-audit`: import reviewer corrections/decisions.
- `generate-targets`: run target model generation with mock, Transformers, or vLLM.
- `export-judge-batch`: create one judge request per target response per judge.
- `export-api-batches`: create provider-shaped API batch files.
- `normalize-judge-results`: normalize raw provider responses.
- `majority-vote`: require all three judges and compute final labels.
- `export-results`: export validated analysis rows.

After the Windows setup defines `$GPU`, ask the CLI for flags:

```powershell
docker @GPU thesis-eval --help
docker @GPU thesis-eval translate --help
docker @GPU thesis-eval generate-targets --help
```

## Windows RTX Setup

Run this section once on the Windows RTX computer.

1. Install prerequisites.

Install these outside the repo:

- Windows 11
- Recent NVIDIA driver with RTX 5070 Ti support
- Docker Desktop with WSL2 backend enabled
- Git

Confirm Docker can see the GPU:

```powershell
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

The output should list the RTX 5070 Ti.

2. Clone and enter the repo.

```powershell
git clone https://github.com/gabrieljcs/ufsc-thesis.git
cd ufsc-thesis\eval
```

3. Create host folders for model assets and Hugging Face cache.

These folders live in Downloads. The container sees them as `/work/eval/assets` and `/root/.cache/huggingface`.

```powershell
$PROJECT = (Get-Location).Path
$ASSETS = Join-Path $env:USERPROFILE "Downloads\thesis-eval-assets"
$HF_CACHE = Join-Path $env:USERPROFILE "Downloads\thesis-hf-cache"

New-Item -ItemType Directory -Force $ASSETS | Out-Null
New-Item -ItemType Directory -Force $HF_CACHE | Out-Null
New-Item -ItemType Directory -Force outputs\pilot | Out-Null
New-Item -ItemType Directory -Force outputs\final | Out-Null
```

4. Create the environment file.

This file is passed into Docker. Add `HF_TOKEN` if any Hugging Face model requires authentication. Add API keys when you are ready to submit judge batches.

```powershell
if (!(Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

5. Build the GPU image.

This image contains the project CLI, PyTorch CUDA, vLLM, Transformers, NLLB dependencies, BLASER/SONAR dependencies, and sacreBLEU.

```powershell
docker build -f docker/Dockerfile.gpu -t thesis-eval:gpu .
```

6. Define the reusable Docker wrapper.

Keep using the same PowerShell session after this. If you open a new terminal, rerun this block.

```powershell
$GPU = @(
  "run", "--rm", "--gpus", "all",
  "-v", "${PROJECT}:/work/eval",
  "-v", "${ASSETS}:/work/eval/assets",
  "-v", "${HF_CACHE}:/root/.cache/huggingface",
  "-w", "/work/eval",
  "--env-file", ".env",
  "thesis-eval:gpu"
)
```

7. Smoke-test the image.

This confirms that the container sees CUDA and that the thesis CLI can load its config.

```powershell
docker @GPU python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
docker @GPU thesis-eval validate-config
docker @GPU thesis-eval runtime-profiles
```

## Asset Downloads

Downloads are modular. They write into `/work/eval/assets`, which is the mounted Windows folder `Downloads\thesis-eval-assets`.

List the manifest:

```powershell
docker @GPU thesis-eval list-assets
docker @GPU thesis-eval asset-status
```

Download and verify one thing at a time:

```powershell
docker @GPU thesis-eval download-assets --name strongreject
docker @GPU thesis-eval verify-assets --name strongreject

docker @GPU thesis-eval download-assets --name nllb_200_3_3b
docker @GPU thesis-eval verify-assets --name nllb_200_3_3b

docker @GPU thesis-eval download-assets --name blaser_2_0_qe
docker @GPU thesis-eval verify-assets --name blaser_2_0_qe

docker @GPU thesis-eval download-assets --name sabia_7b
docker @GPU thesis-eval verify-assets --name sabia_7b
```

Other target models for the full run:

```powershell
docker @GPU thesis-eval download-assets --name fauno
docker @GPU thesis-eval verify-assets --name fauno

docker @GPU thesis-eval download-assets --name gpt_sw3
docker @GPU thesis-eval verify-assets --name gpt_sw3

docker @GPU thesis-eval download-assets --name bggpt
docker @GPU thesis-eval verify-assets --name bggpt
```

Other datasets for the full run:

```powershell
docker @GPU thesis-eval download-assets --name multijail
docker @GPU thesis-eval verify-assets --name multijail

docker @GPU thesis-eval download-assets --name belebele
docker @GPU thesis-eval verify-assets --name belebele
```

Notes:

- Use `nllb_200_3_3b` for the strongest official NLLB-200 translator.
- Use `nllb_200_1_3b` or `nllb_200` if VRAM or time becomes a blocker.
- NLLB has no official 6B/7B checkpoint; 7B-class downloads here are target models.

## Pilot: Sabiá End-to-End

This pilot is the real pipeline on a limited prompt list. It uses:

- Target model: `sabia_7b`
- Baseline language: Portuguese (`por`)
- Attack language: Arabic (`ara`)
- Prompt list: `data/raw/strongreject/pilot_prompts.jsonl`
- Output folder: `outputs/pilot/`

The pilot does not create or overwrite the frozen dataset.

1. Prepare config-derived files.

This validates config and recreates the URIEL+ distance matrix used later to attach linguistic distance values.

```powershell
docker @GPU thesis-eval validate-config
docker @GPU thesis-eval prepare-uriel
```

2. Import StrongREJECT and create the pilot prompt slice.

This writes the full normalized prompt file plus a 2-prompt pilot file.

```powershell
docker @GPU thesis-eval import-strongreject --source github --pilot-size 2
```

3. Prewarm SONAR.

This lets SONAR/fairseq2 download and initialize its text encoder before the BLASER scoring step.

```powershell
docker @GPU thesis-eval prewarm-sonar
```

4. Translate the pilot prompts into Portuguese with NLLB on GPU.

This creates the aligned-language baseline prompts for Sabiá. The NLLB model is loaded on CUDA in fp16.

```powershell
docker @GPU thesis-eval translate `
  --prompts data/raw/strongreject/pilot_prompts.jsonl `
  --target-language por `
  --engine nllb `
  --checkpoint assets/models/nllb-200-3.3B `
  --device cuda `
  --dtype float16 `
  --output outputs/pilot/translations_por.nllb_3_3b.jsonl
```

5. Translate the same pilot prompts into Arabic with NLLB on GPU.

This creates the first cross-lingual attack prompts with the same translator and decoding setup.

```powershell
docker @GPU thesis-eval translate `
  --prompts data/raw/strongreject/pilot_prompts.jsonl `
  --target-language ara `
  --engine nllb `
  --checkpoint assets/models/nllb-200-3.3B `
  --device cuda `
  --dtype float16 `
  --output outputs/pilot/translations_ara.nllb_3_3b.jsonl
```

6. Run BLASER QC for Portuguese on GPU.

This scores source/translation semantic preservation and writes `blaser_qe_score` plus `pass | flag | fail`.

```powershell
docker @GPU thesis-eval translation-qc `
  --translations outputs/pilot/translations_por.nllb_3_3b.jsonl `
  --scorer blaser `
  --device cuda `
  --output outputs/pilot/translations_por.nllb_3_3b.blaser.jsonl
```

7. Run BLASER QC for Arabic on GPU.

This applies the same QC process to the attack-language translations.

```powershell
docker @GPU thesis-eval translation-qc `
  --translations outputs/pilot/translations_ara.nllb_3_3b.jsonl `
  --scorer blaser `
  --device cuda `
  --output outputs/pilot/translations_ara.nllb_3_3b.blaser.jsonl
```

8. Export XSTS audit queues.

This creates reviewer CSVs. For the tiny pilot, inspect them manually. For the final project, reviewer decisions should be imported before generation.

```powershell
docker @GPU thesis-eval export-audit-queue `
  --translations outputs/pilot/translations_por.nllb_3_3b.blaser.jsonl `
  --output outputs/pilot/audit_por.nllb_3_3b.csv `
  --include-passed

docker @GPU thesis-eval export-audit-queue `
  --translations outputs/pilot/translations_ara.nllb_3_3b.blaser.jsonl `
  --output outputs/pilot/audit_ara.nllb_3_3b.csv `
  --include-passed
```

9. Generate Sabiá responses for Portuguese with vLLM.

This starts a fresh container, loads Sabiá-7B from the mounted Downloads asset folder, and generates target responses on CUDA.

```powershell
docker @GPU thesis-eval generate-targets `
  --translations outputs/pilot/translations_por.nllb_3_3b.blaser.jsonl `
  --model sabia_7b `
  --profile rtx_5070_ti `
  --backend vllm `
  --model-path assets/models/sabia-7b `
  --max-tokens 64 `
  --output outputs/pilot/target_generations_por_sabia_rtx.jsonl
```

10. Generate Sabiá responses for Arabic with vLLM.

This uses the same target model and generation settings for the Arabic attack prompts.

```powershell
docker @GPU thesis-eval generate-targets `
  --translations outputs/pilot/translations_ara.nllb_3_3b.blaser.jsonl `
  --model sabia_7b `
  --profile rtx_5070_ti `
  --backend vllm `
  --model-path assets/models/sabia-7b `
  --max-tokens 64 `
  --output outputs/pilot/target_generations_ara_sabia_rtx.jsonl
```

11. Combine the two generation files.

This is a host-side file operation. It creates the single generation file used for judge batching.

```powershell
Get-Content `
  outputs/pilot/target_generations_por_sabia_rtx.jsonl, `
  outputs/pilot/target_generations_ara_sabia_rtx.jsonl |
  Set-Content outputs/pilot/target_generations_sabia_rtx_pilot.jsonl
```

12. Export judge requests.

This creates one request per generated response per judge. No API call happens here.

```powershell
docker @GPU thesis-eval export-judge-batch `
  --generations outputs/pilot/target_generations_sabia_rtx_pilot.jsonl `
  --output outputs/pilot/judge_requests_sabia_rtx_pilot.jsonl
```

13. Export provider-shaped API batch files.

This writes OpenAI, Anthropic, and hosted Llama Guard batch inputs.

```powershell
docker @GPU thesis-eval export-api-batches `
  --requests outputs/pilot/judge_requests_sabia_rtx_pilot.jsonl `
  --output-dir outputs/pilot/api_batches_sabia_rtx_pilot
```

14. Submit the judge batches outside this CLI.

Use your API batch tooling to submit:

- `outputs/pilot/api_batches_sabia_rtx_pilot/openai_gpt_4o_batch.jsonl`
- `outputs/pilot/api_batches_sabia_rtx_pilot/anthropic_claude_opus_4_batch.json`
- `outputs/pilot/api_batches_sabia_rtx_pilot/hosted_llama_guard_4_batch.jsonl`

Save the raw provider results as:

- `outputs/pilot/api_batches_sabia_rtx_pilot/openai_results.jsonl`
- `outputs/pilot/api_batches_sabia_rtx_pilot/anthropic_results.jsonl`
- `outputs/pilot/api_batches_sabia_rtx_pilot/llama_guard_results.jsonl`

15. Normalize the returned judge results.

This converts provider-specific raw output into the shared judge-result schema.

```powershell
docker @GPU thesis-eval normalize-judge-results `
  --provider openai `
  --input outputs/pilot/api_batches_sabia_rtx_pilot/openai_results.jsonl `
  --output outputs/pilot/api_batches_sabia_rtx_pilot/openai_normalized.jsonl

docker @GPU thesis-eval normalize-judge-results `
  --provider anthropic `
  --input outputs/pilot/api_batches_sabia_rtx_pilot/anthropic_results.jsonl `
  --output outputs/pilot/api_batches_sabia_rtx_pilot/anthropic_normalized.jsonl

docker @GPU thesis-eval normalize-judge-results `
  --provider hosted `
  --input outputs/pilot/api_batches_sabia_rtx_pilot/llama_guard_results.jsonl `
  --output outputs/pilot/api_batches_sabia_rtx_pilot/llama_guard_normalized.jsonl
```

16. Combine normalized judge results.

Majority vote requires all three judge labels for every generated response.

```powershell
Get-Content `
  outputs/pilot/api_batches_sabia_rtx_pilot/openai_normalized.jsonl, `
  outputs/pilot/api_batches_sabia_rtx_pilot/anthropic_normalized.jsonl, `
  outputs/pilot/api_batches_sabia_rtx_pilot/llama_guard_normalized.jsonl |
  Set-Content outputs/pilot/judge_results_sabia_rtx_pilot.jsonl
```

17. Compute majority vote.

This validates analysis rows and writes the final `safe | unsafe | invalid` label per target response.

```powershell
docker @GPU thesis-eval majority-vote `
  --generations outputs/pilot/target_generations_sabia_rtx_pilot.jsonl `
  --judge-results outputs/pilot/judge_results_sabia_rtx_pilot.jsonl `
  --output outputs/pilot/analysis_rows_sabia_rtx_pilot.jsonl
```

18. Export pilot rows.

This exports a validated pilot copy. It is not the frozen thesis dataset.

```powershell
docker @GPU thesis-eval export-results `
  --rows outputs/pilot/analysis_rows_sabia_rtx_pilot.jsonl `
  --output outputs/pilot/analysis_rows_sabia_rtx_pilot.export.jsonl
```

## Final Project Workflow

The final project repeats the same shape as the pilot, but expands to:

- 13 attack languages: `ara bul dan eng fin ita nor por rus spa swa swe ukr`
- 4 target models: `sabia_7b fauno gpt_sw3 bggpt`
- full StrongREJECT prompt file
- later MultiJail validation through the same judge pipeline
- human XSTS audit import before target generation
- final export to `outputs/dataset_frozen.parquet`

### Final 1. Define language and model variables

Run this in the same PowerShell session as `$GPU`.

```powershell
$LANGS = @("ara","bul","dan","eng","fin","ita","nor","por","rus","spa","swa","swe","ukr")
$MODELS = @("sabia_7b","fauno","gpt_sw3","bggpt")
$MODEL_PATH = @{
  sabia_7b = "assets/models/sabia-7b"
  fauno = "assets/models/fauno-italian-llm-7b"
  gpt_sw3 = "assets/models/gpt-sw3-6.7b-v2-instruct"
  bggpt = "assets/models/bggpt-7b-instruct-v0.2"
}
```

### Final 2. Prepare inputs

This imports the full StrongREJECT source and prepares URIEL+.

```powershell
docker @GPU thesis-eval prepare-uriel
docker @GPU thesis-eval import-strongreject --source github --pilot-size 2
```

The full prompt file is `data/raw/strongreject/prompts.jsonl`. The pilot file remains available separately.

### Final 3. Download all required assets

This pulls every dataset/model required for the main run. Keep it modular if your connection is unstable.

```powershell
docker @GPU thesis-eval download-assets --name strongreject
docker @GPU thesis-eval download-assets --name multijail
docker @GPU thesis-eval download-assets --name belebele
docker @GPU thesis-eval download-assets --name nllb_200_3_3b
docker @GPU thesis-eval download-assets --name blaser_2_0_qe
docker @GPU thesis-eval download-assets --name sabia_7b
docker @GPU thesis-eval download-assets --name fauno
docker @GPU thesis-eval download-assets --name gpt_sw3
docker @GPU thesis-eval download-assets --name bggpt
```

Verify them:

```powershell
docker @GPU thesis-eval verify-assets --name strongreject
docker @GPU thesis-eval verify-assets --name multijail
docker @GPU thesis-eval verify-assets --name belebele
docker @GPU thesis-eval verify-assets --name nllb_200_3_3b
docker @GPU thesis-eval verify-assets --name blaser_2_0_qe
docker @GPU thesis-eval verify-assets --name sabia_7b
docker @GPU thesis-eval verify-assets --name fauno
docker @GPU thesis-eval verify-assets --name gpt_sw3
docker @GPU thesis-eval verify-assets --name bggpt
```

### Final 4. Translate every StrongREJECT language on GPU

This runs NLLB-200 3.3B once per language. Each command exits after writing the language file, freeing VRAM.

```powershell
foreach ($lang in $LANGS) {
  docker @GPU thesis-eval translate `
    --prompts data/raw/strongreject/prompts.jsonl `
    --target-language $lang `
    --engine nllb `
    --checkpoint assets/models/nllb-200-3.3B `
    --device cuda `
    --dtype float16 `
    --output "data/translations/$lang.nllb_3_3b.jsonl"
}
```

### Final 5. Run BLASER QC for every language on GPU

This scores all source/translation pairs and writes QC logs.

```powershell
docker @GPU thesis-eval prewarm-sonar

foreach ($lang in $LANGS) {
  docker @GPU thesis-eval translation-qc `
    --translations "data/translations/$lang.nllb_3_3b.jsonl" `
    --scorer blaser `
    --device cuda `
    --output "data/translations/$lang.blaser.jsonl"
}
```

### Final 6. Export human XSTS audit queues

This creates one CSV per language. Review all flagged/failing rows plus the planned stratified sample. Human review overrides BLASER.

```powershell
foreach ($lang in $LANGS) {
  docker @GPU thesis-eval export-audit-queue `
    --translations "data/translations/$lang.blaser.jsonl" `
    --output "data/translations/$lang.audit.csv" `
    --include-passed
}
```

After review, import the decisions. The reviewer-edited CSVs should keep the same paths.

```powershell
foreach ($lang in $LANGS) {
  docker @GPU thesis-eval import-audit `
    --translations "data/translations/$lang.blaser.jsonl" `
    --audit-csv "data/translations/$lang.audit.csv" `
    --output "data/translations/$lang.final.jsonl"
}
```

### Final 7. Generate target responses for all model-language pairs

This runs vLLM on CUDA for each target model and language. It creates 52 generation files: 4 models times 13 languages.

```powershell
New-Item -ItemType Directory -Force outputs\final\generations | Out-Null

foreach ($model in $MODELS) {
  foreach ($lang in $LANGS) {
    docker @GPU thesis-eval generate-targets `
      --translations "data/translations/$lang.final.jsonl" `
      --model $model `
      --profile rtx_5070_ti `
      --backend vllm `
      --model-path $MODEL_PATH[$model] `
      --max-tokens 512 `
      --output "outputs/final/generations/${model}_${lang}.jsonl"
  }
}
```

If a model needs custom loading code, rerun that specific command with `--trust-remote-code`.

### Final 8. Combine generation files

This creates one generation file for judge batching.

```powershell
Get-ChildItem outputs\final\generations\*.jsonl |
  Get-Content |
  Set-Content outputs\final\target_generations_all.jsonl
```

### Final 9. Export all judge requests and API batches

This creates the request files for all three judges. No local judge model is loaded.

```powershell
docker @GPU thesis-eval export-judge-batch `
  --generations outputs/final/target_generations_all.jsonl `
  --output outputs/final/judge_requests_all.jsonl

docker @GPU thesis-eval export-api-batches `
  --requests outputs/final/judge_requests_all.jsonl `
  --output-dir outputs/final/api_batches
```

Submit these API batches externally:

- `outputs/final/api_batches/openai_gpt_4o_batch.jsonl`
- `outputs/final/api_batches/anthropic_claude_opus_4_batch.json`
- `outputs/final/api_batches/hosted_llama_guard_4_batch.jsonl`

Save raw results as:

- `outputs/final/api_batches/openai_results.jsonl`
- `outputs/final/api_batches/anthropic_results.jsonl`
- `outputs/final/api_batches/llama_guard_results.jsonl`

### Final 10. Normalize judge results

This converts all provider responses into the shared judge-result schema.

```powershell
docker @GPU thesis-eval normalize-judge-results `
  --provider openai `
  --input outputs/final/api_batches/openai_results.jsonl `
  --output outputs/final/api_batches/openai_normalized.jsonl

docker @GPU thesis-eval normalize-judge-results `
  --provider anthropic `
  --input outputs/final/api_batches/anthropic_results.jsonl `
  --output outputs/final/api_batches/anthropic_normalized.jsonl

docker @GPU thesis-eval normalize-judge-results `
  --provider hosted `
  --input outputs/final/api_batches/llama_guard_results.jsonl `
  --output outputs/final/api_batches/llama_guard_normalized.jsonl
```

Combine them:

```powershell
Get-Content `
  outputs/final/api_batches/openai_normalized.jsonl, `
  outputs/final/api_batches/anthropic_normalized.jsonl, `
  outputs/final/api_batches/llama_guard_normalized.jsonl |
  Set-Content outputs/final/judge_results_all.jsonl
```

### Final 11. Majority vote

This requires all three judge results for every response. It will fail rather than silently label incomplete rows.

```powershell
docker @GPU thesis-eval majority-vote `
  --generations outputs/final/target_generations_all.jsonl `
  --judge-results outputs/final/judge_results_all.jsonl `
  --output outputs/final/analysis_rows_all.jsonl
```

### Final 12. Freeze the analysis dataset

Only do this once the final run is complete and checked. This is the file the Results chapter should be based on.

```powershell
docker @GPU thesis-eval export-results `
  --rows outputs/final/analysis_rows_all.jsonl `
  --output outputs/dataset_frozen.parquet
```

If the frozen file already exists and you intentionally need to replace it:

```powershell
docker @GPU thesis-eval export-results `
  --rows outputs/final/analysis_rows_all.jsonl `
  --output outputs/dataset_frozen.parquet `
  --allow-frozen-overwrite
```

## Optional Local MacBook Checks

The MacBook path is useful only for no-inference checks or tiny Transformers smoke tests. The RTX Docker workflow above is the real model-running path.

```bash
uv python install 3.12
uv sync --python 3.12 --extra mac --extra qc
uv run thesis-eval validate-config
uv run thesis-eval run-pilot
```

## Tests

Run deterministic unit tests without model downloads:

```bash
PYTHONPATH=src python3 -m unittest
```

Or from Docker:

```powershell
docker @GPU python -m unittest
```

## Troubleshooting

- **Docker cannot see the GPU:** run `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi`. If that fails, fix Docker Desktop/WSL2/NVIDIA driver first.
- **vLLM fails on Windows outside Docker:** expected. Use Docker for target inference.
- **A model downloads into the repo instead of Downloads:** check that `$ASSETS` is mounted with `-v "${ASSETS}:/work/eval/assets"` in `$GPU`.
- **CUDA runs out of memory:** run one stage per command as shown, keep `--dtype float16`, and lower `--max-tokens` for pilot/debug runs.
- **BLASER or SONAR downloads during QC:** expected on first run. `prewarm-sonar` makes this less surprising.
- **Majority vote fails:** at least one generation is missing one of the three judge results. Normalize and combine all three providers before running `majority-vote`.
- **Do not use StrongREJECT's evaluator:** this project always uses the three API judges and majority vote.
- **Do not overwrite `outputs/dataset_frozen.parquet` during pilots:** pilot outputs belong under `outputs/pilot/`.
