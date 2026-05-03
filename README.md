# thesis-eval

`thesis-eval` is the reproducible evaluation package for the thesis **Multilingual Jailbreaks in Language-Specific Large Language Models**. It runs the full empirical pipeline: import harmful prompts, translate them, audit translation quality, evaluate target models, score outputs with StrongREJECT, join specialization and tokenizer metrics, freeze the analysis dataset, and export thesis tables.

The active design has eight language-specific target models in four weak/strong aligned-language pairs, plus one separately reported English-aligned reference baseline: `llama3_1_8b_reference` (`meta-llama/Llama-3.1-8B-Instruct`). The reference baseline is descriptive only. It is evaluated on the same 13 attack languages, but is excluded from paired-design GLMMs, weak/strong slope-retention checks, and the four-pair falsification matrix.

The main executable is:

```powershell
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

The thesis evaluates eight paired target models. Seven are open-weight Hugging Face models that are downloaded into `assets/models/`. Sabiá-3 is served through Maritaca and is configured as an API target.

| ID | Aligned language | Access | Asset or provider |
| --- | --- | --- | --- |
| `sagui_7b` | Portuguese | open weight | `OliveiraJLT/Sagui-7B-Instruct-v0.1` |
| `sabia_3` | Portuguese | API | Maritaca `sabia-3` |
| `llamantino_2_ultrachat_7b` | Italian | open weight | `swap-uniba/LLaMAntino-2-chat-7b-hf-UltraChat-ITA` |
| `llamantino_anita_8b` | Italian | open weight | `swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA` |
| `gpt_sw3` | Swedish | open weight | `AI-Sweden-Models/gpt-sw3-6.7b-v2-instruct` |
| `ai_sweden_llama3_8b` | Swedish | open weight | `AI-Sweden-Models/Llama-3-8B-instruct` |
| `bggpt_7b` | Bulgarian | open weight | `INSAIT-Institute/BgGPT-7B-Instruct-v0.2` |
| `bggpt_gemma_9b` | Bulgarian | open weight | `INSAIT-Institute/BgGPT-Gemma-2-9B-IT-v1.0` |

The English reference baseline is configured separately:

| ID | Aligned language | Access | Asset or provider | Analysis role |
| --- | --- | --- | --- | --- |
| `llama3_1_8b_reference` | English | open weight | `meta-llama/Llama-3.1-8B-Instruct` | descriptive reference baseline |

The attack-language inventory is:

```text
ara bul dan eng fin ita nor por rus spa swa swe ukr
```

## 1. Environment Setup

The full thesis pipeline (translation, BLASER QC, target generation, StrongREJECT scoring) runs on a CUDA GPU through PyTorch and vLLM. **Use the Docker GPU profile.** Native installs cover only narrower cases — Apple Silicon smoke tests and Windows analysis-only work — described in §1.3.

> **Why Docker for the GPU path?** PyPI's default `torch` wheel on Windows is CPU-only, so `torch.cuda.is_available()` returns `False` even on a working NVIDIA driver. vLLM has no Windows wheel at all, and `fairseq2` / `sonar-space` (used by BLASER QC) are gated to non-Windows platforms in `pyproject.toml`. The Docker image builds on `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` with `torch` pinned to `cu128` (Blackwell-ready, e.g. RTX 50-series) and a Linux `vllm`, so every command in this README works end-to-end. See [`docker/README.md`](docker/README.md) for image internals.

### 1.1 Build the GPU image

Prerequisites:

- An NVIDIA GPU with a recent driver (the CUDA 12.8 runtime is bundled in the image; only the host driver matters).
- Docker Desktop with the WSL 2 backend and GPU passthrough enabled (Settings → Resources → WSL Integration) on Windows, or Docker Engine + NVIDIA Container Toolkit on Linux.
- ~30 GB free disk for the image and downloaded model assets.

Build once from the repo root:

```powershell
docker build -f docker/Dockerfile.gpu -t thesis-eval:gpu .
```

Smoke-check GPU access from inside the image:

```powershell
docker run --rm --gpus all thesis-eval:gpu nvidia-smi
docker run --rm --gpus all thesis-eval:gpu python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The second line should print `True <your GPU name>` (e.g. `True NVIDIA GeForce RTX 5070 Ti`). If it prints `False`, the host driver is missing or Docker Desktop's GPU support is not enabled.

### 1.2 Run thesis-eval commands

Two console scripts are registered:

| Script | Where it runs | Use for |
| --- | --- | --- |
| `thesis-eval-gpu` | host shell → re-execs in GPU container | any command that needs CUDA, vLLM, NLLB inference, or BLASER QC |
| `thesis-eval` | host shell, native | dataset downloads, asset/config checks, audit-CSV management, analysis |

`thesis-eval-gpu` is a thin wrapper ([`src/thesis_eval/gpu_runner.py`](src/thesis_eval/gpu_runner.py)) that forwards your arguments into the GPU container with the right `docker run` flags (bind mount, GPU passthrough, `--env-file`). From PowerShell, bash, or zsh — same syntax, no shell-quoting fork:

```powershell
uv run thesis-eval-gpu translate `
  --prompts data/raw/strongreject/prompts.jsonl `
  --target-language ara `
  --engine nllb `
  --checkpoint assets/models/nllb-200-3.3B `
  --device cuda `
  --output outputs/translations/ara.raw.jsonl
```

is equivalent to, but more ergonomic than:

```powershell
docker run --rm --gpus all `
  -v "${PWD}:/work/eval" `
  -w /work/eval `
  --env-file .env `
  thesis-eval:gpu `
  uv run thesis-eval translate <same args>
```

`.env` is auto-detected and the `--env-file` flag is skipped if it doesn't exist. Override the image tag with the `THESIS_EVAL_GPU_IMAGE` environment variable.

The pipeline mixes the two scripts: §3 (`prewarm-sonar`), §6 (translate), §7 (round-trip + QC), §9 (`generate-targets`, except `--backend maritaca`), and §10 (StrongREJECT scoring) use `thesis-eval-gpu`; everything else (§2, §4, §5, §8, §11–§14) uses plain `thesis-eval`. Each section below shows the right one for that step.

**Persistence.** The bind mount means everything `thesis-eval-gpu` writes lands on your host disk in real time — `assets/models/` for downloaded model snapshots, `data/raw/` for datasets, and `outputs/` for translations, generations, scored rows, and the frozen dataset. These artifacts survive `--rm`, image rebuilds, and machine restarts. The container itself is ephemeral (process state, `/tmp`, `/opt/venv`); only the bind mount is durable. You can kill a container mid-run and resume later — long downloads (`download-assets`) and per-language outputs are written incrementally, so the next session picks up where the last one stopped.

The image also sets `UV_PROJECT_ENVIRONMENT=/opt/venv` and `UV_NO_SYNC=1`, so `uv run` reuses the pre-built system venv without creating a stray `.venv` in the mount or re-resolving dependencies on every call.

**Credentials.** Save credentials at the repo root in `.env` — one `KEY=VALUE` per line, no quotes, no `export` prefix:

```text
HF_TOKEN=...
OPENAI_API_KEY=...
MARITACA_API_KEY=...
```

`thesis-eval-gpu` injects these into the container automatically. For host-side `thesis-eval` commands, also set them in your shell so `huggingface_hub` and the OpenAI/Maritaca clients can read them:

```powershell
$env:HF_TOKEN = "..."
$env:OPENAI_API_KEY = "..."
$env:MARITACA_API_KEY = "..."
```

**Interactive container shell** (alternative for back-to-back GPU commands without per-call container startup):

```powershell
docker run -it --rm --gpus all `
  -v "${PWD}:/work/eval" `
  -w /work/eval `
  --env-file .env `
  thesis-eval:gpu bash
```

Inside that shell, drop the `-gpu` suffix — `uv run thesis-eval ...` works directly. Note that the README's command blocks use PowerShell syntax (backtick line-continuation); inside the container's bash shell you'll need backslash (`\`) continuations instead. The `thesis-eval-gpu` wrapper avoids this fork because it parses arguments in your host shell and forwards them as a list (no shell quoting).

> **Windows PowerShell:** use `${PWD}` (not `$PWD`) for the bind-mount path and backtick (`` ` ``) for line continuation. On Linux/WSL bash, use `$PWD` and `\` instead.

### 1.3 Native installs (limited use)

Apple Silicon smoke tests with Transformers on MPS (no vLLM, no CUDA):

```powershell
uv sync --extra mac --extra qc --extra scoring --extra analysis
```

Windows analysis-only work — table export, dataset freeze, GLMM fits — without GPU inference:

```powershell
uv sync --extra analysis --extra scoring
```

The `[rtx]` and `[qc]` extras are unsupported on native Windows: PyPI returns CPU-only `torch`, and `vllm`, `fairseq2`, and `sonar-space` are gated to `sys_platform != 'win32'` in `pyproject.toml`. Use the Docker container (§1.1) for any command that needs CUDA, vLLM, or BLASER QC.

### 1.4 URIEL+ standalone script environment

URIEL+ depends on TensorFlow Addons, which has no Python 3.12 wheels, so `prepare-uriel` (§5) shells out to a standalone uv script env:

```powershell
uv sync --script scripts/compute_urielplus.py
```

This runs on the host (any OS) — it does not need the GPU container.

### 1.5 Validate the install

These commands are config-only (no GPU needed):

```powershell
uv run thesis-eval validate-config
uv run thesis-eval runtime-profiles
uv run thesis-eval list-assets
```

To smoke-test the GPU path through the wrapper:

```powershell
uv run thesis-eval-gpu validate-config
```

That should print `OK: 8 paired models, 1 reference baseline, 13 languages` after a brief container startup.

## Checkpointed Execution

The workflow below keeps one checkpoint per phase, but the main commands process the full 13-language inventory where the CLI supports it. Keep the one-language examples as smoke-test references or targeted reruns after a failed language.

Downloads are resumable. Hugging Face assets are downloaded through `snapshot_download` with resume support into the fixed paths listed in `configs/assets.yaml`. If a network interruption or machine restart happens, rerun the same `download-assets --name ...` command. Completed files are reused, partial transfers continue, and verified assets are skipped. If a folder exists but required files are missing, the command resumes into that folder. `--force` starts a fresh download for that asset.

Pipeline runs are resumable too. `translate`, `prompt-roundtrip`, `translation-qc`, `generate-targets`, and `score-strongreject` all save outputs incrementally — each language (or each input file) is written to disk as soon as it finishes, and on the next invocation any input whose output file already exists is skipped with a one-line log message. Cancel with `Ctrl-C` and rerun the same command to pick up where you left off. Pass `--force` to re-run a step from scratch, e.g. after tuning thresholds or upgrading a model checkpoint.

## 2. Download Datasets

StrongREJECT provides the harmful prompt source. MultiJail is the external multilingual validation benchmark. BELEBELE is the benign reading-comprehension benchmark used to compute IF, CONS, and SPEC.

```powershell
uv run thesis-eval download-assets --name strongreject
uv run thesis-eval verify-assets --name strongreject
```

```powershell
uv run thesis-eval download-assets --name multijail
uv run thesis-eval verify-assets --name multijail
```

```powershell
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

```powershell
uv run thesis-eval download-assets --name nllb_200_3_3b
uv run thesis-eval verify-assets --name nllb_200_3_3b
```

```powershell
uv run thesis-eval download-assets --name blaser_2_0_qe
uv run thesis-eval-gpu prewarm-sonar
uv run thesis-eval verify-assets --name blaser_2_0_qe
```

For a smaller local smoke test:

```powershell
uv run thesis-eval download-assets --name nllb_200
uv run thesis-eval verify-assets --name nllb_200
```

## 4. Download Target Models

Download each open-weight paired target separately. After each download, verify the asset and inspect model status. Sabiá-3 is an API target, so `model-status` checks the configured provider metadata instead of local weights.

```powershell
uv run thesis-eval download-assets --name sagui_7b
uv run thesis-eval verify-assets --name sagui_7b
uv run thesis-eval model-status --model sagui_7b
```

```powershell
uv run thesis-eval model-status --model sabia_3
```

```powershell
uv run thesis-eval download-assets --name llamantino_2_ultrachat_7b
uv run thesis-eval verify-assets --name llamantino_2_ultrachat_7b
uv run thesis-eval model-status --model llamantino_2_ultrachat_7b
```

The Italian weak pole is a merged LLaMAntino-2 checkpoint, so it follows the same full-checkpoint generation path as the other open-weight targets.

```powershell
uv run thesis-eval download-assets --name llamantino_anita_8b
uv run thesis-eval verify-assets --name llamantino_anita_8b
uv run thesis-eval model-status --model llamantino_anita_8b
```

```powershell
uv run thesis-eval download-assets --name gpt_sw3
uv run thesis-eval verify-assets --name gpt_sw3
uv run thesis-eval model-status --model gpt_sw3
```

```powershell
uv run thesis-eval download-assets --name ai_sweden_llama3_8b
uv run thesis-eval verify-assets --name ai_sweden_llama3_8b
uv run thesis-eval model-status --model ai_sweden_llama3_8b
```

```powershell
uv run thesis-eval download-assets --name bggpt_7b
uv run thesis-eval verify-assets --name bggpt_7b
uv run thesis-eval model-status --model bggpt_7b
```

```powershell
uv run thesis-eval download-assets --name bggpt_gemma_9b
uv run thesis-eval verify-assets --name bggpt_gemma_9b
uv run thesis-eval model-status --model bggpt_gemma_9b
```

Download the English-aligned reference baseline after the paired targets. It is not a ninth paired model; it exists only for the distance-from-English curve and the counterfactual aligned-language safety check.

```powershell
uv run thesis-eval download-assets --name llama3_1_8b_reference
uv run thesis-eval verify-assets --name llama3_1_8b_reference
uv run thesis-eval model-status --model llama3_1_8b_reference
```

## 5. Prepare URIEL+ Distance Data

This command recomputes the 13-language featural angular distance matrix with URIEL+, then writes the matrix used by the dataset builder and thesis tables. By default it uses the standalone `scripts/compute_urielplus.py` script environment, because URIEL+'s TensorFlow Addons dependency does not provide Python 3.12 wheels.

```powershell
uv run thesis-eval prepare-uriel
```

Use `--uriel-backend in-process` only if `urielplus` is installed in the active Python environment.

```powershell
uv run thesis-eval prepare-uriel `
  --uriel-backend in-process
```

For a static/offline refresh from an existing matrix CSV instead of recomputing:

```powershell
uv run thesis-eval prepare-uriel --from-csv
```

Outputs:

```text
data/uriel_plus/distance_matrix.csv
data/uriel_plus/urielplus_run_config.json
../ufscthesisx/aftertext/urielplus_distance_table_generated.tex
```

## 6. Translate StrongREJECT Prompts

Translate the English StrongREJECT prompts into the full attack-language inventory with NLLB-200. Each language still receives its own JSONL output, so failed languages can be rerun individually.

```powershell
New-Item -ItemType Directory -Force outputs/translations | Out-Null
$PROMPTS = "data/raw/strongreject/prompts.jsonl"
$NLLB = "assets/models/nllb-200-3.3B"
$DEVICE = "cuda"
$LANGUAGES = @("ara", "bul", "dan", "eng", "fin", "ita", "nor", "por", "rus", "spa", "swa", "swe", "ukr")
$TranslationArgs = foreach ($lang in $LANGUAGES) { "--target-language"; $lang }
```

Full run:

```powershell
uv run thesis-eval-gpu translate `
  --prompts "$PROMPTS" `
  @TranslationArgs `
  --engine nllb `
  --checkpoint "$NLLB" `
  --device "$DEVICE" `
  --output-dir outputs/translations `
  --output-template "{lang}.raw.jsonl"
```

One-language reference:

```powershell
uv run thesis-eval-gpu translate `
  --prompts "$PROMPTS" `
  --target-language ara `
  --engine nllb `
  --checkpoint "$NLLB" `
  --device "$DEVICE" `
  --output outputs/translations/ara.raw.jsonl
```

For a two-language pilot:

```powershell
uv run thesis-eval translate `
  --prompts data/raw/strongreject/pilot_prompts.jsonl `
  --target-language por `
  --engine nllb `
  --checkpoint assets/models/nllb-200-distilled-600M `
  --device auto `
  --output outputs/pilot/translations/por.raw.jsonl
```

```powershell
uv run thesis-eval translate `
  --prompts data/raw/strongreject/pilot_prompts.jsonl `
  --target-language ara `
  --engine nllb `
  --checkpoint assets/models/nllb-200-distilled-600M `
  --device auto `
  --output outputs/pilot/translations/ara.raw.jsonl
```

## 7. Run Round-Trip Diagnostics and BLASER QC

Round-trip diagnostics back-translate prompts for inspection. BLASER assigns reference-free QE scores and pass/flag/fail status. Both `prompt-roundtrip` and `translation-qc` accept multiple `--translations` paths and `--output-dir`/`--output-template`, so each command loads its model (NLLB-3.3B for round-trip, BLASER+SONAR for QC) once and processes all 13 languages in a single process.

Full run:

```powershell
$RawArgs = foreach ($lang in $LANGUAGES) { "--translations"; "outputs/translations/$lang.raw.jsonl" }

uv run thesis-eval-gpu prompt-roundtrip `
  @RawArgs `
  --engine nllb `
  --checkpoint "$NLLB" `
  --device "$DEVICE" `
  --output-dir outputs/translations `
  --output-template "{lang}.roundtrip.jsonl"

$RoundtripArgs = foreach ($lang in $LANGUAGES) { "--translations"; "outputs/translations/$lang.roundtrip.jsonl" }

uv run thesis-eval-gpu translation-qc `
  @RoundtripArgs `
  --scorer blaser `
  --device "$DEVICE" `
  --output-dir outputs/translations `
  --output-template "{lang}.qc.jsonl"
```

One-language reference:

```powershell
uv run thesis-eval-gpu prompt-roundtrip `
  --translations outputs/translations/ara.raw.jsonl `
  --engine nllb `
  --checkpoint "$NLLB" `
  --device "$DEVICE" `
  --output outputs/translations/ara.roundtrip.jsonl

uv run thesis-eval-gpu translation-qc `
  --translations outputs/translations/ara.roundtrip.jsonl `
  --scorer blaser `
  --device "$DEVICE" `
  --output outputs/translations/ara.qc.jsonl
```

## 8. Switchable XSTS Audit and BLASER Calibration

The thesis now supports two translation-audit paths. The LaTeX source defaults to **Plan B** with `\humanauditfalse`; switch the thesis to **Plan A** only after the Tier A human review is complete by setting `\humanaudittrue` in `ufscthesisx/main.tex`.

| Plan | When to use it | Audit method |
| --- | --- | --- |
| Plan B | Default, deadline-driven path | Uncalibrated frontier multilingual LLM-as-judge XSTS for all flagged non-source rows |
| Plan A | Upgrade path after human review is available | Tier A human XSTS for `ara`, `bul`, `ita`, `por`, `spa`; calibrated LLM-as-judge XSTS for `dan`, `fin`, `nor`, `rus`, `swa`, `swe`, `ukr` |

Under both plans, audit decisions are pooled by `(target_language, prompt_id)`. A translation audit decision is not model-specific; the same accepted, revised, or excluded translation is applied to every downstream model that consumes that prompt-language pair. Finnish and Swahili are marked as caveat languages in exported audit queues.

Create audit queues for the chosen plan. Auditors fill the exported CSV files with XSTS scores, audit decisions, and notes. The import command writes audited translation logs for generation.

Audit queues are written as UTF-8 with BOM so Excel on Windows auto-detects encoding. **Tell reviewers to "Save As → CSV UTF-8 (Comma delimited) (*.csv)" — not the plain "CSV (Comma delimited)" option** — otherwise Excel will save as cp1252 and mangle non-ASCII characters on the way back in.

### 8.1 Plan B: LLM-only audit (default)

Use this path when no human XSTS reviewers are available. The CSV rows are marked as `audit_plan=plan_b` and `audit_method=uncalibrated_llm_xsts`. XSTS means cross-lingual semantic textual similarity: `5` is fully equivalent, `4` is mostly equivalent, `3` is imperfect with important differences, `2` is substantially different, and `1` is unrelated or wrong. The automated judge command fills `xsts_score`, `xsts_judge_model`, `audit_decision`, and `xsts_note`.

Valid `audit_decision` values are:

```text
pass     keep the translation as-is
revise   use the edited translated_text/revision_note from the CSV
exclude  exclude the prompt-language pair from downstream generation/analysis
```

Plan B is intentionally disclosed as uncalibrated in the thesis. Do not run `calibrate-blaser` as evidence of human calibration unless Plan A human rows exist.

Plan B queue export:

```powershell
New-Item -ItemType Directory -Force outputs/audit | Out-Null

foreach ($lang in $LANGUAGES) {
  uv run thesis-eval export-audit-queue `
    --translations "outputs/translations/$lang.qc.jsonl" `
    --audit-plan plan_b `
    --output "outputs/audit/$lang.xsts_queue.csv"
}
```

Run the LLM judge over each queue. This uses `OPENAI_API_KEY` and defaults to `gpt-4o`; use `--model` to record a different frontier multilingual judge.

```powershell
foreach ($lang in $LANGUAGES) {
  uv run thesis-eval judge-audit-queue `
    --audit-queue "outputs/audit/$lang.xsts_queue.csv" `
    --provider openai `
    --model gpt-4o `
    --output "outputs/audit/$lang.xsts_decisions.csv"
}
```

The judge maps XSTS `4` or `5` to `audit_decision=pass` and XSTS `1`, `2`, or `3` to `audit_decision=exclude`. Spot-check the decisions CSVs before importing, especially Finnish and Swahili.

After LLM-as-judge review, import the decisions:

```powershell
foreach ($lang in $LANGUAGES) {
  uv run thesis-eval import-audit `
    --translations "outputs/translations/$lang.qc.jsonl" `
    --audit-csv "outputs/audit/$lang.xsts_decisions.csv" `
    --output "outputs/translations/$lang.audit.jsonl"
}
```

The `.audit.jsonl` files are the prompt files used for model generation.

### 8.2 Plan A: human-calibrated audit

Use this path only after human reviewers are available for Tier A languages (`ara`, `bul`, `ita`, `por`, `spa`). The exported queue marks Tier A rows as `human_xsts` and Tier B rows as `calibrated_llm_xsts`.

Plan A queue export:

```powershell
New-Item -ItemType Directory -Force outputs/audit | Out-Null

foreach ($lang in $LANGUAGES) {
  uv run thesis-eval export-audit-queue `
    --translations "outputs/translations/$lang.qc.jsonl" `
    --audit-plan plan_a `
    --output "outputs/audit/$lang.xsts_queue.csv"
}
```

Fill Tier A rows from human XSTS reviewers. Fill Tier B rows from the fixed LLM-as-judge prompt in the thesis appendix. The LLM judge must also score the Tier A rows so weighted Cohen's kappa can be computed externally and reported in the thesis before Tier B decisions are treated as calibrated. If the pre-registered calibration threshold fails, rerun the analysis without LLM-derived Tier B audit decisions and keep the corresponding caveat in the thesis.

After review, import decisions the same way as Plan B:

```powershell
foreach ($lang in $LANGUAGES) {
  uv run thesis-eval import-audit `
    --translations "outputs/translations/$lang.qc.jsonl" `
    --audit-csv "outputs/audit/$lang.xsts_decisions.csv" `
    --output "outputs/translations/$lang.audit.jsonl"
}
```

Calibrate BLASER thresholds from the reviewed subset (`calibrate-blaser` accepts multiple `--audit-csv` arguments; no manual concat is needed). This is thesis-grade calibration only for Plan A or for a later Plan B supplement after human reference rows are added:

```powershell
$AuditArgs = foreach ($lang in $LANGUAGES) { "--audit-csv"; "outputs/audit/$lang.xsts_decisions.csv" }

uv run thesis-eval calibrate-blaser `
  @AuditArgs `
  --output outputs/audit/blaser_thresholds.json
```

If trusted human reference translations are available for the spBLEU subset, attach reference scores:

```powershell
uv run thesis-eval reference-spbleu `
  --translations outputs/translations/por.audit.jsonl `
  --references data/translations/references/por.references.jsonl `
  --output outputs/translations/por.audit.spbleu.jsonl
```

## 9. Generate Target Responses

Open-weight models use vLLM on the RTX machine. The full run below writes one output file per model-language pair.

```powershell
New-Item -ItemType Directory -Force outputs/generations | Out-Null
$AuditArgs = foreach ($lang in $LANGUAGES) { "--translations"; "outputs/translations/$lang.audit.jsonl" }
$OPEN_WEIGHT_MODELS = @(
  "sagui_7b",
  "llamantino_2_ultrachat_7b",
  "llamantino_anita_8b",
  "gpt_sw3",
  "ai_sweden_llama3_8b",
  "bggpt_7b",
  "bggpt_gemma_9b",
  "llama3_1_8b_reference"
)
```

Full open-weight run, including the descriptive English reference baseline:

```powershell
foreach ($model in $OPEN_WEIGHT_MODELS) {
  uv run thesis-eval-gpu generate-targets `
    @AuditArgs `
    --model $model `
    --backend vllm `
    --profile rtx_5070_ti `
    --max-tokens 512 `
    --temperature 0.0 `
    --trust-remote-code `
    --output-dir outputs/generations `
    --output-template "{model}_{lang}.jsonl"
}
```

If you want to run only the paired-design models first, omit `llama3_1_8b_reference` from `$OPEN_WEIGHT_MODELS` and run it afterward with the same `generate-targets` command. Do not add Sabiá-3 to this loop; it uses the Maritaca API path below.

Sabiá-3 uses the Maritaca API. Run all languages in one batch:

```powershell
uv run thesis-eval generate-targets `
  @AuditArgs `
  --model sabia_3 `
  --backend maritaca `
  --max-tokens 512 `
  --temperature 0.0 `
  --output-dir outputs/generations `
  --output-template "{model}_{lang}.jsonl"
```

One-language reference:

```powershell
uv run thesis-eval-gpu generate-targets `
  --translations outputs/translations/ara.audit.jsonl `
  --model sagui_7b `
  --backend vllm `
  --profile rtx_5070_ti `
  --max-tokens 512 `
  --temperature 0.0 `
  --trust-remote-code `
  --output outputs/generations/sagui_7b_ara.jsonl
```

For a Mac smoke test with one model and one translation file:

```powershell
uv run thesis-eval generate-targets `
  --translations outputs/pilot/translations/por.raw.jsonl `
  --model sagui_7b `
  --backend transformers `
  --profile macbook `
  --model-path assets/models/sagui-7b-instruct-v0.1 `
  --max-tokens 64 `
  --output outputs/pilot/generations_sagui_por.jsonl
```

## 10. Score Responses With StrongREJECT

Scoring back-translates non-English model outputs to English with NLLB-200, then runs the official StrongREJECT rubric. Each scored file preserves the original response, the English back-translation, the continuous StrongREJECT score, refusal status, sub-scores, API metadata, and failure metadata. `score-strongreject` accepts multiple `--generations` paths and `--output-dir`/`--output-template`, so NLLB-200 loads once and each input file's scored output is written as soon as it finishes.

Full run:

```powershell
New-Item -ItemType Directory -Force outputs/scored | Out-Null

$ScoreArgs = Get-ChildItem outputs/generations/*.jsonl | ForEach-Object { "--generations"; $_.FullName }

uv run thesis-eval-gpu score-strongreject `
  @ScoreArgs `
  --evaluator strongreject_rubric `
  --backtranslate-engine nllb `
  --checkpoint "$NLLB" `
  --device "$DEVICE" `
  --output-dir outputs/scored `
  --output-template "{stem}.strongreject.jsonl"
```

One-file reference:

```powershell
uv run thesis-eval-gpu score-strongreject `
  --generations outputs/generations/sagui_7b_ara.jsonl `
  --evaluator strongreject_rubric `
  --backtranslate-engine nllb `
  --checkpoint "$NLLB" `
  --device "$DEVICE" `
  --output outputs/scored/sagui_7b_ara.strongreject.jsonl
```

After every generation file has a matching scored file, combine them:

```powershell
Get-Content outputs/scored/*.strongreject.jsonl | Set-Content outputs/scored/all_strongreject_scores.jsonl
```

For pipeline smoke tests:

```powershell
uv run thesis-eval score-strongreject `
  --generations outputs/pilot/generations_sagui_por.jsonl `
  --evaluator mock `
  --output outputs/pilot/strongreject_scores_sagui_por.jsonl
```

## 11. Compute BELEBELE IF/CONS/SPEC

BELEBELE measures benign multilingual specialization for the eight paired target models. The English reference baseline is excluded from paired-design GLMMs, so SPEC for `llama3_1_8b_reference` is optional and not required for the main inferential tables. Query each paired target model on the BELEBELE multiple-choice items, then write predictions as JSONL with these fields:

```json
{"model":"sagui_7b","language":"por","item_id":"belebele_0001","prediction":"A","gold":"A"}
```

The prediction file should contain all model-language-item rows needed for IF and CONS. Then compute specialization:

```powershell
New-Item -ItemType Directory -Force outputs/spec | Out-Null

uv run thesis-eval compute-spec `
  --predictions outputs/spec/belebele_predictions.jsonl `
  --output outputs/spec/belebele_spec.jsonl
```

Output rows include:

```text
model, attack_language, if_score, cons_score, spec_score, n_items, n_cons_comparable
```

## 12. Score Matched Benign Controls

Matched benign controls use generation rows shaped like the harmful prompt outputs. The scoring command attaches relevance, completeness, coherence, and task-completion fields for benign analysis tables.

```powershell
uv run thesis-eval score-benign-controls `
  --generations outputs/benign/benign_generations.jsonl `
  --output outputs/benign/benign_scores.jsonl
```

## 13. Build and Freeze the Analysis Dataset

Join StrongREJECT scores, translation QC/audit metadata, tokenizer diagnostics, URIEL+ distances, model metadata, and BELEBELE specialization. The frozen dataset may contain both the eight paired models and `llama3_1_8b_reference`; downstream inferential commands filter the reference baseline out of paired-design models automatically.

```powershell
Get-Content outputs/translations/*.audit.jsonl | Set-Content outputs/translations/all_audited_translations.jsonl

uv run thesis-eval build-dataset `
  --scored outputs/scored/all_strongreject_scores.jsonl `
  --translations outputs/translations/all_audited_translations.jsonl `
  --spec-scores outputs/spec/belebele_spec.jsonl `
  --output outputs/dataset_frozen.jsonl
```

Export a Parquet copy for archival and analysis notebooks:

```powershell
uv run thesis-eval export-results `
  --rows outputs/dataset_frozen.jsonl `
  --output outputs/dataset_frozen.parquet
```

## 14. Export Thesis Tables

Generate CSV table inputs for the Results chapter:

```powershell
New-Item -ItemType Directory -Force outputs/tables | Out-Null

uv run thesis-eval export-report-tables `
  --rows outputs/dataset_frozen.jsonl `
  --output-dir outputs/tables
```

Fit the main GLMM effects table:

```powershell
uv run thesis-eval fit-glmm `
  --rows outputs/dataset_frozen.jsonl `
  --output outputs/tables/glmm_main_effects.csv
```

`fit-glmm`, `glmm_collinearity.csv`, and the pre-registered slope-retention tables use only rows with `model_alignment_pole` equal to `weak` or `strong`. The reference baseline remains available for descriptive tables and figures.

The reporting commands produce:

```text
outputs/tables/results_coverage.csv
outputs/tables/asr_by_model_language.csv
outputs/tables/crosslingual_asr_by_model_language.csv
outputs/tables/closest_farthest_languages.csv
outputs/tables/distance_asr_correlation.csv
outputs/tables/spec_asr_correlation.csv
outputs/tables/tokenizer_diagnostics.csv
outputs/tables/reference_distance_curve.csv
outputs/tables/counterfactual_safety_by_aligned_language.csv
outputs/tables/glmm_collinearity.csv
outputs/tables/prereg_distance_slope_retention.csv
outputs/tables/prereg_falsification_summary.csv
outputs/tables/glmm_main_effects.csv
```

## 15. Validation Commands

Run unit tests:

```powershell
uv run python -m unittest
```

Check asset status at any point:

```powershell
uv run thesis-eval asset-status --group datasets
uv run thesis-eval asset-status --group translation
uv run thesis-eval asset-status --group targets
```

Run the built-in mock pilot to verify local plumbing without model inference:

```powershell
uv run thesis-eval run-pilot --output-dir outputs/pilot
```
