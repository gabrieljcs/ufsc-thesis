# GPU Docker Profile

This profile is the canonical GPU runtime for `thesis-eval`. The image targets the RTX 5070 Ti / Blackwell path with CUDA 12.8 and vLLM, and is the recommended runtime for translation, BLASER QC, target generation, and StrongREJECT scoring on any host (Windows + WSL 2, Linux, or macOS with NVIDIA GPU passthrough).

For the user-facing setup walk-through, see §1 of [`../README.md`](../README.md). This file documents image internals and reference invocations.

## Image contents

`Dockerfile.gpu` builds on `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` and installs:

- Python 3.12 system venv at `/opt/venv` (on `PATH`).
- `uv` and `pip` (latest at build time).
- `thesis-eval` in editable mode via `uv pip install -e .`.
- `torch` from the cu128 index (Blackwell-compatible, sm_120).
- `vllm`, `transformers`, `accelerate`, `sentencepiece`, `sacrebleu`, `fairseq2`, `sonar-space>=0.5,<0.6`.

Two environment variables are set on the image:

- `UV_PROJECT_ENVIRONMENT=/opt/venv` — `uv run` reuses the system venv instead of creating a new `.venv` under the bind-mounted host repo.
- `UV_NO_SYNC=1` — `uv run` does not re-resolve and re-sync project dependencies on every invocation. This keeps the cu128 `torch` wheel (and the rest of the curated install) intact across calls and makes `uv run thesis-eval ...` start in milliseconds instead of seconds.

## Prerequisites

- NVIDIA GPU and recent driver (the CUDA 12.8 runtime is bundled in the image; only the host driver matters).
- Docker Desktop with the WSL 2 backend and GPU passthrough on Windows, or Docker Engine + NVIDIA Container Toolkit on Linux.
- ~30 GB free disk: roughly 12 GB for the image and the rest for downloaded model assets under `assets/models/`.

## Build

From the repo root:

```bash
docker build -f docker/Dockerfile.gpu -t thesis-eval:gpu .
```

PowerShell uses the same command (no syntax differences for `docker build`).

## Smoke checks

Confirm GPU passthrough and that `torch` was built with CUDA:

```bash
docker run --rm --gpus all thesis-eval:gpu nvidia-smi
docker run --rm --gpus all thesis-eval:gpu python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output for the second command:

```text
True NVIDIA GeForce RTX 5070 Ti
```

(Replace with your GPU name.) If `cuda.is_available()` is `False`, the host driver is missing or Docker Desktop's GPU support is not enabled.

## Persistence and the bind mount

The `-v "$PWD":/work/eval` flag bind-mounts your host repo as the container's working directory. **Everything `thesis-eval` writes is written there**, so it lands on your host disk and survives `--rm`, image rebuilds, machine restarts, and `docker rmi thesis-eval:gpu`. This includes:

- `assets/models/` — downloaded Hugging Face model snapshots (NLLB, BLASER, target models)
- `data/raw/`, `data/translations/`, `data/uriel_plus/` — datasets and reference inputs
- `outputs/translations/`, `outputs/generations/`, `outputs/scored/`, `outputs/spec/`, `outputs/tables/`, `outputs/dataset_frozen.*` — every artifact produced by the pipeline

Everything outside the mount is ephemeral: the system venv at `/opt/venv`, `/tmp`, `/root/.cache/`, shell state. The container is disposable; only the bind mount is durable.

Run a long download or generation once, exit the container, restart later, and continue — the host repo is the source of truth. Per-language outputs and `download-assets` (which uses `snapshot_download` with resume support) are both incremental, so a killed run resumes cleanly on the next invocation.

## Run thesis-eval commands

The recommended entry point is the `thesis-eval-gpu` console script (registered alongside `thesis-eval` in `pyproject.toml`, implementation at [`../src/thesis_eval/gpu_runner.py`](../src/thesis_eval/gpu_runner.py)). It runs from your host shell — PowerShell, bash, or zsh — and forwards arguments into this image with the right `docker run` flags:

```powershell
uv run thesis-eval-gpu validate-config
uv run thesis-eval-gpu translate `
  --prompts data/raw/strongreject/prompts.jsonl `
  --target-language por `
  --engine nllb `
  --checkpoint assets/models/nllb-200-3.3B `
  --device cuda `
  --output outputs/translations/por.raw.jsonl
```

The wrapper does the equivalent of `docker run --rm --gpus all -v "$PWD":/work/eval -w /work/eval --env-file .env thesis-eval:gpu uv run thesis-eval ...`. Argument forwarding happens at the `argv` level via `subprocess`, so PowerShell line-continuations (`` ` ``) and bash continuations (`\`) both work without translation. `.env` is auto-detected. Override the image with `THESIS_EVAL_GPU_IMAGE`.

The lower-level patterns below are provided for cases where you don't want the wrapper (e.g., on hosts without a thesis-eval install, debugging, CI).

### Interactive shell (alternative for back-to-back GPU commands)

bash:

```bash
docker run -it --rm --gpus all \
  -v "$PWD":/work/eval \
  -w /work/eval \
  --env-file .env \
  thesis-eval:gpu bash
```

PowerShell:

```powershell
docker run -it --rm --gpus all `
  -v "${PWD}:/work/eval" `
  -w /work/eval `
  --env-file .env `
  thesis-eval:gpu bash
```

From the resulting shell, run any README command directly. Example translate run:

```bash
uv run thesis-eval validate-config
uv run thesis-eval download-assets --name nllb_200_3_3b
uv run thesis-eval translate \
  --prompts data/raw/strongreject/prompts.jsonl \
  --target-language por \
  --engine nllb \
  --checkpoint assets/models/nllb-200-3.3B \
  --device cuda \
  --output outputs/translations/por.raw.jsonl
```

### One-shot

For a single command without an interactive shell:

bash:

```bash
docker run --rm --gpus all \
  -v "$PWD":/work/eval \
  -w /work/eval \
  --env-file .env \
  thesis-eval:gpu \
  uv run thesis-eval validate-config
```

PowerShell:

```powershell
docker run --rm --gpus all `
  -v "${PWD}:/work/eval" `
  -w /work/eval `
  --env-file .env `
  thesis-eval:gpu `
  uv run thesis-eval validate-config
```

## Credentials

Save credentials at the repo root in `.env`. The `--env-file` flag injects them into the container — one `KEY=VALUE` per line, no quotes, no `export` prefix:

```text
HF_TOKEN=...
OPENAI_API_KEY=...
MARITACA_API_KEY=...
```

## Troubleshooting

- **`could not select device driver "" with capabilities: [[gpu]]`** — Docker can't see the GPU. On Windows, enable GPU support in Docker Desktop (Settings → Resources → WSL Integration). On Linux, install `nvidia-container-toolkit` and restart the Docker daemon.
- **`torch.cuda.is_available() == False` inside the container** — host driver is too old or absent. Verify with `nvidia-smi` on the host first; it must work outside Docker before it works inside.
- **`OutOfMemoryError` during NLLB-3.3B translation** — the 3.3B checkpoint plus a long prompt batch exceeds VRAM on smaller GPUs. Drop to the 600M distilled checkpoint (`assets/models/nllb-200-distilled-600M`) for development, or reduce `--max-tokens` / batch size for the 3.3B run.
- **Stray `.venv/` appears under your repo after running** — your image predates the `UV_PROJECT_ENVIRONMENT` env var. Rebuild with the current `Dockerfile.gpu` and delete the stray directory.
