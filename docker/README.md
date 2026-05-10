# GPU Docker profile

Local containerized GPU runtime for `thesis-eval`. This image covers light GPU
checks, translation/BLASER smokes, target-generation smokes, and StrongREJECT
scoring on a CUDA host (Windows + WSL 2, Linux, or macOS with NVIDIA GPU
passthrough).

For the user-facing setup, see the install section in [`../README.md`](../README.md).
This file documents image internals.

## Image contents

`Dockerfile.gpu` builds on `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` and
installs:

- Python 3.12 system venv at `/opt/venv` (on `PATH`).
- `uv` and `pip` (latest at build time).
- `thesis-eval` in editable mode via `uv pip install -e .`.
- `torch` from the cu128 index (Blackwell-compatible, sm_120).
- `vllm`, `transformers`, `accelerate`, `sentencepiece`, `sacrebleu`,
  `fairseq2`, `sonar-space>=0.5,<0.6`.

Two image-level environment variables matter:

- `UV_PROJECT_ENVIRONMENT=/opt/venv`. `uv run` reuses the system venv instead
  of creating a `.venv/` under the bind-mounted host repo.
- `UV_NO_SYNC=1`. `uv run` does not re-resolve and re-sync project
  dependencies on every invocation, which keeps the cu128 `torch` wheel
  intact and makes startup near-instant.

## Prerequisites

- NVIDIA GPU and a recent driver. The CUDA 12.8 runtime is bundled in the
  image; only the host driver matters.
- Docker Desktop with the WSL 2 backend and GPU passthrough on Windows, or
  Docker Engine + NVIDIA Container Toolkit on Linux.
- About 30 GB of free disk: 12 GB for the image, the rest for downloaded
  model assets under `assets/models/`.

## Build

From the repo root:

```sh
docker build -f docker/Dockerfile.gpu -t thesis-eval:gpu .
```

PowerShell uses the same command.

## Smoke checks

```sh
docker run --rm --gpus all thesis-eval:gpu nvidia-smi
docker run --rm --gpus all thesis-eval:gpu \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The second command should print `True <GPU name>`. If `cuda.is_available()`
is `False`, the host driver is missing or Docker Desktop's GPU support is
not enabled.

## Bind mount and persistence

The `-v "$PWD":/work/eval` flag bind-mounts the host repo as the container's
working directory. Every artifact `thesis-eval` writes lands on host disk:

- `assets/models/`: Hugging Face model snapshots (NLLB, BLASER, targets).
- `data/raw/`, `data/uriel_plus/`: imported datasets and reference inputs.
- `outputs/translations/`: prompt translation, round-trip, QC, and audit
  artifacts.
- `outputs/`: translations, generations, scored rows, BELEBELE predictions,
  tables, frozen dataset.

Everything outside the mount is ephemeral (system venv, `/tmp`, caches).
A killed run resumes cleanly: `download-assets` uses `snapshot_download`
with resume support, and per-language outputs are written incrementally.

## Running thesis-eval commands

The recommended entry point is `thesis-eval-gpu`, registered in
`pyproject.toml` and implemented in
[`../src/thesis_eval/gpu_runner.py`](../src/thesis_eval/gpu_runner.py). It
runs from the host shell (PowerShell, bash, or zsh) and forwards arguments
into the container with the right `docker run` flags:

```sh
uv run thesis-eval-gpu validate-config
uv run thesis-eval-gpu translate \
  --prompts data/raw/strongreject/prompts.jsonl \
  --target-language por \
  --engine nllb \
  --checkpoint assets/models/nllb-200-3.3B \
  --device cuda \
  --output outputs/translations/por.raw.jsonl
```

The wrapper is equivalent to
`docker run --rm --gpus all -v "$PWD":/work/eval -w /work/eval --env-file .env thesis-eval:gpu uv run thesis-eval ...`.
Argument forwarding goes through `subprocess`, so PowerShell backticks and
bash backslashes both work without translation. `.env` is auto-detected.
Override the image tag with `THESIS_EVAL_GPU_IMAGE`.

The lower-level invocations below are kept for the cases where the wrapper
is not appropriate (hosts without a thesis-eval install, debugging, CI).

### Interactive shell

```sh
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

From inside the shell, run any documented command directly.

### One-shot

```sh
docker run --rm --gpus all \
  -v "$PWD":/work/eval \
  -w /work/eval \
  --env-file .env \
  thesis-eval:gpu \
  uv run thesis-eval validate-config
```

## Credentials

`.env` at the repo root, one `KEY=VALUE` per line, no quotes, no `export`
prefix:

```text
HF_TOKEN=...
OPENAI_API_KEY=...
MARITACA_API_KEY=...
```

`--env-file` injects these into the container.

## Troubleshooting

`could not select device driver "" with capabilities: [[gpu]]`. Docker
cannot see the GPU. On Windows, enable GPU support in Docker Desktop
(Settings, Resources, WSL Integration). On Linux, install
`nvidia-container-toolkit` and restart the Docker daemon.

`torch.cuda.is_available() == False` inside the container. The host driver
is too old or absent. Verify `nvidia-smi` on the host first; it must work
outside Docker before it works inside.

`OutOfMemoryError` during NLLB-3.3B translation. The 3.3B checkpoint plus a
long prompt batch exceeds VRAM on smaller GPUs. Drop to the 600M distilled
checkpoint (`assets/models/nllb-200-distilled-600M`), or reduce
`--max-tokens` and batch size for the 3.3B run.

Stray `.venv/` under the repo after running. The image predates the
`UV_PROJECT_ENVIRONMENT` env var. Rebuild with the current `Dockerfile.gpu`
and delete the stray directory.
