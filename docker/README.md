# GPU Docker Profile

This profile targets the RTX 5070 Ti / Blackwell path with CUDA 12.8 and vLLM.

Build from `eval/`:

```bash
docker build -f docker/Dockerfile.gpu -t thesis-eval:gpu .
```

Run with the workspace mounted:

```bash
docker run --rm --gpus all \
  -v "$PWD":/work/eval \
  -w /work/eval \
  --env-file .env \
  thesis-eval:gpu thesis-eval validate-config
```

Smoke checks:

```bash
docker run --rm --gpus all thesis-eval:gpu nvidia-smi
docker run --rm --gpus all thesis-eval:gpu python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
