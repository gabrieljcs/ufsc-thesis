# RunPod runbook — rented-GPU path for pending BELEBELE and §9 generation

This is the active runbook for finishing the remaining thesis-eval GPU work on
RunPod. It replaces the TPU execution plan and the heavy §9 generation portion
of the old RTX runbook. The RTX rig still owns the lighter local steps,
especially §10 StrongREJECT scoring, unless you explicitly decide to score on
RunPod before teardown.

RunPod references:

- Pods and connection options: https://docs.runpod.io/pods/connect-to-a-pod
- SSH setup: https://docs.runpod.io/pods/configuration/use-ssh
- Network volumes: https://docs.runpod.io/storage/network-volumes
- `runpodctl` overview: https://docs.runpod.io/runpodctl/overview

## What runs here

Run these steps on a single rented CUDA pod, preferably an A100 80 GB or H100
80 GB:

- `predict-belebele` for the pending open-weight models.
- `generate-targets` for §9 StrongREJECT generations over 13 attack languages.

After this runbook, copy the new `outputs/spec/` and `outputs/generations/`
artifacts to the RTX rig and run [`RTX_RUN.md`](RTX_RUN.md) for §10 scoring.

Already-completed or API-served models should be copied into the RunPod
workspace for file-count checks, but should not be regenerated:

- `sagui_7b` — already completed locally.
- `llamantino_2_ultrachat_7b` — already completed locally.
- `sabia_3` — API-served through Maritaca/Sabiá-3; keep provider metadata.

Pending open-weight models:

```text
llamantino_anita_8b        Italian strong
ai_sweden_llama3_8b        Swedish strong
gpt_sw3                    Swedish weak
bggpt_gemma_9b             Bulgarian strong
bggpt_7b                   Bulgarian weak
llama3_1_8b_reference      optional English reference baseline
```

## Pod sizing

| GPU | Fit | Recommendation |
| --- | --- | --- |
| A100 80 GB | All pending models in fp16 with comfortable KV cache | Recommended default. |
| H100 80 GB | Same, faster | Use if the price/capacity is good. |
| L40S 48 GB | Most 7B/8B jobs; tighter for Gemma 9B and long outputs | Viable fallback, but expect tuning. |
| RTX 4090 24 GB | 7B/8B smoke tests only | Not recommended for the full pending run. |

Use one pod sequentially unless wall-clock time matters more than cost. The
commands are checkpointed by output file, so reruns skip completed model/language
files unless `--force` is passed.

## 0. Final TPU cleanup on the laptop

Before starting RunPod, pull any partial TPU outputs back so they can be reused.
Skip this section if no TPU VMs were ever created.

```bash
cd ~/Developer/ufsc-thesis
for n in 0 1 2 3; do
  gcloud compute tpus tpu-vm scp --recurse \
    --project=mestrado-495311 --zone=europe-west4-a \
    "thesis-belebele-$n":~/ufsc-thesis/eval/outputs/spec ./eval/outputs/ 2>/dev/null
  gcloud compute tpus tpu-vm scp --recurse \
    --project=mestrado-495311 --zone=europe-west4-a \
    "thesis-belebele-$n":~/ufsc-thesis/eval/outputs/generations ./eval/outputs/ 2>/dev/null
done

cd ~/Developer/ufsc-thesis/eval/runbooks/tpu-terraform
terraform destroy
```

## 1. Provision the RunPod pod

Use an official RunPod PyTorch template or your own image built from
`eval/docker/Dockerfile.gpu`. The official PyTorch template is usually simpler:
the bootstrap below installs this project and its vLLM/CUDA dependencies inside
the pod.

Recommended settings:

| Field | Value |
| --- | --- |
| GPU | 1 x A100 80 GB or H100 80 GB |
| Container disk | 80 GB minimum |
| Persistent volume | 150-250 GB mounted at `/workspace` |
| Network volume | Optional but useful if you may recreate pods |
| Ports | SSH enabled; Jupyter/HTTP optional |
| Env vars | `HF_TOKEN`, `OPENAI_API_KEY`, optional `MARITACA_API_KEY`, `HF_HUB_ENABLE_HF_TRANSFER=1` |

Network volumes are available for Pods in Secure Cloud and mount at
`/workspace`. If you do not attach a network volume, terminate carefully only
after pulling outputs back to the laptop.

## 2. Connect and bootstrap

In the RunPod console, open **Connect -> SSH** and copy the SSH command. Use SSH
for long runs; the web terminal is fine only for quick checks.

```bash
# Inside the pod
cd /workspace

# Basic tools. Many RunPod templates already include these, but this is cheap
# and keeps the bootstrap deterministic.
apt-get update
apt-get install -y tmux rsync git curl

# Clone or update the repo.
if [ -d ufsc-thesis/.git ]; then
  cd ufsc-thesis
  git pull --ff-only
else
  git clone https://github.com/gabrieljcs/ufsc-thesis.git
  cd ufsc-thesis
fi
cd eval

# Keep long jobs alive after SSH disconnects. Attach later with:
# tmux attach -t thesis
tmux new -s thesis

# Install uv if the template does not include it.
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
. "$HOME/.local/bin/env" 2>/dev/null || true

# Install thesis-eval with CUDA/vLLM, scoring, and analysis extras.
uv sync --extra rtx --extra scoring --extra analysis --python 3.12

# Confirm CUDA is visible.
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

mkdir -p outputs/translations outputs/generations outputs/spec outputs/scored
```

The CUDA check should print `True` and the pod GPU name.

Create `.env` if the RunPod template did not already inject these variables:

```bash
cat > .env <<'EOF'
HF_TOKEN=<your-hf-token>
OPENAI_API_KEY=<your-openai-key>
MARITACA_API_KEY=<your-maritaca-key-if-needed>
HF_HUB_ENABLE_HF_TRANSFER=1
EOF
set -a && source .env && set +a
```

Move existing local outputs into the pod so reruns skip completed files and
sanity checks see the full panel. If your pod has full SSH/SCP support, push
them from the laptop using the host and port shown in the RunPod Connect panel:

```bash
cd ~/Developer/ufsc-thesis/eval
POD=root@<pod-ip>
PORT=<pod-ssh-port>

rsync -avz -e "ssh -p $PORT" outputs/translations/ "$POD":/workspace/ufsc-thesis/eval/outputs/translations/
rsync -avz -e "ssh -p $PORT" outputs/generations/  "$POD":/workspace/ufsc-thesis/eval/outputs/generations/
rsync -avz -e "ssh -p $PORT" outputs/spec/         "$POD":/workspace/ufsc-thesis/eval/outputs/spec/
```

If the pod only has basic SSH without file transfer, use `runpodctl send` /
`runpodctl receive`, or upload the outputs through a network volume.

Then validate the local checkout and datasets:

```bash
uv run thesis-eval validate-config
uv run thesis-eval runtime-profiles
uv run thesis-eval download-assets --name belebele
uv run thesis-eval verify-assets --name belebele
```

## 3. Shared arguments

Use the dedicated RunPod profile. It resolves to `backend=vllm`, `device=cuda`,
and `dtype=auto`.

```bash
LANGS=(ara bul dan eng fin ita nor por rus spa swa swe ukr)
AUDIT_ARGS=()
for L in "${LANGS[@]}"; do
  AUDIT_ARGS+=(--translations "outputs/translations/$L.audit.jsonl")
done
```

## 4. Per-model loop

Run the pending models sequentially. Keep one tmux pane per active pod. If a
command is interrupted, rerun the same block; completed outputs are skipped.

```bash
MODEL=llamantino_anita_8b
LOCAL_DIR=llamantino-3-anita-8b-inst-dpo-ita

uv run thesis-eval download-assets --name "$MODEL"
uv run thesis-eval verify-assets  --name "$MODEL"

uv run thesis-eval predict-belebele \
  --model "$MODEL" \
  --backend vllm \
  --profile runpod_a100 \
  --output "outputs/spec/${MODEL}.belebele.jsonl"

uv run thesis-eval generate-targets \
  "${AUDIT_ARGS[@]}" \
  --model "$MODEL" \
  --profile runpod_a100 \
  --max-tokens 512 \
  --temperature 0.0 \
  --trust-remote-code \
  --output-dir outputs/generations \
  --output-template "{model}_{lang}.jsonl"

rm -rf "assets/models/$LOCAL_DIR"
```

Repeat with:

| MODEL | LOCAL_DIR | `--trust-remote-code` | Status |
| --- | --- | --- | --- |
| `llamantino_anita_8b` | `llamantino-3-anita-8b-inst-dpo-ita` | yes | done |
| `ai_sweden_llama3_8b` | `ai-sweden-llama-3-8b-instruct` | no | done |
| `gpt_sw3` | `gpt-sw3-6.7b-v2-instruct` | yes | running targets |
| `bggpt_gemma_9b` | `bggpt-gemma-2-9b-it-v1.0` | no | todo |
| `bggpt_7b` | `bggpt-7b-instruct-v0.2` | yes | todo |
| `llama3_1_8b_reference` | `llama-3.1-8b-instruct` | no | todo |

If a model does not require remote code, omit `--trust-remote-code` from the
generation command. Keeping it for a model that does not need it is usually
harmless, but omitting it where possible keeps execution stricter.

## 5. Sanity checks before handoff

Expected file counts:

```bash
find outputs/generations -name '*.jsonl' | wc -l
find outputs/spec -name '*.belebele.jsonl' | wc -l
```

Targets:

- 104 files for the eight paired models: 8 models x 13 languages.
- 117 files if the optional English reference baseline is included.

If counts are low, list missing model/language combinations before handoff:

```bash
uv run thesis-eval model-status
```

## 6. Pull outputs back to the laptop

From the laptop, use the SSH host and port shown in the RunPod Connect panel:

```bash
LAPTOP=~/Developer/ufsc-thesis/eval
POD=root@<pod-ip>
PORT=<pod-ssh-port>

rsync -avz -e "ssh -p $PORT" "$POD":/workspace/ufsc-thesis/eval/outputs/spec/        "$LAPTOP/outputs/spec/"
rsync -avz -e "ssh -p $PORT" "$POD":/workspace/ufsc-thesis/eval/outputs/generations/ "$LAPTOP/outputs/generations/"
```

If SSH transfer is awkward, use `runpodctl send` inside the pod and
`runpodctl receive` on the laptop or another pod.

Then copy `outputs/generations/` and, if needed, `outputs/spec/` to the RTX rig
and follow [`RTX_RUN.md`](RTX_RUN.md) for scoring.

## 7. Optional RunPod scoring fallback

Prefer the RTX rig for §10 scoring. If you cannot use the RTX rig and want to
finish scoring before terminating the pod, run the scoring block from
[`RTX_RUN.md`](RTX_RUN.md) directly inside the pod with plain
`uv run thesis-eval` instead of `uv run thesis-eval-gpu`.

## 8. Tear down

After the generation and BELEBELE files are verified on the laptop or RTX rig,
terminate the pod in the RunPod console. Stopping keeps attached storage
billable; termination is the clean cost cutoff when no network volume needs to
be preserved.

If using a network volume, keep or delete the volume intentionally after
confirming which artifacts it still holds.

## 9. Reconverge on the laptop

Return to the laptop and run the final CPU-side workflow in
[`RECONVERGE.md`](RECONVERGE.md). That runbook handles the per-file BELEBELE
repair pass, SPEC aggregation, frozen dataset build, table export, and GLMM
fits.

```bash
cd ~/Developer/ufsc-thesis/eval
# Then follow eval/runbooks/RECONVERGE.md from §1 onward.
```
