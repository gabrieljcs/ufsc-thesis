# TPU runbook — BELEBELE × 6 + heavy §9 × 4

Copy-paste end-to-end. Each block is self-contained; each command is annotated
only where the why is non-obvious. Defaults: `mestrado-495311` /
`europe-west4-a` / `v6e-1` / `vm_count=4` / spot.

> **Precondition**: the `eval/` repo branch you push must contain the batched
> transformers runner patch (Step 0). Verify locally:
> ```bash
> cd ~/Developer/ufsc-thesis/eval && uv run python -m unittest tests.test_generation_batched -v
> ```
> All 8 tests should pass before you touch a TPU.

---

## 1. Provision the TPU VMs (laptop)

```bash
cd ~/Developer/ufsc-thesis/runbooks/tpu-terraform
terraform init
terraform apply           # confirm with "yes"

# Print the SSH commands and the scp pull commands you'll need later:
terraform output ssh_commands
terraform output scp_pull_outputs_spec_commands
terraform output scp_pull_outputs_generations_commands
```

If you want fewer than 4 VMs (e.g., low quota or debugging): `terraform apply -var "vm_count=1"`.

Provisioning takes ~3 minutes. Each VM gets a 100 GB local SSD and routable
public IP for SSH.

---

## 2. SSH into each VM (laptop, 4 terminal tabs)

The four SSH commands are in `terraform output ssh_commands`. Open 4 terminal
tabs and paste one command each:

```bash
# Tab 0
gcloud compute tpus tpu-vm ssh thesis-belebele-0 --project=mestrado-495311 --zone=europe-west4-a
# Tab 1
gcloud compute tpus tpu-vm ssh thesis-belebele-1 --project=mestrado-495311 --zone=europe-west4-a
# Tab 2
gcloud compute tpus tpu-vm ssh thesis-belebele-2 --project=mestrado-495311 --zone=europe-west4-a
# Tab 3
gcloud compute tpus tpu-vm ssh thesis-belebele-3 --project=mestrado-495311 --zone=europe-west4-a
```

The first SSH may take ~30 s while gcloud propagates your account into the VM.

---

## 3. Bootstrap (paste into EACH VM, one-time)

```bash
# uv (manages Python 3.12; TPU base image ships 3.10)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# gh CLI for cloning the private eval repo
sudo apt-get update -y
sudo apt-get install -y gh
gh auth login --hostname github.com --git-protocol https --web
# ↑ follow the device-flow prompt: copy the one-time code, open the URL on
#   your laptop, paste the code, authorize. Then back in the VM:

gh repo clone gabrieljcs/ufsc-thesis ~/ufsc-thesis
cd ~/ufsc-thesis/eval

# Make sure you're on the branch with the batched runner patch (skip if merged to main)
# git checkout feat/batched-transformers-runner

# Transformers stack via the platform-agnostic mac extra
uv sync --extra mac --python 3.12

# Overlay TPU torch + torch_xla from the libtpu wheel index. This replaces
# the PyPI torch installed by --extra mac with a TPU-compatible build pinned
# to the same 2.8 line. Required for predict-belebele/generate-targets to
# see the TPU.
uv pip install --reinstall \
  'torch~=2.8' 'torch_xla[tpu]~=2.8' \
  -f https://storage.googleapis.com/libtpu-releases/index.html

# Verify torch_xla sees the chip
PJRT_DEVICE=TPU uv run python -c \
  "import torch_xla.core.xla_model as xm; print(xm.get_xla_supported_devices('TPU'))"
# Expected output: ['xla:0']  (or similar — non-empty list).

# Credentials + downloader knobs. Replace <YOUR_HF_TOKEN> with a Hugging Face
# token that has access to the gated meta-llama/Llama-3.1-8B-Instruct repo
# if you plan to run the optional reference baseline; otherwise any read
# token works.
cat > .env <<EOF
HF_TOKEN=<YOUR_HF_TOKEN>
HF_HUB_ENABLE_HF_TRANSFER=1
PJRT_DEVICE=TPU
EOF
set -a && source .env && set +a

# Sanity check + dataset fetch (BELEBELE is small, ~1 GB)
uv run thesis-eval validate-config
uv run thesis-eval download-assets --name belebele
uv run thesis-eval verify-assets --name belebele
```

> **Sanity check (recommended on VM 0 only, ~10 min)**: rerun `sagui_7b`
> BELEBELE on TPU and diff against the existing local result. Skip if you're
> in a hurry; this is just a "is the new batched runner producing the same
> answers as the old single-prompt RTX runs?" gate.
>
> ```bash
> uv run thesis-eval download-assets --name sagui_7b
> uv run thesis-eval verify-assets --name sagui_7b
>
> mkdir -p outputs/spec_validation
> PJRT_DEVICE=TPU uv run thesis-eval predict-belebele \
>   --model sagui_7b \
>   --backend transformers --profile trc_tpu \
>   --gpu-batch-size 32 \
>   --output-dir outputs/spec_validation \
>   --output-template "{model}_{lang}.belebele.jsonl"
>
> # On your laptop, after pulling outputs/spec_validation back:
> #   diff <(jq -c '{item_id, language, prediction}' eval/outputs/spec/sagui_7b.belebele.jsonl | sort) \
> #        <(cat eval/outputs/spec_validation/sagui_7b_*.belebele.jsonl | jq -c '{item_id, language, prediction}' | sort)
> # Both should be identical at temperature=0.
>
> rm -rf assets/models/sagui-7b-instruct-v0.1
> ```

---

## 4. Per-VM model block — paste into the matching tab

Each tab runs ONE block. Tabs 0–3 below are independent; they don't talk to
each other. Each block runs ~6–9 hours; predict-belebele and generate-targets
write per-language `*.partial.jsonl` checkpoints, so a Ctrl-C or spot
preemption is recoverable.

### Tab 0 — `llamantino_anita_8b`

`llamantino_2_ultrachat_7b` BELEBELE was completed locally and repaired via
`repair-belebele-predictions` (see commit `6157c45`); do **not** rerun it
here. Tab 0 only handles its heavy §9 model.

```bash
LANGS=(ara bul dan eng fin ita nor por rus spa swa swe ukr)
AUDIT_ARGS=(); for L in "${LANGS[@]}"; do AUDIT_ARGS+=(--translations "outputs/translations/$L.audit.jsonl"); done

# --- Heavy §9 + BELEBELE for llamantino_anita_8b ---
uv run thesis-eval download-assets --name llamantino_anita_8b
uv run thesis-eval verify-assets  --name llamantino_anita_8b

PJRT_DEVICE=TPU uv run thesis-eval predict-belebele \
  --model llamantino_anita_8b \
  --backend transformers --profile trc_tpu \
  --gpu-batch-size 32 \
  --output-dir outputs/spec \
  --output-template "{model}_{lang}.belebele.jsonl"

PJRT_DEVICE=TPU uv run thesis-eval generate-targets \
  "${AUDIT_ARGS[@]}" \
  --model llamantino_anita_8b \
  --backend transformers --profile trc_tpu \
  --gpu-batch-size 8 \
  --max-tokens 512 --temperature 0.0 --trust-remote-code \
  --output-dir outputs/generations \
  --output-template "{model}_{lang}.jsonl"

rm -rf assets/models/llamantino-3-anita-8b-inst-dpo-ita
echo "Tab 0 done."
```

### Tab 1 — `ai_sweden_llama3_8b` + `gpt_sw3` (BELEBELE only)

```bash
LANGS=(ara bul dan eng fin ita nor por rus spa swa swe ukr)
AUDIT_ARGS=(); for L in "${LANGS[@]}"; do AUDIT_ARGS+=(--translations "outputs/translations/$L.audit.jsonl"); done

# --- Heavy §9 + BELEBELE for ai_sweden_llama3_8b ---
uv run thesis-eval download-assets --name ai_sweden_llama3_8b
uv run thesis-eval verify-assets  --name ai_sweden_llama3_8b

PJRT_DEVICE=TPU uv run thesis-eval predict-belebele \
  --model ai_sweden_llama3_8b \
  --backend transformers --profile trc_tpu \
  --gpu-batch-size 32 \
  --output-dir outputs/spec \
  --output-template "{model}_{lang}.belebele.jsonl"

PJRT_DEVICE=TPU uv run thesis-eval generate-targets \
  "${AUDIT_ARGS[@]}" \
  --model ai_sweden_llama3_8b \
  --backend transformers --profile trc_tpu \
  --gpu-batch-size 8 \
  --max-tokens 512 --temperature 0.0 --trust-remote-code \
  --output-dir outputs/generations \
  --output-template "{model}_{lang}.jsonl"

rm -rf assets/models/ai-sweden-llama-3-8b-instruct

# --- BELEBELE-only for gpt_sw3 ---
uv run thesis-eval download-assets --name gpt_sw3
uv run thesis-eval verify-assets  --name gpt_sw3

PJRT_DEVICE=TPU uv run thesis-eval predict-belebele \
  --model gpt_sw3 \
  --backend transformers --profile trc_tpu \
  --gpu-batch-size 32 --trust-remote-code \
  --output-dir outputs/spec \
  --output-template "{model}_{lang}.belebele.jsonl"

rm -rf assets/models/gpt-sw3-6.7b-v2-instruct
echo "Tab 1 done."
```

### Tab 2 — `bggpt_gemma_9b` + `bggpt_7b` (BELEBELE only)

```bash
LANGS=(ara bul dan eng fin ita nor por rus spa swa swe ukr)
AUDIT_ARGS=(); for L in "${LANGS[@]}"; do AUDIT_ARGS+=(--translations "outputs/translations/$L.audit.jsonl"); done

# --- Heavy §9 + BELEBELE for bggpt_gemma_9b ---
uv run thesis-eval download-assets --name bggpt_gemma_9b
uv run thesis-eval verify-assets  --name bggpt_gemma_9b

PJRT_DEVICE=TPU uv run thesis-eval predict-belebele \
  --model bggpt_gemma_9b \
  --backend transformers --profile trc_tpu \
  --gpu-batch-size 32 \
  --output-dir outputs/spec \
  --output-template "{model}_{lang}.belebele.jsonl"

PJRT_DEVICE=TPU uv run thesis-eval generate-targets \
  "${AUDIT_ARGS[@]}" \
  --model bggpt_gemma_9b \
  --backend transformers --profile trc_tpu \
  --gpu-batch-size 8 \
  --max-tokens 512 --temperature 0.0 --trust-remote-code \
  --output-dir outputs/generations \
  --output-template "{model}_{lang}.jsonl"

rm -rf assets/models/bggpt-gemma-2-9b-it-v1.0

# --- BELEBELE-only for bggpt_7b ---
uv run thesis-eval download-assets --name bggpt_7b
uv run thesis-eval verify-assets  --name bggpt_7b

PJRT_DEVICE=TPU uv run thesis-eval predict-belebele \
  --model bggpt_7b \
  --backend transformers --profile trc_tpu \
  --gpu-batch-size 32 --trust-remote-code \
  --output-dir outputs/spec \
  --output-template "{model}_{lang}.belebele.jsonl"

rm -rf assets/models/bggpt-7b-instruct-v0.2
echo "Tab 2 done."
```

### Tab 3 — `llama3_1_8b_reference` (optional; English-aligned baseline)

Skip this tab if you don't need the reference baseline. Otherwise:

```bash
LANGS=(ara bul dan eng fin ita nor por rus spa swa swe ukr)
AUDIT_ARGS=(); for L in "${LANGS[@]}"; do AUDIT_ARGS+=(--translations "outputs/translations/$L.audit.jsonl"); done

uv run thesis-eval download-assets --name llama3_1_8b_reference
uv run thesis-eval verify-assets  --name llama3_1_8b_reference

PJRT_DEVICE=TPU uv run thesis-eval predict-belebele \
  --model llama3_1_8b_reference \
  --backend transformers --profile trc_tpu \
  --gpu-batch-size 32 \
  --output-dir outputs/spec \
  --output-template "{model}_{lang}.belebele.jsonl"

PJRT_DEVICE=TPU uv run thesis-eval generate-targets \
  "${AUDIT_ARGS[@]}" \
  --model llama3_1_8b_reference \
  --backend transformers --profile trc_tpu \
  --gpu-batch-size 8 \
  --max-tokens 512 --temperature 0.0 --trust-remote-code \
  --output-dir outputs/generations \
  --output-template "{model}_{lang}.jsonl"

rm -rf assets/models/llama-3.1-8b-instruct
echo "Tab 3 done."
```

---

## 5. Pull artifacts back (laptop)

Once at least one VM has finished, pull its `outputs/spec/` and
`outputs/generations/`. You can pull from the others as they finish — the
`scp` is idempotent because filenames are unique per (model, language).

```bash
cd ~/Developer/ufsc-thesis

# Spec (BELEBELE predictions). 4 commands, one per VM:
gcloud compute tpus tpu-vm scp --recurse --project=mestrado-495311 --zone=europe-west4-a thesis-belebele-0:~/ufsc-thesis/eval/outputs/spec ./eval/outputs/
gcloud compute tpus tpu-vm scp --recurse --project=mestrado-495311 --zone=europe-west4-a thesis-belebele-1:~/ufsc-thesis/eval/outputs/spec ./eval/outputs/
gcloud compute tpus tpu-vm scp --recurse --project=mestrado-495311 --zone=europe-west4-a thesis-belebele-2:~/ufsc-thesis/eval/outputs/spec ./eval/outputs/
gcloud compute tpus tpu-vm scp --recurse --project=mestrado-495311 --zone=europe-west4-a thesis-belebele-3:~/ufsc-thesis/eval/outputs/spec ./eval/outputs/

# Generations (StrongREJECT inputs). 4 commands:
gcloud compute tpus tpu-vm scp --recurse --project=mestrado-495311 --zone=europe-west4-a thesis-belebele-0:~/ufsc-thesis/eval/outputs/generations ./eval/outputs/
gcloud compute tpus tpu-vm scp --recurse --project=mestrado-495311 --zone=europe-west4-a thesis-belebele-1:~/ufsc-thesis/eval/outputs/generations ./eval/outputs/
gcloud compute tpus tpu-vm scp --recurse --project=mestrado-495311 --zone=europe-west4-a thesis-belebele-2:~/ufsc-thesis/eval/outputs/generations ./eval/outputs/
gcloud compute tpus tpu-vm scp --recurse --project=mestrado-495311 --zone=europe-west4-a thesis-belebele-3:~/ufsc-thesis/eval/outputs/generations ./eval/outputs/
```

If you used a different `vm_count`, prefer the auto-generated lists from
Terraform: `terraform output scp_pull_outputs_spec_commands` and
`terraform output scp_pull_outputs_generations_commands`.

The `outputs/generations/` files from this step also need to land on the **RTX
rig** for §10 StrongREJECT scoring. After pulling to the laptop, sync to the
RTX rig (LAN copy, USB drive, GitHub LFS, however you usually move files —
they're a few GB total).

---

## 6. Recover from spot preemption

If a VM is preempted mid-run:

```bash
# Laptop
cd ~/Developer/ufsc-thesis/runbooks/tpu-terraform
terraform apply           # recreates the missing VM with the same name
gcloud compute tpus tpu-vm ssh thesis-belebele-N --project=mestrado-495311 --zone=europe-west4-a
```

On the new VM, paste §3 (bootstrap) again, then re-paste the matching tab
block from §4. predict-belebele and generate-targets pick up at the next
unfinished language because of `*.partial.jsonl` checkpoints.

---

## 7. Tear down

After §5 pulled everything back and you've confirmed file counts on the
laptop:

```bash
cd ~/Developer/ufsc-thesis/runbooks/tpu-terraform
terraform destroy

# Sanity check: no live TPU VMs left
gcloud compute tpus tpu-vm list --project=mestrado-495311
```

`terraform destroy` releases TRC quota immediately. Don't skip this — TPU VMs
keep billing (well, against quota) while up.

---

## 8. Verification

After all tabs and `terraform destroy` finish, on the laptop:

```bash
cd ~/Developer/ufsc-thesis/eval

# 13 BELEBELE files per heavy-§9 model + the BELEBELE-only models. ~900 rows each.
# Note: llamantino_2_ultrachat_7b is excluded — its BELEBELE was completed
# locally and is already at outputs/spec/llamantino_2_ultrachat_7b.belebele.repaired.jsonl.
for m in llamantino_anita_8b ai_sweden_llama3_8b bggpt_gemma_9b llama3_1_8b_reference \
         gpt_sw3 bggpt_7b; do
  count=$(ls outputs/spec/${m}_*.belebele.jsonl 2>/dev/null | wc -l)
  printf "%-30s %s files\n" "$m" "$count"
done

# 13 §9 files per heavy-§9 model. ~309 rows each (post-audit retained count).
for m in llamantino_anita_8b ai_sweden_llama3_8b bggpt_gemma_9b llama3_1_8b_reference; do
  count=$(ls outputs/generations/${m}_*.jsonl 2>/dev/null | wc -l)
  printf "%-30s %s files\n" "$m" "$count"
done
```

Each heavy-§9 model should have 13 BELEBELE files **and** 13 generation
files. BELEBELE-only models have 13 BELEBELE files and zero generation files.
