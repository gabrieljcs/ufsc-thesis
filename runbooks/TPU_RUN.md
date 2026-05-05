# TPU runbook — BELEBELE × 6 + heavy §9 × 4

Copy-paste end-to-end. Each block is self-contained; each command is annotated
only where the why is non-obvious.

**Default deployment** (set in `terraform.tfvars`):

| Field | Value |
| --- | --- |
| Project | `mestrado-495311` |
| Zone | `europe-west4-a` |
| Slice | `v6e-1` (1 chip × 32 GB HBM) |
| Pricing | spot (preemptible) |
| `vm_count` | 4 |

> **VMs have no external IPs, but they have internet via Cloud NAT.** The
> Terraform stack creates a regional Cloud Router + NAT gateway
> (`thesis-tpu-router-<region>` / `thesis-tpu-nat-<region>`) attached to
> the default network, plus an IAP firewall rule
> (`thesis-tpu-iap-ssh`, TCP/22 from `35.235.240.0/20`). Net effect:
>
> - **Outbound internet** works on each VM (HF / PyPI / GitHub / libtpu
>   wheels) via the NAT, so `curl ... | sh` / `gh repo clone` / `uv sync`
>   in §3 all succeed.
> - **SSH from your laptop** uses `gcloud ssh`, which auto-tunnels through
>   IAP since the VM has no external IP. Make sure `iap.googleapis.com`
>   is enabled and your account has `roles/iap.tunnelResourceAccessor`
>   on the project.
>
> The `network_config(s)` block on `google_tpu_v2_vm` is intentionally
> omitted — the google-beta provider has a known bug serializing both the
> singular and plural forms, which the Cloud TPU API rejects with
> "NetworkConfig and NetworkConfigs fields are exclusive". Omitting it
> lets apply succeed and Cloud NAT carries the connectivity load.

> **Precondition**: the `eval/` repo branch you push must contain the batched
> transformers runner patch (commit `1b1eb9b`). The `--judge-model` flag
> for §10 (`4783012`) is RTX-only, but pinning to `4783012` or later on
> `main` covers both. Verify:
> ```bash
> cd ~/Developer/ufsc-thesis/eval && uv run python -m unittest tests.test_generation_batched -v
> ```
> All 8 tests should pass before you touch a TPU.

---

## 1. Provision the TPU VMs (laptop)

```bash
cd ~/Developer/ufsc-thesis/eval/runbooks/tpu-terraform
terraform init
terraform apply           # confirm with "yes"
```

If `terraform apply` fails with `Error code 8: There is no more capacity in
the zone` (Cloud TPU runs out of physical capacity *independently* of your
quota — TRC quotas don't reserve capacity), use the supplied retry helper
that polls every 2 minutes:

```bash
bash retry-apply.sh
# Logs to retry-apply.log; bails out on any non-capacity error.
# Cap: 30 attempts / ~1 hour. If that elapses without success, pivot zones
# (see "If capacity stays out" below).
```

Once apply succeeds, print the SSH and scp pull commands you'll use later:

```bash
terraform output ssh_commands
terraform output scp_pull_outputs_spec_commands
terraform output scp_pull_outputs_generations_commands
```

If you want fewer than 4 VMs (e.g., low quota or debugging): `terraform apply -var "vm_count=1"`.

Provisioning takes ~3 minutes per VM. Each VM gets a 100 GB local SSD and a
public IP attached via the `network_configs.enable_external_ips = true`
setting in `main.tf` (without that, `gcloud ssh` falls back to IAP and
times out, and the VM has no outbound internet for `curl`/`git`/HF).

### If capacity stays out

After ~1 hour of failed retries, change `zone` (and matching `region`) in
`terraform.tfvars` to a different TRC-allocated zone with the same
accelerator family, e.g. `us-east1-d` for v6e. Then `terraform apply`.

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

The first SSH may take ~30 s while gcloud generates `~/.ssh/google_compute_engine`
(if not already present) and uploads the public key to the VM via instance
metadata. Same key is reused by `gcloud compute tpus tpu-vm scp` later, so
no per-VM key management.

---

## 3. Bootstrap (paste into EACH VM, one-time)

```bash
# Python 3.12 system-wide (TPU base image ships 3.10; thesis-eval pins
# requires-python >=3.12,<3.13). Installing system-wide lands
# libpython3.12.so.1.0 in /usr/lib so torch_xla's _XLAC.so native module
# can dlopen it at runtime. Without this, you'd need a manual
# LD_LIBRARY_PATH dance pointing at uv's managed Python's lib dir.
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -y
sudo apt-get install -y python3.12 python3.12-dev libpython3.12

# uv (we'll have it use the system python3.12 when creating venvs)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# gh CLI for cloning the private eval repo, plus tmux so the per-VM block
# in §4 survives SSH disconnects (a ~10-13 h block from your laptop will
# almost certainly drop before it finishes).
sudo apt-get install -y gh tmux
gh auth login --hostname github.com --git-protocol https --web
# ↑ follow the device-flow prompt: copy the one-time code, open the URL on
#   your laptop, paste the code, authorize. Then back in the VM:

# Install gh as the git credential helper so `git clone`/`git fetch`/etc
# (including the one inside `gh repo clone`) reuse the gh token without
# prompting for a password.
gh auth setup-git

gh repo clone gabrieljcs/ufsc-thesis ~/ufsc-thesis
cd ~/ufsc-thesis/eval

# Make sure you're on the branch with the batched runner patch (skip if merged to main)
# git checkout feat/batched-transformers-runner

# Transformers stack via the platform-agnostic mac extra. Use the system
# python3.12 (installed above) so libpython3.12.so.1.0 sits in /usr/lib.
uv sync --extra mac --python /usr/bin/python3.12

# Overlay TPU torch + torch_xla from the libtpu wheel index. This replaces
# the torch installed by --extra mac with one paired exactly with the matching
# torch_xla, plus the TPU-flavored libtpu via the [tpu] extra. The PyPI
# `torch_xla` 2.9.0 is built against torch 2.9.0; if you let `~=2.8` resolve
# you get torch 2.8.0 + torch_xla 2.9.0 → ABI mismatch → undefined symbol
# at import time. So pin both to the same exact version.
uv pip install --force-reinstall \
  'torch==2.8.0' 'torch_xla[tpu]==2.8.0' \
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
HF_TOKEN=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
HF_HUB_ENABLE_HF_TRANSFER=0
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
>   --max-input-length 1024 \
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
each other. Each block runs ~10–13 hours (down from ~6–9 since all §9 now
runs on TPU). predict-belebele and generate-targets write per-language
`*.partial.jsonl` checkpoints, so a Ctrl-C or spot preemption is
recoverable.

> **Always `git pull` before running**, on every VM. The runbook's
> `--gpu-batch-size`, `--max-input-length`, sliding-window cache fix,
> and trailing-batch shape padding were all hardened across multiple
> OOM iterations; older clones don't have those patches and will OOM
> again. After SSH'ing in (or reattaching to tmux):
>
> ```bash
> cd ~/ufsc-thesis && git pull
> cd eval
> set -a && source .env && set +a
> ```

> **Why these `--gpu-batch-size` and `--max-input-length` numbers?** v6e-1
> has 31.25 GB usable HBM. An 8B bf16 model is ~16 GB resident
> ("program"); the rest goes to prefill attention scores (`f32[batch,
> heads, seq, seq] × 4 B`), KV cache, and layer activations
> ("arguments"). XLA needs fixed shapes to avoid HLO recompiles. The
> settings below are the safe sweet spot found by retrying through OOMs:
> - **BELEBELE**: `--gpu-batch-size 32 --max-input-length 1024` →
>   attention 4 GB, total ~24 GB. Prompts cap at ~700 tokens; no
>   truncation risk.
> - **§9 generation**: `--gpu-batch-size 4 --max-input-length 2048` →
>   attention 2 GB, arguments ~7.5 GB, total ~24 GB. Doubling batch to 8
>   pushes arguments to ~15 GB and crosses the 31.25 GB cap by a few
>   hundred MB; halving batch is the conservative path with no prompt
>   truncation.
> - **9B Gemma exception** (`bggpt_gemma_9b`): the program block is
>   ~19 GB instead of ~16 GB, leaving only ~12 GB for arguments. batch=4
>   + seq=2048 needs ~13.6 GB and OOMs by ~1.5 GB. Tab 2's
>   `bggpt_gemma_9b` §9 command therefore uses `--gpu-batch-size 2`
>   (~6.8 GB arguments, total ~26 GB, comfortable). bggpt_7b stays at
>   batch=4.

> **Run inside `tmux`** so the block survives SSH disconnects (which WILL
> happen over a 10+ hour window). Per VM:
>
> ```bash
> tmux new -s thesis            # start a new persistent session
> # …paste the model block below…
> # detach with: Ctrl-b then d   (the block keeps running)
> # closing the SSH terminal also detaches; the tmux session lives on
> ```
>
> To reattach from a fresh SSH session:
>
> ```bash
> gcloud compute tpus tpu-vm ssh thesis-belebele-N --project=mestrado-495311 --zone=europe-west4-a
> tmux attach -t thesis
> ```
>
> Without `tmux`, an SSH drop sends SIGHUP to your block and kills
> everything in flight (downloads, model loads, generation). Resumable
> commands recover from `*.partial.jsonl`, but you lose any in-flight HF
> downloads (which would force a `download-assets` rerun).

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
  --max-input-length 1024 \
  --output-dir outputs/spec \
  --output-template "{model}_{lang}.belebele.jsonl"

PJRT_DEVICE=TPU uv run thesis-eval generate-targets \
  "${AUDIT_ARGS[@]}" \
  --model llamantino_anita_8b \
  --backend transformers --profile trc_tpu \
  --gpu-batch-size 4 \
  --max-input-length 2048 \
  --max-tokens 512 --temperature 0.0 --trust-remote-code \
  --output-dir outputs/generations \
  --output-template "{model}_{lang}.jsonl"

rm -rf assets/models/llamantino-3-anita-8b-inst-dpo-ita
echo "Tab 0 done."
```

### Tab 1 — `ai_sweden_llama3_8b` + `gpt_sw3`

```bash
LANGS=(ara bul dan eng fin ita nor por rus spa swa swe ukr)
AUDIT_ARGS=(); for L in "${LANGS[@]}"; do AUDIT_ARGS+=(--translations "outputs/translations/$L.audit.jsonl"); done

# --- §9 + BELEBELE for ai_sweden_llama3_8b ---
uv run thesis-eval download-assets --name ai_sweden_llama3_8b
uv run thesis-eval verify-assets  --name ai_sweden_llama3_8b

PJRT_DEVICE=TPU uv run thesis-eval predict-belebele \
  --model ai_sweden_llama3_8b \
  --backend transformers --profile trc_tpu \
  --gpu-batch-size 32 \
  --max-input-length 1024 \
  --output-dir outputs/spec \
  --output-template "{model}_{lang}.belebele.jsonl"

PJRT_DEVICE=TPU uv run thesis-eval generate-targets \
  "${AUDIT_ARGS[@]}" \
  --model ai_sweden_llama3_8b \
  --backend transformers --profile trc_tpu \
  --gpu-batch-size 4 \
  --max-input-length 2048 \
  --max-tokens 512 --temperature 0.0 --trust-remote-code \
  --output-dir outputs/generations \
  --output-template "{model}_{lang}.jsonl"

rm -rf assets/models/ai-sweden-llama-3-8b-instruct

# --- §9 + BELEBELE for gpt_sw3 ---
uv run thesis-eval download-assets --name gpt_sw3
uv run thesis-eval verify-assets  --name gpt_sw3

PJRT_DEVICE=TPU uv run thesis-eval predict-belebele \
  --model gpt_sw3 \
  --backend transformers --profile trc_tpu \
  --gpu-batch-size 32 --trust-remote-code \
  --max-input-length 1024 \
  --output-dir outputs/spec \
  --output-template "{model}_{lang}.belebele.jsonl"

PJRT_DEVICE=TPU uv run thesis-eval generate-targets \
  "${AUDIT_ARGS[@]}" \
  --model gpt_sw3 \
  --backend transformers --profile trc_tpu \
  --gpu-batch-size 4 \
  --max-input-length 2048 \
  --max-tokens 512 --temperature 0.0 --trust-remote-code \
  --output-dir outputs/generations \
  --output-template "{model}_{lang}.jsonl"

rm -rf assets/models/gpt-sw3-6.7b-v2-instruct
echo "Tab 1 done."
```

### Tab 2 — `bggpt_gemma_9b` + `bggpt_7b`

```bash
LANGS=(ara bul dan eng fin ita nor por rus spa swa swe ukr)
AUDIT_ARGS=(); for L in "${LANGS[@]}"; do AUDIT_ARGS+=(--translations "outputs/translations/$L.audit.jsonl"); done

# --- §9 + BELEBELE for bggpt_gemma_9b ---
uv run thesis-eval download-assets --name bggpt_gemma_9b
uv run thesis-eval verify-assets  --name bggpt_gemma_9b

PJRT_DEVICE=TPU uv run thesis-eval predict-belebele \
  --model bggpt_gemma_9b \
  --backend transformers --profile trc_tpu \
  --gpu-batch-size 32 \
  --max-input-length 1024 \
  --output-dir outputs/spec \
  --output-template "{model}_{lang}.belebele.jsonl"

# bggpt_gemma_9b §9: gpu-batch-size=2 (NOT 4 like the other models). Gemma-2-9B's
# program block is ~19 GB (vs ~16 GB for the 8B Llamas), leaving only ~12 GB for
# arguments — batch=4 + seq=2048 needs ~13.6 GB and OOMs by 1.5 GB. Halving
# the batch to 2 brings arguments down to ~6.8 GB, total ~26 GB, comfortable.
PJRT_DEVICE=TPU uv run thesis-eval generate-targets \
  "${AUDIT_ARGS[@]}" \
  --model bggpt_gemma_9b \
  --backend transformers --profile trc_tpu \
  --gpu-batch-size 2 \
  --max-input-length 2048 \
  --max-tokens 512 --temperature 0.0 --trust-remote-code \
  --output-dir outputs/generations \
  --output-template "{model}_{lang}.jsonl"

rm -rf assets/models/bggpt-gemma-2-9b-it-v1.0

# --- §9 + BELEBELE for bggpt_7b ---
uv run thesis-eval download-assets --name bggpt_7b
uv run thesis-eval verify-assets  --name bggpt_7b

PJRT_DEVICE=TPU uv run thesis-eval predict-belebele \
  --model bggpt_7b \
  --backend transformers --profile trc_tpu \
  --gpu-batch-size 32 --trust-remote-code \
  --max-input-length 1024 \
  --output-dir outputs/spec \
  --output-template "{model}_{lang}.belebele.jsonl"

PJRT_DEVICE=TPU uv run thesis-eval generate-targets \
  "${AUDIT_ARGS[@]}" \
  --model bggpt_7b \
  --backend transformers --profile trc_tpu \
  --gpu-batch-size 4 \
  --max-input-length 2048 \
  --max-tokens 512 --temperature 0.0 --trust-remote-code \
  --output-dir outputs/generations \
  --output-template "{model}_{lang}.jsonl"

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
  --max-input-length 1024 \
  --output-dir outputs/spec \
  --output-template "{model}_{lang}.belebele.jsonl"

PJRT_DEVICE=TPU uv run thesis-eval generate-targets \
  "${AUDIT_ARGS[@]}" \
  --model llama3_1_8b_reference \
  --backend transformers --profile trc_tpu \
  --gpu-batch-size 4 \
  --max-input-length 2048 \
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

If you used a different `zone` or `vm_count`, prefer the auto-generated
lists from Terraform: `terraform output scp_pull_outputs_spec_commands`
and `terraform output scp_pull_outputs_generations_commands` regenerate
against the live config.

The `outputs/generations/` files from this step also need to land on the **RTX
rig** for §10 StrongREJECT scoring. After pulling to the laptop, sync to the
RTX rig (LAN copy, USB drive, GitHub LFS, however you usually move files —
they're a few GB total).

---

## 6. Recover after a failure (OOM / preemption / crash)

The two failure modes in practice:

1. **§9 OOM** (`Used 31.56G of 31.25G hbm. Exceeded hbm capacity`). The
   model and language that failed are still at the partial-checkpoint
   stage; nothing after that point ran. Pull the latest patches and
   rerun the same per-VM block; completed languages are skipped, the
   failed language resumes from its `*.partial.jsonl`.
2. **Spot preemption.** The VM is gone. Recreate it via Terraform, then
   bootstrap and rerun.

### 6a. Reset and restart on a single VM (most common)

On the affected VM:

```bash
# Pull the latest patches (e.g. the gpu-batch-size 4 default for §9, the
# sliding_window cache fix, the trailing-batch shape padding).
cd ~/ufsc-thesis && git pull
cd eval

# Make sure .env is still loaded after a fresh shell/tmux session.
set -a && source .env && set +a

# Reattach to your tmux session (or start a new one — see §4 callout).
tmux attach -t thesis 2>/dev/null || tmux new -s thesis

# Re-paste the matching per-VM block from §4 verbatim. predict-belebele
# and generate-targets are idempotent: a finished `*.belebele.jsonl` /
# `*.jsonl` is skipped with a one-line log; an interrupted language
# resumes from its sibling `*.partial.jsonl`. No prior good work is lost.
```

If you suspect a partial checkpoint is corrupted (e.g., process killed
mid-write), nuke just that one and rerun:

```bash
# Delete the partial for ONE failed language (replace MODEL and LANG):
MODEL=bggpt_7b
LANG=ara
rm -f outputs/spec/${MODEL}_${LANG}.belebele.partial.jsonl
rm -f outputs/generations/${MODEL}_${LANG}.partial.jsonl
# Then re-paste the per-VM block as above.
```

### 6b. Spot preemption (VM gone entirely)

```bash
# Laptop
cd ~/Developer/ufsc-thesis/eval/runbooks/tpu-terraform
terraform apply           # recreates the missing VM with the same name
# (Use bash retry-apply.sh if capacity is tight.)
gcloud compute tpus tpu-vm ssh thesis-belebele-N --project=mestrado-495311 --zone=europe-west4-a
```

On the new VM, paste §3 (bootstrap) again, then re-paste the matching tab
block from §4. `predict-belebele` and `generate-targets` pick up at the
next unfinished language because of `*.partial.jsonl` checkpoints, but
note: the VM lost any in-flight HF model snapshot under `assets/models/`,
so the per-VM block's `download-assets` step will re-download (~5–10 min
per model).

### 6c. Hard reset for one model (start that model's outputs from zero)

If a model's outputs look bad and you want a clean redo (rare — only do
this if you have evidence the partial JSONL is wrong, not just because
something looked slow):

```bash
MODEL=bggpt_7b
# Wipe BELEBELE outputs for that model
rm -f outputs/spec/${MODEL}_*.belebele.jsonl outputs/spec/${MODEL}_*.belebele.partial.jsonl
# Wipe §9 outputs for that model
rm -f outputs/generations/${MODEL}_*.jsonl outputs/generations/${MODEL}_*.partial.jsonl
# Then re-paste the per-VM block; it'll start from scratch for this model.
```

Other models' outputs are untouched.

---

## 7. Tear down

After §5 pulled everything back and you've confirmed file counts on the
laptop:

```bash
cd ~/Developer/ufsc-thesis/eval/runbooks/tpu-terraform
terraform destroy

# Sanity check: no live TPU VMs in the configured zone
gcloud compute tpus tpu-vm list --project=mestrado-495311 --zone=europe-west4-a
```

`terraform destroy` releases TPU VMs and the custom subnet. Don't skip this —
on-demand TPU VMs bill while up.

---

## 8. Verification

After all tabs and `terraform destroy` finish, on the laptop:

```bash
cd ~/Developer/ufsc-thesis/eval

# 13 BELEBELE files per model. ~900 rows each.
# Note: llamantino_2_ultrachat_7b is excluded — its BELEBELE was completed
# locally and is already at outputs/spec/llamantino_2_ultrachat_7b.belebele.repaired.jsonl.
for m in llamantino_anita_8b ai_sweden_llama3_8b bggpt_gemma_9b llama3_1_8b_reference \
         gpt_sw3 bggpt_7b; do
  count=$(ls outputs/spec/${m}_*.belebele.jsonl 2>/dev/null | wc -l)
  printf "%-30s %s files\n" "$m" "$count"
done

# 13 §9 files per model. ~309 rows each (post-audit retained count). All §9
# now runs on TPU, so every paired model in the loop should land 13 files.
for m in llamantino_anita_8b ai_sweden_llama3_8b bggpt_gemma_9b llama3_1_8b_reference \
         gpt_sw3 bggpt_7b; do
  count=$(ls outputs/generations/${m}_*.jsonl 2>/dev/null | wc -l)
  printf "%-30s %s files\n" "$m" "$count"
done
```

Each model should have 13 BELEBELE files **and** 13 §9 generation files
(after TPU runs complete). The exception is `llama3_1_8b_reference` if you
skipped Tab 3 — those files won't exist.
