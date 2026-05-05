# Runbooks — finishing the thesis-eval pipeline

Three parallel execution tracks bring the pipeline from its current state
(translations + audit done, 3/8 §9 generations done, 2/8 BELEBELE done) to a
frozen analysis dataset and the Results-chapter tables.

```
TRACK A  (laptop):  Step 0  Batched transformers runner patch    [DONE in eval/ branch]
TRACK C  (laptop):  This directory's Terraform + runbooks         [DONE]
                     │
                     ▼ A done
TRACK D  (TPU):     Step 2  BELEBELE × 6 + §9 × 6                 [4 VMs in parallel]
                     │
                     ▼ D done
TRACK E  (RTX):     Step 3  §10 StrongREJECT scoring × 8 models
                     │
                     ▼
TRACK F  (laptop):  Step 4  compute-spec → build-dataset → export-report-tables → fit-glmm
```

## Files

| File | Host | Track |
| --- | --- | --- |
| [`TPU_RUN.md`](TPU_RUN.md) | GCP (laptop drives, TPU VMs execute) | D |
| [`RTX_RUN.md`](RTX_RUN.md) | RTX 5070 Ti (Windows + Docker) | B and E |
| [`RECONVERGE.md`](RECONVERGE.md) | Laptop (Mac, CPU only) | F |
| [`tpu-terraform/`](tpu-terraform/) | Laptop, talks to GCP | provisioning for D |

## Prereqs (one-time, on the laptop)

```bash
# 1. gcloud authenticated against the TRC project
gcloud config set project mestrado-495311
gcloud auth application-default login   # for Terraform's google provider
gcloud services enable tpu.googleapis.com

# 2. Terraform >= 1.6
brew install terraform   # or your preferred install method

# 3. github CLI (for cloning the private eval repo onto each TPU VM)
brew install gh
```

## Routing summary (which model goes where)

| Workload | Models | Host |
| --- | --- | --- |
| §11 BELEBELE | 5 paired + optional reference | TPU `v6e-1` (spot, default `europe-west4-a`) |
| §9 generation (all sizes) | `gpt_sw3`, `bggpt_7b`, `llamantino_anita_8b`, `ai_sweden_llama3_8b`, `bggpt_gemma_9b`, optional `llama3_1_8b_reference` | TPU `v6e-1` (after batching patch) |
| §10 StrongREJECT scoring | All 8 generation outputs | RTX 5070 Ti |
| compute-spec, build-dataset, tables, GLMM | n/a | Laptop |

All §9 generations run on TPU now. The 16 GB RTX 5070 Ti can't fit 8B+
models with vLLM's KV cache for `max_tokens=512`, and even the 7B models
(`gpt_sw3`, `bggpt_7b`) turned out to be too slow on the RTX path
(throughput-limited despite fitting in VRAM). The TPU's batched
transformers runner with `--gpu-batch-size 8` clears all six pending
generations in a comparable wall time to a single RTX run, in parallel
across the 4 VMs.

> **TPU VMs in this stack have no external IPs**, but the Terraform stack
> creates a Cloud NAT gateway in the same region so VMs can reach
> `huggingface.co`, `pypi.org`, `github.com`, `astral.sh`,
> `storage.googleapis.com/libtpu-releases`, etc. for the bootstrap. SSH
> from your laptop tunnels through IAP via `gcloud ssh` — make sure
> `iap.googleapis.com` is enabled on the project and your account has
> `roles/iap.tunnelResourceAccessor`. The IAP firewall rule
> (`thesis-tpu-iap-ssh`, allowing TCP/22 from `35.235.240.0/20`) is
> created by Terraform.

## Per-VM model assignments (`vm_count = 4`)

Each TPU VM owns one 8B/9B "primary" model (§9 + BELEBELE) plus, on VMs 1
and 2, a 7B "secondary" model that also runs §9 + BELEBELE on the freed
SSD slot after the primary finishes.

| VM | Primary (§9 + BELEBELE, weights downloaded once) | Secondary (§9 + BELEBELE on re-download) |
| --- | --- | --- |
| 0  | `llamantino_anita_8b`              | —                            |
| 1  | `ai_sweden_llama3_8b`              | `gpt_sw3`                    |
| 2  | `bggpt_gemma_9b`                   | `bggpt_7b`                   |
| 3  | `llama3_1_8b_reference` (optional) | —                            |

Already-done models (do **not** retry on TPU):

- `sagui_7b` — done locally on RTX
- `sabia_3` — done via Maritaca API
- `llamantino_2_ultrachat_7b` — done locally; raw outputs went through the
  `repair-belebele-predictions` pass added in commit `6157c45` to recover
  answers that were emitted as choice text rather than the letter.
  See `outputs/spec/llamantino_2_ultrachat_7b.belebele.repaired.jsonl`.

> **Note on the repair pass**: any model whose `raw_output` contains the
> answer **text** rather than the letter (e.g., outputs "carry the herbs"
> instead of "A") needs `thesis-eval repair-belebele-predictions` applied
> to its `*.belebele.jsonl` before SPEC computation. The
> [`RECONVERGE.md`](RECONVERGE.md) runbook applies this idempotently across
> every BELEBELE output — no per-VM action required.

## Execution order, in plain English

1. **Now (laptop, 1 person-hour)**: confirm `eval/` is on `main` (or your
   feature branch) past commit `4783012` — that's the cumulative tip with
   the batched transformers runner (`1b1eb9b`) for TPU and the
   `--judge-model` flag (`4783012`) for the RTX scoring pass. Quick gate:
   `uv run python -m unittest tests.test_generation_batched -v` should be
   green and `uv run thesis-eval score-strongreject --help` should list a
   `--judge-model` option.
2. **(skipped — all §9 now runs on TPU)**
3. **Now (laptop, ~10 minutes if capacity is available)**:
   `cd ~/Developer/ufsc-thesis/eval/runbooks/tpu-terraform && terraform init &&
   terraform apply`. If `apply` fails with "no more capacity in the zone",
   `bash retry-apply.sh` polls every 2 min until capacity opens (or bails
   out at 30 attempts; if it does, change `zone` and matching `region` in
   `terraform.tfvars` to a different TRC-allocated zone for the same
   accelerator family and re-run apply).
4. **TPU (4 VMs, ~8–9 h wall)**: paste the bootstrap + per-VM model block from
   [`TPU_RUN.md`](TPU_RUN.md) into each VM. v6e is spot — the per-language
   `*.partial.jsonl` checkpoints make recovery from preemption a re-paste
   of the same block.
5. **After all §9 done (RTX rig)**: pull TPU `outputs/generations/` to the
   RTX rig (or laptop, then forward), then run the §10 scoring block from
   [`RTX_RUN.md`](RTX_RUN.md). RTX is scoring-only now; no §9 runs there.
6. **Final (laptop)**: pull all outputs back, run [`RECONVERGE.md`](RECONVERGE.md).

## When something goes wrong

- **Capacity errors (`Error code 8: no more capacity`)**: not a quota
  problem — Cloud TPU runs out of physical capacity per zone independently
  of TRC quota. Use `bash retry-apply.sh` (caps at ~1 h), or change `zone`
  in `terraform.tfvars` to a different TRC zone in the same accelerator
  family.
- **Spot preemption mid-run**: re-run `terraform apply` to recreate the
  missing VM, re-paste the bootstrap block, re-paste the model block —
  predict-belebele and generate-targets resume from `*.partial.jsonl`.
- **`gcloud ssh` times out**: VMs have no external IPs (intentional — the
  google-beta provider has a known bug serializing both `network_config`
  and `network_configs` to the API). `gcloud ssh` should route through
  IAP. Verify `iap.googleapis.com` is enabled on the project, your account
  has `roles/iap.tunnelResourceAccessor`, and the `thesis-tpu-iap-ssh`
  firewall rule was created (`gcloud compute firewall-rules describe
  thesis-tpu-iap-ssh --project=mestrado-495311`).
- **VM has no internet (`curl` hangs, `git clone` fails)**: Cloud NAT
  didn't come up, or the VM is in a different region than the NAT.
  Verify `gcloud compute routers nats list --router=thesis-tpu-router-<region>
  --region=<region> --project=mestrado-495311` shows the
  `thesis-tpu-nat-<region>` gateway in `RUNNING` state.
- **TPU runtime mismatch on apply**: `gcloud compute tpus versions list
  --zone="$(terraform output -raw zone 2>/dev/null || echo europe-west4-a)"`
  and update `runtime_version` in `terraform.tfvars`.
- **Terraform "quota exceeded"**: drop `vm_count` (e.g., to 2 or 1) and run
  the per-VM blocks sequentially on the surviving VM(s).
- **Vague errors during apply**: `terraform apply -refresh=false` to skip
  re-reading state from GCP, or clean local state with `rm -rf .terraform/
  .terraform.lock.hcl && terraform init`.

## Cost / quota guardrails

After Step 4 (Track F) finishes:

```bash
gcloud compute tpus tpu-vm list --project=mestrado-495311 --zone=europe-west4-a
cd ~/Developer/ufsc-thesis/eval/runbooks/tpu-terraform && terraform destroy
```

Both should report zero live VMs.
