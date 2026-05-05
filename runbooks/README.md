# Runbooks — finishing the thesis-eval pipeline

Three parallel execution tracks bring the pipeline from its current state
(translations + audit done, 3/8 §9 generations done, 2/8 BELEBELE done) to a
frozen analysis dataset and the Results-chapter tables.

```
TRACK A  (laptop):  Step 0  Batched transformers runner patch    [DONE in eval/ branch]
TRACK B  (RTX):     §9 light — gpt_sw3, bggpt_7b                  [vLLM, independent of A]
TRACK C  (laptop):  This directory's Terraform + runbooks         [DONE]
                     │
                     ▼ A done
TRACK D  (TPU):     Step 2  BELEBELE × 6 + §9 heavy × 4           [4 VMs in parallel]
                     │
                     ▼ B and D done
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
| §11 BELEBELE | 6 pending paired + optional reference | TPU v6e-1 |
| §9 generation, ≤7B | `gpt_sw3` (6.7B), `bggpt_7b` (7B) | RTX 5070 Ti / vLLM |
| §9 generation, ≥8B | `llamantino_anita_8b`, `ai_sweden_llama3_8b`, `bggpt_gemma_9b`, optional `llama3_1_8b_reference` | TPU v6e-1 (after batching patch) |
| §10 StrongREJECT scoring | All 8 generation outputs | RTX 5070 Ti |
| compute-spec, build-dataset, tables, GLMM | n/a | Laptop |

The 16 GB RTX 5070 Ti can't fit 8B+ models with vLLM's KV cache for
`max_tokens=512`; that's why the heavy §9 work goes to TPU. BELEBELE is
short-output (`max_tokens=8`), so even though TPU's transformers backend is
slower than vLLM per token, the dominant cost is dataset/model load, not
decode.

## Per-VM model assignments (`vm_count = 4`)

Each TPU VM owns one heavy §9 model. VMs 1 and 2 also run a bonus 7B
BELEBELE-only model afterward, reusing the local SSD slot left empty by
the heavy §9 model's deletion.

| VM | Heavy §9 (§9 + BELEBELE on these weights) | BELEBELE-only (re-download then BELEBELE) |
| --- | --- | --- |
| 0  | `llamantino_anita_8b`        | —                            |
| 1  | `ai_sweden_llama3_8b`        | `gpt_sw3`                    |
| 2  | `bggpt_gemma_9b`             | `bggpt_7b`                   |
| 3  | `llama3_1_8b_reference` (optional) | —                       |

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

1. **Now (laptop, 1 person-hour)**: confirm `eval/` is on the
   `feat/batched-transformers-runner` branch (or main, after merge) and
   `python -m unittest discover -s eval/tests` is green for the new
   `test_generation_batched.py` cases.
2. **Now (RTX rig, ~10–15 hours unattended)**: paste the first block of
   [`RTX_RUN.md`](RTX_RUN.md) to start §9 for `gpt_sw3` and `bggpt_7b`.
3. **Now (laptop, ~10 minutes)**: `cd tpu-terraform && terraform init &&
   terraform apply`. Open 4 SSH sessions to the TPU VMs.
4. **TPU (4 VMs, ~8–9 h wall)**: paste the bootstrap + per-VM model block from
   [`TPU_RUN.md`](TPU_RUN.md) into each VM. They self-resume on spot
   preemption.
5. **After all §9 done (RTX rig)**: pull TPU `outputs/generations/` to the RTX
   rig, then run the §10 scoring block from [`RTX_RUN.md`](RTX_RUN.md).
6. **Final (laptop)**: pull all outputs back, run [`RECONVERGE.md`](RECONVERGE.md).

## When something goes wrong

- **Spot preemption mid-run**: the v6e-1 VM is gone. Re-run `terraform apply`
  to recreate it, re-paste the bootstrap block, re-paste the model block —
  predict-belebele and generate-targets resume from `*.partial.jsonl`.
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
gcloud compute tpus tpu-vm list --project=mestrado-495311
cd ~/Developer/ufsc-thesis/runbooks/tpu-terraform && terraform destroy
```

Both should report zero live VMs.
