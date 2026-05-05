# Runbooks — finishing the thesis-eval pipeline

The active execution plan is now hybrid: RunPod handles the rented-GPU model
runs, while the local RTX rig keeps the lighter GPU work.

```text
TRACK A  (laptop):  Reuse completed translations, audits, local generations
                     │
                     ▼
TRACK B  (RunPod):  Pending BELEBELE + §9 target generations
                     │
                     ▼
TRACK C  (RTX):     §10 StrongREJECT scoring over all generation files
                     │
                     ▼
TRACK D  (laptop):  repair-belebele → compute-spec → build-dataset
                     │              → export-report-tables → fit-glmm
                     ▼
                  frozen dataset + Results-chapter tables/figures
```

## Files

| File | Host | Status |
| --- | --- | --- |
| [`RUNPOD_RUN.md`](RUNPOD_RUN.md) | RunPod pod + laptop | Primary path for pending BELEBELE and §9 generation. |
| [`RTX_RUN.md`](RTX_RUN.md) | Local RTX rig | **Primary path** for §10 StrongREJECT scoring and light GPU checks. |
| [`RECONVERGE.md`](RECONVERGE.md) | Laptop | Final CPU-side merge, freeze, and reporting path. |
| [`TPU_RUN.md`](TPU_RUN.md) | GCP TPU VMs | Historical fallback only. |
| [`tpu-terraform/`](tpu-terraform/) | Laptop/GCP | Historical TPU provisioning. |

## Current routing summary

| Workload | Models | Host |
| --- | --- | --- |
| Completed §9 generations | `sagui_7b`, `llamantino_2_ultrachat_7b`, `sabia_3` | Reuse existing laptop outputs. |
| Pending BELEBELE + §9 | `llamantino_anita_8b`, `ai_sweden_llama3_8b`, `gpt_sw3`, `bggpt_gemma_9b`, `bggpt_7b`, optional `llama3_1_8b_reference` | RunPod A100/H100 via `--profile runpod_a100`. |
| §10 StrongREJECT scoring | All 8 paired models, optional reference | RTX 5070 Ti, after copying all generation files into `outputs/generations/`. |
| Dataset freeze and reporting | n/a | Laptop. |

## Execution order

1. On the laptop, pull any useful TPU outputs and destroy live TPU resources if
   they exist. `RUNPOD_RUN.md` includes the cleanup block.
2. Start a RunPod A100 80 GB or H100 80 GB pod, bootstrap `thesis-eval`, and
   sync existing `outputs/translations/`, `outputs/generations/`, and
   `outputs/spec/` into the pod.
3. On RunPod, run the per-model loop in [`RUNPOD_RUN.md`](RUNPOD_RUN.md) for
   each pending model.
4. Pull `outputs/spec/` and `outputs/generations/` back to the laptop, then to
   the RTX rig if that is a separate machine.
5. On the RTX rig, run StrongREJECT scoring with [`RTX_RUN.md`](RTX_RUN.md).
6. Pull `outputs/scored/` back to the laptop.
7. On the laptop, run [`RECONVERGE.md`](RECONVERGE.md).

## Guardrails

- Do not regenerate `sabia_3` locally; it is API-only and provider metadata is
  part of the dataset contract.
- Do not overwrite `outputs/dataset_frozen.parquet` during pilots.
- Do not collapse provider blocks, API failures, and in-model refusals.
- Use `--profile runpod_a100` for RunPod generation/BELEBELE commands so logs
  record the intended runtime.
- Keep §10 scoring on RTX unless the RTX rig is unavailable; RunPod scoring is
  only a fallback before teardown.
- Terminate the RunPod pod only after generation and BELEBELE outputs are
  verified on the laptop or RTX rig.
