# RTX 5070 Ti runbook — StrongREJECT scoring only

The RTX rig used to run §9 for the ≤7B models, but throughput on vLLM was
too slow to be worthwhile. **All §9 generations now run on TPU.** This
runbook is the §10 StrongREJECT scoring pass over every model's outputs.

> **Working directory** for everything below: `path\to\ufsc-thesis\eval` on
> the RTX rig (the dir containing `pyproject.toml`, `assets/`, `outputs/`).
>
> **Required env**: `HF_TOKEN` (HF mirror access) and `OPENAI_API_KEY` (for
> the StrongREJECT rubric judge). Both live in `eval\.env`, one
> `KEY=VALUE` per line, no quotes. The `thesis-eval-gpu` wrapper passes
> `.env` into the Docker container automatically.

---

## §1 — Sync TPU §9 generations onto this rig

After all four tabs in `TPU_RUN.md` finish their `generate-targets` step,
the laptop pulls them via `gcloud compute tpus tpu-vm scp`. Get those
`outputs/generations/*.jsonl` files onto this RTX rig before scoring.

Easiest: copy the laptop's `eval/outputs/generations/` directory over LAN
/ USB / `rsync` so the RTX rig's `outputs/generations/` ends up holding
all 8 × 13 = 104 files (3 from earlier work + the 5–6 new ones from TPU).

Quick count check before continuing:

```powershell
(Get-ChildItem outputs/generations/*.jsonl).Count
# Target: 104 (or 117 if you ran the optional llama3_1_8b_reference on TPU Tab 3).
```

---

## §2 — StrongREJECT scoring

Score every `outputs/generations/*.jsonl` file in one pass. NLLB-600M
loads once, then each input is back-translated and the official
StrongREJECT rubric runs (which calls OpenAI under the hood, hence
`OPENAI_API_KEY`).

```powershell
# Make sure the scored output dir exists.
if (-not (Test-Path outputs/scored)) { New-Item -ItemType Directory -Path outputs/scored | Out-Null }

# Build the --generations arg list from every existing generation file.
$ScoreArgs = Get-ChildItem outputs/generations/*.jsonl | ForEach-Object {
  "--generations"; $_.FullName
}

uv run thesis-eval-gpu score-strongreject `
  @ScoreArgs `
  --evaluator strongreject_rubric `
  --judge-model gpt-5.4 `
  --backtranslate-engine nllb `
  --checkpoint assets/models/nllb-200-distilled-600M `
  --device cuda `
  --output-dir outputs/scored `
  --output-template "{stem}.strongreject.jsonl"

# Sanity check: one scored file per generation file.
$gen   = (Get-ChildItem outputs/generations/*.jsonl).Count
$score = (Get-ChildItem outputs/scored/*.strongreject.jsonl).Count
Write-Host "generations=$gen, scored=$score"
# These should match.
```

Expected wall time: ~2–4 hours for ~50k scored rows total (NLLB
back-translation + per-row OpenAI rubric call).

Bump to NLLB-3.3B for thesis-grade back-translation if you've downloaded
it to `assets/models/nllb-200-3.3B` — change `--checkpoint` accordingly.
The 600M is fine for a first complete pass.

---

## §3 — Hand off to laptop

When §2 finishes, copy the new `outputs/scored/` to the laptop and follow
[`RECONVERGE.md`](RECONVERGE.md).

```powershell
# Quick file-count summary before handing off:
$ts = @(
  @{ name = "translations/audit"; count = (Get-ChildItem outputs/translations/*.audit.jsonl).Count }
  @{ name = "generations";        count = (Get-ChildItem outputs/generations/*.jsonl).Count }
  @{ name = "scored";             count = (Get-ChildItem outputs/scored/*.strongreject.jsonl).Count }
  @{ name = "spec";               count = (Get-ChildItem outputs/spec/*.belebele.jsonl).Count }
)
$ts | ForEach-Object { Write-Host ("{0,-22} {1} files" -f $_.name, $_.count) }
```

Expected counts:

- `translations/audit`: 13 (one per language)
- `generations`: 104 (8 models × 13 langs); 117 if you ran the reference baseline
- `scored`: equals `generations`
- `spec`: 8 × 13 = 104 (or 117 with reference). `llamantino_2_ultrachat_7b`
  already contributes 13 from the local + repaired run; Sabiá-3 contributes
  13 from the Maritaca API run.

If anything's off, find which model/language is missing and rerun §2 —
it's idempotent (existing scored files are skipped; `--force` re-overwrites).
