# RTX 5070 Ti runbook — light §9 + StrongREJECT scoring

Two PowerShell blocks. Run §1 immediately (parallel with the TPU work in
[`TPU_RUN.md`](TPU_RUN.md)). Run §2 after the TPU `outputs/generations/` are
copied onto this machine.

> **Working directory** for everything below: `path\to\ufsc-thesis\eval` on
> the RTX rig (the dir containing `pyproject.toml`, `assets/`, `outputs/`).
>
> **Required env**: `HF_TOKEN` and `OPENAI_API_KEY` are the two credentials
> the pipeline reads. They live in `eval\.env` (one `KEY=VALUE` per line, no
> quotes). The `thesis-eval-gpu` wrapper passes `.env` into the Docker
> container automatically.

---

## §1 — Light §9 generations (start NOW)

Models that fit the 16 GB VRAM with vLLM's KV cache for `--max-tokens 512`:
`gpt_sw3` (6.7B) and `bggpt_7b` (7B). The 8B and 9B targets go to TPU.

```powershell
# Build the audit-translation arg list once (13 languages, post-audit).
$LANGUAGES = @("ara","bul","dan","eng","fin","ita","nor","por","rus","spa","swa","swe","ukr")
$AuditArgs = foreach ($lang in $LANGUAGES) {
  "--translations"; "outputs/translations/$lang.audit.jsonl"
}

foreach ($MODEL in @("gpt_sw3","bggpt_7b")) {
  Write-Host "=== Downloading $MODEL ==="
  uv run thesis-eval download-assets --name $MODEL
  uv run thesis-eval verify-assets  --name $MODEL

  Write-Host "=== Generating §9 for $MODEL ==="
  uv run thesis-eval-gpu generate-targets `
    @AuditArgs `
    --model $MODEL --backend vllm --profile rtx_5070_ti `
    --max-tokens 512 --temperature 0.0 --trust-remote-code `
    --output-dir outputs/generations `
    --output-template "{model}_{lang}.jsonl"
}

# Sanity check: 13 generation files per model.
foreach ($MODEL in @("gpt_sw3","bggpt_7b")) {
  $count = (Get-ChildItem "outputs/generations/${MODEL}_*.jsonl").Count
  Write-Host ("{0,-15} {1} files" -f $MODEL, $count)
}
```

Expected per-model wall time on the 5070 Ti: ~3–5 hours (≈4,021 retained
prompts × 13 languages with vLLM). Both models in series: ~6–10 hours
unattended.

---

## Sync TPU §9 generations onto this rig

After Tab 0/1/2/3 in `TPU_RUN.md` finish their `generate-targets` step on
each TPU VM, the laptop pulls them via `gcloud compute tpus tpu-vm scp`.
Get those `outputs/generations/*.jsonl` files onto this RTX rig before §2.

Easiest: copy the laptop's `eval/outputs/generations/` directory over LAN /
USB / `rsync` so the RTX rig's `outputs/generations/` ends up holding all 8
× 13 = 104 files (3 already there + 2 newly added by §1 above + 3–4 from
TPU). Then proceed.

Quick count check on the RTX rig before continuing:

```powershell
(Get-ChildItem outputs/generations/*.jsonl).Count
# Target: 104 (or 117 if you ran the optional llama3_1_8b_reference on TPU Tab 3).
```

---

## §2 — StrongREJECT scoring (run AFTER §1 + TPU §9 are merged)

Score every `outputs/generations/*.jsonl` file in one pass. NLLB-600M loads
once, then each input is back-translated and the official StrongREJECT
rubric runs (which calls OpenAI under the hood, hence `OPENAI_API_KEY`).

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

Bump to NLLB-3.3B for thesis-grade back-translation if you've downloaded it
to `assets/models/nllb-200-3.3B` — change `--checkpoint` accordingly. The
600M is fine for a first complete pass.

---

## §3 — Hand off to laptop

When §2 finishes, copy the new `outputs/scored/` and (if needed) the
`outputs/generations/` for `gpt_sw3` + `bggpt_7b` over to the laptop, then
follow [`RECONVERGE.md`](RECONVERGE.md).

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
- `spec`: 9 × 13 = 117 (8 paired models + the reference baseline if you ran it; subtract 13 if you skipped it). Sabiá-3 already contributes 13 from before.

If anything's off, find which model/language is missing and rerun the
matching §1 or §2 block — both are idempotent (`--force` re-overwrites).
