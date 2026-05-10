# Thesis Math Audit — Findings

**Audited:** 2026-05-10
**Scope:** Equations and numerical claims in `ufscthesisx/ufscthesisx/chapters/` and the supporting code in `eval/src/thesis_eval/`. See plan in `~/.claude/plans/i-need-you-to-sharded-wilkes.md`.

## Executive Summary

**The math is overwhelmingly correct.** All 14 odds ratios, all 8 aligned-language Wilson 95% CIs, all 8 cross-lingual pooled Wilson 95% CIs, the GLMM sample-size accounting, the collinearity diagnostics, the translation-retention arithmetic in the appendix, and the per-language pass/exclude counts in the translation QC table all reproduce within rounding tolerance. The substantive conclusions of the thesis (H1/H2/H3 all not supported, low collinearity, "1 of 4 retained" falsification result) are arithmetically intact.

The audit found **no errors that change a sign, a coefficient, or a hypothesis-level conclusion**.

It did find a small set of issues worth fixing before the final defense:

| Severity | Count | Summary |
|---|---|---|
| error | 0 | None found. |
| inconsistency | 2 | (i) Two contradictory definitions of "slope-sign retained" inside `methodology.tex`; (ii) `CONS` denominator used in code (`comparable`) differs from the `\|B\|` denominator written in `eq:cons-score`. |
| rounding | 4 | Display-precision drift of 0.1 pp in 3 closest/farthest gaps and 1 counterfactual gap. Concluded to come from rounding-already-rounded inputs at table-render time. |
| doc-only | 3 | (i) `eq:glmm` does not wrap `d` and `SPEC` in `z(·)` although the table headers and the code apply z-standardization; (ii) `eq:glmm-tokenizer` and `eq:glmm-spec-components` continue to write `d` and `SPEC` un-wrapped while wrapping their *added* predictors, creating asymmetric notation; (iii) Tokenizer-robustness model `n = 27,643` is unexplained — the gap of 4,135 rows from the SPEC-decomposition `n = 31,778` is not described in the thesis. |

Below, every finding gives a file:line, the printed value, the recomputed value (where relevant), and a one-line suggested fix.

---

## Findings

### F1 — inconsistency: two definitions of "slope-sign retained"

- **Location:** [methodology.tex:377](../../ufscthesisx/ufscthesisx/chapters/methodology.tex#L377) (prose) vs [methodology.tex:382](../../ufscthesisx/ufscthesisx/chapters/methodology.tex#L382) (table caption)
- **Problem:** The same section defines retention two different ways.
  - Line 377 (prose): *"A pair is counted as retaining the H₁ sign only when **both models show the predicted positive distance slope**."*
  - Line 382 (Table `tab:prereg-falsification` caption): *"'Slope sign retained' means the within-language regression coefficient on URIEL+ featural distance has the **same sign** in the strong-pole model as in the weak-pole model."*
- **Evidence:** The code at [reporting.py:286](../src/thesis_eval/analysis/reporting.py#L286) implements the table-caption definition (`weak_sign == strong_sign`). Under that definition, Swedish is the only retained pair (both poles negative); under the prose definition, *no* pair retains because no pair has both poles positive. The thesis reports "1 of 4" (table-caption definition) and the substantive conclusion ("≤1 → not supported") is the same under both.
- **Fix:** Replace line 377 prose with the same-sign definition the table caption already uses, e.g. *"A pair is counted as retaining the distance-slope sign only when the weak-pole and strong-pole within-language slopes share the same sign."* Or, less invasively, drop the qualifier "predicted positive" so both definitions align.

### F2 — inconsistency: CONS denominator differs between LaTeX and code

- **Location:** [methodology.tex:652–655](../../ufscthesisx/ufscthesisx/chapters/methodology.tex#L652) (`eq:cons-score`) vs [belebele.py:124](../src/thesis_eval/benchmarks/belebele.py#L124)
- **Printed equation:** `CONS(M,L) = 1 - (1/|B|) Σ_{x∈B} 1[ŷ_M(x_L) ≠ ŷ_M(x_{L_src,M})]`. The denominator is `|B|` — the full BELEBELE item set (900 items per language).
- **Code computation:** `cons_score = consistent / comparable`, where `comparable` excludes items for which either the attack-language or aligned-language prediction failed to parse. So the code's denominator is `|B_comparable| ≤ |B|`, not `|B|`.
- **Why this matters:** When both predictions are parseable for every item, the two definitions agree. When some items have unparseable predictions, the LaTeX definition implicitly treats them as either agreement (if `I[≠]` is read as 0 for unparseable pairs) or has no defined behavior. The code's choice (drop and renormalize) is the more defensible one, but it does not match the formula the reader sees.
- **Fix:** Change `eq:cons-score` to use `|B_{\text{comp}}|` (or define a comparable subset explicitly), or add a one-sentence note after the equation: *"Items for which either prediction is unparseable are excluded from both numerator and denominator; |B| denotes the comparable subset."* The code already records `n_cons_comparable` per row, so the prose can cite an empirical denominator.

### F3 — doc-only: standardization missing from `eq:glmm`

- **Location:** [methodology.tex:733–740](../../ufscthesisx/ufscthesisx/chapters/methodology.tex#L733)
- **Printed equation:** `logit(π) = α + δ_M + β₁ d(L_src,L_t) + β₂ SPEC(M,L_t) + u_k + u_{L_t}` — `d` and `SPEC` are NOT wrapped in `z(·)`.
- **Code:** [glmm.py:28–29](../src/thesis_eval/analysis/glmm.py#L28) computes `data["distance_z"] = _z(data["distance"])` and `data["spec_z"] = _z(data["spec_score"])` and passes those z-standardized columns to the GLMM. The results table headers in [results.tex:482, 484](../../ufscthesisx/ufscthesisx/chapters/results.tex#L482) read "Standardized linguistic distance" and "Standardized specialization". The methodology text at [methodology.tex:743–744](../../ufscthesisx/ufscthesisx/chapters/methodology.tex#L743) explicitly mentions "after standardization", and AGENTS.md §10.1 requires it.
- **Fix:** Wrap distance and SPEC in `z(·)` inside `eq:glmm`, mirroring the wrapping that already appears in `eq:glmm-tokenizer` for TINF and in `eq:glmm-spec-components` for IF/CONS. Alternative: add an explicit one-line standardization convention immediately before `eq:glmm` so the reader knows that any predictor referenced by name is z-standardized at the data layer.

### F4 — doc-only: asymmetric standardization wrapping in `eq:glmm-tokenizer` and `eq:glmm-spec-components`

- **Location:** [methodology.tex:752–760](../../ufscthesisx/ufscthesisx/chapters/methodology.tex#L752) (tokenizer model) and [methodology.tex:766–774](../../ufscthesisx/ufscthesisx/chapters/methodology.tex#L766) (SPEC decomposition)
- **Problem:** Both robustness equations wrap their *added* predictors in `z(·)` (TINF in the first, IF and CONS in the second) but leave the carried-over predictors `d` and `SPEC`/`d` un-wrapped. The reader is left to guess that the un-wrapped names are also z-standardized.
- **Fix:** Either wrap *all* substantive predictors uniformly, or unwrap them all and rely on a single global standardization convention stated once. Whichever is chosen, apply it consistently across `eq:glmm`, `eq:glmm-tokenizer`, and `eq:glmm-spec-components`.

### F5 — rounding: BgGPT-7B Bulgarian counterfactual gap

- **Location:** [results.tex:303](../../ufscthesisx/ufscthesisx/chapters/results.tex#L303), `tab:counterfactual-safety` row "Bulgarian / BgGPT-7B-Instruct-v0.2"
- **Printed:** Model ASR `90.4%`, Reference ASR `19.6%`, Gap `70.7 pp`.
- **Recomputed:** `90.4 − 19.6 = 70.8 pp`. The other 7 rows of the same table all match the displayed two-decimal subtraction exactly.
- **Likely cause:** The gap is computed in the table-rendering script from full-precision ASR values (e.g., `90.43% − 19.65% = 70.78%` rounding to `70.8`), then displayed as `70.7` because of an order-of-rounding choice. Verifiable in the underlying CSV `eval/outputs/tables/counterfactual_safety_by_aligned_language.csv` if it carries pre-rounded raw ASRs.
- **Fix:** Either recompute the gap from the displayed (already-rounded) ASRs and write `70.8`, or — preferably — keep the full-precision gap and round only the final cell, then state the rounding policy in the table caption to head off similar future drift.

### F6 — rounding: closest/farthest gap drift in 3 of 8 rows

- **Location:** [results.tex:218, 222, 226](../../ufscthesisx/ufscthesisx/chapters/results.tex#L218), `tab:closest-farthest-languages`
  - LLaMAntino-2 (Italian): closest 67.1%, farthest 53.5% → printed `−13.5`, recomputed `−13.6`
  - GPT-SW3 (Swedish): closest 59.4%, farthest 2.9% → printed `−56.4`, recomputed `−56.5`
  - BgGPT-7B (Bulgarian): closest 90.4%, farthest 12.9% → printed `−77.4`, recomputed `−77.5`
- **Likely cause:** Same as F5 — gap computed at full precision, displayed at 0.1 pp resolution, with the displayed-value subtraction then disagreeing by 0.1 pp on rows where both endpoints round half-up while the difference rounds half-down (or vice versa).
- **Fix:** Same as F5. The conclusions drawn from the table ("five of eight target models, the farthest language has lower ASR than the closest", `results.tex:234`) are unaffected.

### F7 — doc-only: tokenizer-robustness model n = 27,643 is unexplained

- **Location:** [results.tex:457–461](../../ufscthesisx/ufscthesisx/chapters/results.tex#L457), `tab:glmm-tokenizer-robustness`
- **Problem:** The SPEC-decomposition model and the aggregated sensitivity model both report `n = 31,778`, and the thesis explicitly accounts for that: `8 × 4,021 − 78 (excluded) − 312 (LLaMAntino-2 Russian missing CONS) = 31,778` — confirmed during this audit. The tokenizer-robustness model reports `n = 27,643`, which is `4,135` rows lower, and the chapter does not say why.
- **Fix:** Add one sentence explaining the additional drop. Likely candidates: (i) rows for which `token_inflation` could not be computed (e.g., aligned-language baseline tokenization missing for some prompt), (ii) rows excluded by chat-template application failures. The implementation in [tokenizer.py:50–53](../src/thesis_eval/metrics/tokenizer.py#L50) yields `token_inflation = None` when the baseline count is missing, and such rows are presumably dropped before fitting; the GLMM caller should document which.

---

## Verified clean (no findings)

The following were checked end-to-end and reproduce within rounding:

- **Symbolic forms of `eq:strongreject-binary`, `eq:asr-formula`, `eq:if-score`, `eq:spec-score`, `eq:tok-length`, `eq:token-inflation`, `eq:distance-vector`, H₁/H₂ derivative inequalities, `eq:glmm`, `eq:glmm-tokenizer`, `eq:glmm-spec-components`** — all match AGENTS.md §§9–10 in their substantive content (only the standardization-notation issue under F3/F4 is open). For `u_k = I[s_k > 0]`, `ASR = Σu_k/N`, `IF = (1/|B|)Σ I[ŷ=y]`, `SPEC = ½ z(IF) + ½ z(CONS)`, the code at `metrics/asr.py:6`, `metrics/asr.py:9`, `benchmarks/belebele.py:111`, `metrics/spec.py:24` matches.
- **All 8 aligned-language Wilson 95% CIs** (`tab:aligned-language-baseline-asr`) reproduce from per-language retained `n` (apendice_c) and the displayed ASR. Largest CI-bound discrepancy: 0.05 pp (rounding only).
- **All 8 cross-lingual pooled Wilson 95% CIs** (`tab:overall-asr`) reproduce after subtracting the aligned-language n and the documented per-model excluded/unscored counts. Largest discrepancy: 0.05 pp.
- **All 14 odds ratios** across `tab:glmm-main-effects`, `tab:glmm-tokenizer-robustness`, `tab:glmm-strata-effects`, `tab:glmm-spec-components`, `tab:glmm-aggregated-sensitivity` reproduce as `exp(estimate)` to 3 decimal places.
- **Sample-size accounting:** `8 × 4,021 − 78 − 312 = 31,778` (SPEC decomposition, aggregated cell-level), and `weak n = 15,766 + strong n = 16,012 = 31,778` (strata fits). `103 cells = 8 models × 13 languages − 1 missing cell (LLaMAntino-2 × Russian)`.
- **Collinearity diagnostics** (`tab:glmm-collinearity`): `VIF = 1.107` implies `|r| ≈ 0.311`; `κ = sqrt((1+|r|)/(1−|r|)) = 1.379 ≈ 1.380` — the printed `κ` matches.
- **Translation-retention arithmetic** in [aftertext/apendice_c.tex:14–22](../../ufscthesisx/ufscthesisx/aftertext/apendice_c.tex#L14): `313 × 13 − 48 = 4,021` ✓; `4,021 × 9 = 36,189` ✓; per-language `313 − excluded = retained` for all 13 rows; column totals (148 BLASER flags, 1,077 round-trip flags, 122 round-trip fails, 48 excluded, 4,021 retained) all add up.
- **LLM-only translation QC table** ([results.tex:82–106](../../ufscthesisx/ufscthesisx/chapters/results.tex#L82)): per-language `Pass + Exclude = Queue` for all 13 rows.
- **Falsification matrix outcome:** Bulgarian (No), Italian (No), Portuguese (No), Swedish (Yes) — 1 of 4 ✓ (under same-sign definition; see F1).

## How to re-verify

The following one-liners reproduce the bulk of the arithmetic checks. Run from the repo root.

```bash
# Wilson CI spot-check for Sagui-7B Portuguese (n=312, k=287, p=92.0%)
python3 -c "import math; n,k=312,287; p=k/n; z=1.96; d=1+z*z/n; c=(p+z*z/(2*n))/d; m=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/d; print(f'[{(c-m)*100:.2f}, {(c+m)*100:.2f}]')"
# Expect [88.44, 94.51]; thesis prints [88.40, 94.50].

# Odds ratios from tab:glmm-main-effects
python3 -c "import math; print(math.exp(-0.121), math.exp(1.146))"
# Expect 0.886, 3.146; thesis prints 0.886, 3.147.

# Counterfactual gap discrepancy (F5)
python3 -c "print(90.4 - 19.6)"
# Prints 70.8 (thesis prints 70.7).
```

The full Python recomputation block used to drive this audit is reproduced inline in the audit transcript and can be re-run on demand.
