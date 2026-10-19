# StrongREJECT score magnitude: post-review extension

This extension was chosen after inspecting the binary results. It preserves the pre-specified analysis and all frozen scores. The corrected September 9 release is the input. The reference model appears only in descriptive summaries.

## What the outcomes measure

- Binary ASR: proportion with any positive score.
- Mean score: the official graded score averaged over all scored responses, including zeros.
- Positive-only mean: magnitude within a subset selected by its observed outcome; this is not a causal effect or a measure over matched successful prompts.

Mean score = ASR × positive-only mean. Scores are rubric ratings, not calibrated probabilities or quantities of real-world harm.

## Non-aligned-language results

| Model | Scored n | ASR | Mean score (95% CI) | Mean among positives |
|---|---:|---:|---:|---:|
| Sagui-7B | 3709 | 60.39% | 0.377 [0.359, 0.397] | 0.625 |
| Sabiá-3 | 3707 | 8.23% | 0.049 [0.037, 0.063] | 0.600 |
| LLaMAntino-2 | 3702 | 49.62% | 0.259 [0.243, 0.276] | 0.521 |
| ANITA-8B | 3708 | 10.60% | 0.070 [0.057, 0.085] | 0.664 |
| GPT-SW3 | 3711 | 24.04% | 0.132 [0.124, 0.141] | 0.550 |
| AI Sweden Llama-3 | 3710 | 33.40% | 0.194 [0.179, 0.209] | 0.581 |
| BgGPT-7B | 3710 | 73.26% | 0.511 [0.496, 0.526] | 0.697 |
| BgGPT-Gemma-2 | 3641 | 10.77% | 0.077 [0.064, 0.091] | 0.712 |
| Llama-3.1 reference | 3708 | 12.94% | 0.072 [0.060, 0.085] | 0.559 |

Across paired targets: 29598 scored non-aligned responses; ASR 33.84%; mean score 0.209; positive-only mean 0.618.

Intervals bootstrap source prompts as clusters (2,000 draws, fixed seed), preserving model/language rows when a prompt is sampled. The pooled means weight responses, not models equally.

## Matched regression comparison

Fractional-logit GEE models the expected score with a logistic link. Exact zero and one scores are retained. Model and attack-language treatment indicators are fixed effects; robust sandwich covariance clusters all translations/model responses of a source prompt. The binary comparator uses the same rows and design. Positive-only fits retain the parent cohort scaling. Coefficients describe logit conditional means; graded coefficients are not event odds ratios.

| Predictor | Binary | Graded score | Positive-only score |
|---|---:|---:|---:|
| distance | -0.091 [-0.130, -0.051] | -0.112 [-0.142, -0.081] | -0.114 [-0.140, -0.087] |
| spec_score | 1.029 [0.916, 1.142] | 0.954 [0.868, 1.040] | 0.387 [0.321, 0.453] |

## Nuance and robustness

The full-panel distance association stays negative and SPEC stays positive in both binary and graded models, including weak/strong strata. This does not rescue directional H1 or protective H2. The coverage restriction to cells with at least 100 comparable CONS answers reverses distance in both outcomes; it removes the entire weak Italian model, so this is also a model-composition change.

| Specification | Binary distance | Graded distance |
|---|---:|---:|
| weak | -0.194 [-0.256, -0.132] | -0.179 [-0.215, -0.143] |
| strong | -0.218 [-0.283, -0.153] | -0.190 [-0.254, -0.127] |
| non_aligned | -0.238 [-0.273, -0.204] | -0.277 [-0.303, -0.251] |
| cons_at_least_100 | 0.085 [0.042, 0.129] | 0.094 [0.057, 0.130] |
| original_complete_cohort | -0.063 [-0.103, -0.022] | -0.068 [-0.101, -0.035] |
| questionable_items_removed | -0.091 [-0.130, -0.051] | -0.112 [-0.143, -0.081] |
| tokenizer_cohort_main | -0.129 [-0.173, -0.085] | -0.144 [-0.176, -0.113] |
| tokenizer | -0.154 [-0.199, -0.110] | -0.173 [-0.207, -0.140] |
| translation_caveat_languages_removed | -0.152 [-0.205, -0.098] | -0.178 [-0.216, -0.140] |
| blaser_flagged_removed | -0.098 [-0.137, -0.058] | -0.121 [-0.152, -0.090] |

The IF/CONS decomposition shows an additional distinction. Graded outcomes have a larger standardized IF coefficient than the binary comparator. This is a descriptive comparison of fitted coefficients across different outcomes, not a formal test of their difference or a causal capability mechanism.

| Component | Binary | Graded score | Positive-only score |
|---|---:|---:|---:|
| if_score | 0.508 [0.352, 0.664] | 1.054 [0.923, 1.185] | 1.312 [1.197, 1.427] |
| cons_score | 0.571 [0.507, 0.635] | 0.453 [0.406, 0.499] | 0.104 [0.069, 0.140] |

The decomposition is highly collinear: full-design VIF is 38.41 for IF and 5.04 for CONS. This limits claims about their separate contributions. Single-predictor graded coefficients remain positive: IF 1.430 [1.295, 1.565]; CONS 0.629 [0.593, 0.664]. These omit the other substantive predictors and are not estimates of independent component effects.

All 114 fits converged with finite estimates and positive-definite robust covariance. Pooled full-design VIF: distance 2.46, SPEC 13.51. Triggered single-predictor fits are in effects.csv; their estimands omit the other substantive controls, so they do not resolve confounding.

## Limits

Inference is conditional on the observed model/language inventory. Prompt clustering does not allow arbitrary residual dependence between different prompts in the same model-language cell. These are supplementary marginal models, not replacements for the crossed-random-intercept GLMM. Predictors vary largely at model-language level; narrow prompt-cluster intervals should not be mistaken for evidence of broad language/model generalization. Intervals are exploratory and unadjusted for multiple analyses. Positive-only models condition on success and cannot identify causal changes in severity. Human translation/scoring audits remain unavailable. Three nonempty unscored responses are bounded separately in unknown_score_bounds.csv; empty outputs remain missing.

## Reproduction

Run `PYTHONPATH=src .venv/bin/python scripts/analyze_score_magnitude.py --output-dir <new-directory>` from eval. The analysis refuses to overwrite a release. The report generator uses the v2 release. Version 2 corrects positive-subset VIF diagnostics to use the exact positive-only design; the first run is retained as provenance.

[Fractional response method](https://www.nber.org/papers/t0147) · [GEE documentation](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_estimating_equations.GEE.html)
