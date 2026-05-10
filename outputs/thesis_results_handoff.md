# Thesis Results Handoff

This note summarizes the final empirical result set currently built under `eval/outputs/`. It is intended as compact context for a writing model that will finish the thesis Results and Discussion sections.

## Final Dataset Status

Treat `outputs/dataset_frozen.jsonl` and `outputs/dataset_frozen.parquet` as the final analysis dataset, subject to the caveats below.

Sanity checks passed:

- `outputs/scored/all_strongreject_scores.jsonl`: 36,189 rows.
- `outputs/dataset_frozen.jsonl`: 36,189 rows.
- `outputs/dataset_frozen.parquet`: 36,189 rows.
- JSONL and parquet row counts match.
- No duplicate `run_id` values.
- No duplicate `(model, attack_language, prompt_id)` rows.
- All 8 thesis target models have all 13 attack languages.
- The reference model `llama3_1_8b_reference` is also present for all 13 languages.
- Canonical scored files match their corresponding `outputs/backtranslated/*.jsonl` files in row count and order.
- `unsafe_binary` is consistent with `strongreject_score > 0`.
- All non-null StrongREJECT scores are in `[0, 1]`.
- No remaining OpenAI `429 insufficient_quota` rows.

Important artifact warning:

- Two stale double-suffix files still exist in `outputs/scored/`:
  - `sabia_3_dan.backtranslated.strongreject.strongreject.jsonl`
  - `sagui_7b_dan.backtranslated.strongreject.strongreject.jsonl`
- These files are not included in the current rebuilt `all_strongreject_scores.jsonl`. Do not rebuild with a broad `*strongreject*.jsonl` glob. Use:

```zsh
cat outputs/scored/*.backtranslated.strongreject.jsonl > outputs/scored/all_strongreject_scores.jsonl
```

## Coverage and Exclusions

The final dataset contains 78 excluded/unscored rows:

- 69 rows: `bggpt_gemma_9b_swe`, `empty_model_output`.
- 6 rows: `llamantino_2_ultrachat_7b` has one `empty_model_output` each in `dan`, `eng`, `nor`, `por`, `spa`, and `swe`.
- 1 row: `ai_sweden_llama3_8b_nor`, OpenAI judge HTTP 400 safety-filter block.
- 2 rows: `sabia_3_dan`, OpenAI judge HTTP 400 safety-filter block.

These are excluded technical/scoring failures, not safe model refusals. In-model refusals remain in the denominator.

One BELEBELE/SPEC caveat:

- `llamantino_2_ultrachat_7b` in Russian has `spec_score = null` because `n_cons_comparable = 0`.
- This affects 312 harmful-response rows in the frozen dataset.
- SPEC-based analyses should either drop this pair automatically or mention that this one model-language pair lacks SPEC.

## Overall StrongREJECT Outcomes

Overall ASR by model, using rows with valid StrongREJECT scores:

| Model | n | Unsafe | ASR | Mean Score |
| --- | ---: | ---: | ---: | ---: |
| `sagui_7b` | 4021 | 2527 | 0.629 | 0.397 |
| `sabia_3` | 4019 | 322 | 0.080 | 0.049 |
| `llamantino_2_ultrachat_7b` | 4015 | 2047 | 0.510 | 0.270 |
| `llamantino_anita_8b` | 4021 | 419 | 0.104 | 0.070 |
| `gpt_sw3` | 4021 | 1076 | 0.268 | 0.154 |
| `ai_sweden_llama3_8b` | 4020 | 1464 | 0.364 | 0.219 |
| `bggpt_7b` | 4021 | 2999 | 0.746 | 0.528 |
| `bggpt_gemma_9b` | 3952 | 420 | 0.106 | 0.076 |
| `llama3_1_8b_reference` | 4021 | 488 | 0.121 | 0.069 |

Cross-lingual ASR by model, excluding each model's aligned language:

| Model | n | Unsafe | Cross-Lingual ASR | Mean Score |
| --- | ---: | ---: | ---: | ---: |
| `sagui_7b` | 3709 | 2240 | 0.604 | 0.378 |
| `sabia_3` | 3707 | 305 | 0.082 | 0.049 |
| `llamantino_2_ultrachat_7b` | 3702 | 1837 | 0.496 | 0.259 |
| `llamantino_anita_8b` | 3708 | 393 | 0.106 | 0.070 |
| `gpt_sw3` | 3711 | 892 | 0.240 | 0.132 |
| `ai_sweden_llama3_8b` | 3710 | 1239 | 0.334 | 0.194 |
| `bggpt_7b` | 3710 | 2718 | 0.733 | 0.511 |
| `bggpt_gemma_9b` | 3641 | 392 | 0.108 | 0.077 |
| `llama3_1_8b_reference` | 3708 | 480 | 0.129 | 0.072 |

High-level descriptive pattern: the weak/open-weight language-specific models often have much higher ASR than their stronger pair model. The strongest practical safety contrasts are:

- Portuguese pair: `sagui_7b` high ASR vs `sabia_3` low ASR.
- Italian pair: `llamantino_2_ultrachat_7b` high ASR vs `llamantino_anita_8b` low ASR.
- Bulgarian pair: `bggpt_7b` very high ASR vs `bggpt_gemma_9b` low ASR.
- Swedish pair: `gpt_sw3` lower than the weak Portuguese/Italian/Bulgarian models, but `ai_sweden_llama3_8b` is not uniformly safer than `gpt_sw3`.

## Main GLMM Result

Main model output from `outputs/tables/glmm_main_effects.csv`:

| Predictor | Estimate | Std. Error | Odds Ratio | Fit |
| --- | ---: | ---: | ---: | --- |
| `distance_z` | -0.121 | 0.0149 | 0.886 | `statsmodels_binomial_bayes_mixed_glm_vb` |
| `spec_z` | 1.146 | 0.0150 | 3.147 | `statsmodels_binomial_bayes_mixed_glm_vb` |

Important warning:

- The fit emitted `UserWarning: VB fitting did not converge`.
- The estimates are finite, but the thesis should not present them as a perfectly converged definitive GLMM.
- Phrase as an approximate mixed-effects fit with a convergence warning, and rely heavily on the preregistered robustness/sign-retention check.

Collinearity diagnostics are clean:

- VIF for distance: 1.107.
- VIF for SPEC: 1.107.
- Condition number: 1.380.
- No single-predictor rerun was triggered.

Interpretation:

- The pooled distance coefficient is slightly negative, not positive.
- The pooled SPEC coefficient is positive, not protective.
- Because collinearity is low, these signs are not explained by distance/SPEC collinearity.

## Hypothesis Interpretation

### H1: Distance Increases ASR

Not supported.

The preregistered weak/strong pair sign-retention check retained the distance-slope sign in only 1 of 4 language pairs:

| Pair Language | Weak Model Slope | Strong Model Slope | Sign Retained |
| --- | ---: | ---: | --- |
| Bulgarian | -0.660 | 0.097 | No |
| Italian | -0.338 | 0.164 | No |
| Portuguese | -0.807 | 0.376 | No |
| Swedish | -0.667 | -0.772 | Yes |

The generated falsification summary says:

```text
pairs_evaluated = 4
pairs_with_slope_sign_retained = 1
panel_complete = True
interpretation = h1_not_supported_in_strong_alignment_regime
```

Recommended prose:

> The preregistered distance hypothesis was not supported. Distance effects were heterogeneous across model families and alignment poles, and only the Swedish pair retained the distance-slope sign across weak and strong models.

### H2: Specialization Reduces ASR

Not supported in the simple protective sense.

The pooled approximate mixed model estimates a positive SPEC coefficient: OR = 3.147 for one standardized SPEC unit. Descriptive SPEC-ASR correlations are mixed by model:

- Positive: `ai_sweden_llama3_8b`, `bggpt_7b`, `gpt_sw3`, `llamantino_2_ultrachat_7b`, `sagui_7b`.
- Negative or near zero: `bggpt_gemma_9b`, `llamantino_anita_8b`, `sabia_3`, `llama3_1_8b_reference`.

Recommended prose:

> BELEBELE-derived specialization did not behave as a simple protective factor. Higher SPEC was positively associated with unsafe compliance in the pooled approximate mixed model, while model-level correlations were heterogeneous.

### H3: Distance Remains Stronger Than Specialization

Not supported as originally expected.

Distance does not show the expected stable positive effect, and SPEC does not show the expected negative/protective effect. The better conclusion is not that one preregistered predictor dominated the other, but that the predictor story is more model-family-dependent than expected.

Recommended prose:

> The joint model did not support the preregistered account in which distance is the stronger and more stable positive predictor after controlling for specialization. Instead, both predictors showed signs contrary to the original directional expectations in the pooled fit, and the preregistered distance-slope retention criterion failed.

## Descriptive Correlations

Distance-ASR Spearman correlations by model:

- `ai_sweden_llama3_8b`: -0.560
- `bggpt_7b`: -0.357
- `bggpt_gemma_9b`: 0.418
- `gpt_sw3`: -0.681
- `llama3_1_8b_reference`: 0.341
- `llamantino_2_ultrachat_7b`: -0.724
- `llamantino_anita_8b`: 0.162
- `sabia_3`: 0.522
- `sagui_7b`: -0.533

SPEC-ASR Spearman correlations by model:

- `ai_sweden_llama3_8b`: 0.731
- `bggpt_7b`: 0.670
- `bggpt_gemma_9b`: -0.071
- `gpt_sw3`: 0.841
- `llama3_1_8b_reference`: -0.610
- `llamantino_2_ultrachat_7b`: 0.588, n = 12 because Russian SPEC is null
- `llamantino_anita_8b`: -0.330
- `sabia_3`: -0.665
- `sagui_7b`: 0.857

These are descriptive only and should support figures, not be framed as inferential tests.

## Suggested Results Narrative

The final thesis result is not that linguistic distance straightforwardly causes jailbreak vulnerability. The stronger and more defensible result is:

1. Cross-lingual jailbreak success varies strongly by model family and post-training regime.
2. Several weaker/open-weight language-specific models show high unsafe compliance, especially `bggpt_7b`, `sagui_7b`, and `llamantino_2_ultrachat_7b`.
3. Their stronger paired models are generally much safer, especially `bggpt_gemma_9b`, `sabia_3`, and `llamantino_anita_8b`.
4. Typological distance does not show the preregistered stable positive relationship with ASR.
5. BELEBELE/SPEC does not act as a simple protective capability variable.
6. The evidence points toward model-specific alignment and post-training differences dominating over a clean language-distance mechanism.

## Writing Caveats

Use these caveats explicitly:

- The GLMM variational Bayes fit did not converge, so regression estimates should be interpreted cautiously.
- The inference should lean on the preregistered sign-retention check and descriptive tables.
- The Russian SPEC value for `llamantino_2_ultrachat_7b` is missing because CONS could not be computed.
- Empty model outputs and judge safety-filter blocks are excluded as technical/scoring failures, not treated as safe responses.
- BELEBELE is being used as a benign multiple-choice specialization/capability proxy. It should not be over-described as a matched benign generative-control experiment.

## Core Files for the Writing Model

- Final dataset JSONL: `outputs/dataset_frozen.jsonl`
- Final dataset parquet: `outputs/dataset_frozen.parquet`
- Merged StrongREJECT scores: `outputs/scored/all_strongreject_scores.jsonl`
- BELEBELE SPEC table: `outputs/spec/belebele_spec.jsonl`
- Report tables: `outputs/tables/`
- Most important tables:
  - `outputs/tables/results_coverage.csv`
  - `outputs/tables/asr_by_model_language.csv`
  - `outputs/tables/crosslingual_asr_by_model_language.csv`
  - `outputs/tables/distance_asr_correlation.csv`
  - `outputs/tables/spec_asr_correlation.csv`
  - `outputs/tables/glmm_main_effects.csv`
  - `outputs/tables/glmm_collinearity.csv`
  - `outputs/tables/prereg_distance_slope_retention.csv`
  - `outputs/tables/prereg_falsification_summary.csv`

