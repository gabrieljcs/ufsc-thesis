# Inference robustness

Recomputed from the corrected frozen dataset and the completed magnitude fits. No new generation, scoring or translation. All four analyses below are post-review; only the pair sign-retention matrix was specified before results.

## Clustering

Distance and SPEC vary between 104 model-language cells, not between source prompts. Clustering on source prompts therefore cuts across the variation being tested.

| Cohort | Predictor | Prompt-clustered | Cell-clustered |
|---|---|---|---|
| pooled | distance | -0.112 [-0.142, -0.081] | -0.112 [-0.271, 0.047] |
| pooled | spec_score | 0.954 [0.868, 1.040] | 0.954 [0.514, 1.394] |
| non_aligned | distance | -0.277 [-0.303, -0.251] | -0.277 [-0.381, -0.172] |
| non_aligned | spec_score | 1.132 [1.033, 1.231] | 1.132 [0.683, 1.581] |
| cons_at_least_100 | distance | 0.094 [0.057, 0.130] | 0.094 [-0.076, 0.263] |
| cons_at_least_100 | spec_score | 1.648 [1.535, 1.762] | 1.648 [1.214, 2.082] |
| weak | distance | -0.179 [-0.215, -0.143] | -0.179 [-0.317, -0.041] |
| weak | spec_score | 0.429 [0.363, 0.495] | 0.429 [0.131, 0.726] |
| strong | distance | -0.190 [-0.254, -0.127] | -0.190 [-0.486, 0.105] |
| strong | spec_score | 0.707 [0.559, 0.854] | 0.707 [-0.046, 1.459] |

Pooled distance standard error rises from 0.016 to 0.081 and the interval covers zero. The CONS-coverage cohort, previously described as reversing the sign, also covers zero. The two cohorts therefore do not disagree in any inferentially meaningful sense. SPEC remains bounded away from zero except in the strong-pole stratum.

## Outcome decomposition

mean(score) = ASR x mean(score | score > 0): 0.221 = 0.351 x 0.628.

Across models, compliance frequency spans 0.080-0.746 while conditional magnitude spans only 0.529-0.713. Distance behaves almost identically on all three outcomes; SPEC acts chiefly on whether the model complies at all, not on how useful the compliance is.

## Distance sign tally

Distance is negative in 11 of 12 reported specifications and positive in 1. The single positive specification is the CONS-coverage restriction, which removes an entire model.

## Pre-specified pair matrix

1 of 4 pairs retain the distance-slope sign: h1_not_supported. Pairs positive at both poles (the direction H1 predicts): 0. Pairs negative at both poles: 1.
