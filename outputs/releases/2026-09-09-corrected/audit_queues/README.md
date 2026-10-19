# Pending human validation

These are worklists, not evidence that an audit has been completed. All review fields are blank.

- Prompt queue: 1582 rows. Deterministic 12% within each language plus all saved BLASER/round-trip flags and exclusions. This expands the historical LLM-only queue; it does not retrofit human decisions.
- Response queue: 1925 rows. Deterministic 5% (ceiling; minimum one) within model × language × primary binary outcome. Reviewers should not open the separate primary-label file before their in-language judgments. Record bilingual competence and adjudicate disagreement.
- Planned upstream exclusions: 432 model/prompt/language combinations, without fabricated run IDs or generated responses.

Before execution, agree on bilingual reviewer assignment, score rubric, and disagreement resolution. Report model-wise audit n, both unsafe percentages, paired disagreement and uncertainty. Because ceiling/minimum sampling changes inclusion probabilities, use the supplied stratum sizes/probabilities for population summaries; also report the raw audited-sample disagreement against the planned 15% rule. Prompt fidelity review does not replace response review. No trusted human reference translations or spBLEU calibration have been generated.
