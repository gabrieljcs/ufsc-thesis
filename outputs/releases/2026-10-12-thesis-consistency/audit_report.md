# Thesis consistency audit

The reviewed source uses UFSC's `ufscthesisx` class, which loads a bundled abnTeX2 class, with English as its main language. This is a presentation and selected cross-reference/count consistency review, rather than an assertion that the old template alone certifies compliance with every current ABNT rule.

## Applied conventions

- Model display names use the normal document font. Monospace is reserved for exact checkpoint/API identifiers, code, saved field names, and literal serialized tokens. This is an editorial distinction; the consulted ABNT/UFSC guidance does not impose a universal monospace requirement on model names.
- Short display names are Sagui-7B, Sabiá-3, LLaMAntino-2, ANITA, GPT-SW3, AI Sweden, BgGPT-7B, and BgGPT-Gemma. The English reference is described separately. Technical spellings were checked against `eval/configs/models.yaml`.
- English decimal points are explicit in `siunitx` and PGF numeric output. The class's inherited Portuguese decimal-comma output is overridden in `settings.tex`.
- Integer counts of four or more digits use thin-space grouping, e.g. `32\,090` and `4\,021`. Four-digit grouping is a consistent local choice; SI guidance permits grouping and commonly leaves four-digit numbers ungrouped. No comma or full stop serves as the grouping separator.
- Years, version numbers, canonical prompt/question IDs, seeds, URLs, DOIs, filenames, citations, source data, plot coordinates, and RGB/HTML colour definitions retain their literal forms. Decimal precision follows the quantity and the existing generated assets, rather than padding every metric to one arbitrary precision. Percentages retain the consistent English statistical-text form `95%`.
- Figure identifications/captions appear above the illustration, with sources below. Tables retain their upper captions and lower sources. Figure floats are kept within the mechanistic section before the next subject begins.
- Bibliographic source titles, quotations, saved answer fragments, and the embedded published-paper annex are not rewritten to imitate the thesis's house style.

## AUC explanation

AUC is now expanded as area under the receiver operating characteristic curve before the methodology workflow uses it. The paragraph explains ROC thresholds, pairwise ranking, chance and perfect ranking, reversed ranking, half credit for ties, and the distinction from refusal prevalence or a calibrated probability. AUC, ROC and MLP are included in the acronym list. Hanley and McNeil (1982) is cited for the ranking interpretation.

The real Danish AI Sweden example has 24 refused and 24 unsafe rows, and saved AUC 0.9201388888888888 (displayed as 0.920). Its 576 cross-group comparisons illustrate ranking within the reused sample, not an independent accuracy estimate.

## Changes and verification

- The first formatting pass changed 115 integer occurrences, 92 model display fonts, 22 display aliases, and 33 caption placements in 26 source files. These are source-level counts, including macro definitions; they are not counts of distinct rendered models or figures.
- Follow-up passes changed three braced-comma integer occurrences in Appendix C equations. The numeric values are unchanged.
- The final visual pass also removed monospace from the generic reference label and separated the transfer legend entries into two readable rows. The figure generator's legend definition was updated without rerunning its analysis/export pipeline.
- The results' description of the English reference as a base model was corrected: its configured endpoint is `meta-llama/Llama-3.1-8B-Instruct`.
- The configured Bulgarian Gemma checkpoint is `INSAIT-Institute/BgGPT-Gemma-2-9B-IT-v1.0`; technical references now use that leaf identifier. GPT-SW3 and reference checkpoint casing likewise match the configured identifiers.
- Saved summary arithmetic reconciles: 313 × 13 − 48 = 4 021 attempts per model; 4 021 × 9 = 36 189 total rows; 2 492 aligned + 29 598 non-aligned = 32 090 scored paired responses.
- The normalization script checks all 48 TeX files used by the latest build plus settings and reports no further convention changes. Run `python3 eval/scripts/audit_thesis_consistency.py` after a complete build; use `--fix` as a final presentation pass after regenerating assets.
- The audit compares unchanged citation keys, references, figure labels, URLs and graphics paths against the preserved pre-normalization files. Statistical source inputs are unchanged; no generation, judging, regression or activation extraction was rerun.
- The PDF is rebuilt and the affected figure layouts are rendered for visual inspection. Final log status and output checksum are recorded in the manifest.

## Guidance consulted

- [UFSC Library, 2025 tutorial applying ABNT NBR 14724:2024](https://repositorio.ufsc.br/handle/123456789/180829): upper illustration identification, lower source, first-use acronym expansion, and graphical uniformity.
- [BIPM, 22nd CGPM Resolution 10 (2003)](https://www.bipm.org/en/committees/cg/cgpm/22-2003/resolution-10): either decimal marker is permitted; BIPM uses the point in English; digit groups use spaces rather than points or commas.
- [abnTeX2 official documentation](https://github.com/abntex/abntex2/blob/master/doc/latex/abntex2/abntex2.tex): language support is distinct from the ABNT layout/citation framework.
- [Hanley and McNeil (1982)](https://doi.org/10.1148/radiology.143.1.7063747): area-under-ROC ranking interpretation.

The currently accessible UFSC tutorial supplies the layout/acronym rules. Numeric punctuation is explicitly documented here as an English scientific house convention. The IBGE original PDF links on the UFSC site returned HTTP 403 during this check; no unverified claim about an English-specific IBGE exception is made.
