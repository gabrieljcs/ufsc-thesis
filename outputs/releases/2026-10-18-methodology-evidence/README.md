# Methodology documentation release — 18 October 2026

This release documents existing evidence. It does not generate model responses,
rescore outcomes, revise translations, or repair the specialization measure.

- `excluded_translations.csv` and `.jsonl`: all 48 excluded prompt–language
  variants, with their original source, translation, round trip, BLASER score,
  XSTS score, judge identifier, and saved short judge note. Notes are joined from
  the original per-language decision CSVs, since the merged log omits them.
- `unscored_responses.csv`: the 78 unique frozen response records without a
  StrongREJECT score (75 empty responses and three nonempty judge-API blocks).
- `prompt_templates.json` and `xsts_messages.json`: the available model wrappers,
  BELEBELE prompt, and StrongREJECT/XSTS rubric messages. The readable wrappers
  are documentation reconstructions, not recovered historical request bytes.
- `source_templates/`: public model-card/tokenizer snapshots, retrieval URLs,
  checksums, and unavailable-file statuses. Retrieval date is not model-run date.
- `verification.json`: checked counts and confirmation that translation
  exclusions are absent from the frozen target-response universe.
- `manifest.json`: checksums of the immutable inputs used for the export.
- `arxiv-2406.09324v3.tar`: the original arXiv source archive for Xu, Liu, and Liu
  (2024). Figure panels and their attribution manifest are in the thesis folder
  `figures/external/xu2024/`. Original panels are reproduced unchanged.

From the repository root, regenerate the data and LaTeX appendices with:

```sh
python3 eval/scripts/write_methodology_evidence.py
```

The script uses the archived source snapshots and frozen local records; no
network connection or model/judge calls are made. XeLaTeX, Arial, and Geeza Pro
are required to regenerate the foreign-script vector snippets. A snapshot of
the generator is retained in `code/`; run the repository copy so that its paths
resolve from the repository root.

In the manuscript, the concise exclusion overview is in Appendix C, the prompt
inventory in Appendix I, and the complete exclusion/unscored ledger in Appendix J.
Appendix A contains the XSTS messages. These appendices explicitly distinguish
saved evidence from current documentation and unavailable historical provenance.
