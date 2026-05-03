# Thesis Evaluation TODO

Current phase: reconcile the evaluation package with the updated thesis methodology.

## Immediate

- [x] Read the current thesis chapters and identify methodology changes.
- [x] Rewrite `../AGENTS.md` around the updated eight-model paired design and StrongREJECT-primary scoring path.
- [x] Update `configs/models.yaml` and `configs/languages.yaml` for all eight target models.
- [x] Update `configs/assets.yaml` for new open-weight targets and mark Sabiá-3 as API-only.
- [x] Use official StrongREJECT scoring after response back-translation.
- [x] Extend schemas for StrongREJECT score fields, provider/API metadata, round-trip translation diagnostics, and audit fields.
- [x] Add prompt round-trip back-translation diagnostics and response back-translation before scoring.
- [x] Add BLASER threshold calibration from audited subset.
- [x] Add BELEBELE IF/CONS/SPEC pipeline.
- [x] Add matched benign-control prompt and scoring pipeline.
- [x] Add frozen dataset builder for the updated schema.
- [x] Add Results chapter table/figure exporters.

## Blocked / External Inputs

- Real API credentials for Sabiá-3.
- Downloaded NLLB, BLASER, BELEBELE, MultiJail, StrongREJECT, and open-weight target assets.
- Human XSTS audit decisions and trusted reference translations for the spBLEU subset.

## Later

- Run smoke tests for each target model.
- Run full harmful and benign-control generation.
- Run MultiJail validation through the same scoring path.
- Freeze `outputs/dataset_frozen.parquet` only after full coverage checks pass.
