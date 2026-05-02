# Thesis Evaluation TODO

Current phase: pipeline foundation and tiny pilot scaffolding.

## Now

- [x] Adopt `AGENTS.md` as the implementation source of truth.
- [x] Add CLI foundation for config validation, URIEL preparation, translation/QC, target generation, judge batch export/import, majority vote, pilot export, and result export.
- [x] Add tests for deterministic core behavior.
- [x] Add RTX 5070 Ti Docker profile.
- [x] Rewrite `eval/README.md` with usage instructions.
- [x] Add provider-shaped judge batch exports and result normalization.
- [x] Add XSTS audit queue import/export hooks.
- [x] Add reference-subset spBLEU attachment hook.
- [x] Add MacBook/RTX runtime profiles and backend selection.
- [x] Add StrongREJECT importer and two-prompt pilot prompt file path.
- [x] Add modular asset manifest and downloader/status commands.
- [x] Add modular asset verification command.
- [x] Move `urielplus` out of the default install path because its TensorFlow Addons dependency does not resolve on Python 3.12+.
- [x] Add larger NLLB translator assets and fix the distilled NLLB download to include weights.
- [x] Add real BLASER 2.0 QE scorer path behind `translation-qc --scorer blaser`.
- [x] Make the QC extra self-contained by including PyTorch for BLASER/SONAR.
- [x] Pin runtime to Python 3.12 because fairseq2/SONAR does not support Python 3.13 wheels yet.
- [x] Add progress logging and explicit SONAR prewarm command for BLASER QC.
- [x] Make target generation prefer local downloaded model assets and expose `--model-path`.
- [x] Add GPU-aware NLLB translation flags and keep large translator checkpoints loaded across a run.
- [x] Document the ordered RTX Sabiá end-to-end pilot commands in `README.md`.
- [x] Rewrite `README.md` around the Windows host + RTX Docker workflow, with pilot and final end-to-end runbooks.
- [x] Let one NLLB translation command write multiple target languages so checkpoint shards load once per batch.
- [x] Let BLASER QC and target generation batch multiple language files per model load.
- [x] Add target-model chat-template visibility and apply chat templates in vLLM when available.
- [ ] Calibrate BLASER pass/flag/fail thresholds on the audited subset.
- [ ] Run `download-assets --name strongreject` on the MacBook or RTX machine to populate `eval/data/raw/strongreject/`.
- [ ] Download NLLB, BLASER, MultiJail, BELEBELE, and target models slowly with `download-assets`.
- [ ] Run `verify-assets` after each downloaded asset.
- [ ] Run the no-inference 2-prompt pilot on the MacBook Pro.
- [ ] Try optional MacBook Sabiá 2-prompt generation smoke test with `scripts/macbook_sabia_generation_smoke.sh`.
- [ ] Run a real 2-prompt pilot through local vLLM target inference on the RTX machine.
- [ ] Batch query all three API judges and import their results.
- [ ] Add human XSTS audit import examples.

## Blocked / Needs External Inputs

- Real API judge execution needs credentials and provider-side batch jobs.
- Real NLLB/BLASER execution needs model downloads on the GPU machine.
- Human XSTS audit decisions require bilingual reviewer input.

## Later

- Scale from the tiny pilot to all target models and all 13 attack languages.
- Compute BELEBELE IF/CONS and SPEC scores.
- Add tokenizer diagnostics with each model's exact chat template.
- Freeze `eval/outputs/dataset_frozen.parquet` only after the full run is complete.
