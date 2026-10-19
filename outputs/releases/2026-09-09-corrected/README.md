# Corrected thesis analysis — 2026-09-09

This is a local, versioned correction supplement. It has not been published under a new DOI. The original Zenodo deposit (10.5281/zenodo.20113861) contains the original frozen data and earlier analysis exports, not this corrected analysis.

The original `outputs/dataset_frozen.jsonl` and `.parquet` are immutable inputs. `manifest.json` records their checksums, the saved BELEBELE predictions, and the exact fields changed in the separate corrected data. The repair changes question matching for CONS, globally standardized SPEC, and unmeasured context-risk defaults. It does not change target generations, harmfulness scores, IF, or benchmark gold labels.

## Reproduce locally

From `eval/`, using the project environment with its analysis dependencies:

```sh
PYTHONPATH=src .venv/bin/python scripts/build_quality_corrected_release.py
PYTHONPATH=src .venv/bin/python scripts/verify_quality_release.py
.venv/bin/python scripts/write_quality_latex_assets.py
PYTHONPATH=src .venv/bin/python scripts/write_quality_descriptive_assets.py
PYTHONPATH=src .venv/bin/python scripts/write_belebele_quality_appendix.py
.venv/bin/python scripts/write_thesis_results_latex_assets.py --outputs-dir outputs/releases/2026-09-09-corrected
PYTHONPATH=src .venv/bin/python scripts/write_mechanistic_latex_assets.py
PYTHONPATH=src .venv/bin/python scripts/prepare_quality_audit_queues.py
python scripts/draw_model_lineage.py
```

The release builder refuses an existing corrected release unless `--refresh-derived` is explicitly supplied. The original frozen paths are never output targets. `draw_model_lineage.py` requires ReportLab; the other analysis scripts use the locked analysis environment. No reproduction step above calls a model or paid API. A fresh installation needs the relevant packages and saved input artifacts; the lock file pins the known StrongREJECT commit. It does not establish which judge processed historical responses.

The primary corrected panel has 32,090 scored paired-model responses and 117 defined specialization cells in the global standardization universe. The English reference is excluded from paired regressions. Distance is negative in the full panel but positive after restricting CONS to at least 100 comparisons per cell (which removes the entire weak Italian model). The aggregated cell analysis also gives a small positive, uncertain distance coefficient. These sensitivities are part of the reported result.

## Evidence and limitations

- `tables/`: descriptive, full-model, stratum, decomposition, tokenizer, GEE, prior, collinearity, single-predictor and post-review sensitivities. Diagnostic CSVs preserve optimizer status, gradients and positive-definite covariance checks.
- `canonical_prediction_mapping.csv`: identity sidecar for the unchanged 105,300 saved predictions.
- `tables/belebele_coverage.csv`: all 117 cells, with parsing coverage, CONS denominators and Wilson intervals.
- `audit_queues/`: uncompleted human prompt and bilingual response review, plus the 432 upstream excluded potential runs. Blank reviewer fields are not audit results.
- `verification.json`: independent identity/IF/ASR/correlation/OR arithmetic and preservation checks.
- The original target API collection date, dated provider snapshot, primary judge snapshot and raw primary judge replies remain unavailable. Context usage is unmeasured.
- MultiJail external validation, trusted-reference spBLEU calibration, independent mechanistic question holdouts and intervention controls remain unavailable. Matched benign generative controls remain deferred.

The two questionable BELEBELE gold items are excluded only in a labeled sensitivity analysis. Main gold labels remain unchanged. The four-comparison weak-Italian/Russian CONS cell is included under the defined-value primary rule and exposed in the coverage sensitivity; it must not be described as a reliable estimate of perfect consistency.

Before public deposition, supply the final institutional front matter, assign a persistent identifier to this supplement, and follow the original third-party dataset licenses. The package contains saved research prompts/responses and does not include weights, credentials, billing records, or a claimed human validation.
