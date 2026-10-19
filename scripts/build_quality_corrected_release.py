"""Reproduce the post-review correction from saved evidence; no inference/API calls.

Run from eval: PYTHONPATH=src .venv/bin/python scripts/build_quality_corrected_release.py
Outputs are versioned separately from the immutable original frozen dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import numpy as np

from thesis_eval.benchmarks.belebele import (
    canonical_question_id, compute_if_cons, load_belebele_rows, migrate_prediction_identities,
)
from thesis_eval.analysis.glmm import (
    MAIN_PREDICTORS, TOKENIZER_PREDICTORS, Predictor, _fit_laplace_map,
    _prepare_data, fit_glmm_suite,
)
from thesis_eval.analysis.reporting import collinearity_diagnostics
from thesis_eval.cli import cmd_export_report_tables
from thesis_eval.io import read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def join_spec(original: list[dict], spec: list[dict]) -> list[dict]:
    lookup = {(r['model'], r['attack_language']): r for r in spec}
    result = []
    for original_row in original:
        row = dict(original_row)
        value = lookup[(row['model'], row['attack_language'])]
        for key in ('if_score', 'cons_score', 'spec_score'):
            row[key] = value[key]
        # An unmeasured default false is not an observed absence of risk.
        if row.get('context_usage_ratio') is None:
            row['truncation_risk'] = None
        result.append(row)
    return result


def aggregated_sensitivity(rows: list[dict]) -> list[dict]:
    import patsy
    import statsmodels.api as sm
    data = _prepare_data(rows, MAIN_PREDICTORS)
    cells = data.groupby(['model','attack_language']).agg(
        unsafe=('unsafe_binary','sum'), n=('unsafe_binary','size'),
        distance=('distance','first'), spec_score=('spec_score','first')).reset_index()
    for source, target in [('distance','distance_z'),('spec_score','spec_z')]:
        cells[target]=(cells[source]-cells[source].mean())/cells[source].std(ddof=0)
    design=patsy.dmatrix('distance_z + spec_z + C(model)',cells,return_type='dataframe')
    fit=sm.GLM(np.column_stack([cells.unsafe,cells.n-cells.unsafe]),design,
               family=sm.families.Binomial()).fit(cov_type='HC1')
    return [dict(fit='aggregated_cell_glm_hc1',predictor=k,estimate=float(fit.params[k]),
                 std_error=float(fit.bse[k]),odds_ratio=float(np.exp(fit.params[k])),
                 n_cells=len(cells),n_responses=int(cells.n.sum())) for k in ('distance_z','spec_z')]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/releases/2026-09-09-corrected')
    parser.add_argument('--skip-fits', action='store_true')
    parser.add_argument('--refresh-derived', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if (out/'dataset_corrected.jsonl').exists() and not args.refresh_derived:
        raise SystemExit('Release already exists; use --refresh-derived to explicitly refresh derived outputs')
    out.mkdir(parents=True, exist_ok=True)
    tables = out/'tables'; tables.mkdir(exist_ok=True)
    original_path = ROOT/'outputs/dataset_frozen.jsonl'
    parquet_path = ROOT/'outputs/dataset_frozen.parquet'
    predictions_path = ROOT/'outputs/spec/all_belebele_predictions.jsonl'
    input_hashes = {str(p.relative_to(ROOT)): digest(p) for p in (original_path, parquet_path, predictions_path)}
    original = read_jsonl(original_path)
    aligned = {r['model']: r['aligned_language'] for r in original}
    predictions = migrate_prediction_identities(read_jsonl(predictions_path), ROOT/'assets/datasets/belebele')
    # The sidecar supplies a compact, auditable mapping to unchanged saved predictions.
    csv(out/'canonical_prediction_mapping.csv', [{k: r.get(k) for k in
        ('model','language','item_id','belebele_row_index','canonical_item_id','passage_link','passage_split')}
        for r in predictions])
    spec = compute_if_cons(predictions, aligned)
    write_jsonl(out/'belebele_spec.jsonl', spec)
    csv(tables/'belebele_coverage.csv', spec)
    rows = join_spec(original, spec)
    write_jsonl(out/'dataset_corrected.jsonl', rows)
    pd.DataFrame(rows).to_parquet(out/'dataset_corrected.parquet', index=False)
    unchanged = set(original[0]) - {'cons_score', 'spec_score', 'truncation_risk'}
    assert all(all(a[k] == b[k] for k in unchanged) for a, b in zip(original, rows, strict=True))
    cmd_export_report_tables(SimpleNamespace(rows=str(out/'dataset_corrected.jsonl'), output_dir=str(tables)))
    csv(tables/'glmm_aggregated_cell_sensitivity.csv', aggregated_sensitivity(rows))
    # Bounds for nonempty unscored responses; empty outputs are not fabricated outcomes.
    bounds = []
    for model in aligned:
        for scope in ('all_languages', 'nonaligned'):
            panel = [r for r in rows if r['model']==model and
                     (scope=='all_languages' or r['attack_language']!=r['aligned_language'])]
            scored = [r for r in panel if not r['excluded'] and r['unsafe_binary'] is not None]
            unknown = [r for r in panel if r['unsafe_binary'] is None and str(r.get('model_output','')).strip()]
            successes = sum(r['unsafe_binary'] for r in scored); denominator = len(scored)+len(unknown)
            bounds.append(dict(model=model,scope=scope,scored_n=len(scored),nonempty_unscored_n=len(unknown),
                unsafe_n=successes,lower_bound=successes/denominator,upper_bound=(successes+len(unknown))/denominator))
    csv(tables/'unscored_response_bounds.csv', bounds)
    # Gold anomalies are sensitivity exclusions, never silent relabeling.
    english = load_belebele_rows(ROOT/'assets/datasets/belebele', 'eng')
    flagged = {canonical_question_id(english[i-1]) for i in (150,847)}
    clean_spec = compute_if_cons([p for p in predictions if p['canonical_item_id'] not in flagged], aligned)
    csv(tables/'belebele_questionable_item_sensitivity.csv', clean_spec)
    if not args.skip_fits:
        print('Fitting corrected full suite', flush=True)
        for filename, values in fit_glmm_suite(rows).items():
            csv(tables/filename, values)
        print('Fitting correction/coverage sensitivities', flush=True)
        original_keys = {(r['model'],r['attack_language']) for r in original if r['spec_score'] is not None}
        enough = {(r['model'],r['attack_language']) for r in spec if r['n_cons_comparable']>=100}
        sensitivities = [
            ('original_complete_cell_cohort', [r for r in rows if (r['model'],r['attack_language']) in original_keys]),
            ('cons_at_least_100_comparisons', [r for r in rows if (r['model'],r['attack_language']) in enough]),
            ('excluding_two_questionable_gold_items', join_spec(original, clean_spec)),
        ]
        tokenizer_data = _prepare_data(rows, TOKENIZER_PREDICTORS)
        tokenizer_keys = set(tokenizer_data.run_id)
        sensitivities.append(('main_on_tokenizer_cohort',[r for r in rows if r['run_id'] in tokenizer_keys]))
        effects, diagnostics = [], []
        for fit, sensitivity_rows in sensitivities:
            values, diagnostic = _fit_laplace_map(_prepare_data(sensitivity_rows,MAIN_PREDICTORS), MAIN_PREDICTORS,fit=fit)
            effects.extend(values);diagnostics.append(diagnostic)
            print(f'{fit}: n={diagnostic["n"]}', flush=True)
        csv(tables/'glmm_correction_sensitivity.csv', effects)
        csv(tables/'glmm_correction_sensitivity_diagnostics.csv', diagnostics)
        diagnostics_map = {r['diagnostic']:r['value'] for r in collinearity_diagnostics(rows)}
        single, single_diagnostics = [], []
        if diagnostics_map['single_predictor_rerun_triggered']:
            for predictor in MAIN_PREDICTORS:
                values, diagnostic = _fit_laplace_map(_prepare_data(rows,(predictor,)),(predictor,),fit='single_'+predictor.source)
                single.extend(values)
                single_diagnostics.append(diagnostic)
            csv(tables/'glmm_single_predictor_sensitivity.csv', single)
            csv(tables/'glmm_single_predictor_sensitivity_diagnostics.csv', single_diagnostics)
    for relative, expected in input_hashes.items():
        assert digest(ROOT/relative)==expected, f'Original input changed: {relative}'
    manifest = dict(version='2026-09-09-corrected',original_input_sha256=input_hashes,
        canonical_key=['link','split when present','question_number'],n_rows=len(rows),
        n_specialization_cells=len(spec),inference_complete=not args.skip_fits,
        changed_frozen_fields=['cons_score','spec_score','truncation_risk'],
        unchanged='target outputs, scoring labels/components, IF and all other original run fields',
        limitations=['Human audits and MultiJail unavailable','Historical judge/provider revisions incompletely recorded',
                     'Context risk unmeasured','Mechanistic controls and independent question holdout unavailable'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Wrote corrected release: {out}', flush=True)


if __name__ == '__main__':
    main()
