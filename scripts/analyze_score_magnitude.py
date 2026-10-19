"""Run from eval: PYTHONPATH=src .venv/bin/python scripts/analyze_score_magnitude.py."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import scipy
import statsmodels
from thesis_eval.analysis.magnitude import summarize, fit_outcomes
ROOT = Path(__file__).resolve().parents[1]
MAIN = ('distance','spec_score')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/releases/2026-09-29-score-magnitude')
    args=parser.parse_args(); out=args.output_dir.resolve()
    if out.exists(): raise SystemExit('Refusing to overwrite release; choose a new --output-dir')
    out.mkdir(parents=True)
    source=ROOT/'outputs/releases/2026-09-09-corrected'
    inputs=[source/'dataset_corrected.jsonl',ROOT/'outputs/dataset_frozen.jsonl',source/'tables/belebele_coverage.csv',source/'tables/belebele_questionable_item_sensitivity.csv']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    rows=pd.read_json(inputs[0],lines=True)
    scored=rows.loc[~rows.excluded & rows.strongreject_score.notna()].copy()
    assert scored.strongreject_score.between(0,1).all()
    assert (scored.unsafe_binary == (scored.strongreject_score>0)).all()
    paired=scored.loc[scored.model_alignment_pole.isin(['weak','strong'])].copy()
    summaries=[]
    for model, group in [('paired_panel',paired),*scored.groupby('model',sort=True)]:
        for scope in ('all_languages','non_aligned','aligned'):
            g=group if scope=='all_languages' else group.loc[(group.attack_language!=group.aligned_language) if scope=='non_aligned' else (group.attack_language==group.aligned_language)]
            summaries.append(dict(model=model,scope=scope,**summarize(g)))
    pd.DataFrame(summaries).to_csv(out/'model_summary.csv',index=False)
    cells=[]
    for (model, language),group in scored.groupby(['model','attack_language']):
        cells.append(dict(model=model,attack_language=language,distance=group.distance.iloc[0],**summarize(group)))
    pd.DataFrame(cells).to_csv(out/'cell_summary.csv',index=False)
    bounds=[]
    for model,group in rows.groupby('model'):
        valid=group.loc[~group.excluded & group.strongreject_score.notna()]
        missing=group.loc[group.strongreject_score.isna() & group.model_output.fillna('').str.strip().ne('')]
        denominator=len(valid)+len(missing)
        bounds.append(dict(model=model,scored_n=len(valid),nonempty_unscored_n=len(missing),lower=valid.strongreject_score.sum()/denominator,upper=(valid.strongreject_score.sum()+len(missing))/denominator))
    pd.DataFrame(bounds).to_csv(out/'unknown_score_bounds.csv',index=False)
    coverage=pd.read_csv(inputs[2]); clean=pd.read_csv(inputs[3]); old=pd.read_json(inputs[1],lines=True)
    oldkeys=set(map(tuple,old.loc[old.spec_score.notna(),['model','attack_language']].to_numpy()))
    keys=list(zip(paired.model,paired.attack_language))
    enough=set(map(tuple,coverage.loc[coverage.n_cons_comparable>=100,['model','attack_language']].to_numpy()))
    cleaned=paired.drop(columns=['spec_score','cons_score']).merge(clean[['model','attack_language','spec_score','cons_score']],on=['model','attack_language'],validate='many_to_one')
    token=paired.dropna(subset=[*MAIN,'token_inflation'])
    specs=[('pooled',paired,MAIN),('weak',paired.loc[paired.model_alignment_pole=='weak'],MAIN),('strong',paired.loc[paired.model_alignment_pole=='strong'],MAIN),
        ('non_aligned',paired.loc[paired.attack_language!=paired.aligned_language],MAIN),
        ('cons_at_least_100',paired.loc[[k in enough for k in keys]],MAIN),
        ('original_complete_cohort',paired.loc[[k in oldkeys for k in keys]],MAIN),
        ('questionable_items_removed',cleaned,MAIN),('components',paired,('distance','if_score','cons_score')),
        ('tokenizer_cohort_main',token,MAIN),('tokenizer',token,(*MAIN,'token_inflation')),
        ('translation_caveat_languages_removed',paired.loc[~paired.attack_language.isin(['fin','swa','ukr'])],MAIN),
        ('blaser_flagged_removed',paired.loc[paired.translation_blaser_status!='flag'],MAIN)]
    effects=[]; diagnostics=[]
    for name,data,predictors in specs:
        data=data.dropna(subset=list(predictors)).reset_index(drop=True)
        print(f'Fitting {name}: {len(data)} rows',flush=True)
        e,d,trigger=fit_outcomes(data,predictors,name); effects+=e; diagnostics+=d
        if trigger:
            for p in predictors:
                e,d,_=fit_outcomes(data,(p,),name+'_single_'+p); effects+=e; diagnostics+=d
        pd.DataFrame(effects).to_csv(out/'effects.csv',index=False)
        (out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    for p in inputs: assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)]
    manifest=dict(status='complete',post_review=True,source_sha256=hashes,n_scored=len(scored),n_paired=len(paired),
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,statsmodels=statsmodels.__version__,
        bootstrap=dict(seed=20260929,draws=2000,unit='prompt_id',interval='percentile 95%'),
        analysis='Fractional logit GEE with model/language FE and prompt-cluster robust covariance; same-design binary comparator; positive-only exploratory outcome',
        limitations=['Conditional on observed models/languages; cross-prompt residual model-language dependence not captured by prompt clustering',
        'Positive-only cohort is selected by outcome; not a causal intensity effect','Multiple exploratory fits; intervals unadjusted for multiplicity',
        'Graded rubric scores are not calibrated probabilities or direct measures of real-world harm','Historical scoring/translation limitations remain'],
        script_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'src/thesis_eval/analysis/magnitude.py']})
    manifest['output_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); print('Complete',flush=True)

if __name__=='__main__': main()
