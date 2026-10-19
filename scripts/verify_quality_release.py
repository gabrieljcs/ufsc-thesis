"""Independent arithmetic and preservation checks for the corrected release."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
R=Path(__file__).resolve().parents[1];O=R/'outputs/releases/2026-09-09-corrected'
def main():
 manifest=json.loads((O/'manifest.json').read_text())
 for path,expected in manifest['original_input_sha256'].items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==expected,path
 old=pd.read_parquet(R/'outputs/dataset_frozen.parquet');new=pd.read_parquet(O/'dataset_corrected.parquet')
 changed=set(manifest['changed_frozen_fields'])
 pd.testing.assert_frame_equal(old.drop(columns=list(changed)),new.drop(columns=list(changed)),check_dtype=False)
 assert not new.duplicated(['model','attack_language','prompt_id','benchmark']).any()
 valid=new[(~new.excluded)&new.unsafe_binary.notna()]
 assert ((valid.strongreject_score>0).astype(int)==valid.unsafe_binary).all()
 assert valid.strongreject_score.between(0,1).all()
 cells=valid.groupby(['model','attack_language']).agg(asr=('unsafe_binary','mean'),distance=('distance','first'),spec_score=('spec_score','first'))
 csv=pd.read_csv(O/'tables/asr_by_model_language.csv').set_index(['model','attack_language'])
 assert np.max(np.abs(cells.asr-csv.asr))<1e-12
 for pred,filename in [('distance','distance_asr_correlation.csv'),('spec_score','spec_asr_correlation.csv')]:
  table=pd.read_csv(O/'tables'/filename).set_index('model')
  for model,group in cells.groupby(level=0):assert abs(spearmanr(group[pred],group.asr).statistic-table.loc[model,'spearman_rho'])<1e-12
 # Reconstruct canonical joins from raw JSON, independently of the migration helper.
 predictions=[json.loads(line) for line in (R/'outputs/spec/all_belebele_predictions.jsonl').open()]
 sources={}
 for row in predictions:
  dialect=row['dataset_dialect']
  if dialect not in sources:sources[dialect]=[json.loads(line) for line in (R/'assets/datasets/belebele/data'/f'{dialect}.jsonl').open()]
 keyed={}
 for row in predictions:
  src=sources[row['dataset_dialect']][row['belebele_row_index']-1]
  key=(src['link'],src.get('split'),src['question_number'])
  keyed[row['model'],row['language'],key]=row['predicted_choice']
 meta=new.groupby('model').aligned_language.first();coverage=pd.read_csv(O/'tables/belebele_coverage.csv').set_index(['model','attack_language'])
 for (model,lang),group in pd.DataFrame(predictions).groupby(['model','language']):
  correct=int((group.predicted_choice==group.gold).sum());agree=denom=0
  for row in group.to_dict('records'):
   src=sources[row['dataset_dialect']][row['belebele_row_index']-1];key=(src['link'],src.get('split'),src['question_number'])
   a=keyed[model,lang,key];b=keyed[model,meta[model],key]
   if a in ['A','B','C','D'] and b in ['A','B','C','D']:denom+=1;agree+=a==b
  actual=coverage.loc[model,lang]
  assert actual.n_correct==correct and actual.n_cons_comparable==denom
  assert abs(actual.if_score-correct/900)<1e-12 and abs(actual.cons_score-agree/denom)<1e-12
 for path in (O/'tables').glob('glmm*.csv'):
  table=pd.read_csv(path)
  if {'estimate','odds_ratio'}.issubset(table):assert np.max(abs(np.exp(table.estimate)-table.odds_ratio))<1e-10,path
  if 'converged' in table:assert table.converged.all() and (table.covariance_min_eigenvalue>0).all(),path
 counts=json.loads((O/'manifest.json').read_text())
 result=dict(original_hashes_unchanged=True,unchanged_target_outcomes=True,unique_keys=True,
             n_rows=len(new),n_scored=len(valid),n_specialization_cells=len(coverage),
             canonical_cons_independently_verified=True,if_unchanged=True,asr_exports_match=True,
             spearman_exports_match=True,odds_ratio_arithmetic_matches=True,all_exported_fits_converged=True)
 (O/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
