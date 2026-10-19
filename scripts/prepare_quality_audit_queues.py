"""Prepare human review worklists from saved artifacts; no automatic adjudication."""
from pathlib import Path
from collections import defaultdict
import hashlib,json,math
import pandas as pd
from thesis_eval.io import read_jsonl
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/releases/2026-09-09-corrected/audit_queues'
def rank(row,key):return hashlib.sha256(('20260909:'+str(row[key])).encode()).hexdigest()
def main():
 OUT.mkdir(exist_ok=True)
 translations=read_jsonl(ROOT/'outputs/translations/all_audited_translations.jsonl')
 groups=defaultdict(list)
 for r in translations:groups[r['target_language']].append(r)
 selected=[];upstream=[]
 models=sorted({r['model'] for r in read_jsonl(ROOT/'outputs/dataset_frozen.jsonl')})
 for lang,rows in sorted(groups.items()):
  sample={r['prompt_id'] for r in sorted(rows,key=lambda r:rank(r,'prompt_id'))[:math.ceil(.12*len(rows))]}
  for r in rows:
   flags=[]
   if r['prompt_id'] in sample:flags.append('12pct_language_stratum')
   if r.get('blaser_status')!='pass':flags.append('blaser')
   if r.get('roundtrip_status')!='pass':flags.append('roundtrip')
   if r.get('translation_excluded'):flags.append('excluded')
   if flags:
    selected.append({**{k:r.get(k) for k in ['prompt_id','target_language','source_text','translated_text','roundtrip_text','blaser_qe_score','roundtrip_status','translation_excluded']},
                     'selection_reason':'|'.join(flags),'review_status':'not_reviewed',
                     **{k:None for k in ['reviewer','review_date','semantic_equivalence','attack_strength','constraint_preservation','naturalness','no_harmfulness_drift','harm_preservation','decision','notes']}})
   if r.get('translation_excluded'):
    for m in models:upstream.append(dict(model=m,prompt_id=r['prompt_id'],attack_language=lang,benchmark='strongreject',generation_attempted=False,exclusion_reason='upstream_translation_exclusion'))
 pd.DataFrame(selected).to_csv(OUT/'human_prompt_qc.csv',index=False)
 pd.DataFrame(upstream).to_csv(OUT/'upstream_excluded_planned_runs.csv',index=False)
 data=read_jsonl(ROOT/'outputs/dataset_frozen.jsonl');groups=defaultdict(list)
 for r in data:
  if r.get('model_output') and r.get('unsafe_binary') is not None:
   groups[r['model'],r['attack_language'],r['unsafe_binary']].append(r)
 responses=[]
 for key,rows in sorted(groups.items()):
  for r in sorted(rows,key=lambda r:rank(r,'run_id'))[:max(1,math.ceil(.05*len(rows)))]:
   responses.append({**{k:r.get(k) for k in ['run_id','prompt_id','model','attack_language','aligned_language','prompt_text','model_output','model_output_backtranslated']},
                     'review_status':'not_reviewed','stratum_population_n':len(rows),
                     'selection_probability':max(1,math.ceil(.05*len(rows)))/len(rows),
                     **{k:None for k in ['reviewer','review_date','in_language_unsafe','english_path_audit_unsafe','translation_fidelity','adjudication','notes']}})
 # Primary labels are withheld from the reviewer worksheet and kept separately for later comparison.
 ids={r['run_id'] for r in responses}
 pd.DataFrame(responses).to_csv(OUT/'bilingual_response_review.csv',index=False)
 pd.DataFrame([{k:r.get(k) for k in ['run_id','unsafe_binary','strongreject_score','refusal_binary']} for r in data if r['run_id'] in ids]).to_csv(OUT/'response_primary_labels_blinded.csv',index=False)
 (OUT/'README.md').write_text(f'''# Pending human validation\n\nThese are worklists, not evidence that an audit has been completed. All review fields are blank.\n\n- Prompt queue: {len(selected)} rows. Deterministic 12% within each language plus all saved BLASER/round-trip flags and exclusions. This expands the historical LLM-only queue; it does not retrofit human decisions.\n- Response queue: {len(responses)} rows. Deterministic 5% (ceiling; minimum one) within model × language × primary binary outcome. Reviewers should not open the separate primary-label file before their in-language judgments. Record bilingual competence and adjudicate disagreement.\n- Planned upstream exclusions: {len(upstream)} model/prompt/language combinations, without fabricated run IDs or generated responses.\n\nBefore execution, agree on bilingual reviewer assignment, score rubric, and disagreement resolution. Report model-wise audit n, both unsafe percentages, paired disagreement and uncertainty. Because ceiling/minimum sampling changes inclusion probabilities, use the supplied stratum sizes/probabilities for population summaries; also report the raw audited-sample disagreement against the planned 15% rule. Prompt fidelity review does not replace response review. No trusted human reference translations or spBLEU calibration have been generated.\n''')
 print(f'Prepared {len(selected)} prompt reviews, {len(responses)} response reviews, {len(upstream)} upstream exclusions; none adjudicated')
if __name__=='__main__':main()
