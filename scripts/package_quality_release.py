"""Package a local correction supplement from an explicit research-file allowlist."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[2];E=ROOT/'eval';T=ROOT/'ufscthesisx/ufscthesisx'
REL=E/'outputs/releases/2026-09-09-corrected';EV=ROOT/'ufscthesisx/paper-reviews/quality-corrections-2026-09-09-evidence'
def main():
 files=set()
 def add(path):
  if path.is_file():files.add(path.resolve())
 for dirname in ['src','tests','configs']:
  for p in (E/dirname).rglob('*'):
   if p.suffix in ['.py','.yaml','.yml','.json','.toml']:add(p)
 for name in ['build_quality_corrected_release.py','verify_quality_release.py','write_quality_latex_assets.py','write_quality_descriptive_assets.py','write_belebele_quality_appendix.py','write_thesis_results_latex_assets.py','write_mechanistic_latex_assets.py','draw_model_lineage.py','prepare_quality_audit_queues.py','package_quality_release.py']:add(E/'scripts'/name)
 for name in ['pyproject.toml','uv.lock','README.md']:add(E/name)
 for p in REL.rglob('*'):
  if p.is_file() and p.name not in ['package-manifest.json','package.sha256']:add(p)
 for name in ['dataset_frozen.jsonl','dataset_frozen.parquet','spec/all_belebele_predictions.jsonl','translations/all_audited_translations.jsonl']:add(E/'outputs'/name)
 dialects=set()
 for line in (E/'outputs/spec/all_belebele_predictions.jsonl').open():dialects.add(json.loads(line)['dataset_dialect'])
 for dialect in dialects:add(E/f'assets/datasets/belebele/data/{dialect}.jsonl')
 for p in (E/'outputs/audit').glob('*.xsts_decisions.csv'):add(p)
 for p in (E/'outputs/mechanistic/results').rglob('*'):
  if p.suffix in ['.csv','.jsonl','.npz']:add(p)
 add(ROOT/'AGENTS.md')
 # Include the precise local TeX inputs recorded by the successful build.
 for line in (EV/'build/main.fls').read_text().splitlines():
  if not line.startswith('INPUT '):continue
  p=Path(line[6:]);p=p if p.is_absolute() else T/p
  p=p.resolve()
  if p.is_relative_to(T) and p.suffix in ['.tex','.bib','.pdf','.png','.jpg','.jpeg','.svg','.sty','.cls','.def','.bbx','.cbx','.bst']:add(p)
 for p in (T/'figures').glob('model-lineage.*'):add(p)
 add(EV/'thesis-corrected-2026-09-09.pdf')
 add(ROOT/'ufscthesisx/paper-reviews/quality-corrections-2026-09-09.md')
 for name in ['verification.log','tests.log','build.log','visual-inspection.json']:add(EV/name)
 entries=[dict(path=str(p.relative_to(ROOT)),size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)]
 manifest=dict(version='2026-09-09-corrected',publication_status='local_unpublished',files=entries,
               exclusions=['model weights','credentials','billing records','unavailable raw activations','claims of completed human review'])
 (REL/'package-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 archive=E/'outputs/releases/quality-corrections-2026-09-09.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(files):z.write(p,str(p.relative_to(ROOT)))
  z.write(REL/'package-manifest.json','package-manifest.json')
 with zipfile.ZipFile(archive) as z:assert z.testzip() is None
 digest=hashlib.sha256(archive.read_bytes()).hexdigest()
 (REL/'package.sha256').write_text(digest+'  '+archive.name+'\n')
 print(f'Wrote {archive}: {len(files)} files, {archive.stat().st_size:,} bytes; ZIP integrity verified')
if __name__=='__main__':main()
