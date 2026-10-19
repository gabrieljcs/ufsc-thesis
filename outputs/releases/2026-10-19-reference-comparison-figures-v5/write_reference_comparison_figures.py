"""Write the paired comparison figures and a separate English-reference figure.

The three paired figures show only the eight paired targets. The descriptive
English reference gets its own figure, combining its StrongREJECT, BELEBELE
and SPEC profiles. Use saved measurements only; export a new presentation release.
"""
import csv
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import re

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / 'eval/outputs/releases/2026-10-19-reference-comparison-figures-v5'
FIGURES = ROOT / 'ufscthesisx/ufscthesisx/figures'
CELLS = ROOT / 'eval/outputs/releases/2026-09-29-graded-reporting/cell_summary.csv'
COVERAGE = ROOT / 'eval/outputs/releases/2026-09-09-corrected/tables/belebele_coverage.csv'
REF = 'llama3_1_8b_reference'
PAIRS = [
 ('por', [('sagui_7b','Sagui-7B'),('sabia_3',r"Sabi\'a-3")]),
 ('ita', [('llamantino_2_ultrachat_7b','LLaMAntino-2'),('llamantino_anita_8b','ANITA-8B')]),
 ('swe', [('gpt_sw3','GPT-SW3'),('ai_sweden_llama3_8b','AI Sweden')]),
 ('bul', [('bggpt_7b','BgGPT-7B'),('bggpt_gemma_9b','BgGPT-Gemma')]),
]
rows = list(csv.DictReader(CELLS.open()))
coverage = list(csv.DictReader(COVERAGE.open()))
assert len(rows) == len(coverage) == 117
RELEASE.mkdir(parents=True, exist_ok=True)
COLORS = [r'\definecolor{srweak}{RGB}{35,103,159}',r'\definecolor{srstrong}{RGB}{202,86,42}']
PROFILE_STYLE = r'\pgfplotsset{srprofile/.style={width=7.05cm,height=2.65cm,scale only axis,xmin=-.4,xmax=12.4,ymin=0,ymax=1,ytick={0,.2,.4,.6,.8,1},axis lines=box,grid=major,grid style={black!12},tick label style={font=\scriptsize},label style={font=\footnotesize},title style={font=\bfseries\small},xtick={0,...,12},x tick label style={rotate=55,anchor=east,font=\scriptsize},ylabel={Mean score}}}'
SCATTER_STYLE = r'\pgfplotsset{srscatter/.style={width=7.05cm,height=2.65cm,scale only axis,ymin=0,ymax=1,ytick={0,.2,.4,.6,.8,1},xlabel={SPEC},ylabel={Mean score},axis lines=box,grid=major,grid style={black!12},tick label style={font=\scriptsize},label style={font=\footnotesize},title style={font=\bfseries\small},enlarge x limits=.15}}'

def group(model):
 g = [r for r in rows if r['model'] == model]
 assert len(g) == 13
 return sorted(g,key=lambda r:(float(r['distance']),r['attack_language']))

def labels(g, aligned):
 return ','.join(r'\texttt{\textbf{'+r['attack_language']+'}}' if r['attack_language']==aligned else r'\texttt{'+r['attack_language']+'}' for r in g)

def profile_axis(g, aligned, color, options):
 coords=' '.join(f"({i},{float(r['mean_score']):.6f})" for i,r in enumerate(g))
 bars=' '.join(f"({i},{float(r['mean_score']):.6f}) += (0,{float(r['mean_ci_high'])-float(r['mean_score']):.6f}) -= (0,{float(r['mean_score'])-float(r['mean_ci_low']):.6f})" for i,r in enumerate(g))
 return [rf'\begin{{axis}}[srprofile,{options},xticklabels={{{labels(g,aligned)}}}]',r'\path[fill=black!7] (axis cs:-.35,0) rectangle (axis cs:.35,1);',rf'\addplot[color={color},mark=*,mark size=1.6pt,line width=.75pt] coordinates {{{coords}}};',rf'\addplot[color={color},opacity=.65,only marks,mark=none,error bars/.cd,y dir=both,y explicit] coordinates {{{bars}}};',r'\end{axis}']

def scatter_axis(g, aligned, color, options):
 coords=' '.join(f"({float(r['spec_score']):.6f},{float(r['mean_score']):.6f})" for r in g)
 lines = [rf'\begin{{axis}}[srscatter,{options}]',rf'\addplot[only marks,color={color},mark=*,mark size=1.7pt] coordinates {{{coords}}};']
 minimum = min(float(r['mean_score']) for r in g)
 maximum = max(float(r['mean_score']) for r in g)
 chosen = [r for r in g if r['attack_language'] == aligned or float(r['mean_score']) in {minimum,maximum}]
 xs=sorted(float(r['spec_score']) for r in g)
 for index,r in enumerate(chosen):
  value=float(r['mean_score']);v=float(r['spec_score'])
  right=v<xs[10]
  upward=value<.75
  dx=9 if right else -9;dy=8 if upward else -8
  anchor=('south west' if right else 'south east') if upward else ('north west' if right else 'north east')
  identifier='languageLabel'+str(index)
  lines.append(rf"\node[font=\scriptsize,anchor={anchor},xshift={dx}pt,yshift={dy}pt,fill=white,inner sep=1pt] ({identifier}) at (axis cs:{v:.6f},{value:.6f}) {{\texttt{{{r['attack_language']}}}}};")
  lines.append(rf"\draw[black!50,line width=.35pt] ({identifier}.{anchor}) -- (axis cs:{v:.6f},{value:.6f});")
 lines.append(r'\end{axis}')
 return lines

def belebele_axes(ref, labels_text, ypos, height):
 extra=[]
 for col in range(2):
  options=(r'ymin=-.03,ymax=1.06,ytick={0,.2,.4,.6,.8,1},ylabel={IF / CONS},title={BELEBELE IF and CONS}' if col==0 else r'ymin=-2.65,ymax=2.15,ytick={-2,0,2},ylabel={SPEC},title={BELEBELE SPEC}')
  extra += [rf'\begin{{axis}}[belebeleaxis,height={height}cm,at={{({8.55*col}cm,{ypos}cm)}},anchor=north west,{options},xticklabels={{{labels_text}}},x tick label style={{rotate=55,anchor=east,font=\scriptsize}}]',r'\path[fill=black!7] (axis cs:-.35,'+('-.03' if col==0 else '-2.65')+') rectangle (axis cs:.35,'+('1.06' if col==0 else '2.15')+');']
  for field,color in ([('if_score','belebeleif'),('cons_score','belebelecons')] if col==0 else [('spec_score','belebelespec')]):
   coords=' '.join(f"({i},{float(r[field]):.6f})" for i,r in enumerate(ref))
   extra.append(rf'\addplot[color={color},mark=*,mark size=1.5pt,line width=.75pt] coordinates {{{coords}}};')
  if col==0:
   bars=' '.join(f"({i},{float(r['cons_score']):.6f}) += (0,{float(r['cons_ci_high'])-float(r['cons_score']):.6f}) -= (0,{float(r['cons_score'])-float(r['cons_ci_low']):.6f})" for i,r in enumerate(ref))
   extra.append(rf'\addplot[color=belebelecons,opacity=.45,only marks,mark=none,error bars/.cd,y dir=both,y explicit] coordinates {{{bars}}};')
  else: extra.append(r'\addplot[black!45,dashed,mark=none] coordinates {(-.4,0) (12.4,0)};')
  extra.append(r'\end{axis}')
 return extra

entries = [(m,n,a,8.55*x,-4.15*y,'srweak' if x==0 else 'srstrong') for y,(a,models) in enumerate(PAIRS) for x,(m,n) in enumerate(models)]

# StrongREJECT: four paired rows only.
lines = COLORS+[r'\begin{tikzpicture}',PROFILE_STYLE,
 r'\node[anchor=south,font=\small\bfseries] at (3.5,.65) {Weak targets};',r'\node[anchor=south,font=\small\bfseries] at (12.05,.65) {Strong targets};']
for model,name,aligned,x,y,color in entries:
 lines += profile_axis(group(model),aligned,color,rf'at={{({x}cm,{y}cm)}},anchor=north west,title={{{name}}}')
lines.append(r'\end{tikzpicture}')
(FIGURES/'strongreject_profiles.tex').write_text('% Generated by eval/scripts/write_reference_comparison_figures.py.\n'+'\n'.join(lines)+'\n')

# Regenerate the original BELEBELE panels into a new release, then compact their
# vertical spacing so the four paired rows fit one page with the caption.
m=runpy.run_path(str(Path(__file__).with_name('write_belebele_profiles.py')))
f=m['main']
f.__globals__['RELEASE']=RELEASE/'paired-belebele-base'
f()
text=(FIGURES/'belebele_profiles.tex').read_text()
# Axis placements are computed by the original generator, not rounded from pixels.
def compact(match):
 x,y=float(match[1]),float(match[2]); row=round(abs(y)/5.65)
 is_spec=abs(abs(y)-row*5.65)>1
 # Determine row correctly for SPEC coordinates.
 if is_spec: row=round((abs(y)-2.45)/5.65)
 new_y=-row*4.95-(2.05 if is_spec else 0)
 return f'at={{({x:.2f}cm,{new_y:.2f}cm)}}'
text=re.sub(r'at=\{\(([-\d.]+)cm,([-\d.]+)cm\)\}',compact,text)
text=text.replace('height=2.2cm','height=1.8cm').replace('height=1.5cm','height=1.25cm')
belebele_preamble=text[:text.index(r'\node[anchor=south,draw=black!20')]
(FIGURES/'belebele_profiles.tex').write_text('% Compacted by eval/scripts/write_reference_comparison_figures.py.\n'+text)

# SPEC scatter, paired rows only.
lines=COLORS+[r'\begin{tikzpicture}',SCATTER_STYLE]
for model,name,aligned,x,y,color in entries:
 lines += scatter_axis(group(model),aligned,color,rf'at={{({x}cm,{y*(4.7/4.15)}cm)}},anchor=north west,title={{{name}}}')
lines.append(r'\end{tikzpicture}')
(FIGURES/'graded_spec_reference_scatter.tex').write_text('% Generated by eval/scripts/write_reference_comparison_figures.py.\n'+'\n'.join(lines)+'\n')

# English reference: StrongREJECT profile and SPEC scatter on top, BELEBELE below.
g=group(REF); cov={(r['model'],r['attack_language']):r for r in coverage}
ref=[cov[REF,r['attack_language']] for r in g]
assert all(int(r['n_cons_comparable'])>=100 for r in ref)
color='black!65'
lines=COLORS+[l for l in belebele_preamble.splitlines() if l.startswith(r'\definecolor')]+[r'\begin{tikzpicture}',PROFILE_STYLE,SCATTER_STYLE]
lines += [l for l in belebele_preamble.splitlines() if not l.startswith(('%',r'\definecolor',r'\begin{tikzpicture}'))]
lines += profile_axis(g,'eng',color,r'height=3.4cm,at={(0cm,0cm)},anchor=north west,title={StrongREJECT mean score}')
lines += scatter_axis(g,'eng',color,r'height=3.4cm,at={(8.55cm,0cm)},anchor=north west,title={SPEC and StrongREJECT}')
lines += [r'\node[anchor=south,draw=black!20,fill=white,inner sep=4pt,font=\scriptsize] at (7.8,-5.15) {\textcolor{belebeleif}{$\bullet$ IF (accuracy)}\quad',
          r'\textcolor{belebelecons}{$\bullet$ CONS (agreement)}\quad',r'\textcolor{belebelespec}{$\bullet$ SPEC}};']
lines += belebele_axes(ref,labels(g,'eng'),-6.25,3.4)
lines.append(r'\end{tikzpicture}')
(FIGURES/'reference_profiles.tex').write_text('% Generated by eval/scripts/write_reference_comparison_figures.py.\n'+'\n'.join(lines)+'\n')

outputs=[FIGURES/n for n in ['strongreject_profiles.tex','belebele_profiles.tex','graded_spec_reference_scatter.tex','reference_profiles.tex']]
for p in outputs:shutil.copy2(p,RELEASE/p.name)
for p in [Path(__file__),Path(__file__).with_name('write_belebele_profiles.py')]:shutil.copy2(p,RELEASE/p.name)
manifest={'analysis_rerun':False,'revision':'English reference moved out of the three paired figures into its own combined figure; paired panels unchanged.','reference_role':'Separate descriptive baseline; not a fifth weak/strong pair and not added to paired inference.','spec_standardization':'Saved all-117-cell standardization unchanged.','checksums':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [CELLS,COVERAGE,*outputs,Path(__file__),Path(__file__).with_name('write_belebele_profiles.py')]}}
(RELEASE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
