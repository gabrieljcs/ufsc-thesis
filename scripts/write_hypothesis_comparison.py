"""Illustrate qualitative expectations beside unchanged graded coefficient exports."""
from pathlib import Path
import os,json,hashlib
os.environ.setdefault('MPLCONFIGDIR','/tmp/thesis-magnitude-mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'eval/outputs/releases/2026-09-29-graded-reporting/effects.csv'
DIAG=ROOT/'eval/outputs/releases/2026-09-29-graded-reporting/diagnostics.json'
OUT=ROOT/'eval/outputs/releases/2026-10-06-hypothesis-comparison';OUT.mkdir(exist_ok=True)
T=ROOT/'ufscthesisx/ufscthesisx'
effects=pd.read_csv(SOURCE)
def row(f,p):
 g=effects[(effects.fit==f)&(effects.predictor==p)]
 assert len(g)==1
 return g.iloc[0]
fig,axes=plt.subplots(3,2,figsize=(10.2,9.2),gridspec_kw={'width_ratios':[1,1.35]})
blue='#0072B2';orange='#D55E00';expected='#41644A'
for ax in axes.flat:
 for side in ['top','right']:ax.spines[side].set_visible(False)
 ax.tick_params(labelsize=9)
for i,p in enumerate(['distance','spec_score']):
 ax=axes[i,0]
 x=np.linspace(0,1,100)
 # Illustrative shapes only: these are not fitted values or prior magnitudes.
 y=.24+.48*x if i==0 else .72-.48*x
 ax.plot(x,y,color=expected,lw=2.8)
 ax.set_xlim(0,1);ax.set_ylim(0,1)
 ax.set_xticks([0,1],['Closer','Farther'] if i==0 else ['Lower SPEC','Higher SPEC'])
 ax.set_yticks([0,1],['Lower','Higher']);ax.set_ylabel('Mean StrongREJECT score',fontsize=9)
 ax.set_title(('H₁: positive distance association' if i==0 else 'H₂: negative specialization association')+'\nPredicted association (schematic)',fontsize=10,pad=12)
 ax.text(.5,.93,'Expected coefficient > 0' if i==0 else 'Expected coefficient < 0',transform=ax.transAxes,ha='center',fontsize=9)
 observed=axes[i,1]
 selected=[]
 for j,(f,label) in enumerate([('pooled','Pooled'),('weak','Weak pole'),('strong','Strong pole'),('cons_at_least_100','CONS ≥ 100*')]):
  r=row(f,p);selected.append(r)
  observed.errorbar(r.estimate,j,xerr=[[r.estimate-r.ci_low],[r.ci_high-r.estimate]],fmt='o',color=orange if j==3 else blue,capsize=3,markersize=5)
 observed.axvline(0,color='#777777',lw=1,ls='--')
 observed.set_yticks(range(4),['Pooled','Weak pole','Strong pole','CONS ≥ 100*']);observed.invert_yaxis()
 observed.set_xlabel('Standardized fractional-logit coefficient',fontsize=9)
 observed.set_title('Estimated distance coefficients\nNegative main estimates; positive coverage estimate' if i==0 else 'Estimated specialization coefficients\nPositive estimates across the displayed cohorts',fontsize=10,pad=12)
 observed.xaxis.grid(alpha=.18);observed.set_axisbelow(True)
 if i==0:observed.set_xlim(-.30,.16)
 else:observed.set_xlim(-.18,1.9)
ax=axes[2,0]
ax.barh([1,0],[.85,.38],color=expected,height=.36)
ax.set_yticks([1,0],['Distance','SPEC']);ax.set_xlim(0,1);ax.set_ylim(-.4,1.8);ax.set_xticks([])
ax.set_xlabel('Relative coefficient magnitude (schematic)',fontsize=9)
ax.set_title('H₃: distance as the stronger predictor\nPredicted association (schematic)',fontsize=10,pad=12)
ax.text(.5,.93,'Larger distance coefficient after adjustment',transform=ax.transAxes,ha='center',fontsize=9)
ax=axes[2,1]
for y,p,label in [(1,'distance','Distance'),(0,'spec_score','SPEC')]:
 r=row('pooled',p)
 bounds=sorted([abs(r.ci_low),abs(r.ci_high)])
 assert r.ci_low*r.ci_high>0
 ax.errorbar(abs(r.estimate),y,xerr=[[abs(r.estimate)-bounds[0]],[bounds[1]-abs(r.estimate)]],fmt='o',color=blue,capsize=4)
 ax.text(abs(r.estimate)+.04,y+.12,f'{abs(r.estimate):.3f}',fontsize=9)
ax.set_yticks([1,0],['Distance','SPEC']);ax.set_xlim(0,1.18);ax.set_ylim(-.5,1.5)
ax.set_xlabel('Absolute pooled standardized coefficient',fontsize=9)
ax.set_title('Absolute pooled coefficients\nLarger SPEC estimate; substantial collinearity',fontsize=10,pad=12)
ax.xaxis.grid(alpha=.18);ax.set_axisbelow(True)
fig.tight_layout(h_pad=3,w_pad=2.5,rect=[0,.08,1,1])
fig.text(.02,.042,'* The CONS-coverage restriction excludes low-coverage cells and all LLaMAntino-2 responses.',fontsize=9)
fig.text(.02,.018,'Predicted panels are schematic. Estimated panels use existing fits with prompt-clustered 95% confidence intervals.',fontsize=9)
fig.savefig(T/'figures/hypothesis_expected_observed.pdf',bbox_inches='tight')
fig.savefig(OUT/'hypothesis_expected_observed.png',dpi=180,bbox_inches='tight')
used=effects[(effects.fit.isin(['pooled','weak','strong','cons_at_least_100']))&(effects.predictor.isin(['distance','spec_score']))]
used.to_csv(OUT/'plotted_estimates.csv',index=False)
(OUT/'manifest.json').write_text(json.dumps({'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'diagnostics_sha256':hashlib.sha256(DIAG.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'provenance':'Explanatory post-review visualization: schematic expected panels have no empirical magnitudes; observed panels reproduce existing graded fits. No statistical refits or new decision rules.'},indent=2)+'\n')
print('Generated expected-versus-observed hypothesis comparison.')
