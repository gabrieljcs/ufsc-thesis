"""Generate the magnitude report, thesis section and plot from versioned CSVs."""
from pathlib import Path
import csv
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/thesis-magnitude-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'eval/outputs/releases/2026-09-29-score-magnitude-v2'
THESIS=ROOT/'ufscthesisx/ufscthesisx'
NAMES={'sagui_7b':'Sagui-7B','sabia_3':'Sabiá-3','llamantino_2_ultrachat_7b':'LLaMAntino-2','llamantino_anita_8b':'ANITA-8B','gpt_sw3':'GPT-SW3','ai_sweden_llama3_8b':'AI Sweden Llama-3','bggpt_7b':'BgGPT-7B','bggpt_gemma_9b':'BgGPT-Gemma-2','llama3_1_8b_reference':'Llama-3.1 reference'}
rows=list(csv.DictReader((OUT/'model_summary.csv').open()))
s={r['model']:r for r in rows if r['scope']=='non_aligned'}
e=list(csv.DictReader((OUT/'effects.csv').open()))
def effect(f,o,p):return next(r for r in e if r['fit']==f and r['outcome']==o and r['predictor']==p)
def val(r,k):return float(r[k])
def interval(r):return f"{val(r,'estimate'):.3f} [{val(r,'ci_low'):.3f}, {val(r,'ci_high'):.3f}]"
summary=s['paired_panel']
report=['# StrongREJECT score magnitude: post-review extension','',
'This extension was chosen after inspecting the binary results. It preserves the pre-specified analysis and all frozen scores. The corrected September 9 release is the input. The reference model appears only in descriptive summaries.','',
'## What the outcomes measure','',
'- Binary ASR: proportion with any positive score.','- Mean score: the official graded score averaged over all scored responses, including zeros.','- Positive-only mean: magnitude within a subset selected by its observed outcome; this is not a causal effect or a measure over matched successful prompts.','',
'Mean score = ASR × positive-only mean. Scores are rubric ratings, not calibrated probabilities or quantities of real-world harm.','',
'## Non-aligned-language results','',
'| Model | Scored n | ASR | Mean score (95% CI) | Mean among positives |','|---|---:|---:|---:|---:|']
for m,name in NAMES.items():
 r=s[m];report.append(f"| {name} | {r['n']} | {100*val(r,'asr'):.2f}% | {val(r,'mean_score'):.3f} [{val(r,'mean_ci_low'):.3f}, {val(r,'mean_ci_high'):.3f}] | {val(r,'positive_mean'):.3f} |")
report += ['',f"Across paired targets: {summary['n']} scored non-aligned responses; ASR {100*val(summary,'asr'):.2f}%; mean score {val(summary,'mean_score'):.3f}; positive-only mean {val(summary,'positive_mean'):.3f}.",'',
'Intervals bootstrap source prompts as clusters (2,000 draws, fixed seed), preserving model/language rows when a prompt is sampled. The pooled means weight responses, not models equally.','',
'## Matched regression comparison','',
'Fractional-logit GEE models the expected score with a logistic link. Exact zero and one scores are retained. Model and attack-language treatment indicators are fixed effects; robust sandwich covariance clusters all translations/model responses of a source prompt. The binary comparator uses the same rows and design. Positive-only fits retain the parent cohort scaling. Coefficients describe logit conditional means; graded coefficients are not event odds ratios.','',
'| Predictor | Binary | Graded score | Positive-only score |','|---|---:|---:|---:|']
for p in ('distance','spec_score'):
 report.append('| '+p+' | '+' | '.join(interval(effect('pooled',o,p)) for o in ('binary','score','positive_score'))+' |')
report += ['', '## Nuance and robustness','',
'The full-panel distance association stays negative and SPEC stays positive in both binary and graded models, including weak/strong strata. This does not rescue directional H1 or protective H2. The coverage restriction to cells with at least 100 comparable CONS answers reverses distance in both outcomes; it removes the entire weak Italian model, so this is also a model-composition change.','',
'| Specification | Binary distance | Graded distance |','|---|---:|---:|']
for fit in ['weak','strong','non_aligned','cons_at_least_100','original_complete_cohort','questionable_items_removed','tokenizer_cohort_main','tokenizer','translation_caveat_languages_removed','blaser_flagged_removed']:
 report.append('| '+fit+' | '+interval(effect(fit,'binary','distance'))+' | '+interval(effect(fit,'score','distance'))+' |')
report += ['', 'The IF/CONS decomposition shows an additional distinction. Graded outcomes have a larger standardized IF coefficient than the binary comparator. This is a descriptive comparison of fitted coefficients across different outcomes, not a formal test of their difference or a causal capability mechanism.','',
'| Component | Binary | Graded score | Positive-only score |','|---|---:|---:|---:|']
for p in ['if_score','cons_score']:
 report.append('| '+p+' | '+' | '.join(interval(effect('components',o,p)) for o in ('binary','score','positive_score'))+' |')
diag=json.loads((OUT/'diagnostics.json').read_text())
component_diag=next(d for d in diag if d['fit']=='components' and d['outcome']=='score')
report += ['', f"The decomposition is highly collinear: full-design VIF is {component_diag['substantive_vif']['if_score']:.2f} for IF and {component_diag['substantive_vif']['cons_score']:.2f} for CONS. This limits claims about their separate contributions. Single-predictor graded coefficients remain positive: IF {interval(effect('components_single_if_score','score','if_score'))}; CONS {interval(effect('components_single_cons_score','score','cons_score'))}. These omit the other substantive predictors and are not estimates of independent component effects."]
report += ['',f"All {len(diag)} fits converged with finite estimates and positive-definite robust covariance. Pooled full-design VIF: distance {diag[0]['substantive_vif']['distance']:.2f}, SPEC {diag[0]['substantive_vif']['spec_score']:.2f}. Triggered single-predictor fits are in effects.csv; their estimands omit the other substantive controls, so they do not resolve confounding.",'',
'## Limits','',
'Inference is conditional on the observed model/language inventory. Prompt clustering does not allow arbitrary residual dependence between different prompts in the same model-language cell. These are supplementary marginal models, not replacements for the crossed-random-intercept GLMM. Predictors vary largely at model-language level; narrow prompt-cluster intervals should not be mistaken for evidence of broad language/model generalization. Intervals are exploratory and unadjusted for multiple analyses. Positive-only models condition on success and cannot identify causal changes in severity. Human translation/scoring audits remain unavailable. Three nonempty unscored responses are bounded separately in unknown_score_bounds.csv; empty outputs remain missing.','',
'## Reproduction','',
'Run `PYTHONPATH=src .venv/bin/python scripts/analyze_score_magnitude.py --output-dir <new-directory>` from eval. The analysis refuses to overwrite a release. The report generator uses the v2 release. Version 2 corrects positive-subset VIF diagnostics to use the exact positive-only design; the first run is retained as provenance.','',
'[Fractional response method](https://www.nber.org/papers/t0147) · [GEE documentation](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_estimating_equations.GEE.html)','']
(OUT/'REPORT.md').write_text('\n'.join(report))

# Three separate axes emphasize that binary frequency and score magnitude differ.
fig,axes=plt.subplots(1,3,figsize=(11,5.2),sharey=True)
models=sorted(NAMES,key=lambda m:val(s[m],'asr'))
for ax,key,title in zip(axes,['asr','mean_score','positive_mean'],['Positive-score frequency','Mean score, including zeros','Mean among positive scores']):
 for i,m in enumerate(models):
  ax.scatter(val(s[m],key),i,color='#326B86' if 'reference' not in m else '#888888',s=45)
  if key=='mean_score':
   ax.plot([val(s[m],'mean_ci_low'),val(s[m],'mean_ci_high')],[i,i],color='#326B86')
 ax.set_xlim(0,1);ax.set_title(title,fontsize=10);ax.grid(axis='x',alpha=.2);ax.set_xlabel('Fraction' if key=='asr' else 'Rubric score (0–1)')
 ax.spines[['top','right']].set_visible(False)
axes[0].set_yticks(range(len(models)),[NAMES[m] for m in models])
fig.suptitle('Cross-lingual frequency and magnitude',fontsize=15)
fig.text(.5,.015,'Non-aligned responses only • Positive-only means describe selected subsets • Mean-score intervals: prompt bootstrap',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.05,1,.94]);fig.savefig(OUT/'magnitude_comparison.png',dpi=180);fig.savefig(OUT/'magnitude_comparison.pdf');plt.close(fig)

tex=[r'% Generated by eval/scripts/write_magnitude_report.py; do not edit numerical values.',r'\section{Post-review Analysis of StrongREJECT Score Magnitude}',r'\label{sec:score-magnitude}',
r'This supplementary analysis was added after inspecting the binary results. It retains the pre-specified analysis and distinguishes the frequency of positive scores from their magnitude. All scored responses, including zeros, contribute to the mean official score. The identity $\overline{s}=\mathrm{ASR}\,\overline{s}_{s>0}$ is descriptive; the positive-only subset differs across models and does not identify a causal severity effect.',
r'\begin{table}[htbp]\centering\small',r'\caption{Non-aligned-language StrongREJECT frequency and magnitude.}\label{tab:score-magnitude}',r'\begin{tabular}{lrrr}\hline Model & ASR (\%) & Mean score & Positive mean \\\hline']
for m,name in NAMES.items():
 r=s[m];tex.append(f"{name} & {100*val(r,'asr'):.2f} & {val(r,'mean_score'):.3f} & {val(r,'positive_mean'):.3f} \\")
 # Ensure the emitted table row ends with two literal backslashes.
 tex[-1] += '\\'
tex += [r'\hline\end{tabular}',r'\par\smallskip\footnotesize The reference is descriptive only; paired inference uses eight targets. Scores lie in $[0,1]$ and are not probabilities of real-world harm.\end{table}',
f"The paired non-aligned panel contains {summary['n']} scored responses, with mean score {val(summary,'mean_score'):.3f} and positive-only mean {val(summary,'positive_mean'):.3f}. BgGPT-Gemma-2 illustrates relatively infrequent positive scores with substantial conditional magnitude, whereas LLaMAntino-2 produces positive scores more frequently but with lower average magnitude. This comparison concerns observed subsets, not matched successful responses.",
r'\begin{table}[htbp]\centering\small',r'\caption{Matched pooled GEE coefficients with prompt-clustered 95\% confidence intervals.}\label{tab:magnitude-gee}',r'\begin{tabular}{lrr}\hline Outcome & Distance & SPEC \\\hline']
for o,label in [('binary','Binary'),('score','Graded score'),('positive_score','Positive-only score')]:
 tex.append(label+' & '+interval(effect('pooled',o,'distance'))+' & '+interval(effect('pooled',o,'spec_score'))+r' \\')
tex += [r'\hline\end{tabular}\end{table}',
r'The full-panel graded association remains negative for distance and positive for SPEC, as do the separate weak/strong fits. Thus this extension does not reverse the directional conclusions. However, the distance coefficient changes sign when restricting to cells with at least 100 comparable CONS answers: '+interval(effect('cons_at_least_100','score','distance'))+r'. This restriction also removes the weak Italian model; the reversal is not attributable solely to better measurement.',
r'The IF/CONS decomposition adds nuance. For the graded outcome, standardized IF and CONS coefficients are '+interval(effect('components','score','if_score'))+' and '+interval(effect('components','score','cons_score'))+', respectively; the matched binary coefficients are '+interval(effect('components','binary','if_score'))+' and '+interval(effect('components','binary','cons_score'))+r'. The larger IF coefficient for the graded response suggests that reading-comprehension performance is relevant to the magnitude dimension, but comparison across outcomes is not a formal test of coefficient differences and does not establish a mechanism.',
f"All {len(diag)} fitted models, including triggered single-predictor sensitivities, converged with finite estimates and positive-definite robust covariance. The pooled design VIF is {diag[0]['substantive_vif']['distance']:.2f} for distance and {diag[0]['substantive_vif']['spec_score']:.2f} for SPEC. Full tables, bootstrap intervals, tokenizer-matched fits, original-cohort, questionable-item and translation sensitivities are exported in the versioned magnitude supplement.",
r'These exploratory intervals are conditional on the observed model/language inventory and are not adjusted for multiplicity. Prompt clustering does not capture arbitrary residual dependence across different prompts in the same model--language cell. The positive-only models condition on the outcome, and the existing translation/scoring audit limitations remain. The extension enriches the behavioral description without replacing the crossed-random-intercept primary analysis.']
(THESIS/'chapters/results_magnitude_generated.tex').write_text('\n\n'.join(tex)+'\n')
section=THESIS/'chapters/results_magnitude_generated.tex'
section.write_text(section.read_text().replace('The larger IF coefficient for the graded response suggests',f"The decomposition has substantial collinearity (IF VIF {component_diag['substantive_vif']['if_score']:.2f}; CONS VIF {component_diag['substantive_vif']['cons_score']:.2f}), limiting claims about separate contributions. Both retain positive coefficients in single-predictor checks, which omit the other substantive predictors. The larger IF coefficient for the graded response suggests"))
manifest=json.loads((OUT/'manifest.json').read_text())
manifest['plotting_matplotlib_version']=matplotlib.__version__
manifest['report_generator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
manifest['generated_thesis_section_sha256']=hashlib.sha256((THESIS/'chapters/results_magnitude_generated.tex').read_bytes()).hexdigest()
manifest['output_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(OUT/'REPORT.md')
