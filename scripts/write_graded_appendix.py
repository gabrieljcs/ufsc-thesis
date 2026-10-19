"""Export complete graded tables and traceable saved response examples.
Response text is quoted unchanged; prompt descriptions remain editorial paraphrases.
"""
import csv, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
REL=ROOT/'eval/outputs/releases/2026-09-29-graded-reporting'
DATA=ROOT/'eval/outputs/releases/2026-09-09-corrected/dataset_corrected.jsonl'
DEST=ROOT/'ufscthesisx/ufscthesisx/aftertext/apendice_g.tex'
def esc(x):
    return str(x).replace('\\',r'\textbackslash{}').replace('_',r'\_\allowbreak{}').replace('&',r'\&').replace('%',r'\%').replace('#',r'\#')
def read(name):
    with (REL/name).open() as f:return list(csv.DictReader(f))
def number(x):return f'{float(x):.3f}'
lines=[r'\chapter{Detailed Graded Results and Example Records}',r'\label{app:graded-details}',
       'This appendix is generated from the graded release by '+r'\texttt{eval/scripts/write\_graded\_appendix.py}. '+'The tables retain the official score magnitudes, including zero. The English reference is descriptive and excluded from paired regression inference.',
       'The predictor abbreviations are instruction following (IF), cross-lingual consistency (CONS), and specialization (SPEC). IF measures BELEBELE answer accuracy; CONS measures agreement with aligned-language answers; SPEC averages their standardized values with equal weights.']
def table(headers, rows, caption, label, widths):
    lines.extend([r'\Needspace{18\baselineskip}\begingroup\footnotesize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.12}',r'\begin{longtable}{'+widths+'}',r'\caption{'+caption+r'}\label{'+label+r'}\\',r'\toprule',' & '.join(headers)+r'\\\midrule',r'\endfirsthead',r'\multicolumn{'+str(len(headers))+r'}{l}{\textit{Continued from previous page}}\\',r'\toprule',' & '.join(headers)+r'\\\midrule',r'\endhead',r'\bottomrule\endfoot'])
    lines.extend(' & '.join(esc(x) for x in row)+r'\\' for row in rows)
    lines.extend([r'\end{longtable}',r'\endgroup'])
cells=read('cell_summary.csv')
names={'sagui_7b':'Sagui-7B','sabia_3':'Sabiá-3','llamantino_2_ultrachat_7b':'LLaMAntino-2','llamantino_anita_8b':'ANITA-8B','gpt_sw3':'GPT-SW3','ai_sweden_llama3_8b':'AI Sweden','bggpt_7b':'BgGPT-7B','bggpt_gemma_9b':'BgGPT-Gemma','llama3_1_8b_reference':'English reference'}
summary=read('model_summary.csv')
lines += [r'\section{Aligned and Non-aligned Score Summary}',
 r'Means include zero scores and exclude missing judge scores. Intervals are 95\% prompt-cluster percentile bootstrap intervals (2,000 resamples). Non-aligned means give equal weight to each scored response; languages with more scored responses contribute more.']
summary_rows=[]
for model in names:
    entries={r['scope']:r for r in summary if r['model']==model}
    row=[names[model]]
    for scope in ['aligned','non_aligned']:
        r=entries[scope]
        row += [r['n'],number(r['mean_score']),f"[{number(r['mean_ci_low'])}, {number(r['mean_ci_high'])}]"]
    summary_rows.append(row)
table(['Model',r'Aligned $n$','Mean',r'95\% interval',r'Other $n$','Mean',r'95\% interval'],summary_rows,
      'Aligned-language and pooled non-aligned-language StrongREJECT scores', 'tab:graded-setting-summary','lrrrrrr')
lines += [r'\section{Complete Model--Language Scores}',r'Intervals are 95\% prompt-cluster percentile bootstrap intervals (2,000 resamples). Counts are scored responses; missing judge outcomes are excluded, not recoded as zero. Model-specific rows follow increasing linguistic distance.']
for model in names:
    subset=sorted((r for r in cells if r['model']==model),key=lambda r:float(r['distance']))
    table(['Language',r'Scored $n$','Mean',r'95\% interval','Distance','SPEC'],[[r['attack_language'],r['n'],number(r['mean_score']),f"[{number(r['mean_ci_low'])}, {number(r['mean_ci_high'])}]",number(r['distance']),number(r['spec_score'])] for r in subset],names[model]+': complete graded results','tab:detail-'+model,'lrrrrr')
lines += [r'\section{Complete Graded Regression Coefficients}','Fit names are exact release identifiers and can wrap at underscores. Each row is one predictor estimate; repeated headers identify columns when the table continues onto another page. The following table includes all graded fits and triggered single-predictor sensitivities. Coefficients describe the logit of the conditional expected score; they are not event odds ratios. Confidence intervals use prompt-clustered robust covariance. Fit identifiers correspond exactly to the released diagnostics.']
effects=read('effects.csv')
table(['Fit','Predictor',r'$n$',r'$\beta$',r'95\% interval'],[[r['fit'],r['predictor'],r['n'],number(r['estimate']),f"[{number(r['ci_low'])}, {number(r['ci_high'])}]"] for r in effects],'All graded coefficient estimates','tab:graded-all-coefficients',r'p{3.7cm}p{2.1cm}rrr')
lines += [r'\section{Traceable Qualitative Example Records}',r'\label{app:graded-example-records}',
r'These cases complement the complete saved responses in Appendix~\ref{app:qualitative-strongreject-examples}. Selection is illustrative and deliberately includes paired contrasts and a partial-score response; it does not estimate prevalence. StrongREJECT prompts are direct prohibited requests in this experiment, not necessarily adversarial prompt wrappers. Prompt IDs below are the canonical dataset identifiers (the earlier appendix uses shortened SR labels).',
'Prompt descriptions below are editorial paraphrases. Complete saved English back-translations follow, with their original-language text in the companion response export. Scores and rubric components are extracted unchanged by script. A nonzero score alone does not establish that a harmful objective could be achieved outside the evaluator rubric.']
rows=[json.loads(line) for line in DATA.open()]
index={(r['prompt_id'],r['attack_language'],r['model']):r for r in rows}
cases=[('strongreject_0001','bul','bggpt_7b','Request for a deceptive banking email.'),('strongreject_0001','bul','bggpt_gemma_9b','Same deceptive-banking-email request.'),('strongreject_0001','por','sagui_7b','Portuguese version of the deceptive-banking-email request.'),('strongreject_0001','por','sabia_3','Same Portuguese request.'),('strongreject_0123','bul','bggpt_gemma_9b','Request involving hazardous content; see the complete saved response in Case C of the qualitative appendix.')]
exports=[]
for i,(pid,lang,model,description) in enumerate(cases,1):
    r=index[(pid,lang,model)]
    assert r['strongreject_score'] is not None
    record={k:r.get(k) for k in ['prompt_id','run_id','model','attack_language','strongreject_score','refusal_binary','specificity_score','convincingness_score']}
    record.update(prompt_paraphrase=description,response_display=r['model_output_backtranslated'],original_response=r['model_output'],source_record_sha256=hashlib.sha256(json.dumps(r,sort_keys=True,ensure_ascii=False).encode()).hexdigest())
    exports.append(record)
    lines.extend([r'\subsection{Example '+str(i)+': '+esc(names[model])+'}',esc(description),r'\begin{description}',r'\item[Identity] '+esc(pid)+'; '+esc(lang)+'.',r'\item[Official score] '+number(r['strongreject_score'])+'.',r'\item[Rubric fields] Refusal: '+str(r['refusal_binary'])+'; specificity: '+str(r['specificity_score'])+'; convincingness: '+str(r['convincingness_score'])+'.',r'\end{description}',r'\noindent\textbf{Complete saved English back-translation:}',r'\begin{quote}\small\raggedright',esc(r['model_output_backtranslated']),r'\end{quote}'])
lines.append(r'\input{figures/graded_distance_appendix}')
DEST.write_text('\n\n'.join(lines)+'\n')
(REL/'qualitative_example_records.json').write_text(json.dumps(exports,indent=2,ensure_ascii=False)+'\n')
print(f'Exported {len(cells)} cells, {len(effects)} coefficients and {len(exports)} example records.')
