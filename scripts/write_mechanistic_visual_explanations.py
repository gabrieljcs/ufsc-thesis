"""Render existing mechanistic summaries as native thesis figures; no model reruns."""
from pathlib import Path
import csv, hashlib, json, shutil
ROOT = Path(__file__).resolve().parents[2]
THESIS = ROOT / 'ufscthesisx/ufscthesisx'
RESULTS = ROOT / 'eval/outputs/mechanistic/results'
GRADED = ROOT / 'eval/outputs/releases/2026-09-29-graded-reporting'
RELEASE = ROOT / 'eval/outputs/releases/2026-10-12-mechanistic-visual-explanations'
MODELS = ['llama3_1_8b_reference','llamantino_anita_8b','ai_sweden_llama3_8b']
NAMES = ['English reference','ANITA','AI Sweden']
LANGS = ['eng','ita','spa','dan','nor','swe','swa']
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    projection = {(r['model'],r['attack_language']): r for r in read(RESULTS/'refusal_summary.csv')}
    transfer = {(r['model'],r['attack_language']): r for r in read(RESULTS/'transfer_profile.csv')}
    rows = {r['model']:r for r in read(GRADED/'mechanistic_intervention_scores.csv')}
    raw_transfer = read(RESULTS/'transfer.csv')
    # Confirm the saved profile means describe exactly six other source languages.
    for (model,lang), profile in transfer.items():
        values = [float(r['transfer_auc']) for r in raw_transfer if r['model']==model and r['target_language']==lang and r['source_language']!=lang]
        assert len(values)==6
        assert abs(sum(values)/len(values)-float(profile['mean_auc_as_target']))<1e-12
    assert all(0<=float(r['auc_refusal_vs_unsafe'])<=1 for r in projection.values())
    assert len(projection)==len(transfer)==21 and len(rows)==3
    colors = r'\definecolor{mechblue}{HTML}{286F8C}\definecolor{mechorange}{HTML}{A64B39}'
    out = [r'\begin{figure}[p]\centering', r'\begin{tikzpicture}', colors]
    for k,(model,name) in enumerate(zip(MODELS,NAMES)):
        axis = r'\begin{axis}[width=13.1cm,height=3.8cm,at={(0,%scm)},anchor=north west,' % (-4.25*k)
        axis += r'xmin=-.5,xmax=6.5,ymin=.45,ymax=1.04,ytick={.5,.75,1},ylabel={AUC},xtick={0,1,2,3,4,5,6},xticklabels={English,Italian,Spanish,Danish,Norwegian,Swedish,Swahili},tick label style={font=\footnotesize},label style={font=\small},title={'+name+r'},title style={font=\small\bfseries},axis line style={black!35},ymajorgrids,grid style={black!10},legend style={font=\footnotesize,draw=none,at={(.5,1.35)},anchor=south,legend columns=2}]'
        out.append(axis)
        out.append(r'\addplot[black!40,dashed,forget plot] coordinates {(-.5,.5)(6.5,.5)};')
        for shift,key,color,mark,lookup in [(-.1,'auc_refusal_vs_unsafe','mechblue','*',projection),(.1,'mean_auc_as_target','mechorange','diamond*',transfer)]:
            points=' '.join(f'({i+shift:.1f},{float(lookup[model,lang][key]):.12f})' for i,lang in enumerate(LANGS))
            out.append(r'\addplot[only marks,mark='+mark+',mark size=2.4pt,color='+color+'] coordinates {'+points+'};')
        if k==0:
            out.append(r'\legend{Pooled-direction AUC (in-sample),Transfer AUC (target mean)}')
        out.append(r'\end{axis}')
    out += [r'\end{tikzpicture}',r'\caption[Refusal-signal separability by language]{Refusal-signal separability at index 22. Blue circles use each model\textquotesingle s pooled direction on its estimation rows. Orange diamonds average the six directions learned in other languages and evaluated on the indicated target language. The dashed line is chance ranking ($0.5$). The summaries use different direction-estimation procedures; their gap is descriptive, not a held-out performance loss. All transfer pairs share some question identities.}\label{fig:mechanistic-separability-overview}',r'\fonte{Prepared by the author (2026).}',r'\end{figure}']
    (THESIS/'figures/mechanistic_results_overview.tex').write_text('\n'.join(out)+'\n')
    out=[r'\begin{figure}[htbp]\centering',r'\begin{tikzpicture}',colors,r'\begin{axis}[width=13.2cm,height=5.2cm,xmin=0,xmax=1,ymin=-.55,ymax=2.55,ytick={0,1,2},yticklabels={AI Sweden ($\alpha=0.5$),ANITA ($\alpha=1$),Reference ($\alpha=1$)},xlabel={Mean StrongREJECT score after addition},xtick={0,.25,.5,.75,1},tick label style={font=\footnotesize},label style={font=\small},axis line style={black!35},xmajorgrids,grid style={black!10},clip=false]']
    for k,model in enumerate(MODELS):
        r=rows[model]; y=2-k; score=float(r['mean_score']); n=int(r['n']); scored=int(r['scored_n'])
        assert 0<=score<=1 and 0<scored<=n
        out += [f'\\addplot[mechblue,thick,forget plot] coordinates {{(0,{y})({score:.12f},{y})}};',f'\\addplot[only marks,mark=*,mark size=3pt,mechblue,forget plot] coordinates {{({score:.12f},{y})}};',f'\\node[anchor=west,font=\\footnotesize] at (axis cs:{score+.025:.12f},{y}) {{{score:.4f}\\quad({scored}/{n} scored)}};']
    out += [r'\end{axis}',r'\end{tikzpicture}',r'\caption[Graded scores after refusal-direction addition]{Mean graded scores after addition at the reported, post-hoc strengths. The denominator shown beside each point is scored responses out of generated responses; two AI Sweden responses are unscored. These are reused, outcome-selected prompts at different strengths, with no zero-strength rerun or independent controls. A low score can also reflect irrelevant or degenerate output; it is not by itself evidence of successful refusal.}\label{fig:mechanistic-intervention-scores}',r'\fonte{Prepared by the author (2026).}',r'\end{figure}']
    (THESIS/'figures/mechanistic_intervention_scores.tex').write_text('\n'.join(out)+'\n')
    RELEASE.mkdir(exist_ok=True)
    inputs=[RESULTS/'refusal_summary.csv',RESULTS/'transfer_profile.csv',RESULTS/'transfer.csv',GRADED/'mechanistic_intervention_scores.csv']
    outputs=[THESIS/'figures'/f for f in ['residual_stream.tex','mechanistic_results_overview.tex','mechanistic_intervention_scores.tex']]
    for p in outputs: shutil.copy2(p,RELEASE/p.name)
    shutil.copy2(Path(__file__),RELEASE/Path(__file__).name)
    (RELEASE/'manifest.json').write_text(json.dumps({'purpose':'Presentation-only mechanistic figures using saved summaries; no new extraction, scoring, or inference.','inputs':{str(p.relative_to(ROOT)):digest(p) for p in inputs},'outputs':{str(p.relative_to(ROOT)):digest(p) for p in outputs},'generator_sha256':digest(Path(__file__))},indent=2)+'\n')
if __name__=='__main__': main()
