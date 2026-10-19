"""Draw the saved layer-selection diagnostic; no activation extraction or inference."""
from pathlib import Path
import csv, hashlib, json, shutil
ROOT=Path(__file__).resolve().parents[2]
THESIS=ROOT/'ufscthesisx/ufscthesisx'
SOURCE=ROOT/'eval/outputs/mechanistic/results/layer_sweep.csv'
RELEASE=ROOT/'eval/outputs/releases/2026-10-12-mechanistic-interpretation'
def main():
    with SOURCE.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    selected=[r for r in rows if r['selected'].lower()=='true']
    assert len(rows)==9 and len(selected)==1 and all(int(r['n_models'])==3 for r in rows)
    best=selected[0];assert float(best['mean_heldout_auc'])==max(float(r['mean_heldout_auc']) for r in rows)
    coords=' '.join(f"({r['layer']},{r['mean_heldout_auc']})" for r in rows)
    lines=[r'\begin{figure}[htbp]\centering',r'\caption[Layer selection for the refusal-outcome audit]{Selecting a residual-stream index for the subsequent audit. Each point averages three model-specific AUCs: estimate a direction on one label-stratified half and rank outcomes on the other, using the same split at every index. The curve levels off after index 14; index 22 is the highest of the nine checked values. The orange marker shows that choice, not an independently validated optimal layer.}',r'\begin{tikzpicture}',r'\definecolor{layerblue}{HTML}{286F8C}\definecolor{layerorange}{HTML}{A64B39}',r'\begin{axis}[width=.85\linewidth,height=5.8cm,xmin=7.5,xmax=24.5,ymin=.48,ymax=.93,xtick={8,10,12,14,16,18,20,22,24},ytick={.5,.6,.7,.8,.9},xlabel={Residual-stream index (depth)},ylabel={Mean split-sample AUC},label style={font=\small},tick label style={font=\footnotesize},axis line style={black!40},ymajorgrids,grid style={black!10}]',r'\addplot[black!40,densely dotted] coordinates {(7.5,.5)(24.5,.5)};',r'\addplot[thick,color=layerblue,mark=*,mark size=2pt] coordinates {'+coords+'};',f"\\draw[densely dashed,layerorange] (axis cs:{best['layer']},.5) -- (axis cs:{best['layer']},{best['mean_heldout_auc']});",f"\\addplot[only marks,mark=*,mark size=3pt,color=layerorange] coordinates {{({best['layer']},{best['mean_heldout_auc']})}};",f"\\node[font=\\footnotesize,align=right,anchor=north east,text=layerorange] at (axis cs:21.5,.79) {{Selected index: {best['layer']}\\\\Mean AUC: {float(best['mean_heldout_auc']):.3f}}};",r'\node[font=\footnotesize,anchor=south west,text=black!60] at (axis cs:8,.505) {Chance ranking};',r'\end{axis}\end{tikzpicture}',r'\label{fig:mechanistic-layer-sweep}\fonte{Author.}\end{figure}']
    out=THESIS/'figures/mechanistic_layer_selection.tex';out.write_text('\n'.join(lines)+'\n')
    RELEASE.mkdir(exist_ok=True);shutil.copy2(out,RELEASE/out.name);shutil.copy2(Path(__file__),RELEASE/Path(__file__).name)
    (RELEASE/'figure_manifest.json').write_text(json.dumps({'purpose':'Presentation of saved layer-selection values, without new extraction or analysis.','source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'figure_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'selected_index':int(best['layer']),'selected_mean_auc':float(best['mean_heldout_auc'])},indent=2)+'\n')
if __name__=='__main__':main()
