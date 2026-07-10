"""Generate the MP6 reporting assets for the mechanistic refusal-direction audit.

Reads the small artifacts under eval/outputs/mechanistic/results/ (produced by
scripts/run_mechanistic_cloud.sh and scripts/score_mechanistic_interventions.sh)
and writes:

  - eval/outputs/mechanistic/results/intervention_summary.csv
  - ufscthesisx/ufscthesisx/chapters/mechanistic_generated_assets.tex
    (figure/table \\newcommand macros, mirroring results_generated_assets.tex)

Refusal accounting for the intervention table follows the two-signal rule:
the StrongREJECT judge overcounts refusal on degenerate high-alpha text, and
English marker matching undercounts paraphrased refusals, so the conservative
"explicit refusal" figure requires BOTH the judge label and a refusal marker in
the backtranslated output. still-unsafe comes from the judge score alone.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "eval"
RESULTS = EVAL / "outputs" / "mechanistic" / "results"
SCORED = RESULTS / "scored"
THESIS = ROOT / "ufscthesisx" / "ufscthesisx"
OUT = THESIS / "chapters" / "mechanistic_generated_assets.tex"
SUMMARY_CSV = RESULTS / "intervention_summary.csv"

sys.path.insert(0, str(EVAL / "src"))
from thesis_eval.mechanistic.refusal import looks_like_refusal  # noqa: E402

MODELS = ["llama3_1_8b_reference", "llamantino_anita_8b", "ai_sweden_llama3_8b"]
DISPLAY = {
    "llama3_1_8b_reference": "LLaMA-3.1-8B reference",
    "llamantino_anita_8b": "ANITA-8B (safe)",
    "ai_sweden_llama3_8b": "AI Sweden Llama-3 (eroded)",
}
LANGS = ["eng", "ita", "spa", "dan", "nor", "swe", "swa"]
ALPHAS = ["0.5", "1", "2"]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def intervention_summary() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for alpha in ALPHAS:
            path = SCORED / f"intervention_{model}_alpha{alpha}.backtranslated.strongreject.jsonl"
            if not path.exists():
                continue
            scored = read_jsonl(path)
            n = len(scored)
            judge = [bool(r.get("refusal_binary") == 1) for r in scored]
            marker = [looks_like_refusal(str(r.get("model_output_backtranslated") or "")) for r in scored]
            explicit = sum(1 for j, m in zip(judge, marker) if j and m)
            judge_only = sum(1 for j, m in zip(judge, marker) if j and not m)
            unsafe = sum(1 for r in scored if r.get("unsafe_binary") == 1)
            failed = sum(1 for r in scored if r.get("scoring_failed"))
            rows.append(
                {
                    "model": model,
                    "alpha": alpha,
                    "n": n,
                    "explicit_refusal_n": explicit,
                    "explicit_refusal_rate": explicit / n if n else None,
                    "judge_only_refusal_n": judge_only,
                    "judge_only_refusal_rate": judge_only / n if n else None,
                    "still_unsafe_n": unsafe,
                    "still_unsafe_rate": unsafe / n if n else None,
                    "scoring_failed_n": failed,
                }
            )
    return rows


def write_summary_csv(rows: list[dict[str, object]]) -> None:
    fields = list(rows[0].keys())
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pct(value: object) -> str:
    return f"{float(value) * 100:.1f}" if value is not None else "--"


def num(value: object, digits: int = 2) -> str:
    if value in (None, ""):
        return "--"
    return f"{float(value):.{digits}f}"


def similarity_heatmap(similarity: list[dict[str, str]]) -> str:
    cosine = {(r["model"], r["attack_language"]): r["cosine_with_reference_language"] for r in similarity}
    lines = [
        r"\newcommand{\MechDirectionSimilarityFigure}{%",
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tikzpicture}[x=0.85cm,y=0.55cm]",
    ]
    for x, lang in enumerate(LANGS):
        lines.append(f"\\node[rotate=45,anchor=west,font=\\tiny] at ({x + 0.16},0.35) {{\\texttt{{{lang}}}}};")
    for y, model in enumerate(MODELS):
        lines.append(f"\\node[anchor=east,font=\\tiny] at (-0.15,{-y - 0.5}) {{\\texttt{{{DISPLAY[model]}}}}};")
        for x, lang in enumerate(LANGS):
            raw = cosine.get((model, lang))
            if raw in (None, ""):
                lines.append(f"\\fill[gray!15] ({x},{-y}) rectangle ++(1,-1);")
                text = "--"
            else:
                value = float(raw)
                pctval = max(0, min(100, int(round((value + 1.0) / 2.0 * 100))))
                lines.append(f"\\fill[violet!{pctval}!white] ({x},{-y}) rectangle ++(1,-1);")
                lines.append(f"\\draw[white,line width=0.2pt] ({x},{-y}) rectangle ++(1,-1);")
                text = f"{value:.2f}"
            lines.append(f"\\node[font=\\tiny] at ({x + 0.5},{-y - 0.5}) {{{text}}};")
    y0 = -len(MODELS) - 1.25
    lines.extend(
        [
            f"\\node[anchor=west,font=\\tiny,text width=5.6cm,align=left] at (0,{y0 - 0.3}) {{Cosine of each language's refusal-outcome direction to the model's aligned-language direction (residual layer 22).}};",
            r"\end{tikzpicture}",
            r"\caption[Cross-lingual refusal-direction similarity]{Cross-lingual refusal-direction similarity. Cell values are cosines between the per-language difference-in-means direction and the model's aligned-language direction.}",
            r"\label{fig:mechanistic-direction-similarity}",
            r"\fonte{Author.}",
            r"\end{figure}",
            r"}",
        ]
    )
    return "\n".join(lines)


def separability_scatter(linkage: list[dict[str, str]]) -> str:
    marks = {"llama3_1_8b_reference": "*", "llamantino_anita_8b": "square*", "ai_sweden_llama3_8b": "triangle*"}
    colors = {"llama3_1_8b_reference": "black!75!white", "llamantino_anita_8b": "green!55!black", "ai_sweden_llama3_8b": "red!75!black"}
    lines = [
        r"\newcommand{\MechSeparabilityAsrFigure}{%",
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\begin{tikzpicture}",
        r"\begin{axis}[width=0.72\linewidth,height=5.4cm,",
        r"xlabel={Refused-vs-complied projection AUC},ylabel={ASR},",
        r"label style={font=\scriptsize},tick label style={font=\tiny},",
        r"legend style={font=\tiny,at={(0.5,-0.28)},anchor=north,draw=none,fill=none,legend columns=3}]",
    ]
    for model in MODELS:
        coords = " ".join(
            f"({row['auc_refusal_vs_unsafe']},{row['asr']})"
            for row in linkage
            if row["model"] == model and row["auc_refusal_vs_unsafe"] and row["asr"]
        )
        lines.append(f"\\addplot+[only marks,mark={marks[model]},mark size=1.6pt,{colors[model]}] coordinates {{{coords}}};")
        lines.append(f"\\addlegendentry{{{DISPLAY[model]}}}")
    lines.extend(
        [
            r"\end{axis}",
            r"\end{tikzpicture}",
            r"\caption[Refusal-signal separability versus ASR]{Refusal-signal separability versus behavioural ASR across the 21 model--language cells. Higher within-language separability tracks lower ASR (Spearman $\rho$ reported in the text).}",
            r"\label{fig:mechanistic-separability-vs-asr}",
            r"\fonte{Author.}",
            r"\end{figure}",
            r"}",
        ]
    )
    return "\n".join(lines)


def layer_sweep_curve(sweep: list[dict[str, str]]) -> str:
    coords = " ".join(f"({row['layer']},{row['mean_heldout_auc']})" for row in sweep if row["mean_heldout_auc"])
    selected = [row["layer"] for row in sweep if str(row.get("selected")).lower() == "true"]
    lines = [
        r"\newcommand{\MechLayerSweepFigure}{%",
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\begin{tikzpicture}",
        r"\begin{axis}[width=0.72\linewidth,height=4.6cm,",
        r"xlabel={Residual-stream layer},ylabel={Mean held-out AUC},",
        r"label style={font=\scriptsize},tick label style={font=\tiny}]",
        f"\\addplot+[mark=*,mark size=1.4pt,black!75!white] coordinates {{{coords}}};",
    ]
    if selected:
        lines.append(f"\\draw[densely dashed,red!70!black] (axis cs:{selected[0]},\\pgfkeysvalueof{{/pgfplots/ymin}}) -- (axis cs:{selected[0]},\\pgfkeysvalueof{{/pgfplots/ymax}});")
    lines.extend(
        [
            r"\end{axis}",
            r"\end{tikzpicture}",
            r"\caption[Layer sweep for the refusal-outcome direction]{Layer sweep: mean held-out refused-vs-complied AUC across the three models per residual-stream layer. The dashed line marks the selected layer.}",
            r"\label{fig:mechanistic-layer-sweep}",
            r"\fonte{Author.}",
            r"\end{figure}",
            r"}",
        ]
    )
    return "\n".join(lines)


def projections_table(summary: list[dict[str, str]], similarity: list[dict[str, str]], transfer: list[dict[str, str]], linkage: list[dict[str, str]]) -> str:
    cosine = {(r["model"], r["attack_language"]): r["cosine_with_reference_language"] for r in similarity}
    asr = {(r["model"], r["attack_language"]): r["asr"] for r in linkage}
    transfer_in: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in transfer:
        if row["transfer_auc"]:
            transfer_in[(row["model"], row["target_language"])].append(float(row["transfer_auc"]))
    lines = [
        r"\newcommand{\MechRefusalProjectionsTable}{%",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\caption[Refusal-outcome direction diagnostics per model--language cell]{Refusal-outcome direction diagnostics per model--language cell: within-language refused-vs-complied projection AUC, mean cross-lingual transfer AUC into the cell, cosine to the aligned-language direction, and frozen behavioural ASR.}",
        r"\label{tab:mechanistic-refusal-projections}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Lang. & AUC & Transfer-in & Cosine & ASR \\",
        r"\midrule",
    ]
    for model in MODELS:
        model_rows = [r for r in summary if r["model"] == model]
        by_lang = {r["attack_language"]: r for r in model_rows}
        for index, lang in enumerate(LANGS):
            row = by_lang.get(lang)
            if row is None:
                continue
            values = transfer_in.get((model, lang), [])
            mean_transfer = sum(values) / len(values) if values else None
            first = f"\\texttt{{{DISPLAY[model]}}}" if index == 0 else ""
            lines.append(
                f"{first} & \\texttt{{{lang}}} & {num(row['auc_refusal_vs_unsafe'])} & "
                f"{num(mean_transfer)} & {num(cosine.get((model, lang)))} & {pct(asr.get((model, lang)))}\\% \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.extend(
        [
            r"\end{tabular}",
            r"\fonte{Author.}",
            r"\end{table}",
            r"}",
        ]
    )
    return "\n".join(lines)


def similarity_table(similarity: list[dict[str, str]]) -> str:
    lines = [
        r"\newcommand{\MechLanguageDirectionSimilarityTable}{%",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\caption[Cross-lingual refusal-direction cosine similarity]{Cosine similarity of each attack language's refusal-outcome direction to the model's aligned-language direction and to the model's pooled direction.}",
        r"\label{tab:mechanistic-language-direction-similarity}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Model & Lang. & Cos.\ aligned & Cos.\ pooled \\",
        r"\midrule",
    ]
    for model in MODELS:
        model_rows = [r for r in similarity if r["model"] == model]
        by_lang = {r["attack_language"]: r for r in model_rows}
        for index, lang in enumerate(LANGS):
            row = by_lang.get(lang)
            if row is None:
                continue
            first = f"\\texttt{{{DISPLAY[model]}}}" if index == 0 else ""
            lines.append(
                f"{first} & \\texttt{{{lang}}} & {num(row['cosine_with_reference_language'])} & "
                f"{num(row['cosine_with_pooled_direction'])} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.extend(
        [
            r"\end{tabular}",
            r"\fonte{Author.}",
            r"\end{table}",
            r"}",
        ]
    )
    return "\n".join(lines)


def intervention_table(rows: list[dict[str, object]]) -> str:
    lines = [
        r"\newcommand{\MechInterventionTable}{%",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\caption[Addition-only refusal-strengthening intervention outcomes]{Addition-only refusal-strengthening intervention on previously-complied harmful prompts. Explicit refusal requires agreement between the StrongREJECT judge and refusal-marker matching on the backtranslated output; judge-only counts are dominated by degenerate high-$\alpha$ text that the judge mislabels as refusal. Still-unsafe uses the judge score alone.}",
        r"\label{tab:mechanistic-intervention}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & $\alpha$ & $n$ & Explicit refusal & Judge-only & Still unsafe \\",
        r"\midrule",
    ]
    for model in MODELS:
        model_rows = [r for r in rows if r["model"] == model]
        for index, row in enumerate(model_rows):
            first = f"\\texttt{{{DISPLAY[model]}}}" if index == 0 else ""
            lines.append(
                f"{first} & {row['alpha']} & {row['n']} & {pct(row['explicit_refusal_rate'])}\\% & "
                f"{pct(row['judge_only_refusal_rate'])}\\% & {pct(row['still_unsafe_rate'])}\\% \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.extend(
        [
            r"\end{tabular}",
            r"\fonte{Author.}",
            r"\end{table}",
            r"}",
        ]
    )
    return "\n".join(lines)


def write_assets() -> None:
    summary = read_csv_rows(RESULTS / "refusal_summary.csv")
    similarity = read_csv_rows(RESULTS / "language_similarity.csv")
    transfer = read_csv_rows(RESULTS / "transfer.csv")
    sweep = read_csv_rows(RESULTS / "layer_sweep.csv")
    linkage = read_csv_rows(RESULTS / "separability_vs_asr.csv")
    interventions = intervention_summary()
    if interventions:
        write_summary_csv(interventions)
        print(f"Wrote {SUMMARY_CSV}")

    parts = [
        "% Generated by eval/scripts/write_mechanistic_latex_assets.py from the",
        "% mechanistic-results artifacts. Figure and table macros for the exploratory",
        "% mechanistic chapter; invoke e.g. \\MechDirectionSimilarityFigure.",
        similarity_heatmap(similarity),
        separability_scatter(linkage),
        layer_sweep_curve(sweep),
        projections_table(summary, similarity, transfer, linkage),
        similarity_table(similarity),
    ]
    if interventions:
        parts.append(intervention_table(interventions))
    OUT.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    write_assets()
