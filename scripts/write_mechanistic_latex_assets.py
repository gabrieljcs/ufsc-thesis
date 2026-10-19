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
TRANSFER_PROFILE_CSV = RESULTS / "transfer_profile.csv"

sys.path.insert(0, str(EVAL / "src"))
from thesis_eval.mechanistic.refusal import looks_like_refusal  # noqa: E402

MODELS = ["llama3_1_8b_reference", "llamantino_anita_8b", "ai_sweden_llama3_8b"]
DISPLAY = {
    "llama3_1_8b_reference": "Llama-3.1-8B reference",
    "llamantino_anita_8b": "ANITA-8B (safer)",
    "ai_sweden_llama3_8b": "AI Sweden Llama-3 (vulnerable)",
}
MATRIX_DISPLAY = {
    "llama3_1_8b_reference": "Reference",
    "llamantino_anita_8b": "ANITA (safer)",
    "ai_sweden_llama3_8b": "AI Sweden (vulnerable)",
}
LANGS = ["eng", "ita", "spa", "dan", "nor", "swe", "swa"]
LANGUAGE_NAMES = {
    "eng": "English",
    "ita": "Italian",
    "spa": "Spanish",
    "dan": "Danish",
    "nor": "Norwegian",
    "swe": "Swedish",
    "swa": "Swahili",
}
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
                    "other_n": n - explicit - judge_only - unsafe - failed,
                    "other_rate": (n - explicit - judge_only - unsafe - failed) / n if n else None,
                }
            )
    return rows


def write_summary_csv(rows: list[dict[str, object]]) -> None:
    fields = list(rows[0].keys())
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def transfer_profiles(transfer: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for language in LANGS:
            source_values = [
                float(row["transfer_auc"])
                for row in transfer
                if row["model"] == model and row["source_language"] == language and row["transfer_auc"]
            ]
            target_values = [
                float(row["transfer_auc"])
                for row in transfer
                if row["model"] == model and row["target_language"] == language and row["transfer_auc"]
            ]
            rows.append(
                {
                    "model": model,
                    "attack_language": language,
                    "mean_auc_as_source": sum(source_values) / len(source_values),
                    "min_auc_as_source": min(source_values),
                    "max_auc_as_source": max(source_values),
                    "mean_auc_as_target": sum(target_values) / len(target_values),
                    "min_auc_as_target": min(target_values),
                    "max_auc_as_target": max(target_values),
                }
            )
    return rows


def write_transfer_profiles(rows: list[dict[str, object]]) -> None:
    fields = list(rows[0].keys())
    with TRANSFER_PROFILE_CSV.open("w", encoding="utf-8", newline="") as fh:
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
        lines.append(f"\\node[anchor=east,font=\\tiny] at (-0.15,{-y - 0.5}) {{{DISPLAY[model]}}};")
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
            r"\fonte{Prepared by the author (2026).}",
            r"\end{figure}",
            r"}",
        ]
    )
    return "\n".join(lines)


def direction_construction_schematic() -> str:
    """Explain the difference-in-means direction without inventing a 2-D embedding."""
    return "\n".join(
        [
            r"\newcommand{\MechDirectionConstructionFigure}{%",
            r"\begin{figure}[htbp]",
            r"\centering",
            r"\begin{tikzpicture}[x=1.15cm,y=1.0cm,>=Stealth]",
            r"\fill[red!65] (-2.75,-0.35) circle (1.7pt);",
            r"\fill[red!65] (-2.35,0.28) circle (1.7pt);",
            r"\fill[red!65] (-1.75,-0.42) circle (1.7pt);",
            r"\fill[red!65] (-1.45,0.20) circle (1.7pt);",
            r"\fill[blue!70] (1.25,0.10) circle (1.7pt);",
            r"\fill[blue!70] (1.65,0.72) circle (1.7pt);",
            r"\fill[blue!70] (2.20,0.05) circle (1.7pt);",
            r"\fill[blue!70] (2.55,0.70) circle (1.7pt);",
            r"\fill[red!85!black] (-2.05,-0.07) circle (2.5pt) node[below=3pt,font=\scriptsize] {$\mu_{\mathrm{unsafe}}$};",
            r"\fill[blue!85!black] (1.91,0.39) circle (2.5pt) node[below=3pt,font=\scriptsize] {$\mu_{\mathrm{refusal}}$};",
            r"\draw[->,very thick,violet!85!black] (-2.05,-0.07) -- (1.91,0.39) node[midway,above=6pt,font=\scriptsize] {$\hat r=(\mu_{\mathrm{refusal}}-\mu_{\mathrm{unsafe}})/\lVert\mu_{\mathrm{refusal}}-\mu_{\mathrm{unsafe}}\rVert$};",
            r"\fill[black] (-0.15,1.20) circle (1.8pt) node[above,font=\scriptsize] {prompt activation $h$};",
            r"\draw[densely dashed,black!60] (-0.15,1.20) -- (-0.04,0.17);",
            r"\fill[black!70] (-0.04,0.17) circle (1.5pt);",
            r"\node[font=\scriptsize,align=left,anchor=west] at (2.85,-0.65) {$z=(h-m)^\top\hat r$\\$m=(\mu_{\mathrm{refusal}}+\mu_{\mathrm{unsafe}})/2$};",
            r"\node[font=\tiny,text=red!75!black] at (-2.05,-0.85) {unsafe outcomes};",
            r"\node[font=\tiny,text=blue!75!black] at (1.91,-0.85) {refused outcomes};",
            r"\end{tikzpicture}",
            r"\caption[Construction of the refusal-outcome direction]{Construction of the refusal-outcome direction. The vector is the normalized difference between the mean final-prompt-token activation for refused and unsafe outcomes; a prompt is summarized by its signed projection $z$ onto that axis. This is a schematic of the computation, not a two-dimensional embedding of the observed 4\,096-dimensional activations.}",
            r"\label{fig:mechanistic-direction-construction}",
            r"\fonte{Prepared by the author (2026), following \textcite{arditiRefusalLanguageModels2024}.}",
            r"\end{figure}",
            r"}",
        ]
    )


def transfer_matrices(
    transfer: list[dict[str, str]], profiles: list[dict[str, object]]
) -> str:
    """Render ordered transfer AUCs with readable source and target margins."""
    lookup = {
        (row["model"], row["source_language"], row["target_language"]): row["transfer_auc"]
        for row in transfer
    }
    profile_lookup = {
        (str(row["model"]), str(row["attack_language"])): row for row in profiles
    }

    def auc_fill(value: float) -> str:
        if value < 0.5:
            return "red!45!white"
        intensity = max(8, min(92, int(round(8 + (value - 0.5) / 0.5 * 84))))
        return f"teal!{intensity}!white"

    lines = [
        r"\newcommand{\MechTransferMatrixFigure}{%",
        r"\begin{figure}[p]",
        r"\centering",
        r"\begin{tikzpicture}[x=0.72cm,y=0.50cm]",
    ]
    for panel, model in enumerate(MODELS):
        y0 = -panel * 10
        lines.append(
            f"\\node[font=\\small,align=center,text width=7.0cm] at (3.5,{y0 + 1.55}) "
            f"{{{MATRIX_DISPLAY[model]}}};"
        )
        lines.append(
            f"\\node[font=\\tiny,text=black!70,anchor=east] at (-1.05,{y0 - 3.5}) "
            r"{source $\downarrow$};"
        )
        for x, target in enumerate(LANGS):
            lines.append(
                f"\\node[rotate=45,anchor=west,font=\\scriptsize] at ({x + 0.18},{y0 + 0.25}) "
                f"{{\\texttt{{{target}}}}};"
            )
        lines.append(
            f"\\node[rotate=45,anchor=west,font=\\tiny,text=black!75] at (7.38,{y0 + 0.25}) "
            r"{source mean};"
        )
        for y, source in enumerate(LANGS):
            lines.append(
                f"\\node[anchor=east,font=\\scriptsize] at (-0.15,{y0 - y - 0.5}) "
                f"{{\\texttt{{{source}}}}};"
            )
            for x, target in enumerate(LANGS):
                raw = lookup.get((model, source, target))
                if raw in (None, ""):
                    fill = "gray!15"
                    label = "--"
                else:
                    value = float(raw)
                    fill = auc_fill(value)
                    label = f"{value:.2f}"
                lines.append(f"\\fill[{fill}] ({x},{y0 - y}) rectangle ++(1,-1);")
                lines.append(f"\\draw[white,line width=0.25pt] ({x},{y0 - y}) rectangle ++(1,-1);")
                lines.append(
                    f"\\node[font=\\scriptsize] at ({x + 0.5},{y0 - y - 0.5}) {{{label}}};"
                )
            source_mean = float(profile_lookup[(model, source)]["mean_auc_as_source"])
            source_fill = auc_fill(source_mean)
            lines.append(f"\\fill[{source_fill}] (7.35,{y0 - y}) rectangle ++(1,-1);")
            lines.append(f"\\draw[white,line width=0.25pt] (7.35,{y0 - y}) rectangle ++(1,-1);")
            lines.append(
                f"\\node[font=\\scriptsize\\bfseries] at (7.85,{y0 - y - 0.5}) "
                f"{{{source_mean:.2f}}};"
            )
        lines.append(
            f"\\node[anchor=east,font=\\tiny,text=black!75] at (-0.15,{y0 - 7.5}) "
            r"{target mean};"
        )
        for x, target in enumerate(LANGS):
            target_mean = float(profile_lookup[(model, target)]["mean_auc_as_target"])
            target_fill = auc_fill(target_mean)
            lines.append(f"\\fill[{target_fill}] ({x},{y0 - 7.15}) rectangle ++(1,-0.70);")
            lines.append(f"\\draw[white,line width=0.25pt] ({x},{y0 - 7.15}) rectangle ++(1,-0.70);")
            lines.append(
                f"\\node[font=\\scriptsize\\bfseries] at ({x + 0.5},{y0 - 7.50}) "
                f"{{{target_mean:.2f}}};"
            )
    lines.extend(
        [
            r"\node[font=\scriptsize,anchor=east] at (1.0,-28.85) {AUC:};",
            r"\fill[teal!8!white] (1.4,-29.12) rectangle ++(0.65,0.52);",
            r"\node[font=\scriptsize,anchor=west] at (2.25,-28.86) {$0.50$};",
            r"\fill[teal!50!white] (5.0,-29.12) rectangle ++(0.65,0.52);",
            r"\node[font=\scriptsize,anchor=west] at (5.85,-28.86) {$0.75$};",
            r"\fill[teal!92!white] (1.4,-30.27) rectangle ++(0.65,0.52);",
            r"\node[font=\scriptsize,anchor=west] at (2.25,-30.01) {$1.00$};",
            r"\fill[red!45!white] (5.0,-30.27) rectangle ++(0.65,0.52);",
            r"\node[font=\scriptsize,anchor=west] at (5.85,-30.01) {$<0.50$};",
            r"\end{tikzpicture}",
            r"\caption[Cross-lingual transfer matrices for refusal-outcome directions]{Cross-lingual transfer matrices for all 126 ordered language pairs. Each row estimates a direction in the source language; each column evaluates it in a different target language. Cells report AUC and the diagonal is omitted because within-language values are in-sample. Bold right margins average each source row; bold bottom margins average transfer into each target column.}",
            r"\label{fig:mechanistic-transfer-matrices}",
            r"\fonte{Prepared by the author (2026), using the transfer protocol of \textcite{wangRefusalDirectionUniversal2025}.}",
            r"\end{figure}",
            r"}",
        ]
    )
    return "\n".join(lines)


def transfer_analysis_text(profiles: list[dict[str, object]]) -> str:
    lookup = {(str(row["model"]), str(row["attack_language"])): row for row in profiles}
    ai_rows = [row for row in profiles if row["model"] == "ai_sweden_llama3_8b"]
    ai_low = min(ai_rows, key=lambda row: float(row["mean_auc_as_source"]))
    ai_high = max(ai_rows, key=lambda row: float(row["mean_auc_as_source"]))
    ref_swa = lookup[("llama3_1_8b_reference", "swa")]
    anita_swa = lookup[("llamantino_anita_8b", "swa")]
    return "\n".join(
        [
            r"\newcommand{\MechTransferAnalysisText}{%",
            r"Figure~\ref{fig:mechanistic-transfer-matrices} expands the mean transfer scores into all 126 ordered source--target language pairs. Reading across a row asks whether a direction estimated in one source language transfers broadly; reading down a column asks whether a target language is separable under directions learned elsewhere. The reference and ANITA have uniformly strong rows, whereas AI Sweden is heterogeneous: its "
            + f"{LANGUAGE_NAMES[str(ai_low['attack_language'])]} (\\texttt{{{ai_low['attack_language']}}}) source direction has the lowest mean transfer AUC "
            + f"(${float(ai_low['mean_auc_as_source']):.2f}$), while its {LANGUAGE_NAMES[str(ai_high['attack_language'])]} (\\texttt{{{ai_high['attack_language']}}}) source direction has the highest "
            + f"(${float(ai_high['mean_auc_as_source']):.2f}$). The source--target distinction also reveals a pattern hidden by a single mean. "
            + f"For the reference, Swahili is a strong \\emph{{source}} direction (mean transfer ${float(ref_swa['mean_auc_as_source']):.2f}$) "
            + f"but a difficult \\emph{{target}} (mean transfer-in ${float(ref_swa['mean_auc_as_target']):.2f}$); "
            + f"ANITA shows the same asymmetry (${float(anita_swa['mean_auc_as_source']):.2f}$ versus "
            + f"${float(anita_swa['mean_auc_as_target']):.2f}$). This indicates that a language can yield a portable contrast even when examples in that language remain comparatively difficult to separate with directions estimated elsewhere.",
            r"}",
        ]
    )


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
            r"\caption[Refusal-signal separability versus ASR]{In-sample refusal-signal separability versus behavioral ASR across 21 model--language cells. The pooled association is negative, but the within-model direction is heterogeneous; Spearman coefficients are reported in the text.}",
            r"\label{fig:mechanistic-separability-vs-asr}",
            r"\fonte{Prepared by the author (2026).}",
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
        r"xlabel={Residual-stream layer},ylabel={Mean split-sample AUC},",
        r"label style={font=\scriptsize},tick label style={font=\tiny}]",
        f"\\addplot+[mark=*,mark size=1.4pt,black!75!white] coordinates {{{coords}}};",
    ]
    if selected:
        lines.append(f"\\draw[densely dashed,red!70!black] (axis cs:{selected[0]},\\pgfkeysvalueof{{/pgfplots/ymin}}) -- (axis cs:{selected[0]},\\pgfkeysvalueof{{/pgfplots/ymax}});")
    lines.extend(
        [
            r"\end{axis}",
            r"\end{tikzpicture}",
            r"\caption[Layer sweep for the refusal-outcome direction]{Layer sweep: mean split-sample refused-versus-complied AUC across the three models. The dashed line marks the layer selected using these same test splits, so the peak is a selection criterion rather than an independent performance estimate.}",
            r"\label{fig:mechanistic-layer-sweep-legacy}",  # superseded; label kept distinct
            r"\fonte{Prepared by the author (2026).}",
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
        r"\caption[Refusal-outcome direction diagnostics per model--language cell]{Refusal-outcome direction diagnostics per model--language cell: in-sample within-language projection AUC, mean cross-lingual transfer AUC into the cell, cosine to the aligned-language direction, and frozen behavioral ASR.}",
        r"\label{tab:mechanistic-refusal-projections-legacy}",  # superseded; label kept distinct
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
            first = f"{DISPLAY[model]}" if index == 0 else ""
            lines.append(
                f"{first} & \\texttt{{{lang}}} & {num(row['auc_refusal_vs_unsafe'])} & "
                f"{num(mean_transfer)} & {num(cosine.get((model, lang)))} & {pct(asr.get((model, lang)))}\\% \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.extend(
        [
            r"\end{tabular}",
            r"\fonte{Prepared by the author (2026).}",
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
            first = f"{DISPLAY[model]}" if index == 0 else ""
            lines.append(
                f"{first} & \\texttt{{{lang}}} & {num(row['cosine_with_reference_language'])} & "
                f"{num(row['cosine_with_pooled_direction'])} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.extend(
        [
            r"\end{tabular}",
            r"\fonte{Prepared by the author (2026).}",
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
        r"\caption[Addition-only refusal-strengthening intervention outcomes]{Addition-only intervention on previously unsafe prompts from the direction-estimation sample. Explicit refusal requires agreement between the StrongREJECT label and fixed marker matching on the backtranslated output. Manual inspection found substantial degeneration at high $\alpha$; still-unsafe uses the StrongREJECT score alone and zero can therefore include refusal, irrelevant output, or degeneration. Other means non-refusal zero-score output; Failed is a count, and the remaining outcome columns are percentages.}",
        r"\label{tab:mechanistic-intervention-legacy}",  # superseded; label kept distinct
        r"\setlength{\tabcolsep}{3pt}\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Model & $\alpha$ & $n$ & Explicit & Judge-only & Unsafe & Other & Failed \\",
        r"\midrule",
    ]
    for model in MODELS:
        model_rows = [r for r in rows if r["model"] == model]
        for index, row in enumerate(model_rows):
            first = f"{DISPLAY[model]}" if index == 0 else ""
            lines.append(
                f"{first} & {row['alpha']} & {row['n']} & {pct(row['explicit_refusal_rate'])}\\% & "
                f"{pct(row['judge_only_refusal_rate'])}\\% & {pct(row['still_unsafe_rate'])}\\% & "
                f"{pct(row['other_rate'])}\\% & {row['scoring_failed_n']} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.extend(
        [
            r"\end{tabular}",
            r"\fonte{Prepared by the author (2026).}",
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
    profiles = transfer_profiles(transfer)
    write_transfer_profiles(profiles)
    print(f"Wrote {TRANSFER_PROFILE_CSV}")
    interventions = intervention_summary()
    if interventions:
        write_summary_csv(interventions)
        print(f"Wrote {SUMMARY_CSV}")

    parts = [
        "% Generated by eval/scripts/write_mechanistic_latex_assets.py from the",
        "% mechanistic-results artifacts. Figure and table macros for the exploratory",
        "% mechanistic chapter; invoke e.g. \\MechDirectionSimilarityFigure.",
        direction_construction_schematic(),
        similarity_heatmap(similarity),
        transfer_matrices(transfer, profiles),
        transfer_analysis_text(profiles),
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
