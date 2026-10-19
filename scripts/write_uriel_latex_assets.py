from __future__ import annotations

import csv
import runpy
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DISTANCE_CSV = ROOT / "eval" / "data" / "uriel_plus" / "distance_matrix.csv"
OUT = ROOT / "ufscthesisx" / "ufscthesisx" / "chapters" / "uriel_generated_assets.tex"

ORDER = ["ita", "por", "spa", "eng", "dan", "nor", "swe", "fin", "bul", "rus", "ukr", "ara", "swa"]

LANG_DISPLAY = {
    "ara": "Arabic",
    "bul": "Bulgarian",
    "dan": "Danish",
    "eng": "English",
    "fin": "Finnish",
    "ita": "Italian",
    "nor": "Norwegian",
    "por": "Portuguese",
    "rus": "Russian",
    "spa": "Spanish",
    "swa": "Swahili",
    "swe": "Swedish",
    "ukr": "Ukrainian",
}

FAMILY = {
    "ita": ("Romance", "UrielRomance", "F7D7CF"),
    "por": ("Romance", "UrielRomance", "F7D7CF"),
    "spa": ("Romance", "UrielRomance", "F7D7CF"),
    "eng": ("Germanic", "UrielGermanic", "D6E6F6"),
    "dan": ("Germanic", "UrielGermanic", "D6E6F6"),
    "nor": ("Germanic", "UrielGermanic", "D6E6F6"),
    "swe": ("Germanic", "UrielGermanic", "D6E6F6"),
    "fin": ("Uralic", "UrielUralic", "D8F0E1"),
    "bul": ("Slavic", "UrielSlavic", "E8DDF5"),
    "rus": ("Slavic", "UrielSlavic", "E8DDF5"),
    "ukr": ("Slavic", "UrielSlavic", "E8DDF5"),
    "ara": ("Semitic", "UrielSemitic", "F8E2C2"),
    "swa": ("Bantu", "UrielBantu", "DCEBC8"),
}

FAMILY_COLORS = {
    "UrielRomance": "D55E00",
    "UrielGermanic": "0072B2",
    "UrielUralic": "009E73",
    "UrielSlavic": "6A51A3",
    "UrielSemitic": "E69F00",
    "UrielBantu": "5A8F29",
    "UrielClose": "355C7D",
    "UrielMiddle": "4ECDC4",
    "UrielFar": "F0932B",
    "SpeakerTiny": "88CCEE",
    "SpeakerSmall": "44AA99",
    "SpeakerMedium": "DDCC77",
    "SpeakerLarge": "CC6677",
    "SpeakerHuge": "882255",
}

SPEAKER_MAP_DATA = {
    "ara": (39.0, 24.0, "250--500M", "SpeakerLarge", 5.0),
    "bul": (25.0, 43.0, "$<10$M", "SpeakerTiny", 2.8),
    "dan": (9.5, 55.8, "$<10$M", "SpeakerTiny", 2.8),
    "eng": (-2.0, 54.0, "$>500$M", "SpeakerHuge", 6.0),
    "fin": (26.0, 64.0, "$<10$M", "SpeakerTiny", 2.8),
    "ita": (12.0, 43.0, "50--250M", "SpeakerMedium", 4.1),
    "nor": (8.5, 61.5, "$<10$M", "SpeakerTiny", 2.8),
    "por": (-47.0, -15.0, "250--500M", "SpeakerLarge", 5.0),
    "rus": (37.0, 56.0, "250--500M", "SpeakerLarge", 5.0),
    "spa": (-99.0, 19.0, "$>500$M", "SpeakerHuge", 6.0),
    "swa": (37.0, -6.0, "50--250M", "SpeakerMedium", 4.1),
    "swe": (16.0, 62.0, "10--50M", "SpeakerSmall", 3.4),
    "ukr": (31.0, 49.0, "10--50M", "SpeakerSmall", 3.4),
}


def load_matrix() -> tuple[list[str], dict[str, dict[str, float]]]:
    with DISTANCE_CSV.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)[1:]
        matrix: dict[str, dict[str, float]] = {}
        for row in reader:
            matrix[row[0]] = {lang: float(value) for lang, value in zip(header, row[1:])}
    return header, matrix


def interpolate(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def distance_color(value: float, low: float, high: float) -> str:
    stops = [
        (0.00, (53, 92, 125)),
        (0.45, (78, 205, 196)),
        (0.72, (246, 229, 141)),
        (1.00, (240, 147, 43)),
    ]
    t = max(0.0, min(1.0, (value - low) / (high - low)))
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        if t <= p1:
            local = (t - p0) / (p1 - p0)
            rgb = interpolate(c0, c1, local)
            return "".join(f"{channel:02X}" for channel in rgb)
    return "".join(f"{channel:02X}" for channel in stops[-1][1])


def text_color(hex_color: str) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 115 else "black"


def table(matrix: dict[str, dict[str, float]]) -> str:
    values = [matrix[a][b] for a in ORDER for b in ORDER if a != b]
    low, high = min(values), max(values)
    lines = [
        r"\newcommand{\UrielDistanceMatrixTable}{%",
        r"\begin{landscape}",
        r"\begin{table}[p]",
        r"\centering",
        r"\caption[URIEL+ featural distance matrix]{URIEL+ featural angular distance matrix for the selected attack languages. Lower values indicate closer languages; higher values indicate greater typological distance.}",
        r"\label{tab:urielplus-distances}",
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\renewcommand{\arraystretch}{1.18}",
        r"\begin{adjustbox}{width=0.9\linewidth,center}",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{2.85cm}|*{13}{>{\centering\arraybackslash}p{0.88cm}}}",
        r"\hline",
        r"\textbf{Language} & " + " & ".join(f"\\cellcolor[HTML]{{{FAMILY[lang][2]}}}\\textbf{{\\texttt{{{lang}}}}}" for lang in ORDER) + r" \\",
        r"\hline",
    ]
    for lang in ORDER:
        row = [
            f"\\cellcolor[HTML]{{{FAMILY[lang][2]}}}\\textbf{{\\mbox{{{LANG_DISPLAY[lang]} (\\texttt{{{lang}}})}}}}"
        ]
        for other in ORDER:
            if lang == other:
                row.append(r"\cellcolor[HTML]{E6E6E6}\textemdash")
            else:
                color = distance_color(matrix[lang][other], low, high)
                row.append(
                    f"\\cellcolor[HTML]{{{color}}}\\textcolor{{{text_color(color)}}}{{{matrix[lang][other]:.3f}}}"
                )
        lines.append(" & ".join(row) + r" \\")
        if lang in {"spa", "swe", "fin", "ukr", "ara"}:
            lines.append(r"\hline")
    lines.extend(
        [
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\vspace{0.55em}",
            r"\begin{tikzpicture}[x=1cm,y=0.28cm]",
            r"\fill[UrielClose] (0,0) rectangle (0.55,1);\node[anchor=west,font=\scriptsize] at (0.65,0.5) {closer};",
            r"\fill[UrielMiddle] (2.1,0) rectangle (2.65,1);\node[anchor=west,font=\scriptsize] at (2.75,0.5) {intermediate};",
            r"\fill[UrielFar] (5.0,0) rectangle (5.55,1);\node[anchor=west,font=\scriptsize] at (5.65,0.5) {farther};",
            r"\node[anchor=west,font=\scriptsize] at (8.25,0.5) {Rows and columns are grouped by broad family only for readability.};",
            r"\end{tikzpicture}",
            r"\par\fonte{Prepared by the author (2026), based on URIEL+ featural angular distances from \textcite{khan-etal-2025-uriel}.}",
            r"\endgroup",
            r"\end{table}",
            r"\end{landscape}",
            r"}",
        ]
    )
    return "\n".join(lines)


def classical_mds(langs: list[str], matrix: dict[str, dict[str, float]]) -> tuple[dict[str, tuple[float, float]], float]:
    d = np.array([[matrix[a][b] for b in langs] for a in langs], dtype=float)
    n = d.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (d**2) @ j
    values, vectors = np.linalg.eigh(b)
    idx = np.argsort(values)[::-1]
    values = values[idx]
    vectors = vectors[:, idx]
    positive = np.maximum(values, 0)
    coords = vectors[:, :2] * np.sqrt(positive[:2])
    explained = float(positive[:2].sum() / positive.sum())
    return {lang: (float(x), float(y)) for lang, (x, y) in zip(langs, coords)}, explained


def mds_figure(matrix: dict[str, dict[str, float]]) -> str:
    coords, explained = classical_mds(ORDER, matrix)
    lines = [
        r"\newcommand{\UrielLanguageMapFigure}{%",
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"width=0.95\linewidth,height=8.2cm,",
        r"axis lines=left,axis equal image,",
        r"xlabel={MDS dimension 1},ylabel={MDS dimension 2},",
        r"xmin=-0.48,xmax=0.45,ymin=-0.28,ymax=0.29,",
        r"grid=major,grid style={black!8},",
        r"tick label style={font=\scriptsize},",
        r"label style={font=\scriptsize},",
        r"legend columns=3,",
        r"legend style={font=\scriptsize,draw=none,fill=none,at={(0.5,-0.15)},anchor=north,",
        r"/tikz/every even column/.append style={column sep=0.35cm}},",
        r"]",
    ]
    for color, name in [
        ("UrielRomance", "Romance"),
        ("UrielGermanic", "Germanic"),
        ("UrielSlavic", "Slavic"),
        ("UrielUralic", "Uralic"),
        ("UrielSemitic", "Semitic"),
        ("UrielBantu", "Bantu"),
    ]:
        lines.append(f"\\addlegendimage{{only marks,mark=*,mark options={{fill={color},draw={color}}}}}")
        lines.append(f"\\addlegendentry{{{name}}}")
    label_offsets = {
        "ara": (0, 13), "bul": (32, -3), "dan": (0, -14),
        "eng": (0, 13), "fin": (23, -3), "ita": (-26, 13),
        "nor": (0, 13), "por": (-37, -6), "rus": (0, -14),
        "spa": (26, -12), "swa": (0, 13), "swe": (27, 0),
        "ukr": (-34, 10),
    }
    for lang in ORDER:
        x, y = coords[lang]
        _, color, _ = FAMILY[lang]
        dx, dy = label_offsets[lang]
        lines.append(
            f"\\addplot+[only marks,mark=*,mark size=2.8pt,mark options={{fill={color},draw=black,line width=0.25pt}}] coordinates {{({x:.4f},{y:.4f})}};"
        )
        lines.append(
            f"\\draw[black!30,thin] (axis cs:{x:.4f},{y:.4f}) -- ([xshift={dx}pt,yshift={dy}pt]axis cs:{x:.4f},{y:.4f});\\node[font=\\scriptsize,fill=white,inner sep=1.5pt,anchor=center] at ([xshift={dx}pt,yshift={dy}pt]axis cs:{x:.4f},{y:.4f}) {{{LANG_DISPLAY[lang]}}};"
        )
    lines.extend(
        [
            r"\end{axis}",
            r"\end{tikzpicture}",
            rf"\caption[Two-dimensional URIEL+ language-proximity map]{{Classical multidimensional scaling (MDS) of the URIEL+ featural distance matrix \cite{{gower1966distance}}. Languages are colored by broad family.}}",
            r"\label{fig:uriel-language-map}",
            r"\fonte{Prepared by the author (2026), based on URIEL+ featural angular distances from \textcite{khan-etal-2025-uriel}.}",
            r"\end{figure}",
            r"}",
        ]
    )
    return "\n".join(lines)


def sociolinguistic_map_figure() -> str:
    return r"""\newcommand{\AttackLanguageSpeakerMapFigure}{%
\begin{figure}[htbp]\centering
\includegraphics[max width=\linewidth]{figures/compiled/attack_language_map.pdf}
\caption[Attack-language speaker map]{Representative speaker communities for the thirteen attack languages, with a European inset. Transnational languages appear at multiple locations; color identifies the language and marker size encodes its total-speaker bin.}
\label{fig:attack-language-speaker-map}
\fonte{Prepared by the author (2026), based on Natural Earth country boundaries and speaker estimates from \textcite{eberhard2025ethnologue}; reference locations informed by \textcite{hammarstrom2026glottolog}.}
\end{figure}}
"""


def write_assets() -> None:
    runpy.run_path(str(Path(__file__).with_name("write_speaker_map.py")))
    _, matrix = load_matrix()
    definitions = ["% Generated by eval/scripts/write_uriel_latex_assets.py. Do not edit by hand."]
    for name, value in FAMILY_COLORS.items():
        definitions.append(f"\\definecolor{{{name}}}{{HTML}}{{{value}}}")
    definitions.append("")
    definitions.append(table(matrix))
    definitions.append("")
    definitions.append(mds_figure(matrix))
    definitions.append("")
    definitions.append(sociolinguistic_map_figure())
    OUT.write_text("\n".join(definitions) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    write_assets()
