#!/usr/bin/env bash
# Rebuild the precompiled pgfplots figures from their sources in figures/.
# Run from anywhere; requires pdflatex. Outputs replace the PDFs in this folder.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
thesis="$(cd "$here/../.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
for name in strongreject_profiles belebele_profiles graded_spec_reference_scatter reference_profiles attack_language_map; do
  cat > "$tmp/$name.tex" <<TEX
\documentclass[border=2pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[scaled]{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{xcolor}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{calc,positioning,arrows.meta,shapes,backgrounds}
\begin{document}
\input{figures/$name.tex}
\end{document}
TEX
  (cd "$thesis" && pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$tmp" "$tmp/$name.tex" >/dev/null)
  cp "$tmp/$name.pdf" "$here/$name.pdf"
  echo "rebuilt $name.pdf"
done
