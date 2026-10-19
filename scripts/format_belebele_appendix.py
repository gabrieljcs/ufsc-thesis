"""Make BELEBELE appendix tables page-breakable with repeated column headers."""
from pathlib import Path
import re
P=Path(__file__).resolve().parents[2]/'ufscthesisx/ufscthesisx/aftertext/apendice_d.tex'
s=P.read_text()
def convert(m):
 block=m.group(0)
 spec=re.search(r'\\begin\{tabularx\}\{\\linewidth\}\{([^\n]+)\}',block).group(1)
 if spec=='llrX':spec=r'p{2.1cm}p{4.0cm}rp{4.4cm}'
 elif spec=='rrXXcc':spec=r'rrp{4.2cm}p{4.2cm}cc'
 else:spec=spec.replace('X','l')
 caption=block[block.index(r'\caption'):block.index(r'\toprule')]
 rest=block[block.index(r'\toprule'):block.index(r'\end{tabularx}')]
 split=rest.index(r'\midrule')+len(r'\midrule')
 head=rest[:split]
 body=rest[split:]
 # Column count comes from the header row. A hardcoded table of specs silently
 # mislabels any new table, and \multicolumn wider than the table is a LaTeX error.
 n=head.split(r'\toprule')[1].split(r'\\')[0].count('&')+1
 trailing=block[block.index(r'\end{tabularx}')+len(r'\end{tabularx}'):block.rindex(r'\end{table}')]
 return '\n'.join([r'\Needspace{18\baselineskip}\begingroup\footnotesize\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.15}',r'\begin{longtable}{'+spec+'}',caption,head,r'\endfirsthead',r'\multicolumn{'+str(n)+r'}{l}{\textit{Continued from previous page}}\\',head,r'\endhead',body,r'\end{longtable}',trailing,r'\endgroup'])
s=re.sub(r'\\begin\{table\}\[htbp\].*?\\end\{table\}',convert,s,flags=re.S)
# Repair historical dangling reference to an archived binary sensitivity table.
s=s.replace('Table~\\ref{tab:glmm-correction-sensitivity}','Table~\\ref{tab:graded-sensitivity}')
P.write_text(s)
print('Formatted BELEBELE appendix tables.')
