"""Export traceable sparse-CONS examples without modifying saved predictions."""
from pathlib import Path
from collections import Counter
import hashlib,json,subprocess,sys
from thesis_eval.benchmarks.belebele import migrate_prediction_identities
from thesis_eval.io import read_jsonl
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'eval/outputs/spec/llamantino_2_ultrachat_7b.belebele.jsonl'
DEST=ROOT/'ufscthesisx/ufscthesisx/aftertext/apendice_d.tex'
OUT=ROOT/'eval/outputs/releases/2026-10-06-belebele-sparse-examples'
OUT.mkdir(exist_ok=True)
rows=migrate_prediction_identities(read_jsonl(SRC),ROOT/'eval/assets/datasets/belebele')
def esc(s):
 return str(s).replace('&',r'\&').replace('_',r'\_').replace('%',r'\%').replace('#',r'\#')

# --- Evidence that the output cap is uniform and therefore cannot explain the
# --- differences between models or between languages. Computed from the saved
# --- outputs of every model so no figure below is hand-entered.
import re
MODELS={'sagui_7b':'Sagui-7B','sabia_3':'Sabia-3','llamantino_2_ultrachat_7b':'LLaMAntino-2',
 'llamantino_anita_8b':'ANITA','gpt_sw3':'GPT-SW3','ai_sweden_llama3_8b':'AI Sweden',
 'bggpt_7b':'BgGPT-7B','bggpt_gemma_9b':'BgGPT-Gemma','llama3_1_8b_reference':'English reference'}
caps={}
for key in MODELS:
 saved=read_jsonl(ROOT/f'eval/outputs/spec/{key}.belebele.jsonl')
 lengths=[len((r.get('raw_output') or '').strip()) for r in saved]
 words=[len((r.get('raw_output') or '').split()) for r in saved]
 parsed=sum(r.get('predicted_choice') is not None for r in saved)
 caps[key]=dict(max_chars=max(lengths),max_words=max(words),parse=parsed/len(saved))
cap_chars=max(v['max_chars'] for v in caps.values())
cap_words=max(v['max_words'] for v in caps.values())
others={k:v for k,v in caps.items() if k!='llamantino_2_ultrachat_7b'}
worst_other=min(v['parse'] for v in others.values())
# Verbosity is not the discriminator: this model matches LLaMAntino-2's output
# length and still parses completely.
verbose_peer=max(others,key=lambda k:caps[k]['max_chars'])
by_lang={}
for lang in sorted({r['language'] for r in rows}):
 group=[r for r in rows if r['language']==lang]
 by_lang[lang]=dict(parse=sum(r['predicted_choice'] is not None for r in group)/len(group),
  correct=sum(r['predicted_choice']==r['gold'] for r in group)/len(group))
ordered=sorted(by_lang,key=lambda l:-by_lang[l]['parse'])
aligned_rank=ordered.index('ita')+1
# An answer-intent preamble would plausibly complete with more tokens; a
# disclaimer or incapacity statement would not.
ANSWER=re.compile(r'^(la (risposta|scelta|soluzione)|la mia ris|la respuesta|svar|svaret|the answer|vastaus|jibu|kwa muj)',re.I)
NONANSWER=re.compile(r"^(sono un|non ho|non posso|non sono|mi piace|come (modello|assistente)|i am an?|i cannot|as an ai|mi dispiace)",re.I)
unparsed_all=[r for r in rows if r['predicted_choice'] is None]
recoverable=sum(bool(ANSWER.match((r['raw_output'] or '').strip())) for r in unparsed_all)
nonanswer=sum(bool(NONANSWER.match((r['raw_output'] or '').strip())) for r in unparsed_all)
ara_nonanswer=sum(bool(NONANSWER.match((r['raw_output'] or '').strip()))
 for r in unparsed_all if r['language']=='ara')
total_rows=len(rows)
text=[r'''\section{Why the weak Italian model has sparse comparable answers}
\label{app:belebele-sparse-answers}

Here ``parseable'' means that the saved output can be mapped to a single answer option, not that the text is legible or grammatically correct. The parser first looks for an A/B/C/D label and otherwise checks normalized option text, including a unique partial-text match. A sentence can therefore be readable but fail to identify any option. Conversely, a short option-text fragment can be parseable without forming a complete sentence.

LLaMAntino-2 has 900 saved outputs in each language, all marked as completed generations. Only 50 Italian and 20 Russian outputs are parseable, leaving 850 and 880 unparsed, respectively. Many outputs are Italian sentence openings without an answer label, including outputs to Russian questions. Table~\ref{tab:belebele-unparsed-fragments} shows the three most frequent unparsed outputs in each language. Quoted fragments reproduce the complete saved text after removing leading whitespace; they are not excerpts from longer saved answers.

A hard output cap terminated these generations. The fragments break mid-word, as in \texttt{cor} for \emph{corretta} and \texttt{ris} for \emph{risposta}, which an end-of-sequence stop would not produce, and the prediction command defaults to an eight-token budget. Across all nine models no saved output exceeds '''+str(cap_chars)+r''' characters or '''+str(cap_words)+r''' whitespace-separated tokens, confirming that the cap bound every model. The saved rows do not retain the budget, token count or stopping reason, so the cap is inferred from the outputs rather than read from the records.

The cap alone cannot explain these failures, for two reasons. First, it was uniform across models, yet every other model parses at least '''+f'{100*worst_other:.1f}'+r'''\% of its outputs. Output length is not the discriminator either: '''+MODELS[verbose_peer]+r''' reaches '''+str(caps[verbose_peer]['max_chars'])+r''' characters, comparable to LLaMAntino-2, and still parses completely. What differs is the content of those eight tokens. This model opens with an Italian answer preamble that leaves no room for the option, while the others emit the label immediately.

Second, the cap was also uniform across languages, yet this model's parse coverage varies about thirtyfold between them (Table~\ref{tab:belebele-parse-coverage}). A constant cannot produce that spread, and its direction is the substantive problem: Italian, the model's own aligned language, ranks '''+str(aligned_rank)+r''' of '''+str(len(ordered))+r''' by parse coverage. The reason is visible in the fragments: the model writes its longest, most fluent preamble in Italian, a short native preamble in Swedish, and short degenerate text in Swahili that incidentally matches option text. Parse success therefore tracks preamble verbosity, which tracks fluency, so IF for this model partly measures the inverse of language competence. The prompt's formatting instruction is also in English for all thirteen languages, so parse success additionally depends on English meta-instruction compliance.

A larger budget would not repair the measurement. Of the '''+f'{len(unparsed_all):,}'.replace(',','\\,')+r''' unparsed rows, roughly '''+f'{recoverable:,}'.replace(',','\\,')+r''' are answer-intent preambles that more tokens would plausibly complete, but about '''+f'{nonanswer:,}'.replace(',','\\,')+r''' are semantically complete non-answers such as \texttt{Sono un model}\ldots{} (``I am a \ldots{} model'') and \texttt{Non ho accesso} (``I do not have access'') that no budget converts into an answer; '''+str(ara_nonanswer)+r''' of those are Arabic. These failures should therefore not be read as lack of comprehension in the question language, nor as a problem a rerun with more tokens would fully solve. Section~\ref{sec:future-belebele-likelihood} describes the measurement change that would.

\begin{table}[htbp]\centering\small
\begin{tabularx}{\linewidth}{Xrr}
\caption[LLaMAntino-2 parse coverage by language]{LLaMAntino-2 answer-parse coverage by question language, with the accuracy that coverage produces. The aligned language is Italian.}\label{tab:belebele-parse-coverage}\\
\toprule
Question language & Parseable & IF\\
\midrule
'''+'\n'.join(f"{lang}{' (aligned)' if lang=='ita' else ''} & {100*by_lang[lang]['parse']:.1f}\\% & {by_lang[lang]['correct']:.3f}"+r'\\' for lang in ordered)+r'''
\bottomrule\end{tabularx}
\par\smallskip\footnotesize A uniform output cap applies to every row, so it cannot account for differences between these languages. IF counts an unparsed answer as incorrect.
\fonte{Prepared by the author (2026), based on the immutable saved predictions.}
\end{table}

\begin{table}[htbp]\centering\small
\begin{tabularx}{\linewidth}{llrX}
\caption{Most frequent unparsed LLaMAntino-2 answer fragments}\label{tab:belebele-unparsed-fragments}\\
\toprule
Question language & Complete saved output & Count & English gloss\\
\midrule''']
exports=[]
gloss={'La risposta è':'The answer is','La risposta cor':'The correct ans…','La mia ris':'My ans…','La scelta cor':'The correct choice…'}
for lang,name in [('ita','Italian'),('rus','Russian')]:
 unparsed=[r for r in rows if r['language']==lang and r['predicted_choice'] is None]
 for raw,n in Counter(r['raw_output'] for r in unparsed).most_common(3):
  text.append(f'{name} & \\texttt{{{esc(raw.strip())}}} & {n} & {gloss[raw.strip()]}'+r'\\')
  examples=[r for r in unparsed if r['raw_output']==raw]
  exports.append(dict(language=lang,raw_output=raw,count=n,canonical_item_ids=[r['canonical_item_id'] for r in examples],original_row_indices=[r['belebele_row_index'] for r in examples]))
text += [r'''\bottomrule\end{tabularx}
\fonte{Prepared by the author (2026), based on the immutable saved predictions after the canonical identity migration.}
\end{table}

Only four canonical questions have parseable answers in both Russian and Italian. Table~\ref{tab:belebele-four-pairs} lists all four, with the original language-specific row numbers as retrieval anchors. Those row numbers differ because the files have different orderings; matching uses canonical question identity. Two pairs agree on a wrong answer and two agree on the supplied gold. Consequently, CONS $=1.00$ means agreement on this tiny selected subset, not perfect accuracy or evidence of strong Russian specialization. The 95\% Wilson interval $[0.510,1.000]$ is conditional on these four comparable questions and does not account for selective parsing failures.

\begin{table}[htbp]\centering\small
\setlength{\tabcolsep}{4pt}
\begin{tabularx}{\linewidth}{rrXXcc}
\caption{All four comparable Russian--Italian LLaMAntino-2 answer pairs}\label{tab:belebele-four-pairs}\\
\toprule
RU row & IT row & Russian-question output & Italian-question output & Choice & Gold\\
\midrule''']
ita={r['canonical_item_id']:r for r in rows if r['language']=='ita'}
pairs=[]
for r in rows:
 if r['language']!='rus' or r['predicted_choice'] is None:continue
 b=ita[r['canonical_item_id']]
 if b['predicted_choice'] is None:continue
 assert r['predicted_choice']==b['predicted_choice']
 text.append(f"{r['belebele_row_index']} & {b['belebele_row_index']} & \\texttt{{{esc(r['raw_output'].strip())}}} & \\texttt{{{esc(b['raw_output'].strip())}}} & {r['predicted_choice']} & {r['gold']}"+r'\\')
 pairs.append({k:r[k] for k in ['canonical_item_id','belebele_row_index','raw_output','predicted_choice','gold'] } | {'italian_row_index':b['belebele_row_index'],'italian_raw_output':b['raw_output'],'italian_gold':b['gold']})
assert len(pairs)==4
text += [r'''\bottomrule\end{tabularx}
\par\smallskip\footnotesize Both language versions have the displayed gold label. Option-text outputs are mapped against their own language's source options; ``Skype.'' and ``La medaglia d'' illustrate text matching without an explicit letter. Full raw strings and canonical identities are exported in the versioned examples release. No answer or gold label is changed.
\fonte{Prepared by the author (2026), based on saved predictions matched by canonical question identity.}
\end{table}''']
start='% BEGIN GENERATED SPARSE CONS EXAMPLES'
end='% END GENERATED SPARSE CONS EXAMPLES'
s=DEST.read_text()
if start in s:s=s[:s.index(start)]+s[s.index(end)+len(end):]
anchor=r'\section{Complete specialization coverage}'
s=s.replace(anchor,start+'\n'+'\n\n'.join(text)+'\n'+end+'\n\n'+anchor)
DEST.write_text(s)
(OUT/'examples.json').write_text(json.dumps({'unparsed_fragments':exports,'comparable_pairs':pairs},indent=2,ensure_ascii=False)+'\n')
(OUT/'manifest.json').write_text(json.dumps({'source_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'examples_sha256':hashlib.sha256((OUT/'examples.json').read_bytes()).hexdigest(),'provenance':'Post-review descriptive inspection; original outputs and gold unchanged; historical stop reasons unknown.'},indent=2)+'\n')
subprocess.run([sys.executable,str(ROOT/'eval/scripts/format_belebele_appendix.py')],check=True)
print('Exported sparse-CONS examples.')
