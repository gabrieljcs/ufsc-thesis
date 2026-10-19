"""Check/fix presentation consistency in TeX files used by the latest thesis build.

Run after asset generators, before latexmk. Does not alter datasets, plot coordinates,
identifiers, years, bibliographic records, or the embedded published-paper annex.
"""
from pathlib import Path
import argparse, collections, hashlib, json, re
ROOT=Path(__file__).resolve().parents[2]
THESIS=ROOT/'ufscthesisx/ufscthesisx'
RELEASE=ROOT/'eval/outputs/releases/2026-10-12-thesis-consistency'
DISPLAY={
 'Reference','English reference','ANITA','ANITA-8B','ANITA-8B (safer)','ANITA (safer)','AI Sweden',
 'AI Sweden Llama-3','AI Sweden Llama-3 (vulnerable)','AI Sweden (vulnerable)',
 'AI Sweden Llama-3-8B-instruct','AI Sweden LLaMA-3-8B-instruct',
 'Sabiá-3',r"Sabi\'a-3",'Sagui','Sagui-7B','LLaMAntino-2','LLaMAntino-3-ANITA',
 'LLaMAntino-3-ANITA-8B','Sagui-7B-Instruct','GPT-SW3','BgGPT-7B',
 'BgGPT-Gemma','BgGPT-Gemma-9B','BgGPT-Gemma-2','LLaMA-3.1-8B reference',
 'LLaMA-3.1-8B','Llama-3','LLaMA-2','LLaMA-3','Gemma-2',
 'Llama-2-7B','Llama-3-8B','Vicuna-7B','Qwen1.5-7B','Mistral-7B'
}
TECHNICAL_ALIASES={'BgGPT-Gemma-2-9B-Instruct':'BgGPT-Gemma-2-9B-IT-v1.0',
 'LLaMA-3.1-8B-Instruct':'Llama-3.1-8B-Instruct',
 'GPT-SW3-6.7B-v2-instruct':'gpt-sw3-6.7b-v2-instruct'}
ALIASES={'ANITA-8B':'ANITA','LLaMA-3.1-8B reference':'English reference',
 'AI Sweden Llama-3 (vulnerable)':'AI Sweden (vulnerable)',
 'AI Sweden Llama-3-8B-instruct':'AI Sweden','AI Sweden LLaMA-3-8B-instruct':'AI Sweden',
 'AI Sweden Llama-3':'AI Sweden','BgGPT-Gemma-9B':'BgGPT-Gemma',
 'LLaMAntino-3-ANITA-8B':'ANITA','LLaMAntino-3-ANITA':'ANITA'}
def group_end(s,start,opening='{',closing='}'):
    depth=0
    for i in range(start,len(s)):
        if s[i] in (opening,closing) and (i==0 or s[i-1]!='\\'):
            depth+=1 if s[i]==opening else -1
            if depth==0:return i+1
    raise ValueError('Unbalanced TeX group')
def command_end(s,start):
    m=re.match(r'\\[A-Za-z]+\*?',s[start:]);pos=start+m.end()
    while pos<len(s) and s[pos].isspace():pos+=1
    if pos<len(s) and s[pos]=='[':
        pos=group_end(s,pos,'[',']')
        while pos<len(s) and s[pos].isspace():pos+=1
    return group_end(s,pos) if pos<len(s) and s[pos]=='{' else pos
PROTECTED=re.compile(r'\\(?:texttt|url|href|cite\w*|textcite|parencite|autocite|label|ref|cref|Cref|eqref|input|include|includegraphics|num|SI|qty|definecolor)\b')
def protected(s):
    spans=[]
    for m in PROTECTED.finditer(s):
        if any(a<=m.start()<b for a,b in spans):continue
        end=command_end(s,m.start())
        # Multi-argument commands: preserve paths, RGB/HTML values, etc.
        if s[m.start():].startswith(('\\definecolor','\\href')):
            count=2 if s[m.start():].startswith('\\definecolor') else 1
            for _ in range(count):
                while end<len(s) and s[end].isspace():end+=1
                if end<len(s) and s[end]=='{':end=group_end(s,end)
        spans.append((m.start(),end))
    for m in re.finditer(r'\\begin\{(lstlisting|verbatim|Verbatim|minted)\}',s):
        end=s.find(r'\end{'+m.group(1)+'}',m.end())
        if end>=0:spans.append((m.start(),end+len(m.group(0))))
    return spans
def captions(s):
    changes=0
    def move(m):
        nonlocal changes
        body=m.group(2);cap=re.search(r'\\caption(?:\[|\{)',body)
        if not cap:return m.group(0)
        cent=re.match(r'\s*\\centering\b',body)
        insertion=cent.end() if cent else 0
        if not body[insertion:cap.start()].strip():return m.group(0)
        end=command_end(body,cap.start());text=body[cap.start():end]
        rest=body[:cap.start()]+body[end:]
        changes+=1
        return m.group(1)+rest[:insertion]+'\n'+text+'\n'+rest[insertion:]+m.group(3)
    s=re.sub(r'(\\begin\{figure\}(?:\[[^\]]*\])?)(.*?)(\\end\{figure\})',move,s,flags=re.S)
    return s,changes
def normalize(s):
    counts=collections.Counter()
    def display(m):
        name=m.group(1)
        if name in TECHNICAL_ALIASES:
            counts['checkpoint_spelling']+=1
            return r'\texttt{'+TECHNICAL_ALIASES[name]+'}'
        if name in DISPLAY:
            counts['model_display_font']+=1
            return ALIASES.get(name,name)
        return m.group(0)
    s=re.sub(r'\\texttt\{([^{}]+)\}',display,s)
    # Ordinary model display aliases, leaving checkpoint identifiers untouched.
    spans=protected(s)
    replacements=[]
    for old,new in [('ANITA-8B','ANITA'),('BgGPT-Gemma-9B','BgGPT-Gemma')]:
        for m in re.finditer(r'(?<![\w/-])'+re.escape(old)+r'(?![\w/-])',s):
            if not any(a<=m.start()<b for a,b in spans):replacements.append((m.start(),m.end(),new));counts['model_display_alias']+=1
    for a,b,v in reversed(sorted(replacements)):s=s[:a]+v+s[b:]
    spans=protected(s);replacements=[]
    pattern=r'(?<![\w.,/\-])(?:\d{1,3}(?:(?:,|\{,\})\d{3})+|\d{4,})(?![\w/\-]|[.,]\d)'
    for m in re.finditer(pattern,s):
        if any(a<=m.start()<b for a,b in spans):continue
        line=s[s.rfind('\n',0,m.start())+1:s.find('\n',m.end()) if '\n' in s[m.end():] else len(s)]
        if line.lstrip().startswith(('%','\\addplot','\\draw','\\path','\\coordinate','\\foreach')):continue
        prefix=line[:m.start()-(s.rfind('\n',0,m.start())+1)]
        if re.search(r'(?<!\\)%',prefix):continue
        value=m.group().replace('{,}','').replace(',','')
        if value.startswith('0'):continue  # canonical question/prompt identities
        if 1900<=int(value)<=2099 and not (value=='2000' and re.search(r'bootstrap|draw|resampl|replicat|repeat',line,re.I)):continue
        grouped=r'\,'.join([value[max(0,len(value)-i-3):len(value)-i] for i in range(0,len(value),3)][::-1])
        replacements.append((m.start(),m.end(),grouped));counts['integer_grouping']+=1
    for a,b,v in reversed(replacements):s=s[:a]+v+s[b:]
    s,n=captions(s);counts['figure_title_placement']+=n
    return s,dict(counts)
def files():
    paths={THESIS/'settings.tex'}
    for line in (THESIS/'main.fls').read_text().splitlines():
        if line.startswith('INPUT ./') and line.endswith('.tex'):
            p=(THESIS/line[8:]).resolve()
            if p.exists() and p.is_relative_to(THESIS) and p.relative_to(THESIS).parts[0] in ['chapters','figures','beforetext','aftertext','notes']:paths.add(p)
    return sorted(paths)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fix',action='store_true');args=parser.parse_args()
    result={};total=collections.Counter()
    for p in files():
        before=p.read_text();after,counts=normalize(before)
        if before==after:continue
        result[str(p.relative_to(ROOT))]=counts;total.update(counts)
        if args.fix:
            backup=RELEASE/'originals'/p.relative_to(ROOT);backup.parent.mkdir(parents=True,exist_ok=True)
            if not backup.exists():backup.write_text(before)
            p.write_text(after)
    summary={'mode':'fix' if args.fix else 'check','files_scanned':len(files()),'files_needing_changes':len(result),'totals':dict(total),'files':result}
    if args.fix:
        RELEASE.mkdir(parents=True,exist_ok=True)
        (RELEASE/'formatting_changes.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
