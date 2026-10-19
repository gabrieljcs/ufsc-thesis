"""Draw the sourced model-lineage figure as editable vector SVG and PDF."""
from pathlib import Path
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.graphics import renderPDF, renderSVG
from reportlab.lib.colors import HexColor, Color

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'ufscthesisx/ufscthesisx/figures'
W,H = 920,1160
D = Drawing(W,H)
INK = HexColor('#15283B'); MUTED = HexColor('#536575')
COLORS = {'PT':'#17725C','IT':'#2862A2','SV':'#7753A1','BG':'#B66B21','EN':'#52677A'}


def text(x,y,label,size=18,color=INK,bold=False):
    D.add(String(x,H-y,label,fontName='Helvetica-Bold' if bold else 'Helvetica',fontSize=size,fillColor=color))


def card(x,y,lines,width=245,height=65,color='#52677A',target=False,badge=None):
    c=HexColor(color)
    fill=c if target else Color(.965,.977,.984)
    D.add(Rect(x,H-y-height,width,height,rx=10,ry=10,fillColor=fill,strokeColor=c,strokeWidth=1.5))
    for i,label in enumerate(lines):text(x+14,y+25+i*22,label,18,HexColor('#FFFFFF') if target else INK,bold=i==0)
    if badge:
        text(x+10,y+height+23,badge,15,c,True)


def arrow(x1,y1,x2,y2,label='',dashed=False):
    line=Line(x1,H-y1,x2-9,H-y2,strokeColor=MUTED,strokeWidth=2)
    if dashed:line.strokeDashArray=[5,4]
    D.add(line);D.add(Polygon([x2,H-y2,x2-10,H-y2+5,x2-10,H-y2-5],fillColor=MUTED,strokeColor=None))
    if label:text((x1+x2)/2,y1-13,label,13,MUTED)


text(20,37,'Model families and specialization paths',29,bold=True)
text(20,66,'Documented ancestry of eight paired targets and one English reference',18,MUTED)
for x,label in [(20,'UPSTREAM FAMILY / CHECKPOINT'),(334,'ADAPTATION OR INTERMEDIATE'),(648,'EVALUATED ASSISTANT')]:
    text(x,109,label,13,MUTED,True)

# Each horizontal lane is an ancestry path, not a chronological or causal ranking.
lanes=[
 (140,['LLaMA 7B','original base model'],['Sabiá-7B','Portuguese continuation'],['Sagui-7B','Instruct-v0.1'],'PT','PORTUGUESE · weak',False),
 (245,['LLaMA-2 7B','Chat checkpoint'],['Italian UltraChat','QLoRA / merged weights'],['LLaMAntino-2','chat-UltraChat-ITA'],'IT','ITALIAN · weak',False),
 (350,['LLaMA-3 8B','Base checkpoint'],['Meta Llama-3 8B','Instruct checkpoint'],['LLaMAntino-3 ANITA','Italian SFT / DPO'],'IT','ITALIAN · strong',False),
 (455,None,['AI Sweden Llama-3','Nordic continued pretraining'],['AI Sweden Llama-3','8B-instruct'],'SV','SWEDISH · strong comparator',False),
 (560,['GPT-SW3 6.7B v2','Nordic base model'],['Instruction tuning','Nordic / English data'],['GPT-SW3 6.7B','v2-instruct'],'SV','SWEDISH · weak',False),
 (665,['Mistral 7B','v0.1 base checkpoint'],['Bulgarian adaptation','Continued pretraining / SFT'],['BgGPT-7B','Instruct-v0.2'],'BG','BULGARIAN · weak',False),
 (770,['Gemma 2 9B','Base + IT checkpoints'],['Branch-and-merge','Bulgarian adaptation / SFT'],['BgGPT-Gemma 2','9B-IT-v1.0'],'BG','BULGARIAN · strong',False),
 (875,['Llama 3.1 8B','Separate model generation'],['Meta instruction','and safety post-training'],['Llama 3.1 8B','Instruct reference'],'EN','ENGLISH · separate reference',False),
 (980,['Undisclosed upstream','No public ancestry asserted'],['Private model training','Details not established'],['Sabiá-3','Closed API'],'PT','PORTUGUESE · strong',True),
]
for y,base,inter,target,lang,badge,unknown in lanes:
    color=COLORS[lang]
    if base:card(20,y,base,color=color)
    card(334,y,inter,color=color)
    card(648,y,target,color=color,target=True,badge=badge)
    if base:arrow(269,y+32,327,y+32,dashed=unknown)
    arrow(583,y+32,641,y+32,dashed=unknown)
# Explicit fork from Llama-3 BASE, not from its Instruct descendant.
D.add(Line(142,H-415,142,H-487,strokeColor=HexColor(COLORS['SV']),strokeWidth=2.5))
arrow(142,487,327,487)
# The fork is explained in the figure caption; keep the connector unobstructed.
text(20,1100,'Solid arrows: documented derivation or training stage. Dashed arrows: undisclosed API lineage.',16,MUTED)
text(20,1127,'Weak/strong are design labels; arrows do not establish safety gains or losses.',16,MUTED)
OUT.mkdir(exist_ok=True)
renderPDF.drawToFile(D,str(OUT/'model-lineage.pdf'))
renderSVG.drawToFile(D,str(OUT/'model-lineage.svg'))
print('Wrote model-lineage.pdf and model-lineage.svg')
