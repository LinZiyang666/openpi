"""Cost-mode corrections for supplemental arms and a complete readable curve index."""
from pathlib import Path
import csv,json
O=Path(__file__).resolve().parent
R=json.loads((O/'arms.json').read_text())
conflicts=[]
for r in R:
    r['ir_full_reference']=(.152*r['v']+.848*r['m'])*5/r['L'] if r['cell'].startswith('pi05') else (.148*r['v']+.852*r['m'])*5/r['L']
    if r['cell'].startswith('pi05'):
        k=2 if 'k2' in r['arm'] else 10
        c1=.0552 if r['stage1_mode']=='wrist_only' else (.104 if r['stage1_mode']=='dummy_cached' else .152)
        extra=.0499 if r['stage1_mode']=='wrist_only' else 0.
        r['ir']=(c1*r['v']+(.410+.438*k/10+extra)*r['m'])*5/r['L']
        r['price_note']='wrist ratio-transfer assumption' if extra else ('stage-3 linear K2 repricing' if k==2 else 'owner full-policy basis')
    else:r['price_note']='owner full-policy basis'
    if not r['cost_valid']:
        s=json.loads(Path(r['summary']).read_text());m=s['mixed']['misses']/s['client_decisions']
        conflicts.append(dict(arm=r['arm'],sr=r['sr'],raw_N=r['N'],raw_M=r['M'],summary_N=s['client_decisions'],summary_M=s['mixed']['misses'],summary_m=m,summary_ir=.152+.848*m,conflicting_episodes=len(r['conflicts'])))
(O/'legacy_conflicts.json').write_text(json.dumps(conflicts,indent=1))
(O/'arms.json').write_text(json.dumps(R,indent=1))
fields=['run','arm','cell','lib','family','n','sr','N','V','M','L','v','m','m5','ir','ir_full_reference','miss_per_ep','cost_valid','stage1_mode','price_note','summary']
with (O/'arms.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(R)
lines=['# Complete arm index','',
       'Frozen source paths, configuration kwargs and source hashes are in `arms.json`; all realized episode counts are in `episodes.csv`. Costs flagged INVALID are not admitted to fitted curves or frontiers. K2 and wrist prices are supplemental and use the historical assumptions in R4/R5.','']
for cell in sorted({r['cell'] for r in R}):
    for lib in sorted({r['lib'] for r in R if r['cell']==cell}):
        lines += [f'## {cell}, library {lib}','','| Arm | Family | SR | m | m5 | IR | MISS/episode | Cost audit |','|---|---|---|---|---|---|---|---|']
        for r in sorted([r for r in R if r['cell']==cell and r['lib']==lib],key=lambda r:r['ir']):
            lines += [f"| {r['run']}/{r['arm']} | {r['family']} | {r['sr']:.3f} | {r['m']:.5f} | {r['m5']:.5f} | {r['ir']:.5f} | {r['miss_per_ep']:.3f} | {'OK' if r['cost_valid'] else 'INVALID'} |"]
        lines+=['']
(O/'CURVES.md').write_text('\n'.join(lines))
print('indexed',len(R),'arms; conflicting costs',len(conflicts))
