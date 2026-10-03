"""Compact console inspection of generated B-only evidence."""
import json
from exp.offline_search.rounds.r11.astra.boundary import HERE, install
from exp.offline_search.rounds.r11.astra.experiment import CELLS

install()
for m,s,n in CELLS:
    tag=f'{m}_{s}_{n}'
    a=json.loads((HERE/'data'/tag/'analysis.json').read_text())
    pair=next(p for p in a['predicted_vs_disagreement'] if p['stride']==2 and p['dose']==.4)
    print(tag, 'guard',round(a['base']['guard_per_look'],3),'predictor minus disagreement',pair)
    for d in a['adaptive']:
        print(' ADAPT',d['target'],'init',round(d['dose'],3),'validation',round(d['validation']['owner_ir_mean'],4),
              'feasible',d['feasible'])
        for sc in d['scenarios']:
            print('  SC',sc['shift'],sc['stress'],'static',round(sc['static']['owner_ir_mean'],4),
                  'adaptive',round(sc['adaptive']['owner_ir_mean'],4),'sat',round(sc['adaptive']['saturation'],3))
