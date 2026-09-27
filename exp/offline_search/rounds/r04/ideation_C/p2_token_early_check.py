"""Decision-5 token diagnostic restricted to episodes whose first B0 spell starts at >=7."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
OUT=Path(__file__).resolve().parent;out=[]
for m in ['pi05','groot']:
 d=json.loads((OUT/f'p2_tokens_{m}_l10.json').read_text())
 for n in ['50','500']:
  ep=d['scales'][n]['per_episode'];rr=[e for e in ep if e['first_spell']>=7]
  y=np.array([not e['success'] for e in rr]);v=np.array([e['features']['cam1_patch_p95']['step5'] for e in rr])
  auc=(rankdata(v)[y].sum()-y.sum()*(y.sum()+1)/2)/(y.sum()*(~y).sum())
  out.append({'model':m,'scale':int(n),'episodes':len(rr),'failures':int(y.sum()),'auc':float(auc),
              'failed_episodes_spell_at_or_before_5':sum(not e['success'] and e['first_spell']<=5 for e in ep)})
(OUT/'p2_token_early_check.json').write_text(json.dumps(out,indent=2))
