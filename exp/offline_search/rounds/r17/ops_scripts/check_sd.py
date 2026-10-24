# check_sd.py NAME...: every YAML of a seeds_1023 batch points at its own seed/size/segment fit, stall and selection; stall_only.
import json,yaml,collections,sys
from pathlib import Path
for name in sys.argv[1:]:
    root=Path('/data/mw_finetune/eval')/name
    if not root.exists(): print(name,'missing'); continue
    c=json.load(open(root/'campaign.json')); s=json.load(open(root/'assets/CACHE_STUDY.json'))
    seen=collections.Counter(); bad=[]
    for g in s['groups']:
        y=yaml.safe_load(open(root/'assets'/g['file'])); r=y['recipe']
        fit=r['fitted_path']; stall=r.get('stall_path')
        tag=next((t for t in ('x10','x9','x3','x4') if f'_{t}c' in g['group']),'?')
        commit=y.get('adapter_binding',{}).get('commit_horizon'); size=g.get('library_size')
        sel=Path(fit).parent/'selection'/(r['fitted_digest']+'.json')
        ok=(f'sd{tag}_' in fit) and (f'sd{tag}_' in str(stall)) and (f'n{size}_C{commit}' in fit)
        if not ok or not sel.exists() or r.get('miss_mode')!='stall_only': bad.append((g['group'],size,commit,fit[-70:],sel.exists()))
        seen[(tag,commit)]+=1
    print(name,'server',c.get('server_node'),c.get('execution_host'),'R',c.get('replicas'),'groups',len(s['groups']),dict(seen),'bad',len(bad),'episodes',json.load(open(root/'FFK_DEFINITION.json')).get('episodes'))
    for b in bad[:3]: print('  ',b)
