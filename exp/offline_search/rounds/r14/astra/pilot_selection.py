"""Use only already consumed B identities, never consult the R13 reservation."""
from common import *
fence()
from exp.offline_search.closed_loop.devset import r10_subset
import numpy as np

selections=[]
for cell in CELLS:
    model,suite,n=cell.split('_');n=int(n)
    if n==500:continue
    root='r11_devknob_50' if n==50 else 'r12_devlooks_200'
    # The old middle-size recording is pi05; original B identities are valid
    # across models only after checking both actual library exclusions.
    source_model=model if n==50 else 'pi05'
    candidates=[];sources=[]
    for a in read(RUNS/root/'arms.json'):
        if a['model']!=source_model or a['suite_short']!=suite:continue
        jp=RUNS/root/'runs'/a['arm']/'client/journal.jsonl'
        if not jp.exists():continue
        sources.append(dict(path=str(jp),sha256=sha(jp)))
        for line in jp.open():
            r=json.loads(line)
            if 'success' in r and not r.get('error') and r.get('accepted',True):
                assert r['init_pool']=='B';candidates.append(tuple(map(int,r['task_uid'].split(':')[-2:])))
    exclusions=r10_subset(model,suite,n)['excluded_inits_by_task'];pairs=[];missing=[]
    for t in range(10):
        available=sorted({i for tt,i in candidates if tt==t}-set(exclusions[str(t)]),
                         key=lambda i:hashlib.sha256(f'R14-pilot:{suite}:{t}:{i}'.encode()).hexdigest())
        if len(available)<5:missing.append(dict(task=t,available=len(available),needed=5))
        pairs.extend([[t,i] for i in available[:5]])
    selections.append(dict(cell=cell,pairs=sorted(pairs),missing=missing,sources=sources,
        pool='B',selection='already consumed B-dev only; not fresh independent validation',
        excluded_inits_by_task=exclusions))
dump(HERE/'pilot_selection.json',dict(timestamp=now(),cells=selections))
provenance('pilot_selection')
print([(x['cell'],len(x['pairs']),x['missing']) for x in selections])
