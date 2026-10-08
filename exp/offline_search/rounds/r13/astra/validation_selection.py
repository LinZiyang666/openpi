"""Freeze a feasible fresh B identity list without creating a run root."""
from common import *
fence()

def main():
    used=read(HERE/'used_b_identities.json'); exclusions={};counts=[];selected={}
    for size,root in [(50,'r11_dev_size50'),(200,'r11_dev_size200')]:
        for a in read(RUNS/root/'arms.json'):
            cell=f'{a["model"]}_{a["suite_short"]}_{size}'
            if cell not in CELLS:continue
            p=RUNS/root/'runs'/a['arm']/'pool_contract.json';c=read(p)
            assert c['dev'] and c['init_pool']=='B'
            exclusions[cell]={int(t):set(v) for t,v in c['subset']['excluded_inits_by_task'].items()}
    for suite in ('l10','spatial'):
        cells=[c for c in exclusions if c.split('_')[1]==suite]
        seen={tuple(v) for m in COSTS for v in used.get(f'{m}_{suite}',[])}
        pairs=[]
        for task in range(10):
            served=set().union(*(exclusions[c][task] for c in cells))
            available=[i for i in range(50) if i not in served and (task,i) not in seen]
            available.sort(key=lambda i:hashlib.sha256(f'R13-fresh-validation:{suite}:{task}:{i}'.encode()).digest())
            counts.append(dict(suite=suite,task=task,available=len(available),requested=10))
            assert len(available)>=10,(suite,task,'not enough common untouched B identities')
            pairs.extend((task,i) for i in available[:10])
        assert not set(pairs)&seen
        for cell in cells:
            assert all(i not in exclusions[cell][t] for t,i in pairs)
            selected[cell]=pairs
    payload=dict(timestamp=now(),pool='B',dev=True,purpose='frozen independent validation; all arms/replicates share cell pairs',
        selection_rule='common across models and tested sizes within suite, outside all served subsets and old B identity union; SHA256 ordering',
        pairs_by_cell=selected,availability=counts,used_identities_sha256=sha(HERE/'used_b_identities.json'))
    target=HERE/'validation_selection.json'
    if target.exists():
        old=read(target);payload['timestamp']=old['timestamp'];assert clean(payload)==old
    dump(target,payload);provenance('validation_selection')
    print(table(['suite','minimum available per task','selected per cell'],[
        [s,min(r['available'] for r in counts if r['suite']==s),100] for s in ('l10','spatial')]))

if __name__=='__main__':main()
