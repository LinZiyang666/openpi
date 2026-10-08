from common import *
fence()
import pickle
import numpy as np

rows=[]
for root in ROOTS:
    p=RUNS/root/'arms.json'
    if not p.exists():continue
    for a in read(p):
        assert a['dev'] and a['init_pool']=='B'
        arm=RUNS/root/'runs'/a['arm']
        p=arm/'summary.json'
        if not p.exists():continue
        s=read(p); manifests=list(arm.glob('server_*/looks/manifest_*.json'))
        rows.append(dict(root=root,arm=a['arm'],model=a['model'],suite=a['suite_short'],
             size=a.get('r10_size'),method=a.get('r11_method','off'),complete=s.get('complete'),
             seconds=s.get('elapsed_s'),manifests=len(manifests),summary_keys=list(s)))
csvout(HERE/'inventory.csv',rows)
details=[]
for cell in CELLS:
    p=R10/'recipe/artifacts'/f'r10_recipe_{cell}.pkl'
    m=pickle.load(p.open('rb'))['method']; b=m.inner.base; j=m.inner
    subset=read(R10/'subsets'/f'{cell}.json')
    z=np.load(R10/'pca'/'_'.join(cell.split('_')[:2])/cell.split('_')[-1]/'v0.npz')
    details.append(dict(cell=cell,rows=len(m.row_subset),episodes=m.size,
        kref=b.kref,distance_scale=b.distance_scale,disabled_guards=j.disabled_guards,
        judge_attrs=list(vars(j)),base_attrs=list(vars(b)),task_attrs=list(vars(b.tasks[0])),
        pca_keys=z.files,pca_rows=len(z['proj']),fit_sha=sha(p),fit_info=m.fit_info))
dump(HERE/'deployment_audit.json',details)
provenance('inventory')
print(table(['cell','library rows','kref','GC scale'],[[x['cell'],x['rows'],x['kref'],round(x['distance_scale'],4)] for x in details]))
print(table(['root','completed arms','episodes'],[[r,sum(x['root']==r for x in rows),sum(x['complete'] for x in rows if x['root']==r)] for r in ROOTS]))
