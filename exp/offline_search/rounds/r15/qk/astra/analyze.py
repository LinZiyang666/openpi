"""Aggregate prototype evidence, paired episode uncertainty, and phase diagnostics."""
import csv
import json
from pathlib import Path
import numpy as np
from exp.offline_search.rounds.r15.qk.astra.prepare import HERE,DERIVED,dump

def run():
    out={};csvrows=[]
    for suite,size,root in [('l10',50,'r12_devlooks_50L'),('l10',200,'r12_devlooks_50L'),('spatial',50,'r12_devlooks_50S_local')]:
        cell=f'pi05_{suite}_{size}';p=HERE/cell;tr=np.load(DERIVED/cell/'train.npz');dp=DERIVED/f'dev_pi05_{suite}_{root}';keep=np.load(dp/f'unseen_{size}.npy');dev={k:v[keep] for k,v in dict(np.load(dp/'dev.npz')).items()}
        ref=np.load(p/'eval_reference.npz');baseline=ref['err10'];records={}
        for fp in [p/'results.json',*sorted(p.glob('cnn_*_results.json')),p/'kernel_results.json',p/'action_key_results.json']:
            if fp.exists():
                for m in json.loads(fp.read_text())['methods']:records[m['name']]=m
        rng=np.random.default_rng(20261005)
        groups=[]
        for t in range(10):groups.append([np.flatnonzero((dev['uid']==u)&dev['label_valid']) for u in sorted(set(dev['uid'][dev['task_id']==t])) if ((dev['uid']==u)&dev['label_valid']).any()])
        reference_actual=None
        if size==50:
            parent=tr['rows'][ref['top']];act=dev['topk']
            reference_actual={'top1_agreement':float(np.mean(parent[:,0]==act[:,0])),'top16_overlap':float(np.mean([len(set(a)&set(b))/16 for a,b in zip(parent,act)])),'top1_disagreements':int(np.sum(parent[:,0]!=act[:,0]))}
        details={}
        for name,m in records.items():
            f=p/('eval_reference.npz' if name=='deployed_key_geometry' else f'eval_{name}.npz');a=np.load(f)
            # Equal episodes within task; equal task weight. Paired resampling
            # preserves method/reference dependence and repeated-look clustering.
            vals=[np.array([[a['err10'][g].mean(),baseline[g].mean()] for g in gs]) for gs in groups if gs]
            means=np.array([v.mean(0) for v in vals]);mean=means.mean(0)
            draws=[]
            for _ in range(2000):
                d=np.mean([v[rng.integers(len(v),size=len(v))].mean(0) for v in vals],0);draws.append(d)
            draws=np.array(draws);rat=draws[:,0]/draws[:,1]-1
            top=a['top'];prog=tr['progress'][top[:,0]];rprog=tr['progress'][ref['top'][:,0]]
            same=dev['uid'][1:]==dev['uid'][:-1]
            detail={'episode_task_balanced_err10':float(mean[0]),'reference_balanced_err10':float(mean[1]),'balanced_relative_excess':float(mean[0]/mean[1]-1),'balanced_relative_excess_ci95':np.quantile(rat,[.025,.975]).tolist(),'balanced_delta_ci95':np.quantile(draws[:,0]-draws[:,1],[.025,.975]).tolist(),'phase_any_backwards':float(np.mean((prog[1:]<prog[:-1])[same])),'phase_large_backwards':float(np.mean((prog[1:]-prog[:-1]<-.1)[same])),'phase_wrong_third_vs_teacher':float(np.mean(np.abs(prog-rprog)>1/3)),'phase_teacher_mae':float(np.abs(prog-rprog).mean())}
            detail['balanced_by_task']={str(t):{'episodes':len(gs),'err10':float(np.mean([a['err10'][g].mean() for g in gs])),'reference_err10':float(np.mean([baseline[g].mean() for g in gs]))} for t,gs in enumerate(groups) if gs}
            if size==50:
                parent=tr['rows'][top];detail['actual_recorded_top1_agreement']=float(np.mean(parent[:,0]==dev['topk'][:,0]));detail['actual_recorded_top16_overlap']=float(np.mean([len(set(a)&set(b))/16 for a,b in zip(parent,dev['topk'])]))
            details[name]={**m,**detail}
            csvrows.append({'cell':cell,'method':name,'err5':m['err5'],'err10':m['err10'],'err10_p90':m['err10_p90'],'top1':m['top1_agreement'],'top16':m['top16_overlap'],'balanced_err10':detail['episode_task_balanced_err10'],'relative_excess':detail['balanced_relative_excess'],'ci_low':detail['balanced_relative_excess_ci95'][0],'ci_high':detail['balanced_relative_excess_ci95'][1],'phase_backwards':detail['phase_large_backwards'],'phase_teacher_mae':detail['phase_teacher_mae']})
        out[cell]={'reference_vs_actual_recording':reference_actual,'episodes':len(set(dev['uid'])),'looks':len(dev['uid']),'labelled_looks':int(dev['label_valid'].sum()),'methods':details}
    dump(HERE/'analysis.json',out)
    with (HERE/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    for cell,d in out.items():
        ranked=sorted([(m['episode_task_balanced_err10'],name,m['balanced_relative_excess_ci95']) for name,m in d['methods'].items() if name!='deployed_key_geometry'])
        print(cell,'best',ranked[:4],'reference check',d['reference_vs_actual_recording'])

if __name__=='__main__':run()
