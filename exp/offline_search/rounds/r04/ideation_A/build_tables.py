"""Render exact diagnostic tables and paired episode-bootstrap intervals into reviewable markdown."""
import csv,glob,json,pathlib,sys
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[5]))
from exp.offline_search.rounds.r04.ideation_A.measure_blind import *

def read(pattern):return [d for p in sorted(OUT.glob(pattern)) for d in csv.DictReader(p.open())]
def fmt(x):return f'{float(x):.3f}'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'

metrics=read('metrics_*.csv');index={(d['cell'],int(d['scale']),d['method'],d['split'],int(d['h'])):d for d in metrics}
def vector(cell,s,m,split='all',metric='err_mean'):
    return ' / '.join(fmt(index[(cell,s,m,split,h)][metric]) if (cell,s,m,split,h) in index else '—' for h in range(1,5))

lines=['# R4-A generated measurements','All four-value entries list blind horizons 1 / 2 / 3 / 4. Error is current-library σ-normalized RMS on 5×7 executed action blocks. These are immutable-path diagnostics, not closed-loop SR.','']
rows=[]
for s in [50,500]:
    for key in CELLS:
        for arm in ['inf','cache']:
            c=f'{key}_{arm}';rows.append([c,s,*[vector(c,s,m) for m in ['awm','awm_allhit','top1_clock','kernel_clock','phase_abs','anchor_chunk']]])
lines+=['## Action degradation (matched future targets)',table(['query cell','library episodes','vision AWM','vision AWM all-HIT branch','top1 clock','kernel clock','phase abs','anchor chunk'],rows)]
for kind in ['err_mean','delta_awm','grip_mis']:
    rows=[]
    for s in [50,500]:
        for key in CELLS:
            for arm in ['inf','cache']:
                c=f'{key}_{arm}';rows.append([c,s,*[vector(c,s,'phase_abs',split,kind) for split in ['near_transition','far_transition','early','mid','late']]])
    lines+=['## Phase-aligned serving: '+kind,table(['query cell','library episodes','near transition','far transition','early','mid','late'],rows)]
for comparator in ['kernel_clock','anchor_chunk']:
    rows=[]
    for s in [50,500]:
        for key in CELLS:
            for arm in ['inf','cache']:
                c=f'{key}_{arm}';cells=[]
                for split in ['near_transition','far_transition','early','mid','late']:
                    cells.append(' / '.join(fmt(float(index[c,s,comparator,split,h]['err_mean'])-float(index[c,s,'phase_abs',split,h]['err_mean']))
                        if (c,s,comparator,split,h) in index else '—' for h in range(1,5)))
                rows.append([c,s,*cells])
    lines+=['## Phase-conditioned comparator: '+comparator+' minus phase_abs',
        'Positive values favor phase alignment on the action-error proxy. These are paired windows, not SR effects.',
        table(['query cell','library episodes','near transition','far transition','early','mid','late'],rows)]
rows=[]
for s in [50,500]:
    for key in CELLS:
        for arm in ['inf','cache']:
            c=f'{key}_{arm}';rows.append([c,s,*[vector(c,s,m) for m in ['phase_abs','phase_delta','phase_hybrid','phase_reweight_0.25','phase_reweight_1','repeat_anchor_head']]])
lines+=['## Serving ablations',table(['query cell','library episodes','absolute phase','delta phase','hybrid phase','state reweight T=.25','state reweight T=1','repeat anchor head'],rows)]
tr=read('triggers_*.csv');ti={(d['cell'],int(d['scale']),d['trigger'],d['split'],int(d['h'])):d for d in tr}
rows=[]
for s in [50,500]:
    for key in CELLS:
        for arm in ['inf','cache']:
            c=f'{key}_{arm}';rows.append([c,s,*[' / '.join(fmt(ti[c,s,t,'all',h]['rate']) for h in range(1,5)) for t in ['grip_ahead','near_terminal','still2','delta_dev05','absolute_dev05','union']]])
lines+=['## Independent trigger firing rates over all anchor windows',table(['query cell','library episodes','grip ahead','near terminal','still 2','delta residual >.5','absolute residual >.5','union'],rows)]
of=read('offline_schedules*.csv');rows=[]
for s in [50,500]:
    for key in CELLS:
        for arm in ['inf','cache']:
            c=f'{key}_{arm}';data=[d for d in of if d['cell']==c and int(d['scale'])==s and d['method']=='phase_abs' and d['triggers']=='phase_state']
            data=sorted(data,key=lambda d:int(d['cap']))
            rows.append([c,s,*[' / '.join(fmt(d[v]) for d in data) for v in ['vision_share','ir','delta_awm','blind_delta']]])
lines+=['## Scheduled budgets 1 / 2 / 3 / 4, phase + state gates',table(['query cell','library episodes','vision share','pure-cache reference IR','all-row Δerr','blind-row Δerr'],rows)]
ls=read('log_schedules*.csv');aggregate=[]
for arm in sorted(set(d['arm'] for d in ls)):
    for cap in [1,2,3,4]:
        for trigger in ['budget','phase','phase_vote']:
            a=[d for d in ls if d['arm']==arm and int(d['cap'])==cap and d['triggers']==trigger]
            n=sum(int(d['N']) for d in a);v=sum(int(d['nlook']) for d in a)/n;m=sum(int(d['nmiss']) for d in a)/n
            aggregate.append(dict(arm=arm,key=a[0]['key'],scale=int(a[0]['scale']),cap=cap,triggers=trigger,N=n,
                                  vision_share=v,miss_share=m,ir=.152*v+.848*m,old_ir=.152+.848*m,
                                  **{k:sum(float(d[k]) for d in a)/n for k in a[0] if k.startswith('why_')}))
writecsv(OUT/'log_schedule_aggregate.csv',aggregate)
rows=[]
for arm in sorted(set(d['arm'] for d in aggregate)):
    a=[d for d in aggregate if d['arm']==arm];rows.append([arm,a[0]['scale'],fmt(a[0]['miss_share']),fmt(a[0]['old_ir']),
                   *[' / '.join(fmt(d['ir']) for d in a if d['triggers']==t) for t in ['budget','phase','phase_vote']]])
lines+=['## Closed-loop frozen-path IR (MISSes kept at their observed decisions)',table(['arm','library episodes','MISS share','original IR','budget-only caps 1–4','phase caps 1–4','phase + vote caps 1–4'],rows)]
timing=read('log_timing_oscl*.csv');rows=[]
for arm in sorted(set(d['arm'] for d in timing)):
    a=[d for d in timing if d['arm']==arm];un=[d for d in a if d['trigger']=='union_phase']
    rows.append([arm,len(un),*[' / '.join(fmt(np.mean([int(d[k]) for d in a if d['trigger']==t])) for k in ['at_start','within_prior2','within_prior4']) for t in ['grip_ahead','near_terminal','union_phase']]])
lines+=['## Failed first-spell trigger coverage: at onset / onset or previous 2 / onset or previous 4',
        'Retrospective flags use the actual vision kernel at every recorded decision. The next table instead forecasts each flag from the last retained anchor.',
        table(['arm','failed episodes with spell','grip ahead','near terminal','union'],rows)]
rows=[]
for arm in sorted(set(d['arm'] for d in ls if d['arm'].startswith('oscl'))):
    a=[d for d in ls if d['arm']==arm and d['cap']=='2' and d['triggers']=='phase' and int(d['first_failed_spell'])>=0]
    rows.append([arm,len(a),*[' / '.join(fmt(np.mean([int(d[k]) for d in a])) for k in ['predicted_phase_at_spell','predicted_phase_prior2','predicted_phase_prior4'])]])
lines+=['## Causal anchor forecast: phase-gated cap 2, failed first-spell coverage',
        table(['arm','failed episodes with spell','at onset / onset or previous 2 / onset or previous 4'],rows)]
rows=[]
for d in json.loads((OUT/'log_summaries.json').read_text()):
    if not d['arm'].startswith('oscl'):continue
    f=d['N']/(d['N']-d['episodes'])
    rows.append([d['key'],d['scale'],d['N'],fmt(d['sr']),fmt(d['sameep']*f),fmt(d['successor']*f),fmt(d['repeat']*f),
        ' / '.join(fmt(d[p+'_grip']) for p in ['early','mid','late']),
        ' / '.join(fmt(d[p+'_terminal']) for p in ['early','mid','late'])])
lines+=['## Pure-cache path anatomy', 'Consecutive-pick rates exclude episode starts. Phase thirds are fractions of the recorded episode length and are evaluation labels only.',
        table(['cell','episodes','decisions','SR','same episode','exact successor','same row','grip trigger early / mid / late','terminal trigger early / mid / late'],rows)]

# Paired decision-weighted mean error, resampling whole episodes (1000 draws).
ci=[];rng=np.random.default_rng(4817)
for s in [50,500]:
    for key in CELLS:
        for arm in ['inf','cache']:
            c=f'{key}_{arm}';q=store.QueryCell(ROOT,c)
            draws=rng.integers(0,500,(1000,500))
            for h in range(1,5):
                w=np.load(OUT/f'windows_{c}_{s}_h{h}.npz');eps=q.ep[w['anchor']];counts=np.bincount(eps,minlength=500)
                pairs=[('phase_abs','kernel_clock'),('top1_clock','kernel_clock')]
                if f'err_anchor_chunk' in w:pairs.append(('anchor_chunk','phase_abs'))
                for a,b in pairs:
                    dif=w[f'err_{a}']-w[f'err_{b}'];sums=np.bincount(eps,weights=dif,minlength=500)
                    boot=sums[draws].sum(1)/counts[draws].sum(1);low,high=np.percentile(boot,[2.5,97.5])
                    ci.append(dict(cell=c,scale=s,h=h,comparison=a+' minus '+b,delta=float(dif.mean()),ci_low=low,ci_high=high))
writecsv(OUT/'paired_error_ci.csv',ci)
lines+=['## Paired action-error intervals','1000 whole-episode bootstrap draws, seed 4817. Full table: `paired_error_ci.csv`. These intervals quantify a proxy, not SR.']
(OUT/'MEASUREMENTS.md').write_text('\n\n'.join(lines))
sizes=[]
for key in CELLS:
    for scale in [50,500]:
        M=get_method(key,scale);L=library(key,scale)
        sizes.append(dict(key=key,scale=scale,L=L.L,episodes=len(np.unique(L.episode)),representation_b=601,
            action_b=L.H*7*4,fixed_bytes=M.fixed_bytes,compact_bytes=M.fixed_bytes+L.L*(601+L.H*7*4),
            fit_bytes=artifact(key,scale).stat().st_size,added_topology_bytes=L.L*21))
(OUT/'bytes.json').write_text(json.dumps(sizes,indent=2))
print('wrote MEASUREMENTS.md, log_schedule_aggregate.csv, paired_error_ci.csv')
