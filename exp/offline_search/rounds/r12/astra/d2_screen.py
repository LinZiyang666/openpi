"""Post-freeze B-dev mechanism screen, not changed-controller SR evidence."""
from .d2_consensus import *
from exp.offline_search.rounds.r11.knob.recipe import uniform
import math

def due_before(A,i,seq,setting):
    if setting is None:return False
    s=int(A['step'][i]);anchors=[j for j in range(s) if seq[j].get('vision',True)]
    calls=[j for j in anchors if not seq[j]['hit']];last=calls[-1] if calls else -1
    run=sum(j>last for j in anchors);u=uniform('R11-gap-v1',0,int(A['task'][i]),int(A['init'][i]),last,'cap')
    cap=math.floor(setting)+int(setting%1>0 and u<setting%1)
    return run>=cap

def gate_mask(A,Q,C,lib,cfg):
    valid=Q['reference']>=0;h=Q['rows'][:,0];w=Q['witness_rows'];j=np.maximum(Q['reference'],0)
    prev=A['rows'][j,0];l=np.asarray(lib.ep_len)
    adv=(C['progress'][h]-C['progress'][prev])*np.maximum(l[h]-1,1)
    pp=np.abs(C['progress'][h,None]-C['progress'][w])*np.maximum(np.maximum(l[h,None],l[w])-1,1)
    k=cfg['witnesses']
    ok=valid&(Q['support']>=.99)&(Q['safe_mass']>=cfg.get('safe_mass',.75))&(Q['state_error']<=cfg.get('state',.2))
    if k:
        ok&=(Q['witness_rms'][:,:k].max(1)<=cfg['rms'])&~Q['witness_grip'][:,:k].any(1)
        ok&=(pp[:,:k].max(1)<=cfg['progress'])&(adv>cfg.get('advance',.5))
        if cfg.get('exact_top',False):ok&=(w[:,:k]==h[:,None]).all(1)
    return ok

def replay_gate(A,Q,C,lib,decisions,cfg,setting=None):
    n=len(A['step']);ok=gate_mask(A,Q,C,lib,cfg);half=np.zeros(n,bool);rows=A['rows'].copy();corr=A['full'].copy()
    spans=np.zeros(n,int);due=np.zeros(n,bool);calls=np.zeros(n,bool);firsts=[];last_uid=None
    prev=-1;span=0;lastfull=-1;lastcall=-1;run=0;first=None;life=np.zeros(n,bool)
    for i in range(n):
        uid=str(A['uid'][i]);s=int(A['step'][i]);seq=decisions[uid];d=seq[s]
        if uid!=last_uid:
            if last_uid is not None:firsts.append(dict(uid=last_uid,first=first))
            last_uid=uid;prev=-1;span=0;lastfull=-1;lastcall=-1;run=0;first=None
        if setting is not None:
            u=uniform('R11-gap-v1',0,int(A['task'][i]),int(A['init'][i]),lastcall,'cap')
            cap=math.floor(setting)+int(setting%1>0 and u<setting%1);due[i]=run>=cap
        valid_tail=(s>=2 and not seq[s-2]['hit'] and seq[s-2].get('vision',True)
            and seq[s-1].get('source',seq[s-1].get('src'))=='policy_tail'
            and seq[s-1].get('extras',{}).get('policy_tail_lifecycle')==1
            and seq[s-1].get('previous_executed_steps')==5 and d.get('previous_executed_steps')==5
            and seq[s-1].get('exec_ok',False) and d.get('exec_ok',False))
        life[i]=(A['look_reason'][i] not in (6,7,8)) or (A['look_reason'][i]==6 and valid_tail)
        half[i]=bool(ok[i] and Q['reference'][i]==lastfull and lastfull>=0 and span==0 and life[i] and not due[i])
        if half[i]:rows[i]=Q['rows'][i];corr[i]=Q['corrected'][i]
        h=int(rows[i,0]);p=C['progress'][h]
        span=prog_span(span,int(A['step'][prev]) if prev>=0 else -1,C['progress'][int(rows[prev,0])] if prev>=0 else 0,s,p,lib.ep_len[h])
        spans[i]=span;calls[i]=span>=A['threshold'][i] or due[i]
        if calls[i]:lastcall=s;run=0
        else:run+=1
        if calls[i]!=~A['hit'][i] and first is None:first=dict(step=s,kind='extra_call' if calls[i] else 'missed_call')
        if s+1<len(seq):
            expected=bool(span>0 and not calls[i]);actual=seq[s+1].get('vision',True)
            if expected!=actual and first is None:first=dict(step=s+1,kind='extra_look' if expected else 'missed_look')
        if not half[i]:lastfull=i
        elif calls[i]:lastfull=-1
        prev=i
    firsts.append(dict(uid=last_uid,first=first))
    return dict(half=half,rows=rows,corrected=corr,spans=spans,calls=calls,due=due,firsts=firsts,lifecycle=life,gate_ok=ok)

def short(A,Q,R,C,meta,decisions,episode_mask=None):
    if episode_mask is None:episode_mask=np.ones(len(A['step']),bool)
    mask=episode_mask;uids=np.unique(A['uid'][mask]);N=sum(len(decisions[str(u)]) for u in uids)
    half=R['half']&mask;basecall=~A['hit'];miss=basecall&~R['calls']&mask;extra=~basecall&R['calls']&mask
    price=.076 if meta['cell'].startswith('pi05') else .074;policy=1-2*price
    rms=np.sqrt(np.mean(((Q['corrected'][:,:,:6]-A['full'][:,:,:6])/C['sigma'][:6])**2,(1,2)))
    grip=np.any((Q['corrected'][:,:,6]>=0)!=(A['full'][:,:,6]>=0),1)
    has=np.isfinite(A['shadow']).all((1,2))&mask
    out=dict(N=N,V=int(mask.sum()),episodes=len(uids),H=int(half.sum()),half_share=float(half.sum()/mask.sum()),
        saving_fixed_calls=price*int((half&~basecall).sum())/N,
        saving_changed_calls_fixed_schedule=(price*int((half&~R['calls']).sum())-policy*(int(extra.sum())-int(miss.sum())))/N,
        missed=int(miss.sum()),extra=int(extra.sum()),missed_on_half=int((miss&half).sum()),extra_on_half=int((extra&half).sum()),
        timing_divergent_episodes=sum(x['first'] is not None and x['uid'] in uids for x in R['firsts']),
        motion_rms_half=quant(rms[half]),gripper_disagree_half=int((grip&half).sum()),
        half_base_calls=int((half&basecall).sum()),half_candidate_calls=int((half&R['calls']).sum()))
    if has.any():
        err=lambda x:np.mean(((x[:,:,:6]-A['shadow'][:,:,:6])/C['sigma'][:6])**2,(1,2))
        delta=err(R['corrected'])-err(A['full'])
        out['shadow_mse_delta']=mean(delta[has],A['uid'][has]);out['full_shadow_mse']=mean(err(A['full'])[has],A['uid'][has])
    return out

def configs():
    out=[dict(name='frozen_atlas',witnesses=0)]
    for k in (1,2,6,18):
        for rms in (.05,.1,.15):
            out.append(dict(name=f'consensus{k}_r{rms}',witnesses=k,rms=rms,progress=.5,advance=.5))
    out.extend([dict(name='exact_consensus',witnesses=18,rms=.1,progress=0.,advance=.5,exact_top=True),
                dict(name='strict_atlas',witnesses=0,safe_mass=.95,state=.1)])
    for mass in (0.,.5):
        for k in (2,6):
            for rms in (.05,.1,.15):
                out.append(dict(name=f'current{k}_m{mass}_r{rms}',witnesses=k,rms=rms,progress=.5,advance=.5,safe_mass=mass))
    for k in (2,6):
        for rms in (.075,.1,.15):
            out.append(dict(name=f'exact{k}_r{rms}',witnesses=k,rms=rms,progress=0.,advance=.5,exact_top=True))
    return out

def run(root,cell,size):
    src=DEST/root/f'{cell}_{size}';dest=D2/root/f'{cell}_{size}'
    A=dict(np.load(src/'looks.npz',allow_pickle=False));Q=dict(np.load(dest/'candidates.npz',allow_pickle=False))
    meta=json.loads((src/'extract.json').read_text());decisions=json.loads((src/'decisions.json').read_text())
    C,lib=load(cell,size);C['sigma']=np.load(HERE/'final_atlas'/f'{cell}_{size}'/'table.npz')['sigma']
    setting=meta.get('periodic_setting');out=[]
    # Deterministic first five init IDs per task vs remaining five. Split named
    # post-freeze dev calibration/evaluation; earlier atlas saw all these rows.
    cal=np.zeros(len(A['step']),bool)
    for t in range(10):
        ids=np.unique(A['init'][A['task']==t]);cal|=(A['task']==t)&np.isin(A['init'],ids[:len(ids)//2])
    for cfg in configs():
        R=replay_gate(A,Q,C,lib,decisions,cfg,setting)
        out.append(dict(config=cfg,all=short(A,Q,R,C,meta,decisions),calibration=short(A,Q,R,C,meta,decisions,cal),
            evaluation=short(A,Q,R,C,meta,decisions,~cal)))
        np.savez_compressed(dest/f"{cfg['name']}.npz",**{k:v for k,v in R.items() if isinstance(v,np.ndarray)})
    dump(dest/'screen.json',dict(timestamp=now(),root=root,cell=cell,size=size,results=out,split='post-freeze B-dev split; no independent SR estimate'))
    print(root,cell,size,[(x['config']['name'],x['all']['H'],x['all']['missed'],x['all']['gripper_disagree_half']) for x in out],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root');args=ap.parse_args()
    for p in sorted(D2.glob('r12_devlooks_*/*/candidates.json')):
        a=json.loads(p.read_text())
        if a['root'] in VALID_ROOTS and (args.root is None or a['root']==args.root):run(a['root'],a['cell'],a['size'])
