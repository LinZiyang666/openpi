"""Read-only R3-C diagnostics; writes only alongside this file. No policy/model imports."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

OUT = Path(__file__).resolve().parent
REQUESTED = Path('/dev/shm/offline_search_store')
STORE = REQUESTED if REQUESTED.exists() else Path('/home/weiland/trace_runs/offline_search_store')
RUN = Path('/home/weiland/trace_runs/os_closed_loop/r02_g50/runs')
RESULTS = Path('/home/weiland/projects/openpi/exp/offline_search/results/r02')

def arr(path, key):
    p = path / (key + '.npy')
    if not p.exists(): p = path / ('rows.' + key + '.npy')
    return np.load(p, mmap_mode='r')

def mean(x): return float(np.mean(x)) if len(x) else None
def qtiles(x): return [float(v) for v in np.quantile(x, [.1, .5, .9])]

def auc(score, y):
    score, y = np.asarray(score), np.asarray(y, bool)
    if not y.any() or y.all(): return None
    order = np.argsort(score, kind='stable')
    ranks = np.empty(len(y), float)
    _, starts, counts = np.unique(score[order], return_index=True, return_counts=True)
    ranks[order] = np.repeat(starts + (counts + 1)/2, counts)
    n = y.sum()
    return float((ranks[y].sum()-n*(n+1)/2)/(n*(len(y)-n)))

def weights(scores, arm, kref=5):
    s = np.array(scores, float)
    if arm == 0: return np.r_[1., np.zeros(len(s)-1)]
    if arm == 1: return np.r_[np.ones(min(5,len(s)))/min(5,len(s)), np.zeros(max(0,len(s)-5))]
    if arm == 2:
        w = np.exp(-((s-s[0])/max(s[0]-s[min(kref,len(s))-1],1e-6))**2)
    else: w = np.exp(s-s.max())
    return w/w.sum()

def runs(top):
    bounds = np.r_[0, np.flatnonzero(np.diff(top)!=0)+1, len(top)]
    return [(int(a), int(b)) for a,b in zip(bounds[:-1], bounds[1:])]

def load_arm(suite, arm):
    short = {'spatial':'sp','l10':'l10'}[suite]
    path = RUN / f'oscl50_p_{short}_cl{arm}'
    journal = {}
    for line in (path/'client/journal.jsonl').open():
        r = json.loads(line)
        if r.get('accepted') and r.get('status') in ('done','failed'):
            journal[r['task_uid']] = r
    records = defaultdict(dict)
    startups=[]
    for p in sorted(path.glob('server_*/decisions_*.jsonl')):
        for line in p.open():
            r=json.loads(line)
            if r['ev']=='startup': startups.append(r['kwargs'])
            if r['ev']!='dec' or r.get('uid') not in journal: continue
            if r.get('attempt',1)!=journal[r['uid']].get('attempt',1): continue
            records[r['uid']][r['step']]=r
    lib = STORE / 'library'/f'pi05_{suite}'/'current'
    acts=arr(lib,'action'); lep=arr(lib,'episode'); lend=arr(lib,'ep_len'); lst=arr(lib,'step')
    eps=[]
    for uid, bystep in sorted(records.items()):
        rr=[bystep[t] for t in sorted(bystep)]
        assert [r['step'] for r in rr]==list(range(len(rr)))
        features=defaultdict(list)
        prevg=None
        for i,r in enumerate(rr):
            rows=np.array(r['topk'],int); w=weights(r['scores'],arm)
            acts0=acts[rows,:5,:7]
            grip=w @ np.where(acts0[:,:,6]>=0,1.,-1.)
            ew=np.array([w[lep[rows]==ep].sum() for ep in np.unique(lep[rows])])
            vals={'vote':abs(grip[0]),'ep_eff':1/(ew@ew),'ep_max':ew.max(),
                  'row_eff':1/(w@w),'term':float(lst[rows[0]]==lend[rows[0]]-1),
                  'conf':r.get('conf',0.),'top':rows[0], 'gswitch':float(prevg is not None and (grip[0]>=0)!=prevg),
                  'same_run':1 if i==0 or rows[0]!=features['top'][-1] else features['same_run'][-1]+1,
                  'term_w':float(w @ (lst[rows]==lend[rows]-1)),
                  'signed_vote':grip[0]}
            vals.update({k:r.get('extras',{}).get(k,0.) for k in ['d1','d1_rel','disp5','dst','still','recover','stuck_n','rec_level','level']})
            for k,v in vals.items(): features[k].append(v)
            prevg=grip[4]>=0
        f={k:np.array(v) for k,v in features.items()}
        rs=runs(f['top']); starts3=[a for a,b in rs if b-a>=3]; starts6=[a for a,b in rs if b-a>=6]
        spell=np.zeros(len(rr),bool)
        for a,b in rs:
            if b-a>=3: spell[a:b]=True
        f['spell']=spell
        eps.append(dict(uid=uid, task=rr[0]['task_id'], init=rr[0]['init'], success=bool(journal[uid]['success']),
                        n=len(rr), first3=min(starts3,default=len(rr)), first6=min(starts6,default=len(rr)),
                        nspell=len(starts3), longest=max(b-a for a,b in rs), starts6=starts6, f=f))
    assert len(eps)==500, (path,len(eps))
    client_dec=0; client_hits=defaultdict(int)
    for line in (path/'client/per_step.jsonl').open():
        r=json.loads(line)
        if r.get('accepted') and r['task_uid'] in journal and r.get('_kind')!='client_timing':
            client_dec+=1; client_hits[r.get('hit_type')]+=1
    assert client_dec==sum(e['n'] for e in eps),(path,client_dec)
    return eps,dict(kwargs=startups[0],client_dec=client_dec,server_dec=sum(e['n'] for e in eps),client_hits=dict(client_hits))

def summarize(eps):
    out={'sr':mean([e['success'] for e in eps]), 'decisions':sum(e['n'] for e in eps)}
    out['per_task']={str(t):mean([e['success'] for e in eps if e['task']==t]) for t in range(10)}
    for succ in [True,False]:
        ee=[e for e in eps if e['success']==succ]
        f={k:np.concatenate([e['f'][k] for e in ee]) for k in eps[0]['f']}
        out[str(succ)]={'episodes':len(ee),'nmean':mean([e['n'] for e in ee]),
            'no_spell':mean([e['nspell']==0 for e in ee]),'longest':mean([e['longest'] for e in ee]),
            'flips_per_ep':mean([e['f']['gswitch'].sum() for e in ee]),
            'ep_eff':mean(f['ep_eff']), 'vote_split':mean(f['vote']<.8),
            'ep_single':mean(f['ep_eff']<1.5), 'conf_quantiles':qtiles(f['conf']),
            'spell_frac':mean(f['spell']), 'still_198':mean(f['still']>1.98)}
    return out

def gate_result(eps, flags):
    result={'flag_rate':sum(x.sum() for x in flags)/sum(e['n'] for e in eps)}
    for success in [False,True]:
        subset=[(e,x) for e,x in zip(eps,flags) if e['success']==success]
        hits=[int(np.flatnonzero(x)[0]) if x.any() else e['n']+1 for e,x in subset]
        result[str(success)]={'episodes':len(subset),'touched':mean([h<e['n'] for h,(e,x) in zip(hits,subset)]),
            'by_first3_plus1':mean([h<=e['first3']+1 and h<e['n'] for h,(e,x) in zip(hits,subset)]),
            'before_first3':mean([h<e['first3'] and h<e['n'] for h,(e,x) in zip(hits,subset)]),
            'by_first6_plus1':mean([h<=e['first6']+1 and h<e['n'] for h,(e,x) in zip(hits,subset)]),
            'strict_before_first6':mean([h<e['first6'] and h<e['n'] for h,(e,x) in zip(hits,subset)])}
    return result

def gates(eps):
    train=[e for e in eps if e['init']<25]; test=[e for e in eps if e['init']>=25]
    out={}
    for key,direction in [('conf',-1),('disp5',1),('dst',1),('d1_rel',1),('vote',-1),('ep_eff',-1),('still',1)]:
        tr=np.concatenate([direction*e['f'][key] for e in train])
        for rate in [.05,.1,.2]:
            cut=float(np.quantile(tr,1-rate))
            flags=[direction*e['f'][key]>cut for e in test]
            out[f'{key}_{rate}']=dict(cut=cut,**gate_result(test,flags))
    rules={
        'repeat2':lambda f:f['same_run']>=2,
        'repeat3':lambda f:f['same_run']>=3,
        'repeat2_still198':lambda f:(f['same_run']>=2)&(f['still']>1.98),
        'switch_split08':lambda f:(f['gswitch']>0)&(f['vote']<.8),
        'switch_split06':lambda f:(f['gswitch']>0)&(f['vote']<.6),
        'repeat2_or_switch08':lambda f:(f['same_run']>=2)|((f['gswitch']>0)&(f['vote']<.8)),
    }
    for key,fun in rules.items(): out[key]=gate_result(test,[fun(e['f']) for e in test])
    # How much apparent predictive separation disappears when we stop each prefix
    # at the FIRST repeated row? The all-episode signal contains consequence/length bias.
    for key,sgn in [('conf',-1),('ep_eff',-1),('vote',-1),('still',1),('gswitch',1)]:
        vals=[]; early=[]; ys=[]
        for e in test:
            f=e['f'][key]; vals.append(sgn*np.mean(f)); early.append(sgn*np.mean(f[:min(5,len(f))]));ys.append(not e['success'])
        out[f'auc_{key}']={'episode_mean':auc(vals,ys),'first5_mean':auc(early,ys)}
    return out

def library_inventory():
    out={}
    for model in ['pi05','groot']:
        for suite in ['spatial','l10']:
            for name in ['current', 'bpool_cs' if model=='pi05' else 'bpool_all']:
                p=STORE/'library'/f'{model}_{suite}'/name
                a=arr(p,'action'); ep=arr(p,'episode'); tid=arr(p,'task_id')
                out[f'{model}_{suite}_{name}']={'L':len(a),'episodes':len(np.unique(ep)),
                    'action_bytes_valid_full_chunk':a.shape[1]*7*4,
                    'awm_key_bytes':580,'compact_payload_MB':len(a)*(580+a.shape[1]*7*4)/1e6,
                    'plus_pca_MB':2*(32768+32768*64)*4/1e6,
                    'task_rows':[int((tid==t).sum()) for t in np.unique(tid)]}
    return out

def main():
    out={'store_requested':str(REQUESTED),'store_used':str(STORE),'library':library_inventory(),'arms':{},'gates':{},'paired':{}}
    for suite in ['spatial','l10']:
        arms=[]
        for arm in range(4):
            eps,meta=load_arm(suite,arm);arms.append(eps)
            out['arms'][f'{suite}_cl{arm}']={**meta,**summarize(eps)}
            # Compact observations remain under the permitted output directory.
            if arm==2:
                out['gates'][suite]=gates(eps)
                np.savez_compressed(OUT/f'cl2_{suite}.npz',
                    **{k:np.concatenate([e['f'][k] for e in eps]) for k in eps[0]['f']},
                    ep=np.concatenate([np.full(e['n'],i) for i,e in enumerate(eps)]),
                    task=np.concatenate([np.full(e['n'],e['task']) for e in eps]),
                    init=np.concatenate([np.full(e['n'],e['init']) for e in eps]),
                    success=np.concatenate([np.full(e['n'],e['success']) for e in eps]))
            print(suite,arm,'sr',out['arms'][f'{suite}_cl{arm}']['sr'],flush=True)
        for aa,bb in [(0,1),(1,2),(0,2),(2,3)]:
            a={(e['task'],e['init']):e['success'] for e in arms[aa]}; b={(e['task'],e['init']):e['success'] for e in arms[bb]}
            assert set(a)==set(b)
            diff=np.array([int(b[k])-int(a[k]) for k in a])
            delta=diff.mean();se=diff.std(ddof=1)/np.sqrt(len(diff))
            out['paired'][f'{suite}_{aa}_to_{bb}']={'gain':int((diff==1).sum()),'loss':int((diff==-1).sum()),
                'delta':float(delta),'paired_normal_95ci':[float(delta-1.96*se),float(delta+1.96*se)]}
    (OUT/'closed_loop_diagnostics.json').write_text(json.dumps(out,indent=2))
    print('Wrote closed_loop_diagnostics.json; store:',STORE,flush=True)

if __name__=='__main__': main()
