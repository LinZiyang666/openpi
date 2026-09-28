"""R6 Q3, read-only sources, output confined to this directory and /tmp/r6_q3.
All array dimensions here describe the recorded experiment, not a portable controller.
"""
from pathlib import Path
import argparse, collections, json, pickle, hashlib
import numpy as np
from scipy.stats import rankdata, spearmanr

OUT=Path(__file__).parent
TMP=Path('/tmp/r6_q3')
CL=Path('/home/weiland/trace_runs/os_closed_loop')
STORE=Path('/home/weiland/trace_runs/offline_search_store')
SEED=20260928
PAIRS={
 'pi05_l10_50': [('r05_ptail','r5t_p_l10_50_tail1uc'),('r05_q1','r5q1_c10_p_l10_50')],
 'pi05_l10_500':[('r05_ptail','r5t_p_l10_500_tail1uc'),('r05_q1','r5q1_c10_p_l10_500')],
 'pi05_spatial_50':[('r05_ptail','r5t_p_sp_50_tail1uc'),('r05_q1','r5q1_c10_p_sp_50')],
 'pi05_spatial_500':[('r04_blind','r4b3_p_sp_500_tail1uc'),('r05_q1','r5q1_c10_p_sp_500')],
 **{f'groot_{s}_{n}':[('r05_x',f'r5x_g_{"sp" if s=="spatial" else s}_{n}_tail1u'),('r05_q2',f'r5q2_g_{s}_{n}_G10')] for s in ('l10','spatial') for n in (50,500)}
}

def dump(name,x): (OUT/name).write_text(json.dumps(x,indent=2,allow_nan=True,default=lambda a:a.item() if isinstance(a,np.generic) else a.tolist()))
def arr(p,n):return np.load(p/(n+'.npy'),mmap_mode='r')
def percentile(x,v):return np.searchsorted(np.sort(x),v,side='right')/len(x)
def ci_mean(x,boot=4000):
 x=np.asarray(x,float); rng=np.random.default_rng(SEED)
 if not len(x):return None
 # Values can be clustered contributions, one per initialization or demo episode.
 draws=np.array([x[rng.integers(len(x),size=len(x))].mean() for _ in range(boot)])
 return dict(n=len(x),mean=float(x.mean()),ci95=np.quantile(draws,[.025,.975]).tolist())
def auc(y,x):
 y=np.asarray(y,bool);x=np.asarray(x,float)
 if not y.any() or y.all():return float('nan')
 return float((rankdata(x)[y].sum()-y.sum()*(y.sum()+1)/2)/(y.sum()*(~y).sum()))
def group_delta(d,g):
 d=np.asarray(d,float);g=np.asarray(g,bool);rng=np.random.default_rng(SEED)
 if not g.any() or g.all():return dict(n_high=int(g.sum()),n=len(g))
 vals=[]
 for _ in range(4000):
  ii=rng.integers(len(d),size=len(d));dd=d[ii];gg=g[ii]
  if gg.any() and not gg.all():vals.append(dd[gg].mean()-dd[~gg].mean())
 return dict(n_high=int(g.sum()),n=len(g),high=ci_mean(d[g]),low=ci_mean(d[~g]),interaction=float(d[g].mean()-d[~g].mean()),interaction_ci95=np.quantile(vals,[.025,.975]).tolist())

def libload(cell):
 model,suite,scale=cell.split('_');name='current' if scale=='50' else ('bpool_cs' if model=='pi05' else 'bpool_all')
 p=STORE/'library'/f'{model}_{suite}'/name
 a={k:arr(p,k) for k in ['action','rs','task_id','episode','step','next','prev','progress','success','ep_len']}
 a['sigma']=np.maximum(np.std(a['action'][:,:5,:7].reshape(-1,7),axis=0),1e-8)
 a['p']=p
 return a

def calibrate(cell):
 a=libload(cell);run,arm=PAIRS[cell][0];fit=CL/run/'fits'/f'{arm}.pkl'
 m=pickle.load(fit.open('rb'))['method']
 while not hasattr(m,'tasks'):m=m.base
 L=len(a['episode']);head=np.array(a['action'][:,:5,:7],float)/a['sigma'];full=np.array(a['action'][:,:10,:7],float)/a['sigma']
 result={k:np.full(L,np.nan) for k in ['distance','state_distance','disagreement','error','error10','plan_error10','phase_error','phase','event','event_soon','mode_switch_soon','motion','jump','stall']}
 tables={}
 for t,T in m.tasks.items():
  r=T.rows;e=a['episode'][r];z=np.asarray(T.Z,float);rs=np.asarray(a['rs'][r,:8],float);sd=np.maximum(rs.std(0),1e-8);n=len(r)
  nxt=a['next'][r];ok=nxt>=0
  motion=np.full(n,np.nan);motion[ok]=np.sqrt(np.mean(((a['rs'][nxt[ok],:8]-rs[ok])/sd)**2,axis=1))
  jump=np.full(n,np.nan);jump[ok]=np.sqrt(np.mean((head[nxt[ok]]-head[r[ok]])**2,axis=(1,2)))
  event=(jump>np.nanquantile(jump,.9));switch=np.zeros(n,bool)
  switch[ok]=(np.mean(a['action'][r[ok],:5,6],axis=1)>=0)!=(np.mean(a['action'][nxt[ok],:5,6],axis=1)>=0)
  soon=event.copy();gs=switch.copy();lookup={int(v):j for j,v in enumerate(r)}
  for j,k in enumerate(nxt):
   if k>=0:soon[j]|=event[lookup[int(k)]];gs[j]|=switch[lookup[int(k)]]
  result['event'][r]=event;result['event_soon'][r]=soon;result['mode_switch_soon'][r]=gs;result['motion'][r]=motion;result['jump'][r]=jump
  h=head[r].reshape(n,-1);h10=full[r].reshape(n,-1)
  for lo in range(0,n,192):
   hi=min(n,lo+192);zz=z[lo:hi];ss=rs[lo:hi]
   D=np.sqrt(np.maximum(np.sum(zz*zz,1)[:,None]+np.sum(z*z,1)[None,:]-2*zz@z.T,0));D[e[lo:hi,None]==e[None,:]]=np.inf
   ds=np.sqrt(np.maximum(np.sum(ss*ss,1)[:,None]+np.sum(rs*rs,1)[None,:]-2*ss@rs.T,0));ds[e[lo:hi,None]==e[None,:]]=np.inf
   k=min(m.k,np.isfinite(D).sum(1).min());ii=np.argsort(D,axis=1,kind='stable')[:,:k];dd=np.take_along_axis(D,ii,axis=1)
   w=np.exp(-((dd-dd[:,:1])/np.maximum(dd[:,min(k,m.kref)-1,None]-dd[:,:1],1e-6))**2);w/=w.sum(1)[:,None]
   pred=np.einsum('ij,ijk->ik',w,h[ii]);pred10=np.einsum('ij,ijk->ik',w,h10[ii])
   # Mean pairwise RMS of top-five, identical definition to G3 core.dispersion.
   hp=h[ii[:,:5]];dis=np.mean([np.sqrt(np.mean((hp[:,i]-hp[:,j])**2,axis=1)) for i in range(5) for j in range(i+1,5)],axis=0)
   rr=r[lo:hi];result['distance'][rr]=dd[:,0];result['state_distance'][rr]=ds.min(1);result['disagreement'][rr]=dis
   result['error'][rr]=np.sqrt(np.mean((pred-h[lo:hi])**2,axis=1));result['plan_error10'][rr]=np.sqrt(np.mean((pred10-h10[lo:hi])**2,axis=1))
   nx=np.asarray(a['next'][rr]);valid=(nx>=0)&(a['episode'][np.maximum(nx,0)]==a['episode'][rr])&(a['step'][np.maximum(nx,0)]==a['step'][rr]+1)
   # Only the first five controls of each original library decision were executed.
   # The held-out trajectory target is head(t) concatenated with head(t+1), never its unused tail.
   target=np.concatenate([head[rr[valid]],head[nx[valid]]],axis=1).reshape(sum(valid),-1)
   result['error10'][rr[valid]]=np.sqrt(np.mean((pred10[valid]-target)**2,axis=1))
   result['phase'][rr]=a['progress'][r[ii[:,0]]];result['phase_error'][rr]=abs(result['phase'][rr]-a['progress'][rr])
  # Stalls in inferred matched phase over two source transitions, online-legal once past them.
  for eid in np.unique(e):
   er=r[e==eid];pg=result['phase'][er];st=np.zeros(len(er))
   for j in range(1,len(er)):st[j]=(st[j-1]+1) if pg[j]<=pg[j-1] else 0
   result['stall'][er]=st
  tables[str(t)]={k:np.nanquantile(result[k][r],[.1,.5,.9,.95,.99]).tolist() for k in ['distance','state_distance','disagreement','error','error10','motion','jump']}
  tables[str(t)].update(state_sd=sd.tolist(),median_length=float(np.median([sum(e==eid) for eid in np.unique(e)])),rows=n,episodes=len(np.unique(e)))
 # Include row metadata so any reported threshold or error statistic is reproducible.
 np.savez_compressed(TMP/f'library_{cell}.npz',**result,task=a['task_id'],episode=a['episode'],success=a['success'])
 epstats=[]
 for ep in np.unique(a['episode']):
  r=np.flatnonzero(a['episode']==ep);t=str(int(a['task_id'][r[0]]));q=tables[t]
  epstats.append(dict(episode=int(ep),task=int(t),success=bool(a['success'][r[0]]),error=float(result['error'][r].mean()),error10=float(np.nanmean(result['error10'][r])),**{f'{k}_auc':auc(result['error'][r]>q['error'][2],result[k][r]) for k in ['distance','state_distance','disagreement','event_soon','mode_switch_soon','stall']}))
 out=dict(cell=cell,fit=str(fit),library=str(a['p']),rows=L,episodes=len(epstats),sigma=a['sigma'],tables=tables,episode_stats=epstats)
 dump(f'calibration_{cell}.json',out)
 print('calibrated',cell,L,flush=True)
 return out

def read_arm(run,arm):
 assert (CL/run/'state'/f'{arm}.DONE').exists(), (run,arm,'completion marker absent')
 cache=TMP/f'{arm}.pkl'
 if cache.exists():return pickle.load(cache.open('rb'))
 p=CL/run/'runs'/arm;accepted={}
 for line in (p/'client/journal.jsonl').open():
  j=json.loads(line)
  if j.get('accepted') and j.get('status') in ('done','failed') and not j.get('error'):accepted[j['task_uid']]=j
 ds=collections.defaultdict(dict);startup=None;sources=[];duplicates=0
 for path in sorted(p.glob('server_*/decisions_*.jsonl')):
  sources.append(dict(path=str(path),bytes=path.stat().st_size))
  for line in path.open():
   d=json.loads(line)
   if d.get('ev')=='startup':startup=d;continue
   uid=d.get('uid');j=accepted.get(uid)
   if d.get('ev')!='dec' or j is None or int(d.get('attempt',1) or 1)!=int(j.get('attempt',1) or 1):continue
   assert d.get('ok',True)
   if d['step'] in ds[uid]:
    duplicates+=1;old=ds[uid][d['step']]
    assert all(old.get(k)==d.get(k) for k in ['served_head','hit','src','extras','top1'])
   ds[uid][d['step']]=d
 episodes={}
 for uid,j in accepted.items():
  seq=[v for k,v in sorted(ds[uid].items())];assert [d['step'] for d in seq]==list(range(len(seq)))
  task,init=map(int,uid.split(':')[-2:]);episodes[(task,init)]=dict(success=int(j['success']),ds=seq)
 summary=json.loads((p/'summary.json').read_text());assert len(episodes)==summary['complete'];assert sum(e['success'] for e in episodes.values())==summary['success']
 out=dict(run=run,arm=arm,episodes=episodes,startup=startup,summary=summary,sources=sources,duplicates=duplicates)
 pickle.dump(out,cache.open('wb'),protocol=5)
 print('read',arm,len(episodes),sum(len(e['ds']) for e in episodes.values()),flush=True)
 return out

def enrich(d,a,cal,lastmiss=None):
 t=str(d['task_id']);tab=cal['tables'][t];ex=d.get('extras') or {};r=int(d['top1']);phase=float(a['progress'][r])
 rows=np.array(d.get('rows') or d.get('topk') or [r],int);w=np.array(d.get('weights') or [],float)
 if len(w)!=len(rows):
  scores=np.array(d.get('scores') or [0.],float);w=np.exp(scores[:len(rows)]-max(scores))
 w=w/w.sum();heads=np.array(a['action'][rows,:5,:7],float)/a['sigma'];pred=np.einsum('i,ijk->jk',w,heads)
 if 'disp' in ex:
  # Logged wrapper sigma comes from current; adjust by recomputing in own-library units.
  pass
 hp=heads[:min(5,len(heads))];dis=np.mean([np.sqrt(np.mean((hp[i]-hp[j])**2)) for i in range(len(hp)) for j in range(i+1,len(hp))]) if len(hp)>1 else 0.
 sdnn=ex.get('dnn')
 if sdnn is None:
  rr=np.flatnonzero(a['task_id']==int(t));sdnn=float(np.linalg.norm(a['rs'][rr,:8]-np.array(d['robot_state'][:8]),axis=1).min())
 distance=ex.get('d1',np.nan)
 # Known recording schema: action index 6 is gripper; label is mode transition, not verified contact.
 nex=int(a['next'][r]);sw=False;ev=False
 for cur in [r,nex]:
  if cur<0:continue
  nx=int(a['next'][cur])
  if nx<0:continue
  h=np.asarray(a['action'][cur,:5,:7])/a['sigma'];nh=np.asarray(a['action'][nx,:5,:7])/a['sigma']
  ev|=np.sqrt(np.mean((h-nh)**2))>tab['jump'][2]
  sw|=(h[:,6].mean()>=0)!=(nh[:,6].mean()>=0)
 out=dict(phase=phase,state_coverage=sdnn/tab['state_distance'][2],coverage=distance/tab['distance'][2],disagreement=dis/tab['disagreement'][2],event_soon=int(ev),mode_switch_soon=int(sw),reason=int(ex.get('os_reason',0)),flags=int(ex.get('os_flags',0)),stuck=ex.get('stuck_n',0),stall=ex.get('noprog_n',0),overtime=d['step']/tab['median_length'],since_miss=(d['step']-lastmiss)/tab['median_length'] if lastmiss is not None else np.nan,first_miss=lastmiss is None,policy_cache_rms=float(np.sqrt(np.mean((np.array(d['served_head'])/a['sigma']-pred)**2))))
 return out

def pair_analysis(cell):
 A,B=[read_arm(*p) for p in PAIRS[cell]];a=libload(cell);cal=json.loads((OUT/f'calibration_{cell}.json').read_text());keys=sorted(A['episodes']);assert keys==sorted(B['episodes'])
 delta=[];records=[];guards=collections.defaultdict(list);examples=[];risks=collections.defaultdict(list)
 for key in keys:
  ea=A['episodes'][key];eb=B['episodes'][key];da=ea['ds'];db=eb['ds'];t=str(key[0]);tab=cal['tables'][t];dY=eb['success']-ea['success'];delta.append(dY)
  # Fixed library-clock landmark, never actual terminal duration.
  anc=next((d for d in da if d.get('vision',True) and d['step']>=.25*tab['median_length']),None)
  f=enrich(anc,a,cal) if anc else None
  if f is not None:
   hist=[d for d in da[:anc['step']+1] if d.get('vision',True)]
   ph=np.array([a['progress'][d['top1']] for d in hist]);stall=0
   for j in range(len(ph)-1,0,-1):
    if ph[j]>ph[j-1]:break
    stall+=hist[j]['step']-hist[j-1]['step']
   f['phase_late']=int(f['phase']>=.5);f['stall_present']=int(stall>=2)
   if len(hist)>1:
    f['motion_low']=int(np.sqrt(np.mean(((np.array(anc['robot_state'][:8])-np.array(hist[-2]['robot_state'][:8]))/tab['state_sd'])**2))<tab['motion'][0])
   else:f['motion_low']=0
  misses=[d for d in db if not d.get('hit',True)];first=misses[0]['step'] if misses else None
  # Compare physical state separation at aligned control slots, using library p95 one-step motion.
  sep=[];actsep=[]
  for x,y in zip(da,db):
   dist=np.sqrt(np.mean(((np.array(x['robot_state'][:8])-np.array(y['robot_state'][:8]))/tab['state_sd'])**2));sep.append(dist)
   actsep.append(np.sqrt(np.mean(((np.array(x['served_head'])-np.array(y['served_head']))/a['sigma'])**2)))
  divergence=next((i for i,v in enumerate(sep) if v>tab['motion'][3]),None)
  action_divergence=next((i for i,v in enumerate(actsep) if v>tab['error'][2]),None)
  ff=enrich(misses[0],a,cal) if misses else None
  last=None;byreason=collections.defaultdict(list);recs=[]
  for d in misses:
   ef=enrich(d,a,cal,last);s=d['step'];last=s
   nxt=next((x for x in db[s+1:] if x.get('vision',True)),None)
   ef['step']=s;ef['success']=eb['success'];ef['paired_delta']=dY
   ef['next_phase_gain']=(float(a['progress'][nxt['top1']])-ef['phase']) if nxt else None
   ef['next_is_policy']=not nxt.get('hit',True) if nxt else None
   ef['terminal_call']=s==len(db)-1;recs.append(ef);byreason[ef['reason']].append(ef)
  for reason,rr in byreason.items():guards[str(reason)].append(dict(task=key[0],init=key[1],success=eb['success'],paired_delta=dY,n=len(rr),first_step=rr[0]['step'],mean_change=float(np.mean([r['policy_cache_rms'] for r in rr])),calls=rr))
  rec=dict(task=key[0],init=key[1],A_success=ea['success'],B_success=eb['success'],delta=dY,A_N=len(da),B_N=len(db),M=len(misses),first_miss=first,first_miss_features=ff,state_divergence=divergence,action_divergence=action_divergence,landmark=f,calls=recs)
  records.append(rec)
  if dY==1 and first is not None:
   examples.append(dict(task=key[0],init=key[1],first_miss=first,state_divergence=divergence,action_divergence=action_divergence,first=ff,post_first=[dict(step=d['step'],source=d.get('src'),top1=d['top1'],phase=float(a['progress'][d['top1']])) for d in db[first:min(first+7,len(db))]],M=len(misses)))
 out=dict(cell=cell,A=f"{A['run']}/{A['arm']}",B=f"{B['run']}/{B['arm']}",contrast=ci_mean(delta),A_SR=np.mean([r['A_success'] for r in records]),B_SR=np.mean([r['B_success'] for r in records]),B_only=sum(d==1 for d in delta),A_only=sum(d==-1 for d in delta),records=records,guards=dict(guards),examples=examples,sources=A['sources']+B['sources'])
 eligible=[r for r in records if r['landmark'] is not None];dd=[r['delta'] for r in eligible]
 for k in ['coverage','state_coverage','disagreement','event_soon','mode_switch_soon','phase_late','stall_present','motion_low']:
  g=[r['landmark'][k]>(1 if k in ('coverage','state_coverage','disagreement') else .5) for r in eligible]
  risks[k]=group_delta(dd,g)
 out['landmark_interactions']=dict(risks);out['landmark_ineligible']=len(records)-len(eligible)
 for label,p in [('A',A),('B',B)]:
  ledger=p['summary']['cost_ledger'];v=ledger['v'];m=ledger['m'];c=.152 if cell.startswith('pi05') else .148
  out[label+'_IR']=c*v+(1-c)*m;out[label+'_M_per_episode']=sum(not d.get('hit',True) for e in p['episodes'].values() for d in e['ds'])/len(keys)
 rng=np.random.default_rng(SEED);ratios=[];mm=np.array([r['M'] for r in records]);dy=np.array(delta)
 for _ in range(4000):
  ix=rng.integers(len(keys),size=len(keys));ratios.append(dy[ix].sum()/max(mm[ix].sum(),1))
 out['configuration_gain_per_added_call']=dict(value=float(dy.sum()/mm.sum()),ci95=np.quantile(ratios,[.025,.975]).tolist(),warning='configuration-level accounting ratio, not the causal effect of a specific call')
 dump(f'paired_{cell}.json',out);print('paired',cell,out['contrast'],flush=True)
 return out

def causal(scale):
 from exp.offline_search.rounds.r04.k5_rand.estimate import load_arm,validate_pairs,cluster_draws,effect
 rows=[];aud=[];cell=f'pi05_l10_{scale}';a=libload(cell);cal=json.loads((OUT/f'calibration_{cell}.json').read_text())
 for rep in (1,2):
  arm=f'r4k5_p_l10_g{scale}_r{rep}';rr,ad=load_arm(CL/'r04_k5',arm);rows.extend(rr);aud.append(ad)
  dat=read_arm('r04_k5',arm)
  bykey={(r['task_id'],r['init']):r for r in rr}
  for key,e in dat['episodes'].items():
   last=None
   for d in e['ds']:
    if d.get('eligible'):
     f=enrich(d,a,cal,last);f['stall_age']=(d.get('context_values') or {}).get('stall_age');bykey[key]['features']=f
    if not d.get('hit',True):last=d['step']
 validate_pairs(rows);cid,weights=cluster_draws(rows,4000,SEED);expo=np.array([r['exposed'] for r in rows]);allmask=np.ones(len(rows),bool)
 out=dict(scale=scale,audit=aud,ITT=effect(rows,allmask,cid,weights)[0],subgroups={},features=rows)
 masks={'landmark1':np.array([r['landmark_class']==1 for r in rows]),'landmark3':np.array([r['landmark_class']==3 for r in rows])}
 for name,mask in masks.items():out['subgroups'][name]=effect(rows,mask,cid,weights)[0]
 for feature in ['state_coverage','disagreement','phase','event_soon','mode_switch_soon','stuck','stall','overtime','since_miss']:
  val=np.array([r.get('features',{}).get(feature,np.nan) for r in rows],float)
  threshold=.5 if feature in ['phase','event_soon','mode_switch_soon'] else (2 if feature in ['stuck','stall'] else (.1 if feature=='since_miss' else 1))
  usable=expo&np.isfinite(val);hi=usable&(val>=threshold);lo=usable&(val<threshold)
  eh,dh=effect(rows,hi,cid,weights);el,dl=effect(rows,lo,cid,weights)
  entry=dict(threshold=threshold,high=eh,low=el)
  if dh is not None and dl is not None:
   from scipy.stats import norm
   interaction=eh['delta']['Y']-el['delta']['Y'];draws=dh[:,0]-dl[:,0];z=norm.ppf(1-.05/(2*18));se=float(np.nanstd(draws,ddof=1))
   entry.update(interaction=interaction,interaction_ci95=np.nanquantile(draws,[.025,.975]).tolist(),interaction_family18_normal_ci=[interaction-z*se,interaction+z*se])
  out['subgroups'][feature]=entry
 for reason in (1,2,3,4):out['subgroups']['reason'+str(reason)]=effect(rows,expo&np.array([r.get('features',{}).get('reason')==reason for r in rows]),cid,weights)[0]
 # All K5 eligible landmarks already trigger a guard: genuinely preguard prevention has no overlap.
 out['pre_guard_exposures']=sum(r['exposed'] and not r['features']['flags'] for r in rows)
 dump(f'causal_{scale}.json',out);print('causal',scale,out['ITT']['delta'],flush=True)
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['calibrate','paired','causal']);p.add_argument('--cell',choices=list(PAIRS));args=p.parse_args()
 cells=[args.cell] if args.cell else list(PAIRS)
 if args.stage=='calibrate':
  for c in cells:calibrate(c)
 elif args.stage=='paired':
  for c in cells:pair_analysis(c)
 else:
  for n in (50,500):causal(n)

if __name__=='__main__':main()
