"""Read-only snapshot of accepted episodes; causal claims deliberately not made.
Run with the required taskset/BLAS environment. All products live beside this script.
"""
import json, pathlib, collections, datetime
import numpy as np
from scipy.stats import rankdata, spearmanr
from covariance import oas
from scipy.special import expit
OUT=pathlib.Path(__file__).parent
ROOT=pathlib.Path('/home/weiland/trace_runs/os_closed_loop')
STORE=pathlib.Path('/home/weiland/trace_runs/offline_search_store')
FEATS=['dnn','disp','vis','stuck_n','overtime','lag','top1_prog','noprog_n']
RUNS=['r02_g50','r02_g500','r03_full','r03_mx','r04_frontier','r04_blind','r04_k7']
def lines(p):
 if not p.exists():return
 # Snapshot file length; do not follow appenders.
 with p.open('rb') as f:
  size=p.stat().st_size
  while f.tell()<size:
   s=f.readline()
   try: yield json.loads(s)
   except (ValueError,UnicodeError): pass

def auc(y,s):
 y=np.asarray(y,bool);s=np.asarray(s)
 if y.all() or not y.any():return None
 return float((rankdata(s)[y].sum()-y.sum()*(y.sum()+1)/2)/(y.sum()*(~y).sum()))
def fitlda(X,y):
 med=np.nanmedian(X,axis=0);med=np.where(np.isfinite(med),med,0.)
 X=np.where(np.isfinite(X),X,med);mu=X.mean(0);sd=np.maximum(X.std(0),1e-6);Z=(X-mu)/sd
 m0=Z[y==0].mean(0);m1=Z[y==1].mean(0)
 cov, shrink=oas(Z-np.where(y[:,None]==1,m1,m0),assume_centered=True)
 w=np.linalg.solve(cov+1e-6*np.eye(Z.shape[1]),m1-m0)
 b=float(-.5*(m0+m1)@w+np.log(y.mean()/(1-y.mean())))
 return dict(median=med.tolist(),mean=mu.tolist(),std=sd.tolist(),w=w.tolist(),b=b,shrinkage=float(shrink))
def predict(f,X):
 X=np.where(np.isfinite(X),X,f['median']);return expit(((X-f['mean'])/f['std'])@f['w']+f['b'])

def main():
 summaries=[];sample=[];missing=[];all_ep={};replays={};snapshot=[]
 for rn in RUNS:
  root=ROOT/rn
  for meta in json.loads((root/'arms.json').read_text()):
   arm=meta['arm']; p=root/'runs'/arm
   if rn.startswith('r02') and not arm.endswith(('cl2','cl3')):continue
   j={r['task_uid']:r for r in lines(p/'client/journal.jsonl') if r.get('accepted') and r.get('status') in ('done','failed') and not r.get('error')}
   if not j:missing.append(arm);continue
   rec={}; files=sorted(p.glob('server_*/decisions_*.jsonl'))
   for fp in files:
    snapshot.append({'path':str(fp),'bytes':fp.stat().st_size})
    for r in lines(fp):
     if r.get('ev')!='dec' or r.get('uid') not in j:continue
     jr=j[r['uid']]
     if jr.get('attempt') is not None and r.get('attempt') is not None and jr['attempt']!=r['attempt']:continue
     rec[(r['uid'],int(r['step']))]=r
   groups=collections.defaultdict(list)
   for (uid,_),r in rec.items():groups[uid].append(r)
   N=V=M=0;eps={};nr=collections.Counter();flags=collections.Counter();noncontig=[];fx=[];q=[];frozen={k:0 for k in (3,4,5)}
   reason_outcomes=collections.defaultdict(lambda:[0,0])
   for uid,rs in groups.items():
    rs.sort(key=lambda r:r['step']);steps=[r['step'] for r in rs]
    if steps!=list(range(len(rs))):noncontig.append(uid);continue
    jr=j[uid];pair=tuple(map(int,uid.split(':')[-2:]));succ=int(bool(jr.get('success')))
    n=len(rs);m=sum(r.get('hit') is False for r in rs);v=sum(r.get('vision',True) for r in rs)
    N+=n;M+=m;V+=v;eps[pair]=succ
    for r in rs:
     e=r.get('extras') or {};fl=int(e.get('os_flags',0));npn=e.get('noprog_n',e.get('noprog_span',0));sn=e.get('stuck_n',0)
     nr[str(r.get('judge','cache'))]+=1
     for bit,name in [(1,'stuck'),(2,'terminal'),(4,'overtime'),(8,'noprog'),(16,'disp'),(32,'grip')]:
      if fl&bit:flags[name]+=1
     for k in frozen:frozen[k]+=int(bool(fl&~8) or npn>=k-1)
     if 'os_flags' in e:
      reason_outcomes[str(fl)][0]+=1;reason_outcomes[str(fl)][1]+=1-succ
    # One fixed landmark per episode prevents length-weighted failure labels.
    # Vision can be unavailable at exactly 10; use first anchor step>=10, at most 13.
    land=next((r for r in rs if 10<=r['step']<=13 and r.get('vision',True)),None)
    if land is not None and 'dnn' in (land.get('extras') or {}):
     e=land['extras'];vals=[float(e.get(k,e.get('noprog_span',0) if k=='noprog_n' else np.nan)) for k in FEATS]
     item={'arm':arm,'run':rn,'task':pair[0],'init':pair[1],'failure':1-succ,'step':land['step'],'x':vals,'model':meta['model'],'suite':meta['suite']}
     sample.append(item)
   all_ep[arm]=eps
   summary={'run':rn,'arm':arm,'model':meta['model'],'suite':meta['suite'],'kwargs':meta.get('kwargs',{}),'judge':meta.get('judge'),'journal_accepted':len(j),'episodes':len(eps),'N':N,'V':V,'M':M,'SR':sum(eps.values())/len(eps) if eps else None,'IR':(.152*V+.848*M)/N if N else None,'noncontiguous':noncontig,'reason':dict(nr),'flags':dict(flags),'landmark_n':sum(s['arm']==arm for s in sample),'flag_failure_counts':dict(reason_outcomes)}
   summaries.append(summary)
   if arm in ('r3mx_p_l10_g','r3mx_p_l10_g500','r3mx_p_sp_g','r4_p_sp_g500'):
    replays[arm]={'N':N,'actual_M':M,'fixed_path_M':frozen,'fixed_path_IR':{k:.152+.848*m/N for k,m in frozen.items()}}
   print(arm,len(eps),N,summary['SR'],summary['IR'],flush=True)
 pairs=[]
 comparisons=[('r3mx_p_l10_g','r4_p_l10_g50_np4'),('r3mx_p_l10_g500','r4_p_l10_g500_np4'),('r3mx_p_l10_perk5','r4_p_l10_per6_50'),('r4_p_l10_per8_500','r4_p_l10_per12_500'),('r4k7_p_l10_500_ph1g','r4k7_p_l10_500_ph2g'),('r4b3_p_l10_500_b0g','r4k7_p_l10_500_b0g'),('r4b3_p_l10_500_ph2g','r4k7_p_l10_500_ph2g'),('r4k7_p_l10_500_ph2g','r4k7_p_l10_500_tail1ug'),('r4k7_p_l10_50_ph2g','r4k7_p_l10_50_tail1ug')]
 for a,b in comparisons:
  ka=all_ep.get(a,{});kb=all_ep.get(b,{});common=sorted(set(ka)&set(kb));d=np.array([kb[k]-ka[k] for k in common]);
  pairs.append({'a':a,'b':b,'n':len(d),'delta_SR':float(d.mean()),'SE':float(d.std(ddof=1)/np.sqrt(len(d))),'S_to_F':int((d<0).sum()),'F_to_S':int((d>0).sum())})
 lda={}
 for scale,trainarm in [('50','r3mx_p_l10_g'),('500','r3mx_p_l10_g500')]:
  ss=[s for s in sample if s['arm']==trainarm];X=np.array([s['x'] for s in ss]);y=np.array([s['failure'] for s in ss]);t=np.array([s['task'] for s in ss]);i=np.array([s['init'] for s in ss]);
  preds=np.zeros(len(ss));fits={}
  for task in np.unique(t):
   tr=t!=task;f=fitlda(X[tr],y[tr]);preds[~tr]=predict(f,X[~tr]);fits[int(task)]=f
  # Initiative-held-out validation, stratified same tasks, still observational.
  ip=np.zeros(len(ss))
  for fold in range(5):
   tr=i%5!=fold;f=fitlda(X[tr],y[tr]);ip[~tr]=predict(f,X[~tr])
  full=fitlda(X,y);extern=[]
  for a in sorted(set(s['arm'] for s in sample if s['run'].startswith('r04') and s['model']=='pi05' and s['suite']=='libero_10' and ('_50_' in s['arm'] or 'g50_' in s['arm'] if scale=='50' else '_500_' in s['arm'] or 'g500_' in s['arm'] or '_500'==s['arm'][-4:]))):
   zz=[s for s in sample if s['arm']==a];xx=np.array([s['x'] for s in zz]);yy=np.array([s['failure'] for s in zz]);pp=np.array([predict(fits[s['task']],xx[n:n+1])[0] for n,s in enumerate(zz)])
   extern.append({'arm':a,'n':len(zz),'failure':float(yy.mean()),'predicted_failure':float(pp.mean()),'auc':auc(yy,pp)})
  lda[scale]={'features':FEATS,'train':trainarm,'label':'eventual failure conditional on surviving landmark 10; observational, not CALL benefit','n':len(y),'failures':int(y.sum()),'LOTO_AUC':auc(y,preds),'init_fold_AUC':auc(y,ip),'LOTO_Brier':float(np.mean((preds-y)**2)),'LOTO_constant_Brier':float(np.mean([(y[t==tt]-y[t!=tt].mean())@ (y[t==tt]-y[t!=tt].mean()) for tt in []])) if False else float(np.mean((y-np.array([y[t!=tt].mean() for tt in t]))**2)),'fit':full,'external':extern}
 out={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'inputs':snapshot,'missing':missing,'summaries':summaries,'paired':pairs,'fixed_path':replays,'risk_lda':lda}
 (OUT/'log_results.json').write_text(json.dumps(out,indent=2,allow_nan=False))
 # Small scalar feature artifact; NaNs use null in JSON.
 for s in sample:s['x']=[v if np.isfinite(v) else None for v in s['x']]
 (OUT/'landmarks.json').write_text(json.dumps(sample))
if __name__=='__main__':main()
