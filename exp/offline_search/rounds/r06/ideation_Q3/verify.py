"""Audit the actual sources/units used by this analysis; no serving imports or launches."""
from analyze import *

def main():
 records=[]
 for cell,pairs in PAIRS.items():
  a=libload(cell)
  for run,arm in pairs:
   p=read_arm(run,arm);episodes=list(p['episodes'].values());ds=[d for e in episodes for d in e['ds']]
   ledger=p['summary']['cost_ledger'];counts=dict(N=len(ds),V=sum(d.get('vision',True) for d in ds),M=sum(not d.get('hit',True) for d in ds))
   assert (counts['N'],counts['V'],counts['M'])==(ledger['decisions'],ledger['vision_decisions'],ledger['misses'])
   candidates=[d for d in ds if d.get('vision',True) and d.get('src')=='cache' and d.get('rows') and d.get('weights')]
   errs=[]
   for j in np.linspace(0,len(candidates)-1,min(64,len(candidates)),dtype=int):
    d=candidates[j];w=np.asarray(d['weights'],float);w/=w.sum();pred=np.einsum('i,ijk->jk',w,np.asarray(a['action'][d['rows'],:5,:7],float));errs.append(np.max(np.abs(pred-np.asarray(d['served_head']))))
   assert errs and max(errs)<1e-5,(arm,max(errs) if errs else None)
   records.append(dict(arm=arm,run=run,episodes=len(episodes),duplicates=p['duplicates'],counts=counts,cache_action_samples=len(errs),max_cache_reconstruction_error=max(errs)))
  lo=np.load(TMP/f'library_{cell}.npz');nxt=np.asarray(a['next']);valid=(nxt>=0)&(a['episode'][np.maximum(nxt,0)]==a['episode'])&(a['step'][np.maximum(nxt,0)]==a['step']+1)
  assert np.isfinite(lo['error10']).sum()==valid.sum()
 for n in (50,500):
  ca=json.loads((OUT/f'causal_{n}.json').read_text());old=json.loads((CL/'r04_k5'/f'k5_g{n}_estimate.json').read_text())
  assert ca['ITT']['delta']==old['ITT']['delta']
 out=dict(status='PASS',arms=records,accepted_decisions=sum(r['counts']['N'] for r in records),cache_reconstruction_samples=sum(r['cache_action_samples'] for r in records),max_cache_reconstruction_error=max(r['max_cache_reconstruction_error'] for r in records),k5_point_estimates_equal_coordinator=True)
 dump('verification.json',out);print(json.dumps({k:v for k,v in out.items() if k!='arms'}))

if __name__=='__main__':main()
