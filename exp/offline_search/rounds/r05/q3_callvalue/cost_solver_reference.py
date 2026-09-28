"""Cost inverses and conservative one-landmark K5 policy fitter.
Input budget (rho), allowed SR loss and confidence level are owner decisions, not learnable constants.
Never extrapolate K5 to every decision, to blind operation, or across library scales.
"""
import argparse,json,pathlib,math
import numpy as np
from scipy.optimize import linprog
from scipy.stats import norm
from exp.offline_search.rounds.r04.k5_rand import estimate as k5
O=pathlib.Path(__file__).parent;R=pathlib.Path('/home/weiland/trace_runs/os_closed_loop')
def h_for_ir(rho,v=1.):
 m=(rho-.152*v)/.848
 return {'rho':rho,'v':v,'required_m':m,'required_h':1-m,'feasible_without_forced_calls':0<=m<=v}
def periodic(lengths,rho):
 n=sum(lengths);out=[]
 for k in range(1,101):
  m=sum(l//k for l in lengths);ir=.152+.848*m/n;out.append({'k':k,'M_frozen_lengths':m,'IR_frozen_lengths':ir})
 eligible=[r for r in out if r['IR_frozen_lengths']<=rho]
 return {'budget':rho,'minimum_k':eligible[0]['k'] if eligible else None,'all':out}
def lengths(run,arm):
 from log_audit import lines
 p=R/run/'runs'/arm;j={r['task_uid']:r for r in lines(p/'client/journal.jsonl') if r.get('accepted') and r.get('status') in ('done','failed') and not r.get('error')};ds={}
 for f in p.glob('server_*/decisions_*.jsonl'):
  for r in lines(f):
   if r.get('ev')=='dec' and r.get('uid') in j and r.get('attempt')==j[r['uid']].get('attempt'):ds[(r['uid'],r['step'])]=1
 return [sum(u==uid for u,s in ds) for uid in j]
def causal_fit(rows,rho,max_loss=.01):
 """One override only; LP uses simultaneous conservative cluster-normal bounds, then init-held-out OPE.
 A production decision needs the output LP+held-out check, not the in-sample optimum.
 """
 k5.validate_pairs(rows)
 keys=sorted({(r['landmark_class'],json.dumps(r['context'],sort_keys=True)) for r in rows if r['exposed']})
 # At most 48 leaf contexts; Bonferroni normal bounds, finite-sample approximate.
 z=float(norm.ppf(1-.05/(2*max(len(keys),1))));allclusters=sorted({(r['task_id'],r['init']) for r in rows});ct=len(allclusters);index={k:i for i,k in enumerate(allclusters)};cells=[]
 for key in keys:
  rr=[r for r in rows if r['exposed'] and (r['landmark_class'],json.dumps(r['context'],sort_keys=True))==key]
  groups={(r['task_id'],r['init']) for r in rr};nc=sum(r['assigned_treatment']=='CALL' for r in rr);n0=len(rr)-nc
  # Cluster contribution already includes context frequency in the whole-population policy effect.
  vals=np.zeros((ct,4))
  for r in rr:
   sign=1 if r['assigned_treatment']=='CALL' else -1
   vals[index[(r['task_id'],r['init'])]]+=sign*np.array([r['Y'],r['N'],r['M'],.848*r['M']+(.152-rho)*r['N']])
  mean=vals.mean(0);se=vals.std(0,ddof=1)/np.sqrt(ct)
  cells.append({'key':list(key),'n':len(rr),'clusters':len(groups),'n_call':nc,'n_cache':n0,'delta_Y_N_M_C_population':mean.tolist(),'SE':se.tolist(),'loss_ucb':float(mean[0]+z*se[0]),'saving_lcb':float(mean[3]-z*se[3]),'supported':len(groups)>=30 and min(nc,n0)>=10})
 # x_c is probability to suppress this single landmark's baseline CALL.
 bounds=[(0,1) if c['supported'] and c['saving_lcb']>0 else (0,0) for c in cells]
 if not cells:return {'status':'no exposed contexts','cells':[]}
 sol=linprog(-np.array([c['saving_lcb'] for c in cells]),A_ub=[np.array([max(c['loss_ucb'],0) for c in cells])],b_ub=[max_loss],bounds=bounds,method='highs')
 if not sol.success:raise RuntimeError(sol.message)
 for c,x in zip(cells,sol.x):c['suppress_probability']=float(x)
 return {'status':'fit, not deployed','rho':rho,'max_SR_loss':max_loss,'z_simultaneous_normal':z,'approximate_bounds':True,'cells':cells,'conservative_saving_C':float(-sol.fun)}
def ope(rows,fit,rho):
 policies={tuple(c['key']):c['suppress_probability'] for c in fit.get('cells',[])};groups={}
 for r in rows:
  key=(r['task_id'],r['init']);v=groups.setdefault(key,np.zeros(4));ctx=(r['landmark_class'],json.dumps(r['context'],sort_keys=True));x=policies.get(ctx,0.) if r['exposed'] else 0.
  # Difference vs always CALL: x * (CACHE - CALL), inverse propensities .5, /2 replicas.
  s=1 if r['assigned_treatment']=='CACHE' else -1;v+=x*s*np.array([r['Y'],r['N'],r['M'],.848*r['M']+(.152-rho)*r['N']])
 vv=np.array(list(groups.values()));return {'n_clusters':len(vv),'policy_minus_CALL_Y_N_M_C':vv.mean(0).tolist(),'SE':(vv.std(0,ddof=1)/np.sqrt(len(vv))).tolist()}
def main():
 out={'inverses':[h_for_ir(x) for x in [.178,.203,.242,.28,.315]],'vision_saved_per_added_MISS':.848/.152,'full_call_price':1.,'k5':{},'periodic':{}}
 for scale,arm in [(50,'r3mx_p_l10_g'),(500,'r3mx_p_l10_g500')]:
  le=lengths('r03_mx',arm);out['periodic'][str(scale)]=[periodic(le,b) for b in ([.242,.28,.315] if scale==50 else [.178,.203,.245])]
  arms=[f'r4k5_p_l10_g{scale}_r{i}' for i in (1,2)];root=R/'r04_k5'
  if not all((root/'runs'/a/'client/journal.jsonl').exists() for a in arms):out['k5'][str(scale)]={'status':'formal randomized data absent; no fit/no effects invented','arms':arms};continue
  rows=[]
  for a in arms:rr,audit=k5.load_arm(root,a);rows+=rr
  gg=k5.validate_pairs(rows)
  if len(gg)!=500:out['k5'][str(scale)]={'status':'incomplete formal pair; no fit','clusters':len(gg)};continue
  rho=k5.RHOS[f'g{scale}'];cv=[]
  for f in range(5):
   tr=[r for r in rows if r['init']%5!=f];te=[r for r in rows if r['init']%5==f];fit=causal_fit(tr,rho);cv.append(ope(te,fit,rho))
  out['k5'][str(scale)]={'fit':causal_fit(rows,rho),'init_fold_OPE':cv,'provenance':'borrowed big-library information','deployment_status':'research artifact; independent closed-loop pilot required'}
 (O/'cost_results.json').write_text(json.dumps(out,indent=2,allow_nan=False))
 print({k:v.get('status','fitted') for k,v in out['k5'].items()})
if __name__=='__main__':main()
