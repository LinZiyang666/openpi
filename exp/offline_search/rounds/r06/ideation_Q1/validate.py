"""Artifact invariants, online adapter numerical check, independent rank check."""
import json,types,ast,warnings
import numpy as np
from scipy.stats import spearmanr,rankdata
from score import *
from quality import load_audit_calibration
checks=[];max_adapter=0;states=0
for j in joblist():
 tag='_'.join([j['cell'],j['library'],j['variant']]);sm=json.loads((O/(tag+'.json')).read_text());p=S/'library'/j['cell']/j['library'];d=dataset(p,True)
 assert sm['rows']==len(d['ep']) and sm['episodes']==len(np.unique(d['ep']))
 for source in ['loeo','inf','cache']:
  a=np.load(O/(tag+'_'+source+'.npz'));finite=np.isfinite(a['d1']);assert np.all(a['d1'][finite]>=0);assert np.all((a['q'][np.isfinite(a['q'])]>0)&(a['q'][np.isfinite(a['q'])]<=1))
  if source=='loeo':assert finite.all()
  if source!='loeo' and j['library']=='grow250':assert np.all(a['init'][finite]>=25)
  for key in ['edge1','edge2']:assert np.nanmin(a[key])>=0 and np.nanmax(a[key])<=1.00001
  states+=int(finite.sum())
 if j['fit']:
  m=loadmethod(j);a=np.load(O/(tag+'_inf.npz'));qp=S/'queries'/(j['cell']+'_inf');ca=load_audit_calibration(O,tag);valid=np.flatnonzero(np.isfinite(a['q_online'])&(a['step']>0));qs=valid[np.linspace(0,len(valid)-1,5,dtype=int)]
  for r in qs:
   q=types.SimpleNamespace(key_v0=ar(qp,'key_v0')[r],key_v1=ar(qp,'key_v1')[r],rs=ar(qp,'rs')[r],step=int(a['step'][r]),prev_hit=True,task_id=int(a['task'][r]));dd=m._dist(q)[-1];ix=np.argpartition(dd,15)[:16];ix=ix[np.lexsort((ix,dd[ix]))];dk=dd[ix].astype(float);w=np.exp(-((dk-dk[0])/max(dk[m.kref-1]-dk[0],1e-6))**2);value=ca.at_anchor(q.task_id,m.tasks[q.task_id].rows[ix],w,dd.min());delta=abs(value['quality']-a['q_online'][r]);max_adapter=max(delta,max_adapter)
   assert delta<.005,(tag,r,delta)
  checks.append({'tag':tag,'n_online_adapter':len(qs)})
# Independent scipy check for every reported ordinary global/per-task coefficient.
R=json.loads((O/'closed_loop_records.json').read_text());co=json.loads((O/'rank_correlations.json').read_text());max_rho=0
for panel,sc in co.items():
 for key,modes in sc.items():
  rr=[r for r in R[panel] if r.get(key) is not None];x=np.array([r[key] for r in rr]);y=np.array([r['y'] for r in rr]);labels=list(dict.fromkeys(r['label'] for r in rr))
  for mode in ['global','task']:
   if mode=='global':xx=[np.mean([r[key] for r in rr if r['label']==l]) for l in labels];yy=[np.mean([r['y'] for r in rr if r['label']==l]) for l in labels]
   else:xx=x;yy=y
   val=spearmanr(xx,yy).statistic;delta=abs(val-modes[mode]['rho']);max_rho=max(delta,max_rho);assert delta<1e-12
summary={'states_scored':states,'fits':len(joblist()),'online_adapter_queries':sum(c['n_online_adapter'] for c in checks),'online_adapter_max_q_abs_difference':max_adapter,'independent_rho_max_difference':max_rho,'all_assertions_passed':True}
(O/'validation.json').write_text(json.dumps(summary,indent=2));print(summary)
