"""Numerical, population and contract checks for this research artifact; no simulator/model.
Run after select_configs.py. Existing test suites are neither read nor run.
"""
import json,pathlib,hashlib,importlib,numpy as np
from covariance import oas
from cost_solver import h_for_ir,periodic,causal_fit
from exp.offline_search.rounds.r02.g1_awm.awm import fit_metric
from exp.offline_search.rounds.r04.k5_rand.estimate import load_arm
O=pathlib.Path(__file__).parent;S=pathlib.Path('/home/weiland/trace_runs/offline_search_store');results={}
rng=np.random.default_rng(53);X=rng.normal(size=(50,11));H=rng.normal(size=(50,35));ep=np.repeat(np.arange(5),10)
mu,sd,W=fit_metric(X,H,ep,nn=3,lam=.1);z=(X-X.mean(0))/(X.std(0)+1e-6);D=np.maximum(np.sum(H*H,1)[:,None]+np.sum(H*H,1)[None,:]-2*H@H.T,0);D[ep[:,None]==ep[None,:]]=np.inf;ix=np.argsort(D,axis=1)[:,:3];dx=(z[:,None]-z[ix]).reshape(-1,11);sw=dx.T@dx/len(dx);mine=np.linalg.cholesky(np.linalg.inv(sw+.1*np.trace(sw)/11*np.eye(11)))
results['metric_factor_max_abs_difference_from_AWM']=float(np.max(abs(W-mine)));assert np.allclose(W,mine,atol=1e-12)
C,g=oas(dx);assert 0<=g<=1 and np.linalg.eigvalsh(C).min()>0;results['OAS_gamma_synthetic']=g
for v in [0.,.5,1.]:
 for m in [0.,v/2,v]:
  rho=.152*v+.848*m;assert abs(h_for_ir(rho,v)['required_m']-m)<1e-12
le=[1,5,8,17,60];p=periodic(le,.3)
for r in p['all']:assert r['M_frozen_lengths']==sum(sum(s%r['k']==r['k']-1 for s in range(n)) for n in le)
results['cost_inverse_and_periodic_exact']=True
records={}
for f in sorted([*O.glob('verified_library_*.json'),*O.glob('stock_library_*.json')]):
 d=json.loads(f.read_text());counts={}
 for r in d['episode_scores']:counts.setdefault((r['task'],r['episode']),set()).add(r['n_successor'])
 assert all(len(v)==1 for v in counts.values()),f'candidate-dependent population in {f}'
 assert len(d['LOTO']['action_RMS']['folds'])==10
 records[f.name]={'heldout_episodes':d['heldout_episodes'],'same_successor_population':True,'sigma_source':d.get('sigma_source','deployed library only'),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
assert len(records)==12, len(records)
results['library_outputs']=records
arms=json.loads((O/'arms_proposed.json').read_text());names=[]
for a in arms:
 mod,cl=a['method'].split(':');obj=getattr(importlib.import_module(mod),cl)(**a['kwargs']);names.append(obj.name)
results['proposed_constructor_checks']=len(names)
# Existing K5 smoke: it must yield no supported controller cells, never stand in for formal data.
rr=[]
for rep in [1,2]:
 rows,audit=load_arm('/home/weiland/trace_runs/os_closed_loop/r04_k5_smoke',f'r4k5_p_l10_g50_r{rep}');rr+=rows
fit=causal_fit(rr,.3234160220826887);assert all(c['suppress_probability']==0 for c in fit['cells']);results['K5_smoke_no_supported_override']={'episodes':len(rr),'contexts':len(fit['cells'])}
log=json.loads((O/'log_results.json').read_text());results['log_audit']={'arms':len(log['summaries']),'episodes':sum(r['episodes'] for r in log['summaries']),'decisions':sum(r['N'] for r in log['summaries']),'noncontiguous_episodes':sum(len(r['noncontiguous']) for r in log['summaries'])}
# Fit provenance uses identities; no multi-GB arrays copied or hashed into memory.
prov=[]
for cell in ['pi05_spatial','pi05_l10','groot_spatial','groot_l10']:
 for lib in ['current','bpool_cs' if cell.startswith('pi05') else 'bpool_all']:
  p=S/'library'/cell/lib
  prov.append({'library':str(p),'ids_sha256':hashlib.sha256((p/'ids.json').read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest()})
results['library_identity']=prov;results['status']='PASS'
(O/'verification.json').write_text(json.dumps(results,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in results.items() if k not in ['library_outputs','library_identity']},indent=1))
