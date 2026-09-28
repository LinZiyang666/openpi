from pathlib import Path
import sys,json,re,math
import numpy as np
from scipy.stats import binomtest
sys.path.insert(0,'/tmp/r5_analysis');from tables import *
f=Path('/home/weiland/projects/openpi/exp/offline_search/rounds/r05/ANALYSIS.md');md=f.read_text()
heads=re.findall(r'^## (\d+)\.',md,re.M);assert heads==[str(i) for i in range(1,12)],heads
broken=[]
for target in re.findall(r'\]\(([^)]+)\)',md):
 if '://' not in target and not (f.parent/target).resolve().exists():broken.append(target)
print('Broken relative links',broken)
for lab,r in A.items():
 if r['status']!='OK':continue
 s=json.loads(Path(r['summary']).read_text());cl=s.get('cost_ledger')
 if cl:
  v=cl['vision_decisions']/cl['decisions'];m=cl['misses']/cl['decisions'];L=cl['l_per_request']['mean']
  expect=((.148*v+.852*m) if r['cell'][0]=='g' else (.0552*v+.8979*m) if r['stage1_mode']=='wrist_only' else (.152*v+.848*m))*5/L
  assert math.isclose(r['ir'],expect,abs_tol=1e-12),(lab,r['ir'],expect)
 assert f"`{r['run']}/{r['arm']}`" in md,lab
 if r['run'].startswith('r05'):
  d=D[lab];assert d['N']==r['N'] and d['V']==r['V'] and d['M']==r['M']
  assert not d['duplicates'] and not d['incomplete'] and not d['bad_decisions']
# Independently reconstruct several central paired tests from exported raw journal keys.
for a,b in [('r5t_p_l10_50_tail1uc','cl2_l10_50'),('r5x_g_sp_50_tail1u','gb_sp_50_tail2u'),('r5q4_p_sp_grow250_refit','cl2_sp_50')]:
 keys=set(J[a])&set(J[b]);w=sum(J[a][k]>J[b][k] for k in keys);l=sum(J[a][k]<J[b][k] for k in keys);p=pair(a,b)
 assert p['n']==len(keys) and (w,l)==(p['wins'],p['losses']) and abs(p['p']-binomtest(w,w+l).pvalue)<1e-12
# Verify explicitly reported within-family Holm conclusions.
def holm(ps):
 order=np.argsort(ps);out=np.empty(len(ps));c=0
 for rank,j in enumerate(order):c=max(c,min(1,ps[j]*(len(ps)-rank)));out[j]=c
 return out.tolist()
print('Pure-tail Holm',holm([pair(pt(s,n),f'cl2_{s}_{n}')['p'] for s in ['l10','sp'] for n in [50,500]]))
print('Refit-frozen Holm',holm([pair(f'r5q4_p_{s}_grow250_refit',f'r5q4_p_{s}_grow250_frozen')['p'] for s in ['l10','sp']]))
print('R5 checks passed;',len(D),'audited arms,',sum(d['N'] for d in D.values()),'decisions;',len(md.split()),'words')
# Final replicate and report checks.
assert not broken and 'TODO' not in md and 'PENDING' not in md and not X['pending']
repb='r5x_p_l10_500_k7tail_repb'
assert A[repb]['success']==441 and A[repb]['n']==500
assert Path('/home/weiland/trace_runs/os_closed_loop/r05_x/state/'+repb+'.DONE').exists()
assert 'IDLE_RECHECK.md' in md
for name in ['K7_all4_vs_stock3','K7_planned3_vs_stock3','r5q1_c10_p_l10_500_vs_K7_all4','r5q6_p_l10_500_tail_vs_K7_all4']:
 p=P[name];keys=sorted(set.intersection(*(set(J[l]) for l in p['armsA']+p['armsB'])))
 d=np.array([np.mean([J[l][k] for l in p['armsA']])-np.mean([J[l][k] for l in p['armsB']]) for k in keys])
 u,c=np.unique(d,return_counts=True);boot=np.random.default_rng(20260928).multinomial(len(keys),c/len(keys),10000)@u/len(keys)
 assert len(keys)==500 and abs(d.mean()-p['delta'])<1e-12
 assert np.allclose(np.quantile(boot,[.025,.975]),p['ci'],atol=1e-12)
old=json.loads((S/'analysis_before_repb.json').read_text())
for lab,r in old['arms'].items():
 if r['status']=='OK':assert r==A[lab],lab
for key,value in old['pairs'].items():assert X['pairs'][key]==value,key
prior=json.loads((S/'frontiers_before_repb.json').read_text())
now=json.loads((S/'frontiers.json').read_text())
assert now['pooled']==prior['pooled']
assert len([r for r in A.values() if r['status']=='OK'])==137
print('Final checks passed: four-run / prescribed-three-run pools independently recomputed; prior arm data unchanged; frontier membership unchanged; no TODO markers; links valid.')
