from pathlib import Path
import ast, json, math, hashlib, datetime, collections, itertools
import numpy as np
from scipy.stats import binomtest
ROOT=Path('/home/weiland/projects/openpi'); OUT=Path('/tmp/r5_analysis'); CL=Path('/home/weiland/trace_runs/os_closed_loop')
PENDING=set()
old=ast.parse(Path('/home/weiland/.claude/jobs/a607dd74/tmp/analysis_r04/an.py').read_text())
LITERALS={n.targets[0].id:ast.literal_eval(n.value) for n in old.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['ARMS','K8']}
ARMS=LITERALS['ARMS']; K8=LITERALS['K8']
RUNS=['r05_ptail','r05_growth','r05_demo_curve','r05_q1','r05_q2','r05_q6','r05_b1','r05_q5','r05_x']
SPECS={}
for run in {x[0] for x in ARMS.values()}-{'DUAL'} | set(RUNS):
 p=CL/run/'arms.json'
 if p.exists():
  for a in json.loads(p.read_text()): SPECS[(run,a['arm'])]=a
for run in RUNS:
 for (rr,arm),s in list(SPECS.items()):
  if rr!=run:continue
  model='p' if s['model']=='pi05' else 'g'; suite='l10' if s['suite']=='libero_10' else 'sp'; cell=f'{model}_{suite}'
  kw=s.get('kwargs',{}); base=kw.get('base_kwargs',kw)
  lib=base.get('library',base.get('lib','big'))
  lib={'current':'50','big':'500','bpool_cs':'500','demo100':'100','demo200':'200','demo300':'300'}.get(lib,lib)
  fam=run.removeprefix('r05_');mk='BlindAWMtail'
  if 'policy_L10' in arm: lib='-'; fam='pure_inf_L10';mk='none'
  elif fam=='q5':mk='AWM'
  elif fam=='growth':mk='AWM'
  elif fam=='b1':mk='K7tail' if model=='p' else 'AWM'
  elif fam=='q1':mk='K7tail'
  elif fam=='q6':mk='WristTail'
  elif 'wrist_rep' in arm:mk='WristMixedJudge';lib='50'
  elif 'k7tail' in arm:mk='K7tail'
  ARMS[arm]=(run,arm,cell,lib,fam,mk)
ARMS['k7_sp_tail50']=('r04_k7','r4k7_p_sp_50_tail1ug','p_sp','50','blind_k7','K7tail')
def readj(path):
 acc={}; rejected=collections.Counter(); dup=0
 for line in path.open():
  if not line.strip():continue
  x=json.loads(line)
  if not (x.get('accepted') and x.get('status') in ('done','failed') and not x.get('error')):
   rejected[str(x.get('status'))]+=1;continue
  parts=x['task_uid'].split(':');key=(int(parts[-2]),int(parts[-1]))
  if key in acc:
   assert acc[key]['success']==x['success'];dup+=1
  acc[key]=x
 return acc,dict(rejected),dup
def wilson(k,n):
 z=1.95996398454;p=k/n;den=1+z*z/n;cen=(p+z*z/(2*n))/den;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [float(cen-h),float(cen+h)]
def cost(v,m,L,cell,mode):
 c=.148*v+.852*m if cell[0]=='g' else (.0552*v+.8979*m if mode=='wrist_only' else .152*v+.848*m)
 return c*5/L
rows={}; J={}; exclusions=[]; sources=[]
for lab,(run,arm,cell,lib,fam,mk) in ARMS.items():
 if run=='DUAL':continue # fixed historical references, not like-for-like pairing
 r={'label':lab,'run':run,'arm':arm,'cell':cell,'lib':lib,'family':fam,'search_method':mk}; rows[lab]=r
 p=CL/run;d=p/'runs'/arm;sm=d/'summary.json';done=p/'state'/f'{arm}.DONE';skip=p/'state'/f'{arm}.SKIPPED'
 if not done.exists():
  candidates=list((p/'state').glob(f'{arm}.manifest_*.DONE'))
  if candidates:
   manifest=ROOT/'exp/offline_search/rounds/r05/q4_growth/evaluation_pairs.json'
   assert run=='r05_growth' and len(candidates)==1
   saved=json.loads((d/'manifest.json').read_text())
   assert saved==json.loads(manifest.read_text())
   done=candidates[0]
   r['manifest_completion_marker']=str(done)
 if arm in PENDING:
  r['status']='TODO-PENDING: deferred by user until coordinator resume';continue
 if not done.exists() or not sm.exists() or skip.exists():
  r['status']='MISSING_OR_SKIPPED';exclusions.append([run,arm,done.exists(),sm.exists(),skip.exists()]);continue
 s=json.loads(sm.read_text());journal=d/'client/journal.jsonl';acc,reject,dup=readj(journal)
 n=len(acc);succ=sum(bool(x['success']) for x in acc.values());expect=250 if run=='r05_growth' else 500
 assert n==expect,(arm,n);assert s['complete']==n and s['success']==succ,(arm,'summary mismatch')
 expected={(t,i) for t in range(10) for i in range(25 if n==250 else 0,50)}
 assert set(acc)==expected,(arm,'pair IDs');assert not dup,(arm,'accepted duplicates')
 J[lab]={k:int(x['success']) for k,x in acc.items()}
 cl=s.get('cost_ledger');mode='full';L=5
 if cl:
  v,m=cl['v'],cl['m'];L=cl['l_per_request']['mean'];N=cl['decisions'];V=cl['vision_decisions'];M=cl['misses'];mode=next(iter(cl['stage1_modes']))
  assert abs(v-V/N)<1e-10 and abs(m-M/N)<1e-10
  csrc='per-arm cost_ledger'
 else:
  N=s['client_decisions'];V=N;M=(s.get('mixed') or {}).get('misses',0);v=1.;m=M/N;csrc='legacy summary counts; raw decision audit required'
  if fam.startswith('pure_inf'):v=m=1.;M=N
 owner=cost(v,m,L,cell,mode)
 spec=SPECS.get((run,arm),{});args=spec.get('plugin_args',[]);fit=None
 if '--os-fit-artifact' in args:fit=Path(args[args.index('--os-fit-artifact')+1])
 if fit is None or not fit.exists():
  fit=p/'fits'/f'{arm}.pkl'
 r.update(status='OK',n=n,success=succ,sr=succ/n,wilson=wilson(succ,n),v=v,m=m,L=L,N=N,V=V,M=M,stage1_mode=mode,ir=owner,ir_full=cost(v,m,L,cell,'full'),cost_source=csrc,fit_path=str(fit),fit_bytes=fit.stat().st_size if fit.exists() else None,bytes_per_entry=(s.get('server')or{}).get('bytes_per_entry'),kwargs=s.get('kwargs'),reject=reject,duplicates=dup,per_task=[sum(J[lab][t,i] for i in range(25 if n==250 else 0,50)) for t in range(10)],journal=str(journal),summary=str(sm),error_rows=s.get('error_rows'),eager_ir=cl.get('ir_per_five_controls') if cl else None)
 q=K8.get((mk,cell,lib));proxy=False
 if q is None and mk=='BlindAWMtail':
  z=K8.get(('BlindAWM',cell,lib));q=(z[0],.223) if z else None;proxy=True
 if q is None and mk=='WristTail':
  z=K8.get(('WristMixedJudge',cell,lib));q=(z[0],.223) if z else None;proxy=True
 if q is None and mk=='K7tail' and cell=='p_sp' and lib=='50':q=(1.530,.238);proxy=True
 if run in ['r05_q1','r05_q6','r05_q2','r05_b1']:proxy=True
 if q:
  ms=(q[0]*v+(q[1] or 0)*(1-v))*5/L
  r.update(search_ir=owner+ms/67.5,search_ms=ms,search_proxy=proxy,search_q=q)
 else:r.update(search_ir=None,search_proxy=True)
 # Dummy contract only supplied for pi05 full-camera arms. Wrist stacking and GR00T not validated.
 r['dummy_ir']=owner-.048*v*5/L if cell[0]=='p' and mode=='full' else owner
 r['dummy_applicable']=cell[0]=='p' and mode=='full'
 for f in [sm,journal,done]:sources.append({'path':str(f),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
# All pairwise comparisons within same model/suite; growth automatically intersects manifest.
def pair(a,b):
 keys=sorted(set(a)&set(b));d=np.array([a[k]-b[k] for k in keys]);n=len(d);wins=int(sum(d==1));loss=int(sum(d==-1));u,c=np.unique(d,return_counts=True)
 draws=np.random.default_rng(20260928).multinomial(n,c/n,10000)@u/n
 return {'n':n,'srA':sum(a[k] for k in keys)/n,'srB':sum(b[k] for k in keys)/n,'delta':float(d.mean()),'wins':wins,'losses':loss,'discordant':wins+loss,'p':float(binomtest(wins,wins+loss,.5).pvalue) if wins+loss else 1.,'ci':np.quantile(draws,[.025,.975]).tolist(),'per_task':[sum(a[k]-b[k] for k in keys if k[0]==t) for t in range(10)]}
pairs={}
for a,b in itertools.combinations(J,2):
 if rows[a]['cell']==rows[b]['cell']:pairs[a+'__'+b]=pair(J[a],J[b])
# pooled uses same 500 init clusters, one per-init average per configuration. Never 1500 iid episodes.
def pool(labs):
 keys=sorted(set.intersection(*(set(J[x]) for x in labs)));return {k:float(np.mean([J[x][k] for x in labs])) for k in keys}
pooled={}
def comp(name,A,B):
 a,b=pool(A),pool(B);p=pair(a,b);p['armsA']=A;p['armsB']=B;p['p']=None
 p['wins']=p['losses']=p['discordant']=None;pooled[name]=p
stock=['g500','g500_repa','g500_repb'];tail=['k7_tail1ug','r5b_p_l10_500_hand','r5x_p_l10_500_k7tail_repa','r5x_p_l10_500_k7tail_repb']
planned=['k7_tail1ug','r5x_p_l10_500_k7tail_repa','r5x_p_l10_500_k7tail_repb']
comp('K7_all4_vs_stock3',tail,stock)
comp('K7_planned3_vs_stock3',planned,stock)
for lab in ['r5q1_c10_p_l10_500','r5q6_p_l10_500_tail','r5q1_d1_p_l10_500','r5b_p_l10_500_solve','r5t_p_l10_500_tail1uc']:
 comp(lab+'_vs_K7_all4',[lab],tail);comp(lab+'_vs_K7_planned3',[lab],planned);comp(lab+'_vs_stock3',[lab],stock)
comp('wrist_sp50_available2_vs_stock',['wrist_sp_50','r5t_p_sp_g50_wrist_rep'],['sp_g50'])
comp('K7_sp500_available2_vs_C10',['k7_sp_tail1ug','r5b_p_sp_500_hand'],['r5q1_c10_p_sp_500'])
comp('Q6_sp500_vs_K7_available2',['r5q6_p_spatial_500_tail'],['k7_sp_tail1ug','r5b_p_sp_500_hand'])
comp('L10_l10_2_vs_L5_seeded2',['inf_l10_L10','inf_l10_L10b'],['inf_l10_s1001','inf_l10_s2001'])
for model in ['p','g','all']:
 labs=[x for x in J if x.startswith('r5b_') and x.endswith('_solve') and (model=='all' or x.startswith('r5b_'+model+'_'))]
 comp('B1_'+model,labs,[x.replace('_solve','_hand') for x in labs])
from pool_b1 import refresh
refresh(pooled,rows,J)
# Availability includes all configured R4 exclusions; smokes never read.
for run in sorted({x[0] for x in ARMS.values()}-{'DUAL'}):
 for p in (CL/run/'state').glob('*.SKIPPED'):exclusions.append([run,p.stem,'SKIPPED'])
out={'snapshot_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'arms':rows,'pairs':pairs,'pooled':pooled,'exclusions':exclusions,'pending':sorted(PENDING),'sources':sources}
(OUT/'analysis.json').write_text(json.dumps(out,indent=1))
(OUT/'journals.json').write_text(json.dumps({lab:{f'{t}:{i}':v for (t,i),v in j.items()} for lab,j in J.items()}))
print('Completed',len(J),'arms;',len(pairs),'pairs; exclusions',exclusions)
for lab,r in rows.items():
 if r['run'] in RUNS or lab=='k7_sp_tail50':print(lab,r.get('sr'),r.get('ir'),r.get('cost_source'),r.get('fit_bytes'))
