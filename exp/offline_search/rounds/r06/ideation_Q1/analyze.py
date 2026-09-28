"""Associations with accepted closed-loop journals, paired per task/init; no new experiments."""
import json,pathlib,collections,hashlib,sys,warnings
import numpy as np
from scipy.stats import spearmanr,rankdata
from score import O,S,C,joblist,macro
warnings.filterwarnings('ignore',message='An input array is constant')
SEED=20260928;BOOT=2000
runs=['r05_demo_curve','r05_growth','r05_ptail','r05_x','r04_blind','r04_gblind','r02_g50','r02_g500','r05_q1','r05_q2']
# R5 audit is only an index: reread each summary/journal and completion marker here.
index_path=pathlib.Path('/tmp/r5_analysis/analysis.json')
old=json.loads(index_path.read_text())['arms'] if index_path.exists() else json.loads((O/'source_index.json').read_text())['arms']
J={};arms={};sources=[]
for label,a in old.items():
 if a.get('status')!='OK':continue
 p=pathlib.Path(a['summary']);jp=pathlib.Path(a['journal']);root=C/a['run'];arm=a['arm']
 markers=list((root/'state').glob(arm+'.DONE'))+list((root/'state').glob(arm+'.manifest_*.DONE'))
 if not markers:continue
 sm=json.loads(p.read_text());acc={}
 for line in jp.read_text().splitlines():
  r=json.loads(line)
  if r.get('accepted') and r.get('status') in ('done','failed') and not r.get('error'):
   key=tuple(map(int,r['task_uid'].split(':')[-2:]));assert key not in acc;acc[key]=int(r['success'])
 assert len(acc)==sm['complete'] and sum(acc.values())==sm['success']
 J[label]=acc;arms[label]=a
 sources.extend({'path':str(q),'sha256':hashlib.sha256(q.read_bytes()).hexdigest()} for q in [p,jp,markers[0]])

scores={}
for job in joblist():
 tag='_'.join([job['cell'],job['library'],job['variant']]);p=O/(tag+'.json')
 if p.exists():scores[tag]=json.loads(p.read_text())

def skey(a):
 cell=a['cell'].replace('p_','pi05_',1) if a['cell'].startswith('p_') else a['cell'].replace('g_','groot_',1);cell=cell.replace('_sp','_spatial')
 lib={'50':'current','500':'bpool_cs' if cell.startswith('pi05') else 'bpool_all','100':'demo100','200':'demo200','300':'demo300'}.get(a['lib'],a['lib'])
 return cell+'_'+lib+('_frozen' if 'frozen' in a['arm'] else '_refit')

# Sixteen pure 10-control A arms (eight endpoints and eight demo geometry/size arms).
A=[]
for k,a in arms.items():
 if (a['run']=='r05_demo_curve' or a['run']=='r05_ptail' and 'tail1uc' in a['arm'] or a['run']=='r05_x' and a['arm'].startswith('r5x_g_') or a['arm']=='r4b3_p_sp_500_tail1uc'):A.append(k)
A5=[k for k,a in arms.items() if a['run']=='r05_growth' or a['run'] in ['r02_g50','r02_g500'] and a['arm'].endswith('_cl2')]
ALL=[k for k,a in arms.items() if a['run'] in runs and skey(a) in scores]
# all includes non-A methods at same bank deliberately as stress/falsification, separately labeled.
REF10={cell:[k for k,a in arms.items() if a['cell']==cell and a['family']=='pure_inf_L10'] for cell in {a['cell'] for a in arms.values()}}
REF5={cell:[k for k,a in arms.items() if a['cell']==cell and a['family']=='pure_inf'] for cell in {a['cell'] for a in arms.values()}}

def poolj(kk):
 keys=set.intersection(*(set(J[k]) for k in kk));return {pair:float(np.mean([J[k][pair] for k in kk])) for pair in keys}

# Attach paired baseline outcomes and all score sources, only from their matching fit.
metrics=[(src,key) for src in ['loeo','inf','cache'] for key in ['d_pair','d_self','covered','err5','err10','exec10','succ1','succ2','q','q_online','episodes_any_break','longest_bad_fraction'] if src!='loeo' or key!='q_online']
signs={k:(1 if k in ['covered','q','q_online'] else -1) for _,k in metrics}

def records(labels,target='sr',gain=False,half=False):
 out=[]
 for lab in labels:
  a=arms[lab];tag=skey(a)
  if tag not in scores:continue
  cell=a['cell'];s=scores[tag];y=J[lab];base=None
  if target.startswith('gap'):
   refs=(REF10 if target=='gap10' else REF5).get(cell,[])
   if refs:base=poolj(refs)
   elif target=='gap5':
    qp=S/'queries'/(s['job']['cell']+'_inf');eps=json.loads((qp/'episodes.json').read_text());base={(int(e['task_id']),int(e['init'])):int(e['success']) for e in eps}
   else:continue
  if gain:
   al=next((k for k in A if arms[k]['cell']==cell and arms[k]['lib']==a['lib']),None)
   if not al:continue
   base=J[al]
  for t in range(10):
   pairs=[p for p in y if p[0]==t and (base is None or p in base) and (not half or p[1]>=25)]
   vals=np.array([y[p]-(base[p] if base else 0) for p in pairs]);row={'label':lab,'cell':cell,'suite':'spatial' if cell.endswith('sp') else 'l10','task':t,'lib':a['lib'],'tag':tag,'y':float(vals.mean()),'n':len(vals)}
   for src,key in metrics:
    val=s['metrics'][src]['tasks'][str(t)].get(key)
    if half and src!='loeo':
     arr=np.load(O/(tag+'_'+src+'.npz'));v=(arr['task']==t)&(arr['init']>=25)&np.isfinite(arr['d1'])
     if key in arr:val=macro(arr[key][v],arr['ep'][v])
    row[src+'_'+key]=None if val is None else signs[key]*val
   row['library_success']=sum(e['success'] for e in json.loads((S/'library'/s['job']['cell']/s['job']['library']/'episodes.json').read_text()) if e['task_id']==t)/sum(e['task_id']==t for e in json.loads((S/'library'/s['job']['cell']/s['job']['library']/'episodes.json').read_text()))
   row['log_episodes']=float(np.log(s['metrics']['loeo']['tasks'][str(t)]['episodes']))
   out.append(row)
 return out

def rho(x,y):
 if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return np.nan
 return float(spearmanr(x,y).statistic)

def corr(rows,key,mode='global',boot=BOOT):
 rows=[r for r in rows if r.get(key) is not None and np.isfinite(r[key])]
 labs=list(dict.fromkeys(r['label'] for r in rows));suites=sorted({r['suite'] for r in rows});tasks={s:sorted({r['task'] for r in rows if r['suite']==s}) for s in suites}
 def rc(x,y):
  x=rankdata(x,axis=-1);y=rankdata(y,axis=-1);x-=x.mean(-1,keepdims=True);y-=y.mean(-1,keepdims=True)
  with np.errstate(divide='ignore',invalid='ignore'):return np.sum(x*y,-1)/np.sqrt(np.sum(x*x,-1)*np.sum(y*y,-1))
 rng=np.random.default_rng(SEED);draws={s:np.vstack([np.arange(len(ts)),rng.integers(len(ts),size=(boot,len(ts)))]) for s,ts in tasks.items()}
 X=[];Y=[]
 for lab in labs:
  rr=sorted([r for r in rows if r['label']==lab],key=lambda r:r['task']);suite=rr[0]['suite'];assert [r['task'] for r in rr]==tasks[suite]
  X.append(np.array([r[key] for r in rr])[draws[suite]]);Y.append(np.array([r['y'] for r in rr])[draws[suite]])
 X=np.stack(X,axis=1);Y=np.stack(Y,axis=1)
 if mode=='global':result=rc(X.mean(-1),Y.mean(-1))
 elif mode=='task':result=rc(X.reshape(boot+1,-1),Y.reshape(boot+1,-1))
 elif mode=='within_config':result=np.nanmean(rc(X,Y),axis=1)
 else:
  vals=[]
  for cell in sorted({r['cell'] for r in rows}):
   rr=[r for r in rows if r['cell']==cell];suite=rr[0]['suite'];v=[]
   for t in tasks[suite]:
    tt=[r for r in rr if r['task']==t];v.append(rho([r[key] for r in tt],[r['y'] for r in tt]))
   vals.append(np.array(v)[draws[suite]])
  with warnings.catch_warnings():
   warnings.simplefilter('ignore');result=np.nanmean(np.concatenate(vals,axis=1),axis=1)
 good=result[1:][np.isfinite(result[1:])]
 return {'rho':float(result[0]) if np.isfinite(result[0]) else None,'ci':np.quantile(good,[.025,.975]).tolist() if len(good) else None,'n_configs':len(labs),'n_task_cells':len(rows),'mode':mode,'valid_boot':len(good)}

B=[k for k,a in arms.items() if a['run']=='r05_q1' and 'c10_' in a['arm'] or a['run']=='r05_q2' and a['arm'].endswith('_G10')]
sets={'A':records(A),'A_gap10':records(A,'gap10'),'A_gap5':records(A,'gap5'),'A5_growth':records(A5,half=True),'all_requested':records(ALL),'B_gain':records(B,gain=True)}
(O/'closed_loop_records.json').write_text(json.dumps(sets,indent=1));(O/'source_index.json').write_text(json.dumps({'sources':sources,'arms':arms,'A':A,'A5':A5,'B':B,'all_requested':ALL},indent=1))
# Efficiently bootstrap a focused prespecified subset; retain point estimates for every metric.
focus=['loeo_err5','loeo_exec10','loeo_succ1','inf_err5','inf_succ1','cache_err5','cache_exec10','cache_succ1','cache_d_self','cache_covered','cache_q_online','loeo_d_self','loeo_covered','loeo_longest_bad_fraction','inf_longest_bad_fraction','cache_longest_bad_fraction','loeo_d_pair','loeo_err10','loeo_succ2','loeo_q','inf_d_pair','inf_d_self','inf_covered','inf_err10','inf_exec10','inf_succ2','inf_q','inf_q_online','cache_d_pair','cache_err10','cache_succ2','cache_q','library_success','log_episodes']
out={}
for name,rows in sets.items():
 out[name]={}
 for key in focus:
  # Stats preserve suite/task clusters; intervals are task-generalization conditional on recorded library/run fits.
  out[name][key]={mode:corr(rows,key,mode,BOOT) for mode in ['global','task']}
  print(name,key,out[name][key]['global']['rho'],flush=True)
 for key in ['loeo_q','inf_q','inf_d_pair','inf_err10','inf_succ2','inf_q_online','log_episodes']:
  out[name][key].update({mode:corr(rows,key,mode,BOOT) for mode in ['within_config','within_task']})
 (O/'rank_correlations.json').write_text(json.dumps(out,indent=2,allow_nan=False))
# Family holdout: no SR fitting used; these expose sign stability on each model/suite.
bycell={}
for cell in sorted({r['cell'] for r in sets['A']}):
 rs=[r for r in sets['A'] if r['cell']==cell];bycell[cell]={k:corr(rs,k,'global',BOOT) for k in ['loeo_q','inf_q','inf_d_pair','inf_err10','inf_succ2','log_episodes']}
(O/'rank_within_cell.json').write_text(json.dumps(bycell,indent=2,allow_nan=False))
print('DONE stats',len(A),len(A5),len(ALL),len(B),flush=True)
