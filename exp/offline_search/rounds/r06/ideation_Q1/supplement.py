import json,csv,hashlib,pathlib,collections
import numpy as np
from scipy.stats import spearmanr
from score import O,S,joblist,macro
scores={};rows=[];tasks=[];overlaps=[]
for j in joblist():
 tag='_'.join([j['cell'],j['library'],j['variant']]);r=json.loads((O/(tag+'.json')).read_text());scores[tag]=r
 for source,values in r['metrics'].items():
  rows.append({'tag':tag,'source':source,'episodes':r['episodes'],'rows':r['rows'],**values['global']})
  for t,vals in values['tasks'].items():tasks.append({'tag':tag,'source':source,'task':t,**{k:v for k,v in vals.items() if not isinstance(v,dict)}})
 eps=json.loads((S/'library'/j['cell']/j['library']/'episodes.json').read_text());files={e['file'] for e in eps}
 for source in ['inf','cache']:
  qe=json.loads((S/'queries'/(j['cell']+'_'+source)/'episodes.json').read_text());overlaps.append({'tag':tag,'source':source,'shared_source_files':len(files&{e['file'] for e in qe})})
def csvwrite(name,data):
 keys=list(dict.fromkeys(k for r in data for k in r))
 with (O/name).open('w') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(data)
csvwrite('library_scores.csv',rows);csvwrite('task_scores.csv',tasks)
C=json.loads((O/'rank_correlations.json').read_text());csvwrite('correlations.csv',[{'panel':p,'score':s,'mode':mode,'rho':v['rho'],'ci_low':v['ci'][0] if v['ci'] else None,'ci_high':v['ci'][1] if v['ci'] else None,'n_configs':v['n_configs'],'n_task_cells':v['n_task_cells']} for p,ss in C.items() for s,modes in ss.items() for mode,v in modes.items()])
R=json.loads((O/'closed_loop_records.json').read_text());csvwrite('closed_loop_task_scores.csv',[{'panel':p,**r} for p,rr in R.items() for r in rr])
# Small calibration stream: deterministic first 5 inits/task, estimate scalar mean residual correction,
# evaluate only inits >= 5. Grow: first 5 HELD-OUT inits 25..29, evaluate 30..49.
cal=[]
for j in joblist():
 tag='_'.join([j['cell'],j['library'],j['variant']]);a=np.load(O/(tag+'_inf.npz'));lo=25 if j['library']=='grow250' else 0;valid=np.isfinite(a['q_online'])&np.isfinite(a['succ2']);split=valid&(a['init']>=lo)&(a['init']<lo+5);test=valid&(a['init']>=lo+5)
 # Fit no success labels and no retrieval parameters; match each predicted residual's episode-macro mean.
 for key,pred in [('err10','pred_err'),('succ2','pred_succ')]:
  num=np.mean([macro(a[key][split&(a['task']==t)],a['ep'][split&(a['task']==t)]) for t in np.unique(a['task'])]);den=np.mean([macro(a[pred][split&(a['task']==t)],a['ep'][split&(a['task']==t)]) for t in np.unique(a['task'])]);factor=num/den
  task=[]
  for t in np.unique(a['task']):
   z=test&(a['task']==t);obs=macro(a[key][z],a['ep'][z]);before=macro(a[pred][z],a['ep'][z]);task.append((obs,before,before*factor))
  arr=np.array(task);cal.append({'tag':tag,'metric':key,'calibration_episodes':len(np.unique(a['ep'][split])),'test_episodes':len(np.unique(a['ep'][test])),'factor':factor,'actual_mean':float(arr[:,0].mean()),'before_mean':float(arr[:,1].mean()),'after_mean':float(arr[:,2].mean()),'task_MAE_before':float(np.mean(abs(arr[:,0]-arr[:,1]))),'task_MAE_after':float(np.mean(abs(arr[:,0]-arr[:,2])))})
# Whole-trajectory influence of successful/failed logs, full and first 20% observation only.
status=[]
for j in joblist():
 tag='_'.join([j['cell'],j['library'],j['variant']]);a=np.load(O/(tag+'_inf.npz'))
 for suc in [0,1]:
  z=np.isfinite(a['d1'])&(a['success']==suc);status.append({'tag':tag,'query_success':suc,'episodes':len(np.unique(a['ep'][z])),'covered':macro(a['covered'][z],a['ep'][z]),'q':macro(a['q'][z],a['ep'][z])})
# Show common library/fit assigning one score to many different closed-loop controllers.
ranges=[]
for tag in sorted({r['tag'] for r in R['all_requested']}):
 gr=collections.defaultdict(list)
 for r in R['all_requested']:
  if r['tag']==tag:gr[r['label']].append(r['y'])
 vals={k:float(np.mean(v)) for k,v in gr.items()};ranges.append({'tag':tag,'min_arm':min(vals,key=vals.get),'max_arm':max(vals,key=vals.get),'min_sr':min(vals.values()),'max_sr':max(vals.values()),'n_arms':len(vals)})
(O/'supplement.json').write_text(json.dumps({'overlap':overlaps,'small_calibration':cal,'query_success_split':status,'same_library_controller_range':ranges},indent=2))
# Artifact and source provenance; source manifests contain identity/hash inventory for huge arrays.
inputs=[]
for j in joblist():
 ps=[S/'library'/j['cell']/j['library']/f for f in ['manifest.json','ids.json','episodes.json']]
 if j['fit']:ps.append(pathlib.Path(j['fit']))
 for p in ps:
  inputs.append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(O/'input_hashes.json').write_text(json.dumps(inputs,indent=2))
print('wrote',len(rows),'library-source rows;',len(tasks),'task-source rows')
