"""Replay pre-existing plugin NPZ logs through the measured fit objects."""
import os;os.sched_setaffinity(0,{35})
from common import *
from types import SimpleNamespace
from exp.offline_search.closed_loop.blind import BlindQueryView,BlindResult
from benchmark import reset_prof
results=[];configs=json.loads((OUT/'configs.json').read_text())
for c in configs:
 dirs=[]
 if c['family']=='BlindMixedJudge':
  dirs=[pathlib.Path('/tmp/k6_installed/existing/k2')/f"final_{'l10' if '_l10_' in c['cell'] else 'spatial'}_p{c['scale']}"]
 elif c['family']=='MixedJudge' and '_l10_' in c['cell']:
  dirs=[pathlib.Path('/tmp/k6_installed_concurrency')/f"mixed_{c['scale']}"/'threaded']
 if not dirs:continue
 m=load(c['path'])['method'];reset_prof(m,api.NULL_PROFILER)
 for dr in dirs:
  for path in sorted(dr.glob('inputs/*.npz'))[:4]:
   z=np.load(path,allow_pickle=False);meta=json.loads(str(z['meta']));n=len(z['step'])
   ep=api.EpisodeView(meta['uid'],meta['task'],meta['task_id'],meta['init'],meta['index'],meta['seed'])
   s=SimpleNamespace(rt=SimpleNamespace(model=meta['model'],api=api,opts=SimpleNamespace(os_tokens='off')),hits=np.asarray(z['hit'],int).tolist(),has_vision=np.asarray(z['has_vision'],bool).tolist(),last_vision_step=0,_age_before=0)
   for name,key in [('b_v0','key_v0'),('b_v1','key_v1'),('b_rs','rs'),('b_raw','raw_state'),('b_aex','a_exec')]:
    ar=np.array(z[key]);buf=plugin._Buf(ar.shape[1:],ar.dtype);buf.a=ar;buf.n=len(ar);setattr(s,name,buf)
   m.reset(ep);match=0;act=0;scores=0;blind=0;fail=[];max_delta=0.
   for j in range(n):
    q=plugin.OnlineQueryView(s,j,ep.task_id,ep);s._age_before=int(z['blind_age'][j]) if 'blind_age' in z else 0
    if s.has_vision[j]:
     # Failed blind attempt can update gate diagnostics but not selection.
     if hasattr(m,'blind_step') and j and s.hits[j-1] and int(z['look_reason'][j]) not in (6,7):
      bq=BlindQueryView(q.step,q.task_id,q.episode,q.rs,q.raw_state,q.prev_hit,q.prev_a_exec,q.hist_a_exec,q.hist_hit,q.hist_rs,q.hist_has_vision,s._age_before);m.blind_step(bq)
     r=m.query(q);rr=r.topk[:10];expect=z['topk'][j];expect=expect[expect>=0]
     good=np.array_equal(rr,expect);match+=int(good);scores+=int(np.array_equal(r.scores[:10],z['scores'][j,:len(rr)]));action=r.action
     if not good:fail.append([j,'topk'])
     s.last_vision_step=j
    else:
     bq=BlindQueryView(q.step,q.task_id,q.episode,q.rs,q.raw_state,q.prev_hit,q.prev_a_exec,q.hist_a_exec,q.hist_hit,q.hist_rs,q.hist_has_vision,s._age_before);r=m.blind_step(bq);assert isinstance(r,BlindResult)
     rows=z['blind_rows'][j];rows=rows[rows>=0];good=np.array_equal(r.rows,rows);match+=int(good);blind+=1;action=r.action
     if not good:fail.append([j,'blind_rows'])
    if action is not None:
     same=np.array_equal(action,z['synth'][j]);act+=int(same);max_delta=max(max_delta,float(np.abs(action-z['synth'][j]).max()))
     if not same:fail.append([j,'action'])
    if not s.hits[j] and hasattr(m,'invalidate_anchor'):m.invalidate_anchor()
   results.append(dict(config=c['id'],fit=c['path'],log=str(path),decisions=n,topk_equal=match,action_equal=act,vision_scores_equal=scores,blind=blind,max_action_delta=max_delta,failures=fail))
   print(c['id'],n,match,act,blind,len(fail),flush=True)
   dump(OUT/'replay_verification.json',results)
assert all(r['decisions']==r['topk_equal']==r['action_equal'] and not r['failures'] for r in results)
