"""Write a reviewable eight-arm OFFLINE plan; never launches a worker/server."""
import json,hashlib
from score import O,S
selections={}
for cell in ['groot_l10','groot_spatial']:
 eps=json.loads((S/'library'/cell/'bpool_all'/'episodes.json').read_text());banks=[[],[]]
 for t in sorted({e['task_id'] for e in eps}):
  chosen=sorted([e for e in eps if e['task_id']==t],key=lambda e:(hashlib.sha256(('r6q1_equal50_v1|'+cell+'|'+str(t)+'|'+e['stem']).encode()).hexdigest(),e['stem']))
  for b in range(2):banks[b].extend(chosen[b*5:(b+1)*5])
 for b,eps in enumerate(banks):
  name=cell+'_hash50_'+str(b);selections[name]=[{'file':e['file'],'stem':e['stem'],'task_id':e['task_id'],'source_start':e['start'],'source_end':e['end']} for e in eps]
  (O/(name+'_episodes.json')).write_text(json.dumps(selections[name],indent=2))
common={'execution':'A: anchor every 2 decisions, execute anchor chunk controls 0:10; no policy','method':'exp.offline_search.rounds.r04.k1_blind.blind_awm:BlindAWM','kwargs':{'lib':'current','kref':5,'k':16,'lam':.1,'nn':3,'early':True,'state_scale':1.,'serving':'anchor_tail','budget':1,'gates':'budget_only'},'evaluation':'all task IDs of suite, init IDs 5..49, paired across comparison','seed':20260929,'calibration':'recorded inference init IDs 0..4; no success labels used','status':'PROPOSED_ONLY; no execution authorized by this artifact'}
plan=[]
for i,(cell,lib,variant,pair) in enumerate([('pi05_spatial','demo200','refit',1),('pi05_spatial','demo300','refit',1),('pi05_l10','demo200','refit',2),('pi05_l10','demo200','frozen50',2),('groot_l10','hash50_0','refit',3),('groot_l10','hash50_1','refit',3),('groot_spatial','hash50_0','refit',4),('groot_spatial','hash50_1','refit',4)],1):
 a=dict(common,arm='r6q1_'+str(i),cell=cell,library=lib,variant=variant,pair=pair)
 if cell.startswith('pi05'):
  a['method']='exp/offline_search/rounds/r05/q4_growth/demo_method.py:DemoBlindAWM';a['kwargs']={'library':lib,'variant':variant,'kref':5,'serving':'anchor_tail','budget':1,'gates':'budget_only'}
 else:
  a['episode_manifest']=str(O/(cell+'_'+lib+'_episodes.json'));a['preparation']='Coordinator exports these complete episodes to current of an isolated store; fit standard AWM PCA+main/early metric on that bank only, including action sigma; score/freeze Q before running either arm.'
 plan.append(a)
(O/'proposed_arms.json').write_text(json.dumps(plan,indent=2));print('wrote 8 proposed arms, 4 disjoint-within-cell 50-episode manifests')
