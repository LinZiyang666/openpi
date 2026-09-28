import os;os.sched_setaffinity(0,{34})
from common import *
inv=json.loads((OUT/'inventory.json').read_text());configs=[];missing=[]
for model in ('pi05','groot'):
 for suite in ('spatial','l10'):
  for scale in (50,500):
   cell=f'{model}_{suite}_cache'
   for family in ('AWM','MixedJudge','BlindMixedJudge','BlindAWM','WristAWM','WristMixedJudge','BlindWristMixedJudge','ControlG','ControlGS'):
    def ok(r):
     cls=r.get('class_name');kw=r.get('kwargs',{});bk=kw.get('base_kwargs',kw)
     if r.get('cell')!=cell or r.get('scale')!=scale:return False
     if family=='AWM':return cls=='AWM' and 'cl2.pkl' in r['path']
     if family=='MixedJudge':return cls==family and kw.get('events')=='none' and kw.get('noprog_n',3)==3
     if family in ('BlindMixedJudge','BlindAWM','BlindWristMixedJudge'):return cls==family and bk.get('serving','phase_particles')=='phase_particles' and bk.get('budget',2)==2 and bk.get('gates','all')=='all'
     if family=='WristAWM':return cls=='WristMixedJudge'
     if family.startswith('Control'):return cls=='ControlStepLibrary' and kw.get('ablation')==family[7:]
     return cls==family
    candidates=[r for r in inv if ok(r)]
    if not candidates:
     missing.append(dict(model=model,suite=suite,scale=scale,family=family,reason='implementation pi05-only' if 'Wrist' in family and model=='groot' else 'no matching fitted artifact in inventory'));continue
    candidates.sort(key=lambda r:(not r['path'].startswith(str(RUNS)),not ('_g.pkl' in r['path'] or '_g500.pkl' in r['path']),r['path']))
    r=dict(candidates[0]);r.update(id=f'{model}_{suite}_{scale}_{family}',family=family,extract_base=family=='WristAWM');configs.append(r)
dump(OUT/'configs.json',configs);dump(OUT/'missing.json',missing)
print(len(configs),'configs;',len(missing),'missing')
