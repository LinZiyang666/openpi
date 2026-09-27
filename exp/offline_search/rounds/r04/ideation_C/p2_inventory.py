import json
from pathlib import Path
import numpy as np
from exp.offline_search.harness.store import LibraryView, QueryCell
ROOT=Path('/home/weiland/trace_runs/offline_search_store'); OUT=Path(__file__).resolve().parent
out={}
for m in ['pi05','groot']:
 for s in ['spatial','l10']:
  k=f'{m}_{s}';lib=LibraryView(ROOT,k)
  out[k]={'library_meta':lib.meta,'library_episode_example':lib.episodes[0]}
  for n in ['current','bpool_cs' if m=='pi05' else 'bpool_all']:
   l=LibraryView(ROOT,k,n)
   out[k][n]={'L':l.L,'episodes':len(l.episodes),'tok':{f.stem:{'shape':list(np.load(f,mmap_mode='r').shape),'bytes':f.stat().st_size} for f in (l.dir/'tok').glob('*.npy')}}
  for a in ['inf','cache']:
   q=QueryCell(ROOT,k+'_'+a)
   out[k][a]={'episode_example':q.episodes[0],'N':q.N,'tok':{f.stem:{'shape':list(np.load(f,mmap_mode='r').shape),'bytes':f.stat().st_size} for f in (q.dir/'tok').glob('*.npy')}}
run=Path('/home/weiland/trace_runs/os_closed_loop/r03_mx')
out['mixed_arms']=json.loads((run/'arms.json').read_text())
out['r3_input_npz']=list(map(str,run.glob('runs/*/server_*/*.npz')))
f=next(run.glob('runs/r3mx_p_l10_g/server_*/decisions*.jsonl'))
for line in f.open():
 d=json.loads(line)
 if d.get('ev')=='dec' and not d.get('hit',True):out['mixed_row_example']=d;break
(OUT/'p2_inventory.json').write_text(json.dumps(out,indent=2))
for k in ['pi05_l10','groot_l10']:
 print(k, out[k]['library_meta'],out[k]['cache']['episode_example'],out[k]['current']['tok'])
print('r3 input files',len(out['r3_input_npz']))
print('mixed miss example',out['mixed_row_example'])
