"""Read-only asset/accelerator audit; outputs only inside ideation_B."""
import os,json,pathlib,subprocess,numpy as np
OUT=pathlib.Path(__file__).resolve().parent
ROOT=pathlib.Path('/home/weiland/trace_runs/offline_search_store')
r={'affinity':sorted(os.sched_getaffinity(0)), 'hot_exists':pathlib.Path('/dev/shm/offline_search_store').exists(), 'cold':str(ROOT), 'datasets':{}}
p=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free','--format=csv'],capture_output=True,text=True)
r['nvidia_smi']={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
for part in ('library','queries'):
 for model in ('pi05','groot'):
  for suite in ('spatial','l10'):
   names=('current','bpool_cs' if model=='pi05' else 'bpool_all') if part=='library' else ('inf','cache')
   for name in names:
    d=ROOT/part/(model+'_'+suite)/(name) if part=='library' else ROOT/part/(model+'_'+suite+'_'+name)
    a={}
    for f in sorted(d.glob('*.npy'))+sorted((d/'tok').glob('*.npy')):
     x=np.load(f,mmap_mode='r'); a[str(f.relative_to(d))]={'shape':list(x.shape),'dtype':str(x.dtype),'file_bytes':f.stat().st_size}
    r['datasets'][str(d.relative_to(ROOT))]=a
(OUT/'inventory.json').write_text(json.dumps(r,indent=2))
print(json.dumps({'hot_exists':r['hot_exists'],'gpu':r['nvidia_smi'],'datasets':len(r['datasets'])},indent=2))
