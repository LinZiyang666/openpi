"""Validate the handoff through installed emit_arms and CacheConfig without starting a server."""
import json,pathlib,subprocess,sys
import yaml
from exp.offline_search.closed_loop import plugin
from openpi.cache.config import load_cache_config

HERE=pathlib.Path(__file__).resolve().parent
OUT=HERE/'results'
run=pathlib.Path('/tmp/k1_r4_emit_final')
rows=json.loads((HERE/'arms_r4.json').read_text())
resolved=json.loads(json.dumps(rows).replace('<RUN>',str(run)))
path=OUT/'arms_resolved_test.json';path.write_text(json.dumps(resolved,indent=2))
prefix=['taskset','-c','18-21,62-65','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
        'CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1',sys.executable]
command=prefix+['-m','exp.offline_search.closed_loop.ops.emit_arms','--run-root',str(run),'--spec',str(path)]
with (OUT/'emit_arms_final.log').open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
emitted=json.loads((run/'arms.json').read_text());assert len(emitted)==len(rows)
batch={}
for s,r in zip(resolved,emitted):
    cls,_=plugin.load_method_class(s['method']);m=cls(**s['kwargs'])
    cfg=load_cache_config(r['yaml'])
    if 'yaml_patch' in s:
        y=yaml.safe_load(pathlib.Path(r['yaml']).read_text());assert y['miss']['num_steps']==2
        assert '<RUN>' not in y['miss']['evidence_dir']
    if 'replan_steps' in s:
        assert yaml.safe_load(pathlib.Path(r['matrix']).read_text())['arms'][0]['client_overrides']['replan_steps']==10
    # Server-owned stage flags are consumed before the plugin parses its options.
    args=['--os-method',s['method'],'--os-kwargs',json.dumps(s['kwargs']),'--os-cell',r['cell'],
          '--os-log-dir',str(run/'logs')]
    pargs=list(s['plugin_args'])
    if '--os-stage1-mode' in pargs:
        i=pargs.index('--os-stage1-mode');del pargs[i:i+2]
    if '--os-pack-prefix' in pargs:pargs.remove('--os-pack-prefix')
    _,rest=plugin.parse_cli(args+pargs);assert not rest,rest
    key=(s['method'],m.name)
    item=batch.setdefault(key,dict(method=s['method'],kwargs=s['kwargs'],family='k1_blind',cells=[],subsample=None,
                                  allow_gpu_fit=False))
    for arm in ('inf','cache'):
        cell=f"{s['model']}_{s['suite']}_{arm}"
        if cell not in item['cells']:item['cells'].append(cell)
(HERE/'batch.json').write_text(json.dumps(list(batch.values()),indent=2)+'\n')
report=dict(arms=len(rows),configs=len(emitted),batch_variants=len(batch),command=command,PASS=True)
(OUT/'arms_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
