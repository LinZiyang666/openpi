from common import *
fence()
import numpy as np

rows = []
samples = {}
for root in ROOTS:
    p = RUNS / root / 'arms.json'
    if not p.exists(): continue
    for a in read(p):
        assert a['dev'] is True and a['init_pool'] == 'B'
        arm = RUNS / root / 'runs' / a['arm']
        summary = read(arm/'summary.json') if (arm/'summary.json').exists() else {}
        manifests = sorted(arm.glob('server_*/looks/manifest_*.json'))
        looks = shadows = 0
        for mp in manifests:
            mm = read(mp)
            assert mm['dev'] and mm['init_pool'] == 'B'
            looks += len(mm['shards'])
            if root not in samples and mm['shards']:
                shard = mp.parent / mm['shards'][0]['file']
                with np.load(shard, allow_pickle=False) as z:
                    samples[root] = dict(manifest_keys=list(mm), options=mm.get('options'),
                        fields={k:list(z[k].shape) for k in z.files},
                        guard_state=json.loads(str(z['guard_state'])),
                        meta=json.loads(str(z['meta'])),
                        distances=z['distances'].tolist())
            shadows += int(bool(mm.get('options', {}).get('shadow')))*len(mm['shards'])
        led = summary.get('cost_ledger', {})
        rows.append(dict(root=root, arm=a['arm'], model=a['model'], suite=a.get('suite_short'),
            size=a.get('r10_size', 'current'), method=a.get('r11_method', 'off'), target=a.get('target_ir'),
            complete=summary.get('complete', 0), expected=summary.get('expected'), success=summary.get('success'),
            N=led.get('decisions'), V=led.get('vision_decisions'), M=led.get('misses'),
            looks=looks, shadows=shadows, log_files=len(list(arm.glob('server_*/decisions_*.jsonl'))),
            fit_path=a.get('plugin_args', [])))
csvout(HERE/'inventory.csv', [{k:v for k,v in r.items() if k != 'fit_path'} for r in rows])
dump(HERE/'inventory.json', rows)
dump(HERE/'recording_schema.json', samples)
for cell in CELLS:
    p = R11/'astra/data'/cell/'signals.npz'
    with np.load(p, allow_pickle=False) as z:
        samples[cell] = {k:list(z[k].shape) for k in z.files}
dump(HERE/'signal_schema.json', {k:v for k,v in samples.items() if k in CELLS})
provenance('inventory')
print(table(['root', 'arms', 'completed episodes', 'looks'], [[r, sum(x['root']==r for x in rows),
    sum(x['complete'] for x in rows if x['root']==r), sum(x['looks'] for x in rows if x['root']==r)] for r in ROOTS]))
print(json.dumps(next(iter(samples.values())), indent=2)[:9000])
