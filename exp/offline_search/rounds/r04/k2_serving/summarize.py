from pathlib import Path
import hashlib
import json
import numpy as np
base=Path(__file__).resolve().parent
reports={p.parent.name:json.loads(p.read_text()) for p in sorted((base/'results').glob('final_*/selftest_report.json'))}
assert all(x['PASS'] for x in reports.values()), reports
summary={'checks':reports,'total_decisions':sum(r['decisions'] for r in reports.values()),
         'total_checks':len(reports), 'gpu':{},'legacy_bytes':{}}
for model in ('pi05','groot'):
    folder=base/'results'/f'gpu_{model}'
    replay=json.loads((folder/'replay.json').read_text())
    verify=json.loads((folder/'verify.json').read_text())
    assert verify['PASS'] and replay['offline_wire']['bit_equal'] == replay['offline_wire']['checked']
    assert replay['blind']['stage1_equals_vision'] and replay['blind']['s1_null_on_blind']
    summary['gpu'][model]={'final_replay':{k:replay[k] for k in ('n','episodes','blind','offline_wire','rs_max_abs')},
                           'normalized_replay':verify,'memory_observation':(folder/'gpu_memory.txt').read_text().strip()}
for mode in ('pure','mixed'):
    before=(base/'results'/f'parity_{mode}_before.jsonl').read_bytes()
    after=(base/'results'/f'parity_{mode}_after.jsonl').read_bytes()
    assert before==after
    summary['legacy_bytes'][mode]={'bytes':len(after),'lines':len(after.splitlines()),'bit_equal':True}
files=['blind.py','plugin.py','probe.py','selftest.py','verify_logs.py','replay_client.py']
summary['installed_sha256']={name:hashlib.sha256((base.parents[2]/'closed_loop'/name).read_bytes()).hexdigest() for name in files}
root=Path('/dev/shm/offline_search_store')
summary['libraries']={}
for model in ('pi05','groot'):
    for suite in ('spatial','l10'):
        for scale, lib in ((50,'current'),(500,'bpool_cs' if model=='pi05' else 'bpool_all')):
            path=root/'library'/f'{model}_{suite}'/lib
            meta=json.loads((path/'manifest.json').read_text())
            src=meta.get('sources',{}).get('pkl')
            summary['libraries'][f'{model}_{suite}_{scale}']={
                'rows':int(np.load(path/'action.npy',mmap_mode='r').shape[0]),
                'episodes':len(json.loads((path/'episodes.json').read_text())),
                'library':lib,'pkl':src,'pkl_bytes':Path(src).stat().st_size if src and Path(src).exists() else None}
(base/'results'/'validation_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in ('checks','installed_sha256','libraries','gpu')},indent=2))
for name,r in reports.items():
    mix=r.get('mixed',{})
    print(name,r['decisions'],r.get('vision','all'),r.get('blind',0),r.get('miss',mix.get('n_miss',0)),r['PASS'])
