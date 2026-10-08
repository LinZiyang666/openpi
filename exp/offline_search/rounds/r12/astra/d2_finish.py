"""Final deliverable/source boundary and append-only freeze audit."""
from .d2_screen import *

E=json.loads((D2/'evidence.json').read_text());V=json.loads((D2/'validation.json').read_text())
G=json.loads((D2/'document_generation.json').read_text());P=(HERE/'PREDICTION.md').read_bytes()
assert sha(D2/'evidence.json')==G['evidence_sha256']
assert sha(D2/'validation.json')==G['validation_sha256']
for name,digest in G['files'].items():assert sha(HERE/name)==digest
for prefix in (D2/'original_prediction.md',D2/'superseded_1822'/'PREDICTION.md',D2/'prediction_before_local_revision.md'):
    assert P.startswith(prefix.read_bytes()),str(prefix)
assert len({(r['root'],r['cell'],r['size']) for r in E['cells']})==len(E['cells'])
assert all(r['root'] in VALID_ROOTS for r in E['cells'])
assert all(r['root'] not in ('r12_devlooks_50S','r12_devlooks_200') for r in E['cells'])
assert set((r['root'],r['cell'],r['size']) for r in E['cells'])==set((r['root'],r['cell'],r['size']) for r in V['checks'])
assert sum(r['V'] for r in E['cells'])==V['looks']
assert sum(r['episodes'] for r in E['cells'])==V['episodes']
assert {'r12_devlooks_50L','r12_devlooks_400','r12_devlooks_knob'}<=set(r['root'] for r in E['cells'])
for r in E['cells']:
    for m in r['methods'].values():
        assert abs(m['saving_fixed_calls']-m['saving_ci']['point'])<1e-12
    assert 0<=r['methods']['consensus2_r0.1']['saving_fixed_calls']<=r['allhalf_ceiling']
for p in list(HERE.glob('d2_*.py'))+[HERE/'dev_validation.py',HERE/'dev_assess.py']:
    compile(p.read_text(),str(p),'exec')
allowed={10,11,12,13,14,15,16,17,18,19,20,21,54,55,56,57,58,59,60,61,62,63,64,65}
assert set(os.sched_getaffinity(0))<=allowed
gpu=json.loads((D2/'groot_mismatch'/'summary.json').read_text())
assert gpu['peak_reserved_mib']<10*1024
files=list(HERE.glob('d2_*.py'))+[HERE/x for x in ('REPORT.md','SPEC.md','OFFLINE.md','PREDICTION.md','EXPERIMENTS.md','HANDBACK.md','REQUESTS.md','dev_validation.py','dev_assess.py')]
files += [D2/x for x in ('evidence.json','validation.json','document_generation.json','source_correction.json','gpu_after.txt')]
for r in E['cells']:
    dest=D2/r['root']/f"{r['cell']}_{r['size']}"
    files += [dest/'candidates.npz',dest/'screen.json',dest/'consensus2_r0.1.npz',dest/'consensus2_r0.05.npz']
    for name in ('numeric_stress.json','camera_choice.json'):
        if (dest/name).exists():files.append(dest/name)
out=dict(timestamp=now(),checks_passed=True,episodes=V['episodes'],looks=V['looks'],roots=sorted(set(r['root'] for r in E['cells'])),
    pending=['r12_devlooks_200_local'],retired=['r12_devlooks_50S','r12_devlooks_200'],
    all_prior_prediction_sections_byte_preserved=True,source_correction_disclosed=True,
    changed_controller_SR_measured=False,goal_demonstrated=False,GPU_peak_reserved_MiB=gpu['peak_reserved_mib'],
    directive_sha256=sha(HERE.parent/'DIRECTIVE_2.md'),files={str(p.relative_to(HERE)):sha(p) for p in files},
    operational_note='Offline work only; no new-controller/test-A/server/worker/sync launch. Exact GR00T single-camera original-pixel and policy-prefix validation still required.')
dump(D2/'FINAL_MANIFEST.json',out)
print({k:v for k,v in out.items() if k!='files'},flush=True)
