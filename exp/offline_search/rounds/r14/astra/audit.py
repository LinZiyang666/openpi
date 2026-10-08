"""Review the final frozen design and input/output boundary without A access."""
from common import *
fence()
import numpy as np
import ast

freeze=read(HERE/'freeze.json');p=freeze['plan'];errors=[]
for name,value in freeze['files'].items():
    if sha(HERE/name)!=value:errors.append('changed frozen file '+name)
for path in HERE.glob('*.py'):ast.parse(path.read_text())
test=list(csv.DictReader((HERE/'test_grid.csv').open()));pilot=list(csv.DictReader((HERE/'pilot_grid.csv').open()))
assert len(test)==p['test_arms'] and sum(int(r['episodes']) for r in test)==p['test_episodes']
assert len(pilot)==p['pilot_arms'] and sum(int(r['episodes']) for r in pilot)==p['pilot_episodes']
assert len({r['arm'] for r in test})==len(test) and len({r['arm'] for r in pilot})==len(pilot)
for grid in (test,pilot):
    for cell in CELLS:
        hosts={r['host'] for r in grid if r['cell']==cell}
        assert not hosts or hosts=={'local_4090' if cell.startswith('groot') else 'h100'}
for cell in CELLS:
    assert sum(r['cell']==cell and r['part']=='A' for r in test)==len(p['variants'])
    for dose in p['doses']:
        for rep in range(p['B_replicates']):
            rr=[r for r in test if r['cell']==cell and r['part']=='B' and float(r['target'])==dose and int(r['replicate'])==rep]
            assert {r['variant'] for r in rr}=={'periodic','task_periodic'}
            assert len({r['rollout_seed'] for r in rr})==1 and {r['coin_seed'] for r in rr}=={'0'}
            assert abs(int(rr[0]['order'])-int(rr[1]['order']))==1
selection=read(HERE/'pilot_selection.json')
for s in selection['cells']:
    assert not s['missing'] and len(s['pairs'])==50
    assert len({tuple(pair) for pair in s['pairs']})==50
    for t,i in s['pairs']:assert i not in s['excluded_inits_by_task'][str(t)]
fits=read(HERE/'allocation_fits.json')
for f in fits:
    assert len(f['K'])==len(f['priority']['z'])==10
    assert np.isfinite(f['K']).all() and min(f['K'])>=0
    from allocation import settings
    np.testing.assert_array_equal(f['K'],settings(f['reference_K'],f['priority']['z'],f['bias']))
    assert f['gain']==p['allocation_gain'] and f['prior_episodes']==p['allocation_prior_episodes']
    assert len(f['reference_irs'])==len(f['calibration_sources'])
for cell in CELLS:
    a=read(HERE/f'partition_{cell}_audit.json')
    assert len(a['folds'])==5
    for f in a['folds']:
        tr,va=set(f['train_episodes']),set(f['heldout_episodes'])
        assert not tr&va
        inner=f['inner'];it=set(inner['metric_episodes']);iv=set(inner['heldout_episodes'])
        assert not it&iv and it|iv==tr
    rows=list(csv.DictReader((HERE/f'partition_{cell}.csv').open()))
    for name in ('base','scope','metric','head','scope_metric','scope_only','all_pool'):
        rr=[r for r in rows if r['variant']==name]
        assert len(rr)==int(cell.split('_')[-1]) and len({r['ep'] for r in rr})==len(rr)
        assert all(np.isfinite(float(r['mse_corrected'])) and np.isfinite(float(r['mse_blend05'])) for r in rr)
paths=set()
for q in HERE.glob('*_reads.json'):
    paths.update(read(q)['paths'])
for name in paths:
    q=Path(name)
    assert not ('r13' in q.parts and q.name.startswith('validation_selection'))
    if RUNS in q.parents:assert q.relative_to(RUNS).parts[0] in ROOTS
    if STORE in q.parents:assert STORE/'library' in q.parents
assert freeze['affinity']==list(range(10,22))+list(range(54,66))
assert freeze['cuda_visible_devices']==''
assert not errors,errors
dump(HERE/'audit.json',dict(timestamp=now(),frozen_files_verified=len(freeze['files']),
    code_files=len(list(HERE.glob('*.py'))),read_paths=len(paths),test_arms=len(test),pilot_arms=len(pilot),
    all_episode_folds_disjoint=True,all_task_caps_recompute_exactly=True,host_pinning=True,
    paired_B_arms_adjacent=True,reserved_identity_file_unread=True,read_fence_pass=True,
    source_freeze_sha256=sha(HERE/'freeze.json'),errors=errors))
print((HERE/'audit.json').read_text())
