import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from exp.offline_search.rounds.r17.autoscale_1017 import acceptance, formal_trial, kernel_evidence, release
from exp.offline_search.rounds.r17.autoscale_1017.node import atomic
from exp.offline_search.rounds.r17.autoscale_1017.policy import GIB, Tuner


def test_formal_maxima_change_only_resource_fields():
    original = dict(run_name='fa_co_r_f_l', cache_plan_name='frozen-science', domain='libero',
                    seed=7, init_indices=[0, 49], replicas=8, batch=1, ports=[11, 12],
                    worker_nodes=dict(timan1=35, timan107=25), workers=60,
                    cache_study='original/STUDY.json', filter_failed_knob=dict(arm='filtered'),
                    worker_max_episodes=4)
    saved = copy.deepcopy(original)
    maxima = formal_trial.counts('timan1=150,timan107=100')
    c = formal_trial.configure(original, maxima)
    assert original == saved and c['autoscale_maximum'] == 254
    assert c['execution_worker_nodes'] == dict(timan1=150,timan107=104) and Tuner(c['workers']).target == 254
    for key in ('cache_plan_name', 'seed', 'init_indices', 'replicas', 'batch', 'ports', 'cache_study', 'filter_failed_knob'):
        assert c[key] == original[key]
    mw = formal_trial.configure(dict(original, domain='metaworld', worker_max_episodes=0), dict(timan1=200, timan107=150))
    assert mw['workers'] == 772 and mw['worker_max_episodes'] == 32 and mw['worker_max_seconds'] == 1800
    with pytest.raises(ValueError):formal_trial.counts('timan108=1')


def test_overlay_changes_only_listed_orchestration_files(tmp_path):
    from exp.r17_integration.ops.build_client_bundle import build_client_bundle, verified_members
    checkout = tmp_path / 'checkout'
    for name in release.worker_files():
        p = checkout / name;p.parent.mkdir(parents=True, exist_ok=True);p.write_text('# new orchestration\n')
    scientific = ['exp/metaworld/episode_runner.py', 'examples/libero/worker_entry.py']
    for name in scientific:
        p = checkout / name;p.parent.mkdir(parents=True, exist_ok=True);p.write_text('# frozen science\n')
    original = tmp_path / 'old.tar'
    build_client_bundle(checkout, scientific + ['src/openpi/conductor/worker.py'], original)
    (checkout / scientific[0]).write_text('# checkout science must not enter frozen release\n')
    (checkout / 'src/openpi/conductor/worker.py').write_text('# new observer hook\n')
    new = tmp_path / 'new.tar'
    proof = release.overlay_worker(original, new, checkout)
    _, content = verified_members(new, 'PAYLOAD.json')
    assert all(content[n] == b'# frozen science\n' for n in scientific)
    assert set(proof['ops_files']) == set(release.worker_files())
    assert content['src/openpi/conductor/worker.py'] == b'# new observer hook\n'


def test_post_apply_capabilities_available():
    release.require_patch(formal_trial.REPO)


def evidence(root):
    atomic(root / 'campaign.json', dict(autoscale_node_maxima=dict(timan107=150), autoscale=dict(critical=10 * GIB)))
    atomic(root / 'EPISODE_PLAN.json', dict(tasks=[dict(task_uid='a'), dict(task_uid='b')]))
    folder = root / 'workers/timan107';folder.mkdir(parents=True)
    (root / 'driver').mkdir()
    journal = [dict(accepted=True, task_uid=u, error=None) for u in ('a', 'b')]
    (root / 'driver/journal.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in journal))
    windows = [dict(time=t, node='global', delta=0, reason='hold', counts=dict(active=44),
                    signals=dict(rate=86 / 60, settled=True, comparable=True, calibrated=True, window_s=180, window_end=t))
               for t in range(180, 1081, 180)]
    windows[0]['reason'] = 'server_queue'
    (root / 'AUTOSCALE_DECISIONS.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in windows))
    sample = dict(time=1, node='timan107', delta=0, reason='node_sample', signals=dict(available=22 * GIB, minimum_available=20 * GIB), counts=dict(active=100))
    (folder / 'AUTOSCALE_SAMPLES.jsonl').write_text(json.dumps(sample) + '\n')
    event = dict(time=1, node='timan107', delta=0, reason='memory_growth', signals=dict(available=20 * GIB), counts=dict(active=100, target=150))
    (folder / 'AUTOSCALE_DECISIONS.jsonl').write_text(json.dumps(event) + '\n')
    atomic(folder / 'AUTOSCALE_KERNEL.json', dict(status='PASS', oom_lines=[], source='journal', coverage=True))
    return folder


def test_acceptance_enforces_own_evidence_duplicates_critical_oom_and_rate(tmp_path, monkeypatch):
    monkeypatch.setattr('exp.offline_search.rounds.r17.autoscale_1017.compare.wait_io', lambda: None)
    node = evidence(tmp_path)
    report = acceptance.audit(tmp_path, 86, case='server')
    assert report['status'] == 'PASS' and report['threshold_per_min'] == 81.7
    assert (tmp_path / 'AUTOSCALE_ACTIVE_CURVE.csv').exists()
    assert acceptance.audit(tmp_path, 86, case='memory')['checks']['scaling_exercised']
    assert not acceptance.audit(tmp_path, 100, case='server')['checks']['throughput']
    atomic(node / 'AUTOSCALE_KERNEL.json', dict(status='UNKNOWN'))
    assert not acceptance.audit(tmp_path, 86, case='server')['checks']['zero_oom']
    atomic(node / 'AUTOSCALE_KERNEL.json', dict(status='FAIL', oom_lines=['Killed process']))
    assert not acceptance.audit(tmp_path, 86, case='server')['checks']['zero_oom']
    atomic(node / 'AUTOSCALE_NODE.json', dict(minimum_available=9 * GIB))
    assert not acceptance.audit(tmp_path, 86, case='server')['checks']['above_critical']
    with (tmp_path / 'driver/journal.jsonl').open('a') as out:out.write(json.dumps(dict(accepted=True, task_uid='a', error='bad')) + '\n')
    checks = acceptance.audit(tmp_path, 86, case='server')['checks']
    assert not checks['every_uid_once'] and not checks['accepted_errors_zero']


def test_kernel_collection_once_at_end_and_unknown_is_not_zero_oom(tmp_path):
    calls = []
    def execute(argv, **kw):
        calls.append(argv)
        if argv[0]=='dmesg':return SimpleNamespace(returncode=1,stderr='permission denied',stdout='')
        return SimpleNamespace(returncode=0, stderr='', stdout=json.dumps(dict(__REALTIME_TIMESTAMP='150000000', MESSAGE='Out of memory: Killed process 123 worker')))
    result = kernel_evidence.collect(tmp_path / 'one', 100, now=lambda: 200, execute=execute)
    assert sum(c[0]=='journalctl' for c in calls)==1 and len(calls)<=2 and result['status'] == 'FAIL' and len(result['oom_lines']) == 1
    calls.clear()
    def denied(argv, **kw):
        calls.append(argv)
        return SimpleNamespace(returncode=1, stderr='permission denied', stdout='')
    result = kernel_evidence.collect(tmp_path / 'two', 100, now=lambda: 200, execute=denied)
    assert len(calls) == 2 and result['status'] == 'UNKNOWN'


def test_acceptance_separates_scheduled_probe_windows_without_rate_cherry_picking(tmp_path, monkeypatch):
    monkeypatch.setattr('exp.offline_search.rounds.r17.autoscale_1017.compare.wait_io',lambda:None)
    evidence(tmp_path)
    path=tmp_path/'AUTOSCALE_DECISIONS.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    probe=dict(time=1260,node='global',delta=0,reason='marginal_plateau',counts=dict(active=48),
        signals=dict(rate=10/60,settled=True,comparable=True,calibrated=True,probing=True,window_s=180,window_end=1260))
    rows.append(probe)
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert acceptance.audit(tmp_path,86,case='server')['checks']['throughput']
    probe['signals']['probing']=False
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert not acceptance.audit(tmp_path,86,case='server')['checks']['throughput']


def test_shared_patch_defaults_and_dynamic_probe_set():
    base = json.loads((Path(__file__).parents[1]/'PATCH_BASE.json').read_text())
    after = {p:Path(p).read_text() for p in base}
    r16 = after['exp/offline_search/rounds/r16/conductor.py']
    assert "'--workers',type=int,default=None" in r16
    assert "'--workers',type=int,default=8" not in r16
    assert "worker['autoscale_initial'] = count" in after['exp/r17_integration/ops/formal_eval.py']
    assert 'autoscale_node_maxima' in after['exp/offline_search/rounds/r17/filter_failed/worker_resilience.py']
    assert 'collect(run, autoscaler.started_wall)' in after['exp/r17_integration/ops/formal_workers.py']


def test_formal_prepare_uses_new_worker_pin_preserves_plan_and_native_server(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from exp.r17_integration.ops import formal_eval as f, driver_release, immutable_release, formal_runtime, cache_control, formal_ports, formal_prestage
    from exp.offline_search.rounds.r17.filter_failed import stage, timan1_gpu2, worker_resilience
    from exp.r17_integration.ops.build_client_bundle import build_client_bundle, sha256
    root, source, checkout = (tmp_path / n for n in ('campaign', 'source', 'checkout'))
    root.mkdir();source.mkdir()
    for name in release.worker_files():
        p = checkout / name;p.parent.mkdir(parents=True, exist_ok=True);p.write_text('# orchestration\n')
    build_client_bundle(checkout, ['src/openpi/conductor/worker.py'], source / 'workers.tar')
    for folder in (root, source):(folder / 'runtime.tar').write_bytes(b'frozen server')
    atomic(source / 'WORKER_DEPLOYMENT.json', {})
    tasks = [dict(task_uid='native-uid', orig_init_state_idx=49, extra=dict(seed=7))]
    atomic(root / 'EPISODE_PLAN.json', dict(tasks=tasks))
    old_plan = (root / 'EPISODE_PLAN.json').read_bytes()
    atomic(root / 'campaign.json', dict(run_name='formal', domain='metaworld', filter_failed_source=str(source),
        runtime=str(tmp_path / 'immutable-server'), runtime_identity='original', replicas=4, ports=[100]))
    monkeypatch.setattr(formal_trial, 'REPO', checkout)
    monkeypatch.setattr(timan1_gpu2, 'coordinator', nullcontext)
    monkeypatch.setattr(worker_resilience, 'coordinator', nullcontext)
    monkeypatch.setattr(f, 'retry_scope', lambda *a: nullcontext())
    monkeypatch.setattr(formal_ports, 'prepare_admission', lambda *a: None)
    monkeypatch.setattr(immutable_release, 'install', lambda *a: None)
    monkeypatch.setattr(formal_runtime, 'attach', lambda *a: None)
    monkeypatch.setattr(cache_control, 'attach', lambda *a: None)
    monkeypatch.setattr(driver_release, 'prepare', lambda *a: None)
    monkeypatch.setattr(driver_release, 'planned_tasks', lambda *a: copy.deepcopy(tasks))
    monkeypatch.setattr('exp.offline_search.rounds.r17.autoscale_1017.compare.wait_io', lambda: None)
    seen = []
    def workers(c, directory, *, payload_override):
        seen.append((copy.deepcopy(c), payload_override))
        assert sha256(payload_override) != sha256(source / 'workers.tar')
        return {n: {} for n in c['execution_worker_nodes']}
    monkeypatch.setattr(f, 'prepare_workers', workers)
    monkeypatch.setattr(f, 'remote', lambda *a, **k: pytest.fail('No remote in CPU test'))
    monkeypatch.setattr(stage, 'stage_server_inputs', lambda *a: None)
    monkeypatch.setattr(stage, 'stage_worker_inputs', lambda *a: None)
    monkeypatch.setattr(formal_prestage, 'seal_prepared', lambda c, r, d: atomic(r / 'PREPARED.json', dict(status='PASS', nodes=list(d))))
    formal_trial.prepare(root, dict(timan1=200, timan107=150))
    assert (root / 'EPISODE_PLAN.json').read_bytes() == old_plan
    assert (root / 'runtime.tar').read_bytes() == b'frozen server'
    assert seen[0][0]['workers'] == 772 and seen[0][0]['worker_max_episodes'] == 32
    assert seen[0][0]['worker_vram_mib'] == timan1_gpu2.SIZES['metaworld'][0]
    # A sealed campaign can never be silently retargeted.
    with pytest.raises(ValueError, match='newly rebuilt'):
        formal_trial.prepare(root, dict(timan1=100))
    # Any unlaunched prepared R17 campaign, including one with no ff source,
    # can explicitly reseal only orchestration while retaining server bytes.
    c = json.loads((root/'campaign.json').read_text());c.pop('filter_failed_source')
    atomic(root/'campaign.json', c)
    (root/'workers.tar').write_bytes((source/'workers.tar').read_bytes())
    formal_trial.prepare(root, dict(timan1=100), reprepare=True)
    assert list(root.glob('PREPARED.before_autoscale.*.json'))
    assert (root/'EPISODE_PLAN.json').read_bytes() == old_plan
    assert (root/'runtime.tar').read_bytes() == b'frozen server'
    atomic(root/'LAUNCH.json', {})
    with pytest.raises(ValueError, match='launched'):
        formal_trial.prepare(root, dict(timan1=100), reprepare=True)


def test_zero_oom_requires_log_retention_covering_run_start(tmp_path):
    calls = []
    def execute(argv, **kw):
        calls.append(argv)
        if argv[0]=='dmesg':return SimpleNamespace(returncode=1,stderr='permission denied',stdout='')
        stamp = '50000000' if '-n' in argv else '150000000'
        return SimpleNamespace(returncode=0, stderr='', stdout=json.dumps(dict(__REALTIME_TIMESTAMP=stamp, MESSAGE='ordinary kernel event')))
    result = kernel_evidence.collect(tmp_path / 'covered', 100, now=lambda: 200, execute=execute)
    assert result['status'] == 'PASS' and result['coverage'] and sum(c[0]=='journalctl' for c in calls)==2 and len(calls)<=3
    def rotated(argv, **kw):
        return SimpleNamespace(returncode=0, stderr='', stdout=json.dumps(dict(__REALTIME_TIMESTAMP='150000000', MESSAGE='ordinary kernel event')) if '-n' not in argv else '')
    result = kernel_evidence.collect(tmp_path / 'rotated', 100, now=lambda: 200, execute=rotated)
    assert result['status'] == 'UNKNOWN'


def test_waiting_formal_queue_drops_dead_node_before_native_topology(tmp_path, monkeypatch):
    from exp.offline_search.rounds.r17.filter_failed_knob import formal_a_queue as q
    monkeypatch.setattr(formal_trial, 'EVAL', tmp_path)
    monkeypatch.setattr(q, 'INDEX', tmp_path / 'queue.json')
    unit = dict(name='fa_co_r_f_l', status='WAITING_OFF', model='cosmos', mode='reduced', arm='filtered',
                workers=dict(timan1=35, timan107=25, timan108=25))
    atomic(q.INDEX, dict(units=[unit]))
    seen = []
    def materialize(u, keep, levels):
        seen.append(u)
        root = tmp_path / u['name'];atomic(root / 'campaign.json', {})
        assert keep == [('libero_10', 50)] and levels == ['k1']
        return dict(u, jobs=[dict(run=str(root))])
    monkeypatch.setattr(q, 'materialize', materialize)
    monkeypatch.setattr(q, 'build_levels', lambda *a: q.materialize(copy.deepcopy(unit), [('libero_10', 50)], ['k1']))
    root = formal_trial.resolve('fa_co_r_f_l', 'prepare', dict(timan1=150, timan107=100))
    assert root == tmp_path / 'fa_co_r_f_l'
    assert seen[0]['workers'] == dict(timan1=150, timan107=100)
    assert unit['workers']['timan108'] == 25  # original input object untouched


@pytest.mark.parametrize('window',['','-- No entries --\n'])
@pytest.mark.parametrize('coverage',['covered','missing','denied'])
def test_empty_kernel_window_needs_retention_anchor_not_a_new_message(tmp_path,window,coverage):
    calls=[]
    def execute(argv,**kw):
        calls.append(argv)
        if argv[0]=='dmesg':return SimpleNamespace(returncode=1,stdout='',stderr='permission denied')
        if '--since' in argv:return SimpleNamespace(returncode=0,stdout=window,stderr='')
        return SimpleNamespace(returncode=0,
            stdout='' if coverage=='missing' else json.dumps(dict(__REALTIME_TIMESTAMP='90000000',MESSAGE='previous kernel entry')),
            stderr='not seeing messages from other users' if coverage=='denied' else '')
    result=kernel_evidence.collect(tmp_path,100,now=lambda:200,execute=execute)
    assert result['status']==('PASS' if coverage=='covered' else 'UNKNOWN')
    assert len(calls)<=3 and sum('--since' in c for c in calls)==1
    if coverage=='covered':assert result['coverage'] is True and result['oom_lines']==[]
