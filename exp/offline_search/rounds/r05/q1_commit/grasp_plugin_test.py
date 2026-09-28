"""Eight real plugin connections; synthetic contact table stresses D1 execution.

The historical helper test separately verifies the real table/thresholds. Here
the successor aperture is deliberately set to one and the observed aperture to
zero so actual closed executed heads deterministically expose the lifecycle.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time

import numpy as np

from exp.offline_search.closed_loop import plugin, selftest
from exp.offline_search.harness import store
from exp.offline_search.rounds.r05.q1_commit.judge import GraspCheckJudge
import openpi.cache.config as cc
from openpi.cache.orchestrator import CacheOrchestrator


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    kw = dict(base_kwargs=dict(lib='current', kref=5, serving='anchor_tail', budget=1, gates='budget_only'),
              guards=False, ncal=64)
    opts, _ = plugin.parse_cli(['--os-method', 'exp.offline_search.rounds.r05.q1_commit.judge:GraspCheckJudge',
        '--os-kwargs', json.dumps(kw), '--os-cell', 'pi05_l10_cache', '--os-root',
        '/home/weiland/trace_runs/offline_search_store', '--os-log-dir', str(out), '--os-tag', 'grasp',
        '--os-blind', '--os-judge', 'guard_only', '--os-no-shadow-native', '--os-log-inputs'])
    rt = plugin.install(opts, 'pi05')
    n = len(rt.method.contact['next'])
    rt.method.contact = dict(width=np.ones(n, np.float32), pattern=np.full(n, 31, np.uint8),
                             next=np.arange(n, dtype=np.int32), lo=np.float32(-1), hi=np.float32(1))
    qc = store.QueryCell(opts.os_root, opts.os_cell)
    cfg = cc.load_cache_config('exp/trace_dual/config/tr_pi05_l10_cache.yaml')
    shared = cc.build_shared_storage(cfg)
    active = peak = 0
    lock = threading.Lock()

    class KB(selftest.FakeKB):
        def build(self, checkpoint_id):
            keys = super().build(checkpoint_id)
            keys['robot_state'][6], keys['robot_state'][7] = -1., 1.
            return keys

    class Fake(selftest.FakePolicy):
        def stage1(self, obs):
            nonlocal active, peak
            assert not rt.decision_lock._is_owned()
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(.015)
            with lock:
                active -= 1
        def infer(self, obs):
            self.stage1(obs)
            return super().infer(obs)
        def _osp_prepare_blind(self, obs):
            rs = np.array(qc.rs[obs['_row']], copy=True)
            rs[6], rs[7] = -1., 1.
            return rs, obs['observation/state']
        def _osp_blind_output(self, action, state):
            return {'actions': np.array(action, copy=True)}

    def factory(_base, bundle_id='default'):
        comps = cc.build_per_connection_components(cfg, shared, quiet=True)
        kb = KB(qc)
        for s in plugin._TLS.new_sessions:
            s.kb = kb
        orch = CacheOrchestrator(storage=comps['storage'], key_builder=kb, gates=comps['gates'],
                judges=comps['judges'], search_strategies=comps['search_strategies'], timer=comps['timer'],
                write_policy=comps.get('write_policy'), offline_writers=comps.get('offline_writers', ()),
                library_stats=comps.get('library_stats'))
        return Fake(orch, kb, np.asarray(qc.a_inf))

    conns = [plugin._wrap_factory(factory)(None, str(i)) for i in range(8)]
    e = qc.episodes[0]
    barrier = threading.Barrier(8)

    def obs(step):
        row = min(e['start'] + step, e['end'] - 1)
        return {'observation/state': np.asarray(qc.raw_state[row], np.float64), 'prompt': e['task'],
                '_row': row, '__extra__': {'decision_id': step, 'executed_steps': 5}}

    def drive(i):
        c, counts = conns[i], []
        barrier.wait()
        # First episode continues after intervention; second ends exactly on it.
        for episode in range(2):
            c.on_episode_start(task=e['task'], episode_id=e['init'], extra_metadata={
                'task_uid': f'grasp:{i}', 'task_id': e['task_id'], 'orig_init_state_idx': e['init']})
            s = c._osp_sessions[0]
            alarm_steps = []
            for step in range(80):
                result = c.infer(obs(step))
                assert s.step == step + 1 == s.b_aex.n == len(s.has_vision) == len(s.hits)
                if step == 0:
                    assert s.has_vision[0] and not s.method._grasp_used and s.method._grasp_pending is None
                if s.method._grasp_issued == step:
                    alarm_steps.append(step)
                    assert result['hit_type'] == 'MISS' and s.has_vision[-1] and s.hits[-1] == 0
                    assert s._look_reason == GraspCheckJudge.LOOK_CODE
                    assert not s.method._grasp_used  # consumes only on later committed history
                    snapshot = (s.step, s.b_aex.n, len(s.has_vision), list(s.hits), s.method._grasp_issued)
                    try:
                        c.infer(obs(step))
                    except ValueError as exc:
                        assert 'duplicate decision_id' in str(exc)
                    else:
                        raise AssertionError('duplicate D1 decision accepted')
                    assert snapshot == (s.step, s.b_aex.n, len(s.has_vision), list(s.hits), s.method._grasp_issued)
                    if episode:
                        break
                if alarm_steps and step > alarm_steps[0]:
                    assert s.method._grasp_used
            assert len(alarm_steps) == 1, (i, episode, alarm_steps)
            counts.append(alarm_steps[0])
            c.on_episode_end(success=False)
        # Same external identity and terminal forced MISS must not cross reset.
        c.on_episode_start(task=e['task'], episode_id=e['init'], extra_metadata={
            'task_uid': f'grasp:{i}', 'task_id': e['task_id'], 'orig_init_state_idx': e['init']})
        c.infer(obs(0))
        assert not c._osp_sessions[0].method._grasp_used and c._osp_sessions[0].method._grasp_issued is None
        c.on_episode_end(success=False)
        return counts

    with ThreadPoolExecutor(max_workers=8) as pool:
        counts = list(pool.map(drive, range(8)))
    rows = [json.loads(line) for line in rt.dec_path.read_text().splitlines()]
    dec = [r for r in rows if r['ev'] == 'dec']
    alarms = [r for r in dec if r['extras'].get('grasp_check') == 1]
    assert len(alarms) == 16
    assert all(r['vision'] and not r['hit'] and r['extras']['os_reason'] == GraspCheckJudge.MISS_CODE for r in alarms)
    assert peak >= 2
    report = dict(PASS=True, synthetic_contact_table=True, connections=8, episodes=24, interventions=16,
                  duplicate_rejections=16, decisions=len(dec), peak_stage1=peak, alarm_steps=counts,
                  costs_per_intervention=.152+.848)
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args().out)
