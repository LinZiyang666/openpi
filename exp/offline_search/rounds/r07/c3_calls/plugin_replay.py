"""Real in-process plugin/orchestrator, recorded B-val inputs, fake CPU policy."""
import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

from .common import CELLS, SCRATCH, output_path, write_json, stall_path
from .recordings import streams


def replay(cell, variant, out):
    from exp.offline_search.closed_loop import plugin, selftest
    from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import STORE
    import openpi.cache.config as config
    from openpi.cache.orchestrator import CacheOrchestrator
    plugin._git_head = lambda: None  # no indirect git invocation
    model, suite, _ = cell.split('_')
    out = output_path(out)
    out.mkdir(parents=True, exist_ok=False)
    episodes = list(streams(cell, 'bval'))
    rows = [row for _, data in episodes for row in data]
    qc = NS(key_v0=np.array([r['vision_0'] for r in rows]), key_v1=np.array([r['vision_1'] for r in rows]),
            rs=np.array([r['robot_state'] for r in rows]), raw_state=np.array([r['raw_state'] for r in rows]))
    policy = np.array([r['policy_chunk'] for r in rows])
    method = ('exp.offline_search.rounds.r06.ideation_Q1.method_c.methods:CalibratedRescue'
              if variant == 'R6' else 'exp.offline_search.rounds.r07.c3_calls.methods:CallController')
    kwargs = dict(rho=.30, placement='uniform', stall_model_path=str(stall_path(cell)),
        calibration_path=str(SCRATCH / 'cal' / cell / ('CT' if variant == 'CT' else 'CU') / 'calibration.json'),
        random_seed=26092903, randomization_key='R6-C-v2/' + cell, cooldown_scope='stall')
    if variant != 'R6':
        kwargs['tilt'] = variant == 'CT'
    argv = ['--os-method', method, '--os-kwargs', json.dumps(kwargs), '--os-cell', cell.rsplit('_', 1)[0] + '_cache',
        '--os-root', str(STORE), '--os-log-dir', str(out), '--os-tag', 'r7_cpu_replay', '--os-no-shadow-native',
        '--os-blind', '--os-policy-tail', '--os-policy-tail-blocks', '1', '--os-judge', 'guard_only']
    opts, rest = plugin.parse_cli(argv)
    assert not rest
    rt = plugin.install(opts, model)
    assert rt.gpu is None and not rt.shadow_native
    cfg = config.load_cache_config(f'exp/trace_dual/config/tr_{model}_{"sp" if suite == "spatial" else suite}_cache.yaml')
    shared = config.build_shared_storage(cfg)
    class Fake(selftest.FakePolicy):
        stage1_calls = 0
        def stage1(self, obs):
            Fake.stage1_calls += 1
        def infer(self, obs):
            self.stage1(obs)
            return super().infer(obs)
        def _osp_prepare_blind(self, obs):
            return np.array(qc.rs[int(obs['_row'])]), None
        def _osp_blind_output(self, action, state):
            return {'actions': action.copy()}
    def factory(_base, bundle_id='default'):
        components = config.build_per_connection_components(cfg, shared, quiet=True)
        kb = selftest.FakeKB(qc)
        for session in plugin._TLS.new_sessions:
            session.kb = kb
        orch = CacheOrchestrator(storage=components['storage'], key_builder=kb, gates=components['gates'],
            judges=components['judges'], search_strategies=components['search_strategies'], timer=components['timer'],
            write_policy=components.get('write_policy'), offline_writers=components.get('offline_writers', ()),
            library_stats=components.get('library_stats'))
        return Fake(orch, kb, policy)
    conn = plugin._wrap_factory(factory)(None, '0')
    session = conn._osp_sessions[0]
    manifest = json.loads((STORE / 'library' / cell.rsplit('_', 1)[0] / 'current' / 'manifest.json').read_text())
    actions, vision, hit = [], [], []
    image = np.zeros((2, 2, 3), np.uint8)
    index = 0
    for episode, data in episodes:
        conn.on_episode_start(task=manifest['tasks'][str(episode.task_id)], episode_id=episode.init,
            extra_metadata=dict(task_uid=episode.uid, task_id=episode.task_id, orig_init_state_idx=episode.init, attempt=1))
        for step, row in enumerate(data):
            obs = {'observation/state': qc.raw_state[index], '_row': index, 'prompt': manifest['tasks'][str(episode.task_id)],
                   'observation/image': image, 'observation/wrist_image': image,
                   '__extra__': {'decision_id': step, 'executed_steps': session.method.block_controls}}
            action = np.asarray(conn.infer(obs)['actions'], np.float32)
            assert np.array_equal(action, session.b_aex.a[step])
            actions.append(action); vision.append(bool(session.has_vision[-1])); hit.append(bool(session.hits[-1]))
            index += 1
        conn.on_episode_end(success=False)  # terminal placeholder; no SR estimate
    decisions = [json.loads(line) for line in rt.dec_path.read_text().splitlines() if '"ev": "dec"' in line]
    assert len(decisions) == len(actions) and Fake.stage1_calls == sum(vision)
    keys = None
    if variant == 'CT':
        required = {'os_c3_weight', 'os_c3_lambda', 'os_c3_event_mass', 'os_c3_h', 'os_c3_h_dev',
                    'os_c3_dev_entry', 'os_c3_dev_latched', 'os_c3_unanimous', 'os_c3_unknown',
                    'os_c3_deviation', 'os_c3_p75'}
        for row in decisions:
            ex = row['extras']
            assert required <= ex.keys(), (row['step'], sorted(required - ex.keys()))
            assert len(ex) <= plugin.EXTRA_SCALARS_MAX_MIXED
            if row['vision']:
                assert ex['os_c_nominal_p'] == min(1., ex['os_c3_lambda'] * ex['os_c3_weight'])
                assert ex['os_c_call'] == float(ex['os_c_coin'] < ex['os_c_p'])
                assert {'os_force_miss', 'os_reason'} <= ex.keys()
        keys = sorted(set.intersection(*(set(row['extras']) for row in decisions)))
    np.savez_compressed(out / 'served.npz', action=actions, vision=vision, hit=hit,
                        source=[r['src'] for r in decisions])
    c1 = .152 if model == 'pi05' else .148
    report = dict(status='PASS', cell=cell, variant=variant, episodes=len(episodes), decisions=len(actions),
        anchors=sum(vision), calls=sum(not h for h in hit),
        IR=(c1 * sum(vision) + (1 - c1) * sum(not h for h in hit)) / len(actions),
        sources=dict(Counter(r['src'] for r in decisions)), logged_ct_keys=keys,
        no_server_no_model_no_simulator=True)
    write_json(out / 'report.json', report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', choices=CELLS, required=True)
    parser.add_argument('--variant', choices=['R6', 'CU', 'CT'], required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    replay(args.cell, args.variant, args.out or SCRATCH / 'plugin_checked' / f'{args.cell}_{args.variant}')


if __name__ == '__main__':
    main()
