"""CPU in-process plugin replay, recorded observations and recorded policy chunks.

No simulator, network server, policy model, or closed-loop episode is run.
Observations never respond to served actions: this cannot estimate success rate.
"""
from __future__ import annotations
import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/weiland/trace_runs/offline_search_store')
REPO = HERE.parents[6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--cases', type=Path, default=HERE/'replay_cases.json')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if not args.out.resolve().is_relative_to(HERE):
        raise ValueError('replay output must remain under adapters/')
    args.out.mkdir(parents=True, exist_ok=False)
    case = next(r for r in json.loads(args.cases.read_text()) if r['name'] == args.case)
    from exp.offline_search.closed_loop import plugin, selftest
    from exp.offline_search.closed_loop.blind import policy_tail_chunk
    from exp.offline_search.harness import store
    import openpi.cache.config as cc
    from openpi.cache.orchestrator import CacheOrchestrator
    # Honor the owner's no-git rule, including indirect startup provenance reads.
    plugin._git_head = lambda: None
    spec = case['spec']
    argv = ['--os-method',spec['method'],'--os-kwargs',json.dumps(spec['kwargs']),
            '--os-cell',f'{spec["model"]}_{spec["suite"]}_cache', '--os-root',str(ROOT),
            '--os-log-dir',str(args.out),'--os-tag','replay','--os-no-shadow-native','--os-blind',
            '--os-fit-artifact',case['artifact']]
    if case['judge']:
        argv += ['--os-judge',case['judge']]
    if case['policy_tail']:
        argv += ['--os-policy-tail','--os-policy-tail-blocks','1']
    opts, rest = plugin.parse_cli(argv)
    assert not rest
    if not Path(case['artifact']).exists():
        raise ValueError('replay must use a prebuilt fit, never fit into a run/output directory')
    t0 = time.monotonic()
    rt = plugin.install(opts, spec['model'])
    assert rt.shadow_native is False and rt.randomized is False and rt.gpu is None
    short = 'sp' if spec['suite'] == 'spatial' else 'l10'
    cfg = cc.load_cache_config(str(REPO/f'exp/trace_dual/config/tr_{spec["model"]}_{short}_cache.yaml'))
    shared = cc.build_shared_storage(cfg)
    qc = store.QueryCell(ROOT, f'{spec["model"]}_{spec["suite"]}_cache')

    class Fake(selftest.FakePolicy):
        calls = 0
        def stage1(self, obs):
            Fake.calls += 1
        def infer(self, obs):
            self.stage1(obs)
            return super().infer(obs)
        def _osp_prepare_blind(self, obs):
            return np.array(qc.rs[int(obs['_row'])], copy=True), None
        def _osp_blind_output(self, action, state):
            return {'actions':action.copy()}

    def factory(_base, bundle_id='default'):
        comps = cc.build_per_connection_components(cfg, shared, quiet=True)
        kb = selftest.FakeKB(qc)
        for s in plugin._TLS.new_sessions:
            s.kb = kb
        orch = CacheOrchestrator(storage=comps['storage'], key_builder=kb, gates=comps['gates'],
            judges=comps['judges'], search_strategies=comps['search_strategies'], timer=comps['timer'],
            write_policy=comps.get('write_policy'), offline_writers=comps.get('offline_writers',()),
            library_stats=comps.get('library_stats'))
        return Fake(orch, kb, np.asarray(qc.a_inf, np.float32))

    conn = plugin._wrap_factory(factory)(None,'0')
    session = conn._osp_sessions[0]
    selected = [i for i,e in enumerate(qc.episodes) if int(e['init']) in (0,49)]
    assert len(selected) == 20 and len({qc.episodes[i]['task_id'] for i in selected}) == 10
    if case.get('reverse'):
        selected.reverse()
    rows, eps, steps, acts, visions, hits = [], [], [], [], [], []
    img = np.zeros((2,2,3), np.uint8)
    for ei in selected:
        e = qc.episodes[ei]
        uid = ('renamed/' if case.get('reverse') else '')+e['uid']
        conn.on_episode_start(task=e['task'], episode_id=e['init'], extra_metadata=dict(
            task_uid=uid, task_id=e['task_id'], orig_init_state_idx=e['init'], attempt=1))
        for step, r in enumerate(range(e['start'],e['end'])):
            obs = {'observation/state':np.asarray(qc.raw_state[r],np.float64), 'prompt':e['task'], '_row':r,
                   'observation/image':img,'observation/wrist_image':img,
                   '__extra__':{'decision_id':step,'executed_steps':5}}
            action = np.asarray(conn.infer(obs)['actions'],np.float32)
            assert action.shape == (rt.H,32)
            assert np.array_equal(session.b_aex.a[step],action)
            rows.append(r);eps.append(ei);steps.append(step);acts.append(action)
            visions.append(bool(session.has_vision[-1]));hits.append(bool(session.hits[-1]))
        conn.on_episode_end(success=False)  # no recorded terminal label is used or estimated
    decs = [json.loads(line) for line in rt.dec_path.read_text().splitlines() if '"ev": "dec"' in line]
    assert len(decs) == len(acts)
    sources = [r['src'] for r in decs]
    tails = 0
    for j,r in enumerate(decs):
        if steps[j] == 0:
            assert visions[j]
        if not visions[j]:
            assert j > 0 and visions[j-1] and eps[j-1] == eps[j]
            assert r['miss_k'] is None and r['s1_ms'] is None and r['s23_ms'] is None
        if not hits[j]:
            assert visions[j] and sources[j] == 'policy'
            assert np.array_equal(acts[j],qc.a_inf[rows[j]])
            if case['policy_tail'] and j+1 < len(acts) and eps[j+1] == eps[j]:
                assert sources[j+1] == 'policy_tail'
                assert np.array_equal(acts[j+1],policy_tail_chunk(acts[j],5))
                tails += 1
                if j+2 < len(acts) and eps[j+2] == eps[j]:
                    assert visions[j+2]
        if 'os_q2_p_call' in r.get('extras',{}):
            ex = r['extras']
            if 'os_q2_episode_dose' in ex:
                assert 0 < ex['os_q2_episode_propensity'] <= 1
            if ex.get('os_q2_base_miss'):
                assert not hits[j] and ex['os_q2_extra'] == 0
            else:
                assert (ex['os_q2_coin'] < ex['os_q2_dose']) == (not hits[j])
    assert Fake.calls == sum(visions)
    act_array = np.stack(acts)
    np.savez(args.out/'served.npz', row=rows,ep=eps,step=steps,action=act_array,vision=visions,hit=hits,source=sources)
    eligible = [r['extras'] for r in decs if r.get('vision') and r.get('extras',{}).get('os_q2_eligible')]
    nominal = sum(r['os_q2_dose'] for r in eligible)
    variance = sum(r['os_q2_dose']*(1-r['os_q2_dose']) for r in eligible)
    extra = sum(r['os_q2_extra'] for r in eligible)
    report = dict(case=args.case, cell=case['cell'], kind=case['kind'], episodes=len(selected), decisions=len(acts),
        anchors=sum(visions), misses=sum(not h for h in hits), policy_tails=tails, stage1_calls=Fake.calls,
        eligible_anchors=len(eligible), extra_calls=extra, conditional_expected_extra_calls=nominal,
        conditional_variance=variance, rate_z=(extra-nominal)/np.sqrt(variance) if variance else None,
        realized_extra_rate=extra/len(eligible) if eligible else None,
        nominal_average_rate=nominal/len(eligible) if eligible else None,
        source_counts=dict(collections.Counter(sources)), served_sha256=hashlib.sha256(act_array.tobytes()).hexdigest(),
        seconds=time.monotonic()-t0, assertions='PASS', no_policy_model=True, no_simulator=True,
        scope='fixed-observation plugin replay; no SR/counterfactual trajectory claim')
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
