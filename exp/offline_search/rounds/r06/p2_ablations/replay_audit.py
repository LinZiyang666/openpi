"""Same-history leave-one-out audit inside P1's REAL-plugin recorded replay driver.

Baseline B determines the served actions/hits/vision schedule. Independent fresh
prefit variants receive exactly those online QueryViews. This isolates direct
trigger effects; counterfactual future histories are not asserted equal.
"""
import argparse
import collections
import copy
import json
import pickle
from pathlib import Path

import numpy as np

from exp.offline_search.closed_loop import plugin
from exp.offline_search.closed_loop.blind import BlindResult, LookReason
from exp.offline_search.rounds.r05.q1_commit.judge import CommitJudge
from exp.offline_search.rounds.r06.p1_groot_commit.judge import GrootCommitJudge
from exp.offline_search.rounds.r06.p1_groot_commit import replay
from exp.offline_search.rounds.r06.p2_ablations.judge import BITS
from exp.offline_search.rounds.r06.p2_ablations.make_arms import HERE, RUN, artifact, sources

COUNTS = collections.Counter()
CHANGES = {g: collections.Counter() for g in (*BITS, 'none', 'all')}


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)['method']


def equal_payload(a, b, confidence=True):
    for key in ('topk', 'scores', 'action'):
        assert np.array_equal(getattr(a, key), getattr(b, key)), key
    assert a.library == b.library
    if confidence:
        assert np.float64(a.confidence).tobytes() == np.float64(b.confidence).tobytes()


def equal_blind(a, b):
    assert type(a) is type(b), (a, b)
    if isinstance(a, LookReason):
        assert a == b
    else:
        for k in ('action', 'rows', 'weights'):
            assert np.array_equal(getattr(a, k), getattr(b, k)), k
        assert a.library == b.library and a.extras == b.extras


def flags_from_diagnostics(m, ex):
    n = ex['stuck_n']
    closed = ex.get('gexec', 0.) * (-1 if m.model == 'groot' else 1) > 0
    span = ex.get('noprog_span', ex.get('noprog_n', 0))
    return (int(n >= m.stuck_thr) | (int(bool(ex['term1']) and closed) << 1)
            | (int(ex.get('overtime', 0) > 1 and ex.get('lag', 0) > m.lag_thr and n >= 1) << 2)
            | (int(span >= m.noprog_n - 1) << 3))


class AuditMixin:
    def reset(self, ep):
        super().reset(ep)
        for m in self.shadows:
            m.reset(ep)

    def invalidate_anchor(self):
        super().invalidate_anchor()
        for m in self.shadows:
            m.invalidate_anchor()

    def query(self, q):
        ref = super().query(q)
        flags = int(ref.extras['os_flags'])
        assert flags == flags_from_diagnostics(self, ref.extras)
        COUNTS['vision_queries'] += 1
        COUNTS['baseline_misses'] += bool(flags)
        COUNTS['allvision_queries' if np.all(q.hist_has_vision) else 'gap_queries'] += 1
        for m in self.shadows:
            got = m.query(q)
            equal_payload(ref, got)
            want = flags & ~m.disabled_mask
            expected = {**ref.extras, 'os_flags': float(want), 'os_force_miss': float(bool(want)),
                        'os_reason': float((want & -want).bit_length() if want else 0)}
            assert got.extras == expected, (m.disabled_guard, got.extras, expected)
            assert m._s['stuck_n'] == self._s['stuck_n']
            assert m._noprog_span == self._noprog_span
            assert m._s['flag'][-1] == int(bool(want))
            c = CHANGES[m.disabled_guard]
            c['queries'] += 1
            c['misses'] += bool(want)
            c['removed_misses'] += bool(flags) and not want
            c['sole_disabled_reason'] += bool(flags) and flags & ~m.disabled_mask == 0
            c['changed_reason_with_miss_retained'] += bool(want) and got.extras['os_reason'] != ref.extras['os_reason']
            c['overtime_fires_retained'] += bool(want & 4)
            assert c['removed_misses'] == c['sole_disabled_reason']
        return ref

    def blind_step(self, q):
        ref = super().blind_step(q)
        for m in self.shadows:
            got = m.blind_step(q)
            if isinstance(ref, LookReason) and ref.name == 'noprog_span' and m.disabled_mask & 8:
                CHANGES[m.disabled_guard]['no_progress_look_veto_removed'] += 1
            else:
                equal_blind(ref, got)
            CHANGES[m.disabled_guard]['blind_calls'] += 1
        return ref

    def policy_tail_step(self, q):
        ref = super().policy_tail_step(q)
        for m in self.shadows:
            equal_blind(ref, m.policy_tail_step(q))
            CHANGES[m.disabled_guard]['policy_tail_calls'] += 1
        return ref


class AuditPi(AuditMixin, CommitJudge):
    pass


class AuditGroot(AuditMixin, GrootCommitJudge):
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--suite', required=True)
    p.add_argument('--stream', choices=['cache', 'inf'], required=True)
    p.add_argument('--episodes', default='all')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    src = sources()[a.model, a.suite, 50, 'B'][1]
    path = artifact(src).replace('<RUN>', '/home/weiland/trace_runs/os_closed_loop/r06_paper')
    rows = [r for r in json.loads((HERE / 'arms_trigger_loo.json').read_text())
            if (r['model'], r['suite']) == (a.model, a.suite)]
    original_install = plugin.install

    def install(opts, model):
        rt = original_install(opts, model)
        rt.method.__class__ = AuditPi if model == 'pi05' else AuditGroot
        shadows = [load(artifact(r).replace('<RUN>', str(RUN))) for r in rows]
        for guard, mask in [('none', 0), ('all', 15)]:
            m, _ = plugin.clone_method(shadows[0])
            m.disabled_guard, m.disabled_mask = guard, mask
            shadows.append(m)
        rt.method.shadows = shadows
        return rt

    plugin.install = install
    args = ['--method', src['method'], '--kwargs', json.dumps(src['kwargs']),
            '--cell', f'{a.model}_{a.suite}_cache', '--replay-cell', f'{a.model}_{a.suite}_{a.stream}',
            '--fit-artifact', path, '--judge', 'guard_only', '--policy-tail', '--blocks', '1',
            '--episodes', a.episodes, '--tag', 'baseline_B_audit', '--out', str(a.out)]
    replay.main(args)
    result = dict(PASS=True, model=a.model, suite=a.suite, stream=a.stream, episodes=a.episodes,
                  baseline=dict(COUNTS), variants={g: dict(c) for g, c in CHANGES.items()}, mismatches=0,
                  protocol='same online histories as real-plugin B replay; observations fixed')
    (a.out / 'audit.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
