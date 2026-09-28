"""Real-plugin new all-off path == old B guards=False == A (audit original kth ties)."""
import argparse
import collections
import copy
import json
from pathlib import Path

import numpy as np

from exp.offline_search.closed_loop import plugin
from exp.offline_search.closed_loop.blind import LookReason
from exp.offline_search.rounds.r06.p1_groot_commit import replay
from exp.offline_search.rounds.r06.p1_groot_commit.tie_rule import StableTieBlindAWM
from exp.offline_search.rounds.r06.p2_ablations.judge import TriggerCommitJudge, TriggerGrootCommitJudge
from exp.offline_search.rounds.r06.p2_ablations.make_arms import HERE, RUN, artifact, sources
from exp.offline_search.rounds.r06.p2_ablations.replay_audit import load, equal_payload, equal_blind

COUNTS = collections.Counter()
TIES = []


class NestMixin:
    def reset(self, ep):
        super().reset(ep)
        for x in self.controls:
            x.reset(ep)

    def query(self, q):
        got = super().query(q)
        old, raw, stable = [x.query(q) for x in self.controls]
        equal_payload(got, old)
        assert got.extras == old.extras
        assert got.extras['os_force_miss'] == 0
        assert np.array_equal(got.topk, stable.topk)
        assert np.array_equal(got.action, stable.action)
        COUNTS['anchors'] += 1
        if not np.array_equal(raw.action, got.action):
            A = self.controls[1]
            T, *_, dt = A._dist(q)
            ra, rb = set(raw.topk) - set(got.topk), set(got.topk) - set(raw.topk)
            assert ra and len(ra) == len(rb)
            boundary = np.sort(dt)[15]
            assert all(dt[np.searchsorted(T.rows, r)] == boundary for r in ra | rb)
            TIES.append(dict(episode=q.episode.uid, step=int(q.step), A_only=sorted(map(int, ra)),
                             B_only=sorted(map(int, rb)), distance=float(boundary)))
            COUNTS['deployed_A_tied_anchors'] += 1
        return got

    def blind_step(self, q):
        got = super().blind_step(q)
        old, raw, stable = [x.blind_step(q) for x in self.controls]
        equal_blind(got, old)
        equal_blind(got, stable)
        if not isinstance(got, LookReason):
            COUNTS['blind_tails'] += 1
            assert not isinstance(raw, LookReason)
            if not np.array_equal(got.action, raw.action):
                COUNTS['deployed_A_tied_tails'] += 1
        else:
            assert raw == got
        return got


class NestPi(NestMixin, TriggerCommitJudge):
    pass


class NestGroot(NestMixin, TriggerGrootCommitJudge):
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--suite', required=True)
    p.add_argument('--episodes', default='all')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    row, = [r for r in json.loads((HERE / 'arms_trigger_loo.json').read_text())
            if (r['model'], r['suite'], r['kwargs']['disabled_guard']) == (a.model, a.suite, 'stuck')]
    orig = plugin.install

    def install(opts, model):
        rt = orig(opts, model)
        m = rt.method
        m.disabled_guard, m.disabled_mask = 'all', 15
        m.__class__ = NestPi if model == 'pi05' else NestGroot
        s = sources()
        old = load(artifact(s[model, a.suite, 50, 'B'][1]).replace('<RUN>', '/home/weiland/trace_runs/os_closed_loop/r06_paper'))
        old.guards = False
        raw = load(artifact(s[model, a.suite, 50, 'A'][1]))
        stable, _ = plugin.clone_method(raw)
        stable.__class__ = StableTieBlindAWM
        m.controls = [old, raw, stable]
        return rt

    plugin.install = install
    replay.main(['--method', row['method'], '--kwargs', json.dumps(row['kwargs']),
                 '--cell', f'{a.model}_{a.suite}_cache', '--replay-cell', f'{a.model}_{a.suite}_cache',
                 '--fit-artifact', artifact(row).replace('<RUN>', str(RUN)), '--judge', 'guard_only',
                 '--policy-tail', '--blocks', '1', '--episodes', a.episodes, '--tag', 'nesting', '--out', str(a.out)])
    result = dict(PASS=True, model=a.model, suite=a.suite, episodes=a.episodes, counts=dict(COUNTS),
                  new_all_off_equals_original_B_guards_false=True, stable_tie_A_equal=True,
                  original_A_ties=TIES)
    (a.out / 'nesting.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
