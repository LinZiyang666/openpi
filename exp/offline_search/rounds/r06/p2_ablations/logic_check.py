"""Exhaustive bitmask and explicit no-progress blind-veto tests, no simulator."""
import collections
import json

import numpy as np

from exp.offline_search.harness import api, store
from exp.offline_search.closed_loop.blind import BlindResult, LookReason
from exp.offline_search.rounds.r04.k1_blind.checks import view, blind_view
from exp.offline_search.rounds.r06.p2_ablations.judge import _TriggerMask, BITS, TriggerCommitJudge, TriggerGrootCommitJudge
from exp.offline_search.rounds.r06.p2_ablations.make_arms import HERE, RUN, STORE, artifact, sources
from exp.offline_search.rounds.r06.p2_ablations.replay_audit import load, equal_payload


class ToyParent:
    def query(self, q):
        return q


class Toy(_TriggerMask, ToyParent):
    pass


def main():
    c = collections.Counter()
    for guard, mask in [('none', 0), ('all', 15), *BITS.items()]:
        for flags in range(16):
            m = object.__new__(Toy)
            m.disabled_mask = mask
            m._s = dict(burst_end=0, ret_end=0, flag=[int(bool(flags))])
            ex = dict(os_flags=float(flags), os_reason=float((flags & -flags).bit_length()),
                      os_force_miss=float(bool(flags)), stuck_n=3., overtime=2., lag=6.)
            ref = api.Result(np.arange(16), np.arange(16.), .5, action=np.zeros((10, 32), np.float32), extras=ex)
            got = m.query(ref)
            new = flags & ~mask
            equal_payload(ref, got)
            assert got.extras == {**ex, 'os_flags': float(new), 'os_reason': float((new & -new).bit_length()),
                                  'os_force_miss': float(bool(new))}
            assert m._s['flag'] == [int(bool(new))]
            c['bitmask_cases'] += 1
    rows = json.loads((HERE / 'arms_trigger_loo.json').read_text())
    for model in ('pi05', 'groot'):
        cls = TriggerCommitJudge if model == 'pi05' else TriggerGrootCommitJudge
        for suite in ('l10', 'spatial'):
            src = sources()[model, suite, 50, 'B'][1]
            for field, value in [('disabled_guard', 'invalid'), ('burst', 2), ('events', 'disp'), ('guards', False),
                                 ('monitor', 'loeo_xyz99'), ('policy_tail_gate', 'inherited'),
                                 ('stuck_guard', 'dense'), ('progress_guard', 'noprog_n')]:
                kw = {**src['kwargs'], field: value}
                try:
                    cls(**kw)
                except ValueError:
                    c['refusals'] += 1
                else:
                    raise AssertionError((field, value))
            qc = store.QueryCell(STORE, f'{model}_{suite}_cache')
            A = api.QueryArrays(qc)
            q, q1 = view(qc, A, 0), view(qc, A, 1)
            bq = blind_view(q1, prev_hit=True)
            for row in rows:
                if (row['model'], row['suite']) != (model, suite):
                    continue
                m = load(artifact(row).replace('<RUN>', str(RUN)))
                m.reset(q.episode)
                m.query(q)
                m._noprog_span = 1
                got = m.blind_step(bq)
                if m.disabled_guard == 'no_progress':
                    assert isinstance(got, BlindResult)
                else:
                    assert isinstance(got, LookReason) and got.name == 'noprog_span'
                assert m._noprog_span == 1
                c['explicit_progress_veto_cases'] += 1
    out = dict(PASS=True, **c)
    (HERE / 'results/logic.json').write_text(json.dumps(out, indent=1)+'\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
