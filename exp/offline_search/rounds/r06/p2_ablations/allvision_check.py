"""Exercise the stock all-vision branch, including GR00T's terminal sign fix."""
import argparse
import collections
import json
from pathlib import Path

from exp.offline_search.harness import api, store
from exp.offline_search.rounds.r04.k1_blind.checks import view
from exp.offline_search.rounds.r06.p2_ablations.make_arms import HERE, RUN, STORE, sources, artifact
from exp.offline_search.rounds.r06.p2_ablations.replay_audit import load, equal_payload, flags_from_diagnostics
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--suite', required=True)
    a = p.parse_args()
    source = sources()[a.model, a.suite, 50, 'B'][1]
    base = load(artifact(source).replace('<RUN>', '/home/weiland/trace_runs/os_closed_loop/r06_paper'))
    rows = [r for r in json.loads((HERE / 'arms_trigger_loo.json').read_text())
            if (r['model'], r['suite']) == (a.model, a.suite)]
    variants = [load(artifact(r).replace('<RUN>', str(RUN))) for r in rows]
    counts = {m.disabled_guard: collections.Counter() for m in variants}
    n = 0
    for stream in ('cache', 'inf'):
        qc = store.QueryCell(STORE, f'{a.model}_{a.suite}_{stream}')
        A = api.QueryArrays(qc)
        for ei in np.linspace(0, len(qc.episodes)-1, 30, dtype=int):
            e = qc.episodes[ei]
            for m in [base, *variants]:
                m.reset(view(qc, A, e['start']).episode)
            for i in range(e['start'], e['end']):
                q = view(qc, A, i)
                ref = base.query(q)
                flags = int(ref.extras['os_flags'])
                assert flags == flags_from_diagnostics(base, ref.extras)
                for m in variants:
                    got = m.query(q)
                    equal_payload(got, ref)
                    f = flags & ~m.disabled_mask
                    assert got.extras == {**ref.extras, 'os_flags': float(f), 'os_force_miss': float(bool(f)),
                                          'os_reason': float((f & -f).bit_length() if f else 0)}
                    c = counts[m.disabled_guard]
                    c['baseline_misses'] += bool(flags)
                    c['removed_misses'] += bool(flags) and not f
                    c['overtime_retained'] += bool(f & 4)
                    c['terminal_retained'] += bool(f & 2)
                n += 1
    out = dict(PASS=True, model=a.model, suite=a.suite, recorded_episodes=60, queries=n, mismatches=0,
               variants={k: dict(v) for k, v in counts.items()})
    (HERE / 'results' / f'allvision_{a.model}_{a.suite}.json').write_text(json.dumps(out, indent=1)+'\n')
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
