"""DP pure-level deficit check: rerun the DP pure-inference controls with the knob TEST topology.

hs7dpctl_full / hs7dpctl_ddim10 are plain pure-inference runs on the knob TEST topology (local 4090 R4/B8/MPS,
local x48 workers). Compare them by init with the original controls (R2, timan107 x32) and with the knob cells
that ended at the pure level. Read-only; prints to stdout.

Run from the repository root: .venv/bin/python -B exp/offline_search/rounds/r17/knob/dpctl_compare.py [dp] [dp_red]
"""
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from exp.offline_search.rounds.r17.knob import common as c
from exp.offline_search.rounds.r17.knob.noise_analysis import HERE, SPECS, load, pair_stats

RERUNS = {'dp': 'hs7dpctl_full', 'dp_red': 'hs7dpctl_ddim10'}


def control(run):
    refs = {}
    for (_, _, group, _), result in load(run).items():
        assert group == 'control'
        refs.update(result['data'])
    return refs


def main():
    for g in sys.argv[1:] or RERUNS:
        rerun = RERUNS[g]
        final, lanes, old_run = SPECS[g]
        old, new = control(old_run), control(rerun)
        print(f'== {g}: {old_run} (old control) vs {rerun} (rerun)')
        for suite in ('libero_10', 'libero_spatial'):
            keys = sorted(k for k in old if k[0] == suite)
            assert all(k in new for k in keys), (g, suite, len(keys), sum(k in new for k in keys))
            y = np.array([int(new[k]['success']) for k in keys]); z = np.array([int(old[k]['success']) for k in keys])
            s = pair_stats(y, z)
            print(f'  {suite}: rerun {y.sum()}/{len(y)}  old {z.sum()}/{len(z)}  rerun-only {s["gains"]}  old-only {s["losses"]}  p_two {s["p_two"]:.3f}')
        data = defaultdict(dict)
        for lane in lanes:
            for (model, size, group, suite), result in load(lane).items():
                c.merge(data[suite, size, group.rsplit('_', 1)[-1]], result['data'])
        cells = json.loads((HERE / f'test_results_{final}/test.json').read_text())['cells']
        ys, zo, zn, per_task = [], [], [], []
        for cell in cells:
            suite, size = cell['suite'], cell['size']
            for name, t in cell['tasks'].items():
                if t['level'] != 'pure':
                    continue
                keys = [(suite, name, i) for i in range(50)]
                arm = data[suite, size, 'pure']
                y = np.array([int(arm[k]['success']) for k in keys])
                assert y.sum() == t['successes']
                a = np.array([int(old[k]['success']) for k in keys]); b = np.array([int(new[k]['success']) for k in keys])
                ys.append(y); zo.append(a); zn.append(b)
                per_task.append((size, name.split('/', 1)[1][:48], int(y.sum()), int(a.sum()), int(b.sum())))
        y, a, b = map(np.concatenate, (ys, zo, zn))
        so, sn = pair_stats(y, a), pair_stats(y, b)
        print(f'  pure-level knob cells: {len(ys)} cells, {y.sum()} successes; old control {a.sum()} '
              f'(deficit {so["deficit"]}, p_two {so["p_two"]:.4f}); rerun {b.sum()} (deficit {sn["deficit"]}, p_two {sn["p_two"]:.4f})')
        for row in sorted(per_task):
            print('   size %d %-48s knob %2d old %2d rerun %2d' % row)


if __name__ == '__main__':
    main()
