"""Research-output consistency checks; does not touch any existing repo method."""
import csv
import json
import pathlib
import sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))
from exp.offline_search.rounds.r04.ideation_A.measure_blind import OUT, ROOT, CELLS, get_method, store

def read(name):
    return list(csv.DictReader((OUT / name).open()))

checks = {"query_cells": 0, "horizon_windows": 0, "fitted_libraries": [], "metric_rows": 0}
for cell in CELLS:
    for scale in [50, 500]:
        method = get_method(cell, scale)
        expected = 'current' if scale == 50 else ('bpool_cs' if cell.startswith('pi05') else 'bpool_all')
        # R2 g50 artifacts predate the explicit G3 os_* aliases.
        assert method.cand_name == expected and method.fit_data == 'same'
        assert method.fit_src == ('current' if scale == 50 else 'big')
        assert getattr(method, 'os_fit_library', expected) == expected
        checks['fitted_libraries'].append(dict(cell=cell, scale=scale, fit=expected, kref=method.kref))
        data = read(f'metrics_{cell}_{scale}.csv')
        idx = {(d['cell'], d['method'], d['split'], int(d['h'])): d for d in data}
        assert len(idx) == len(data)
        for d in data:
            assert int(d['n']) > 0
            assert all(np.isfinite(float(d[k])) for k in ['err_mean', 'err_median', 'err_p90', 'grip_mis', 'delta_awm'])
        checks['metric_rows'] += len(data)
        for arm in ['inf', 'cache']:
            key = cell + '_' + arm
            q = store.QueryCell(ROOT, key)
            checks['query_cells'] += 1
            with np.load(OUT / f'anchors_{key}_{scale}.npz') as a:
                assert a['rows'].shape == (q.N, 16)
                assert np.allclose(a['weights'].sum(1), 1, atol=2e-6)
                assert np.isfinite(a['action']).all()
            for h in range(1, 5):
                with np.load(OUT / f'windows_{key}_{scale}_h{h}.npz') as w:
                    assert len(w['anchor']) == q.N - 500*h
                    assert np.array_equal(w['target'], w['anchor'] + h)
                    assert np.array_equal(q.ep[w['anchor']], q.ep[w['target']])
                    assert ('err_anchor_chunk' in w) == ((h + 1)*5 <= q.H)
                    checks['horizon_windows'] += len(w['anchor'])
                for method_name in ['awm', 'kernel_clock', 'phase_abs']:
                    n = int(idx[key, method_name, 'all', h]['n'])
                    assert n == sum(int(idx[key, method_name, p, h]['n']) for p in ['early', 'mid', 'late'])
                    assert n == sum(int(idx[key, method_name, p, h]['n']) for p in ['near_transition', 'far_transition'])

logs = json.loads((OUT / 'log_summaries.json').read_text())
assert len(logs) == 16 and all(d['episodes'] == 500 for d in logs)
checks['accepted_closed_loop_episodes'] = sum(d['episodes'] for d in logs)
checks['accepted_closed_loop_decisions'] = sum(d['N'] for d in logs)
for row in read('log_schedule_aggregate.csv'):
    v, m, ir = (float(row[k]) for k in ['vision_share', 'miss_share', 'ir'])
    assert 0 <= m <= v <= 1 and np.isclose(ir, .152*v+.848*m)
    assert ir <= float(row['old_ir']) + 1e-12
for path in OUT.glob('*.npz'):
    assert path.stat().st_size < 50_000_000, path
checks['maximum_array_file_bytes'] = max(p.stat().st_size for p in OUT.glob('*.npz'))
checks['status'] = 'passed'
(OUT / 'validation.json').write_text(json.dumps(checks, indent=2))
print(json.dumps(checks, indent=2))
