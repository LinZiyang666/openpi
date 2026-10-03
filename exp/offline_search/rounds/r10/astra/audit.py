"""Post-assembly audits using only B queries and newly generated fit/plan artifacts."""
import copy
import json
import pickle
import numpy as np
from .boundary import HERE, NEW, install
from .method import DistanceCorrectedBase
from .build import audit_fits, snapshot, REPO
from exp.offline_search.rounds.r10 import data, train
from exp.offline_search.rounds.r09.recipe.recipe import RecipeCorrectedBase
from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM
from exp.offline_search.rounds.r02.g1_awm.awm import _kernel_w


def main():
    install()
    freeze = json.loads((HERE / 'freeze.json').read_text())
    for path, digest in freeze['hashes'].items():
        assert data.sha(path) == digest
    audit_fits()
    rows = []
    relocated = []
    for run in NEW:
        receipt = json.loads((run / 'emission_receipt.json').read_text())
        assert receipt['timestamp_utc'] > freeze['timestamp_utc']
        assert receipt['prediction_sha256'] == data.sha(HERE / 'PREDICTION.md')
        for arm in json.loads((run / 'arms.json').read_text()):
            with (run / 'fits' / (arm['arm'] + '.pkl')).open('rb') as f:
                fitted = pickle.load(f)['method']
            base = fitted.inner.base
            original_scale = base.distance_scale
            lib, _, _, table = train.load_size(arm['model'], arm['suite_short'], fitted.size)
            checks = 0
            for task in (0, 9):
                qlist = list(train.queries(lib, table, task))
                # Early query and three nonzero decisions, including a different episode.
                for i, q in (qlist[0], qlist[1], qlist[len(qlist)//2], qlist[-1]):
                    n, d, _, _ = train.neighbors(base, q, int(lib.episode[i]))
                    w = _kernel_w(d.astype(float) - float(d[0]), base.kref)
                    base.reset(q.episode)
                    raw = BlindAWM.os_synth(base, q, n, w)
                    corr = RecipeCorrectedBase._correction(base, q, raw)
                    # Exercise peak, interior and zero distance gates against raw G synthesis.
                    T, _, _, _, _, _, _, geometric, _, _, _ = base._dist(q)
                    actual_d = float(np.min(geometric[np.searchsorted(T.rows, n)]))
                    for ratio in (.5, 1.25, 2.5):
                        base.distance_scale = actual_d / ratio
                        base.reset(q.episode)
                        got = base.os_synth(q, n, w)
                        factor = 0. if int(q.step) == 0 else np.clip((2.-ratio)/1.25, 0, 1)
                        expected = np.array(raw, copy=True)
                        expected[:10, :6] += .5 * (corr[:, :6] * np.float32(factor))
                        np.testing.assert_allclose(got, expected, rtol=2e-6, atol=2e-7)
                        np.testing.assert_array_equal(got[:, 6:], raw[:, 6:])
                        np.testing.assert_array_equal(got[10:], raw[10:])
                        np.testing.assert_array_equal(base._anchor['action'], got)
                        if ratio >= 2 or int(q.step) == 0:
                            np.testing.assert_array_equal(got, raw)
                        checks += 1
            base.distance_scale = original_scale
            rows.append(dict(arm=arm['arm'], peak_taper_zero_early_anchor_checks=checks, PASS=True))
        plan = json.loads((run / 'h100_sync/plan.json').read_text())
        for item in plan['files']:
            if not any('fitted method' in reason for reason in item['reasons']):
                continue
            assert data.sha(item['source']) == item['sha256']
            with open(item['source'], 'rb') as f:
                blob = pickle.load(f)
            m = blob['method']
            assert isinstance(m.inner.base, DistanceCorrectedBase)
            assert m.inner.base.act.path.startswith('/data/oscl_h100/store/library/')
            assert '/r10/astra/calibration/' in m.calibration_path
            assert m.calibration_path.startswith('/data/oscl_h100/mirror/')
            arm = next(r for r in plan['arms'] if r['kwargs'] == blob['kwargs'])
            assert blob['spec'] == arm['method'] and blob['cell'] == arm['cell']
            assert set(m.inner.base.heads) == set(map(str, range(10)))
            relocated.append(dict(arm=arm['arm'], PASS=True, indexed_action_path=m.inner.base.act.path))
    data.write_json(HERE / 'serving_gate_audit.json', rows)
    data.write_json(HERE / 'relocation_audit.json', relocated)
    data.write_json(HERE / 'final_audit.json', dict(PASS=True, arms=len(rows), relocated=len(relocated),
        serving_gate_checks=sum(r['peak_taper_zero_early_anchor_checks'] for r in rows),
        freeze_unchanged=True, emitted_after_prediction=True, protected_inputs_unchanged=snapshot()))
    runtime = [HERE / '__init__.py', HERE / 'method.py']
    all_sources = sorted(HERE.glob('*.py'))
    for name, paths in (('H100_SOURCES.sha256', runtime), ('ALL_SOURCES.sha256', all_sources)):
        (HERE / name).write_text(''.join(f'{data.sha(p)}  {p.relative_to(REPO)}\n' for p in paths))
    data.write_json(HERE / 'source_files.json', dict(new_h100_sources=[dict(path=str(p.relative_to(REPO)),
        bytes=p.stat().st_size, sha256=data.sha(p)) for p in runtime],
        offline_only_sources=[str(p.relative_to(REPO)) for p in all_sources if p not in runtime]))
    print('AUDIT PASS', len(rows), 'arms;', len(relocated), 'relocated fits')


if __name__ == '__main__':
    main()
