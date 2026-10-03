"""B-library feature/weight audits and read-only deployment/preservation checks."""
from __future__ import annotations
import itertools
import json
import pickle
from unittest.mock import patch
import numpy as np

from exp.offline_search.rounds.r02.g1_awm.awm import _kernel_w
from exp.offline_search.rounds.r09.recipe import recipe
from exp.offline_search.closed_loop.ops.h100 import assets
from .data import HERE, SIZES, CELLS, IndexedRows, sha, write_json, assert_fit_input
from .build import run_root
from .train import TRAINING, HEADS, REVISION, load_size, queries, neighbors, feature_row


def preservation():
    before = json.loads((HERE / 'stage2b_preservation_before.json').read_text())
    changed = [p for p, digest in before.items() if sha(p) != digest]
    assert not changed, f'protected Stage 1/v1 files changed: {changed}'
    write_json(HERE / 'stage2b_preservation_audit.json',
               dict(PASS=True, protected_files=len(before), changed_files=changed,
                    purpose='hash-only preservation check; these outputs are never fitting inputs'))


def library_audits():
    assert len(list(HEADS.glob('*.npz'))) == 48
    records, coverage = [], []
    for model, suite in CELLS:
        for size in SIZES:
            lib, base, f, table = load_size(model, suite, size)
            cell = f'{model}_{suite}_{size}'
            source_sha = sha(HERE / 'artifacts' / f'r10_{cell}_G.pkl')
            reports = [json.loads((TRAINING / cell / f'task{t}.json').read_text()) for t in range(10)]
            radius = json.loads((TRAINING / cell / 'radius.json').read_text())
            assert radius['rule'] == 'per-task median LOEO 16th-neighbor distance'
            assert radius['quantile'] == .5 and radius['pair_cap'] == radius['k'] == 16
            for task, report in enumerate(reports):
                assert report['revision'] == REVISION
                assert report['source_fit_sha256'] == source_sha
                selected = set(int(e) for e in np.unique(lib.episode[lib.task_id == task]))
                for variant in ('loeo', 'pair'):
                    r = report[variant]
                    assert r['episodes'] == len(selected)
                    assert np.isclose(r['weight_sum'], r['anchors'])
                    if variant == 'pair': assert r['rows'] <= 16 * r['anchors']
                    heldout = []
                    for fold in r['folds']:
                        tr, val = set(fold['train_episodes']), set(fold['heldout_episodes'])
                        assert not tr & val and tr | val == selected
                        assert np.isclose(fold['weight_sum'], fold['n_anchor'])
                        heldout.extend(val)
                    assert len(heldout) == len(selected) and set(heldout) == selected
                coverage.append(dict(cell=cell, task=task, episodes=len(selected),
                                     loeo_anchors=report['loeo']['anchors'], pair_anchors=report['pair']['anchors'],
                                     pair_rows=report['pair']['rows'], pair_weight=report['pair']['weight_sum']))
            for variant in ('loeo', 'pair'):
                f.heads, f.head_meta = recipe.load_head(assert_fit_input(HEADS / f'{cell}_{variant}.npz'))
                assert f.head_meta['revision'] == REVISION
                assert f.head_meta['alpha'] == 100 and f.head_meta['n_rff'] == 384
                assert f.head_meta['blend'] == .5 and f.head_meta['chans'] == 6
                assert set(f.heads) == set(map(str, range(10)))
                np.testing.assert_array_equal(np.asarray(f.head_meta['sigma'], np.float32), base.sig)
                for head in f.heads.values():
                    assert head['coef'].shape == (60, 601) and head['w'].shape == (217, 384)
                for task in (0, 1):
                    for i, q in itertools.islice(queries(lib, table, task), 2):
                        rows, d, xv, rs8 = neighbors(base, q, int(lib.episode[i]))
                        assert np.all(lib.episode[rows] != lib.episode[i])
                        base.reset(q.episode)
                        kd = d.astype(np.float64)
                        cached = (base.os_synth(q, rows, _kernel_w(kd-kd[0], base.kref))
                                  if variant == 'loeo' else lib.action[rows[0]])
                        expected = feature_row(f, q, xv, rs8, cached)[None]
                        captured = []
                        def predict(head, x):
                            captured.append(x.copy())
                            return np.zeros((1, 60), np.float32)
                        with patch.object(recipe, '_predict', predict):
                            f._correction(q, cached)
                        assert expected.tobytes() == captured[0].tobytes()
                        regime = base._dist(q)[2]
                        assert regime == (0 if q.step == 0 else 2)
                        records.append(dict(cell=cell, variant=variant, row=i, regime=regime,
                                            feature_bytes=expected.nbytes))
    write_json(HERE / 'stage2b_feature_audit.json', dict(PASS=True, fit_pool='B', records=records))
    write_json(HERE / 'stage2b_episode_weight_audit.json', dict(PASS=True, fit_pool='B', records=coverage))


def relocation_and_bytes():
    prior, current, relocations = {}, {}, []
    for stage in (1, 2, '2b'):
        for model in ('pi05', 'groot'):
            run = run_root(stage, model)
            plan = json.loads((run / 'h100_sync/plan.json').read_text())
            for item in plan['files']:
                if stage != '2b':
                    prior[item['rel']] = item
                    continue
                if item['rel'] in current:
                    assert current[item['rel']]['sha256'] == item['sha256']
                current[item['rel']] = item
                if ': fitted method' not in ' '.join(item['reasons']): continue
                # Relocated GC outputs are validation subjects, never fit inputs.
                with open(item['source'], 'rb') as stream:
                    blob = pickle.load(stream)
                method = blob['method']
                base = method.inner.base
                assert base.head_meta['revision'] == REVISION
                assert isinstance(base, recipe.RecipeCorrectedBase)
                assert isinstance(base.act, IndexedRows) and method.inner.C.act is base.act
                assert base.act._array is None
                assert base.act.path.startswith(str(assets.BASE / 'store') + '/')
                for value in (method.source_fit, method.head_path, base.base_fit, base.head_path):
                    assert value.startswith(str(assets.BASE / 'mirror') + '/')
                assert blob['kwargs'] == assets.remap(
                    next(r['kwargs'] for r in json.loads((run/'arms.json').read_text())
                         if r['arm'] == item['original'].split('/')[-1][:-4]), run)
                relocations.append(dict(path=item['source'], bytes=item['size'], sha256=item['sha256']))
    assert len(relocations) == 48
    incremental = [f for rel, f in current.items() if rel not in prior or prior[rel]['sha256'] != f['sha256']]
    assert not any(f['rel'].startswith('store/') for f in incremental)
    sources = sorted(HERE.glob('*.py'))
    report = dict(PASS=True, extra_h100_store_bytes=0,
                  basis='additional destinations beyond accepted Stage 1 and superseded v1 plans',
                  incremental_run_and_mirror_bytes=sum(f['size'] for f in incremental),
                  all_three_stages_union_bytes=sum(f['size'] for f in {**prior, **current}.values()),
                  source_bytes=sum(p.stat().st_size for p in sources),
                  runtime_source_bytes=sum((HERE/p).stat().st_size for p in ('__init__.py','data.py','method.py')),
                  incremental_dependencies=incremental)
    write_json(HERE / 'stage2b_incremental_deployment.json', report)
    write_json(HERE / 'stage2b_relocation_audit.json', dict(PASS=True, artifacts=relocations))


def main():
    preservation()
    library_audits()
    relocation_and_bytes()
    print('PASS: preservation, B features/episodes/weights, relocation, exact bytes')


if __name__ == '__main__':
    main()
