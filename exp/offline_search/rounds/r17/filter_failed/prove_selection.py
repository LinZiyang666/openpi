"""Required no-filter proof for frozen-fit row selection in all three pipelines."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from .audit_counts import HERE, read
from .prove_retrieval import compare
from .row_filter import load_r10, r10_candidates, r17_candidates, save_r10


def proof_r17(pipeline, study_path, bundle, output):
    from exp.r17_integration.search.fit import load_fitted_recipe, save_fitted_recipe
    source = Path(study_path)
    row = next(r for r in read(source)['models'][bundle] if r['group'] == 'c')
    ready = read(Path(row['fit']) / 'READY.json')
    binding = ready['receipt']['binding']
    original = load_fitted_recipe(row['fit'], expected_binding=binding)
    selected = r17_candidates(original, np.unique(original.episode))
    destination = output / pipeline
    save_fitted_recipe(selected, ready['receipt'], destination)
    loaded = load_fitted_recipe(destination, expected_binding=binding)
    records = []
    for name in ('action', 'state', 'task', 'episode', 'step', 'fields', 'pca', 'projected', 'sigma', 'tasks', 'heads'):
        compare(getattr(original, name), getattr(loaded, name), name, records=records)
    if hasattr(original, 'distance_scale'):
        compare(original.distance_scale, loaded.distance_scale, 'distance_scale', records=records)
    return dict(pipeline=pipeline, source_fit=row['fit'], original_digest=ready['digest'],
                status='PASS' if all(r['equal'] for r in records) else 'FAIL', comparisons=records)


def run(output):
    from .audit_counts import REPO
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    results = [proof_r17('r17_mt50', '/data/r17_library/mt50/studies/pi_k2/STUDY.json', 'pi05_mw', output),
               proof_r17('r17_libero', REPO / 'exp/r17_integration/ops/cache_study/sizes/STUDY.json', 'cosmos_libero', output)]
    source = REPO / 'exp/offline_search/rounds/r10/artifacts/r10_pi05_spatial_50_A.pkl'
    original = load_r10(source)
    selected = r10_candidates(original, np.unique(original['method'].inner.lib_ep))
    path = output / 'r10_vla.pkl'
    save_r10(selected, path)
    loaded = load_r10(path)
    records = compare(vars(original['method']), vars(loaded['method']), 'method')
    results.append(dict(pipeline='r10_vla', source_fit=str(source),
                        status='PASS' if all(r['equal'] for r in records) else 'FAIL', comparisons=records))
    report = dict(method='frozen_fit_candidate_selection_no_refit',
                  status='PASS' if all(r['status']=='PASS' for r in results) else 'FAIL', pipelines=results)
    (output / 'PROOF.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], pipelines=[dict(pipeline=r['pipeline'], status=r['status'],
          comparisons=len(r['comparisons'])) for r in results]), indent=2))
    return report


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=HERE / 'selection_proof')
    args = p.parse_args()
    raise SystemExit(0 if run(args.output)['status']=='PASS' else 1)
