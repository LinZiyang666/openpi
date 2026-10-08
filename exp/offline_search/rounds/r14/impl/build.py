"""Reuse verified cell PCA; fit every production row once per geometry."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from exp.offline_search.harness import api
from exp.offline_search.rounds.r10.data import SubsetLibrary
from exp.offline_search.rounds.r10.recipe.fitting import library_boundary
from exp.offline_search.rounds.r14.knob.recipe import R14Knob, _PartitionCache, SWITCHES
from exp.offline_search.rounds.r14.knob.controller import caps
from exp.offline_search.rounds.r14.knob import fitting
from .common import *

def frozen():
    freeze = json.loads((ASTRA / 'freeze.json').read_text())
    names = ['SPEC.md', 'design.json', 'allocation_fits.json', 'pilot_grid.csv',
             'pilot_selection.json', 'test_grid.csv', 'continuity_summary.json',
             'controller.py', 'selfcheck.py', 'REQUESTS.md']
    hashes = {name:sha(ASTRA / name) for name in names}
    for name, digest in hashes.items():
        if freeze['files'][name] != digest:
            raise ValueError(f'changed frozen source: {name}')
    return hashes

def converted(blob, switches, method='off', target=None):
    obj = blob['method']
    obj.__class__ = R14Knob
    obj.knob_method, obj.target = method, target
    for name in SWITCHES:
        setattr(obj, 'r14_' + name, switches[name])
    obj.r14_fit, obj.r14_diagnostics = None, None
    obj._accepted_prefix, obj._fallback_counts = ((), ()), {}
    kwargs = dict(library=obj.library, episode_subset=obj.episode_subset, **switches, method=method, target=target)
    blob.update(spec=SPEC, kwargs=kwargs)
    return obj

def publish(blob, cell, variant, target, manifest):
    obj = blob['method']
    obj.r14_fit = manifest
    # The manifest hash identifies the numerical fit; the plugin adds the
    # actual artifact SHA after loading, avoiding a self-referential pickle.
    obj.r14_fit['manifest_sha256'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    obj.name = f'R14_{cell}_{variant}_{target}'
    obj.reset(api.EpisodeView('r14-build', '', 0, 0, 0, 0))
    blob['r14_manifest'] = obj.r14_fit
    name = key(cell, variant, target).replace(':', '_').replace('.', 'p') + '.pkl'
    path = KNOB / 'artifacts' / name
    dump_pickle(path, blob)
    return dict(config_key=key(cell, variant, target), path=str(path), sha256=sha(path),
                bytes=path.stat().st_size, spec=SPEC, kwargs=blob['kwargs'], manifest=manifest)

def cell_build(cell):
    started = time.monotonic()
    source_path, original = source(cell)
    baseline = original['method']
    design = json.loads((ASTRA / 'design.json').read_text())
    hashes = frozen()
    model, suite, size = cell.split('_')
    lib = SubsetLibrary(STORE, f'{model}_{suite}', int(size))
    np.testing.assert_array_equal(lib.rows, baseline.row_subset)
    source_sha = sha(source_path)
    source_paths = [source_path, ROUNDS / f'r10/subsets/{cell}.json', ROUNDS / f'r10/subsets/{cell}.npy',
                    lib.dir / 'episodes.json']
    ps = []
    for cam in (0, 1):
        p = ROUNDS / f'r10/pca/{model}_{suite}/{size}/v{cam}.npz'
        source_paths.append(p)
        with np.load(p, allow_pickle=False) as z:
            np.testing.assert_array_equal(z['basis'].T, getattr(baseline.inner.base, f'B{cam}T'))
            assert str(z['rows_sha256']) == sha(ROUNDS / f'r10/subsets/{cell}.npy')
            ps.append(np.array(z['proj'], np.float32))
    source_hashes = {str(p):sha(p) for p in source_paths}
    choices = baseline.fit_info['episode_ids_by_task']
    ef = {int(e):i % 5 for es in choices.values() for i,e in enumerate(es)}
    with library_boundary(STORE, lib.dir) as reads:
        C = dict(X=np.c_[ps[0], ps[1], lib.rs[:, :8]].astype(np.float32),
                 act=np.asarray(lib.action[:, :10, :7], np.float32), ep=lib.episode,
                 step=lib.step, task=lib.task_id, fold=np.asarray([ef[int(e)] for e in lib.episode]))
        pooled_c = fitting.continuity(C, baseline.inner.base.sig)
        reference_c = next(r['pooled_scale'] for r in json.loads((ASTRA / 'continuity_summary.json').read_text()) if r['cell'] == cell)
        assert pooled_c == reference_c, (pooled_c, reference_c)
        records, geometries, datasets, heads, scales = [], {}, {}, {}, {}
        for variant, switches in design['variants'].items():
            blob = copy.deepcopy(original)
            obj = converted(blob, switches)
            base = obj.inner.base
            scope, metric, sharing = (switches[s] for s in ('scope', 'metric', 'head'))
            geomkey = scope, metric
            changed = scope == 'pool' or metric == 'pool'
            if changed:
                if geomkey not in geometries:
                    geometries[geomkey] = fitting.geometry(C, base, scope, metric)
                base.tasks = copy.deepcopy(geometries[geomkey])
            if changed or sharing == 'pool':
                if geomkey not in datasets:
                    datasets[geomkey] = fitting.full_heads(lib, base, obj.inner.C, scope, metric)
                headkey = scope, metric, sharing
                if headkey not in heads:
                    heads[headkey] = fitting.fit_heads(datasets[geomkey], sharing)
                base.heads, counts = heads[headkey]
                base.head_meta=dict(base.head_meta,task_onehot=sharing=='task',per_task=sharing=='task')
                obj.fit_info = copy.deepcopy(obj.fit_info)
                obj.fit_info['head_counts'] = counts
            if changed:
                if geomkey not in scales:
                    scales[geomkey] = fitting.calibrate(C, base, scope, metric)
                base.distance_scale, folds = scales[geomkey]
                obj.fit_info.update(distance_scale=base.distance_scale, distance_folds=folds)
            if switches['continuity'] == 'pool':
                for T in base.tasks.values():
                    T.s_c = pooled_c
            if switches['guard'] == 'pool':
                fitting.pool_guard(obj.inner, lib, C)
            if variant != 'base':
                base.__class__ = _PartitionCache
                base.r14_head = sharing
            manifest = dict(cell=cell, variant=variant, switches=switches,
                source_artifact_sha256=source_sha, sources_sha256=source_hashes,
                frozen_sources_sha256=hashes, row_subset_sha256=obj.fit_info['row_subset_sha256'],
                rows=lib.L, episodes=int(size), full_training_rows=True,
                selected_library_PCA_reused=True, action_sigma_byte_preserved=True, kref=base.kref,
                head_counts=obj.fit_info['head_counts'], distance_scale=base.distance_scale,
                distance_folds=obj.fit_info['distance_folds'], pooled_continuity=pooled_c,
                continuity_exact_frozen=True, fit_reads=sorted(reads))
            records.append(publish(blob, cell, variant, None, manifest))
            print('PUBLISHED', cell, variant, flush=True)
        # Guard-only null is CPU-only, never inserted into either grid.
        blob = copy.deepcopy(original)
        switches = dict(design['variants']['base'], guard='pool')
        obj = converted(blob, switches)
        fitting.pool_guard(obj.inner, lib, C)
        obj.inner.base.__class__ = _PartitionCache
        obj.inner.base.r14_head = 'task'
        records.append(publish(blob, cell, 'guard_null', None, dict(cell=cell,variant='guard_null',
            switches=switches, cpu_only=True, sources_sha256=source_hashes, frozen_sources_sha256=hashes,
            row_subset_sha256=obj.fit_info['row_subset_sha256'], rows=lib.L, full_training_rows=True)))
    for fit in (r for r in json.loads((ASTRA / 'allocation_fits.json').read_text()) if r['cell'] == cell):
        target = fit['target']
        path, periodic = source(cell, 'periodic', target)
        assert periodic['method'].knob_settings['setting'] == fit['reference_K']
        K = caps(fit['reference_K'], fit['priority']['z'], fit['bias'], fit['gain'])
        np.testing.assert_array_equal(K, fit['K'])
        records.append(dict(config_key=key(cell, 'periodic', target), path=str(path), sha256=sha(path),
                            bytes=path.stat().st_size, spec=periodic['spec'], kwargs=periodic['kwargs'], reused_R11=True))
        blob = copy.deepcopy(original)
        obj = converted(blob, design['variants']['base'], 'task_periodic', target)
        obj.knob_settings = dict(K=fit['K'], reference_K=fit['reference_K'], priority=fit['priority']['z'],
                                 gain=fit['gain'], bias=fit['bias'], prior_episodes=fit['prior_episodes'])
        records.append(publish(blob, cell, 'task_periodic', target, dict(cell=cell,variant='task_periodic',
            frozen_allocation=fit, frozen_sources_sha256=hashes, sources_sha256=source_hashes,
            lower_layers_byte_preserved=True, row_subset_sha256=obj.fit_info['row_subset_sha256'])))
    write_json(HERE / 'cells' / f'{cell}.json', records)
    write_json(HERE / 'cells' / f'{cell}_build.json', dict(cell=cell,seconds=time.monotonic()-started,
        rows=lib.L, artifacts=len(records), reads_sha256={p:sha(p) for p in sorted(reads)}))
    print('DONE', cell, time.monotonic() - started, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--recover', action='store_true', help='verify completed component fits and finish controller artifacts')
    a = ap.parse_args()
    if a.recover:
        complete_controller_artifacts()
        return
    if a.cell:
        cell_build(a.cell)
        return
    cells = json.loads((ASTRA / 'design.json').read_text())['cells']
    def run(cell):
        log = HERE / 'logs' / f'build_{cell}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('w') as stream:
            p = subprocess.run(PREFIX + ['-m', 'exp.offline_search.rounds.r14.impl.build', '--cell', cell],
                               cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
        if p.returncode:
            raise RuntimeError(f'build failed: {log}')
        print('CELL', cell, flush=True)
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(run, cells))
    write_json(HERE / 'fits.json', [r for c in cells for r in json.loads((HERE / 'cells' / f'{c}.json').read_text())])

def complete_controller_artifacts():
    """Resume a completed expensive component fit without refitting its rows."""
    design=json.loads((ASTRA/'design.json').read_text());hashes=frozen()
    allrecords=[]
    for cell in design['cells']:
        started=time.monotonic();records=[]
        for variant in [*design['variants'],'guard_null']:
            p=KNOB/'artifacts'/(key(cell,variant).replace(':','_')+'.pkl')
            blob=pickle.loads(p.read_bytes());manifest=blob['r14_manifest']
            if manifest['cell']!=cell or manifest['variant']!=variant:
                raise ValueError('completed component metadata differs')
            for name,digest in manifest['frozen_sources_sha256'].items():
                assert sha(ASTRA/name)==digest
            for path,digest in manifest['sources_sha256'].items():
                assert sha(path)==digest
            records.append(dict(config_key=key(cell,variant),path=str(p),sha256=sha(p),bytes=p.stat().st_size,
                                spec=blob['spec'],kwargs=blob['kwargs'],manifest=manifest))
        source_path,original=source(cell);source_hashes=records[0]['manifest']['sources_sha256']
        for fit in (r for r in json.loads((ASTRA/'allocation_fits.json').read_text()) if r['cell']==cell):
            target=fit['target'];path,periodic=source(cell,'periodic',target)
            assert periodic['method'].knob_settings['setting']==fit['reference_K']
            np.testing.assert_array_equal(caps(fit['reference_K'],fit['priority']['z'],fit['bias'],fit['gain']),fit['K'])
            records.append(dict(config_key=key(cell,'periodic',target),path=str(path),sha256=sha(path),
                bytes=path.stat().st_size,spec=periodic['spec'],kwargs=periodic['kwargs'],reused_R11=True))
            blob=copy.deepcopy(original)
            obj=converted(blob,design['variants']['base'],'task_periodic',target)
            obj.knob_settings=dict(K=fit['K'],reference_K=fit['reference_K'],priority=fit['priority']['z'],
                gain=fit['gain'],bias=fit['bias'],prior_episodes=fit['prior_episodes'])
            records.append(publish(blob,cell,'task_periodic',target,dict(cell=cell,variant='task_periodic',
                frozen_allocation=fit,frozen_sources_sha256=hashes,sources_sha256=source_hashes,
                lower_layers_byte_preserved=True,row_subset_sha256=obj.fit_info['row_subset_sha256'])))
        write_json(HERE/'cells'/f'{cell}.json',records)
        allrecords.extend(records)
        print('VERIFIED_COMPLETE',cell,round(time.monotonic()-started,2),flush=True)
    write_json(HERE/'fits.json',allrecords)

if __name__ == '__main__':
    main()
