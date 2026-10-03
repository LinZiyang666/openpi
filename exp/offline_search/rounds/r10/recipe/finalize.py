"""Local integrity, standard control plan, relocation, source manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

from exp.offline_search.rounds.r10.data import HERE as R10, STORE, RUNS, CELLS, SIZES, sha, write_json
from .build import HERE, REPO, CURRENT, artifact_path, run_cpu


def protected_paths():
    paths = list(R10.glob('*.py')) + list((R10 / 'astra').glob('*.py'))
    for model, suite in CELLS:
        for size in SIZES:
            stem = f'{model}_{suite}_{size}'
            paths += [R10 / 'subsets' / f'{stem}.json', R10 / 'subsets' / f'{stem}.npy',
                      R10 / 'artifacts' / f'r10_{stem}_G.pkl', R10 / 'heads2b' / f'{stem}_loeo.npz',
                      R10 / 'astra/calibration' / f'{stem}.json']
    for root in sorted(RUNS.glob('r10_corr3_*')):
        paths += list((root / 'fits').glob('*.pkl')) + [root / 'arms.json']
    return sorted(set(p for p in paths if p.is_file()))


def snapshot():
    path = HERE / 'protected_inputs.json'
    current = {str(p): sha(p) for p in protected_paths()}
    if path.exists():
        if current != json.loads(path.read_text()):
            raise ValueError('frozen R10 inputs changed')
    else:
        write_json(path, current)
    # Astra explicitly froze these sources and calibrations before this work.
    freeze = json.loads((R10 / 'astra/freeze.json').read_text())
    for filename, digest in freeze['hashes'].items():
        if sha(filename) != digest:
            raise ValueError(f'astra frozen input changed: {filename}')
    return len(current)


def graph_contract(method):
    """Inspect actual serialization state without materializing parent payloads."""
    seen, modules, paths = set(), set(), []
    def walk(obj):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, np.ndarray):
            return
        if isinstance(obj, (str, Path)):
            if str(obj).startswith('/'):
                paths.append(str(obj))
            return
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (tuple, list)):
            for v in obj:
                walk(v)
        else:
            module = sys.modules.get(type(obj).__module__)
            filename = getattr(module, '__file__', '') or ''
            if filename and Path(filename).resolve().is_relative_to(REPO / 'exp/offline_search'):
                modules.add(type(obj).__module__)
                for v in vars(obj).values():
                    walk(v)
    walk(method)
    if any('explore_' in module for module in modules):
        raise ValueError('exploration class in deployment graph')
    if any('/os_closed_loop/' in path for path in paths):
        raise ValueError('closed-loop dependency in deployment method')
    if any(path.endswith(('.pkl', '.npz')) for path in paths):
        raise ValueError('external fit/head/calibration dependency in deployment method')
    return dict(class_modules=sorted(modules), external_payload=method.inner.base.act.path,
                fitted_state_self_contained=True, no_exploration_classes=True)


def artifacts_audit():
    records = []
    for model, suite in CELLS:
        for size in (*SIZES, None):
            path = artifact_path(model, suite, size)
            with path.open('rb') as f:
                method = pickle.load(f)['method']
            info = method.fit_info
            parent = Path(info['parent']).resolve()
            if any(parent not in Path(p).resolve().parents for p in info['fit_reads']):
                raise ValueError('fit read outside selected library')
            records.append(dict(model=model, suite=suite, size=method.size, kind='current' if size is None else 'R10 subset',
                artifact=str(path), sha256=sha(path), bytes=path.stat().st_size,
                fit_reads=info['fit_reads'], library=method.library, rows=len(method.row_subset),
                distance_scale=info['distance_scale'], **graph_contract(method)))
    write_json(HERE / 'artifact_audit.json', dict(PASS=True, artifacts=28, all_fit_reads_selected_library_only=True,
                                               records=records))
    return records


def plan():
    names = [r['arm'] for r in json.loads((CURRENT / 'arms.json').read_text())]
    if len(names) != 4:
        raise ValueError('current root must contain four arms')
    run_cpu(['-m', 'exp.offline_search.closed_loop.ops.h100.control', 'plan', str(CURRENT), *names],
            HERE / 'validation/control_plan.log')
    standard = json.loads((CURRENT / 'h100_sync/plan.json').read_text())
    new = [f for f in standard['files'] if f['rel'].startswith(f'runs/{CURRENT.name}/')]
    fits = [f for f in new if 'fitted method' in ' '.join(f['reasons'])]
    relocated = []
    for f in fits:
        with Path(f['source']).open('rb') as handle:
            blob = pickle.load(handle)
        method = blob['method']
        if not method.inner.base.act.path.startswith('/data/oscl_h100/store/library/'):
            raise ValueError('indexed action payload was not relocated to h100')
        # A standard relocated artifact has no original local filesystem roots.
        for value in [method.inner.base.act.path, method.inner.C.act.path, method.fit_info['parent'],
                      *method.fit_info['fit_reads']]:
            if '/home/weiland/' in value:
                raise ValueError('original fit path remained in relocated artifact')
        relocated.append(dict(artifact=f['rel'], sha256=f['sha256'], bytes=f['size'],
                              action_payload=method.inner.base.act.path, embedded_heads=True))
    if len(relocated) != 4:
        raise ValueError('plan must contain exactly four fitted recipe artifacts')
    write_json(HERE / 'relocation_audit.json', dict(PASS=True, artifacts=relocated))
    write_json(HERE / 'deployment.json', dict(PASS=True, run_root=str(CURRENT), arms=names,
        extra_h100_store_bytes=0, new_store_arrays=[], new_run_dependency_bytes=sum(f['size'] for f in new),
        artifact_bytes=sum(f['size'] for f in fits), full_dependency_plan_bytes=standard['bytes'],
        full_dependency_plan_files=len(standard['files']), explicit_head_or_calibration_dependencies=[],
        new_dependencies=new, no_sync_or_launch=True))
    write_json(CURRENT / 'validation/artifact_shas.json',
               [dict(path=str(artifact_path(m, s)), sha256=sha(artifact_path(m, s)),
                     bytes=artifact_path(m, s).stat().st_size) for m, s in CELLS])


def runtime_sources():
    # Enumerate imports and dynamic base-spec imports in a clean subprocess;
    # exclude offline fitting helpers from the serving manifest.
    out = HERE / 'runtime_modules.json'
    code = """import json, pickle, sys
from pathlib import Path
from exp.offline_search.rounds.r10.recipe.recipe import R10Recipe
with Path(sys.argv[1]).open('rb') as f: pickle.load(f)
root = Path(sys.argv[3]) / 'exp/offline_search'
files = sorted({str(Path(m.__file__).resolve()) for name, m in sys.modules.items()
                if name != '__main__' and (getattr(m, '__file__', '') or '').endswith('.py')
                and Path(m.__file__).resolve().is_relative_to(root)})
Path(sys.argv[2]).write_text(json.dumps(files, indent=2) + '\\n')
"""
    script = HERE / 'runtime_modules_helper.py'
    script.write_text(code)
    run_cpu([str(script), str(artifact_path('groot', 'spatial')), str(out), str(REPO)], HERE / 'validation/runtime_modules.log')
    files = [Path(p) for p in json.loads(out.read_text())]
    if any('explore_' in str(p) for p in files):
        raise ValueError('exploration import in runtime source closure')
    return files


def sources():
    runtime = runtime_sources()
    all_new = sorted(HERE.glob('*.py'))
    for filename, paths in [('H100_SOURCES.sha256', runtime), ('ALL_SOURCES.sha256', all_new)]:
        (HERE / filename).write_text(''.join(f'{sha(p)}  {p.relative_to(REPO)}\n' for p in paths))
    write_json(HERE / 'source_list.json', dict(runtime_files=[str(p.relative_to(REPO)) for p in runtime],
        runtime_bytes=sum(p.stat().st_size for p in runtime), new_source_files=[str(p.relative_to(REPO)) for p in all_new],
        new_source_bytes=sum(p.stat().st_size for p in all_new),
        note='H100_SOURCES includes frozen transitive modules; existing deployment sources may already be present'))
    (HERE / 'DOCUMENTS.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(REPO)}\n'
        for p in (HERE / 'README.md', R10 / 'HANDBACK.md')))


def finalize(reuse_plan=False):
    count = snapshot()
    artifacts_audit()
    eq = json.loads((HERE / 'equivalence.json').read_text())
    if eq['cell_sizes'] != 24 or eq['differing_decisions'] != 0:
        raise ValueError('all 24 cell-size equivalence checks must pass')
    from .replay import fitted_check, reference_rows
    fitted = []
    for model, suite in CELLS:
        for size in SIZES:
            for root, row in reference_rows(model, suite, size):
                oldpath = row['plugin_args'][row['plugin_args'].index('--os-fit-artifact') + 1]
                fitted.append(dict(model=model, suite=suite, size=size, reference=str(root),
                    **fitted_check(artifact_path(model, suite, size), oldpath)))
    write_json(HERE / 'fitted_state_audit.json', dict(PASS=True, records=fitted))
    current = json.loads((HERE / 'current_selftests.json').read_text())
    if len(current) != 4 or not all(r['PASS'] for r in current):
        raise ValueError('all current selftests must pass')
    if reuse_plan:
        from exp.offline_search.closed_loop.ops.h100.assets import remap
        standard = json.loads((CURRENT / 'h100_sync/plan.json').read_text())
        local_rows = json.loads((CURRENT / 'arms.json').read_text())
        expected_rows = {row['arm']: remap(row, CURRENT) for row in local_rows}
        if {row['arm']: row for row in standard['arms']} != expected_rows:
            raise ValueError('cached standard plan arms changed')
        files = {f['original']: f for f in standard['files']}
        for model, suite in CELLS:
            path = artifact_path(model, suite)
            if files[str(path)]['source_sha256'] != sha(path):
                raise ValueError('cached standard plan fit changed')
        if not json.loads((HERE / 'relocation_audit.json').read_text())['PASS']:
            raise ValueError('cached relocation audit did not pass')
    else:
        run_cpu(['-m', 'exp.offline_search.rounds.r10.recipe.tests'], HERE / 'tests.log')
        plan()
    sources()
    write_json(HERE / 'final_audit.json', dict(PASS=True, protected_files=count,
        protected_inputs_unchanged=snapshot(), artifacts=28, cell_size_equivalence=24,
        differing_decisions=0, current_arms=4, current_selftest_decisions=sum(r['decisions'] for r in current),
        compared_reference_artifacts=eq['compared_reference_artifacts'],
        guard_calibration_audits=len(fitted),
        compared_decisions=eq['compared_decisions'], forced_equivalence_guard_triggers=eq['forced_guard_triggers'],
        standard_control_plan=True, extra_h100_store_bytes=0, no_sync_or_launch=True))
    print(json.dumps(json.loads((HERE / 'final_audit.json').read_text())), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['snapshot', 'finalize', 'sources'])
    p.add_argument('--reuse-plan', action='store_true', help='verify and reuse the completed standard plan')
    a = p.parse_args()
    if a.action == 'snapshot':
        print(snapshot())
    elif a.action == 'sources':
        sources()
    else:
        finalize(a.reuse_plan)
