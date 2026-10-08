"""Emit unused B inits and frozen both/wrist_sub arms. Never launch or fit."""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path

import numpy as np

from exp.offline_search.closed_loop import devset
from exp.offline_search.closed_loop.cache_ablate_config import validate_row
from exp.offline_search.closed_loop.ops.emit_arms import _emit
from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import check_manifest, load_manifest

HERE = Path(__file__).resolve().parent
ABLATE = HERE.parent
CELLS = ('pi05_l10_50', 'pi05_spatial_50', 'pi05_spatial_200',
         'groot_spatial_50', 'groot_l10_50')
ARMS = ('both', 'wrist_sub')
REQUESTED = 40


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + '\n')


def selection(cell, source_record, base):
    model, suite, size = cell.split('_')
    prior_path = Path(source_record['source_selection'])
    if prior_path.parent.name != f'r11_dev_k10_size{size}':
        raise ValueError('camera choice requires the original R11 K10 selection')
    prior = load_manifest(prior_path)
    devset.check_manifest_pool(prior, 'B')
    check_manifest(prior, model, suite)
    if any(sum(t == task for t, _ in prior['selected']) != 10 for task in range(10)):
        raise ValueError('previous screen must contain exactly K10 per task')
    if Path(base['manifest']).read_bytes() != prior_path.read_bytes():
        raise ValueError('existing ablation B screen differs from the R11 K10 selection')
    previous_selections, previous_pairs = [], set()
    for old_size in (50, 200):
        old_root = prior_path.parent.parent / f'r11_dev_k10_size{old_size}'
        old_rows = json.loads((old_root / 'arms.json').read_text())
        paths = {r['manifest'] for r in old_rows
                 if r['model'] == model and r['suite_short'] == suite}
        if len(paths) != 1:
            raise ValueError(f'{old_root}: need one original K10 selection for {model}/{suite}')
        old_path = Path(paths.pop())
        if old_path.parent != old_root:
            raise ValueError('prior K10 manifest escaped its original root')
        old = load_manifest(old_path)
        devset.check_manifest_pool(old, 'B')
        check_manifest(old, model, suite)
        if any(sum(t == task for t, _ in old['selected']) != 10 for task in range(10)):
            raise ValueError('both previous screens must contain exactly K10 per task')
        previous_pairs.update(old['selected'])
        previous_selections.append(dict(path=str(old_path), file_sha256=devset.sha(old_path),
                                        selection_sha256=old['sha256'],
                                        pairs=[list(p) for p in sorted(old['selected'])]))
    subset = devset.r10_subset(model, suite, int(size))
    served = devset.subset_record(model, suite, base['kwargs']['library'],
                                 base['kwargs']['episode_subset'])
    if subset != served:
        raise ValueError('R10 subset differs from the served frozen artifact selection')
    devset.require_disjoint(prior, subset)
    capacities = [50 - len(subset['excluded_inits_by_task'][str(t)]) for t in range(10)]
    # These frozen R10 cells select exactly size/10 distinct library inits per
    # task. Draw their complete held-out complement with the existing selector,
    # including its PCG64(seed + task) convention, then remove BOTH prior K10s.
    if len(set(capacities)) != 1:
        raise ValueError('frozen R10 cell has an unbalanced library selection')
    seed = prior['data']['seed']
    data = devset.build_manifest(model, suite, r10_size=int(size),
                                 per_task=capacities[0], seed=seed)
    pairs, counts, available_by_task = [], {}, {}
    for task in range(10):
        available = sorted(i for t, i in data['pairs']
                           if t == task and (t, i) not in previous_pairs)
        expected = sorted(set(range(50)) - set(subset['excluded_inits_by_task'][str(task)])
                          - {i for t, i in previous_pairs if t == task})
        if available != expected or not available:
            raise ValueError(f'task {task}: incomplete or empty unused B complement')
        count = min(REQUESTED, len(available))
        # Currently all cells fall below K40, so every remaining init is used.
        # If more are available, retain devset's exact per-task sampling rule.
        rng = np.random.Generator(np.random.PCG64(seed + task))
        pairs.extend([[task, int(i)] for i in sorted(rng.choice(available, count, replace=False))])
        counts[str(task)] = count
        available_by_task[str(task)] = available
    data.update(pairs=pairs, per_task=next(iter(counts.values())) if len(set(counts.values())) == 1 else None,
                requested_per_task=REQUESTED, per_task_counts=counts,
                selection_rule='devset.build_manifest full B library complement; exclude union of R11 size50 AND size200 K10 pairs; PCG64(seed+task); min(40, available)',
                previous_selection=dict(path=str(prior_path), file_sha256=devset.sha(prior_path),
                                        selection_sha256=prior['sha256'],
                                        pairs=[list(p) for p in sorted(prior['selected'])]),
                previous_selections=previous_selections,
                excluded_K10_pairs=[list(p) for p in sorted(previous_pairs)],
                available_inits_by_task=available_by_task,
                a_used=False)
    return data


def emit(root, cell, source_record):
    source = Path(source_record['root'])
    originals = json.loads((source / 'arms.json').read_text())
    base = next(r for r in originals if r['kwargs']['method'] == 'both')
    data = selection(cell, source_record, base)
    specs, artifacts = [], {}
    catalog = json.loads((ABLATE / 'artifacts' / cell / 'build.json').read_text())['artifacts']
    for arm in ARMS:
        original = next(r for r in originals if r['kwargs']['method'] == arm)
        name = f'{root.name}_{arm}'
        frozen = source_record['frozen_artifacts'][original['arm']]
        expected = next(r['sha256'] for r in catalog if r['method'] == arm)
        if expected != frozen['sha256'] or devset.sha(frozen['path']) != expected:
            raise ValueError('existing frozen ablation artifact changed')
        args = copy.deepcopy(original['plugin_args'])
        args.remove('--os-record-looks')
        index = args.index('--os-record-look-images')
        del args[index:index + 2]
        if any(a.startswith(('--os-record-look', '--os-look-contract')) for a in args):
            raise ValueError('camera dev recording must be off')
        specs.append(dict(name=name, model=original['model'], suite=original['suite_short'],
                          mode=original['mode'], method=original['method'], kwargs=original['kwargs'],
                          plugin_args=args, manifest=str(root / 'dev_selection.json'),
                          full_model=original['full_model'], cost_ledger=True,
                          client_overrides=original['client_overrides']))
        artifacts[name] = dict(path=frozen['path'], sha256=expected)
    root.mkdir(parents=True, exist_ok=False)
    manifest_path = root / 'dev_selection.json'
    write(manifest_path, data)
    manifest = load_manifest(manifest_path)
    devset.check_manifest_pool(manifest, 'B')
    devset.require_disjoint(manifest, data['subset'])
    if set(manifest['selected']) & {tuple(p) for p in data['excluded_K10_pairs']}:
        raise ValueError('camera dev overlaps a previous K10 screen')
    write(root / 'arms_in.json', specs)
    with contextlib.redirect_stdout(io.StringIO()):
        _emit(root, root / 'arms_in.json')
    rows = json.loads((root / 'arms.json').read_text())
    for row in rows:
        row.update(dev=True, init_pool='B', server_host='local')
        validate_row(row)
    write(root / 'arms.json', rows)
    for row in rows:
        devset.arm_contract(root, row, manifest)
    record = dict(root=str(root), cell=cell, source_root=str(source),
                  arms=[r['arm'] for r in rows], methods=list(ARMS), arm_count=2,
                  dev=True, init_pool='B', requested_per_task=REQUESTED,
                  per_task_counts=data['per_task_counts'], episodes_per_arm=len(manifest['selected']),
                  selection_sha256=manifest['sha256'], selection_file_sha256=devset.sha(manifest_path),
                  previous_selection=data['previous_selection'],
                  previous_selections=data['previous_selections'], excluded_K10_pairs=data['excluded_K10_pairs'],
                  library_subset=data['subset'], available_inits_by_task=data['available_inits_by_task'],
                  prior_overlap=[], library_overlap=[], a_used=False,
                  frozen_artifacts=artifacts, refits=0, look_recording=False,
                  cost_ledger=True, per_episode_journal=True,
                  execution_worker='exp.offline_search.closed_loop.execution_worker',
                  engagement_requirements={f'{root.name}_{arm}': dict(calls=0, **{
                      'one_camera_looks' if arm == 'both' else 'two_camera_looks': 0}) for arm in ARMS},
                  launches=0, remote_operations=0)
    write(root / 'ROOT.json', record)
    return record


def prepare(destination=HERE):
    destination = Path(destination).resolve()
    sources = {r['cell']: r for r in json.loads((ABLATE / 'pilot_roots.json').read_text())}
    for cell in CELLS:
        path = destination / f'r12_camdev_{cell}'
        if path.exists():
            raise FileExistsError(f'fresh root required: {path}')
    records = [emit(destination / f'r12_camdev_{cell}', cell, sources[cell]) for cell in CELLS]
    write(destination / 'roots.json', records)
    print(json.dumps([dict(root=r['root'], per_task=r['per_task_counts'],
                           episodes_per_arm=r['episodes_per_arm']) for r in records], indent=2))
    return records


if __name__ == '__main__':
    prepare()
