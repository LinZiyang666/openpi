"""Emit R12 A roots and the Spatial-200 B screen from frozen pilot artifacts.

No fitting, calibration, planning, transport or launches occur here. A roots can
be staged in the repository and installed later when trace_runs is writable.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import pickle
import shutil

from exp.offline_search.closed_loop import devset
from exp.offline_search.closed_loop.look_recording import validate_root
from exp.offline_search.closed_loop.lookhalf_config import validate_row, value
from exp.offline_search.closed_loop.ops.emit_arms import _emit
from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import load_manifest

HERE = Path(__file__).resolve().parent
TRACE = Path('/home/weiland/trace_runs/os_closed_loop')
STAGED = HERE / 'test_a_staged'
CELLS = (
    ('pi05_l10_50', ('off', 'opus_m1', 'h2', 'm2'), False),
    ('pi05_spatial_50', ('off', 'opus_m1', 'h2', 'm2'), False),
    ('groot_spatial_50', ('off', 'opus_m1', 'h2', 'm2'), False),
    ('groot_l10_50', ('off', 'opus_m1', 'h2', 'm2'), False),
    ('pi05_spatial_200', ('off', 'opus_m1'), False),
    ('pi05_spatial_200_guard_pending', ('h2', 'm2'), True),
    ('pi05_l10_50_periodic_ir32', ('off', 'h2', 'm2'), False),
)


def _json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def _strip_recording(args):
    remove = {'--os-record-looks': 0, '--os-record-look-images': 1,
              '--os-groot-path-check-every': 1}
    out, skip = [], 0
    for item in args:
        if skip:
            skip -= 1
        elif item.split('=', 1)[0] in remove:
            skip = remove[item] if '=' not in item else 0
        else:
            out.append(item)
    return out


def _frozen(group, cell, selected):
    artifact = HERE / group / 'artifacts' / cell / f'{selected}.pkl'
    catalog = json.loads((artifact.parent / 'build.json').read_text())['artifacts']
    expected = next(r['sha256'] for r in catalog if r['method'] == selected)
    if devset.sha(artifact) != expected:
        raise ValueError(f'frozen artifact changed: {artifact}')
    return artifact, expected


def _pilot_row(cell, selected):
    group = 'guard' if selected in ('h2', 'm2') or 'periodic' in cell else 'lookhalf'
    pilot = HERE / group / 'pilots' / f'r12_{group}_fix1_{cell}' / 'arms.json'
    artifact, expected = _frozen(group, cell, selected)
    if pilot.exists():
        row = next(r for r in json.loads(pilot.read_text()) if r['kwargs']['method'] == selected)
        if value(row['plugin_args'], '--os-fit-artifact') != str(artifact):
            raise ValueError('pilot does not reference the exact frozen artifact')
        return copy.deepcopy(row), artifact, expected, str(pilot)
    if (group, cell) != ('guard', 'pi05_spatial_200'):
        raise FileNotFoundError(pilot)
    # These two already-frozen fits are the only methods without fix1 pilots.
    with artifact.open('rb') as stream:
        blob = pickle.load(stream)
    row = _pilot_row(cell, 'off')[0]
    row.update(method=blob['spec'], kwargs=blob['kwargs'])
    args = row['plugin_args']
    args[args.index('--os-fit-artifact') + 1] = str(artifact)
    args[args.index('--os-lookhalf') + 1] = selected
    validate_row(row)
    return row, artifact, expected, str(artifact.parent / 'build.json')


def _emit_root(root, cell, methods, manifest_source, pool, *, pending=False, final_root=None):
    root.mkdir(parents=True, exist_ok=False)
    manifest = root / ('eval500.json' if pool == 'A' else 'dev_selection.json')
    manifest.write_bytes(manifest_source.read_bytes())
    selected = load_manifest(manifest)
    devset.check_manifest_pool(selected, pool)
    pairs = selected['selected']
    count = 500 if pool == 'A' else 100
    if len(pairs) != count or any(sum(t == task for t, _ in pairs) != count // 10 for task in range(10)):
        raise ValueError('selection must have 50 A or 10 B inits for every task')
    specs, artifacts = [], {}
    for method in methods:
        pilot, artifact, digest, provenance = _pilot_row(cell, method)
        args = pilot['plugin_args'] if pool == 'B' else _strip_recording(pilot['plugin_args'])
        arm = f'{root.name}_{method}'
        specs.append(dict(name=arm, model=pilot['model'], suite=pilot['suite_short'], mode='plugin',
                          method=pilot['method'], kwargs=pilot['kwargs'], plugin_args=args,
                          manifest=str(manifest), full_model=True, cost_ledger=True,
                          client_overrides=pilot['client_overrides']))
        artifacts[arm] = dict(path=str(artifact), sha256=digest, source=provenance)
    _json(root / 'arms_in.json', specs)
    with contextlib.redirect_stdout(io.StringIO()):
        _emit(root, root / 'arms_in.json')
    rows = json.loads((root / 'arms.json').read_text())
    for row in rows:
        row.update(init_pool=pool, server_host='local')
        if pool == 'B':
            row['dev'] = True
        validate_row(row)
    _json(root / 'arms.json', rows)
    if pool == 'B':
        validate_root(root)
    contracts = [devset.arm_contract(root, row, selected) for row in rows] if pool == 'B' else []
    record = dict(root=str(root), final_root=str(final_root or root), cell=cell,
                  arms=[r['arm'] for r in rows], methods=list(methods), arm_count=len(rows),
                  episodes_per_arm=count, init_pool=pool, launches=0,
                  pending_B_screen=pending, source_selection=str(manifest_source),
                  selection_sha256=devset.sha(manifest), selection_exact_copy=True,
                  frozen_artifacts=artifacts, contracts=contracts)
    _json(root / 'ROOT.json', record)
    return record


def prepare(test_dir=STAGED):
    """Emit seven fresh A roots and one fresh B screening root, without launches."""
    test_dir = Path(test_dir).resolve()
    roots = [test_dir / f'r12_test_{cell}' for cell, _, _ in CELLS]
    b_root = HERE / 'guard/pilots/r12_guard_fix2_pi05_spatial_200'
    for root in [*roots, b_root]:
        if root.exists():
            raise FileExistsError(f'fresh root required: {root}')
    manifests = [TRACE / f'r11_knob_{i}/eval500.json' for i in range(1, 5)]
    if len({p.read_bytes() for p in manifests}) != 1:
        raise ValueError('R11 A selection templates differ')
    templates = [row for i in range(1, 5)
                 for row in json.loads((TRACE / f'r11_knob_{i}/arms.json').read_text())]
    records = []
    for (cell, methods, pending), root in zip(CELLS, roots, strict=True):
        fit_cell = cell.removesuffix('_guard_pending')
        record = _emit_root(root, fit_cell, methods, manifests[0], 'A', pending=pending,
                            final_root=TRACE / root.name)
        # Template yaml/client construction is independent of the library size.
        for row in json.loads((root / 'arms.json').read_text()):
            template = next(r for r in templates if r['model'] == row['model'] and r['suite'] == row['suite'])
            for key in ('src_yaml', 'src_sha256', 'yaml_sha256', 'client_overrides'):
                if row[key] != template[key]:
                    raise ValueError(f'R11 template mismatch: {row["arm"]} {key}')
        records.append(record)
    base = next(r for r in json.loads((TRACE / 'r11_dev_k10_size200/arms.json').read_text())
                if r['model'] == 'pi05' and r['suite_short'] == 'spatial')
    b_record = _emit_root(b_root, 'pi05_spatial_200', ('h2', 'm2'), Path(base['manifest']), 'B')
    _json(HERE / 'test_a_roots.json', records)
    _json(HERE / 'guard/fix2_roots.json', [b_record])
    return records, b_record


def _replace(value, source, target):
    if isinstance(value, str):
        return str(target) + value[len(str(source)):] if value == str(source) or value.startswith(str(source) + '/') else value
    if isinstance(value, dict):
        return {k: _replace(v, source, target) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace(v, source, target) for v in value]
    return value


def install_test_a(destination=TRACE):
    """Copy staged root definitions to fresh final roots, rewriting owned paths only."""
    records = json.loads((HERE / 'test_a_roots.json').read_text())
    pairs = [(Path(r['root']), Path(destination).resolve() / Path(r['final_root']).name) for r in records]
    for source, target in pairs:
        if target.exists():
            raise FileExistsError(f'fresh destination required: {target}')
        if source.resolve() == target.resolve():
            raise ValueError('source and destination must differ')
        for artifact in json.loads((source / 'ROOT.json').read_text())['frozen_artifacts'].values():
            if devset.sha(artifact['path']) != artifact['sha256']:
                raise ValueError('frozen artifact changed since emission')
    for source, target in pairs:
        target.mkdir(parents=True)
        # Only definitions are copied. Test scratch and coordinator run state
        # can never be carried into a new run by this installer.
        for name in ('arms.json', 'arms_in.json', 'ROOT.json', 'eval500.json'):
            if name == 'eval500.json':
                shutil.copyfile(source / name, target / name)
            else:
                data = _replace(json.loads((source / name).read_text()), source, target)
                if name == 'ROOT.json':
                    data['root'] = data['final_root'] = str(target)
                _json(target / name, data)
        shutil.copytree(source / 'config', target / 'config')
    return [str(target) for _, target in pairs]


def main():
    """Prepare definitions, or install staged A definitions; never launch anything."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-dir', type=Path, default=STAGED)
    parser.add_argument('--install-test-a', action='store_true')
    parser.add_argument('--destination', type=Path, default=TRACE)
    args = parser.parse_args()
    if args.install_test_a:
        print(json.dumps(install_test_a(args.destination), indent=2))
    else:
        records, b_record = prepare(args.test_dir)
        print(json.dumps([dict(root=r['root'], arms=r['arm_count'], pool=r['init_pool'])
                          for r in [*records, b_record]], indent=2))


if __name__ == '__main__':
    main()
