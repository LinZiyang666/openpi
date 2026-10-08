"""Stage exact deployed R12 test-A off replicates; optionally install definitions.

No fitting, planning, transport, servers or drivers run from this module.
Installed source roots and all frozen inputs are read-only.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from exp.offline_search.closed_loop import devset
from exp.offline_search.closed_loop.lookhalf_config import validate_row, value
from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import load_manifest

HERE = Path(__file__).resolve().parent
TRACE = Path('/home/weiland/trace_runs/os_closed_loop')
STAGED = HERE / 'test_a_rep_staged'
CELLS = (
    'pi05_l10_50', 'pi05_spatial_50', 'groot_spatial_50', 'groot_l10_50',
    'pi05_spatial_200', 'pi05_l10_50_periodic_ir32',
)


def _json_bytes(data):
    return (json.dumps(data, indent=2) + '\n').encode()


def _rewrite(data, source, target):
    """Move owned paths and root tags, leaving external input paths intact."""
    if isinstance(data, str):
        return data.replace(str(source), str(target)).replace(source.name, target.name)
    if isinstance(data, list):
        return [_rewrite(item, source, target) for item in data]
    if isinstance(data, dict):
        return {_rewrite(key, source, target): _rewrite(val, source, target)
                for key, val in data.items()}
    return data


def _definitions(cell, root, final_root):
    source = TRACE / f'r12_test_{cell}'
    arm = source.name + '_off'
    rows = json.loads((source / 'arms.json').read_text())
    matches = [row for row in rows if row['arm'] == arm]
    if len(matches) != 1:
        raise ValueError(f'exactly one deployed off arm required: {source}')
    row = matches[0]
    if validate_row(row) != 'off' or devset.pool_for(row) != 'A':
        raise ValueError(f'deployed test-A off arm required: {arm}')
    specs = [spec for spec in json.loads((source / 'arms_in.json').read_text())
             if spec['name'] == arm]
    if len(specs) != 1:
        raise ValueError(f'exactly one off input definition required: {arm}')
    record = json.loads((source / 'ROOT.json').read_text())
    frozen = record['frozen_artifacts'][arm]
    if value(row['plugin_args'], '--os-fit-artifact') != frozen['path']:
        raise ValueError(f'off artifact path differs from frozen provenance: {arm}')
    if devset.sha(frozen['path']) != frozen['sha256']:
        raise ValueError(f'frozen off artifact changed: {arm}')
    manifest = source / 'eval500.json'
    if row['manifest'] != str(manifest):
        raise ValueError(f'off manifest must be owned by its deployed root: {arm}')
    selected = load_manifest(manifest)
    devset.check_manifest_pool(selected, 'A')
    if (len(selected['selected']) != 500
            or any(sum(t == task for t, _ in selected['selected']) != 50 for task in range(10))
            or devset.sha(manifest) != record['selection_sha256']):
        raise ValueError(f'frozen eval500 selection changed: {source}')
    configs = {}
    for key, name in (('yaml', arm + '.yaml'), ('matrix', 'matrix_' + arm + '.yaml')):
        path = source / 'config' / name
        if row[key] != str(path):
            raise ValueError(f'config must be owned by its deployed root: {arm} {key}')
        data = path.read_bytes()
        configs['config/' + name.replace(source.name, root.name)] = (
            _rewrite(data.decode(), source, root).encode())
    if devset.sha(row['yaml']) != row['yaml_sha256']:
        raise ValueError(f'frozen off YAML changed: {arm}')
    replica_arm = root.name + '_off'
    replica_record = copy.deepcopy(record)
    replica_record.update(
        root=str(root), final_root=str(final_root), cell=cell,
        arms=[replica_arm], methods=['off'], arm_count=1, episodes_per_arm=500,
        launches=0, pending_B_screen=False, contracts=[],
        source_root=str(source), source_arm=arm, source_selection=str(manifest),
        frozen_artifacts={replica_arm: copy.deepcopy(frozen)},
    )
    return {
        'arms.json': _json_bytes([_rewrite(row, source, root)]),
        'arms_in.json': _json_bytes(_rewrite(specs, source, root)),
        'ROOT.json': _json_bytes(replica_record),
        'eval500.json': manifest.read_bytes(),
        **configs,
    }


def _write(root, definitions):
    root.mkdir(parents=True, exist_ok=False)
    for name, data in definitions.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def prepare(staged_dir=STAGED):
    """Preflight all six sources/destinations, then stage definition files only."""
    staged_dir = Path(staged_dir).resolve()
    catalog = staged_dir / 'roots.json'
    roots = [staged_dir / f'r12_test_rep_{cell}' for cell in CELLS]
    for path in [catalog, *roots]:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f'fresh staging destination required: {path}')
    definitions = [_definitions(cell, root, TRACE / root.name)
                   for cell, root in zip(CELLS, roots, strict=True)]
    records = []
    for root, files in zip(roots, definitions, strict=True):
        _write(root, files)
        records.append(json.loads(files['ROOT.json']))
    catalog.write_bytes(_json_bytes(records))
    return records


def install(destination=TRACE, staged_dir=STAGED):
    """Install only verified definitions; refuse every existing destination."""
    staged_dir, destination = Path(staged_dir).resolve(), Path(destination).resolve()
    records = json.loads((staged_dir / 'roots.json').read_text())
    if [record['cell'] for record in records] != list(CELLS):
        raise ValueError('replicate catalog must contain exactly the six requested cells')
    prepared = []
    for cell, record in zip(CELLS, records, strict=True):
        source = staged_dir / f'r12_test_rep_{cell}'
        target = destination / source.name
        if target.exists() or target.is_symlink():
            raise FileExistsError(f'fresh destination required: {target}')
        if source == target:
            raise ValueError('source and destination must differ')
        expected = _definitions(cell, source, TRACE / source.name)
        if record != json.loads(expected['ROOT.json']):
            raise ValueError(f'replicate catalog differs from deployed off definition: {cell}')
        if {p.name for p in (source / 'config').iterdir()} != {
                Path(name).name for name in expected if name.startswith('config/')}:
            raise ValueError(f'replicate must contain only off configs: {source}')
        for name, data in expected.items():
            if (source / name).read_bytes() != data:
                raise ValueError(f'staged definition differs from deployed off arm: {source / name}')
        files = {}
        for name, data in expected.items():
            if name == 'eval500.json':
                files[name] = data
            elif name == 'ROOT.json':
                installed_record = _rewrite(json.loads(data), source, target)
                installed_record.update(root=str(target), final_root=str(target))
                files[name] = _json_bytes(installed_record)
            else:
                files[name] = _rewrite(data.decode(), source, target).encode()
        prepared.append((target, files))
    for target, files in prepared:
        _write(target, files)
    return [str(target) for target, _ in prepared]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--staged-dir', type=Path, default=STAGED)
    parser.add_argument('--destination', type=Path, default=TRACE)
    args = parser.parse_args()
    result = (install(args.destination, args.staged_dir) if args.install
              else prepare(args.staged_dir))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
