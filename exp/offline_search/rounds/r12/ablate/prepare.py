"""Emit B screen and staged A pure-cache definitions; install only on request."""
from __future__ import annotations
import argparse
import contextlib
import io
import json
from pathlib import Path
import pickle
import shutil

from exp.offline_search.closed_loop import devset
from exp.offline_search.closed_loop.cache_ablate import ARMS, SPEC
from exp.offline_search.closed_loop.cache_ablate_config import validate_row
from exp.offline_search.closed_loop.ops.emit_arms import _emit
from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import load_manifest
from exp.offline_search.closed_loop.look_recording import validate_root
from exp.offline_search.rounds.r12.prepare_test_a import TRACE, _replace
from .build import HERE, CELLS


def write(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def emit(root, cell, base, source, pool):
    root.mkdir(parents=True, exist_ok=False)
    manifest = root / ('eval500.json' if pool == 'A' else 'dev_selection.json')
    manifest.write_bytes(source.read_bytes())
    selected = load_manifest(manifest)
    devset.check_manifest_pool(selected, pool)
    count = 500 if pool == 'A' else 100
    if len(selected['selected']) != count or any(sum(t == task for t, _ in selected['selected']) != count//10 for task in range(10)):
        raise ValueError('selection has wrong per-task count')
    specs, artifacts = [], {}
    catalog = json.loads((HERE/'artifacts'/cell/'build.json').read_text())['artifacts']
    for arm in ARMS:
        path = HERE / 'artifacts' / cell / f'{arm}.pkl'
        digest = next(r['sha256'] for r in catalog if r['method'] == arm)
        if devset.sha(path) != digest:
            raise ValueError('frozen ablation changed')
        with path.open('rb') as stream:
            blob = pickle.load(stream)
        name = f'{root.name}_{arm}'
        args = ['--os-root', str(devset.STORE), '--os-no-shadow-native', '--os-tokens', 'off',
                '--os-blind', '--os-request-cameras', '--os-fit-artifact', str(path), '--os-cache-ablate', arm,
                '--os-cache-ablate-sha256', digest]
        if pool == 'B':
            args += ['--os-record-looks', '--os-record-look-images', '56']
        specs.append(dict(name=name, model=base['model'], suite=base['suite_short'], mode='plugin',
            method=SPEC, kwargs=blob['kwargs'], plugin_args=args, manifest=str(manifest),
            full_model=True, cost_ledger=True, client_overrides=base['client_overrides']))
        artifacts[name] = dict(path=str(path), sha256=digest)
    write(root/'arms_in.json', specs)
    with contextlib.redirect_stdout(io.StringIO()):
        _emit(root, root/'arms_in.json')
    rows = json.loads((root/'arms.json').read_text())
    for row in rows:
        row.update(init_pool=pool, server_host='local')
        if pool == 'B':
            row['dev'] = True
        validate_row(row)
    write(root/'arms.json', rows)
    if pool == 'B':
        validate_root(root)
    record = dict(root=str(root), final_root=str(TRACE/root.name if pool == 'A' else root), cell=cell,
        arms=[r['arm'] for r in rows], methods=list(ARMS), arm_count=5, init_pool=pool,
        episodes_per_arm=count, source_selection=str(source), selection_sha256=devset.sha(manifest),
        selection_exact_copy=True, frozen_artifacts=artifacts, launches=0)
    write(root/'ROOT.json', record)
    return record


def prepare():
    for model, suite, size in CELLS:
        for group, prefix in (('pilots', 'r12_ablate'), ('test_a_staged', 'r12_ablate_test')):
            path = HERE/group/f'{prefix}_{model}_{suite}_{size}'
            if path.exists():
                raise FileExistsError(f'fresh root required: {path}')
    templates = [TRACE/f'r11_knob_{i}/eval500.json' for i in range(1, 5)]
    if len({p.read_bytes() for p in templates}) != 1:
        raise ValueError('R11 A templates differ')
    test_rows = [row for i in range(1, 5) for row in json.loads((TRACE/f'r11_knob_{i}/arms.json').read_text())]
    B, A = [], []
    for model, suite, size in CELLS:
        cell = f'{model}_{suite}_{size}'
        base = next(r for r in json.loads((TRACE/f'r11_dev_k10_size{size}/arms.json').read_text())
                    if r['model'] == model and r['suite_short'] == suite)
        B.append(emit(HERE/'pilots'/f'r12_ablate_{cell}', cell, base, Path(base['manifest']), 'B'))
        test_base = next(r for r in test_rows if r['model'] == model and r['suite_short'] == suite)
        A.append(emit(HERE/'test_a_staged'/f'r12_ablate_test_{cell}', cell, test_base, templates[0], 'A'))
    write(HERE/'pilot_roots.json', B)
    write(HERE/'test_a_roots.json', A)
    print(json.dumps([dict(root=r['root'], pool=r['init_pool'], arms=5) for r in [*B, *A]], indent=2))


def install_test_a(destination=TRACE):
    records = json.loads((HERE/'test_a_roots.json').read_text())
    pairs = [(Path(r['root']), Path(destination).resolve()/Path(r['final_root']).name) for r in records]
    for source, target in pairs:
        if target.exists() or source.resolve() == target.resolve():
            raise FileExistsError(f'fresh destination required: {target}')
        for fit in json.loads((source/'ROOT.json').read_text())['frozen_artifacts'].values():
            if devset.sha(fit['path']) != fit['sha256']:
                raise ValueError('frozen fit changed since emission')
    for source, target in pairs:
        target.mkdir(parents=True)
        for name in ('arms.json', 'arms_in.json', 'ROOT.json', 'eval500.json'):
            if name == 'eval500.json':
                shutil.copyfile(source/name, target/name)
            else:
                data = _replace(json.loads((source/name).read_text()), source, target)
                if name == 'ROOT.json':
                    data['root'] = data['final_root'] = str(target)
                write(target/name, data)
        shutil.copytree(source/'config', target/'config')
    return [str(target) for _, target in pairs]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--install-test-a', action='store_true')
    p.add_argument('--destination', type=Path, default=TRACE)
    args = p.parse_args()
    if args.install_test_a:
        print(json.dumps(install_test_a(args.destination), indent=2))
    else:
        prepare()


if __name__ == '__main__':
    main()
