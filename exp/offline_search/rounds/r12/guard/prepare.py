"""Emit fifteen K10 B-dev arms using exact R11 selections, with no launches."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import pickle

from exp.offline_search.closed_loop import devset
from exp.offline_search.closed_loop.ops.emit_arms import _emit
from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import load_manifest
from exp.offline_search.closed_loop.look_recording import validate_root
from .build import HERE, PILOT_CELLS, SPEC


def prepare(prefix='r12_guard'):
    """Copy B selections and verify disjoint served-library contracts per arm."""
    if not prefix.replace('_', '').isalnum():
        raise ValueError('root prefix must contain only letters, numbers and underscores')
    output = []
    for model, suite, size, knob in PILOT_CELLS:
        source = Path('/home/weiland/trace_runs/os_closed_loop')/f'r11_dev_k10_size{size}'
        base = next(row for row in json.loads((source/'arms.json').read_text())
                    if row['model'] == model and row['suite_short'] == suite)
        suffix = f'{model}_{suite}_{size}'+('_periodic_ir32' if knob == 'periodic' else '')
        root = HERE/'pilots'/f'{prefix}_{suffix}'
        if root.exists():
            raise FileExistsError(f'fresh pilot root required: {root}')
        root.mkdir(parents=True)
        original = Path(base['manifest'])
        manifest = root/'dev_selection.json'
        manifest.write_bytes(original.read_bytes())
        selected = load_manifest(manifest)
        if len(selected['selected']) != 100 or any(sum(t == task for t, _ in selected['selected']) != 10 for task in range(10)):
            raise ValueError('pilot requires exact K10 B-dev selection')
        specs = []
        for method in ('off', 'h2', 'm2'):
            name = f'r12_guard_{suffix}_{method}'
            artifact = HERE/'artifacts'/suffix/f'{method}.pkl'
            with artifact.open('rb') as stream:
                blob = pickle.load(stream)
            kwargs = blob['kwargs']
            args = ['--os-root', str(devset.STORE), '--os-no-shadow-native', '--os-tokens', 'off',
                    '--os-blind', '--os-policy-tail', '--os-policy-tail-blocks', '1',
                    '--os-judge', 'guard_only', '--os-fit-artifact', str(artifact), '--os-lookhalf', method,
                    '--os-record-looks', '--os-record-look-images', '56']
            if model == 'groot':
                args += ['--os-groot-path-check-every', '20']
            specs.append(dict(name=name, model=model, suite=suite, mode='plugin', method=SPEC,
                kwargs=kwargs, plugin_args=args, full_model=True, cost_ledger=True,
                manifest=str(manifest), client_overrides=base['client_overrides']))
        spec = root/'arms_in.json'
        spec.write_text(json.dumps(specs, indent=2)+'\n')
        with contextlib.redirect_stdout(io.StringIO()):
            _emit(root, spec)
        rows = json.loads((root/'arms.json').read_text())
        for row in rows:
            row.update(dev=True, init_pool='B', server_host='local')
        (root/'arms.json').write_text(json.dumps(rows, indent=2)+'\n')
        validate_root(root)
        contracts = [devset.arm_contract(root, row, selected) for row in rows]
        record = dict(root=str(root), model=model, suite=suite, size=size, knob=knob,
            target=.32 if knob == 'periodic' else None, arms=[r['arm'] for r in rows], episodes_per_arm=100,
            init_pool='B', server_host='local', source_selection=str(original),
            selection_sha256=devset.sha(manifest), selection_exact_copy=True, contracts=contracts, launches=0)
        (root/'ROOT.json').write_text(json.dumps(record, indent=2)+'\n')
        output.append(record)
    (HERE/'pilot_roots.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps([dict(root=r['root'], arms=len(r['arms']), knob=r['knob']) for r in output], indent=2))


def main():
    """Prepare fresh B roots only; never sync or start anything."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', default='r12_guard')
    prepare(parser.parse_args().prefix)


if __name__ == '__main__':
    main()
