"""Emit local-4090 B-dev roots without sync, servers, workers or chains."""
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
from .build import HERE, CELLS, SPEC

SOURCES = Path('/home/weiland/trace_runs/os_closed_loop')


def prepare(prefix='r12_lookhalf'):
    if not prefix.replace('_', '').isalnum():
        raise ValueError('root prefix must contain only letters, numbers and underscores')
    output = []
    for model, suite, size in CELLS:
        source = SOURCES/f'r11_dev_k10_size{size}'
        base = next(r for r in json.loads((source/'arms.json').read_text())
                    if r['model'] == model and r['suite_short'] == suite)
        root = HERE/'pilots'/f'{prefix}_{model}_{suite}_{size}'
        if root.exists():
            raise FileExistsError(f'fresh root required: {root}')
        root.mkdir(parents=True)
        original = Path(base['manifest'])
        manifest = root/'dev_selection.json'
        manifest.write_bytes(original.read_bytes())
        selected = load_manifest(manifest)
        if len(selected['selected']) != 100 or any(sum(t == task for t,i in selected['selected']) != 10 for task in range(10)):
            raise ValueError('pilot must use exact K10 B selection')
        specs = []
        methods = ['off', 'opus_m1', 'astra_consensus2']
        if model == 'pi05' and size == 50:
            methods.append('astra_strict')
        for method in methods:
            name = f'r12_lh_{model}_{suite}_{size}_{method}'
            artifact = HERE/'artifacts'/f'{model}_{suite}_{size}'/f'{method}.pkl'
            with artifact.open('rb') as f:
                blob = pickle.load(f)
            kwargs = blob['kwargs']
            if kwargs.get('knob_method', 'off') != 'off':
                raise ValueError('pilot knob must be off')
            args = ['--os-root', str(devset.STORE), '--os-no-shadow-native', '--os-tokens', 'off',
                    '--os-blind', '--os-policy-tail', '--os-policy-tail-blocks', '1',
                    '--os-judge', 'guard_only', '--os-fit-artifact', str(artifact), '--os-lookhalf', method,
                    '--os-record-looks', '--os-record-look-images', '56']
            if model == 'groot':
                args += ['--os-groot-path-check-every', '20']
            specs.append(dict(name=name, model=model, suite=suite, mode='plugin', method=SPEC,
                              kwargs=kwargs, plugin_args=args, full_model=True,
                              cost_ledger=True, manifest=str(manifest), client_overrides=base['client_overrides']))
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
        record = dict(root=str(root), model=model, suite=suite, size=size, arms=[r['arm'] for r in rows],
                      episodes_per_arm=100, init_pool='B', knob='off', server_host='local',
                      source_selection=str(original), selection_sha256=devset.sha(manifest),
                      selection_exact_copy=True, contracts=contracts, launches=0,
                      numerical_check_every=20 if model == 'groot' else 0)
        (root/'ROOT.json').write_text(json.dumps(record, indent=2)+'\n')
        output.append(record)
    (HERE/'pilot_roots.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ('root','arms','episodes_per_arm','init_pool','server_host')} for r in output], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', default='r12_lookhalf')
    prepare(parser.parse_args().prefix)


if __name__ == '__main__':
    main()
