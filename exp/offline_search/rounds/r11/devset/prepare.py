"""Emit fresh JSON-only B-dev roots from completed R10Recipe prefits."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from exp.offline_search.closed_loop import devset
from exp.offline_search.closed_loop.ops.h100 import control
from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import load_manifest

HERE = Path(__file__).resolve().parent
SOURCE = Path('/home/weiland/trace_runs/os_closed_loop/r10_recipe_current')


def prepare(output, *, r10_size=None, library=None, per_task=5, seed=20261003):
    output = Path(output).resolve()
    if output.parent != SOURCE.parent or not output.name.startswith('r11_dev_'):
        raise ValueError('dev example roots must be /home/weiland/trace_runs/os_closed_loop/r11_dev_*')
    if output.exists():
        raise FileExistsError('fresh root required')
    rows, files = [], {}
    for template in json.loads((SOURCE / 'arms.json').read_text()):
        row = copy.deepcopy(template)
        model, suite = row['model'], devset.suite_short(row['suite'])
        manifest = devset.build_manifest(model, suite, r10_size=r10_size, library=library, per_task=per_task, seed=seed)
        tag = str(r10_size) if r10_size is not None else library
        name = f'r11_dev_{model}_{suite}_{tag}'
        manifest_path = output / f'dev_{model}_{suite}.json'
        old_name = row['arm']
        row.update(arm=name, dev=True, init_pool='B', manifest=str(manifest_path),
                   remote_yaml=f'os_cl/cfg/{name}.yaml', remote_matrix=f'os_cl/cfg/matrix_{name}.yaml')
        if r10_size is not None:
            subset = manifest['subset']
            row['kwargs'] = dict(library=subset['library'], episode_subset=subset['episode_ids_by_task'])
            fit = devset.REPO / 'exp/offline_search/rounds/r10/recipe/artifacts' / f'r10_recipe_{model}_{suite}_{r10_size}.pkl'
            index = row['plugin_args'].index('--os-fit-artifact')
            row['plugin_args'][index + 1] = str(fit)
        elif library != 'current':
            raise ValueError('prepare supports deployed current or completed --r10-size prefits')
        matrix = yaml.safe_load(Path(row['matrix']).read_text())
        matrix = json.loads(json.dumps(matrix).replace(old_name, name))
        row['matrix'] = str(output / f'matrix_{name}.json')
        files[Path(row['matrix'])] = matrix
        files[manifest_path] = manifest
        rows.append(row)
    output.mkdir()
    (output / 'arms.json').write_text(json.dumps(rows, indent=1) + '\n')
    for p, data in files.items():
        p.write_text(json.dumps(data, indent=1) + '\n')
    contracts, references = {}, {}
    for row in rows:
        manifest = load_manifest(row['manifest'])
        contracts[row['arm']] = devset.arm_contract(output, row, manifest)
        references[row['arm']] = devset.pure_reference(manifest)
        control.selection(output, row)
    (output / 'dev_contracts.json').write_text(json.dumps(contracts, indent=1) + '\n')
    (output / 'pure_policy_reference.json').write_text(json.dumps(references, indent=1) + '\n')
    (output / 'protocol.json').write_text(json.dumps(dict(dev=True, init_pool='B', per_task=per_task, seed=seed,
        r10_size=r10_size, library=library, fits='completed R10Recipe artifacts; no refitting',
        evaluation_launched=False, source_root=str(SOURCE)), indent=1) + '\n')
    print(f'DEV_ROOT_READY root={output} arms={len(rows)} pairs_per_arm={10*per_task}', flush=True)
    for name, ref in references.items():
        print(f'DEV_REFERENCE {name} success={ref["success"]}/{ref["complete"]} sr={ref["sr"]:.4f}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--library', choices=['current'])
    g.add_argument('--r10-size', type=int, choices=(50, 100, 200, 300, 400, 500))
    p.add_argument('--per-task', type=int, default=5)
    p.add_argument('--seed', type=int, default=20261003)
    a = p.parse_args()
    prepare(a.output, r10_size=a.r10_size, library=a.library, per_task=a.per_task, seed=a.seed)


if __name__ == '__main__':
    main()
