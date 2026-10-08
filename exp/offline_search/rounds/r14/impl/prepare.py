"""Emit one B root per cell and stage A roots by model/suite/replicate."""
from __future__ import annotations
import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import shutil
from exp.offline_search.closed_loop import devset, look_recording
from exp.offline_search.closed_loop.ops.emit_arms import _emit
from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import load_manifest
from .common import *

def spec(row, manifest, catalog, *, pilot):
    model, suite, size = row['cell'].split('_')
    target = float(row['target']) if row['target'] else None
    fit = catalog[key(row['cell'], row['variant'], target)]
    seed = int(row.get('rollout_seed', 20261401 + int(row['replicate'])))
    args = ['--os-root', str(STORE), '--os-no-shadow-native', '--os-tokens', 'off',
            '--os-blind', '--os-policy-tail', '--os-policy-tail-blocks', '1',
            '--os-judge', 'guard_only', '--os-fit-artifact', fit['path'],
            '--os-rollout-seed', str(seed), '--os-r14-audit']
    if pilot:
        args += ['--os-record-looks', '--os-record-look-raw-distance']
        if row['part'] == 'A':
            args += ['--os-record-look-shadow', '--os-record-look-cache-proposal']
    return dict(name=row['arm'], model=model, suite=suite, mode='plugin', full_model=True,
        cost_ledger=True, client_overrides=dict(replan_steps=5, **({'resize_size':256} if model == 'groot' else {})),
        manifest=str(manifest), method=fit['spec'], kwargs=fit['kwargs'], plugin_args=args)

def emit(root, specs, grid, *, pilot, final_root):
    write_json(root / 'arms_in.json', specs)
    with contextlib.redirect_stdout(io.StringIO()):
        _emit(root, root / 'arms_in.json')
    rows = json.loads((root / 'arms.json').read_text())
    by_name = {r['arm']:r for r in grid}
    contracts = {}
    for row in rows:
        g = by_name[row['arm']]
        row.update(r14_grid=g, init_pool='B' if pilot else 'A', intended_host=g['host'],
                   r10_size=int(g['cell'].split('_')[2]),
                   rollout_replicate_seed=int(g.get('rollout_seed', 20261401 + int(g['replicate']))),
                   knob_coin_seed=0, target_ir=float(g['target']) if g['target'] else None,
                   global_order=int(g.get('order', grid.index(g))), fleet_exclusive=True)
        if pilot:
            row['dev'] = True
        else:
            row.update(prerequisite='R14 frozen per-cell mechanics and B budget/engagement gates; R13 coordinator release')
        if 'server_host' in row:
            raise ValueError('a root must not bind a worker fleet')
    # Preserve the frozen relative order within each root; the queue records
    # order across batches, while each batch is served by one fleet at a time.
    rows.sort(key=lambda r:r['global_order'])
    write_json(root / 'arms.json', rows)
    if pilot:
        look_recording.validate_root(root)
        for row in rows:
            contract = devset.arm_contract(root, row, load_manifest(row['manifest']))
            look_recording.validate_arm(row, contract)
            contracts[row['arm']] = contract
        write_json(root / 'dev_contracts.json', contracts)
    selections = {Path(r['manifest']) for r in rows}
    fits = {r['plugin_args'][r['plugin_args'].index('--os-fit-artifact') + 1] for r in rows}
    definitions = [p for p in root.rglob('*') if p.is_file() and p.name != 'ROOT.json']
    record = dict(root=str(root), final_root=str(final_root), staged=not pilot,
        arms=[r['arm'] for r in rows], arm_count=len(rows), episodes_per_arm=50 if pilot else 500,
        init_pool='B' if pilot else 'A', intended_host=rows[0]['intended_host'],
        fleet_exclusive=True, launches=0, recording=pilot, contracts=contracts,
        definitions_sha256={str(p.relative_to(root)):sha(p) for p in sorted(definitions)},
        selections_sha256={str(p):sha(p) for p in sorted(selections)},
        fit_sha256={p:sha(p) for p in sorted(fits)})
    write_json(root / 'ROOT.json', record)
    return record

def prepare(resume=False):
    fits = json.loads((HERE / 'fits.json').read_text())
    for r in fits:
        if sha(r['path']) != r['sha256']:
            raise ValueError(f'changed prefit {r["config_key"]}')
    catalog = {r['config_key']:r for r in fits}
    pilots = list(csv.DictReader((ASTRA / 'pilot_grid.csv').open()))
    tests = list(csv.DictReader((ASTRA / 'test_grid.csv').open()))
    selection = {r['cell']:r for r in json.loads((ASTRA / 'pilot_selection.json').read_text())['cells']}
    batch = lambda r:'_'.join([*r['cell'].split('_')[:2], 'rep' + r['replicate']])
    cells = list(dict.fromkeys(r['cell'] for r in pilots))
    batches = list(dict.fromkeys(batch(r) for r in tests))
    destinations = [*(TRACE / f'r14_pilot_{c}' for c in cells), *(STAGED / f'r14_test_{b}' for b in batches)]
    for p in destinations:
        if p.exists():
            if not resume or any((p/name).exists() for name in ('runs','state','h100_sync')):
                raise FileExistsError(f'fresh unserved root required: {p}')
            old=json.loads((p/'arms.json').read_text())
            if not old or any('r14_grid' not in r for r in old):
                raise FileExistsError('resume only owns R14 definitions')
    pilot_roots, test_roots = [], []
    for cell in cells:
        model, suite, n = cell.split('_')
        root = TRACE / f'r14_pilot_{cell}'
        root.mkdir(parents=True,exist_ok=resume)
        sel = selection[cell]
        pairs = sel['pairs']
        if sel['missing'] or len(pairs) != 50 or len({tuple(p) for p in pairs}) != 50:
            raise ValueError('frozen pilot requires fifty distinct consumed B identities')
        if any(sum(p[0] == t for p in pairs) != 5 for t in range(10)):
            raise ValueError('need five pairs per task')
        subset = devset.r10_subset(model, suite, int(n))
        if subset['excluded_inits_by_task'] != sel['excluded_inits_by_task']:
            raise ValueError('served subset differs from frozen pilot exclusion')
        manifest = root / 'dev_selection.json'
        write_json(manifest, dict(dev=True,init_pool='B',model=model,suite=suite,per_task=5,
            seed=20261014,pairs=pairs,subset=subset, selection_sources=sel['sources'],
            frozen_selection_sha256=sha(ASTRA / 'pilot_selection.json')))
        grid = [r for r in pilots if r['cell'] == cell]
        pilot_roots.append(emit(root, [spec(r, manifest, catalog, pilot=True) for r in grid], grid,
                                pilot=True, final_root=root))
    templates = [TRACE / f'r11_knob_{i}/eval500.json' for i in range(1, 5)]
    if len({p.read_bytes() for p in templates}) != 1:
        raise ValueError('R11 eval500 templates differ')
    selected = load_manifest(templates[0])
    devset.check_manifest_pool(selected, 'A')
    if set(selected['selected']) != {(t,i) for t in range(10) for i in range(50)}:
        raise ValueError('A must keep the complete inherited eval500 manifest')
    for b in batches:
        root = STAGED / f'r14_test_{b}'
        root.mkdir(parents=True,exist_ok=resume)
        manifest = root / 'eval500.json'
        atomic(manifest, templates[0].read_bytes())
        grid = [r for r in tests if batch(r) == b]
        test_roots.append(emit(root,[spec(r,manifest,catalog,pilot=False) for r in grid],grid,
                              pilot=False,final_root=TRACE / root.name))
    write_json(HERE / 'pilot_roots.json', pilot_roots)
    write_json(HERE / 'test_a_roots.json', test_roots)
    # A batch root occurs in multiple frozen groups. Reuse that same root in
    # successive queue segments rather than regrouping/shuffling the freeze.
    by_batch={Path(r['final_root']).name.removeprefix('r14_test_'):r for r in test_roots}
    segments=[]
    for row in tests:
        rec=by_batch[batch(row)]
        if not segments or segments[-1]['final_root']!=rec['final_root']:
            segments.append(dict(final_root=rec['final_root'],intended_host=rec['intended_host'],arms=[]))
        segments[-1]['arms'].append(row['arm'])
    write_json(HERE/'test_queue_segments.json',segments)
    for name, records in [('queue_pilot.txt', pilot_roots), ('queue_test_a.txt', segments)]:
        atomic(HERE / name, ''.join(f'@{"local" if r["intended_host"] == "local_4090" else "h100"} {r["final_root"]}|{" ".join(r["arms"])}\n' for r in records).encode())
    result = dict(pilot_roots=len(pilot_roots),pilot_arms=sum(r['arm_count'] for r in pilot_roots),
        pilot_episode_runs=3400, staged_roots=len(test_roots),test_arms=sum(r['arm_count'] for r in test_roots),
        test_episode_runs=64000, launches=0,test_roots_installed=False)
    write_json(HERE / 'build.json', result)
    return result

def replace_owned(value, source, target):
    if isinstance(value, str):
        return str(target) + value[len(str(source)):] if value == str(source) or value.startswith(str(source) + '/') else value
    if isinstance(value, dict):
        return {k:replace_owned(v,source,target) for k,v in value.items()}
    if isinstance(value, list):
        return [replace_owned(v,source,target) for v in value]
    return value

def install_test_a(destination=TRACE):
    records = json.loads((HERE / 'test_a_roots.json').read_text())
    pairs = [(Path(r['root']),Path(destination).resolve() / Path(r['final_root']).name) for r in records]
    for record,(src,dst) in zip(records,pairs):
        if dst.exists():
            raise FileExistsError(f'fresh destination required: {dst}')
        for name,digest in record['definitions_sha256'].items():
            if sha(src / name) != digest:
                raise ValueError(f'changed staged definition {name}')
        for p,digest in record['fit_sha256'].items():
            if sha(p) != digest:
                raise ValueError(f'changed fit {p}')
    for record,(src,dst) in zip(records,pairs):
        # Every staged input is a regular file; rewrite owned absolute paths in
        # JSON and preserve YAML bytes. No installed A root is produced by prepare.
        dst.mkdir(parents=True)
        for p in sorted(src.rglob('*')):
            if p.is_dir():
                (dst / p.relative_to(src)).mkdir(exist_ok=True)
            elif p.is_file() and not p.is_symlink() and p.name != 'ROOT.json':
                target = dst / p.relative_to(src)
                if p.suffix == '.json' and p.name != 'eval500.json':
                    write_json(target,replace_owned(json.loads(p.read_text()),src,dst))
                else:
                    atomic(target,p.read_bytes())
            else:
                if p.name != 'ROOT.json':
                    raise ValueError('only regular staged files may be installed')
        new = replace_owned(record,src,dst)
        new.update(root=str(dst),final_root=str(dst),staged=False,
            definitions_sha256={str(p.relative_to(dst)):sha(p) for p in dst.rglob('*') if p.is_file()})
        write_json(dst / 'ROOT.json',new)
    return [str(dst) for _,dst in pairs]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action',choices=('prepare','install-test-a'))
    ap.add_argument('--resume',action='store_true',help='refresh only owned, unserved R14 definitions')
    a = ap.parse_args()
    print(json.dumps(prepare(a.resume) if a.action == 'prepare' else install_test_a()))
