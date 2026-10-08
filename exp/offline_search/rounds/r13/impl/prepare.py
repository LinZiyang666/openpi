"""Build frozen R13 fits, isolated B pilot roots and staged A batch roots.

No outcomes, inference, network, transport or launch are used. Installation is a
separate explicitly invoked command; prepare leaves A roots in the repository.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import pickle
import shutil
from types import SimpleNamespace

from exp.offline_search.closed_loop import devset, look_recording
from exp.offline_search.closed_loop.ops.emit_arms import _emit
from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import load_manifest
from exp.offline_search.rounds.r11.knob.recipe import R11Knob
from exp.offline_search.rounds.r13.knob.recipe import R13Knob

HERE = Path(__file__).resolve().parent
R13 = HERE.parent
ASTRA = R13 / 'astra'
R11 = R13.parent / 'r11/knob'
TRACE = Path('/home/weiland/trace_runs/os_closed_loop')
SPEC = 'exp.offline_search.rounds.r13.knob.recipe:R13Knob'
STAGED = HERE / 'test_a_staged'


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=1, allow_nan=False) + '\n')
    tmp.replace(path)


def key(cell, method, target):
    return f'{cell}:{method}' + ('' if target is None else f':{target}')


def source_artifact(cell, method='off', target=None):
    suffix = '' if target is None else f'_ir{round(100*target):02d}'
    path = R11 / 'artifacts' / f'r11_{cell}_{method}{suffix}.pkl'
    catalog = json.loads((R11 / 'calibration_audit.json').read_text())['records']
    record = next(r for r in catalog if r['artifact'] == str(path))
    if devset.sha(path) != record['sha256']:
        raise ValueError(f'changed frozen R11 fit: {path}')
    return path, record


def build_fits():
    configs = json.loads((ASTRA / 'configs.json').read_text())
    static = json.loads((ASTRA / 'static_pilot_grid.json').read_text())
    frozen = json.loads((ASTRA / 'freeze.json').read_text())
    for filename, digest in frozen['sources_sha256'].items():
        if devset.sha(ASTRA / filename) != digest:
            raise ValueError(f'frozen source SHA changed: {filename}')
    records = []
    cells = list(dict.fromkeys(c['cell'] for c in configs))
    base_blobs = {}
    for cell in cells:
        source, original = source_artifact(cell)
        blob = pickle.loads(source.read_bytes())
        base_blobs[cell] = blob
        records.append(dict(config_key=key(cell, 'off', None), path=str(source), sha256=original['sha256'],
                            spec=blob['spec'], kwargs=blob['kwargs'], reused_R11=True))
    jobs = [c for c in configs if c['method'] in ('distance_feedback', 'periodic')]
    jobs += [dict(c, method='distance_static_b') for c in static]
    jobs += [dict(cell=c, method='pure_c10', target=None, setting=0.) for c in cells]
    for c in jobs:
        cell, method, target = c['cell'], c['method'], c['target']
        model, suite, size = cell.split('_')
        config_key = key(cell, method, target)
        source, source_record = source_artifact(cell)
        if method == 'periodic':
            candidate = R11 / 'artifacts' / f'r11_{cell}_periodic_ir{round(100*target):02d}.pkl'
            if candidate.exists():
                path, record = source_artifact(cell, 'periodic', target)
                blob = pickle.loads(path.read_bytes())
                if blob['method'].knob_settings['setting'] != c['setting']:
                    raise ValueError('frozen periodic setting differs from R11')
                records.append(dict(config_key=config_key, path=str(path), sha256=record['sha256'],
                                    spec=blob['spec'], kwargs=blob['kwargs'], reused_R11=True))
                continue
        path = R13 / 'knob/artifacts' / (config_key.replace(':', '_').replace('.', 'p') + '.pkl')
        if path.exists():
            raise FileExistsError(path)
        # Reuse the complete frozen lower-layer graph, without any refitting.
        blob = copy.deepcopy(base_blobs[cell])
        obj = blob['method']
        obj.__class__ = R13Knob
        obj.knob_method = 'periodic' if method == 'pure_c10' else method
        obj.target = .5 if method == 'pure_c10' else target
        obj._controller = None
        obj._ledger_checkpoint = (0, 0, 0)
        obj._accepted_prefix = ((), ())
        if method == 'distance_feedback':
            settings = {k:c[k] for k in ('lo', 'hi', 'width', 'bias')}
        elif method == 'distance_static_b':
            settings = dict(threshold=c['threshold'], dev_only=True)
        else:
            settings = dict(setting=c['setting'], dose=c['setting'],
                            pred_IR_lib=c.get('pred_ir_library', .5))
        obj.configure(settings, model)
        kwargs = dict(blob['kwargs'], method=method, target=obj.target)
        sources = ('SPEC.md', 'configs.json', 'periodic_fits.json', 'static_pilot_grid.json', 'controller.py')
        manifest = dict(config_key=config_key, settings=settings,
                        settings_sha256=hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                        source_artifact=str(source), source_artifact_sha256=source_record['sha256'],
                        frozen_sources_sha256={s:devset.sha(ASTRA / s) for s in sources},
                        serving_source_sha256=devset.sha(R13 / 'knob/recipe.py'),
                        initialization=('r11_library_schedule' if method in ('periodic','pure_c10') else
                                        'library_only' if int(size) == 500 or cell == 'groot_l10_200' else 'external_B_scalars'),
                        lower_layers_byte_preserved=True, no_refitting=True)
        blob.update(spec=SPEC, kwargs=kwargs, method=obj, r13_manifest=manifest)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.tmp')
        with temp.open('xb') as stream:
            pickle.dump(blob, stream, protocol=4)
        temp.replace(path)
        records.append(dict(config_key=config_key, path=str(path), sha256=devset.sha(path),
                            spec=SPEC, kwargs=kwargs, reused_R11=False, manifest=manifest))
        print('FIT', config_key, flush=True)
    write_json(HERE / 'fits.json', records)
    records += build_pure_c5_fits()
    write_json(HERE / 'fits.json', records)
    return records


def build_pure_c5_fits():
    """Preserve the earlier standard pure-policy proposal graph; seeding is opt-in."""
    from exp.offline_search.harness.baselines import B0Current
    rows = []
    for model in ('pi05', 'groot'):
        for suite in ('l10', 'spatial'):
            if model == 'pi05':
                short = 'sp' if suite == 'spatial' else 'l10'
                source = TRACE / f'r04_cost/fits/r4f_p_{short}_inf_s1001.pkl'
            else:
                source = TRACE / f'r05_q2/fits/r5q2_g_{suite}_policy_L10.pkl'
            blob = pickle.loads(source.read_bytes())
            blob['method'].__class__ = B0Current
            # The ordinary planner walks __dict__ objects. Preserve every B0
            # array byte in a field-equivalent carrier, since its legacy _Block
            # uses slots and otherwise cannot be relocated by that planner.
            blob['method'].blocks = {t:SimpleNamespace(**{k:getattr(b,k) for k in b.__slots__ if hasattr(b,k)})
                                     for t,b in blob['method'].blocks.items()}
            blob.update(spec='exp.offline_search.harness.baselines:B0Current', kwargs={},
                        r13_manifest=dict(source_artifact=str(source), source_artifact_sha256=devset.sha(source),
                                          standard_pure_policy=True, process_seed_flag='--os-rollout-seed'))
            path = R13 / f'knob/artifacts/{model}_{suite}_pure_c5.pkl'
            if path.exists():
                raise FileExistsError(path)
            tmp = path.with_suffix('.tmp')
            with tmp.open('xb') as f:
                pickle.dump(blob, f, protocol=4)
            tmp.replace(path)
            rows.append(dict(config_key=key(f'{model}_{suite}_50', 'pure_c5', None), path=str(path),
                             sha256=devset.sha(path), spec=blob['spec'], kwargs={}, reused_R11=False,
                             manifest=blob['r13_manifest']))
            print('FIT', rows[-1]['config_key'], flush=True)
    return rows


def _spec(arm, cell, method, target, manifest, catalog, *, recording=False, shadow=False, seed=None):
    model, suite, _ = cell.split('_')
    out = dict(name=arm, model=model, suite=suite, mode='plugin', full_model=True,
               cost_ledger=True, client_overrides=dict(replan_steps=5, **({'resize_size':256} if model == 'groot' else {})),
               manifest=str(manifest))
    args = ['--os-root', str(devset.STORE), '--os-no-shadow-native', '--os-tokens', 'off']
    if method == 'pure_c5':
        args += ['--os-fit-artifact', catalog[key(cell, method, None)]['path']]
        out.update(pure_inference=True, plugin_args=args)
    else:
        fit = catalog[key(cell, method, target)]
        args += ['--os-blind', '--os-policy-tail', '--os-policy-tail-blocks', '1',
                 '--os-judge', 'guard_only', '--os-fit-artifact', fit['path']]
        out.update(method=fit['spec'], kwargs=fit['kwargs'], plugin_args=args)
    if recording:
        args += ['--os-record-looks', '--os-record-look-raw-distance']
        if shadow:
            args += ['--os-record-look-shadow', '--os-record-look-cache-proposal']
    if seed is not None:
        args += ['--os-rollout-seed', str(seed)]
    return out


def emit_root(root, specs, metadata, *, pool, final_root=None):
    write_json(root / 'arms_in.json', specs)
    with contextlib.redirect_stdout(io.StringIO()):
        _emit(root, root / 'arms_in.json')
    rows = json.loads((root / 'arms.json').read_text())
    for row in rows:
        row.update(metadata[row['arm']], init_pool=pool)
        if pool == 'B':
            row['dev'] = True
        # Hosts are queue intentions; runtime host is still selected by fleet env.
        if 'server_host' in row:
            raise ValueError('root must not hard-wire the serving host')
    write_json(root / 'arms.json', rows)
    contracts = {}
    if pool == 'B':
        look_recording.validate_root(root)
        for row in rows:
            contract = devset.arm_contract(root, row, load_manifest(row['manifest']))
            look_recording.validate_arm(row, contract)
            contracts[row['arm']] = contract
        write_json(root / 'dev_contracts.json', contracts)
    record = dict(root=str(root), final_root=str(final_root or root), arms=[r['arm'] for r in rows],
                  arm_count=len(rows), episodes_per_arm=100 if pool == 'B' else 500,
                  init_pool=pool, intended_host=rows[0]['intended_host'], launches=0,
                  recording=pool == 'B', staged=pool == 'A', contracts=contracts,
                  definitions_sha256={name:devset.sha(root / name) for name in ('arms.json', 'arms_in.json')},
                  selections_sha256={str(p):devset.sha(p) for p in {Path(r['manifest']) for r in rows}},
                  fit_sha256={f:devset.sha(f) for f in {r['plugin_args'][r['plugin_args'].index('--os-fit-artifact')+1]
                                for r in rows if '--os-fit-artifact' in r['plugin_args']}})
    write_json(root / 'ROOT.json', record)
    return record


def prepare():
    configs = json.loads((ASTRA / 'configs.json').read_text())
    pilot = json.loads((ASTRA / 'pilot_grid.json').read_text())
    pilot += [dict(c, method='distance_static_b', shadow=False, replicate=0) for c in json.loads((ASTRA / 'static_pilot_grid.json').read_text())]
    grid = list(csv.DictReader((ASTRA / 'test_grid.csv').open()))
    cells = list(dict.fromkeys(c['cell'] for c in pilot))
    batches = list(dict.fromkeys(r['batch'] for r in grid))
    for root in [*(TRACE / f'r13_pilot_{c}' for c in cells), *(STAGED / f'r13_test_{b}' for b in batches)]:
        if root.exists():
            raise FileExistsError(f'fresh root required: {root}')
    fits = build_fits()
    catalog = {r['config_key']:r for r in fits}
    selection = json.loads((ASTRA / 'validation_selection.json').read_text())
    used = json.loads((ASTRA / 'used_b_identities.json').read_text())
    if selection['used_identities_sha256'] != devset.sha(ASTRA / 'used_b_identities.json'):
        raise ValueError('validation exclusion SHA changed')
    pilots, tests = [], []
    for cell in cells:
        model, suite, size = cell.split('_')
        root = TRACE / f'r13_pilot_{cell}'
        root.mkdir(parents=True)
        pairs = selection['pairs_by_cell'][cell]
        old_union = {tuple(p) for k, pp in used.items() if k.endswith('_' + suite) for p in pp}
        if len(pairs) != 100 or len({tuple(p) for p in pairs}) != 100 or old_union & {tuple(p) for p in pairs}:
            raise ValueError('validation is not 100 fresh distinct identities')
        if any(sum(t == task for t, _ in pairs) != 10 for task in range(10)):
            raise ValueError('validation must be K=10 per task')
        subset = devset.r10_subset(model, suite, int(size))
        manifest = root / 'dev_selection.json'
        write_json(manifest, dict(dev=True, init_pool='B', model=model, suite=suite, seed=20261004,
                                 per_task=10, pairs=pairs, subset=subset,
                                 frozen_selection_sha256=devset.sha(ASTRA / 'validation_selection.json'),
                                 used_identities_sha256=selection['used_identities_sha256']))
        specs, metadata = [], {}
        for c in (c for c in pilot if c['cell'] == cell):
            specs.append(_spec(c['arm'], cell, c['method'], c['target'], manifest, catalog,
                               recording=True, shadow=c['shadow'], seed=2026100400+c['replicate']))
            metadata[c['arm']] = dict(r13_grid=c, r13_method=c['method'], r10_size=int(size),
                                     intended_host=c['host'], rollout_replicate_seed=2026100400+c['replicate'],
                                     knob_coin_seed=0, target_ir=c['target'])
        pilots.append(emit_root(root, specs, metadata, pool='B'))
        print('PILOT', cell, len(specs), flush=True)
    templates = [TRACE / f'r11_knob_{i}/eval500.json' for i in range(1, 5)]
    if len({p.read_bytes() for p in templates}) != 1:
        raise ValueError('R11 A selections differ')
    selected = load_manifest(templates[0]); devset.check_manifest_pool(selected, 'A')
    if set(selected['selected']) != {(t,i) for t in range(10) for i in range(50)}:
        raise ValueError('R11 A template must include all unchanged 500 pairs')
    for batch in batches:
        root = STAGED / f'r13_test_{batch}'
        root.mkdir(parents=True)
        manifest = root / 'eval500.json'; manifest.write_bytes(templates[0].read_bytes())
        specs, metadata = [], {}
        for r in (r for r in grid if r['batch'] == batch):
            cell = r['cell'] if r['size'] else f'{r["model"]}_{r["suite"]}_50'
            target = float(r['target']) if r['target'] else None
            specs.append(_spec(r['arm'], cell, r['method'], target, manifest, catalog,
                               seed=int(r['rollout_replicate_seed'])))
            metadata[r['arm']] = dict(r13_grid=r, r13_method=r['method'], r10_size=int(r['size']) if r['size'] else None,
                                     intended_host=r['host'], global_order=int(r['global_order']),
                                     rollout_replicate_seed=int(r['rollout_replicate_seed']), knob_coin_seed=0,
                                     target_ir=target, prerequisite=r['prerequisite'])
        tests.append(emit_root(root, specs, metadata, pool='A', final_root=TRACE / root.name))
        print('STAGED', batch, len(specs), flush=True)
    write_json(HERE / 'pilot_roots.json', pilots)
    write_json(HERE / 'test_a_roots.json', tests)
    for name, records in [('queue_pilot.txt', pilots), ('queue_test_a.txt', tests)]:
        (HERE / name).write_text(''.join(f'@{"local" if r["intended_host"] == "local_4090" else "h100"} {r["final_root"]}|{" ".join(r["arms"])}\n' for r in records))
    return dict(pilot_roots=len(pilots), pilot_arms=sum(r['arm_count'] for r in pilots),
                staged_roots=len(tests), staged_arms=sum(r['arm_count'] for r in tests), artifacts=len(fits),
                pilot_episode_runs=6400, test_episode_runs=128000, launches=0)


def replace_owned(value, source, target):
    if isinstance(value, str):
        return str(target)+value[len(str(source)):] if value == str(source) or value.startswith(str(source)+'/') else value
    if isinstance(value, dict):
        return {k:replace_owned(v, source, target) for k,v in value.items()}
    if isinstance(value, list):
        return [replace_owned(v, source, target) for v in value]
    return value


def install_test_a(destination=TRACE):
    records = json.loads((HERE / 'test_a_roots.json').read_text())
    pairs = [(Path(r['root']), Path(destination).resolve()/Path(r['final_root']).name) for r in records]
    for source, target in pairs:
        if target.exists():
            raise FileExistsError(f'fresh destination required: {target}')
        rec = json.loads((source / 'ROOT.json').read_text())
        for file, digest in rec['fit_sha256'].items():
            if devset.sha(file) != digest:
                raise ValueError('changed fit before install')
        for file, digest in rec['definitions_sha256'].items():
            if devset.sha(source / file) != digest:
                raise ValueError('changed staged definition before install')
        for file, digest in rec['selections_sha256'].items():
            if devset.sha(file) != digest:
                raise ValueError('changed A selection before install')
    for source, target in pairs:
        target.mkdir(parents=True)
        for name in ('arms.json', 'arms_in.json', 'ROOT.json'):
            data = replace_owned(json.loads((source / name).read_text()), source, target)
            if name == 'ROOT.json':
                data.update(root=str(target), final_root=str(target), staged=False)
                data['definitions_sha256'] = {n:devset.sha(target / n) for n in ('arms.json', 'arms_in.json')}
                data['selections_sha256'] = {str(target/'eval500.json'):devset.sha(source/'eval500.json')}
            write_json(target / name, data)
        shutil.copyfile(source / 'eval500.json', target / 'eval500.json')
        shutil.copytree(source / 'config', target / 'config')
    return [str(t) for _, t in pairs]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=('prepare', 'install-test-a'))
    opts = ap.parse_args()
    result = prepare() if opts.action == 'prepare' else install_test_a()
    if opts.action == 'prepare':
        write_json(HERE / 'build.json', result)
    print(json.dumps(result), flush=True)
