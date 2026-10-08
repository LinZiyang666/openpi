"""Build and CPU-verify the frozen R13 B-dev recording root; never launch."""
from __future__ import annotations

import argparse
import ast
import copy
import json
import os
from pathlib import Path
import pickle
import sys

import yaml

from exp.offline_search.closed_loop import devset, look_recording
from exp.offline_search.closed_loop.ops.h100 import control
from exp.offline_search.closed_loop.ops.h100.assets import flag
from exp.offline_search.closed_loop.ops.h100.node import sha
from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import load_manifest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
ROOT = Path('/home/weiland/trace_runs/os_closed_loop/r13_devrec_50')
SOURCE = ROOT.with_name('r11_devknob_50')
SELECTION = ROOT.with_name('r11_dev_k10_size50')
KNOB = REPO / 'exp/offline_search/rounds/r11/knob'
CELLS = ('pi05_l10_50', 'pi05_spatial_50', 'groot_l10_50', 'groot_spatial_50')
VARIANTS = ('off_shadow', 'periodic_ir25', 'periodic_ir32', 'periodic_ir40',
            'distance_ir25', 'distance_ir32', 'distance_ir40')
EVIDENCE = HERE / 'evidence'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1) + '\n')


def build():
    if ROOT.exists():
        raise FileExistsError('Fresh root required; use verify for an existing root')
    templates = {r['arm']: r for r in json.loads((SOURCE / 'arms.json').read_text())}
    rows, files = [], {}
    for cell in CELLS:
        model, suite, _ = cell.split('_')
        manifest_name = f'dev_{model}_{suite}.json'
        files[ROOT / manifest_name] = (SELECTION / manifest_name).read_bytes()
        frozen = {r['arm']: r for r in json.loads((KNOB / 'calibration' / f'{cell}.json').read_text())['records']}
        for variant in VARIANTS:
            method = variant.split('_')[0]
            suffix = 'off' if method == 'off' else variant
            template_name = f'r11_devknob_{cell}_{"off" if method == "off" else method + "_ir32"}'
            template = templates[template_name]
            row = copy.deepcopy(template)
            name = f'r13_devrec_{cell}_{variant}'
            record = frozen[f'r11_{cell}_{suffix}']
            artifact = Path(record['artifact'])
            assert sha(artifact) == record['sha256'], artifact
            blob = pickle.loads(artifact.read_bytes())
            assert blob['spec'] == row['method'] and blob['cell'] == row['cell']
            row.update(arm=name, kwargs=blob['kwargs'], manifest=str(ROOT / manifest_name),
                       pred_IR_lib=blob['method'].knob_settings['pred_IR_lib'],
                       r11_method=method, target_ir=blob['kwargs']['target'])
            assert row['pred_IR_lib'] == record['pred_IR_lib']
            args = row['plugin_args']
            args[args.index('--os-fit-artifact') + 1] = str(artifact)
            args += ['--os-record-looks']
            if method == 'off':
                args += ['--os-record-look-shadow']
            for field in ('yaml', 'matrix'):
                old = Path(template[field])
                new = ROOT / 'config' / (f'{name}.yaml' if field == 'yaml' else f'matrix_{name}.yaml')
                content = old.read_text().replace(template_name, name)
                files[new] = content.encode()
                row[field] = str(new)
                row['remote_' + field] = 'os_cl/cfg/' + new.name
            rows.append(row)
    ROOT.mkdir(parents=True)
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    write_json(ROOT / 'arms.json', rows)
    write_json(ROOT / 'protocol.json', dict(
        dev=True, init_pool='B', per_task=10, pairs_per_arm=100, arm_count=len(rows),
        seed=20261004, r10_size=50, library_only=True, fixed_settings=True,
        source_root=str(SOURCE), selection_root=str(SELECTION),
        record_looks=True, record_look_tokens=False, record_look_images=0,
        shadow='off arms only', evaluation_launched=False,
        layer4_task_indexed=False, artifacts='referenced byte-identical frozen R11 fits'))
    verify()


def verify():
    rows = json.loads((ROOT / 'arms.json').read_text())
    templates = {r['arm']: r for r in json.loads((SOURCE / 'arms.json').read_text())}
    assert [r['arm'] for r in rows] == [f'r13_devrec_{c}_{v}' for c in CELLS for v in VARIANTS]
    assert devset.root_pool(ROOT) == 'B'
    look_recording.validate_root(ROOT)
    worker_ast = ast.parse((REPO / 'examples/libero/worker_entry.py').read_text())
    worker_seed = next(ast.literal_eval(k.value) for n in ast.walk(worker_ast)
                       if isinstance(n, ast.Call) and n.args and isinstance(n.args[0], ast.Constant)
                       and n.args[0].value == '--seed' for k in n.keywords if k.arg == 'default')
    assert worker_seed == 7
    driver_ast = ast.parse((REPO / 'exp/gate_threshold_pareto/run_gtp.py').read_text())
    assert not any(k.arg == 'seed' for n in ast.walk(driver_ast) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Name) and n.func.id == 'WorkerSpec' for k in n.keywords)
    contracts, checks = {}, []
    for row in rows:
        name, model, suite = row['arm'], row['model'], row['suite_short']
        cell, method = f'{model}_{suite}_50', row['r11_method']
        template = templates[f'r11_devknob_{cell}_{"off" if method == "off" else method + "_ir32"}']
        args = row['plugin_args']
        source_args = template['plugin_args'].copy()
        artifact = Path(flag(args, '--os-fit-artifact'))
        source_args[source_args.index('--os-fit-artifact') + 1] = str(artifact)
        assert args == source_args + ['--os-record-looks'] + (['--os-record-look-shadow'] if method == 'off' else [])
        for key in ('model', 'suite', 'suite_short', 'cell', 'mode', 'method', 'library',
                    'cost_ledger', 'client_overrides', 'replan_steps', 'full_model', 'judge',
                    'dev', 'init_pool', 'src_yaml', 'src_sha256', 'yaml_sha256', 'r10_size'):
            assert row[key] == template[key], (name, key)
        assert 'server_seed' not in row and 'server_seed' not in template
        assert flag(args, '--os-seed') is None
        manifest = load_manifest(row['manifest'])
        original = SELECTION / Path(row['manifest']).name
        assert Path(row['manifest']).read_bytes() == original.read_bytes()
        assert manifest['data']['seed'] == 20261004 and manifest['data']['per_task'] == 10
        assert len(manifest['selected']) == 100
        assert all(sum(t == task for t, _ in manifest['selected']) == 10 for task in range(10))
        assert control.selection(ROOT, row)[0] == 100
        contract = devset.arm_contract(ROOT, row, manifest)
        assert look_recording.validate_arm(row, contract)
        contracts[name] = contract
        frozen = json.loads((KNOB / 'calibration' / f'{cell}.json').read_text())['records']
        record = next(r for r in frozen if r['artifact'] == str(artifact))
        assert sha(artifact) == record['sha256'] == contract['fit_sha256']
        blob = pickle.loads(artifact.read_bytes())
        assert row['kwargs'] == blob['kwargs'] == record['kwargs']
        assert blob['method'].size == 50 and blob['method'].knob_method == method
        assert blob['method'].target == row['target_ir']
        assert row['pred_IR_lib'] == record['pred_IR_lib'] == blob['method'].knob_settings['pred_IR_lib']
        assert sha(row['yaml']) == row['yaml_sha256'] == sha(template['yaml'])
        matrix = yaml.safe_load(Path(row['matrix']).read_text())
        assert matrix == dict(arms=[dict(arm=name, yaml=row['remote_yaml'], suite=row['suite'],
                                       client_overrides=row['client_overrides'])])
        assert control.execution_env(row) == dict(OSCL_RECORD_LOOKS='1', OSCL_REPLAN_STEPS='5')
        assert '--os-lookhalf' not in args
        spec = control.server_spec(ROOT, row, 23100)
        assert spec['env']['STAGE1_ONLY'] == '0'
        checks.append(dict(arm=name, episodes=100, per_task=10,
                           selection_sha256=manifest['sha256'], manifest_file_sha256=sha(row['manifest']),
                           artifact=str(artifact), artifact_sha256=contract['fit_sha256'],
                           frozen_r11_sha256=record['sha256'], pred_IR_lib=row['pred_IR_lib'],
                           yaml_sha256=sha(row['yaml']), execution_env=control.execution_env(row)))
    write_json(ROOT / 'dev_contracts.json', contracts)
    write_json(EVIDENCE / 'verification.json', dict(
        PASS=True, root=str(ROOT), arms_sha256=sha(ROOT / 'arms.json'), arm_count=len(rows),
        episodes_per_arm=100, total_planned_episodes=100 * len(rows), shadow_arms=len(CELLS),
        selection_byte_identical=True, paired_selection=True, served_subset_disjoint=True,
        frozen_artifact_sha256_match=True, configs_consistent=True,
        recording_contracts_valid=True, execution_audit_required=False,
        common_run_seed=0, common_simulator_seed=worker_seed,
        common_selection_seed=20261004, checks=checks))
    print(f'PASS: {len(rows)} arms; paired K=10 B selection; disjoint; frozen SHA256; configs; recording worker')


def plan(fleet):
    """Exercise the real plan CLI with remote/process launch entry points refused."""
    os.environ.update(SERVER_HOST='local' if fleet == 'local' else 'h100',
                      WORKER_HOST='timan107' if fleet == 'local' else 'timan108')
    for key in ('OSCL_MANIFEST', 'OSCL_EPISODES', 'OSCL_TASKS', 'OSCL_INIT_POOL',
                'OSCL_RECORD_LOOKS', 'OSCL_EXECUTION_AUDIT'):
        os.environ.pop(key, None)
    def forbidden(*args, **kwargs):
        raise AssertionError('Remote or launch operation attempted by configuration-only plan')
    for name in ('run', 'remote', 'push', 'rpc', 'bundle', 'daemon', 'setup', 'chain', 'collect',
                 'fleet_lock', 'chain_lock'):
        setattr(control, name, forbidden)
    rows = json.loads((ROOT / 'arms.json').read_text())
    work = EVIDENCE / f'plan_{fleet}'
    sys.argv = ['control', 'plan', '--work-dir', str(work), str(ROOT), *[r['arm'] for r in rows]]
    control.main()
    result = json.loads((work / 'plan.json').read_text())
    assert len(result['arms']) == len(result['dev_contracts']) == len(rows)
    assert result['init_pool'] == 'B'
    assert not any('/queries/' in f['original'] for f in result['files'])
    fits, looks = [], []
    for source, ready in zip(rows, result['arms'], strict=True):
        assert ready['arm'] == source['arm'] and ready['kwargs'] == source['kwargs']
        assert control.execution_env(ready) == control.execution_env(source)
        artifact = flag(source['plugin_args'], '--os-fit-artifact')
        entry = next(f for f in result['files'] if f['original'] == artifact)
        assert entry['source_sha256'] == sha(artifact)
        fits.append(dict(arm=source['arm'], source_sha256=entry['source_sha256'],
                         planned_sha256=entry['sha256'], relocated=entry['sha256'] != entry['source_sha256']))
        contract_sha = flag(ready['plugin_args'], '--os-look-contract-sha256')
        entry = next(f for f in result['files'] if source['arm'] + ': B recording contract' in f['reasons'])
        assert entry['sha256'] == contract_sha == sha(entry['source'])
        contract = json.loads(Path(entry['source']).read_text())
        assert contract['pairs'] == [list(p) for p in sorted(load_manifest(source['manifest'])['selected'])]
        assert contract['fit_sha256'] == sha(artifact)
        looks.append(dict(arm=source['arm'], contract_sha256=contract_sha))
    write_json(EVIDENCE / f'plan_{fleet}_audit.json', dict(
        PASS=True, fleet=fleet, worker_host=os.environ['WORKER_HOST'],
        server_host=os.environ['SERVER_HOST'], remote_operations=0,
        arm_count=len(rows), look_contract_count=len(rows), file_count=len(result['files']),
        bytes=result['bytes'], plan_sha256=sha(work / 'plan.json'), fits=fits, looks=looks))
    print(f'PASS: {fleet} configuration-only plan; {len(rows)} SHA-bound look contracts; no remote operations')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=('build', 'verify', 'plan-local', 'plan-h100'))
    ap.add_argument('--pi05-spatial-only', action='store_true',
                    help='Use the isolated seven-arm root and separate evidence directory')
    options = ap.parse_args()
    if options.pi05_spatial_only:
        ROOT = ROOT.with_name('r13_devrec_pi05_spatial_50')
        CELLS = ('pi05_spatial_50',)
        EVIDENCE = EVIDENCE / 'pi05_spatial_50'
    action = options.action
    if action == 'build':
        build()
    elif action == 'verify':
        verify()
    else:
        plan(action.removeprefix('plan-'))
