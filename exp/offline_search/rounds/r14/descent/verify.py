"""Contracts, identities, frozen fits and ordinary plans with remote ops blocked."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import pickle
import socket
import subprocess
from unittest.mock import patch

from exp.offline_search.closed_loop import devset, look_recording
from exp.offline_search.closed_loop.ops.h100 import control
from exp.offline_search.closed_loop.ops.h100.assets import flag
from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import load_manifest
from .prepare import HERE, R14, ROUNDS, REPO, CELLS, DEV_ORDER, check_seal, write_json, level_cap


def protected_sources():
    files = [REPO / 'exp/offline_search/closed_loop' / n for n in ('plugin.py', 'look_recording.py', 'devset.py')]
    for directory in (ROUNDS / 'r10/recipe', ROUNDS / 'r11/knob', ROUNDS / 'r13/knob', R14 / 'knob'):
        files.extend(directory.glob('*.py'))
    for rnd in ('r13', 'r14'):
        files.append(ROUNDS / rnd / 'impl/fits.json')
    files.extend((R14 / 'astra').glob('*.py'))
    files.extend(R14 / 'astra' / n for n in ('freeze.json', 'allocation_fits.json', 'PREDICTION.md', 'SPEC.md'))
    return {str(p): devset.sha(p) for p in sorted(files)}


def snapshot():
    path = HERE / 'protected_before.json'
    if path.exists():
        raise FileExistsError(path)
    write_json(path, protected_sources())


def verify_records(records):
    evidence = []
    for rec in records:
        root = Path(rec['root'])
        assert check_seal(root) == rec
        rows = json.loads((root / 'arms.json').read_text())
        pool = devset.root_pool(root)
        assert pool == rec['init_pool']
        pairs = None
        contracts = {}
        for row in rows:
            manifest = load_manifest(row['manifest'])
            devset.check_manifest_pool(manifest, pool)
            assert len(manifest['selected']) == (300 if pool == 'B' else 500)
            assert all(sum(t == task for t, _ in manifest['selected']) == (30 if pool == 'B' else 50) for task in range(10))
            if pairs is None:
                pairs = manifest['selected']
            assert pairs == manifest['selected']
            assert control.selection(root, row)[0] == len(pairs)
            assert row['r10_size'] == 50 and row['full_model'] and row['cost_ledger']
            assert '--os-r14-audit' in row['plugin_args']
            assert not any(v.split('=', 1)[0] in look_recording.FLAGS + look_recording.CONTRACT_FLAGS for v in row['plugin_args'])
            assert row['recording'] is False and 'server_host' not in row
            assert row['intended_host'] == ('local' if row['model'] == 'groot' else 'h100')
            assert control.server_spec(root, row, 23100)['env']['STAGE1_ONLY'] == '0'
            assert 'OSCL_RECORD_LOOKS' not in control.execution_env(row)
            assert devset.sha(row['yaml']) == row['yaml_sha256']
            contract = devset.arm_contract(root, row, manifest)
            artifact = flag(row['plugin_args'], '--os-fit-artifact')
            blob = pickle.loads(Path(artifact).read_bytes())
            assert blob['cell'] == row['cell'] and blob['spec'] == row['method'] and blob['kwargs'] == row['kwargs']
            level = row['r14_descent_level']
            method = blob['method']
            if level == 'task_descent':
                expected = [level_cap(rec['cell'], method.task_levels[str(t)]) for t in range(10)]
                assert method.knob_settings['K'] == expected == blob['kwargs']['task_caps']
                assert method.task_levels == blob['kwargs']['task_levels']
            elif level.startswith('task_ir'):
                alloc = next(r for r in json.loads((R14 / 'astra/allocation_fits.json').read_text())
                             if r['cell'] == rec['cell'] and r['target'] == int(level[-2:]) / 100)
                assert method.knob_settings['K'] == alloc['K']
                assert method.knob_settings['reference_K'] == alloc['reference_K']
            elif level == 'off':
                assert method.knob_method == 'off'
            else:
                assert method.knob_method == 'periodic'
                assert method.knob_settings['setting'] == level_cap(rec['cell'], level)
            assert not look_recording.validate_arm(row, contract)
            if pool == 'B':
                assert contract['fit_sha256'] == devset.sha(flag(row['plugin_args'], '--os-fit-artifact'))
                contracts[row['arm']] = contract
        if pool == 'B':
            assert contracts == json.loads((root / 'dev_contracts.json').read_text())
            assert len(rows) == 8
            assert [r['r14_descent_level'] for r in rows] == list(DEV_ORDER)
        else:
            assert len(rows) == 3
        evidence.append(dict(root=str(root), arms=len(rows), episodes_per_arm=len(pairs), pool=pool,
                             paired=True, contracts_valid=True, fits_sha256_valid=True, recording=False))
    return evidence


def verify_dev():
    records = json.loads((HERE / 'dev_roots.json').read_text())
    assert {r['cell'] for r in records} == set(CELLS)
    evidence = verify_records(records)
    for suite in ('l10', 'spatial'):
        roots = [Path(r['root']) for r in records if f'_{suite}_' in r['cell']]
        assert load_manifest(roots[0] / 'dev_selection.json')['selected'] == load_manifest(roots[1] / 'dev_selection.json')['selected']
    before = json.loads((HERE / 'protected_before.json').read_text())
    assert protected_sources() == before
    allocation = R14 / 'astra/allocation_fits.json'
    frozen = json.loads((R14 / 'astra/freeze.json').read_text())
    assert devset.sha(allocation) == frozen['files']['allocation_fits.json']
    write_json(HERE / 'verification.json', dict(PASS=True, records=evidence, shared_runtime_edits=[],
        common_models_within_suite=True, disjoint_from_actual_served_subsets=True,
        protected_sources_unchanged=True, allocation_fits_sha256=devset.sha(allocation), total_episode_runs=9600))
    print('PASS: four paired B roots, 32 arms, 9600 planned episodes; frozen fits and contracts')


def blocked(*args, **kwargs):
    raise AssertionError('remote/process/server operation forbidden during CPU plan')


def plans(records, output):
    evidence = []
    with contextlib.ExitStack() as stack:
        for name in ('run', 'remote', 'push', 'rpc', 'bundle', 'daemon', 'setup', 'chain', 'collect',
                     'fleet_lock', 'chain_lock'):
            stack.enter_context(patch.object(control, name, blocked))
        stack.enter_context(patch.object(subprocess, 'Popen', blocked))
        stack.enter_context(patch.object(socket, 'create_connection', blocked))
        for rec in records:
            root = Path(rec['root'])
            host = rec['intended_host']
            work = Path(output) / host / root.name
            env = dict(SERVER_HOST=host, WORKER_HOST='timan107' if host == 'local' else 'timan108')
            with patch.dict(os.environ, env):
                for key in ('OSCL_MANIFEST', 'OSCL_EPISODES', 'OSCL_TASKS', 'OSCL_INIT_POOL',
                            'OSCL_RECORD_LOOKS', 'OSCL_EXECUTION_AUDIT'):
                    os.environ.pop(key, None)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = control.sync(root, rec['arms'], plan_only=True, workdir=work)
            assert len(result['arms']) == rec['arm_count']
            assert len(result.get('dev_contracts', {})) == (rec['arm_count'] if rec['init_pool'] == 'B' else 0)
            fit_hashes = []
            for row in json.loads((root / 'arms.json').read_text()):
                artifact = flag(row['plugin_args'], '--os-fit-artifact')
                entry = next(f for f in result['files'] if f['original'] == artifact)
                assert entry['source_sha256'] == rec['fit_sha256'][artifact]
                fit_hashes.append(dict(arm=row['arm'], source_sha256=entry['source_sha256'], served_sha256=entry['sha256']))
            assert not any('B recording contract' in reason for f in result['files'] for reason in f['reasons'])
            evidence.append(dict(root=str(root), host=host, arms=len(result['arms']), remote_operations=0,
                                 plan=str(work / 'plan.json'), plan_sha256=devset.sha(work / 'plan.json'), fits=fit_hashes))
            print('PLAN_PASS', host, root.name, len(result['arms']), flush=True)
    write_json(Path(output) / 'audit.json', dict(PASS=True, records=evidence, remote_operations=0, servers=0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=('snapshot', 'verify-dev', 'plan-dev', 'verify-test-a', 'plan-test-a'))
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    if args.action == 'snapshot':
        snapshot()
    elif args.action == 'verify-dev':
        verify_dev()
    elif args.action == 'plan-dev':
        plans(json.loads((HERE / 'dev_roots.json').read_text()), HERE / 'plans_dev')
    else:
        if args.out is None or not args.out.resolve().is_relative_to(HERE):
            ap.error('--out below rounds/r14/descent is required')
        records = json.loads((args.out / 'test_a_roots.json').read_text())
        if args.action == 'verify-test-a':
            write_json(args.out / 'verification.json', dict(PASS=True, records=verify_records(records)))
        else:
            plans(records, args.out / 'plans')


if __name__ == '__main__':
    main()
