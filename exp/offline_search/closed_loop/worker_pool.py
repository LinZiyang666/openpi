"""Worker-side pool checks and pool-labelled journals, installed by subset entry.

Only the isolated closed-loop driver is adapted. Shared conductor/LIBERO sources
are unchanged. Validate before importing or starting the conductor's main loop.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

from .devset import check_manifest_pool, pool_record, require_disjoint, sha, suite_long, validate_journal_pool


def arg(args, name, default=None):
    values = []
    for i, value in enumerate(args):
        if value == name:
            values.append(args[i + 1])
        elif value.startswith(name + '='):
            values.append(value.split('=', 1)[1])
    if len(values) > 1:
        raise ValueError(f'duplicate {name} overrides refused')
    return values[0] if values else default


def validate(args, manifest, repo=None):
    from .devset import REPO
    import yaml
    repo = Path(repo or REPO)
    pool = os.environ.get('OSCL_INIT_POOL', 'A')
    if pool not in ('A', 'B'):
        raise ValueError('POOL_MISMATCH: unknown worker init pool')
    suite = suite_long(arg(args, '--task-suite'))
    record_path, expected = pool_record(suite, pool, repo)
    path = Path(arg(args, '--apool-record', str(record_path)))
    record = yaml.safe_load(path.read_text())
    directory = Path(arg(args, '--apool-dir', record['apool_dir'])).resolve()
    canonical = (repo / 'exp/common/data/db_init/libero' / (suite if pool == 'B' else suite + '_apool')).resolve()
    if directory != canonical or record.get('per_task_digests') != expected['per_task_digests'] or record.get('rollup_sha256') != expected['rollup_sha256']:
        raise ValueError('POOL_MISMATCH: driver init path/digest differs from frozen pool')
    if pool == 'B' and (record.get('init_pool') != 'B' or record.get('dev') is not True):
        raise ValueError('POOL_MISMATCH: B driver record must declare dev')
    if pool == 'A' and record.get('init_pool', 'A') != 'A':
        raise ValueError('POOL_MISMATCH: A driver record cannot declare B')
    # The worker loader prefers .pruned_init to .init. Never allow an un-hashed
    # higher-precedence file to bypass run_gtp's .init attestation.
    if list(directory.glob('*.pruned_init')):
        raise ValueError('unattested .pruned_init shadows the frozen .init pool')
    if arg(args, '--init-map') or arg(args, '--init-map-key'):
        raise ValueError('pool index remapping is forbidden in closed-loop roots')
    if manifest:
        check_manifest_pool(manifest, pool)
    contract = None
    if pool == 'B':
        if manifest is None:
            raise ValueError('dev requires an exact B manifest')
        path = Path(os.environ.get('OSCL_POOL_CONTRACT', ''))
        if not path.is_file() or sha(path) != os.environ.get('OSCL_POOL_CONTRACT_SHA256'):
            raise ValueError('dev worker contract missing or SHA mismatch')
        contract = json.loads(path.read_text())
        if contract.get('dev') is not True or contract.get('init_pool') != 'B' or contract.get('suite') != suite or contract.get('manifest_sha256') != manifest['sha256']:
            raise ValueError('POOL_MISMATCH: dev contract differs from worker selection')
        if manifest['data'].get('model') != contract['model']:
            raise ValueError('dev contract model differs from manifest')
        matrix = yaml.safe_load(Path(arg(args, '--arm-matrix')).read_text())
        if [r['arm'] for r in matrix['arms']] != [contract['arm']]:
            raise ValueError('dev contract arm differs from matrix')
        require_disjoint(manifest, contract['subset'])
    journal = Path(arg(args, '--journal'))
    if journal.exists():
        rows = []
        for line in journal.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # same torn-line semantics as conductor
        validate_journal_pool(rows, pool)
    return pool, contract


def install_journal(pool):
    from openpi.conductor import driver
    from openpi.conductor.journal import Journal

    class PoolJournal(Journal):
        def record(self, **kw):
            record = {k: kw[k] for k in ('task_uid', 'yaml_id', 'phase', 'status', 'success')}
            record['ts'] = time.time()
            for key in ('attempt', 'accepted', 'error', 'duration_s', 'run_id'):
                value = kw.get(key)
                if value is not None:
                    record[key] = round(float(value), 3) if key == 'duration_s' else value
            record['init_pool'] = pool
            with self._lock, self._path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()

    driver.Journal = PoolJournal
    return PoolJournal


def install(run_gtp, args, manifest):
    pool, contract = validate(args, manifest)
    # Run the existing attestor against the exact selected file paths before
    # run_gtp can create a driver or workers; run_gtp repeats this at launch.
    from .devset import pool_record
    record_path, record = pool_record(arg(args, '--task-suite'), pool)
    run_gtp.load_apool_digest(str(record_path), required=True, verify_contents=True)
    install_journal(pool)
    print(f'INIT_POOL_OK pool={pool} suite={record["suite"]} sha={record["rollup_sha256"]}', flush=True)
