"""Snapshot A startup paths and fit hashes only; never read outcomes or apply."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from exp.offline_search.closed_loop.ops.h100.assets import flag
from .analyze import write_outputs
from .analyze_test_a import (AServedFitVerifier, CELLS, HERE, TRACE, Reader,
                             require, sha, verify_protocol)


def decision_paths_hashes(path, verifier):
    """Scan the initial file length, allowing append-only running-arm tails."""
    digest = hashlib.sha256()
    startups = decisions = tail = 0
    verifier.active_path_branch = None
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        remaining = before.st_size
        while remaining:
            line = stream.readline(remaining)
            require(line, f'Decision shard truncated: {path}')
            remaining -= len(line)
            digest.update(line)
            if not line.endswith(b'\n'):
                tail = len(line)
                break
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('ev') == 'startup':
                value = flag(row['argv'], '--os-fit-artifact')
                require(isinstance(value, str), f'Missing startup fit path: {path}')
                verifier.check_path(value)
                startups += 1
            elif row.get('ev') == 'dec':
                value = row.get('r14', {}).get('fit_sha256')
                require(isinstance(value, str), f'Missing served fit hash: {path}')
                verifier.check(value)
                decisions += 1
        after = os.fstat(stream.fileno())
        current = path.stat()
        require(before.st_ino == after.st_ino == current.st_ino and
                after.st_size >= before.st_size and current.st_size >= before.st_size,
                f'Decision shard replaced/truncated: {path}')
    return dict(path=str(path), snapshot_bytes=before.st_size, sha256=digest.hexdigest(),
                startup_paths=startups, decisions_with_fit_hash=decisions, incomplete_tail_bytes=tail)


def audit_root(root, cell, reader, caps, shared=None):
    seal = reader.json(root / 'ROOT.json')
    require(root.parent == TRACE and seal['init_pool'] == 'A' and seal['episodes_per_arm'] == 500,
            'Disallowed A root')
    for filename, digest in seal['definitions_sha256'].items():
        path = (root / filename).resolve()
        require(path.is_relative_to(root.resolve()) and sha(reader.read(path)) == digest,
                'Changed A definition')
    rows = reader.json(root / 'arms.json')
    if shared is None:
        require(root.name == 'r15_descent_test_' + cell and seal['cell'] == cell and
                seal['arm_count'] == len(rows) == 2, 'Wrong R15 A root design')
        table = reader.json(root / 'descent_table.json')
        require(table['caps'] == [caps['cells'][cell]['periodic_cap_by_level'][table['levels'][str(t)]]
                                 for t in range(10)], 'Wrong frozen caps')
        for level, digest in table['source_artifacts'].items():
            require(digest == caps['cells'][cell]['test_fit_sha256_by_level'][level], 'Wrong cap source')
        require({r['arm'] for r in rows} == {
            f'r15dt_{cell}_uniform_' + table['uniform'], f'r15dt_{cell}_task_descent'}, 'Wrong A arm set')
    else:
        require(sha(reader.read(root / 'arms.json')) == shared['arms_sha256'], 'Changed shared C10 definition')
        rows = [r for r in rows if r['arm'] == shared['arm']]
        require(len(rows) == 1, 'Missing/extra shared C10 arm')
    records = []
    for row in rows:
        require(row['init_pool'] == 'A' and row['dev'] is False and row['episodes'] == 500,
                'Wrong A arm metadata')
        fit = Path(flag(row['plugin_args'], '--os-fit-artifact'))
        source = seal['fit_sha256'][str(fit)]
        if shared is not None:
            require(source == shared['fit_sha256'], 'Wrong frozen C10 source')
        elif row['r14_descent_level'] == 'task_descent':
            require(table['fit']['sha256'] == source and Path(table['fit']['path']) == fit and
                    row['kwargs']['task_levels'] == table['levels'] and row['kwargs']['task_caps'] == table['caps'],
                    'Wrong frozen adapter fit or settings')
        else:
            require(row['r14_descent_level'] == table['uniform'] and
                    source == caps['cells'][cell]['test_fit_sha256_by_level'][table['uniform']],
                    'Wrong frozen uniform fit')
        verifier = AServedFitVerifier(root, row, source, reader)
        run = root / 'runs' / row['arm']
        shards = sorted(run.glob('server_*/decisions_*.jsonl'))
        snapshots = [decision_paths_hashes(path, verifier) for path in shards]
        paths_seen = bool(verifier.observed_paths)
        hashes_seen = bool(verifier.observed)
        records.append(dict(arm=row['arm'], root=str(root), cell=cell, model=row['model'],
            shared_reference=shared is not None, frozen_source_matches=True,
            served_paths_match=paths_seen, served_hashes_match=hashes_seen,
            status='verified_observed_paths_and_hashes' if paths_seen and hashes_seen else 'source_verified_observations_pending',
            **verifier.audit(), decision_shard_snapshots=snapshots))
    print('A_FIT_PATH_HASH_PASS', root.name, sum(r['served_paths_match'] and r['served_hashes_match'] for r in records),
          '/', len(records), 'observed arms', flush=True)
    return records


def verify(models=('pi05', 'groot')):
    require(set(os.sched_getaffinity(0)) <= set(range(2, 10)) | set(range(46, 54)), 'Use required CPU affinity')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Hide GPUs')
    reader = Reader()
    p = verify_protocol(reader)
    caps = reader.json(HERE / 'cap_tables.json')
    records = []
    for cell in CELLS:
        if cell.split('_')[0] not in models:
            continue
        records.extend(audit_root(TRACE / ('r15_descent_test_' + cell), cell, reader, caps))
    # References are audited once each, without triple-counting library sizes.
    for key, shared in p['test_references'].items():
        if key.split('_')[0] in models:
            records.extend(audit_root(Path(shared['root']), key + '_reference', reader, caps, shared))
    reader.stable()
    pending = [r['arm'] for r in records if not (r['served_paths_match'] and r['served_hashes_match'])]
    return dict(PASS=True, verified_at_utc=datetime.now(timezone.utc).isoformat(), models=list(models),
        arms=len(records), arms_with_verified_paths_and_hashes=len(records) - len(pending),
        all_frozen_sources_match=True, all_observed_served_paths_and_hashes_match=True,
        all_arms_have_observed_paths_and_hashes=not pending, pending_observations=pending,
        journals_read=False, summaries_read=False, outcome_tables_read=False, apply_run=False,
        completion_not_inferred_from_outcomes=True, inputs=reader.inputs, records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=('pi05', 'groot'), default=['pi05', 'groot'])
    args = parser.parse_args()
    result = verify(args.models)
    write_outputs({'test_a_served_fit_verification.json': result}, HERE)
    print(json.dumps({k: result[k] for k in ('PASS', 'arms', 'arms_with_verified_paths_and_hashes',
        'all_frozen_sources_match', 'all_observed_served_paths_and_hashes_match', 'pending_observations')}))


if __name__ == '__main__':
    main()
