"""Read-only B-pool/R10 identity audit and exact init-file transport manifest."""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

from exp.r17_integration.debug.validate import digest, file_sha

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = Path('/home/weiland/trace_runs/offline_search_store/library')
SUBSETS = ROOT / 'exp/offline_search/rounds/r10/subsets'
SIZES = (50, 100, 200, 300, 400, 500)
SEED = 20261002
SUITES = {'libero_spatial': 'spatial', 'libero_10': 'l10'}


def array_identity(value):
    raw = value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)
    if raw.dtype.hasobject or raw.dtype.kind not in 'biuf' or not np.isfinite(raw).all():
        raise ValueError('unsafe/nonfinite B init state')
    import hashlib
    return dict(dtype=str(raw.dtype), shape=list(raw.shape), state_sha256=hashlib.sha256(raw.tobytes(order='C')).hexdigest())


def task_data_identity(value, canonical_name):
    from exp.r17_integration.benchmarks.spec import digest as task_data_digest
    return task_data_digest(dict(array_identity(value), task_name=canonical_name))


def _load_states(path):
    # Only historical, owner-specified files already checked against pinned SHA.
    import torch
    return torch.load(path, map_location='cpu', weights_only=False)


def audit_alignment(suite, *, reference=REFERENCE, subsets=SUBSETS, pool_record=None,
                    identity_certificate=None, state_loader=_load_states, trace_root=None):
    """Certify all500 coordinates; never infer missing GR indices by row order.

    Input identity is proved by historical file SHA + per-array bytes, not by
    claiming that a post-settle H5 state is the original simulator reset state.
    """
    import yaml
    if suite not in SUITES:
        raise ValueError('only the two owner-selected LIBERO suites are supported')
    short = SUITES[suite]
    reference, subsets = Path(reference), Path(subsets)
    pool_record = Path(pool_record or ROOT / f'exp/offline_search/closed_loop/ops/h100/bpool_{suite}.yaml')
    pool = yaml.safe_load(pool_record.read_text())
    if pool.get('init_pool') != 'B' or pool.get('suite') != suite or pool.get('total_inits') != 500:
        raise ValueError('historical record is not the B500 pool')
    source_dir = Path(pool['apool_dir'])  # Historical field name; init_pool is B.
    pi_path = reference / f'pi05_{short}/bpool_cs/episodes.json'
    gr_path = reference / f'groot_{short}/bpool_all/episodes.json'
    pi, gr = (json.loads(p.read_text()) for p in (pi_path, gr_path))
    if len(pi) != 500 or len(gr) != 500:
        raise ValueError('reference parents must contain all500 episodes including failures')
    pi_by = {}
    for ep in pi:
        task, init, eid = ep['task_id'], ep['orig_init_state_idx'], ep['episode_id']
        if type(init) is not int or not 0 <= task < 10 or not 0 <= init < 50 or eid != task*50+init:
            raise ValueError('pi05 B coordinate/id mismatch')
        if (task, init) in pi_by:
            raise ValueError('duplicate pi05 B coordinate')
        pi_by[task, init] = ep
    certificate = None
    identities = {}
    if any(ep.get('orig_init_state_idx') is None for ep in gr):
        identity_certificate = Path(identity_certificate or ROOT / f'exp/offline_search/rounds/r11/devset/identities/groot_{short}_bpool_all.json')
        certificate = json.loads(identity_certificate.read_text())
        if certificate['episodes_sha256'] != file_sha(gr_path) or len(certificate['identities']) != 500:
            raise ValueError('GR identity certificate is stale/incomplete')
        if certificate.get('rs_sha256') != file_sha(gr_path.parent / 'rs.npy'):
            raise ValueError('GR identity certificate state seal mismatch')
        identities = {r['stem']: r for r in certificate['identities']}
    gr_by = {}
    recovered = 0
    for i, ep in enumerate(gr):
        init = ep.get('orig_init_state_idx')
        if init is None:
            row = identities[ep['stem']]
            if row['episode_index'] != i or row['task_id'] != ep['task_id']:
                raise ValueError('GR identity certificate coordinate mismatch')
            init = row['orig_init_state_idx']; recovered += 1
        if (ep['task_id'], init) not in pi_by or ep['episode_id'] != pi_by[ep['task_id'], init]['episode_id']:
            raise ValueError('GR/pi05 episode alignment mismatch')
        if (ep['task_id'], init) in gr_by:
            raise ValueError('duplicate GR B coordinate')
        gr_by[ep['task_id'], init] = ep
    task_files, states = {}, {}
    for task in range(10):
        prompt = pi_by[task, 0]['task']
        matches = [name for name in pool['per_task_digests'] if
                   re.sub(r'^(?:KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+_', '', name).replace('_', ' ').lower() == prompt.lower()]
        if len(matches) != 1 or any(pi_by[task, j]['task'] != prompt or gr_by[task, j]['task'] != prompt for j in range(50)):
            raise ValueError('historical task/file language alignment mismatch')
        name = matches[0]
        path = source_dir / (name+'.init')
        if file_sha(path) != pool['per_task_digests'][name]:
            raise ValueError('B init file differs from historical input bytes')
        loaded = state_loader(path)
        if len(loaded) != 50:
            raise ValueError('B task must have exactly50 init arrays')
        states[task] = loaded
        task_files[str(task)] = dict(name=name, file=str(path), sha256=file_sha(path), count=50)
    permutations, subset_hashes, subset_seed = None, {}, None
    selections = {}
    for model in ('pi05', 'groot'):
        for size in SIZES:
            path = subsets / f'{model}_{short}_{size}.json'
            document = json.loads(path.read_text())
            order = document['permutation_by_task']
            if document['size'] != size or document['suite'] != short:
                raise ValueError('subset size/suite mismatch')
            if permutations is not None and (order != permutations or document['seed'] != subset_seed):
                raise ValueError('subset seeds/permutations differ across models/sizes')
            subset_seed = document['seed']
            permutations = order
            expected = {str(t): order[str(t)][:size//10] for t in range(10)}
            if document['episode_ids_by_task'] != expected:
                raise ValueError('subset membership mismatch')
            selections[str(size)] = expected
            subset_hashes[str(path)] = file_sha(path)
    rows = []
    trace_root = Path(trace_root) if trace_root else None
    for task in range(10):
        for init in range(50):
            ep = pi_by[task, init]
            row = dict(task_id=task, orig_init_state_idx=init, episode_id=ep['episode_id'],
                pi05_episode_id=ep['episode_id'], groot_episode_id=gr_by[task, init]['episode_id'],
                task_file=task_files[str(task)]['name'], init_array=array_identity(states[task][init]),
                pi05_source_h5=ep['file'], input_file_sha256=task_files[str(task)]['sha256'])
            if trace_root is not None:
                import h5py
                path = trace_root / suite / f'task_{task}/episode_{init}.h5'
                with h5py.File(path, 'r') as h:
                    if (h.attrs['task_id'], h.attrs['orig_init_state_idx'], h.attrs['seed'], h.attrs['num_steps_wait']) != (task, init, 7, 10):
                        raise ValueError('pi05 physical trace reset metadata mismatch')
                row['pi05_reset_metadata_h5'] = str(path)
            rows.append(row)
    result = dict(schema='r17.libero_bpool_alignment.v1', status='PASS', suite=suite, init_pool='B',
        episodes=500, aligned=500, init_arrays=500, original_input_files_byte_identical=True,
        input_proof='historical pinned .init file SHA; same original index; per-array dtype/shape/bytes; transported files must retain SHA',
        historical_h5_contains_full_reset_array=False,
        source_init_dir=str(source_dir), pool_record=str(pool_record), pool_record_sha256=file_sha(pool_record),
        pool_rollup_sha256=pool['rollup_sha256'], task_files=task_files, rows=rows,
        references={str(pi_path):file_sha(pi_path), str(gr_path):file_sha(gr_path)},
        groot_missing_resolved=recovered, identity_certificate_sha256=file_sha(identity_certificate) if certificate else None,
        subset_seed=subset_seed, permutation_by_task=permutations, selections=selections, subset_hashes=subset_hashes,
        failed_episodes_kept=True, official_A_fit_reads=0)
    result['digest'] = digest(result)
    return result


def load_alignment(path):
    document = json.loads(Path(path).read_text())
    if document.get('status') != 'PASS':
        raise ValueError('alignment receipt is partial')
    return document


def main(argv=None):
    import argparse
    from .formal_process import write_json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', required=True, choices=tuple(SUITES))
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--trace-root', type=Path)
    args = p.parse_args(argv)
    document = audit_alignment(args.suite, trace_root=args.trace_root)
    write_json(args.output, document)
    print(json.dumps({k:document[k] for k in ('status','suite','aligned','init_arrays','groot_missing_resolved','digest')}))


if __name__ == '__main__':
    main()
