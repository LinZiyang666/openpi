"""Extract compact, CPU-only GR00T replay inputs from read-only server artifacts.

Run on weilandserver with the GR00T environment and the repository on PYTHONPATH.
Writes only to the explicitly selected output directory. It preserves the binding
between the original grid's library/normalizers and its 500-episode outcome table;
W13 libraries are separate datasets and never inherit those outcomes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import torch
import yaml

FIELDS = ('vision_0', 'vision_1', 'robot_state')


def digest(path: Path) -> str:
    """Hash a source without allocating its full bytes a second time."""
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def extract(suite: str, regime: str, output: Path) -> None:
    """Save exact task-block scores, physical action coordinates, and provenance."""
    started = time.perf_counter()
    root = Path('/data/libero_cache/search')/suite
    if regime == 'original':
        config_path = root/'r1/v0@2_v1@2_rs@2.yaml'
    else:
        config_path = Path(f'/tmp/groot_{suite}_template.yaml')
    config = yaml.safe_load(config_path.read_text())
    library = Path(config['backend']['in_memory']['preload_path'])
    assert ('libraries_w13' in str(library)) == (regime == 'w13')
    params = config['checkpoints']['cp1']['search_strategy']['score_normalization']['fields']
    with library.open('rb') as handle:
        artifact = pickle.load(handle)
    entries = artifact['entries']
    names = sorted({e.payload.task_key for e in entries})
    trajectories = sorted({e.trajectory_id for e in entries})
    tm = {x: i for i, x in enumerate(names)}
    trm = {x: i for i, x in enumerate(trajectories)}
    task = np.array([tm[e.payload.task_key] for e in entries])
    traj = np.array([trm[e.trajectory_id] for e in entries])
    step = np.array([e.step_idx for e in entries])
    length = np.array([max(step[traj == t])+1 for t in traj])
    actions = np.stack([np.asarray(e.payload.action_chunk) for e in entries])[:, :5]
    arrays = dict(task=task, traj=traj, step=step, length=length, actions=actions)
    for t in np.unique(task):
        ids = np.flatnonzero(task == t)
        scores = []
        for f in FIELDS:
            x = torch.stack([torch.as_tensor(np.asarray(entries[i].query_keys[f]), dtype=torch.float32) for i in ids])
            if f == 'robot_state':
                raw = -torch.cdist(x, x, p=2)
            else:
                x = torch.nn.functional.normalize(x, dim=1)
                raw = x @ x.T
            p = params[f]['params']
            assert params[f]['method'] == 'zscore' and p['squash'] == 'tanh'
            scores.append((.5*(torch.tanh((raw-p['mu'])/p['sigma'])+1)).numpy())
        arrays[f'ids_{t}'] = ids
        arrays[f'scores_{t}'] = np.stack(scores)
    metadata = dict(suite=suite, regime=regime, library=str(library), library_sha256=digest(library),
                    config_path=str(config_path), config_sha256=digest(config_path),
                    normalizers=params, entries=len(entries), trajectories=len(trajectories),
                    task_names=names, action_layout='first seven of 32 are the LIBERO action coordinates')
    if regime == 'original':
        records = []
        for path in sorted((root/'r1').glob('*.yaml')):
            raw = yaml.safe_load(path.read_text())
            assert raw['backend']['in_memory']['preload_path'] == str(library)
            rows_path = root/'r1_results'/f'{path.stem}.json'
            rows = json.loads(rows_path.read_text())
            keys = [(r['task_id'], r['init_state_idx']) for r in rows]
            assert len(rows) == len(set(keys)) == 500
            records.append(dict(name=path.stem, weights=[raw['keys'].get(f, {}).get('weight', 0) for f in FIELDS],
                                result_sha256=digest(rows_path), result_source=str(rows_path),
                                episodes=[dict(task=r['task_id'], init=r['init_state_idx'],
                                               success=r['success'], seed=r['seed']) for r in rows]))
        assert len(records) == 28
        metadata['closed_loop'] = records
    tag = f'groot_{suite}_{regime}'
    np.savez_compressed(output/f'{tag}.npz', **arrays)
    metadata['seconds'] = time.perf_counter()-started
    (output/f'{tag}.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({k: metadata[k] for k in ['suite','regime','entries','trajectories','seconds','library']}), flush=True)
    del artifact, entries, arrays
    gc.collect()


def main() -> None:
    """Extract both original-grid and W13 inputs with at most two CPU threads."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    for regime in ['original', 'w13']:
        for suite in ['libero_spatial', 'libero_10']:
            extract(suite, regime, args.output)


if __name__ == '__main__':
    main()
