"""Certify missing init metadata using B-library evidence; never edit libraries."""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

from exp.offline_search.closed_loop.devset import STORE, sha

HERE = Path(__file__).resolve().parent


def write_identity(model, suite, library, eps, values, evidence):
    d = STORE / 'library' / f'{model}_{suite}' / library
    rows = []
    for idx, (e, init) in enumerate(zip(eps, values, strict=True)):
        if e.get('orig_init_state_idx') not in (None, init):
            raise ValueError('identity recovery conflicts with known metadata')
        rows.append(dict(episode_index=idx, stem=e['stem'], task_id=e['task_id'], orig_init_state_idx=int(init)))
    result = dict(model=model, suite=suite, library=library, episodes_sha256=sha(d / 'episodes.json'),
                  rs_sha256=sha(d / 'rs.npy'), identities=rows, evidence=evidence,
                  missing_original=sum(e.get('orig_init_state_idx') is None for e in eps))
    out = HERE / 'identities' / f'{model}_{suite}_{library}.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + '\n')
    print('IDENTITY_CERTIFIED',model,suite,library,'episodes='+str(len(eps)),'recovered='+str(result['missing_original']),flush=True)


def main():
    # GR00T's collection uses the explicit global episode ID formula in the
    # shared client. Validate all surviving coordinates, the exact stem, all
    # 500 coordinates and the source dataset binding before recovering nulls.
    formula = Path('examples/libero/collect_util.py')
    for suite in ('spatial', 'l10'):
        d = STORE / 'library' / f'groot_{suite}' / 'bpool_all'
        eps = json.loads((d / 'episodes.json').read_text())
        values = []
        for e in eps:
            eid = int(e['episode_id'])
            match = re.match(r'episode_(\d+)_', e['stem'])
            if not match or int(match[1]) != eid or eid // 50 != e['task_id']:
                raise ValueError('GR00T parent is not the full 50-trial collection')
            if e.get('orig_init_state_idx') not in (None, eid % 50):
                raise ValueError('collector formula disagrees with known original')
            values.append(eid % 50)
        if {(e['task_id'], i) for e, i in zip(eps, values)} != {(t, i) for t in range(10) for i in range(50)} or len(eps) != 500:
            raise ValueError('GR00T parent does not cover full B grid')
        if any(e.get('orig_init_state_idx') is None for e in eps):
            write_identity('groot', suite, 'bpool_all', eps, values, dict(rule='compute_global_episode_id(task, init, 50) = task*50+init',
                source=str(formula), source_sha256=sha(formula), known_coordinates_verified=sum(e.get('orig_init_state_idx') is not None for e in eps),
                cpu_reset_proof='mapping_proof.json'))
        cd = d.with_name('current')
        current = json.loads((cd / 'episodes.json').read_text())
        by_stem = {e['stem']: (e, i) for e, i in zip(eps, values)}
        rs, crs = np.load(d / 'rs.npy', mmap_mode='r'), np.load(cd / 'rs.npy', mmap_mode='r')
        current_values = []
        for e in current:
            p, init = by_stem[e['stem']]
            if e['file'] != p['file'] or e['task_id'] != p['task_id'] or e['success'] != p['success'] or not np.array_equal(rs[p['start']], crs[e['start']]):
                raise ValueError('GR00T current is not a byte-exact parent episode')
            current_values.append(init)
        if any(e.get('orig_init_state_idx') is None for e in current):
            write_identity('groot', suite, 'current', current, current_values, dict(rule='exact same source H5/stem/task/success and first state as certified B parent',
                parent_episodes_sha256=sha(d / 'episodes.json'), parent_rs_sha256=sha(d / 'rs.npy')))
    # Older pi05 l10 episode numbers are not reliable coordinates. Match the
    # full first-state float32 bytes within task, and insist on a unique match.
    d = STORE / 'library/pi05_l10/bpool_cs'
    eps = json.loads((d / 'episodes.json').read_text())
    cd = d.with_name('current')
    current = json.loads((cd / 'episodes.json').read_text())
    rs, crs = np.load(d / 'rs.npy', mmap_mode='r'), np.load(cd / 'rs.npy', mmap_mode='r')
    vals, proof = [], []
    for e in current:
        same = [p for p in eps if p['task_id'] == e['task_id']]
        errors = np.array([np.max(np.abs(rs[p['start'], :8] - crs[e['start'], :8])) for p in same])
        hits = np.flatnonzero(errors == 0)
        if len(hits) != 1:
            raise ValueError(f'unknown pi05 current init: {e["stem"]}, exact matches={len(hits)}, best_delta={errors.min()}')
        hit = int(hits[0]);vals.append(same[hit]['orig_init_state_idx'])
        proof.append(dict(stem=e['stem'], orig_init_state_idx=vals[-1], max_abs_delta=0.,
                          next_best_delta=float(np.sort(errors)[1])))
    write_identity('pi05', 'l10', 'current', current, vals, dict(rule='unique byte-exact first 8-D float32 robot state within task of B parent',
        parent_episodes_sha256=sha(d / 'episodes.json'), parent_rs_sha256=sha(d / 'rs.npy'), comparisons=proof))
    historical = Path('exp/common/data/db/libero_cache/libero_10_init_map.json')
    old = json.loads(historical.read_text())
    sets_equal = all({i for e, i in zip(current, vals) if e['task_id'] == t} ==
                     {r['orig_init_state_idx'] for r in old if r['task_id'] == t} for t in range(10))
    if not sets_equal:
        raise ValueError('recovered current B init sets differ from historical materialization map')
    audit = dict(pi05_l10_current_recovered=50, max_abs_delta=0.,
                 minimum_next_best_delta=min(r['next_best_delta'] for r in proof),
                 historical_init_map_sha256=sha(historical), historical_per_task_init_sets_equal=sets_equal,
                 groot_spatial_parent_recovered=464, groot_spatial_current_recovered=47)
    (HERE / 'identity_audit.json').write_text(json.dumps(audit, indent=1) + '\n')
    print('IDENTITY_AUDIT', json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
