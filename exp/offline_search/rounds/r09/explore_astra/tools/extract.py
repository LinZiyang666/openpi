"""Extract discovery-only compact observations, actions and factual outcome ledgers.

Reads small NPZ members lazily, never images; joins all augmentations by decision_id.
No dependency on reader-derived caches. Source stat digest and augmentation metadata
hashes accompany each output. Existing outputs require --overwrite to be replaced.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import time
import numpy as np
from .common import RUN, DERIVED, HERE, accepted, identity, jsonl, dump, sha


def join_aug(debug, kind, index, names):
    arrays, seen, metadata = {}, set(), set()
    paths = sorted((debug / 'aug' / kind).glob('part_*.npz'))
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            ids = z['decision_id']
            take = [j for j, did in enumerate(ids) if str(did) in index]
            if not take:
                continue
            dest = np.array([index[str(ids[j])] for j in take])
            if seen.intersection(dest.tolist()):
                raise ValueError(f'duplicate {kind} decision')
            seen.update(dest.tolist())
            if '_meta_json' in z:
                metadata.add(hashlib.sha256(str(z['_meta_json']).encode()).hexdigest())
            for name in names:
                if name not in z:
                    continue
                x = z[name][take]
                if 'chunk' in name and name != 'chunks':
                    x = x[:, :10, :7]
                if name == 'chunks':
                    x = x[:, :, :10, :7]
                if name not in arrays:
                    fill = np.nan if x.dtype.kind == 'f' else -1
                    arrays[name] = np.full((len(index), *x.shape[1:]), fill, dtype=x.dtype)
                arrays[name][dest] = x
    if kind != 'policy_draws' and len(seen) != len(index):
        raise ValueError(f'{kind} missing {len(index)-len(seen)} decisions')
    return arrays, dict(parts=len(paths), decisions=len(seen), metadata_sha256=sorted(metadata))


def one(spec, detail=True, overwrite=False):
    start = time.monotonic()
    name = spec['arm']
    out = DERIVED / 'compact' / f'{name}.npz'
    meta_path = out.with_suffix('.json')
    if out.exists() and meta_path.exists() and not overwrite:
        return json.loads(meta_path.read_text())
    arm = RUN / 'runs' / name
    debug = arm / 'debug'
    acc = accepted(arm / 'client/journal.jsonl')
    attempts = {(r['task_uid'], int(r.get('attempt', 1))): k for k, r in acc.items()}
    rows, sources = [], [arm / 'client/journal.jsonl', debug / 'MANIFEST.json']
    for path in sorted(debug.glob('server_*/decisions*.jsonl')):
        sources.append(path)
        for r in jsonl(path):
            key = (r.get('task_uid'), int(r.get('attempt', 1)))
            if key not in attempts:
                continue
            r['_task'], r['_init'] = attempts[key]
            r['_block'] = path.parent / 'blocks' / r['blk']
            rows.append(r)
    rows.sort(key=lambda r: (r['_task'], r['_init'], r['decision_seq']))
    index = {r['decision_id']: i for i, r in enumerate(rows)}
    if len(index) != len(rows):
        raise ValueError('duplicate live decisions')
    # Check each accepted client's declared decision count without reading holdout.
    groups = defaultdict(list)
    for r in rows:
        groups[r['episode_key']].append(r)
    for ep, rr in groups.items():
        ep_path = debug / 'client' / ep / 'episode.json'
        sources.append(ep_path)
        e = json.loads(ep_path.read_text())
        if len(rr) != e['n_decisions'] or [r['decision_seq'] for r in rr] != list(range(e['n_decisions'])):
            raise ValueError('incomplete decision sequence')
    cost, n = np.zeros((10, 30)), np.zeros((10, 30), int)
    success = np.array([[acc[t, i]['success'] for i in range(30)] for t in range(10)], float)
    for r in rows:
        t, i = r['_task'], r['_init']
        cost[t, i] += r['owner_cost']
        n[t, i] += 1
    arrays = dict(task=np.array([r['_task'] for r in rows], np.int16),
                  init=np.array([r['_init'] for r in rows], np.int16),
                  seq=np.array([r['decision_seq'] for r in rows], np.int16),
                  decision_id=np.array([r['decision_id'] for r in rows]),
                  vision=np.array([r['vision'] for r in rows], bool),
                  treatment=np.array([r.get('treatment') is True for r in rows]),
                  p=np.array([r.get('p_effective') if r.get('p_effective') is not None else np.nan for r in rows]),
                  coin=np.array([r.get('coin') if r.get('coin') is not None else np.nan for r in rows]),
                  eligible=np.array([r.get('eligible') is True for r in rows]),
                  src=np.array([r['src'] for r in rows]),
                  conf=np.array([r.get('conf') if r.get('conf') is not None else np.nan for r in rows]),
                  cost=cost, decisions=n, success=success)
    for key in ['d1', 'd1_rel', 'disp5', 'dst', 'lib_ep', 'lib_step', 'w_eff', 'regime']:
        arrays[key] = np.array([r.get('extras', {}).get(key, np.nan) for r in rows])
    aug_meta = {}
    if detail:
        blocks = defaultdict(list)
        for j, r in enumerate(rows):
            blocks[r['_block']].append((j, r['blk_i'], r['decision_id']))
        for path, entries in blocks.items():
            sources.append(path)
            with np.load(path, allow_pickle=False) as z:
                jj, kk, ids = zip(*entries)
                jj, kk = np.array(jj), np.array(kk)
                if not np.array_equal(z['decision_id'][kk], ids):
                    raise ValueError('block identity mismatch')
                for key in ['served_chunk', 'cache_chunk', 'policy_chunk', 'state_norm', 'state_wire']:
                    x = z[key][kk]
                    x = x[:, :10, :7] if 'chunk' in key else x[:, :8]
                    if key not in arrays:
                        arrays[key] = np.full((len(rows), *x.shape[1:]), np.nan, x.dtype)
                    arrays[key][jj] = x
        for kind, names in [('policy_shadow', ['chunk']),
                            ('shadow_look', ['rows','weights','cache_chunk','keys_pca_third','keys_pca_wrist']),
                            ('policy_draws', ['chunks'])]:
            # P10 has dual-library shadow names; only the local policy error is
            # required for arm calibration, so it need not choose a library.
            if spec['r8']['variant'] == 'P10' and kind == 'shadow_look':
                continue
            values, aug_meta[kind] = join_aug(debug, kind, index, names)
            arrays.update({kind + '_' + k: v for k, v in values.items()})
            sources.extend(sorted((debug / 'aug' / kind).glob('part_*.npz')))
        if spec['model'] == 'pi05' and spec['r8']['variant'] == 'A':
            values, aug_meta['camera_shadow'] = join_aug(debug, 'camera_shadow', index,
                                                        ['wrist_cache_chunk', 'third_cache_chunk'])
            arrays.update({'camera_shadow_' + k: v for k, v in values.items()})
            sources.extend(sorted((debug / 'aug/camera_shadow').glob('part_*.npz')))
    fingerprint = hashlib.sha256(json.dumps([(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sources]).encode()).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    meta = dict(arm=name, model=spec['model'], suite=spec['suite_short'],
                library_size=spec['r8']['library_size'], variant=spec['r8']['variant'],
                episodes=300, decisions=len(rows), sr=float(success.mean()), ir=float(cost.sum()/n.sum()),
                source_stat_sha256=fingerprint, arm_spec_sha256=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest(),
                source_count=len(sources), tool_sha256=sha(__file__), augmentation=aug_meta,
                split='discovery: tasks 0..9, inits 0..29; no holdout outcomes admitted',
                seconds=time.monotonic()-start, output=str(out), detailed=detail)
    dump(meta_path, meta)
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--arms', nargs='*')
    p.add_argument('--ledger-only', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    specs = json.loads((RUN / 'arms.json').read_text())
    if args.arms:
        specs = [s for s in specs if s['arm'] in args.arms or s['r8']['variant'] in args.arms]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fut = {pool.submit(one, s, not args.ledger_only, args.overwrite): s['arm'] for s in specs}
        for f in as_completed(fut):
            r = f.result()
            results.append(r)
            print(json.dumps({k:r[k] for k in ['arm','decisions','sr','ir','seconds']}), flush=True)
    dump(HERE / 'results' / ('ledger_extract.json' if args.ledger_only else 'extract.json'), results)


if __name__ == '__main__':
    main()
