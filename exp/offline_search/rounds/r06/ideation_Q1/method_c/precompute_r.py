"""Exact scalar deployed retrieval, whole-source-episode exclusion; no rollout."""
from __future__ import annotations
import argparse
import time
from types import SimpleNamespace
import numpy as np
from .common import OUT, STORE, SCHEMA, sources, load_base, fingerprint, sha, write_json, output_path


def compute(cell, destination=None):
    start = time.monotonic()
    out = output_path(destination or OUT / cell)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / 'r_bank.json'
    if meta_path.exists():
        from .common import read_bank
        m, _ = read_bank(meta_path)
        print(cell, 'EXISTING_VERIFIED', m['rows'], flush=True)
        return m
    source = sources()[cell]
    base, _ = load_base(source)
    from exp.offline_search.rounds.r02.g1_awm.awm import _kernel_w
    libdir = STORE / 'library' / cell.rsplit('_', 1)[0] / base.cand_name
    arr = lambda name: np.load(libdir / (name + '.npy'), mmap_mode='r')
    manifest = __import__('json').loads((libdir / 'manifest.json').read_text())
    episodes = __import__('json').loads((libdir / 'episodes.json').read_text())
    ep, step, task = arr('episode'), arr('step'), arr('task_id')
    k0, k1, state = arr('key_v0'), arr('key_v1'), arr('rs')
    if not np.array_equal(ep, base.lib_ep) or not np.array_equal(step, base.lib_step):
        raise ValueError('base/library row identity mismatch')
    if not np.array_equal(arr('action'), base.act):
        raise ValueError('base/library action mismatch')
    b, H, dims = manifest['exec_steps'], manifest['H'], list(range(manifest['act_valid_dims']))
    horizon = min(H, 2*b)
    sigma = np.asarray(base.act[:, :b, dims], float).reshape(-1, len(dims)).std(0)
    sigma = np.where(sigma > 1e-12, sigma, 1.)
    # An acquisition source may have aliases; all copies of that source are held out.
    source_ids = []
    for e in episodes:
        source_ids.append((e.get('task_id'), e.get('file') or e.get('stem') or e.get('uid') or str(e.get('episode_id'))))
    ids = {key: i for i, key in enumerate(dict.fromkeys(source_ids))}
    source_ep = np.array([ids[source_ids[int(e)]] for e in ep], np.int64)
    residual = np.empty(len(ep), np.float64)
    nearest = np.empty(len(ep), np.float64)
    for j in range(len(ep)):
        q = SimpleNamespace(key_v0=k0[j], key_v1=k1[j], rs=state[j], step=int(step[j]),
                            prev_hit=True, task_id=int(task[j]))
        T, *_, distance = base._dist(q)
        distance = distance.copy()
        distance[source_ep[T.rows] == source_ep[j]] = np.inf
        eligible = int(np.isfinite(distance).sum())
        if eligible < base.k:
            raise ValueError(f'{cell} row {j}: not enough other-episode candidates')
        ix = np.argpartition(distance, base.k - 1)[:base.k]
        ix = ix[np.lexsort((ix, distance[ix]))]
        rows = T.rows[ix]
        assert np.all(source_ep[rows] != source_ep[j])
        weights = _kernel_w(distance[ix].astype(float) - float(distance[ix[0]]), base.kref)
        _, _, action = base._mix(rows, weights)
        residual[j] = np.sqrt(np.mean(((action[:horizon][:, dims] - base.act[j, :horizon][:, dims]) / sigma)**2))
        nearest[j] = distance[ix[0]]
        if j % 4000 == 0:
            print(cell, j, len(ep), round(time.monotonic()-start, 1), flush=True)
    np.savez_compressed(out / 'r_bank.npz', r=residual, sigma=sigma, episode=ep, step=step,
                        task_id=task, source_episode=source_ep, d1=nearest)
    lengths = {}
    for t in np.unique(task):
        _, counts = np.unique(ep[task == t], return_counts=True)
        lengths[str(int(t))] = dict(h=float(counts.mean()), a=float(np.ceil(counts*b/horizon).mean()),
                                    traffic=1., risk=1., episodes=len(counts))
    meta = dict(schema=SCHEMA+'.r_bank', cell=cell, source=source, rows=len(ep),
        source_episodes=len(np.unique(source_ep)), library_directory=str(libdir),
        library_hashes={name: sha(libdir/name) for name in ('manifest.json','episodes.json','action.npy','episode.npy','step.npy','task_id.npy')},
        base_fit_sha256=sha(source['source_artifact']), retrieval_fingerprint=fingerprint(base),
        interface=dict(block_controls=b, H=H, commit_controls=horizon, valid_action_indices=dims),
        arrays='r_bank.npz', arrays_sha256=sha(out/'r_bank.npz'), library_lengths=lengths,
        definition='conditional LOEO, exact deployed scalar projection/distance/argpartition/kernel/mix; whole source episode excluded; no outcome fit',
        seconds=time.monotonic()-start, finite=bool(np.isfinite(residual).all()), mean_r=float(residual.mean()))
    write_json(meta_path, meta)
    print(cell, 'DONE', meta['seconds'], flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cell', choices=[*sources(), 'all'], required=True)
    a = ap.parse_args()
    for cell in sources() if a.cell == 'all' else [a.cell]:
        compute(cell)


if __name__ == '__main__':
    main()
