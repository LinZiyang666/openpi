"""Fixed spatial projections of only this cell's stored token maps; CPU only."""
import argparse
import json
import time
import numpy as np
from exp.offline_search.rounds.r15.qk.astra.prepare import HERE, DERIVED, load_cell, sha, dump


def run(size):
    _, _, lib, rows, provenance = load_cell('l10', size)
    out = DERIVED / 'round2' / f'pi05_l10_{size}'
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026100502)
    perm = rng.permutation(2048)
    signs = rng.choice([-1., 1.], size=2048).astype(np.float32)
    token_rows = np.load(lib/'tok/rows.npy')
    inverse = np.full(len(np.load(lib/'episode.npy')), -1, np.int64)
    inverse[token_rows] = np.arange(len(token_rows))
    assert np.all(inverse[rows] >= 0)
    values = np.empty((len(rows), 2, 16, 4, 4), np.float32)
    start = time.perf_counter()
    for cam in (0, 1):
        tokens = np.load(lib/f'tok/v{cam}.npy', mmap_mode='r')
        assert tokens.shape[1:] == (256, 2048)
        for lo in range(0, len(rows), 16):
            rr = inverse[rows[lo:lo+16]]
            block = np.asarray(tokens[rr], dtype=np.float32)
            pooled = block.reshape(len(rr), 4, 4, 4, 4, 2048).mean((2, 4))
            projected = (pooled[..., perm] * signs).reshape(len(rr), 4, 4, 16, 128).sum(-1) / np.sqrt(128.)
            values[lo:lo+len(rr), cam] = projected.transpose(0, 3, 1, 2)
        print('tokens', size, cam, 'seconds', time.perf_counter()-start, flush=True)
    np.savez(out/'spatial_tokens.npz', target=values, rows=rows, channel_permutation=perm, channel_signs=signs)
    dump(out/'spatial_tokens_meta.json', {'cell': f'pi05_l10_{size}', 'provenance': provenance,
        'rows':len(rows), 'row_sha256':sha(DERIVED/f'pi05_l10_{size}/train.npz'),
        'projection':'fixed disjoint signed orthonormal channel groups, seed2026100502, 2048->16; 16x16 tokens average pooled to4x4',
        'fit_data':'none for projection; only selected cell rows read', 'seconds':time.perf_counter()-start})


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--size',type=int,required=True);args=ap.parse_args();run(args.size)
