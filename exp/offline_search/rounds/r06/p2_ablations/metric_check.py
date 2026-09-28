"""Independent Euclidean-on-z-scored reference on recorded queries (no policy MISS)."""
import argparse
import collections
import json
import pickle

import numpy as np

from exp.offline_search.harness import api, dims, store
from exp.offline_search.rounds.r02.g1_awm import awm
from exp.offline_search.rounds.r04.k1_blind.checks import view
from exp.offline_search.rounds.r06.p2_ablations.make_arms import HERE, RUN, STORE, artifact, sources


class CacheQuery:
    def __init__(self, q):
        self.q = q
        self.prev_hit = True if q.step else None  # A never invokes policy
    def __getattr__(self, k):
        return getattr(self.q, k)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', required=True)
    a = p.parse_args()
    row, = [r for r in json.loads((HERE / 'arms_metric.json').read_text()) if r['name'] == a.arm]
    with open(artifact(row).replace('<RUN>', str(RUN)), 'rb') as f:
        m = pickle.load(f)['method']
    scale = 50 if m.lib == 'current' else 500
    src = sources()[row['model'], row['suite'], scale, 'A'][1]
    with open(artifact(src), 'rb') as f:
        old = pickle.load(f)['method']
    for field in ('B0T', 'B1T', 'mu0', 'mu1', 'muB0', 'muB1', 'act', 'lib_step', 'lib_ep', 'blind_next',
                  'blind_rs', 'blind_event', 'blind_terminal'):
        assert np.array_equal(getattr(m, field), getattr(old, field)), field
    for field in ('k', 'kref', 'early', 'features', 'codes', 'state_scale', 'serving', 'budget', 'gates',
                  'norm_cap', 'hyst', 'lam_c', 'fit_src', 'feat0'):
        assert getattr(m, field) == getattr(old, field), field
    lib = store.LibraryView(STORE, f"{row['model']}_{row['suite']}", m.cand_name)
    P = []
    for v in ('v0', 'v1'):
        if m.lib == 'big':
            d = awm.PCA_BIG / f"{row['model']}_{row['suite']}" / m.cand_name / v
            P.append(np.asarray(np.load(d / 'proj.npy', mmap_mode='r')[:, :64], np.float64))
        else:
            P.append(np.asarray(awm.pca_current(lib, f"{row['model']}_{row['suite']}", v)[2], np.float64))
    X = np.concatenate([*P, np.asarray(dims.valid_state(lib.rs, m.model), np.float64)], axis=1)
    assert X.shape == (lib.L, 136)
    refs = {}
    for task, T in m.tasks.items():
        x = X[T.rows]
        early = x[np.asarray(lib.step)[T.rows] <= 2]
        mu, sd = x.mean(0), x.std(0) + 1e-6
        mu0, sd0 = early.mean(0), early.std(0) + 1e-6
        assert np.array_equal(T.Wf, np.asarray(np.diag(1 / sd), np.float32))
        assert np.array_equal(T.W0f, np.asarray(np.diag(1 / sd0), np.float32))
        assert np.array_equal(T.Z, ((x - mu) / sd).astype(np.float32))
        assert T.Z0 is None and T.As0 is None  # same full-rank early affine branch
        refs[task] = [(mu0, sd0, (x - mu0) / sd0), (mu, sd, (x - mu) / sd)]
    counts = collections.Counter()
    mismatches = []
    max_dist_error = 0.
    min_boundary_gap = float('inf')
    for stream in ('cache', 'inf'):
        qc = store.QueryCell(STORE, f"{row['model']}_{row['suite']}_{stream}")
        A = api.QueryArrays(qc)
        # Every step-0 query plus 1,000 evenly spaced noninitial recorded queries.
        ids = sorted(set([e['start'] for e in qc.episodes] + np.linspace(0, qc.N - 1, 1000, dtype=int).tolist()))
        for i in ids:
            q = CacheQuery(view(qc, A, i))
            m.reset(q.episode)
            result = m.query(q)
            T = m.tasks[int(q.task_id)]
            xv = np.r_[m.B0T @ np.asarray(q.key_v0, np.float32) - m.muB0,
                       m.B1T @ np.asarray(q.key_v1, np.float32) - m.muB1,
                       dims.valid_state(q.rs, m.model)].astype(np.float64)
            mu, sd, Z = refs[int(q.task_id)][0 if q.step == 0 else 1]
            # Direct differences in float64; independent of AWM's quadratic-distance / affine implementation.
            d = np.linalg.norm(Z - (xv - mu) / sd, axis=1)
            idx = np.argsort(d, kind='stable')[:16]
            refrows = T.rows[idx]
            if not np.array_equal(result.topk, refrows):
                mismatches.append(dict(stream=stream, row=i, step=q.step,
                                       got=result.topk.tolist(), reference=refrows.tolist(),
                                       reference_distances=d[idx].tolist()))
            max_dist_error = max(max_dist_error, float(np.max(np.abs(-result.scores - d[np.searchsorted(T.rows, result.topk)]))))
            ordered = np.sort(d)
            min_boundary_gap = min(min_boundary_gap, float(ordered[16] - ordered[15]))
            # Exact stock kernel/synthesis on the served scores; whole chunk, including padding.
            w = awm._kernel_w(-result.scores + result.scores[0], m.kref)
            ref_action = np.tensordot((w / w.sum()).astype(np.float32), m.act[result.topk], 1)
            assert np.array_equal(result.action, ref_action)
            counts[f'{stream}_early' if q.step == 0 else f'{stream}_main'] += 1
    out = dict(PASS=not mismatches, arm=a.arm, feature_dims=136, M='exact identity; no large-lambda approximation',
               queries=sum(counts.values()), counts=dict(counts), exact_ordered_top16_mismatches=len(mismatches),
               max_selected_distance_absolute_error=max_dist_error, minimum_reference_16_17_gap=min_boundary_gap,
               pca_and_nonmetric_configuration_equal_to_A=True, mismatches=mismatches)
    (HERE / 'results' / (a.arm + '_metric.json')).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'mismatches'}), flush=True)
    assert not mismatches, f"{len(mismatches)} floating-point ranking differences: see result JSON"


if __name__ == '__main__':
    main()
