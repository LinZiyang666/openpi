"""Stage 0: build exact [N,N] raw-score matrices for every (suite, builder) combo.

Also cross-checks the numpy normalizer ports in ``fusion_theory_common`` against
the production torch implementations in ``openpi.cache.components.score_normalizers``
(max abs diff must be < 1e-5), so every downstream number is faithful to src.

Usage:
    uv run exp/weighted_sum/analysis/fusion_theory/collect_scores.py --cache-dir <dir>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fusion_theory_common as C  # noqa: E402

from openpi.cache.components.score_normalizers import (  # noqa: E402
    LegacyPercentileNormalizer,
    ZScoreNormalizer,
)


def parity_check() -> None:
    rng = np.random.default_rng(0)
    cos = rng.uniform(-1, 1, 4096)
    dist = rng.uniform(0, 5, 4096)

    zs_t = ZScoreNormalizer("cosine", mu=0.97, sigma=0.007)
    ours = C.apply_zscore_tanh(cos, "cosine", {"mu": 0.97, "sigma": 0.007})
    ref = zs_t(torch.from_numpy(cos)).numpy()
    assert np.abs(ours - ref).max() < 1e-5, "zscore cosine parity failed"

    zs_l = ZScoreNormalizer("l2", mu=-1.84, sigma=1.0)
    ours = C.apply_zscore_tanh(dist, "l2", {"mu": -1.84, "sigma": 1.0})
    ref = zs_l(torch.from_numpy(dist)).numpy()
    assert np.abs(ours - ref).max() < 1e-5, "zscore l2 parity failed"

    lp_c = LegacyPercentileNormalizer("cosine", p5=0.93, p95=0.99)
    ours = C.apply_legacy_percentile(cos, "cosine", {"p5": 0.93, "p95": 0.99})
    ref = lp_c(torch.from_numpy(cos)).numpy()
    assert np.abs(ours - ref).max() < 1e-5, "legacy percentile cosine parity failed"

    lp_l = LegacyPercentileNormalizer("l2", p5=0.001, p95=0.4, tau=C.LEGACY_TAU)
    ours = C.apply_legacy_percentile(dist, "l2", {"p5": 0.001, "p95": 0.4, "tau": C.LEGACY_TAU})
    ref = lp_l(torch.from_numpy(dist)).numpy()
    assert np.abs(ours - ref).max() < 1e-5, "legacy percentile l2 parity failed"

    # tanh <-> logistic identity: 0.5*(tanh(z)+1) == sigmoid(2z)
    z = rng.normal(0, 3, 4096)
    assert np.abs(C.SQUASHES["tanh"](z) - C.SQUASHES["logistic2z"](z)).max() < 1e-12
    print("parity checks passed (numpy ports == src torch implementations; tanh==sigmoid(2z))")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args()
    cache = Path(args.cache_dir)

    parity_check()
    for suite, builder in C.COMBOS:
        t0 = time.time()
        art = C.load_artifact(suite, builder, cache)
        stats = []
        for f in C.FIELDS:
            pool = C.pool_loeo(art.raw[f], art.same_traj)
            stats.append(f"{f}: LOEO mean={pool.mean():.4f} std={pool.std():.4f}")
        print(
            f"{suite}/{builder}: N={art.n} trajs={len(np.unique(art.traj))} "
            f"tasks={len(art.task_names)} ({time.time() - t0:.1f}s)\n  " + "\n  ".join(stats)
        )


if __name__ == "__main__":
    main()
