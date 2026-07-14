"""Tests for the offline pre-GPU go/no-go (plan §6.7)."""

import numpy as np

from exp.zixuan_proposal.projection_offline_gate import offline_go_no_go


def _synthetic(n_clusters=60, rows_per=5, seed=0):
    rng = np.random.default_rng(seed)
    cluster, safe, sa, sb = [], [], [], []
    for c in range(n_clusters):
        for _ in range(rows_per):
            y = int(rng.integers(0, 2))
            cluster.append((0, c))
            safe.append(y)
            sa.append(rng.normal())  # A: uninformative gate score
            sb.append(y + 0.5 * rng.normal())  # B: correlated with safe-reuse
    return (
        np.array(cluster),
        np.array(safe),
        np.array(sa),
        np.array(sb),
    )


def test_go_when_projected_gate_separates_better():
    cluster, safe, sa, sb = _synthetic()
    v = offline_go_no_go(cluster, safe, sa, sb, b_replicates=1000)
    assert v["status"] == "GO"
    assert v["auroc_b"] > v["auroc_a"]
    assert v["ci_low"] > 0


def test_no_go_when_identical():
    cluster, safe, sa, _ = _synthetic()
    v = offline_go_no_go(cluster, safe, sa, sa, b_replicates=1000)
    assert v["status"] == "NO_GO"
    assert abs(v["point"]) < 1e-9
