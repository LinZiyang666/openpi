"""Tests for the Phase-6.0 batch-separability acceptance gate (plan §6.0)."""

import numpy as np

from exp.zixuan_proposal.phase6_batch_separability import (
    APRIL,
    JULY,
    Episode,
    episode_feature_digest,
    episode_feature_from_entries,
    gate_input_digest,
    run_gate,
)


def _episodes(n_cells: int, dim: int, separable: bool, seed: int = 0) -> list[Episode]:
    """One April + one July episode per cell. If `separable`, features encode the
    batch (gate should FAIL = confound present); else pure noise (gate should PASS)."""
    rng = np.random.default_rng(seed)
    eps: list[Episode] = []
    for c in range(n_cells):
        for batch in (APRIL, JULY):
            noise = rng.normal(size=dim)
            if separable:
                noise[0] += 4.0 * batch  # linearly separable by batch
            eps.append(Episode(cell=(0, c), batch=batch, features=noise, episode_id=f"ep_{batch}_{c}"))
    return eps


def test_gate_fail_when_batch_separable():
    v = run_gate(_episodes(40, 6, separable=True), b_replicates=1000)
    assert v["status"] == "FAIL"
    assert v["ci_high"] > 0.55


def test_gate_pass_when_batch_unpredictable():
    # The CI-upper<=0.55 gate is STRICT: the cluster-bootstrap CI width is driven by
    # the number of matched cells, so certifying "confound broken" on truly-null data
    # needs many matched cells (a real estimability constraint on Phase 6.0). With
    # enough cells the null-AUROC CI tightens below 0.55 -> PASS.
    v = run_gate(_episodes(800, 6, separable=False), b_replicates=400)
    assert v["status"] == "PASS"
    assert v["ci_high"] <= 0.55


def test_gate_inconclusive_when_too_few_episodes():
    v = run_gate(_episodes(4, 6, separable=True), b_replicates=1000)
    assert v["status"] == "INCONCLUSIVE"


def test_gate_inconclusive_when_no_matched_cells():
    # April-only cells and July-only cells never share a cell -> no matched cells.
    rng = np.random.default_rng(1)
    eps = [Episode(cell=(0, c), batch=APRIL, features=rng.normal(size=4)) for c in range(20)]
    eps += [Episode(cell=(0, 100 + c), batch=JULY, features=rng.normal(size=4)) for c in range(20)]
    assert run_gate(eps, b_replicates=500)["status"] == "INCONCLUSIVE"


def test_pass_verdict_carries_tamper_evident_per_episode_manifest():
    v = run_gate(_episodes(800, 6, separable=False), b_replicates=400)
    assert v["status"] == "PASS"
    assert "input_digest" in v and v["episode_manifest"]
    # each manifest row is [batch, episode_id, cell, feature_digest]
    assert len(v["episode_manifest"]) == 1600  # 800 cells x 2 batches
    assert all(len(r) == 4 and isinstance(r[3], str) for r in v["episode_manifest"])
    assert len({tuple(r[:2]) for r in v["episode_manifest"]}) == 1600  # distinct (batch, episode_id)
    # the recorded digest binds the exact per-episode manifest (incl. feature digests) + counts + CI
    assert v["input_digest"] == gate_input_digest(v["episode_manifest"], v["n_april"], v["n_july"], v["ci_high"])
    # dropping any episode invalidates the digest (count cannot be inflated for free)
    assert v["input_digest"] != gate_input_digest(v["episode_manifest"][:-1], v["n_april"], v["n_july"], v["ci_high"])


def test_feature_digest_binds_classifier_bytes():
    # Same identity/cell, DIFFERENT features -> different manifest row (feature digest changes),
    # so a chance-PASS obtained on fabricated (e.g. constant) features cannot reuse a real row.
    import numpy as np

    real = Episode(cell=(0, 0), batch=APRIL, features=np.arange(6, dtype=float), episode_id="ep_a")
    const = Episode(cell=(0, 0), batch=APRIL, features=np.zeros(6), episode_id="ep_a")
    assert episode_feature_digest(real.features) != episode_feature_digest(const.features)
    # canonical reconstruction from per-step entries matches the pooled feature digest
    class _E:
        def __init__(self, v0):
            self.query_keys = {"vision_0": np.asarray(v0, dtype=float)}

    rows = [_E([1.0, 3.0]), _E([3.0, 1.0])]  # mean -> [2.0, 2.0]
    assert episode_feature_digest(episode_feature_from_entries(rows)) == episode_feature_digest(np.array([2.0, 2.0]))
