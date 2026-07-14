"""Tests for the projection trainset driver (plan §A/§C wiring)."""

import json
import pathlib
import pickle

import pytest

import numpy as np

from exp.zixuan_proposal.build_projection_trainset import (
    _dminus_fold,
    _fold_map,
    _verify_verdict_binding,
    build_trainset,
)
from exp.zixuan_proposal.phase6_batch_separability import (
    episode_feature_digest,
    episode_feature_from_entries,
    gate_input_digest,
)
from exp.zixuan_proposal.projection_labels import Entry

_REPO = pathlib.Path(__file__).resolve().parents[2]
_SPATIAL_PKL = _REPO / "exp/common/data/cache_artifacts/libero_spatial/cp1_mean_pool_dual.pkl"


def _episode_manifest(n_per_batch=12, init=2):
    """A per-episode manifest with n_per_batch distinct April + July episodes (>= min-10 floor).
    Each row is [batch, episode_id, [task, init], feature_digest]; a placeholder feature digest is
    fine for tests that are rejected BEFORE the artifact feature-binding stage."""
    return [[b, f"ep_{b}_{i}", [i % 3, init], f"fd_{b}_{i}"] for b in (0, 1) for i in range(n_per_batch)]


def _valid_verdict(manifest=None, *, ci_high=0.53):
    """A structurally-valid PASS verdict whose input_digest matches its per-episode manifest (so
    it reaches the artifact-binding / control checks rather than being rejected as malformed)."""
    em = manifest if manifest is not None else _episode_manifest()
    n_april = len({e[1] for e in em if e[0] == 0})
    n_july = len({e[1] for e in em if e[0] == 1})
    return {
        "status": "PASS",
        "ci_high": ci_high,
        "n_april": n_april,
        "n_july": n_july,
        "matched_cells": sorted({tuple(e[2]) for e in em}),
        "episode_manifest": em,
        "input_digest": gate_input_digest(em, n_april, n_july, ci_high),
    }


def _e(ident, outcome):
    import numpy as np

    return Entry(ident=ident, outcome=outcome, action_flat=np.zeros(4), snap_flat={})


def test_dminus_fold_stable_and_only_train_or_val():
    for ident in [(0, 0), (3, 12), (9, 48)]:
        assert _dminus_fold(ident) == _dminus_fold(ident)
        assert _dminus_fold(ident) in ("train", "val")


def test_fold_map_is_per_entry_and_dminus_never_test():
    entries = [
        _e((0, 0), 1),
        _e((0, 2), 1),
        _e((0, 4), 1),
        _e((0, 6), 1),  # task 0: 4 even D+ -> last=test, second-last=val, rest=train
        _e((5, 10), -1),  # D- unique -> train/val by hash
        _e((0, 6), -1),  # D- sharing the D+ mechanism-test identity -> MUST be excluded, not test
    ]
    fold = _fold_map(entries)  # per-entry list, aligned to entries
    assert fold[3] == "test"  # the D+ at (0,6)
    assert fold[2] == "val"  # the D+ at (0,4)
    assert fold[4] in ("train", "val")  # unique D-
    assert fold[5] == "excluded"  # the D- at the D+ test identity -> excluded (never test)


def test_gate_not_bypassable_by_string():
    # No persisted verdict -> blocked (a caller can no longer type PASS).
    with pytest.raises(SystemExit, match="persisted batch-separability verdict"):
        build_trainset("libero_spatial", eta=1.0, control_artifact_path=None, batch_sep_verdict_path=None)


def test_gate_requires_pass_verdict(tmp_path):
    fail = tmp_path / "verdict.json"
    fail.write_text(json.dumps({"status": "FAIL", "ci_high": 0.7}))
    with pytest.raises(SystemExit, match="not PASS"):
        build_trainset("libero_spatial", eta=1.0, control_artifact_path=tmp_path / "c.pkl", batch_sep_verdict_path=fail)


def _synthetic_entries(dim=8):
    """6 tasks x 4 even D+ inits, within-task actions close (positives), cross-task far."""
    import numpy as np

    from exp.zixuan_proposal.projection_labels import Entry

    rng = np.random.default_rng(0)
    entries, feats = [], []
    for t in range(6):
        base = np.zeros(dim)
        base[t] = 5.0  # task-separated action cluster
        for init in (0, 2, 4, 6):
            entries.append(Entry(ident=(t, init), outcome=1, action_flat=base + 0.05 * rng.normal(size=dim), snap_flat={}))
            feats.append({"vision_0": rng.normal(size=dim), "vision_1": rng.normal(size=dim)})
    return entries, feats


def test_finalize_successful_build_and_fit_schema():
    # One successful end-to-end assembly on controlled data + build->fit schema integration.
    from exp.zixuan_proposal.build_projection_trainset import _finalize
    from exp.zixuan_proposal.fit_projection import fit_from_trainset

    entries, feats = _synthetic_entries(dim=8)
    ts = _finalize("libero_spatial", entries, feats, eta=1.0, seed=7, meta_extra={"control": {"n_kept": 24}})
    assert ts["meta"]["valid_anchor_frac"] >= 0.5
    assert len(ts["meta"]["represented_tasks"]) >= 5  # all 6 tasks represented
    assert any(r["fold"] == "test" for r in ts["rows"]) and all(r["fold"] != "excluded" for r in ts["rows"])
    params, prov = fit_from_trainset(ts, out_dim=4, epochs=5)
    assert params.head("vision_0").weight.shape == (4, 8)
    assert prov["vision_0"]["selected_epoch"] >= 0


def test_verdict_and_control_authenticity(tmp_path):
    # A forged / under-threshold verdict is rejected (not just a status string).
    bad = tmp_path / "forged.json"
    bad.write_text(json.dumps({"status": "PASS", "ci_high": 0.9, "n_april": 50, "n_july": 50}))
    with pytest.raises(SystemExit, match="0.55"):
        build_trainset("libero_spatial", eta=1.0, control_artifact_path=tmp_path / "c.pkl", batch_sep_verdict_path=bad)
    thin = tmp_path / "thin.json"
    thin.write_text(json.dumps({"status": "PASS", "ci_high": 0.5, "n_april": 3, "n_july": 50}))
    with pytest.raises(SystemExit, match="min-10 floor"):
        build_trainset("libero_spatial", eta=1.0, control_artifact_path=tmp_path / "c.pkl", batch_sep_verdict_path=thin)


def test_nan_ci_high_is_not_pass(tmp_path):
    # A NaN CI upper bound compares false to `> 0.55`; it must be rejected as non-finite,
    # never silently accepted as PASS (independent probe).
    v = tmp_path / "nan.json"
    v.write_text(json.dumps({**_valid_verdict(), "ci_high": float("nan")}))
    with pytest.raises(SystemExit, match="finite"):
        build_trainset("libero_spatial", eta=1.0, control_artifact_path=tmp_path / "c.pkl", batch_sep_verdict_path=v)


def test_hand_edited_verdict_fails_digest(tmp_path):
    # Bumping a recorded metric without recomputing input_digest is caught.
    d = _valid_verdict()
    d["ci_high"] = 0.10  # tampered lower CI, stale digest
    v = tmp_path / "tampered.json"
    v.write_text(json.dumps(d))
    with pytest.raises(SystemExit, match="input_digest mismatch"):
        build_trainset("libero_spatial", eta=1.0, control_artifact_path=tmp_path / "c.pkl", batch_sep_verdict_path=v)


class _Ep:
    def __init__(self, tid, v0):
        self.outcome = 1
        self.trajectory_id = tid
        self.step_idx = 0
        self.query_keys = {"vision_0": np.asarray(v0, dtype=float)}


def _fd(v0):
    """The feature digest a manifest row must carry for a one-row episode with vision_0 = v0."""
    return episode_feature_digest(episode_feature_from_entries([_Ep("x", v0)]))


def test_verify_verdict_binding_exact_episode_correspondence():
    base = [_Ep("a0", [1.0, 0.0]), _Ep("a1", [0.0, 1.0])]  # 2 independent April episodes
    control = [_Ep("j0", [2.0, 0.0]), _Ep("j1", [0.0, 2.0])]  # 2 independent July episodes
    ident_of = {
        id(base[0]): (0, 0), id(base[1]): (1, 2),
        id(control[0]): (0, 0), id(control[1]): (1, 2),
    }
    em = [
        [0, "a0", [0, 0], _fd([1.0, 0.0])], [0, "a1", [1, 2], _fd([0.0, 1.0])],
        [1, "j0", [0, 0], _fd([2.0, 0.0])], [1, "j1", [1, 2], _fd([0.0, 2.0])],
    ]
    verdict = {"episode_manifest": em, "n_april": 2, "n_july": 2}
    kept = _verify_verdict_binding(verdict, ident_of, control, base)
    assert kept == 2  # both July cells even-init (0, 2)

    # n_april=10 backed by only 2 real April episodes -> rejected (the inflation attack).
    inflated = {"episode_manifest": em, "n_april": 10, "n_july": 2}
    with pytest.raises(SystemExit, match="distinct episodes"):
        _verify_verdict_binding(inflated, ident_of, control, base)

    # a verdict episode absent from the supplied artifacts -> rejected.
    ghost = {"episode_manifest": em + [[1, "ghost", [0, 0], _fd([9.0, 9.0])]], "n_april": 2, "n_july": 3}
    with pytest.raises(SystemExit, match="does not correspond"):
        _verify_verdict_binding(ghost, ident_of, control, base)

    # SAME identities/ids but a FABRICATED feature digest (e.g. gate run on constants) -> rejected.
    faked = {
        "episode_manifest": [[0, "a0", [0, 0], _fd([0.0, 0.0])]] + em[1:],  # a0 digest doesn't match real vision
        "n_april": 2, "n_july": 2,
    }
    with pytest.raises(SystemExit, match="does not correspond"):
        _verify_verdict_binding(faked, ident_of, control, base)


@pytest.mark.skipif(not _SPATIAL_PKL.exists(), reason="real spatial artifact not present")
def test_real_control_from_april_batch_is_rejected(tmp_path):
    # A control built from the April D+ artifact (not the July batch) must be rejected, so
    # July source/batch identity is genuinely enforced (independent forged-control probe).
    with open(_SPATIAL_PKL, "rb") as fh:
        art = pickle.load(fh)
    dplus = [e for e in art["entries"] if e.outcome == 1][:40]  # April-batch D+
    control = tmp_path / "control.pkl"
    with open(control, "wb") as fh:
        pickle.dump({"entries": dplus, "vector_dims": art["vector_dims"]}, fh)
    verdict = tmp_path / "pass.json"
    verdict.write_text(json.dumps(_valid_verdict()))
    with pytest.raises(SystemExit, match="batch"):
        build_trainset("libero_spatial", eta=1.0, control_artifact_path=control, batch_sep_verdict_path=verdict)
