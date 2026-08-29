"""Unit tests for SurfaceJudge, SurfaceArtifact and the shared (v, Y) metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from openpi.cache.components.judge import HitType, SimilarityJudge
from openpi.cache.components.surface_judge import (
    CERTIFICATION_CONFORMAL,
    SURFACE_ARTIFACT_SCHEMA_VERSION,
    SurfaceArtifact,
    SurfaceJudge,
    load_surface_artifact,
    save_surface_artifact,
    weighted_chunk_deviation,
    weighted_topk_disagreement,
)
from openpi.cache.storage_types import CachePayload, SearchResultLite
from openpi.cache.types import CheckpointID

ACTION_DIM = 4
HORIZON = 6
H_EXEC = 5


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _contract() -> dict:
    return {
        "key_builder_digest": "kb", "search_digest": "sd", "top_k": 3,
        "library_sha256": "lib", "library_entry_count": 10,
        "action_dim": ACTION_DIM, "num_steps": 10, "h_exec": H_EXEC,
        "policy_fingerprint": "fp",
    }


def make_artifact(tmp_path, *, uses_disagreement=True, s_min_full=None,
                  s_min_warm=None, v_edges=None, conformal_c=0.01, **overrides):
    if uses_disagreement:
        v_edges = np.array([0.0, 1.0, 2.0]) if v_edges is None else np.asarray(v_edges)
        s_min_full = np.array([0.90, 0.95]) if s_min_full is None else np.asarray(s_min_full)
        s_min_warm = np.array([0.80, 0.85]) if s_min_warm is None else np.asarray(s_min_warm)
    else:
        v_edges = np.array([-np.inf, np.inf]) if v_edges is None else np.asarray(v_edges)
        s_min_full = np.array([0.90]) if s_min_full is None else np.asarray(s_min_full)
        s_min_warm = np.array([0.80]) if s_min_warm is None else np.asarray(s_min_warm)
    kwargs = dict(
        schema_version=SURFACE_ARTIFACT_SCHEMA_VERSION,
        k=3, h_exec=H_EXEC,
        w=np.ones(ACTION_DIM, dtype=np.float32),
        active_mask=np.ones(ACTION_DIM, dtype=bool),
        start_t_ws=0.3, delta=0.5, quantile_alpha=0.05,
        certification_mode=CERTIFICATION_CONFORMAL,
        uses_disagreement=uses_disagreement,
        v_bin_edges=v_edges, s_min_full=s_min_full, s_min_warm=s_min_warm,
        conformal_c=conformal_c, n_calibration_episodes=100,
        retrieval_contract=_contract(), meta={"ref_mode": "fresh"},
    )
    kwargs.update(overrides)
    artifact = SurfaceArtifact(**kwargs)
    path = tmp_path / "surface.npz"
    save_surface_artifact(artifact, str(path))
    return str(path)


class FakeView:
    """PayloadView stub returning identical or perturbed chunks."""

    def __init__(self, chunks_by_id: dict[str, torch.Tensor], fail: bool = False):
        self._chunks = chunks_by_id
        self.fail = fail
        self.calls = 0

    def get_many(self, ids):
        self.calls += 1
        if self.fail:
            raise RuntimeError("backend down")
        return [CachePayload(action_chunk=self._chunks[i]) for i in ids]

    def get(self, entry_id):
        return CachePayload(action_chunk=self._chunks[entry_id])

    def get_entry(self, entry_id):  # pragma: no cover - protocol completeness
        raise NotImplementedError


def results(*scores):
    return [SearchResultLite(id=f"e{i}", score=s, checkpoint_id=CheckpointID.CP1)
            for i, s in enumerate(scores)]


def tight_view(base=0.0):
    chunk = torch.full((HORIZON, ACTION_DIM), base)
    return FakeView({f"e{i}": chunk.clone() for i in range(5)})


# ------------------------------------------------------------------
# Metric functions
# ------------------------------------------------------------------


def test_disagreement_zero_for_identical_chunks():
    chunks = torch.zeros(3, HORIZON, ACTION_DIM)
    v = weighted_topk_disagreement(chunks, torch.ones(ACTION_DIM),
                                   torch.ones(ACTION_DIM, dtype=torch.bool), H_EXEC)
    assert v == 0.0


def test_disagreement_nan_below_two_candidates():
    chunks = torch.zeros(1, HORIZON, ACTION_DIM)
    v = weighted_topk_disagreement(chunks, torch.ones(ACTION_DIM),
                                   torch.ones(ACTION_DIM, dtype=torch.bool), H_EXEC)
    assert np.isnan(v)


def test_deviation_shape_mismatch_raises():
    a = torch.zeros(HORIZON, ACTION_DIM)
    b = torch.zeros(HORIZON + 1, ACTION_DIM)
    with pytest.raises(ValueError):
        weighted_chunk_deviation(a, b, torch.ones(ACTION_DIM),
                                 torch.ones(ACTION_DIM, dtype=torch.bool), H_EXEC)


def test_deviation_identical_chunks_is_zero():
    a = torch.randn(HORIZON, ACTION_DIM)
    d = weighted_chunk_deviation(a, a.clone(), torch.ones(ACTION_DIM),
                                 torch.ones(ACTION_DIM, dtype=torch.bool), H_EXEC)
    assert d == 0.0


def test_tau_start_t_sign_convention():
    # tau = 7 skipped steps of N = 10 <-> start_t = (N - tau)/N = 0.3. Pinned.
    n, tau = 10, 7
    assert round(1.0 - tau / n, 4) == 0.3


# ------------------------------------------------------------------
# Artifact validation
# ------------------------------------------------------------------


def test_artifact_roundtrip(tmp_path):
    path = make_artifact(tmp_path)
    art = load_surface_artifact(path)
    assert art.k == 3 and art.uses_disagreement
    assert art.retrieval_contract["library_sha256"] == "lib"


@pytest.mark.parametrize("mutation", [
    {"s_min_warm": np.array([0.95, 0.96])},              # warm > full
    {"s_min_full": np.array([0.95, 0.90])},              # decreasing in v
    {"v_edges": np.array([0.0, 0.0, 1.0])},              # non-strict edges
    {"s_min_full": np.array([0.9, float("nan")])},       # NaN boundary
    {"s_min_full": np.array([-np.inf, 0.9])},            # -inf boundary
    {"start_t_ws": 0.35},                                 # non-canonical tier
    {"k": 1},                                             # k too small
    {"quantile_alpha": 0.9},                              # quantile_alpha domain
    {"conformal_c": 1.0, "s_min_full": np.array([np.inf, np.inf])},  # placeholder
])
def test_artifact_validation_rejects(tmp_path, mutation):
    if mutation.get("conformal_c") == 1.0:
        # inverse case: c=+inf must force all-inf boundaries.
        with pytest.raises(ValueError):
            make_artifact(tmp_path, conformal_c=float("inf"),
                          s_min_full=np.array([0.9, 0.95]),
                          s_min_warm=np.array([0.8, 0.85]))
        return
    with pytest.raises(ValueError):
        make_artifact(tmp_path, **mutation)


def test_pinned_start_t_rejects_other_canonical_tiers(tmp_path):
    """G2-B7: 0.5 is a canonical timestep but NOT this line's frozen tier."""
    with pytest.raises(ValueError):
        make_artifact(tmp_path, start_t_ws=0.5)


@pytest.mark.parametrize("field, value", [
    ("h_exec", 99),        # contract h_exec != artifact h_exec
    ("top_k", 7),          # contract top_k != artifact k
    ("action_dim", 32),    # contract action_dim != len(w)
])
def test_contract_cross_field_invariants(tmp_path, field, value):
    contract = _contract()
    contract[field] = value
    with pytest.raises(ValueError):
        make_artifact(tmp_path, retrieval_contract=contract)


def test_nonpositive_h_exec_rejected(tmp_path):
    contract = _contract()
    contract["h_exec"] = 0
    with pytest.raises(ValueError):
        make_artifact(tmp_path, h_exec=0, retrieval_contract=contract)


def test_all_miss_degenerate_artifact_loads(tmp_path):
    path = make_artifact(
        tmp_path, conformal_c=float("inf"),
        s_min_full=np.array([np.inf, np.inf]), s_min_warm=np.array([np.inf, np.inf]),
    )
    judge = SurfaceJudge(path)
    r = judge(results(0.99, 0.99, 0.99), CheckpointID.CP1, {}, view=tight_view())
    assert r.hit_type is HitType.MISS


def test_s_only_sentinel_enforced(tmp_path):
    with pytest.raises(ValueError):
        make_artifact(tmp_path, uses_disagreement=False,
                      v_edges=np.array([0.0, 1.0]),
                      s_min_full=np.array([0.9]), s_min_warm=np.array([0.8]))


# ------------------------------------------------------------------
# Verdicts
# ------------------------------------------------------------------


def test_full_hit_above_full_boundary(tmp_path):
    judge = SurfaceJudge(make_artifact(tmp_path))
    r = judge(results(0.92, 0.9, 0.9), CheckpointID.CP1, {}, view=tight_view())
    assert r.hit_type is HitType.FULL_HIT and r.winner_id == "e0"


def test_warm_start_between_boundaries(tmp_path):
    judge = SurfaceJudge(make_artifact(tmp_path))
    r = judge(results(0.85, 0.8, 0.8), CheckpointID.CP1, {}, view=tight_view())
    assert r.hit_type is HitType.WARM_START and r.start_t == 0.3


def test_miss_below_warm_boundary(tmp_path):
    judge = SurfaceJudge(make_artifact(tmp_path))
    r = judge(results(0.5, 0.4), CheckpointID.CP1, {}, view=tight_view())
    assert r.hit_type is HitType.MISS


def test_high_v_bin_uses_stricter_boundary(tmp_path):
    # v in the second bin (edges [0,1,2]): boundary 0.95/0.85. Build a view
    # whose chunks disagree enough to land v in bin 1.
    chunks = {f"e{i}": torch.full((HORIZON, ACTION_DIM), float(i)) for i in range(3)}
    judge = SurfaceJudge(make_artifact(tmp_path))
    r = judge(results(0.92, 0.9, 0.9), CheckpointID.CP1, {}, view=FakeView(chunks))
    # v ≈ mean squared deviation of {0,1,2} * dims = large -> above right edge -> MISS
    assert r.hit_type is HitType.MISS


@pytest.mark.parametrize("case", ["empty", "s_nonfinite", "view_none", "fetch_fail", "k_eff"])
def test_fail_closed_paths(tmp_path, case):
    judge = SurfaceJudge(make_artifact(tmp_path))
    if case == "empty":
        r = judge([], CheckpointID.CP1, {})
    elif case == "s_nonfinite":
        r = judge(results(float("inf"), 0.9), CheckpointID.CP1, {}, view=tight_view())
    elif case == "view_none":
        r = judge(results(0.99, 0.99), CheckpointID.CP1, {}, view=None)
    elif case == "fetch_fail":
        r = judge(results(0.99, 0.99), CheckpointID.CP1, {},
                  view=FakeView({}, fail=True))
    else:
        r = judge(results(0.99), CheckpointID.CP1, {}, view=tight_view())
    assert r.hit_type is HitType.MISS


def test_v_below_left_edge_clamps_to_first_bin(tmp_path):
    # Identical chunks -> v = 0.0 == left edge of bin 0; still a valid verdict.
    judge = SurfaceJudge(make_artifact(tmp_path))
    r = judge(results(0.91, 0.9), CheckpointID.CP1, {}, view=tight_view())
    assert r.hit_type is HitType.FULL_HIT


def test_cp3_defensive_miss(tmp_path):
    judge = SurfaceJudge(make_artifact(tmp_path))
    r = judge(results(0.99, 0.99), CheckpointID.CP3, {}, view=tight_view())
    assert r.hit_type is HitType.MISS


def test_min_required_top_k_exposed(tmp_path):
    assert SurfaceJudge(make_artifact(tmp_path)).min_required_top_k == 3
    assert SurfaceJudge(
        make_artifact(tmp_path, uses_disagreement=False)
    ).min_required_top_k == 1


def test_s_only_never_touches_view_and_invariant_to_payloads(tmp_path):
    judge = SurfaceJudge(make_artifact(tmp_path, uses_disagreement=False))
    view_a = tight_view()
    r_a = judge(results(0.85, 0.2), CheckpointID.CP1, {}, view=view_a)
    chunks = {f"e{i}": torch.randn(HORIZON, ACTION_DIM) * 100 for i in range(5)}
    view_b = FakeView(chunks)
    r_b = judge(results(0.85, 0.2), CheckpointID.CP1, {}, view=view_b)
    assert view_a.calls == 0 and view_b.calls == 0
    assert r_a.hit_type is r_b.hit_type is HitType.WARM_START


def test_export_factor_outputs_attached(tmp_path):
    judge = SurfaceJudge(make_artifact(tmp_path), export_factor_outputs=True)
    r = judge(results(0.92, 0.9), CheckpointID.CP1, {}, view=tight_view())
    assert r.factor_outputs["s"] == 0.92
    assert r.factor_outputs["v"] == 0.0
    assert r.factor_outputs["verdict_src"] == "full"
    r_off = SurfaceJudge(make_artifact(tmp_path))(
        results(0.92, 0.9), CheckpointID.CP1, {}, view=tight_view()
    )
    assert r_off.factor_outputs is None


def test_protocol_conformance(tmp_path):
    assert isinstance(SurfaceJudge(make_artifact(tmp_path)), SimilarityJudge)


def test_npz_rejects_pickle_payload(tmp_path):
    # An object-array npz must be rejected by allow_pickle=False loading.
    path = tmp_path / "evil.npz"
    np.savez(path, w=np.array([{"a": 1}], dtype=object))
    with pytest.raises(Exception):
        load_surface_artifact(str(path))


# ---------------- Rev 1 certification mode ----------------

def test_empirical_mode_forbids_a_conformal_correction(tmp_path):
    """An empirical artifact carrying a c would read as a certificate."""
    from openpi.cache.components.surface_judge import CERTIFICATION_EMPIRICAL

    with pytest.raises(ValueError, match="conformal_c == 0"):
        make_artifact(tmp_path, certification_mode=CERTIFICATION_EMPIRICAL, conformal_c=0.01)


def test_empirical_mode_forbids_a_calibration_episode_count(tmp_path):
    from openpi.cache.components.surface_judge import CERTIFICATION_EMPIRICAL

    with pytest.raises(ValueError, match="n_calibration_episodes == 0"):
        make_artifact(tmp_path, certification_mode=CERTIFICATION_EMPIRICAL,
                      conformal_c=0.0, n_calibration_episodes=100)


def test_empirical_mode_round_trips_when_it_claims_nothing(tmp_path):
    from openpi.cache.components.surface_judge import (
        CERTIFICATION_EMPIRICAL, load_surface_artifact,
    )

    path = make_artifact(tmp_path, certification_mode=CERTIFICATION_EMPIRICAL,
                         conformal_c=0.0, n_calibration_episodes=0)
    back = load_surface_artifact(path)
    assert back.certification_mode == CERTIFICATION_EMPIRICAL
    assert back.conformal_c == 0.0 and back.n_calibration_episodes == 0


def test_unknown_certification_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="certification_mode"):
        make_artifact(tmp_path, certification_mode="probably_fine")


def test_v1_artifact_without_a_mode_is_refused_not_defaulted(tmp_path):
    """A pre-Rev-1 artifact must not acquire a certification claim by default."""
    import json

    import numpy as np

    from openpi.cache.components.surface_judge import load_surface_artifact

    p = tmp_path / "v1.npz"
    scalars = {"schema_version": 2, "k": 1, "h_exec": 5, "start_t_ws": 0.3,
               "delta": 0.5, "uses_disagreement": False,
               "conformal_c": 0.0, "n_calibration_episodes": 0}
    np.savez(
        p, w=np.ones(2, dtype=np.float32), active_mask=np.ones(2, dtype=bool),
        v_bin_edges=np.array([-np.inf, np.inf]), s_min_full=np.array([0.9]),
        s_min_warm=np.array([0.8]),
        scalars_json=np.frombuffer(json.dumps(scalars).encode(), dtype=np.uint8),
        contract_json=np.frombuffer(json.dumps({"h_exec": 5}).encode(), dtype=np.uint8),
        meta_json=np.frombuffer(json.dumps({}).encode(), dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="predates schema"):
        load_surface_artifact(str(p))


# ---------------- Rev 1 effective top-k contract (four tamper classes) ----------------

def test_sv_contract_top_k_must_equal_artifact_k(tmp_path):
    """The blocker the old stop-loss hid: SV saved with the yaml's configured 1."""
    with pytest.raises(ValueError, match=r"contract top_k=1 != artifact k=5"):
        make_artifact(tmp_path, uses_disagreement=True, k=5,
                      v_edges=np.array([0.0, 1.0, 2.0]),
                      retrieval_contract={**_contract(), "top_k": 1})


def test_sv_contract_top_k_must_not_exceed_artifact_k(tmp_path):
    with pytest.raises(ValueError, match="contract top_k"):
        make_artifact(tmp_path, uses_disagreement=True, k=5,
                      v_edges=np.array([0.0, 1.0, 2.0]),
                      retrieval_contract={**_contract(), "top_k": 7})


def test_sv_contract_top_k_missing_is_rejected(tmp_path):
    contract = {k: v for k, v in _contract().items() if k != "top_k"}
    with pytest.raises(ValueError, match="contract top_k"):
        make_artifact(tmp_path, uses_disagreement=True, k=5,
                      v_edges=np.array([0.0, 1.0, 2.0]), retrieval_contract=contract)


def test_sv_effective_width_round_trips_at_five(tmp_path):
    """configured yaml=1, artifact k=contract=5, judge hint lifts runtime to 5."""
    from openpi.cache.components.surface_judge import SurfaceJudge, load_surface_artifact

    path = make_artifact(tmp_path, uses_disagreement=True, k=5,
                         v_edges=np.array([0.0, 1.0, 2.0]),
                         s_min_full=np.array([0.9, 0.95]),
                         s_min_warm=np.array([0.8, 0.85]),
                         retrieval_contract={**_contract(), "top_k": 5})
    art = load_surface_artifact(path)
    assert art.k == 5 and art.retrieval_contract["top_k"] == 5
    assert SurfaceJudge(path).min_required_top_k == 5


def test_s0_declares_width_one_and_hints_one(tmp_path):
    from openpi.cache.components.surface_judge import SurfaceJudge, load_surface_artifact

    path = make_artifact(tmp_path, uses_disagreement=False, k=1,
                         retrieval_contract={**_contract(), "top_k": 1})
    assert load_surface_artifact(path).k == 1
    assert SurfaceJudge(path).min_required_top_k == 1
