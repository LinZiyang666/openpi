"""run_gtp --checkpoint validation and the serving-side CP2 stage-placement guard."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from exp.actioncache_baseline import libs
from exp.actioncache_baseline.export_arms import cp2_arm_yaml
from exp.gate_threshold_pareto.run_gtp import validate_arms

PROJ = libs.ProjectionArgs(seed=7, d=8, p=0.25, input_dim=64)


def _cp1_doc():
    return {
        "enabled": True,
        "keys": {"robot_state": {"enabled": True, "weight": 1.0}},
        "key_builder": {"type": "placeholder"},
        "checkpoints": {"cp1": {"enabled": True, "gate": {"type": "always_search"},
                                "judge": {"type": "threshold", "threshold": 0.9},
                                "search_strategy": {"type": "weighted_rrf_knn", "top_k": 1}}},
        "backend": {"type": "in_memory", "vector_dims": {"robot_state": 32}},
        "write_policy": {"type": "never"},
    }


def _write(tmp_path, doc, name):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return {"arm": name[:-5], "yaml": str(p), "suite": "libero_spatial"}


def test_validate_arms_default_checkpoint_is_cp1_non_regression(tmp_path):
    row = _write(tmp_path, _cp1_doc(), "cp1arm.yaml")
    paths = validate_arms([row], phase="eval", eval_gate="always_search")
    assert paths == {"cp1arm": row["yaml"]}
    with pytest.raises(SystemExit, match="missing cp2"):
        validate_arms([row], phase="eval", eval_gate="always_search", checkpoint="cp2")


N0 = "acb_sp_lib50_n0_ir60"
N1 = "acb_sp_lib50_n1_ir60"


def _n0():
    return cp2_arm_yaml(preload_path="/x.pkl", projection=PROJ, tier="n0", theta_raw=0.85)


def _n1():
    return cp2_arm_yaml(preload_path="/x.pkl", projection=PROJ, tier="n1", theta_raw=0.85)


def test_validate_arms_cp2_arms(tmp_path):
    n0 = _write(tmp_path, _n0(), f"{N0}.yaml")
    n1 = _write(tmp_path, _n1(), f"{N1}.yaml")
    paths = validate_arms([n0, n1], phase="eval", eval_gate="always_search",
                          warm_tiers=(libs.WARM_START_T,), checkpoint="cp2")
    assert set(paths) == {N0, N1}
    # The GTP warm-tier prohibition still bites when no ladder is declared.
    with pytest.raises(SystemExit, match="warm tier present"):
        validate_arms([n1], phase="eval", eval_gate="always_search", checkpoint="cp2")
    # A CP2 arm is not a CP1 arm.
    with pytest.raises(SystemExit, match="missing cp1"):
        validate_arms([n0], phase="eval", eval_gate="always_search")
    with pytest.raises(SystemExit, match="checkpoint must be one of"):
        validate_arms([n0], phase="eval", checkpoint="cp3")


def _sn(d):
    return d["checkpoints"]["cp2"]["search_strategy"]


@pytest.mark.parametrize(
    "name, doc_fn, mutate, fragment",
    [
        # protocol clauses that pass config validation but break the frozen arm contract
        ("top_k", _n0, lambda d: _sn(d).update({"top_k": 2}), "top_k 2 != 1"),
        ("step_filter", _n0, lambda d: _sn(d).update({"step_filter": "exact"}), "step_filter"),
        ("task_scoped", _n0, lambda d: _sn(d).update({"task_scoped": True}), "task_scoped"),
        ("cache_disabled", _n0, lambda d: d.update({"enabled": False}), "config must be enabled"),
        ("key_weight", _n0, lambda d: d["keys"]["vlm_out"].update({"weight": 2.0}),
         "weight 2.0 != 1.0"),
        ("similarity", _n0, lambda d: _sn(d).update({"field_similarity": {"vlm_out": {"type": "l2"}}}), "cosine"),
        ("normalization", _n0, lambda d: _sn(d)["score_normalization"]["fields"].update(
            {"vlm_out": {"method": "affine_clip", "params": {"lo": 0.0, "hi": 1.0}}}), "score_normalization"),
        ("n0_with_tier", _n0, lambda d: d["checkpoints"]["cp2"]["judge"].update(
            {"warm_tiers": [{"threshold": 0.9, "start_t": 0.1}]}), "judge shape"),
        ("n1_full_reachable", _n1, lambda d: d["checkpoints"]["cp2"]["judge"].update({"threshold": 0.99}), "judge shape"),
        ("n1_full_noncanonical", _n1,
         lambda d: d["checkpoints"]["cp2"]["judge"].update({"threshold": 2.0}), "threshold == 1.5"),
        ("n1_wrong_start_t", _n1, lambda d: d["checkpoints"]["cp2"]["judge"].update(
            {"warm_tiers": [{"threshold": 0.925, "start_t": 0.3}]}), "judge shape"),
        ("n1_two_tiers", _n1, lambda d: d["checkpoints"]["cp2"]["judge"].update(
            {"warm_tiers": [{"threshold": 0.925, "start_t": 0.1}, {"threshold": 0.9, "start_t": 0.3}]}), "judge shape"),
    ],
)
def test_validate_arms_cp2_contract_rejects(tmp_path, name, doc_fn, mutate, fragment):
    doc = doc_fn()
    mutate(doc)
    arm = N0 if doc_fn is _n0 else N1
    row = _write(tmp_path, doc, f"{arm}.yaml")
    with pytest.raises(SystemExit) as exc:
        validate_arms([row], phase="eval", eval_gate="always_search",
                      warm_tiers=(libs.WARM_START_T, 0.3), checkpoint="cp2")
    assert "CP2 contract violated" in str(exc.value) and fragment in str(exc.value), (name, str(exc.value))


def test_validate_arms_cp2_arm_id_must_match_shape_and_suite(tmp_path):
    # n1 judge shape under an n0 arm id
    row = _write(tmp_path, _n1(), f"{N0}.yaml")
    with pytest.raises(SystemExit, match="arm id tier 'n0' != judge shape 'n1'"):
        validate_arms([row], phase="eval", eval_gate="always_search", warm_tiers=(0.1,), checkpoint="cp2")
    # spatial arm id in a libero_10 matrix row
    row = {**_write(tmp_path, _n0(), f"{N0}.yaml"), "suite": "libero_10"}
    with pytest.raises(SystemExit, match="suite tag"):
        validate_arms([row], phase="eval", eval_gate="always_search", checkpoint="cp2")
    # not an acb arm id at all
    row = _write(tmp_path, _n0(), "gtp_something.yaml")
    with pytest.raises(SystemExit, match="arm id is not acb_"):
        validate_arms([row], phase="eval", eval_gate="always_search", checkpoint="cp2")


def test_serve_policy_cp2_stage_placement_guard():
    from scripts.serve_policy import _validate_cp2_stage_placement

    cp2_cfg = SimpleNamespace(checkpoints={"cp2": object()})
    cp1_cfg = SimpleNamespace(checkpoints={"cp1": object()})
    real = SimpleNamespace(stage1="cuda:0", stage2="cuda:0", stage3="cuda:0")
    _validate_cp2_stage_placement(cp2_cfg, real)
    _validate_cp2_stage_placement(cp2_cfg, None)
    for bad in (SimpleNamespace(stage1="cuda:0", stage2="meta", stage3="cuda:0"),
                SimpleNamespace(stage1="cuda:0", stage2="cuda:0", stage3="meta")):
        with pytest.raises(ValueError, match="real devices"):
            _validate_cp2_stage_placement(cp2_cfg, bad)
        _validate_cp2_stage_placement(cp1_cfg, bad)  # legacy configs untouched
