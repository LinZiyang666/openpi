"""X15 U2/U6 — the P0-b emitter and the launch gate that precedes evaluation.

The launch gate is the half of the pre-registration protocol that can actually
prevent something: an analysis-time check can only refuse to quote a result that
already exists, by which point the parameters could have been chosen from the
episodes. These tests therefore assert refusal *at the entry point*, and that a
refused launch leaves no trace that would make the pool look consumed.

Key dependencies: ``exp.rl_router.emit_p0b_yamls``,
``exp.rl_router.launch_evaluation``.
"""

from __future__ import annotations

import json

import pytest
import yaml

from exp.rl_router.analysis.cluster_stats import PreregistrationError, freeze_parameter
from exp.rl_router.emit_p0b_yamls import (
    build_blind_yaml,
    build_risk_router_yaml,
    build_threshold_yaml,
    emit,
)
from exp.rl_router.launch_evaluation import clear_for_launch


def _base() -> dict:
    """A minimal base that the REAL loader accepts.

    weighted-score-sum needs an in-memory backend and per-field normalization;
    a base that cannot load would make the emitter's own validation vacuous.
    """
    return {
        "enabled": True,
        "keys": {"robot_state": {"enabled": True, "weight": 1.0}},
        "key_builder": {"type": "placeholder"},
        "backend": {"type": "in_memory", "vector_dims": {"robot_state": 32}},
        "checkpoints": {
            "cp1": {
                "search_strategy": {
                    "type": "weighted_score_sum_knn",
                    "top_k": 5,
                    "field_similarity": {"robot_state": {"type": "cosine"}},
                    "score_normalization": {
                        "type": "per_field",
                        "fields": {
                            "robot_state": {
                                "method": "zscore",
                                "params": {"mu": 0.0, "sigma": 1.0},
                            }
                        },
                    },
                }
            }
        },
    }


# ------------------------------------------------------------------
# Emitter (U2)
# ------------------------------------------------------------------


def test_threshold_config_uses_the_tier_judge() -> None:
    cfg = build_threshold_yaml(_base(), threshold=0.95, dump_dir="/tmp/d")
    assert cfg["checkpoints"]["cp1"]["judge"]["type"] == "threshold"
    assert cfg["checkpoints"]["cp1"]["judge"]["threshold"] == pytest.approx(0.95)


def test_every_emitted_config_can_build_the_top_k_features() -> None:
    """A top-1 search cannot produce the top-k score features, so the emitter
    raises the depth rather than emitting a config that fails at load."""
    for cfg in (
        build_threshold_yaml(_base(), threshold=0.9, dump_dir="/tmp/d"),
        build_blind_yaml(_base(), p=0.3, dump_dir="/tmp/d", seed=0),
    ):
        assert cfg["checkpoints"]["cp1"]["search_strategy"]["top_k"] >= 5


def test_risk_router_config_spells_out_both_replan_intervals() -> None:
    """The library's interval has no safe default: assuming it matches the
    client's silently rescales every phase feature."""
    cfg = build_risk_router_yaml(
        _base(),
        risk_model_path="/tmp/r.pt",
        tau=0.5,
        task_index=2,
        replan_steps=5,
        library_replan_steps=10,
    )
    judge = cfg["checkpoints"]["cp1"]["judge"]
    assert judge["replan_steps"] == 5
    assert judge["library_replan_steps"] == 10


def test_shadow_collection_is_opt_in() -> None:
    """A measurement-only sweep must not pay for a teacher forward per step."""
    plain = build_risk_router_yaml(
        _base(),
        risk_model_path="/tmp/r.pt",
        tau=0.5,
        task_index=0,
        replan_steps=5,
        library_replan_steps=5,
    )
    assert "shadow_teacher" not in plain

    collecting = build_risk_router_yaml(
        _base(),
        risk_model_path="/tmp/r.pt",
        tau=0.5,
        task_index=0,
        replan_steps=5,
        library_replan_steps=5,
        shadow_path="/tmp/s.jsonl",
    )
    assert collecting["shadow_teacher"] == {"enabled": True, "path": "/tmp/s.jsonl"}


def test_emit_writes_the_matched_sweep(tmp_path) -> None:
    base_path = tmp_path / "arm.yaml"
    base_path.write_text(yaml.safe_dump(_base()), encoding="utf-8")

    dump_root = tmp_path / "dump"
    out_dir = tmp_path / "out"
    written = emit(
        str(base_path),
        str(out_dir),
        shares=[0.25, 0.40],
        thresholds=[0.90, 0.95],
        dump_root=str(dump_root),
    )
    assert len(written) == 4
    for name in ("threshold_0.900", "threshold_0.950", "blind_0.25", "blind_0.40"):
        assert (dump_root / name).is_dir()
    assert (out_dir / "weights" / "constant_p0.25.pt").is_file()
    assert (out_dir / "weights" / "constant_p0.40.pt").is_file()
    # Every emitted file must parse and carry a judge.
    for path in written:
        cfg = yaml.safe_load(open(path, encoding="utf-8"))
        assert "judge" in cfg["checkpoints"]["cp1"]


# ------------------------------------------------------------------
# Launch gate (U6)
# ------------------------------------------------------------------


def _ledger(tmp_path, *, frozen: bool = True) -> str:
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({"gradient": ["g"], "a": ["a1"]}), encoding="utf-8")
    if frozen:
        freeze_parameter(str(path), "p_hat", 0.29, at="2026-08-23T09:00:00Z")
    return str(path)


def test_launch_is_refused_when_p_hat_is_not_frozen(tmp_path) -> None:
    """The refusal that matters: it happens before any episode runs."""
    ledger = _ledger(tmp_path, frozen=False)
    with pytest.raises(PreregistrationError, match="is not frozen"):
        clear_for_launch(
            ledger,
            pool="a",
            requires=["p_hat"],
            phase="evaluate_a",
            at="2026-08-23T10:00:00Z",
        )


def test_a_refused_launch_leaves_no_touch_stamp(tmp_path) -> None:
    """Otherwise the pool would look consumed by a run that never happened."""
    ledger = _ledger(tmp_path, frozen=False)
    with pytest.raises(PreregistrationError):
        clear_for_launch(
            ledger,
            pool="a",
            requires=["p_hat"],
            phase="evaluate_a",
            at="2026-08-23T10:00:00Z",
        )
    assert "touched" not in json.loads(open(ledger, encoding="utf-8").read())


def test_a_cleared_launch_stamps_first_touch(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    out = clear_for_launch(
        ledger,
        pool="a",
        requires=["p_hat"],
        phase="evaluate_a",
        at="2026-08-23T10:00:00Z",
    )
    assert out["touched"]["a"] == "2026-08-23T10:00:00Z"


def test_a_second_launch_on_the_same_pool_is_refused(tmp_path) -> None:
    """Each evaluation pool is measured once; a relaunch would make the
    reported result a best-of-N."""
    ledger = _ledger(tmp_path)
    clear_for_launch(
        ledger,
        pool="a",
        requires=["p_hat"],
        phase="evaluate_a",
        at="2026-08-23T10:00:00Z",
    )
    with pytest.raises(PreregistrationError, match="already first touched"):
        clear_for_launch(
            ledger,
            pool="a",
            requires=["p_hat"],
            phase="evaluate_a",
            at="2026-08-23T11:00:00Z",
        )


def test_a_freeze_stamped_after_the_launch_is_refused(tmp_path) -> None:
    """Back-filling the ledger cannot buy a pass: the times are compared."""
    path = tmp_path / "ledger.json"
    path.write_text(
        json.dumps(
            {
                "a": ["a1"],
                "frozen": {"p_hat": {"value": 0.29, "at": "2026-08-23T12:00:00Z"}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PreregistrationError, match="not before"):
        clear_for_launch(
            str(path),
            pool="a",
            requires=["p_hat"],
            phase="evaluate_a",
            at="2026-08-23T10:00:00Z",
        )


def test_freezing_after_a_touch_is_refused(tmp_path) -> None:
    """Closes the other direction: once A has been seen, nothing more may be
    frozen, because it could have been chosen from what A showed."""
    ledger = _ledger(tmp_path)
    clear_for_launch(
        ledger,
        pool="a",
        requires=["p_hat"],
        phase="evaluate_a",
        at="2026-08-23T10:00:00Z",
    )
    with pytest.raises(PreregistrationError, match="already touched"):
        freeze_parameter(ledger, "tau_star", 0.61, at="2026-08-23T11:00:00Z")


def test_a_fitting_pool_launch_needs_no_freeze(tmp_path) -> None:
    """Training on the gradient slice is not an evaluation and is not gated."""
    ledger = _ledger(tmp_path, frozen=False)
    clear_for_launch(
        ledger,
        pool="gradient",
        requires=["p_hat"],
        phase="train",
        at="2026-08-23T10:00:00Z",
    )


def test_threshold_recipe_enables_the_real_dump_wrapper() -> None:
    """``judge.dump`` is what makes _build_judge wrap in a DumpingJudge; a bare
    ``dump_dir`` means nothing to a threshold judge and would collect nothing."""
    cfg = build_threshold_yaml(_base(), threshold=0.95, dump_dir="/tmp/d")
    dump = cfg["checkpoints"]["cp1"]["judge"].get("dump")
    assert dump and dump["path"].endswith(".jsonl") and dump["config_id"]


def test_blind_recipe_is_not_a_degenerate_always_cache() -> None:
    """``constant_arm`` pins one arm for every step, and there is no
    ``constant_p`` field for the loader to read — either way the control would
    stop being a mixture at share p."""
    cfg = build_blind_yaml(
        _base(), p=0.3, dump_dir="/tmp/d", seed=0, weights_root="/tmp/w"
    )
    judge = cfg["checkpoints"]["cp1"]["judge"]
    assert "constant_arm" not in judge
    assert "constant_p" not in judge
    assert judge["mode"] == "sample" and judge["weights_path"].endswith(".pt")


def test_emitted_judge_fields_exist_on_the_config_schema() -> None:
    """Anything the loader does not know is dropped silently, so every key the
    emitter writes must be a real JudgeConfig field."""
    import dataclasses

    from openpi.cache.config import JudgeConfig

    known = {f.name for f in dataclasses.fields(JudgeConfig)}
    for cfg in (
        build_threshold_yaml(_base(), threshold=0.9, dump_dir="/tmp/d"),
        build_blind_yaml(
            _base(), p=0.3, dump_dir="/tmp/d", seed=0, weights_root="/tmp/w"
        ),
        build_risk_router_yaml(
            _base(),
            risk_model_path="/tmp/r.pt",
            tau=0.5,
            task_index=0,
            replan_steps=5,
            library_replan_steps=5,
        ),
    ):
        unknown = set(cfg["checkpoints"]["cp1"]["judge"]) - known
        assert not unknown, f"emitter writes fields the loader drops: {unknown}"


def _routed_base() -> dict:
    """A base yaml shaped like the real X14 arm config: it routes hits to the
    student sidecar."""
    cfg = _base()
    cfg["routing"] = {"hit_to": "student", "student_endpoint": "127.0.0.1:7002"}
    return cfg


def test_inherited_student_routing_is_stripped_from_every_recipe() -> None:
    """Left in place it breaks the two controls in different ways: the blind arm
    (``arms="tc"``, no student) is rejected by config validation outright, and
    the threshold recipe loads fine but executes the STUDENT on every FULL_HIT
    — so the "cache replay baseline" would measure a different policy entirely.
    """
    for cfg in (
        build_threshold_yaml(_routed_base(), threshold=0.95, dump_dir="/tmp/d"),
        build_blind_yaml(
            _routed_base(), p=0.3, dump_dir="/tmp/d", seed=0, weights_root="/tmp/w"
        ),
        build_risk_router_yaml(
            _routed_base(),
            risk_model_path="/tmp/r.pt",
            tau=0.5,
            task_index=0,
            replan_steps=5,
            library_replan_steps=5,
        ),
    ):
        assert "routing" not in cfg


def test_stripping_does_not_mutate_the_caller_s_base() -> None:
    """The emitter derives several recipes from one base; mutating it in place
    would make the result depend on emission order."""
    base = _routed_base()
    build_threshold_yaml(base, threshold=0.9, dump_dir="/tmp/d")
    assert "routing" in base


def test_emit_validates_every_recipe_through_the_real_loader(tmp_path) -> None:
    """Field-name checks cannot catch a config that loads into a different
    policy; only the loader the server uses can."""
    import inspect

    from exp.rl_router import emit_p0b_yamls

    src = inspect.getsource(emit_p0b_yamls._validate_emitted)
    assert "load_cache_config" in src


def test_constant_share_weights_actually_mix_at_p(tmp_path) -> None:
    """The blind control's whole job is to mix at share p; weights that do
    anything else make it a different policy under the same name."""
    import torch

    from exp.rl_router.emit_p0b_yamls import build_constant_share_weights
    from openpi.cache.components.mlp_router_judge import RouterWeights

    path = build_constant_share_weights(0.3, str(tmp_path / "w.pt"))
    weights = RouterWeights.load(path)

    # The formal serving consumer accepts the artifact. Its constant logits are
    # independent of input and softmax reproduces p exactly.
    assert torch.equal(weights.W1, torch.zeros_like(weights.W1))
    probs = torch.softmax(weights.b2, dim=0)
    assert float(probs[0]) == pytest.approx(0.3, abs=1e-6)
    assert weights.arms == "tc" and weights.hidden == 8


def test_a_degenerate_share_is_refused() -> None:
    from exp.rl_router.emit_p0b_yamls import build_constant_share_weights

    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="constant share"):
            build_constant_share_weights(bad, "/tmp/never.pt")


def test_emission_builds_the_weights_its_recipes_reference(tmp_path) -> None:
    """A dangling weights path surfaces only at server start, after the
    campaign has been scheduled around it."""
    import yaml as _yaml

    base_path = tmp_path / "arm.yaml"
    base_path.write_text(_yaml.safe_dump(_base()), encoding="utf-8")
    dump_root = tmp_path / "dump"
    for name in ("blind_0.30",):
        (dump_root / name).mkdir(parents=True, exist_ok=True)
    weights_root = tmp_path / "w"

    emit(
        str(base_path),
        str(tmp_path / "out"),
        shares=[0.30],
        thresholds=[],
        dump_root=str(dump_root),
        weights_root=str(weights_root),
    )

    assert (weights_root / "constant_p0.30.pt").exists()


def test_a_missing_weights_artifact_fails_emission(tmp_path) -> None:
    """Validation must cover the files a config points at, not just its shape."""
    import yaml as _yaml

    from exp.rl_router.emit_p0b_yamls import _validate_emitted

    cfg = build_blind_yaml(
        _base(), p=0.3, dump_dir=str(tmp_path), seed=0, weights_root="/nonexistent"
    )
    path = tmp_path / "blind.yaml"
    path.write_text(_yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match="do not exist"):
        _validate_emitted([str(path)])


def test_the_launcher_wraps_the_runner_rather_than_only_checking(tmp_path) -> None:
    """A separate check-then-run step can always be skipped; a wrapper cannot.

    The gate must be on the path that actually starts the runner.
    """
    import inspect

    from exp.rl_router import launch_evaluation

    src = inspect.getsource(launch_evaluation.main)
    assert "execvp" in src, "the cleared launch must exec the runner itself"
    assert "REMAINDER" in src, "the runner command must be accepted"


def test_a_missing_ledger_refuses_rather_than_assuming_the_good_case(tmp_path) -> None:
    with pytest.raises(PreregistrationError, match="does not exist"):
        clear_for_launch(
            str(tmp_path / "nope.json"),
            pool="a",
            requires=["p_hat"],
            phase="evaluate_a",
            at="2026-08-23T10:00:00Z",
        )
