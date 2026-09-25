"""Plan resolution, arm coverage, grids and self-start seeds (plan §4.3, §4.4, §9.1, §9.4)."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from exp.step_diag import envs as E
from exp.step_diag import recorder as R
from exp.step_diag.serve_diag_groot import GROOT_VARIANT_MODES
from openpi.cache.types import PI05_V1, groot_n15_schedule
from openpi.cache.warm_reset.types import (
    EpisodeDigestSeed,
    SelfStartPlan,
    WarmResetPlan,
    groot_loop_grid,
    native_grid,
    private_noise,
    resolve_plan,
    resolve_self_plan,
    stable_digest_int,
    validate_plan,
    validate_self_plan,
)
from openpi.models_pytorch.pi0_pytorch import _warm_start_num_steps
from tests.cache.warm_reset._support import BASE_MODES, spec_for, split_mode, warm_arms


def _schedule(policy: str, arm_id: str):
    if policy == "pi05":
        return PI05_V1
    return groot_n15_schedule(8 if E.warm_steps_of(arm_id) is not None else 4)


# ------------------------------------------------------------------
# §4.3.4 table: coverage and agreement with the step_diag constants
# ------------------------------------------------------------------


def test_every_step_diag_warm_family_arm_has_a_spec():
    arms = warm_arms()
    assert len(arms) > 40
    for policy, arm in arms:
        spec = spec_for(policy, arm)
        if E.warm_mode_of(arm) == "warm":
            assert spec is None, arm  # the exact resume is the absent block
        else:
            assert spec is not None, arm
    # every served mode of the reference is in the table
    modes = {*E.WARM_VARIANT_MODES, *E.SELF_VARIANT_MODES, *E.GROOT_SHOOT_MODES, *E.SELF_SHOOT_MODES}
    assert {split_mode(m)[1] for m in modes} <= set(BASE_MODES)


@pytest.mark.parametrize("policy", ["pi05", "groot"])
def test_table_levels_are_the_step_diag_constants(policy):
    variants = (
        {**E.WARM_VARIANT_MODES, **E.SELF_VARIANT_MODES} if policy == "pi05" else GROOT_VARIANT_MODES
    )
    for mode, variant in variants.items():
        _, base = split_mode(mode)
        point, kind, levels = BASE_MODES[base]
        if variant in E.MID_ENTRY_T_BY_VARIANT:
            want = E.MID_ENTRY_T_BY_VARIANT[variant][policy]
        elif variant in E.SHOOT_ENTRY_T:
            want = E.SHOOT_ENTRY_T[variant]
        else:
            want = 1.0
        assert levels[policy] == want, mode
        assert kind == ("shoot" if variant in E.SHOOT_ENTRY_T else "reset"), mode
        final = variant in ("reset_final", "mid_final", "mid_final50")
        assert point == ("final" if final else "snapshot"), mode


@pytest.mark.parametrize("policy,arm", warm_arms())
def test_resolved_plan_matches_the_arm(policy, arm):
    spec = spec_for(policy, arm)
    if spec is None:
        return
    schedule = _schedule(policy, arm)
    start_t = E.warm_t_of(arm)
    plan = resolve_plan(spec, schedule, start_t)
    n = E.warm_steps_of(arm)
    assert plan.n_steps == (schedule.remaining_steps(start_t) if n is None else n)
    assert (plan.kind, plan.source, plan.point, plan.level) == (spec.kind, spec.source, spec.point, spec.level)
    assert len(plan.flow_times()) == plan.n_steps
    if spec.self_start:
        sp = resolve_self_plan(spec, schedule, start_t)
        want = schedule.snapshot_index(start_t) if spec.point == "snapshot" else None
        assert sp.capture_index == want and sp.grid_key() == (schedule.schedule_id, schedule.num_steps)


@pytest.mark.parametrize("arm,t", [
    ("warmreset_t0.2", [1.0, 0.5]),
    ("resetfinal_t0.2", [1.0, 0.5]),
    ("midfinal_t0.2", [0.9, 0.45]),
    ("midfinal50_t0.2", [0.5, 0.25]),
    ("midreset_t0.2", [0.9, 0.45]),
    ("midreset50_t0.2", [0.5, 0.25]),
    ("warmshoot_t0.2", [0.2, -0.3]),
])
def test_pi05_flow_times_follow_the_table(arm, t):
    plan = resolve_plan(spec_for("pi05", arm), PI05_V1, 0.2)
    assert plan.flow_times() == pytest.approx(t, abs=1e-6)


@pytest.mark.parametrize("policy_k,arm,start_t,tau", [
    (4, "warmreset_t0.75", 0.75, [0.0]),
    (4, "warmreset_t0.5", 0.5, [0.0, 0.5]),
    (4, "midreset_t0.5", 0.5, [0.25, 0.625]),
    (4, "midreset50_t0.5", 0.5, [0.5, 0.75]),
    (4, "warmshoot_t0.5", 0.5, [0.5, 1.0]),
    (4, "midshoot_t0.5", 0.5, [0.5, 0.875]),
    (4, "midshoot50_t0.75", 0.75, [0.75]),
    (8, "midreset_t0.75_n1", 0.75, [0.25]),
    (8, "warmreset_t0.5_n2", 0.5, [0.0, 0.5]),
])
def test_groot_native_grid_follows_the_table(policy_k, arm, start_t, tau):
    plan = resolve_plan(spec_for("groot", arm), groot_n15_schedule(policy_k), start_t)
    taus, _ = native_grid(plan)
    assert list(taus) == pytest.approx(tau)
    assert plan.flow_times() == pytest.approx([1.0 - x for x in tau])
    assert (groot_loop_grid(plan)[0] is None) == (plan.kind == "reset" and plan.level == 1.0)


@pytest.mark.parametrize("t", PI05_V1.timesteps)
def test_pi05_index_arithmetic_equals_the_model_formulas(t):
    assert PI05_V1.snapshot_index(t) == round((1.0 - t) * 10)
    assert PI05_V1.remaining_steps(t) == _warm_start_num_steps(t, 10)


# ------------------------------------------------------------------
# Grid keys
# ------------------------------------------------------------------


def test_grid_key_ignores_source_point_and_reset_start_t():
    snap = resolve_plan(spec_for("pi05", "warmreset_t0.2"), PI05_V1, 0.2)
    final = resolve_plan(spec_for("pi05", "resetfinal_t0.2"), PI05_V1, 0.2)
    self_ = resolve_plan(spec_for("pi05", "selfresetfinal_t0.2"), PI05_V1, 0.2)
    assert snap.grid_key() == final.grid_key() == self_.grid_key()
    # different N, level or schedule -> different grid
    assert resolve_plan(spec_for("pi05", "warmreset_t0.3"), PI05_V1, 0.3).grid_key() != snap.grid_key()
    assert resolve_plan(spec_for("pi05", "midreset_t0.2"), PI05_V1, 0.2).grid_key() != snap.grid_key()
    # a reset grid does not depend on start_t; a shoot grid does
    k4 = groot_n15_schedule(4)
    a = resolve_plan(dataclasses.replace(spec_for("groot", "warmreset_t0.75"), num_steps=1), k4, 0.75)
    b = resolve_plan(dataclasses.replace(spec_for("groot", "warmreset_t0.75"), num_steps=1), k4, 0.5)
    assert a.grid_key() == b.grid_key()
    s1 = resolve_plan(dataclasses.replace(spec_for("groot", "warmshoot_t0.75"), num_steps=1), k4, 0.75)
    s2 = resolve_plan(dataclasses.replace(spec_for("groot", "warmshoot_t0.75"), num_steps=1), k4, 0.5)
    assert s1.grid_key() != s2.grid_key()
    k8 = resolve_plan(spec_for("groot", "warmreset_t0.75_n1"), groot_n15_schedule(8), 0.75)
    assert k8.grid_key() != a.grid_key()  # K is part of the key


# ------------------------------------------------------------------
# Validation of resolver input and of hand-built plans
# ------------------------------------------------------------------


def test_explicit_n_does_not_skip_the_recoverable_point_check():
    spec = spec_for("groot", "midreset_t0.75_n1")
    with pytest.raises(ValueError, match="not a recoverable"):
        resolve_plan(spec, groot_n15_schedule(8), 0.7)
    with pytest.raises(ValueError, match="not a recoverable"):
        resolve_plan(spec_for("pi05", "warmreset_t0.2"), PI05_V1, 0.25)
    with pytest.raises(ValueError):
        resolve_self_plan(spec_for("pi05", "warmreset_t0.2"), PI05_V1, 0.2)  # cache spec


@pytest.mark.parametrize("change", [
    {"n_steps": True},
    {"n_steps": 0},
    {"n_steps": 11},
    {"k": True},
    {"k": 4},
    {"schedule_id": "groot_n15_k10_v1"},
    {"start_t": 0.25},
    {"level": 1},
    {"level": 1.5},
    {"kind": "exact"},
    {"point": "final", "kind": "shoot"},
])
def test_hand_built_continuation_plans_are_refused(change):
    plan = resolve_plan(spec_for("pi05", "warmshoot_t0.2"), PI05_V1, 0.2)
    with pytest.raises(ValueError):
        validate_plan(dataclasses.replace(plan, **change), PI05_V1)


@pytest.mark.parametrize("change", [
    {"capture_index": True},
    {"capture_index": 7},
    {"start_t": 0.25, "capture_index": None},
    {"k": 9},
])
def test_hand_built_self_plans_are_refused(change):
    plan = resolve_self_plan(spec_for("pi05", "selfwarmreset_t0.2"), PI05_V1, 0.2)
    assert plan == SelfStartPlan("pi05_v1", PI05_V1.direction, 10, 0.2, 8)
    validate_self_plan(plan, PI05_V1)
    with pytest.raises(ValueError):
        validate_self_plan(dataclasses.replace(plan, **change), PI05_V1)
    with pytest.raises(ValueError):
        validate_plan(plan, PI05_V1)  # wrong plan type for the entry
    with pytest.raises(ValueError):
        validate_self_plan(resolve_plan(spec_for("pi05", "warmreset_t0.2"), PI05_V1, 0.2), PI05_V1)


def test_plans_are_frozen_and_hashable():
    plan = resolve_plan(spec_for("pi05", "warmreset_t0.2"), PI05_V1, 0.2)
    assert isinstance(plan, WarmResetPlan) and hash(plan) == hash(dataclasses.replace(plan))
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.n_steps = 3  # type: ignore[misc]


# ------------------------------------------------------------------
# Seeds and private noise
# ------------------------------------------------------------------

_IDENTITY = {"experiment": "libero_10", "task": "put both pots", "episode_id": 3, "orig_init_state_idx": 3,
             "attempt": 1, "task_uid": "arm_a:eval:4:3", "task_id": 4}


def test_digest_is_the_step_diag_algorithm():
    for parts in [("a", 1, 2.5, None), ("sdiag", "libero_10", "t", (7, 3, "sha"), 1, 0, "self")]:
        assert stable_digest_int(*parts) == R.stable_digest_int(*parts)


def test_private_noise_is_the_step_diag_noise_and_leaves_the_global_rng():
    cpu = torch.get_rng_state()
    assert torch.equal(private_noise(123, (10, 8)), R.make_noise(123, (10, 8)))
    assert private_noise(123, (10, 8)).dtype == torch.float32
    assert torch.equal(torch.get_rng_state(), cpu)


def test_episode_digest_seed_pairs_arms_and_separates_decisions():
    policy = EpisodeDigestSeed("ns", ("experiment", "task", "orig_init_state_idx", "attempt"))
    base = policy.seed(_IDENTITY, 0)
    assert base == stable_digest_int("ns", "libero_10", "put both pots", 3, 1, 0, "self")
    assert policy.seed(dict(_IDENTITY), 0) == base  # deterministic
    assert policy.seed(_IDENTITY, 1) != base
    assert policy.seed({**_IDENTITY, "attempt": 2}, 0) != base
    assert policy.seed({**_IDENTITY, "task": "other"}, 0) != base
    # yaml / bundle identity never enters the seed: every self arm pairs
    assert policy.seed({**_IDENTITY, "task_uid": "arm_b:eval:4:3", "yaml_id": "x", "bundle_id": "y"}, 0) == base
    assert EpisodeDigestSeed("other", policy.keys).seed(_IDENTITY, 0) != base
    with pytest.raises(KeyError):
        policy.seed({"experiment": "e", "task": "t"}, 0)
