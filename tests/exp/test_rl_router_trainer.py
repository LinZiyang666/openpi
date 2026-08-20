"""REINFORCE trainer, warm-start graft, and on-policy parity (X14 M2).

The objective is frozen in the plan, so the tests here are deliberately
value-level rather than behavioural: a refactor that changes the reward by a
constant, drops the baseline, or quietly runs a second epoch would still "work"
and would still be a different experiment.

Covered:
  - the frozen reward / loss formulas, to the number;
  - exactly one Adam step per batch, and idempotent re-entry after a crash;
  - bitwise on-policy verification, including the rejection path;
  - real 65,568-dim fp16 parity between the serving judge and the trainer's
    reference forward — the contract the whole dump format exists to support;
  - uninterrupted-vs-resume equivalence through a checkpoint;
  - the warm-start graft's three mappings and its initial arm probabilities;
  - convergence on a synthetic bandit, so the plumbing is known to learn.
"""

from __future__ import annotations

import json
import math

import pytest
import torch
import yaml

from openpi.cache.components.mlp_router_judge import (
    ARM_SETS,
    FEATURE_DTYPE,
    MlpRouterJudge,
    RouterFeatureEncoder,
    RouterWeights,
    save_router_weights,
)
from openpi.cache.types import CheckpointID, ROBOT_STATE, VISION_0, VISION_1

from exp.rl_router.fit_warmstart import (
    SuccessHead,
    fit_head,
    fit_feature_stats,
    graft,
    graft_disclosure,
    grouped_folds,
    initial_student_bias,
    zero_init,
)
from exp.rl_router.train_router import (
    AdmissionError,
    RouterPolicy,
    RouterTrainer,
    TrainEpisode,
    TrainerHParams,
    episode_reward,
    reinforce_loss,
)

D = 16
H = 8
COSTS = {"teacher": 1.0, "student": 0.2, "cache": 0.02}


def _hparams(**kw) -> TrainerHParams:
    params = {"arm_costs": COSTS, "lam": 0.2, "t_max": 100}
    params.update(kw)
    return TrainerHParams(**params)


def _policy(arms: str = "tsc", seed: int = 0, scale: float = 0.3) -> RouterPolicy:
    torch.manual_seed(seed)
    n = len(ARM_SETS[arms])
    return RouterPolicy(torch.randn(H, D) * scale, torch.zeros(H),
                        torch.randn(n, H) * scale, torch.zeros(n), arms=arms)


def _episode(policy: RouterPolicy | None, *, k: int = 4, success: bool = True,
             arm: str = "teacher", uid: str = "y:eval:0:0", seed: int = 1) -> TrainEpisode:
    """One rollout. ``policy=None`` omits the behaviour logits, which is how a
    synthetic episode says "there is nothing to verify against"."""
    g = torch.Generator().manual_seed(seed)
    features = torch.randn(k, D, generator=g).to(FEATURE_DTYPE)
    return TrainEpisode(
        task_uid=uid, attempt=1, success=success, features=features,
        arm_sampled=[arm] * k, arm_executed=[arm] * k,
        dumped_logits=None if policy is None else policy.reference_logits(features),
    )


# ---------------------------------------------------------------------------
# Frozen formulas
# ---------------------------------------------------------------------------


def test_episode_reward_matches_the_frozen_formula() -> None:
    r = episode_reward(success=True, arm_executed=["teacher", "cache", "student"],
                       arm_costs=COSTS, lam=0.5, t_max=10)
    assert r == pytest.approx(1.0 - 0.5 * (1.0 + 0.02 + 0.2) / 10)

    # Failure contributes 0, not -1: the cost term is the only penalty.
    r_fail = episode_reward(success=False, arm_executed=["teacher"],
                            arm_costs=COSTS, lam=0.5, t_max=10)
    assert r_fail == pytest.approx(-0.05)


def test_reward_bills_the_executed_arm_not_the_sampled_one() -> None:
    """A cache arm that hit an empty library ran the teacher; charging it the
    cache price would make the router look cheaper than it is."""
    executed_as_teacher = episode_reward(success=True, arm_executed=["teacher"] * 3,
                                         arm_costs=COSTS, lam=1.0, t_max=10)
    executed_as_cache = episode_reward(success=True, arm_executed=["cache"] * 3,
                                       arm_costs=COSTS, lam=1.0, t_max=10)
    assert executed_as_teacher < executed_as_cache


def test_reward_rejects_an_unmeasured_arm() -> None:
    with pytest.raises(ValueError, match="microbench"):
        episode_reward(success=True, arm_executed=["student"],
                       arm_costs={"teacher": 1.0}, lam=0.1, t_max=10)


def test_reinforce_loss_matches_the_frozen_formula() -> None:
    logprob_sums = torch.tensor([-2.0, -4.0], requires_grad=True)
    rewards = torch.tensor([1.0, 0.0])
    entropy_mean = torch.tensor(0.5)
    loss, advantages = reinforce_loss(
        logprob_sums=logprob_sums, rewards=rewards,
        entropy_mean=entropy_mean, entropy_beta=0.01,
    )
    # baseline = 0.5 -> advantages = [+0.5, -0.5]
    assert advantages.tolist() == [0.5, -0.5]
    expected = -((0.5 * -2.0) + (-0.5 * -4.0)) / 2 - 0.01 * 0.5
    assert float(loss) == pytest.approx(expected)


def test_baseline_is_detached_from_the_gradient() -> None:
    """The batch mean is a variance-reduction constant; letting gradients flow
    through it silently changes the estimator."""
    logprob_sums = torch.tensor([-2.0, -4.0], requires_grad=True)
    loss, _ = reinforce_loss(
        logprob_sums=logprob_sums, rewards=torch.tensor([1.0, 0.0]),
        entropy_mean=torch.tensor(0.0), entropy_beta=0.0,
    )
    loss.backward()
    # d/dlogp_i of -(1/N) * sum(adv_i * logp_i) = -adv_i / N
    assert logprob_sums.grad.tolist() == pytest.approx([-0.25, 0.25])


# ---------------------------------------------------------------------------
# One batch, one step
# ---------------------------------------------------------------------------


def test_exactly_one_optimizer_step_per_batch() -> None:
    """Two batches, two steps. A second epoch over one batch would make the
    data off-policy for the updated parameters, which plain REINFORCE has no
    correction for — so the count is the contract."""
    trainer = RouterTrainer(_policy(), _hparams())
    for batch in range(2):
        # Fresh episodes per batch: after the first step the previous batch's
        # recorded logits are, correctly, no longer on-policy.
        episodes = [_episode(trainer.policy, uid=f"b{batch}u{i}", seed=batch * 10 + i,
                             success=bool(i % 2)) for i in range(4)]
        trainer.train_batch(episodes, batch_id=f"b{batch}")
        assert trainer.optimizer.state_dict()["state"][0]["step"].item() == batch + 1


def test_weights_version_advances_once_per_batch() -> None:
    trainer = RouterTrainer(_policy(), _hparams(), weights_version="v3")
    trainer.train_batch([_episode(trainer.policy)], batch_id="b0")
    assert trainer.weights_version == "v4"


def test_replaying_a_consumed_batch_is_a_noop() -> None:
    """Trainer crash between the Adam step and the checkpoint write: the batch
    is re-pushed and must not be applied twice."""
    trainer = RouterTrainer(_policy(), _hparams())
    ep = [_episode(trainer.policy)]
    trainer.train_batch(ep, batch_id="b0")
    before = trainer.policy.W2.data.clone()
    metrics = trainer.train_batch(ep, batch_id="b0")
    assert metrics["skipped"] == "already_consumed"
    assert torch.equal(trainer.policy.W2.data, before)


def test_empty_batch_is_refused() -> None:
    trainer = RouterTrainer(_policy(), _hparams())
    with pytest.raises(ValueError, match="empty batch"):
        trainer.train_batch([], batch_id="b0")


# ---------------------------------------------------------------------------
# On-policy verification
# ---------------------------------------------------------------------------


def test_matching_logits_pass_verification() -> None:
    trainer = RouterTrainer(_policy(), _hparams())
    metrics = trainer.train_batch([_episode(trainer.policy)], batch_id="b0")
    assert metrics["n_rejected"] == 0


def test_one_bad_episode_aborts_the_whole_batch_untouched() -> None:
    """§3.5/§3.6: full N or nothing.

    Dropping the failure and stepping on the remainder would divide by a
    smaller N than the one frozen before dispatch — a different estimator
    reported under the same name. The abort must leave parameters, optimizer,
    version and the consumed ledger completely untouched so the caller can
    repair the batch and update once, on the full N.
    """
    trainer = RouterTrainer(_policy(), _hparams(), weights_version="v7")
    good = _episode(trainer.policy, uid="good")
    stale = _episode(_policy(seed=99), uid="stale", seed=2)
    before = {n: p.detach().clone() for n, p in trainer.policy.named_parameters()}

    with pytest.raises(AdmissionError) as excinfo:
        trainer.train_batch([good, stale], batch_id="b0")

    assert [r["task_uid"] for r in excinfo.value.rejected] == ["stale"]
    assert excinfo.value.rejected[0]["reason"] == "logits_not_bitwise_equal"
    for name, param in trainer.policy.named_parameters():
        assert torch.equal(param.detach(), before[name]), name
    assert trainer.weights_version == "v7"
    assert trainer.consumed_batches == []
    assert trainer.optimizer.state_dict()["state"] == {}
    # The repaired batch then updates once, on the full N.
    repaired = _episode(trainer.policy, uid="stale_repaired", seed=2)
    trainer.train_batch([good, repaired], batch_id="b0")
    assert trainer.weights_version == "v8"


def test_verification_is_bitwise_not_approximate() -> None:
    """A one-ULP perturbation must fail: an "almost equal" behaviour policy is
    exactly the silent-drift failure the check exists to catch."""
    trainer = RouterTrainer(_policy(), _hparams())
    ep = _episode(trainer.policy)
    ep.dumped_logits = ep.dumped_logits.clone()
    ep.dumped_logits[0, 0] = torch.nextafter(ep.dumped_logits[0, 0], torch.tensor(1e30))
    with pytest.raises(AdmissionError):
        trainer.train_batch([ep, _episode(trainer.policy, uid="ok", seed=5)], batch_id="b0")


def test_sampled_logprob_is_verified_not_just_the_logits() -> None:
    """The logits pin the forward pass; the sampled log-probability pins the
    sampling the gradient is weighted by. A temperature drift leaves the logits
    identical and makes every recorded action's probability wrong."""
    trainer = RouterTrainer(_policy(), _hparams())
    ep = _episode(trainer.policy)
    logp = torch.log_softmax(ep.dumped_logits, dim=1)
    idx = torch.tensor([trainer.policy.arms.index(a) for a in ep.arm_sampled])
    ep.dumped_logprobs = logp[torch.arange(len(idx)), idx]
    trainer.train_batch([ep], batch_id="ok")            # consistent -> admitted

    trainer2 = RouterTrainer(_policy(), _hparams())
    bad = _episode(trainer2.policy)
    bad.dumped_logprobs = torch.full((bad.features.shape[0],), -0.5)
    with pytest.raises(AdmissionError) as excinfo:
        trainer2.train_batch([bad], batch_id="b0")
    assert excinfo.value.rejected[0]["reason"] == "logprob_not_bitwise_equal"


def test_batch_from_another_weights_version_is_refused() -> None:
    """A hot-swap or resume race: crediting this batch's gradient to the
    trainer's version would attribute the rollouts to the wrong policy."""
    trainer = RouterTrainer(_policy(), _hparams(), weights_version="v3")
    with pytest.raises(AdmissionError, match="weights_version"):
        trainer.train_batch([_episode(trainer.policy)], batch_id="b0",
                            expected_weights_version="v2")
    assert trainer.weights_version == "v3" and trainer.consumed_batches == []


def test_all_episodes_rejected_fails_the_batch() -> None:
    trainer = RouterTrainer(_policy(), _hparams())
    with pytest.raises(AdmissionError, match="on-policy verification"):
        trainer.train_batch([_episode(_policy(seed=42))], batch_id="b0")


def test_episode_shape_mismatch_fails_loud() -> None:
    with pytest.raises(ValueError, match="feature rows"):
        TrainEpisode(task_uid="u", attempt=1, success=True,
                     features=torch.zeros(3, D, dtype=FEATURE_DTYPE),
                     arm_sampled=["teacher"] * 3, arm_executed=["teacher"] * 2)


# ---------------------------------------------------------------------------
# Real-dimension parity with the serving judge
# ---------------------------------------------------------------------------


def test_production_dimension_parity_between_judge_and_trainer(tmp_path) -> None:
    """The whole dump format exists so this holds: the trainer's reference
    forward reproduces the judge's logits bitwise at the real 65,568-dim width
    (libero_10 ``cp1_spatial_pool_16``: vision_0 32768 + vision_1 32768 +
    robot_state 32)."""
    dims = {VISION_0: 32768, VISION_1: 32768, ROBOT_STATE: 32}
    dim = sum(dims.values())
    assert dim == 65568
    hidden, arms = 4, "tsc"
    torch.manual_seed(0)
    path = tmp_path / "prod.pt"
    save_router_weights(
        path,
        W1=(torch.randn(hidden, dim) * 0.01), b1=torch.zeros(hidden),
        W2=(torch.randn(3, hidden) * 0.1), b2=torch.zeros(3),
        arms=arms, fields=tuple(dims), dims=dims, weights_version="v1",
        mu=torch.zeros(dim), sigma=torch.ones(dim),
    )
    # Drive the real dump path: the trainer reads shard bytes and sidecar
    # logits off disk, so that is what the parity claim has to cover.
    dump = tmp_path / "dump"
    judge = MlpRouterJudge(arms=arms, weights_path=str(path), hidden=hidden,
                           feature_fields=list(dims), mode="argmax", dump_dir=str(dump))
    judge.on_episode_start(extra_metadata={
        "run_id": "r", "batch_id": "b", "task_uid": "u", "attempt": 1,
    })
    query_keys = {
        VISION_0: torch.randn(32768), VISION_1: torch.randn(32768),
        ROBOT_STATE: torch.randn(32),
    }
    verdict = judge(results=[], checkpoint_id=CheckpointID.CP1, cached_data={},
                    query_keys=query_keys)
    judge.on_episode_end()

    shard_dir = dump / "r" / "b"
    entry = json.loads((shard_dir / "manifest.jsonl").read_text().splitlines()[0])
    raw = (shard_dir / entry["shard"]).read_bytes()
    features = torch.frombuffer(bytearray(raw), dtype=FEATURE_DTYPE).reshape(
        entry["rows"], entry["dim"]
    )
    assert entry["dim"] == 65568
    sidecar = json.loads((shard_dir / entry["sidecar"]).read_text().splitlines()[0])
    dumped = torch.tensor(sidecar["logits"], dtype=torch.float32).reshape(1, 3)

    policy = RouterPolicy.from_weights(RouterWeights.load(str(path)))
    assert torch.equal(policy.reference_logits(features), dumped)

    # And the dumped bytes really are the encoder's output for that observation.
    encoder = RouterFeatureEncoder(tuple(dims), mu=torch.zeros(dim), sigma=torch.ones(dim))
    assert torch.equal(features[0], encoder.encode(query_keys))
    assert verdict.router_outputs["arm_sampled"] in ARM_SETS[arms]


# ---------------------------------------------------------------------------
# Resume equivalence
# ---------------------------------------------------------------------------


def test_uninterrupted_and_resumed_runs_are_identical(tmp_path) -> None:
    """A crash between batches must continue the same experiment, not start a
    statistically similar one — so optimizer moments and RNG ride along."""
    def batches():
        # Behaviour logits omitted: this test is about optimizer + RNG state
        # crossing the checkpoint, and the two runs must see identical inputs.
        return [
            [_episode(None, uid=f"b{b}u{i}", seed=b * 10 + i, success=bool(i % 2))
             for i in range(3)]
            for b in range(3)
        ]

    straight = RouterTrainer(_policy(), _hparams())
    for idx, batch in enumerate(batches()):
        straight.train_batch(batch, batch_id=f"b{idx}")

    interrupted = RouterTrainer(_policy(), _hparams())
    all_batches = batches()
    interrupted.train_batch(all_batches[0], batch_id="b0")
    interrupted.save_checkpoint(tmp_path / "ck.pt")

    resumed = RouterTrainer.load_checkpoint(tmp_path / "ck.pt")
    for idx in (1, 2):
        resumed.train_batch(all_batches[idx], batch_id=f"b{idx}")

    for name in ("W1", "b1", "W2", "b2"):
        assert torch.equal(getattr(straight.policy, name).data,
                           getattr(resumed.policy, name).data), name
    assert resumed.weights_version == straight.weights_version
    assert [b["batch_id"] for b in resumed.consumed_batches] == ["b0", "b1", "b2"]


def test_checkpoint_carries_the_consumed_batch_ledger(tmp_path) -> None:
    trainer = RouterTrainer(_policy(), _hparams())
    trainer.train_batch([_episode(trainer.policy)], batch_id="b0", package_sha256="deadbeef")
    trainer.save_checkpoint(tmp_path / "ck.pt")
    reloaded = RouterTrainer.load_checkpoint(tmp_path / "ck.pt")
    assert reloaded.consumed_batches[0]["package_sha256"] == "deadbeef"
    # Re-pushing the same batch after a resume is still a no-op.
    assert reloaded.train_batch([_episode(reloaded.policy)], batch_id="b0")["skipped"]


def test_exported_weights_reload_into_a_judge(tmp_path) -> None:
    """The training loop's output must be directly servable, or the hot-swap
    would need a conversion step that could drift."""
    dims = {ROBOT_STATE: D}
    trainer = RouterTrainer(_policy(arms="ts"), _hparams(), weights_version="v0")
    trainer.train_batch([_episode(trainer.policy, arm="teacher")], batch_id="b0")
    out = tmp_path / "v1.pt"
    trainer.encoder_meta = {"fields": [ROBOT_STATE], "dims": dims,
                            "mu": [0.0] * D, "sigma": [1.0] * D}
    trainer.export_router_weights(out)
    judge = MlpRouterJudge(arms="ts", weights_path=str(out), hidden=H,
                           feature_fields=[ROBOT_STATE], mode="argmax")
    assert judge.weights_version == "v1"


# ---------------------------------------------------------------------------
# Synthetic convergence
# ---------------------------------------------------------------------------


def test_trainer_learns_a_synthetic_contextual_bandit() -> None:
    """A feature dimension decides which arm succeeds. If the plumbing learns
    this, the gradient signs, the baseline, and the arm indexing all line up;
    if it does not, no amount of real data will help."""
    torch.manual_seed(0)
    policy = RouterPolicy(torch.zeros(H, D), torch.zeros(H),
                          torch.zeros(2, H), torch.zeros(2), arms="ts")
    # A zero trunk has zero gradient; start the trunk random but the head at 0
    # so the initial policy is uniform.
    policy.W1.data = torch.randn(H, D) * 0.5
    trainer = RouterTrainer(
        policy,
        # lr well above the production 3e-4: this is a 60-batch sanity harness,
        # not the real 4k-episode schedule.
        _hparams(arm_costs={"teacher": 0.0, "student": 0.0}, lr=0.05),
    )

    gen = torch.Generator().manual_seed(7)
    for batch in range(60):
        episodes = []
        for i in range(32):
            features = torch.randn(1, D, generator=gen).to(FEATURE_DTYPE)
            with torch.no_grad():
                probs = torch.softmax(trainer.policy(features)[0], dim=0)
            arm_idx = int(torch.multinomial(probs, 1, generator=gen))
            arm = ARM_SETS["ts"][arm_idx]
            # Ground truth: positive first feature -> student wins, else teacher.
            winner = "student" if float(features[0, 0]) > 0 else "teacher"
            episodes.append(TrainEpisode(
                task_uid=f"b{batch}e{i}", attempt=1, success=(arm == winner),
                features=features, arm_sampled=[arm], arm_executed=[arm],
            ))
        trainer.train_batch(episodes, batch_id=f"b{batch}", verify=False)

    # Grade the learned policy on fresh contexts.
    correct = 0
    for _ in range(200):
        features = torch.randn(1, D, generator=gen).to(FEATURE_DTYPE)
        with torch.no_grad():
            arm = ARM_SETS["ts"][int(trainer.policy(features)[0].argmax())]
        correct += arm == ("student" if float(features[0, 0]) > 0 else "teacher")
    assert correct / 200 > 0.7


# ---------------------------------------------------------------------------
# Warm-start graft
# ---------------------------------------------------------------------------


def test_graft_copies_the_trunk_and_zeroes_the_other_arms() -> None:
    torch.manual_seed(0)
    head = SuccessHead(D, H)
    params = graft(head, arms="tsc", delta0=0.75)
    arms = ARM_SETS["tsc"]
    assert torch.equal(params["W1"], head.l1.weight.data)
    assert torch.equal(params["b1"], head.l1.bias.data)
    student = arms.index("student")
    assert torch.equal(params["W2"][student], head.l2.weight.data[0])
    assert params["b2"][student] == pytest.approx(float(head.l2.bias.data[0]) - 0.75)
    for idx, arm in enumerate(arms):
        if arm == "student":
            continue
        assert torch.count_nonzero(params["W2"][idx]) == 0
        assert params["b2"][idx] == 0.0


@pytest.mark.parametrize("arms", ["ts", "tsc"])
def test_initial_student_rate_is_the_mean_not_the_median(arms: str) -> None:
    """The frozen contract is a 50% initial student RATE.

    On an asymmetric logit distribution the median sample and the mean rate are
    different numbers, and the earlier closed-form (median = 0.5) left the
    realized rate elsewhere. delta0 is therefore solved for the mean.
    """
    torch.manual_seed(0)
    head = SuccessHead(D, H)
    features = torch.randn(256, D)
    # Deliberately skewed out-of-fold logits: symmetric data would hide the bug.
    oof = torch.cat([torch.randn(200) * 0.3 - 1.0, torch.randn(56) * 0.3 + 4.0])
    delta0 = initial_student_bias(oof, target_rate=0.5, arms=arms)
    params = graft(head, arms=arms, delta0=delta0)

    n_other = len(ARM_SETS[arms]) - 1
    realized = torch.sigmoid(oof.double() - delta0 - math.log(n_other)).mean()
    assert float(realized) == pytest.approx(0.5, abs=1e-6)

    student = ARM_SETS[arms].index("student")
    others = [i for i in range(len(ARM_SETS[arms])) if i != student]
    if len(others) > 1:
        with torch.no_grad():
            hidden = torch.relu(features @ params["W1"].T + params["b1"])
            probs = torch.softmax(hidden @ params["W2"].T + params["b2"], dim=1)
        # teacher and cache both start from a zero row: identical, by construction.
        assert torch.allclose(probs[:, others[0]], probs[:, others[1]])


def test_the_shipped_head_is_the_one_calibrated_on() -> None:
    """The returned head is trained WITHOUT the calibration fold, and the
    returned logits are that same head's predictions on that fold.

    Cross-validated out-of-fold logits come from five fold-specific models, none
    of which ships; calibrating on them makes the report say 50% while the
    deployed router starts elsewhere.
    """
    torch.manual_seed(0)
    features = torch.randn(40, D)
    labels = (features[:, 0] > 0).float()
    episode = torch.arange(40) // 4
    folds = grouped_folds(episode, torch.zeros_like(episode), n_folds=5)
    head, selection, calibration = fit_head(features, labels, hidden=H, folds=folds, epochs=3)

    held = folds[selection["calibration_fold"]]
    assert calibration.shape == (int(held.sum()),)
    with torch.no_grad():
        assert torch.allclose(calibration, head(features[held]))   # same model
    assert selection["calibration_fold"] not in selection["final_head_trained_on_folds"]


def test_deployed_router_starts_at_the_declared_student_rate() -> None:
    """End of the chain: the GRAFTED parameters, on the held-out fold, must
    average 50% student. Asserting on an intermediate logit tensor would not
    have caught the graft being applied to a different head."""
    from exp.rl_router.fit_warmstart import deployed_student_rate

    torch.manual_seed(0)
    features = torch.randn(120, D)
    labels = (features[:, 0] + 0.5 * features[:, 1] > 0).float()
    episode = torch.arange(120) // 4
    folds = grouped_folds(episode, torch.zeros_like(episode), n_folds=5)
    head, selection, calibration = fit_head(features, labels, hidden=H, folds=folds, epochs=5)
    held = folds[selection["calibration_fold"]]

    for arms in ("ts", "tsc"):
        delta0 = initial_student_bias(calibration, arms=arms)
        params = graft(head, arms=arms, delta0=delta0)
        assert deployed_student_rate(params, features[held], arms=arms) == \
            pytest.approx(0.5, abs=1e-6)


def test_r_tc_has_no_student_rate_and_says_so() -> None:
    """R_tc must not be asked for a student rate it structurally cannot have.

    ``graft`` supports the no-student arm set on purpose (uniform start, §3.8),
    so the fit must not die one line later while reporting on it. Before the
    fix this surfaced as ``tuple.index(x): x not in tuple`` from inside
    ``deployed_student_rate``, i.e. the one variant graft() went out of its way
    to support was the one the fit could not produce.
    """
    from exp.rl_router.fit_warmstart import deployed_student_rate

    torch.manual_seed(0)
    features = torch.randn(20, D)
    labels = (features[:, 0] > 0).float()
    episode = torch.arange(20) // 4
    folds = grouped_folds(episode, torch.zeros_like(episode), n_folds=5)
    head, _, calibration = fit_head(features, labels, hidden=H, folds=folds, epochs=2)

    params = graft(head, arms="tc", delta0=initial_student_bias(calibration, arms="tc"))
    # every row zero -> uniform over {teacher, cache}, on the inherited trunk
    assert torch.count_nonzero(params["W2"]) == 0
    assert torch.count_nonzero(params["b2"]) == 0
    assert torch.count_nonzero(params["W1"]) > 0

    disclosure = graft_disclosure("tc")
    assert disclosure["initial_policy"] == "uniform"
    assert disclosure["student_row_from_head"] is False

    with pytest.raises(ValueError, match="no student arm"):
        deployed_student_rate(params, features, arms="tc")


def test_r_tc_grafts_the_trunk_and_starts_uniform() -> None:
    """R_tc has no student arm, so the same frozen rule leaves every output row
    at zero: the representation is inherited, the policy starts uniform. It must
    produce a usable checkpoint — R_tc is one of the three variants the
    experiment has to ship — with the limitation disclosed, not raise."""
    torch.manual_seed(0)
    head = SuccessHead(D, H)
    params = graft(head, arms="tc", delta0=0.0)
    assert torch.equal(params["W1"], head.l1.weight.data)   # trunk inherited
    assert torch.count_nonzero(params["W2"]) == 0           # no informed row
    assert torch.count_nonzero(params["b2"]) == 0

    features = torch.randn(32, D)
    hidden = torch.relu(features @ params["W1"].T + params["b1"])
    probs = torch.softmax(hidden @ params["W2"].T + params["b2"], dim=1)
    assert torch.allclose(probs, torch.full_like(probs, 0.5))

    disclosure = graft_disclosure("tc")
    assert disclosure["student_row_from_head"] is False
    assert disclosure["initial_policy"] == "uniform"
    assert graft_disclosure("tsc")["student_row_from_head"] is True


def test_zero_init_is_a_uniform_policy() -> None:
    params = zero_init(dim=D, hidden=H, arms="tsc")
    logits = torch.relu(torch.randn(5, D) @ params["W1"].T + params["b1"]) @ params["W2"].T
    probs = torch.softmax(logits + params["b2"], dim=1)
    assert torch.allclose(probs, torch.full_like(probs, 1 / 3))


def test_robot_state_stats_guard_against_a_frozen_joint() -> None:
    """A joint that never moved has zero variance; dividing by it would poison
    every subsequent feature with inf/NaN. The whole-vector affine also has to
    bring ||x|| to O(1): leaving it at sqrt(D) is what let one Adam step at the
    frozen trainer lr saturate the policy to a single arm."""
    from openpi.cache.components.mlp_router_judge import ROBOT_STATE

    features = torch.zeros(10, 8)
    features[:, 0] = torch.arange(10, dtype=torch.float32)   # only this one varies
    features[:, 4:] = torch.randn(10, 4) * 50.0              # a wildly unscaled field
    slices = {"vision_0": slice(0, 4), ROBOT_STATE: slice(4, 8)}
    mu, sigma = fit_feature_stats(features, field_slices=slices)

    assert mu.numel() == 8 and sigma.numel() == 8       # spans the WHOLE vector
    assert torch.all(sigma > 0)
    normalized = (features - mu) / sigma
    assert torch.all(torch.isfinite(normalized))
    # sqrt(D) is folded into sigma, so the per-row L1 lands near O(sqrt(D)),
    # not near D -- that ratio is exactly what the Adam step size multiplies.
    assert float(normalized.abs().sum(dim=1).mean()) < float(
        ((features - mu) * 0 + features).abs().sum(dim=1).mean()
    )
    assert float(normalized.abs().mean()) < 1.0


def test_folds_never_split_one_episode() -> None:
    """Steps inside an episode are near-duplicates: a step-level split would
    grade the head on data it memorised."""
    episode = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9])
    task = torch.zeros_like(episode)
    folds = grouped_folds(episode, task, n_folds=5)
    assert sum(int(f.sum()) for f in folds) == episode.numel()
    for fold in folds:
        for ep_id in torch.unique(episode[fold]):
            assert bool(fold[episode == ep_id].all()), "episode split across folds"


def test_folds_are_stratified_by_task_and_deterministic() -> None:
    episode = torch.arange(20)
    task = torch.tensor([i % 2 for i in range(20)])
    first = grouped_folds(episode, task, n_folds=5, seed=0)
    again = grouped_folds(episode, task, n_folds=5, seed=0)
    assert all(torch.equal(a, b) for a, b in zip(first, again))
    for fold in first:
        tasks_in_fold = task[fold].tolist()
        assert math.isclose(sum(tasks_in_fold), len(tasks_in_fold) / 2, abs_tol=1.0)


# ---------------------------------------------------------------------------
# Pre-registration guards and batch composition (§3.9 / §3.10)
# ---------------------------------------------------------------------------


def test_eval_yaml_guard_catches_a_sampling_eval_config(tmp_path) -> None:
    """The one-shot A-pool measurement must not be able to sample or dump —
    a copy-paste slip would turn it into another training run."""
    from exp.rl_router.emit_router_yamls import check_eval_guards, emit

    emit(tmp_path, suite="libero_10", weights_path=str(tmp_path / "w.pt"),
         student_endpoint="127.0.0.1:7002", dump_dir="", temperature=1.0,
         seed=0, hidden=8, variants=["R_ts"])
    assert check_eval_guards(tmp_path) == []

    bad = yaml.safe_load((tmp_path / "r_ts_eval.yaml").read_text())
    bad["checkpoints"]["cp1"]["judge"].update(
        {"mode": "sample", "temperature": 1.0, "seed": 3, "dump_dir": "/tmp/x"}
    )
    (tmp_path / "r_ts_eval.yaml").write_text(yaml.safe_dump(bad, sort_keys=False))
    problems = check_eval_guards(tmp_path)
    assert any("mode=" in p for p in problems)
    assert any("dump_dir" in p for p in problems)
    assert any("sampling seed" in p for p in problems)


def test_init_pool_exclusion_is_bidirectional(tmp_path) -> None:
    """Training may only touch B-train; the A pool is measured once. One shared
    or nested directory would leak the evaluation set into training."""
    from exp.rl_router.emit_router_yamls import check_init_pools

    a, b = tmp_path / "a_pool", tmp_path / "b_train"
    a.mkdir()
    b.mkdir()
    assert check_init_pools(str(b), str(a)) == []
    assert check_init_pools(str(a), str(a))          # identical
    nested = b / "inner"
    nested.mkdir()
    assert check_init_pools(str(nested), str(b))     # train inside eval
    assert check_init_pools(str(b), str(nested))     # eval inside train


def test_only_seed_zero_enters_the_primary_family() -> None:
    from exp.rl_router.emit_router_yamls import holm_family, seed_roles

    runs = [
        {"suite": "libero_10", "variant": "R_ts", "seed": 0, "flagship": True},
        {"suite": "libero_10", "variant": "R_ts", "seed": 1, "flagship": True},
        {"suite": "libero_10", "variant": "R_tsc", "seed": 0},
    ]
    roles = seed_roles(runs)
    assert [r["seed"] for r in roles["primary"]] == [0, 0]
    assert [r["seed"] for r in roles["robustness"]] == [1]

    family = holm_family(runs)
    assert {(f["variant"], f["comparator"]) for f in family} == {
        ("R_ts", "tier_two_tier"), ("R_tsc", "tier_three_tier"),
    }
    assert all(f["seed"] == 0 for f in family)


def test_pooling_two_seeds_into_one_comparison_is_rejected() -> None:
    from exp.rl_router.emit_router_yamls import reject_seed_pooling

    reject_seed_pooling([{"seed": 0}, {"seed": 0}])
    with pytest.raises(ValueError, match="multiple training seeds"):
        reject_seed_pooling([{"seed": 0}, {"seed": 1}])


def test_batch_sampling_is_deterministic_and_without_replacement() -> None:
    """The interaction-efficiency x-axis counts episodes, so a resume that
    re-drew different inits would silently redefine it."""
    from exp.rl_router.run_rl_router import sample_batch

    pairs = [(t, i) for t in range(10) for i in range(45)]  # B-train shape
    b0 = sample_batch(pairs, batch_size=100, batch_idx=0, seed=0)
    assert b0 == sample_batch(pairs, batch_size=100, batch_idx=0, seed=0)
    assert len(set(b0)) == 100
    assert b0 != sample_batch(pairs, batch_size=100, batch_idx=1, seed=0)
    assert b0 != sample_batch(pairs, batch_size=100, batch_idx=0, seed=1)

    # One epoch covers the pool exactly once before anything repeats.
    epoch = [p for b in range(4) for p in sample_batch(pairs, batch_size=100,
                                                       batch_idx=b, seed=0)]
    assert len(set(epoch)) == 400


def test_batch_sampling_wraps_across_epoch_boundaries() -> None:
    from exp.rl_router.run_rl_router import sample_batch

    pairs = [(0, i) for i in range(10)]
    spanning = sample_batch(pairs, batch_size=4, batch_idx=2, seed=0)  # covers 8..11
    assert len(spanning) == 4


def test_batch_larger_than_the_pool_is_refused() -> None:
    from exp.rl_router.run_rl_router import sample_batch

    with pytest.raises(ValueError, match="exceeds the B-train pool"):
        sample_batch([(0, 0)], batch_size=2, batch_idx=0, seed=0)


def test_pilot_split_holds_out_what_it_measures_on() -> None:
    from exp.rl_router.pilot_lambda import pilot_split

    doc = {f"task_{t}": {"train": list(range(45))} for t in range(10)}
    split = pilot_split(doc, inits_per_task=30, seed=0)
    for key in doc:
        pilot, remainder = set(split["pilot"][key]), set(split["remainder"][key])
        assert len(pilot) == 30
        assert not (pilot & remainder)              # measured on unseen inits
        assert pilot | remainder == set(range(45))
    assert split == pilot_split(doc, inits_per_task=30, seed=0)


def test_lambda_selection_picks_the_closest_realized_rates() -> None:
    from exp.rl_router.pilot_lambda import select_lambdas

    result = select_lambdas({0.05: 0.62, 0.2: 0.41, 0.5: 0.18})
    assert result["separated"] is True
    assert result["selected"] == {"lambda_1": 0.2, "lambda_2": 0.5}


def test_lambda_selection_flags_an_unseparated_grid() -> None:
    """All three candidates in one regime: the protocol inserts one
    supplementary candidate rather than picking arbitrarily."""
    from exp.rl_router.pilot_lambda import select_lambdas

    result = select_lambdas({0.05: 0.9, 0.2: 0.88, 0.5: 0.87})
    assert result["separated"] is False
    assert 0.05 < result["supplementary_candidate"] < 0.5


def test_interaction_budget_charges_shared_costs_to_every_variant() -> None:
    """The headline curve answers "what did this router cost to obtain", so the
    warm-start pass and the pilot cannot start it from a free lunch."""
    from exp.rl_router.pilot_lambda import interaction_budget

    budget = interaction_budget(warmstart_episodes=450)
    assert budget["pilot_episodes"] == 3 * (5 * 100 + 100)
    assert budget["shared_offset"] == 450 + budget["pilot_episodes"]


def test_collect_yaml_is_derived_from_the_arm_yaml(tmp_path) -> None:
    """The collection pass must observe the same feature space the RL run will,
    so it is derived from the arm config rather than written independently."""
    from exp.rl_router.collect_warmstart import build_collect_yaml
    from exp.rl_router.emit_router_yamls import emit

    emit(tmp_path, suite="libero_10", weights_path=str(tmp_path / "w.pt"),
         student_endpoint="127.0.0.1:7002", dump_dir="/tmp/d", temperature=1.0,
         seed=0, hidden=8, variants=["R_ts"])
    arm = yaml.safe_load((tmp_path / "r_ts_train.yaml").read_text())
    collect = build_collect_yaml(tmp_path / "r_ts_train.yaml", arm="student",
                                 dump_dir="/tmp/collect")
    judge = collect["checkpoints"]["cp1"]["judge"]
    assert judge["constant_arm"] == "student" and judge["mode"] == "argmax"
    assert "weights_path" not in judge and "seed" not in judge
    assert judge["dump_dir"] == "/tmp/collect"
    # Same observation space and same retrieval as the RL run.
    assert judge["feature_fields"] == arm["checkpoints"]["cp1"]["judge"]["feature_fields"]
    assert collect["key_builder"] == arm["key_builder"]
    assert collect["backend"] == arm["backend"]


# ---------------------------------------------------------------------------
# Warm-start trunk health (the failure that cost three pilot batches)
# ---------------------------------------------------------------------------


def test_dead_trunk_is_refused_not_shipped() -> None:
    """A head whose ReLUs are all dead makes the grafted router a CONSTANT.

    It ignores the observation, argmax collapses onto one arm, and
    ``relu'(pre) == 0`` means REINFORCE's gradient into the trunk is identically
    zero — the RL run can never learn, however long it runs. Every surface
    metric still looks plausible (the checkpoint loads, batches join, rates read
    ~50/50), so this has to fail loudly at fit time.
    """
    import torch

    from exp.rl_router.fit_warmstart import SuccessHead, assert_trunk_alive, trunk_health

    features = torch.randn(64, 32)
    head = SuccessHead(32, 8)
    with torch.no_grad():                      # every unit pushed past its hinge
        head.l1.weight.zero_()
        head.l1.bias.fill_(-1.0)
    health = trunk_health(head, features)
    assert health["live_units"] == 0
    # Float noise, not a clean zero — which is why the guard uses a tolerance.
    assert health["output_std"] < 1e-6
    with pytest.raises(SystemExit, match="degenerate warm-start head"):
        assert_trunk_alive(health)


def test_healthy_trunk_passes_and_reports_its_margin() -> None:
    import torch

    from exp.rl_router.fit_warmstart import SuccessHead, assert_trunk_alive, trunk_health

    torch.manual_seed(0)
    features = torch.randn(64, 32)
    health = trunk_health(SuccessHead(32, 8), features)
    assert 0.0 < health["live_unit_fraction"] < 1.0
    assert health["output_std"] > 0.0
    assert_trunk_alive(health)                 # must not raise




def test_head_learning_rate_follows_the_feature_scale() -> None:
    """Pinning a learning rate instead of the per-step displacement is what
    produced two degenerate warm starts: 1e-3 killed the trunk on raw features
    (sum|x|~2.75e5) and 1e-5 left the head at initialisation on scaled ones
    (sum|x|~89). Deriving it makes both impossible at any width or scaling."""
    import torch

    from exp.rl_router.fit_warmstart import HEAD_STEP_TARGET, head_learning_rate

    for scale in (2.75e5, 89.0, 1.0):
        x = torch.full((4, 8), scale / 8.0)
        lr = head_learning_rate(x)
        assert abs(lr * scale - HEAD_STEP_TARGET) < 1e-6 * HEAD_STEP_TARGET
    with pytest.raises(SystemExit, match="all zero"):
        head_learning_rate(torch.zeros(4, 8))


def test_a_head_no_better_than_the_base_rate_is_refused() -> None:
    """Both degenerate warm starts were visible here first: CV loss 0.911 and
    0.687 against a 0.518 bar. Grafting such a head spends the interaction
    budget the headline curve is measured in and dresses a coin flip up as an
    initialisation."""
    import torch

    from exp.rl_router.fit_warmstart import assert_head_beats_base_rate, base_rate_entropy

    labels = torch.cat([torch.ones(787), torch.zeros(213)])       # 78.7% success
    bar = base_rate_entropy(labels)
    assert abs(bar - 0.5184) < 1e-3

    with pytest.raises(SystemExit, match="carries no signal"):
        assert_head_beats_base_rate(
            {"cv_loss": 0.911, "base_rate_entropy": bar, "head_lr": 1e-3})
    with pytest.raises(SystemExit, match="carries no signal"):
        assert_head_beats_base_rate(
            {"cv_loss": 0.687, "base_rate_entropy": bar, "head_lr": 1e-5})
    assert_head_beats_base_rate(
        {"cv_loss": 0.456, "base_rate_entropy": bar, "head_lr": 1e-3})   # must not raise
