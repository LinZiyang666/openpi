"""Warm-start the router from a success-prediction head (§3.8).

"Warm-start" here means **RL weight initialisation**, not the cache's
denoising warm-start tier — the TIER experiment configs replay clean actions
only and never use WARM_START.

Pipeline::

    constant-arm collection (student arm, B-train)   -> v0-raw features + outcomes
      -> fit a whole-vector mu/sigma from those features -> encoder v2
      -> 2-layer BCE head, SAME architecture (D -> hidden -> 1)
      -> graft the head into the router's first layer

The head is the same shape as the router on purpose. A logistic regression
would have to be *mapped* into a 2-layer MLP, and there is no exact mapping —
the grafted initialisation would differ from the thing that was validated. With
an identical trunk the graft is a copy.

Initialisation mapping (frozen)::

    router.W1, router.b1  <-  head.W1, head.b1          (whole layer)
    router.W2[student]    <-  head.W2                   (the success logit row)
    router.W2[others]     <-  0
    router.b2[student]    <-  head.b2 - delta0          (delta0 sets the initial
    router.b2[others]     <-  0                          student rate to 50%)

Model selection is grouped 5-fold: folds split by *episode*, stratified by task.
Steps within one episode are near-duplicates, so a step-level split would leak
across folds and report a head far better than it is.

Fallback (predeclared): if more than 10% of the collected episodes failed for
*infrastructure* reasons (error, missing identity, incomplete shard), the suite
falls back to zero initialisation and that is disclosed. A student rollout that
simply did not solve the task is a normal negative label, not a failure.

Usage::

    uv run exp/rl_router/fit_warmstart.py \
        --shards <dump_dir>/<run_id>/<batch_id> --journal <journal.jsonl> \
        --arms tsc --out exp/rl_router/data/warmstart_l10.pt
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Optional

import numpy as np
import torch

from openpi.cache.components.mlp_router_judge import (
    ARM_SETS,
    FEATURE_DTYPE,
    ROBOT_STATE,
    RouterFeatureEncoder,
    save_router_weights,
)

from exp.rl_router.batch_package import load_shard_manifest, read_jsonl

# Predeclared infrastructure-failure ceiling (§3.8). Above this the suite drops
# to a zero initialisation rather than warm-starting from a biased sample.
INFRA_FAILURE_CEILING = 0.10
SEED = 0
N_FOLDS = 5

# How far one Adam step may move a pre-activation. Adam displaces every
# parameter by ~lr per step regardless of gradient magnitude, so a step moves
# each pre-activation by about ``lr * sum_j |x_j|`` — a quantity that scales
# with BOTH the input width and the feature scale. Pinning a learning rate
# instead of pinning this product is what produced two degenerate warm starts in
# a row on the same trunk:
#
#   raw features (sum|x| ~ 2.75e5), lr 1e-3  -> 275 per step: every ReLU driven
#       past its hinge, 0/256 live units, a router that ignores its observation
#       and whose trunk gradient is identically zero;
#   v2 features  (sum|x| ~ 8.9e1),  lr 1e-5  -> 9e-4 per step: 30 steps move the
#       head by 0.03 total, so it never leaves initialisation and its CV loss
#       sits at ln 2.
#
# Deriving the rate from the measured scale makes both impossible, at any
# feature width or scaling. 0.1 is inside the band both sweeps found healthy.
HEAD_STEP_TARGET = 0.1
HEAD_EPOCHS = 30


def head_learning_rate(features: torch.Tensor, *, target: float = HEAD_STEP_TARGET) -> float:
    """Adam lr that moves a pre-activation by ~``target`` per step on ``features``."""
    scale = float(features.abs().sum(dim=1).mean())
    if scale <= 0:
        raise SystemExit("features are all zero; the collection produced nothing to fit on")
    return target / scale


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_collection(
    shard_dir: str | pathlib.Path,
    journal_path: str | pathlib.Path,
    *,
    expected_slots: list[str],
) -> dict:
    """Load v0-raw features + per-episode outcomes from a collection run.

    Admission mirrors the RL batch packager exactly — accepted, no error, shard
    complete, and only one attempt per slot — because the same failure modes
    apply: a stale attempt is journaled like the live one, and an episode whose
    shard never finalized has a truncated trajectory.

    The infrastructure-failure rate is computed against the **full expected
    set**, not against the shards that happen to exist. An episode that was
    dispatched and vanished (worker died before any shard was written) leaves no
    manifest entry at all; scoring only what exists would divide the failures by
    the survivors and report a healthy collection precisely when it went worst.

    Returns ``{"features": [S, D] fp32, "labels": [S], "episode": [S],
    "task": [S], "uids": [...], "infra_failure_rate": float}`` where every step
    of an episode carries that episode's outcome.
    """
    shard_dir = pathlib.Path(shard_dir)
    expected = list(dict.fromkeys(expected_slots))
    # Keyed by (uid, attempt), not by uid. A requeued episode produces two
    # journal rows and two shards for one uid; collapsing on uid alone lets the
    # ACCEPTED attempt's success label be pinned onto the STALE attempt's
    # features — a mislabelled training set that nothing downstream can detect.
    journal_by_key: dict[tuple[str, int], dict] = {}
    accepted_attempt: dict[str, int] = {}
    for row in read_jsonl(journal_path):
        uid, attempt = str(row.get("task_uid")), int(row.get("attempt", -1))
        journal_by_key[(uid, attempt)] = row
        if row.get("accepted") is True and row.get("error") is None:
            accepted_attempt[uid] = attempt
    shard_by_key = {
        (str(e.get("task_uid")), int(e.get("attempt", -1))): e
        for e in load_shard_manifest(shard_dir)
    }

    usable, failures = [], []
    for uid in expected:
        attempt = accepted_attempt.get(uid)
        if attempt is None:
            # Either no journal row at all (dispatched and vanished) or every
            # attempt was rejected / errored. Both are infrastructure failures.
            failures.append((uid, "no_accepted_attempt"))
            continue
        entry = journal_by_key[(uid, attempt)]
        shard = shard_by_key.get((uid, attempt))
        if shard is None:
            failures.append((uid, "shard_missing"))
        elif shard.get("status") != "complete" or int(shard.get("rows", 0)) <= 0:
            failures.append((uid, f"shard_{shard.get('status')}"))
        else:
            usable.append((shard, entry))
    total = len(expected) or 1

    features, labels, episodes, tasks, uids = [], [], [], [], []
    for idx, (entry, journal_row) in enumerate(usable):
        raw = (shard_dir / entry["shard"]).read_bytes()
        mat = torch.frombuffer(bytearray(raw), dtype=FEATURE_DTYPE).reshape(
            int(entry["rows"]), -1
        ).to(torch.float32)
        features.append(mat)
        uid = str(entry["task_uid"])
        uids.append(uid)
        # The label comes from the journal row of the SAME attempt these
        # features were dumped under.
        labels.append(torch.full((mat.shape[0],), float(journal_row.get("success"))))
        episodes.append(torch.full((mat.shape[0],), idx, dtype=torch.long))
        tasks.append(torch.full((mat.shape[0],), _task_of(uid), dtype=torch.long))

    return {
        "features": torch.cat(features) if features else torch.zeros(0, 0),
        "labels": torch.cat(labels) if labels else torch.zeros(0),
        "episode": torch.cat(episodes) if episodes else torch.zeros(0, dtype=torch.long),
        "task": torch.cat(tasks) if tasks else torch.zeros(0, dtype=torch.long),
        "uids": uids,
        "failures": [{"task_uid": u, "reason": r} for u, r in failures],
        "infra_failure_rate": len(failures) / total,
        "n_expected": len(expected),
        "n_episodes": len(usable),
    }


def _task_of(task_uid: str) -> int:
    """``<yaml>:<phase>:<task_id>:<episode_idx>`` -> task_id (for stratification)."""
    parts = task_uid.rsplit(":", 3)
    try:
        return int(parts[1]) if len(parts) == 3 else int(parts[-2])
    except (ValueError, IndexError):
        return 0


def fit_feature_stats(
    features: torch.Tensor, *, field_slices: dict[str, slice]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Whole-vector affine: standardize each field, then bring ``||x||`` to O(1).

    Returned as one ``(mu, sigma)`` pair spanning the concatenated vector, which
    is exactly what ``RouterFeatureEncoder`` applies — one operator, one version
    hash, nothing for the online and offline paths to disagree about.

    Two decisions, both forced by measurement rather than taste:

      - **robot_state is per-dimension, the vision fields are per-field
        scalars.** robot_state mixes joint angles with a gripper flag, so its
        dimensions genuinely differ in scale. A pooled vision embedding does
        not: standardizing its 32,768 dimensions independently would amplify
        whichever ones happen to be near-constant in this collection into pure
        noise at serving time.
      - **1/sqrt(D) is folded into sigma.** Per-dimension standardization alone
        still leaves ``||x|| ~ sqrt(D) = 256``, and with Adam displacing every
        weight by ~lr per step that puts ~82 on each pre-activation per update
        at the frozen trainer lr — measured to saturate the policy to one arm in
        a single step. Dividing by sqrt(D) is what makes the frozen §3.5
        constants train instead of detonate.
    """
    dim = int(features.shape[1])
    mu = torch.zeros(dim)
    sigma = torch.ones(dim)
    for name, sl in field_slices.items():
        block = features[:, sl]
        if name == ROBOT_STATE:
            m = block.mean(dim=0)
            s = block.std(dim=0, unbiased=False)
        else:
            m = block.mean().expand(block.shape[1]).clone()
            s = block.std(unbiased=False).expand(block.shape[1]).clone()
        # A dimension (or a field) that never varied would divide to inf/NaN.
        mu[sl] = m
        sigma[sl] = torch.where(s > 1e-6, s, torch.ones_like(s))
    return mu, sigma * (dim ** 0.5)


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------


class SuccessHead(torch.nn.Module):
    """``D -> hidden -> 1`` BCE head; the router's trunk, one output."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(dim, hidden)
        self.l2 = torch.nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(torch.relu(self.l1(x))).squeeze(-1)


def grouped_folds(episode: torch.Tensor, task: torch.Tensor, *, n_folds: int = N_FOLDS,
                  seed: int = SEED) -> list[torch.Tensor]:
    """Episode-grouped, task-stratified fold assignment (per-step mask list).

    Grouping is the point: two steps from one episode are near-duplicates, so a
    step-level split would let the model memorise an episode in training and be
    graded on the same episode in validation.
    """
    rng = np.random.RandomState(seed)
    ep_ids = torch.unique(episode)
    ep_task = {int(e): int(task[episode == e][0]) for e in ep_ids}
    assignment: dict[int, int] = {}
    for task_id in sorted(set(ep_task.values())):
        members = [e for e in sorted(ep_task) if ep_task[e] == task_id]
        rng.shuffle(members)
        for pos, e in enumerate(members):
            assignment[e] = pos % n_folds
    fold_of_step = torch.tensor([assignment[int(e)] for e in episode])
    return [fold_of_step == k for k in range(n_folds)]


def fit_head(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    hidden: int,
    weight_decays: tuple[float, ...] = (0.0, 1e-4, 1e-3),
    epochs: int = HEAD_EPOCHS,
    lr: Optional[float] = None,
    folds: Optional[list[torch.Tensor]] = None,
    seed: int = SEED,
    calibration_fold: int = 0,
) -> tuple[SuccessHead, dict, torch.Tensor]:
    """Select weight decay by grouped CV, fit the FINAL head, return its
    held-out calibration logits.

    The head that is returned — and therefore grafted and shipped — is trained
    on every fold EXCEPT ``calibration_fold``, and the third return value is
    that same head's predictions on the fold it never saw.

    This is the difference that matters: cross-validated out-of-fold logits come
    from five different fold-specific models, none of which is the model that
    ships. Calibrating δ₀ on them makes the *report* say 50% while the deployed
    router starts somewhere else entirely. Holding one fold out of the final fit
    costs a little data and buys the guarantee that the number describes the
    thing actually being deployed.
    """
    torch.manual_seed(seed)
    dim = int(features.shape[1])
    # Derived from the features actually being fit, so the rate follows any
    # change of width or scaling instead of silently mis-matching it.
    if lr is None:
        lr = head_learning_rate(features)
    scores: dict[float, float] = {}
    oof_by_decay: dict[float, torch.Tensor] = {}
    for decay in weight_decays:
        losses = []
        oof = torch.zeros(features.shape[0])
        for fold in (folds or []):
            head = SuccessHead(dim, hidden)
            _train_head(head, features[~fold], labels[~fold], decay=decay, epochs=epochs, lr=lr)
            with torch.no_grad():
                held = head(features[fold])
                oof[fold] = held
                losses.append(float(torch.nn.functional.binary_cross_entropy_with_logits(
                    held, labels[fold]
                )))
        scores[decay] = float(np.mean(losses)) if losses else float("nan")
        oof_by_decay[decay] = oof
    best = min(scores, key=lambda d: scores[d]) if folds else weight_decays[0]

    torch.manual_seed(seed)
    head = SuccessHead(dim, hidden)
    if folds:
        held = folds[calibration_fold]
        _train_head(head, features[~held], labels[~held], decay=best, epochs=epochs, lr=lr)
        with torch.no_grad():
            calibration_logits = head(features[held])
    else:
        _train_head(head, features, labels, decay=best, epochs=epochs, lr=lr)
        calibration_logits = torch.zeros(0)
    del oof_by_decay
    selection = {
        "head_lr": lr,
        "cv_loss_by_weight_decay": scores,
        "selected_weight_decay": best,
        "calibration_fold": calibration_fold,
        "final_head_trained_on_folds": [i for i in range(len(folds or []))
                                        if i != calibration_fold],
        "base_rate_entropy": base_rate_entropy(labels),
        "cv_loss": scores.get(best),
    }
    return head, selection, calibration_logits


def base_rate_entropy(labels: torch.Tensor) -> float:
    """BCE of the constant predictor — the bar any useful head must clear."""
    p = float(labels.mean())
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)))


def assert_head_beats_base_rate(selection: dict) -> None:
    """A head no better than "always predict the base rate" carries no signal.

    This is the check that would have caught both degenerate warm starts on
    sight, before any GPU time was spent on them: an over-large learning rate
    killed the trunk and scored 0.911 against a 0.518 bar, and an over-small one
    left the head at initialisation and scored 0.687 against the same bar. Both
    times every other surface — checkpoint loads, batches join, arm rates near
    the calibrated target — looked perfectly healthy.

    Grafting such a head is worse than not warm-starting at all: it spends the
    interaction budget the headline curve is measured in, and it dresses a coin
    flip up as an initialisation.
    """
    cv, bar = selection.get("cv_loss"), selection.get("base_rate_entropy")
    if cv is None or bar is None:
        return
    if cv >= bar:
        raise SystemExit(
            f"warm-start head carries no signal: held-out CV loss {cv:.4f} is not "
            f"below the base-rate entropy {bar:.4f} (predicting the success rate "
            "with a constant would do as well or better). Fit lr was "
            f"{selection.get('head_lr'):.3g}. Check the feature scaling and "
            "HEAD_STEP_TARGET before grafting this into a router."
        )


def _train_head(head: SuccessHead, x: torch.Tensor, y: torch.Tensor, *,
                decay: float, epochs: int, lr: float) -> None:
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=decay)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(head(x), y)
        loss.backward()
        opt.step()


# ---------------------------------------------------------------------------
# Graft
# ---------------------------------------------------------------------------


def fold_manifest(episode: torch.Tensor, task: torch.Tensor, uids: list[str],
                  folds: list[torch.Tensor]) -> dict:
    """Tracked record of which episode landed in which fold.

    Model selection that cannot be re-derived is not auditable: without this,
    a later reader has to trust that the folds were grouped and stratified
    rather than being able to check it.
    """
    assignment: dict[str, int] = {}
    for k, mask in enumerate(folds):
        for ep_id in torch.unique(episode[mask]).tolist():
            if ep_id < len(uids):
                assignment[uids[ep_id]] = k
    return {
        "n_folds": len(folds),
        "seed": SEED,
        "grouped_by": "episode",
        "stratified_by": "task",
        "fold_of_episode": assignment,
        "episodes_per_fold": {
            str(k): int(torch.unique(episode[mask]).numel()) for k, mask in enumerate(folds)
        },
    }


def initial_student_bias(calibration_logits: torch.Tensor, *, target_rate: float = 0.5,
                         arms: str = "tsc", tol: float = 1e-9) -> float:
    """delta0 such that the **mean** initial student probability is ``target_rate``.

    Two things this is careful about:

      - it takes the logits **the shipped head** produced on the fold it never
        trained on, so the rate describes the deployed router rather than an
        intermediate cross-validation artefact;
      - it solves for the *mean* rate, not the median sample. With every other
        arm at logit 0 the student probability is
        ``sigmoid(z - delta0 - log(n_other))``, which is non-linear in ``z``:
        setting the median sample to 0.5 leaves the realized rate somewhere else
        entirely on an asymmetric logit distribution. The frozen contract is a
        50% initial student rate, so that is what is solved for, by bisection.
    """
    n_other = len(ARM_SETS[arms]) - 1
    if n_other <= 0:
        raise ValueError(f"arms={arms!r} has no non-student arm to trade against")
    z = calibration_logits.detach().to(torch.float64)
    offset = float(np.log(n_other))

    def mean_rate(delta: float) -> float:
        return float(torch.sigmoid(z - delta - offset).mean())

    lo, hi = -1e3, 1e3                      # mean_rate is monotonically decreasing in delta
    for _ in range(200):
        mid = (lo + hi) / 2
        if mean_rate(mid) > target_rate:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


def trunk_health(head: SuccessHead, features: torch.Tensor) -> dict:
    """Is the fitted head actually a function of its input?

    A trunk whose ReLUs are all dead makes the grafted router a constant: it
    ignores the observation, argmax collapses onto one arm, and — worst —
    ``relu'(pre) == 0`` means REINFORCE's gradient into ``W1``/``b1`` is
    identically zero, so the RL run can never learn no matter how long it runs.
    That failure is silent end to end: the checkpoint loads, the judge decides,
    the batches join, and every metric looks plausible while the policy is a
    fixed coin flip. It cost three pilot batches to notice, so it is checked
    here, on the same features the head was fit on.
    """
    with torch.no_grad():
        pre = features @ head.l1.weight.T + head.l1.bias
        out = head(features)
    return {
        "live_unit_fraction": float((pre > 0).float().mean()),
        "live_units": int((pre > 0).any(dim=0).sum()),
        "hidden_units": int(head.l1.out_features),
        "output_std": float(out.std()),
        "mean_pre_activation": float(pre.mean()),
    }


# A head whose output varies by less than this across the whole fit set is a
# constant for every practical purpose: softmax over such logits is uniform to
# well past fp16 resolution, so the deployed router still ignores its input.
# Compared against a tolerance rather than exact zero because a dead trunk
# leaves float noise, not a clean zero.
MIN_HEAD_OUTPUT_STD = 1e-6


def assert_trunk_alive(health: dict) -> None:
    """Fail loudly on a degenerate head rather than shipping a constant router."""
    if health["live_units"] == 0 or health["output_std"] < MIN_HEAD_OUTPUT_STD:
        raise SystemExit(
            "degenerate warm-start head: "
            f"{health['live_units']}/{health['hidden_units']} live hidden units, "
            f"output std {health['output_std']:.3e}, mean pre-activation "
            f"{health['mean_pre_activation']:.3e}. The grafted router would ignore its "
            "observation and REINFORCE could never move the trunk (relu'(pre)==0). "
            "This is what an over-large head learning rate does at this input width — "
            "This is what a mis-scaled head learning rate does at this input "
            f"width — see HEAD_STEP_TARGET ({HEAD_STEP_TARGET:g}) and the "
            "reasoning beside it."
        )


def graft(head: SuccessHead, *, arms: str, delta0: float) -> dict[str, torch.Tensor]:
    """Copy the head's trunk into a router parameter set (frozen mapping).

    The mapping is one rule applied to whatever arm set is configured: the whole
    first layer is copied; the **student** output row takes the head's success
    logit (shifted by ``delta0``); every other row starts at zero.

    R_tc has no student arm, so every row is zero and the initial policy is
    uniform over {teacher, cache} on top of the inherited representation. That
    is not a special case bolted on — it is the same rule, and it is the
    pre-registered "no informative head for the cache arm" position (§3.8). It
    returns a usable initialisation rather than raising, because R_tc is one of
    the three variants the experiment must produce.
    """
    arm_names = ARM_SETS[arms]
    hidden = head.l1.out_features
    W2 = torch.zeros(len(arm_names), hidden)
    b2 = torch.zeros(len(arm_names))
    if "student" in arm_names:
        student = arm_names.index("student")
        W2[student] = head.l2.weight.data[0].clone()
        b2[student] = float(head.l2.bias.data[0]) - delta0
    return {
        "W1": head.l1.weight.data.clone(),
        "b1": head.l1.bias.data.clone(),
        "W2": W2,
        "b2": b2,
    }


def graft_disclosure(arms: str) -> dict:
    """What the pre-registration must say about this variant's initialisation."""
    has_student = "student" in ARM_SETS[arms]
    return {
        "arms": arms,
        "student_row_from_head": has_student,
        "initial_policy": "student-biased" if has_student else "uniform",
        "note": (
            "the success head predicts the student arm's outcome, so only that "
            "row carries information; every other arm starts at zero"
            if has_student else
            "no student arm: the head informs no output row, so the trunk is "
            "inherited and the policy starts uniform (pre-registered §3.8 limit "
            "on the steelman narrative for the cache arm)"
        ),
    }


def deployed_student_rate(params: dict[str, torch.Tensor], features: torch.Tensor,
                          *, arms: str) -> float:
    """Mean student probability of the GRAFTED router on ``features``.

    Measured through the actual grafted parameters rather than an intermediate
    logit tensor: the contract is about what the deployed policy does on its
    first step, and only this path includes the graft itself.

    Undefined for an arm set without a student arm (R_tc), where ``graft``
    deliberately returns a uniform policy. Callers must branch on the arm set —
    see ``graft_disclosure`` — rather than ask this for a rate that does not
    exist; the explicit raise is here so that mistake is legible instead of
    surfacing as ``tuple.index(x): x not in tuple``.
    """
    if "student" not in ARM_SETS[arms]:
        raise ValueError(
            f"arms={arms!r} has no student arm, so there is no deployed student "
            "rate; graft() starts this variant uniform (see graft_disclosure)"
        )
    with torch.no_grad():
        hidden = torch.relu(features.to(torch.float32) @ params["W1"].T + params["b1"])
        probs = torch.softmax(hidden @ params["W2"].T + params["b2"], dim=1)
    return float(probs[:, ARM_SETS[arms].index("student")].mean())


def zero_init(*, dim: int, hidden: int, arms: str) -> dict[str, torch.Tensor]:
    """Uniform-policy fallback used when the collection failed too often."""
    n_arms = len(ARM_SETS[arms])
    return {
        "W1": torch.zeros(hidden, dim), "b1": torch.zeros(hidden),
        "W2": torch.zeros(n_arms, hidden), "b2": torch.zeros(n_arms),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def dims_from_arm_yaml(arm_yaml: str | pathlib.Path, fields: tuple[str, ...]) -> dict[str, int]:
    """Read the per-field widths from the arm yaml's ``backend.vector_dims``.

    The authoritative source, and the same one the serving judge validates
    against. Deriving them by splitting a total width evenly is a guess that
    happens to be right only for equal-sized vision fields, and it would produce
    a checkpoint whose meta silently disagrees with the artifact.
    """
    import yaml as _yaml

    cfg = _yaml.safe_load(pathlib.Path(arm_yaml).read_text(encoding="utf-8"))
    declared = (cfg.get("backend") or {}).get("vector_dims") or {}
    missing = [f for f in fields if f not in declared]
    if missing:
        raise SystemExit(
            f"{arm_yaml}: backend.vector_dims has no entry for {missing}; the "
            "warm-start checkpoint's meta must come from the artifact, not a guess"
        )
    encoder = RouterFeatureEncoder(fields)
    return {f: int(declared[f]) for f in encoder.fields}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", required=True, help="<dump_dir>/<run_id>/<batch_id>")
    parser.add_argument("--journal", required=True)
    parser.add_argument("--expected-slots", required=True,
                        help="collect_warmstart.py's expected_slots.json (the dispatched set)")
    parser.add_argument("--arm-yaml", required=True,
                        help="the arm yaml whose backend.vector_dims define the feature widths")
    parser.add_argument("--arms", required=True, choices=sorted(ARM_SETS))
    parser.add_argument("--fields", default="vision_0,vision_1,robot_state")
    parser.add_argument("--robot-state-dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument(
        "--initial-student-rate", type=float, default=0.5,
        help=(
            "mean student probability the grafted router starts at. The default 0.5 is "
            "what M5b shipped; the SR(p) sweep showed it lands on the flat part of the "
            "success curve, where no state-dependent signal exists and only a global "
            "bias can move -- set this at the measured knee instead."))
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--folds-out", required=True,
                        help="tracked fold assignment manifest (required: model selection "
                             "that cannot be re-derived is not auditable)")
    args = parser.parse_args()

    fields = tuple(args.fields.split(","))
    dims = dims_from_arm_yaml(args.arm_yaml, fields)
    dim = sum(dims.values())
    expected = json.loads(pathlib.Path(args.expected_slots).read_text(encoding="utf-8"))
    data = load_collection(args.shards, args.journal, expected_slots=expected)

    report = {
        "arms": args.arms,
        "n_expected": data["n_expected"],
        "n_episodes": data["n_episodes"],
        "n_steps": int(data["features"].shape[0]),
        "infra_failure_rate": data["infra_failure_rate"],
        "failures": data["failures"][:50],
        "dims": dims,
        "disclosure": graft_disclosure(args.arms),
        "fallback": False,
    }

    usable = data["n_episodes"] > 0 and int(data["features"].shape[0]) > 0
    if data["infra_failure_rate"] > INFRA_FAILURE_CEILING or not usable:
        # Predeclared fallback. Dims still come from the artifact, so even the
        # degenerate checkpoint has a meta the serving side can validate.
        report["fallback"] = (
            "infrastructure_failure_rate_above_ceiling" if usable else "no_usable_episodes"
        )
        params = zero_init(dim=dim, hidden=args.hidden, arms=args.arms)
        mu = sigma = None
        # The fold manifest is a required product even here: the disclosure has
        # to record that no selection happened and why.
        pathlib.Path(args.folds_out).write_text(json.dumps({
            "n_folds": 0, "seed": SEED, "grouped_by": "episode", "stratified_by": "task",
            "fold_of_episode": {}, "episodes_per_fold": {},
            "fallback": report["fallback"],
            "infra_failure_rate": data["infra_failure_rate"],
        }, indent=2), encoding="utf-8")
    else:
        if int(data["features"].shape[1]) != dim:
            raise SystemExit(
                f"collected features are {data['features'].shape[1]}-dim but the arm "
                f"yaml declares {dim}; the collection ran on a different artifact"
            )
        # Field slices in canonical concatenation order — robot_state last.
        offset, field_slices = 0, {}
        for name in fields:
            field_slices[name] = slice(offset, offset + dims[name])
            offset += dims[name]
        mu, sigma = fit_feature_stats(data["features"], field_slices=field_slices)
        normalized = (data["features"] - mu) / sigma
        # Re-quantize: the online encoder decides on Q(normalize(raw)), so the
        # head must be fit on that same representation.
        normalized = normalized.to(FEATURE_DTYPE).to(torch.float32)
        report["feature_scaling"] = {
            "per_field": {
                name: {"mu": float(mu[sl][0]), "sigma_pre_sqrtD": float(sigma[sl][0] / dim ** 0.5)}
                for name, sl in field_slices.items() if name != ROBOT_STATE
            },
            "robot_state": "per-dimension",
            "sqrtD_folded_into_sigma": dim ** 0.5,
            "normalized_abs_mean": float(normalized.abs().mean()),
            "normalized_l1_per_row": float(normalized.abs().sum(dim=1).mean()),
        }
        folds = grouped_folds(data["episode"], data["task"])
        head, selection, calibration_logits = fit_head(
            normalized, data["labels"], hidden=args.hidden, folds=folds,
        )
        # Before anything is grafted: a dead trunk is a constant router, and
        # every downstream number would describe a fixed coin flip.
        health = trunk_health(head, normalized)
        report["trunk_health"] = health
        assert_trunk_alive(health)
        # ...and a live trunk that learned nothing is just as useless.
        assert_head_beats_base_rate(selection)
        # Calibrated on the SHIPPED head's held-out fold, so the recorded rate
        # is the rate the deployed router will actually start at.
        delta0 = initial_student_bias(
            calibration_logits, arms=args.arms, target_rate=args.initial_student_rate)
        params = graft(head, arms=args.arms, delta0=delta0)
        report.update(selection)
        report["delta0"] = delta0
        report["initial_student_rate_target"] = args.initial_student_rate
        report["delta0_basis"] = {
            "source": "held_out_fold_of_the_shipped_head",
            "fold": selection["calibration_fold"],
            "n_steps": int(calibration_logits.numel()),
        }
        # R_tc has no student row to calibrate, so there is no realized rate to
        # report -- graft() starts it uniform on purpose (§3.8). Reporting the
        # disclosure instead keeps the record honest for every arm set, and
        # keeps the fit from dying on the one variant graft() went out of its
        # way to support.
        if "student" in ARM_SETS[args.arms]:
            report["delta0_basis"]["realized_student_rate"] = deployed_student_rate(
                params, normalized[folds[selection["calibration_fold"]]], arms=args.arms,
            )
        else:
            report["delta0_basis"]["realized_student_rate"] = None
            report["delta0_basis"]["no_student_arm"] = graft_disclosure(args.arms)
        manifest = fold_manifest(data["episode"], data["task"], data["uids"], folds)
        report["folds"] = {k: v for k, v in manifest.items() if k != "fold_of_episode"}
        pathlib.Path(args.folds_out).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    meta = save_router_weights(
        args.out, arms=args.arms, fields=fields, dims=dims,
        weights_version="v0", mu=mu, sigma=sigma, **params,
    )
    report["meta"] = meta
    payload = json.dumps(report, indent=2, default=str)
    if args.report:
        pathlib.Path(args.report).write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
