"""Batch on-policy REINFORCE trainer for the X14 router (§3.5 / §3.7).

One batch, one Adam step. That is a frozen contract, not a default: a second
epoch over the same rollouts makes the data off-policy for the updated
parameters, and plain REINFORCE has no importance correction to pay for it.
Multiple epochs would require PPO-style clipping, which is a different
algorithm and would have to go back through plan review.

Frozen objective::

    R_ep  = success − λ · (Σ_t cost(arm_executed_t)) / T_max
    b     = mean_ep R_ep                       (batch-mean baseline)
    loss  = −(1/N) Σ_ep (R_ep − b) · Σ_t log π(a_t | f_t)  −  β · mean_t H_t

with β = 0.01, lr = 3e-4, grad-norm clip = 1.0, Adam defaults.

On-policy verification
----------------------
The behaviour policy's authority is the dump (the logits recorded at sampling
time). Before any gradient is taken, the trainer recomputes those logits from
the dumped features on the same CPU reference — fp32, single-threaded, the same
``addmv`` the judge used — and requires **bitwise** equality. A mismatch means
the batch was not produced by the weights it claims, so the whole episode is
rejected rather than silently biasing the gradient. Rejections are counted per
batch; a sustained rate above ~1% is an alert, not an accepted loss.

Artifacts per step: a versioned router checkpoint (consumable by the serving
yaml), an atomic trainer checkpoint (model + optimizer + RNG + consumed
batches), and one ``metrics.jsonl`` line.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import random
from typing import Optional

import numpy as np
import torch

from openpi.cache.components.mlp_router_judge import (
    ARM_SETS,
    FEATURE_DTYPE,
    RouterWeights,
    pin_router_threads,
    save_router_weights,
)

from exp.rl_router.batch_package import BatchManifest, read_jsonl, verify_package

# Alert threshold for the rejection rate across batches (risk register R10). A
# single rejection now aborts its batch, so this is monitored over the run's
# repair rounds rather than inside one step.
REJECT_RATE_ALERT = 0.01


class AdmissionError(RuntimeError):
    """A batch failed admission and was aborted untouched.

    Carries the per-episode rejection reasons so the caller can dispatch a
    repair round for exactly those slots. Raised *before* any parameter,
    optimizer, version, or ledger mutation, so catching it and retrying the
    repaired batch is safe.

    Both admission stages raise it: loading (shard digest, sidecar identity /
    continuity, judge-vs-interceptor arm disagreement) and the bitwise
    on-policy check. Loading failures used to escape as bare exceptions, which
    the run loop could only turn into an ALERT — the same defect was repairable
    if it named its slot and fatal if it did not.
    """

    def __init__(self, message: str, *, rejected: list[dict]) -> None:
        super().__init__(message)
        self.rejected = rejected


class EpisodeAdmissionError(AdmissionError):
    """One episode could not be loaded; carries the slot to repair."""

    def __init__(self, *, task_uid: str, attempt: int, reason: str, detail: str) -> None:
        super().__init__(
            f"episode {task_uid} (attempt {attempt}) failed load admission: {detail}",
            rejected=[{"task_uid": task_uid, "attempt": attempt, "reason": reason,
                       "detail": detail}],
        )


# ---------------------------------------------------------------------------
# Hyper-parameters (frozen; see module docstring)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TrainerHParams:
    """Every knob the objective depends on, recorded into each checkpoint."""

    arm_costs: dict[str, float]      # normalized GPU-time, teacher == 1.0 (M5a)
    lam: float                       # λ, from the M5c pilot
    t_max: int                       # episode step cap, normalizes the cost term
    lr: float = 3e-4
    entropy_beta: float = 0.01
    grad_clip: float = 1.0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Episode payload
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TrainEpisode:
    """One rollout, already joined and verified."""

    task_uid: str
    attempt: int
    success: bool
    features: torch.Tensor            # [K, D] fp16 — exactly the bytes the judge saw
    arm_sampled: list[str]
    arm_executed: list[str]
    dumped_logits: Optional[torch.Tensor] = None    # [K, A] fp32, None only in synthetic tests
    dumped_logprobs: Optional[torch.Tensor] = None  # [K] fp32, the sampled arm's log pi

    def __post_init__(self) -> None:
        k = int(self.features.shape[0])
        if not (len(self.arm_sampled) == len(self.arm_executed) == k):
            raise ValueError(
                f"episode {self.task_uid!r}: {k} feature rows but "
                f"{len(self.arm_sampled)} sampled / {len(self.arm_executed)} executed arms"
            )


# ---------------------------------------------------------------------------
# Frozen objective pieces (pure functions so the golden test can pin them)
# ---------------------------------------------------------------------------


def episode_reward(
    *, success: bool, arm_executed: list[str], arm_costs: dict[str, float], lam: float, t_max: int
) -> float:
    """``success − λ · (Σ_t cost(executed_t)) / T_max``.

    Cost is billed to the arm that *executed*, not the one that was sampled: a
    cache arm that found an empty library ran the teacher, and charging it the
    cache price would make the router look cheaper than it is.
    """
    if t_max <= 0:
        raise ValueError(f"t_max must be positive, got {t_max}")
    try:
        total = sum(arm_costs[a] for a in arm_executed)
    except KeyError as exc:
        raise ValueError(f"no measured cost for arm {exc.args[0]!r}; run microbench_cost.py") from None
    return float(success) - lam * total / t_max


def reinforce_loss(
    *,
    logprob_sums: torch.Tensor,   # [N] sum_t log pi(a_t) per episode
    rewards: torch.Tensor,        # [N]
    entropy_mean: torch.Tensor,   # scalar, averaged over every step in the batch
    entropy_beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(loss, advantages)`` for the frozen objective.

    The baseline is the batch mean and is detached: it is a variance-reduction
    constant, and letting gradients flow through it would change the estimator.
    """
    baseline = rewards.mean().detach()
    advantages = (rewards - baseline).detach()
    policy_term = -(advantages * logprob_sums).mean()
    return policy_term - entropy_beta * entropy_mean, advantages


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class RouterPolicy(torch.nn.Module):
    """The trainable twin of ``MlpRouterJudge``'s network.

    Two forward paths on purpose:

      - ``reference_logits`` reproduces the judge step by step with the same
        ``addmv``, for the bitwise on-policy check;
      - ``forward`` is the batched equivalent used for the gradient, where a
        different reduction order costs nothing.
    """

    def __init__(self, W1: torch.Tensor, b1: torch.Tensor, W2: torch.Tensor,
                 b2: torch.Tensor, *, arms: str, temperature: float = 1.0) -> None:
        super().__init__()
        self.W1 = torch.nn.Parameter(W1.detach().clone().to(torch.float32))
        self.b1 = torch.nn.Parameter(b1.detach().clone().to(torch.float32))
        self.W2 = torch.nn.Parameter(W2.detach().clone().to(torch.float32))
        self.b2 = torch.nn.Parameter(b2.detach().clone().to(torch.float32))
        self.arms_key = arms
        self.arms = ARM_SETS[arms]
        self.temperature = float(temperature)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """[K, D] -> [K, A] logits."""
        hidden = torch.relu(features.to(torch.float32) @ self.W1.T + self.b1)
        return hidden @ self.W2.T + self.b2

    @torch.no_grad()
    def reference_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Per-row replay of the judge's forward, for the bitwise check."""
        rows = []
        for row in features:
            hidden = torch.addmv(self.b1, self.W1, row.to(torch.float32)).clamp_min(0.0)
            rows.append(torch.addmv(self.b2, self.W2, hidden))
        return torch.stack(rows) if rows else torch.zeros(0, len(self.arms))

    @classmethod
    def from_weights(cls, weights: RouterWeights, *, temperature: float = 1.0) -> "RouterPolicy":
        return cls(weights.W1, weights.b1, weights.W2, weights.b2,
                   arms=weights.arms, temperature=temperature)


def encoder_meta_from_weights(weights: RouterWeights) -> dict:
    """Everything ``save_router_weights`` needs, taken from the loaded checkpoint."""
    return {
        "fields": list(weights.fields),
        "dims": dict(weights.dims),
        "mu": None if weights.mu is None else weights.mu.tolist(),
        "sigma": None if weights.sigma is None else weights.sigma.tolist(),
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class RouterTrainer:
    """Owns the policy, the optimizer, and the consumed-batch ledger."""

    def __init__(
        self,
        policy: RouterPolicy,
        hparams: TrainerHParams,
        *,
        weights_version: str = "v0",
        temperature: float = 1.0,
        encoder_meta: Optional[dict] = None,
    ) -> None:
        pin_router_threads()
        self.policy = policy
        self.hparams = hparams
        self.temperature = float(temperature)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=hparams.lr)
        self.weights_version = weights_version
        self.consumed_batches: list[dict] = []
        # The serving encoder's identity (fields / dims / robot_state stats)
        # lives IN the checkpoint. Without it a recovery export has nothing to
        # stamp into the weights meta and cannot produce a file the server will
        # load — the checkpoint would hold a durable update that can never be
        # served.
        self.encoder_meta: dict = dict(encoder_meta or {})

    # -- one batch, one step ------------------------------------------------

    def train_batch(
        self,
        episodes: list[TrainEpisode],
        *,
        batch_id: str = "",
        package_sha256: str = "",
        expected_weights_version: Optional[str] = None,
        verify: bool = True,
    ) -> dict:
        """Take exactly one Adam step on the FULL batch. Returns the metrics row.

        Admission is all-or-nothing (§3.5 / §3.6). If any episode fails
        verification the batch is aborted by raising ``AdmissionError`` with
        **no** change to parameters, optimizer state, weights version, or the
        consumed-batch ledger, and the caller schedules a repair round. Dropping
        the failures and stepping on what is left would silently change the
        estimator: the objective divides by N, and N is frozen before dispatch,
        so a short batch is a different experiment reported under the same name.
        """
        if not episodes:
            raise ValueError("refusing to step on an empty batch")
        if any(b["batch_id"] == batch_id for b in self.consumed_batches) and batch_id:
            # Idempotent re-entry after a trainer crash: the checkpoint already
            # contains this batch's update.
            return {"batch_id": batch_id, "skipped": "already_consumed"}
        if expected_weights_version is not None and expected_weights_version != self.weights_version:
            # The batch was collected under a different policy than the one this
            # trainer holds — a hot-swap or resume race. Its gradient would be
            # credited to the wrong version.
            raise AdmissionError(
                f"batch {batch_id!r} was collected under weights_version "
                f"{expected_weights_version!r} but the trainer holds "
                f"{self.weights_version!r}",
                rejected=[],
            )

        rejected: list[dict] = []
        if verify:
            for ep in episodes:
                reason = self._verify(ep)
                if reason is not None:
                    rejected.append(
                        {"task_uid": ep.task_uid, "attempt": ep.attempt, "reason": reason}
                    )
        if rejected:
            raise AdmissionError(
                f"batch {batch_id!r}: {len(rejected)}/{len(episodes)} episodes failed "
                "on-policy verification; aborting the batch untouched so the caller "
                "can repair it and update on the full N",
                rejected=rejected,
            )
        kept = list(episodes)

        logprob_sums, entropies, rewards = [], [], []
        for ep in kept:
            logits = self.policy(ep.features)
            scaled = logits / self.temperature
            logp = torch.log_softmax(scaled, dim=1)
            idx = torch.tensor([self.policy.arms.index(a) for a in ep.arm_sampled])
            logprob_sums.append(logp[torch.arange(len(idx)), idx].sum())
            entropies.append(-(logp.exp() * logp).sum(dim=1))
            rewards.append(episode_reward(
                success=ep.success, arm_executed=ep.arm_executed,
                arm_costs=self.hparams.arm_costs, lam=self.hparams.lam,
                t_max=self.hparams.t_max,
            ))

        loss, advantages = reinforce_loss(
            logprob_sums=torch.stack(logprob_sums),
            rewards=torch.tensor(rewards, dtype=torch.float32),
            entropy_mean=torch.cat(entropies).mean(),
            entropy_beta=self.hparams.entropy_beta,
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.hparams.grad_clip
        )
        self.optimizer.step()

        self.weights_version = _next_version(self.weights_version)
        ledger_entry = {
            "batch_id": batch_id, "package_sha256": package_sha256,
            "weights_version": self.weights_version, "n_episodes": len(kept),
        }
        self.consumed_batches.append(ledger_entry)
        metrics = {
            "batch_id": batch_id,
            "weights_version": self.weights_version,
            "n_episodes": len(kept),
            "n_rejected": 0,          # a batch only reaches here fully admitted
            "loss": float(loss.item()),
            "grad_norm": float(grad_norm),
            "mean_reward": float(np.mean(rewards)),
            "mean_success": float(np.mean([e.success for e in kept])),
            "mean_advantage_abs": float(advantages.abs().mean()),
            "arm_executed_rate": _arm_rates(kept),
        }
        # The whole row goes INTO the checkpoint. A ledger that only remembers
        # the batch id cannot rebuild loss / grad_norm / reward / arm rates after
        # a crash, so those numbers would be lost for that batch forever — and
        # the training curve would have a hole exactly where something went
        # wrong.
        ledger_entry["metrics"] = metrics
        return metrics

    def _verify(self, ep: TrainEpisode) -> Optional[str]:
        """Bitwise on-policy check; returns a rejection reason or None.

        Both halves of the behaviour record are checked. The logits pin the
        forward pass; the sampled log-probability pins the *sampling* the
        gradient is weighted by — a temperature or arm-index drift would leave
        the logits identical while making every recorded action's probability
        wrong, which the logits check alone cannot see.
        """
        if ep.dumped_logits is None:
            return None
        recomputed = self.policy.reference_logits(ep.features)
        if recomputed.shape != ep.dumped_logits.shape:
            return "logits_shape_mismatch"
        if not torch.equal(recomputed, ep.dumped_logits.to(torch.float32)):
            return "logits_not_bitwise_equal"
        if ep.dumped_logprobs is not None:
            try:
                idx = torch.tensor([self.policy.arms.index(a) for a in ep.arm_sampled])
            except ValueError:
                return "unknown_arm_in_dump"
            logp = torch.log_softmax(recomputed / self.temperature, dim=1)
            sampled = logp[torch.arange(len(idx)), idx]
            if not torch.equal(sampled, ep.dumped_logprobs.to(torch.float32)):
                return "logprob_not_bitwise_equal"
        return None

    # -- checkpointing ------------------------------------------------------

    def save_checkpoint(self, path: str | os.PathLike) -> None:
        """Atomically persist everything needed to resume identically.

        The RNG states ride along so a resumed run is not merely "close" to the
        uninterrupted one — the equivalence is what lets a crash mid-run keep
        the same experiment rather than start a new one.
        """
        blob = {
            "model": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "consumed_batches": list(self.consumed_batches),
            "hparams": self.hparams.to_dict(),
            "weights_version": self.weights_version,
            "arms": self.policy.arms_key,
            "temperature": self.temperature,
            "encoder_meta": dict(self.encoder_meta),
        }
        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        torch.save(blob, tmp)
        os.replace(tmp, target)

    @classmethod
    def load_checkpoint(cls, path: str | os.PathLike) -> "RouterTrainer":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        state = blob["model"]
        policy = RouterPolicy(
            state["W1"], state["b1"], state["W2"], state["b2"],
            arms=blob["arms"], temperature=blob["temperature"],
        )
        trainer = cls(
            policy, TrainerHParams(**blob["hparams"]),
            weights_version=blob["weights_version"], temperature=blob["temperature"],
            encoder_meta=blob.get("encoder_meta"),
        )
        trainer.optimizer.load_state_dict(blob["optimizer"])
        trainer.consumed_batches = list(blob["consumed_batches"])
        rng = blob["rng"]
        torch.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        return trainer

    def export_router_weights(self, path: str | os.PathLike) -> dict:
        """Write the serving-side checkpoint for the next bundle version.

        Takes its encoder identity from the trainer's own durable state, so a
        recovery export needs nothing but the checkpoint.
        """
        meta = self.encoder_meta
        if not meta.get("fields"):
            raise ValueError(
                "trainer has no encoder meta: cannot stamp a servable weights file. "
                "Construct it from RouterWeights (or resume from a checkpoint that "
                "carries encoder_meta)."
            )
        mu = None if meta.get("mu") is None else torch.tensor(meta["mu"], dtype=torch.float32)
        sigma = (None if meta.get("sigma") is None
                 else torch.tensor(meta["sigma"], dtype=torch.float32))
        return save_router_weights(
            path, W1=self.policy.W1.data, b1=self.policy.b1.data,
            W2=self.policy.W2.data, b2=self.policy.b2.data,
            arms=self.policy.arms_key, fields=tuple(meta["fields"]),
            dims={k: int(v) for k, v in meta["dims"].items()},
            weights_version=self.weights_version, mu=mu, sigma=sigma,
        )

    def state_summary(self) -> dict:
        """Small JSON the conductor fetches instead of reading remote artifacts.

        The trainer's artifacts live on the serving host and the loop runs on
        the conductor; the two do not share a filesystem. Rather than have the
        loop stat paths that do not exist for it, the trainer publishes this
        summary next to its checkpoint and the loop resumes from the summary.
        """
        return {
            "weights_version": self.weights_version,
            # Ledger entries carry their full metrics row; strip it from the
            # published summary (the conductor only needs the identities) but
            # keep the ids so resume can detect a metrics gap.
            "consumed_batches": [
                {k: v for k, v in entry.items() if k != "metrics"}
                for entry in self.consumed_batches
            ],
            "arms": self.policy.arms_key,
            "hparams": self.hparams.to_dict(),
        }


def _next_version(current: str) -> str:
    """``v3`` -> ``v4``; anything unparsable restarts the counter explicitly."""
    if current.startswith("v") and current[1:].isdigit():
        return f"v{int(current[1:]) + 1}"
    return "v1"


def _arm_rates(episodes: list[TrainEpisode]) -> dict[str, float]:
    counts: dict[str, int] = {}
    total = 0
    for ep in episodes:
        for arm in ep.arm_executed:
            counts[arm] = counts.get(arm, 0) + 1
            total += 1
    return {k: v / total for k, v in sorted(counts.items())} if total else {}


# ---------------------------------------------------------------------------
# Batch loading (three-source join already decided by batch_package)
# ---------------------------------------------------------------------------


def load_batch(
    manifest: BatchManifest,
    *,
    shard_dir: str | pathlib.Path,
    package_dir: str | pathlib.Path,
    n_arms: int,
) -> list[TrainEpisode]:
    """Materialise the ``training_selected`` episodes of a complete batch.

    Refuses to load a short batch: the frozen N is part of the experiment, and
    ``batch_package`` already knows which slots are missing and needs a repair
    round rather than a smaller Adam step.
    """
    if not manifest.complete:
        raise ValueError(
            f"batch {manifest.batch_id} is short {len(manifest.missing_slots)} slot(s); "
            "run a repair round before training"
        )
    shard_dir = pathlib.Path(shard_dir)
    executed = _executed_arms(read_jsonl(pathlib.Path(package_dir) / "per_step_rows_batch.jsonl"))

    episodes: list[TrainEpisode] = []
    for record in manifest.selected:
        try:
            episodes.append(_load_episode(record, shard_dir, executed, n_arms))
        except _EpisodeDefect as defect:
            raise EpisodeAdmissionError(
                task_uid=record.task_uid, attempt=record.attempt,
                reason=defect.reason, detail=str(defect),
            ) from defect
        except Exception as exc:  # noqa: BLE001 - see below
            # Catch-all by design: a malformed artifact can raise almost
            # anything (KeyError, ValueError, RuntimeError from a bad reshape).
            # Whatever it is, it belongs to ONE slot and the correct response is
            # a zero-update repair of that slot, not an ALERT that stops the run.
            raise EpisodeAdmissionError(
                task_uid=record.task_uid, attempt=record.attempt,
                reason="episode_unreadable", detail=f"{type(exc).__name__}: {exc}",
            ) from exc
    return episodes


class _EpisodeDefect(ValueError):
    """One episode's artifacts are unusable; ``reason`` names the slot's fault."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _load_episode(record, shard_dir: pathlib.Path, executed: dict, n_arms: int) -> TrainEpisode:
    """Materialise one verified episode, or raise ``_EpisodeDefect``."""
    raw = (shard_dir / record.shard).read_bytes()
    if record.sha256 is not None and _sha(raw) != record.sha256:
        raise _EpisodeDefect(
            "shard_digest_mismatch",
            f"shard {record.shard} digest differs from the manifest — the file "
            "changed after it was declared complete",
        )
    features = torch.frombuffer(bytearray(raw), dtype=FEATURE_DTYPE).reshape(
        record.rows, -1
    )
    sidecar = _load_sidecar(shard_dir / record.sidecar, record)
    key = (record.task_uid, record.attempt, record.weights_version)
    arm_executed = executed.get(key, [])
    if len(arm_executed) != record.rows:
        raise _EpisodeDefect(
            "client_row_count_mismatch",
            f"episode {record.task_uid}: {record.rows} feature rows vs "
            f"{len(arm_executed)} client rows",
        )
    for row, mapped in zip(sidecar, arm_executed, strict=True):
        if row["arm_mapped"] != mapped:
            raise _EpisodeDefect(
                "arm_mapping_disagreement",
                f"episode {record.task_uid} step {row['decision_idx']}: judge mapped "
                f"{row['arm_mapped']!r} but the interceptor executed {mapped!r}",
            )
    return (TrainEpisode(
        task_uid=record.task_uid, attempt=record.attempt, success=record.success,
        features=features,
        arm_sampled=[r["arm_sampled"] for r in sidecar],
        arm_executed=list(arm_executed),
        dumped_logits=torch.tensor([r["logits"] for r in sidecar],
                                   dtype=torch.float32).reshape(record.rows, n_arms),
        dumped_logprobs=torch.tensor([r["logprob_sampled"] for r in sidecar],
                                     dtype=torch.float32),
    ))


def _load_sidecar(path: pathlib.Path, record) -> list[dict]:
    """Read and fully validate one episode's behaviour sidecar.

    The sidecar is the behaviour-policy authority — the logits and log-probs the
    gradient is weighted by — so it gets the same treatment as the features
    rather than being trusted for its row count alone:

      - **digest** — it must be the file the manifest declared complete;
      - **five-key identity, per row** — every row must belong to this exact
        ``(task_uid, attempt, batch_id, weights_version)``. Without this a row
        from a neighbouring episode or a stale attempt joins silently, and the
        gradient is credited to the wrong trajectory;
      - **dense ``0..K-1``** — duplicates and holes both produce a
        correctly-sized list, so a length check alone cannot see them, and a
        repeated ``decision_idx`` would double-weight one step.
    """
    raw = path.read_bytes()
    if record.sidecar_sha256 is not None and _sha(raw) != record.sidecar_sha256:
        raise _EpisodeDefect(
            "sidecar_digest_mismatch",
            f"sidecar {path.name} digest differs from the manifest — the behaviour "
            "record changed after it was declared complete",
        )
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    for row in rows:
        actual = (str(row.get("task_uid")), int(row.get("attempt", -1)),
                  str(row.get("batch_id")), str(row.get("weights_version")))
        expected = (record.task_uid, record.attempt, record.batch_id, record.weights_version)
        if actual != expected:
            raise _EpisodeDefect(
                "sidecar_identity_mismatch",
                f"sidecar {path.name} row identity {actual} != episode identity {expected}",
            )
    indices = sorted(int(r["decision_idx"]) for r in rows)
    if indices != list(range(record.rows)):
        raise _EpisodeDefect(
            "sidecar_decision_idx_discontinuous",
            f"sidecar {path.name} decision_idx {indices[:8]}... is not dense "
            f"0..{record.rows - 1} (duplicate or missing verdict)",
        )
    rows.sort(key=lambda r: int(r["decision_idx"]))
    return rows


def _executed_arms(client_rows: list[dict]) -> dict[tuple, list[str]]:
    """``arm_executed`` per episode, ordered by decision_idx."""
    grouped: dict[tuple, list[tuple[int, str]]] = {}
    for row in client_rows:
        if row.get("_kind") is not None:
            continue
        ro = row.get("router_outputs")
        if not ro:
            continue
        key = (str(row.get("task_uid")), int(row.get("attempt", -1)),
               str(ro.get("weights_version")))
        grouped.setdefault(key, []).append((int(ro["decision_idx"]), str(ro["arm_executed"])))
    return {k: [a for _, a in sorted(v)] for k, v in grouped.items()}


def _sha(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="batch_package manifest json")
    parser.add_argument("--package", required=True)
    parser.add_argument("--shards", required=True)
    parser.add_argument("--checkpoint", required=True, help="trainer checkpoint path")
    parser.add_argument("--weights-in", default="", help="router weights to start from")
    parser.add_argument("--weights-out", required=True, help="router weights for the next bundle")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--lam", type=float, required=True)
    parser.add_argument("--t-max", type=int, required=True)
    parser.add_argument("--arm-costs", required=True, help='json, e.g. {"teacher":1.0,"cache":0.02}')
    parser.add_argument("--export-meta-out", default="",
                        help="small json the conductor fetches to verify the export")
    parser.add_argument("--state-out", default="",
                        help="small json holding weights_version + consumed ledger")
    parser.add_argument("--rejected-out", default="",
                        help="written on admission failure: the slots to repair")
    parser.add_argument("--state-only", action="store_true",
                        help="regenerate the state summary from the checkpoint and exit")
    parser.add_argument("--export-only", action="store_true",
                        help="re-export weights from an existing checkpoint and exit "
                             "(idempotent recovery for a crash between save and export)")
    args = parser.parse_args()

    if args.state_only:
        # The state file is a CACHE of the checkpoint's ledger, not a second
        # source of truth. Regenerating it before every resume closes the window
        # between "checkpoint written" and "state published": a crash in there
        # would otherwise have the loop re-dispatch a batch the trainer already
        # consumed.
        trainer = RouterTrainer.load_checkpoint(args.checkpoint)
        _write_json(args.state_out, trainer.state_summary())
        print(json.dumps(trainer.state_summary(), indent=2, default=str))
        return

    if args.export_only:
        # Crash window between save_checkpoint and export: the update IS durable
        # (it is in the checkpoint), only the serving artifact is missing. Replay
        # just the export rather than re-running the batch.
        trainer = RouterTrainer.load_checkpoint(args.checkpoint)
        meta = _export(trainer, args)
        # The update is already in the checkpoint, so its metrics row belongs in
        # the ledger too — otherwise the batch that was applied leaves no trace
        # in metrics.jsonl and the training curve loses a step.
        backfilled = _append_recovery_metrics(trainer, args)
        _write_json(args.state_out, trainer.state_summary())
        print(json.dumps({"re_exported": meta["weights_version"],
                          "metrics_backfilled": backfilled}, indent=2))
        return

    # Verify the received package BEFORE anything reads it: a partially-copied
    # package would otherwise present as a short batch and trigger a pointless
    # repair round for episodes that actually ran.
    package_meta = verify_package(args.package)
    raw = json.loads(pathlib.Path(args.manifest).read_text(encoding="utf-8"))
    manifest = _manifest_from_dict(raw)
    if manifest.batch_id != package_meta["batch_id"]:
        raise SystemExit(
            f"manifest batch {manifest.batch_id!r} does not belong to package "
            f"{package_meta['batch_id']!r}"
        )

    if pathlib.Path(args.checkpoint).exists():
        trainer = RouterTrainer.load_checkpoint(args.checkpoint)
    else:
        if not args.weights_in:
            raise SystemExit("--weights-in is required for the first batch")
        weights = RouterWeights.load(args.weights_in)
        trainer = RouterTrainer(
            RouterPolicy.from_weights(weights),
            TrainerHParams(arm_costs=json.loads(args.arm_costs), lam=args.lam, t_max=args.t_max),
            weights_version=weights.weights_version,
            encoder_meta=encoder_meta_from_weights(weights),
        )

    try:
        # Loading is part of admission, not a preamble to it. A bad shard or a
        # malformed sidecar has to surface as "repair these slots" with the same
        # zero-update guarantee as a parity rejection; raising a bare exception
        # here would give the run loop nothing to act on but an ALERT.
        episodes = load_batch(
            manifest, shard_dir=args.shards, package_dir=args.package,
            n_arms=len(trainer.policy.arms),
        )
        metrics = trainer.train_batch(
            episodes, batch_id=manifest.batch_id,
            package_sha256=package_meta["package_sha256"],
            expected_weights_version=manifest.weights_version,
        )
    except AdmissionError as exc:
        # Exit non-zero WITHOUT writing a checkpoint or weights: the run loop
        # treats this as "repair these slots", and any artifact written here
        # would make a rejected batch look consumed. The rejected list is
        # published so the loop can quarantine exactly those attempts.
        pathlib.Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
        with pathlib.Path(args.metrics).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "batch_id": manifest.batch_id, "admission_failed": True,
                "rejected": exc.rejected, "error": str(exc),
            }) + "\n")
        _write_json(args.rejected_out, {
            "batch_id": manifest.batch_id, "weights_version": manifest.weights_version,
            "rejected": exc.rejected, "error": str(exc),
        })
        raise SystemExit(f"ADMISSION FAILED: {exc}") from exc

    # Checkpoint BEFORE exporting serving weights: a crash between the two
    # replays the export from the checkpoint (see --export-only), whereas the
    # reverse order could serve a version the trainer has no record of.
    if metrics.get("skipped"):
        # Idempotent re-entry: this batch's update is already in the checkpoint,
        # so there is nothing to write. Exporting here would stamp the CURRENT
        # (unadvanced) version into the NEXT version's filename and hand the
        # fleet a mislabelled policy.
        print(json.dumps(metrics, indent=2))
        return

    # Transaction order: the checkpoint makes the update durable, the export
    # makes it servable, metrics record it, and the state summary is published
    # LAST so it never advertises a step whose artifacts are missing. Every
    # earlier crash point is replayed by --export-only, which is idempotent.
    trainer.save_checkpoint(args.checkpoint)
    meta = _export(trainer, args)
    with pathlib.Path(args.metrics).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(metrics) + "\n")
    _write_json(args.state_out, trainer.state_summary())
    print(json.dumps({**{k: v for k, v in metrics.items() if k != "rejected"},
                      "export": meta}, indent=2))


def _export(trainer: "RouterTrainer", args) -> dict:
    """Write the serving weights + the small meta the conductor fetches back."""
    meta = trainer.export_router_weights(args.weights_out)
    _write_json(args.export_meta_out, {
        "weights_version": meta["weights_version"],
        "model_sha": meta["model_sha"],
        "encoder_version": meta["encoder_version"],
        "arms": meta["arms"],
        "path": str(args.weights_out),
    })
    return meta


def _append_recovery_metrics(trainer: "RouterTrainer", args) -> list[str]:
    """Backfill the metrics rows of EVERY consumed batch that is missing one.

    The full row was stored in the checkpoint's ledger at update time, so this
    replays the real numbers rather than a four-field stub — a recovered curve
    that silently lost loss / grad_norm / reward would be worse than no curve.

    Every consumed batch is checked, not just the last one: a crash after the
    export still leaves the weights present, so keying recovery off "are the
    weights missing" would leave that batch's row absent forever.
    """
    if not args.metrics or not trainer.consumed_batches:
        return []
    existing = {
        row.get("batch_id") for row in read_jsonl(args.metrics)
        if not row.get("admission_failed")
    }
    backfilled: list[str] = []
    pathlib.Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    with pathlib.Path(args.metrics).open("a", encoding="utf-8") as fh:
        for entry in trainer.consumed_batches:
            batch_id = entry.get("batch_id")
            if batch_id in existing:
                continue
            row = dict(entry.get("metrics") or {})
            if not row:
                row = {k: v for k, v in entry.items() if k != "metrics"}
                row["metrics_incomplete"] = True
            fh.write(json.dumps({
                **row, "batch_id": batch_id, "recovered": True,
                "note": "metrics row replayed from the checkpoint ledger after a "
                        "crash in the update transaction tail",
            }) + "\n")
            backfilled.append(str(batch_id))
    return backfilled


def _write_json(path: str, payload: dict) -> None:
    if not path:
        return
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, target)


def _manifest_from_dict(raw: dict) -> BatchManifest:
    from exp.rl_router.batch_package import EpisodeRecord

    return BatchManifest(
        batch_id=raw["batch_id"],
        weights_version=raw["weights_version"],
        expected_slots=raw["expected_slots"],
        selected=[EpisodeRecord(**r) for r in raw["training_selected"]],
        superseded=raw.get("superseded", []),
        rejected=raw.get("rejected", []),
        missing_slots=raw.get("missing_slots", []),
    )


if __name__ == "__main__":
    main()
