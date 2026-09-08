"""Teacher-driven shadow pass that labels every step for the RIT ladder fit.

Why a shadow rather than an offline replay
------------------------------------------
The risk column a RIT rung is fitted on is "how far this rung's action lands
from what full inference would have produced at this state". Producing it
offline means rebuilding the language model's input sequence from the stored
per-modality slices, and the collector's HDF5 keeps the slices but not the
scatter that produced them -- the state token and the image-token layout are
gone. Running the label inline, while the real stage-1 tensors are in hand,
removes that reconstruction entirely.

The executed action stays the teacher's own full-inference chunk, so the
trajectory this cohort walks is the teacher's; the cache is queried in the
shadow and never steers. That is what makes one pass yield both the score
distribution the gate's theta is cut from and the per-rung deviations the
ladder is fitted on.

Parity: the query keys and the search come from the same ``CacheOrchestrator``
the deployed arms run, driven by a calibration yaml whose judge is
``always_hit`` -- so the reported score is the winner's score under exactly the
production retrieval, and the winner is the entry a deployed FULL_HIT would
have taken.

Public interface: ``GrootRitShadow``, ``library_action_weights``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from openpi.cache.components.surface_judge import (
    compute_library_action_weights,
    weighted_chunk_deviation,
)
from openpi.cache.groot.interceptor import (
    _is_batched,
    _squeeze_values,
    _unsqueeze_values,
)
from openpi.cache.types import CheckpointID, DenoiseSchedule

logger = logging.getLogger(__name__)


def library_action_weights(pkl_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dim inverse-sigma weights and active mask of a library artifact.

    Loaded straight from the pickle rather than from a live backend so the
    numbers can be frozen once and handed to every server of the fleet; five
    processes each re-deriving them would also each hold a second copy of the
    chunks.
    """
    import pickle

    with open(pkl_path, "rb") as f:
        lib = pickle.load(f)
    chunks = torch.stack(
        [torch.as_tensor(e.payload.action_chunk, dtype=torch.float32) for e in lib["entries"]]
    )
    return compute_library_action_weights(chunks)


class GrootRitShadow:
    """Teacher policy that also writes one RIT calibration row per inference.

    Satisfies the same ``get_action`` protocol as the raw policy, the cache
    interceptor and the collector, so the server still hands the adapter
    exactly one object.

    Args:
        policy: the ``Gr00tPolicy`` whose transforms and model are used.
        runner: staged runner over the same model.
        orchestrator: built from the calibration yaml (always_search /
            always_hit / write never). Its verdict is read, never applied.
        out_path: JSONL sink, one row per step.
        warm_ts: resume timesteps to label, in ladder order.
        w / active_mask: library action weights from ``library_action_weights``.
        h_exec: executed window the deviation is averaged over (the driver's
            ``--replan-steps``; steps past it are never executed).
    """

    def __init__(
        self,
        policy,
        runner,
        *,
        orchestrator,
        out_path: str,
        warm_ts: list[float],
        w: torch.Tensor,
        active_mask: torch.Tensor,
        h_exec: int,
        experiment: str = "",
    ) -> None:
        self._policy = policy
        self._runner = runner
        self._orchestrator = orchestrator
        self._path = Path(out_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._warm_ts = [round(float(t), 4) for t in warm_ts]
        self._w = w
        self._active_mask = active_mask
        self._h_exec = int(h_exec)
        self._experiment = experiment
        self._task = ""
        self._episode_id = -1
        self._step = 0
        self._rows: list[dict] = []
        self._schedule: DenoiseSchedule | None = None

    # -- lifecycle -------------------------------------------------------

    def on_task_begin(self, task_key: str = "") -> None:
        if self._orchestrator is not None:
            self._orchestrator.on_task_begin(task_key)

    def on_task_end(self) -> None:
        if self._orchestrator is not None:
            self._orchestrator.on_task_end()

    def on_episode_start(
        self,
        experiment: str = "",
        task: str = "",
        episode_id: int = -1,
        episode_name: str = "",
        extra_metadata: dict | None = None,
    ) -> None:
        del episode_name
        self._task = task
        self._episode_id = episode_id
        self._step = 0
        self._rows = []
        self._schedule = self._runner.live_schedule()
        if self._orchestrator is not None:
            # Same identity the interceptor binds, so the search session this
            # cohort is scored under is the one a deployed arm would open.
            self._orchestrator.on_episode_start(
                task_key=task,
                episode_id=str(episode_id),
                extra_metadata=extra_metadata,
            )

    def on_episode_end(self, success: bool) -> None:
        """Flush the episode's rows, stamped with the outcome.

        Stamped at the end rather than per step because the fit reports
        success-conditioned diagnostics, and a row cannot know its episode's
        outcome while the episode is still running.
        """
        with self._path.open("a") as f:
            for row in self._rows:
                row["episode_success"] = bool(success)
                f.write(json.dumps(row) + "\n")
        self._rows = []
        if self._orchestrator is not None:
            self._orchestrator.on_episode_end()

    # -- inference -------------------------------------------------------

    def get_action(self, observations: dict[str, Any]) -> dict[str, Any]:
        obs_copy = observations.copy()
        is_batch = _is_batched(obs_copy)
        if not is_batch:
            obs_copy = _unsqueeze_values(obs_copy)
        for key, value in obs_copy.items():
            if not isinstance(value, np.ndarray):
                obs_copy[key] = np.array(value)

        normalized_input = self._policy.apply_transforms(obs_copy)

        with self._runner.session():
            stage1 = self._runner.run_stage1(normalized_input)

        cp1 = None
        try:
            cp1 = self._orchestrator.check(CheckpointID.CP1, stage1=stage1)
            with self._runner.session():
                stage2 = self._runner.run_stage2_llm(stage1)
                teacher = self._runner.run_stage3(stage2).action_pred
                row = self._label(cp1, stage2, teacher)
            action_cpu = self._to_storage_tensor(teacher)
            self._orchestrator.broadcast_action(action_cpu)
        finally:
            if self._orchestrator is not None:
                self._orchestrator.clear()

        self._rows.append(row)
        self._step += 1

        unnormalized = self._policy.unapply_transforms(
            {"action": action_cpu[None, ...]}
        )
        return unnormalized if is_batch else _squeeze_values(unnormalized)

    # -- labelling -------------------------------------------------------

    def _label(self, cp1, stage2, teacher) -> dict:
        """One calibration row: the score, the winner, and one y per rung."""
        ref = self._to_storage_tensor(teacher)
        row: dict[str, Any] = {
            "experiment": self._experiment,
            "task": self._task,
            "episode_id": int(self._episode_id),
            "step_idx": int(self._step),
            "s": None,
            "winner_id": None,
        }
        payload = None if cp1 is None else getattr(cp1, "payload", None)
        if cp1 is not None:
            score = getattr(cp1, "score", None)
            row["s"] = None if score is None else float(score)
            row["winner_id"] = getattr(cp1, "entry_id", None)
        if payload is None:
            # No winner: the step contributes a score-less row so the coverage
            # audit can see it, but it carries no risk label to fit on.
            return row

        row["y_full"] = weighted_chunk_deviation(
            torch.as_tensor(payload.action_chunk, dtype=torch.float32),
            ref,
            self._w,
            self._active_mask,
            self._h_exec,
        )
        for t in self._warm_ts:
            start_x = payload.intermediates.get(t)
            if start_x is None:
                raise RuntimeError(
                    f"library payload {row['winner_id']!r} has no intermediate at "
                    f"t={t}; the ladder cannot be calibrated against it"
                )
            chunk = self._runner.run_stage3_from(
                stage2, start_x, t, schedule=self._schedule
            ).action_pred
            rem = self._schedule.remaining_steps(t)
            row[f"y_rem{rem}"] = weighted_chunk_deviation(
                self._to_storage_tensor(chunk), ref, self._w, self._active_mask, self._h_exec
            )
        return row

    @staticmethod
    def _to_storage_tensor(chunk: torch.Tensor) -> torch.Tensor:
        out = chunk[0].detach().cpu().float().contiguous()
        if out.is_inference():
            out = out.clone()
        return out


def main() -> None:
    """Freeze one library's action weights so the fleet labels on one scale."""
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--library", required=True, help="library .pkl to derive (w, active_mask) from")
    ap.add_argument("--out", required=True, help="NPZ sink read by --rit-weights")
    args = ap.parse_args()
    w, active = library_action_weights(args.library)
    np.savez(args.out, w=w.numpy(), active_mask=active.numpy())
    print(
        f"wrote {args.out}: {int(active.sum())}/{active.numel()} active dims, "
        f"w in [{float(w[active].min()):.4f}, {float(w[active].max()):.4f}]"
    )


if __name__ == "__main__":
    main()
