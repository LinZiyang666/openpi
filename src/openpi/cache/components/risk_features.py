"""RiskFeatureBuilder — the X15 A-tier runtime feature vector (59 dims).

Turns one step's retrieval evidence into the exact vector the risk model was
trained on. Everything here is teacher-free by construction: reading the
teacher at decision time would cost precisely what the gate exists to save.

Layout (A-tier, frozen; see the X15 plan 6.2):

    f1  fused top-k scores                       5        (k = 5)
    f2  winner's per-field scores                3
    f3  fused margin + per-field own margins     4
    f5  robot_state delta (query - neighbour)    32
    f6  ||f5||                                   1
    f7  neighbour phase, |phase gap|             2
    f8  top-k action-chunk variance              1
    f9  top-k same-trajectory rate               1
    f11 task embedding (frozen artifact table)   8
    f12 query step fraction                      1
    f13 retrieval coverage (n_results / k)       1
                                                 --
                                                 59

f8 and f9 are the library's own uncertainty: neighbours can all score highly
and still disagree about what to do (f8), or all come from one trajectory so
that their agreement is not independent evidence (f9). Neither is visible in
the similarity scores alone.

Two time axes meet in f7 and they are NOT the same unit. The query side counts
inference cycles (``decision_idx``), while a library entry's ``step_idx`` also
counts inference cycles — of the episode that built the library, which may have
used a different replan interval. Both are converted to physical environment
steps before comparison; ``exp/markov_sufficiency/_timeaxis.py`` documents the
same hazard for the offline analysis path.

Missing or malformed inputs raise rather than silently zero-fill: the judge
catches and fails safe to teacher, which keeps "we could not tell" distinct
from "we looked and it was fine".

Key dependencies: ``StepRetrievalFeatures`` (openpi.cache.storage_types),
``PayloadView`` (neighbour entries), and the frozen feature schema recorded in
the risk-model artifact.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

import torch

# Canonical field order — the vector is positional, so this must never be
# reordered without retraining and bumping the schema digest.
CANONICAL_FIELDS: tuple[str, ...] = ("vision_0", "vision_1", "robot_state")
TOP_K = 5
N_TASKS = 10
TASK_EMBED_DIM = 8
ROBOT_STATE_DIM = 32
FEATURE_DIM = (
    TOP_K + 3 + 4 + ROBOT_STATE_DIM + 1 + 2 + 1 + 1 + TASK_EMBED_DIM + 1 + 1
)  # 59


def feature_schema_digest(
    *,
    fields: tuple[str, ...] = CANONICAL_FIELDS,
    top_k: int = TOP_K,
    n_tasks: int = N_TASKS,
    task_embed_dim: int = TASK_EMBED_DIM,
    robot_state_dim: int = ROBOT_STATE_DIM,
) -> str:
    """Stable digest of the feature layout, stored in the model artifact.

    A plain content hash of the layout parameters: if any of them changes the
    vector means something different, and loading an old checkpoint against the
    new builder must fail loudly instead of scoring garbage.
    """
    payload = (
        f"{'|'.join(fields)}|k={top_k}|tasks={n_tasks}|te={task_embed_dim}"
        f"|rs={robot_state_dim}|f8f9|v2"
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def default_task_embedding_table() -> torch.Tensor:
    """The frozen [N_TASKS, TASK_EMBED_DIM] task embedding table.

    Deterministic by construction so every process builds the identical table;
    it is nevertheless stored in the model artifact rather than regenerated,
    because a table the runtime re-derives is a table nobody can audit against
    the weights that were fitted on it.
    """
    gen = torch.Generator().manual_seed(0x15A5)
    return torch.randn(N_TASKS, TASK_EMBED_DIM, generator=gen)


class RiskFeatureBuilder:
    """Build the 59-dim A-tier feature vector for one decision."""

    def __init__(
        self,
        *,
        task_index: int,
        replan_steps: int,
        library_replan_steps: int,
        t_max: int = 520,
        top_k: int = TOP_K,
        task_embedding_table: Optional[torch.Tensor] = None,
        action_sigma: Optional[torch.Tensor] = None,
    ) -> None:
        if replan_steps < 1 or library_replan_steps < 1:
            raise ValueError(
                "risk_features: replan_steps and library_replan_steps must be >= 1 "
                f"(got {replan_steps}, {library_replan_steps})"
            )
        if not 0 <= task_index < N_TASKS:
            raise ValueError(f"risk_features: task_index out of range: {task_index}")
        self._task_index = task_index
        # Task embedding: an 8-dim row looked up from a table. The table is a
        # frozen artifact, not learned in this revision — it is generated once
        # by ``default_task_embedding_table`` and SAVED WITH THE MODEL, so a
        # reload reproduces the exact rows the risk head was fitted against.
        # 8 dims rather than a 10-way one-hot keeps task identity from taking a
        # sixth of the vector. Passing a fitted table here is what a future
        # revision would do without changing the layout.
        table = (
            default_task_embedding_table()
            if task_embedding_table is None
            else task_embedding_table
        )
        if tuple(table.shape) != (N_TASKS, TASK_EMBED_DIM):
            raise ValueError(
                f"risk_features: task embedding table must be "
                f"[{N_TASKS}, {TASK_EMBED_DIM}], got {tuple(table.shape)}"
            )
        self._task_embedding_table = table.detach().float()
        self._task_embedding = self._task_embedding_table[task_index]
        self._action_sigma = (
            None if action_sigma is None else action_sigma.detach().float()
        )
        self._replan = replan_steps
        self._lib_replan = library_replan_steps
        self._t_max = float(t_max)
        self._top_k = top_k

    @property
    def dim(self) -> int:
        return FEATURE_DIM

    @property
    def task_embedding_table(self) -> torch.Tensor:
        """The table in force, for persistence into the model artifact."""
        return self._task_embedding_table

    def build(
        self,
        *,
        results,
        step_features,
        query_keys: dict[str, torch.Tensor],
        view: Any,
        decision_idx: int,
    ) -> torch.Tensor:
        """Assemble the feature vector. Raises on any missing input."""
        parts: list[torch.Tensor] = []

        # f1 — fused top-k, zero-padded on a thin library (f13 carries coverage).
        fused = torch.zeros(self._top_k)
        for i, (_, score) in enumerate(step_features.fused_topk[: self._top_k]):
            fused[i] = float(score)
        parts.append(fused)

        # f2 / f3 — per-field decomposition of the winner and each field's own
        # discriminating power, in canonical order.
        parts.append(self._by_field(step_features.winner_per_field))
        parts.append(torch.tensor([float(step_features.fused_margin)]))
        parts.append(self._by_field(step_features.field_own_margin))

        # f5 / f6 — how far the query sits from the neighbour it would replay.
        delta = self._state_delta(results, query_keys, view)
        parts.append(delta)
        parts.append(torch.linalg.vector_norm(delta).reshape(1))

        # f7 — both sides converted to physical env steps before comparison.
        t_env = decision_idx * self._replan
        neighbour_env = self._neighbour_env_step(results, view)
        parts.append(torch.tensor([
            neighbour_env / self._t_max,
            abs(t_env - neighbour_env) / self._t_max,
        ]))

        # f8 / f9 — the library's own uncertainty about this neighbourhood.
        parts.append(torch.tensor([self._chunk_variance(results, view)]))
        parts.append(torch.tensor([self._same_trajectory_rate(results, view)]))

        # f11 / f12 / f13
        parts.append(self._task_embedding)
        parts.append(torch.tensor([t_env / self._t_max]))
        parts.append(torch.tensor([step_features.n_results / self._top_k]))

        x = torch.cat(parts).to(torch.float32)
        if x.numel() != FEATURE_DIM:
            raise ValueError(
                f"risk_features: built {x.numel()} dims, expected {FEATURE_DIM}"
            )
        return x

    # -- internals ----------------------------------------------------

    def _by_field(self, mapping: dict[str, float]) -> torch.Tensor:
        """Project a per-field dict onto the canonical order.

        A field absent from the mapping scores 0.0: for margins that is the
        honest reading (a field with fewer than two candidates cannot separate
        anything), and it keeps the vector positional.
        """
        return torch.tensor([float(mapping.get(name, 0.0)) for name in CANONICAL_FIELDS])

    def _state_delta(
        self, results, query_keys: dict[str, torch.Tensor], view: Any
    ) -> torch.Tensor:
        query_state = query_keys.get("robot_state")
        if query_state is None:
            raise KeyError("risk_features: query_keys lacks 'robot_state'")
        entry = view.get_entry(results[0].id)
        neighbour_state = entry.query_keys.get("robot_state")
        if neighbour_state is None:
            raise KeyError(
                f"risk_features: neighbour {results[0].id} lacks 'robot_state'"
            )
        delta = (
            query_state.detach().reshape(-1).float()
            - neighbour_state.detach().reshape(-1).float()
        )
        if delta.numel() != ROBOT_STATE_DIM:
            raise ValueError(
                f"risk_features: robot_state delta has {delta.numel()} dims, "
                f"expected {ROBOT_STATE_DIM}"
            )
        return delta

    def _chunk_variance(self, results, view: Any) -> float:
        """Variance across the top-k neighbours' action chunks.

        High similarity scores with disagreeing actions is precisely the case a
        score threshold cannot see: the neighbourhood looks confident and is not.

        Normalisation matters for comparability across steps and joints. The
        chunks are truncated to their common horizon, divided per dimension by
        ``action_sigma`` when supplied, and reduced by mean per-dimension
        variance — so the value does not grow simply because a chunk is longer
        or because one joint has a wider range. Returns 0.0 with fewer than two
        neighbours (nothing to disagree about).
        """
        chunks = []
        for result in results[: self._top_k]:
            payload = view.get(result.id)
            chunk = getattr(payload, "action_chunk", None)
            if chunk is not None:
                chunks.append(chunk.detach().float())
        if len(chunks) < 2:
            return 0.0
        horizon = min(c.shape[0] for c in chunks)
        stacked = torch.stack([c[:horizon] for c in chunks])   # [K, H, D]
        if self._action_sigma is not None:
            stacked = stacked / self._action_sigma.reshape(1, 1, -1).clamp_min(1e-6)
        # Unbiased variance across neighbours, averaged over horizon and dims.
        return float(stacked.var(dim=0, unbiased=True).mean().item())

    def _same_trajectory_rate(self, results, view: Any) -> float:
        """Fraction of top-k neighbours sharing the winner's trajectory.

        Neighbours drawn from one trajectory are consecutive frames, so their
        agreement is one observation rather than k independent ones.
        """
        top = results[: self._top_k]
        if not top:
            return 0.0
        anchor = getattr(view.get_entry(top[0].id), "trajectory_id", None)
        if anchor is None:
            return 0.0
        same = sum(
            1 for r in top
            if getattr(view.get_entry(r.id), "trajectory_id", None) == anchor
        )
        return same / len(top)

    def _neighbour_env_step(self, results, view: Any) -> float:
        entry = view.get_entry(results[0].id)
        step_idx = getattr(entry, "step_idx", None)
        if step_idx is None:
            raise ValueError(
                f"risk_features: neighbour {results[0].id} has no step_idx; the "
                "library artifact predates the phase feature"
            )
        # step_idx counts the LIBRARY episode's inference cycles.
        env_step = float(step_idx) * self._lib_replan
        if env_step > 1.2 * self._t_max:
            # A library_replan_steps that does not match the library would
            # rescale every phase feature; catching it here beats training on
            # silently wrong inputs.
            raise ValueError(
                f"risk_features: neighbour step_idx={step_idx} x "
                f"library_replan_steps={self._lib_replan} = {env_step:.0f} env steps "
                f"exceeds 1.2 x T_max={self._t_max:.0f}; library_replan_steps is "
                "probably wrong for this library"
            )
        return env_step
