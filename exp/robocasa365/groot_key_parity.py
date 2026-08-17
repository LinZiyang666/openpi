"""The G0-D2 gate: do online and offline keys agree closely enough to retrieve the same entry?

Two paths produce a cache key for the same step. Online, stage 1 hands bf16
activations straight to the pooling. Offline, the collector stores fp16 in
HDF5 and the artifact builder upcasts to fp32 before pooling. They cannot be
bit-identical — bf16 has eight exponent bits and seven mantissa bits, fp16 has
five and ten, so the round trip is lossy by construction. The structural
question (does mask-slicing equal offset-slicing?) is answered separately, in
fp32, where equality *is* exact.

What remains is whether the quantisation is small enough to be irrelevant, and
"relative error below a threshold" is a weak way to ask that. Two refinements:

* **Per-field metric.** The backend scores `vision_*` and `prompt_emb` by
  cosine and `robot_state` by L2. A margin measured in raw L2 says nothing
  about a cosine ranking, so each field is checked under the metric that will
  actually rank it. (The L2-to-similarity `tau` is a monotone transform and
  cannot change a within-field ranking, so it is irrelevant here.)

* **Degenerate fields are excluded, not fudged.** Within one episode the task
  text never changes, so every step's `prompt_emb` key is identical. "Distance
  to the nearest other step" is zero there — not a small number, a
  structurally meaningless one. Such fields keep the error gate and are
  reported by name rather than being silently averaged away.

The gate is the retrieval question stated directly: querying the offline
library with the online key must still return that same step, with a positive
margin. Thresholds are pre-registered in the plan; a run that misses them is a
finding, not a reason to widen them.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Mapping, Sequence

import torch

REL_ERROR_TOL = 1e-3
REQUIRED_NON_DEGENERATE = ("vision_0", "robot_state")


@dataclass
class FieldParity:
    """Per-field outcome. ``rank_ok`` / ``min_margin`` are None for degenerate fields."""

    field: str
    max_rel_error: float
    degenerate: bool
    rank_ok: bool | None = None
    min_margin: float | None = None
    first_rank_failure: int | None = None


@dataclass
class ParityReport:
    fields: dict[str, FieldParity] = dataclass_field(default_factory=dict)
    n_steps: int = 0

    @property
    def degenerate_fields(self) -> list[str]:
        return sorted(f for f, r in self.fields.items() if r.degenerate)

    @property
    def error_gate_passed(self) -> bool:
        return all(r.max_rel_error <= REL_ERROR_TOL for r in self.fields.values())

    @property
    def rank_gate_passed(self) -> bool:
        judged = [r for r in self.fields.values() if not r.degenerate]
        return bool(judged) and all(r.rank_ok for r in judged)

    @property
    def required_fields_judged(self) -> bool:
        """Required fields must be non-degenerate: an episode where they are not is uninformative."""
        return all(
            name in self.fields and not self.fields[name].degenerate
            for name in REQUIRED_NON_DEGENERATE
        )

    @property
    def passed(self) -> bool:
        return (
            self.error_gate_passed
            and self.required_fields_judged
            and self.rank_gate_passed
        )

    def summary(self) -> str:
        lines = [f"G0-D2 over {self.n_steps} steps: {'PASS' if self.passed else 'FAIL'}"]
        for name in sorted(self.fields):
            r = self.fields[name]
            if r.degenerate:
                lines.append(
                    f"  {name:12s} rel_err={r.max_rel_error:.3e}  "
                    "DEGENERATE (constant within the episode; rank gate not applicable)"
                )
            else:
                lines.append(
                    f"  {name:12s} rel_err={r.max_rel_error:.3e}  "
                    f"rank_ok={r.rank_ok}  min_margin={r.min_margin:.3e}"
                    + (
                        ""
                        if r.first_rank_failure is None
                        else f"  first failure at step {r.first_rank_failure}"
                    )
                )
        if not self.required_fields_judged:
            missing = [
                n
                for n in REQUIRED_NON_DEGENERATE
                if n not in self.fields or self.fields[n].degenerate
            ]
            lines.append(
                f"  !! {missing} degenerate or absent: this episode cannot answer the "
                "gate. Re-run on an episode with movement rather than relaxing it."
            )
        return "\n".join(lines)


def _similarity(query: torch.Tensor, library: torch.Tensor, metric: str) -> torch.Tensor:
    """Score one query against every library row, higher is better."""
    if metric == "cosine":
        return torch.nn.functional.cosine_similarity(query.unsqueeze(0), library, dim=1)
    if metric == "l2":
        # Negated distance: monotone in the configured exp(-d/tau) transform, so
        # the ranking is the same and no tau has to be guessed here.
        return -torch.linalg.vector_norm(library - query.unsqueeze(0), dim=1)
    raise ValueError(f"unknown similarity metric {metric!r}")


def _is_degenerate(library: torch.Tensor) -> bool:
    """True when two steps share a bit-identical key, making 'nearest other' meaningless."""
    return len(torch.unique(library, dim=0)) < library.shape[0]


def check_key_parity(
    online: Sequence[Mapping[str, torch.Tensor]],
    offline: Sequence[Mapping[str, torch.Tensor]],
    metrics: Mapping[str, str],
) -> ParityReport:
    """Run the (a) error gate and (b) rank-preservation gate over one episode.

    Args:
        online: per-step key dicts from the live path.
        offline: per-step key dicts rebuilt from the artifact pipeline, same order.
        metrics: field name -> "cosine" | "l2", matching the YAML's
            ``field_similarity``.

    Returns:
        A ParityReport; ``passed`` is the gate verdict.
    """
    if len(online) != len(offline):
        raise ValueError(f"{len(online)} online steps vs {len(offline)} offline")
    if not online:
        raise ValueError("no steps to compare")

    report = ParityReport(n_steps=len(online))
    names = sorted(set(online[0]) & set(offline[0]))

    for name in names:
        metric = metrics.get(name, "cosine")
        on_stack = torch.stack([step[name].float() for step in online])
        off_stack = torch.stack([step[name].float() for step in offline])

        diff = torch.linalg.vector_norm(on_stack - off_stack, dim=1)
        norms = torch.linalg.vector_norm(off_stack, dim=1)
        # A zero-norm reference has no relative scale; fall back to the absolute
        # difference rather than dividing.
        rel = torch.where(norms > 0, diff / norms.clamp(min=1e-30), diff)
        result = FieldParity(
            field=name,
            max_rel_error=float(rel.max()),
            degenerate=_is_degenerate(off_stack),
        )

        if not result.degenerate:
            rank_ok = True
            margins = []
            for t in range(len(online)):
                scores = _similarity(on_stack[t], off_stack, metric)
                self_score = scores[t].clone()
                others = torch.cat([scores[:t], scores[t + 1 :]])
                margins.append(float(self_score - others.max()))
                if int(scores.argmax()) != t:
                    rank_ok = False
                    if result.first_rank_failure is None:
                        result.first_rank_failure = t
            result.rank_ok = rank_ok and min(margins) > 0.0
            result.min_margin = min(margins)

        report.fields[name] = result

    return report
