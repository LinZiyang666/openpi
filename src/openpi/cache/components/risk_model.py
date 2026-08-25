"""RiskModel — the X15 calibrated cache-risk scorer.

A two-head MLP over the 59-dim A-tier feature vector. The regression head
predicts the teacher/cache action deviation ``u``; the auxiliary binary head
predicts ``d = 1[u > delta]`` and exists only as a training regulariser. What
gets deployed is neither head's raw output but ``isotonic(u_hat)``, a monotone
map fitted on the held-out calibration slice that turns an uncalibrated
regression into a probability the router can threshold. Calibrate first,
threshold second: an isotonic step is the cheapest way to make one frozen tau
mean the same thing across tasks.

Determinism is part of the contract — the same artifact and the same features
must produce the same risk on any machine, so inference runs in eval mode,
under ``no_grad``, on CPU float32, and the isotonic map is a pure lookup.

The artifact carries the feature-schema digest it was trained against and
refuses to load against a mismatched builder: a checkpoint scored through a
different feature layout is silently wrong in exactly the way a fail-safe
cannot catch.

Key dependency: ``risk_features.feature_schema_digest``.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class RiskNet(nn.Module):
    """59 -> 128 -> 128 -> {u, d} two-head MLP."""

    def __init__(self, in_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.u_head = nn.Linear(hidden, 1)
        self.d_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(u_hat, d_logit)``; ``x`` is ``[B, in_dim]``."""
        h = self.trunk(x)
        return self.u_head(h).squeeze(-1), self.d_head(h).squeeze(-1)


class IsotonicMap:
    """Monotone piecewise-linear calibration map, stored as knots.

    Fitted on the calibration slice by the trainer (pool-adjacent-violators);
    at serving time it is a clamped interpolation, which keeps the deployed
    scalar deterministic and dependency-free.
    """

    def __init__(self, x_knots: torch.Tensor, y_knots: torch.Tensor) -> None:
        if x_knots.numel() != y_knots.numel() or x_knots.numel() == 0:
            raise ValueError("isotonic: knot arrays must be non-empty and equal length")
        self._x = x_knots.detach().flatten().float()
        self._y = y_knots.detach().flatten().float()

    def __call__(self, value: float) -> float:
        x, y = self._x, self._y
        if value <= float(x[0]):
            return float(y[0])
        if value >= float(x[-1]):
            return float(y[-1])
        idx = int(torch.searchsorted(x, torch.tensor(float(value))).item())
        idx = max(1, min(idx, x.numel() - 1))
        x0, x1 = float(x[idx - 1]), float(x[idx])
        y0, y1 = float(y[idx - 1]), float(y[idx])
        if x1 == x0:
            return y1
        return y0 + (y1 - y0) * (value - x0) / (x1 - x0)

    def state(self) -> dict[str, torch.Tensor]:
        return {"x": self._x, "y": self._y}


class RiskModel:
    """Deployment wrapper: features in, calibrated risk out."""

    def __init__(
        self,
        net: RiskNet,
        isotonic: Optional[IsotonicMap],
        *,
        feature_schema_sha: str,
        delta: float,
        cp_tau0: Optional[float] = None,
        seed: Optional[int] = None,
        git_sha: str = "",
        task_embedding_table: Optional[torch.Tensor] = None,
    ) -> None:
        self._net = net.eval()
        self._isotonic = isotonic
        self.feature_schema_sha = feature_schema_sha
        self.delta = delta
        # Provenance the plan requires for a recomputable artifact: the
        # conformal starting threshold, the training seed, and the code
        # revision. Carried so a deployed tau can be traced to what produced it.
        self.cp_tau0 = cp_tau0
        self.seed = seed
        self.git_sha = git_sha
        # The exact task-embedding rows the head was fitted against. Stored
        # rather than re-derived: a table the runtime regenerates cannot be
        # audited against these weights.
        self.task_embedding_table = task_embedding_table

    @torch.no_grad()
    def risk(self, x: torch.Tensor) -> float:
        """Calibrated risk for one feature vector.

        Without a calibration map (a raw checkpoint, training-time only) the
        squashed regression output stands in, so the value still lands in
        ``[0, 1]`` and a threshold remains meaningful.
        """
        u_hat, _ = self._net(x.reshape(1, -1).float())
        raw = float(u_hat.item())
        if self._isotonic is None:
            return float(torch.sigmoid(torch.tensor(raw)).item())
        return self._isotonic(raw)

    # -- persistence --------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self._net.state_dict(),
                "in_dim": self._net.trunk[0].in_features,
                "hidden": self._net.trunk[0].out_features,
                "isotonic": None if self._isotonic is None else self._isotonic.state(),
                "feature_schema_sha": self.feature_schema_sha,
                "delta": self.delta,
                "cp_tau0": self.cp_tau0,
                "seed": self.seed,
                "git_sha": self.git_sha,
                "task_embedding_table": self.task_embedding_table,
            },
            path,
        )

    @staticmethod
    def load(path: str, *, expected_schema_sha: str) -> "RiskModel":
        """Load an artifact, refusing a feature-schema mismatch.

        The refusal is deliberate and loud: a model scored through a different
        feature layout produces confident nonsense, which no downstream
        fail-safe can distinguish from a real verdict.
        """
        blob = torch.load(path, map_location="cpu", weights_only=False)
        actual = blob.get("feature_schema_sha")
        if actual != expected_schema_sha:
            raise ValueError(
                f"risk model {path} was trained on feature schema {actual!r} but the "
                f"runtime builder produces {expected_schema_sha!r}; refusing to load"
            )
        net = RiskNet(blob["in_dim"], blob["hidden"])
        net.load_state_dict(blob["state_dict"])
        iso_state = blob.get("isotonic")
        isotonic = None if iso_state is None else IsotonicMap(iso_state["x"], iso_state["y"])
        return RiskModel(
            net, isotonic,
            feature_schema_sha=actual,
            delta=float(blob.get("delta", 0.0)),
            cp_tau0=blob.get("cp_tau0"),
            seed=blob.get("seed"),
            git_sha=blob.get("git_sha", ""),
            task_embedding_table=blob.get("task_embedding_table"),
        )
