#!/usr/bin/env python3
"""Fit the X15 cache-risk model from shadow-teacher labels.

    python train_risk_model.py --features feats.jsonl --ledger x15_init_ledger.json \
        --out risk.pt [--epochs 200] [--seed 0]

Supervised, not RL: every decision carries its own label (the teacher/cache
action deviation ``u`` recorded by ``ShadowTeacherRecorder``), which is what
replaces X14's one-bit-per-episode credit assignment.

The pipeline is deliberately split three ways by init, and the split is
enforced from the ledger rather than trusted:

  * **gradient**  — the only slice that produces weight updates.
  * **delta**     — picks the ``u > delta`` binarisation threshold (Youden J on
                    episode outcome) and does model selection / early stopping.
  * **cal**       — fits the isotonic map and, later, the tau grid.

``test`` never appears here at all; a ledger that routes a test init into any
fitting slice is a hard error. Selecting a threshold on the same data that
reports the result is the failure this structure exists to prevent.

Key dependencies: ``openpi.cache.components.risk_model`` (net + artifact) and
``openpi.cache.components.risk_features`` (schema digest).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import subprocess
from typing import Iterable, Optional

import numpy as np
import torch
from torch import nn

from openpi.cache.components.risk_features import (
    FEATURE_DIM,
    default_task_embedding_table,
    feature_schema_digest,
)
from openpi.cache.components.risk_model import IsotonicMap, RiskModel, RiskNet

FIT_SLICES = ("gradient", "delta", "cal")
FORBIDDEN_SLICE = "test"


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------


def load_rows(path: str | pathlib.Path) -> list[dict]:
    """Read the joined feature/label rows written by the offline pipeline."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def assert_pools_are_disjoint(ledger: dict) -> None:
    """Refuse a ledger whose fitting slices overlap or touch the test pool.

    Cheap to check, catastrophic to miss: a single shared init silently turns
    the held-out estimate into an in-sample one.
    """
    slices = {name: set(ledger.get(name, [])) for name in FIT_SLICES}
    test = set(ledger.get(FORBIDDEN_SLICE, []))

    for name, inits in slices.items():
        overlap = inits & test
        if overlap:
            raise ValueError(
                f"ledger: fitting slice {name!r} contains {len(overlap)} init(s) "
                f"from the {FORBIDDEN_SLICE!r} pool, e.g. {sorted(overlap)[:3]}; "
                "the test pool must never be fitted on"
            )
    names = list(slices)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = slices[a] & slices[b]
            if overlap:
                raise ValueError(
                    f"ledger: slices {a!r} and {b!r} share {len(overlap)} init(s), "
                    f"e.g. {sorted(overlap)[:3]}; threshold selection and "
                    "calibration must not see gradient data"
                )


def split_rows(rows: Iterable[dict], ledger: dict) -> dict[str, list[dict]]:
    """Route rows to slices by their init id, dropping anything unassigned."""
    lookup: dict[str, str] = {}
    for name in FIT_SLICES:
        for init in ledger.get(name, []):
            lookup[str(init)] = name
    out: dict[str, list[dict]] = {name: [] for name in FIT_SLICES}
    for row in rows:
        slice_name = lookup.get(str(row["init_id"]))
        if slice_name is not None:
            out[slice_name].append(row)
    return out


def to_tensors(rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack rows into ``(features, u, success)``."""
    if not rows:
        raise ValueError("train_risk_model: empty slice")
    x = torch.tensor([r["features"] for r in rows], dtype=torch.float32)
    if x.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"features have {x.shape[1]} dims, runtime builder produces {FEATURE_DIM}"
        )
    u = torch.tensor([r["u"] for r in rows], dtype=torch.float32)
    success = torch.tensor([bool(r.get("success", True)) for r in rows], dtype=torch.bool)
    return x, u, success


# ------------------------------------------------------------------
# Threshold and calibration
# ------------------------------------------------------------------


def select_delta(u: torch.Tensor, success: torch.Tensor) -> float:
    """Pick the ``u`` cut that best separates failed from successful episodes.

    Youden's J over the observed u values. This is model selection, so it runs
    on the delta slice only.
    """
    failed = ~success
    if not failed.any() or not success.any():
        # Degenerate slice: fall back to the median so the binary head still
        # trains on a balanced-ish target rather than a constant.
        return float(u.median().item())
    best_j, best_delta = -1.0, float(u.median().item())
    for candidate in torch.unique(u):
        predicted = u > candidate
        tpr = float((predicted & failed).sum()) / float(failed.sum())
        fpr = float((predicted & success).sum()) / float(success.sum())
        j = tpr - fpr
        if j > best_j:
            best_j, best_delta = j, float(candidate.item())
    return best_delta


def split_conformal_tau0(
    risks: torch.Tensor, success: torch.Tensor, *, alpha: float = 0.1
) -> Optional[float]:
    """Split-conformal starting point for the tau grid.

    Nonconformity is the calibrated risk observed on steps of SUCCESSFUL
    episodes: those are the steps where replaying the cache turned out fine, so
    the ``1 - alpha`` quantile of their risk is the level below which the gate
    would have left a successful episode alone.

    This is only the centre of the tau grid, never a guarantee — intervening on
    the policy changes the very distribution the calibration was drawn from, so
    exchangeability does not hold. Returns None when the slice contains no
    successful steps to calibrate against.
    """
    good = risks[success]
    if good.numel() == 0:
        return None
    return float(torch.quantile(good, 1.0 - alpha).item())


def fit_isotonic(u_hat: torch.Tensor, target: torch.Tensor) -> IsotonicMap:
    """Pool-adjacent-violators fit of ``u_hat -> P(target)``.

    Monotone by construction, which is what lets a single tau keep one meaning
    across tasks.
    """
    order = torch.argsort(u_hat)
    xs = u_hat[order].tolist()
    ys = target[order].float().tolist()

    # PAVA: merge blocks until means are non-decreasing.
    values: list[float] = []
    weights: list[float] = []
    for y in ys:
        values.append(y)
        weights.append(1.0)
        while len(values) > 1 and values[-2] > values[-1]:
            w = weights[-2] + weights[-1]
            v = (values[-2] * weights[-2] + values[-1] * weights[-1]) / w
            values[-2:] = [v]
            weights[-2:] = [w]

    fitted: list[float] = []
    for v, w in zip(values, weights):
        fitted.extend([v] * int(w))
    return IsotonicMap(torch.tensor(xs), torch.tensor(fitted))


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------


def _git_sha() -> str:
    """Current revision, empty when it cannot be determined."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return ""


def train(
    slices: dict[str, list[dict]],
    *,
    epochs: int = 200,
    hidden: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
    alpha: float = 0.1,
) -> RiskModel:
    """Fit the two-head net, then calibrate. Deterministic given ``seed``."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    x_tr, u_tr, _ = to_tensors(slices["gradient"])
    x_de, u_de, s_de = to_tensors(slices["delta"])
    x_cal, u_cal, s_cal = to_tensors(slices["cal"])

    delta = select_delta(u_de, s_de)
    d_tr = (u_tr > delta).float()
    d_de = (u_de > delta).float()

    net = RiskNet(FEATURE_DIM, hidden=hidden)
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)
    huber, bce = nn.HuberLoss(delta=1.0), nn.BCEWithLogitsLoss()

    best_state, best_loss = None, float("inf")
    for _ in range(epochs):
        net.train()
        optimiser.zero_grad()
        u_hat, d_logit = net(x_tr)
        # The binary head is an auxiliary regulariser at half weight; only the
        # regression output is deployed.
        loss = huber(u_hat, u_tr) + 0.5 * bce(d_logit, d_tr)
        loss.backward()
        optimiser.step()

        # Early stopping on the delta slice — never on gradient or cal.
        net.eval()
        with torch.no_grad():
            u_val, d_val = net(x_de)
            val_loss = float(huber(u_val, u_de) + 0.5 * bce(d_val, d_de))
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

    if best_state is not None:
        net.load_state_dict(best_state)

    net.eval()
    with torch.no_grad():
        u_hat_cal, _ = net(x_cal)
    isotonic = fit_isotonic(u_hat_cal, (u_cal > delta))

    # tau grid centre, computed on the SAME calibration slice and through the
    # deployed scalar (isotonic output), so the number the grid is built around
    # is the number the gate will actually threshold.
    calibrated = torch.tensor([isotonic(float(v)) for v in u_hat_cal])
    cp_tau0 = split_conformal_tau0(calibrated, s_cal, alpha=alpha)

    return RiskModel(
        net, isotonic,
        feature_schema_sha=feature_schema_digest(),
        delta=delta,
        cp_tau0=cp_tau0,
        seed=seed,
        git_sha=_git_sha(),
        task_embedding_table=default_task_embedding_table(),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--features", required=True, help="joined feature/label jsonl")
    ap.add_argument("--ledger", required=True, help="init ledger with the pool split")
    ap.add_argument("--out", required=True, help="artifact path")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.1,
                    help="split-conformal miscoverage for the tau grid centre")
    args = ap.parse_args()

    ledger = json.loads(pathlib.Path(args.ledger).read_text(encoding="utf-8"))
    assert_pools_are_disjoint(ledger)

    slices = split_rows(load_rows(args.features), ledger)
    for name in FIT_SLICES:
        print(f"{name:>9}: {len(slices[name])} rows")

    model = train(
        slices, epochs=args.epochs, hidden=args.hidden, lr=args.lr,
        seed=args.seed, alpha=args.alpha,
    )
    model.save(args.out)
    print(f"delta = {model.delta:.6f}")
    print(f"cp_tau0 = {model.cp_tau0}")
    print(f"schema = {model.feature_schema_sha}")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
