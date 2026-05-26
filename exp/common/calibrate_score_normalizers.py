"""Offline calibration of Layer-1 score normalizers (Phase 1).

For each library artifact (keybuilder) and each active field, this script
estimates the *real* query-vs-library raw-similarity distribution using a
leave-one-episode-out (LOEO) scheme — every entry serves as a query and is
scored against all library entries that are NOT in its own trajectory
(``trajectory_id``), which removes self-match and intra-chain near-duplicate
contamination and mirrors serving (the live episode is never in the library).

Candidate monotone normalizers (``score_normalizers.CANDIDATE_NORMALIZERS``,
all bounded to ``[0, 1]``) are fit on the pooled distribution and ranked by a
magnitude-aware diagnostic ``J = mag_sep + beta*intra_spread - lam*sat`` (rank
metrics are invariant under the monotone transforms, so a magnitude metric is
required). The top-2 per field are written as a shortlist; the final method is
chosen by Phase-2 task success, not by ``J`` alone.

Usage:
    uv run exp/common/calibrate_score_normalizers.py \
        --artifact-dir exp/common/data/cache_artifacts/libero_spatial \
        --output exp/weighted_sum/data/calibration_normalizers.json \
        --max-queries 300

Output JSON (keyed by artifact stem so the two CLIP variants do not collide):
    {stem: {"builder_type": .., "vector_dims": {..},
            "fields": {field: {"sim_type": .., "shortlist": [{method, params, J, ...}],
                               "selected": {"method": .., "params": ..}}}}}
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from openpi.cache.components.score_normalizers import (
    SCORE_NORMALIZER_REGISTRY,
    candidate_normalizers_for,
)

logger = logging.getLogger(__name__)

# Fields excluded from calibration: prompt_emb is dropped from the experiment
# (task-constant / near-binary, see plan D7).
_EXCLUDED_FIELDS = frozenset({"prompt_emb"})

# Default field -> similarity type (matches the experiment's YAML convention).
_FIELD_SIM_TYPE = {
    "vision_0": "cosine",
    "vision_1": "cosine",
    "vision_2": "cosine",
    "robot_state": "l2",
}


def _raw_score(query: torch.Tensor, mat: torch.Tensor, sim_type: str) -> torch.Tensor:
    """Raw per-entry score: cosine value (cosine) or L2 distance (l2)."""
    q = query.float().unsqueeze(0)
    if sim_type == "cosine":
        return F.cosine_similarity(q, mat.float())
    return torch.norm(q - mat.float(), p=2, dim=1)


def collect_loeo_scores(
    entries: list,
    field: str,
    sim_type: str,
    *,
    max_queries: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect LOEO query-vs-library raw scores, split into same/cross task.

    Returns (same_task_scores, cross_task_scores) as 1-D float arrays.
    """
    idx = [i for i, e in enumerate(entries) if field in e.query_keys]
    if len(idx) < 2:
        return np.array([]), np.array([])

    # query_keys may be torch.Tensor (CLIP artifacts) or numpy.ndarray (cp1_*
    # artifacts store numpy); as_tensor coerces both to a float32 tensor.
    vecs = torch.stack(
        [torch.as_tensor(entries[i].query_keys[field], dtype=torch.float32) for i in idx]
    )  # [M, D]
    traj = np.array([getattr(entries[i], "trajectory_id", None) for i in idx], dtype=object)
    task = np.array([entries[i].payload.task_key for i in idx], dtype=object)

    rng = np.random.default_rng(seed)
    q_local = np.arange(len(idx))
    if len(idx) > max_queries:
        q_local = rng.choice(len(idx), size=max_queries, replace=False)

    same: list[np.ndarray] = []
    cross: list[np.ndarray] = []
    for qi in q_local:
        keep = traj != traj[qi]  # leave-one-episode-out: drop own trajectory
        if not keep.any():
            continue
        scores = _raw_score(vecs[qi], vecs[keep], sim_type).numpy()
        same_mask = task[keep] == task[qi]
        same.append(scores[same_mask])
        cross.append(scores[~same_mask])

    same_arr = np.concatenate(same) if same else np.array([])
    cross_arr = np.concatenate(cross) if cross else np.array([])
    return same_arr, cross_arr


def _diagnostics(norm, same: np.ndarray, cross: np.ndarray) -> dict[str, float]:
    ns = norm(torch.from_numpy(same.astype(np.float64))).numpy() if same.size else np.array([])
    nc = norm(torch.from_numpy(cross.astype(np.float64))).numpy() if cross.size else np.array([])
    allv = np.concatenate([ns, nc]) if (ns.size or nc.size) else np.array([0.5])
    mag_sep = float(ns.mean() - nc.mean()) if (ns.size and nc.size) else 0.0
    intra = 0.5 * (float(ns.std()) if ns.size else 0.0) + 0.5 * (float(nc.std()) if nc.size else 0.0)
    sat = float(np.mean(np.isclose(allv, 0.0) | np.isclose(allv, 1.0)))
    return {"mag_sep": mag_sep, "intra_spread": intra, "sat": sat}


def select_shortlist(
    same: np.ndarray,
    cross: np.ndarray,
    sim_type: str,
    *,
    beta: float,
    lam: float,
    top_n: int = 2,
) -> list[dict]:
    """Fit each compatible candidate, rank by J, return the top-N shortlist."""
    # Guard against non-finite scores propagating into percentile/moment fits.
    same = same[np.isfinite(same)] if same.size else same
    cross = cross[np.isfinite(cross)] if cross.size else cross
    pooled = np.concatenate([same, cross]) if (same.size or cross.size) else np.array([])
    if pooled.size < 2:
        return []
    ranked = []
    for name in candidate_normalizers_for(sim_type):
        cls = SCORE_NORMALIZER_REGISTRY[name]
        try:
            norm = cls.fit_from_scores(pooled, sim_type)
        except Exception as exc:  # noqa: BLE001 - skip a candidate that cannot fit
            logger.warning("fit failed for %s: %s", name, exc)
            continue
        diag = _diagnostics(norm, same, cross)
        j = diag["mag_sep"] + beta * diag["intra_spread"] - lam * diag["sat"]
        ranked.append({"method": name, "params": norm.params_to_dict(), "J": float(j), **diag})
    ranked.sort(key=lambda r: r["J"], reverse=True)
    return ranked[:top_n]


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Calibrate Layer-1 score normalizers (Phase 1)")
    ap.add_argument("--artifact-dir", required=True, help="Directory with library .pkl artifacts")
    ap.add_argument("--output", required=True, help="Output calibration JSON path")
    ap.add_argument("--max-queries", type=int, default=300, help="LOEO query subsample per field")
    ap.add_argument("--beta", type=float, default=0.5, help="intra_spread weight in J")
    ap.add_argument("--lam", type=float, default=0.5, help="saturation penalty weight in J")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pkls = sorted(Path(args.artifact_dir).glob("*.pkl"))
    pkls = [p for p in pkls if ".bak" not in p.name]
    if not pkls:
        raise SystemExit(f"No .pkl artifacts in {args.artifact_dir}")

    result: dict[str, dict] = {}
    for pkl in pkls:
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        entries = data.get("entries", [])
        if not entries:
            logger.warning("%s: no entries, skipping", pkl.name)
            continue
        # LOEO requires trajectory grouping; fail fast if absent.
        if any(getattr(e, "trajectory_id", None) is None for e in entries):
            raise SystemExit(
                f"{pkl.name}: entries missing trajectory_id; cannot run LOEO calibration"
            )
        builder = data.get("key_builder_type") or data.get("builder_type") or pkl.stem
        vector_dims = data.get("vector_dims", {})
        fields = [f for f in vector_dims if f not in _EXCLUDED_FIELDS]

        logger.info("%s (builder=%s): calibrating fields %s", pkl.name, builder, fields)
        builder_out: dict[str, dict] = {}
        for field in fields:
            sim_type = _FIELD_SIM_TYPE.get(field, "cosine")
            same, cross = collect_loeo_scores(
                entries, field, sim_type, max_queries=args.max_queries, seed=args.seed,
            )
            shortlist = select_shortlist(same, cross, sim_type, beta=args.beta, lam=args.lam)
            if not shortlist:
                logger.warning("  %s: insufficient data, skipping", field)
                continue
            best = shortlist[0]
            builder_out[field] = {
                "sim_type": sim_type,
                "shortlist": shortlist,
                "selected": {"method": best["method"], "params": best["params"]},
            }
            logger.info(
                "  %s (%s): best=%s J=%.4f mag_sep=%.4f intra=%.4f sat=%.4f",
                field, sim_type, best["method"], best["J"],
                best["mag_sep"], best["intra_spread"], best["sat"],
            )
        # Key by artifact stem: builder_type "clip" is shared by both CLIP
        # variants, so it cannot disambiguate them; the stem can.
        result[pkl.stem] = {
            "builder_type": builder,
            "vector_dims": vector_dims,
            "fields": builder_out,
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    logger.info("Calibration written to %s", out)


if __name__ == "__main__":
    main()
