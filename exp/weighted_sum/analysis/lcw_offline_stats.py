"""Zero-cost modality-weight priors from library statistics, read against the closed-loop landscape.

The served fusion is ``S(c) = sum_f w_f * n_f(sim_f(q, c))`` over the task-scoped
library, with ``n_f`` the per-field zscore+tanh normalizer whose ``mu``/``sigma``
live in the served yaml. Everything here is computed from the library alone
(leave-one-trajectory-out, same-task candidates only, exactly the population
the online search ranks) and then *read* against the closed-loop cells that
the grid search already paid for:

* per-field statistics that could serve as zero-cost weight priors — the
  within-task spread of the normalized score (how much of the vote a field
  actually casts among contending candidates), Fisher separability of
  phase-aligned vs phase-misaligned candidates, field-only top-1 phase error,
  and lockstep coherence of consecutive retrievals;
* offline objectives evaluated over the whole simplex and at every closed-loop
  cell, so each objective gets a rank correlation with closed-loop success
  across cells rather than a single "does the winner rank first" reading;
* for every candidate prior: the nearest closed-loop cell, its success rate,
  the kernel-smoothed success rate at the prior, and the gap to the leader.

Nothing under ``src/`` is touched; the production strategy is only used for a
parity check of the normalized scores.

Usage:
  python exp/weighted_sum/analysis/lcw_offline_stats.py \
      --library <pkl> --template <served yaml> --cells <cells json> \
      --output <dir> [--J a,b,c] [--served a,b,c] [--parity 5] [--delta 0.1]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import time

import numpy as np
import torch
import yaml

FIELDS = ("vision_0", "vision_1", "robot_state")


# ------------------------------------------------------------------
# Library loading and normalized score matrices
# ------------------------------------------------------------------
def load_entries(pkl: pathlib.Path):
    with pkl.open("rb") as f:
        art = pickle.load(f)
    return art["entries"], art


def normalized_scores(keys: np.ndarray, sim_type: str, mu: float, sigma: float, device) -> np.ndarray:
    """Return the n x n matrix of served normalized scores for one field."""
    k = torch.as_tensor(keys, dtype=torch.float32, device=device)
    if sim_type == "cosine":
        kn = torch.nn.functional.normalize(k, dim=1)
        x = kn @ kn.T
    elif sim_type == "l2":
        x = -torch.cdist(k, k)
    else:
        raise ValueError(sim_type)
    s = sigma if sigma > 1e-12 else 1.0
    n = 0.5 * (torch.tanh((x - mu) / s) + 1.0)
    return n.float().cpu().numpy()


# ------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------
def simplex(step: int):
    for a in range(step + 1):
        for b in range(step + 1 - a):
            yield (a / step, b / step, (step - a - b) / step)


def normalize_w(v) -> list[float]:
    v = np.asarray(v, dtype=np.float64)
    v = np.clip(v, 0.0, None)
    return (v / v.sum()).tolist()


def spearman(a, b) -> float:
    from scipy.stats import spearmanr

    r = spearmanr(a, b).correlation
    return float(r) if np.isfinite(r) else float("nan")


def l1(w1, w2) -> float:
    return float(np.abs(np.asarray(w1) - np.asarray(w2)).sum())


def nearest_cell(w, cells):
    best = min(cells, key=lambda c: l1(w, c["w"]))
    return best, l1(w, best["w"])


def smoothed_sr(w, cells, bandwidth: float) -> float:
    d = np.array([l1(w, c["w"]) for c in cells])
    k = np.exp(-((d / bandwidth) ** 2)) * np.array([c["n"] for c in cells])
    return float(np.sum(k * np.array([c["sr"] for c in cells])) / np.sum(k))


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--library", type=pathlib.Path, required=True)
    ap.add_argument("--template", type=pathlib.Path, required=True)
    ap.add_argument("--cells", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--J", default="", help="Phase-1 J per field v0,v1,rs (memo baseline)")
    ap.add_argument("--served", default="", help="closed-loop winner v0,v1,rs for reference")
    ap.add_argument("--parity", type=int, default=5, help="queries checked against the production stack")
    ap.add_argument("--delta", type=float, default=0.1, help="phase window for the positive label")
    ap.add_argument("--horizon", type=int, default=5, help="executed action steps (replan length)")
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--action-dims", type=int, default=7,
                    help="leading executed action dims (LIBERO: 7; padding dims are excluded); 0 = detect by variance")
    ap.add_argument("--bandwidth", type=float, default=0.15)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()

    entries, art = load_entries(args.library)
    n = len(entries)
    tpl = yaml.safe_load(args.template.read_text())
    ss = tpl["checkpoints"]["cp1"]["search_strategy"]
    sim_types = {f: ss["field_similarity"][f]["type"] for f in FIELDS}
    norm = {f: ss["score_normalization"]["fields"][f]["params"] for f in FIELDS}
    served_yaml = [tpl["keys"][f]["weight"] for f in FIELDS]

    task_names = sorted({e.payload.task_key for e in entries})
    task = np.array([task_names.index(e.payload.task_key) for e in entries])
    traj_names = sorted({e.trajectory_id for e in entries})
    traj = np.array([traj_names.index(e.trajectory_id) for e in entries])
    step = np.array([int(e.step_idx) for e in entries])
    traj_len = np.zeros(len(traj_names), dtype=int)
    for t in range(len(traj_names)):
        s = np.sort(step[traj == t])
        if not np.array_equal(s, np.arange(len(s))):
            raise SystemExit(f"trajectory {traj_names[t]} steps are not contiguous: {s[:5]}")
        traj_len[t] = len(s)
    length = traj_len[traj]
    progress = step / np.maximum(length - 1, 1)
    remaining = (length - 1 - step).astype(float)

    actions = np.stack([np.asarray(e.payload.action_chunk, dtype=np.float32) for e in entries])
    if args.action_dims > 0:
        live = np.arange(min(args.action_dims, actions.shape[-1]))
    else:
        live = np.flatnonzero(actions.reshape(-1, actions.shape[-1]).std(axis=0) > 1e-2)
    h = min(args.horizon, actions.shape[1])
    act = actions[:, :h, :][:, :, live].reshape(n, -1)
    grip = actions[:, :h, live[-1]].mean(axis=1)

    N = {}
    for f in FIELDS:
        keys = np.stack([np.asarray(e.query_keys[f], dtype=np.float32) for e in entries])
        N[f] = normalized_scores(keys, sim_types[f], norm[f]["mu"], norm[f]["sigma"], device)
    M = (task[:, None] == task[None, :]) & (traj[:, None] != traj[None, :])
    queries = np.flatnonzero(M.any(axis=1))
    print(f"{n} entries, {len(traj_names)} trajectories, {len(task_names)} tasks, "
          f"{len(queries)} queries, live action dims {live.tolist()}, "
          f"scores in {time.perf_counter() - t0:.0f}s on {device}", flush=True)

    # ------------------------------------------------------------------
    # Parity check against the production strategy (one-hot weights)
    # ------------------------------------------------------------------
    parity = None
    if args.parity > 0:
        from openpi.cache.backends.in_memory_backend import InMemoryBackend
        from openpi.cache.cache_storage import CacheStorage
        from openpi.cache.components.search_strategy import SearchContext
        from openpi.cache.config import _build_search_strategy, _keys_iter, load_cache_config
        from openpi.cache.types import CheckpointID

        raw = dict(tpl)
        raw["backend"]["in_memory"]["preload_path"] = str(args.library)
        used = args.output / "config_used.yaml"
        used.write_text(yaml.safe_dump(raw, sort_keys=False))
        cfg = load_cache_config(str(used))
        enabled = [name for name, key in _keys_iter(cfg.keys) if key.enabled]
        for e in entries:
            e.query_keys = {k: torch.as_tensor(v).float().contiguous() for k, v in e.query_keys.items()}
            e.payload.action_chunk = torch.as_tensor(e.payload.action_chunk).float().contiguous()
        backend = InMemoryBackend(vector_dims=cfg.backend.vector_dims)
        storage = CacheStorage(backend)
        storage.batch_insert(entries)
        if hasattr(backend, "freeze"):
            backend.freeze()
        ids = {e.id: i for i, e in enumerate(entries)}
        rng = np.random.default_rng(0)
        picks = rng.choice(queries, size=min(args.parity, len(queries)), replace=False)
        worst = 0.0
        for f in FIELDS:
            one_hot = {name: (1.0 if name == f else 0.0) for name in enabled}
            strategy = _build_search_strategy(cfg.checkpoints["cp1"].search_strategy, storage, one_hot, min_top_k_hint=n)
            for i in picks:
                e = entries[i]
                ctx = SearchContext({k: e.query_keys[k] for k in enabled}, CheckpointID.CP1, e.step_idx, e.payload.task_key)
                for hit in strategy.search(ctx):
                    worst = max(worst, abs(hit.score - float(N[f][i, ids[hit.id]])))
        parity = worst
        print(f"parity vs production stack: max |diff| = {worst:.2e}", flush=True)

    # ------------------------------------------------------------------
    # Per-field statistics
    # ------------------------------------------------------------------
    stats = {f: {} for f in FIELDS}
    nxt = {}
    for i in queries:
        j = np.flatnonzero((traj == traj[i]) & (step == step[i] + 1))
        if j.size:
            nxt[i] = int(j[0])
    for f in FIELDS:
        S = N[f]
        spread_all, spread_top, d_delta, var_delta, d_nn, var_nn = [], [], [], [], [], []
        top1 = np.full(n, -1)
        for i in queries:
            cand = np.flatnonzero(M[i])
            s = S[i, cand]
            spread_all.append(s.std())
            spread_top.append(np.sort(s)[-10:].std() if s.size >= 3 else s.std())
            top1[i] = cand[np.argmax(s)]
            pos = np.abs(progress[cand] - progress[i]) <= args.delta
            if pos.any() and (~pos).any():
                d_delta.append(s[pos].mean() - s[~pos].mean())
                var_delta.append(0.5 * (s[pos].var() + s[~pos].var()))
            nn = np.zeros(cand.size, dtype=bool)
            for t in np.unique(traj[cand]):
                sub = np.flatnonzero(traj[cand] == t)
                nn[sub[np.argmin(np.abs(progress[cand[sub]] - progress[i]))]] = True
            if nn.any() and (~nn).any():
                d_nn.append(s[nn].mean() - s[~nn].mean())
                var_nn.append(0.5 * (s[nn].var() + s[~nn].var()))
        q = queries
        j = top1[q]
        phase_err = np.abs(progress[j] - progress[q])
        rem_err = np.abs(remaining[j] - remaining[q])
        act_l2 = np.sqrt(((act[j] - act[q]) ** 2).sum(axis=1))
        flip = (np.sign(grip[j]) * np.sign(grip[q])) < 0
        lock, same = [], []
        for i, i2 in nxt.items():
            a, b = top1[i], top1[i2]
            same.append(traj[a] == traj[b])
            lock.append(traj[a] == traj[b] and step[b] == step[a] + 1)
        stats[f] = {
            "spread_all": float(np.mean(spread_all)),
            "spread_top10": float(np.mean(spread_top)),
            "fisher_delta_d": float(np.mean(d_delta)),
            "fisher_delta_var": float(np.mean(var_delta)),
            "fisher_nn_d": float(np.mean(d_nn)),
            "fisher_nn_var": float(np.mean(var_nn)),
            "top1_phase_err": float(phase_err.mean()),
            "top1_phase_err_p90": float(np.quantile(phase_err, 0.9)),
            "top1_phase_gt_0.2": float((phase_err > 0.2).mean()),
            "top1_phase_gt_0.3": float((phase_err > 0.3).mean()),
            "top1_remaining_err": float(rem_err.mean()),
            "top1_action_l2": float(act_l2.mean()),
            "top1_grip_flip": float(flip.mean()),
            "lockstep": float(np.mean(lock)),
            "same_traj": float(np.mean(same)),
        }
    print("per-field stats:")
    for k in stats[FIELDS[0]]:
        print(f"  {k:22s} " + "  ".join(f"{f}={stats[f][k]:.4f}" for f in FIELDS))

    # ------------------------------------------------------------------
    # Objectives at any weight vector
    # ------------------------------------------------------------------
    Nst = np.stack([N[f] for f in FIELDS]).astype(np.float32)
    penalty = np.where(M, 0.0, -np.inf).astype(np.float32)
    q = queries
    nxt_from = np.array(list(nxt.keys()))
    nxt_to = np.array(list(nxt.values()))

    def objectives(w) -> dict[str, float]:
        F = np.tensordot(np.asarray(w, dtype=np.float32), Nst, axes=1) + penalty
        top = np.argmax(F, axis=1)
        j = top[q]
        phase_err = np.abs(progress[j] - progress[q])
        out = {
            "phase_err": float(phase_err.mean()),
            "phase_err_p90": float(np.quantile(phase_err, 0.9)),
            "phase_gt_0.1": float((phase_err > 0.1).mean()),
            "phase_gt_0.2": float((phase_err > 0.2).mean()),
            "phase_gt_0.3": float((phase_err > 0.3).mean()),
            "phase_clip_0.2": float(np.minimum(phase_err / 0.2, 1.0).mean()),
            "remaining_err": float(np.abs(remaining[j] - remaining[q]).mean()),
            "action_l2": float(np.sqrt(((act[j] - act[q]) ** 2).sum(axis=1)).mean()),
            "grip_flip": float(((np.sign(grip[j]) * np.sign(grip[q])) < 0).mean()),
        }
        a, b = top[nxt_from], top[nxt_to]
        out["lockstep"] = float(np.mean((traj[a] == traj[b]) & (step[b] == step[a] + 1)))
        out["same_traj"] = float(np.mean(traj[a] == traj[b]))
        return out

    cells = json.loads(args.cells.read_text())["cells"]
    leader = max(cells, key=lambda c: c["sr"])
    for c in cells:
        c["smoothed"] = smoothed_sr(c["w"], cells, args.bandwidth)
    smooth_leader = max(cells, key=lambda c: c["smoothed"])
    print(f"smoothed landscape: max {smooth_leader['smoothed']:.3f} at w={np.round(smooth_leader['w'], 3).tolist()} "
          f"(cell sr {smooth_leader['sr']:.3f}); leader cell smoothed {smoothed_sr(leader['w'], cells, args.bandwidth):.3f}")
    grid = [dict(w=list(w), **objectives(w)) for w in simplex(args.grid)]
    at_cells = [dict(w=c["w"], sr=c["sr"], n=c["n"], **objectives(c["w"])) for c in cells]
    obj_names = [k for k in grid[0] if k != "w"]
    interior = [c for c in at_cells if min(c["w"]) > 1e-9]
    corr = {}
    for name in obj_names:
        corr[name] = {
            "all": spearman([c[name] for c in at_cells], [c["sr"] for c in at_cells]),
            "interior": spearman([c[name] for c in interior], [c["sr"] for c in interior]),
            "n_all": len(at_cells), "n_interior": len(interior),
        }
    # Which direction is "good": smaller error is good, larger coherence is good.
    good_high = {"lockstep", "same_traj"}
    print("Spearman(offline objective, closed-loop SR) across cells [sign flipped so + means agrees]:")
    for name in obj_names:
        sgn = 1.0 if name in good_high else -1.0
        print(f"  {name:14s} all={sgn * corr[name]['all']:+.3f} (n={corr[name]['n_all']})  "
              f"interior={sgn * corr[name]['interior']:+.3f} (n={corr[name]['n_interior']})")
    argmax = {}
    for name in obj_names:
        best = (max if name in good_high else min)(grid, key=lambda r: r[name])
        cell, dist = nearest_cell(best["w"], cells)
        argmax[name] = {"w": best["w"], "value": best[name], "nearest_cell": cell, "dist": dist,
                        "smoothed_sr": smoothed_sr(best["w"], cells, args.bandwidth)}

    # ------------------------------------------------------------------
    # Candidate zero-cost priors
    # ------------------------------------------------------------------
    priors = {"uniform": [1 / 3] * 3}
    if args.J:
        priors["J_ratio"] = normalize_w([float(x) for x in args.J.split(",")])
    priors["inv_spread_all"] = normalize_w([1 / stats[f]["spread_all"] for f in FIELDS])
    priors["inv_spread_top10"] = normalize_w([1 / stats[f]["spread_top10"] for f in FIELDS])
    priors["spread_all"] = normalize_w([stats[f]["spread_all"] for f in FIELDS])
    priors["fisher_delta_d_over_var"] = normalize_w([stats[f]["fisher_delta_d"] / stats[f]["fisher_delta_var"] for f in FIELDS])
    priors["fisher_delta_d_over_sd"] = normalize_w([stats[f]["fisher_delta_d"] / np.sqrt(stats[f]["fisher_delta_var"]) for f in FIELDS])
    priors["fisher_delta_d"] = normalize_w([stats[f]["fisher_delta_d"] for f in FIELDS])
    priors["fisher_nn_d_over_var"] = normalize_w([stats[f]["fisher_nn_d"] / stats[f]["fisher_nn_var"] for f in FIELDS])
    priors["fisher_nn_d_over_sd"] = normalize_w([stats[f]["fisher_nn_d"] / np.sqrt(stats[f]["fisher_nn_var"]) for f in FIELDS])
    priors["inv_phase_err"] = normalize_w([1 / stats[f]["top1_phase_err"] for f in FIELDS])
    priors["inv_phase_err_sq"] = normalize_w([1 / stats[f]["top1_phase_err"] ** 2 for f in FIELDS])
    priors["inv_remaining_err"] = normalize_w([1 / stats[f]["top1_remaining_err"] for f in FIELDS])
    priors["inv_phase_gt_0.2"] = normalize_w([1 / max(stats[f]["top1_phase_gt_0.2"], 1e-3) for f in FIELDS])
    priors["inv_phase_gt_0.3"] = normalize_w([1 / max(stats[f]["top1_phase_gt_0.3"], 1e-3) for f in FIELDS])
    priors["inv_grip_flip"] = normalize_w([1 / max(stats[f]["top1_grip_flip"], 1e-3) for f in FIELDS])
    priors["lockstep"] = normalize_w([stats[f]["lockstep"] for f in FIELDS])
    priors["same_traj"] = normalize_w([stats[f]["same_traj"] for f in FIELDS])
    priors["served_yaml"] = normalize_w(served_yaml)
    if args.served:
        priors["closed_loop_winner"] = normalize_w([float(x) for x in args.served.split(",")])

    rows = {}
    print(f"leader cell w={np.round(leader['w'], 3).tolist()} sr={leader['sr']:.3f} n={leader['n']}")
    print(f"{'prior':26s} {'w (v0/v1/rs)':22s} {'nearest cell':22s} {'dist':>5s} {'cell SR':>7s} {'smooth':>7s} {'gap':>6s}")
    for name, w in priors.items():
        cell, dist = nearest_cell(w, cells)
        sm = smoothed_sr(w, cells, args.bandwidth)
        rows[name] = {"w": w, "nearest_cell": cell, "dist": dist, "cell_sr": cell["sr"],
                      "smoothed_sr": sm, "gap_to_leader_pp": 100 * (leader["sr"] - cell["sr"]),
                      "smoothed_gap_pp": 100 * (leader["sr"] - sm), "objectives": objectives(w)}
        fmt = lambda v: "/".join(f"{x:.2f}" for x in v)  # noqa: E731
        print(f"{name:26s} {fmt(w):22s} {fmt(cell['w']):22s} {dist:5.2f} {cell['sr']:7.3f} {sm:7.3f} "
              f"{100 * (leader['sr'] - cell['sr']):+6.1f}")

    # Effective-vote reading of the leader: w_f * spread_f, normalized.
    plateau = [c for c in cells if c["sr"] >= leader["sr"] - 0.02]
    centroid = normalize_w(np.mean([c["w"] for c in plateau], axis=0))
    effective = {
        "leader_w_times_spread_all": normalize_w([leader["w"][k] * stats[f]["spread_all"] for k, f in enumerate(FIELDS)]),
        "leader_w_times_spread_top10": normalize_w([leader["w"][k] * stats[f]["spread_top10"] for k, f in enumerate(FIELDS)]),
        "plateau_centroid": centroid,
        "plateau_size": len(plateau),
        "centroid_w_times_spread_all": normalize_w([centroid[k] * stats[f]["spread_all"] for k, f in enumerate(FIELDS)]),
    }
    print("effective-vote reading:", json.dumps({k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in effective.items()}))

    result = {
        "library": str(args.library), "template": str(args.template), "cells": str(args.cells),
        "entries": n, "queries": int(len(queries)), "trajectories": len(traj_names), "tasks": len(task_names),
        "live_action_dims": live.tolist(), "horizon": h, "delta": args.delta, "device": device,
        "parity_max_abs": parity, "normalizers": norm, "served_yaml": served_yaml,
        "stats": stats, "priors": rows, "effective": effective,
        "leader": leader, "smooth_leader": smooth_leader, "correlations": corr, "argmax": argmax,
        "grid": grid, "at_cells": at_cells, "seconds": time.perf_counter() - t0,
    }
    (args.output / "lcw_offline_stats.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"done in {time.perf_counter() - t0:.0f}s -> {args.output / 'lcw_offline_stats.json'}")


if __name__ == "__main__":
    main()
