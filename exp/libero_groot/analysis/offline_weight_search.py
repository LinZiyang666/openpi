"""Offline modality-weight search by leave-one-trajectory-out action distance.

Hold each trajectory out of the library, query the library with its entries,
take the top-1 hit's action chunk, and measure how far it is from the entry's
own chunk. Summed over every entry of every trajectory this is a number per
weight vector, and the weight vector with the smallest total wins. The
question is whether that offline winner lands where the closed-loop grid
search (500 episodes per cell) landed.

Per-field normalized similarities come from the production strategy itself,
run once per field with a one-hot weight vector; the served fusion is a plain
weighted sum of those, so every point of the simplex is then a matrix
combination and the whole grid costs seconds.

Usage:
  PYTHONPATH=. .venv/bin/python exp/libero_groot/analysis/offline_weight_search.py \\
      --library <pkl> --config <template yaml with fitted normalizers> \\
      --reference 5,4,3 --step 48 --horizon 5 --output <dir>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import pickle
import time

import numpy as np
import torch
import yaml

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.search_strategy import SearchContext
from openpi.cache.config import _build_search_strategy, _keys_iter, load_cache_config
from openpi.cache.types import CheckpointID

FIELDS = ("vision_0", "vision_1", "robot_state")


def simplex(step: int):
    for a in range(step + 1):
        for b in range(step + 1 - a):
            yield (a / step, b / step, (step - a - b) / step)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", default="", help="closed-loop winner as integers, e.g. 5,4,3")
    parser.add_argument("--step", type=int, default=48, help="simplex grid denominator")
    parser.add_argument("--horizon", type=int, default=5, help="action steps compared (replan length)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    logging.basicConfig(level=logging.ERROR)

    raw = yaml.safe_load(args.config.read_text())
    raw["backend"]["in_memory"]["preload_path"] = str(args.library)
    tmp = args.output / "config_used.yaml"
    tmp.write_text(yaml.safe_dump(raw, sort_keys=False))
    cfg = load_cache_config(str(tmp))
    enabled = [name for name, key in _keys_iter(cfg.keys) if key.enabled]
    served = {name: key.weight for name, key in _keys_iter(cfg.keys) if key.enabled}
    ss_cfg = cfg.checkpoints["cp1"].search_strategy

    with args.library.open("rb") as handle:
        artifact = pickle.load(handle)
    entries = artifact["entries"]
    for entry in entries:
        entry.query_keys = {k: torch.as_tensor(v).float().contiguous() for k, v in entry.query_keys.items()}
        entry.payload.action_chunk = torch.as_tensor(entry.payload.action_chunk).float().contiguous()
    n = len(entries)
    ids = {e.id: i for i, e in enumerate(entries)}
    trajectory = np.array([e.trajectory_id for e in entries])
    task = np.array([e.payload.task_key for e in entries])
    actions = np.stack([e.payload.action_chunk.numpy() for e in entries])
    live = np.flatnonzero(actions.reshape(-1, actions.shape[-1]).std(axis=0) > 0)
    horizon = min(args.horizon, actions.shape[1])
    act = actions[:, :horizon, live].reshape(n, -1)
    print(f"{n} entries, {len(set(trajectory))} trajectories, {len(set(task))} tasks, "
          f"action block {horizon}x{len(live)}", flush=True)

    backend = InMemoryBackend(vector_dims=cfg.backend.vector_dims)
    storage = CacheStorage(backend)
    storage.batch_insert(entries)
    backend.freeze()

    def context(entry):
        return SearchContext({k: entry.query_keys[k] for k in enabled}, CheckpointID.CP1,
                             entry.step_idx, entry.payload.task_key)

    started = time.perf_counter()
    per_field = {}
    for field in FIELDS:
        one_hot = {name: (1.0 if name == field else 0.0) for name in enabled}
        strategy = _build_search_strategy(ss_cfg, storage, one_hot, min_top_k_hint=n)
        matrix = np.full((n, n), -np.inf, dtype=np.float32)
        for i, entry in enumerate(entries):
            for hit in strategy.search(context(entry)):
                matrix[i, ids[hit.id]] = hit.score
        per_field[field] = matrix
        print(f"{field}: scored in {time.perf_counter() - started:.0f}s", flush=True)
    cross = (task[:, None] == task[None, :]) & (trajectory[:, None] != trajectory[None, :])
    has_cross = cross.any(axis=1)
    queries = np.flatnonzero(has_cross)
    penalty = np.where(cross, 0.0, -np.inf).astype(np.float32)
    S = np.stack([np.where(np.isfinite(per_field[f]), per_field[f], 0.0) for f in FIELDS])

    def evaluate(w):
        fused = np.tensordot(np.asarray(w, dtype=np.float32), S, axes=1) + penalty
        winner = np.argmax(fused[queries], axis=1)
        diff = act[queries] - act[winner]
        l2 = np.sqrt((diff ** 2).sum(axis=1))
        return float(l2.mean()), float(np.sqrt((diff ** 2).mean())), winner

    grid = list(simplex(args.step))
    rows = []
    for w in grid:
        mean_l2, rmse, _ = evaluate(w)
        rows.append({"w": w, "mean_l2": mean_l2, "rmse": rmse})
    rows.sort(key=lambda r: r["mean_l2"])
    best = rows[0]
    reference = None
    if args.reference:
        ref = [int(x) for x in args.reference.split(",")]
        ref_w = tuple(x / sum(ref) for x in ref)
        mean_l2, rmse, _ = evaluate(ref_w)
        rank = 1 + sum(r["mean_l2"] < mean_l2 for r in rows)
        reference = {"w": ref_w, "mean_l2": mean_l2, "rmse": rmse, "rank": rank, "of": len(rows)}
    corners = {f: evaluate(tuple(1.0 if g == f else 0.0 for g in FIELDS))[0] for f in FIELDS}
    result = {
        "library": str(args.library), "config": str(args.config), "entries": n,
        "queries": int(len(queries)), "horizon": horizon, "action_dims": live.tolist(),
        "served_weights": served, "step": args.step,
        "best": best, "reference": reference, "corners": corners,
        "top20": rows[:20], "grid": rows,
    }
    (args.output / "offline_weight_search.json").write_text(json.dumps(result, indent=2) + "\n")
    fmt = lambda w: "/".join(f"{x:.3f}" for x in w)  # noqa: E731
    print(f"\nbest offline: v0/v1/rs = {fmt(best['w'])}  mean L2 {best['mean_l2']:.4f}")
    print("top 10:")
    for r in rows[:10]:
        print(f"  {fmt(r['w'])}  L2 {r['mean_l2']:.4f}")
    print("corners: " + ", ".join(f"{f} {v:.4f}" for f, v in corners.items()))
    if reference:
        print(f"closed-loop winner {fmt(reference['w'])}: mean L2 {reference['mean_l2']:.4f}, "
              f"rank {reference['rank']}/{reference['of']} "
              f"(spread best..worst {best['mean_l2']:.4f}..{rows[-1]['mean_l2']:.4f})")


if __name__ == "__main__":
    main()
