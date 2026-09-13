"""Offline pruning audit of a RoboCasa365 pi0.5 library, through its production retrieval.

Same questions as the LIBERO audit (``exp/rit_pareto/analysis/prune_feasibility.py``),
asked of a RoboCasa library: at a given fusion-score threshold, how many entries
have a cross-trajectory near-duplicate that is further along, how much closer
to the goal the substitute is, how much the retrieved action changes for a
held-out trajectory, and what the search costs. Scores come from the served
strategy (``text_ivf_knn``: prompt-bucket screening, then the weighted score
sum), so "same task" here means "same prompt bucket", exactly as at serving.

Nothing here is a closed-loop result; it is the offline side of the question
"is there anything for pruning to shorten".

Usage:
  PYTHONPATH=. .venv/bin/python exp/robocasa365/analysis/prune_feasibility_rc.py \\
      --library /data/robocasa365_cache/cache_artifacts_w13/pi05_spatial_pool_16_w13_S3.pkl \\
      --config exp/robocasa365/config/rit_k1_v2/pi05/main/always_hit.yaml \\
      --output exp/robocasa365/analysis/prune_feasibility_pi05_w13_S3
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path
import pickle
import tempfile
import time

import numpy as np
import torch
import yaml

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.search_strategy import SearchContext
from openpi.cache.config import _build_search_strategy, _keys_iter, load_cache_config
from openpi.cache.types import CheckpointID


def _stats(values):
    x = np.asarray(values, dtype=float)
    if not x.size:
        return {"n": 0}
    return dict(n=int(x.size), mean=float(x.mean()), median=float(np.median(x)),
                p90=float(np.quantile(x, 0.9)), max=float(x.max()))


def _prune(pool, scores, remaining, trajectory, threshold):
    """Strict-shorter greedy pruning with a direct retained witness per deletion."""
    ordered = sorted(pool, key=lambda i: (remaining[i], i))
    keep, pairs = np.empty(len(pool), dtype=int), []
    size = 0
    for i in ordered:
        active = keep[:size]
        eligible = ((remaining[active] < remaining[i]) & (trajectory[active] != trajectory[i])
                    & (scores[i, active] >= threshold))
        candidates = active[eligible]
        if len(candidates):
            order = np.lexsort((candidates, remaining[candidates], -scores[i, candidates]))
            pairs.append((i, candidates[order[0]]))
        else:
            keep[size] = i
            size += 1
    return np.sort(keep[:size]), pairs


def _diagnose(query, winner, baseline, remaining, actions):
    err = np.sqrt(np.mean((actions[query] - actions[winner]) ** 2, axis=(1, 2)))
    old = np.sqrt(np.mean((actions[query] - actions[baseline]) ** 2, axis=(1, 2)))
    return {
        "changed_winner_pct": float(np.mean(winner != baseline) * 100),
        "mean_remaining_reduction": float(np.mean(remaining[baseline] - remaining[winner])),
        "baseline_mean_remaining": float(remaining[baseline].mean()),
        "action_rmse_change_pct": float((err.mean() / old.mean() - 1) * 100),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="served YAML; preload_path is replaced")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantiles", default="0.95,0.9,0.85,0.8,0.7,0.6,0.5,0.4,0.3,0.2",
                        help="thresholds are these quantiles of each entry's nearest cross-trajectory score")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    logging.basicConfig(level=logging.ERROR)

    raw = yaml.safe_load(args.config.read_text())
    raw["backend"]["in_memory"]["preload_path"] = str(args.library)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(raw, tmp, sort_keys=False)
    cfg = load_cache_config(tmp.name)
    enabled = [name for name, key in _keys_iter(cfg.keys) if key.enabled]
    fusion_weights = {name: key.weight for name, key in _keys_iter(cfg.keys) if key.enabled}
    ss_cfg = cfg.checkpoints["cp1"].search_strategy

    with args.library.open("rb") as handle:
        artifact = pickle.load(handle)
    entries = artifact["entries"]
    # The served backend receives tensors (load_artifact converts); the pickle
    # holds arrays, so convert before inserting the way the loader would.
    for entry in entries:
        entry.query_keys = {k: torch.as_tensor(v).float().contiguous() for k, v in entry.query_keys.items()}
        entry.payload.action_chunk = torch.as_tensor(entry.payload.action_chunk).float().contiguous()
    n = len(entries)
    ids = {e.id: i for i, e in enumerate(entries)}
    trajectories, buckets = defaultdict(list), defaultdict(list)
    for i, entry in enumerate(entries):
        trajectories[entry.trajectory_id].append(i)
    remaining = np.zeros(n, dtype=int)
    trajectory = np.array([e.trajectory_id for e in entries])
    for members in trajectories.values():
        members.sort(key=lambda i: entries[i].step_idx)
        for j, i in enumerate(members):
            remaining[i] = len(members) - 1 - j
    action_full = np.stack([e.payload.action_chunk.numpy() for e in entries])
    live_dims = np.flatnonzero(action_full.reshape(-1, action_full.shape[-1]).std(axis=0) > 0)
    actions = action_full[:, :5, live_dims]
    print(f"{n} entries, {len(trajectories)} trajectories, action dims used {live_dims.tolist()}", flush=True)

    def build(subset, top_k):
        backend = InMemoryBackend(vector_dims=cfg.backend.vector_dims,
                                  text_ivf=cfg.backend.in_memory.text_ivf
                                  if cfg.backend.in_memory.index_type == "text_ivf" else None)
        storage = CacheStorage(backend)
        storage.batch_insert([entries[i] for i in subset])
        backend.freeze()
        return _build_search_strategy(ss_cfg, storage, fusion_weights, min_top_k_hint=top_k)

    def context(entry):
        keys = {k: torch.as_tensor(entry.query_keys[k]).float() for k in enabled}
        return SearchContext(keys, CheckpointID.CP1, entry.step_idx, entry.payload.task_key)

    strategy = build(range(n), n)
    scores = np.full((n, n), -np.inf, dtype=np.float32)
    started = time.perf_counter()
    for i, entry in enumerate(entries):
        for hit in strategy.search(context(entry)):
            scores[i, ids[hit.id]] = hit.score
        if (i + 1) % 500 == 0:
            print(f"scored {i + 1}/{n}", flush=True)
    same_bucket = np.isfinite(scores)
    for i in range(n):
        buckets[int(np.flatnonzero(same_bucket[i])[0])].append(i)
    cross = same_bucket & (trajectory[:, None] != trajectory[None, :])
    nearest_cross = np.max(np.where(cross, scores, -np.inf), axis=1)
    has_cross = np.isfinite(nearest_cross)
    print(f"scoring took {time.perf_counter() - started:.0f}s; {len(buckets)} prompt buckets; "
          f"{int(has_cross.sum())}/{n} entries have a cross-trajectory neighbour; "
          f"nearest cross score {_stats(nearest_cross[has_cross])}", flush=True)
    np.savez_compressed(args.output / "pair_scores.npz", scores=scores, remaining=remaining,
                        ids=np.array([e.id for e in entries]), trajectories=trajectory)

    quantiles = [float(q) for q in args.quantiles.split(",")]
    thresholds = [float(np.quantile(nearest_cross[has_cross], q)) for q in quantiles]
    baseline = np.argmax(np.where(cross, scores, -np.inf), axis=1)
    query = np.flatnonzero(has_cross)
    rows = []
    masks = {}
    for q, threshold in zip(quantiles, thresholds):
        keep, pairs = _prune(np.arange(n), scores, remaining, trajectory, threshold)
        masks[threshold] = keep
        ii = np.array([i for i, _ in pairs], dtype=int)
        jj = np.array([j for _, j in pairs], dtype=int)
        # Held-out: each trajectory's queries are answered by a library pruned
        # without that trajectory, against the unpruned nearest neighbour.
        winners = np.full(n, -1, dtype=int)
        for members in trajectories.values():
            pool = np.array(sorted({j for i in members for j in np.flatnonzero(cross[i])}))
            if not pool.size:
                continue
            selected, _ = _prune(pool, scores, remaining, trajectory, threshold)
            sub = scores[np.ix_(members, selected)]
            winners[members] = selected[np.argmax(sub, axis=1)]
        valid = query[winners[query] >= 0]
        diag = _diagnose(valid, winners[valid], baseline[valid], remaining, actions)
        row = dict(
            quantile=q, threshold=threshold, kept=int(len(keep)), removed=int(n - len(keep)),
            removed_pct=float((n - len(keep)) / n * 100),
            pair_gap_mean=float(np.mean(remaining[ii] - remaining[jj])) if len(ii) else 0.0,
            pair_gap_median=float(np.median(remaining[ii] - remaining[jj])) if len(ii) else 0.0,
            pair_gap_le2_pct=float(np.mean((remaining[ii] - remaining[jj]) <= 2) * 100) if len(ii) else 0.0,
            **diag,
        )
        rows.append(row)
        print(f"threshold {threshold:.4f}: removed {row['removed_pct']:.1f}%, held-out remaining "
              f"reduction {diag['mean_remaining_reduction']:.3f} of {diag['baseline_mean_remaining']:.2f}, "
              f"action RMSE {diag['action_rmse_change_pct']:+.2f}%", flush=True)

    # Warm search cost: 100 fixed queries, seven shuffled rounds, baseline and pruned libraries.
    rng = np.random.default_rng(20260912)
    chosen = rng.choice(n, min(100, n), replace=False)
    contexts = [context(entries[i]) for i in chosen]
    timed = {"baseline": build(range(n), 1)}
    for threshold in thresholds[::2]:
        timed[f"{threshold:.4f}"] = build(masks[threshold], 1)
    for s in timed.values():
        for ctx in contexts:
            s.search(ctx)
    timings = defaultdict(list)
    for _ in range(7):
        for label in rng.permutation(list(timed)):
            for ctx in contexts:
                t0 = time.perf_counter_ns()
                timed[label].search(ctx)
                timings[label].append((time.perf_counter_ns() - t0) / 1e6)
    timing = {label: _stats(values) for label, values in timings.items()}

    result = {
        "library": str(args.library), "config": str(args.config), "entries": n,
        "trajectories": len(trajectories), "buckets": len(buckets),
        "entries_with_cross_neighbour": int(has_cross.sum()),
        "fusion_weights": fusion_weights, "action_dims_used": live_dims.tolist(),
        "rows": rows, "timing_ms": timing,
    }
    (args.output / "feasibility.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["| 融合分数阈值 | 保留点 | 删点比例 | 剩余长度减少 | 动作 RMSE 变化 | 配对步差中位 | 检索中位 ms |",
             "|---|---:|---:|---:|---:|---:|---:|",
             f"| 原库 | {n:,} | 0% | 0 | 0% | — | {timing['baseline']['median']:.3f} |"]
    for row in rows:
        ms = timing.get(f"{row['threshold']:.4f}")
        lines.append(f"| {row['threshold']:.4f} | {row['kept']:,} | {row['removed_pct']:.1f}% | "
                     f"{row['mean_remaining_reduction']:.3f} | {row['action_rmse_change_pct']:+.2f}% | "
                     f"{row['pair_gap_median']:.0f} | {ms['median']:.3f} |" if ms else
                     f"| {row['threshold']:.4f} | {row['kept']:,} | {row['removed_pct']:.1f}% | "
                     f"{row['mean_remaining_reduction']:.3f} | {row['action_rmse_change_pct']:+.2f}% | "
                     f"{row['pair_gap_median']:.0f} | 未测 |")
    table = "\n".join(lines)
    (args.output / "table.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
