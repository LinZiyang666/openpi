"""Decompose the pi0.5 retrieval score into its fields, step by step.

The calibration score decays 0.985 -> 0.849 across an episode while the
deployed score stays flat. Two readings fit that: the teacher's trajectory
genuinely drifts away from the library, or one field's normalizer is scaled so
tightly that it collapses as soon as the state moves. They are separable --
this replays calibration episodes through the production search and records the
FUSED WINNER's per-field normalized score (``StepRetrievalFeatures.winner_per_field``),
so the decay can be attributed to a field rather than to the score as a whole.

Reads the same cache yaml, library and checkpoint the offline table used, so
any difference from that table is this probe's own instrumentation and nothing
else.

Usage:
  uv run python -m exp.robocasa365.probe_field_decay_pi05 \\
      --calib-h5-dir ... --cache-yaml ... --library-pkl ... --checkpoint-dir ... \\
      --episodes 5 --out-jsonl ...
"""

from __future__ import annotations

import argparse
import json
import pathlib

import h5py

from openpi.cache.components.search_strategy import SearchContext
from openpi.cache.storage_types import CheckpointID

from exp.common.build_in_memory_cache_artifact import (
    _build_fake_stage1_with_masks,
    _load_pi05_for_llm_extract,
    resolve_h5_paths,
)
from exp.dispatch_surface.build_dispatch_table import _load_components


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calib-h5-dir", required=True)
    ap.add_argument("--cache-yaml", required=True)
    ap.add_argument("--library-pkl", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--config-name", default="pi05_robocasa")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--out-jsonl", required=True)
    args = ap.parse_args()

    _, storage, components = _load_components(args.cache_yaml, args.library_pkl, 1)
    strategy = components["search_strategies"][CheckpointID.CP1]
    key_builder = components["key_builder"]

    model, tokenizer = _load_pi05_for_llm_extract(
        args.checkpoint_dir, args.config_name, args.device
    )
    import torch

    dev = torch.device(args.device)

    paths = resolve_h5_paths(args.calib_h5_dir, None)[: args.episodes]
    out = pathlib.Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as sink:
        for h5_path in paths:
            with h5py.File(h5_path, "r") as h5:
                task = str(h5.attrs.get("task", ""))
                task_str = h5.attrs.get("prompt") or task
                episode_id = int(h5.attrs.get("episode_id", -1))
                steps = sorted(
                    (k for k in h5 if k.startswith("step_")),
                    key=lambda k: int(k.split("_")[1]),
                )
                for name in steps:
                    step_idx = int(name.split("_")[1])
                    stage1 = _build_fake_stage1_with_masks(
                        h5[name], str(task_str), tokenizer, model, dev
                    )
                    key_builder.collect(CheckpointID.CP1, stage1=stage1)
                    ctx = SearchContext(
                        query_keys=key_builder.build(CheckpointID.CP1),
                        checkpoint_id=CheckpointID.CP1,
                        current_step=step_idx,
                    )
                    # Go through the strategy's own search so the spec carries
                    # whatever filters and top-k widening the deployed path
                    # applies; the facade stashes that search's diagnostics.
                    results = strategy.search(ctx)
                    diag = storage.last_step_features()   # a method, not a property
                    row = {
                        "task": task, "episode_id": episode_id, "step_idx": step_idx,
                        "s": float(results[0].score) if results else None,
                        "winner_id": results[0].id if results else None,
                        "per_field": {k: float(v) for k, v in
                                      ((diag.winner_per_field if diag else None) or {}).items()},
                        "field_own_margin": {k: float(v) for k, v in
                                             ((diag.field_own_margin if diag else None) or {}).items()},
                    }
                    sink.write(json.dumps(row) + "\n")
                    n += 1
            print(f"[probe] {h5_path.name}: {n} rows", flush=True)
    print(f"[probe] wrote {n} rows to {out}")


if __name__ == "__main__":
    main()
