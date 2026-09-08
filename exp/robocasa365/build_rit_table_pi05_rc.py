"""Offline RIT calibration table for the RoboCasa365 pi0.5 line.

pi0.5 gets an offline replay where GR00T needs an in-server shadow, and the
asymmetry is a property of the two models rather than a choice. pi0.5's stage-1
prefix is the vision tokens followed by the prompt tokens, which is exactly
what the collector's HDF5 keeps, so ``_build_fake_stage1_with_masks`` can hand
the language model the same sequence it saw online. GR00T scatters its image
tokens into the text sequence and the collector stores the slices without the
scatter, so the same reconstruction is not available there.

What one row is
---------------
For every step of every calibration episode: the retrieval score under the
production key builder and search strategy, the winner it picked, and one
deviation per ladder rung -- the cached chunk for FULL_HIT, and the completion
of the winner's stored ``x_t`` under **this step's** conditioning for each warm
rung. The reference is the query step's own ``clean_action``: what full
inference did produce here, which is the quantity a deployed verdict is
trading against. That matches the reference the GR00T shadow uses, so the two
teachers' risk columns mean the same thing.

Usage:
  uv run python -m exp.robocasa365.build_rit_table_pi05_rc \\
      --calib-h5-dir /data/robocasa365_cache/calib_rit_w13_pi05/pi05 \\
      --cache-yaml /tmp/rit/calib_cell_pi05.yaml \\
      --library-pkl /data/.../pi05_spatial_pool_16_w13_full.pkl \\
      --checkpoint-dir /home/weiland/ckpt_pi05_robocasa_pytorch \\
      --config-name pi05_robocasa --h-exec 5 --out-jsonl /data/.../shadow_pi05.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib

import h5py
import numpy as np
import torch

from openpi.cache.components.payload_view import StoragePayloadView
from openpi.cache.components.search_strategy import SearchContext
from openpi.cache.components.surface_judge import (
    compute_library_action_weights,
    weighted_chunk_deviation,
)
from openpi.cache.types import CheckpointID

from exp.common.build_in_memory_cache_artifact import (
    _build_fake_stage1_with_masks,
    _load_pi05_for_llm_extract,
    resolve_h5_paths,
)
from exp.dispatch_surface.build_dispatch_table import _load_components
from exp.robocasa365.emit_rit_rc import WARM_TS


def rungs(warm_ts: list[float], num_steps: int) -> list[tuple[float, str]]:
    """``(start_t, y_key)`` per warm rung, keyed by remaining steps.

    The key is the step count rather than the timestep so a row reads the same
    for either teacher: pi0.5 resuming at 0.3 and GR00T resuming at 0.75 both
    say how much of the loop is left, which is what the cost model prices.
    """
    out = []
    for t in warm_ts:
        rem = int(round(float(t) * num_steps + 0.5 - 1e-9))
        out.append((round(float(t), 4), f"y_rem{rem}"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calib-h5-dir", required=True)
    ap.add_argument("--cache-yaml", required=True)
    ap.add_argument("--library-pkl", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--config-name", default="pi05_robocasa")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--h-exec", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--out-jsonl", required=True)
    args = ap.parse_args()

    _, storage, components = _load_components(args.cache_yaml, args.library_pkl, args.top_k)
    strategy = components["search_strategies"][CheckpointID.CP1]
    key_builder = components["key_builder"]
    view = StoragePayloadView(storage)

    import pickle

    with open(args.library_pkl, "rb") as f:
        lib = pickle.load(f)
    w, active_mask = compute_library_action_weights(
        torch.stack([
            torch.as_tensor(e.payload.action_chunk, dtype=torch.float32)
            for e in lib["entries"]
        ])
    )
    del lib

    model, tokenizer = _load_pi05_for_llm_extract(
        args.checkpoint_dir, args.config_name, args.device
    )
    dev = torch.device(args.device)

    out_path = pathlib.Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    paths = resolve_h5_paths(args.calib_h5_dir, None)
    if not paths:
        raise SystemExit(f"no HDF5 under {args.calib_h5_dir}")
    n_rows = 0
    with out_path.open("w") as out:
        for h5_path in paths:
            with h5py.File(h5_path, "r") as h5:
                task = str(h5.attrs.get("task", ""))
                episode_id = int(h5.attrs.get("episode_id", -1))
                success = bool(h5.attrs.get("success", False))
                task_str = h5.attrs.get("prompt") or task
                steps = sorted(
                    (k for k in h5 if k.startswith("step_")),
                    key=lambda k: int(k.split("_")[1]),
                )
                for name in steps:
                    step_idx = int(name.split("_")[1])
                    group = h5[name]
                    stage1 = _build_fake_stage1_with_masks(
                        group, str(task_str), tokenizer, model, dev
                    )
                    key_builder.collect(CheckpointID.CP1, stage1=stage1)
                    ctx = SearchContext(
                        query_keys=key_builder.build(CheckpointID.CP1),
                        checkpoint_id=CheckpointID.CP1,
                        current_step=step_idx,
                    )
                    results = strategy.search(ctx)
                    row = {
                        "experiment": "pi05", "task": task, "episode_id": episode_id,
                        "step_idx": step_idx, "s": None, "winner_id": None,
                        "episode_success": success,
                    }
                    if not results:
                        out.write(json.dumps(row) + "\n")
                        n_rows += 1
                        continue
                    winner_id = results[0].id
                    payload = view.get(winner_id)
                    row["s"] = float(results[0].score)
                    row["winner_id"] = winner_id
                    a_ref = torch.from_numpy(
                        np.array(group["clean_action"], dtype=np.float32)
                    )
                    row["y_full"] = weighted_chunk_deviation(
                        torch.as_tensor(payload.action_chunk, dtype=torch.float32),
                        a_ref, w, active_mask, args.h_exec,
                    )
                    with torch.no_grad():
                        stage2 = model.run_stage2(stage1)
                        for t, y_key in rungs(WARM_TS["pi05"], payload.denoising_num_steps):
                            xt = payload.intermediates.get(t)
                            if xt is None:
                                raise SystemExit(
                                    f"{winner_id}: library payload has no intermediate at "
                                    f"t={t}; the ladder cannot be calibrated against it"
                                )
                            chunk = model.run_stage3_from(
                                stage2, xt.to(dev)[None], t,
                                num_steps=payload.denoising_num_steps,
                            ).action_chunk[0]
                            row[y_key] = weighted_chunk_deviation(
                                chunk.float().cpu(), a_ref, w, active_mask, args.h_exec
                            )
                    out.write(json.dumps(row) + "\n")
                    n_rows += 1
            print(f"[rit-pi05] {h5_path.name}: {n_rows} rows so far", flush=True)
    print(f"[rit-pi05] wrote {n_rows} rows to {out_path}")


if __name__ == "__main__":
    main()
