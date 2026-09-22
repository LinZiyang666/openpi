"""Measure the query-vs-library similarity population the normalizer should be fitted on.

The per-field normalizer's mu/sigma were fitted on the LIBRARY's internal
pairwise similarities. What the served score actually normalizes is a
QUERY-vs-library similarity, and if those two populations sit at different
centres the z-score saturates and the fused score loses resolution exactly
where the thresholds cut.

This replays calibration steps through the production key builder and search,
then recomputes the RAW per-field similarity between the query key and the
winner's stored key -- the quantity the normalizer consumes -- so the two
populations can be compared directly.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import pickle

import h5py
import numpy as np
import torch

from openpi.cache.components.search_strategy import SearchContext
from openpi.cache.types import CheckpointID

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
    ap.add_argument("--per-task", type=int, default=1)
    ap.add_argument("--out-jsonl", required=True)
    args = ap.parse_args()

    _, storage, components = _load_components(args.cache_yaml, args.library_pkl, 1)
    strategy = components["search_strategies"][CheckpointID.CP1]
    key_builder = components["key_builder"]

    with open(args.library_pkl, "rb") as f:
        lib = pickle.load(f)
    by_id = {e.id: e for e in lib["entries"]}
    fields = list(lib["vector_dims"])
    del lib

    model, tokenizer = _load_pi05_for_llm_extract(
        args.checkpoint_dir, args.config_name, args.device
    )
    dev = torch.device(args.device)

    # one episode per task, so the population is not one task's geometry
    chosen: list = []
    per_task: dict[str, int] = {}
    for path in resolve_h5_paths(args.calib_h5_dir, None):
        name = path.parent.name
        if per_task.get(name, 0) >= args.per_task:
            continue
        per_task[name] = per_task.get(name, 0) + 1
        chosen.append(path)
    out = pathlib.Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as sink:
        for h5_path in chosen:
            with h5py.File(h5_path, "r") as h5:
                task = str(h5.attrs.get("task", ""))
                task_str = h5.attrs.get("prompt") or task
                steps = sorted((k for k in h5 if k.startswith("step_")),
                               key=lambda k: int(k.split("_")[1]))
                for name in steps:
                    step_idx = int(name.split("_")[1])
                    stage1 = _build_fake_stage1_with_masks(
                        h5[name], str(task_str), tokenizer, model, dev)
                    key_builder.collect(CheckpointID.CP1, stage1=stage1)
                    qk = key_builder.build(CheckpointID.CP1)
                    res = strategy.search(SearchContext(
                        query_keys=qk, checkpoint_id=CheckpointID.CP1,
                        current_step=step_idx))
                    if not res:
                        continue
                    win = by_id.get(res[0].id)
                    if win is None:
                        continue
                    row = {"task": task, "step_idx": step_idx,
                           "fused": float(res[0].score), "winner": res[0].id, "raw": {}}
                    for fld in fields:
                        q = qk.get(fld)
                        w = win.query_keys.get(fld)
                        if q is None or w is None:
                            continue
                        qa = np.asarray(q.detach().cpu() if hasattr(q, "detach") else q,
                                        dtype=np.float32).ravel()
                        wa = np.asarray(w, dtype=np.float32).ravel()
                        if fld == "robot_state":
                            row["raw"][fld] = float(-np.linalg.norm(qa - wa))
                        else:
                            d = np.linalg.norm(qa) * np.linalg.norm(wa)
                            row["raw"][fld] = float(qa @ wa / d) if d > 0 else 0.0
                    sink.write(json.dumps(row) + "\n")
                    n += 1
            print(f"[raw] {h5_path.parent.name}/{h5_path.name}: {n} rows", flush=True)
    print(f"[raw] wrote {n} rows to {out}")


if __name__ == "__main__":
    main()
