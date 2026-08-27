"""Build the dispatch-surface calibration table (s, v, Y_tau7, Y_tau10).

Offline replay of the query-cohort H5 against the frozen rebuilt library
(query and library are init-disjoint by construction, so no LOEO machinery is
needed; a fail-loud assertion still rejects any winner that shares a
trajectory with the query episode).

Per step of every fit / calibration episode:
  1. Reconstruct conditioning through the production KeyBuilder chain
     (``_build_fake_stage1_with_masks``) and search the frozen library with
     the production strategy -> s, top-k ids, winner.
  2. v = ``weighted_topk_disagreement`` over the top-k payload chunks (the
     same src function the judge runs — parity by construction).
  3. Reference action a_ref per ``--ref-mode``:
       fresh     (primary)  F^{0->N}(z_j) under the CURRENT conditioning,
                            z_j from the rebuilt library's noise sidecar.
       uncoupled (sens.)    the query step's own H5 ``clean_action``.
       tau1      (sens.)    9-step completion from winner intermediates[0.9].
  4. Y_tau10 = dev(winner.action_chunk, a_ref)   (no denoising needed)
     Y_tau7  = dev(run_stage3_from(intermediates[0.3], 0.3), a_ref)
     with ``weighted_chunk_deviation`` (shared src implementation).

Output: JSONL rows
  {episode_id, task_id, step_idx, split, s, v, k_eff, winner_id,
   y_tau7, y_tau10, ref_mode, episode_success}
plus a weights NPZ (W, active_mask) computed from the library and consumed by
``fit_surface.py`` so every downstream stage shares one W.

Usage:
  uv run python -m exp.dispatch_surface.build_dispatch_table \
      --query-h5-dir exp/dispatch_surface/data/query_cohort \
      --library-pkl exp/dispatch_surface/data/cache_artifacts/dispatch_lib_cp1_spatial_pool_16.pkl \
      --noise-sidecar exp/dispatch_surface/data/cache_artifacts/dispatch_lib_noise_sidecar.npz \
      --split-manifest exp/dispatch_surface/data/init_pools/split_manifest.json \
      --cache-yaml exp/dispatch_surface/config/calibration_retrieval.yaml \
      --config-name pi05_libero --checkpoint-dir <ckpt> \
      --ref-mode fresh --top-k 5 --h-exec 5 \
      --out-jsonl exp/dispatch_surface/data/dispatch_table_fresh.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib

import h5py
import numpy as np
import torch

from openpi.cache.components.surface_judge import (
    compute_library_action_weights,
    weighted_chunk_deviation,
    weighted_topk_disagreement,
)

REF_MODES = ("fresh", "uncoupled", "tau1")
START_T_WS = 0.3
START_T_TAU1 = 0.9


def _load_components(cache_yaml: str, library_pkl: str):
    """Assemble storage + search strategy through the production config path."""
    from openpi.cache.config import (
        build_per_connection_components,
        build_shared_storage,
        load_cache_config,
    )

    config = load_cache_config(cache_yaml)
    if config.backend.in_memory.preload_path != library_pkl:
        raise SystemExit(
            f"cache yaml preload_path={config.backend.in_memory.preload_path} "
            f"!= --library-pkl {library_pkl}; refusing a split-brain calibration"
        )
    storage = build_shared_storage(config)
    components = build_per_connection_components(config, storage)
    return config, storage, components


def _build_split_lookup(manifest_path: str) -> dict[tuple[int, int], str]:
    """(task_id, official init idx) -> 'fit' | 'cal' from the split manifest."""
    manifest = json.loads(pathlib.Path(manifest_path).read_text())
    lookup: dict[tuple[int, int], str] = {}
    for tid_str, info in manifest["assignment"].items():
        tid = int(tid_str)
        for split in ("fit", "cal"):
            for idx in info[split]:
                lookup[(tid, int(idx))] = split
    return lookup


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query-h5-dir", required=True)
    ap.add_argument("--library-pkl", required=True)
    ap.add_argument("--noise-sidecar", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--cache-yaml", required=True)
    ap.add_argument("--config-name", default="pi05_libero")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--ref-mode", choices=REF_MODES, default="fresh")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--h-exec", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--weights-out", default=None,
                    help="NPZ for (w, active_mask); default <out-jsonl>.weights.npz")
    args = ap.parse_args()

    from exp.common.build_in_memory_cache_artifact import (
        _build_fake_stage1_with_masks,
        _load_pi05_for_llm_extract,
        resolve_h5_paths,
    )
    from openpi.cache.components.payload_view import StoragePayloadView
    from openpi.cache.components.search_strategy import SearchContext
    from openpi.cache.types import CheckpointID

    config, storage, components = _load_components(args.cache_yaml, args.library_pkl)
    strategy = components["search_strategies"][CheckpointID.CP1]
    key_builder = components["key_builder"]
    view = StoragePayloadView(storage)

    # W / active mask from the frozen library — one shared W for v, Y and the
    # eventual artifact.
    import pickle

    with open(args.library_pkl, "rb") as f:
        lib = pickle.load(f)
    lib_chunks = torch.stack(
        [torch.as_tensor(e.payload.action_chunk, dtype=torch.float32) for e in lib["entries"]]
    )
    w, active_mask = compute_library_action_weights(lib_chunks)

    sidecar = np.load(args.noise_sidecar) if args.ref_mode in ("fresh",) else None
    model, tokenizer = _load_pi05_for_llm_extract(args.checkpoint_dir, args.config_name, args.device)
    dev = torch.device(args.device)
    split_lookup = _build_split_lookup(args.split_manifest)

    out_path = pathlib.Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Pre-scan ALL query trajectory ids before any search: an incremental set
    # would only catch winners from already-visited query episodes, letting a
    # hit on a later-visited one slip through (G2-B9).
    query_paths = resolve_h5_paths(args.query_h5_dir, None)
    query_traj_ids = {p.stem for p in query_paths}
    n_rows = 0
    with open(out_path, "w") as out:
        for h5_path in query_paths:
            with h5py.File(h5_path, "r") as h5:
                if "task_id" not in h5.attrs or "orig_init_state_idx" not in h5.attrs:
                    raise SystemExit(
                        f"{h5_path}: missing task_id / orig_init_state_idx attrs — "
                        "cohort was not collected through the identity-stamping recipe"
                    )
                task_id = int(h5.attrs["task_id"])
                init_idx = int(h5.attrs["orig_init_state_idx"])  # OFFICIAL index
                split = split_lookup.get((task_id, init_idx))
                if split is None:
                    raise SystemExit(
                        f"{h5_path}: (task {task_id}, official init {init_idx}) not in "
                        "the split manifest — cohort and pool manifest are out of sync"
                    )
                if "split" in h5.attrs and str(h5.attrs["split"]) != split:
                    raise SystemExit(
                        f"{h5_path}: stamped split {h5.attrs['split']!r} disagrees with "
                        f"the manifest ({split!r})"
                    )
                success = bool(h5.attrs.get("success", False))
                task_str = h5.attrs.get("prompt") or h5.attrs.get("task", "")

                step_names = sorted(
                    (k for k in h5.keys() if k.startswith("step_")),
                    key=lambda k: int(k.split("_")[1]),
                )
                for step_name in step_names:
                    step_idx = int(step_name.split("_")[1])
                    group = h5[step_name]
                    stage1 = _build_fake_stage1_with_masks(
                        group, str(task_str), tokenizer, model, dev,
                    )
                    key_builder.collect(CheckpointID.CP1, stage1=stage1)
                    query_keys = key_builder.build(CheckpointID.CP1)
                    ctx = SearchContext(
                        query_keys=query_keys, checkpoint_id=CheckpointID.CP1,
                        current_step=step_idx,
                    )
                    results = strategy.search(ctx)
                    if not results:
                        continue
                    winner_id = results[0].id
                    winner_entry = view.get_entry(winner_id)
                    if winner_entry.trajectory_id in query_traj_ids:
                        raise SystemExit(
                            f"winner {winner_id} shares trajectory with a query episode "
                            "— library / query separation is broken"
                        )
                    s = float(results[0].score)
                    ids = [r.id for r in results[: args.top_k]]
                    chunks = torch.stack(
                        [torch.as_tensor(p.action_chunk, dtype=torch.float32)
                         for p in view.get_many(ids)]
                    )
                    v = weighted_topk_disagreement(chunks, w, active_mask, args.h_exec)
                    winner_payload = view.get(winner_id)

                    with torch.no_grad():
                        stage2 = model.run_stage2(stage1)
                        # Reference branch under CURRENT conditioning.
                        if args.ref_mode == "fresh":
                            z = torch.from_numpy(sidecar[winner_id]).to(dev)[None]
                            a_ref = model.run_stage3(stage2, noise=z).action_chunk[0]
                        elif args.ref_mode == "tau1":
                            x9 = winner_payload.intermediates[START_T_TAU1].to(dev)[None]
                            a_ref = model.run_stage3_from(
                                stage2, x9, START_T_TAU1,
                                num_steps=winner_payload.denoising_num_steps,
                            ).action_chunk[0]
                        else:  # uncoupled
                            a_ref = torch.from_numpy(np.array(group["clean_action"])).to(dev)
                        # Warm branch from the winner's stored tier.
                        x3 = winner_payload.intermediates[START_T_WS].to(dev)[None]
                        a_warm = model.run_stage3_from(
                            stage2, x3, START_T_WS,
                            num_steps=winner_payload.denoising_num_steps,
                        ).action_chunk[0]

                    a_ref_cpu = a_ref.float().cpu()
                    y_tau10 = weighted_chunk_deviation(
                        torch.as_tensor(winner_payload.action_chunk, dtype=torch.float32),
                        a_ref_cpu, w, active_mask, args.h_exec,
                    )
                    y_tau7 = weighted_chunk_deviation(
                        a_warm.float().cpu(), a_ref_cpu, w, active_mask, args.h_exec,
                    )
                    out.write(json.dumps({
                        "episode_id": h5_path.stem,
                        "task_id": task_id,
                        "init_idx": init_idx,
                        "step_idx": step_idx,
                        "split": split,
                        "s": s,
                        "v": None if not np.isfinite(v) else v,
                        "k_eff": len(ids),
                        "winner_id": winner_id,
                        "y_tau7": y_tau7,
                        "y_tau10": y_tau10,
                        "ref_mode": args.ref_mode,
                        "episode_success": success,
                    }) + "\n")
                    n_rows += 1

    weights_out = args.weights_out or f"{args.out_jsonl}.weights.npz"
    np.savez(weights_out, w=w.numpy(), active_mask=active_mask.numpy())
    print(f"wrote {n_rows} rows -> {out_path}; weights -> {weights_out}")


if __name__ == "__main__":
    main()
