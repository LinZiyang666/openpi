"""Rebuild the dispatch-surface library with known initial noise (fresh coupling).

The fresh coupled reference of the dispatch note requires the initial noise
z_j of every library entry to be known, and the previously referenced nine-tier
warm library was deleted in the factor-artifact rebuild. This script therefore
rebuilds the calibration/precheck library from the library-source H5 in two
passes:

  1. ``build_artifact`` (the production offline builder) produces the base
     artifact — query keys, trajectory chain, library_stats — through exactly
     the pipeline the server preloads, so retrieval semantics are identical.
  2. A model pass regenerates every entry's payload under the SAME
     conditioning: z_j is drawn from a deterministic per-entry generator
     (``stable_seed``), Stage 2 + Stage 3 run with
     ``return_intermediates=True`` over all nine canonical timesteps, and the
     payload's ``action_chunk`` / ``intermediates`` are replaced with the
     regenerated ones. z_j goes to a separate NPZ sidecar consumed only by the
     calibration table builder — runtime payloads are unchanged in shape.

Outputs (under --out-dir):
  dispatch_lib_<builder>.pkl        the library (standard artifact format)
  dispatch_lib_noise_sidecar.npz    entry_id -> z_j float32 [H, D]
  rebuild_record.json               entry count, per-tier completeness, sha256

Usage:
  uv run python -m exp.dispatch_surface.rebuild_dispatch_library \
      --h5-dir exp/common/data/db/libero_cache/libero_spatial \
      --split-manifest exp/dispatch_surface/data/init_pools/split_manifest.json \
      --builder cp1_spatial_pool_16 \
      --config-name pi05_libero --checkpoint-dir <ckpt> \
      --out-dir exp/dispatch_surface/data/cache_artifacts --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import pickle

import h5py
import numpy as np
import torch

NINE_TIERS = tuple(round(1.0 - i / 10, 4) for i in range(1, 10))  # 0.9 .. 0.1
NUM_STEPS = 10


def _file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def validate_split_manifest_binding(
    split_manifest_path: pathlib.Path, h5_dir: pathlib.Path,
) -> tuple[dict, dict]:
    """Recompute D_lib authority from source and bind it to the rebuild.

    A digest copied from the split manifest is only a claim.  Re-running the
    census against the live init map, H5 tree and official pools makes the
    rebuild refuse a swapped official identity or post-split source drift.
    """
    from exp.dispatch_surface.split_init_pools import census_dlib_inits

    if not split_manifest_path.is_file():
        raise SystemExit(f"split manifest missing: {split_manifest_path}")
    manifest = json.loads(split_manifest_path.read_text())
    expected = manifest.get("dlib_content_digests")
    if not isinstance(expected, dict):
        raise SystemExit("split manifest has no dlib_content_digests authority")
    init_map = pathlib.Path(str(manifest.get("init_map", "")))
    official_pool_dir = pathlib.Path(str(manifest.get("apool_dir", "")))
    _, actual = census_dlib_inits(
        init_map,
        h5_dir=h5_dir,
        official_pool_dir=official_pool_dir,
    )
    if actual != expected:
        raise SystemExit(
            "live D_lib H5/init/official-pool contents differ from the split manifest"
        )
    return manifest, actual


def _regen_payloads(
    data: dict,
    h5_dir: pathlib.Path,
    config_name: str,
    checkpoint_dir: str,
    device: str,
) -> dict[str, np.ndarray]:
    """Second pass: regenerate every entry payload with a known z_j (in place)."""
    from exp.common.build_in_memory_cache_artifact import (
        _build_fake_stage1_with_masks,
        _load_pi05_for_llm_extract,
        resolve_h5_paths,
    )
    from openpi.cache.shadow_teacher import stable_seed

    model, tokenizer = _load_pi05_for_llm_extract(checkpoint_dir, config_name, device)
    dev = torch.device(device)

    # Index entries by (trajectory_id, step_idx) to walk H5 groups in order.
    by_traj: dict[str, dict[int, object]] = {}
    for entry in data["entries"]:
        by_traj.setdefault(entry.trajectory_id, {})[entry.step_idx] = entry

    sidecar: dict[str, np.ndarray] = {}
    n_regen = 0
    for h5_path in resolve_h5_paths(h5_dir, None):
        traj_id = h5_path.stem
        steps = by_traj.get(traj_id)
        if not steps:
            continue
        with h5py.File(h5_path, "r") as h5:
            task_str = h5.attrs.get("prompt") or h5.attrs.get("task", "")
            step_names = sorted(
                (k for k in h5.keys() if k.startswith("step_")),
                key=lambda k: int(k.split("_")[1]),
            )
            for step_name in step_names:
                step_idx = int(step_name.split("_")[1])
                entry = steps.get(step_idx)
                if entry is None:
                    continue
                group = h5[step_name]
                stage1 = _build_fake_stage1_with_masks(group, str(task_str), tokenizer, model, dev)
                with torch.no_grad():
                    stage2 = model.run_stage2(stage1)
                    gen = torch.Generator(device=dev)
                    gen.manual_seed(stable_seed(traj_id, 0, step_idx))
                    noise = model.sample_noise(
                        (1, entry.payload.action_chunk.shape[0], entry.payload.action_chunk.shape[1]),
                        dev,
                        generator=gen,
                    )
                    stage3 = model.run_stage3(
                        stage2,
                        noise=noise,
                        return_intermediates=True,
                        save_timesteps=NINE_TIERS,
                    )
                entry.payload.action_chunk = stage3.action_chunk[0].float().cpu()
                entry.payload.intermediates = {
                    round(t, 4): x[0].float().cpu() for t, x in stage3.intermediates.items()
                }
                entry.payload.denoising_num_steps = NUM_STEPS
                sidecar[entry.id] = noise[0].float().cpu().numpy()
                n_regen += 1

    missing = [
        e.id for e in data["entries"] if e.id not in sidecar
    ]
    if missing:
        raise SystemExit(
            f"{len(missing)} entries had no matching H5 step (first: {missing[:3]}); "
            "library and H5 source are out of sync — aborting"
        )
    print(f"regenerated payloads for {n_regen} entries")
    return sidecar


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5-dir", required=True)
    ap.add_argument("--split-manifest", required=True,
                    help="authoritative split manifest; D_lib content is recomputed "
                         "and compared before rebuilding")
    ap.add_argument("--builder", default="cp1_spatial_pool_16")
    ap.add_argument("--config-name", default="pi05_libero")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    split_manifest_path = pathlib.Path(args.split_manifest)
    _, dlib_content_digests = validate_split_manifest_binding(
        split_manifest_path, pathlib.Path(args.h5_dir),
    )

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_pkl = out_dir / f"dispatch_lib_{args.builder}.pkl"

    # Pass 1: production builder (query keys / chain / library_stats).
    from exp.common.build_in_memory_cache_artifact import build_artifact

    data = build_artifact(
        data_dir=args.h5_dir,
        builder_type=args.builder,
    )

    # Pass 2: regenerate payloads with known z_j (mutates `data` in place).
    sidecar = _regen_payloads(
        data, pathlib.Path(args.h5_dir), args.config_name, args.checkpoint_dir, args.device,
    )
    with open(base_pkl, "wb") as f:
        pickle.dump(data, f)

    sidecar_path = out_dir / "dispatch_lib_noise_sidecar.npz"
    np.savez(sidecar_path, **sidecar)

    # Rebuild record: completeness + content identity.
    n = len(data["entries"])
    tier_counts = {f"{t:.4f}": 0 for t in NINE_TIERS}
    for entry in data["entries"]:
        for t in entry.payload.intermediates or {}:
            tier_counts[f"{t:.4f}"] += 1
    record = {
        "builder": args.builder,
        "h5_dir": str(args.h5_dir),
        "config_name": args.config_name,
        "checkpoint_dir": str(args.checkpoint_dir),
        "entry_count": n,
        "tier_completeness": {k: v / n for k, v in sorted(tier_counts.items())},
        "library_sha256": _file_sha256(base_pkl),
        "noise_sidecar_sha256": _file_sha256(sidecar_path),
        "split_manifest_sha256": _file_sha256(split_manifest_path),
        "dlib_content_digests": dlib_content_digests,
    }
    incomplete = {k: v for k, v in record["tier_completeness"].items() if v < 1.0}
    if incomplete:
        raise SystemExit(f"tier completeness below 100%: {incomplete}")
    record_path = out_dir / "rebuild_record.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True))
    print(f"library: {base_pkl}\nsidecar: {sidecar_path}\nrecord: {record_path}")


if __name__ == "__main__":
    main()
