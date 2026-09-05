"""Build a CP2 (post-backbone single-key) library one-to-one from a CP1 library.

Every entry of the source CP1 pickle is copied verbatim — same ``id``, same
payload (action chunk + denoising intermediates), same trajectory / step /
outcome / chain edges — and only two fields change: ``query_keys`` becomes
``{"vlm_out": k}`` with ``k`` the projected backbone prefix output of that
step, and ``checkpoint_id`` becomes ``CP2`` (the in-memory backend filters on
it). The backbone output is recomputed offline from the collection H5 of the
same step: the stored Stage-1 tensors (``vision_*`` / ``prompt_emb`` /
``robot_state``) are re-assembled into a Stage-1 output with the production
helper ``_build_fake_stage1_with_masks`` (tokenizer self-check included) and
pushed through ``PI0Pytorch.run_stage2_capture``.

Artifact metadata binds the library to what produced it: the projection spec
(seed / d / p / D / nnz / accumulation dtype / index digest), the id policy,
the source pickle sha256, a per-file H5 manifest, the model checkpoint
identity, the tokenizer and the git commit. ``verify_cp2_artifact.py`` checks
the result and ``config.build_shared_storage`` enforces the projection binding
at load time.

Usage:
  uv run python -m exp.actioncache_baseline.build_cp2_artifact \\
      --source-pkl <cp1 library>.pkl --h5-root <collection dir> \\
      --out-pkl <cp2 library>.pkl --config-name pi05_libero \\
      --checkpoint-dir <ckpt> --seed 20260904 [--device cuda:0] [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pathlib
import pickle
import time

import h5py
import torch

from exp.actioncache_baseline import libs
from exp.common.build_in_memory_cache_artifact import (
    _PI05_TOKENIZER_MAX_LEN,
    _PI05_TOKENIZER_SOURCE,
    _build_fake_stage1_with_masks,
    _load_pi05_for_llm_extract,
    _self_check_tokenizer_consistency,
)
from openpi.cache.components.cp2_vlm_key_builder import CP2VlmTernaryKeyBuilder
from openpi.cache.storage_types import CacheEntry
from openpi.cache.types import CheckpointID

logger = logging.getLogger("actioncache_baseline.build")


# ------------------------------------------------------------------
# Identity helpers
# ------------------------------------------------------------------


def _entry_copy_with_key(src: CacheEntry, key: torch.Tensor) -> CacheEntry:
    """Copy every entry field, replacing only ``query_keys`` and ``checkpoint_id``.

    Built explicitly (not ``dataclasses.replace``) so entries unpickled from
    older builds that predate optional fields still copy with their defaults.
    """
    return CacheEntry(
        id=src.id,
        checkpoint_id=CheckpointID.CP2,
        query_keys={libs.FIELD: key},
        payload=src.payload,
        step_idx=getattr(src, "step_idx", None),
        timestamp=getattr(src, "timestamp", 0.0),
        prev_ids=list(getattr(src, "prev_ids", []) or []),
        next_ids=list(getattr(src, "next_ids", []) or []),
        trajectory_id=getattr(src, "trajectory_id", None),
        outcome=getattr(src, "outcome", None),
    )


# ------------------------------------------------------------------
# Build
# ------------------------------------------------------------------


def build(args: argparse.Namespace) -> dict:
    t0 = time.time()
    source_path = pathlib.Path(args.source_pkl).resolve()
    src = libs.load_pickle(source_path)
    src_entries: list[CacheEntry] = list(src["entries"])
    if args.limit:
        src_entries = src_entries[: args.limit]
    logger.info("source %s: %d entries", source_path, len(src_entries))

    by_traj: dict[str, list[CacheEntry]] = {}
    for e in src_entries:
        tid = getattr(e, "trajectory_id", None)
        if tid is None or getattr(e, "step_idx", None) is None:
            raise SystemExit(f"entry {e.id} lacks trajectory_id / step_idx; cannot join to H5")
        by_traj.setdefault(tid, []).append(e)

    index = libs.H5Index(args.h5_root)
    logger.info("H5 index: %d files under %s", len(index), index.root)
    device = torch.device(args.device)
    model, tokenizer = _load_pi05_for_llm_extract(args.checkpoint_dir, args.config_name, args.device)
    model.eval()
    builder = CP2VlmTernaryKeyBuilder(seed=args.seed, d=args.d, p=args.p, input_dim=args.input_dim)

    out_entries: list[CacheEntry] = []
    manifest_files: list[dict] = []
    checked_tokenizer = False
    n_done = 0
    with torch.no_grad():
        for tid, entries in sorted(by_traj.items()):
            h5_path = index.resolve(tid)
            manifest_files.append({"trajectory_id": tid, "path": str(h5_path),
                                   "sha256": libs.sha256_file(h5_path)})
            if not checked_tokenizer:
                _self_check_tokenizer_consistency(h5_path, model, tokenizer, device)
                checked_tokenizer = True
            wanted = {e.step_idx: e for e in entries}
            with h5py.File(h5_path, "r") as f:
                task = str(f.attrs.get("task", ""))
                for step_idx, group in libs.iter_steps(f):
                    e = wanted.pop(step_idx, None)
                    if e is None:
                        continue
                    fake = _build_fake_stage1_with_masks(
                        group, task_str=task, tokenizer=tokenizer, model=model, device=device,
                    )
                    stage2 = model.run_stage2_capture(fake)
                    builder.collect(CheckpointID.CP2, stage2=stage2)
                    key = builder.build(CheckpointID.CP2)[libs.FIELD]
                    builder.clear()
                    out_entries.append(_entry_copy_with_key(e, key))
                    n_done += 1
            if wanted:
                raise SystemExit(
                    f"{h5_path}: steps {sorted(wanted)} of trajectory {tid} are not in the H5"
                )
            if n_done % 500 < len(entries):
                logger.info("%d / %d entries (%.0fs)", n_done, len(src_entries), time.time() - t0)

    if len(out_entries) != len(src_entries):
        raise SystemExit(f"built {len(out_entries)} entries for {len(src_entries)} source entries")

    manifest_digest = hashlib.sha256()
    for row in sorted(manifest_files, key=lambda r: r["trajectory_id"]):
        manifest_digest.update(f"{row['trajectory_id']}:{row['sha256']}\n".encode())

    artifact = {
        "key_builder_type": libs.KEY_BUILDER_TYPE,
        "checkpoint_id": "CP2",
        "vector_dims": {libs.FIELD: args.d},
        "entries": out_entries,
        "projection": builder.projection_meta(),
        "id_policy": libs.ID_POLICY,
        "source_pkl": str(source_path),
        "source_pkl_sha256": libs.sha256_file(source_path),
        "source_key_builder_type": src.get("key_builder_type"),
        "h5_manifest": {"root": str(index.root), "files": manifest_files,
                        "digest": manifest_digest.hexdigest()},
        # ``weights_digest``: full-content sha256 of the checkpoint dir — the
        # binding every shadow / overhead / parity run must re-derive.
        "model": {"config_name": args.config_name, **libs.weights_digest(args.checkpoint_dir)},
        "tokenizer": {"source": _PI05_TOKENIZER_SOURCE, "max_len": _PI05_TOKENIZER_MAX_LEN},
        "build_git_commit": libs.git_commit(),
        "protocol": libs.PROTOCOL,
    }
    if src.get("library_stats") is not None:
        artifact["library_stats"] = src["library_stats"]

    out_path = pathlib.Path(args.out_pkl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
    record = {k: v for k, v in artifact.items() if k not in ("entries", "library_stats")}
    record["n_entries"] = len(out_entries)
    record["out_pkl"] = str(out_path.resolve())
    record["out_pkl_sha256"] = libs.sha256_file(out_path)
    record["elapsed_s"] = round(time.time() - t0, 1)
    libs.dump_json(out_path.with_suffix(".record.json"), record)
    logger.info("wrote %s (%d entries, %.0fs)", out_path, len(out_entries), time.time() - t0)
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-pkl", required=True)
    ap.add_argument("--h5-root", required=True)
    ap.add_argument("--out-pkl", required=True)
    ap.add_argument("--config-name", default="pi05_libero")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--d", type=int, default=libs.ProjectionArgs.d)
    ap.add_argument("--p", type=float, default=libs.ProjectionArgs.p)
    ap.add_argument("--input-dim", type=int, default=libs.ProjectionArgs.input_dim)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N source entries")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rec = build(args)
    print(json.dumps({k: rec[k] for k in ("out_pkl", "n_entries", "out_pkl_sha256", "elapsed_s")}))


if __name__ == "__main__":
    main()
