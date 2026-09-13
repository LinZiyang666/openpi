"""Build the GR00T CP2 library entry-for-entry from a W13 CP1 library (plan §3.6).

Same contract as the Pi0.5 builder (``exp/actioncache_baseline/build_cp2_artifact.py``):
every entry of the source pickle is copied with only ``query_keys`` (-> the
projected encoded key) and ``checkpoint_id`` (-> CP2) changed; ids, payloads,
chain edges, trajectory / step / outcome fields are inherited
(``id_policy: inherited_from_source``). The key of each step comes from the
serving code path itself -- ``run_stage2_llm`` + ``run_cp2_key_source`` on the
stage-1 sequence rebuilt by ``cp2_reconstruct`` -- and the same
``GrootCP2TernaryKeyBuilder`` the server runs.

The artifact's top-level ``schedule_id`` is copied from the source (it is what
``InMemoryBackend.load_artifact`` reads; an absent stamp is back-filled as
Pi0.5's and then collides with every GR00T payload), and before anything is
built the source stamp, every payload stamp, the profile schedule and the H5
files' ``denoise_schedule_id`` must agree.

Environment: the GR00T island venv (torch + gr00t, no jax). Usage:
  python -m exp.libero_groot.build_cp2_artifact_groot \\
      --source-pkl /data/libero_cache/libraries_w13/<suite>/<suite>_w13_S3.pkl \\
      --h5-root /archive/libero_cache/build_<suite>_w13/<suite> \\
      --checkpoint <ckpt dir> --seed 20260904 \\
      --out-pkl /data/libero_cache/libraries_w13_cp2/<suite>/<suite>_w13_S3_cp2.pkl
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import pathlib
import pickle
import time

import h5py
import torch

from openpi.cache.groot.cp2_key_builder import (
    DEFAULT_D,
    DEFAULT_FEATURE_DIM,
    DEFAULT_P,
    DEFAULT_STATE_FEAT_DIM,
    DEFAULT_TOKEN_LEN,
    KEY_BUILDER_TYPE,
    GrootCP2TernaryKeyBuilder,
)
from openpi.cache.groot.staged import GrootStagedRunner
from openpi.cache.storage_types import CacheEntry
from openpi.cache.types import CheckpointID, schedule_from_id

from exp.actioncache_baseline import libs
from exp.libero_groot.cp2_reconstruct import (
    STAGE1_PATH,
    TemplateCache,
    h5_task,
    load_groot_libero_policy,
    reconstruct_stage1,
)

logger = logging.getLogger("build_cp2_artifact_groot")
PROFILE = libs.GROOT_LIBERO


def entry_copy_with_key(src: CacheEntry, key: torch.Tensor) -> CacheEntry:
    """Copy every entry field, replacing only ``query_keys`` and ``checkpoint_id``."""
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


def assert_schedule_consistency(src: dict, entries: list[CacheEntry], h5_paths: list[pathlib.Path]) -> str:
    """Source stamp == every payload stamp == profile == every H5 stamp, k = 8."""
    want = PROFILE.denoise_schedule
    if src.get("schedule_id") != want:
        raise SystemExit(f"source pkl schedule_id {src.get('schedule_id')!r} != {want!r}")
    n_steps = schedule_from_id(want).num_steps
    for e in entries:
        p = e.payload
        if getattr(p, "schedule_id", None) != want or p.denoising_num_steps != n_steps:
            raise SystemExit(
                f"entry {e.id}: payload schedule_id={getattr(p, 'schedule_id', None)!r} "
                f"denoising_num_steps={p.denoising_num_steps}; expected {want!r} / {n_steps}"
            )
        inter = p.intermediates or {}
        if not any(abs(float(t) - PROFILE.warm_start_t) < 1e-6 for t in inter):
            raise SystemExit(f"entry {e.id}: no snapshot at start_t={PROFILE.warm_start_t}")
    for path in h5_paths:
        with h5py.File(path, "r") as f:
            sid = f.attrs.get("denoise_schedule_id", None)
            n = int(f.attrs.get("denoising_num_steps", -1))
            if sid != want or n != n_steps:
                raise SystemExit(f"{path}: denoise_schedule_id={sid!r} denoising_num_steps={n}; expected {want!r} / {n_steps}")
    return want


def build(args: argparse.Namespace) -> dict:
    """Copy the source CP1 library entry for entry, replacing only the key with the CP2 projection.

    Every entry is joined to its H5 step through ``trajectory_id`` /
    ``step_idx``, reconstructed from the instruction template, run through
    stage 2 and the key-source encoders, then projected. Schedule stamps
    (source, payloads, H5, live head) must all be the profile's k=8 loop and
    every payload must hold the 0.875 snapshot. Writes ``--out-pkl`` and its
    ``.record.json``; returns the record.
    """
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
    h5_paths = [index.resolve(tid) for tid in sorted(by_traj)]
    schedule_id = assert_schedule_consistency(src, src_entries, h5_paths)

    model_identity = libs.weights_digest(args.checkpoint)
    policy = load_groot_libero_policy(args.checkpoint, denoising_steps=args.denoising_steps, device=args.device)
    runner = GrootStagedRunner(policy.model)
    live = runner.live_schedule()
    if live.schedule_id != schedule_id:
        raise SystemExit(f"served head runs {live.schedule_id}, library is {schedule_id}")
    builder = GrootCP2TernaryKeyBuilder(
        seed=args.seed, d=args.d, p=args.p, token_len=args.token_len,
        feature_dim=args.feature_dim, state_feat_dim=args.state_feat_dim,
    )
    templates = TemplateCache(policy, runner)

    out_entries: list[CacheEntry] = []
    manifest_files: list[dict] = []
    n_done = 0
    max_tokens = 0
    with runner.session():
        for tid, entries in sorted(by_traj.items()):
            h5_path = index.resolve(tid)
            manifest_files.append({"trajectory_id": tid, "path": str(h5_path), "sha256": libs.sha256_file(h5_path)})
            wanted = {e.step_idx: e for e in entries}
            with h5py.File(h5_path, "r") as f:
                template = templates.get(h5_task(f))
                max_tokens = max(max_tokens, template.n_tokens)
                for step_idx, group in libs.iter_steps(f):
                    e = wanted.pop(step_idx, None)
                    if e is None:
                        continue
                    stage1 = reconstruct_stage1(template, group)
                    stage2 = runner.run_stage2_llm(stage1)
                    source = runner.run_cp2_key_source(stage2)
                    builder.collect(CheckpointID.CP2, cp2_source=source, stage2=stage2)
                    key = builder.build(CheckpointID.CP2)[libs.FIELD]
                    builder.clear()
                    out_entries.append(entry_copy_with_key(e, key))
                    n_done += 1
            if wanted:
                raise SystemExit(f"{h5_path}: steps {sorted(wanted)} of trajectory {tid} are not in the H5")
            logger.info("%d / %d entries (%.0fs, %d templates)", n_done, len(src_entries), time.time() - t0, len(templates))
    if len(out_entries) != len(src_entries):
        raise SystemExit(f"built {len(out_entries)} entries for {len(src_entries)} source entries")

    manifest_digest = hashlib.sha256()
    for row in sorted(manifest_files, key=lambda r: r["trajectory_id"]):
        manifest_digest.update(f"{row['trajectory_id']}:{row['sha256']}\n".encode())

    artifact = {
        "key_builder_type": KEY_BUILDER_TYPE,
        "checkpoint_id": "CP2",
        "vector_dims": {libs.FIELD: args.d},
        "schedule_id": schedule_id,
        "entries": out_entries,
        "projection": builder.projection_meta(),
        "id_policy": libs.ID_POLICY,
        "teacher": PROFILE.name,
        "stage1_path": STAGE1_PATH,
        "source_pkl": str(source_path),
        "source_pkl_sha256": libs.sha256_file(source_path),
        "source_key_builder_type": src.get("key_builder_type"),
        "h5_manifest": {"root": str(index.root), "files": manifest_files, "digest": manifest_digest.hexdigest()},
        "model": {"config_name": "groot_n15_libero", "denoising_steps": int(args.denoising_steps),
                  "max_template_tokens": int(max_tokens), **model_identity},
        "tokenizer": {"source": "gr00t eagle chat template (zero-image template per instruction)",
                      "n_templates": len(templates)},
        "build_git_commit": libs.git_commit(),
        "protocol": libs.PROTOCOL,
    }
    if src.get("library_stats") is not None:
        artifact["library_stats"] = src["library_stats"]
    if src.get("prompt_pool") is not None:
        artifact["prompt_pool"] = src["prompt_pool"]

    out_path = pathlib.Path(args.out_pkl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
    record = {k: v for k, v in artifact.items() if k not in ("entries", "library_stats", "prompt_pool")}
    record["n_entries"] = len(out_entries)
    record["out_pkl"] = str(out_path.resolve())
    record["out_pkl_sha256"] = libs.sha256_file(out_path)
    record["elapsed_s"] = round(time.time() - t0, 1)
    libs.dump_json(out_path.with_suffix(".record.json"), record)
    logger.info("wrote %s (%d entries, %.0fs)", out_path, len(out_entries), time.time() - t0)
    return record


def main() -> None:
    """CLI entry: see the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-pkl", required=True)
    ap.add_argument("--h5-root", required=True)
    ap.add_argument("--out-pkl", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--denoising-steps", type=int, default=8)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--d", type=int, default=DEFAULT_D)
    ap.add_argument("--p", type=float, default=DEFAULT_P)
    ap.add_argument("--token-len", type=int, default=DEFAULT_TOKEN_LEN)
    ap.add_argument("--feature-dim", type=int, default=DEFAULT_FEATURE_DIM)
    ap.add_argument("--state-feat-dim", type=int, default=DEFAULT_STATE_FEAT_DIM)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N source entries")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build(args)


if __name__ == "__main__":
    main()
