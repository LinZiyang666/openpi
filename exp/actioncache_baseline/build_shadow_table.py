"""Shadow table: top-1 CP2 cosine of every decision of a teacher cohort.

"Cache in the shadow": the pure-teacher cohort H5s (the RIT-Pareto 150-episode
dev cohort per suite) are replayed offline — Stage 1 re-assembled from the
stored tensors, the backbone run with ``run_stage2_capture``, the CP2 key
built with the production builder — and searched against a CP2 library
(suite-wide, no task filter, exactly the deployed retrieval). The per-decision
raw cosine ``s`` is what ``export_arms.py`` inverts into IR-addressed cuts
(GST K=1).

Scoring uses one matrix product over the library keys (``cosine`` of the
normalised vectors); the first ``--backend-check`` rows are cross-checked
against the real ``InMemoryBackend`` search (same winner, ``2*score-1 == s``
to 1e-4) so the table and the deployed path cannot drift apart.

Output: JSONL rows ``{episode, task, step_idx, s_raw, winner_id, success}``
plus a sibling ``.record.json`` with library / cohort / projection identity.

Usage:
  uv run python -m exp.actioncache_baseline.build_shadow_table \\
      --cohort-h5-root <dir with the cohort H5s> --library-pkl <cp2>.pkl \\
      --config-name pi05_libero --checkpoint-dir <ckpt> --out-jsonl <path>
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import time

import h5py
import numpy as np
import torch

from exp.actioncache_baseline import libs
from exp.actioncache_baseline.verify_cp2_artifact import cp2_query_spec
from exp.actioncache_baseline.stage1_paths import STAGE1_PATHS, rebuild_stage1
from exp.common.build_in_memory_cache_artifact import (
    _load_pi05_for_llm_extract,
    _self_check_tokenizer_consistency,
)
from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.cp2_vlm_key_builder import CP2VlmTernaryKeyBuilder
from openpi.cache.types import CheckpointID

logger = logging.getLogger("actioncache_baseline.shadow")


def load_library_matrix(library_pkl: str) -> tuple[list[str], torch.Tensor, dict]:
    """``(ids, unit-normalised keys [N, d] float32, artifact dict without entries)``."""
    art = libs.load_pickle(library_pkl)
    ids, rows = [], []
    for e in art["entries"]:
        k = e.query_keys[libs.FIELD]
        rows.append(torch.as_tensor(np.asarray(k)).float())
        ids.append(e.id)
    mat = torch.stack(rows)
    mat = mat / mat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    meta = {k: v for k, v in art.items() if k not in ("entries", "library_stats")}
    return ids, mat, meta


def top1_cosine(mat: torch.Tensor, key: torch.Tensor) -> tuple[int, float]:
    q = key.float() / key.float().norm().clamp_min(1e-12)
    sims = mat @ q
    i = int(torch.argmax(sims))
    return i, float(sims[i])


def build(args: argparse.Namespace) -> dict:
    t0 = time.time()
    ids, mat, lib_meta = load_library_matrix(args.library_pkl)
    proj = lib_meta["projection"]
    d = int(lib_meta["vector_dims"][libs.FIELD])
    device = torch.device(args.device)
    mat = mat.to(device)
    # Fail closed unless the checkpoint we are about to load is the one the
    # library keys were built with (full-content weights digest, plan §3.7).
    model_binding = libs.assert_model_binding(lib_meta.get("model"), args.checkpoint_dir)
    # The shadow keys must come from the same Stage-1 path as the library keys.
    lib_path = lib_meta.get("stage1_path")
    if lib_path not in STAGE1_PATHS:
        raise SystemExit(f"library carries no valid stage1_path ({lib_path!r}); rebuild it with build_cp2_artifact")
    stage1_path = args.stage1_path or lib_path
    if stage1_path != lib_path:
        raise SystemExit(f"--stage1-path {stage1_path} != library stage1_path {lib_path}")
    model, tokenizer = _load_pi05_for_llm_extract(args.checkpoint_dir, args.config_name, args.device)
    model.eval()
    builder = CP2VlmTernaryKeyBuilder(seed=proj["seed"], d=proj["d"], p=proj["p"], input_dim=proj["D"])
    if builder.projection_meta() != proj:
        raise SystemExit("library projection metadata does not re-derive from its (seed, d, p, D)")

    backend = InMemoryBackend(vector_dims={libs.FIELD: d})
    backend.load_artifact(args.library_pkl)
    storage = CacheStorage(backend)

    h5_files = sorted(pathlib.Path(args.cohort_h5_root).rglob("*.h5"))
    if not h5_files:
        raise SystemExit(f"no H5 under {args.cohort_h5_root}")
    if args.limit_episodes:
        h5_files = h5_files[: args.limit_episodes]
    out_path = pathlib.Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    n_checked = 0
    cohort_manifest = []
    with torch.no_grad(), out_path.open("w", encoding="utf-8") as out:
        for i, h5_path in enumerate(h5_files):
            cohort_manifest.append({"path": str(h5_path), "sha256": libs.sha256_file(h5_path)})
            if i == 0:
                _self_check_tokenizer_consistency(h5_path, model, tokenizer, device)
            with h5py.File(h5_path, "r") as f:
                task = str(f.attrs.get("task", ""))
                success = bool(f.attrs.get("success", False))
                episode = h5_path.relative_to(pathlib.Path(args.cohort_h5_root)).with_suffix("").as_posix()
                for step_idx, group in libs.iter_steps(f):
                    stage1 = rebuild_stage1(group, task, tokenizer, model, device, stage1_path)
                    stage2 = model.run_stage2_capture(stage1)
                    builder.collect(CheckpointID.CP2, stage2=stage2)
                    key = builder.build(CheckpointID.CP2)[libs.FIELD]
                    builder.clear()
                    j, s_raw = top1_cosine(mat, key.to(device))
                    if n_checked < args.backend_check:
                        res = storage.search(cp2_query_spec(key))
                        if not res or res[0].id != ids[j] or abs(libs.theta_raw(float(res[0].score)) - s_raw) > 1e-4:
                            raise SystemExit(
                                f"backend cross-check failed at {episode} step {step_idx}: "
                                f"matrix winner {ids[j]} s={s_raw:.6f} vs backend {res[0].id if res else None} "
                                f"score={res[0].score if res else None}"
                            )
                        n_checked += 1
                    out.write(json.dumps({
                        "episode": episode, "task": task, "step_idx": step_idx,
                        "s_raw": s_raw, "winner_id": ids[j], "success": success,
                    }) + "\n")
                    n_rows += 1
            if (i + 1) % 10 == 0:
                logger.info("%d / %d episodes, %d rows (%.0fs)", i + 1, len(h5_files), n_rows, time.time() - t0)

    record = {
        "protocol": libs.PROTOCOL,
        "library_pkl": str(pathlib.Path(args.library_pkl).resolve()),
        "library_sha256": libs.sha256_file(args.library_pkl),
        "library_entries": len(ids),
        "projection": proj,
        "stage1_path": stage1_path,
        "cohort_root": str(pathlib.Path(args.cohort_h5_root).resolve()),
        "cohort_files": len(h5_files),
        "cohort_manifest_sha256": _manifest_digest(cohort_manifest),
        "n_rows": n_rows,
        "backend_checked_rows": n_checked,
        "config_name": args.config_name,
        "checkpoint_dir": str(pathlib.Path(args.checkpoint_dir).resolve()),
        "model": {**model_binding, "bound_to_library": True,
                  "library_model": lib_meta.get("model")},
        "git_commit": libs.git_commit(),
        "out_jsonl": str(out_path.resolve()),
        "out_jsonl_sha256": libs.sha256_file(out_path),
        "elapsed_s": round(time.time() - t0, 1),
    }
    libs.dump_json(out_path.with_suffix(".record.json"), record)
    return record


def _manifest_digest(rows: list[dict]) -> str:
    import hashlib

    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: r["path"]):
        h.update(f"{r['path']}:{r['sha256']}\n".encode())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort-h5-root", required=True)
    ap.add_argument("--library-pkl", required=True)
    ap.add_argument("--config-name", default="pi05_libero")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stage1-path", choices=STAGE1_PATHS, default="",
                    help="must equal the library's stage1_path (default: taken from the library)")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--backend-check", type=int, default=50, help="rows cross-checked against the real backend")
    ap.add_argument("--limit-episodes", type=int, default=0, help="debug: only the first N H5 files")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rec = build(args)
    print(json.dumps({k: rec[k] for k in ("out_jsonl", "n_rows", "backend_checked_rows", "elapsed_s")}))


if __name__ == "__main__":
    main()
