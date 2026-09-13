"""Shadow table for the GR00T CP2 arm: top-1 cosine of every cohort step against the CP2 library.

Same role as ``exp/actioncache_baseline/build_shadow_table.py`` (Pi0.5): the
threshold cuts of every arm are addressed on this distribution, so the keys
here must be produced by the exact online code path -- ``run_stage2_llm`` +
``run_cp2_key_source`` + ``GrootCP2TernaryKeyBuilder`` -- on the stage-1
sequences ``cp2_reconstruct`` rebuilds from the cohort HDF5. Only episodes
in the ``accepted_shadow_manifest.json`` that ``verify_shadow_h5`` produced
are read (never a raw glob of the attempt directories), each file's sha256 is
re-checked, and the checkpoint is bound to the library's ``weights_digest``
before the model is loaded.

Every row: ``{episode, task, subset, orig, step_idx, s_raw, winner_id,
success}``; the first ``--backend-check`` decisions are cross-checked
against a real ``InMemoryBackend`` search (``task_scoped: false``).

Usage:
  python -m exp.libero_groot.build_shadow_table_groot --suite libero_spatial \\
      --accepted-manifest <accepted_shadow_manifest.json> \\
      --library-pkl <cp2.pkl> --checkpoint <ckpt> --out-jsonl <shadow.jsonl>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pathlib
import time

import h5py
import numpy as np
import torch

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.groot.cp2_key_builder import KEY_BUILDER_TYPE, GrootCP2TernaryKeyBuilder
from openpi.cache.groot.staged import GrootStagedRunner
from openpi.cache.types import CheckpointID

from exp.actioncache_baseline import libs
from exp.actioncache_baseline.verify_cp2_artifact import cp2_query_spec
from exp.libero_groot.cp2_reconstruct import (
    STAGE1_PATH,
    TemplateCache,
    h5_task,
    load_groot_libero_policy,
    reconstruct_stage1,
)

logger = logging.getLogger("build_shadow_table_groot")
PROFILE = libs.GROOT_LIBERO


def load_library_matrix(library_pkl: str) -> tuple[list[str], torch.Tensor, dict]:
    """``(ids, unit-normalised keys [N, d] float32, artifact dict without entries)``."""
    art = libs.load_pickle(library_pkl)
    if art.get("key_builder_type") != KEY_BUILDER_TYPE or art.get("teacher") != PROFILE.name:
        raise SystemExit(f"{library_pkl} is not a {PROFILE.name} CP2 library "
                         f"(key_builder_type={art.get('key_builder_type')!r}, teacher={art.get('teacher')!r})")
    ids, rows = [], []
    for e in art["entries"]:
        if set(e.query_keys) != {libs.FIELD}:
            raise SystemExit(f"{library_pkl}: entry {e.id} keys {sorted(e.query_keys)} != [{libs.FIELD!r}]")
        rows.append(torch.as_tensor(np.asarray(e.query_keys[libs.FIELD])).float())
        ids.append(e.id)
    mat = torch.stack(rows)
    mat = mat / mat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    meta = {k: v for k, v in art.items() if k not in ("entries", "library_stats", "prompt_pool")}
    return ids, mat, meta


def top1_cosine(mat: torch.Tensor, key: torch.Tensor) -> tuple[int, float]:
    """``(row index, cosine)`` of the best library row for ``key`` over unit-normalised ``mat``."""
    q = key.float() / key.float().norm().clamp_min(1e-12)
    sims = mat @ q
    i = int(torch.argmax(sims))
    return i, float(sims[i])


def builder_from_meta(proj: dict) -> GrootCP2TernaryKeyBuilder:
    """The builder whose ``projection_meta()`` re-derives the library's ``projection`` block."""
    lay = proj["layout"]
    b = GrootCP2TernaryKeyBuilder(seed=proj["seed"], d=proj["d"], p=proj["p"], token_len=lay["token_len"],
                                  feature_dim=lay["feature_dim"], state_feat_dim=lay["state_feat_dim"])
    if b.projection_meta() != proj:
        raise SystemExit("library projection metadata does not re-derive from its (seed, d, p, layout)")
    return b


def build(args: argparse.Namespace) -> dict:
    """Replay the accepted cohort through the CP2 key path and write one row per decision.

    Inputs: the accepted cohort manifest (``verify_shadow_h5.py``, ``ok`` and
    task-map bound), the verified CP2 library and the checkpoint it was built
    from. Each row is ``{episode, task, task_id, subset, orig, step_idx, s_raw,
    winner_id, success}`` with ``s_raw`` the top-1 cosine over the whole
    library; the sidecar ``.record.json`` binds the table to the library,
    model, cohort and (when ``--limit-episodes`` was used) marks it
    ``limited`` so the exporter refuses it as a frozen cohort.
    """
    t0 = time.time()
    accepted_path = pathlib.Path(args.accepted_manifest)
    acc = json.loads(accepted_path.read_text(encoding="utf-8"))
    if acc.get("suite") != args.suite or not acc.get("ok") or acc.get("n_accepted") != acc.get("expected"):
        raise SystemExit(f"{accepted_path}: not an accepted, complete cohort for {args.suite}")
    if not acc.get("task_map_bound"):
        raise SystemExit(f"{accepted_path}: cohort was verified without the benchmark task map")
    ids, mat, lib_meta = load_library_matrix(args.library_pkl)
    if lib_meta.get("key_builder_type") != KEY_BUILDER_TYPE or lib_meta.get("teacher") != PROFILE.name:
        raise SystemExit(f"--library-pkl is not a {PROFILE.name} CP2 library")
    if lib_meta.get("stage1_path") != STAGE1_PATH:
        raise SystemExit(f"library stage1_path {lib_meta.get('stage1_path')!r} != {STAGE1_PATH!r}")
    model_binding = libs.assert_model_binding(lib_meta.get("model"), args.checkpoint)
    proj = lib_meta["projection"]
    d = int(lib_meta["vector_dims"][libs.FIELD])
    builder = builder_from_meta(proj)

    policy = load_groot_libero_policy(args.checkpoint, denoising_steps=args.denoising_steps, device=args.device)
    runner = GrootStagedRunner(policy.model)
    if runner.live_schedule().schedule_id != lib_meta.get("schedule_id"):
        raise SystemExit(f"served head runs {runner.live_schedule().schedule_id}, library {lib_meta.get('schedule_id')}")
    device = torch.device(args.device)
    mat = mat.to(device)
    backend = InMemoryBackend(vector_dims={libs.FIELD: d})
    backend.load_artifact(args.library_pkl)
    storage = CacheStorage(backend)
    templates = TemplateCache(policy, runner)

    out_path = pathlib.Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    episodes = list(acc["accepted"])
    if args.limit_episodes:
        episodes = episodes[: args.limit_episodes]
    n_rows = n_checked = 0
    cohort_digest = hashlib.sha256()
    with out_path.open("w", encoding="utf-8") as out:
        for i, ep in enumerate(episodes):
            h5_path = pathlib.Path(ep["h5"])
            sha = libs.sha256_file(h5_path)
            if sha != ep["h5_sha256"]:
                raise SystemExit(f"{h5_path}: sha256 changed since acceptance")
            cohort_digest.update(f"{ep['task_id']}:{ep['subset_init_state_idx']}:{sha}\n".encode())
            with h5py.File(h5_path, "r") as f:
                task = h5_task(f)
                if task != ep["task_language"]:
                    raise SystemExit(f"{h5_path}: instruction {task!r} != accepted {ep['task_language']!r}")
                with runner.session():
                    template = templates.get(task)
                for step_idx, group in libs.iter_steps(f):
                    # Model path inside the session; the key projection is
                    # float32 by contract. The cosine / backend search run
                    # *outside* it, as the served check() does -- under the
                    # session's autocast a matmul would be bf16 and the table
                    # would not be the deployed score.
                    with runner.session():
                        stage1 = reconstruct_stage1(template, group)
                        stage2 = runner.run_stage2_llm(stage1)
                        source = runner.run_cp2_key_source(stage2)
                        builder.collect(CheckpointID.CP2, cp2_source=source, stage2=stage2)
                        key = builder.build(CheckpointID.CP2)[libs.FIELD]
                        builder.clear()
                    source = None
                    j, s_raw = top1_cosine(mat, key.to(device))
                    if n_checked < args.backend_check:
                        res = storage.search(cp2_query_spec(key))
                        if not res or res[0].id != ids[j] or abs(libs.theta_raw(float(res[0].score)) - s_raw) > 1e-4:
                            raise SystemExit(
                                f"backend cross-check failed at {h5_path.name} step {step_idx}: matrix winner "
                                f"{ids[j]} s={s_raw:.6f} vs backend {res[0].id if res else None} "
                                f"score={res[0].score if res else None}")
                        n_checked += 1
                    out.write(json.dumps({
                        "episode": h5_path.stem, "task": task, "task_id": ep["task_id"],
                        "subset": ep["subset_init_state_idx"], "orig": ep["orig_init_state_idx"],
                        "step_idx": step_idx, "s_raw": s_raw, "winner_id": ids[j], "success": bool(ep["success"]),
                    }) + "\n")
                    n_rows += 1
            if (i + 1) % 10 == 0:
                logger.info("%d / %d episodes, %d rows (%.0fs)", i + 1, len(episodes), n_rows, time.time() - t0)
    record = {
        "protocol": libs.PROTOCOL, "teacher": PROFILE.name, "suite": args.suite,
        "library_pkl": str(pathlib.Path(args.library_pkl).resolve()),
        "library_sha256": libs.sha256_file(args.library_pkl), "library_entries": len(ids),
        "projection": proj, "stage1_path": STAGE1_PATH, "schedule_id": lib_meta.get("schedule_id"),
        "accepted_manifest": str(accepted_path.resolve()), "accepted_manifest_sha256": libs.sha256_file(accepted_path),
        "cohort_episodes": len(episodes), "cohort_expected": int(acc["expected"]),
        "limited": bool(args.limit_episodes), "complete": len(episodes) == int(acc["expected"]) and not args.limit_episodes,
        "cohort_manifest_sha256": cohort_digest.hexdigest(), "task_map_sha256": acc.get("task_map_sha256"),
        "n_rows": n_rows, "backend_checked_rows": n_checked,
        "checkpoint": str(pathlib.Path(args.checkpoint).resolve()),
        "model": {**model_binding, "bound_to_library": True, "library_model": lib_meta.get("model")},
        "git_commit": libs.git_commit(), "out_jsonl": str(out_path.resolve()),
        "out_jsonl_sha256": libs.sha256_file(out_path), "elapsed_s": round(time.time() - t0, 1),
    }
    libs.dump_json(out_path.with_suffix(".record.json"), record)
    logger.info("wrote %s (%d rows)", out_path, n_rows)
    return record


def main() -> None:
    """CLI entry: see the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True, choices=sorted(libs.SUITE_TAGS))
    ap.add_argument("--accepted-manifest", required=True)
    ap.add_argument("--library-pkl", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--denoising-steps", type=int, default=8)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--backend-check", type=int, default=50)
    ap.add_argument("--limit-episodes", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build(args)


if __name__ == "__main__":
    main()
