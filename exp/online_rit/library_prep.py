"""M0 assets from the S3 library alone: scales, checks, knots, S3b and init pools.

Three subcommands, none of which runs the model (the common score knots come
from the M1 table's fit half: ``fit_init_curves.py knots``):

  scales   ``update_scales.npz`` -- executed-dimension mask (``EXEC_DIMS``),
           the D-column mask/scale (library action chunks) and one mask/scale
           per tier index for the d column (library updates ``(x_{i+1}-x_i)*N``),
           each restricted to the executed dims and to dims whose library
           variance exceeds 1e-8. Also asserts every entry carries all N-1
           snapshots plus its chunk (``library_check.json``).
  s3b      a second library, same size and recipe as S3, sliced from the W13
           *full* artifact by whole trajectories the S3 library does not hold
           (seed 1); writes the pkl and a manifest with both sha256s.
  pools    the A_adapt / A_terminal init pools (25 + 25 per task, seed
           20260914) materialised from the official A pool.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import random

import numpy as np
import torch

from exp.online_rit.common import (
    DENOISING_STEPS,
    EXEC_DIMS,
    TIER_TS,
    canonical_sha256,
    sha256_file,
    tier_index_of,
    write_json,
)
from openpi.cache.types import DenoiseSchedule, groot_n15_schedule, schedule_from_id

VAR_EPS = 1e-8
S3B_SEED = 1
POOL_SEED = 20260914
POOL_PER_SIDE = 25


# ------------------------------------------------------------------
# scales
# ------------------------------------------------------------------


def library_check(entries, schedule: DenoiseSchedule) -> dict:
    """Every entry has all N-1 snapshots, a chunk, matching shapes and the schedule."""
    missing = 0
    bad_schedule = 0
    shapes: set[tuple] = set()
    for e in entries:
        pl = e.payload
        keys = sorted(round(float(k), 4) for k in (pl.intermediates or {}))
        if keys != list(schedule.timesteps) or pl.action_chunk is None:
            missing += 1
            continue
        if pl.schedule_id is not None and pl.schedule_id != schedule.schedule_id:
            bad_schedule += 1
        shapes.add(tuple(torch.as_tensor(pl.action_chunk).shape))
    return {
        "n_entries": len(entries),
        "n_missing_snapshots": missing,
        "n_bad_schedule": bad_schedule,
        "chunk_shapes": sorted(shapes),
        "schedule_id": schedule.schedule_id,
        "ok": missing == 0 and bad_schedule == 0 and len(shapes) == 1,
    }


def compute_update_scales(entries, schedule: DenoiseSchedule, *, exec_dims: int = EXEC_DIMS, tier_ts=TIER_TS) -> dict:
    """Masks and inverse-sigma scales for the D column and each tier's d column."""
    chunks = torch.stack([torch.as_tensor(e.payload.action_chunk, dtype=torch.float32) for e in entries])  # [N, H, D]
    dim = chunks.shape[-1]
    exec_mask = np.zeros(dim, dtype=bool)
    exec_mask[:exec_dims] = True
    out: dict = {"exec_mask": exec_mask}

    def _mask_scale(x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        flat = x.reshape(-1, dim)
        var = flat.var(dim=0, unbiased=False).numpy()
        sigma = np.sqrt(var)
        mask = exec_mask & (var > VAR_EPS)
        scale = np.zeros(dim, dtype=np.float32)
        scale[mask] = 1.0 / sigma[mask]
        return mask, scale

    out["mask_D"], out["scale_D"] = _mask_scale(chunks)
    n = float(schedule.num_steps)
    for t in tier_ts:
        i = tier_index_of(t, schedule)
        t4 = round(float(t), 4)
        ups = []
        for e in entries:
            pl = e.payload
            x_t = torch.as_tensor(pl.intermediates[t4], dtype=torch.float32)
            if i + 1 < schedule.num_steps:
                x_next = torch.as_tensor(pl.intermediates[schedule.snapshot_t(i + 1)], dtype=torch.float32)
            else:
                x_next = torch.as_tensor(pl.action_chunk, dtype=torch.float32)
            ups.append((x_next - x_t) * n)
        out[f"mask_d_{i}"], out[f"scale_d_{i}"] = _mask_scale(torch.stack(ups))
    return out


def write_scales_npz(path: pathlib.Path, arrays: dict, meta: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, meta_json=np.array(json.dumps(meta, sort_keys=True)), **arrays)
    return sha256_file(path)


def _load_pkl(path: str | pathlib.Path) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _schedule_of(artifact: dict) -> DenoiseSchedule:
    sid = artifact.get("schedule_id")
    if sid is None:
        raise SystemExit("artifact carries no schedule_id")
    return schedule_from_id(str(sid))


def cmd_scales(args) -> None:
    art = _load_pkl(args.library_pkl)
    schedule = _schedule_of(art)
    if schedule != groot_n15_schedule(args.denoising_steps):
        raise SystemExit(f"library schedule {schedule.schedule_id} != k{args.denoising_steps}")
    entries = art["entries"]
    check = library_check(entries, schedule)
    out_dir = pathlib.Path(args.out_dir)
    write_json(out_dir / "library_check.json", {**check, "library_sha256": sha256_file(args.library_pkl)})
    if not check["ok"]:
        raise SystemExit(f"library check failed: {check}")
    arrays = compute_update_scales(entries, schedule, exec_dims=args.exec_dims)
    meta = {
        "library_sha256": sha256_file(args.library_pkl),
        "library_path": str(args.library_pkl),
        "schedule_id": schedule.schedule_id,
        "exec_dims": int(args.exec_dims),
        "var_eps": VAR_EPS,
        "tier_ts": [float(t) for t in TIER_TS],
        "n_entries": len(entries),
        "active_dims": {k: int(v.sum()) for k, v in arrays.items() if k.startswith("mask")},
    }
    sha = write_scales_npz(out_dir / "update_scales.npz", arrays, meta)
    print(f"wrote {out_dir / 'update_scales.npz'} sha256={sha} active={meta['active_dims']}")


# ------------------------------------------------------------------
# s3b
# ------------------------------------------------------------------


def _by_trajectory(entries) -> dict[str, list]:
    out: dict[str, list] = {}
    for e in entries:
        out.setdefault(str(e.trajectory_id), []).append(e)
    return out


def _task_of_entry(e) -> str:
    return str(getattr(e.payload, "task_key", "") or "")


def sample_s3b(full_entries, s3_trajectories: set[str], *, per_task: int, seed: int) -> dict[str, list[str]]:
    """Per task, ``per_task`` full-artifact trajectories outside S3, seeded and sorted."""
    by_task: dict[str, list[str]] = {}
    for traj, ents in _by_trajectory(full_entries).items():
        if traj in s3_trajectories:
            continue
        by_task.setdefault(_task_of_entry(ents[0]), []).append(traj)
    picked: dict[str, list[str]] = {}
    for task in sorted(by_task):
        cands = sorted(by_task[task])
        if len(cands) < per_task:
            raise SystemExit(f"task {task!r}: only {len(cands)} trajectories outside S3, need {per_task}")
        rng = random.Random(f"{seed}:{task}")
        rng.shuffle(cands)
        picked[task] = sorted(cands[:per_task])
    return picked


def cmd_s3b(args) -> None:
    full = _load_pkl(args.full_pkl)
    s3 = _load_pkl(args.s3_pkl)
    if full.get("schedule_id") != s3.get("schedule_id"):
        raise SystemExit("full and S3 artifacts carry different schedules")
    s3_traj = {str(e.trajectory_id) for e in s3["entries"]}
    per_task = len(s3_traj) // max(1, len({_task_of_entry(e) for e in s3["entries"]}))
    picked = sample_s3b(full["entries"], s3_traj, per_task=per_task, seed=args.seed)
    kept = {t for ts in picked.values() for t in ts}
    if kept & s3_traj:
        raise SystemExit("S3b overlaps S3")
    entries = [e for e in full["entries"] if str(e.trajectory_id) in kept]
    out_pkl = pathlib.Path(args.out_pkl)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with out_pkl.open("wb") as fh:
        pickle.dump({**full, "entries": entries}, fh, protocol=4)
    manifest = {
        "protocol": "online_rit_s3b_v1",
        "seed": int(args.seed),
        "per_task": per_task,
        "trajectories": picked,
        "n_entries": len(entries),
        "s3_pkl": str(args.s3_pkl),
        "s3_sha256": sha256_file(args.s3_pkl),
        "full_pkl": str(args.full_pkl),
        "full_sha256": sha256_file(args.full_pkl),
        "s3b_pkl": str(out_pkl),
        "s3b_sha256": sha256_file(out_pkl),
        "overlap_with_s3": 0,
    }
    write_json(out_pkl.with_suffix(".manifest.json"), manifest)
    print(f"wrote {out_pkl}: {len(entries)} entries over {len(kept)} trajectories, sha256={manifest['s3b_sha256']}")


# ------------------------------------------------------------------
# pools
# ------------------------------------------------------------------


def split_indices(n: int, per_side: int, seed: int, task_id: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng([int(seed), int(task_id)])
    perm = [int(i) for i in rng.permutation(n)]
    if 2 * per_side > n:
        raise SystemExit(f"cannot split {n} inits into two sides of {per_side}")
    return sorted(perm[:per_side]), sorted(perm[per_side : 2 * per_side])


def cmd_pools(args) -> None:
    from exp.dispatch_surface.split_init_pools import OFFICIAL_PER_TASK, materialize_pool

    order = json.loads(pathlib.Path(args.task_order).read_text(encoding="utf-8"))
    assignment = {int(t): {"task_name": v["task_name"]} for t, v in order["assignment"].items()}
    adapt: dict[int, dict] = {}
    terminal: dict[int, dict] = {}
    for tid in sorted(assignment):
        a, b = split_indices(OFFICIAL_PER_TASK, args.per_side, args.seed, tid)
        adapt[tid] = {"task_name": assignment[tid]["task_name"], "adapt": a}
        terminal[tid] = {"task_name": assignment[tid]["task_name"], "terminal": b}
    out_dir = pathlib.Path(args.out_dir)
    for name in ("adapt_pool", "terminal_pool", "smoke_pool"):
        d = out_dir / name
        if d.exists() and any(d.iterdir()):
            raise SystemExit(f"pool dir must be empty or absent: {d}")
    dig_a = materialize_pool(pathlib.Path(args.apool_dir), out_dir / "adapt_pool", adapt, ["adapt"])
    dig_b = materialize_pool(pathlib.Path(args.apool_dir), out_dir / "terminal_pool", terminal, ["terminal"])
    smoke = {t: {"task_name": v["task_name"], "smoke": v["adapt"][:1]} for t, v in adapt.items()}
    dig_smoke = materialize_pool(pathlib.Path(args.apool_dir), out_dir / "smoke_pool", smoke, ["smoke"])
    # Frozen records in the run_gtp ``--apool-record`` schema, sized to the subset
    # (run_gtp verifies them with ``expect_per_task=--trials``).
    from exp.ablation_study.cache_size.run_size_eval import rehash_apool

    import yaml as _yaml

    records = {}
    for name in ("adapt", "terminal", "smoke"):
        pool_dir = out_dir / f"{name}_pool"
        rec = rehash_apool(pool_dir, expect_per_task=1 if name == "smoke" else args.per_side)
        record = {"suite": order.get("suite"), "apool_dir": str(pool_dir.resolve()), **rec}
        records[name] = rec
        (out_dir / f"apool_{name}.yaml").write_text(_yaml.safe_dump(record, sort_keys=False), encoding="utf-8")
    manifest = {
        "protocol": "online_rit_init_pools_v1",
        "suite": order.get("suite"),
        "seed": int(args.seed),
        "per_side": int(args.per_side),
        "adapt": {str(t): v["adapt"] for t, v in adapt.items()},
        "terminal": {str(t): v["terminal"] for t, v in terminal.items()},
        "task_names": {str(t): v["task_name"] for t, v in assignment.items()},
        "smoke": {str(t): v["smoke"] for t, v in smoke.items()},
        "pool_digests": {"adapt": dig_a, "terminal": dig_b, "smoke": dig_smoke},
        "pool_records": records,
        "parent_pool_sha256": rehash_apool(pathlib.Path(args.apool_dir))["rollup_sha256"],
        "apool_dir": str(args.apool_dir),
        "task_order_sha256": sha256_file(args.task_order),
    }
    for name, rec in records.items():
        rec["index_map_sha256"] = canonical_sha256(manifest[name])
        rec["record_sha256"] = sha256_file(out_dir / f"apool_{name}.yaml")
    for t in adapt:
        assert not set(adapt[t]["adapt"]) & set(terminal[t]["terminal"])
    write_json(out_dir / "init_pools_manifest.json", manifest)
    print(f"wrote {out_dir / 'init_pools_manifest.json'} ({len(adapt)} tasks x {args.per_side}+{args.per_side})")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scales")
    p.add_argument("--library-pkl", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--denoising-steps", type=int, default=DENOISING_STEPS)
    p.add_argument("--exec-dims", type=int, default=EXEC_DIMS)
    p.set_defaults(fn=cmd_scales)

    p = sub.add_parser("s3b")
    p.add_argument("--full-pkl", required=True)
    p.add_argument("--s3-pkl", required=True)
    p.add_argument("--out-pkl", required=True)
    p.add_argument("--seed", type=int, default=S3B_SEED)
    p.set_defaults(fn=cmd_s3b)

    p = sub.add_parser("pools")
    p.add_argument("--task-order", required=True, help="exp/rit_pareto/config/task_order_<suite>.json")
    p.add_argument("--apool-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=POOL_SEED)
    p.add_argument("--per-side", type=int, default=POOL_PER_SIDE)
    p.set_defaults(fn=cmd_pools)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
