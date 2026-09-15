"""M1 table: continuation disagreement and executed-dim deviation per stored decision.

Reuses the LOTO line's reconstruction and retrieval helpers
(``exp.rit_loto.build_loto_table``): the same stage-1 rebuild from the W13
HDF5 slices, the same production search with the template's enabled fields
(disabled ones at weight 0), and the same leave-one-trajectory-out winner
rule. What is new is the label: for the winner's snapshot at every tier
(t = 0.875 / 0.75 / 0.5) the loop is resumed **once** with the first step
captured, so one continuation yields both

* ``d_<i>``  -- the continuation disagreement (first-step update against the
  library's stored update, executed dims, per-tier scale), and
* ``y_rem<k>`` -- the old method's deviation of the continued chunk from the
  teacher's ``clean_action`` on the executed-dim scale (``m_D`` / ``W_D``);
  ``y_full`` is the library end chunk against the same reference, and the
  ``*_legacy`` columns repeat the D labels with the LOTO helper's variance
  mask so the two lines can be cross-checked on identical inputs.

Rows cover every out-of-library trajectory (successes and failures; this is
the calibration query set, ``in_library: false``) and, for trajectories that
are in the library, ``in_library: true`` diagnostic rows carrying the
``d_self`` columns (the query's own stored decision resumed under its own
conditioning: the numeric floor of the signal). Rows without a candidate are
kept with ``s: null``. Every trajectory is stamped ``split: fit|test`` by task
x success stratified seeded assignment (whole trajectories, never split).

Numerical gate (plan §4 M1-3). ``--parity-only`` draws 200 stratified decisions
(20 per task) and replays each from its own stored noise / snapshots at full,
warm875, warm75 and warm50, measuring ``parity_D`` against the stored
``clean_action`` on the same executed-dim ``W_D`` / ``m_D`` / ``H_EXEC`` scale
as the table, and the two-noise floor ``D(ref1, ref2)`` on that same scale.
PASS requires, for **all four** replays, ``p90(parity_D) <= 0.1 x median floor``
(all zero if the floor is zero), finite values, the full sample and every
task; the gate is bound to the library / scales / checkpoint / h_exec /
schedule identity and a full table run refuses to start without a matching
PASS. Smoke runs (``--limit-episodes``) may pass ``--allow-ungated-smoke``.

Runs in the GR00T island venv. Usage::

  python -m exp.online_rit.build_disagreement_table --suite libero_10 --parity-only \\
      --corpus-dir ... --library-pkl ... --template-yaml ... --scales ... --checkpoint ... --out-dir ...
  python -m exp.online_rit.build_disagreement_table --suite libero_10 --parity-gate <out>/parity_gate.json ...
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import time

import h5py
import numpy as np
import torch

from exp.online_rit.common import DENOISING_STEPS, H_EXEC, TIER_TS, canonical_sha256, sha256_file, tier_index_of, write_json
from exp.online_rit.provenance import GATE_IDENTITY_KEYS, validate_parity
from exp.rit_loto.build_loto_table import (
    build_identity,
    build_retrieval,
    build_storage,
    checkpoint_identity,
    corpus_files,
    corpus_manifest,
    enabled_fields,
    episode_attrs,
    h5_chunk,
    load_library,
    load_template,
    loto_winner,
    make_noise,
    make_query_builder,
    query_keys_for,
    search,
    stage2_in_session,
    step_groups,
    stratified_decisions,
    to_chunk,
)
from exp.robocasa365.rit_shadow import library_action_weights
from openpi.cache.components.online_rit import continuation_disagreement, load_update_scales, reference_update
from openpi.cache.components.surface_judge import weighted_chunk_deviation
from openpi.cache.types import groot_n15_schedule
from openpi.collect.h5_intermediates import episode_schedule

logger = logging.getLogger("online_rit.table")
SPLIT_SEED = 20260914
PARITY_SEED = 20260914
PARITY_SAMPLE = 200
PARITY_MIN_PER_TASK = 20
PARITY_RATIO = 0.1
NUM_TASKS = 10
PARITY_REPLAYS = ("full", "warm875", "warm750", "warm500")
NOISE_SEEDS = (11, 23)


def assign_splits(files: list[pathlib.Path], seed: int) -> dict[str, str]:
    """Whole-trajectory fit/test halves, stratified by (task_id, success)."""
    strata: dict[tuple, list[str]] = {}
    for p in files:
        with h5py.File(p, "r") as f:
            a = episode_attrs(f, p.stem)
        strata.setdefault((a["task_id"], bool(a["success"])), []).append(p.stem)
    out: dict[str, str] = {}
    for key in sorted(strata):
        stems = sorted(strata[key])
        rng = np.random.default_rng([int(seed), int(key[0]), int(key[1])])
        perm = rng.permutation(len(stems))
        half = len(stems) // 2
        for j, idx in enumerate(perm):
            out[stems[idx]] = "fit" if j < half else "test"
    return out


def _scales_for_D(npz_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(npz_path, allow_pickle=False) as npz:
        return torch.as_tensor(np.asarray(npz["scale_D"], np.float32)), torch.as_tensor(np.asarray(npz["mask_D"], bool))


# ------------------------------------------------------------------
# labels
# ------------------------------------------------------------------


def label_decision(runner, templates, task, group, payload, *, schedule, scales, masks, w_D, mask_D, w_legacy, mask_legacy, h_exec, own_payload=None) -> dict:
    """One row's labels: d per tier, D per tier (executed and legacy scale), and d_self if own_payload given."""
    ref = h5_chunk(group, "clean_action")
    row: dict = {}
    n = float(schedule.num_steps)
    with runner.session():
        _, stage2 = stage2_in_session(runner, templates, task, group)
        for t in TIER_TS:
            i = tier_index_of(t, schedule)
            out = runner.run_stage3_from(stage2, payload.intermediates[t], t, schedule=schedule, capture_first_step=True)
            chunk = to_chunk(out.action_pred)
            u_now = (to_chunk(out.first_step_x) - to_chunk(out.first_step_input)) * n
            u_ref = reference_update(payload, t, schedule)
            try:
                row[f"d_{i}"] = continuation_disagreement(u_now, u_ref, scales[i], masks[i], h_exec)
            except ValueError:
                row[f"d_{i}"] = None
            rem = schedule.remaining_steps(t)
            row[f"y_rem{rem}"] = weighted_chunk_deviation(chunk, ref, w_D, mask_D, h_exec)
            row[f"y_rem{rem}_legacy"] = weighted_chunk_deviation(chunk, ref, w_legacy, mask_legacy, h_exec)
        if own_payload is not None:
            for t in TIER_TS:
                i = tier_index_of(t, schedule)
                out = runner.run_stage3_from(stage2, own_payload.intermediates[t], t, schedule=schedule, capture_first_step=True)
                u_now = (to_chunk(out.first_step_x) - to_chunk(out.first_step_input)) * n
                u_ref = reference_update(own_payload, t, schedule)
                try:
                    row[f"d_self_{i}"] = continuation_disagreement(u_now, u_ref, scales[i], masks[i], h_exec)
                except ValueError:
                    row[f"d_self_{i}"] = None
    end = torch.as_tensor(payload.action_chunk, dtype=torch.float32)
    row["y_full"] = weighted_chunk_deviation(end, ref, w_D, mask_D, h_exec)
    row["y_full_legacy"] = weighted_chunk_deviation(end, ref, w_legacy, mask_legacy, h_exec)
    return row


# ------------------------------------------------------------------
# parity gate (self-replay + two-noise floor on the executed-dim scale)
# ------------------------------------------------------------------


def parity_row(runner, templates, task: str, group, *, schedule, w_D, mask_D, h_exec: int) -> dict:
    """Replay full / warm875 / warm750 / warm500 from the step's OWN noise and snapshots, plus the two-noise floor."""
    clean = h5_chunk(group, "clean_action")
    shape = (1,) + tuple(clean.shape)
    noise0 = h5_chunk(group, "noise_action_0")[None]
    snaps = {t: h5_chunk(group, f"noise_action_{schedule.snapshot_index(t)}") for t in TIER_TS}
    with runner.session():
        _, stage2 = stage2_in_session(runner, templates, task, group)
        replay = {"full": runner.run_stage3(stage2, noise=noise0).action_pred}
        for name, t in zip(PARITY_REPLAYS[1:], TIER_TS):
            replay[name] = runner.run_stage3_from(stage2, snaps[t], t, schedule=schedule).action_pred
        ref1 = runner.run_stage3(stage2, noise=make_noise(NOISE_SEEDS[0], shape)).action_pred
        ref2 = runner.run_stage3(stage2, noise=make_noise(NOISE_SEEDS[1], shape)).action_pred
    out = {}
    for name in PARITY_REPLAYS:
        out[f"parity_D_{name}"] = weighted_chunk_deviation(to_chunk(replay[name]), clean, w_D, mask_D, h_exec)
    out["floor_D_ref1_ref2"] = weighted_chunk_deviation(to_chunk(ref1), to_chunk(ref2), w_D, mask_D, h_exec)
    return out


def parity_gate(rows: list[dict], identity: dict, *, ratio: float = PARITY_RATIO, min_rows: int = PARITY_SAMPLE, min_per_task: int = PARITY_MIN_PER_TASK, num_tasks: int = NUM_TASKS) -> dict:
    """PASS iff every replay's p90(parity_D) <= ratio x median(floor) on a full, finite, every-task sample."""
    reasons: list[str] = []
    n = len(rows)
    per_task: dict[int, int] = {}
    for r in rows:
        per_task[int(r["task_id"])] = per_task.get(int(r["task_id"]), 0) + 1
    if n < min_rows:
        reasons.append(f"parity sample has {n} rows < {min_rows}")
    missing = sorted(set(range(num_tasks)) - set(per_task))
    if missing:
        reasons.append(f"tasks without parity rows: {missing}")
    thin = {t: c for t, c in per_task.items() if c < min_per_task}
    if thin:
        reasons.append(f"tasks below {min_per_task} parity rows: {thin}")
    floor_vals = np.array([float(r["floor_D_ref1_ref2"]) for r in rows], dtype=np.float64) if n else np.array([])
    if floor_vals.size == 0 or not np.isfinite(floor_vals).all():
        reasons.append("noise floor sample is empty or non-finite")
        floor = float("nan")
    else:
        floor = float(np.quantile(floor_vals, 0.5))
    threshold = ratio * floor if np.isfinite(floor) else float("nan")
    replays: dict[str, dict] = {}
    for name in PARITY_REPLAYS:
        vals = np.array([float(r[f"parity_D_{name}"]) for r in rows], dtype=np.float64) if n else np.array([])
        stats = {"n": int(vals.size)}
        if vals.size == 0 or not np.isfinite(vals).all():
            reasons.append(f"parity_D_{name} is empty or non-finite")
        else:
            stats.update(p50=float(np.quantile(vals, 0.5)), p90=float(np.quantile(vals, 0.9)), max=float(vals.max()))
            if np.isfinite(floor):
                if floor == 0.0:
                    if float(vals.max()) > 0.0:
                        reasons.append(f"zero noise floor but parity_D_{name} max {vals.max():.3g} > 0")
                elif stats["p90"] > threshold:
                    reasons.append(f"parity_D_{name} p90 {stats['p90']:.4g} > {threshold:.4g} (= {ratio} x floor {floor:.4g})")
        replays[name] = stats
    return {
        "status": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "ratio": ratio,
        "floor_median": None if not np.isfinite(floor) else floor,
        "floor_p90": None if floor_vals.size == 0 or not np.isfinite(floor_vals).all() else float(np.quantile(floor_vals, 0.9)),
        "threshold": None if not np.isfinite(threshold) else threshold,
        "n_rows": n,
        "per_task_rows": {str(k): v for k, v in sorted(per_task.items())},
        "replays": replays,
        "identity": {k: identity.get(k) for k in GATE_IDENTITY_KEYS},
        "sample": [{"trajectory_id": r["trajectory_id"], "decision_id": r["decision_id"]} for r in rows],
    }


def require_parity_pass(gate_path: str, identity: dict) -> str:
    """Refuse anything but an explicit PASS bound to this run's identity; returns the gate sha."""
    path = pathlib.Path(gate_path)
    if not path.is_file():
        raise SystemExit(f"parity gate missing: {path}")
    gate = json.loads(path.read_text(encoding="utf-8"))
    validate_parity(gate, identity)
    return sha256_file(path)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--library-pkl", required=True)
    ap.add_argument("--template-yaml", required=True)
    ap.add_argument("--scales", required=True, help="update_scales.npz from library_prep scales")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--denoising-steps", type=int, default=DENOISING_STEPS)
    ap.add_argument("--h-exec", type=int, default=H_EXEC)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--task-ids", default="", help="comma list; empty = all")
    ap.add_argument("--limit-episodes", type=int, default=0, help="smoke: first N files only")
    ap.add_argument("--allow-ungated-smoke", action="store_true", help="permit a --limit-episodes run without a PASS parity gate")
    ap.add_argument("--parity-only", action="store_true", help="draw the parity sample, write parity_gate.json, exit")
    ap.add_argument("--parity-gate", default="", help="parity_gate.json that must be PASS for a full run")
    ap.add_argument("--parity-sample", type=int, default=PARITY_SAMPLE)
    ap.add_argument("--parity-seed", type=int, default=PARITY_SEED)
    ap.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-every", type=int, default=500)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from exp.rit_loto.build_loto_table import ModelSide

    schedule = groot_n15_schedule(args.denoising_steps)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg, cfg_used = load_template(args.template_yaml, args.library_pkl, out_dir)
    library = load_library(args.library_pkl, expected_schedule=schedule, warm_ts=TIER_TS)
    tier_indices = [tier_index_of(t, schedule) for t in TIER_TS]
    scales, masks, scales_meta, scales_sha = load_update_scales(args.scales, tier_indices)
    if scales_meta.get("library_sha256") != library.sha256:
        raise SystemExit("update_scales.npz was computed on a different library")
    w_D, mask_D = _scales_for_D(args.scales)
    w_legacy, mask_legacy = library_action_weights(args.library_pkl)
    w_legacy = torch.as_tensor(np.asarray(w_legacy), dtype=torch.float32)
    mask_legacy = torch.as_tensor(np.asarray(mask_legacy), dtype=torch.bool)
    enabled, weights = enabled_fields(cfg)
    files = corpus_files(args.corpus_dir)
    task_filter = {int(x) for x in args.task_ids.split(",") if x.strip()} or None
    corpus = corpus_manifest(files)
    ckpt = checkpoint_identity(args.checkpoint)
    identity = build_identity(
        suite=args.suite, library=library, library_path=args.library_pkl, template_path=args.template_yaml,
        ckpt=ckpt, corpus=corpus, corpus_dir=args.corpus_dir, w=w_D, mask=mask_D, h_exec=args.h_exec,
        schedule=schedule, warm_ts=TIER_TS, enabled=enabled, weights=weights,
    )
    identity["scales_sha256"] = scales_sha
    identity["split_seed"] = int(args.split_seed)
    root = pathlib.Path(__file__).resolve().parents[2]
    identity["label_code_sha256"] = canonical_sha256({
        name: sha256_file(root / name) for name in (
            "exp/online_rit/build_disagreement_table.py", "exp/rit_loto/build_loto_table.py",
            "src/openpi/cache/groot/staged.py", "src/openpi/cache/components/online_rit.py",
        )
    })

    model = ModelSide(args.checkpoint, denoising_steps=args.denoising_steps, device=args.device)
    if model.schedule != schedule:
        raise SystemExit(f"live schedule {model.schedule.schedule_id} != {schedule.schedule_id}")
    runner, templates = model.runner, model.templates

    # -------------------------------------------------------------- parity
    if args.parity_only:
        sample = stratified_decisions(files, args.parity_sample // NUM_TASKS, args.parity_seed)
        by_stem: dict[str, list[dict]] = {}
        for d in sample:
            by_stem.setdefault(d["trajectory_id"], []).append(d)
        rows: list[dict] = []
        t0 = time.time()
        for stem, picks in by_stem.items():
            with h5py.File(pathlib.Path(args.corpus_dir) / f"{stem}.h5", "r") as f:
                if episode_schedule(f) != schedule:
                    raise SystemExit(f"{stem}: file schedule differs from {schedule.schedule_id}")
                for d in picks:
                    g = f[f"step_{d['decision_id']:04d}"]
                    rows.append({**d, **parity_row(runner, templates, d["task"], g, schedule=schedule, w_D=w_D, mask_D=mask_D, h_exec=args.h_exec)})
        logger.info("parity sample: %d rows in %.0fs", len(rows), time.time() - t0)
        write_json(out_dir / "parity_sample.json", rows)
        gate = parity_gate(rows, identity)
        gate["parity_sample_path"] = str(out_dir / "parity_sample.json")
        write_json(out_dir / "parity_gate.json", gate)
        logger.info("parity gate: %s %s", gate["status"], gate["reasons"])
        print(f"PARITY_GATE={gate['status']}")
        return

    # -------------------------------------------------------------- full table
    smoke = args.limit_episodes > 0
    gate_sha = None
    if args.parity_gate:
        gate_sha = require_parity_pass(args.parity_gate, identity)
    elif not (smoke and args.allow_ungated_smoke):
        raise SystemExit("a full run needs --parity-gate <PASS gate>; smoke runs may pass --allow-ungated-smoke")

    splits = assign_splits(files, args.split_seed)
    storage = build_storage(cfg, library.entries)
    strategy = build_retrieval(cfg, storage, weights, n_hint=max(library.per_task_entries.values()))
    builder = make_query_builder(cfg.key_builder.type)
    own_index = {(library.traj_of[e.id], int(e.step_idx)): e for e in library.entries if e.step_idx is not None}

    selected = []
    for p in files:
        with h5py.File(p, "r") as f:
            tid = episode_attrs(f, p.stem)["task_id"]
        if task_filter is None or tid in task_filter:
            selected.append(p)
    if smoke:
        selected = selected[: args.limit_episodes]
    out_path = out_dir / ("disagreement_table.smoke.jsonl" if smoke else "disagreement_table.jsonl")
    stats = {"episodes": 0, "rows": 0, "no_candidate": 0, "calibration_rows": 0, "in_library_rows": 0, "d_self_rows": 0, "rejected_d": 0}
    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as out:
        for p in selected:
            with h5py.File(p, "r") as f:
                a = episode_attrs(f, p.stem)
                if episode_schedule(f) != schedule:
                    raise SystemExit(f"{p.stem}: file schedule differs from {schedule.schedule_id}")
                in_library = p.stem in library.trajectory_ids
                stats["episodes"] += 1
                for step_idx, name in step_groups(f):
                    g = f[name]
                    base = {
                        "suite": args.suite, "trajectory_id": p.stem, "episode_id": a["episode_id"], "decision_id": step_idx,
                        "task": a["task"], "task_id": a["task_id"], "orig_init_state_idx": a["orig_init_state_idx"],
                        "in_library": in_library, "episode_success": a["success"], "split": splits[p.stem],
                        "library_sha256": library.sha256,
                    }
                    keys = query_keys_for(builder, g, enabled)
                    hits = search(strategy, keys, step_idx, a["task"])
                    cand = None
                    if hits:
                        try:
                            cand, s, skipped = loto_winner(hits, p.stem, in_library, library.traj_of)
                        except RuntimeError:
                            cand = None
                    if cand is None:
                        stats["no_candidate"] += 1
                        row = {**base, "s": None, "candidate_id": None, "no_candidate": "no_hit" if not hits else "only_self"}
                    else:
                        own = own_index.get((p.stem, step_idx)) if in_library else None
                        labels = label_decision(
                            runner, templates, a["task"], g, library.by_id[cand].payload, schedule=schedule, scales=scales, masks=masks,
                            w_D=w_D, mask_D=mask_D, w_legacy=w_legacy, mask_legacy=mask_legacy, h_exec=args.h_exec,
                            own_payload=own.payload if own is not None else None,
                        )
                        stats["rejected_d"] += sum(1 for k, v in labels.items() if k.startswith("d_") and v is None)
                        stats["d_self_rows"] += int(own is not None)
                        row = {**base, "s": s, "candidate_id": cand, "n_self_skipped": skipped, **labels}
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stats["rows"] += 1
                    stats["in_library_rows"] += int(in_library)
                    stats["calibration_rows"] += int(not in_library)
                    if stats["rows"] % args.log_every == 0:
                        el = time.time() - t0
                        logger.info("%d rows / %d episodes in %.0fs (%.1f rows/s)", stats["rows"], stats["episodes"], el, stats["rows"] / max(el, 1e-9))
            out.flush()
    record = {
        "protocol": "online_rit_table_v2", "smoke": smoke or task_filter is not None, "identity": identity, "stats": stats,
        "splits": {"fit": sum(v == "fit" for v in splits.values()), "test": sum(v == "test" for v in splits.values())},
        "parity_gate": args.parity_gate or None, "parity_gate_sha256": gate_sha,
        "config_used": str(cfg_used), "out_jsonl": str(out_path), "out_sha256": sha256_file(out_path),
        "splits_sha256": canonical_sha256(splits),
        "elapsed_s": time.time() - t0, "python": sys.version.split()[0], "torch": torch.__version__,
    }
    write_json(pathlib.Path(str(out_path) + ".record.json"), record)
    write_json(out_dir / "splits.json", splits)
    logger.info("wrote %s: %s", out_path, stats)


if __name__ == "__main__":
    main()
