"""Closed-loop verification of the offline RIT calibration (LIBERO-10, K=2 arm).

Every stage binds to the *frozen run record* written by ``emit_verify_arm arm``
(``frozen_run.json``): the fits digest and table identity it carries, the arm
yaml digest, the pool manifest digest and the run tags. A stage that cannot
prove it is looking at that run stops with the discrepancy printed.

``collect``  Merge the server-side logs of the formal run: every HDF5 episode
             must be on the pool manifest (exactly the frozen 10 x 5), appear
             once, be finished, carry the frozen identity attrs, hold exactly the
             step groups its sidecar rows name (decision ids 0..n-1, control
             step = H_exec x decision, same task / outcome), and hold the fields
             the offline replay needs. Then draw the sample: ``base`` = uniform
             without replacement over all decisions, plus every remaining WARM
             decision as ``warm_extra``.

``label``    Island / GPU. Binds library / checkpoint / template / W / H_exec /
             schedule to the frozen identity, then for every sampled decision
             retrieves the candidate offline on the full S3 library (also for
             MISS and gate-skip rows), confirms it against the online winner
             where one exists, draws a row-stable reference noise, runs full
             inference for the reference and fills BOTH K=2 labels from the
             plan's table. A ``--limit`` run is a *partial* product under its
             own file name and can never feed a report.

``report``   Joins the labelled rows against the sampling manifest (every sampled
             key labelled exactly once, nothing else), rejects rows that have a
             candidate but lack a finite label, then per dispatched tier computes
             the decision-weighted exceedance of the frozen curve on in-support
             rows, a task-stratified episode cluster bootstrap over the frozen
             episode list, the information / degeneracy gates and the three-way
             pre-registered reading. Also the joint K=2 re-fit on ``base`` rows,
             the realised inference ratio of the whole run under the measured
             cost, and the success count.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time
from typing import Any

import numpy as np

from exp.libero_groot.emit_rit_arms import WARM_TS, load_cost
from exp.rit_pareto.rit_k import predict
from exp.robocasa365 import emit_rit_rc as er
from exp.robocasa365 import rit_cost_rc as rc

from exp.rit_loto.build_loto_table import (
    NOISE_CONTRACT,
    build_retrieval,
    build_storage,
    checkpoint_identity,
    code_sha256,
    derive_seed,
    enabled_fields,
    git_commit,
    h5_chunk,
    load_library,
    load_template,
    loto_winner,
    make_noise,
    make_query_builder,
    query_keys_for,
    read_jsonl,
    require_code_identity,
    search,
    sha256_file,
    stage2_in_session,
    to_chunk,
    weights_sha256,
    write_jsonl,
)
from exp.rit_loto.emit_verify_arm import load_frozen_record, validate_pool_manifest
from exp.rit_loto.fit_loto import _cost_to_json, deserialize_fit
from exp.rit_loto.loto_logger import SIDECAR_NAME

VERIFY_WARM_T = 0.75
SCORE_TOL = 3e-5
TIER_Y = {"full": "y_full", "warm75": "y_rem2"}
HIT_OF_TIER = {"full": "FULL_HIT", "warm75": "WARM_START"}
DISPATCHED_OF_HIT = {"FULL_HIT": "y_full", "WARM_START": "y_rem2", "MISS": None}
REQUIRED_STEP_FIELDS = ("vision_0", "vision_1", "prompt_emb", "robot_state", "clean_action")
FROZEN_ATTR_OF = {
    "loto_frozen_record_sha256": "record_sha256",
    "loto_arm_yaml_sha256": "arm_yaml_sha256",
    "loto_library_sha256": "library_sha256",
    "loto_checkpoint_identity_sha256": "checkpoint_identity_sha256",
    "loto_pool_manifest_sha256": "pool_manifest_sha256",
    "loto_fits_sha256": "fits_sha256",
}
ROW_KEY = ("run_tag", "connection_id", "episode_id", "decision_id")
LABEL_COPY_FIELDS = ROW_KEY + ("control_step_idx", "task", "task_id", "orig_init_state_idx", "hit_type",
                               "start_t", "winner_id", "searched", "sample_group", "episode_success")
DEFAULTS = {"n_sample": 2000, "alpha": 0.05, "tol": 0.05, "n_boot": 1000, "min_rows": 200, "min_episodes": 20,
            "min_event_episodes": 5, "min_valid_fraction": 0.9}


def _attr(a, key, default=None):
    if key not in a:
        return default
    v = a[key]
    return v.decode() if isinstance(v, bytes) else v


def _rid(row: dict) -> str:
    return f"{row.get('run_tag')}/{row.get('connection_id')}/ep{row.get('episode_id')}/d{row.get('decision_id')}"


def row_key(row: dict) -> tuple:
    """The decision's primary key ``(run_tag, connection_id, episode_id, decision_id)``."""
    return (str(row["run_tag"]), str(row["connection_id"]), int(row["episode_id"]), int(row["decision_id"]))


def frozen_attrs(record: dict, record_sha: str) -> dict[str, str]:
    """The attrs every logged episode of a run must carry, derived from the frozen record."""
    values = {"record_sha256": record_sha, "arm_yaml_sha256": record["arm"]["yaml_sha256"],
              "library_sha256": record["identity"]["library_sha256"],
              "checkpoint_identity_sha256": record["identity"]["checkpoint_identity_sha256"],
              "pool_manifest_sha256": record["pool_manifest_sha256"], "fits_sha256": record["fits_sha256"]}
    return {attr: values[src] for attr, src in FROZEN_ATTR_OF.items()}


# ------------------------------------------------------------------
# collect
# ------------------------------------------------------------------


def collect_decisions(log_root: str | pathlib.Path, manifest: dict, record: dict, record_sha: str,
                      run_tag: str = "verify") -> tuple[list[dict], dict]:
    """All decisions of the formal run, or SystemExit with every discrepancy listed.

    Checks: run tag is the record's formal tag; the manifest is the frozen pool
    (sha, per-task count, exactly the expected episode set); every HDF5 carries
    the frozen attrs, the run tag, its connection id and schedule; step groups
    are exactly ``step_0000..step_{n-1}`` with the required fields; sidecar rows
    cover decisions ``0..n-1`` once each with ``control_step_idx = H_exec *
    decision_id`` and the H5's task / identity; no ``.h5.tmp``; no sidecar row
    without a file. Returns ``(rows, episodes)``.
    """
    import h5py

    validate_pool_manifest(manifest)
    if manifest != record.get("pool"):
        raise SystemExit("pool manifest differs from the frozen pool")
    if run_tag != record["run_tags"].get("verify"):
        raise SystemExit(f"collect only admits the formal run tag {record['run_tags'].get('verify')!r}, got {run_tag!r}")
    root = pathlib.Path(log_root) / run_tag
    if not root.is_dir():
        raise SystemExit(f"no log directory for run_tag {run_tag!r}: {root}")
    per_task = int(record.get("pool_per_task", {}).get("verify", 0))
    expected = {(int(t), int(i)) for t, idxs in manifest["verify"].items() for i in idxs}
    n_tasks = 10
    if per_task <= 0 or len(expected) != n_tasks * per_task or any(len(v) != per_task for v in manifest["verify"].values()):
        raise SystemExit(f"pool manifest does not hold {n_tasks} x {per_task} verify episodes: "
                         f"{ {t: len(v) for t, v in manifest['verify'].items()} }")
    want_attrs = frozen_attrs(record, record_sha)
    h_exec = int(record["identity"]["h_exec"])
    schedule_id = record["identity"]["schedule_id"]
    problems: list[str] = []
    for tmp in sorted(root.rglob("*.h5.tmp")):
        problems.append(f"unfinished episode file: {tmp}")
    episodes: dict[tuple[int, int], dict] = {}
    rows: list[dict] = []
    duplicates, extras = [], []
    for conn_dir in sorted(root.glob("conn_*")):
        cid = conn_dir.name[len("conn_"):]
        side_rows: list[dict] = []
        for sc in sorted(conn_dir.rglob(SIDECAR_NAME)):
            side_rows.extend(read_jsonl(sc))
        by_ep: dict[int, list[dict]] = {}
        for r in side_rows:
            by_ep.setdefault(int(r["episode_id"]), []).append(r)
        for h5 in sorted(conn_dir.rglob("*.h5")):
            with h5py.File(h5, "r") as f:
                a = f.attrs
                tid, orig = int(a["task_id"]), int(a["orig_init_state_idx"])
                ep_id, n_steps = int(a["episode_id"]), int(a["num_steps"])
                success = bool(a["success"])
                task = str(_attr(a, "task", ""))
                if n_steps <= 0:
                    problems.append(f"{h5}: completed episode has no decisions")
                if int(_attr(a, "denoising_num_steps", -1)) != 8:
                    problems.append(f"{h5}: denoising_num_steps differs from the frozen schedule")
                if str(_attr(a, "run_tag", "")) != run_tag:
                    problems.append(f"{h5}: run_tag {_attr(a, 'run_tag')!r} != {run_tag!r}")
                if str(_attr(a, "connection_id", "")) != cid:
                    problems.append(f"{h5}: connection_id {_attr(a, 'connection_id')!r} != dir {cid!r}")
                if str(_attr(a, "denoise_schedule_id", "")) != schedule_id:
                    problems.append(f"{h5}: schedule {_attr(a, 'denoise_schedule_id')!r} != frozen {schedule_id!r}")
                if int(_attr(a, "h_exec", -1)) != h_exec:
                    problems.append(f"{h5}: h_exec {_attr(a, 'h_exec')!r} != frozen {h_exec}")
                for key, want in want_attrs.items():
                    if str(_attr(a, key, None)) != str(want):
                        problems.append(f"{h5}: attr {key} = {_attr(a, key, None)!r} != frozen {want[:12]!r}")
                groups = sorted(k for k in f.keys() if k.startswith("step_"))
                expected_groups = [f"step_{i:04d}" for i in range(n_steps)]
                if groups != expected_groups:
                    problems.append(f"{h5}: step groups {groups[:3]}..(n={len(groups)}) != step_0000..step_{n_steps - 1:04d}")
                for g in groups:
                    missing = [k for k in REQUIRED_STEP_FIELDS if k not in f[g]]
                    if missing:
                        problems.append(f"{h5}/{g}: missing fields {missing}")
                    elif tuple(f[g]["clean_action"].shape) != (16, 32):
                        problems.append(f"{h5}/{g}: clean_action must be [16,32]")
            key = (tid, orig)
            if key in episodes:
                duplicates.append((key, str(h5), episodes[key]["h5"]))
                continue
            if key not in expected:
                extras.append((key, str(h5)))
                continue
            srows = by_ep.pop(ep_id, [])
            ids = sorted(int(r["decision_id"]) for r in srows)
            if ids != list(range(n_steps)):
                problems.append(f"{h5}: sidecar decisions {ids[:5]}..(n={len(ids)}) != 0..{n_steps - 1}")
            for r in srows:
                d = int(r["decision_id"])
                if int(r.get("task_id", -1)) != tid or int(r.get("orig_init_state_idx", -1)) != orig \
                        or r.get("run_tag") != run_tag or r.get("connection_id") != cid:
                    problems.append(f"{h5}: sidecar identity mismatch at decision {d}")
                if int(r.get("control_step_idx", -1)) != h_exec * d:
                    problems.append(f"{h5}: decision {d} control_step_idx {r.get('control_step_idx')} != {h_exec * d}")
                if str(r.get("task", "")) != task:
                    problems.append(f"{h5}: decision {d} sidecar task differs from the file's task")
                if "episode_success" not in r or bool(r["episode_success"]) != success:
                    problems.append(f"{h5}: decision {d} sidecar episode_success {r.get('episode_success')!r} != file {success}")
                if r.get("hit_type") not in DISPATCHED_OF_HIT:
                    problems.append(f"{h5}: decision {d} unknown hit_type {r.get('hit_type')!r}")
                else:
                    try:
                        validate_online_row(r)
                    except SystemExit as exc:
                        problems.append(f"{h5}: {exc}")
            file_sha = sha256_file(h5)
            episodes[key] = {"h5": str(h5), "connection_id": cid, "episode_id": ep_id, "task_id": tid,
                             "orig_init_state_idx": orig, "n_steps": n_steps, "success": success, "task": task,
                             "h5_sha256": file_sha}
            for r in sorted(srows, key=lambda r: int(r["decision_id"])):
                rows.append({**r, "h5": str(h5), "step": f"step_{int(r['decision_id']):04d}", "h5_sha256": file_sha})
        for ep_id, leftover in by_ep.items():
            problems.append(f"conn {cid}: {len(leftover)} sidecar rows for episode {ep_id} without an HDF5 file")
    missing = sorted(expected - set(episodes))
    if problems or duplicates or extras or missing:
        lines = [f"collect_decisions({run_tag}) rejected the logs:"]
        lines += [f"  missing episode {k}" for k in missing]
        lines += [f"  duplicate episode {k}: {a} vs {b}" for k, a, b in duplicates]
        lines += [f"  episode not on the manifest {k}: {p}" for k, p in extras]
        lines += [f"  {p}" for p in problems]
        raise SystemExit("\n".join(lines))
    return rows, {f"{t}:{i}": v for (t, i), v in sorted(episodes.items())}


def sample_decisions(rows: list[dict], n: int, seed: int) -> tuple[list[dict], dict]:
    """Uniform ``base`` sample without replacement plus every other WARM row as ``warm_extra``."""
    ordered = sorted(rows, key=row_key)
    if len({row_key(r) for r in ordered}) != len(ordered):
        raise SystemExit("duplicate decision in the sampling population")
    if n <= 0:
        raise SystemExit("sample size must be positive")
    total = len(ordered)
    rng = np.random.default_rng(int(seed))
    n_base = min(int(n), total)
    base_idx = set(int(i) for i in rng.choice(total, size=n_base, replace=False)) if total else set()
    sampled: list[dict] = []
    n_warm_total = n_warm_in_base = n_full_total = n_full_in_base = n_miss_total = 0
    for i, r in enumerate(ordered):
        hit = r.get("hit_type")
        is_warm = hit == "WARM_START"
        n_warm_total += int(is_warm)
        n_full_total += int(hit == "FULL_HIT")
        n_miss_total += int(hit == "MISS")
        if i in base_idx:
            sampled.append({**r, "sample_group": "base"})
            n_warm_in_base += int(is_warm)
            n_full_in_base += int(hit == "FULL_HIT")
        elif is_warm:
            sampled.append({**r, "sample_group": "warm_extra"})
    manifest = {"seed": int(seed), "n_requested": int(n), "n_total": total, "n_base": n_base,
                "n_warm_total": n_warm_total, "n_warm_in_base": n_warm_in_base, "n_warm_extra": n_warm_total - n_warm_in_base,
                "n_full_total": n_full_total, "n_full_in_base": n_full_in_base, "n_miss_total": n_miss_total,
                "n_sampled": len(sampled),
                "inclusion_probability": {"base": (n_base / total) if total else None, "warm": 1.0 if n_warm_total else None},
                "keys": [list(row_key(r)) for r in sampled], "selected_rows": sampled}
    return sampled, manifest


# ------------------------------------------------------------------
# label
# ------------------------------------------------------------------


def offline_candidate(strategy, builder, group, enabled: list[str], task_key: str, decision_id: int,
                      traj_of: dict[str, str]) -> dict:
    """Top-1 of the full library for one logged observation, independent of the online gate / verdict."""
    keys = query_keys_for(builder, group, enabled)
    hits = search(strategy, keys, decision_id, task_key)
    if not hits:
        return {"candidate_id": None, "s_offline": None}
    cand, s_off, _ = loto_winner(hits, "", False, traj_of)
    return {"candidate_id": cand, "s_offline": s_off}


def _finite(x) -> bool:
    try:
        return x is not None and not isinstance(x, bool) and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def validate_online_row(row: dict) -> None:
    """Reject unknown dispatches, invalid scores and rungs before sampling or replay."""
    hit, searched, score = row.get("hit_type"), row.get("searched"), row.get("s")
    if hit not in DISPATCHED_OF_HIT or type(searched) is not bool:
        raise SystemExit(f"{_rid(row)}: invalid hit_type/searched")
    if score is not None and not _finite(score):
        raise SystemExit(f"{_rid(row)}: non-finite online score")
    if hit in ("FULL_HIT", "WARM_START"):
        if not searched or not row.get("winner_id") or not _finite(score):
            raise SystemExit(f"{_rid(row)}: cache hit without a searched winner and finite score")
    elif row.get("winner_id") is not None or (not searched and score is not None):
        raise SystemExit(f"{_rid(row)}: inconsistent MISS metadata")
    if hit == "WARM_START":
        if not _finite(row.get("start_t")) or float(row["start_t"]) != VERIFY_WARM_T:
            raise SystemExit(f"{_rid(row)}: unknown WARM start_t")
    elif row.get("start_t") is not None:
        raise SystemExit(f"{_rid(row)}: non-WARM decision has start_t")


def check_online_consistency(row: dict, cand: dict) -> None:
    """Refuse a decision whose offline candidate disagrees with what the server logged.

    FULL / WARM rows need a finite online score equal to the offline one within
    ``SCORE_TOL`` and the same winner; a WARM row must be at the frozen rung. A
    searched MISS with a finite online score must agree on the score. Unknown hit
    types and non-finite online scores on searched rows are rejected outright.
    """
    hit = row.get("hit_type")
    online_s, winner = row.get("s"), row.get("winner_id")
    if hit in ("FULL_HIT", "WARM_START"):
        if cand["candidate_id"] is None or not _finite(online_s) or not _finite(cand["s_offline"]):
            raise SystemExit(f"{_rid(row)}: online {hit} but offline candidate {cand} / online score {online_s!r}")
        if cand["candidate_id"] != winner or abs(float(cand["s_offline"]) - float(online_s)) > SCORE_TOL:
            raise SystemExit(f"{_rid(row)}: offline ({cand['candidate_id']}, {cand['s_offline']}) != online ({winner}, {online_s})")
        if hit == "WARM_START" and round(float(row.get("start_t") or -1), 4) != VERIFY_WARM_T:
            raise SystemExit(f"{_rid(row)}: WARM_START at start_t={row.get('start_t')} is not the frozen K=2 arm")
    elif hit == "MISS":
        if row.get("searched"):
            if online_s is not None and not _finite(online_s):
                raise SystemExit(f"{_rid(row)}: searched MISS with a non-finite online score {online_s!r}")
            if _finite(online_s) and (cand["candidate_id"] is None or not _finite(cand["s_offline"])
                                      or abs(float(cand["s_offline"]) - float(online_s)) > SCORE_TOL):
                raise SystemExit(f"{_rid(row)}: searched MISS score {online_s} != offline {cand['s_offline']}")
    else:
        raise SystemExit(f"{_rid(row)}: unknown hit_type {hit!r}")


def k2_labels(runner, templates, task: str, group, row: dict, payload, executed: Any, w, mask, h_exec: int,
              schedule, *, ref_seed: int) -> dict:
    """Both K=2 labels for one decision per the plan's table; the reference is a fresh full inference."""
    from openpi.cache.components.surface_judge import weighted_chunk_deviation

    hit = row["hit_type"]
    executed = to_chunk(torch_as(executed))
    z = make_noise(ref_seed, (1,) + tuple(executed.shape))
    with runner.session():
        _, stage2 = stage2_in_session(runner, templates, task, group)
        ref_t = runner.run_stage3(stage2, noise=z).action_pred
        warm_c = None
        if hit != "WARM_START":
            warm_c = runner.run_stage3_from(stage2, payload.intermediates[VERIFY_WARM_T], VERIFY_WARM_T,
                                            schedule=schedule).action_pred
    ref = to_chunk(ref_t)
    cand_full = to_chunk(payload.action_chunk)

    def dev(a, b):
        """Compare two chunks over the frozen executed window and action weights."""
        return weighted_chunk_deviation(a, b, w, mask, h_exec)

    if hit == "FULL_HIT":
        return {"y_full": dev(executed, ref), "y_rem2": dev(to_chunk(warm_c), ref), "dispatched_y_key": "y_full"}
    if hit == "WARM_START":
        return {"y_full": dev(cand_full, ref), "y_rem2": dev(executed, ref), "dispatched_y_key": "y_rem2"}
    return {"y_full": dev(cand_full, ref), "y_rem2": dev(to_chunk(warm_c), ref), "dispatched_y_key": None}


def torch_as(x):
    """``torch.as_tensor`` behind a lazy import so the CPU stages never need torch."""
    import torch

    return torch.as_tensor(x)


def bind_label_inputs(record: dict, *, library_sha: str, template_path: str, checkpoint: str, w, mask, h_exec: int,
                      schedule_id: str) -> dict:
    """Prove the label stage runs on the frozen library / checkpoint / template / W / H_exec / schedule."""
    ident = record["identity"]
    actual = {"library_sha256": library_sha, "template_sha256": sha256_file(template_path),
              "checkpoint_identity_sha256": checkpoint_identity(checkpoint)["sha256"],
              "weights_sha256": weights_sha256(w, mask), "h_exec": int(h_exec), "schedule_id": schedule_id}
    problems = [f"{k}: frozen {str(ident.get(k))[:16]!r} vs actual {str(v)[:16]!r}" for k, v in actual.items()
                if ident.get(k) != v]
    if problems:
        raise SystemExit("label inputs do not match the frozen run record: " + "; ".join(problems))
    return actual


def cmd_label(args) -> None:
    """Island / GPU: offline candidates, references and both K=2 labels for the sampled decisions."""
    import h5py
    import torch

    from openpi.cache.types import groot_n15_schedule

    from exp.rit_loto.build_loto_table import ModelSide
    from exp.robocasa365.rit_shadow import library_action_weights

    record, record_sha = load_frozen_record(args.frozen_record)
    if record["suite"] != args.suite:
        raise SystemExit(f"frozen record is for {record['suite']!r}, not {args.suite!r}")
    sample_manifest = json.loads(pathlib.Path(args.sample_manifest).read_text(encoding="utf-8"))
    if sample_manifest.get("frozen_record_sha256") != record_sha:
        raise SystemExit("sample manifest was collected under a different frozen record")
    validate_sample_manifest(sample_manifest, record)
    sampled = read_jsonl(args.sampled_jsonl)
    sampled_sha = sha256_file(args.sampled_jsonl)
    if sampled_sha != sample_manifest.get("sampled_sha256") or sampled != sample_manifest["selected_rows"]:
        raise SystemExit("sampled_decisions.jsonl bytes/metadata differ from the sampling manifest")
    if args.root_seed != record["verify_protocol"]["root_seed"] or args.limit < 0:
        raise SystemExit("label root_seed/limit differs from the frozen protocol")
    # Validate the captured observations before allocating the model or replaying any row.
    checked = set()
    for row in sampled:
        validate_online_row(row)
        if row["h5"] not in checked:
            if sha256_file(row["h5"]) != row.get("h5_sha256"):
                raise SystemExit(f"{row['h5']}: HDF5 bytes changed since collect")
            checked.add(row["h5"])
        with h5py.File(row["h5"], "r") as f:
            expected = {**frozen_attrs(record, record_sha), "run_tag": row["run_tag"],
                        "connection_id": row["connection_id"], "episode_id": row["episode_id"],
                        "task_id": row["task_id"], "orig_init_state_idx": row["orig_init_state_idx"],
                        "task": row["task"], "success": row["episode_success"], "h_exec": 5,
                        "denoising_num_steps": 8, "denoise_schedule_id": record["identity"]["schedule_id"]}
            if any(_attr(f.attrs, k) != v for k, v in expected.items()) \
                    or row["step"] != f"step_{int(row['decision_id']):04d}" or row["step"] not in f:
                raise SystemExit(f"{_rid(row)}: HDF5 identity/schedule/step differs from collected row")
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schedule = groot_n15_schedule(args.denoising_steps)
    cfg, cfg_used = load_template(args.template_yaml, args.library_pkl, out_dir)
    library = load_library(args.library_pkl, expected_schedule=schedule, warm_ts=(VERIFY_WARM_T,))
    w, mask = library_action_weights(args.library_pkl)
    w = torch.as_tensor(np.asarray(w), dtype=torch.float32)
    mask = torch.as_tensor(np.asarray(mask), dtype=torch.bool)
    bound = bind_label_inputs(record, library_sha=library.sha256, template_path=args.template_yaml,
                              checkpoint=args.checkpoint, w=w, mask=mask, h_exec=args.h_exec,
                              schedule_id=schedule.schedule_id)
    enabled, weights = enabled_fields(cfg)
    storage = build_storage(cfg, library.entries)
    strategy = build_retrieval(cfg, storage, weights, n_hint=max(library.per_task_entries.values()))
    builder = make_query_builder(cfg.key_builder.type)
    model = ModelSide(args.checkpoint, denoising_steps=args.denoising_steps, device=args.device)
    if model.schedule != schedule:
        raise SystemExit(f"live schedule {model.schedule.schedule_id} != requested {schedule.schedule_id}")
    partial = args.limit > 0
    if partial:
        sampled = sampled[: args.limit]
    stats = {"rows": 0, "no_candidate": 0, "by_hit": {}, "consistency_checked": 0}
    out_rows: list[dict] = []
    t0 = time.time()
    open_files: dict[str, Any] = {}
    try:
        for row in sampled:
            f = open_files.get(row["h5"]) or open_files.setdefault(row["h5"], h5py.File(row["h5"], "r"))
            g = f[row["step"]]
            cand = offline_candidate(strategy, builder, g, enabled, row["task"], int(row["decision_id"]), library.traj_of)
            check_online_consistency(row, cand)
            stats["consistency_checked"] += 1
            base = {k: row.get(k) for k in LABEL_COPY_FIELDS}
            base.update({"suite": args.suite, "online_s": row.get("s"), "candidate_id": cand["candidate_id"],
                         "s": cand["s_offline"],
                         "s_source": "online_confirmed" if row.get("searched") and row.get("s") is not None else "offline_only"})
            stats["by_hit"][row["hit_type"]] = stats["by_hit"].get(row["hit_type"], 0) + 1
            if cand["candidate_id"] is None:
                stats["no_candidate"] += 1
                out_rows.append({**base, "y_full": None, "y_rem2": None, "dispatched_y_key": None, "label_reason": "no_offline_candidate"})
                continue
            ident = {"suite": args.suite, "run_tag": row["run_tag"], "task_id": int(row["task_id"]),
                     "orig_init_state_idx": int(row["orig_init_state_idx"]), "decision_id": int(row["decision_id"]),
                     "tag": "verify_ref"}
            seed = derive_seed(args.root_seed, ident)
            payload = library.by_id[cand["candidate_id"]].payload
            labels = k2_labels(model.runner, model.templates, row["task"], g, row, payload, h5_chunk(g, "clean_action"),
                               w, mask, args.h_exec, schedule, ref_seed=seed)
            out_rows.append({**base, **labels, "ref_seed": seed, "noise_contract": NOISE_CONTRACT})
            stats["rows"] += 1
            if stats["rows"] % 200 == 0:
                print(f"labelled {stats['rows']} rows in {time.time() - t0:.0f}s", flush=True)
    finally:
        for f in open_files.values():
            f.close()
    rows_path = out_dir / ("verify_rows.partial.jsonl" if partial else "verify_rows.jsonl")
    validate_labelled_rows(out_rows, {**sample_manifest, "selected_rows": sampled,
                                     "keys": [list(row_key(r)) for r in sampled]})
    write_jsonl(rows_path, out_rows)
    label_record = {
        "protocol": "rit_loto_verify_labels_v1", "suite": args.suite, "partial": partial, "limit": int(args.limit),
        "frozen_record_sha256": record_sha, "sampled_sha256": sampled_sha, "n_sampled": len(sampled), "stats": stats,
        "sample_manifest_sha256": sha256_file(args.sample_manifest),
        "bound_inputs": bound, "noise_contract": {"version": NOISE_CONTRACT, "root_seed": int(args.root_seed),
                                                  "tag": "verify_ref", "torch": torch.__version__, "shape": [1, 16, 32]},
        "config_used": str(cfg_used), "code_sha256": code_sha256(), "git_commit": git_commit(),
        "elapsed_s": time.time() - t0, "rows_path": str(rows_path), "rows_sha256": sha256_file(rows_path),
    }
    rec_path = out_dir / ("verify_labels.partial.record.json" if partial else "verify_labels.record.json")
    rec_path.write_text(json.dumps(label_record, indent=1) + "\n", encoding="utf-8")
    print(f"label{' (PARTIAL)' if partial else ''}: {stats} -> {rows_path}")


# ------------------------------------------------------------------
# report
# ------------------------------------------------------------------


def validate_labelled_rows(rows: list[dict], sample_manifest: dict) -> dict:
    """Join the labelled rows against the sampling manifest and refuse anything incomplete.

    Every sampled key must appear exactly once and nothing else may appear; a
    row with a candidate must carry finite ``y_full`` / ``y_rem2`` and a
    ``dispatched_y_key`` consistent with its hit type; only
    ``label_reason == "no_offline_candidate"`` rows may lack labels.
    """
    want = [tuple(k) for k in sample_manifest.get("keys", [])]
    want_set = set(want)
    if len(want_set) != len(want):
        raise SystemExit("sampling manifest lists a key twice")
    sources = {row_key(r): r for r in sample_manifest.get("selected_rows", [])}
    if set(sources) != want_set:
        raise SystemExit("sampling manifest lacks the selected rows' complete metadata")
    seen: dict[tuple, int] = {}
    problems: list[str] = []
    for r in rows:
        key = row_key(r)
        seen[key] = seen.get(key, 0) + 1
        if key not in want_set:
            problems.append(f"{_rid(r)}: labelled row not in the sampling manifest")
        else:
            source = sources[key]
            for field in LABEL_COPY_FIELDS:
                if field not in r or field not in source or r[field] != source[field]:
                    problems.append(f"{_rid(r)}: {field} differs from the sampled decision")
            if r.get("online_s") != source.get("s"):
                problems.append(f"{_rid(r)}: online_s differs from the sampled decision")
            try:
                validate_online_row(source)
                check_online_consistency(source, {"candidate_id": r.get("candidate_id"), "s_offline": r.get("s")})
            except SystemExit as exc:
                problems.append(str(exc))
            expected_source = "online_confirmed" if source.get("searched") and source.get("s") is not None else "offline_only"
            if r.get("s_source") != expected_source:
                problems.append(f"{_rid(r)}: invalid score provenance")
        group = r.get("sample_group")
        if group not in ("base", "warm_extra"):
            problems.append(f"{_rid(r)}: unknown sample_group {group!r}")
        if group == "warm_extra" and r.get("hit_type") != "WARM_START":
            problems.append(f"{_rid(r)}: warm_extra contains a non-WARM decision")
        hit = r.get("hit_type")
        if hit not in DISPATCHED_OF_HIT:
            problems.append(f"{_rid(r)}: unknown hit_type {hit!r}")
            continue
        if r.get("candidate_id") is None:
            if r.get("label_reason") != "no_offline_candidate" or hit != "MISS":
                problems.append(f"{_rid(r)}: no candidate but hit_type {hit!r} / reason {r.get('label_reason')!r}")
            if any(r.get(k) is not None for k in ("s", "y_full", "y_rem2", "dispatched_y_key")):
                problems.append(f"{_rid(r)}: no candidate but scores/labels are present")
            continue
        if not _finite(r.get("s")):
            problems.append(f"{_rid(r)}: missing or non-finite offline score")
        for k in ("y_full", "y_rem2"):
            if not _finite(r.get(k)) or float(r[k]) < 0:
                problems.append(f"{_rid(r)}: {k} missing or non-finite with a candidate present")
        if r.get("dispatched_y_key") != DISPATCHED_OF_HIT[hit]:
            problems.append(f"{_rid(r)}: dispatched_y_key {r.get('dispatched_y_key')!r} != {DISPATCHED_OF_HIT[hit]!r} for {hit}")
        if hit in ("FULL_HIT", "WARM_START") and not _finite(r.get("online_s")):
            problems.append(f"{_rid(r)}: dispatched cache row without a finite online score")
    dup = [k for k, c in seen.items() if c > 1]
    missing = sorted(want_set - set(seen))
    problems += [f"labelled twice: {k}" for k in dup]
    problems += [f"sampled but not labelled: {k}" for k in missing]
    if problems:
        raise SystemExit("labelled rows rejected:\n  " + "\n  ".join(problems[:50]) + (f"\n  ... {len(problems)} problems" if len(problems) > 50 else ""))
    return {"n_rows": len(rows), "n_no_candidate": int(sum(r.get("candidate_id") is None for r in rows)),
            "n_base": int(sum(r.get("sample_group") == "base" for r in rows)),
            "n_warm_extra": int(sum(r.get("sample_group") == "warm_extra" for r in rows))}


def _quantile_bins(s: np.ndarray, n_bins: int = 4) -> list[float]:
    edges = np.unique(np.quantile(s, np.linspace(0.0, 1.0, n_bins + 1)))
    return edges.tolist()


def exceedance_report(rows: list[dict], episode_manifest: dict[str, dict], fit, tiers: tuple[str, ...] = ("full", "warm75"),
                      *, alpha: float = DEFAULTS["alpha"], tol: float = DEFAULTS["tol"], n_boot: int = DEFAULTS["n_boot"],
                      seed: int = 0, min_rows: int = DEFAULTS["min_rows"], min_episodes: int = DEFAULTS["min_episodes"],
                      min_event_episodes: int = DEFAULTS["min_event_episodes"],
                      min_valid_fraction: float = DEFAULTS["min_valid_fraction"]) -> dict:
    """Per dispatched tier: decision-weighted exceedance on in-support rows, episode cluster bootstrap, gates, reading.

    The information gates count episodes that *contain* an exceedance and episodes
    that *contain* a non-exceedance (a mixed episode counts for both); zero / full
    exceedance and degenerate intervals are refused as evidence.
    """
    knots = np.asarray(fit.knots, dtype=np.float64)
    s_lo, s_hi = float(knots[0]), float(knots[-1])
    eps_by_task: dict[int, list[str]] = {}
    for key, ep in episode_manifest.items():
        eps_by_task.setdefault(int(ep["task_id"]), []).append(key)
    for t in eps_by_task:
        eps_by_task[t].sort()
    upper = float(alpha) + float(tol)
    out: dict[str, Any] = {"alpha": alpha, "tol": tol, "upper_bound_for_pass": upper, "n_boot": n_boot, "seed": seed,
                           "support": [s_lo, s_hi], "gates": {"min_rows": min_rows, "min_episodes": min_episodes,
                                                              "min_event_episodes": min_event_episodes,
                                                              "min_valid_fraction": min_valid_fraction},
                           "tiers": {}}
    for tier in tiers:
        y_key = TIER_Y[tier]
        sel = [r for r in rows if r.get("dispatched_y_key") == y_key and r.get("hit_type") == HIT_OF_TIER[tier]
               and (tier != "full" or r.get("sample_group") == "base")]
        for r in sel:
            if not _finite(r.get("online_s")) or not _finite(r.get(y_key)):
                raise SystemExit(f"{_rid(r)}: dispatched {tier} row without finite online_s / {y_key}; run validate_labelled_rows first")
        s = np.array([float(r["online_s"]) for r in sel])
        d = np.array([float(r[y_key]) for r in sel])
        inside = (s >= s_lo) & (s <= s_hi) if s.size else np.array([], dtype=bool)
        q = predict(fit, s, tier) if s.size else np.array([])
        exceed = d > q if s.size else np.array([], dtype=bool)
        n_in, n_out = int(inside.sum()), int((~inside).sum())
        block: dict[str, Any] = {"y_key": y_key, "n_rows_total": len(sel), "n_in_support": n_in, "n_out_of_support": n_out,
                                 "sample_groups": {g: int(sum(1 for r in sel if r.get("sample_group") == g)) for g in ("base", "warm_extra")},
                                 "out_of_support": {"n": n_out, "n_exceed": int((exceed & ~inside).sum()) if s.size else 0,
                                                    "status": "unverified"}}
        if n_in == 0:
            block.update(point_estimate=None, interval=None, reading="insufficient_evidence", reasons=["no in-support rows"])
            out["tiers"][tier] = block
            continue
        ex_in = exceed[inside]
        point = float(ex_in.mean())
        per_ep: dict[str, list[int]] = {k: [0, 0] for k in episode_manifest}
        for r, e, ins in zip(sel, exceed, inside):
            if not ins:
                continue
            key = f"{int(r['task_id'])}:{int(r['orig_init_state_idx'])}"
            if key not in per_ep:
                raise SystemExit(f"row episode {key} is not on the episode manifest")
            per_ep[key][0] += int(e)
            per_ep[key][1] += 1
        eps_with_rows = [k for k, v in per_ep.items() if v[1] > 0]
        eps_with_event = [k for k in eps_with_rows if per_ep[k][0] > 0]
        eps_with_nonevent = [k for k in eps_with_rows if per_ep[k][0] < per_ep[k][1]]
        rng = np.random.default_rng([int(seed), tiers.index(tier)])
        ratios: list[float] = []
        invalid = 0
        for _ in range(int(n_boot)):
            num = den = 0
            for t in sorted(eps_by_task):
                keys = eps_by_task[t]
                draw = rng.choice(len(keys), size=len(keys), replace=True)
                for i in draw.tolist():
                    c = per_ep[keys[i]]
                    num += c[0]
                    den += c[1]
            if den == 0:
                invalid += 1
            else:
                ratios.append(num / den)
        arr = np.asarray(ratios, dtype=np.float64)
        n_valid = int(arr.size)
        interval = None
        if n_valid:
            interval = [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))]
        degenerate = interval is None or interval[0] == interval[1] or (n_valid and float(arr.min()) == float(arr.max()))
        reasons = []
        if n_in < min_rows:
            reasons.append(f"in-support rows {n_in} < {min_rows}")
        if len(eps_with_rows) < min_episodes:
            reasons.append(f"episodes with rows {len(eps_with_rows)} < {min_episodes}")
        if len(eps_with_event) < min_event_episodes:
            reasons.append(f"episodes containing an exceedance {len(eps_with_event)} < {min_event_episodes}")
        if len(eps_with_nonevent) < min_event_episodes:
            reasons.append(f"episodes containing a non-exceedance {len(eps_with_nonevent)} < {min_event_episodes}")
        if n_valid < int(math.ceil(min_valid_fraction * n_boot)):
            reasons.append(f"valid bootstrap replicates {n_valid} < {min_valid_fraction} x {n_boot}")
        if degenerate:
            reasons.append("degenerate bootstrap interval")
        usable_interval = interval if not reasons else None
        if usable_interval is None:
            reading = "insufficient_evidence"
        elif usable_interval[1] <= upper:
            reading = "within_preset_tolerance_in_support"
        elif usable_interval[0] > upper:
            reading = "risk_above_preset_tolerance_in_support"
        else:
            reading = "insufficient_evidence"
        edges = _quantile_bins(s[inside])
        bins = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = inside & (s >= lo) & (s <= hi if hi == edges[-1] else s < hi)
            if m.sum():
                bins.append({"s_lo": float(lo), "s_hi": float(hi), "n": int(m.sum()), "exceedance": float(exceed[m].mean())})
        block.update({
            "point_estimate": point, "n_exceed": int(ex_in.sum()),
            "n_episodes_with_rows": len(eps_with_rows), "n_episodes_with_event": len(eps_with_event),
            "n_episodes_with_nonevent": len(eps_with_nonevent), "n_tasks": len({k.split(":")[0] for k in eps_with_rows}),
            "bootstrap": {"n_valid": n_valid, "n_invalid": invalid, "raw_interval": interval, "degenerate": bool(degenerate)},
            "interval": usable_interval, "reasons": reasons, "reading": reading,
            "score_bins": bins, "n_score_bins": len(bins),
        })
        out["tiers"][tier] = block
    return out


def joint_refit(rows: list[dict], cost: rc.StageCost, *, alpha: float) -> dict:
    """K=2 re-fit on base rows with a candidate (descriptive overlay), or ``fit_unavailable``.

    Rows are expected to have passed ``validate_labelled_rows``: a base row with
    a candidate but a missing label is an error here, never a silent drop.
    """
    from exp.rit_pareto.rit_k import fit_record_fields

    base = [r for r in rows if r.get("sample_group") == "base" and r.get("candidate_id") is not None]
    for r in base:
        if not all(_finite(r.get(k)) for k in ("s", "y_full", "y_rem2")):
            raise SystemExit(f"{_rid(r)}: base row with a candidate lacks a finite label; refusing to fit around it")
    out: dict[str, Any] = {"n_rows": len(base), "source": "closed_loop_base_rows"}
    if not base:
        out.update(fit_unavailable=True, reason="no base rows with a candidate")
        return out
    s = np.array([float(r["s"]) for r in base])
    try:
        fits = er.fit_ladders(base, cost, [VERIFY_WARM_T], [2], float(alpha), ir_sample=s)
    except SystemExit as exc:
        out.update(fit_unavailable=True, reason=str(exc))
        return out
    rec = fit_record_fields(fits[2]["fit"])
    rec.update({"k": 2, "n_rows": int(fits[2]["n_rows"]), "s_range": [float(s.min()), float(s.max())]})
    out["fit"] = rec
    return out


def run_summary(decisions: list[dict], episodes: dict[str, dict], cost: rc.StageCost) -> dict:
    """Dispatch counts, realised IR under the measured cost, and the success count of the whole run."""
    counts: dict[str, int] = {}
    for r in decisions:
        hit = r.get("hit_type")
        if hit == "FULL_HIT":
            key = "full"
        elif hit == "WARM_START":
            key = f"warm{int(round(float(r['start_t']) * 100)):02d}"
        else:
            key = "miss"
        counts[key] = counts.get(key, 0) + 1
    tiers = rc.ladder(cost, [VERIFY_WARM_T])
    return {"n_decisions": len(decisions), "counts": counts,
            "realized_ir_measured_cost": rc.realized_ir(counts, tiers, cost),
            "n_episodes": len(episodes), "n_success": int(sum(bool(e["success"]) for e in episodes.values())),
            "note": "priced with the measured stage costs; not a wall-clock measurement of the logged server"}


def cmd_collect(args) -> None:
    """Merge the formal logs under the frozen record and write decisions / sample / episodes / sample manifest."""
    record, record_sha = load_frozen_record(args.frozen_record)
    if args.n_sample != record["verify_protocol"]["n_sample"] or args.seed != record["verify_protocol"]["sample_seed"]:
        raise SystemExit("collect sample size/seed differs from the frozen protocol")
    pool_sha = sha256_file(args.pool_manifest)
    if pool_sha != record["pool_manifest_sha256"]:
        raise SystemExit(f"pool manifest sha {pool_sha[:12]} != frozen {record['pool_manifest_sha256'][:12]}")
    manifest = json.loads(pathlib.Path(args.pool_manifest).read_text(encoding="utf-8"))
    rows, episodes = collect_decisions(args.log_root, manifest, record, record_sha, run_tag=args.run_tag)
    sampled, sample_manifest = sample_decisions(rows, args.n_sample, args.seed)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "decisions.jsonl", rows)
    write_jsonl(out_dir / "sampled_decisions.jsonl", sampled)
    (out_dir / "episodes.json").write_text(json.dumps(episodes, indent=1) + "\n", encoding="utf-8")
    sample_manifest.update({"protocol": "rit_loto_sample_v1", "run_tag": args.run_tag,
                            "frozen_record_sha256": record_sha, "pool_manifest_sha256": pool_sha,
                            "decisions_sha256": sha256_file(out_dir / "decisions.jsonl"),
                            "episodes_sha256": sha256_file(out_dir / "episodes.json"),
                            "sampled_sha256": sha256_file(out_dir / "sampled_decisions.jsonl"),
                            "code_sha256": code_sha256(), "git_commit": git_commit()})
    (out_dir / "sample_manifest.json").write_text(json.dumps(sample_manifest, indent=1) + "\n", encoding="utf-8")
    print(f"collect: {len(episodes)} episodes, {len(rows)} decisions, sampled "
          f"{ {k: v for k, v in sample_manifest.items() if k not in ('keys', 'selected_rows', 'code_sha256')} }")


def validate_sample_manifest(manifest: dict, record: dict) -> None:
    """Require the frozen sampling protocol, complete selected metadata and artifact digests."""
    expected = {"protocol": "rit_loto_sample_v1", "run_tag": record["run_tags"]["verify"],
                "pool_manifest_sha256": record["pool_manifest_sha256"],
                "n_requested": record["verify_protocol"]["n_sample"], "seed": record["verify_protocol"]["sample_seed"]}
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise SystemExit("sampling manifest differs from the frozen protocol")
    for field in ("decisions_sha256", "episodes_sha256", "sampled_sha256"):
        if not manifest.get(field):
            raise SystemExit(f"sampling manifest lacks {field}")
    selected = manifest.get("selected_rows", [])
    keys = [list(row_key(r)) for r in selected]
    if not selected or keys != manifest.get("keys") or len(selected) != manifest.get("n_sampled") \
            or len({tuple(k) for k in keys}) != len(keys):
        raise SystemExit("sampling manifest selected rows/keys/count disagree")
    require_code_identity(manifest.get("code_sha256"))


def validate_population(decisions: list[dict], episodes: dict, record: dict) -> None:
    """Require exactly the frozen 50 episodes and their complete, consistent decision sequences."""
    expected = {f"{t}:{i}" for t, indices in record["pool"]["verify"].items() for i in indices}
    if set(episodes) != expected:
        raise SystemExit("episodes.json does not contain the frozen task/init episode set")
    grouped = {k: [] for k in expected}
    keys = set()
    episode_ids = set()
    for key, ep in episodes.items():
        if key != f"{ep['task_id']}:{ep['orig_init_state_idx']}" or int(ep["n_steps"]) <= 0 \
                or type(ep.get("success")) is not bool or not ep.get("h5_sha256"):
            raise SystemExit(f"episode {key}: inconsistent identity/completion metadata")
        ep_id = (ep["connection_id"], ep["episode_id"])
        if ep_id in episode_ids:
            raise SystemExit("episode manifest reuses a connection/episode identity")
        episode_ids.add(ep_id)
    for row in decisions:
        validate_online_row(row)
        key = f"{row['task_id']}:{row['orig_init_state_idx']}"
        if key not in episodes or row_key(row) in keys:
            raise SystemExit("decisions contain an unknown episode or duplicate primary key")
        keys.add(row_key(row))
        ep = episodes[key]
        for field in ("connection_id", "episode_id", "task_id", "orig_init_state_idx", "task", "h5", "h5_sha256"):
            if row.get(field) != ep.get(field):
                raise SystemExit(f"{_rid(row)}: {field} differs from the episode manifest")
        decision = int(row["decision_id"])
        if row.get("run_tag") != record["run_tags"]["verify"] or row.get("episode_success") != ep["success"] \
                or row.get("control_step_idx") != 5 * decision or row.get("step") != f"step_{decision:04d}":
            raise SystemExit(f"{_rid(row)}: inconsistent decision metadata")
        grouped[key].append(decision)
    for key, ids in grouped.items():
        if sorted(ids) != list(range(int(episodes[key]["n_steps"]))):
            raise SystemExit(f"episode {key}: decision sequence is incomplete")


def cmd_report(args) -> None:
    """Bind every input to the frozen record, join the labels, then judge and summarise."""
    record, record_sha = load_frozen_record(args.frozen_record)
    label_record = json.loads(pathlib.Path(args.label_record).read_text(encoding="utf-8"))
    sample_manifest = json.loads(pathlib.Path(args.sample_manifest).read_text(encoding="utf-8"))
    validate_sample_manifest(sample_manifest, record)
    require_code_identity(label_record.get("code_sha256"))
    problems = []
    if record["suite"] != args.suite:
        problems.append(f"frozen record suite {record['suite']!r} != {args.suite!r}")
    if label_record.get("partial") is not False or label_record.get("limit") != 0:
        problems.append("label record is a partial (--limit) product")
    if label_record.get("protocol") != "rit_loto_verify_labels_v1" or label_record.get("suite") != record["suite"]:
        problems.append("label record protocol/suite differs from the frozen run")
    if label_record.get("frozen_record_sha256") != record_sha:
        problems.append("label record was produced under a different frozen record")
    if sample_manifest.get("frozen_record_sha256") != record_sha:
        problems.append("sample manifest was collected under a different frozen record")
    if sha256_file(args.verify_rows) != label_record.get("rows_sha256"):
        problems.append("verify rows bytes differ from the label record")
    if sha256_file(args.fits) != record["fits_sha256"]:
        problems.append("fits.json is not the one the arm was emitted from")
    if sha256_file(args.sample_manifest) != label_record.get("sample_manifest_sha256") \
            or label_record.get("sampled_sha256") != sample_manifest.get("sampled_sha256") \
            or label_record.get("n_sampled") != sample_manifest.get("n_sampled"):
        problems.append("label record does not bind the complete sampling manifest")
    for field in ("decisions", "episodes"):
        if sha256_file(getattr(args, field)) != sample_manifest.get(f"{field}_sha256"):
            problems.append(f"{field} bytes differ from the collected artifact")
    bound_keys = ("library_sha256", "template_sha256", "checkpoint_identity_sha256", "weights_sha256", "h_exec", "schedule_id")
    if label_record.get("bound_inputs") != {k: record["identity"][k] for k in bound_keys}:
        problems.append("label inputs differ from the frozen identity")
    noise = label_record.get("noise_contract", {})
    expected_noise = {"version": NOISE_CONTRACT, "root_seed": record["verify_protocol"]["root_seed"],
                      "tag": "verify_ref", "shape": [1, 16, 32]}
    if any(noise.get(k) != v for k, v in expected_noise.items()) or not noise.get("torch"):
        problems.append("label noise contract differs from the frozen protocol")
    for k in ("tol", "n_boot", "seed", "min_rows", "min_episodes", "min_event_episodes", "min_valid_fraction"):
        if getattr(args, k) != record["verify_protocol"][k]:
            problems.append(f"report {k} differs from the frozen protocol")
    if problems:
        raise SystemExit("report inputs rejected: " + "; ".join(problems))
    fits = json.loads(pathlib.Path(args.fits).read_text(encoding="utf-8"))
    if fits.get("suite") != record["suite"] or fits.get("identity") != record["identity"] \
            or fits.get("cost") != record["cost"] or fits.get("parity_gate_status") != "PASS":
        raise SystemExit("fits identity/cost/parity differs from the frozen record")
    block = fits[record["fit_source"]]
    if float(block["alpha"]) != float(record["alpha"]):
        raise SystemExit("fit source alpha differs from the frozen record")
    cost = load_cost(pathlib.Path(args.cost))
    if _cost_to_json(cost) != record["cost"]:
        raise SystemExit("report measured cost differs from the frozen record")
    fit = deserialize_fit(block["fits"][str(int(record["k"]))], cost, WARM_TS)
    if fit.alpha != record["alpha"]:
        raise SystemExit("curve alpha differs from the frozen record")
    rows = read_jsonl(args.verify_rows)
    join = validate_labelled_rows(rows, sample_manifest)
    episodes = json.loads(pathlib.Path(args.episodes).read_text(encoding="utf-8"))
    decisions = read_jsonl(args.decisions)
    validate_population(decisions, episodes, record)
    _, replayed_sample = sample_decisions(decisions, record["verify_protocol"]["n_sample"], record["verify_protocol"]["sample_seed"])
    if any(sample_manifest.get(k) != v for k, v in replayed_sample.items()):
        raise SystemExit("sampling manifest does not reproduce the frozen draw from all decisions")
    for row in rows:
        if row.get("suite") != record["suite"]:
            raise SystemExit(f"{_rid(row)}: label suite differs from the frozen run")
        if row.get("candidate_id") is not None:
            ident = {"suite": record["suite"], "run_tag": row["run_tag"], "task_id": int(row["task_id"]),
                     "orig_init_state_idx": int(row["orig_init_state_idx"]), "decision_id": int(row["decision_id"]), "tag": "verify_ref"}
            if row.get("ref_seed") != derive_seed(noise["root_seed"], ident) or row.get("noise_contract") != NOISE_CONTRACT:
                raise SystemExit(f"{_rid(row)}: reference seed/noise contract differs from the frozen run")
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": "rit_loto_verify_report_v1", "suite": args.suite, "fit_source": record["fit_source"], "k": int(record["k"]),
        "alpha": float(record["alpha"]), "frozen_record_sha256": record_sha, "fits_sha256": record["fits_sha256"],
        "label_record_sha256": sha256_file(args.label_record), "verify_rows_sha256": label_record["rows_sha256"],
        "join": join, "sampling": {k: v for k, v in sample_manifest.items() if k not in ("keys", "selected_rows")},
        "exceedance": exceedance_report(rows, episodes, fit, alpha=float(record["alpha"]), tol=args.tol, n_boot=args.n_boot,
                                        seed=args.seed, min_rows=args.min_rows, min_episodes=args.min_episodes,
                                        min_event_episodes=args.min_event_episodes, min_valid_fraction=args.min_valid_fraction),
        "joint_refit": joint_refit(rows, cost, alpha=float(record["alpha"])),
        "run": run_summary(decisions, episodes, cost),
        "code_sha256": code_sha256(), "git_commit": git_commit(),
        "reading_legend": {
            "within_preset_tolerance_in_support": "interval upper bound <= alpha + tol on in-support rows (approximate cluster-bootstrap diagnostic, not a proof of exact alpha calibration)",
            "risk_above_preset_tolerance_in_support": "interval lower bound > alpha + tol on in-support rows",
            "insufficient_evidence": "an information / degeneracy gate failed or the interval straddles alpha + tol",
        },
    }
    (out_dir / "verify.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for tier, b in report["exceedance"]["tiers"].items():
        print(f"{tier}: n_in={b.get('n_in_support')} point={b.get('point_estimate')} interval={b.get('interval')} -> {b.get('reading')}")


def main() -> None:
    """CLI with the three stages ``collect`` / ``label`` / ``report``."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--log-root", required=True)
    c.add_argument("--run-tag", default="verify")
    c.add_argument("--pool-manifest", required=True)
    c.add_argument("--frozen-record", required=True)
    c.add_argument("--n-sample", type=int, default=DEFAULTS["n_sample"])
    c.add_argument("--seed", type=int, default=20260914)
    c.add_argument("--out-dir", required=True)
    c.set_defaults(func=cmd_collect)
    lb = sub.add_parser("label")
    lb.add_argument("--suite", required=True)
    lb.add_argument("--frozen-record", required=True)
    lb.add_argument("--sampled-jsonl", required=True)
    lb.add_argument("--sample-manifest", required=True)
    lb.add_argument("--library-pkl", required=True)
    lb.add_argument("--template-yaml", required=True)
    lb.add_argument("--checkpoint", required=True)
    lb.add_argument("--denoising-steps", type=int, default=8)
    lb.add_argument("--h-exec", type=int, default=5)
    lb.add_argument("--root-seed", type=int, default=20260914)
    lb.add_argument("--limit", type=int, default=0, help="debug: label the first N rows into a *partial* product")
    lb.add_argument("--device", default="cuda")
    lb.add_argument("--out-dir", required=True)
    lb.set_defaults(func=cmd_label)
    r = sub.add_parser("report")
    r.add_argument("--suite", required=True)
    r.add_argument("--frozen-record", required=True)
    r.add_argument("--label-record", required=True)
    r.add_argument("--sample-manifest", required=True)
    r.add_argument("--verify-rows", required=True)
    r.add_argument("--episodes", required=True)
    r.add_argument("--decisions", required=True)
    r.add_argument("--fits", required=True)
    r.add_argument("--cost", required=True)
    r.add_argument("--tol", type=float, default=DEFAULTS["tol"])
    r.add_argument("--n-boot", type=int, default=DEFAULTS["n_boot"])
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--min-rows", type=int, default=DEFAULTS["min_rows"])
    r.add_argument("--min-episodes", type=int, default=DEFAULTS["min_episodes"])
    r.add_argument("--min-event-episodes", type=int, default=DEFAULTS["min_event_episodes"])
    r.add_argument("--min-valid-fraction", type=float, default=DEFAULTS["min_valid_fraction"])
    r.add_argument("--out-dir", required=True)
    r.set_defaults(func=cmd_report)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
