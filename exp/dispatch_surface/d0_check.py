"""D0 data-semantics stop-loss: replay logged D_dev observations, never a rollout.

Rev 1 makes the existing Y tables its input, so before anything is refitted the
tables have to be shown to mean what the pipeline assumes. D0 does that by
re-deriving, on the real model, the two identities the table construction relies
on, plus a read-only decomposition of the extreme rows that set the residual
tail. It touches only observations already recorded in the query cohort and the
library source; it starts no simulator and never reads A'.

Three checks (plan section 8):

  1. self-resume parity -- under CURRENT conditioning with the winner's z_j, a
     full stage-3 run and a resume from that run's own x_{c,0.3} must agree.
     Failing this means run_stage3_from does not reproduce the trajectory it
     claims to continue, and every Y_7 in the table is meaningless.
  2. payload/sidecar identity -- replaying the winner under ITS OWN conditioning
     with the sidecar z_j must reproduce the pickled action chunk and the stored
     t=0.3 intermediate. Failing this means the library, the sidecar and the
     table are not describing the same generation.
  3. sampled table-semantics replay -- for every selected extreme/control row,
     recompute y7/y10 with the deployed warm path and require the stored table
     values to match. The additional path decomposition remains diagnostic.

All three checks are hard gates. In particular, check 3 prevents a table built
before a warm-path implementation change from being certified for a later
deployment merely because both runs point at files with plausible names.

The non-GPU census (schema, dtypes, joins, sidecar coverage) runs over EVERY
D_dev row and EVERY library entry -- the GPU sample is a sample, the identity
census is not.

Usage:
  uv run python -m exp.dispatch_surface.d0_check \
      --table exp/dispatch_surface/data/dispatch_table_fresh.jsonl \
      --query-h5-dir exp/dispatch_surface/data/query_cohort \
      --library-pkl <lib.pkl> --noise-sidecar <sidecar.npz> \
      --library-h5-dir <dlib h5 dir> \
      --cache-yaml <calibration yaml> --checkpoint-dir <ckpt> \
      --out exp/dispatch_surface/analysis/d0_<suite>.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import pickle

import h5py
import numpy as np
import torch

from openpi.cache.components.surface_judge import weighted_chunk_deviation

START_T_WS = 0.3
EXPECTED_NUM_STEPS = 10
CONTROL_ROWS_PER_TASK = 2
CONTROL_SEED = 20260828
# Frozen tolerance for both parity checks; a looser one would let a genuinely
# different trajectory pass as "close enough".
PARITY_RTOL = 1e-4
PARITY_ATOL = 1e-4
D0_PROTOCOL = "dispatch_surface_rev1_d0_v2"
FORMAL_SUITES = frozenset({"libero_10", "libero_spatial"})
FORMAL_H_EXEC = 5
EXPECTED_TASKS = 10


def accumulated_start_t(start_t: float, num_steps: int) -> float:
    """The timestep the full denoising loop holds at ``start_t`` (reference).

    Kept as the executable record of why the parity check once failed, and as
    the value ``run_stage3_from`` now seeds internally. D0 itself calls the
    frozen literal tier; this helper is not on the gate path.

    ``_stage3_with_intermediates`` walks ``timestep`` from 1.0 by repeated
    float32 addition of ``dt=-1/num_steps``. ``run_stage3_from`` now reproduces
    that accumulator internally while retaining the public literal tier
    argument. This helper documents the invariant; D0 deliberately calls the
    public method with the frozen literal, exactly like deployment.
    """
    t = torch.tensor(1.0, dtype=torch.float32)
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32)
    for _ in range(round((1.0 - start_t) * num_steps)):
        t = t + dt
    return float(t)


def step_group_index(h5) -> dict[int, str]:
    """Map parsed step index -> the group's ACTUAL key name.

    The H5 writes zero-padded names (``step_0000``), while every table column
    carries the parsed integer. Reconstructing ``f"step_{i}"`` therefore misses
    every group; the table builder only ever parses, so the checker must too.
    """
    out: dict[int, str] = {}
    for key in h5.keys():
        if not key.startswith("step_"):
            continue
        try:
            idx = int(key.split("_", 1)[1])
        except ValueError as exc:
            raise ValueError(f"malformed H5 step group {key!r}") from exc
        if idx in out:
            raise ValueError(
                f"H5 step groups {out[idx]!r} and {key!r} both parse as step {idx}"
            )
        out[idx] = key
    return out


def _load_rows(table_path: str) -> list[dict]:
    return [json.loads(line) for line in open(table_path)]


def select_gpu_sample(
    rows: list[dict], seed: int, *, expected_tasks: set[int] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Every row above the y7 p99, plus a fixed per-task control draw.

    The extremes are taken whole rather than sampled: they are the rows the
    residual tail is made of, so a sample of them would leave the question open.
    Controls are drawn without replacement from what remains, per task, with a
    frozen seed so the sample is a property of the table and not of the run.
    """
    y7 = np.array([r["y_tau7"] for r in rows], dtype=float)
    thr = float(np.percentile(y7, 99))
    extreme = [r for r in rows if r["y_tau7"] > thr]
    extreme_keys = {(r["episode_id"], r["step_idx"]) for r in extreme}

    rng = np.random.default_rng(seed)
    by_task: dict[int, list[dict]] = collections.defaultdict(list)
    for r in rows:
        if (r["episode_id"], r["step_idx"]) not in extreme_keys:
            by_task[r["task_id"]].append(r)
    tasks = set(by_task) if expected_tasks is None else expected_tasks
    if set(by_task) != tasks:
        raise ValueError(
            f"control pools cover tasks {sorted(by_task)}, expected {sorted(tasks)}"
        )
    controls: list[dict] = []
    for task_id in sorted(tasks):
        pool = sorted(by_task[task_id], key=lambda r: (r["episode_id"], r["step_idx"]))
        if len(pool) < CONTROL_ROWS_PER_TASK:
            raise ValueError(
                f"task {task_id} has only {len(pool)} non-extreme rows; "
                f"D0 requires exactly {CONTROL_ROWS_PER_TASK} controls"
            )
        for i in rng.choice(len(pool), size=CONTROL_ROWS_PER_TASK, replace=False):
            controls.append(pool[int(i)])
    return extreme, controls


def census_identity(rows: list[dict], entries: list, sidecar) -> dict:
    """Non-GPU identity/schema census over EVERY row and EVERY library entry."""
    problems: list[str] = []

    # A duplicate entry id would be silently collapsed by the id->entry map the
    # replay uses, so one payload would stand in for another.
    seen_ids: set[str] = set()
    for e in entries:
        if e.id in seen_ids:
            problems.append(f"duplicate library entry id {e.id}")
        seen_ids.add(e.id)

    seen: set[tuple] = set()
    for r in rows:
        key = (r["episode_id"], r["step_idx"])
        if key in seen:
            problems.append(f"duplicate table row for {key}")
        seen.add(key)

    by_entry_key: dict[tuple, str] = {}
    entry_ids: set[str] = set()
    for e in entries:
        if e.id in entry_ids:
            problems.append(f"duplicate library entry id {e.id!r}")
        entry_ids.add(e.id)
        payload = e.payload
        action = np.asarray(payload.action_chunk)
        if action.dtype != np.float32:
            problems.append(f"entry {e.id}: action dtype {action.dtype} != float32")
        if payload.denoising_num_steps != EXPECTED_NUM_STEPS:
            problems.append(
                f"entry {e.id}: denoising_num_steps={payload.denoising_num_steps} != {EXPECTED_NUM_STEPS}"
            )
        chunk = torch.as_tensor(payload.action_chunk)
        if chunk.dtype != torch.float32:
            problems.append(f"entry {e.id}: action_chunk dtype {chunk.dtype} != float32")
        if chunk.ndim != 2:
            problems.append(f"entry {e.id}: action_chunk ndim {chunk.ndim} != 2")
        inter = payload.intermediates
        if not inter or START_T_WS not in inter:
            problems.append(f"entry {e.id}: no intermediates[{START_T_WS}]")
        else:
            x3 = inter[START_T_WS]
            if tuple(x3.shape) != tuple(np.asarray(payload.action_chunk).shape):
                problems.append(f"entry {e.id}: intermediate shape {tuple(x3.shape)} != chunk shape")
            if torch.as_tensor(x3).dtype != torch.float32:
                problems.append(f"entry {e.id}: intermediate dtype {torch.as_tensor(x3).dtype} != float32")
        if e.id not in sidecar:
            problems.append(f"entry {e.id}: missing from noise sidecar")
        else:
            z = np.asarray(sidecar[e.id])
            if z.shape != tuple(np.asarray(payload.action_chunk).shape):
                problems.append(f"entry {e.id}: sidecar shape {z.shape} != chunk shape")
            if z.dtype != np.float32:
                problems.append(f"entry {e.id}: sidecar dtype {z.dtype} != float32")
        key = (e.trajectory_id, e.step_idx)
        if key in by_entry_key:
            problems.append(f"entry join collision at {key}: {by_entry_key[key]} and {e.id}")
        by_entry_key[key] = e.id

    winners = {r["winner_id"] for r in rows}
    known = {e.id for e in entries}
    for wid in sorted(winners - known):
        problems.append(f"table winner_id {wid} is not in the library")

    return {
        "rows": len(rows),
        "entries": len(entries),
        "unique_winners": len(winners),
        "problems": problems,
        "passed": not problems,
    }


def _index_library_h5(h5_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    from exp.common.build_in_memory_cache_artifact import resolve_h5_paths

    return _index_h5_paths(resolve_h5_paths(h5_dir, None), label="library H5")


def _index_h5_paths(paths, *, label: str) -> dict[str, pathlib.Path]:
    out: dict[str, pathlib.Path] = {}
    for raw in paths:
        path = pathlib.Path(raw).resolve()
        if path.stem in out:
            raise SystemExit(
                f"{label} basename collision for {path.stem!r}: {out[path.stem]} and {path}"
            )
        out[path.stem] = path
    return out


def _file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def _canonical_digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _h5_tree_attestation(root: pathlib.Path, paths) -> dict:
    resolved_root = root.resolve()
    files = []
    seen_rel: set[str] = set()
    for raw in sorted((pathlib.Path(p).resolve() for p in paths), key=str):
        try:
            rel = raw.relative_to(resolved_root).as_posix()
        except ValueError as exc:
            raise SystemExit(f"H5 path escapes declared root {resolved_root}: {raw}") from exc
        if rel in seen_rel:
            raise SystemExit(f"duplicate H5 path in attestation: {rel}")
        seen_rel.add(rel)
        files.append({"relative_path": rel, "sha256": _file_sha256(raw)})
    if not files:
        raise SystemExit(f"no H5 files under {resolved_root}")
    return {
        "path": str(resolved_root),
        "files": files,
        "rollup_sha256": _canonical_digest(files),
    }


def build_input_attestation(args, query_paths, library_paths) -> dict:
    from openpi.serving.policy_identity import compute_policy_fingerprint, resolve_checkpoint_root

    files = {}
    for name, raw in (
        ("table", args.table),
        ("library_pkl", args.library_pkl),
        ("noise_sidecar", args.noise_sidecar),
        ("cache_yaml", args.cache_yaml),
        ("weights_npz", args.weights_npz),
    ):
        path = pathlib.Path(raw).resolve()
        if not path.is_file():
            raise SystemExit(f"D0 input {name} is missing: {path}")
        files[name] = {"path": str(path), "sha256": _file_sha256(path)}
    checkpoint = resolve_checkpoint_root(args.checkpoint_dir)
    out = {
        "files": files,
        "query_h5": _h5_tree_attestation(pathlib.Path(args.query_h5_dir), query_paths),
        "library_h5": _h5_tree_attestation(pathlib.Path(args.library_h5_dir), library_paths),
        "policy": {
            "checkpoint_dir": str(checkpoint),
            "config_name": args.config_name,
            "policy_fingerprint": compute_policy_fingerprint(str(checkpoint), args.config_name),
        },
    }
    out["rollup_sha256"] = _canonical_digest(out)
    return out


def validate_input_attestation(attestation: dict) -> None:
    """Recompute every D0 input digest from the recorded paths.

    The D0 JSON is consumed later by ``fit_surface``. Merely hashing the JSON
    would authenticate a statement about paths, not the bytes that were
    replayed, so the consumer must be able to re-attest those bytes.
    """
    expected_files = {"table", "library_pkl", "noise_sidecar", "cache_yaml", "weights_npz"}
    files = attestation.get("files")
    if not isinstance(files, dict) or set(files) != expected_files:
        raise ValueError("D0 attestation file set is incomplete or contains extras")
    for name, item in files.items():
        path = pathlib.Path(item.get("path", "")).resolve()
        if not path.is_file() or _file_sha256(path) != item.get("sha256"):
            raise ValueError(f"D0 attested file {name!r} is missing or content-drifted")

    for name in ("query_h5", "library_h5"):
        tree = attestation.get(name)
        if not isinstance(tree, dict) or not isinstance(tree.get("files"), list):
            raise ValueError(f"D0 attestation lacks {name} tree")
        root = pathlib.Path(tree.get("path", "")).resolve()
        records = []
        for item in tree["files"]:
            path = (root / str(item.get("relative_path", ""))).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"D0 {name} path escapes its root: {path}") from exc
            if not path.is_file() or _file_sha256(path) != item.get("sha256"):
                raise ValueError(f"D0 {name} file is missing or content-drifted: {path}")
            records.append({"relative_path": item["relative_path"], "sha256": item["sha256"]})
        if not records or _canonical_digest(records) != tree.get("rollup_sha256"):
            raise ValueError(f"D0 {name} rollup is empty or invalid")

    policy = attestation.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("D0 attestation lacks policy identity")
    from openpi.serving.policy_identity import compute_policy_fingerprint

    actual_fp = compute_policy_fingerprint(policy.get("checkpoint_dir", ""), policy.get("config_name", ""))
    if actual_fp != policy.get("policy_fingerprint"):
        raise ValueError("D0 policy checkpoint/config content drifted")

    payload = {k: v for k, v in attestation.items() if k != "rollup_sha256"}
    if _canonical_digest(payload) != attestation.get("rollup_sha256"):
        raise ValueError("D0 input attestation rollup is invalid")


def main() -> None:  # noqa: PLR0915 - one linear replay script; splitting hides the order
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", required=True)
    ap.add_argument("--query-h5-dir", required=True)
    ap.add_argument("--library-pkl", required=True)
    ap.add_argument("--noise-sidecar", required=True)
    ap.add_argument("--library-h5-dir", required=True)
    ap.add_argument("--cache-yaml", required=True)
    ap.add_argument("--config-name", default="pi05_libero")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--weights-npz", required=True)
    ap.add_argument("--h-exec", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.suite not in FORMAL_SUITES:
        raise SystemExit(f"formal D0 suite must be one of {sorted(FORMAL_SUITES)}")
    if args.h_exec != FORMAL_H_EXEC:
        raise SystemExit(f"formal D0 requires --h-exec {FORMAL_H_EXEC}")

    from exp.common.build_in_memory_cache_artifact import (
        _build_fake_stage1_with_masks,
        _load_pi05_for_llm_extract,
        resolve_h5_paths,
    )

    query_h5_paths = resolve_h5_paths(args.query_h5_dir, None)
    library_h5_paths = resolve_h5_paths(args.library_h5_dir, None)
    query_paths = _index_h5_paths(query_h5_paths, label="query H5")
    lib_paths = _index_h5_paths(library_h5_paths, label="library H5")

    rows = _load_rows(args.table)
    with open(args.library_pkl, "rb") as f:
        data = pickle.load(f)
    entries = data["entries"]
    by_id = {e.id: e for e in entries}
    sidecar = np.load(args.noise_sidecar)
    weights = np.load(args.weights_npz)
    w = torch.as_tensor(weights["w"], dtype=torch.float32)
    active_mask = torch.as_tensor(weights["active_mask"], dtype=torch.bool)

    report: dict = {"protocol": D0_PROTOCOL, "suite": args.suite, "table": args.table,
                    "tolerance": {"rtol": PARITY_RTOL, "atol": PARITY_ATOL},
                    "resume_start_t": START_T_WS, "h_exec": args.h_exec,
                    "inputs": build_input_attestation(args, query_h5_paths, library_h5_paths)}

    # ---- non-GPU census over everything -----------------------------
    report["census"] = census_identity(rows, entries, sidecar)

    # ---- GPU replay sample ------------------------------------------
    extreme, controls = select_gpu_sample(
        rows, CONTROL_SEED, expected_tasks=set(range(EXPECTED_TASKS)),
    )
    sample = extreme + controls
    extreme_keys = {(r["episode_id"], r["step_idx"]) for r in extreme}
    report["sample"] = {
        "y7_p99": float(np.percentile([r["y_tau7"] for r in rows], 99)),
        "extreme_rows": len(extreme),
        "control_rows": len(controls),
        "control_seed": CONTROL_SEED,
        "tasks_covered": sorted({r["task_id"] for r in sample}),
        "rows_sha256": _canonical_digest([
            {
                "kind": "extreme" if (r["episode_id"], r["step_idx"]) in extreme_keys else "control",
                "episode_id": r["episode_id"], "step_idx": r["step_idx"],
                "task_id": r["task_id"], "winner_id": r["winner_id"],
            }
            for r in sample
        ]),
    }

    dev = torch.device(args.device)
    model, tokenizer = _load_pi05_for_llm_extract(args.checkpoint_dir, args.config_name, args.device)

    check1: list[dict] = []
    check2: list[dict] = []
    check3: list[dict] = []
    c1_fail = c2_fail = 0

    for r in sample:
        ep, step_idx, wid = r["episode_id"], r["step_idx"], r["winner_id"]
        entry = by_id[wid]
        z = torch.from_numpy(np.asarray(sidecar[wid])).to(dev)[None]

        # -- current conditioning from the QUERY observation ----------
        qpath = query_paths.get(ep)
        if qpath is None:
            raise SystemExit(f"table episode {ep!r} has no unique query H5")
        with h5py.File(qpath, "r") as h5:
            task_str = h5.attrs.get("prompt") or h5.attrs.get("task", "")
            names = step_group_index(h5)
            if step_idx not in names:
                raise SystemExit(
                    f"{qpath.name}: table row references step {step_idx} which the H5 lacks"
                )
            group = h5[names[step_idx]]
            stage1_c = _build_fake_stage1_with_masks(group, str(task_str), tokenizer, model, dev)
            with torch.no_grad():
                stage2_c = model.run_stage2(stage1_c)
                full = model.run_stage3(
                    stage2_c, noise=z, return_intermediates=True, save_timesteps=(START_T_WS,),
                )
                x_c_03 = full.intermediates[START_T_WS]
                # The frozen call: the pinned tier, exactly as the deployed warm
                # path invokes it. Nothing here substitutes a friendlier
                # argument -- the parity being checked is the one production
                # relies on.
                resumed = model.run_stage3_from(
                    stage2_c, x_c_03, START_T_WS, num_steps=EXPECTED_NUM_STEPS,
                )
        a_full = full.action_chunk[0].float().cpu()
        a_res = resumed.action_chunk[0].float().cpu()
        d1 = float((a_full - a_res).abs().max())
        ok1 = bool(torch.allclose(a_full, a_res, rtol=PARITY_RTOL, atol=PARITY_ATOL))
        c1_fail += (not ok1)
        check1.append({"episode_id": ep, "step_idx": step_idx,
                       "max_abs_diff": d1, "passed": ok1})

        # -- winner's OWN conditioning from the LIBRARY observation ---
        lpath = lib_paths.get(entry.trajectory_id)
        if lpath is None:
            check2.append({"winner_id": wid, "passed": False,
                           "error": f"no library H5 for trajectory {entry.trajectory_id}"})
            c2_fail += 1
            continue
        with h5py.File(lpath, "r") as h5:
            task_str = h5.attrs.get("prompt") or h5.attrs.get("task", "")
            names = step_group_index(h5)
            if entry.step_idx not in names:
                raise SystemExit(
                    f"{lpath.name}: library entry {wid} references step {entry.step_idx} "
                    "which the H5 lacks"
                )
            group = h5[names[entry.step_idx]]
            stage1_j = _build_fake_stage1_with_masks(group, str(task_str), tokenizer, model, dev)
            with torch.no_grad():
                stage2_j = model.run_stage2(stage1_j)
                regen = model.run_stage3(
                    stage2_j, noise=z, return_intermediates=True, save_timesteps=(START_T_WS,),
                )
        a_regen = regen.action_chunk[0].float().cpu()
        x_j_03_regen = regen.intermediates[START_T_WS][0].float().cpu()
        a_stored = torch.as_tensor(entry.payload.action_chunk)
        x_j_03 = torch.as_tensor(entry.payload.intermediates[START_T_WS])
        d2a = float((a_regen - a_stored).abs().max())
        d2x = float((x_j_03_regen - x_j_03).abs().max())
        ok2 = bool(
            torch.allclose(a_regen, a_stored, rtol=PARITY_RTOL, atol=PARITY_ATOL)
            and torch.allclose(x_j_03_regen, x_j_03, rtol=PARITY_RTOL, atol=PARITY_ATOL)
            and entry.payload.denoising_num_steps == EXPECTED_NUM_STEPS
        )
        c2_fail += (not ok2)
        check2.append({"winner_id": wid, "trajectory_id": entry.trajectory_id,
                       "step_idx": entry.step_idx, "max_abs_diff_action": d2a,
                       "max_abs_diff_intermediate": d2x, "passed": ok2})

        # -- diagnostic decomposition (no gate) -----------------------
        with torch.no_grad():
            warm = model.run_stage3_from(
                stage2_c, x_j_03.to(dev)[None], START_T_WS, num_steps=EXPECTED_NUM_STEPS,
            )
        a_warm = warm.action_chunk[0].float().cpu()
        direct_dev = weighted_chunk_deviation(a_stored, a_full, w, active_mask, args.h_exec)
        warm_dev = weighted_chunk_deviation(a_warm, a_full, w, active_mask, args.h_exec)
        table_match = bool(
            np.isclose(direct_dev, r["y_tau10"], rtol=PARITY_RTOL, atol=PARITY_ATOL)
            and np.isclose(warm_dev, r["y_tau7"], rtol=PARITY_RTOL, atol=PARITY_ATOL)
        )
        check3.append({
            "episode_id": ep, "step_idx": step_idx, "winner_id": wid,
            "is_extreme": r["y_tau7"] > report["sample"]["y7_p99"],
            "intermediate_gap": float((x_j_03 - x_c_03[0].float().cpu()).norm()),
            "dev_direct_full": direct_dev,
            "dev_warm": warm_dev,
            "table_y_tau7": r["y_tau7"], "table_y_tau10": r["y_tau10"],
            "table_semantics_match": table_match, "s": r["s"], "v": r["v"],
        })

    report["check1_self_resume_parity"] = {
        "n": len(check1), "failures": c1_fail, "passed": c1_fail == 0,
        "max_abs_diff": max((c["max_abs_diff"] for c in check1), default=0.0),
        "rows": check1,
    }
    report["check2_payload_sidecar_identity"] = {
        "n": len(check2), "failures": c2_fail, "passed": c2_fail == 0,
        "rows": check2,
    }
    report["check3_path_decomposition"] = {
        "n": len(check3), "complete": len(check3) == len(sample),
        "table_semantics_passed": bool(check3) and all(r["table_semantics_match"] for r in check3),
        "rows": check3,
    }

    gates = (report["census"]["passed"]
             and report["check1_self_resume_parity"]["passed"]
             and report["check2_payload_sidecar_identity"]["passed"]
             and report["check3_path_decomposition"]["complete"]
             and report["check3_path_decomposition"]["table_semantics_passed"])
    report["D0"] = "PASS" if gates else "FAIL"

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("check1_self_resume_parity", "check2_payload_sidecar_identity",
                                   "check3_path_decomposition")}, indent=2))
    print(f"check1 failures={c1_fail}  check2 failures={c2_fail}  -> D0 {report['D0']}")
    if not gates:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
