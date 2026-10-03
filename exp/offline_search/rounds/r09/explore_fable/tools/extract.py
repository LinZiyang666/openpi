"""Extract compact per-episode / per-decision tables from an osdebug.v1 arm.

For every accepted episode of an arm this writes

* ``<out>/episodes/<arm>.parquet`` -- one row per accepted (task, init) pair with
  success, decision counts, look / wrist / call counts and summed owner cost;
* ``<out>/decisions/<arm>.parquet`` -- one row per decision with the scalar
  serving fields (vision, src, owner_cost, d1, conf, ...), flattened method
  extras and the client join (control_idx_start, n_applied);
* ``<out>/arrays/<arm>.npz`` -- decision-aligned arrays: executed 10x7 served
  block, the deferred policy shadow (10x7), PCA keys, wire state, the 16
  retrieval rows / weights (look decisions only; -1 / NaN elsewhere).

The capture root is never written to. ``reader.ArmData`` caches are disabled.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from exp.offline_search.debug import reader

from . import common

EXTRA_KEYS = ("d1", "d1_rel", "disp5", "dst", "w_eff", "regime", "lib_ep", "lib_step", "c0", "still",
              "os_force_miss", "os_reason", "os_c_R", "os_c_Ehat", "os_c_p", "os_c_call", "os_c_stall_call",
              "os_c_extra_look", "os_c_stall_state", "os_q2_episode_dose", "os_q2_task_p")
RAND_KEYS = ("p_nominal", "p_effective", "coin", "treatment", "eligible", "override", "budget_state", "stall_state", "cooldown")
SCALAR_KEYS = ("decision_id", "episode_key", "task_id", "init", "decision_seq", "vision", "src", "hit", "camera_mode",
               "blind_age_controls", "look_reason", "miss_reason", "owner_cost", "policy_calls", "stage1_calls",
               "anchor_decision_id", "chunk_offset", "served_len", "conf", "d1", "lib", "member_k_eff",
               "infer_ms", "control_idx_start", "n_applied", "journal_success")


def _flat_extras(series, keys, prefix):
    out = {f"{prefix}{k}": np.full(len(series), np.nan) for k in keys}
    for i, ex in enumerate(series):
        if isinstance(ex, dict):
            for k in keys:
                v = ex.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out[f"{prefix}{k}"][i] = v
                elif isinstance(v, bool):
                    out[f"{prefix}{k}"][i] = float(v)
    return out


def extract_arm(arm, run_root=common.RUN_ROOT, out=common.DERIVED, with_arrays=True, force=False):
    out = Path(out)
    ep_path, dec_path, arr_path = out / "episodes" / f"{arm}.parquet", out / "decisions" / f"{arm}.parquet", out / "arrays" / f"{arm}.npz"
    if not force and ep_path.exists() and dec_path.exists() and (arr_path.exists() or not with_arrays):
        return dict(arm=arm, status="cached")
    t0 = time.time()
    data = reader.open_arm(run_root, arm)
    data.cache_enabled = False
    model, suite, lib, variant = common.parse_arm(arm)
    eps = data.episodes()
    dec = data.decisions()
    dec = dec[dec.server_join.eq("verified")].copy()
    dec["task_id"] = dec.task_id.astype(int)
    dec["init"] = dec["init"].astype(int)
    dec["decision_seq"] = dec.decision_seq.astype(int)
    dec = dec.sort_values(["episode_key", "decision_seq"]).reset_index(drop=True)

    scal = {k: dec[k].values if k in dec else np.full(len(dec), np.nan) for k in SCALAR_KEYS}
    scal.update(_flat_extras(dec["extras"] if "extras" in dec else [None] * len(dec), EXTRA_KEYS, "x_"))
    for k in RAND_KEYS:
        if k in dec:
            v = dec[k]
            scal[f"r_{k}"] = pd.to_numeric(v, errors="coerce").values if k != "override" else v.astype(str).values
    table = pd.DataFrame(scal)
    table["is_call"] = table.policy_calls.fillna(0).astype(int) > 0
    table["is_look"] = table.vision.fillna(False).astype(bool)
    table["is_wrist"] = table.camera_mode.eq("wrist_only")
    table["arm"], table["model"], table["suite"], table["lib"], table["variant"] = arm, model, suite, lib if lib else -1, variant

    # per-episode aggregates
    g = table.groupby("episode_key")
    agg = pd.DataFrame(dict(n_dec=g.size(), n_look=g.is_look.sum(), n_call=g.is_call.sum(), n_wrist=g.is_wrist.sum(),
                            cost=g.owner_cost.sum(), n_applied=g.n_applied.sum()))
    epi = eps.set_index("episode_key")[["task_id", "init", "journal_success", "n_controls", "n_decisions", "termination_reason"]].join(agg, how="left")
    epi = epi.reset_index()
    epi["success"] = epi.journal_success.astype(bool)
    epi["arm"], epi["model"], epi["suite"], epi["lib"], epi["variant"] = arm, model, suite, lib if lib else -1, variant
    epi["task_id"] = epi.task_id.astype(int)
    epi["init"] = epi["init"].astype(int)
    epi["ir"] = epi.cost / epi.n_dec

    ep_path.parent.mkdir(parents=True, exist_ok=True)
    dec_path.parent.mkdir(parents=True, exist_ok=True)
    epi.to_parquet(ep_path, index=False)
    table.to_parquet(dec_path, index=False)

    status = dict(arm=arm, n_episodes=int(len(epi)), n_decisions=int(len(table)), sr=float(epi.success.mean()),
                  ir=float(epi.cost.sum() / epi.n_dec.sum()), seconds=round(time.time() - t0, 1))
    if not with_arrays:
        return status
    ids = table.decision_id.tolist()
    H = int(data.server_meta.get("H", 10))
    arrays = data.decision_arrays(["served_chunk", "keys_pca_third", "keys_pca_wrist", "state_wire"], ids)
    served = arrays["served_chunk"][:, :10, :7].astype(np.float32)
    shadow = np.full_like(served, np.nan)
    try:
        sh = data.aug("policy_shadow", ids)
        shadow = sh["chunk"][:, :10, :7].astype(np.float32)
    except (KeyError, ValueError) as exc:  # arm without finished augmentation
        status["shadow"] = f"unavailable: {exc}"[:200]
    rows = np.full((len(ids), 16), -1, np.int32)
    weights = np.full((len(ids), 16), np.nan, np.float32)
    for i, (r, w) in enumerate(zip(dec["rows"], dec["weights"])):
        if isinstance(r, (list, np.ndarray)) and len(r) == 16:
            rows[i] = np.asarray(r, np.int32)
            weights[i] = np.asarray(w, np.float32)
    arr_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = arr_path.with_suffix(".tmp.npz")
    np.savez(tmp, decision_id=np.asarray(ids), served=served, shadow=shadow, keys_third=arrays["keys_pca_third"].astype(np.float32),
             keys_wrist=arrays["keys_pca_wrist"].astype(np.float32), state=arrays["state_wire"].astype(np.float32),
             rows=rows, weights=weights, H=np.int32(H))
    tmp.rename(arr_path)
    status["seconds"] = round(time.time() - t0, 1)
    return status


def _worker(args):
    arm, with_arrays, force = args
    try:
        return extract_arm(arm, with_arrays=with_arrays, force=force)
    except Exception as exc:  # report, never abort the pool
        return dict(arm=arm, status="error", error=repr(exc)[:500])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms", nargs="*", help="arm names (default: all arms in arms.json)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-arrays", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    arms = a.arms or [r["arm"] for r in common.load_arms()]
    jobs = [(arm, not a.no_arrays, a.force) for arm in arms]
    results = []
    with ProcessPoolExecutor(a.workers) as pool:
        for res in pool.map(_worker, jobs):
            results.append(res)
            print(json.dumps(res), flush=True)
    common.write_json(common.DERIVED / "extract_status.json", results)
    bad = [r for r in results if r.get("status") == "error"]
    print(f"done: {len(results)} arms, {len(bad)} errors", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
