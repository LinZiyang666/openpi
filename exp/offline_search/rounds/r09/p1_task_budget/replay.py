"""CPU replay of discovery-only A/CU/P10 logged decisions.

Compare every returned field/action byte between each emitted task switch and
its original R8 branch. Reconstruct online histories from saved normalized
state, exact post-projection PCA keys, hits and served chunks, without policy calls or simulation.
The compact pi05 table retains its full 10-control valid action. GR00T's
full 16-control chunks are read from the original capture for tail replay.
"""
from __future__ import annotations

import argparse
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd

from exp.offline_search.closed_loop.blind import BlindResult, LookReason, policy_tail_chunk
from exp.offline_search.closed_loop.plugin import clone_method

from .common import CELLS, DERIVED, HERE, R8, RUN, fit_path, r8_name, write_json


def assert_same(a, b):
    if type(a) is not type(b):
        raise AssertionError(f"result type changed: {type(a)} vs {type(b)}")
    if isinstance(a, LookReason):
        assert (a.code, a.name) == (b.code, b.name)
        return
    attrs = ("action", "rows", "weights") if isinstance(a, BlindResult) else ("action", "topk", "scores")
    for name in attrs:
        x, y = getattr(a, name), getattr(b, name)
        assert (x.shape, x.dtype, x.tobytes()) == (y.shape, y.dtype, y.tobytes()), name
    assert a.library == b.library
    if not isinstance(a, BlindResult):
        assert a.confidence == b.confidence
    assert a.extras == b.extras


def load_method(row):
    with fit_path(row).open("rb") as f:
        b = pickle.load(f)
    assert b["spec"] == row["method"] and b["kwargs"] == row["kwargs"] and b["cell"] == row["cell"]
    method = clone_method(b["method"], strict=True)[0]
    controllers = method.controllers.values() if hasattr(method, "controllers") else [method]
    for controller in controllers:
        base = controller.base if hasattr(controller, "base") else controller
        # observer.diagnostics records xv AFTER projection and mean subtraction.
        # Raw 32768-D keys were sampled, so use identity projection for replay
        # clones only. Frozen files and downstream distance/synthesis are exact.
        for camera in (0, 1):
            width = getattr(base, f"B{camera}T").shape[0]
            setattr(base, f"B{camera}T", np.eye(width, dtype=np.float32))
            setattr(base, f"muB{camera}", np.zeros(width, np.float32))
            # AWM's optional adjacent-key diagnostic uses the original key mean;
            # it does not affect retrieval/calls and is removed after blind gaps.
            setattr(base, f"mu{camera}", np.zeros(width, np.float32))
    return method


def logged_tape(cell, variant, inits):
    name = r8_name(cell, variant)
    df = pd.read_parquet(DERIVED / "decisions" / (name + ".parquet"),
                         filters=[("init", ">=", 0), ("init", "<", inits)])
    df = df.sort_values(["task_id", "init", "decision_seq"])
    ids = df.decision_id.tolist()
    with np.load(DERIVED / "arrays" / (name + ".npz")) as f:
        lookup = {v: i for i, v in enumerate(f["decision_id"])}
        sel = np.array([lookup[v] for v in ids])
        arr = {k: f[k][sel] for k in ("served", "keys_third", "keys_wrist", "state", "rows", "weights")}
    with np.load(DERIVED / "arrays_norm" / (name + ".npz")) as f:
        lookup = {v: i for i, v in enumerate(f["decision_id"])}
        arr["rs"] = f["state_norm"][[lookup[v] for v in ids]]
    df = df.reset_index(drop=True)
    if cell.startswith("groot"):
        from exp.offline_search.debug import reader
        capture = reader.open_arm(R8, name)
        capture.cache_enabled = False
        arr["full_served"] = capture.decision_arrays(["served_chunk"], ids)["served_chunk"]
    else:
        arr["full_served"] = np.zeros((len(df), 10, 32), np.float32)
        arr["full_served"][:, :, :7] = arr["served"]
    assert set(zip(df.task_id, df.init)) == {(t, i) for t in range(10) for i in range(inits)}
    return df, arr


def replay_episode(reference, switches, df, arr, indices, horizon, stats):
    first = df.iloc[indices[0]]
    ep = SimpleNamespace(uid=str(first.episode_key), task_id=int(first.task_id), init=int(first.init))
    methods = [reference, *switches]
    for m in methods:
        m.reset(ep)
    rows = df.iloc[indices]
    assert rows.decision_seq.tolist() == list(range(len(indices)))
    served = arr["full_served"][indices]
    assert served.shape == (len(indices), horizon, 32)
    rs, v0, v1, raw = (arr[k][indices] for k in ("rs", "keys_third", "keys_wrist", "state"))
    vision, hits = rows.is_look.to_numpy(bool), rows.hit.to_numpy(bool)
    age = 0
    last_policy = -2
    for step, ix in enumerate(indices):
        q = SimpleNamespace(episode=ep, task_id=ep.task_id, step=step, rs=rs[step], raw_state=raw[step],
            key_v0=v0[step], key_v1=v1[step], hist_rs=rs[:step], hist_raw_state=raw[:step],
            hist_key_v0=v0[:step], hist_key_v1=v1[:step], hist_a_exec=served[:step],
            hist_has_vision=vision[:step], hist_hit=hits[:step].astype(np.int8),
            prev_a_exec=served[step-1] if step else None, prev_hit=bool(hits[step-1]) if step else None,
            blind_age=age, executed_steps=5)
        use_tail = bool(step and (not hits[step-1] or step == last_policy + 1))
        proposals = []
        for m in methods:
            if step == 0 or (use_tail and step != last_policy + 1):
                result = LookReason(6, "lifecycle")
            else:
                hook = m.policy_tail_step if use_tail else m.blind_step
                result = hook(q)
            if isinstance(result, LookReason):
                # Missing keys on a logged blind decision cannot be invented.
                assert vision[step], (ep.task_id, ep.init, step, "unexpected replay LOOK", result)
                result = m.query(q)
            else:
                assert not vision[step], (ep.task_id, ep.init, step, "unexpected replay BLIND")
            proposals.append(result)
        for result in proposals[1:]:
            assert_same(proposals[0], result)
            stats["branch_exact_decisions"] += 1
        call = bool(proposals[0].extras.get("os_force_miss", 0))
        assert call == bool(rows.iloc[step].is_call), (ep.task_id, ep.init, step, "call mismatch")
        stats["decisions"] += 1
        stats["calls"] += int(call)
        stats["policy_tails"] += int(use_tail and not vision[step])
        if call:
            last_policy = step
            for m in methods:
                m.invalidate_anchor()
        else:
            action = proposals[0].action[:, :7]
            logged = arr["full_served"][ix, :, :7]
            stats["logged_action_exact"] += int(action.tobytes() == logged.tobytes())
            stats["logged_action_checked"] += 1
            err = float(np.max(np.abs(action - logged)))
            stats["logged_action_max_abs"] = max(stats["logged_action_max_abs"], err)
            np.testing.assert_allclose(action, logged, rtol=0, atol=2e-5)
            if use_tail and not vision[step]:
                np.testing.assert_array_equal(action, policy_tail_chunk(served[step-1])[:, :7])
        age = 0 if vision[step] else age + 1
    stats["episodes"] += 1


def run(inits=30):
    if not 1 <= inits <= 30:
        raise ValueError("replay restricted to discovery inits 0-29")
    r8 = json.loads((HERE / "source_lock.json").read_text())["cells"]
    r9 = {r["arm"]: r for r in json.loads((RUN / "arms.json").read_text())}
    report = dict(status="PASS", inits=list(range(inits)), scope="post-projection logged observations; no policy/simulator",
                  limitation="raw keys sampled only; replay replaces projection in detached clones with identity on exact captured PCA codes",
                  logged_action_horizons={"pi05": 10, "groot": 16}, logged_action_valid_dims=7,
                  policy_action_limitation="policy calls replay historical served chunks; no policy forward is recomputed",
                  branch_equality="all result fields, full action bytes, extras, LOOK/call decisions",
                  logged_action_atol=2e-5, checks=[])
    for cell in CELLS:
        wrappers = {v: load_method(r9[f"r9p1_{cell}_{v}"]) for v in ("topk_cu", "topk_p10")}
        horizon = 10 if cell.startswith("pi05") else 16
        for variant, branch in (("A", "A"), ("CU", "CU:0.3"), ("P10", "P10")):
            baseline = load_method(r8[cell][variant]["row"])
            df, arrays = logged_tape(cell, variant, inits)
            stats = dict(cell=cell, branch=variant, episodes=0, decisions=0, branch_exact_decisions=0,
                         calls=0, policy_tails=0, logged_action_exact=0, logged_action_checked=0,
                         logged_action_max_abs=0.)
            for (task, init), group in df.groupby(["task_id", "init"], sort=True):
                selected = [m for m in wrappers.values() if m.branch(task) == branch]
                if selected:
                    replay_episode(baseline, selected, df, arrays, group.index.to_numpy(), horizon, stats)
            print(json.dumps(stats, sort_keys=True), flush=True)
            report["checks"].append(stats)
    target = HERE / "evidence" / ("replay.json" if inits == 30 else f"replay_{inits}.json")
    write_json(target, report)
    print(f"P1_REPLAY_OK checks={len(report['checks'])} inits={inits} branch_exact="
          f"{sum(c['branch_exact_decisions'] for c in report['checks'])} report={target}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inits", type=int, default=30)
    run(ap.parse_args().inits)
