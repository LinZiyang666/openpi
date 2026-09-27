"""Replay the plugin's logged per-episode inputs through the OFFLINE harness and compare decision by decision.

For every inputs/*.npz a plugin server wrote with --os-log-inputs, a mini store is built in the harness layout
(queries/<cell>/ from the logged keys / rs / raw_state / executed chunks; library/<m>_<s> symlinked to the real store),
the logged method is fitted exactly as the harness fits it and run with harness.run.run_jobs_inprocess, i.e. the
QueryView the offline harness would construct for that trajectory. Checks:

  * offline topk / scores / confidence / library / synthesized action / scalar extras == online (bit-exact); with
    ProbeB0 / ProbeHist the extras include crc32 digests of every QueryView field (keys, rs, raw_state, hist_*,
    prev_a_exec, prev_hit, hist_hit, episode identity), so equality proves the online inputs match
  * logged executed chunk == the chunk the method selected (library row action, or the synthesized action) on every
    HIT decision
  * native shadow (method mode) / native winner (native mode) vs offline B0Current on the same live keys
  * mixed mode (server started with --os-judge; the npz carries hit / judge / tau / run): MISS rows enter the mini
    store as policy rows (a_inf = the logged executed chunk, a_hit != it), so the offline replay sees prev_hit False /
    prev_a_exec = policy chunk exactly like the online session; every verdict is re-derived from the logged
    confidence / tau / run / step / os_force_miss and must equal the logged one (thr: conf >= tau; quantile: conf >=
    logged tau_t; force: os_force_miss == 1 -> MISS; cap: run >= R -> MISS; periodic: step % k == k-1 -> MISS; step0)

    taskset -c 34-37,78-81 .venv/bin/python -m exp.offline_search.closed_loop.verify_logs --log-dir <server log dir> \
        [--tag <tag>] [--work /tmp/osplug_verify] [--b0-check]
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pathlib  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

IR_PI05 = (0.152, 0.848)


def load_logs(log_dir: pathlib.Path, tag: str | None):
    pat = f"{tag}_c*.npz" if tag else "*.npz"
    eps = []
    for f in sorted(glob.glob(str(log_dir / "inputs" / pat))):
        z = np.load(f, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        eps.append((f, m, {k: z[k] for k in z.files if k != "meta"}))
    return eps


def hit_flags(z) -> np.ndarray:
    """int8 per logged decision: 1 HIT (served payload executed), 0 MISS (policy chunk executed). Pure-cache logs
    (no 'hit' array) are all HIT."""
    n = int(z["step"].shape[0])
    if "hit" in z:
        return np.asarray(z["hit"], np.int8)[:n]
    return np.ones(n, np.int8)


def build_ministore(work: pathlib.Path, root: pathlib.Path, cell: str, eps, H: int):
    from exp.offline_search.harness import store

    m, s, _ = store.parse_cell(cell)
    qd = work / "queries" / cell
    if work.exists():
        shutil.rmtree(work)
    qd.mkdir(parents=True)
    (work / "library").mkdir()
    os.symlink(root / "library" / f"{m}_{s}", work / "library" / f"{m}_{s}")
    episodes, cols = [], {k: [] for k in ("ep", "step", "key_v0", "key_v1", "rs", "raw_state", "a_exec", "a_hit", "a_inf")}
    start = 0
    for i, (_f, meta, z) in enumerate(eps):
        n = int(z["step"].shape[0])
        aex = np.full((n, H, 32), np.nan, np.float32)
        aex[: z["a_exec"].shape[0]] = z["a_exec"][:n]
        hit = hit_flags(z)
        # exec_hit_flag (store.QueryCell) compares a_exec with a_hit / a_inf on [:, :, :7]: HIT rows -> a_hit ==
        # a_exec, a_inf != ; MISS rows (policy chunk executed) -> a_inf == a_exec, a_hit != ; NaN rows stay undecidable
        other = np.nan_to_num(aex) + 1.0
        ahit = np.where((hit == 1)[:, None, None], aex, other)
        ainf = np.where((hit == 0)[:, None, None], aex, other)
        episodes.append({"uid": meta["uid"], "file": _f, "task": meta["task"], "task_id": int(meta["task_id"]),
                         "init": int(meta["init"]), "success": bool(meta.get("success") or False),
                         "num_steps": n, "start": start, "end": start + n})
        cols["ep"].append(np.full(n, i, np.int32))
        cols["step"].append(z["step"].astype(np.int16))
        for k in ("key_v0", "key_v1", "rs", "raw_state"):
            cols[k].append(z[k].astype(np.float32))
        cols["a_exec"].append(aex)
        cols["a_hit"].append(ahit.astype(np.float32))
        cols["a_inf"].append(ainf.astype(np.float32))
        start += n
    arr = {k: np.concatenate(v) for k, v in cols.items()}
    N = arr["ep"].shape[0]
    for k, v in arr.items():
        np.save(qd / f"{k}.npy", v)
    np.save(qd / "rec_top1.npy", np.zeros(N, np.int32))
    np.save(qd / "rec_score.npy", np.zeros(N, np.float32))
    np.save(qd / "rec_perfield.npy", np.zeros((N, 3), np.float32))
    (qd / "episodes.json").write_text(json.dumps(episodes))
    (qd / "manifest.json").write_text(json.dumps({"schema": "offline_search.queries.v1", "arm": cell,
                                                  "complete": True, "source": "closed_loop logs"}))
    return episodes


def run_offline(method_spec, kwargs, cell, work, seed, episodes):
    from exp.offline_search.harness import run, store

    cls, _ = run.load_method_class(method_spec)
    F = run._fit_cell(cls, kwargs, cell, root=work, out_dir=work / "out", seed=seed, profile=False)
    qc = store.QueryCell(work, cell)
    jobs = [(i, np.arange(e["start"], e["end"], dtype=np.int64)) for i, e in enumerate(episodes)]
    return run.run_jobs_inprocess(F["method"], qc, jobs, lib_sizes=F["lib_sizes"], seed=seed, cell=cell), F


def expected_verdict(J: dict, *, step: int, conf: float, tau: float, run: int, forced: bool, prev_reason: str | None,
                     burst_left: int, decision_index=None):
    """Re-derive (hit, reason, burst_left_after) from the logged inputs with the plugin's rule (plugin._verdict)."""
    mode = J["mode"]
    if mode == "periodic" and decision_index is not None and decision_index % int(J["k"]) == int(J["k"])-1:
        hit, why = False, "periodic"
    elif step == 0 and J.get("step0", "judge") != "judge":
        hit, why = J["step0"] == "hit", "step0"
    elif mode == "always":
        hit, why = True, "always"
    elif mode == "periodic":
        k = int(J["k"])
        hit, why = ((step if decision_index is None else decision_index) % k != k - 1), "periodic"
    elif forced:
        hit, why = False, "force"
    elif burst_left > 0:
        hit, why = False, "burst"
    elif int(J.get("cap", 0)) > 0 and run >= int(J["cap"]):
        hit, why = False, "cap"
    elif mode == "guard_only":
        hit, why = True, "guard_only"
    else:
        hit, why = bool(conf >= tau), ("thr" if mode == "threshold" else "quantile")
    if why == "burst":
        burst_left -= 1
    elif forced and not hit and int(J.get("burst", 1)) > 1 and mode not in ("always", "periodic"):
        burst_left = int(J["burst"]) - 1
    return hit, why, burst_left


def check_verdicts(J: dict, z: dict) -> dict:
    """Per-episode verdict audit of a mixed-mode npz: counts by reason and the rows whose logged verdict differs from
    the rule applied to the logged inputs. hist consistency: run == trailing HITs of the logged flags."""
    n = int(z["step"].shape[0])
    hit = hit_flags(z)
    judge = np.asarray(z["judge"]).astype(str)[:n]
    tau = np.asarray(z["tau"], np.float64)[:n]
    run = np.asarray(z["run"], np.int64)[:n]
    conf = np.asarray(z["conf"], np.float64)[:n]
    force = np.asarray(z["x_os_force_miss"], np.float64)[:n] if "x_os_force_miss" in z else np.zeros(n)
    out = {"n": n, "mix": {}, "bad": [], "run_bad": 0, "force_rows": int(np.sum(force == 1))}
    burst_left = 0
    for s in range(n):
        forced = bool(force[s] == 1)
        exp_hit, why, burst_left = expected_verdict(J, step=s, conf=float(conf[s]), tau=float(tau[s]), run=int(run[s]),
                                                    forced=forced, prev_reason=None, burst_left=burst_left,
                                                    decision_index=int(z["decision_index"][s]) if "has_vision" in z
                                                    and ("periodic_global" not in z or z["periodic_global"][s]) else None)
        if "has_vision" in z and not z["has_vision"][s]:
            exp_hit, why = True, "blind"
        logged_why = judge[s].split(":")[0]
        out["mix"][judge[s]] = out["mix"].get(judge[s], 0) + 1
        if bool(hit[s]) != exp_hit or logged_why != why:
            out["bad"].append({"step": s, "hit": int(hit[s]), "judge": judge[s], "expected": (exp_hit, why),
                               "conf": float(conf[s]), "tau": float(tau[s]), "run": int(run[s]), "forced": forced})
        if J["mode"] == "threshold" and math.isfinite(tau[s]) and tau[s] != float(J["tau"]):
            out["bad"].append({"step": s, "tau_logged": float(tau[s]), "tau_flag": float(J["tau"])})
        exp_run = 0
        for f in hit[:s][::-1]:
            if f != 1:
                break
            exp_run += 1
        if exp_run != run[s]:
            out["run_bad"] += 1
    return out


def verify_blind_logs(log_dir, eps, *, out=None):
    """Replay the method on dense histories, calling blind_step only before vision.

    This uses logged normalized states/actions as observations, independently builds
    the contract facades, and recomputes all selected/propagated action chunks.
    """
    from types import SimpleNamespace
    from exp.offline_search.closed_loop.blind import BlindQueryView, BlindResult, LookReason
    from exp.offline_search.harness import api, run, store

    configs = {(m["method_spec"], json.dumps(m["kwargs"], sort_keys=True), m["cell"], m["root"], m["run_seed"],
                json.dumps(m.get("judge"), sort_keys=True)) for _, m, _ in eps}
    if len(configs) != 1:
        raise ValueError("blind logs mix configurations; select one server tag")
    _, meta, _ = eps[0]
    root, cell = meta["root"], meta["cell"]
    cls, _ = run.load_method_class(meta["method_spec"])
    F = run._fit_cell(cls, meta["kwargs"], cell, root=root, out_dir=pathlib.Path(log_dir) / "blind_verify_fit",
                      seed=meta["run_seed"], profile=False)
    method = F["method"]
    lib = store.LibraryView(root, store.lib_key(cell), "current")
    tables = {"current": lib.action}
    for name in store.library_names(root, store.lib_key(cell)):
        tables[name] = store.LibraryView(root, store.lib_key(cell), name).action
    tables.update({k: v["action"] for k, v in F.get("registered", {}).items()})
    total = vision_n = blind_n = hit_n = 0
    checks = {k: 0 for k in ("topk", "scores", "conf", "lib", "action", "extras", "look_reason", "dense_history")}
    errors = []
    decs = {}
    for path in pathlib.Path(log_dir).glob("decisions_*.jsonl"):
        for line in path.read_text().splitlines():
            d = json.loads(line)
            if d.get("ev") == "dec":
                decs[(d["conn"], d["uid"], d["step"])] = d
    for _, m, z in eps:
        ep = api.EpisodeView(m["uid"], m["task"], m["task_id"], m["init"], m["index"], m["seed"])
        method.reset(ep)
        age, last_vision, burst_left = 0, -1, 0
        J = m.get("judge")
        if J:
            verdict = check_verdicts(J, z)
            errors.extend(verdict["bad"])
            if verdict["run_bad"]:
                errors.append({"run_bad": verdict["run_bad"]})
        hit = hit_flags(z)
        for s in range(len(z["step"])):
            total += 1
            v = bool(z["has_vision"][s])
            vision_n += v
            blind_n += not v
            hit_n += bool(hit[s])
            common = dict(step=s, task_id=ep.task_id, episode=ep, rs=z["rs"][s], raw_state=z["raw_state"][s],
                          prev_hit=None if s == 0 else bool(hit[s-1]),
                          prev_a_exec=None if s == 0 else z["a_exec"][s-1], hist_a_exec=z["a_exec"][:s],
                          hist_hit=hit[:s], hist_rs=z["rs"][:s], hist_has_vision=z["has_vision"][:s], blind_age=age)
            reason = None
            if J and J["mode"] == "periodic" and int(z["decision_index"][s]) % J["k"] == J["k"]-1:
                reason = LookReason(7, "periodic")
            elif s == 0 or hit[s-1] == 0:
                reason = LookReason(6, "lifecycle")
            elif J and (burst_left > 0 or (J.get("cap", 0) > 0 and J["mode"] not in ("always", "periodic")
                                          and int(z["run"][s]) >= J["cap"])):
                reason = LookReason(8, "judge requires vision")
            elif not hasattr(method, "blind_step"):
                reason = LookReason(8, "unsupported")
            else:
                reason = method.blind_step(BlindQueryView(**common))
            if v:
                if not isinstance(reason, LookReason):
                    errors.append({"uid": m["uid"], "step": s, "error": "offline chose blind on vision row"})
                elif reason.code == z["look_reason"][s]:
                    checks["look_reason"] += 1
                last_vision = s
                q = SimpleNamespace(**common, model=m["model"], key_v0=z["key_v0"][s], key_v1=z["key_v1"][s],
                                    hist_key_v0=z["key_v0"][:s], hist_key_v1=z["key_v1"][:s],
                                    hist_raw_state=z["raw_state"][:s], has_vision=True, has_tok=False,
                                    last_vision_step=last_vision)
                res = method.query(q)
                rows, scores, conf, lname, action, extras = api.validate_result(
                    res, lib_sizes=F["lib_sizes"], H=int(m["H"]), where=f"blind replay {m['uid']}:{s}")
                if action is None:
                    action = np.asarray(tables[lname][rows[0]], np.float32)
                checks["scores"] += np.array_equal(z["scores"][s, :min(len(scores), api.TOPK_SAVE)], scores[:api.TOPK_SAVE])
                checks["conf"] += z["conf"][s] == conf
                age = 0
            else:
                if not isinstance(reason, BlindResult):
                    errors.append({"uid": m["uid"], "step": s, "error": "offline requested vision on blind row"})
                    continue
                rows, lname, action, extras = reason.rows, reason.library, reason.action, reason.extras
                checks["look_reason"] += np.isnan(z["look_reason"][s])
                checks["scores"] += bool(np.all(z["scores"][s, :min(len(rows), api.TOPK_SAVE)] == 0))
                checks["conf"] += bool(np.isnan(z["conf"][s]))
                assert np.array_equal(z["blind_rows"][s, :len(rows)], rows)
                assert np.array_equal(z["blind_weights"][s, :len(rows)], reason.weights)
                age += 1
            checks["topk"] += np.array_equal(z["topk"][s, :min(len(rows), api.TOPK_SAVE)], rows[:api.TOPK_SAVE])
            checks["lib"] += str(z["lib"][s]) == lname
            checks["action"] += bool(not hit[s] or np.array_equal(z["a_exec"][s], action))
            ex_ok = all(f"x_{k}" in z and np.array_equal(np.asarray(z[f"x_{k}"][s]), np.asarray(v).reshape(-1)[0],
                                                       equal_nan=True)
                        for k, v in (extras or {}).items() if np.asarray(v).size == 1)
            checks["extras"] += ex_ok
            d = decs[(m["conn"], m["uid"], s)]
            dense_ok = (len(common["hist_rs"]) == s == len(common["hist_a_exec"]) == len(common["hist_has_vision"])
                        and (v or (hit[s] == 1 and np.isnan(z["key_v0"][s]).all() and np.isnan(z["key_v1"][s]).all()))
                        and d["vision"] == v and d["served_head"] == z["a_exec"][s, :5, :7].tolist()
                        and (v or (d["s1_ms"] is None and d["s23_ms"] is None and not d["shadow_available"]))
                        and d["blind_age"] == common["blind_age"])
            checks["dense_history"] += dense_ok
            if J:
                why = str(z["judge"][s])
                forced = (z.get("x_os_force_miss", np.zeros(len(hit)))[s] == 1)
                if why == "burst":
                    burst_left -= 1
                elif forced and not hit[s] and J.get("burst", 1) > 1 and J["mode"] not in ("always", "periodic"):
                    burst_left = J["burst"] - 1
    rep = dict(PASS=not errors and all(n == total for n in checks.values()), episodes=len(eps), decisions=total,
               vision=vision_n, blind=blind_n, hit=hit_n, miss=total-hit_n,
               equal={k: int(v) for k, v in checks.items()}, errors=errors[:10])
    path = pathlib.Path(out) if out else pathlib.Path(log_dir) / "verify_blind.json"
    path.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0 if rep["PASS"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--work", default="/tmp/osplug_verify")
    ap.add_argument("--b0-check", action="store_true", help="also run offline B0Current and compare with the native "
                    "winners (native mode) / native shadow (method mode)")
    ap.add_argument("--root", default="/dev/shm/offline_search_store", help="store root when the logs lack it")
    ap.add_argument("--out", default="", help="write the report JSON here (default <log-dir>/verify_<tag>.json)")
    a = ap.parse_args(argv)

    from exp.offline_search.harness import run

    log_dir = pathlib.Path(a.log_dir)
    eps = load_logs(log_dir, a.tag or None)
    if not eps:
        raise SystemExit(f"no inputs/*.npz under {log_dir}")
    if any(m.get("blind", False) for _, m, _ in eps):
        return verify_blind_logs(log_dir, eps, out=a.out or None)
    metas = {(m["method_spec"], json.dumps(m["kwargs"], sort_keys=True), m["cell"], m.get("run_seed", 0),
              m.get("root"), json.dumps(m.get("judge"), sort_keys=True)) for _, m, _ in eps}
    if len(metas) != 1:
        raise SystemExit(f"logs mix several configurations {metas}; pass --tag")
    spec, kw, cell, seed, root, judge_js = next(iter(metas))
    root = root or a.root
    kwargs = json.loads(kw)
    J = json.loads(judge_js)
    H = int(eps[0][1]["H"])
    native_mode = spec == "native"
    work = pathlib.Path(a.work) / (a.tag or "all")
    episodes = build_ministore(work, pathlib.Path(root), cell, eps, H)
    rep = {"log_dir": str(log_dir), "tag": a.tag, "method_spec": spec, "kwargs": kwargs, "cell": cell, "judge": J,
           "episodes": len(eps), "decisions": int(sum(e["num_steps"] for e in episodes))}

    # seeds as the harness derives them
    rep["episode_seed_equal"] = all(int(m["seed"]) == run.ep_seed(seed, m["uid"]) for _, m, _ in eps)
    from exp.offline_search.harness import store

    lib = store.LibraryView(root, store.lib_key(cell), "current")
    tables = {"current": np.asarray(lib.action, np.float32)}
    for ln in store.library_names(root, store.lib_key(cell)):
        if ln != "current":
            tables[ln] = store.LibraryView(root, store.lib_key(cell), ln).action

    # executed == selected on HIT rows (MISS rows executed the policy chunk: nothing to compare)
    ex_ok = ex_n = miss_rows = 0
    for _, m, z in eps:
        hit = hit_flags(z)
        n_ex = z["a_exec"].shape[0]
        for s in range(n_ex):
            if hit[s] != 1:
                miss_rows += 1
                continue
            sel = z["synth"][s] if z["used_synth"][s] else (
                np.asarray(tables[str(z["lib"][s])][int(z["top1"][s])], np.float32) if str(z["lib"][s]) in tables
                else None)
            if sel is None:
                continue
            ex_n += 1
            ex_ok += int(np.array_equal(z["a_exec"][s], sel))
    rep["executed_equals_selected"] = {"n": ex_n, "equal": ex_ok, "miss_rows_skipped": miss_rows}

    verdict_bad = 0
    if J is not None:
        hits = np.concatenate([hit_flags(z) for _, _, z in eps])
        n = int(hits.size)
        nm = int((hits == 0).sum())
        mix: dict = {}
        bad = []
        run_bad = 0
        for _, m, z in eps:
            c = check_verdicts(J, z)
            for k, v in c["mix"].items():
                mix[k] = mix.get(k, 0) + v
            bad += [{"uid": m["uid"], **b} for b in c["bad"]]
            run_bad += c["run_bad"]
        verdict_bad = len(bad) + run_bad
        s23 = np.concatenate([np.asarray(z["s23_ms"], np.float64) for _, _, z in eps])
        s23 = s23[np.isfinite(s23)]
        rep["mixed"] = {"n": n, "n_hit": n - nm, "n_miss": nm, "h": (n - nm) / n if n else None,
                        "ir_pi05_formula": IR_PI05[0] + IR_PI05[1] * (nm / n) if n else None,
                        "judge_mix": mix, "verdict_violations": len(bad), "run_violations": run_bad,
                        "first_violations": bad[:5],
                        "s23_ms": {"n": int(s23.size), "mean": float(s23.mean()) if s23.size else None,
                                   "p50": float(np.percentile(s23, 50)) if s23.size else None}}

    if not native_mode:
        off, F = run_offline(spec, kwargs, cell, work, seed, episodes)
        eq = dict(topk=0, scores=0, conf=0, lib=0, synth=0, extras=0)
        first = None
        j = 0
        for (_, m, z), e in zip(eps, episodes):
            for s in range(e["num_steps"]):
                k = off["topk"][j].size
                c = {"topk": np.array_equal(z["topk"][s, :k], off["topk"][j]),
                     "scores": np.array_equal(z["scores"][s, :k], off["scores"][j]),
                     "conf": bool(z["conf"][s] == off["conf"][j]),
                     "lib": str(z["lib"][s]) == off["lib"][j]}
                osyn = off["synth"][j]
                c["synth"] = (osyn is None and not z["used_synth"][s]) or (
                    osyn is not None and np.array_equal(np.nan_to_num(z["synth"][s, :5, :7]), np.nan_to_num(osyn)))
                exo = off["extras"][j] or {}
                bad = {kk: (float(z[f"x_{kk}"][s]) if f"x_{kk}" in z else None, float(np.asarray(v).reshape(-1)[0]))
                       for kk, v in exo.items() if np.asarray(v).size == 1 and
                       not (f"x_{kk}" in z and np.float64(z[f"x_{kk}"][s]) == np.float64(np.asarray(v).reshape(-1)[0]))}
                c["extras"] = not bad
                for kk, v in c.items():
                    eq[kk] += int(bool(v))
                if first is None and not all(c.values()):
                    first = {"uid": m["uid"], "step": s, **{kk: bool(v) for kk, v in c.items()}, "extras_diff": bad}
                j += 1
        n = rep["decisions"]
        rep["offline_equal"] = {kk: v / n for kk, v in eq.items()}
        rep["first_diff"] = first
        rep["fit_s_offline"] = round(F["fit_s"], 3)
    if a.b0_check or native_mode:
        off0, _ = run_offline("exp.offline_search.harness.baselines:B0Current", {}, cell, work, seed, episodes)
        native = np.concatenate([z["native_top1"] for _, _, z in eps])
        b0 = np.asarray(off0["top1"])
        valid = native >= 0
        rep["offline_b0_vs_native"] = {"n": int(valid.sum()), "agree": int((b0[valid] == native[valid]).sum())}
        if not native_mode:
            online = np.concatenate([z["top1"] for _, _, z in eps])
            libs = np.concatenate([z["lib"] for _, _, z in eps])
            rep["online_vs_offline_b0"] = {"n": int(online.size),
                                           "agree": int(((online == b0) & (libs == "current")).sum())}
    q = np.concatenate([z["q_us"] for _, _, z in eps])
    im = np.concatenate([z["infer_ms"] for _, _, z in eps])
    rep["q_us"] = {"p50": float(np.percentile(q, 50)), "p95": float(np.percentile(q, 95))}
    rep["infer_ms"] = {"p50": float(np.percentile(im, 50)), "p95": float(np.percentile(im, 95))}
    out = pathlib.Path(a.out) if a.out else log_dir / f"verify_{a.tag or 'all'}.json"
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    ok = rep["episode_seed_equal"] and ex_ok == ex_n and verdict_bad == 0
    if not native_mode:
        ok = ok and all(v == 1.0 for v in rep["offline_equal"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
