"""CPU tests for exp/step_diag/analysis/aggregate_arms.py on synthetic driver + server files: the
strict equal-NFE gate, the paired statistics, the discordant-pair flat interval at n=50 vs n=100,
the macro bootstrap's degenerate flag, the verdict branches and the end-to-end aggregate."""

import json
import hashlib
import pathlib

import numpy as np
import pytest

from exp.step_diag import envs as E
from exp.step_diag.analysis import aggregate_arms as AG

POLICY, TEACHER = "pi05", "pi05"
M_STAR, T_STAR = E.QB_MAIN_M[POLICY], E.QB_MAIN_T[POLICY]
ARMS = ("full", f"plain_k{M_STAR}", f"warm_t{T_STAR:g}")


def _write_arm(root: pathlib.Path, server_root: pathlib.Path, arm_id: str, kind: str, m: int, outcomes: dict,
               *, n_decisions=3, hit_type=None, steps=None, drop_server=(), drop_journal=(), error_uids=(),
               stamp=None, dup_journal=(), summary_decisions=None, drop_summary=()):
    """outcomes: {task: [success per init_idx]}; writes launch/journal + server rows for one arm."""
    d = root / TEACHER / arm_id
    d.mkdir(parents=True)
    s = server_root / TEACHER / arm_id
    s.mkdir(parents=True)
    expected, journal, rows, per_step = [], [], [], []
    hit = hit_type or ("MISS" if kind == "plain" else "WARM_START")
    manifest = E.RunManifest(experiment_id="e", env=E.ENVS["pi05_rc"].to_json(), arm_id=arm_id,
        mode="warm" if kind == "warm" else "plain", exec_steps=m, checkpoint="fixture", checkpoint_sha256="weights",
        cache_config="fixture", cache_config_sha256="cache", library="fixture", library_sha256="library",
        code_commit="fixture", extras={"h_exec": 5, "runtime": {"source_sha256": "source"}})
    cfg = manifest.write(s / f"manifest_{arm_id}.json")
    stamp = {"launch_id": "L0", "arm_id": arm_id, "experiment_id": "e", "config_sha": cfg} if stamp is None else stamp
    for task, succ in outcomes.items():
        for i, ok in enumerate(succ):
            uid = f"{task}:{i}"
            ident = {"task_uid": uid, "task": task, "init_idx": i, "env_seed": 2_000_000 + i,
                     "lane": E.lane_of(task), "pin_id": E.canonical_pin_id() if E.lane_of(task) == "pnp" else None, "layout": 1, "style": 1}
            expected.append(ident)
            if uid not in drop_journal:
                journal.append({"accepted": True, "run_id": "driver", "task_uid": uid, "status": "done", "success": bool(ok), "attempt": 1,
                                "error": ("boom" if uid in error_uids else None)})
                if uid in dup_journal:
                    journal.append(dict(journal[-1], run_id="second-launch"))
            if uid not in drop_summary:
                per_step.append({"worker_runtime": {"host": "worker", "source_sha256": "source"}, **ident, "success": bool(ok), "run_id": "driver", "task_uid": uid, "attempt": 1, "row": "episode_summary", "accepted": True,
                                 "n_decisions": (n_decisions if summary_decisions is None else summary_decisions),
                                 "n_env_steps": 5 * n_decisions, "reported_n_steps": 5 * n_decisions - 1})
            if uid not in drop_server:
                rel = f"arrays/{uid}.npz"
                (s / "arrays").mkdir(exist_ok=True)
                np.savez(s / rel, **{f"a_exec_{j:04d}": np.zeros((50, 32), dtype=np.float32) for j in range(n_decisions)})
                array_meta = {"arrays": rel, "arrays_sha256": hashlib.sha256((s / rel).read_bytes()).hexdigest(),
                              "env_id": "pi05_rc"}
                for j in range(n_decisions):
                    rows.append({**ident, **array_meta, "config_sha": cfg, "start_t": T_STAR if kind == "warm" else None, "schedule_id": "pi05_v1", "task_uid": uid, "attempt": 1, "decision_idx": j, "status": "ok", "hit_type": hit,
                                 "executed_steps": (steps if steps is not None else m), "n_stage3_calls": 1})
                rows.append({**ident, **array_meta, "config_sha": cfg, "outcome": bool(ok), "task_uid": uid, "attempt": 1, "status": "finalize", "terminal": True, "n_decisions": n_decisions,
                             "client_stamp": stamp, "stamp_mismatch": []})
    (d / "launch_0.json").write_text(json.dumps({"launch_id": "L0", "driver_run_id": "driver", "config_sha": cfg, "experiment_id": "e", "arm_id": arm_id, "expected": expected}))
    (d / "journal_0.jsonl").write_text("\n".join(json.dumps(r) for r in journal) + "\n")
    (s / "rows_x.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (d / "per_step_x.jsonl").write_text("\n".join(json.dumps(r) for r in per_step) + "\n")


def _outcomes(rng, tasks, n, p):
    return {t: list((rng.random(n) < p).astype(int)) for t in tasks}


def test_equal_nfe_gate(tmp_path):
    arms, srv = tmp_path / "arms", tmp_path / "srv"
    _write_arm(arms, srv, "plain_k2", "plain", 2, {"CloseFridge": [1, 0, 1]})
    arm = AG.load_arm(arms / TEACHER / "plain_k2")
    server = AG.load_server_rows(srv / TEACHER / "plain_k2")
    c = AG.cell_admission(arm, server, "CloseFridge", kind="plain", m=2)
    assert c["complete"] and c["equal_nfe"] and c["n_decisions"] == 9 and c["mean_executed_steps"] == 2.0
    assert c["env_steps_mean"] == 15.0 and c["decisions_per_episode_mean"] == 3.0  # Q-C.1 from the worker summaries
    # wrong step count on the server rows
    _write_arm(arms, srv, "plain_k2_bad", "plain", 2, {"CloseFridge": [1, 0, 1]}, steps=3)
    c2 = AG.cell_admission(AG.load_arm(arms / TEACHER / "plain_k2_bad"), AG.load_server_rows(srv / TEACHER / "plain_k2_bad"),
                           "CloseFridge", kind="plain", m=2)
    assert c2["complete"] and not c2["equal_nfe"] and c2["problems"]["steps_mismatch"] == 9
    # warm arm that fell back to MISS on every decision
    _write_arm(arms, srv, "warm_t0.2", "warm", 2, {"CloseFridge": [1, 0, 1]}, hit_type="MISS")
    c3 = AG.cell_admission(AG.load_arm(arms / TEACHER / "warm_t0.2"), AG.load_server_rows(srv / TEACHER / "warm_t0.2"),
                           "CloseFridge", kind="warm", m=2)
    assert not c3["equal_nfe"] and c3["miss_fraction"] == 1.0 and c3["problems"]["hit_type_mismatch"] == 9
    # missing server evidence / missing journal record / terminal error
    _write_arm(arms, srv, "p_missing", "plain", 2, {"CloseFridge": [1, 0, 1]}, drop_server=("CloseFridge:1",),
               drop_journal=("CloseFridge:2",), error_uids=("CloseFridge:0",))
    c4 = AG.cell_admission(AG.load_arm(arms / TEACHER / "p_missing"), AG.load_server_rows(srv / TEACHER / "p_missing"),
                           "CloseFridge", kind="plain", m=2)
    assert not c4["complete"] and c4["problems"] == {"terminal_error": 1, "server_evidence_missing": 1, "missing_terminal": 1}
    assert c4["n_outcomes"] == 2  # outcomes retained, errors excluded by the gate
    # rows stamped by another launch / arm are evidence for a different cell
    _write_arm(arms, srv, "p_stamp", "plain", 2, {"CloseFridge": [1, 0, 1]},
               stamp={"launch_id": "L-other", "arm_id": "p_stamp", "experiment_id": "e", "config_sha": "c"})
    c5 = AG.cell_admission(AG.load_arm(arms / TEACHER / "p_stamp"), AG.load_server_rows(srv / TEACHER / "p_stamp"),
                           "CloseFridge", kind="plain", m=2)
    assert c5["complete"] and not c5["equal_nfe"] and c5["problems"]["arm_stamp_mismatch"] == 3
    _write_arm(arms, srv, "p_arm", "plain", 2, {"CloseFridge": [1, 0, 1]},
               stamp={"launch_id": "L0", "arm_id": "warm_t0.2", "experiment_id": "e", "config_sha": "c"})
    c6 = AG.cell_admission(AG.load_arm(arms / TEACHER / "p_arm"), AG.load_server_rows(srv / TEACHER / "p_arm"),
                           "CloseFridge", kind="plain", m=2)
    assert c6["problems"]["arm_stamp_mismatch"] == 3
    # a second accepted terminal record for one identity is a conflict, not "latest wins"
    _write_arm(arms, srv, "p_dup", "plain", 2, {"CloseFridge": [1, 0, 1]}, dup_journal=("CloseFridge:0",))
    c7 = AG.cell_admission(AG.load_arm(arms / TEACHER / "p_dup"), AG.load_server_rows(srv / TEACHER / "p_dup"),
                           "CloseFridge", kind="plain", m=2)
    assert not c7["complete"] and c7["problems"] == {"duplicate_accepted_terminal": 1} and c7["n_outcomes"] == 3
    # worker summary disagreeing with the server's decision count, or missing
    _write_arm(arms, srv, "p_cnt", "plain", 2, {"CloseFridge": [1, 0, 1]}, summary_decisions=4, drop_summary=("CloseFridge:2",))
    c8 = AG.cell_admission(AG.load_arm(arms / TEACHER / "p_cnt"), AG.load_server_rows(srv / TEACHER / "p_cnt"),
                           "CloseFridge", kind="plain", m=2)
    assert c8["complete"] and not c8["equal_nfe"]
    assert c8["problems"] == {"decision_count_mismatch": 2, "worker_summary_missing": 1}


def test_paired_stats_and_flat_interval_power():
    triples = [(1, 0, 1), (1, 0, 0), (1, 1, 1), (0, 0, 0)]
    st = AG.task_stats(triples)
    assert st["n"] == 4 and st["g"] == pytest.approx(0.5) and st["delta"] == pytest.approx(0.25)
    assert st["H50"] == pytest.approx(0.0) and st["H25"] == pytest.approx(0.125) and st["recovery"] == pytest.approx(0.5)
    assert AG.task_stats([])["g"] is None
    # identical arms: the discordant-pair interval at n=50 is too wide for the -0.10 margin, n=100 is not
    same50 = [(1, 1)] * 45 + [(0, 0)] * 5
    same100 = same50 * 2
    i50, i100 = AG.flat_delta_interval(same50, AG.ALPHA_FAMILY), AG.flat_delta_interval(same100, AG.ALPHA_FAMILY)
    assert i50["delta"] == 0.0 and i50["n10"] == 0 and i50["n01"] == 0
    assert i50["lower"] < -0.10 < i100["lower"]
    assert i100["upper"] == -i100["lower"]
    lo, hi = AG.clopper_pearson(0, 10, 0.95)
    assert lo == 0.0 and 0 < hi < 1
    assert AG.clopper_pearson(10, 10, 0.95)[1] == 1.0


def test_macro_bootstrap_shared_indices_and_degenerate_flag():
    rng = np.random.default_rng(0)
    cliff = {t: [tuple(int(v) for v in rng.integers(0, 2, size=3)) for _ in range(40)] for t in ("A", "B", "C")}
    m = AG.macro_bootstrap(cliff, boot=300, alpha=AG.ALPHA_FAMILY)
    assert m["n_tasks"] == 3 and m["g"]["lower"] <= m["g"]["point"] <= m["g"]["upper"]
    assert not m["delta"]["degenerate"]
    # deterministic outcomes -> zero-width bootstrap -> degenerate
    const = {t: [(1, 0, 1)] * 20 for t in ("A", "B", "C")}
    m2 = AG.macro_bootstrap(const, boot=50, alpha=AG.ALPHA_FAMILY)
    assert m2["g"]["degenerate"] and m2["g"]["point"] == 1.0
    assert AG.macro_bootstrap({}, boot=10, alpha=0.05) == {"n_tasks": 0}


def _macro(g, d, lo_shift=0.05):
    q = lambda p: {"point": p, "lower": p - lo_shift, "upper": p + lo_shift, "degenerate": False}  # noqa: E731
    return {"n_tasks": 4, "g": q(g), "delta": q(d), "H50": q(d - 0.5 * g), "H25": q(d - 0.25 * g)}


def test_verdict_branches():
    flat_ok = {"S": {"lower": -0.05, "upper": 0.05}}
    assert AG.verdict(_macro(0.4, 0.35), flat_ok, True)["verdict"] == "supported"
    assert AG.verdict(_macro(0.4, 0.02), flat_ok, True)["verdict"] == "not_supported"  # H25 upper = 0.02-0.1+0.05 < 0
    assert AG.verdict(_macro(0.4, 0.2), flat_ok, True)["verdict"] == "inconclusive"  # H50 lower = 0 - 0.05 < 0
    assert AG.verdict(_macro(0.4, 0.35), flat_ok, False)["verdict"] == "inconclusive"  # gate failed
    harmful = {"S": {"lower": -0.3, "upper": -0.15}}
    v = AG.verdict(_macro(0.4, 0.35), harmful, True)
    assert v["verdict"] == "inconclusive" and v["harmful_on_flat"]
    deg = _macro(0.4, 0.35)
    deg["g"]["degenerate"] = True
    assert AG.verdict(deg, flat_ok, True)["verdict"] == "inconclusive"
    few = _macro(0.4, 0.35)
    few["n_tasks"] = 2
    assert AG.verdict(few, flat_ok, True)["verdict"] == "inconclusive"


def test_aggregate_end_to_end_and_missing_arm(tmp_path):
    rng = np.random.default_rng(1)
    arms, srv = tmp_path / "arms", tmp_path / "srv"
    tasks = E.qb_tasks(POLICY)
    full = {t: [1] * E.qb_episode_count(POLICY, t, ARMS[0]) for t in tasks}
    plain = {t: list((rng.random(E.qb_episode_count(POLICY, t, ARMS[1])) < (0.3 if t in E.QB_CLIFF[POLICY] else 0.95)).astype(int)) for t in tasks}
    warm = {t: [1] * E.qb_episode_count(POLICY, t, ARMS[2]) for t in tasks}
    _write_arm(arms, srv, ARMS[0], "plain", 10, full)
    _write_arm(arms, srv, ARMS[1], "plain", M_STAR, plain)
    _write_arm(arms, srv, ARMS[2], "warm", M_STAR, warm)
    res = AG.aggregate(POLICY, arms, srv, boot=200)
    assert res["missing_arms"] == [] and set(res["per_task"]) == set(tasks)
    assert all(res["per_task"][t]["status"] == "ok" for t in tasks)
    assert res["cliff_macro"]["n_tasks"] == len(E.QB_CLIFF[POLICY]) and res["cliff_macro"]["g"]["point"] > 0.5
    assert set(res["flat"]) == set(E.QB_FLAT[POLICY])
    assert res["verdict"]["verdict"] in ("supported", "inconclusive")
    md = AG.markdown(res)
    assert "Q-B pi05" in md and "cliff macro" in md
    # a warm arm without server rows -> not admissible -> inconclusive with the cells reported
    import shutil
    shutil.rmtree(srv / TEACHER / ARMS[2])
    res2 = AG.aggregate(POLICY, arms, srv, boot=50)
    assert all(res2["per_task"][t]["status"] == "not_admissible" for t in tasks)
    assert res2["verdict"]["verdict"] == "inconclusive"
    shutil.rmtree(arms / TEACHER / ARMS[1])
    res3 = AG.aggregate(POLICY, arms, srv, boot=50)
    assert res3["missing_arms"] == [ARMS[1]] and res3["per_task"][tasks[0]]["status"] == "arm_missing"
    assert "arm missing" in AG.markdown(res3)


def test_similarity_bins_split_at_median_with_ties_low_and_min_bin(tmp_path):
    outcomes_p = {"CloseFridge": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1], "OpenCabinet": [0] * 10}
    outcomes_w = {"CloseFridge": [1, 1, 1, 0, 1, 1, 1, 1, 1, 1], "OpenCabinet": [1] * 10}
    arms, srv = tmp_path / "arms", tmp_path / "srv"
    _write_arm(arms, srv, "plain_k2", "plain", 2, outcomes_p)
    _write_arm(arms, srv, "warm_t0.2", "warm", 2, outcomes_w)
    cells = {}
    for task in outcomes_p:
        cells[task] = {aid: AG.cell_admission(AG.load_arm(arms / TEACHER / aid), AG.load_server_rows(srv / TEACHER / aid), task,
                                              kind=kind, m=2) for aid, kind in (("plain_k2", "plain"), ("warm_t0.2", "warm"))}
    # scores: CloseFridge idx 0..3 low (0.1), 4..9 high (0.5..0.9, with a tie at the median);
    # OpenCabinet scored on 5 identities only (< 2 x MIN_BIN); idx >= 10 is ignored
    shadow_eps = [{"task": "CloseFridge", "init_idx": i, "top1_score_median": 0.1 if i < 4 else 0.5 + 0.1 * (i - 4) if i < 8 else 0.9}
                  for i in range(10)]
    shadow_eps += [{"task": "OpenCabinet", "init_idx": i, "top1_score_median": 0.3} for i in range(5)]
    shadow_eps += [{"task": "CloseFridge", "init_idx": 12, "top1_score_median": 0.99}]
    for ep in shadow_eps:
        ep.update(env_seed=2_000_000 + ep['init_idx'], lane=E.lane_of(ep['task']), pin_id=None, layout=1, style=1,
                  env_id="pi05_rc",  # analyze_shadow episodes name their environment (part of the pairing identity)
                  comparison_identity=cells[ep['task']]['plain_k2']['comparison_identities'][0])
    bins = AG.similarity_bins(shadow_eps, cells, "plain_k2", "warm_t0.2")
    cf = bins["CloseFridge"]
    assert cf["status"] == "ok" and cf["n_joined"] == 10 and cf["n_scored"] == 10
    assert cf["low"]["n"] + cf["high"]["n"] == 10 and cf["low"]["n"] >= 5  # ties at the cut go low
    assert cf["low"]["delta"] == pytest.approx(3 / cf["low"]["n"]) and cf["high"]["delta"] == 0.0
    assert bins["OpenCabinet"]["status"] == "too_few"
    # end to end through aggregate(): a shadow json is optional
    (tmp_path / "shadow.json").write_text(json.dumps({"env_id": "pi05_rc", "episodes": shadow_eps}))
    for aid, kind, outs in (("full", "plain", {t: [1] * 10 for t in outcomes_p}),):
        _write_arm(arms, srv, aid, kind, 10, outs)
    res = AG.aggregate(POLICY, arms, srv, boot=20, shadow_json=tmp_path / "shadow.json")
    assert res["similarity_bins"]["CloseFridge"]["status"] == "ok" and "similarity bins" in AG.markdown(res)
    assert AG.aggregate(POLICY, arms, srv, boot=20)["similarity_bins"] is None


def test_resumed_attempt_collision_selects_the_accepted_launch_session(tmp_path):
    """A driver crash leaves the partial session of attempt 1 on the server; the resumed driver
    re-dispatches the identity as attempt 1 of a new launch. The accepted terminal's launch picks
    the session; the stale one is a stray note, and two sessions of the same launch stay a duplicate."""
    arms, srv = tmp_path / "arms", tmp_path / "srv"
    _write_arm(arms, srv, "plain_k2", "plain", 2, {"CloseFridge": [1, 0]})
    rows_path = srv / TEACHER / "plain_k2" / "rows_x.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    uid = "CloseFridge:0"
    for r in rows:
        r["session_id"] = 1
    stale = [dict(r, session_id=7) for r in rows if r["task_uid"] == uid]
    for r in stale:
        if r["status"] == "finalize":
            r["client_stamp"] = dict(r["client_stamp"], launch_id="Lcrashed")
            r["terminal"] = False
    stale = stale[:1] + [r for r in stale if r["status"] == "finalize"]  # one decision row + finalize
    rows_path.write_text("\n".join(json.dumps(r) for r in rows + stale) + "\n")
    arm = AG.load_arm(arms / TEACHER / "plain_k2")
    server = AG.load_server_rows(srv / TEACHER / "plain_k2")
    assert len(server[(uid, 1)]["sessions"]) == 2
    cell = AG.cell_admission(arm, server, "CloseFridge", kind="plain", m=2)
    assert cell["equal_nfe"] and cell["problems"] == {} and cell["stray_sessions"] == 1
    # a single-connection server has ONE session id per process: the stale occurrence and the
    # accepted one share it, and only the finalize row separates them (file order)
    same = [dict(r, session_id=1) for r in stale]
    rows_path.write_text("\n".join(json.dumps(r) for r in same + rows) + "\n")
    cell = AG.cell_admission(arm, AG.load_server_rows(srv / TEACHER / "plain_k2"), "CloseFridge", kind="plain", m=2)
    assert cell["equal_nfe"] and cell["problems"] == {} and cell["stray_sessions"] == 1
    # the same launch twice on one identity is still a duplicate, never "take the first"
    twin = [dict(r, session_id=9) for r in rows if r["task_uid"] == uid]
    rows_path.write_text("\n".join(json.dumps(r) for r in rows + twin) + "\n")
    cell = AG.cell_admission(arm, AG.load_server_rows(srv / TEACHER / "plain_k2"), "CloseFridge", kind="plain", m=2)
    assert not cell["equal_nfe"] and cell["problems"].get("duplicate_finalize") == 1
