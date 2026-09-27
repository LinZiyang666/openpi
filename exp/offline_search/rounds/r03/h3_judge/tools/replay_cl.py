"""Table (b): MixedJudge's judge signals replayed as SHADOW triggers on the real closed-loop decision logs of the R2
pure-cache arms (r02_g50: oscl50_{p,g}_{sp,l10}_cl2 = AWM kr5, cl3 = V6 StuckRecovery over AWM), joined with the
episode outcome. Protocol of ideation C: thresholds on inits 0-24 per task, everything measured on inits 25-49.
A spell = run of >= 3 identical top-1 picks; "early" = the first flag is no later than (first spell start + 1).

What the logs carry / lack (exact vs approximate features):
  exact   top-10 rows + scores (-> kernel weights over 10 of AWM's 16 members), top-1 -> library step / progress /
          terminal / episode; overtime; lag (top-5); disp (pairwise RMS of the top-5 heads = the wrapper's feature);
          no-progress guard (top-1 progress, library steps); cl3: stuck_n / motion / vself (V6 extras)
  approx  gripper vote and the served gripper sign (10 of 16 members); cl2: stuck_n via AWM's `still` extra
          (centred-cosine sum > 1.98, ideation C's stillness proxy) because rs / keys are not logged; the V7 confidence:
          `vis` is never logged (-> calibration mean, z contribution 0), `dnn` = dst * s_d(task) is exact in cl2 (AWM's
          dst extra) and missing in cl3 (-> mean); stuck from the proxy in cl2. So the V7 rows are approximate; the
          guard / event rows are exact except where marked.
The wrapper is fitted on the arm's store cell (same library) to take med_len, disp_thr, the calibration maps, s_d.

    taskset -c <cpus> .venv/bin/python exp/offline_search/rounds/r03/h3_judge/tools/replay_cl.py [arm ...]
Writes <derived>/r03/h3_judge/replay/<arm>.json and replay_cl.md (markdown tables) next to them; prints the tables.
"""
from __future__ import annotations

import glob
import json
import math
import os
import pathlib
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))
from exp.offline_search.harness import run, store  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parents[1]
JUDGE = str(HERE / "judge.py") + ":MixedJudge"
RUNS = pathlib.Path("/home/weiland/trace_runs/os_closed_loop/r02_g50/runs")
ROOT = "/dev/shm/offline_search_store"
OUT = pathlib.Path("/home/weiland/trace_runs/offline_search_store/derived/r03/h3_judge/replay")
SUITE = {"sp": "spatial", "l10": "l10"}
MODEL = {"p": "pi05", "g": "groot"}
KREF = 5
STILL_THR = 1.98            # ideation C's stillness proxy on AWM's `still` extra (sum of two centred cosines)
TRAIN_MAX_INIT = 24
NOMINAL = (0.1, 0.3, 0.5)   # nominal MISS rates for the confidence gates (h = .9 / .7 / .5)
REASON_NAMES = {1: "stuck", 2: "terminal", 3: "overtime", 4: "noprog", 5: "disp", 6: "grip", 7: "burst", 8: "conf", 9: "noprog_se"}


def load_arm(arm):
    """-> list of episodes (dicts with per-decision arrays), sorted by uid; accepted attempts only, joined to success."""
    d = RUNS / arm
    eps_rows = {}
    decs = {}
    for f in sorted(glob.glob(str(d / "server_*" / "decisions_*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                ev = r.get("ev")
                if ev == "episode":
                    eps_rows[(r["uid"], r["attempt"])] = r
                elif ev == "dec":
                    decs.setdefault((r["uid"], r["attempt"]), {})[int(r["step"])] = r
    journal = {}
    jp = d / "client" / "journal.jsonl"
    if jp.exists():
        for line in open(jp):
            r = json.loads(line)
            if r.get("accepted") and r.get("status") in ("done", "failed"):
                journal[r["task_uid"]] = r
    out = []
    for key, er in sorted(eps_rows.items()):
        if key not in decs or er.get("success") is None:
            continue
        uid = key[0]
        if journal and uid in journal and journal[uid].get("attempt", 1) != key[1]:
            continue
        bystep = decs[key]
        rr = [bystep[t] for t in sorted(bystep)]
        if [r["step"] for r in rr] != list(range(len(rr))):
            continue
        succ = bool(er["success"]) if uid not in journal else bool(journal[uid]["success"])
        out.append({"uid": uid, "task": int(rr[0]["task_id"]), "init": int(rr[0]["init"]), "success": succ, "rows": rr})
    return out


def kernel_w_awm(scores, kref=KREF):
    dt = -np.asarray(scores, np.float64)
    rel = dt - dt[0]
    ref = max(float(rel[min(kref, rel.size) - 1]), 1e-6)
    return np.exp(-(rel / ref) ** 2)


def features(eps, L, mj, is_cl3):
    """Per-decision feature arrays for one arm (concatenated over episodes; 'ep' indexes eps)."""
    HD, ep_of, step_of, eplen_of, prog_of, nxt_of = L
    C = mj.C
    cal = mj.cal
    F = {k: [] for k in ("ep", "step", "task", "init", "success", "top1", "conf_awm", "disp", "lag", "overtime", "term1",
                         "gexec", "gprop", "vote", "prog", "stuck_n", "noprog_n", "noprog_se", "conf_v7", "pred_v7",
                         "spell", "first3", "n", "same_run")}
    for ei, e in enumerate(eps):
        rr = e["rows"]
        n = len(rr)
        t = e["task"]
        med = mj.med_len.get(t, float("nan"))
        s_d = mj.base.tasks[t].s_d if hasattr(mj.base, "tasks") else float("nan")
        prev_g4 = 0.0
        prog_hist = []
        se_hist = []
        still_run = 0
        tops = np.asarray([r["top1"] for r in rr], np.int64)
        # spells (runs of identical top-1)
        bounds = np.r_[0, np.flatnonzero(np.diff(tops) != 0) + 1, n]
        spell = np.zeros(n, bool)
        first3 = n
        same_run = np.zeros(n, np.int64)
        for a, b in zip(bounds[:-1], bounds[1:]):
            same_run[a:b] = np.arange(1, b - a + 1)
            if b - a >= 3:
                spell[a:b] = True
                first3 = min(first3, int(a))
        for i, r in enumerate(rr):
            rows = np.asarray(r["topk"], np.int64)
            sc = np.asarray(r["scores"], np.float64)
            w = np.exp(sc - sc[0]) if is_cl3 else kernel_w_awm(sc)
            wn = w / w.sum()
            H = wn @ HD[rows]                                       # served head approx (10 of 16 members)
            top1 = int(rows[0])
            ex = r.get("extras") or {}
            disp = 0.0
            if rows.size >= 2:
                X = HD[rows[:5]]
                iu = np.triu_indices(X.shape[0], 1)
                Dm = X[iu[0]] - X[iu[1]]
                disp = float(np.sqrt(np.einsum("ij,ij->i", Dm, Dm) / X.shape[1]).mean())
            lag = i - float(step_of[rows[:5]].mean())
            overtime = i / med
            gsign = np.where(HD[rows, 6] >= 0, 1.0, -1.0)
            vote = float(wn @ gsign)
            gprop = 1.0 if H[6] >= 0 else -1.0
            gexec = prev_g4 if i > 0 else 0.0
            prog = float(prog_of[top1])
            prog_hist.append((prog, float(max(int(eplen_of[top1]) - 1, 1))))
            noprog = 0
            for j in range(i, 0, -1):
                (a, na), (b, _) = prog_hist[j], prog_hist[j - 1]
                if (a - b) * na > mj.prog_eps:
                    break
                noprog += 1
            # alternative: consecutive picks in the SAME library episode whose library step does not advance
            se_hist.append((int(ep_of[top1]), int(step_of[top1])))
            noprog_se = 0
            for j in range(i, 0, -1):
                (ea, sa), (eb, sb) = se_hist[j], se_hist[j - 1]
                if ea != eb or sa > sb:
                    break
                noprog_se += 1
            if is_cl3:
                stuck_n = int(ex.get("stuck_n", 0))
            else:
                still = float(ex.get("still", -9.0)) if i > 0 else -9.0
                still_run = still_run + 1 if still > STILL_THR else 0
                stuck_n = still_run
            # V7 confidence from the logged features (approximate: vis missing; dnn from dst in cl2 only)
            reg = 0 if i == 0 else 2
            c = cal.get(reg) or cal[2]
            names = ["dnn", "disp", "overtime", "vis", "stuck", "abslag"]
            dst = ex.get("dst")
            fv = {"dnn": (float(dst) * s_d) if (dst is not None and math.isfinite(s_d)) else float("nan"),
                  "disp": disp, "overtime": overtime, "vis": float("nan"), "stuck": float(min(stuck_n, 5)),
                  "abslag": abs(lag)}
            x = np.asarray([fv[k] for k in names], np.float64)
            x = np.where(np.isfinite(x), x, c["mu"])
            z = float(((x - c["mu"]) / c["sd"]) @ c["w"])
            pred = float(np.interp(z, c["kx"], c["ky"]))
            conf_v7 = -pred + 1e-6 * z
            for k, v in (("ep", ei), ("step", i), ("task", t), ("init", e["init"]), ("success", e["success"]),
                         ("top1", top1), ("conf_awm", float(r["conf"])), ("disp", disp), ("lag", lag),
                         ("overtime", overtime), ("term1", float(nxt_of[top1] < 0)), ("gexec", gexec), ("gprop", gprop),
                         ("vote", vote), ("prog", prog), ("stuck_n", stuck_n), ("noprog_n", noprog), ("noprog_se", noprog_se),
                         ("conf_v7", conf_v7), ("pred_v7", pred), ("spell", bool(spell[i])), ("first3", first3),
                         ("n", n), ("same_run", int(same_run[i]))):
                F[k].append(v)
            prev_g4 = 1.0 if H[4 * 7 + 6] >= 0 else -1.0
    return {k: np.asarray(v) for k, v in F.items()}


def flags_of(F, mj, which, tau_v7=None):
    """Shadow flags of one variant. which: set of reason codes {1..6} (+ 'v7' with tau_v7); burst adds reason 7."""
    n = F["step"].size
    fl = np.zeros(n, np.int64)
    if 1 in which:
        fl |= (F["stuck_n"] >= mj.stuck_thr) << 0
    if 2 in which:
        fl |= ((F["term1"] > 0) & (F["gexec"] > 0)) << 1
    if 3 in which:
        fl |= ((F["overtime"] > 1) & (F["lag"] > mj.lag_thr) & (F["stuck_n"] >= 1)) << 2
    if 4 in which:
        fl |= (F["noprog_n"] >= mj.noprog_n - 1) << 3
    if 5 in which:
        fl |= (F["disp"] >= mj.disp_thr) << 4
    if 6 in which:
        fl |= ((F["gexec"] != 0) & (F["gprop"] != F["gexec"]) & (np.abs(F["vote"]) < mj.vote_thr)) << 5
    if 9 in which:
        fl |= (F["noprog_se"] >= mj.noprog_n - 1) << 8
    force = fl != 0
    low = fl & -fl
    reason = np.zeros(fl.shape, np.int64)
    for r in range(1, 10):
        reason[low == (1 << (r - 1))] = r
    if tau_v7 is not None:
        low = F["conf_v7"] < tau_v7
        reason = np.where(~force & low, 8, reason)        # 8 = confidence below tau (not a forced reason code)
        force = force | low
    # burst continuation (only after a guard / event, not after a threshold MISS)
    if mj.burst > 1 and which:
        ep = F["ep"]
        burst_end = -1
        cur_ep = -1
        for i in range(n):
            if ep[i] != cur_ep:
                cur_ep, burst_end = ep[i], -1
            if reason[i] in (1, 2, 3, 4, 5, 6, 9):
                burst_end = F["step"][i] + mj.burst
            elif F["step"][i] < burst_end and not force[i]:
                force[i] = True
                reason[i] = 7
    return force, reason


def summarize(F, force, reason, test):
    """Per-variant metrics on the test episodes (mask over decisions)."""
    ep = F["ep"]
    succ = F["success"]
    out = {"flag_rate": float(force[test].mean()), "flag_rate_fail": float(force[test & ~succ].mean()) if (test & ~succ).any() else None,
           "flag_rate_succ": float(force[test & succ].mean()),
           "flags_in_fail_share": float(force[test & ~succ].sum() / max(force[test].sum(), 1)),
           "dec_in_fail_share": float((test & ~succ).sum() / max(test.sum(), 1))}
    rc = {}
    for r in range(1, 10):
        m = test & (reason == r)
        if m.any():
            rc[REASON_NAMES.get(r, "conf")] = {"rate": float(m.sum() / test.sum()), "in_fail": float((~succ[m]).mean())}
    out["reasons"] = rc
    res = {True: [], False: []}
    for e in np.unique(ep[test]):
        m = ep == e
        f = np.flatnonzero(force[m])
        first = int(f[0]) if f.size else None
        first3 = int(F["first3"][m][0])
        n = int(F["n"][m][0])
        s = bool(succ[m][0])
        res[s].append((first, first3, n))
    for s, lab in ((False, "fail"), (True, "succ")):
        L = res[s]
        touched = [x for x in L if x[0] is not None]
        out[f"n_{lab}"] = len(L)
        out[f"{lab}_touched"] = len(touched) / max(len(L), 1)
        if touched:
            out[f"{lab}_first_flag_pos_median"] = float(np.median([x[0] / max(x[2] - 1, 1) for x in touched]))
        if not s:
            early = [x for x in touched if x[0] <= x[1] + 1]
            with_spell = [x for x in L if x[1] < x[2]]
            lead = [x[1] - x[0] for x in touched if x[1] < x[2]]
            out["fail_touched_early"] = len(early) / max(len(L), 1)
            out["fail_with_spell"] = len(with_spell) / max(len(L), 1)
            out["fail_lead_median"] = float(np.median(lead)) if lead else None
            out["fail_lead_ge0"] = float(np.mean([x >= 0 for x in lead])) if lead else None
            out["fail_first_flag_pos_median"] = float(np.median([x[0] / max(x[2] - 1, 1) for x in touched])) if touched else None
    return out


def analyse(arm, mj_cache):
    parts = arm.split("_")
    model, suite = MODEL[parts[1]], SUITE[parts[2]]
    is_cl3 = parts[3] == "cl3"
    cell = f"{model}_{suite}_cache"
    if cell not in mj_cache:
        cls, _ = run.load_method_class(JUDGE)
        Fc = run._fit_cell(cls, {"guards": True, "events": "all", "burst": 2}, cell, root=ROOT,
                           out_dir=OUT / "fit" / cell, seed=0, profile=False)
        mj_cache[cell] = Fc["method"]
    mj = mj_cache[cell]
    lib = store.LibraryView(ROOT, f"{model}_{suite}", "current")
    sig = np.asarray(store.action_sigma(ROOT, f"{model}_{suite}"), np.float64)
    HD = (np.asarray(lib.action[:, :5, :7], np.float64) / sig).reshape(lib.L, 35)
    L = (HD, np.asarray(lib.episode), np.asarray(lib.step), np.asarray(lib.ep_len), np.asarray(lib.progress, np.float64),
         np.asarray(lib.next))
    eps = load_arm(arm)
    F = features(eps, L, mj, is_cl3)
    train = F["init"] <= TRAIN_MAX_INIT
    test = ~train
    rep = {"arm": arm, "cell": cell, "episodes": len(eps), "decisions": int(F["step"].size),
           "sr": float(np.mean([e["success"] for e in eps])), "test_episodes": int(len(np.unique(F["ep"][test]))),
           "test_fail": int(len(np.unique(F["ep"][test & ~F["success"]]))),
           "exact": {"disp": True, "lag": True, "overtime": True, "terminal": True, "noprog": True,
                     "stuck_n": is_cl3, "vote/gexec": "10 of 16 members", "v7_conf": "approx (vis missing" +
                     (", dnn missing" if is_cl3 else "") + (", stuck proxy" if not is_cl3 else "") + ")",
                     "cl3_weights": "exp(S_i - S_0), blend override ignored" if is_cl3 else None},
           "fit": {"disp_thr": mj.disp_thr, "med_len": mj.med_len, "prog_eps": mj.prog_eps},
           "variants": {}}
    taus = {r: float(np.quantile(F["conf_v7"][train], r)) for r in NOMINAL}
    taus_awm = {r: float(np.quantile(F["conf_awm"][train], r)) for r in NOMINAL}
    rep["tau_v7_train"] = taus
    rep["tau_awm_train"] = taus_awm
    V = {}
    for r in NOMINAL:
        V[f"awm_conf@{r}"] = (set(), None, F["conf_awm"] < taus_awm[r])
        V[f"v7@{r}"] = (set(), taus[r], None)
    V["guards"] = ({1, 2, 3, 4}, None, None)
    V["guard_stuck"] = ({1}, None, None)
    V["guard_terminal"] = ({2}, None, None)
    V["guard_overtime"] = ({3}, None, None)
    V["guard_noprog"] = ({4}, None, None)
    V["guard_noprog_sameep"] = ({9}, None, None)
    for nn in (4, 5, 6):
        raw_np = F["noprog_n"] >= nn - 1
        V[f"guard_noprog_n{nn}"] = (set(), None, raw_np)
        V[f"guards_np{nn}"] = ({1, 2, 3}, None, raw_np)
    V["guards_sameep"] = ({1, 2, 3, 9}, None, None)
    V["ev_disp"] = ({5}, None, None)
    V["ev_grip"] = ({6}, None, None)
    V["events"] = ({5, 6}, None, None)
    V["guards+events(+burst)"] = ({1, 2, 3, 4, 5, 6}, None, None)
    for r in NOMINAL:
        V[f"v7+guards@{r}"] = ({1, 2, 3, 4}, taus[r], None)
        V[f"v7+guards+events@{r}"] = ({1, 2, 3, 4, 5, 6}, taus[r], None)
    # disp threshold sweep over the library quantile grid (train flag rate -> test metrics)
    grid = getattr(mj, "disp_qgrid", {})
    sweep = {}
    for q in (0.9, 0.95, 0.975, 0.98, 0.985, 0.99, 0.995, 1.0):
        thr = grid.get(q)
        if thr is None:
            continue
        fl = F["disp"] >= thr
        sweep[q] = {"thr": thr, "train_rate": float(fl[train].mean()), **summarize(F, fl, np.where(fl, 5, 0), test)}
    rep["disp_sweep"] = sweep
    # the smallest grid quantile whose TRAIN flag rate is <= 10 % (ideation C's nominal budget)
    q10 = None
    for q in sorted(grid):
        if float((F["disp"][train] >= grid[q]).mean()) <= 0.10:
            q10 = q
            break
    rep["disp_q_for_10pct_train"] = q10
    if q10 is not None:
        V[f"ev_disp@q{q10}"] = (set(), None, F["disp"] >= grid[q10])
        V[f"ev_disp@q{q10}+grip"] = ({6}, None, F["disp"] >= grid[q10])
        V[f"guards+ev_disp@q{q10}+grip"] = ({1, 2, 3, 4, 6}, None, F["disp"] >= grid[q10])
    for name, (which, tau, raw) in V.items():
        if raw is not None and which:
            force2, reason2 = flags_of(F, mj, which, tau)
            force, reason = force2 | raw, np.where(force2, reason2, np.where(raw, 5 if "disp" in name else 4, 0))
        elif raw is not None:
            force, reason = raw.copy(), np.where(raw, 8, 0)
        else:
            force, reason = flags_of(F, mj, which, tau)
        rep["variants"][name] = summarize(F, force, reason, test)
    # spell / decision statistics for context
    rep["spell_frac_test"] = {"fail": float(F["spell"][test & ~F["success"]].mean()) if (test & ~F["success"]).any() else None,
                              "succ": float(F["spell"][test & F["success"]].mean())}
    return rep


def md_table(rep):
    hdr = "| variant | flag rate (test dec) | in failed eps (share of flags) | failed touched early | failed touched ever | successful touched (first flag at) | median lead (dec) | lead >= 0 |"
    lines = [f"**{rep['arm']}** (SR {rep['sr']:.3f}; test {rep['test_episodes']} eps, {rep['test_fail']} failed; "
             f"disp_thr(q.9) {rep['fit']['disp_thr']:.3f}; disp_q for 10% train flags = {rep.get('disp_q_for_10pct_train')}; "
             f"V7 conf approx: {rep['exact']['v7_conf']})", "", hdr, "|" + "---|" * 8]
    for name, v in rep["variants"].items():
        lead = "-" if v.get("fail_lead_median") is None else f"{v['fail_lead_median']:+.1f}"
        ge0 = "-" if v.get("fail_lead_ge0") is None else f"{v['fail_lead_ge0']:.2f}"
        lines.append(f"| {name} | {v['flag_rate']:.3f} | {v['flags_in_fail_share']:.2f} (dec share {v['dec_in_fail_share']:.2f}) | "
                     f"{v['fail_touched_early']:.2f} | {v['fail_touched']:.2f} | {v['succ_touched']:.2f} ({v.get('succ_first_flag_pos_median', float('nan')):.2f}) | {lead} | {ge0} |")
    sw = rep.get("disp_sweep") or {}
    if sw:
        lines += ["", "disp threshold sweep (library quantile -> train flag rate; test: flag rate / failed early / failed ever / succ touched):",
                  "  " + "; ".join(f"q{q}: thr {v['thr']:.3f} train {v['train_rate']:.3f} -> {v['flag_rate']:.3f} / {v['fail_touched_early']:.2f} / {v['fail_touched']:.2f} / {v['succ_touched']:.2f}"
                                   for q, v in sw.items())]
    return "\n".join(lines)


def main(argv):
    OUT.mkdir(parents=True, exist_ok=True)
    arms = argv or [p.name for p in sorted(RUNS.iterdir()) if p.is_dir() and (p / "summary.json").exists()
                    and p.name.split("_")[-1] in ("cl2", "cl3")]
    mj_cache = {}
    md = ["# Shadow-trigger replay on the r02_g50 closed-loop logs (thresholds inits 0-24, measured on 25-49)", ""]
    for arm in arms:
        try:
            rep = analyse(arm, mj_cache)
        except Exception as e:  # noqa: BLE001
            print(f"{arm}: ERROR {e!r}", flush=True)
            continue
        (OUT / f"{arm}.json").write_text(json.dumps(rep, indent=1, default=float))
        t = md_table(rep)
        md += [t, ""]
        print(t, flush=True)
        print()
    (OUT / "replay_cl.md").write_text("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1:])
