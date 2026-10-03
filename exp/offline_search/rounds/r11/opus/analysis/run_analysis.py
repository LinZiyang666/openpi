"""R11 post-sweep analysis (opus): one entry point, rerunnable on whatever arms are complete.

    cd /home/weiland/projects/openpi && taskset -c 2-9,46-53 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r11.opus.analysis.run_analysis

Reads (read-only) the four R11 test roots (r11_knob_1, r11_knob_2, r11_local_k3, r11_local_k4), the 4090 identity
root r11_local_idg, the R10 3-layer GC_dist reference arms (r10_corr3_*) and the R8 pure-policy arms (r08_main
r8_*_P10); the library-only settings in rounds/r11/knob/calibration/ and opus's frozen predictions in ../out/.
Writes only analysis/out/{TABLES.md, results.json, arms.csv} and analysis/cache/.

Measurement only: nothing here refits or retunes a knob. Every 'library' number is the frozen pre-run prediction.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

from exp.offline_search.rounds.r11.opus.analysis import extract as X
from exp.offline_search.rounds.r11.opus.analysis import stats as S

HERE = os.path.dirname(os.path.abspath(__file__))
OPUS = os.path.dirname(HERE)
METHODS = ["random", "periodic", "periodic_pgt1", "random_tail2",
           "distance", "error_hybrid", "adaptive_error_hybrid", "disagreement"]
MINE = ["random", "periodic", "periodic_pgt1", "random_tail2"]
STATE = ["distance", "error_hybrid", "adaptive_error_hybrid", "disagreement"]
SHORT = dict(off="off", random="R", periodic="P", periodic_pgt1="P+tail", random_tail2="R2", distance="D",
             error_hybrid="E", adaptive_error_hybrid="AE", disagreement="Dis", off_r10="R10 3-layer",
             pure="pure policy")
CELL_LABEL = {"pi05_l10_50": "π0.5 L10-50", "groot_l10_50": "GR00T L10-50", "pi05_spatial_50": "π0.5 Sp-50",
              "groot_spatial_50": "GR00T Sp-50", "pi05_l10_200": "π0.5 L10-200", "groot_l10_200": "GR00T L10-200",
              "pi05_l10_500": "π0.5 L10-500", "groot_l10_500": "GR00T L10-500"}
EXPECTED_TEST_ARMS = 94
IR_GRID = (0.25, 0.30, 0.35, 0.40)


def f3(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.3f}"


def f4s(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:+.4f}"


def pps(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:+.{d}f}"


def pci(lo, hi, d=1):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "—"
    return f"[{100 * lo:+.{d}f}, {100 * hi:+.{d}f}]"


def ir_f(model, v, g, k, o=0.0):
    cv, cm = X.COSTS[model]
    return v * (cv + cm * (g + (1 - g) * k + o))


class Arm:
    def __init__(self, res):
        sp = res["spec"]
        self.res, self.spec = res, sp
        self.root, self.arm, self.kind = sp["root"], sp["arm"], sp["kind"]
        self.model, self.cell, self.method = sp["model"], sp["cell"], sp["method"]
        self.target, self.pred, self.setting = sp["target"], sp["pred"], sp["setting"]
        self.ep, self.an = res["episodes"], res["anchors"]
        self.hw = res.get("host") or ("H100" if self.kind == "r8" else "?")
        self.n = len(self.ep)
        self.sr = float(self.ep["success"].mean()) if self.n else float("nan")
        led = res["ledger"]
        cv, cm = X.COSTS[self.model]
        self.ir = cv * led["v"] + cm * led["m"] if led.get("v") is not None else float("nan")
        if self.kind == "r8" and not np.isfinite(self.ir):
            self.ir = 0.5
        e = self.ep
        if self.kind != "r8":
            N, V, M = e["N"].sum(), e["V"].sum(), e["M"].sum()
            G, K, O = e["G"].sum(), e["K"].sum(), e["O"].sum()
            self.N, self.V, self.M, self.G, self.K, self.O = map(int, (N, V, M, G, K, O))
            self.v = V / N
            self.g = G / V
            self.k = K / max(V - G, 1)
            self.o = O / V
            self.ir_log = (cv * V + cm * M) / N
            self.ledger_ok = (led.get("decisions") == N and led.get("vision") == V and led.get("misses") == M)
        self.name = f"{self.root}/{self.arm}"

    @property
    def label(self):
        t = "" if self.target is None else f"@{self.target:.2f}"
        return f"{SHORT.get(self.method, self.method)}{t}"


def lib_base():
    d = json.load(open(os.path.join(OPUS, "out", "curves.json")))
    return {x["cell"]: x["base"] for x in d}


def lib_predictions():
    """opus's frozen per-arm predictions (PREDICTION.md, 2026-10-02 21:10 CDT)."""
    d = json.load(open(os.path.join(OPUS, "out", "predictions.json")))
    return {(x["cell"], x["method"], round(x["target"], 4)): x for x in d}


def lib_sim(arms):
    """Library-replay schedule diagnostics for opus's methods at the frozen settings (ir_model, library only)."""
    from exp.offline_search.rounds.r11.opus import ir_model as M
    out, cache = {}, {}
    for a in arms:
        if a.method not in MINE + ["off"] or a.kind not in ("r11", "idg"):
            continue
        model, suite, size = a.cell.split("_")
        key = (a.cell, a.method, a.setting)
        if key in out:
            continue
        if a.cell not in cache:
            cache[a.cell] = M.load_episodes(model, suite, int(size))
        eps = cache[a.cell]
        K = {"off": lambda: M.Knob(), "random": lambda: M.Random(a.setting),
             "random_tail2": lambda: M.Random(a.setting, tail=2),
             "periodic": lambda: M.DitheredGapCap(a.setting),
             "periodic_pgt1": lambda: M.DitheredGapCap(a.setting, post_guard_tail=1)}[a.method]()
        out[key] = M.simulate(model, eps, K)
    return out


# ================================================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)

    specs = X.discover()
    done = [s for s in specs if s["complete"]]
    with Pool(args.workers) as pool:
        res = pool.map(X.load_arm, done, chunksize=1)
    arms = []
    for r in res:
        a = Arm(r)
        if a.kind in ("r11", "idg") and a.n < X.MIN_N:
            continue
        arms.append(a)
    test = [a for a in arms if a.kind == "r11"]
    n_test_total = sum(1 for s in specs if s["kind"] == "r11")
    final = len(test) >= EXPECTED_TEST_ARMS
    missing = sorted(f"{s['root']}/{s['arm']}" for s in specs if s["kind"] == "r11"
                     and f"{s['root']}/{s['arm']}" not in {a.name for a in test})

    base = lib_base()
    preds = lib_predictions()
    sims = lib_sim(arms)

    # ------------------------------------------------------------------ references per cell
    by_cell = defaultdict(list)
    for a in arms:
        if a.kind in ("r11", "idg", "r10"):
            by_cell[a.cell].append(a)
    pure = {a.cell: a for a in arms if a.kind == "r8"}
    off_ref, off_src = {}, {}
    for c, L in by_cell.items():
        cand = ([a for a in L if a.kind == "r11" and a.method == "off"] +
                [a for a in L if a.kind == "idg"] + [a for a in L if a.kind == "r10"])
        if cand:
            off_ref[c] = cand[0]
            off_src[c] = {"r11": "R11 same-batch", "idg": "R11 4090 duplicate", "r10": "R10 GC_dist (earlier batch)"}[
                cand[0].kind]

    # ------------------------------------------------------------------ bootstrap per cell
    boots = {}
    for ci_, (c, L) in enumerate(sorted(by_cell.items())):
        keys = set()
        for a in L:
            keys |= {(int(e["task"]), int(e["init"])) for e in a.ep}
        cb = S.CellBoot(keys, args.boot, args.seed + ci_)
        for a in L:
            cb.add(a.name, a.ep, costs=X.COSTS[a.model])
        pc = c.rsplit("_", 1)[0]
        if pc in pure:
            cb.add("pure", pure[pc].ep, ir_const=pure[pc].ir)
        offs = [a for a in L if a.method in ("off", "off_r10")]
        if offs:
            cb.add_pool("offpool", [a.ep for a in offs], X.COSTS[offs[0].model])
        boots[c] = cb

    out = []
    P = out.append
    R = {}  # results.json
    stamp = time.strftime("%Y-%m-%d %H:%M %Z")
    P(f"# R11 opus analysis tables — {'FINAL' if final else 'INTERIM'}\n")
    P(f"Generated {stamp} by `exp/offline_search/rounds/r11/opus/analysis/run_analysis.py` "
      f"(bootstrap B = {args.boot}, stratified by task, seed {args.seed}).\n")
    P(f"Test arms complete: **{len(test)} / {n_test_total}** (expected {EXPECTED_TEST_ARMS}). "
      f"Reference arms: {sum(a.kind == 'r10' for a in arms)} R10 3-layer, {len(pure)} R8 pure policy, "
      f"{sum(a.kind == 'idg' for a in arms)} 4090 duplicate.\n")
    cnt = defaultdict(int)
    for a in test:
        cnt[a.method] += 1
    P("Arms per method: " + ", ".join(f"{SHORT[m]} {cnt[m]}" for m in ["off"] + METHODS) + "\n")
    P("Knob-off reference used per cell-size: " + "; ".join(
        f"{CELL_LABEL[c]} = {off_src[c]} ({off_ref[c].hw})" for c in X.CELLS if c in off_ref) + "\n")
    if missing:
        P(f"<details><summary>{len(missing)} test arms not yet complete</summary>\n\n" +
          "\n".join(f"- {m}" for m in missing) + "\n</details>\n")
    R.update(final=final, n_test=len(test), generated=stamp, off_source={c: off_src[c] for c in off_src})

    # ================================================================== overview grid
    P("\n## Overview: SR @ realized IR (Δ SR vs knob-off in pp)\n")
    for c in X.CELLS:
        if c not in boots:
            continue
        off = off_ref.get(c)
        pc = c.rsplit("_", 1)[0]
        pure_txt = f"{pure[pc].sr:.3f} @ {pure[pc].ir:.3f}" if pc in pure else "—"
        off_txt = f"{off.sr:.3f} @ {off.ir:.3f} ({off_src[c]}, {off.hw})" if off is not None else "—"
        P(f"\n**{CELL_LABEL[c]}** — knob off {off_txt}; pure policy {pure_txt}\n")
        tg = sorted({a.target for a in test if a.cell == c and a.target is not None})
        if not tg:
            continue
        ms = [m for m in METHODS if any(a.cell == c and a.method == m for a in test)]
        P("| target | " + " | ".join(SHORT[m] for m in ms) + " |")
        P("|---|" + "---|" * len(ms))
        for t in tg:
            row = []
            for m in ms:
                L = [a for a in test if a.cell == c and a.method == m and a.target == t]
                if not L:
                    row.append("")
                    continue
                a = L[0]
                d = f" ({pps(a.sr - off.sr)})" if off is not None else ""
                row.append(f"{a.sr:.3f} @ {a.ir:.3f}{d}")
            P(f"| {t:.2f} | " + " | ".join(row) + " |")

    # ================================================================== 0. audits
    P("\n## 0. Integrity audits (decision logs vs ledger, schedule rules, keyed coins)\n")
    P("| arm | hw | n | ledger N/V/M match | cadence violations | rule checks (bad) | coin checks (bad) | "
      "run/cap bad | p checks (bad) | guard-flag bad | other calls | miss reasons |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|")
    bad_any = 0
    for a in sorted([a for a in arms if a.kind in ("r11", "idg", "r10")], key=lambda a: (a.cell, a.method, a.target or 0)):
        au = a.res["audit"]
        cad = int(a.ep["cad_bad"].sum())
        nbad = (not a.ledger_ok) + cad + au["rule_bad"] + au["coin_bad"] + au["run_bad"] + au["cap_bad"] + \
            au["p_bad"] + au["guard_flag_bad"] + au["missing_dec"]
        bad_any += nbad > 0
        P(f"| {a.arm} | {a.hw} | {a.n} | {'yes' if a.ledger_ok else '**NO**'} | {cad} | "
          f"{au['rule_checked']} ({au['rule_bad']}) | {au['coin_checked']} ({au['coin_bad']}) | "
          f"{au['run_bad']}/{au['cap_bad']} | {au['p_checked']} ({au['p_bad']}) | {au['guard_flag_bad']} | {a.O} | "
          f"{', '.join(f'{k}:{v}' for k, v in sorted(au['miss_reasons'].items()))} |")
    P(f"\nArms with any audit failure: **{bad_any}**.\n")
    R["audit_failures"] = bad_any

    # ================================================================== 1. IR calibration
    P("\n## 1. IR calibration and accounting decomposition (all methods)\n")
    P("Realized IR from the cost ledger: `IR = v·(c_v + c_m·f)`, `f = g + (1 − g)·k + o` with v = looks/slots, "
      "g = guard-call share of looks, k = knob-call share of non-guard looks, o = other calls per look. Library "
      "values: v, g from the frozen calibration (astra's own v, g for her methods; opus's base otherwise), "
      "k backed out of `pred_IR_lib`. The realized-minus-library gap is attributed exactly (Shapley over 5 factors) "
      "to: look rate v; **base guard gap** (closed-loop knob-off g minus library g); **guard feedback** (this arm's g "
      "minus knob-off g); knob share k; other calls. `informed` = library knob share k applied with the measured "
      "closed-loop v and knob-off g, i.e. what the model predicts once the only offline unknown is known.\n")
    P("| cell-size | method | target | hw | SR | IR | lib pred | IR − target | v | g (off → arm) | k real / lib | "
      "Δ look | Δ base guard | Δ guard feedback | Δ knob share | Δ other | informed | IR − informed |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    calib = []
    for a in sorted(test, key=lambda a: (X.CELLS.index(a.cell), METHODS.index(a.method) if a.method in METHODS else -1,
                                         a.target or 0)):
        if a.method == "off":
            continue
        b = base[a.cell]
        v_l = a.spec.get("lib_v") or b["v"]
        g_l = a.spec.get("lib_g") if a.spec.get("lib_g") is not None else b["g"]
        cv, cm = X.COSTS[a.model]
        k_l = float(np.clip(((a.pred / v_l - cv) / cm - g_l) / (1 - g_l), 0, 1))
        off = off_ref.get(a.cell)
        g_off = off.g if off is not None else float("nan")
        fn = lambda v, d1, d2, k, o: ir_f(a.model, v, g_l + d1 + d2, k, o)
        if np.isfinite(g_off):
            x0, x1 = (v_l, 0.0, 0.0, k_l, 0.0), (a.v, g_off - g_l, a.g - g_off, a.k, a.o)
            sh = S.shapley(fn, x0, x1)
            informed = ir_f(a.model, a.v, g_off, k_l)
        else:
            sh = np.full(5, np.nan)
            informed = float("nan")
        row = dict(arm=a.arm, cell=a.cell, method=a.method, target=a.target, hw=a.hw, sr=a.sr, ir=a.ir, pred=a.pred,
                   v=a.v, g=a.g, k=a.k, o=a.o, v_lib=v_l, g_lib=g_l, k_lib=k_l, g_off=g_off,
                   d_look=sh[0], d_base_guard=sh[1], d_guard_feedback=sh[2], d_knob=sh[3], d_other=sh[4],
                   informed=informed, off_source=off_src.get(a.cell))
        calib.append(row)
        P(f"| {CELL_LABEL[a.cell]} | {SHORT[a.method]} | {a.target:.2f} | {a.hw} | {f3(a.sr)} | {f3(a.ir)} | "
          f"{f3(a.pred)} | {f4s(a.ir - a.target)} | {a.v:.3f} | {f3(g_off)} → {a.g:.3f} | {a.k:.3f} / {k_l:.3f} | "
          f"{f4s(sh[0])} | {f4s(sh[1])} | {f4s(sh[2])} | {f4s(sh[3])} | {f4s(sh[4])} | {f3(informed)} | "
          f"{f4s(a.ir - informed)} |")
    R["calibration"] = calib

    P("\n### 1b. Per-method summary\n")
    P("| method | arms | mean IR − target | mean abs | max abs | within ±.02 | mean Δ look | mean Δ base guard | "
      "mean Δ guard feedback | mean Δ knob share | mean abs(IR − informed) | within ±.01 of informed |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|")
    msum = {}
    for m in METHODS:
        L = [r for r in calib if r["method"] == m]
        if not L:
            continue
        d = np.array([r["ir"] - r["target"] for r in L])
        inf = np.array([r["ir"] - r["informed"] for r in L])
        mean = lambda k: float(np.nanmean([r[k] for r in L]))
        msum[m] = dict(n=len(L), bias=float(d.mean()), mae=float(np.abs(d).mean()), max=float(np.abs(d).max()),
                       within02=int((np.abs(d) <= .02).sum()), d_look=mean("d_look"),
                       d_base_guard=mean("d_base_guard"), d_guard_feedback=mean("d_guard_feedback"),
                       d_knob=mean("d_knob"), mae_informed=float(np.nanmean(np.abs(inf))),
                       within01_informed=int((np.abs(inf) <= .01).sum()))
        s = msum[m]
        P(f"| {SHORT[m]} | {s['n']} | {s['bias']:+.4f} | {s['mae']:.4f} | {s['max']:.4f} | {s['within02']}/{s['n']} | "
          f"{s['d_look']:+.4f} | {s['d_base_guard']:+.4f} | {s['d_guard_feedback']:+.4f} | {s['d_knob']:+.4f} | "
          f"{s['mae_informed']:.4f} | {s['within01_informed']}/{s['n']} |")
    R["calibration_summary"] = msum

    P("\n### 1c. Knob-off base: closed-loop vs library\n")
    P("| cell-size | source | hw | SR | IR | library base IR | v real / lib | g real / lib | IR − lib |")
    P("|---|---|---|---|---|---|---|---|---|")
    for c in X.CELLS:
        for a in [x for x in by_cell.get(c, []) if x.method in ("off", "off_r10")]:
            b = base[c]
            P(f"| {CELL_LABEL[c]} | {a.root} | {a.hw} | {f3(a.sr)} | {f3(a.ir)} | {f3(b['IR'])} | "
              f"{a.v:.3f} / {b['v']:.3f} | {a.g:.3f} / {b['g']:.3f} | {f4s(a.ir - b['IR'])} |")

    P("\n### 1d. Random knob: realized knob share vs ρ (binomial check)\n")
    P("| cell-size | method | target | ρ | eligible looks | realized k | z |")
    P("|---|---|---|---|---|---|---|")
    for a in test:
        if a.method != "random":
            continue
        E = a.V - a.G
        rho = float(a.setting)
        z = (a.k - rho) / np.sqrt(rho * (1 - rho) / E)
        P(f"| {CELL_LABEL[a.cell]} | R | {a.target:.2f} | {rho:.4f} | {E} | {a.k:.4f} | {z:+.2f} |")

    # ================================================================== 2. frontier
    P("\n## 2. Efficiency frontier (SR vs realized IR)\n")
    P("Paired on the 500 test-A episodes. Δ vs off: knob arm minus the cell's knob-off reference (+b / −c discordant "
      "episodes, exact McNemar p, bootstrap 95% CI in pp). Efficiency = ΔSR / ΔIR (SR points per unit IR).\n")
    front = {}
    for c in X.CELLS:
        if c not in boots:
            continue
        cb = boots[c]
        off = off_ref.get(c)
        P(f"\n### {CELL_LABEL[c]}\n")
        P("| arm | hw | SR [95% CI] | IR | Δ vs off (pp) [95% CI] | +b/−c | p | ΔIR | ΔSR/ΔIR |")
        P("|---|---|---|---|---|---|---|---|---|")
        rows = []
        for a in by_cell[c]:
            rows.append(a)
        rows.sort(key=lambda a: (a.kind != "r10", a.method not in ("off",), a.ir))
        for a in rows:
            A = cb.arms[a.name]
            lo, hi = S.ci(A["sr_b"])
            if off is not None and a is not off:
                O_ = cb.arms[off.name]
                d_b = A["sr_b"] - O_["sr_b"]
                b_, c_, n_ = cb.paired(a.name, off.name)
                dlo, dhi = S.ci(d_b)
                dIR = a.ir - off.ir
                eff = (a.sr - off.sr) / dIR if dIR > 0.01 else float("nan")
                P(f"| {a.label} ({a.root.replace('r11_', '')}) | {a.hw} | {a.sr:.3f} [{lo:.3f}, {hi:.3f}] | "
                  f"{a.ir:.3f} | {pps(a.sr - off.sr)} {pci(dlo, dhi)} | +{b_}/−{c_} | {S.mcnemar_p(b_, c_):.2g} | "
                  f"{dIR:+.3f} | {'—' if not np.isfinite(eff) else f'{eff:+.2f}'} |")
            else:
                P(f"| **{a.label} ({a.root.replace('r11_', '')})** | {a.hw} | {a.sr:.3f} [{lo:.3f}, {hi:.3f}] | "
                  f"{a.ir:.3f} | (reference) | | | | |")
            front.setdefault(c, []).append(dict(arm=a.arm, root=a.root, method=a.method, target=a.target, sr=a.sr,
                                                ir=a.ir, hw=a.hw))
        if "pure" in cb.arms:
            A = cb.arms["pure"]
            lo, hi = S.ci(A["sr_b"])
            s = f"| pure policy (R8 P10) | H100 | {A['sr']:.3f} [{lo:.3f}, {hi:.3f}] | {A['ir']:.3f} |"
            if off is not None:
                b_, c_, n_ = cb.paired("pure", off.name)
                dlo, dhi = S.ci(A["sr_b"] - cb.arms[off.name]["sr_b"])
                dIR = A["ir"] - off.ir
                s += (f" {pps(A['sr'] - off.sr)} {pci(dlo, dhi)} | +{b_}/−{c_} | {S.mcnemar_p(b_, c_):.2g} | "
                      f"{dIR:+.3f} | {(A['sr'] - off.sr) / dIR:+.2f} |")
            else:
                s += " | | | | |"
            P(s)
    R["frontier"] = front

    # ---------------------------------------------------------------- 2b. matched realized IR along each curve
    P("\n### 2b. SR at matched realized IR (piecewise-linear along knob-off → each method's arms; linear extrapolation ≤ .01 IR past the end points, none further)\n")
    P("| cell-size | method | " + " | ".join(f"SR @ IR {x:.2f}" for x in IR_GRID) + " |")
    P("|---|---|" + "---|" * len(IR_GRID))
    curves = {}
    for c in X.CELLS:
        if c not in boots or c not in off_ref:
            continue
        cb, off = boots[c], off_ref[c]
        for m in METHODS:
            L = sorted([a for a in by_cell[c] if a.kind == "r11" and a.method == m], key=lambda a: a.ir)
            if not L:
                continue
            pts = [off] + L
            xs = np.array([cb.arms[a.name]["ir"] for a in pts])
            ys = np.array([cb.arms[a.name]["sr"] for a in pts])
            xsb = np.stack([cb.arms[a.name]["ir_b"] for a in pts])
            ysb = np.stack([cb.arms[a.name]["sr_b"] for a in pts])
            curves[(c, m)] = (xs, ys, xsb, ysb)
            cells_ = []
            for x in IR_GRID:
                y = S.interp_curve(x, xs, ys)
                if np.isfinite(y):
                    yb = S.interp_curve_boot(np.full(args.boot, x), xsb, ysb)
                    lo, hi = S.ci(yb)
                    cells_.append(f"{y:.3f} [{lo:.3f}, {hi:.3f}]")
                else:
                    cells_.append("—")
            P(f"| {CELL_LABEL[c]} | {SHORT[m]} | " + " | ".join(cells_) + " |")

    # ---------------------------------------------------------------- 2c. advantage over random / periodic at matched IR
    def advantage(ref_m):
        rows, per_m = [], defaultdict(list)
        for c in X.CELLS:
            if (c, ref_m) not in curves:
                continue
            cb = boots[c]
            xs, ys, xsb, ysb = curves[(c, ref_m)]
            for a in by_cell[c]:
                if a.kind != "r11" or a.method in ("off", ref_m):
                    continue
                A = cb.arms[a.name]
                y = S.interp_curve(A["ir"], xs, ys)
                if not np.isfinite(y):
                    rows.append((a, float("nan"), None))
                    continue
                yb = S.interp_curve_boot(A["ir_b"], xsb, ysb)
                adv_b = A["sr_b"] - yb
                rows.append((a, A["sr"] - y, adv_b))
                per_m[a.method].append((A["sr"] - y, adv_b))
        return rows, per_m

    adv_res = {}
    for ref_m in ("random", "periodic"):
        rows, per_m = advantage(ref_m)
        P(f"\n### 2c. SR advantage over {SHORT[ref_m]} at the same realized IR (arm SR minus {SHORT[ref_m]}'s "
          f"interpolated curve, incl. knob-off point)\n")
        P("| method | arms in range | mean advantage (pp) [95% CI] | bootstrap p | per-arm (cell, target: pp) |")
        P("|---|---|---|---|---|")
        adv_res[ref_m] = {}
        for m in METHODS:
            if m == ref_m or m not in per_m:
                continue
            L = per_m[m]
            pt = float(np.mean([x for x, _ in L]))
            bb = np.nanmean(np.stack([b for _, b in L]), axis=0)
            lo, hi = S.ci(bb)
            arms_txt = "; ".join(f"{CELL_LABEL[a.cell]} {a.target:.2f}: {pps(v)}" for a, v, _ in rows
                                 if a.method == m and np.isfinite(v))
            oor = sum(1 for a, v, _ in rows if a.method == m and not np.isfinite(v))
            P(f"| {SHORT[m]} | {len(L)}{f' (+{oor} out of range)' if oor else ''} | {pps(pt, 2)} {pci(lo, hi, 2)} | "
              f"{S.two_sided_p_boot(bb):.2g} | {arms_txt} |")
            adv_res[ref_m][m] = dict(n=len(L), pp=100 * pt, lo=100 * lo, hi=100 * hi, p=S.two_sided_p_boot(bb))
    R["advantage_matched_ir"] = adv_res

    # ---------------------------------------------------------------- 2d / 2e. efficiency and chord, two references
    def ref_name(c, mode):
        if mode == "same":
            return off_ref[c].name if c in off_ref else None
        return "offpool" if "offpool" in boots[c].arms else None

    def eff_table(mode, verbose):
        res = {}
        for m in METHODS + ["pure"]:
            num = den = 0.0
            num_b = np.zeros(args.boot)
            den_b = np.zeros(args.boot)
            k = 0
            dsr = []
            for c in X.CELLS:
                if c not in boots or ref_name(c, mode) is None:
                    continue
                cb = boots[c]
                O_ = cb.arms[ref_name(c, mode)]
                names = ["pure"] if m == "pure" and "pure" in cb.arms else \
                    [a.name for a in by_cell[c] if a.kind == "r11" and a.method == m]
                if m == "pure" and c not in ("pi05_l10_50", "groot_l10_50", "pi05_spatial_50", "groot_spatial_50"):
                    names = []
                for nm in names:
                    A = cb.arms[nm]
                    num += A["sr"] - O_["sr"]
                    den += A["ir"] - O_["ir"]
                    num_b += A["sr_b"] - O_["sr_b"]
                    den_b += A["ir_b"] - O_["ir_b"]
                    dsr.append(A["sr"] - O_["sr"])
                    k += 1
            if not k:
                continue
            e_b = num_b / den_b
            lo, hi = S.ci(e_b)
            res[m] = dict(n=k, dsr=num, dir=den, eff=num / den, lo=lo, hi=hi, mean_dsr=float(np.mean(dsr)))
            P(f"| {SHORT[m]}{' (50-cells)' if m == 'pure' else ''} | {k} | {100 * num:+.1f} | {den:+.3f} | "
              f"{10 * num / den:+.2f} [{10 * lo:+.2f}, {10 * hi:+.2f}] | {100 * np.mean(dsr):+.2f} |")
        return res

    def chord_table(mode, verbose):
        res = {}
        for m in METHODS:
            vals, boots_ = [], []
            txt = []
            for c in X.CELLS:
                if c not in boots or ref_name(c, mode) is None or "pure" not in boots[c].arms:
                    continue
                cb = boots[c]
                O_, Pu = cb.arms[ref_name(c, mode)], cb.arms["pure"]
                for a in by_cell[c]:
                    if a.kind != "r11" or a.method != m:
                        continue
                    A = cb.arms[a.name]
                    lam = (A["ir"] - O_["ir"]) / (Pu["ir"] - O_["ir"])
                    y = O_["sr"] + lam * (Pu["sr"] - O_["sr"])
                    lam_b = (A["ir_b"] - O_["ir_b"]) / (Pu["ir_b"] - O_["ir_b"])
                    yb = O_["sr_b"] + lam_b * (Pu["sr_b"] - O_["sr_b"])
                    vals.append(A["sr"] - y)
                    boots_.append(A["sr_b"] - yb)
                    txt.append(f"{CELL_LABEL[c]} {a.target:.2f}: {pps(A['sr'] - y)}")
            if not vals:
                continue
            bb = np.mean(np.stack(boots_), axis=0)
            lo, hi = S.ci(bb)
            res[m] = dict(n=len(vals), pp=100 * float(np.mean(vals)), lo=100 * lo, hi=100 * hi,
                          p=S.two_sided_p_boot(bb))
            P(f"| {SHORT[m]} | {len(vals)} | {pps(np.mean(vals), 2)} {pci(lo, hi, 2)} | {S.two_sided_p_boot(bb):.2g} |"
              + (f" {'; '.join(txt)} |" if verbose else ""))
        return res

    P("\n### 2d. Pooled efficiency over knob-off: ΣΔSR / ΣΔIR (paired bootstrap 95% CI)\n")
    P("| method | arms | ΣΔSR (pp) | ΣΔIR | SR pp per +0.1 IR [95% CI] | mean ΔSR per arm (pp) |")
    P("|---|---|---|---|---|---|")
    R["efficiency"] = eff_table("same", True)
    P("\n### 2e. SR advantage over the cache ↔ pure-policy chord at the same IR\n")
    P("Chord = running knob-off on a share (1 − λ) of episodes and pure policy on the rest, which reaches any IR "
      "between them on a (nearly) straight line. A knob is only worth having where it sits above this line.\n")
    P("| method | arms | mean advantage over chord (pp) [95% CI] | bootstrap p | per-arm (cell, target: pp) |")
    P("|---|---|---|---|---|")
    R["chord_advantage"] = chord_table("same", True)
    P("\n### 2d′ / 2e′. Sensitivity: knob-off reference = average of every knob-off run of the cell\n")
    P("Each cell has 2–3 knob-off runs of the identical configuration (R11 same batch, R10 GC_dist, and the 4090 "
      "duplicate for GR00T L10-50); their per-episode average damps single-run batch noise (largest: GR00T L10-500, "
      "R11 .858 vs R10 .906, §4).\n")
    P("| method | arms | ΣΔSR (pp) | ΣΔIR | SR pp per +0.1 IR [95% CI] | mean ΔSR per arm (pp) |")
    P("|---|---|---|---|---|---|")
    R["efficiency_pooled_ref"] = eff_table("pool", False)
    P("")
    P("| method | arms | mean advantage over chord (pp) [95% CI] | bootstrap p |")
    P("|---|---|---|---|")
    R["chord_advantage_pooled_ref"] = chord_table("pool", False)

    # ---------------------------------------------------------------- 2f. matched-target paired contrasts
    P("\n### 2f. Matched-target paired contrasts (same cell-size and target label; check ΔIR before reading)\n")
    P("| contrast | pairs (arms) | episodes | +b / −c | pp | exact p | mean ΔIR | IR-adjusted pp* |")
    P("|---|---|---|---|---|---|---|---|")
    pair_res = {}
    for m1, m2 in [("periodic", "random"), ("periodic_pgt1", "periodic"), ("periodic_pgt1", "random"),
                   ("random_tail2", "random"), ("random_tail2", "periodic"), ("distance", "random"),
                   ("error_hybrid", "random"), ("adaptive_error_hybrid", "random"), ("disagreement", "random"),
                   ("adaptive_error_hybrid", "error_hybrid")]:
        B_ = C_ = N_ = 0
        dIR, adj, k = [], [], 0
        for c in X.CELLS:
            if c not in boots:
                continue
            cb = boots[c]
            for a in by_cell[c]:
                if a.kind != "r11" or a.method != m1:
                    continue
                o = [x for x in by_cell[c] if x.kind == "r11" and x.method == m2 and x.target == a.target]
                if not o:
                    continue
                o = o[0]
                b_, c_, n_ = cb.paired(a.name, o.name)
                B_ += b_; C_ += c_; N_ += n_; k += 1
                dIR.append(a.ir - o.ir)
                # IR adjustment: slope of the random curve between its neighbouring points around the pair
                if (c, "random") in curves:
                    xs, ys, _, _ = curves[(c, "random")]
                    yb = S.interp_curve(a.ir, xs, ys)
                    yo = S.interp_curve(o.ir, xs, ys)
                    if np.isfinite(yb) and np.isfinite(yo):
                        adj.append((a.sr - o.sr) - (yb - yo))
        if not k:
            continue
        pp_ = 100 * (B_ - C_) / N_
        adj_txt = f"{100 * np.mean(adj):+.2f} ({len(adj)} pairs)" if adj else "—"
        P(f"| {SHORT[m1]} − {SHORT[m2]} | {k} | {N_} | +{B_}/−{C_} | {pp_:+.2f} | {S.mcnemar_p(B_, C_):.2g} | "
          f"{np.mean(dIR):+.4f} | {adj_txt} |")
        pair_res[f"{m1}-{m2}"] = dict(arms=k, eps=N_, b=B_, c=C_, pp=pp_, p=S.mcnemar_p(B_, C_),
                                      dIR=float(np.mean(dIR)), adj=float(np.mean(adj)) * 100 if adj else None)
    P("\n*IR-adjusted: SR difference minus the random curve's SR difference between the two realized IRs.\n")
    R["matched_target"] = pair_res

    # ================================================================== 3. where the calls landed
    P("\n## 3. Where the extra calls landed\n")
    P("Anchors = vision decisions (one per ten controls). 'Eligible' = non-guard anchors (where a knob may act). "
      "Rates are pooled over the method's complete arms; relative intensity = bin rate / the arm's own overall rate, "
      "averaged over arms (1.0 = spread evenly).\n")

    def per_arm_bins(a, binvals, edges):
        an = a.an
        elig = an["kind"] != X.KIND_GUARD
        knob = an["kind"] == X.KIND_KNOB
        guard = an["kind"] == X.KIND_GUARD
        b = np.clip(np.digitize(binvals, edges) - 1, 0, len(edges) - 2)
        nb = len(edges) - 1
        E_ = np.bincount(b[elig], minlength=nb)
        K_ = np.bincount(b[knob], minlength=nb)
        G_ = np.bincount(b[guard], minlength=nb)
        A_ = np.bincount(b, minlength=nb)
        return A_, E_, K_, G_

    def bin_table(title, fn, edges, labels, suite=None):
        P(f"\n### {title}\n")
        P("| method | arms | " + " | ".join(labels) + " |")
        P("|---|---|" + "---|" * len(labels))
        res_ = {}
        for m in ["off"] + METHODS:
            L = [a for a in test if a.method == m and len(a.an) and (suite is None or f"_{suite}_" in a.cell)]
            if not L:
                continue
            rel_k, rel_g, share = [], [], []
            for a in L:
                A_, E_, K_, G_ = per_arm_bins(a, fn(a), edges)
                kr = K_.sum() / max(E_.sum(), 1)
                gr = G_.sum() / max(A_.sum(), 1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    rel_k.append((K_ / E_) / kr if kr > 0 else np.full(len(A_), np.nan))
                    rel_g.append((G_ / A_) / gr)
                share.append(A_ / A_.sum())
            rk, rg, sh = np.nanmean(rel_k, 0), np.nanmean(rel_g, 0), np.mean(share, 0)
            txt = " | ".join(f"{sh[i]:.2f} / g {rg[i]:.2f}" + (f" / k {rk[i]:.2f}" if m != "off" else "")
                             for i in range(len(labels)))
            P(f"| {SHORT[m]} | {len(L)} | {txt} |")
            res_[m] = dict(share=sh.tolist(), rel_guard=rg.tolist(), rel_knob=rk.tolist())
        P("\nCell format: share of anchors in the bin / relative guard-call intensity / relative knob-call intensity.\n")
        return res_

    q3 = {}
    q3["by_progress"] = bin_table(
        "3a. By retrieved-demo progress (top-1 library progress at the anchor, a task-progress proxy)",
        lambda a: np.nan_to_num(a.an["prog"], nan=0.0), np.array([0, .2, .4, .6, .8, 1.0001]),
        ["0–.2", ".2–.4", ".4–.6", ".6–.8", ".8–1"])

    def norm_time(a):
        N_ep = a.ep["N"][a.an["ep"]].astype(float)
        return a.an["step"] / np.maximum(N_ep, 1)

    q3["by_time"] = bin_table("3b. By normalized episode time (step / episode length)", norm_time,
                              np.array([0, .2, .4, .6, .8, 1.0001]), ["0–.2", ".2–.4", ".4–.6", ".6–.8", ".8–1"])

    def abs_time(a):
        return a.an["step"].astype(float)

    q3["by_abs_time"] = bin_table("3c. LIBERO-10 cells only: by absolute decision step (1 step = 5 controls; "
                                  "step limit 104 = failure)", abs_time, np.array([0, 10, 20, 40, 60, 80, 1e9]),
                                  ["0–9", "10–19", "20–39", "40–59", "60–79", "80+"], suite="l10")

    # 3d. lag after guard / after any call
    P("\n### 3d. Knob-call rate per eligible anchor vs anchors since the last guard call\n")
    P("| method | arms | 1 (right after) | 2 | 3 | 4–6 | ≥7 | no guard call yet | overall |")
    P("|---|---|---|---|---|---|---|---|---|")
    lag_res = {}
    for m in METHODS:
        L = [a for a in test if a.method == m and len(a.an)]
        if not L:
            continue
        cnt_e = np.zeros(6)
        cnt_k = np.zeros(6)
        for a in L:
            an = a.an
            lag = S.since_last(an["kind"] == X.KIND_GUARD, an["ep"])
            b = np.select([lag == 1, lag == 2, lag == 3, lag <= 6, lag < 10 ** 6], [0, 1, 2, 3, 4], 5)
            el = an["kind"] != X.KIND_GUARD
            kn = an["kind"] == X.KIND_KNOB
            cnt_e += np.bincount(b[el], minlength=6)
            cnt_k += np.bincount(b[kn], minlength=6)
        r_ = cnt_k / np.maximum(cnt_e, 1)
        lag_res[m] = r_.tolist()
        P(f"| {SHORT[m]} | {len(L)} | " + " | ".join(f"{r_[i]:.3f} ({int(cnt_e[i])})" for i in range(6)) +
          f" | {cnt_k.sum() / cnt_e.sum():.3f} |")
    q3["lag_after_guard"] = lag_res

    P("\n### 3e. Guard-call rate at the next anchor, by what happened at this anchor (guard feedback mechanism)\n")
    P("| method | arms | after cache look | after guard call | after knob call | arm guard share g |")
    P("|---|---|---|---|---|---|")
    fb_res = {}
    for m in ["off"] + METHODS:
        L = [a for a in test if a.method == m and len(a.an)]
        if not L:
            continue
        num = np.zeros(3)
        den = np.zeros(3)
        for a in L:
            an = a.an
            same = an["ep"][1:] == an["ep"][:-1]
            prev, nxt = an["kind"][:-1][same], an["kind"][1:][same]
            for i, kd in enumerate((X.KIND_LOOK, X.KIND_GUARD, X.KIND_KNOB)):
                s_ = prev == kd
                den[i] += s_.sum()
                num[i] += (nxt[s_] == X.KIND_GUARD).sum()
        r_ = num / np.maximum(den, 1)
        gs = np.mean([a.g for a in L])
        fb_res[m] = r_.tolist()
        P(f"| {SHORT[m]} | {len(L)} | {r_[0]:.3f} | {r_[1]:.3f} | " +
          (f"{r_[2]:.3f}" if den[2] else "—") + f" | {gs:.3f} |")
    q3["guard_after"] = fb_res

    # 3f. success vs failure
    P("\n### 3f. Calls on eventually successful vs failed episodes\n")
    P("| method | arms | fail share of episodes | fail share of slots | fail share of knob calls | "
      "knob calls / ep (succ / fail) | knob rate per eligible (succ / fail) | guard share g (succ / fail) | "
      "slots / ep (succ / fail) |")
    P("|---|---|---|---|---|---|---|---|---|")
    sf_res = {}
    for m in ["off"] + METHODS:
        L = [a for a in test if a.method == m]
        if not L:
            continue
        acc = defaultdict(float)
        for a in L:
            e = a.ep
            for tag, sel in (("s", e["success"] == 1), ("f", e["success"] == 0)):
                acc[f"n{tag}"] += sel.sum()
                acc[f"N{tag}"] += e["N"][sel].sum()
                acc[f"K{tag}"] += e["K"][sel].sum()
                acc[f"G{tag}"] += e["G"][sel].sum()
                acc[f"V{tag}"] += e["V"][sel].sum()
        g = lambda k: acc[k]
        fs_ep = g("nf") / (g("nf") + g("ns"))
        fs_sl = g("Nf") / (g("Nf") + g("Ns"))
        fs_k = g("Kf") / max(g("Kf") + g("Ks"), 1)
        kpe = (g("Ks") / max(g("ns"), 1), g("Kf") / max(g("nf"), 1))
        kre = (g("Ks") / max(g("Vs") - g("Gs"), 1), g("Kf") / max(g("Vf") - g("Gf"), 1))
        gsh = (g("Gs") / max(g("Vs"), 1), g("Gf") / max(g("Vf"), 1))
        spe = (g("Ns") / max(g("ns"), 1), g("Nf") / max(g("nf"), 1))
        sf_res[m] = dict(fail_ep=fs_ep, fail_slots=fs_sl, fail_knob=fs_k, kpe=kpe, kre=kre, g=gsh, slots=spe)
        P(f"| {SHORT[m]} | {len(L)} | {fs_ep:.3f} | {fs_sl:.3f} | " + (f"{fs_k:.3f}" if m != "off" else "—") +
          f" | {kpe[0]:.1f} / {kpe[1]:.1f} | {kre[0]:.3f} / {kre[1]:.3f} | {gsh[0]:.3f} / {gsh[1]:.3f} | "
          f"{spe[0]:.1f} / {spe[1]:.1f} |")
    q3["success_failure"] = sf_res

    # 3g. cache-only stretches
    P("\n### 3g. Cache-only stretches (consecutive anchors without any policy call; 1 anchor = 10 controls)\n")
    P("Library columns: opus's ir_model replay at the same frozen setting (open-loop, library guard flags; "
      "opus methods and knob-off only).\n")
    P("| arm | hw | mean max stretch / ep | p90 max | share of looks in stretches ≥ 4 | longest | "
      "library mean max | library p90 max |")
    P("|---|---|---|---|---|---|---|---|")
    run_res = {}
    for a in sorted([a for a in test if a.method in ["off"] + METHODS],
                    key=lambda a: (["off"] + METHODS).index(a.method) * 100 + X.CELLS.index(a.cell) + (a.target or 0)):
        an = a.an
        call = an["kind"] != X.KIND_LOOK
        eps_, lens = S.runs_without(call, an["ep"])
        mx = np.zeros(a.n)
        np.maximum.at(mx, eps_, lens) if len(eps_) else None
        looks = (~call).sum()
        ge4 = lens[lens >= 4].sum() / max(looks, 1)
        sim = sims.get((a.cell, a.method, a.setting))
        run_res[a.arm] = dict(mean_max=float(mx.mean()), p90=float(np.percentile(mx, 90)), ge4=float(ge4),
                              longest=int(mx.max()), lib_mean_max=sim["mean_max_cache_run"] if sim else None,
                              lib_p90=sim["p90_max_cache_run"] if sim else None)
        P(f"| {CELL_LABEL[a.cell]} {a.label} | {a.hw} | {mx.mean():.2f} | {np.percentile(mx, 90):.0f} | {ge4:.3f} | "
          f"{int(mx.max())} | " + (f"{sim['mean_max_cache_run']:.2f} | {sim['p90_max_cache_run']:.0f} |" if sim else "— | — |"))
    q3["cache_runs"] = run_res

    # 3i. random vs periodic on matched (cell, target) pairs
    P("\n### 3i. Random vs periodic on matched (cell-size, target) pairs\n")
    P("| cell-size | target | SR R / P | IR R / P | P vs R +b/−c | fail share of knob calls R / P | "
      "knob rate per eligible in failed eps R / P | mean max cache stretch R / P | guard share g R / P |")
    P("|---|---|---|---|---|---|---|---|---|")
    rp_rows = []
    for c in X.CELLS:
        for t in sorted({a.target for a in test if a.cell == c and a.target is not None}):
            r = [a for a in test if a.cell == c and a.method == "random" and a.target == t]
            q = [a for a in test if a.cell == c and a.method == "periodic" and a.target == t]
            if not (r and q):
                continue
            r, q = r[0], q[0]
            b_, c_, n_ = boots[c].paired(q.name, r.name)

            def stats_(a):
                e = a.ep
                f = e["success"] == 0
                fk = e["K"][f].sum() / max(e["K"].sum(), 1)
                kf = e["K"][f].sum() / max((e["V"][f] - e["G"][f]).sum(), 1)
                return fk, kf, run_res[a.arm]["mean_max"]
            sr_, sq_ = stats_(r), stats_(q)
            rp_rows.append(dict(cell=c, target=t, b=b_, c=c_))
            P(f"| {CELL_LABEL[c]} | {t:.2f} | {r.sr:.3f} / {q.sr:.3f} | {r.ir:.3f} / {q.ir:.3f} | +{b_}/−{c_} | "
              f"{sr_[0]:.3f} / {sq_[0]:.3f} | {sr_[1]:.3f} / {sq_[1]:.3f} | {sr_[2]:.2f} / {sq_[2]:.2f} | "
              f"{r.g:.3f} / {q.g:.3f} |")
    q3["random_vs_periodic"] = rp_rows

    # 3j. where do extra calls pay off: gain by init difficulty measured in an independent knob-off run
    P("\n### 3j. Gain by init difficulty (difficulty from an independent knob-off run of the same init)\n")
    P("Difficulty proxy per (task, init): guard calls among the first 10 anchors (split at its 1/3 and 2/3 quantiles; ties make the low bin larger), and the outcome, in a knob-off run "
      "that is *not* the comparison reference (R10 batch when the reference is R11, and vice versa), so chance "
      "failures of the reference do not leak into the split. Cells without two knob-off runs are skipped.\n")
    P("| method | arms | eps | ΔSR low early-guard (pp) | mid | high early-guard | ΔSR on inits the other run failed | "
      "on inits it solved |")
    P("|---|---|---|---|---|---|---|---|")
    het = {}
    for m in METHODS + ["pure"]:
        acc = np.zeros((5, 2))   # rows: tercile0..2, failed, solved ; cols: sum diff, count
        k = 0
        for c in X.CELLS:
            off = off_ref.get(c)
            if off is None or c not in boots:
                continue
            others = [a for a in by_cell[c] if a.method in ("off", "off_r10") and a is not off and a.kind != off.kind]
            if not others:
                continue
            cov = others[0]
            an = cov.an
            first10 = np.zeros(cov.n)
            # anchors are sorted by episode then step; count guard calls among each episode's first 10 anchors
            start = np.r_[0, np.flatnonzero(an["ep"][1:] != an["ep"][:-1]) + 1]
            idx_in_ep = np.arange(len(an)) - np.repeat(start, np.diff(np.r_[start, len(an)]))
            sel = (idx_in_ep < 10) & (an["kind"] == X.KIND_GUARD)
            np.add.at(first10, an["ep"][sel], 1)
            key_cov = {(int(e["task"]), int(e["init"])): (first10[j], int(e["success"])) for j, e in enumerate(cov.ep)}
            vals = np.array([v[0] for v in key_cov.values()])
            q1, q2 = np.quantile(vals, [1 / 3, 2 / 3])
            cb = boots[c]
            O_ = cb.arms[off.name]
            names = (["pure"] if "pure" in cb.arms else []) if m == "pure" else \
                [a.name for a in by_cell[c] if a.kind == "r11" and a.method == m]
            for nm in names:
                A = cb.arms[nm]
                k += 1
                for key, j in cb.pos.items():
                    if key not in key_cov or not (A["m"][j] and O_["m"][j]):
                        continue
                    d = A["s"][j] - O_["s"][j]
                    g0, sc = key_cov[key]
                    t_ = 0 if g0 <= q1 else (1 if g0 <= q2 else 2)
                    acc[t_] += (d, 1)
                    acc[3 if sc == 0 else 4] += (d, 1)
        if not k:
            continue
        r_ = acc[:, 0] / np.maximum(acc[:, 1], 1)
        het[m] = dict(arms=k, diff=r_.tolist(), n=acc[:, 1].tolist())
        P(f"| {SHORT[m]} | {k} | {int(acc[:3, 1].sum())} | " + " | ".join(
            f"{100 * r_[i]:+.1f} ({int(acc[i, 1])})" for i in range(5)) + " |")
    q3["gain_by_difficulty"] = het

    # 3h. state knobs: closed-loop score vs calibration threshold
    P("\n### 3h. State knobs: closed-loop score vs the frozen calibration threshold\n")
    P("Static methods: the threshold was set so that a share q (= dose) of *calibration* eligible looks lies above it; "
      "the closed-loop share above it should equal q if closed-loop scores were distributed like the calibration "
      "reference. Adaptive: the online dose q starts at q0 and is pushed by the realized spend.\n")
    P("| arm | q (dose) | threshold | closed-loop share above threshold | score p10 / p50 / p90 | realized k | "
      "mean online q (adaptive) |")
    P("|---|---|---|---|---|---|---|")
    st_res = {}
    for a in sorted([a for a in test if a.method in STATE], key=lambda a: (a.method, a.cell, a.target)):
        an = a.an
        el = an["kind"] != X.KIND_GUARD
        sc = an["score"][el]
        sc = sc[np.isfinite(sc)]
        thr = a.spec.get("threshold")
        above = float((sc > thr).mean()) if thr is not None and len(sc) else float("nan")
        qs = np.percentile(sc, [10, 50, 90]) if len(sc) else [np.nan] * 3
        mq = float(np.nanmean(an["q"][el])) if a.method == "adaptive_error_hybrid" else float("nan")
        st_res[a.arm] = dict(dose=a.setting, thr=thr, above=above, q=list(map(float, qs)), k=a.k, mean_q=mq)
        P(f"| {CELL_LABEL[a.cell]} {a.label} | {float(a.setting):.3f} | {f3(thr) if thr is not None else '—'} | "
          f"{f3(above)} | {qs[0]:.3f} / {qs[1]:.3f} / {qs[2]:.3f} | {a.k:.3f} | {f3(mq)} |")
    q3["state_scores"] = st_res
    R["where_calls"] = q3

    # ================================================================== 4. hardware
    P("\n## 4. Hardware check (RTX 4090 vs H100, identical config and checkpoints)\n")
    P("Same-configuration knob-off pairs on the same 500 episodes. 'same-hw' pairs (H100 vs H100, R11 vs R10 batch) "
      "measure run-to-run nondeterminism; 'cross-hw' pairs add any hardware effect. First divergence = first anchor "
      "where the retrieved top-1 row or the call decision differs between the two runs of an episode.\n")
    P("| cell-size | arm A (hw) | arm B (hw) | type | SR A / B | +b/−c | discordant | p | IR A / B | g A / B | "
      "slots/ep A / B | step-0 top-1 equal | diverge before first policy call | identical episodes | "
      "median first-divergence step |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    hw_rows = []
    for c in X.CELLS:
        L = [a for a in by_cell.get(c, []) if a.method in ("off", "off_r10")]
        for i in range(len(L)):
            for j in range(i + 1, len(L)):
                a, b = L[i], L[j]
                if a.kind == b.kind == "r10":
                    continue
                cb = boots[c]
                b_, c_, n_ = cb.paired(a.name, b.name)
                typ = "same-hw" if a.hw == b.hw else "cross-hw"
                ident, fdiv, s0, pre = first_divergence(a, b)
                hw_rows.append(dict(cell=c, arm_a=a.name, arm_b=b.name, hw_a=a.hw, hw_b=b.hw, type=typ, sr_a=a.sr, sr_b=b.sr,
                                    nb=b_, nc=c_, n=n_, p=S.mcnemar_p(b_, c_), ir_a=a.ir, ir_b=b.ir, g_a=a.g, g_b=b.g,
                                    ident=ident, fdiv=fdiv, step0_equal=s0, pre_call_div=pre))
                P(f"| {CELL_LABEL[c]} | {a.root} ({a.hw}) | {b.root} ({b.hw}) | {typ} | {a.sr:.3f} / {b.sr:.3f} | "
                  f"+{b_}/−{c_} | {(b_ + c_) / n_:.3f} | {S.mcnemar_p(b_, c_):.2g} | {a.ir:.3f} / {b.ir:.3f} | "
                  f"{a.g:.3f} / {b.g:.3f} | {a.ep['N'].mean():.1f} / {b.ep['N'].mean():.1f} | {s0:.3f} | {pre:.3f} | "
                  f"{ident:.3f} | {fdiv:.0f} |")
    for typ in ("same-hw", "cross-hw"):
        L = [r for r in hw_rows if r["type"] == typ]
        if L:
            P(f"\n- {typ}: {len(L)} pairs; mean discordant share {np.mean([(r['nb'] + r['nc']) / r['n'] for r in L]):.3f}; "
              f"mean |ΔSR| {100 * np.mean([abs(r['sr_a'] - r['sr_b']) for r in L]):.1f} pp; "
              f"mean ΔIR (A − B) {np.mean([r['ir_a'] - r['ir_b'] for r in L]):+.4f}; "
              f"step-0 top-1 equal {np.mean([r['step0_equal'] for r in L]):.3f}; diverge before first policy call "
              f"{np.mean([r['pre_call_div'] for r in L]):.3f}; identical episodes {np.mean([r['ident'] for r in L]):.3f}")
    # knob arms by hardware: realized - library IR
    P("\n**Knob arms by serving hardware** (IR calibration error and Δ vs off, by hw):\n")
    P("| hw | knob arms | mean IR − library | mean IR − informed | mean ΔSR vs off (pp) |")
    P("|---|---|---|---|---|")
    for hw in sorted({r["hw"] for r in calib}):
        L = [r for r in calib if r["hw"] == hw]
        dsr = [r["sr"] - off_ref[r["cell"]].sr for r in L if r["cell"] in off_ref]
        P(f"| {hw} | {len(L)} | {np.mean([r['ir'] - r['pred'] for r in L]):+.4f} | "
          f"{np.nanmean([r['ir'] - r['informed'] for r in L]):+.4f} | {100 * np.mean(dsr):+.2f} |")
    R["hardware"] = hw_rows

    # ================================================================== 5. opus prediction scorecard
    P("\n## 5. opus pre-registered predictions (PREDICTION.md, 2026-10-02 21:10 CDT) vs measurement\n")
    P("Only arms whose (cell-size, method, target) was in opus's 38-arm grid; the coordinator's final grid moved "
      "Spatial targets to .25/.32/.40 (those arms have library predictions only, from sol's addendum).\n")
    P("| cell-size | method | target | pred IR lib / informed | real IR | pred SR (Δ pp) | real SR (Δ vs off pp) |")
    P("|---|---|---|---|---|---|---|")
    sc_rows = []
    for a in sorted([a for a in test if a.method in MINE], key=lambda a: (a.cell, a.method, a.target)):
        p = preds.get((a.cell, a.method, round(a.target, 4)))
        if not p:
            continue
        off = off_ref.get(a.cell)
        dsr = a.sr - off.sr if off is not None else float("nan")
        sc_rows.append(dict(cell=a.cell, method=a.method, target=a.target, model=a.model, pred_lib=p["pred_IR_lib"],
                            pred_inf=p["pred_IR_informed"], ir=a.ir, pred_sr=p["pred_SR"], pred_dsr=p["pred_dSR_pp"],
                            sr=a.sr, dsr=dsr))
        P(f"| {CELL_LABEL[a.cell]} | {SHORT[a.method]} | {a.target:.2f} | {p['pred_IR_lib']:.3f} / "
          f"{p['pred_IR_informed']:.3f} | {a.ir:.3f} | {p['pred_SR']:.3f} ({p['pred_dSR_pp']:+.1f}) | "
          f"{a.sr:.3f} ({pps(dsr)}) |")
    if sc_rows:
        e_lib = np.array([r["ir"] - r["pred_lib"] for r in sc_rows])
        e_inf = np.array([r["ir"] - r["pred_inf"] for r in sc_rows])
        P(f"\n- **P1** (IR): within ±.02 of library prediction {int((np.abs(e_lib) <= .02).sum())}/{len(sc_rows)} "
          f"(predicted ≥ 34/38 ⇒ ≥ {int(np.ceil(34 / 38 * len(sc_rows)))}/{len(sc_rows)}); "
          f"|real − informed| ≤ .012 in {int((np.abs(e_inf) <= .012).sum())}/{len(sc_rows)} "
          f"(predicted ≥ 32/38). Mean signed error vs library: π0.5 "
          f"{np.mean([r['ir'] - r['pred_lib'] for r in sc_rows if r['model'] == 'pi05'] or [np.nan]):+.4f} "
          f"(predicted ≈ −.008), GR00T L10-50 "
          f"{np.mean([r['ir'] - r['pred_lib'] for r in sc_rows if r['cell'] == 'groot_l10_50'] or [np.nan]):+.4f} "
          f"(predicted ≈ +.005).")
        pr = {}
        for r in sc_rows:
            pr.setdefault((r["cell"], r["target"]), {})[r["method"]] = r
        pairs = [(v["random"], v["periodic"]) for v in pr.values() if "random" in v and "periodic" in v]
        if pairs:
            close = sum(abs(r["ir"] - p["ir"]) <= .01 for r, p in pairs)
            pcl = sum(abs(p["ir"] - p["pred_lib"]) < abs(r["ir"] - r["pred_lib"]) for r, p in pairs)
            P(f"- **P2** (matched spend): R/P within .01 IR in {close}/{len(pairs)} pairs (predicted ≥ 14/16); "
              f"P closer to its library prediction in {pcl}/{len(pairs)} (predicted ≥ 10/16).")
        if "periodic-random" in pair_res:
            x = pair_res["periodic-random"]
            P(f"- **P4** (P − R, matched target, pooled): {x['pp']:+.2f} pp over {x['arms']} pairs, p = {x['p']:.2g} "
              f"(predicted +0.6, 80% PI −0.9 to +2.1; refuted if ≤ −1).")
        if "periodic_pgt1-periodic" in pair_res:
            x = pair_res["periodic_pgt1-periodic"]
            P(f"- **P6a** (P+tail − P): {x['pp']:+.2f} pp over {x['arms']} pairs (predicted +0.3, no direction claimed).")
        if "random_tail2-random" in pair_res:
            x = pair_res["random_tail2-random"]
            P(f"- **P6b** (R2 − R): {x['pp']:+.2f} pp over {x['arms']} pairs (predicted −0.5, no direction claimed).")
        d_err = [r["dsr"] * 100 - r["pred_dsr"] for r in sc_rows if np.isfinite(r["dsr"])]
        if d_err:
            P(f"- SR change vs off: realized − predicted = {np.mean(d_err):+.2f} pp on average over {len(d_err)} arms "
              f"(sd {np.std(d_err):.2f}).")
    R["prediction_scorecard"] = sc_rows

    # ================================================================== 6. near-pure cells / default-off rule
    P("\n## 6. Near-pure cells: where the knob should default to off\n")
    P("Library-only signal: mean whole-episode-out error between the cache's served (GC_dist-corrected) chunk and the "
      "policy's own chunk at the same library state (opus anchor tables, the same replay that calibrates the knob). "
      "Measured columns are test-A aggregates (knob-off runs: R11 same batch, R10 GC_dist; pure = R8 P10); they are "
      "used only to *check* the library signal, not to fit anything. Δ columns use the pooled knob-off reference.\n")
    P("| cell-size | eps/task | library served-chunk error | library guard rate | knob-off SR (runs) | pure SR | "
      "gap to pure (pp) | P @ .25 Δ (pp) | E @ .25 Δ (pp) | R @ .25 Δ (pp) | rule: error < .40 ⇒ off |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    from exp.offline_search.rounds.r11.opus import ir_model as IM
    near = []
    for model in ("pi05", "groot"):
        for suite in ("l10", "spatial"):
            for size in (50, 200, 500):
                c = f"{model}_{suite}_{size}"
                eps = IM.load_episodes(model, suite, size)
                err = float(np.mean(np.concatenate([e.risk for e in eps])))
                gl = base[c]["g"]
                srs = []
                for root in (f"r10_corr3_{model}", f"r10_corr3_{model}_b"):
                    pth = f"{X.BASE}/{root}/runs/r10_{model}_{suite}_{size}_GC_dist/summary.json"
                    if os.path.exists(pth):
                        srs.append(("R10", json.load(open(pth))["sr"]))
                        break
                for a in by_cell.get(c, []):
                    if a.kind == "idg":
                        srs.append(("R11-4090dup", a.sr))
                    elif a.method == "off":
                        srs.append(("R11", a.sr))
                pu = pure.get(f"{model}_{suite}")
                off_mean = float(np.mean([x for _, x in srs])) if srs else float("nan")
                gap = (pu.sr - off_mean) if pu is not None else float("nan")
                d = {}
                if c in boots and "offpool" in boots[c].arms:
                    cb = boots[c]
                    for m in ("periodic", "error_hybrid", "random"):
                        L = [a for a in by_cell[c] if a.kind == "r11" and a.method == m and a.target == 0.25]
                        if L:
                            d[m] = cb.arms[L[0].name]["sr"] - cb.arms["offpool"]["sr"]
                rule_off = err < 0.40
                near.append(dict(cell=c, err=err, g=gl, offs=srs, pure=pu.sr if pu else None, gap=gap,
                                 d=d, rule_off=rule_off))
                P(f"| {c} | {size // 10} | {err:.3f} | {gl:.3f} | " +
                  ", ".join(f"{k} {v:.3f}" for k, v in srs) + f" | {pu.sr:.3f} | {100 * gap:+.1f} | " +
                  " | ".join(pps(d[m]) if m in d else "—" for m in ("periodic", "error_hybrid", "random")) +
                  f" | {'off' if rule_off else 'on'} |")
    ok = sum((r["gap"] <= 0.02) == r["rule_off"] for r in near)
    P(f"\nRule 'library served-chunk error < .40 ⇒ knob off' agrees with 'measured gap to pure ≤ 2 pp' in "
      f"{ok}/{len(near)} cell-sizes (threshold chosen by eye on these aggregates: a hypothesis to validate on "
      f"non-test data, not a fitted result).\n")
    R["near_pure"] = near

    # ------------------------------------------------------------------ write
    with open(os.path.join(args.out, "TABLES.md"), "w") as fh:
        fh.write("\n".join(out) + "\n")
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(R, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    with open(os.path.join(args.out, "arms.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["root", "arm", "kind", "hw", "cell", "method", "target", "setting", "n", "sr", "ir", "pred",
                    "v", "g", "k", "o", "N", "V", "M", "G", "K", "O"])
        for a in arms:
            w.writerow([a.root, a.arm, a.kind, a.hw, a.cell, a.method, a.target, a.setting, a.n, f"{a.sr:.4f}",
                        f"{a.ir:.5f}", a.pred] + ([f"{a.v:.5f}", f"{a.g:.5f}", f"{a.k:.5f}", f"{a.o:.5f}", a.N, a.V,
                                                   a.M, a.G, a.K, a.O] if a.kind != "r8" else [""] * 10))
    print(f"wrote {args.out}/TABLES.md ({len(test)} test arms, final={final}) in {time.time() - t0:.0f}s")


def first_divergence(a, b):
    """Paired-episode determinism probe between two runs of the same configuration.
    Returns (share of episodes with identical anchor sequences, median first differing decision step,
    share with the same top-1 row at step 0, share diverging strictly before either run's first policy call).
    The policy's sampling noise is not seeded across server processes, so divergence right after the first call
    is expected even on identical hardware; divergence in the cache-only prefix (retrieval on the same rendered
    observation) is the hardware-sensitive part."""
    def seqs(x):
        an = x.an
        out = {}
        bounds = np.r_[0, np.flatnonzero(an["ep"][1:] != an["ep"][:-1]) + 1, len(an)]
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            e = x.ep[an["ep"][lo]]
            call = an["kind"][lo:hi] != X.KIND_LOOK
            out[(int(e["task"]), int(e["init"]))] = (an["step"][lo:hi], an["top1"][lo:hi], call)
        return out
    A, B = seqs(a), seqs(b)
    ident, fd, s0, pre = 0, [], 0, 0
    common = set(A) & set(B)
    for k in common:
        sa, ta, ca = A[k]
        sb, tb, cb = B[k]
        s0 += int(len(ta) and len(tb) and ta[0] == tb[0])
        n = min(len(sa), len(sb))
        diff = np.flatnonzero((sa[:n] != sb[:n]) | (ta[:n] != tb[:n]) | (ca[:n] != cb[:n]))
        fa = np.flatnonzero(ca)
        fb = np.flatnonzero(cb)
        first_call = min(fa[0] if len(fa) else 10 ** 6, fb[0] if len(fb) else 10 ** 6)
        if len(diff):
            fd.append(int(sa[diff[0]]))
            pre += int(diff[0] < first_call)
        elif len(sa) == len(sb):
            ident += 1
        else:
            fd.append(int(sa[n - 1]) if n else 0)
    m = max(len(common), 1)
    return ident / m, (float(np.median(fd)) if fd else float("nan")), s0 / m, pre / m


if __name__ == "__main__":
    sys.exit(main())
