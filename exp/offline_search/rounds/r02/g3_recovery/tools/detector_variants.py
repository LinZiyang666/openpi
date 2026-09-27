"""Replay V6's stuck detector + escalation on the per-decision extras of a V6 run for alternative clause settings
(no re-run needed: stuck_n, terminal, overtime, lag are recorded; with anchor = the executed row they do not depend
on V6's own recovery). Reports, per cache cell (stale decisions): STUCK rate, the base's err on flagged / unflagged
decisions and the R1 / R2 / R3 shares -- to pick the closed-loop (CL3) detector strictness.

    .venv/bin/python exp/offline_search/rounds/r02/g3_recovery/tools/detector_variants.py --res <dir> \
        --v6 V6sr_bl1rc1__<base> --base <base>
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np

MS = [f"{m}_{s}" for m in ("pi05", "groot") for s in ("spatial", "l10")]
VARIANTS = {
    "spec: sn>=2 | term | (ot & lag>5)": dict(sn=2, term=True, lag=5.0, ot_needs_still=False, ot=True),
    "lag>10": dict(sn=2, term=True, lag=10.0, ot_needs_still=False, ot=True),
    "(ot & lag>5) only while still": dict(sn=2, term=True, lag=5.0, ot_needs_still=True, ot=True),
    "no ot&lag clause": dict(sn=2, term=True, lag=5.0, ot_needs_still=False, ot=False),
    "sn>=3 | term": dict(sn=3, term=True, lag=5.0, ot_needs_still=False, ot=False),
}


def simulate(x, v, esc=2):
    ep, sn, term, ot, lag = x["ep"], x["stuck_n"], x["terminal"] > 0, x["overtime"], x["lag"]
    stale = x["regime"] == 2
    stuck = (sn >= v["sn"]) | (term if v["term"] else False)
    if v["ot"]:
        c = (ot > 1) & (lag > v["lag"])
        if v["ot_needs_still"]:
            c &= sn >= 1
        stuck |= c
    level = np.zeros(sn.size, np.int64)
    e_n = m_n = 0
    prev_ep = None
    for i in range(sn.size):
        if ep[i] != prev_ep:
            e_n = m_n = 0
            prev_ep = ep[i]
        m_n = m_n + 1 if sn[i] == 0 else 0
        if not stale[i]:
            continue
        if m_n >= 2:
            e_n = 0
        if stuck[i]:
            e_n += 1
            level[i] = min(3, 1 + (e_n - 1) // esc)
    return stuck, level


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--v6", required=True)
    ap.add_argument("--base", required=True)
    a = ap.parse_args(argv)
    for ms in MS:
        c = f"{ms}_cache"
        pv, pb = pathlib.Path(a.res) / a.v6 / f"{c}.npz", pathlib.Path(a.res) / a.base / f"{c}.npz"
        if not (pv.exists() and pb.exists()):
            continue
        zv, zb = np.load(pv), np.load(pb)
        x = {k[2:]: np.nan_to_num(zv[k], nan=0.0) for k in zv.files if k.startswith("x_")}
        x["ep"] = zv["ep"]
        e = zb["err"]
        s = zv["step"] > 0
        print(f"{ms}  (stale n={int(s.sum())}, base err {e[s].mean():.3f})")
        for name, v in VARIANTS.items():
            st, lv = simulate(x, v)
            f = st[s]
            ee = e[s]
            print(f"   {name:34s} STUCK {f.mean():.3f}  err flagged {ee[f].mean() if f.any() else np.nan:.3f} / "
                  f"unflagged {ee[~f].mean():.3f}  R1/R2/R3 {np.mean(lv[s] == 1):.3f}/{np.mean(lv[s] == 2):.3f}/"
                  f"{np.mean(lv[s] == 3):.3f}")
    return 0


if __name__ == "__main__":
    main()
