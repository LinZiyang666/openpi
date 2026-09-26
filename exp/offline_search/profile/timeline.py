"""timeline -- per-episode decision timelines and trajectory-switching statistics of a run.

For every decision (in episode/step order): err, oracle_err, confidence, the chosen library row as
(library trajectory, library step), the oracle row likewise, phase_err, and a move flag relative
to the previous decision of the same episode:
  T track   chosen == library `next` of the previous choice (advanced exactly one entry)
  A advance same library trajectory, later step (skipping)          = stay (same row)
  B back    same trajectory, earlier step                           S switch (different trajectory)
Per cell summary: switch / track / advance / stay / back rates, mean run length on one trajectory,
mean err after each move type, and the same move rates for the oracle row (does the oracle itself
stay on one trajectory? -- i.e. is trajectory tracking even the right prior).

  python -m exp.offline_search.profile.timeline RUN_OR_METHOD_DIR [--method M] [--cells all]
         [--episode UID|IDX ...] [--worst-episodes N] [--random-episodes N]
"""
from __future__ import annotations

import argparse
import csv as csv_mod
import io
import sys
from pathlib import Path

import numpy as np

from . import common as C

FLAG = {0: ".", 1: "T", 2: "A", 3: "=", 4: "B", 5: "S", 6: "?"}


def moves(sel: np.ndarray, contiguous: np.ndarray, traj, lstep, nxt) -> np.ndarray:
    """Move code per decision (0 first/no-prev, 1 track, 2 advance, 3 stay, 4 back, 5 switch, 6 unknown)."""
    sel = np.asarray(sel, np.int64)
    n = sel.size
    code = np.zeros(n, np.int8)
    if n == 0:
        return code
    prev = np.r_[-1, sel[:-1]]
    v = contiguous & (prev >= 0) & (sel >= 0)
    code[v] = 6
    if traj is None:
        code[v & (sel == prev)] = 3
        return code
    t, tp = traj[np.maximum(sel, 0)], traj[np.maximum(prev, 0)]
    s, sp = lstep[np.maximum(sel, 0)], lstep[np.maximum(prev, 0)]
    same = v & (t == tp)
    code[v & (t != tp)] = 5
    code[same & (s > sp)] = 2
    code[same & (s < sp)] = 4
    code[same & (sel == prev)] = 3
    if nxt is not None:
        code[v & (nxt[np.maximum(prev, 0)] == sel) & (sel != prev)] = 1
    else:
        code[same & (s == sp + 1)] = 1
    return code


def _rates(code: np.ndarray, err: np.ndarray | None = None) -> dict:
    v = code > 0
    n = int(v.sum())
    d = {"trans": n}
    for k, name in ((1, "track"), (2, "advance"), (3, "stay"), (4, "back"), (5, "switch")):
        d[name] = float((code[v] == k).mean()) if n else np.nan
        if err is not None:
            m = code == k
            d[f"err|{name}"] = float(err[m].mean()) if m.any() else np.nan
    d["forward"] = d["track"] + d["advance"] if n else np.nan
    return d


def run_lengths(code: np.ndarray, first: np.ndarray) -> float:
    """Mean number of consecutive decisions on one library trajectory (breaks at switch / new episode)."""
    brk = first | (code == 5) | (code == 6) | (code == 0)
    idx = np.flatnonzero(brk)
    if idx.size == 0:
        return float(code.size)
    lens = np.diff(np.r_[idx, code.size])
    return float(lens.mean())


def lib_arrays(root, ms: str, names: list[str]) -> dict | None:
    """Trajectory arrays of several libraries concatenated with row / trajectory offsets, so decisions that refer
    to different libraries live in one index space (a library change is then always a switch)."""
    off, traj, tloc, lstep, nxt, lname = {}, [], [], [], [], []
    tot = tmax = 0
    has_next = True
    for n in names:
        lib = C.open_library(root, ms, n)
        if lib is None:
            return None
        t = lib.opt("traj", lib.opt("episode"))
        if t is None:
            return None
        t = np.asarray(t, np.int64)
        st = lib.opt("step")
        st = np.asarray(st, np.int64) if st is not None else np.zeros_like(t)
        nx = lib.opt("next")
        has_next &= nx is not None
        off[n] = tot
        traj.append(t + tmax); tloc.append(t); lstep.append(st)
        nxt.append(np.where(np.asarray(nx) >= 0, np.asarray(nx, np.int64) + tot, -1) if nx is not None else np.full(t.size, -1))
        lname += [n] * t.size
        tot += t.size
        tmax += int(t.max()) + 1 if t.size else 0
    return {"off": off, "traj": np.concatenate(traj), "traj_local": np.concatenate(tloc), "lstep": np.concatenate(lstep),
            "next": np.concatenate(nxt) if has_next else None, "lname": np.array(lname)}


def cell_timeline(md: Path, cell: str, root) -> dict:
    rc = C.RunCell(md, cell)
    qc = C.QueryCell(root, cell)
    names = sorted(set(rc.lib_names) | {"current"}, key=lambda n: (n != "current", n))
    LA = lib_arrays(root, qc.ms, names)
    rows = rc.rows
    o = np.lexsort((qc.step[rows], qc.ep[rows]))
    r_ = rows[o]
    ep, st = qc.ep[r_], qc.step[r_]
    contiguous = np.r_[False, (ep[1:] == ep[:-1]) & (st[1:] == st[:-1] + 1)]
    first = np.r_[True, ep[1:] != ep[:-1]]
    top1 = np.asarray(rc["top1"], np.int64)[o]
    if LA is not None:
        offs = np.array([LA["off"][n] for n in rc.lib_names], np.int64)
        top1g = top1 + offs[rc.lib_code[o]]
        traj, lstep, nxt = LA["traj"], LA["lstep"], LA["next"]
    else:
        top1g, traj, lstep, nxt = top1, None, None, None
    err = rc["err"][o].astype(np.float64)
    code = moves(top1g, contiguous, traj, lstep, nxt)
    out = {"cell": cell, "n": int(rc.n), **_rates(code, err), "runlen": run_lengths(code, first),
           "has_traj": traj is not None, "has_next": nxt is not None, "libs": ",".join(rc.lib_names)}
    ocode = orowg = None
    if rc.has("oracle_row"):
        orow = np.asarray(rc["oracle_row"], np.int64)[o]          # harness oracle: library current
        orowg = orow + (LA["off"]["current"] if LA is not None else 0)
        ocode = moves(orowg, contiguous, traj, lstep, nxt)
        orr = _rates(ocode)
        out.update({f"orc_{k}": v for k, v in orr.items() if k in ("switch", "track", "forward", "stay")})
        out["orc_runlen"] = run_lengths(ocode, first)
    table = {"o": o, "rows": r_, "ep": ep, "step": st, "code": code, "ocode": ocode, "top1": top1, "top1g": top1g,
             "orowg": orowg, "err": err}
    return {"summary": out, "table": table, "rc": rc, "qc": qc, "LA": LA}


def _where(LA, g) -> str:
    if LA is None or g < 0:
        return str(g)
    n = LA["lname"][g]
    return ("" if n == "current" else f"{n}/") + f"{LA['traj_local'][g]}:{LA['lstep'][g]}"


def episode_rows(T: dict, ei: int) -> list[dict]:
    t, rc, qc, LA = T["table"], T["rc"], T["qc"], T["LA"]
    m = np.flatnonzero(t["ep"] == ei)
    out = []
    for j in m:
        p = t["o"][j]
        r = int(t["rows"][j])
        d = {"step": int(t["step"][j]), "p": float(qc.p[r]), "err": float(t["err"][j]),
             "orc_err": float(rc["oracle_err"][p]) if rc.has("oracle_err") else None,
             "conf": float(rc["confidence"][p]) if rc.has("confidence") else None,
             "top1": int(t["top1"][j]), "choice": _where(LA, int(t["top1g"][j])), "move": FLAG[int(t["code"][j])],
             "phase": float(rc["phase_err"][p]) if rc.has("phase_err") else None}
        if t["orowg"] is not None:
            d["oracle"] = _where(LA, int(t["orowg"][j]))
            d["omove"] = FLAG[int(t["ocode"][j])]
        out.append(d)
    return out


SUM_COLS = [("cell", "cell"), ("n", "n"), ("switch", "switch"), ("track", "track"), ("advance", "adv"), ("stay", "stay"),
            ("back", "back"), ("runlen", "runlen", ".1f"), ("err|track", "err|T"), ("err|advance", "err|A"),
            ("err|stay", "err|="), ("err|switch", "err|S"), ("orc_switch", "orc_switch"), ("orc_track", "orc_track"),
            ("orc_stay", "orc_stay"), ("orc_runlen", "orc_runlen", ".1f")]
EP_COLS = [("step", "step"), ("p", "p", ".2f"), ("err", "err"), ("orc_err", "orc_err"), ("conf", "conf", ".4g"),
           ("choice", "choice ep:step"), ("move", "mv"), ("oracle", "oracle ep:step"), ("omove", "omv"), ("phase", "phase")]


def _csv_rows(T: dict) -> list[dict]:
    t, rc, qc, LA = T["table"], T["rc"], T["qc"], T["LA"]
    out = []
    for j in range(t["rows"].size):
        p = t["o"][j]
        r = int(t["rows"][j])
        g = int(t["top1g"][j])
        d = {"cell": T["summary"]["cell"], "row": r, "ep": int(t["ep"][j]), "step": int(t["step"][j]),
             "p": float(qc.p[r]), "err": float(t["err"][j]),
             "oracle_err": float(rc["oracle_err"][p]) if rc.has("oracle_err") else None,
             "conf": float(rc["confidence"][p]) if rc.has("confidence") else None,
             "library": rc.lib_of(p), "top1": int(t["top1"][j]),
             "top1_traj": int(LA["traj_local"][g]) if LA is not None else None,
             "top1_step": int(LA["lstep"][g]) if LA is not None else None,
             "move": FLAG[int(t["code"][j])],
             "phase_err": float(rc["phase_err"][p]) if rc.has("phase_err") else None}
        if t["orowg"] is not None:
            og = int(t["orowg"][j])
            d.update(oracle_row=int(rc["oracle_row"][p]),
                     oracle_traj=int(LA["traj_local"][og]) if LA is not None and og >= 0 else None,
                     oracle_step=int(LA["lstep"][og]) if LA is not None and og >= 0 else None,
                     oracle_move=FLAG[int(t["ocode"][j])])
        out.append(d)
    return out


def cell_job(a) -> dict:
    """Worker: timeline of one (method, cell) reduced to plain data (summary, selected episodes, csv rows)."""
    md, cell, root, episodes, worst, rand, seed, want_csv = a
    T = cell_timeline(Path(md), cell, root)
    qc = T["qc"]
    sel = []
    for e in episodes:
        e = str(e)
        ei = qc.uid_to_ep.get(e, int(e) if e.lstrip("-").isdigit() else None)
        if ei is not None and ei < len(qc.episodes):
            sel.append(ei)
    t = T["table"]
    if worst or rand:
        eps = np.unique(t["ep"])
        me = np.array([t["err"][t["ep"] == e].mean() for e in eps])
        if worst:
            sel += [int(e) for e in eps[np.argsort(-me, kind="stable")[:worst]]]
        if rand:
            sel += [int(e) for e in np.random.default_rng(seed).choice(eps, min(rand, eps.size), replace=False)]
    eps_out = []
    for ei in dict.fromkeys(sel):
        ep = qc.episodes[ei]
        title = (f"{cell} ep {ei} ({ep.get('uid')}) task {ep.get('task_id')} "
                 f"{'SUCCESS' if ep.get('success') else 'failure'} len {ep.get('num_steps')}")
        eps_out.append((f"{cell}:{ei}", title, episode_rows(T, ei)))
    out = {"summary": T["summary"], "episodes": eps_out, "csv_cols": None, "csv_text": None}
    if want_csv:  # render the CSV body here (in parallel); the parent only concatenates
        rows = _csv_rows(T)
        cols = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        buf = io.StringIO()
        w = csv_mod.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
        for r in rows:
            w.writerow({k: C._csv_val(r.get(k)) for k in cols})
        out["csv_cols"], out["csv_text"] = cols, buf.getvalue()
    return out


def write_csv_parts(path: Path, parts: list[dict]) -> None:
    """Byte-identical to common.write_csv over all rows when every part has the same columns (the normal case)."""
    parts = [p for p in parts if p["csv_cols"]]
    cols0 = parts[0]["csv_cols"] if parts else []
    if all(p["csv_cols"] == cols0 for p in parts):
        with open(path, "w", newline="") as f:
            csv_mod.DictWriter(f, fieldnames=cols0).writeheader()
            for p in parts:
                f.write(p["csv_text"])
        return
    rows = [dict(zip(p["csv_cols"], vals)) for p in parts for vals in csv_mod.reader(io.StringIO(p["csv_text"]))]
    C.write_csv(path, rows)


def run(run_path, method=None, root=C.DEFAULT_ROOT, cells="all", episodes=(), worst=0, rand=0, seed=0, out=None,
        csv=True, quiet=False, procs=None) -> dict:
    root = Path(root)
    res = {}
    mds = C.method_dirs(run_path, method)
    jobs = [(str(md), c, str(root), tuple(episodes), worst, rand, seed, csv)
            for md in mds for c in C.run_cells(md, C.expand_cells(cells))]
    done = C.pmap(cell_job, jobs, procs)                      # one pool over every (method, cell)
    for md in mds:
        Ts = [r for j, r in zip(jobs, done) if j[0] == str(md)]
        L = [f"## timeline {md.name}  (moves: T track=next entry, A advance, = stay, B back, S switch)"]
        L.append(C.md_table([T["summary"] for T in Ts], SUM_COLS, "per cell move statistics (rates over decisions with a previous decision)"))
        ep_dump = {}
        for T in Ts:
            for key, title, er in T["episodes"]:
                ep_dump[key] = er
                L.append("")
                L.append(C.md_table(er, EP_COLS, title))
        s = "\n".join(L)
        if not quiet:
            print(s)
        rdir = C.report_dir(root, "timeline", md.name, out, multi=len(mds) > 1)
        C.write_json(rdir / "timeline.json", {"method": md.name, "summary": [T["summary"] for T in Ts], "episodes": ep_dump})
        C.write_csv(rdir / "timeline_summary.csv", [T["summary"] for T in Ts])
        if csv:
            write_csv_parts(rdir / "timeline.csv", Ts)
        if not quiet:
            print(f"\n[timeline] {md.name}: {len(Ts)} cells -> {rdir}")
        res[md.name] = {"summary": [T["summary"] for T in Ts], "episodes": ep_dump, "report_dir": str(rdir)}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--method", default=None)
    ap.add_argument("--root", default=str(C.DEFAULT_ROOT))
    ap.add_argument("--cells", default="all")
    ap.add_argument("--episode", action="append", default=[], help="episode uid or index (repeatable)")
    ap.add_argument("--worst-episodes", type=int, default=0)
    ap.add_argument("--random-episodes", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-csv", action="store_true", help="skip the all-decisions timeline.csv")
    ap.add_argument("--procs", type=int, default=C.DEFAULT_PROCS, help="worker processes (default: all cores)")
    ap.add_argument("--out", default=None, help="report dir (one subdir per method when several are reported)")
    a = ap.parse_args(argv)
    run(a.run, a.method, a.root, a.cells, a.episode, a.worst_episodes, a.random_episodes, a.seed, a.out, not a.no_csv,
        procs=a.procs)


if __name__ == "__main__":
    sys.exit(main())
