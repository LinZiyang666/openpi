"""explain -- decision inspector for one method x cell of a runner output.

Per selected decision prints: query (episode uid, task, success, step/num_steps, p, third), the
method's top-k (library row, library episode/step/progress, score, err to a_inf, good-flag), the
oracle candidate (row, err, rank inside the method's top-k), confidence, gap top1-top2, extras
(x_* fields), coverage context (n_cand, n_good, recorded-B0 pick and its rank) and, for the B0
reference formula, the raw per-field sims (cos v0 / cos v1 on pooled keys, rs L2 on valid dims)
plus the normalized/fused B0 scores of the method's top-1 vs the oracle vs the recorded B0 pick,
with the field that favoured the top-1 over the oracle the most ("misled by").

Selectors (one or more): --episode UID|EP_INDEX, --row R, --worst N, --best N, --random N
(ranked by --sort err|regret). Episode timelines are in `timeline`.

  python -m exp.offline_search.profile.explain RUN_OR_METHOD_DIR --cell pi05_spatial_inf --worst 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import common as C


def select(rc: C.RunCell, qc: C.QueryCell, episodes=(), rows=(), worst=0, best=0, rand=0, sort="err", seed=0) -> list[int]:
    """-> positions into the run arrays."""
    rr = rc.rows
    pos_of_row = {int(r): i for i, r in enumerate(rr)}
    sel: list[int] = []
    for e in episodes:
        e = str(e)
        ei = qc.uid_to_ep.get(e)
        if ei is None and e.lstrip("-").isdigit():
            ei = int(e)
        if ei is None:
            raise SystemExit(f"episode {e!r} not found in {qc.cell}")
        idx = np.flatnonzero(qc.ep[rr] == ei)
        sel += list(idx[np.argsort(qc.step[rr[idx]], kind="stable")])
    for r in rows:
        if int(r) not in pos_of_row:
            raise SystemExit(f"row {r} not in run {rc.method}/{rc.cell}")
        sel.append(pos_of_row[int(r)])
    key = rc["err"].astype(np.float64)
    if sort == "regret" and rc.has("oracle_err"):
        key = key - rc["oracle_err"]
    key = np.where(np.isfinite(key), key, -np.inf)
    if worst:
        sel += list(np.argsort(-key, kind="stable")[:worst])
    if best:
        sel += list(np.argsort(np.where(np.isfinite(key), key, np.inf), kind="stable")[:best])
    if rand:
        sel += list(np.random.default_rng(seed).choice(rc.n, min(rand, rc.n), replace=False))
    seen, out = set(), []
    for s in sel:
        if s not in seen:
            seen.add(int(s)); out.append(int(s))
    return out


def _scalar(v):
    v = np.asarray(v)
    if v.ndim == 0:
        return v.item()
    if v.size <= 6:
        return [x.item() if hasattr(x, "item") else x for x in v.ravel()]
    return f"<{v.dtype} {tuple(v.shape)}>"


def inspect(rc: C.RunCell, qc: C.QueryCell, pos: int, libs: dict, sigma, thr: float, covs: dict,
            b0cfg: dict | None, k: int = 10) -> dict:
    """libs / covs: {library name: Library | None} / {name: CoverageCache | None}; 'current' always present.
    The decision's own library is rc.lib_of(pos); the harness oracle_row always refers to library `current`."""
    row = int(rc.rows[pos])
    ei = int(qc.ep[row])
    ep = qc.episodes[ei]
    lname = rc.lib_of(pos)
    lib, cur = libs.get(lname), libs.get("current")
    d = {"pos": pos, "row": row, "ep": ei, "uid": ep.get("uid"), "task_id": int(qc.task_id[row]),
         "success": bool(qc.success[row]), "step": int(qc.step[row]), "num_steps": int(qc.num_steps[row]),
         "p": float(qc.p[row]), "third": C.THIRDS[int(qc.third[row])], "thr": thr, "library": lname,
         "library_in_store": lib is not None}
    for f in ("err", "oracle_err", "oracle_row", "regret", "confidence", "grip_mis", "phase_err", "used_synth",
              "t_query_us", "top1", "flip"):
        if rc.has(f):
            d[f] = _scalar(rc[f][pos])
    d["extras"] = {x: _scalar(rc[x][pos]) for x in rc.extras if not x.endswith("__len")}
    topk = np.asarray(rc["topk"][pos]) if rc.has("topk") else np.array([rc["top1"][pos]])
    sc = np.asarray(rc["topk_scores"][pos], np.float64) if rc.has("topk_scores") else np.full(topk.shape, np.nan)
    valid = topk >= 0
    topk, sc = topk[valid][:k], sc[valid][:k]
    d["gap12"] = float(sc[0] - sc[1]) if sc.size > 1 and np.isfinite(sc[:2]).all() else None
    a_ref = qc["a_inf"][row]
    lib_info = lib.describe_rows(topk) if lib is not None else [{"row": int(r)} for r in topk]
    errs = C.seg_err(lib["action"][topk], a_ref[None], sigma) if (lib is not None and topk.size) else np.full(topk.size, np.nan)
    orow = int(d.get("oracle_row", -1)) if "oracle_row" in d else -1
    same_as_cur = lname == "current"
    cand = []
    for j, (r, s_, e, info) in enumerate(zip(topk, sc, errs, lib_info)):
        cand.append({"rank": j + 1, **info, "score": float(s_), "err": float(e), "good": bool(e <= thr),
                     "is_oracle": same_as_cur and int(r) == orow})
    d["topk"] = cand
    d["oracle_rank_in_topk"] = next((c["rank"] for c in cand if c["is_oracle"]), None)
    if cur is not None and orow >= 0:
        d["oracle_info"] = cur.describe_rows([orow])[0]
    cc, cl = covs.get("current"), covs.get(lname)
    if cc is not None and cc.has(qc.cell):
        g = lambda f: cc.get(qc.cell, f)[row]  # noqa: E731
        d["coverage"] = {"n_cand": int(g("n_cand")), "n_good": int(g("n_good")), "oracle_err": float(g("oracle_err")),
                         "oracle_row": int(g("top_rows")[0]), "med_err": float(g("med_err")),
                         "rec_err": float(g("rec_err")), "rec_rank": int(g("rec_rank"))}
    if cl is not None and cl.has(qc.cell) and "top1" in d:
        tr = cl.get(qc.cell, "top_rows")[row]
        hit = np.flatnonzero(tr == int(d["top1"]))
        d.setdefault("coverage", {})["top1_err_rank"] = int(hit[0]) + 1 if hit.size else f">{tr.size}"
        d["coverage"]["n_good_in_" + lname] = int(cl.get(qc.cell, "n_good")[row])
        d["coverage"]["oracle_err_in_" + lname] = float(cl.get(qc.cell, "oracle_err")[row])
    # B0 per-field view: raw sims of the query vs the method's top-1 (its library), the oracle and the recorded
    # B0 pick (both library current)
    if qc.rows.has("key_v0"):
        who = []
        if lib is not None and topk.size:
            who.append(("top1", lib, int(topk[0])))
        if cur is not None and orow >= 0:
            who.append(("oracle", cur, orow))
        if cur is not None and qc.rows.has("rec_top1"):
            who.append(("b0_rec", cur, int(qc["rec_top1"][row])))
        b0 = {}
        for name, L_, r in who:
            f = C.b0_fields(qc, row, L_, np.array([r]))
            nz = C.b0_score(f, b0cfg) if b0cfg else {}
            e1 = float(C.seg_err(L_["action"][r], a_ref, sigma))
            b0[name] = {"row": r, "lib": L_.name, "err": e1, **{kk: float(vv[0]) for kk, vv in f.items()},
                        **{kk: float(vv[0]) for kk, vv in nz.items()}}
        if "top1" in b0 and "oracle" in b0 and b0cfg and "fused" in b0["top1"] and "fused" in b0["oracle"]:
            contrib = {fn: b0cfg["w"][fn] * (b0["top1"]["n_" + fn] - b0["oracle"]["n_" + fn])
                       for fn in ("vision_0", "vision_1", "robot_state")}
            b0["contrib_top1_minus_oracle"] = contrib
            if same_as_cur and b0["top1"]["row"] == b0["oracle"]["row"]:
                b0["verdict"] = "top1 is the oracle"
            elif sum(contrib.values()) > 0:  # B0 itself ranks top1 above the oracle: which field pushed it
                b0["misled_by"] = max(contrib, key=contrib.get)
                b0["verdict"] = f"B0 prefers top1; misled by {b0['misled_by']}"
            else:
                b0["verdict"] = "B0 would prefer the oracle (the method's own choice, not the B0 fields)"
        if b0:
            d["b0"] = b0
    return d


def render(d: dict, method: str, cell: str) -> str:
    L = []
    L.append(f"--- {method} | {cell} | row {d['row']} | ep {d['ep']} ({d['uid']}) task {d['task_id']} "
             f"{'SUCCESS' if d['success'] else 'failure'} | step {d['step']}/{d['num_steps']} p={d['p']:.2f} ({d['third']})"
             + ("" if d["library"] == "current" else f" | library {d['library']}"))
    orr = d.get("oracle_rank_in_topk")
    L.append(f"    err {C.fmt_val(d.get('err'))}  oracle {C.fmt_val(d.get('oracle_err'))} (row {d.get('oracle_row')}, "
             f"rank in top-k: {orr if orr else 'not in top-k'})  thr {C.fmt_val(d['thr'])}  conf {C.fmt_val(d.get('confidence'), '.4g')}  "
             f"gap12 {C.fmt_val(d.get('gap12'), '.4g')}  phase {C.fmt_val(d.get('phase_err'))}  grip_mis {C.fmt_val(d.get('grip_mis'))}"
             + ("  [synth]" if d.get("used_synth") else ""))
    if d.get("extras"):
        L.append("    extras: " + "  ".join(f"{k}={C.fmt_val(v, '.4g') if not isinstance(v, (list, str)) else v}" for k, v in d["extras"].items()))
    cv = d.get("coverage")
    if cv:
        if "n_cand" in cv:
            L.append(f"    coverage(current): n_cand {cv['n_cand']}  n_good {cv['n_good']}  oracle {cv['oracle_err']:.3f} "
                     f"(row {cv['oracle_row']})  median cand {cv['med_err']:.3f}  B0-rec err {C.fmt_val(cv['rec_err'])} "
                     f"rank {cv['rec_rank']}")
        extra = {k_: v for k_, v in cv.items() if k_ == "top1_err_rank" or k_.startswith(("n_good_in_", "oracle_err_in_"))}
        if extra:
            L.append("    coverage(method library): " + "  ".join(f"{k_} {C.fmt_val(v)}" for k_, v in extra.items()))
    rows = []
    for c in d["topk"]:
        rows.append({"k": c["rank"], "row": c["row"], "lib_ep": c.get("episode", c.get("traj")), "lib_step": c.get("step"),
                     "lib_p": c.get("progress"), "score": c["score"], "err": c["err"],
                     "flag": ("ORACLE " if c["is_oracle"] else "") + ("good" if c["good"] else "")})
    oi = d.get("oracle_info")
    if oi and not d.get("oracle_rank_in_topk"):
        where = "outside top-k" if d["library"] == "current" else "library current"
        rows.append({"k": "orc", "row": oi["row"], "lib_ep": oi.get("episode", oi.get("traj")), "lib_step": oi.get("step"),
                     "lib_p": oi.get("progress"), "score": None, "err": d.get("oracle_err"), "flag": f"ORACLE ({where})"})
    L.append(_indent(C.md_table(rows, [("k", "k"), ("row", "lib_row"), ("lib_ep", "lib_ep"), ("lib_step", "lib_step"),
                                       ("lib_p", "lib_p", ".2f"), ("score", "score", ".5g"), ("err", "err"), ("flag", "")])))
    b0 = d.get("b0")
    if b0:
        brow = []
        for name in ("top1", "oracle", "b0_rec"):
            if name in b0:
                x = b0[name]
                brow.append({"who": name if x.get("lib", "current") == "current" else f"{name}@{x['lib']}", "row": x["row"], "err": x["err"], "cos_v0": x.get("cos_v0"), "cos_v1": x.get("cos_v1"),
                             "rs_l2": x.get("rs_l2"), "n_v0": x.get("n_vision_0"), "n_v1": x.get("n_vision_1"),
                             "n_rs": x.get("n_robot_state"), "fused": x.get("fused")})
        L.append("    B0 per-field (raw cos on pooled keys, rs L2 on valid dims; n_* = 0.5(tanh(z)+1) with trace_dual mu/sigma):")
        L.append(_indent(C.md_table(brow, [("who", "who"), ("row", "row"), ("err", "err"), ("cos_v0", "cos_v0", ".5f"),
                                           ("cos_v1", "cos_v1", ".5f"), ("rs_l2", "rs_l2", ".4f"), ("n_v0", "n_v0", ".3f"),
                                           ("n_v1", "n_v1", ".3f"), ("n_rs", "n_rs", ".3f"), ("fused", "fused", ".4f")])))
        if b0.get("contrib_top1_minus_oracle"):
            c = b0["contrib_top1_minus_oracle"]
            L.append("    weighted B0 contribution top1 - oracle: " + "  ".join(f"{k}={v:+.4f}" for k, v in c.items())
                     + f"  -> {b0.get('verdict', '')}")
    return "\n".join(L)


def _indent(s: str, n: int = 4) -> str:
    return "\n".join(" " * n + ln for ln in s.splitlines())


PAR_MIN = 16      # below this many decisions a pool costs more than it saves


class _Ctx:
    """Everything inspect() needs for one (method dir, cell); built once per process."""

    def __init__(self, md, cell, root, floor_mode, k):
        self.rc = C.RunCell(md, cell)
        self.qc = C.QueryCell(root, cell)
        names = sorted(set(self.rc.lib_names) | {"current"})
        self.libs = {n: C.open_library(root, self.qc.ms, n) for n in names}
        self.covs = {n: C.load_coverage(root, n) for n in names}
        cur = self.libs["current"]
        self.sigma = C.lib_sigma(cur["action"]) if cur is not None else np.ones(C.N_ACT)
        self.floor = C.load_floor(root, self.qc.ms, self.sigma)
        self.b0cfg = C.b0_config(cell)
        self.floor_mode, self.k = floor_mode, k

    def inspect(self, pos: int) -> dict:
        row = int(self.rc.rows[pos])
        thr = float(self.floor.thr(self.qc.third[row:row + 1], self.floor_mode)[0]) if self.floor is not None else float("nan")
        return inspect(self.rc, self.qc, pos, self.libs, self.sigma, thr, self.covs, self.b0cfg, self.k)


def _chunk_job(a) -> list[dict]:
    md, cell, root, floor_mode, k, positions = a
    ctx = _Ctx(md, cell, root, floor_mode, k)
    return [ctx.inspect(p) for p in positions]


def run(run_path, cell, method=None, root=C.DEFAULT_ROOT, episodes=(), rows=(), worst=0, best=0, rand=0, sort="err",
        seed=0, k=10, floor_mode="third", out=None, quiet=False, procs=None) -> list[dict]:
    root = Path(root)
    mds = C.method_dirs(run_path, method)
    if len(mds) != 1:
        raise SystemExit(f"need exactly one method dir under {run_path} (got {[m.name for m in mds]}); use --method")
    md = mds[0]
    ctx = _Ctx(md, cell, root, floor_mode, k)
    rc, floor, b0cfg = ctx.rc, ctx.floor, ctx.b0cfg
    if not (episodes or rows or worst or best or rand):
        worst = 3
    sel = select(rc, ctx.qc, episodes, rows, worst, best, rand, sort, seed)
    P = C.DEFAULT_PROCS if procs is None else max(1, int(procs))
    if len(sel) >= PAR_MIN and P > 1:            # batch modes: contiguous chunks, order preserved
        n = min(P, -(-len(sel) // 4))
        bounds = np.linspace(0, len(sel), n + 1).astype(int)
        chunks = [sel[a:b] for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
        res = [d for part in C.pmap(_chunk_job, [(str(md), cell, str(root), floor_mode, k, c) for c in chunks], P) for d in part]
    else:
        res = [ctx.inspect(pos) for pos in sel]
    missing = [n for n in rc.lib_names if ctx.libs.get(n) is None]
    txt = [f"## explain {md.name} / {cell}  (library {rc.library}"
           + (f" [not in store: {missing} -> row details unavailable]" if missing else "")
           + f"; floor {'none' if floor is None else floor.source}; B0 cfg {'yes' if b0cfg else 'no'}; "
           "oracle = harness oracle over library current)"]
    txt += [render(d, md.name, cell) for d in res]
    s = "\n".join(txt)
    if not quiet:
        print(s)
    rdir = C.report_dir(root, "explain", f"{md.name}__{cell}", out)
    C.write_json(rdir / "explain.json", {"method": md.name, "cell": cell, "library": rc.library, "decisions": res})
    (rdir / "explain.txt").write_text(s + "\n")
    if not quiet:
        print(f"\n[explain] {len(res)} decisions -> {rdir}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="method dir (holds <cell>.npz) or run root (+ --method)")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--method", default=None)
    ap.add_argument("--root", default=str(C.DEFAULT_ROOT))
    ap.add_argument("--episode", action="append", default=[], help="episode uid or index (repeatable)")
    ap.add_argument("--row", action="append", type=int, default=[], help="query-store row (repeatable)")
    ap.add_argument("--worst", type=int, default=0)
    ap.add_argument("--best", type=int, default=0)
    ap.add_argument("--random", type=int, default=0)
    ap.add_argument("--sort", default="err", choices=["err", "regret"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--floor-mode", default="third", choices=["third", "overall"])
    ap.add_argument("--procs", type=int, default=C.DEFAULT_PROCS,
                    help=f"worker processes for batches of >= {PAR_MIN} decisions (default: all cores)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    run(a.run, a.cell, a.method, a.root, a.episode, a.row, a.worst, a.best, a.random, a.sort, a.seed, a.k, a.floor_mode, a.out,
        procs=a.procs)


if __name__ == "__main__":
    sys.exit(main())
