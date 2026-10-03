"""Per-decision ledger from the R8 debug collection's *server* records (discovery inits 0-29 only).

The debug server records carry task_uid, so ``iter_jsonl_discovery`` drops init >= 30 before decoding (RULE 1).
Kept per decision: task, init, seq, src, vision, hit, top-4 rows + weights, d1, conf, owner_cost.
Output: ``DERIVED/r8ledger/<arm>.parquet``.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .common import RUNS, DERIVED, iter_jsonl_discovery, parse_uid

R8 = RUNS / "r08_main"


def _list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return []
    return []


def extract(arm, overwrite=False):
    out = DERIVED / "r8ledger" / f"{arm}.parquet"
    if out.exists() and not overwrite:
        return str(out)
    recs = []
    for path in sorted((R8 / "runs" / arm / "debug").glob("server_*/decisions*.jsonl")):
        for r in iter_jsonl_discovery(path):
            task, init = parse_uid(r["task_uid"])
            rows = _list(r.get("rows"))[:4]
            w = _list(r.get("weights"))[:4]
            rows += [-1] * (4 - len(rows))
            w += [np.nan] * (4 - len(w))
            recs.append(dict(task=task, init=init, attempt=int(r.get("attempt", 1)), episode_key=r.get("episode_key"),
                             seq=int(r.get("decision_seq", -1)), src=r.get("src"), vision=bool(r.get("vision")),
                             hit=r.get("hit"), d1=r.get("d1"), conf=r.get("conf"), cost=r.get("owner_cost"),
                             camera=r.get("camera_mode"), r0=rows[0], r1=rows[1], r2=rows[2], r3=rows[3],
                             w0=w[0], w1=w[1], w2=w[2], w3=w[3], lib=r.get("lib")))
    D = pd.DataFrame(recs)
    if D.empty:
        return None
    assert D["init"].max() < 30
    last = D.groupby(["task", "init"]).attempt.transform("max")
    D = D[D.attempt == last].sort_values(["task", "init", "seq"]).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    for c in ("d1", "conf", "cost", "w0", "w1", "w2", "w3"):
        D[c] = pd.to_numeric(D[c], errors="coerce")
    D.to_parquet(out)
    return str(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)
    arms = a.arms.split(",") if a.arms else sorted(p.name for p in (R8 / "runs").iterdir() if p.is_dir())
    with ProcessPoolExecutor(a.workers) as ex:
        for arm, res in zip(arms, ex.map(extract, arms)):
            print(arm, res)


if __name__ == "__main__":
    main()
