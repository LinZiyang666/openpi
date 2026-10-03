"""Per-decision ledger from standard-mode *server* decision logs (they carry uid, step, top1, rows, weights, extras).

Needed for GR00T, whose client per_step logs do not record the retrieved row.  RULE 1: records are admitted through
``iter_jsonl_discovery(uid_key="uid")`` which drops init >= 30 before decoding; only the accepted attempt (from the
discovery journal catalog) is kept.  Output: ``DERIVED/serverledger/<root>__<arm>.parquet``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .common import RUNS, DERIVED, PRICE, iter_jsonl_discovery, parse_uid, check_root


def extract(root, arm, accepted_attempts, overwrite=False):
    out = DERIVED / "serverledger" / f"{root}__{arm}.parquet"
    if out.exists() and not overwrite:
        return str(out)
    recs = []
    for path in sorted((check_root(RUNS / root) / "runs" / arm).glob("server_*/decisions*.jsonl")):
        for r in iter_jsonl_discovery(path, uid_key="uid"):
            if r.get("ev") != "dec":
                continue
            t, i = parse_uid(r["uid"])
            if int(r.get("attempt", 1)) != accepted_attempts.get((t, i), -1):
                continue
            rows = list(r.get("rows") or r.get("topk") or [])[:4]
            w = list(r.get("weights") or [])[:4]
            rows += [-1] * (4 - len(rows))
            w += [np.nan] * (4 - len(w))
            recs.append(dict(task=t, init=i, seq=int(r.get("step", -1)), vision=bool(r.get("vision")),
                             hit=bool(r.get("hit")), src=r.get("src"), r0=int(rows[0]), r1=int(rows[1]), r2=int(rows[2]),
                             r3=int(rows[3]), w0=float(w[0]), w1=float(w[1]), w2=float(w[2]), w3=float(w[3])))
    D = pd.DataFrame(recs)
    if D.empty:
        return None
    assert D["init"].max() < 30
    D = D.sort_values(["task", "init", "seq"]).drop_duplicates(["task", "init", "seq"], keep="last").reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    D.to_parquet(out)
    return str(out)


def episodes(path, outcomes, model):
    """Episode objects (triggers.Episode) with top-4 rows; owner cost from vision/hit."""
    from .triggers import Episode
    X = pd.read_parquet(path)
    v, m = PRICE[model]
    eps = []
    for (t, i), e in X.groupby(["task", "init"], sort=True):
        if (int(t), int(i)) not in outcomes:
            continue
        e = e.sort_values("seq")
        fresh = e.vision.values & (e.r0.values >= 0)
        cost = np.where(e.vision.values, v, 0.0) + np.where(e.vision.values & ~e.hit.values, m, 0.0)
        eps.append(Episode(str(path), int(t), int(i), bool(outcomes[(int(t), int(i))]), len(e), e.seq.values[fresh],
                           e[["r0", "r1", "r2", "r3"]].values[fresh].astype(int),
                           e[["w0", "w1", "w2", "w3"]].values[fresh].astype(float), cost))
    return eps


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="cache")
    ap.add_argument("--models", default="groot")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)
    from .replicates import load
    arms, outc = load()
    sel = arms[arms.family.isin(a.families.split(",")) & arms.model.isin(a.models.split(","))].drop_duplicates(["root", "arm"])
    jobs = []
    for r in sel.itertuples():
        o = outc[(outc.root == r.root) & (outc.arm == r.arm)]
        jobs.append((r.root, r.arm, {(int(x.task), int(x.init)): int(x.attempt) for x in o.itertuples()}))
    with ProcessPoolExecutor(a.workers) as ex:
        for (root, arm, _), res in zip(jobs, ex.map(extract, *zip(*jobs))):
            print(root, arm, res)


if __name__ == "__main__":
    main()
