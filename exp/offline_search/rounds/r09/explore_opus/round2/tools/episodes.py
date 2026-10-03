"""Standard-mode per-decision ledger for selected arms (discovery inits 0-29 only).

Reads ``runs/<arm>/client/per_step.jsonl`` (one record per decision plus one ``client_timing`` record per
episode).  RULE 1: ``iter_jsonl_discovery`` drops every record whose task_uid init is >= 30 before decoding.

Output per arm: ``DERIVED/episodes/<root>__<arm>.parquet`` with one row per decision:
task, init, seq (decision order), step (control index), kind (0 blind, 1 look/hit, 2 call/miss), judge, row
and one row per episode in ``..._ep.parquet``: controls, decisions, looks, calls, success, termination.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .common import RUNS, DERIVED, iter_jsonl_discovery, parse_uid, check_root

JUDGE_CODES = {}


def judge_code(j):
    if j is None:
        return -1
    if j == "blind":
        return 0
    if j in ("guard_only", "always", "periodic"):
        return 1
    if isinstance(j, str) and j.startswith("force:"):
        try:
            return 100 + int(j.split(":")[1])
        except ValueError:
            return 99
    return 98


def extract(root, arm, overwrite=False):
    out = DERIVED / "episodes" / f"{root}__{arm}.parquet"
    if out.exists() and not overwrite:
        return str(out), None
    path = check_root(RUNS / root) / "runs" / arm / "client" / "per_step.jsonl"
    if not path.exists():
        return None, "missing per_step"
    dec, eps = [], []
    for r in iter_jsonl_discovery(path):
        if r.get("accepted") is False:
            continue
        task, init = parse_uid(r["task_uid"])
        if r.get("_kind") == "client_timing":
            eps.append(dict(task=task, init=init, attempt=int(r.get("attempt", 1)), controls=r.get("steps"),
                            infers=r.get("infers"), termination=r.get("termination_reason"),
                            success=bool(r.get("success"))))
            continue
        if "hit_type" not in r:
            continue
        fo = (r.get("factor_outputs") or {}).get("osplug") or {}
        ht = r.get("hit_type")
        kind = 2 if ht == "MISS" else (1 if r.get("searched") else 0)
        dec.append(dict(task=task, init=init, attempt=int(r.get("attempt", 1)), step=int(r.get("step_idx", -1)),
                        kind=kind, judge=judge_code(fo.get("os_judge")), row=int(fo.get("os_row", -1) if fo.get("os_row") is not None else -1),
                        conf=float(fo["os_conf"]) if fo.get("os_conf") is not None else np.nan,
                        success=bool(r.get("success"))))
    D = pd.DataFrame(dec)
    E = pd.DataFrame(eps)
    if D.empty:
        return None, "empty"
    assert D["init"].max() < 30
    # keep the last accepted attempt per pair
    last = D.groupby(["task", "init"]).attempt.transform("max")
    D = D[D.attempt == last].sort_values(["task", "init", "step"]).reset_index(drop=True)
    D["seq"] = D.groupby(["task", "init"]).cumcount()
    if not E.empty:
        lastE = E.groupby(["task", "init"]).attempt.transform("max")
        E = E[E.attempt == lastE].drop_duplicates(["task", "init"], keep="last")
    agg = D.groupby(["task", "init"]).agg(decisions=("seq", "size"), looks=("kind", lambda k: int((k >= 1).sum())),
                                          calls=("kind", lambda k: int((k == 2).sum())), success=("success", "last"),
                                          last_step=("step", "max")).reset_index()
    if not E.empty:
        agg = agg.merge(E[["task", "init", "controls", "infers", "termination"]], on=["task", "init"], how="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    D.to_parquet(out)
    agg.to_parquet(out.with_name(out.stem + "_ep.parquet"))
    return str(out), None


def selected_arms():
    from .replicates import family
    arms = pd.read_parquet(DERIVED / "catalog" / "arms.parquet")
    arms = arms[arms.n_pairs >= 290].copy()
    arms["family"] = [family(r) for r in arms.itertuples()]
    return arms[arms.family.notna()]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--families", default="")
    a = ap.parse_args(argv)
    arms = selected_arms()
    if a.families:
        arms = arms[arms.family.isin(a.families.split(","))]
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(extract, r.root, r.arm, a.overwrite): (r.root, r.arm) for r in arms.itertuples()}
        for f, (root, arm) in futs.items():
            path, err = f.result()
            if err:
                print("skip", root, arm, err)
    print("done", len(arms))


if __name__ == "__main__":
    main()
