"""Catalog every closed-loop arm of every run root and its discovery outcomes (inits 0-29 only).

Output: ``DERIVED/catalog/outcomes.parquet`` (one row per accepted discovery episode) and
``DERIVED/catalog/arms.parquet`` (one row per arm with a behavioural signature).

RULE 1: journal records with init >= 30 are dropped by ``iter_jsonl_discovery`` before decoding;
the two holdout roots are refused by ``check_root``.  Nothing else is read from a journal.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from .common import RUNS, DERIVED, forbidden, iter_jsonl_discovery, parse_uid, sha_text

NONBEHAVIOURAL = {"yaml", "matrix", "remote_yaml", "remote_matrix", "src_yaml", "src_sha256", "yaml_sha256",
                  "manifest", "r8", "arm", "priority", "note", "notes", "description"}


def fit_artifact(plugin_args):
    args = list(plugin_args or [])
    for i, a in enumerate(args):
        if a == "--os-fit-artifact" and i + 1 < len(args):
            return args[i + 1]
    return None


def flags(plugin_args):
    args = list(plugin_args or [])
    out = []
    skip = False
    for i, a in enumerate(args):
        if skip:
            skip = False
            continue
        if a in ("--os-root", "--os-fit-artifact"):
            skip = True
            continue
        out.append(a)
    return " ".join(out)


def lib_size(spec):
    kw = spec.get("kwargs") or {}
    for d in (kw, kw.get("base_kwargs") or {}, (kw.get("corrected_kwargs") or {})):
        lib = d.get("lib") if isinstance(d, dict) else None
        if lib == "current":
            return 50
        if lib in ("big", "bpool_all", "bpool_cs"):
            return 500
    r8 = spec.get("r8") or {}
    if r8.get("library_size"):
        return int(r8["library_size"])
    return None


def signature(spec):
    kw = json.loads(json.dumps(spec.get("kwargs") or {}, sort_keys=True))
    for k in ("seed", "replicate", "provenance", "calibration_path"):
        kw.pop(k, None)
    fa = fit_artifact(spec.get("plugin_args"))
    sig = dict(model=spec.get("model"), suite=spec.get("suite_short") or spec.get("suite"), method=spec.get("method"),
               kwargs=kw, flags=flags(spec.get("plugin_args")), fit=Path(fa).name if fa else None,
               client=spec.get("client_overrides"), full_model=spec.get("full_model"), mode=spec.get("mode"))
    return sha_text(json.dumps(sig, sort_keys=True, default=str)), sig


def one_arm(root, spec):
    arm = spec["arm"]
    jp = RUNS / root / "runs" / arm / "client" / "journal.jsonl"
    rows = []
    if not jp.exists():
        return arm, rows
    for r in iter_jsonl_discovery(jp):           # inits >= 30 never decoded
        if not r.get("accepted"):
            continue
        if r.get("phase") not in (None, "eval"):
            continue
        task, init = parse_uid(r["task_uid"])
        rows.append(dict(root=root, arm=arm, task=task, init=init, attempt=int(r.get("attempt", 1)),
                         status=r.get("status"), success=bool(r.get("success")), duration_s=r.get("duration_s")))
    return arm, rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)
    specs = []
    for root in sorted(p.name for p in RUNS.iterdir() if p.is_dir() and p.name.startswith("r0")):
        if forbidden(root):
            continue
        af = RUNS / root / "arms.json"
        if not af.exists():
            continue
        try:
            arms = json.loads(af.read_text())
        except ValueError:
            continue
        if isinstance(arms, dict):
            arms = list(arms.values())
        for s in arms:
            if isinstance(s, dict) and "arm" in s:
                specs.append((root, s))
    arm_rows, out_rows = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(one_arm, root, s) for root, s in specs]
        for (root, s), f in zip(specs, futs):
            arm, rows = f.result()
            key, sig = signature(s)
            arm_rows.append(dict(root=root, arm=arm, sig=key, model=s.get("model"),
                                 suite=s.get("suite_short") or s.get("suite"), lib=lib_size(s), method=s.get("method"),
                                 fit=sig["fit"], flags=sig["flags"], kwargs=json.dumps(sig["kwargs"], sort_keys=True),
                                 client=json.dumps(sig["client"], sort_keys=True), full_model=s.get("full_model"),
                                 variant=(s.get("r8") or {}).get("variant"), n_disc=len(rows),
                                 n_pairs=len({(r["task"], r["init"]) for r in rows})))
            out_rows += rows
    DERIVED.joinpath("catalog").mkdir(parents=True, exist_ok=True)
    arms = pd.DataFrame(arm_rows)
    outc = pd.DataFrame(out_rows)
    assert outc.empty or outc["init"].max() < 30, "RULE 1 violated"
    arms.to_parquet(DERIVED / "catalog" / "arms.parquet")
    outc.to_parquet(DERIVED / "catalog" / "outcomes.parquet")
    print(len(arms), "arms;", len(outc), "discovery episodes;", arms.n_disc.gt(0).sum(), "arms with discovery data")


if __name__ == "__main__":
    main()
