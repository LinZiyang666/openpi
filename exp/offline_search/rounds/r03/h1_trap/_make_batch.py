"""Writes batch.json (the offline full round of H1: every variant on all 8 cells) and prints the variant table.
`python3 _make_batch.py` from anywhere. VARIANTS is the single source of the variant list (smoke drivers import it).
"""
from __future__ import annotations

import json
import pathlib

METHOD = "exp/offline_search/rounds/r03/h1_trap/awm3.py:AWM3"
FAMILY = "h1_trap"

# (short tag, kwargs, group) -- names are produced by AWM3 itself (AWM3_joint_<cur|big>_<fcur|fbig>_kr5[_...]).
VARIANTS = [
    # non-borrowing (current candidates, current fit) -- the CL2 twin family
    ("ref",        {},                                                                     "current"),
    ("ridge1",     {"ridge_main": 1.0},                                                    "current"),
    ("sb05",       {"shared_beta": 0.5},                                                   "current (offline-only ablation)"),
    ("gc",         {"grip_commit": True},                                                  "current"),
    ("tg",         {"term_guard": True},                                                   "current"),
    ("gc_tg",      {"grip_commit": True, "term_guard": True},                              "current"),
    ("gcs_tg",     {"grip_commit": True, "grip_confirm": "sign", "term_guard": True},      "current"),
    # borrowed big-library information (current candidates; big PCA basis + standardization; blended S_w)
    ("a1",         {"prior_alpha": 1.0},                                                   "borrowed"),
    ("a05",        {"prior_alpha": 0.5},                                                   "borrowed"),
    ("a0",         {"prior_alpha": 0.0},                                                   "borrowed (basis only)"),
    ("a1_gc",      {"prior_alpha": 1.0, "grip_commit": True},                              "borrowed"),
    ("a1_tg",      {"prior_alpha": 1.0, "term_guard": True},                               "borrowed"),
    ("a1_gc_tg",   {"prior_alpha": 1.0, "grip_commit": True, "term_guard": True},          "borrowed"),
    ("a05_gc",     {"prior_alpha": 0.5, "grip_commit": True},                              "borrowed"),
    ("a05_tg",     {"prior_alpha": 0.5, "term_guard": True},                               "borrowed"),
    ("a05_gc_tg",  {"prior_alpha": 0.5, "grip_commit": True, "term_guard": True},          "borrowed"),
    ("a1_gcm_tg",  {"prior_alpha": 1.0, "grip_commit": True, "grip_class_mean": True, "term_guard": True}, "borrowed (class-mean variant)"),
    ("a1_gc_tg1",  {"prior_alpha": 1.0, "grip_commit": True, "term_guard": True, "term_rows": 1}, "borrowed (last-1 variant)"),
    ("a1_gct06_tg", {"prior_alpha": 1.0, "grip_commit": True, "grip_thr": 0.6, "term_guard": True}, "borrowed (vote thr .6 variant)"),
    ("a1_gc_tgp",  {"prior_alpha": 1.0, "grip_commit": True, "term_guard": True, "term_gate": "late"}, "borrowed (progress-only gate variant)"),
    # 10x library (500 episodes candidates + fit)
    ("big",        {"lib": "big"},                                                         "10x"),
    ("big_gc_tg",  {"lib": "big", "grip_commit": True, "term_guard": True},                "10x"),
]


def rows():
    return [{"method": METHOD, "kwargs": kw, "family": FAMILY, "cells": "all", "subsample": None,
             "allow_gpu_fit": False} for _, kw, _ in VARIANTS]


def names():
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))
    from exp.offline_search.rounds.r03.h1_trap.awm3 import AWM3

    return [(tag, AWM3(**kw).name, kw, grp) for tag, kw, grp in VARIANTS]


if __name__ == "__main__":
    here = pathlib.Path(__file__).resolve().parent
    (here / "batch.json").write_text(json.dumps(rows(), indent=1) + "\n")
    for tag, name, kw, grp in names():
        print(f"{tag:12s} {name:48s} {grp:32s} {json.dumps(kw)}")
    print(f"wrote {here / 'batch.json'} ({len(VARIANTS)} variants)")
