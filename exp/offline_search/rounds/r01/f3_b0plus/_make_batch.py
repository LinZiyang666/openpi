"""Writes batch.json (every F3 variant of the R1 full round)."""
import json
import pathlib

P = "exp/offline_search/rounds/r01/f3_b0plus/method.py"
V = [
    ("B0TopkConsensus", {"k": 5, "synth": "mean"}),
    ("B0TopkConsensus", {"k": 5, "synth": "kernel"}),
    ("B0TopkConsensus", {"k": 5, "synth": "med"}),
    ("B0TopkConsensus", {"k": 3, "synth": "med"}),
    ("B0TopkConsensus", {"k": 8, "synth": "mean"}),
    ("B0ShortlistContRerank", {"lam": 1.0, "k": 5}),
    ("B0ShortlistContRerank", {"lam": 2.0, "k": 3}),
    ("B0ShortlistContRerank", {"lam": 2.0, "k": 5}),
    ("B0ShortlistContRerank", {"lam": "inf", "k": 3}),
    ("ConsistencyConfidence", {"base": "b0", "weights": "equal", "veto": False}),
    ("ConsistencyConfidence", {"base": "b0", "weights": "split", "veto": False}),
    ("ConsistencyConfidence", {"base": "b0", "weights": "split", "veto": True, "tau": 0.3}),
    ("ConsistencyConfidence", {"base": "m4k5med", "weights": "split", "veto": False}),
    ("ConsistencyConfidence", {"base": "m5l2k3", "weights": "split", "veto": False}),
    ("B0BigLibCons", {"synth": "top1"}),
    ("B0BigLibCons", {"k": 5, "synth": "mean"}),
]
spec = [{"method": f"{P}:{c}", "kwargs": kw, "family": "f3_b0plus", "cells": "all", "subsample": None,
         "allow_gpu_fit": False} for c, kw in V]
pathlib.Path(__file__).with_name("batch.json").write_text(json.dumps(spec, indent=1) + "\n")
print(len(spec), "entries")
