"""Writes batch.json of family G3 (stand-in base; see README.md for instantiating the wrappers over G1 / G2)."""
import json
import pathlib

D = "exp/offline_search/rounds/r02/g3_recovery"
BASE = f"{D}/standins.py:VZSKernel"
W = f"{D}/wrappers.py"


def entry(method, kwargs):
    return {"method": method, "kwargs": kwargs, "family": "g3_recovery", "cells": "all", "subsample": None,
            "allow_gpu_fit": False}


spec = []
for lib in ("current", "big"):
    bk = {"lib": lib}
    spec.append(entry(BASE, bk))                                                              # the base itself
    spec.append(entry(f"{W}:StuckRecovery", {"base": BASE, "base_kwargs": bk, "recover": False}))   # blend only
    spec.append(entry(f"{W}:StuckRecovery", {"base": BASE, "base_kwargs": bk, "recover": False, "anchor": "self"}))
    spec.append(entry(f"{W}:StuckRecovery", {"base": BASE, "base_kwargs": bk}))                     # full V6
    spec.append(entry(f"{W}:StuckRecovery", {"base": BASE, "base_kwargs": bk, "ot_still": True}))   # CL3 candidate
    spec.append(entry(f"{W}:DriftCalibratedConfidence", {"base": BASE, "base_kwargs": bk, "calib": "iso"}))
    spec.append(entry(f"{W}:DriftCalibratedConfidence", {"base": BASE, "base_kwargs": bk, "calib": "bin10"}))
(pathlib.Path(__file__).parent / "batch.json").write_text(json.dumps(spec, indent=1) + "\n")
print(len(spec), "entries")
