"""Write batch.json of family g2_pcacos (V4 / V5 variants, one factor at a time around the primary
lib=big fit=big k=32 st=1 kernel-8 T=.5 step-0 alignment on)."""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
M = "exp/offline_search/rounds/r02/g2_pcacos/method.py"
V4 = [
    {},                                          # primary: LbigFbig p32 st1 km8 T.5 align0
    {"lib": "current"},                          # MANDATORY current + fit on current (50-episode closed-loop group)
    {"lib": "current", "fit": "big"},            # current candidates, big-library basis ("borrowed")
    {"st": 0.5},
    {"st": 3.0},
    {"st": "pm"},                                # per model: pi0.5 0.5 / GR00T 3
    {"synth": "mean", "kk": 5},                  # plain mean-5
    {"k": 64},                                   # PCA-64
    {"align0": False},                           # step-0 alignment off
    {"lib": "current", "st": "pm"},              # current size, per-model st
]
V5 = [
    {"lam": 0.5},
    {"lam": 1.0},
    {"lib": "current", "lam": 0.5},              # current size (B F7: lam must stay <= .5 there)
]
spec = [{"method": f"{M}:PcaCosV4", "kwargs": kw, "family": "g2_pcacos", "cells": "all", "subsample": None,
         "allow_gpu_fit": False} for kw in V4]
spec += [{"method": f"{M}:PcaCosV5Fresh", "kwargs": kw, "family": "g2_pcacos", "cells": "all", "subsample": None,
          "allow_gpu_fit": False} for kw in V5]
(HERE / "batch.json").write_text(json.dumps(spec, indent=1) + "\n")
print(len(spec), "entries")
