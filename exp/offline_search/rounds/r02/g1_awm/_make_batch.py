"""Write batch.json for family g1_awm: one-factor-at-a-time variants around the primary
AWM_joint_big_fbig (lib big, fit big, joint 136-d, kref 8, full codes, lam .1, state_scale 1, early fit on, lam_c .5,
insurance off), plus the effect-decomposition variants (lib x fit_data) and the vision-only arm."""
import json
import pathlib

D = pathlib.Path(__file__).resolve().parent
M = "exp/offline_search/rounds/r02/g1_awm/awm.py:AWM"
GROOT = ["groot_spatial_inf", "groot_spatial_cache", "groot_l10_inf", "groot_l10_cache"]
variants = [
    ({}, "all"),                                                   # V1 primary: 10x candidates, 10x fit
    ({"lib": "current"}, "all"),                                   # V1 current + fit on current only (CL 50-ep group)
    ({"lib": "current", "fit_data": "big"}, "all"),                # V1 current candidates, BORROWED big-library fit
    ({"features": "vision"}, "all"),                               # V2 vision only (10x)
    ({"features": "vision", "lib": "current"}, "all"),             # V2 vision only, current + fit on current
    ({"features": "vision", "step0_joint": True}, "all"),          # V2 with the joint early-fit metric at step 0
    ({"kref": 5}, "all"),
    ({"codes": 32}, "all"),                                        # rank-32 codes
    ({"lam": 0.01}, "all"),
    ({"state_scale": 3.0}, GROOT),                                 # state block x3 (GR00T cells only)
    ({"early": False}, "all"),                                     # step 0 with the full-step metric
    ({"insure": True}, "all"),                                     # V3: norm-preserving mean + gripper hysteresis
    ({"lib": "current", "insure": True}, "all"),                   # V3 on current + fit on current
    ({"lam_c": 1.0}, "all"),                                       # fresh-regime continuity weight
    ({"lib": "current", "kref": 5}, "all"),                        # narrower kernel on the 50-ep library (ideation A's
    #                                                                current-library numbers were kref 5)
]
out = [{"method": M, "kwargs": kw, "family": "g1_awm", "cells": cells, "subsample": None, "allow_gpu_fit": False}
       for kw, cells in variants]
(D / "batch.json").write_text(json.dumps(out, indent=1) + "\n")
print(len(out), "entries")
