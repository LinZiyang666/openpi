"""Write batch_v6.json: G3's V6 StuckRecovery (ot_still=True, G3's recommended CL3 detector) around the two AWM
configs the coordinator named -- current + fit on current with kref 5 (the 50-ep closed-loop group) and the 10x
primary -- each plain and with V3 insurance (CL3 = CL2 + V3 + V6). Family g3_recovery (the wrapper's)."""
import json
import pathlib

D = pathlib.Path(__file__).resolve().parent
W = "exp/offline_search/rounds/r02/g3_recovery/wrappers.py:StuckRecovery"
B = "exp/offline_search/rounds/r02/g1_awm/awm.py:AWM"
bases = [{"lib": "current", "kref": 5}, {},
         {"lib": "current", "kref": 5, "insure": True}, {"insure": True}]
out = [{"method": W, "kwargs": {"base": B, "base_kwargs": bk, "ot_still": True}, "family": "g3_recovery",
        "cells": "all", "subsample": None, "allow_gpu_fit": False} for bk in bases]
(D / "batch_v6.json").write_text(json.dumps(out, indent=1) + "\n")
print(len(out), "entries")
