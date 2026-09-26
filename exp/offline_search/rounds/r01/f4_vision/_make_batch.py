"""Write batch.json for family f4_vision (M7 cascade_biglib variants + M9(c) vision_zsum variants)."""
import json
import pathlib

D = pathlib.Path(__file__).resolve().parent
M7 = "exp/offline_search/rounds/r01/f4_vision/m7_cascade.py:CascadeBigLib"
M9 = "exp/offline_search/rounds/r01/f4_vision/m9_vision_zsum.py:VisionZSum"
m7 = [("b0fused", "raw"), ("v1only", "raw"), ("none", "raw"), ("v1only", "pca64"), ("v1only", "pca32"),
      ("v1only", "pca128"), ("both", "raw"), ("both", "pca32"), ("both", "pca32", 64)]
# the raw-key big-library twin (lib=big keys=raw center=tm fields=both st=1 top1) is smoke-tested but left out: it
# equals pca128 within 0.01 err in the smoke and costs 60-90 ms/query + 80 s fit (memory-bandwidth bound, ~2 GB of
# keys streamed per l10 query)
m9 = [dict(lib="big", keys="pca128", center="tm", fields="both", st=1.0, synth="top1"),   # cheap twin (128 floats/field)
      dict(lib="big", keys="pca128", center="tm", fields="both", st=1.0, synth="med3"),
      dict(lib="big", keys="pca128", center="tm", fields="both", st=0.0, synth="top1"),   # vision only
      dict(lib="current", keys="raw", center="tm", fields="both", st=1.0, synth="top1")]  # B0-cost comparator
out = [{"method": M7, "kwargs": {"rerank": v[0], "keys": v[1], **({"M": v[2]} if len(v) > 2 else {})},
        "family": "f4_vision", "cells": "all", "subsample": None, "allow_gpu_fit": False} for v in m7]
out += [{"method": M9, "kwargs": kw, "family": "f4_vision", "cells": "all", "subsample": None, "allow_gpu_fit": False}
        for kw in m9]
(D / "batch.json").write_text(json.dumps(out, indent=1) + "\n")
print(len(out), "entries")
