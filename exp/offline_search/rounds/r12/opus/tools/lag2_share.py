"""R12 opus: library-replay counterpart of the closed-loop (R6 B-val) lag-2 consistency statistics: share of looks
whose |W - WL(lag 2)| + gripper-sign disagreement <= tau, and dF(W vs F) on those looks. Library only."""
import json
import numpy as np
from exp.offline_search.rounds.r12.opus.tools import common as cm
from exp.offline_search.rounds.r12.opus.tools.analyze_replay import load

out = {}
for m in ("pi05", "groot"):
    for s in ("l10", "spatial"):
        for n in (50, 500):
            Z = load(m, s, n)
            sig = Z["sig"]
            look = (Z["step"] % 2 == 0) & (Z["step"] > 0)
            sW, sWL, sF = Z["W|served"][look], Z["WL|served"][look], Z["F|served"][look]
            sc = cm.motion_err(sW, sWL, sig) + np.mean(np.sign(sW[:, :5, 6]) != np.sign(sWL[:, :5, 6]), axis=1)
            dF = cm.motion_err(sF, sW, sig)
            r = {str(t): dict(half=float((sc <= t).mean()), dF_half=float(dF[sc <= t].mean())) for t in (0.1, 0.15, 0.2, 0.3)}
            out[f"{m}_{s}_{n}"] = r
            print(f"{m}_{s}_{n}: " + "  ".join(f"tau {t}: half {d['half']:.3f} dF {d['dF_half']:.3f}" for t, d in r.items()))
cm.save_json(out, cm.OUT / "lag2_share_library.json")
