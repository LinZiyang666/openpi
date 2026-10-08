"""R12 opus: which cues available at a one-camera look predict a two-camera guard call that follows a HIT?
(closed-loop dev looks; AUC of each cue; library-row rs for the anchor-continuation cue)."""
import glob
from collections import defaultdict
import numpy as np
from scipy.stats import rankdata
from exp.offline_search.rounds.r10.analysis_opus.tools import core
from exp.offline_search.rounds.r12.opus.tools import common as cm
from exp.offline_search.rounds.r12.opus.tools.dev_eval import load


def auc(s, y):
    r = rankdata(s); n1 = y.sum(); n0 = (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


out = {}
for p in sorted(glob.glob(str(cm.OUT / "dev" / "*.npz"))):
    d, meta = load(p)
    C = cm.load_cell(meta["model"], meta["suite"])
    S = cm.stage_arrays(meta["model"], meta["suite"])
    rsL = C["rs"]
    by = defaultdict(list)
    for i, u in enumerate(d["uid"]):
        by[u].append(i)
    F = defaultdict(list)
    for u, idx in by.items():
        idx = sorted(idx, key=lambda i: d["step"][i])
        for a, b in zip(idx[:-1], idx[1:]):
            if not d["hit"][a]:
                continue                      # only transitions after a HIT
            pp = float(d["F_top1_prog"][a])
            r0 = int(d["F_top1"][a])
            r2 = cm.advance(S["next"], np.array([r0]), 2)[0]
            F["dW"].append(-(float(d["W_top1_prog"][b]) - pp) * max(int(d["W_top1_eplen"][b]) - 1, 1))
            F["dWL"].append(-(float(d["WL2_top1_prog"][b]) - pp) * max(int(d["WL2_top1_eplen"][b]) - 1, 1))
            F["dmin"].append(max(F["dW"][-1], F["dWL"][-1]))
            F["drs"].append(-float(d["drs"][b]))
            F["dP1"].append(-float(d["dP1"][b]))
            F["W_d1"].append(float(d["W_d1"][b]))
            F["y"].append(not bool(d["hit"][b]))
    y = np.array(F.pop("y"))
    res = {k: auc(np.nan_to_num(np.array(v, float)), y) for k, v in F.items()}
    key = f"{meta['model']}_{meta['suite']}_{meta['size']}"
    out[key] = dict(n=int(len(y)), calls=float(y.mean()), auc=res)
    print(key, len(y), round(y.mean(), 3), {k: round(v, 3) for k, v in res.items()})
cm.save_json(out, cm.OUT / "stall_cues.json")
