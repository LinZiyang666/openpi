"""R11 opus: offline value diagnostics of schedule knobs at the frozen arm settings (library only).

For each proposed arm (out/arm_grid.json) and the knob-off base, on the held-out library episodes:
  * capture: mean LOEO policy-vs-(GC_dist-corrected)-cache motion error at knob-called anchors / mean over
    eligible (non-guard) anchors (1.0 = what a random placement captures by construction);
  * uncovered high-error streaks: maximal runs of >= 3 consecutive anchors that are all "high error" (above the
    cell's 75th percentile over non-guard anchors) and none of which is a policy call, per episode. This is the
    one library-only proxy for the "long cache run drifts" hypothesis that motivates even spacing; the open-loop
    replay itself cannot show compounding, so treat it as a structural statistic, not an SR predictor;
  * cache-only stretch statistics (mean / p90 of the per-episode longest run of uncalled anchors).

  .venv/bin/python -m exp.offline_search.rounds.r11.opus.value_diag
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from exp.offline_search.rounds.r11.opus import ir_model as M
from exp.offline_search.rounds.r11.opus.arm_grid import MAKERS

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def streaks(ep, called, thr, min_len=3):
    hi = (ep.risk > thr) & ~called
    n, run = 0, 0
    for h in list(hi) + [False]:
        if h:
            run += 1
        else:
            n += run >= min_len
            run = 0
    return n


def diag(model, eps, knob, thr, reps=16):
    det = getattr(knob, "deterministic", True)
    reps = 1 if det else reps
    s = []
    for ep in eps:
        for r in range(reps):
            g, k = knob.calls(ep, r)
            s.append(streaks(ep, g | k, thr))
    return float(np.mean(s))


def main():
    grid = json.loads((OUT / "arm_grid.json").read_text())
    cells = sorted({r["cell"] for r in grid})
    res = {}
    for cell in cells:
        model, suite, size = cell.split("_")[0], cell.split("_")[1], int(cell.split("_")[2])
        eps = M.load_episodes(model, suite, size)
        R = np.concatenate([e.risk for e in eps]); G = np.concatenate([e.guard for e in eps])
        thr = float(np.percentile(R[~G], 75))
        rows = [dict(method="base", target=None, streaks=diag(model, eps, M.Knob(), thr))]
        for r in grid:
            if r["cell"] != cell:
                continue
            make = MAKERS[r["method"]][0]
            rows.append(dict(method=r["method"], target=r["target"], streaks=diag(model, eps, make(r["setting"]), thr),
                             capture=r["capture"], mean_max_run=r["mean_max_cache_run"], p90_max_run=r["p90_max_cache_run"]))
        res[cell] = dict(threshold=thr, rows=rows)
    (OUT / "value_diag.json").write_text(json.dumps(res, indent=1) + "\n")
    L = ["| cell-size | method | target | uncovered high-error streaks >=3 per episode | capture | mean / p90 longest cache run |",
         "|---|---|---|---|---|---|"]
    for cell, d in res.items():
        for r in d["rows"]:
            tgt = "" if r["target"] is None else f"{r['target']:.2f}"
            L.append(f"| {cell} | {r['method']} | {tgt} | "
                     f"{r['streaks']:.3f} | {r.get('capture', float('nan')):.2f} | "
                     f"{r.get('mean_max_run', float('nan')):.2f} / {r.get('p90_max_run', float('nan')):.0f} |")
    (OUT / "value_diag.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
