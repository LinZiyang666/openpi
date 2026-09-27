"""Table of the full-cell vectorized references (ref/*.json from _ref_all.sh): per method, the 4 model x suite cells
(pi0.5-sp / pi0.5-l10 / GR00T-sp / GR00T-l10) for stale (cache arms, step >= 1), step 0 (inf arms), fresh (inf arms)."""
import glob
import json
import pathlib
import sys

D = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/home/weiland/trace_runs/offline_search_store/derived/r02/g2_pcacos/ref")
R = {}
for f in glob.glob(str(D / "*.json")):
    r = json.load(open(f))
    R[(r["method"], r["cell"])] = r
KEYS = ["pi05_spatial", "pi05_l10", "groot_spatial", "groot_l10"]
meths = sorted({m for m, _ in R})


def get(m, key, arm, reg, met):
    r = R.get((m, f"{key}_{arm}"))
    if r is None and m.startswith("V5fr") and arm == "cache":     # V5 == V4 on cache cells (every decision after a HIT)
        r = R.get((m.replace("V5fr", "V4pc").rsplit("_lam", 1)[0], f"{key}_{arm}"))
    try:
        return r[reg][met]
    except (TypeError, KeyError):
        return None


def fmt(v):
    return "  -  " if v is None else f"{v:.3f}"


for title, arm, reg in (("stale (cache arms, step>=1)", "cache", "stale"), ("step 0 (inf arms)", "inf", "step0"),
                        ("fresh (inf arms, step>=1)", "inf", "fresh")):
    print(f"\n### {title}: err | err_snap | AURC   (cells {' / '.join(KEYS)})")
    print("| method | err | err_snap | AURC |")
    print("|---|---|---|---|")
    for m in meths:
        e = [get(m, k, arm, reg, "err") for k in KEYS]
        if all(x is None for x in e):
            continue
        s = [get(m, k, arm, reg, "err_snap") for k in KEYS]
        a = [get(m, k, arm, reg, "aurc") for k in KEYS]
        print(f"| {m} | {' '.join(fmt(x) for x in e)} | {' '.join(fmt(x) for x in s)} | {' '.join(fmt(x) for x in a)} |")
