"""Frozen retrieval fits for deferred augmentation, per arm (coordinator tool).

full  = the cell's frozen A fit (R5/R6/R7 `sources()`), wrist = R7's frozen wrist bank (π0.5 only).
Pure-policy arms have no library: both libraries of the model/suite are listed. Every path carries its SHA256.
"""
import argparse
import hashlib
import json
from pathlib import Path

from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import sources

WRIST = Path("/home/weiland/trace_runs/os_closed_loop/r07_profile_bval1/fits")
STORE = "/home/weiland/trace_runs/offline_search_store"
DENSE = {"pi05": "bpool_cs", "groot": "bpool_all"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def spec(name, path, cache):
    path = str(path)
    if path not in cache:
        cache[path] = sha(path)
    return {"name": name, "path": path, "sha256": cache[path]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True, help="run-root arms.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, cache, out = sources(), {}, {}
    for arm in json.loads(Path(a.arms).read_text()):
        model, suite = arm["model"], arm["suite_short"]
        size = arm.get("r8", {}).get("library_size")
        sizes = [size] if size else [50, 500]
        cfg = {"store_root": STORE, "cell": arm["cell"], "full": []}
        for n in sizes:
            name = "current" if n == 50 else DENSE[model]
            cfg["full"].append(spec(name, src["{}_{}_{}".format(model, suite, n)]["source_artifact"], cache))
            if model == "pi05":
                short = "l10" if suite == "l10" else "sp"
                cfg.setdefault("wrist", []).append(spec(name, WRIST / "wrist_pi05_{}_{}.pkl".format(short, n), cache))
        out[arm["arm"]] = cfg
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"arms": len(out), "files": len(cache), "out": a.out}))


if __name__ == "__main__":
    main()
