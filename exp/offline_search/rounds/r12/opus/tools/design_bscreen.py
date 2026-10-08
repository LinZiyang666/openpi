"""B-screen (pure-cache camera ablation, K=10/task) per-task success, paired by init. B-pool only."""
import glob, json, os, sys
from collections import defaultdict
import numpy as np

P = "/home/weiland/projects/openpi/exp/offline_search/rounds/r12/ablate/pilots"
CELLS = ["pi05_l10_50", "pi05_spatial_50", "pi05_spatial_200", "groot_spatial_50", "groot_l10_50"]
ARMS = ["both", "wrist_sub", "wrist_refit", "agent_sub", "agent_refit"]


def journal(run):
    comp = {}
    for line in open(run + "/client/journal.jsonl"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("accepted") and r.get("status") in ("done", "failed") and not r.get("error"):
            t, i = r["task_uid"].split(":")[-2:]
            comp[(int(t), int(i))] = bool(r.get("success"))
    return comp


out = {}
for cell in CELLS:
    root = f"{P}/r12_ablate_{cell}"
    res = {}
    for a in ARMS:
        run = f"{root}/runs/r12_ablate_{cell}_{a}"
        if os.path.exists(run + "/client/journal.jsonl"):
            res[a] = journal(run)
    tasks = sorted({t for (t, i) in res["both"]})
    print(f"\n== {cell}")
    print("task  " + " ".join(f"{a:>11s}" for a in res) + "   d(wsub-both) d(wref-both)")
    rows = {}
    for t in tasks:
        line = f"{t:4d}  "
        sr = {}
        for a in res:
            v = [res[a][k] for k in res[a] if k[0] == t]
            sr[a] = (np.mean(v), len(v))
            line += f"{sr[a][0]:7.2f}({sr[a][1]:2d})"
        common = [k for k in res["both"] if k[0] == t and k in res["wrist_sub"]]
        d1 = np.mean([res["wrist_sub"][k] - res["both"][k] for k in common])
        common2 = [k for k in res["both"] if k[0] == t and k in res.get("wrist_refit", {})]
        d2 = np.mean([res["wrist_refit"][k] - res["both"][k] for k in common2]) if common2 else float("nan")
        line += f"   {d1:+.2f}  {d2:+.2f}"
        print(line)
        rows[t] = {a: [int(res[a][k]) for k in sorted(res[a]) if k[0] == t] for a in res}
        rows[t]["inits"] = [k[1] for k in sorted(res["both"]) if k[0] == t]
    tot = {a: np.mean(list(res[a].values())) for a in res}
    print("all   " + " ".join(f"{tot[a]:11.3f}" for a in res))
    out[cell] = {str(t): rows[t] for t in rows}
json.dump(out, open("/home/weiland/projects/openpi/exp/offline_search/rounds/r12/opus/out/design/bscreen_pertask.json", "w"))
