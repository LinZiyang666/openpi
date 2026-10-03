"""R8 arm table: accepted SR and owner IR (pooled over decisions, from the live per-decision owner_cost) per arm.

owner_cost is written by the server observer with the arm's price table (pi05 .152/.848, GR00T .148/.852,
measured wrist look .0646 / completion .0502). Pure-policy arms are library-independent.
"""
import argparse
import glob
import json
import os

import numpy as np


def arm_row(run, arm):
    root = os.path.join(run, "runs", arm)
    acc = {}
    for line in open(os.path.join(root, "client", "journal.jsonl")):
        r = json.loads(line)
        if r.get("accepted"):
            acc[r["task_uid"]] = bool(r.get("success"))
    cost, n, vis, pol = 0.0, 0, 0, 0
    for path in glob.glob(os.path.join(root, "debug", "server_*", "decisions*.jsonl")):
        with open(path) as f:
            for line in f:
                if not line.endswith("\n"):
                    continue  # truncated final line of a killed process
                r = json.loads(line)
                if r.get("task_uid") not in acc:
                    continue
                c = r.get("owner_cost")
                if c is None:
                    continue
                cost += float(c)
                n += 1
                vis += int(bool(r.get("vision")))
                pol += int(r.get("policy_calls") or 0) > 0
    return dict(arm=arm, episodes=len(acc), sr=float(np.mean(list(acc.values()))), decisions=n,
                owner_ir=cost / n if n else None, vision_share=vis / n if n else None, call_share=pol / n if n else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = json.load(open(os.path.join(a.run_root, "arms.json")))
    rows = []
    for spec in arms:
        row = arm_row(a.run_root, spec["arm"])
        row.update(model=spec["model"], suite=spec["suite_short"], variant=spec["r8"]["variant"],
                   library=spec["r8"].get("library_size"), priority=spec["r8"]["priority"])
        rows.append(row)
        print(json.dumps(row))
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
