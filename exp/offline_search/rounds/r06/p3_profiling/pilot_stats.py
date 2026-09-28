"""Episode-level within-init random-intercept ICC; clusters nested in task/cell/cohort.

Use success and episode-mean residual/cost summaries, not repeated terminal Y
copied onto anchors. Negative estimates are retained; no fictitious precision.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def icc(groups, strata=None):
    groups = list(groups)
    strata = [0] * len(groups) if strata is None else list(strata)
    if len(strata) != len(groups):
        raise ValueError("one task stratum required per init group")
    pairs = [(np.asarray(x, float), task) for x, task in zip(groups, strata) if len(x) >= 2]
    values = [x for x, _ in pairs]
    k, n = len(values), sum(map(len, values))
    tasks = set(task for _, task in pairs)
    between_df = k - len(tasks)
    if between_df < 1 or n <= k:
        return dict(icc=None, clusters=k, reason="replicated init groups and a within-task between-init contrast required")
    sizes = np.asarray(list(map(len, values)), float)
    task_n = {task: sum(len(x) for x, t in pairs if t == task) for task in tasks}
    means = {task: sum(x.sum() for x, t in pairs if t == task)/task_n[task] for task in tasks}
    between = sum(len(x)*(x.mean()-means[task])**2 for x, task in pairs)/between_df
    within = sum(((x-x.mean())**2).sum() for x in values)/(n-k)
    # Task fixed effects consume one between-cluster degree of freedom each.
    # This coefficient also handles unequal repeat counts within each task.
    n0 = (n - sum(sum(len(x)**2 for x, t in pairs if t == task)/task_n[task]
                  for task in tasks))/between_df
    denom = between+(n0-1)*within
    return dict(icc=float((between-within)/denom) if denom else None, clusters=k, episodes=n,
                mean_cluster_size=float(sizes.mean()), ms_between=between, ms_within=within,
                between_df=between_df, within_df=n-k, task_strata=len(tasks),
                scope="init random intercept with task fixed effects; replicated clusters only; binary Y approximate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=Path, required=True)
    ap.add_argument("--value", default="Y")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    groups = defaultdict(lambda: defaultdict(list))
    with a.episodes.open() as f:
        for r in csv.DictReader(f):
            # Generated arm IDs end in _r0/_r1/_r2. Remove ONLY the seed suffix.
            cell = r["arm"].rsplit("_r", 1)[0]
            groups[cell][r["task_id"], r["init"]].append(float(r[a.value]))
    result = {}
    for cell, group in groups.items():
        result[cell] = icc(group.values(), [task for task, init in group])
        result[cell]["task_means_removed"] = True
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
