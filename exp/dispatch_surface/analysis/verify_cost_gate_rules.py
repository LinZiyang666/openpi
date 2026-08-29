"""Cost-gate rule numbers under the Review Authority ruling (section 9).

Run this to reproduce every figure in section 10 of
logs/dispatch_surface_cost_axis_change.md.

Two corrections against the proposal's own figures:

  1. estimand (ruling 9.4): the cost of an arm is the decision-weighted
     ratio-of-sums, sum_d c(h_d) / N_decisions, recomputed inside every
     bootstrap replicate from the resampled init clusters. The proposal
     averaged per-episode means over episodes, which estimates the cost of a
     random EPISODE, not of a random DECISION -- different quantities once
     episodes differ in length.
  2. boundary size (ruling 9.3): report the release probability at the
     non-inferiority boundary (true +5%), which is what decides whether a rule
     controls its type-I error. The proposal only tabulated +8% and +15%.
"""

import collections
import itertools
import json
import statistics
from math import erf, sqrt

import numpy as np

STAGE1, STAGE2, STAGE3 = 10.26, 27.69, 29.57
TRIALS_PER_TASK = 30
N_BOOT = 2000
SEED = 20260827
SRC = "exp/gate_threshold_pareto/data/eval/libero_spatial/per_step.jsonl"


def unit_cost(hit_type, start_t):
    if hit_type == "FULL_HIT":
        return STAGE1
    if hit_type == "WARM_START":
        return STAGE1 + STAGE2 + float(start_t if start_t is not None else 1.0) * STAGE3
    return STAGE1 + STAGE2 + STAGE3


# Keep the SUM and the COUNT separately -- the ratio must be formed after
# resampling, never before.
cost_sum = collections.defaultdict(float)
n_dec = collections.defaultdict(int)
seen_timing = set()
for line in open(SRC):
    row = json.loads(line)
    if row.get("_kind") == "client_timing":
        seen_timing.add(row["task_uid"])
        continue
    uid = row.get("task_uid")
    if uid is None:
        continue
    cost_sum[uid] += unit_cost(row.get("hit_type"), row.get("start_t"))
    n_dec[uid] += 1

by_arm_sum = collections.defaultdict(dict)
by_arm_n = collections.defaultdict(dict)
for uid in cost_sum:
    if uid not in seen_timing:
        continue
    arm, _phase, t, i = uid.split(":")
    by_arm_sum[arm][(int(t), int(i))] = cost_sum[uid]
    by_arm_n[arm][(int(t), int(i))] = n_dec[uid]

arms = sorted(by_arm_sum)
common = sorted(set.intersection(*(set(d) for d in by_arm_sum.values())))
tasks = sorted({t for t, _ in common})
per_task = {t: sorted(i for (tt, i) in common if tt == t)[:TRIALS_PER_TASK]
            for t in tasks}
cells = [(t, i) for t in tasks for i in per_task[t]]
idx_by_task = {t: np.array([k for k, (tt, _) in enumerate(cells) if tt == t])
               for t in tasks}
rng = np.random.default_rng(SEED)
print(f"{len(arms)} arms, {len(cells)} episodes/arm")

# Episode length really does vary, which is why the two estimands differ.
lens = [by_arm_n[arms[0]][c] for c in cells]
print(f"decisions per episode: min={min(lens)} median={statistics.median(lens):.0f} "
      f"max={max(lens)}  (equal-weight vs decision-weight only coincide if these "
      f"were constant)")


def se_both(a: str, b: str) -> tuple[float, float, float, float]:
    """Bootstrap SE of the relative cost difference under both estimands."""
    sa = np.array([by_arm_sum[a][c] for c in cells])
    na = np.array([by_arm_n[a][c] for c in cells], dtype=float)
    sb = np.array([by_arm_sum[b][c] for c in cells])
    nb = np.array([by_arm_n[b][c] for c in cells], dtype=float)
    ratio_pt = (sa.sum() / na.sum()) / (sb.sum() / nb.sum()) - 1
    mean_pt = (sa / na).mean() / (sb / nb).mean() - 1
    d_ratio = np.empty(N_BOOT)
    d_mean = np.empty(N_BOOT)
    for s in range(N_BOOT):
        pick = np.concatenate([
            rng.choice(idx_by_task[t], size=len(idx_by_task[t]), replace=True)
            for t in tasks
        ])
        d_ratio[s] = (sa[pick].sum() / na[pick].sum()) / \
                     (sb[pick].sum() / nb[pick].sum()) - 1
        d_mean[s] = (sa[pick] / na[pick]).mean() / (sb[pick] / nb[pick]).mean() - 1
    return ratio_pt, d_ratio.std(ddof=1), mean_pt, d_mean.std(ddof=1)


rows = []
for a, b in itertools.combinations(arms, 2):
    rp, sr, mp, sm = se_both(a, b)
    rows.append({"a": a, "b": b, "gap_ratio": rp, "se_ratio": sr,
                 "gap_mean": mp, "se_mean": sm})

near = sorted(rows, key=lambda r: abs(r["gap_ratio"]))[:20]
se_ratio = statistics.median([r["se_ratio"] for r in near])
se_mean = statistics.median([r["se_mean"] for r in near])
print(f"\nnested-pair SE (median over {len(near)} near-zero-gap pairs):")
print(f"  ruling 9.4 estimand (decision-weighted ratio-of-sums): {se_ratio:.5f}")
print(f"  proposal's estimand (equal-weight over episodes):      {se_mean:.5f}")


def phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def release(thr, z, se, true_effect):
    return phi((thr - z * se - true_effect) / se)


OPTIONS = [("A  97.5% upper <= +5%", 0.05, 1.9600),
           ("C  point estimate <= +5%", 0.05, 0.0),
           ("D  95% upper <= +5%  (ruling)", 0.05, 1.6449),
           ("E  90% upper <= +5%", 0.05, 1.2816)]

for label, se in (("ruling estimand", se_ratio), ("proposal estimand", se_mean)):
    print(f"\nrelease probability under the {label} (SE={se:.5f}):")
    print(f"  {'rule':32s} {'true 0%':>8s} {'true +5%':>9s} {'true +8%':>9s}"
          f" {'true +15%':>10s}")
    for name, thr, z in OPTIONS:
        print(f"  {name:32s} {release(thr, z, se, 0.0):8.3f} "
              f"{release(thr, z, se, 0.05):9.3f} {release(thr, z, se, 0.08):9.3f} "
              f"{release(thr, z, se, 0.15):10.3f}")
    print("   'true +5%' is the boundary: it IS the rule's type-I error rate.")
