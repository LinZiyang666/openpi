"""Power of the cost gates when cost is computed analytically from the precheck.

Cost on this line is the GPU inference ratio: per decision it is the verdict
weighted by the CUDA-graph stage costs (FULL_HIT = stage1; WARM_START =
stage1 + stage2 + start_t * stage3; MISS = all three). The precheck's per_step
journal already records hit_type and start_t for every step under a full
(arm, task, init, episode) join key, so every arm's cost falls out of the SR
run itself -- 300 episodes per arm, paired within an init. No separate bench,
no blocks, no extra A' inits, and nothing to measure with a wall clock.

That removes the reason plan section 4.6 introduced an isolated cost
experiment ("existing interfaces cannot support per-decision paired cost"):
they cannot support a *timed* one, but an analytic one needs only the verdict
counts, which are already joined.

Arm pairs are grouped by what each Gate actually compares, because the
bootstrap SE of a paired relative cost difference depends strongly on how
far apart the two arms are: when their verdict mixes are close they move
together on every init and the pairing cancels most of the variance. So the
question "what SE do the gates get" has to be asked on the pairs each gate
actually faces:

  Gate 2   SV vs S0 -- nested, same delta*, differing only in the v guard.
           Analogue: the arm pairs whose true cost gap is near zero.
  Gate 1   SV vs a point interpolated on the threshold frontier, where the
           pre-registered surrogate effect is -10%.
           Analogue: pairs whose true cost gap is around -10%.

Both are measured on 300 episodes/arm (--trials 30 on the A' pool), paired
within an init, task-stratified init-cluster bootstrap.
"""

import collections
import itertools
import json
import statistics
from math import erf, sqrt

import numpy as np

from exp.dispatch_surface.analysis.block_variance_probe import load_episodes

TRIALS_PER_TASK = 30
N_BOOT = 2000
SEED = 20260827
GATES = (("gate1_compute", -0.10, -0.05, 1.6449),
         ("gate2_compute", 0.00, 0.05, 1.9600))

eps = load_episodes(
    "exp/gate_threshold_pareto/data/eval/libero_spatial/per_step.jsonl"
)
by_arm = collections.defaultdict(dict)
for e in eps:
    by_arm[e["arm"]][(e["task_id"], e["init_idx"])] = e
arms = sorted(by_arm)
common = sorted(set.intersection(*(set(d) for d in by_arm.values())))
tasks = sorted({t for t, _ in common})
per_task = {t: sorted(i for (tt, i) in common if tt == t)[:TRIALS_PER_TASK]
            for t in tasks}
cells = [(t, i) for t in tasks for i in per_task[t]]
idx_by_task = {t: np.array([k for k, (tt, _) in enumerate(cells) if tt == t])
               for t in tasks}
rng = np.random.default_rng(SEED)
print(f"{len(arms)} arms, {len(cells)} episodes/arm "
      f"({len(per_task[tasks[0]])} inits/task x {len(tasks)} tasks)")


def stats_for(a: str, b: str) -> tuple[float, float]:
    ca = np.array([by_arm[a][c]["compute_ms"] for c in cells])
    cb = np.array([by_arm[b][c]["compute_ms"] for c in cells])
    point = (ca.mean() - cb.mean()) / cb.mean()
    draws = np.empty(N_BOOT)
    for s in range(N_BOOT):
        pick = np.concatenate([
            rng.choice(idx_by_task[t], size=len(idx_by_task[t]), replace=True)
            for t in tasks
        ])
        draws[s] = (ca[pick].mean() - cb[pick].mean()) / cb[pick].mean()
    return point, draws.std(ddof=1)


rows = []
for a, b in itertools.combinations(arms, 2):
    point, se = stats_for(a, b)
    rows.append({"a": a, "b": b, "gap": point, "se": se})


def power(effect: float, thr: float, z: float, se: float) -> float:
    x = (thr - z * se - effect) / se
    return 0.5 * (1 + erf(x / sqrt(2)))


def report(label: str, subset: list[dict], gate) -> None:
    name, effect, thr, z = gate
    ses = sorted(r["se"] for r in subset)
    med = statistics.median(ses)
    print(f"\n{label}  (n={len(subset)} pairs)")
    print(f"  |gap| range: {min(abs(r['gap']) for r in subset):.3%} .. "
          f"{max(abs(r['gap']) for r in subset):.3%}")
    print(f"  SE: min={ses[0]:.5f} median={med:.5f} max={ses[-1]:.5f}")
    for tag, se in (("median SE", med), ("worst SE", ses[-1])):
        print(f"  {name} power at {tag} ({se:.5f}): "
              f"{power(effect, thr, z, se):.3f}")


# Gate 2 analogue: near-zero true gap.
near = sorted(rows, key=lambda r: abs(r["gap"]))[:20]
report("Gate 2 analogue -- nested pairs, true gap ~ 0", near, GATES[1])

# Gate 1 analogue: true gap near the pre-registered -10% surrogate.
near10 = sorted(rows, key=lambda r: abs(abs(r["gap"]) - 0.10))[:20]
report("Gate 1 analogue -- pairs whose true gap is ~10%", near10, GATES[0])

# Largest SE anywhere, as a worst case.
allrows = sorted(rows, key=lambda r: r["se"])
print(f"\nacross ALL {len(rows)} pairs: SE min={allrows[0]['se']:.5f} "
      f"median={allrows[len(allrows)//2]['se']:.5f} max={allrows[-1]['se']:.5f}")
print("  (pairs with a large true gap have large SE; the gates never face those)")

out = {
    "episodes_per_arm": len(cells),
    "inits_per_task": len(per_task[tasks[0]]),
    "n_boot": N_BOOT,
    "gate2_analogue": {
        "pairs": len(near),
        "se_median": statistics.median([r["se"] for r in near]),
        "se_max": max(r["se"] for r in near),
        "power_at_median": power(*GATES[1][1:],
                                 statistics.median([r["se"] for r in near])),
        "power_at_worst": power(*GATES[1][1:], max(r["se"] for r in near)),
    },
    "gate1_analogue": {
        "pairs": len(near10),
        "se_median": statistics.median([r["se"] for r in near10]),
        "se_max": max(r["se"] for r in near10),
        "power_at_median": power(*GATES[0][1:],
                                 statistics.median([r["se"] for r in near10])),
        "power_at_worst": power(*GATES[0][1:], max(r["se"] for r in near10)),
    },
}
print("\n" + json.dumps(out, indent=2))
