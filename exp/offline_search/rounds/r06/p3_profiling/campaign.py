"""One superset campaign, source A/B fits copied by provenance, never refitted here."""
import argparse
import json
import math
from pathlib import Path

from exp.offline_search.rounds.r06.p1_groot_commit.make_arms import source_rows

HERE = Path(__file__).resolve().parent
ROOT = "/home/weiland/trace_runs/offline_search_store"


def fitpath(row):
    args = row["plugin_args"]
    return args[args.index("--os-fit-artifact") + 1].replace("<RUN>", "/home/weiland/trace_runs/os_closed_loop/r06_paper")


def make_kwargs(model, cell, p=.1, replicate=0, enabled=True, baseline="A"):
    source = source_rows()
    base = source[model, baseline, cell]["row"]
    guard = source[model, "B", cell]["row"]
    kw = dict(base_spec=base["method"], base_kwargs=base["kwargs"], base_fit=fitpath(base),
              enabled=enabled, p=p, seed=603, replicate=replicate)
    if baseline == "A":
        kw.update(guard_spec=guard["method"], guard_kwargs=guard["kwargs"], guard_fit=fitpath(guard))
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=4)
    ap.add_argument("--p", type=float, default=.2)
    a = ap.parse_args()
    arms = []
    for model in ("pi05", "groot"):
        for cell in ("l10_50", "l10_500", "sp_50", "sp_500"):
            suite = "spatial" if cell.startswith("sp_") else "l10"
            for rep in range(a.replicates):
                name = f"r6p3_{model}_{cell}_r{rep}"
                arms.append(dict(name=name, model=model, suite=suite, mode="plugin",
                    method="exp.offline_search.rounds.r06.p3_profiling.method:Profile",
                    kwargs=make_kwargs(model, cell, a.p, rep), full_model=True, cost_ledger=True,
                    client_overrides=dict(replan_steps=5, resize_size=256 if model == "groot" else 224),
                    plugin_args=["--os-root", ROOT, "--os-blind", "--os-policy-tail",
                        "--os-policy-tail-blocks", "1", "--os-judge", "guard_only", "--os-no-shadow-native",
                        "--os-log-inputs",
                        "--os-fit-artifact", f"<RUN>/fits/{name}.pkl"]))
    (HERE / "arms.json").write_text(json.dumps(arms, indent=2) + "\n")
    # Power is a sensitivity analysis, not a promise of independent anchors.
    power = []
    for rho in (0., .01, .05, .1, .2, 1.):
        anchors_per_episode = 30
        # The 4 replicas reuse the SAME 500 inits. Cluster them together.
        anchors_per_cluster = anchors_per_episode * a.replicates
        deff = 1 + (anchors_per_cluster - 1) * rho
        n = 500 * a.replicates
        effective = n * anchors_per_episode / deff
        se = math.sqrt(.25 / (effective * a.p * (1 - a.p)))
        power.append(dict(episodes=n, unique_init_clusters=500, anchors_per_episode=anchors_per_episode,
            anchors_per_cluster=anchors_per_cluster, within_init_score_icc=rho,
            design_effect=deff, effective_anchors=effective, ci95_halfwidth=1.96 * se,
            mde80=(1.96 + .841621) * se,
            episodes_for_5pp=math.ceil((1.96 + .841621)**2 * .25 * deff /
                                     (.05**2 * anchors_per_episode * a.p * (1 - a.p)))))
    costs = json.loads(Path("exp/offline_search/closed_loop/ops/cost_table.json").read_text())["models"]
    gpu = {m: costs[m]["full_cost_ms"] * 30 * 4 * 500 * a.replicates / 3.6e6 for m in costs}
    report = dict(p=a.p, seed=603, arms=len(arms), episodes_per_cell=500 * a.replicates,
                  episodes_total=500 * len(arms), assumed_anchors_per_episode=30, power=power,
                  gpu_compute_hours=gpu, gpu_compute_hours_total=sum(gpu.values()),
                  gpu_budget_hours_2x=sum(gpu.values()) * 2,
                  note="old eager batch-1 timing basis; wall occupancy and simulator costs unmeasured")
    (HERE / "results/campaign_calculation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
