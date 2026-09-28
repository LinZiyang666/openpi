"""One campaign: a 4,320-episode pilot and disjoint continuation to 36,000.

Same arm IDs/fits/designs across stages. Three seed blocks share init clusters.
Only emits specifications and manifests; never launches anything.
"""
import argparse
import json
import math
from pathlib import Path

from .campaign import make_kwargs, ROOT, HERE


COHORTS = [
    ("A", "A", 0., {}), ("dose125", "A", .125, {}),
    ("dose25", "A", .25, {}), ("dose50", "A", .5, {}),
    ("P10", "A", 1., {}), ("B", "B", 0., {}),
    ("factorial", "B", .25, dict(pre_guard=True, durations=[5, 10], holds=[1, 2, 3], cooldown=1)),
    ("window", "A", .125, dict(cap=1, delays=[0, 1, 2], start_anchor=2)),
    ("dose_mix", "A", .25, dict(episode_doses=[0., .125, .25, .5, 1.])),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "results").mkdir(exist_ok=True)
    arms, schedule = [], []
    for model in ("pi05", "groot"):
        for cell in ("l10_50", "l10_500", "sp_50", "sp_500"):
            suite = "spatial" if cell.startswith("sp_") else "l10"
            for cohort, baseline, p, design in COHORTS:
                for rep, ninit in enumerate((25, 15, 10)):
                    name = f"r6p3v2_{model}_{cell}_{cohort}_r{rep}"
                    kw = make_kwargs(model, cell, p=p, replicate=rep, baseline=baseline)
                    kw.update(design=dict(cohort=cohort, cohort_probability=1., **design),
                        blind_shadow=True, resample_p=1/16, resample_draws=4,
                        calibration_path=f"<RUN>/calibration/{model}_{cell}.json",
                        provenance=dict(population="equal weight on ten LIBERO tasks", task_probability=.1,
                            init_pool_size=50, init_pool="stock benchmark; repeated IDs are not new scenes",
                            block=rep, validation_split="init % 5 != 0; freeze score before unsealing",
                            design_assignment="complete matched block; equal cohort allocation, not stochastic assignment"))
                    if cohort == "factorial":
                        kw["strata"] = dict(edges=[1/3, 2/3], p=[.125, .25, .5])
                    arms.append(dict(name=name, model=model, suite=suite, mode="plugin",
                        method="exp.offline_search.rounds.r06.p3_profiling.v2:Profile", kwargs=kw,
                        full_model=True, cost_ledger=True,
                        client_overrides=dict(replan_steps=5, resize_size=256 if model == "groot" else 224),
                        plugin_args=["--os-root", ROOT, "--os-blind", "--os-policy-tail", "--os-policy-tail-blocks", "1",
                            "--os-judge", "guard_only", "--os-no-shadow-native", "--os-log-inputs",
                            "--os-fit-artifact", f"<RUN>/fits/{name}.pkl"]))
                    schedule.append(dict(arm=name, pilot_manifest=f"<RUN>/manifests/pilot_r{rep}.json",
                        continuation_manifest=f"<RUN>/manifests/continuation_r{rep}.json",
                        pilot_episodes=20, continuation_episodes=10*(ninit-2), full_episodes=10*ninit,
                        client_env=dict(P3_ENV_SEED=str(603+rep)),
                        client_launcher="exp/offline_search/rounds/r06/p3_profiling/run_arm_v2.sh"))
    (a.out / "arms_v2.json").write_text(json.dumps(arms, indent=2) + "\n")
    (a.out / "schedule_v2.json").write_text(json.dumps(schedule, indent=2) + "\n")
    manifests = a.out / "manifests_v2"
    manifests.mkdir(exist_ok=True)
    for rep, ninit in enumerate((25, 15, 10)):
        for stage, indices in (("pilot", range(2)), ("continuation", range(2, ninit))):
            (manifests / f"{stage}_r{rep}.json").write_text(json.dumps(dict(stage=stage, replicate=rep,
                selected=[dict(task=t, init=i) for t in range(10) for i in indices]), indent=2) + "\n")
    costs = json.loads(Path("exp/offline_search/closed_loop/ops/cost_table.json").read_text())["models"]
    cost = {}
    # Conservative head share=1 prices each extra head at an entire full call.
    # 60 decisions, >=30 anchors per episode is an explicit planning assumption;
    # duration5 cohort may have more anchors, priced at 60 for resampling below.
    equivalents = 60 + 60 * 3/16
    for stage, episodes in (("pilot", 4320), ("full_including_pilot", 36000), ("continuation_only", 31680)):
        gpu = {m: (episodes/2)*equivalents*costs[m]["full_cost_ms"]/3.6e6 for m in ("pi05", "groot")}
        cost[stage] = dict(episodes=episodes, episodes_per_cell=episodes//8,
            gpu_compute_hours=gpu, gpu_compute_hours_total=sum(gpu.values()),
            gpu_occupancy_budget_2x_hours=2*sum(gpu.values()),
            raw_keys_gib=episodes*60*2*32768*4/2**30,
            collection_storage_budget_gib=episodes*100/1024,
            storage_assumption="100 MiB/episode incl keys/actions/images/physics/JSON; not measured; tables+fits additional")
    power = []
    for stage, repeats, clusters in (("pilot", [3]*20, 20), ("full", [3]*100+[2]*50+[1]*100, 250)):
        n = sum(repeats)
        avg_pair_count = sum(x*(x-1) for x in repeats)/n
        for rho in (0., .1, .3, .5, 1.):
            deff = 1 + avg_pair_count*rho
            neff = n/deff
            power.append(dict(stage=stage, episodes_per_cohort_cell=n, unique_init_clusters=clusters,
                repeat_outcome_icc=rho, effective_episodes=neff,
                paired_dose_halfwidth_assumed_discordance02=1.96*math.sqrt(.2/neff),
                paired_dose_mde80_assumed_discordance02=2.801621*math.sqrt(.2/neff),
                unpaired_worstcase_halfwidth=1.96*math.sqrt(.5/neff)))
    summary = dict(arms=len(arms), cohorts=[x[0] for x in COHORTS], cost=cost, power=power,
        pilot_per_cohort_cell=60, full_per_cohort_cell=500,
        notes=["Pilot estimates ICC, logging fidelity, score dispersion, support; does not settle small SR effects.",
            "Direct matched whole-dose contrasts use task/init clusters across seed blocks, never anchor independence.",
            "For a 5pp paired halfwidth with variance .2: ceil(1.96^2*.2/.05^2)=308 effective paired units.",
            "For 80% power on a 5pp paired effect: ceil(2.801621^2*.2/.05^2)=628 effective paired units.",
            "Full 250 distinct initializations/cohort/cell cannot promise those targets at high ICC; expand only using pilot variance.",
            "Shared GPU: admit 1 line, optionally 2 if coordinator memory permits; never infer twice the GPU throughput from two lines.",
            "Full phase uses disjoint manifests with SAME arm identities; pilot episodes are retained, not rerun."])
    (a.out / "results/campaign_v2.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
