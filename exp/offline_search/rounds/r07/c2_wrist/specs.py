"""Emit SW and SF+SW × the four pi05 cells in emit_arms input format."""
import argparse
import json
from pathlib import Path

from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import sources
from .method import BASE

HERE = Path(__file__).resolve().parent
SPEC = "exp.offline_search.rounds.r07.c2_wrist.method:StageWrist"
FOLLOW = "exp.offline_search.rounds.r07.c1_follow.methods:StageFollow"


def make_specs(run="<RUN>"):
    result = []
    has_follow = (HERE.parent / "c1_follow/methods.py").is_file()
    for cell in ("l10_50", "l10_500", "sp_50", "sp_500"):
        source = sources()["pi05_" + cell.replace("sp_", "spatial_")]
        for variant in (["sw", "sf_sw"] if has_follow else ["sw"]):
            base_kwargs = dict(source["kwargs"])
            if variant == "sf_sw":
                base_kwargs.update(extend_blocks=1, stage_gate=True, state_valve=True,
                                   stages_path=f"{run}/fits/stages_pi05_{cell}.pkl")
            name = f"r7_{variant}_pi05_{cell}"
            result.append(dict(name=name, model="pi05", suite="spatial" if cell.startswith("sp_") else "l10",
                mode="plugin", method=SPEC,
                kwargs=dict(enabled=True, base_spec=FOLLOW if variant == "sf_sw" else BASE,
                    base_kwargs=base_kwargs, base_fit=source["source_artifact"],
                    wrist_fit=f"{run}/fits/wrist_pi05_{cell}.pkl", stage_fit=f"{run}/fits/stages_pi05_{cell}.pkl"),
                full_model=True, cost_ledger=True, server_env={"BATCHING_MAX_BATCH_SIZE": "1"},
                client_overrides=dict(replan_steps=5, resize_size=224),
                plugin_args=["--os-root", "/home/weiland/trace_runs/offline_search_store", "--os-no-shadow-native",
                    "--os-tokens", "off", "--os-blind", "--os-request-cameras", "--os-fit-artifact", f"{run}/fits/{name}.pkl"]))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="<RUN>")
    ap.add_argument("--out", type=Path, default=HERE / "arms_in.json")
    a = ap.parse_args()
    arms = make_specs(a.run_root)
    a.out.write_text(json.dumps(arms, indent=2) + "\n")
    print(json.dumps({"arms": len(arms), "variants": sorted({r["name"].split("_pi05")[0] for r in arms}), "out": str(a.out)}))


if __name__ == "__main__":
    main()
