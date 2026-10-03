"""CPU-only fake stage/kind profile, optionally decoding a real read-only arm."""
import argparse
import json
from pathlib import Path

from .. import reader
from ..fixtures import make_synthetic_arm
from .cpu_fake import CpuGrootModel
from .job import KINDS, run_arm, private_seed, wire_observations
from .models import FakeModel
from .retrieval import FakeRetrieval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="fresh scratch directory")
    parser.add_argument("--rows", type=int, default=96)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--arm")
    parser.add_argument("--model", choices=("pi05", "groot"))
    args = parser.parse_args()
    if args.out.exists():
        parser.error("use a fresh --out directory")
    if args.run_root and not (args.arm and args.model):
        parser.error("real capture requires --arm and --model")
    args.out.mkdir(parents=True)
    profiles = []
    for name in ([args.model] if args.model else ["pi05", "groot"]):
        root = args.run_root or make_synthetic_arm(args.out / (name + "_capture"), n_episodes=12, model=name)
        arm = reader.open_arm(root, args.arm or "synthetic")
        for scope in ["all"] + list(KINDS if name == "pi05" else KINDS[:-1]):
            kinds = None if scope == "all" else [scope]
            for mode, batch, prefetch in [("serial", 1, False), ("batched", 32, False), ("batched", 32, True)]:
                model = CpuGrootModel(mode) if name == "groot" else FakeModel(name)
                model.stage_mode = mode
                first = arm.decisions().head(1).to_dict("records")
                arm.cache_enabled = False
                encoded = model.encode(wire_observations(arm, first))
                row = first[0]
                model.draw(encoded, [private_seed(arm.manifest.get("campaign", arm.run_root.name), row["task_id"], row["init"], row["decision_seq"], 0)])
                model.synchronize()
                if name == "groot":
                    model.prepared_rows = 0
                    model.runner._eagle.batches.clear()
                    model.runner._eagle.language_model.batches.clear()
                    model.runner.action_batches.clear()
                else:
                    model.encode_batches.clear()
                    model.draw_batches.clear()
                target = args.out / ("%s_%s_%s_b%d_prefetch%d" % (name, scope, mode, batch, prefetch))
                libraries = ("current", "bpool_cs" if name == "pi05" else "bpool_all") if arm.manifest.get("pure_policy") else ("current",)
                report = run_arm(arm, model, FakeRetrieval(name, libraries=libraries), batch, kinds, checkpoint_sha="cpu-fake-not-checkpoint",
                                 limit=args.rows, profile=True, prefetch=prefetch, output_dir=target)
                if name == "groot":
                    report["dispatches"] = dict(stage1=model.runner._eagle.batches,
                        stage2=model.runner._eagle.language_model.batches, stage3=model.runner.action_batches)
                else:
                    report["dispatches"] = dict(stage1=model.encode_batches, stage3=model.draw_batches)
                profiles.append(dict(model=name, scope=scope, **report))
    target = args.out / "profile.json"
    target.write_text(json.dumps(dict(profiles=profiles,
        interpretation="CPU fake includes real NPZ decode/publication and production GR00T stage slicing/buckets. Toy encoder, LLM, action and retrieval timings are not real checkpoint/GPU latency estimates."), indent=2) + "\n")
    print(str(target))


if __name__ == "__main__":
    main()
