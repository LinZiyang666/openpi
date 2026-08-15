"""Per-task ACT distillation training wrapper (Phase 1, lerobot venv).

Trains one ACT policy per task (ACT has no language conditioning; the sidecar
routes by exact prompt match) and emits the prompt -> checkpoint manifest the
sidecar consumes, plus the freeze manifest. Chunk size follows the student
action contract (10-step env_action_chunk labels).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import yaml

from exp.ablation_study.select_student_checkpoint import resolve_pretrained_dir
from exp.ablation_study.select_student_checkpoint import sha256_tree


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--dataset-root", required=True, help="per-task dataset roots parent")
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest-out", required=True, help="prompt->checkpoint json")
    parser.add_argument("--freeze-manifest-out", required=True)
    args = parser.parse_args()

    recipe = yaml.safe_load(pathlib.Path(args.recipe).read_text())
    manifest: dict[str, str] = {}
    for task_dir in sorted(pathlib.Path(args.dataset_root).glob("task_*")):
        prompt = (task_dir / "prompt.txt").read_text().strip()
        out_dir = pathlib.Path(args.out) / task_dir.name
        # API-level entry (train_student.py): the 0.3.3 CLI cannot consume the
        # pre-chunked [10,7] O5 labels — see train_student.py module docstring.
        trainer = pathlib.Path(__file__).with_name("train_student.py")
        cmd = [
            sys.executable, str(trainer),
            "--student", "act",
            f"--chunk-size={recipe.get('chunk_size', 10)}",
            f"--dataset={task_dir / 'dataset'}",
            f"--out={out_dir}",
            f"--steps={recipe['steps']}",
            f"--batch-size={recipe['batch_size']}",
        ] + [str(x) for x in recipe.get("extra_args", [])]
        subprocess.run(cmd, check=True)
        # LeRobot 0.3.3 layout: loadable policy lives under
        # checkpoints/<step>/pretrained_model (checkpoints/last is the link).
        manifest[prompt] = str(resolve_pretrained_dir(out_dir))
    pathlib.Path(args.manifest_out).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    import lerobot

    lock = pathlib.Path(args.out) / "env_lock.txt"
    lock.write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout)

    freeze = {
        "student": "act_per_task",
        "lerobot_version": lerobot.__version__,
        "recipe": args.recipe,
        "dataset_root": args.dataset_root,
        "prompt_manifest": args.manifest_out,
        "env_lock": str(pathlib.Path(args.out) / "env_lock.txt"),
        "checkpoint_sha256": {t: sha256_tree(p) for t, p in manifest.items()},
        "action_contract": "env_action_chunk [10, 7] env-space, replan client-side",
        "obs_contract": "resize_with_pad uint8 images + 8-dim float32 state",
        "selection_protocol": "student-val only (build_distill_dataset --emit-val-inits)",
    }
    pathlib.Path(args.freeze_manifest_out).write_text(yaml.safe_dump(freeze, sort_keys=False))


if __name__ == "__main__":
    main()
