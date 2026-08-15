"""SmolVLA distillation finetune wrapper (Phase 1, lerobot venv on ziyang10).

Thin shell around the lerobot training CLI: pins the base checkpoint and
hyper-parameters from a recipe yaml, trains on the distillation dataset built
by ``build_distill_dataset.py``, and emits the freeze manifest (versions,
preprocessing contract, checkpoint sha256) required by the plan §5 Phase 1.

Checkpoint selection reads ONLY the student-val slice (leakage guard);
pruned_init is never touched here.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import subprocess
import sys

import yaml


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, help="training recipe yaml (config/)")
    parser.add_argument("--dataset", required=True, help="lerobot dataset root")
    parser.add_argument("--out", required=True, help="checkpoint output dir")
    parser.add_argument("--freeze-manifest-out", required=True)
    args = parser.parse_args()

    recipe = yaml.safe_load(pathlib.Path(args.recipe).read_text())
    # API-level entry (train_student.py): the 0.3.3 CLI cannot consume the
    # pre-chunked [10,7] O5 labels — see train_student.py module docstring.
    trainer = pathlib.Path(__file__).with_name("train_student.py")
    cmd = [
        sys.executable, str(trainer),
        "--student", "smolvla",
        f"--base-checkpoint={recipe['base_checkpoint']}",
        f"--dataset={args.dataset}",
        f"--out={args.out}",
        f"--steps={recipe['steps']}",
        f"--batch-size={recipe['batch_size']}",
    ] + [str(x) for x in recipe.get("extra_args", [])]
    subprocess.run(cmd, check=True)

    import lerobot

    lock = pathlib.Path(args.out) / "env_lock.txt"
    lock.write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout)

    ckpt = sorted(pathlib.Path(args.out).rglob("*.safetensors"))
    manifest = {
        "student": "smolvla",
        "lerobot_version": lerobot.__version__,
        "base_checkpoint": recipe["base_checkpoint"],
        "recipe": args.recipe,
        "dataset": args.dataset,
        "action_contract": "env_action_chunk [10, 7] env-space, replan client-side",
        "obs_contract": "resize_with_pad uint8 images + 8-dim float32 state",
        "checkpoint_sha256": {str(p): _sha256(p) for p in ckpt},
        "env_lock": str(pathlib.Path(args.out) / "env_lock.txt"),
    }
    pathlib.Path(args.freeze_manifest_out).write_text(yaml.safe_dump(manifest, sort_keys=False))


if __name__ == "__main__":
    main()
