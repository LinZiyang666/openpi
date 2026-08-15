"""Validation-only student checkpoint selection + strength gate (Phase 1).

Scores candidate checkpoints from their student-val rollout trajectory dirs
(``main.py --save_trajectory`` against the sidecar on the val-only inits from
``build_distill_dataset --emit-val-inits``), tallies SR from ``attrs["success"]``,
enforces the strength admission band (SR must sit well below the Pi0.5 anchor
and well above zero), selects the best admitted checkpoint, and freezes it
into the manifest with sha256 hashes. pruned_init is never read here.

Usage:
    python exp/ablation_study/select_student_checkpoint.py \
        --candidates ckptA:/traj/dirA ckptB:/traj/dirB \
        --anchor-sr 0.95 --band-low 0.10 --band-high 0.85 \
        --freeze-manifest exp/ablation_study/config/freeze_manifest_<suite>.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import h5py
import yaml


def val_sr(traj_dir: str) -> tuple[float, int]:
    files = sorted(pathlib.Path(traj_dir).rglob("episode_*.h5"))
    if not files:
        # EN-2 farm keeps only the per-rollout results.json (H5s are pruned to
        # save disk); the success records are the same attrs the H5s carried.
        rj = pathlib.Path(traj_dir) / "results.json"
        if rj.exists():
            rows = json.loads(rj.read_text())
            if rows:
                return sum(bool(r["success"]) for r in rows) / len(rows), len(rows)
        raise SystemExit(f"no val trajectories under {traj_dir}")
    succ = sum(bool(h5py.File(f, "r").attrs.get("success", False)) for f in files)
    return succ / len(files), len(files)


def resolve_pretrained_dir(out_dir) -> pathlib.Path:
    """Resolve a LeRobot 0.3.3 train output root to its loadable policy dir:
    ``<out>/checkpoints/last/pretrained_model`` (or the highest step when the
    ``last`` link is absent). Raises when no checkpoint exists."""
    root = pathlib.Path(out_dir)
    ckpts = root / "checkpoints"
    last = ckpts / "last" / "pretrained_model"
    if last.exists():
        return last
    steps = sorted(d for d in ckpts.glob("*") if (d / "pretrained_model").exists())
    if not steps:
        raise SystemExit(f"no LeRobot checkpoints under {ckpts}")
    return steps[-1] / "pretrained_model"


def sha256_tree(root: str) -> dict[str, str]:
    out = {}
    for p in sorted(pathlib.Path(root).rglob("*.safetensors")):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[str(p)] = h.hexdigest()
    return out


def update_act_manifest(manifest_path: str, prompt: str, checkpoint: str) -> None:
    """Atomically point the sidecar's prompt manifest at the SELECTED model
    (resolving a train-output root to its loadable pretrained_model dir)."""
    import json
    import os

    ckpt = pathlib.Path(checkpoint)
    if (ckpt / "checkpoints").exists():
        ckpt = resolve_pretrained_dir(ckpt)
    mp = pathlib.Path(manifest_path)
    manifest = json.loads(mp.read_text()) if mp.exists() else {}
    manifest[prompt] = str(ckpt)
    tmp = mp.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    os.replace(tmp, mp)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="+", required=True,
                        help="<checkpoint_dir>:<val_traj_dir> pairs")
    parser.add_argument("--anchor-sr", type=float, required=True)
    parser.add_argument("--band-low", type=float, default=0.10)
    parser.add_argument("--band-high", type=float, default=0.85,
                        help="admission band as fractions of anchor SR")
    parser.add_argument("--freeze-manifest", required=True)
    parser.add_argument("--update-act-manifest", default=None,
                        help="ACT prompt->checkpoint json to update atomically")
    parser.add_argument("--prompt", default=None,
                        help="prompt key to point at the selected checkpoint")
    args = parser.parse_args()
    if bool(args.update_act_manifest) != bool(args.prompt):
        raise SystemExit("--update-act-manifest and --prompt must be given together")

    lo = args.band_low * args.anchor_sr
    hi = args.band_high * args.anchor_sr
    scored = []
    for spec in args.candidates:
        ckpt, traj = spec.split(":", 1)
        sr, n = val_sr(traj)
        admitted = lo <= sr <= hi
        scored.append({"checkpoint": ckpt, "val_sr": sr, "n_val": n, "admitted": admitted})
        print(f"{ckpt}: val SR={sr:.3f} (n={n}) admitted={admitted} band=[{lo:.3f},{hi:.3f}]")
    admitted = [s for s in scored if s["admitted"]]
    if not admitted:
        raise SystemExit(
            "no candidate inside the strength admission band — adjust the "
            "training knobs (plan §11-O4) and re-run; do NOT touch pruned_init."
        )
    best = max(admitted, key=lambda s: s["val_sr"])
    manifest = yaml.safe_load(pathlib.Path(args.freeze_manifest).read_text())
    manifest["selection"] = {
        "candidates": scored,
        "selected_checkpoint": best["checkpoint"],
        "selected_val_sr": best["val_sr"],
        "admission_band": [lo, hi],
        "anchor_sr": args.anchor_sr,
        "selected_sha256": sha256_tree(best["checkpoint"]),
    }
    pathlib.Path(args.freeze_manifest).write_text(yaml.safe_dump(manifest, sort_keys=False))
    if args.update_act_manifest:
        update_act_manifest(args.update_act_manifest, args.prompt, best["checkpoint"])
    print(f"frozen: {best['checkpoint']}")


if __name__ == "__main__":
    main()
