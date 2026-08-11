"""Distillation dataset builder: trajectory H5 -> lerobot datasets (Phase 1).

Consumes the client-side ``--save_trajectory`` episodes (canonical schema:
``attrs["task_name"] / attrs["success"] / attrs["init_state_idx"]`` + per-cycle
groups, examples/libero/main.py) collected over the 500-init diff pool and
produces:

- a deterministic per-task **45/5 train/val split** (tracked yaml) with the
  leakage-guard constraint that the cache-library init positions (the 5/task
  states in ``db_init/libero_cache``) always stay in **train** — preserving the
  same-source-superset property; student calibration only ever reads the val
  slice, and pruned_init stays untouched until models are frozen;
- lerobot datasets for the requested split part: one combined multi-task
  dataset (SmolVLA) and/or a per-task layout ``task_<id>/{dataset, prompt.txt}``
  (ACT trainer + sidecar manifest input);
- per-task ``.init`` files holding ONLY the val inits (``--emit-val-inits``),
  so student-val rollouts run standalone main.py against the sidecar without
  touching pruned_init.

Also hosts the pre-collection guard ``--check-init-dir`` (asserts no
``.pruned_init`` shadowing, main.py's loader prefers that extension).

lerobot imports are lazy: split/init emission and the guard run in the main
openpi venv; dataset writing runs in the lerobot venv on ziyang10.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import re

import h5py
import numpy as np
import yaml

logger = logging.getLogger(__name__)

SPLIT_SEED = 20260811
VAL_PER_TASK = 5


def check_init_dir(init_dir: str) -> None:
    """Fail fast if the init pool directory could be shadowed by pruned files."""
    d = pathlib.Path(init_dir)
    stray = sorted(p.name for p in d.glob("*.pruned_init"))
    if stray:
        raise SystemExit(
            f"init dir {init_dir} contains .pruned_init files {stray}: "
            "collection would silently run on the test set. Aborting."
        )
    n = len(list(d.glob("*.init")))
    logger.info("init dir OK: %d .init files, no .pruned_init", n)


# ----------------------------------------------------------------------
# Split (pure helpers, unit-tested with fixtures)
# ----------------------------------------------------------------------


def cache_positions_for_task(pool_states, cache_states) -> list[int]:
    """Positions (row indices) of the cache-library states inside the diff
    pool init array — same matching rule as sample_cache.find_original_indices."""
    pool = np.asarray(pool_states)
    positions = []
    for row in np.asarray(cache_states):
        matches = np.where(np.all(np.isclose(pool, row, atol=1e-7), axis=1))[0]
        if len(matches) != 1:
            raise SystemExit(f"cache init row matched {len(matches)} pool rows (expected 1)")
        positions.append(int(matches[0]))
    return positions


def constrained_split(episodes: list[int], protected: list[int], val_n: int, rng) -> dict:
    """Sample ``val_n`` val episodes from ``episodes`` excluding ``protected``
    (cache-library positions must stay in train)."""
    candidates = [e for e in episodes if e not in set(protected)]
    if len(candidates) < val_n:
        raise SystemExit(f"not enough non-protected episodes for val: {len(candidates)} < {val_n}")
    val = sorted(rng.choice(candidates, size=val_n, replace=False).tolist())
    return {"train": [e for e in episodes if e not in val], "val": val,
            "protected_in_train": sorted(protected)}


def emit_split(raw_dir: str, pool_init_dir: str, cache_init_dir: str, out_path: str,
               stem_map: dict | None = None) -> dict:
    """Deterministic stratified split with the cache-subset-in-train constraint."""
    import torch

    rng = np.random.RandomState(SPLIT_SEED)
    split: dict[str, dict] = {}
    for task_dir in sorted(pathlib.Path(raw_dir).glob("task_*")):
        eps = sorted(int(p.stem.split("_")[1]) for p in task_dir.glob("episode_*.h5"))
        task_name = _task_name_of(task_dir)
        stem = resolve_init_stem(task_name, pool_init_dir, stem_map)
        pool = torch.load(pathlib.Path(pool_init_dir) / f"{stem}.init", weights_only=False)
        cache = torch.load(pathlib.Path(cache_init_dir) / f"{stem}.init", weights_only=False)
        protected = cache_positions_for_task(pool, cache)
        split[task_dir.name] = constrained_split(eps, protected, VAL_PER_TASK, rng)
        split[task_dir.name]["task_name"] = task_name
        split[task_dir.name]["init_stem"] = stem
    pathlib.Path(out_path).write_text(yaml.safe_dump(split, sort_keys=True))
    logger.info("split written: %s", out_path)
    return split


def _task_name_of(task_dir: pathlib.Path) -> str:
    """Read the canonical task_name from the first episode's attrs."""
    first = sorted(task_dir.glob("episode_*.h5"))[0]
    with h5py.File(first, "r") as h:
        return str(h.attrs["task_name"])


def resolve_init_stem(task_name: str, init_dir: str, stem_map: dict | None = None) -> str:
    """Map the natural-language task_name (HDF5 ``attrs["task_name"]``) onto the
    ``task.name`` stem of the ``.init`` files. Resolution order:

    1. explicit ``stem_map`` (task_name -> stem; authoritative, emitted from the
       LIBERO benchmark on the client via --emit-stem-map);
    2. exact normalized match (libero_spatial: stems carry no scene prefix);
    3. UNIQUE suffix match (libero_10/90: stems are ``<SCENE_PREFIX>_<language>``,
       e.g. ``KITCHEN_SCENE3_turn_on_the_stove_...``); ambiguity or zero -> exit.
    """
    if stem_map and task_name in stem_map:
        return str(stem_map[task_name])
    stem = re.sub(r"[^A-Za-z0-9]+", "_", task_name.strip()).strip("_")
    candidates = sorted(p.stem for p in pathlib.Path(init_dir).glob("*.init"))
    exact = [c for c in candidates if c.lower() == stem.lower()]
    if len(exact) == 1:
        return exact[0]
    suffix = [c for c in candidates if c.lower().endswith("_" + stem.lower())]
    if len(suffix) == 1:
        return suffix[0]
    raise SystemExit(
        f"task_name {task_name!r} (stem {stem!r}) resolves to {exact or suffix} "
        f"in {init_dir} (need exactly one; pass --stem-map for authoritative "
        f"mapping); available: {candidates[:5]}..."
    )


def iter_frames(raw_dir: str, split: dict, part: str, only_task: str | None = None):
    """Yield (frame dict, task_name) for successful episodes of a split part."""
    for task_key, parts in sorted(split.items()):
        if only_task is not None and task_key != only_task:
            continue
        for ep_idx in parts[part]:
            path = pathlib.Path(raw_dir) / task_key / f"episode_{ep_idx}.h5"
            if not path.exists():
                continue
            with h5py.File(path, "r") as h:
                if not bool(h.attrs.get("success", False)):
                    continue
                task_name = str(h.attrs["task_name"])
                for cycle in sorted(k for k in h.keys() if k.startswith("cycle_") or k.startswith("step_")):
                    g = h[cycle]
                    yield {
                        "image": np.asarray(g["agentview_image"]),
                        "wrist_image": np.asarray(g["eye_in_hand_image"]),
                        "state": np.asarray(g["robot_state"], dtype=np.float32),
                        "actions": np.asarray(g["env_action_chunk"], dtype=np.float32),
                    }, task_name


# ----------------------------------------------------------------------
# lerobot dataset writing (lerobot venv only)
# ----------------------------------------------------------------------


def build_lerobot_dataset(raw_dir, split_path, out_dir, part="train", only_task=None, fps=20):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    split = yaml.safe_load(pathlib.Path(split_path).read_text())
    frames = iter_frames(raw_dir, split, part, only_task)
    try:
        first, _ = next(frames)
    except StopIteration:
        raise SystemExit(f"no successful frames for part={part} task={only_task}") from None
    features = {
        "observation.images.image": {"dtype": "image", "shape": first["image"].shape},
        "observation.images.wrist_image": {"dtype": "image", "shape": first["wrist_image"].shape},
        "observation.state": {"dtype": "float32", "shape": first["state"].shape},
        "actions": {"dtype": "float32", "shape": first["actions"].shape},
    }
    ds = LeRobotDataset.create(repo_id=str(out_dir), root=str(out_dir), fps=fps, features=features)
    current_task, n = None, 0
    for frame, task in iter_frames(raw_dir, split, part, only_task):
        if current_task not in (None, task):
            ds.save_episode()
        current_task = task
        ds.add_frame(
            {f"observation.images.{k}": frame[k] for k in ("image", "wrist_image")}
            | {"observation.state": frame["state"], "actions": frame["actions"]},
            task=task,
        )
        n += 1
    ds.save_episode()
    logger.info("lerobot dataset written: %s part=%s (%d frames)", out_dir, part, n)


def build_per_task_datasets(raw_dir, split_path, out_root, part="train", fps=20):
    """ACT input layout: ``<out_root>/task_<id>/`` dataset + ``prompt.txt``."""
    split = yaml.safe_load(pathlib.Path(split_path).read_text())
    for task_key, parts in sorted(split.items()):
        task_dir = pathlib.Path(out_root) / task_key
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "prompt.txt").write_text(str(parts["task_name"]) + "\n")
        build_lerobot_dataset(raw_dir, split_path, task_dir / "dataset", part, task_key, fps)


def emit_val_inits(split_path: str, pool_init_dir: str, out_dir: str) -> None:
    """Per-task ``.init`` files holding only the val inits (student-val rollouts)."""
    import torch

    split = yaml.safe_load(pathlib.Path(split_path).read_text())
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for _task_key, parts in sorted(split.items()):
        stem = parts["init_stem"]
        pool = torch.load(pathlib.Path(pool_init_dir) / f"{stem}.init", weights_only=False)
        val_states = np.asarray(pool)[parts["val"]]
        torch.save(val_states, out / f"{stem}.init")
    logger.info("val init files written: %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-init-dir", default=None)
    parser.add_argument("--raw-dir", default=None, help="distill_raw/<suite> root")
    parser.add_argument("--pool-init-dir", default=None, help="db_init/libero/<suite>")
    parser.add_argument("--cache-init-dir", default=None, help="db_init/libero_cache/<suite>")
    parser.add_argument("--split-out", default=None, help="emit constrained train/val split yaml")
    parser.add_argument("--split", default=None, help="existing split yaml (build modes)")
    parser.add_argument("--dataset-out", default=None, help="combined lerobot dataset (SmolVLA)")
    parser.add_argument("--per-task-out", default=None, help="per-task ACT layout root")
    parser.add_argument("--part", default="train", choices=["train", "val"])
    parser.add_argument("--emit-val-inits", default=None, help="dir for val-only .init files")
    parser.add_argument("--stem-map", default=None,
                        help="authoritative task_name->init-stem yaml (from LIBERO benchmark)")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.check_init_dir:
        check_init_dir(args.check_init_dir)
        return
    if args.split_out:
        if not (args.raw_dir and args.pool_init_dir and args.cache_init_dir):
            raise SystemExit("--split-out needs --raw-dir --pool-init-dir --cache-init-dir")
        stem_map = (yaml.safe_load(pathlib.Path(args.stem_map).read_text())
                    if args.stem_map else None)
        emit_split(args.raw_dir, args.pool_init_dir, args.cache_init_dir,
                   args.split_out, stem_map)
        args.split = args.split_out
    if args.dataset_out:
        build_lerobot_dataset(args.raw_dir, args.split, args.dataset_out, args.part, fps=args.fps)
    if args.per_task_out:
        build_per_task_datasets(args.raw_dir, args.split, args.per_task_out, args.part, args.fps)
    if args.emit_val_inits:
        emit_val_inits(args.split, args.pool_init_dir, args.emit_val_inits)


if __name__ == "__main__":
    main()
