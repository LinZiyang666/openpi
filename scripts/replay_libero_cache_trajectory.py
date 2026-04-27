"""Map cache trajectories back to LIBERO init states, and optionally replay them.

This is intentionally a standalone utility: the old cache artifacts only store
``trajectory_id`` on each CacheEntry, while the init-state identity lives in the
source HDF5 naming convention / attrs and in the sampled ``.init`` files.

Examples
--------
Build a mapping from the 50-episode LIBERO spatial cache dataset:

    uv run python scripts/replay_libero_cache_trajectory.py map \\
        --suite libero_spatial \\
        --artifact exp/common/data/cache_artifacts/libero_spatial/cp1_mean_pool.pkl \\
        --h5-dir exp/common/data/db/libero_cache/libero_spatial \\
        --init-states-dir exp/common/data/db_init/libero_cache/libero_spatial \\
        --full-init-states-dir exp/common/data/db_init/libero/libero_spatial \\
        --num-trials-per-task 5 \\
        --out /tmp/libero_spatial_cache_init_map.json

Replay one trajectory without a policy server.  Run this in the LIBERO conda env:

    MUJOCO_GL=egl conda run -n libero_sim python scripts/replay_libero_cache_trajectory.py render \\
        --suite libero_spatial \\
        --mapping /tmp/libero_spatial_cache_init_map.json \\
        --trajectory-id episode_0004_20260410_011001_080633 \\
        --h5-dir exp/common/data/db/libero_cache/libero_spatial \\
        --out /tmp/episode_0004_replay.mp4

Render every mapped trajectory into a directory:

    conda run -n libero_sim python scripts/replay_libero_cache_trajectory.py render-all \\
        --suite libero_spatial \\
        --mapping exp/common/data/db/libero_cache/libero_spatial_init_map.json \\
        --h5-dir exp/common/data/db/libero_cache/libero_spatial \\
        --init-states-dir exp/common/data/db_init/libero_cache/libero_spatial \\
        --out-dir exp/common/data/db/libero_cache/libero_spatial_replay_videos
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

LOGGER = logging.getLogger(__name__)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
DEFAULT_NORM_STATS = "assets/pi05_libero/physical-intelligence/libero/norm_stats.json"


@dataclass
class SourceTrajectory:
    trajectory_id: str
    prompt: str = ""
    h5_path: str = ""
    entry_count: int = 0
    episode_number: int | None = None
    attrs: dict[str, Any] | None = None


@dataclass
class ReplayAction:
    action: np.ndarray
    pkl_step: int | None
    action_offset: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _as_py(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _norm(text: str) -> str:
    text = text.replace("_", " ").lower().strip()
    return " ".join(text.split())


def _episode_number_from_id(trajectory_id: str) -> int | None:
    match = re.match(r"episode_(\d+)", Path(trajectory_id).name)
    return int(match.group(1)) if match else None


def _task_name_to_prompt(task_name: str) -> str:
    return task_name.replace("_", " ")


def _load_pickle_artifact(path: Path) -> dict:
    src_dir = str(_repo_root() / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    with path.open("rb") as f:
        return pickle.load(f)


def _sources_from_artifact(path: Path | None) -> dict[str, SourceTrajectory]:
    if path is None:
        return {}
    artifact = _load_pickle_artifact(path)
    grouped: dict[str, list] = defaultdict(list)
    for entry in artifact.get("entries", []):
        trajectory_id = getattr(entry, "trajectory_id", None) or str(entry.id).split(":", 1)[0]
        grouped[str(trajectory_id)].append(entry)

    out: dict[str, SourceTrajectory] = {}
    for trajectory_id, entries in grouped.items():
        entries.sort(key=lambda e: (-1 if getattr(e, "step_idx", None) is None else int(e.step_idx), str(e.id)))
        prompt = ""
        if entries:
            prompt = str(getattr(entries[0].payload, "task_key", "") or "")
        out[trajectory_id] = SourceTrajectory(
            trajectory_id=trajectory_id,
            prompt=prompt,
            entry_count=len(entries),
            episode_number=_episode_number_from_id(trajectory_id),
        )
    return out


def _sources_from_h5_dir(h5_dir: Path | None) -> dict[str, SourceTrajectory]:
    if h5_dir is None or not h5_dir.exists():
        return {}
    out: dict[str, SourceTrajectory] = {}
    for path in sorted(h5_dir.rglob("*.h5")):
        with h5py.File(path, "r") as f:
            attrs = {k: _as_py(v) for k, v in f.attrs.items()}
            prompt = str(attrs.get("task") or attrs.get("task_name") or attrs.get("prompt") or "")
            step_count = len([k for k in f.keys() if k.startswith("step_")])
        trajectory_id = path.stem
        out[trajectory_id] = SourceTrajectory(
            trajectory_id=trajectory_id,
            prompt=prompt,
            h5_path=str(path),
            entry_count=step_count,
            episode_number=_episode_number_from_id(trajectory_id),
            attrs=attrs,
        )
    return out


def _merge_sources(*sources: dict[str, SourceTrajectory]) -> list[SourceTrajectory]:
    merged: dict[str, SourceTrajectory] = {}
    for source in sources:
        for trajectory_id, item in source.items():
            existing = merged.get(trajectory_id)
            if existing is None:
                merged[trajectory_id] = item
                continue
            if item.prompt:
                existing.prompt = item.prompt
            if item.h5_path:
                existing.h5_path = item.h5_path
            if item.entry_count:
                existing.entry_count = item.entry_count
            if item.attrs:
                existing.attrs = item.attrs
            if item.episode_number is not None:
                existing.episode_number = item.episode_number
    return sorted(merged.values(), key=lambda s: (s.episode_number is None, s.episode_number or -1, s.trajectory_id))


def _load_libero_task_index(suite_name: str) -> tuple[dict[int, Any], dict[str, int]]:
    try:
        from libero.libero import benchmark  # type: ignore[import]
    except Exception as exc:
        LOGGER.info("LIBERO import unavailable; falling back to filename/prompt matching: %s", exc)
        return {}, {}

    suite = benchmark.get_benchmark_dict()[suite_name]()
    by_id: dict[int, Any] = {}
    prompt_to_id: dict[str, int] = {}
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        by_id[task_id] = task
        prompt_to_id[_norm(getattr(task, "language", ""))] = task_id
        prompt_to_id[_norm(getattr(task, "name", ""))] = task_id
    return by_id, prompt_to_id


def _init_files_by_prompt(init_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not init_dir.exists():
        return out
    for path in sorted(init_dir.glob("*.init")) + sorted(init_dir.glob("*.pruned_init")):
        suffix = ".pruned_init" if path.name.endswith(".pruned_init") else ".init"
        task_name = path.name[: -len(suffix)]
        out[_norm(_task_name_to_prompt(task_name))] = task_name
    return out


def _resolve_task_name(
    *,
    task_id: int | None,
    prompt: str,
    libero_tasks: dict[int, Any],
    prompt_to_task_id: dict[str, int],
    init_prompt_index: dict[str, str],
) -> tuple[int | None, str]:
    if task_id is None and prompt:
        task_id = prompt_to_task_id.get(_norm(prompt))
    if task_id is not None and task_id in libero_tasks:
        return task_id, str(libero_tasks[task_id].name)
    if prompt:
        task_name = init_prompt_index.get(_norm(prompt))
        if task_name:
            return task_id, task_name
    return task_id, ""


def _load_init_file(path: Path) -> np.ndarray:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - only happens outside project envs
        raise RuntimeError("torch is required to read LIBERO .init files") from exc
    return np.asarray(torch.load(path, weights_only=False))


def _find_orig_init_idx(init_path: Path, full_init_path: Path, subset_idx: int) -> int | None:
    if not init_path.exists() or not full_init_path.exists():
        return None
    subset = _load_init_file(init_path)
    full = _load_init_file(full_init_path)
    if subset_idx < 0 or subset_idx >= len(subset):
        return None
    row = np.asarray(subset[subset_idx])
    matches = np.where(np.all(np.isclose(np.asarray(full), row, atol=1e-7), axis=1))[0]
    if len(matches) != 1:
        return None
    return int(matches[0])


def _choose_subset_idx(source: SourceTrajectory, num_trials_per_task: int) -> int | None:
    attrs = source.attrs or {}
    if "init_state_idx" in attrs:
        return int(attrs["init_state_idx"])
    if source.episode_number is not None and num_trials_per_task > 0:
        return int(source.episode_number % num_trials_per_task)
    return None


def _choose_task_id(source: SourceTrajectory, num_trials_per_task: int) -> int | None:
    attrs = source.attrs or {}
    if "task_id" in attrs:
        return int(attrs["task_id"])
    if source.episode_number is not None and num_trials_per_task > 0:
        return int(source.episode_number // num_trials_per_task)
    return None


def build_mapping(
    *,
    suite: str,
    artifact: Path | None,
    h5_dir: Path | None,
    init_states_dir: Path,
    full_init_states_dir: Path | None,
    num_trials_per_task: int,
) -> list[dict[str, Any]]:
    artifact_sources = _sources_from_artifact(artifact)
    h5_sources = _sources_from_h5_dir(h5_dir)
    sources = _merge_sources(artifact_sources, h5_sources)

    libero_tasks, prompt_to_task_id = _load_libero_task_index(suite)
    init_prompt_index = _init_files_by_prompt(init_states_dir)

    records: list[dict[str, Any]] = []
    for source in sources:
        task_id = _choose_task_id(source, num_trials_per_task)
        task_id, task_name = _resolve_task_name(
            task_id=task_id,
            prompt=source.prompt,
            libero_tasks=libero_tasks,
            prompt_to_task_id=prompt_to_task_id,
            init_prompt_index=init_prompt_index,
        )
        subset_idx = _choose_subset_idx(source, num_trials_per_task)

        init_path = init_states_dir / f"{task_name}.pruned_init"
        if not init_path.exists():
            init_path = init_states_dir / f"{task_name}.init"
        full_init_path = None
        if full_init_states_dir is not None and task_name:
            candidate = full_init_states_dir / f"{task_name}.init"
            if candidate.exists():
                full_init_path = candidate

        attrs = source.attrs or {}
        orig_idx = int(attrs["orig_init_state_idx"]) if "orig_init_state_idx" in attrs else None
        if orig_idx is None and subset_idx is not None and full_init_path is not None:
            orig_idx = _find_orig_init_idx(init_path, full_init_path, subset_idx)

        records.append(
            {
                "suite": suite,
                "trajectory_id": source.trajectory_id,
                "h5_path": source.h5_path,
                "task_id": task_id,
                "task_name": task_name,
                "prompt": source.prompt,
                "episode_number": source.episode_number,
                "subset_init_state_idx": subset_idx,
                "orig_init_state_idx": orig_idx,
                "init_path": str(init_path) if init_path.exists() else "",
                "full_init_path": str(full_init_path) if full_init_path is not None else "",
                "entry_count": source.entry_count,
                "attrs": attrs,
            }
        )
    return records


def _step_sort_key(name: str) -> tuple[int, str]:
    suffix = name.split("_", 1)[1] if "_" in name else ""
    return (int(suffix) if suffix.isdigit() else 10**9, name)


def _load_action_quantiles(norm_stats_path: Path | None) -> tuple[np.ndarray, np.ndarray] | None:
    if norm_stats_path is None:
        return None
    data = json.loads(norm_stats_path.read_text())
    stats = data.get("norm_stats", data)
    action_stats = stats["actions"]
    return (
        np.asarray(action_stats["q01"], dtype=np.float32),
        np.asarray(action_stats["q99"], dtype=np.float32),
    )


def _decode_model_actions(action_chunk: np.ndarray, quantiles: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    """Mirror Policy output transforms for pi05_libero model-space actions."""
    chunk = np.asarray(action_chunk, dtype=np.float32)
    if quantiles is None:
        return chunk[..., :7]
    q01, q99 = quantiles
    dim = q01.shape[-1]
    decoded = chunk.copy()
    decoded[..., :dim] = (decoded[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    return decoded[..., :7]


def _load_h5_actions(
    h5_path: Path,
    replan_steps: int,
    action_quantiles: tuple[np.ndarray, np.ndarray] | None,
) -> list[ReplayAction]:
    actions: list[ReplayAction] = []
    with h5py.File(h5_path, "r") as f:
        for step_name in sorted((k for k in f.keys() if k.startswith("step_")), key=_step_sort_key):
            step_idx, _ = _step_sort_key(step_name)
            pkl_step = None if step_idx == 10**9 else step_idx
            group = f[step_name]
            if "executed_actions" in group:
                chunk = np.asarray(group["executed_actions"], dtype=np.float32)
            elif "clean_action" in group:
                chunk = _decode_model_actions(
                    np.asarray(group["clean_action"], dtype=np.float32),
                    action_quantiles,
                )[:replan_steps]
            elif "env_action_chunk" in group:
                chunk = np.asarray(group["env_action_chunk"], dtype=np.float32)[:replan_steps, :7]
            else:
                continue
            if chunk.ndim == 1:
                chunk = chunk[None, :]
            for action_offset, action in enumerate(chunk):
                actions.append(
                    ReplayAction(
                        action=np.asarray(action[:7], dtype=np.float32),
                        pkl_step=pkl_step,
                        action_offset=action_offset,
                    )
                )
    return actions


def _load_artifact_actions(
    artifact_path: Path,
    trajectory_id: str,
    replan_steps: int,
    action_quantiles: tuple[np.ndarray, np.ndarray] | None,
) -> list[ReplayAction]:
    artifact = _load_pickle_artifact(artifact_path)
    entries = [
        e for e in artifact.get("entries", [])
        if (getattr(e, "trajectory_id", None) or str(e.id).split(":", 1)[0]) == trajectory_id
    ]
    entries.sort(key=lambda e: (-1 if getattr(e, "step_idx", None) is None else int(e.step_idx), str(e.id)))
    actions: list[ReplayAction] = []
    for entry in entries:
        pkl_step = getattr(entry, "step_idx", None)
        pkl_step = None if pkl_step is None else int(pkl_step)
        chunk = _decode_model_actions(
            np.asarray(entry.payload.action_chunk, dtype=np.float32),
            action_quantiles,
        )[:replan_steps]
        for action_offset, action in enumerate(chunk):
            actions.append(
                ReplayAction(
                    action=np.asarray(action[:7], dtype=np.float32),
                    pkl_step=pkl_step,
                    action_offset=action_offset,
                )
            )
    return actions


def _frame_from_obs(obs: dict[str, Any]) -> np.ndarray:
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])


def _annotate_frame(
    frame: np.ndarray,
    *,
    trajectory_id: str,
    pkl_step: int | None,
    action_offset: int | None = None,
) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    label = "pkl step: init" if pkl_step is None else f"pkl step: {pkl_step:04d}"
    if action_offset is not None:
        label += f"  action: {action_offset + 1}"
    label += f"  {trajectory_id}"

    image = Image.fromarray(frame.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        text_w, text_h = draw.textsize(label, font=font)

    pad = 6
    draw.rectangle((6, 6, 6 + text_w + 2 * pad, 6 + text_h + 2 * pad), fill=(0, 0, 0))
    draw.text((6 + pad, 6 + pad), label, fill=(255, 255, 255), font=font)
    return np.asarray(image)


def _render_record(
    record: dict[str, Any],
    *,
    artifact: Path | None,
    h5_dir: Path | None,
    out_path: Path,
    resolution: int,
    seed: int,
    num_steps_wait: int,
    replan_steps: int,
    fps: int,
    max_action_steps: int | None,
    stop_on_done: bool,
    action_quantiles: tuple[np.ndarray, np.ndarray] | None,
) -> dict[str, Any]:
    _prepare_headless_render_env()

    from imageio import v2 as imageio
    from libero.libero import get_libero_path  # type: ignore[import]
    from libero.libero.benchmark import get_benchmark_dict  # type: ignore[import]
    from libero.libero.envs import OffScreenRenderEnv  # type: ignore[import]

    task_id = record.get("task_id")
    subset_idx = record.get("subset_init_state_idx")
    if task_id is None or subset_idx is None:
        raise ValueError(f"Mapping record lacks task_id/subset_init_state_idx: {record}")
    init_path = Path(record.get("init_path") or "")
    if not init_path.exists():
        raise FileNotFoundError(f"init_path missing for {record['trajectory_id']}: {init_path}")

    if artifact is not None:
        actions = _load_artifact_actions(
            artifact,
            record["trajectory_id"],
            replan_steps,
            action_quantiles,
        )
    else:
        h5_path = Path(record.get("h5_path") or "")
        if not h5_path.exists() and h5_dir is not None:
            h5_path = h5_dir / f"{record['trajectory_id']}.h5"
        if not h5_path.exists():
            raise FileNotFoundError(f"HDF5 source missing for {record['trajectory_id']}: {h5_path}")
        actions = _load_h5_actions(h5_path, replan_steps, action_quantiles)
    if max_action_steps is not None:
        actions = actions[:max_action_steps]
    if not actions:
        raise ValueError(f"No actions found for {record['trajectory_id']}")

    suite = get_benchmark_dict()[record["suite"]]()
    task = suite.get_task(int(task_id))
    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    init_states = _load_init_file(init_path)
    initial_state = init_states[int(subset_idx)]

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)

    frames: list[np.ndarray] = []
    done = False
    try:
        env.reset()
        obs = env.set_init_state(initial_state)
        frames.append(
            _annotate_frame(
                _frame_from_obs(obs),
                trajectory_id=record["trajectory_id"],
                pkl_step=None,
            )
        )
        for _ in range(num_steps_wait):
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            frames.append(
                _annotate_frame(
                    _frame_from_obs(obs),
                    trajectory_id=record["trajectory_id"],
                    pkl_step=None,
                )
            )

        executed = 0
        for replay_action in actions:
            obs, _, done, _ = env.step(replay_action.action.tolist())
            frames.append(
                _annotate_frame(
                    _frame_from_obs(obs),
                    trajectory_id=record["trajectory_id"],
                    pkl_step=replay_action.pkl_step,
                    action_offset=replay_action.action_offset,
                )
            )
            executed += 1
            if done and stop_on_done:
                break
    finally:
        env.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out_path), frames, fps=fps)
    return {
        "out_path": str(out_path),
        "frames": len(frames),
        "fps": int(fps),
        "duration_sec": len(frames) / float(fps),
        "executed_actions": executed,
        "done": bool(done),
        "stop_on_done": bool(stop_on_done),
    }


def _prepare_headless_render_env() -> None:
    """Set writable cache dirs before importing robosuite / matplotlib / numba."""
    defaults = {
        "MUJOCO_GL": "egl",
        "NUMBA_CACHE_DIR": "/tmp/numba_cache",
        "MPLCONFIGDIR": "/tmp/matplotlib",
        "MESA_SHADER_CACHE_DIR": "/tmp/mesa_shader_cache",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
        if key.endswith("_DIR") or key in {"MPLCONFIGDIR"}:
            Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def _print_mapping_summary(records: list[dict[str, Any]]) -> None:
    unresolved = [r for r in records if r["task_id"] is None or r["subset_init_state_idx"] is None or not r["init_path"]]
    by_task = Counter(r["task_id"] for r in records)
    print(f"mapped trajectories: {len(records)}")
    print(f"by task: {dict(sorted(by_task.items(), key=lambda kv: (-1 if kv[0] is None else kv[0])))}")
    print(f"unresolved: {len(unresolved)}")
    for row in records[:8]:
        print(
            f"  {row['trajectory_id']} -> task={row['task_id']} "
            f"subset={row['subset_init_state_idx']} orig={row['orig_init_state_idx']} "
            f"entries={row['entry_count']}"
        )
    if unresolved:
        print("first unresolved:")
        for row in unresolved[:5]:
            print(f"  {row['trajectory_id']}: prompt={row['prompt']!r}, init_path={row['init_path']!r}")


def _cmd_map(args: argparse.Namespace) -> None:
    records = build_mapping(
        suite=args.suite,
        artifact=Path(args.artifact) if args.artifact else None,
        h5_dir=Path(args.h5_dir) if args.h5_dir else None,
        init_states_dir=Path(args.init_states_dir),
        full_init_states_dir=Path(args.full_init_states_dir) if args.full_init_states_dir else None,
        num_trials_per_task=args.num_trials_per_task,
    )
    _print_mapping_summary(records)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(records, indent=2))
        print(f"wrote mapping: {out}")


def _load_mapping(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.mapping:
        return json.loads(Path(args.mapping).read_text())
    return build_mapping(
        suite=args.suite,
        artifact=Path(args.artifact) if args.artifact else None,
        h5_dir=Path(args.h5_dir) if args.h5_dir else None,
        init_states_dir=Path(args.init_states_dir),
        full_init_states_dir=Path(args.full_init_states_dir) if args.full_init_states_dir else None,
        num_trials_per_task=args.num_trials_per_task,
    )


def _cmd_render(args: argparse.Namespace) -> None:
    records = _load_mapping(args)
    matches = [r for r in records if r["trajectory_id"] == args.trajectory_id]
    if not matches:
        raise KeyError(f"trajectory_id not found in mapping: {args.trajectory_id}")
    out_path = Path(args.out) if args.out else Path("renders") / f"{args.trajectory_id}.mp4"
    action_quantiles = _load_action_quantiles(Path(args.norm_stats)) if args.norm_stats else None
    result = _render_record(
        matches[0],
        artifact=Path(args.artifact) if args.artifact else None,
        h5_dir=Path(args.h5_dir) if args.h5_dir else None,
        out_path=out_path,
        resolution=args.resolution,
        seed=args.seed,
        num_steps_wait=args.num_steps_wait,
        replan_steps=args.replan_steps,
        fps=args.fps,
        max_action_steps=args.max_action_steps,
        stop_on_done=args.stop_on_done and not args.ignore_done,
        action_quantiles=action_quantiles,
    )
    print(json.dumps(result, indent=2))


def _cmd_render_all(args: argparse.Namespace) -> None:
    records = _load_mapping(args)
    if args.limit is not None:
        records = records[: args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    action_quantiles = _load_action_quantiles(Path(args.norm_stats)) if args.norm_stats else None

    manifest: list[dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        out_path = out_dir / f"{record['trajectory_id']}.mp4"
        if args.skip_existing and out_path.exists():
            result = {
                "out_path": str(out_path),
                "trajectory_id": record["trajectory_id"],
                "skipped": True,
            }
            print(f"[{idx}/{len(records)}] SKIP {out_path}")
            manifest.append(result)
            continue

        print(f"[{idx}/{len(records)}] render {record['trajectory_id']} -> {out_path}", flush=True)
        result = _render_record(
            record,
            artifact=Path(args.artifact) if args.artifact else None,
            h5_dir=Path(args.h5_dir) if args.h5_dir else None,
            out_path=out_path,
            resolution=args.resolution,
            seed=args.seed,
            num_steps_wait=args.num_steps_wait,
            replan_steps=args.replan_steps,
            fps=args.fps,
            max_action_steps=args.max_action_steps,
            stop_on_done=args.stop_on_done and not args.ignore_done,
            action_quantiles=action_quantiles,
        )
        result["trajectory_id"] = record["trajectory_id"]
        result["task_id"] = record.get("task_id")
        result["subset_init_state_idx"] = record.get("subset_init_state_idx")
        result["orig_init_state_idx"] = record.get("orig_init_state_idx")
        manifest.append(result)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote manifest: {manifest_path}")


def _add_common_mapping_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suite", required=True, help="LIBERO suite, e.g. libero_spatial")
    parser.add_argument("--artifact", default="", help="Optional cache artifact .pkl")
    parser.add_argument("--h5-dir", default="", help="Optional source HDF5 directory")
    parser.add_argument("--init-states-dir", required=True, help="Sample/subset init states directory")
    parser.add_argument("--full-init-states-dir", default="", help="Full init states directory for orig-index matching")
    parser.add_argument("--num-trials-per-task", type=int, default=5)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    map_p = sub.add_parser("map", help="Write/print trajectory -> init-state mapping")
    _add_common_mapping_args(map_p)
    map_p.add_argument("--out", default="", help="Optional JSON output path")
    map_p.set_defaults(func=_cmd_map)

    render_p = sub.add_parser("render", help="Replay one mapped trajectory in LIBERO and save mp4")
    _add_common_mapping_args(render_p)
    render_p.add_argument("--mapping", default="", help="Mapping JSON from the map subcommand")
    render_p.add_argument("--trajectory-id", required=True)
    render_p.add_argument("--out", default="", help="Output mp4 path")
    render_p.add_argument("--resolution", type=int, default=256)
    render_p.add_argument("--seed", type=int, default=7)
    render_p.add_argument("--num-steps-wait", type=int, default=10)
    render_p.add_argument("--replan-steps", type=int, default=5)
    render_p.add_argument("--fps", type=int, default=10)
    render_p.add_argument("--max-action-steps", type=int, default=None)
    render_p.add_argument(
        "--norm-stats",
        default=DEFAULT_NORM_STATS,
        help="pi05_libero norm_stats.json used to decode model-space pkl actions; empty string disables decoding",
    )
    render_p.add_argument(
        "--stop-on-done",
        action="store_true",
        help="Stop when LIBERO reports success/done; default is to replay all pkl steps.",
    )
    render_p.add_argument("--ignore-done", action="store_true", help=argparse.SUPPRESS)
    render_p.set_defaults(func=_cmd_render)

    render_all_p = sub.add_parser("render-all", help="Replay every mapped trajectory and save mp4 files")
    _add_common_mapping_args(render_all_p)
    render_all_p.add_argument("--mapping", default="", help="Mapping JSON from the map subcommand")
    render_all_p.add_argument(
        "--out-dir",
        default="exp/common/data/db/libero_cache/libero_spatial_replay_videos",
        help="Directory for rendered mp4 files",
    )
    render_all_p.add_argument("--resolution", type=int, default=256)
    render_all_p.add_argument("--seed", type=int, default=7)
    render_all_p.add_argument("--num-steps-wait", type=int, default=10)
    render_all_p.add_argument("--replan-steps", type=int, default=5)
    render_all_p.add_argument("--fps", type=int, default=10)
    render_all_p.add_argument("--max-action-steps", type=int, default=None)
    render_all_p.add_argument(
        "--norm-stats",
        default=DEFAULT_NORM_STATS,
        help="pi05_libero norm_stats.json used to decode model-space pkl actions; empty string disables decoding",
    )
    render_all_p.add_argument(
        "--stop-on-done",
        action="store_true",
        help="Stop when LIBERO reports success/done; default is to replay all pkl steps.",
    )
    render_all_p.add_argument("--ignore-done", action="store_true", help=argparse.SUPPRESS)
    render_all_p.add_argument("--skip-existing", action="store_true")
    render_all_p.add_argument("--limit", type=int, default=None, help="Debug limit; omit to render all")
    render_all_p.set_defaults(func=_cmd_render_all)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
