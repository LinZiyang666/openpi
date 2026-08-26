"""Library-side bucket -> instruction-variant map for ws2 attribution (plan §3-W7).

Loads the full704 text-IVF artifact, groups entries into prompt buckets by the
backend's exact rule (byte-identical ``prompt_emb`` float32 buffers; bucket
indices follow the backend's ascending-key order), picks one representative
episode per bucket, and — unless ``--skip-replay`` — recovers each
representative's ground-truth instruction by replaying its collection seed
(``seed = episode index``; collection used base_seed=0, the T5b id-is-seed
discipline) through a bare ``env.reset``. No policy runs.

Env discipline: representatives are resolved task by task, ONE env alive at a
time, closed before the next task's env is built (evict-1) — building ~13
kitchens in one process exhausts the EGL framebuffer (seed-anatomy report).

Output ``bucket_variants.json``::

    {"artifact_sha256": ..., "n_entries": ..., "buckets": [
        {"bucket_index": i, "tasks": [...], "n_entries": n,
         "representative": {"trajectory": relpath, "seed": s,
                            "prompt": str|None, "status": "resolved|unresolved|skipped"},
         "ambiguous": bool},   # >1 task in one bucket (should not happen)
        ...]}

Entry ids are ``<trajectory-relpath>:<step>`` (offline builder, relpath id
mode); the ws2 join maps ``winner_id`` -> trajectory -> bucket through the
same table. Buckets whose replay fails are marked ``unresolved`` — kept and
counted, never dropped silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import pickle
import re
from typing import Any

# Collected files are ``<Task>/episode_{idx:04d}_a{attempt:02d}.h5``
# (episode_runner's episode_name), and the artifact's relpath ids are those
# paths with the suffix stripped — so the attempt tail is part of the id and
# the pattern must tolerate it. The seed is the episode index regardless of
# which attempt finally succeeded (collection seeds are id-is-seed).
_EPISODE_RE = re.compile(r"episode_(\d+)(?:_a\d+)?$")


def eval_env_kwargs(teacher: str) -> dict[str, Any]:
    """The extra ``gym.make`` kwargs the REAL eval runner passes for this teacher.

    Read from the production adapter (``ADAPTERS[teacher]().env_kwargs()``),
    never re-declared here: GR00T renders at its own resolution and a replay
    built from a second copy of those numbers could drift from the environment
    the library was actually collected in.
    """
    from exp.robocasa365.episode_runner import ADAPTERS

    try:
        factory = ADAPTERS[teacher]
    except KeyError:
        raise SystemExit(f"unknown teacher {teacher!r}; expected one of {sorted(ADAPTERS)}") from None
    return dict(factory().env_kwargs())


def env_provenance(layout: int, style: int, teacher: str) -> dict[str, Any]:
    """What a replayed prompt is only valid against: sim revision + env args.

    The instruction text a seed produces is a function of the robocasa build
    and of every argument ``default_gym_make`` pins, so the map is unreadable
    later without them recorded next to the artifact hash.
    """
    import subprocess

    import robocasa

    root = pathlib.Path(robocasa.__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001 - recorded, never guessed
        commit = f"unavailable: {exc}"
    return {
        "robocasa_commit": commit,
        "robocasa_path": str(root),
        "robocasa_version": getattr(robocasa, "__version__", None),
        # Mirrors default_gym_make plus the teacher adapter's own extras, which
        # is what the eval runner actually builds (episode_runner _ensure_env).
        "teacher": teacher,
        "env_kwargs": {
            "env_id_template": "robocasa/<task_name>",
            "split": None,
            "obj_instance_split": "target",
            "layout_and_style_ids": [[layout, style]],
            **eval_env_kwargs(teacher),
        },
    }


def bucket_entries(entries: list[Any]) -> list[dict[str, Any]]:
    """Group entries by exact prompt_emb bytes; index in ascending key order."""
    buckets: dict[bytes, list[Any]] = {}
    for entry in entries:
        vec = entry.query_keys.get("prompt_emb")
        if vec is None:
            raise SystemExit(f"entry {entry.id!r} has no prompt_emb — not a text-IVF artifact")
        buckets.setdefault(vec.float().contiguous().numpy().tobytes(), []).append(entry)
    out = []
    for index, key in enumerate(sorted(buckets)):
        members = buckets[key]
        trajectories = sorted({e.id.rsplit(":", 1)[0] for e in members})
        tasks = sorted({t.split("/")[0] for t in trajectories})
        out.append({
            "bucket_index": index,
            "tasks": tasks,
            "n_entries": len(members),
            "trajectories": trajectories,
            "ambiguous": len(tasks) > 1,
        })
    return out


def representative_of(bucket: dict[str, Any]) -> dict[str, Any]:
    """Pick one member trajectory per bucket and recover its collection seed."""
    trajectory = bucket["trajectories"][0]  # smallest relpath = deterministic
    m = _EPISODE_RE.search(trajectory)
    if not m:
        return {"trajectory": trajectory, "seed": None, "prompt": None,
                "object_class": None, "status": "unresolved"}
    return {"trajectory": trajectory, "seed": int(m.group(1)), "prompt": None,
            "object_class": None, "status": "skipped"}


def object_class_of(env: Any) -> Any:
    """The task object's category for this reset, or None when unexposed.

    RoboCasa surfaces the sampled object registry on the unwrapped env; the
    attribute set differs per task family, so this reads what is there and
    records ``None`` rather than guessing a shape.
    """
    inner = getattr(env, "unwrapped", env)
    objects = getattr(inner, "object_cfgs", None)
    if not objects:
        return None
    out = []
    for cfg in objects:
        info = cfg.get("info") if isinstance(cfg, dict) else None
        name = cfg.get("name") if isinstance(cfg, dict) else None
        out.append({"name": name, "cat": (info or {}).get("cat")})
    return out or None


def resolve_prompts(
    buckets: list[dict[str, Any]], *, layout: int, style: int, teacher: str,
) -> None:
    """Replay each representative's seed for its prompt text. Evict-1 envs.

    The env is built exactly as the eval runner builds it — same
    ``default_gym_make`` plus the same adapter kwargs — so a recovered
    instruction is the one that episode really ran under.
    """
    from exp.robocasa365.episode_runner import PROMPT_SOURCE_KEY, default_gym_make

    extra = eval_env_kwargs(teacher)

    by_task: dict[str, list[dict[str, Any]]] = {}
    for bucket in buckets:
        rep = bucket["representative"]
        if rep["seed"] is None:
            continue
        by_task.setdefault(rep["trajectory"].split("/")[0], []).append(rep)
    for task_name in sorted(by_task):
        try:
            env = default_gym_make(task_name, layout, style, **extra)
        except Exception as exc:  # noqa: BLE001 - recorded, never silent
            # One unbuildable task must not abort the map: its buckets stay
            # unresolved and counted, which is this tool's stated policy.
            for rep in by_task[task_name]:
                rep["status"] = "unresolved"
                rep["error"] = f"env build failed: {exc}"
            continue
        try:
            for rep in by_task[task_name]:
                try:
                    obs, _ = env.reset(seed=rep["seed"])
                    rep["prompt"] = str(obs[PROMPT_SOURCE_KEY])
                    rep["object_class"] = object_class_of(env)
                    rep["status"] = "resolved"
                except Exception as exc:  # noqa: BLE001 - recorded, never silent
                    rep["status"] = "unresolved"
                    rep["error"] = str(exc)
        finally:
            env.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", required=True, help="full704 text-IVF pkl")
    ap.add_argument("--out", required=True, help="bucket_variants.json path")
    ap.add_argument("--layout", type=int, default=1)
    ap.add_argument("--style", type=int, default=1)
    ap.add_argument("--teacher", default="groot_tp",
                    help="whose adapter env kwargs the replay must reuse")
    ap.add_argument("--skip-replay", action="store_true",
                    help="map buckets only (no sim available); representatives stay 'skipped'")
    args = ap.parse_args()

    artifact_path = pathlib.Path(args.artifact)
    sha = hashlib.sha256()
    with artifact_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            sha.update(chunk)
    with artifact_path.open("rb") as fh:
        data = pickle.load(fh)

    buckets = bucket_entries(data["entries"])
    for bucket in buckets:
        bucket["representative"] = representative_of(bucket)
    if not args.skip_replay:
        resolve_prompts(buckets, layout=args.layout, style=args.style, teacher=args.teacher)

    # The ws2 join walks winner_id -> trajectory -> bucket without reloading
    # the 20GB artifact, so the trajectory->bucket map ships in the json
    # (~704 rows) while per-bucket lists collapse to counts.
    trajectory_to_bucket: dict[str, int] = {}
    for bucket in buckets:
        for trajectory in bucket["trajectories"]:
            trajectory_to_bucket[trajectory] = bucket["bucket_index"]
        bucket["n_trajectories"] = len(bucket.pop("trajectories"))

    unresolved = sum(1 for b in buckets if b["representative"]["status"] == "unresolved")
    payload = {
        "artifact": str(artifact_path),
        "artifact_sha256": sha.hexdigest(),
        "prompt_pool": data.get("prompt_pool"),
        "n_entries": len(data["entries"]),
        "n_buckets": len(buckets),
        "n_unresolved": unresolved,
        "layout": args.layout,
        "style": args.style,
        "provenance": (
            env_provenance(args.layout, args.style, args.teacher) if not args.skip_replay
            else {"robocasa_commit": None, "teacher": args.teacher,
                  "note": "--skip-replay: no env was built"}
        ),
        "trajectory_to_bucket": trajectory_to_bucket,
        "buckets": buckets,
    }
    pathlib.Path(args.out).write_text(json.dumps(payload, indent=1) + "\n")
    print(
        f"[bucket_variants] {len(buckets)} buckets ({unresolved} unresolved, "
        f"{sum(1 for b in buckets if b['ambiguous'])} ambiguous) -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
