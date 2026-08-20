"""Per-arm execution cost — the single authority for the reward's cost term (§3.10 / D4).

``cost(arm)`` is the **batch=1 GPU time** of one execution on that arm,
normalized so ``cost(teacher) == 1``.

Why this cannot be measured from a client
-----------------------------------------
A websocket round trip is wall-clock: it folds in serialization, network and
queueing, and it varies with the fleet rather than with the model. Worse, a
client process cannot synchronize the *server's* CUDA context, so even the
compute part of that number is only "time until the kernels were enqueued".
Each arm is therefore measured **on the host that owns its compute**, in
process, with CUDA events around the actual execution path:

  - ``teacher`` — the routed server: the full staged Pi0.5 forward
    (stage1 → stage2 → stage3), which is exactly what a MISS runs;
  - ``student`` — the sidecar host: one forward of the sidecar policy, which is
    what a payloadless FULL_HIT runs;
  - ``cache``  — the routed server: the real replay path (payload fetch through
    ``PayloadView`` + ``broadcast_action`` + output transform), i.e. the cost of
    *not* running a model. It is small but it is not zero, and measuring it as
    zero would flatter the cache arm in every reward.

Run once per arm on its own host, then ``combine`` the per-arm records into the
normalized artifact the trainer consumes. Every record carries provenance
(host, device, torch version, timing method) so a launch gate can tell a real
measurement from a placeholder — ``launch_gates.check_launch_gates`` refuses to
start a run whose costs are not marked ``gpu_timed``.

Usage::

    # on the routed server
    uv run exp/rl_router/microbench_cost.py arm --arm teacher --yaml <arm.yaml> \\
        --checkpoint <pi05 ckpt> --out costs/teacher.json
    uv run exp/rl_router/microbench_cost.py arm --arm cache --yaml <arm.yaml> \\
        --out costs/cache.json
    # on the sidecar host
    # in the SIDECAR's own venv, on the sidecar host (never an RPC wrapper)
    uv run exp/rl_router/microbench_cost.py arm --arm student \\
        --student-kind act --student-checkpoint <act manifest> --out costs/student.json
    # anywhere
    uv run exp/rl_router/microbench_cost.py combine --inputs costs/*.json \\
        --out exp/rl_router/data/arm_costs.json
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import platform
import statistics
import time
from typing import Callable, Optional

import numpy as np

# Discarded before timing: the first calls pay one-off compilation, cuDNN
# autotuning and allocator growth that no steady-state episode ever pays again.
WARMUP_CALLS = 5


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def gpu_timed(fn: Callable[[], object], *, repeats: int, in_process: bool = True) -> dict:
    """Time ``fn`` with CUDA events; fall back to a synchronized host timer.

    CUDA events are recorded on the same stream as the work, so the measurement
    is device time and does not include host-side launch latency. Without a GPU
    (unit tests) the fallback still synchronizes, and the record says so — the
    launch gate reads ``gpu_timed`` and refuses a run whose costs were taken
    without a device.
    """
    import torch

    # ``in_process`` is not a formality: CUDA events observe THIS process's
    # context, so timing a call that merely sends an RPC to another process
    # records the local (idle) stream and would report a near-zero GPU time for
    # work that never ran here. The flag rides into the artifact's provenance and
    # the launch gate refuses a cost that was not measured in-process.
    has_cuda = torch.cuda.is_available() and in_process
    for _ in range(WARMUP_CALLS):
        fn()
    if has_cuda:
        torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(repeats):
        if has_cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) / 1000.0)
        else:
            t0 = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - t0)
    return {
        "n": len(samples),
        "mean_s": statistics.fmean(samples),
        "median_s": statistics.median(samples),
        "p90_s": float(np.percentile(samples, 90)),
        "min_s": min(samples),
        "gpu_timed": bool(has_cuda),
        "in_process": bool(in_process),
        "method": ("cuda_event" if has_cuda else
                   ("perf_counter_out_of_process" if not in_process else "perf_counter_no_cuda")),
        "device": torch.cuda.get_device_name(0) if has_cuda else "cpu",
        "torch": torch.__version__,
        "host": platform.node(),
    }


def wall_clock(fn: Callable[[], object], *, repeats: int) -> dict:
    """Host wall clock — reported alongside, never fed into the reward (D4)."""
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return {"mean_s": statistics.fmean(samples), "n": len(samples)}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_costs(raw: dict[str, dict]) -> dict[str, float]:
    """Scale measured means so the teacher arm costs exactly 1.0.

    Normalizing rather than using seconds keeps λ comparable across machines: a
    faster GPU shifts every arm together and leaves the trade-off the reward
    encodes unchanged.
    """
    if "teacher" not in raw:
        raise ValueError("the teacher arm must be measured — it defines the unit")
    base = raw["teacher"]["mean_s"]
    if base <= 0:
        raise ValueError(f"teacher mean time {base} is not positive")
    return {arm: stats["mean_s"] / base for arm, stats in raw.items()}


def combine_records(records: dict[str, dict]) -> dict:
    """Fold per-arm records into the artifact the trainer and gates consume."""
    raw = {arm: rec["gpu"] for arm, rec in records.items()}
    return {
        "normalized_costs": normalize_costs(raw),
        "raw_gpu_seconds": raw,
        "wall_clock_seconds": {arm: rec.get("wall") for arm, rec in records.items()},
        "router_cpu_forward": {
            arm: rec.get("router_cpu") for arm, rec in records.items()
            if rec.get("router_cpu")
        },
        "provenance": {
            arm: {
                "gpu_timed": bool(rec["gpu"].get("gpu_timed")),
                "method": rec["gpu"].get("method"),
                "device": rec["gpu"].get("device"),
                "host": rec["gpu"].get("host"),
                "torch": rec["gpu"].get("torch"),
                "path": rec.get("path"),
                "in_process": bool(rec["gpu"].get("in_process")),
                "task_prompts": rec.get("task_prompts"),
                "per_task_gpu_mean_s": rec.get("per_task_gpu_mean_s"),
            }
            for arm, rec in records.items()
        },
        "note": (
            "normalized to teacher=1 from server-side batch=1 GPU time (D4). "
            "wall-clock and the router's own CPU forward are reported for "
            "context and do NOT enter the reward."
        ),
    }


# ---------------------------------------------------------------------------
# Per-arm measurement
# ---------------------------------------------------------------------------


def measure_teacher(policy, obs: dict, *, repeats: int) -> dict:
    """Full staged forward — what a router MISS actually executes."""
    return _record(lambda: policy.infer(dict(obs)), repeats=repeats,
                   path="stage1+stage2+stage3 (in-process)")


def measure_student(policy, obs: dict, *, repeats: int) -> dict:
    """One student policy forward, timed **inside the process that owns it**.

    ``policy`` must be the loaded student model (ACT / SmolVLA), not a
    ``SidecarExecutor``. Wrapping the RPC client would time a websocket round
    trip: the sidecar's kernels run in a different process and a different CUDA
    context, which this process's events cannot observe — being on the same host
    does not change that. The measurement therefore has to be taken by the
    sidecar's own interpreter, which is why this is a separate CLI invocation
    run there.
    """
    # The guard must survive running where openpi is NOT importable: this call
    # is made from the sidecar's own venv (which holds lerobot + openpi_client
    # but not openpi), so a hard import would abort the very measurement the
    # docstring above demands be taken there. The class-name check is the one
    # that always applies; the isinstance check is added back whenever the real
    # class can be imported.
    if type(policy).__name__ == "SidecarExecutor":
        raise ValueError(
            "measure_student needs the loaded student policy, not a SidecarExecutor: "
            "an RPC round trip cannot be GPU-timed from the calling process (D4 "
            "requires server-side batch=1 GPU time)"
        )
    try:
        from openpi.cache.sidecar_executor import SidecarExecutor
    except ImportError:
        SidecarExecutor = None  # sidecar venv: no openpi, hence no wrapper either
    if SidecarExecutor is not None and isinstance(policy, SidecarExecutor):
        raise ValueError(
            "measure_student needs the loaded student policy, not a SidecarExecutor: "
            "an RPC round trip cannot be GPU-timed from the calling process (D4 "
            "requires server-side batch=1 GPU time)"
        )
    return _record(lambda: policy(dict(obs)), repeats=repeats,
                   path="student policy forward (in-process on the sidecar host)")


def measure_cache(orchestrator, payload_id: str, action_chunk, *, repeats: int) -> dict:
    """The real replay path: payload fetch + broadcast, no model forward.

    Measured rather than assumed zero — a cache hit still pays a storage lookup
    and the history broadcast, and an arm whose cost is hard-coded to zero is
    exactly the kind of flattering assumption the reward must not contain.
    """
    from openpi.cache.components.payload_view import StoragePayloadView

    view = StoragePayloadView(orchestrator._storage)  # noqa: SLF001 - benchmark probe

    def replay() -> None:
        payload = view.get(payload_id)
        orchestrator.broadcast_action(
            payload.action_chunk if payload is not None else action_chunk
        )

    return _record(replay, repeats=repeats, path="PayloadView.get + broadcast_action")


def measure_router_cpu(judge, query_keys: dict, *, repeats: int) -> dict:
    """The router's own CPU forward — reported separately, never in the reward."""
    from openpi.cache.types import CheckpointID

    def forward() -> None:
        judge(results=[], checkpoint_id=CheckpointID.CP1, cached_data={},
              query_keys=query_keys)

    return wall_clock(forward, repeats=repeats)


def _aggregate_tasks(per_task: dict[str, dict]) -> dict:
    """Mean over the ensemble's tasks, keeping every per-task measurement."""
    means = [rec["gpu"]["mean_s"] for rec in per_task.values()]
    first = next(iter(per_task.values()))
    return {
        "gpu": {**first["gpu"], "mean_s": statistics.fmean(means),
                "n_tasks": len(per_task)},
        "wall": {"mean_s": statistics.fmean([r["wall"]["mean_s"] for r in per_task.values()]),
                 "n": first["wall"]["n"]},
        "path": first["path"],
        "per_task_gpu_mean_s": {p: r["gpu"]["mean_s"] for p, r in per_task.items()},
        "task_prompts": sorted(per_task),
    }


def _record(fn: Callable[[], object], *, repeats: int, path: str) -> dict:
    return {"gpu": gpu_timed(fn, repeats=repeats),
            "wall": wall_clock(fn, repeats=repeats), "path": path}


def act_manifest_prompts(manifest_path: str | pathlib.Path) -> list[str]:
    """The exact prompts the ACT ensemble routes on.

    ``make_act_policy`` matches ``obs["prompt"]`` against the manifest keys and
    raises on anything else, so a synthetic prompt cannot even complete warmup —
    the cost artifact for the owner's primary student arm could never be
    produced. The measurement therefore has to use real task prompts.
    """
    manifest = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
    return sorted(manifest)


def synthetic_obs(suite: str, *, prompt: Optional[str] = None) -> dict:
    """One fixed LIBERO-shaped observation, reused for every call.

    Holding the observation fixed removes input variation from the comparison
    between arms: the measurement is of the execution path, not of the data.
    """
    rng = np.random.RandomState(0)
    return {
        "observation/image": rng.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/state": rng.randn(8).astype(np.float32),
        "prompt": prompt if prompt is not None else f"microbench {suite}",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_arm(args) -> None:
    obs = synthetic_obs(args.suite)
    if args.arm == "teacher":
        policy = _load_local_policy(args.checkpoint, args.yaml)
        record = measure_teacher(policy, obs, repeats=args.repeats)
    elif args.arm == "student":
        if not args.student_checkpoint:
            raise SystemExit(
                "--student-checkpoint is required: the student arm must be timed "
                "inside the process that holds the model (run this in the sidecar's "
                "own venv on the sidecar host)"
            )
        policy = _load_student_policy(args.student_checkpoint, args.student_kind)
        if args.student_kind == "act":
            # ACT is a per-task ensemble routed by EXACT prompt match, so the
            # measurement runs over the manifest's real prompts. One task would
            # under-represent an ensemble whose members differ; the arm's cost is
            # the mean over tasks, and every per-task mean is recorded.
            prompts = act_manifest_prompts(args.student_checkpoint)
            if args.student_task:
                prompts = [p for p in prompts if p == args.student_task] or prompts[:1]
            per_task = {
                prompt: measure_student(policy, synthetic_obs(args.suite, prompt=prompt),
                                        repeats=args.repeats)
                for prompt in prompts
            }
            record = _aggregate_tasks(per_task)
        else:
            record = measure_student(policy, obs, repeats=args.repeats)
    else:
        orchestrator, entry_id, chunk = _load_cache_probe(args.yaml)
        record = measure_cache(orchestrator, entry_id, chunk, repeats=args.repeats)
    payload = {"arm": args.arm, "suite": args.suite, **record}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "wall"}, indent=2))


def _cmd_combine(args) -> None:
    records: dict[str, dict] = {}
    for pattern in args.inputs:
        for path in sorted(glob.glob(pattern)):
            doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
            records[doc["arm"]] = doc
    artifact = combine_records(records)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact["normalized_costs"], indent=2))


def _load_student_policy(checkpoint: str, kind: str, device: str = "cuda:0"):
    """Load the student model in-process (sidecar venv, sidecar host).

    Uses the very factories ``exp/ablation_study/sidecar_server.py`` serves
    from, so the thing being timed is the thing the student arm executes — not a
    re-implementation that could drift from it.
    """
    from exp.ablation_study.sidecar_server import make_act_policy, make_smolvla_policy

    if kind == "act":
        return make_act_policy(checkpoint, device)     # checkpoint = ACT manifest path
    return make_smolvla_policy(checkpoint, device)


def _load_local_policy(checkpoint: str, yaml_path: str):
    """Build the routed server's policy in-process (server host only)."""
    if not checkpoint:
        raise SystemExit("--checkpoint is required for the teacher arm")
    from openpi.policies import policy_config as _pc
    from openpi.training import config as _train_config

    cfg = _train_config.get_config("pi05_libero")
    return _pc.create_trained_policy(cfg, checkpoint, pytorch_device="cuda:0")


def _load_cache_probe(yaml_path: str):
    """Build storage + orchestrator from an arm yaml and pick one payload to replay.

    A production arm yaml preloads its library and the backend is frozen from
    that moment on (``write_policy: never``), so seeding a synthetic entry is
    impossible there — and also the wrong thing to measure: the cache arm's cost
    is a fetch of a *real* library payload, whose action chunk is the size the
    replay path actually moves. The synthetic seed is therefore only used when
    the backend is still writable (no preload, i.e. unit fixtures).
    """
    import torch

    from openpi.cache.storage_types import CacheEntry, CachePayload
    from openpi.cache.types import CheckpointID

    orchestrator, storage = _build_probe_components(yaml_path)
    if storage.is_frozen:
        entry_id = _first_preloaded_id(storage)
        return orchestrator, entry_id, storage.fetch_payload(entry_id).action_chunk
    chunk = torch.randn(50, 32)
    entry_id = "microbench:0"
    storage.insert(CacheEntry(
        id=entry_id, checkpoint_id=CheckpointID.CP1,
        query_keys={"robot_state": torch.zeros(32)},
        payload=CachePayload(action_chunk=chunk),
    ))
    return orchestrator, entry_id, chunk


def _build_probe_components(yaml_path: str):
    """Build ``(orchestrator, storage)`` from an arm yaml — the heavy half."""
    from openpi.cache.config import build_cache_components, load_cache_config
    from openpi.cache.orchestrator import CacheOrchestrator

    components = build_cache_components(load_cache_config(yaml_path))
    storage = components["storage"]
    orchestrator = CacheOrchestrator(
        storage=storage, key_builder=components["key_builder"],
        gates=components["gates"], judges=components["judges"],
        search_strategies=components["search_strategies"], timer=components["timer"],
    )
    return orchestrator, storage


def _first_preloaded_id(storage) -> str:
    """Deterministically pick one id out of a frozen, preloaded library.

    Sorted rather than "whatever the dict yields first" so two runs on the same
    artifact time the same entry; an id that varied per run would make the
    cache arm's cost irreproducible for no benefit.
    """
    entries = getattr(storage._backend, "_entries", None)  # noqa: SLF001 - benchmark probe
    if not entries:
        raise SystemExit(
            f"the cache arm needs a non-empty preloaded library; {storage.count()} "
            "entries were loaded from the arm yaml's artifact"
        )
    return sorted(entries)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_arm = sub.add_parser("arm", help="measure ONE arm on the host that owns its compute")
    p_arm.add_argument("--arm", required=True, choices=["teacher", "student", "cache"])
    p_arm.add_argument("--suite", default="libero_10")
    p_arm.add_argument("--repeats", type=int, default=50)
    p_arm.add_argument("--yaml", default="", help="arm yaml (teacher / cache)")
    p_arm.add_argument("--checkpoint", default="", help="pi05 checkpoint (teacher)")
    p_arm.add_argument("--student-checkpoint", default="",
                       help="student model checkpoint, loaded IN THIS PROCESS")
    p_arm.add_argument("--student-kind", default="act", choices=["act", "smolvla"])
    p_arm.add_argument("--student-task", default="",
                       help="measure one ACT manifest prompt instead of the whole ensemble")
    p_arm.add_argument("--out", required=True)
    p_arm.set_defaults(func=_cmd_arm)

    p_comb = sub.add_parser("combine", help="normalize per-arm records to teacher=1")
    p_comb.add_argument("--inputs", nargs="+", required=True)
    p_comb.add_argument("--out", required=True)
    p_comb.set_defaults(func=_cmd_combine)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
