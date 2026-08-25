"""Fan one eval stage out across every server, so a single-arm phase uses the pool.

Why this exists
---------------
``EpisodeScheduler.next_task(server_key)`` only hands a worker episodes whose
stage is pinned to *that* server, and ``assign_servers`` places each yaml on
exactly one server. A phase holding a single yaml therefore runs at 1/N of an
N-server pool: measured on the 2026-08-24 gate-Pareto run, the two single-arm
phases took 48.0 of libero_spatial's 190.5 minutes (25.2%) while five of six
slots sat idle.

Nothing in the core requires one stage per yaml -- ``plan()`` is the strategy's
to write. Splitting one yaml into N sibling stages, one per server, makes every
server eligible for that yaml's episodes without touching the scheduler:

  * the activation cap counts per ``(server.key, phase)``, so each server sees
    exactly one sibling and activates it normally;
  * ``make_task_uid`` does not encode the server, so an episode keeps its
    identity across a reshard -- a run resumed with a different shard count
    still replays its journal exactly.

Eval only, and that is enforced
-------------------------------
A warmup stage publishes a calibration artifact: its ``on_stage_complete`` hook
fetches the server's dump and calls ``StageContext.publish``, which is a plain
assignment. Sharding it would leave each sibling holding 1/N of the dump and
each publish overwriting the last, so the downstream eval would preload one
shard's calibration and nothing would raise -- ``CalibrationArtifact`` names a
single producing stage, and ``TaskGraph.validate`` does not check it. The
gate-Pareto line happens to calibrate offline and never calls ``fetch_dump``,
but the failure is silent for any line that does, so this module refuses the
shape rather than relying on the caller to know.

Coupling map:
  DEPENDS ON:  openpi.conductor.task + stdlib only (same boundary as task.py)
  CONSUMED BY: concrete ExperimentStrategy implementations in exp/
"""

from __future__ import annotations

import dataclasses

from openpi.conductor import task as _task


def stride_partition(items: list, n: int) -> list[list]:
    """Split ``items`` into ``n`` shards by stride, preserving order within each.

    Stride (``items[k::n]``) rather than contiguous blocks: episodes arrive
    grouped by task, and a contiguous split hands one shard every episode of the
    slowest task. Measured on the finished ``gpgo_sp`` arm (500 episodes, per-episode
    cost spread cv=0.35), the slowest of six contiguous blocks ran 8.5% over an even
    split against 4.9% for stride; at 48 shards the gap widens to 41.7% vs 26.9%
    (``exp/libero_groot/analysis/gate_pareto/shard_imbalance_probe.py``).

    Shard lengths differ by at most one and their union is exactly ``items``.
    Shards may be empty when ``len(items) < n``; callers must keep them (see
    ``shard_eval_stage``).
    """
    if n < 1:
        raise ValueError(f"shard count must be >= 1, got {n}")
    return [items[k::n] for k in range(n)]


def shard_eval_stage(
    *,
    stage_id: str,
    yaml_id: str,
    episodes: list[_task.EpisodeTask],
    servers: list[_task.ServerEndpoint],
    episodes_are_idempotent: bool,
    setup: dict | None = None,
    consumes_calib_id: str | None = None,
) -> list[_task.Stage]:
    """One logical eval stage -> one sibling ``Stage`` per server.

    Each sibling carries a stride shard of ``episodes`` with every task's
    ``server_host``/``server_port`` rewritten to that sibling's endpoint: the
    worker connects to the server named on the task, so a shard whose tasks
    still named the original endpoint would send every worker back to one box.

    Empty shards are returned, not dropped: the shard count is a property of the
    topology, and dropping empties would make the stage set depend on how much
    work happens to be left, so a resume would plan a different graph than the
    run it resumes. The scheduler already sends a zero-episode stage straight to
    completion after its setup hook -- but that hook runs *first*
    (``driver.py`` calls ``on_stage_begin`` before ``mark_setup_done``), so a
    strategy whose setup loads a cache bundle must return early on an empty
    stage or pay a full library reload for no work.

    ``episodes_are_idempotent`` has no default on purpose: the caller must
    state it. ``phase`` is not a sufficient test -- RoboCasa's collection
    stages are ``phase="eval"`` (``exp/robocasa365/run_collect.py:158,180``)
    yet write one HDF5 file through a single writer, so sharding them would put
    several collectors on one output with nothing raising. Only the strategy
    knows whether its episodes are replayable side-effect-free.

    Raises:
        ValueError: if ``servers`` is empty, if any episode is not ``eval``, or
            if the caller has not asserted idempotence. Warmup is refused by
            construction -- see the module docstring.
    """
    if not servers:
        raise ValueError("shard_eval_stage requires at least one server")
    if not episodes_are_idempotent:
        raise ValueError(
            "shard_eval_stage requires episodes_are_idempotent=True: running the "
            "same stage on several servers is only safe when an episode leaves no "
            "exclusive side effect. A collection stage is phase='eval' but writes "
            "one HDF5 through a single writer -- sharding it corrupts the artifact "
            "silently."
        )
    non_eval = sorted({ep.phase for ep in episodes} - {"eval"})
    if non_eval:
        raise ValueError(
            f"shard_eval_stage is eval-only; got episodes with phase {non_eval}. "
            "A warmup stage publishes one calibration artifact, so sharding it "
            "would silently keep 1/N of the dump (see module docstring)."
        )
    shards = stride_partition(episodes, len(servers))
    out: list[_task.Stage] = []
    for k, (server, shard) in enumerate(zip(servers, shards, strict=True)):
        out.append(
            _task.Stage(
                stage_id=f"{stage_id}__s{k}",
                yaml_id=yaml_id,
                phase="eval",
                server=server,
                episodes=[
                    dataclasses.replace(ep, server_host=server.host, server_port=server.port)
                    for ep in shard
                ],
                consumes_calib_id=consumes_calib_id,
                setup=dict(setup or {}),
            )
        )
    return out
