"""Assembly of the trace runtime for one connection (plan §4.2, §4.3, §10).

``build_trace_runtime`` turns the effective ``TraceConfig`` plus the
connection's identity into a ``TraceRuntime`` (plan + sink + twin component
set), or ``None`` when tracing is off. ``executable_warm_tiers`` enumerates
the warm variants to compute for the configured judge (distinct from
``required_warm_timesteps``, which is the library-completeness set), and
``build_trace_twins`` assembles the second, isolated component set that the
orchestrator drives with ``force_search=True``.

Public interface: ``build_trace_runtime``, ``build_groot_trace_runtime``, ``executable_warm_tiers``,
``build_trace_twins``, ``strip_twin_config``, ``TWIN_STRIP_FIELDS``.
Depends on ``openpi.cache.config`` (factories, ``TraceConfig``),
``openpi.cache.trace.types`` and ``openpi.cache.trace.h5_sink``. Imports the
orchestrator lazily inside ``build_trace_twins`` (the types module must stay
import-light for the GR00T island).
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
from typing import Any, Optional

from openpi.cache.config import CacheConfig, TraceConfig
from openpi.cache.trace.h5_sink import H5TraceSink, TraceWriter
from openpi.cache.trace.types import TracePlan, TraceRuntime
from openpi.cache.types import CheckpointID, DenoiseSchedule

logger = logging.getLogger("openpi.cache.trace.runtime")

#: Judge-config fields that name an external write target. A twin is built
#: from a config copy with every one of them cleared: observation and dump
#: settings are not part of the "same parameters" contract (plan §4.2).
TWIN_STRIP_FIELDS: dict[str, tuple[str, ...]] = {
    "JudgeConfig": ("dump", "dump_dir", "state_log_dir", "snapshot_every"),
    "TimerConfig": ("output_csv_dir",),
}

#: Path-like config fields that are read-only inputs and therefore kept on
#: the twin. Every field whose name matches the scan pattern must be listed
#: either here or in ``TWIN_STRIP_FIELDS`` (guard test in tests/cache/trace).
TWIN_READONLY_PATH_FIELDS: dict[str, tuple[str, ...]] = {
    "JudgeConfig": (
        "surface_artifact_path",
        "weights_path",
        "risk_model_path",
        "update_scales_path",
        "init_state_path",
        "artifact_path",
    ),
    "BackendConfig": (),
    "InMemoryConfig": ("preload_path",),
    "KeyBuilderConfig": ("weights_path",),
    "ProjectionKeyBuilderConfig": ("weights_path",),
    "StatsSourceOfflineConfig": ("path",),
    "SamplesSourceOfflineConfig": ("path",),
    "DumpConfig": ("path",),
    "ShadowTeacherConfig": ("path",),
    "TraceConfig": ("out_dir",),
    # Never reaches a twin: warm_reset and trace are mutually exclusive.
    "WarmResetConfig": ("evidence_dir",),
    "CollectionConfig": (),
    # Not a path at all (composer sign directions); listed so the name-pattern
    # guard does not flag it.
    "ComposerConfig": ("directions",),
}


# ---------------------------------------------------------------------------
# Executable warm tiers
# ---------------------------------------------------------------------------


def _unwrap_judge(judge: Any) -> Any:
    inner = getattr(judge, "_inner", None)
    return inner if inner is not None else judge


def _round4(value: float) -> float:
    return round(float(value), 4)


def executable_warm_tiers(
    config: Optional[CacheConfig],
    *,
    checkpoint: Optional[CheckpointID],
    assembled_judge: Any,
    schedule: DenoiseSchedule,
) -> tuple[int, ...]:
    """Snapshot indices of the warm variants to compute for ``checkpoint``.

    Enumerated per judge type from the config, or from the assembled judge's
    artifact where the config does not carry the value (dispatch surface /
    CRD). Online RIT contributes its tiers only, never the successor
    snapshots its feedback reads. Values are validated against ``schedule``
    and returned as loop-step indices in loop-execution order.
    """
    if config is None or checkpoint is None:
        return ()
    cp_name = checkpoint.name.lower()
    cp = config.checkpoints.get(cp_name)
    if cp is None or not cp.enabled:
        return ()
    judge = cp.judge
    found: set[float] = set()
    jtype = judge.type
    if jtype in ("threshold", "failure_aware_gate", "always_hit"):
        for tier in judge.warm_tiers or []:
            if "start_t" in tier:
                found.add(_round4(tier["start_t"]))
    elif jtype == "always_warm_start":
        if judge.start_t is not None:
            found.add(_round4(judge.start_t))
    elif jtype == "composite":
        composer = judge.composer
        if composer is not None:
            for value in (composer.warm_start_t, composer.warm_fallback_start_t):
                if value is not None:
                    found.add(_round4(value))
    elif jtype in ("dispatch_surface", "crd"):
        art = getattr(_unwrap_judge(assembled_judge), "artifact", None)
        start_t_ws = getattr(art, "start_t_ws", None)
        if start_t_ws is None:
            raise ValueError(
                f"checkpoints.{cp_name}.judge.type={jtype!r}: the assembled judge "
                "exposes no artifact.start_t_ws; cannot enumerate its warm tier"
            )
        found.add(_round4(start_t_ws))
    elif jtype == "online_rit":
        for t in judge.tiers or []:
            found.add(_round4(t))
    elif jtype in ("mlp_router", "risk_router"):
        pass
    else:
        raise ValueError(
            f"checkpoints.{cp_name}.judge.type={jtype!r}: trace does not know "
            "how to enumerate this judge's warm tiers"
        )
    indices: list[int] = []
    for t in found:
        if t not in schedule.timestep_set:
            raise ValueError(
                f"checkpoints.{cp_name}: warm tier t={t} is not a snapshot of "
                f"{schedule.schedule_id} ({schedule.timesteps})"
            )
        indices.append(schedule.snapshot_index(t))
    return tuple(sorted(indices))


# ---------------------------------------------------------------------------
# Twin assembly
# ---------------------------------------------------------------------------


def strip_twin_config(config: CacheConfig) -> CacheConfig:
    """Config copy with every external write target of the components cleared."""
    checkpoints = {}
    for name, cp in config.checkpoints.items():
        judge = cp.judge
        judge_kwargs = {f: None for f in TWIN_STRIP_FIELDS["JudgeConfig"] if hasattr(judge, f)}
        checkpoints[name] = dataclasses.replace(cp, judge=dataclasses.replace(judge, **judge_kwargs))
    timer = dataclasses.replace(config.timer, enabled=False, output_csv_dir=None)
    return dataclasses.replace(config, checkpoints=checkpoints, timer=timer)


def build_trace_twins(
    config: CacheConfig,
    shared_storage: Any,
    *,
    real_components: dict[str, Any],
    yaml_id: Optional[str],
) -> Any:
    """Assemble the twin component set (plan §4.2).

    A second call of the production factory on the stripped config copy: same
    component types and parameters, separate instances, separate storage
    facade, timer disabled, no dump / shard / snapshot output. ``online_rit``
    twins share one process-level twin ``CurveRegistry`` that is distinct from
    the real one. The real judges' effective ``min_required_top_k`` is carried
    over so removing a wrapper never narrows the search width.
    """
    from openpi.cache.config import build_per_connection_components
    from openpi.cache.orchestrator import TwinSet

    stripped = strip_twin_config(config)
    real_judges = real_components.get("judges") or {}
    checkpoints = dict(stripped.checkpoints)
    for cp_id, real_judge in real_judges.items():
        name = cp_id.name.lower()
        if name not in checkpoints:
            continue
        cp = checkpoints[name]
        floor = int(getattr(real_judge, "min_required_top_k", 0) or 0)
        checkpoints[name] = dataclasses.replace(
            cp, search_strategy=dataclasses.replace(
                cp.search_strategy, top_k=max(cp.search_strategy.top_k, floor)
            )
        )
    stripped = dataclasses.replace(stripped, checkpoints=checkpoints)
    registry = _twin_online_registry(config)
    components = build_per_connection_components(
        stripped,
        shared_storage,
        yaml_id=yaml_id,
        quiet=True,
        online_registry=registry,
    )
    return TwinSet(
        key_builder=components["key_builder"],
        gates=components["gates"],
        judges=components["judges"],
        strategies=components["search_strategies"],
        storage=components["storage"],
        timer=components["timer"],
        library_stats=components.get("library_stats"),
    )


_TWIN_REGISTRY: dict[str, Any] = {}


def _twin_online_registry(config: CacheConfig) -> Any:
    """Process-level twin ``CurveRegistry`` (one per process, never persisted)."""
    needs = any(
        cp.enabled and cp.judge.type == "online_rit"
        for name, cp in config.checkpoints.items()
        if not name.startswith("_")
    )
    if not needs:
        return None
    from openpi.cache.online_state import CurveRegistry

    reg = _TWIN_REGISTRY.get("registry")
    if reg is None:
        reg = CurveRegistry(state_log_root=None, require_persistence=False)
        _TWIN_REGISTRY["registry"] = reg
    return reg


# ---------------------------------------------------------------------------
# Runtime assembly
# ---------------------------------------------------------------------------


def _yaml_sha256(text: Optional[str]) -> str:
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_trace_runtime(
    trace_config: Optional[TraceConfig],
    *,
    cache_config: Optional[CacheConfig],
    model: str,
    schedule: DenoiseSchedule,
    checkpoint: Optional[CheckpointID],
    real_components: Optional[dict[str, Any]],
    shared_storage: Any,
    artifact_meta: Optional[dict[str, Any]],
    bundle_id: str,
    yaml_id: Optional[str],
    yaml_text: Optional[str],
    connection_id: str,
    concurrent: bool,
) -> Optional[TraceRuntime]:
    """Build the per-connection trace runtime, or ``None`` when tracing is off.

    ``trace_config`` is the effective (yaml + CLI) block from
    ``validate_effective_trace``. With no cache config there is no
    orchestrator and therefore no checkpoint, no warm tiers and no twins:
    that is the build-cache form (plan §8).
    """
    if trace_config is None or not trace_config.enabled:
        return None
    assembled_judge = None
    if real_components is not None and checkpoint is not None:
        assembled_judge = (real_components.get("judges") or {}).get(checkpoint)
    warm_tiers = executable_warm_tiers(
        cache_config,
        checkpoint=checkpoint,
        assembled_judge=assembled_judge,
        schedule=schedule,
    )
    meta = artifact_meta if isinstance(artifact_meta, dict) else {}
    plan = TracePlan(
        model=model,
        schedule=schedule,
        checkpoint=checkpoint.name if checkpoint is not None else None,
        warm_tiers=warm_tiers,
        record_noise_actions=bool(trace_config.record_noise_actions),
        save_timesteps=tuple(schedule.timesteps) if trace_config.record_noise_actions else None,
        record_prefix_tokens=bool(trace_config.record_prefix_tokens),
        record_raw_images=bool(trace_config.record_raw_images),
        record_model_images=bool(trace_config.record_model_images),
        record_query_keys=bool(trace_config.record_query_keys),
        record_search=bool(trace_config.record_search),
        record_tokenized_prompt=bool(trace_config.record_tokenized_prompt),
        raw_image_keys=tuple(trace_config.raw_image_keys),
        rng_isolation=trace_config.rng_isolation,
        sidecar_jsonl=bool(trace_config.sidecar_jsonl),
        yaml_id=yaml_id or "",
        yaml_sha256=_yaml_sha256(yaml_text),
        bundle_id=bundle_id,
        connection_id=connection_id,
        key_builder_type=(cache_config.key_builder.type if cache_config is not None else None),
        library_sha256=meta.get("library_sha256"),
        concurrent=bool(concurrent),
        fail_loud=bool(trace_config.record_noise_actions),
    )
    writer = TraceWriter.get(trace_config.out_dir, queue_steps=int(trace_config.queue_steps))
    sink = H5TraceSink(trace_config.out_dir, plan=plan, writer=writer)
    twins = None
    if cache_config is not None and real_components is not None:
        twins = build_trace_twins(
            cache_config,
            shared_storage,
            real_components=real_components,
            yaml_id=yaml_id,
        )
    logger.info(
        "trace runtime: model=%s checkpoint=%s warm_tiers=%s build=%s out_dir=%s",
        model,
        plan.checkpoint,
        warm_tiers,
        plan.record_noise_actions,
        trace_config.out_dir,
    )
    return TraceRuntime(plan=plan, sink=sink, twins=twins)


def build_groot_trace_runtime(
    trace_config: Optional[TraceConfig],
    *,
    cache_config: Optional[CacheConfig],
    components: Optional[dict[str, Any]],
    orchestrator: Optional[Any],
    runner: Any,
    bundle_id: str,
    yaml_id: Optional[str],
    yaml_path: Optional[str],
    concurrent: bool,
) -> Optional[TraceRuntime]:
    """``build_trace_runtime`` for a GR00T server (plan §9-8).

    The schedule is the served action head's live one (``runner.live_schedule``);
    the checkpoint is CP2 when the orchestrator serves it, CP1 otherwise, None
    without an orchestrator (build form). Shared by both GR00T entry points so
    the two never disagree on how a connection's runtime is assembled.
    """
    if trace_config is None or not trace_config.enabled:
        return None
    import uuid

    checkpoint = None
    if orchestrator is not None:
        checkpoint = (
            CheckpointID.CP2 if orchestrator.has_checkpoint(CheckpointID.CP2) else CheckpointID.CP1
        )
    yaml_text = None
    if yaml_path:
        try:
            with open(yaml_path, encoding="utf-8") as fh:
                yaml_text = fh.read()
        except OSError:
            yaml_text = None
    return build_trace_runtime(
        trace_config,
        cache_config=cache_config,
        model="groot_n15",
        schedule=runner.live_schedule(),
        checkpoint=checkpoint,
        real_components=components,
        shared_storage=components["storage"] if components is not None else None,
        artifact_meta=getattr(orchestrator, "artifact_meta", None),
        bundle_id=bundle_id,
        yaml_id=yaml_id,
        yaml_text=yaml_text,
        connection_id=uuid.uuid4().hex[:12],
        concurrent=bool(concurrent),
    )
