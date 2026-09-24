"""Trace serving mode: run every module on every decision, record everything.

Plan: ``logs/cache_trace_mode_plan.log.md``. The executed action still follows
the yaml; the trace adds the variants that were not executed (full inference,
every executable warm tier, the top-1 replay), the twin component set's
search / verdict, the raw observation, the prefix tokens and the query keys,
and writes one HDF5 file per episode that is a strict superset of the legacy
``--collect`` file.

This package is jax-free and does not import the interceptor, the models or
the serving stack at module level; ``pi05`` is the only model-specific module
and is imported lazily by the Pi0.5 interceptor.
"""

from openpi.cache.trace.types import (
    ARM_FULL_HIT,
    ARM_FULL_INFERENCE,
    ARM_WARM_EXEC,
    TRACE_SCHEMA_VERSION,
    DrainReport,
    EpisodeIdentity,
    PerFieldTrace,
    SearchTrace,
    StepTrace,
    TracePlan,
    TraceRuntime,
    TraceSink,
    TraceWriteError,
    TraceWriteFailed,
    warm_arm_name,
)

__all__ = [
    "ARM_FULL_HIT",
    "ARM_FULL_INFERENCE",
    "ARM_WARM_EXEC",
    "TRACE_SCHEMA_VERSION",
    "DrainReport",
    "EpisodeIdentity",
    "PerFieldTrace",
    "SearchTrace",
    "StepTrace",
    "TracePlan",
    "TraceRuntime",
    "TraceSink",
    "TraceWriteError",
    "TraceWriteFailed",
    "warm_arm_name",
]
