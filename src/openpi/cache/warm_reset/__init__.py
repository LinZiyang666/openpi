"""Warm reset continuation family (``warm_reset:`` yaml block) as a production feature.

Plan ``logs/warm_continuation_first_class_plan.log.md``. A warm reset replaces
only the stage-3 continuation of a WARM_START verdict -- start (cache / self x
snapshot / final), t grid (reset / shoot) and Euler step count -- and mirrors
the exact WARM_START chain: verdict ``start_t`` -> interceptor -> stage-3 entry
-> batching coordinator. Judges, retrieval, orchestrator, conductor and
workers are unaware of it; without the block every path is unchanged.

This package root exports only the model-agnostic modules (``types``,
``runtime``, ``evidence``). The executors live in ``pi05`` (Pi0.5 only, imports
the model) and ``groot`` (jax-free), and are imported by their serving entry
points directly so neither import chain reaches the other.
"""

from openpi.cache.warm_reset.evidence import (
    EVIDENCE_SCHEMA,
    ExpectedEpisode,
    GrootWarmResetEvidencePolicy,
    WarmResetEvidencePolicy,
    decision_problems,
    episode_problems,
)
from openpi.cache.warm_reset.runtime import (
    META_SCHEMA,
    WarmResetParts,
    WarmResetSession,
    decision_meta,
    refuse_warm_reset,
)
from openpi.cache.warm_reset.types import (
    EpisodeDigestSeed,
    SelfSeedPolicy,
    SelfStartPlan,
    WarmResetPlan,
    WarmResetSpec,
    private_noise,
    resolve_plan,
    resolve_self_plan,
    stable_digest_int,
    validate_plan,
    validate_self_plan,
)

__all__ = [
    "EVIDENCE_SCHEMA",
    "META_SCHEMA",
    "EpisodeDigestSeed",
    "ExpectedEpisode",
    "GrootWarmResetEvidencePolicy",
    "SelfSeedPolicy",
    "SelfStartPlan",
    "WarmResetEvidencePolicy",
    "WarmResetParts",
    "WarmResetPlan",
    "WarmResetSession",
    "WarmResetSpec",
    "decision_meta",
    "decision_problems",
    "episode_problems",
    "private_noise",
    "refuse_warm_reset",
    "resolve_plan",
    "resolve_self_plan",
    "stable_digest_int",
    "validate_plan",
    "validate_self_plan",
]
