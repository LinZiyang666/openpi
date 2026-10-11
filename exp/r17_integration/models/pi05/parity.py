"""π-MW qualification parity: exact R16 eager stages, bounded other-path study.

The alternate whole-call compiler is not the serving/reference path. Its bounds
apply only to the sealed Q synthetic cohort below; they are neither a policy
quality tolerance nor permission to relax reference/debug/suffix equality.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .gpu_diagnostic import action_difference, r16_reference

REFERENCE_PATH = "r16.concurrent.eager_sdpa.public_stages.return_intermediates_true"
CONTRACT_VERSION = "r17.pi05.parity.r16_eager.v1"
FIXTURE_CHECKPOINT_IDENTITY = "4008289bc944015dbf3eba6cea0a701f6ef9dbb53e0bfd47dea7b55585d8b62e"
FIXTURE_NORM_IDENTITY = "ab3e2f7380ccbd0967fe391e07d4bc67492ceb7e3e06a9c03397aa6c28309220"
FIXTURE_ARCHIVE_SHA256 = "ff0ae420a969cd6d48208a7982ac4542e4fcefd8c82cd0f49e65fd7aa0f01adb"
FIXTURE_SCOPE = "Q.pi05_mw_fix2_b3a3_v1.three_saved_observations.decision_seed_reach_init0_seq0_to2"


def compiled_fixture_limits(spec, observation_archive):
    """Explicit admission of the measured alternate-path diagnostic cohort.

    Q still validates its runtime source seal. Bounds are absolute native wire
    units, rtol=0, with an additional per-observation mean bound. No fallback
    profile exists for another checkpoint/norm/archive or sampler.
    """
    if (spec.model_id != "pi05_rlinf_mt50" or
            spec.checkpoint_identity != FIXTURE_CHECKPOINT_IDENTITY or
            spec.norm_identity != FIXTURE_NORM_IDENTITY):
        raise ValueError("compiled comparison profile model/norm mismatch")
    if hashlib.sha256(Path(observation_archive).read_bytes()).hexdigest() != FIXTURE_ARCHIVE_SHA256:
        raise ValueError("compiled comparison profile observation archive mismatch")
    return {k: {"max_abs_diff": maximum, "mean_abs_diff": mean, "rtol": 0,
                "scope": FIXTURE_SCOPE, "quality_claim": False}
            for k, maximum, mean in ((2, .08, .02), (1, .13, .03))}


def assert_native_parity(reference_normalized, reference_wire, plain, wire, debug, debug_wire,
                         *, alternate_wire=None, alternate_limits=None):
    """Reference/debug bytes remain mandatory, irrespective of alternate limits."""
    k = plain.sampler.num_steps
    if plain.actual_nfe != k or debug.actual_nfe != k or debug.sampler != plain.sampler:
        raise AssertionError("native sampler/NFE parity mismatch")
    comparisons = {
        "reference_normalized_vs_plain": action_difference(reference_normalized, plain.normalized_chunk),
        "reference_wire_vs_plain": action_difference(reference_wire, wire.actions),
        "debug_normalized_vs_plain": action_difference(plain.normalized_chunk, debug.normalized_chunk),
        "debug_wire_vs_plain": action_difference(wire.actions, debug_wire.actions),
    }
    for name, comparison in comparisons.items():
        if not comparison["exact"]:
            raise AssertionError(f"exact R16/debug parity failed: {name}: {comparison}")
    result = {"contract": CONTRACT_VERSION, "baseline_path": REFERENCE_PATH,
              "exact_r16": True, "exact_debug": True, "comparisons": comparisons,
              "quality_claim": False}
    if alternate_wire is not None:
        comparison = action_difference(alternate_wire, wire.actions)
        # K10's stock staged timing helper is the existing exact comparison.
        if k == 10:
            if not comparison["exact"]:
                raise AssertionError("exact stock staged K10 parity failed")
            comparison["scope"] = "stock.Policy.staged_timing.default_K10"
        else:
            limits = (alternate_limits or {}).get(k)
            if limits is None or limits.get("scope") != FIXTURE_SCOPE or limits.get("rtol") != 0:
                raise ValueError("explicit sealed alternate comparison profile required")
            if (comparison["max_abs_diff"] > limits["max_abs_diff"] or
                    comparison["mean_abs_diff"] > limits["mean_abs_diff"]):
                raise AssertionError(f"compiled alternate diagnostic exceeded bound: {comparison}")
            comparison["limits"] = dict(limits)
        result["alternate_path"] = comparison
    return result
