"""Separate NVIDIA inference protocol; preserves the complete R17 LIBERO cohort."""
from dataclasses import replace

PROPRIO_KEY = "observation/cosmos_proprio"
PROTOCOL_PREFIX = "libero_cosmos_official_"


def benchmark_profile(base, official_50=False):
    return replace(base, seed=195, seed_mode="fixed", replan_steps=16,
        protocol_id=PROTOCOL_PREFIX + ("50" if official_50 else "10") + "_r17_v1",
        formal_indices=tuple(range(50 if official_50 else 10)),
        library_indices=() if official_50 else tuple(range(10, 30)),
        dev_indices=() if official_50 else tuple(range(30, 50)))


def uses_raw_proprio(spec):
    return spec.protocol_id.startswith(PROTOCOL_PREFIX)
