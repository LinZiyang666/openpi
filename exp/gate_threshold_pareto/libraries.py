"""The four libraries this experiment sweeps, and the per-suite d1 base template.

One place to state what a run is bound to. Every other module takes a
``Library`` from here rather than re-deriving paths, so a swapped pkl or a
mis-set suite is a one-line diff in a reviewable table instead of a string
buried in a launch command.

Each library is also registered in ``exp/data_authority/records/``; the
``dataset_id`` field is the join. The content digests recorded there are what a
reader should verify before quoting any number this experiment produces.
"""

from __future__ import annotations

import dataclasses
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Server-side N4 (hybrid) yaml from the gate line, used as the structural
#: template: it already carries the d1 retrieval stack (weights, per-field
#: zscore+tanh normalizers, trajectory depth 1) for its suite.
TEMPLATE = {
    "libero_spatial": (
        "exp/gate_research/config/libero_spatial/n4_server/"
        "cp1_spatial_pool_16__grid3_vision_0@6_vision_1@50_robot_state@43__d1__fh75_ws10_quantile.yaml"
    ),
    "libero_10": (
        "exp/gate_research/config/libero_10/n4_server/"
        "cp1_spatial_pool_16__grid3_vision_0@56_vision_1@25_robot_state@18__d1__fh5_ws40_quantile.yaml"
    ),
}

#: Frozen A-pool evaluation records (the 500-init official test set per suite).
APOOL_RECORD = {
    "libero_spatial": "exp/ablation_study/cache_size/config/apool_libero_spatial.yaml",
    "libero_10": "exp/ablation_study/cache_size/config/apool_libero_10.yaml",
}


@dataclasses.dataclass(frozen=True)
class Library:
    """One retrieval library: its id, suite, server-side path and ledger join."""

    lib_id: str
    suite: str
    preload_path: str
    dataset_id: str
    note: str


LIBRARIES: tuple[Library, ...] = (
    Library(
        lib_id="ws",
        suite="libero_spatial",
        preload_path="exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl",
        dataset_id="weighted_sum/libero_spatial/cp1_spatial_pool_16",
        note="weighted_sum / threshold-pareto base library; 1018 entries, 49 trajectories",
    ),
    Library(
        lib_id="ws",
        suite="libero_10",
        preload_path="exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl",
        dataset_id="weighted_sum/libero_10/cp1_spatial_pool_16",
        note="weighted_sum / threshold-pareto base library; 2640 entries, 50 trajectories",
    ),
    Library(
        lib_id="cs",
        suite="libero_spatial",
        preload_path=(
            "/data/openpi/ablation_study/cache_size/artifacts/"
            "cache_size_libero_spatial_all_S3.pkl"
        ),
        dataset_id="cache_size/libero_spatial/all_s3",
        note="cache_size X9b tier S3 (5 trajectories/task); 1072 entries, 50 trajectories",
    ),
    Library(
        lib_id="cs",
        suite="libero_10",
        preload_path=(
            "/data/openpi/ablation_study/cache_size/artifacts/"
            "cache_size_libero_10_all_S3.pkl"
        ),
        dataset_id="cache_size/libero_10/all_s3",
        note="cache_size X9b tier S3 (5 trajectories/task); 2741 entries, 50 trajectories",
    ),
)


def arm_key(lib: Library) -> str:
    """Stable short name for one (library, suite) pair."""
    return f"{lib.lib_id}_{'sp' if lib.suite == 'libero_spatial' else 'l10'}"


def for_suite(suite: str) -> list[Library]:
    return [lib for lib in LIBRARIES if lib.suite == suite]


def by_arm(arm: str) -> Library:
    for lib in LIBRARIES:
        if arm_key(lib) == arm:
            return lib
    raise KeyError(f"unknown arm {arm!r}; known: {[arm_key(x) for x in LIBRARIES]}")
