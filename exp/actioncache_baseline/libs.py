"""Shared constants and helpers of the ActionCache-style CP2 baseline line.

Everything the scripts of this package agree on lives here: the CP2 tier
cost model (two unit-cost tables: the frozen CUDA-graph stage costs from the
dispatch-surface cost authority and the eager stage split recorded by the
latency bench), the raw<->normalised threshold mapping of the single cosine
field, the arm naming scheme, H5 lookup by trajectory id and small file
helpers (sha256, git commit).

Depends on ``exp.dispatch_surface.analysis.analytic_cost`` (CUDA-graph stage
constants) and ``openpi.cache.components.cp2_vlm_key_builder`` (field name /
projection defaults).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import pickle
import subprocess
from typing import Any, Iterator

import h5py

from exp.dispatch_surface.analysis.analytic_cost import STAGE1_MS, STAGE2_MS, STAGE3_MS
from openpi.cache.components.cp2_vlm_key_builder import (
    DEFAULT_D,
    DEFAULT_INPUT_DIM,
    DEFAULT_P,
    KEY_BUILDER_TYPE,
)
from openpi.cache.types import VLM_OUT

PROTOCOL = "actioncache_baseline/v1"
FIELD = VLM_OUT
ID_POLICY = "inherited_from_source"
WARM_START_T = 0.1  # ActionCache N_hit=1 == one refinement step == start_t 0.1
N1_FULL_THRESHOLD = 1.5  # deliberately outside affine_clip's [0, 1] score range

#: Tier tag -> (hit_type, start_t). ``n0`` = ActionCache N_hit=0 (direct
#: execution of the cached chunk), ``n1`` = N_hit=1 (one denoising step).
TIERS: dict[str, tuple[str, float | None]] = {
    "n0": ("FULL_HIT", None),
    "n1": ("WARM_START", WARM_START_T),
}

#: Per-stage unit costs in ms. ``cuda_graph`` is the project's frozen cost
#: authority (same numbers every frontier in this repo is priced with);
#: ``eager`` is the eager stage split measured by the latency bench
#: (exp/data_authority/records/latency_bench__libero_spatial__executor_costs.json,
#: ``pi05_stage_split_ms.eager``).
COST_TABLES: dict[str, dict[str, float]] = {
    "cuda_graph": {"stage1": STAGE1_MS, "stage2": STAGE2_MS, "stage3": STAGE3_MS},
    "eager": {"stage1": 63.06, "stage2": 35.27, "stage3": 349.81},
}
DEFAULT_COST_TABLE = "cuda_graph"


def cp2_tier_cost(hit_type: str, start_t: float | None, table: str = DEFAULT_COST_TABLE) -> float:
    """Per-decision model-forward cost of a CP2 verdict.

    The backbone always runs at CP2: FULL_HIT pays stage 1 + 2, WARM_START
    pays stage 1 + 2 + ``start_t`` of stage 3, MISS pays all three stages.
    """
    t = COST_TABLES[table]
    if hit_type == "FULL_HIT":
        return t["stage1"] + t["stage2"]
    if hit_type == "MISS":
        return t["stage1"] + t["stage2"] + t["stage3"]
    if hit_type == "WARM_START":
        if start_t is None:
            raise ValueError("WARM_START needs a start_t")
        st = round(float(start_t), 4)
        if not (0.0 < st < 1.0):
            raise ValueError(f"start_t={start_t} is not a canonical denoise timestep")
        return t["stage1"] + t["stage2"] + st * t["stage3"]
    raise ValueError(f"unknown hit_type {hit_type!r}")


def miss_cost(table: str = DEFAULT_COST_TABLE) -> float:
    return cp2_tier_cost("MISS", None, table)


def theta_norm(theta_raw: float) -> float:
    """Raw cosine cut -> deployed ``affine_clip(lo=-1, hi=1)`` score cut."""
    return (float(theta_raw) + 1.0) / 2.0


def theta_raw(theta_norm_value: float) -> float:
    """Inverse of :func:`theta_norm`."""
    return 2.0 * float(theta_norm_value) - 1.0


# ------------------------------------------------------------------
# Arm naming
# ------------------------------------------------------------------

SUITE_TAGS = {"libero_spatial": "sp", "libero_10": "l10"}


def arm_name(suite: str, lib_tag: str, tier: str, target: str) -> str:
    """``acb_<suite>_<lib>_<tier>_<target>`` — e.g. ``acb_sp_lib50_n0_ir60``."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return f"acb_{SUITE_TAGS[suite]}_{lib_tag}_{tier}_{target}"


def parse_arm(arm: str) -> dict[str, str] | None:
    parts = arm.split("_")
    if len(parts) != 5 or parts[0] != "acb" or parts[3] not in TIERS:
        return None
    return {"suite_tag": parts[1], "lib": parts[2], "tier": parts[3], "target": parts[4]}


# ------------------------------------------------------------------
# Files
# ------------------------------------------------------------------


def sha256_file(path: str | pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001 - provenance only
        return "unknown"


def load_pickle(path: str | pathlib.Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_json(path: str | pathlib.Path, obj: Any) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ------------------------------------------------------------------
# H5 lookup by trajectory id
# ------------------------------------------------------------------


class H5Index:
    """Map a library ``trajectory_id`` to its H5 file under a collection root.

    Both id conventions of the offline builders are accepted: the bare stem
    (``episode_0004_20260410_011001_080633``, flat layouts) and the
    suffix-stripped relative path (``task_3/episode_12``, ``task_N/`` layouts).
    A stem that occurs in several sub-directories is ambiguous and rejected.
    """

    def __init__(self, root: str | pathlib.Path) -> None:
        self.root = pathlib.Path(root).resolve()
        self._by_rel: dict[str, pathlib.Path] = {}
        self._by_stem: dict[str, list[pathlib.Path]] = {}
        for p in sorted(self.root.rglob("*.h5")):
            rel = p.relative_to(self.root).with_suffix("").as_posix()
            self._by_rel[rel] = p
            self._by_stem.setdefault(p.stem, []).append(p)

    def __len__(self) -> int:
        return len(self._by_rel)

    def resolve(self, trajectory_id: str) -> pathlib.Path:
        if trajectory_id in self._by_rel:
            return self._by_rel[trajectory_id]
        cands = self._by_stem.get(trajectory_id, [])
        if len(cands) == 1:
            return cands[0]
        if not cands:
            raise KeyError(f"no H5 under {self.root} for trajectory_id {trajectory_id!r}")
        raise KeyError(
            f"trajectory_id {trajectory_id!r} is ambiguous under {self.root}: {cands}"
        )


def iter_steps(h5: h5py.File) -> Iterator[tuple[int, h5py.Group]]:
    """Yield ``(step_idx, group)`` in numeric order for ``step_XXXX`` groups."""
    names = []
    for name in h5.keys():
        if not name.startswith("step_"):
            continue
        suffix = name.split("_", 1)[1]
        if suffix.isdigit():
            names.append((int(suffix), name))
    for idx, name in sorted(names):
        yield idx, h5[name]


@dataclasses.dataclass(frozen=True)
class ProjectionArgs:
    seed: int
    d: int = DEFAULT_D
    p: float = DEFAULT_P
    input_dim: int = DEFAULT_INPUT_DIM


# ------------------------------------------------------------------
# Frozen protocol constants (plan §3.7 / §3.9 / §3.11)
# ------------------------------------------------------------------

#: Every library entry's cached chunk (Pi0.5 LIBERO: horizon 10, action dim 32).
ACTION_CHUNK_SHAPE = (10, 32)

#: Per-tier cap on IR-addressed target arms (§3.9); the fixed reference arm
#: (``theta_raw = 0.85``) is on top of the cap. 8 + 1 + 7 + 1 = 17 arms/group.
TIER_TARGET_CAP: dict[str, int] = {"n0": 8, "n1": 7}
GROUP_ARM_CAP = 17

#: Completeness-gate constants (mirrors exp/rit_pareto/ops/audit_k3_group.py):
#: a ``failed`` episode whose client-side step count is below the suite's
#: policy-step cap was truncated by a client exception, not a real failure,
#: and the per-step hit-row floor a full failed episode must reach.
STEP_CAP: dict[str, int] = {"libero_spatial": 200, "libero_10": 500}
MIN_HIT_ROWS: dict[str, int] = {"libero_spatial": 42, "libero_10": 100}


def suite_from_tag(tag: str) -> str:
    for suite, t in SUITE_TAGS.items():
        if t == tag:
            return suite
    raise KeyError(f"unknown suite tag {tag!r}")


# ------------------------------------------------------------------
# Model provenance: full-content digest of a checkpoint directory
# ------------------------------------------------------------------


def weights_digest(checkpoint_dir: str | pathlib.Path) -> dict:
    """Falsifiable identity of a checkpoint directory.

    ``weights_digest`` is one sha256 over every regular file under the
    directory in sorted relative-path order — header ``<rel>:<size>:`` then
    the *complete* file content — so any byte of any weight file changes it.
    (Hashing ~7 GB costs ~10 s per build/shadow/bench run; a partial digest
    was rejected at G2 R1 because a tail-modified weight file kept its id.)
    """
    root = pathlib.Path(checkpoint_dir).resolve()
    files = sorted(p for p in root.rglob("*") if p.is_file())
    h = hashlib.sha256()
    total = 0
    for p in files:
        rel = p.relative_to(root).as_posix()
        size = p.stat().st_size
        total += size
        h.update(f"{rel}:{size}:".encode())
        with open(p, "rb") as f:
            while chunk := f.read(8 * 1024 * 1024):
                h.update(chunk)
    return {"checkpoint_dir": str(root), "files": len(files), "bytes": total,
            "weights_digest": h.hexdigest()}


def assert_model_binding(artifact_model: dict | None, checkpoint_dir: str | pathlib.Path) -> dict:
    """Fail closed unless ``checkpoint_dir`` re-derives the artifact's ``model.weights_digest``."""
    here = weights_digest(checkpoint_dir)
    want = (artifact_model or {}).get("weights_digest")
    if not want:
        raise SystemExit("artifact carries no model.weights_digest; rebuild it with build_cp2_artifact")
    if here["weights_digest"] != want:
        raise SystemExit(
            f"model binding failed: {checkpoint_dir} digest {here['weights_digest'][:16]}... != "
            f"artifact model.weights_digest {want[:16]}... ({(artifact_model or {}).get('checkpoint_dir')})"
        )
    return here


# ------------------------------------------------------------------
# CP2 arm contract (plan §3.4 / §3.5) on a loaded CacheConfig
# ------------------------------------------------------------------

_NORMALIZATION = {"method": "affine_clip", "params": {"lo": -1.0, "hi": 1.0}}


def cp2_tier_of_config(cfg) -> str | None:
    """``"n0"`` (threshold-only judge) / ``"n1"`` (FULL disabled + one warm tier) / None."""
    cp2 = cfg.checkpoints.get("cp2")
    if cp2 is None or cp2.judge.type != "threshold":
        return None
    tiers = list(cp2.judge.warm_tiers or [])
    if not tiers:
        return "n0" if 0.0 <= float(cp2.judge.threshold) <= 1.0 else None
    if len(tiers) == 1 and abs(float(cp2.judge.threshold) - N1_FULL_THRESHOLD) < 1e-12 \
            and abs(float(tiers[0].get("start_t", -1)) - WARM_START_T) < 1e-9 \
            and 0.0 <= float(tiers[0].get("threshold", -1)) <= 1.0:
        return "n1"
    return None


def cp2_contract_problems(cfg) -> list[str]:
    """Every §3.4 / §3.5 clause a deployed CP2 arm must satisfy; [] when clean.

    Config validation (R-CP2) guarantees the structural rules (exclusive cp2,
    builder / key pairing, in_memory preload, write never); this adds the
    *experiment* protocol on top: single-step top-1 over the whole suite,
    cosine + ``affine_clip(-1, 1)``, and the two N_hit judge shapes.
    """
    from openpi.cache.types import CACHE_QUERY_FIELDS

    p: list[str] = []
    if not cfg.enabled:
        p.append("cache config must be enabled")
    cps = sorted(n for n in cfg.checkpoints if not str(n).startswith("_"))
    if cps != ["cp2"]:
        return [f"checkpoints {cps} != ['cp2']"]
    if cfg.key_builder.type != KEY_BUILDER_TYPE:
        p.append(f"key_builder.type {cfg.key_builder.type!r} != {KEY_BUILDER_TYPE!r}")
    enabled = [n for n in CACHE_QUERY_FIELDS if getattr(getattr(cfg.keys, n, None), "enabled", False)]
    if enabled != [FIELD]:
        p.append(f"enabled keys {enabled} != [{FIELD!r}]")
    key_weight = getattr(cfg.keys.vlm_out, "weight", None)
    if isinstance(key_weight, bool) or not isinstance(key_weight, (int, float)) or float(key_weight) != 1.0:
        p.append(f"keys.{FIELD}.weight {key_weight!r} != 1.0")
    d = getattr(cfg.key_builder.cp2_vlm, "d", None)
    if cfg.backend.type != "in_memory" or not cfg.backend.in_memory.preload_path:
        p.append("backend must be in_memory with preload_path")
    if cfg.backend.in_memory.index_type != "brute_force":
        p.append(f"backend.in_memory.index_type {cfg.backend.in_memory.index_type!r} != 'brute_force'")
    if dict(cfg.backend.vector_dims) != {FIELD: d}:
        p.append(f"vector_dims {dict(cfg.backend.vector_dims)} != {{{FIELD!r}: {d}}}")
    cp2 = cfg.checkpoints["cp2"]
    if not cp2.enabled:
        p.append("checkpoints.cp2.enabled must be true")
    ss = cp2.search_strategy
    if ss.type != "weighted_score_sum_knn":
        p.append(f"search_strategy.type {ss.type!r}")
    if ss.top_k != 1:
        p.append(f"search_strategy.top_k {ss.top_k} != 1")
    if ss.step_filter != "all":
        p.append(f"search_strategy.step_filter {ss.step_filter!r} != 'all'")
    if getattr(ss, "task_scoped", True) is not False:
        p.append("search_strategy.task_scoped must be false (whole-suite library, D14)")
    if getattr(ss, "trajectory_depth", 1) != 1:
        p.append(f"search_strategy.trajectory_depth {ss.trajectory_depth} != 1")
    fs = ss.field_similarity or {}
    if sorted(fs) != [FIELD] or getattr(fs.get(FIELD), "type", None) != "cosine":
        p.append(f"field_similarity must be exactly {{{FIELD!r}: cosine}}, got {sorted(fs)}")
    sn = ss.score_normalization
    fields = dict(getattr(sn, "fields", None) or {}) if sn is not None else {}
    if sn is None or sn.type != "per_field" or sorted(fields) != [FIELD] or fields.get(FIELD) != _NORMALIZATION:
        p.append(f"score_normalization must be per_field {{{FIELD!r}: {_NORMALIZATION}}}, got {sn}")
    if cp2.gate.type != "always_search":
        p.append(f"gate.type {cp2.gate.type!r} != 'always_search'")
    if cp2.judge.type != "threshold":
        p.append(f"judge.type {cp2.judge.type!r} != 'threshold'")
    elif cp2_tier_of_config(cfg) is None:
        p.append(
            f"judge shape is neither n0 (threshold in [0,1], no warm_tiers) nor n1 "
            f"(threshold == {N1_FULL_THRESHOLD}, exactly one warm tier at start_t={WARM_START_T} "
            f"with threshold in [0,1]): "
            f"threshold={cp2.judge.threshold} warm_tiers={cp2.judge.warm_tiers}"
        )
    if getattr(cp2.judge, "dump", None):
        p.append("judge.dump must be off")
    if cfg.write_policy.type != "never":
        p.append(f"write_policy.type {cfg.write_policy.type!r} != 'never'")
    if getattr(cfg, "routing", None) is not None:
        p.append("routing must be absent")
    if getattr(getattr(cfg, "shadow_teacher", None), "enabled", False):
        p.append("shadow_teacher must be off")
    if getattr(getattr(cfg, "collection", None), "export_collect_meta", False):
        p.append("collection.export_collect_meta must be false")
    return p


__all__ = [
    "ACTION_CHUNK_SHAPE",
    "GROUP_ARM_CAP",
    "MIN_HIT_ROWS",
    "N1_FULL_THRESHOLD",
    "STEP_CAP",
    "TIER_TARGET_CAP",
    "assert_model_binding",
    "cp2_contract_problems",
    "cp2_tier_of_config",
    "suite_from_tag",
    "weights_digest",
    "COST_TABLES",
    "DEFAULT_COST_TABLE",
    "FIELD",
    "H5Index",
    "ID_POLICY",
    "KEY_BUILDER_TYPE",
    "PROTOCOL",
    "ProjectionArgs",
    "SUITE_TAGS",
    "TIERS",
    "WARM_START_T",
    "arm_name",
    "cp2_tier_cost",
    "dump_json",
    "git_commit",
    "iter_steps",
    "load_pickle",
    "miss_cost",
    "parse_arm",
    "sha256_file",
    "theta_norm",
    "theta_raw",
]
