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
import math
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
    """Projection identity of one arm: (seed, d, p, D) plus, for the GR00T
    builder, the ``layout`` block (token_len / feature_dim / state_feat_dim)
    that fixes how ``D`` is laid out. ``layout`` is ``None`` for Pi0.5."""

    seed: int
    d: int = DEFAULT_D
    p: float = DEFAULT_P
    input_dim: int = DEFAULT_INPUT_DIM
    layout: dict | None = None

    @classmethod
    def from_projection_meta(cls, meta: dict) -> "ProjectionArgs":
        """Read the identity back from an artifact's ``projection`` block (``layout`` optional)."""
        layout = meta.get("layout")
        return cls(seed=int(meta["seed"]), d=int(meta["d"]), p=float(meta["p"]),
                   input_dim=int(meta["D"]), layout=dict(layout) if layout else None)

    def builder_block(self, prof: "TeacherProfile") -> dict:
        """The ``key_builder.<block>`` yaml mapping for this projection."""
        if prof.projection_block == "cp2_vlm":
            return {"seed": int(self.seed), "d": int(self.d), "p": float(self.p),
                    "input_dim": int(self.input_dim)}
        if self.layout is None:
            raise ValueError("GR00T projection needs a layout block")
        return {"seed": int(self.seed), "d": int(self.d), "p": float(self.p),
                "token_len": int(self.layout["token_len"]), "feature_dim": int(self.layout["feature_dim"]),
                "state_feat_dim": int(self.layout["state_feat_dim"])}

    def expected_projection_meta(self, prof: "TeacherProfile") -> dict:
        """The metadata the profile's builder emits for this projection (offline verifier)."""
        if prof.projection_block == "cp2_vlm":
            from openpi.cache.components.cp2_vlm_key_builder import get_projection_spec

            return get_projection_spec(self.seed, self.d, self.p, self.input_dim).meta()
        from openpi.cache.groot.cp2_key_builder import GrootCP2TernaryKeyBuilder

        if self.layout is None:
            raise ValueError("GR00T projection needs a layout block")
        return GrootCP2TernaryKeyBuilder(
            seed=self.seed, d=self.d, p=self.p, token_len=int(self.layout["token_len"]),
            feature_dim=int(self.layout["feature_dim"]), state_feat_dim=int(self.layout["state_feat_dim"]),
        ).projection_meta()


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

# ------------------------------------------------------------------
# Teacher profiles (plan §3.5): what differs between the Pi0.5 and the GR00T arm
# ------------------------------------------------------------------

GROOT_KEY_BUILDER_TYPE = "cp2_groot_ternary"
GROOT_SCHEDULE_ID = "groot_n15_k8_v1"
GROOT_WARM_START_T = 0.875  # k=8 ascending loop: snapshot 7, one Euler step remaining
SHADOW_COHORT_EPISODES = 150  # 10 tasks x 15 (fit + cal) dev-cohort episodes per suite
COST_FORMULA_PI05 = "pi05_cp2_v1"
COST_FORMULA_GROOT = "groot_cp2_encoded_additive_v1"


@dataclasses.dataclass(frozen=True)
class TeacherProfile:
    """Teacher-specific constants of the CP2 baseline; everything else is shared."""

    name: str
    builder: str
    projection_block: str          # KeyBuilderConfig attribute holding seed/d/p
    warm_start_t: float            # N_hit=1 == this snapshot, one Euler step left
    action_chunk_shape: tuple[int, int]
    denoise_schedule: str | None   # None == Pi0.5 legacy (pi05_v1, descending)
    cost_tables: tuple[str, ...]
    default_cost_table: str
    stage1_paths: tuple[str, ...]
    cost_formula_version: str
    reference_theta_raw: float     # the paper's default T_hit for this teacher

    @property
    def tiers(self) -> dict[str, tuple[str, float | None]]:
        """``{"n0": ("FULL_HIT", None), "n1": ("WARM_START", start_t)}``: the two N_hit tiers' verdicts."""
        return {"n0": ("FULL_HIT", None), "n1": ("WARM_START", self.warm_start_t)}


PI05 = TeacherProfile(
    name="pi05", builder=KEY_BUILDER_TYPE, projection_block="cp2_vlm", warm_start_t=WARM_START_T,
    action_chunk_shape=(10, 32), denoise_schedule=None, cost_tables=("cuda_graph", "eager"),
    default_cost_table="cuda_graph", stage1_paths=("offline", "online"),
    cost_formula_version=COST_FORMULA_PI05, reference_theta_raw=0.85,
)
GROOT_LIBERO = TeacherProfile(
    name="groot_libero", builder=GROOT_KEY_BUILDER_TYPE, projection_block="cp2_groot",
    warm_start_t=GROOT_WARM_START_T, action_chunk_shape=(16, 32), denoise_schedule=GROOT_SCHEDULE_ID,
    cost_tables=("measured",), default_cost_table="measured",
    stage1_paths=("groot_reconstructed_template",), cost_formula_version=COST_FORMULA_GROOT,
    reference_theta_raw=0.65,
)
PROFILES: dict[str, TeacherProfile] = {PI05.name: PI05, GROOT_LIBERO.name: GROOT_LIBERO}
PROFILES_BY_BUILDER: dict[str, TeacherProfile] = {p.builder: p for p in PROFILES.values()}


def profile_for_builder(builder_type: str) -> TeacherProfile | None:
    """The teacher profile whose CP2 key builder is ``builder_type``, or None for a non-CP2 builder."""
    return PROFILES_BY_BUILDER.get(builder_type)


def profile(name: str) -> TeacherProfile:
    """The teacher profile named ``name`` (``pi05`` / ``groot_libero``); ``KeyError`` otherwise."""
    if name not in PROFILES:
        raise KeyError(f"unknown teacher profile {name!r}; known: {sorted(PROFILES)}")
    return PROFILES[name]


# ------------------------------------------------------------------
# Cost records: CP2 numerators and the teacher denominator (plan §3.5, R2-B8)
# ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CostRecord:
    """One priced teacher: ``P = s1 + s2``, ``L`` = full stage-3 loop, ``M = P + L``,
    and the CP2 arm's extra encoder forward ``E`` (0 for Pi0.5, whose key needs
    no extra model call).

    A warm tier is priced by the fraction of the loop still to run. For the
    Pi0.5 descending schedule that fraction *is* ``start_t``; for a GR00T
    ascending schedule it is ``remaining_steps(start_t) / num_steps``, read
    through the ``DenoiseSchedule`` so the direction is never re-derived.
    """

    teacher: str
    table: str
    stage1_ms: float
    stage2_ms: float
    stage3_full_loop_ms: float
    encoder_ms: float = 0.0
    schedule_id: str | None = None
    provenance: dict = dataclasses.field(default_factory=dict)

    @property
    def prefix_ms(self) -> float:
        """``P = s1 + s2``: what every decision pays before stage 3."""
        return self.stage1_ms + self.stage2_ms

    @property
    def teacher_forward_ms(self) -> float:
        """``M = P + L``: the no-cache teacher's per-decision cost."""
        return self.prefix_ms + self.stage3_full_loop_ms

    def stage3_fraction(self, start_t: float) -> float:
        """Fraction of the full stage-3 loop a warm start at ``start_t`` still runs."""
        st = round(float(start_t), 4)
        if self.schedule_id is None:
            if not (0.0 < st < 1.0):
                raise ValueError(f"start_t={start_t} is not a canonical denoise timestep")
            return st
        from openpi.cache.types import schedule_from_id

        schedule = schedule_from_id(self.schedule_id)
        return schedule.remaining_steps(st) / schedule.num_steps


def teacher_forward_cost(record: CostRecord) -> float:
    """The no-cache teacher's per-decision cost ``M`` (the IR denominator)."""
    return record.teacher_forward_ms


def cp2_verdict_cost(record: CostRecord, hit_type: str, start_t: float | None) -> float:
    """Per-decision model-forward cost of a CP2 verdict (the IR numerator).

    FULL_HIT = P + E; WARM_START = P + E + L * fraction(start_t); MISS = M + E.
    With ``E > 0`` an all-MISS arm prices above 100 % of the teacher.
    """
    if hit_type == "FULL_HIT":
        return record.prefix_ms + record.encoder_ms
    if hit_type == "MISS":
        return record.teacher_forward_ms + record.encoder_ms
    if hit_type == "WARM_START":
        if start_t is None:
            raise ValueError("WARM_START needs a start_t")
        return record.prefix_ms + record.encoder_ms + record.stage3_full_loop_ms * record.stage3_fraction(start_t)
    raise ValueError(f"unknown hit_type {hit_type!r}")


def pi05_cost_record(table: str = DEFAULT_COST_TABLE) -> CostRecord:
    """The Pi0.5 tables as records (``E = 0``); identical prices to ``cp2_tier_cost``."""
    t = COST_TABLES[table]
    return CostRecord(teacher=PI05.name, table=table, stage1_ms=t["stage1"], stage2_ms=t["stage2"],
                      stage3_full_loop_ms=t["stage3"], encoder_ms=0.0, schedule_id=None,
                      provenance={"cost_formula_version": COST_FORMULA_PI05})


#: Plan §3.10 / §3.11 acceptance rule on the warm total P95 of a CP2 decision.
VERDICT_OK_MS = 10.0
VERDICT_HALT_MS = 40.0


def preflight_verdict(warm_p95_ms: float | None) -> str:
    """``ok_report`` (<= 10 ms), ``report_with_caption`` (<= 40 ms), ``halt_profile_segments``, or
    ``insufficient_decisions`` when there is no finite warm P95."""
    if (isinstance(warm_p95_ms, bool) or not isinstance(warm_p95_ms, (int, float))
            or not math.isfinite(warm_p95_ms) or warm_p95_ms < 0):
        return "insufficient_decisions"
    if warm_p95_ms <= VERDICT_OK_MS:
        return "ok_report"
    if warm_p95_ms <= VERDICT_HALT_MS:
        return "report_with_caption"
    return "halt_profile_segments"


#: Fields the GR00T encoder-cost record (``bench_cp2_overhead_groot.py``) must carry.
GROOT_ENCODER_RECORD_FIELDS = (
    "suite", "teacher", "cp2_key_encoder_ms", "teacher_cost_record_sha256", "schedule_id",
    "mode", "certified", "valid", "layout", "model", "ckpt_weights_digest", "ckpt_sha256", "n_tokens",
    "warmup", "iters", "gpu_uuid", "hardware", "cudagraph_launch_count", "expected_cudagraph_launch_count",
)
#: The frozen sampling identity of the teacher table (plan §3.11): every E record
#: must have been measured under exactly these, and so must the table itself.
GROOT_E_SAMPLING = {"n_tokens": 566, "warmup": 30, "iters": 200, "mode": "reduce-overhead"}
#: Provenance keys a GR00T cost summary must keep so a downstream consumer can
#: re-bind the pricing (model / hardware / sampling) without the source files.
GROOT_COST_PROVENANCE_KEYS = (
    "suite", "cost_record_sha256", "encoder_cost_record_sha256", "mode", "layout", "model",
    "ckpt_sha256", "teacher_ckpt_sha256", "gpu_uuid", "gpu_name", "n_tokens", "warmup", "iters",
)


def _is_sha256(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _finite_positive(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0.0


def _validate_groot_provenance(provenance: dict) -> None:
    missing = [k for k in GROOT_COST_PROVENANCE_KEYS if k not in provenance]
    if missing:
        raise SystemExit(f"GR00T cost summary lacks provenance {missing}; it cannot be re-bound")
    for key in ("cost_record_sha256", "encoder_cost_record_sha256", "ckpt_sha256", "teacher_ckpt_sha256"):
        if not _is_sha256(provenance[key]):
            raise SystemExit(f"GR00T cost provenance {key} must be a full sha256")
    if provenance["suite"] not in SUITE_TAGS:
        raise SystemExit(f"unknown GR00T cost suite {provenance['suite']!r}")
    if {k: provenance[k] for k in GROOT_E_SAMPLING} != GROOT_E_SAMPLING:
        raise SystemExit("GR00T cost sampling differs from the frozen encoder measurement")
    if (not isinstance(provenance["gpu_uuid"], str) or not provenance["gpu_uuid"].startswith("GPU-")
            or provenance["gpu_name"] != "NVIDIA GeForce RTX 4090"):
        raise SystemExit("GR00T cost provenance must identify the calibration RTX 4090")
    model, layout = provenance["model"], provenance["layout"]
    if not isinstance(model, dict) or not _is_sha256(model.get("weights_digest")):
        raise SystemExit("GR00T cost model.weights_digest must be a full sha256")
    if not isinstance(layout, dict) or layout.get("kind") != "groot_encoded_v1" or not all(
            isinstance(layout.get(k), int) and not isinstance(layout[k], bool) and layout[k] > 0
            for k in ("token_len", "feature_dim", "state_feat_dim")):
        raise SystemExit("GR00T cost layout must describe positive groot_encoded_v1 dimensions")


def groot_cost_record(cost_json: str | pathlib.Path, encoder_json: str | pathlib.Path,
                      *, suite: str) -> CostRecord:
    """Bind the owner's measured GR00T stage costs to this suite's encoder cost ``E``.

    Fail-closed on every identity the two records share (G2-B4): the encoder
    record must name this suite and teacher, carry the sha256 of the very cost
    table it was measured against, its own checkpoint and the same GPU
    (``gpu_uuid`` / name) as that table, the
    frozen sampling (``GROOT_E_SAMPLING``: N=566, 30 warmup, 200 iterations,
    CUDA-Graph ``reduce-overhead``) which the table must also declare, a
    certified *and* valid CUDA-graph replay (launch count == expected > 0), a
    finite ``cp2_key_encoder_ms > 0`` and a full-content ``weights_digest`` of
    the model. ``E`` is never defaulted to zero; the returned record's
    provenance keeps everything a consumer needs to re-bind it. The frozen
    teacher table is shared by both suites; its calibration checkpoint is
    recorded separately from each suite's encoder checkpoint (plan §3.11).
    """
    cost_path, enc_path = pathlib.Path(cost_json), pathlib.Path(encoder_json)
    cost = json.loads(cost_path.read_text(encoding="utf-8"))
    enc = json.loads(enc_path.read_text(encoding="utf-8"))
    missing = [k for k in GROOT_ENCODER_RECORD_FIELDS if k not in enc]
    if missing:
        raise SystemExit(f"{enc_path}: encoder cost record lacks {missing}")
    cost_sha = sha256_file(cost_path)
    if enc["teacher_cost_record_sha256"] != cost_sha:
        raise SystemExit(f"{enc_path}: teacher_cost_record_sha256 does not match {cost_path}")
    if enc["suite"] != suite:
        raise SystemExit(f"{enc_path}: encoder cost record is for suite {enc['suite']!r}, not {suite!r}")
    if enc["teacher"] != GROOT_LIBERO.name or cost.get("teacher") != GROOT_LIBERO.name:
        raise SystemExit(f"cost records must be for teacher {GROOT_LIBERO.name!r}")
    if cost.get("schedule_id") != GROOT_SCHEDULE_ID or enc["schedule_id"] != GROOT_SCHEDULE_ID:
        raise SystemExit(f"cost records must be stamped {GROOT_SCHEDULE_ID!r}")
    e = enc["cp2_key_encoder_ms"]
    if not _finite_positive(e):
        raise SystemExit(f"{enc_path}: cp2_key_encoder_ms must be a finite number > 0, got {e!r}")
    if enc["mode"] != cost.get("mode") or not enc["certified"] or not enc["valid"] or not cost.get("certified"):
        raise SystemExit(f"{enc_path}: encoder record must be certified and valid under the same mode "
                         f"({cost.get('mode')!r}) as the certified teacher table")
    launches, expected = enc["cudagraph_launch_count"], enc["expected_cudagraph_launch_count"]
    if not isinstance(launches, int) or launches <= 0 or launches != expected:
        raise SystemExit(f"{enc_path}: cudagraph_launch_count {launches!r} != expected {expected!r} (> 0)")
    # Frozen sampling identity, on both records, against the plan's constants.
    table_sampling = {"n_tokens": cost.get("prompt_shape_n"), "warmup": cost.get("warmup"),
                      "iters": cost.get("iters"), "mode": cost.get("mode")}
    enc_sampling = {k: enc.get(k) for k in GROOT_E_SAMPLING}
    if table_sampling != GROOT_E_SAMPLING:
        raise SystemExit(f"{cost_path}: teacher table sampling {table_sampling} != frozen {GROOT_E_SAMPLING}")
    if enc_sampling != GROOT_E_SAMPLING:
        raise SystemExit(f"{enc_path}: encoder sampling {enc_sampling} != frozen {GROOT_E_SAMPLING}")
    # Suite checkpoints differ; both identities survive alongside the shared table.
    if not _is_sha256(enc["ckpt_sha256"]) or not _is_sha256(cost.get("ckpt_sha256")):
        raise SystemExit("encoder and teacher table must each identify their checkpoint by sha256")
    if not enc["gpu_uuid"] or enc["gpu_uuid"] != cost.get("gpu_uuid"):
        raise SystemExit(f"{enc_path}: gpu_uuid {enc['gpu_uuid']!r} != teacher table {cost.get('gpu_uuid')!r}")
    gpu_name = (enc.get("hardware") or {}).get("gpu")
    if not gpu_name or gpu_name != cost.get("gpu_name"):
        raise SystemExit(f"{enc_path}: hardware.gpu {gpu_name!r} != teacher table gpu_name {cost.get('gpu_name')!r}")
    model = enc["model"] if isinstance(enc["model"], dict) else {}
    if not _is_sha256(model.get("weights_digest")) or model.get("weights_digest") != enc["ckpt_weights_digest"]:
        raise SystemExit(f"{enc_path}: model.weights_digest must be a full-content sha256 equal to ckpt_weights_digest")
    layout = enc["layout"] if isinstance(enc["layout"], dict) else {}
    if layout.get("kind") != "groot_encoded_v1" or not all(
            isinstance(layout.get(k), int) and layout[k] > 0 for k in ("token_len", "feature_dim", "state_feat_dim")):
        raise SystemExit(f"{enc_path}: layout {layout!r} is not a groot_encoded_v1 layout")
    for k in ("stage1_ms", "stage2_ms", "stage3_full_loop_ms"):
        if not _finite_positive(cost.get(k)):
            raise SystemExit(f"{cost_path}: {k} missing or non-positive")
    record = CostRecord(
        teacher=GROOT_LIBERO.name, table="measured", stage1_ms=float(cost["stage1_ms"]),
        stage2_ms=float(cost["stage2_ms"]), stage3_full_loop_ms=float(cost["stage3_full_loop_ms"]),
        encoder_ms=float(e), schedule_id=GROOT_SCHEDULE_ID,
        provenance={
            "cost_formula_version": COST_FORMULA_GROOT, "suite": suite,
            "cost_record": str(cost_path.resolve()), "cost_record_sha256": cost_sha,
            "encoder_cost_record": str(enc_path.resolve()), "encoder_cost_record_sha256": sha256_file(enc_path),
            "mode": enc["mode"], "layout": layout, "model": model, "ckpt_sha256": enc["ckpt_sha256"],
            "teacher_ckpt_sha256": cost["ckpt_sha256"],
            "gpu_uuid": enc["gpu_uuid"], "gpu_name": gpu_name,
            "n_tokens": int(enc["n_tokens"]), "warmup": int(enc["warmup"]), "iters": int(enc["iters"]),
        },
    )
    _validate_groot_provenance(record.provenance)
    return record


def cost_record_summary(record: CostRecord, prof: TeacherProfile) -> dict:
    """What an export / aggregate record stores about the pricing (plan §3.5)."""
    tiers = prof.tiers
    return {
        "teacher": record.teacher, "cost_table": record.table, "schedule_id": record.schedule_id,
        "cost_formula_version": prof.cost_formula_version,
        "teacher_forward_ms": record.teacher_forward_ms, "encoder_ms": record.encoder_ms,
        "stage_ms": {"stage1": record.stage1_ms, "stage2": record.stage2_ms,
                     "stage3_full_loop": record.stage3_full_loop_ms},
        "verdict_unit_ms": {
            "FULL_HIT": cp2_verdict_cost(record, *tiers["n0"]),
            f"WARM_START@{tiers['n1'][1]:g}": cp2_verdict_cost(record, *tiers["n1"]),
            "MISS": cp2_verdict_cost(record, "MISS", None),
        },
        **{k: v for k, v in record.provenance.items() if k != "cost_formula_version"},
    }


def cost_record_from_summary(summary: dict) -> CostRecord:
    """Rebuild the record an export/aggregate record stored (no re-measurement)."""
    teacher = summary.get("teacher")
    if teacher == PI05.name:
        return pi05_cost_record(summary.get("cost_table", DEFAULT_COST_TABLE))
    if teacher == GROOT_LIBERO.name:
        _validate_groot_provenance(summary)
        prov = {k: summary[k] for k in GROOT_COST_PROVENANCE_KEYS}
        if summary.get("cost_formula_version") != COST_FORMULA_GROOT:
            raise SystemExit(f"unknown GR00T cost formula {summary.get('cost_formula_version')!r}")
        if not _finite_positive(summary.get("encoder_ms")):
            raise SystemExit(f"GR00T cost summary encoder_ms={summary.get('encoder_ms')!r} is not a finite positive E")
        if summary.get("cost_table") != "measured" or summary.get("schedule_id") != GROOT_SCHEDULE_ID:
            raise SystemExit("GR00T cost summary must use the measured teacher8 table")
        st = summary.get("stage_ms")
        if not isinstance(st, dict) or not all(_finite_positive(st.get(k)) for k in (
                "stage1", "stage2", "stage3_full_loop")):
            raise SystemExit("GR00T cost summary stage times must be finite and positive")
        record = CostRecord(teacher=teacher, table=summary["cost_table"], stage1_ms=float(st["stage1"]),
                          stage2_ms=float(st["stage2"]), stage3_full_loop_ms=float(st["stage3_full_loop"]),
                          encoder_ms=float(summary["encoder_ms"]), schedule_id=summary["schedule_id"],
                          provenance={"cost_formula_version": COST_FORMULA_GROOT, **prov})
        recomputed = cost_record_summary(record, GROOT_LIBERO)
        for key in ("teacher_forward_ms", "verdict_unit_ms"):
            expected = recomputed[key] if isinstance(recomputed[key], dict) else {key: recomputed[key]}
            actual = summary.get(key) if isinstance(recomputed[key], dict) else {key: summary.get(key)}
            if not isinstance(actual, dict) or actual.keys() != expected.keys() or any(
                    not _finite_positive(actual[k]) or not math.isclose(actual[k], value, rel_tol=1e-12)
                    for k, value in expected.items()):
                raise SystemExit(f"GR00T cost summary {key} differs from recomputed stage costs")
        return record
    raise SystemExit(f"cost summary names unknown teacher {teacher!r}")


_NORMALIZATION = {"method": "affine_clip", "params": {"lo": -1.0, "hi": 1.0}}


def cp2_tier_of_config(cfg, profile: "TeacherProfile | None" = None) -> str | None:
    """``"n0"`` (threshold-only judge) / ``"n1"`` (FULL disabled + one warm tier) / None.

    The N_hit=1 tier's ``start_t`` is the teacher's (Pi0.5 0.1, GR00T 0.875);
    without an explicit ``profile`` it is inferred from ``key_builder.type``.
    """
    if profile is None:
        profile = profile_for_builder(cfg.key_builder.type)
        if profile is None:
            return None
    cp2 = cfg.checkpoints.get("cp2")
    if cp2 is None or cp2.judge.type != "threshold":
        return None
    tiers = list(cp2.judge.warm_tiers or [])
    if not tiers:
        return "n0" if 0.0 <= float(cp2.judge.threshold) <= 1.0 else None
    if len(tiers) == 1 and abs(float(cp2.judge.threshold) - N1_FULL_THRESHOLD) < 1e-12 \
            and abs(float(tiers[0].get("start_t", -1)) - profile.warm_start_t) < 1e-9 \
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
    profile = profile_for_builder(cfg.key_builder.type)
    if profile is None:
        return [
            f"key_builder.type {cfg.key_builder.type!r} is not a known CP2 teacher builder "
            f"({sorted(PROFILES_BY_BUILDER)})"
        ]
    if profile.denoise_schedule is not None and getattr(cfg, "denoise_schedule", None) != profile.denoise_schedule:
        p.append(
            f"denoise_schedule {getattr(cfg, 'denoise_schedule', None)!r} != {profile.denoise_schedule!r} "
            f"({profile.name} warm tiers are stamped under that loop)"
        )
    enabled = [n for n in CACHE_QUERY_FIELDS if getattr(getattr(cfg.keys, n, None), "enabled", False)]
    if enabled != [FIELD]:
        p.append(f"enabled keys {enabled} != [{FIELD!r}]")
    key_weight = getattr(cfg.keys.vlm_out, "weight", None)
    if isinstance(key_weight, bool) or not isinstance(key_weight, (int, float)) or float(key_weight) != 1.0:
        p.append(f"keys.{FIELD}.weight {key_weight!r} != 1.0")
    d = getattr(getattr(cfg.key_builder, profile.projection_block), "d", None)
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
    elif cp2_tier_of_config(cfg, profile) is None:
        p.append(
            f"judge shape is neither n0 (threshold in [0,1], no warm_tiers) nor n1 "
            f"(threshold == {N1_FULL_THRESHOLD}, exactly one warm tier at "
            f"start_t={profile.warm_start_t} with threshold in [0,1]): "
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
    "COST_FORMULA_GROOT",
    "COST_FORMULA_PI05",
    "CostRecord",
    "GROOT_ENCODER_RECORD_FIELDS",
    "GROOT_KEY_BUILDER_TYPE",
    "GROOT_LIBERO",
    "GROOT_SCHEDULE_ID",
    "GROOT_WARM_START_T",
    "PI05",
    "PROFILES",
    "PROFILES_BY_BUILDER",
    "TeacherProfile",
    "cost_record_from_summary",
    "cost_record_summary",
    "cp2_verdict_cost",
    "groot_cost_record",
    "pi05_cost_record",
    "profile",
    "profile_for_builder",
    "teacher_forward_cost",
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
