"""Rebuild the RIT shadow table offline, leave-one-trajectory-out, from the W13 corpus.

What one row is
---------------
For every decision (one ``step_XXXX`` group) of every collected episode:

  1. the CP1 query key is built from the stored stage-1 slices with the *same*
     builder chain that built the library entries, then filtered to the fields
     the template yaml enables, and searched with the production strategy over
     the full S3 library (``fusion_weights`` carries every field, disabled
     ones at 0.0, so a disabled field can never leak into the score);
  2. if the episode's trajectory sits in the library, hits from that trajectory
     are skipped and the best *other* hit is the winner (LOTO). The fused score
     is a per-pair quantity, so skipping is identical to searching a library
     that never held the trajectory; ``--orchestrator-check`` and the unit
     tests pin that claim rather than assume it;
  3. the winner's stored intermediates are resumed under the reconstructed
     stage-1 of the current observation (template + stored slices,
     ``cp2_reconstruct``), one continuation per warm rung;
  4. every rung is compared with the chunk the teacher executed at collection
     time (``clean_action``) through ``weighted_chunk_deviation`` with the
     library-wide ``W`` (Eq. shadow-deviation), over the executed window.

Parity gate
-----------
The action-head continuation runs on a *reconstructed* stage-1. Before any fit
is trusted, ``--parity-only`` replays a stratified sample of decisions from
their own stored noise / snapshots and measures how far the offline replay
lands from the stored ``clean_action`` on the same W / mask / H_exec scale as
the table. The gate compares that against the two-seed noise floor
(``noise_floor.py``) and writes ``parity_gate.json``; the full table run and
``fit_loto.py`` refuse to proceed without a matching PASS.

Environment: the GR00T island (torch, no jax). Retrieval helpers are also
importable in the main venv for tests; model loading is lazy.

Usage (weilandserver, island venv):
  python -m exp.rit_loto.build_loto_table --suite libero_10 \
      --corpus-dir /archive/libero_cache/build_libero10_w13/libero_10 \
      --library-pkl /data/libero_cache/libraries_w13/libero_10/libero_10_w13_S3.pkl \
      --template-yaml exp/libero_groot/config/rit/libero_10/template.yaml \
      --checkpoint /home/weiland/ckpt_n15_libero_10 \
      --noise-floor-record <data>/libero_10/noise_floor.json \
      --parity-only --out-dir <data>/libero_10
  python -m exp.rit_loto.build_loto_table ... --parity-gate <data>/libero_10/parity_gate.json \
      --out-dir <data>/libero_10 --orchestrator-check 200
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import pathlib
import pickle
import subprocess
import sys
import time
from typing import Any

import h5py
import numpy as np
import torch

from openpi.cache.backends.in_memory_backend import InMemoryBackend
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.components.search_strategy import SearchContext
from openpi.cache.components.surface_judge import weighted_chunk_deviation
from openpi.cache.config import _build_search_strategy, _keys_iter, load_cache_config
from openpi.cache.types import CheckpointID, DenoiseSchedule, groot_n15_schedule, schedule_from_id
from openpi.collect.h5_intermediates import episode_schedule

from exp.libero_groot.cp2_reconstruct import TemplateCache, reconstruct_stage1
from exp.robocasa365.rit_shadow import library_action_weights

logger = logging.getLogger("rit_loto.build")

SUITES = ("libero_spatial", "libero_10")
DEFAULT_WARM_TS = (0.75, 0.5)
DEFAULT_H_EXEC = 5
DEFAULT_DENOISING_STEPS = 8
NUM_TASKS = 10
TRIALS_PER_TASK = 50
DEFAULT_PARITY_SAMPLE = 200
PARITY_RATIO = 0.1
PARITY_MIN_ROWS = 200
PARITY_MIN_ROWS_PER_TASK = 20
PARITY_TIERS = ("full", "warm75", "warm50")
PARITY_WARM_TS = {"warm75": 0.75, "warm50": 0.5}
NOISE_CONTRACT = "rit_loto_noise_v1"
#: Fields whose equality binds a parity gate / noise floor to one table run. ``code_sha256``
#: is a dict of content digests, so a changed forward or key path invalidates an old gate.
IDENTITY_KEYS = (
    "suite", "library_sha256", "template_sha256", "checkpoint_identity_sha256",
    "corpus_manifest_sha256", "weights_sha256", "h_exec", "schedule_id", "code_sha256",
)
#: The LIBERO GR00T action chunk: 16 steps x 32 (7 live + padding) dims.
ACTION_SHAPE = (16, 32)
#: Source files whose content hash goes into every record (git commit alone does
#: not describe an uncommitted working tree).
CODE_FILES = (
    "exp/rit_loto/build_loto_table.py",
    "exp/rit_loto/noise_floor.py",
    "exp/rit_loto/fit_loto.py",
    "exp/rit_loto/loto_logger.py",
    "exp/rit_loto/emit_verify_arm.py",
    "exp/rit_loto/verify_closed_loop.py",
    "exp/libero_groot/serve_groot_libero.py",
    "exp/libero_groot/cp2_reconstruct.py",
    "exp/robocasa365/rit_shadow.py",
    "exp/common/build_in_memory_cache_artifact.py",
    "src/openpi/cache/groot/staged.py",
    "src/openpi/cache/groot/interceptor.py",
    "src/openpi/cache/groot/load_guard.py",
    "src/openpi/cache/orchestrator.py",
    "src/openpi/cache/groot/key_builder.py",
    "src/openpi/cache/components/surface_judge.py",
    "src/openpi/cache/components/search_strategy.py",
    "src/openpi/cache/backends/in_memory_backend.py",
    "src/openpi/cache/config.py",
    "exp/rit_pareto/rit_k.py",
    "exp/robocasa365/emit_rit_rc.py",
    "exp/robocasa365/rit_cost_rc.py",
    "exp/libero_groot/emit_rit_arms.py",
)
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------
# Hashing, seeds, small io
# ------------------------------------------------------------------


def sha256_file(path: str | pathlib.Path) -> str:
    """Hex sha256 of a file's bytes, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, UTF-8 characters kept."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_canonical(obj: Any) -> str:
    """Hex sha256 of ``canonical_json(obj)``."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def derive_seed(root_seed: int, identity: dict) -> int:
    """Row-stable seed: sha256 of the canonical JSON of ``{root_seed, **identity}``, first 8 bytes big-endian."""
    payload = {"root_seed": int(root_seed), **identity}
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def make_noise(seed: int, shape: tuple[int, ...] = (1, 16, 32)) -> torch.Tensor:
    """Standard-normal fp32 noise on CPU from a per-row generator; the runner moves it to the model."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return torch.randn(shape, generator=gen, dtype=torch.float32)


def git_commit() -> str:
    """HEAD commit of the repo, or ``"unknown"`` outside git; never a substitute for ``code_sha256``."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def code_sha256(files: tuple[str, ...] = CODE_FILES) -> dict[str, str]:
    """Content digest of every source file the numbers depend on; a missing file is an error, not ``None``."""
    out = {}
    for rel in files:
        path = REPO_ROOT / rel
        if not path.is_file():
            raise SystemExit(f"code identity: {rel} is missing from {REPO_ROOT}")
        out[rel] = sha256_file(path)
    return out


def require_code_identity(recorded: dict) -> None:
    """Reject absent, incomplete or stale source digests before consuming a frozen artifact."""
    actual = code_sha256()
    if recorded != actual:
        old = recorded if isinstance(recorded, dict) else {}
        changed = sorted(k for k in set(actual) | set(old) if actual.get(k) != old.get(k))
        raise SystemExit(f"code identity differs from the frozen artifact: {changed}")


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    """Write ``rows`` as one JSON object per line (parent directory created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | pathlib.Path) -> list[dict]:
    """Read a JSONL file, skipping blank lines."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ------------------------------------------------------------------
# Corpus and checkpoint identity
# ------------------------------------------------------------------


def corpus_files(corpus_dir: str | pathlib.Path) -> list[pathlib.Path]:
    """Sorted ``episode_*.h5`` paths of a collection directory; empty is an error."""
    files = sorted(pathlib.Path(corpus_dir).glob("episode_*.h5"))
    if not files:
        raise SystemExit(f"no episode_*.h5 under {corpus_dir}")
    return files


def corpus_manifest(files: list[pathlib.Path]) -> dict:
    """Cheap corpus identity: sorted (stem, size, mtime_ns); hashing 60 GB per run is not worth it."""
    entries = []
    for p in files:
        st = p.stat()
        entries.append({"stem": p.stem, "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)})
    return {"definition": "sha256(canonical_json(sorted [{stem,size,mtime_ns}]))",
            "n_files": len(entries), "sha256": sha256_canonical(entries), "entries": entries}


def _content_sha256_of_dir(root: pathlib.Path, listing: list[dict]) -> str:
    """sha256 over the concatenated bytes of every file in ``listing`` order, each prefixed by its path."""
    h = hashlib.sha256()
    for item in listing:
        h.update(canonical_json({"path": item["path"], "size": item["size"]}).encode("utf-8") + b"\0")
        with open(root / item["path"], "rb") as f:
            while chunk := f.read(1 << 24):
                h.update(chunk)
    return h.hexdigest()


def checkpoint_identity(ckpt_dir: str | pathlib.Path, *, cache_dir: str | pathlib.Path | None = None) -> dict:
    """Content identity of a checkpoint directory: sha256 over every file's bytes.

    The audit record is cached under ``cache_dir`` (default
    ``~/.cache/rit_loto/ckpt_identity``), keyed by the resolved path and file
    metadata. Every admission rehashes the contents: timestamp granularity and
    restored mtimes make metadata-only cache hits insufficient evidence of
    unchanged model weights. ``cache_hit`` means the cached digest was verified.
    """
    root = pathlib.Path(ckpt_dir).resolve()
    if not root.is_dir():
        raise SystemExit(f"checkpoint dir missing: {root}")
    listing = sorted(
        ({"path": str(p.relative_to(root)), "size": int(p.stat().st_size), "mtime_ns": int(p.stat().st_mtime_ns),
          "ctime_ns": int(p.stat().st_ctime_ns), "inode": int(p.stat().st_ino)}
         for p in root.rglob("*") if p.is_file()),
        key=lambda d: d["path"],
    )
    if not listing:
        raise SystemExit(f"checkpoint dir is empty: {root}")
    listing_sha = sha256_canonical({"version": 2, "root": str(root), "listing": listing})
    cache_root = pathlib.Path(cache_dir) if cache_dir else pathlib.Path.home() / ".cache" / "rit_loto" / "ckpt_identity"
    cache_file = cache_root / f"{listing_sha}.json"
    content = _content_sha256_of_dir(root, listing)
    if cache_file.is_file():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached.get("listing_sha256") == listing_sha and cached.get("sha256") == content:
            return {**cached, "cache_hit": True}
    out = {"path": str(root), "n_files": len(listing), "bytes": int(sum(i["size"] for i in listing)),
           "listing_sha256": listing_sha, "sha256": content, "cache_hit": False}
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(out) + "\n", encoding="utf-8")
    return out


def step_groups(h5: h5py.File) -> list[tuple[int, str]]:
    """``(step_idx, group name)`` of every decision, contiguous from 0, else RuntimeError."""
    names = [k for k in h5.keys() if k.startswith("step_")]
    out = []
    for name in names:
        suffix = name.split("_", 1)[1]
        if not suffix.isdigit():
            raise RuntimeError(f"{h5.filename}: unexpected step group {name!r}")
        out.append((int(suffix), name))
    out.sort()
    if [i for i, _ in out] != list(range(len(out))):
        raise RuntimeError(f"{h5.filename}: step groups are not contiguous from 0")
    return out


def episode_attrs(h5: h5py.File, stem: str) -> dict:
    """The episode identity attrs of a collector file; missing required attrs raise."""
    a = h5.attrs
    for key in ("task", "task_id", "success", "num_steps"):
        if key not in a:
            raise RuntimeError(f"{stem}: missing attr {key!r}")
    return {
        "trajectory_id": stem,
        "task": str(a["task"]),
        "task_id": int(a["task_id"]),
        "orig_init_state_idx": int(a["orig_init_state_idx"]) if "orig_init_state_idx" in a else None,
        "episode_id": int(a["episode_id"]) if "episode_id" in a else None,
        "success": bool(a["success"]),
        "num_steps": int(a["num_steps"]),
    }


# ------------------------------------------------------------------
# Library
# ------------------------------------------------------------------


@dataclasses.dataclass
class Library:
    """A loaded S3 artifact plus the lookups the LOTO search needs."""

    entries: list
    by_id: dict[str, Any]
    traj_of: dict[str, str]
    trajectory_ids: set[str]
    per_task_entries: dict[str, int]
    meta: dict
    sha256: str
    schedule: DenoiseSchedule


def load_library(pkl: str | pathlib.Path, *, expected_schedule: DenoiseSchedule,
                 warm_ts: tuple[float, ...]) -> Library:
    """Load the artifact; every payload must prove its own schedule identity before any snapshot is resumed.

    Checks per entry (``CachePayload.validate_for_warm_start`` plus the fixed
    LIBERO chunk shape): ``schedule_id`` equals the expected schedule,
    ``denoising_num_steps`` matches, a snapshot exists at every warm ``t`` with
    the chunk's shape and dtype, chunk and snapshots are ``[16, 32]`` and finite.
    """
    path = pathlib.Path(pkl)
    with open(path, "rb") as f:
        art = pickle.load(f)
    schedule_id = art.get("schedule_id")
    if schedule_id is None:
        raise SystemExit(f"{path}: artifact carries no schedule_id; refusing to guess one")
    schedule = schedule_from_id(str(schedule_id))
    if schedule != expected_schedule:
        raise SystemExit(f"{path}: library schedule {schedule.schedule_id} != expected {expected_schedule.schedule_id}")
    entries = art["entries"]
    by_id, traj_of, per_task = {}, {}, {}
    for e in entries:
        pl = e.payload
        pl.action_chunk = torch.as_tensor(pl.action_chunk).float().contiguous()
        if tuple(pl.action_chunk.shape) != ACTION_SHAPE or not torch.isfinite(pl.action_chunk).all():
            raise SystemExit(f"{path}: entry {e.id} action_chunk is not a finite {ACTION_SHAPE} chunk")
        if not pl.intermediates:
            raise SystemExit(f"{path}: entry {e.id} has no intermediates; the warm rungs cannot be labelled")
        for t in list(pl.intermediates):
            snap = torch.as_tensor(pl.intermediates[t]).float().contiguous()
            if tuple(snap.shape) != ACTION_SHAPE or not torch.isfinite(snap).all():
                raise SystemExit(f"{path}: entry {e.id} snapshot@{t} is not a finite {ACTION_SHAPE} chunk")
            pl.intermediates[t] = snap
        for t in warm_ts:
            try:
                pl.validate_for_warm_start(expected_schedule, t)
            except ValueError as exc:
                raise SystemExit(f"{path}: entry {e.id} rejected for warm t={t}: {exc}") from exc
        e.query_keys = {k: torch.as_tensor(v).float().contiguous() for k, v in e.query_keys.items()}
        if e.trajectory_id is None:
            raise SystemExit(f"{path}: entry {e.id} has no trajectory_id")
        by_id[e.id] = e
        traj_of[e.id] = str(e.trajectory_id)
        task_key = getattr(pl, "task_key", None) or ""
        per_task[task_key] = per_task.get(task_key, 0) + 1
    meta = {k: v for k, v in art.items() if k not in ("entries", "library_stats", "prompt_pool")}
    return Library(entries=entries, by_id=by_id, traj_of=traj_of, trajectory_ids=set(traj_of.values()),
                   per_task_entries=per_task, meta=meta, sha256=sha256_file(path), schedule=schedule)


# ------------------------------------------------------------------
# Retrieval (importable without the model)
# ------------------------------------------------------------------


def enabled_fields(cfg) -> tuple[list[str], dict[str, float]]:
    """Enabled key names in ``_keys_iter`` order, and the full fusion-weight map with 0.0 for disabled fields."""
    names, weights = [], {}
    for name, key in _keys_iter(cfg.keys):
        weights[name] = float(key.weight) if key.enabled else 0.0
        if key.enabled:
            names.append(name)
    if not names:
        raise SystemExit("template enables no key field")
    return names, weights


def build_storage(cfg, entries) -> CacheStorage:
    """Frozen in-memory storage over ``entries`` with the template's vector dims."""
    backend = InMemoryBackend(vector_dims=cfg.backend.vector_dims)
    storage = CacheStorage(backend)
    storage.batch_insert(entries)
    backend.freeze()
    return storage


def build_retrieval(cfg, storage, weights: dict[str, float], n_hint: int):
    """The production CP1 search strategy with an explicit full fusion-weight map and ``min_top_k_hint``."""
    return _build_search_strategy(
        cfg.checkpoints["cp1"].search_strategy, storage, weights, min_top_k_hint=int(n_hint)
    )


def make_query_builder(builder_type: str):
    """The pooling key builder the library artifact was built with (same class, same reduction)."""
    from exp.common.build_in_memory_cache_artifact import _create_builder

    return _create_builder(builder_type)


def query_keys_for(builder, group, enabled: list[str]) -> dict[str, torch.Tensor]:
    """The CP1 key of one stored step through the library's own builder chain, enabled fields only."""
    from exp.common.build_in_memory_cache_artifact import _build_fake_stage1

    stage1 = _build_fake_stage1(group)
    builder.collect(CheckpointID.CP1, stage1=stage1)
    keys = builder.build(CheckpointID.CP1)
    builder.clear()
    missing = [k for k in enabled if k not in keys]
    if missing:
        raise RuntimeError(f"builder produced no key for enabled fields {missing}")
    return {k: torch.as_tensor(keys[k]).float().contiguous() for k in enabled}


def search(strategy, keys: dict[str, torch.Tensor], step_idx: int, task_key: str):
    """Task-scoped search of ``keys``; hits come back best first."""
    return strategy.search(SearchContext(
        query_keys=keys, checkpoint_id=CheckpointID.CP1, current_step=int(step_idx), task_key=task_key,
    ))


def loto_winner(hits, query_traj: str, in_library: bool, traj_of: dict[str, str]) -> tuple[str, float, int]:
    """First hit not from the query's own trajectory (when it is in the library): (id, score, n_self_skipped)."""
    skipped = 0
    for hit in hits:
        if in_library and traj_of.get(hit.id) == query_traj:
            skipped += 1
            continue
        return str(hit.id), float(hit.score), skipped
    raise RuntimeError(
        f"no admissible hit for trajectory {query_traj!r}: {len(hits)} hits, {skipped} from itself"
    )


def with_permuted_prompt(keys: dict[str, torch.Tensor], seed: int, dim: int = 2048) -> dict[str, torch.Tensor]:
    """The enabled keys plus a random ``prompt_emb``: a disabled field must not move the score."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed) & ((1 << 63) - 1))
    return {**keys, "prompt_emb": torch.randn(dim, generator=gen, dtype=torch.float32)}


# ------------------------------------------------------------------
# Model side (island only)
# ------------------------------------------------------------------


class ModelSide:
    """Served policy, staged runner and per-instruction templates."""

    def __init__(self, checkpoint: str | pathlib.Path, *, denoising_steps: int = DEFAULT_DENOISING_STEPS,
                 device: str = "cuda") -> None:
        from openpi.cache.groot.staged import GrootStagedRunner

        from exp.libero_groot.cp2_reconstruct import load_groot_libero_policy

        self.policy = load_groot_libero_policy(checkpoint, denoising_steps=denoising_steps, device=device)
        self.runner = GrootStagedRunner(self.policy.model)
        self.templates = TemplateCache(self.policy, self.runner)
        self.schedule = self.runner.live_schedule()


def stage2_in_session(runner, templates, task: str, group):
    """Template (built on first use), reconstructed stage-1 and stage-2 of one stored step.

    Must be called inside ``runner.session()``: a template miss runs stage 1,
    and both forwards need the bf16 autocast the runner asserts.
    """
    template = templates.get(task)
    stage1 = reconstruct_stage1(template, group)
    stage2 = runner.run_stage2_llm(stage1)
    return stage1, stage2


def to_chunk(x: torch.Tensor) -> torch.Tensor:
    """``[1,H,D]`` or ``[H,D]`` model output -> ``[H,D]`` CPU fp32, safe to keep outside the session."""
    t = x.detach()
    if t.dim() == 3:
        if t.shape[0] != 1:
            raise ValueError(f"expected batch 1, got {tuple(t.shape)}")
        t = t[0]
    if t.dim() != 2:
        raise ValueError(f"expected [H, D], got {tuple(t.shape)}")
    out = t.to(device="cpu", dtype=torch.float32).contiguous()
    if out.is_inference():
        out = out.clone()
    return out


def h5_chunk(group, name: str) -> torch.Tensor:
    """A ``[H, D]`` float32 CPU tensor from a step dataset; any other rank raises."""
    arr = np.asarray(group[name], dtype=np.float32)
    if arr.ndim != 2:
        raise RuntimeError(f"{name}: expected [H, D], got {arr.shape}")
    return torch.from_numpy(np.ascontiguousarray(arr))


def maxabs_active(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, h_exec: int) -> float:
    """Unweighted max |a - b| over active dims and the executed window (diagnostic only)."""
    diff = (a[:h_exec] - b[:h_exec])[..., mask]
    if diff.numel() == 0:
        raise ValueError("no active dims to compare")
    return float(diff.abs().max())


def label_row(runner, templates, task: str, group, payload, warm_ts: tuple[float, ...],
              schedule: DenoiseSchedule, w: torch.Tensor, mask: torch.Tensor, h_exec: int) -> dict:
    """``y_full`` and one ``y_rem<k>`` per warm rung against the stored teacher chunk."""
    ref = h5_chunk(group, "clean_action")
    with runner.session():
        _, stage2 = stage2_in_session(runner, templates, task, group)
        warm = {
            t: runner.run_stage3_from(stage2, payload.intermediates[t], t, schedule=schedule).action_pred
            for t in warm_ts
        }
    row = {"y_full": weighted_chunk_deviation(to_chunk(payload.action_chunk), ref, w, mask, h_exec)}
    for t in warm_ts:
        row[f"y_rem{schedule.remaining_steps(t)}"] = weighted_chunk_deviation(
            to_chunk(warm[t]), ref, w, mask, h_exec
        )
    return row


def parity_row(runner, templates, task: str, group, schedule: DenoiseSchedule,
               w: torch.Tensor, mask: torch.Tensor, h_exec: int) -> dict:
    """Replay full / warm75 / warm50 from the step's OWN stored noise and snapshots.

    Diagnostic ``*_maxabs`` (unweighted, active dims, executed window) and the
    gate columns ``parity_D_*`` on the same W / mask / H_exec scale as the table.
    """
    clean = h5_chunk(group, "clean_action")
    noise0 = h5_chunk(group, "noise_action_0")[None]
    snaps = {name: h5_chunk(group, f"noise_action_{schedule.snapshot_index(t)}")
             for name, t in PARITY_WARM_TS.items()}
    with runner.session():
        _, stage2 = stage2_in_session(runner, templates, task, group)
        replay = {"full": runner.run_stage3(stage2, noise=noise0).action_pred}
        for name, t in PARITY_WARM_TS.items():
            replay[name] = runner.run_stage3_from(stage2, snaps[name], t, schedule=schedule).action_pred
    out = {}
    for name in PARITY_TIERS:
        chunk = to_chunk(replay[name])
        out[f"parity_{name}_maxabs"] = maxabs_active(chunk, clean, mask, h_exec)
        out[f"parity_D_{name}"] = weighted_chunk_deviation(chunk, clean, w, mask, h_exec)
    return out


# ------------------------------------------------------------------
# Parity gate
# ------------------------------------------------------------------


def identity_mismatches(expected: dict, actual: dict, keys: tuple[str, ...] = IDENTITY_KEYS) -> list[str]:
    """Human-readable list of identity keys that differ; a key missing on either side counts as a mismatch."""
    out = []
    for k in keys:
        if k not in expected or k not in actual or expected[k] in (None, "", "unknown", "missing") \
                or actual[k] in (None, "", "unknown", "missing") or expected[k] != actual[k] \
                or (k == "code_sha256" and not expected[k]):
            out.append(f"{k}: expected {expected.get(k)!r}, got {actual.get(k)!r}")
    return out


def parity_gate(parity_rows: list[dict], noise_floor_record: dict, expected_identity: dict, *,
                ratio: float = PARITY_RATIO, min_rows: int = PARITY_MIN_ROWS,
                min_rows_per_task: int = PARITY_MIN_ROWS_PER_TASK, num_tasks: int = NUM_TASKS) -> dict:
    """PASS iff every replay's p90(parity_D) <= ratio * median(D(ref1, ref2)) on a full, finite sample."""
    reasons: list[str] = []
    mism = identity_mismatches(expected_identity, noise_floor_record.get("identity", {}))
    if mism:
        reasons.append("noise floor identity mismatch: " + "; ".join(mism))
    floor_block = noise_floor_record.get("d_ref1_ref2") or {}
    floor_sample = noise_floor_record.get("sample", {})
    if floor_sample.get("per_task", 0) < 50 or floor_sample.get("n_rows", 0) < 500 \
            or floor_block.get("n") != floor_sample.get("n_rows"):
        reasons.append("noise floor needs at least 500 observations (50 per task) before formal parity admission")
    floor = floor_block.get("median")
    if floor is None or not math.isfinite(float(floor)) or float(floor) < 0.0:
        raise ValueError(f"noise floor median is not a finite non-negative number: {floor!r}")
    if int(noise_floor_record.get("n_active_dims", 0)) <= 0:
        raise ValueError("noise floor record reports no active action dims")
    floor = float(floor)
    n = len(parity_rows)
    if len({(r["trajectory_id"], r["decision_id"]) for r in parity_rows}) != n:
        reasons.append("parity sample repeats decision identities")
    per_task: dict[int, int] = {}
    for r in parity_rows:
        per_task[int(r["task_id"])] = per_task.get(int(r["task_id"]), 0) + 1
    if n < min_rows:
        reasons.append(f"parity sample has {n} rows < {min_rows}")
    missing_tasks = sorted(set(range(num_tasks)) - set(per_task))
    if missing_tasks:
        reasons.append(f"tasks without parity rows: {missing_tasks}")
    thin = {t: c for t, c in per_task.items() if c < min_rows_per_task}
    if thin:
        reasons.append(f"tasks below {min_rows_per_task} parity rows: {thin}")
    threshold = ratio * floor
    tiers: dict[str, dict] = {}
    for name in PARITY_TIERS:
        vals = np.array([float(r[f"parity_D_{name}"]) for r in parity_rows], dtype=np.float64) if n else np.array([])
        if vals.size and (not np.isfinite(vals).all() or (vals < 0).any()):
            reasons.append(f"non-finite or negative parity_D_{name}")
        stats = {"n": int(vals.size)}
        if vals.size and np.isfinite(vals).all():
            stats.update(p50=float(np.quantile(vals, 0.5)), p90=float(np.quantile(vals, 0.9)),
                         max=float(vals.max()), mean=float(vals.mean()))
            if floor == 0.0:
                if float(vals.max()) > 0.0:
                    reasons.append(f"zero noise floor but parity_D_{name} max {vals.max():.3g} > 0")
            elif stats["p90"] > threshold:
                reasons.append(f"parity_D_{name} p90 {stats['p90']:.4g} > {threshold:.4g} (= {ratio} x floor {floor:.4g})")
        tiers[name] = stats
    return {
        "status": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "ratio": ratio,
        "floor_median": floor,
        "threshold": threshold,
        "n_rows": n,
        "per_task_rows": {str(k): v for k, v in sorted(per_task.items())},
        "tiers": tiers,
        "identity": dict(expected_identity),
        "noise_floor_identity": dict(noise_floor_record.get("identity", {})),
        "sample": [{"trajectory_id": r["trajectory_id"], "decision_id": r["decision_id"]} for r in parity_rows],
    }


def require_pass_gate(gate_path: str | pathlib.Path, expected_identity: dict) -> dict:
    """Load a parity gate and refuse anything but an explicit PASS bound to this identity."""
    path = pathlib.Path(gate_path)
    if not path.is_file():
        raise SystemExit(f"parity gate missing: {path}")
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS":
        raise SystemExit(f"parity gate {path} is {gate.get('status')!r}, not PASS: {gate.get('reasons')}")
    mism = identity_mismatches(expected_identity, gate.get("identity", {}))
    if mism:
        raise SystemExit(f"parity gate {path} was computed for a different input: " + "; ".join(mism))
    return gate


# ------------------------------------------------------------------
# Identity record
# ------------------------------------------------------------------


def weights_sha256(w: torch.Tensor, mask: torch.Tensor) -> str:
    """Digest of the deviation weights and active mask (fp32 / bool bytes)."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(w.detach().cpu().numpy().astype(np.float32)).tobytes())
    h.update(np.ascontiguousarray(mask.detach().cpu().numpy().astype(np.bool_)).tobytes())
    return h.hexdigest()


def build_identity(*, suite: str, library: Library, library_path: str, template_path: str, ckpt: dict,
                   corpus: dict, corpus_dir: str, w: torch.Tensor, mask: torch.Tensor, h_exec: int,
                   schedule: DenoiseSchedule, warm_ts: tuple[float, ...], enabled: list[str],
                   weights: dict[str, float]) -> dict:
    """The identity record every product of this line carries and every downstream stage compares."""
    return {
        "suite": suite,
        "library_sha256": library.sha256,
        "library_path": str(library_path),
        "library_entries": len(library.entries),
        "library_trajectories": len(library.trajectory_ids),
        "template_sha256": sha256_file(template_path),
        "template_path": str(template_path),
        "checkpoint_identity_sha256": ckpt["sha256"],
        "checkpoint_path": ckpt["path"],
        "corpus_manifest_sha256": corpus["sha256"],
        "corpus_dir": str(corpus_dir),
        "corpus_files": corpus["n_files"],
        "weights_sha256": weights_sha256(w, mask),
        "n_active_dims": int(mask.sum()),
        "h_exec": int(h_exec),
        "schedule_id": schedule.schedule_id,
        "denoising_steps": schedule.num_steps,
        "warm_ts": [float(t) for t in warm_ts],
        "enabled_fields": list(enabled),
        "fusion_weights": dict(weights),
        "git_commit": git_commit(),
        "code_sha256": code_sha256(),
    }


def load_template(template_yaml: str, library_pkl: str, out_dir: pathlib.Path):
    """The template with its preload path pointed at the library actually loaded here."""
    import yaml

    raw = yaml.safe_load(pathlib.Path(template_yaml).read_text(encoding="utf-8"))
    raw["backend"]["in_memory"]["preload_path"] = str(library_pkl)
    used = out_dir / "config_used.yaml"
    used.parent.mkdir(parents=True, exist_ok=True)
    used.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return load_cache_config(str(used)), used


# ------------------------------------------------------------------
# Sampling helpers
# ------------------------------------------------------------------


def stratified_decisions(files: list[pathlib.Path], per_task: int, seed: int, *,
                         num_tasks: int = NUM_TASKS) -> list[dict]:
    """``per_task`` distinct (trajectory, decision) pairs per task, drawn uniformly with a per-task child seed."""
    pool: dict[int, list[tuple[str, int, str]]] = {t: [] for t in range(num_tasks)}
    for p in files:
        with h5py.File(p, "r") as f:
            a = episode_attrs(f, p.stem)
            for i in range(a["num_steps"]):
                pool[a["task_id"]].append((p.stem, i, a["task"]))
    out = []
    for t in range(num_tasks):
        cands = pool[t]
        if len(cands) < per_task:
            raise SystemExit(f"task {t}: only {len(cands)} decisions, cannot draw {per_task}")
        rng = np.random.default_rng([int(seed), int(t)])
        for j in sorted(int(i) for i in rng.choice(len(cands), size=per_task, replace=False)):
            stem, step, task = cands[j]
            out.append({"task_id": t, "trajectory_id": stem, "decision_id": step, "task": task})
    return out


class OrchestratorProbe:
    """The production CP1 stack (always_search / always_hit) on reconstructed stage-1 tensors."""

    def __init__(self, config, storage_config_path: str) -> None:
        from openpi.cache.config import build_per_connection_components, build_shared_storage
        from openpi.cache.orchestrator import CacheOrchestrator

        shared = build_shared_storage(config)
        comps = build_per_connection_components(config, shared, quiet=True)
        self.orchestrator = CacheOrchestrator(
            storage=comps["storage"], key_builder=comps["key_builder"], gates=comps["gates"],
            judges=comps["judges"], search_strategies=comps["search_strategies"], timer=comps["timer"],
            write_policy=comps["write_policy"], offline_writers=comps["offline_writers"],
            library_stats=comps["library_stats"],
        )
        self.source = storage_config_path

    def check(self, runner, templates, task: str, group, episode_id: str) -> tuple[str | None, float | None, str]:
        """``(entry_id, score, hit_type)`` the production stack returns for one reconstructed step."""
        with runner.session():
            template = templates.get(task)
            stage1 = reconstruct_stage1(template, group)
        self.orchestrator.on_task_begin(task)
        self.orchestrator.on_episode_start(task_key=task, episode_id=str(episode_id))
        try:
            res = self.orchestrator.check(CheckpointID.CP1, stage1=stage1)
        finally:
            self.orchestrator.clear()
            self.orchestrator.on_episode_end()
            self.orchestrator.on_task_end()
        return res.entry_id, (None if res.score is None else float(res.score)), res.hit_type.name


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def _parse_ts(text: str) -> tuple[float, ...]:
    vals = tuple(round(float(x), 4) for x in text.split(",") if x.strip())
    if not vals:
        raise SystemExit("--warm-ts is empty")
    return vals


def _parse_ids(text: str) -> set[int] | None:
    text = (text or "").strip()
    return {int(x) for x in text.split(",") if x.strip()} if text else None


def main() -> None:
    """CLI: parity sample + gate (``--parity-only``) or the full LOTO table bound to a PASS gate."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True, choices=SUITES)
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--library-pkl", required=True)
    ap.add_argument("--library-seed", type=int, default=0, help="the W13 manifest's sampling seed, recorded per row")
    ap.add_argument("--template-yaml", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--denoising-steps", type=int, default=DEFAULT_DENOISING_STEPS)
    ap.add_argument("--warm-ts", default=",".join(str(t) for t in DEFAULT_WARM_TS))
    ap.add_argument("--h-exec", type=int, default=DEFAULT_H_EXEC)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--task-ids", default="", help="comma list; empty = all tasks")
    ap.add_argument("--limit-episodes", type=int, default=0, help="smoke: first N episodes only")
    ap.add_argument("--allow-ungated-smoke", action="store_true",
                    help="permit a --limit-episodes run without a PASS parity gate (smoke only)")
    ap.add_argument("--parity-gate", default="", help="parity_gate.json that must be PASS for a full run")
    ap.add_argument("--parity-only", action="store_true", help="draw the parity sample, write parity_gate.json, exit")
    ap.add_argument("--parity-sample", type=int, default=DEFAULT_PARITY_SAMPLE)
    ap.add_argument("--parity-seed", type=int, default=20260914)
    ap.add_argument("--noise-floor-record", default="", help="noise_floor.json of this suite (parity gate scale)")
    ap.add_argument("--orchestrator-check", type=int, default=0,
                    help="every N-th out-of-library decision is re-checked through the production orchestrator")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-every", type=int, default=500)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    warm_ts = _parse_ts(args.warm_ts)
    schedule = groot_n15_schedule(args.denoising_steps)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg, cfg_used = load_template(args.template_yaml, args.library_pkl, out_dir)
    library = load_library(args.library_pkl, expected_schedule=schedule, warm_ts=warm_ts)
    w, mask = library_action_weights(args.library_pkl)
    w = torch.as_tensor(np.asarray(w), dtype=torch.float32)
    mask = torch.as_tensor(np.asarray(mask), dtype=torch.bool)
    enabled, weights = enabled_fields(cfg)
    files = corpus_files(args.corpus_dir)
    task_filter = _parse_ids(args.task_ids)
    corpus = corpus_manifest(files)
    ckpt = checkpoint_identity(args.checkpoint)
    identity = build_identity(
        suite=args.suite, library=library, library_path=args.library_pkl, template_path=args.template_yaml,
        ckpt=ckpt, corpus=corpus, corpus_dir=args.corpus_dir, w=w, mask=mask, h_exec=args.h_exec,
        schedule=schedule, warm_ts=warm_ts, enabled=enabled, weights=weights,
    )
    logger.info("library %s: %d entries / %d trajectories; enabled=%s weights=%s",
                pathlib.Path(args.library_pkl).name, len(library.entries), len(library.trajectory_ids),
                enabled, weights)

    model = ModelSide(args.checkpoint, denoising_steps=args.denoising_steps, device=args.device)
    if model.schedule != schedule:
        raise SystemExit(f"live schedule {model.schedule.schedule_id} != requested {schedule.schedule_id}")
    runner, templates = model.runner, model.templates

    # -------------------------------------------------------------- parity
    if args.parity_only:
        if not args.noise_floor_record:
            raise SystemExit("--parity-only needs --noise-floor-record")
        nf = json.loads(pathlib.Path(args.noise_floor_record).read_text(encoding="utf-8"))
        sample = stratified_decisions(files, args.parity_sample // NUM_TASKS, args.parity_seed)
        by_stem: dict[str, list[dict]] = {}
        for d in sample:
            by_stem.setdefault(d["trajectory_id"], []).append(d)
        rows: list[dict] = []
        t0 = time.time()
        for stem, picks in by_stem.items():
            with h5py.File(pathlib.Path(args.corpus_dir) / f"{stem}.h5", "r") as f:
                if episode_schedule(f) != schedule:
                    raise SystemExit(f"{stem}: file schedule differs from {schedule.schedule_id}")
                for d in picks:
                    g = f[f"step_{d['decision_id']:04d}"]
                    rows.append({**d, **parity_row(runner, templates, d["task"], g, schedule, w, mask, args.h_exec)})
        logger.info("parity sample: %d rows in %.0fs", len(rows), time.time() - t0)
        write_jsonl(out_dir / "parity_sample.jsonl", rows)
        gate = parity_gate(rows, nf, identity)
        gate["parity_sample_path"] = str(out_dir / "parity_sample.jsonl")
        gate["noise_floor_record"] = str(args.noise_floor_record)
        (out_dir / "parity_gate.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        logger.info("parity gate: %s %s", gate["status"], gate["reasons"])
        print(f"PARITY_GATE={gate['status']}")
        return

    # -------------------------------------------------------------- full table
    smoke = args.limit_episodes > 0
    gate_sha = None
    if args.parity_gate:
        require_pass_gate(args.parity_gate, identity)
        gate_sha = sha256_file(args.parity_gate)
    elif not (smoke and args.allow_ungated_smoke):
        raise SystemExit("a full run needs --parity-gate <PASS gate>; smoke runs may pass --allow-ungated-smoke")

    storage = build_storage(cfg, library.entries)
    strategy = build_retrieval(cfg, storage, weights, n_hint=max(library.per_task_entries.values()))
    builder = make_query_builder(cfg.key_builder.type)
    probe = OrchestratorProbe(cfg, str(cfg_used)) if args.orchestrator_check > 0 else None

    selected = [p for p in files if task_filter is None or _task_of(p) in task_filter]
    if smoke:
        selected = selected[: args.limit_episodes]
    out_path = out_dir / ("loto_table.smoke.jsonl" if smoke else "loto_table.jsonl")
    stats = {"episodes": 0, "rows": 0, "rows_in_library": 0, "self_skipped_total": 0,
             "orchestrator_checks": 0, "orchestrator_mismatch": 0, "prompt_perm_checks": 0,
             "prompt_perm_mismatch": 0, "no_hit": 0}
    seen_keys: set[tuple[str, int]] = set()
    t0 = time.time()
    library_id = library.sha256[:12]
    with open(out_path, "w", encoding="utf-8") as out:
        for p in selected:
            with h5py.File(p, "r") as f:
                a = episode_attrs(f, p.stem)
                if task_filter is not None and a["task_id"] not in task_filter:
                    continue
                if episode_schedule(f) != schedule:
                    raise SystemExit(f"{p.stem}: file schedule differs from {schedule.schedule_id}")
                in_library = p.stem in library.trajectory_ids
                stats["episodes"] += 1
                for step_idx, name in step_groups(f):
                    g = f[name]
                    key = (p.stem, step_idx)
                    if key in seen_keys:
                        raise SystemExit(f"duplicate row key {key}")
                    seen_keys.add(key)
                    keys = query_keys_for(builder, g, enabled)
                    hits = search(strategy, keys, step_idx, a["task"])
                    if not hits:
                        stats["no_hit"] += 1
                        continue
                    cand, s, skipped = loto_winner(hits, p.stem, in_library, library.traj_of)
                    payload = library.by_id[cand].payload
                    row = {
                        "suite": args.suite, "trajectory_id": p.stem, "episode_id": a["episode_id"],
                        "decision_id": step_idx, "task": a["task"], "task_id": a["task_id"],
                        "orig_init_state_idx": a["orig_init_state_idx"], "library_id": library_id,
                        "library_seed": int(args.library_seed), "s": s, "candidate_id": cand,
                        **label_row(runner, templates, a["task"], g, payload, warm_ts, schedule, w, mask, args.h_exec),
                        "in_library": in_library, "episode_success": a["success"], "n_self_skipped": skipped,
                    }
                    if probe is not None and not in_library and stats["rows"] % args.orchestrator_check == 0:
                        stats["orchestrator_checks"] += 1
                        pid, pscore, phit = probe.check(runner, templates, a["task"], g, f"{p.stem}:{step_idx}")
                        ok = pid == cand and pscore is not None and abs(pscore - s) <= 3e-5
                        row["orchestrator_check"] = {"entry_id": pid, "score": pscore, "hit_type": phit, "ok": ok}
                        if not ok:
                            stats["orchestrator_mismatch"] += 1
                            logger.error("orchestrator mismatch at %s:%d offline=(%s,%.6f) online=(%s,%s)",
                                         p.stem, step_idx, cand, s, pid, pscore)
                        perm = search(strategy, with_permuted_prompt(keys, stats["rows"]), step_idx, a["task"])
                        pc, ps, _ = loto_winner(perm, p.stem, in_library, library.traj_of)
                        stats["prompt_perm_checks"] += 1
                        if pc != cand or abs(ps - s) > 1e-6:
                            stats["prompt_perm_mismatch"] += 1
                            logger.error("permuted prompt changed the verdict at %s:%d", p.stem, step_idx)
                        row["prompt_perm_ok"] = pc == cand and abs(ps - s) <= 1e-6
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stats["rows"] += 1
                    stats["rows_in_library"] += int(in_library)
                    stats["self_skipped_total"] += skipped
                    if stats["rows"] % args.log_every == 0:
                        el = time.time() - t0
                        logger.info("%d rows (%d episodes) in %.0fs, %.1f rows/s", stats["rows"], stats["episodes"], el, stats["rows"] / max(el, 1e-9))
            out.flush()
    if stats["orchestrator_mismatch"] or stats["prompt_perm_mismatch"]:
        raise SystemExit(f"self-check failed: {stats}")
    record = {
        "protocol": "rit_loto_table_v1", "smoke": smoke, "identity": identity, "stats": stats,
        "library_seed": int(args.library_seed), "config_used": str(cfg_used),
        "parity_gate": args.parity_gate or None, "parity_gate_sha256": gate_sha,
        "out_jsonl": str(out_path), "out_sha256": sha256_file(out_path),
        "elapsed_s": time.time() - t0, "python": sys.version.split()[0], "torch": torch.__version__,
    }
    np.savez(str(out_path) + ".weights.npz", w=w.numpy(), active_mask=mask.numpy())
    (pathlib.Path(str(out_path) + ".record.json")).write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("wrote %s: %s", out_path, stats)


def _task_of(path: pathlib.Path) -> int:
    """Task id from the collector's file name (``episode_<gid>`` with gid = task*50 + init)."""
    gid = int(path.stem.split("_")[1])
    return gid // TRIALS_PER_TASK


if __name__ == "__main__":
    main()
