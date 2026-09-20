"""Emit the served cache yamls of the step-vs-warm-start line from the RIT lines' frozen retrieval.

One source template per environment (plan §3.0), never a name transplanted across benchmarks:

* ``pi05_rc`` / ``groot_rc``: the deployed RoboCasa RIT ``always_hit`` cell of the teacher
  (``config/base/<teacher>_rit_always_hit.yaml``: W13 ``full`` library, frozen weight cell
  ``emit_rit_rc.FROZEN_WEIGHT_CID``, per-field z-score normalisers, ``text_ivf`` search);
* ``pi05_libero_<suite>``: the gate line's N4 server template of the suite
  (``exp.gate_threshold_pareto.libraries.TEMPLATE``: d1 retrieval stack + the ``ws``
  ``cp1_spatial_pool_16`` library the RIT-Pareto arms ran on);
* ``groot_libero_<suite>``: ``exp/libero_groot/config/rit/<suite>/template.yaml`` (W13 S3 library,
  ``groot_n15_k8_v1``).

Only the verdict layer changes:

* ``shadow`` (read-only retrieval, every environment): ``gate: always_search``; judge ``threshold``
  with an unreachable threshold for pi0.5 (every decision is a teacher MISS with the winner
  recorded by ``Pi05DiagInterceptor``; the template's warm tiers are dropped), ``always_hit`` for
  GR00T (``GrootDiagPolicy`` reads the verdict and never applies it, like ``GrootRitShadow``);
  ``write_policy: never``;
* ``warm_t<t>`` (RoboCasa only, Q-B): ``gate: always_search``; judge ``always_warm_start`` at
  ``start_t = t`` (top-1, no gate); ``write_policy: never``.

``index.json`` binds every emitted yaml (sha256), its source template sha, the library path named
in it, the schedule and the arm's resume steps, plus the plain arms and the Q-B design. The library
files are checked on the serving host at Step 0 (``--check-libraries``: chunk shape, executed
dims, the snapshots every warm ``t`` needs, the loop length), not assumed from their names.

usage: python -m exp.step_diag.emit_arms --out exp/step_diag/config/arms [--check-libraries]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
from typing import Dict, Optional

import yaml

from openpi.cache.types import PI05_V1, groot_n15_schedule

from exp.step_diag import envs as _envs

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE_DIR = HERE / "config" / "base"
UNREACHABLE_THRESHOLD = 99.0


def source_template(env_id: str) -> pathlib.Path:
    """The frozen RIT template this environment's cells derive from."""
    env = _envs.resolve_env(env_id)
    if env.benchmark == "robocasa365":
        return BASE_DIR / f"{_teacher_of(env)}_rit_always_hit.yaml"
    if env.policy == "pi05":
        from exp.gate_threshold_pareto.libraries import TEMPLATE

        return REPO / TEMPLATE[env.benchmark]
    return REPO / "exp" / "libero_groot" / "config" / "rit" / env.benchmark / "template.yaml"


def _teacher_of(env: _envs.EnvSpec) -> str:
    return "pi05" if env.policy == "pi05" else "groot_tp"


def load_base(env_id: str) -> tuple[dict, str]:
    path = source_template(env_id)
    text = path.read_text()
    cfg = yaml.safe_load(text)
    env = _envs.resolve_env(env_id)
    if env.benchmark == "robocasa365" and cfg["checkpoints"]["cp1"]["judge"]["type"] != "always_hit":
        raise ValueError(f"{path}: the RoboCasa base must be the RIT always_hit cell")
    if env.policy == "groot" and cfg.get("denoise_schedule") not in (None, env.schedule_id):
        raise ValueError(f"{path}: denoise_schedule {cfg.get('denoise_schedule')!r} != {env.schedule_id}")
    return cfg, hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strip_verdict(cfg: dict) -> dict:
    out = copy.deepcopy(cfg)
    cp1 = out["checkpoints"]["cp1"]
    cp1["gate"] = {"type": "always_search"}
    out["write_policy"] = {"type": "never"}
    return out


def build_shadow_cell(env_id: str, cfg: dict) -> dict:
    env = _envs.resolve_env(env_id)
    out = _strip_verdict(cfg)
    if env.policy == "pi05":
        out["checkpoints"]["cp1"]["judge"] = {"type": "threshold", "threshold": UNREACHABLE_THRESHOLD}
    else:
        out["checkpoints"]["cp1"]["judge"] = {"type": "always_hit"}
        out["denoise_schedule"] = env.schedule_id
    return out


def build_warm_cell(env_id: str, cfg: dict, start_t: float) -> dict:
    env = _envs.resolve_env(env_id)
    if env.benchmark != "robocasa365":
        raise ValueError(f"{env_id}: warm-start arms are RoboCasa-only in this line (Q-B)")
    out = _strip_verdict(cfg)
    schedule = PI05_V1 if env.policy == "pi05" else groot_n15_schedule(env.k_full)
    t = round(float(start_t), 4)
    if t not in schedule.timestep_set:
        raise ValueError(f"{env_id}: start_t {t} is not a snapshot timestep of {schedule.schedule_id}")
    out["checkpoints"]["cp1"]["judge"] = {"type": "always_warm_start", "start_t": t}
    if schedule is not PI05_V1:
        out["denoise_schedule"] = schedule.schedule_id
    return out


def verify_cell(env_id: str, cfg: dict, base: dict) -> None:
    """The retrieval identity must be the template's; only gate / judge / write policy may differ."""
    env = _envs.resolve_env(env_id)
    comparable = [copy.deepcopy(c) for c in (cfg, base)]
    for c in comparable:
        c["checkpoints"]["cp1"]["gate"] = None
        c["checkpoints"]["cp1"]["judge"] = None
        c["write_policy"] = None
        c.pop("denoise_schedule", None)
        c["backend"]["in_memory"].pop("preload_path")
    if comparable[0] != comparable[1]:
        raise ValueError(f"{env_id}: settings outside the permitted verdict/path changes differ from the source template")
    for key in ("keys", "key_builder"):
        if cfg[key] != base[key]:
            raise ValueError(f"{env_id}: {key} differs from the source template")
    # A host may remap the artifact path. The emitted YAML and server manifest
    # bind the effective file's bytes; all other backend settings stay frozen.
    backends = [copy.deepcopy(c["backend"]) for c in (cfg, base)]
    for backend in backends:
        backend["in_memory"].pop("preload_path")
    if backends[0] != backends[1]:
        raise ValueError(f"{env_id}: backend differs from the source template")
    if cfg["checkpoints"]["cp1"]["search_strategy"] != base["checkpoints"]["cp1"]["search_strategy"]:
        raise ValueError(f"{env_id}: search_strategy differs from the source template")
    if cfg["checkpoints"]["cp1"]["gate"] != {"type": "always_search"}:
        raise ValueError(f"{env_id}: gate must be always_search")
    if cfg["write_policy"] != {"type": "never"}:
        raise ValueError(f"{env_id}: write_policy must be never")
    judge = cfg["checkpoints"]["cp1"]["judge"]
    if env.policy == "pi05" and judge["type"] == "threshold" and judge.get("warm_tiers"):
        raise ValueError(f"{env_id}: a shadow cell must not carry warm tiers")
    if env.policy == "groot" and cfg.get("denoise_schedule") != env.schedule_id:
        raise ValueError(f"{env_id}: denoise_schedule must be stamped {env.schedule_id}")


def verify_library(pkl_path: str | pathlib.Path, env_id: str) -> dict:
    """Load the library and check what the arms need of it (Step 0, on the serving host)."""
    import pickle

    import torch

    env = _envs.resolve_env(env_id)
    with open(pkl_path, "rb") as f:
        lib = pickle.load(f)
    entries = lib["entries"]
    if not entries:
        raise ValueError(f"{pkl_path}: empty library")
    shape = tuple(torch.as_tensor(entries[0].payload.action_chunk).shape)
    if shape != (env.action_horizon, env.action_dim):
        raise ValueError(f"{pkl_path}: action_chunk {shape} != env ({env.action_horizon}, {env.action_dim})")
    need = {round(float(t), 4) for t in env.warm_ts}
    missing = 0
    for e in entries:
        chunk = torch.as_tensor(e.payload.action_chunk)
        if tuple(chunk.shape) != shape or not torch.isfinite(chunk).all():
            raise ValueError(f"{pkl_path}: invalid action_chunk shape or nonfinite values")
        inter = getattr(e.payload, "intermediates", None) or {}
        have = {round(float(t), 4) for t in inter}
        if not need <= have:
            missing += 1
        for t, value in inter.items():
            if round(float(t), 4) in need:
                tensor = torch.as_tensor(value)
                if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
                    raise ValueError(f"{pkl_path}: invalid snapshot at t={t}")
    if missing:
        raise ValueError(f"{pkl_path}: {missing}/{len(entries)} entries lack a snapshot for warm_ts {sorted(need)}")
    steps = {getattr(e.payload, "denoising_num_steps", None) for e in entries}
    if steps != {env.k_full}:
        raise ValueError(f"{pkl_path}: denoising_num_steps {steps} != {env.k_full}")
    # The schedule is stamped per payload by newer builders and at library level by the in-memory
    # backend's ``artifact_meta`` (``schedule_id`` / ``denoise_schedule``); pi0.5 libraries without
    # either are ``pi05_v1`` by construction (same default the backend applies).
    lib_schedule = lib.get("schedule_id") or lib.get("denoise_schedule") or (PI05_V1.schedule_id if env.policy == "pi05" else None)
    schedules = {getattr(e.payload, "schedule_id", None) or lib_schedule for e in entries}
    if schedules != {env.schedule_id}:
        raise ValueError(f"{pkl_path}: schedule {schedules} differs from {env.schedule_id}")
    # Library action_active_mask describes variance, not the adapter's executed dimensions.
    n_exec = env.n_executed
    return {"env_id": env_id, "path": str(pkl_path), "n_entries": len(entries), "action_shape": list(shape),
            "snapshot_ts": sorted(need), "n_executed": n_exec, "sha256": _envs.library_digest(pkl_path)}


def arm_id_of(kind: str, t: float | None = None, k: int | None = None) -> str:
    if kind == "warm":
        return f"warm_t{t:g}"
    if kind == "plain":
        return f"plain_k{k}"
    return kind


def emit(out_root: pathlib.Path, env_ids=tuple(_envs.ENVS), *, check_libraries: bool = False,
         library_paths: Optional[Dict[str, str]] = None) -> Dict[str, dict]:
    """Write ``<out>/<env_id>/{shadow,warm_t*}.yaml`` + ``index.json``; returns the index."""
    index: Dict[str, dict] = {}
    for env_id in env_ids:
        env = _envs.resolve_env(env_id)
        base, base_sha = load_base(env_id)
        override = (library_paths or {}).get(env_id)
        if override:
            base["backend"]["in_memory"]["preload_path"] = override
        edir = out_root / env_id
        edir.mkdir(parents=True, exist_ok=True)
        cells = {"shadow": build_shadow_cell(env_id, base)}
        if env.benchmark == "robocasa365":
            for t in _envs.QB_WARM_TS[env.policy]:
                cells[arm_id_of("warm", t=t)] = build_warm_cell(env_id, base, t)
        entries = {}
        for arm_id, cfg in cells.items():
            verify_cell(env_id, cfg, base)
            text = yaml.safe_dump(cfg, sort_keys=False)
            path = edir / f"{arm_id}.yaml"
            path.write_text(text)
            judge = cfg["checkpoints"]["cp1"]["judge"]
            entries[arm_id] = {
                "file": str(path.relative_to(out_root)),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "judge": judge,
                "start_t": judge.get("start_t"),
                "remaining_steps": None if judge.get("start_t") is None else env.remaining_steps(judge["start_t"]),
                "library": cfg["backend"]["in_memory"]["preload_path"],
                "schedule_id": env.schedule_id,
            }
        record: dict = {
            "env_id": env_id, "policy": env.policy, "benchmark": env.benchmark,
            "source_template": str(source_template(env_id).relative_to(REPO)), "source_sha256": base_sha,
            "cells": entries, "k_full": env.k_full, "k_set": list(env.k_set), "warm_ts": list(env.warm_ts),
            "server_host": env.server_host, "worker_host": env.worker_host,
        }
        if env.benchmark == "robocasa365":
            plain = {arm_id_of("plain", k=k): {"exec_steps": k} for k in _envs.QB_PLAIN_KS[env.policy]}
            plain["full"] = {"exec_steps": env.k_full}
            record["weight_cid"] = _rit_weight_cid(_teacher_of(env))
            record["plain_arms"] = plain
            record["qb"] = {"main_m": _envs.QB_MAIN_M[env.policy], "main_t": _envs.QB_MAIN_T[env.policy],
                            "cliff": list(_envs.QB_CLIFF[env.policy]), "flat": list(_envs.QB_FLAT[env.policy]),
                            "episodes": _envs.QB_EPISODES, "flat_episodes": _envs.QB_FLAT_EPISODES,
                            "base_seed": _envs.RC_FORMAL_BASE_SEED}
        if check_libraries:
            lib_path = (library_paths or {}).get(env_id, entries["shadow"]["library"])
            record["library_check"] = verify_library(lib_path, env_id)
        index[env_id] = record
    (out_root / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
    return index


def _rit_weight_cid(teacher: str) -> str:
    try:
        from exp.robocasa365.emit_rit_rc import FROZEN_WEIGHT_CID

        return FROZEN_WEIGHT_CID[teacher]
    except Exception:  # noqa: BLE001 - informational only
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "config" / "arms"))
    ap.add_argument("--env-ids", default=",".join(_envs.ENVS))
    ap.add_argument("--check-libraries", action="store_true", help="load every library and verify the payload contract")
    ap.add_argument("--library", action="append", default=[], help="env_id=path override for --check-libraries")
    a = ap.parse_args()
    overrides = dict(item.split("=", 1) for item in a.library)
    index = emit(pathlib.Path(a.out), tuple(x for x in a.env_ids.split(",") if x), check_libraries=a.check_libraries,
                 library_paths=overrides)
    for env_id, entry in index.items():
        print(f"{env_id}: {sorted(entry['cells'])} + plain {sorted(entry.get('plain_arms', {}))} "
              f"{'library ok' if 'library_check' in entry else ''} -> {a.out}/{env_id}")


if __name__ == "__main__":
    main()
