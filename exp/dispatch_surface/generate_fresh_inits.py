"""Fresh initial-state generator for the confirmation pools P and C
(confirmation plan 3.7, G1R1-B7 / G1R2-B3, G2R1-B5).

Per (suite, task, pool, k) the generator runs a frozen state machine:
attempt ``a = 0`` uses the base authority, ``a = 1..MAX_RETRIES`` are retries;
each attempt derives a 256-bit authority ``sha256("dsp_rev2_fresh|suite|
task|pool|k|attempt|a")``, maps it through ``SeedSequence`` to ONE uint32
seed that is handed to ``random``, ``numpy`` and ``env.seed`` alike, builds a
fresh ``OffScreenRenderEnv``, calls ``env.reset()`` and reads the simulator
state. A state is accepted when reset raised nothing, the shape matches the
TASK's official ``.init`` width (LIBERO-10 widths differ per task: 45..123)
and every entry is finite. Failed or
colliding indices occupy ``k`` (never re-sampled); P needs 10/10 ``ok`` and
C 60/60 ``ok`` or the pool is ``generator_validation_failed``.

Nothing a manifest says about itself is authority. ``validate_pools`` is the
fresh-pool finalizer: it re-derives the state dimension from the official
``.init`` files, checks the task manifest (10 unique ids / names / BDDL
digests), every ``k``-ordered entry's attempts and seed derivation, the
materialised ``.init`` bytes state by state, the three-way exclusivity
(official 50 / P / C), the asset rollup and the cross-machine records
(re-running ``compare_manifests`` on the local and peer manifests, whose
hosts must differ). The validation artifact it writes is the only thing the
seal accepts, and the seal re-runs it.

The LIBERO environment is only imported inside ``make_env``; unit tests
inject a stub factory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import re
import socket

import numpy as np

PROTOCOL = "dispatch_surface_rev2_fresh_inits"
CROSS_MACHINE_PROTOCOL = "dispatch_surface_rev2_fresh_cross_machine"
VALIDATION_PROTOCOL = "dispatch_surface_rev2_fresh_pool_validation"
SEED_NAMESPACE = "dsp_rev2_fresh"
MAX_RETRIES = 4          # attempts a = 0..MAX_RETRIES (5 in total)
POOL_QUOTA = {"P": 10, "C": 60}
STATUS_OK, STATUS_FAILED, STATUS_COLLISION = "ok", "failed", "collision"
NUM_TASKS = 10
OFFICIAL_QUOTA = 50
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
VALIDATION_PATH_KEYS = ("p_manifest_path", "c_manifest_path", "p_dir", "c_dir", "apool_dir", "task_manifest_path",
                        "bddl_root", "cross_p_record", "cross_p_peer", "cross_c_record", "cross_c_peer", "assets_dir")


def state_sha256(state) -> str:
    arr = np.ascontiguousarray(np.asarray(state, dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def seed_authority(suite: str, task_name: str, pool: str, k: int, attempt: int) -> bytes:
    return hashlib.sha256(f"{SEED_NAMESPACE}|{suite}|{task_name}|{pool}|{k}|attempt|{attempt}".encode("utf-8")).digest()


def seed32_from_authority(authority: bytes) -> int:
    ss = np.random.SeedSequence(int.from_bytes(authority, "big"))
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def derive_seeds(suite: str, task_name: str, pool: str, k: int, attempt: int) -> dict:
    auth = seed_authority(suite, task_name, pool, k, attempt)
    s32 = seed32_from_authority(auth)
    if not (0 <= s32 <= 2**32 - 1):
        raise SystemExit("derived seed outside the uint32 domain")
    return {"a": attempt, "authority_sha256": auth.hex(), "seed32": s32,
            "py_seed": s32, "np_seed": s32, "env_seed": s32}


def make_env(bddl_file: str):
    """Real LIBERO env (same constructor arguments as examples/libero/main.py)."""
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=256, camera_widths=256)


def read_state(env) -> np.ndarray:
    if hasattr(env, "get_sim_state"):
        return np.asarray(env.get_sim_state(), dtype=np.float64).reshape(-1)
    return np.asarray(env.sim.get_state().flatten(), dtype=np.float64)


def sample_one(suite: str, task_name: str, pool: str, k: int, *, bddl_file: str, state_dim: int,
               env_factory=make_env) -> dict:
    """One index of the state machine: returns the manifest entry for k."""
    attempts = []
    for a in range(MAX_RETRIES + 1):
        seeds = derive_seeds(suite, task_name, pool, k, a)
        outcome = None
        state = None
        env = None
        try:
            random.seed(seeds["py_seed"])
            np.random.seed(seeds["np_seed"])
            env = env_factory(bddl_file)
            if hasattr(env, "seed"):
                env.seed(seeds["env_seed"])
            env.reset()
            state = read_state(env)
            if state.shape != (state_dim,):
                outcome = f"bad_shape:{state.shape}"
            elif not np.isfinite(state).all():
                outcome = "non_finite"
            else:
                outcome = "accepted"
        except Exception as exc:  # any reset failure is a retry, never a crash
            outcome = f"exception:{type(exc).__name__}"
        finally:
            if env is not None:
                close = getattr(env, "close", None)
                if close:
                    try:
                        close()
                    except Exception:
                        pass
        attempts.append({**seeds, "outcome": outcome})
        if outcome == "accepted":
            return {"k": k, "attempts": attempts, "status": STATUS_OK, "shape": list(state.shape),
                    "dtype": "float64", "state_sha256": state_sha256(state), "_state": state}
    return {"k": k, "attempts": attempts, "status": STATUS_FAILED, "shape": None, "dtype": None,
            "state_sha256": None, "_state": None}


def load_init_states(path):
    from exp.ablation_study.cache_size.verify_apool import load_init_states as _load

    return _load(pathlib.Path(path))


def official_state_digests(apool_dir: pathlib.Path, task_name: str) -> set[str]:
    states = np.asarray(load_init_states(pathlib.Path(apool_dir) / f"{task_name}.init"))
    if states.ndim != 2 or states.shape[0] != OFFICIAL_QUOTA or not np.isfinite(states).all():
        raise SystemExit(f"{task_name}: official init file must hold exactly {OFFICIAL_QUOTA} finite 2-D states")
    digests = {state_sha256(st) for st in states}
    if len(digests) != OFFICIAL_QUOTA:
        raise SystemExit(f"{task_name}: official init file contains duplicate states")
    return digests


def official_state_dim(apool_dir: pathlib.Path, task_name: str) -> int:
    """The suite's state width, derived from the official per-task ``.init`` file."""
    states = np.asarray(load_init_states(pathlib.Path(apool_dir) / f"{task_name}.init"))
    if states.ndim != 2 or states.shape[0] != OFFICIAL_QUOTA or not np.isfinite(states).all():
        raise SystemExit(f"{task_name}: official init file must hold exactly {OFFICIAL_QUOTA} finite 2-D states")
    return int(states.shape[1])


def load_task_manifest(path) -> dict:
    """The suite's task manifest: exactly 10 tasks with unique ids 0..9, names and BDDL digests."""
    tm = json.loads(pathlib.Path(path).read_text())
    if tm.get("schema") != 1 or not isinstance(tm.get("suite"), str) or not tm["suite"]:
        raise SystemExit("task manifest requires schema=1 and a non-empty suite")
    tasks = tm.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != NUM_TASKS:
        raise SystemExit(f"task manifest must list exactly {NUM_TASKS} tasks")
    ids = [t.get("task_id") for t in tasks]
    names = [t.get("task_name") for t in tasks]
    if sorted(ids) != list(range(NUM_TASKS)) or any(isinstance(i, bool) or not isinstance(i, int) for i in ids):
        raise SystemExit("task manifest ids are not exactly 0..9")
    if len(set(names)) != NUM_TASKS or any(not isinstance(n, str) or not n for n in names):
        raise SystemExit("task manifest names are not unique non-empty strings")
    for t in tasks:
        if not isinstance(t.get("bddl_file"), str) or not t["bddl_file"]:
            raise SystemExit(f"task {t.get('task_name')}: bddl_file missing")
        if not isinstance(t.get("bddl_sha256"), str) or not _SHA_RE.match(t["bddl_sha256"]):
            raise SystemExit(f"task {t.get('task_name')}: bddl_sha256 missing or malformed")
    return tm


def assets_rollup(assets_dir) -> dict:
    """Content rollup of every file under the simulator asset directory.

    Hidden entries (any path component starting with ``.``) are excluded: the
    HuggingFace download bookkeeping under ``assets/.cache/huggingface`` holds
    per-download ``*.metadata`` files with timestamps that differ between
    machines while the asset content itself is identical."""
    if not assets_dir:
        raise SystemExit("assets directory is required (the rollup is part of the generation authority)")
    root = pathlib.Path(assets_dir)
    if not root.is_dir():
        raise SystemExit(f"assets directory does not exist: {root}")
    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts))
    if not files:
        raise SystemExit(f"assets directory is empty: {root}")
    lines = [f"{p.relative_to(root).as_posix()} {_file_sha256(p)}" for p in files]
    return {"root_name": root.name, "file_count": len(files),
            "sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest()}


def task_state_dims(apool_dir: pathlib.Path, task_names) -> dict[str, int]:
    """Per-task official state width (the ONLY source of the width)."""
    return {name: official_state_dim(apool_dir, name) for name in task_names}


def _width_for(state_dim, name: str) -> int:
    return int(state_dim[name]) if isinstance(state_dim, dict) else int(state_dim)


def generate_pool(suite: str, pool: str, tasks: list[dict], *, apool_dir: pathlib.Path, state_dim,
                  env_factory=make_env, exclude: dict[str, set[str]] | None = None) -> dict:
    """tasks: [{task_id, task_name, bddl_file, bddl_sha256?}]; ``state_dim`` is the per-task
    width map from ``task_state_dims`` (an int applies to every task); returns pools[pool] block + states."""
    quota = POOL_QUOTA[pool]
    block = {"quota": quota, "tasks": {}}
    states_by_task: dict[str, list[np.ndarray]] = {}
    for t in tasks:
        name = t["task_name"]
        width = _width_for(state_dim, name)
        banned = set(official_state_digests(apool_dir, name)) | set((exclude or {}).get(name, set()))
        entries = []
        seen: set[str] = set()
        states = []
        for k in range(quota):
            e = sample_one(suite, name, pool, k, bddl_file=t["bddl_file"], state_dim=width, env_factory=env_factory)
            st = e.pop("_state")
            if e["status"] == STATUS_OK and (e["state_sha256"] in banned or e["state_sha256"] in seen):
                e["status"] = STATUS_COLLISION
                st = None
            if e["status"] == STATUS_OK:
                seen.add(e["state_sha256"])
                states.append(st)
            entries.append(e)
        bddl_path = pathlib.Path(t["bddl_file"])
        bddl_sha = t.get("bddl_sha256") or (_file_sha256(bddl_path) if bddl_path.is_file() else None)
        block["tasks"][name] = {"task_id": int(t["task_id"]), "bddl_file": t["bddl_file"],
                               "bddl_sha256": bddl_sha, "entries": entries}
        states_by_task[name] = states
    return block, states_by_task


def assert_pool_complete(block: dict, pool: str) -> None:
    quota = POOL_QUOTA[pool]
    for name, info in block["tasks"].items():
        bad = [e for e in info["entries"] if e["status"] != STATUS_OK]
        if len(info["entries"]) != quota or bad:
            raise SystemExit(f"generator_validation_failed: pool {pool} task {name}: "
                             f"{len(info['entries']) - len(bad)}/{quota} ok ({[(e['k'], e['status']) for e in bad][:5]})")


def materialize(block: dict, states_by_task: dict[str, list], out_dir: pathlib.Path) -> dict[str, str]:
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name, states in states_by_task.items():
        arr = np.stack(states).astype(np.float64)
        dest = out_dir / f"{name}.init"
        torch.save(arr, dest)
        digests[name] = _file_sha256(dest)
    return digests


def _file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def load_pool_manifest(path) -> dict:
    m = json.loads(pathlib.Path(path).read_text())
    if m.get("protocol") != PROTOCOL or m.get("schema") != 1 or "pools" not in m:
        raise SystemExit(f"{path}: not a fresh-init pool manifest")
    return m


def environment_record(assets_dir=None) -> dict:
    rec = {"host": socket.gethostname(), "numpy": np.__version__,
           "assets_rollup": assets_rollup(assets_dir) if assets_dir else None}
    for mod in ("libero", "robosuite", "mujoco", "torch"):
        try:
            m = __import__(mod)
            rec[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            rec[mod] = None
    return rec


def compare_manifests(a: dict, b: dict, pool: str) -> list[str]:
    """Cross-machine determinism check under the same generation authority."""
    problems = []
    for key in ("suite", "seed_namespace", "max_retries", "state_dim", "task_manifest_sha256"):
        if a.get(key) != b.get(key):
            problems.append(f"authority field {key} differs")
    if (a.get("environment") or {}).get("assets_rollup") != (b.get("environment") or {}).get("assets_rollup"):
        problems.append("asset rollups differ")
    pa, pb = (a.get("pools") or {}).get(pool), (b.get("pools") or {}).get(pool)
    if not isinstance(pa, dict) or not isinstance(pb, dict):
        return problems + [f"one manifest lacks pool {pool}"]
    if pa.get("quota") != pb.get("quota"):
        problems.append("pool quotas differ")
    ta, tb = pa.get("tasks") or {}, pb.get("tasks") or {}
    if set(ta) != set(tb):
        return problems + ["task sets differ"]
    for name in sorted(ta):
        for key in ("task_id", "bddl_sha256"):
            if ta[name].get(key) != tb[name].get(key):
                problems.append(f"{name}: {key} differs")
        ea, eb = ta[name]["entries"], tb[name]["entries"]
        if len(ea) != len(eb):
            problems.append(f"{name}: entry counts differ")
            continue
        for x, y in zip(ea, eb):
            for key in ("k", "attempts", "status", "shape", "dtype", "state_sha256"):
                if x.get(key) != y.get(key):
                    problems.append(f"{name} k={x.get('k')}: {key} differs")
    return problems


# ------------------------------------------------------------------
# cross-machine record and the fresh-pool validation artifact (G2R1-B5)
# ------------------------------------------------------------------

def _side(manifest_path) -> dict:
    m = load_pool_manifest(manifest_path)
    env = m.get("environment") or {}
    host = env.get("host")
    if not isinstance(host, str) or not host:
        raise SystemExit(f"{manifest_path}: manifest lacks environment.host")
    return {"manifest_sha256": _file_sha256(pathlib.Path(manifest_path)), "host": host, "environment": env}


def build_cross_machine_record(local_manifest_path, peer_manifest_path, pool: str) -> dict:
    """Re-run ``compare_manifests`` on the local and peer manifests of one pool."""
    a = load_pool_manifest(local_manifest_path)
    b = load_pool_manifest(peer_manifest_path)
    if pool not in POOL_QUOTA or pool not in a.get("pools", {}) or pool not in b.get("pools", {}):
        raise SystemExit(f"both manifests must carry pool {pool!r}")
    local, peer = _side(local_manifest_path), _side(peer_manifest_path)
    problems = compare_manifests(a, b, pool)
    if local["host"] == peer["host"]:
        problems = [f"local and peer manifests come from the same host {local['host']!r}"] + problems
    if a.get("suite") != b.get("suite"):
        problems = ["suites differ"] + problems
    return {"protocol": CROSS_MACHINE_PROTOCOL, "pool": pool, "suite": a.get("suite"), "local": local, "peer": peer,
            "problems": problems, "verified": not problems}


def validate_cross_machine(record: dict, pool: str, local_manifest_path, peer_manifest_path) -> None:
    """A cross-machine record is only evidence if it can be recomputed now."""
    if not isinstance(record, dict) or record.get("protocol") != CROSS_MACHINE_PROTOCOL or record.get("pool") != pool:
        raise SystemExit(f"not a cross-machine record for pool {pool}")
    fresh = build_cross_machine_record(local_manifest_path, peer_manifest_path, pool)
    if fresh["problems"] or fresh != record:
        raise SystemExit(f"cross-machine record for pool {pool} cannot be reproduced or reports problems: "
                         f"{fresh['problems'][:3] or 'record content differs'}")


def _validate_task_block(pool: str, suite: str, name: str, task_id: int, info: dict, state_dim: int) -> list[str]:
    quota = POOL_QUOTA[pool]
    if int(info.get("task_id", -1)) != task_id:
        raise SystemExit(f"{pool}/{name}: task id != task manifest")
    entries = info.get("entries") or []
    if len(entries) != quota or [e.get("k") for e in entries] != list(range(quota)):
        raise SystemExit(f"{pool}/{name}: entries are not k = 0..{quota - 1} in order")
    digests = []
    for e in entries:
        k = e["k"]
        if e.get("status") != STATUS_OK:
            raise SystemExit(f"generator_validation_failed: {pool}/{name} k={k} is {e.get('status')!r}")
        attempts = e.get("attempts") or []
        if not attempts or len(attempts) > MAX_RETRIES + 1:
            raise SystemExit(f"{pool}/{name} k={k}: attempt count {len(attempts)} outside 1..{MAX_RETRIES + 1}")
        for i, at in enumerate(attempts):
            expect = derive_seeds(suite, name, pool, k, i)
            if any(at.get(f) != expect[f] for f in ("a", "authority_sha256", "seed32", "py_seed", "np_seed", "env_seed")):
                raise SystemExit(f"{pool}/{name} k={k} attempt {i}: seed derivation != the frozen authority")
            if (at.get("outcome") == "accepted") != (i == len(attempts) - 1):
                raise SystemExit(f"{pool}/{name} k={k}: only the last attempt may be accepted")
        if e.get("shape") != [state_dim] or e.get("dtype") != "float64":
            raise SystemExit(f"{pool}/{name} k={k}: shape/dtype != official state width {state_dim}")
        d = e.get("state_sha256")
        if not isinstance(d, str) or not _SHA_RE.match(d):
            raise SystemExit(f"{pool}/{name} k={k}: state digest malformed")
        digests.append(d)
    if len(set(digests)) != quota:
        raise SystemExit(f"{pool}/{name}: duplicate state digests inside the pool")
    return digests


def validate_pools(*, p_manifest_path, c_manifest_path, p_dir, c_dir, apool_dir, task_manifest_path, bddl_root,
                   cross_p_record, cross_p_peer, cross_c_record, cross_c_peer, assets_dir) -> dict:
    """The fresh-pool finalizer: everything re-derived from files, nothing self-reported."""
    paths = {k: (str(pathlib.Path(v).resolve()) if v else None) for k, v in {
        "p_manifest_path": p_manifest_path, "c_manifest_path": c_manifest_path, "p_dir": p_dir, "c_dir": c_dir,
        "apool_dir": apool_dir, "task_manifest_path": task_manifest_path, "bddl_root": bddl_root,
        "cross_p_record": cross_p_record,
        "cross_p_peer": cross_p_peer, "cross_c_record": cross_c_record, "cross_c_peer": cross_c_peer,
        "assets_dir": assets_dir}.items()}
    manifests = {"P": load_pool_manifest(p_manifest_path), "C": load_pool_manifest(c_manifest_path)}
    dirs = {"P": pathlib.Path(p_dir), "C": pathlib.Path(c_dir)}
    tm = load_task_manifest(task_manifest_path)
    tm_sha = _file_sha256(pathlib.Path(task_manifest_path))
    suite = tm.get("suite")
    tasks = sorted(tm["tasks"], key=lambda t: t["task_id"])
    bddl_dir = pathlib.Path(bddl_root)
    for t in tasks:
        rel = pathlib.Path(t["bddl_file"])
        if rel.is_absolute() or ".." in rel.parts:
            raise SystemExit(f"task {t['task_name']}: bddl_file must be relative to --bddl-root")
        bddl = bddl_dir / rel
        if not bddl.is_file() or _file_sha256(bddl) != t["bddl_sha256"]:
            raise SystemExit(f"task {t['task_name']}: BDDL bytes != task manifest")
    apool = pathlib.Path(apool_dir)
    dims = task_state_dims(apool, [t["task_name"] for t in tasks])
    official = {t["task_name"]: official_state_digests(apool, t["task_name"]) for t in tasks}
    rollups = {}
    for pool, m in manifests.items():
        if m.get("suite") != suite:
            raise SystemExit(f"{pool} manifest suite != task manifest suite")
        if m.get("task_manifest_sha256") != tm_sha:
            raise SystemExit(f"{pool} manifest does not bind this task manifest")
        if m.get("state_dim") != dims:
            raise SystemExit(f"{pool} manifest state_dim {m.get('state_dim')} != official per-task widths {dims}")
        if m.get("seed_namespace") != SEED_NAMESPACE or m.get("max_retries") != MAX_RETRIES:
            raise SystemExit(f"{pool} manifest seed namespace / retry budget != frozen")
        rollups[pool] = (m.get("environment") or {}).get("assets_rollup")
        if not isinstance(rollups[pool], dict) or not _SHA_RE.match(str(rollups[pool].get("sha256"))):
            raise SystemExit(f"{pool} manifest lacks the asset rollup")
    if rollups["P"] != rollups["C"]:
        raise SystemExit("P and C were generated against different asset rollups")
    if assets_rollup(assets_dir) != rollups["P"]:
        raise SystemExit("asset rollup recomputed from --assets-dir != the manifests")
    pools_out = {}
    digests_by_pool: dict[str, dict[str, list[str]]] = {}
    for pool, m in manifests.items():
        block = m["pools"].get(pool)
        if block is None or set(m["pools"]) != {pool}:
            raise SystemExit(f"{pool} manifest must carry exactly the {pool} pool")
        if block.get("quota") != POOL_QUOTA[pool]:
            raise SystemExit(f"{pool} manifest quota != {POOL_QUOTA[pool]}")
        if set(block["tasks"]) != {t["task_name"] for t in tasks}:
            raise SystemExit(f"{pool} manifest tasks != task manifest")
        file_sha = block.get("init_file_sha256") or {}
        per_task: dict[str, list[str]] = {}
        files: dict[str, str] = {}
        for t in tasks:
            name = t["task_name"]
            info = block["tasks"][name]
            if info.get("bddl_sha256") != t["bddl_sha256"]:
                raise SystemExit(f"{pool}/{name}: bddl digest != task manifest")
            per_task[name] = _validate_task_block(pool, suite, name, t["task_id"], info, dims[name])
            path = dirs[pool] / f"{name}.init"
            if not path.is_file():
                raise SystemExit(f"{pool}/{name}: materialised init file missing: {path}")
            got = _file_sha256(path)
            if file_sha.get(name) != got:
                raise SystemExit(f"{pool}/{name}: init file bytes != manifest init_file_sha256")
            states = np.asarray(load_init_states(path))
            if states.shape != (POOL_QUOTA[pool], dims[name]):
                raise SystemExit(f"{pool}/{name}: init file shape {states.shape} != ({POOL_QUOTA[pool]}, {dims[name]})")
            if [state_sha256(st) for st in states] != per_task[name]:
                raise SystemExit(f"{pool}/{name}: materialised states != manifest digests (k order)")
            files[name] = got
        digests_by_pool[pool] = per_task
        rec_path, peer_path = (cross_p_record, cross_p_peer) if pool == "P" else (cross_c_record, cross_c_peer)
        record = json.loads(pathlib.Path(rec_path).read_text())
        validate_cross_machine(record, pool, p_manifest_path if pool == "P" else c_manifest_path, peer_path)
        pools_out[pool] = {"manifest_sha256": _file_sha256(pathlib.Path(p_manifest_path if pool == "P" else c_manifest_path)),
                           "quota": POOL_QUOTA[pool], "init_file_sha256": files,
                           "cross_machine": {"record_sha256": _file_sha256(pathlib.Path(rec_path)),
                                             "peer_manifest_sha256": record["peer"]["manifest_sha256"],
                                             "local_host": record["local"]["host"], "peer_host": record["peer"]["host"]}}
    excl = {"official_vs_P": 0, "official_vs_C": 0, "P_vs_C": 0}
    for name in official:
        p_set, c_set = set(digests_by_pool["P"][name]), set(digests_by_pool["C"][name])
        excl["official_vs_P"] += len(official[name] & p_set)
        excl["official_vs_C"] += len(official[name] & c_set)
        excl["P_vs_C"] += len(p_set & c_set)
    if any(excl.values()):
        raise SystemExit(f"fresh pools are not exclusive: {excl}")
    return {"protocol": VALIDATION_PROTOCOL, "suite": suite, "state_dim": dims, "task_manifest_sha256": tm_sha,
            "tasks": [{"task_id": t["task_id"], "task_name": t["task_name"], "bddl_sha256": t["bddl_sha256"]} for t in tasks],
            "official": {"states_per_task": {n: len(s) for n, s in official.items()},
                         "file_sha256": {t["task_name"]: _file_sha256(apool / f"{t['task_name']}.init") for t in tasks}},
            "pools": pools_out, "exclusivity": excl, "assets_rollup": rollups["P"], "inputs": {"paths": paths}, "passed": True}


def validate_pool_validation(artifact_path, *, p_manifest_path, c_manifest_path) -> dict:
    """The seal's entry point: re-run the finalizer from the artifact's own inputs."""
    art = json.loads(pathlib.Path(artifact_path).read_text())
    if not isinstance(art, dict) or art.get("protocol") != VALIDATION_PROTOCOL or art.get("passed") is not True:
        raise SystemExit("not a passing fresh-pool validation artifact")
    paths = (art.get("inputs") or {}).get("paths") or {}
    if set(paths) != set(VALIDATION_PATH_KEYS):
        raise SystemExit("fresh-pool validation artifact input paths are not exact")
    if _file_sha256(pathlib.Path(p_manifest_path)) != art["pools"]["P"]["manifest_sha256"] \
            or _file_sha256(pathlib.Path(c_manifest_path)) != art["pools"]["C"]["manifest_sha256"]:
        raise SystemExit("fresh-pool validation artifact binds different P / C manifests")
    fresh = validate_pools(**paths)
    if fresh != art:
        raise SystemExit("fresh-pool validation artifact cannot be reproduced from its inputs")
    return art


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _generate(args) -> None:
    tm = load_task_manifest(args.task_manifest)
    if tm["suite"] != args.suite:
        raise SystemExit("task manifest suite != --suite")
    tasks = []
    for t in tm["tasks"]:
        tasks.append({"task_id": int(t["task_id"]), "task_name": t["task_name"],
                      "bddl_file": str(pathlib.Path(args.bddl_root) / t["bddl_file"]), "bddl_sha256": t["bddl_sha256"]})
    for t in tasks:
        if not pathlib.Path(t["bddl_file"]).is_file() or _file_sha256(pathlib.Path(t["bddl_file"])) != t["bddl_sha256"]:
            raise SystemExit(f"{t['task_name']}: BDDL file missing or its digest != task manifest")
    apool = pathlib.Path(args.apool_dir)
    dims = task_state_dims(apool, [t["task_name"] for t in tasks])
    exclude = {}
    if args.exclude_manifest:
        other = load_pool_manifest(args.exclude_manifest)
        for pool_block in other["pools"].values():
            for name, info in pool_block["tasks"].items():
                exclude.setdefault(name, set()).update(e["state_sha256"] for e in info["entries"] if e["state_sha256"])
    block, states = generate_pool(args.suite, args.pool, tasks, apool_dir=apool, state_dim=dims, exclude=exclude)
    out = pathlib.Path(args.out_dir)
    manifest = {"schema": 1, "protocol": PROTOCOL, "suite": args.suite, "seed_namespace": SEED_NAMESPACE,
                "max_retries": MAX_RETRIES, "state_dim": dims, "environment": environment_record(args.assets_dir),
                "task_manifest_sha256": _file_sha256(pathlib.Path(args.task_manifest)),
                "pools": {args.pool: block}}
    (out / "pool_manifest_draft.json").parent.mkdir(parents=True, exist_ok=True)
    (out / "pool_manifest_draft.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    assert_pool_complete(block, args.pool)
    file_digests = materialize(block, states, out / args.pool)
    manifest["pools"][args.pool]["init_file_sha256"] = file_digests
    (out / f"pool_manifest_{args.pool}.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"pool {args.pool}: {sum(len(i['entries']) for i in block['tasks'].values())} states, manifest written")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--suite", required=True)
    g.add_argument("--pool", required=True, choices=sorted(POOL_QUOTA))
    g.add_argument("--task-manifest", required=True, help="exp/dispatch_surface/config/task_manifest_<suite>.json")
    g.add_argument("--bddl-root", required=True, help="directory holding the suite's BDDL files")
    g.add_argument("--apool-dir", required=True, help="official 50/task super-pool (exclusivity + state width)")
    g.add_argument("--assets-dir", required=True, help="simulator asset directory (content rollup)")
    g.add_argument("--exclude-manifest", default="", help="another pool manifest (P<->C exclusivity)")
    g.add_argument("--out-dir", required=True)
    x = sub.add_parser("cross_machine")
    x.add_argument("--pool", required=True, choices=sorted(POOL_QUOTA))
    x.add_argument("--local-manifest", required=True)
    x.add_argument("--peer-manifest", required=True, help="the manifest regenerated on the other machine")
    x.add_argument("--out", required=True)
    v = sub.add_parser("validate")
    for key in VALIDATION_PATH_KEYS:
        v.add_argument("--" + key.replace("_", "-"), required=True)
    v.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "generate":
        _generate(args)
    elif args.cmd == "cross_machine":
        rec = build_cross_machine_record(args.local_manifest, args.peer_manifest, args.pool)
        pathlib.Path(args.out).write_text(json.dumps(rec, indent=2, sort_keys=True))
        if rec["problems"]:
            raise SystemExit(f"cross-machine determinism FAILED for pool {args.pool}: {rec['problems'][:5]}")
        print(f"cross-machine record for pool {args.pool}: verified")
    else:
        art = validate_pools(**{k: getattr(args, k) for k in VALIDATION_PATH_KEYS})
        pathlib.Path(args.out).write_text(json.dumps(art, indent=2, sort_keys=True))
        print(json.dumps({"state_dim": art["state_dim"], "exclusivity": art["exclusivity"], "passed": art["passed"]}, indent=2))


if __name__ == "__main__":
    main()
