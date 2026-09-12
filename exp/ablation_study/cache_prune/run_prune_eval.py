"""Run one immutable pruning arm per owned server process and evidence batch.

Invoke on the serving node, where preload hashes and process ownership can be
measured directly. The existing cache-size runner still owns worker scheduling.
Invalid batches retain their files; recovery uses a new full-arm batch only.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import uuid

import psutil

from .common import (
    VERSION,
    check_identity,
    check_seal,
    digest,
    environment,
    file_identity,
    implementation_identity,
    read_json,
    require,
    seal,
    write_json,
)
from .emit_prune_arms import validate_freeze
from .prepare_membership import checked_apool

GIB = 1024**3
SERVER_LIMIT = 64 * GIB
RESERVE = 16 * GIB
LOADED = re.compile(r"Loaded (\d+) entries from (.+?)\s*$", re.MULTILINE)


def gpu_snapshot(owned_pids: set[int]) -> list[dict]:
    """Measure only this batch's GPU allocations through NVML, per physical GPU."""
    import pynvml

    pynvml.nvmlInit()
    try:
        devices = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            allocations = {}
            for query in (
                pynvml.nvmlDeviceGetComputeRunningProcesses,
                pynvml.nvmlDeviceGetGraphicsRunningProcesses,
            ):
                for process in query(handle):
                    if process.pid in owned_pids:
                        used = process.usedGpuMemory
                        require(
                            isinstance(used, int) and 0 <= used <= memory.total,
                            "GPU allocation is not measurable",
                        )
                        allocations[process.pid] = max(
                            allocations.get(process.pid, 0), used
                        )
            devices.append(
                {
                    "index": index,
                    "uuid": str(pynvml.nvmlDeviceGetUUID(handle)),
                    "name": str(pynvml.nvmlDeviceGetName(handle)),
                    "total": memory.total,
                    "free": memory.free,
                    "owned_bytes": sum(allocations.values()),
                }
            )
        require(
            bool(devices), "no GPU telemetry; cannot validate the smoke memory budget"
        )
        return devices
    finally:
        pynvml.nvmlShutdown()


def process_high_water(process: psutil.Process) -> int:
    """Read Linux VmHWM so short-lived allocation peaks between samples are retained."""
    rows = Path(f"/proc/{process.pid}/status").read_text().splitlines()
    values = [int(line.split()[1]) * 1024 for line in rows if line.startswith("VmHWM:")]
    require(len(values) == 1, "process high-water RSS is unavailable")
    return max(values[0], process.memory_info().rss)


def host_budget(total: int, available: int) -> int:
    """Reserve 16 GiB and cap this experiment at three quarters of node RAM."""
    return max(0, int(min(0.75 * total, available - RESERVE)))


def validate_load_log(text: str, identity: dict, entries: int) -> None:
    """Require one actual artifact load, allowing repeated deduplicated config loads."""
    events = LOADED.findall(text)
    require(len(events) == 1, "server must log exactly one actual artifact load")
    count, path = events[0]
    require(
        int(count) == entries
        and Path(path).resolve() == Path(identity["path"]).resolve(),
        "server loaded another artifact or entry count",
    )


def _process_identity(process: psutil.Process) -> dict:
    return {"pid": process.pid, "start_time": process.create_time()}


def _alive(identity):
    try:
        p = psutil.Process(identity["pid"])
        return (
            p.create_time() == identity["start_time"]
            and p.status() != psutil.STATUS_ZOMBIE
        )
    except psutil.NoSuchProcess:
        return False


def _terminate_owned(identities):
    processes = []
    for identity in identities:
        if _alive(identity):
            process = psutil.Process(identity["pid"])
            process.terminate()
            processes.append(process)
    _, alive = psutil.wait_procs(processes, timeout=10)
    for process in alive:
        process.kill()
    psutil.wait_procs(alive, timeout=10)
    require(
        not any(_alive(identity) for identity in identities),
        "owned processes did not exit",
    )


def _client_probe(spec: dict, apool: dict, out: Path) -> dict:
    # Use the worker's interpreter and import environment, not the server's
    # installed defaults. The probe is deliberately Python 3.8 compatible.
    script = """import hashlib,json,pathlib,sys
from examples.libero import main as m
import numpy as np
import torch,libero,robosuite
base=pathlib.Path(sys.argv[1])
if list(base.glob('*.pruned_init')): raise ValueError('unexpected pruned_init override')
files=sorted(base.glob('*.init'))
result={'num_steps_wait':m.Args().num_steps_wait,'replan_steps':m.Args().replan_steps,
'resize_size':m.Args().resize_size,'max_steps':{s:m._get_max_steps(s) for s in ('libero_spatial','libero_10')},
'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,
'libero_file':str(pathlib.Path(libero.__file__).resolve()),'robosuite_version':getattr(robosuite,'__version__','unknown'),
'libero_code':{str(p.relative_to(pathlib.Path(libero.__file__).parent)):hashlib.sha256(p.read_bytes()).hexdigest()
 for p in sorted(pathlib.Path(libero.__file__).parent.rglob('*.py'))},
'per_task_digests':{p.stem:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
print('CACHE_PRUNE_PROBE='+json.dumps(result,sort_keys=True))
"""
    conda_env = spec["conda_env"]
    flag = "-p" if "/" in conda_env else "-n"
    command = [
        "conda",
        "run",
        "--no-capture-output",
        flag,
        conda_env,
        "python",
        "-c",
        script,
        apool["apool_dir"],
    ]
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")
    }
    venv_bin = str(Path(os.environ.get("VIRTUAL_ENV", sys.prefix)) / "bin")
    env["PATH"] = os.pathsep.join(
        p for p in env.get("PATH", "").split(os.pathsep) if p != venv_bin
    )
    env["PYTHONPATH"] = os.pathsep.join((str(Path.cwd()), str(Path.cwd() / "src")))
    env["MUJOCO_GL"] = "egl"
    try:
        completed = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=600, check=True
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        captured = []
        for part in (exc.stdout, exc.stderr):
            captured.append(
                part.decode(errors="replace") if isinstance(part, bytes) else part or ""
            )
        (out / "client_probe.log").write_text("".join(captured))
        write_json(
            out / "preflight_failure.json",
            seal(
                {
                    "version": VERSION,
                    "kind": "infra_failure",
                    "phase": "client_probe",
                    "status": "invalid",
                    "error_type": type(exc).__name__,
                    "timeout_seconds": 600,
                    "argv": command,
                    "log": file_identity(out / "client_probe.log"),
                }
            ),
        )
        raise
    (out / "client_probe.log").write_text(completed.stdout + completed.stderr)
    lines = [
        line.split("=", 1)[1]
        for line in completed.stdout.splitlines()
        if line.startswith("CACHE_PRUNE_PROBE=")
    ]
    require(len(lines) == 1, "client environment probe did not return one result")
    probe = json.loads(lines[0])
    require(
        probe["num_steps_wait"] == 10
        and probe["replan_steps"] == 5
        and probe["max_steps"] == {"libero_spatial": 220, "libero_10": 520},
        "client timing semantics changed",
    )
    require(
        probe["per_task_digests"] == apool["per_task_digests"],
        "client is loading different A-pool bytes",
    )
    return probe


def _model_identity(spec):
    root = Path(spec["checkpoint_dir"]).resolve(strict=True)
    require(root.is_dir(), "checkpoint must be an existing local directory")
    files = [file_identity(p) for p in sorted(root.rglob("*")) if p.is_file()]
    require(
        any(p["path"].endswith(".safetensors") for p in files),
        "checkpoint has no model weights",
    )
    require(
        any("norm_stats" in p["path"] for p in files),
        "checkpoint has no normalization statistics",
    )
    return {
        "config": spec["model_config"],
        "directory": str(root),
        "files": files,
        "digest": digest(files),
    }


def validate_run(
    freeze_manifest: dict, suite: str, apool_record: str | Path, run_dir: str | Path
) -> dict:
    """Revalidate frozen artifacts and A-pool before constructing any subprocess."""
    path = Path(run_dir)
    # The first batch validates all frozen inputs. Subsequent batches rehash
    # their served PKL/YAML and A-pool, not 900 unrelated collection HDF5s.
    validate_freeze(freeze_manifest, rehash=not (path / "run.json").exists())
    require(suite in freeze_manifest["memberships"], "unknown suite")
    membership = freeze_manifest["memberships"][suite]
    require(
        file_identity(apool_record) == membership["apool_record"],
        "launch points to another A-pool record",
    )
    require(
        checked_apool(apool_record, suite) == membership["apool"],
        "launch A-pool changed",
    )
    if (path / "run.json").exists():
        record = read_json(path / "run.json")
        check_seal(record)
        require(
            record["freeze_digest"] == freeze_manifest["digest"],
            "resume input changed; create a new run_id",
        )
    return {
        "freeze_digest": freeze_manifest["digest"],
        "suite": suite,
        "apool_record": membership["apool_record"],
        "apool": membership["apool"],
    }


def _server_command(spec, arm):
    require(
        Path(spec["server_python"]).absolute() == Path(sys.executable).absolute(),
        "server_python must be this runner's interpreter so recorded server versions are authoritative",
    )
    require(
        spec["model_config"] == "pi05_libero",
        "this experiment freezes the pi05_libero teacher",
    )
    require(
        spec["host"] in ("127.0.0.1", "localhost"),
        "run on the serving node with an owned loopback endpoint",
    )
    require(
        type(spec["port"]) is int and 1024 <= spec["port"] <= 65535,
        "invalid dedicated server port",
    )
    return [
        spec["server_python"],
        "scripts/serve_policy.py",
        "--port",
        str(spec["port"]),
        "--replicas",
        "1",
        "--cache-config",
        arm["yaml"]["path"],
        "--stage1-device",
        spec["stage1_device"],
        "--stage2-device",
        spec["stage2_device"],
        "--stage3-device",
        spec["stage3_device"],
        "policy:checkpoint",
        "--policy.config",
        spec["model_config"],
        "--policy.dir",
        spec["checkpoint_dir"],
    ]


def _eval_command(spec, freeze, arm, directory, smoke):
    return [
        sys.executable,
        "-m",
        "exp.ablation_study.cache_size.run_size_eval",
        "--arm-matrix",
        freeze["matrices"][arm["suite"]]["path"],
        "--task-suite",
        arm["suite"],
        "--arms",
        arm["arm"],
        "--servers",
        f"{spec['host']}:{spec['port']}",
        "--workers",
        str(spec["workers"]),
        "--trials",
        "1" if smoke else "50",
        "--journal",
        str(directory / "journal.jsonl"),
        "--per-step-out",
        str(directory / "per_step.jsonl"),
        "--apool-record",
        freeze["memberships"][arm["suite"]]["apool_record"]["path"],
        "--conda-env",
        spec["conda_env"],
        "--gpus",
        str(spec["gpus"]),
        "--eval-concurrency",
        str(spec["workers"]),
        "--episode-timeout-s",
        "1800",
        "--min-full-hit",
        "1",
        *(["--smoke"] if smoke else []),
    ]


def validate_batch_evidence(directory: str | Path, freeze: dict) -> dict:
    """Recheck node hashes, one actual load, memory and owned-process termination."""
    directory = Path(directory)
    launch = read_json(directory / "launch.json")
    pre, post = (
        read_json(directory / "node_preflight.json"),
        read_json(directory / "node_postflight.json"),
    )
    for item in (launch, pre, post):
        check_seal(item)
    arm = next(a for a in freeze["arms"] if a["arm"] == launch["arm"])
    require(
        launch["implementation"] == pre["implementation"] == post["implementation"],
        "runtime implementation changed",
    )
    require(
        launch["model"] == pre["model"] == post["model"], "runtime model bytes changed"
    )
    require(
        launch["client_probe"]["num_steps_wait"] == launch["num_steps_wait"] == 10
        and launch["client_probe"]["replan_steps"] == launch["replan_steps"] == 5,
        "client defaults differ from launch",
    )
    require(
        launch["client_probe"]["max_steps"][arm["suite"]] == launch["max_steps"]
        and launch["client_probe"]["per_task_digests"]
        == freeze["memberships"][arm["suite"]]["apool"]["per_task_digests"],
        "client environment/pool differs from freeze",
    )
    require(
        launch["freeze_digest"] == freeze["digest"]
        and launch["batch_id"] == directory.name,
        "batch freeze/identity mismatch",
    )
    for key in ("batch_id", "arm", "endpoint", "process", "yaml", "library", "node"):
        require(pre[key] == post[key], f"node binding changed: {key}")
    require(
        pre["arm"] == arm["arm"]
        and pre["library"] == arm["artifact"]["file"]
        and pre["yaml"] == arm["yaml"]
        and pre["endpoint"] == launch["endpoint"]
        and pre["process"] == launch["server_process"]
        and pre["node"] == launch["node"],
        "node evidence differs from launch",
    )
    require(
        post["owned_processes_exited"] is True
        and post["server_returncode"] in (0, -15),
        "server did not finish cleanly",
    )
    require(
        post["peak_server_rss"] <= SERVER_LIMIT
        and post["peak_task_rss"] <= launch["host_budget"],
        "memory budget violated",
    )
    require(
        launch["host_budget"]
        == host_budget(launch["mem_total"], launch["mem_available_before"]),
        "invalid host budget",
    )
    require(
        launch["estimated_rss"] <= launch["host_budget"]
        and launch["estimated_server_rss"] <= SERVER_LIMIT,
        "underbudgeted launch",
    )
    require(
        post["resource_samples"] > 0 and post["worker_processes"] is not None,
        "missing process telemetry",
    )
    for name, identity in post["evidence_files"].items():
        require(
            Path(identity["path"]).resolve() == (directory / name).resolve(),
            "batch evidence path escapes its directory",
        )
        check_identity(identity)
    require(
        set(post["evidence_files"]) >= {"server.log", "resources.jsonl"},
        "missing server/resource evidence",
    )
    count, server_present = 0, False
    peaks = {"server": 0, "task": 0, "gpu": 0}
    actual_workers = {}
    with (directory / "resources.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            count += 1
            require(
                sample["gpus"] and sample["threads"], "missing GPU/thread telemetry"
            )
            peaks["server"] = max(peaks["server"], sample["server_rss"])
            peaks["task"] = max(peaks["task"], sample["task_rss"])
            peaks["gpu"] = max(
                peaks["gpu"], sum(g["owned_bytes"] for g in sample["gpus"])
            )
            server_present |= pre["process"] in sample["processes"]
            actual_workers.update((p["pid"], p) for p in sample["workers"])
    require(count == post["resource_samples"], "resource sample count mismatch")
    require(
        post["peak_gpu_bytes"] == peaks["gpu"]
        and peaks["gpu"] <= launch["launch_spec"]["gpu_budget_gib"] * GIB,
        "GPU memory budget violated",
    )
    require(
        peaks["server"] == post["peak_server_rss"]
        and peaks["task"] == post["peak_task_rss"],
        "resource peak differs from raw samples",
    )
    require(server_present, "server PID is absent from resource evidence")
    require(
        list(actual_workers.values()) == post["worker_processes"],
        "worker provenance differs from raw samples",
    )
    if not launch["probe"]:
        runner_record = read_json(directory / "per_step.jsonl.launch.json")
        require(
            runner_record
            == {
                "suite": arm["suite"],
                "arms": [arm["arm"]],
                "trials_per_task": launch["trials"],
                "smoke": launch["smoke"],
                "apool": freeze["memberships"][arm["suite"]]["apool"],
            },
            "conductor launch metadata mismatch",
        )
        require(
            "per_step.jsonl.launch.json" in post["evidence_files"],
            "unbound conductor launch metadata",
        )
        require(
            len(actual_workers) >= launch["launch_spec"]["workers"],
            "worker launch evidence missing",
        )
    for worker in actual_workers.values():
        argv = worker["argv"]
        require(
            "examples.libero.worker_entry" in argv, "unexpected worker implementation"
        )
        for flag, expected in (
            ("--task-suite-name", arm["suite"]),
            ("--server-key", launch["endpoint"]),
            (
                "--init-states-dir",
                freeze["memberships"][arm["suite"]]["apool"]["apool_dir"],
            ),
        ):
            require(
                argv.count(flag) == 1 and argv[argv.index(flag) + 1] == expected,
                f"actual worker {flag} mismatch",
            )
        require(
            "--replan-steps" not in argv and "--resize-size" not in argv,
            "unexpected worker override",
        )
    validate_load_log(
        (directory / "server.log").read_text(),
        arm["artifact"]["file"],
        arm["artifact"]["entries"],
    )
    return launch


def validate_completion(directory: str | Path, freeze: dict) -> tuple[dict, dict]:
    """Bind a valid completion to its launch, node evidence and whole result files."""
    directory = Path(directory)
    completion = read_json(directory / "completion.json")
    check_seal(completion)
    launch = validate_batch_evidence(directory, freeze)
    require(
        completion["status"] == "valid" and completion["failure"] is None,
        "batch is not valid",
    )
    require(
        completion["freeze_digest"] == freeze["digest"]
        and completion["launch_digest"] == launch["digest"]
        and completion["arm"] == launch["arm"]
        and completion["smoke"] is launch["smoke"]
        and completion["probe"] is launch["probe"],
        "completion/launch mismatch",
    )
    require(
        completion["node_postflight"]
        == file_identity(directory / "node_postflight.json"),
        "completed node evidence changed",
    )
    if not launch["probe"]:
        require(
            completion["journal"] == file_identity(directory / "journal.jsonl")
            and completion["per_step"] == file_identity(directory / "per_step.jsonl"),
            "completed batch results changed",
        )
    return launch, completion


def accepted_batches(
    run_dir: str | Path, freeze: dict, *, require_complete: bool = False
) -> dict:
    """Choose at most one complete valid whole batch per arm; never splice episodes."""
    from .analysis.analyze_prune import episode_ledger

    catalog = Path(run_dir) / "run.json"
    if catalog.exists() and read_json(catalog).get("kind") == "concurrent_run":
        from .run_concurrent import accepted_concurrent_batches

        return accepted_concurrent_batches(
            run_dir, freeze, require_complete=require_complete
        )

    result = {}
    for directory in sorted(Path(run_dir).glob("batch_*")):
        completion_path = directory / "completion.json"
        if not completion_path.exists():
            launch_path = directory / "launch.json"
            if launch_path.exists():
                launch = read_json(launch_path)
                require(
                    not _alive(launch["server_process"]),
                    "previous batch server is still alive",
                )
            resources = directory / "resources.jsonl"
            if resources.exists():
                for line in resources.read_text().splitlines():
                    if line.strip():
                        sample = json.loads(line)
                        require(
                            not any(_alive(p) for p in sample["processes"]),
                            "previous incomplete batch still has live owned processes",
                        )
            continue
        completion = read_json(completion_path)
        check_seal(completion)
        require(
            completion["freeze_digest"] == freeze["digest"],
            "batch belongs to another freeze",
        )
        if (
            completion["status"] != "valid"
            or completion["smoke"]
            or completion["probe"]
        ):
            continue
        launch, completion = validate_completion(directory, freeze)
        require(
            launch["arm"] == completion["arm"]
            and launch["digest"] == completion["launch_digest"],
            "completion/launch mismatch",
        )
        arm = next(a for a in freeze["arms"] if a["arm"] == launch["arm"])
        journal, per_step = directory / "journal.jsonl", directory / "per_step.jsonl"
        require(
            file_identity(journal) == completion["journal"]
            and file_identity(per_step) == completion["per_step"],
            "completed batch results changed",
        )
        episode_ledger(freeze, arm, journal, per_step, launch)
        require(
            arm["arm"] not in result,
            "two valid batches for one arm; selective replacement is forbidden",
        )
        result[arm["arm"]] = {
            "launch": launch,
            "journal": str(journal),
            "per_step": str(per_step),
            "directory": str(directory),
        }
    if require_complete:
        require(
            set(result) == {a["arm"] for a in freeze["arms"]},
            "formal analysis requires 40 valid whole batches",
        )
    return result


def _require_smokes(run_dir, freeze):
    seen = set()
    for path in Path(run_dir).glob("batch_*/completion.json"):
        completion = read_json(path)
        check_seal(completion)
        if (
            completion["status"] == "valid"
            and completion["smoke"]
            and not completion["probe"]
        ):
            launch, completion = validate_completion(path.parent, freeze)
            from .analysis.analyze_prune import episode_ledger

            arm = next(a for a in freeze["arms"] if a["arm"] == completion["arm"])
            episode_ledger(
                freeze,
                arm,
                path.parent / "journal.jsonl",
                path.parent / "per_step.jsonl",
                launch,
                smoke=True,
            )
            seen.add((arm["suite"], arm["regime"], arm["point"]))
    required = {
        (a["suite"], a["regime"], a["point"])
        for a in freeze["arms"]
        if a["point"] in ("P00", "P09")
    }
    require(
        required <= seen,
        "formal rollout requires all four sources' P00/P09 smoke batches",
    )


def run_batch(
    freeze: dict,
    name: str,
    spec: dict,
    run_dir: str | Path,
    *,
    smoke: bool = False,
    probe: bool = False,
    p00_probe: str | Path | None = None,
) -> dict:
    """Launch one owned server and one thin conductor wrapper, retaining failures."""
    from .analysis.analyze_prune import episode_ledger

    require(name in {a["arm"] for a in freeze["arms"]}, "select exactly one known arm")
    arm = next(a for a in freeze["arms"] if a["arm"] == name)
    require(not probe or arm["point"] == "P00", "memory probes must use P00")
    require(
        type(spec["workers"]) is int and 0 < spec["workers"] and spec["gpus"] > 0,
        "invalid worker capacity",
    )
    require(
        spec["client_budget_gib"] > 0 and bool(spec["conda_env"]),
        "client environment and memory budget required",
    )
    require(
        math.isfinite(spec["gpu_budget_gib"]) and spec["gpu_budget_gib"] > 0,
        "finite positive GPU budget required",
    )
    directory = Path(run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    node_lock = Path(
        f"/tmp/openpi-cache-prune-{os.getuid()}-{socket.gethostname()}.lock"
    ).open("a")
    try:
        fcntl.flock(node_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validated = validate_run(
            freeze,
            arm["suite"],
            freeze["memberships"][arm["suite"]]["apool_record"]["path"],
            directory,
        )
        check_identity(arm["artifact"]["file"])
        check_identity(arm["yaml"])
        accepted = accepted_batches(directory, freeze)
        require(
            name not in accepted, "a complete valid batch already exists for this arm"
        )
        if not probe and not smoke:
            _require_smokes(directory, freeze)
        batch_id = "batch_" + uuid.uuid4().hex
        batch = directory / batch_id
        batch.mkdir()
        server_command = _server_command(spec, arm)
        eval_command = _eval_command(spec, freeze, arm, batch, smoke)
        client = _client_probe(spec, validated["apool"], batch)
        model = _model_identity(spec)
        implementation = implementation_identity()
        model_record = {
            "version": VERSION,
            "kind": "run",
            "freeze_digest": freeze["digest"],
            "model_digest": model["digest"],
            "model_config": spec["model_config"],
            "stage_devices": [spec[f"stage{i}_device"] for i in (1, 2, 3)],
            "workers": spec["workers"],
            "conda_env": spec["conda_env"],
            "gpus": spec["gpus"],
            "implementation": implementation,
            "client_version": {
                k: v for k, v in client.items() if k != "per_task_digests"
            },
            "gpu_budget_gib": spec["gpu_budget_gib"],
            "runtime_environment": {
                key: os.environ.get(key)
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        }
        if (directory / "run.json").exists():
            require(
                read_json(directory / "run.json") == seal(model_record),
                "run model/worker settings changed; create a new run_id",
            )
        else:
            write_json(directory / "run.json", seal(model_record))
        memory = psutil.virtual_memory()
        budget = host_budget(memory.total, memory.available)
        estimated_server = 2 * arm["artifact"]["file"]["bytes"]
        if not probe:
            require(p00_probe is not None, "a measured P00 preload probe is required")
            prior = read_json(p00_probe)
            check_seal(prior)
            require(
                prior["status"] == "valid"
                and prior["probe"] is True
                and prior["source_digest"] == arm["source_digest"],
                "wrong/incomplete P00 memory probe",
            )
            require(
                prior["model_digest"] == model["digest"]
                and prior["node"] == socket.gethostname(),
                "probe uses another model/node",
            )
            prior_launch, _ = validate_completion(Path(p00_probe).parent, freeze)
            require(
                prior_launch["probe"] is True
                and prior_launch["digest"] == prior["launch_digest"],
                "memory probe lacks valid process evidence",
            )
            estimated_server = max(
                int(1.25 * prior["peak_server_rss"]), estimated_server
            )
        estimated = (
            estimated_server
            + int(spec["client_budget_gib"] * GIB)
            + psutil.Process().memory_info().rss
        )
        require(
            estimated_server <= SERVER_LIMIT and estimated <= budget,
            "estimated memory exceeds server/node budget",
        )
        with socket.socket() as reservation:
            reservation.bind((spec["host"], spec["port"]))
        server = evaluator = None
        owned, workers = {}, {}
        peaks = {"server": 0, "task": 0, "gpu": 0}
        samples = 0
        last_resource_write = 0.0
        last_process_set = None
        success = False
        failure = None
        launch = None
        pre = None
        server_log = (batch / "server.log").open("x")
        resource_log = (batch / "resources.jsonl").open("x")
        eval_log = (batch / "eval.log").open("x")

        def sample():
            nonlocal samples, last_resource_write, last_process_set
            server_rss, task_rss = 0, psutil.Process().memory_info().rss
            threads = {}
            for proc, is_server in ((server, True), (evaluator, False)):
                if proc is None or proc.poll() is not None:
                    continue
                parent = psutil.Process(proc.pid)
                for child in [parent, *parent.children(recursive=True)]:
                    try:
                        identity = _process_identity(child)
                        rss = child.memory_info().rss
                        threads[str(child.pid)] = child.num_threads()
                        owned[child.pid] = identity
                        task_rss += rss
                        if is_server:
                            server_rss += process_high_water(child)
                        elif "examples.libero.worker_entry" in child.cmdline():
                            workers[child.pid] = {**identity, "argv": child.cmdline()}
                    except (
                        psutil.NoSuchProcess,
                        psutil.ZombieProcess,
                        FileNotFoundError,
                    ):
                        continue
            gpus = gpu_snapshot(
                {pid for pid, identity in owned.items() if _alive(identity)}
            )
            gpu_bytes = sum(g["owned_bytes"] for g in gpus)
            old_peaks = dict(peaks)
            peaks["server"] = max(peaks["server"], server_rss)
            peaks["task"] = max(peaks["task"], task_rss)
            peaks["gpu"] = max(peaks["gpu"], gpu_bytes)
            process_set = (tuple(owned), tuple(workers))
            now = time.monotonic()
            # Retain every newly observed peak and ownership change while
            # reducing unchanged telemetry. Measurement still runs every 0.5s.
            if (
                peaks != old_peaks
                or process_set != last_process_set
                or now - last_resource_write >= 2
            ):
                samples += 1
                resource_log.write(
                    json.dumps(
                        {
                            "time": time.time(),
                            "server_rss": server_rss,
                            "task_rss": task_rss,
                            "processes": list(owned.values()),
                            "workers": list(workers.values()),
                            "gpus": gpus,
                            "threads": threads,
                        }
                    )
                    + "\n"
                )
                resource_log.flush()
                last_resource_write, last_process_set = now, process_set
            require(
                server_rss <= SERVER_LIMIT
                and task_rss <= budget
                and gpu_bytes <= spec["gpu_budget_gib"] * GIB,
                "measured memory exceeds server/node budget",
            )

        try:
            server = subprocess.Popen(
                server_command,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            identity = _process_identity(psutil.Process(server.pid))
            owned[server.pid] = identity
            common = {
                "batch_id": batch_id,
                "arm": name,
                "endpoint": f"{spec['host']}:{spec['port']}",
                "process": identity,
                "yaml": file_identity(arm["yaml"]["path"]),
                "library": file_identity(arm["artifact"]["file"]["path"]),
                "node": socket.gethostname(),
                "implementation": implementation,
                "model": model,
            }
            pre = seal({"version": VERSION, "kind": "node_preflight", **common})
            write_json(batch / "node_preflight.json", pre)
            launch = seal(
                {
                    "version": VERSION,
                    "kind": "launch",
                    "freeze_digest": freeze["digest"],
                    "arm": name,
                    "batch_id": batch_id,
                    "server_process": identity,
                    "node": common["node"],
                    "endpoint": common["endpoint"],
                    "server_argv": server_command,
                    "eval_argv": eval_command,
                    "num_steps_wait": 10,
                    "replan_steps": 5,
                    "max_steps": freeze["protocol"]["max_steps"][arm["suite"]],
                    "trials": 1 if smoke else 50,
                    "smoke": smoke,
                    "probe": probe,
                    "client_probe": client,
                    "model": model,
                    "implementation": implementation,
                    "launch_spec": spec,
                    "environment": environment(),
                    "host_budget": budget,
                    "mem_total": memory.total,
                    "mem_available_before": memory.available,
                    "estimated_server_rss": estimated_server,
                    "estimated_rss": estimated,
                    "p00_probe": file_identity(p00_probe) if p00_probe else None,
                    "supersedes_invalid_batches": sorted(
                        p.name
                        for p in directory.glob("batch_*")
                        if p != batch
                        and (p / "launch.json").exists()
                        and read_json(p / "launch.json")["arm"] == name
                        and (
                            not (p / "completion.json").exists()
                            or read_json(p / "completion.json")["status"] == "invalid"
                        )
                    ),
                }
            )
            write_json(batch / "launch.json", launch)
            deadline = time.monotonic() + 1200
            while True:
                sample()
                require(server.poll() is None, "server exited before ready")
                events = LOADED.findall((batch / "server.log").read_text())
                require(len(events) <= 1, "server loaded more than one artifact")
                connections = psutil.Process(server.pid).net_connections(kind="tcp")
                listening = any(
                    c.status == psutil.CONN_LISTEN and c.laddr.port == spec["port"]
                    for c in connections
                )
                if events and listening:
                    validate_load_log(
                        (batch / "server.log").read_text(),
                        arm["artifact"]["file"],
                        arm["artifact"]["entries"],
                    )
                    break
                require(time.monotonic() < deadline, "server startup timeout")
                time.sleep(0.5)
            if probe:
                for _ in range(10):
                    sample()
                    time.sleep(0.5)
            else:
                evaluator = subprocess.Popen(
                    eval_command,
                    stdout=eval_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                owned[evaluator.pid] = _process_identity(psutil.Process(evaluator.pid))
                while evaluator.poll() is None:
                    sample()
                    require(server.poll() is None, "server exited during rollout")
                    time.sleep(0.5)
                require(evaluator.returncode == 0, "conductor evaluation failed")
                episode_ledger(
                    freeze,
                    arm,
                    batch / "journal.jsonl",
                    batch / "per_step.jsonl",
                    launch,
                    smoke=smoke,
                )
                require(
                    len(workers) >= spec["workers"],
                    "missing actual worker launch evidence",
                )
            check_identity(arm["artifact"]["file"])
            check_identity(arm["yaml"])
            validate_load_log(
                (batch / "server.log").read_text(),
                arm["artifact"]["file"],
                arm["artifact"]["entries"],
            )
            success = True
        except BaseException as exc:
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            for proc in (server, evaluator):
                if proc is not None and proc.poll() is None:
                    try:
                        for child in psutil.Process(proc.pid).children(recursive=True):
                            owned[child.pid] = _process_identity(child)
                    except psutil.NoSuchProcess:
                        pass
            _terminate_owned(list(owned.values()))
            if server is not None:
                server.wait(timeout=10)
            if evaluator is not None:
                evaluator.wait(timeout=10)
            server_log.close()
            resource_log.close()
            eval_log.close()
        require(
            launch is not None and pre is not None, f"server did not launch: {failure}"
        )
        post = seal(
            {
                "version": VERSION,
                "kind": "node_postflight",
                **common,
                "library": file_identity(arm["artifact"]["file"]["path"]),
                "yaml": file_identity(arm["yaml"]["path"]),
                "implementation": implementation_identity(),
                "model": _model_identity(spec),
                "server_returncode": server.returncode,
                "owned_processes_exited": not any(_alive(i) for i in owned.values()),
                "peak_server_rss": peaks["server"],
                "peak_task_rss": peaks["task"],
                "peak_gpu_bytes": peaks["gpu"],
                "resource_samples": samples,
                "worker_processes": list(workers.values()),
                "evidence_files": {
                    name: file_identity(batch / name)
                    for name in (
                        "server.log",
                        "resources.jsonl",
                        "eval.log",
                        "client_probe.log",
                    )
                    + (
                        ()
                        if probe or not (batch / "per_step.jsonl.launch.json").exists()
                        else ("per_step.jsonl.launch.json",)
                    )
                },
            }
        )
        write_json(batch / "node_postflight.json", post)
        if success:
            try:
                validate_batch_evidence(batch, freeze)
            except ValueError as exc:
                success, failure = False, str(exc)
        completion = seal(
            {
                "version": VERSION,
                "kind": "completion",
                "status": "valid" if success else "invalid",
                "failure": failure,
                "freeze_digest": freeze["digest"],
                "arm": name,
                "launch_digest": launch["digest"],
                "source_digest": arm["source_digest"],
                "node": socket.gethostname(),
                "model_digest": model["digest"],
                "probe": probe,
                "smoke": smoke,
                "peak_server_rss": peaks["server"],
                "peak_task_rss": peaks["task"],
                "node_postflight": file_identity(batch / "node_postflight.json"),
                "journal": file_identity(batch / "journal.jsonl")
                if (batch / "journal.jsonl").exists()
                else None,
                "per_step": file_identity(batch / "per_step.jsonl")
                if (batch / "per_step.jsonl").exists()
                else None,
            }
        )
        write_json(batch / "completion.json", completion)
        require(
            success,
            f"invalid batch retained at {batch}: {failure}; remedy conditions and use a fresh full batch",
        )
        return completion
    finally:
        node_lock.close()
        lock.close()


def main() -> None:
    """Run an explicitly requested probe, smoke or formal single-arm batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--arm", "--arms", dest="arm", required=True)
    parser.add_argument("--launch-spec", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--p00-probe", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--probe", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = run_batch(
        read_json(args.freeze),
        args.arm,
        read_json(args.launch_spec),
        args.run_dir,
        smoke=args.smoke,
        probe=args.probe,
        p00_probe=args.p00_probe,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
