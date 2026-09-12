"""Run bounded multi-YAML waves with the existing concurrent server and conductor.

Servers belong to one wave, preload its full library set, and serve episode-level
worker pulls across YAMLs. Process, input and raw-output evidence is retained for
every attempt. Analysis accepts only complete waves and lossless per-arm views.
"""

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import uuid

import psutil
import yaml

from . import run_prune_eval as legacy
from .common import (
    VERSION,
    check_identity,
    check_seal,
    file_identity,
    implementation_identity,
    read_json,
    require,
    seal,
    write_json,
)
from .concurrent_plan import arm_launch, build_schedule, partition_rows
from .emit_prune_arms import validate_freeze
from .prepare_membership import checked_apool


def validate_loads(text: str, arms: list[dict]) -> None:
    """Require exactly the assigned library set, loaded once per physical path."""
    wanted = {
        (a["artifact"]["entries"], str(Path(a["artifact"]["file"]["path"]).resolve()))
        for a in arms
    }
    actual = Counter(
        (int(n), str(Path(p).resolve())) for n, p in legacy.LOADED.findall(text)
    )
    require(
        actual == Counter(wanted), "server loaded a missing, extra or duplicate library"
    )


def _server_spec(spec, server):
    return {**spec, **server, "workers": server["workers"]}


def _eval_command(spec, freeze, plan, directory):
    return [
        sys.executable,
        "-m",
        "exp.ablation_study.cache_size.run_size_eval",
        "--arm-matrix",
        str(directory / "matrix.yaml"),
        "--task-suite",
        plan["suite"],
        "--servers",
        ",".join(s["endpoint"] for s in plan["servers"]),
        "--server-workers",
        ",".join(str(s["workers"]) for s in plan["servers"]),
        "--workers",
        str(sum(s["workers"] for s in plan["servers"])),
        "--trials",
        "1" if plan["phase"] == "smoke" else "50",
        "--journal",
        str(directory / "journal.jsonl"),
        "--per-step-out",
        str(directory / "per_step.jsonl"),
        "--apool-record",
        freeze["memberships"][plan["suite"]]["apool_record"]["path"],
        "--conda-env",
        spec["conda_env"],
        "--gpus",
        str(spec["gpus"]),
        "--eval-concurrency",
        str(spec["eval_concurrency"]),
        "--episode-timeout-s",
        "1800",
        "--min-full-hit",
        "1",
        *(["--smoke"] if plan["phase"] == "smoke" else []),
    ]


def _preload(server, arms):
    from examples.libero.episode_runner import default_client_factory
    from openpi.conductor.task import ServerEndpoint

    client = default_client_factory(
        ServerEndpoint(server["endpoint"].rsplit(":", 1)[0], server["port"])
    )
    receipts = []
    try:
        for arm in arms:
            check_identity(arm["yaml"])
            reply = client.load_cache_config(
                yaml_content=Path(arm["yaml"]["path"]).read_text(),
                yaml_id=arm["arm"],
                bundle_id=arm["arm"],
            )
            require(
                reply.get("__ack__") == "load_cache_config"
                and reply.get("bundle_id") == arm["arm"],
                "wrong bundle preload acknowledgement",
            )
            receipts.append(
                {
                    "arm": arm["arm"],
                    "yaml": arm["yaml"],
                    "library": arm["artifact"]["file"],
                    "reply": reply,
                }
            )
    finally:
        client.close()
    return receipts


def _monitored_call(function, sample, timeout=1200):
    result, errors = [], []

    def invoke():
        try:
            result.append(function())
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=invoke, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    while thread.is_alive():
        sample()
        require(time.monotonic() < deadline, "bundle preload timeout")
        thread.join(0.5)
    if errors:
        raise errors[0]
    return result[0]


def _views(directory, freeze, launch, post):
    from .analysis.analyze_prune import episode_ledger

    arms = {a["arm"]: a for a in freeze["arms"]}
    locations = {}
    for name in launch["plan"]["arms"]:
        view = directory.parent / f"view_{directory.name}_{name}"
        view.mkdir()
        projected = arm_launch(freeze, arms[name], launch, directory)
        write_json(view / "launch.json", projected)
        shared_peak = post["server_peaks"][projected["endpoint"]]
        write_json(
            view / "node_postflight.json",
            seal(
                {
                    "version": VERSION,
                    "kind": "concurrent_arm_resources",
                    "wave_postflight": file_identity(directory / "postflight.json"),
                    "rss_scope": "shared_server_wave",
                    "peak_server_rss": None,
                    "shared_server_peak_rss": shared_peak,
                }
            ),
        )
        locations[name] = view
    for filename in ("journal.jsonl", "per_step.jsonl"):
        partition_rows(
            directory / filename,
            {n: p / filename for n, p in locations.items()},
            list(locations),
        )
    records = {}
    for name, view in locations.items():
        projected = read_json(view / "launch.json")
        episode_ledger(
            freeze,
            arms[name],
            view / "journal.jsonl",
            view / "per_step.jsonl",
            projected,
            smoke=projected["smoke"],
        )
        records[name] = {
            "directory": str(view),
            "launch": file_identity(view / "launch.json"),
            "journal": file_identity(view / "journal.jsonl"),
            "per_step": file_identity(view / "per_step.jsonl"),
            "resources": file_identity(view / "node_postflight.json"),
        }
    return records


def validate_wave(directory: str | Path, freeze: dict, run: dict) -> dict:
    """Reconcile all shared process evidence and every per-arm episode partition."""
    from .analysis.analyze_prune import episode_ledger

    directory = Path(directory).resolve()
    launch, post, completion = [
        read_json(directory / name)
        for name in ("launch.json", "postflight.json", "completion.json")
    ]
    for value in (run, launch, post, completion):
        check_seal(value)
    require(
        completion["status"] == "valid"
        and completion["launch_digest"] == launch["digest"]
        and launch["run_digest"] == run["digest"]
        and launch["freeze_digest"] == run["freeze_digest"] == freeze["digest"],
        "wave/run/freeze identity mismatch",
    )
    require(
        launch["attempt_id"] == directory.name
        and post["launch_digest"] == launch["digest"],
        "wave launch changed",
    )
    check_identity(completion["postflight"])
    require(
        Path(completion["postflight"]["path"]) == directory / "postflight.json",
        "postflight path escaped wave",
    )
    plan = launch["plan"]
    require(
        run["schedule"] == build_schedule(freeze, run["spec"]),
        "schedule placement changed",
    )
    require(
        plan in run["schedule"]["waves"][plan["phase"]],
        "wave differs from frozen schedule",
    )
    require(
        completion["phase"] == plan["phase"]
        and completion["wave_id"] == plan["wave_id"]
        and completion["failure"] is None,
        "completion phase/identity mismatch",
    )
    arms = {a["arm"]: a for a in freeze["arms"]}
    require(
        post["implementation"] == run["implementation"]
        and post["model"] == run["model"],
        "code or model changed during wave",
    )
    require(
        post["inputs"] == launch["inputs"], "library/YAML bytes changed during wave"
    )
    wanted_inputs = {
        n: {"yaml": arms[n]["yaml"], "library": arms[n]["artifact"]["file"]}
        for n in plan["arms"]
    }
    require(launch["inputs"] == wanted_inputs, "wave loaded another frozen input")
    require(
        post["owned_processes_exited"]
        and post["evaluator_returncode"] == 0
        and all(code in (0, -15) for code in post["server_returncodes"].values()),
        "wave processes did not finish cleanly",
    )
    require(
        set(post["server_returncodes"]) == {s["endpoint"] for s in plan["servers"]},
        "missing server exit evidence",
    )
    apool = freeze["memberships"][plan["suite"]]["apool"]
    client = launch["client_probe"]
    require(
        client["num_steps_wait"] == 10
        and client["replan_steps"] == 5
        and client["max_steps"] == freeze["protocol"]["max_steps"]
        and client["per_task_digests"] == apool["per_task_digests"],
        "client pool/protocol mismatch",
    )
    for filename, identity in post["files"].items():
        require(
            Path(identity["path"]) == directory / filename, "evidence path escaped wave"
        )
        check_identity(identity)
    required = {
        "journal.jsonl",
        "per_step.jsonl",
        "per_step.jsonl.launch.json",
        "matrix.yaml",
        "resources.jsonl",
        "preloads.json",
        "eval.log",
        "client_probe.log",
    }
    required |= {f"server_{s['port']}.log" for s in plan["servers"]}
    require(set(post["files"]) == required, "missing raw wave evidence")
    runner_record = read_json(directory / "per_step.jsonl.launch.json")
    require(
        runner_record
        == {
            "suite": plan["suite"],
            "arms": sorted(plan["arms"]),
            "trials_per_task": 1 if plan["phase"] == "smoke" else 50,
            "smoke": plan["phase"] == "smoke",
            "apool": apool,
        },
        "conductor launched another episode grid",
    )
    receipts = read_json(directory / "preloads.json")
    require(
        set(receipts) == {s["endpoint"] for s in plan["servers"]},
        "foreign preload endpoint",
    )
    require(
        yaml.safe_load((directory / "matrix.yaml").read_text())
        == {
            "arms": [
                {"arm": n, "yaml": arms[n]["yaml"]["path"], "sidecar": None}
                for n in plan["arms"]
            ]
        },
        "wave matrix changed",
    )
    require(
        launch["eval_argv"] == _eval_command(run["spec"], freeze, plan, directory),
        "conductor command differs from plan",
    )
    for server in plan["servers"]:
        require(
            launch["server_argv"][server["endpoint"]]
            == legacy._server_command(
                _server_spec(run["spec"], server), arms[server["arms"][0]]
            ),
            "server command changed",
        )
        validate_loads(
            (directory / f"server_{server['port']}.log").read_text(),
            [arms[n] for n in server["arms"]],
        )
        rows = receipts[server["endpoint"]]
        require(
            [r["arm"] for r in rows] == server["arms"], "incomplete bundle receipts"
        )
        for row in rows:
            require(
                row["yaml"] == arms[row["arm"]]["yaml"]
                and row["library"] == arms[row["arm"]]["artifact"]["file"]
                and row["reply"].get("__ack__") == "load_cache_config"
                and row["reply"].get("bundle_id") == row["arm"],
                "bundle receipt mismatch",
            )
    server_peaks = {s["endpoint"]: 0 for s in plan["servers"]}
    task_peak = gpu_peak = count = 0
    workers, observed = {}, set()
    with (directory / "resources.jsonl").open() as handle:
        for line in handle:
            sample = json.loads(line)
            count += 1
            require(sample["gpus"] and sample["threads"], "missing GPU/thread evidence")
            for endpoint, value in sample["server_rss"].items():
                require(endpoint in server_peaks, "unknown sampled server")
                server_peaks[endpoint] = max(server_peaks[endpoint], value)
            task_peak = max(task_peak, sample["task_rss"])
            gpu_peak = max(gpu_peak, sum(g["owned_bytes"] for g in sample["gpus"]))
            observed.update((p["pid"], p["start_time"]) for p in sample["processes"])
            workers.update((p["pid"], p) for p in sample["workers"])
    budget = min(
        int(run["spec"]["host_budget_gib"] * legacy.GIB),
        legacy.host_budget(launch["mem_total"], launch["mem_available"]),
    )
    require(
        count == post["resource_samples"]
        and count > 0
        and server_peaks == post["server_peaks"]
        and task_peak == post["task_peak"]
        and gpu_peak == post["gpu_peak"]
        and launch["host_budget"] == budget
        and task_peak <= budget
        and plan["estimated_rss"] <= budget
        and gpu_peak <= run["spec"]["gpu_budget_gib"] * legacy.GIB
        and all(v <= legacy.SERVER_LIMIT for v in server_peaks.values()),
        "resource budget/evidence mismatch",
    )
    require(list(workers.values()) == post["workers"], "worker census changed")
    for server in plan["servers"]:
        pid = launch["server_processes"][server["endpoint"]]
        require(
            (pid["pid"], pid["start_time"]) in observed, "server absent from telemetry"
        )
        matching = []
        for worker in workers.values():
            argv = worker["argv"]
            require("examples.libero.worker_entry" in argv, "unknown worker")
            for flag, expected in (
                ("--task-suite-name", plan["suite"]),
                ("--init-states-dir", apool["apool_dir"]),
            ):
                require(
                    flag in argv and argv[argv.index(flag) + 1] == expected,
                    "worker pool/suite mismatch",
                )
            require("--server-key" in argv, "worker server missing")
            endpoint = argv[argv.index("--server-key") + 1]
            require(endpoint in server_peaks, "worker uses another endpoint")
            if endpoint == server["endpoint"]:
                matching.append(worker)
        require(len(matching) >= server["workers"], "missing endpoint worker evidence")
    require(set(completion["views"]) == set(plan["arms"]), "incomplete arm views")
    partitions = {
        key: partition_rows(directory / filename, None, plan["arms"])
        for key, filename in (
            ("journal", "journal.jsonl"),
            ("per_step", "per_step.jsonl"),
        )
    }
    for name, view in completion["views"].items():
        expected_dir = directory.parent / f"view_{directory.name}_{name}"
        require(Path(view["directory"]) == expected_dir, "arm view directory changed")
        for key, filename in (
            ("launch", "launch.json"),
            ("journal", "journal.jsonl"),
            ("per_step", "per_step.jsonl"),
            ("resources", "node_postflight.json"),
        ):
            require(
                Path(view[key]["path"]) == expected_dir / filename,
                "arm view path changed",
            )
            check_identity(view[key])
        require(
            all(view[key]["sha256"] == partitions[key][name] for key in partitions),
            "arm view omits or alters raw attempts",
        )
        projected = read_json(view["launch"]["path"])
        require(
            projected == arm_launch(freeze, arms[name], launch, directory),
            "arm launch changed",
        )
        resource = read_json(view["resources"]["path"])
        check_seal(resource)
        require(
            resource["wave_postflight"] == completion["postflight"]
            and resource["peak_server_rss"] is None
            and resource["shared_server_peak_rss"]
            == server_peaks[projected["endpoint"]],
            "shared RSS incorrectly attributed",
        )
        episode_ledger(
            freeze,
            arms[name],
            view["journal"]["path"],
            view["per_step"]["path"],
            projected,
            smoke=projected["smoke"],
        )
    return completion


def accepted_concurrent_batches(
    run_dir, freeze, *, require_complete=False, phase="eval"
) -> dict:
    """Admit complete waves once, retaining unsuccessful attempts without splicing."""
    run_dir = Path(run_dir).resolve()
    run = read_json(run_dir / "run.json")
    check_seal(run)
    require(
        run["kind"] == "concurrent_run" and run["freeze_digest"] == freeze["digest"],
        "wrong concurrent run catalog",
    )
    result, waves = {}, set()
    for directory in sorted(run_dir.glob("wave_*")):
        path = directory / "completion.json"
        if not path.exists():
            telemetry = directory / "resources.jsonl"
            if telemetry.exists():
                with telemetry.open() as handle:
                    for line in handle:
                        require(
                            not any(
                                legacy._alive(p) for p in json.loads(line)["processes"]
                            ),
                            "unfinished wave still owns live processes",
                        )
            continue
        completion = read_json(path)
        check_seal(completion)
        require(completion["freeze_digest"] == freeze["digest"], "foreign wave")
        if completion["status"] != "valid":
            require(
                completion.get("owned_processes_exited") is True
                and not any(legacy._alive(p) for p in completion["owned_processes"]),
                "failed wave still owns live processes",
            )
            continue
        if completion["phase"] != phase:
            continue
        validate_wave(directory, freeze, run)
        require(completion["wave_id"] not in waves, "duplicate valid wave")
        waves.add(completion["wave_id"])
        for name, view in completion["views"].items():
            require(name not in result, "duplicate valid arm")
            result[name] = {
                "launch": read_json(view["launch"]["path"]),
                "journal": view["journal"]["path"],
                "per_step": view["per_step"]["path"],
                "directory": view["directory"],
            }
    if require_complete:
        expected = {
            a["arm"]
            for a in freeze["arms"]
            if phase == "eval" or a["point"] in ("P00", "P09")
        }
        require(set(result) == expected, "incomplete concurrent phase")
    return result


def run_wave(freeze: dict, run: dict, plan: dict, run_dir: str | Path) -> dict:
    """Own a complete concurrent wave, including failure evidence and process cleanup."""
    root = Path(run_dir).resolve()
    spec = run["spec"]
    require(plan in run["schedule"]["waves"][plan["phase"]], "unplanned wave")
    existing = accepted_concurrent_batches(root, freeze, phase=plan["phase"])
    require(not set(plan["arms"]) & set(existing), "valid wave/arm already completed")
    if plan["phase"] == "eval":
        accepted_concurrent_batches(root, freeze, require_complete=True, phase="smoke")
    memory = psutil.virtual_memory()
    budget = min(
        int(spec["host_budget_gib"] * legacy.GIB),
        legacy.host_budget(memory.total, memory.available),
    )
    require(plan["estimated_rss"] <= budget, "current host headroom below planned wave")
    arms = {a["arm"]: a for a in freeze["arms"]}
    inputs = {
        name: {"yaml": arms[name]["yaml"], "library": arms[name]["artifact"]["file"]}
        for name in plan["arms"]
    }
    for pair in inputs.values():
        for identity in pair.values():
            check_identity(identity)
    apool = checked_apool(
        freeze["memberships"][plan["suite"]]["apool_record"]["path"], plan["suite"]
    )
    require(apool == freeze["memberships"][plan["suite"]]["apool"], "A-pool changed")
    directory = root / f"wave_{plan['wave_id']}_{uuid.uuid4().hex}"
    directory.mkdir()
    processes, owned, workers, server_pids, logs = {}, {}, {}, {}, {}
    peaks = {s["endpoint"]: 0 for s in plan["servers"]}
    task_peak = gpu_peak = samples = 0
    last_write = 0.0
    last_owned = None
    evaluator = None
    launch = None
    error = None
    resources = (directory / "resources.jsonl").open("x")
    eval_log = (directory / "eval.log").open("x")

    def sample():
        nonlocal task_peak, gpu_peak, samples, last_write, last_owned
        before = (dict(peaks), task_peak, gpu_peak)
        task_rss = psutil.Process().memory_info().rss
        server_rss, threads = {}, {}
        for endpoint, process in list(processes.items()) + (
            [("evaluator", evaluator)] if evaluator else []
        ):
            if process.poll() is not None:
                continue
            parent = psutil.Process(process.pid)
            subtotal = 0
            for child in [parent, *parent.children(recursive=True)]:
                try:
                    identity = legacy._process_identity(child)
                    owned[child.pid] = identity
                    threads[str(child.pid)] = child.num_threads()
                    task_rss += child.memory_info().rss
                    subtotal += legacy.process_high_water(child)
                    argv = child.cmdline()
                    if "examples.libero.worker_entry" in argv:
                        workers[child.pid] = {**identity, "argv": argv}
                except (psutil.NoSuchProcess, psutil.ZombieProcess, FileNotFoundError):
                    continue
            if endpoint != "evaluator":
                server_rss[endpoint] = subtotal
                peaks[endpoint] = max(peaks[endpoint], subtotal)
        gpus = legacy.gpu_snapshot(
            {p for p, identity in owned.items() if legacy._alive(identity)}
        )
        task_peak = max(task_peak, task_rss)
        gpu_peak = max(gpu_peak, sum(g["owned_bytes"] for g in gpus))
        identities = (tuple(owned), tuple(workers))
        now = time.monotonic()
        if (
            before != (peaks, task_peak, gpu_peak)
            or identities != last_owned
            or now - last_write >= 2
        ):
            resources.write(
                json.dumps(
                    {
                        "time": time.time(),
                        "server_rss": server_rss,
                        "task_rss": task_rss,
                        "gpus": gpus,
                        "threads": threads,
                        "processes": list(owned.values()),
                        "workers": list(workers.values()),
                    }
                )
                + "\n"
            )
            resources.flush()
            samples += 1
            last_write, last_owned = now, identities
        require(
            task_peak <= budget
            and gpu_peak <= spec["gpu_budget_gib"] * legacy.GIB
            and all(v <= legacy.SERVER_LIMIT for v in peaks.values()),
            "wave memory budget exceeded",
        )

    try:
        client = legacy._client_probe(spec, apool, directory)
        require(
            implementation_identity() == run["implementation"]
            and legacy._model_identity(spec) == run["model"],
            "run code/model changed",
        )
        matrix = {
            "arms": [
                {"arm": n, "yaml": arms[n]["yaml"]["path"], "sidecar": None}
                for n in plan["arms"]
            ]
        }
        (directory / "matrix.yaml").write_text(yaml.safe_dump(matrix, sort_keys=False))
        commands = {}
        for server in plan["servers"]:
            endpoint = server["endpoint"]
            with socket.socket() as reservation:
                reservation.bind((spec["host"], server["port"]))
            command = legacy._server_command(
                _server_spec(spec, server), arms[server["arms"][0]]
            )
            commands[endpoint] = command
            logs[endpoint] = (directory / f"server_{server['port']}.log").open("x")
            process = subprocess.Popen(
                command,
                stdout=logs[endpoint],
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes[endpoint] = process
            identity = legacy._process_identity(psutil.Process(process.pid))
            server_pids[endpoint] = identity
            owned[process.pid] = identity
            deadline = time.monotonic() + 1200
            while True:
                sample()
                require(process.poll() is None, "server exited before ready")
                listening = any(
                    c.status == psutil.CONN_LISTEN and c.laddr.port == server["port"]
                    for c in psutil.Process(process.pid).net_connections(kind="tcp")
                )
                if listening and legacy.LOADED.search(
                    (directory / f"server_{server['port']}.log").read_text()
                ):
                    break
                require(time.monotonic() < deadline, "server startup timeout")
                time.sleep(0.5)
        launch = seal(
            {
                "version": VERSION,
                "kind": "concurrent_wave_launch",
                "freeze_digest": freeze["digest"],
                "run_digest": run["digest"],
                "attempt_id": directory.name,
                "plan": plan,
                "inputs": inputs,
                "server_processes": server_pids,
                "server_argv": commands,
                "eval_argv": _eval_command(spec, freeze, plan, directory),
                "client_probe": client,
                "node": socket.gethostname(),
                "mem_total": memory.total,
                "mem_available": memory.available,
                "host_budget": budget,
            }
        )
        write_json(directory / "launch.json", launch)
        receipts = {}
        for server in plan["servers"]:
            receipts[server["endpoint"]] = _monitored_call(
                lambda s=server: _preload(s, [arms[n] for n in s["arms"]]), sample
            )
            validate_loads(
                (directory / f"server_{server['port']}.log").read_text(),
                [arms[n] for n in server["arms"]],
            )
        write_json(directory / "preloads.json", receipts)
        evaluator = subprocess.Popen(
            launch["eval_argv"],
            stdout=eval_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        owned[evaluator.pid] = legacy._process_identity(psutil.Process(evaluator.pid))
        while evaluator.poll() is None:
            sample()
            require(
                all(p.poll() is None for p in processes.values()),
                "server exited during evaluation",
            )
            time.sleep(0.5)
        require(evaluator.returncode == 0, "conductor evaluation failed")
        sample()
        for pair in inputs.values():
            for identity in pair.values():
                check_identity(identity)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        for process in [*processes.values(), *([evaluator] if evaluator else [])]:
            if process.poll() is None:
                try:
                    for child in psutil.Process(process.pid).children(recursive=True):
                        owned[child.pid] = legacy._process_identity(child)
                except psutil.NoSuchProcess:
                    pass
        legacy._terminate_owned(list(owned.values()))
        for process in [*processes.values(), *([evaluator] if evaluator else [])]:
            process.wait(timeout=10)
        for handle in [*logs.values(), eval_log, resources]:
            handle.close()
    completion = {
        "version": VERSION,
        "kind": "concurrent_wave_completion",
        "freeze_digest": freeze["digest"],
        "wave_id": plan["wave_id"],
        "phase": plan["phase"],
        "status": "invalid",
        "failure": error,
        "launch_digest": launch["digest"] if launch else None,
        "owned_processes": list(owned.values()),
        "owned_processes_exited": not any(legacy._alive(p) for p in owned.values()),
    }
    if not error:
        try:
            files = [
                "matrix.yaml",
                "journal.jsonl",
                "per_step.jsonl",
                "per_step.jsonl.launch.json",
                "preloads.json",
                "resources.jsonl",
                "eval.log",
                "client_probe.log",
            ]
            files += [f"server_{s['port']}.log" for s in plan["servers"]]
            post = seal(
                {
                    "version": VERSION,
                    "kind": "concurrent_wave_postflight",
                    "launch_digest": launch["digest"],
                    "inputs": {
                        n: {k: file_identity(v["path"]) for k, v in pair.items()}
                        for n, pair in inputs.items()
                    },
                    "implementation": implementation_identity(),
                    "model": legacy._model_identity(spec),
                    "owned_processes_exited": not any(
                        legacy._alive(p) for p in owned.values()
                    ),
                    "server_returncodes": {
                        k: p.returncode for k, p in processes.items()
                    },
                    "evaluator_returncode": evaluator.returncode,
                    "server_peaks": peaks,
                    "task_peak": task_peak,
                    "gpu_peak": gpu_peak,
                    "resource_samples": samples,
                    "workers": list(workers.values()),
                    "files": {name: file_identity(directory / name) for name in files},
                }
            )
            write_json(directory / "postflight.json", post)
            views = _views(directory, freeze, launch, post)
            completion.update(
                status="valid",
                views=views,
                postflight=file_identity(directory / "postflight.json"),
            )
            write_json(directory / "completion.json", seal(completion))
            validate_wave(directory, freeze, run)
        except BaseException as exc:
            completion.update(status="invalid", failure=f"{type(exc).__name__}: {exc}")
            (directory / "completion.json").write_text(
                json.dumps(seal(completion), indent=2) + "\n"
            )
    else:
        write_json(directory / "completion.json", seal(completion))
    return seal(completion)


def main() -> None:
    """Freeze a schedule or execute its smoke/eval phase with exclusive node ownership."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("plan", "smoke", "eval"), required=True)
    args = parser.parse_args()
    freeze, spec = read_json(args.freeze), read_json(args.spec)
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (
        (root / ".run.lock").open("a") as lock,
        Path(f"/tmp/openpi-cache-prune-{os.getuid()}-{socket.gethostname()}.lock").open(
            "a"
        ) as node_lock,
    ):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(node_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate_freeze(freeze, rehash=not (root / "run.json").exists())
        schedule = build_schedule(freeze, spec)
        if (root / "schedule.json").exists():
            require(
                read_json(root / "schedule.json") == schedule,
                "schedule changed; use a new run",
            )
        else:
            write_json(root / "schedule.json", schedule)
        if args.phase == "plan":
            print(json.dumps({p: len(w) for p, w in schedule["waves"].items()}))
            return
        run = seal(
            {
                "version": VERSION,
                "kind": "concurrent_run",
                "freeze_digest": freeze["digest"],
                "spec": spec,
                "schedule": schedule,
                "implementation": implementation_identity(),
                "model": legacy._model_identity(spec),
                "node": socket.gethostname(),
                "runtime_environment": {
                    k: os.environ.get(k)
                    for k in (
                        "CUDA_VISIBLE_DEVICES",
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                    )
                },
            }
        )
        if (root / "run.json").exists():
            require(
                read_json(root / "run.json") == run, "runtime changed; use a new run"
            )
        else:
            write_json(root / "run.json", run)
        accepted = accepted_concurrent_batches(root, freeze, phase=args.phase)
        for plan in schedule["waves"][args.phase]:
            if set(plan["arms"]) <= set(accepted):
                continue
            result = run_wave(freeze, run, plan, root)
            print(json.dumps(result), flush=True)
            require(
                result["status"] == "valid",
                "invalid wave retained; fix before resuming",
            )
        accepted_concurrent_batches(
            root, freeze, require_complete=True, phase=args.phase
        )


if __name__ == "__main__":
    main()
