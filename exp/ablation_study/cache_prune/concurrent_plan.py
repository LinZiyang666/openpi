"""Freeze bounded multi-library waves and lossless per-arm conductor views.

Placement uses the existing conductor algorithm. All libraries ever resident in
a wave count towards RAM, independently of the scheduler's activation limit.
Raw mixed-arm journals remain authoritative; views preserve every attempt.
"""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path

from openpi.conductor.driver import assign_servers
from openpi.conductor.task import ServerEndpoint

from .common import SUITES, VERSION, require, seal
from .run_prune_eval import GIB, SERVER_LIMIT


def validate_spec(spec: dict) -> None:
    """Reject ambiguous placement, impossible budgets and unsupported worker sizes."""
    require(spec["model_config"] == "pi05_libero", "pi05_libero is required")
    require(bool(spec["conda_env"]), "a LIBERO client environment is required")
    require(spec["host"] in ("127.0.0.1", "localhost"), "run on the serving node")
    require(spec["endpoints"], "at least one concurrent endpoint is required")
    ports = [e["port"] for e in spec["endpoints"]]
    require(len(set(ports)) == len(ports), "duplicate server ports")
    require(all(type(p) is int and 1024 <= p <= 65535 for p in ports), "invalid ports")
    for endpoint in spec["endpoints"]:
        require(
            type(endpoint["workers"]) is int and endpoint["workers"] > 0,
            "each endpoint needs workers",
        )
        require(bool(endpoint["stage1_device"]), "missing Stage 1 device")
    require(type(spec["gpus"]) is int and spec["gpus"] > 0, "invalid render GPU count")
    require(
        sum(e["workers"] for e in spec["endpoints"]) <= 15 * spec["gpus"],
        "more than 15 LIBERO workers per render GPU",
    )
    for key in ("arms_per_server", "eval_concurrency"):
        require(
            type(spec[key]) is int and spec[key] >= 2, f"{key} must allow YAML overlap"
        )
    for key in (
        "host_budget_gib",
        "gpu_budget_gib",
        "model_budget_gib",
        "client_worker_budget_gib",
        "resident_multiplier",
    ):
        require(
            isinstance(spec[key], (int, float))
            and math.isfinite(spec[key])
            and spec[key] > 0,
            f"invalid {key}",
        )
    require(
        spec["resident_multiplier"] >= 2, "resident multiplier must include load copies"
    )


def placement(arms: list[dict], spec: dict) -> list[dict]:
    """Match run_size_eval's ordered equal-weight, worker-capacity placement."""
    endpoints = spec["endpoints"][: min(len(arms), len(spec["endpoints"]))]
    servers = [ServerEndpoint(spec["host"], e["port"]) for e in endpoints]
    capacities = {s.key: e["workers"] for s, e in zip(servers, endpoints, strict=True)}
    assigned = assign_servers(
        {a["arm"]: 100 for a in arms}, servers, capacities=capacities
    )
    result = []
    for server, endpoint in zip(servers, endpoints, strict=True):
        members = [a for a in arms if assigned[a["arm"]] == server]
        if not members:
            continue
        # BackendPool deduplicates paths, not YAML IDs, and never evicts them.
        libraries = {
            a["artifact"]["file"]["path"]: a["artifact"]["file"]["bytes"]
            for a in members
        }
        estimate = int(
            spec["model_budget_gib"] * GIB
            + spec["resident_multiplier"] * sum(libraries.values())
        )
        result.append(
            {
                "endpoint": server.key,
                "port": endpoint["port"],
                "stage1_device": endpoint["stage1_device"],
                "workers": endpoint["workers"],
                "arms": [a["arm"] for a in members],
                "estimated_rss": estimate,
            }
        )
    return result


def build_schedule(freeze: dict, spec: dict) -> dict:
    """Plan both phases before outcomes, shrinking waves to fit cumulative RAM."""
    validate_spec(spec)
    waves = {"smoke": [], "eval": []}
    for phase in waves:
        for suite in SUITES:
            pending = [
                a
                for a in freeze["arms"]
                if a["suite"] == suite
                and (phase == "eval" or a["point"] in ("P00", "P09"))
            ]
            pending.sort(key=lambda a: a["arm"])
            while pending:
                maximum = min(
                    len(pending), len(spec["endpoints"]) * spec["arms_per_server"]
                )
                for size in range(maximum, 0, -1):
                    chosen = pending[:size]
                    servers = placement(chosen, spec)
                    estimate = sum(
                        s["estimated_rss"]
                        + int(s["workers"] * spec["client_worker_budget_gib"] * GIB)
                        for s in servers
                    )
                    if estimate <= spec["host_budget_gib"] * GIB and all(
                        s["estimated_rss"] <= SERVER_LIMIT
                        and len(s["arms"]) <= spec["arms_per_server"]
                        for s in servers
                    ):
                        break
                else:
                    raise ValueError(
                        f"no memory-feasible placement for {pending[0]['arm']}"
                    )
                waves[phase].append(
                    {
                        "wave_id": f"{phase}_{len(waves[phase]):03d}",
                        "phase": phase,
                        "suite": suite,
                        "arms": [a["arm"] for a in chosen],
                        "servers": servers,
                        "estimated_rss": estimate,
                    }
                )
                pending = pending[size:]
    return seal(
        {
            "version": VERSION,
            "kind": "concurrent_schedule",
            "freeze_digest": freeze["digest"],
            "spec": spec,
            "waves": waves,
        }
    )


def arm_launch(freeze: dict, arm: dict, wave: dict, wave_directory: Path) -> dict:
    """Project shared launch identity while retaining the unchanged episode contract."""
    server = next(s for s in wave["plan"]["servers"] if arm["arm"] in s["arms"])
    return seal(
        {
            "version": VERSION,
            "kind": "concurrent_arm_launch",
            "freeze_digest": freeze["digest"],
            "arm": arm["arm"],
            "num_steps_wait": 10,
            "replan_steps": 5,
            "max_steps": freeze["protocol"]["max_steps"][arm["suite"]],
            "trials": 1 if wave["plan"]["phase"] == "smoke" else 50,
            "smoke": wave["plan"]["phase"] == "smoke",
            "probe": False,
            "endpoint": server["endpoint"],
            "wave_digest": wave["digest"],
            "wave_directory": str(wave_directory.resolve()),
            "rss_scope": "shared_server_wave",
            "latency_scope": "concurrent_call",
        }
    )


def partition_rows(
    source: Path, destinations: dict[str, Path] | None, names: list[str]
) -> dict[str, str]:
    """Hash or write byte-preserving partitions, rejecting foreign and truncated rows."""
    hashes = {name: hashlib.sha256() for name in names}
    with ExitStack() as stack:
        outputs = (
            {
                name: stack.enter_context(path.open("xb"))
                for name, path in destinations.items()
            }
            if destinations is not None
            else {}
        )
        with source.open("rb") as handle:
            for line in handle:
                require(
                    line.endswith(b"\n") and bool(line.strip()),
                    "truncated or blank JSONL row",
                )
                row = json.loads(line)
                name = row.get("yaml_id")
                require(
                    name in hashes
                    and str(row.get("task_uid", "")).startswith(name + ":eval:"),
                    "foreign arm or episode in conductor output",
                )
                hashes[name].update(line)
                if outputs:
                    outputs[name].write(line)
    return {name: value.hexdigest() for name, value in hashes.items()}
