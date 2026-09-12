"""Synthetic on-disk node evidence for testing catalog and analysis validation.

These records explicitly model completed test processes; separate runner tests
exercise real process ownership, startup, failure and termination.
"""

from __future__ import annotations

from pathlib import Path

from exp.ablation_study.cache_prune.common import (
    VERSION,
    file_identity,
    seal,
    write_json,
)
from exp.ablation_study.cache_prune.run_prune_eval import GIB, host_budget


def write_batch_evidence(
    directory: Path, freeze: dict, arm: dict, launch: dict
) -> dict:
    """Complete synthetic journal files with internally cross-checked node records."""
    process = {"pid": 999991, "start_time": 100.0}
    endpoint = "127.0.0.1:12345"
    apool = freeze["memberships"][arm["suite"]]["apool"]
    implementation = [file_identity(__file__)]
    model = {"test_model": "synthetic"}
    launch = seal(
        {
            **launch,
            "version": VERSION,
            "kind": "launch",
            "probe": False,
            "server_process": process,
            "batch_id": directory.name,
            "node": "synthetic-node",
            "endpoint": endpoint,
            "launch_spec": {"workers": 1, "gpu_budget_gib": 4},
            "implementation": implementation,
            "model": model,
            "client_probe": {
                "num_steps_wait": 10,
                "replan_steps": 5,
                "max_steps": freeze["protocol"]["max_steps"],
                "per_task_digests": apool["per_task_digests"],
            },
            "host_budget": host_budget(128 * GIB, 100 * GIB),
            "mem_total": 128 * GIB,
            "mem_available_before": 100 * GIB,
            "estimated_rss": 8 * GIB,
            "estimated_server_rss": 4 * GIB,
        }
    )
    write_json(directory / "launch.json", launch)
    common = {
        "batch_id": directory.name,
        "arm": arm["arm"],
        "endpoint": endpoint,
        "process": process,
        "yaml": arm["yaml"],
        "library": arm["artifact"]["file"],
        "node": launch["node"],
        "implementation": implementation,
        "model": model,
    }
    write_json(
        directory / "node_preflight.json",
        seal({"version": VERSION, "kind": "node_preflight", **common}),
    )
    (directory / "server.log").write_text(
        f"Loaded {arm['artifact']['entries']} entries from {arm['artifact']['file']['path']}\n"
    )
    worker = {
        "pid": 999992,
        "start_time": 100.1,
        "argv": [
            "python",
            "-m",
            "examples.libero.worker_entry",
            "--task-suite-name",
            arm["suite"],
            "--server-key",
            endpoint,
            "--init-states-dir",
            apool["apool_dir"],
        ],
    }
    sample = {
        "server_rss": 3 * GIB,
        "task_rss": 5 * GIB,
        "processes": [process],
        "workers": [worker],
        "threads": {str(process["pid"]): 4},
        "gpus": [{"owned_bytes": GIB}],
    }
    import json

    (directory / "resources.jsonl").write_text(json.dumps(sample) + "\n")
    write_json(
        directory / "per_step.jsonl.launch.json",
        {
            "suite": arm["suite"],
            "arms": [arm["arm"]],
            "trials_per_task": launch["trials"],
            "smoke": launch["smoke"],
            "apool": apool,
        },
    )
    post = seal(
        {
            "version": VERSION,
            "kind": "node_postflight",
            **common,
            "owned_processes_exited": True,
            "server_returncode": -15,
            "peak_server_rss": 3 * GIB,
            "peak_task_rss": 5 * GIB,
            "resource_samples": 1,
            "peak_gpu_bytes": GIB,
            "worker_processes": [worker],
            "evidence_files": {
                name: file_identity(directory / name)
                for name in (
                    "server.log",
                    "resources.jsonl",
                    "per_step.jsonl.launch.json",
                )
            },
        }
    )
    write_json(directory / "node_postflight.json", post)
    completion = seal(
        {
            "version": VERSION,
            "kind": "completion",
            "status": "valid",
            "failure": None,
            "freeze_digest": freeze["digest"],
            "arm": arm["arm"],
            "launch_digest": launch["digest"],
            "smoke": launch["smoke"],
            "probe": False,
            "node_postflight": file_identity(directory / "node_postflight.json"),
            "journal": file_identity(directory / "journal.jsonl"),
            "per_step": file_identity(directory / "per_step.jsonl"),
        }
    )
    write_json(directory / "completion.json", completion)
    return launch
