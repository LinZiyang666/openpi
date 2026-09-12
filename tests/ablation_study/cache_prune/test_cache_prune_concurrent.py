"""Exercise real conductor placement, tail overlap and owned multi-library waves.

Integration uses actual subprocesses and sockets while replacing GPU/model load
and simulator execution. Raw attempt partitions and shared resource attribution
remain subject to the production validators.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest

from exp.ablation_study.cache_size.run_size_eval import PureCacheEvalStrategy
from openpi.conductor.scheduler import EpisodeScheduler
from openpi.conductor.task import ServerEndpoint

from exp.ablation_study.cache_prune import concurrent_plan as planner
from exp.ablation_study.cache_prune import run_concurrent as concurrent
from exp.ablation_study.cache_prune import run_prune_eval as legacy
from exp.ablation_study.cache_prune.common import (
    file_identity,
    implementation_identity,
    read_json,
    seal,
    write_json,
)
from .test_cache_prune_evidence import episode_files as episode_files
from .test_cache_prune_runner import _augment


def _spec(port):
    return {
        "server_python": sys.executable,
        "model_config": "pi05_libero",
        "checkpoint_dir": "/synthetic",
        "host": "127.0.0.1",
        "conda_env": "synthetic",
        "stage2_device": "meta",
        "stage3_device": "meta",
        "gpus": 1,
        "endpoints": [{"port": port, "workers": 2, "stage1_device": "cuda:0"}],
        "arms_per_server": 2,
        "eval_concurrency": 2,
        "resident_multiplier": 3,
        "model_budget_gib": 0.1,
        "client_worker_budget_gib": 0.1,
        "host_budget_gib": 8,
        "gpu_budget_gib": 4,
    }


def _two_arms(fixture):
    freeze, first, launch, journal, steps, root = _augment(fixture)
    second = copy.deepcopy(first)
    second["arm"] = first["arm"].replace("P00", "P09")
    second["point"] = "P09"
    library, yaml_path = root / "pruned.pkl", root / "pruned.yaml"
    library.write_bytes(b"another library")
    yaml_path.write_text("another config\n")
    second["artifact"]["file"] = file_identity(library)
    second["yaml"] = file_identity(yaml_path)
    freeze["arms"] = [first, second]

    def clone(row):
        return json.loads(json.dumps(row).replace(first["arm"], second["arm"]))

    return freeze, [*journal, *map(clone, journal)], [*steps, *map(clone, steps)], root


def test_real_conductor_can_pull_next_yaml_before_previous_tail_finishes():
    """With an in-flight tail, a worker can immediately pull the next YAML's episode."""
    endpoint = ServerEndpoint("localhost", 8801)
    names = ["arm_a", "arm_b"]
    strategy = PureCacheEvalStrategy("libero_spatial", {n: "unused" for n in names}, 1)
    graph = strategy.plan(names, {n: endpoint for n in names})
    scheduler = EpisodeScheduler(graph, eval_concurrency=2)
    for stage in scheduler.pending_setups():
        scheduler.mark_setup_running(stage.stage_id)
        scheduler.mark_setup_done(stage.stage_id)
    issued = [scheduler.next_task(endpoint.key) for _ in range(10)]
    assert {task.yaml_id for task in issued} == {"arm_a"}
    next_episode = scheduler.next_task(endpoint.key)
    assert next_episode.yaml_id == "arm_b"
    assert next_episode.bundle_id == "arm_b"
    assert all(task.server == endpoint for task in [*issued, next_episode])


def test_schedule_counts_resident_libraries_and_rejects_impossible_budget(
    episode_files,
):
    """Activation=2 cannot excuse keeping arbitrarily many different PKLs resident."""
    freeze, _, _, _ = _two_arms(episode_files)
    spec = _spec(8801)
    spec["host_budget_gib"] = 2
    for arm in freeze["arms"]:
        arm["artifact"]["file"]["bytes"] = int(0.4 * legacy.GIB)
    schedule = planner.build_schedule(freeze, spec)
    assert len(schedule["waves"]["eval"]) == 2
    assert all(len(w["arms"]) == 1 for w in schedule["waves"]["eval"])
    spec["host_budget_gib"] = 0.5
    with pytest.raises(ValueError, match="no memory-feasible"):
        planner.build_schedule(freeze, spec)


@pytest.mark.parametrize("problem", ["foreign", "duplicate", "missing"])
def test_multi_library_log_requires_the_exact_group(episode_files, problem):
    """Each distinct PKL must be loaded exactly once on its assigned server."""
    freeze, _, _, _ = _two_arms(episode_files)
    lines = [
        f"Loaded {a['artifact']['entries']} entries from {a['artifact']['file']['path']}\n"
        for a in freeze["arms"]
    ]
    concurrent.validate_loads("".join(lines), freeze["arms"])
    if problem == "foreign":
        lines.append("Loaded 10 entries from /another.pkl\n")
    elif problem == "duplicate":
        lines.append(lines[0])
    else:
        lines.pop()
    with pytest.raises(ValueError, match="library"):
        concurrent.validate_loads("".join(lines), freeze["arms"])


@pytest.mark.parametrize("endpoint_count", [1, 2])
def test_mixed_wave_failure_smoke_eval_and_raw_partition_corruption(
    episode_files, monkeypatch, endpoint_count
):
    """Shared and separate servers retain failed waves and bind valid views to raw data."""
    freeze, journal, steps, root = _two_arms(episode_files)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    spec = _spec(port)
    if endpoint_count == 2:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            second_port = listener.getsockname()[1]
        spec["endpoints"].append(
            {"port": second_port, "workers": 2, "stage1_device": "cuda:0"}
        )
    model = {"digest": "synthetic-model"}
    # Node admission is a deployment check; this process test must also run on
    # CI machines smaller than the production 16 GiB free-memory reserve.
    monkeypatch.setattr(
        legacy.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=128 * legacy.GIB, available=100 * legacy.GIB),
    )
    monkeypatch.setattr(legacy, "_model_identity", lambda value: model)
    monkeypatch.setattr(legacy, "gpu_snapshot", lambda pids: [{"owned_bytes": 1024}])
    apool = freeze["memberships"]["libero_spatial"]["apool"]
    monkeypatch.setattr(concurrent, "checked_apool", lambda *args: apool)

    def probe(spec, pool, directory):
        (directory / "client_probe.log").write_text("synthetic environment\n")
        return {
            "num_steps_wait": 10,
            "replan_steps": 5,
            "max_steps": freeze["protocol"]["max_steps"],
            "per_task_digests": pool["per_task_digests"],
        }

    monkeypatch.setattr(legacy, "_client_probe", probe)
    schedule = planner.build_schedule(freeze, spec)
    catalog = root / "concurrent_run"
    catalog.mkdir()
    run = seal(
        {
            "version": concurrent.VERSION,
            "kind": "concurrent_run",
            "freeze_digest": freeze["digest"],
            "spec": spec,
            "schedule": schedule,
            "model": model,
            "implementation": implementation_identity(),
        }
    )
    write_json(catalog / "run.json", run)
    data_path = root / "mixed.json"
    data_path.write_text(
        json.dumps(
            {"journal": journal, "steps": steps, "apool": apool, "arms": freeze["arms"]}
        )
    )
    server_code = """import json,socket,sys,time
data=json.load(open(sys.argv[2]));s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
s.bind(('127.0.0.1',int(sys.argv[1])));s.listen()
for a in data['arms']:
 if a['arm'] in json.loads(sys.argv[3]):print('Loaded 10 entries from '+a['artifact']['file']['path'],flush=True)
time.sleep(120)
"""
    monkeypatch.setattr(
        legacy,
        "_server_command",
        lambda spec, arm: [
            sys.executable,
            "-c",
            server_code,
            str(spec["port"]),
            str(data_path),
            json.dumps(spec["arms"]),
        ],
    )
    monkeypatch.setattr(
        concurrent,
        "_preload",
        lambda server, arms: [
            {
                "arm": a["arm"],
                "yaml": a["yaml"],
                "library": a["artifact"]["file"],
                "reply": {"__ack__": "load_cache_config", "bundle_id": a["arm"]},
            }
            for a in arms
        ],
    )
    failure_flag = root / "fail"
    evaluator_code = """import json,pathlib,subprocess,sys,time
out=pathlib.Path(sys.argv[1]);data=json.load(open(sys.argv[2]));smoke=sys.argv[3]=='smoke'
workers=[subprocess.Popen([sys.executable,'-c','import time;time.sleep(2)','examples.libero.worker_entry',
'--task-suite-name','libero_spatial','--server-key',server['endpoint'],'--init-states-dir',data['apool']['apool_dir']])
 for server in json.loads(sys.argv[4]) for _ in range(server['workers'])]
for filename,key in [('journal.jsonl','journal'),('per_step.jsonl','steps')]:
 rows=[r for r in data[key] if not smoke or r['task_uid'].rsplit(':',1)[1]=='0']
 (out/filename).write_text(''.join(json.dumps(r)+'\\n' for r in rows))
(out/'per_step.jsonl.launch.json').write_text(json.dumps({'suite':'libero_spatial',
'arms':sorted(a['arm'] for a in data['arms']),'trials_per_task':1 if smoke else 50,'smoke':smoke,'apool':data['apool']}))
for p in workers:p.wait()
if pathlib.Path(sys.argv[5]).exists():sys.exit(1)
"""
    monkeypatch.setattr(
        concurrent,
        "_eval_command",
        lambda spec, freeze, plan, directory: [
            sys.executable,
            "-c",
            evaluator_code,
            str(directory),
            str(data_path),
            plan["phase"],
            json.dumps(plan["servers"], sort_keys=True),
            str(failure_flag),
        ],
    )
    with pytest.raises(ValueError, match="incomplete concurrent phase"):
        concurrent.run_wave(freeze, run, schedule["waves"]["eval"][0], catalog)
    failure_flag.touch()
    failed = concurrent.run_wave(freeze, run, schedule["waves"]["smoke"][0], catalog)
    assert failed["status"] == "invalid" and failed["owned_processes_exited"]
    assert all(not legacy._alive(p) for p in failed["owned_processes"])
    failure_flag.unlink()
    smoke = concurrent.run_wave(freeze, run, schedule["waves"]["smoke"][0], catalog)
    assert smoke["status"] == "valid", smoke
    assert (
        len(concurrent.accepted_concurrent_batches(catalog, freeze, phase="smoke")) == 2
    )
    formal = concurrent.run_wave(freeze, run, schedule["waves"]["eval"][0], catalog)
    assert formal["status"] == "valid", formal
    accepted = legacy.accepted_batches(catalog, freeze, require_complete=True)
    assert len(accepted) == 2
    assert all(
        v["launch"]["latency_scope"] == "concurrent_call" for v in accepted.values()
    )
    for record in accepted.values():
        resource = read_json(Path(record["directory"]) / "node_postflight.json")
        assert (
            resource["peak_server_rss"] is None
            and resource["shared_server_peak_rss"] > 0
        )
    with pytest.raises(ValueError, match="already completed"):
        concurrent.run_wave(freeze, run, schedule["waves"]["eval"][0], catalog)
    wave_dir = Path(next(iter(accepted.values()))["launch"]["wave_directory"])
    completion_path = wave_dir / "completion.json"
    completion = read_json(completion_path)
    name = next(iter(completion["views"]))
    view = completion["views"][name]
    path = Path(view["journal"]["path"])
    path.write_bytes(
        path.read_bytes().replace(b'"duration_s": 1.2', b'"duration_s": 1.3')
    )
    view["journal"] = file_identity(path)
    completion_path.write_text(json.dumps(seal(completion)))
    with pytest.raises(ValueError, match="omits or alters raw attempts"):
        concurrent.validate_wave(wave_dir, freeze, run)


def test_partition_preserves_stale_attempts_and_rejects_truncation(tmp_path):
    """A view must retain stale attempts even though outcome selection later excludes them."""
    source = tmp_path / "raw.jsonl"
    rows = [
        {"yaml_id": n, "task_uid": n + ":eval:0:0", "attempt": attempt}
        for n, attempt in [("a", 1), ("b", 1), ("a", 2)]
    ]
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    paths = {n: tmp_path / (n + ".jsonl") for n in ("a", "b")}
    hashes = planner.partition_rows(source, paths, list(paths))
    assert hashes == planner.partition_rows(source, None, list(paths))
    assert len(paths["a"].read_text().splitlines()) == 2
    source.write_bytes(source.read_bytes().rstrip())
    with pytest.raises(ValueError, match="truncated"):
        planner.partition_rows(source, None, list(paths))
