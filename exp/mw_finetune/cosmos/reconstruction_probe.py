"""Four immutable reconstruction arms, concurrent A servers, a single 20-min GPU lock.

Default/libero/chunk32 arms use A.evaluate unchanged. Only corner3/256 uses the
local render/horizon shim, with A's episode function, server and model wrapper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


CONFIGS = [
    {"name": "01_default", "layout": "single", "chunk_size": 16, "corner3": False},
    {"name": "02_libero_duplicate", "layout": "libero_duplicate", "chunk_size": 16, "corner3": False},
    {"name": "03_flowdagger_view_300", "layout": "single", "chunk_size": 16, "corner3": True},
    {"name": "04_chunk32", "layout": "single", "chunk_size": 32, "corner3": False},
]


def stop_owned(process):
    if process is None:
        return
    # Every child starts its own session: this group contains only our child
    # and its spawned evaluator workers, never another agent's processes.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=3)


def evaluation_command(config, stage, root, sim_python, workers):
    task, trials = ("assembly-v3", 10) if stage == "assembly" else ("reach-v3", 5)
    args = [sim_python, "-m"]
    if config["corner3"]:
        args += ["exp.mw_finetune.cosmos.probe_evaluate", "--task", task,
                 "--max-steps", "300" if stage == "assembly" else "160"]
    else:
        args += ["exp.mw_finetune.evaluate", "--tasks", task]
    return args + ["--run", str(root / config["name"] / stage), "--model-id", "cosmos_" + config["name"],
                   "--port", str(config["port"]), "--workers", str(workers), "--trials", str(trials)]


def summarize(root, runtime):
    report = {"runtime": runtime, "configs": []}
    for config in CONFIGS:
        directory = root / config["name"]
        arm = {**config, "tasks": {}, "cuda_graph_parity": None}
        parity = directory / "graph_parity.json"
        if parity.exists():
            arm["cuda_graph_parity"] = json.loads(parity.read_text())
        query_file = directory / "queries.jsonl"
        queries = [json.loads(line) for line in query_file.read_text().splitlines()] if query_file.exists() else []
        for stage in ("assembly", "reach"):
            path = directory / stage / "episodes.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
            good = [r for r in rows if not r.get("error")]
            task = "assembly-v3" if stage == "assembly" else "reach-v3"
            q = [row for row in queries if row["task"] == task]
            summaries = [r["per_step_rows"][-1] for r in good]
            decisions = sum(r["n_decisions"] for r in summaries)
            arm["tasks"][task] = {"completed": len(good), "successes": sum(r["success"] for r in good),
                "errors": len(rows) - len(good), "n_queries": decisions,
                "first_160_successes": sum(r["success"] and r["n_steps"] <= 160 for r in good),
                "mean_roundtrip_ms": 1000 * sum(r["infer_s"] for r in summaries) / decisions if decisions else None,
                "mean_service_ms": sum(r["service_ms"] for r in q) / len(q) if q else None,
                "episodes": [{k: r[k] for k in ("idx", "success", "n_steps", "duration_s")} for r in good]}
        report["configs"].append(arm)
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--checkpoint", default="/data/mw_finetune/models/cosmos_metaworld")
    ap.add_argument("--sim-python", default="/home/weiland/projects/openpi_ext/envs/metaworld_sim/bin/python")
    ap.add_argument("--workers", type=int, choices=(4, 5, 6), default=5)
    ap.add_argument("--base-port", type=int, default=8064)
    ap.add_argument("--seconds", type=int, default=1170)
    ap.add_argument("--only-flowdagger", action="store_true", help="corrected standalone 300-step arm within remaining GPU budget")
    ap.add_argument("--assembly-only", action="store_true", help="omit reach when repeating only the assembly horizon")
    a = ap.parse_args()
    if not 1 <= a.seconds <= 1170:
        ap.error("seconds must be <=1170, leaving cleanup time inside 20 minutes")
    if a.only_flowdagger:
        CONFIGS[:] = [{"name": "03_flowdagger_view_300", "layout": "single", "chunk_size": 16, "corner3": True}]
    a.run.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + a.seconds
    memory = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    required_mib = 30000 if len(CONFIGS) > 1 else 10000
    if int(memory.splitlines()[0]) < required_mib:
        raise RuntimeError(f"need >={required_mib} MiB free, found {memory.strip()} MiB")
    runtime = {"gpu_free_mib_before_start": int(memory.splitlines()[0]), "budget_s": a.seconds,
               "workers_per_config": a.workers, "concurrent_A_servers": len(CONFIGS), "denoising_steps": 5,
               "pure_inference": True, "assembly_only": a.assembly_only, "configs": []}
    for index, config in enumerate(CONFIGS):
        config["port"] = a.base_port + index
        with socket.socket() as check:
            check.bind(("127.0.0.1", config["port"]))
    source_dir = Path(__file__).parent
    runtime["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [source_dir / "policy.py", source_dir / "probe_server.py", source_dir / "probe_evaluate.py", Path(__file__)]}
    snapshot = a.run / "source_snapshot"
    snapshot.mkdir()
    for name in runtime["source_sha256"]:
        (snapshot / name).write_bytes((source_dir / name).read_bytes())
    (a.run / "plan.json").write_text(json.dumps({"runtime": runtime, "configs": CONFIGS}, indent=2) + "\n")
    cpu_env = dict(os.environ)
    cpu_env.update(MUJOCO_GL="osmesa", LIBGL_ALWAYS_SOFTWARE="1", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", LP_NUM_THREADS="1")
    cpu_env["LD_LIBRARY_PATH"] = "/data/mw_finetune/envs/osmesa/usr/lib/x86_64-linux-gnu:" + cpu_env.get("LD_LIBRARY_PATH", "")
    contexts = []

    def interrupted(signum, frame):
        raise KeyboardInterrupt("probe interrupted; clean up only owned children")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for config in CONFIGS:
            directory = a.run / config["name"]
            directory.mkdir()
            log = (directory / "server.log").open("x")
            cmd = [sys.executable, "-m", "exp.mw_finetune.cosmos.probe_server", "--parity-file", str(directory / "graph_parity.json"),
                   "--model", "cosmos", "--checkpoint", a.checkpoint, "--port", str(config["port"]),
                   "--host", "127.0.0.1", "--denoising-steps", "5", "--policy-option", "camera_layout=" + json.dumps(config["layout"]),
                   "--policy-option", "chunk_size=" + str(config["chunk_size"])]
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            contexts.append({"config": config, "server": process, "server_log": log, "eval": None, "eval_log": None,
                             "stage": "startup", "server_command": cmd})
        last_update = 0
        while any(c["stage"] != "done" for c in contexts):
            if time.monotonic() >= deadline:
                runtime["error"] = "20-minute budget reached before all requested episodes completed"
                break
            for context in contexts:
                stage, config = context["stage"], context["config"]
                if stage == "done":
                    continue
                if context["server"].poll() is not None:
                    context["error"] = f"server exited {context['server'].returncode}"
                    stop_owned(context["eval"])
                    context["stage"] = "done"
                    continue
                next_stage = None
                if stage == "startup":
                    try:
                        with socket.create_connection(("127.0.0.1", config["port"]), timeout=.1):
                            next_stage = "assembly"
                    except OSError:
                        continue
                elif context["eval"].poll() is not None:
                    status = context["eval"].returncode
                    context["eval_log"].close()
                    if status:
                        context["error"] = f"{stage} evaluator exited {status}"
                        next_stage = "done"
                    else:
                        print(f"PROBE {config['name']} {stage} complete", flush=True)
                        next_stage = "reach" if stage == "assembly" and not a.assembly_only else "done"
                if next_stage == "done":
                    stop_owned(context["server"])
                    context["stage"] = "done"
                elif next_stage:
                    log = (a.run / config["name"] / (next_stage + ".log")).open("x")
                    cmd = evaluation_command(config, next_stage, a.run, a.sim_python, a.workers)
                    context["eval"] = subprocess.Popen(cmd, env=cpu_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    context["eval_log"] = log
                    context["stage"] = next_stage
            if time.monotonic() - last_update >= 30:
                print(json.dumps({"elapsed_s": round(time.monotonic() - started, 1),
                                  "stages": {c["config"]["name"]: c["stage"] for c in contexts}}), flush=True)
                last_update = time.monotonic()
            time.sleep(.2)
    finally:
        for context in contexts:
            stop_owned(context["eval"])
            stop_owned(context["server"])
            for key in ("server_log", "eval_log"):
                if context[key] is not None:
                    context[key].close()
        runtime["wall_s"] = time.monotonic() - started
        runtime["configs"] = [{"name": c["config"]["name"], "stage": c["stage"], "error": c.get("error")} for c in contexts]
        report = summarize(a.run, runtime)
        print(json.dumps({"wall_s": runtime["wall_s"], "counts": {c["name"]: {t: (v["successes"], v["completed"])
                                                                                for t, v in c["tasks"].items()} for c in report["configs"]}}), flush=True)
    if runtime.get("error") or any(c.get("error") for c in contexts):
        raise SystemExit("probe incomplete; see report.json")


if __name__ == "__main__":
    main()
