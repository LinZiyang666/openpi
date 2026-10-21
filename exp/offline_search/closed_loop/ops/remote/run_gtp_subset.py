"""Exact manifest or legacy Cartesian episode selection; importable without run_gtp.

OSCL_MANIFEST/--manifest accepts C's {strata, selected:[{task, init, ...}]} or a
list of [task_id, episode_idx] pairs / dictionaries. OSCL_EPISODES and OSCL_TASKS
retain their old meaning when no manifest is supplied. Full pool attestation stays 50.
"""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def pair(row):
    if isinstance(row, dict):
        t = row.get("task_id", row.get("task"))
        e = row.get("episode_idx", row.get("init"))
    else:
        t, e = row
    if isinstance(t, bool) or isinstance(e, bool) or int(t) != t or int(e) != e:
        raise ValueError(f"manifest pair must contain integer IDs: {row!r}")
    t, e = int(t), int(e)
    if not 0 <= t < 10 or not 0 <= e < 50:
        raise ValueError(f"manifest pair outside the 10-task / 50-init pool: {(t, e)}")
    return t, e


def load_manifest(path):
    data = json.loads(Path(path).read_text())
    rows = data if isinstance(data, list) else data.get("selected", data.get("pairs"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest must have a nonempty selected/pairs list")
    selected = {}
    for r in rows:
        p = pair(r)
        value = dict(r) if isinstance(r, dict) else {"task_id": p[0], "episode_idx": p[1]}
        if p in selected and selected[p] != value:
            raise ValueError(f"conflicting duplicate manifest pair {p}")
        selected[p] = value
    canonical = json.dumps(sorted(selected), separators=(",", ":"))
    return {"data": data if isinstance(data, dict) else {}, "selected": selected,
            "sha256": hashlib.sha256(canonical.encode()).hexdigest(), "path": str(Path(path).resolve())}


def uid_pair(uid):
    try:
        return tuple(map(int, uid.rsplit(":", 2)[-2:])) if uid.count(":") >= 2 else None
    except (TypeError, ValueError, AttributeError):
        return None


def check_manifest(manifest, model=None, suite=None):
    aliases = {"libero_spatial": "spatial", "sp": "spatial", "libero_10": "l10"}
    data = manifest["data"]
    if model and data.get("model") and model != data["model"]:
        raise ValueError(f"manifest model {data['model']} != arm model {model}")
    if suite and data.get("suite") and aliases.get(suite, suite) != aliases.get(data["suite"], data["suite"]):
        raise ValueError(f"manifest suite {data['suite']} != arm suite {suite}")


def selected_tasks(tasks, manifest=None, keep_ep=None, keep_t=None):
    if manifest is not None:
        return [t for t in tasks if (t.task_id, t.episode_idx) in manifest["selected"]]
    return [t for t in tasks if (keep_ep is None or t.episode_idx in keep_ep)
            and (not keep_t or t.task_id in keep_t)]


class AgentOnly:
    """The driver's stand-in on another fleet of the arm: run() returns once that driver reports the arm done
    or has been unreachable for 10 minutes (supervisor pulls refresh AUTOSCALE_CONTROL.json every 15 s)."""
    per_step_rows = ()

    def __init__(self, port):
        self.port = port
        self.control = Path(os.environ["OPENPI_AUTOSCALE_CONFIG"]).parent / "AUTOSCALE_CONTROL.json"
        self.control.unlink(missing_ok=True)  # a previous attempt's "done" must not end or stall this one

    def run(self):
        started = time.time()
        while True:
            time.sleep(5)
            try:
                if json.loads(self.control.read_text()).get("done"):
                    return
                last = self.control.stat().st_mtime
            except (OSError, ValueError):
                last = started
            if time.time() - last > 600:
                return


def multinode(run_gtp):
    """One arm, agents on several fleets (closed_loop/ops/h100/fleet.py). OSCL_DRIVER_PORT: the driver listens on
    0.0.0.0:<port> (its own agent still uses 127.0.0.1). OSCL_AGENT_OF=<host>:<port>: no driver here, only this
    fleet's agent, connected to that driver, worker ids prefixed with WORKER_HOST."""
    port, agent_of = os.environ.get("OSCL_DRIVER_PORT"), os.environ.get("OSCL_AGENT_OF")
    if port:
        driver = run_gtp.ConductorDriver
        run_gtp.ConductorDriver = lambda *a, **k: driver(*a, **dict(k, bind_host="0.0.0.0", bind_port=int(port)))
    if agent_of:
        host, _, port = agent_of.rpartition(":")
        agent, prefix = run_gtp.WorkerAgent, os.environ["WORKER_HOST"] + "_"
        run_gtp.ConductorDriver = lambda *a, **k: AgentOnly(int(port))
        run_gtp.WorkerAgent = lambda specs, driver_host, driver_port: agent(
            [dataclasses.replace(s, worker_id=prefix + s.worker_id) for s in specs], host, int(port))
        print(f"AGENT_ONLY driver={host}:{port}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--manifest", default=os.environ.get("OSCL_MANIFEST"))
    args, rest = ap.parse_known_args(sys.argv[1:] if argv is None else argv)
    manifest = load_manifest(args.manifest) if args.manifest else None
    if manifest and "--task-suite" in rest:
        check_manifest(manifest, suite=rest[rest.index("--task-suite") + 1])
    keep_ep = {int(x) for x in os.environ.get("OSCL_EPISODES", "").split(",") if x.strip()}
    keep_t = {int(x) for x in os.environ.get("OSCL_TASKS", "").split(",") if x.strip()}
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this island (<island>/os_cl/run_gtp_subset.py)
    from exp.gate_threshold_pareto import run_gtp
    from exp.offline_search.closed_loop.worker_pool import install
    install(run_gtp, rest, manifest)
    multinode(run_gtp)
    original = run_gtp.SweepStrategy._episodes

    def episodes(self, yaml_id, server):
        return selected_tasks(original(self, yaml_id, server), manifest, keep_ep or None, keep_t)

    run_gtp.SweepStrategy._episodes = episodes
    # The base driver's whole-arm resume count is not exact-pair aware.
    if manifest and "--no-resume-filter" not in rest:
        rest.append("--no-resume-filter")
    sys.argv = [sys.argv[0], *rest]
    run_gtp.main()


if __name__ == "__main__":
    main()
