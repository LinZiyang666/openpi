"""R17 queue shim; transport, lifecycle, journal, census stay in stock driver."""
import argparse
import copy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import signal
import threading
import time
from openpi.conductor.driver import ConductorDriver
from .queue import CompatibleEpisodeQueue


class MetaWorldConductorDriver(ConductorDriver):
    def __init__(self, strategy, *, dispatch_path, worker_probe_validator=None, **kwargs):
        super().__init__(strategy, **kwargs)
        self._worker_probe_validator = worker_probe_validator
        if worker_probe_validator is not None:
            # Admission and the stock census publication must be atomic. The
            # stock recorder reacquires this lock; no transport override needed.
            self._census_lock = threading.RLock()
        self.rejected_worker_probes = []
        self._scheduler = CompatibleEpisodeQueue(self._graph, endpoints=strategy.endpoints,
                                                dispatch_path=dispatch_path,
                                                **(kwargs.get("scheduler_kwargs") or {}))
        self._scheduler.mark_preexisting_done(self._journal.replay_done_uids())
        if hasattr(strategy, "bind_driver"):
            strategy.bind_driver(self)
        strategy.preload(self._ctl)

    def _record_pull(self, payload, peer):
        # Stock census is observe-only. The explicit R17 launcher enables a
        # validator before the stock transport can dispatch an episode.
        if self._worker_probe_validator is None:
            return super()._record_pull(payload, peer)
        with self._census_lock:
            entry = self._census.get(payload.get("worker_id"), {})
            previous = entry.get("probe")
            peer_text = f"{peer[0]}:{peer[1]}" if isinstance(peer, tuple) and len(peer) >= 2 else None
            try:
                if previous is not None and entry.get("server_key") != payload.get("server_host"):
                    raise ValueError("worker endpoint changed during the campaign")
                if "probe" in payload:
                    probe = payload["probe"]
                else:
                    # Stock WorkerLoop sends a probe only on a connection's
                    # first pull. A worker id alone cannot attest a new peer.
                    if previous is None or peer_text is None or peer_text not in entry.get("peers", []):
                        raise ValueError("worker first pull on this connection lacks probe")
                    probe = previous
                if not isinstance(probe, dict) or not probe:
                    raise ValueError("worker probe must be a nonempty mapping")
                self._worker_probe_validator(probe)
                if "probe" in payload and previous is not None and previous != probe:
                    raise ValueError("worker probe changed during the campaign")
            except Exception as exc:
                self.rejected_worker_probes.append({"worker_id": payload.get("worker_id"),
                    "server_key": payload.get("server_host"), "reason": str(exc), "time": time.time(),
                    "probe_present": "probe" in payload, "probe": copy.deepcopy(payload.get("probe")),
                    "accepted_probe": copy.deepcopy(previous), "pull_payload": copy.deepcopy(payload),
                    "peer": list(peer) if isinstance(peer, tuple) else copy.deepcopy(peer)})
                raise
            # Own accepted probe data as well as rejected evidence. This also
            # preserves the exact wire distinction between absent and empty.
            return super()._record_pull(copy.deepcopy(payload), peer)

    def handle_pull(self, server_key):
        message, payload = super().handle_pull(server_key)
        if "task" in payload:
            from openpi.conductor.protocol import task_from_wire
            task = task_from_wire(payload["task"])
            try:
                if hasattr(self._strategy, "on_dispatch"):
                    self._strategy.on_dispatch(task)
                self._strategy.ensure_bundle(task.server, task.bundle_id)
            except Exception as exc:
                from openpi.conductor.protocol import result_to_wire
                from openpi.conductor.task import EpisodeResult
                # Use the same retry classification, durable terminal journal,
                # accepted-attempt fence and strategy notification as worker errors.
                self.handle_result(result_to_wire(EpisodeResult(task.task_uid, False, 0, repr(exc),
                                                                attempt=task.attempt)))
                raise
        return message, payload

    def handle_result(self, payload):
        # Stock driver only journals non-retriable errors. At the retry limit
        # make the live attempt terminal explicitly, retaining its real error.
        uid = payload.get("task_uid")
        q = self._scheduler
        if (payload.get("error") and q._ep_attempts.get(uid, 0) >= q._max_ep_retries
                and payload.get("attempt") == q._dispatch_gen.get(uid)):
            payload = dict(payload, error="FatalEpisodeError: retry budget exhausted; " + payload["error"])
        with q._lock:
            entry = q._ep_index.get(uid)
            accepted = (entry is not None and uid in q._rt[entry[0]].dispatched
                        and payload.get("attempt") == q._dispatch_gen.get(uid))
            super().handle_result(payload)
            if entry is not None and hasattr(self._strategy, "observe_result"):
                from openpi.conductor.protocol import result_from_wire
                self._strategy.observe_result(entry[1], result_from_wire(payload), accepted)

    def close(self):
        for ctl in self._ctls.values():
            ctl.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True, help="immutable manifest containing r17_model_spec/model_spec, or π content manifest")
    ap.add_argument("--endpoint", required=True, help="host:port; qualified owned inference endpoint")
    ap.add_argument("--full-yaml", type=Path)
    ap.add_argument("--k2-yaml", type=Path)
    ap.add_argument("--arm", action="append", default=[], help="name:YAML:K; repeat for model-neutral bundles")
    ap.add_argument("--benchmark-spec", type=Path, help="sealed explicit LIBERO benchmark profile JSON")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--campaign", default="pi05_mw_mechanical")
    ap.add_argument("--smoke", action="store_true", help="one first-task episode per arm, alternating original idx0/1; not SR")
    ap.add_argument("--plan-only", action="store_true", help="CPU manifest only; no clients or environments")
    ap.add_argument("--bind-host", default="127.0.0.1")
    ap.add_argument("--bind-port", type=int, default=0)
    ap.add_argument("--deadline-seconds", type=int, default=540)
    a = ap.parse_args(argv)
    if not a.smoke and not a.plan_only:
        ap.error("live launch requires --smoke; formal evaluation uses coordinator-qualified frozen strategy")
    from .manifest import EndpointSpec, episode_tasks, validate_worker_probe
    from .strategy import MetaWorldStrategy
    from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
    from openpi.conductor.task import ServerEndpoint
    m = json.loads(a.manifest.read_text())
    from exp.r17_integration.common.contracts import digest
    from exp.r17_integration.common.recipe import bundle_identity, recipe_identity
    from .manifest import load_benchmark_spec, load_plan_config
    benchmark = load_benchmark_spec(a.benchmark_spec) if a.benchmark_spec else FORMAL_SPEC
    model = m.get("r17_model_spec", m.get("model_spec", m.get("spec", m if "layout" in m else None)))
    if model is None:
        from exp.r17_integration.models.pi05.adapter import model_spec
        model = asdict(model_spec(m["checkpoint_identity"], m["norm_identity"], code_digest=m["code_digest"]))
    if benchmark.identity not in model["allowed_benchmark_protocols"]:
        raise ValueError("model does not bind selected benchmark protocol")
    host, port = a.endpoint.rsplit(":", 1)
    endpoint = ServerEndpoint(host, int(port))
    arms = []
    if a.arm:
        for raw in a.arm:
            name, path, k = raw.rsplit(":", 2)
            arms.append((name, Path(path), int(k)))
    else:
        if not a.full_yaml or not a.k2_yaml:
            ap.error("provide --arm name:YAML:K or both --full-yaml/--k2-yaml")
        arms = [("full", a.full_yaml, 10), ("retrieval", a.k2_yaml, 2)]
    if len({arm for arm, _, _ in arms}) != len(arms):
        raise ValueError("arm names must be unique")
    tasks, bundles, identities = [], {}, []
    for index, (arm, path, k) in enumerate(arms):
        config = load_plan_config(path)
        binding = config.adapter_binding
        identity = bundle_identity(config)
        if (binding.get("schema") != "r17.adapter_binding.v1" or binding.get("model_id") != model["model_id"]
                or binding.get("feature_identity") != digest(model["feature_spec"])
                or binding["checkpoint_identity"] != model["checkpoint_identity"] or binding["norm_identity"] != model["norm_identity"]
                or binding["layout_identity"] != digest(model["layout"]) or binding["benchmark_protocol"] != benchmark.identity
                or binding["num_steps"] != k):
            raise ValueError("arm YAML/model/benchmark/K mismatch")
        from exp.r17_integration.common.contracts import ActionStateLayout
        layout = ActionStateLayout(**model["layout"])
        layout.validate_commit(binding["commit_horizon"], allow_partial=binding.get("allow_partial_commit", False))
        if binding.get("allow_partial_commit", False) and benchmark.benchmark != "libero":
            raise ValueError("partial-tail negotiation is only implemented by LIBERO runner")
        bundle = arm + "_" + identity[:16]
        identities.append(identity)
        bundles[bundle] = path
        episodes = episode_tasks(campaign=a.campaign, arm=arm, bundle_id=bundle, endpoint=endpoint,
                     model_identity=model["checkpoint_identity"], norm_identity=model["norm_identity"],
                     layout_identity=digest(model["layout"]), num_steps=k, partition="smoke" if a.smoke else "test",
                     names=["reach-v3" if benchmark.benchmark != "libero" else benchmark.tasks[0][0]] if a.smoke else None, indices=[index % 2] if a.smoke else None,
                     spec=benchmark, yaml_identity=__import__("hashlib").sha256(path.read_bytes()).hexdigest(),
                     bundle_generation=bundle, bundle_identity=identity, recipe_identity=recipe_identity(config),
                     commit_horizon=binding["commit_horizon"])
        study = {name: binding[name] for name in ("allow_lossy_sampler", "sampler_study_label") if name in binding}
        if study:
            episodes = [replace(ep, extra={**ep.extra, **study, "sampler_quality_status": "lossy_study"}) for ep in episodes]
        tasks.extend(episodes)
    endpoints = [EndpointSpec(endpoint, model["checkpoint_identity"], model["norm_identity"], digest(model["layout"]),
                              benchmark.identity, allowed_k=tuple(k for _, _, k in arms), bundle_identities=tuple(identities))]
    a.output.mkdir(parents=True, exist_ok=True)
    plan = {"schema": "r17.episode_plan.v1", "benchmark": benchmark.manifest(),
            "model": model, "expected_episodes": len(tasks), "smoke": a.smoke,
            "quality_sr_claim": False if a.smoke else "requires complete qualified formal protocol",
            "tasks": [asdict(t) for t in tasks]}
    path = a.output / "episode_manifest.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(plan)):
        raise ValueError("resume episode manifest differs")
    path.write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n")
    if a.plan_only:
        return
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    ctl_factory = lambda s: WebsocketClientPolicy(s.host, s.port)
    client = ctl_factory(endpoint)
    try:
        if client.get_server_metadata().get("r17_model_spec", {}).get("checkpoint_identity") != model["checkpoint_identity"]:
            pass
    finally:
        client.close()
    strategy = MetaWorldStrategy(tasks, endpoints=endpoints,
                                  bundles=bundles)
    def write_rows(stage, rows):
        with (a.output / "per_step.jsonl").open("a") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
    driver = MetaWorldConductorDriver(strategy, dispatch_path=a.output / "dispatch.jsonl",
             yaml_weights={t.yaml_id: sum(e.yaml_id == t.yaml_id for e in tasks) for t in tasks}, servers=[endpoint],
             journal_path=str(a.output / "journal.jsonl"), ctl_factory=ctl_factory,
             bind_host=a.bind_host, bind_port=a.bind_port, per_step_writer=write_rows,
             worker_probe_validator=lambda probe: validate_worker_probe(probe, spec=benchmark))
    stopped = False
    def stop(*args):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    deadline = time.monotonic() + a.deadline_seconds
    try:
        driver.run(stop=lambda: stopped or time.monotonic() > deadline)
    finally:
        (a.output / "census.json").write_text(json.dumps(driver.worker_census, indent=2) + "\n")
        (a.output / "rejected_worker_probes.json").write_text(json.dumps(driver.rejected_worker_probes, indent=2) + "\n")
        driver.close()
    if not driver.scheduler.all_done():
        raise SystemExit("partial: driver deadline or interruption; accepted evidence retained")


if __name__ == "__main__":
    main()

BenchmarkConductorDriver = MetaWorldConductorDriver
