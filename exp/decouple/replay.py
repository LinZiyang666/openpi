"""B-pool replay with real frozen R16 methods and model-stage stubs.

Reuses the plugin's existing CPU tape adapter for its blind/tail contract.
Only stage execution is replaced; retrieval, correction, histories, decisions
and JSONL logging run through the actual plugin implementation.
"""
from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from exp.decouple.recorded import ROOT, load, observation
from exp.offline_search.closed_loop import plugin
from exp.offline_search.debug.tests.test_server_parity import FakePolicy, runtime
from exp.offline_search.harness import api, store
from openpi.serving.batching_core import Stage3MissPayload


def frozen_runtime(model, directory, suite="spatial"):
    path = Path(f"/home/weiland/trace_runs/os_closed_loop/r16_conductor_dev_{model}_k1_v1/group.json")
    group = json.loads(path.read_text())
    row = next(e["row"] for e in group["entries"]
               if suite in e["row"]["suite"] and "100" in e["row"]["arm"])
    args = row["plugin_args"]
    fit = Path(args[args.index("--os-fit-artifact") + 1])
    with fit.open("rb") as stream:
        blob = pickle.load(stream)
    rt = runtime(blob["method"], blob, Path(directory), "CU", model)
    # Load only immutable library arrays (all mmap_mode='r') and the frozen
    # artifact's registered actions. Never refit, write or alter a trace file.
    for name in store.library_names(ROOT, f"{model}_{suite}"):
        library = store.LibraryView(ROOT, f"{model}_{suite}", name)
        rt.tables[name] = library.action
        rt.lib_sizes[name] = library.L
    for name, value in blob.get("registered", {}).items():
        rt.tables[name] = value["action"]
        rt.lib_sizes[name] = len(value["action"])
    rt.opts.os_fit_artifact = str(fit)
    return rt, fit


def queries(data):
    a = data.arrays
    return SimpleNamespace(rs=a["rs"], key_v0=a["key_v0"], key_v1=a["key_v1"])


class ReplayPolicy(FakePolicy):
    """Real method/decision CPU path, with stage stubs for MISS and encoder."""
    def __init__(self, session, data, coordinator, *, k=1):
        super().__init__(session, queries(data))
        self.data, self.coordinator, self.k = data, coordinator, k

    def stage1(self, obs):
        super().stage1(obs)
        state = np.zeros(32, np.float32)
        state[:self.queries.rs.shape[1]] = self.queries.rs[obs["_row"]]
        self.stage1_out = self.coordinator.submit_to_stage(1, "replay", {"state": state})

    def on_task_begin(self):
        pass

    def on_task_end(self):
        pass

    def infer(self, obs):
        self.stage1(obs)
        row, session = obs["_row"], self.session
        state = np.array(self.queries.rs[row], copy=True)
        ctx = SimpleNamespace(task_key=obs["prompt"], current_step=self.orch._real.state.step_counter,
                              query_keys=dict(vision_0=np.array(self.queries.key_v0[row]),
                                              vision_1=np.array(self.queries.key_v1[row]), robot_state=state),
                              checkpoint_id="CP1")
        session.on_search(ctx)
        hit = session._dec.get("hit", True)
        if hit:
            action = session._dec["served"]
        else:
            self.policy_calls += 1
            noise = self.rng.standard_normal((10, 32)).astype(np.float32)
            stage2 = self.coordinator.submit_to_stage(2, "replay", self.stage1_out)
            out = self.coordinator.submit_to_stage(3, "replay", Stage3MissPayload(stage2, torch.from_numpy(noise), self.k))
            action = out.action_chunk[0].numpy()
            # The two model tapes keep their original horizon; GR00T's stage
            # harness is tested separately with the real staged runner types.
            if session.rt.H != 10:
                action = np.tile(action, (2, 1))[:session.rt.H]
        self.orch._real.state.step_counter += 1
        self.orch._real.state.state_history.append(state)
        self.orch.broadcast_action(action)
        return {**self._osp_blind_output(action, state), "diag": session.wire_diag(),
                "hit_type": "FULL_HIT" if hit else "MISS"}


def make_connections(rt, data, coordinator, *, k=1, count=2):
    connections = []
    for i in range(count):
        session = plugin.PluginSession(rt, None, None, None, "bundle_old")
        inner = ReplayPolicy(session, data, coordinator, k=k)
        conn = plugin._ConnPolicy(inner, [session], "bundle_old")
        connections.append((conn, session, inner))
    return connections


def tape(data, *, steps=12, count=2):
    # Exercise identical inputs across two interleaved connections; consecutive
    # rows make histories available to the real method's blind_step.
    tid = int(data.arrays["task_id"][int(data.rows[0])])
    same = np.flatnonzero(data.arrays["task_id"] == tid)[:steps]
    task = data.manifest["tasks"][str(tid)]
    start = dict(task=task, extra_metadata=dict(task_uid="decouple:B:0", attempt=1,
                                              task_id=tid, orig_init_state_idx=0))
    for seq, row in enumerate(same):
        for connection in range(count):
            obs = observation(data, 0)
            obs["observation/state"] = np.asarray(data.arrays["rs"][row, :8], np.float64)
            obs.update(prompt=task, _row=int(row),
                       __extra__=dict(decision_id=seq, executed_steps=5, exec_ok=True))
            yield connection, start, obs
