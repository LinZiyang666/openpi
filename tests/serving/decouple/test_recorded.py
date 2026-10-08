import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from openpi_client import msgpack_numpy

from exp.decouple.recorded import load
from exp.decouple.replay import frozen_runtime, make_connections, tape
from exp.offline_search.closed_loop import plugin
from openpi.cache.components.key_builder import CP1SpatialPool16KeyBuilder
from openpi.cache.groot.key_builder import GrootLiberoCP1SpatialPool16KeyBuilder
from openpi.cache.groot.staged import GrootStage1Output
from openpi.cache.types import CheckpointID
from openpi.models_pytorch.pi0_pytorch import Stage1Output
from openpi.serving.batching_coordinator import BatchingCoordinator
from openpi.serving.split_runtime import DecisionPolicy
from openpi.serving.split_service import RemoteCoordinator, RemoteKeyBuilder, StageService
from tests.serving.decouple.stubs import PiModel
from tests.serving.decouple.test_stages import Harness


@pytest.mark.parametrize("model", ["pi05", "groot"])
def test_recorded_gpu_key_reduction_transport_exact(model):
    data = load(model, limit=3)
    harness = Harness(StageService(None))
    try:
        for index in range(3):
            row = int(data.rows[index])
            tokens = torch.cat([torch.from_numpy(np.array(data.token[name][index])) for name in ("v0", "v1")])
            state = torch.from_numpy(np.array(data.arrays["rs"][row]))
            if model == "pi05":
                tokens = torch.cat((tokens, torch.zeros_like(tokens[:256])))
                stage = Stage1Output(state[None], tokens[None], torch.ones(1, len(tokens), dtype=torch.bool), None, None)
                builder = CP1SpatialPool16KeyBuilder(enabled_fields=["vision_0", "vision_1", "robot_state"])
            else:
                stage = GrootStage1Output(tokens[None], torch.ones(1, len(tokens)), torch.ones(1, len(tokens), dtype=torch.bool),
                                         {"state": state[None, None], "state_mask": torch.ones(1, 1, len(state), dtype=torch.bool)})
                # Insert text gap to make the two image runs distinct.
                stage.input_embeds = torch.cat((tokens[:256], torch.zeros(1, tokens.shape[1]), tokens[256:]))[None]
                stage.image_token_mask = torch.cat((torch.ones(256), torch.zeros(1), torch.ones(256))).bool()[None]
                builder = GrootLiberoCP1SpatialPool16KeyBuilder(enabled_fields=["vision_0", "vision_1", "robot_state"])
            remote = RemoteKeyBuilder(builder, harness.client)
            builder.collect(CheckpointID.CP1, stage1=stage)
            expected = builder.build(CheckpointID.CP1)
            remote.collect(CheckpointID.CP1, stage1=stage)
            actual = remote.build(CheckpointID.CP1)
            for field in expected:
                assert torch.equal(actual[field], expected[field])
            remote.clear()
            remote.close()
    finally:
        harness.close()


@pytest.mark.parametrize("model", ["pi05", "groot"])
def test_frozen_b_pool_method_corrector_blind_tail_replay_exact(tmp_path, model):
    data = load(model, limit=2)
    rt, fit = frozen_runtime(model, tmp_path / "direct")
    split_rt, _ = frozen_runtime(model, tmp_path / "split")
    pi_model = PiModel()
    direct = BatchingCoordinator(pi_model, max_batch_size=1, max_wait_ms=0)
    core = BatchingCoordinator(pi_model, max_batch_size=1, max_wait_ms=0)
    direct.start()
    core.start()
    harness = Harness(StageService(pi_model, coordinator=core))
    baseline = make_connections(rt, data, direct)
    split = make_connections(split_rt, data, RemoteCoordinator(harness.client))
    split = [(DecisionPolicy(conn, harness.client), session, inner) for conn, session, inner in split]
    try:
        # Freeze only nondeterministic telemetry clocks. Compare full response
        # bytes, including the plugin's diagnostics, and all committed histories.
        with patch.object(plugin.time, "perf_counter_ns", return_value=1_000_000), \
             patch.object(plugin.time, "perf_counter", return_value=1.), \
             patch.object(plugin.time, "time", return_value=1.):
            for ci, start, obs in tape(data, steps=8):
                if baseline[ci][1].pending:
                    for conn, _, _ in (baseline[ci], split[ci]):
                        conn.on_episode_start(**start)
                before = (baseline[ci][2].encodes, baseline[ci][2].policy_calls)
                expected = baseline[ci][0].infer(dict(obs))
                actual = split[ci][0].infer(dict(obs))
                assert msgpack_numpy.Packer().pack(actual) == msgpack_numpy.Packer().pack(expected)
                a, b = baseline[ci][1], split[ci][1]
                assert a.hits == b.hits and a.has_vision == b.has_vision and a.blind_age == b.blind_age
                for name in ("b_v0", "b_v1", "b_rs", "b_raw", "b_aex"):
                    x, y = getattr(a, name), getattr(b, name)
                    assert np.array_equal(x.a[:x.n], y.a[:y.n], equal_nan=True)
            for old, new in zip(baseline, split):
                old[0].on_episode_end(success=False)
                new[0].on_episode_end(success=False)
        # At least one model-skipping decision must actually have executed.
        assert sum(inner.encodes for _, _, inner in baseline) < 16
        assert all(not handles for handles in harness.service.sessions.values())
    finally:
        harness.close()
        direct.stop()
        core.stop()
