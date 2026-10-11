from dataclasses import replace
from types import SimpleNamespace as NS
import json
import numpy as np
import pytest
import torch

from exp.r17_integration.common.contracts import CaptureSpec, RequestContext, SamplerSpec
from exp.r17_integration.common.request_rng import RequestRNG
from exp.r17_integration.common.serving import full_call
from exp.r17_integration.models.pi05.adapter import Pi05MetaWorldAdapter, model_spec, sampler_for_steps, content_manifest


class ToyModel:
    """Real native π sampler methods; only the neural velocity is a CPU fixture."""
    def __init__(self):
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        self.config = NS(action_horizon=5, action_dim=32)
        self.counts = [0, 0, 0]
        self.PI = PI0Pytorch
        from types import MethodType
        for name in ("run_stage3", "run_stage3_from", "_stage3_action_expert", "_stage3_with_intermediates"):
            setattr(self, name, MethodType(getattr(PI0Pytorch, name), self))

    def to(self, device):
        return self

    def eval(self):
        return self

    def run_stage1(self, obs):
        from openpi.models_pytorch.pi0_pytorch import Stage1Output
        self.counts[0] += 1
        base = torch.arange(256*2048, dtype=torch.float32).reshape(256, 2048) / 4096
        base += obs.images["base_0_rgb"].float().mean() / 255
        prefix = torch.cat([base, torch.full((512, 2048), 777), torch.ones((2, 2048))])[None]
        return Stage1Output(obs.state, prefix, torch.ones((1, 770), dtype=torch.bool),
                            torch.zeros((1, 1, 1, 1)), torch.arange(770)[None])

    def _stage1_token_prep(self, obs):
        s = self.run_stage1(obs)
        return s.state, s.prefix_embs, s.prefix_pad_masks, s.prefix_att_2d_masks_4d, s.prefix_position_ids

    def _stage2_llm_backbone(self, *args):
        self.counts[1] += 1
        return None

    def run_stage2(self, s):
        from openpi.models_pytorch.pi0_pytorch import Stage2Output
        self._stage2_llm_backbone()
        return Stage2Output(s, None)

    def run_stage2_capture(self, s):
        return self.run_stage2(s)

    def denoise_step(self, state, mask, kv, x, t):
        self.counts[2] += 1
        return (x*.2 + t[:, None, None]*.07 + state[:, None, :]*.005).float()

    def sample_actions(self, device, obs, noise, num_steps=10):
        return self.run_stage3(self.run_stage2(self.run_stage1(obs)), noise=noise, num_steps=num_steps).action_chunk


def fixture_adapter():
    from openpi.policies.policy import Policy
    from openpi.policies.metaworld_policy import MetaworldInputs, MetaworldOutputs
    from openpi.transforms import Normalize, Unnormalize, PadStatesAndActions
    from openpi.shared.normalize import NormStats
    stats = {"state": NormStats(mean=np.zeros(4), std=np.ones(4), q01=np.array([-2]*4), q99=np.array([3]*4)),
             "actions": NormStats(mean=np.zeros(4), std=np.ones(4), q01=np.array([-3, -4, -5, -1]), q99=np.array([2, 3, 4, 1]))}
    model = ToyModel()
    # Drop prompt after testing the true benchmark transform; tokenizer/vendor
    # language encoding is outside this small CPU velocity fixture.
    def drop_prompt(x):
        return {k: v for k, v in x.items() if k != "prompt"}
    policy = Policy(model, is_pytorch=True, pytorch_device="cpu",
                    transforms=[MetaworldInputs(), Normalize(stats, use_quantiles=True), PadStatesAndActions(32), drop_prompt],
                    output_transforms=[Unnormalize(stats, use_quantiles=True), MetaworldOutputs()])
    return Pi05MetaWorldAdapter(policy, model_spec("cpu_fixture_weights", "cpu_fixture_stats")), model


def observation():
    return {"observation/image": np.arange(480*480*3, dtype=np.uint8).reshape(480, 480, 3),
            "observation/state": np.array([.1, .2, .3, .4], dtype=np.float32), "prompt": "Reach a goal position"}


def request():
    return RequestContext("cpu", "mt50", "reach-v3", 8, 0)


@pytest.mark.parametrize("k", [10, 2, 1])
def test_real_native_sampler_full_staged_norm_and_nfe(k):
    a, model = fixture_adapter()
    obs, r = observation(), request()
    rng = RequestRNG(r)
    # Original Policy uses the same explicit noise; disable its timing-only
    # stage helper for K!=10 because that helper fixes the full default10.
    a.policy._staged_inference = k == 10
    a.policy._sample_kwargs = {"num_steps": k}
    legacy = a.policy.infer(obs, noise=rng.normal((5, 32)))["actions"]
    model.counts[:] = [0, 0, 0]
    result, wire = full_call(a, obs, r, sampler=sampler_for_steps(k))
    assert model.counts == [1, 1, k] and result.actual_nfe == k
    assert wire.actions.shape == (5, 4) and result.normalized_chunk.shape == (5, 32)
    assert legacy.tobytes() == wire.actions.tobytes()
    # Explicit quantile inverse once, independently of wrapper output chain.
    expected = (result.normalized_chunk[:, :4] + 1) / 2 * (np.array([5, 7, 9, 2]) + 1e-6) + np.array([-3, -4, -5, -1])
    np.testing.assert_allclose(wire.actions, expected, rtol=0, atol=1e-6)


def test_one_real_view_matches_existing_pool_math_and_ignores_masked_wrist():
    from openpi.cache.components.key_builder import _spatial_pool_tokens
    a, model = fixture_adapter()
    s = a.run_stage1(a.prepare(observation(), request()))
    fields = a.retrieval_fields(s)
    assert fields.keys() == {"vision_0", "robot_state"}
    assert fields["vision_0"].shape == (32768,) and fields["robot_state"].shape == (4,)
    reference = _spatial_pool_tokens(s.data.prefix_embs[0, :256], 16, 4)
    assert torch.equal(fields["vision_0"], reference)
    s.data.prefix_embs[:, 256:768] = -1000
    assert torch.equal(a.retrieval_fields(s)["vision_0"], fields["vision_0"])
    s.data.prefix_embs[:, :256] += 1
    assert not torch.equal(a.retrieval_fields(s)["vision_0"], fields["vision_0"])
    assert model.counts == [1, 0, 0]


def test_native_capture_resume_exact_and_foreign_snapshot_rejected():
    a, model = fixture_adapter()
    s1 = a.run_stage1(a.prepare(observation(), request()))
    s2 = a.run_stage2(s1)
    rng, sampler = RequestRNG(request()), sampler_for_steps(10)
    plain = a.sample(s2, sampler, rng)
    captured = a.sample(s2, sampler, rng, CaptureSpec(True, True))
    assert captured.normalized_chunk.tobytes() == plain.normalized_chunk.tobytes()
    assert len(captured.snapshots) == 9
    for snap in captured.snapshots:
        model.counts[2] = 0
        resumed = a.resume(s2, snap, sampler, rng, CaptureSpec())
        assert resumed.normalized_chunk.tobytes() == plain.normalized_chunk.tobytes()
        assert resumed.actual_nfe == model.counts[2] == 10 - snap.loop_index
    with pytest.raises(ValueError, match="identity"):
        a.resume(s2, replace(captured.snapshots[0], norm_identity="other"), sampler, rng)
    reduced = a.sample(s2, sampler_for_steps(2), rng, CaptureSpec(True, True))
    assert len(reduced.snapshots) == 1
    assert a.resume(s2, reduced.snapshots[0], sampler_for_steps(2), rng).normalized_chunk.tobytes() == reduced.normalized_chunk.tobytes()
    assert a.sample(s2, sampler_for_steps(1), rng, CaptureSpec(True, True)).snapshots == ()


@pytest.mark.parametrize("k", [1, 2, 10])
def test_sampler_grid_is_native_float32_accumulation(k):
    native = sampler_for_steps(k)
    t, dt, times = torch.tensor(1., dtype=torch.float32), torch.tensor(-1/k, dtype=torch.float32), []
    for _ in range(k):
        times.append(t.item())
        t += dt
    assert native.native_timesteps == tuple(times)
    assert native.native_deltas == (dt.item(),)*k


def test_h5_rejects_commit10_and_replan1_tail_is_distinct():
    layout = model_spec("m", "n").layout
    layout.validate_commit(5)
    with pytest.raises(ValueError, match="commit"):
        layout.validate_commit(10)
    experimental = model_spec("m", "n", replan_steps=1).layout
    experimental.validate_commit(5)
    assert experimental.identity != layout.identity
    assert sum(experimental.valid_action_mask) == 4


@pytest.mark.parametrize("mutation", ["nan", "state_width", "image_shape", "prompt"])
def test_invalid_observation_fails_before_model(mutation):
    a, model = fixture_adapter()
    obs = observation()
    if mutation == "nan":
        obs["observation/state"][0] = np.nan
    elif mutation == "state_width":
        obs["observation/state"] = np.zeros(8)
    elif mutation == "image_shape":
        obs["observation/image"] = np.zeros((224, 224, 3), dtype=np.uint8)
    else:
        obs["prompt"] = ""
    with pytest.raises(ValueError):
        a.prepare(obs, request())
    assert model.counts == [0, 0, 0]


def test_debug_and_control_envelopes_never_enter_transforms():
    a, _ = fixture_adapter()
    original = a.policy._input_transform
    def transform(obs):
        assert not any(k.startswith("__") for k in obs)
        return original(obs)
    a.policy._input_transform = transform
    obs = {**observation(), "__debug__": {"x": "debug"}, "__execution__": {"executed_count": 2}, "__request__": {"attempt": 8}}
    full_call(a, obs, request())


def test_checkpoint_content_hash_detects_same_size_change_and_missing_norm(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights-test")
    norm = tmp_path / "assets/metaworld_mt50/norm_stats.json"
    norm.parent.mkdir(parents=True)
    norm.write_text('{"a":1}')
    original = content_manifest(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"weights-best")
    changed = content_manifest(tmp_path)
    assert original["checkpoint_identity"] != changed["checkpoint_identity"]
    assert original["norm_identity"] == changed["norm_identity"]
    norm.write_text('{"a":2}')
    assert changed["norm_identity"] != content_manifest(tmp_path)["norm_identity"]


def test_tokenizer_content_seal_tamper_and_missing_fail_before_load(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from exp.r17_integration.models.pi05.adapter import load_adapter, _verify_loaded_tokenizer
    monkeypatch.setenv("OPENPI_DATA_HOME", str(tmp_path / "cache"))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"dummy never load")
    norm = checkpoint / "assets/metaworld_mt50/norm_stats.json"
    norm.parent.mkdir(parents=True)
    norm.write_text("{}")
    missing = content_manifest(checkpoint)
    assert missing["tokenizer"]["status"] == "missing_local_content"
    with pytest.raises(ValueError, match="tokenizer content must be sealed"):
        load_adapter({"checkpoint": checkpoint, "device": "cpu"})
    token = tmp_path / "cache/big_vision/paligemma_tokenizer.model"
    token.parent.mkdir(parents=True)
    token.write_bytes(b"native tokenizer proto")
    sealed = content_manifest(checkpoint)
    assert sealed["checkpoint_identity"] != missing["checkpoint_identity"]
    processor = SimpleNamespace(serialized_model_proto=lambda: b"native tokenizer proto")
    policy = SimpleNamespace(_input_transform=SimpleNamespace(transforms=[
        SimpleNamespace(tokenizer=SimpleNamespace(_tokenizer=processor))]))
    _verify_loaded_tokenizer(policy, sealed["tokenizer"]["sha256"])
    token.write_bytes(b"changedtokenizer proto")
    changed = content_manifest(checkpoint)
    assert changed["checkpoint_identity"] != sealed["checkpoint_identity"]
    with pytest.raises(ValueError, match="differs from sealed"):
        _verify_loaded_tokenizer(policy, changed["tokenizer"]["sha256"])


def test_default_off_import_registers_nothing_and_legacy_sources_match_head():
    import subprocess
    import sys
    code = "import sys; from exp.r17_integration.common.provider import _providers; import exp.r17_integration.models.pi05.adapter, exp.r17_integration.models.pi05.provider; assert not _providers; assert 'torch' not in sys.modules and 'jax' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, timeout=15)
    from pathlib import Path
    for name in ("exp/metaworld/tasks.py", "exp/metaworld/env.py", "exp/metaworld/episode_runner.py",
                 "src/openpi/models_pytorch/pi0_pytorch.py", "src/openpi/policies/metaworld_policy.py",
                 "src/openpi/conductor/worker.py", "src/openpi/conductor/driver.py"):
        original = subprocess.run(["git", "show", "HEAD:" + name], check=True, capture_output=True).stdout
        assert original == Path(name).read_bytes(), name


def test_replan1_ack_tail_uses_at_most_h5_and_no_extra_model_calls(tmp_path):
    import yaml
    from exp.r17_integration.models.pi05.provider import register_provider, config_document, make_connection_factory
    from exp.r17_integration.common.config import load_cache_config, build_shared_storage
    a, model = fixture_adapter()
    a.spec = model_spec("m", "n", replan_steps=1)
    provider = register_provider(a, provider_id="p2_tail_" + tmp_path.name)
    path = tmp_path / "tail.yaml"
    path.write_text(yaml.safe_dump(config_document(provider, commit=5, evidence_dir=tmp_path / "miss")))
    cfg = load_cache_config(path)
    p = make_connection_factory(provider, cfg, build_shared_storage(cfg))()
    p.on_task_begin()
    p.on_episode_start(experiment="metaworld_mt50", task="reach-v3", episode_id=0,
                       extra_metadata={"campaign": "tail_fixture", "arm": "full", "bundle_generation": "default",
                                       "task_id": 43, "task_name": "reach-v3", "original_init_idx": 0,
                                       "seed": 7, "explicit_seed": 123, "ack_protocol": "r17.execution.v1"})
    obs = observation()
    first = p.infer(obs)
    counts = list(model.counts)
    for seq in range(1, 5):
        result = p.infer({**obs, "__execution__": {"decision_seq": seq-1, "executed_count": 1}})
        assert model.counts == counts
        assert result["actions"].tobytes() == first["actions"][seq:].tobytes()
        assert result["__hit_meta__"]["stage_calls"] == (0, 0, 0)
    p.infer({**obs, "__execution__": {"decision_seq": 4, "executed_count": 1}})
    assert model.counts == [2, 2, 20]
    assert a.spec.allowed_benchmark_protocols[0] != model_spec("m", "n").allowed_benchmark_protocols[0]
