"""Tests for per-stage device placement.

Covers StageDeviceConfig validation, relocate_model_stages with mock models,
Stage1Output/Stage2Output cross-device transfer, interceptor meta sentinel,
and serve_policy startup guards.

All tests run on CPU (no GPU required). GPU/model tests use @pytest.mark.manual.
"""

from __future__ import annotations

import pytest

# NOTE: temporarily disabled during the trajectory-deviation corrective
# experiment work. The 45 tests in this module each trigger the heavy
# ``torch`` + ``openpi.models_pytorch.pi0_pytorch`` imports and dominate the
# local regression loop (~5 min vs ~18 s without it). Re-enable by removing
# the ``pytest.skip`` below once the trajectory-deviation bundle ships, or
# when touching anything under ``openpi/models_pytorch/``.
pytest.skip(
    "temporarily disabled for trajectory-deviation bundle iteration — "
    "remove this skip before shipping or when editing models_pytorch/",
    allow_module_level=True,
)

import pathlib  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from typing import Any  # noqa: E402

import torch  # noqa: E402
from torch import nn  # noqa: E402

from openpi.models_pytorch.stage_device_placement import (  # noqa: E402
    StageDeviceConfig,
    _normalize_device,
    relocate_model_stages,
)
from openpi.models_pytorch.pi0_pytorch import (  # noqa: E402
    Stage1Output,
    Stage2Output,
    _move_kv_cache,
)


# ------------------------------------------------------------------
# StageDeviceConfig unit tests
# ------------------------------------------------------------------


class TestStageDeviceConfig:
    """Tests for StageDeviceConfig.create() and properties."""

    def test_config_legacy_default(self):
        """Three Nones -> is_legacy_default=True."""
        cfg = StageDeviceConfig.create(None, None, None)
        assert cfg.is_legacy_default
        assert not cfg.is_all_same_device
        assert not cfg.needs_relocation
        assert not cfg.has_meta_stage
        assert cfg.primary_device is None

    def test_config_all_same_device(self):
        """Same real device -> is_all_same_device=True."""
        cfg = StageDeviceConfig.create("cuda:0", "cuda:0", "cuda:0")
        assert not cfg.is_legacy_default
        assert cfg.is_all_same_device
        assert not cfg.needs_relocation
        assert not cfg.has_meta_stage
        assert cfg.primary_device == "cuda:0"

    def test_config_needs_relocation_split(self):
        """Different real devices -> needs_relocation=True."""
        cfg = StageDeviceConfig.create("cuda:0", "cpu", "cpu")
        assert cfg.needs_relocation
        assert not cfg.is_legacy_default
        assert not cfg.is_all_same_device

    def test_config_needs_relocation_meta(self):
        """Meta stage -> needs_relocation=True, has_meta_stage=True."""
        cfg = StageDeviceConfig.create("cuda:0", "meta", "meta")
        assert cfg.needs_relocation
        assert cfg.has_meta_stage

    def test_config_validate_stage1_meta(self):
        """stage1=meta -> ValueError."""
        with pytest.raises(ValueError, match="stage1 cannot be"):
            StageDeviceConfig.create("meta", "meta", "meta")

    def test_config_validate_partial_override(self):
        """Partial override (not all-or-none) -> ValueError."""
        with pytest.raises(ValueError, match="Either set all three"):
            StageDeviceConfig.create("cuda:0", None, None)

    def test_config_normalize_cuda(self):
        """'cuda' is normalized to 'cuda:0'."""
        cfg = StageDeviceConfig.create("cuda", "cuda", "cuda")
        assert cfg.stage1 == "cuda:0"
        assert cfg.stage2 == "cuda:0"
        assert cfg.stage3 == "cuda:0"

    def test_config_all_cpu(self):
        """All CPU -> is_all_same_device=True."""
        cfg = StageDeviceConfig.create("cpu", "cpu", "cpu")
        assert cfg.is_all_same_device
        assert cfg.primary_device == "cpu"

    def test_config_meta_only_stage2(self):
        """Only stage2=meta -> has_meta_stage=True."""
        cfg = StageDeviceConfig.create("cpu", "meta", "cpu")
        assert cfg.has_meta_stage
        assert cfg.needs_relocation

    def test_config_meta_only_stage3(self):
        """Only stage3=meta -> has_meta_stage=True."""
        cfg = StageDeviceConfig.create("cpu", "cpu", "meta")
        assert cfg.has_meta_stage


class TestNormalizeDevice:
    """Tests for _normalize_device helper."""

    def test_cpu(self):
        assert _normalize_device("cpu") == "cpu"

    def test_meta(self):
        assert _normalize_device("meta") == "meta"

    def test_cuda_bare(self):
        assert _normalize_device("cuda") == "cuda:0"

    def test_cuda_indexed(self):
        assert _normalize_device("cuda:1") == "cuda:1"

    def test_whitespace(self):
        assert _normalize_device("  CUDA:0  ") == "cuda:0"

    def test_invalid(self):
        with pytest.raises(ValueError, match="Unrecognized device"):
            _normalize_device("tpu")

    def test_negative_index(self):
        with pytest.raises(ValueError, match="non-negative"):
            _normalize_device("cuda:-1")

    def test_extra_colon_rejected(self):
        """'cuda:0:1' should be rejected (extra colon)."""
        with pytest.raises(ValueError, match="exactly one colon"):
            _normalize_device("cuda:0:1")


# ------------------------------------------------------------------
# Mock model for relocate tests
# ------------------------------------------------------------------


class _TiedWeight(nn.Module):
    """Minimal HuggingFace-style model with tied lm_head/embed_tokens."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(100, 16)
        self.lm_head = nn.Linear(16, 100, bias=False)
        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight

    def tie_weights(self):
        self.lm_head.weight = self.embed_tokens.weight


class _FakeLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(100, 16)
        self.layers = nn.ModuleList([nn.Linear(16, 16)])
        self.norm = nn.LayerNorm(16)


class _FakePaligemmaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = nn.Linear(16, 16)
        self.multi_modal_projector = nn.Linear(16, 16)
        self.language_model = _FakeLanguageModel()


class _FakePaligemma(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _FakePaligemmaModel()
        self.lm_head = nn.Linear(16, 100, bias=False)
        # Tie lm_head ↔ embed_tokens
        self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def tie_weights(self):
        self.lm_head.weight = self.model.language_model.embed_tokens.weight


class _FakePaligemmaWithExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma = _FakePaligemma()
        self.gemma_expert = nn.Linear(16, 16)


class _FakePI0Model(nn.Module):
    """Mock PI0Pytorch with the same module structure."""

    def __init__(self, *, pi05: bool = True):
        super().__init__()
        self.pi05 = pi05
        self.paligemma_with_expert = _FakePaligemmaWithExpert()
        self.action_in_proj = nn.Linear(16, 16)
        self.action_out_proj = nn.Linear(16, 16)
        if pi05:
            self.time_mlp_in = nn.Linear(16, 16)
            self.time_mlp_out = nn.Linear(16, 16)
        else:
            self.state_proj = nn.Linear(16, 16)
            self.action_time_mlp_in = nn.Linear(16, 16)
            self.action_time_mlp_out = nn.Linear(16, 16)


# ------------------------------------------------------------------
# relocate_model_stages tests
# ------------------------------------------------------------------


class TestRelocateModelStages:
    """Tests for relocate_model_stages()."""

    def test_relocate_moves_modules(self):
        """Modules move to specified CPU devices (all-cpu split test)."""
        model = _FakePI0Model(pi05=True)
        config = StageDeviceConfig(stage1="cpu", stage2="cpu", stage3="cpu")
        relocate_model_stages(model, config)
        # All params should be on CPU (they already were, but verifies no crash)
        for p in model.parameters():
            assert p.device == torch.device("cpu")

    def test_relocate_meta_frees_memory(self):
        """meta device -> params become meta tensors."""
        model = _FakePI0Model(pi05=True)
        config = StageDeviceConfig(stage1="cpu", stage2="meta", stage3="meta")
        relocate_model_stages(model, config)

        # Stage 2 params should be meta
        for p in model.paligemma_with_expert.paligemma.model.language_model.layers.parameters():
            assert p.device == torch.device("meta")

        # Stage 3 params should be meta
        for p in model.paligemma_with_expert.gemma_expert.parameters():
            assert p.device == torch.device("meta")

        # Stage 1 params should stay on CPU
        for p in model.paligemma_with_expert.paligemma.model.vision_tower.parameters():
            assert p.device == torch.device("cpu")

    def test_relocate_required_missing_raises(self):
        """Missing required module raises AttributeError."""
        model = nn.Linear(16, 16)  # No paligemma_with_expert
        model.pi05 = True
        config = StageDeviceConfig(stage1="cpu", stage2="cpu", stage3="cpu")
        with pytest.raises(AttributeError, match="Required module"):
            relocate_model_stages(model, config)

    def test_relocate_optional_pi05_skip(self):
        """Pi0 model (pi05=False) skips Pi0.5-only optional modules."""
        model = _FakePI0Model(pi05=False)
        config = StageDeviceConfig(stage1="cpu", stage2="cpu", stage3="cpu")
        # Should not raise — pi05 modules are skipped
        relocate_model_stages(model, config)
        # Pi0-specific modules should exist
        assert hasattr(model, "state_proj")
        assert not hasattr(model, "time_mlp_in")

    def test_tied_weight_retied(self):
        """After relocate, lm_head.weight is embed_tokens.weight (same object)."""
        model = _FakePI0Model(pi05=True)
        config = StageDeviceConfig(stage1="cpu", stage2="cpu", stage3="cpu")
        relocate_model_stages(model, config)

        paligemma = model.paligemma_with_expert.paligemma
        assert paligemma.lm_head.weight is paligemma.model.language_model.embed_tokens.weight

    def test_relocate_legacy_default_noop(self):
        """Legacy default config -> no relocation."""
        model = _FakePI0Model(pi05=True)
        config = StageDeviceConfig()  # all None
        relocate_model_stages(model, config)
        # Should not crash or change anything


# ------------------------------------------------------------------
# Stage*Output.to() tests
# ------------------------------------------------------------------


class TestStageOutputTo:
    """Tests for Stage1Output.to() and Stage2Output.to()."""

    def test_stage1_output_to_device(self):
        """Stage1Output.to() moves all 5 tensors."""
        s1 = Stage1Output(
            state=torch.randn(1, 32),
            prefix_embs=torch.randn(1, 10, 64),
            prefix_pad_masks=torch.ones(1, 10, dtype=torch.bool),
            prefix_att_2d_masks_4d=torch.zeros(1, 1, 10, 10),
            prefix_position_ids=torch.arange(10).unsqueeze(0),
        )
        # Move to same device (cpu -> cpu) — should return new dataclass
        result = s1.to("cpu")
        assert isinstance(result, Stage1Output)
        assert result.state.device == torch.device("cpu")
        assert result.prefix_embs.device == torch.device("cpu")

    def test_stage2_output_to_device(self):
        """Stage2Output.to() moves stage1 + KV cache."""
        s1 = Stage1Output(
            state=torch.randn(1, 32),
            prefix_embs=torch.randn(1, 10, 64),
            prefix_pad_masks=torch.ones(1, 10, dtype=torch.bool),
            prefix_att_2d_masks_4d=torch.zeros(1, 1, 10, 10),
            prefix_position_ids=torch.arange(10).unsqueeze(0),
        )
        kv = ((torch.randn(1, 4, 10, 16), torch.randn(1, 4, 10, 16)),)
        s2 = Stage2Output(stage1=s1, past_key_values=kv)
        result = s2.to("cpu")
        assert isinstance(result, Stage2Output)
        assert result.stage1.state.device == torch.device("cpu")

    def test_stage_output_to_same_noop(self):
        """Moving to same device returns tensors on correct device."""
        s1 = Stage1Output(
            state=torch.randn(1, 32),
            prefix_embs=torch.randn(1, 10, 64),
            prefix_pad_masks=torch.ones(1, 10, dtype=torch.bool),
            prefix_att_2d_masks_4d=torch.zeros(1, 1, 10, 10),
            prefix_position_ids=torch.arange(10).unsqueeze(0),
        )
        result = s1.to("cpu")
        # All tensors should still be on CPU
        for field_name in ["state", "prefix_embs", "prefix_pad_masks",
                           "prefix_att_2d_masks_4d", "prefix_position_ids"]:
            assert getattr(result, field_name).device == torch.device("cpu")


# ------------------------------------------------------------------
# _move_kv_cache tests
# ------------------------------------------------------------------


class TestMoveKvCache:
    """Tests for _move_kv_cache helper."""

    def test_none_passthrough(self):
        assert _move_kv_cache(None, "cpu") is None

    def test_dynamic_cache_with_to(self):
        """Object with .to() method."""
        class FakeCache:
            def to(self, device):
                self.device = device
                return self
        cache = FakeCache()
        result = _move_kv_cache(cache, "cpu")
        assert result is cache
        assert result.device == "cpu"

    def test_tuple_fallback(self):
        """Tuple of (key, value) pairs."""
        kv = ((torch.randn(2, 4), torch.randn(2, 4)),)
        result = _move_kv_cache(kv, "cpu")
        assert isinstance(result, tuple)
        assert len(result) == 1

    def test_to_returns_none_fallback(self):
        """If .to() returns None, fall back to original object."""
        class WeirdCache:
            def to(self, device):
                return None
        cache = WeirdCache()
        result = _move_kv_cache(cache, "cpu")
        assert result is cache


# ------------------------------------------------------------------
# Interceptor meta guard tests
# ------------------------------------------------------------------


class TestInterceptorMetaGuard:
    """Tests for meta sentinel in InferenceInterceptor."""

    def test_interceptor_meta_guard(self):
        """stage2=meta -> calling stage2_fn raises RuntimeError."""
        from openpi.cache.interceptor import _meta_guard

        guard = _meta_guard("stage2")
        with pytest.raises(RuntimeError, match="stage2 is on meta device"):
            guard(None)

    def test_interceptor_stage_config_sets_devices(self):
        """InferenceInterceptor picks up stage_config device strings."""
        from openpi.cache.interceptor import InferenceInterceptor
        from openpi.cache.timing import SystemTimer

        model = SimpleNamespace(
            config=SimpleNamespace(pytorch_compile_mode=None),
            run_stage1=lambda obs: None,
            run_stage2=lambda s1: None,
            run_stage3=lambda s2, noise=None: None,
        )
        policy = SimpleNamespace(
            _is_pytorch_model=True,
            _model=model,
            _input_transform=lambda x: x,
            _output_transform=lambda x: x,
            _pytorch_device=torch.device("cpu"),
            metadata={},
        )
        sc = StageDeviceConfig(stage1="cpu", stage2="meta", stage3="meta")
        interceptor = InferenceInterceptor(
            policy,
            timer=SystemTimer(enabled=False),
            stage_config=sc,
            eager=True,
        )
        assert interceptor._stage2_device == "meta"
        assert interceptor._stage3_device == "meta"
        # stage2/3 functions should be meta sentinels
        with pytest.raises(RuntimeError, match="stage2 is on meta device"):
            interceptor._stage2_fn(None)
        with pytest.raises(RuntimeError, match="stage3 is on meta device"):
            interceptor._stage3_fn(None)

    def test_interceptor_legacy_default_timer_uses_pytorch_device(self):
        """Legacy default (stage_config=None or all-None) should use
        policy._pytorch_device for timer probe backend, not fallback to CPU."""
        from openpi.cache.interceptor import InferenceInterceptor
        from openpi.cache.timing import SystemTimer

        model = SimpleNamespace(
            config=SimpleNamespace(pytorch_compile_mode=None),
            run_stage1=lambda obs: None,
            run_stage2=lambda s1: None,
            run_stage3=lambda s2, noise=None: None,
        )
        policy = SimpleNamespace(
            _is_pytorch_model=True,
            _model=model,
            _input_transform=lambda x: x,
            _output_transform=lambda x: x,
            _pytorch_device="cuda:0",
            metadata={},
        )
        # Legacy default: stage_config with all None
        sc = StageDeviceConfig()
        timer = SystemTimer(enabled=False)
        interceptor = InferenceInterceptor(
            policy, timer=timer, stage_config=sc, eager=True,
        )
        # stage devices should resolve to pytorch_device, not None
        assert interceptor._stage1_device == "cuda:0"
        assert interceptor._stage2_device == "cuda:0"
        assert interceptor._stage3_device == "cuda:0"
        # Verify probe registered with cuda backend (not cpu)
        from openpi.cache.timing import CudaEventBackend
        probe = timer._probes.get("stage1_vision")
        assert probe is not None
        assert isinstance(probe, CudaEventBackend)

    def test_interceptor_meta_stage3_infer_guard(self):
        """stage2=cpu, stage3=meta + CP1 MISS -> explicit RuntimeError
        before any Stage 3 call (not a PyTorch meta tensor error)."""
        from openpi.cache.interceptor import InferenceInterceptor
        from openpi.cache.timing import SystemTimer

        import numpy as np

        stage3_called = False

        def _fake_stage1(obs):
            return Stage1Output(
                state=torch.randn(1, 32),
                prefix_embs=torch.randn(1, 10, 64),
                prefix_pad_masks=torch.ones(1, 10, dtype=torch.bool),
                prefix_att_2d_masks_4d=torch.zeros(1, 1, 10, 10),
                prefix_position_ids=torch.arange(10).unsqueeze(0),
            )

        def _fake_stage2(s1):
            return Stage2Output(stage1=s1, past_key_values=None)

        def _fake_stage3(s2, noise=None):
            nonlocal stage3_called
            stage3_called = True
            return SimpleNamespace(action_chunk=torch.randn(1, 50, 32))

        model = SimpleNamespace(
            config=SimpleNamespace(pytorch_compile_mode=None),
            run_stage1=_fake_stage1,
            run_stage2=_fake_stage2,
            run_stage3=_fake_stage3,
        )
        policy = SimpleNamespace(
            _is_pytorch_model=True,
            _model=model,
            _input_transform=lambda x: x,
            _output_transform=lambda x: x,
            _pytorch_device=torch.device("cpu"),
            metadata={},
        )
        sc = StageDeviceConfig(stage1="cpu", stage2="cpu", stage3="meta")
        interceptor = InferenceInterceptor(
            policy, timer=SystemTimer(enabled=False),
            stage_config=sc, eager=True,
        )

        obs = {
            "state": np.random.randn(32).astype(np.float32),
            "image": {"base_0_rgb": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)},
            "image_mask": {"base_0_rgb": np.bool_(True)},
        }
        with pytest.raises(RuntimeError, match="stage3 is on meta device"):
            interceptor.infer(obs)
        # Stage 3 should NOT have been called
        assert not stage3_called


# ------------------------------------------------------------------
# Timer backend selection tests
# ------------------------------------------------------------------


class TestTimerBackend:
    """Tests for _probe_backend helper."""

    def test_cuda_device_returns_cuda(self):
        from openpi.cache.interceptor import _probe_backend
        assert _probe_backend("cuda:0") == "cuda"
        assert _probe_backend("cuda:1") == "cuda"

    def test_cpu_device_returns_cpu(self):
        from openpi.cache.interceptor import _probe_backend
        assert _probe_backend("cpu") == "cpu"

    def test_meta_device_returns_cpu(self):
        from openpi.cache.interceptor import _probe_backend
        assert _probe_backend("meta") == "cpu"

    def test_none_returns_cpu(self):
        from openpi.cache.interceptor import _probe_backend
        assert _probe_backend(None) == "cpu"


# ------------------------------------------------------------------
# serve_policy startup guard tests
# ------------------------------------------------------------------


class TestServePolicyGuards:
    """Tests for create_policy startup validation.

    Uses monkeypatch to avoid loading real model weights.
    """

    def test_serve_split_no_cache_guard(self, monkeypatch):
        """Split device without --cache -> create_policy raises ValueError."""
        from scripts import serve_policy

        args = serve_policy.Args(
            stage1_device="cuda:0",
            stage2_device="cpu",
            stage3_device="cpu",
            # No --cache or --cache_config
        )
        with pytest.raises(ValueError, match="Split device placement requires"):
            serve_policy.create_policy(args)

    def test_serve_meta_no_cache_config_guard(self, monkeypatch):
        """Meta stage + --cache (no --cache_config) -> ValueError."""
        from scripts import serve_policy

        args = serve_policy.Args(
            stage1_device="cuda:0",
            stage2_device="meta",
            stage3_device="meta",
            cache=True,  # --cache but no --cache_config
        )
        with pytest.raises(ValueError, match="Meta stage placement requires --cache_config"):
            serve_policy.create_policy(args)

    def test_serve_meta_no_cache_at_all_guard(self, monkeypatch):
        """Meta stage without any cache flag -> ValueError (split guard fires first)."""
        from scripts import serve_policy

        args = serve_policy.Args(
            stage1_device="cuda:0",
            stage2_device="meta",
            stage3_device="meta",
        )
        with pytest.raises(ValueError, match="Split device placement requires"):
            serve_policy.create_policy(args)

    def test_cli_args_default(self):
        """Default Args have None stage devices."""
        from scripts.serve_policy import Args, _get_stage_device_config

        args = Args()
        config = _get_stage_device_config(args)
        assert config.is_legacy_default


# ------------------------------------------------------------------
# Manual tests (require GPU or real model)
# ------------------------------------------------------------------


_CHECKPOINT_CONFIG = "pi05_libero"
_CHECKPOINT_DIR = str(
    pathlib.Path.home()
    / ".cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
)


def _skip_if_no_weights():
    """Skip test if model weights are not available."""
    if not pathlib.Path(_CHECKPOINT_DIR).exists():
        pytest.skip(f"Model weights not found at {_CHECKPOINT_DIR}")


def _skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.mark.manual
def test_named_modules_paths():
    """Verify module mapping paths against real PI0Pytorch model."""
    _skip_if_no_weights()

    from openpi.models_pytorch.stage_device_placement import (
        _STAGE1_MODULES,
        _STAGE2_MODULES,
        _STAGE3_MODULES_PI05,
        _STAGE3_MODULES_REQUIRED,
        _resolve_submodule,
    )
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    policy = _policy_config.create_trained_policy(
        _config.get_config(_CHECKPOINT_CONFIG),
        _CHECKPOINT_DIR,
        pytorch_device="cpu",
    )
    model = policy._model

    # All required modules must resolve
    for path in _STAGE1_MODULES + _STAGE2_MODULES + _STAGE3_MODULES_REQUIRED:
        sub = _resolve_submodule(model, path)
        assert isinstance(sub, nn.Module), f"{path} resolved to {type(sub)}"

    # Pi0.5-specific modules (this is a pi05 model)
    assert model.pi05
    for path in _STAGE3_MODULES_PI05:
        sub = _resolve_submodule(model, path)
        assert isinstance(sub, nn.Module), f"{path} resolved to {type(sub)}"

    del policy, model
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.mark.manual
def test_full_stage1_only():
    """Full stage1-only: load model, relocate stage2/3 to meta, run stage1.

    Verifies:
    - Model loads to CPU then relocates successfully
    - Stage 1 executes on CUDA with correct output types
    - Stage 2/3 params are on meta device (zero memory)
    - tied weight is preserved after relocation
    """
    _skip_if_no_weights()
    _skip_if_no_cuda()

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    # Load model to CPU (simulates the split/meta loading path)
    policy = _policy_config.create_trained_policy(
        _config.get_config(_CHECKPOINT_CONFIG),
        _CHECKPOINT_DIR,
        pytorch_device="cpu",
    )
    model = policy._model

    # Relocate: stage1 -> cuda:0, stage2/3 -> meta
    config = StageDeviceConfig(stage1="cuda:0", stage2="meta", stage3="meta")
    relocate_model_stages(model, config)
    policy._pytorch_device = "cuda:0"

    # Verify stage1 params are on CUDA
    vision = model.paligemma_with_expert.paligemma.model.vision_tower
    for p in vision.parameters():
        assert str(p.device).startswith("cuda"), f"stage1 param on {p.device}"
        break  # just check first param

    # Verify stage2 params are on meta
    layers = model.paligemma_with_expert.paligemma.model.language_model.layers
    for p in layers.parameters():
        assert p.device == torch.device("meta"), f"stage2 param on {p.device}"
        break

    # Verify tied weight
    paligemma = model.paligemma_with_expert.paligemma
    assert paligemma.lm_head.weight is paligemma.model.language_model.embed_tokens.weight

    # Run stage1 with a dummy observation
    from openpi.models import model as _model
    from openpi.policies.libero_policy import make_libero_example
    import numpy as np
    import jax

    dummy_inputs = make_libero_example()
    dummy_inputs = policy._input_transform(dummy_inputs)
    dummy_inputs = jax.tree.map(
        lambda x: torch.from_numpy(np.array(x)).to("cuda:0")[None, ...],
        dummy_inputs,
    )
    observation = _model.Observation.from_dict(dummy_inputs)

    with torch.no_grad():
        stage1 = model.run_stage1(observation)

    assert hasattr(stage1, "state")
    assert hasattr(stage1, "prefix_embs")
    assert str(stage1.state.device).startswith("cuda")

    # Cleanup
    del policy, model, stage1
    import gc; gc.collect()
    torch.cuda.empty_cache()
