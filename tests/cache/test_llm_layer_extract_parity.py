"""Online vs offline parity for cp1_llm_layer_extract.

Hard requirement from `logs/cp1_llm_layer_extract_key_builder_plan.log.md` §6.4:
given the same observation, the KeyBuilder must produce the same query keys
whether driven by an online Stage1Output (Path A) or by HDF5-derived
synthetic Stage 1 reconstructed via `_build_fake_stage1_with_masks`
(Path B). Any drift would invalidate offline-built artifacts as queries
against an online cache.

This test is `@pytest.mark.manual` because it loads a real Pi0.5 checkpoint
(~5 GB) on GPU. CI does not run manual tests; the local Verify step (§9 in
the plan) MUST run it.

Trigger:
    PI05_CHECKPOINT_DIR=/path/to/checkpoint \
    PI05_CONFIG_NAME=pi05_libero \
    uv run pytest tests/cache/test_llm_layer_extract_parity.py -m manual -v
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.manual


# ----------------------------------------------------------------------
# Skip control
# ----------------------------------------------------------------------


def _checkpoint_dir() -> Path:
    p = os.environ.get("PI05_CHECKPOINT_DIR")
    if not p:
        pytest.skip("PI05_CHECKPOINT_DIR not set; skipping parity test")
    return Path(p)


def _config_name() -> str:
    return os.environ.get("PI05_CONFIG_NAME", "pi05_libero")


def _device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for parity test (model is too large for CPU)")
    return torch.device("cuda")


# ----------------------------------------------------------------------
# Loaded once per session
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def loaded_model_and_tokenizer():
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    ckpt = _checkpoint_dir()
    cfg = _config_name()
    device = _device()

    train_config = _config.get_config(cfg)
    policy = _policy_config.create_trained_policy(
        train_config, ckpt, pytorch_device=str(device),
    )
    model = policy._model  # noqa: SLF001
    tokenizer = PaligemmaTokenizer(max_len=200)
    return model, tokenizer, device


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _synthetic_observation(
    model,
    device,
    *,
    seed: int = 42,
    task: str = "pick up the red block and place it on the plate",
):
    """Construct an Observation with deterministic synthetic content.

    Uses the model's preprocessing pipeline to ensure the resulting
    Stage1Output has the canonical shape that production Stage 1 produces.

    Different `seed` -> different images and state. Different `task` ->
    different prompt tokens. Both knobs are required for end-to-end
    retrieval tests that need genuinely distinct observations.
    """
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.models.model import Observation

    rng = np.random.default_rng(seed)
    image_shape = (224, 224, 3)
    images = {
        "base_0_rgb": rng.integers(0, 256, image_shape, dtype=np.uint8),
        "left_wrist_0_rgb": rng.integers(0, 256, image_shape, dtype=np.uint8),
        "right_wrist_0_rgb": rng.integers(0, 256, image_shape, dtype=np.uint8),
    }
    image_masks = {k: True for k in images}

    state = rng.uniform(-1, 1, size=(32,)).astype(np.float32)

    tokenizer = PaligemmaTokenizer(max_len=200)
    tokens, mask = tokenizer.tokenize(task, state=state)

    obs = Observation(
        images={k: torch.from_numpy(v).to(device) for k, v in images.items()},
        image_masks={k: torch.tensor(v, device=device) for k, v in image_masks.items()},
        state=torch.from_numpy(state).to(device),
        tokenized_prompt=torch.from_numpy(tokens).long().to(device),
        tokenized_prompt_mask=torch.from_numpy(mask).bool().to(device),
    )
    return obs, task, state


def _write_hdf5_from_stage1(stage1, task: str, action: np.ndarray, out_path: Path):
    """Mirror EpisodeDataCollector schema for one step."""
    from openpi.cache.components.key_builder import _PROMPT_START

    prefix = stage1.prefix_embs[0].detach().cpu().numpy()  # (968, 2048)
    state = stage1.state[0].detach().cpu().numpy()         # (32,)

    with h5py.File(out_path, "w") as f:
        f.attrs["task"] = task
        f.attrs["success"] = True
        f.attrs["episode_id"] = 0
        f.attrs["num_steps"] = 1
        grp = f.create_group("step_0000")
        grp.create_dataset("vision_0", data=prefix[0:256].astype(np.float16))
        grp.create_dataset("vision_1", data=prefix[256:512].astype(np.float16))
        grp.create_dataset("vision_2", data=prefix[512:768].astype(np.float16))
        grp.create_dataset("prompt_emb", data=prefix[_PROMPT_START:].astype(np.float16))
        grp.create_dataset("robot_state", data=state)
        grp.create_dataset("clean_action", data=action)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_online_offline_layer0_hidden_states_match(loaded_model_and_tokenizer, tmp_path):
    """End-to-end parity: same obs -> online layer-0 hidden == offline layer-0 hidden
    at every non-padding position (bf16 tolerance)."""
    from exp.common.build_in_memory_cache_artifact import (
        _build_fake_stage1_with_masks,
    )

    model, tokenizer, device = loaded_model_and_tokenizer
    obs, task, _state = _synthetic_observation(model, device)

    # Online Stage 1 -> Stage1Output (canonical reference).
    with torch.no_grad():
        stage1_online = model.run_stage1(obs)

    # Persist this exact obs to HDF5, then reconstruct.
    h5_path = tmp_path / "step.h5"
    _write_hdf5_from_stage1(
        stage1_online, task,
        action=np.zeros((1, 32), dtype=np.float32),
        out_path=h5_path,
    )
    with h5py.File(h5_path, "r") as f:
        group = f["step_0000"]
        stage1_offline = _build_fake_stage1_with_masks(
            group, task_str=task, tokenizer=tokenizer, model=model, device=device,
        )

    # 1) Pad masks must match.
    assert torch.equal(
        stage1_online.prefix_pad_masks, stage1_offline.prefix_pad_masks
    ), "pad mask mismatch — re-tokenize did not recover lang_masks"

    # 2) Layer-0 hidden states at non-padding positions must match.
    layer0 = model.paligemma_with_expert.paligemma.language_model.layers[0]
    rotary = model.paligemma_with_expert.paligemma.language_model.rotary_emb

    def run_layer0(stage1):
        h = stage1.prefix_embs.to(layer0.self_attn.q_proj.weight.dtype)
        cos, sin = rotary(h, stage1.prefix_position_ids)
        out = layer0(
            h,
            attention_mask=stage1.prefix_att_2d_masks_4d,
            position_ids=stage1.prefix_position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=(cos, sin),
            adarms_cond=None,
        )
        return out[0]

    with torch.no_grad():
        h_online = run_layer0(stage1_online)[0]   # [968, 2048]
        h_offline = run_layer0(stage1_offline)[0]

    mask = stage1_online.prefix_pad_masks[0]
    diff = (h_online[mask].float() - h_offline[mask].float()).abs().max().item()
    assert torch.allclose(
        h_online[mask].float(), h_offline[mask].float(), rtol=1e-2, atol=1e-2
    ), f"layer-0 hidden state drift at non-padding positions (max abs diff: {diff:.4g})"


def test_online_offline_query_keys_match(loaded_model_and_tokenizer, tmp_path):
    """Final key-level parity: KeyBuilder.build() output must match across paths."""
    from exp.common.build_in_memory_cache_artifact import (
        _build_fake_stage1_with_masks,
    )
    from openpi.cache.components.llm_layer_key_builder import (
        CP1LLMLayerExtractKeyBuilder,
    )
    from openpi.cache.components.prefix_reducer import PrefixMeanPoolReducer
    from openpi.cache.types import CheckpointID

    model, tokenizer, device = loaded_model_and_tokenizer
    obs, task, _state = _synthetic_observation(model, device)

    with torch.no_grad():
        stage1_online = model.run_stage1(obs)

    h5_path = tmp_path / "step.h5"
    _write_hdf5_from_stage1(
        stage1_online, task,
        action=np.zeros((1, 32), dtype=np.float32),
        out_path=h5_path,
    )
    with h5py.File(h5_path, "r") as f:
        group = f["step_0000"]
        stage1_offline = _build_fake_stage1_with_masks(
            group, task_str=task, tokenizer=tokenizer, model=model, device=device,
        )

    builder = CP1LLMLayerExtractKeyBuilder(
        reducer=PrefixMeanPoolReducer(), extract_layer=0,
    )
    builder.attach_model(model)

    builder.collect(CheckpointID.CP1, stage1=stage1_online)
    keys_online = builder.build(CheckpointID.CP1)
    builder.clear()

    builder.collect(CheckpointID.CP1, stage1=stage1_offline)
    keys_offline = builder.build(CheckpointID.CP1)
    builder.clear()

    assert set(keys_online.keys()) == set(keys_offline.keys())
    for f in keys_online:
        diff = (keys_online[f] - keys_offline[f]).abs().max().item()
        assert torch.allclose(
            keys_online[f], keys_offline[f], rtol=1e-2, atol=1e-2
        ), f"key '{f}' drifts between online and offline (max abs diff: {diff:.4g})"


# ----------------------------------------------------------------------
# Plan §8.5 — Layer 0 partial replay vs full-forward output_hidden_states[1]
# ----------------------------------------------------------------------


def test_layer0_partial_replay_matches_full_forward_hidden_states(
    loaded_model_and_tokenizer,
):
    """The KeyBuilder runs `layers[0..N]` standalone; HF's full forward with
    `output_hidden_states=True` exposes `hidden_states[1]` = layer-0 output.
    For the same Stage1 inputs, both paths must produce the same tensor at
    non-padding positions (bf16 tolerance).

    This pins down the contract that partial replay does not drift from
    Stage 2's actual layer-0 computation."""
    model, _, device = loaded_model_and_tokenizer
    obs, _task, _state = _synthetic_observation(model, device)

    with torch.no_grad():
        stage1 = model.run_stage1(obs)

    layer0 = model.paligemma_with_expert.paligemma.language_model.layers[0]
    rotary = model.paligemma_with_expert.paligemma.language_model.rotary_emb
    language_model = model.paligemma_with_expert.paligemma.language_model

    # Path A: KeyBuilder-style partial replay of layer 0 only.
    h_in = stage1.prefix_embs.to(layer0.self_attn.q_proj.weight.dtype)
    cos, sin = rotary(h_in, stage1.prefix_position_ids)
    with torch.no_grad():
        layer0_out = layer0(
            h_in,
            attention_mask=stage1.prefix_att_2d_masks_4d,
            position_ids=stage1.prefix_position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=(cos, sin),
            adarms_cond=None,
        )
    h_partial = layer0_out[0][0].float()  # [968, 2048]

    # Path B: HF full forward with output_hidden_states=True; index [1] is
    # the output of layer 0 (index [0] is `inputs_embeds` pre-layer-0).
    # Mirror the current model backend. Stage 2 configures sdpa once in
    # PI0Pytorch.__init__; the keybuilder no longer mutates shared config.
    with torch.no_grad():
        full_output = language_model.forward(
            inputs_embeds=stage1.prefix_embs,
            attention_mask=stage1.prefix_att_2d_masks_4d,
            position_ids=stage1.prefix_position_ids,
            past_key_values=None,
            use_cache=True,
            output_hidden_states=True,
        )
    h_full = full_output.hidden_states[1][0].float()

    mask = stage1.prefix_pad_masks[0]
    diff = (h_partial[mask] - h_full[mask]).abs().max().item()
    assert torch.allclose(
        h_partial[mask], h_full[mask], rtol=1e-2, atol=1e-2
    ), (
        f"KeyBuilder layer-0 replay drifts from HF full-forward "
        f"hidden_states[1] (max abs diff: {diff:.4g}). Likely cause: "
        f"missing or extra step in the partial-replay loop "
        f"(check llm_layer_key_builder._extract)."
    )


# ----------------------------------------------------------------------
# Plan §8.5 — End-to-end real-model collect -> build -> search -> fetch
# ----------------------------------------------------------------------


def test_full_chain_with_real_model_collect_build_search_fetch(
    loaded_model_and_tokenizer,
):
    """Drive the new builder through the full Orchestrator pipeline against an
    InMemoryBackend: insert two genuinely-distinct entries (different prompts
    AND different image/state seeds), then re-query each observation and
    verify the top-1 retrieval picks **its own** entry, not the other one.

    This is the discrimination test: a degenerate "self-retrieval-only"
    pass would not catch search bugs (e.g. always returning the first
    entry). Pre-asserting `keys_a != keys_b` rules out a silent same-query
    regression where the two observations collapse to identical keys."""
    from openpi.cache.backends.in_memory_backend import InMemoryBackend
    from openpi.cache.cache_storage import CacheStorage
    from openpi.cache.components.gate import AlwaysSearchGate
    from openpi.cache.components.judge import ThresholdJudge
    from openpi.cache.components.llm_layer_key_builder import (
        CP1LLMLayerExtractKeyBuilder,
    )
    from openpi.cache.components.prefix_reducer import PrefixMeanPoolReducer
    from openpi.cache.orchestrator import CacheOrchestrator
    from openpi.cache.storage_types import CacheEntry, CachePayload
    from openpi.cache.types import CheckpointID, HitType

    from tests.cache.conftest import TestStorageSearchStrategy

    model, _tokenizer, device = loaded_model_and_tokenizer

    backend = InMemoryBackend({"vision_0": 2048, "robot_state": 32})
    storage = CacheStorage(backend)
    builder = CP1LLMLayerExtractKeyBuilder(
        reducer=PrefixMeanPoolReducer(), extract_layer=0,
    )
    builder.attach_model(model)
    orchestrator = CacheOrchestrator(
        storage=storage,
        key_builder=builder,
        gates={CheckpointID.CP1: AlwaysSearchGate()},
        judges={CheckpointID.CP1: ThresholdJudge(
            cp1_threshold=0.0, cp3_threshold=0.0,  # threshold=0 -> top-1 always hits
        )},
        search_strategies={CheckpointID.CP1: TestStorageSearchStrategy(storage, top_k=1)},
    )

    # Two observations that differ in BOTH images/state (different seeds) AND
    # prompt text. Same-seed-different-prompt would collapse vision keys; the
    # combination guarantees distinct multi-modal content.
    obs_a, _task_a, _ = _synthetic_observation(
        model, device, seed=42,
        task="pick up the red block and place it on the plate",
    )
    obs_b, _task_b, _ = _synthetic_observation(
        model, device, seed=99,
        task="open the top drawer and put the apple inside it",
    )

    payload_a = CachePayload(
        action_chunk=torch.full((1, 32), 1.0), task_key="A",
    )
    payload_b = CachePayload(
        action_chunk=torch.full((1, 32), 2.0), task_key="B",
    )

    # Insert entry A.
    with torch.no_grad():
        s1_a = model.run_stage1(obs_a)
    builder.collect(CheckpointID.CP1, stage1=s1_a)
    keys_a = builder.build(CheckpointID.CP1)
    builder.clear()
    storage.insert(CacheEntry(
        id="entry_a", checkpoint_id=CheckpointID.CP1,
        query_keys=keys_a, payload=payload_a,
    ))

    # Insert entry B.
    with torch.no_grad():
        s1_b = model.run_stage1(obs_b)
    builder.collect(CheckpointID.CP1, stage1=s1_b)
    keys_b = builder.build(CheckpointID.CP1)
    builder.clear()
    storage.insert(CacheEntry(
        id="entry_b", checkpoint_id=CheckpointID.CP1,
        query_keys=keys_b, payload=payload_b,
    ))

    # Pre-condition: A's and B's keys are genuinely different. Without this
    # the subsequent retrieval assertions would be vacuous.
    assert not torch.allclose(keys_a["vision_0"], keys_b["vision_0"]), (
        "Synthetic observations collapsed to identical keys — test is vacuous."
    )

    # Query A retrieves entry_a (not entry_b).
    result_a = orchestrator.check(CheckpointID.CP1, stage1=s1_a)
    assert result_a.hit_type == HitType.FULL_HIT
    assert result_a.entry_id == "entry_a", (
        f"Query A retrieved {result_a.entry_id!r}; search picked the wrong entry."
    )
    assert torch.equal(result_a.payload.action_chunk, payload_a.action_chunk)
    orchestrator.clear()

    # Query B retrieves entry_b (not entry_a).
    result_b = orchestrator.check(CheckpointID.CP1, stage1=s1_b)
    assert result_b.hit_type == HitType.FULL_HIT
    assert result_b.entry_id == "entry_b", (
        f"Query B retrieved {result_b.entry_id!r}; search picked the wrong entry."
    )
    assert torch.equal(result_b.payload.action_chunk, payload_b.action_chunk)
    orchestrator.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "manual"])
