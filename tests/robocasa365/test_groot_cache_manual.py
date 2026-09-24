"""Real-model checks for the two-stage split. Island B only.

Everything else in this suite runs against a stub, which can show that the
split calls the right things in the right order but cannot show that the six
statements copied out of upstream's forward still reproduce it. That is what
these tests are for, and they need the actual 7.2 GB checkpoint.

Run (the PYTHONPATH entry is not optional -- `gr00t` is a worktree, not an
installed package, and without it `importorskip` turns this file into a silent
skip that reads as a pass)::

    cd /home/weiland/projects/openpi
    PYTHONPATH=/home/weiland/gr00t_n15:/home/weiland/projects/openpi/src:/home/weiland/projects/openpi \\
      /home/weiland/gr00t_n15_venv/.venv/bin/python -m pytest \\
      tests/robocasa365/test_groot_cache_manual.py --run-manual -v

Negative controls are first-class here. `max|delta| == 0` between two paths
proves nothing on its own: if the flow-matching noise were accidentally pinned,
or if both paths shared a cached result, the equality would hold for the wrong
reason. So each equality assertion is paired with a condition under which the
difference must be non-zero.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.manual

CHECKPOINT = pathlib.Path(
    "/home/weiland/ckpt_n15_robocasa_tp/gr00t_n1-5/foundation_model_learning/"
    "target_posttraining/atomic_seen/checkpoint-60000"
)
EMBODIMENT_TAG = "new_embodiment"
SEED = 12345


@pytest.fixture(scope="module")
def policy():
    pytest.importorskip("gr00t", reason="requires the GR00T island")
    if not CHECKPOINT.exists():
        pytest.skip(f"checkpoint not present: {CHECKPOINT}")

    from gr00t.model.policy import Gr00tPolicy

    from exp.robocasa365.groot_data_config import RoboCasa365DataConfig

    data_config = RoboCasa365DataConfig()
    return Gr00tPolicy(
        model_path=str(CHECKPOINT),
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        device="cuda",
    )


@pytest.fixture(scope="module")
def runner(policy):
    from openpi.cache.groot.staged import GrootStagedRunner

    # verify_upstream left on: the pinned hash is part of what is under test.
    return GrootStagedRunner(policy.model)


def _observation(step: int = 0) -> dict:
    """A legal observation; `step` varies it so successive keys differ."""
    from exp.robocasa365 import groot_keys

    import json

    stats = json.loads((CHECKPOINT / "experiment_cfg" / "metadata.json").read_text())
    state_stats = stats[EMBODIMENT_TAG]["statistics"]["state"]

    obs: dict = {}
    rng = np.random.default_rng(1000 + step)
    resolution = groot_keys.MODEL_IMAGE_RESOLUTION
    for key in groot_keys.VIDEO_KEYS:
        obs[key] = rng.integers(0, 255, (resolution, resolution, 3), dtype=np.uint8)

    for key in groot_keys.STATE_KEYS:
        vector = np.asarray(
            state_stats[key.removeprefix("state.")]["mean"], dtype=np.float64
        )
        if key in groot_keys.QUATERNION_STATE_KEYS:
            norm = float(np.linalg.norm(vector))
            vector = (
                np.asarray(groot_keys.IDENTITY_QUATERNION_WXYZ, dtype=np.float64)
                if norm < 1e-8
                else vector / norm
            )
        else:
            vector = vector + 0.01 * step
        obs[key] = vector

    for key in groot_keys.LANGUAGE_KEYS:
        obs[key] = "pick up the object"
    return obs


def _normalized(policy, obs):
    """Wire-format obs -> normalized model input, via the PRODUCTION reshaping.

    ``_observation`` builds wire-format values (video ``(H, W, 3)``, state
    ``(D,)``); upstream ``apply_transforms`` consumes batched ``(B, T, ...)``.
    The T axis is added by the adapter's ``build_groot_observation`` and the B
    axis by the interceptor's unsqueeze — using both here means the test eats
    the same reshaping code the server runs, instead of a hand-rolled copy
    that can drift (the first real-machine run caught exactly that: a missing
    T axis sent PIL frames into ``prepare_input``).
    """
    from exp.robocasa365.groot_policy_adapter import build_groot_observation
    from openpi.cache.groot.interceptor import _is_batched, _unsqueeze_values

    shaped = build_groot_observation(obs)  # wire -> [T=1, ...] (production contract)
    if not _is_batched(shaped):
        shaped = _unsqueeze_values(shaped)  # add the batch axis, as the interceptor does
    for key, value in shaped.items():
        if not isinstance(value, np.ndarray):
            shaped[key] = np.array(value)
    return policy.apply_transforms(shaped)


def _warm_up(policy, inputs):
    """Discard one full call so cudnn's algorithm choice is already cached."""
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        policy.model.get_action(inputs)


# ------------------------------------------------------------------
# G0-C: the split reproduces the unsplit forward
# ------------------------------------------------------------------


def test_two_stage_split_is_bit_exact(policy, runner):
    inputs = _normalized(policy, _observation())
    _warm_up(policy, inputs)

    torch.manual_seed(SEED)
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        reference = policy.model.get_action(inputs)["action_pred"].float().cpu()

    torch.manual_seed(SEED)
    with runner.session():
        stage1 = runner.run_stage1(inputs)
        split = runner.run_stage2(stage1).action_pred.float().cpu()

    assert torch.equal(split, reference), (
        f"max|delta| = {(split - reference).abs().max().item()}"
    )


def test_unseeded_calls_differ(policy):
    """The control for the test above: without it, equality could be an artefact."""
    inputs = _normalized(policy, _observation())
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        first = policy.model.get_action(inputs)["action_pred"].float().cpu()
        second = policy.model.get_action(inputs)["action_pred"].float().cpu()
    assert not torch.equal(first, second), (
        "two unseeded flow-matching calls produced identical actions; the noise "
        "is pinned somewhere and the equality test above proves nothing"
    )


def test_backbone_features_match_the_upstream_backbone(policy, runner):
    """Our language-model call must land on the same tensor upstream selects."""
    inputs = _normalized(policy, _observation())
    captured = {}
    original = policy.model.action_head.get_action

    def _capture(backbone_outputs, action_inputs):
        captured["features"] = backbone_outputs["backbone_features"].float().cpu().clone()
        return original(backbone_outputs, action_inputs)

    policy.model.action_head.get_action = _capture
    try:
        with runner.session():
            runner.run_stage2(runner.run_stage1(inputs))
    finally:
        policy.model.action_head.get_action = original

    backbone_inputs, _ = policy.model.prepare_input(inputs)
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        reference = policy.model.backbone(backbone_inputs)["backbone_features"]
    reference = reference.float().cpu()

    assert torch.equal(captured["features"], reference), (
        f"max|delta| = {(captured['features'] - reference).abs().max().item()}"
    )


def test_running_stage2_twice_on_one_stage1_is_reproducible(policy, runner):
    """Pins that the action head's in-place vlln is not applied to a reused mapping."""
    inputs = _normalized(policy, _observation())
    with runner.session():
        stage1 = runner.run_stage1(inputs)
        torch.manual_seed(SEED)
        first = runner.run_stage2(stage1).action_pred.float().cpu()
        torch.manual_seed(SEED)
        second = runner.run_stage2(stage1).action_pred.float().cpu()
    assert torch.equal(first, second)


def test_running_outside_autocast_changes_the_numbers(policy, runner):
    """The control for the session contract: if this passed, the guard would be pointless."""
    inputs = _normalized(policy, _observation())

    torch.manual_seed(SEED)
    with runner.session():
        inside = runner.run_stage2(runner.run_stage1(inputs)).action_pred.float().cpu()

    # Bypass the guard deliberately to measure what it is protecting against.
    torch.manual_seed(SEED)
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16):
        stage1 = runner.run_stage1(inputs)
    torch.manual_seed(SEED)
    with torch.inference_mode():
        backbone_inputs, action_inputs = policy.model.prepare_input(inputs)
        del backbone_inputs
        outputs = policy.model.backbone.eagle_model.language_model(
            inputs_embeds=stage1.input_embeds,
            attention_mask=stage1.attention_mask,
            position_ids=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=True,
        )
        features = outputs.hidden_states[policy.model.backbone.select_layer]
        features = policy.model.backbone.eagle_linear(features)
        from transformers.feature_extraction_utils import BatchFeature

        no_autocast = policy.model.action_head.get_action(
            BatchFeature(
                data={
                    "backbone_features": features,
                    "backbone_attention_mask": stage1.attention_mask,
                }
            ),
            action_inputs,
        )["action_pred"].float().cpu()

    assert not torch.equal(inside, no_autocast), (
        "autocast made no difference; LayerNorm is no longer being promoted to "
        "fp32 and the session contract has lost its reason to exist"
    )


def test_stage1_guard_refuses_to_run_without_a_session(policy, runner):
    inputs = _normalized(policy, _observation())
    with pytest.raises(RuntimeError, match="must run inside"):
        runner.run_stage1(inputs)


# ------------------------------------------------------------------
# Real token layout
# ------------------------------------------------------------------


def test_image_runs_differ_from_the_pi05_fixed_offset_table(policy, runner):
    """Mask-derived runs are 3x256 contiguous and NOT at pi0.5's 0/256/512.

    The first real-machine A/B corrected an earlier belief: this chat template
    places the instruction AFTER the image blocks, so the image offsets do NOT
    move with prompt length — the TOTAL sequence length does (measured
    812/814/834 for three prompts, runs pinned at 20/283/546 in all of them).
    What makes mask-driven slicing load-bearing is therefore not offset
    motion but that the true offsets differ from the pi0.5 fixed table
    (0/256/512): a copied table would slice system/text tokens while every
    shape stays plausible.
    """
    from openpi.cache.groot.key_builder import _contiguous_runs

    short = _normalized(policy, _observation())
    obs_long = _observation()
    for key in ("annotation.human.task_description",):
        obs_long[key] = (
            "please carefully open the leftmost cabinet door and then wait "
            "patiently for further detailed instructions in this kitchen"
        )
    long = _normalized(policy, obs_long)

    with runner.session():
        s_short = runner.run_stage1(short)
        s_long = runner.run_stage1(long)

    # The prompt genuinely changes the sequence — so an equal-runs outcome
    # below is a statement about the template, not a fixture that fed the
    # same input twice.
    len_short = int(s_short.image_token_mask.shape[-1])
    len_long = int(s_long.image_token_mask.shape[-1])
    assert len_short != len_long, "the long prompt did not change the sequence length"

    runs_short = _contiguous_runs(s_short.image_token_mask[0])
    runs_long = _contiguous_runs(s_long.image_token_mask[0])
    for runs in (runs_short, runs_long):
        assert len(runs) == 3
        assert all(length == 256 for _, length in runs)
    # Template-constant w.r.t. the instruction (it follows the images).
    assert runs_short == runs_long

    starts = [start for start, _ in runs_short]
    assert starts != [0, 256, 512], (
        "image runs sit exactly at the pi0.5 fixed offset table; the "
        "mask-vs-table distinction this suite protects would be vacuous"
    )


def test_scattered_positions_hold_the_vision_output(policy, runner):
    inputs = _normalized(policy, _observation())
    with runner.session():
        stage1 = runner.run_stage1(inputs)
        backbone_inputs, _ = policy.model.prepare_input(inputs)
        vit = policy.model.backbone.eagle_model.extract_feature(
            backbone_inputs["eagle_pixel_values"]
        )
    scattered = stage1.input_embeds[0][stage1.image_token_mask[0]]
    assert torch.equal(scattered, vit.reshape(-1, vit.shape[-1]))


# ------------------------------------------------------------------
# G0-D2: online keys still retrieve their own step from the offline library
# ------------------------------------------------------------------


def _trace_build_episode(policy, runner, out_dir, *, n_steps: int, seed: int):
    """Run ``n_steps`` decisions through the trace build form (no orchestrator).

    Returns the per-step wire outputs and the produced H5 path. The trace
    replaces the legacy ``GrootCacheCollector``; this is the island-B gate
    that its file is the same file (plan §12-J).
    """
    from exp.robocasa365.groot_policy_adapter import build_groot_observation
    from openpi.cache.groot.interceptor import GrootCacheInterceptor
    from openpi.cache.trace.h5_sink import H5TraceSink, TraceWriter
    from openpi.cache.trace.types import TracePlan, TraceRuntime

    schedule = runner.live_schedule()
    plan = TracePlan(
        model="groot_n15", schedule=schedule, checkpoint=None, warm_tiers=(),
        record_noise_actions=True, save_timesteps=schedule.timesteps,
        record_prefix_tokens=True, record_raw_images=True, record_model_images=False,
        record_query_keys=True, record_search=True, record_tokenized_prompt=True,
        raw_image_keys=(), rng_isolation="verdict_aware", sidecar_jsonl=True, fail_loud=True,
    )
    writer = TraceWriter(str(out_dir), queue_steps=8)
    rt = TraceRuntime(plan=plan, sink=H5TraceSink(out_dir, plan=plan, writer=writer), twins=None)
    it = GrootCacheInterceptor(
        policy, runner, trace=rt, trace_vision_fields=("vision_0", "vision_1", "vision_2")
    )
    it.on_task_begin()
    it.on_episode_start("manual", "ManualParity", 0, "ep0", {"task_uid": "u0", "attempt": 1})
    outs = []
    for step in range(n_steps):
        torch.manual_seed(seed + step)
        outs.append(it.get_action(build_groot_observation(_observation(step))))
    it.on_episode_end(True)
    it.on_task_end()
    report = writer.stop(timeout=30)
    assert report.ok, report
    return outs, pathlib.Path(out_dir) / "manual" / "ep0.h5"


def test_trace_build_matches_the_legacy_hook_capture(policy, runner, tmp_path):
    """Same seed: the trace file holds what the legacy collector's hook saw.

    The legacy ``GrootCacheCollector`` captured ``action_encoder``'s input on
    every Euler step around upstream's own ``get_action`` (noise drawn
    inside). The trace draws that noise itself (``sample_noise``) and runs
    the transcribed loop; under one seed both must consume the same noise
    and produce the same snapshots and the same chunk.
    """
    import h5py

    from openpi.cache.groot.key_builder import slice_groot_cp1_fields

    seed = SEED + 7
    n_steps = 3
    outs, h5 = _trace_build_episode(policy, runner, tmp_path / "trace", n_steps=n_steps, seed=seed)
    schedule = runner.live_schedule()
    with h5py.File(h5, "r") as f:
        assert f.attrs["denoise_schedule_id"] == schedule.schedule_id
        assert int(f.attrs["num_steps"]) == n_steps
        for step in range(n_steps):
            inputs = _normalized(policy, _observation(step))
            captures: list[torch.Tensor] = []
            handle = policy.model.action_head.action_encoder.register_forward_hook(
                lambda m, i, o: captures.append(i[0].detach().clone())
            )
            try:
                torch.manual_seed(seed + step)
                with runner.session():
                    stage1 = runner.run_stage1(inputs)
                    stage2 = runner.run_stage2(stage1)  # upstream get_action, noise inside
            finally:
                handle.remove()
            assert len(captures) == schedule.num_steps
            raw = slice_groot_cp1_fields(
                stage1.input_embeds, stage1.image_token_mask, stage1.state, stage1.state_mask, None
            )
            g = f[f"step_{step:04d}"]
            for i, name in enumerate(("vision_0", "vision_1", "vision_2")):
                np.testing.assert_array_equal(g[f"vision_{i}"][...], raw[name].cpu().to(torch.float16).numpy())
            np.testing.assert_array_equal(g["prompt_emb"][...], raw["prompt_emb"].cpu().to(torch.float16).numpy())
            np.testing.assert_array_equal(g["robot_state"][...], raw["robot_state"].cpu().float().numpy())
            for i, x in enumerate(captures):
                np.testing.assert_array_equal(g[f"noise_action_{i}"][...], x[0].cpu().float().numpy())
            assert f"noise_action_{schedule.num_steps}" not in g
            np.testing.assert_array_equal(g["clean_action"][...], stage2.action_pred[0].cpu().float().numpy())
            served = policy.unapply_transforms({"action": stage2.action_pred[0].detach().cpu().float()[None]})
            for key, value in served.items():
                np.testing.assert_array_equal(np.asarray(outs[step][key])[None], np.asarray(value))


def test_online_and_offline_keys_retrieve_the_same_entry(policy, runner, tmp_path):
    from exp.common.build_in_memory_cache_artifact import build_artifact
    from exp.robocasa365.groot_key_parity import check_key_parity
    from openpi.cache.groot.key_builder import GrootCP1SpatialPool16KeyBuilder
    from openpi.cache.types import CheckpointID

    n_steps = 6
    data_dir = tmp_path / "episodes"
    _, h5 = _trace_build_episode(policy, runner, data_dir, n_steps=n_steps, seed=SEED)
    # A trace build only reads an audited selection (plan §7.3): audit, then list.
    from exp.robocasa365.verify_collection_artifacts import _check_h5_schema

    problems = _check_h5_schema(h5, "ManualParity", require_schedule=True)
    assert problems == [], problems
    selected = tmp_path / "audited_episodes.txt"
    selected.write_text(h5.name + "\n")

    online_keys = []
    builder = GrootCP1SpatialPool16KeyBuilder()
    for step in range(n_steps):
        inputs = _normalized(policy, _observation(step))
        with runner.session():
            stage1 = runner.run_stage1(inputs)
        builder.collect(CheckpointID.CP1, stage1=stage1)
        online_keys.append(builder.build(CheckpointID.CP1))

    artifact = build_artifact(
        str(h5.parent), "cp1_groot_spatial_pool_16", "CP1", workers=-1, episode_list=str(selected)
    )
    entries = sorted(artifact["entries"], key=lambda e: e.step_idx)
    assert len(entries) == n_steps
    assert artifact["schedule_id"] == runner.live_schedule().schedule_id
    offline_keys = [
        {k: torch.as_tensor(v).float() for k, v in entry.query_keys.items()}
        for entry in entries
    ]

    metrics = {
        "vision_0": "cosine",
        "vision_1": "cosine",
        "vision_2": "cosine",
        "prompt_emb": "cosine",
        "robot_state": "l2",
    }
    report = check_key_parity(online_keys, offline_keys, metrics)
    print("\n" + report.summary())
    assert report.passed, report.summary()


def test_concurrent_trace_batches_same_shape_and_splits_lengths(policy, runner, tmp_path):
    """Three connections trace-build concurrently through one stage-3 core (plan §12-I / §12-K).

    Connections 0 and 1 share a prompt but see different images and state, so
    their conditioning has one shape with different content and must meet in a
    single stage-3 forward; connection 2 has a longer prompt and must never
    share a bucket with them (no padding). Every stage-3 call is reproduced
    directly with its composition and order, and the served replies must be
    bit-identical to it (no cross-talk, exact split). Against the batch-1 loop
    a B>1 call differs only by kernel numerics; that drift must stay an order of
    magnitude below the spread two noise draws give on the same conditioning --
    the noise-floor rule of the online-RIT real-model gate (a bf16 batch changes
    GEMM tiling, so a fixed relative tolerance is not the right yardstick).
    """
    import threading

    import h5py

    from exp.robocasa365 import groot_keys
    from exp.robocasa365.groot_policy_adapter import build_groot_observation
    from openpi.cache.groot import batcher as gb
    from openpi.cache.groot.interceptor import GrootCacheInterceptor
    from openpi.cache.groot.staged import GrootStagedRunner
    from openpi.cache.trace.h5_sink import H5TraceSink, TraceWriter
    from openpi.cache.trace.types import TracePlan, TraceRuntime
    from openpi.serving.batching_core import BatchingCore

    prompts = (
        "pick up the object",
        "pick up the object",
        "pick up the object from the counter and place it in the cabinet next to the sink",
    )
    n_steps = 2
    schedule = runner.live_schedule()
    plan = TracePlan(
        model="groot_n15", schedule=schedule, checkpoint=None, warm_tiers=(),
        record_noise_actions=True, save_timesteps=schedule.timesteps,
        record_prefix_tokens=True, record_raw_images=True, record_model_images=False,
        record_query_keys=True, record_search=True, record_tokenized_prompt=True,
        raw_image_keys=(), rng_isolation="verdict_aware", sidecar_jsonl=True, fail_loud=True,
        concurrent=True,
    )
    out_dir = tmp_path / "trace"
    writer = TraceWriter(str(out_dir), queue_steps=16)
    batcher = gb.GrootStageBatcher(runner)
    calls: list[list[tuple[int, float]]] = []  # per call: (conditioning length, noise checksum) in order
    real_miss = batcher.run_stage3_miss

    def spy(payloads, **kw):
        calls.append([
            (int(p.stage2_out.stage2.backbone_features.shape[1]), float(p.noise.double().sum()))
            for p in payloads
        ])
        return real_miss(payloads, **kw)

    batcher.run_stage3_miss = spy

    def obs_for(conn: int, step: int) -> dict:
        obs = _observation(step + 10 * conn)  # different images and state per connection
        for key in groot_keys.LANGUAGE_KEYS:
            obs[key] = prompts[conn]
        return obs

    lock = threading.Lock()
    barrier = threading.Barrier(len(prompts))
    errors: list[BaseException] = []

    with BatchingCore(batcher, device="cuda", max_batch_size=8, max_wait_ms=300.0) as core:

        def connection(i: int) -> None:
            try:
                rt = TraceRuntime(plan=plan, sink=H5TraceSink(out_dir, plan=plan, writer=writer))
                it = GrootCacheInterceptor(
                    policy, GrootStagedRunner(policy.model), trace=rt, coordinator=core,
                    model_lock=lock, trace_vision_fields=("vision_0", "vision_1", "vision_2"),
                )
                it.on_task_begin()
                it.on_episode_start("manual_conc", "ManualConcurrent", i, f"c{i}", {"task_uid": f"u{i}", "attempt": 1})
                for step in range(n_steps):
                    barrier.wait(timeout=300)
                    it.get_action(build_groot_observation(obs_for(i, step)))
                it.on_episode_end(True)
                it.on_task_end()
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                errors.append(exc)
                barrier.abort()

        threads = [threading.Thread(target=connection, args=(i,)) for i in range(len(prompts))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(600)
    assert not errors, errors
    report = writer.stop(timeout=60)
    assert report.ok, report

    # What each connection's decision was served from, read back from its file.
    served: dict[tuple[int, int], dict] = {}
    for i in range(len(prompts)):
        with h5py.File(out_dir / "manual_conc" / f"c{i}.h5", "r") as f:
            assert int(f.attrs["num_steps"]) == n_steps and bool(f.attrs["trace_terminal"])
            for step in range(n_steps):
                g = f[f"step_{step:04d}"]
                inputs = _normalized(policy, obs_for(i, step))
                with runner.session():
                    stage2 = runner.run_stage2_llm(runner.run_stage1(inputs))
                    probe = runner.sample_noise(stage2, generator=torch.Generator(device="cuda").manual_seed(0))
                noise = torch.from_numpy(g["noise_action_0"][...]).to(device=probe.device, dtype=probe.dtype)
                served[(i, step)] = {
                    "stage2": stage2,
                    "noise": noise,
                    "length": int(stage2.backbone_features.shape[1]),
                    "clean": g["clean_action"][...],
                }
    assert served[(0, 0)]["length"] == served[(1, 0)]["length"] != served[(2, 0)]["length"]
    print(f"\nstage-3 calls (conditioning length per payload): {[[n for n, _ in c] for c in calls]}")
    assert all(len({n for n, _ in c}) == 1 for c in calls), calls  # never a mixed-length bucket
    assert any(len(c) >= 2 for c in calls), calls  # same shape met in one forward
    assert sum(len(c) for c in calls) == len(prompts) * n_steps

    def member(noise_sum: float) -> tuple[int, int]:
        (key,) = [k for k, v in served.items() if abs(float(v["noise"].double().sum()) - noise_sum) < 1e-6]
        return key

    def loop(stage2s, noises) -> np.ndarray:
        out, _ = gb.run_miss(
            runner, gb.cat_stage2([gb.stage3_input(runner, s) for s in stage2s]),
            torch.stack(noises), schedule=schedule, capture=True,
        )
        return out.action_pred.float().cpu().numpy()

    for call in calls:
        keys = [member(noise_sum) for _, noise_sum in call]
        direct = loop([served[k]["stage2"] for k in keys], [served[k]["noise"] for k in keys])
        for row, k in enumerate(keys):
            np.testing.assert_array_equal(served[k]["clean"], direct[row])  # exact, no cross-talk
        if len(keys) < 2:
            continue
        for k in keys:
            other = next(o for o in keys if o != k)
            serial = loop([served[k]["stage2"]], [served[k]["noise"]])[0]
            spread = loop([served[k]["stage2"]], [served[other]["noise"]])[0]
            drift = float(np.linalg.norm(served[k]["clean"] - serial))
            floor = float(np.linalg.norm(spread - serial))
            print(f"{k}: batch drift {drift:.4f}, noise spread {floor:.4f}")
            assert drift <= 0.1 * floor, (k, drift, floor)
