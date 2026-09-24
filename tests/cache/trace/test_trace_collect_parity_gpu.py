"""GPU parity gate: the trace build file vs the legacy forward-hook capture (plan §12-J).

The legacy ``--collect`` recorded the prefix tokens and the loop inputs with
forward hooks on the real Pi0.5 model; the trace build slices the same
tensors from the staged API. This gate proves, on the real checkpoint and
the real GPU, that every field of the trace file is what the hooks would
have captured under the same observation and the same explicit noise:
``vision_*`` / ``prompt_emb`` (fp16, bitwise), ``robot_state``,
``noise_action_0..9`` (bitwise), ``clean_action`` and the served action, and
that the file passes the collection auditor and builds into a library that
loads. Deleting the legacy collector is conditional on this gate (plan D7).

Run (fails -- never skips -- on a missing checkpoint or GPU once
``--run-manual`` is passed)::

    uv run pytest tests/cache/trace/test_trace_collect_parity_gpu.py --run-manual -q
"""

from __future__ import annotations

import gc
import math
import pathlib
import pickle

import numpy as np
import pytest

pytestmark = pytest.mark.manual

CKPT = pathlib.Path("/home/weiland/ckpt_pi05_robocasa_pytorch")
EVIDENCE = pathlib.Path(__file__).resolve().parents[3] / "exp" / "robocasa365" / "analysis" / "trace_collect_parity.txt"


def _observation() -> dict:
    rng = np.random.default_rng(1234)
    frame = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
    return {
        "observation/state": rng.standard_normal(16),
        "observation/image": frame,
        "observation/wrist_image": frame[::2, ::2].copy(),
        "observation/right_image": frame[::2, ::2].copy(),
        "prompt": "open the cabinet",
    }


@pytest.fixture(scope="module")
def _gate_environment():
    if not CKPT.is_dir():
        pytest.fail(f"checkpoint dir missing: {CKPT} -- run on weilandserver")
    import torch

    if not torch.cuda.is_available():
        pytest.fail("CUDA unavailable -- the parity gate must run on the serving GPU")


def _hook_capture(policy, obs: dict, noise: np.ndarray) -> dict:
    """The legacy collector's capture, transcribed (same hooks, same source tensors)."""
    import torch

    model = policy._model  # noqa: SLF001
    vision, lang, a_in, a_out = [], [None], [], []
    handles = [
        model.paligemma_with_expert.paligemma.multi_modal_projector.register_forward_hook(
            lambda m, i, o: vision.append(o.detach().clone())
        ),
        model.paligemma_with_expert.paligemma.language_model.embed_tokens.register_forward_hook(
            lambda m, i, o: lang.__setitem__(0, (o * math.sqrt(o.shape[-1])).detach().clone())
        ),
        model.action_in_proj.register_forward_hook(lambda m, i, o: a_in.append(i[0].detach().clone())),
        model.action_out_proj.register_forward_hook(lambda m, i, o: a_out.append(o.detach().clone())),
    ]
    try:
        actions = np.asarray(policy.infer(dict(obs), noise=noise)["actions"])
    finally:
        for h in handles:
            h.remove()
    assert len(a_in) == 10 and len(a_out) == 10
    dt = -1.0 / 10
    x_last = a_in[-1].squeeze(0).cpu().float()
    v_last = a_out[-1].squeeze(0).cpu().float()
    return {
        "vision": [v.squeeze(0).cpu().to(torch.float16).numpy() for v in vision],
        "prompt_emb": lang[0].squeeze(0).cpu().to(torch.float16).numpy(),
        "init_noise": a_in[0].squeeze(0).cpu().numpy().astype(np.float32),
        "noise_steps": [a_in[i].squeeze(0).cpu().numpy().astype(np.float32) for i in range(1, 10)],
        "clean_action": (x_last + dt * v_last).numpy().astype(np.float32),
        "actions": actions,
    }


@pytest.fixture(scope="module")
def _artifacts(_gate_environment, tmp_path_factory):
    import torch

    from openpi.cache.interceptor import InferenceInterceptor
    from openpi.cache.timing import SystemTimer
    from openpi.cache.trace.h5_sink import H5TraceSink, TraceWriter
    from openpi.cache.trace.types import TracePlan, TraceRuntime
    from openpi.cache.types import PI05_V1
    from openpi.policies import policy_config as _pc
    from openpi.training import config as _config

    obs = _observation()
    noise = np.random.default_rng(0).standard_normal((50, 32)).astype(np.float32)
    out_dir = tmp_path_factory.mktemp("trace_parity")

    policy = _pc.create_trained_policy(_config.get_config("pi05_robocasa"), CKPT, pytorch_device="cuda")
    legacy = _hook_capture(policy, obs, noise)

    plan = TracePlan(
        model="pi05", schedule=PI05_V1, checkpoint=None, warm_tiers=(),
        record_noise_actions=True, save_timesteps=PI05_V1.timesteps,
        record_prefix_tokens=True, record_raw_images=True, record_model_images=True,
        record_query_keys=True, record_search=True, record_tokenized_prompt=True,
        raw_image_keys=("observation/image", "observation/wrist_image", "observation/right_image"),
        rng_isolation="verdict_aware", sidecar_jsonl=True, fail_loud=True,
    )
    writer = TraceWriter(str(out_dir), queue_steps=8)
    rt = TraceRuntime(plan=plan, sink=H5TraceSink(out_dir, plan=plan, writer=writer), twins=None)
    it = InferenceInterceptor(policy, timer=SystemTimer(enabled=False), eager=True, trace=rt)
    it.on_task_begin()
    it.on_episode_start("parity", "open the cabinet", 1, "ep1", {"task_uid": "u1", "attempt": 1})
    traced_actions = np.asarray(it.infer(dict(obs), noise=noise)["actions"])
    it.on_episode_end(True)
    it.on_task_end()
    report = writer.stop(timeout=30)
    assert report.ok, report

    del it, policy
    gc.collect()
    torch.cuda.empty_cache()
    return {"legacy": legacy, "traced_actions": traced_actions, "h5": out_dir / "parity" / "ep1.h5"}


def test_prefix_tokens_and_loop_inputs_are_bitwise_equal(_artifacts):
    import h5py

    legacy = _artifacts["legacy"]
    with h5py.File(_artifacts["h5"], "r") as f:
        g = f["step_0000"]
        assert f.attrs["denoise_schedule_id"] == "pi05_v1" and f.attrs["denoising_num_steps"] == 10
        for i, v in enumerate(legacy["vision"]):
            np.testing.assert_array_equal(g[f"vision_{i}"][...], v)
        assert f"vision_{len(legacy['vision'])}" not in g
        np.testing.assert_array_equal(g["prompt_emb"][...], legacy["prompt_emb"])
        np.testing.assert_array_equal(g["noise_action_0"][...], legacy["init_noise"])
        for i, x in enumerate(legacy["noise_steps"], start=1):
            np.testing.assert_array_equal(g[f"noise_action_{i}"][...], x)
        assert "noise_action_10" not in g
        # The legacy clean action was reconstructed on the CPU from the last
        # hook pair; the trace stores the model's own output (same numbers up
        # to the fp32 reconstruction rounding).
        np.testing.assert_allclose(g["clean_action"][...], legacy["clean_action"], atol=1e-5, rtol=1e-5)
        assert g["robot_state"].shape == (32,)
        assert "input_images" in g and len(g["input_images"]) == 3  # legacy slots present
    np.testing.assert_array_equal(_artifacts["traced_actions"], legacy["actions"])


def test_trace_file_passes_the_auditor_and_builds_a_library(_artifacts, tmp_path):
    from exp.common.build_in_memory_cache_artifact import build_artifact
    from exp.robocasa365.verify_collection_artifacts import _check_h5_schema
    from openpi.cache.backends.in_memory_backend import InMemoryBackend

    h5 = _artifacts["h5"]
    problems = _check_h5_schema(h5, "open the cabinet", require_schedule=True)
    assert problems == [], problems
    selected = tmp_path / "audited_episodes.txt"
    selected.write_text(h5.name + "\n")
    artifact = build_artifact(str(h5.parent), "cp1_mean_pool", workers=-1, device="cpu",
                              episode_list=str(selected))
    assert artifact["entries"], "no entries built from the trace file"
    pkl = tmp_path / "trace_parity.pkl"
    with open(pkl, "wb") as fh:
        pickle.dump(artifact, fh)
    backend = InMemoryBackend(artifact["vector_dims"])
    backend.load_artifact(str(pkl))
    entry = artifact["entries"][0]
    assert entry.payload.schedule_id == "pi05_v1" and len(entry.payload.intermediates) == 9
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(
        "trace vs legacy-hook parity: PASS\n"
        f"checkpoint: {CKPT}\nentries built: {len(artifact['entries'])}\n"
    )
