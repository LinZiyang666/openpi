"""Contract tests for the W1 three-stage CUDA-Graph benchmark.

Everything here runs without a GPU, a checkpoint, or the gr00t package: the
subject is the benchmark's *certification logic*, which is where a wrong answer
would be dangerous. A latency number that is merely noisy gets noticed; a cell
that certifies itself valid on empty evidence does not, and that is precisely
the failure mode plan §5 G-M was written to prevent.

The real-model parity and trace tests live in the island-B manual suite; they
cannot run here because the model is not importable in this environment.
"""

from __future__ import annotations

import inspect
import json
import pathlib
from types import SimpleNamespace

import pytest
import torch

from exp.robocasa365 import bench_groot_stages as bench


RAW_CELL_IDENTITY = {
    "schedule_id": bench.SCHEDULE_ID,
    "k": 1,
    "prompt_sha256": "prompt",
    "proc_idx": 0,
    "mode": bench.COMPILE_MODE,
    "warmup": 1,
    "iters": 1,
    "seed": 0,
    "expected_cudagraph_launch_count": 3,
    "ckpt_sha256": "checkpoint",
    "gpu_uuid": "GPU-test",
    "ts": "2026-09-06T00:00:00.000001-05:00",
}
TRACE_MARKER = bench.trace_marker(RAW_CELL_IDENTITY)


def _trace(*api_rows: str, marker: str = TRACE_MARKER) -> str:
    """Build the combined CUDA-API/NVTX export consumed by certification."""
    return (
        "Name,Num Calls\n"
        + "\n".join(api_rows)
        + "\n\nTime (%),Total Time (ns),Instances,Range\n"
        + f"100.0,1,1,{marker}\n"
    )


# ---------------------------------------------------------------------------
# argument validation (non-blocking suggestion NB1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["0", "-1", "-200"])
def test_positive_int_rejects_non_positive(bad: str) -> None:
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        bench.positive_int(bad)


def test_positive_int_accepts_positive() -> None:
    assert bench.positive_int("30") == 30


# ---------------------------------------------------------------------------
# atomic artifact write
# ---------------------------------------------------------------------------


def test_atomic_write_json_leaves_no_temp_file(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "cell.json"
    bench.atomic_write_json(out, {"a": 1})
    assert json.loads(out.read_text()) == {"a": 1}
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_json_replaces_previous_record(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "cell.json"
    bench.atomic_write_json(out, {"valid": False})
    bench.atomic_write_json(out, {"valid": True})
    assert json.loads(out.read_text()) == {"valid": True}


# ---------------------------------------------------------------------------
# trace parsing -- the "file exists" hole this replaced
# ---------------------------------------------------------------------------


def test_parse_cuda_trace_missing_file_fails_closed(tmp_path: pathlib.Path) -> None:
    with pytest.raises(SystemExit):
        bench.parse_cuda_trace(tmp_path / "nope.csv")


def test_parse_cuda_trace_empty_file_is_not_evidence(tmp_path: pathlib.Path) -> None:
    """An empty file used to satisfy the old existence-only check."""
    trace = tmp_path / "empty.csv"
    trace.write_text("")
    with pytest.raises(SystemExit):
        bench.parse_cuda_trace(trace)


def test_parse_cuda_trace_counts_launches_and_captures(tmp_path: pathlib.Path) -> None:
    trace = tmp_path / "t.csv"
    trace.write_text(
        "Time (%),Total Time (ns),Num Calls,Avg (ns),Name\n"
        '90.0,1000,37,27.0,"cudaGraphLaunch"\n'
        '10.0,100,2,50.0,"cudaStreamBeginCapture"\n'
    )
    counts = bench.parse_cuda_trace(trace)
    assert counts["cudagraph_launch_count"] == 37
    assert counts["graph_capture_calls"] == 2


def test_parse_cuda_trace_ignores_following_nvtx_table(tmp_path: pathlib.Path) -> None:
    trace = tmp_path / "combined.csv"
    trace.write_text(_trace("cudaGraphLaunch,2"))
    counts = bench.parse_cuda_trace(trace, expected_marker=TRACE_MARKER)
    assert counts == {"cudagraph_launch_count": 2, "graph_capture_calls": 0}


# ---------------------------------------------------------------------------
# certification -- a cell must not be able to certify itself
# ---------------------------------------------------------------------------


def _raw_cell(**over: object) -> dict:
    cell = dict.fromkeys(bench.MEASURE_RECORD_FIELDS)
    cell.update(
        {
            "valid": False,
            "certified": False,
            "void_reasons": ["uncertified: no CUDA trace parsed"],
            "cudagraph_launch_count": None,
            "capture_calls_after_warmup": None,
            "expected_cudagraph_launch_count": 3,
            "host": bench.EXPECTED_HOST,
            **RAW_CELL_IDENTITY,
        }
    )
    cell["trace_marker"] = bench.trace_marker(cell)
    cell.update(over)
    return cell


def test_certify_clears_only_the_uncertified_reason(tmp_path: pathlib.Path) -> None:
    rec = tmp_path / "cell.json"
    bench.atomic_write_json(rec, _raw_cell())
    trace = tmp_path / "t.csv"
    trace.write_text(_trace("cudaGraphLaunch,3"))

    out = bench.certify_cell(rec, trace)
    assert out["certified"] is True
    assert out["valid"] is True
    assert out["cudagraph_launch_count"] == 3
    assert out["void_reasons"] == []


def test_certify_keeps_unrelated_void_reasons(tmp_path: pathlib.Path) -> None:
    """Certification must never launder a parity or RNG failure into validity."""
    rec = tmp_path / "cell.json"
    bench.atomic_write_json(
        rec,
        _raw_cell(
            void_reasons=[
                "stage1 parity cos_min=0.871600",
                "uncertified: no CUDA trace parsed",
            ]
        ),
    )
    trace = tmp_path / "t.csv"
    trace.write_text(_trace("cudaGraphLaunch,3"))

    out = bench.certify_cell(rec, trace)
    assert out["valid"] is False
    assert out["void_reasons"] == ["stage1 parity cos_min=0.871600"]


def test_certify_without_graph_launch_is_void(tmp_path: pathlib.Path) -> None:
    """A trace that never launched a graph proves the opposite of what is claimed."""
    rec = tmp_path / "cell.json"
    bench.atomic_write_json(rec, _raw_cell())
    trace = tmp_path / "t.csv"
    trace.write_text(_trace("cudaLaunchKernel,3", "cudaMemcpyAsync,2"))

    out = bench.certify_cell(rec, trace)
    assert out["valid"] is False
    assert any("no cudaGraphLaunch" in r for r in out["void_reasons"])
    assert out["cudagraph_launch_count"] == 0


def test_certify_rejects_missing_stage_replays(tmp_path: pathlib.Path) -> None:
    rec = tmp_path / "cell.json"
    bench.atomic_write_json(rec, _raw_cell())
    trace = tmp_path / "t.csv"
    trace.write_text(_trace("cudaGraphLaunch,2"))

    out = bench.certify_cell(rec, trace)
    assert out["valid"] is False
    assert any("count mismatch" in reason for reason in out["void_reasons"])


def test_certify_rejects_capture_in_measurement_range(tmp_path: pathlib.Path) -> None:
    rec = tmp_path / "cell.json"
    bench.atomic_write_json(rec, _raw_cell())
    trace = tmp_path / "t.csv"
    trace.write_text(_trace("cudaGraphLaunch,3", "cudaStreamBeginCapture,1"))

    out = bench.certify_cell(rec, trace)
    assert out["valid"] is False
    assert out["capture_calls_after_warmup"] == 1
    assert any("capture" in reason.lower() for reason in out["void_reasons"])


def test_certify_rejects_trace_from_another_cell(tmp_path: pathlib.Path) -> None:
    rec = tmp_path / "cell.json"
    bench.atomic_write_json(rec, _raw_cell())
    trace = tmp_path / "other-cell.csv"
    trace.write_text(_trace("cudaGraphLaunch,3", marker="openpi_w1_other"))

    with pytest.raises(SystemExit, match="does not contain cell marker"):
        bench.certify_cell(rec, trace)


def test_certify_rejects_tampered_cell_identity(tmp_path: pathlib.Path) -> None:
    rec = tmp_path / "cell.json"
    bench.atomic_write_json(rec, _raw_cell(prompt_sha256="mutated-after-measurement"))
    trace = tmp_path / "trace.csv"
    trace.write_text(_trace("cudaGraphLaunch,3"))

    with pytest.raises(SystemExit, match="does not match its identity"):
        bench.certify_cell(rec, trace)


@pytest.mark.parametrize("expected", [None, 0, True, "2"])
def test_certify_requires_frozen_expected_replay_count(
    tmp_path: pathlib.Path, expected: object
) -> None:
    rec = tmp_path / "cell.json"
    cell = _raw_cell(expected_cudagraph_launch_count=expected)
    # Model a raw record written this way rather than a later mutation.
    cell["trace_marker"] = bench.trace_marker(cell)
    bench.atomic_write_json(rec, cell)
    trace = tmp_path / "trace.csv"
    trace.write_text(_trace("cudaGraphLaunch,3", marker=cell["trace_marker"]))

    with pytest.raises(SystemExit, match="positive integer"):
        bench.certify_cell(rec, trace)


# ---------------------------------------------------------------------------
# host binding
# ---------------------------------------------------------------------------


def test_assert_host_rejects_the_wrong_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench.socket, "gethostname", lambda: "some-other-host")
    with pytest.raises(SystemExit):
        bench.assert_host()


def test_assert_host_accepts_only_frozen_machine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bench.socket, "gethostname", lambda: "weilandserver")
    assert bench.assert_host() == "weilandserver"


def test_assert_host_accepts_frozen_machine_fqdn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bench.socket, "gethostname", lambda: "weilandserver.example.test"
    )
    assert bench.assert_host() == "weilandserver.example.test"


def test_assert_host_rejects_substring_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        bench.socket, "gethostname", lambda: "not-weilandserver.example.test"
    )
    with pytest.raises(SystemExit):
        bench.assert_host()


def test_measurement_gpu_guard_checks_both_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        bench,
        "assert_idle_gpu",
        lambda device_index, own_pid: calls.append((device_index, own_pid)),
    )
    with bench.measurement_gpu_guard(2, allow_busy=False):
        pass
    assert [device for device, _ in calls] == [2, 2]


def test_cuda_profiler_range_brackets_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    cudart = SimpleNamespace(
        cudaProfilerStart=lambda: calls.append("start") or 0,
        cudaProfilerStop=lambda: calls.append("stop") or 0,
    )
    monkeypatch.setattr(bench.torch.cuda, "cudart", lambda: cudart)
    with bench.cuda_profiler_range():
        calls.append("measure")
    assert calls == ["start", "measure", "stop"]


def test_nvtx_measurement_range_binds_trace_to_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        bench.torch.cuda,
        "nvtx",
        SimpleNamespace(
            range_push=lambda marker: calls.append(("push", marker)),
            range_pop=lambda: calls.append(("pop", None)),
        ),
    )
    with bench.nvtx_measurement_range(TRACE_MARKER):
        calls.append(("measure", None))
    assert calls == [
        ("push", TRACE_MARKER),
        ("measure", None),
        ("pop", None),
    ]


def test_trace_marker_changes_with_cell_identity() -> None:
    record = {
        "schedule_id": bench.SCHEDULE_ID,
        "k": 4,
        "prompt_sha256": "prompt-a",
        "proc_idx": 0,
        "ckpt_sha256": "checkpoint",
        "gpu_uuid": "GPU-1",
        "ts": "2026-09-06T00:00:00.000001-05:00",
        "expected_cudagraph_launch_count": 6,
    }
    marker = bench.trace_marker(record)
    assert marker.startswith("openpi_w1_")
    assert marker == bench.trace_marker(record)
    assert marker != bench.trace_marker({**record, "prompt_sha256": "prompt-b"})


def test_compile_stages_requires_three_full_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict]] = []

    def fake_compile(fn, **kwargs):
        calls.append((fn.__name__, kwargs))
        return fn

    class Runner:
        _model = object()
        _eagle = object()
        _backbone = object()

    monkeypatch.setattr(bench.torch, "compile", fake_compile)
    stages = bench.compile_stages(Runner(), torch.tensor([0]))
    assert set(stages) == {
        "compiled_stage1",
        "compiled_stage2_llm",
        "compiled_denoise_step",
    }
    assert [name for name, _ in calls] == [
        "stage1_full",
        "stage2_llm",
        "denoise_step",
    ]
    assert all(kwargs["fullgraph"] is True for _, kwargs in calls)


def test_stage1_copy_has_no_dynamo_graph_break() -> None:
    from openpi.cache.groot.staged import GrootStagedRunner
    from tests.cache.groot.conftest import StubGrootModel

    model = StubGrootModel()
    runner = GrootStagedRunner(model, verify_upstream=False, compile_vision=False)
    normalized = model.build_inputs()
    with runner.session():
        eager = runner.run_stage1(normalized)
        positions = torch.nonzero(
            eager.image_token_mask.reshape(-1), as_tuple=False
        ).flatten()
        stage1, _ = bench._stage_callables(runner, positions)
        compiled = torch.compile(stage1, backend="eager", fullgraph=True, dynamic=False)
        values = compiled(normalized)
    assert torch.equal(values[0], eager.input_embeds)
    assert torch.equal(values[2], eager.image_token_mask)


def test_stage1_parity_checks_all_consumed_outputs() -> None:
    action = SimpleNamespace(
        state=torch.ones(1, 1, 2),
        state_mask=torch.ones(1, 1, 2, dtype=torch.bool),
        embodiment_id=torch.tensor([1]),
    )
    eager = SimpleNamespace(
        input_embeds=torch.ones(1, 2, 3),
        attention_mask=torch.ones(1, 2, dtype=torch.bool),
        image_token_mask=torch.tensor([[True, False]]),
        action_inputs=action,
        state=action.state,
        state_mask=action.state_mask,
    )
    compiled = SimpleNamespace(**vars(eager))
    stats = bench.stage1_parity(compiled, eager)
    assert stats["attention_mask_equal"] is True
    assert stats["image_token_mask_equal"] is True
    assert stats["state_mask_equal"] is True
    assert stats["embodiment_id_equal"] is True
    assert stats["state_rel_err"] == pytest.approx(0.0)


def test_unique_graph_count_uses_exact_counter() -> None:
    counters = {"stats.unique_graphs": 3, "inductor.compile_threads": 99}
    assert bench.unique_graph_count(counters) == 3


def test_perf_hints_are_captured_into_record() -> None:
    logger = torch._logging.getArtifactLogger("torch._inductor.test", "perf_hints")
    with bench.capture_inductor_perf_hints() as hints:
        logger.warning("sentinel perf hint")
    assert any("sentinel perf hint" in hint for hint in hints)


def test_build_production_input_runs_time_then_batch_shaping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    from exp.robocasa365 import groot_keys
    from exp.robocasa365 import groot_policy_adapter
    from exp.robocasa365 import serve_groot_n15
    from openpi.cache.groot import interceptor

    events: list[str] = []
    wire = {key: "old prompt" for key in groot_keys.LANGUAGE_KEYS}
    monkeypatch.setattr(serve_groot_n15, "_dummy_observation", lambda _: wire.copy())

    def add_time_axis(obs):
        events.append("time")
        assert all(obs[key] == "new prompt" for key in groot_keys.LANGUAGE_KEYS)
        return {"state": torch.zeros(1, 2).numpy()}

    def add_batch_axis(obs):
        events.append("batch")
        return {"state": obs["state"][None, ...]}

    monkeypatch.setattr(groot_policy_adapter, "build_groot_observation", add_time_axis)
    monkeypatch.setattr(interceptor, "_is_batched", lambda _: False)
    monkeypatch.setattr(interceptor, "_unsqueeze_values", add_batch_axis)

    class Policy:
        def apply_transforms(self, obs):
            events.append("transform")
            assert obs["state"].shape == (1, 1, 2)
            return obs

    result = bench.build_production_input(Policy(), tmp_path, "new prompt")
    assert result["state"].shape == (1, 1, 2)
    assert events == ["time", "batch", "transform"]


# ---------------------------------------------------------------------------
# numerical reporting -- the stage-1 diagnosis
# ---------------------------------------------------------------------------


def test_tensor_stats_identical_tensors() -> None:
    x = torch.randn(4, 8)
    stats = bench.tensor_stats(x.clone(), x)
    assert stats["cos_min"] == pytest.approx(1.0, abs=1e-6)
    assert stats["max_abs_delta"] == pytest.approx(0.0, abs=1e-6)
    assert stats["rel_frobenius"] == pytest.approx(0.0, abs=1e-6)


def test_tensor_stats_exposes_the_low_norm_token_effect() -> None:
    """One tiny token can sink min-cosine while the tensor is globally fine.

    This is the whole reason the production 0.999 min-cosine gate cannot, on its
    own, tell "miscompiled" from "one small token" -- and why the diagnosis mode
    reports quantiles and the worst token's norm alongside it.
    """
    eager = torch.randn(64, 16)
    eager[0] *= 1e-4  # one very low-norm token
    compiled = eager.clone()
    compiled[0] += 1e-3  # a fixed absolute perturbation

    stats = bench.tensor_stats(compiled, eager)
    assert stats["cos_min"] < 0.999, "the worst token should dominate min-cosine"
    assert stats["cos_p50"] > 0.999, "the bulk of tokens should be untouched"
    assert stats["rel_frobenius"] < 1e-2, "globally the tensors are close"
    assert stats["worst_token_norm"] < 1e-2, "the offender is the low-norm token"


def test_rel_err_is_normalised_by_the_reference() -> None:
    ref = torch.full((3,), 10.0)
    got = torch.full((3,), 10.1)
    assert bench.rel_err(got, ref) == pytest.approx(0.01, rel=1e-3)


# ---------------------------------------------------------------------------
# checkpoint identity
# ---------------------------------------------------------------------------


def test_checkpoint_identity_requires_content(tmp_path: pathlib.Path) -> None:
    with pytest.raises(SystemExit):
        bench.checkpoint_identity(tmp_path)


def test_checkpoint_identity_tracks_weights_not_just_config(
    tmp_path: pathlib.Path,
) -> None:
    """Hashing config.json alone would call two different weight sets identical."""
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"weights-A")
    first = bench.checkpoint_identity(tmp_path)

    (tmp_path / "model.safetensors").write_bytes(b"weights-B")
    assert bench.checkpoint_identity(tmp_path) != first


# ---------------------------------------------------------------------------
# frozen constants and repo conventions
# ---------------------------------------------------------------------------


def test_five_pinned_prompts_are_distinct() -> None:
    assert len(bench.PROMPTS) == 5
    assert len(set(bench.PROMPTS)) == 5


def test_compile_mode_is_cuda_graph_only() -> None:
    """Owner froze the CUDA-Graph tier; there is deliberately no fallback path."""
    assert bench.COMPILE_MODE == "reduce-overhead"


def test_public_functions_have_docstrings() -> None:
    """WORKING_AGREEMENT §3.2."""
    missing = [
        name
        for name, obj in vars(bench).items()
        if inspect.isfunction(obj)
        and obj.__module__ == bench.__name__
        and not name.startswith("_")
        and not (obj.__doc__ or "").strip()
    ]
    assert missing == [], f"public functions without docstrings: {missing}"
