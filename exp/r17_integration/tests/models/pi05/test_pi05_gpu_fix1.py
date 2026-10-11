"""CPU regression of the exact R16 reference boundary and grid evidence."""
from unittest.mock import Mock

import numpy as np
import pytest

from exp.r17_integration.common.contracts import CaptureSpec
from exp.r17_integration.common.serving import full_call
from exp.r17_integration.models.pi05.adapter import sampler_for_steps
from exp.r17_integration.models.pi05.gpu_diagnostic import (
    action_difference, native_grid_witness, r16_reference,
)
from exp.r17_integration.models.pi05.parity import (
    FIXTURE_CHECKPOINT_IDENTITY, FIXTURE_NORM_IDENTITY, FIXTURE_SCOPE,
    assert_native_parity, compiled_fixture_limits,
)
from exp.r17_integration.tests.models.pi05.test_pi05 import fixture_adapter, observation, request


@pytest.mark.parametrize("k,times,dt", [(2, [1.0, .5], -.5), (1, [1.0], -1.0)])
def test_stock_and_r16_original_loops_have_identical_exact_grid(k, times, dt):
    witness = native_grid_witness(k)
    assert witness["stock"] == witness["r16_staged"] == times
    assert witness["dt_float32"] == dt and witness["cpu_native_loop_exact"]
    assert tuple(times) == sampler_for_steps(k).native_timesteps


@pytest.mark.parametrize("k", [10, 2, 1])
def test_exact_r16_reference_avoids_compiled_stock_and_adapter_methods(k, monkeypatch):
    adapter, model = fixture_adapter()
    obs, req = observation(), request()
    plain, wire = full_call(adapter, obs, req, sampler=sampler_for_steps(k))
    debug, debug_wire = full_call(adapter, obs, req, sampler=sampler_for_steps(k), capture=CaptureSpec(True, True))
    model.counts[:] = [0, 0, 0]
    poison = Mock(side_effect=AssertionError("wrong parity baseline path"))
    monkeypatch.setattr(adapter.policy, "infer", poison)
    monkeypatch.setattr(adapter.policy, "_sample_actions", poison)
    for name in ("prepare", "sample", "to_wire"):
        monkeypatch.setattr(adapter, name, poison)
    normalized, reference = r16_reference(adapter, obs, req, k)
    assert model.counts == [1, 1, k]
    assert normalized.tobytes() == plain.normalized_chunk.tobytes() == debug.normalized_chunk.tobytes()
    assert reference.tobytes() == wire.actions.tobytes() == debug_wire.actions.tobytes()
    poison.assert_not_called()


def test_diagnostic_statistics_do_not_hide_one_bit_mismatch():
    left = np.zeros((5, 4), dtype=np.float32)
    right = left.copy()
    right[0, 0] = np.nextafter(np.float32(0), np.float32(1))
    report = action_difference(left, right)
    assert not report["exact"] and report["max_abs_diff"] > 0
    assert report["mean_abs_diff"] == report["max_abs_diff"] / 20
    with pytest.raises(ValueError, match="nonfinite"):
        action_difference(left, np.full_like(left, np.nan))
    with pytest.raises(ValueError, match="geometry/dtype"):
        action_difference(left, left.astype(np.float64))


@pytest.mark.parametrize("k", [2, 1])
def test_alternate_tolerance_never_relaxes_exact_reference_or_debug(k):
    adapter, _ = fixture_adapter()
    obs, req = observation(), request()
    plain, wire = full_call(adapter, obs, req, sampler=sampler_for_steps(k))
    debug, debug_wire = full_call(adapter, obs, req, sampler=sampler_for_steps(k), capture=CaptureSpec(True, True))
    normalized, reference = r16_reference(adapter, obs, req, k)
    limits = {k: {"max_abs_diff": .08 if k == 2 else .13, "mean_abs_diff": .02 if k == 2 else .03,
                  "rtol": 0, "scope": FIXTURE_SCOPE}}
    alternate = wire.actions.copy()
    alternate[0, 0] += .01
    receipt = assert_native_parity(normalized, reference, plain, wire, debug, debug_wire,
                                   alternate_wire=alternate, alternate_limits=limits)
    assert receipt["exact_r16"] and receipt["exact_debug"]
    assert not receipt["alternate_path"]["exact"]
    # Old Q's unconditional equality fails the same alternate witness.
    assert alternate.tobytes() != wire.actions.tobytes()
    with pytest.raises(ValueError, match="sealed alternate"):
        assert_native_parity(normalized, reference, plain, wire, debug, debug_wire, alternate_wire=alternate)
    with pytest.raises(AssertionError, match="exceeded bound"):
        assert_native_parity(normalized, reference, plain, wire, debug, debug_wire,
                             alternate_wire=wire.actions + 1, alternate_limits=limits)
    wrong_reference = reference.copy()
    wrong_reference[0, 0] = np.nextafter(wrong_reference[0, 0], np.inf)
    with pytest.raises(AssertionError, match="exact R16/debug"):
        assert_native_parity(normalized, wrong_reference, plain, wire, debug, debug_wire,
                             alternate_wire=alternate, alternate_limits=limits)
    debug.normalized_chunk[0, 0] = np.nextafter(debug.normalized_chunk[0, 0], np.float32(np.inf))
    with pytest.raises(AssertionError, match="exact R16/debug"):
        assert_native_parity(normalized, reference, plain, wire, debug, debug_wire,
                             alternate_wire=alternate, alternate_limits=limits)


def test_alternate_fixture_limits_reject_unmeasured_model_and_archive(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    import hashlib
    from exp.r17_integration.models.pi05 import parity
    archive = tmp_path / "observations.npz"
    archive.write_bytes(b"CPU fixture cohort")
    spec = NS(model_id="pi05_rlinf_mt50", checkpoint_identity=FIXTURE_CHECKPOINT_IDENTITY,
              norm_identity=FIXTURE_NORM_IDENTITY)
    with pytest.raises(ValueError, match="observation archive"):
        compiled_fixture_limits(spec, archive)
    monkeypatch.setattr(parity, "FIXTURE_ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest())
    limits = compiled_fixture_limits(spec, archive)
    assert limits[2]["max_abs_diff"] == .08 and limits[1]["max_abs_diff"] == .13
    assert limits[2]["mean_abs_diff"] == .02 and limits[1]["mean_abs_diff"] == .03
    spec.norm_identity = "another norm"
    with pytest.raises(ValueError, match="model/norm"):
        compiled_fixture_limits(spec, archive)
