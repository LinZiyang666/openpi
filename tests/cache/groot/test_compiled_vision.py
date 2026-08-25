"""Compiled vision tower (compile_vision=True): equivalence, sharing, gating.

Owner directive 2026-08-22: stage1 is launch-bound at B=1, so the tower runs
under torch.compile(mode="reduce-overhead") behind an opt-in flag. These tests
pin the safety contract on the CPU stub: first-call eager-vs-compiled check,
process-level sharing of the compiled artifact, and the refuse-to-serve gate.
"""

from __future__ import annotations

import pytest
import torch

import openpi.cache.groot.staged as staged_mod
from openpi.cache.groot.staged import GrootStagedRunner


@pytest.fixture(autouse=True)
def _clean_registry():
    staged_mod._COMPILED_VISION_REGISTRY.clear()
    yield
    staged_mod._COMPILED_VISION_REGISTRY.clear()


def _make_runner(stub_model, **kw):
    return GrootStagedRunner(stub_model, verify_upstream=False, **kw)


def test_compiled_stage1_matches_eager(stub_model):
    eager = _make_runner(stub_model)
    with eager.session():
        ref = eager.run_stage1(stub_model.build_inputs())

    compiled = _make_runner(stub_model, compile_vision=True)
    with compiled.session():
        out = compiled.run_stage1(stub_model.build_inputs())
    assert torch.allclose(out.input_embeds, ref.input_embeds, atol=1e-6)
    # The one-time check has passed and is not repeated.
    assert staged_mod._COMPILED_VISION_REGISTRY[id(stub_model.backbone.eagle_model)]["checked"]


def test_compiled_artifact_shared_across_runners(stub_model):
    r1 = _make_runner(stub_model, compile_vision=True)
    r2 = _make_runner(stub_model, compile_vision=True)
    assert r1._compiled_entry is r2._compiled_entry
    assert len(staged_mod._COMPILED_VISION_REGISTRY) == 1


def test_divergent_compiled_tower_refuses_to_serve(stub_model):
    runner = _make_runner(stub_model, compile_vision=True)
    good = runner._compiled_entry["fn"]
    runner._compiled_entry["fn"] = lambda pv: -good(pv)  # poisoned: cos = -1
    with runner.session(), pytest.raises(RuntimeError, match="diverges from eager"):
        runner.run_stage1(stub_model.build_inputs())
    assert not runner._compiled_entry["checked"]


def test_compiled_output_is_copied_out_of_the_shared_buffer(stub_model):
    """A concurrent replay must not be able to rewrite the tokens we scattered.

    Under ``mode="reduce-overhead"`` the compiled callable hands back a
    CUDA-graph *static* output buffer, and the registry entry holding it is
    shared by every runner built on the same base model -- so between the
    compiled call and the scatter into the language sequence, that tensor is
    live shared state. The inference lock is what keeps two connections out of
    that window today; copying makes the output side correct on its own terms.
    (The graph's *input* buffer is static too, so the lock is still required --
    this closes one of the two directions, not both.)

    The CPU stub produces a fresh tensor per call, so a static buffer is
    installed here. The overwrite is injected inside the one-time verifier
    because that is the only seam the runner offers between the compiled call
    and the scatter -- which is exactly the window at issue. Asserting on the
    tensor *after* ``run_stage1`` returns would prove nothing: the scatter at
    ``flat_embeds[selected] = ...`` is an index assignment, i.e. already a copy.
    """
    inputs = stub_model.build_inputs()

    eager = _make_runner(stub_model)
    with eager.session():
        ref = eager.run_stage1(inputs)

    runner = _make_runner(stub_model, compile_vision=True)
    compiled_fn = runner._compiled_entry["fn"]
    shared: dict[str, torch.Tensor] = {}

    def static_buffer_fn(pixel_values):
        out = compiled_fn(pixel_values)
        buf = shared.get("buf")
        if buf is None:
            buf = out.clone()
            shared["buf"] = buf
        buf.copy_(out)
        return buf  # the same tensor object every call, like a graph output

    runner._compiled_entry["fn"] = static_buffer_fn

    def concurrent_replay_lands_here(pixel_values, compiled_out):
        shared["buf"].fill_(-999.0)
        runner._compiled_entry["checked"] = True
        return compiled_out

    runner._verify_compiled_vision = concurrent_replay_lands_here

    with runner.session():
        out = runner.run_stage1(inputs)

    assert torch.allclose(out.input_embeds, ref.input_embeds, atol=1e-6)
