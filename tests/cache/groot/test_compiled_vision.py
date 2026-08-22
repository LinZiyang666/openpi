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
