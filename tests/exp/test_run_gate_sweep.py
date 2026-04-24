"""Tests for ``exp/random_periodic_gate/run_gate_sweep.py``.

Covers pure-Python helpers: slug parsing, unit-key encode/decode, cost
metric derivation, and batch-dir -> cfg mapping. LIBERO env / websocket
paths are not exercised here (they live in §6.5 smoke).

Plan: ``logs/random_periodic_gate_plan.log.md`` §6.3.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.random_periodic_gate import run_gate_sweep as rgs


# ---------------------------------------------------------------------------
# Slug parsing
# ---------------------------------------------------------------------------


def test_parse_slug_periodic():
    meta = rgs.parse_slug("periodic_k5_n2")
    assert meta.slug == "periodic_k5_n2"
    assert meta.gate_type == "periodic"
    assert meta.gate_params == {"cache_len": 5, "inference_len": 2}
    assert meta.seed is None


def test_parse_slug_random():
    meta = rgs.parse_slug("random_p0p20_s1")
    assert meta.gate_type == "random"
    assert meta.gate_params == {"p_inference": 0.20}
    assert meta.seed == 1


def test_parse_slug_random_edge_probabilities():
    assert rgs.parse_slug("random_p0p05_s0").gate_params["p_inference"] == pytest.approx(0.05)
    assert rgs.parse_slug("random_p0p70_s2").gate_params["p_inference"] == pytest.approx(0.70)


def test_parse_slug_rejects_unknown():
    with pytest.raises(ValueError, match="unrecognized slug"):
        rgs.parse_slug("custom_gate")


# ---------------------------------------------------------------------------
# Unit-key encode/decode roundtrip
# ---------------------------------------------------------------------------


def test_unit_key_roundtrip():
    key = rgs._encode_unit("periodic_k5_n2.yaml", 3, 7)
    name, tid, iid = rgs._decode_unit(key)
    assert name == "periodic_k5_n2.yaml"
    assert tid == 3
    assert iid == 7


def test_unit_key_decode_rejects_malformed():
    with pytest.raises(ValueError, match="Malformed"):
        rgs._decode_unit("not-a-key")
    with pytest.raises(ValueError, match="Malformed"):
        rgs._decode_unit("periodic_k5_n2.yaml:3")


# ---------------------------------------------------------------------------
# Cost metric derivation
# ---------------------------------------------------------------------------


def test_derive_periodic_num_inference_full_period():
    # k=3, n=2, total=10 -> 2 full periods -> 4 skips
    assert rgs.derive_periodic_num_inference(10, 3, 2) == 4


def test_derive_periodic_num_inference_partial_period():
    # k=3, n=2, total=7 -> first period 5 (3 cache, 2 skip) + next 2 positions
    # (both in cache block) -> 2 skips total
    assert rgs.derive_periodic_num_inference(7, 3, 2) == 2


def test_derive_periodic_num_inference_total_less_than_cache_len():
    # All cycles land in the opening cache block.
    assert rgs.derive_periodic_num_inference(2, 5, 3) == 0


def test_derive_periodic_num_inference_k1_n1_alternating():
    # k=1, n=1 -> skip every other cycle
    assert rgs.derive_periodic_num_inference(10, 1, 1) == 5
    assert rgs.derive_periodic_num_inference(11, 1, 1) == 5


def test_derive_periodic_num_inference_k1_n10():
    # k=1, n=10, total=22: periods of length 11; period 0: 1 cache + 10 skip;
    # period 1 (cycles 11-21): 1 cache + 10 skip. total skips = 20.
    assert rgs.derive_periodic_num_inference(22, 1, 10) == 20


def test_estimated_random_num_inference_rounding():
    # 44 cycles * 0.2 = 8.8 -> round-half-to-even gives 9; and * 0.3 -> 13.2 -> 13.
    assert rgs.estimated_random_num_inference(44, 0.20) == 9
    assert rgs.estimated_random_num_inference(44, 0.30) == 13


def test_estimated_random_num_inference_endpoints():
    assert rgs.estimated_random_num_inference(100, 0.0) == 0
    assert rgs.estimated_random_num_inference(100, 1.0) == 100


# ---------------------------------------------------------------------------
# cfg inference from batch-dir name
# ---------------------------------------------------------------------------


def test_infer_cfg_from_batch_mapping():
    assert rgs._infer_cfg_from_batch(Path("batch1")) == "clip_w7_d4"
    assert rgs._infer_cfg_from_batch(Path("batch2")) == "spatial16_w8_d4"
    assert rgs._infer_cfg_from_batch(Path("batch3")) == "max_pool_w3_d5"


def test_infer_cfg_from_batch_rejects_unknown_name():
    with pytest.raises(ValueError, match="batch<N>"):
        rgs._infer_cfg_from_batch(Path("results"))
    with pytest.raises(ValueError, match="unexpected batch index"):
        rgs._infer_cfg_from_batch(Path("batch4"))


# ---------------------------------------------------------------------------
# --resume CLI flag propagation (G2 Round 1 Blocking #1 regression).
# ---------------------------------------------------------------------------


class _RunCallRecorder:
    """Stand-in for GateSweepRunner that records run() / parallel_run() kwargs."""

    def __init__(self, *args, **kwargs):
        self.calls: list[tuple[str, dict]] = []
        _RunCallRecorder.instances.append(self)

    instances: list["_RunCallRecorder"] = []

    def run(self, *, resume: bool = False) -> None:
        self.calls.append(("run", {"resume": resume}))

    def parallel_run(self, *, num_workers: int, resume: bool = False) -> None:
        self.calls.append(("parallel_run", {"num_workers": num_workers, "resume": resume}))

    def close_all(self) -> None:
        pass


def _stub_main_path(monkeypatch, tmp_path):
    """Wire up the tiniest possible runnable environment for main()."""
    _RunCallRecorder.instances = []
    monkeypatch.setattr(rgs, "GateSweepRunner", _RunCallRecorder)
    # Avoid hitting the real LIBERO benchmark + websocket.
    monkeypatch.setattr(rgs, "enumerate_full_suite", lambda _name: [(0, 0)])
    # Bundle version must strictly increase across YAMLs (real server's
    # invariant — monotonic counter on load_cache_config).
    counter = {"n": 0}
    def _fake_load(_url, _path):
        counter["n"] += 1
        return counter["n"]
    monkeypatch.setattr(rgs, "send_load_cache_config", _fake_load)

    batch_dir = tmp_path / "batch1"
    batch_dir.mkdir()
    (batch_dir / "periodic_k1_n1.yaml").write_text("enabled: false\n", encoding="utf-8")
    (batch_dir / "random_p0p30_s0.yaml").write_text("enabled: false\n", encoding="utf-8")
    out_root = tmp_path / "data"
    return batch_dir, out_root


def test_main_propagates_resume_true_to_serial_run(monkeypatch, tmp_path):
    batch_dir, out_root = _stub_main_path(monkeypatch, tmp_path)
    rgs.main([
        "--batch-dir", str(batch_dir),
        "--host", "localhost", "--port", "8001",
        "--out-root", str(out_root),
        "--num-workers", "1",
        "--resume",
    ])
    # Two YAMLs in the batch -> two runner instances -> two run() calls.
    assert len(_RunCallRecorder.instances) == 2
    for inst in _RunCallRecorder.instances:
        assert inst.calls == [("run", {"resume": True})]


def test_main_defaults_resume_false_to_serial_run(monkeypatch, tmp_path):
    batch_dir, out_root = _stub_main_path(monkeypatch, tmp_path)
    rgs.main([
        "--batch-dir", str(batch_dir),
        "--host", "localhost", "--port", "8001",
        "--out-root", str(out_root),
        "--num-workers", "1",
    ])
    for inst in _RunCallRecorder.instances:
        assert inst.calls == [("run", {"resume": False})]


def test_main_propagates_resume_true_to_parallel_run(monkeypatch, tmp_path):
    batch_dir, out_root = _stub_main_path(monkeypatch, tmp_path)
    rgs.main([
        "--batch-dir", str(batch_dir),
        "--host", "localhost", "--port", "8001",
        "--out-root", str(out_root),
        "--num-workers", "3",
        "--resume",
    ])
    for inst in _RunCallRecorder.instances:
        assert inst.calls == [("parallel_run", {"num_workers": 3, "resume": True})]
