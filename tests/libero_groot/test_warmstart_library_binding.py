"""Schedule binding for LIBERO warm-start size-tier libraries."""

from __future__ import annotations

import json
import pickle
import types

import pytest

from exp.libero_groot import build_size_libraries as build
from exp.libero_groot import verify_libraries as verify


def test_full_builder_forwards_the_requested_schedule(monkeypatch, tmp_path):
    captured = {}

    def fake_run(cmd, *, check):
        captured["cmd"] = cmd
        captured["check"] = check

    monkeypatch.setattr(build.subprocess, "run", fake_run)
    build.build_full(
        data_dir="/data/warm",
        builder_type="cp1_groot_libero_mean_pool",
        out=tmp_path / "full.pkl",
        workers=2,
        denoise_schedule="groot_n15_k8_v1",
    )
    index = captured["cmd"].index("--expected-schedule")
    assert captured["cmd"][index + 1] == "groot_n15_k8_v1"
    assert captured["check"] is True


def test_posthoc_verifier_rejects_a_tier_from_another_schedule(monkeypatch, tmp_path):
    tier = tmp_path / "suite_S1.pkl"
    entry = types.SimpleNamespace(id="e0", trajectory_id="t0")
    with tier.open("wb") as handle:
        pickle.dump(
            {
                "key_builder_type": "cp1_groot_libero_mean_pool",
                "vector_dims": {"robot_state": 8},
                "schedule_id": "groot_n15_k4_v1",
                "entries": [entry],
            },
            handle,
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "prefix": "suite",
                "builder_type": "cp1_groot_libero_mean_pool",
                "vector_dims": {"robot_state": 8},
                "schedule_id": "groot_n15_k8_v1",
                "tiers": [
                    {
                        "tier": "S1",
                        "path": str(tier),
                        "entries": 1,
                        "trajectories": 1,
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(
        "sys.argv", ["verify_libraries.py", str(manifest), "--skip-backend"]
    )
    with pytest.raises(SystemExit) as excinfo:
        verify.main()
    assert excinfo.value.code == 1
