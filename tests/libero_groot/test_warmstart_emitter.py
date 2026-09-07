"""LIBERO warm-start emitter: every cell is a warm cell under a named GR00T schedule."""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib

import pytest
import yaml

from exp.libero_groot import emit_gate_yamls as gate_emit
from exp.libero_groot import emit_warmstart_yamls as emit
from exp.libero_groot import gate_pareto_bindings as gpb
from openpi.cache.config import load_cache_config
from tests.libero_groot.test_gate_pareto_emit import _TEMPLATE


@pytest.fixture
def binding(tmp_path, monkeypatch):
    monkeypatch.setattr(gpb, "REPO_ROOT", tmp_path)
    b = gpb.for_suite("libero_spatial")
    b.config_root.mkdir(parents=True, exist_ok=True)
    b.template_path.write_text(
        yaml.safe_dump(copy.deepcopy(_TEMPLATE), sort_keys=False), "utf-8"
    )
    return b


def test_sweep_covers_every_recoverable_point_of_the_live_loop(binding):
    written = emit.emit_warm_sweep(binding, library="/x/warm.pkl", denoising_steps=8)
    assert sorted(written) == [
        f"gpws_{binding.tag}_k8_t{i / 8:.4f}" for i in range(1, 8)
    ]
    for path in written.values():
        cfg = load_cache_config(path)
        assert cfg.denoise_schedule == "groot_n15_k8_v1"
        assert cfg.checkpoints["cp1"].judge.type == "always_warm_start"
        assert cfg.checkpoints["cp1"].gate.type == "always_search"
        assert cfg.backend.in_memory.preload_path == "/x/warm.pkl"
        assert cfg.write_policy.type == "never"


def test_only_gate_judge_schedule_and_library_move(binding):
    written = emit.emit_warm_sweep(
        binding, library="/x/warm.pkl", denoising_steps=4, timesteps=[0.5]
    )
    cfg = yaml.safe_load(open(next(iter(written.values()))))
    untouched = copy.deepcopy(cfg)
    untouched["checkpoints"]["cp1"].pop("gate")
    untouched["checkpoints"]["cp1"].pop("judge")
    untouched.pop("denoise_schedule")
    untouched["backend"]["in_memory"].pop("preload_path")
    untouched.pop("write_policy")
    base = copy.deepcopy(_TEMPLATE)
    base["checkpoints"]["cp1"].pop("gate", None)
    base["checkpoints"]["cp1"].pop("judge", None)
    base["backend"]["in_memory"].pop("preload_path", None)
    base.pop("write_policy", None)
    assert untouched == base


def test_a_timestep_the_loop_cannot_resume_from_is_refused(binding):
    with pytest.raises(SystemExit, match="not a recoverable point"):
        emit.emit_warm_sweep(
            binding, library="/x/warm.pkl", denoising_steps=4, timesteps=[0.3]
        )


def test_index_records_library_schedule_and_emitter_identity(binding):
    emit.emit_warm_sweep(binding, library="/x/warm.pkl", denoising_steps=4)
    index = json.loads((binding.config_root / emit.ARM / "index.json").read_text())
    assert index["schedule_id"] == "groot_n15_k4_v1"
    assert index["library"] == "/x/warm.pkl"
    assert len(index["cells"]) == 3
    assert len(index["emitter_sha256"]) == 64
    verified = emit.verify_warm_sweep(binding.config_root / emit.ARM)
    assert verified["schedule_id"] == "groot_n15_k4_v1"


def test_eval_preflight_rejects_a_cell_changed_after_indexing(binding):
    written = emit.emit_warm_sweep(
        binding, library="/x/warm.pkl", denoising_steps=4, timesteps=[0.5]
    )
    path = next(iter(written.values()))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("# changed after digest\n")
    with pytest.raises(SystemExit, match="sha256"):
        emit.verify_warm_sweep(binding.config_root / emit.ARM)


def test_eval_preflight_requires_the_warm_index(binding):
    emit.emit_warm_sweep(
        binding, library="/x/warm.pkl", denoising_steps=4, timesteps=[0.5]
    )
    (binding.config_root / emit.ARM / "index.json").unlink()
    with pytest.raises(SystemExit, match="require index.json"):
        emit.verify_warm_sweep(binding.config_root / emit.ARM)


def test_eval_preflight_rejects_removal_of_the_last_warm_judge(binding):
    written = emit.emit_warm_sweep(
        binding, library="/x/warm.pkl", denoising_steps=4, timesteps=[0.5]
    )
    path = pathlib.Path(next(iter(written.values())))
    cfg = yaml.safe_load(path.read_text())
    cfg["checkpoints"]["cp1"]["judge"] = {"type": "always_hit"}
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(SystemExit, match="only indexed warm"):
        emit.verify_warm_sweep(path.parent)


def test_eval_preflight_keeps_the_legacy_search_index_compatible(tmp_path):
    (tmp_path / "cell.yaml").write_text(yaml.safe_dump(_TEMPLATE))
    (tmp_path / "index.json").write_text(
        json.dumps({"cell": {"file": "cell.yaml", "weights": {"robot_state": 1.0}}})
    )
    assert emit.verify_warm_sweep(tmp_path) is None


@pytest.mark.parametrize("change", ["disabled", "gate"])
def test_eval_preflight_checks_execution_shape_even_with_matching_digest(
    binding, change
):
    written = emit.emit_warm_sweep(
        binding, library="/x/warm.pkl", denoising_steps=4, timesteps=[0.5]
    )
    yaml_id, filename = next(iter(written.items()))
    path = pathlib.Path(filename)
    cfg = yaml.safe_load(path.read_text())
    if change == "disabled":
        cfg["enabled"] = False
    else:
        cfg["checkpoints"]["cp1"]["gate"] = {"type": "always_skip"}
    path.write_text(yaml.safe_dump(cfg))
    index_path = path.parent / "index.json"
    index = json.loads(index_path.read_text())
    index["cells"][yaml_id] = hashlib.sha256(path.read_bytes()).hexdigest()
    index_path.write_text(json.dumps(index))
    with pytest.raises(SystemExit, match="invalid cp1 judge"):
        emit.verify_warm_sweep(path.parent)


def test_the_gate_emitter_still_refuses_warm_tiers(binding):
    """Releasing warm start did not weaken the gate line's template-leak guard."""
    leaked = copy.deepcopy(_TEMPLATE)
    leaked["checkpoints"]["cp1"]["judge"] = {
        "type": "threshold",
        "threshold": 0.9,
        "warm_tiers": [{"threshold": 0.8, "start_t": 0.5}],
    }
    binding.template_path.write_text(yaml.safe_dump(leaked, sort_keys=False), "utf-8")
    cfg = gate_emit.build_warmup(binding)
    assert not cfg["checkpoints"]["cp1"]["judge"].get("warm_tiers")
