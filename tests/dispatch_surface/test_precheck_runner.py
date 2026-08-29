"""Execution-boundary regressions for the formal dispatch precheck runner."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from examples.libero.worker_entry import _init_state_index
from exp.dispatch_surface.run_precheck import (
    FROZEN_LAUNCH_KEYS,
    PrecheckSweepStrategy,
    arms_with_accepted_work_left,
    build_worker_specs,
    validate_aprime_pool,
    validate_existing_launch_ledger,
)
from openpi.conductor import ExperimentStrategy, ServerEndpoint


def test_strategy_builds_a_valid_graph_and_loads_each_arm(tmp_path):
    yaml_path = tmp_path / "arm.yaml"
    yaml_path.write_text("enabled: true\n")
    officials = {task: [task, task + 10] for task in range(10)}
    strategy = PrecheckSweepStrategy(
        "libero_spatial", {"dsp_sv": str(yaml_path)}, 2, officials
    )
    assert isinstance(strategy, ExperimentStrategy)

    endpoint = ServerEndpoint("localhost", 8000)
    graph = strategy.plan(["dsp_sv"], {"dsp_sv": endpoint})
    graph.validate()
    stage = graph.stages["eval__dsp_sv"]
    assert len(stage.episodes) == 20
    assert stage.episodes[0].episode_idx == 0
    assert stage.episodes[0].orig_init_state_idx == 0
    assert stage.episodes[1].episode_idx == 1
    assert stage.episodes[1].orig_init_state_idx == 10

    calls = []

    class Control:
        def load_cache_config(self, **kwargs):
            calls.append(kwargs)

    strategy.on_stage_begin(stage, Control(), None)
    assert calls == [{
        "yaml_content": "enabled: true\n",
        "yaml_id": "dsp_sv",
        "bundle_id": "dsp_sv",
    }]


def test_resume_filter_does_not_count_fenced_terminal_rows(tmp_path):
    journal = tmp_path / "journal.jsonl"
    journal.write_text("\n".join([
        json.dumps({"yaml_id": "dsp_sv", "task_uid": "dsp_sv:eval:0:0",
                    "phase": "eval", "accepted": False}),
        json.dumps({"yaml_id": "dsp_sv", "task_uid": "dsp_sv:eval:0:1",
                    "phase": "eval", "accepted": True}),
    ]))
    remaining, counts = arms_with_accepted_work_left(journal, ["dsp_sv"], expected=2)
    assert remaining == ["dsp_sv"]
    assert counts == {"dsp_sv": 1}


def test_materialised_aprime_uses_subset_position_without_losing_official_identity():
    task = type("Task", (), {
        "episode_idx": 7,
        "orig_init_state_idx": 43,
    })()
    assert _init_state_index(task, "subset") == 7
    assert _init_state_index(task, "orig") == 43


def test_formal_worker_specs_receive_subset_mode_replan_and_seed():
    specs = build_worker_specs(
        ["server:8000"],
        gpus=1,
        conda_env="libero",
        task_suite="libero_spatial",
        pool_dir="test_aprime",
        replan_steps=5,
        seed=17,
    )
    assert len(specs) == 1
    spec = specs[0]
    assert spec.init_state_index_mode == "subset"
    assert spec.init_states_dir == "test_aprime"
    assert spec.replan_steps == 5
    assert spec.seed == 17


def _write_aprime_fixture(tmp_path, *, trials=2):
    pool_dir = tmp_path / "test_aprime"
    pool_dir.mkdir()
    assignment = {}
    digests = {}
    for task in range(10):
        task_name = f"task_{task}"
        states = np.array([[task, 1.0], [task, 3.0]], dtype=np.float64)[:trials]
        torch.save(states, pool_dir / f"{task_name}.init")
        assignment[str(task)] = {"task_name": task_name, "test": [1, 3][:trials]}
        digests[task_name] = {
            "count": trials,
            "indices": [1, 3][:trials],
            "sha256": hashlib.sha256(np.ascontiguousarray(states).tobytes()).hexdigest(),
        }
    split = {
        "suite": "libero_spatial",
        "quota": {"test": trials},
        "assignment": assignment,
        "pool_digests": {"test_aprime": digests},
    }
    split_path = tmp_path / "split_manifest.json"
    split_path.write_text(json.dumps(split))
    return split_path, pool_dir


def test_aprime_validator_binds_actual_worker_pool_to_split(tmp_path):
    split, pool_dir = _write_aprime_fixture(tmp_path)
    record = validate_aprime_pool(str(split), pool_dir, 2)
    assert record["suite"] == "libero_spatial"
    assert record["total_inits"] == 20
    assert len(record["per_task_digests"]) == 10


def test_aprime_validator_rejects_tampered_state_bytes(tmp_path):
    split, pool_dir = _write_aprime_fixture(tmp_path)
    torch.save(np.zeros((2, 2)), pool_dir / "task_4.init")
    with pytest.raises(SystemExit, match="state bytes"):
        validate_aprime_pool(str(split), pool_dir, 2)


def _launch_entry(*, run_id=None, executed=("dsp_sv", "dsp_s0")):
    frozen = {"dsp_sv": "sv-sha", "dsp_s0": "s0-sha"}
    entry = {
        "protocol": "dispatch_surface_rev1",
        "layer": "primary",
        "suite": "libero_spatial",
        "core_arms": ["dsp_s0", "dsp_sv"],
        "descriptive_arms": [],
        "trials_per_task": 30,
        "replan_steps": 5,
        "env_seed": 7,
        "policy_fingerprint": "fp",
        "library_sha256": "lib",
        "aprime_content_sha256": "pool",
        "split_manifest_sha256": "split",
        "arm_matrix_sha256": "matrix",
        "frozen_yaml_sha256": frozen,
        "artifact_sha256": {"dsp_sv": "art-sv", "dsp_s0": "art-s0"},
        "fit_record_sha256": {"sv": "fit-sv", "s0": "fit-s0"},
        "executed_arms": list(executed),
        "executed_yaml_sha256": {arm: frozen[arm] for arm in executed},
    }
    assert set(FROZEN_LAUNCH_KEYS).issubset(entry)
    if run_id is not None:
        entry["run_id"] = run_id
    return entry


def test_resume_ledger_allows_a_strict_executed_arm_subset():
    prior = _launch_entry(run_id="run-1")
    resumed = _launch_entry(executed=("dsp_sv",))
    validate_existing_launch_ledger(
        {"schema_version": 2, "launches": [prior]}, resumed
    )


def test_resume_ledger_rejects_experiment_drift():
    prior = _launch_entry(run_id="run-1")
    resumed = _launch_entry(executed=("dsp_sv",))
    resumed["aprime_content_sha256"] = "other-pool"
    with pytest.raises(SystemExit, match="aprime_content_sha256"):
        validate_existing_launch_ledger(
            {"schema_version": 2, "launches": [prior]}, resumed
        )


# ---------------- Rev 1 layers (G2R1-B6) ----------------

def test_secondary_roster_is_a_strict_subset_of_primary_without_s0():
    from exp.dispatch_surface.run_precheck import FORMAL_CORE_ARMS, SECONDARY_CORE_ARMS

    assert SECONDARY_CORE_ARMS < FORMAL_CORE_ARMS
    assert "dsp_s0" in FORMAL_CORE_ARMS and "dsp_s0" not in SECONDARY_CORE_ARMS
    assert len(SECONDARY_CORE_ARMS) == 4 and len(FORMAL_CORE_ARMS) == 5


def test_each_layer_pins_its_own_gate():
    from exp.dispatch_surface.run_precheck import (
        LAYER_EXPECTED_GATE, LAYER_PRIMARY, LAYER_SECONDARY,
    )

    assert LAYER_EXPECTED_GATE[LAYER_PRIMARY] == "always_search"
    assert LAYER_EXPECTED_GATE[LAYER_SECONDARY] == "score_hysteresis"


def test_emitter_and_runner_agree_on_both_rosters():
    """The bug this pins: emitter froze 4 arms, runner demanded 5, so a legal
    secondary matrix could never start."""
    from exp.dispatch_surface.emit_precheck_yamls import (
        PRIMARY_CORE_ARMS as EMIT_PRIMARY,
        SECONDARY_ARMS as EMIT_SECONDARY,
    )
    from exp.dispatch_surface.run_precheck import FORMAL_CORE_ARMS, SECONDARY_CORE_ARMS

    assert set(EMIT_PRIMARY) == set(FORMAL_CORE_ARMS)
    assert set(EMIT_SECONDARY) == set(SECONDARY_CORE_ARMS)


def test_emitter_and_runner_agree_on_both_gates():
    from exp.dispatch_surface.emit_precheck_yamls import (
        LAYER_PRIMARY as E_PRIMARY, LAYER_SECONDARY as E_SECONDARY, gate_section,
    )
    from exp.dispatch_surface.run_precheck import LAYER_EXPECTED_GATE

    for layer in (E_PRIMARY, E_SECONDARY):
        assert gate_section(layer, 0.97)["type"] == LAYER_EXPECTED_GATE[layer]


# ---------------- launch dry validation (section 9 release gate) ----------------

def test_dry_validate_exists_and_precedes_any_driver_construction():
    """The gate must run every check and then stop BEFORE episodes or ledger writes."""
    import inspect

    from exp.dispatch_surface import run_precheck as rp

    src = inspect.getsource(rp.main)
    dry = src.index("if args.dry_validate:")
    assert dry < src.index("ConductorDriver("), "dry exit must precede driver construction"
    assert dry < src.index('ledger["launches"].append'), "dry run must not touch the ledger"
    assert dry < src.index("driver_thread.start()"), "dry run must not start episodes"


def test_dry_validate_is_opt_in():
    """A default run must never silently become a no-op."""
    import inspect

    from exp.dispatch_surface import run_precheck as rp

    src = inspect.getsource(rp.main)
    assert '"--dry-validate", action="store_true"' in src


def test_dry_validate_reports_the_frozen_launch_keys():
    """The summary has to show what a real launch WOULD freeze, or it proves nothing."""
    import inspect

    from exp.dispatch_surface import run_precheck as rp

    src = inspect.getsource(rp.main)
    block = src[src.index("if args.dry_validate:"):src.index("strategy = PrecheckSweepStrategy(")]
    assert "FROZEN_LAUNCH_KEYS" in block
    assert "contract_binding" in block
