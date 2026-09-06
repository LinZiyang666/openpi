"""Pinned-object plumbing: identity, payload delivery, and realized provenance.

The failure this suite exists to catch is not a crash — it is a run that looks
completely normal while sampling random objects. Three ways that can happen, and
one test class each:

* the payload never reaches the env (only its hash travels);
* the worker accepts a payload that disagrees with the table it was started on;
* the scene is built unpinned but the episode still records a correct-looking
  identity.

Everything is injected; nothing here builds a kitchen or touches a GPU.
"""

from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace
from typing import Any

import pytest

from openpi.cache.config import CacheConfig, ConfigValidationError
from openpi.conductor.task import ServerEndpoint

from exp.robocasa365.pinned_objects import (
    PNP_CACHE_ARM,
    PNP_ROSTER,
    PNP_TEACHER_ARM,
    assert_pnp_eval_identity,
    assert_pnp_run_plan_identity,
    compute_pin_id,
    compute_pin_task_id,
    load_pin_manifest,
    normalize_mjcf_path,
    realized_objects_of,
    resolve_manifest_path,
)
from exp.robocasa365.run_collect import RobocasaCollectStrategy, build_run_plan

TASK = "PickPlaceCounterToStove"
SLOTS = {
    "container": "objects/objaverse/pan/pan_4/model.xml",
    "obj": "objects/lightwheel/chicken_drumstick/ChickenDrumstick010/model.xml",
    "obj_container": "objects/objaverse/plate/plate_4/model.xml",
}
TABLE = {TASK: dict(SLOTS)}
PIN_ID = compute_pin_id(TABLE)
PIN_TASK_ID = compute_pin_task_id(TASK, SLOTS)


def _write_manifest(tmp_path: pathlib.Path, table: dict, pin_id: str | None = None) -> str:
    path = tmp_path / "pins.json"
    path.write_text(json.dumps({"pin_id": pin_id or compute_pin_id(table), "pinned_objects": table}))
    return str(path)


# ------------------------------------------------------------------
# Identity
# ------------------------------------------------------------------


class TestIdentity:
    def test_paths_from_different_machines_normalize_to_one_string(self):
        # The three hosts keep their asset trees at different prefixes; an
        # identity that changed with the prefix would make every cross-machine
        # comparison fail for a reason unrelated to the objects.
        weiland = (
            "/home/weiland/Isaac-GR00T/external_dependencies/robocasa365/robocasa"
            "/models/assets/objects/objaverse/plate/plate_9/model.xml"
        )
        timan = (
            "/scratch/zixuans8/Isaac-GR00T/external_dependencies/robocasa365/robocasa"
            "/models/assets/objects/objaverse/plate/plate_9/model.xml"
        )
        assert normalize_mjcf_path(weiland) == normalize_mjcf_path(timan)
        assert normalize_mjcf_path(weiland) == "objects/objaverse/plate/plate_9/model.xml"

    def test_path_outside_an_asset_tree_is_rejected(self):
        with pytest.raises(ValueError, match="objects"):
            normalize_mjcf_path("/tmp/somewhere/model.xml")

    def test_pin_id_ignores_key_order(self):
        a = {"B": {"y": "objects/2/model.xml", "x": "objects/1/model.xml"}}
        b = {"B": {"x": "objects/1/model.xml", "y": "objects/2/model.xml"}}
        assert compute_pin_id(a) == compute_pin_id(b)

    def test_pin_id_changes_when_one_mesh_changes(self):
        other = {TASK: {**SLOTS, "obj": "objects/lightwheel/apple/Apple001/model.xml"}}
        assert compute_pin_id(other) != PIN_ID

    def test_task_identity_is_not_the_global_identity(self):
        # A worker verifying only its own slice must not be able to satisfy the
        # global check by accident.
        assert PIN_TASK_ID != PIN_ID

    def test_manifest_edited_after_the_fact_is_rejected(self, tmp_path):
        stale = _write_manifest(tmp_path, TABLE, pin_id="0" * 64)
        with pytest.raises(ValueError, match="contents hash to"):
            load_pin_manifest(stale)

    def test_manifest_roundtrips(self, tmp_path):
        pin_id, table = load_pin_manifest(_write_manifest(tmp_path, TABLE))
        assert (pin_id, table) == (PIN_ID, TABLE)


# ------------------------------------------------------------------
# Realized provenance
# ------------------------------------------------------------------


class _Inner:
    def __init__(self, cfgs):
        self.object_cfgs = cfgs


class _Wrapper:
    """A Gym-style wrapper that does NOT forward attributes.

    Gymnasium's forwarding is not a stable contract across versions, so the
    reader must unwrap rather than rely on it.
    """

    def __init__(self, inner):
        self.unwrapped = inner


class TestRealizedProvenance:
    def _cfgs(self):
        return [
            {"name": name, "info": {"mjcf_path": f"/any/prefix/{path}"}}
            for name, path in SLOTS.items()
        ]

    def test_reads_through_a_wrapper_that_forwards_nothing(self):
        assert realized_objects_of(_Wrapper(_Inner(self._cfgs()))) == SLOTS

    def test_reads_an_unwrapped_env_too(self):
        assert realized_objects_of(_Inner(self._cfgs())) == SLOTS

    def test_unnamed_slot_gets_the_same_default_name_kitchen_uses(self):
        cfgs = [{"info": {"mjcf_path": "/p/objects/a/b/model.xml"}}]
        assert set(realized_objects_of(_Inner(cfgs))) == {"obj_1"}

    def test_slot_without_info_is_an_error_not_a_silent_gap(self):
        cfgs = [{"name": "obj"}]
        with pytest.raises(ValueError, match="no info.mjcf_path"):
            realized_objects_of(_Inner(cfgs))


# ------------------------------------------------------------------
# Dispatch: does the payload actually travel?
# ------------------------------------------------------------------


def _strategy(**kw) -> RobocasaCollectStrategy:
    base = dict(
        teacher="pi05",
        layout=1,
        style=1,
        base_seed=0,
        replan_steps=5,
        tasks=[(TASK, 3)],
    )
    base.update(kw)
    return RobocasaCollectStrategy(**base)


def _plan(strategy) -> Any:
    server = ServerEndpoint("127.0.0.1", 8000)
    return strategy.plan(sorted(strategy.yaml_ids), {yid: server for yid in strategy.yaml_ids})


class TestDispatch:
    def test_pinned_run_ships_the_slot_map_not_only_its_hash(self):
        graph = _plan(_strategy(pin_id=PIN_ID, pinned_objects=TABLE))
        extra = next(iter(graph.stages.values())).episodes[0].extra
        # The map itself is what builds the scene; the hashes only prove it.
        assert extra["pinned_objects"] == SLOTS
        assert extra["pin_id"] == PIN_ID
        assert extra["pin_task_id"] == PIN_TASK_ID

    def test_unpinned_run_ships_exactly_the_legacy_key_set(self):
        graph = _plan(_strategy())
        extra = next(iter(graph.stages.values())).episodes[0].extra
        assert set(extra) == {
            "task_name", "layout", "style", "teacher", "base_seed", "replan_steps", "batch",
        }

    def test_task_missing_from_the_table_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="no slot map"):
            _strategy(pin_id=PIN_ID, pinned_objects={"SomeOtherTask": SLOTS})

    def test_identity_without_payload_is_refused(self):
        with pytest.raises(ValueError, match="supplied together"):
            _strategy(pin_id=PIN_ID)

    def test_run_plan_hash_separates_two_pin_tables(self):
        # Otherwise a pinned batch resumes onto an unpinned journal and the two
        # distributions merge into one ledger with nothing to flag it.
        other = {TASK: {**SLOTS, "obj": "objects/lightwheel/apple/Apple001/model.xml"}}
        hashes = set()
        for table in (None, TABLE, other):
            kw = {} if table is None else {"pin_id": compute_pin_id(table), "pinned_objects": table}
            strategy = _strategy(**kw)
            hashes.add(build_run_plan(strategy, _plan(strategy), "/data/x")["plan_hash"])
        assert len(hashes) == 3


class TestSeedSegregation:
    def test_collection_batch_that_would_reach_the_eval_segment_is_refused(self):
        with pytest.raises(ValueError, match="eval segment"):
            _strategy(base_seed=0, tasks=[(TASK, 5)], episode_lo={TASK: 999_999})

    def test_an_eval_run_may_start_at_the_eval_base(self):
        # The guard must not reject the segment it is protecting.
        _strategy(base_seed=1_000_000, tasks=[(TASK, 8)])


# ------------------------------------------------------------------
# Serving: wrong library must not load
# ------------------------------------------------------------------


class _FakeStorage:
    def __init__(self, meta):
        self.artifact_meta = meta


def _pin_config(expected: str | None) -> CacheConfig:
    config = CacheConfig()
    config.backend.type = "in_memory"
    config.backend.in_memory.preload_path = "/tmp/lib.pkl"
    config.backend.in_memory.expected_pin_id = expected
    return config


class TestArtifactBinding:
    def _check(self, storage, config):
        from openpi.cache.config import _check_pin_identity_binding

        _check_pin_identity_binding(storage, config)

    def test_matching_identity_passes(self):
        self._check(_FakeStorage({"pin_id": PIN_ID}), _pin_config(PIN_ID))

    def test_library_from_a_different_pin_table_is_refused(self):
        with pytest.raises(ConfigValidationError, match="different object pin tables"):
            self._check(_FakeStorage({"pin_id": "0" * 64}), _pin_config(PIN_ID))

    def test_unpinned_library_under_a_pinned_config_is_refused(self):
        with pytest.raises(ConfigValidationError, match="records none"):
            self._check(_FakeStorage({"pin_id": None}), _pin_config(PIN_ID))

    def test_legacy_library_still_loads_when_nothing_is_expected(self):
        # Every library built before pinning existed must keep working.
        self._check(_FakeStorage({"pin_id": None}), _pin_config(None))


# ------------------------------------------------------------------
# Worker-side binding: the asymmetric cases are the dangerous ones
# ------------------------------------------------------------------


class _NullAdapter:
    def env_kwargs(self):
        return {}


def _runner(tmp_path, table=None):
    from exp.robocasa365.episode_runner import RobocasaEpisodeRunner

    made: list[dict[str, Any]] = []

    def gym_make(task_name, layout, style, **kw):
        made.append({"task_name": task_name, "layout": layout, "style": style, **kw})
        return object()

    runner = RobocasaEpisodeRunner(
        _NullAdapter(),
        client_factory=lambda server: object(),
        gym_make=gym_make,
        horizon_fn=lambda name: 5,
        handshake_probe=lambda server, deadline: None,
        connect_deadline_s=1.0,
        pinned_objects_path=_write_manifest(tmp_path, table) if table else None,
    )
    return runner, made


def _task(**extra_overrides):
    extra = {
        "task_name": TASK,
        "layout": 1,
        "style": 1,
        "teacher": "pi05",
        "base_seed": 1_000_000,
        "replan_steps": 5,
    }
    extra.update(extra_overrides)
    return SimpleNamespace(task_uid="uid-1", extra=extra)


def _pin_extra(**overrides):
    extra = {"pin_id": PIN_ID, "pin_task_id": PIN_TASK_ID, "pinned_objects": dict(SLOTS)}
    extra.update(overrides)
    return extra


class TestWorkerBinding:
    def test_matching_payload_is_accepted(self, tmp_path):
        runner, _ = _runner(tmp_path, TABLE)
        slots, task_id = runner._verify_pin(_task(**_pin_extra()), _task(**_pin_extra()).extra, TASK)
        assert (slots, task_id) == (SLOTS, PIN_TASK_ID)

    def test_pinned_worker_refuses_a_task_with_no_pin_keys(self, tmp_path):
        runner, _ = _runner(tmp_path, TABLE)
        task = _task()
        with pytest.raises(ValueError, match="missing pin keys"):
            runner._verify_pin(task, task.extra, TASK)

    def test_unpinned_worker_refuses_a_task_that_carries_pins(self, tmp_path):
        # The asymmetry that matters most: --pinned-objects forgotten on the
        # worker while the driver pins. Falling through to the legacy path would
        # build random-object scenes under a valid-looking identity.
        runner, _ = _runner(tmp_path, None)
        task = _task(**_pin_extra())
        with pytest.raises(ValueError, match="without --pinned-objects"):
            runner._verify_pin(task, task.extra, TASK)

    def test_neither_side_pinning_keeps_the_legacy_path(self, tmp_path):
        runner, _ = _runner(tmp_path, None)
        task = _task()
        assert runner._verify_pin(task, task.extra, TASK) == (None, None)

    def test_wrong_global_identity_is_refused(self, tmp_path):
        runner, _ = _runner(tmp_path, TABLE)
        task = _task(**_pin_extra(pin_id="0" * 64))
        with pytest.raises(ValueError, match="pin_id mismatch"):
            runner._verify_pin(task, task.extra, TASK)

    def test_slot_map_disagreeing_with_the_local_manifest_is_refused(self, tmp_path):
        runner, _ = _runner(tmp_path, TABLE)
        bad = {**SLOTS, "obj": "objects/lightwheel/apple/Apple001/model.xml"}
        task = _task(**_pin_extra(pinned_objects=bad))
        with pytest.raises(ValueError, match="pinned_objects mismatch"):
            runner._verify_pin(task, task.extra, TASK)

    def test_task_identity_not_matching_its_own_payload_is_refused(self, tmp_path):
        runner, _ = _runner(tmp_path, TABLE)
        task = _task(**_pin_extra(pin_task_id="0" * 64))
        with pytest.raises(ValueError, match="pin_task_id mismatch"):
            runner._verify_pin(task, task.extra, TASK)


class TestEnvCacheIdentity:
    def test_two_pin_identities_do_not_share_one_env(self, tmp_path):
        # The conductor interleaves tasks across one worker; without the pin in
        # the key the second cell silently reuses the first cell's scene.
        runner, made = _runner(tmp_path, TABLE)
        first = runner._ensure_env(TASK, 1, 1, pinned_objects=SLOTS, pin_task_id="a")
        second = runner._ensure_env(TASK, 1, 1, pinned_objects=SLOTS, pin_task_id="b")
        assert first is not second
        assert len(made) == 2

    def test_same_identity_reuses_the_cached_env(self, tmp_path):
        runner, made = _runner(tmp_path, TABLE)
        first = runner._ensure_env(TASK, 1, 1, pinned_objects=SLOTS, pin_task_id="a")
        again = runner._ensure_env(TASK, 1, 1, pinned_objects=SLOTS, pin_task_id="a")
        assert first is again
        assert len(made) == 1

    def test_pin_map_reaches_gym_make(self, tmp_path):
        runner, made = _runner(tmp_path, TABLE)
        runner._ensure_env(TASK, 1, 1, pinned_objects=SLOTS, pin_task_id="a")
        assert made[0]["pinned_objects"] == SLOTS

    def test_unpinned_call_shape_is_unchanged(self, tmp_path):
        # S2: an unpinned run must call gym_make exactly as it did before, or
        # every injected fake in the existing suites changes behaviour.
        runner, made = _runner(tmp_path, None)
        runner._ensure_env(TASK, 1, 1)
        assert "pinned_objects" not in made[0]


# ------------------------------------------------------------------
# Manifest path and roster gate
# ------------------------------------------------------------------


class TestManifestPath:
    def test_relative_path_becomes_absolute(self, tmp_path, monkeypatch):
        # Workers run with the external RoboCasa checkout as cwd, so a relative
        # path that opens on the driver opens nothing in the child.
        _write_manifest(tmp_path, TABLE)
        monkeypatch.chdir(tmp_path)
        resolved = resolve_manifest_path("pins.json")
        assert pathlib.Path(resolved).is_absolute()
        assert pathlib.Path(resolved).is_file()

    def test_missing_manifest_is_reported_at_resolution_time(self, tmp_path):
        with pytest.raises(ValueError, match="does not resolve to a file"):
            resolve_manifest_path(tmp_path / "nope.json")


class TestRosterGate:
    def _tasks(self, names=PNP_ROSTER, n=8):
        return [(name, n) for name in names]

    def test_frozen_cache_arm_shape_passes(self):
        assert_pnp_eval_identity(self._tasks(), cells=132, arm=PNP_CACHE_ARM, label="t")

    def test_frozen_teacher_arm_shape_passes(self):
        assert_pnp_eval_identity(
            self._tasks(n=50), cells=1, arm=PNP_TEACHER_ARM, label="t"
        )

    def test_reordered_roster_is_refused(self):
        # Order is the task_id assignment; reordering relabels every uid.
        reordered = list(PNP_ROSTER[1:]) + [PNP_ROSTER[0]]
        with pytest.raises(ValueError, match="not the frozen ordered roster"):
            assert_pnp_eval_identity(self._tasks(reordered), cells=132, arm=PNP_CACHE_ARM, label="t")

    def test_missing_task_is_refused(self):
        with pytest.raises(ValueError, match="not the frozen ordered roster"):
            assert_pnp_eval_identity(self._tasks(PNP_ROSTER[:4]), cells=132, arm=PNP_CACHE_ARM, label="t")

    def test_default_thirteen_task_roster_is_refused(self):
        names = list(PNP_ROSTER) + ["OpenCabinet", "CloseDrawer"]
        with pytest.raises(ValueError, match="not the frozen ordered roster"):
            assert_pnp_eval_identity(self._tasks(names), cells=132, arm=PNP_CACHE_ARM, label="t")

    def test_wrong_trials_per_task_is_refused(self):
        with pytest.raises(ValueError, match="episodes per task"):
            assert_pnp_eval_identity(self._tasks(n=7), cells=132, arm=PNP_CACHE_ARM, label="t")

    def test_wrong_cell_count_is_refused(self):
        with pytest.raises(ValueError, match="cell count"):
            assert_pnp_eval_identity(self._tasks(), cells=131, arm=PNP_CACHE_ARM, label="t")

    def test_ragged_episode_counts_are_refused(self):
        ragged = [(name, 8) for name in PNP_ROSTER[:-1]] + [(PNP_ROSTER[-1], 9)]
        with pytest.raises(ValueError, match="episodes per task"):
            assert_pnp_eval_identity(ragged, cells=132, arm=PNP_CACHE_ARM, label="t")


def _formal_run_plan(prefix: str, n: int, pin_id: str = PIN_ID) -> dict[str, Any]:
    tasks = [
        {
            "task_name": name,
            "task_id": task_id,
            "episode_lo": 0,
            "episode_hi": n - 1,
        }
        for task_id, name in enumerate(PNP_ROSTER)
    ]
    uids = [
        f"{prefix}__{name}:eval:{task_id}:{episode_idx}"
        for task_id, name in enumerate(PNP_ROSTER)
        for episode_idx in range(n)
    ]
    return {
        "params": {"pin_id": pin_id, "tasks": tasks},
        "uids": uids,
        "prefixes": {uid: f"out/{i}" for i, uid in enumerate(uids)},
        "plan_hash": f"hash-{prefix}",
    }


class TestSerializedRunPlanGate:
    def test_frozen_teacher_plan_passes(self):
        assert_pnp_run_plan_identity(
            [_formal_run_plan("teacher", 50)],
            arm=PNP_TEACHER_ARM,
            pin_id=PIN_ID,
            label="teacher",
        )

    def test_frozen_cache_matrix_passes(self):
        plans = [_formal_run_plan(f"cell-{i:03d}", 8) for i in range(132)]
        assert_pnp_run_plan_identity(
            plans, arm=PNP_CACHE_ARM, pin_id=PIN_ID, label="cache"
        )

    def test_serialized_task_range_drift_is_refused(self):
        plan = _formal_run_plan("teacher", 50)
        plan["params"]["tasks"][0]["episode_hi"] = 48
        with pytest.raises(ValueError, match="task ranges"):
            assert_pnp_run_plan_identity(
                [plan], arm=PNP_TEACHER_ARM, pin_id=PIN_ID, label="teacher"
            )

    def test_serialized_episode_loss_is_refused(self):
        plan = _formal_run_plan("teacher", 50)
        plan["uids"].pop()
        with pytest.raises(ValueError, match="serialized episode"):
            assert_pnp_run_plan_identity(
                [plan], arm=PNP_TEACHER_ARM, pin_id=PIN_ID, label="teacher"
            )

    def test_uid_task_name_must_match_its_task_id(self):
        plan = _formal_run_plan("teacher", 50)
        plan["uids"][0] = plan["uids"][0].replace(PNP_ROSTER[0], PNP_ROSTER[1])
        with pytest.raises(ValueError, match="names"):
            assert_pnp_run_plan_identity(
                [plan], arm=PNP_TEACHER_ARM, pin_id=PIN_ID, label="teacher"
            )

    def test_runtime_manifest_identity_must_match_the_plan(self):
        plan = _formal_run_plan("teacher", 50, pin_id="0" * 64)
        with pytest.raises(ValueError, match="manifest"):
            assert_pnp_run_plan_identity(
                [plan], arm=PNP_TEACHER_ARM, pin_id=PIN_ID, label="teacher"
            )

    def test_duplicate_cell_plan_is_refused(self):
        plans = [_formal_run_plan(f"cell-{i:03d}", 8) for i in range(131)]
        plans.append(plans[0])
        with pytest.raises(ValueError, match="not unique"):
            assert_pnp_run_plan_identity(
                plans, arm=PNP_CACHE_ARM, pin_id=PIN_ID, label="cache"
            )
