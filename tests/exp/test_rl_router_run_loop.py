"""Run-loop contracts: pool isolation, transport, weight rollover, recovery.

These are the paths a multi-day unattended run actually depends on, and each of
them fails silently rather than loudly if it regresses:

  - **pool isolation** — an empty init dir makes LIBERO fall back to the
    official pruned_init pool, so training would run on the frozen test set and
    every reported number would be contaminated with no error anywhere;
  - **transport** — a package interrupted mid-copy would present as a short
    batch and burn the repair budget re-running episodes that already ran;
  - **weight rollover** — a loop that ships the same static yaml every round
    keeps the workers on the first checkpoint while the task metadata advertises
    a new version, so every episode is isolated and no batch ever fills;
  - **recovery** — client rows lost with a crashed process can never be
    re-collected, because the journal makes a restarted driver skip the very
    episodes whose evidence is missing;
  - **reclamation** — without it the frozen estimate is ~105 GB over a run
    instead of a ~5-6 GB steady state.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest
import torch
import yaml

from openpi.cache.components.mlp_router_judge import FEATURE_DTYPE, save_router_weights
from openpi.cache.types import ROBOT_STATE, VISION_0, VISION_1

from exp.rl_router.batch_package import (
    COMPLETE_MARKER,
    LocalTransport,
    TransportError,
    assemble_package,
    fetch_missing_slots,
    push_package,
    reclaim_batch_shards,
    steady_state_bytes,
)
from exp.rl_router.launch_gates import CAPACITY_CEILING_BYTES, check_launch_gates, m4_smoke
from exp.rl_router.train_router import EpisodeAdmissionError
from exp.rl_router.run_rl_router import (
    make_slots,
    resolve_init_states_dir,
    resume_state,
    sample_batch,
    write_versioned_yaml,
)

D = 32
H = 8


# ---------------------------------------------------------------------------
# B-pool isolation
# ---------------------------------------------------------------------------


def test_empty_init_dir_is_refused_with_the_reason() -> None:
    """The single most damaging silent failure in the whole pipeline."""
    with pytest.raises(SystemExit, match="pruned_init"):
        resolve_init_states_dir("")


def test_pruned_init_shadowing_is_refused(tmp_path) -> None:
    """``_load_init_states`` prefers ``<task>.pruned_init`` inside a custom
    directory, so one stray file silently redirects that task to the test set."""
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "TASK_A.init").write_bytes(b"x")
    (pool / "TASK_A.pruned_init").write_bytes(b"x")
    with pytest.raises(SystemExit, match="pruned_init"):
        resolve_init_states_dir(str(pool))


def test_a_clean_diff_pool_is_accepted(tmp_path) -> None:
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "TASK_A.init").write_bytes(b"x")
    assert resolve_init_states_dir(str(pool)) == str(pool)


def test_missing_or_empty_pool_is_refused(tmp_path) -> None:
    with pytest.raises(SystemExit, match="not a directory"):
        resolve_init_states_dir(str(tmp_path / "nope"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="no .init files"):
        resolve_init_states_dir(str(empty))


def test_b_index_maps_into_the_diff_pool_not_the_benchmark(tmp_path) -> None:
    """The real loader, on the real index: split index 7 must address the 7th
    state of the diff-pool file, not of the benchmark's pruned_init."""
    # main.py imports the LIBERO sim stack at module level; stub it exactly the
    # way tests/examples/test_libero_main.py does so the loader itself is the
    # thing under test, not the simulator.
    import importlib
    import sys
    from unittest.mock import MagicMock

    for name in ("libero", "libero.libero", "libero.libero.benchmark", "libero.libero.envs"):
        sys.modules.setdefault(name, MagicMock())
    libero_dir = pathlib.Path(__file__).resolve().parents[2] / "examples" / "libero"
    if str(libero_dir) not in sys.path:
        sys.path.insert(0, str(libero_dir))
    libero_main = importlib.import_module("main")

    pool = tmp_path / "pool"
    pool.mkdir()
    diff_states = torch.arange(50).reshape(50, 1).float()
    torch.save(diff_states, pool / "TASK_A.init")

    class _Task:
        name = "TASK_A"

    class _Suite:
        def get_task_init_states(self, task_id):
            return torch.full((50, 1), -1.0)      # the A pool: distinguishable

    loaded = libero_main._load_init_states(_Task(), _Suite(), 0, str(pool))
    assert float(loaded[7][0]) == 7.0             # diff pool
    fallback = libero_main._load_init_states(_Task(), _Suite(), 0, "")
    assert float(fallback[7][0]) == -1.0          # what an empty dir silently gives


def test_worker_specs_carry_the_init_dir(tmp_path) -> None:
    """The guard is worthless if the validated path is not handed to the
    workers — that is where the pool is actually selected."""
    from openpi.conductor import WorkerSpec

    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "TASK_A.init").write_bytes(b"x")
    resolved = resolve_init_states_dir(str(pool))
    specs = [
        WorkerSpec(worker_id=f"w{i}", server_key="h:1", gpu_id="0",
                   task_suite_name="libero_10", init_states_dir=resolved)
        for i in range(4)
    ]
    assert all(s.init_states_dir == str(pool) for s in specs)
    assert WorkerSpec(worker_id="w", server_key="h:1", gpu_id="0").init_states_dir == ""


# ---------------------------------------------------------------------------
# Cross-machine transport
# ---------------------------------------------------------------------------


def _package(tmp_path, *, batch_id="b0", rows=None) -> pathlib.Path:
    return assemble_package(
        tmp_path / f"local_{batch_id}", batch_id=batch_id, weights_version="v1",
        journal_rows=rows or [{"task_uid": "u0", "accepted": True, "success": True}],
        client_rows=[{"task_uid": "u0", "attempt": 1, "router_outputs": {"decision_idx": 0}}],
        expected_slots=["u0"],
    )


def test_push_delivers_the_marker_last(tmp_path) -> None:
    """Marker-last IS the protocol: an interrupted copy leaves no marker, and a
    package without a marker is rejected rather than read as a short batch."""
    order: list[str] = []

    class _RecordingTransport(LocalTransport):
        def push_dir(self, local, remote, *, names):
            order.extend(names)
            super().push_dir(local, remote, names=names)

    remote = str(tmp_path / "remote")
    assert push_package(_RecordingTransport(), _package(tmp_path), remote) == "pushed"
    assert order[-1] == COMPLETE_MARKER
    assert order[:-1] and COMPLETE_MARKER not in order[:-1]


def test_interrupted_push_is_detectable_and_repushable(tmp_path) -> None:
    local = _package(tmp_path)
    remote = tmp_path / "remote"

    class _FailsBeforeMarker(LocalTransport):
        def push_dir(self, local, remote, *, names):
            if names == [COMPLETE_MARKER]:
                raise OSError("link dropped")
            super().push_dir(local, remote, names=names)

    from exp.rl_router.batch_package import verify_package

    with pytest.raises(TransportError, match="ALERT"):
        push_package(_FailsBeforeMarker(), local, str(remote), retries=2)
    with pytest.raises(ValueError, match=COMPLETE_MARKER):
        verify_package(remote)                      # detectable, not a short batch
    assert push_package(LocalTransport(), local, str(remote)) == "pushed"
    verify_package(remote)                          # the retry lands cleanly


def test_repushing_identical_content_is_idempotent(tmp_path) -> None:
    local, remote = _package(tmp_path), str(tmp_path / "remote")
    assert push_package(LocalTransport(), local, remote) == "pushed"
    assert push_package(LocalTransport(), local, remote) == "already_delivered"


def test_a_second_batch_claiming_a_delivered_id_is_refused(tmp_path) -> None:
    """Two different batches under one id would have the trainer's idempotence
    ledger silently skip the second."""
    remote = str(tmp_path / "remote")
    push_package(LocalTransport(), _package(tmp_path), remote)
    other = _package(tmp_path.joinpath("second"), rows=[{"task_uid": "u1", "accepted": True}])
    with pytest.raises(TransportError, match="different digest"):
        push_package(LocalTransport(), other, remote)


def test_push_retries_then_alerts(tmp_path) -> None:
    attempts = {"n": 0}

    class _Flaky(LocalTransport):
        def push_dir(self, local, remote, *, names):
            attempts["n"] += 1
            raise OSError("boom")

    with pytest.raises(TransportError, match="ALERT"):
        push_package(_Flaky(), _package(tmp_path), str(tmp_path / "remote"), retries=3)
    assert attempts["n"] == 3


def test_missing_slots_travel_back_from_the_remote_join(tmp_path) -> None:
    """The join runs where the shards are; the conductor only learns which
    slots to repair by fetching the remote manifest back."""
    remote_manifest = tmp_path / "remote" / "manifest.json"
    remote_manifest.parent.mkdir(parents=True)
    remote_manifest.write_text(json.dumps({"missing_slots": ["u1", "u2"]}), encoding="utf-8")
    missing = fetch_missing_slots(
        LocalTransport(), remote_manifest=str(remote_manifest),
        local_path=tmp_path / "local_manifest.json",
    )
    assert missing == ["u1", "u2"]
    assert (tmp_path / "local_manifest.json").exists()

    with pytest.raises(TransportError, match="was not produced"):
        fetch_missing_slots(LocalTransport(), remote_manifest=str(tmp_path / "nope.json"),
                            local_path=tmp_path / "x.json")


# ---------------------------------------------------------------------------
# Weight rollover through the real run-loop helpers
# ---------------------------------------------------------------------------


# The emitted arm yaml carries the production feature set, so a checkpoint the
# loop ships has to match it field for field — that agreement is the point.
ARM_DIMS = {VISION_0: 32768, VISION_1: 32768, ROBOT_STATE: 32}
ARM_DIM = sum(ARM_DIMS.values())


def _weights(path, version: str) -> str:
    torch.manual_seed(abs(hash(version)) % 997)
    save_router_weights(
        path, W1=torch.zeros(H, ARM_DIM), b1=torch.zeros(H),
        W2=torch.randn(2, H) * 0.3, b2=torch.zeros(2),
        arms="ts", fields=tuple(ARM_DIMS), dims=ARM_DIMS,
        weights_version=version, mu=torch.zeros(32), sigma=torch.ones(32),
    )
    return str(path)


def _base_arm_yaml(tmp_path) -> pathlib.Path:
    from exp.rl_router.emit_router_yamls import emit

    emit(tmp_path / "cfg", suite="libero_10", weights_path=str(tmp_path / "v0.pt"),
         student_endpoint="127.0.0.1:7002", dump_dir="/tmp/d", temperature=1.0,
         seed=0, hidden=H, variants=["R_ts"])
    return tmp_path / "cfg" / "r_ts_train.yaml"


def test_each_round_ships_a_yaml_pointing_at_the_current_checkpoint(tmp_path) -> None:
    """Without this the workers keep building a judge from the FIRST checkpoint
    while their task metadata advertises a new version, and every episode is
    isolated on the mismatch — no batch would ever fill."""
    base = _base_arm_yaml(tmp_path)
    v1 = _weights(tmp_path / "v1.pt", "v1")
    v2 = _weights(tmp_path / "v2.pt", "v2")

    y1 = write_versioned_yaml(base, weights_path=v1, weights_version="v1",
                              out_path=tmp_path / "b0" / "arm__v1.yaml")
    y2 = write_versioned_yaml(base, weights_path=v2, weights_version="v2",
                              out_path=tmp_path / "b1" / "arm__v2.yaml")

    assert yaml.safe_load(y1.read_text())["checkpoints"]["cp1"]["judge"]["weights_path"] == v1
    assert yaml.safe_load(y2.read_text())["checkpoints"]["cp1"]["judge"]["weights_path"] == v2

    # And a judge built from each yaml really reports that version.
    from openpi.cache.config import _build_judge, load_cache_config

    for path, expected in ((y1, "v1"), (y2, "v2")):
        cfg = load_cache_config(str(path))
        assert _build_judge(cfg.checkpoints["cp1"].judge).weights_version == expected


def test_a_mislabelled_checkpoint_cannot_be_shipped(tmp_path) -> None:
    base = _base_arm_yaml(tmp_path)
    v1 = _weights(tmp_path / "v1.pt", "v1")
    with pytest.raises(ValueError, match="mislabelled"):
        write_versioned_yaml(base, weights_path=v1, weights_version="v9",
                             out_path=tmp_path / "arm.yaml")


def test_versioned_yaml_is_written_atomically(tmp_path) -> None:
    base = _base_arm_yaml(tmp_path)
    out = write_versioned_yaml(base, weights_path=_weights(tmp_path / "v1.pt", "v1"),
                               weights_version="v1", out_path=tmp_path / "arm.yaml")
    assert out.exists() and not out.with_name(out.name + ".tmp").exists()


# ---------------------------------------------------------------------------
# Resume: the checkpoint is the authority
# ---------------------------------------------------------------------------


def _trainer(tmp_path, version: str):
    from exp.rl_router.train_router import RouterPolicy, RouterTrainer, TrainerHParams

    torch.manual_seed(0)
    policy = RouterPolicy(torch.randn(H, D) * 0.3, torch.zeros(H),
                          torch.randn(2, H) * 0.3, torch.zeros(2), arms="ts")
    return RouterTrainer(
        policy, TrainerHParams(arm_costs={"teacher": 1.0, "student": 0.2}, lam=0.2, t_max=100),
        weights_version=version,
    )


def _remote(tmp_path):
    """A RemoteRun over LocalTransport with a *separate* directory tree.

    Separate on purpose: the loop must never read a remote artifact as if it
    were local, and pointing the two namespaces at different roots is what makes
    such a slip fail here instead of on the first real cross-machine update.
    """
    from exp.rl_router.run_rl_router import RemoteRun

    return RemoteRun(LocalTransport(), root=str(tmp_path / "REMOTE"),
                     run_id="r0", shard_root=str(tmp_path / "REMOTE_SHARDS"))


def _publish_state(remote, *, version: str, consumed: list[dict]) -> None:
    path = pathlib.Path(remote.state)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"weights_version": version,
                                "consumed_batches": consumed}), encoding="utf-8")


def test_resume_reads_the_next_batch_and_version_from_the_remote_state(tmp_path) -> None:
    """An operator-supplied start index is a guess; the trainer's published
    state records exactly which batches were consumed and which policy it holds
    — and it lives on the serving host, so it is fetched, never opened."""
    remote = _remote(tmp_path)
    _publish_state(remote, version="v2",
                   consumed=[{"batch_id": "b0000"}, {"batch_id": "b0001"}])
    _weights(pathlib.Path(remote.weights("v2")), "v2")

    state = resume_state(remote, tmp_path / "scratch")
    assert state["next_batch_idx"] == 2
    assert state["weights_version"] == "v2"
    assert state["needs_export"] is False


def test_fresh_run_starts_at_batch_zero(tmp_path) -> None:
    state = resume_state(_remote(tmp_path), tmp_path / "scratch")
    assert state == {"next_batch_idx": 0, "weights_version": None, "consumed": [],
                     "needs_export": False, "missing_metrics": []}


def test_missing_export_is_reported_as_recoverable_not_fatal(tmp_path) -> None:
    """The save-checkpoint/export crash window: the update IS durable inside the
    checkpoint, so the loop replays the export rather than re-running a batch."""
    remote = _remote(tmp_path)
    _publish_state(remote, version="v5", consumed=[{"batch_id": "b0000"}])
    state = resume_state(remote, tmp_path / "scratch")
    assert state["needs_export"] is True
    assert state["weights_version"] == "v5" and state["next_batch_idx"] == 1


def test_remote_namespace_never_collides_with_local_paths(tmp_path) -> None:
    """The two hosts do not share a mount; the addresses must not overlap."""
    remote = _remote(tmp_path)
    local_out = tmp_path / "LOCAL"
    for address in (remote.checkpoint, remote.metrics, remote.state,
                    remote.weights("v1"), remote.package("b0000", 0), remote.shards("b0000")):
        assert not str(address).startswith(str(local_out))
        assert "REMOTE" in str(address)


# ---------------------------------------------------------------------------
# Client-row durability
# ---------------------------------------------------------------------------


def test_client_rows_survive_a_crash_between_rounds(tmp_path) -> None:
    """The journal makes a restarted driver skip completed uids, so rows lost
    with the process can never be re-collected. Persisting per round is what
    keeps the three-source join possible after a crash."""
    from exp.rl_router.batch_package import read_jsonl

    rows_path = tmp_path / "client_rows.jsonl"
    first = [{"task_uid": "u0", "attempt": 1, "router_outputs": {"decision_idx": 0}}]
    with rows_path.open("a", encoding="utf-8") as fh:
        for row in first:
            fh.write(json.dumps(row) + "\n")
    # "crash" — a fresh process reads what landed and appends the repair round.
    assert read_jsonl(rows_path) == first
    second = [{"task_uid": "u1#r1", "attempt": 1, "router_outputs": {"decision_idx": 0}}]
    with rows_path.open("a", encoding="utf-8") as fh:
        for row in second:
            fh.write(json.dumps(row) + "\n")
    assert [r["task_uid"] for r in read_jsonl(rows_path)] == ["u0", "u1#r1"]


def test_repair_round_dispatches_only_the_missing_slots(tmp_path) -> None:
    """A repair round re-runs the missing inits under ``#r1`` uids and leaves
    the rest alone; the second round must not re-dispatch a full batch."""
    from exp.rl_router.run_rl_router import run_batch_with_repair

    slots = make_slots([(0, i) for i in range(4)], yaml_id="rlr")
    dispatched: list[list[str]] = []

    def runner(strategy, repair):
        dispatched.append([uid for _, _, uid in strategy._slots])
        return [], []

    calls = {"n": 0}

    def join(journal_rows, client_rows, round_idx, quarantine):
        calls["n"] += 1
        if calls["n"] == 1:
            return False, [slots[1][2], slots[3][2]]
        return True, []

    run_batch_with_repair(
        round_runner=runner,
        strategy_factory=lambda pending, repair: type("S", (), {"_slots": pending})(),
        slots=slots, join=join, train=lambda r: (True, []), batch_id="b0",
    )
    assert len(dispatched) == 2
    assert len(dispatched[0]) == 4
    assert dispatched[1] == [f"{slots[1][2]}#r1", f"{slots[3][2]}#r1"]


def test_a_batch_still_short_after_two_repairs_halts(tmp_path) -> None:
    from exp.rl_router.run_rl_router import run_batch_with_repair

    slots = make_slots([(0, 0)], yaml_id="rlr")
    with pytest.raises(RuntimeError, match="ALERT"):
        run_batch_with_repair(
            round_runner=lambda s, r: ([], []),
            strategy_factory=lambda pending, repair: type("S", (), {"_slots": pending})(),
            slots=slots, join=lambda j, c, r, q: (False, [slots[0][2]]),
            train=lambda r: (True, []), batch_id="b0",
        )


# ---------------------------------------------------------------------------
# Reclamation + M4 capacity
# ---------------------------------------------------------------------------


def _checkpoint_with(tmp_path, *, batches: list[str], package_sha: str = "pkg") -> pathlib.Path:
    """A trainer checkpoint whose ledger names exactly ``batches``."""
    path = tmp_path / "ck.pt"
    torch.save({"consumed_batches": [{"batch_id": b, "package_sha256": package_sha}
                                     for b in batches]}, path)
    return path


def _shard_dir(tmp_path, *, n: int = 3, size: int = 1024) -> pathlib.Path:
    directory = tmp_path / "shards"
    directory.mkdir(parents=True)
    for i in range(n):
        (directory / f"u{i}.bin").write_bytes(b"\x00" * size)
        (directory / f"u{i}.jsonl").write_text('{"decision_idx": 0}\n', encoding="utf-8")
    (directory / "manifest.jsonl").write_text('{"task_uid": "u0"}\n', encoding="utf-8")
    return directory


def test_reclaim_frees_features_and_keeps_the_audit_trail(tmp_path) -> None:
    directory = _shard_dir(tmp_path)
    checkpoint = _checkpoint_with(tmp_path, batches=["b0000"])

    report = reclaim_batch_shards(directory, checkpoint_path=checkpoint,
                                  batch_id="b0000", package_sha256="pkg")

    assert report["bytes_freed"] == 3 * 1024
    assert not list(directory.glob("*.bin"))
    assert len(list(directory.glob("*.jsonl"))) == 4      # 3 sidecars + manifest
    assert (directory / "manifest.jsonl").exists()


def test_reclaim_refuses_before_the_checkpoint_lands(tmp_path) -> None:
    """Deleting before the update is durable could lose rollouts that still
    have to be replayed."""
    directory = _shard_dir(tmp_path)
    with pytest.raises(ValueError, match="not proven durable"):
        reclaim_batch_shards(directory, checkpoint_path=tmp_path / "absent.pt",
                             batch_id="b0000")
    assert len(list(directory.glob("*.bin"))) == 3


def test_reclaim_refuses_a_checkpoint_that_did_not_consume_this_batch(tmp_path) -> None:
    """A stale checkpoint from an earlier batch is a file on disk too. Letting
    mere existence authorise a deletion would destroy rollouts whose update
    never happened."""
    directory = _shard_dir(tmp_path)
    checkpoint = _checkpoint_with(tmp_path, batches=["b9999"])
    with pytest.raises(ValueError, match="has not consumed batch"):
        reclaim_batch_shards(directory, checkpoint_path=checkpoint, batch_id="b0000")
    assert len(list(directory.glob("*.bin"))) == 3


def test_reclaim_refuses_a_different_package_digest(tmp_path) -> None:
    directory = _shard_dir(tmp_path)
    checkpoint = _checkpoint_with(tmp_path, batches=["b0000"], package_sha="pkg-a")
    with pytest.raises(ValueError, match="was consumed from package"):
        reclaim_batch_shards(directory, checkpoint_path=checkpoint,
                             batch_id="b0000", package_sha256="pkg-b")
    assert len(list(directory.glob("*.bin"))) == 3


def test_reclaim_also_sweeps_torn_writes(tmp_path) -> None:
    directory = _shard_dir(tmp_path, n=1)
    (directory / "torn.bin.tmp").write_bytes(b"\x00" * 512)
    checkpoint = _checkpoint_with(tmp_path, batches=["b0000"])
    report = reclaim_batch_shards(directory, checkpoint_path=checkpoint, batch_id="b0000")
    assert report["bytes_freed"] == 1024 + 512
    assert not list(directory.glob("*.tmp"))


def test_steady_state_bytes_measures_live_features(tmp_path) -> None:
    directory = _shard_dir(tmp_path, n=2, size=2048)
    assert steady_state_bytes(directory.parent) == 4096


def test_m4_smoke_passes_on_a_healthy_batch(tmp_path) -> None:
    directory = _shard_dir(tmp_path, n=1, size=64)
    manifest = {
        "complete": True, "rejected": [], "missing_slots": [],
        "training_selected": [{"rows": 10, "dim": 65568} for _ in range(20)],
    }
    report = m4_smoke(
        manifest=manifest,
        metrics=[{"batch_id": "b0", "arm_executed_rate": {"teacher": 1.0}}],
        checkpoint_versions=["v0", "v1"],
        next_batch_shards=[{"weights_version": "v1"}],
        dump_root=directory.parent,
    )
    assert report["passed"], report["violations"]
    assert report["bytes_per_step"] == 65568 * 2
    assert report["episodes"] == 20


@pytest.mark.parametrize("break_it", ["join", "updates", "rollover", "fallback"])
def test_m4_smoke_catches_each_failure(tmp_path, break_it: str) -> None:
    directory = _shard_dir(tmp_path, n=1, size=64)
    manifest = {"complete": True, "rejected": [], "missing_slots": [],
                "training_selected": [{"rows": 10, "dim": 8} for _ in range(20)]}
    metrics = [{"batch_id": "b0", "arm_executed_rate": {"teacher": 1.0}}]
    versions = ["v0", "v1"]
    next_shards = [{"weights_version": "v1"}]

    if break_it == "join":
        manifest = {**manifest, "complete": False, "missing_slots": ["u3"]}
    elif break_it == "updates":
        metrics = metrics + [{"batch_id": "b0", "arm_executed_rate": {}}]
    elif break_it == "rollover":
        next_shards = [{"weights_version": "v0"}]       # workers never rebound
    else:
        metrics = [{"batch_id": "b0"}]                  # no arm/fallback record

    report = m4_smoke(manifest=manifest, metrics=metrics, checkpoint_versions=versions,
                      next_batch_shards=next_shards, dump_root=directory.parent)
    assert not report["passed"] and report["violations"]


# ---------------------------------------------------------------------------
# Launch gates
# ---------------------------------------------------------------------------


MATRIX = {
    "batch_size": 100, "episodes_per_run": 4000,
    "decisions": {"D3_training_init_domain": "b_train"},
    "suites": {"libero_10": {"split": "s.yaml"}},
    "lambda": {"lambda_1": 0.2, "lambda_2": None},
    "runs": [{"id": "r0", "suite": "libero_10", "variant": "R_ts",
              "lambda": "lambda_1", "seed": 0, "flagship": True}],
    "eval": {"primary_seed": 0},
}


def _artifacts(tmp_path, *, gpu_timed: bool = True, arms=("teacher", "student")) -> dict:
    import hashlib

    costs = tmp_path / "costs.json"
    costs.write_text(json.dumps({
        "normalized_costs": {a: (1.0 if a == "teacher" else 0.2) for a in arms},
        "provenance": {
            a: {"gpu_timed": gpu_timed, "in_process": True,
                **({"task_prompts": ["pick up the bowl"]} if a == "student" else {})}
            for a in arms
        },
    }), encoding="utf-8")
    (tmp_path / "warm.pt").write_text("{}", encoding="utf-8")
    WARM_SHA = hashlib.sha256((tmp_path / "warm.pt").read_bytes()).hexdigest()
    # A real pilot record: the gate reads the selection, the per-candidate
    # manifests and the digests, so an empty {} must NOT clear it.
    from exp.rl_router.pilot_lambda import PILOT_SCHEMA

    (tmp_path / "pilot.json").write_text(json.dumps({
        "schema": PILOT_SCHEMA, "separated": True, "seed": 0, "candidates_run": 3,
        "selected": {"lambda_1": 0.2, "lambda_2": 0.5},
        "protocol": {"batches": 5, "batch_size": 100, "eval_episodes": 100,
                     "mode": "argmax", "eval_pool": "b_train_remainder"},
        "pilot_split_sha256": "p" * 64, "remainder_split_sha256": "r" * 64,
"runs": {
            str(lam): {
                "lambda": lam, "lambda_recorded": lam, "seed": 0,
                "batch_size": 100, "train_judge_mode": "sample",
                "batches_trained": 5, "version_chain_contiguous": True,
                "weights_versions": [["v0", "v1"], ["v1", "v2"], ["v2", "v3"],
                                     ["v3", "v4"], ["v4", "v5"]],
                "start_weights_version": "v0", "final_weights_version": "v5",
                "eval_mode": "argmax", "eval_episodes": 100,
                "eval_weights_versions": ["v5"], "teacher_rate": rate,
                "train_split_sha256": "p" * 64,
                "expected_warmstart_sha256": WARM_SHA,
                "expected_pilot_split_sha256": "p" * 64,
                "expected_remainder_split_sha256": "r" * 64,
            }
            for lam, rate in ((0.05, 0.62), (0.2, 0.41), (0.5, 0.18))
        },
    }), encoding="utf-8")
    from exp.rl_router.launch_gates import M4_SCHEMA

    smoke = tmp_path / "smoke.json"
    smoke.write_text(json.dumps({
        "schema": M4_SCHEMA, "passed": True, "violations": [],
        "run_id": "r0", "batch_id": "b0000", "package_sha256": "s" * 64,
        "episodes": 20, "bytes_per_step": 131136, "peak_bytes": 3 * 1024**3,
        "bytes_before_reclaim": 3 * 1024**3, "bytes_after_reclaim": 1024,
        "weights_versions": ["v0", "v1"],
    }), encoding="utf-8")
    return {"arm_costs": str(costs), "warmstart_weights": str(tmp_path / "warm.pt"),
            "pilot": str(tmp_path / "pilot.json"), "capacity_smoke": str(smoke)}


def test_launch_gates_clear_a_ready_run(tmp_path) -> None:
    problems = check_launch_gates(
        MATRIX, MATRIX["runs"][0], artifacts=_artifacts(tmp_path),
        batch_size=100, seed=0, variant="R_ts", suite="libero_10",
    )
    assert problems == []


def test_launch_gates_block_a_null_lambda(tmp_path) -> None:
    run = {**MATRIX["runs"][0], "lambda": "lambda_2"}
    problems = check_launch_gates(
        MATRIX, run, artifacts=_artifacts(tmp_path),
        batch_size=100, seed=0, variant="R_ts", suite="libero_10",
    )
    assert any("still null" in p for p in problems)


def test_launch_gates_block_a_wall_clock_cost_artifact(tmp_path) -> None:
    """D4 is batch=1 GPU time. A wall-clock number would price every arm by the
    fleet's latency instead of its compute."""
    problems = check_launch_gates(
        MATRIX, MATRIX["runs"][0], artifacts=_artifacts(tmp_path, gpu_timed=False),
        batch_size=100, seed=0, variant="R_ts", suite="libero_10",
    )
    assert any("GPU-time" in p for p in problems)


def test_launch_gates_block_a_missing_arm_cost(tmp_path) -> None:
    problems = check_launch_gates(
        MATRIX, MATRIX["runs"][0], artifacts=_artifacts(tmp_path, arms=("teacher",)),
        batch_size=100, seed=0, variant="R_ts", suite="libero_10",
    )
    assert any("missing measurements" in p for p in problems)


def test_launch_gates_block_a_mismatched_batch_size_or_seed(tmp_path) -> None:
    artifacts = _artifacts(tmp_path)
    assert any("batch_size" in p for p in check_launch_gates(
        MATRIX, MATRIX["runs"][0], artifacts=artifacts, batch_size=64, seed=0,
        variant="R_ts", suite="libero_10"))
    assert any("seed" in p for p in check_launch_gates(
        MATRIX, MATRIX["runs"][0], artifacts=artifacts, batch_size=100, seed=7,
        variant="R_ts", suite="libero_10"))


@pytest.mark.parametrize("missing", ["warmstart_weights", "pilot", "capacity_smoke"])
def test_launch_gates_block_a_missing_prerequisite(tmp_path, missing: str) -> None:
    artifacts = _artifacts(tmp_path)
    artifacts[missing] = str(tmp_path / "absent")
    problems = check_launch_gates(
        MATRIX, MATRIX["runs"][0], artifacts=artifacts,
        batch_size=100, seed=0, variant="R_ts", suite="libero_10",
    )
    assert problems


def test_launch_gates_block_an_over_full_remote_dump_root(tmp_path) -> None:
    """Capacity is checked WHERE THE SHARDS ARE; a local stat of a remote path
    would report zero and clear the gate on a full disk."""
    problems = check_launch_gates(
        MATRIX, MATRIX["runs"][0], artifacts=_artifacts(tmp_path),
        batch_size=100, seed=0, variant="R_ts", suite="libero_10",
        remote_live_bytes=CAPACITY_CEILING_BYTES + 1,
    )
    assert any("ceiling" in p and "serving host" in p for p in problems)
    assert check_launch_gates(
        MATRIX, MATRIX["runs"][0], artifacts=_artifacts(tmp_path),
        batch_size=100, seed=0, variant="R_ts", suite="libero_10",
        remote_live_bytes=1024,
    ) == []


# ---------------------------------------------------------------------------
# Pilot closed loop
# ---------------------------------------------------------------------------


def test_realized_teacher_rate_counts_executed_arms() -> None:
    """Measured on what ran, not on what was sampled: a cache arm that hit an
    empty library executed the teacher."""
    from exp.rl_router.pilot_lambda import realized_teacher_rate

    rows = [
        {"router_outputs": {"arm_sampled": "cache", "arm_executed": "teacher"}},
        {"router_outputs": {"arm_sampled": "student", "arm_executed": "student"}},
        {"_kind": "episode_summary"},
    ]
    assert realized_teacher_rate(rows) == 0.5
    with pytest.raises(ValueError, match="no executed arms"):
        realized_teacher_rate([{"_kind": "client_timing"}])


def test_pilot_candidates_share_the_checkpoint_and_seed(tmp_path) -> None:
    """Only λ may vary between candidates; anything else would make the
    comparison uninterpretable."""
    from exp.rl_router.pilot_lambda import LAMBDA_GRID, candidate_command

    template = ("train --lam {lam} --out {candidate_dir} --split {pilot_split} "
                "--eval-split {remainder_split} --init {warmstart_weights} "
                "--batches {batches}")
    rendered = [
        candidate_command(template=template, lam=lam, candidate_dir=tmp_path / f"l{lam}",
                          pilot_split_path="p.yaml", remainder_split_path="r.yaml",
                          warmstart_weights="warm.pt")
        for lam in LAMBDA_GRID
    ]
    assert all("--init warm.pt" in cmd for cmd in rendered)
    assert all("--split p.yaml --eval-split r.yaml" in cmd for cmd in rendered)
    assert len({cmd.split("--lam ")[1].split()[0] for cmd in rendered}) == len(LAMBDA_GRID)


def test_pilot_split_yaml_is_consumable_by_the_run_loop(tmp_path) -> None:
    from exp.rl_router.pilot_lambda import pilot_split, write_split_yaml
    from exp.rl_router.run_rl_router import btrain_pairs

    source = {f"task_{t}": {"train": list(range(45)), "val": [45]} for t in range(2)}
    split = pilot_split(source, inits_per_task=30, seed=0)
    pilot_yaml = write_split_yaml(split["pilot"], source, tmp_path / "pilot.yaml")
    remainder_yaml = write_split_yaml(split["remainder"], source, tmp_path / "rem.yaml")

    pilot_pairs = btrain_pairs(pilot_yaml)
    remainder_pairs = btrain_pairs(remainder_yaml)
    assert len(pilot_pairs) == 60 and len(remainder_pairs) == 30
    assert not (set(pilot_pairs) & set(remainder_pairs))


# ---------------------------------------------------------------------------
# Sampling still honours the frozen batch composition
# ---------------------------------------------------------------------------


def test_sampling_is_unchanged_by_the_rewrite() -> None:
    pairs = [(t, i) for t in range(10) for i in range(45)]
    b0 = sample_batch(pairs, batch_size=100, batch_idx=0, seed=0)
    assert b0 == sample_batch(pairs, batch_size=100, batch_idx=0, seed=0)
    assert len(set(b0)) == 100


# ---------------------------------------------------------------------------
# Cost artifact shape (D4)
# ---------------------------------------------------------------------------


def test_costs_normalize_to_teacher_one() -> None:
    from exp.rl_router.microbench_cost import normalize_costs

    costs = normalize_costs({
        "teacher": {"mean_s": 0.4}, "student": {"mean_s": 0.08}, "cache": {"mean_s": 0.002},
    })
    assert costs["teacher"] == 1.0
    assert costs["student"] == pytest.approx(0.2)
    assert costs["cache"] == pytest.approx(0.005)


def test_costs_require_the_teacher_as_the_unit() -> None:
    from exp.rl_router.microbench_cost import normalize_costs

    with pytest.raises(ValueError, match="defines the unit"):
        normalize_costs({"student": {"mean_s": 0.1}})


def test_combined_artifact_carries_gpu_provenance() -> None:
    """The launch gate reads this provenance to tell a real GPU measurement
    from a placeholder or a client-side round trip."""
    from exp.rl_router.microbench_cost import combine_records

    records = {
        "teacher": {"gpu": {"mean_s": 0.4, "gpu_timed": True, "method": "cuda_event",
                            "device": "RTX 4090", "host": "wls", "torch": "2.7"},
                    "wall": {"mean_s": 0.45}, "path": "stage1+2+3"},
        "cache": {"gpu": {"mean_s": 0.002, "gpu_timed": True, "method": "cuda_event",
                          "device": "RTX 4090", "host": "wls", "torch": "2.7"},
                  "wall": {"mean_s": 0.003}, "path": "PayloadView.get + broadcast_action"},
    }
    artifact = combine_records(records)
    assert artifact["normalized_costs"]["teacher"] == 1.0
    assert artifact["provenance"]["teacher"]["gpu_timed"] is True
    assert artifact["provenance"]["cache"]["path"].startswith("PayloadView")
    # wall-clock is present but explicitly outside the reward.
    assert "wall_clock_seconds" in artifact and "do NOT enter the reward" in artifact["note"]


def test_cache_arm_cost_is_measured_not_assumed_zero() -> None:
    """A cache hit still pays a storage lookup and the history broadcast.
    Hard-coding it to zero would flatter the cache arm in every reward."""
    from exp.rl_router.microbench_cost import measure_cache

    from tests.cache.conftest import insert_entry, make_orchestrator

    from openpi.cache.storage_types import CachePayload
    from openpi.cache.types import CheckpointID

    orch, _, storage = make_orchestrator()
    state = torch.zeros(1, 32)
    state[0, 0] = 1.0
    entry = insert_entry(storage, CheckpointID.CP1, state,
                         CachePayload(action_chunk=torch.randn(50, 32)))
    record = measure_cache(orch, entry.id, torch.randn(50, 32), repeats=3)
    assert record["gpu"]["mean_s"] > 0.0
    assert record["path"].startswith("PayloadView.get")
    assert "gpu_timed" in record["gpu"]


# ---------------------------------------------------------------------------
# Sidecar admission (the behaviour authority gets the same gate as the features)
# ---------------------------------------------------------------------------


def _episode_on_disk(shard_dir: pathlib.Path, *, uid="u0", attempt=1, batch="b0",
                     version="v3", rows=3, sidecar_rows=None, dim=8):
    import hashlib

    shard_dir.mkdir(parents=True, exist_ok=True)
    payload = b"\x00\x3c" * (rows * dim)
    (shard_dir / f"{uid}.bin").write_bytes(payload)
    entries = sidecar_rows if sidecar_rows is not None else [
        {"task_uid": uid, "attempt": attempt, "batch_id": batch, "weights_version": version,
         "decision_idx": i, "arm_sampled": "teacher", "arm_mapped": "teacher",
         "logits": [0.1, 0.2, 0.3], "logprob_sampled": -1.0}
        for i in range(rows)
    ]
    sidecar_bytes = "".join(json.dumps(e) + "\n" for e in entries).encode("utf-8")
    (shard_dir / f"{uid}.jsonl").write_bytes(sidecar_bytes)
    return {
        "run_id": "run0", "batch_id": batch, "task_uid": uid, "attempt": attempt,
        "weights_version": version, "shard": f"{uid}.bin", "sidecar": f"{uid}.jsonl",
        "rows": rows, "dim": dim, "dtype": "float16",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "sidecar_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
        "status": "complete",
    }


def _load(tmp_path, shard, *, rows=3):
    from exp.rl_router.batch_package import build_batch_manifest
    from exp.rl_router.train_router import load_batch

    client = [
        {"task_uid": shard["task_uid"], "attempt": shard["attempt"],
         "router_outputs": {"decision_idx": i, "arm_executed": "teacher",
                            "weights_version": shard["weights_version"]}}
        for i in range(rows)
    ]
    (tmp_path / "per_step_rows_batch.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in client), encoding="utf-8")
    manifest = build_batch_manifest(
        batch_id="b0", weights_version="v3", expected_slots=[shard["task_uid"]],
        journal=[{"task_uid": shard["task_uid"], "attempt": shard["attempt"],
                  "accepted": True, "success": True, "error": None}],
        client_rows=client, shards=[shard],
    )
    return load_batch(manifest, shard_dir=tmp_path / "shards",
                      package_dir=tmp_path, n_arms=3)


def test_sidecar_with_duplicate_decision_idx_is_rejected(tmp_path) -> None:
    """A repeated index yields a correctly-sized list, so a length check cannot
    see it — and it would double-weight one step's gradient."""
    rows = [
        {"task_uid": "u0", "attempt": 1, "batch_id": "b0", "weights_version": "v3",
         "decision_idx": idx, "arm_sampled": "teacher", "arm_mapped": "teacher",
         "logits": [0.1, 0.2, 0.3], "logprob_sampled": -1.0}
        for idx in (0, 0, 1)
    ]
    shard = _episode_on_disk(tmp_path / "shards", sidecar_rows=rows)
    with pytest.raises(EpisodeAdmissionError) as excinfo:
        _load(tmp_path, shard)
    assert excinfo.value.rejected[0]["reason"] == "sidecar_decision_idx_discontinuous"
    assert excinfo.value.rejected[0]["task_uid"] == "u0"


def test_sidecar_from_another_episode_is_rejected(tmp_path) -> None:
    """Five-key identity, per row: a neighbouring episode's rows would join
    silently and credit the gradient to the wrong trajectory."""
    rows = [
        {"task_uid": "OTHER", "attempt": 1, "batch_id": "b0", "weights_version": "v3",
         "decision_idx": i, "arm_sampled": "teacher", "arm_mapped": "teacher",
         "logits": [0.1, 0.2, 0.3], "logprob_sampled": -1.0}
        for i in range(3)
    ]
    shard = _episode_on_disk(tmp_path / "shards", sidecar_rows=rows)
    with pytest.raises(EpisodeAdmissionError) as excinfo:
        _load(tmp_path, shard)
    assert excinfo.value.rejected[0]["reason"] == "sidecar_identity_mismatch"
    assert excinfo.value.rejected[0]["task_uid"] == "u0"


def test_sidecar_from_a_stale_attempt_is_rejected(tmp_path) -> None:
    rows = [
        {"task_uid": "u0", "attempt": 99, "batch_id": "b0", "weights_version": "v3",
         "decision_idx": i, "arm_sampled": "teacher", "arm_mapped": "teacher",
         "logits": [0.1, 0.2, 0.3], "logprob_sampled": -1.0}
        for i in range(3)
    ]
    shard = _episode_on_disk(tmp_path / "shards", sidecar_rows=rows)
    with pytest.raises(EpisodeAdmissionError) as excinfo:
        _load(tmp_path, shard)
    assert excinfo.value.rejected[0]["reason"] == "sidecar_identity_mismatch"
    assert excinfo.value.rejected[0]["task_uid"] == "u0"


def test_tampered_sidecar_fails_its_digest(tmp_path) -> None:
    shard = _episode_on_disk(tmp_path / "shards")
    (tmp_path / "shards" / "u0.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(EpisodeAdmissionError) as excinfo:
        _load(tmp_path, shard)
    assert excinfo.value.rejected[0]["reason"] == "sidecar_digest_mismatch"
    assert excinfo.value.rejected[0]["task_uid"] == "u0"


def test_a_well_formed_sidecar_loads(tmp_path) -> None:
    shard = _episode_on_disk(tmp_path / "shards")
    (episode,) = _load(tmp_path, shard)
    assert episode.features.shape == (3, 8)
    assert episode.dumped_logprobs.tolist() == [-1.0, -1.0, -1.0]


# ---------------------------------------------------------------------------
# Quarantine: a parity-rejected attempt must not be re-selected
# ---------------------------------------------------------------------------


def test_quarantined_attempt_is_excluded_from_the_next_join() -> None:
    """Without this the deterministic priority re-picks the same parity-bad
    attempt every round and the repair loop can never converge."""
    from exp.rl_router.batch_package import build_batch_manifest

    def manifest(quarantine):
        return build_batch_manifest(
            batch_id="b0", weights_version="v3", expected_slots=["u0"],
            journal=[{"task_uid": "u0", "attempt": 1, "accepted": True,
                      "success": True, "error": None},
                     {"task_uid": "u0#r1", "attempt": 1, "accepted": True,
                      "success": True, "error": None}],
            client_rows=[
                {"task_uid": uid, "attempt": 1,
                 "router_outputs": {"decision_idx": i, "weights_version": "v3"}}
                for uid in ("u0", "u0#r1") for i in range(2)
            ],
            shards=[
                {"task_uid": uid, "attempt": 1, "batch_id": "b0", "weights_version": "v3",
                 "rows": 2, "dim": 8, "status": "complete", "shard": f"{uid}.bin",
                 "sidecar": f"{uid}.jsonl", "sha256": "x"}
                for uid in ("u0", "u0#r1")
            ],
            quarantine=quarantine,
        )

    assert [r.task_uid for r in manifest(None).selected] == ["u0"]
    repaired = manifest([("u0", 1)])
    assert [r.task_uid for r in repaired.selected] == ["u0#r1"]
    assert any(r["reason"] == "quarantined_by_trainer_admission" for r in repaired.rejected)


def test_trainer_parity_failure_drives_a_repair_round() -> None:
    """A parity rejection is repairable, not fatal: §3.5 says repair, then
    update once on the full N."""
    from exp.rl_router.run_rl_router import run_batch_with_repair

    slots = make_slots([(0, i) for i in range(3)], yaml_id="rlr")
    dispatched, quarantines = [], []

    def runner(strategy, repair):
        dispatched.append([uid for _, _, uid in strategy._slots])
        return [], []

    def join(journal_rows, client_rows, round_idx, quarantine):
        quarantines.append(list(quarantine))
        return True, []

    calls = {"n": 0}

    def train(round_idx):
        calls["n"] += 1
        if calls["n"] == 1:
            return False, [{"task_uid": slots[1][2], "attempt": 1, "reason": "logits"}]
        return True, []

    landed = run_batch_with_repair(
        round_runner=runner,
        strategy_factory=lambda pending, repair: type("S", (), {"_slots": pending})(),
        slots=slots, join=join, train=train, batch_id="b0",
    )
    assert landed == 1
    assert len(dispatched) == 2 and dispatched[1] == [f"{slots[1][2]}#r1"]
    assert quarantines[1] == [(slots[1][2], 1)]      # carried into the next join


# ---------------------------------------------------------------------------
# main-level golden: two isolated filesystems, no shared mount
# ---------------------------------------------------------------------------


def test_main_loop_runs_two_batches_across_isolated_filesystems(tmp_path, monkeypatch) -> None:
    """The whole loop against a REMOTE tree the conductor never opens directly.

    This is the shape the earlier rounds got wrong: local paths handed to a
    remote trainer, then read back locally. Here the remote root is a distinct
    directory, the fake trainer only ever touches remote paths, and any local
    read of a remote artifact would surface as a missing file.
    """
    import subprocess

    from exp.rl_router import run_rl_router as loop

    local, remote_root, shard_root = tmp_path / "LOCAL", tmp_path / "REMOTE", tmp_path / "SHARDS"
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "TASK.init").write_bytes(b"x")

    split = tmp_path / "split.yaml"
    split.write_text(yaml.safe_dump(
        {f"task_{t}": {"train": list(range(4)), "val": [4]} for t in range(2)}), encoding="utf-8")
    base_yaml = _base_arm_yaml(tmp_path)
    cfg = yaml.safe_load(base_yaml.read_text())
    cfg["checkpoints"]["cp1"]["judge"]["seed"] = 0
    base_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    warm = tmp_path / "warm_v0.pt"
    _weights(warm, "v0")
    WARM_SHA = hashlib.sha256(warm.read_bytes()).hexdigest()
    costs = tmp_path / "costs.json"
    costs.write_text(json.dumps({
        "normalized_costs": {"teacher": 1.0, "student": 0.2},
        "provenance": {
            "teacher": {"gpu_timed": True, "in_process": True},
            "student": {"gpu_timed": True, "in_process": True,
                        "task_prompts": ["pick up the bowl"]},
        },
    }), encoding="utf-8")
    from exp.rl_router.launch_gates import M4_SCHEMA
    from exp.rl_router.pilot_lambda import PILOT_SCHEMA

    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({
        "schema": PILOT_SCHEMA, "separated": True,
        "selected": {"lambda_1": 0.2, "lambda_2": 0.5},
        "protocol": {"batches": 5, "batch_size": 100, "eval_episodes": 100,
                     "mode": "argmax", "eval_pool": "b_train_remainder"},
        "pilot_split_sha256": "p" * 64, "remainder_split_sha256": "r" * 64,
        "seed": 0, "candidates_run": 3,
"runs": {
            str(lam): {
                "lambda": lam, "lambda_recorded": lam, "seed": 0,
                "batch_size": 100, "train_judge_mode": "sample",
                "batches_trained": 5, "version_chain_contiguous": True,
                "weights_versions": [["v0", "v1"], ["v1", "v2"], ["v2", "v3"],
                                     ["v3", "v4"], ["v4", "v5"]],
                "start_weights_version": "v0", "final_weights_version": "v5",
                "eval_mode": "argmax", "eval_episodes": 100,
                "eval_weights_versions": ["v5"], "teacher_rate": rate,
                "train_split_sha256": "p" * 64,
                "expected_warmstart_sha256": WARM_SHA,
                "expected_pilot_split_sha256": "p" * 64,
                "expected_remainder_split_sha256": "r" * 64,
            }
            for lam, rate in ((0.05, 0.62), (0.2, 0.41), (0.5, 0.18))
        },
    }), encoding="utf-8")
    smoke = tmp_path / "smoke.json"
    smoke.write_text(json.dumps({
        "schema": M4_SCHEMA, "passed": True, "violations": [], "run_id": "r0",
        "batch_id": "b0000", "package_sha256": "s" * 64, "episodes": 20,
        "bytes_per_step": 131136, "peak_bytes": 1024,
        "bytes_before_reclaim": 1024, "bytes_after_reclaim": 0,
        "weights_versions": ["v0", "v1"],
    }), encoding="utf-8")
    artifacts = tmp_path / "artifacts.json"
    artifacts.write_text(json.dumps({
        "arm_costs": str(costs), "warmstart_weights": str(warm),
        "pilot": str(pilot), "capacity_smoke": str(smoke),
    }), encoding="utf-8")

    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(yaml.safe_dump({
        "batch_size": 4, "episodes_per_run": 8,
        "decisions": {"D3_training_init_domain": "b_train"},
        "suites": {"libero_10": {"split": str(split)}},
        "lambda": {"lambda_1": 0.2},
        "runs": [{"id": "r0", "suite": "libero_10", "variant": "R_ts",
                  "lambda": "lambda_1", "seed": 0, "flagship": True}],
        "eval": {"primary_seed": 0},
    }), encoding="utf-8")

    # --- fakes: the episodes and the trainer, both writing only where they may ---
    dispatched_yamls: list[str] = []

    def fake_run_round(*, strategy, yaml_id, servers, worker_specs, journal_path,
                       rows_path, bind_host, episode_timeout_s):
        dispatched_yamls.append(strategy._yaml_path)
        assert all(w.init_states_dir == str(pool) for w in worker_specs)
        batch_dir = pathlib.Path(journal_path).parent
        shard_dir = shard_root / "r0" / strategy._batch_id
        shard_dir.mkdir(parents=True, exist_ok=True)
        journal, client, manifest_lines = [], [], []
        for _, _, uid in strategy._slots:
            journal.append({"task_uid": uid, "attempt": 1, "accepted": True,
                            "success": True, "error": None})
            client.append({"task_uid": uid, "attempt": 1,
                           "router_outputs": {"decision_idx": 0, "arm_executed": "teacher",
                                              "weights_version": strategy._weights_version}})
            manifest_lines.append(json.dumps({
                "run_id": "r0", "batch_id": strategy._batch_id, "task_uid": uid,
                "attempt": 1, "weights_version": strategy._weights_version,
                "rows": 1, "dim": 8, "status": "complete", "shard": f"{uid}.bin",
                "sidecar": f"{uid}.jsonl", "sha256": "x",
            }))
            (shard_dir / f"{uid}.bin").write_bytes(b"\x00" * 16)
        (shard_dir / "manifest.jsonl").write_text("\n".join(manifest_lines) + "\n",
                                                  encoding="utf-8")
        pathlib.Path(journal_path).write_text(
            "".join(json.dumps(r) + "\n" for r in journal), encoding="utf-8")
        pathlib.Path(rows_path).write_text(
            "".join(json.dumps(r) + "\n" for r in client), encoding="utf-8")
        del batch_dir
        return journal, client

    monkeypatch.setattr(loop, "run_round", fake_run_round)

    def fake_remote_manifest(transport, *, remote_package, remote_shards,
                             remote_manifest, workdir=".", python="uv run"):
        pkg = pathlib.Path(remote_package)
        accepted = json.loads((pkg / "accepted_manifest.json").read_text())
        rows = [json.loads(x) for x in
                (pkg / "per_step_rows_batch.jsonl").read_text().splitlines() if x.strip()]
        doc = {"batch_id": accepted["batch_id"],
               "weights_version": accepted["weights_version"],
               "complete": True, "missing_slots": [],
               "training_selected": [{"task_uid": r["task_uid"], "rows": 1, "dim": 8}
                                     for r in rows],
               "rejected": [], "superseded": []}
        pathlib.Path(remote_manifest).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(remote_manifest).write_text(json.dumps(doc), encoding="utf-8")
        return True, "ok"

    monkeypatch.setattr(loop, "remote_build_manifest", fake_remote_manifest)

    def fake_run(command: str):
        """Stand-in for the remote shell: only ever touches the REMOTE tree."""
        if command.startswith("test -f "):
            return (0 if pathlib.Path(command.split()[-1]).exists() else 1), ""
        if "batch_package.py capacity" in command:
            root = command.split("--root ")[1].split()[0]
            live = sum(f.stat().st_size for f in pathlib.Path(root).rglob("*.bin")) \
                if pathlib.Path(root).exists() else 0
            return 0, json.dumps({"root": root, "live_bytes": live})
        if "train_router.py --state-only" in command:
            return 0, ""
        if "train_router.py" in command:
            version = command.split("--weights-out ")[1].split()[0].rsplit("/", 1)[1][:-3]
            batch_dir = pathlib.Path(command.split("--export-meta ")[1].split()[0]).parent
            _weights(pathlib.Path(command.split("--weights-out ")[1].split()[0]), version)
            batch_dir.mkdir(parents=True, exist_ok=True)
            (batch_dir / "export_meta.json").write_text(
                json.dumps({"weights_version": version}), encoding="utf-8")
            state = pathlib.Path(command.split("--state ")[1].split()[0])
            prior = json.loads(state.read_text())["consumed_batches"] if state.exists() else []
            batch_id = batch_dir.name
            state.parent.mkdir(parents=True, exist_ok=True)
            state.write_text(json.dumps({
                "weights_version": version,
                "consumed_batches": prior + [{"batch_id": batch_id, "package_sha256": "s"}],
            }), encoding="utf-8")
            return 0, ""
        if "reclaim" in command:
            return 0, ""
        return 0, ""

    monkeypatch.setattr(LocalTransport, "run", lambda self, command: fake_run(command))
    # Guard only OUR shell invocations: the stdlib itself shells out (e.g.
    # platform.processor()), and blanketing that would fail for the wrong reason.
    real_run = subprocess.run

    def _no_shell(*a, **k):
        if k.get("shell"):
            raise AssertionError(f"unexpected shell command in an isolated test: {a[:1]}")
        return real_run(*a, **k)

    monkeypatch.setattr(subprocess, "run", _no_shell)

    monkeypatch.setattr("sys.argv", [
        "run_rl_router.py", "--matrix", str(matrix), "--run-id", "r0",
        "--arm-yaml", str(base_yaml), "--init-states-dir", str(pool),
        "--artifacts", str(artifacts), "--servers", "h:8000", "--workers", "1",
        "--out-dir", str(local), "--shard-root", str(shard_root),
        "--remote-root", str(remote_root),
        "--trainer-cmd",
        ("uv run exp/rl_router/train_router.py --manifest {manifest} --package {package} "
         "--shards {shards} --weights-in {weights_in} --weights-out {weights_out} "
         "--checkpoint {checkpoint} --metrics {metrics} --export-meta-out {export_meta} "
         "--state-out {state} --rejected-out {rejected} --lam {lam} --t-max {t_max} "
         "--arm-costs {arm_costs} --export-meta {export_meta} --state {state}"),
    ])

    loop.main()

    # Two batches ran, each shipping a yaml pointing at the version before it.
    assert len(dispatched_yamls) == 2
    versions = [json.loads((local / b / "versions.json").read_text())
                for b in ("b0000", "b0001")]
    assert versions == [["v0", "v1"], ["v1", "v2"]]
    for batch, expected in (("b0000", "v0"), ("b0001", "v1")):
        cfg = yaml.safe_load((local / batch / f"r_ts_train__{expected}.yaml").read_text())
        weights_path = cfg["checkpoints"]["cp1"]["judge"]["weights_path"]
        assert weights_path.startswith(str(remote_root))       # remote address
        assert not weights_path.startswith(str(local))
    # Round-scoped packages, so a repair generation can never collide.
    assert (remote_root / "r0" / "b0000" / "package" / "r0" / "COMPLETE.json").exists()
    assert (local / "run_manifest.json").exists()


# ---------------------------------------------------------------------------
# SshTransport command construction (the real two-host path)
# ---------------------------------------------------------------------------


def test_ssh_transport_builds_port_scoped_quoted_commands(monkeypatch) -> None:
    """The tether exposes ssh on a non-default port, and paths can contain
    characters a shell would re-interpret; both have to survive."""
    from exp.rl_router.batch_package import SshTransport

    seen: list[str] = []

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr("subprocess.run", lambda cmd, **kw: (seen.append(cmd), _Proc())[1])
    transport = SshTransport("wls", port=14024, user="ziyang")

    transport.push_dir(pathlib.Path("/local/pkg"), "/remote/run/b0",
                       names=["journal_slice.jsonl"])
    assert any(c.startswith("ssh -p 14024 ziyang@wls mkdir -p ") for c in seen)
    assert any(c.startswith("scp -P 14024 /local/pkg/journal_slice.jsonl "
                            "ziyang@wls:/remote/run/b0/journal_slice.jsonl") for c in seen)

    seen.clear()
    transport.run("cd /repo && uv run x.py --a 'b c'")
    assert seen[0].startswith("ssh -p 14024 ziyang@wls ")
    assert "'cd /repo && uv run x.py --a '\"'\"'b c'\"'\"''" in seen[0]


def test_ssh_transport_raises_on_a_failed_push(monkeypatch) -> None:
    from exp.rl_router.batch_package import SshTransport, TransportError

    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "permission denied"

    monkeypatch.setattr("subprocess.run", lambda cmd, **kw: _Proc())
    with pytest.raises(TransportError, match="permission denied"):
        SshTransport("wls", port=14024).push_dir(pathlib.Path("/l"), "/r", names=["f"])


# ---------------------------------------------------------------------------
# Per-result row persistence (the crash window the journal opens)
# ---------------------------------------------------------------------------


def test_rows_land_when_the_result_arrives_not_at_stage_end(tmp_path) -> None:
    """``per_step_writer`` fires at stage completion; ``monitor.on_result``
    fires per episode, next to the journal write. A crash after the journal but
    before stage end must not lose that episode's evidence — the journal makes a
    restarted driver skip the uid, so those rows can never be re-collected."""
    from types import SimpleNamespace

    from exp.rl_router.run_rl_router import _ResultRowPersister
    from exp.rl_router.batch_package import read_jsonl

    rows_path = tmp_path / "client_rows.jsonl"
    persister = _ResultRowPersister(rows_path)

    persister.on_result(SimpleNamespace(
        task_uid="u0", attempt=1,
        per_step_rows=[{"router_outputs": {"decision_idx": 0}}],
    ))
    # Mid-stage: the file already holds the episode.
    assert [r["task_uid"] for r in read_jsonl(rows_path)] == ["u0"]

    persister.on_result(SimpleNamespace(
        task_uid="u1", attempt=1,
        per_step_rows=[{"router_outputs": {"decision_idx": 0}}],
    ))
    assert len(read_jsonl(rows_path)) == 2


def test_the_stage_end_drain_cannot_double_append(tmp_path) -> None:
    from types import SimpleNamespace

    from exp.rl_router.run_rl_router import _ResultRowPersister
    from exp.rl_router.batch_package import read_jsonl

    rows_path = tmp_path / "client_rows.jsonl"
    persister = _ResultRowPersister(rows_path)
    row = {"task_uid": "u0", "attempt": 1, "router_outputs": {"decision_idx": 0}}
    persister.on_result(SimpleNamespace(task_uid="u0", attempt=1, per_step_rows=[dict(row)]))
    persister.per_step_writer("y", [dict(row)])          # stage-completion drain
    persister.write([dict(row)])                          # run() finalizer
    assert len(read_jsonl(rows_path)) == 1


def test_a_restarted_persister_does_not_duplicate_what_landed(tmp_path) -> None:
    from types import SimpleNamespace

    from exp.rl_router.run_rl_router import _ResultRowPersister
    from exp.rl_router.batch_package import read_jsonl

    rows_path = tmp_path / "client_rows.jsonl"
    first = _ResultRowPersister(rows_path)
    first.on_result(SimpleNamespace(task_uid="u0", attempt=1,
                                    per_step_rows=[{"router_outputs": {"decision_idx": 0}}]))
    # "crash", then a fresh process for the repair round.
    second = _ResultRowPersister(rows_path)
    second.on_result(SimpleNamespace(task_uid="u0", attempt=1,
                                     per_step_rows=[{"router_outputs": {"decision_idx": 0}}]))
    second.on_result(SimpleNamespace(task_uid="u1#r1", attempt=1,
                                     per_step_rows=[{"router_outputs": {"decision_idx": 0}}]))
    assert [r["task_uid"] for r in read_jsonl(rows_path)] == ["u0", "u1#r1"]


# ---------------------------------------------------------------------------
# M4 against a REAL manifest produced by the packager
# ---------------------------------------------------------------------------


def test_m4_measures_bytes_from_a_real_packager_manifest(tmp_path) -> None:
    """The smoke used to read a ``dim`` the manifest never carried, so a real
    run could only ever report "could not measure"."""
    from exp.rl_router.batch_package import build_batch_manifest

    shard_dir = _shard_dir(tmp_path, n=1, size=64)
    uids = [f"u{i}" for i in range(20)]
    manifest = build_batch_manifest(
        batch_id="b0000", weights_version="v1", expected_slots=uids,
        journal=[{"task_uid": u, "attempt": 1, "accepted": True, "success": True,
                  "error": None} for u in uids],
        client_rows=[{"task_uid": u, "attempt": 1,
                      "router_outputs": {"decision_idx": i, "weights_version": "v1"}}
                     for u in uids for i in range(2)],
        shards=[{"task_uid": u, "attempt": 1, "batch_id": "b0000", "weights_version": "v1",
                 "rows": 2, "dim": 65568, "status": "complete", "shard": f"{u}.bin",
                 "sidecar": f"{u}.jsonl", "sha256": "x"} for u in uids],
    )
    report = m4_smoke(
        manifest=manifest.to_dict(),
        metrics=[{"batch_id": "b0000", "arm_executed_rate": {"teacher": 1.0}}],
        checkpoint_versions=["v0", "v1"],
        next_batch_shards=[{"weights_version": "v1"}],
        dump_root=shard_dir.parent,
    )
    assert report["passed"], report["violations"]
    assert report["bytes_per_step"] == 65568 * 2      # measured, not "unknown"
    assert report["episodes"] == 20


@pytest.mark.parametrize("versions", [["v0", "v9"], ["v3", "v3"], ["v2", "v1"]])
def test_m4_requires_exactly_one_version_increment(tmp_path, versions) -> None:
    """"Changed" is not the contract: v0 -> v9 means eight updates vanished."""
    shard_dir = _shard_dir(tmp_path, n=1, size=64)
    report = m4_smoke(
        manifest={"complete": True, "rejected": [], "missing_slots": [],
                  "training_selected": [{"rows": 1, "dim": 8} for _ in range(20)]},
        metrics=[{"batch_id": "b0", "arm_executed_rate": {}}],
        checkpoint_versions=versions,
        next_batch_shards=[{"weights_version": versions[-1]}],
        dump_root=shard_dir.parent,
    )
    assert any("advance by exactly one" in v for v in report["violations"])


# ---------------------------------------------------------------------------
# Warm-start admission by (uid, attempt)
# ---------------------------------------------------------------------------


def test_warmstart_never_pins_a_label_onto_a_stale_attempt(tmp_path) -> None:
    """Two attempts of one uid: the accepted attempt's success label must ride
    the accepted attempt's features. Collapsing on uid alone mislabels the
    training set in a way nothing downstream can detect."""
    from exp.rl_router.fit_warmstart import load_collection

    shards = tmp_path / "shards"
    shards.mkdir()
    manifest_rows = []
    for attempt, marker in ((1, 1.0), (2, 2.0)):
        payload = torch.full((2, 4), marker, dtype=torch.float16).numpy().tobytes()
        (shards / f"u0__a{attempt}.bin").write_bytes(payload)
        manifest_rows.append(json.dumps({
            "task_uid": "u0", "attempt": attempt, "batch_id": "collect",
            "weights_version": "constant-student", "rows": 2, "dim": 4,
            "status": "complete", "shard": f"u0__a{attempt}.bin",
            "sidecar": f"u0__a{attempt}.jsonl", "sha256": "x",
        }))
    (shards / "manifest.jsonl").write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")

    journal = tmp_path / "journal.jsonl"
    journal.write_text("\n".join([
        json.dumps({"task_uid": "u0", "attempt": 1, "accepted": True,
                    "success": True, "error": None}),
        json.dumps({"task_uid": "u0", "attempt": 2, "accepted": False,
                    "success": False, "error": None}),
    ]) + "\n", encoding="utf-8")

    data = load_collection(shards, journal, expected_slots=["u0"])
    assert data["n_episodes"] == 1
    # attempt 1's features (marker 1.0), attempt 1's label (success=True).
    assert float(data["features"][0, 0]) == 1.0
    assert data["labels"].tolist() == [1.0, 1.0]


def test_warmstart_counts_a_vanished_episode_against_the_failure_rate(tmp_path) -> None:
    """Scoring only the shards that exist divides failures by the survivors and
    reports a healthy collection exactly when it went worst."""
    from exp.rl_router.fit_warmstart import load_collection

    shards = tmp_path / "shards"
    shards.mkdir()
    payload = torch.ones(1, 4, dtype=torch.float16).numpy().tobytes()
    (shards / "u0.bin").write_bytes(payload)
    (shards / "manifest.jsonl").write_text(json.dumps({
        "task_uid": "u0", "attempt": 1, "rows": 1, "dim": 4, "status": "complete",
        "shard": "u0.bin", "sidecar": "u0.jsonl", "sha256": "x",
    }) + "\n", encoding="utf-8")
    journal = tmp_path / "journal.jsonl"
    journal.write_text(json.dumps({
        "task_uid": "u0", "attempt": 1, "accepted": True, "success": True, "error": None,
    }) + "\n", encoding="utf-8")

    data = load_collection(shards, journal, expected_slots=["u0", "u1", "u2", "u3"])
    assert data["n_expected"] == 4 and data["n_episodes"] == 1
    assert data["infra_failure_rate"] == 0.75
    assert {f["reason"] for f in data["failures"]} == {"no_accepted_attempt"}


# ---------------------------------------------------------------------------
# Real trainer CLI: publish -> update -> crash -> resume, no fakes
# ---------------------------------------------------------------------------


def _real_batch_on_disk(tmp_path, *, batch_id: str, version: str, uids: list[str],
                        weights_path: str, arms: str = "ts", dim: int = 32):
    """A package + shards the REAL trainer can consume, with real behaviour logits."""
    import hashlib

    from openpi.cache.components.mlp_router_judge import RouterWeights
    from exp.rl_router.train_router import RouterPolicy

    shard_dir = tmp_path / "SHARDS" / batch_id
    shard_dir.mkdir(parents=True, exist_ok=True)
    policy = RouterPolicy.from_weights(RouterWeights.load(weights_path))
    n_arms = len(policy.arms)

    manifest_lines, journal, client = [], [], []
    for idx, uid in enumerate(uids):
        features = torch.full((2, dim), 0.25 * (idx + 1), dtype=torch.float16)
        payload = features.numpy().tobytes()
        (shard_dir / f"{uid}.bin").write_bytes(payload)
        logits = policy.reference_logits(features)
        logp = torch.log_softmax(logits, dim=1)
        rows = []
        for step in range(2):
            rows.append({
                "task_uid": uid, "attempt": 1, "batch_id": batch_id,
                "weights_version": version, "decision_idx": step,
                "arm_sampled": policy.arms[0], "arm_mapped": policy.arms[0],
                "logits": [float(x) for x in logits[step].tolist()],
                "logprob_sampled": float(logp[step, 0]),
            })
        sidecar_bytes = "".join(json.dumps(r) + "\n" for r in rows).encode("utf-8")
        (shard_dir / f"{uid}.jsonl").write_bytes(sidecar_bytes)
        manifest_lines.append(json.dumps({
            "run_id": "r0", "batch_id": batch_id, "task_uid": uid, "attempt": 1,
            "weights_version": version, "rows": 2, "dim": dim, "status": "complete",
            "shard": f"{uid}.bin", "sidecar": f"{uid}.jsonl",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "sidecar_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
        }))
        journal.append({"task_uid": uid, "attempt": 1, "accepted": True,
                        "success": bool(idx % 2), "error": None})
        client.extend({
            "task_uid": uid, "attempt": 1,
            "router_outputs": {"decision_idx": s, "arm_executed": policy.arms[0],
                               "weights_version": version},
        } for s in range(2))
    (shard_dir / "manifest.jsonl").write_text("\n".join(manifest_lines) + "\n",
                                              encoding="utf-8")

    pkg = assemble_package(tmp_path / "PKG" / batch_id, batch_id=batch_id,
                           weights_version=version, journal_rows=journal,
                           client_rows=client, expected_slots=uids)
    from exp.rl_router.batch_package import build_batch_manifest

    manifest = build_batch_manifest(
        batch_id=batch_id, weights_version=version, expected_slots=uids,
        journal=journal, client_rows=client,
        shards=[json.loads(x) for x in manifest_lines], shard_dir=shard_dir,
    )
    assert manifest.complete, manifest.rejected
    manifest_path = tmp_path / f"{batch_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    return shard_dir, pkg, manifest_path, n_arms


def _run_trainer_cli(args: list[str]) -> None:
    """Invoke the real ``train_router`` CLI in-process."""
    import sys

    from exp.rl_router import train_router

    old = sys.argv
    sys.argv = ["train_router.py", *args]
    try:
        train_router.main()
    finally:
        sys.argv = old


def test_real_trainer_cli_publishes_a_servable_checkpoint_and_resumes(tmp_path) -> None:
    """First launch -> update -> crash before export -> recovery, with the real
    trainer, the real weights loader, and a real judge built from the result.

    Every earlier round's cross-machine bug hid behind a fake trainer: a fake
    never fails to stamp encoder meta, and never proves the exported file is one
    the serving side can actually load.
    """
    from openpi.cache.components.mlp_router_judge import MlpRouterJudge

    v0 = tmp_path / "weights" / "v0.pt"
    _weights_ts(v0, "v0")
    shard_dir, pkg, manifest_path, _ = _real_batch_on_disk(
        tmp_path, batch_id="b0000", version="v0", uids=["u0", "u1"], weights_path=str(v0))

    checkpoint = tmp_path / "trainer_checkpoint.pt"
    metrics = tmp_path / "metrics.jsonl"
    state = tmp_path / "trainer_state.json"
    v1 = tmp_path / "weights" / "v1.pt"
    common = [
        "--manifest", str(manifest_path), "--package", str(pkg), "--shards", str(shard_dir),
        "--checkpoint", str(checkpoint), "--metrics", str(metrics),
        "--lam", "0.2", "--t-max", "100",
        "--arm-costs", json.dumps({"teacher": 1.0, "student": 0.2}),
    ]
    _run_trainer_cli([*common, "--weights-in", str(v0), "--weights-out", str(v1),
                      "--export-meta-out", str(tmp_path / "export_meta.json"),
                      "--state-out", str(state)])

    # The exported file is servable: a real judge loads it and reports v1.
    assert MlpRouterJudge(arms="ts", weights_path=str(v1), hidden=H,
                          feature_fields=[ROBOT_STATE], mode="argmax").weights_version == "v1"
    assert json.loads(state.read_text())["weights_version"] == "v1"
    assert json.loads((tmp_path / "export_meta.json").read_text())["weights_version"] == "v1"

    # --- crash right after save_checkpoint: export, metrics and state all
    # missing, but the update itself is durable inside the checkpoint ---
    v1.unlink()
    state.unlink()
    metrics.write_text("", encoding="utf-8")
    _run_trainer_cli(["--export-only", "--checkpoint", str(checkpoint),
                      "--weights-out", str(v1), "--state-out", str(state),
                      "--metrics", str(metrics), "--manifest", "/dev/null",
                      "--package", "/dev/null", "--shards", "/dev/null",
                      "--lam", "0", "--t-max", "1", "--arm-costs", "{}",
                      "--export-meta-out", str(tmp_path / "export_meta.json")])
    # Recovery re-exports a SERVABLE file without re-running the batch, and the
    # batch's metrics row is rebuilt rather than lost.
    assert MlpRouterJudge(arms="ts", weights_path=str(v1), hidden=H,
                          feature_fields=[ROBOT_STATE], mode="argmax").weights_version == "v1"
    rows = [json.loads(x) for x in metrics.read_text().splitlines() if x.strip()]
    assert any(r.get("recovered") for r in rows)

    # --- state is a cache of the ledger: regenerate it from the checkpoint ---
    state.unlink()
    _run_trainer_cli(["--state-only", "--checkpoint", str(checkpoint),
                      "--state-out", str(state), "--manifest", "/dev/null",
                      "--package", "/dev/null", "--shards", "/dev/null",
                      "--weights-out", "/dev/null", "--metrics", "/dev/null",
                      "--lam", "0", "--t-max", "1", "--arm-costs", "{}"])
    published = json.loads(state.read_text())
    assert published["weights_version"] == "v1"
    assert [c["batch_id"] for c in published["consumed_batches"]] == ["b0000"]

    # --- re-pushing the consumed batch is a no-op, not a second update ---
    # Exporting here would stamp v1's parameters into a file named v2 and hand
    # the fleet a mislabelled policy.
    _run_trainer_cli([*common, "--weights-in", str(v0),
                      "--weights-out", str(tmp_path / "weights" / "v2.pt"),
                      "--state-out", str(state)])
    assert not (tmp_path / "weights" / "v2.pt").exists()
    assert json.loads(state.read_text())["weights_version"] == "v1"


def _weights_ts(path: pathlib.Path, version: str) -> str:
    torch.manual_seed(abs(hash(version)) % 997)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_router_weights(
        path, W1=torch.randn(H, 32) * 0.2, b1=torch.zeros(H),
        W2=torch.randn(2, H) * 0.2, b2=torch.zeros(2),
        arms="ts", fields=(ROBOT_STATE,), dims={ROBOT_STATE: 32},
        weights_version=version, mu=torch.zeros(32), sigma=torch.ones(32),
    )
    return str(path)


def test_export_only_fails_loud_without_encoder_meta(tmp_path) -> None:
    """A checkpoint that does not carry the serving encoder identity cannot
    produce a loadable weights file — that has to be an error, not a file the
    server will reject at start-up."""
    from exp.rl_router.train_router import RouterPolicy, RouterTrainer, TrainerHParams

    torch.manual_seed(0)
    trainer = RouterTrainer(
        RouterPolicy(torch.randn(H, 32) * 0.2, torch.zeros(H),
                     torch.randn(2, H) * 0.2, torch.zeros(2), arms="ts"),
        TrainerHParams(arm_costs={"teacher": 1.0, "student": 0.2}, lam=0.2, t_max=100),
    )
    with pytest.raises(ValueError, match="encoder meta"):
        trainer.export_router_weights(tmp_path / "v1.pt")


def test_checkpoint_round_trips_the_encoder_meta(tmp_path) -> None:
    from openpi.cache.components.mlp_router_judge import RouterWeights
    from exp.rl_router.train_router import (
        RouterPolicy, RouterTrainer, TrainerHParams, encoder_meta_from_weights,
    )

    path = _weights_ts(tmp_path / "v0.pt", "v0")
    weights = RouterWeights.load(path)
    trainer = RouterTrainer(
        RouterPolicy.from_weights(weights),
        TrainerHParams(arm_costs={"teacher": 1.0, "student": 0.2}, lam=0.2, t_max=100),
        weights_version="v0", encoder_meta=encoder_meta_from_weights(weights),
    )
    trainer.save_checkpoint(tmp_path / "ck.pt")
    reloaded = RouterTrainer.load_checkpoint(tmp_path / "ck.pt")
    assert reloaded.encoder_meta["fields"] == [ROBOT_STATE]
    reloaded.weights_version = "v1"
    reloaded.export_router_weights(tmp_path / "v1.pt")     # self-sufficient
    assert RouterWeights.load(str(tmp_path / "v1.pt")).weights_version == "v1"


# ---------------------------------------------------------------------------
# Destination-aware upload
# ---------------------------------------------------------------------------


def test_push_file_honours_the_destination_name(tmp_path) -> None:
    """The fleet reads ``weights/v0.pt``; the source is ``warmstart_l10.pt``.
    Pushing by source basename put the file where nothing looked for it and the
    very first batch could not find its policy."""
    source = tmp_path / "warmstart_l10.pt"
    source.write_bytes(b"weights")
    dest = str(tmp_path / "REMOTE" / "weights" / "v0.pt")
    LocalTransport().push_file(source, dest)
    assert pathlib.Path(dest).read_bytes() == b"weights"
    assert not (tmp_path / "REMOTE" / "weights" / "warmstart_l10.pt").exists()
    assert not list((tmp_path / "REMOTE" / "weights").glob("*.uploading"))


def test_ssh_push_file_stages_then_renames(monkeypatch) -> None:
    from exp.rl_router.batch_package import SshTransport

    seen: list[str] = []

    class _Proc:
        returncode = 0
        stdout = stderr = ""

    monkeypatch.setattr("subprocess.run", lambda cmd, **kw: (seen.append(cmd), _Proc())[1])
    SshTransport("wls", port=14024).push_file(pathlib.Path("/l/warm.pt"),
                                              "/remote/weights/v0.pt")
    assert any("scp -P 14024 /l/warm.pt wls:/remote/weights/v0.pt.uploading" in c for c in seen)
    assert any("mv /remote/weights/v0.pt.uploading /remote/weights/v0.pt" in c for c in seen)


# ---------------------------------------------------------------------------
# Sidecar defects are caught by the REMOTE join, not only by the trainer
# ---------------------------------------------------------------------------


def test_remote_join_rejects_a_duplicate_decision_idx_sidecar(tmp_path) -> None:
    """Caught during admission it is a repairable missing slot; caught later in
    the trainer's loader it was a generic crash the repair loop could not act on."""
    from exp.rl_router.batch_package import build_batch_manifest

    rows = [
        {"task_uid": "u0", "attempt": 1, "batch_id": "b0", "weights_version": "v3",
         "decision_idx": idx, "arm_sampled": "teacher", "arm_mapped": "teacher",
         "logits": [0.1], "logprob_sampled": -1.0}
        for idx in (0, 0, 1)
    ]
    shard = _episode_on_disk(tmp_path / "shards", sidecar_rows=rows)
    manifest = build_batch_manifest(
        batch_id="b0", weights_version="v3", expected_slots=["u0"],
        journal=[{"task_uid": "u0", "attempt": 1, "accepted": True, "success": True,
                  "error": None}],
        client_rows=[{"task_uid": "u0", "attempt": 1,
                      "router_outputs": {"decision_idx": i, "weights_version": "v3"}}
                     for i in range(3)],
        shards=[shard], shard_dir=tmp_path / "shards",
    )
    assert not manifest.complete and manifest.missing_slots == ["u0"]
    assert manifest.rejected[0]["reason"] == "sidecar_decision_idx_discontinuous"


def test_remote_join_accepts_a_clean_sidecar(tmp_path) -> None:
    from exp.rl_router.batch_package import build_batch_manifest

    shard = _episode_on_disk(tmp_path / "shards")
    manifest = build_batch_manifest(
        batch_id="b0", weights_version="v3", expected_slots=["u0"],
        journal=[{"task_uid": "u0", "attempt": 1, "accepted": True, "success": True,
                  "error": None}],
        client_rows=[{"task_uid": "u0", "attempt": 1,
                      "router_outputs": {"decision_idx": i, "weights_version": "v3"}}
                     for i in range(3)],
        shards=[shard], shard_dir=tmp_path / "shards",
    )
    assert manifest.complete and manifest.selected[0].dim == 8


# ---------------------------------------------------------------------------
# ACT microbench routes on real manifest prompts
# ---------------------------------------------------------------------------


def test_act_microbench_uses_real_manifest_prompts(tmp_path) -> None:
    """ACT is a per-task ensemble routed by EXACT prompt match, so a synthetic
    prompt cannot even complete warmup — the owner's primary student arm would
    have no cost artifact at all."""
    from exp.ablation_study.sidecar_server import route_prompt
    from exp.rl_router.microbench_cost import act_manifest_prompts, synthetic_obs

    manifest = tmp_path / "act_manifest.json"
    manifest.write_text(json.dumps({
        "pick up the black bowl": "/ckpt/task0",
        "open the top drawer": "/ckpt/task1",
    }), encoding="utf-8")

    prompts = act_manifest_prompts(manifest)
    assert prompts == ["open the top drawer", "pick up the black bowl"]

    policies = {p: object() for p in prompts}
    for prompt in prompts:
        obs = synthetic_obs("libero_10", prompt=prompt)
        assert route_prompt(policies, obs["prompt"]) is policies[prompt]

    with pytest.raises(KeyError):
        route_prompt(policies, synthetic_obs("libero_10")["prompt"])


def test_measure_student_refuses_an_rpc_wrapper() -> None:
    from openpi.cache.sidecar_executor import SidecarExecutor
    from exp.rl_router.microbench_cost import measure_student

    executor = SidecarExecutor("127.0.0.1:7002", label="probe")
    with pytest.raises(ValueError, match="not a SidecarExecutor"):
        measure_student(executor, {"prompt": "x"}, repeats=1)


def test_out_of_process_timing_is_never_marked_gpu_timed() -> None:
    """A near-zero event time on an idle local stream must not masquerade as the
    other process's GPU time."""
    from exp.rl_router.microbench_cost import gpu_timed

    record = gpu_timed(lambda: None, repeats=2, in_process=False)
    assert record["gpu_timed"] is False
    assert record["in_process"] is False
    assert record["method"] == "perf_counter_out_of_process"


# ---------------------------------------------------------------------------
# Round-7 boundaries
# ---------------------------------------------------------------------------


def test_full_metrics_ride_in_the_durable_ledger(tmp_path) -> None:
    """A four-field stub is not a recovered metrics row: loss / grad_norm /
    reward / arm rates would be lost for that batch forever, leaving a hole in
    the training curve exactly where something went wrong."""
    from exp.rl_router.train_router import (
        RouterPolicy, RouterTrainer, TrainEpisode, TrainerHParams,
    )

    torch.manual_seed(0)
    policy = RouterPolicy(torch.randn(H, 32) * 0.2, torch.zeros(H),
                          torch.randn(2, H) * 0.2, torch.zeros(2), arms="ts")
    trainer = RouterTrainer(
        policy, TrainerHParams(arm_costs={"teacher": 1.0, "student": 0.2},
                               lam=0.2, t_max=100),
    )
    features = torch.randn(3, 32).to(FEATURE_DTYPE)
    episode = TrainEpisode(task_uid="u0", attempt=1, success=True, features=features,
                           arm_sampled=["teacher"] * 3, arm_executed=["teacher"] * 3)
    metrics = trainer.train_batch([episode], batch_id="b0000", package_sha256="pkg")

    stored = trainer.consumed_batches[-1]["metrics"]
    for key in ("loss", "grad_norm", "mean_reward", "mean_success",
                "mean_advantage_abs", "arm_executed_rate"):
        assert key in stored, key
    assert stored["loss"] == metrics["loss"]

    # And the published state summary stays lean — the conductor only needs ids.
    assert "metrics" not in trainer.state_summary()["consumed_batches"][0]


def test_metrics_are_backfilled_even_when_the_weights_survived(tmp_path) -> None:
    """Crash AFTER the export: the weights exist, so keying recovery off "are
    the weights missing" would leave that batch's row absent forever."""
    from exp.rl_router.train_router import _append_recovery_metrics

    class _Args:
        metrics = str(tmp_path / "metrics.jsonl")

    class _Trainer:
        consumed_batches = [
            {"batch_id": "b0000", "metrics": {"batch_id": "b0000", "loss": 0.5,
                                              "grad_norm": 1.0}},
            {"batch_id": "b0001", "metrics": {"batch_id": "b0001", "loss": 0.25,
                                              "grad_norm": 0.9}},
        ]

    pathlib.Path(_Args.metrics).write_text(
        json.dumps({"batch_id": "b0000", "loss": 0.5}) + "\n", encoding="utf-8")
    backfilled = _append_recovery_metrics(_Trainer(), _Args())
    assert backfilled == ["b0001"]
    rows = [json.loads(x) for x in pathlib.Path(_Args.metrics).read_text().splitlines()]
    recovered = [r for r in rows if r.get("recovered")]
    assert recovered[0]["loss"] == 0.25 and recovered[0]["grad_norm"] == 0.9


def test_repair_generation_survives_a_process_restart(tmp_path) -> None:
    """Journal and client rows are cumulative, so a restart at round 0 would
    rebuild an "r0" package containing round-1 rows — a different digest for a
    directory the server already holds, which the duplicate guard refuses."""
    from exp.rl_router.run_rl_router import run_batch_with_repair

    slots = make_slots([(0, i) for i in range(3)], yaml_id="rlr")
    state_path = tmp_path / "repair_state.json"
    rounds_seen: list[int] = []

    def runner(strategy, repair):
        rounds_seen.append(repair)
        return [], []

    # First process: round 0 finds a gap, persists round 1, then "crashes".
    with pytest.raises(RuntimeError):
        run_batch_with_repair(
            round_runner=runner,
            strategy_factory=lambda pending, repair: type("S", (), {"_slots": pending})(),
            slots=slots,
            join=lambda j, c, r, q: (_ for _ in ()).throw(RuntimeError("crash"))
            if r == 1 else (False, [slots[1][2]]),
            train=lambda r: (True, []), batch_id="b0", state_path=state_path,
        )
    assert json.loads(state_path.read_text())["round"] == 1

    # Second process resumes at round 1 — not 0 — and only the missing slot.
    rounds_seen.clear()
    dispatched: list[list[str]] = []

    def runner2(strategy, repair):
        rounds_seen.append(repair)
        dispatched.append([uid for _, _, uid in strategy._slots])
        return [], []

    run_batch_with_repair(
        round_runner=runner2,
        strategy_factory=lambda pending, repair: type("S", (), {"_slots": pending})(),
        slots=slots, join=lambda j, c, r, q: (True, []),
        train=lambda r: (True, []), batch_id="b0", state_path=state_path,
    )
    assert rounds_seen == [1]
    assert dispatched == [[f"{slots[1][2]}#r1"]]


@pytest.mark.parametrize("mutation,reason", [
    ("not_object", "sidecar_row_not_an_object"),
    ("missing_field", "sidecar_row_missing_field"),
    ("bad_logits", "sidecar_logits_malformed"),
    ("ragged_logits", "sidecar_logits_ragged"),
    ("bad_logprob", "sidecar_logprob_malformed"),
    ("bad_attempt", "sidecar_identity_malformed"),
])
def test_malformed_sidecars_are_named_rejections_not_crashes(tmp_path, mutation, reason) -> None:
    """Anything that would raise inside the trainer's loader must be a named
    per-slot rejection here, or it escapes the bounded repair loop."""
    from exp.rl_router.batch_package import sidecar_defect

    base = {"task_uid": "u0", "attempt": 1, "batch_id": "b0", "weights_version": "v3",
            "decision_idx": 0, "arm_sampled": "teacher", "arm_mapped": "teacher",
            "logits": [0.1, 0.2], "logprob_sampled": -1.0}
    rows: list = [dict(base), {**base, "decision_idx": 1}]
    if mutation == "not_object":
        rows[0] = [1, 2, 3]
    elif mutation == "missing_field":
        rows[0].pop("logprob_sampled")
    elif mutation == "bad_logits":
        rows[0]["logits"] = "not a list"
    elif mutation == "ragged_logits":
        rows[0]["logits"] = [0.1]
    elif mutation == "bad_logprob":
        rows[0]["logprob_sampled"] = "nan"
    else:
        rows[0]["attempt"] = "one"

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    payload = "".join(json.dumps(r) + "\n" for r in rows).encode("utf-8")
    (shard_dir / "u0.jsonl").write_bytes(payload)
    shard = {"task_uid": "u0", "attempt": 1, "batch_id": "b0", "weights_version": "v3",
             "sidecar": "u0.jsonl",
             "sidecar_sha256": hashlib.sha256(payload).hexdigest()}
    assert sidecar_defect(shard_dir, shard, expected_rows=2) == reason


def test_production_join_refuses_to_skip_sidecar_admission() -> None:
    from exp.rl_router.batch_package import build_batch_manifest

    with pytest.raises(ValueError, match="requires shard_dir"):
        build_batch_manifest(batch_id="b0", weights_version="v1", expected_slots=[],
                             journal=[], client_rows=[], shards=[], require_sidecar=True)


def test_loader_turns_any_defect_into_a_repairable_slot(tmp_path) -> None:
    """Even an unforeseen exception belongs to ONE slot; the right response is a
    zero-update repair of that slot, not an ALERT that stops the run."""
    from exp.rl_router.train_router import EpisodeAdmissionError

    shard_dir = tmp_path / "shards"
    shard = _episode_on_disk(shard_dir)
    (shard_dir / "u0.bin").write_bytes(b"\x00")        # truncated: reshape will raise
    with pytest.raises(EpisodeAdmissionError) as excinfo:
        _load(tmp_path, shard)
    assert excinfo.value.rejected[0]["task_uid"] == "u0"
    assert excinfo.value.rejected[0]["reason"] in {"episode_unreadable", "shard_digest_mismatch"}


def test_capacity_probe_failure_blocks_the_launch(tmp_path, monkeypatch) -> None:
    """An unmeasurable disk is not an empty disk."""
    from exp.rl_router import run_rl_router as loop

    class _Broken(LocalTransport):
        def run(self, command):
            return 1, "ssh: connect refused"

    args = type("A", (), {"remote_workdir": ".", "shard_root": "/data/x"})()
    with pytest.raises(SystemExit, match="could not measure remote capacity"):
        loop._remote_live_bytes(_Broken(), args)

    class _Garbled(LocalTransport):
        def run(self, command):
            return 0, "not json"

    with pytest.raises(SystemExit, match="unparsable"):
        loop._remote_live_bytes(_Garbled(), args)


def test_m4_gate_requires_bound_evidence(tmp_path) -> None:
    """A self-reported JSON without run/package identity or reclaim evidence is
    a claim, not capacity measurement."""
    from exp.rl_router.launch_gates import M4_SCHEMA, _smoke_problems

    path = tmp_path / "smoke.json"
    path.write_text(json.dumps({
        "schema": M4_SCHEMA, "passed": True, "violations": [], "batch_id": "b0000",
        "episodes": 20, "bytes_per_step": 131136, "peak_bytes": 1024,
        "weights_versions": ["v0", "v1"],
    }), encoding="utf-8")
    problems = _smoke_problems(str(path))
    assert any("run id" in p for p in problems)
    assert any("package" in p for p in problems)
    assert any("reclaim" in p for p in problems)


def test_pilot_gate_rejects_a_self_reported_record_with_no_candidates(tmp_path) -> None:
    """``candidates_run: 0`` with an empty ``runs`` selects a λ that nothing
    shows was ever trained."""
    from exp.rl_router.launch_gates import _pilot_problems
    from exp.rl_router.pilot_lambda import PILOT_SCHEMA

    path = tmp_path / "pilot.json"
    path.write_text(json.dumps({
        "schema": PILOT_SCHEMA, "seed": 0, "separated": True, "candidates_run": 0,
        "selected": {"lambda_1": 0.2, "lambda_2": 0.5}, "runs": {},
        "protocol": {"batches": 5, "batch_size": 100, "eval_episodes": 100,
                     "mode": "argmax"},
    }), encoding="utf-8")
    problems = _pilot_problems(str(path), {"lambda": {"lambda_1": 0.2}},
                               {"lambda": "lambda_1"})
    assert any("no per-candidate manifests" in p for p in problems)
    assert any("frozen grid" in p for p in problems)


def test_pilot_candidate_manifest_reads_the_candidates_own_artifacts(tmp_path) -> None:
    """A command string proves nothing — an arbitrary template can ignore the
    placeholders. The manifest is built from what the candidate produced."""
    from exp.rl_router.pilot_lambda import candidate_manifest

    candidate = tmp_path / "lam_0.2"
    for idx, versions in enumerate((["v0", "v1"], ["v1", "v2"], ["v2", "v3"],
                                    ["v3", "v4"], ["v4", "v5"])):
        batch = candidate / f"b{idx:04d}"
        batch.mkdir(parents=True)
        (batch / "versions.json").write_text(json.dumps(versions), encoding="utf-8")
    warm = tmp_path / "warm.pt"
    warm.write_bytes(b"w")
    split_a, split_b = tmp_path / "p.yaml", tmp_path / "r.yaml"
    split_a.write_bytes(b"a")
    split_b.write_bytes(b"b")

    record = candidate_manifest(
        candidate, lam=0.2, seed=0,
        eval_rows=[{"task_uid": f"u{i}", "attempt": 1} for i in range(100)],
        teacher_rate=0.41, command="…", warmstart_weights=str(warm),
        pilot_split=str(split_a), remainder_split=str(split_b),
    )
    assert record["batches_trained"] == 5
    assert record["final_weights_version"] == "v5"
    assert record["version_chain_contiguous"] is True
    assert record["eval_episodes"] == 100
    assert record["expected_warmstart_sha256"] == hashlib.sha256(b"w").hexdigest()
    # Nothing was recorded about the training run itself, because the candidate
    # produced no run manifest: missing evidence reads as None, not agreement.
    assert record["seed"] is None and record["train_judge_mode"] is None


# ---------------------------------------------------------------------------
# Round-9 boundaries
# ---------------------------------------------------------------------------


def test_smoke_mode_runs_exactly_two_twenty_episode_batches(tmp_path, monkeypatch) -> None:
    """``--smoke`` must be mechanically 20 x 2, never the formal matrix size.

    A ``--smoke`` that silently ran 100 x 40 would spend the entire run budget
    and then emit a report the M4 gate rejects for covering the wrong episode
    count — the operator would have paid for the whole experiment to learn
    nothing.
    """
    from exp.rl_router import run_rl_router as loop
    from exp.rl_router.launch_gates import SMOKE_EPISODES

    calls: list[tuple[int, int]] = []

    def fake_run_one_batch(**kwargs):
        calls.append((kwargs["batch_idx"], kwargs["batch_size"]))
        return _next_version_str(kwargs["weights_version"])

    monkeypatch.setattr(loop, "run_one_batch", fake_run_one_batch)
    monkeypatch.setattr(loop, "emit_m4_report",
                        lambda **kw: {"passed": True, "violations": [],
                                      "episodes": kw["episodes"]})
    _run_main_in_smoke_mode(tmp_path, monkeypatch)

    assert calls == [(0, SMOKE_EPISODES), (1, SMOKE_EPISODES)]
    report = json.loads((tmp_path / "LOCAL" / "m4_smoke.json").read_text())
    assert report["episodes"] == SMOKE_EPISODES


def _next_version_str(current: str) -> str:
    return f"v{int(current[1:]) + 1}"


def _run_main_in_smoke_mode(tmp_path, monkeypatch) -> None:
    """Drive ``main()`` in bootstrap mode against a matrix that is 100 x 40."""
    import subprocess

    from exp.rl_router import run_rl_router as loop
    from exp.rl_router.launch_gates import M4_SCHEMA
    from exp.rl_router.pilot_lambda import PILOT_SCHEMA

    local, remote_root, shard_root = tmp_path / "LOCAL", tmp_path / "REMOTE", tmp_path / "SHARDS"
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "TASK.init").write_bytes(b"x")
    split = tmp_path / "split.yaml"
    split.write_text(yaml.safe_dump(
        {f"task_{t}": {"train": list(range(50)), "val": []} for t in range(4)}),
        encoding="utf-8")
    base_yaml = _base_arm_yaml(tmp_path)
    cfg = yaml.safe_load(base_yaml.read_text())
    cfg["checkpoints"]["cp1"]["judge"]["seed"] = 0
    base_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    warm = tmp_path / "warm_v0.pt"
    _weights(warm, "v0")
    warm_sha = hashlib.sha256(warm.read_bytes()).hexdigest()

    costs = tmp_path / "costs.json"
    costs.write_text(json.dumps({
        "normalized_costs": {"teacher": 1.0, "student": 0.2},
        "provenance": {"teacher": {"gpu_timed": True, "in_process": True},
                       "student": {"gpu_timed": True, "in_process": True,
                                   "task_prompts": ["p"]}},
    }), encoding="utf-8")
    pilot = tmp_path / "pilot.json"
    pilot.write_text(json.dumps({
        "schema": PILOT_SCHEMA, "separated": True, "seed": 0, "candidates_run": 3,
        "selected": {"lambda_1": 0.2, "lambda_2": 0.5},
        "protocol": {"batches": 5, "batch_size": 100, "eval_episodes": 100,
                     "mode": "argmax", "eval_pool": "b_train_remainder"},
        "pilot_split_sha256": "p" * 64, "remainder_split_sha256": "r" * 64,
        "runs": {
            str(lam): {
                "lambda": lam, "lambda_recorded": lam, "seed": 0,
                "batch_size": 100, "train_judge_mode": "sample",
                "batches_trained": 5, "version_chain_contiguous": True,
                "weights_versions": [["v0", "v1"], ["v1", "v2"], ["v2", "v3"],
                                     ["v3", "v4"], ["v4", "v5"]],
                "final_weights_version": "v5", "eval_mode": "argmax",
                "eval_episodes": 100, "eval_weights_versions": ["v5"],
                "train_split_sha256": "p" * 64,
                "expected_warmstart_sha256": warm_sha,
                "expected_pilot_split_sha256": "p" * 64,
                "expected_remainder_split_sha256": "r" * 64,
            } for lam in (0.05, 0.2, 0.5)
        },
    }), encoding="utf-8")
    artifacts = tmp_path / "artifacts.json"
    artifacts.write_text(json.dumps({
        "arm_costs": str(costs), "warmstart_weights": str(warm), "pilot": str(pilot),
        # Deliberately absent: bootstrap is the one mode allowed to start
        # without a capacity report, because it is what produces one.
        "capacity_smoke": str(tmp_path / "absent.json"),
    }), encoding="utf-8")
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(yaml.safe_dump({
        "batch_size": 100, "episodes_per_run": 4000,      # the FORMAL size
        "decisions": {"D3_training_init_domain": "b_train"},
        "suites": {"libero_10": {"split": str(split)}},
        "lambda": {"lambda_1": 0.2},
        "runs": [{"id": "r0", "suite": "libero_10", "variant": "R_ts",
                  "lambda": "lambda_1", "seed": 0, "flagship": True}],
        "eval": {"primary_seed": 0},
    }), encoding="utf-8")
    del M4_SCHEMA

    def fake_run(command: str):
        if command.startswith("test -f "):
            return (0 if pathlib.Path(command.split()[-1]).exists() else 1), ""
        if "batch_package.py capacity" in command:
            return 0, json.dumps({"root": "x", "live_bytes": 1024})
        return 0, ""

    monkeypatch.setattr(LocalTransport, "run", lambda self, command: fake_run(command))
    # Guard only OUR shell invocations: the stdlib itself shells out (e.g.
    # platform.processor()), and blanketing that would fail for the wrong reason.
    real_run = subprocess.run

    def _no_shell(*a, **k):
        if k.get("shell"):
            raise AssertionError(f"unexpected shell command in an isolated test: {a[:1]}")
        return real_run(*a, **k)

    monkeypatch.setattr(subprocess, "run", _no_shell)
    monkeypatch.setattr("sys.argv", [
        "run_rl_router.py", "--matrix", str(matrix), "--run-id", "r0",
        "--arm-yaml", str(base_yaml), "--init-states-dir", str(pool),
        "--artifacts", str(artifacts), "--servers", "h:8000", "--workers", "1",
        "--out-dir", str(local), "--shard-root", str(shard_root),
        "--remote-root", str(remote_root), "--trainer-cmd", "true", "--smoke",
    ])
    loop.main()


def test_resume_recovers_metrics_when_the_export_already_landed(tmp_path) -> None:
    """The two tail windows are independent: "export never happened" leaves the
    weights missing, "export happened, metrics did not" leaves them present.
    Keying recovery off the weights alone loses that batch's row forever."""
    remote = _remote(tmp_path)
    _publish_state(remote, version="v1", consumed=[{"batch_id": "b0000"}])
    _weights(pathlib.Path(remote.weights("v1")), "v1")      # export DID land
    pathlib.Path(remote.metrics).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(remote.metrics).write_text("", encoding="utf-8")   # metrics did not

    state = resume_state(remote, tmp_path / "scratch")
    assert state["needs_export"] is False        # weights are fine...
    assert state["missing_metrics"] == ["b0000"]  # ...but the row is missing


def test_resume_reports_no_gap_when_the_tail_completed(tmp_path) -> None:
    remote = _remote(tmp_path)
    _publish_state(remote, version="v1", consumed=[{"batch_id": "b0000"}])
    _weights(pathlib.Path(remote.weights("v1")), "v1")
    pathlib.Path(remote.metrics).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(remote.metrics).write_text(
        json.dumps({"batch_id": "b0000", "loss": 0.5}) + "\n", encoding="utf-8")
    state = resume_state(remote, tmp_path / "scratch")
    assert state["needs_export"] is False and state["missing_metrics"] == []


@pytest.mark.parametrize("defect,expected", [
    ("eval_pool", "eval_pool"),
    ("broken_chain", "contiguous"),
    ("wrong_lambda", "actually trained"),
    ("eval_on_wrong_weights", "batch-5 policy"),
    ("eval_sampled", "argmax"),
    ("wrong_train_split", "different split"),
])
def test_pilot_gate_rejects_a_candidate_that_did_not_run_the_frozen_protocol(
    tmp_path, defect, expected,
) -> None:
    """A pilot that trained on the wrong split, skipped updates, or measured a
    policy other than the batch-5 one would poison λ₁/λ₂ and every formal run
    that cites them."""
    from exp.rl_router.launch_gates import _pilot_problems
    from exp.rl_router.pilot_lambda import PILOT_SCHEMA

    warm = tmp_path / "warm.pt"
    warm.write_bytes(b"w")
    warm_sha = hashlib.sha256(b"w").hexdigest()
    protocol = {"batches": 5, "batch_size": 100, "eval_episodes": 100,
                "mode": "argmax", "eval_pool": "b_train_remainder"}
    record = {
        "lambda": 0.2, "seed": 0, "batch_size": 100, "train_judge_mode": "sample",
        "batches_trained": 5, "version_chain_contiguous": True,
        "weights_versions": [["v0", "v1"], ["v1", "v2"], ["v2", "v3"],
                             ["v3", "v4"], ["v4", "v5"]],
        "final_weights_version": "v5", "eval_mode": "argmax", "eval_episodes": 100,
        "eval_weights_versions": ["v5"], "train_split_sha256": "p" * 64,
        "expected_warmstart_sha256": warm_sha,
        "expected_pilot_split_sha256": "p" * 64,
        "expected_remainder_split_sha256": "r" * 64,
    }
    if defect == "eval_pool":
        protocol = {**protocol, "eval_pool": "b_val"}
    elif defect == "broken_chain":
        # Keep the self-reported boolean true: the gate must recompute from the
        # marker files and notice that none of the five batches advanced.
        record = {**record, "weights_versions": [["v9", "v9"]] * 5}
    elif defect == "eval_on_wrong_weights":
        record = {**record, "eval_weights_versions": ["v2"]}
    elif defect == "eval_sampled":
        record = {**record, "eval_mode": "sample"}
    elif defect == "wrong_train_split":
        record = {**record, "train_split_sha256": "z" * 64}

    doc = {
        "schema": PILOT_SCHEMA, "separated": True, "seed": 0, "candidates_run": 3,
        "selected": {"lambda_1": 0.2, "lambda_2": 0.5}, "protocol": protocol,
        "pilot_split_sha256": "p" * 64, "remainder_split_sha256": "r" * 64,
        "runs": {
            str(lam): dict(record, **{"lambda": lam, "lambda_recorded": lam})
            for lam in (0.05, 0.2, 0.5)
        },
    }
    if defect == "wrong_lambda":
        # Models a command template that ignored {lam}: all three directories
        # claim different candidates, but each run manifest records lambda=0.2.
        for candidate in doc["runs"].values():
            candidate["lambda_recorded"] = 0.2
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    problems = _pilot_problems(str(path), {"lambda": {"lambda_1": 0.2}},
                               {"lambda": "lambda_1"}, warmstart_path=str(warm))
    assert any(expected in p for p in problems), problems
