"""Unit tests for ``MlpRouterJudge`` — the X14 online-RL router baseline.

The suite is organised around the contracts the experiment's validity rests on:

  1. **Masking** — the network is blind to everything on the library side. If
     this ever regresses, the whole "why not train a router" comparison becomes
     a comparison against a router that cheats.
  2. **Arm -> verdict mapping** — including the empty-library fallback, which
     decides how execution cost is billed.
  3. **Weights fail-fast** — a silently mismatched checkpoint produces plausible
     numbers, which is worse than a crash.
  4. **Per-episode RNG** — replay equivalence is what makes an interrupted run
     and its resume the same experiment.
  5. **Shard lifecycle** — the manifest, not the journal, is the authority on
     batch completeness, so every way an episode can end is pinned here.
  6. **Identity** — a stale attempt must never overwrite a live one's shard.
"""

from __future__ import annotations

import inspect
import json
import pathlib

import pytest
import torch

from openpi.cache.components.judge import HitType
from openpi.cache.components.mlp_router_judge import (
    FEATURE_DTYPE,
    MlpRouterJudge,
    RouterFeatureEncoder,
    RouterWeights,
    save_router_weights,
)
from openpi.cache.storage_types import SearchResultLite
from openpi.cache.types import CheckpointID, ROBOT_STATE, VISION_0

D = 32
H = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(entry_id: str = "e1", score: float = 0.9) -> SearchResultLite:
    return SearchResultLite(id=entry_id, score=score, checkpoint_id=CheckpointID.CP1)


def _weights(
    tmp_path: pathlib.Path,
    *,
    arms: str = "tsc",
    version: str = "v1",
    seed: int = 0,
    normalized: bool = True,
    name: str = "w.pt",
) -> str:
    torch.manual_seed(seed)
    n_arms = {"ts": 2, "tc": 2, "tsc": 3}[arms]
    path = tmp_path / name
    save_router_weights(
        path,
        W1=torch.randn(H, D) * 0.5,
        b1=torch.zeros(H),
        W2=torch.randn(n_arms, H) * 0.5,
        b2=torch.zeros(n_arms),
        arms=arms,
        fields=(ROBOT_STATE,),
        dims={ROBOT_STATE: D},
        weights_version=version,
        mu=torch.zeros(D) if normalized else None,
        sigma=torch.ones(D) if normalized else None,
    )
    return str(path)


def _judge(tmp_path: pathlib.Path, **kwargs) -> MlpRouterJudge:
    params = {
        "arms": "tsc",
        "feature_fields": [ROBOT_STATE],
        "hidden": H,
        "mode": "sample",
        "temperature": 1.0,
        "seed": 7,
    }
    params.update(kwargs)
    if "weights_path" not in params and "constant_arm" not in params:
        params["weights_path"] = _weights(tmp_path, arms=params["arms"])
    return MlpRouterJudge(**params)


def _identity(**overrides) -> dict:
    base = {"run_id": "run0", "batch_id": "b0", "task_uid": "y:eval:1:2", "attempt": 1}
    base.update(overrides)
    return base


def _verdicts(judge: MlpRouterJudge, features: list[torch.Tensor], results=None):
    out = []
    for f in features:
        out.append(judge(
            results=[_result()] if results is None else results,
            checkpoint_id=CheckpointID.CP1,
            cached_data={},
            query_keys={ROBOT_STATE: f},
        ))
    return out


def _features(n: int = 6, seed: int = 3) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(D, generator=g) for _ in range(n)]


def _manifest(dump_dir: pathlib.Path, run_id: str = "run0", batch_id: str = "b0") -> list[dict]:
    path = dump_dir / run_id / batch_id / "manifest.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# 1. Masking contract
# ---------------------------------------------------------------------------


def test_decide_signature_excludes_every_library_side_input() -> None:
    """The masking contract, enforced statically.

    A future edit that widens ``_decide`` to accept results / view / history /
    retrieval signals would let the router see what TIER's retrieval sees after
    it touches the library — exactly the confound this experiment exists to
    avoid. The signature is the cheapest place to lock it.
    """
    params = set(inspect.signature(MlpRouterJudge._decide).parameters) - {"self"}
    assert params == {"features"}


def test_decision_is_invariant_to_scores_results_and_cached_data(tmp_path) -> None:
    """Permuting, rescaling and negating the retrieval scores — and feeding
    junk ``cached_data`` — must not move a single sampled arm."""
    feats = _features()
    # Same weights, same features, fresh judge each time: the ONLY thing that
    # varies across the three runs is what the library side hands over.
    shared = _weights(tmp_path, name="shared.pt")

    def arms_for(results, cached):
        j = _judge(tmp_path, weights_path=shared)
        j.on_episode_start(extra_metadata=_identity())
        return [
            j(results=results, checkpoint_id=CheckpointID.CP1, cached_data=cached,
              query_keys={ROBOT_STATE: f}).router_outputs["arm_sampled"]
            for f in feats
        ]

    permuted = arms_for([_result("b", 0.01), _result("a", 0.99)], {})
    rescaled = arms_for([_result("a", -7.0), _result("b", 123.4)], {})
    noisy = arms_for([_result("a", 0.99), _result("b", 0.01)],
                     {"state": torch.randn(4), "prefix_embs": torch.randn(2, 3)})
    assert permuted == rescaled == noisy
    # Guard against the assertion passing vacuously (e.g. a degenerate all-one-arm
    # policy would make any two runs equal).
    assert len(set(permuted)) > 1


def test_judge_requires_query_keys(tmp_path) -> None:
    judge = _judge(tmp_path)
    with pytest.raises(ValueError, match="query_keys"):
        judge(results=[], checkpoint_id=CheckpointID.CP1, cached_data={})


# ---------------------------------------------------------------------------
# 2. Arm -> verdict mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arms", ["ts", "tc", "tsc"])
@pytest.mark.parametrize("has_results", [True, False])
def test_action_mapping_full_table(tmp_path, arms: str, has_results: bool) -> None:
    """Every (arm set x library state) combination maps to the frozen verdict."""
    for arm in {"ts": ("teacher", "student"), "tc": ("teacher", "cache"),
                "tsc": ("teacher", "student", "cache")}[arms]:
        judge = MlpRouterJudge(
            arms=arms, constant_arm=arm, feature_fields=[ROBOT_STATE], mode="argmax",
        )
        judge.on_episode_start(extra_metadata=_identity())
        res = judge(
            results=[_result()] if has_results else [],
            checkpoint_id=CheckpointID.CP1, cached_data={},
            query_keys={ROBOT_STATE: torch.zeros(D)},
        )
        ro = res.router_outputs
        assert ro["arm_sampled"] == arm
        if arm == "teacher":
            assert (res.hit_type, res.winner_id, res.hit_override) == (HitType.MISS, None, None)
            assert ro["fallback"] is False
        elif arm == "student":
            # Payloadless FULL_HIT: no winner, override True, zero fetch.
            assert (res.hit_type, res.winner_id, res.hit_override) == (HitType.FULL_HIT, None, True)
            assert ro["fallback"] is False
        elif has_results:
            assert (res.hit_type, res.winner_id, res.hit_override) == (HitType.FULL_HIT, "e1", False)
            assert ro["fallback"] is False
        else:
            # Empty library: nothing to replay, so the cache arm degrades to a
            # teacher MISS and says so — cost is billed to what actually ran.
            assert (res.hit_type, res.winner_id, res.hit_override) == (HitType.MISS, None, None)
            assert ro["fallback"] is True


def test_router_outputs_schema_is_frozen(tmp_path) -> None:
    """The wire schema is fixed; features and logits must never appear on it."""
    judge = _judge(tmp_path)
    judge.on_episode_start(extra_metadata=_identity())
    ro = _verdicts(judge, _features(1))[0].router_outputs
    assert set(ro) == {
        "decision_idx", "arm_sampled", "arm_executed", "probs",
        "temperature", "weights_version", "seed_ep", "fallback",
    }
    assert ro["arm_executed"] is None  # the interceptor stamps this


def test_decision_idx_is_a_dense_per_episode_counter(tmp_path) -> None:
    judge = _judge(tmp_path)
    judge.on_episode_start(extra_metadata=_identity())
    first = [v.router_outputs["decision_idx"] for v in _verdicts(judge, _features(5))]
    judge.on_episode_end()
    judge.on_episode_start(extra_metadata=_identity(task_uid="other"))
    second = [v.router_outputs["decision_idx"] for v in _verdicts(judge, _features(3))]
    assert first == [0, 1, 2, 3, 4]
    assert second == [0, 1, 2]


# ---------------------------------------------------------------------------
# 3. Weights / meta fail-fast
# ---------------------------------------------------------------------------


def test_weights_roundtrip_validates(tmp_path) -> None:
    path = _weights(tmp_path)
    w = RouterWeights.load(path)
    assert w.arms == "tsc" and w.weights_version == "v1" and w.hidden == H


@pytest.mark.parametrize("corruption", ["model_sha", "arms", "encoder_version", "dims", "shape"])
def test_weights_meta_mismatch_fails_fast(tmp_path, corruption: str) -> None:
    path = pathlib.Path(_weights(tmp_path))
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if corruption == "model_sha":
        blob["meta"]["model_sha"] = "0" * 64
    elif corruption == "arms":
        blob["meta"]["arms"] = "nope"
    elif corruption == "encoder_version":
        blob["meta"]["encoder_version"] = "0" * 64
    elif corruption == "dims":
        blob["meta"]["dims"] = {ROBOT_STATE: D + 1}
    else:
        blob["W1"] = torch.randn(H + 1, D)
    torch.save(blob, path)

    with pytest.raises(ValueError):
        MlpRouterJudge(
            arms="tsc", weights_path=str(path), feature_fields=[ROBOT_STATE],
            hidden=H, mode="argmax",
        )


def test_configured_arms_must_match_weights(tmp_path) -> None:
    path = _weights(tmp_path, arms="ts")
    with pytest.raises(ValueError, match="arms"):
        MlpRouterJudge(arms="tsc", weights_path=path, feature_fields=[ROBOT_STATE],
                       hidden=H, mode="argmax")


def test_feature_dims_are_checked_against_weights_meta(tmp_path) -> None:
    """A key builder change that alters the feature width must not be absorbed
    silently — the router would be reading a different observation space."""
    judge = _judge(tmp_path, mode="argmax")
    judge.on_episode_start(extra_metadata=_identity())
    with pytest.raises(ValueError, match="do not match weights meta"):
        judge(results=[], checkpoint_id=CheckpointID.CP1, cached_data={},
              query_keys={ROBOT_STATE: torch.zeros(D + 4)})


def test_weights_and_constant_arm_are_mutually_exclusive(tmp_path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        MlpRouterJudge(arms="ts", mode="argmax", feature_fields=[ROBOT_STATE])
    with pytest.raises(ValueError, match="exactly one"):
        MlpRouterJudge(arms="ts", mode="argmax", feature_fields=[ROBOT_STATE],
                       weights_path=_weights(tmp_path, arms="ts"), constant_arm="student")


# ---------------------------------------------------------------------------
# 4. Constant-arm collection mode
# ---------------------------------------------------------------------------


def test_constant_arm_needs_no_weights_and_emits_one_hot_probs() -> None:
    """The warm-start collection pass runs before any statistics exist; that is
    what breaks the mu/sigma circular dependency."""
    judge = MlpRouterJudge(arms="ts", constant_arm="student", mode="argmax",
                           feature_fields=[ROBOT_STATE])
    judge.on_episode_start(extra_metadata=_identity())
    res = judge(results=[], checkpoint_id=CheckpointID.CP1, cached_data={},
                query_keys={ROBOT_STATE: torch.randn(D)})
    assert res.router_outputs["probs"] == [0.0, 1.0]
    assert res.router_outputs["weights_version"] == "constant-student"
    assert judge.weights_version == "constant-student"


def test_constant_arm_rejects_sampling_and_foreign_arms() -> None:
    with pytest.raises(ValueError, match="argmax"):
        MlpRouterJudge(arms="ts", constant_arm="student", mode="sample",
                       feature_fields=[ROBOT_STATE])
    with pytest.raises(ValueError, match="not in arms"):
        MlpRouterJudge(arms="tc", constant_arm="student", mode="argmax",
                       feature_fields=[ROBOT_STATE])


def test_constant_arm_without_dump_does_no_encoding(tmp_path) -> None:
    """No weights and no dump => the router touches no feature tensor, so its
    own cost stays out of an evaluation it is not part of."""
    judge = MlpRouterJudge(arms="tc", constant_arm="cache", mode="argmax",
                           feature_fields=[ROBOT_STATE])
    judge.on_episode_start(extra_metadata=_identity())
    # A malformed query_keys dict would raise inside the encoder if it ran.
    res = judge(results=[_result()], checkpoint_id=CheckpointID.CP1, cached_data={},
                query_keys={})
    assert res.hit_type is HitType.FULL_HIT


# ---------------------------------------------------------------------------
# 5. Per-episode RNG / replay equivalence
# ---------------------------------------------------------------------------


def test_same_identity_replays_the_same_arm_sequence(tmp_path) -> None:
    """Uninterrupted vs resume equivalence: a fresh process, a fresh judge and a
    different global torch seed still reproduce the episode exactly."""
    shared, feats = _weights(tmp_path, name="rng.pt"), _features()

    def run(identity: dict, global_seed: int) -> list[str]:
        torch.manual_seed(global_seed)
        judge = _judge(tmp_path, weights_path=shared)
        judge.on_episode_start(extra_metadata=identity)
        return [v.router_outputs["arm_sampled"] for v in _verdicts(judge, feats)]

    base = run(_identity(), 11)
    assert base == run(_identity(), 999)                      # resume equivalence
    assert base != run(_identity(attempt=2), 11)              # requeue is a new stream
    assert base != run(_identity(task_uid="y:eval:1:3"), 11)  # per-episode stream


def test_run_seed_changes_the_stream(tmp_path) -> None:
    shared, feats = _weights(tmp_path, name="seeded.pt"), _features()

    def run(seed: int) -> list[str]:
        judge = _judge(tmp_path, weights_path=shared, seed=seed)
        judge.on_episode_start(extra_metadata=_identity())
        return [v.router_outputs["arm_sampled"] for v in _verdicts(judge, feats)]

    assert run(7) != run(8)


def test_missing_identity_forces_argmax_and_produces_no_training_data(tmp_path) -> None:
    dump = tmp_path / "dump"
    judge = _judge(tmp_path, dump_dir=str(dump))
    judge.on_episode_start(extra_metadata={"task_uid": "u"})  # run_id/batch_id absent
    feats = _features(4)
    first = [v.router_outputs["arm_sampled"] for v in _verdicts(judge, feats)]
    judge.on_episode_end()

    judge.on_episode_start(extra_metadata={"task_uid": "u"})
    assert [v.router_outputs["arm_sampled"] for v in _verdicts(judge, feats)] == first
    judge.on_episode_end()

    assert _manifest(dump) == []                       # never a training slot
    orphans = list((dump / "_orphan").glob("*.jsonl"))
    assert len(orphans) == 2                           # metadata kept for auditing
    assert not list(dump.rglob("*.bin"))               # but never the features


def test_weights_version_mismatch_isolates_the_episode(tmp_path) -> None:
    """A hot-swap race (task dispatched against another version) drops the
    episode from training instead of failing it — the batch can still repair."""
    dump = tmp_path / "dump"
    judge = _judge(tmp_path, dump_dir=str(dump))
    judge.on_episode_start(extra_metadata=_identity(weights_version="v-stale"))
    _verdicts(judge, _features(2))
    judge.on_episode_end()
    assert _manifest(dump) == []
    assert len(list((dump / "_orphan").glob("*.jsonl"))) == 1


# ---------------------------------------------------------------------------
# 6. Encoder / dump parity
# ---------------------------------------------------------------------------


def test_dumped_bytes_are_exactly_the_network_input(tmp_path) -> None:
    """On-policy parity by construction: the MLP decides on the fp32 upcast of
    the very bytes the trainer will read back."""
    dump = tmp_path / "dump"
    judge = _judge(tmp_path, dump_dir=str(dump))
    judge.on_episode_start(extra_metadata=_identity())
    feats = _features(5)
    _verdicts(judge, feats)
    judge.on_episode_end()

    entry = _manifest(dump)[0]
    raw = (dump / "run0" / "b0" / entry["shard"]).read_bytes()
    restored = torch.frombuffer(bytearray(raw), dtype=FEATURE_DTYPE).reshape(
        entry["rows"], entry["dim"]
    )
    encoder = RouterFeatureEncoder((ROBOT_STATE,), mu=torch.zeros(D), sigma=torch.ones(D))
    expected = torch.stack([encoder.encode({ROBOT_STATE: f}) for f in feats])
    assert torch.equal(restored, expected)
    assert entry["rows"] == 5 and entry["dim"] == D and entry["dtype"] == "float16"


def test_encoder_version_tracks_fields_order_and_normalization() -> None:
    v0 = RouterFeatureEncoder((VISION_0, ROBOT_STATE))
    # Canonical ordering: the version is a property of the field SET, not of the
    # order the yaml happened to list them in.
    assert RouterFeatureEncoder((ROBOT_STATE, VISION_0)).version == v0.version
    assert RouterFeatureEncoder((ROBOT_STATE,)).version != v0.version
    v1 = RouterFeatureEncoder((VISION_0, ROBOT_STATE),
                              mu=torch.zeros(4), sigma=torch.ones(4))
    assert v1.version != v0.version and v1.normalized and not v0.normalized
    shifted = RouterFeatureEncoder((VISION_0, ROBOT_STATE),
                                   mu=torch.ones(4), sigma=torch.ones(4))
    assert shifted.version != v1.version


def test_encoder_normalizes_only_robot_state() -> None:
    enc = RouterFeatureEncoder(
        (VISION_0, ROBOT_STATE), mu=torch.full((2,), 5.0), sigma=torch.full((2,), 2.0)
    )
    out = enc.encode({VISION_0: torch.tensor([1.0, 2.0]), ROBOT_STATE: torch.tensor([7.0, 9.0])})
    assert torch.equal(out, torch.tensor([1.0, 2.0, 1.0, 2.0], dtype=FEATURE_DTYPE))


def test_encoder_rejects_a_missing_field() -> None:
    enc = RouterFeatureEncoder((VISION_0, ROBOT_STATE))
    with pytest.raises(ValueError, match="missing from query_keys"):
        enc.encode({ROBOT_STATE: torch.zeros(4)})


# ---------------------------------------------------------------------------
# 7. Shard lifecycle
# ---------------------------------------------------------------------------


def test_normal_end_writes_shard_sidecar_and_manifest(tmp_path) -> None:
    dump = tmp_path / "dump"
    judge = _judge(tmp_path, dump_dir=str(dump))
    judge.on_episode_start(extra_metadata=_identity())
    _verdicts(judge, _features(3))
    judge.on_episode_end()

    (entry,) = _manifest(dump)
    assert entry["status"] == "complete" and entry["rows"] == 3
    directory = dump / "run0" / "b0"
    assert (directory / entry["shard"]).exists()
    sidecar = [json.loads(x) for x in (directory / entry["sidecar"]).read_text().splitlines()]
    assert [r["decision_idx"] for r in sidecar] == [0, 1, 2]
    assert all(r["task_uid"] == "y:eval:1:2" and r["attempt"] == 1 for r in sidecar)
    assert not list(directory.glob("*.tmp"))


def test_zero_step_episode_is_recorded_as_empty(tmp_path) -> None:
    """A dispatched episode that never reached a verdict still terminates its
    slot — the packager needs to tell "empty" from "crashed before writing"."""
    dump = tmp_path / "dump"
    judge = _judge(tmp_path, dump_dir=str(dump))
    judge.on_episode_start(extra_metadata=_identity())
    judge.on_episode_end()
    (entry,) = _manifest(dump)
    assert entry["status"] == "empty" and entry["rows"] == 0 and entry["shard"] is None


def test_episode_end_is_idempotent(tmp_path) -> None:
    """``on_task_end`` follows ``on_episode_end`` on a clean connection close;
    the second call must not append a second manifest entry."""
    dump = tmp_path / "dump"
    judge = _judge(tmp_path, dump_dir=str(dump))
    judge.on_episode_start(extra_metadata=_identity())
    _verdicts(judge, _features(2))
    judge.on_episode_end()
    judge.on_episode_end()
    judge.on_task_end()
    assert len(_manifest(dump)) == 1


def test_missing_episode_end_is_closed_as_partial_by_the_next_start(tmp_path) -> None:
    """The client's ``episode_end`` RPC is best-effort (its exception is
    suppressed in episode_runner), so a lost end must not bleed into the next
    episode's shard."""
    dump = tmp_path / "dump"
    judge = _judge(tmp_path, dump_dir=str(dump))
    judge.on_episode_start(extra_metadata=_identity())
    _verdicts(judge, _features(4))
    judge.on_episode_start(extra_metadata=_identity(task_uid="y:eval:1:3"))  # no end
    _verdicts(judge, _features(2))
    judge.on_episode_end()

    entries = {e["task_uid"]: e for e in _manifest(dump)}
    assert entries["y:eval:1:2"]["status"] == "partial" and entries["y:eval:1:2"]["rows"] == 4
    assert entries["y:eval:1:3"]["status"] == "complete" and entries["y:eval:1:3"]["rows"] == 2


def test_disconnect_mid_episode_is_closed_as_partial(tmp_path) -> None:
    dump = tmp_path / "dump"
    judge = _judge(tmp_path, dump_dir=str(dump))
    judge.on_episode_start(extra_metadata=_identity())
    _verdicts(judge, _features(3))
    judge.on_task_end()
    (entry,) = _manifest(dump)
    assert entry["status"] == "partial" and entry["rows"] == 3


def test_write_never_decline_path_still_finalizes(tmp_path) -> None:
    """The orchestrator's ``on_episode_end`` has three early returns; routed
    router configs take the write-policy decline one. The finalize broadcast
    lives in ``finally`` precisely so this path still terminates the shard."""
    from openpi.cache.components.write_policy import NeverWritePolicy

    from tests.cache.conftest import make_orchestrator

    dump = tmp_path / "dump"
    judge = _judge(tmp_path, dump_dir=str(dump))
    orch, _, _ = make_orchestrator(judge=judge, write_policy=NeverWritePolicy())
    orch.on_episode_start("task", "ep", extra_metadata=_identity())
    orch.check(CheckpointID.CP1, stage1=type("S", (), {"state": torch.randn(1, D)})())
    orch.on_episode_end()
    (entry,) = _manifest(dump)
    assert entry["status"] == "complete" and entry["rows"] == 1


def test_crash_in_the_rename_window_leaves_no_manifest_entry(tmp_path, monkeypatch) -> None:
    """Finalize must not raise into the episode-end path, and a torn write must
    not be claimed in the ledger — the missing slot is what triggers repair."""
    import openpi.cache.components.mlp_router_judge as mod

    dump = tmp_path / "dump"
    judge = _judge(tmp_path, dump_dir=str(dump))
    judge.on_episode_start(extra_metadata=_identity())
    _verdicts(judge, _features(2))

    real_replace = mod.os.replace

    def boom(src, dst):
        if str(dst).endswith(".bin"):
            raise OSError("simulated crash in the rename window")
        return real_replace(src, dst)

    monkeypatch.setattr(mod.os, "replace", boom)
    judge.on_episode_end()  # must not raise
    assert _manifest(dump) == []


def test_crash_in_the_manifest_commit_window_leaves_no_entry(tmp_path, monkeypatch) -> None:
    dump = tmp_path / "dump"
    judge = _judge(tmp_path, dump_dir=str(dump))
    judge.on_episode_start(extra_metadata=_identity())
    _verdicts(judge, _features(2))
    monkeypatch.setattr(
        MlpRouterJudge, "_append_manifest",
        lambda *a, **k: (_ for _ in ()).throw(OSError("commit crash")),
    )
    judge.on_episode_end()  # must not raise
    assert _manifest(dump) == []


def test_startup_sweep_quarantines_torn_writes(tmp_path) -> None:
    """A ``.tmp`` can only outlive its writer if the process died inside the
    atomic write, so it is by definition torn and must never be read back."""
    dump = tmp_path / "dump"
    stale = dump / "run0" / "b0" / "prev.bin.tmp"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"\x00\x01")
    _judge(tmp_path, dump_dir=str(dump))
    assert not stale.exists()
    assert len(list((dump / "_quarantine").glob("*"))) == 1


def test_dump_dir_unset_means_zero_io(tmp_path) -> None:
    judge = _judge(tmp_path, mode="argmax")
    judge.on_episode_start(extra_metadata=_identity())
    _verdicts(judge, _features(3))
    judge.on_episode_end()
    assert list(tmp_path.glob("**/*.bin")) == []
    assert list(tmp_path.glob("**/manifest.jsonl")) == []


# ---------------------------------------------------------------------------
# 8. Identity: stale attempts and run isolation
# ---------------------------------------------------------------------------


def test_two_generations_of_one_uid_write_different_shards(tmp_path) -> None:
    """A late writer from a superseded dispatch must not be able to land on the
    live attempt's path — that is what keeps a finalized shard trustworthy."""
    dump = tmp_path / "dump"
    judge = _judge(tmp_path, dump_dir=str(dump))
    for attempt in (1, 2):
        judge.on_episode_start(extra_metadata=_identity(attempt=attempt))
        _verdicts(judge, _features(2))
        judge.on_episode_end()
    entries = _manifest(dump)
    assert {e["attempt"] for e in entries} == {1, 2}
    assert len({e["shard"] for e in entries}) == 2


def test_stale_attempt_then_current_attempt_both_land(tmp_path) -> None:
    dump = tmp_path / "dump"
    stale = _judge(tmp_path, dump_dir=str(dump))
    stale.on_episode_start(extra_metadata=_identity(attempt=1))
    _verdicts(stale, _features(2))

    live = _judge(tmp_path, dump_dir=str(dump))
    live.on_episode_start(extra_metadata=_identity(attempt=2))
    _verdicts(live, _features(5))
    live.on_episode_end()
    stale.on_task_end()  # the superseded worker reports late

    by_attempt = {e["attempt"]: e for e in _manifest(dump)}
    assert by_attempt[2]["status"] == "complete" and by_attempt[2]["rows"] == 5
    assert by_attempt[1]["status"] == "partial"


def test_runs_and_batches_do_not_share_a_directory(tmp_path) -> None:
    dump = tmp_path / "dump"
    judge = _judge(tmp_path, dump_dir=str(dump))
    for run_id, batch_id in (("runA", "b0"), ("runB", "b0"), ("runA", "b1")):
        judge.on_episode_start(extra_metadata=_identity(run_id=run_id, batch_id=batch_id))
        _verdicts(judge, _features(1))
        judge.on_episode_end()
    assert len(_manifest(dump, "runA", "b0")) == 1
    assert len(_manifest(dump, "runB", "b0")) == 1
    assert len(_manifest(dump, "runA", "b1")) == 1


def test_repair_uid_is_a_distinct_slot(tmp_path) -> None:
    """Repair rounds re-run an init under a new ``<orig>#r<n>`` uid, so the two
    attempts coexist instead of racing for one path."""
    dump = tmp_path / "dump"
    judge = _judge(tmp_path, dump_dir=str(dump))
    for uid in ("y:eval:1:2", "y:eval:1:2#r1"):
        judge.on_episode_start(extra_metadata=_identity(task_uid=uid))
        _verdicts(judge, _features(2))
        judge.on_episode_end()
    entries = _manifest(dump)
    assert {e["task_uid"] for e in entries} == {"y:eval:1:2", "y:eval:1:2#r1"}
    assert len({e["shard"] for e in entries}) == 2
