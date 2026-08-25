"""X15 U3 — shadow-teacher recording: RNG isolation, schema, and fail-open.

The recorder runs an extra teacher forward inside a live episode, so the tests
that matter are the ones about what it must NOT change: the executed action,
the global RNG stream, and the episode's ability to finish. The row schema is
tested alongside because the label joiner depends on error and terminal rows
being expressible at all.

Key dependency: ``openpi.cache.shadow_teacher``.
"""

from __future__ import annotations

import json

import pytest
import torch

from openpi.cache.shadow_teacher import (
    ShadowTeacherRecorder,
    chunk_deviation,
    stable_seed,
)

# ------------------------------------------------------------------
# Seeds
# ------------------------------------------------------------------


def test_seed_is_stable_across_processes() -> None:
    """A content digest, not ``hash()`` — otherwise PYTHONHASHSEED makes a run
    unreplayable."""
    assert stable_seed("u1", 1, 7) == stable_seed("u1", 1, 7)
    assert stable_seed("u1", 1, 7) != stable_seed("u1", 1, 8)
    assert stable_seed("u1", 1, 7) != stable_seed("u1", 2, 7)
    assert stable_seed("u1", 1, 7) != stable_seed("u2", 1, 7)
    assert 0 <= stable_seed("u1", 1, 7) < 2 ** 64


def test_seed_is_a_digest_not_pythons_hash() -> None:
    assert stable_seed("u1", 1, 0) != abs(hash("u1|1|0"))


# ------------------------------------------------------------------
# RNG isolation — the reason the generator seam exists
# ------------------------------------------------------------------


def test_shadow_noise_does_not_advance_the_global_rng(tmp_path) -> None:
    """The whole point: an extra forward must not shift the random sequence the
    main trajectory's later teacher steps will draw from."""
    rec = ShadowTeacherRecorder(path=str(tmp_path / "s.jsonl"))
    rec.begin_episode("u1", 1)

    torch.manual_seed(0)
    before = torch.random.get_rng_state().clone()
    rec.record(
        decision_idx=0,
        cache_chunk=torch.zeros(4, 3),
        teacher_fn=lambda noise: torch.zeros(4, 3),
        noise_shape=(4, 3),
    )
    after = torch.random.get_rng_state().clone()

    assert torch.equal(before, after)
    rec.close()


def test_shadow_noise_is_reproducible_for_the_same_decision(tmp_path) -> None:
    seen: list[torch.Tensor] = []

    def teacher(noise):
        seen.append(noise.clone())
        return torch.zeros(4, 3)

    for _ in range(2):
        rec = ShadowTeacherRecorder(path=str(tmp_path / "s.jsonl"))
        rec.begin_episode("u1", 1)
        rec.record(
            decision_idx=5, cache_chunk=torch.zeros(4, 3),
            teacher_fn=teacher, noise_shape=(4, 3),
        )
        rec.close()

    assert torch.equal(seen[0], seen[1])


# ------------------------------------------------------------------
# Labels
# ------------------------------------------------------------------


def test_identical_chunks_have_zero_deviation() -> None:
    chunk = torch.randn(10, 4)
    assert chunk_deviation(chunk, chunk.clone()) == pytest.approx(0.0, abs=1e-6)


def test_a_batched_teacher_chunk_does_not_inflate_the_label() -> None:
    """``Stage3Output.action_chunk`` keeps a unit batch dim while the cache
    payload's does not.

    Subtracting them directly broadcasts ``[1, H, D]`` against ``[H, D]`` into
    ``[H, H, D]`` — every step compared against every other step — so two
    IDENTICAL chunks scored ~8.0 instead of 0.0, and every recorded label
    carried that inflation.
    """
    chunk = torch.randn(50, 32)
    assert chunk_deviation(chunk, chunk.clone()[None, ...]) == pytest.approx(0.0, abs=1e-6)
    assert chunk_deviation(chunk[None, ...], chunk.clone()) == pytest.approx(0.0, abs=1e-6)


def test_a_real_batch_is_refused_rather_than_silently_reduced() -> None:
    """A unit batch is the known serving shape; anything wider means the caller
    handed over something other than one decision's chunk."""
    with pytest.raises(ValueError, match="batch size"):
        chunk_deviation(torch.zeros(4, 2), torch.zeros(3, 4, 2))


def test_a_wrongly_ranked_chunk_is_refused() -> None:
    with pytest.raises(ValueError, match=r"\[H, D\]"):
        chunk_deviation(torch.zeros(4, 2), torch.zeros(8))


def test_deviation_is_normalised_per_dimension() -> None:
    """Without sigma the widest-range joint would dominate the label."""
    a = torch.zeros(2, 2)
    b = torch.tensor([[10.0, 1.0], [10.0, 1.0]])
    raw = chunk_deviation(a, b)
    scaled = chunk_deviation(a, b, sigma=torch.tensor([10.0, 1.0]))
    assert scaled < raw


def test_deviation_compares_the_overlapping_horizon() -> None:
    a = torch.zeros(4, 2)
    b = torch.zeros(10, 2)
    assert chunk_deviation(a, b) == pytest.approx(0.0)


def test_empty_chunk_is_refused() -> None:
    with pytest.raises(ValueError, match="empty action chunk"):
        chunk_deviation(torch.zeros(0, 2), torch.zeros(0, 2))


# ------------------------------------------------------------------
# Schema and fail-open
# ------------------------------------------------------------------


def _rows(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_ok_rows_carry_the_join_key_and_label(tmp_path) -> None:
    path = tmp_path / "s.jsonl"
    rec = ShadowTeacherRecorder(path=str(path))
    rec.begin_episode("u1", 2)
    rec.record(
        decision_idx=3, cache_chunk=torch.zeros(4, 3),
        teacher_fn=lambda noise: torch.ones(4, 3), noise_shape=(4, 3),
    )
    rec.finalize_episode()
    rec.close()

    rows = _rows(path)
    ok = [r for r in rows if r["status"] == "ok"][0]
    assert (ok["task_uid"], ok["attempt"], ok["decision_idx"]) == ("u1", 2, 3)
    assert ok["u"] > 0
    # The frozen union schema keeps the chunk so a different deviation metric
    # can be applied later without re-running the rollout.
    assert len(ok["teacher_chunk"]) == 4 and len(ok["teacher_chunk"][0]) == 3


def test_a_failing_shadow_records_an_error_and_does_not_raise(tmp_path) -> None:
    """Fail-open: a broken shadow costs a label, never the episode."""
    path = tmp_path / "s.jsonl"
    rec = ShadowTeacherRecorder(path=str(path))
    rec.begin_episode("u1", 1)

    def boom(noise):
        raise RuntimeError("teacher exploded")

    result = rec.record(
        decision_idx=0, cache_chunk=torch.zeros(4, 3),
        teacher_fn=boom, noise_shape=(4, 3),
    )
    rec.close()

    assert result is None
    assert rec.error_count == 1
    row = _rows(path)[0]
    assert row["status"] == "error"
    assert row["error_type"] == "RuntimeError"


def test_finalize_marks_the_terminal_row(tmp_path) -> None:
    """An aborted episode must be distinguishable from one still buffering."""
    path = tmp_path / "s.jsonl"
    rec = ShadowTeacherRecorder(path=str(path))
    rec.begin_episode("u1", 1)
    rec.finalize_episode(terminal=False)
    rec.close()

    row = _rows(path)[0]
    assert row["status"] == "finalize"
    assert row["terminal"] is False
    assert row["decision_idx"] == -1


def test_disabled_recorder_writes_nothing(tmp_path) -> None:
    """The production default: wired but off must be a true no-op."""
    path = tmp_path / "s.jsonl"
    rec = ShadowTeacherRecorder(path=str(path), enabled=False)
    rec.begin_episode("u1", 1)
    assert rec.record(
        decision_idx=0, cache_chunk=torch.zeros(4, 3),
        teacher_fn=lambda noise: torch.zeros(4, 3), noise_shape=(4, 3),
    ) is None
    rec.finalize_episode()
    rec.close()

    assert not path.exists()
    assert rec.row_count == 0


def test_retries_are_separable_by_attempt(tmp_path) -> None:
    """The driver replays failed slots; labels from attempt 1 and 2 must not
    collide on the join key."""
    path = tmp_path / "s.jsonl"
    rec = ShadowTeacherRecorder(path=str(path))
    for attempt in (1, 2):
        rec.begin_episode("u1", attempt)
        rec.record(
            decision_idx=0, cache_chunk=torch.zeros(4, 3),
            teacher_fn=lambda noise: torch.ones(4, 3), noise_shape=(4, 3),
        )
    rec.close()

    attempts = {r["attempt"] for r in _rows(path) if r["status"] == "ok"}
    assert attempts == {1, 2}
