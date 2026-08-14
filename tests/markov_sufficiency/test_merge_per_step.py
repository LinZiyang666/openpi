"""Tests for the per-step shard merge.

The bug this exists to prevent is silent: both init pools index their inits from
zero, so a plain concatenation makes ``episode_table`` fuse 450 pairs of
unrelated episodes -- the sample collapses from 950 to 500, ``n_cycle`` becomes
the max of two episodes and their deviation counts are summed. Nothing raises;
the numbers just come out wrong.
"""

from __future__ import annotations

import json

import pytest

from exp.markov_sufficiency import _timeaxis, merge_per_step as mps
from exp.markov_sufficiency.journal_to_arms import POOL_OFFSET


def _row(arm="A0", task=0, subset=0, step=0, attempt=1, **extra):
    return {
        "yaml_id": arm, "task_id": task, "subset_init_state_idx": subset,
        "orig_init_state_idx": subset + 1, "episode_id": 0, "attempt": attempt,
        "task_uid": f"{arm}:eval:{task}:{subset}", "phase": "eval",
        "step_idx": step, "hit_type": "FULL_HIT", "winner_id": "ep:0",
        "cp1_score": 0.9, "searched": True, "success": True, **extra,
    }


def _shard(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(p)


def test_pools_stay_distinct_after_merge(tmp_path):
    a = _shard(tmp_path, "off.jsonl", [_row(subset=0), _row(subset=1)])
    b = _shard(tmp_path, "db.jsonl", [_row(subset=0), _row(subset=1)])
    rows, stats = mps.merge([("official", a), ("db_init", b)])
    assert stats["distinct_episode_keys"] == 4, "the two pools were fused"
    idx = sorted(r["subset_init_state_idx"] for r in rows)
    assert idx == [0, 1, POOL_OFFSET["db_init"], POOL_OFFSET["db_init"] + 1]


def test_provenance_index_is_left_alone(tmp_path):
    b = _shard(tmp_path, "db.jsonl", [_row(subset=7)])
    rows, _ = mps.merge([("db_init", b)])
    assert rows[0]["orig_init_state_idx"] == 8, "orig index is the link back to the init files"
    assert rows[0]["subset_init_state_idx"] == 7 + POOL_OFFSET["db_init"]
    assert rows[0]["_pool"] == "db_init"


def test_official_pool_is_unshifted(tmp_path):
    a = _shard(tmp_path, "off.jsonl", [_row(subset=3)])
    rows, _ = mps.merge([("official", a)])
    assert rows[0]["subset_init_state_idx"] == 3


def test_downstream_grouping_sees_two_episodes_not_one(tmp_path):
    """The real consequence: the time-axis gate must not fuse the two pools."""
    a = _shard(tmp_path, "off.jsonl", [_row(subset=0, step=s) for s in (0, 5, 10)])
    b = _shard(tmp_path, "db.jsonl", [_row(subset=0, step=s) for s in (0, 5, 10)])
    rows, _ = mps.merge([("official", a), ("db_init", b)])
    kept, report = _timeaxis.to_cycles(rows, 5)
    keys = {_timeaxis._episode_key(r) for r in kept}
    assert len(keys) == 2, "both pools collapsed into one episode key"
    assert report.quarantined_yamls == set()


def test_sidecar_rows_pass_through_untouched(tmp_path):
    a = _shard(tmp_path, "off.jsonl", [_row(), {"_kind": "client_timing", "yaml_id": "A0", "infer_ms": 1.0}])
    rows, stats = mps.merge([("official", a)])
    assert stats["sidecar"] == {"official": 1}
    sidecar = [r for r in rows if r.get("_kind")]
    assert sidecar[0]["_pool"] == "official"
    assert "subset_init_state_idx" not in sidecar[0]


def test_multiple_shards_of_one_pool_are_concatenated(tmp_path):
    a1 = _shard(tmp_path, "a1.jsonl", [_row(subset=0)])
    a2 = _shard(tmp_path, "a2.jsonl", [_row(subset=1)])
    rows, stats = mps.merge([("official", a1), ("official", a2)])
    assert stats["per_pool"]["official"] == 2
    assert len(stats["shards"]) == 2


def test_residual_collision_raises_rather_than_fusing(tmp_path, monkeypatch):
    """If the offset ever stopped separating the pools, that must be loud."""
    monkeypatch.setitem(mps.POOL_OFFSET, "db_init", 0)
    a = _shard(tmp_path, "off.jsonl", [_row(subset=0)])
    b = _shard(tmp_path, "db.jsonl", [_row(subset=0)])
    with pytest.raises(ValueError, match="still collide"):
        mps.merge([("official", a), ("db_init", b)])


def test_unknown_pool_is_rejected(tmp_path):
    a = _shard(tmp_path, "off.jsonl", [_row()])
    with pytest.raises(ValueError, match="unknown pool"):
        mps.merge([("nope", a)])


def test_cli_writes_rows_and_a_manifest(tmp_path):
    a = _shard(tmp_path, "off.jsonl", [_row(subset=0)])
    b = _shard(tmp_path, "db.jsonl", [_row(subset=0)])
    out = tmp_path / "merged.jsonl"
    mps.main(["--shard", f"official={a}", "--shard", f"db_init={b}", "--out", str(out)])
    lines = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    assert len(lines) == 2
    manifest = json.loads((tmp_path / "merged.manifest.json").read_text())
    assert manifest["distinct_episode_keys"] == 2
    assert manifest["total_rows"] == 2


def test_cli_rejects_a_malformed_shard(tmp_path):
    with pytest.raises(SystemExit):
        mps.main(["--shard", "no-equals", "--out", str(tmp_path / "o.jsonl")])


def test_arm_filter_keeps_only_the_named_arm(tmp_path):
    """E2-primary's estimand is A0 alone; other arms are other configurations."""
    a = _shard(tmp_path, "off.jsonl", [_row(arm="A0", subset=0), _row(arm="A1", subset=0),
                                       _row(arm="A2", subset=1)])
    rows, stats = mps.merge([("official", a)], arm="A0")
    assert {r["yaml_id"] for r in rows} == {"A0"}
    assert stats["dropped_other_arms"] == 2
    assert stats["arm_filter"] == "A0"


def test_arm_filter_also_drops_other_arms_sidecar_rows(tmp_path):
    a = _shard(tmp_path, "off.jsonl", [
        _row(arm="A0"),
        {"_kind": "client_timing", "yaml_id": "A1", "infer_ms": 1.0},
        {"_kind": "client_timing", "yaml_id": "A0", "infer_ms": 2.0},
    ])
    rows, stats = mps.merge([("official", a)], arm="A0")
    assert stats["sidecar"] == {"official": 1}
    assert all(r["yaml_id"] == "A0" for r in rows)


def test_no_arm_filter_keeps_everything(tmp_path):
    a = _shard(tmp_path, "off.jsonl", [_row(arm="A0"), _row(arm="A1")])
    rows, stats = mps.merge([("official", a)])
    assert len(rows) == 2 and stats["dropped_other_arms"] == 0
