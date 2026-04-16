"""Tests for exp/trajectory_deviation/merge_step3_cfgs.py."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from exp.trajectory_deviation.merge_step3_cfgs import (
    _load_jsonl_records,
    aggregate,
    main,
    write_csv,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _rec(
    *, cfg="clip_w7_d4", ep="task_0/episode_0", tau=5, n=2,
    success=True, inference_ratio=0.5, total_cycles=10,
    num_inference_cycles=5, T_gt_cycles=10, burst_count=3,
) -> dict:
    return {
        "cfg": cfg, "ep": ep, "tau": int(tau), "n": int(n),
        "success": bool(success), "inference_ratio": float(inference_ratio),
        "total_cycles": int(total_cycles),
        "num_inference_cycles": int(num_inference_cycles),
        "T_gt_cycles": int(T_gt_cycles), "burst_count": int(burst_count),
    }


# ---------------------------------------------------------------------------
# _load_jsonl_records: dedup + malformed detection
# ---------------------------------------------------------------------------


class TestLoadJsonlRecords:
    def test_reads_multiple_files_in_order(self, tmp_path):
        p1 = tmp_path / "a.jsonl"
        p2 = tmp_path / "b.jsonl"
        _write_jsonl(p1, [_rec(ep="task_0/episode_0")])
        _write_jsonl(p2, [_rec(ep="task_0/episode_1")])

        out = _load_jsonl_records([p1, p2])
        eps = sorted(r["ep"] for r in out)
        assert eps == ["task_0/episode_0", "task_0/episode_1"]

    def test_last_write_wins_on_duplicate_key(self, tmp_path):
        """BaseRunState retry re-appends on failure; later records must win."""
        path = tmp_path / "r.jsonl"
        _write_jsonl(path, [
            _rec(success=False, inference_ratio=0.9),
            _rec(success=True, inference_ratio=0.3),
        ])
        out = _load_jsonl_records([path])
        assert len(out) == 1
        assert out[0]["success"] is True
        assert out[0]["inference_ratio"] == 0.3

    def test_dedupe_across_files(self, tmp_path):
        """Same (cfg, ep, tau, n) across files collapses to one record."""
        p1 = tmp_path / "a.jsonl"
        p2 = tmp_path / "b.jsonl"
        _write_jsonl(p1, [_rec(success=False)])
        _write_jsonl(p2, [_rec(success=True)])
        out = _load_jsonl_records([p1, p2])
        assert len(out) == 1
        # Second file is read after the first, so it wins.
        assert out[0]["success"] is True

    def test_blank_lines_ignored(self, tmp_path):
        path = tmp_path / "blanks.jsonl"
        path.write_text("\n" + json.dumps(_rec()) + "\n\n")
        out = _load_jsonl_records([path])
        assert len(out) == 1

    def test_malformed_json_raises_with_line_number(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text("{not json}\n")
        with pytest.raises(ValueError, match=r"bad\.jsonl:1"):
            _load_jsonl_records([path])

    def test_missing_key_raises_with_line_number(self, tmp_path):
        path = tmp_path / "missing.jsonl"
        rec = _rec()
        del rec["tau"]
        _write_jsonl(path, [rec])
        with pytest.raises(ValueError, match="missing required key"):
            _load_jsonl_records([path])

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_jsonl_records([tmp_path / "does_not_exist.jsonl"])


# ---------------------------------------------------------------------------
# aggregate: grouping + statistics
# ---------------------------------------------------------------------------


class TestAggregate:
    def test_single_group_mean_and_std(self):
        recs = [
            _rec(ep="task_0/episode_0", inference_ratio=0.4, success=True),
            _rec(ep="task_0/episode_1", inference_ratio=0.6, success=False),
        ]
        rows = aggregate(recs)
        assert len(rows) == 1
        row = rows[0]
        assert row["cfg"] == "clip_w7_d4"
        assert row["tau"] == 5
        assert row["n"] == 2
        assert row["episodes"] == 2
        assert row["success_rate"] == 0.5
        assert row["mean_inference_ratio"] == pytest.approx(0.5)
        # Population std (ddof=0) of {0.4, 0.6}.
        assert row["std_inference_ratio"] == pytest.approx(0.1)

    def test_groups_split_by_tau_and_n(self):
        recs = [
            _rec(tau=5, n=2, inference_ratio=0.5, success=True),
            _rec(tau=5, n=3, inference_ratio=0.7, success=False),
            _rec(tau=7, n=2, inference_ratio=0.3, success=True),
        ]
        rows = aggregate(recs)
        assert len(rows) == 3
        keys = sorted((r["tau"], r["n"]) for r in rows)
        assert keys == [(5, 2), (5, 3), (7, 2)]

    def test_groups_split_by_cfg(self):
        recs = [
            _rec(cfg="clip_w7_d4", ep="task_0/episode_0"),
            _rec(cfg="spatial16_w8_d4", ep="task_0/episode_0"),
        ]
        rows = aggregate(recs)
        assert len(rows) == 2
        assert sorted(r["cfg"] for r in rows) == ["clip_w7_d4", "spatial16_w8_d4"]

    def test_success_rate_all_success(self):
        recs = [
            _rec(ep="task_0/episode_0", success=True),
            _rec(ep="task_0/episode_1", success=True),
        ]
        rows = aggregate(recs)
        assert rows[0]["success_rate"] == 1.0

    def test_success_rate_all_fail(self):
        recs = [
            _rec(ep="task_0/episode_0", success=False),
            _rec(ep="task_0/episode_1", success=False),
        ]
        rows = aggregate(recs)
        assert rows[0]["success_rate"] == 0.0

    def test_output_rows_sorted(self):
        recs = [
            _rec(cfg="b", tau=5, n=1),
            _rec(cfg="a", tau=7, n=2),
            _rec(cfg="a", tau=5, n=1),
        ]
        rows = aggregate(recs)
        keys = [(r["cfg"], r["tau"], r["n"]) for r in rows]
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# write_csv
# ---------------------------------------------------------------------------


def test_write_csv_roundtrip(tmp_path):
    rows = [
        {
            "cfg": "clip_w7_d4", "tau": 5, "n": 2, "episodes": 3,
            "success_rate": 0.3333333,
            "mean_inference_ratio": 0.4,
            "std_inference_ratio": 0.1,
        },
    ]
    out = tmp_path / "summary.csv"
    write_csv(rows, out)
    with out.open() as fh:
        reader = list(csv.DictReader(fh))
    assert len(reader) == 1
    r = reader[0]
    assert r["cfg"] == "clip_w7_d4"
    assert r["tau"] == "5"
    assert r["n"] == "2"
    assert r["episodes"] == "3"


def test_write_csv_creates_parent_dirs(tmp_path):
    rows = [{
        "cfg": "c", "tau": 1, "n": 1, "episodes": 1,
        "success_rate": 1.0, "mean_inference_ratio": 0.0, "std_inference_ratio": 0.0,
    }]
    out = tmp_path / "nested" / "dir" / "summary.csv"
    write_csv(rows, out)
    assert out.exists()


# ---------------------------------------------------------------------------
# main CLI smoke test
# ---------------------------------------------------------------------------


def test_main_end_to_end(tmp_path):
    p1 = tmp_path / "a" / "results.jsonl"
    p2 = tmp_path / "b" / "results.jsonl"
    _write_jsonl(p1, [
        _rec(cfg="clip_w7_d4", ep="task_0/episode_0", tau=5, n=2, success=True),
        _rec(cfg="clip_w7_d4", ep="task_0/episode_1", tau=5, n=2, success=False),
    ])
    _write_jsonl(p2, [
        _rec(cfg="spatial16_w8_d4", ep="task_0/episode_0", tau=5, n=2, success=True),
    ])
    out = tmp_path / "summary.csv"

    main([
        "--jsonl", str(p1),
        "--jsonl", str(p2),
        "--out", str(out),
    ])

    assert out.exists()
    with out.open() as fh:
        rows = list(csv.DictReader(fh))
    # Two (cfg, tau, n) groups.
    assert len(rows) == 2
    cfgs = sorted(r["cfg"] for r in rows)
    assert cfgs == ["clip_w7_d4", "spatial16_w8_d4"]


def test_main_empty_inputs_exits(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    out = tmp_path / "summary.csv"
    with pytest.raises(SystemExit):
        main(["--jsonl", str(empty), "--out", str(out)])
