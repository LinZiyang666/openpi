"""Snapshot tests for the weighted-sum search emitter + cell summarizer.

Pin the round-1 matrix invariants (owner-approved plan robocasa365_ws_search):
76 cells = 4 iso + 42 grid2 + 30 grid3(rs-dominant) per teacher, weights on a
unit simplex, the forced-enabled keybuilder fields, the cp3 in-memory pin, and
the journal summarizer's success/error split.
"""

from __future__ import annotations

import json

import pytest
import yaml

from exp.robocasa365.emit_ws_search_yamls import FIELDS, TEACHERS, emit_teacher, weight_matrix
from exp.robocasa365.run_ws_search import WsSearchStrategy, summarize_journal


# Minimal calibration doc: zscore for every field, both teacher stems.
def _calib() -> dict:
    fields = {
        f: {
            "sim_type": "l2" if f == "robot_state" else "cosine",
            "selected": {"method": "zscore", "params": {"mu": 0.5, "sigma": 0.1, "squash": "tanh"}},
        }
        for f in FIELDS
    }
    dims = {"vision_0": 8, "vision_1": 8, "vision_2": 8, "prompt_emb": 4, "robot_state": 2}
    return {
        spec["stem"]: {"builder_type": f"cp1_{t}", "vector_dims": dims, "fields": fields}
        for t, spec in TEACHERS.items()
    }


class TestWeightMatrix:
    def test_cell_count_is_76(self):
        m = weight_matrix()
        assert len(m) == 76
        assert sum(1 for c in m if c.startswith("iso_")) == 4
        assert sum(1 for c in m if c.startswith("grid_")) == 42
        assert sum(1 for c in m if c.startswith("grid3_")) == 30

    def test_weights_sum_to_one(self):
        for cid, weights in weight_matrix().items():
            assert abs(sum(weights.values()) - 1.0) < 1e-6, cid
            assert all(w > 0 for w in weights.values()), cid

    def test_grid3_is_rs_dominant(self):
        for cid, weights in weight_matrix().items():
            if cid.startswith("grid3_"):
                assert weights["robot_state"] >= 0.375, cid


class TestEmit:
    def test_emitted_yaml_shape(self, tmp_path):
        cids = emit_teacher("groot_tp", _calib(), tmp_path)
        assert len(cids) == 76
        index = json.loads((tmp_path / "groot_tp" / "index.json").read_text())
        assert sorted(index) == cids

        cfg = yaml.safe_load((tmp_path / "groot_tp" / "iso_robot_state.yaml").read_text())
        # Forced keybuilder fields: enabled at weight 0, not searched.
        assert cfg["keys"]["vision_0"] == {"enabled": True, "weight": 0.0}
        assert cfg["keys"]["robot_state"]["weight"] == 1.0
        assert cfg["keys"]["prompt_emb"]["enabled"] is False
        cp1 = cfg["checkpoints"]["cp1"]
        assert cp1["gate"]["type"] == "always_search"
        assert cp1["judge"]["type"] == "always_hit"
        ss = cp1["search_strategy"]
        assert ss["type"] == "weighted_score_sum_knn"
        assert ss["top_k"] == 1
        # Normalizers exist exactly for the weighted fields.
        assert list(ss["score_normalization"]["fields"]) == ["robot_state"]
        # The qdrant-default trap pin.
        assert cfg["checkpoints"]["cp3"] == {
            "enabled": False,
            "search_strategy": {"type": "weighted_rrf_knn"},
        }
        assert cfg["write_policy"] == {"type": "never"}
        assert cfg["timer"] == {"enabled": False}
        # Backend keeps the artifact's FULL dim set (prompt_emb included).
        assert set(cfg["backend"]["vector_dims"]) == {
            "vision_0", "vision_1", "vision_2", "prompt_emb", "robot_state",
        }


class TestStrategyIdentity:
    def test_run_id_and_uids_carry_the_cell(self):
        s = WsSearchStrategy(
            cid="grid_vision_0@25_robot_state@75", teacher="groot_tp",
            layout=1, style=1, base_seed=1_000_000, replan_steps=5,
            tasks=[("OpenCabinet", 2)],
        )
        assert s.run_id == "ws1-grid_vision_0@25_robot_state@75__l1s1_groot_tp"
        assert "collect" not in s.run_id
        assert s.yaml_ids == [f"{s.run_id}__OpenCabinet"]


class TestSummarizeJournal:
    def test_success_failure_error_split(self, tmp_path):
        run_id = "ws1-iso_robot_state__l1s1_groot_tp"
        yid = f"{run_id}__OpenCabinet"
        rows = [
            # robot success / robot failure / infra error / superseded stale row
            {"task_uid": f"{yid}:eval:0:0", "yaml_id": yid, "status": "done", "success": True},
            {"task_uid": f"{yid}:eval:0:1", "yaml_id": yid, "status": "failed", "success": False},
            {"task_uid": f"{yid}:eval:0:2", "yaml_id": yid, "status": "failed", "success": False,
             "error": "worker died"},
            {"task_uid": f"{yid}:eval:0:3", "yaml_id": yid, "status": "failed", "success": False,
             "accepted": False},
            {"task_uid": f"{yid}:eval:0:3", "yaml_id": yid, "status": "done", "success": True,
             "accepted": True},
        ]
        journal = tmp_path / "journal.jsonl"
        journal.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        # uid :0:4 has NO terminal record at all — the retry-exhausted case the
        # driver never journals; it must surface as missing, not vanish.
        expected = [f"{yid}:eval:0:{i}" for i in range(5)]
        summary = summarize_journal(journal, expected_uids=expected)
        task = summary["tasks"]["OpenCabinet"]
        assert task == {"succ": 2, "fail": 1, "err": 1, "missing": 1,
                        "n_scored": 3, "sr": pytest.approx(2 / 3)}
        assert summary["macro_sr"] == pytest.approx(2 / 3)
        assert summary["n_err"] == 1
        assert summary["n_missing"] == 1
        assert summary["complete"] is False

    def test_clean_cell_is_complete(self, tmp_path):
        run_id = "ws1-iso_robot_state__l1s1_groot_tp"
        yid = f"{run_id}__OpenCabinet"
        rows = [
            {"task_uid": f"{yid}:eval:0:0", "yaml_id": yid, "status": "done", "success": True},
            {"task_uid": f"{yid}:eval:0:1", "yaml_id": yid, "status": "failed", "success": False},
        ]
        journal = tmp_path / "journal.jsonl"
        journal.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        summary = summarize_journal(journal, expected_uids=[r["task_uid"] for r in rows])
        assert summary["complete"] is True
        assert summary["macro_sr"] == pytest.approx(1 / 2)
