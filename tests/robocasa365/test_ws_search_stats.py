"""Tests for the paired read-out of the weighted-sum search and the round tag.

Two things are easy to get wrong and expensive to notice late:

* a cell caught mid-flight can hold a whole number of finished tasks, so a
  completeness check against the cell's *own* task count lets it through and
  the paired analysis then dies on a missing key;
* the orchestrator must keep driving remotes whose ``run_ws_search.py``
  predates ``--run-prefix``, so the flag may only be spelled out when it is
  not the default.
"""

from __future__ import annotations

import json

import pytest

from exp.robocasa365.analyze_ws_search_stats import load_journals, macro_sr, signflip_p
from exp.robocasa365.orchestrate_ws_search import (
    CellQueue,
    Slot,
    pack_agent_c_gpus,
    stratify_by_family,
)

TASKS = ("OpenCabinet", "CloseFridge")


def _journal(tmp_path, cid, outcomes, prefix="ws1"):
    """outcomes: {(task, idx): success} -> one journal file for that cell."""
    run_id = f"{prefix}-{cid}__l1s1_groot_tp"
    rows = []
    for (task, idx), success in outcomes.items():
        yid = f"{run_id}__{task}"
        rows.append({
            "task_uid": f"{yid}:eval:0:{idx}", "yaml_id": yid, "phase": "eval",
            "status": "done" if success else "failed", "success": success,
            "attempt": 1, "accepted": True,
        })
    path = tmp_path / f"journal_{run_id}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


class TestLoadJournals:
    def test_full_grid_cell_is_kept(self, tmp_path):
        outcomes = {(t, i): (i % 2 == 0) for t in TASKS for i in range(4)}
        _journal(tmp_path, "cellA", outcomes)
        cells, grid = load_journals(tmp_path, episodes=4)
        assert set(cells) == {"cellA"}
        assert len(grid) == len(TASKS) * 4
        assert macro_sr(cells["cellA"], sorted(TASKS), range(4)) == pytest.approx(0.5)

    def test_cell_missing_whole_tasks_is_dropped(self, tmp_path):
        full = {(t, i): True for t in TASKS for i in range(4)}
        _journal(tmp_path, "cellA", full)
        _journal(tmp_path, "cellB", full)
        # cellC finished only one of the two tasks: a whole number of episodes,
        # so a per-cell "n_tasks * episodes" check would wave it through.
        _journal(tmp_path, "cellC", {(TASKS[0], i): True for i in range(4)})
        cells, grid = load_journals(tmp_path, episodes=4)
        assert set(cells) == {"cellA", "cellB"}
        assert len(grid) == len(TASKS) * 4

    def test_run_prefix_selects_the_round(self, tmp_path):
        outcomes = {(t, i): True for t in TASKS for i in range(4)}
        _journal(tmp_path, "cellA", outcomes, prefix="ws1")
        _journal(tmp_path, "cellA", outcomes, prefix="ws2")
        assert set(load_journals(tmp_path, 4, "ws1")[0]) == {"cellA"}
        assert set(load_journals(tmp_path, 4, "ws2")[0]) == {"cellA"}
        assert load_journals(tmp_path, 4, "ws3")[0] == {}


class TestSignFlip:
    def test_identical_cells_are_not_separated(self):
        import random
        d, p = signflip_p([0] * 40, weight=1 / 40, rng=random.Random(0), resamples=200)
        assert d == 0.0
        assert p == 1.0

    def test_one_sided_discordance_is_separated(self):
        import random
        # every discordant pair favours the same cell: the sign-flip reference
        # distribution reaches that extreme only at probability 2^-20.
        d, p = signflip_p([1] * 20 + [0] * 20, weight=1 / 40, rng=random.Random(0), resamples=500)
        assert d == pytest.approx(0.5)
        assert p < 0.01


class TestStratifyByFamily:
    def test_prefix_is_balanced_across_families(self):
        cids = [f"grid_{i}" for i in range(42)] + [f"grid3_{i}" for i in range(30)] + \
               [f"iso_{i}" for i in range(4)]
        order = stratify_by_family(cids)
        assert sorted(order) == sorted(cids)          # a permutation, nothing dropped
        families = {c.split("_", 1)[0] for c in order[:9]}
        assert families == {"grid", "grid3", "iso"}   # every family present in a short prefix
        # the cid sort order, by contrast, is family-blocked ("grid3_" sorts
        # ahead of "grid_" because '3' < '_', which is how the real matrix's
        # alphabetical prefix came out all-grid3)
        assert len({c.split("_", 1)[0] for c in sorted(cids)[:9]}) == 1

    def test_order_is_deterministic(self):
        cids = [f"grid_{i}" for i in range(10)] + [f"iso_{i}" for i in range(3)]
        assert stratify_by_family(cids) == stratify_by_family(list(reversed(cids)))


class TestCellQueue:
    def test_shared_pop_until_empty(self):
        q = CellQueue(["a", "b", "c"])
        assert len(q) == 3
        assert [q.pop(), q.pop(), q.pop(), q.pop()] == ["a", "b", "c", None]
        assert q.pop() is None  # drained stays drained

    def test_concurrent_pops_never_duplicate(self):
        import threading
        q = CellQueue([str(i) for i in range(500)])
        got, lock = [], threading.Lock()

        def drain():
            while (item := q.pop()) is not None:
                with lock:
                    got.append(item)

        threads = [threading.Thread(target=drain) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(got, key=int) == [str(i) for i in range(500)]


class TestPackAgentCGpus:
    def test_fills_one_card_before_the_next(self):
        # 4 slots x 6 workers, cap 24: every slot lands on the FIRST card.
        assert [pack_agent_c_gpus(i, 6, "0,2,3", 24) for i in range(4)] == ["0", "0", "0", "0"]

    def test_overflows_in_order_once_the_cap_is_hit(self):
        # 4 slots x 16 workers, cap 24: 0,16 -> card0; 32,48 -> card2, card3.
        assert [pack_agent_c_gpus(i, 16, "0,2,3", 24) for i in range(4)] == ["0", "0", "2", "3"]

    def test_never_runs_past_the_last_card(self):
        assert pack_agent_c_gpus(9, 16, "0,2", 24) == "2"


class TestOrchestratorRunPrefix:
    def _slot(self, **kw):
        return Slot(teacher="groot_tp", port=23160, pull_port=23180, cids=["cellA"],
                    timan_workers=8, weiland_workers=2, timan_gpus="0", **kw)

    def test_default_prefix_emits_no_flag(self):
        slot = self._slot()
        assert slot.prefix_flag == ""
        assert slot.run_id("cellA") == "ws1-cellA__l1s1_groot_tp"

    def test_custom_prefix_is_passed_through(self):
        slot = self._slot(run_prefix="ws2")
        assert slot.prefix_flag == "--run-prefix ws2 "
        assert slot.run_id("cellA") == "ws2-cellA__l1s1_groot_tp"
