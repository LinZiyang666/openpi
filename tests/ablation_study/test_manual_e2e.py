"""Manual 1-cell end-to-end acceptance for Phase 2 (plan §5).

Requires live infrastructure (GPU main server with a routing yaml loaded +
sidecar + LIBERO client run already executed); asserts the smoke acceptance
criteria over the produced per-step rows: every episode completed, per-step
rows carry the executor field, executor counts match hit_type counts, and the
sidecar timing log row count equals the routed step count.

Run: ABLATION_PER_STEP=<rows.jsonl> ABLATION_SIDECAR_LOG=<timing.jsonl> \
     ABLATION_JOURNAL=<journal.jsonl> ABLATION_EXPECTED_EPISODES=<n> \
     ABLATION_CLIENT_LOG=<client.log> \
     uv run pytest tests/ablation_study/test_manual_e2e.py -m manual
"""

from __future__ import annotations

import json
import os

import pytest

pytestmark = pytest.mark.manual


def _rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_one_cell_smoke_acceptance():
    per_step = os.environ.get("ABLATION_PER_STEP")
    sidecar_log = os.environ.get("ABLATION_SIDECAR_LOG")
    if not per_step or not sidecar_log:
        pytest.skip("set ABLATION_PER_STEP and ABLATION_SIDECAR_LOG")
    rows = [r for r in _rows(per_step) if r.get("hit_type")]
    assert rows, "no per-step verdict rows recorded"
    assert all("executor" in r for r in rows), "executor field missing from rows"
    routed = [r for r in rows if r.get("executor") == "override"]
    arm = rows[0]["yaml_id"]
    if "hit_" in arm:
        expected = [r for r in rows if r["hit_type"] == "FULL_HIT"]
    else:
        expected = [r for r in rows if r["hit_type"] == "MISS"]
    assert len(routed) == len(expected), (len(routed), len(expected))
    assert len(_rows(sidecar_log)) == len(routed)
    # Completion + cleanliness gates: every scheduled episode must have a
    # terminal journal record, and the client log must be Traceback-free.
    # Completion + cleanliness are MANDATORY acceptance criteria: a missing
    # input fails the smoke instead of silently skipping the check.
    journal = os.environ.get("ABLATION_JOURNAL")
    expected = int(os.environ.get("ABLATION_EXPECTED_EPISODES", "0"))
    client_log = os.environ.get("ABLATION_CLIENT_LOG")
    if not (journal and expected and client_log):
        pytest.fail(
            "smoke acceptance requires ABLATION_JOURNAL, "
            "ABLATION_EXPECTED_EPISODES and ABLATION_CLIENT_LOG to be set"
        )
    terminal = {r["task_uid"] for r in _rows(journal) if r.get("status") in ("done", "failed")}
    assert len(terminal) == expected, (len(terminal), expected)
    text = open(client_log, encoding="utf-8", errors="replace").read()
    assert "Traceback" not in text and "ERROR" not in text
