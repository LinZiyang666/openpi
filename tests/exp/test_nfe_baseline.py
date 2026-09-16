"""Tests for the reduced-step teacher baseline tooling (exp/nfe_baseline).

Covers the three places a silent error would move a published number: the
shard filters (every pool position exactly once), the per-k pricing (bit-equal
to the RIT lines' own cost authorities at the tiers they share), and the shard
join (duplicates, wrong suite and short counts are refusals). The Pi0.5 server
wrapper's argument gate and class patch are exercised without loading a model.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from exp.dispatch_surface.analysis import analytic_cost as ac
from exp.libero_groot.aggregate_rit_lg import load_cost
from exp.nfe_baseline import aggregate_nfe, make_shards, serve_pi05_ksweep
from exp.rit_pareto import rit_k
from exp.robocasa365 import rit_cost_rc as rc

GROOT_COST = pathlib.Path("exp/libero_groot/config/rit/cost_groot_libero_measured.json")


# ------------------------------------------------------------------
# Shards
# ------------------------------------------------------------------


def test_shards_tile_the_pool_exactly():
    n_per_task = {t: 50 for t in range(10)}
    shards = make_shards.shard_entries(n_per_task, 5)
    assert [len(s) for s in shards] == [100] * 5
    seen = [(e["task_id"], e["orig_init_state_idx"]) for s in shards for e in s]
    assert len(seen) == len(set(seen)) == 500
    assert set(seen) == {(t, p) for t in range(10) for p in range(50)}
    for e in shards[0]:
        assert e["subset_init_state_idx"] == e["orig_init_state_idx"]


def test_shards_uneven_pool_and_single_shard():
    shards = make_shards.shard_entries({0: 7, 1: 3}, 4)
    assert sum(len(s) for s in shards) == 10
    assert [len(s) for s in make_shards.shard_entries({0: 50}, 1)] == [50]
    with pytest.raises(ValueError):
        make_shards.shard_entries({0: 50}, 0)


# ------------------------------------------------------------------
# Pricing
# ------------------------------------------------------------------


def test_pi05_pricing_matches_the_rit_cost_authority():
    cm = aggregate_nfe.CostModel("pi05")
    assert cm.num_steps == 10
    assert cm.decision_ms(10) == ac.unit_cost("MISS", None)
    assert cm.ir_percent(10) == pytest.approx(100.0, abs=1e-12)
    # k=3 is the warm@0.3 tier, k=5 the warm@0.5 tier: same x on the frontier.
    assert cm.decision_ms(3) == pytest.approx(ac.unit_cost("WARM_START", 0.3), abs=1e-9)
    assert cm.decision_ms(5) == pytest.approx(
        rit_k.tier_cost("WARM_START", 0.5), abs=1e-9
    )
    assert cm.decision_ms(1) == pytest.approx(
        ac.STAGE1_MS + ac.STAGE2_MS + 0.1 * ac.STAGE3_MS
    )
    irs = [cm.ir_percent(k) for k in range(1, 11)]
    assert irs == sorted(irs) and irs[0] > 60.0
    with pytest.raises(ValueError):
        cm.decision_ms(0)
    with pytest.raises(ValueError):
        cm.decision_ms(11)


def test_groot_pricing_matches_the_measured_ledger():
    cm = aggregate_nfe.CostModel("groot", GROOT_COST)
    cost = load_cost(GROOT_COST)
    assert cm.num_steps == 8
    assert cm.decision_ms(8) == rc.miss_cost(cost)
    assert cm.ir_percent(8) == pytest.approx(100.0, abs=1e-12)
    # k=2 leaves the same two steps as the t=0.75 rung, k=4 the same four as t=0.5.
    assert cm.decision_ms(2) == pytest.approx(
        rc.tier_cost(cost, "WARM_START", 0.75), abs=1e-9
    )
    assert cm.decision_ms(4) == pytest.approx(
        rc.tier_cost(cost, "WARM_START", 0.5), abs=1e-9
    )
    assert cm.ir_percent(1) == pytest.approx(
        100 * (cost.stage1_ms + cost.stage2_ms + cost.stage3_ms(1)) / rc.miss_cost(cost)
    )


def test_unknown_policy_is_refused():
    with pytest.raises(ValueError):
        aggregate_nfe.CostModel("smolvla")


# ------------------------------------------------------------------
# Shard join
# ------------------------------------------------------------------


def _rows(suite, task_ids, positions, success=True):
    return [
        {
            "task_id": t,
            "orig_init_state_idx": p,
            "init_state_idx": p,
            "success": success,
            "task_suite_name": suite,
        }
        for t in task_ids
        for p in positions
    ]


def _write(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text(json.dumps(rows))
    return p


def test_join_shards_counts_and_rates(tmp_path):
    a = _write(
        tmp_path,
        "libero_spatial_k3_s0.json",
        _rows("libero_spatial", range(10), range(0, 50, 2), True),
    )
    b = _write(
        tmp_path,
        "libero_spatial_k3_s1.json",
        _rows("libero_spatial", range(10), range(1, 50, 2), False),
    )
    rec = aggregate_nfe.join_shards([a, b], suite="libero_spatial", expect=500)
    assert rec["n_ep"] == 500
    assert rec["success_rate"] == pytest.approx(0.5)
    assert rec["per_task"]["0"] == {"n": 50, "success_rate": 0.5}
    assert set(rec["shard_files"]) == {a.name, b.name}
    assert aggregate_nfe.shard_files(tmp_path, "libero_spatial", 3) == [a, b]
    assert aggregate_nfe.shard_files(tmp_path, "libero_spatial", 4) == []


def test_join_shards_refuses_duplicates_short_counts_and_wrong_suite(tmp_path):
    a = _write(tmp_path, "a.json", _rows("libero_10", [0], range(10)))
    dup = _write(tmp_path, "dup.json", _rows("libero_10", [0], range(5, 15)))
    with pytest.raises(SystemExit, match="duplicate"):
        aggregate_nfe.join_shards([a, dup], suite="libero_10", expect=None)
    with pytest.raises(SystemExit, match="expected 500"):
        aggregate_nfe.join_shards([a], suite="libero_10", expect=500)
    with pytest.raises(SystemExit, match="suite"):
        aggregate_nfe.join_shards([a], suite="libero_spatial", expect=None)
    # expect=None (smoke) accepts a partial join.
    assert aggregate_nfe.join_shards([a], suite="libero_10", expect=None)["n_ep"] == 10


def test_parse_ks():
    assert aggregate_nfe.parse_ks("1-9") == list(range(1, 10))
    assert aggregate_nfe.parse_ks("5,1,3") == [1, 3, 5]
    assert aggregate_nfe.parse_ks("1-3,7") == [1, 2, 3, 7]


# ------------------------------------------------------------------
# Aggregate + figure spec
# ------------------------------------------------------------------


def test_aggregate_and_figure_spec(tmp_path):
    for k, sr in ((1, 0.2), (2, 0.6)):
        _write(
            tmp_path,
            f"libero_10_k{k}_s0.json",
            _rows("libero_10", range(10), range(50), True)[: int(500 * sr)]
            + _rows("libero_10", range(10), range(50), False)[int(500 * sr) :],
        )
    cm = aggregate_nfe.CostModel("groot", GROOT_COST)
    anchor = {
        "success_rate": 0.868,
        "n_ep": 500,
        "source": "test",
        "k": 8,
        "ir_percent": 100.0,
    }
    agg = aggregate_nfe.aggregate(
        "groot", "libero_10", tmp_path, [1, 2], expect=500, cost=cm, anchor=anchor
    )
    assert agg["num_steps"] == 8 and set(agg["per_k"]) == {"1", "2"}
    assert agg["per_k"]["1"]["success_rate"] == pytest.approx(0.2)
    assert agg["per_k"]["2"]["ir_percent"] == pytest.approx(cm.ir_percent(2))
    spec = aggregate_nfe.figure_spec(
        "groot", "libero_10", {int(k): v for k, v in agg["per_k"].items()}, cm, anchor
    )
    assert (
        spec["schema"] == "rit_pareto.figure/v1"
        and spec["figure_id"] == "nfe_groot_libero_10"
    )
    pts = spec["series"][0]["points"]
    assert [p["label"] for p in pts] == ["k=1", "k=2", "k=8 (policy alone)"]
    assert pts[-1]["x"] == 100.0 and pts[-1]["y"] == 0.868
    assert all(p["n_ep"] == 500 for p in pts)
    ref = aggregate_nfe.reference_overlay(
        "groot_libero_10", "RIT k=2 (+ warm, 2 steps left)", "RIT k=2", "#000"
    )
    assert (
        ref["source_figure"] == "groot_libero_10"
        and ref["show_frontier"]
        and not ref["show_points"]
    )


def test_aggregate_refuses_missing_k(tmp_path):
    cm = aggregate_nfe.CostModel("pi05")
    with pytest.raises(SystemExit, match="no shard files"):
        aggregate_nfe.aggregate(
            "pi05", "libero_spatial", tmp_path, [1], expect=None, cost=cm, anchor=None
        )


# ------------------------------------------------------------------
# Pi0.5 ksweep wrapper
# ------------------------------------------------------------------


def test_ksweep_argument_gate():
    k, rest, batched = serve_pi05_ksweep._parse(
        ["--denoising-steps", "3", "--port", "23150", "policy:checkpoint"]
    )
    assert (
        k == 3 and rest == ["--port", "23150", "policy:checkpoint"] and batched is False
    )
    assert (
        serve_pi05_ksweep._parse(["--denoising-steps", "2", "--cache", "--port", "1"])[
            2
        ]
        is True
    )
    assert serve_pi05_ksweep._parse(["--denoising-steps", "2", "--replicas", "1"])[
        1
    ] == ["--replicas", "1"]
    with pytest.raises(SystemExit, match=">= 1"):
        serve_pi05_ksweep._parse(["--denoising-steps", "0"])
    with pytest.raises(SystemExit, match="replicas"):
        serve_pi05_ksweep._parse(["--denoising-steps", "2", "--replicas", "4"])
    with pytest.raises(SystemExit, match="replicas"):
        serve_pi05_ksweep._parse(["--denoising-steps", "2", "--replicas=2"])
    with pytest.raises(SystemExit, match="policy alone"):
        serve_pi05_ksweep._parse(["--denoising-steps", "2", "--cache_config", "x.yaml"])
    with pytest.raises(SystemExit, match="policy alone"):
        serve_pi05_ksweep._parse(["--denoising-steps", "2", "--cache-config=x.yaml"])


def test_ksweep_pins_num_steps_on_both_entry_points(monkeypatch, capsys):
    from openpi.models_pytorch import pi0_pytorch

    calls = []

    def fake_expert(
        self, state, prefix_pad_masks, past_key_values, noise, num_steps=10
    ):
        calls.append(("expert", num_steps))
        return "chunk"

    def fake_stage3(
        self,
        stage2,
        *,
        noise=None,
        num_steps=10,
        return_intermediates=False,
        save_timesteps=(0.7, 0.5, 0.3),
    ):
        calls.append(("stage3", num_steps, return_intermediates))
        return "out"

    monkeypatch.setattr(pi0_pytorch.PI0Pytorch, "_stage3_action_expert", fake_expert)
    monkeypatch.setattr(pi0_pytorch.PI0Pytorch, "run_stage3", fake_stage3)
    serve_pi05_ksweep.pin_num_steps(4)
    obj = object.__new__(pi0_pytorch.PI0Pytorch)
    assert (
        pi0_pytorch.PI0Pytorch._stage3_action_expert(obj, None, None, None, None)
        == "chunk"
    )
    assert (
        pi0_pytorch.PI0Pytorch._stage3_action_expert(
            obj, None, None, None, None, num_steps=10
        )
        == "chunk"
    )
    assert (
        pi0_pytorch.PI0Pytorch.run_stage3(obj, None, return_intermediates=True) == "out"
    )
    assert (
        pi0_pytorch.PI0Pytorch.run_stage3(
            obj, None, num_steps=10, return_intermediates=True
        )
        == "out"
    )
    assert calls == [
        ("expert", 4),
        ("expert", 4),
        ("stage3", 4, True),
        ("stage3", 4, True),
    ]
    out = capsys.readouterr().out
    assert out.count("KSWEEP stage3 first call num_steps=4") == 1
    assert out.count("KSWEEP run_stage3 first call num_steps=4") == 1


def test_ksweep_lock_serialises_infer(monkeypatch):
    from openpi.policies import policy as _policy

    seen = []

    def fake_infer(self, obs, *, noise=None):
        seen.append((obs, noise))
        return {"ok": True}

    monkeypatch.setattr(_policy.Policy, "infer", fake_infer)
    serve_pi05_ksweep.lock_infer()
    obj = object.__new__(_policy.Policy)
    assert _policy.Policy.infer(obj, {"a": 1}, noise=None) == {"ok": True}
    assert seen == [({"a": 1}, None)]


def test_ksweep_stamps_metadata(monkeypatch):
    from openpi.serving import websocket_policy_server as wps

    captured = {}

    def fake_init(self, *a, **kw):
        captured.update(kw)

    monkeypatch.setattr(wps.WebsocketPolicyServer, "__init__", fake_init)
    serve_pi05_ksweep.stamp_metadata(6)
    wps.WebsocketPolicyServer(
        policy=None, host="0.0.0.0", port=1, metadata={"concurrent": True}
    )
    assert captured["metadata"] == {"concurrent": True, "nfe_num_steps": 6}
    wps.WebsocketPolicyServer(policy=None, host="0.0.0.0", port=1)
    assert captured["metadata"] == {"nfe_num_steps": 6}


# ------------------------------------------------------------------
# Metadata probe (client side)
# ------------------------------------------------------------------


def test_probe_metadata_reports_down_and_step_keys(monkeypatch):
    import socket

    from exp.nfe_baseline import probe_metadata

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    free_port = sock.getsockname()[1]
    sock.close()
    assert probe_metadata.probe("127.0.0.1", free_port, timeout=0.5) == "down"

    class FakeClient:
        metadata = {}

        def __init__(self, host, port):
            self._ws = type("W", (), {"close": lambda self: None})()

        def get_server_metadata(self):
            return dict(FakeClient.metadata)

    import openpi_client.websocket_client_policy as wcp

    monkeypatch.setattr(wcp, "WebsocketClientPolicy", FakeClient)
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: socket.socket())
    FakeClient.metadata = {"denoising_steps": 3}
    assert probe_metadata.probe("h", 1) == "3"
    FakeClient.metadata = {"nfe_num_steps": 7, "concurrent": True}
    assert probe_metadata.probe("h", 1) == "7"
    FakeClient.metadata = {"concurrent": True}
    assert probe_metadata.probe("h", 1) == "none"


# ------------------------------------------------------------------
# RoboCasa365 branch
# ------------------------------------------------------------------

RC_COST_GROOT = pathlib.Path("exp/nfe_baseline/config/rc_cost_groot_tp.json")
RC_COST_PI05 = pathlib.Path("exp/nfe_baseline/config/rc_cost_pi05.json")


def test_rc_pricing_matches_the_rit_ledgers():
    g = aggregate_nfe.CostModel("groot_rc")
    gc = aggregate_nfe.load_rc_cost(RC_COST_GROOT, "groot_tp")
    assert g.num_steps == 4 and g.decision_ms(4) == rc.miss_cost(gc)
    # k steps from noise cost what a warm start with k steps left costs (ascending schedule).
    assert g.decision_ms(1) == pytest.approx(
        rc.tier_cost(gc, "WARM_START", 0.75), abs=1e-9
    )
    assert g.decision_ms(2) == pytest.approx(
        rc.tier_cost(gc, "WARM_START", 0.5), abs=1e-9
    )
    assert 74 < g.ir_percent(1) < 75 and g.ir_percent(4) == pytest.approx(
        100.0, abs=1e-12
    )
    p = aggregate_nfe.CostModel("pi05_rc")
    pc = aggregate_nfe.load_rc_cost(RC_COST_PI05, "pi05")
    assert p.num_steps == 10 and p.decision_ms(10) == rc.miss_cost(pc)
    assert p.decision_ms(3) == pytest.approx(
        rc.tier_cost(pc, "WARM_START", 0.3), abs=1e-9
    )
    assert p.decision_ms(5) == pytest.approx(
        rc.tier_cost(pc, "WARM_START", 0.5), abs=1e-9
    )
    assert 59 < p.ir_percent(1) < 60.5


def _summary(teacher, tasks, sr=0.5, n=50, complete=True):
    return {
        "cid": "teacher",
        "teacher": teacher,
        "tasks": {
            t: {
                "succ": int(n * sr),
                "fail": n - int(n * sr),
                "err": 0,
                "missing": 0,
                "n_scored": n,
                "sr": sr,
            }
            for t in tasks
        },
        "macro_sr": sr,
        "n_err": 0,
        "n_missing": 0,
        "complete": complete,
    }


def _write_rc(root, teacher, lane, k, payload, prefix="nfek"):
    p = aggregate_nfe.rc_summary_path(root, teacher, lane, k, prefix)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload))
    return p


def test_rc_join_macro_over_13_tasks(tmp_path):
    main = _write_rc(
        tmp_path,
        "groot_tp",
        "main",
        2,
        _summary("groot_tp", aggregate_nfe.RC_LANES["main"], sr=0.6),
    )
    pnp = _write_rc(
        tmp_path,
        "groot_tp",
        "pnp",
        2,
        _summary("groot_tp", aggregate_nfe.RC_LANES["pnp"], sr=0.8),
    )
    rec = aggregate_nfe.join_rc_summaries(
        {"main": main, "pnp": pnp}, teacher="groot_tp", per_task=50
    )
    assert rec["n_tasks"] == 13 and rec["n_ep"] == 650
    assert rec["success_rate"] == pytest.approx((8 * 0.6 + 5 * 0.8) / 13)
    cm = aggregate_nfe.CostModel("groot_rc")
    agg = aggregate_nfe.aggregate_rc(
        "groot_rc", tmp_path, [2], per_task=50, cost=cm, anchor=None
    )
    assert agg["suite"] == "robocasa365" and agg["per_k"]["2"][
        "ir_percent"
    ] == pytest.approx(cm.ir_percent(2))


def test_rc_join_refuses_incomplete_short_and_wrong_roster(tmp_path):
    pnp = _write_rc(
        tmp_path, "pi05", "pnp", 1, _summary("pi05", aggregate_nfe.RC_LANES["pnp"])
    )
    bad = _write_rc(
        tmp_path,
        "pi05",
        "main",
        1,
        _summary("pi05", aggregate_nfe.RC_LANES["main"], complete=False),
    )
    with pytest.raises(SystemExit, match="not complete"):
        aggregate_nfe.join_rc_summaries(
            {"main": bad, "pnp": pnp}, teacher="pi05", per_task=50
        )
    short = _write_rc(
        tmp_path,
        "pi05",
        "main",
        1,
        _summary("pi05", aggregate_nfe.RC_LANES["main"], n=49),
    )
    with pytest.raises(SystemExit, match="expected 50"):
        aggregate_nfe.join_rc_summaries(
            {"main": short, "pnp": pnp}, teacher="pi05", per_task=50
        )
    roster = _write_rc(
        tmp_path,
        "pi05",
        "main",
        1,
        _summary("pi05", aggregate_nfe.RC_LANES["main"][:7]),
    )
    with pytest.raises(SystemExit, match="roster"):
        aggregate_nfe.join_rc_summaries(
            {"main": roster, "pnp": pnp}, teacher="pi05", per_task=50
        )
    with pytest.raises(SystemExit, match="missing summaries"):
        aggregate_nfe.aggregate_rc(
            "pi05_rc",
            tmp_path,
            [9],
            per_task=50,
            cost=aggregate_nfe.CostModel("pi05_rc"),
            anchor=None,
        )
