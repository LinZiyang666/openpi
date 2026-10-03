"""CPU tests for dispatch, state isolation, frozen packaging and paired inference."""
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from exp.offline_search.closed_loop.plugin import clone_method
from exp.offline_search.harness import api
from exp.offline_search.rounds.r08.methods.methods import PolicyEveryTen
from exp.offline_search.rounds.r08.methods.test_methods import fixture, qview, tape

from . import analyze
from .common import CELLS, HERE, R8, RUN, SPEC, VARIANTS, fit_path, sha
from .methods import TaskBudgetController
from .replay import assert_same


def attached(rates, controllers, policy_tasks=()):
    method = TaskBudgetController(rates, {}, policy_tasks)
    method.controllers = controllers
    return method


@pytest.mark.parametrize("rates,policies", [({}, []), ({"0": -1}, []), ({"0": float("nan")}, []),
    ({"0": True}, []), ({"0": 1.1}, []), ({"00": 0}, []), ({"-1": 0}, []),
    ({"0": 1}, [1]), ({"0": .3}, [0]), ({"0": 1}, [0, 0]), ({"0": 1}, [False])])
def test_invalid_tables_fail_closed(rates, policies):
    with pytest.raises(ValueError):
        TaskBudgetController(rates, {}, policies)


@pytest.mark.parametrize("model", ["pi05", "groot"])
def test_zero_rate_raw_key_replay_is_exact_a(model):
    base, lib, _, _ = fixture(model)
    wrapped, _ = clone_method(base, strict=True)
    method = attached({"0": 0}, {"A": wrapped})
    ep = SimpleNamespace(uid="raw-key-equivalence", task_id=0, init=4)
    ref, _ = tape(base, lib, np.tile(np.arange(16), 4), ep)
    actual, _ = tape(method, lib, np.tile(np.arange(16), 4), ep)
    for (rl, r), (al, a) in zip(ref, actual):
        assert rl == al
        assert_same(r, a)
    assert method.last_blind_extras == base.last_blind_extras


class RecordingBranch:
    def __init__(self, label):
        self.label, self.resets, self.calls = label, [], []
        self.last_blind_extras = {"label": label}

    def reset(self, ep):
        self.resets.append((ep.task_id, ep.init))

    def query(self, q):
        self.calls.append((q.task_id, q.step))
        return self.label

    blind_step = query
    policy_tail_step = query

    def invalidate_anchor(self):
        self.calls.append("invalidate")


def test_task_switch_and_implicit_reset_do_not_leak_stall_calls():
    a, c, p = [RecordingBranch(x) for x in ("A", "CU", "P10")]
    m = attached({"0": 0, "1": .3, "2": 1}, {"A": a, "CU:0.3": c, "P10": p}, [2])
    for task in (1, 0, 2, 1, 0):
        ep = SimpleNamespace(uid="episode" + str(task), task_id=task, init=3)
        q = SimpleNamespace(episode=ep, task_id=task, step=0)
        assert m.blind_step(q) == ("A", "CU", "P10")[task]
        assert m.query(q) == ("A", "CU", "P10")[task]
        m.invalidate_anchor()
        assert m.last_blind_extras == {"label": ("A", "CU", "P10")[task]}
    assert len(a.resets) == 2 and len(c.resets) == 2 and len(p.resets) == 1
    with pytest.raises(api.ContractError):
        m.reset(SimpleNamespace(uid="unknown", task_id=9, init=0))
    with pytest.raises(api.ContractError):
        m.query(SimpleNamespace(episode=ep, task_id=1, step=0))


def test_p10_dispatch_preserves_policy_tail_and_invalidates_only_active():
    base, lib, _, kwargs = fixture()
    p = PolicyEveryTen(base_kwargs=kwargs)
    p.base, _ = clone_method(base, strict=True)
    m = attached({"0": 1}, {"P10": clone_method(p, strict=True)[0]}, [0])
    ep = SimpleNamespace(uid="P10", task_id=0, init=4)
    p.reset(ep)
    m.reset(ep)
    history = dict(rs=[], action=[], vision=[], hit=[], v0=[], v1=[], age=0)
    q = qview(lib, 0, 0, ep, history)
    assert_same(p.query(q), m.query(q))
    p.invalidate_anchor()
    m.invalidate_anchor()
    for key, value in (("rs", q.rs), ("action", lib.action[0]), ("vision", True),
                       ("hit", False), ("v0", q.key_v0), ("v1", q.key_v1)):
        history[key].append(value)
    q = qview(lib, 1, 1, ep, history)
    assert_same(p.policy_tail_step(q), m.policy_tail_step(q))


def test_connection_clone_shares_fit_arrays_and_isolates_episode_state():
    base, _, _, _ = fixture()
    m = attached({"0": 0}, {"A": base})
    other, mode = clone_method(m, strict=True)
    assert mode == "deepcopy_shared_arrays"
    assert other.controllers["A"].act is base.act
    assert other.controllers["A"] is not base
    ep = SimpleNamespace(uid="clone", task_id=0, init=4)
    other.reset(ep)
    assert m.active is None and other.active is other.controllers["A"]


def test_emitted_arm_contracts_and_frozen_source_sha():
    rows = json.loads((RUN / "arms.json").read_text())
    r8 = {r["arm"]: r for r in json.loads((R8 / "arms.json").read_text())}
    lock = json.loads((HERE / "source_lock.json").read_text())
    table = json.loads((HERE / "task_tables.json").read_text())
    assert len(rows) == 12
    assert {r["arm"] for r in rows} == {f"r9p1_{c}_{v}" for c in CELLS for v in VARIANTS}
    manifest = json.loads((RUN / "manifests/discovery300.json").read_text())
    assert len(manifest) == 300 and set(map(tuple, manifest)) == analyze.PAIRS
    for row in rows:
        cell, variant = row["p1"]["cell"], row["p1"]["variant"]
        assert row["manifest"] == str(RUN / "manifests/discovery300.json")
        assert not row["p1"]["debug_required"]
        assert not any(x.startswith(("--os-debug", "--os-log-inputs", "--os-oracle", "--trace")) for x in row["plugin_args"])
        assert sha(row["yaml"]) == lock["cells"][cell]["A"]["config_sha256"]
        with fit_path(row).open("rb") as f:
            blob = pickle.load(f)
        assert blob["spec"] == row["method"] and blob["kwargs"] == row["kwargs"] and blob["cell"] == row["cell"]
        if variant in ("A", "CU"):
            old = r8["r8_" + cell + "_" + variant]
            for key in ("method", "kwargs", "plugin_args", "client_overrides"):
                assert row[key] == old[key]
            assert sha(fit_path(row)) == lock["cells"][cell][variant]["source"]["sha256"]
        else:
            assert row["method"] == SPEC and row["full_model"] and row["judge"] == "guard_only"
            assert row["kwargs"]["task_rates"] == table["cells"][cell][variant]
            assert sum(v > 0 for v in row["kwargs"]["task_rates"].values()) == 3


def test_accepted_retry_semantics_and_manifest_completeness():
    rows = [dict(task_uid="arm:eval:0:0", accepted=True, status="failed", success=False, attempt=1),
            dict(task_uid="arm:eval:0:0", accepted=False, status="done", success=True, attempt=2),
            dict(task_uid="arm:eval:0:0", accepted=True, status="done", success=True, attempt=3)]
    assert analyze.accepted_journals(rows, {(0, 0)})["arm:eval:0:0"]["attempt"] == 3
    with pytest.raises(ValueError, match="incomplete"):
        analyze.accepted_journals(rows, {(0, 0), (0, 1)})
    with pytest.raises(ValueError, match="multiple accepted"):
        analyze.accepted_journals(rows + [dict(rows[0], task_uid="other:eval:0:0")], {(0, 0)})


def test_paired_simulation_reproduces_frozen_predictions_and_pooled_ir():
    df = analyze.simulated_episode_table()
    report = analyze.analyze(df, n_boot=100, kind="test-only prior simulation")
    pred = json.loads((HERE / "predictions.json").read_text())["arms"]
    for cell in CELLS:
        for variant in VARIANTS:
            got = report["cells"][cell]["arms"][variant]
            ref = pred[f"r9p1_{cell}_{variant}"]["fixed_table_simulator"]
            assert got["sr"] == ref["sr"] and got["ir"] == ref["ir"]
        for c in report["cells"][cell]["comparisons"]:
            assert c["mcnemar_holm_p"] >= c["mcnemar_exact_p"]
    with pytest.raises(ValueError, match="duplicate"):
        analyze.validate_episode_table(__import__("pandas").concat([df, df.iloc[:1]]))
    with pytest.raises(ValueError, match="incomplete"):
        analyze.validate_episode_table(df.iloc[1:])


def test_holm_adjustment_known_values():
    assert analyze.holm([.01, .04, .03]) == pytest.approx([.03, .06, .06])


def test_fit_refuses_tampered_source_and_wrong_task_coverage(tmp_path):
    base, _, _, kwargs = fixture()
    path = tmp_path / "source.pkl"
    spec = "exp.offline_search.rounds.r04.k1_blind.blind_awm:BlindAWM"
    path.write_bytes(pickle.dumps(dict(method=base, spec=spec, kwargs=kwargs, cell="pi05_l10_cache", registered={})))
    source = dict(artifact=str(path), sha256="0"*64, spec=spec, kwargs=kwargs)
    m = TaskBudgetController({"0": 0}, {"A": source})
    with pytest.raises(ValueError, match="SHA"):
        m.fit(None, SimpleNamespace(cell="pi05_l10_cache"))
    source["sha256"] = sha(path)
    m.fit(None, SimpleNamespace(cell="pi05_l10_cache"))
    bad = TaskBudgetController({"0": 0, "1": 0}, {"A": source})
    with pytest.raises(ValueError, match="cover"):
        bad.fit(None, SimpleNamespace(cell="pi05_l10_cache"))


def test_standard_log_parser_uses_final_accepted_attempt_and_owner_prices(tmp_path, monkeypatch):
    cell, arm = "groot_l10_50", "r9p1_groot_l10_50_A"
    monkeypatch.setattr(analyze, "CELLS", (cell,))
    monkeypatch.setattr(analyze, "VARIANTS", ("A",))
    (tmp_path / "arms.json").write_text(json.dumps([dict(arm=arm, model="groot")]))
    root = tmp_path / "runs" / arm
    (root / "client").mkdir(parents=True)
    (root / "server_23240").mkdir()
    journal, server = [], []
    for t, i in sorted(analyze.PAIRS):
        uid = f"{arm}:eval:{t}:{i}"
        journal.extend([dict(task_uid=uid, attempt=1, status="failed", success=False, accepted=True),
                        dict(task_uid=uid, attempt=2, status="done", success=True, accepted=True)])
        server.extend([dict(ev="dec", uid=uid, attempt=1, step=0, vision=True, hit=False),
                       dict(ev="dec", uid=uid, attempt=2, step=0, vision=True, hit=False),
                       dict(ev="dec", uid=uid, attempt=2, step=1, vision=False, hit=True)])
    (root / "client/journal.jsonl").write_text("".join(json.dumps(x)+"\n" for x in journal))
    (root / "server_23240/decisions_fixture.jsonl").write_text("".join(json.dumps(x)+"\n" for x in server))
    df = analyze.standard_episode_table(tmp_path)
    assert len(df) == 300 and df.success.eq(1).all()
    assert df.n_dec.eq(2).all() and df.cost.eq(1.).all()
    assert df.n_look.eq(1).all() and df.n_call.eq(1).all()
