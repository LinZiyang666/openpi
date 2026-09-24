"""RIT offline calibration (LOTO): the seams that fail silently, pinned.

GPU-side self-checks (reconstruction parity, orchestrator agreement) live in
the scripts themselves; here every model call is a stub and the questions are
about bookkeeping: which hit is the LOTO winner, which fields reach the score,
what a decision row carries, which frozen identity every stage must prove, and
when a gate, a merge or a report must refuse.
"""

from __future__ import annotations

import json
import math
import pathlib
import pickle
import sys
import threading
import types

import h5py
import numpy as np
import pytest
import torch

from openpi.cache.components.judge import ThresholdJudge
from openpi.cache.components.surface_judge import weighted_chunk_deviation
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID, groot_n15_schedule

import exp.rit_loto.build_loto_table as blt
import exp.rit_loto.emit_verify_arm as eva
import exp.rit_loto.fit_loto as fl
import exp.rit_loto.loto_logger as ll
import exp.rit_loto.verify_closed_loop as vcl
from exp.libero_groot.emit_rit_arms import WARM_TS, load_cost
from exp.rit_pareto.rit_k import PLFitK, predict
from exp.robocasa365 import rit_cost_rc as rc

REPO = pathlib.Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "exp/libero_groot/config/rit/libero_10/template.yaml"
COST = REPO / "exp/libero_groot/config/rit/cost_groot_libero_measured.json"
SCHEDULE = groot_n15_schedule(8)
H, D = 16, 32


# ------------------------------------------------------------------
# Fixtures: a tiny library shaped like the real one
# ------------------------------------------------------------------


def _entry(traj: str, step: int, task: str, rng: np.random.Generator, *, schedule_id=SCHEDULE.schedule_id,
           shape=(H, D)) -> CacheEntry:
    keys = {
        "vision_0": torch.tensor(rng.standard_normal(32768), dtype=torch.float32),
        "vision_1": torch.tensor(rng.standard_normal(32768), dtype=torch.float32),
        "vision_2": torch.zeros(32768),
        "prompt_emb": torch.tensor(rng.standard_normal(2048), dtype=torch.float32),
        "robot_state": torch.tensor(rng.standard_normal(8), dtype=torch.float32),
    }
    chunk = torch.tensor(rng.standard_normal(shape), dtype=torch.float32)
    chunk[:, 7:] = 0.0  # padding dims like the real LIBERO chunks
    inter = {t: chunk + t for t in SCHEDULE.timesteps}
    return CacheEntry(id=f"{traj}:{step}", checkpoint_id=CheckpointID.CP1, query_keys=keys,
                      payload=CachePayload(action_chunk=chunk, intermediates=inter, denoising_num_steps=8, task_key=task,
                                           schedule_id=schedule_id),
                      step_idx=step, trajectory_id=traj)


@pytest.fixture(scope="module")
def tiny_library():
    rng = np.random.default_rng(0)
    entries = []
    for task in ("task A", "task B"):
        for traj in ("ep_a", "ep_b", "ep_c"):
            for step in range(3):
                entries.append(_entry(f"{task[-1]}_{traj}", step, task, rng))
    return entries


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cfg")
    cfg, _ = blt.load_template(str(TEMPLATE), str(tmp / "fake_lib.pkl"), tmp)
    return cfg


@pytest.fixture(scope="module")
def cost():
    return load_cost(COST)


def _query_from(entry: CacheEntry, enabled: list[str], jitter: float = 0.01) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(1)
    return {k: entry.query_keys[k] + jitter * torch.randn(entry.query_keys[k].shape, generator=gen) for k in enabled}


# ------------------------------------------------------------------
# 1 / 1b / 1c  retrieval
# ------------------------------------------------------------------


def test_loto_winner_skips_only_the_query_trajectory():
    hits = [types.SimpleNamespace(id="t1:0", score=0.99), types.SimpleNamespace(id="t1:1", score=0.98),
            types.SimpleNamespace(id="t2:0", score=0.97), types.SimpleNamespace(id="t3:0", score=0.90)]
    traj_of = {"t1:0": "t1", "t1:1": "t1", "t2:0": "t2", "t3:0": "t3"}
    assert blt.loto_winner(hits, "t1", True, traj_of) == ("t2:0", 0.97, 2)
    assert blt.loto_winner(hits, "t1", False, traj_of) == ("t1:0", 0.99, 0)
    assert blt.loto_winner(hits, "t9", True, traj_of) == ("t1:0", 0.99, 0)
    with pytest.raises(RuntimeError):
        blt.loto_winner(hits[:2], "t1", True, traj_of)


def test_post_hoc_exclusion_equals_a_library_without_the_trajectory(cfg, tiny_library):
    enabled, weights = blt.enabled_fields(cfg)
    query_entry = next(e for e in tiny_library if e.trajectory_id == "A_ep_a" and e.step_idx == 1)
    keys = _query_from(query_entry, enabled)
    full = blt.build_storage(cfg, tiny_library)
    hits = blt.search(blt.build_retrieval(cfg, full, weights, len(tiny_library)), keys, 1, "task A")
    assert hits[0].id.startswith("A_ep_a:")
    winner, score, skipped = blt.loto_winner(hits, "A_ep_a", True, {e.id: e.trajectory_id for e in tiny_library})
    assert skipped >= 1
    without = [e for e in tiny_library if e.trajectory_id != "A_ep_a"]
    shrunk = blt.build_storage(cfg, without)
    hits2 = blt.search(blt.build_retrieval(cfg, shrunk, weights, len(without)), keys, 1, "task A")
    assert (hits2[0].id, float(hits2[0].score)) == (winner, score)


def test_enabled_fields_contract_and_disabled_prompt_never_scores(cfg, tiny_library):
    enabled, weights = blt.enabled_fields(cfg)
    assert enabled == ["vision_0", "vision_1", "robot_state"]
    assert weights["prompt_emb"] == 0.0 and weights["vision_2"] == 0.0 and weights["vision_0"] > 0
    storage = blt.build_storage(cfg, tiny_library)
    strategy = blt.build_retrieval(cfg, storage, weights, len(tiny_library))
    entry = next(e for e in tiny_library if e.trajectory_id == "B_ep_b" and e.step_idx == 2)
    keys = _query_from(entry, enabled)
    base = blt.search(strategy, keys, 2, "task B")
    perm = blt.search(strategy, blt.with_permuted_prompt(keys, 7), 2, "task B")
    assert [(h.id, float(h.score)) for h in perm] == [(h.id, float(h.score)) for h in base]
    partial = {k: v for k, v in weights.items() if k != "prompt_emb"}
    leaky = blt.build_retrieval(cfg, storage, partial, len(tiny_library))
    try:
        leaked = blt.search(leaky, blt.with_permuted_prompt(keys, 7), 2, "task B")
    except ValueError:
        leaked = None  # the disabled field became active and had no normalizer: the leak surfaces as a crash
    if leaked is not None:
        assert float(leaked[0].score) != float(base[0].score)


def _artifact(entries):
    return {"entries": entries, "schedule_id": SCHEDULE.schedule_id, "key_builder_type": "x"}


def test_load_library_validates_every_payloads_schedule_and_shape(tmp_path, tiny_library):
    pkl = tmp_path / "lib.pkl"
    pkl.write_bytes(pickle.dumps(_artifact(tiny_library)))
    lib = blt.load_library(pkl, expected_schedule=SCHEDULE, warm_ts=(0.75, 0.5))
    assert len(lib.trajectory_ids) == 6 and lib.per_task_entries == {"task A": 9, "task B": 9}
    (tmp_path / "bad_art.pkl").write_bytes(pickle.dumps({**_artifact(tiny_library), "schedule_id": "groot_n15_k4_v1"}))
    with pytest.raises(SystemExit):
        blt.load_library(tmp_path / "bad_art.pkl", expected_schedule=SCHEDULE, warm_ts=(0.75,))
    with pytest.raises(SystemExit):
        blt.load_library(pkl, expected_schedule=SCHEDULE, warm_ts=(0.3,))
    rng = np.random.default_rng(1)
    # B4: artifact says k8 but one payload is stamped pi05_v1 -> rejected before any snapshot is resumed
    bad_payload = [_entry("A_x", 0, "task A", rng, schedule_id="pi05_v1")] + tiny_library[1:]
    (tmp_path / "bad_pl.pkl").write_bytes(pickle.dumps(_artifact(bad_payload)))
    with pytest.raises(SystemExit, match="schedule"):
        blt.load_library(tmp_path / "bad_pl.pkl", expected_schedule=SCHEDULE, warm_ts=(0.75,))
    bad_shape = [_entry("A_y", 0, "task A", rng, shape=(1, 32))] + tiny_library[1:]
    (tmp_path / "bad_shape.pkl").write_bytes(pickle.dumps(_artifact(bad_shape)))
    with pytest.raises(SystemExit, match="finite"):
        blt.load_library(tmp_path / "bad_shape.pkl", expected_schedule=SCHEDULE, warm_ts=(0.75,))


# ------------------------------------------------------------------
# 2 / 2b / 2c  labels, parity gate, session contract, identities
# ------------------------------------------------------------------


class _StubRunner:
    """Refuses every forward outside ``session()``; produces recognisable chunks."""

    def __init__(self):
        self.in_session = False
        self.calls: list[tuple] = []

    def session(self):
        runner = self

        class _Ctx:
            def __enter__(self_inner):
                runner.in_session = True

            def __exit__(self_inner, *exc):
                runner.in_session = False

        return _Ctx()

    def _require(self):
        if not self.in_session:
            raise RuntimeError("must run inside session()")

    def run_stage1(self, x):
        self._require()
        self.calls.append(("stage1",))
        return ("stage1", x)

    def run_stage2_llm(self, stage1):
        self._require()
        self.calls.append(("stage2",))
        return ("stage2", stage1)

    def run_stage3(self, stage2, *, noise=None):
        self._require()
        self.calls.append(("stage3", None if noise is None else float(noise.sum())))
        return types.SimpleNamespace(action_pred=noise * 2.0)

    def run_stage3_from(self, stage2, start_x, t, *, schedule, capture_first_step=False):
        assert not capture_first_step
        self._require()
        self.calls.append(("stage3_from", float(t)))
        return types.SimpleNamespace(action_pred=start_x[None] * 10.0 + t)

    def live_schedule(self):
        return SCHEDULE


class _StubTemplates:
    def __init__(self, runner):
        self.runner = runner
        self.built: list[str] = []

    def get(self, task):
        if task not in self.built:
            self.runner.run_stage1(None)  # a template miss runs stage 1, as the real cache does
            self.built.append(task)
        return f"template:{task}"


def _group(tmp_path, name="g", *, clean=None):
    f = h5py.File(tmp_path / f"{name}.h5", "w")
    g = f.create_group("step_0000")
    rng = np.random.default_rng(3)
    g.create_dataset("clean_action", data=(rng.standard_normal((H, D)) if clean is None else clean).astype(np.float32))
    for i in range(8):
        g.create_dataset(f"noise_action_{i}", data=np.full((H, D), 0.1 * i, np.float32))
    return f, g


@pytest.fixture
def stub_model(monkeypatch):
    runner = _StubRunner()
    monkeypatch.setattr(blt, "reconstruct_stage1", lambda template, group: ("stage1", template))
    return runner, _StubTemplates(runner)


def _w_mask():
    w = torch.ones(D)
    mask = torch.zeros(D, dtype=torch.bool)
    mask[:7] = True
    w[7:] = 0.0
    return w, mask


def test_label_row_uses_the_shared_w_and_executed_window(tmp_path, stub_model, tiny_library):
    runner, templates = stub_model
    w, mask = _w_mask()
    f, g = _group(tmp_path)
    payload = tiny_library[0].payload
    row = blt.label_row(runner, templates, "task A", g, payload, (0.75, 0.5), SCHEDULE, w, mask, 5)
    ref = torch.from_numpy(np.asarray(g["clean_action"], dtype=np.float32))
    assert row["y_full"] == weighted_chunk_deviation(payload.action_chunk, ref, w, mask, 5)
    assert row["y_rem2"] == weighted_chunk_deviation(payload.intermediates[0.75] * 10.0 + 0.75, ref, w, mask, 5)
    assert row["y_rem4"] == weighted_chunk_deviation(payload.intermediates[0.5] * 10.0 + 0.5, ref, w, mask, 5)
    assert ("stage3_from", 0.75) in runner.calls and ("stage3_from", 0.5) in runner.calls
    assert not runner.in_session
    f.close()


def test_template_miss_and_forwards_stay_inside_session(tmp_path, stub_model):
    runner, templates = stub_model
    f, g = _group(tmp_path)
    with pytest.raises(RuntimeError):
        blt.stage2_in_session(runner, templates, "new task", g)
    with runner.session():
        blt.stage2_in_session(runner, templates, "new task", g)
    assert templates.built == ["new task"]
    w, mask = _w_mask()
    out = blt.parity_row(runner, templates, "another", g, SCHEDULE, w, mask, 5)
    assert templates.built == ["new task", "another"]
    assert set(out) == {f"parity_{n}_maxabs" for n in blt.PARITY_TIERS} | {f"parity_D_{n}" for n in blt.PARITY_TIERS}
    f.close()


def _identity():
    ident = {k: f"v_{k}" for k in blt.IDENTITY_KEYS}
    ident["code_sha256"] = {"exp/rit_loto/build_loto_table.py": "abc", "src/openpi/cache/groot/staged.py": "def"}
    return ident


def _parity_rows(n_per_task, err_dim0):
    rows = []
    for t in range(blt.NUM_TASKS):
        for i in range(n_per_task):
            rows.append({"task_id": t, "trajectory_id": f"e{t}_{i}", "decision_id": i,
                         **{f"parity_D_{k}": err_dim0 for k in blt.PARITY_TIERS},
                         **{f"parity_{k}_maxabs": 0.01 for k in blt.PARITY_TIERS}})
    return rows


def _floor_record(median, identity, n_active=2):
    return {"identity": json.loads(json.dumps(identity)), "d_ref1_ref2": {"median": median, "n": 500},
            "sample": {"per_task": 50, "n_rows": 500}, "n_active_dims": n_active}


def test_parity_gate_scales_with_w_and_refuses_thin_or_mismatched_samples():
    ident = _identity()
    w = torch.tensor([1.0, 1.0])
    mask = torch.tensor([True, True])
    a = torch.zeros(5, 2)
    b = torch.zeros(5, 2)
    b[:, 0] = 0.01  # parity error only on dim 0
    c = torch.zeros(5, 2)
    c[:, 1] = 1.0  # noise floor only on dim 1
    parity_d = weighted_chunk_deviation(a, b, w, mask, 5)
    floor = weighted_chunk_deviation(a, c, w, mask, 5)
    gate = blt.parity_gate(_parity_rows(20, parity_d), _floor_record(floor, ident), ident)
    assert gate["status"] == "PASS", gate["reasons"]
    w100 = torch.tensor([100.0, 1.0])
    parity_d100 = weighted_chunk_deviation(a, b, w100, mask, 5)
    assert math.isclose(parity_d100, 100 * parity_d, rel_tol=1e-5)
    assert blt.maxabs_active(a, b, mask, 5) == blt.maxabs_active(a, b, mask, 5)
    gate100 = blt.parity_gate(_parity_rows(20, parity_d100), _floor_record(floor, ident), ident)
    assert gate100["status"] == "FAIL" and any("p90" in r for r in gate100["reasons"])
    assert blt.parity_gate(_parity_rows(19, parity_d), _floor_record(floor, ident), ident)["status"] == "FAIL"
    rows = _parity_rows(20, parity_d)
    assert blt.parity_gate([r for r in rows if r["task_id"] != 3], _floor_record(floor, ident), ident)["status"] == "FAIL"
    assert blt.parity_gate(rows, _floor_record(0.0, ident), ident)["status"] == "FAIL"
    assert blt.parity_gate(_parity_rows(20, 0.0), _floor_record(0.0, ident), ident)["status"] == "PASS"
    bad = [dict(r, parity_D_full=float("nan")) for r in rows]
    assert blt.parity_gate(bad, _floor_record(floor, ident), ident)["status"] == "FAIL"
    other = dict(ident, library_sha256="other")
    assert blt.parity_gate(rows, _floor_record(floor, other), ident)["status"] == "FAIL"
    # B3: a changed forward file is a different identity even when everything else matches
    changed = json.loads(json.dumps(ident))
    changed["code_sha256"]["src/openpi/cache/groot/staged.py"] = "changed"
    assert blt.parity_gate(rows, _floor_record(floor, changed), ident)["status"] == "FAIL"
    with pytest.raises(ValueError):
        blt.parity_gate(rows, _floor_record(-1.0, ident), ident)
    with pytest.raises(ValueError):
        blt.parity_gate(rows, _floor_record(floor, ident, n_active=0), ident)
    thin_floor = _floor_record(floor, ident)
    thin_floor["sample"] = {"per_task": 2, "n_rows": 20}
    thin_floor["d_ref1_ref2"]["n"] = 20
    assert blt.parity_gate(rows, thin_floor, ident)["status"] == "FAIL"
    assert blt.parity_gate(rows[:-1] + rows[:1], _floor_record(floor, ident), ident)["status"] == "FAIL"
    assert blt.parity_gate(_parity_rows(20, -1), _floor_record(floor, ident), ident)["status"] == "FAIL"


def test_require_pass_gate_refuses_missing_failed_foreign_or_stale_code_gates(tmp_path):
    ident = _identity()
    with pytest.raises(SystemExit):
        blt.require_pass_gate(tmp_path / "none.json", ident)
    p = tmp_path / "gate.json"
    p.write_text(json.dumps({"status": "FAIL", "reasons": ["x"], "identity": ident}))
    with pytest.raises(SystemExit):
        blt.require_pass_gate(p, ident)
    p.write_text(json.dumps({"status": "PASS", "reasons": [], "identity": dict(ident, h_exec="9")}))
    with pytest.raises(SystemExit):
        blt.require_pass_gate(p, ident)
    stale = json.loads(json.dumps(ident))
    stale["code_sha256"]["exp/rit_loto/build_loto_table.py"] = "older"
    p.write_text(json.dumps({"status": "PASS", "reasons": [], "identity": stale}))
    with pytest.raises(SystemExit, match="code_sha256"):
        blt.require_pass_gate(p, ident)
    no_code = {k: v for k, v in ident.items() if k != "code_sha256"}
    p.write_text(json.dumps({"status": "PASS", "reasons": [], "identity": no_code}))
    with pytest.raises(SystemExit, match="code_sha256"):
        blt.require_pass_gate(p, ident)
    p.write_text(json.dumps({"status": "unknown", "identity": ident}))
    with pytest.raises(SystemExit):
        blt.require_pass_gate(p, ident)
    p.write_text(json.dumps({"status": "PASS", "reasons": [], "identity": ident}))
    assert blt.require_pass_gate(p, ident)["status"] == "PASS"


def test_checkpoint_identity_hashes_content_and_caches_by_listing(tmp_path):
    a, b = tmp_path / "ckpt_a", tmp_path / "ckpt_b"
    for d, byte in ((a, b"A"), (b, b"B")):
        d.mkdir()
        (d / "config.json").write_text("{}")
        (d / "model.safetensors").write_bytes(byte * 4096)  # same name, same size, different bytes
    cache = tmp_path / "cache"
    ia = blt.checkpoint_identity(a, cache_dir=cache)
    ib = blt.checkpoint_identity(b, cache_dir=cache)
    assert ia["sha256"] != ib["sha256"] and not ia["cache_hit"]
    again = blt.checkpoint_identity(a, cache_dir=cache)
    assert again["sha256"] == ia["sha256"] and again["cache_hit"]
    (a / "model.safetensors").write_bytes(b"C" * 4096)
    import os
    os.utime(a / "model.safetensors", ns=(1, 1))  # a different mtime invalidates the cache
    changed = blt.checkpoint_identity(a, cache_dir=cache)
    assert changed["sha256"] != ia["sha256"] and not changed["cache_hit"]
    with pytest.raises(SystemExit):
        blt.checkpoint_identity(tmp_path / "missing")


def test_code_sha256_refuses_a_missing_file():
    with pytest.raises(SystemExit):
        blt.code_sha256(("exp/rit_loto/does_not_exist.py",))
    assert set(blt.code_sha256()) == set(blt.CODE_FILES)


def test_derive_seed_is_stable_and_order_free():
    a = blt.derive_seed(1, {"suite": "s", "trajectory_id": "t", "decision_id": 3, "tag": "x"})
    b = blt.derive_seed(1, {"tag": "x", "decision_id": 3, "trajectory_id": "t", "suite": "s"})
    assert a == b and a != blt.derive_seed(2, {"suite": "s", "trajectory_id": "t", "decision_id": 3, "tag": "x"})
    assert torch.equal(blt.make_noise(a, (1, 2, 2)), blt.make_noise(a, (1, 2, 2)))


# ------------------------------------------------------------------
# 3 / 4 / 4b  fits, audit, curves, bootstrap
# ------------------------------------------------------------------


def _synthetic_rows(n=400, seed=0, n_eps=20):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        s = float(rng.uniform(0.6, 1.0))
        base = 8.0 - 3.0 * s + 0.2 * rng.standard_normal()
        rows.append({"s": s, "y_full": base + 0.5, "y_rem2": base + 0.2, "y_rem4": base,
                     "task": f"task {i % 2}", "episode_id": i % n_eps, "episode_success": True, "in_library": i % 5 == 0})
    return rows


def _fit_k(rows, k, cost, alpha=0.05):
    from exp.robocasa365 import emit_rit_rc as er
    s = np.array([r["s"] for r in rows])
    return er.fit_ladders(rows, cost, list(WARM_TS), [k], alpha, ir_sample=s)[k]


def test_fit_source_roundtrip_and_measured_cost_tiers(cost):
    rows = _synthetic_rows()
    block = fl.fit_source(rows, cost, source="loto_all", input_sha256="abc", ks=(1, 2, 3))
    assert not block.get("fit_unavailable") and set(block["fits"]) == {"1", "2", "3"}
    rec = json.loads(json.dumps(block["fits"]["2"]))
    fit = fl.deserialize_fit(rec, cost, WARM_TS)
    live = _fit_k(rows, 2, cost)["fit"]
    grid = np.linspace(0.6, 1.0, 50)
    for t in fit.tiers:
        assert np.allclose(predict(fit, grid, t.name), predict(live, grid, t.name))
        assert math.isclose(t.cost_ms, rc.tier_cost(cost, t.hit_type, t.start_t))
    assert rc.cuts_for(fit, 4.0) == rc.cuts_for(live, 4.0)
    assert fl.fit_source([], cost, source="x", input_sha256="")["fit_unavailable"]
    assert fl.fit_source(rows[:10], cost, source="x", input_sha256="")["fit_unavailable"]


def _arm_record_from(rows, cost, *, table_sha, template_sha, alpha=None, fit_alpha=0.05, ks=(1, 2, 3)):
    fits = {k: _fit_k(rows, k, cost, alpha=fit_alpha) for k in ks}
    arms = {}
    for k in ks:
        fit = fits[k]["fit"]
        q_min = min(float(np.min(v)) for v in fit.q.values())
        for j, delta in enumerate((q_min + 0.05, q_min + 1.0)):
            cuts = rc.cuts_for(fit, delta)
            arms[f"k{k}_arm{j}"] = {"rule": "rit", "k": k, "delta": delta,
                                   "cuts": [None if not math.isfinite(c) else c for c in cuts]}
    rec = {"shadow_sha256": table_sha, "template_sha256": template_sha, "warm_ts": list(WARM_TS),
           "fits": {str(k): {"knots": fits[k]["knots"], "tiers": [t.name for t in fits[k]["tiers"]]} for k in fits},
           "arms": arms, "gate_theta": 0.99}
    if alpha is not None:
        rec["alpha"] = alpha
    return rec


def test_refit_shadow_audit_accepts_the_deployed_record_and_rejects_tampering(cost):
    rows = _synthetic_rows(seed=1)
    rec = _arm_record_from(rows, cost, table_sha="T", template_sha="P")
    assert any(c is None for a in rec["arms"].values() for c in a["cuts"])
    out = fl.refit_shadow_and_check(rows, cost, rec, table_sha="T", template_sha="P")
    assert out["alpha_source"] == "legacy_reconstruction" and out["n_arms_checked"] == 6
    rec_alpha = _arm_record_from(rows, cost, table_sha="T", template_sha="P", alpha=0.05)
    assert fl.refit_shadow_and_check(rows, cost, rec_alpha, table_sha="T", template_sha="P")["alpha_source"] == "record"
    with pytest.raises(SystemExit):
        fl.refit_shadow_and_check(rows, cost, rec, table_sha="other", template_sha="P")
    bad = json.loads(json.dumps(rec))
    arm = next(a for a in bad["arms"].values() if any(c is not None for c in a["cuts"]))
    idx = next(i for i, c in enumerate(arm["cuts"]) if c is not None)
    arm["cuts"][idx] += 1e-6
    with pytest.raises(SystemExit):
        fl.refit_shadow_and_check(rows, cost, bad, table_sha="T", template_sha="P")
    bad2 = json.loads(json.dumps(rec))
    arm2 = next(a for a in bad2["arms"].values() if any(c is None for c in a["cuts"]))
    arm2["cuts"][arm2["cuts"].index(None)] = 0.5
    with pytest.raises(SystemExit):
        fl.refit_shadow_and_check(rows, cost, bad2, table_sha="T", template_sha="P")
    # B7: a self-consistent record at alpha=0.20 reproduces its own knots and cuts, and is still refused
    rec20 = _arm_record_from(rows, cost, table_sha="T", template_sha="P", alpha=0.20, fit_alpha=0.20)
    with pytest.raises(SystemExit, match="alpha"):
        fl.refit_shadow_and_check(rows, cost, rec20, table_sha="T", template_sha="P")
    rec_k = _arm_record_from(rows, cost, table_sha="T", template_sha="P", ks=(1, 2))
    with pytest.raises(SystemExit, match="K set"):
        fl.refit_shadow_and_check(rows, cost, rec_k, table_sha="T", template_sha="P")
    with pytest.raises(ValueError):
        fl._cut_matches(float("nan"), float("nan"))


def _pl(knots, q):
    tiers = rc.ladder(load_cost(COST), [0.75])
    return PLFitK(knots=np.asarray(knots, float), q={t.name: np.asarray(v, float) for t, v in zip(tiers, q)},
                  tiers=tiers, eps_total=0.02, n_seg_req=2, n_seg=len(knots) - 1, alpha=0.05)


def test_curve_max_diff_is_exact_on_the_knot_union_and_marginals_match_on_identical_rows():
    a = _pl([0.0, 0.5, 1.0], [[3.0, 2.0, 1.0], [2.0, 1.5, 0.5]])
    b = _pl([0.0, 0.25, 1.0], [[3.0, 2.0, 1.0], [2.0, 1.5, 0.5]])
    d = fl.curve_max_diff(a, b, "full", 0.0, 1.0)
    assert math.isclose(d["max_abs"], 0.5) and math.isclose(d["at_s"], 0.25)
    assert fl.curve_max_diff(a, a, "full", 0.0, 1.0)["max_abs"] == 0.0
    s = np.linspace(0, 1, 100)
    assert fl.score_marginal(s, s)["ks_statistic"] == 0.0
    assert fl.common_support(a, _pl([2.0, 3.0, 4.0], [[1, 1, 1], [1, 1, 1]])) is None


def test_episode_bootstrap_band_is_reproducible_and_records_failures(cost, monkeypatch):
    rows = _synthetic_rows(seed=2, n=600, n_eps=30)
    grid = [0.62, 0.7, 0.8, 0.9, 0.98]
    b1 = fl.episode_bootstrap_band(rows, cost, WARM_TS, 2, 0.05, grid, n_boot=4, seed=5)
    b2 = fl.episode_bootstrap_band(rows, cost, WARM_TS, 2, 0.05, grid, n_boot=4, seed=5)
    assert b1["tiers"] == b2["tiers"] and b1["n_ok"] == 4
    assert all(v <= 4 for v in b1["tiers"]["full"]["valid_per_point"])
    real = fl.er.fit_ladders

    def flaky(sample, *a, **k):
        if flaky.calls == 1:
            flaky.calls += 1
            raise SystemExit("knot ladder exhausted (simulated)")
        flaky.calls += 1
        return real(sample, *a, **k)

    flaky.calls = 0
    monkeypatch.setattr(fl.er, "fit_ladders", flaky)
    b3 = fl.episode_bootstrap_band(rows, cost, WARM_TS, 2, 0.05, grid, n_boot=3, seed=5)
    assert b3["n_failures"] == 1 and b3["n_ok"] == 2 and "simulated" in b3["failures"][0]["reason"]
    assert all(v is None for v in b3["tiers"]["full"]["lo"])


# ------------------------------------------------------------------
# 5  logger
# ------------------------------------------------------------------


def _stage1(n_text=5, state_dim=8):
    n = 2 * 256 + n_text
    mask = torch.zeros(1, n, dtype=torch.bool)
    mask[0, :256] = True
    mask[0, 256 + n_text: 512 + n_text] = True
    embeds = torch.arange(n, dtype=torch.float32)[None, :, None].expand(1, n, 4).clone()
    state = torch.zeros(1, 1, 64)
    state[0, -1, :state_dim] = torch.arange(state_dim, dtype=torch.float32)
    smask = torch.zeros(1, 1, 64, dtype=torch.bool)
    smask[0, -1, :state_dim] = True
    return types.SimpleNamespace(input_embeds=embeds, attention_mask=torch.ones(1, n, dtype=torch.bool),
                                 image_token_mask=mask, action_inputs={"state": state, "state_mask": smask},
                                 state=state, state_mask=smask)


class _StubOrch:
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.broadcasts = []
        self.calls = []

    def has_checkpoint(self, cp):
        return False

    def on_task_begin(self, *a, **k):
        self.calls.append("task_begin")

    def on_task_end(self, *a, **k):
        self.calls.append("task_end")

    def on_episode_start(self, **k):
        self.calls.append(("ep_start", k.get("episode_id")))

    def on_episode_end(self, *a, **k):
        self.calls.append("ep_end")

    def clear(self):
        self.calls.append("clear")

    def buffer_for_write(self, *a, **k):
        pass

    def check(self, cp, *, stage1):
        v = self.verdicts.pop(0)
        if isinstance(v, Exception):
            raise v
        return v

    def broadcast_action(self, chunk):
        self.broadcasts.append(chunk)


class _LoggerRunner(_StubRunner):
    def run_stage1(self, x):
        self._require()
        return _stage1()

    def run_stage2(self, stage1):
        self._require()
        return types.SimpleNamespace(action_pred=torch.full((1, H, D), 7.0))


def _verdict(hit, entry_id=None, start_t=None, score=None, payload=None):
    from openpi.cache.components.judge import HitType
    from openpi.cache.orchestrator import CheckResult
    return CheckResult(hit_type=HitType[hit], payload=payload, score=score, entry_id=entry_id, start_t=start_t, searched=True)


def _policy():
    return types.SimpleNamespace(apply_transforms=lambda obs: obs,
                                 unapply_transforms=lambda d: {"action": d["action"].numpy()})


def test_logger_writes_the_executed_chunk_for_every_tier_and_isolates_failures(tmp_path, tiny_library):
    payload = tiny_library[0].payload
    verdicts = [
        _verdict("FULL_HIT", "e:0", None, 0.99, payload),
        _verdict("WARM_START", "e:1", 0.75, 0.98, payload),
        RuntimeError("boom"),
        _verdict("MISS", None, None, 0.5, None),
    ]
    orch = _StubOrch(verdicts)
    logger = ll.GrootLotoLogger(_policy(), _LoggerRunner(), orchestrator=orch, timer=None, out_dir=str(tmp_path / "logs"),
                                experiment="groot_libero", run_tag="verify", connection_id="c1",
                                identity_attrs={"loto_arm_yaml_sha256": "armsha"}, h_exec=5)
    obs = {"o": np.zeros(3)}
    logger.on_task_begin()
    logger.on_episode_start(experiment="groot_libero", task="task A", episode_id=11,
                            extra_metadata={"task_id": 0, "orig_init_state_idx": 42})
    r0 = logger.get_action(obs)
    assert r0["__hit_meta__"]["hit_type"] == "FULL_HIT"
    assert logger._runner.last_stage1 is None and logger._orch.last_broadcast is None  # N2: nothing outlives the call
    logger.get_action(obs)
    with pytest.raises(RuntimeError):
        logger.get_action(obs)
    assert logger._orch.last_broadcast is None
    logger.get_action(obs)
    # N2: mutating the broadcast tensor after the call must not change what was buffered
    orch.broadcasts[2].fill_(-1.0)
    logger.on_episode_end(True)
    logger.on_task_end()
    h5s = list((tmp_path / "logs" / "verify" / "conn_c1").rglob("*.h5"))
    assert len(h5s) == 1
    with h5py.File(h5s[0], "r") as f:
        assert f.attrs["num_steps"] == 3 and f.attrs["run_tag"] == "verify" and f.attrs["connection_id"] == "c1"
        assert f.attrs["task_id"] == 0 and f.attrs["orig_init_state_idx"] == 42 and f.attrs["loto_arm_yaml_sha256"] == "armsha"
        assert f.attrs["h_exec"] == 5 and f.attrs["denoise_schedule_id"] == SCHEDULE.schedule_id
        assert np.array_equal(f["step_0000"]["clean_action"][()], payload.action_chunk.numpy())
        assert np.allclose(f["step_0002"]["clean_action"][()], 7.0)  # the MISS chunk as executed, not the later mutation
        g = f["step_0000"]
        assert "noise_action_0" not in g and g["vision_0"].shape == (256, 4) and g["robot_state"].shape == (8,)
    side = blt.read_jsonl(tmp_path / "logs" / "verify" / "conn_c1" / "groot_libero" / ll.SIDECAR_NAME)
    assert [r["decision_id"] for r in side] == [0, 1, 2]
    assert [r["hit_type"] for r in side] == ["FULL_HIT", "WARM_START", "MISS"]
    assert side[1]["start_t"] == 0.75 and side[1]["winner_id"] == "e:1" and side[0]["s"] == 0.99
    assert all(r["episode_success"] and r["task_id"] == 0 and r["orig_init_state_idx"] == 42 for r in side)
    assert side[0]["control_step_idx"] == 0 and side[2]["control_step_idx"] == 10
    other = ll.GrootLotoLogger(_policy(), _LoggerRunner(), orchestrator=_StubOrch([]), timer=None,
                               out_dir=str(tmp_path / "logs"), experiment="groot_libero", run_tag="verify", connection_id="c2")
    assert other.root != logger.root and other.root.is_dir()
    with pytest.raises(ValueError):
        ll.GrootLotoLogger(_policy(), _LoggerRunner(), orchestrator=None, timer=None, out_dir=str(tmp_path), experiment="e", run_tag="t")


# ------------------------------------------------------------------
# 6  pools, arm, frozen record
# ------------------------------------------------------------------


def _fake_apool(tmp_path, names):
    d = tmp_path / "apool"
    d.mkdir()
    for i, name in enumerate(names):
        torch.save(torch.arange(50 * 4, dtype=torch.float32).reshape(50, 4) + i, d / f"{name}.init")
    return d


def _shadow_manifest(tmp_path, names):
    assignment = {str(t): {"task_name": names[t], "fit": list(range(0, 50, 10)), "cal": list(range(1, 50, 5))} for t in range(10)}
    man = tmp_path / "shadow_manifest.json"
    man.write_text(json.dumps({"suite": "libero_10", "assignment": assignment}))
    return man, assignment


def test_sample_pools_are_disjoint_from_the_cohort_and_filters_carry_official_indices(tmp_path):
    names = [f"TASK_{t}" for t in range(10)]
    apool = _fake_apool(tmp_path, names)
    man, assignment = _shadow_manifest(tmp_path, names)
    out = eva.sample_pools("libero_10", apool, man, 5, 1, 7, tmp_path / "pools")
    out2 = eva.sample_pools("libero_10", apool, man, 5, 1, 7, tmp_path / "pools2")
    assert out["verify"] == out2["verify"] and out["smoke"] == out2["smoke"]
    for t in map(str, range(10)):
        used = set(assignment[t]["fit"]) | set(assignment[t]["cal"])
        assert len(out["verify"][t]) == 5 and len(out["smoke"][t]) == 1
        assert not (set(out["verify"][t]) & used) and not (set(out["smoke"][t]) & used)
        assert not (set(out["verify"][t]) & set(out["smoke"][t]))
    flt = json.loads(pathlib.Path(out["filters"]["verify"]).read_text())
    for t in range(10):
        rows = [e for e in flt if e["task_id"] == t]
        assert [e["subset_init_state_idx"] for e in rows] == [0, 1, 2, 3, 4]
        assert [e["orig_init_state_idx"] for e in rows] == sorted(out["verify"][str(t)])
        pool = torch.load(tmp_path / "pools" / "verify_pool" / f"{names[t]}.init", weights_only=False)
        assert pool.shape[0] == 5
    with pytest.raises(SystemExit):
        eva.sample_pools("libero_spatial", apool, man, 5, 1, 7, tmp_path / "pools3")


def test_formal_table_rejects_shards_and_inconsistent_episode_identity():
    rows = [{"suite": "libero_10", "trajectory_id": f"episode_{t}_{i}", "decision_id": 0,
             "task_id": t, "orig_init_state_idx": i, "in_library": i < 5} for t in range(10) for i in range(50)]
    record = {"protocol": "rit_loto_table_v1", "smoke": False,
              "identity": {"suite": "libero_10", "corpus_files": 500}, "stats": {"episodes": 500, "rows": 500}}
    fl.validate_table_record(rows, record)
    with pytest.raises(SystemExit, match="shard"):
        fl.validate_table_record(rows[:50], dict(record, stats={"episodes": 50, "rows": 50}))
    with pytest.raises(SystemExit, match="duplicate"):
        fl.validate_table_record(rows[:-1] + rows[:1], record)
    wrong = [dict(r) for r in rows]
    wrong[0]["orig_init_state_idx"] = 49
    with pytest.raises(SystemExit, match="repeats an episode"):
        fl.validate_table_record(wrong, record)


def test_fit_cli_checks_actual_code_and_fixed_k_before_reading_data(tmp_path, monkeypatch):
    record = tmp_path / "record.json"
    record.write_text(json.dumps({"identity": {"code_sha256": {"source": "stale"}}}))
    argv = ["fit_loto", "--suite", "libero_10", "--loto-record", str(record), "--out-dir", str(tmp_path / "out")]
    for flag in ("--loto-table", "--parity-gate", "--shadow-rows", "--arm-record", "--template-yaml", "--cost"):
        argv += [flag, "must-not-open"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="code identity"):
        fl.main()
    monkeypatch.setattr(sys, "argv", argv + ["--ks", "1,2"])
    with pytest.raises(SystemExit, match="K="):
        fl.main()


def _fits_json(tmp_path, cost, *, template=TEMPLATE, alpha=0.05, identity_over=None):
    rows = _synthetic_rows(n=500, seed=3)
    block = fl.fit_source(rows, cost, source="loto_all", input_sha256="tbl", ks=(1, 2, 3), alpha=alpha)
    identity = {"suite": "libero_10", "library_sha256": "LIB", "template_sha256": blt.sha256_file(template),
                "checkpoint_identity_sha256": "CKPT", "weights_sha256": "W", "h_exec": 5,
                "schedule_id": SCHEDULE.schedule_id, "corpus_manifest_sha256": "CORP",
                "code_sha256": blt.code_sha256(), **(identity_over or {})}
    fits = {"protocol": "rit_loto_fits_v1", "suite": "libero_10", "identity": identity, "loto_all": block,
            "cost": fl._cost_to_json(cost), "parity_gate_status": "PASS"}
    p = tmp_path / "fits.json"
    p.write_text(json.dumps(fits))
    return p


def _test_pool(suite="libero_10"):
    verify = {str(t): list(range(15, 20)) for t in range(10)}
    verify.update({"0": [3, 9, 15, 16, 17], "1": [4, 6, 15, 16, 17]})
    return {"suite": suite, "n_episodes": {"verify": 50, "smoke": 10}, "per_task": {"verify": 5, "smoke": 1},
            "verify": verify, "smoke": {str(t): [20] for t in range(10)},
            "exclude": {str(t): list(range(21, 36)) for t in range(10)}}


def _pool_manifest(tmp_path, suite="libero_10"):
    p = tmp_path / "verify_pool_manifest.json"
    p.write_text(json.dumps(_test_pool(suite)))
    return p


def test_emit_arm_writes_a_frozen_record_bound_to_fits_template_and_pool(tmp_path, cost):
    fits = _fits_json(tmp_path, cost)
    pool = _pool_manifest(tmp_path)
    rec = eva.emit_arm("libero_10", "loto_all", fits, cost, 70.0, TEMPLATE, tmp_path / "cfg", pool_manifest=pool)
    assert rec["protocol"] == eva.FROZEN_PROTOCOL and rec["arm"]["name"] == "loto_k2_ir70"
    assert rec["fits_sha256"] == blt.sha256_file(fits) and rec["pool_manifest_sha256"] == blt.sha256_file(pool)
    assert rec["identity"]["library_sha256"] == "LIB" and rec["alpha"] == 0.05 and rec["run_tags"] == eva.RUN_TAGS
    assert set(rec["code_sha256"]) == set(blt.CODE_FILES)
    assert (tmp_path / "cfg" / "loto_k2_ir70.yaml").is_file() and rec["arm"]["yaml_sha256"] == blt.sha256_file(tmp_path / "cfg" / "loto_k2_ir70.yaml")
    frozen = tmp_path / "frozen_run.json"
    frozen.write_text(json.dumps(rec))
    loaded, sha = eva.load_frozen_record(frozen)
    assert loaded["arm"]["name"] == "loto_k2_ir70" and sha == blt.sha256_file(frozen)
    stale = json.loads(frozen.read_text())
    stale["identity"]["code_sha256"][blt.CODE_FILES[0]] = "older-source"
    frozen.write_text(json.dumps(stale))
    with pytest.raises(SystemExit, match="code identity"):
        eva.load_frozen_record(frozen)
    # B1: a template that is not the one the fits were built on is refused
    other_tpl = tmp_path / "other_template.yaml"
    other_tpl.write_text(TEMPLATE.read_text() + "\n# touched\n")
    with pytest.raises(SystemExit, match="template"):
        eva.emit_arm("libero_10", "loto_all", fits, cost, 70.0, other_tpl, tmp_path / "cfg2", pool_manifest=pool)
    # B7: a fit at another alpha never becomes the deployed arm
    fits20 = _fits_json(tmp_path / "a20", cost, alpha=0.20) if (tmp_path / "a20").mkdir() is None else None
    with pytest.raises(SystemExit, match="alpha"):
        eva.emit_arm("libero_10", "loto_all", fits20, cost, 70.0, TEMPLATE, tmp_path / "cfg3", pool_manifest=pool)
    with pytest.raises(SystemExit, match="pool manifest"):
        eva.emit_arm("libero_10", "loto_all", fits, cost, 70.0, TEMPLATE, tmp_path / "cfg4", pool_manifest=_pool_manifest(tmp_path / "sp", "libero_spatial") if (tmp_path / "sp").mkdir() is None else None)
    with pytest.raises(SystemExit):
        eva.load_frozen_record(tmp_path / "none.json")
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({k: v for k, v in rec.items() if k != "arm"}))
    with pytest.raises(SystemExit, match="no emitted arm"):
        eva.load_frozen_record(incomplete)


# ------------------------------------------------------------------
# 6b  merge
# ------------------------------------------------------------------


def _frozen(tmp_path):
    rec = {"protocol": eva.FROZEN_PROTOCOL, "suite": "libero_10", "fit_source": "loto_all", "k": 2, "alpha": 0.05,
           "fits_sha256": "FITS", "template_sha256": "TPL", "pool_manifest_sha256": "POOL",
           "pool_counts": {"verify": 50, "smoke": 10}, "pool_per_task": {"verify": 5, "smoke": 1},
           "pool": _test_pool(), "target_ir": 70.0, "verify_protocol": dict(eva.VERIFY_PROTOCOL), "compile_stage1": False,
           "cost": fl._cost_to_json(load_cost(COST)), "run_tags": dict(eva.RUN_TAGS), "code_sha256": blt.code_sha256(),
           "identity": {"library_sha256": "LIB", "checkpoint_identity_sha256": "CKPT", "weights_sha256": "W",
                        "h_exec": 5, "schedule_id": SCHEDULE.schedule_id, "template_sha256": "TPL", "suite": "libero_10",
                        "corpus_manifest_sha256": "CORP", "code_sha256": blt.code_sha256()},
           "arm": {"name": "loto_k2_ir70", "yaml_sha256": "ARM", "tiers": ["full", "warm75"]}}
    p = tmp_path / "frozen_run.json"
    p.write_text(json.dumps(rec))
    return rec, blt.sha256_file(p)


def _write_episode(root, conn, run_tag, task_id, orig, ep_id, n_steps, *, frozen_attrs, side=True, tmp=False,
                   bad_side=False, attrs=None, group_names=None, control_offset=0, side_task=None, drop_field=None,
                   h_exec=5):
    d = root / run_tag / f"conn_{conn}" / "groot_libero"
    d.mkdir(parents=True, exist_ok=True)
    name = f"episode_{ep_id:04d}_x{'.h5.tmp' if tmp else '.h5'}"
    with h5py.File(d / name, "w") as f:
        f.attrs.update({"task_id": task_id, "orig_init_state_idx": orig, "episode_id": ep_id, "num_steps": n_steps,
                        "success": True, "task": "t", "run_tag": run_tag, "connection_id": conn, "h_exec": h_exec,
                        "denoise_schedule_id": SCHEDULE.schedule_id, "denoising_num_steps": 8, **frozen_attrs, **(attrs or {})})
        names = group_names or [f"step_{i:04d}" for i in range(n_steps)]
        for gname in names:
            g = f.create_group(gname)
            for field in ("vision_0", "vision_1", "prompt_emb", "robot_state", "clean_action"):
                if field == drop_field:
                    continue
                g.create_dataset(field, data=np.zeros((16, 32) if field == "clean_action" else (2, 2), np.float32))
    if side:
        with (d / ll.SIDECAR_NAME).open("a") as f:
            for i in range(n_steps):
                r = {"run_tag": run_tag, "connection_id": conn, "episode_id": ep_id, "decision_id": i,
                     "control_step_idx": 5 * i + control_offset, "task": side_task or "t", "task_id": task_id,
                     "orig_init_state_idx": orig, "s": 0.99, "hit_type": "FULL_HIT" if i % 2 else "WARM_START",
                     "start_t": None if i % 2 else 0.75, "winner_id": "e:0", "searched": True, "episode_success": True}
                if bad_side and i == 0:
                    r["orig_init_state_idx"] = orig + 1
                f.write(json.dumps(r) + "\n")


def _good_logs(root, fa, **over):
    _write_episode(root, "a", "verify", 0, 3, 3, 4, frozen_attrs=fa, **over)
    _write_episode(root, "a", "verify", 0, 9, 9, 2, frozen_attrs=fa)
    _write_episode(root, "b", "verify", 1, 4, 54, 3, frozen_attrs=fa)
    _write_episode(root, "b", "verify", 1, 6, 56, 1, frozen_attrs=fa)
    for t, indices in _test_pool()["verify"].items():
        for i in indices:
            if (int(t), i) not in ((0, 3), (0, 9), (1, 4), (1, 6)):
                _write_episode(root, "rest", "verify", int(t), i, int(t) * 50 + i, 1, frozen_attrs=fa)


def test_collect_decisions_admits_only_the_frozen_run_and_names_every_discrepancy(tmp_path):
    rec, sha = _frozen(tmp_path)
    fa = vcl.frozen_attrs(rec, sha)
    manifest = rec["pool"]
    root = tmp_path / "logs"
    _good_logs(root, fa)
    _write_episode(root, "a", "smoke", 0, 7, 7, 2, frozen_attrs=fa)
    rows, eps = vcl.collect_decisions(root, manifest, rec, sha, run_tag="verify")
    assert len(eps) == 50 and len(rows) == 56 and all("h5" in r for r in rows)
    with pytest.raises(SystemExit, match="formal run tag"):
        vcl.collect_decisions(root, manifest, rec, sha, run_tag="smoke")
    with pytest.raises(SystemExit, match="pool manifest"):
        vcl.collect_decisions(root, {"verify": {"0": [3, 9], "1": [4]}}, rec, sha)
    _write_episode(root, "b", "verify", 0, 3, 103, 2, frozen_attrs=fa)
    with pytest.raises(SystemExit, match="duplicate"):
        vcl.collect_decisions(root, manifest, rec, sha)

    def fresh(name, **over):
        r = tmp_path / name
        _good_logs(r, fa, **over)
        return r

    with pytest.raises(SystemExit, match="missing"):
        r = tmp_path / "missing"
        _write_episode(r, "a", "verify", 0, 3, 3, 4, frozen_attrs=fa)
        vcl.collect_decisions(r, manifest, rec, sha)
    r = fresh("extra")
    _write_episode(r, "b", "verify", 2, 1, 101, 1, frozen_attrs=fa)
    with pytest.raises(SystemExit, match="not on the manifest"):
        vcl.collect_decisions(r, manifest, rec, sha)
    with pytest.raises(SystemExit, match="identity mismatch"):
        vcl.collect_decisions(fresh("badside", bad_side=True), manifest, rec, sha)
    r = fresh("tmp")
    _write_episode(r, "c", "verify", 1, 4, 154, 3, frozen_attrs=fa, tmp=True)
    with pytest.raises(SystemExit, match="unfinished"):
        vcl.collect_decisions(r, manifest, rec, sha)
    # B1: every episode must carry the frozen record's digests
    with pytest.raises(SystemExit, match="loto_frozen_record_sha256"):
        vcl.collect_decisions(fresh("nofrozen", attrs={"loto_frozen_record_sha256": "other"}), manifest, rec, sha)
    with pytest.raises(SystemExit, match="loto_arm_yaml_sha256"):
        vcl.collect_decisions(fresh("otherarm", attrs={"loto_arm_yaml_sha256": "other"}), manifest, rec, sha)
    # B5: step groups, control step, task and required fields are all compared, not just counted
    with pytest.raises(SystemExit, match="step groups"):
        vcl.collect_decisions(fresh("groups", group_names=["step_0001", "step_0002", "step_0003", "step_0004"]), manifest, rec, sha)
    with pytest.raises(SystemExit, match="control_step_idx"):
        vcl.collect_decisions(fresh("ctrl", control_offset=99), manifest, rec, sha)
    with pytest.raises(SystemExit, match="task"):
        vcl.collect_decisions(fresh("task", side_task="other"), manifest, rec, sha)
    with pytest.raises(SystemExit, match="missing fields"):
        vcl.collect_decisions(fresh("field", drop_field="clean_action"), manifest, rec, sha)
    with pytest.raises(SystemExit, match="h_exec"):
        vcl.collect_decisions(fresh("hexec", h_exec=3), manifest, rec, sha)


# ------------------------------------------------------------------
# 7 / 7b / 7c  labels, sampling, join, exceedance
# ------------------------------------------------------------------


def test_threshold_judge_miss_carries_no_winner_and_labels_follow_the_table(tmp_path, stub_model, tiny_library):
    judge = ThresholdJudge(cp1_threshold=0.99, warm_tiers=[{"threshold": 0.98, "start_t": 0.75}])
    res = judge([types.SimpleNamespace(id="e:0", score=0.5)], CheckpointID.CP1, {})
    assert res.hit_type.name == "MISS" and res.winner_id is None
    runner, templates = stub_model
    w, mask = _w_mask()
    payload = tiny_library[0].payload
    executed = torch.full((H, D), 3.0)
    f, g = _group(tmp_path)

    def labels(hit, start_t=None):
        row = {"hit_type": hit, "start_t": start_t}
        return vcl.k2_labels(runner, templates, "task A", g, row, payload, executed, w, mask, 5, SCHEDULE, ref_seed=9)

    z = blt.make_noise(9, (1, H, D))
    ref = (z * 2.0)[0]
    warm_c = payload.intermediates[0.75] * 10.0 + 0.75
    full = labels("FULL_HIT")
    assert full["dispatched_y_key"] == "y_full"
    assert full["y_full"] == weighted_chunk_deviation(executed, ref, w, mask, 5)
    assert full["y_rem2"] == weighted_chunk_deviation(warm_c, ref, w, mask, 5)
    warm = labels("WARM_START", 0.75)
    assert warm["dispatched_y_key"] == "y_rem2"
    assert warm["y_full"] == weighted_chunk_deviation(payload.action_chunk, ref, w, mask, 5)
    assert warm["y_rem2"] == weighted_chunk_deviation(executed, ref, w, mask, 5)
    miss = labels("MISS")
    assert miss["dispatched_y_key"] is None
    assert miss["y_full"] == warm["y_full"] and miss["y_rem2"] == full["y_rem2"]
    f.close()


def test_online_consistency_checks():
    ok = {"hit_type": "FULL_HIT", "s": 0.99, "winner_id": "e:0", "searched": True, "run_tag": "v", "connection_id": "c", "episode_id": 1, "decision_id": 0}
    vcl.check_online_consistency(ok, {"candidate_id": "e:0", "s_offline": 0.99 + 1e-5})
    with pytest.raises(SystemExit):
        vcl.check_online_consistency(ok, {"candidate_id": "e:1", "s_offline": 0.99})
    with pytest.raises(SystemExit):
        vcl.check_online_consistency(ok, {"candidate_id": "e:0", "s_offline": 0.99 + 1e-3})
    with pytest.raises(SystemExit):  # B2: a NaN online score is not "close enough"
        vcl.check_online_consistency(dict(ok, s=float("nan")), {"candidate_id": "e:0", "s_offline": 0.99})
    with pytest.raises(SystemExit):
        vcl.check_online_consistency(dict(ok, hit_type="WARM_START", start_t=0.5), {"candidate_id": "e:0", "s_offline": 0.99})
    miss = dict(ok, hit_type="MISS", winner_id=None, s=0.7)
    vcl.check_online_consistency(miss, {"candidate_id": "e:9", "s_offline": 0.7})
    with pytest.raises(SystemExit):
        vcl.check_online_consistency(miss, {"candidate_id": "e:9", "s_offline": 0.6})
    with pytest.raises(SystemExit):
        vcl.check_online_consistency(dict(miss, s=float("nan")), {"candidate_id": "e:9", "s_offline": 0.7})
    vcl.check_online_consistency(dict(miss, searched=False, s=None), {"candidate_id": None, "s_offline": None})
    with pytest.raises(SystemExit):
        vcl.check_online_consistency(dict(ok, hit_type="ODD"), {"candidate_id": "e:0", "s_offline": 0.99})


def test_sample_decisions_keeps_base_uniform_and_warm_extra_apart():
    rows = [{"run_tag": "verify", "connection_id": "c", "episode_id": i // 10, "decision_id": i % 10,
             "hit_type": "WARM_START" if i % 4 == 0 else "FULL_HIT"} for i in range(100)]
    sampled, man = vcl.sample_decisions(rows, 20, 1)
    base = [r for r in sampled if r["sample_group"] == "base"]
    extra = [r for r in sampled if r["sample_group"] == "warm_extra"]
    assert len(base) == 20 and man["n_base"] == 20 and all(r["hit_type"] == "WARM_START" for r in extra)
    assert man["n_warm_total"] == 25 and man["n_warm_in_base"] + man["n_warm_extra"] == 25
    assert len({vcl.row_key(r) for r in sampled}) == len(sampled) == len(man["keys"])
    sampled2, _ = vcl.sample_decisions(rows, 20, 1)
    assert [r["decision_id"] for r in sampled2] == [r["decision_id"] for r in sampled]
    _, man_all = vcl.sample_decisions(rows, 500, 1)
    assert man_all["n_base"] == 100 and man_all["n_warm_extra"] == 0


def _labelled(i, hit, group="base", **over):
    row = {"run_tag": "verify", "connection_id": "c", "episode_id": i // 10, "decision_id": i % 10, "task_id": 0,
           "orig_init_state_idx": i // 10, "hit_type": hit, "sample_group": group, "candidate_id": "x",
           "online_s": 0.9, "s": 0.9, "y_full": 1.0, "y_rem2": 1.0, "dispatched_y_key": vcl.DISPATCHED_OF_HIT[hit],
           "control_step_idx": 5 * (i % 10), "task": "t", "episode_success": True, "searched": True,
           "start_t": 0.75 if hit == "WARM_START" else None, "winner_id": "x" if hit != "MISS" else None,
           "s_source": "online_confirmed"}
    row.update(over)
    return row


def test_validate_labelled_rows_requires_the_whole_sample_with_finite_labels():
    rows = [_labelled(i, "FULL_HIT") for i in range(6)] + [_labelled(6, "MISS", candidate_id=None, y_full=None, y_rem2=None,
                    s=None, online_s=None, searched=False, s_source="offline_only",
                    dispatched_y_key=None, label_reason="no_offline_candidate")]
    manifest = {"keys": [list(vcl.row_key(r)) for r in rows],
                "selected_rows": [{**{k: r[k] for k in vcl.LABEL_COPY_FIELDS}, "s": r["online_s"]} for r in rows]}
    join = vcl.validate_labelled_rows(rows, manifest)
    assert join["n_rows"] == 7 and join["n_no_candidate"] == 1
    with pytest.raises(SystemExit, match="sampled but not labelled"):  # B2: half the sample missing
        vcl.validate_labelled_rows(rows[:3], manifest)
    with pytest.raises(SystemExit, match="labelled twice"):
        vcl.validate_labelled_rows(rows + rows[:1], manifest)
    with pytest.raises(SystemExit, match="not in the sampling manifest"):
        vcl.validate_labelled_rows(rows + [_labelled(99, "FULL_HIT")], manifest)
    nan = [dict(r) for r in rows]
    nan[0]["y_rem2"] = float("nan")
    with pytest.raises(SystemExit, match="non-finite"):
        vcl.validate_labelled_rows(nan, manifest)
    wrong = [dict(r) for r in rows]
    wrong[1]["dispatched_y_key"] = "y_rem2"
    with pytest.raises(SystemExit, match="dispatched_y_key"):
        vcl.validate_labelled_rows(wrong, manifest)
    nocand = [dict(r) for r in rows]
    nocand[2].update(candidate_id=None, y_full=None, y_rem2=None)
    with pytest.raises(SystemExit, match="no candidate"):
        vcl.validate_labelled_rows(nocand, manifest)


def _verify_rows(n_eps, rows_per_ep, exceed_rate, seed, s_lo=0.7, s_hi=0.99, event_free_eps=0):
    rng = np.random.default_rng(seed)
    rows, manifest = [], {}
    for e in range(n_eps):
        task, orig = e % 10, e
        manifest[f"{task}:{orig}"] = {"task_id": task, "orig_init_state_idx": orig, "success": True}
        rate = 0.0 if e < event_free_eps else exceed_rate
        for d in range(rows_per_ep):
            s = float(rng.uniform(s_lo, s_hi))
            rows.append({"task_id": task, "orig_init_state_idx": orig, "decision_id": d, "hit_type": "FULL_HIT",
                         "dispatched_y_key": "y_full", "sample_group": "base", "online_s": s, "candidate_id": "x",
                         "y_full": 100.0 if rng.uniform() < rate else 0.0, "y_rem2": 0.0})
    return rows, manifest


def _flat_fit():
    return _pl([0.6, 0.8, 1.0], [[5.0, 4.0, 3.0], [4.0, 3.0, 2.0]])


def test_exceedance_report_gates_readings_and_is_invariant_to_row_duplication():
    fit = _flat_fit()
    rows, man = _verify_rows(30, 15, 0.05, seed=3)
    rep = vcl.exceedance_report(rows, man, fit, tiers=("full",), n_boot=200, seed=1)["tiers"]["full"]
    assert rep["reading"] == "within_preset_tolerance_in_support", rep
    doubled = rows + [dict(r, decision_id=r["decision_id"] + 1000) for r in rows]
    rep2 = vcl.exceedance_report(doubled, man, fit, tiers=("full",), n_boot=200, seed=1)["tiers"]["full"]
    assert rep2["point_estimate"] == rep["point_estimate"] and rep2["bootstrap"]["raw_interval"] == rep["bootstrap"]["raw_interval"]
    # B6: every episode mixed (events and non-events in each) at 24% must read as risk above tolerance
    mixed, man_m = _verify_rows(50, 20, 0.24, seed=8)
    rep_m = vcl.exceedance_report(mixed, man_m, fit, tiers=("full",), n_boot=300, seed=1)["tiers"]["full"]
    assert rep_m["n_episodes_with_nonevent"] == 50 and rep_m["n_episodes_with_event"] >= 45
    assert rep_m["reading"] == "risk_above_preset_tolerance_in_support", rep_m
    high, man_h = _verify_rows(40, 10, 0.3, seed=4, event_free_eps=10)
    rep_h = vcl.exceedance_report(high, man_h, fit, tiers=("full",), n_boot=200, seed=1)["tiers"]["full"]
    assert rep_h["reading"] == "risk_above_preset_tolerance_in_support", rep_h
    zero, man_z = _verify_rows(25, 12, 0.0, seed=5)
    rep_z = vcl.exceedance_report(zero, man_z, fit, tiers=("full",), n_boot=200, seed=1)["tiers"]["full"]
    assert rep_z["reading"] == "insufficient_evidence" and rep_z["interval"] is None and rep_z["point_estimate"] == 0.0
    small, man_s = _verify_rows(20, 1, 0.1, seed=6)
    rep_s = vcl.exceedance_report(small, man_s, fit, tiers=("full",), n_boot=200, seed=1)["tiers"]["full"]
    assert rep_s["reading"] == "insufficient_evidence" and any("rows" in r for r in rep_s["reasons"])
    outside, man_o = _verify_rows(25, 12, 0.02, seed=7, s_lo=0.1, s_hi=0.5)
    rep_o = vcl.exceedance_report(outside, man_o, fit, tiers=("full",), n_boot=50, seed=1)["tiers"]["full"]
    assert rep_o["n_in_support"] == 0 and rep_o["out_of_support"]["n"] == 300 and rep_o["reading"] == "insufficient_evidence"
    assert 1 <= rep["n_score_bins"] <= 4
    bad = [dict(r) for r in rows]
    bad[0]["y_full"] = float("nan")
    with pytest.raises(SystemExit):
        vcl.exceedance_report(bad, man, fit, tiers=("full",), n_boot=5, seed=1)


def test_joint_refit_uses_base_rows_only_and_refuses_missing_labels(cost):
    rows = []
    rng = np.random.default_rng(0)
    for i in range(400):
        s = float(rng.uniform(0.6, 1.0))
        rows.append({"sample_group": "base" if i % 4 else "warm_extra", "candidate_id": "c", "s": s,
                     "y_full": 8 - 3 * s + 0.5, "y_rem2": 8 - 3 * s, "run_tag": "v", "connection_id": "c",
                     "episode_id": 0, "decision_id": i})
    out = vcl.joint_refit(rows, cost, alpha=0.05)
    assert out["n_rows"] == 300 and not out.get("fit_unavailable") and out["fit"]["k"] == 2
    assert vcl.joint_refit(rows[:5], cost, alpha=0.05)["fit_unavailable"]
    broken = [dict(r) for r in rows]
    broken[1]["y_rem2"] = None  # a base row (i % 4 != 0) with a candidate but no label
    with pytest.raises(SystemExit):
        vcl.joint_refit(broken, cost, alpha=0.05)


def test_run_summary_prices_with_measured_cost(cost):
    decisions = [{"hit_type": "FULL_HIT"}] * 50 + [{"hit_type": "WARM_START", "start_t": 0.75}] * 30 + [{"hit_type": "MISS"}] * 20
    eps = {"0:1": {"success": True}, "0:2": {"success": False}}
    out = vcl.run_summary(decisions, eps, cost)
    tiers = rc.ladder(cost, [0.75])
    assert out["counts"] == {"full": 50, "warm75": 30, "miss": 20} and out["n_success"] == 1
    assert math.isclose(out["realized_ir_measured_cost"], rc.realized_ir(out["counts"], tiers, cost))


@pytest.fixture
def verification_pipeline(tmp_path, cost, monkeypatch):
    """Exercise the real collect/label/report commands with only model/retrieval computation stubbed."""
    import exp.robocasa365.rit_shadow as shadow

    w, mask = np.ones(32, np.float32), np.ones(32, bool)
    fits = _fits_json(tmp_path, cost, identity_over={"weights_sha256": blt.weights_sha256(torch.as_tensor(w), torch.as_tensor(mask))})
    pool = _pool_manifest(tmp_path)
    rec = eva.emit_arm("libero_10", "loto_all", fits, cost, 70.0, TEMPLATE, tmp_path / "cfg", pool_manifest=pool)
    frozen = tmp_path / "frozen_run.json"
    frozen.write_text(json.dumps(rec))
    frozen_sha = blt.sha256_file(frozen)
    logs, merged, labelled = (tmp_path / k for k in ("logs", "merged", "labelled"))
    _good_logs(logs, vcl.frozen_attrs(rec, frozen_sha))
    vcl.cmd_collect(types.SimpleNamespace(frozen_record=str(frozen), pool_manifest=str(pool), log_root=str(logs),
                   run_tag="verify", n_sample=2000, seed=20260914, out_dir=str(merged)))
    monkeypatch.setattr(vcl, "load_library", lambda *a, **kw: types.SimpleNamespace(
        sha256="LIB", entries=[], per_task_entries={"t": 1}, traj_of={}, by_id={"e:0": types.SimpleNamespace(payload=None)}))
    monkeypatch.setattr(shadow, "library_action_weights", lambda *a: (w, mask))
    monkeypatch.setattr(vcl, "checkpoint_identity", lambda *a: {"sha256": "CKPT"})
    monkeypatch.setattr(vcl, "build_storage", lambda *a: object())
    monkeypatch.setattr(vcl, "build_retrieval", lambda *a, **kw: object())
    monkeypatch.setattr(vcl, "make_query_builder", lambda *a: object())
    monkeypatch.setattr(blt, "ModelSide", lambda *a, **kw: types.SimpleNamespace(schedule=SCHEDULE, runner=None, templates=None))
    monkeypatch.setattr(vcl, "offline_candidate", lambda *a: {"candidate_id": "e:0", "s_offline": 0.99})
    monkeypatch.setattr(vcl, "k2_labels", lambda runner, templates, task, group, row, *a, **kw: {
        "y_full": 1.0, "y_rem2": 1.0, "dispatched_y_key": vcl.DISPATCHED_OF_HIT[row["hit_type"]]})
    label_args = types.SimpleNamespace(suite="libero_10", frozen_record=str(frozen),
                    sample_manifest=str(merged / "sample_manifest.json"), sampled_jsonl=str(merged / "sampled_decisions.jsonl"),
                    template_yaml=str(TEMPLATE), library_pkl="fixture.pkl", checkpoint="fixture_ckpt", denoising_steps=8,
                    h_exec=5, device="cpu", root_seed=20260914, limit=0, out_dir=str(labelled))
    vcl.cmd_label(label_args)
    report_args = types.SimpleNamespace(suite="libero_10", frozen_record=str(frozen),
                    label_record=str(labelled / "verify_labels.record.json"), sample_manifest=label_args.sample_manifest,
                    verify_rows=str(labelled / "verify_rows.jsonl"), episodes=str(merged / "episodes.json"),
                    decisions=str(merged / "decisions.jsonl"), fits=str(fits), cost=str(COST),
                    tol=0.05, n_boot=1000, seed=0, min_rows=200, min_episodes=20, min_event_episodes=5, min_valid_fraction=0.9,
                    out_dir=str(tmp_path / "out"))
    return types.SimpleNamespace(report=report_args, label=label_args, record=rec, frozen_sha=frozen_sha)


def test_report_refuses_partial_labels_and_foreign_curves(tmp_path, cost, verification_pipeline):
    """B1 / B2: exercise valid collected artifacts before checking report refusals."""
    pipe = verification_pipeline
    args = pipe.report
    vcl.cmd_report(args)
    report = json.loads((tmp_path / "out" / "verify.json").read_text())
    assert report["frozen_record_sha256"] == pipe.frozen_sha
    assert report["run"]["n_episodes"] == 50 and report["run"]["n_decisions"] == 56
    assert all(t["reading"] == "insufficient_evidence" for t in report["exceedance"]["tiers"].values())
    with pytest.raises(SystemExit, match="suite"):
        vcl.cmd_report(types.SimpleNamespace(**(vars(args) | {"suite": "libero_spatial"})))
    other = tmp_path / "other"
    other.mkdir()
    other_fits = _fits_json(other, cost, identity_over={"library_sha256": "OTHER"})
    with pytest.raises(SystemExit, match="fits.json is not the one"):
        vcl.cmd_report(types.SimpleNamespace(**(vars(args) | {"fits": str(other_fits)})))
    vcl.cmd_label(types.SimpleNamespace(**(vars(pipe.label) | {"limit": 2})))
    with pytest.raises(SystemExit, match="partial"):
        vcl.cmd_report(types.SimpleNamespace(**(vars(args) | {
            "label_record": str(pathlib.Path(args.label_record).with_name("verify_labels.partial.record.json"))})))
    rows = blt.read_jsonl(args.verify_rows)
    blt.write_jsonl(pathlib.Path(args.verify_rows), rows[:2])
    record = json.loads(pathlib.Path(args.label_record).read_text())
    record["rows_sha256"] = blt.sha256_file(args.verify_rows)
    pathlib.Path(args.label_record).write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="sampled but not labelled"):
        vcl.cmd_report(args)


@pytest.mark.parametrize("field,value", [
    ("sample_group", "warm_extra"), ("task_id", 9), ("orig_init_state_idx", 49), ("hit_type", "MISS"),
    ("online_s", 0.7), ("s", None), ("s", float("nan")), ("s_source", "offline_only"),
    ("winner_id", "other"), ("candidate_id", "other"), ("y_full", float("nan")),
    ("ref_seed", 42), ("suite", "libero_spatial"),
])
def test_report_revalidates_label_semantics_even_with_a_matching_digest(verification_pipeline, field, value):
    args = verification_pipeline.report
    rows = blt.read_jsonl(args.verify_rows)
    rows[0][field] = value
    blt.write_jsonl(pathlib.Path(args.verify_rows), rows)
    rec = json.loads(pathlib.Path(args.label_record).read_text())
    rec["rows_sha256"] = blt.sha256_file(args.verify_rows)
    pathlib.Path(args.label_record).write_text(json.dumps(rec))
    with pytest.raises(SystemExit):
        vcl.cmd_report(args)


@pytest.mark.parametrize("field", ["decisions", "episodes"])
def test_report_rejects_modified_population_bytes(verification_pipeline, field):
    args = verification_pipeline.report
    path = pathlib.Path(getattr(args, field))
    path.write_text(path.read_text() + "\n")
    with pytest.raises(SystemExit, match=f"{field} bytes"):
        vcl.cmd_report(args)


@pytest.mark.parametrize("field,value", [("tol", 1.0), ("n_boot", 20), ("seed", 42), ("min_rows", 1),
                                         ("min_episodes", 1), ("min_event_episodes", 0), ("min_valid_fraction", 0.0)])
def test_report_cannot_relax_frozen_thresholds(verification_pipeline, field, value):
    args = types.SimpleNamespace(**(vars(verification_pipeline.report) | {field: value}))
    with pytest.raises(SystemExit, match=f"report {field}"):
        vcl.cmd_report(args)


def test_label_checks_sample_and_h5_before_loading_the_model(verification_pipeline, monkeypatch):
    args = verification_pipeline.label
    monkeypatch.setattr(blt, "ModelSide", lambda *a, **kw: pytest.fail("must reject before model allocation"))
    sampled = pathlib.Path(args.sampled_jsonl)
    original = sampled.read_text()
    rows = blt.read_jsonl(sampled)
    rows[0]["sample_group"] = "warm_extra"
    blt.write_jsonl(sampled, rows)
    with pytest.raises(SystemExit, match="bytes/metadata"):
        vcl.cmd_label(args)
    sampled.write_text(original)
    with h5py.File(rows[0]["h5"], "r+") as f:
        f.attrs["denoise_schedule_id"] = "wrong"
    with pytest.raises(SystemExit, match="HDF5 bytes changed"):
        vcl.cmd_label(args)


# ------------------------------------------------------------------
# 8  serve entry point
# ------------------------------------------------------------------


@pytest.fixture
def serve(monkeypatch):
    stub = types.ModuleType("gr00t")
    stub.__spec__ = types.SimpleNamespace(name="gr00t")  # keep find_spec callers happy (N4)
    if "gr00t" not in sys.modules:
        monkeypatch.setitem(sys.modules, "gr00t", stub)
    import exp.libero_groot.serve_groot_libero as sgl
    return sgl


@pytest.mark.parametrize("extra", [
    ["--loto-log-out", "/tmp/x"],
    ["--loto-log-out", "/tmp/x", "--loto-run-tag", "smoke"],
    ["--loto-log-out", "/tmp/x", "--loto-run-tag", "smoke", "--loto-frozen-record", "f.json"],
    ["--loto-log-out", "/tmp/x", "--loto-run-tag", "smoke", "--loto-frozen-record", "f.json", "--cache-config", "a.yaml"],
    ["--loto-log-out", "/tmp/x", "--loto-run-tag", "smoke", "--loto-frozen-record", "f.json", "--cache-config", "a.yaml", "--concurrent", "--trace-out", "/tmp/h"],
    ["--loto-log-out", "/tmp/x", "--loto-run-tag", "smoke", "--loto-frozen-record", "f.json", "--cache-config", "a.yaml", "--concurrent", "--rit-shadow-out", "/tmp/s.jsonl"],
    ["--loto-log-out", "/tmp/x", "--loto-run-tag", "smoke", "--loto-frozen-record", "f.json", "--cache-config", "a.yaml", "--concurrent", "--allow-dynamic-bundles"],
])
def test_serve_rejects_incomplete_or_conflicting_loto_flags(serve, monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["serve_groot_libero.py", *extra])
    with pytest.raises(SystemExit) as exc:
        serve.main()
    assert exc.value.code == 2


def test_loto_factory_binds_the_frozen_record_before_serving(serve, monkeypatch, tmp_path):
    import openpi.cache.config as cache_config
    import openpi.cache.groot.load_guard as lg
    import openpi.cache.groot.staged as gs
    import openpi.cache.orchestrator as orch

    created = []

    class FakeLogger:
        def __init__(self, policy, runner, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(cache_config, "build_per_connection_components", lambda config, storage, *, quiet=False: {
        "timer": object(), "storage": object(), "key_builder": object(), "gates": [], "judges": [],
        "search_strategies": [], "write_policy": object(), "offline_writers": [], "library_stats": object()})
    monkeypatch.setattr(orch, "CacheOrchestrator", lambda **kw: object())
    monkeypatch.setattr(gs, "GrootStagedRunner", lambda model, *, timer=None, compile_vision=False: object())
    monkeypatch.setattr(lg, "live_num_inference_timesteps", lambda policy: 8)
    monkeypatch.setattr(blt, "checkpoint_identity", lambda path, **kw: {"sha256": "CKPT"})
    monkeypatch.setattr(ll, "GrootLotoLogger", FakeLogger)
    arm = tmp_path / "arm.yaml"
    arm.write_text("x: 1\n")
    lib = tmp_path / "lib.pkl"
    lib.write_bytes(b"lib")
    rec, _ = _frozen(tmp_path)
    rec["identity"]["library_sha256"] = blt.sha256_file(lib)
    rec["arm"]["yaml_sha256"] = blt.sha256_file(arm)
    frozen = tmp_path / "frozen_run.json"
    frozen.write_text(json.dumps(rec))
    cfg = types.SimpleNamespace(backend=types.SimpleNamespace(in_memory=types.SimpleNamespace(preload_path=str(lib))),
                                checkpoints={"cp1": types.SimpleNamespace(enabled=True)})

    def args(**over):
        base = dict(cache_config=str(arm), loto_log_out=str(tmp_path / "logs"), loto_run_tag="verify",
                    loto_frozen_record=str(frozen), checkpoint="/ckpt", experiment="groot_libero", rit_h_exec=5)
        base.update(over)
        return types.SimpleNamespace(**base)

    factory, label = serve._build_loto_factory(args(), cfg, object(), threading.Lock())
    assert "loto-log" in label
    factory(types.SimpleNamespace(model=object()))
    factory(types.SimpleNamespace(model=object()))
    assert len(created) == 2 and created[0]["run_tag"] == "verify" and created[0]["h_exec"] == 5
    attrs = created[0]["identity_attrs"]
    assert attrs["loto_frozen_record_sha256"] == blt.sha256_file(frozen) and attrs["loto_arm_yaml_sha256"] == blt.sha256_file(arm)
    assert attrs["loto_checkpoint_identity_sha256"] == "CKPT" and attrs["loto_pool_manifest_sha256"] == "POOL"
    with pytest.raises(SystemExit, match="run tag"):
        serve._build_loto_factory(args(loto_run_tag="other"), cfg, object(), threading.Lock())
    with pytest.raises(SystemExit, match="h_exec"):
        serve._build_loto_factory(args(rit_h_exec=3), cfg, object(), threading.Lock())
    with pytest.raises(SystemExit, match="eager stage-1"):
        serve._build_loto_factory(args(compile_stage1=True), cfg, object(), threading.Lock())
    arm.write_text("x: 2\n")  # the served arm is not the frozen one
    with pytest.raises(SystemExit, match="arm yaml"):
        serve._build_loto_factory(args(), cfg, object(), threading.Lock())
    arm.write_text("x: 1\n")
    monkeypatch.setattr(blt, "checkpoint_identity", lambda path, **kw: {"sha256": "OTHER"})
    with pytest.raises(SystemExit, match="checkpoint identity"):
        serve._build_loto_factory(args(), cfg, object(), threading.Lock())
    monkeypatch.setattr(blt, "checkpoint_identity", lambda path, **kw: {"sha256": "CKPT"})
    monkeypatch.setattr(lg, "live_num_inference_timesteps", lambda policy: 4)
    factory, _ = serve._build_loto_factory(args(), cfg, object(), threading.Lock())
    with pytest.raises(SystemExit, match="live schedule"):
        factory(types.SimpleNamespace(model=object()))
    cfg2 = types.SimpleNamespace(backend=cfg.backend, checkpoints={"cp1": types.SimpleNamespace(enabled=True),
                                                                   "cp2": types.SimpleNamespace(enabled=True)})
    with pytest.raises(SystemExit):
        serve._build_loto_factory(args(), cfg2, object(), threading.Lock())
