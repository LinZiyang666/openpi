"""Staged / integration evidence for the CP2 arm (plan §3.1, §3.3, §7; G2 R1 item 8).

- ``run_stage2`` vs ``run_stage2_capture``: one identical backbone forward,
  bit-identical KV, the capture only retains the prefix output.
- Real orchestrator + real CP2 components from the deployed YAML: every tier
  (FULL / WARM / MISS) on both the direct and the coordinator path is exactly
  one ``check(CP2)`` — no CP1 before, no CP3 after — one step increment, one
  broadcast, one clear, and the wire carries the library digest.
- Same-bundle-id hot replacement through the production seams
  (``wps._bundles`` registry -> ``serve_policy._wrap_policy``): CP2 -> CP1 and
  CP1 -> CP2 while the old wrapper's request is in flight inside stage 2.
- ``bench_cp2_overhead`` dry run: forced timer, every core segment recorded
  per decision, schema-complete CSV / JSON.
"""

from __future__ import annotations

import csv
import json
import pickle
import threading
import types
from types import SimpleNamespace

import pytest
import torch
import yaml

from exp.actioncache_baseline import bench_cp2_overhead as bench
from exp.actioncache_baseline import libs
from exp.actioncache_baseline.export_arms import cp2_arm_yaml
from openpi.cache.components.cp2_vlm_key_builder import get_projection_spec, project
from openpi.cache.config import build_per_connection_components, build_shared_storage, load_cache_config
from openpi.cache.interceptor import InferenceInterceptor
from openpi.cache.orchestrator import CacheOrchestrator
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.serving import websocket_policy_server as wps
from tests.actioncache_baseline.test_acb_interceptor_cp2 import _Model, _obs, _Policy

PROJ = libs.ProjectionArgs(seed=5, d=8, p=0.25, input_dim=64)
SPEC = get_projection_spec(PROJ.seed, PROJ.d, PROJ.p, PROJ.input_dim)
#: Deterministic "backbone output" of the fake model: [1, 4, 16] -> 64 dims.
PREFIX = (torch.arange(64, dtype=torch.float32).reshape(1, 4, 16) / 64.0) - 0.5
QKEY = project(PREFIX.reshape(1, -1), SPEC)[0]


class _DetModel(_Model):
    def run_stage2_capture(self, stage1):
        self.capture_calls += 1
        return SimpleNamespace(stage1=stage1, past_key_values=None, prefix_out=PREFIX.clone())


class _Coord:
    """Coordinator stand-in recording the per-request capture capability."""

    def __init__(self, model):
        self.model = model
        self.stage2_flags: list[tuple[str, bool]] = []
        self.stage3_payloads: list = []

    def submit_to_stage(self, stage_id, bundle_id, payload, **kw):
        if stage_id == 1:
            return self.model.run_stage1(payload)
        if stage_id == 2:
            flag = bool(kw.get("requires_stage2_capture"))
            self.stage2_flags.append((bundle_id, flag))
            self._before_stage2()
            return (self.model.run_stage2_capture if flag else self.model.run_stage2)(payload)
        self.stage3_payloads.append(payload)
        return SimpleNamespace(action_chunk=torch.randn(1, 50, 32), intermediates=None)

    def _before_stage2(self):
        pass


def _write_artifact(path, keys):
    entries = [CacheEntry(
        id=f"e{i}", checkpoint_id=CheckpointID.CP2, query_keys={libs.FIELD: k.clone().float()},
        payload=CachePayload(action_chunk=torch.full((10, 32), float(i)), intermediates={0.1: torch.zeros(10, 32)},
                             denoising_num_steps=10, task_key="t"),
        step_idx=i, trajectory_id="traj", prev_ids=[], next_ids=[]) for i, k in enumerate(keys)]
    art = {"key_builder_type": libs.KEY_BUILDER_TYPE, "checkpoint_id": "CP2",
           "vector_dims": {libs.FIELD: PROJ.d}, "entries": entries, "projection": SPEC.meta(),
           "id_policy": libs.ID_POLICY, "model": {"checkpoint_dir": "c", "weights_digest": "0" * 64},
           "stage1_path": "online"}
    with open(path, "wb") as f:
        pickle.dump(art, f)


def _cp2_config(tmp_path, name, tier, keys):
    pkl = tmp_path / f"{name}.pkl"
    _write_artifact(pkl, keys)
    doc = cp2_arm_yaml(preload_path=str(pkl), projection=PROJ, tier=tier, theta_raw=0.85)
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path, load_cache_config(path)


def _cp1_config(tmp_path):
    doc = {
        "enabled": True,
        "keys": {"robot_state": {"enabled": True, "weight": 1.0}},
        "key_builder": {"type": "placeholder"},
        "checkpoints": {"cp1": {"enabled": True, "gate": {"type": "always_search"},
                                "judge": {"type": "threshold", "threshold": 0.9},
                                "search_strategy": {"type": "weighted_rrf_knn", "top_k": 1}}},
        "backend": {"type": "in_memory", "vector_dims": {"robot_state": 32}},
        "write_policy": {"type": "never"},
    }
    path = tmp_path / "cp1.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return load_cache_config(path)


def _interceptor(cfg, storage, model, *, coordinator=None, bundle_id="default"):
    comps = build_per_connection_components(cfg, storage, yaml_id="acb", quiet=True)
    orch = CacheOrchestrator(
        storage=comps["storage"], key_builder=comps["key_builder"], gates=comps["gates"],
        judges=comps["judges"], search_strategies=comps["search_strategies"], timer=comps["timer"],
        write_policy=comps.get("write_policy"), offline_writers=comps.get("offline_writers", ()),
        library_stats=comps.get("library_stats"),
    )
    it = InferenceInterceptor(_Policy(model), timer=comps["timer"], orchestrator=orch,
                              coordinator=coordinator, bundle_id=bundle_id)
    it.on_task_begin()
    return it, orch


# ------------------------------------------------------------------
# 1. staged KV parity
# ------------------------------------------------------------------


class _KV:
    def __init__(self, emb):
        self.k = emb.cumsum(1) * 3.0
        self.v = emb.flip(1) - 1.0


def test_run_stage2_and_capture_issue_identical_forward_and_kv():
    calls: list[dict] = []

    class _Backbone:
        def forward(self, **kw):
            calls.append(kw)
            emb = kw["inputs_embeds"][0]
            return [emb * 2.0 + 1.0, None], _KV(emb)

    fake = SimpleNamespace(paligemma_with_expert=_Backbone())
    fake._stage2_llm_backbone = types.MethodType(PI0Pytorch._stage2_llm_backbone, fake)
    stage1 = SimpleNamespace(
        state=torch.zeros(1, 32), prefix_embs=torch.randn(1, 4, 16),
        prefix_pad_masks=torch.ones(1, 4, dtype=torch.bool),
        prefix_att_2d_masks_4d=torch.zeros(1, 1, 4, 4), prefix_position_ids=torch.arange(4)[None],
    )
    plain = PI0Pytorch.run_stage2(fake, stage1)
    cap = PI0Pytorch.run_stage2_capture(fake, stage1)
    assert len(calls) == 2
    a, b = calls
    assert a["attention_mask"] is stage1.prefix_att_2d_masks_4d and b["attention_mask"] is stage1.prefix_att_2d_masks_4d
    assert a["position_ids"] is stage1.prefix_position_ids and b["position_ids"] is stage1.prefix_position_ids
    assert a["inputs_embeds"][0] is stage1.prefix_embs and b["inputs_embeds"][0] is stage1.prefix_embs
    assert a["inputs_embeds"][1] is None and b["inputs_embeds"][1] is None
    assert a["past_key_values"] is None and b["past_key_values"] is None and a["use_cache"] is b["use_cache"] is True
    assert set(a) == set(b) == {"attention_mask", "position_ids", "past_key_values", "inputs_embeds", "use_cache"}
    assert torch.equal(plain.past_key_values.k, cap.past_key_values.k)
    assert torch.equal(plain.past_key_values.v, cap.past_key_values.v)
    assert plain.stage1 is stage1 and cap.stage1 is stage1
    assert plain.prefix_out is None and torch.equal(cap.prefix_out, stage1.prefix_embs * 2.0 + 1.0)


# ------------------------------------------------------------------
# 2. one check(CP2) per decision, every tier, both paths
# ------------------------------------------------------------------


@pytest.mark.parametrize("via_coordinator", [False, True], ids=["direct", "coordinator"])
def test_cp2_decision_cycle_per_tier(tmp_path, via_coordinator):
    cases = {
        "full": ("n0", [QKEY, QKEY, QKEY], "FULL_HIT"),
        "warm": ("n1", [QKEY, QKEY, QKEY], "WARM_START"),
        "miss": ("n0", [-QKEY, -QKEY, -QKEY], "MISS"),   # cosine -1 -> normalised 0
    }
    for name, (tier, keys, expect) in cases.items():
        _path, cfg = _cp2_config(tmp_path, f"{name}_{via_coordinator}", tier, keys)
        storage = build_shared_storage(cfg)
        model = _DetModel()
        coord = _Coord(model) if via_coordinator else None
        it, orch = _interceptor(cfg, storage, model, coordinator=coord, bundle_id="b")
        checks: list[CheckpointID] = []
        counts = {"broadcast": 0, "clear": 0}
        real_check, real_bc, real_clear = orch.check, orch.broadcast_action, orch.clear
        orch.check = lambda cp, **kw: (checks.append(cp), real_check(cp, **kw))[1]
        orch.broadcast_action = lambda a: (counts.__setitem__("broadcast", counts["broadcast"] + 1), real_bc(a))[1]
        orch.clear = lambda: (counts.__setitem__("clear", counts["clear"] + 1), real_clear())[1]
        step0 = orch._step_counter
        out = it.infer(_obs())
        meta = out["__hit_meta__"]
        assert meta["hit_type"] == expect and meta["checkpoint"] == "CP2", name
        assert checks == [CheckpointID.CP2], (name, checks)               # no CP1 before, no CP3 after
        assert orch._step_counter == step0 + 1                              # exactly one step increment
        assert counts == {"broadcast": 1, "clear": 1}
        assert meta["library_sha256"] == storage.artifact_meta["library_sha256"]
        assert meta["cp1_score"] is None and meta["score"] is not None
        assert model.capture_calls == 1 and model.stage2_calls == 0        # capture variant only
        if via_coordinator:
            assert coord.stage2_flags == [("b", True)]
        if expect == "FULL_HIT":
            assert model.stage3_calls == 0 and model.stage3_from == []
            assert out["actions"].shape == (10, 32) and float(out["actions"][0, 0]) in (0.0, 1.0, 2.0)
            if via_coordinator:
                assert coord.stage3_payloads == []
        elif expect == "WARM_START":
            assert model.stage3_calls == 0
            if via_coordinator:
                from openpi.serving.batching_coordinator import Stage3WarmStartPayload

                (pl,) = coord.stage3_payloads
                assert isinstance(pl, Stage3WarmStartPayload) and pl.start_t == 0.1 and pl.num_steps == 10
                assert tuple(pl.start_x.shape) == (10, 32) and model.stage3_from == []
            else:
                assert model.stage3_from == [((1, 10, 32), 0.1, 10)]
        else:
            assert model.stage3_from == []
            if via_coordinator:
                from openpi.serving.batching_coordinator import Stage3MissPayload

                (pl,) = coord.stage3_payloads
                assert isinstance(pl, Stage3MissPayload) and model.stage3_calls == 0
            else:
                assert model.stage3_calls == 1
        # second decision on the same wrapper: still one check, counter +1
        checks.clear()
        if via_coordinator:
            coord.stage3_payloads.clear()
        it.infer(_obs())
        assert checks == [CheckpointID.CP2] and orch._step_counter == step0 + 2
        it.on_task_end()


# ------------------------------------------------------------------
# 3. same-bundle-id hot replacement with an in-flight request
# ------------------------------------------------------------------


@pytest.fixture
def clean_bundles():
    saved = dict(wps._bundles)
    wps._bundles.clear()
    yield wps._bundles
    wps._bundles.clear()
    wps._bundles.update(saved)


def _bundle(cfg, version, yaml_id):
    return wps.CurrentCacheBundle(config_path=f"{yaml_id}.yaml", cache_config=cfg,
                                  shared_storage=build_shared_storage(cfg), version=version, yaml_id=yaml_id)


class _HoldCoord(_Coord):
    """Blocks the FIRST stage-2 submit until released (an in-flight request)."""

    def __init__(self, model):
        super().__init__(model)
        self.entered = threading.Event()
        self.release = threading.Event()
        self._n = 0

    def _before_stage2(self):
        self._n += 1
        if self._n == 1:
            self.entered.set()
            assert self.release.wait(10), "in-flight request was never released"


@pytest.mark.parametrize("direction", ["cp2_to_cp1", "cp1_to_cp2"])
def test_same_bundle_id_hot_swap_with_in_flight_request(tmp_path, clean_bundles, direction):
    from scripts.serve_policy import Args, _wrap_policy

    _path, cp2_cfg = _cp2_config(tmp_path, "swap", "n0", [QKEY, QKEY, QKEY])
    cp1_cfg = _cp1_config(tmp_path)
    first, second = (cp2_cfg, cp1_cfg) if direction == "cp2_to_cp1" else (cp1_cfg, cp2_cfg)
    model = _DetModel()
    coord = _HoldCoord(model)

    clean_bundles["same"] = _bundle(first, 1, "v1")
    a = _wrap_policy(_Policy(model), Args(), quiet=True, eager=True,
                     shared_cache={"coordinator": coord}, bundle_id="same")
    a.on_task_begin()
    out_a: dict = {}
    t = threading.Thread(target=lambda: out_a.update(a.infer(_obs())))
    t.start()
    assert coord.entered.wait(10)

    # Replacement published under the SAME id while a's request sits in stage 2;
    # a new connection binds through the production factory seam.
    clean_bundles["same"] = _bundle(second, 2, "v2")
    b = _wrap_policy(_Policy(model), Args(), quiet=True, eager=True,
                     shared_cache={"coordinator": coord}, bundle_id="same")
    b.on_task_begin()
    out_b = b.infer(_obs())
    coord.release.set()
    t.join(10)
    assert not t.is_alive()

    first_is_cp2, second_is_cp2 = first is cp2_cfg, second is cp2_cfg
    assert a._cp2_only is first_is_cp2 and b._cp2_only is second_is_cp2
    # Each wrapper froze its own capability: the old one keeps it across the swap.
    assert coord.stage2_flags == [("same", first_is_cp2), ("same", second_is_cp2)]
    ma, mb = out_a["__hit_meta__"], out_b["__hit_meta__"]
    assert ma["checkpoint"] == ("CP2" if first_is_cp2 else "CP1")
    assert mb["checkpoint"] == ("CP2" if second_is_cp2 else "CP1")
    cp2_digest = clean_bundles["same"].shared_storage.artifact_meta["library_sha256"] if second_is_cp2 \
        else build_shared_storage(cp2_cfg).artifact_meta["library_sha256"]
    for meta, is_cp2 in ((ma, first_is_cp2), (mb, second_is_cp2)):
        if is_cp2:
            assert meta["hit_type"] == "FULL_HIT" and meta["library_sha256"] == cp2_digest
        else:  # CP1 wrapper: legacy wire, no library digest; empty library -> MISS, cp1_score mirrors score
            assert meta["hit_type"] == "MISS" and "library_sha256" not in meta and meta["cp1_score"] == meta["score"]
    a.on_task_end()
    b.on_task_end()


# ------------------------------------------------------------------
# 4. overhead harness dry run
# ------------------------------------------------------------------


def test_bench_dry_run_records_every_segment(tmp_path):
    from openpi.serving import monitor as _monitor

    prev = _monitor.get_monitor_level()
    try:
        path, _cfg = _cp2_config(tmp_path, "bench", "n0", [QKEY, -QKEY, -QKEY])
        config, comps, orch, timer = bench.build_orchestrator(path)
        assert timer._enabled and config.timer.enabled
        decisions = [(f"ep{i // 3}", i % 3, SimpleNamespace(prefix_out=torch.randn(1, 4, 16))) for i in range(6)]
        orch.on_task_begin()
        out_dir = tmp_path / "out"
        measured = bench.run_decisions(orch, timer, iter(decisions), device=torch.device("cpu"),
                                       out_dir=out_dir, cold=2, max_decisions=100)
        rec = bench.write_record(out_dir, suite="libero_spatial", cache_yaml=path, config=config,
                                 components=comps, device=torch.device("cpu"), cohort_root=None,
                                 model_binding=None, measured=measured)
        assert rec["suite"] == "libero_spatial" and rec["timer_enabled"] is True
        assert rec["n_decisions"] == 6 and rec["cold"]["count"] == 2 and rec["warm"]["count"] == 4
        for s in bench.CORE_SEGMENTS:
            seg = rec["per_segment"][s]
            assert seg["count"] == 6 and seg["median"] is not None and seg["p95"] >= 0.0, s
        assert rec["verdict"] in {"ok_report", "report_with_caption", "halt_profile_segments"}
        assert rec["library_sha256"] == comps["storage"].artifact_meta["library_sha256"]
        assert rec["model"]["library_model"]["weights_digest"] == "0" * 64
        assert json.loads((out_dir / "overhead.json").read_text())["suite"] == "libero_spatial"
        with (out_dir / "per_decision.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 6 and set(rows[0]) == {"episode", "step_idx", "total_ms", *[f"{s}_ms" for s in bench.SEGMENTS]}
        for r in rows:
            assert float(r["total_ms"]) >= 0.0
            for s in bench.CORE_SEGMENTS:
                assert r[f"{s}_ms"] not in ("", "None") and float(r[f"{s}_ms"]) >= 0.0
    finally:
        _monitor.set_monitor_level(prev)
