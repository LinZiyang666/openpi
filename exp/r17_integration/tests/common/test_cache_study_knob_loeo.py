"""CPU serving regressions for the default-off cache-study knobs and LOEO."""
import copy
from dataclasses import asdict, fields, replace
import json
from pathlib import Path
import uuid

import numpy as np
import pytest
import yaml

from exp.r17_integration.common.config import config_from_dict, build_shared_storage
from exp.r17_integration.common.contracts import RequestContext, digest
from exp.r17_integration.common.layers import LayoutRecipe
from exp.r17_integration.common.query_view import QueryView
from exp.r17_integration.common.recipe import bundle_identity, fit_identity, make_recipe_factory, recipe_identity
from exp.r17_integration.common.serving import make_connection_factory
from exp.r17_integration.search.fit import fit_recipe, save_fitted_recipe
from exp.r17_integration.search.store import Store, atomic_json
from exp.r17_integration.search.teacher_library import nested_views
from exp.r17_integration.tests.common.conftest import config_for, with_recipe
from exp.r17_integration.tests.common.test_layers import library, q_for
from exp.r17_integration.tests.ops.test_cache_study import fixture_model, freeze


class Tracker:
    def __init__(self, *args):
        self.count = 0

    def observe(self, *args):
        self.count += 1

    def status(self):
        return dict(state="slow_confirmed" if self.count % 4 == 2 else "ok", W=2)


@pytest.mark.parametrize("gap", [None, 0., 1.9])
def test_complete_group_periodic_calls_and_long_horizon(gap, tmp_path):
    provider = fixture_model("dp_libero")
    cfg, fitted = freeze(provider, tmp_path, 10, gated=True)
    cfg.recipe.periodic_gap = gap
    # Exercise YAML parsing/validation as well as the native serving branch.
    cfg = config_from_dict(asdict(cfg))
    policy = make_connection_factory(provider, cfg, build_shared_storage(cfg))()
    policy.on_episode_start(extra_metadata={"task_name": "reach", "task_id": 0, "original_init_idx": 10})
    policy.recipe.tracker_class = Tracker
    obs = {"state": fitted.state[2], **{name: np.arange(width, dtype=np.float32)
           for name, width in provider.model_spec.feature_spec.fields.items() if name != "robot_state"}}
    metas = []
    for seq in range(32):
        payload = dict(obs)
        if seq:
            payload["__execution__"] = {"decision_seq": seq - 1, "executed_count": 5}
        meta = policy.infer(payload)["__hit_meta__"]
        metas.append(meta)
        if gap is None:
            assert "knob_call" not in meta  # Preserve default-off diagnostics.
            assert (meta["source"] == "policy") == meta["stall_call"]
        else:
            assert type(meta["knob_call"]) is bool
            assert (meta["source"] == "policy") == (meta["stall_call"] or meta["knob_call"])
        if meta["source"] == "policy":
            assert meta["stage_calls"] == (1, 1, 100)
            assert meta["os_force_miss"] == 1.
            assert meta["os_reason"] == (63. if meta["stall_call"] else 62.)
            assert not meta["corrector_applied"]
        elif meta["source"].endswith("_tail"):
            assert meta["stage_calls"] == (0, 0, 0)
        else:
            assert meta["stage_calls"] == (1, 0, 0)
    if gap == 0.:
        assert all(meta["source"] == "policy" and meta["knob_call"] for meta in metas)
    else:
        assert any(meta["source"] == "library_tail" for meta in metas)
        assert any(meta["corrector_applied"] for meta in metas)
        assert any(meta["stall_call"] for meta in metas)
        if gap is not None:
            assert any(meta["knob_call"] and not meta["stall_call"] for meta in metas)
            assert any(meta["stall_call"] and not meta["knob_call"] for meta in metas)


def test_periodic_knob_without_stall_gate(tmp_path):
    provider = fixture_model("dp_libero")
    cfg, fitted = freeze(provider, tmp_path, 10)
    cfg.recipe.periodic_gap = 0.
    policy = make_connection_factory(provider, cfg, build_shared_storage(cfg))()
    policy.on_episode_start(extra_metadata={"task_name": "reach", "task_id": 0, "original_init_idx": 10})
    for seq in range(3):
        obs = {"state": fitted.state[0]}
        if seq:
            obs["__execution__"] = {"decision_seq": seq - 1, "executed_count": 5}
        meta = policy.infer(obs)["__hit_meta__"]
        assert meta["source"] == "policy" and meta["knob_call"] and not meta["stall_call"]
        assert meta["os_reason"] == 62.


@pytest.mark.parametrize("view", [False, True])
def test_factory_serving_loeo_parent_map_and_subset(provider, tmp_path, view):
    cfg = with_recipe(provider, config_for(provider, commit=5), tmp_path, episodes=6)
    cfg.recipe.guard = False
    parent = Store(cfg.recipe.store_path)
    manifest = copy.deepcopy(parent.manifest)
    episode_ids = [701, 17, 809, 23, 997, 41]
    for row in manifest["rows"]:
        row["library_episode_id"] = episode_ids[row["orig_init_idx"] - 10]
    manifest.pop("content_digest")
    manifest["content_digest"] = digest(manifest)
    atomic_json(parent.directory / "READY.json", manifest)
    parent = Store(parent.directory)
    store = parent
    if view:
        alignment = dict(selections={"5": {"reach": episode_ids[:5]}}, digest="fixture", subset_seed=0)
        store = nested_views(parent, alignment, sizes=(5,))[5]
    cfg.recipe.store_path = str(store.directory)
    cfg.recipe.store_digest = store.manifest["content_digest"]
    cfg.recipe.k = len(store.rows)  # All donors: verify that only the own episode disappears.
    fitted, receipt = fit_recipe(store, task_names=["reach"], corrector=False,
                                commit_horizon=5, k=cfg.recipe.k, kref=cfg.recipe.kref,
                                early=True, guard=False, periodic_gap=None)
    ready = save_fitted_recipe(fitted, receipt, tmp_path / "fit")
    cfg.recipe.fitted_path, cfg.recipe.fitted_digest = str(tmp_path / "fit"), ready["digest"]
    before_fit, before_recipe, before_bundle = fit_identity(cfg), recipe_identity(cfg), bundle_identity(cfg)
    factory = make_recipe_factory(provider)
    baseline = factory(cfg, None)
    cfg.recipe.serve_loeo = True
    enabled = factory(config_from_dict(asdict(cfg)), None)
    assert fit_identity(cfg) == before_fit
    assert recipe_identity(cfg) != before_recipe and bundle_identity(cfg) != before_bundle
    assert enabled.action is baseline.action
    assert enabled.serve_loeo_episodes is factory(cfg, None).serve_loeo_episodes
    assert enabled.serve_loeo_episodes[(0, 15)] == episode_ids[-1]
    for init in [10, 11, 15, 99]:
        for step, prev_hit in [(0, None), (1, True), (1, False)]:
            q = QueryView(RequestContext("cpu", "fake_mw.v1", "reach", init, step, task_id=0),
                          {name: parent.array("key_" + name)[0] for name in fitted.fields}, parent.array("state")[0],
                          [True] * step, [1] * step, previous_action=fitted.action[0], prev_hit=prev_hit)
            own = enabled.serve_loeo_episodes.get((0, init))
            result = enabled.fork().query(q)
            expected = baseline.query(q, exclude_episode=own) if own is not None else baseline.fork().query(q)
            assert own not in enabled.episode[result.rows]
            assert set(result.rows) == set(np.flatnonzero(enabled.episode != own))
            for name in ("rows", "weights", "distances", "action"):
                assert getattr(result, name).tobytes() == getattr(expected, name).tobytes()
            # An explicit exclusion retains precedence over serving LOEO.
            other = episode_ids[2]
            shortcut = enabled.query(q, exclude_episode=other)
            assert set(shortcut.rows) == set(np.flatnonzero(enabled.episode != other))
            assert episode_ids[0] in enabled.episode[shortcut.rows]
    off_again = copy.deepcopy(cfg)
    off_again.recipe.serve_loeo = False
    assert not factory(off_again, None).serve_loeo_episodes
    assert recipe_identity(off_again) == before_recipe and bundle_identity(off_again) == before_bundle


def test_serving_loeo_key_includes_task():
    layout, lib = library()
    lib = replace(lib, features={name: np.concatenate([values, values]) for name, values in lib.features.items()},
                  state=np.concatenate([lib.state, lib.state]), action=np.concatenate([lib.action, lib.action]),
                  task_id=np.repeat([0, 1], len(lib.state)), episode=np.concatenate([lib.episode, lib.episode + 100]),
                  step=np.concatenate([lib.step, lib.step]))
    recipe = LayoutRecipe(layout, 2, k=24, corrector=False, guard=False).fit(lib)
    recipe.serve_loeo_episodes = {(0, 10): 0, (1, 10): 100}
    for task, own in [(0, 0), (1, 100)]:
        q = q_for(lib, step=0)
        q.request = replace(q.request, task_id=task, original_init_idx=10)
        result = recipe.fork().query(q)
        assert set(result.rows) == set(np.flatnonzero((recipe.task == task) & (recipe.episode != own)))


@pytest.mark.env_dependent(reason="requires the published n50 DP YAML, fits and B-pool arrays; CPU only")
def test_real_n50_dp_serving_loeo():
    from exp.r17_integration.benchmarks.libero_bridge import LiberoBenchmarkSpec
    from exp.r17_integration.common.benchmarks import register_benchmark
    from exp.r17_integration.common.provider import AdapterProvider, register_provider
    from exp.r17_integration.models.dp.provider import configure_library_provider
    from exp.r17_integration.ops.cache_study import model_spec

    assets = Path("/data/mw_finetune/eval/csztd_r3_l48/assets")
    bpool = Path("/data/r17_library/dp_libero_libero_spatial/bpool")
    raw = yaml.safe_load((assets / "cache_yamls/n50_sp_chrg.yaml").read_text())
    names = {field.name for field in fields(LiberoBenchmarkSpec)}
    serving = LiberoBenchmarkSpec(**{key: value for key, value in raw["recipe"]["library_benchmark"]["serving"].items()
                                    if key in names})
    register_benchmark(serving)
    spec = model_spec(json.loads((assets / "model_spec.json").read_text()))
    provider = configure_library_provider(AdapterProvider("loeo.cpu." + uuid.uuid4().hex, spec,
        lambda: pytest.fail("real-data retrieval must never load a model"), lambda k: spec.default_sampler))
    register_provider(provider)
    raw["adapter_binding"]["provider"] = provider.provider_id
    cfg = config_from_dict(raw)
    factory = make_recipe_factory(provider)
    baseline = factory(cfg, None)
    cfg.recipe.serve_loeo = True
    enabled = factory(cfg, None)
    parent = json.loads((bpool / "READY.json").read_text())
    arrays = {name: np.load(bpool / entry["file"], mmap_mode="r", allow_pickle=False)
              for name, entry in parent["arrays"].items() if name == "state" or name.startswith("key_")}
    task = cfg.recipe.ordered_tasks[0]
    queries = [(row_id, row) for row_id, row in enumerate(parent["rows"])
               if row["task_name"] == task and row["decision_seq"] == 0]
    excluded = unchanged = 0
    for row_id, row in queries:
        init, own = row["orig_init_idx"], row["library_episode_id"]
        assert enabled.serve_loeo_episodes[(0, init)] == own
        q = QueryView(RequestContext("loeo.cpu", cfg.adapter_binding["benchmark_protocol"], task, init, 0, task_id=0),
                      {name: arrays["key_" + name][row_id] for name in enabled.fields}, arrays["state"][row_id])
        enabled.reset(None)
        baseline.reset(None)
        result = enabled.query(q)
        assert own not in enabled.episode[result.rows]
        expected = baseline.base.query(q, exclude_episode=own)
        for name in ("rows", "weights", "distances", "action"):
            assert getattr(result, name).tobytes() == getattr(expected, name).tobytes()
        if own in enabled.episode:
            excluded += 1
        else:
            before = baseline.query(q)
            for name in ("rows", "weights", "distances", "action"):
                assert getattr(result, name).tobytes() == getattr(before, name).tobytes()
            unchanged += 1
    assert excluded and unchanged
    print(f"task 0: {len(queries)} B inits checked; {excluded} own episodes excluded, {unchanged} absent episodes byte-identical")
