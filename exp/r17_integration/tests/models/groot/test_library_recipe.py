"""The P1 recipe consumes true P5 widths and excludes padded GR action channels."""
from dataclasses import replace
import numpy as np
from exp.r17_integration.common.layers.recipe import LibraryData, fit_library
from exp.r17_integration.common.query_view import QueryView
from .test_shared_factory import published


def test_one_camera_awmetric_corrector_loeo_padding_and_fork(manifest, context):
    rng = np.random.default_rng(51)
    n = 24
    features = rng.normal(size=(n, 128)).astype(np.float32)
    state = rng.normal(size=(n, 4)).astype(np.float32)
    action = rng.normal(size=(n, 16, 32)).astype(np.float32)
    episodes = np.repeat(np.arange(3), 8)
    step = np.tile(np.arange(8), 3)
    task = np.zeros(n, np.int64)
    data = LibraryData({"vision_0": features}, state, action, task, episodes, step, source_digest="CPU-stub-only")
    layout = manifest.model_spec().layout
    recipe = fit_library(data, layout=layout, task_count=50, commit_horizon=10, k=4, kref=2,
                         corrector=True, guard=True, early=False)
    poisoned = action.copy()
    poisoned[..., 4:] = rng.normal(0, 100000, (n, 16, 28))
    other = fit_library(replace(data, action=poisoned), layout=layout, task_count=50, commit_horizon=10,
                        k=4, kref=2, corrector=True, guard=True, early=False)
    for evidence in recipe.fit_evidence["loeo"]:
        assert evidence["query_episode"] not in evidence["donor_episodes"]
    raw = fit_library(data, layout=layout, task_count=50, commit_horizon=10, k=4, kref=2,
                     corrector=False, guard=False, early=False)
    query = QueryView(replace(context, decision_seq=1, task_id=0), {"vision_0": features[1] + .01}, state[1],
                      hist_has_vision=(True,), hist_hit=(1,), prev_hit=True)
    a, b, unc = recipe.query(query), other.query(query), raw.query(query)
    np.testing.assert_array_equal(a.rows, b.rows)
    np.testing.assert_array_equal(a.weights, b.weights)
    np.testing.assert_array_equal(a.action[..., :4], b.action[..., :4])
    np.testing.assert_array_equal(a.action[:, 3], unc.action[:, 3])  # corrector changes motion only
    fork = recipe.fork()
    assert fork.action is recipe.action and not recipe.action.flags.writeable
    assert fork.guard is not recipe.guard and fork._memo is None
    assert tuple(recipe.fields) == ("vision_0",)


def test_shared_periodic_fallback_with_p5_model_and_recipe(published, manifest, obs, tmp_path):
    from exp.r17_integration.models.groot.serve import make_connection_factory
    from exp.r17_integration.common.config import build_shared_storage
    from .test_shared_factory import config
    cfg = config(published, commit=10)
    from .test_fix_round1 import teacher_store, recipe_config
    from .conftest import begin_episode
    store = teacher_store(tmp_path / "teacher", manifest)
    cfg = recipe_config(cfg, store, corrector=False, guard=False, periodic_gap=1)
    captures = []
    factory = make_connection_factory(published, cfg, build_shared_storage(cfg), capture_sink=captures.append)
    policy = factory()
    begin_episode(policy)
    # First decision uses library; the shared periodic rule subsequently forces real policy work.
    first = policy.infer(obs)
    assert captures[-1].source == "library" and captures[-1].stage_calls == (1, 0, 0)
    policy.infer(dict(obs, __execution__={"decision_seq": 0, "executed_count": 5}))
    assert captures[-1].source == "policy" and captures[-1].stage_calls == (1, 1, 4)
    assert captures[-1].diagnostics["knob"] is True
