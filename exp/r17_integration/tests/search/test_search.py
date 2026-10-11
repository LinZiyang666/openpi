import copy
import json

import numpy as np
import pytest

from exp.r17_integration.debug.validate import digest
from exp.r17_integration.search.audit import audit_campaign, audit_freeze
from exp.r17_integration.search.build_library import (
    build_library, load_stock_artifact, mw_split, nested_episodes, save_stock_artifact, stock_artifact)
from exp.r17_integration.search.fit import (
    LibraryView, fit_library, fit_recipe, library_data, load_fitted_recipe, save_fitted_recipe)
from exp.r17_integration.search.freeze import DEPENDENCIES, freeze
from exp.r17_integration.search.replay import compare_replays, replay
from exp.r17_integration.search.store import Store, write_store
from exp.r17_integration.tests.debug.fixtures import binding, captured, record


def make_store(tmp_path, b=None, *, episodes=3, decisions=2):
    b = b or binding()
    path, eps = captured(tmp_path, b, episodes=episodes, decisions=decisions)
    return build_library([path], tmp_path / "store", b, eps,
                         key_fields=b["model"]["feature_spec"]["fields"])


@pytest.mark.parametrize("model", ["pi05", "groot", "fastwam", "dp"])
@pytest.mark.parametrize("benchmark", ["mw", "libero"])
def test_dynamic_store_stock_roundtrip(tmp_path, model, benchmark):
    b = binding(model, benchmark)
    store = make_store(tmp_path, b, episodes=1)
    loaded = Store(tmp_path / "store", expected_binding=b)
    for field in store.fields:
        assert loaded.array(field).tobytes() == store.array(field).tobytes()
        assert not loaded.array(field).flags.writeable
    assert loaded.array("active_actions").shape == (2, b["model"]["layout"]["horizon"], b["model"]["layout"]["valid_action_dim"])
    artifact = stock_artifact(store)
    assert artifact["entries"][0].next_ids == [artifact["entries"][1].id]
    assert artifact["entries"][1].prev_ids == [artifact["entries"][0].id]
    assert artifact["entries"][0].payload.task_key == "reach-v3"
    path = tmp_path / "library.pkl"
    save_stock_artifact(store, path)
    backend = load_stock_artifact(path, b)
    assert backend.artifact_meta["entry_count"] == 2
    assert backend.artifact_meta["native_warm_supported"] is False
    assert backend.artifact_meta["native_sampler"] == b["model"]["default_sampler"]


def test_store_ragged_native_grid_and_rng_exact(tmp_path):
    grids = [np.array([99, 49, 0], np.int64), np.array([99], np.int64)]
    store = write_store(tmp_path / "ragged", binding(), [record(), record(seq=1)], {"native_grid": grids})
    for source, loaded in zip(grids, store.array("native_grid")):
        assert source.tobytes() == loaded.tobytes() and not loaded.flags.writeable


def test_partial_and_tampered_store_rejected(tmp_path):
    partial = tmp_path / "partial"
    with pytest.raises(ValueError, match="empty"):
        write_store(partial, binding(), [], {})
    assert (partial / "PARTIAL.json").exists() and not (partial / "READY.json").exists()
    store = make_store(tmp_path)
    data = tmp_path / "store" / "native_grid.npy"
    data.write_bytes(data.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        Store(tmp_path / "store")


def test_stock_receipt_checks_identity_and_tamper(tmp_path):
    store = make_store(tmp_path)
    path = tmp_path / "library.pkl"
    save_stock_artifact(store, path)
    wrong = copy.deepcopy(binding())
    wrong["model"]["norm_identity"] = "wrong"
    with pytest.raises(ValueError, match="binding"):
        load_stock_artifact(path, wrong)
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_stock_artifact(path, binding())


def test_nested_complete_episode_selection_and_small_corrector_limits():
    eps = [record(init=10+i, task=task) for task in ("reach-v3", "push-v3") for i in range(4)]
    sizes = nested_episodes(list(reversed(eps)), [2, 4, 6, 8], task_names=["reach-v3", "push-v3"])
    previous = set()
    for size, selection in sizes.items():
        chosen = {ep["task_uid"] for ep in selection["episodes"]}
        assert previous <= chosen and len(chosen) == size
        assert selection["corrector_supported"] == (size >= 6)
        previous = chosen
    bad = dict(eps[0], split="dev")
    with pytest.raises(ValueError, match="library-only"):
        nested_episodes([bad], [1], task_names=["reach-v3"])


@pytest.mark.parametrize("idx,partition", [(0, "test"), (9, "test"), (10, "library"), (29, "library"), (30, "dev"), (49, "dev")])
def test_sealed_mw_split(idx, partition):
    assert mw_split(idx) == partition


def test_library_excludes_queries_outcomes_and_whole_loeo_episode(tmp_path):
    store = make_store(tmp_path)
    view = LibraryView(store)
    assert all("success" not in row and "cost_ms" not in row for row in view.rows)
    with pytest.raises(KeyError, match="allowlist"):
        view.array("posterior_rng")
    donors = view.donor_indices(exclude_episode=view.episode_ids[0])
    assert donors == (2, 3, 4, 5)
    with pytest.raises(ValueError, match="complete source"):
        view.donor_indices(exclude_episode="query-episode")
    observed = []
    def fitter(v, *, config, corrector):
        observed.append(v.donor_indices(exclude_episode=v.episode_ids[0]))
        return v.array("active_actions").sum()
    product, receipt = fit_library(store, fitter=fitter, corrector=True)
    assert observed == [(2, 3, 4, 5)] and receipt["corrector"]
    assert receipt["store_digest"] == store.manifest["content_digest"]


def test_corrector_refuses_small_library(tmp_path):
    store = make_store(tmp_path, episodes=2)
    with pytest.raises(ValueError, match=">=3"):
        fit_library(store, fitter=lambda *a, **k: None, corrector=True)


def test_builder_rejects_incomplete_episode_and_feature_geometry(tmp_path):
    b = binding()
    path, eps = captured(tmp_path, b, episodes=1)
    eps[0]["n_decisions"] = 3
    with pytest.raises(ValueError, match="complete episode"):
        build_library([path], tmp_path / "bad", b, eps, key_fields=b["model"]["feature_spec"]["fields"])
    with pytest.raises(ValueError, match="FeatureSpec"):
        build_library([path], tmp_path / "bad", b, eps, key_fields=["vision_0", "vision_1"])


def test_hashed_mw_protocol_still_checks_explicit_split_policy(tmp_path):
    b = binding()
    b["benchmark_protocol"] = "protocol-content-digest"
    path, episodes = captured(tmp_path, b, episodes=1)
    episodes[0]["orig_init_idx"] = 0
    with pytest.raises(ValueError, match="split leakage"):
        build_library([path], tmp_path / "leak", b, episodes, key_fields=b["model"]["feature_spec"]["fields"],
                      split_policy="mt50_init_v1")


def test_shared_datacollector_h5_builder_bridge(tmp_path):
    import h5py
    from exp.r17_integration.debug.validate import file_sha
    from exp.r17_integration.tests.debug.fixtures import arrays
    from openpi.collect.data_collector import InferenceEmbeddings, write_episode_attrs, write_step_group
    b = binding()
    source = tmp_path / "shared.h5"
    a, keys = arrays(b)
    with h5py.File(source, "w") as f:
        write_episode_attrs(f, experiment="fixture", task="reach-v3", episode_id=0, num_steps=1, success=True, episode_attrs={})
        embs = InferenceEmbeddings([keys["vision_0"]], np.zeros((1, 4), np.float32), a["state"],
                                   [a["snapshot_x"]], a["normalized_actions"])
        write_step_group(f.create_group("step_0000"), embs)
        f["step_0000"].create_dataset("native_grid", data=a["native_grid"])
    episode = record(n_decisions=1)
    manifest = {str(source): dict(binding=b, sha256=file_sha(source), episode=episode)}
    store = build_library([source], tmp_path / "from_shared_h5", b, [episode], key_fields=["vision_0", "robot_state"],
                          collected_source_manifests=manifest,
                          feature_builder=lambda raw: {"vision_0": raw["vision_0"], "robot_state": raw["robot_state"]})
    assert store.array("normalized_actions")[0].tobytes() == a["normalized_actions"].tobytes()
    assert store.array("native_grid")[0].tobytes() == a["native_grid"].tobytes()


def test_p1_published_libraryview_fit_bridge(tmp_path):
    from exp.r17_integration.common.layers import fit_library_view
    b = binding(key_width=8)
    store = make_store(tmp_path, b, episodes=3, decisions=4)
    recipe, receipt = fit_library(store, fitter=fit_library_view, corrector=False,
        config=dict(layout=b["model"]["layout"], ordered_tasks=["reach-v3"], early=False, guard=False, k=4))
    assert recipe.layout.horizon == 16 and receipt["fitter_status"] == "injected"


def test_shared_p1_fit_recipe_and_numeric_reload(tmp_path):
    from exp.r17_integration.common.contracts import RequestContext
    from exp.r17_integration.common.query_view import QueryView
    b = binding(key_width=8)
    store = make_store(tmp_path, b, episodes=6, decisions=4)
    recipe, receipt = fit_recipe(store, task_names=["reach-v3"], corrector=True, early=False, guard=False,
                                 k=4, kref=3, commit_horizon=5)
    assert recipe.fit_evidence["partition"] == "library"
    assert all(r["query_episode"] not in r["donor_episodes"] for r in recipe.fit_evidence["loeo"])
    save_fitted_recipe(recipe, receipt, tmp_path / "fit")
    restored = load_fitted_recipe(tmp_path / "fit", expected_binding=b)
    q = QueryView(RequestContext("query", "mt50_seed7_corner2_r17_v1", "reach-v3", 0, 1),
                  {"vision_0": store.array("key_vision_0")[0]}, store.array("state")[0],
                  hist_has_vision=(True,), hist_hit=(0,))
    a, c = recipe.query(q), restored.query(q)
    assert a.action.tobytes() == c.action.tobytes()
    np.testing.assert_array_equal(a.rows, c.rows)
    assert a.force_miss == c.force_miss
    raw = restored.fork()
    raw.use_corrector = False
    assert a.action[:, 3].tobytes() == raw.query(q).action[:, 3].tobytes()


@pytest.mark.parametrize("dependency", DEPENDENCIES)
def test_freeze_change_fails_audit(dependency):
    deps = {name: "sealed-hash" for name in DEPENDENCIES}
    seal = freeze(deps)
    assert audit_freeze(seal, deps)
    altered = dict(deps, **{dependency: "changed"})
    with pytest.raises(ValueError, match="changed"):
        audit_freeze(seal, altered)


def test_audit_accepted_attempt_contiguity_and_missing_cost():
    row = dict(record(n_decisions=2), accepted=True, status="ok")
    decisions = [record(seq=0), record(seq=1)]
    report = audit_campaign([row["task_uid"], "missing"], [row], decisions)
    assert report["accepted"] == 1 and report["missing"] == ["missing"]
    del decisions[1]["actual_nfe"]
    assert audit_campaign([row["task_uid"]], [row], decisions)["cost_status"] == "unsupported"
    with pytest.raises(ValueError, match="noncontiguous"):
        audit_campaign([row["task_uid"]], [row], decisions[:1])


def test_generic_offline_resident_and_transport_replay_contract():
    class Method:
        def fit(self, library):
            self.library = library
        def reset(self, episode):
            self.seq = 0
        def query(self, query):
            self.seq += 1
            return dict(actions=np.array([query, self.seq], np.float32), donor_ids=["donor1"], weights=[1.],
                        guard=False, knob=self.seq % 2 == 0, stage_calls=[1, 0, 0], actual_nfe=0, cost_ms=1.)
    reference = replay(Method(), "sealed-library", [("episode", [2, 3])])
    resident = replay(Method(), "sealed-library", [("episode", [2, 3])])
    # This exercises JSON+array boundary; actual WS model receipt is a separate gate.
    transported = copy.deepcopy(resident)
    assert compare_replays(reference, transported)["decisions"] == 2
    transported[1]["metadata"]["guard"] = True
    with pytest.raises(AssertionError, match="guard"):
        compare_replays(reference, transported)
