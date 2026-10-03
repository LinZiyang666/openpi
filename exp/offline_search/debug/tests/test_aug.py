import hashlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from exp.offline_search.debug import reader, schema
from exp.offline_search.debug.fixtures import make_synthetic_arm
from exp.offline_search.debug.aug.job import run_arm, private_seed, wire_observations, benchmark, fit_specs_sha, code_sha
from exp.offline_search.debug.aug.models import FakeModel, input_sha
from exp.offline_search.debug.aug.retrieval import FakeRetrieval, FrozenRetrieval, load_fit, make_third_fit, used_fit_specs
from exp.offline_search.debug.aug.validate import agreement
from exp.offline_search.debug.validate import validate_arm


def empty_arm(path, model="pi05"):
    arm = reader.open_arm(make_synthetic_arm(path, model=model), "synthetic")
    for part in (arm.debug_dir / "aug").glob("*/*.npz"):
        part.unlink()
    return arm


def test_private_seeds_input_sha_and_rng_do_not_depend_on_order():
    np.random.seed(15)
    state_before = np.random.get_state()
    model = FakeModel()
    a = {"observation/state": np.arange(8, dtype=np.float64), "prompt": "a",
         "observation/image": np.zeros((16, 16, 3), np.uint8), "observation/wrist_image": np.ones((16, 16, 3), np.uint8)}
    b = dict(a, prompt="b")
    seeds = [private_seed("campaign", 0, 1, seq, 0) for seq in (1, 2)]
    first = model.draw(model.encode([a, b]), seeds)
    second = model.draw(model.encode([b, a]), seeds[::-1])
    assert np.array_equal(first, second[::-1])
    state_after = np.random.get_state()
    assert state_before[0] == state_after[0]
    assert np.array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]
    assert input_sha(a) != input_sha(b)
    assert input_sha(a) == input_sha(dict(reversed(list(a.items()))))
    assert private_seed("campaign", 0, 1, 1, 1) != seeds[0]
    assert private_seed("campaign", 0., np.float64(1.), 1., 0.) == private_seed("campaign", 0, 1, 1, 0)


def test_resume_ignores_campaign_config_but_rejects_changed_used_fit(tmp_path):
    arm = empty_arm(tmp_path / "run")
    model = FakeModel()
    first = run_arm(arm, model, kinds=["policy_shadow"], checkpoint_sha="fake", fit_config_sha="old-campaign", limit=1)
    assert first["newly_written"]["policy_shadow"] == 1
    resumed = run_arm(arm, model, kinds=["policy_shadow"], checkpoint_sha="fake", fit_config_sha="unrelated-campaign-edit")
    assert resumed["newly_written"]["policy_shadow"] == 53
    class ConfiguredFake(FakeRetrieval):
        def __init__(self, config):
            super().__init__()
            self.config = config
        def fit_specs(self, kind):
            return used_fit_specs(self.config, kind)
    config = dict(full=[dict(name="current", path="A.pkl", sha256="full")],
                  wrist=[dict(name="current", path="W.pkl", sha256="wrist")], unrelated="old")
    retrieval = ConfiguredFake(config)
    run_arm(arm, model, retrieval, kinds=["shadow_look"], checkpoint_sha="fake", limit=1)
    config["unrelated"] = "new"
    config["wrist"][0]["sha256"] = "unrelated-wrist-change"
    assert run_arm(arm, model, retrieval, kinds=["shadow_look"], checkpoint_sha="fake")["newly_written"]["shadow_look"] == 53
    config["full"][0]["sha256"] = "changed-used-fit"
    with pytest.raises(ValueError, match="fit_config_sha"):
        run_arm(arm, model, retrieval, kinds=["shadow_look"], checkpoint_sha="fake")


def test_used_camera_fit_specs_and_code_hash_exclude_unrelated_inputs(monkeypatch):
    config = dict(full=[dict(name="current", path="A.pkl", sha256="full", note="unused")],
                  wrist=[dict(name="current", path="W.pkl", sha256="wrist")],
                  third=[dict(name="current", path="T.pkl", sha256="third")], store_root="unused", cell="unused")
    before = used_fit_specs(config, "camera_shadow")
    config["full"][0]["sha256"] = "unused-full-when-third-is-frozen"
    config["store_root"] = "unused-again"
    assert used_fit_specs(config, "camera_shadow") == before
    assert used_fit_specs(config, "policy_shadow") == {}
    assert fit_specs_sha(None, "policy_shadow") == "none"
    config["wrist"][0]["sha256"] = "changed-wrist"
    assert used_fit_specs(config, "camera_shadow") != before
    actual = Path.read_bytes
    accessed = []
    def track(path):
        accessed.append(path)
        return actual(path)
    monkeypatch.setattr(Path, "read_bytes", track)
    assert len(code_sha()) == 64
    assert {p.name for p in accessed} == {"job.py", "models.py", "retrieval.py"}
    assert all(p.parent.name == "aug" for p in accessed)


def test_wire_observations_do_not_decompress_cache_keys_or_chunks(tmp_path, monkeypatch):
    arm = empty_arm(tmp_path / "run")
    records = arm.decisions().head(2).to_dict("records")
    accessed = []
    actual = np.lib.npyio.NpzFile.__getitem__
    def track(archive, key):
        accessed.append(key)
        return actual(archive, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", track)
    assert len(wire_observations(arm, records)) == 2
    assert set(accessed) <= {"decision_id", "img_third", "img_wrist", "img_third_available", "img_wrist_available", "state_wire", "prompts", "prompt_idx"}


@pytest.mark.parametrize("model_name", ["pi05", "groot"])
def test_all_kinds_roundtrip_capture_untouched_resume_no_model_work(tmp_path, model_name):
    arm = empty_arm(tmp_path / model_name, model_name)
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in arm.debug_dir.rglob("*")
              if p.is_file() and "aug" not in p.relative_to(arm.debug_dir).parts and "derived" not in p.relative_to(arm.debug_dir).parts}
    model, retrieval = FakeModel(model_name), FakeRetrieval(model_name)
    report = run_arm(arm, model, retrieval, batch_size=8, checkpoint_sha="fake")
    assert not (arm.debug_dir / "derived").exists()
    assert report["newly_written"]["policy_shadow"] == 54
    assert report["newly_written"]["shadow_look"] == 54
    assert report["newly_written"].get("camera_shadow", 0) == (54 if model_name == "pi05" else 0)
    ids = arm.decisions().decision_id.tolist()
    arrays = arm.aug("policy_shadow", ids)
    assert arrays["chunk"].shape == (54, model.H, 32)
    assert arrays["seed"].dtype == np.int64
    for i, record in enumerate(arm.decisions().to_dict("records")):
        observation = wire_observations(arm, [record])[0]
        assert arrays["input_sha"][i] == input_sha(observation)
    check = validate_arm(arm, 6)
    assert check["status"] == "PASS", check["missing"]
    counts_before = len(model.encode_batches), len(model.draw_batches)
    parts = {str(p): p.stat().st_mtime_ns for p in (arm.debug_dir / "aug").glob("*/*.npz")}
    resumed = run_arm(arm, model, retrieval, batch_size=32, checkpoint_sha="fake")
    assert not any(resumed["newly_written"].values())
    assert counts_before == (len(model.encode_batches), len(model.draw_batches))
    assert parts == {str(p): p.stat().st_mtime_ns for p in (arm.debug_dir / "aug").glob("*/*.npz")}
    assert before == {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in before}


def test_partial_resume_matches_single_pass_noise_across_batches(tmp_path):
    arm = empty_arm(tmp_path / "same_campaign")
    model = FakeModel()
    first = run_arm(arm, model, kinds=["policy_shadow"], batch_size=1, checkpoint_sha="fake", limit=19)
    assert first["newly_written"]["policy_shadow"] == 19
    second = run_arm(arm, model, kinds=["policy_shadow"], batch_size=8, checkpoint_sha="fake")
    assert second["newly_written"]["policy_shadow"] == 35
    records = arm.decisions().to_dict("records")
    ids = [r["decision_id"] for r in records]
    observations = wire_observations(arm, records)
    seeds = [private_seed(arm.manifest["campaign"], r["task_id"], r["init"], r["decision_seq"], 0) for r in records]
    expected = model.draw(model.encode(observations), seeds)
    assert np.array_equal(arm.aug("policy_shadow", ids)["chunk"], expected)


def test_changed_checkpoint_is_rejected_and_corrupt_parts_are_not_overwritten(tmp_path):
    arm = empty_arm(tmp_path / "run")
    run_arm(arm, FakeModel(), kinds=["policy_shadow"], checkpoint_sha="first", limit=1)
    with pytest.raises(ValueError, match="provenance mismatch"):
        run_arm(arm, FakeModel(), kinds=["policy_shadow"], checkpoint_sha="changed")
    part = next((arm.debug_dir / "aug/policy_shadow").glob("*.npz"))
    data = reader.read_npz(part)
    data["decision_id"] = np.repeat(data["decision_id"], 2)
    schema.write_npz_block(part, data)
    with pytest.raises(ValueError, match="duplicate published"):
        run_arm(arm, FakeModel(), kinds=["policy_shadow"], checkpoint_sha="first")


def test_invalid_shapes_and_camera_models_fail_before_publish(tmp_path):
    arm = empty_arm(tmp_path / "run")
    class BrokenModel(FakeModel):
        def draw(self, encoded, seeds):
            return np.zeros((len(seeds), 1, 7))
    with pytest.raises(ValueError, match="malformed"):
        run_arm(arm, BrokenModel(), kinds=["policy_shadow"], checkpoint_sha="fake")
    assert not list((arm.debug_dir / "aug/policy_shadow").glob("*.npz"))
    with pytest.raises(ValueError, match="unsupported"):
        run_arm(arm, FakeModel("groot"), kinds=["camera_shadow"], checkpoint_sha="fake")


def test_pure_policy_both_libraries_required(tmp_path):
    arm = empty_arm(tmp_path / "run")
    arm.manifest["pure_policy"] = True
    with pytest.raises(ValueError, match="both current and big"):
        run_arm(arm, FakeModel(), FakeRetrieval(), kinds=["shadow_look"], checkpoint_sha="fake")
    run_arm(arm, FakeModel(), FakeRetrieval(libraries=("current", "bpool_cs")), kinds=["shadow_look"], checkpoint_sha="fake")
    arrays = arm.aug("shadow_look")
    assert arrays["cache_chunk_bpool_cs"].shape == (54, 10, 32)
    assert set(arrays["lib_bpool_cs"]) == {"bpool_cs"}


def test_benchmark_has_explicit_batches_and_missing_miss_agreement(tmp_path):
    arm = empty_arm(tmp_path / "run")
    model = FakeModel()
    result = benchmark(arm, model, [1, 8, 32], 32)
    assert [r["batch_size"] for r in result] == [1, 8, 32]
    assert all(r["decisions_per_second"] > 0 for r in result)
    run_arm(arm, model, FakeRetrieval(), checkpoint_sha="fake")
    assert agreement(arm)["status"] == "unavailable"


def test_frozen_fit_sha_and_third_metric_use_correct_camera(tmp_path, monkeypatch):
    from exp.offline_search.harness import api
    from exp.offline_search.rounds.r02.g1_awm import awm
    from exp.offline_search.rounds.r04.k3_cost.method import WristAWM
    rng = np.random.default_rng(1)
    libdir = tmp_path / "library"
    libdir.mkdir()
    (libdir / "ids.json").write_text("{}")
    lib = SimpleNamespace(L=128, H=10, dir=libdir, key_v0=rng.standard_normal((128, 128)).astype(np.float32),
                          key_v1=rng.standard_normal((128, 128)).astype(np.float32),
                          rs=rng.standard_normal((128, 32)).astype(np.float32), action=rng.standard_normal((128, 10, 32)).astype(np.float32),
                          episode=np.repeat(np.arange(8), 16), step=np.tile(np.arange(16), 8))
    lib.tasks = lambda: [0]
    lib.rows_of_task = lambda task: np.arange(128)
    context = SimpleNamespace(model="pi05", lib_key="pi05_l10", action_sigma=np.ones(7), open_library=lambda name: lib)
    monkeypatch.setattr(awm, "PCA_CUR", tmp_path / "pca_cache")
    monkeypatch.setattr(api, "Context", lambda **kwargs: context)
    full = awm.AWM(lib="current", kref=5)
    full.fit(lib, context)
    wrist = WristAWM(lib="current", kref=5)
    wrist.fit(lib, context)
    third = make_third_fit(full, tmp_path, "pi05_l10_cache", tmp_path / "scratch")
    assert next(iter(third.tasks.values())).Wf.shape[0] == 72
    assert np.array_equal(third.B0T, full.B0T)
    assert third.B1T.shape[0] == 0
    query = SimpleNamespace(key_v0=lib.key_v0[0], key_v1=lib.key_v1[0], rs=lib.rs[0], task_id=0, step=0, prev_hit=None,
                            episode=SimpleNamespace(uid="test"), hist_key_v0=[], hist_key_v1=[])
    response = third.query(query)
    query.key_v1 = lib.key_v1[1]
    third.reset(query.episode)
    assert np.array_equal(response.action, third.query(query).action)
    query.key_v0 = lib.key_v0[1]
    third.reset(query.episode)
    assert not np.array_equal(response.action, third.query(query).action)
    specs = {}
    for name, method in (("full", full), ("wrist", wrist)):
        path = tmp_path / (name + ".pkl")
        with path.open("wb") as f:
            pickle.dump(dict(method=method), f)
        specs[name] = [dict(name="current", path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())]
    retriever = FrozenRetrieval(dict(specs, cell="pi05_l10_cache", store_root=str(tmp_path)), tmp_path / "aug", "pi05")
    retriever.ensure_cameras()
    unused = dict(specs, third=[dict(name="current", path="does-not-exist.pkl", sha256="bad")])
    full_only = FrozenRetrieval(unused, tmp_path / "aug", "pi05", kinds=["shadow_look"])
    assert set(full_only.fits) == {("full", "current")}
    groot_only = FrozenRetrieval(dict(unused, wrist=unused["third"]), tmp_path / "aug", "groot")
    assert set(groot_only.fits) == {("full", "current")}
    with pytest.raises(ValueError, match="SHA"):
        load_fit(dict(specs["full"][0], sha256="wrong"))
    encoded = dict(v0=lib.key_v0[:1], v1=lib.key_v1[:1], state=lib.rs[:1])
    records = [dict(task_id=0, init=0, episode_key="episode", decision_seq=0, decision_id="episode:1:0")]
    result = retriever.retrieve(encoded, records)
    direct = full.query(SimpleNamespace(key_v0=lib.key_v0[0], key_v1=lib.key_v1[0], rs=lib.rs[0], task_id=0, step=0,
                                        prev_hit=None, episode=SimpleNamespace(uid="episode")))
    assert np.array_equal(result["rows"][0], direct.topk)
    assert np.array_equal(result["cache_chunk"][0], direct.action)
    assert np.isclose(result["weights"].sum(), 1)
