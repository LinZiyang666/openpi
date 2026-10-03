import hashlib
import json
import threading

import numpy as np
import pytest
import torch

from exp.offline_search.debug import reader, schema
from exp.offline_search.debug.aug import job
from exp.offline_search.debug.aug.compare import compare
from exp.offline_search.debug.aug.cpu_fake import CpuGrootModel
from exp.offline_search.debug.aug.models import FakeModel
from exp.offline_search.debug.aug.retrieval import FakeRetrieval
from exp.offline_search.debug.tests.test_aug import empty_arm


def observations():
    def obs(prompt, value, embodiment=1):
        return dict(prompt=prompt, embodiment_id=embodiment,
                    **{"observation/state": np.arange(8, dtype=float) + value,
                       "observation/image": np.full((16, 16, 3), value, np.uint8),
                       "observation/wrist_image": np.full((16, 16, 3), value + 1, np.uint8)})
    return [obs("short", 1), obs("a longer prompt", 2), obs("short", 3), obs("short", 4, 2)]


def test_groot_all_stages_bucket_parity_and_conditioning_reuse():
    before = torch.get_rng_state().clone()
    serial, batched = CpuGrootModel("serial"), CpuGrootModel("batched")
    rows, seeds = observations(), [13, 14, 15, 16]
    a, b = serial.encode(rows), batched.encode(rows)
    for key in ("v0", "v1", "state"):
        assert np.array_equal(a[key], b[key])
    old, new = serial.draw(a, seeds), batched.draw(b, seeds)
    assert np.array_equal(old, new)
    assert serial.runner._eagle.batches == [1, 1, 1, 1]
    assert sorted(batched.runner._eagle.batches) == [1, 1, 2]
    assert sorted(batched.runner._eagle.language_model.batches) == [1, 1, 2]
    for x, y in zip(a["stage1"], b["stage1"]):
        assert torch.equal(x.input_embeds, y.input_embeds)
        assert torch.equal(x.image_token_mask, y.image_token_mask)
    chosen = batched.select(b, [2, 0])
    assert np.array_equal(batched.draw(chosen, [15, 13]), new[[2, 0]])
    assert len(batched.runner._eagle.language_model.batches) == 3
    reverse = CpuGrootModel("batched")
    assert np.array_equal(reverse.draw(reverse.encode(rows[::-1]), seeds[::-1]), new[::-1])
    assert torch.equal(before, torch.get_rng_state())


def test_groot_changed_image_token_layout_fails_before_keys():
    model = CpuGrootModel()
    actual = model.runner._model.prepare_input
    def broken(obs):
        eagle, action = actual(obs)
        eagle["eagle_input_ids"][0, 0] = 0
        return eagle, action
    model.runner._model.prepare_input = broken
    with pytest.raises(RuntimeError, match="image-token count"):
        model.encode(observations())


@pytest.mark.parametrize("name", ["pi05", "groot"])
def test_pipeline_bitwise_serial_equivalence_all_kinds_and_no_reencoding(tmp_path, name):
    arm = empty_arm(tmp_path / "input", name)
    serial = CpuGrootModel("serial") if name == "groot" else FakeModel(name)
    batched = CpuGrootModel() if name == "groot" else FakeModel(name)
    old, new = tmp_path / "old", tmp_path / "new"
    a = job.run_arm(arm, serial, FakeRetrieval(name), batch_size=1, checkpoint_sha="fake", prefetch=False,
                    output_dir=old, profile=True)
    b = job.run_arm(arm, batched, FakeRetrieval(name), batch_size=8, checkpoint_sha="fake", prefetch=True,
                    output_dir=new, profile=True)
    kinds = list(a["newly_written"])
    checked = compare(old, new, kinds, atol=0., rtol=0.)
    assert checked["status"] == "PASS", checked
    assert a["newly_written"] == b["newly_written"]
    if name == "groot":
        assert batched.prepared_rows == 54
        assert sum(batched.runner._eagle.batches) == 54
        assert sum(batched.runner._eagle.language_model.batches) == 54
        assert len(batched.runner._eagle.language_model.batches) < 54
    else:
        assert sum(batched.encode_batches) == 54
    assert "decode_io" in b["profile"]["wall_seconds"]
    assert "publish_io.policy_shadow" in b["profile"]["wall_seconds"]
    assert "retrieval.shadow_look" in b["profile"]["wall_seconds"]


def test_prefetch_retrieval_overlaps_model_and_uses_ordered_worker(tmp_path):
    arm = empty_arm(tmp_path / "input")
    retrieval_started, next_forward = threading.Event(), threading.Event()
    class Model(FakeModel):
        def encode(self, obs):
            if len(self.encode_batches) == 1:
                assert retrieval_started.wait(3)
                next_forward.set()
            return super().encode(obs)
    class Retrieval(FakeRetrieval):
        def retrieve(self, *args, **kwargs):
            assert threading.current_thread().name.startswith("aug-retrieve")
            retrieval_started.set()
            assert next_forward.wait(3)
            return super().retrieve(*args, **kwargs)
    job.run_arm(arm, Model(), Retrieval(), batch_size=2, kinds=["policy_shadow", "shadow_look"],
                checkpoint_sha="fake", output_dir=tmp_path / "output")
    assert next_forward.is_set()


def test_prefetch_exception_leaves_resumable_parts_and_releases_lock(tmp_path):
    arm = empty_arm(tmp_path / "input")
    # Force the failure after two successful prepare calls, independent of
    # fixture prompt strings.
    model = FakeModel()
    calls = []
    def prepare(obs):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("decoder failed")
        return obs
    model.prepare = prepare
    with pytest.raises(RuntimeError, match="decoder failed"):
        job.run_arm(arm, model, kinds=["policy_shadow"], batch_size=2, checkpoint_sha="fake")
    resumed = job.run_arm(arm, FakeModel(), kinds=["policy_shadow"], batch_size=8, checkpoint_sha="fake")
    assert resumed["completed"]["policy_shadow"] == 54
    assert 0 < resumed["newly_written"]["policy_shadow"] < 54


def test_explicit_code_upgrade_keeps_parts_records_hashes_and_remains_strict(tmp_path, monkeypatch):
    arm = empty_arm(tmp_path / "input")
    actual_sha = job.code_sha
    monkeypatch.setattr(job, "code_sha", lambda: "old-code")
    job.run_arm(arm, FakeModel(), kinds=["policy_shadow"], batch_size=2, checkpoint_sha="fake", limit=3)
    before = {p: p.read_bytes() for p in (arm.debug_dir / "aug/policy_shadow").glob("*.npz")}
    monkeypatch.setattr(job, "code_sha", actual_sha)
    with pytest.raises(ValueError, match="code_sha"):
        job.run_arm(arm, FakeModel(), kinds=["policy_shadow"], checkpoint_sha="fake")
    with pytest.raises(ValueError, match="checkpoint_sha"):
        job.run_arm(arm, FakeModel(), kinds=["policy_shadow"], checkpoint_sha="other", allow_code_upgrade=True)
    assert not list((arm.debug_dir / "aug/provenance").glob("*.json"))
    result = job.run_arm(arm, FakeModel(), kinds=["policy_shadow"], checkpoint_sha="fake", allow_code_upgrade=True)
    assert result["newly_written"]["policy_shadow"] == 51
    note_path = arm.debug_dir / "aug" / result["code_upgrade_note"]
    note = json.loads(note_path.read_text())
    assert note["authorization"] == "explicit --allow-code-upgrade"
    assert note["new_code_sha"] == actual_sha()
    assert len(note["kept_parts"]) == len(before)
    for part in note["kept_parts"]:
        path = arm.debug_dir / "aug" / part["path"]
        assert part["sha256"] == hashlib.sha256(before[path]).hexdigest()
        assert path.read_bytes() == before[path]
    newer = [p for p in (arm.debug_dir / "aug/policy_shadow").glob("*.npz") if p not in before]
    assert all(json.loads(str(reader.read_npz(p, ["_meta_json"])["_meta_json"]))["code_upgrade_note"] == result["code_upgrade_note"] for p in newer)
    again = job.run_arm(arm, FakeModel(), kinds=["policy_shadow"], checkpoint_sha="fake", allow_code_upgrade=True)
    assert again["newly_written"]["policy_shadow"] == 0
    assert again["code_upgrade_note"] == result["code_upgrade_note"]


def test_code_upgrade_cannot_authorize_changed_used_fit(tmp_path, monkeypatch):
    arm = empty_arm(tmp_path / "input")
    class Retrieval(FakeRetrieval):
        def fit_specs(self, kind):
            return dict(identity=self.identity)
    retrieval = Retrieval()
    retrieval.identity = "first"
    actual_sha = job.code_sha
    monkeypatch.setattr(job, "code_sha", lambda: "old")
    job.run_arm(arm, FakeModel(), retrieval, kinds=["shadow_look"], checkpoint_sha="fake", limit=2)
    monkeypatch.setattr(job, "code_sha", actual_sha)
    retrieval.identity = "changed"
    with pytest.raises(ValueError, match="fit_config_sha"):
        job.run_arm(arm, FakeModel(), retrieval, kinds=["shadow_look"], checkpoint_sha="fake", allow_code_upgrade=True)
    assert not list((arm.debug_dir / "aug/provenance").glob("*.json"))


def test_all_kind_benchmark_cli_includes_io_and_refuses_reusing_parts(tmp_path, monkeypatch, capsys):
    arm = empty_arm(tmp_path / "input")
    args = ["job", "--run-root", str(arm.run_root), "--arms", "synthetic", "--model", "pi05",
            "--checkpoint-sha", "fake", "--fake", "--benchmark-all", "--batch-sizes", "1", "8",
            "--limit", "5", "--output-root", str(tmp_path / "bench")]
    monkeypatch.setattr("sys.argv", args)
    job.main()
    reports = json.loads(capsys.readouterr().out)
    assert len(reports) == 2
    assert all(r["newly_written"]["policy_shadow"] == 5 for r in reports)
    assert all("publish_io.policy_shadow" in r["profile"]["wall_seconds"] for r in reports)
    with pytest.raises(ValueError, match="already contains parts"):
        job.main()


def test_comparison_joins_ids_requires_seeds_and_reports_numeric_deltas(tmp_path):
    arm = empty_arm(tmp_path / "input")
    old, new = tmp_path / "old", tmp_path / "new"
    for target in (old, new):
        job.run_arm(arm, FakeModel(), kinds=["policy_shadow"], checkpoint_sha="fake", limit=5, output_dir=target)
    path = next((new / "policy_shadow").glob("*.npz"))
    data = reader.read_npz(path)
    for key in data:
        if key != "_meta_json":
            data[key] = data[key][::-1]
    data["chunk"] += 0.001
    schema.write_npz_block(path, data)
    assert compare(old, new, ["policy_shadow"], atol=0.002, rtol=0.)["status"] == "PASS"
    assert compare(old, new, ["policy_shadow"], atol=0., rtol=0.)["status"] == "FAIL"
    data["seed"][0] += 1
    schema.write_npz_block(path, data)
    assert compare(old, new, ["policy_shadow"], atol=1., rtol=1.)["status"] == "FAIL"
