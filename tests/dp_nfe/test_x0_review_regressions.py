"""Regression coverage for the Owner-authorized G2 R2 corrections."""

import copy
import json
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from exp.dp_nfe.analysis import aggregate_x0 as A
from exp.dp_nfe.analysis import dispersion_index as D
from exp.dp_nfe import x0_queue as Q
from exp.dp_nfe.eval_dp_steps_v2 import KeyedNoise
from exp.dp_nfe.x0_identity import sha256_path, verify_data_identity
from tests.dp_nfe.test_x0_aggregate import GOOD, _cells_dir, _matrix
from tests.dp_nfe.test_x0_queue import TASKS_CFG, _cells, _final, _summary


@pytest.mark.parametrize("kind", ["duplicate", "different_checkpoint", "invalid_duplicate", "missing_ladder"])
def test_conflict_or_missing_ladder_cannot_leave_a_formal_verdict(tmp_path, kind):
    cells = _cells_dir(tmp_path)
    root = tmp_path / "res"
    _matrix(root, GOOD)
    source = root / "pusht_lowdim_U_sample_s42_B50k" / "test_ddim_4" / "summary.json"
    r = json.loads(source.read_text())
    if kind == "missing_ladder":
        source.unlink()
    else:
        if kind == "different_checkpoint":
            r["manifest"]["checkpoint_sha256"] = "changed"
        if kind == "invalid_duplicate":
            r["complete"] = False
        duplicate = source.parent.parent / "duplicate"
        duplicate.mkdir()
        (duplicate / "summary.json").write_text(json.dumps(r))
    out = A.aggregate(cells, root, boot=40, seed=1)
    by = {t["task"]: t for t in out["tasks"]}
    assert by["pusht"]["verdict"] is None
    assert by["blockpush"]["verdict"] is not None
    # Conflict handling must not depend on filesystem traversal order.
    _, frozen = A.load_cells(cells)
    records = A.load_results(root)
    assert A.validate(records, frozen)[0] == A.validate(reversed(records), frozen)[0]


@pytest.mark.parametrize("mutation", [
    lambda r: r["sampler"].pop("eps_mode"),
    lambda r: r["sampler"].update(k="broken"),
    lambda r: r["sampler"].update(clip_sample=False),
    lambda r: r["sampler"].update(T=1000),
    lambda r: r["sampler"].update(sampling_seed=7),
    lambda r: r["episodes"].update({"100000": 1.1}),
    lambda r: r["cell"].update(cell_yaml_sha256="changed"),
])
def test_invalid_protocol_never_counts_as_complete(tmp_path, mutation):
    cells = _cells_dir(tmp_path)
    root = tmp_path / "res"
    _matrix(root, GOOD)
    p = root / "pusht_lowdim_U_sample_s42_B50k" / "test_ddim_1" / "summary.json"
    r = json.loads(p.read_text()); mutation(r); p.write_text(json.dumps(r))
    out = A.aggregate(cells, root, boot=40, seed=1)
    assert next(t for t in out["tasks"] if t["task"] == "pusht")["verdict"] is None
    assert out["records"]["invalid"]


def test_single_seed_manifest_and_image_formal_verdict_are_refused(tmp_path):
    cells = _cells_dir(tmp_path)
    p = cells / "cells_manifest.json"
    m = json.loads(p.read_text()); m["train_seeds"] = [42]; p.write_text(json.dumps(m))
    with pytest.raises(A.AggregationError, match="train_seeds"):
        A.aggregate(cells, tmp_path / "res", boot=40, seed=1)
    with pytest.raises(A.AggregationError, match="lowdim"):
        A.evaluate_task({}, "pusht", "image", [42], "B20k", 40, 1)


def test_json_array_is_reported_as_invalid(tmp_path):
    p = tmp_path / "c" / "test_ddim_1"; p.mkdir(parents=True)
    (p / "summary.json").write_text("[]")
    table, report = A.validate(A.load_results(tmp_path), {})
    assert not table and len(report["invalid"]) == 1


def test_queue_revalidates_done_and_legacy_gated_jobs(tmp_path):
    cells = Q.load_matrix(_cells(tmp_path, ("cA",)), ["core"])
    runs, res = tmp_path / "runs", tmp_path / "res"
    sha = _final(runs / "cA", "cA")
    _summary(res / "cA" / "screen_ddim_100", "screen", "ddim", 100, 32, sha, q=0)
    _summary(res / "cA" / "test_ddim_1", "test", "ddim", 1, 100, sha)
    p = res / "cA" / "test_ddim_1" / "summary.json"
    r = json.loads(p.read_text()); r["sampler"]["clip_sample"] = False; p.write_text(json.dumps(r))
    state = Q.QueueState(tmp_path / "q.json")
    state.set("eval/cA/test_ddim_1", status="done", attempts=1)
    state.set("eval/cA/test_ddim_4", status="gated")
    calls = []
    def run(argv, log, env):
        out = pathlib.Path(argv[argv.index("-o") + 1]); k = int(argv[argv.index("--k") + 1])
        calls.append(k); _summary(out, "test", "ddim", k, 100, sha, q=0); return 0
    def queue():
        return Q.run_eval_queue(cells, TASKS_CFG, runs, res, state, "python", "/dp", ["test"], ["ddim_1", "ddim_4"], 3, run_cmd=run)
    assert queue() == {"done": 2} and calls == [1, 4]
    calls.clear()
    assert queue() == {"done": 2} and not calls
    (runs / "cA" / "checkpoints" / "final.ckpt").write_bytes(b"tampered")
    assert queue() == {"blocked": 2} and not calls
    assert Q.summarize({"gated": 1}) == 3


def test_train_verifier_checks_frozen_fields(tmp_path):
    cells = Q.load_matrix(_cells(tmp_path, ("cA",)), ["core"])
    run = tmp_path / "runs" / "cA"
    _final(run, "cA")
    assert Q.verify_train(run, cells["cA"]) is None
    altered = copy.deepcopy(cells["cA"]); altered["head"] = "sample"
    assert "head" in Q.verify_train(run, altered)


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_frozen_data_detects_same_size_mutation(tmp_path, kind):
    path = tmp_path / "data"
    if kind == "directory":
        path.mkdir(); payload = path / "chunk"
    else:
        payload = path
    payload.write_bytes(b"original")
    ident = {"subset_path": str(path), "subset_sha256": sha256_path(path)}
    verify_data_identity(ident)
    payload.write_bytes(b"modified")
    with pytest.raises(ValueError, match="subset_sha256"):
        verify_data_identity(ident)


def test_launch_rejects_dataset_path_different_from_frozen_identity(tmp_path):
    from exp.dp_nfe.train_x0 import build_identity
    frozen = tmp_path / "frozen"; frozen.write_bytes(b"data")
    cfg = OmegaConf.create({"task": {"dataset": {"zarr_path": str(tmp_path / "other")}},
                           "x0": {"budget_steps": 10, "identity": {"subset_path": str(frozen), "variant": "full"}}})
    cell = {"cell_id": "test", "head": "epsilon", "train_seed": 42, "budget_steps": 10}
    with pytest.raises(ValueError, match="dataset path"):
        build_identity(cfg, cell, tmp_path / "unused.yaml", tmp_path)


@pytest.mark.parametrize("image", [False, True])
def test_diagnostic_policy_schema_and_clip_noise_replay(image):
    obs = {"image": torch.zeros(2, 3, 8, 8), "agent_pos": torch.zeros(2, 2)} if image else torch.zeros(4, 3)
    batched = D.batch_obs({"obs": obs}, 4, "cpu")
    assert set(batched) == ({"image", "agent_pos"} if image else {"obs"})
    keyed = KeyedNoise(0, 0, 4)
    ts = SimpleNamespace(clip=True)
    noises = []
    def predict(data):
        x = keyed.initial_noise((4, 3, 2), torch.float32, "cpu")
        noises.append(x.clone())
        x += torch.rand_like(x)  # image encoder randomness is replayed too
        return {"action": x.clamp(-1, 1) if ts.clip else x}
    policy = SimpleNamespace(device=torch.device("cpu"), predict_action=predict)
    for history in range(2):
        clipped, raw = D.paired_clip_samples(policy, batched, ts, keyed, history)
        np.testing.assert_array_equal(clipped, np.clip(raw, -1, 1))
        assert ts.clip and keyed.chunk_idx == history
    assert torch.equal(noises[0], noises[1]) and torch.equal(noises[2], noises[3])
    assert not torch.equal(noises[0], noises[2])


def test_diagnostic_zero_variance_and_action_alignment():
    p = SimpleNamespace(n_obs_steps=2, n_action_steps=3, oa_step_convention=False)
    assert D.executed_slice(p) == slice(2, 5)
    p.oa_step_convention = True
    assert D.executed_slice(p) == slice(1, 4)
    x = np.zeros((4, 3, 2)); std = np.zeros(2)
    assert D.dispersion(x, std)["mean_pairwise_l2"] == 0
    assert D.mixture_diagnostic(x, std)["delta_bic_2_vs_1"] is None
