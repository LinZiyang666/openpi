"""The GR00T CP2 island tooling end to end on a stub model: build the CP2 library
from a W13-shaped CP1 source + H5 root, verify it, build the shadow table over an
accepted cohort, and run the decision-overhead preflight -- with every fail-closed
branch the plan names (schedule stamps, model binding, accepted-manifest gate,
sample count, probe wiring, verdict).
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import pickle

import h5py
import numpy as np
import pytest
import torch

from exp.actioncache_baseline import libs
from exp.actioncache_baseline.bench_cp2_overhead import verdict_for
from exp.actioncache_baseline.verify_cp2_artifact import VerificationError, verify
from exp.libero_groot import bench_cp2_overhead_groot as bench
from exp.libero_groot import build_cp2_artifact_groot as builder_mod
from exp.libero_groot import build_shadow_table_groot as shadow_mod
from exp.libero_groot.cp2_reconstruct import STAGE1_PATH, build_template
from openpi.cache.groot.staged import GrootStagedRunner
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID
from tests.cache.groot.conftest import EMB_DIM, STATE_FEAT_DIM, STATE_VALID, TOKENS_PER_IMAGE

K8 = "groot_n15_k8_v1"
SNAPS = [i / 8 for i in range(1, 8)]
TASKS = ["open the drawer", "put the bowl on the plate"]
STEPS_PER_EPISODE = 30
SEED = 20260904
D = 8
TOKEN_LEN = 640


def _write_h5(path: pathlib.Path, policy, runner, task: str, n_steps: int, rng) -> None:
    with runner.session():
        template = build_template(policy, runner, task)
    text = template.input_embeds[0][~template.image_token_mask[0]].float().to(torch.float16).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["task"] = task
        f.attrs["success"] = True
        f.attrs["num_steps"] = n_steps
        f.attrs["denoise_schedule_id"] = K8
        f.attrs["denoising_num_steps"] = 8
        for i in range(n_steps):
            g = f.create_group(f"step_{i:04d}")
            g.create_dataset("vision_0", data=rng.standard_normal((TOKENS_PER_IMAGE, EMB_DIM)).astype(np.float16))
            g.create_dataset("vision_1", data=rng.standard_normal((TOKENS_PER_IMAGE, EMB_DIM)).astype(np.float16))
            g.create_dataset("prompt_emb", data=text)
            g.create_dataset("robot_state", data=rng.uniform(-1, 1, STATE_VALID).astype(np.float32))
            g.create_dataset("clean_action", data=np.zeros((16, 32), np.float16))
            for k in range(8):
                g.create_dataset(f"noise_action_{k}", data=np.zeros((16, 32), np.float16))


def _source_pkl(path: pathlib.Path, episodes: list[tuple[str, str]], *, schedule=K8, warm_ts=SNAPS) -> pathlib.Path:
    entries = []
    for stem, task in episodes:
        ids = [f"{stem}:{i}" for i in range(STEPS_PER_EPISODE)]
        for i, eid in enumerate(ids):
            entries.append(CacheEntry(
                id=eid, checkpoint_id=CheckpointID.CP1, query_keys={"vision_0": torch.zeros(4)},
                payload=CachePayload(action_chunk=torch.full((16, 32), float(i)),
                                     intermediates={t: torch.full((16, 32), t) for t in warm_ts},
                                     denoising_num_steps=8, task_key=task, schedule_id=schedule),
                step_idx=i, trajectory_id=stem, prev_ids=[ids[i - 1]] if i else [], next_ids=[ids[i + 1]] if i + 1 < len(ids) else [],
            ))
    art = {"key_builder_type": "cp1_groot_libero_spatial_pool_16", "checkpoint_id": "CP1",
           "vector_dims": {"vision_0": 4}, "schedule_id": schedule, "entries": entries,
           "library_stats": {"n": len(entries)}, "prompt_pool": {"tasks": sorted({t for _, t in episodes})}}
    with open(path, "wb") as f:
        pickle.dump(art, f)
    return path


@pytest.fixture
def island(tmp_path, stub_island_policy):
    policy = stub_island_policy
    runner = GrootStagedRunner(policy.model, verify_upstream=False)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"stub weights " * 100)
    rng = np.random.default_rng(0)
    h5_root = tmp_path / "h5"
    episodes = [("episode_0000_ts", TASKS[0]), ("episode_0001_ts", TASKS[1])]
    for stem, task in episodes:
        _write_h5(h5_root / f"{stem}.h5", policy, runner, task, STEPS_PER_EPISODE, rng)
    src = _source_pkl(tmp_path / "w13_S3.pkl", episodes)
    return argparse.Namespace(policy=policy, runner=runner, ckpt=ckpt, h5_root=h5_root, src=src,
                              episodes=episodes, tmp=tmp_path, rng=rng)


def _build_args(island, **over):
    d = dict(source_pkl=str(island.src), h5_root=str(island.h5_root), out_pkl=str(island.tmp / "cp2.pkl"),
             checkpoint=str(island.ckpt), device="cpu", denoising_steps=8, seed=SEED, d=D, p=0.5,
             token_len=TOKEN_LEN, feature_dim=EMB_DIM, state_feat_dim=STATE_FEAT_DIM, limit=0)
    d.update(over)
    return argparse.Namespace(**d)


def _accepted_manifest(island, path: pathlib.Path, *, ok=True, episodes=None, task_map_bound=True) -> pathlib.Path:
    """A ``verify_shadow_h5`` acceptance record over the island's episodes (canonical name + instruction)."""
    eps = island.episodes if episodes is None else episodes
    rows = []
    for i, (stem, task) in enumerate(eps):
        h5 = island.h5_root / f"{stem}.h5"
        rows.append({"task_id": i, "task_name": task.replace(" ", "_"), "task_language": task, "subset_init_state_idx": 0,
                     "orig_init_state_idx": 3 + i, "attempt": "attempt_0", "h5": str(h5), "h5_sha256": libs.sha256_file(h5),
                     "success": True})
    path.write_text(json.dumps({"suite": "libero_spatial", "ok": ok, "expected": len(rows), "n_accepted": len(rows),
                                "task_map_bound": task_map_bound, "task_map_sha256": "4" * 64, "accepted": rows}),
                    encoding="utf-8")
    return path


def test_build_verify_shadow_and_preflight_on_the_stub(island, tmp_path):
    record = builder_mod.build(_build_args(island))
    assert record["n_entries"] == 2 * STEPS_PER_EPISODE
    assert record["schedule_id"] == K8 and record["teacher"] == "groot_libero" and record["stage1_path"] == STAGE1_PATH
    assert record["projection"]["layout"] == {"kind": "groot_encoded_v1", "token_len": TOKEN_LEN,
                                              "feature_dim": EMB_DIM, "state_feat_dim": STATE_FEAT_DIM}
    assert record["model"]["weights_digest"] == libs.weights_digest(island.ckpt)["weights_digest"]
    assert record["source_key_builder_type"] == "cp1_groot_libero_spatial_pool_16"
    assert len(record["h5_manifest"]["files"]) == 2 and record["tokenizer"]["n_templates"] == 2
    art = libs.load_pickle(island.tmp / "cp2.pkl")
    assert art["library_stats"] == {"n": 60} and art["prompt_pool"]["tasks"] == sorted(TASKS)
    keys = torch.stack([torch.as_tensor(np.asarray(e.query_keys[libs.FIELD])) for e in art["entries"]])
    assert keys.shape == (60, D) and keys.dtype is torch.float32
    assert len({tuple(k.tolist()) for k in keys}) == 60  # every step got its own key

    # The shared verifier under the GR00T profile: one-to-one copy, schedule identity, backend round trip.
    report = verify(str(island.tmp / "cp2.pkl"), str(island.src), teacher="groot_libero", search_samples=10)
    assert report["ok"] and report["action_chunk_shape"] == (16, 32) and report["n_entries"] == 60
    with pytest.raises(VerificationError, match="key_builder_type"):
        verify(str(island.tmp / "cp2.pkl"), str(island.src), teacher="pi05")

    # Rebuilding is deterministic (fixed projection seed, deterministic stub).
    record2 = builder_mod.build(_build_args(island, out_pkl=str(island.tmp / "cp2b.pkl")))
    art2 = libs.load_pickle(island.tmp / "cp2b.pkl")
    keys2 = torch.stack([torch.as_tensor(np.asarray(e.query_keys[libs.FIELD])) for e in art2["entries"]])
    assert torch.equal(keys, keys2) and record2["projection"] == record["projection"]

    # Shadow table over an accepted cohort: library steps self-match at cosine 1.
    acc = _accepted_manifest(island, tmp_path / "accepted_shadow_manifest.json")
    shadow_args = argparse.Namespace(suite="libero_spatial", accepted_manifest=str(acc), library_pkl=str(island.tmp / "cp2.pkl"),
                                     checkpoint=str(island.ckpt), device="cpu", denoising_steps=8,
                                     out_jsonl=str(tmp_path / "shadow" / "rows.jsonl"), backend_check=50, limit_episodes=0)
    srec = shadow_mod.build(shadow_args)
    rows = [json.loads(line) for line in (tmp_path / "shadow" / "rows.jsonl").read_text().splitlines()]
    assert srec["n_rows"] == len(rows) == 60 and srec["backend_checked_rows"] == 50
    assert all(abs(r["s_raw"] - 1.0) < 1e-5 and r["winner_id"] == f"{r['episode']}:{r['step_idx']}" for r in rows)
    assert set(rows[0]) == {"episode", "task", "task_id", "subset", "orig", "step_idx", "s_raw", "winner_id", "success"}
    assert srec["teacher"] == "groot_libero" and srec["stage1_path"] == STAGE1_PATH
    assert srec["model"]["bound_to_library"] and srec["accepted_manifest_sha256"] == libs.sha256_file(acc)
    assert srec["complete"] and not srec["limited"] and srec["cohort_expected"] == 2 and srec["task_map_sha256"] == "4" * 64
    assert srec["out_jsonl_sha256"] == libs.sha256_file(tmp_path / "shadow" / "rows.jsonl")
    # A debug run over a subset is marked so the exporter refuses it.
    lrec = shadow_mod.build(argparse.Namespace(**{**vars(shadow_args), "out_jsonl": str(tmp_path / "shadow" / "lim.jsonl"),
                                                   "limit_episodes": 1}))
    assert lrec["limited"] and not lrec["complete"] and lrec["cohort_episodes"] == 1

    # Decision-overhead preflight: schema, boundary, probes, verdict.
    out_dir = tmp_path / "overhead"
    bargs = argparse.Namespace(mode="overhead", suite="libero_spatial", checkpoint=str(island.ckpt), device="cpu",
                               denoising_steps=8, library_pkl=str(island.tmp / "cp2.pkl"), accepted_manifest=str(acc),
                               out_dir=str(out_dir), max_decisions=2000, cold=50)
    brec = bench.run_overhead(bargs)
    assert brec["n_decisions"] == 60 and brec["cold_decisions"] == 50 and brec["warm"]["count"] == 10
    assert brec["verdict"] in ("ok_report", "report_with_caption", "halt_profile_segments")
    assert brec["verdict"] == verdict_for(brec["warm"]["p95"])
    assert brec["timer_enabled"] and brec["stage1_path"] == STAGE1_PATH and brec["schedule_id"] == K8
    assert brec["library_sha256"] == libs.sha256_file(island.tmp / "cp2.pkl") and brec["library_entries"] == 60
    assert brec["accepted_manifest_sha256"] == libs.sha256_file(acc) and brec["projection"] == art["projection"]
    assert brec["model"]["bound"]["weights_digest"] == libs.weights_digest(island.ckpt)["weights_digest"]
    assert brec["record_kind"] == "cp2_decision_overhead" and brec["teacher"] == "groot_libero"
    for seg in bench.CORE_SEGMENTS_GROOT:
        assert brec["per_segment"][seg]["count"] == 60
    assert brec["per_segment"]["cp2_fetch"]["count"] == 60  # every decision is a self-hit here
    with (out_dir / "per_decision.csv").open() as fh:
        header, *lines = list(csv.reader(fh))
    assert header[:4] == ["episode", "step_idx", "total_ms", "check_total_ms"] and "cp2_encode_ms" in header
    assert len(lines) == 60 and all(float(r[2]) >= float(r[3]) > 0 for r in lines)
    assert (out_dir / "bench_internal_cp2_n0.yaml").exists()
    assert json.loads((out_dir / "overhead.json").read_text())["record_kind"] == "cp2_decision_overhead"


def test_builder_refuses_schedule_drift_and_missing_steps(island):
    bad = _source_pkl(island.tmp / "k4.pkl", island.episodes, schedule="groot_n15_k4_v1", warm_ts=[0.25, 0.5, 0.75])
    with pytest.raises(SystemExit, match="schedule_id"):
        builder_mod.build(_build_args(island, source_pkl=str(bad)))
    no_warm = _source_pkl(island.tmp / "nowarm.pkl", island.episodes, warm_ts=[0.125, 0.5])
    with pytest.raises(SystemExit, match="no snapshot at start_t=0.875"):
        builder_mod.build(_build_args(island, source_pkl=str(no_warm)))
    with h5py.File(island.h5_root / "episode_0001_ts.h5", "a") as f:
        f.attrs["denoising_num_steps"] = 4
    with pytest.raises(SystemExit, match="denoising_num_steps=4"):
        builder_mod.build(_build_args(island))
    with h5py.File(island.h5_root / "episode_0001_ts.h5", "a") as f:
        f.attrs["denoising_num_steps"] = 8
        del f["step_0029"]
    with pytest.raises(SystemExit, match=r"steps \[29\]"):
        builder_mod.build(_build_args(island))


def test_builder_refuses_a_sequence_longer_than_the_layout(island):
    with pytest.raises(RuntimeError, match="exceed token_len"):
        builder_mod.build(_build_args(island, token_len=300))


def test_shadow_and_preflight_gates(island, tmp_path):
    builder_mod.build(_build_args(island))
    lib = str(island.tmp / "cp2.pkl")
    acc = _accepted_manifest(island, tmp_path / "acc.json")
    base = dict(suite="libero_spatial", accepted_manifest=str(acc), library_pkl=lib, checkpoint=str(island.ckpt),
                device="cpu", denoising_steps=8)
    shadow = lambda **o: shadow_mod.build(argparse.Namespace(**{**base, "out_jsonl": str(tmp_path / "s.jsonl"),  # noqa: E731
                                                                 "backend_check": 50, "limit_episodes": 0, **o}))
    # accepted-manifest gate
    not_ok = _accepted_manifest(island, tmp_path / "notok.json", ok=False)
    with pytest.raises(SystemExit, match="not an accepted, complete cohort"):
        shadow(accepted_manifest=str(not_ok))
    with pytest.raises(SystemExit, match="not an accepted, complete cohort"):
        shadow(suite="libero_10")
    unbound = _accepted_manifest(island, tmp_path / "unbound.json", task_map_bound=False)
    with pytest.raises(SystemExit, match="without the benchmark task map"):
        shadow(accepted_manifest=str(unbound))
    # The H5 instruction must be the accepted record's task_language (not its canonical task_name).
    wrong_lang = _accepted_manifest(island, tmp_path / "lang.json")
    doc = json.loads(wrong_lang.read_text())
    doc["accepted"][0]["task_language"] = doc["accepted"][0]["task_name"]
    wrong_lang.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="instruction .* != accepted"):
        shadow(accepted_manifest=str(wrong_lang))
    # model binding
    other = island.tmp / "other_ckpt"
    other.mkdir()
    (other / "model.safetensors").write_bytes(b"different")
    with pytest.raises(SystemExit, match="model binding failed"):
        shadow(checkpoint=str(other))
    # H5 changed since acceptance
    with h5py.File(island.h5_root / "episode_0000_ts.h5", "a") as f:
        f.attrs["touched"] = 1
    with pytest.raises(SystemExit, match="sha256 changed since acceptance"):
        shadow()
    acc = _accepted_manifest(island, tmp_path / "acc2.json")
    # Pi0.5-shaped library refused
    with pytest.raises(SystemExit, match="not a groot_libero CP2 library"):
        shadow(accepted_manifest=str(acc), library_pkl=str(island.src))

    bench_args = lambda **o: argparse.Namespace(**{**base, "accepted_manifest": str(acc), "mode": "overhead",  # noqa: E731
                                                   "out_dir": str(tmp_path / "ov"), "max_decisions": 2000, "cold": 50, **o})
    with pytest.raises(SystemExit, match="only 30 decisions"):
        bench.run_overhead(bench_args(max_decisions=30))
    with pytest.raises(SystemExit, match="not an accepted, task-map-bound cohort"):
        bench.run_overhead(bench_args(accepted_manifest=str(not_ok)))
    with pytest.raises(SystemExit, match="not a GR00T CP2 library"):
        bench.run_overhead(bench_args(library_pkl=str(island.src)))


def test_preflight_halts_when_a_probe_is_not_wired(island, tmp_path, monkeypatch):
    builder_mod.build(_build_args(island))
    acc = _accepted_manifest(island, tmp_path / "acc.json")
    monkeypatch.setattr(bench, "CORE_SEGMENTS_GROOT", (*bench.CORE_SEGMENTS_GROOT, "cp2_not_a_probe"))
    args = argparse.Namespace(mode="overhead", suite="libero_spatial", checkpoint=str(island.ckpt), device="cpu",
                              denoising_steps=8, library_pkl=str(island.tmp / "cp2.pkl"), accepted_manifest=str(acc),
                              out_dir=str(tmp_path / "ov"), max_decisions=2000, cold=50)
    with pytest.raises(SystemExit, match=r"recorded no \['cp2_not_a_probe'\] probe"):
        bench.run_overhead(args)


@pytest.mark.parametrize("p95, verdict", [(9.9, "ok_report"), (10.0, "ok_report"), (25.0, "report_with_caption"),
                                          (40.0, "report_with_caption"), (40.1, "halt_profile_segments"),
                                          (None, "insufficient_decisions")])
def test_verdict_rule(p95, verdict):
    assert verdict_for(p95) == verdict


# ------------------------------------------------------------------
# encoder-cost certification (trace parsing only; the measurement needs the GPU)
# ------------------------------------------------------------------


def _raw_encoder_record(tmp_path, **over):
    rec = {"suite": "libero_spatial", "schedule_id": K8, "prompt_sha256": "p" * 64, "mode": "reduce-overhead",
           "warmup": 30, "iters": 200, "n_tokens": 566, "debug_sampling": False, "graphs_per_call": 3,
           "expected_cudagraph_launch_count": 600, "ckpt_weights_digest": "c" * 64, "gpu_uuid": "GPU-x", "ts": "t",
           "cp2_key_encoder_ms": 1.5, "certified": False, "valid": False, "void_reasons": [bench.UNCERTIFIED]}
    rec.update(over)
    rec["trace_marker"] = bench.encoder_trace_marker(rec)
    p = tmp_path / "enc.json"
    p.write_text(json.dumps(rec))
    return p


def _trace_csv(tmp_path, marker, launches, captures=0):
    lines = ["Time (%),Total Time (ns),Num Calls,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name",
             f"50.0,1000,{launches},1,1,1,1,0,cudaGraphLaunch",
             f"1.0,10,{captures},1,1,1,1,0,cudaStreamBeginCapture_v10000" if captures else "1.0,10,5,1,1,1,1,0,cudaMemcpyAsync",
             "", "Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Style,Range",
             f"100.0,1,1,1,1,1,1,0,PushPop,:{marker}"]
    p = tmp_path / "trace.csv"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_certify_encoder_needs_the_frozen_sampling_and_a_matching_trace(tmp_path):
    rec_path = _raw_encoder_record(tmp_path, warmup=5, iters=20, debug_sampling=True)
    with pytest.raises(SystemExit, match="debug measurement cannot be certified"):
        bench.certify_encoder(argparse.Namespace(out=str(rec_path), cuda_trace=str(tmp_path / "none.csv")))
    rec_path = _raw_encoder_record(tmp_path, n_tokens=512)  # a debug shape not flagged by the writer
    with pytest.raises(SystemExit, match="not the frozen"):
        bench.certify_encoder(argparse.Namespace(out=str(rec_path), cuda_trace=str(tmp_path / "none.csv")))
    rec_path = _raw_encoder_record(tmp_path)
    marker = json.loads(rec_path.read_text())["trace_marker"]
    trace = _trace_csv(tmp_path, marker, launches=600)
    out = bench.certify_encoder(argparse.Namespace(out=str(rec_path), cuda_trace=str(trace)))
    assert out["certified"] and out["valid"] and out["cudagraph_launch_count"] == 600 and out["void_reasons"] == []
    # Already certified records are not re-certified; a short trace voids the record.
    with pytest.raises(SystemExit, match="not a raw uncertified"):
        bench.certify_encoder(argparse.Namespace(out=str(rec_path), cuda_trace=str(trace)))
    rec_path = _raw_encoder_record(tmp_path)
    marker = json.loads(rec_path.read_text())["trace_marker"]
    with pytest.raises(SystemExit):
        bench.certify_encoder(argparse.Namespace(out=str(rec_path), cuda_trace=str(_trace_csv(tmp_path, marker, launches=599))))
    voided = json.loads(rec_path.read_text())
    assert not voided["certified"] and any("mismatch" in r for r in voided["void_reasons"])
