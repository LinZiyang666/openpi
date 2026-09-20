"""CPU tests for exp/dp_nfe/x0_queue.py (G2 R1-B10) with an injected command runner: artifacts are verified before a
skip and after every attempt, retries are bounded, permanent failures end as ``failed`` (no spinning), evals of
untrained cells stay ``blocked``, test arms of a cell with screening Q < 0.5 still run, and the terminal line /
exit code never claim completion for pending work."""

import hashlib
import json
import pathlib

import numpy as np
import pytest
import yaml

from exp.dp_nfe import x0_queue as Q
from exp.dp_nfe.x0_identity import expected_identity, sha256_file
from tests.dp_nfe.test_x0_aggregate import _rec

TASKS_CFG = {"dp_root": "/dp", "eval": {"screen": {"n": 32, "start_seed": 90000}, "test_lowdim": {"n": 100, "start_seed": 100000, "n_envs": 25},
                                        "test_image": {"n": 50, "start_seed": 100000, "n_envs": 25}}}


def _cells(tmp_path, ids=("cA", "cB")):
    d = tmp_path / "cells"; d.mkdir(exist_ok=True)
    man = {"cells": list(ids), "by_arm": {"core": list(ids), "explore": [], "image": []}, "skipped": [], "budgets": {"lowdim": 10, "image": 5}, "train_seeds": [42]}
    for cid in ids:
        (d / f"{cid}.yaml").write_text(yaml.safe_dump({"cell_id": cid, "budget_steps": 10, "head": "epsilon", "train_seed": 42,
                                                       "identity": {"task_name": cid, "modality": "lowdim", "variant": "U", "budget_id": "B0k"},
                                                       "runner_overrides": ["task.env_runner.dataset_path=/x"]}))
    (d / "cells_manifest.json").write_text(json.dumps(man))
    return d


def _cell_at(path, cid):
    for parent in (path, *path.parents):
        f = parent / "cells" / f"{cid}.yaml"
        if f.is_file():
            c = yaml.safe_load(f.read_text()); c["_yaml_sha256"] = sha256_file(f)
            return c
    raise AssertionError(f"no fixture cell {cid}")


def _final(run_dir: pathlib.Path, cell_id: str, global_step=10, finished=True, sha_ok=True):
    ck = run_dir / "checkpoints"; ck.mkdir(parents=True, exist_ok=True)
    (ck / "final.ckpt").write_bytes(b"weights-" + cell_id.encode())
    sha = hashlib.sha256((ck / "final.ckpt").read_bytes()).hexdigest()
    (ck / "final.done").write_text(json.dumps({"global_step": global_step, "finished": finished, "identity": expected_identity(_cell_at(run_dir, cell_id)),
                                               "checkpoint_sha256": sha if sha_ok else "0" * 64}))
    return sha


def _summary(out_dir: pathlib.Path, split, sampler, k, n, sha, q=1.0, complete=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = [str(Q.SPLIT_SEEDS[split] + i) for i in range(n)]
    cell = _cell_at(out_dir, out_dir.parent.name)
    r = _rec(expected_identity(cell), sampler, k, split, [1.0 if j < q * n else 0.0 for j in range(n)])
    r["cell"].update(expected_identity(cell))
    r["manifest"]["checkpoint_sha256"] = sha
    r["complete"] = complete
    (out_dir / "summary.json").write_text(json.dumps(r))


# ---------------------------------------------------------------- training queue
def test_train_queue_skips_only_verified_and_bounds_retries(tmp_path):
    cells_dir = _cells(tmp_path); cells = Q.load_matrix(cells_dir, ["core"])
    runs = tmp_path / "runs"; state = Q.QueueState(tmp_path / "q.json")
    _final(runs / "cA", "cA")                                   # cA verified
    _final(runs / "cB", "cB", global_step=7, finished=False)    # cB: stale partial artifact -> must run
    calls = []
    def fake_run(argv, log, env):
        calls.append(argv); return 1                            # cB keeps failing deterministically
    counts = Q.run_train_queue(cells, cells_dir, runs, state, "python", "/openpi", "/dp", max_attempts=3, run_cmd=fake_run)
    assert counts == {"done": 1, "failed": 1}
    assert len(calls) == 3 and all("cB.yaml" in " ".join(a) for a in calls)   # never touched cA, exactly 3 attempts
    st = json.loads((tmp_path / "q.json").read_text())
    assert st["train/cA"]["status"] == "done" and st["train/cB"]["status"] == "failed" and st["train/cB"]["attempts"] == 3
    assert "max attempts" in st["train/cB"]["reason"] and "global_step 7" in st["train/cB"]["reason"]
    # a re-run of the queue does not retry a failed job (no spinning) and re-verifies cA
    calls.clear()
    counts = Q.run_train_queue(cells, cells_dir, runs, state, "python", "/openpi", "/dp", max_attempts=3, run_cmd=fake_run)
    assert counts == {"done": 1, "failed": 1} and calls == []
    assert Q.summarize(counts) == 2


def test_load_matrix_task_name_filter(tmp_path):
    cells_dir = _cells(tmp_path, ("cA", "cB", "cC"))
    assert sorted(Q.load_matrix(cells_dir, ["core"])) == ["cA", "cB", "cC"]
    assert sorted(Q.load_matrix(cells_dir, ["core"], task_names=["cA", "cC"])) == ["cA", "cC"]
    assert sorted(Q.load_matrix(cells_dir, ["core"], only=["cB"], task_names=["cB"])) == ["cB"]
    assert Q.load_matrix(cells_dir, ["core"], only=["cB"], task_names=["cA"]) == {}


def test_train_queue_succeeds_on_retry_and_verifies_sha(tmp_path):
    cells_dir = _cells(tmp_path, ("cA",)); cells = Q.load_matrix(cells_dir, ["core"])
    runs = tmp_path / "runs"; state = Q.QueueState(tmp_path / "q.json")
    n = {"i": 0}
    def fake_run(argv, log, env):
        n["i"] += 1
        if n["i"] == 1:
            _final(runs / "cA", "cA", sha_ok=False); return 0   # exit 0 but corrupt artifact -> not done
        _final(runs / "cA", "cA"); return 0
    counts = Q.run_train_queue(cells, cells_dir, runs, state, "python", "/openpi", "/dp", max_attempts=3, run_cmd=fake_run)
    assert counts == {"done": 1} and n["i"] == 2 and Q.summarize(counts) == 0
    assert state.get("train/cA")["attempts"] == 2
    assert Q.verify_train(runs / "cA", cells["cA"]) is None
    (runs / "cA" / "checkpoints" / "final.ckpt").write_bytes(b"tampered")
    assert "sha256 differs" in Q.verify_train(runs / "cA", cells["cA"])


# ---------------------------------------------------------------- evaluation queue
def test_eval_queue_blocked_weak_models_retried_and_verified(tmp_path):
    cells_dir = _cells(tmp_path, ("cA", "cB", "cC")); cells = Q.load_matrix(cells_dir, ["core"])
    runs = tmp_path / "runs"; res = tmp_path / "res"; state = Q.QueueState(tmp_path / "q.json")
    shaA = _final(runs / "cA", "cA"); shaB = _final(runs / "cB", "cB")        # cC never trained
    samplers = ["ddim_100", "ddim_1"]
    attempts = []
    def fake_run(argv, log, env):
        i = argv.index("-o"); out = pathlib.Path(argv[i + 1]); split = argv[argv.index("--split") + 1]
        sampler = argv[argv.index("--sampler") + 1]; k = int(argv[argv.index("--k") + 1]); n = int(argv[argv.index("--n-test") + 1])
        attempts.append((out.parent.name, out.name, env.get("CUDA_VISIBLE_DEVICES")))
        cell = out.parent.name
        if cell == "cA" and split == "screen":
            _summary(out, split, sampler, k, n, shaA, q=1.0)                     # passes the gate
        elif cell == "cA" and sampler == "ddim" and k == 1:
            if sum(1 for a in attempts if a[0] == "cA" and a[1] == "test_ddim_1") < 2:
                _summary(out, split, sampler, k, n, shaA, complete=False)        # first attempt: crashed mid-way
            else:
                _summary(out, split, sampler, k, n, shaA)
        elif cell == "cA":
            _summary(out, split, sampler, k, n, "WRONGSHA")                       # stale checkpoint -> never verifies
        elif cell == "cB":
            _summary(out, split, sampler, k, n, shaB, q=0.25)                     # fails the gate
        return 0
    counts = Q.run_eval_queue(cells, TASKS_CFG, runs, res, state, "python", "/dp", ["screen", "test"], samplers,
                              max_attempts=3, parallel=2, gpus=["0", "1"], run_cmd=fake_run)
    st = {k: v["status"] for k, v in state.jobs.items()}
    assert st["eval/cA/screen_ddim_100"] == "done"
    assert st["eval/cA/test_ddim_1"] == "done" and state.get("eval/cA/test_ddim_1")["attempts"] == 2
    assert st["eval/cA/test_ddim_100"] == "failed" and state.get("eval/cA/test_ddim_100")["attempts"] == 3
    assert "sha256 differs" in state.get("eval/cA/test_ddim_100")["reason"]
    assert st["eval/cB/screen_ddim_100"] == "done" and st["eval/cB/test_ddim_100"] == "done" and st["eval/cB/test_ddim_1"] == "done"
    assert st["eval/cC/screen_ddim_100"] == "blocked" and st["eval/cC/test_ddim_1"] == "blocked"
    assert counts == {"done": 5, "failed": 1, "blocked": 3}
    assert Q.summarize(counts) == 2
    assert {a[2] for a in attempts} <= {"0", "1"}
    # the queue never launched a test arm before its screening result existed
    order = [a for a in attempts if a[0] == "cA"]
    assert order[0][1] == "screen_ddim_100"
    # training cC later -> its jobs become runnable; a second pass leaves verified done/failed untouched
    shaC = _final(runs / "cC", "cC"); before = len(attempts)
    def fake_run2(argv, log, env):
        out = pathlib.Path(argv[argv.index("-o") + 1]); split = argv[argv.index("--split") + 1]
        sampler = argv[argv.index("--sampler") + 1]; k = int(argv[argv.index("--k") + 1]); n = int(argv[argv.index("--n-test") + 1])
        attempts.append((out.parent.name, out.name, None)); _summary(out, split, sampler, k, n, shaC); return 0
    counts = Q.run_eval_queue(cells, TASKS_CFG, runs, res, state, "python", "/dp", ["screen", "test"], samplers,
                              max_attempts=3, parallel=1, run_cmd=fake_run2)
    assert counts == {"done": 8, "failed": 1} and all(a[0] == "cC" for a in attempts[before:])


def test_eval_verification_rejects_wrong_arm_ids_and_incomplete(tmp_path):
    cells = Q.load_matrix(_cells(tmp_path, ("cA",)), ["core"])
    cell = cells["cA"]
    d = tmp_path / "res" / "cA" / "test_ddim_1"
    _summary(d, "test", "ddim", 1, 100, "S")
    assert Q.verify_eval(d, "test", "ddim", 1, 100, "S", cell) is None
    assert "another arm" in Q.verify_eval(d, "test", "ddim", 4, 100, "S", cell)
    assert "episode id set" in Q.verify_eval(d, "test", "ddim", 1, 50, "S", cell)
    assert "sha256" in Q.verify_eval(d, "test", "ddim", 1, 100, "T", cell)
    _summary(d, "test", "ddim", 1, 100, "S", complete=False)
    assert "not complete" in Q.verify_eval(d, "test", "ddim", 1, 100, "S", cell)
    (d / "summary.json").write_text("{not json")
    assert "unreadable" in Q.verify_eval(d, "test", "ddim", 1, 100, "S", cell)
    assert Q.verify_eval(tmp_path / "missing", "test", "ddim", 1, 100, "S", cell) == "summary.json missing"


def test_summarize_exit_codes_and_status_mode(tmp_path, capsys):
    assert Q.summarize({"done": 3}) == 0 and "QUEUE DONE" in capsys.readouterr().out
    assert Q.summarize({"done": 3, "gated": 1}) == 3
    assert Q.summarize({"done": 3, "pending": 1}) == 3 and "QUEUE INCOMPLETE" in capsys.readouterr().out
    assert Q.summarize({"done": 3, "blocked": 1}) == 3
    assert Q.summarize({"failed": 1, "pending": 4}) == 2
    cells_dir = _cells(tmp_path, ("cA",)); tasks = tmp_path / "tasks.yaml"; tasks.write_text(yaml.safe_dump(TASKS_CFG))
    rc = Q.main(["status", "--tasks", str(tasks), "--cells", str(cells_dir), "--runs", str(tmp_path / "runs"), "--results", str(tmp_path / "res"),
                 "--state", str(tmp_path / "q.json"), "--samplers", "ddim_100,ddim_1"])
    out = capsys.readouterr().out
    assert rc == 3 and "pending" in out and "train/cA" in out and "eval/cA/test_ddim_1" in out
