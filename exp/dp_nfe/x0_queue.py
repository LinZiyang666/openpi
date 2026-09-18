"""Bounded-retry job queue of the x0-head experiment (training cells and the evaluation ladder), driven by the frozen
cell matrix and verified artifacts instead of file existence (G2 R1-B10).

    python -m exp.dp_nfe.x0_queue train --tasks exp/dp_nfe/config/x0_multimodal/tasks.yaml --cells <cells dir> \
        --runs <runs root> --state <state.json> [--arms core,explore,image] [--only cell_id,...] [--max-attempts 3]
    python -m exp.dp_nfe.x0_queue eval  --tasks ... --cells <cells dir> --runs <runs root> --results <results root> \
        --state <state.json> [--arms core,...] [--splits screen,test] [--samplers ddpm_100,ddim_100,...] [--parallel 2] [--gpus 0]
    python -m exp.dp_nfe.x0_queue status --tasks ... --cells <cells dir> --runs <runs root> --results <results root> --state <state.json>

Job states: ``pending`` (never ran or retry allowed), ``done`` (artifact verified), ``failed`` (max attempts reached or a
permanent error), ``blocked`` (training or screening artifacts are not verified yet). Screening scores never suppress
test evaluation; legacy ``gated`` states are requeued. A job is *done* only when its artifact
verifies: training -> ``checkpoints/final.done`` with ``global_step == budget``, ``finished``, matching ``cell_id`` and
the recorded sha256 of ``final.ckpt``; evaluation -> ``summary.json`` complete, without error, with the exact frozen
episode-id set of its split and the checkpoint sha256 of the verified ``final.ckpt``. Verification happens before every
skip and after every attempt; a stale or corrupt artifact is treated as absent.

Retries: at most ``--max-attempts`` (default 3 = the plan's "two retries with the same ids") per job, then ``failed``.
The queue never spins: it exits 0 with ``QUEUE DONE`` only when every selected job is done, otherwise it prints
``QUEUE INCOMPLETE`` with the counts and exits 2 (failures) or 3 (pending/blocked). The actual command runner is injected
(``run_cmd``) so the queue logic is testable without a GPU, tmux or the DP env.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence

import yaml

from exp.dp_nfe.x0_identity import cell_key, expected_identity, frozen_identity_diff, sha256_file

DEFAULT_SAMPLERS = ("ddpm_100", "ddim_100", "ddim_10", "ddim_4", "ddim_2", "ddim_1")
SPLIT_SEEDS = {"screen": 90000, "test": 100000}

RunCmd = Callable[[List[str], pathlib.Path, Dict[str, str]], int]


def default_run_cmd(argv: List[str], log: pathlib.Path, env: Dict[str, str]) -> int:
    """Run ``argv`` with stdout+stderr appended to ``log``; returns the exit code (never raises)."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as lf:
        lf.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(argv)}\n".encode())
        lf.flush()
        try:
            return subprocess.run(argv, stdout=lf, stderr=subprocess.STDOUT, env={**os.environ, **env}).returncode
        except Exception as e:  # noqa: BLE001
            lf.write(f"launcher error: {e}\n".encode())
            return 127


# ----------------------------------------------------------------------------- artifact verification
def verify_train(run_dir: pathlib.Path, cell: dict, check_sha: bool = True) -> Optional[str]:
    """``None`` when ``run_dir`` holds a verified final checkpoint of ``cell``; otherwise the reason it does not."""
    done = run_dir / "checkpoints" / "final.done"
    ckpt = run_dir / "checkpoints" / "final.ckpt"
    if not done.is_file() or not ckpt.is_file():
        return "final.done/final.ckpt missing"
    try:
        d = json.loads(done.read_text())
        if not isinstance(d, dict) or not isinstance(d.get("identity"), dict):
            raise ValueError("final.done must contain an identity object")
    except Exception as e:  # noqa: BLE001
        return f"final.done unreadable: {e}"
    if d.get("global_step") != int(cell["budget_steps"]):
        return f"global_step {d.get('global_step')} != budget {cell['budget_steps']}"
    if d.get("finished") is not True:
        return "final.done without finished=true"
    differences = frozen_identity_diff(d.get("identity") or {}, cell)
    if differences:
        return "final.done frozen identity differs: " + "; ".join(differences)
    if not d.get("checkpoint_sha256"):
        return "final.done without checkpoint_sha256"
    if check_sha and sha256_file(ckpt) != d["checkpoint_sha256"]:
        return "final.ckpt sha256 differs from final.done"
    return None


def final_sha(run_dir: pathlib.Path) -> Optional[str]:
    """``checkpoint_sha256`` recorded in ``final.done`` (None when unreadable)."""
    try:
        return json.loads((run_dir / "checkpoints" / "final.done").read_text()).get("checkpoint_sha256")
    except Exception:  # noqa: BLE001
        return None


def verify_eval(out_dir: pathlib.Path, split: str, sampler: str, k: int, n_test: int, ckpt_sha: Optional[str],
                cell: dict) -> Optional[str]:
    """``None`` when ``out_dir/summary.json`` is a complete, protocol-consistent result of this job; else the reason."""
    p = out_dir / "summary.json"
    if not p.is_file():
        return "summary.json missing"
    try:
        s = json.loads(p.read_text())
        if not isinstance(s, dict):
            raise ValueError("summary must be a JSON object")
    except Exception as e:  # noqa: BLE001
        return f"summary.json unreadable: {e}"
    if s.get("complete") is not True or s.get("error"):
        return "not complete" if not s.get("error") else f"error: {str(s['error'])[:60]}"
    sm = s.get("sampler") or {}
    if not isinstance(sm, dict) or not isinstance(s.get("episodes"), dict) or not isinstance(s.get("manifest"), dict):
        return "malformed sampler/episodes/manifest"
    if sm.get("sampler") != sampler or sm.get("k") != int(k) or s.get("split") != split:
        return "summary belongs to another arm/split"
    ids = [str(SPLIT_SEEDS[split] + i) for i in range(int(n_test))]
    if sorted(s.get("episodes") or {}) != sorted(ids):
        return "episode id set differs from the frozen set"
    if ckpt_sha and (s.get("manifest") or {}).get("checkpoint_sha256") != ckpt_sha:
        return "checkpoint sha256 differs from the verified final.ckpt"
    from exp.dp_nfe.analysis.aggregate_x0 import check_record
    cid = cell_key(expected_identity(cell))
    checked_id, why = check_record(s, {cid: cell})
    if checked_id != cid or why:
        return why or "summary belongs to another cell"
    return None


# ----------------------------------------------------------------------------- state
class QueueState:
    """Persistent per-job ledger ``{job: {"status", "attempts", "rc", "reason", "updated"}}`` in a json file."""

    def __init__(self, path: pathlib.Path):
        self.path = path
        self.jobs: Dict[str, dict] = {}
        self._lock = threading.Lock()
        if path.is_file():
            self.jobs = json.loads(path.read_text())

    def get(self, job: str) -> dict:
        """The (possibly new, pending) record of ``job``."""
        with self._lock:
            return self.jobs.setdefault(job, {"status": "pending", "attempts": 0, "rc": None, "reason": None})

    def set(self, job: str, **kw) -> None:
        """Update fields of ``job`` and persist the ledger atomically."""
        rec = self.get(job)
        with self._lock:
            rec.update(kw); rec["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self._save_locked()

    def save(self) -> None:
        """Persist the ledger (tmp file + rename)."""
        with self._lock:
            self._save_locked()

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.jobs, indent=1)); os.replace(tmp, self.path)

    def counts(self, jobs: Sequence[str]) -> Dict[str, int]:
        """``{status: count}`` over ``jobs``."""
        c: Dict[str, int] = {}
        for j in jobs:
            st = self.get(j)["status"]; c[st] = c.get(st, 0) + 1
        return c


# ----------------------------------------------------------------------------- job construction
def load_matrix(cells_dir: pathlib.Path, arms: Sequence[str], only: Sequence[str] = ()) -> Dict[str, dict]:
    """``{cell_id: cell yaml}`` of the selected arms (in manifest order), optionally restricted to ``only``."""
    man = json.loads((cells_dir / "cells_manifest.json").read_text())
    out = {}
    for arm in arms:
        for cid in man["by_arm"][arm]:
            if only and cid not in only:
                continue
            out[cid] = yaml.safe_load((cells_dir / f"{cid}.yaml").read_text())
            out[cid]["_yaml_sha256"] = sha256_file(cells_dir / f"{cid}.yaml")
    return out


def train_argv(python: str, openpi_root: str, dp_root: str, cell_yaml: pathlib.Path, run_dir: pathlib.Path) -> List[str]:
    """argv of one training launch (``python -m exp.dp_nfe.train_x0 ...``)."""
    return [python, "-m", "exp.dp_nfe.train_x0", "--dp-root", dp_root, "--cell", str(cell_yaml), "--out", str(run_dir)]


def eval_argv(python: str, dp_root: str, ckpt: pathlib.Path, out_dir: pathlib.Path, split: str, sampler: str, k: int,
              n_test: int, n_envs: int, attempt: int, runner_overrides: Sequence[str]) -> List[str]:
    """argv of one evaluation launch (``python -m exp.dp_nfe.eval_dp_steps_v2 ...`` with the runner overrides of the cell)."""
    argv = [python, "-m", "exp.dp_nfe.eval_dp_steps_v2", "--dp-root", dp_root, "-c", str(ckpt), "-o", str(out_dir),
            "--sampler", sampler, "--k", str(k), "--split", split, "--n-test", str(n_test), "--n-envs", str(n_envs),
            "--attempt", str(attempt)]
    for ov in runner_overrides:
        argv += ["--runner-override", ov]
    return argv


def eval_sizes(tasks_cfg: dict, modality: str, split: str) -> Dict[str, int]:
    """``{n_test, n_envs}`` of a split from the ``eval`` block of tasks.yaml (image test uses ``test_image``)."""
    ev = tasks_cfg["eval"]
    if split == "screen":
        return {"n_test": int(ev["screen"]["n"]), "n_envs": int(ev["screen"].get("n_envs", ev["screen"]["n"]))}
    key = "test_image" if modality == "image" else "test_lowdim"
    return {"n_test": int(ev[key]["n"]), "n_envs": int(ev[key]["n_envs"])}


# ----------------------------------------------------------------------------- drivers
def run_train_queue(cells: Dict[str, dict], cells_dir: pathlib.Path, runs: pathlib.Path, state: QueueState,
                    python: str, openpi_root: str, dp_root: str, max_attempts: int, run_cmd: RunCmd = default_run_cmd,
                    env: Optional[Dict[str, str]] = None) -> Dict[str, int]:
    """Train every cell sequentially (skip verified, retry bounded). Returns the status counts."""
    env = dict(env or {})
    jobs = [f"train/{cid}" for cid in cells]
    for cid, cell in cells.items():
        job = f"train/{cid}"; run_dir = runs / cid
        rec = state.get(job)
        if verify_train(run_dir, cell) is None:
            state.set(job, status="done", reason=None); continue
        if rec["status"] == "failed":
            continue
        while rec["attempts"] < max_attempts:
            state.set(job, status="running", attempts=rec["attempts"] + 1)
            rc = run_cmd(train_argv(python, openpi_root, dp_root, cells_dir / f"{cid}.yaml", run_dir), run_dir / "train.log", env)
            why = verify_train(run_dir, cell)
            if why is None:
                state.set(job, status="done", rc=rc, reason=None); break
            state.set(job, status="pending", rc=rc, reason=f"attempt {rec['attempts']}: rc={rc}; {why}")
        else:
            state.set(job, status="failed", reason=f"max attempts ({max_attempts}) reached: {rec.get('reason')}")
    return state.counts(jobs)


def eval_jobs(cells: Dict[str, dict], splits: Sequence[str], samplers: Sequence[str]) -> List[dict]:
    """Ordered job list: screening ddim_100 first for every cell, then the remaining (split, sampler) arms."""
    jobs = []
    for cid, cell in cells.items():
        if "screen" in splits:
            jobs.append({"cell": cid, "split": "screen", "sampler": "ddim", "k": 100})
        if "test" in splits:
            for sk in samplers:
                s, k = sk.split("_"); jobs.append({"cell": cid, "split": "test", "sampler": s, "k": int(k)})
    return jobs


def _job_name(j: dict) -> str:
    return f"eval/{j['cell']}/{j['split']}_{j['sampler']}_{j['k']}"


def run_eval_queue(cells: Dict[str, dict], tasks_cfg: dict, runs: pathlib.Path, results: pathlib.Path, state: QueueState,
                   python: str, dp_root: str, splits: Sequence[str], samplers: Sequence[str], max_attempts: int,
                   parallel: int = 1, gpus: Sequence[str] = ("0",), run_cmd: RunCmd = default_run_cmd,
                   env: Optional[Dict[str, str]] = None) -> Dict[str, int]:
    """Evaluate all scores, including weak models; screening only gates inference in the aggregator.

    Revalidate artifacts on every invocation. Hash each checkpoint once before scheduling its jobs.
    """
    env = dict(env or {})
    jobs = eval_jobs(cells, splits, samplers)
    names = [_job_name(j) for j in jobs]
    training_errors = {cid: verify_train(runs / cid, cell) for cid, cell in cells.items()}
    for name in names:
        if state.get(name)["status"] in ("done", "gated", "running"):
            state.set(name, status="pending")

    def attempt(j: dict, slot: int) -> None:
        name = _job_name(j); cell = cells[j["cell"]]; run_dir = runs / j["cell"]
        out_dir = results / j["cell"] / f"{j['split']}_{j['sampler']}_{j['k']}"
        modality = cell["identity"]["modality"]
        sizes = eval_sizes(tasks_cfg, modality, j["split"])
        rec = state.get(name)
        why_train = training_errors[j["cell"]]
        if why_train is not None:
            state.set(name, status="blocked", reason=f"training not verified: {why_train}"); return
        sha = final_sha(run_dir)
        if j["split"] == "test":
            scr = results / j["cell"] / "screen_ddim_100"
            if verify_eval(scr, "screen", "ddim", 100, eval_sizes(tasks_cfg, modality, "screen")["n_test"], sha, cell) is not None:
                state.set(name, status="blocked", reason="screening result not verified yet"); return
        if verify_eval(out_dir, j["split"], j["sampler"], j["k"], sizes["n_test"], sha, cell) is None:
            state.set(name, status="done", reason=None); return
        if rec["status"] == "failed":
            return
        if rec["attempts"] >= max_attempts:
            state.set(name, status="failed", reason=f"max attempts ({max_attempts}) reached: {rec.get('reason')}"); return
        n = rec["attempts"] + 1
        state.set(name, status="running", attempts=n)
        argv = eval_argv(python, dp_root, run_dir / "checkpoints" / "final.ckpt", out_dir, j["split"], j["sampler"], j["k"],
                         sizes["n_test"], sizes["n_envs"], n, cell.get("runner_overrides", []))
        job_env = {**env, "CUDA_VISIBLE_DEVICES": str(gpus[slot % len(gpus)])}
        rc = run_cmd(argv, out_dir / "eval.log", job_env)
        why = verify_eval(out_dir, j["split"], j["sampler"], j["k"], sizes["n_test"], sha, cell)
        if why is None:
            state.set(name, status="done", rc=rc, reason=None)
        elif n >= max_attempts:
            state.set(name, status="failed", rc=rc, reason=f"max attempts ({max_attempts}) reached: rc={rc}; {why}")
        else:
            state.set(name, status="pending", rc=rc, reason=f"attempt {n}: rc={rc}; {why}")

    # Each pass rechecks unfinished jobs, including failed jobs whose artifacts may have been recovered externally.
    # Retries interleave; stop when a pass
    # changes nothing
    for _pass in range(max_attempts + 2):
        before = json.dumps({n: state.get(n)["status"] for n in names}, sort_keys=True)
        todo = [j for j in jobs if state.get(_job_name(j))["status"] != "done"]
        if not todo:
            break
        screen_first = [j for j in todo if j["split"] == "screen"] + [j for j in todo if j["split"] != "screen"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
            list(ex.map(lambda p: attempt(p[1], p[0]), enumerate(screen_first)))
        after = json.dumps({n: state.get(n)["status"] for n in names}, sort_keys=True)
        attempts_left = any(state.get(_job_name(j))["status"] == "pending" for j in jobs)
        if after == before and not attempts_left:
            break
    return state.counts(names)


def summarize(counts: Dict[str, int]) -> int:
    """Print the terminal line and return the exit code (0 all done, 2 failures, 3 pending/blocked)."""
    terminal = bool(counts) and all(k == "done" for k in counts)
    line = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    if terminal:
        print(f"QUEUE DONE {line}"); return 0
    print(f"QUEUE INCOMPLETE {line}")
    return 2 if counts.get("failed", 0) else 3


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry: ``train`` / ``eval`` / ``status``; returns the exit code of :func:`summarize`."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["train", "eval", "status"])
    ap.add_argument("--tasks", required=True); ap.add_argument("--cells", required=True)
    ap.add_argument("--runs", required=True); ap.add_argument("--results", default=None)
    ap.add_argument("--state", required=True)
    ap.add_argument("--arms", default="core"); ap.add_argument("--only", default="")
    ap.add_argument("--splits", default="screen,test"); ap.add_argument("--samplers", default=",".join(DEFAULT_SAMPLERS))
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--parallel", type=int, default=1); ap.add_argument("--gpus", default="0")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dp-root", default=None, help="default: tasks.yaml dp_root")
    a = ap.parse_args(argv)
    tasks_cfg = yaml.safe_load(open(a.tasks))
    dp_root = a.dp_root or tasks_cfg["dp_root"]
    openpi_root = str(pathlib.Path(__file__).resolve().parents[2])
    cells = load_matrix(pathlib.Path(a.cells), a.arms.split(","), [c for c in a.only.split(",") if c])
    state = QueueState(pathlib.Path(a.state))
    env = {"PYTHONPATH": f"{openpi_root}:{dp_root}"}
    if a.mode == "train":
        counts = run_train_queue(cells, pathlib.Path(a.cells), pathlib.Path(a.runs), state, a.python, openpi_root, dp_root,
                                 a.max_attempts, env=env)
    elif a.mode == "eval":
        counts = run_eval_queue(cells, tasks_cfg, pathlib.Path(a.runs), pathlib.Path(a.results), state, a.python, dp_root,
                                a.splits.split(","), a.samplers.split(","), a.max_attempts, a.parallel, a.gpus.split(","), env=env)
    else:
        print("Recorded ledger status; artifacts are revalidated when train/eval runs.")
        names = [f"train/{c}" for c in cells] + [_job_name(j) for j in eval_jobs(cells, a.splits.split(","), a.samplers.split(","))]
        for n in names:
            r = state.get(n); print(f"{r['status']:8s} attempts={r['attempts']} {n} {r.get('reason') or ''}")
        counts = state.counts(names)
    return summarize(counts)


if __name__ == "__main__":
    sys.exit(main())
