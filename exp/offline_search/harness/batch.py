"""Batch runner: many (method, cell) jobs at once, packed onto the whole machine.

    python -m exp.offline_search.harness.batch --spec <batch.json> --root <store> --out <results/rNN>
        [--budget <#CPUs in mask>] [--cpus 26-35,70-79] [--max-job-workers budget//2] [--round rNN]
        [--scoreboard default|none|PATH] [--timing-concurrency 40] [--no-timing-phase] [--no-numa] [--tag NAME]

spec = JSON list of
    {"method": "<module_or_file>:<Class>", "kwargs": {...}, "family": "name", "cells": "all" | [cell, ...],
     "subsample": null | "tok", "allow_gpu_fit": false, "seed": 0, "profile": false}
(only "method" is required).

Phase 1 (accuracy): the spec expands to one job per (method, cell). Each job is a separate subprocess
`run.py --job --no-timing --workers W` for that single cell, so fits, forked workers and metrics of different jobs
run in parallel. W is proportional to the cell's decision count (W = round(max_job_workers * n / n_max), min 4,
max max_job_workers). Jobs are started biggest first and packed first-fit so that the sum of W over running jobs
stays <= budget; jobs with allow_gpu_fit run at most one at a time. A failed job does not stop the others; there
are no retries.
Phase 2 (timing, unless --no-timing-phase): for every successful job, `run.py --timing-only` (re-fit, then the
300-query single-thread timing pass) with at most --timing-concurrency processes at a time, each pinned to its own
physical core (one logical CPU per core, sched_setaffinity). Timing fields are merged into <cell>.json in place.
NUMA: accuracy jobs are packed per NUMA node (node budget = budget x node share of the CPUs) and pinned to their
node, so a job's fitted arrays live in node-local memory (the bandwidth-bound kNN scans read them per query);
--no-numa disables it.
Finalize: per method <out>/<name>/{summary.json, run_meta.json, DONE|ERROR} (same marker format as run.py) and one
scoreboard row per successful (method, cell), appended once after the timing phase (so every row carries its
ms_per_query; with --no-timing-phase those columns are NaN).
Batch files: <out>/_batch/<tag>.log, <tag>.progress.jsonl (batch/phase/job start + end, wall, workers, core),
<tag>.DONE | <tag>.ERROR, and per-job logs <out>/_batch/<tag>/{acc,timing}/<name>__<cell>.log.
"""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

from . import metrics, run, store  # noqa: E402

POLL_S = 0.2
TAIL_LINES = 40


def _cpulist(text: str) -> set:
    out = set()
    for part in text.strip().split(","):
        if part:
            lo, _, hi = part.partition("-")
            out.update(range(int(lo), int(hi or lo) + 1))
    return out


def numa_nodes() -> dict:
    """{node id: sorted logical CPUs} within our affinity mask (one pseudo-node if the topology is unknown)."""
    allowed = os.sched_getaffinity(0)
    out = {}
    for d in sorted(pathlib.Path("/sys/devices/system/node").glob("node[0-9]*")):
        try:
            cpus = _cpulist((d / "cpulist").read_text()) & allowed
        except Exception:
            continue
        if cpus:
            out[int(d.name[4:])] = sorted(cpus)
    return out or {-1: sorted(allowed)}


class Batch:
    def __init__(self, a):
        self.a = a
        self.root = pathlib.Path(a.root).resolve()
        self.out = pathlib.Path(a.out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.bdir = self.out / "_batch"
        self.bdir.mkdir(exist_ok=True)
        stem = pathlib.Path(a.spec).stem
        self.tag = a.tag or f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.logp = self.bdir / f"{self.tag}.log"
        self.progp = self.bdir / f"{self.tag}.progress.jsonl"
        for mk in ("DONE", "ERROR"):
            p = self.bdir / f"{self.tag}.{mk}"
            if p.exists():
                p.unlink()
        self.jobdir = self.bdir / self.tag
        self.budget = max(1, int(a.budget))
        allowed = sorted(os.sched_getaffinity(0))
        self.nodes = numa_nodes() if not a.no_numa else {-1: allowed}
        ncpu = sum(len(c) for c in self.nodes.values())
        self.node_budget = {n: max(1, round(self.budget * len(c) / ncpu)) for n, c in self.nodes.items()}
        self.max_job = max(1, min(max(self.node_budget.values()), int(a.max_job_workers or self.budget // 2)))
        self.rnd = a.round or self.out.name
        self.sb = (self.out.parent / "scoreboard.csv") if a.scoreboard == "default" else (
            None if a.scoreboard in (None, "none") else pathlib.Path(a.scoreboard))
        self.run_ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.procs = {}

    # -------------------------------------------------------------------------------------- logging
    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.logp, "a") as f:
            f.write(line + "\n")

    def ev(self, **obj):
        run.plog(self.progp, **obj)

    # ---------------------------------------------------------------------------------------- spec
    def expand(self):
        spec = json.loads(pathlib.Path(self.a.spec).read_text())
        if not isinstance(spec, list) or not spec:
            raise ValueError(f"{self.a.spec}: spec must be a non-empty JSON list of job entries")
        jobs, seen, methods = [], set(), {}
        for k, e in enumerate(spec):
            if not isinstance(e, dict) or "method" not in e:
                raise ValueError(f"spec entry {k}: needs at least {{'method': '<module_or_file>:<Class>'}}")
            unknown = set(e) - {"method", "kwargs", "family", "cells", "subsample", "allow_gpu_fit", "seed", "profile"}
            if unknown:
                raise ValueError(f"spec entry {k}: unknown keys {sorted(unknown)}")
            kw = e.get("kwargs") or {}
            cls, src = run.load_method_class(e["method"])
            probe = run.build_method(cls, kw)
            name = probe.name
            fam = e.get("family") or run.infer_family(probe, src)
            del probe
            cells = e.get("cells", "all")
            cells = store.resolve_cells(cells if isinstance(cells, str) else ",".join(cells), self.root)
            sub = e.get("subsample")
            if sub not in (None, "tok"):
                raise ValueError(f"spec entry {k}: subsample must be null or 'tok'")
            methods.setdefault(name, {"entry": e, "method_spec": e["method"], "src": src, "family": fam, "kwargs": kw,
                                      "cells": []})
            for c in cells:
                if (name, c) in seen:
                    raise ValueError(f"spec: ({name}, {c}) appears twice -- give variants distinct names")
                seen.add((name, c))
                qc = store.QueryCell(self.root, c)
                n = int(qc.tok_rows.size) if sub == "tok" else qc.N
                methods[name]["cells"].append(c)
                jobs.append(dict(id=len(jobs), method_spec=e["method"], kwargs=kw, name=name, family=fam, cell=c,
                                 subsample=sub, allow_gpu=bool(e.get("allow_gpu_fit", False)),
                                 seed=int(e.get("seed", 0)), profile=bool(e.get("profile", False)), n=n))
        nmax = max(j["n"] for j in jobs) or 1
        for j in jobs:
            j["W"] = int(min(self.max_job, max(4, round(self.max_job * j["n"] / nmax))))
        self.jobs, self.methods = jobs, methods
        # stale per-method markers / job records of this batch's cells
        for name, m in methods.items():
            mdir = self.out / name
            mdir.mkdir(parents=True, exist_ok=True)
            for mk in ("DONE", "ERROR"):
                if (mdir / mk).exists():
                    (mdir / mk).unlink()
            for c in m["cells"]:
                for suf in (".json", ".timing.json"):
                    p = mdir / "_jobs" / f"{c}{suf}"
                    if p.exists():
                        p.unlink()

    # --------------------------------------------------------------------------------------- procs
    def _cmd(self, j, timing, core=None):
        c = [sys.executable, "-m", "exp.offline_search.harness.run", "--method", j["method_spec"],
             "--kwargs", json.dumps(j["kwargs"]), "--cells", j["cell"], "--root", str(self.root), "--out", str(self.out),
             "--round", self.rnd, "--family", j["family"] or "", "--scoreboard", "none", "--seed", str(j["seed"])]
        if j["subsample"]:
            c += ["--subsample", j["subsample"]]
        if j["allow_gpu"]:
            c += ["--allow-gpu-fit"]
        if timing:
            c += ["--timing-only", "--workers", "1", "--pin-core", str(core)]
        else:
            c += ["--job", "--no-timing", "--workers", str(j["W"])]
            if j["profile"]:
                c += ["--profile"]
        return c

    def _start(self, j, phase, core=None, cpus=None, node=None):
        d = self.jobdir / phase
        d.mkdir(parents=True, exist_ok=True)
        lp = d / f"{j['name']}__{j['cell']}.log"
        lf = open(lp, "w")
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        if phase == "timing":      # recorded in <cell>.json: concurrent timing processes share memory bandwidth
            env["OFFLINE_SEARCH_TIMING_CONCURRENCY"] = str(self.timing_slots)
        if j["allow_gpu"]:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        mask = {core} if core is not None else (set(cpus) if cpus else None)
        pre = (lambda m=mask: os.sched_setaffinity(0, m)) if mask else None
        p = subprocess.Popen(self._cmd(j, phase == "timing", core), cwd=str(store.REPO), stdout=lf,
                             stderr=subprocess.STDOUT, env=env, preexec_fn=pre, start_new_session=True)
        self.procs[(phase, j["id"])] = dict(p=p, j=j, t0=time.time(), log=lp, lf=lf, core=core, node=node)
        self.ev(ev="job_start", phase=phase, id=j["id"], method=j["name"], cell=j["cell"], workers=j["W"] if
                phase == "acc" else 1, core=core, node=node, n=j["n"], pid=p.pid)

    def _finish(self, key):
        r = self.procs.pop(key)
        r["lf"].close()
        phase, j = key[0], r["j"]
        wall = time.time() - r["t0"]
        rc = r["p"].returncode
        rp = self.out / j["name"] / "_jobs" / f"{j['cell']}{'.timing' if phase == 'timing' else ''}.json"
        rec = json.loads(rp.read_text()) if rp.exists() else None
        if rec is None:
            tail = "".join(open(r["log"]).readlines()[-TAIL_LINES:]) if r["log"].exists() else ""
            rec = {"status": "error", "error": f"job exited with code {rc} without a result record (see {r['log']})",
                   "traceback": tail}
        elif rc != 0 and rec.get("status") == "ok":
            rec = dict(rec, status="error", error=f"exit code {rc}")
        j[f"{phase}_status"] = rec.get("status")
        j[f"{phase}_rec"] = rec
        j[f"{phase}_wall"] = wall
        self.ev(ev="job_end", phase=phase, id=j["id"], method=j["name"], cell=j["cell"], status=rec.get("status"),
                rc=rc, wall_s=round(wall, 2), workers=j["W"] if phase == "acc" else 1, core=r["core"],
                node=r["node"], error=rec.get("error"))
        msg = f"{phase:6s} {rec.get('status'):5s} {j['name']} {j['cell']} ({wall:.1f}s"
        msg += f", W={j['W']}, node {r['node']})" if phase == "acc" else f", core {r['core']})"
        if rec.get("status") == "error":
            msg += f": {rec.get('error')} -- log {r['log']}"
        self.log(msg)

    def _reap(self):
        done = [k for k, r in self.procs.items() if r["p"].poll() is not None]
        for k in done:
            self._finish(k)
        return done

    def kill_all(self):
        for r in list(self.procs.values()):
            try:
                os.killpg(r["p"].pid, signal.SIGTERM)
            except Exception:
                pass

    # -------------------------------------------------------------------------------------- phases
    def phase_acc(self):
        """First-fit decreasing over NUMA nodes: a job (W workers) goes to the node with the most free worker
        slots that can hold it and is pinned to that node's CPUs (fit memory is then node-local and the forked
        workers inherit the mask). --no-numa: one pseudo-node, no pinning."""
        pending = sorted(self.jobs, key=lambda j: (-j["W"], -j["n"], j["id"]))
        free = dict(self.node_budget)
        cap = max(self.node_budget.values())
        gpu = 0
        t0 = time.time()
        self.ev(ev="phase_start", phase="acc", jobs=len(pending), budget=self.budget, max_job_workers=self.max_job,
                node_budget=self.node_budget)
        self.log(f"accuracy phase: {len(pending)} jobs, budget {self.budget} workers "
                 f"(per NUMA node {self.node_budget}), max {self.max_job} per job")
        while pending or self.procs:
            started = True
            while started:
                started = False
                for j in pending:
                    w = min(j["W"], cap)
                    fits = [n for n in free if free[n] >= w]
                    if fits and (not j["allow_gpu"] or gpu == 0):
                        n = max(fits, key=lambda x: (free[x], -x))
                        self._start(j, "acc", cpus=None if n == -1 else self.nodes[n], node=n)
                        j["node"], j["w_used"] = n, w
                        pending.remove(j)
                        free[n] -= w
                        gpu += j["allow_gpu"]
                        started = True
                        break
            for k in self._reap():
                j = self.jobs[k[1]]
                free[j["node"]] += j["w_used"]
                gpu -= j["allow_gpu"]
            if pending or self.procs:
                time.sleep(POLL_S)
        wall = time.time() - t0
        self.ev(ev="phase_done", phase="acc", wall_s=round(wall, 2))
        self.log(f"accuracy phase done in {wall:.1f}s")
        return wall

    @staticmethod
    def physical_cores():
        """One logical CPU per physical core (the lowest id of each sibling set), within our affinity mask."""
        allowed = os.sched_getaffinity(0)
        firsts = set()
        for c in sorted(allowed):
            p = pathlib.Path(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list")
            try:
                sib = p.read_text().strip()
                first = int(sib.replace("-", ",").split(",")[0])
            except Exception:
                first = c
            firsts.add(first if first in allowed else c)
        return sorted(firsts)

    def phase_timing(self):
        todo = [j for j in self.jobs if j.get("acc_status") == "ok"]
        # longest first: fit + 600 queries at the accuracy run's mean query time
        for j in todo:
            try:
                s = json.loads((self.out / j["name"] / f"{j['cell']}.json").read_text())
                j["t_est"] = s["fit"]["fit_s"] + 600 * s["profile"]["query_total_us"]["mean"] / 1e6
            except Exception:
                j["t_est"] = 0.0
        todo.sort(key=lambda j: (-j["t_est"], j["id"]))
        cores = self.physical_cores()[: max(1, int(self.a.timing_concurrency))]
        self.timing_slots = len(cores)
        free = list(cores)
        gpu = 0
        t0 = time.time()
        self.ev(ev="phase_start", phase="timing", jobs=len(todo), cores=cores)
        self.log(f"timing phase: {len(todo)} jobs on {len(cores)} pinned physical cores {cores[0]}..{cores[-1]}")
        while todo or self.procs:
            started = True
            while started and free:
                started = False
                for j in todo:
                    if not j["allow_gpu"] or gpu == 0:
                        core = free.pop(0)
                        self._start(j, "timing", core)
                        todo.remove(j)
                        gpu += j["allow_gpu"]
                        started = True
                        break
            for k in self._reap():
                pass
            busy = {r["core"] for k, r in self.procs.items() if k[0] == "timing"}
            free = [c for c in cores if c not in busy]
            gpu = sum(r["j"]["allow_gpu"] for k, r in self.procs.items() if k[0] == "timing")
            if todo or self.procs:
                time.sleep(POLL_S)
        wall = time.time() - t0
        self.ev(ev="phase_done", phase="timing", wall_s=round(wall, 2))
        self.log(f"timing phase done in {wall:.1f}s")
        return wall

    # ------------------------------------------------------------------------------------- finalize
    def finalize(self, walls):
        finished = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        failed_jobs = []
        for name, m in self.methods.items():
            mdir = self.out / name
            js = [j for j in self.jobs if j["name"] == name]
            ok = [j for j in js if j.get("acc_status") == "ok"]
            skipped = {j["cell"]: j["acc_rec"].get("reason") for j in js if j.get("acc_status") == "skip"}
            errs = {}
            for j in js:
                for ph in ("acc", "timing"):
                    if j.get(f"{ph}_status") == "error":
                        rec = j[f"{ph}_rec"]
                        errs[f"{j['cell']} [{ph}]"] = f"{rec.get('error')}\n{rec.get('traceback', '')}"
                        failed_jobs.append(f"{name} {j['cell']} [{ph}]")
                if j.get("acc_status") is None:
                    errs[f"{j['cell']} [acc]"] = "job never finished"
                    failed_jobs.append(f"{name} {j['cell']} [acc]")
            if ok:
                metrics.summarize_method(mdir)
            if self.sb is not None:
                for j in ok:
                    s = json.loads((mdir / f"{j['cell']}.json").read_text())
                    metrics.append_scoreboard(self.sb, run.scoreboard_row(s, rnd=self.rnd, run_ts=self.run_ts,
                                                                          name=name, fam=m["family"]))
            meta = {"argv": sys.argv, "mode": "batch", "batch_tag": self.tag, "spec": str(self.a.spec),
                    "spec_entry": m["entry"], "method_spec": m["method_spec"], "method_src": m["src"], "method": name,
                    "family": m["family"], "kwargs": m["kwargs"], "cells": m["cells"], "root": str(self.root),
                    "out": str(self.out), "workers": {j["cell"]: j["W"] for j in js}, "round": self.rnd,
                    "scoreboard": str(self.sb), "host": socket.gethostname(), "started": self.started,
                    "finished": finished, "errors": sorted(errs), "skipped": skipped,
                    "job_wall_s": {j["cell"]: {"acc": j.get("acc_wall"), "timing": j.get("timing_wall")} for j in js}}
            (mdir / "run_meta.json").write_text(json.dumps(meta, indent=1))
            if errs:
                (mdir / "ERROR").write_text("failed cells: " + ", ".join(sorted(errs)) + "\n\n" +
                                            "\n\n".join(f"== {c}\n{tb}" for c, tb in sorted(errs.items())))
            else:
                (mdir / "DONE").write_text(f"ok {len(ok)} cells, skipped {sorted(skipped)} {finished}\n")
        summ = {"tag": self.tag, "jobs": len(self.jobs), "failed": failed_jobs, "walls_s": walls,
                "budget": self.budget, "finished": finished}
        self.ev(ev="batch_done", **summ)
        if failed_jobs:
            (self.bdir / f"{self.tag}.ERROR").write_text(json.dumps(summ, indent=1) + "\n")
        else:
            (self.bdir / f"{self.tag}.DONE").write_text(json.dumps(summ, indent=1) + "\n")
        return summ


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, help="JSON list of {method, kwargs, family, cells, subsample, ...}")
    ap.add_argument("--root", default=str(store.FULL_ROOT))
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=None,
                    help="max sum of forked workers of running jobs (default: #CPUs in the affinity mask)")
    ap.add_argument("--cpus", default=None, help="pin the batch and every job to this CPU list, e.g. 26-35,70-79 "
                    "(timing jobs then use distinct physical cores inside it)")
    ap.add_argument("--max-job-workers", type=int, default=None, help="workers of the largest cell (default budget//2)")
    ap.add_argument("--round", default=None)
    ap.add_argument("--scoreboard", default="default", help="CSV path, 'default' (<out>/../scoreboard.csv) or 'none'")
    ap.add_argument("--timing-concurrency", type=int, default=40)
    ap.add_argument("--no-timing-phase", action="store_true")
    ap.add_argument("--tag", default=None, help="batch tag (default <spec stem>_<timestamp>)")
    ap.add_argument("--no-numa", action="store_true", help="do not pin accuracy jobs to NUMA nodes")
    a = ap.parse_args(argv)
    if a.cpus:
        os.sched_setaffinity(0, _cpulist(a.cpus))      # inherited by every job; NUMA / core picks stay inside it
    if a.budget is None:
        a.budget = len(os.sched_getaffinity(0))
    b = Batch(a)
    walls = {}
    t0 = time.time()
    try:
        b.ev(ev="batch_start", tag=b.tag, spec=str(a.spec), root=str(b.root), out=str(b.out), budget=b.budget,
             cpus=sorted(os.sched_getaffinity(0)))
        b.expand()
        b.log(f"batch {b.tag}: {len(b.jobs)} jobs ({len(b.methods)} methods) root={b.root} out={b.out}")
        walls["acc"] = round(b.phase_acc(), 2)
        if not a.no_timing_phase:
            walls["timing"] = round(b.phase_timing(), 2)
        walls["total"] = round(time.time() - t0, 2)
        s = b.finalize(walls)
        b.log(f"batch {b.tag} {'ERROR' if s['failed'] else 'DONE'}: walls {walls}; failed {s['failed']}")
        return 1 if s["failed"] else 0
    except BaseException as exc:
        b.kill_all()
        tb = traceback.format_exc()
        b.log(f"batch {b.tag} crashed: {exc!r}")
        b.ev(ev="batch_error", error=repr(exc), traceback=tb)
        (b.bdir / f"{b.tag}.ERROR").write_text(f"batch crashed: {exc!r}\n{tb}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
