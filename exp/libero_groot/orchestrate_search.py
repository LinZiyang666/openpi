"""Cell scheduler for the GR00T x LIBERO weighted-sum search.

Runs ON weilandserver (which carries the tether CLI), so it survives the
operator's session dying. Each slot owns one public port and pulls cells from a
shared queue:

    start local cache server with the cell's YAML  ->  launch the timan107 sim
    fleet against it  ->  wait  ->  merge + pull the per-episode results  ->
    tear the server down  ->  next cell

Why a server restart per cell: the GR00T serving stack has no yaml hot-swap
(``allow_dynamic_bundles=False``), so configuration identity is carried by the
process. That is a feature here -- a cell can never be evaluated under another
cell's weights.

Why the sim fleet lives off-box: six servers hold ~36 GB of the 4090's 48 GB,
which leaves room for barely twenty local EGL contexts. timan107 has eight idle
GPUs and 48 cores, and pure-cache inference is ~9 ms/call, so the wire round
trip is cheap relative to the ~200 ms环境 step it overlaps with.

A shared queue rather than a static split: cells finish at different speeds
(episode length tracks success), and a static shard leaves the fast slots idle
while one slow slot drains.

Resume is by artifact: a cell whose merged results file already exists is
skipped, so re-running after a crash costs nothing.

Four opt-in flags extend the same scheduler to the gate-threshold Pareto line
without changing anything the search does: ``--per-step-dir`` turns on per-step
verdict capture, ``--init-subdir`` and ``--shards-dir`` retarget the init pool
and the work partition (the Pareto's warmup phase runs on the B pool), and
``--phase`` is stamped into each captured row. All four default to the search's
behaviour, so a search re-run is byte-identical.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import queue
import socket
import subprocess
import threading
import time

# Resolved from this file, never restated as a constant: the scheduler runs on
# whichever host holds the checkout it was launched from, and a second absolute
# path would silently pin the whole chain to one machine. The control-plane box
# and the serving box carry the checkout at different paths, so a hardcoded root
# is wrong on one of them by construction.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
WEILAND_REPO = str(REPO_ROOT)
# The GR00T island is a separate install with no relation to this checkout, so
# it stays a constant -- but an overridable one, since it is the single thing
# here that a different host would legitimately place elsewhere.
GR00T_HOME = os.environ.get("GROOT_N15_HOME", "/home/weiland/gr00t_n15")
TIMAN = "timan107"
TIMAN_REPO = "/scratch/zixuans8/openpi"
TIMAN_HOME = "/home/zixuans8"
TIMAN_CONDA = "/shared/nas/data/m1/zixuans8/miniconda3/bin/conda"
TIMAN_SIM = "/scratch/zixuans8/libero_sim"
# Outside the checkout on purpose: /scratch/zixuans8/openpi belongs to the
# rl_router line, and an untracked directory there would surface in its
# git status and risk being swept into someone else's commit.
TIMAN_SHARDS = "/tmp/libsearch/shards"
GR00T_PATH = f"{GR00T_HOME}:{GR00T_HOME}/examples/Libero:{WEILAND_REPO}:{WEILAND_REPO}/src"
ISLAND_PY = os.environ.get(
    "GROOT_N15_PYTHON", "/home/weiland/gr00t_n15_venv/.venv/bin/python"
)


def preflight() -> None:
    """Fail before a single episode is dispatched if the host is not set up.

    Without this the first symptom is a server that never binds its port and a
    slot that waits out its 150-second startup budget, once per cell -- a whole
    phase can burn before anyone reads the log and finds a missing interpreter.
    """
    missing = [
        str(path)
        for path in (
            REPO_ROOT / "exp/libero_groot/serve_groot_libero.py",
            pathlib.Path(ISLAND_PY),
            pathlib.Path(GR00T_HOME),
        )
        if not path.exists()
    ]
    if missing:
        raise SystemExit(
            f"host {socket.gethostname()!r} is not set up to serve GR00T: missing "
            + ", ".join(missing)
            + f" (repo root resolved from __file__ = {REPO_ROOT}; override the "
            "island with GROOT_N15_HOME / GROOT_N15_PYTHON)"
        )
_LOCK = threading.Lock()
# The broker caps concurrent file transfers: six slots finishing within the
# same minute all issued a pull and were rejected with too_many_in_flight,
# losing three fully-computed cells to a transport limit. Pulls are
# serialized here -- the payload is a few hundred KB, so the queue costs
# nothing next to the 27 minutes that produced it.
_PULL_LOCK = threading.Lock()


def log(msg: str) -> None:
    with _LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(cmd: str, timeout: int = 540) -> str:
    """Run a shell command, returning stdout (stderr folded in on failure)."""
    proc = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=timeout
    )
    return proc.stdout if proc.returncode == 0 else proc.stdout + proc.stderr


def tether(node: str, script: str, timeout: int = 540) -> str:
    # tether exec caps a single call around ten minutes and buffers output, so
    # every call here is a short probe or a detached launch -- never a wait.
    return sh(
        f"export HOME=/home/weiland; tether exec {node} -- bash -lc {json.dumps(script)}",
        timeout=timeout,
    )


class Slot:
    """One public port: a local cache server plus its remote sim fleet."""

    def __init__(self, port: int, args: argparse.Namespace) -> None:
        self.port = port
        self.a = args

    # -- server side (local) ------------------------------------------------
    def start_server(self, cell: str) -> bool:
        yaml_path = f"{self.a.yaml_dir}/{cell}.yaml"
        sh(f"tmux kill-session -t libsrv{self.port} 2>/dev/null")
        time.sleep(3)
        cmd = (
            f"cd {WEILAND_REPO} && PYTHONPATH={GR00T_PATH} OPENPI_MONITOR_LEVEL=BASIC "
            f"{ISLAND_PY} exp/libero_groot/serve_groot_libero.py "
            f"--checkpoint {self.a.checkpoint} --port {self.port} --concurrent "
            f"--cache-config {yaml_path} 2>&1 | tee /tmp/libsrv{self.port}.log"
        )
        sh(f"tmux new -s libsrv{self.port} -d {json.dumps(cmd)}")
        for _ in range(60):
            if f":{self.port} " in sh(f"ss -tln | grep ':{self.port} ' || true"):
                return True
            time.sleep(5)
        log(f"slot {self.port}: server did not come up for {cell}")
        return False

    def stop_server(self) -> None:
        sh(f"tmux kill-session -t libsrv{self.port} 2>/dev/null")

    # -- client side (timan107) --------------------------------------------
    def out_dir(self, cell: str) -> str:
        """Per-cell, not per-slot: a slot wipes its directory when it claims the
        next cell, so a per-slot path destroys the evidence of a collection that
        failed -- exactly when it is needed."""
        return f"/tmp/libsearch/{self.port}/{cell}"

    def launch_clients(self, cell: str) -> None:
        out = self.out_dir(cell)
        parts = [f"export HOME={TIMAN_HOME}; rm -rf {out}; mkdir -p {out}"]
        for w in range(self.a.workers):
            gpu = w % self.a.gpus
            shard = f"{self.a.shards_dir}/{self.a.shard_prefix}_lane{w}.json"
            # Per-worker directory, not a shared one. PerStepWriterPool names its
            # temp file by (yaml_id, worker slot) and truncates it on
            # construction; every worker here is its own process holding slot 0,
            # so one directory would have them truncating each other's file,
            # writing the same path concurrently, and overwriting each other's
            # merged output -- losing rows with nothing reporting a failure.
            per_step = (
                f" --per-step-log-dir {out}/per_step/w{w} --yaml-id {cell} "
                f"--phase {self.a.phase}"
                if self.a.per_step_dir
                else ""
            )
            cmd = (
                f"cd {TIMAN_REPO} && MUJOCO_EGL_DEVICE_ID={gpu} PYTHONPATH=. "
                f"{TIMAN_CONDA} run -p {TIMAN_SIM} --no-capture-output python "
                f"examples/libero/main.py --host {self.a.public_host} --port {self.port} "
                f"--task-suite-name {self.a.suite} --num-trials-per-task 50 "
                f"--num-workers 1 --resize-size 256 --replan-steps 5 "
                f"--init-states-dir {TIMAN_REPO}/exp/common/data/db_init/libero/{self.a.init_subdir} "
                f"--cuda-visible-devices {gpu} --episode-filter {shard} "
                f"--save-episode-results --episode-results-path {out}/r{w}.json"
                f"{per_step}"
            )
            parts.append(f"tmux new -s lw{self.port}_{w} -d {json.dumps(cmd)}")
        tether(TIMAN, "; ".join(parts))

    def collect_per_step(self, cell: str) -> None:
        """Merge the fleet's per-step JSONL and pull it back with a sidecar.

        The sidecar carries ``lanes_expected`` / ``lanes_found``: a worker whose
        per-step file never appeared is the one failure the analysis cannot
        detect from the merged content, because the remaining episodes are
        internally consistent and simply fewer.
        """
        out = self.out_dir(cell)
        merged = f"{out}/per_step.jsonl"
        sidecar = f"{out}/per_step.merge.json"
        dst_dir = pathlib.Path(self.a.per_step_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        script = (
            f"export HOME={TIMAN_HOME}; python3 -c \"import glob,json;"
            f"ps=sorted(glob.glob('{out}/per_step/w*/{cell}.jsonl'));"
            f"rows=[l for p in ps for l in open(p) if l.strip()];"
            f"open('{merged}','w').writelines(rows);"
            f"json.dump({{'lanes_expected':{self.a.workers},'lanes_found':len(ps),"
            f"'rows':len(rows)}},open('{sidecar}','w'));"
            f"print('PERSTEP',len(ps),len(rows))\""
        )
        merge_out = tether(TIMAN, script)
        for name, remote in ((f"{cell}.jsonl", merged), (f"{cell}.merge.json", sidecar)):
            dst = dst_dir / name
            for attempt in range(1, 7):
                with _PULL_LOCK:
                    sh(f"export HOME=/home/weiland; tether pull {TIMAN}:{remote} "
                       f"{dst} --force 2>&1 | tail -2")
                if dst.exists():
                    break
                time.sleep(20 * attempt)
            if not dst.exists():
                log(f"slot {self.port}: {cell} per-step {name} UNPULLED -- "
                    f"recoverable at {TIMAN}:{remote} (merge said "
                    f"{merge_out.strip()[-40:]!r})")

    def clients_alive(self) -> int:
        out = tether(TIMAN, f"export HOME={TIMAN_HOME}; tmux ls 2>/dev/null | grep -c '^lw{self.port}_' || true")
        for line in reversed(out.strip().splitlines()):
            if line.strip().isdigit():
                return int(line.strip())
        return -1  # unreadable: treat as alive, the stall guard handles a hang

    def collect(self, cell: str) -> int:
        """Merge the fleet's per-episode results and pull them back. Returns rows."""
        out = self.out_dir(cell)
        merged = f"{out}/merged.json"
        dst = pathlib.Path(self.a.results_dir) / f"{cell}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Two attempts: six slot threads issue tether calls concurrently, and a
        # transient failure here costs a whole 26-minute cell. Both the merge
        # and the pull output are logged when the file does not land -- the
        # first occurrence was undiagnosable because neither was captured.
        merge_out = tether(
            TIMAN,
            f"export HOME={TIMAN_HOME}; python3 -c \"import glob,json;"
            f"rows=[r for p in sorted(glob.glob('{out}/r*.json')) for r in json.load(open(p))];"
            f"json.dump(rows, open('{merged}','w'));print('MERGED',len(rows))\"",
        )
        pull_out = ""
        for attempt in range(1, 7):
            with _PULL_LOCK:
                pull_out = sh(
                    f"export HOME=/home/weiland; tether pull {TIMAN}:{merged} {dst} --force 2>&1 | tail -2"
                )
            if dst.exists():
                break
            log(f"slot {self.port}: {cell} pull attempt {attempt} failed: "
                f"{pull_out.strip()[-160:]!r}")
            time.sleep(20 * attempt)  # linear backoff: the limit is transient
        if not dst.exists():
            log(f"slot {self.port}: {cell} UNPULLED -- rows are merged at "
                f"{TIMAN}:{merged}, recoverable without re-running "
                f"(merge said {merge_out.strip()[-40:]!r})")
        if not dst.exists():
            return 0
        try:
            rows = len(json.loads(dst.read_text()))
        except json.JSONDecodeError:
            dst.unlink()
            return 0
        # A short result is a silently truncated cell -- one worker OOM-killed
        # or timed out drops its whole shard, and the cell would then be
        # compared against the others on fewer episodes. Keep the evidence
        # under a .partial name and leave the cell unclaimed so resume re-runs
        # it; a result file existing is what "done" means to the queue.
        if rows != self.a.expect:
            dst.rename(dst.with_suffix(".partial.json"))
            log(f"slot {self.port}: {cell} INCOMPLETE {rows}/{self.a.expect} rows, kept as .partial")
            return 0
        # Only after the results file is accepted: a short cell is re-run, and
        # its per-step evidence would be superseded anyway.
        if self.a.per_step_dir:
            self.collect_per_step(cell)
        return rows

    def kill_clients(self) -> None:
        tether(
            TIMAN,
            f"export HOME={TIMAN_HOME}; for s in $(tmux ls 2>/dev/null | "
            f"grep -oE '^lw{self.port}_[0-9]+'); do tmux kill-session -t $s 2>/dev/null; done; echo ok",
        )

    # -- one cell ----------------------------------------------------------
    def run_cell(self, cell: str) -> None:
        t0 = time.time()
        self.kill_clients()
        if not self.start_server(cell):
            self.stop_server()
            return
        self.launch_clients(cell)
        log(f"slot {self.port}: {cell} launched ({self.a.workers} workers)")

        idle = 0
        while True:
            time.sleep(self.a.poll)
            alive = self.clients_alive()
            if alive == 0:
                break
            idle = idle + 1 if alive == -1 else 0
            if idle * self.a.poll > self.a.stall:
                log(f"slot {self.port}: {cell} unreadable for {self.a.stall}s, giving up")
                break
            if time.time() - t0 > self.a.cell_timeout:
                log(f"slot {self.port}: {cell} exceeded cell timeout")
                break

        rows = self.collect(cell)
        self.kill_clients()
        self.stop_server()
        mins = (time.time() - t0) / 60
        if not rows:
            log(f"slot {self.port}: {cell} FAILED (no rows, {mins:.1f} min)")
            return
        records = json.loads((pathlib.Path(self.a.results_dir) / f"{cell}.json").read_text())
        ok = sum(1 for r in records if r["success"])
        log(f"slot {self.port}: {cell} DONE rows={rows} sr={ok / rows:.3f} ({mins:.1f} min)")


def reap_stale_workers() -> None:
    """Kill every sim worker on timan107 before claiming the first cell.

    This process is the only thing that creates ``lw<port>_<n>`` sessions, so
    anything alive at startup belongs to a run that is already over. That
    matters more than it sounds: a previous stage's workers keep hammering the
    same six ports, and because every server serializes ``infer``, they double
    the queue on every call -- 88 ms became 2325 ms, lanes died mid-shard, and
    the cells came back short. They are invisible to a process-count check
    because their sessions carry exactly the same name pattern as the live
    ones; only the suite in their argv distinguishes them.
    """
    out = tether(
        TIMAN,
        f"export HOME={TIMAN_HOME}; n=0; "
        f"for s in $(tmux ls 2>/dev/null | grep -oE '^lw[0-9]+_[0-9]+'); do "
        f"tmux kill-session -t $s 2>/dev/null; n=$((n+1)); done; "
        f"sleep 3; for p in $(pgrep -f 'task-suite-nam[e]'); do kill -TERM $p 2>/dev/null; done; "
        f"echo REAPED $n",
    )
    line = next((x for x in out.splitlines() if x.startswith("REAPED")), "REAPED ?")
    log(f"startup reap: {line.split()[-1]} stale worker session(s)")


def prepare_shards(args: argparse.Namespace) -> None:
    """(Re)cut the eval shards on timan107 to match the worker count exactly.

    Generating them here rather than by hand removes a whole failure class: a
    shard set cut for N lanes but driven with M<N workers silently drops the
    (N-M) tail shards, so every cell comes back short by the tasks those shards
    owned -- with no error anywhere, because each launched worker completed its
    own assignment perfectly. The completeness guard in ``collect`` catches it,
    but only after a cell has burned its wall clock.

    ``--skip-shard-prep`` suppresses only the cutting, never the verification:
    the Pareto warmup supplies its own shard set (a B-pool selection this
    module has no business re-deriving), but the guarantee that the set covers
    exactly ``--expect`` episodes across exactly ``--workers`` lanes is the
    valuable half and applies either way.
    """
    shards, prefix = args.shards_dir, args.shard_prefix
    if args.skip_shard_prep:
        log(f"shards: prep skipped, using the externally supplied set in {shards}")
    else:
        script = (
            f"export HOME={TIMAN_HOME}; rm -rf {shards}; mkdir -p {shards}; "
            f"python3 /tmp/make_shards.py --num-tasks {args.tasks} --trials {args.trials} "
            f"--lanes {args.workers} --out-dir {shards} --prefix {prefix} | tail -2; "
            f"ls {shards}/{prefix}_lane*.json | wc -l"
        )
        out = tether(TIMAN, script)
        log(f"shards: {out.strip().splitlines()[-1] if out.strip() else 'NO OUTPUT'} lanes for "
            f"{args.workers} workers")
    verify = tether(
        TIMAN,
        f"export HOME={TIMAN_HOME}; python3 -c \"import glob,json;"
        f"n=sum(len(json.load(open(p))) for p in glob.glob('{shards}/{prefix}_lane*.json'));"
        f"f=len(glob.glob('{shards}/{prefix}_lane*.json'));print(f'SHARDS {{f}} {{n}}')\"",
    )
    line = next((x for x in verify.splitlines() if x.startswith("SHARDS ")), "")
    if not line:
        raise SystemExit(f"could not verify shards on {TIMAN}: {verify!r}")
    files, episodes = (int(x) for x in line.split()[1:3])
    if files != args.workers or episodes != args.expect:
        raise SystemExit(
            f"shard set is {files} files / {episodes} episodes, expected "
            f"{args.workers} / {args.expect}: every cell would come back short"
        )
    log(f"shards verified: {files} lanes covering {episodes} episodes")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yaml-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--checkpoint", default="/home/weiland/ckpt_n15_libero_spatial")
    ap.add_argument("--ports", default="23160,23161,23162,23163,23164,23165")
    ap.add_argument("--workers", type=int, default=16, help="sim workers per slot")
    ap.add_argument("--gpus", type=int, default=8, help="timan107 GPUs to spread EGL over")
    ap.add_argument("--public-host", default="ziyanglin.com")
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--expect", type=int, default=500,
                    help="Episodes a complete cell must return (tasks x trials).")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--stall", type=int, default=900)
    ap.add_argument("--cell-timeout", type=int, default=7200)
    ap.add_argument("--only", default="", help="comma-separated cell ids (default: all)")
    # Opt-in extensions for the gate-threshold Pareto line. Every default
    # reproduces the search's behaviour exactly.
    ap.add_argument("--per-step-dir", default="",
                    help="capture per-step verdicts and pull them here (empty: off)")
    ap.add_argument("--phase", default="eval", help="stamped into each per-step row")
    ap.add_argument("--init-subdir", default="",
                    help="init pool under exp/common/data/db_init/libero "
                         "(default: <suite>_apool)")
    ap.add_argument("--shards-dir", default=TIMAN_SHARDS)
    ap.add_argument("--shard-prefix", default="eval")
    ap.add_argument("--skip-shard-prep", action="store_true",
                    help="shards are supplied externally; verify but do not cut")
    args = ap.parse_args()
    # Resolved once, here, so every consumer sees the same pool: an empty
    # --init-subdir means "the suite's frozen A pool", which is what every
    # evaluation phase uses. The Pareto warmup overrides it with the B pool.
    if not args.init_subdir:
        args.init_subdir = f"{args.suite}_apool"

    preflight()
    reap_stale_workers()
    prepare_shards(args)

    yaml_dir = pathlib.Path(args.yaml_dir)
    cells = sorted(p.stem for p in yaml_dir.glob("*.yaml"))
    if args.only:
        wanted = set(args.only.split(","))
        cells = [c for c in cells if c in wanted]
    results = pathlib.Path(args.results_dir)
    # .partial files are deliberately not counted as done.
    done = ({p.stem for p in results.glob("*.json") if not p.name.endswith(".partial.json")}
            if results.exists() else set())
    todo = [c for c in cells if c not in done]
    log(f"{len(cells)} cells, {len(done)} already done, {len(todo)} to run")

    q: queue.Queue[str] = queue.Queue()
    for c in todo:
        q.put(c)

    def worker(port: int) -> None:
        slot = Slot(port, args)
        while True:
            try:
                cell = q.get_nowait()
            except queue.Empty:
                log(f"slot {port}: queue drained")
                return
            try:
                slot.run_cell(cell)
            except Exception as exc:  # noqa: BLE001 - one bad cell must not kill the slot
                log(f"slot {port}: {cell} raised {exc!r}")
                slot.kill_clients()
                slot.stop_server()

    threads = [threading.Thread(target=worker, args=(int(p),), daemon=False)
               for p in args.ports.split(",")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # A slot swallows its own cell failures on purpose (one bad cell must not
    # strand the other five), which means the queue draining is not evidence
    # that the phase succeeded. Establish that separately, from the artifacts,
    # and exit non-zero if it does not hold: an unattended chain that reads a
    # zero exit status as "phase complete" would carry a hole in the sweep all
    # the way through to a published frontier.
    complete = {
        p.stem for p in results.glob("*.json") if not p.name.endswith(".partial.json")
    }
    partial = sorted(p.stem for p in results.glob("*.partial.json"))
    incomplete = sorted(set(cells) - complete)
    missing_evidence = []
    if args.per_step_dir:
        per_step = pathlib.Path(args.per_step_dir)
        for cell in sorted(complete):
            for name in (f"{cell}.jsonl", f"{cell}.merge.json"):
                if not (per_step / name).is_file():
                    missing_evidence.append(name)

    log(f"ALL-CELLS-DONE complete={len(complete)}/{len(cells)} partial={len(partial)}")
    if incomplete or missing_evidence:
        log(f"PHASE-FAILED incomplete={incomplete} partial={partial} "
            f"missing_evidence={missing_evidence}")
        raise SystemExit(1)
    log("PHASE-OK")


if __name__ == "__main__":
    main()
