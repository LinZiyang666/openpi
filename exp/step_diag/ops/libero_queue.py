"""Job queue of the LIBERO self-start round (``sdiag_libero_self``, logs/step_diag_libero_selfstart_plan.log.md):
every (env, arm, task) cell of ``envs.LIBERO_SELF_ARMS_BY_POLICY`` on libero_spatial / libero_10, 50 episodes each
(init_idx 0..49 of the frozen pruned A pool), on the pairs h100 <-> timan108 and weilandserver <-> timan107.

The round queues behind the RoboCasa ``sdiag_self13`` queue (``self13_queue.py``; plan §5): a host takes LIBERO
servers only once no RoboCasa job is waiting to be claimed, and its budget is the RoboCasa host budget minus the
servers the RoboCasa slots still hold (read from that queue's state file) minus this queue's own servers. So the
two queues never plan the same GPU / RAM, and every RoboCasa slot that exits hands its capacity to LIBERO.

Otherwise the mechanics are ``self13_queue.py``'s: one slot per server port; a slot keeps its (env, arm) server
while jobs of it remain; a single-task cell (``ops/run_lib_cell.sh``) runs on the paired worker under a per-job run
prefix; a failed cell resumes from its journal up to ``MAX_TRIES`` times; per-teacher caps apply while the other
teacher still has work; a restarted queue resumes its persisted state (done jobs skipped, running cells re-attached).

The servers and workers run from the LIBERO trees (``SERVER_REPO`` / ``WORKER_REPO``), separate from the trees the
RoboCasa round uses. Runs on weilandserver in tmux::

    tmux new -s sdlq -d "cd ~/projects/openpi && python3 -u exp/step_diag/ops/libero_queue.py run 2>&1 | tee -a /data/step_diag_libero_self/queue.out"
    python3 exp/step_diag/ops/libero_queue.py status

Log lines (``/data/step_diag_libero_self/queue.log``): ``JOB_START`` / ``JOB_DONE`` / ``JOB_FAIL`` / ``JOB_GIVEUP`` /
``SERVER_UP`` / ``SERVER_FAIL`` / ``SLOT_EXIT`` / ``ALL_DONE``.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from exp.step_diag import envs as _envs  # noqa: E402
from exp.step_diag.ops import self13_queue as RC  # noqa: E402

EXP = _envs.LIBERO_SELF_EXPERIMENT_ID
STATE_DIR = Path("/data/step_diag_libero_self")
STATE = STATE_DIR / "queue_state.json"
LOG = STATE_DIR / "queue.log"
MAX_TRIES = 3
POLL_S = 60
TEACHER_DIR = {"pi05": "pi05", "groot": "groot_tp"}
ENV_IDS = {"pi05": ("pi05_libero_spatial", "pi05_libero_10"), "groot": ("groot_libero_spatial", "groot_libero_10")}
SUITE_TAG = {"libero_spatial": "sp", "libero_10": "l10"}
COST = {"pi05": (7.8, 20.0), "groot": (6.4, 16.0)}  # per LIBERO server: GPU GB, RSS GB (smoke 2026-09-24)
SERVER_REPO = "/data/openpi_sdlib"
WORKER_REPO = "/scratch/zixuans8/step_diag/openpi_lib"
CELL_SCRIPT = f"{WORKER_REPO}/exp/step_diag/ops/run_lib_cell.sh"

# LIBERO server ports (outside the RoboCasa queue's ranges) and teacher paths per server host; the address, worker,
# home and host budgets are ``RC.HOSTS``'s
HOSTS = {
    "h100": {
        "ports": {"pi05": list(range(23260, 23268)), "groot": list(range(23270, 23280))},
        "env": {"pi05": "SD_PY=/home/exouser/openpi/.venv/bin/python SD_CKPT=/home/exouser/ckpt/pi05_libero_pytorch",
                "groot": "SD_GROOT=/home/exouser/gr00t_n15 SD_GROOT_PY=/home/exouser/gr00t_n15_venv/.venv/bin/python"},
        "groot_ckpt": "/home/exouser/ckpt/n15_libero_{suite}",
        "worker_gpus": ["0", "1", "2"],
    },
    "wls": {
        "ports": {"pi05": list(range(23160, 23168)), "groot": list(range(23170, 23180))},
        "env": {"pi05": "SD_PY=/home/weiland/projects/openpi/.venv/bin/python "
                        "SD_CKPT=/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch",
                "groot": "SD_GROOT=/home/weiland/gr00t_n15 SD_GROOT_PY=/home/weiland/gr00t_n15_venv/.venv/bin/python"},
        "groot_ckpt": "/home/weiland/ckpt_n15_libero_{suite}",
        "worker_gpus": ["0", "1", "2", "4", "5", "6", "7"],  # GPU 3 of timan107 belongs to another project
    },
}
# per-host server caps per teacher while the other teacher still has work (LIBQ_CAPS overrides); GR00T has
# 29 arms to pi0.5's 9
DEFAULT_CAPS = {"h100": {"pi05": 3, "groot": 7}, "wls": {"pi05": 2, "groot": 4}}

_lock = threading.Lock()
_stop = threading.Event()


def log(msg: str) -> None:
    """Print one timestamped queue event and append it to ``LOG``."""
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


# ------------------------------------------------------------------
# Jobs and state
# ------------------------------------------------------------------


def suite_of(env_id: str) -> str:
    """The LIBERO suite of an environment id (``groot_libero_10`` -> ``libero_10``)."""
    return _envs.resolve_env(env_id).benchmark


def build_jobs() -> list[dict]:
    """Every (teacher, env, arm, task) cell of the round as a pending job."""
    jobs = []
    for teacher, env_ids in ENV_IDS.items():
        for env_id in env_ids:
            for arm in _envs.LIBERO_SELF_ARMS_BY_POLICY[teacher]:
                for task in range(_envs.LIBERO_N_TASKS):
                    jobs.append({"id": f"{teacher}|{env_id}|{arm}|{task}", "teacher": teacher, "env_id": env_id,
                                 "arm": arm, "task": task, "status": "pending", "tries": 0, "slot": None, "cell": None})
    return jobs


def load_state() -> dict:
    """The persisted queue state, or a fresh one."""
    if not STATE.exists():
        return {"jobs": build_jobs(), "slots": {}}
    return json.loads(STATE.read_text())


def save_state(state: dict) -> None:
    """Persist the queue state atomically (write a temporary file, then rename it over ``STATE``)."""
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, STATE)


# ------------------------------------------------------------------
# Budget shared with the RoboCasa queue
# ------------------------------------------------------------------


def rc_usage(host: str, rc_state_path: Path = RC.STATE) -> tuple[bool, float, float]:
    """``(RoboCasa blocks, GPU GB, RSS GB)`` for ``host``: it blocks while a RoboCasa job waits to be claimed and a
    RoboCasa slot is alive to claim it; the GB are the servers the live RoboCasa slots on ``host`` hold. A missing
    state file means no RoboCasa round; an unreadable one blocks (retried next poll)."""
    try:
        state = json.loads(rc_state_path.read_text())
    except FileNotFoundError:
        return False, 0.0, 0.0
    except (OSError, ValueError):
        return True, 0.0, 0.0
    slots = state.get("slots", {})
    blocks = bool(slots) and any(j["status"] == "pending" for j in state["jobs"])
    gpu = ram = 0.0
    for key in slots:
        slot_host, port = key.rsplit(":", 1)
        if slot_host != host:
            continue
        teacher = "pi05" if int(port) in RC.HOSTS[host]["ports"]["pi05"] else "groot"
        gpu += RC.COST[teacher][0]
        ram += RC.COST[teacher][1]
    return blocks, gpu, ram


class Budget:
    """This queue's servers per host and teacher, fitted into the RoboCasa host budget next to the RoboCasa
    queue's live servers; a per-teacher cap applies while the other teacher still has work."""

    def __init__(self, caps: dict, rc_state_path: Path = RC.STATE) -> None:
        self.used = {h: [0.0, 0.0] for h in HOSTS}
        self.count = {h: {"pi05": 0, "groot": 0} for h in HOSTS}
        self.caps = caps
        self.rc_state_path = rc_state_path

    def try_take(self, host: str, teacher: str, other_pending: bool, force: bool = False) -> bool:
        """Reserve one ``teacher`` server on ``host`` if RoboCasa does not block the host, it fits next to the
        RoboCasa servers, and the teacher is under its cap; ``force`` reserves regardless (resumed cells)."""
        g, r = COST[teacher]
        u = self.used[host]
        if not force:
            blocks, rc_g, rc_r = rc_usage(host, self.rc_state_path)
            cap = self.caps[host][teacher] if other_pending else 99
            fits = (u[0] + rc_g + g <= RC.HOSTS[host]["gpu_budget"] and u[1] + rc_r + r <= RC.HOSTS[host]["ram_budget"])
            if blocks or not fits or self.count[host][teacher] >= cap:
                return False
        u[0] += g
        u[1] += r
        self.count[host][teacher] += 1
        return True

    def give_back(self, host: str, teacher: str) -> None:
        """Release one server reservation of ``teacher`` on ``host``."""
        g, r = COST[teacher]
        self.used[host][0] -= g
        self.used[host][1] -= r
        self.count[host][teacher] -= 1


# ------------------------------------------------------------------
# Servers and cells
# ------------------------------------------------------------------


def server_args(env_id: str, arm: str) -> tuple[str, str]:
    """``(mode, last argument)`` of ``ops/serve_{pi05,groot}.sh`` for one arm: full ``-``, plain its step count,
    every warm-family arm (ours, cache and self warm reset) its env's ``warm_t<start_t>.yaml``."""
    if arm == "full":
        return "full", "-"
    if arm.startswith("plain_k"):
        return "plain", arm[len("plain_k"):]
    t = _envs.warm_t_of(arm)
    return _envs.warm_mode_of(arm), f"{SERVER_REPO}/exp/step_diag/config/arms/{env_id}/warm_t{t:g}.yaml"


def server_out(teacher: str, env_id: str, arm: str) -> str:
    """The server's diagnostics cell (its ``manifest_<arm>.json`` holds the config_sha)."""
    return f"{SERVER_REPO}/exp/step_diag/data/server_libero_self/{TEACHER_DIR[teacher]}/{env_id}/{arm}"


def server_env(host: str, teacher: str, env_id: str) -> str:
    """The shell ``export`` prefix of a server launch on ``host`` (experiment id, out root, teacher paths)."""
    h, rc = HOSTS[host], RC.HOSTS[host]
    ckpt = f" SD_CKPT={h['groot_ckpt'].format(suite=suite_of(env_id).removeprefix('libero_'))}" if teacher == "groot" else ""
    return (f"export HOME={rc['home']} SD_REPO={SERVER_REPO} SD_EXP={EXP} "
            f"SD_OUT={SERVER_REPO}/exp/step_diag/data/server_libero_self SD_HOME={rc['home']} {h['env'][teacher]}{ckpt}")


def stop_server(host: str, port: int) -> None:
    """Stop this queue's server on ``port`` of ``host`` (``ops/stop_servers.sh``: our sdsrv session only)."""
    RC.sh(host, f"cd {SERVER_REPO} && bash exp/step_diag/ops/stop_servers.sh {port}")


def _server_probe(host: str, teacher: str, env_id: str, arm: str, port: int) -> str | None:
    manifest = f"{server_out(teacher, env_id, arm)}/manifest_{arm}.json"
    probe = (f"ss -tlnH sport = :{port} | grep -q . && grep -q 'STEP_DIAG arm={arm} mode=.* env={env_id} ' "
             f"/tmp/sdiag/sdsrv{port}.log && python3 -c \"import json;print('CS='+json.load(open('{manifest}'))['config_sha'])\"")
    _, out = RC.sh(host, probe)
    for tok in out.split():
        if tok.startswith("CS=") and len(tok) > 20:
            return tok[3:]
    return None


def start_server(host: str, teacher: str, env_id: str, arm: str, port: int) -> str | None:
    """Start the (env, arm) server on ``port`` and wait until it listens; returns its config_sha or None.
    A server of the same (env, arm) already listening on the port (queue restart) is reused as is."""
    live = _server_probe(host, teacher, env_id, arm, port)
    if live is not None:
        return live
    mode, arg = server_args(env_id, arm)
    script = f"exp/step_diag/ops/serve_{'pi05' if teacher == 'pi05' else 'groot'}.sh"
    launch = (f"cd {SERVER_REPO} && {server_env(host, teacher, env_id)} && mkdir -p /tmp/sdiag && "
              f"nohup bash {script} {env_id} {mode} {arm} {port} {arg} >> /tmp/sdiag/serve_{port}.out 2>&1 < /dev/null &")
    stop_server(host, port)  # clears a stale claim of this port (our sdsrv session only)
    RC.sh(host, launch)
    deadline = time.time() + 1500
    while time.time() < deadline and not _stop.is_set():
        time.sleep(30)
        cs = _server_probe(host, teacher, env_id, arm, port)
        if cs is not None:
            return cs
        _, tail = RC.sh(host, f"tail -3 /tmp/sdiag/serve_{port}.out 2>/dev/null")
        if "readiness failed" in tail or "refusing" in tail or "missing" in tail:
            log(f"SERVER_FAIL host={host} port={port} env={env_id} arm={arm}: {tail.strip()[-200:]}")
            return None
    log(f"SERVER_FAIL host={host} port={port} env={env_id} arm={arm}: not ready in time")
    return None


def cell_tag(port: int, job: dict) -> str:
    """The cell launcher tag of one (slot, suite, task): names its tmux session and log."""
    return f"q{port}{SUITE_TAG[suite_of(job['env_id'])]}t{job['task']}"


def cell_log(port: int, job: dict) -> str:
    """Path of one cell's log on the worker host (``ops/run_lib_cell.sh`` writes ``SDCELL_EXIT`` there)."""
    return f"/tmp/sdiag/sdlib_{cell_tag(port, job)}_{job['arm'].replace('.', '_')}_{EXP}.log"


def start_cell(host: str, job: dict, port: int, cs: str, gpu: str) -> tuple[int, str]:
    """Launch (or re-attach) one single-task cell of ``job`` on the paired worker against ``port``; returns the
    launcher's exit code and output. The run prefix is fixed at the job's first start, so a retry on the same slot
    resumes the same journal, and names the task, so sequential cells of one arm on a slot never share a run id."""
    rc = RC.HOSTS[host]
    job.setdefault("prefix", f"sdl{cell_tag(port, job)}")
    env = (f"export SD_EXP={EXP} SD_LIB_REPO={WORKER_REPO} SD_OUT_ROOT={WORKER_REPO}/exp/step_diag/data/libero_self "
           f"SD_TMUX_SOCKET=sdiag SD_RUN_PREFIX={job['prefix']} SD_GPUS={gpu}")
    cmd = (f"cd {WORKER_REPO} && {env} && bash {CELL_SCRIPT} {job['env_id']} {job['arm']} {rc['addr']}:{port} "
           f"{job['task']} {cs} {cell_tag(port, job)}")
    return RC.sh(rc["worker"], cmd)


def reap_orphans(host: str, port: int) -> None:
    """Terminate this slot's orphaned LIBERO workers: ``run_libero_diag --role worker`` processes re-parented to init
    (their driver is gone) whose ``--server-key`` is this slot's server."""
    key = f"{RC.HOSTS[host]['addr']}:{port}"
    cmd = ("ps -eo pid=,ppid=,args= | awk -v key='--server-key " + key + " ' "
           "'$2 == 1 && /exp\\.step_diag\\.run_libero_diag/ && /--role worker/ && index($0, key) {print $1}' | "
           "while read -r pid; do kill \"$pid\" && echo REAPED $pid; done; true")
    _, out = RC.sh(RC.HOSTS[host]["worker"], cmd)
    reaped = [tok for tok in out.split() if tok.isdigit()]
    if reaped:
        log(f"REAPED {host}:{port} orphan workers {reaped}")


# ------------------------------------------------------------------
# Slot threads
# ------------------------------------------------------------------


def claim(state: dict, teacher: str, current: tuple[str, str] | None, key: str) -> dict | None:
    """Mark and return the next pending job of ``teacher`` for slot ``key``: the (env, arm) already served there if
    any, else the (env, arm) with the most pending jobs; a job pinned to another slot is skipped."""
    pending = [j for j in state["jobs"] if j["teacher"] == teacher and j["status"] == "pending"
               and j.get("pin") in (None, key)]
    if not pending:
        return None
    same = [j for j in pending if (j["env_id"], j["arm"]) == current]
    if same:
        job = same[0]
    else:
        counts: dict[tuple[str, str], int] = {}
        for j in pending:
            counts[(j["env_id"], j["arm"])] = counts.get((j["env_id"], j["arm"]), 0) + 1
        best = max(counts, key=lambda k: (counts[k], k))
        job = next(j for j in pending if (j["env_id"], j["arm"]) == best)
    job["status"] = "running"
    return job


def slot_main(state: dict, budget: Budget, host: str, teacher: str, port: int, gpu: str, resume_job: str | None) -> None:
    """One server slot's loop: take budget, claim a job, (re)start the server when the (env, arm) changes, run the
    cell to ``SDCELL_EXIT`` with up to ``MAX_TRIES`` resumed relaunches; ``resume_job`` re-attaches a cell running at
    restart."""
    key = f"{host}:{port}"
    other = "groot" if teacher == "pi05" else "pi05"
    loaded: tuple[str, str] | None = None
    cs: str | None = None
    holding = resume_job is not None  # run() took the budget of every resumed job before any slot started
    attach = resume_job is not None
    try:
        while not _stop.is_set():
            job = None
            with _lock:
                if resume_job is not None:
                    job = next((j for j in state["jobs"] if j["id"] == resume_job and j["status"] == "running"), None)
                    resume_job = None
                if job is None:
                    if not any(j["teacher"] == teacher and j["status"] == "pending" for j in state["jobs"]):
                        break
                    other_busy = any(j["teacher"] == other and j["status"] in ("pending", "running") for j in state["jobs"])
                    if not holding:
                        holding = budget.try_take(host, teacher, other_busy)
                    if holding:
                        job = claim(state, teacher, loaded, key)
                        if job is None:
                            break
                if job is not None:
                    job["slot"] = key
                    state["slots"][key] = {"env_id": job["env_id"], "arm": job["arm"], "job": job["id"]}
                    save_state(state)
            if job is None:  # no budget yet: wait for RoboCasa or another slot to release some
                time.sleep(POLL_S)
                continue
            if loaded != (job["env_id"], job["arm"]) or cs is None:
                cs = start_server(host, teacher, job["env_id"], job["arm"], port)
                if cs is None:
                    with _lock:
                        job["status"], job["slot"] = "pending", None
                        save_state(state)
                    log(f"SLOT_EXIT {key} server did not come up; job {job['id']} requeued")
                    return
                loaded = (job["env_id"], job["arm"])
                log(f"SERVER_UP {key} env={job['env_id']} arm={job['arm']} cs={cs[:12]}")
            logpath = cell_log(port, job)
            while not _stop.is_set():
                code = None
                if attach and job.get("cell"):
                    code = RC.cell_exit(host, logpath)  # a re-attached cell may have finished while the queue was down
                if code is None:
                    if not attach:
                        job["tries"] += 1
                    rc, out = start_cell(host, job, port, cs, gpu)
                    job["cell"] = logpath
                    with _lock:
                        save_state(state)
                    log(f"JOB_START {job['id']} slot={key} try={job['tries']} :: {out.strip()[-160:]}")
                attach = False
                while code is None and not _stop.is_set():
                    time.sleep(POLL_S)
                    code = RC.cell_exit(host, logpath)
                if _stop.is_set():
                    return
                reap_orphans(host, port)
                if code == 0:
                    with _lock:
                        job["status"] = "done"
                        save_state(state)
                    log(f"JOB_DONE {job['id']} slot={key}")
                    break
                log(f"JOB_FAIL {job['id']} slot={key} exit={code} try={job['tries']}")
                if job["tries"] >= MAX_TRIES:
                    with _lock:
                        job["status"] = "failed"
                        save_state(state)
                    log(f"JOB_GIVEUP {job['id']} after {job['tries']} tries")
                    break
                # keep the failed log; the relaunched cell resumes from its journal
                RC.sh(RC.HOSTS[host]["worker"], f"mv {logpath} {logpath}.try{job['tries']} 2>/dev/null; true")
                _, up = RC.sh(host, f"ss -tlnH sport = :{port} | grep -q . && echo UP")
                if "UP" not in up:
                    cs = start_server(host, teacher, job["env_id"], job["arm"], port)
                    if cs is None:
                        with _lock:
                            job["status"], job["slot"], job["pin"] = "pending", None, key
                            save_state(state)
                        log(f"SLOT_EXIT {key} server lost; job {job['id']} requeued (pinned to {key})")
                        return
    finally:
        stop_server(host, port)
        with _lock:
            if holding:
                budget.give_back(host, teacher)
            state["slots"].pop(key, None)
            save_state(state)
        log(f"SLOT_EXIT {key} teacher={teacher}")


def run() -> None:
    """Start the queue: load / resume the state, reserve budget for resumed cells, start every slot thread and wait."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    for j in state["jobs"]:
        if j["status"] == "running" and not j.get("slot"):
            j["status"] = "pending"
    save_state(state)
    caps = json.loads(os.environ.get("LIBQ_CAPS", json.dumps(DEFAULT_CAPS)))
    budget = Budget(caps)
    for j in state["jobs"]:  # resumed cells hold their server before any slot may claim new work
        if j["status"] == "running" and j.get("slot"):
            budget.try_take(j["slot"].split(":")[0], j["teacher"], True, force=True)
    log(f"QUEUE_START caps={caps} jobs={len(state['jobs'])} done={sum(j['status'] == 'done' for j in state['jobs'])}")
    threads = []
    for host, h in HOSTS.items():
        order = []
        for i in range(max(len(h["ports"]["pi05"]), len(h["ports"]["groot"]))):
            for teacher in ("groot", "pi05"):
                if i < len(h["ports"][teacher]):
                    order.append((teacher, h["ports"][teacher][i]))
        for n, (teacher, port) in enumerate(order):
            key = f"{host}:{port}"
            resume = next((j["id"] for j in state["jobs"] if j["status"] == "running" and j.get("slot") == key), None)
            gpu = h["worker_gpus"][n % len(h["worker_gpus"])]
            t = threading.Thread(target=slot_main, args=(state, budget, host, teacher, port, gpu, resume), daemon=True,
                                 name=key)
            threads.append(t)
            t.start()
            time.sleep(5)  # stagger the slots' first budget checks
    for t in threads:
        t.join()
    with _lock:
        left = [j["id"] for j in state["jobs"] if j["status"] != "done"]
    log(f"ALL_DONE done={sum(j['status'] == 'done' for j in state['jobs'])}/{len(state['jobs'])} not_done={left}")


def status() -> None:
    """Print the job counts per teacher and status, and what every busy slot runs."""
    state = load_state()
    by: dict[tuple[str, str], int] = {}
    for j in state["jobs"]:
        by[(j["teacher"], j["status"])] = by.get((j["teacher"], j["status"]), 0) + 1
    print(json.dumps({f"{t}:{s}": n for (t, s), n in sorted(by.items())}))
    for k, v in sorted(state["slots"].items()):
        print(k, v)


if __name__ == "__main__":
    {"run": run, "status": status}[sys.argv[1] if len(sys.argv) > 1 else "status"]()
