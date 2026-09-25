"""Job queue of the self-start ablation round (``sdiag_self13``, owner 2026-09-24): every (arm, task) cell of
``envs.SELF13_ARMS_BY_POLICY`` x the 13-task roster, 50 episodes each, seed 2M, on the two server/worker pairs
h100 <-> timan108 and weilandserver <-> timan107.

Each server port is a slot. A slot thread takes the next job of its teacher (preferring the arm its server
already runs, else the arm with the most pending jobs), (re)starts its server if the arm changes, launches a
single-task cell on the paired worker (``ops/run_rc_cell.sh`` with a per-slot ``SD_RUN_PREFIX`` / tag, so
concurrent cells of one arm never share a journal file) and waits for ``SDCELL_EXIT``. A failed cell is
relaunched (the driver resumes from its journal) up to ``MAX_TRIES`` times. A slot may only hold a server while
its host's GPU / RAM budget allows, so when one teacher's queue drains, its budget goes to the other teacher's
waiting slots (no machine idles while work is left). State is persisted after every change; a restarted queue
resumes (done jobs are skipped, running cells are re-attached through the idempotent cell launcher).

Runs on weilandserver in tmux::

    tmux new -s sdq -d "cd ~/projects/openpi && python3 exp/step_diag/ops/self13_queue.py run 2>&1 | tee -a /data/step_diag_self13/queue.out"
    python3 exp/step_diag/ops/self13_queue.py status

Log lines (``/data/step_diag_self13/queue.log``): ``JOB_START`` / ``JOB_DONE`` / ``JOB_FAIL`` / ``JOB_GIVEUP`` /
``SERVER_UP`` / ``SERVER_FAIL`` / ``SLOT_EXIT`` / ``ALL_DONE``.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from exp.step_diag import envs as _envs  # noqa: E402

EXP = _envs.SELF13_EXPERIMENT_ID
EPISODES = _envs.SELF13_EPISODES
SEED = _envs.RC_FORMAL_BASE_SEED
STATE_DIR = Path("/data/step_diag_self13")
STATE = STATE_DIR / "queue_state.json"
LOG = STATE_DIR / "queue.log"
MAX_TRIES = 3
POLL_S = 60
ROSTER_MAIN = tuple(_envs.RC_MAIN_LANE)
ROSTER_PNP = tuple(_envs.RC_PNP_LANE)
TEACHER_DIR = {"pi05": "pi05", "groot": "groot_tp"}
ENV_ID = {"pi05": "pi05_rc", "groot": "groot_rc"}
COST = {"pi05": (7.8, 31.0), "groot": (6.4, 21.0)}  # measured per server: GPU GB, RSS GB (2026-09-24)

HOSTS = {
    "h100": {
        "remote": True, "repo": "/data/openpi_sdiag", "home": "/home/exouser", "gpu_budget": 76.0, "ram_budget": 175.0,
        "addr": "149.165.153.233", "worker": "timan108", "worker_env": "", "worker_gpus": ["0", "1", "2"],
        "ports": {"pi05": list(range(23240, 23248)), "groot": list(range(23250, 23260))},
        "env": {"pi05": "SD_PY=/home/exouser/openpi/.venv/bin/python SD_CKPT=/home/exouser/ckpt/pi05_robocasa_pytorch",
                "groot": "SD_GROOT=/home/exouser/gr00t_n15 SD_GROOT_PY=/home/exouser/gr00t_n15_venv/.venv/bin/python "
                         "SD_CKPT=/home/exouser/ckpt/n15_robocasa_tp/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000"},
    },
    "wls": {
        "remote": False, "repo": "/data/openpi_sdiag", "home": "/home/weiland", "gpu_budget": 44.0, "ram_budget": 220.0,
        "addr": "ziyanglin.com", "worker": "timan107",
        "worker_env": "SD_RC_ENV=/scratch/zixuans8/step_diag/openpi/exp/step_diag/config/rc_timan107.env",
        "worker_gpus": ["0", "1", "2", "4", "5", "6", "7"],  # GPU 3 belongs to another project
        "ports": {"pi05": list(range(23140, 23148)), "groot": list(range(23150, 23160))},
        "env": {"pi05": "SD_PY=/home/weiland/projects/openpi/.venv/bin/python SD_CKPT=/data/ckpt/pi05_robocasa_pytorch",
                "groot": "SD_GROOT=/home/weiland/gr00t_n15 SD_GROOT_PY=/home/weiland/gr00t_n15_venv/.venv/bin/python "
                         "SD_CKPT=/home/weiland/ckpt_n15_robocasa_tp/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000"},
    },
}
# initial per-host server caps per teacher while the other teacher still has work (SDQ_CAPS overrides)
DEFAULT_CAPS = {"h100": {"pi05": 4, "groot": 2}, "wls": {"pi05": 5, "groot": 0}}  # pi0.5 first (owner 2026-09-24)
WORKER_REPO = "/scratch/zixuans8/step_diag/openpi"
CELL_SCRIPT = "/tmp/sdiag/run_rc_cell.sh.self"

_lock = threading.Lock()
_stop = threading.Event()


def log(msg: str) -> None:
    """Print one timestamped queue event and append it to ``LOG``."""
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def sh(host: str, cmd: str, timeout: int = 110) -> tuple[int, str]:
    """Run ``cmd`` with bash on ``host`` (local for wls, tether exec otherwise); never raises."""
    if host == "wls":
        argv = ["bash", "-lc", cmd]
    else:
        argv = ["tether", "exec", "--timeout", f"{timeout - 10}s", host, "--", "bash", "-lc", cmd]
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout + 30)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as exc:  # noqa: BLE001 - a transport hiccup is retried by the caller
        return 99, f"{type(exc).__name__}: {exc}"


# -- state ------------------------------------------------------------------------------------------------


def build_jobs() -> list[dict]:
    """Every (teacher, arm, task) cell of the round as a pending job (arm table x the 13-task roster, by lane)."""
    jobs = []
    for teacher, arms in _envs.SELF13_ARMS_BY_POLICY.items():
        for arm in arms:
            for lane, tasks in (("main", ROSTER_MAIN), ("pnp", ROSTER_PNP)):
                for task in tasks:
                    jobs.append({"id": f"{teacher}|{arm}|{task}", "teacher": teacher, "arm": arm, "lane": lane,
                                 "task": task, "status": "pending", "tries": 0, "slot": None, "cell": None})
    return jobs


def load_state() -> dict:
    """The persisted queue state, or a fresh one; jobs of arms added to the round since the last start are appended."""
    if not STATE.exists():
        return {"jobs": build_jobs(), "slots": {}}
    state = json.loads(STATE.read_text())
    known = {j["id"] for j in state["jobs"]}
    added = [j for j in build_jobs() if j["id"] not in known]  # arms added to the round since the last start
    state["jobs"].extend(added)
    if added:
        print(f"queue: {len(added)} new jobs appended", flush=True)
    return state


def save_state(state: dict) -> None:
    """Persist the queue state atomically (write a temporary file, then rename it over ``STATE``)."""
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, STATE)


class Budget:
    """Per-host GPU / RAM budget plus a per-teacher server cap; the cap lifts once the other teacher has no
    pending job, so a drained queue hands its capacity to the other teacher."""

    def __init__(self, caps: dict) -> None:
        self.used = {h: [0.0, 0.0] for h in HOSTS}
        self.count = {h: {"pi05": 0, "groot": 0} for h in HOSTS}
        self.caps = caps

    def try_take(self, host: str, teacher: str, other_pending: bool, force: bool = False) -> bool:
        """Reserve one server of ``teacher`` on ``host`` if it fits the GPU / RAM budget and the teacher's cap (the cap
        applies only while the other teacher has work); ``force`` reserves regardless (resumed cells)."""
        g, r = COST[teacher]
        u = self.used[host]
        cap = self.caps[host][teacher] if other_pending else 99
        fits = u[0] + g <= HOSTS[host]["gpu_budget"] and u[1] + r <= HOSTS[host]["ram_budget"]
        if force or (fits and self.count[host][teacher] < cap):
            u[0] += g
            u[1] += r
            self.count[host][teacher] += 1
            return True
        return False

    def give_back(self, host: str, teacher: str) -> None:
        """Release one server reservation of ``teacher`` on ``host``."""
        g, r = COST[teacher]
        self.used[host][0] -= g
        self.used[host][1] -= r
        self.count[host][teacher] -= 1


# -- servers and cells ------------------------------------------------------------------------------------


def yaml_for(host: str, teacher: str, arm: str) -> str:
    """The warm yaml an arm serves on ``host`` (a self / shoot arm serves its cache arm's ``warm_t<start_t>.yaml``)."""
    t = _envs.warm_t_of(arm)
    return f"{HOSTS[host]['repo']}/exp/step_diag/config/arms/{ENV_ID[teacher]}/warm_t{t:g}.yaml"


def server_env(host: str, teacher: str) -> str:
    """The shell ``export`` prefix of a server launch on ``host`` (experiment id, server out root, teacher paths)."""
    h = HOSTS[host]
    return (f"export HOME={h['home']} SD_REPO={h['repo']} SD_EXP={EXP} SD_OUT={h['repo']}/exp/step_diag/data/server_self13 "
            f"SD_HOME={h['home']} {h['env'][teacher]}")


def stop_server(host: str, port: int) -> None:
    """Stop this queue's server on ``port`` of ``host`` (``ops/stop_servers.sh``, our session only)."""
    sh(host, f"cd {HOSTS[host]['repo']} && bash exp/step_diag/ops/stop_servers.sh {port}")


def _server_probe(host: str, teacher: str, arm: str, port: int) -> str | None:
    h = HOSTS[host]
    manifest = f"{h['repo']}/exp/step_diag/data/server_self13/{TEACHER_DIR[teacher]}/{arm}/manifest_{arm}.json"
    probe = (f"ss -tlnH sport = :{port} | grep -q . && grep -q STEP_DIAG.arm={arm}.mode= /tmp/sdiag/sdsrv{port}.log "
             f"&& python3 -c \"import json;print('CS='+json.load(open('{manifest}'))['config_sha'])\"")
    rc, out = sh(host, probe)
    for tok in out.split():
        if tok.startswith("CS=") and len(tok) > 20:
            return tok[3:]
    return None


def start_server(host: str, teacher: str, arm: str, port: int) -> str | None:
    """Start the arm's server on ``port`` and wait until it listens; returns the arm's config_sha or None.
    A server of the same arm already listening on the port (queue restart) is reused as is."""
    h = HOSTS[host]
    live = _server_probe(host, teacher, arm, port)
    if live is not None:
        return live
    mode = _envs.warm_mode_of(arm)
    script = f"exp/step_diag/ops/serve_{'pi05' if teacher == 'pi05' else 'groot'}.sh"
    launch = (f"cd {h['repo']} && {server_env(host, teacher)} && mkdir -p /tmp/sdiag && "
              f"nohup bash {script} {ENV_ID[teacher]} {mode} {arm} {port} {yaml_for(host, teacher, arm)} "
              f">> /tmp/sdiag/serve_{port}.out 2>&1 < /dev/null &")
    stop_server(host, port)  # clears a stale claim of this port (our sdsrv session only)
    sh(host, launch)
    deadline = time.time() + 1500
    while time.time() < deadline and not _stop.is_set():
        time.sleep(30)
        cs = _server_probe(host, teacher, arm, port)
        if cs is not None:
            return cs
        rc2, out2 = sh(host, f"tail -3 /tmp/sdiag/serve_{port}.out 2>/dev/null")
        if "readiness failed" in out2 or "refusing" in out2:
            log(f"SERVER_FAIL host={host} port={port} arm={arm}: {out2.strip()[-200:]}")
            return None
    log(f"SERVER_FAIL host={host} port={port} arm={arm}: not ready in time")
    return None


def cell_tag(port: int, task: str) -> str:
    """The cell launcher tag of one (slot, task): names its tmux session and log."""
    return f"q{port}{task}"


def cell_log(teacher: str, arm: str, lane: str, port: int, task: str) -> str:
    """Path of one cell's log on the worker host (``ops/run_rc_cell.sh`` writes ``SDCELL_EXIT`` there)."""
    return f"/tmp/sdiag/sdcell_{cell_tag(port, task)}_{TEACHER_DIR[teacher]}_{arm.replace('.', '_')}_{lane}_{EXP}.log"


def start_cell(host: str, job: dict, port: int, cs: str, gpu: str) -> tuple[int, str]:
    """Launch (or re-attach) one single-task cell of ``job`` on the paired worker against ``port``; returns the
    launcher's exit code and output. The job's run prefix is fixed at its first start (see below)."""
    h = HOSTS[host]
    # The driver's run_id is <prefix>-<arm>-<lane>-<sha(exp, config)>: without the task in the prefix, the next task
    # of the same arm on the same slot would reuse the previous task's journal / run plan (refused as a mismatch).
    # Fixed at the job's first start so a retry on the same slot resumes the same journal.
    job.setdefault("prefix", f"sdq{port}{job['task']}")
    env = (f"export SD_EXP={EXP} SD_BASE_SEED={SEED} SD_OUT_ROOT={WORKER_REPO}/exp/step_diag/data/rc_self13 "
           f"SD_TMUX_SOCKET=sdiag SD_RUN_PREFIX={job['prefix']} SD_GPUS={gpu} {h['worker_env']}")
    cmd = (f"cd {WORKER_REPO} && {env} && bash {CELL_SCRIPT} {TEACHER_DIR[job['teacher']]} {job['arm']} {job['lane']} "
           f"{h['addr']}:{port} {job['task']} {EPISODES} - {cs} {cell_tag(port, job['task'])}")
    return sh(h["worker"], cmd)


def cell_exit(host: str, logpath: str) -> int | None:
    """Exit code of a cell from its log, or None while it runs.

    ``grep -a``: a log can read as binary (NUL holes written by an orphaned worker through a stale descriptor).
    A cell whose tmux session is gone without an ``SDCELL_EXIT`` marker counts as 0 when the driver printed its
    ``[step_diag] DONE`` line (the journal is complete) and as 1 otherwise (the relaunch resumes the journal).
    Transport errors and incomplete probes remain unknown and never spend a cell retry."""
    session = Path(logpath).name[: -len(".log")]
    path = shlex.quote(logpath)
    probe = (
        f"if tmux -L sdiag has-session -t {shlex.quote('=' + session)} 2>/dev/null; then "
        "echo SESSION_STATE=alive; else probe_rc=$?; "
        'if [ "$probe_rc" -eq 1 ]; then echo SESSION_STATE=gone; else exit "$probe_rc"; fi; fi; '
        f"if [ -f {path} ]; then "
        f"grep -a -h -o 'SDCELL_EXIT=[0-9][0-9]*' {path} 2>/dev/null | tail -1; "
        f"done_lines=$(grep -a -c '\\[step_diag\\] DONE arm=' {path} 2>/dev/null); probe_rc=$?; "
        '[ "$probe_rc" -le 1 ] || exit "$probe_rc"; '
        'else done_lines=0; fi; echo DONE_LINES="$done_lines"; echo CELL_PROBE_DONE'
    )
    rc, out = sh(HOSTS[host]["worker"], probe)
    tokens = out.split()
    if rc != 0 or "CELL_PROBE_DONE" not in tokens:
        return None
    if "SESSION_STATE=alive" in tokens or "SESSION_STATE=gone" not in tokens:
        return None
    done = next((tok.split("=", 1)[1] for tok in tokens if tok.startswith("DONE_LINES=")), "")
    if not done.isdigit():
        return None
    for tok in tokens:
        if tok.startswith("SDCELL_EXIT="):
            code = tok.split("=", 1)[1]
            return int(code) if code.isdigit() else None
    return 0 if int(done) > 0 else 1


def reap_orphans(host: str, port: int) -> None:
    """Terminate this slot's orphaned workers: ``worker_entry`` processes re-parented to init (their driver is gone)
    whose ``--server-key`` is this slot's server. They otherwise keep retrying the old driver port and write into the
    reused cell log through a stale descriptor."""
    key = f"{HOSTS[host]['addr']}:{port}"
    cmd = ("ps -eo pid=,ppid=,args= | awk -v key='--server-key " + key + " ' "
           "'$2 == 1 && /exp\\.step_diag\\.worker_entry/ && index($0, key) {print $1}' | "
           "while read -r pid; do kill \"$pid\" && echo REAPED $pid; done; true")
    rc, out = sh(HOSTS[host]["worker"], cmd)
    reaped = [tok for tok in out.split() if tok.isdigit()]
    if reaped:
        log(f"REAPED {host}:{port} orphan workers {reaped}")


# -- slot threads -----------------------------------------------------------------------------------------


def claim(state: dict, teacher: str, current_arm: str | None, key: str) -> dict | None:
    """Mark and return the next pending job of ``teacher`` for slot ``key``: the arm already served there if any,
    else the arm with the most pending jobs; a job pinned to another slot is skipped. None when nothing is left."""
    # a job that already ran under a run prefix may only resume on the slot (server) its run plan names
    pending = [j for j in state["jobs"] if j["teacher"] == teacher and j["status"] == "pending"
               and j.get("pin") in (None, key)]
    if not pending:
        return None
    same = [j for j in pending if j["arm"] == current_arm]
    if same:
        job = same[0]
    else:
        counts: dict[str, int] = {}
        for j in pending:
            counts[j["arm"]] = counts.get(j["arm"], 0) + 1
        arm = max(counts, key=lambda a: (counts[a], a))
        job = next(j for j in pending if j["arm"] == arm)
    job["status"] = "running"
    return job


def slot_main(state: dict, budget: Budget, host: str, teacher: str, port: int, gpu: str, resume_job: str | None) -> None:
    """One server slot's loop: take budget, claim a job, (re)start the server when the arm changes, run the cell to
    ``SDCELL_EXIT`` with up to ``MAX_TRIES`` resumed relaunches; ``resume_job`` re-attaches a cell running at restart."""
    key = f"{host}:{port}"
    other = "groot" if teacher == "pi05" else "pi05"
    arm_loaded: str | None = None
    cs: str | None = None
    holding = resume_job is not None  # run() took the budget of every resumed job before any slot started
    attach = resume_job is not None
    try:
        while not _stop.is_set():
            job = None
            release = False
            with _lock:
                if resume_job is not None:  # a cell this slot was running when the queue stopped
                    job = next((j for j in state["jobs"] if j["id"] == resume_job and j["status"] == "running"), None)
                    resume_job = None
                if job is None:
                    if not any(j["teacher"] == teacher and j["status"] == "pending" for j in state["jobs"]):
                        break
                    other_busy = any(j["teacher"] == other and j["status"] in ("pending", "running") for j in state["jobs"])
                    if holding and other_busy and budget.count[host][teacher] > budget.caps[host][teacher]:
                        # over this teacher's cap (caps changed at restart): hand the capacity to the other teacher
                        budget.give_back(host, teacher)
                        holding = False
                        release = True
                    else:
                        if not holding:
                            holding = budget.try_take(host, teacher, other_busy)
                        if holding:
                            job = claim(state, teacher, arm_loaded, key)
                            if job is None:
                                break
                if job is not None:
                    job["slot"] = key
                    state["slots"][key] = {"arm": job["arm"], "job": job["id"]}
                    save_state(state)
            if job is None:  # no budget yet: wait for another slot to release some
                if release:
                    stop_server(host, port)
                    arm_loaded, cs = None, None
                    log(f"SLOT_YIELD {key} teacher={teacher} (over cap; capacity handed to {other})")
                time.sleep(POLL_S)
                continue
            if arm_loaded != job["arm"] or cs is None:
                cs = start_server(host, teacher, job["arm"], port)
                if cs is None:
                    with _lock:
                        job["status"], job["slot"] = "pending", None
                        save_state(state)
                    log(f"SLOT_EXIT {key} server did not come up; job {job['id']} requeued")
                    return
                arm_loaded = job["arm"]
                log(f"SERVER_UP {key} arm={arm_loaded} cs={cs[:12]}")
            logpath = cell_log(teacher, job["arm"], job["lane"], port, job["task"])
            while not _stop.is_set():
                code = None
                if attach and job.get("cell"):
                    code = cell_exit(host, logpath)  # a re-attached cell may have finished while the queue was down
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
                    code = cell_exit(host, logpath)
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
                sh(HOSTS[host]["worker"], f"mv {logpath} {logpath}.try{job['tries']} 2>/dev/null; true")
                _, up = sh(host, f"ss -tlnH sport = :{port} | grep -q . && echo UP")
                if "UP" not in up:
                    cs = start_server(host, teacher, job["arm"], port)
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
    caps = json.loads(os.environ.get("SDQ_CAPS", json.dumps(DEFAULT_CAPS)))
    budget = Budget(caps)
    for j in state["jobs"]:  # resumed cells hold their server before any slot may claim new work
        if j["status"] == "running" and j.get("slot"):
            host = j["slot"].split(":")[0]
            budget.try_take(host, j["teacher"], True, force=True)
    log(f"QUEUE_START caps={caps} jobs={len(state['jobs'])} done={sum(j['status'] == 'done' for j in state['jobs'])}")
    threads = []
    gpu_rr = {h: 0 for h in HOSTS}
    # start slots in an order that interleaves teachers per host; each slot waits for budget on its own
    for host, h in HOSTS.items():
        order = []
        for i in range(max(len(h["ports"]["pi05"]), len(h["ports"]["groot"]))):
            for teacher in ("pi05", "groot"):
                if i < len(h["ports"][teacher]):
                    order.append((teacher, h["ports"][teacher][i]))
        for teacher, port in order:
            key = f"{host}:{port}"
            resume = next((j["id"] for j in state["jobs"] if j["status"] == "running" and j.get("slot") == key), None)
            gpu = h["worker_gpus"][gpu_rr[host] % len(h["worker_gpus"])]
            gpu_rr[host] += 1
            t = threading.Thread(target=slot_main, args=(state, budget, host, teacher, port, gpu, resume), daemon=True,
                                 name=key)
            threads.append(t)
            t.start()
            time.sleep(20)  # stagger server starts
    for t in threads:
        t.join()
    with _lock:
        left = [j["id"] for j in state["jobs"] if j["status"] != "done"]
    log(f"ALL_DONE done={sum(j['status'] == 'done' for j in state['jobs'])}/{len(state['jobs'])} not_done={left}")


def status() -> None:
    """Print the job counts per teacher and status, and what every busy slot runs."""
    state = load_state()
    by = {}
    for j in state["jobs"]:
        by.setdefault((j["teacher"], j["status"]), 0)
        by[(j["teacher"], j["status"])] += 1
    print(json.dumps({f"{t}:{s}": n for (t, s), n in sorted(by.items())}))
    for k, v in sorted(state["slots"].items()):
        print(k, v)


if __name__ == "__main__":
    {"run": run, "status": status}[sys.argv[1] if len(sys.argv) > 1 else "status"]()
