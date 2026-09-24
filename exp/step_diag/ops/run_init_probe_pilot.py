"""Supervise the authorized two-model pilot using existing tether cell launchers.

Each model runs CloseFridge then PickPlaceCounterToStove, ten episodes each.
Servers are started separately after smoke validation. This controller waits for
readiness, dispatches one worker per model, pulls closed evidence through /tmp,
validates the cohort, and stops only the dedicated pilot's server process.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path("exp/step_diag/data/init_probe_20260922/pilot")
REPO = "/scratch/zixuans8/step_diag/openpi"
REMOTE_ROOT = f"{REPO}/exp/step_diag/data/init_probe_20260922/pilot"
EXPERIMENT = "init_probe_pilot_20260922"
MODELS = {"groot": ("groot_tp", 23158, 1), "pi05": ("pi05", 23147, 2)}
CELLS = (("main", "CloseFridge"), ("pnp", "PickPlaceCounterToStove"))


def remote(args) -> str:
    """Run a bounded tether command, retrying only transient transport failures."""
    for attempt in range(3):
        result = subprocess.run(["tether", "exec", "--timeout", "30s", "timan107", "--", *args],
                                capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        if result.returncode not in (69, 70, 75) or attempt == 2:
            raise RuntimeError(result.stdout + result.stderr)
        time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def log_name(teacher, lane) -> str:
    """Return the existing cell launcher's exact terminal-log path."""
    return f"/tmp/sdiag/sdcell_initprobe_{teacher}_shadow_{lane}_{EXPERIMENT}.log"


def cell_status(teacher, lane) -> str:
    """Read terminal status without modifying or inferring episode outcomes."""
    path = log_name(teacher, lane)
    code = f"from pathlib import Path; p=Path({path!r}); print(p.read_text()[-1200:] if p.exists() else 'ABSENT')"
    tail = remote(["python3", "-c", code])
    if "SDCELL_EXIT=0" in tail and "DONE arm=shadow" in tail:
        return "done"
    if "SDCELL_EXIT=" in tail:
        raise RuntimeError(f"cell failed: {path}\n{tail}")
    return "absent" if tail.strip() == "ABSENT" else "running"


def start_cell(policy, lane, task) -> None:
    """Launch a cell with the server's actual fingerprint, never a typed copy."""
    teacher, port, gpu = MODELS[policy]
    manifest = json.loads((ROOT / policy / "server/manifest_shadow.json").read_text())
    sha = manifest["config_sha"]
    assert len(sha) == 64 and manifest["experiment_id"] == EXPERIMENT
    args = ["env", f"SD_EXP={EXPERIMENT}", "SD_BASE_SEED=2000000", f"SD_GPUS={gpu}",
            f"SD_RC_ENV={REPO}/exp/step_diag/config/rc_timan107.env", f"SD_OUT_ROOT={REMOTE_ROOT}",
            "bash", f"{REPO}/exp/step_diag/ops/run_rc_cell.sh", teacher, "shadow", lane,
            f"ziyanglin.com:{port}", task, "10", "-", sha, "initprobe"]
    print(remote(["bash", "-lc", shlex.join(args)]).strip(), flush=True)


def pull_and_validate(policy) -> None:
    """Copy closed artifacts, verify transfer checksums, then reconcile the run."""
    teacher, _, _ = MODELS[policy]
    source = f"{REMOTE_ROOT}/{teacher}/shadow"
    stage = f"/tmp/sdiag/initprobe_transfer/{teacher}"
    code = (
        "from pathlib import Path; import hashlib,json,shutil; "
        f"p=Path({source!r}); out=Path({stage!r}); out.mkdir(parents=True,exist_ok=True); rows=[]\n"
        "for pattern in ('journal_*.jsonl','launch_*.json','summary_*.json','run_plan_*.json','per_step_*.jsonl'):\n"
        " for src in p.glob(pattern):\n"
        "  dst=out/src.name; shutil.copyfile(src,dst); rows.append({'path':str(dst),'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()})\n"
        "print(json.dumps(rows))"
    )
    files = json.loads(remote(["python3", "-c", code]))
    target = ROOT / policy / "client"
    target.mkdir(exist_ok=True)
    for item in files:
        local = target / Path(item["path"]).name
        if not local.exists():
            subprocess.run(["tether", "pull", "timan107:" + item["path"], str(local)], check=True)
        if hashlib.sha256(local.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"transfer checksum mismatch: {local}")
    summary = ROOT / policy / "summary.json"
    env = dict(os.environ, MPLCONFIGDIR="/tmp/init_probe_mpl", PYTHONPATH="src:.")
    subprocess.run([sys.executable, "-B", "-m", "exp.step_diag.analysis.analyze_init_probe",
                    "--probe-dir", str(ROOT / policy / "probe"), "--client-dir", str(target),
                    "--out-json", str(summary), "--out-figure",
                    f"exp/step_diag/analysis/init_probe_20260922/{policy}_sensitivity.png"], env=env, check=True)
    result = json.loads(summary.read_text())
    assert result["accepted_episodes"] == 20 and result["requested_cohort_checked"]
    assert all(v["episodes"] == 10 for v in result["tasks"].values())
    print(f"VALIDATED {policy}: {result['accepted_episodes']} episodes, {result['observations']} observations", flush=True)


def stop_own_server(policy) -> None:
    """Stop only a process whose command line identifies this pilot output path."""
    expected = str(ROOT / policy / "probe")
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = path.read_bytes().decode().split("\0")
            if "exp.step_diag.serve_init_probe" in argv and "--probe-out" in argv:
                if argv[argv.index("--probe-out") + 1] == expected:
                    os.kill(int(path.parent.name), signal.SIGTERM)
                    print(f"STOPPED own {policy} pilot server pid={path.parent.name}", flush=True)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue


def main() -> None:
    """Poll readiness and terminal conditions until both pilot cohorts validate."""
    complete = set()
    while len(complete) < len(MODELS):
        states = {}
        for policy, (teacher, port, _) in MODELS.items():
            if policy in complete:
                states[policy] = "validated"
                continue
            ready = subprocess.check_output(["ss", "-ltnH", f"sport = :{port}"], text=True).strip()
            if not ready:
                states[policy] = "server_loading"
                continue
            for lane, task in CELLS:
                state = cell_status(teacher, lane)
                if state != "done":
                    if state == "absent":
                        start_cell(policy, lane, task)
                    states[policy] = f"{lane}_running"
                    break
            else:
                pull_and_validate(policy)
                stop_own_server(policy)
                complete.add(policy)
                states[policy] = "validated"
        (ROOT / "controller_status.json").write_text(json.dumps(states, indent=2) + "\n")
        print("PROBE " + json.dumps(states), flush=True)
        if len(complete) < len(MODELS):
            time.sleep(30)
    print("INIT_PROBE_PILOT_DONE", flush=True)


if __name__ == "__main__":
    main()
