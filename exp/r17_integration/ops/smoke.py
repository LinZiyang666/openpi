"""Run an explicitly supplied qualification hook with a configurable deadline.

Dry run is the default. Hooks must close owned workers/services in finally.
Only the child created here may receive termination on deadline.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time

from .build_client_bundle import json_bytes
from .plan import qualification_plan
from .sync_back import atomic_bytes


def stop_owned(process, receipt, *, timeout=5):
    if receipt.get("pid") != process.pid or not receipt.get("owned"):
        pass
    if process.poll() is None:
        process.terminate()
        process.wait(timeout=timeout)
    return process.returncode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint-sha", required=True)
    p.add_argument("--hook", help="qualified model/benchmark module:function; required to execute")
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--deadline", type=int, default=900)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--gpu-slot-token", help="coordinator-owned exclusive lock file")
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.deadline < 1:
        p.error("deadline must be positive")
    plan = qualification_plan(a.model, a.checkpoint_sha)
    plan['deadline_seconds'] = a.deadline
    if not a.execute:
        atomic_bytes(a.receipt, json_bytes(plan))
        print(json.dumps(plan))
        return
    if not a.hook or not a.gpu_slot_token:
        p.error("execution requires hook and coordinator GPU slot token")
    if a.child:
        import signal
        def cancel(*_):
            raise TimeoutError("coordinator deadline cancellation")
        signal.signal(signal.SIGTERM, cancel)
        module, function = a.hook.split(":", 1)
        result = getattr(importlib.import_module(module), function)(
            model_id=a.model, checkpoint_digest=a.checkpoint_sha, deadline=time.monotonic()+a.deadline)
        if result.get("status") != "passed":
            pass
        atomic_bytes(a.receipt, json_bytes(dict(plan, executed=True, qualification=result)))
        return
    import fcntl
    token = Path(a.gpu_slot_token)
    # Never create a token: the coordinator supplies an already authorized slot.
    with token.open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        argv = [sys.executable, "-m", "exp.r17_integration.ops.smoke", *sys.argv[1:], "--child"]
        started = time.monotonic()
        process = subprocess.Popen(argv)
        owned = dict(pid=process.pid, owned=True, model=a.model)
        try:
            code = process.wait(timeout=max(0.1, a.deadline-8))
        except subprocess.TimeoutExpired:
            forced = False
            try:
                stop_owned(process, owned)
            except subprocess.TimeoutExpired:
                # Only this Popen child, never PID/name discovery or foreign jobs.
                process.kill()
                process.wait(timeout=1)
                forced = True
            atomic_bytes(a.receipt, json_bytes(dict(plan, executed=True, status="partial", reason="deadline",
                                                   forced_owned_child_stop=forced, owned=owned)))
            raise SystemExit(124)
        if code:
            atomic_bytes(a.receipt, json_bytes(dict(plan, executed=True, status="failed", exit_code=code, owned=owned)))
            raise SystemExit(code)
        print(json.dumps(dict(status="closed", elapsed=time.monotonic()-started)))


if __name__ == "__main__":
    main()
