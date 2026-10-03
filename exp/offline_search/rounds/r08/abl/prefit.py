"""Fresh CPU prefits through R6's plugin entry point, with bounded concurrency."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

from exp.offline_search.rounds.r08.abl.make_arms import PREFIX, RUN, STORE, artifact, write_json


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--parallel", type=int, default=7)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.parallel <= 7:
        parser.error("--parallel must be between 1 and 7")
    rows = json.loads((args.run / "arms_in.json").read_text())
    logs = args.run / "prefit_logs"
    logs.mkdir(parents=True, exist_ok=True)
    jobs = []
    for row in rows:
        if Path(artifact(row)).exists():
            if args.resume:
                continue
            raise FileExistsError(artifact(row))
        cmd = PREFIX + ["-m", "exp.offline_search.closed_loop.plugin", "--os-method", row["method"],
                       "--os-kwargs", json.dumps(row["kwargs"]), "--os-cell", f"{row['model']}_{row['suite']}_cache",
                       "--os-log-dir", str(logs / row["name"]), "--os-tag", row["name"]]
        if "--os-root" not in row["plugin_args"]:
            cmd += ["--os-root", STORE]
        jobs.append((row["name"], cmd + row["plugin_args"]))
    write_json(args.run / "validation/prefit_commands.json",
               [dict(arm=name, argv=cmd, command=shlex.join(cmd)) for name, cmd in jobs])
    running, failures, completed = [], [], 0
    try:
        while jobs or running:
            while jobs and len(running) < args.parallel:
                name, cmd = jobs.pop(0)
                log = (logs / (name + ".log")).open("w")
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                running.append((name, proc, log))
            for item in list(running):
                name, proc, log = item
                if proc.poll() is not None:
                    log.close()
                    print(name, proc.returncode, flush=True)
                    completed += 1
                    if proc.returncode:
                        failures.append(name)
                    running.remove(item)
            if running:
                time.sleep(1)
    finally:
        # Only children created here are eligible for termination.
        for _, proc, log in running:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
            log.close()
    if failures:
        raise RuntimeError(f"prefit failures: {failures}")
    print(f"PASS: {completed} fresh prefits completed", flush=True)


if __name__ == "__main__":
    main()
