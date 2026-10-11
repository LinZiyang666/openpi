"""Bound only an E-owned server; no process search, shared lock or other-session signals."""
import argparse
import signal
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=int, default=780)
    ap.add_argument("--module", default="exp.mw_finetune.dp.serve_compat")
    a, extra = ap.parse_known_args()
    if not 1 <= a.seconds <= 850:
        ap.error("Smoke server must be bounded to850s")
    child = subprocess.Popen([sys.executable, "-m", a.module, *extra])

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        try:
            code = child.wait(timeout=a.seconds)
            if code:
                raise subprocess.CalledProcessError(code, child.args)
        except subprocess.TimeoutExpired:
            print("DP smoke server reached its authorized deadline", flush=True)
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()


if __name__ == "__main__":
    main()
