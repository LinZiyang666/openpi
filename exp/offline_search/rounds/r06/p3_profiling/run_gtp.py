"""Small driver wrapper selecting the snapshot worker without src/ edits."""
import os
import sys


def main():
    from exp.gate_threshold_pareto import run_gtp
    original = run_gtp.WorkerSpec
    directory = os.environ["P3_SNAPSHOT_DIR"]

    def spec(*args, **kw):
        kw["worker_module"] = "exp.offline_search.rounds.r06.p3_profiling.worker"
        kw["env"] = {**kw.get("env", {}), "P3_SNAPSHOT_DIR": directory,
                     "P3_SNAPSHOT_EVERY": os.environ.get("P3_SNAPSHOT_EVERY", "1")}
        return original(*args, **kw)

    run_gtp.WorkerSpec = spec
    if os.environ.get("OSCL_EPISODES") or os.environ.get("OSCL_MANIFEST") or "--manifest" in sys.argv:
        from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import main as subset
        subset()
    else:
        run_gtp.main()


if __name__ == "__main__":
    main()
