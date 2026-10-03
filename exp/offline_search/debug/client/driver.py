"""Stock driver/subset selection, with a fence in BOTH file and stream modes.

Adapted from P3 run_gtp_v2.py; no telemetry-specific controller or P3 marker.
"""
import os
from pathlib import Path
import runpy
import sys


def main():
    from .compat import install_driver_compat, install_import_compat
    install_import_compat()
    from exp.gate_threshold_pareto import run_gtp
    install_driver_compat()
    constructor = run_gtp.ConductorDriver

    def driver(*args, **kw):
        from exp.offline_search.debug.transport.dispatch_fence import install
        install(Path(os.environ["OSDEBUG_CLIENT_DIR"]).parent / ".osdebug_dispatch", kw.get("journal_path"))
        return constructor(*args, **kw)

    run_gtp.ConductorDriver = driver
    original = run_gtp.WorkerSpec

    def spec(*args, **kw):
        kw["worker_module"] = "exp.offline_search.debug.client.worker"
        kw["env"] = dict(kw.get("env", {}), **{k: v for k, v in os.environ.items() if k.startswith("OSDEBUG_")})
        if not kw["env"].get("OSDEBUG_CLIENT_DIR"):
            raise ValueError("OSDEBUG_CLIENT_DIR required")
        return original(*args, **kw)

    run_gtp.WorkerSpec = spec
    if (os.environ.get("OSCL_EPISODES") or os.environ.get("OSCL_MANIFEST")
            or any(x == "--manifest" or x.startswith("--manifest=") for x in sys.argv[1:])):
        subset = Path(os.environ.get("OSDEBUG_SUBSET_SCRIPT", "/scratch/zixuans8/openpi_trace/os_cl/run_gtp_subset.py"))
        if not subset.is_file():
            raise FileNotFoundError("stock subset script missing: " + str(subset))
        runpy.run_path(str(subset), run_name="__main__")
    else:
        run_gtp.main()


if __name__ == "__main__":
    main()
