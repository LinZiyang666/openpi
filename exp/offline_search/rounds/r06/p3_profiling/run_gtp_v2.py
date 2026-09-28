"""Same stock driver and subset selectors, with V2 client dependency injection."""
import os
from pathlib import Path
import runpy
import sys


def main():
    from .client_compat import install_driver_compat, install_import_compat
    install_import_compat()
    from exp.gate_threshold_pareto import run_gtp
    install_driver_compat()
    original = run_gtp.WorkerSpec
    def spec(*args, **kw):
        kw["worker_module"] = "exp.offline_search.rounds.r06.p3_profiling.worker_v2"
        kw["env"] = {**kw.get("env", {}), **{k: os.environ.get(k, default) for k, default in (
            ("P3_SNAPSHOT_DIR", ""), ("P3_SNAPSHOT_EVERY", "1"), ("P3_SNAPSHOT_P", "1"), ("P3_ENV_SEED", "7"))}}
        if not kw["env"]["P3_SNAPSHOT_DIR"]:
            raise ValueError("P3_SNAPSHOT_DIR required")
        if os.environ.get("P3_STREAM"):
            kw["env"].update({k: v for k, v in os.environ.items() if k == "P3_STREAM" or k.startswith("P3_STREAM_")})
        return original(*args, **kw)
    run_gtp.WorkerSpec = spec
    if (os.environ.get("OSCL_EPISODES") or os.environ.get("OSCL_MANIFEST")
            or any(x == "--manifest" or x.startswith("--manifest=") for x in sys.argv[1:])):
        # The client island only has the stock standalone script in os_cl/.
        # run_gtp is already imported, so the script sees our WorkerSpec patch.
        subset = Path(os.environ.get("P3_SUBSET_SCRIPT", str(
            Path(__file__).resolve().parents[5] / "os_cl/run_gtp_subset.py")))
        if not subset.is_file():
            raise FileNotFoundError("stock subset script is missing: " + str(subset))
        runpy.run_path(str(subset), run_name="__main__")
    else:
        run_gtp.main()


if __name__ == "__main__":
    main()
