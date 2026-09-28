"""Coordinator-only import/API check; no driver, worker, socket or env is started."""
import argparse
import importlib
import inspect
import json
from pathlib import Path
import sys


MODULES = ("run_gtp_v2", "worker_v2", "telemetry", "snapshots", "client_compat", "client_preflight", "stream_protocol", "stream_sink")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock-imports", action="store_true")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[5]
    info = dict(python=sys.version, executable=sys.executable, island_root=str(root), files={})
    for name in MODULES:
        path = Path(__file__).with_name(name+".py")
        compile(path.read_text(), str(path), "exec")
        importlib.import_module("exp.offline_search.rounds.r06.p3_profiling."+name)
        info["files"][name] = str(path)
    import numpy as np
    info["numpy"] = np.__version__
    if a.stock_imports:
        from .client_compat import install_import_compat
        install_import_compat()
        from examples.libero import episode_runner, worker_entry, main as libero_main
        from exp.gate_threshold_pareto import run_gtp
        from .client_compat import install_driver_compat
        install_driver_compat()
        assert {"client_factory", "run_episode_fn"} <= set(inspect.signature(episode_runner.LiberoEpisodeRunner).parameters)
        assert {"worker_module", "env"} <= set(inspect.signature(run_gtp.WorkerSpec).parameters)
        assert callable(worker_entry.main) and callable(libero_main._run_episode)
        assert callable(episode_runner.default_client_factory) and hasattr(run_gtp.SweepStrategy, "_episodes")
        script = root / "os_cl/run_gtp_subset.py"
        compile(script.read_text(), str(script), "exec")
        info["stock_modules"] = {x.__name__: x.__file__ for x in (episode_runner, worker_entry, libero_main, run_gtp)}
        info["subset_script"] = str(script)
        info["simulator_attributes"] = "not verified without an actual episode; smoke required"
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
