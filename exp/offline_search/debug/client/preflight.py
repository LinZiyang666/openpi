"""Import/syntax inventory only; creates no environments, workers or sockets."""
import argparse
import ast
import importlib
import json
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock-imports", action="store_true")
    a = ap.parse_args()
    from .compat import install_driver_compat, install_import_compat
    install_import_compat()
    package = Path(__file__).resolve().parents[1]
    paths = [package / "schema.py"] + [p for sub in ("client", "transport", "backfill") for p in (package / sub).rglob("*.py")]
    for path in paths:
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        if sys.version_info >= (3, 8):
            ast.parse(path.read_text(), feature_version=8)
    names = ["client.worker", "client.driver", "client.capture", "client.adapter", "transport.sink", "transport.dispatch_fence"]
    for name in names:
        importlib.import_module("exp.offline_search.debug." + name)
    if a.stock_imports:
        for name in ("examples.libero.worker_entry", "examples.libero.episode_runner", "examples.libero.main", "exp.gate_threshold_pareto.run_gtp"):
            importlib.import_module(name)
        install_driver_compat()
    print(json.dumps(dict(PASS=True, compiled=len(paths), imported=names, stock_imports=a.stock_imports, python=sys.version)))


if __name__ == "__main__":
    main()
