"""Repository-Python wrapper for an available local Python 3.8 compiler, no installs."""
import argparse
import json
from pathlib import Path
import subprocess


CHECK = r'''
import importlib, importlib.util, json, pathlib, sys
inventory, bundle, repository = map(pathlib.Path, sys.argv[1:])
assert sys.version_info[:2] == (3, 8), sys.version
report = json.loads(inventory.read_text())
compiled = []
for row in report["files"]:
    path = repository / row["path"]
    compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    compiled.append(row["path"])
sys.path.insert(0, str(bundle / "payload"))
prefix = "exp.offline_search.rounds.r06.p3_profiling."
names = ("run_gtp_v2", "worker_v2", "client_compat", "client_preflight")
for name in names:
    mod = importlib.import_module(prefix + name)
    assert str(bundle / "payload") in mod.__file__
compat = importlib.import_module(prefix + "client_compat")
compat.install_import_compat()
compat.install_import_compat()
assert sum(bool(getattr(x, "_p3_python38", False)) for x in sys.meta_path) == 1
path = repository / "src/openpi/cache/types.py"
loader = compat.DeferredSourceLoader("p3_types_runtime_fixture", str(path))
module = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
sys.modules[loader.name] = module
loader.exec_module(module)
assert isinstance(module.__annotations__["CACHE_QUERY_FIELDS"], str)
assert len(module.CACHE_QUERY_FIELDS) > 0
assert list(compat.zip_compat([1,2], [3,4], strict=True)) == [(1,3),(2,4)]
assert list(compat.zip_compat([], [], strict=True)) == []
assert list(compat.zip_compat([1], [2,3])) == [(1,2)]
for a, b in (([1], [2,3]), ([1,2], [3])):
    try:
        list(compat.zip_compat(a, b, strict=True))
    except ValueError:
        pass
    else:
        raise AssertionError("strict length mismatch was accepted")
print(json.dumps(dict(PASS=True, python=sys.version, executable=sys.executable,
    compiled_files=len(compiled), owned_compiled=sum(x["owned_client"] for x in report["files"]),
    isolated_import_modules=list(names), real_stock_types_loader=True,
    idempotent_import_hook=True, strict_zip_cases=5,
    numpy_and_simulator_runtime="not tested: isolated cached interpreter has no third-party environment",
    compiled_paths=compiled), indent=2))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python38", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    here = Path(__file__).resolve().parent
    command = ["taskset", "-c", "2-5,46-49", "env", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
               "MKL_NUM_THREADS=1", "CUDA_VISIBLE_DEVICES=", "PYTHONDONTWRITEBYTECODE=1",
               str(a.python38), "-I", "-S", "-c", CHECK,
               str(here / "results/client_compat.json"), str(here / "client_bundle"), str(here.parents[4])]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    report = json.loads(result.stdout)
    report["command"] = command
    a.out.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("compiled_paths", "command")}))


if __name__ == "__main__":
    main()
