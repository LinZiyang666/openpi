"""Record already-run Fix round 1 CPU evidence; never run model/GPU commands."""
from dataclasses import asdict
from datetime import datetime, timezone
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

from exp.r17_integration.common.contracts import VerificationReceipt


ROOT = Path("exp/r17_integration")
TESTS = ROOT / "tests/models/groot"
MODELS = ROOT / "models/groot"
ENV = ("env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 "
       "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
       "PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python")
LEGACY = ("tests/cache/groot/test_groot_staged.py tests/cache/groot/test_groot_stage3.py "
          "tests/cache/groot/test_groot_key_builder.py tests/cache/groot/test_libero_key_builder.py "
          "tests/cache/groot/test_groot_interceptor.py tests/cache/groot/test_groot_batcher.py "
          "tests/cache/groot/test_groot_load_guard.py tests/cache/groot/test_trace_groot.py "
          "tests/libero_groot/test_dynamic_bundle_guards.py tests/libero_groot/test_policy_adapter.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(path):
    suite = ET.parse(path).getroot().find("testsuite")
    return {name: int(suite.attrib[name]) for name in ("tests", "failures", "errors", "skipped")}


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "run with CUDA hidden"
    legacy = json.loads((MODELS / "legacy_sources.json").read_text())
    assert len(legacy) == 18 and all(sha(path) == expected for path, expected in legacy.items())
    owned_python = sorted([*MODELS.glob("*.py"), *TESTS.glob("*.py")])
    for path in owned_python:
        ast.parse(path.read_text(), filename=str(path))
    sample = json.loads((MODELS / "samples/cpu_stub_capture.json").read_text())
    assert sample["diagnostic_fixture"] and not sample["real_checkpoint"] and not sample["gpu_qualified"]
    assert sample["actual_nfe"] == 4
    with np.load(MODELS / "samples/cpu_stub_capture.npz", allow_pickle=False) as arrays:
        assert set(arrays.files) == set(sample["array_shapes"]) == set(sample["array_dtypes"])
        for name in arrays.files:
            assert list(arrays[name].shape) == sample["array_shapes"][name]
            assert str(arrays[name].dtype) == sample["array_dtypes"][name]
            assert np.isfinite(arrays[name]).all()
    handback = ROOT / "handbacks/HANDBACK_P5.md"
    blocks = re.findall(r"```bash\n(.*?)```", handback.read_text(), re.S)
    for block in blocks:
        result = subprocess.run(["bash", "-n"], input=block, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
    before = metrics(TESTS / "fix1_before.xml")
    assert before == dict(tests=6, failures=6, errors=0, skipped=0)
    paths = set(owned_python) | {handback, MODELS / "legacy_sources.json", MODELS / ".gitignore", TESTS / ".gitignore"}
    paths.update((MODELS / "samples").glob("*"))
    paths.update(ROOT / "REQUESTS" / f"P5_to_{owner}.md" for owner in ("P1", "P2", "P6"))
    for directory in ("common", "benchmarks", "conductor", "debug", "search"):
        paths.update((ROOT / directory).rglob("*.py"))
    paths.update(Path(path) for path in legacy)
    paths.update(Path(path) for path in ("src/openpi/cache/config.py", "src/openpi/cache/types.py",
                 "src/openpi/cache/storage_types.py", "src/openpi/serving/websocket_policy_server.py"))
    paths.update(TESTS / name for name in ("fix1_before.xml", "fix1_final.xml", "fix1_legacy.xml"))
    source = {str(path): sha(path) for path in sorted(paths) if path.is_file()}
    limits = ("CPU stub model and native-shaped fake environment; no real weights/renderer/GPU",
              "Live conductor pull/journal transport is replaced only in worker entry fixture",
              "Direct advanced hooks remain implemented_unverified; GPU quality/cost/BF16 pending",
              "Dependency hashes captured after verification in a concurrently edited checkout")
    checks = []
    for xml, target, scratch, count in (
        ("fix1_final.xml", str(TESTS), "scratch_fix1_final_guard", 61),
        ("fix1_legacy.xml", LEGACY, "scratch_fix1_legacy", 153),
    ):
        values = metrics(TESTS / xml)
        assert values == dict(tests=count, failures=0, errors=0, skipped=0), values
        command = f"{ENV} -m pytest {target} -p no:cacheprovider --basetemp={TESTS}/{scratch} --junitxml={TESTS}/{xml} --tb=short -q"
        check = asdict(VerificationReceipt(command, 0, count, 0, "cpu", source, limits))
        check.update(xml=str(TESTS / xml), metrics=values, unique_counted=True)
        checks.append(check)
    original = json.loads((TESTS / "verification_receipt.json").read_text())["checks"][0]["source_digests"]
    receipt = dict(schema="r17.p5_verification_bundle.v1", fix_round=1,
        created_utc=datetime.now(timezone.utc).isoformat(), interface_version="r17.contracts.v1",
        device="cpu", gpu_executed=False, unique_passed=214, skipped=0, checks=checks,
        failing_before=dict(xml=str(TESTS / "fix1_before.xml"), metrics=before,
                            note="Six initial regression cases before P5 fixes; current tests add deployment cases"),
        legacy_byte_certificate=legacy, parsed_python_modules=len(owned_python),
        diagnostic_sample_checks="array shape/dtype/finite checks passed; regenerated actual NFE4",
        bash_syntax_blocks=len(blocks),
        auxiliary_commands=[dict(command=f"{ENV} -m exp.r17_integration.models.groot.{name}", exit_code=0)
                            for name in ("build_cpu_sample", "serve --help", "qualify --help")],
        receipt_command=f"{ENV} -m exp.r17_integration.tests.models.groot.record_fix1_receipt",
        owned_modified_since_initial=[str(p) for p in owned_python if str(p) in original and sha(p) != original[str(p)]],
        owned_added_since_initial=[str(p) for p in owned_python if str(p) not in original],
        limits=limits)
    (TESTS / "verification_fix1.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": 214, "skipped": 0, "python_modules": len(owned_python),
                      "legacy_bytes": len(legacy), "bash_blocks": len(blocks), "gpu_executed": False}))


if __name__ == "__main__":
    main()
