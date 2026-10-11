"""Record existing CPU XMLs and post-test sources; never execute qualification."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import xml.etree.ElementTree as ET


def main():
    base = Path("exp/r17_integration")
    owned_dirs = ["benchmarks", "conductor", "models/pi05", "tests/benchmarks", "tests/conductor", "tests/models/pi05"]
    python_files = [p for directory in owned_dirs for p in sorted((base / directory).glob("*.py"))]
    for path in python_files:
        compile(path.read_text(), str(path), "exec")
    owned = python_files + [base / "handbacks/HANDBACK_P2.md"] + [base / "REQUESTS" / name for name in
        ("P2_to_P1.md", "P2_to_P3.md", "P2_to_P4.md", "P2_to_P5.md", "P2_to_P6.md")]
    originals = [Path(p) for p in (
        "exp/metaworld/tasks.py", "exp/metaworld/env.py", "exp/metaworld/episode_runner.py",
        "examples/libero/main.py", "examples/libero/episode_runner.py", "examples/libero/worker_entry.py", "examples/libero/collect_util.py",
        "src/openpi/cache/config.py", "src/openpi/cache/types.py", "src/openpi/cache/storage_types.py", "src/openpi/serving/websocket_policy_server.py",
        "exp/mw_finetune/rendering.py", "src/openpi/models_pytorch/pi0_pytorch.py",
        "src/openpi/policies/metaworld_policy.py", "src/openpi/conductor/worker.py",
        "src/openpi/conductor/agent.py", "src/openpi/conductor/driver.py", "src/openpi/conductor/scheduler.py",
        "exp/offline_search/debug/schema.py", "exp/offline_search/debug/server/writer.py")]
    unchanged, untracked = {}, []
    for path in originals:
        original = subprocess.run(["git", "show", "HEAD:" + str(path)], capture_output=True)
        if original.returncode:
            # A's mw_finetune baseline is untracked in this shared checkout.
            untracked.append(str(path))
        else:
            unchanged[str(path)] = original.stdout == path.read_bytes()
    assert all(unchanged.values()), unchanged
    shared = [p for directory in ("common", "debug", "search", "ops") for p in sorted((base / directory).rglob("*.py"))]
    shared.extend(Path(p) for p in (
        "src/openpi/cache/config.py", "src/openpi/cache/types.py", "src/openpi/cache/storage_types.py",
        "src/openpi/serving/websocket_policy_server.py", "src/openpi/policies/policy.py",
        "src/openpi/policies/policy_config.py", "src/openpi/transforms.py", "src/openpi/models/tokenizer.py",
        "src/openpi/training/config.py"))
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    source = {"schema": "r17.p2.source_snapshot.v1", "captured_utc": datetime.now(timezone.utc).isoformat(),
        "timing": "post-test; shared agents can still change their files", "branch": "Ziyang",
        "head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
        "python": platform.python_version(), "owned": {str(p): sha(p) for p in owned},
        "original_reused": {str(p): sha(p) for p in originals}, "original_equal_head": unchanged,
        "original_untracked_sha_only": untracked, "cross_package_dependencies": {str(p): sha(p) for p in shared},
        "compile": {"passed": len(python_files), "skipped": 0, "pyc_written": False}}
    artifacts = base / "tests/conductor"
    (artifacts / "SOURCES_P2.json").write_text(json.dumps(source, sort_keys=True, indent=2) + "\n")
    prefix = "env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:packages/openpi-client/src .venv/bin/python -m pytest "
    runs = [
        ("cpu_fix1_release4.xml", "exp/r17_integration/tests/benchmarks exp/r17_integration/tests/conductor exp/r17_integration/tests/models/pi05", "scratch_fix1_release4", "", 0),
        ("cpu_fix1_cross_release4.xml", "exp/r17_integration/tests/common exp/r17_integration/tests/benchmarks exp/r17_integration/tests/conductor exp/r17_integration/tests/models/pi05 exp/r17_integration/tests/debug exp/r17_integration/tests/search exp/r17_integration/tests/ops", "scratch_fix1_cross_release4", "", 0),
        ("cpu_fix1_legacy_release4.xml", "tests/serving/test_policy_identity.py tests/serving/test_per_step_recorder.py tests/serving/test_replica_proxy.py tests/serving/test_router_bundle_rebind.py tests/exp/metaworld/test_env.py tests/exp/metaworld/test_runner.py tests/conductor exp/mw_finetune/tests exp/offline_search/rounds/r16/test_conductor.py exp/offline_search/rounds/r16/test_groot_servers.py exp/r17_integration/tests/common/test_default_off.py tests/cache/test_config.py tests/cache/test_denoise_schedule_config.py tests/cache/test_cache_storage.py", "scratch_fix1_legacy_release4", "-k 'not same_identity_same_initial_state_across_processes'", 1),
        ("cpu_fix1_proxy.xml", "tests/serving/test_replica_proxy.py", "scratch_fix1_proxy", "-m env_dependent --run-manual", 23),
        ("cpu_fix1_groot_worker_release.xml", "exp/r17_integration/tests/models/groot/test_fix_round1.py", "scratch_fix1_groot_worker_release", "-k 'real_serve_cli'", 12),
    ]
    receipts = []
    for filename, targets, scratch, options, deselected in runs:
        xml = artifacts / filename
        root = ET.parse(xml).getroot()
        cases = list(root.iter("testcase"))
        failed = sum(c.find("failure") is not None or c.find("error") is not None for c in cases)
        skipped = sum(c.find("skipped") is not None for c in cases)
        assert failed == 0
        p2_cases = [c for c in cases if c.attrib.get("classname", "").startswith((
            "exp.r17_integration.tests.benchmarks.", "exp.r17_integration.tests.conductor.",
            "exp.r17_integration.tests.models.pi05."))]
        receipts.append({"schema": "r17.verification_receipt.v1",
            "command": prefix + targets + " " + options + " -p no:cacheprovider --basetemp=" + str(artifacts / scratch)
                       + " --junitxml=" + str(xml) + " -q --tb=short",
            "exit_code": 0, "passed": len(cases)-skipped, "skipped": skipped, "failed": failed,
            "duration_seconds": sum(float(s.attrib.get("time", 0)) for s in root.iter("testsuite")),
            "deselected": deselected, "p2_cases": len(p2_cases), "device": "cpu", "junit_path": str(xml),
            "junit_sha256": sha(xml), "source_digests": source["owned"],
            "source_snapshot_timing": "post-test; later handback/sample/receipt helpers included; see SOURCES_P2.json",
            "baseline_paths": [str(p) for p in originals],
            "tolerance": "exact wire bytes; independent inverse assertion atol1e-6 in named test",
            "limits": ["no real checkpoint/encoder/physics/renderer/GPU qualification", "overlaps other runs; do not sum passes",
                       "post-test snapshot; other packages may continue editing"]})
    samples = base / "tests/models/pi05/samples/native_k2"
    manifest = json.loads((samples / "sample_manifest.json").read_text())
    assert manifest["synthetic_fixture"] and not manifest["checkpoint_loaded"] and not manifest["renderer_used"]
    assert len(json.loads((samples / "formal_500.json").read_text())) == 500
    receipt = {"schema": "r17.p2.handback_receipts.v1", "interface_version": "r17.contracts.v1", "runs": receipts,
        "source_snapshot": str(artifacts / "SOURCES_P2.json"), "compile": source["compile"],
        "sample_command": prefix.removesuffix("pytest ") + "exp.r17_integration.tests.models.pi05.build_samples",
        "sample_exit_code": 0, "sample_readback": "passed",
        "sample_label": "synthetic CPU fixture; synthetic three-control ack; no actual checkpoint/physics/quality",
        "sample_hashes": {str(p): sha(p) for p in samples.rglob("*") if p.is_file() and not any("diagnostic" in part for part in p.parts)},
        "fix_round1_sample_command": prefix.removesuffix("pytest ") + "exp.r17_integration.tests.models.pi05.build_fix1_samples",
        "fix_round1_samples": {str(p): sha(p) for p in (base / "tests/models/pi05/samples/fix_round1").rglob("*") if p.is_file()},
        "old_regression_red": {"xml": str(artifacts / "cpu_fix1_old_red.xml"), "expected_failures": 2,
                               "preserved_old_runner_sha256": sha(artifacts / "scratch_p2_binding_release/test_client_payload_copied_fil0/exp/r17_integration/benchmarks/metaworld.py")},
        "context_default_red": {"xml": str(artifacts / "cpu_fix1_context_old_red.xml"), "expected_failures": 2,
            "command": prefix + "exp/r17_integration/tests/benchmarks/test_metaworld.py "
                "-k framework_request_ack_and_original_task_identity -p no:cacheprovider "
                "--basetemp=exp/r17_integration/tests/conductor/scratch_fix1_context_old_red "
                "--junitxml=exp/r17_integration/tests/conductor/cpu_fix1_context_old_red.xml -q --tb=short",
            "deselected": 14, "exit_code": 1, "junit_sha256": sha(artifacts / "cpu_fix1_context_old_red.xml"),
            "reason": "both R17 constructors omit __request__ before enabling framework_context by default"},
        "gpu_commands_executed": False, "gpu_use_today": False, "git_commit_or_push": False}
    (artifacts / "RECEIPT_P2.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"owned_python_compile": len(python_files), "original_source_byte_identical": len(unchanged),
        "untracked_baseline_sha_only": untracked,
        "runs": [{"xml": r["junit_path"], "passed": r["passed"], "skipped": r["skipped"], "p2_cases": r["p2_cases"]} for r in receipts],
        "receipt": str(artifacts / "RECEIPT_P2.json")}))


if __name__ == "__main__":
    main()
