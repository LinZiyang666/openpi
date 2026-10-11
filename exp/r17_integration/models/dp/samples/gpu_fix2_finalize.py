"""Audit GPU fix2's current CPU evidence and separate historical GPU diagnostic."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[5]
DP = ROOT / "exp/r17_integration/models/dp"
SAMPLES = DP / "samples"
TESTS = ROOT / "exp/r17_integration/tests/models/dp"


def sha(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    runtime = read(SAMPLES / "gpu_fix2_runtime_under_test.json")
    for group in ("model_sources", "framework_sources"):
        for path, expected in runtime[group].items():
            assert sha(ROOT / path) == expected, f"source drift: {path}"
    suites = {}
    for name, tests, failures in (("full", 153, 0), ("legacy", 49, 0),
                                  ("targeted", 23, 0), ("negative", 1, 1)):
        suite = ET.parse(DP / f"cpu_gpu_fix2_{name}.xml").getroot().find("testsuite")
        counts = {key: int(suite.attrib[key]) for key in ("tests", "failures", "skipped", "errors")}
        assert counts == dict(tests=tests, failures=failures, skipped=0, errors=0)
        suites[name] = dict(counts, passed=tests-failures, seconds=float(suite.attrib["time"]),
                            expected_exit_code=int(bool(failures)))
    probes = {}
    for name in ("q_libero", "release_libero", "release_metaworld"):
        receipt = read(SAMPLES / f"gpu_fix2_{name}_cpu/receipt.json")
        assert receipt["status"] == "pass" and receipt["exit_code"] == 0
        assert receipt["passed"] == len(receipt["checks"]) == 13 and receipt["skipped"] == 0
        assert receipt["device"] == "cpu" and receipt["diagnostic"] is True
        manifest = receipt["manifest"]
        assert manifest["strict_load"] == dict(missing=[], unexpected=[], keys=475)
        assert {path: manifest["content_digests"][path] for path in runtime["model_sources"]} == runtime["model_sources"]
        assert manifest["framework_digests"] == runtime["framework_sources"]
        for capability in ("batch_stage1", "batch_stage2", "batch_stage3"):
            assert manifest["spec"]["capabilities"]["features"][capability]["status"] == "unsupported"
        checks = {row["name"] for row in receipt["checks"]}
        assert "real_production_B1_only" in checks and "real_B1_B2_sampler_equivalence" not in checks
        diagnostic = receipt["diagnostics"][0]
        assert diagnostic["status"] == "diagnostic_only" and diagnostic["production_batching"] == "unsupported"
        assert diagnostic["rng_states_exact"] is True and diagnostic["label"]
        probes[name] = dict(passed=13, skipped=0, device="cpu", diagnostic=True,
                            elapsed_seconds=receipt["elapsed_seconds"], B2_diagnostic=diagnostic,
                            model_identity=manifest["spec"]["checkpoint_identity"])
    q = read(SAMPLES / "gpu_fix2_q_libero_cpu/receipt.json")
    seal = read(SAMPLES / "gpu_fix2_q_libero_seal/receipt.json")
    source = Path(seal["source_load"])
    assert sha(source) == seal["source_sha256"]
    sealed = read(SAMPLES / "gpu_fix2_q_libero_seal/load.json")
    assert sealed == dict(read(source), expected_model_identity=seal["new_model_identity"])
    assert q["manifest"]["spec"] == read(SAMPLES / "gpu_fix2_q_libero_seal/model_spec.json")
    assert read(SAMPLES / "gpu_fix2_q_libero_cpu/load_cpu.json") == dict(sealed, device="cpu")
    provenance = q["observation_provenance"]
    assert sha(Path(provenance["path"])) == provenance["npz_sha256"] == runtime["Q_npz_sha256"]
    assert provenance["image_shapes"] == {key: [256, 256, 3] for key in ("observation/image", "observation/wrist_image")}
    assert q["load_spec_provenance"]["overridden_fields"] == ["device"]
    bundles = read(SAMPLES / "gpu_fix2_release_loader_bundle_receipt.json")
    assert bundles["passed"] == len(bundles["checks"]) == 18 and bundles["skipped"] == bundles["exit_code"] == 0
    gpu = read(SAMPLES / "gpu_fix2_measurement/receipt.json")
    assert gpu["status"] == "pass" and gpu["exit_code"] == 0 and gpu["gpu_executed"] is True
    assert gpu["diagnostic_only"] is True and gpu["elapsed_seconds"] <= 300
    assert gpu["manifest"]["spec"]["checkpoint_identity"] == seal["old_model_identity"]
    assert gpu["load_sha256"] == seal["source_sha256"] and gpu["observations_sha256"] == runtime["Q_npz_sha256"]
    assert sha(Path(gpu["job"])) == gpu["job_sha256"]
    assert len(gpu["measurements"]) == 8 and all(row["rng_exact"] for row in gpu["measurements"])
    assert {(row["K"], row["B"], row["scope"]) for row in gpu["measurements"]} == {
        (k, b, scope) for k in (100, 2) for b in (2, 4) for scope in ("sampler_only", "end_to_end")}
    gpu_source_delta = {path: dict(measured=gpu["manifest"]["content_digests"][path], current=now)
                        for path, now in runtime["model_sources"].items()
                        if gpu["manifest"]["content_digests"][path] != now}
    assert len(gpu_source_delta) == 7
    assert gpu["manifest"]["framework_digests"] == runtime["framework_sources"]
    b3 = read(SAMPLES / "gpu_fix1_runtime_under_test.json")
    protocol_path = "exp/r17_integration/models/dp/protocols.py"
    assert runtime["model_sources"][protocol_path] == b3["model_sources"][protocol_path]
    stock = ("src/openpi/cache/config.py", "src/openpi/cache/types.py",
             "src/openpi/cache/storage_types.py", "src/openpi/serving/websocket_policy_server.py")
    for name in stock:
        assert subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT) == (ROOT / name).read_bytes()
    code = sorted(DP.glob("*.py")) + sorted(TESTS.glob("*.py")) + sorted(SAMPLES.glob("gpu_fix2_*.py"))
    for path in code:
        ast.parse(path.read_text(), filename=str(path))
    paths = set(code)
    paths.update(path for folder in SAMPLES.glob("gpu_fix2_*") if folder.is_dir()
                 for path in folder.rglob("*") if path.is_file())
    paths.update(SAMPLES / name for name in ("gpu_fix2_runtime_under_test.json", "gpu_fix2_release_loader_bundle_receipt.json"))
    paths.update(DP.glob("cpu_gpu_fix2_*.xml"))
    paths.add(DP / "README.md")
    paths.update(ROOT / "exp/r17_integration" / name for name in
                 ("handbacks/HANDBACK_P4.md", "REQUESTS/P4_to_Q.md", "REQUESTS/P4_to_P1.md", "REQUESTS/P4_to_P6.md"))
    artifacts = {str(path.relative_to(ROOT)): sha(path) for path in sorted(paths)}
    summary = dict(schema="r17.p4.gpu_fix2.summary.v1", production_max_batch_size=1,
        production_batching="unsupported", current_release_GPU_qualified=False,
        gpu_measurement_count=1, gpu_executed=True, gpu_measurement_seconds=gpu["elapsed_seconds"],
        gpu_measured_previous_model_identity=seal["old_model_identity"],
        gpu_measurements=gpu["measurements"], gpu_measurement_source_delta=gpu_source_delta,
        suites=suites, distinct_accepted_pytest_passed=202, current_probe_checks=39, probes=probes,
        bundle_checks=dict(passed=18, skipped=0, elapsed_seconds=bundles["elapsed_seconds"]),
        runtime=runtime, current_model_source_drift={}, current_framework_source_drift={},
        reviewer_B3_protocol_source_unchanged=True, frozen_stock_HEAD_byte_parity=list(stock),
        Q_reseal=seal, Q_input=provenance, parsed_sources=len(code), artifacts=artifacts,
        consumed_request_sha256=sha(ROOT / "exp/r17_integration/REQUESTS/Q_to_P4.md"),
        limits=["Single GPU run measured the pinned GPU fix1 release before implementation changes",
                "No current-release GPU/renderer/SR/throughput/quality qualification",
                "Q must reseal and retry B1 native/concurrent gates; no qualification writes by P4",
                "Production cache/MISS/recipe/teacher arms require B1; batch diagnostics do not qualify cost",
                "Local MW CPU probes use diagnostic ML45, not remote formal MT50"])
    (SAMPLES / "gpu_fix2_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status="pass", pytest_passed=202, CPU_probe_checks=39, bundle_checks=18,
                         GPU_diagnostic_runs=1, GPU_diagnostic_comparisons=8, current_source_drift=0,
                         model_dependencies=len(runtime["model_sources"]), parsed_sources=len(code),
                         artifact_hashes=len(artifacts))))


if __name__ == "__main__":
    main()
