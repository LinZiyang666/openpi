"""Audit existing CPU receipts/seals; no inference, GPU or external writes."""
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
    snapshot = read(SAMPLES / "gpu_fix1_runtime_under_test.json")
    for group in ("model_sources", "framework_sources"):
        for path, expected in snapshot[group].items():
            assert sha(ROOT / path) == expected, f"source drift: {path}"
    pytest = {}
    for name, total, failures in (("full", 134, 0), ("legacy", 49, 0),
                                   ("targeted", 9, 0), ("negative", 1, 1)):
        suite = ET.parse(DP / f"cpu_gpu_fix1_{name}.xml").getroot().find("testsuite")
        assert suite is not None
        counts = {key: int(suite.attrib[key]) for key in ("tests", "failures", "skipped", "errors")}
        assert counts == dict(tests=total, failures=failures, skipped=0, errors=0)
        pytest[name] = dict(counts, passed=total-failures, seconds=float(suite.attrib["time"]),
                            expected_exit_code=int(bool(failures)))
    probes = {}
    for name in ("q_libero", "release_libero", "release_metaworld", "mw_canonical"):
        receipt = read(SAMPLES / f"gpu_fix1_{name}_cpu/receipt.json")
        assert receipt["status"] == "pass" and receipt["exit_code"] == 0
        assert receipt["passed"] == len(receipt["checks"]) == 13 and receipt["skipped"] == 0
        assert receipt["device"] == "cpu" and receipt["diagnostic"] is True
        assert receipt["manifest"]["strict_load"] == dict(missing=[], unexpected=[], keys=475)
        manifest = receipt["manifest"]
        assert {path: manifest["content_digests"][path] for path in snapshot["model_sources"]} == snapshot["model_sources"]
        assert manifest["framework_digests"] == snapshot["framework_sources"]
        probes[name] = dict(passed=13, skipped=0, diagnostic=True, device="cpu",
                            elapsed_seconds=receipt["elapsed_seconds"],
                            model_identity=manifest["spec"]["checkpoint_identity"])
    q = read(SAMPLES / "gpu_fix1_q_libero_cpu/receipt.json")
    q_seal = read(SAMPLES / "gpu_fix1_q_libero_seal/receipt.json")
    original_path = Path(q_seal["source_load"])
    assert sha(original_path) == q_seal["source_sha256"]
    original = read(original_path)
    sealed = read(SAMPLES / "gpu_fix1_q_libero_seal/load.json")
    assert sealed == dict(original, expected_model_identity=q_seal["new_model_identity"])
    assert q["manifest"]["spec"] == read(SAMPLES / "gpu_fix1_q_libero_seal/model_spec.json")
    assert read(SAMPLES / "gpu_fix1_q_libero_cpu/load_cpu.json") == dict(sealed, device="cpu")
    provenance = q["observation_provenance"]
    assert sha(Path(provenance["path"])) == provenance["npz_sha256"] == snapshot["Q_npz_sha256"]
    assert provenance["image_shapes"] == {key: [256, 256, 3] for key in
                                          ("observation/image", "observation/wrist_image")}
    assert provenance["available_rows"] == 3 and provenance["native_check_row"] == 0
    assert q["load_spec_provenance"]["overridden_fields"] == ["device"]
    assert q["load_spec_provenance"]["sha256"] == sha(SAMPLES / "gpu_fix1_q_libero_seal/load.json")
    bundles = read(SAMPLES / "gpu_fix1_release_loader_bundle_receipt.json")
    assert bundles["passed"] == len(bundles["checks"]) == 18 and bundles["skipped"] == bundles["exit_code"] == 0
    stock = ("src/openpi/cache/config.py", "src/openpi/cache/types.py",
             "src/openpi/cache/storage_types.py", "src/openpi/serving/websocket_policy_server.py")
    for name in stock:
        head = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT)
        assert head == (ROOT / name).read_bytes(), f"frozen HEAD mismatch: {name}"
    source_files = sorted(DP.glob("*.py")) + sorted(TESTS.glob("*.py"))
    for path in source_files:
        ast.parse(path.read_text(), filename=str(path))
    paths = set(SAMPLES.glob("gpu_fix1_*.py"))
    paths.update(path for folder in SAMPLES.glob("gpu_fix1_*") if folder.is_dir()
                 for path in folder.rglob("*") if path.is_file())
    paths.update(SAMPLES / name for name in ("gpu_fix1_runtime_under_test.json",
                                            "gpu_fix1_release_loader_bundle_receipt.json"))
    paths.update(DP.glob("cpu_gpu_fix1_*.xml"))
    paths.update([DP / "README.md", TESTS / "test_dp_gpu_fix1.py", TESTS / "run_gpu_fix1_negative.py"])
    paths.update(ROOT / "exp/r17_integration" / name for name in
                 ("handbacks/HANDBACK_P4.md", "REQUESTS/P4_to_P1.md", "REQUESTS/P4_to_P6.md"))
    artifacts = {str(path.relative_to(ROOT)): sha(path) for path in sorted(paths)}
    summary = dict(schema="r17.p4.gpu_fix1.summary.v1", device="cpu", gpu_executed=False,
        pytest=pytest, distinct_accepted_pytest_passed=183,
        negative_note="One expected failure from scoped missing-option mutation; not historical checkout",
        model_sources=snapshot["model_sources"], framework_sources=snapshot["framework_sources"],
        model_source_drift={}, framework_source_drift={},
        probes=probes, probe_check_total=52,
        pinned_bundle_checks=dict(passed=18, skipped=0, exit_code=0, seconds=bundles["elapsed_seconds"]),
        q_input=provenance, q_reseal=q_seal, frozen_stock_HEAD_byte_parity=list(stock),
        parsed_source_count=len(source_files), artifacts=artifacts,
        formal_mt50_export_local=Path("/data/mw_finetune/dp/mt50_policy_100000/complete.json").is_file(),
        limits=["Only row0 plus existing B2 prompt variant tested by bounded native probe",
                "Formal diagnostic=False test stops at descriptor/receipt boundary; no local formal weight qualification",
                "MW examples use diagnostic ML45; canonical MW NPZ is explicitly synthetic",
                "Q must reseal asset/runtime/payload before GPU retry; qualification files untouched",
                "No GPU, renderer, SR or new quality/cost qualification"])
    (SAMPLES / "gpu_fix1_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status="pass", pytest_passed=183, probe_checks=52, bundle_checks=18,
                         model_dependencies=len(snapshot["model_sources"]),
                         framework_dependencies=len(snapshot["framework_sources"]),
                         source_drift=0, parsed_sources=len(source_files), artifact_hashes=len(artifacts))))


if __name__ == "__main__":
    main()
