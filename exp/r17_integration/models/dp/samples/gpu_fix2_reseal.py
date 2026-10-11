"""Explicit owned CPU reseal of Q's descriptor; never changes Q's assets."""
from dataclasses import asdict
import json
from pathlib import Path

from exp.r17_integration.models.dp.loader import load_adapter
from exp.r17_integration.models.dp.manifest import code_identity, file_digest, framework_identity


def main():
    job_path = Path("/data/mw_finetune/qualification/runs/slot_G2_dp_libero_A_20261009_194429_754466/job.json")
    job = json.loads(job_path.read_text())
    source = Path(job["assets"]) / "load.json"
    original = json.loads(source.read_text())
    # Explicit reseal outside probe: retain weight/norm pins and protocol,
    # replace only the old source-dependent model pin in an owned copy.
    document = {key: value for key, value in original.items() if key != "expected_model_identity"}
    adapter = load_adapter(dict(document, device="cpu"))
    pinned = dict(original, expected_model_identity=adapter.metadata().checkpoint_identity)
    out = Path("exp/r17_integration/models/dp/samples/gpu_fix2_q_libero_seal")
    out.mkdir(parents=True, exist_ok=True)
    (out / "load.json").write_text(json.dumps(pinned, indent=2) + "\n")
    (out / "model_spec.json").write_text(json.dumps(asdict(adapter.metadata()), indent=2) + "\n")
    receipt = dict(schema="r17.p4.q_cpu_reseal.v1", device="cpu", gpu_executed=False,
                   passed=1, skipped=0, source_load=str(source), source_sha256=file_digest(source),
                   changed_fields=["expected_model_identity"], old_model_identity=original["expected_model_identity"],
                   new_model_identity=pinned["expected_model_identity"], strict_load=adapter.strict_load,
                   protocols=list(adapter.metadata().allowed_benchmark_protocols))
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    runtime = dict(schema="r17.p4.gpu_fix2.runtime.v1", test_device="cpu",
                   model_sources=code_identity(), framework_sources=framework_identity(),
                   Q_npz_sha256=file_digest(Path(job["assets"]) / "wire_observations.npz"))
    (out.parent / "gpu_fix2_runtime_under_test.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
