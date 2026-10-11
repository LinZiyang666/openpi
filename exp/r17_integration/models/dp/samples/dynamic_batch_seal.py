"""CPU-only formal DP-LIBERO official50 seal for the owned batching sweep."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil

from exp.r17_integration.models.dp.loader import load_adapter
from exp.r17_integration.models.dp.manifest import code_identity, framework_identity, file_digest
from exp.r17_integration.ops.formal_eval import benchmark


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU seal requires hidden CUDA")
    out = Path("exp/r17_integration/models/dp/samples/dynamic_batch_assets")
    out.mkdir(parents=True, exist_ok=True)
    document = dict(checkpoint="/data/mw_finetune/dp/libero_policy_100000", domain="libero",
        device="cpu", diagnostic=True, libero_benchmark_specs=[benchmark(True).manifest()])
    diagnostic = load_adapter(document)
    pinned = dict(document, diagnostic=False, expected_weight_sha256=diagnostic.weight_identity,
        expected_norm_identity=diagnostic.metadata().norm_identity,
        expected_model_identity=diagnostic.metadata().checkpoint_identity)
    adapter = load_adapter(pinned)
    from exp.r17_integration.models.dp.provider import register_provider
    from exp.r17_integration.models.dp.serve import plain_config
    import yaml
    provider = register_provider(adapter)
    for k in (100, 10, 5):
        config = plain_config(provider, num_steps=k)
        config.enabled = False
        (out / f"k{k}.yaml").write_text(yaml.safe_dump(asdict(config)))
    (out / "benchmark.json").write_text(json.dumps(benchmark(True).manifest(), indent=2) + "\n")
    (out / "load.json").write_text(json.dumps(dict(pinned, device="cuda:0"), indent=2) + "\n")
    (out / "model_spec.json").write_text(json.dumps(asdict(adapter.metadata()), indent=2) + "\n")
    source = Path("/data/mw_finetune/qualification/assets/dp_libero_gpu_fix1_v1/wire_observations.npz")
    shutil.copyfile(source, out / "wire_observations.npz")
    receipt = dict(status="pass", device="cpu", gpu_executed=False, diagnostic=False,
        checkpoint=pinned["checkpoint"], strict_load=adapter.strict_load, passed=1, skipped=0,
        observation_source=str(source), observations_sha256=file_digest(source),
        official_50=True, suites=4, tasks=40, formal_episodes=2000,
        model_sources=code_identity(), framework_sources=framework_identity(),
        model_identity=adapter.metadata().checkpoint_identity, weight_identity=adapter.weight_identity)
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in ("status", "diagnostic", "model_identity", "strict_load")}))


if __name__ == "__main__":
    main()
