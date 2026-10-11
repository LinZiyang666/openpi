"""CPU pinned loads and YAML admission, including rejection before publication."""
from dataclasses import asdict
import json
from pathlib import Path
import time

from exp.r17_integration.common.config import config_from_dict, load_cache_config
from exp.r17_integration.common.provider import validate_binding
from exp.r17_integration.models.dp.loader import load_adapter
from exp.r17_integration.models.dp.provider import register_provider
from exp.r17_integration.models.dp.serve import plain_config

base = Path("exp/r17_integration/models/dp/samples")
# Optional release prefix (default: P4 fix round 2); Review B3 resealed set: review_b3_release.
import sys
prefix = sys.argv[1] if len(sys.argv) > 1 else "fix_round2_release"
started = time.monotonic()
checks = []


def rejected(domain, name, document):
    try:
        config_from_dict(document)
    except ValueError as exc:
        assert "K1" in str(exc)
        checks.append({"check": name, "domain": domain, "status": "pass", "rejection": str(exc)})
    else:
        raise AssertionError("forbidden K1 binding accepted")


for domain in ("libero", "metaworld"):
    source = base / f"{prefix}_{domain}_cpu"
    adapter = load_adapter(json.loads((source / "load_cpu.json").read_text()))
    provider = register_provider(adapter)
    assert adapter.strict_load == {"missing": [], "unexpected": [], "keys": 475}
    checks.append({"check": "actual_pinned_loader", "domain": domain, "status": "pass",
                   "model_identity": adapter.metadata().identity})
    for k in (1, 2, 4, 10, 100):
        config = load_cache_config(source / f"k{k}.yaml")
        bound, sampler = validate_binding(config.adapter_binding)
        assert bound is provider and sampler.native_timesteps[0] == 99
        assert sampler.family == ("ddpm" if k == 100 else "ddim")
        if k == 1:
            assert config.adapter_binding["allow_lossy_sampler"] is True
            assert config.adapter_binding["sampler_study_label"].strip()
            assert config.adapter_binding["commit_horizon"] == 5 and not config.checkpoints and not config.recipe.enabled
        else:
            assert "allow_lossy_sampler" not in config.adapter_binding and "sampler_study_label" not in config.adapter_binding
        checks.append({"check": "bound_yaml_actual_resident", "domain": domain, "K": k, "status": "pass"})
    k1 = load_cache_config(source / "k1.yaml")
    unlabelled = asdict(k1)
    unlabelled["adapter_binding"].pop("allow_lossy_sampler")
    unlabelled["adapter_binding"].pop("sampler_study_label")
    rejected(domain, "K1_without_study_rejected", unlabelled)
    miss = asdict(k1)
    miss["checkpoints"] = asdict(plain_config(provider, num_steps=2, retrieval=True))["checkpoints"]
    miss["checkpoints"]["cp1"]["judge"]["type"] = "always_miss"
    rejected(domain, "K1_cache_MISS_rejected", miss)
    teacher = asdict(plain_config(provider, num_steps=2))
    teacher["adapter_binding"]["library_sampler_identity"] = provider.sampler_for_steps(1).identity
    rejected(domain, "K1_teacher_K2_MISS_rejected", teacher)

receipt = {"schema": "r17.p4.bundle_load_receipt.v2", "device": "cpu", "gpu_executed": False,
           "exit_code": 0, "passed": len(checks), "skipped": 0, "checks": checks,
           "elapsed_seconds": time.monotonic() - started}
(base / ("fix_round2_loader_bundle_receipt.json" if prefix == "fix_round2_release" else f"{prefix}_loader_bundle_receipt.json")).write_text(json.dumps(receipt, indent=2) + "\n")
print({key: receipt[key] for key in ("passed", "skipped", "elapsed_seconds")})
