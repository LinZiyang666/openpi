"""Q_to_P4: pinned descriptor, recorded256 input, formal flag and fallback."""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from exp.r17_integration.models.dp import probe
from exp.r17_integration.models.dp.config import LoadSpec
from exp.r17_integration.models.dp.manifest import file_digest

Q_LIBERO = Path("/data/mw_finetune/qualification/assets/dp_libero_fix2_b3a3_v1")


def test_Q_sealed_official_LIBERO256_probe_passes_all_native_checks(monkeypatch, tmp_path):
    """Real475 tensors and Q's original paired256, no native inference mocks."""
    from exp.r17_integration.models.dp.loader import load_adapter
    from exp.r17_integration.common import provider as registry
    from exp.r17_integration.models.dp.protocols import libero_spec_from_document
    if not all((Q_LIBERO / name).is_file() for name in ("load.json", "wire_observations.npz")):
        pytest.skip("Q original paired256 and official sealed descriptor unavailable")
    monkeypatch.setattr(registry, "_providers", {})
    monkeypatch.setattr(registry, "_traces", {})
    original = json.loads((Q_LIBERO / "load.json").read_text())
    assert original["domain"] == "libero" and original["diagnostic"] is True
    # Reseal in owned scratch against this probe source. Never remove any pin
    # in the probe itself or mutate Q's previous source-sealed assets.
    unpinned = {k: v for k, v in original.items() if k != "expected_model_identity"}
    adapter = load_adapter(dict(unpinned, device="cpu"))
    sealed = dict(original, expected_model_identity=adapter.metadata().checkpoint_identity)
    path = tmp_path / "load.json"
    path.write_text(json.dumps(sealed))
    seen, prepared = [], []
    real_prepare = adapter.prepare
    def checked_prepare(obs, request):
        prepared.append((obs, request))
        return real_prepare(obs, request)
    monkeypatch.setattr(adapter, "prepare", checked_prepare)
    def pinned_loader(document):
        assert document == dict(sealed, device="cpu")
        seen.append(document)
        # Exercise the real loader and all three supplied pins.
        resident = load_adapter(document)
        assert asdict(resident.metadata()) == asdict(adapter.metadata())
        return adapter
    monkeypatch.setattr(probe, "load_adapter", pinned_loader)
    # The optional pair path must bypass the standalone96 fallback entirely.
    from exp.r17_integration.models.dp import fixtures
    monkeypatch.setattr(fixtures, "libero_observation", lambda: pytest.fail("sealed256 probe used fallback96"))
    output = tmp_path / "native"
    monkeypatch.setattr(sys, "argv", ["dp.probe", "--load-spec", str(path),
        "--observations", str(Q_LIBERO / "wire_observations.npz"), "--device", "cpu", "--output", str(output)])
    probe.main()
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "pass" and receipt["passed"] == len(receipt["checks"]) == 13
    assert receipt["skipped"] == 0 and receipt["diagnostic"] is True and seen
    assert "real_production_B1_only" in {c["name"] for c in receipt["checks"]}
    assert "real_B1_B2_sampler_equivalence" not in {c["name"] for c in receipt["checks"]}
    assert receipt["diagnostics"][0]["status"] == "diagnostic_only"
    assert receipt["manifest"]["spec"] == json.loads(json.dumps(asdict(adapter.metadata())))
    assert receipt["manifest"]["strict_load"] == {"missing": [], "unexpected": [], "keys": 475}
    benchmark = libero_spec_from_document(sealed["libero_benchmark_specs"][0], replan_steps=5)
    assert receipt["manifest"]["spec"]["allowed_benchmark_protocols"] == [benchmark.identity]
    assert len(benchmark.tasks) == 40 and benchmark.wire_size == 256
    with np.load(Q_LIBERO / "wire_observations.npz", allow_pickle=False) as data:
        for key in ("observation/image", "observation/wrist_image", "observation/state"):
            assert prepared[0][0][key].tobytes() == data[key][0].tobytes()
        assert not np.array_equal(data["observation/wrist_image"][0], data["observation/image"][0][::-1])
    assert prepared[0][1].benchmark_protocol == benchmark.identity
    assert receipt["observation_provenance"]["npz_sha256"] == file_digest(Q_LIBERO / "wire_observations.npz")
    assert json.loads((output / "load_cpu.json").read_text()) == dict(sealed, device="cpu")


def test_sealed_formal_descriptor_only_device_overridden_and_actual_flag_reported(monkeypatch, tmp_path):
    """Descriptor/receipt boundary witness; no claim of loading remote MT50."""
    class StopAfterDescriptor(RuntimeError):
        pass
    document = asdict(LoadSpec("/immutable/mt50_policy_100000", "metaworld", text_encoder="/sealed/clip",
        device="cuda:0", diagnostic=False, expected_weight_sha256="sha256:weights",
        expected_norm_identity="sha256:norm", expected_model_identity="sha256:model"))
    path = tmp_path / "formal.json"
    path.write_text(json.dumps(document))
    def stop():
        raise StopAfterDescriptor("descriptor boundary only")
    def loader(actual):
        assert actual == dict(document, device="cpu")
        return SimpleNamespace(domain="metaworld", diagnostic=False, manifest=stop)
    monkeypatch.setattr(probe, "load_adapter", loader)
    output = tmp_path / "formal"
    monkeypatch.setattr(sys, "argv", ["dp.probe", "--load-spec", str(path), "--device", "cpu",
        "--text-encoder", "/must/not/override/sealed/text", "--output", str(output)])
    with pytest.raises(StopAfterDescriptor):
        probe.main()
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["diagnostic"] is False and receipt["domain"] == "metaworld"
    assert receipt["load_spec_provenance"]["overridden_fields"] == ["device"]
    assert json.loads(path.read_text()) == document


def test_existing_checkpoint_domain_CLI_keeps_identical_fallback_descriptor(monkeypatch, tmp_path):
    class StopAtLoader(RuntimeError):
        pass
    def loader(spec):
        assert spec == LoadSpec("immutable", "metaworld", "clip", "cpu")
        raise StopAtLoader
    monkeypatch.setattr(probe, "load_adapter", loader)
    monkeypatch.setattr(sys, "argv", ["dp.probe", "--checkpoint", "immutable", "--domain", "metaworld",
        "--text-encoder", "clip", "--output", str(tmp_path / "fallback")])
    with pytest.raises(StopAtLoader):
        probe.main()


@pytest.mark.parametrize("conflict", [("--domain", "metaworld"), ("--checkpoint", "/other/weights")])
def test_sealed_descriptor_rejects_conflicting_fallback_arguments(monkeypatch, tmp_path, conflict):
    path = tmp_path / "load.json"
    path.write_text(json.dumps(asdict(LoadSpec("/immutable/libero", "libero"))))
    monkeypatch.setattr(probe, "load_adapter", lambda _: pytest.fail("conflicting descriptor reached loader"))
    monkeypatch.setattr(sys, "argv", ["dp.probe", "--load-spec", str(path), *conflict,
        "--output", str(tmp_path / "conflict")])
    with pytest.raises(ValueError, match="differs from the sealed"):
        probe.main()


@pytest.mark.parametrize("fault", ["missing_wrist", "unaligned", "empty", "pickle"])
def test_optional_observation_NPZ_rejects_invalid_pair_without_fallback(tmp_path, fault):
    arrays = {"observation/image": np.zeros((2,256,256,3), np.uint8),
              "observation/wrist_image": np.ones((2,256,256,3), np.uint8),
              "observation/state": np.zeros((2,8), np.float32), "prompt": np.asarray(["pick", "place"])}
    if fault == "missing_wrist":
        arrays.pop("observation/wrist_image")
    elif fault == "unaligned":
        arrays["observation/wrist_image"] = arrays["observation/wrist_image"][:1]
    elif fault == "empty":
        arrays = {k: v[:0] for k, v in arrays.items()}
    else:
        arrays["prompt"] = np.asarray(["pick", "place"], dtype=object)
    path = tmp_path / "observations.npz"
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError):
        probe.fixed_observation(path, "libero")
