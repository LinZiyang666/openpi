"""Isolated fault recreation; expected exit1 and three regression failures."""
import json

import numpy as np
import pytest

from exp.r17_integration.models.dp import provider as dp_provider
from exp.r17_integration.models.dp.manifest import file_digest
from exp.r17_integration.tests.models.dp import test_dp_fix_round2 as tests


class ReviewedFaults:
    @pytest.fixture(autouse=True)
    def recreate(self, monkeypatch, request):
        original = dp_provider.admission_validators
        def missing_role_gate(spec, sampler_for_steps):
            # Round1 had only the lossy metadata, with no admission consumer.
            original(spec, sampler_for_steps)
            return lambda *args: None, lambda *args: None
        monkeypatch.setattr(dp_provider, "admission_validators", missing_role_gate)
        writer = request.node.module.write_libero_qualification_fixture
        def fabricated_wrist(path, **kwargs):
            receipt = writer(path, **kwargs)
            with np.load(path, allow_pickle=False) as data:
                arrays = {key: data[key].copy() for key in data.files}
            # Recreate qualification/model_runtime's reviewed old camera path.
            arrays["observation/wrist_image"] = arrays["observation/image"][:, ::-1].copy()
            np.savez_compressed(path, **arrays)
            receipt["npz_sha256"] = file_digest(path)
            path.with_suffix(".json").write_text(json.dumps(receipt))
            return receipt
        monkeypatch.setattr(request.node.module, "write_libero_qualification_fixture", fabricated_wrist)


if __name__ == "__main__":
    raise SystemExit(pytest.main([
        "exp/r17_integration/tests/models/dp/test_dp_fix_round2.py",
        "-k", "shared_binding_rejects_default_K1 or qualification_fixture_uses_recorded_pair_verbatim",
        "-p", "no:cacheprovider", "-o", "junit_family=legacy",
        "--basetemp=exp/r17_integration/tests/models/dp/scratch_fix_round2_negative",
        "--junitxml=exp/r17_integration/models/dp/cpu_fix_round2_negative.xml", "-q", "--tb=short"],
        plugins=[ReviewedFaults()]))
