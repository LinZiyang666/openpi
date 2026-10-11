"""Scoped reproduction of the old missing production B>1 guard."""
import pytest


class MissingBatchGuard:
    @pytest.fixture(autouse=True)
    def recreate(self, monkeypatch):
        from exp.r17_integration.models.dp.staged import DPAdapter
        monkeypatch.setattr(DPAdapter, "_check_batch", lambda self, count: None)


if __name__ == "__main__":
    raise SystemExit(pytest.main([
        "exp/r17_integration/tests/models/dp/test_dp_gpu_fix2.py",
        "-k", "rejects_B2_before and stage1 and libero", "-p", "no:cacheprovider", "-o", "junit_family=legacy",
        "--basetemp=exp/r17_integration/tests/models/dp/scratch_gpu_fix2_negative",
        "--junitxml=exp/r17_integration/models/dp/cpu_gpu_fix2_negative.xml", "-q", "--tb=short"],
        plugins=[MissingBatchGuard()]))
