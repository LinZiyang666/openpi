"""Recreate the old missing probe flags in isolation; expected one failure."""
import argparse

import pytest


class MissingProbeFlags:
    @pytest.fixture(autouse=True)
    def recreate(self, monkeypatch):
        original = argparse.ArgumentParser.add_argument
        def add_argument(parser, *names, **kwargs):
            if any(name in {"--load-spec", "--observations"} for name in names):
                return None
            return original(parser, *names, **kwargs)
        monkeypatch.setattr(argparse.ArgumentParser, "add_argument", add_argument)


if __name__ == "__main__":
    raise SystemExit(pytest.main([
        "exp/r17_integration/tests/models/dp/test_dp_gpu_fix1.py",
        "-k", "sealed_formal_descriptor_only", "-p", "no:cacheprovider", "-o", "junit_family=legacy",
        "--basetemp=exp/r17_integration/tests/models/dp/scratch_gpu_fix1_negative",
        "--junitxml=exp/r17_integration/models/dp/cpu_gpu_fix1_negative.xml", "-q", "--tb=short"],
        plugins=[MissingProbeFlags()]))
