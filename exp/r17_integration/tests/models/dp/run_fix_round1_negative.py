"""Isolated negative control: recreate reviewed faults, run only regressions.

Never edits source files. Intentional pytest failures prove the assertions
detect the old behaviors; these are not acceptance-suite failures.
"""
import argparse
from dataclasses import replace

import pytest


class OldFaults:
    def pytest_sessionstart(self, session):
        from exp.r17_integration.models.dp.sampler import SamplerProfile
        from exp.r17_integration.models.dp.staged import DPAdapter
        from exp.r17_integration.models.dp import fixtures
        from exp.r17_integration.common.contracts import Capabilities, Capability
        real_scheduler = SamplerProfile.scheduler
        def leading(self, cfg):
            scheduler = real_scheduler(self, cfg)
            scheduler.register_to_config(timestep_spacing="leading")
            scheduler.set_timesteps(self.num_steps)
            return scheduler
        SamplerProfile.scheduler = leading
        real_snapshot = DPAdapter._snapshot
        def per_step_digest(self, s2, sampler, index, x, generator, **kwargs):
            return real_snapshot(self, s2, sampler, index, x, generator)
        DPAdapter._snapshot = per_step_digest
        real_init = DPAdapter.__init__
        def overclaim(self, *args, **kwargs):
            real_init(self, *args, **kwargs)
            values = dict(self._spec.capabilities.features)
            for name in ("native_warm_resume", "native_warm_reset", "library", "recipe"):
                values[name] = Capability("supported", "old reviewed claim")
            self._spec = replace(self._spec, capabilities=Capabilities(values))
        DPAdapter.__init__ = overclaim
        real_pair = fixtures.libero_observation
        def fake_wrist(*args, **kwargs):
            obs, origin = real_pair(*args, **kwargs)
            obs["observation/wrist_image"] = obs["observation/image"][::-1].copy()
            return obs, origin
        fixtures.libero_observation = fake_wrist
        real_prepare = DPAdapter.prepare
        def arbitrary_resolution(self, obs, request):
            if self.domain == "libero":
                from PIL import Image
                import numpy as np
                obs = dict(obs)
                for key in ("observation/image", "observation/wrist_image"):
                    obs[key] = np.asarray(Image.fromarray(obs[key]).resize((96,96), Image.Resampling.BILINEAR)).copy()
            return real_prepare(self, obs, request)
        DPAdapter.prepare = arbitrary_resolution
        real_add = argparse.ArgumentParser.add_argument
        real_group_add = argparse._MutuallyExclusiveGroup.add_argument
        removed = {"--recipe-factory", "--capture-sink", "--debug-directory"}
        def without_hooks(self, *args, **kwargs):
            if removed.intersection(args):
                return None
            return real_add(self, *args, **kwargs)
        def group_without_hooks(self, *args, **kwargs):
            if removed.intersection(args):
                return None
            return real_group_add(self, *args, **kwargs)
        argparse.ArgumentParser.add_argument = without_hooks
        argparse._MutuallyExclusiveGroup.add_argument = group_without_hooks
        # The old CLI had no capture argument group at all.
        argparse.ArgumentParser.add_mutually_exclusive_group = lambda self, **kwargs: self


if __name__ == "__main__":
    base = "exp/r17_integration/tests/models/dp/"
    checks = ["test_sampler.py::test_every_reduced_K_starts_at_Tminus1_and_binds_trailing_identity",
              "test_dp_fix_round1.py::test_snapshot_condition_digest_once_per_request_not_once_per_step",
              "test_dp_fix_round1.py::test_capabilities_do_not_claim_generic_warm_library_recipe_qualification[metaworld]",
              "test_dp_fix_round1.py::test_libero_rejects_resampled_wire_and_binds_expected_size[224]",
              "test_dp_fix_round1.py::test_libero_true_paired_recorded_views_use_both_adapter_branches",
              "test_dp_fix_round1.py::test_serve_CLI_deploys_shared_five_layers_and_capture_even_at_full_K[sink-100]"]
    raise SystemExit(pytest.main([*(base+test for test in checks), "-p", "no:cacheprovider", "-q", "--tb=short",
        "--basetemp="+base+"scratch_fix_round1_negative",
        "--junitxml=exp/r17_integration/models/dp/cpu_fix_round1_negative.xml"], plugins=[OldFaults()]))
