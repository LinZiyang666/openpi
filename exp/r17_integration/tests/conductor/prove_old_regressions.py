"""Replay new behavioral regressions against the preserved pre-fix runner.

This intentionally returns pytest's nonzero exit status. Only owned in-process
classes are replaced; files, stock modules, and processes are unchanged.
"""
from pathlib import Path
import importlib.util
import pytest


def main():
    old_path = Path('exp/r17_integration/tests/conductor/scratch_p2_binding_release/test_client_payload_copied_fil0/exp/r17_integration/benchmarks/metaworld.py')
    spec = importlib.util.spec_from_file_location('exp.r17_integration.benchmarks.pre_fix_metaworld', old_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Import the actual regression module once and replace just its runner.
    from exp.r17_integration.tests.conductor import test_fix_round1 as regression
    regression.R17MetaWorldEpisodeRunner = module.R17MetaWorldEpisodeRunner
    # pytest's default basename import differs from the namespace import.
    import sys
    sys.modules['test_fix_round1'] = regression
    return pytest.main(['exp/r17_integration/tests/conductor/test_fix_round1.py',
                        '-k', 'mw_cost_final_ack or terminal_ack_transport_failure', '-q', '--tb=short',
                        '-p', 'no:cacheprovider', '--junitxml=exp/r17_integration/tests/conductor/cpu_fix1_old_red.xml'])


if __name__ == '__main__':
    raise SystemExit(main())
