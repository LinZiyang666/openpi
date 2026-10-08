"""Run the existing R13 mechanics and regression runner in /tmp scratch."""
from __future__ import annotations
import argparse
import tempfile
from pathlib import Path
import pytest
from .common import HERE, REPO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('suite', choices=('r13','regression'))
    a = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix='r14_regression_') as scratch:
        scratch = Path(scratch)
        if a.suite == 'r13':
            from exp.offline_search.rounds.r13.impl import prepare
            original = prepare.write_json
            def isolated(path, value):
                path = Path(path)
                if path.is_relative_to(prepare.HERE):
                    path = HERE / 'r13_rerun' / path.relative_to(prepare.HERE)
                return original(path,value)
            prepare.write_json = isolated
            files = [str(prepare.HERE / 'test_r13.py'),str(prepare.HERE / 'test_compatibility.py')]
            plugins = []
        else:
            from exp.offline_search.rounds.r13.impl import regression_runner as runner
            # Keep its original suite and isolation plugin. Its temporary
            # symlink read views remain entirely outside exp/offline_search.
            runner.HERE = scratch
            runner.view(runner.lookhalf,'lookhalf')
            runner.view(runner.guard,'guard')
            files = [str(p) for p in sorted((REPO / 'exp/offline_search/closed_loop').glob('test_*.py'))]
            files += ['exp/offline_search/rounds/r11/knob/tests.py',
                      'exp/offline_search/rounds/r12/lookhalf/selftest.py',
                      'exp/offline_search/rounds/r12/guard/selftest.py']
            plugins = [runner.IsolateScratch()]
        return pytest.main(['-q','-p','no:cacheprovider',*files,
            '--basetemp=' + str(scratch / 'pytest'), '--junitxml=' + str(HERE / (a.suite + '_tests.xml'))], plugins=plugins)

if __name__ == '__main__':
    raise SystemExit(main())
