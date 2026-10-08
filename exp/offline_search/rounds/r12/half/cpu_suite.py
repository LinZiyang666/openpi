"""Run prior R12 equivalence suites on temporary definition/scratch snapshots."""
from pathlib import Path
import importlib
import sys
import tempfile
import pytest
from exp.offline_search.rounds.r12.ablate.camdev.cpu_suite import (
    MODULES, DefinitionSnapshots, isolate_selftest_scratch)
from .build import HERE


def main():
    extra = ('exp.offline_search.rounds.r07.c2_wrist.test_dispatch',
             'exp.offline_search.closed_loop.ops.h100.test_collect_split')
    with tempfile.TemporaryDirectory(prefix='r12_half_regression_') as directory:
        scratch = Path(directory)
        isolate_selftest_scratch(scratch)
        snapshots = DefinitionSnapshots(scratch)
        args = [str(Path(importlib.import_module(name).__file__)) for name in (*MODULES,*extra)]
        args += ['-q','-p','no:cacheprovider','--junitxml='+str(HERE/'CPU_REGRESSION.xml')]
        print('Legacy definitions and selftest writes isolated in /tmp; mocked transport only.',flush=True)
        return pytest.main(args,plugins=[snapshots])


if __name__=='__main__':
    sys.exit(main())
