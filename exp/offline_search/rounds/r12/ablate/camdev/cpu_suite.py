"""Run unchanged ablation/R12 CPU tests with legacy outputs isolated in /tmp.

Historical integration tests expect roots without runs/state. Copy only their
definition JSON/configs, rewrite owned paths, and keep all fitted inputs in
place. Selftests also get temporary scratch directories. No test is deselected.
"""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import pytest

from .prepare import HERE

MODULES = (
    'exp.offline_search.closed_loop.test_cache_ablate',
    'exp.offline_search.closed_loop.test_r12_pipeline',
    'exp.offline_search.closed_loop.test_r12_test_a',
    'exp.offline_search.closed_loop.test_look_recording',
    'exp.offline_search.rounds.r12.lookhalf.selftest',
    'exp.offline_search.rounds.r12.guard.selftest',
    'exp.offline_search.closed_loop.test_devset',
)


def replace_owned(data, source, target):
    if isinstance(data, str):
        return str(target) + data[len(str(source)):] if data == str(source) or data.startswith(str(source) + '/') else data
    if isinstance(data, list):
        return [replace_owned(v, source, target) for v in data]
    if isinstance(data, dict):
        return {k: replace_owned(v, source, target) for k, v in data.items()}
    return data


class DefinitionSnapshots:
    def __init__(self, scratch):
        self.scratch = scratch
        self.roots = {}

    def snapshot(self, record):
        source = Path(record['root'])
        target = self.scratch / 'definitions' / source.name
        if source not in self.roots:
            target.mkdir(parents=True)
            for path in source.glob('*.json'):
                data = replace_owned(json.loads(path.read_text()), source, target)
                (target / path.name).write_text(json.dumps(data, indent=2) + '\n')
            # Manifest bytes, unlike owned config paths, must remain exact.
            for name in ('dev_selection.json', 'eval500.json'):
                if (source / name).is_file():
                    shutil.copyfile(source / name, target / name)
            shutil.copytree(source / 'config', target / 'config')
            self.roots[source] = target
        return replace_owned(copy.deepcopy(record), source, target)

    def pytest_collection_modifyitems(self, items):
        for item in items:
            if item.module.__name__ in (MODULES[0], MODULES[2]) and hasattr(item, 'callspec'):
                record = item.callspec.params.get('record')
                if record is not None:
                    item.callspec.params['record'] = self.snapshot(record)


def isolate_selftest_scratch(scratch):
    r12 = HERE.parents[1]
    mirror = scratch / 'r12'
    mirror.mkdir()
    for source in r12.iterdir():
        if source.name not in ('lookhalf', 'guard'):
            (mirror / source.name).symlink_to(source, target_is_directory=source.is_dir())
            continue
        target = mirror / source.name
        target.mkdir()
        for child in source.iterdir():
            if child.name == 'scratch':
                continue
            if child.name == 'coordinator_commands.sh':
                (target / child.name).write_text(child.read_text().replace(f'BASE={source / "pilots"}', f'BASE={target / "pilots"}'))
            else:
                (target / child.name).symlink_to(child, target_is_directory=child.is_dir())
        importlib.import_module(f'exp.offline_search.rounds.r12.{source.name}.selftest').HERE = target


def main():
    with tempfile.TemporaryDirectory(prefix='r12_camdev_cpu_') as directory:
        scratch = Path(directory)
        isolate_selftest_scratch(scratch)
        snapshots = DefinitionSnapshots(scratch)
        args = [str(Path(importlib.import_module(name).__file__)) for name in MODULES]
        args += ['-q', '-p', 'no:cacheprovider', '--junitxml=' + str(HERE / 'CPU_REGRESSION.xml')]
        print('Legacy roots: definition snapshots in /tmp; frozen artifacts unchanged; no tests deselected.', flush=True)
        return pytest.main(args, plugins=[snapshots])


if __name__ == '__main__':
    sys.exit(main())
