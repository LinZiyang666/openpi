"""Read-only 19-file overlay, preserving original absolute scientific pin paths.

Used by sitecustomize in baseline test subprocesses. The full checkout and
installed dependencies remain available. Reads/imports of the 19 source files
use verified backups; writes to those paths are refused. No worktree restoration.
"""
import builtins
import hashlib
import importlib.abc
import importlib.util
import io
import json
import os
from pathlib import Path
import sys


def install():
    root = Path(os.environ['AUTOSCALE_BASELINE_ROOT'])
    backup = Path('/home/weiland/.claude/jobs/35f2869b/tmp/autoscale_backup')
    pins = json.loads((root/'exp/offline_search/rounds/r17/autoscale_1017/PATCH_BASE.json').read_text())
    mapping, modules = {}, {}
    for name, row in pins.items():
        original, saved = root/name, backup/name.replace('/', '__')
        assert hashlib.sha256(saved.read_bytes()).hexdigest() == row['sha256'], name
        mapping[str(original)] = str(saved)
        if name.endswith('.py'):
            modules[name.removeprefix('src/').removesuffix('.py').replace('/', '.')] = (original, saved)
    # A concurrent agent may update a shared scientific dependency between
    # serial runs. Replay the baseline against the exact after-side dependency
    # bytes without restoring or editing any file in the shared checkout.
    snapshot = os.environ.get('AUTOSCALE_DEPENDENCY_SNAPSHOT')
    if snapshot:
        manifest = Path(snapshot)
        for name, row in json.loads(manifest.read_text()).items():
            original, saved = root/name, manifest.parent/row['file']
            assert hashlib.sha256(saved.read_bytes()).hexdigest() == row['sha256'], name
            assert str(original) not in mapping, name
            mapping[str(original)] = str(saved)
            modules[name.removeprefix('src/').removesuffix('.py').replace('/', '.')] = (original, saved)
    def wrap(native):
        def opened(file, mode='r', *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)):
                saved = mapping.get(os.path.abspath(os.fsdecode(file)))
                if saved:
                    if any(c in mode for c in 'wax+'):
                        raise PermissionError('baseline source is read-only: '+str(file))
                    file = saved
            return native(file, mode, *args, **kwargs)
        return opened
    builtins.open = wrap(builtins.open)
    io.open = wrap(io.open)
    class Loader(importlib.abc.Loader):
        def __init__(self, original, saved):self.original, self.saved = original, saved
        def create_module(self, spec):return None
        def exec_module(self, module):
            exec(compile(self.saved.read_bytes(), str(self.original), 'exec'), module.__dict__)
        def get_source(self, fullname):return self.saved.read_text()
        def get_code(self, fullname):return compile(self.saved.read_bytes(), str(self.original), 'exec')
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in modules:
                original, saved = modules[fullname]
                return importlib.util.spec_from_file_location(fullname, original, loader=Loader(original, saved))
    sys.meta_path.insert(0, Finder())
