"""Process-local read-only aliases for pinned cache files; canonical YAML stays intact.

Only explicitly listed files and their directory metadata are aliased. No model
math, configuration identity, write target or global filesystem is changed.
"""
def install(files):
    import builtins
    import io
    import os
    from pathlib import Path
    from contextlib import ExitStack
    from unittest.mock import patch
    files = {os.path.abspath(k): os.path.abspath(v) for k, v in files.items()}
    directories = {}
    for old, new in files.items():
        for a, b in zip(Path(old).parents, Path(new).parents):
            if str(a) in ('/', '/data', '/home'):break
            previous = directories.setdefault(str(a), str(b))
            if previous != str(b):raise ValueError('ambiguous alias parent')
    def translated(path, directory=False):
        if isinstance(path, int):return path
        key = os.path.abspath(os.fsdecode(path))
        target = files.get(key, directories.get(key) if directory else None)
        return os.fsencode(target) if target and isinstance(path, bytes) else target or path
    stack = ExitStack()
    for owner, name in ((builtins, 'open'), (io, 'open')):
        native = getattr(owner, name)
        def opening(path, mode='r', *args, _native=native, **kwargs):
            target = translated(path)
            if target != path and any(x in mode for x in 'wax+'):
                raise PermissionError('pinned cache alias is read-only')
            return _native(target, mode, *args, **kwargs)
        stack.enter_context(patch.object(owner, name, opening))
    for name in ('stat', 'lstat', 'access', 'scandir', 'listdir'):
        native = getattr(os, name)
        def call(path, *args, _native=native, **kwargs):
            return _native(translated(path, True), *args, **kwargs)
        stack.enter_context(patch.object(os, name, call))
    return stack
