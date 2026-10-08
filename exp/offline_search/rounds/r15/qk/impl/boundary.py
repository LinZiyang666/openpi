"""Fit-read audit also covers symlinked query corpora and their resolved aliases."""
from contextlib import contextmanager
import os
from pathlib import Path
import sys

_ACTIVE = None
_INSTALLED = False


@contextmanager
def library_boundary(root, library):
    global _ACTIVE, _INSTALLED
    if not _INSTALLED:
        def audit(event,args):
            if event!='open' or _ACTIVE is None or isinstance(args[0],int):
                return
            path,mode,flags=args
            if flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
                return
            lexical=Path(os.path.abspath(os.fsdecode(path)))
            resolved=lexical.resolve()
            store,lib,query,reads=_ACTIVE
            if ('queries' in lexical.parts or 'os_closed_loop' in lexical.parts
                    or resolved==query or query in resolved.parents
                    or 'os_closed_loop' in resolved.parts):
                raise ValueError(f'non-library fitting input refused before open: {lexical}')
            if lexical==store or store in lexical.parents or resolved==store or store in resolved.parents:
                if lib!=resolved and lib not in resolved.parents:
                    raise ValueError(f'fitting read escaped selected cell library: {lexical}')
                reads.add(str(resolved))
        sys.addaudithook(audit)
        _INSTALLED=True
    if _ACTIVE is not None:
        raise RuntimeError('fit-read scopes must not overlap')
    reads=set()
    root=Path(root).resolve()
    _ACTIVE=(root,Path(library).resolve(),(root/'queries').resolve(),reads)
    try:
        yield reads
    finally:
        _ACTIVE=None
