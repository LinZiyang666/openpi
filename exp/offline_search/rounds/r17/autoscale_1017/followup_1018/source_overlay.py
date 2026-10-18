"""Test staged source bytes without activating them for future prepare calls."""
import builtins
import importlib.abc
import importlib.util
import io
import json
import os
from pathlib import Path
import sys


def install():
    here=Path(__file__).resolve().parent
    root=here.parents[5]
    variant=Path(os.environ['AUTOSCALE_SOURCE_VARIANT'])
    pins=json.loads((here/'BEFORE_SOURCES.json').read_text())
    mapping={str(root/name):variant/Path(name).name for name in pins}
    modules={name.removesuffix('.py').replace('/','.'):(root/name,variant/Path(name).name) for name in pins}
    def wrap(native):
        def opened(file,mode='r',*a,**kw):
            if isinstance(file,(str,bytes,os.PathLike)):
                mapped=mapping.get(os.path.abspath(os.fsdecode(file)))
                if mapped:
                    if any(c in mode for c in 'wax+'):raise PermissionError('test overlay is read only')
                    file=mapped
            return native(file,mode,*a,**kw)
        return opened
    builtins.open=wrap(builtins.open);io.open=wrap(io.open)
    class Loader(importlib.abc.Loader):
        def __init__(self,original,saved):self.original,self.saved=original,saved
        def create_module(self,spec):return None
        def get_source(self,name):return self.saved.read_text()
        def get_code(self,name):return compile(self.saved.read_bytes(),str(self.original),'exec')
        def exec_module(self,module):exec(self.get_code(module.__name__),module.__dict__)
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self,name,path=None,target=None):
            if name in modules:
                original,saved=modules[name]
                return importlib.util.spec_from_file_location(name,original,loader=Loader(original,saved))
    sys.meta_path.insert(0,Finder())
