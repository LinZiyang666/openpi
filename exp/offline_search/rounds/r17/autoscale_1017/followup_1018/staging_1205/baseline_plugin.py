"""Read-only imports of the four pre-slot4 source snapshots."""
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).parent
FILES={name[:-3].replace('/','.') : HERE/'before'/Path(name).name
       for name in json.loads((HERE/'BEFORE_SOURCES.json').read_text())}
class Snapshot(importlib.abc.MetaPathFinder,importlib.abc.Loader):
    def find_spec(self,fullname,path=None,target=None):
        if fullname in FILES:return importlib.util.spec_from_loader(fullname,self)
    def create_module(self,spec):return None
    def exec_module(self,module):
        path=FILES[module.__name__];module.__file__=str(path)
        exec(compile(path.read_text(),str(path),'exec'),module.__dict__)
sys.meta_path.insert(0,Snapshot())
