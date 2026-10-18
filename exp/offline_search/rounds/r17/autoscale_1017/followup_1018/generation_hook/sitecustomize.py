"""Read-only extra source overlay for the urgent WorkerLoop fix."""
import builtins
import importlib.abc
import importlib.util
import io
import os
from pathlib import Path
import sys

here=Path(__file__).resolve().parent.parent
root=here.parents[5]
if os.environ.get('AUTOSCALE_SOURCE_VARIANT'):
    spec=importlib.util.spec_from_file_location('fairness_source_overlay',here/'source_overlay.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.install()
original=root/'src/openpi/conductor/worker.py'
saved=here/os.environ.get('AUTOSCALE_GENERATION_VARIANT','generation_candidate')/'worker.py'
def wrap(native):
    def opened(file,mode='r',*a,**kw):
        if isinstance(file,(str,bytes,os.PathLike)) and os.path.abspath(os.fsdecode(file))==str(original):
            if any(c in mode for c in 'wax+'):raise PermissionError('read-only generation overlay')
            file=saved
        return native(file,mode,*a,**kw)
    return opened
builtins.open=wrap(builtins.open);io.open=wrap(io.open)
class Loader(importlib.abc.Loader):
    def create_module(self,spec):return None
    def get_source(self,name):return saved.read_text()
    def get_code(self,name):return compile(saved.read_bytes(),str(original),'exec')
    def exec_module(self,module):exec(self.get_code(module.__name__),module.__dict__)
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name=='openpi.conductor.worker':return importlib.util.spec_from_file_location(name,original,loader=Loader())
sys.meta_path.insert(0,Finder())
