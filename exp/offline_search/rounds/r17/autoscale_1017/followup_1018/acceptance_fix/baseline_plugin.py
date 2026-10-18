"""Import the five pre-followup modules without changing the shared tree."""
import importlib.abc
import importlib.util
from pathlib import Path
import sys

HERE=Path(__file__).parent
BASE=HERE.parents[1]
PREFIX='exp.offline_search.rounds.r17.autoscale_1017.'


class Baseline(importlib.abc.MetaPathFinder,importlib.abc.Loader):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.startswith(PREFIX):
            name=fullname[len(PREFIX):]+'.py'
            if '/' not in name and (HERE/'before'/name).is_file():
                return importlib.util.spec_from_loader(fullname,self)
    def create_module(self,spec):return None
    def exec_module(self,module):
        name=module.__name__[len(PREFIX):]+'.py'
        module.__file__=str(BASE/name)
        exec(compile((HERE/'before'/name).read_text(),module.__file__,'exec'),module.__dict__)


sys.meta_path.insert(0,Baseline())
