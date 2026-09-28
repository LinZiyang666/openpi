"""Candidate modules at their canonical names; no live-path edits."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
BASE=Path(__file__).resolve().parent
REPO=BASE.parents[4]
def load(source):
    if source != 'installed':
        for short in ('blind','plugin','verify_logs','selftest'):
            name='exp.offline_search.closed_loop.'+short
            spec=importlib.util.spec_from_file_location(name,BASE/source/(short+'.py'))
            module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
            if short=='plugin': module.REPO=REPO
    from exp.offline_search.closed_loop import plugin
    return plugin
if __name__=='__main__':
    source=sys.argv.pop(1);mode=sys.argv.pop(1)
    plugin=load(source)
    from exp.offline_search.closed_loop import selftest
    if mode.startswith('parity'):
        plugin.time=types.SimpleNamespace(time=lambda:1790550000.,perf_counter=lambda:10.,perf_counter_ns=lambda:10000000000)
        plugin.os=types.SimpleNamespace(**{**vars(os),'getpid':lambda:12345})
        sys.argv=['k10_parity']
    if mode in ('log_r4','parity_r4'):
        parse=plugin.parse_cli
        plugin.parse_cli=lambda argv:parse([*argv,'--os-log-r4'])
        original=selftest.FakePolicy.infer
        selftest.FakePolicy.stage1=lambda self,obs:None
        selftest.FakePolicy._osp_prepare_blind=lambda self,obs:None
        selftest.FakePolicy._osp_blind_output=lambda self,action,state:{'actions':action}
        def infer(self,obs):
            self.stage1(obs)
            return original(self,obs)
        selftest.FakePolicy.infer=infer
    args=json.loads(os.environ['K10_TEST_ARGS']) if 'K10_TEST_ARGS' in os.environ else sys.argv[1:]
    raise SystemExit(selftest.main(args))
