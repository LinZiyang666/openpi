"""Load development/baseline shared modules without editing a live import path."""
import importlib.util
import os
from pathlib import Path
import sys
import types

BASE=Path(__file__).resolve().parent
REPO=BASE.parents[4]
source=sys.argv.pop(1)
mode=sys.argv.pop(1)
if source != 'installed':
    for name in ('plugin', 'verify_logs', 'selftest'):
        spec=importlib.util.spec_from_file_location('exp.offline_search.closed_loop.'+name, BASE/source/(name+'.py'))
        module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module
        spec.loader.exec_module(module)
        if name=='plugin': module.REPO=REPO
from exp.offline_search.closed_loop import plugin, selftest
if mode.startswith('parity'):
    plugin.time=types.SimpleNamespace(time=lambda:1790550000., perf_counter=lambda:10., perf_counter_ns=lambda:10000000000)
    plugin.os=types.SimpleNamespace(**{**vars(os),'getpid':lambda:12345})
    # Keep invocation stable; baseline and candidate use the identical output path.
    sys.argv=['k5_parity']
if mode in ('log_r4','parity_r4'):
    parse=plugin.parse_cli
    plugin.parse_cli=lambda argv: parse([*argv,'--os-log-r4'])
    original=selftest.FakePolicy.infer
    selftest.FakePolicy.stage1=lambda self, obs: None
    selftest.FakePolicy._osp_prepare_blind=lambda self, obs: None
    selftest.FakePolicy._osp_blind_output=lambda self, action, state: {'actions':action}
    def infer(self, obs):
        self.stage1(obs)
        return original(self,obs)
    selftest.FakePolicy.infer=infer
args=os.environ.get('K5_TEST_ARGS')
if args:
    import json
    args=json.loads(args)
else: args=sys.argv[1:]
rc=selftest.main(args)
if mode in ('log_r4','parity_r4'):
    import json
    rows=[json.loads(l) for l in plugin.RUNTIME.dec_path.read_text().splitlines() if '"ev": "dec"' in l]
    assert all(r['vision'] and r['s1_ms'] is not None and r['served_head'] is not None for r in rows)
    assert rows[-1]['stage1_calls']==len(rows)
raise SystemExit(rc)
