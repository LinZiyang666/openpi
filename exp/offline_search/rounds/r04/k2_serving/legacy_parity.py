"""Literal decision-log parity under fixed clocks/pid; identical invocation/output path."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import types

base=Path(__file__).resolve().parent
variant, mode=sys.argv[1:]
source=base / ('baseline' if variant == 'before' else 'dev') / 'plugin.py'
spec=importlib.util.spec_from_file_location('exp.offline_search.closed_loop.plugin', source)
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
module.REPO=base.parents[3]
module.time=types.SimpleNamespace(time=lambda:1790550000., perf_counter=lambda:10., perf_counter_ns=lambda:10000000000)
module.os=types.SimpleNamespace(**{**vars(os),'getpid':lambda:12345})
from exp.offline_search.closed_loop import selftest
sys.argv=['k2_legacy_parity']
out=base / 'results' / ('parity_' + mode)
args=['--cell','pi05_spatial_cache','--yaml','/home/weiland/trace_runs/os_closed_loop/r02_g50/config/oscl50_p_sp_cl0.yaml',
      '--method','exp.offline_search.closed_loop.probe:' + ('ProbeB0' if mode == 'pure' else 'ProbeForce'),
      '--episodes','2','--out',str(out)]
if mode == 'mixed': args += ['--judge','guard_only']
rc=selftest.main(args)
assert rc == 0
shutil.copyfile(out/'decisions_selftest.jsonl',base/'results'/f'parity_{mode}_{variant}.jsonl')
if variant == 'after':
    before=(base/'results'/f'parity_{mode}_before.jsonl').read_bytes()
    after=(base/'results'/f'parity_{mode}_after.jsonl').read_bytes()
    assert before==after, (mode, len(before), len(after))
    print(json.dumps({'byte_identical':True,'mode':mode,'bytes':len(after),'lines':len(after.splitlines())}))
