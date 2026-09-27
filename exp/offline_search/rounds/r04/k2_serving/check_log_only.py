"""Exercise the real --os-log-r4 parser/wrapper with the legacy CPU interceptor stand-in."""
from exp.offline_search.closed_loop import plugin, selftest
parse=plugin.parse_cli
plugin.parse_cli=lambda argv: parse([*argv, '--os-log-r4'])
orig_infer=selftest.FakePolicy.infer
selftest.FakePolicy._osp_prepare_blind=lambda self, obs: None
selftest.FakePolicy._osp_blind_output=lambda self, action, state: {'actions':action}
selftest.FakePolicy.stage1=lambda self, obs: None
def infer(self, obs):
    self.stage1(obs)
    return orig_infer(self, obs)
selftest.FakePolicy.infer=infer
rc=selftest.main(['--cell','pi05_spatial_cache','--yaml',
 '/home/weiland/trace_runs/os_closed_loop/r02_g50/config/oscl50_p_sp_cl0.yaml',
 '--method','exp.offline_search.closed_loop.probe:ProbeForce','--judge','periodic:5',
 '--episodes','2','--out','exp/offline_search/rounds/r04/k2_serving/results/final_log_only'])
assert rc == 0
import json
rows=[json.loads(l) for l in plugin.RUNTIME.dec_path.read_text().splitlines() if '"ev": "dec"' in l]
assert all(r['vision'] and r['s1_ms'] is not None and r['served_head'] is not None for r in rows)
assert rows[-1]['stage1_calls'] == len(rows)
print(json.dumps({'log_only':True,'decisions':len(rows),'stage1_calls':rows[-1]['stage1_calls']}))
