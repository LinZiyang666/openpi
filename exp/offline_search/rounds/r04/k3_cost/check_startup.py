"""Exercise the server-owned emit hook against a real fitted PluginRuntime."""
import json,pathlib
from exp.offline_search.closed_loop import plugin,stage_overrides as cost
HERE=pathlib.Path(__file__).resolve().parent
out=HERE/'results/startup_smoke'
flags=['--os-stage1-mode','wrist_only','--os-method','exp.offline_search.rounds.r04.k3_cost.method:WristMixedJudge',
       '--os-kwargs','{"lib":"current","guards":true,"events":"none"}',
       '--os-fit-artifact','/home/weiland/trace_runs/offline_search_store/derived/r04/k3_cost/fits/pi05_spatial_current.pkl',
       '--os-no-shadow-native','--os-tokens','off','--os-cell','pi05_spatial_cache','--os-root','/dev/shm/offline_search_store',
       '--os-log-dir',str(out),'--os-tag','hook','--os-judge','guard_only']
cs,remaining=cost.parse_flags(flags);opts,rest=plugin.parse_cli(remaining);assert not rest
cost.validate_method(opts,cs.os_stage1_mode)
cost.install_startup_hook(plugin,stage1_mode=cs.os_stage1_mode,miss_steps=2)
rt=plugin.install(opts,model='pi05')
row=json.loads(rt.dec_path.read_text().splitlines()[-1])
assert row['stage1_mode']=='wrist_only' and row['miss_steps']==2 and row['prefix_packing'] is False
(HERE/'results/startup_check.json').write_text(json.dumps({'passed':True,'row':row},indent=2))
print('PASS: real startup row wrist_only, miss_steps=2, prefix_packing=false')
