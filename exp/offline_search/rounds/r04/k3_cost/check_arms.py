"""Read-only artifact validation plus K4 generic YAML patch integration (local output)."""
import importlib.util,json,pathlib,pickle,sys,types
from exp.offline_search.closed_loop import plugin,stage_overrides
from openpi.cache.config import load_cache_config
from openpi.cache.interceptor import InferenceInterceptor
from openpi.cache.groot.interceptor import GrootCacheInterceptor
HERE=pathlib.Path(__file__).resolve().parent
REPO=HERE.parents[4]


def main():
    path=REPO/'exp/offline_search/closed_loop/ops/emit_arms.py'
    if 'yaml_patch' not in path.read_text():path=HERE.parent/'k4_eval/dev/emit_arms.py'
    spec=importlib.util.spec_from_file_location('k3_emit_checked',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    out=HERE/'results/emitted';mod.main(['--spec',str(HERE/'arms_r4.json'),'--run-root',str(out)])
    arms=json.loads((out/'arms.json').read_text());checks=[]
    for arm in arms:
        cfg=load_cache_config(arm['yaml']);assert cfg.miss.num_steps==2 and cfg.write_policy.type=='never'
        co,args=stage_overrides.parse_flags(arm['plugin_args'])
        opts,rest=plugin.parse_cli(['--os-method',arm['method'],'--os-kwargs',json.dumps(arm['kwargs']),
            '--os-cell',arm['cell'],'--os-log-dir',str(out),'--os-root','/dev/shm/offline_search_store',*args])
        assert not rest;stage_overrides.validate_method(opts,co.os_stage1_mode)
        plugin.load_method_class(arm['method'])
        if opts.os_fit_artifact:
            artifact=pathlib.Path(opts.os_fit_artifact)
            with artifact.open('rb') as f:blob=pickle.load(f)
            for key,want in [('spec',arm['method']),('kwargs',arm['kwargs']),('cell',arm['cell'])]:
                assert blob[key]==want,(arm['arm'],key,blob[key],want)
        checks.append({'arm':arm['arm'],'miss_steps':cfg.miss.num_steps,'stage1_mode':co.os_stage1_mode,
                       'artifact_validated':bool(opts.os_fit_artifact)})
    pi=object.__new__(InferenceInterceptor);pi._miss_num_steps=None;assert pi._miss_steps()==10
    pi._miss_num_steps=2;assert pi._miss_steps()==2
    gr=object.__new__(GrootCacheInterceptor);gr._runner=types.SimpleNamespace(_model=types.SimpleNamespace(action_head=types.SimpleNamespace(num_inference_timesteps=2)))
    assert gr._require_miss_steps(2)==2
    try:gr._require_miss_steps(8)
    except RuntimeError:pass
    else:raise AssertionError('GR00T K mismatch accepted')
    report={'emitter':str(path),'arms':len(checks),'yaml_and_artifacts':checks,'pi05_default_and_K2':True,'groot_match_and_mismatch':True}
    (HERE/'results/arm_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
