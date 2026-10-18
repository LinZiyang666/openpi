"""Exercise a real frozen cache/fit with synthetic CPU keys; no quality claim."""
import argparse
import dataclasses
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from artifacts import write_json
from replay import make_cache, cache_query, verify_runtime


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--campaign',required=True);p.add_argument('--cache-yaml',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    c=json.loads(Path(a.campaign).read_text());r=Path(c['runtime'])
    verify_runtime(r,c['runtime_identity'])
    sys.path[:0]=[str(r),str(r/'src'),str(r/'packages/openpi-client/src')]
    from exp.r17_integration.common.contracts import ActionStateLayout, FeatureSpec, SamplerSpec, Capability, Capabilities, ModelSpec, RequestContext
    from exp.r17_integration.qualification.benchmark_binding import benchmark_spec
    benchmark_spec(dict(c,benchmark_spec=c['assets']+'/benchmark.json'))
    d=json.loads(Path(c['assets']+'/model_spec.json').read_text())
    d['layout']=ActionStateLayout(**d['layout']);d['feature_spec']=FeatureSpec(**d['feature_spec'])
    d['default_sampler']=SamplerSpec(**d['default_sampler'])
    d['capabilities']=Capabilities({k:Capability(**v) for k,v in d['capabilities']['features'].items()})
    spec=ModelSpec(**d)
    import yaml
    from exp.r17_integration.common.provider import AdapterProvider, register_provider
    config_doc=yaml.safe_load(Path(a.cache_yaml).read_text())
    provider=AdapterProvider(config_doc['adapter_binding']['provider'],spec,lambda:None,
        lambda k:spec.default_sampler)
    from exp.r17_integration.models.dp.library import configure_library_provider
    provider=configure_library_provider(provider)
    register_provider(provider)
    recipe,seal=make_cache(provider,a.cache_yaml)
    fields={k:np.zeros(v,np.float32) for k,v in spec.feature_spec.fields.items()}
    state=np.zeros(len(spec.layout.valid_state_indices),np.float32)
    decisions=[]
    for i in range(3):
        req=RequestContext('cpu_fixture',spec.allowed_benchmark_protocols[0],
            'libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate',
            0,i,explicit_seed=7)
        history=dict(hist_has_vision=[True]*i,hist_hit=[1]*i,hist_state=[state.copy() for _ in range(i)],
                     prev_hit=True if i else None, executed_count=int(i>0))
        result=cache_query(recipe,dict(episode_start=i==0,history=history,
            executed_controls_since_last_query=5 if i else 0),req,fields,state)
        decisions.append({k:result[k] for k in ('hit','top1','corrector','stall','neighbour_margin','corrector_margin')})
    import torch
    if torch.cuda.is_initialized():raise RuntimeError('CPU check initialized CUDA')
    # Import the actual CPU policy dependency surface without loading weights.
    from exp.mw_finetune.dp import policy as _policy
    import importlib.metadata as md
    mapping=md.packages_distributions()
    used={name for module in sys.modules for name in mapping.get(module.split('.')[0],[])}
    dependencies={name:md.version(name) for name in sorted(used)}
    report=dict(status='PASS',synthetic=True,quality_claim=False,cuda_initialized=False,
                runtime_identity=c['runtime_identity'],cache_seal=seal,decisions=decisions,
                imported_distributions=dependencies)
    write_json(a.output,report);print(json.dumps(report))


if __name__=='__main__':main()
