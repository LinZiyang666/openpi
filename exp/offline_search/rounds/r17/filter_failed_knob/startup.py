"""Read-only frozen model identity probes; no staging, serving or CUDA.

On remote servers, inspect the source assets. Their descriptor bytes must equal
the new local campaign's assets; changed retrieval paths are checked locally.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import inspect
import textwrap
import ast
import base64
import zlib

from .common import HERE, read, dump, sha


def descriptor_code():
    """Import the existing probe's descriptor half, before its Cache-only tests.

    All strict model loader/content checks remain. Compare the resulting spec
    against sealed metadata AND actual source ready.json in the caller.
    """
    tree=ast.parse((HERE.parent/'filter_failed/cpu_startup.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='probe')
    block=next(n for n in fn.body if isinstance(n,ast.Try))
    kept=[]
    for stmt in block.body:
        kept.append(stmt)
        if isinstance(stmt,ast.Assign) and any(isinstance(t,ast.Subscript) and isinstance(t.slice,ast.Constant) and t.slice.value=='spec_equal' for t in stmt.targets):break
    else:raise ValueError('Existing descriptor probe boundary changed')
    block.body=kept+ast.parse('result["status"]="DESCRIPTOR_PASS"').body
    class NativeContent(ast.NodeTransformer):
        def visit_Assert(self,node):
            if isinstance(node.msg,ast.Constant) and node.msg.value=='Fast-WAM loader content seal differs':
                return ast.parse('native_fastwam_content(descriptor,actual,result)').body[0]
            return node
    tree=NativeContent().visit(tree)
    helper=ast.parse('''
def native_fastwam_content(descriptor,actual,result):
    import ast,inspect
    from exp.r17_integration.models.fastwam.loader import load_adapter
    tree=ast.parse(inspect.getsource(load_adapter))
    checks=[n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='actual != spec.expected_digests']
    if not checks:raise ValueError('Inspect frozen Fast-WAM content check')
    native_raises=any(isinstance(n,ast.Raise) for statement in checks[0].body for n in ast.walk(statement))
    result['native_content_check']='raise' if native_raises else 'record_only'
    result['content_seal_equal']=actual==descriptor.expected_digests
    result['content_differences']={k:{'expected':descriptor.expected_digests.get(k),'actual':actual.get(k)} for k in actual.keys()|descriptor.expected_digests.keys() if actual.get(k)!=descriptor.expected_digests.get(k)}
    if native_raises and not result['content_seal_equal']:raise ValueError('Frozen Fast-WAM loader content check failed')
''').body
    tree.body=helper+tree.body
    return ast.unparse(ast.fix_missing_locations(tree))


def check(job, *, virtual_assets=False, installed_source=None):
    from exp.r17_integration.ops import formal_eval as f
    root=Path(job['run']);c=read(root/'campaign.json');source=Path(job['source'])
    names=('load.json','model_spec.json','model_manifest.json','full.yaml','benchmark.json','execution_sampler.json')
    equality={n:sha(root/'assets'/n)==sha(source/'assets'/n) for n in names if (source/'assets'/n).exists()}
    assert all(equality.values())
    if installed_source is not None:
        installed_source=Path(installed_source)
        assert all(sha(source/'assets'/n)==sha(installed_source/'assets'/n)
                   for n in names if (source/'assets'/n).exists())
        old,new=read(source/'campaign.json'),read(installed_source/'campaign.json')
        for key in ('runtime_identity','runtime_archive','replicas','fastwam_batching','mps'):
            assert old.get(key)==new.get(key),(key,old.get(key),new.get(key))
        original_inputs=read(source/'assets/cache_inputs/manifest.json')['files']
        installed_inputs=read(installed_source/'assets/cache_inputs/manifest.json')['files']
        assert all(installed_inputs.get(n)==pin for n,pin in original_inputs.items())
        source=installed_source
    local=c['server_node']=='local'
    # CPU loader reads exactly the descriptors copied into this new campaign.
    actual=dict(c)
    if not local:actual.update(assets=str(source/'assets'),local_assets=str(source/'assets'))
    overrides={}
    if virtual_assets and not local:
        # Probe new grids on installed, byte-identical source inputs without
        # staging or writing on H100. Only YAML paths and the group list move.
        seal=read(root/'assets/CACHE_STUDY.json')
        overrides[str(source/'assets/CACHE_STUDY.json')]=json.dumps(seal)
        for row in seal['groups']:
            overrides[str(source/'assets'/row['file'])]=(root/'assets'/row['file']).read_text().replace(
                str(c['assets']),str(source/'assets')).replace(str(root/'assets'),str(source/'assets'))
    body=descriptor_code()
    from exp.offline_search.rounds.r17.filter_failed.build import register_metadata
    payload=(root/'assets/ffk_bootstrap.py').read_text()
    if "ffk_runtime=dict(__name__='__main__')" not in payload:
        raise ValueError('CPU probe requires the frozen bootstrap entry marker')
    payload=payload.replace("ffk_runtime=dict(__name__='__main__')","ffk_runtime=dict(__name__='ffk1015_cpu_runtime')",1)
    extra=('\n'+payload+'\n'
           'ffk_runtime["install"]('+repr(c['filter_failed_knob']['selection'])+')\n'
           '_metadata_samplers={}\nread=lambda p:result["actual_spec"] if str(p)=='+repr(str(source/'assets/model_spec.json'))+' else json.loads(Path(p).read_text())\n'+inspect.getsource(register_metadata)+'\n'
           'import yaml\nfrom types import SimpleNamespace\nfrom exp.r17_integration.common.config import config_from_dict\n'
           'from exp.r17_integration.common.provider import get_provider\n'
           'from exp.r17_integration.common.recipe import make_recipe_factory\n'
           'if '+repr(c['model'])+' == "dp" and '+repr(bool(c.get('dp_batching')))+':\n'
           '    from exp.r17_integration.models.dp.batch_runtime import BatchConfig\n'
           '    from exp.r17_integration.models.dp.formal_batch_server import cache_study_admission\n'
           '    full_doc=yaml.safe_load(Path('+repr(actual['assets']+'/full.yaml')+').read_text())\n'
           '    register_metadata('+repr(str(source))+',full_doc)\n'
           '    full_cfg=config_from_dict(full_doc)\n'
           '    b='+repr(c.get('dp_batching'))+'\n'
           '    admission_c='+repr(actual)+'\n'
           '    if admission_c.get("ffk_dp_capacity_bridge"):admission_c["replicas"]=2\n'
           '    cache_study_admission(admission_c,full_cfg,BatchConfig(b["max_batch"],b["batch_wait_ms"],b["numerics_policy"]),get_provider(full_cfg.adapter_binding["provider"]))\n'
           '    result["native_dp_startup_admission_checked"]=True\n'
           'full_checks=[]\n'
           'for row in read('+repr(actual['assets']+'/CACHE_STUDY.json')+')["groups"]:\n'
           '    doc=yaml.safe_load(Path('+repr(actual['assets'])+',row["file"]).read_text())\n'
           '    register_metadata('+repr(str(source))+',doc)\n'
           '    cfg=config_from_dict(doc)\n'
           '    provider=get_provider(cfg.adapter_binding["provider"])\n'
           '    module=importlib.import_module('+repr('exp.r17_integration.models.'+c['model']+'.provider')+')\n'
           '    configure=getattr(module,"configure_library_provider",None)\n'
           '    if configure:provider=configure(provider)\n'
           '    recipe=make_recipe_factory(provider)(cfg,None)\n'
           '    full_checks.append(dict(group=row["group"],size=row["library_size"],suite=row["suite"],corrector=recipe.base.use_corrector,stall=recipe.model is not None))\n'
           'result["full_system_startup_checks"]=full_checks\n'
           'result["source_ready_specs"]=[]\n'
           'for p in Path('+repr(str(source/'servers'))+').glob("*/ready.json"):\n'
           '    r=read(p)\n    spec=r.get("r17_model_spec",r.get("spec"))\n'
           '    if spec:result["source_ready_specs"].append(spec)\n')
    virtual='/__ffk_cpu_campaign__.json'
    code=(body+'\nfrom unittest.mock import patch\n'
          'native=Path.read_text\n'
          'def read_text(self,*a,**kw):\n'
          '    if str(self)=='+repr(virtual)+':return '+repr(json.dumps(actual))+'\n'
          '    overrides='+repr(overrides)+'\n'
          '    return overrides[str(self)] if str(self) in overrides else native(self,*a,**kw)\n'
          'with patch.object(Path,"read_text",read_text):\n'
          '    result=probe('+repr(virtual)+')\ntry:\n'+textwrap.indent(extra,'    ')+
          '\nexcept Exception:\n    import traceback\n    result["full_system_startup_error"]=traceback.format_exc()\n'+
          'print(json.dumps(result))\n')
    # A complete test grid carries many repeated YAMLs. Compress the CPU-only
    # probe to fit the OS argv limit without staging files on the server.
    packed=base64.b64encode(zlib.compress(code.encode(),9)).decode()
    code='import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))'
    env=dict(f.server_env(c,c['runtime']),CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',PYTHONDONTWRITEBYTECODE='1')
    argv=['env',*[k+'='+str(v) for k,v in env.items()],c['server_python'],'-c',code]
    if local:
        output=subprocess.run(argv,cwd=c['runtime'],capture_output=True,text=True,check=True).stdout
    else:
        output=f.remote('h100',argv,seconds=600,replay_safe=True)
    result=json.loads(output.splitlines()[-1])
    # The reused pure-Cache probe stops at its group-c assertion. Identity and
    # native startup admission have already completed before that assertion.
    identity=bool(result.get('spec_equal') and result.get('pinned_packages_equal'))
    expected_stop=False
    result.update(identity_passes=identity,descriptor_bytes_equal=equality,
                  source_assets_read_only=not local,pure_cache_probe_expected_stop=expected_stop,
                  virtual_grid_assets=bool(overrides))
    ready_specs=list(result.get('source_ready_specs',[]))
    ready=[s['checkpoint_identity'] for s in ready_specs]
    ready_source=source
    if not list(source.glob('servers/*/ready.json')) and source.name=='knobdevdpred2':
        actual_source=source.with_name('knobdevdpred3')
        if read(actual_source/'campaign.json')['runtime_identity']==c['runtime_identity'] and all(sha(actual_source/'assets'/n)==sha(source/'assets'/n) for n in names if (source/'assets'/n).exists()):ready_source=actual_source
    result['source_serving_run']=str(ready_source)
    for p in ready_source.glob('servers/*/ready.json'):
        r=read(p);s=r.get('r17_model_spec',r.get('spec',{}));ready.append(s.get('checkpoint_identity',r.get('model_identity')))
        if s:ready_specs.append(s)
    result['source_actual_identities']=sorted(set(ready))
    same_actual=bool(ready_specs) and all(s==result.get('actual_spec') for s in ready_specs)
    # A source may have stale sealed descriptive metadata (pimk1_fuld records
    # its updated code/preprocess identity in ready.json). Preserve both source
    # asset bytes and its actual frozen serving identity; never repair the seal.
    descriptor_pass=bool(result.get('pinned_packages_equal') and result.get('identity_path'))
    recorded_drift=same_actual and not result.get('spec_equal')
    result['equals_source_actual']=same_actual
    result['identity_passes']=descriptor_pass and (identity or recorded_drift)
    result['sealed_metadata_drift_matches_source_ready']=recorded_drift
    result['status']='PASS' if result['identity_passes'] and result['status']=='DESCRIPTOR_PASS' and not result.get('full_system_startup_error') else 'FAIL'
    dump(HERE/'startup_checks'/(job['name']+'.json'),result)
    print(json.dumps(dict(job=job['name'],passes=result['status']=='PASS',equals_source_actual=result['equals_source_actual'],error=result.get('error'))),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',action='append')
    p.add_argument('--run',type=Path,action='append',help='New official-A/raise definition to check')
    args=p.parse_args()
    catalog=read(HERE/'CAMPAIGNS.json')['jobs']
    jobs=[read(root/'FFK_DEFINITION.json') for root in args.run] if args.run else catalog
    for job in jobs:
        if not args.job or job['name'] in args.job:check(job)
    active={j['name'] for j in catalog+jobs}
    results=[read(p) for p in sorted((HERE/'startup_checks').glob('*.json')) if p.stem in active]
    dump(HERE/'STARTUP.json',dict(jobs=results,cpu_only=True))


if __name__=='__main__':main()
