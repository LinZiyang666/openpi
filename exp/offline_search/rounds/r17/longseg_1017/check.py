"""Small CPU startup/plan proofs, reusing the sealed strict loader receipts.

No weight/fit reread is needed: their unchanged identities were CPU-probed
already. Run the new pinned bootstrap in each actual frozen interpreter and
check its package pins, then bind it to that strict descriptor proof.

With --local-only, remote interpreter probes are retained as prior evidence,
explicitly marked PASS_LOCAL; local interpreter and plan checks still run.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import inspect
from pathlib import Path
import subprocess
import zlib

from .audit import HERE,EVAL,io_ready
from .io_budget import dump
from exp.offline_search.rounds.r17.filter_failed_knob.common import read,sha

def proofs():
    base=HERE.parent
    return [(p,read(p)) for directory in (base/'filter_failed_knob/startup_checks',base/'filter_failed/startup_checks')
            for p in directory.glob('*.json') if read(p).get('status')=='PASS']

def check(job,evidence,*,local_only=False,output_dir=None):
    from exp.offline_search.rounds.r17.filter_failed_knob.build import cpu_plan
    from exp.r17_integration.ops import formal_eval as f
    root=Path(job['run']);c=read(root/'campaign.json');spec=read(root/'assets/model_spec.json')
    names=('load.json','model_spec.json','model_manifest.json')
    matches=[(p,r) for p,r in evidence if r.get('identity_path') and r.get('pinned_packages_equal') and
             r.get('assets') and all((Path(r['assets'])/n).exists() and sha(root/'assets'/n)==sha(Path(r['assets'])/n) for n in names)]
    if not matches:raise ValueError('No strict CPU loader evidence for exact model descriptor: '+job['name'])
    p,prior=matches[0];old=Path(prior['assets'])
    equality={name:sha(root/'assets'/name)==sha(old/name) for name in ('load.json','model_spec.json','model_manifest.json','benchmark.json','full.yaml')}
    # full.yaml and benchmark may intentionally differ by benchmark or pure
    # startup sampler. Model load/spec/manifest must match byte for byte.
    assert all(equality[n] for n in ('load.json','model_spec.json','model_manifest.json')),equality
    code=(root/'assets/ffk_bootstrap.py').read_text().replace(
        "ffk_runtime=dict(__name__='__main__')","ffk_runtime=dict(__name__='cpu_probe')",1)
    pins=read(root/'assets/model_manifest.json').get('dependency_versions') or prior['package_versions']
    code+='\nffk_runtime["install"]('+repr(c['filter_failed_knob']['selection'])+')\n'
    code+='\nfrom importlib.metadata import version\nimport json,sys\n'
    code+='pins='+repr(pins)+'\nversions={p:version(p) for p in pins}\nassert versions==pins,(versions,pins)\n'
    code+='from exp.r17_integration.common.serving import ConnectionPolicy\n'
    code+='assert "_decision_seq" in ConnectionPolicy._request.__code__.co_names\n'
    code+='assert "force_look" not in ConnectionPolicy.infer.__code__.co_names\n'
    if c.get('longseg_1017',{}).get('recalibration'):
        code+='\n_recal_evidence={}\nexec('+repr((root/'assets/ffk_evidence.py').read_text())+',_recal_evidence)\n_recal_evidence["install"]()\n'
        code+='from exp.r17_integration.ops import cache_study as _recal_study\nassert "task_namespace" in _recal_study.plan.__code__.co_consts\n'
    if c['model']=='dp' and c.get('dp_batching'):
        from exp.offline_search.rounds.r17.filter_failed.build import register_metadata
        code+='\nfrom pathlib import Path\nfrom types import SimpleNamespace\nfrom unittest.mock import patch\nimport yaml\n'
        virtual={str(root/'assets/benchmark.json'):(root/'assets/benchmark.json').read_text()}
        code+='native_read_text=Path.read_text\nvirtual='+repr(virtual)+'\n'
        code+='read=lambda p:'+repr(prior['actual_spec'])+'\n_metadata_samplers={}\n'+inspect.getsource(register_metadata)+'\n'
        code+='with patch.object(Path,"read_text",lambda p,*a,**kw:virtual[str(p)] if str(p) in virtual else native_read_text(p,*a,**kw)):\n'
        code+='    doc=yaml.safe_load('+repr((root/'assets/full.yaml').read_text())+')\n'
        code+='    register_metadata('+repr(str(root))+',doc)\n'
        code+='    from exp.r17_integration.common.config import config_from_dict\n    cfg=config_from_dict(doc)\n'
        code+='from exp.r17_integration.common.provider import get_provider\nfrom exp.r17_integration.models.dp import formal_batch_server as server\nfrom exp.r17_integration.models.dp.batch_runtime import BatchConfig\n'
        code+='from dataclasses import replace\nfrom exp.mw_finetune.dp.checkpoint import read_metadata\nfrom exp.r17_integration.models.dp.sampler import SamplerProfile\nfrom exp.r17_integration.models.dp.admission import admission_validators\n'
        code+='checkpoint_config=read_metadata('+repr(read(root/'assets/load.json')['checkpoint'])+')["config"]["model"]\n'
        code+='provider=get_provider(cfg.adapter_binding["provider"])\nnative_sampler=lambda k:SamplerProfile("auto",k,0.).to_spec(checkpoint_config)\nbinding_validator,config_validator=admission_validators(provider.model_spec,native_sampler)\nprovider=replace(provider,sampler_for_steps=native_sampler,binding_validator=binding_validator,config_validator=config_validator)\n'
        if c.get('ffk_dp_capacity_bridge'):
            code+='from exp.offline_search.rounds.r17.filter_failed.server_cli import bridge\nbridge(server)\n'
        b=c['dp_batching']
        code+='server.cache_study_admission('+repr(c)+',cfg,BatchConfig('+repr(b['max_batch'])+','+repr(b['batch_wait_ms'])+','+repr(b['numerics_policy'])+'),provider)\n'
    code+='print(json.dumps(dict(packages=versions,python=sys.executable,bootstrap_imported=True,cuda_initialized=(sys.modules["torch"].cuda.is_initialized() if "torch" in sys.modules else False))))\n'
    packed=base64.b64encode(zlib.compress(code.encode())).decode()
    command='import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))'
    env=dict(f.server_env(c,c['runtime']),CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',PYTHONDONTWRITEBYTECODE='1')
    argv=['env',*[k+'='+str(v) for k,v in env.items()],c['server_python'],'-c',command]
    cached_probe=None
    if local_only and c['server_node']!='local':
        # A CPU-only coordinator audit must not silently use tether. Keep the
        # earlier remote receipt separate from a fresh interpreter result.
        receipt=read(HERE/'cpu'/f'{job["name"]}.json')
        assert receipt['status']=='PASS' and receipt['name']==job['name']
        cached_probe=receipt['fresh_frozen_interpreter_probe']
        assert cached_probe['packages']==pins and not cached_probe['cuda_initialized']
        assert sha(root/'assets/ffk_bootstrap.py')==c['ffk_bootstrap_sha256']
        compile(code,str(root/'assets/ffk_bootstrap.py'),'exec')
        fresh=None
    elif c['server_node']=='local':
        output=subprocess.run(argv,cwd=c['runtime'],capture_output=True,text=True)
        if output.returncode:raise RuntimeError(output.stderr[-3000:])
        output=output.stdout
    else:output=f.remote('h100',argv,seconds=180,replay_safe=True)
    if cached_probe is None:
        fresh=json.loads(output.splitlines()[-1]);assert not fresh['cuda_initialized']
    _,_,tasks,_=cpu_plan(c);uids=[t.task_uid for t in tasks]
    assert len(uids)==len(set(uids))==job['episodes']
    assert hashlib.sha256('\n'.join(uids).encode()).hexdigest()==job['task_uid_sha256']
    native_manifest=read(root/'assets/cache_inputs/manifest.json')
    # Every sealed numeric/view input is the same inode as the source copy.
    src=Path(job['source'])/'assets/cache_inputs'
    linked=all((root/'assets/cache_inputs'/n).samefile(src/n) for n in native_manifest['files'])
    assert linked
    old_proof=read(HERE.parent/'filter_failed_knob/formal_a_1016/short_names_1017/CPU_PROOF.json')['rows']
    gpu_hash=next(r['gpu_hash'] for r in old_proof if ('fa_dp' if c['server_node']=='local' else 'fa_co') in r['name'])
    pipe=str(root/'mps'/gpu_hash/'p');assert len(pipe.encode())<=85
    result=dict(name=job['name'],status='PASS_LOCAL' if cached_probe else 'PASS',native_strict_loader_proof=str(p),
        native_strict_loader_path=prior['identity_path'],descriptor_bytes_equal=equality,
        fresh_frozen_interpreter_probe=fresh,model_identity_equal=True,
        sealed_metadata_equal_to_actual=prior.get('spec_equal'),
        weights_and_fit_reread=False,immutable_inputs_same_inodes=True,
        planner_episodes=len(tasks),unique_task_uids=len(set(uids)),
        source_native_plan_equal=job.get('source_full_native_plan_equal'),mps_pipe=pipe,mps_pipe_bytes=len(pipe.encode()))
    if cached_probe:
        result.update(remote_interpreter_repeated=False,cached_remote_cpu_probe=cached_probe,
            cached_remote_receipt=str(HERE/'cpu'/f'{job["name"]}.json'),
            local_bootstrap_syntax_and_pin_checked=True)
    result['fresh_native_dp_startup_admission_checked']=bool(fresh and c['model']=='dp' and c.get('dp_batching'))
    if c.get('longseg_1017',{}).get('recalibration'):
        index=read(HERE/'RECALIBRATION.json');checked=[]
        for group in read(root/'assets/CACHE_STUDY.json')['groups']:
            if group['settings']['commit']<=5 or not group['settings'].get('gated') or not group.get('stall_digest'):continue
            q=next(q for q in index['Q3'].values() if q['digest']==group['stall_digest'])
            alias=root/'assets/cache_inputs'/group['stall_rel']
            assert (alias/'stall.pkl').samefile(Path(q['path'])/'stall.pkl')
            assert read(alias/'READY.json')['digest']==q['digest']
            assert q['new_parameters']['exec_steps']==q['new_parameters']['commit_controls']==1
            assert q['metric_identical']
            checked.append(q['digest'])
        result['decision_clock_recalibration']=dict(Q3_artifacts=sorted(set(checked)),
            new_artifact_inodes_verified=True,frozen_metric_identical=True,UID_namespace_preserved=True,
            frozen_driver_plan_overlay_imported=bool(fresh),level_ir='M/N_segment_heads',tolerance=.005)
    dump((output_dir or HERE/'cpu')/f'{job["name"]}.json',result);print(job['name'],result['status'],len(tasks),flush=True)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--name',action='append')
    p.add_argument('--local-only',action='store_true',help='Never contact remote nodes; verify local plans and retain remote CPU receipts as prior evidence')
    p.add_argument('--out',type=Path,default=HERE/'CPU_PROOF.json');a=p.parse_args()
    jobs=read(HERE/'CAMPAIGNS.json')['jobs']
    for u in read(HERE/'FORMAL_QUEUE.json')['units']:
        for j in u['jobs']:
            if j['name'] not in {v['name'] for v in jobs}:jobs.append(j)
    if a.name and set(a.name)-{j['name'] for j in jobs}:raise ValueError('Named campaign is not built yet')
    evidence=proofs();rows=[]
    for job in jobs:
        if a.name and job['name'] not in a.name:continue
        io_ready();rows.append(check(job,evidence,local_only=a.local_only,output_dir=a.out.parent/'cpu'))
    dump(a.out,dict(status='PASS_LOCAL' if any(r['status']=='PASS_LOCAL' for r in rows) else 'PASS',
        rows=rows,cpu_only=True,local_only=a.local_only,strict_loader_evidence_reused=True))
if __name__=='__main__':main()
