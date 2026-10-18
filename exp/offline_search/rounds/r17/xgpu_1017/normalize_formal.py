"""CPU: bind formal outcomes and optional socket traces to one sealed paired PLAN.

No failed/missing episode is silently removed. This does not launch or modify a
formal run. Policy seed means the pinned base seed; actual per-decision noise
seeds remain in the trace. Results from different model/suite/arms stay separate.
"""
import argparse
import collections
import json
from pathlib import Path
from artifacts import digest, sha, write_json


def lines(path):
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():yield number, json.loads(line)


def index_plan(plan, benchmark):
    body=dict(plan); identity=body.pop('identity')
    if digest(body)!=identity:raise ValueError('PLAN identity changed')
    tasks={}
    for task in plan['tasks']:
        uid=task['task_uid']
        if uid in tasks:raise ValueError('duplicate planned UID')
        extra=task.get('extra',{})
        seed=extra.get('seed',benchmark.get('seed'))
        if type(seed) is not int:raise ValueError('explicit benchmark/task seed required')
        if extra.get('benchmark_protocol')!=plan['benchmark_protocol']:
            raise ValueError('task benchmark protocol differs')
        # Frozen libero_bridge.checked_setup fixes Cosmos environment RNG0,
        # independently of its policy diffusion base seed195.
        env_seed=0 if benchmark.get('protocol_id','').startswith('libero_cosmos_official_') else seed
        key=dict(model=plan['model'],suite=plan['suite'],arm=plan['arm'],
            task_id=task['task_id'],init_idx=task['orig_init_state_idx'],
            env_seed=env_seed,policy_seed=seed)
        tasks[uid]=(task,key)
    if not tasks:raise ValueError('empty PLAN')
    return tasks


def outcomes(tasks, attempts):
    accepted={}; rejected=collections.Counter()
    for row in attempts:
        uid=row.get('task_uid')
        if uid not in tasks:raise ValueError('outcome outside paired PLAN: '+str(uid))
        if row.get('accepted') is not True:
            rejected[str(uid)]+=1;continue
        if uid in accepted:raise ValueError('multiple accepted outcomes for '+uid)
        if row.get('error') not in (None,''):
            raise ValueError('accepted infrastructure/runner error needs resolution: '+uid)
        if type(row.get('success')) is not bool or type(row.get('attempt')) is not int:
            raise ValueError('accepted binary success and attempt required')
        accepted[uid]={**tasks[uid][1], 'task_uid':uid, 'accepted':True,
            'success':row['success'],'attempt':row['attempt'],'n_steps':row.get('n_steps'),
            'duration_s':row.get('duration_s')}
    missing=set(tasks)-set(accepted)
    if missing:raise ValueError(f'{len(missing)} planned outcomes missing; cannot compute paired SR')
    return accepted,dict(rejected)


def trace_index(tasks):
    result={}
    for uid,(task,key) in tasks.items():
        extra=task.get('extra',{})
        signature=(key['task_id'],key['init_idx'],extra.get('bundle_generation',task['bundle_id']))
        if signature in result:raise ValueError('ambiguous trace-to-task mapping')
        result[signature]=uid
    return result


def validate_manifest(tasks, manifest):
    """Only endpoint binding may differ from the once-generated scientific plan."""
    def scientific(task):
        return {k:v for k,v in task.items() if k not in ('server_host','server_port')}
    actual={t['task_uid']:scientific(t) for t in manifest['tasks']}
    if len(actual)!=len(manifest['tasks']) or actual!={u:scientific(t) for u,(t,_) in tasks.items()}:
        raise ValueError('formal driver task population differs from sealed PLAN')


def trace_row(row, plan, tasks, by_trace, accepted):
    req=row['request']
    if req.get('benchmark_protocol')!=plan['benchmark_protocol']:
        raise ValueError('trace benchmark mismatch')
    signature=(req['task_id'],req['original_init_idx'],req['bundle_generation'])
    if signature not in by_trace:raise ValueError('trace outside paired PLAN')
    uid=by_trace[signature]; outcome=accepted[uid]; task,key=tasks[uid]
    if req.get('seed')!=key['policy_seed']:raise ValueError('trace base seed mismatch')
    if req.get('task_name')!=task['extra']['task_name']:raise ValueError('trace task-name mismatch')
    if req.get('attempt')!=outcome['attempt']:return None
    if type(row.get('env_step')) is not int or row['env_step']<0:
        raise ValueError('physical execution clock required')
    if type(row.get('wire_delivery')) is not int or row['wire_delivery']<0:
        raise ValueError('wire delivery index required')
    return {**row,**key,'task_uid':uid,'attempt':outcome['attempt']}


def normalize(plan_path, attempts_path, driver_receipt, output, bundle, trace_paths=(), execution_paths=()):
    from concurrent_server import checked_pin
    plan_path=Path(plan_path); plan=json.loads(plan_path.read_text())
    benchmark_path=plan_path.parent/'assets/benchmark.json'
    benchmark=json.loads(benchmark_path.read_text())
    pin=checked_pin(bundle)
    if plan['pin']!=pin['identity'] or sha(benchmark_path)!=pin['files']['benchmark.json']:
        raise ValueError('benchmark/PLAN is not bound to the pinned serving bundle')
    receipt=json.loads(Path(driver_receipt).read_text())
    if receipt.get('status')!='PASS' or receipt.get('all_done') is not True:
        raise ValueError('complete PASS formal driver receipt required')
    tasks=index_plan(plan,benchmark)
    manifest_path=Path(driver_receipt).parent/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    validate_manifest(tasks,manifest)
    if manifest.get('runtime_identity')!=pin['runtime_identity']:
        raise ValueError('driver manifest runtime differs from pinned serving release')
    accepted,rejected=outcomes(tasks,(r for _,r in lines(attempts_path)))
    binding=dict(plan_identity=plan['identity'],serving_pin=pin['identity'],
        benchmark_protocol=plan['benchmark_protocol'],runtime_identity=pin['runtime_identity'])
    for outcome in accepted.values():outcome.update(binding)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with (output/'outcomes.jsonl').open('x') as f:
        for uid in sorted(accepted):f.write(json.dumps(accepted[uid],sort_keys=True)+'\n')
    report=dict(schema='xgpu.formal_normalization.v1',plan_identity=plan['identity'],
        plan_sha256=sha(plan_path),benchmark_sha256=sha(benchmark_path),
        attempts_sha256=sha(attempts_path),driver_receipt_sha256=sha(driver_receipt),
        driver_manifest_sha256=sha(manifest_path),
        outcomes=len(accepted),rejected_attempts=rejected,trace_rows=0,
        discarded_unaccepted_attempt_trace_rows=0,trace_sources=[],server_clock_mismatches=0,
        seed_scope='frozen worker contract: Cosmos environment seed0, policy base195; other selected protocols use benchmark/task base seed; actual noise seeds stay in traces',
        SR_evidence='normalized outcomes only; paired comparison is still required')
    by_trace=trace_index(tasks); seen=set(); coverage=collections.defaultdict(set);returned={}
    with (output/'traces.jsonl').open('x') as f:
        for path in trace_paths:
            for number,row in lines(path):
                if row.get('schema')=='xgpu.client_execution.v1':
                    raise ValueError('execution events belong in --execution-trace, not --trace')
                result=trace_row(row,plan,tasks,by_trace,accepted)
                if result is None:
                    report['discarded_unaccepted_attempt_trace_rows']+=1;continue
                for field,value in binding.items():
                    if field in result and result[field]!=value:raise ValueError('trace scientific binding differs: '+field)
                result.update(binding)
                report['server_clock_mismatches']+=result.get('server_clock_agrees')is False
                identity=(result['task_uid'],result['env_step'])
                if identity in seen:raise ValueError('duplicate physical trace step')
                seen.add(identity);coverage[result['task_uid']].add(result['wire_delivery'])
                query=(result['task_uid'],result['wire_delivery'])
                if query in returned:raise ValueError('duplicate wire delivery in trace')
                returned[query]={k:result[k]for k in ('env_step','actions','request')}
                f.write(json.dumps(result,allow_nan=False,sort_keys=True)+'\n')
                report['trace_rows']+=1
            report['trace_sources'].append(dict(path=str(path),sha256=sha(path)))
    from executed_controls import validate_ack
    acknowledged=set();report.update(execution_rows=0,execution_sources=[])
    with (output/'executions.jsonl').open('x') as f:
        for path in execution_paths:
            for number,row in lines(path):
                result=trace_row(row,plan,tasks,by_trace,accepted)
                if result is None:
                    report['discarded_unaccepted_attempt_trace_rows']+=1;continue
                for field,value in binding.items():
                    if field in result and result[field]!=value:raise ValueError('execution scientific binding differs: '+field)
                result.update(binding)
                query=(result['task_uid'],result['wire_delivery'])
                if query in acknowledged:raise ValueError('duplicate execution ACK')
                if query not in returned:raise ValueError('execution ACK has no corresponding query trace')
                validate_ack(result,returned[query]);acknowledged.add(query)
                f.write(json.dumps(result,allow_nan=False,sort_keys=True)+'\n');report['execution_rows']+=1
            report['execution_sources'].append(dict(path=str(path),sha256=sha(path)))
    report['queries_without_execution_ack']=len(set(returned)-acknowledged)
    report['episodes_without_execution_ack']=sorted(set(tasks)-{key[0]for key in acknowledged})
    report['trace_coverage']={uid:dict(deliveries=len(v),starts_at_zero=min(v)==0,
        continuous=v==set(range(max(v)+1))) for uid,v in coverage.items()}
    report['episodes_without_trace']=sorted(set(tasks)-set(coverage))
    write_json(output/'NORMALIZATION.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--attempts',required=True)
    p.add_argument('--bundle',required=True)
    p.add_argument('--driver-receipt',required=True);p.add_argument('--output',required=True)
    p.add_argument('--trace',action='append',default=[])
    p.add_argument('--execution-trace',action='append',default=[])
    a=p.parse_args()
    report=normalize(a.plan,a.attempts,a.driver_receipt,a.output,a.bundle,a.trace,a.execution_trace)
    print(json.dumps({k:report[k] for k in ('outcomes','trace_rows','episodes_without_trace')}))
