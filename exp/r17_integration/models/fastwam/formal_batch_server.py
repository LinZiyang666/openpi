"""Local/H100 batched resident entry; P1 factory/S07, P6 loader/receipts.

Fast-WAM LIBERO/MT50 inference, cache bundles and opt-in library capture.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import threading
import time


def campaign_prompts(adapter,campaign):
    """Every wire prompt this campaign can send (prompt_template wraps the task language)."""
    benchmark=json.loads((Path(campaign['assets'])/'benchmark.json').read_text())
    from .text_table import libero_language
    if campaign.get('domain')=='metaworld':
        return sorted({adapter.wire.prompt_template.format(task=row[1]) for row in benchmark['tasks']})
    return sorted({adapter.wire.prompt_template.format(task=libero_language(n)) for n in benchmark['ordered_tasks']
                   if n.split('/')[0] in campaign['suites']})


def table_text(adapter,campaign):
    """Serve prompts from the precomputed table; the model was loaded without T5."""
    from .text_table import load,install
    if adapter.model.text_encoder is not None:
        raise ValueError('fastwam_text_table requires a model loaded without its text encoder')
    table=load(Path(campaign['assets'])/campaign['fastwam_text_table'],adapter.model.device)
    prompts=campaign_prompts(adapter,campaign)
    missing=[p for p in prompts if p not in table]
    if missing:
        raise ValueError(f'text table lacks {len(missing)} campaign prompts, e.g. {missing[0]!r}')
    install(adapter.model,table)
    return prompts


def cache_text(adapter,campaign):
    """Encode each prompt of the campaign once (batch 1), then move T5 off the GPU.

    Frees ~11 GB so two replicas fit on the 48 GB 4090 (dev lane only).
    """
    import torch
    benchmark=json.loads((Path(campaign['assets'])/'benchmark.json').read_text())
    names=benchmark['ordered_tasks']
    def language(name):  # libero.benchmark.grab_language_from_filename
        task=name.split('/',1)[1]
        if task[0].isupper():
            task=task[task.find('SCENE')+(8 if 'SCENE10' in task else 7):]
        return ' '.join(task.split('_'))
    model=adapter.model
    # Same string the wire policy sends to T5 (prompt_template wraps the task language).
    if campaign.get('domain')=='metaworld':
        prompts=sorted({adapter.wire.prompt_template.format(task=row[1]) for row in benchmark['tasks']})
    else:
        prompts=sorted({adapter.wire.prompt_template.format(task=language(n)) for n in names if n.split('/')[0] in campaign['suites']})
    with torch.inference_mode():
        table={p:model.encode_prompt([p]) for p in prompts}
    model.text_encoder.to('cpu')
    torch.cuda.empty_cache()
    def encode_prompt(prompt):
        rows=[table[p] for p in ([prompt] if isinstance(prompt,str) else prompt)]
        return torch.cat([r[0] for r in rows]),torch.cat([r[1] for r in rows])
    model.encode_prompt=encode_prompt
    return prompts


def serve(campaign,port,directory):
    if campaign['model']!='fastwam' or campaign['domain'] not in ('libero','metaworld'):
        raise ValueError('batched resident supports Fast-WAM LIBERO or MetaWorld')
    config_doc=campaign.get('fastwam_batching')
    if not config_doc or config_doc['max_batch']<=1:
        pass
    from exp.r17_integration.common.config import load_cache_config,build_shared_storage
    from exp.r17_integration.common.serving import make_connection_factory,make_bundle_validator
    from exp.r17_integration.common.websocket_policy_server import WebsocketPolicyServer
    from exp.r17_integration.qualification.model_runtime import load
    from exp.r17_integration.ops.formal_process import write_json
    from .batch_runtime import create_runtime,BatchConfig
    from .batcher import FastWAMBatchedAdapter
    from .serve import serve_metadata
    from .formal_batching import validate_runtime
    config=BatchConfig(config_doc['max_batch'],config_doc['batch_wait_ms'],config_doc['numerics_policy'])
    if config.metadata()!=config_doc:
        pass
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    # Same closed-run transfer contract as the ordinary resident entry.
    if not (directory.parent/'manifest.json').exists():
        write_json(directory.parent/'manifest.json',campaign)
    started=time.monotonic()
    treatment = validate_runtime(campaign) if campaign.get('fastwam_batch_treatment') else None
    if treatment:
        for stage in (1,2,3):
            if os.environ.get(f'BATCHING_STAGE{stage}_WORKERS','1') != '1':
                raise ValueError('formal batch treatment requires one worker per stage')
        if os.environ.get('OPENPI_DISABLE_STAGE_STREAMS') == '1':
            raise ValueError('formal batch treatment requires stage streams')
    import torch
    if treatment:
        free,total=torch.cuda.mem_get_info(campaign['device'])
        budget=campaign['memory_budget_gib']*2**30/campaign['replicas']
        if free < budget:
            raise ValueError('insufficient free VRAM; existing residents are untouched')
        torch.cuda.set_per_process_memory_fraction(budget/total,campaign['device'])
        if torch.cuda.get_device_name(campaign['device']) != treatment['gpu_family']:
            pass
    adapter,original_provider=load(dict(campaign,benchmark_spec=str(Path(campaign['assets'])/'benchmark.json')),campaign['device'])
    if campaign.get('fastwam_text_table'):
        cached_prompts=table_text(adapter,campaign)
    else:
        cached_prompts=cache_text(adapter,campaign) if campaign.get('fastwam_text_cache') else None
    total=torch.cuda.get_device_properties(campaign['device']).total_memory
    torch.cuda.set_per_process_memory_fraction(campaign['memory_budget_gib']*2**30/campaign['replicas']/total,campaign['device'])
    qualification = None
    if campaign.get('batch_qualification_observations'):
        from .formal_qualification import qualify
        qualification=qualify(adapter,campaign,directory)
    runtime=create_runtime(adapter,config.max_batch,config.batch_wait_ms,directory/'batching.jsonl')
    captures=None
    try:
        provider=original_provider
        cache_config=load_cache_config(Path(campaign['assets'])/'full.yaml')
        if cache_config.enabled or cache_config.checkpoints or cache_config.recipe.enabled:
            raise ValueError('formal batch campaign requires cache/checkpoints/recipe disabled')
        storage=build_shared_storage(cache_config)
        capture_kwargs={}
        if campaign.get('library'):
            from exp.r17_integration.debug.library_capture import campaign_capture_factory
            from exp.r17_integration.common.contracts import CaptureSpec
            captures=campaign_capture_factory(campaign,directory/'capture')
            capture_kwargs=dict(capture_sink_factory=captures,
                                capture_spec=CaptureSpec(enabled=True,snapshots=False))
        # Preserve the sealed provider/YAML identity for worker hot-switches.
        # The existing factory accepts a connection-local adapter constructor.
        base=runtime.wrap_connection_factory(make_connection_factory(provider,cache_config,storage,
            adapter_factory=lambda:FastWAMBatchedAdapter(provider.load_adapter(),runtime.core), **capture_kwargs))
        evidence_lock=threading.Lock()
        def factory(*args,**kwargs):
            core=base(*args,**kwargs)
            original=core.infer
            def infer(obs):
                begin=time.monotonic()
                if captures is not None:
                    core.capture_sink.bind_observation(obs)
                result=original(obs)
                elapsed=time.monotonic()-begin
                hit=result['__hit_meta__']
                row=dict(replica_port=port,task_uid=core._episode.get('task_uid'),task_name=core._episode.get('task_name'),
                    decision_seq=hit['decision_seq'],source=hit['source'],nfe=hit['actual_nfe'],elapsed_seconds=elapsed,
                    service_seconds=elapsed,queue_seconds=None,time=time.time(),batching=hit['batching'],
                    timing_scope='end_to_end_including_stage_queue_wait; service_seconds_is_not_kernel_time')
                if treatment:
                    row['fastwam_batch_treatment_identity'] = campaign['fastwam_batch_treatment_identity']
                with evidence_lock,(directory/'serving.jsonl').open('a') as handle:
                    handle.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
                return result
            core.infer=infer
            return core
        metadata=serve_metadata(adapter)
        active=provider.sampler_for_steps(cache_config.adapter_binding['num_steps'])
        metadata.update(formal_pure_inference=True,cache_enabled=False,debug_capture=False,cuda_graph=False,
            resident_execution='S07 cross-connection native batches; one worker per stage',replica_port=port,
            execution_sampler=asdict(active),execution_commit_horizon=cache_config.adapter_binding['commit_horizon'])
        if treatment:
            metadata.update(fastwam_batch_treatment_identity=campaign['fastwam_batch_treatment_identity'],
                            gpu_family=torch.cuda.get_device_name(campaign['device']))
        if qualification:
            metadata['batching_self_check'] = qualification
        if cached_prompts:
            metadata['text_cache']=(dict(prompts=cached_prompts,text_encoder_device='none',table=campaign['fastwam_text_table'])
                if campaign.get('fastwam_text_table') else dict(prompts=cached_prompts,text_encoder_device='cpu'))
        if captures is not None:
            metadata.update(formal_pure_inference=False,library_teacher=True,debug_capture=True,
                            capture_profile='B-full-native-library-allowlist; received_wire_uint8')
        server=WebsocketPolicyServer(factory(),host='0.0.0.0',port=port,metadata=metadata,concurrent=True,
            connection_policy_factory=factory,bundle_validator=make_bundle_validator(provider),
            ready_callback=lambda:write_json(directory/'ready.json',dict(metadata,pid=os.getpid(),
                elapsed_load_seconds=time.monotonic()-started)))
        signal.signal(signal.SIGTERM,lambda *unused:server.request_stop())
        server.serve_forever(stop_on_request=True)
    finally:
        runtime.close()
        if captures is not None:
            write_json(directory/'CAPTURE_CLOSED.json',captures.close())
        write_json(directory/'closed.json',dict(status='CLOSED',pid=os.getpid(),
            elapsed_seconds=time.monotonic()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,batching=config.metadata()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',required=True)
    p.add_argument('--port',required=True,type=int)
    p.add_argument('--directory',required=True)
    a=p.parse_args()
    serve(json.loads(Path(a.campaign).read_text()),a.port,a.directory)


if __name__=='__main__':
    main()
