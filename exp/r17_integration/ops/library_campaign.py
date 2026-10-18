"""Teacher libraries using the stock formal resident/conductor launcher.

No GPUs are started without --run or --repair. --finish uses a closed receipt on
CPU, useful when a fit job is run separately from collection. LIBERO smoke
uses two episodes; MT50 smoke uses two tasks x two seed8 initial states.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time
import uuid

from .formal_process import write_json
from .library_alignment import ROOT, SUITES, load_alignment

LIBRARY_ROOT = Path('/data/r17_library')
AUDIT_ROOT = ROOT/'exp/r17_integration/ops/samples/libero_library_alignment'
PORT_BASE = {'dp_libero':23320,'fastwam_libero':23340,'cosmos_libero':23360,
             'pi05_libero':23160, 'groot_libero':23160,
             'dp_mw':23180, 'fastwam_mw':23170, 'pi05_mw':23160, 'groot_mw':23160, 'cosmos_mw':23240}


def teacher_tasks(campaign, benchmark, tasks):
    """B is a separate input pool; keep the sealed A-serving protocol/table."""
    if benchmark.benchmark != 'libero':
        return [replace(task, extra=dict(task.extra, partition='library', split='library',
            seed=benchmark.library_seed, init_pool='seed8', trial=0, suite_task_id=task.task_id,
            library_episode_id=task.task_id*50+task.orig_init_state_idx)) for task in tasks]
    audit = load_alignment(campaign['alignment'])
    if audit['suite'] != campaign['suite']:
        raise ValueError('B audit belongs to another suite')
    witnesses = {(r['task_id'],r['orig_init_state_idx']):r for r in audit['rows']}
    result = []
    for task in tasks:
        # Library IDs are suite-local even when the canonical task id is30..39.
        name = benchmark.tasks[task.task_id][0]
        suite, task_file = name.split('/',1)
        if suite != audit['suite']:
            raise ValueError('teacher dispatch selected a task from another suite')
        local = next(int(t) for t,r in audit['task_files'].items() if r['name']==task_file)
        row = witnesses[local,task.orig_init_state_idx]
        # Match R17LiberoEpisodeRunner.checked_setup's state/task identity,
        # including its sha256: namespace. The raw state SHA remains unchanged.
        from exp.r17_integration.benchmarks.spec import digest
        extra = dict(task.extra, partition='library',split='library',init_pool='B',trial=0,
            suite_task_id=local,library_episode_id=row['episode_id'],
            init_state_sha256=row['init_array']['state_sha256'],
            task_data_hash=digest(dict(row['init_array'],task_name=name)))
        result.append(replace(task,extra=extra))
    expected = 2 if campaign['smoke'] else 500
    if len(result) != expected or len({t.extra['library_episode_id'] for t in result}) != expected:
        raise ValueError('B teacher plan must have one trial per selected init')
    return result


def validate_b_probe(probe, campaign):
    audit = load_alignment(campaign['alignment'])
    expected = {r['name']:r['sha256'] for r in audit['task_files'].values()}
    actual = probe.get('per_task_digests',{}).get(campaign['suite'])
    if actual != expected:
        raise ValueError('worker did not load the byte-identical historical B pool')


def require_fastwam_ab(receipt_path, descriptor_path, teacher_descriptor):
    """Optional P3 loader-parity check; never substitutes for Q's G2 gate."""
    if not receipt_path or not descriptor_path:
        raise ValueError('Fast-WAM LIBERO library waits for passing GPU A/B receipt and sealed descriptor')
    from exp.r17_integration.debug.validate import file_sha
    receipt = json.loads(Path(receipt_path).read_text())
    descriptor = json.loads(Path(descriptor_path).read_text())
    if (receipt.get('schema') != 'r17.fastwam.loader_parity.v1' or receipt.get('status') != 'passed' or
        not receipt.get('device','').startswith('cuda') or receipt.get('observation_count') != 2 or
        receipt.get('k') != 10 or receipt.get('execution_protocol') != 'released_libero' or
        receipt.get('source_digests',{}).get('descriptor') != file_sha(descriptor_path) or
        descriptor.get('execution_protocol') != 'released_libero' or
        not receipt.get('rows') or not all(r.get('passed') is True for r in receipt['rows'])):
        raise ValueError('Fast-WAM GPU A/B receipt is absent, stale or failed')
    if not descriptor.get('expected_digests') or descriptor['expected_digests'] != teacher_descriptor.get('expected_digests'):
        raise ValueError('Fast-WAM A/B checkpoint/code/norm assets differ from sealed teacher')
    return dict(receipt_sha256=file_sha(receipt_path),descriptor_sha256=file_sha(descriptor_path),
                coverage='released LIBERO B1 two observations/K10; no SR certificate')


def require_fastwam_g2(a_path, b_path, checkpoint, *, teacher_descriptor=None, teacher_model=None):
    """Require PASS G2 slots and their recorded ModelManifest v1 identities.

    A checkpoint directory or today's mutable qualification assets cannot bind
    yesterday's GPU run. Q must attach the run's native model manifest to its
    PASS receipt (inside remote_receipt for a relayed run). Missing pins fail
    before deployment; the fresh teacher seal is compared again after sealing.
    """
    from exp.r17_integration.debug.validate import canonical, digest, file_sha
    if not a_path or not b_path:
        raise ValueError('Fast-WAM requires --g2-a-receipt and --g2-b-receipt')
    if teacher_model is not None:
        teacher_model = json.loads(canonical(teacher_model))
    results, problems = {}, []
    for phase, path in (('A', a_path), ('B', b_path)):
        outer = json.loads(Path(path).read_text())
        slot = 'slot_G2_fastwam_libero_'+phase
        receipt = outer.get('remote_receipt', outer)
        if (outer.get('status') != 'PASS' or outer.get('slot') != slot or
            outer.get('gpu_executed') is not True or receipt.get('status') != 'PASS' or
            receipt.get('slot') != slot or receipt.get('schema') != 'r17.Q.slot.v1' or
            receipt.get('gpu_executed') is not True):
            raise ValueError(f'{slot}: expected PASS GPU receipt for this exact slot')
        if receipt.get('checkpoint') != str(checkpoint):
            raise ValueError(f'{slot}: checkpoint path differs from teacher')
        manifest = receipt.get('model_manifest', {})
        spec = manifest.get('spec', {})
        pins = manifest.get('content_digests', {})
        missing = ['model_manifest.spec.'+k for k in
                   ('checkpoint_identity', 'code_digest', 'norm_identity') if not spec.get(k)]
        if not pins:
            missing.append('model_manifest.content_digests (weight/code/norm asset SHA256s)')
        if missing:
            problems.append(slot+': missing '+', '.join(missing))
            continue
        if (manifest.get('schema') != 'r17.model_manifest.v1' or spec.get('model_id') != 'fastwam_libero' or
            not any('weights' in key for key in pins) or not any(key.startswith('code:') for key in pins) or
            not any(key.startswith('norm:') for key in pins) or any(
                not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v)
                for v in pins.values())):
            raise ValueError(f'{slot}: invalid model manifest / weight/code/norm SHA256s')
        if (spec['checkpoint_identity'] != digest(pins) or
            spec['code_digest'] != digest({k:v for k,v in pins.items() if k.startswith('code:')})):
            raise ValueError(f'{slot}: checkpoint/code digest does not bind recorded assets')
        if teacher_descriptor is not None and pins != teacher_descriptor.get('expected_digests'):
            changed = sorted(k for k in set(pins)|set(teacher_descriptor.get('expected_digests', {}))
                if pins.get(k) != teacher_descriptor.get('expected_digests', {}).get(k))
            raise ValueError(f'{slot}: teacher asset digests differ: {changed}')
        if teacher_model is not None:
            changed = [k for k in ('model_id','checkpoint_identity','code_digest','norm_identity',
                                   'layout','feature_spec','default_sampler') if spec.get(k) != teacher_model.get(k)]
            if changed:
                raise ValueError(f'{slot}: teacher model seal differs: {changed}')
        results[phase] = dict(path=str(path), receipt_sha256=file_sha(path), slot=slot,
            model_manifest_digest=digest(manifest), checkpoint_identity=spec['checkpoint_identity'],
            code_digest=spec['code_digest'], norm_identity=spec['norm_identity'], content_digests=pins)
    if problems:
        raise ValueError('G2 receipts lack qualified teacher bindings; '+'; '.join(problems))
    if any(results['A'][k] != results['B'][k] for k in
           ('checkpoint_identity','code_digest','norm_identity','content_digests')):
        raise ValueError('G2 A/B model identities differ')
    return dict(schema='r17.fastwam.g2_library_gate.v1', status='PASS', slots=results,
                coverage='Q G2 A native mechanics + G2 B framework mechanics; no quality claim')


def disk_estimate(model, suite, *, wire_size=256, normalized_action_bytes=4, state_bytes=4, capture_images=True):
    """Uncompressed ceiling, including two received images for every decision."""
    import math
    fields,layout = model['feature_spec']['fields'],model['layout']
    # Current hooks expose float32 normalized chunks/state, including Cosmos;
    # its solver/physical inverse float64 dtype does not describe fit inputs.
    width = layout.get('model_action_width') or layout['valid_action_dim']
    keys = sum(fields.values())*4
    images = len(model['feature_spec']['real_cameras'])*wire_size*wire_size*3 if capture_images else 0
    fit = keys+(max(layout['valid_state_indices'])+1)*state_bytes+layout['horizon']*width*normalized_action_bytes
    maximum = {'libero_spatial':220,'libero_10':520,'mt50':160}[suite]
    episodes = 2500 if suite == 'mt50' else 500
    rows = episodes*math.ceil(maximum/layout['replan_steps'])
    return dict(schema='r17.teacher_disk_estimate.v1',bundle=model['model_id'],suite=suite,
        max_decisions=rows, bytes_per_decision=dict(keys=keys,images=images,state_actions=fit-keys),
        parent_ceiling_bytes=rows*(fit+images),
        collection_transfer_parent_ceiling_bytes=3*rows*(fit+images),
        excludes='small JSON/NPY headers and fitted arrays; allocator/filesystem overhead',
        assumptions=dict(normalized_action_bytes=normalized_action_bytes,state_bytes=state_bytes,wire_size=wire_size),
        formula=f'{episodes}*ceil(max_episode_controls/native_replan); no image compression assumed')


def finish(campaign, directory):
    """Publish only after accepted episodes, closed replicas and checked sync."""
    from .library_staging import bounded, require_build_host
    require_build_host(campaign)
    from exp.r17_integration.debug.validate import model_binding
    from exp.r17_integration.search.teacher_library import build_teacher_parent, fit_nested_libraries, calibrate_teacher_knobs
    from exp.r17_integration.search.store import Store
    from .formal_driver import plan
    directory = Path(directory)
    receipt = json.loads((directory/'RECEIPT.json').read_text())
    if (receipt.get('status') != 'PASS' or receipt.get('all_owned_closed') is not True or
            receipt.get('worker_storage_cleanup_pending')):
        raise ValueError('library requires passing closed formal collection receipt')
    if campaign['server_node']=='h100' and receipt.get('transfers',{}).get('h100_servers',{}).get('verified',{}).get('status') != 'verified':
        raise ValueError('remote teacher captures lack SHA-checked closed sync')
    if bounded(campaign) and campaign.get('h100_staging'):
        shards = receipt.get('transfers',{}).get('h100_shards',{})
        if shards.get('status')!='verified' or not shards.get('shards'):
            raise ValueError('bounded H100 teacher lacks verified streaming shard receipts')
    benchmark,model,tasks,_ = plan(campaign)
    from .library_repair import capture_sources, outcomes, verify_original_pin
    from exp.r17_integration.debug.attempts import capture_counts
    sources = capture_sources(campaign,directory)
    binding = model_binding(model,benchmark.identity)
    observed, manifests = capture_counts(sources,binding=binding)
    if campaign.get('repair_merge'):
        provenance = json.loads((directory/'REPAIR_PROVENANCE.json').read_text())
        verify_original_pin(provenance['pin'])
        if manifests != campaign['repair_merge']['capture_manifests'] or manifests != provenance['captures']:
            raise ValueError('repair merged capture evidence changed')
    terminal, counts, histories, _ = outcomes(directory,tasks,observed)
    task_data = {}
    if campaign.get('domain', 'libero') != 'libero':
        from exp.r17_integration.debug.reader_profiles import CaptureReader
        for source in sources:
            for row,_,_ in CaptureReader(source,expected_binding=binding,max_cached_blocks=2).decisions():
                task_data[row['task_uid'],row['attempt']] = row['task_data_hash']
    episodes = []
    for task in tasks:
        row = terminal[task.task_uid]
        ep = dict(task.extra, task_uid=task.task_uid,attempt=row['attempt'],orig_init_idx=task.orig_init_state_idx,
            closed_ok=True,success=row['success'],error=None,n_decisions=counts.get((task.task_uid,row['attempt']),0))
        if histories[task.task_uid]:
            ep['prior_attempts'] = histories[task.task_uid]
        if campaign.get('domain', 'libero') != 'libero':
            ep['task_data_hash'] = task_data[task.task_uid,row['attempt']]
        episodes.append(ep)
    if not sources:
        raise ValueError('closed replica teacher capture missing')
    if bounded(campaign) and campaign.get('h100_staging') and (len(shards['shards'])!=len(tasks) or len(sources)!=len(tasks)):
        raise ValueError('teacher streamed shard/episode cohort is incomplete')
    destination = directory/'smoke_parent' if campaign['smoke'] else Path(campaign['library_parent'])
    alignment = load_alignment(campaign['alignment']) if campaign.get('domain', 'libero') == 'libero' else None
    capture_images = campaign.get('capture_images', True)
    parent = Store(destination,expected_binding=binding) if (destination/'READY.json').exists() else build_teacher_parent(
        sources,destination,binding,episodes,alignment,expected_episodes=len(tasks),capture_images=capture_images)
    if (parent.manifest['metadata']['alignment_digest'] != (alignment['digest'] if alignment else None) or
        parent.manifest['metadata']['episodes'] != episodes):
        raise ValueError('existing parent belongs to another aligned collection')
    fit_options = {} if alignment else dict(sizes=campaign.get('subset_counts', [10,20,30,40,50]))
    products = {} if campaign['smoke'] else fit_nested_libraries(parent,alignment,task_names=benchmark.ordered_tasks,**fit_options)
    knobs = {} if campaign['smoke'] else calibrate_teacher_knobs(parent,products,task_names=benchmark.ordered_tasks,
        cost_profile=campaign.get('ir_costs'),miss_sampler=campaign.get('miss_sampler'))
    result = dict(status='PASS',schema='r17.teacher_pipeline.v1',smoke=campaign['smoke'],
        parent=str(destination),parent_digest=parent.manifest['content_digest'],rows=len(parent.rows),
        episodes=len(episodes),failed_episodes=sum(not e['success'] for e in episodes),fits=products,
        alignment_digest=alignment['digest'] if alignment else None,official_A_fit_reads=0,actual_gpu_collection=not campaign.get('synthetic',False),
        knob_products=knobs,knob_cost_qualification_pending=bool(knobs) and any(v['status']=='pending_cost_qualification' for v in knobs.values()))
    write_json(directory/'LIBRARY_PIPELINE.json',result)
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=tuple(PORT_BASE))
    p.add_argument('--suite',required=True,choices=(*SUITES,'mt50'))
    p.add_argument('--checkpoint'); p.add_argument('--alignment',type=Path)
    p.add_argument('--g2-a-receipt',type=Path,help='Q slot_G2_fastwam_libero_A PASS receipt with model_manifest')
    p.add_argument('--g2-b-receipt',type=Path,help='Q slot_G2_fastwam_libero_B PASS receipt with model_manifest')
    p.add_argument('--ab-qualification',type=Path,help='Optional extra P3 loader-parity GPU receipt')
    p.add_argument('--ab-descriptor',type=Path,help='Descriptor for optional P3 loader-parity receipt')
    p.add_argument('--ir-costs',type=Path,help='Measured r17.ir_costs.v1; no π/GR cost reuse')
    p.add_argument('--miss-sampler',type=Path,help='Sealed native SamplerSpec for the cost profile; teacher grid by default')
    p.add_argument('--server-node',choices=('local','h100')); p.add_argument('--worker-node',choices=('timan107','timan108','timan1'))
    from .local_workers import counts
    p.add_argument('--worker-counts',type=counts,help='Per-node counts, e.g. timan107=24,timan1=40')
    p.add_argument('--capture-images',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--subset-counts',type=lambda s:[int(v) for v in s.split(',')],default=[10,20,30,40,50],
                   help='MT50 nested episodes per task, default10,20,30,40,50')
    p.add_argument('--replicas',type=int); p.add_argument('--workers',type=int,default=24)
    p.add_argument('--replica-startup-width',type=int,help='Replicas loaded concurrently (default min(replicas,4))')
    p.add_argument('--max-batch',type=int,default=1)
    p.add_argument('--batch-wait-ms',type=float,default=5)
    p.add_argument('--mps',action='store_true')
    p.add_argument('--driver-mode',choices=('legacy','group-commit'),default='legacy')
    p.add_argument('--memory-budget-gib',type=float); p.add_argument('--gpu',default='cuda:0')
    p.add_argument('--worker-egl-device',type=int,default=0)
    p.add_argument('--worker-egl-devices',type=lambda s:[int(v) for v in s.split(',')])
    p.add_argument('--worker-memory-mib',type=int)
    p.add_argument('--worker-vram-mib',type=int,default=1024)
    p.add_argument('--worker-max-episodes',type=int)
    p.add_argument('--worker-min-free-gib',type=float,default=8,
                   help='Available worker run/stage filesystem GiB required at launch/recycle (default8)')
    p.add_argument('--port',type=int); p.add_argument('--driver-port',type=int)
    p.add_argument('--smoke',action='store_true'); p.add_argument('--run-name')
    p.add_argument('--seconds',type=int); p.add_argument('--episode-seconds',type=int)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run',action='store_true'); mode.add_argument('--prepare-only',action='store_true')
    mode.add_argument('--finish',type=Path,metavar='CLOSED_CAMPAIGN')
    mode.add_argument('--repair',type=Path,metavar='FAILED_CAMPAIGN',help='Launch missing episodes after zero-progress proof or sealed salvage quarantine')
    mode.add_argument('--salvage',type=Path,metavar='FAILED_CAMPAIGN',help='CPU-only closure/quarantine of missing results; bounded H100 drain when remote')
    a = p.parse_args(argv)
    if not 0 < a.worker_min_free_gib < 1024:
        p.error('--worker-min-free-gib must be positive and below1024')
    vla = a.model in ('pi05_libero', 'groot_libero')
    a.server_node = a.server_node or ('local' if a.model in ('dp_libero', 'pi05_libero') else 'h100')
    mw = a.model.endswith('_mw')
    from exp.r17_integration.models.dp.batch_runtime import BatchConfig
    BatchConfig(a.max_batch,a.batch_wait_ms)
    if a.max_batch>1 and (not mw or a.model not in ('dp_mw','fastwam_mw')):
        p.error('library batching supports DP/Fast-WAM MT50')
    if mw != (a.suite == 'mt50'): p.error('MetaWorld models require --suite mt50; LIBERO models require a LIBERO suite')
    a.capture_images = not (mw or vla) if a.capture_images is None else a.capture_images
    a.worker_node = a.worker_node or ('timan107' if mw or vla or a.model=='dp_libero' else 'timan108')
    a.worker_counts = a.worker_counts or {a.worker_node:a.workers}
    a.workers = sum(a.worker_counts.values())
    if 'local' in a.worker_counts: p.error('library simulators require remote worker nodes')
    if 'timan1' in a.worker_counts and (not mw or a.worker_counts['timan1']>40):
        p.error('timan1 MetaWorld workers require GPU3 only, maximum40')
    if sorted(set(a.subset_counts)) != a.subset_counts or any(n<1 or n>50 for n in a.subset_counts):
        p.error('nested per-task counts must be distinct increasing values in1..50')
    a.replicas = a.replicas if a.replicas is not None else (1 if mw else 2 if a.smoke and a.model=='dp_libero' else 12 if a.model=='dp_libero' else 1)
    a.port = a.port if a.port is not None else PORT_BASE[a.model]
    a.driver_port = a.driver_port if a.driver_port is not None else (23112 if a.model=='cosmos_mw' else PORT_BASE[a.model]+9 if mw or vla else PORT_BASE[a.model]+19)
    ports = [a.port,*range(a.port+1,a.port+1+a.replicas),a.driver_port]
    if a.replicas<1 or a.workers<1 or len(set(ports))!=len(ports):
        p.error('positive workers/replicas and distinct ports required')
    if (mw or vla) and a.driver_port not in range(23100,23200): p.error('driver port must lie in23100..23199')
    a.worker_memory_mib = a.worker_memory_mib if a.worker_memory_mib is not None else (2048 if mw else 6144)
    a.worker_max_episodes = a.worker_max_episodes if a.worker_max_episodes is not None else (0 if mw else 4)
    a.seconds = a.seconds or (900 if a.smoke else 172800)
    a.episode_seconds = a.episode_seconds or (min(600,a.seconds) if a.smoke else 3600)
    if not 60<=a.seconds<=172800 or not 30<=a.episode_seconds<=a.seconds or a.smoke and a.seconds>900:
        p.error('bounded episode/campaign required; smoke maximum900s')
    if a.memory_budget_gib is not None and (a.memory_budget_gib<=0 or not math.isfinite(a.memory_budget_gib)):
        p.error('positive finite resident memory budget required')
    if a.miss_sampler and not a.ir_costs: p.error('--miss-sampler requires --ir-costs')
    if bool(a.ab_qualification) != bool(a.ab_descriptor):
        p.error('optional P3 --ab-qualification and --ab-descriptor must be supplied together')
    return a


def main(argv=None):
    from . import formal_eval as formal
    from .library_init import build_init_archive
    a = parse_args(argv)
    if a.salvage:
        from .library_salvage import salvage
        from .tether_retry import retry_scope
        with retry_scope(a.salvage/'salvage'):
            print(json.dumps(salvage(a.salvage,model=a.model.rsplit('_',1)[0],suite=a.suite,
                remote=formal.remote,pull=lambda path,target:formal.transfer(['rsync','pull','h100:'+path,str(target)]))))
        return 0
    if a.finish:
        campaign = json.loads((a.finish/'campaign.json').read_text())
        if not campaign.get('library') or campaign['suite'] != a.suite or campaign['model'] != a.model.rsplit('_',1)[0]:
            raise ValueError('finish must match the closed library campaign')
        if a.ir_costs: campaign['ir_costs']=json.loads(a.ir_costs.read_text())
        if a.miss_sampler: campaign['miss_sampler']=json.loads(a.miss_sampler.read_text())
        print(json.dumps(finish(campaign,a.finish))); return 0
    repair = None
    if a.repair:
        from .library_repair import inspect_repair
        repair = inspect_repair(a.repair,model=a.model.rsplit('_',1)[0],suite=a.suite,
                                checkpoint=a.checkpoint,alignment=a.alignment)
        if a.model=='fastwam_libero':
            original = repair[0]
            a.g2_a_receipt = a.g2_a_receipt or original.get('g2_a_receipt')
            a.g2_b_receipt = a.g2_b_receipt or original.get('g2_b_receipt')
    if a.model=='cosmos_libero' and not (ROOT/'exp/r17_integration/handbacks/HANDBACK_COSMOS.md').exists():
        raise ValueError('Cosmos library waits for sol D HANDBACK_COSMOS.md and debug-capable bundle')
    checkpoint = repair[0]['checkpoint'] if repair else a.checkpoint or (
        '/home/exouser/ckpt' if a.model == 'groot_libero' and a.server_node == 'h100' else formal.DEFAULT_CHECKPOINTS[a.model])
    if a.model=='fastwam_libero':
        require_fastwam_g2(a.g2_a_receipt,a.g2_b_receipt,checkpoint)
    mw = a.suite == 'mt50'
    alignment = a.alignment or AUDIT_ROOT/(a.suite+'.json')
    audit = None if mw else repair[1] if repair else load_alignment(alignment)
    name = a.run_name or ('teacher_'+a.model+'_'+a.suite+('_repair_' if repair else
        ('_smoke4_' if mw else '_smoke2_') if a.smoke else ('_2500_' if mw else '_500_'))+
        time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    if a.mps and a.run_name is None:
        name = 't_'+a.model.rsplit('_',1)[0]+'_'+time.strftime('%H%M%S')+'_'+uuid.uuid4().hex[:6]
    if not name or len(name)>150 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
        raise ValueError('unique plain campaign identifier required')
    home = LIBRARY_ROOT/(a.model+'_'+a.suite)
    if not a.smoke and (home/'bpool').exists():
        raise ValueError('B500 parent already exists; use --finish for a closed collection')
    directory = home/'campaigns'/name
    directory.mkdir(parents=True,exist_ok=False)
    if audit is not None:
        write_json(directory/'ALIGNMENT.json',audit)
    # Both small B pools avoid the stock LIBERO import/probe selecting A for
    # global task0 during a libero_10-only campaign. Only selected suite is run.
    audits = [] if mw else [audit if suite==a.suite else load_alignment(AUDIT_ROOT/(suite+'.json')) for suite in SUITES]
    if repair:
        import shutil
        shutil.copy2(repair[0]['bpool_archive'],directory/'bpool_inputs.tar')
        init = repair[0]['bpool_transfer']
    elif not mw:
        init = build_init_archive(audits,directory/'bpool_inputs.tar')
    else:
        init = None
    model = a.model.rsplit('_',1)[0]
    host = 'ziyanglin.com' if a.server_node=='local' else formal.h100_host()
    campaign = dict(library=True,library_intent_fence='r17.library_input_intent.v1',suite=a.suite,alignment=str(directory/'ALIGNMENT.json'),
        library_parent=str(home/'bpool'),bpool_archive=str(directory/'bpool_inputs.tar'),bpool_transfer=init,
        ab_qualification=str(a.ab_qualification) if a.ab_qualification else None,
        ab_descriptor=str(a.ab_descriptor) if a.ab_descriptor else None,
        g2_a_receipt=str(a.g2_a_receipt) if a.g2_a_receipt else None,
        g2_b_receipt=str(a.g2_b_receipt) if a.g2_b_receipt else None,
        run_name=name,model=model,domain='metaworld' if mw else 'libero',checkpoint=checkpoint,driver_mode=a.driver_mode,
        device=a.gpu,server_node=a.server_node,worker_nodes=a.worker_counts,workers=a.workers,
        capture_images=a.capture_images,subset_counts=a.subset_counts,prepare_only=a.prepare_only,
        worker_egl_device=a.worker_egl_device,replicas=a.replicas,public_port=a.port,
        worker_egl_devices=a.worker_egl_devices,worker_memory_mib=a.worker_memory_mib,
        worker_vram_mib=a.worker_vram_mib,worker_max_episodes=a.worker_max_episodes,
        replica_startup_width=a.replica_startup_width or min(a.replicas,4),
        worker_min_free_gib=a.worker_min_free_gib,
        backend_ports=list(range(a.port+1,a.port+1+a.replicas)),ports=[a.port,*range(a.port+1,a.port+1+a.replicas),a.driver_port],
        driver_host='ziyanglin.com',driver_port=a.driver_port,endpoint_host=host,endpoint=host+':'+str(a.port),
        official_50=False,smoke=a.smoke,smoke_episodes=4 if mw else 2,native_parity_only=False,
        seconds=a.seconds,episode_seconds=a.episode_seconds,retries=2,
        memory_budget_gib=a.memory_budget_gib or (4 if mw and model=='dp' else 20 if model in ('pi05','groot') else 14 if model=='dp' else 36 if model=='fastwam' else 10),
        local_assets=str(directory/'assets'),assets=str(directory/'assets'),server_run=str(directory/'servers'),
        runtime=str(ROOT),vendor='/data/cosmos/cosmos-policy' if a.server_node=='h100' else '/home/weiland/projects/openpi_ext/lines/cosmos-policy',live_k=4)
    if mw:
        for key in ('alignment','bpool_archive','bpool_transfer'):
            campaign.pop(key)
        campaign.update(worker_memory_mib=2048,worker_vram_mib=1024,worker_max_episodes=0)
    if model == 'groot':
        campaign['vendor'] = '/home/weiland/projects/openpi_ext/third_party/gr00t_n15'
    if repair:
        original,_,pin = repair
        if a.smoke or a.server_node != original['server_node']:
            raise ValueError('repair requires the original full suite and execution host')
        campaign.update(semantic_campaign=original.get('semantic_campaign',original['run_name']),
            repair_pin=pin,repair_attempts=pin['repair_attempts'],repair_prior_attempts=pin['prior_attempts'])
        write_json(directory/'REPAIR_PLAN.json',pin)
    if a.server_node=='h100':
        from .library_staging import DEFAULT_POLICY, require_build_host
        campaign['h100_staging'] = dict(DEFAULT_POLICY)
        if not a.capture_images:
            # Image-free episodes are a few MB; 256 MiB slots capped concurrency near 66 episodes.
            campaign['h100_staging']['shard_bytes'] = 64*1024**2
        campaign['h100_batch'] = True
        require_build_host(campaign)
    if a.ir_costs: campaign['ir_costs']=json.loads(a.ir_costs.read_text())
    if a.miss_sampler: campaign['miss_sampler']=json.loads(a.miss_sampler.read_text())
    # Direct endpoints only (formal.open_tunnels never exposes through the broker).
    formal.open_tunnels(campaign,directory)
    campaign['backend_host'] = '127.0.0.1' if a.server_node=='local' else formal.h100_host()
    if a.model=='dp_mw':
        from exp.r17_integration.models.dp.batch_runtime import BatchConfig
        campaign['dp_batching'] = BatchConfig(a.max_batch,a.batch_wait_ms).metadata()
    elif a.model=='fastwam_mw':
        from exp.r17_integration.models.fastwam.formal_batching import bind
        bind(campaign,a.max_batch,a.batch_wait_ms)
    from exp.r17_integration.models.fastwam.ops.mps import bind
    bind(campaign,a.mps)
    deployment = formal.prepare(campaign,directory)
    if a.prepare_only:
        print(json.dumps(dict(status='PASS',cpu_only=True,run=str(directory)))); return 0
    code = formal.run_campaign(campaign,directory,deployment)
    if code: return code
    if repair:
        from .library_repair import merge_repair
        campaign,directory = merge_repair(campaign,directory)
    print(json.dumps(finish(campaign,directory))); return 0


if __name__ == '__main__':
    raise SystemExit(main())
