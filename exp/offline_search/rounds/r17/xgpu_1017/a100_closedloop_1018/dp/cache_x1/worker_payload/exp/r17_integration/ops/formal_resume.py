"""Fail-closed continuation of sealed pure-inference campaigns (CPU only)."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil

from .formal_process import write_json


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


# Resource/transport knobs may change; episode and algorithm settings may not.
SEMANTICS = ('model', 'domain', 'checkpoint', 'server_node', 'device', 'official_50', 'smoke',
             'native_parity_only', 'commit_horizon', 'denoising_steps', 'suites',
             'fastwam_batching', 'fastwam_batch_treatment', 'fastwam_batch_treatment_identity', 'dp_batching',
             'episode_seconds', 'retries', 'worker_egl_device')
SEALED = ('model_manifest.json', 'model_spec.json', 'benchmark.json', 'full.yaml')
PROBE_FIELDS = ('benchmark_protocol', 'task_table_identity', 'libero_code_digest',
                'robosuite_version', 'torch', 'numpy', 'task_data', 'per_task_digests',
                'packages', 'camera_identity')


def same_campaign(old, new):
    if old.get('library') or new.get('library') or old.get('smoke') or old.get('native_parity_only'):
        raise ValueError('resume supports formal pure inference only')
    for key in SEMANTICS:
        if old.get(key) != new.get(key):
            raise ValueError('resume campaign mismatch: '+key)


def probe_binding(probe):
    return {key: probe[key] for key in PROBE_FIELDS if key in probe}


def accepted_evidence(directory, plan, bundle_identity):
    """Keep successes AND legitimate failures, never select on success rate."""
    directory = Path(directory)
    tasks = {t['task_uid']: t for t in plan['tasks']}
    if not tasks or len(tasks) != len(plan['tasks']):
        raise ValueError('resume duplicate or empty planned UID set')
    journal, attempts, steps = (rows(directory/'driver'/name) for name in
                               ('journal.jsonl', 'attempts.jsonl', 'per_step.jsonl'))
    terminal, by_attempt, traces = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in journal:
        if row.get('accepted') is True and row.get('status') in ('done', 'failed') and row.get('error') is None:
            terminal[row.get('task_uid')].append(row)
    for row in attempts:
        if row.get('accepted') is True:
            by_attempt[row.get('task_uid')].append(row)
    for row in steps:
        if row.get('accepted') is True and not row.get('_kind'):
            traces[row.get('task_uid')].append(row)
    kept = {}
    for uid, endings in terminal.items():
        if uid not in tasks or len(endings) != 1:
            raise ValueError('resume unknown or duplicate accepted episode: '+str(uid))
        ending, task = endings[0], tasks[uid]
        attempt = ending.get('attempt')
        cohort = by_attempt[uid]
        final = [r for r in cohort if r.get('attempt') == attempt]
        if (len(final) != 1 or final[0].get('error') is not None or
            final[0].get('success') != ending.get('success') or final[0].get('n_steps', 0) < 1 or
            any(r.get('attempt', 1) > attempt for r in cohort)):
            raise ValueError('resume unresolved or mismatched accepted attempt: '+uid)
        trace = [r for r in traces[uid] if r.get('attempt') == attempt]
        extra = task['extra']
        expected = {key: extra[key] for key in ('model_identity', 'norm_identity', 'layout_identity',
                    'benchmark_protocol', 'yaml_identity', 'num_steps', 'commit_horizon') if key in extra}
        expected.update(bundle_id=task['bundle_id'], bundle_identity=bundle_identity,
                        task_name=extra['task_name'], original_init_idx=task['orig_init_state_idx'],
                        capture_enabled=False, source='policy')
        k = extra['num_steps']
        if (not trace or any(any(r.get(key) != value for key, value in expected.items()) or
                r.get('actual_nfe') != k or r.get('stage_calls') != [1, 1, k] for r in trace) or
                sorted(r.get('decision_seq', -1) for r in trace) != list(range(len(trace)))):
            raise ValueError('resume execution trace differs from sealed full arm: '+uid)
        kept[uid] = ending
    return dict(journal=journal, attempts=attempts, steps=steps, accepted=kept)


def inspect_source(directory, *, closure_waiver=None):
    directory = Path(directory).resolve(strict=True)
    if (directory/'FULL_EPISODE_PLAN.json').exists() and not (directory/'RESUME_MERGE.json').exists():
        raise ValueError('resume source is an unmerged continuation; select its original resume_from run')
    receipt = read(directory/'RECEIPT.json')
    if receipt.get('status') == 'PENDING_EVIDENCE':
        raise ValueError('resume source evidence sealing is pending; wait for its final receipt')
    if receipt.get('all_owned_closed') is not True:
        if closure_waiver is None:
            raise ValueError('resume source must be closed; refusing a running campaign')
        from .formal_resume_waiver import validate
        validate(directory, closure_waiver)
    campaign = read(directory/'campaign.json')
    same_campaign(campaign, campaign)
    plan = read(directory/'EPISODE_PLAN.json')
    if plan.get('cache_enabled') is not False or plan.get('debug_capture') is not False:
        raise ValueError('resume source is not pure inference')
    model = read(directory/'assets/model_spec.json')
    manifest = read(directory/'assets/model_manifest.json')
    if manifest.get('spec') != model or not manifest.get('content_digests') or plan.get('model') != model:
        raise ValueError('resume source has no bound ModelManifest')
    attestations = read(directory/'SERVER_ATTESTATIONS.json')
    if len(attestations) != campaign['replicas'] or any(
            a['metadata'].get('r17_model_spec') != model or
            a['metadata'].get('cache_enabled') is not False or
            a['metadata'].get('debug_capture') is not False or
            a['metadata'].get('formal_pure_inference') is not True for a in attestations):
        raise ValueError('resume source resident attestation mismatch')
    ack = read(directory/'BUNDLE_ACK.json')
    if ack.get('bundle_id') != 'formal_full_v1' or not ack.get('bundle_identity'):
        raise ValueError('resume source bundle was not acknowledged')
    evidence = accepted_evidence(directory, plan, ack['bundle_identity'])
    if not evidence['accepted'] or len(evidence['accepted']) >= len(plan['tasks']):
        raise ValueError('resume needs a nonempty partial set of accepted episodes')
    census = read(directory/'driver/census.json')
    probes = [probe_binding(value['probe']) for value in census.values() if value.get('probe')]
    if not probes or any(probe != probes[0] for probe in probes):
        raise ValueError('resume source worker probes disagree')
    required = ['campaign.json', 'RECEIPT.json', 'EPISODE_PLAN.json', 'SERVER_ATTESTATIONS.json',
                'BUNDLE_ACK.json', 'driver/DRIVER_RECEIPT.json', 'driver/census.json',
                'WORKER_DEPLOYMENT.json',
                'driver/journal.jsonl', 'driver/attempts.jsonl', 'driver/per_step.jsonl',
                *['assets/'+name for name in SEALED]]
    if (directory/'assets/execution_sampler.json').exists():
        required.append('assets/execution_sampler.json')
    if campaign.get('server_node') == 'h100':
        required.append('runtime.tar')
    return dict(source=str(directory), campaign=campaign, plan=plan, evidence=evidence,
                probe=probes[0], bundle_identity=ack['bundle_identity'],
                pins={name: sha(directory/name) for name in required})


def attach(campaign, source):
    """Validate CLI semantics before runtime deployment or any GPU launch."""
    waiver = None
    if campaign.get('resume_waive_unreachable_source_node'):
        from .formal_resume_waiver import issue
        waiver = issue(source, campaign, campaign['resume_waive_unreachable_source_node'])
    original = inspect_source(source, closure_waiver=waiver)
    same_campaign(original['campaign'], campaign)
    campaign.update(resume_from=original['source'],
                    semantic_campaign=original['campaign'].get('semantic_campaign', original['campaign']['run_name']),
                    resume_source_pins=original['pins'], resume_worker_probe=original['probe'])
    if waiver is not None:
        campaign['resume_source_closure_waiver'] = waiver


def check_pins(campaign):
    source = Path(campaign['resume_from'])
    if any(sha(source/name) != value for name, value in campaign['resume_source_pins'].items()):
        raise ValueError('resume source evidence changed after selection')
    if campaign.get('resume_source_closure_waiver'):
        from .formal_resume_waiver import validate, validate_destination
        validate(source, campaign['resume_source_closure_waiver'])
        validate_destination(campaign, campaign['resume_source_closure_waiver'])


def copy_runtime(campaign, destination):
    """Use preserved model code, never a currently edited package or live copy."""
    check_pins(campaign)
    source = Path(campaign['resume_from'])/'runtime.tar'
    shutil.copyfile(source, destination)
    if sha(destination) != campaign['resume_source_pins']['runtime.tar']:
        raise ValueError('resume archived runtime SHA mismatch')
    return sha(destination)


def bind(campaign, directory, full_plan):
    """Require fresh sealed byte identities; preserve UID/seed/cell identities."""
    directory = Path(directory)
    check_pins(campaign)
    source = inspect_source(campaign['resume_from'], closure_waiver=campaign.get('resume_source_closure_waiver'))
    same_campaign(source['campaign'], campaign)
    old_assets, assets = Path(source['source'])/'assets', Path(campaign['local_assets'])
    for name in (*SEALED, 'execution_sampler.json'):
        a, b = old_assets/name, assets/name
        if a.exists() != b.exists() or a.exists() and sha(a) != sha(b):
            raise ValueError('resume sealed identity mismatch: '+name)
    root = Path(__file__).resolve().parents[3]
    for deployment in read(Path(source['source'])/'WORKER_DEPLOYMENT.json').values():
        for name, pin in deployment['installed']['files'].items():
            # Lifecycle fixes are permitted; simulator/episode/seed code is not.
            if '/ops/' not in name and '/qualification/' not in name:
                if not (root/name).is_file() or sha(root/name) != pin:
                    raise ValueError('resume worker episode code changed: '+name)
    def semantic(task):
        return {key: value for key, value in task.items() if key not in ('server_host', 'server_port')}
    old = {t['task_uid']: semantic(t) for t in source['plan']['tasks']}
    fresh = {t['task_uid']: semantic(t) for t in full_plan['tasks']}
    if (fresh != old or len(fresh) != len(full_plan['tasks']) or
            full_plan['benchmark_protocol'] != source['plan']['benchmark_protocol']):
        raise ValueError('resume episode/config/bundle/seed/cell mismatch')
    retained = set(source['evidence']['accepted'])
    missing = [t['task_uid'] for t in full_plan['tasks'] if t['task_uid'] not in retained]
    campaign['resume_task_uids'] = missing
    selection = dict(schema='r17.formal_resume.v1', source=source['source'], source_pins=source['pins'],
                     source_status=read(Path(source['source'])/'RECEIPT.json')['status'],
                     semantic_campaign=campaign['semantic_campaign'],
                     retained_episodes=len(retained), dispatched_episodes=len(missing),
                     expected_episodes=len(full_plan['tasks']), retained_task_uids=sorted(retained),
                     missing_task_uids=missing, bundle_identity=source['bundle_identity'],
                     worker_probe=source['probe'])
    if campaign.get('resume_source_closure_waiver'):
        selection['source_closure_waiver'] = campaign['resume_source_closure_waiver']
    write_json(directory/'FULL_EPISODE_PLAN.json', full_plan)
    write_json(directory/'RESUME.json', selection)
    return dict(full_plan, expected_episodes=len(missing),
                tasks=[t for t in full_plan['tasks'] if t['task_uid'] in set(missing)])


def validate_probe(probe, campaign):
    if campaign.get('resume_from') and probe_binding(probe) != campaign['resume_worker_probe']:
        raise ValueError('resume worker simulator/init binding mismatch')


def cpu_probe(path):
    """Same native probe and original-state hash, without constructing EGL."""
    from exp.r17_integration.conductor.manifest import load_benchmark_spec
    from exp.r17_integration.benchmarks.libero_bridge import LiberoBenchmarkAdapter, SUITES
    from examples.libero import main as native
    from examples.libero.worker_entry import build_probe
    from libero.libero import benchmark
    spec = load_benchmark_spec(path, register=False)
    args = native.Args(task_suite_name=SUITES[0], seed=spec.seed, resize_size=spec.wire_size,
                       replan_steps=spec.replan_steps, num_steps_wait=spec.settle_steps)
    probe = LiberoBenchmarkAdapter(spec).worker_probe(build_probe(args, ''))
    state = benchmark.get_benchmark_dict()[SUITES[0]]().get_task_init_states(0)[0]
    import numpy as np
    raw = state.detach().cpu().numpy() if hasattr(state, 'detach') else np.asarray(state)
    probe['task_data'] = dict(task_name=spec.tasks[0][0], original_init_idx=0,
                             state_sha256=hashlib.sha256(raw.tobytes()).hexdigest())
    return probe


def merge(campaign, directory, completion):
    """Run only after the missing-cell driver and remote fleets have closed."""
    directory = Path(directory)
    check_pins(campaign)
    source = inspect_source(campaign['resume_from'], closure_waiver=campaign.get('resume_source_closure_waiver'))
    selection = read(directory/'RESUME.json')
    if selection.get('source_closure_waiver') != campaign.get('resume_source_closure_waiver'):
        raise ValueError('resume selection closure waiver evidence changed')
    dispatch_plan = read(directory/'EPISODE_PLAN.json')
    full = read(directory/'FULL_EPISODE_PLAN.json')
    if read(directory/'BUNDLE_ACK.json').get('bundle_identity') != source['bundle_identity']:
        raise ValueError('resume resident acknowledged a different bundle')
    segment = accepted_evidence(directory, dispatch_plan, source['bundle_identity'])
    old, new = set(source['evidence']['accepted']), set(segment['accepted'])
    wanted = {t['task_uid'] for t in full['tasks']}
    if (old & new or old | new != wanted or new != set(selection['missing_task_uids']) or
            completion['accepted_episodes'] != len(new)):
        raise ValueError('resume merged journal is incomplete or overlaps retained cells')
    for value in read(directory/'driver/census.json').values():
        if value.get('probe'):
            validate_probe(value['probe'], campaign)
    raw = directory/'resume_segment/driver'
    shutil.copytree(directory/'driver', raw)
    merged = {}
    for name, key in (('journal.jsonl', 'journal'), ('attempts.jsonl', 'attempts'), ('per_step.jsonl', 'steps')):
        prior = [row for row in source['evidence'][key] if row.get('task_uid') in old]
        own = [row for row in segment[key] if row.get('task_uid') in new]
        merged[name] = prior+own
    # Preserve the raw segment before changing only this new run's aggregate inputs.
    for name, content in merged.items():
        path = directory/'driver'/name
        temporary = path.with_suffix('.merged')
        with temporary.open('x') as f:
            for row in content:
                f.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
        temporary.replace(path)
    old_driver = read(Path(source['source'])/'driver/DRIVER_RECEIPT.json')
    own_driver = read(raw/'DRIVER_RECEIPT.json')
    proof = {key: value for key, value in selection.items() if key not in
             ('retained_task_uids', 'missing_task_uids', 'worker_probe')}
    proof.update(status='PASS', segment_completion=completion, segment_driver=str(raw),
                 abandoned_prior_attempts=sum(r.get('task_uid') not in old for r in source['evidence']['attempts']),
                 abandoned_prior_error_attempts=sum(r.get('task_uid') not in old and r.get('error') is not None
                                                   for r in source['evidence']['attempts']))
    write_json(directory/'driver/DRIVER_RECEIPT.json', dict(status='PASS', all_done=True,
        rejected_worker_probes=0, elapsed_seconds=old_driver['elapsed_seconds']+own_driver['elapsed_seconds'],
        resume=proof, segment_receipt=own_driver))
    write_json(directory/'EPISODE_PLAN.json', full)
    write_json(directory/'RESUME_MERGE.json', proof)
    return proof
