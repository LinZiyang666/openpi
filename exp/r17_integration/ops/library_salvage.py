"""CPU-only node-failure salvage; originals remain evidence, never edited.

Only a proved dead campaign may be drained. Entire incomplete episodes are
quarantined; a separate sealed collection, rather than a rewritten FAIL
receipt, is passed to repair. No simulator, model, port or GPU lock is started.
"""
from __future__ import annotations

from dataclasses import asdict
import inspect
import json
from pathlib import Path
import re

from exp.r17_integration.debug.attempts import capture_counts, json_rows
from exp.r17_integration.debug.reader_profiles import canonical_episode, merge_decision_ledgers
from exp.r17_integration.debug.validate import decision_identity, digest, file_sha, model_binding
from .formal_process import alive, write_json
from .library_repair import capture_sources, seal_pins


def process_probe(entries, campaign):
    """Read /proc only: refuse live leaders, group members and campaign argv."""
    import os
    result = []
    groups = {e['pgid'] for e in entries if e.get('pgid')}
    for e in entries:
        if alive(e):
            raise ValueError('salvage owned process still alive: '+str(e['pid']))
        result.append(dict(pid=e['pid'], start_ticks=e['start_ticks'], alive=False))
    for p in Path('/proc').glob('[0-9]*'):
        try:
            fields = (p/'stat').read_text().split(') ', 1)[1].split()
            if fields[0] != 'Z' and int(fields[2]) in groups:
                raise ValueError('salvage campaign/group member still alive: '+p.name)
            argv = (p/'cmdline').read_bytes().decode(errors='replace').split('\0')
            modules = {'exp.r17_integration.ops.'+m for m in
                       ('formal_workers','formal_server','formal_driver','formal_gpu_lock')}
            campaign_worker = any(arg in modules for arg in argv) and any(campaign in arg for arg in argv)
            if fields[0] != 'Z' and campaign_worker:
                raise ValueError('salvage campaign/group member still alive: '+p.name)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return result


def receipt_processes(references):
    """Read existing node receipts; transport only paths and ownership pins."""
    entries = []
    for reference in references:
        path = Path(reference['path'])
        if not path.is_absolute():raise ValueError('absolute ownership receipt path required')
        if not path.exists():
            continue  # wiped with its machine (jupyter pod restart, 10-21): nothing it recorded can still run
        recorded = json.loads(path.read_text())
        for role, pin in reference['identities'].items():
            observed = recorded.get(role, {})
            if {k: observed.get(k) for k in pin} != pin:
                raise ValueError('remote ownership receipt identity changed: '+str(path)+' '+role)
            entries.append(pin)
    return entries


def prove_closed(receipt, campaign, remote):
    local = receipt.get('owned_local', [])
    if not local or not receipt.get('owned_remote'):
        raise ValueError('salvage requires original process ownership receipts')
    proof = dict(local=process_probe(local, campaign['run_name']), remote={})
    grouped = {}
    for owned in receipt['owned_remote']:
        identities = {role: {k: owned['receipt'][role][k] for k in ('pid','start_ticks','pgid')
                             if k != 'pgid' or k in owned['receipt'][role]}
                      for role in ('child','supervisor') if role in owned['receipt']}
        if not identities:
            raise ValueError('missing remote process identity')
        if owned['node'] == 'timan108':raise ValueError('timan108 is disabled; cannot probe closure')
        path = (owned.get('request') or owned['receipt'].get('request', {})).get('receipt')
        if not path:raise ValueError('missing remote ownership receipt path')
        grouped.setdefault(owned['node'], []).append(dict(path=path, identities=identities))
    # Full argv/env stay in the existing receipt FILE on the node. They may
    # contain megabytes of embedded source, exceeding Linux MAX_ARG_STRLEN.
    source = inspect.getsource(alive)
    from .formal_process import ticks
    source = inspect.getsource(ticks)+'\n'+source+'\n'+inspect.getsource(process_probe)+'\n'+inspect.getsource(receipt_processes)
    for node, references in grouped.items():
        batches, batch = [], []
        for reference in references:
            if len(json.dumps(reference).encode()) > 8192:raise ValueError('ownership receipt reference too large')
            if batch and len(json.dumps(batch+[reference]).encode()) > 8192:
                batches.append(batch);batch=[]
            batch.append(reference)
        if batch:batches.append(batch)
        proof['remote'][node] = []
        for batch in batches:
            code = ('from pathlib import Path; import json\n'+source+
                    '\nprint(json.dumps(process_probe(receipt_processes('+repr(batch)+'),'+repr(campaign['run_name'])+')))')
            proof['remote'][node].extend(json.loads(remote(node,['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','python3','-c',code])))
    return dict(status='PASS', all_owned_closed=True, processes=proof)


def remote_shards(campaign, action, shard_id=None, receipt=None):
    """Executed on H100 only, using the campaign's existing stdlib runtime.

    Open shards may be snapshotted only AFTER process closure proof. A SHA
    transfer proves receipt, not episode validity; open/torn payloads stay
    quarantined locally. One bounded /tmp stage at a time, with durable ack.
    """
    import shutil
    from .library_staging import ShardBudget, policy, TRANSFER_ROOT
    from .sync_back import snapshot_run, atomic_bytes
    from .build_client_bundle import json_bytes, sha256
    from exp.r17_integration.qualification.archive_stage import stage_archive, cleanup_stage
    limits = policy(campaign)
    budget = ShardBudget.__new__(ShardBudget)
    budget.campaign, budget.limits = campaign, limits
    budget.root = Path(campaign['server_run'])
    budget.state_path = budget.root/'SHARD_BUDGET.json'
    if budget.root.resolve() != budget.root or not budget.state_path.is_file():
        raise ValueError('missing/extraneous campaign shard budget')
    # No admission for new GPU work: reserve only this bounded transfer. Do not
    # require a new20GB capture reservation while another campaign is running.
    if shutil.disk_usage('/tmp').free < limits['min_free_tmp_bytes']+limits['transfer_reserve_bytes']:
        raise ValueError('salvage /tmp free-space admission failed')
    state = json.loads(budget.state_path.read_text())
    if state['campaign'] != campaign['run_name'] or state['policy'] != limits:
        raise ValueError('foreign campaign/policy quota')
    if action == 'inventory':
        return dict(shards=state['shards'], stage=state['stage'], failure=state['failure'],
                    high_water_reserved_bytes=state['high_water_reserved_bytes'])
    if action == 'batch_ack':
        from .library_batch import batch_ack
        with budget.locked() as state:
            return batch_ack(budget,state,receipt)
    if not re.fullmatch('[0-9a-f]{64}', shard_id or ''):
        raise ValueError('invalid owned shard id')
    with budget.locked() as state:
        entry = state['shards'].get(shard_id)
        if entry is None:
            saved = budget.root/'SHARD_RECEIPTS'/(shard_id+'.json')
            if action=='ack' and saved.is_file() and json.loads(saved.read_text())==receipt:
                return dict(status='PASS',shard_id=shard_id,already_cleaned=True)
            raise ValueError('missing owned shard')
        leaf = budget.path(entry)
        if action == 'stage':
            if state['stage']:
                stage = state['stage']
                if stage['shard_id'] != shard_id or not stage.get('transfer'):
                    raise ValueError('another/partial transfer occupies bounded staging')
                return stage['transfer']
            size = sum(p.stat().st_size for p in leaf.rglob('*') if p.is_file())
            if size > limits['shard_bytes'] or sum(e.get('reserved_bytes',limits['shard_bytes']) for e in state['shards'].values())+limits['transfer_reserve_bytes']+limits['metadata_reserve_bytes'] > limits['limit_bytes']:
                raise ValueError('salvage shard/staging exceeds capture bound')
            state['stage'] = dict(shard_id=shard_id)
            atomic_bytes(budget.state_path,json_bytes(state))
            archive = TRANSFER_ROOT/campaign['run_name']/'server/shards'/(shard_id+'.tar')
            transfer = snapshot_run(leaf,archive)
            if archive.stat().st_size>limits['shard_bytes']:
                raise ValueError('salvage archive exceeds bounded transfer')
            transfer.update(stage_archive(archive,campaign['run_name']+'_shards','workers',transfer['archive_sha256']))
            transfer.update(shard_id=shard_id,relative=entry['relative'],campaign=campaign['run_name'],original_status=entry['status'])
            state['stage']['transfer'] = transfer
            return transfer
        if action != 'ack':
            raise ValueError('unknown salvage transfer action')
        stage = state['stage']
        if not stage or stage['shard_id']!=shard_id:
            raise ValueError('no matching staged shard')
        transfer = stage['transfer']
        if (receipt.get('schema')!='r17.salvage_shard_receipt.v1' or receipt.get('status')!='verified' or
                receipt.get('campaign')!=campaign['run_name'] or receipt.get('shard_id')!=shard_id or
                receipt.get('archive_sha256')!=transfer['archive_sha256'] or receipt.get('snapshot_digest')!=transfer['manifest']['digest'] or
                receipt.get('build_host')!='weilandserver' or receipt.get('partials')!=transfer['manifest']['partials']):
            raise ValueError('missing exact durable salvage receipt; remote copy retained')
        actual = {str(p.relative_to(leaf)):sha256(p) for p in leaf.rglob('*') if p.is_file() and p.name not in ('.eval.lock','TRANSFER_SHA256.json')}
        if actual!=transfer['manifest']['files'] or any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in leaf.rglob('*')):
            raise ValueError('salvage shard changed after verified transfer')
        removed_bytes = sum(p.stat().st_size for p in leaf.rglob('*') if p.is_file())
        source = TRANSFER_ROOT/campaign['run_name']/'server/shards'/(shard_id+'.tar')
        if Path(transfer['source_archive'])!=source or source.resolve()!=source or sha256(source)!=transfer['archive_sha256']:
            raise ValueError('foreign/changed salvage source archive')
        # Publish remote acknowledgment before cleanup: it is also restart proof.
        atomic_bytes(budget.root/'SHARD_RECEIPTS'/(shard_id+'.json'),json_bytes(receipt))
        cleanup_stage(transfer['archive'],campaign['run_name']+'_shards','workers',transfer['archive_sha256'])
        source.unlink()
        for path in sorted(leaf.rglob('*'),key=lambda p:len(p.parts),reverse=True):
            if path.is_file():path.unlink()
            elif path.is_dir():path.rmdir()
        leaf.rmdir()
        del state['shards'][shard_id]
        state['stage'] = None
        return dict(status='PASS',shard_id=shard_id,removed_after_verified_receipt=True,removed_bytes=removed_bytes)


def evidence_rows(path):
    """Retain complete lines; identify a torn tail explicitly, never parse it."""
    path = Path(path)
    if not path.exists():return [], []
    raw = path.read_bytes(); rows=[]; issues=[]
    for index,line in enumerate(raw.splitlines(keepends=True)):
        try:
            if not line.endswith(b'\n'):raise ValueError('torn tail')
            rows.append(json.loads(line))
        except (ValueError,TypeError) as exc:
            issues.append(dict(path=str(path),line=index+1,reason=str(exc),raw_sha256=digest(line.hex())))
    return rows, issues


def source_pins(original):
    original=Path(original)
    campaign=json.loads((original/'campaign.json').read_text())
    return {str(p):file_sha(p) for p in [original/'campaign.json',original/'EPISODE_PLAN.json',original/'RECEIPT.json',
        Path(campaign['alignment']),Path(campaign['bpool_archive']),original/'H100_SHARD_TRANSFERS.json',
        *[original/'driver'/n for n in ('journal.jsonl','attempts.jsonl','per_step.jsonl','dispatch.jsonl')]] if p.is_file()}


def verify_salvage(campaign):
    salvage=campaign.get('node_failure_salvage')
    if not salvage:return
    path=Path(salvage['provenance']);proof=json.loads(path.read_text())
    if digest({k:v for k,v in proof.items() if k!='digest'})!=salvage['digest'] or proof['digest']!=salvage['digest']:
        raise ValueError('salvage provenance seal differs')
    if proof.get('closure',{}).get('all_owned_closed') is not True:
        raise ValueError('salvage lacks process closure proof')
    for name,sha in proof['source_pins'].items():
        if file_sha(name)!=sha:raise ValueError('salvage original evidence changed: '+name)
    for name,sha in proof['capture_files'].items():
        if file_sha(name)!=sha:raise ValueError('salvage capture/quarantine evidence changed: '+name)
    if seal_pins(campaign)!=proof['seal']:
        raise ValueError('salvage teacher seal differs')


def verify_discard(row, counts):
    reference=row.get('salvage_proof',{})
    proof=json.loads(Path(reference['path']).read_text())
    if digest({k:v for k,v in proof.items() if k!='digest'})!=reference['digest'] or proof['digest']!=reference['digest']:
        raise ValueError('discarded attempt lacks sealed salvage evidence')
    key=row['task_uid'],row['attempt']
    matches=[r for r in proof['discarded_attempts'] if (r['task_uid'],r['attempt'])==key]
    plain={k:v for k,v in row.items() if k!='salvage_proof'}
    if len(matches)!=1 or matches[0]!=plain or counts.get(key,0)!=0 or proof['closure'].get('all_owned_closed') is not True:
        raise ValueError('discarded node-failure attempt is not quarantined/proven')


def verify_received(sources, campaign, receipts):
    """Recheck original and newly pulled shard install receipts and all bytes."""
    import hashlib
    from .build_client_bundle import json_bytes
    for leaf in {Path(source).parent for source in sources}:
        receipt=receipts.get(leaf.name,{}).get('receipt',{})
        snapshot=json.loads((leaf/'TRANSFER_SHA256.json').read_text())
        sealed={k:v for k,v in snapshot.items() if k!='digest'}
        if (receipt.get('status')!='verified' or receipt.get('campaign')!=campaign['run_name'] or
                receipt.get('shard_id')!=leaf.name or receipt.get('destination')!=str(leaf) or
                receipt.get('build_host')!='weilandserver' or receipt.get('partials')!=snapshot['partials'] or
                receipt.get('snapshot_digest')!=snapshot['digest'] or
                hashlib.sha256(json_bytes(sealed)).hexdigest()!=snapshot['digest']):
            raise ValueError('capture lacks exact SHA-verified shard receipt: '+str(leaf))
        actual={str(p.relative_to(leaf)):file_sha(p) for p in leaf.rglob('*') if p.is_file() and
                p.name not in ('.eval.lock','TRANSFER_SHA256.json')}
        if actual!=snapshot['files'] or any(p.is_symlink() for p in leaf.rglob('*')):
            raise ValueError('received shard changed/unsealed since verified install: '+str(leaf))


def reconstruct(original, destination, campaign, sources, closure, transfers):
    from .formal_driver import plan
    from .library_alignment import load_alignment
    original,destination=Path(original),Path(destination)
    if closure.get('all_owned_closed') is not True:raise ValueError('salvage requires closed owned processes')
    benchmark,model,tasks,_=plan(campaign)
    recorded=json.loads((original/'EPISODE_PLAN.json').read_text())
    if recorded['model']!=model or recorded['tasks']!=[asdict(t) for t in tasks]:
        raise ValueError('salvage episode plan/alignment differs')
    audit=load_alignment(campaign['alignment'])
    if file_sha(campaign['bpool_archive'])!=campaign['bpool_transfer']['archive_sha256']:
        raise ValueError('salvage B archive differs')
    planned={t.task_uid:t for t in tasks};binding=model_binding(model,benchmark.identity)
    original_capture_pins={}
    if campaign['server_node']=='local':
        from .library_capture_views import episode_views
        sources,original_capture_pins=episode_views(sources,destination/'capture_views',binding)
    rows={};issues=[]
    for name in ('journal','attempts','per_step','dispatch'):
        rows[name],bad=evidence_rows(original/'driver'/(name+'.jsonl'));issues.extend(bad)
    if any(r.get('task_uid') not in planned for name in rows for r in rows[name]):
        raise ValueError('foreign episode in original salvage journal')
    decisions={};valid_sources={};bad_sources=[];captured={};all_source_pins=dict(original_capture_pins)
    for source in sorted(set(map(Path,sources))):
        keys=set()
        for p in source.glob('decisions_*.jsonl'):
            rr,_=evidence_rows(p);keys.update((r['task_uid'],r['attempt']) for r in rr)
        for name in ('decision_intents.jsonl','execution.jsonl'):
            rr,_=evidence_rows(source/name);keys.update((r['task_uid'],r['attempt']) for r in rr)
        shard_manifest=source.parent/'manifest.json'
        if shard_manifest.is_file():
            shard=json.loads(shard_manifest.read_text())
            identity=shard.get('episode_identity')
            if identity:
                if (shard.get('schema')!='r17.teacher_shard.v1' or
                        shard.get('campaign')!=campaign['run_name'] or len(identity)!=9 or
                        identity[-2] not in planned or keys and keys!={tuple(identity[-2:])}):
                    raise ValueError('shard/episode ownership differs from original campaign')
                keys.add(tuple(identity[-2:]))
        all_source_pins.update({str(p):file_sha(p) for p in source.rglob('*') if p.is_file()})
        all_source_pins.update({str(p):file_sha(p) for p in
            (source.parent/'manifest.json',source.parent/'TRANSFER_SHA256.json') if p.is_file()})
        for key in keys:
            decisions_n=sum(r.get('task_uid')==key[0] and r.get('attempt')==key[1] for p in source.glob('decisions_*.jsonl') for r in evidence_rows(p)[0])
            intents_n=sum((r.get('task_uid'),r.get('attempt'))==key for r in evidence_rows(source/'decision_intents.jsonl')[0])
            captured[key]=captured.get(key,0)+max(decisions_n,intents_n)
        try:
            counts,_=capture_counts([source],binding=binding)
            raw=[r for p in source.glob('decisions_*.jsonl') for r in json_rows(p)]
            acks=json_rows(source/'execution.jsonl')
            joined=merge_decision_ledgers(raw,execution_rows=acks)
            if not raw or any((r['task_uid'],r['attempt']) not in keys for r in joined):raise ValueError('empty/foreign capture')
            for r in joined:decisions.setdefault((r['task_uid'],r['attempt']),[]).append(r)
            valid_sources[source]=keys
        except (ValueError,KeyError,OSError) as exc:
            bad_sources.append(dict(source=str(source),attempts=[list(k) for k in sorted(keys)],reason=repr(exc)))
    selected={};dropped=[]
    node_failure='node_offline:' in str(json.loads((original/'RECEIPT.json').read_text()))
    if not node_failure:
        from exp.r17_integration.debug.attempts import pre_decision_error
        for row in rows['attempts']:
            if row.get('error') is not None and not pre_decision_error(row,captured.get((row['task_uid'],row['attempt']),0),
                    reported_rows=[r for r in rows['per_step'] if r.get('task_uid')==row['task_uid'] and r.get('attempt')==row['attempt']]):
                raise ValueError('local salvage cannot retry an error after progress: '+row['task_uid'])
    # A torn driver line has unknown identity; conservatively quarantine its
    # entire affected ledger rather than invent the omitted fields.
    uncertain=bool(issues)
    for uid,task in planned.items():
        terminal=[r for r in rows['journal'] if r.get('accepted') is True and r.get('task_uid')==uid]
        attempts=[r for r in rows['attempts'] if r.get('accepted') is True and r.get('task_uid')==uid]
        reported=[r for r in rows['per_step'] if r.get('accepted') is True and r.get('task_uid')==uid and not r.get('_kind')]
        reason=None
        if len(terminal)!=1 or terminal[0].get('error') is not None or type(terminal[0].get('success')) is not bool:reason='no unique accepted non-error outcome'
        else:
            end=terminal[0];key=uid,end.get('attempt');caps=sorted(decisions.get(key,[]),key=lambda r:r['decision_seq'])
            cost=sorted([r for r in reported if r['attempt']==key[1]],key=lambda r:r['decision_seq'])
            if uncertain:reason='torn original driver ledger; identity unknown'
            elif len(attempts)!=1 or attempts[0].get('error') is not None or attempts[0]['attempt']!=key[1] or attempts[0].get('success')!=end['success'] or end.get('status') not in ('done','failed'):reason='attempt/terminal outcome disagreement'
            elif not caps or [r['decision_seq'] for r in caps]!=list(range(len(caps))) or [r['decision_seq'] for r in cost]!=list(range(len(caps))):reason='incomplete/noncontiguous decisions or driver ledger'
            elif any(r.get('executed_count') is None or r.get('execution_status')!='acknowledged' for r in caps) or caps[-1].get('terminal') is not True or any(r.get('terminal') for r in caps[:-1]):reason='missing/disagreeing execution ack or final ack'
            elif attempts[0].get('n_steps')!=sum(r['executed_count'] for r in caps) or ('n_steps' in end and end['n_steps']!=attempts[0]['n_steps']):reason='executed steps disagree with terminal outcome'
            elif any(r.get('source')!='policy' or r.get('actual_nfe')!=binding['sampler']['num_steps'] or r.get('stage_calls')!=[1,1,binding['sampler']['num_steps']] or r.get('sampler_identity')!=digest(binding['sampler']) for r in caps):reason='non-native teacher capture'
            elif any(any(r.get(k)!=task.extra.get(k) for k in ('init_pool','split','library_episode_id','suite_task_id','init_state_sha256','task_data_hash')) for r in caps):reason='capture differs from sealed B coordinates/input'
            elif any(decision_identity(r)!=decision_identity(canonical_episode(c)) for r,c in zip(caps,cost)):reason='capture/driver decision identity disagreement'
            elif any(uid in [k[0] for k in b['attempts']] for b in bad_sources):reason='partial/torn/checksum-failed capture for this episode'
            else:selected[uid]=end
        if reason:
            if not node_failure and any(r.get('error') is None for r in terminal+attempts):
                raise ValueError('local salvage cannot discard a returned teacher outcome: '+uid+'; '+reason)
            dispatched=[r for r in rows['dispatch'] if r.get('task_uid')==uid]
            generations=sorted(set(r.get('attempt') for r in terminal+attempts+dispatched+reported)|{k[1] for k in captured if k[0]==uid})
            if generations and generations!=list(range(1,max(generations)+1)):raise ValueError('unknown prior attempt generations: '+uid)
            dropped.append(dict(task_uid=uid,reason=reason,terminal=terminal,attempts=attempts,
                dispatches=dispatched,captured_decisions=sum(n for k,n in captured.items() if k[0]==uid),reported_decisions=len(reported),generations=generations))
    # Per-episode H100 shards only. Never keep a source containing another,
    # incomplete episode, even if one of its episodes otherwise looks complete.
    for source,keys in valid_sources.items():
        if any(uid not in selected for uid,_ in keys):
            for uid,_ in keys:
                if uid in selected:
                    selected.pop(uid);dropped.append(dict(task_uid=uid,reason='shared capture includes quarantined episode',generations=[1],captured_decisions=captured.get((uid,1),0),reported_decisions=0))
    kept_sources=[s for s,keys in valid_sources.items() if all(uid in selected and selected[uid]['attempt']==a for uid,a in keys)]
    if any(uid not in planned for key in captured for uid in [key[0]]):raise ValueError('foreign capture episode')
    discarded=[]
    for drop in dropped:
        for attempt in drop['generations']:
            disposition='discarded_node_failure' if node_failure else 'discarded_missing_worker_result'
            discarded.append(dict(task_uid=drop['task_uid'],attempt=attempt,disposition=disposition,
                reason=drop['reason'],capture_decisions=captured.get((drop['task_uid'],attempt),0),
                reported_decisions=sum(r.get('task_uid')==drop['task_uid'] and r.get('attempt')==attempt for r in rows['per_step']),
                original_attempts=[r for r in rows['attempts'] if r.get('task_uid')==drop['task_uid'] and r.get('attempt')==attempt]))
    proof=dict(schema='r17.node_failure_salvage.v1',original=str(original),closure=closure,source_pins=source_pins(original),
        seal=seal_pins(campaign),alignment_digest=audit['digest'],capture_files=all_source_pins,
        kept_uids=sorted(selected),dropped_episodes=dropped,discarded_attempts=discarded,
        bad_sources=bad_sources,driver_issues=issues,transfers=transfers)
    quarantine=dict(schema='r17.capture_quarantine.v1',original=str(original),
        sources=[str(p) for p in sources if p not in kept_sources],dropped_episodes=dropped,
        policy='entire episode excluded; shared array blocks retained only as sealed evidence')
    write_json(destination/'QUARANTINE.json',quarantine)
    proof['capture_files'][str(destination/'QUARANTINE.json')]=file_sha(destination/'QUARANTINE.json')
    proof['digest']=digest(proof)
    destination.mkdir(parents=True,exist_ok=True)
    provenance=destination/'SALVAGE_PROVENANCE.json';write_json(provenance,proof)
    histories={}
    for row in discarded:histories.setdefault(row['task_uid'],[]).append(dict(row,salvage_proof=dict(path=str(provenance),digest=proof['digest'])))
    _,manifests=capture_counts(kept_sources,binding=binding)
    derived=dict(campaign,semantic_campaign=campaign.get('semantic_campaign',campaign['run_name']),
        node_failure_salvage=dict(provenance=str(provenance),digest=proof['digest']),node_failure_prior_attempts=histories,
        repair_merge=dict(capture_sources=list(map(str,kept_sources)),capture_manifests=manifests,provenance_digest=proof['digest']))
    write_json(destination/'campaign.json',derived)
    for name in ('journal','attempts','per_step','dispatch'):
        p=destination/'driver'/(name+'.jsonl');p.parent.mkdir(exist_ok=True)
        with p.open('w') as f:
            for row in rows[name]:
                if row.get('task_uid') in selected and row.get('attempt')==selected[row['task_uid']]['attempt']:f.write(json.dumps(row,sort_keys=True)+'\n')
    write_json(destination/'EPISODE_PLAN.json',recorded)
    write_json(destination/'RECEIPT.json',dict(status='FAIL',all_owned_closed=True,derived_collection=True,
        original_collection_status='FAIL',salvage_digest=proof['digest'],kept_episodes=len(selected),missing_episodes=len(tasks)-len(selected),
        transfers=dict(h100_servers=dict(verified=dict(status='verified',derived_from_proved_closed=True)),
            h100_shards=dict(status='verified',shards=[str(s) for s in kept_sources]))))
    verify_salvage(derived)
    return dict(status='PASS',cpu_only=True,salvaged=str(destination),kept_episodes=len(selected),missing_episodes=len(tasks)-len(selected),
        dropped_attempted_episodes=sum(bool(d['generations']) for d in dropped),unattempted_episodes=sum(not d['generations'] for d in dropped),digest=proof['digest'])


def salvage(directory, *, model, suite, remote, pull):
    from .library_staging import require_build_host
    from .sync_back import install_snapshot, atomic_bytes
    from .build_client_bundle import json_bytes
    from exp.r17_integration.qualification.archive_stage import stage_identity
    directory=Path(directory).resolve();campaign=json.loads((directory/'campaign.json').read_text())
    original=json.loads((directory/'RECEIPT.json').read_text())
    if not campaign.get('library') or campaign['smoke'] or campaign['server_node'] not in ('local','h100') or campaign['model']!=model or campaign['suite']!=suite or original['status']!='FAIL':
        raise ValueError('salvage requires matching FAILED full library campaign')
    require_build_host(campaign)
    node_failure='node_offline:' in str(original.get('error'))+json.dumps(original.get('cleanup',[]))
    if not node_failure and campaign['server_node']=='h100':
        raise ValueError('salvage requires explicit node-failure evidence')
    destination=directory/'salvage';destination.mkdir(exist_ok=True)
    closure=prove_closed(original,campaign,remote);write_json(destination/'CLOSURE.json',closure)
    if campaign['server_node']=='local':
        dispatched=evidence_rows(directory/'driver/dispatch.jsonl')[0]
        outcomes=evidence_rows(directory/'driver/journal.jsonl')[0]
        accepted={r['task_uid'] for r in outcomes if r.get('accepted') is True}
        missing={r['task_uid'] for r in dispatched if r['task_uid'] not in accepted}
        if not node_failure and not missing:raise ValueError('local salvage requires a dispatched missing worker result')
        sources=capture_sources(campaign,directory)
        result=reconstruct(directory,destination,campaign,sources,closure,{})
        write_json(destination/'SALVAGE_RESULT.json',result)
        return result
    from .remote_json import stage_json
    from .formal_eval import transfer as push
    control_reference=None
    def call(action,shard_id=None,receipt=None):
        nonlocal control_reference
        # Send only this stdlib helper to the old sealed runtime, without editing
        # or deploying any shared bundle used by Fast-WAM.
        payload=repr(campaign)
        if len(payload.encode())>8192:
            if control_reference is None:
                control_reference=stage_json('h100',campaign,destination,
                    '/tmp/r17_formal/'+campaign['run_name']+'/server',remote=remote,transfer=push)
            payload=('json.loads(Path('+repr(control_reference['path'])+').read_text())')
        verify=('assert hashlib.sha256(Path('+repr(control_reference['path'])+').read_bytes()).hexdigest()=='+
                repr(control_reference['sha256'])+',"salvage control SHA mismatch"\n') if control_reference else ''
        code='import json,re,hashlib;from pathlib import Path\n'+verify+inspect.getsource(remote_shards).replace('from .','from exp.r17_integration.ops.')+'\nprint(json.dumps(remote_shards('+payload+','+repr(action)+','+repr(shard_id)+','+repr(receipt)+')))'
        from .formal_eval import pythonpath
        return json.loads(remote('h100',['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+pythonpath(campaign['runtime']),'python3','-c',code]))
    receipts_path=destination/'SALVAGE_TRANSFERS.json'
    receipts=json.loads(receipts_path.read_text()) if receipts_path.exists() else {}
    inventory=call('inventory')
    if not (destination/'H100_INITIAL_INVENTORY.json').exists():
        write_json(destination/'H100_INITIAL_INVENTORY.json',inventory)
    if inventory['stage'] and inventory['stage'].get('batch_id'):
        from .library_batch import install_batch
        stage=inventory['stage'];batch=stage.get('transfer')
        if not batch:raise ValueError('partial bounded batch stage; cannot invent its transfer')
        archive=destination/'batch_transfers'/(stage['batch_id']+'.tar');archive.parent.mkdir(exist_ok=True)
        try:
            pull(batch['archive'],archive)
            ack=install_batch(archive,destination,batch)
            atomic_bytes(archive.with_suffix('.receipt.json'),json_bytes(ack))
            for receipt in ack['receipts']:receipts[receipt['shard_id']]=dict(receipt=receipt)
            write_json(receipts_path,receipts)
            cleaned=call('batch_ack',receipt=ack)
            for receipt in ack['receipts']:receipts[receipt['shard_id']]['cleaned']=cleaned
            write_json(receipts_path,receipts);archive.unlink()
            inventory=call('inventory')
        except Exception as exc:
            failure=dict(campaign=campaign['run_name'],shard_id=stage['batch_id'],step='salvage_batch',
                         cause=repr(exc),stderr=str(getattr(exc,'stderr','')))
            write_json(destination/'SALVAGE_FAILURE.json',failure);print(json.dumps(failure),flush=True);raise
    ids=list(inventory['shards'])
    if inventory['stage']:ids.sort(key=lambda s:s!=inventory['stage']['shard_id'])
    for index,shard_id in enumerate(ids):
        context=dict(shard_id=shard_id,step='stage',campaign=campaign['run_name'])
        try:
            transfer=call('stage',shard_id)
            if stage_identity(transfer['archive'],campaign['run_name']+'_shards','workers')!=transfer['archive_sha256'] or transfer['campaign']!=campaign['run_name'] or transfer['relative']!=inventory['shards'][shard_id]['relative']:
                raise ValueError('foreign salvage stage identity')
            archive=destination/'transfers'/(shard_id+'.tar');archive.parent.mkdir(exist_ok=True)
            context['step']='pull';pull(transfer['archive'],archive)
            context['step']='verify_install';target=destination/'servers'/transfer['relative']
            verified=install_snapshot(archive,target,expected_sha256=transfer['archive_sha256'])
            receipt=dict(schema='r17.salvage_shard_receipt.v1',status='verified',campaign=campaign['run_name'],shard_id=shard_id,
                archive_sha256=transfer['archive_sha256'],snapshot_digest=verified['digest'],partials=verified['partials'],
                build_host='weilandserver',destination=str(target),original_status=inventory['shards'][shard_id]['status'])
            context['step']='persist_receipt';atomic_bytes(archive.with_suffix('.receipt.json'),json_bytes(receipt))
            receipts[shard_id]=dict(receipt=receipt);write_json(receipts_path,receipts)
            context['step']='ack_cleanup';receipts[shard_id]['cleaned']=call('ack',shard_id,receipt)
            write_json(receipts_path,receipts);archive.unlink()
            if (index+1)%10==0 or index+1==len(ids):print('salvage verified/cleaned shards '+str(index+1)+'/'+str(len(ids)),flush=True)
        except Exception as exc:
            failure=dict(context,cause=repr(exc),stderr=str(getattr(exc,'stderr','')))
            write_json(destination/'SALVAGE_FAILURE.json',failure);print(json.dumps(failure),flush=True);raise
    final=call('inventory');write_json(destination/'H100_FINAL_INVENTORY.json',final)
    if final['shards'] or final['stage']:raise ValueError('salvage did not drain all owned shards')
    sources=capture_sources(campaign,directory)+capture_sources(campaign,destination)
    # Include partial bindings: their failures and bytes are quarantine evidence.
    sources+=list((destination/'servers').glob('replica_*/capture/shards/*/binding_*'))
    prior=json.loads((directory/'H100_SHARD_TRANSFERS.json').read_text()).get('shards',{})
    all_receipts=dict(prior,**receipts)
    verify_received(sources,campaign,all_receipts)
    result=reconstruct(directory,destination,campaign,sources,closure,all_receipts)
    write_json(destination/'SALVAGE_RESULT.json',result)
    return result
