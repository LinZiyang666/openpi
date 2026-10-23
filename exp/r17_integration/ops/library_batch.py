"""Bounded multi-shard /tmp snapshots; one pull, exact per-shard receipts."""
import hashlib
import json
import os
from pathlib import Path
import time

from .build_client_bundle import json_bytes, sha256
from .sync_back import atomic_bytes, snapshot_run, install_snapshot


def remove_tree(root):
    """Only a caller-validated, exact campaign stage; no blanket deletion."""
    for path in sorted(root.rglob('*'),key=lambda p:len(p.parts),reverse=True):
        if path.is_symlink():raise ValueError('symlink batch stage')
        if path.is_file():path.unlink()
        elif path.is_dir():path.rmdir()
    root.rmdir()


def budget_for(campaign, *, salvage=False):
    from .library_staging import ShardBudget,policy
    if not salvage:return ShardBudget(campaign)
    budget=ShardBudget.__new__(ShardBudget)
    budget.campaign,budget.limits=campaign,policy(campaign)
    budget.root=Path(campaign['server_run']);budget.state_path=budget.root/'SHARD_BUDGET.json'
    if not budget.state_path.is_file():raise ValueError('missing batch salvage quota')
    return budget


def batch_ack(budget, state, acknowledgement):
    from exp.r17_integration.qualification.archive_stage import cleanup_stage
    batch_id=acknowledgement['batch_id']
    if not isinstance(batch_id,str) or len(batch_id)!=64 or any(c not in '0123456789abcdef' for c in batch_id):
        raise ValueError('invalid batch ack identity')
    record=budget.root/'BATCH_RECEIPTS'/(batch_id+'.json')
    stage=state['stage']
    if not stage or stage.get('batch_id')!=batch_id:
        if record.is_file() and not record.is_symlink() and json.loads(record.read_text())==acknowledgement:
            return dict(status='PASS',batch_id=batch_id,already_cleaned=True)
        raise ValueError('batch ack lacks exact owned transfer')
    transfer=stage['transfer'];receipts=acknowledgement['receipts']
    if {r['shard_id'] for r in receipts}!=set(transfer['shards']) or len(receipts)!=len(transfer['shards']):
        raise ValueError('batch ack must receive every shard exactly once')
    for receipt in receipts:
        shard_id=receipt['shard_id'];meta=transfer['shards'][shard_id]
        if (receipt.get('schema')!='r17.h100_shard_receipt.v1' or receipt.get('status')!='verified' or
            receipt.get('campaign')!=budget.campaign['run_name'] or receipt.get('archive_sha256')!=transfer['archive_sha256'] or
            receipt.get('snapshot_digest')!=meta['manifest']['digest'] or receipt.get('partials')!=[]):
            raise ValueError('missing/mismatched SHA-verified batch shard receipt')
    source=Path(transfer['source_archive']);tree=budget.root/'STAGED_BATCHES'/batch_id
    from .library_staging import TRANSFER_ROOT
    if source!=TRANSFER_ROOT/budget.campaign['run_name']/'server/batches'/(batch_id+'.tar'):
        raise ValueError('batch source archive belongs to another transfer')
    atomic_bytes(record,json_bytes(acknowledgement))
    for receipt in receipts:atomic_bytes(budget.root/'SHARD_RECEIPTS'/(receipt['shard_id']+'.json'),json_bytes(receipt))
    cleanup_stage(transfer['archive'],budget.campaign['run_name']+'_shards','workers',transfer['archive_sha256'])
    source.unlink();remove_tree(tree)
    for receipt in receipts:
        shard_id=receipt['shard_id'];remove_tree(budget.path(state['shards'][shard_id]));del state['shards'][shard_id]
    state['stage']=None
    return dict(status='PASS',batch_id=batch_id,removed_after_verified_receipt=True,shards=len(receipts))


def next_batch(campaign, acknowledgement=None, force=False, *, salvage=False):
    from .library_staging import TRANSFER_ROOT
    from exp.r17_integration.qualification.archive_stage import stage_archive
    budget=budget_for(campaign,salvage=salvage)
    with budget.locked() as state:
        if not salvage:budget._check(state)
        cleaned=batch_ack(budget,state,acknowledgement) if acknowledgement else None
        result=dict(status='PASS',active=sum(e['status']=='writing' for e in state['shards'].values()),
            high_water_reserved_bytes=state['high_water_reserved_bytes'],cleaned=cleaned,transfer=None)
        if state['stage']:
            if 'batch_id' not in state['stage'] or not state['stage'].get('transfer'):
                raise ValueError('another/partial transfer occupies batch staging')
            return dict(result,transfer=state['stage']['transfer'])
        closed=[s for s,e in state['shards'].items() if e['status']=='closed']
        if not closed:
            state.pop('batch_wait_started',None);return result
        started=state.setdefault('batch_wait_started',time.time())
        if not force and len(closed)<4 and time.time()-started<120:return result
        # Data blocks are hardlinked below the same /data campaign root, so
        # only the source tar and Q relay tar consume transfer byte headroom.
        limit=min(511*1024**2,(budget.limits['transfer_reserve_bytes']-1024**2)//2)
        chosen={};estimate=16384
        for shard_id in sorted(closed):
            leaf=budget.path(state['shards'][shard_id]);files=[p for p in leaf.rglob('*') if p.is_file() and p.name!='.eval.lock']
            # Include tar/PAX padding and both per-shard/outer SHA manifests.
            # Raw payload bytes alone underestimate a many-file capture.
            size=sum(p.stat().st_size+4096+4*len(str(p.relative_to(leaf)).encode()) for p in files)+8192
            if estimate+size>limit:continue
            manifest=dict(schema='r17.snapshot.v1',files={str(p.relative_to(leaf)):sha256(p) for p in files},partials=[])
            if any(p.name.endswith(('.part','.tmp','.failed')) for p in files):raise ValueError('partial closed batch shard '+shard_id)
            manifest['digest']=hashlib.sha256(json_bytes(manifest)).hexdigest()
            chosen[shard_id]=dict(relative=state['shards'][shard_id]['relative'],manifest=manifest)
            estimate+=size
        if not chosen:raise ValueError('closed shards cannot fit bounded batch transfer: '+repr(closed))
        batch_id=hashlib.sha256(json_bytes(chosen)).hexdigest()
        state['stage']=dict(batch_id=batch_id,shard_ids=list(chosen))
        atomic_bytes(budget.state_path,json_bytes(state))
        tree=budget.root/'STAGED_BATCHES'/batch_id;tree.mkdir(parents=True,exist_ok=False)
        atomic_bytes(tree/'manifest.json',json_bytes(dict(schema='r17.shard_batch.v1',campaign=campaign['run_name'],batch_id=batch_id,shards=chosen)))
        for shard_id,meta in chosen.items():
            leaf=budget.path(state['shards'][shard_id]);target=tree/meta['relative']
            for name in meta['manifest']['files']:
                dst=target/name;dst.parent.mkdir(parents=True,exist_ok=True);os.link(leaf/name,dst)
            atomic_bytes(target/'TRANSFER_SHA256.json',json_bytes(meta['manifest']))
        archive=TRANSFER_ROOT/campaign['run_name']/'server/batches'/(batch_id+'.tar')
        transfer=snapshot_run(tree,archive)
        if archive.stat().st_size>limit:raise ValueError('batch '+batch_id+' archive exceeded reserved /tmp bytes')
        transfer.update(stage_archive(archive,campaign['run_name']+'_shards','workers',transfer['archive_sha256']))
        transfer.update(batch_id=batch_id,campaign=campaign['run_name'],shards=chosen)
        state['stage']['transfer']=transfer;state.pop('batch_wait_started',None)
        return dict(result,transfer=transfer)


def install_batch(archive, directory, transfer):
    from exp.r17_integration.qualification.archive_stage import stage_identity
    if stage_identity(transfer['archive'],transfer['campaign']+'_shards','workers')!=transfer['archive_sha256']:
        raise ValueError('batch relay stage identity differs')
    received=Path(directory)/'batch_transfers'/transfer['batch_id']/'received'
    verified=install_snapshot(archive,received,expected_sha256=transfer['archive_sha256'])
    if verified['partials']:raise ValueError('partial batch cannot release remote capture')
    manifest=json.loads((received/'manifest.json').read_text())
    if manifest!=dict(schema='r17.shard_batch.v1',campaign=transfer['campaign'],batch_id=transfer['batch_id'],shards=transfer['shards']):
        raise ValueError('batch manifest differs from transfer description')
    receipts=[]
    for shard_id,meta in transfer['shards'].items():
        relative=Path(meta['relative'])
        import re
        if (relative.is_absolute() or '..' in relative.parts or len(relative.parts)!=4 or
            not re.fullmatch(r'replica_[0-9]+',relative.parts[0]) or relative.parts[1:3]!=('capture','shards') or relative.name!=shard_id):
            raise ValueError('batch shard escaped exact replica/campaign')
        leaf=received/relative;target=Path(directory)/'servers'/relative
        snapshot=json.loads((leaf/'TRANSFER_SHA256.json').read_text())
        if snapshot!=meta['manifest'] or snapshot['partials']:
            raise ValueError('batch shard snapshot digest differs')
        if target.exists():
            if {str(p.relative_to(target)):sha256(p) for p in target.rglob('*') if p.is_file() and p.name!='TRANSFER_SHA256.json'}!=snapshot['files']:
                raise ValueError('batch destination differs')
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            # Keep the complete installed batch as restart proof until ack;
            # hardlinks avoid a second permanent copy of large camera blocks.
            for p in leaf.rglob('*'):
                if p.is_file():
                    dst=target/p.relative_to(leaf);dst.parent.mkdir(parents=True,exist_ok=True);os.link(p,dst)
        receipts.append(dict(schema='r17.h100_shard_receipt.v1',status='verified',campaign=transfer['campaign'],
            shard_id=shard_id,archive_sha256=transfer['archive_sha256'],snapshot_digest=snapshot['digest'],
            partials=[],destination=str(target),build_host='weilandserver'))
    return dict(batch_id=transfer['batch_id'],receipts=receipts)
