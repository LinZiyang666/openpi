"""Bounded, receipt-owned H100 teacher shards; CPU-only control/transfer path."""
from __future__ import annotations

from contextvars import copy_context

from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import re
import shutil
import socket
import sys
import threading
import time

from .build_client_bundle import json_bytes, sha256
from .sync_back import atomic_bytes, install_snapshot, snapshot_run
from .tether_retry import RETRY_SECONDS, current_journal, current_step, retry

GIB = 1024**3
DEFAULT_POLICY = dict(limit_bytes=20_000_000_000, shard_bytes=256*1024**2,
    transfer_reserve_bytes=GIB, metadata_reserve_bytes=GIB,
    min_free_data_bytes=10*GIB, min_free_tmp_bytes=2*GIB)
TRANSFER_ROOT = Path('/tmp/r17_formal')


def bounded(campaign):
    return campaign.get('library') is True and campaign.get('server_node')=='h100'


def policy(campaign):
    value = campaign.get('h100_staging')
    if not isinstance(value, dict) or set(value)!=set(DEFAULT_POLICY):
        raise ValueError('H100 library requires the bounded staging policy')
    if any(type(n) is not int or n<=0 for n in value.values()):
        raise ValueError('positive integer staging limits required')
    if (value['limit_bytes']>20_000_000_000 or value['shard_bytes']>256*1024**2 or
        value['transfer_reserve_bytes']<2*value['shard_bytes']+1024**2 or
        value['shard_bytes']+value['transfer_reserve_bytes']+value['metadata_reserve_bytes']>value['limit_bytes']):
        raise ValueError('unsafe H100 capture/staging bounds')
    return value


def admit(campaign, *, data='/data', temporary='/tmp', usage=None):
    """Called before remote runtime deployment and again before GPU startup."""
    limits = policy(campaign)
    usage = usage or shutil.disk_usage
    free_data, free_tmp = usage(data).free, usage(temporary).free
    if (free_data<limits['min_free_data_bytes']+limits['limit_bytes'] or
        free_tmp<limits['min_free_tmp_bytes']+limits['transfer_reserve_bytes']):
        raise RuntimeError('H100 free-space admission failed; no resident may start')
    return dict(status='PASS', free_data_bytes=free_data, free_tmp_bytes=free_tmp, policy=limits)


def require_build_host(campaign):
    if bounded(campaign) and socket.gethostname().split('.')[0]!='weilandserver':
        raise ValueError('H100 library parent/fits must be built only on weilandserver')


class ShardBudget:
    """One flock-protected quota across all replicas and their connections.

    Active episodes retain a full reservation so they can always finish. After
    closure, immutable bytes plus bounded metadata slack remain reserved until
    verified receipt; unused capacity is available to the next episode.
    Two transfer copies and control metadata have separate reserved headroom.
    The conservative charge is checked BEFORE each writer submission, so queued
    buffers and asynchronous block writes cannot overshoot the disk budget.
    """
    def __init__(self, campaign, *, temporary='/tmp', usage=None):
        self.campaign, self.limits = campaign, policy(campaign)
        self.root = Path(campaign['server_run'])
        self.root.mkdir(parents=True,exist_ok=True)
        if self.root.resolve()!=self.root:
            raise ValueError('symlink/relative H100 capture root')
        self.temporary, self.usage = Path(temporary), usage or shutil.disk_usage
        self.state_path = self.root/'SHARD_BUDGET.json'
        with self.locked() as state:
            self._check(state)

    @contextmanager
    def locked(self):
        with (self.root/'.shard_budget.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state = json.loads(self.state_path.read_text()) if self.state_path.exists() else dict(
                schema='r17.h100_shard_budget.v1', campaign=self.campaign['run_name'],
                policy=self.limits, shards={}, stage=None, failure=None, high_water_reserved_bytes=0)
            if state['campaign']!=self.campaign['run_name'] or state['policy']!=self.limits:
                raise ValueError('capture quota belongs to another campaign/policy')
            try:
                yield state
            except Exception as exc:
                state['failure'] = state['failure'] or repr(exc)
                raise
            finally:
                atomic_bytes(self.state_path,json_bytes(state))

    def _check(self, state):
        if state['failure']:
            raise RuntimeError('H100 capture stopped: '+state['failure'])
        free_data, free_tmp = self.usage(self.root).free, self.usage(self.temporary).free
        if (free_data<self.limits['min_free_data_bytes']+self.limits['shard_bytes'] or
            free_tmp<self.limits['min_free_tmp_bytes']+self.limits['transfer_reserve_bytes']):
            raise RuntimeError('H100 free space fell below the capture reserve')
        state.update(last_free_data_bytes=free_data,last_free_tmp_bytes=free_tmp)

    def check(self):
        with self.locked() as state:
            self._check(state)
            return dict(state)

    def path(self, entry):
        relative = Path(entry['relative'])
        if (relative.is_absolute() or '..' in relative.parts or len(relative.parts)!=4 or
            not re.fullmatch(r'replica_[0-9]+',relative.parts[0]) or
            relative.parts[1:3]!=('capture','shards') or not re.fullmatch('[0-9a-f]{64}',relative.name)):
            raise ValueError('shard is outside its exact replica/campaign')
        result = self.root/relative
        if result.resolve()!=result:
            raise ValueError('symlink shard path')
        return result

    def reserved(self, state):
        return (sum(e.get('reserved_bytes',self.limits['shard_bytes']) for e in state['shards'].values())+
                self.limits['transfer_reserve_bytes']+self.limits['metadata_reserve_bytes'])

    def reserve(self, shard_id, directory, *, wait_seconds=RETRY_SECONDS+90, poll_seconds=.2):
        """Capacity is backpressure, not a fatal error while transfers progress.

        Poll without holding the flock so active writers can close and the
        independent coordinator pump can acknowledge/delete verified shards.
        Disk/identity errors still latch immediately; a bounded stall fails closed.
        """
        if not 0<=wait_seconds<=RETRY_SECONDS+90 or not 0<poll_seconds<=5 or not math.isfinite(poll_seconds):
            raise ValueError('bounded nonnegative reservation wait required')
        deadline = time.monotonic()+wait_seconds
        while True:
            with self.locked() as state:
                self._check(state)
                entry = dict(relative=str(Path(directory).relative_to(self.root)),status='writing',
                    charged_bytes=65536,reserved_bytes=self.limits['shard_bytes'])
                self.path(entry)
                if shard_id!=Path(entry['relative']).name or shard_id in state['shards']:
                    raise ValueError('duplicate/colliding teacher episode shard')
                total = self.reserved(state)+self.limits['shard_bytes']
                if total<=self.limits['limit_bytes']:
                    state['shards'][shard_id] = entry
                    state['high_water_reserved_bytes'] = max(state['high_water_reserved_bytes'],total)
                    return
                if time.monotonic()>=deadline:
                    raise RuntimeError(f'H100 capture bound reached after {wait_seconds}s backpressure: '
                        f'shard={shard_id}, reserved={self.reserved(state)}, requested={self.limits["shard_bytes"]}, '
                        f'limit={self.limits["limit_bytes"]}, active='+str(sum(e['status']=='writing' for e in state['shards'].values())))
                state['backpressure_events'] = state.get('backpressure_events',0)+1
            time.sleep(min(poll_seconds,max(0,deadline-time.monotonic())))

    def charge(self, shard_id, size):
        with self.locked() as state:
            self._check(state)
            entry = state['shards'][shard_id]
            if entry['status']!='writing' or type(size) is not int or size<0:
                raise ValueError('invalid active-shard charge')
            if entry['charged_bytes']+size>self.limits['shard_bytes']:
                raise RuntimeError('teacher episode exceeds bounded capture shard')
            entry['charged_bytes'] += size

    def closed(self, shard_id):
        with self.locked() as state:
            entry = state['shards'][shard_id]
            leaf = self.path(entry)
            capture = list(leaf.glob('binding_*/r17_capture.json'))
            if len(capture)!=1 or json.loads(capture[0].read_text())['status']!='closed':
                raise ValueError('only drained, closed captures may be streamed')
            size = sum(p.stat().st_size for p in leaf.rglob('*') if p.is_file())
            if size>entry['charged_bytes'] or size>self.limits['shard_bytes']:
                raise RuntimeError('capture exceeded conservative shard reservation')
            entry.update(status='closed',capture_sha256=sha256(capture[0]),actual_bytes=size)
            # A closed writer is immutable; reserve rounded actual bytes plus
            #64KiB, rather than256MiB for every tiny Cosmos episode in the queue.
            mib = 1024**2
            entry['reserved_bytes'] = min(self.limits['shard_bytes'],((size+65536+mib-1)//mib)*mib)


def pending(campaign):
    budget = ShardBudget(campaign)
    state = budget.check()
    return dict(status='PASS',shards={k:v for k,v in state['shards'].items() if v['status']=='closed'},
        active=sum(v['status']=='writing' for v in state['shards'].values()),
        high_water_reserved_bytes=state['high_water_reserved_bytes'])


def stage_shard(campaign, shard_id):
    from exp.r17_integration.qualification.archive_stage import stage_archive
    budget = ShardBudget(campaign)
    with budget.locked() as state:
        budget._check(state)
        entry = state['shards'][shard_id]
        if entry['status']!='closed':
            raise ValueError('open teacher shard cannot be snapshotted')
        if state['stage']:
            if state['stage']['shard_id']!=shard_id or not state['stage'].get('transfer'):
                raise RuntimeError('another/partial transfer occupies bounded staging')
            return state['stage']['transfer']
        # Claim before writing; a failed snapshot cannot be retried into more copies.
        state['stage'] = dict(shard_id=shard_id)
        atomic_bytes(budget.state_path,json_bytes(state))
        leaf = budget.path(entry)
        archive = TRANSFER_ROOT/campaign['run_name']/'server/shards'/(shard_id+'.tar')
        result = snapshot_run(leaf,archive)
        if archive.stat().st_size>budget.limits['shard_bytes']:
            raise RuntimeError('closed shard archive exceeds transfer reservation')
        result.update(stage_archive(archive,campaign['run_name']+'_shards','workers',result['archive_sha256']))
        result.update(shard_id=shard_id,relative=entry['relative'],campaign=campaign['run_name'])
        state['stage']['transfer'] = result
        return result


def acknowledge_shard(campaign, shard_id, receipt):
    """Only called with the durable weilandserver install receipt, never on pull."""
    from exp.r17_integration.qualification.archive_stage import cleanup_stage
    budget = ShardBudget(campaign)
    with budget.locked() as state:
        if not re.fullmatch('[0-9a-f]{64}',shard_id):
            raise ValueError('invalid shard receipt identity')
        record = budget.root/'SHARD_RECEIPTS'/(shard_id+'.json')
        if shard_id not in state['shards'] and record.exists():
            if record.is_symlink() or json.loads(record.read_text())!=receipt:
                raise ValueError('replayed shard receipt differs from durable verified receipt')
            return dict(status='PASS',shard_id=shard_id,removed_after_verified_receipt=True,already_cleaned=True)
        stage = state['stage']
        if not stage or stage['shard_id']!=shard_id:
            raise ValueError('receipt has no matching owned transfer')
        transfer = stage['transfer']
        if (receipt.get('schema')!='r17.h100_shard_receipt.v1' or receipt.get('status')!='verified' or
            receipt.get('campaign')!=campaign['run_name'] or receipt.get('shard_id')!=shard_id or
            receipt.get('archive_sha256')!=transfer['archive_sha256'] or
            receipt.get('snapshot_digest')!=transfer['manifest']['digest'] or receipt.get('partials')!=[] or
            receipt.get('build_host')!='weilandserver'):
            raise ValueError('missing/mismatched verified shard receipt; remote copy retained')
        leaf = budget.path(state['shards'][shard_id])
        actual = {str(p.relative_to(leaf)):sha256(p) for p in leaf.rglob('*') if p.is_file() and p.name!='.eval.lock'}
        if actual!=transfer['manifest']['files'] or any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in leaf.rglob('*')):
            raise ValueError('closed shard changed after transfer; remote copy retained')
        source = Path(transfer['source_archive'])
        expected_source = TRANSFER_ROOT/campaign['run_name']/'server/shards'/(shard_id+'.tar')
        if source!=expected_source or source.resolve()!=source or sha256(source)!=transfer['archive_sha256']:
            raise ValueError('source archive ownership/SHA changed')
        # Persist the receipt before deleting any received byte. A lost reply
        # after quota commit can be replayed against this exact receipt.
        atomic_bytes(record,json_bytes(receipt))
        cleanup_stage(transfer['archive'],campaign['run_name']+'_shards','workers',transfer['archive_sha256'])
        source.unlink()
        # Enumerate only this immutable leaf; no recursive blanket deletion.
        for path in sorted(leaf.rglob('*'),key=lambda p:len(p.parts),reverse=True):
            if path.is_file(): path.unlink()
            elif path.is_dir(): path.rmdir()
            else: raise ValueError('nonregular owned capture file')
        leaf.rmdir()
        del state['shards'][shard_id]
        state['stage'] = None
        return dict(status='PASS',shard_id=shard_id,removed_after_verified_receipt=True)


def batch_next(campaign, acknowledgement=None, force=False):
    from .library_batch import next_batch
    return next_batch(campaign,acknowledgement,force)


class ShardPump:
    """Coordinator background transfer; leases/status checks stay responsive."""
    def __init__(self,campaign,directory,remote_call,pull):
        self.campaign,self.directory = campaign,Path(directory)
        def traced(operation, label, *args):
            token = current_step.set(dict(self.context))
            try:
                return retry(lambda: operation(*args), label=label, node='h100',replay_safe=True)
            finally:
                current_step.reset(token)
        self.remote_call = lambda action,*args: traced(remote_call, 'shard '+action, action, *args)
        self.pull = lambda source,target: traced(pull, 'shard pull', source, target)
        self.stop = threading.Event()
        self.error = None
        self.failure = None
        self.context = dict(shard_id=None,step='pending')
        self.receipts = {}
        self.batch_ack = None
        self.batch_transfer = None
        self.batches = 0
        self.high_water_reserved_bytes = 0
        context = copy_context()
        self.thread = threading.Thread(target=lambda: context.run(self._run),name='h100-library-shards',daemon=True)

    def _failed(self, exc):
        """Persist/print the original exception, including subprocess stderr."""
        if self.error is exc:
            return
        self.error = exc
        chain, current, seen = [], exc, set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            row = dict(type=type(current).__name__,message=str(current))
            for field in ('stderr','stdout','returncode'):
                value = getattr(current,field,None)
                if isinstance(value,bytes):value=value.decode(errors='replace')
                if value is not None:row[field]=value
            chain.append(row)
            current = current.__cause__ or current.__context__
        self.failure = dict(schema='r17.h100_shard_failure.v1',status='FAIL',
            campaign=self.campaign['run_name'],**self.context,cause=repr(exc),exceptions=chain,
            stderr='\n'.join(str(e['stderr']) for e in chain if e.get('stderr')))
        try:
            atomic_bytes(self.directory/'H100_SHARD_FAILURE.json',json_bytes(self.failure))
            atomic_bytes(self.directory/'H100_SHARD_TRANSFERS.json',json_bytes(self.summary()))
        except Exception as logging_error:
            self.failure['receipt_write_error'] = repr(logging_error)
            try:
                atomic_bytes(self.directory/'H100_SHARD_FAILURE.json',json_bytes(self.failure))
            except Exception:
                pass  # stderr below still carries both concrete failures.
        print('H100 shard transfer failure: '+json_bytes(self.failure).decode(),file=sys.stderr,flush=True)

    def drain(self):
        try:
            return self._drain()
        except Exception as exc:
            self._failed(exc)
            raise

    def _drain(self):
        if self.campaign.get('h100_batch'):
            return self._drain_batch()
        from exp.r17_integration.qualification.archive_stage import stage_identity
        self.context = dict(shard_id=None,step='pending')
        value = self.remote_call('pending')
        self.high_water_reserved_bytes = max(self.high_water_reserved_bytes,value['high_water_reserved_bytes'])
        for shard_id in value['shards']:
            if self.stop.is_set(): break
            self.context = dict(shard_id=shard_id,step='stage_shard')
            transfer = self.remote_call('stage_shard',shard_id)
            self.context.update(step='validate_stage',remote_archive=transfer.get('archive'))
            relative = Path(transfer['relative'])
            if (transfer['campaign']!=self.campaign['run_name'] or transfer['shard_id']!=shard_id or
                relative.is_absolute() or '..' in relative.parts or len(relative.parts)!=4 or
                not re.fullmatch(r'replica_[0-9]+',relative.parts[0]) or
                relative.parts[1:3]!=('capture','shards') or relative.name!=shard_id or
                stage_identity(transfer['archive'],self.campaign['run_name']+'_shards','workers')!=transfer['archive_sha256']):
                raise ValueError('remote transfer escaped the exact campaign/shard stage')
            destination = self.directory/'servers'/transfer['relative']
            archive = self.directory/'shard_transfers'/(shard_id+'.tar')
            archive.parent.mkdir(parents=True,exist_ok=True)
            self.context.update(step='pull',local_archive=str(archive))
            self.pull(transfer['archive'],archive)
            self.context['step'] = 'install_snapshot'
            verified = install_snapshot(archive,destination,expected_sha256=transfer['archive_sha256'])
            if verified['partials']:
                raise ValueError('partial shard may not release H100 storage')
            receipt = dict(schema='r17.h100_shard_receipt.v1',status='verified',campaign=self.campaign['run_name'],
                shard_id=shard_id,archive_sha256=transfer['archive_sha256'],snapshot_digest=verified['digest'],
                partials=verified['partials'],destination=str(destination),build_host='weilandserver')
            self.context['step'] = 'persist_receipt'
            atomic_bytes(archive.with_suffix('.receipt.json'),json_bytes(receipt))
            self.context['step'] = 'acknowledge_shard'
            cleaned = self.remote_call('acknowledge_shard',shard_id,receipt)
            self.receipts[shard_id] = dict(receipt=receipt,cleaned=cleaned)
            self.context['step'] = 'persist_summary'
            atomic_bytes(self.directory/'H100_SHARD_TRANSFERS.json',json_bytes(self.summary()))
            # Durable installed arrays + receipt remain on weilandserver.
            self.context['step'] = 'remove_local_archive'
            archive.unlink()
        return value

    def _drain_batch(self, force=False):
        from .library_batch import install_batch
        while True:
            self.context=dict(shard_id=self.batch_ack['batch_id'] if self.batch_ack else None,step='batch_next')
            value=self.remote_call('batch_next',self.batch_ack,force)
            self.high_water_reserved_bytes=max(self.high_water_reserved_bytes,value['high_water_reserved_bytes'])
            if self.batch_ack:
                for receipt in self.batch_ack['receipts']:
                    self.receipts[receipt['shard_id']]=dict(receipt=receipt,cleaned=value['cleaned'])
                archive=self.directory/'batch_transfers'/(self.batch_ack['batch_id']+'.tar')
                if archive.exists():archive.unlink()
                self.batch_ack=None
                atomic_bytes(self.directory/'H100_SHARD_TRANSFERS.json',json_bytes(self.summary()))
            transfer=value['transfer']
            if transfer is None or self.stop.is_set():return value
            if transfer['campaign']!=self.campaign['run_name']:raise ValueError('foreign batch campaign')
            self.batch_transfer=transfer
            self.context=dict(shard_id=transfer['batch_id'],shard_ids=list(transfer['shards']),step='batch_pull')
            archive=self.directory/'batch_transfers'/(transfer['batch_id']+'.tar');archive.parent.mkdir(parents=True,exist_ok=True)
            self.pull(transfer['archive'],archive)
            self.context['step']='batch_install'
            ack=install_batch(archive,self.directory,transfer)
            for receipt in ack['receipts']:
                atomic_bytes(self.directory/'shard_transfers'/(receipt['shard_id']+'.receipt.json'),json_bytes(receipt))
            atomic_bytes(archive.with_suffix('.receipt.json'),json_bytes(ack))
            self.batch_ack=ack;self.batches+=1

    def summary(self):
        result = dict(status='FAIL' if self.failure else 'PARTIAL' if self.stop.is_set() else 'verified',
            schema='r17.h100_shard_transfers.v1',shards=dict(self.receipts),failure=self.failure,
            high_water_reserved_bytes=self.high_water_reserved_bytes,limit_bytes=policy(self.campaign)['limit_bytes'])
        result.update(batch_transfers=self.batches,poll_seconds=30 if self.campaign.get('h100_batch') else 2)
        if current_journal.get():
            result['tether_retry'] = current_journal.get().summary()
        return result

    def _run(self):
        try:
            while not self.stop.is_set():
                self.drain()
                self.stop.wait(30 if self.campaign.get('h100_batch') else 2)
        except Exception as exc:
            self._failed(exc)

    def start(self):
        self.thread.start()

    def check(self):
        if self.error:
            details = self.failure or dict(self.context,cause=repr(self.error),stderr='')
            raise RuntimeError('H100 bounded shard transfer failed: '
                f'shard={details["shard_id"]}, step={details["step"]}, cause={details["cause"]}; '
                'stderr='+details['stderr']) from self.error

    def finish(self):
        self.stop.set()
        # stage/pull/ack each have their own bounded transport retry and timeout.
        self.thread.join(timeout=3*RETRY_SECONDS+180)
        if self.thread.is_alive():
            exc = RuntimeError('H100 shard transfer did not close within its deadline; step='+self.context['step'])
            self._failed(exc)
            raise exc
        self.check()
        self.stop.clear()
        try:
            value = self._drain_batch(force=True) if self.campaign.get('h100_batch') else self.drain()
            self.context = dict(shard_id=None,step='finish_pending')
            if value['active']:
                raise ValueError('unacknowledged active teacher shards remain')
            final = self.remote_call('batch_next',None,True) if self.campaign.get('h100_batch') else self.remote_call('pending')
            if final['active'] or (final.get('transfer') if self.campaign.get('h100_batch') else final['shards']):
                raise ValueError('H100 capture was not fully drained to weilandserver')
        except Exception as exc:
            self._failed(exc)
            raise
        return self.summary()

    def abort(self):
        self.stop.set()
        self.thread.join(timeout=3*RETRY_SECONDS+180)
        if self.thread.is_alive():
            exc = RuntimeError('H100 shard transfer abort deadline exceeded; step='+self.context['step'])
            self._failed(exc)
            raise exc
