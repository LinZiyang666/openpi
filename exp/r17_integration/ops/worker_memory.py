"""Cooperative node-memory admission; observe other fleets, never signal them."""
import json
from pathlib import Path

from .formal_process import alive

MIB = 1024*1024
RAM_HEADROOM_MIB = 20480
CACHE_RAM_MIB = 3584


def reservation_mib(campaign):
    """Explicit admission policy overrides the legacy fleet receipt's RAM floor."""
    mib = campaign.get('worker_ram_reservation_mib',
        campaign.get('worker_memory_mib', 4096 if campaign['domain'] == 'libero' else 2048))
    if type(mib) is not int or mib < 1:
        raise ValueError('positive integer worker RAM reservation required')
    if campaign.get('cache_study'):
        mib = max(mib, CACHE_RAM_MIB)
    return mib


def capacity():
    fields = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable'):
            fields[key] = int(value.split()[0])*1024
    total, available = fields['MemTotal'], fields['MemAvailable']
    # Account for both cgroup generations, without altering their limits.
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _, controllers, relative = line.split(':', 2)
        if controllers == '':
            root = Path('/sys/fs/cgroup')/relative.lstrip('/')
            pairs = [('memory.max', 'memory.current')]
        elif 'memory' in controllers.split(','):
            root = Path('/sys/fs/cgroup/memory')/relative.lstrip('/')
            pairs = [('memory.limit_in_bytes', 'memory.usage_in_bytes')]
        else:
            continue
        for limit, used in pairs:
            try:
                bound = int((root/limit).read_text())
                total = min(total, bound)
                available = min(available, max(0, bound-int((root/used).read_text())))
            except (FileNotFoundError, ValueError):
                pass
    return total, available


def rss(entry):
    if not entry.get('start_ticks') or not alive(entry):
        return None
    try:
        values = Path('/proc/'+str(entry['pid'])+'/status').read_text().splitlines()
        return next(int(line.split()[1])*1024 for line in values if line.startswith('VmRSS:'))
    except (FileNotFoundError, ProcessLookupError, StopIteration):
        return None


def admission(campaign, run, *, memory=None, rss_reader=rss, replacement=False):
    run = Path(run)
    total, available = capacity() if memory is None else memory
    mib = reservation_mib(campaign)
    if type(campaign['node_workers']) is not int or campaign['node_workers'] < 1:
        raise ValueError('positive integer worker memory/count required')
    per_worker = mib*MIB
    floor = campaign.get('worker_memory_headroom_mib', 0)
    if type(floor) is not int or floor < 0:
        raise ValueError('nonnegative worker RAM headroom required')
    headroom = max(RAM_HEADROOM_MIB, floor)*MIB
    pending, observed = 0, []
    for path in run.parent.glob('*/receipt.json'):
        own = path.parent == run
        if own and not replacement:
            continue
        try:
            receipt = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if receipt.get('status') not in ('RUNNING','STARTING','PARTIAL'):
            continue
        amounts = [value for p in receipt.get('pids', []) if (value := rss_reader(p)) is not None]
        owner = receipt.get('fleet_owner')
        live_owner = owner is not None and rss_reader(owner) is not None
        count = max(receipt.get('workers', len(amounts)), len(amounts)) if live_owner else len(amounts)
        if own:
            count = max(0, count-1)  # Replacement has a separate requested reservation.
        if not count:
            continue
        config = dict(domain='libero', worker_memory_mib=receipt.get('worker_memory_mib', 4096))
        manifest = path.parent/'manifest.json'
        if manifest.exists():
            config.update(json.loads(manifest.read_text()))
            # The explicit CPU-prepared reservation is authoritative even when
            # an unchanged legacy fleet helper publishes its historical floor.
            if 'worker_ram_reservation_mib' not in config:
                config['worker_memory_mib'] = max(config.get('worker_memory_mib', 0),
                    receipt.get('worker_memory_mib', 0))
        reserve = reservation_mib(config)*MIB
        # MemAvailable already reflects RSS. Charge only the missing reservation
        # for each worker, including owner-held slots between N4 generations.
        # One oversized worker must not cancel another worker's reservation.
        accounted = sum(max(reserve, value) for value in amounts) + max(0, count-len(amounts))*reserve
        future = max(0, accounted-sum(amounts))
        pending += future
        observed.append(dict(receipt=str(path), workers=count, rss_bytes=sum(amounts), pending_bytes=future,
                             reservation_bytes=count*reserve, accounted_bytes=accounted,
                             worker_memory_mib=reserve//MIB, own_fleet=own))
    requested = campaign['node_workers']*per_worker
    result = dict(total_bytes=total, available_bytes=available, headroom_bytes=headroom,
                  other_pending_bytes=pending, requested_bytes=requested, other_fleets=observed,
                  worker_memory_mib=per_worker//MIB, admitted=available-requested-pending >= headroom)
    if not result['admitted']:
        result['reason'] = ('node worker memory admission refused: requested '+str(campaign['node_workers'])+
            ' workers at '+str(per_worker//MIB)+' MiB each; available after fleet reservations/headroom '+
            str(max(0, available-pending-headroom)//MIB)+' MiB; reduce --workers or use another node')
    return result
