"""Worker eligibility, hard caps and the batch fields of the central scheduler (R17).

Prepare: apply() describes the batch for its BatchClient and NodeExecutors.
Launch: require() checks that the scheduler answers.
"""
import json
import os
from pathlib import Path
import urllib.request

GIB = 1024 ** 3
URL = 'http://149.165.152.232:23500'  # the A100 (owner 10-19)
PACKAGE = 'exp/offline_search/rounds/r17/scheduler'
ROOT = Path(__file__).resolve().parents[5]
# Shared pool: any batch may use these. Every worker machine has every worker environment (owner 10-20).
SHARED = {'a100': ('libero', 'metaworld'), 'timan1': ('libero', 'metaworld'), 'timan107': ('libero', 'metaworld')}
# Local class: renderers on the server's own machine serve only that server.
# Servers on h100-temp (Xid 31 renderer faults) and jupyter pods get none.
LOCAL = {'h100': ('libero', 'metaworld'), 'local': ('libero', 'metaworld')}
# NodeExecutor RAM safety lines, bytes. A100: worker-only machine; timan107: 12 GiB coordinator guard below.
NODE_CONFIG = {'a100': dict(floor=16 * GIB, critical=6 * GIB), 'timan107': dict(floor=24 * GIB, critical=16 * GIB)}
DEFAULT_CONFIG = dict(floor=20 * GIB, critical=10 * GIB)


def physical_maximum(node, domain):
    """Renderer hard cap of one node; the scheduler decides actual counts below it."""
    from exp.offline_search.rounds.r17.filter_failed.timan1_gpu2 import CAPS, SIZES
    from exp.r17_integration.ops.h100_workers import CPU_WORKERS
    if node == 'timan1':
        return CAPS[domain]
    if node in CPU_WORKERS:
        return CPU_WORKERS[node]
    cards = {'timan107': [8192] * 8, 'local': [49140]}[node]
    # Ten percent framebuffer reserve; residents and server growth are charged per spawn on the node.
    return sum((total - (total + 9) // 10 - 10) // SIZES[domain][0] for total in cards)


def server_host(campaign):
    """Machine of the model server: 'local', 'h100', or the R17_H100_NODE stand-in (h100-temp, a pod)."""
    if campaign['server_node'] == 'local':
        return 'local'
    return campaign.get('execution_host') or os.environ.get('R17_H100_NODE') or 'h100'


def nodes(campaign):
    domain, host = campaign['domain'], server_host(campaign)
    result = {n: {'class': 'shared', 'max': physical_maximum(n, domain)} for n, kinds in SHARED.items() if domain in kinds}
    if domain in LOCAL.get(host, ()):
        result[host] = {'class': 'local', 'max': physical_maximum(host, domain)}
    return result


def apply(campaign, directory=None, *, force=False):
    """Write the scheduler fields; a campaign that has them keeps them unless forced.

    worker_resilience prepares single nodes from a narrowed copy, which must not
    widen again. ``directory`` is accepted for the existing call sites.
    """
    c = campaign
    if 'scheduler_nodes' in c and not force:
        return c
    for key in [k for k in c if k.startswith('autoscale') or k == 'demand_targets']:
        del c[key]
    from exp.offline_search.rounds.r17.filter_failed.timan1_gpu2 import SIZES
    eligible = nodes(c)
    maxima = {n: v['max'] for n, v in eligible.items()}
    c.update(scheduler_url=URL, scheduler_nodes=eligible, server_host=server_host(c),
             scheduler_node_config={n: dict(NODE_CONFIG.get(n, DEFAULT_CONFIG)) for n in eligible},
             wire_zlib=True, execution_worker_nodes=maxima, execution_workers=sum(maxima.values()),
             worker_vram_mib=SIZES[c['domain']][0])
    # Canonical (scientific) topology labels stay as built.
    c.setdefault('worker_nodes', dict(maxima))
    c.setdefault('workers', sum(maxima.values()))
    if c['domain'] == 'metaworld' and not c.get('worker_max_episodes'):
        c.update(worker_max_episodes=32, worker_max_seconds=1800)
    return c


def files():
    """This package's modules, as packed into driver runtimes and worker payloads."""
    return sorted(PACKAGE + '/' + p.name for p in Path(__file__).parent.glob('*.py'))


def require(campaign, directory):
    """Check scheduler availability before dispatch."""
    url = campaign['scheduler_url'] + '/healthz'
    try:
        with urllib.request.urlopen(url, timeout=10) as reply:
            json.load(reply)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f'central worker scheduler not answering at {url}: {exc!r}') from exc
