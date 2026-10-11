"""CPU-only coverage of the coordinator's 60-worker admission policy."""
import copy

import pytest

from exp.r17_integration.ops import cache_capacity_lanes, local_workers, worker_memory, worker_gpu
from exp.r17_integration.ops.formal_process import write_json
from exp.r17_integration.ops.tests.test_cache_size_fleets import frozen

MIB = 1024**2
GIB = 1024**3


def configured(tmp_path):
    c = frozen(tmp_path)
    c.update(cache_capacity_lanes.DP_RESOURCES, execution_worker_nodes={'local': 60})
    c['cache_size_lane_authorization'].update(worker_nodes={'local': 60},
        resource_reservations=copy.deepcopy(cache_capacity_lanes.DP_RESOURCES))
    return c


def test_sixty_local_workers_and_four_replicas_use_the_requested_policy(tmp_path):
    c = configured(tmp_path)
    w = local_workers.route(dict(c, worker_node='local', node_workers=60), c)
    assert cache_capacity_lanes.FLEETS['dp_libero'] == {'local': 60}
    assert (w['worker_memory_mib'], w['worker_vram_mib'], w['worker_memory_headroom_mib']) == (3584, 717, 8192)
    assert local_workers.authorize(c, today='2026-10-13')
    result = worker_memory.admission(w, tmp_path/'next', memory=(1024*GIB, 220*GIB))
    assert result['admitted'] and result['requested_bytes'] == 210*GIB
    assert result['headroom_bytes'] == 8*GIB  # Never 10% of a large node.


@pytest.mark.parametrize('field', list(cache_capacity_lanes.DP_RESOURCES))
def test_resource_policy_is_bound_to_the_frozen_preparation(tmp_path, field):
    c = configured(tmp_path); c[field] += 1
    with pytest.raises(ValueError, match='resource reservations changed'):
        local_workers.authorize(c, today='2026-10-13')


def test_running_workers_use_individual_maxima_and_hold_recycling_slots(tmp_path):
    run = tmp_path/'existing'
    receipt = dict(status='RUNNING', workers=3, fleet_owner=dict(pid=99),
        worker_memory_mib=3584, pids=[dict(pid=1), dict(pid=2)])
    write_json(run/'receipt.json', receipt)
    original = (run/'receipt.json').read_bytes()
    rss = lambda p: {1: 6*GIB, 2: GIB, 99: MIB}.get(p['pid'])
    result = worker_memory.admission(dict(domain='libero', node_workers=1, worker_memory_mib=3584),
        tmp_path/'next', memory=(1024*GIB, 20*GIB), rss_reader=rss)
    fleet = result['other_fleets'][0]
    assert fleet['rss_bytes'] == 7*GIB
    assert fleet['reservation_bytes'] == 10.5*GIB
    assert fleet['accounted_bytes'] == 13*GIB  # max(3.5,6)+max(3.5,1)+3.5
    assert fleet['pending_bytes'] == 6*GIB  # RSS already deducted by MemAvailable.
    assert result['admitted']
    assert (run/'receipt.json').read_bytes() == original


def test_explicit_policy_survives_unchanged_legacy_fleet_floor(tmp_path):
    run = tmp_path/'existing'
    write_json(run/'receipt.json', dict(status='STARTING', workers=60, fleet_owner=dict(pid=99),
        worker_memory_mib=4096, pids=[]))
    write_json(run/'manifest.json', dict(domain='libero', cache_study='frozen',
        worker_memory_mib=4096, worker_ram_reservation_mib=3584))
    result = worker_memory.admission(dict(domain='libero', node_workers=1, worker_memory_mib=3584),
        tmp_path/'next', memory=(1024*GIB, 230*GIB), rss_reader=lambda p: MIB)
    assert result['other_pending_bytes'] == 210*GIB
    assert result['other_fleets'][0]['worker_memory_mib'] == 3584


def test_historical_larger_reservation_and_measured_rss_are_retained(tmp_path):
    write_json(tmp_path/'old/receipt.json', dict(status='RUNNING', workers=2,
        worker_memory_mib=6144, pids=[dict(pid=1), dict(pid=2)]))
    result = worker_memory.admission(dict(domain='libero', node_workers=1, worker_memory_mib=3584),
        tmp_path/'next', memory=(512*GIB, 20*GIB), rss_reader=lambda p: {1:GIB, 2:8*GIB}[p['pid']])
    assert result['other_fleets'][0]['accounted_bytes'] == 14*GIB
    assert result['other_pending_bytes'] == 5*GIB


def test_renderer_budget_does_not_silently_remove_remaining_gpu_guards(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_gpu, 'reservations', lambda *a, **k: ({0:0}, []))
    w = dict(worker_node='local', node_workers=60, worker_egl_devices=[0], worker_vram_mib=717,
        worker_server_vram_headroom_mib=8192)
    card = dict(device=0, uuid='local', total_mib=49140, used_mib=0, usable=True)
    result = worker_gpu.admission(w, tmp_path/'next', observed=[card])
    assert not result['admitted'] and len(result['assignments']) == 50
    assert result['cards'][0]['headroom_mib'] == 8192 + 4914
