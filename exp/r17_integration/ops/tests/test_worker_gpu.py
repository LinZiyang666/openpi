"""CPU gates for cross-campaign EGL admission and descendant closure."""
import json
import os
from pathlib import Path
import sys
from unittest import mock

import pytest

from exp.r17_integration.ops import formal_process as process, worker_gpu as gpu


def worker(root, name, device=0, *, live=True, owner=False):
    run = root / name
    run.mkdir()
    campaign = dict(run_name=name, worker_node='cpu', worker_run=str(run), worker_egl_device=device)
    process.write_json(run / 'manifest.json', campaign)
    entry = dict(pid=os.getpid(), start_ticks=process.ticks(os.getpid()))
    if not live:
        entry['start_ticks'] = str(int(entry['start_ticks']) + 1)
    process.write_json(run / 'receipt.json', dict(status='CLOSED', pids=[] if owner else [entry],
        fleet_owner=entry if owner else None))
    return campaign, run, entry


def test_same_gpu_is_exclusive_but_different_gpu_can_run(tmp_path):
    first = tmp_path / 'first'; first.mkdir()
    second = tmp_path / 'second'; second.mkdir()
    lock = gpu.acquire(dict(worker_egl_device=0), first)
    try:
        for _ in range(2):
            with pytest.raises(RuntimeError, match='leased by another'):
                gpu.acquire(dict(worker_egl_device=0), second)
        other = gpu.acquire(dict(worker_egl_device=3), second)
        other.close()
    finally:
        lock.close()
    gpu.acquire(dict(worker_egl_device=0), second).close()


@pytest.mark.parametrize('owner', [False, True])
def test_legacy_or_orphaned_fleet_blocks_admission_without_signalling_it(tmp_path, owner):
    _, old, _ = worker(tmp_path, 'legacy', owner=owner)
    before = (old / 'receipt.json').read_bytes()
    new = tmp_path / 'new'; new.mkdir()
    with mock.patch('os.kill') as kill, mock.patch('os.killpg') as killpg:
        with pytest.raises(RuntimeError, match='still belongs to legacy'):
            gpu.acquire(dict(worker_egl_device=0), new)
        kill.assert_not_called(); killpg.assert_not_called()
    assert (old / 'receipt.json').read_bytes() == before


def test_reused_pid_and_other_gpu_do_not_block_admission(tmp_path):
    worker(tmp_path, 'reused', live=False)
    worker(tmp_path, 'other_device', device=3)
    new = tmp_path / 'new'; new.mkdir()
    gpu.acquire(dict(worker_egl_device=0), new).close()


def request(tmp_path, *, live=True):
    campaign, run, entry = worker(tmp_path, 'fleet', live=live)
    config = tmp_path / 'campaign.json'; process.write_json(config, campaign)
    receipt = tmp_path / 'process.json'
    dead = dict(pid=99999999, pgid=99999999, start_ticks='0')
    process.write_json(receipt, dict(status='CLOSED', child=dead, supervisor=dead))
    req = dict(receipt=str(receipt), argv=[sys.executable, '-m',
        'exp.r17_integration.ops.formal_workers', '--campaign', str(config)])
    return req, entry, run


@pytest.mark.parametrize('mode', ['status', 'stop'])
def test_dead_parent_with_live_worker_never_claims_closed(tmp_path, mode):
    req, entry, _ = request(tmp_path)
    with mock.patch.object(process, 'stop_group', return_value=dict(status='closed')):
        result = process.manage(req, mode)
    assert not result['supervisor_alive'] and not result['child_alive']
    assert result['alive'] and result['owned_worker_survivors'] == [entry]


def test_dead_or_reused_worker_is_closed_and_missing_receipt_is_unconfirmed(tmp_path):
    req, _, run = request(tmp_path, live=False)
    assert process.manage(req, 'status')['alive'] is False
    (run / 'receipt.json').rename(run / 'receipt_unavailable.json')
    result = process.manage(req, 'status')
    assert result['alive'] and result['worker_closure_error']


def test_gpu_refusal_happens_before_any_worker_spawn_and_preserves_other_fleet(tmp_path, monkeypatch):
    from exp.r17_integration.ops import formal_workers
    _, old, _ = worker(tmp_path, 'other')
    before = (old / 'receipt.json').read_bytes()
    campaign = dict(run_name='refused', worker_node='cpu', worker_run=str(tmp_path / 'refused'),
        domain='libero', node_workers=1, seconds=30, worker_egl_device=0)
    config = tmp_path / 'refused.json'; process.write_json(config, campaign)
    monkeypatch.setattr(sys, 'argv', ['formal_workers', '--campaign', str(config)])
    monkeypatch.setattr(gpu, 'inventory', lambda *a: [card(0, used=7670)])
    with mock.patch.object(formal_workers.subprocess, 'Popen') as popen, \
         mock.patch('exp.r17_integration.ops.sync_back.snapshot_run', return_value=dict(archive_sha256='cpu')), \
         mock.patch('exp.r17_integration.qualification.archive_stage.stage_archive', return_value={}):
        with pytest.raises(SystemExit) as exit_info:
            formal_workers.main()
        assert exit_info.value.code == 2
        popen.assert_not_called()
    receipt = json.loads((tmp_path / 'refused/receipt.json').read_text())
    assert receipt['status'] == 'PARTIAL' and receipt['pids'] == []
    assert 'EGL VRAM admission refused' in receipt['reason']
    assert (old / 'receipt.json').read_bytes() == before


def card(device, used=2, total=8192, **kw):
    return dict(device=device, uuid=f'GPU-{device}', total_mib=total,
                used_mib=used, processes=[], usable=True, **kw)


def test_round_robin_skips_full_card_and_records_physical_devices(tmp_path):
    config = dict(domain='libero', worker_node='timan107', node_workers=12, worker_egl_device=0)
    result = gpu.admission(config, tmp_path/'new', observed=[card(0, used=8085),*[card(i) for i in range(1,8)]])
    assert result['admitted']
    assert [e['device'] for e in result['assignments']] == [1,2,3,4,5,6,7,1,2,3,4,5]
    assert all(e['uuid']==f'GPU-{e["device"]}' for e in result['assignments'])
    assert gpu.admission(dict(config,node_workers=50),tmp_path/'new',observed=[card(i) for i in range(1,8)])['admitted'] is False


def test_protected_and_other_user_cards_are_never_eligible(tmp_path):
    config = dict(domain='libero', worker_node='timan108', node_workers=4, worker_egl_device=0)
    result = gpu.admission(config,tmp_path/'new',observed=[card(0),card(1),card(2),dict(card(3),usable=False)])
    assert result['admitted'] and {e['device'] for e in result['assignments']} == {0}
    assert result['excluded_devices'] == [1,2]
    refused = gpu.admission(dict(config,node_workers=8),tmp_path/'new',observed=[card(0),card(1),card(2)])
    assert not refused['admitted']


def test_pending_reservations_account_for_two_fleets_without_double_counting(tmp_path, monkeypatch):
    campaign, old, entry = worker(tmp_path,'first',owner=True)
    campaign.update(domain='libero', node_workers=6, worker_vram_mib=1024)
    process.write_json(old/'manifest.json',campaign)
    process.write_json(old/'GPU_RESERVATION.json',dict(owner=entry,worker_vram_mib=1024,
        assignments=[dict(index=i,device=0) for i in range(6)],released=False))
    # One current worker occupies1GiB; the other five slots remain reserved.
    process.write_json(old/'receipt.json',dict(pids=[dict(entry,index=0,egl_device=0)],fleet_owner=entry))
    observed = [dict(card(0,used=1026),processes=[dict(pid=entry['pid'],used_mib=1024)])]
    result = gpu.admission(dict(domain='libero',worker_node='cpu',node_workers=2,worker_egl_device=0),
                           tmp_path/'second',observed=observed)
    assert not result['admitted'] and len(result['assignments']) == 1
    assert result['other_fleets'][0]['pending_mib'] == 5120
    assert result['cards'][0]['used_mib'] == 1026


def test_released_reservation_still_accounts_for_live_orphan(tmp_path):
    _,old,entry = worker(tmp_path,'orphan')
    process.write_json(old/'GPU_RESERVATION.json',dict(owner=dict(pid=99999999,start_ticks='0'),
        worker_vram_mib=1024,assignments=[dict(index=0,device=0)],released=True))
    result = gpu.admission(dict(node_workers=7,worker_egl_device=0),tmp_path/'new',observed=[card(0)])
    assert not result['admitted'] and result['other_fleets'][0]['workers'] == 1


def test_acquisition_publishes_reservation_before_other_fleet_can_admit(tmp_path,monkeypatch):
    monkeypatch.setattr(gpu,'inventory',lambda *a:[card(0)])
    first=tmp_path/'first'; first.mkdir()
    second=tmp_path/'second'; second.mkdir()
    config=dict(domain='libero',worker_node='cpu',node_workers=7,worker_egl_device=0)
    process.write_json(first/'manifest.json',config)
    process.write_json(second/'manifest.json',dict(config,node_workers=1))
    lease=gpu.acquire(config,first)
    try:
        with pytest.raises(RuntimeError,match='VRAM admission refused'):
            gpu.acquire(dict(config,node_workers=1),second)
    finally: lease.close()
    gpu.acquire(dict(config,node_workers=1),second).close()


def test_inventory_never_queries_protected_devices_and_fails_closed(monkeypatch):
    xml='''<nvidia_smi_log><gpu><minor_number>3</minor_number><uuid>GPU-3</uuid>
        <product_name>GPU</product_name><fb_memory_usage><total>8192 MiB</total>
        <used>123 MiB</used></fb_memory_usage><processes/></gpu></nvidia_smi_log>'''
    with mock.patch.object(gpu.subprocess,'run',return_value=mock.Mock(stdout=xml)) as query:
        result=gpu.inventory('timan108',[1,2,3])
        assert [c['device'] for c in result]==[3]
        assert query.call_count==1 and query.call_args.args[0][2]=='3'
    with mock.patch.object(gpu.subprocess,'run',side_effect=OSError('unavailable')):
        assert gpu.inventory('timan107',[7])[0]['usable'] is False
    foreign=xml.replace('<processes/>',f'<processes><process_info><pid>{os.getpid()}</pid>'
        '<used_memory>77 MiB</used_memory></process_info></processes>')
    with mock.patch.object(gpu.subprocess,'run',return_value=mock.Mock(stdout=foreign)), \
         mock.patch.object(gpu.os,'getuid',return_value=99999999):
        observed=gpu.inventory('timan107',[3])[0]
        assert observed['usable'] is False and observed['processes'][0]['used_mib']==77


def test_assigned_device_overrides_fleet_default_in_worker_argv():
    from exp.r17_integration.ops.formal_workers import worker_command
    config=dict(run_name='new',worker_node='timan107',endpoint='host:1',driver_host='host',
        driver_port=2,domain='libero',worker_benchmark='/sealed',worker_egl_device=0,
        worker_gpu_assignments=[dict(index=0,device=3,uuid='GPU-3')])
    _,argv=worker_command(config,0)
    assert argv[argv.index('--egl-device')+1]=='3'


def test_fleet_sets_import_time_device_environment_and_attests_each_child(tmp_path,monkeypatch):
    from exp.r17_integration.ops import formal_workers
    monkeypatch.setattr(gpu,'inventory',lambda *a:[card(0),card(3)])
    config=dict(run_name='devices',worker_node='timan108',worker_run=str(tmp_path/'devices'),
        endpoint='host:1',driver_host='host',driver_port=2,domain='libero',worker_benchmark='/sealed',
        node_workers=2,seconds=30,worker_egl_device=0)
    process.write_json(tmp_path/'campaign.json',config)
    monkeypatch.setattr(sys,'argv',['fleet','--campaign',str(tmp_path/'campaign.json')])
    child=mock.Mock(pid=os.getpid()); child.poll.return_value=0; child.wait.return_value=0
    with mock.patch.object(formal_workers.subprocess,'Popen',return_value=child) as popen, \
         mock.patch('exp.r17_integration.ops.worker_memory.admission',return_value=dict(admitted=True)), \
         mock.patch('exp.r17_integration.ops.sync_back.snapshot_run',return_value=dict(archive_sha256='cpu')), \
         mock.patch('exp.r17_integration.qualification.archive_stage.stage_archive',return_value={}):
        with pytest.raises(SystemExit) as ended: formal_workers.main()
        assert ended.value.code==0
        assert [c.kwargs['env']['MUJOCO_EGL_DEVICE_ID'] for c in popen.call_args_list]==['0','3']
        assert all(c.kwargs['env']['CUDA_VISIBLE_DEVICES']=='' for c in popen.call_args_list)
    receipt=json.loads((tmp_path/'devices/receipt.json').read_text())
    assert [e['egl_device'] for e in receipt['pids']]==[0,3]
    assert [e['gpu_uuid'] for e in receipt['pids']]==['GPU-0','GPU-3']
    assert receipt['gpu_admission']['excluded_devices']==[1,2]


def test_replacement_cannot_exceed_reserved_card_after_external_usage_rises(tmp_path,monkeypatch):
    first=tmp_path/'first'; first.mkdir()
    config=dict(domain='libero',worker_node='cpu',node_workers=2,worker_egl_device=0)
    process.write_json(first/'manifest.json',config)
    monkeypatch.setattr(gpu,'inventory',lambda *a:[card(0)])
    lease=gpu.acquire(config,first)
    try:
        assert lease.check(0)
        monkeypatch.setattr(gpu,'inventory',lambda *a:[card(0,used=6500)])
        assert not lease.check(0)
    finally: lease.close()


def test_formal_cli_uses_safe_defaults_and_explicit_allowlist():
    from exp.r17_integration.ops.formal_eval import parse_args
    args=parse_args(['--prepare-only'])
    assert (args.worker_max_episodes,args.worker_memory_mib,args.worker_vram_mib)==(4,6144,1024)
    args=parse_args(['--prepare-only','--worker-egl-devices','1,3'])
    assert args.worker_egl_devices==[1,3]
    for values in ('-1','1,1'):
        with pytest.raises(SystemExit):parse_args(['--prepare-only','--worker-egl-devices',values])
