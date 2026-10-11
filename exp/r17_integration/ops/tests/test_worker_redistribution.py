"""CPU regressions for guarded T1 leases and same-count cache continuations."""
import copy
import json
from pathlib import Path
from unittest import mock

import pytest

from exp.r17_integration.ops import cache_lane,local_workers,timan1_workers as t1,worker_gpu,worker_memory,worker_topology


def test_t1_lease_and_wrapper_never_fall_back_to_another_device():
    from exp.r17_integration.ops.formal_workers import worker_command
    from exp.r17_integration.ops.formal_eval import worker_paths
    c=dict(domain='libero',cache_study='/sealed',worker_nodes={'timan1':16},workers=16,
           run_name='q',endpoint='ziyanglin.com:23420',driver_host='ziyanglin.com',driver_port=23439,
           worker_benchmark='/sealed/benchmark',worker_node='timan1',node_workers=16)
    w=t1.route(dict(c),c)
    _,argv=worker_command(w,0,python='unprotected-python')
    assert argv[0]==t1.WRAPPER and argv[argv.index('--egl-device')+1]=='3'
    assert '--max-episodes' in argv and w['worker_memory_mib']==6144
    assert worker_paths(c,'timan1')['run'].endswith('/q_timan1')
    env=t1.environment({'PYTHONPATH':'frozen','MUJOCO_EGL_DEVICE_ID':'0'})
    assert env['PYTHONPATH'].startswith(t1.GUARD+':') and env['CUDA_VISIBLE_DEVICES']==t1.UUID
    for bad in (dict(c,worker_nodes={'timan1':17}),dict(c,domain='metaworld'),dict(c,cache_study=None)):
        with pytest.raises(ValueError):t1.authorize(bad)


def test_t1_inventory_queries_only_authorized_uuid_and_verifies_mapping():
    xml='<nvidia_smi_log><gpu><minor_number>0</minor_number><uuid>'+t1.UUID+'</uuid><product_name>A6000</product_name><pci><pci_bus_id>00000000:C1:00.0</pci_bus_id></pci><fb_memory_usage><total>49140 MiB</total><used>10 MiB</used></fb_memory_usage></gpu></nvidia_smi_log>'
    with mock.patch.object(worker_gpu.subprocess,'run',return_value=mock.Mock(stdout=xml)) as query:
        row=worker_gpu.inventory('timan1')[0]
        assert row['usable'] and row['device']==3 and row['device_minor']==0
        assert query.call_args.args[0]==['nvidia-smi','-i',t1.UUID,'-q','-x']
        for devices in ([0],[1],[2],[0,3],[]):
            with pytest.raises(ValueError):worker_gpu.inventory('timan1',devices)
        assert query.call_count==1
        query.return_value.stdout=xml.replace('C1:00.0','C2:00.0')
        assert not worker_gpu.inventory('timan1',[3])[0]['usable']


def test_historical_cache_fleets_reserve_measured_ram_without_mutation(tmp_path):
    mib=1024**2
    old=tmp_path/'running_dp';old.mkdir()
    receipt=dict(status='RUNNING',workers=32,worker_memory_mib=2048,fleet_owner={'pid':1},pids=[])
    (old/'receipt.json').write_text(json.dumps(receipt))
    (old/'manifest.json').write_text(json.dumps({'cache_study':'sealed'}))
    c=dict(domain='libero',cache_study='sealed',worker_memory_mib=2048,node_workers=16)
    r=worker_memory.admission(c,tmp_path/'next',memory=(220*1024*mib,100*1024*mib),rss_reader=lambda _:1)
    assert r['worker_memory_mib']==3584 and r['other_pending_bytes']==32*3584*mib
    assert not r['admitted'] and json.loads((old/'receipt.json').read_text())==receipt


def test_t1_root_display_is_accounted_but_foreign_compute_still_refuses():
    xml='<nvidia_smi_log><gpu><minor_number>0</minor_number><uuid>'+t1.UUID+'</uuid><product_name>A6000</product_name><pci><pci_bus_id>00000000:C1:00.0</pci_bus_id></pci><fb_memory_usage><total>49140 MiB</total><used>10 MiB</used></fb_memory_usage><processes><process_info><pid>3016</pid><type>G</type><process_name>/usr/lib/xorg/Xorg</process_name><used_memory>4 MiB</used_memory></process_info></processes></gpu></nvidia_smi_log>'
    with mock.patch.object(worker_gpu.subprocess,'run',return_value=mock.Mock(stdout=xml)) as query,mock.patch.object(Path,'stat',return_value=mock.Mock(st_uid=0)),mock.patch.object(Path,'read_bytes',return_value=b'/usr/lib/xorg/Xorg\0vt1\0'),mock.patch.object(worker_gpu.os,'getuid',return_value=1000):
        card=worker_gpu.inventory('timan1')[0]
        assert card['usable'] and card['used_mib']==10 and card['processes'][0]['system_display']
        query.return_value.stdout=xml.replace('<type>G</type>','<type>C</type>')
        assert not worker_gpu.inventory('timan1')[0]['usable']


def test_worker_redistribution_and_resize_rebind_only_orchestration(tmp_path):
    from exp.r17_integration.ops import cache_control
    control=Path('/data/mw_finetune/eval/hs6fw_res_152931')
    runtime=Path('/data/mw_finetune/eval/hs6fw_ctl_143332')
    c=json.loads((control/'campaign.json').read_text())
    c.update(cache_study='/sealed/cache/STUDY.json',cache_control=str(control),cache_control_runtime=str(runtime))
    before=copy.deepcopy(c)
    worker_topology.redistribute(c,'timan107=16,local=16,timan1=16',tmp_path)
    assert c['workers']==before['workers']==48 and c['worker_nodes']==before['worker_nodes']
    assert c['execution_worker_nodes']=={'timan107':16,'local':16,'timan1':16}
    for key in ('checkpoint','model','fastwam_batching','mps','replicas','worker_max_episodes','worker_memory_mib'):
        assert c.get(key)==before.get(key)
    assert c['timan1_worker_lease']['excluded_devices']==[0,1,2]
    from exp.r17_integration.models.fastwam.formal_batching import validate,treatment
    assert validate(c)==treatment(before)  # Original sealed batch identity stays exact.
    resized=copy.deepcopy(before)
    worker_topology.redistribute(resized,'local=16,timan1=16',tmp_path)
    assert resized['execution_workers']==32
    assert resized['workers']==before['workers']==48
    assert resized['worker_nodes']==before['worker_nodes']
    assert resized['cache_worker_orchestration']['source_total_workers']==48
    assert validate(resized)==treatment(before)
    for key in ('checkpoint','model','fastwam_batching','mps','replicas','worker_max_episodes','worker_memory_mib'):
        assert resized.get(key)==before.get(key)
    # Scientific changes remain rejected by the original control validator.
    bad=copy.deepcopy(c);bad['checkpoint']='other'
    with pytest.raises(ValueError,match='checkpoint'):cache_control.control_proof(control,runtime,bad)


def test_lane_model_scoped_overrides_keep_unequal_total_fleets():
    r=cache_lane.scoped_overrides(['fastwam_libero:timan107=16,local=16,timan1=16',
                                 'cosmos_libero:local=4,timan1=4'],['fastwam_libero','cosmos_libero'],fleet=True)
    assert sum(r['fastwam_libero'].values())==48 and sum(r['cosmos_libero'].values())==8
    with pytest.raises(ValueError):cache_lane.scoped_overrides(['dp_libero:local=16'],['fastwam_libero'],fleet=True)
    assert all(p['ram']>=4096 for p in cache_lane.PROFILES.values())


def test_cache_resume_passes_topology_to_clone_before_any_deployment(tmp_path):
    from exp.r17_integration.ops import cache_resume,queue_retry,formal_prestage
    source=tmp_path/'source';source.mkdir();destination=tmp_path/'new';destination.mkdir()
    files={'campaign.json':{'run_name':'old'},'EPISODE_PLAN.json':{'tasks':[{'task_uid':'keep'},{'task_uid':'missing'}]},
           'WORKER_DEPLOYMENT.json':{}}
    for root in (source,destination):
        for name,value in files.items():(root/name).write_text(json.dumps(value))
    selection={'missing_task_uids':['missing'],'dropped':[],'source_pins':{}}
    with mock.patch.object(cache_resume,'select',return_value=selection),mock.patch.object(queue_retry,'clone',return_value=destination) as clone,mock.patch.object(formal_prestage,'seal_prepared'):
        cache_resume.prepare(source,'new',refresh_workers=True,worker_counts={'local':16,'timan107':16,'timan1':16})
    assert clone.call_args.kwargs['worker_counts']['timan1']==16
    assert json.loads((destination/'EPISODE_PLAN.json').read_text())['tasks']==[{'task_uid':'missing'}]
    assert json.loads((destination/'FULL_EPISODE_PLAN.json').read_text())['tasks']==files['EPISODE_PLAN.json']['tasks']


def test_t1_evidence_transfer_rejects_unowned_roots_before_transport(tmp_path):
    from exp.r17_integration.ops import evidence_transfer
    with pytest.raises(ValueError,match='owned SHA'):
        evidence_transfer.pull('timan1','/scratch/foreign',tmp_path/'tar',remote=mock.Mock(),transfer=mock.Mock())


def test_reuse_sealed_dependencies_does_not_require_mutable_legacy_config():
    from exp.r17_integration.ops import formal_eval as launch
    old=dict(dependencies=dict(private_dependencies='/sealed/release',content_identity='native-pin',
                              config_source='/deleted/legacy/config.yaml'),
             private_config=dict(path='/sealed/campaign/config',sha256='config-pin'))
    expected=dict(old['dependencies'],config_source='/sealed/campaign/config/config.yaml',config_sha256='config-pin')
    with mock.patch.object(launch,'remote',return_value=json.dumps(expected)) as remote:
        assert launch.install_worker_dependencies('timan107','unused','/payload',inherited=old)==expected
    code=remote.call_args.args[1][-1]
    assert 'verify_dependencies' in code and 'sealed simulator config changed' in code
    assert '/deleted/legacy' not in code and 'install_dependencies' not in code
