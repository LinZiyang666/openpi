"""CPU-only hot handoff of a frozen, undispatched lane job."""
import json
from pathlib import Path

import pytest

from exp.r17_integration.ops import prepared_handoff as h,cache_capacity as capacity
from exp.r17_integration.ops.formal_process import write_json
from exp.r17_integration.ops.formal_prestage import seal_prepared,validate_prepared
from exp.r17_integration.ops.tests.test_cache_capacity import campaign as capacity_campaign


def make(root,name,**changes):
    p=root/name;p.mkdir()
    (p/'assets').mkdir();(p/'assets/science.yaml').write_text('size: 500\n')
    (p/'runtime.tar').write_bytes(b'unchanged frozen native runtime')
    c=dict(run_name=name,model='cosmos',domain='libero',replicas=2,workers=8,
        execution_workers=8,execution_worker_nodes={'timan1':8},runtime='frozen',runtime_identity='pin',
        checkpoint='same-checkpoint',cache_study='same-study')
    c.update(changes)
    for n,d in [('campaign.json',c),('PREFLIGHT.json',{}),('WORKER_DEPLOYMENT.json',{}),
                ('EPISODE_PLAN.json',{'tasks':[{'task_uid':'same'}]})]:write_json(p/n,d)
    seal_prepared(c,p,{})
    return p


def test_atomic_prepared_alias_resolves_new_seal_and_preserves_original(tmp_path):
    old=make(tmp_path,'old');new=make(tmp_path,'new',execution_workers=57,
        execution_worker_nodes={'timan107':17,'timan1':40})
    original=(old/'PREPARED.json').read_bytes();backup=tmp_path/'original_preserved'
    record=h.exchange_prepared(old,new,backup)
    assert old.is_symlink() and old.resolve()==new.resolve()
    assert record['atomic_exchange'] and record['no_process_signals']
    assert (backup/'PREPARED.json').read_bytes()==original
    assert validate_prepared(old)[0]['run_name']=='new'
    assert validate_prepared(old)[0]['execution_workers']==57
    # The already running parent keeps its old path; the new launcher resolves it.
    assert (old/'assets/science.yaml').read_bytes()==(backup/'assets/science.yaml').read_bytes()


def test_second_atomic_handoff_preserves_both_undispatched_preparations(tmp_path):
    old=make(tmp_path,'old');first=make(tmp_path,'first',execution_workers=57)
    second=make(tmp_path,'second',execution_workers=85)
    h.exchange_prepared(old,first,tmp_path/'original_preserved')
    saved=tmp_path/'previous_alias'
    record=h.exchange_prepared(old,second,saved)
    assert record['previous_path_was_alias'] and old.resolve()==second.resolve()
    assert saved.resolve()==first.resolve() and first.is_dir()
    assert validate_prepared(old)[0]['execution_workers']==85
    assert validate_prepared(saved)[0]['execution_workers']==57


@pytest.mark.parametrize('change',[{'checkpoint':'different'},{'replicas':4}])
def test_scientific_or_server_changes_refused_before_exchange(tmp_path,change):
    old=make(tmp_path,'old');new=make(tmp_path,'new',**change)
    with pytest.raises(ValueError,match='scientific/server'):h.exchange_prepared(old,new,tmp_path/'saved')
    assert not old.is_symlink() and not (tmp_path/'saved').exists()


def test_dispatched_source_refused_without_affecting_directory(tmp_path):
    old=make(tmp_path,'old');new=make(tmp_path,'new');write_json(old/'LAUNCH.json',{'pid':1})
    with pytest.raises(ValueError,match='already launched'):h.exchange_prepared(old,new,tmp_path/'saved')
    assert not old.is_symlink()


def test_renderer_spread_and_gpu3_restriction_are_bound_to_capacity_record():
    c=capacity_campaign('cosmos',{'timan107':17,'timan1':40})
    record=c['cache_capacity_authorization']
    record['worker_egl_devices_by_node']={'timan107':list(range(8)),'timan1':[3]}
    record['digest']=capacity.digest({k:v for k,v in record.items() if k!='digest'})
    assert capacity.renderer_devices(c,'timan107')==list(range(8))
    assert capacity.renderer_devices(c,'timan1')==[3]
    record['worker_egl_devices_by_node']['timan1']=[0]
    record['digest']=capacity.digest({k:v for k,v in record.items() if k!='digest'})
    with pytest.raises(ValueError,match='authorized node scope'):capacity.renderer_devices(c,'timan1')
