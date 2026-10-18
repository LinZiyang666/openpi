"""The replacement chains' real prepare entries, with all effects redirected."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from exp.offline_search.rounds.r17.autoscale_1017.node import atomic, read
from exp.offline_search.rounds.r17.autoscale_1017.tests.test_defaults import dry_stage, test_each_queued_prepare_dispatch_cpu_only as legacy_prepare

HERE=Path(__file__).resolve().parents[1]
LONGSEG=HERE.parent/'longseg_1017'
EVAL=Path('/data/mw_finetune/eval')
NAMES=('fa_co_f_f_oc1','fa_ls_fw_r_p_m','fa_ls_gr_r_p_m',
       'fa_pi_r_f_l','fa_gr_r_u_l','fa_ls_dp_r_b_l','fa_ls_dp_r_u_m')


def entry(name):
    commands=HERE.parent/'filter_failed_knob/commands'
    p=commands/f'prepare_{name}.sh'
    texts=[p.read_text()]
    if 'exec bash ' in texts[-1]:
        import shlex
        p=Path(shlex.split(texts[-1].split('exec bash ',1)[1])[0])
        texts.append(p.read_text())
    return texts


@pytest.fixture
def no_campaign_writes(monkeypatch):
    import builtins,io
    def guard(native):
        def opened(path,mode='r',*a,**kw):
            if isinstance(path,(str,Path)) and str(Path(path).absolute()).startswith(str(EVAL)+'/'):
                assert not any(k in mode for k in 'wax+'),'dry run attempted campaign write'
            return native(path,mode,*a,**kw)
        return opened
    monkeypatch.setattr(builtins,'open',guard(builtins.open))
    monkeypatch.setattr(io,'open',guard(io.open))


@pytest.mark.parametrize('name',NAMES)
def test_new_chain_prepare(tmp_path,monkeypatch,no_campaign_writes,name):
    scripts=entry(name)
    lists={p.stem:p.read_text().splitlines() for p in (LONGSEG/'H100_NAMES.txt',LONGSEG/'4090_NAMES.txt')}
    assert any(name in names for names in lists.values())
    if name in ('fa_pi_r_f_l','fa_gr_r_u_l'):
        assert 'formal_a_queue dispatch prepare --name '+name in scripts[-1]
        legacy_prepare(tmp_path,monkeypatch,name,'R16')
        release=read(tmp_path/name/'LEGACY_AUTOSCALE.json')
        result=dict(name=name,pipeline='R16',mode=release['mode'],maximum=release['maximum'])
    else:
        from exp.offline_search.rounds.r17.filter_failed import stage
        from exp.offline_search.rounds.r17.longseg_1017.io_budget import filesystem_budget
        source=EVAL/name
        # One bounded reader, including the 90,000-task MetaWorld plan.
        with filesystem_budget():
            c=read(source/'campaign.json')
            plan=read(source/'EPISODE_PLAN.json')
            definition=read(source/'FFK_DEFINITION.json')
        # Replay the queue's not-yet-prepared resource input. A now-launched
        # source may contain a failover execution map (including local workers).
        if name!='fa_co_f_f_oc1':
            c.update(execution_worker_nodes=dict(definition['workers']),
                     execution_workers=sum(definition['workers'].values()))
        stamps={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in (source/'campaign.json',source/'EPISODE_PLAN.json')}
        assert 'filter_failed.stage --run '+str(source) in scripts[-1] or 'filter_failed_knob.stage --run '+str(source) in scripts[-1]
        expected=copy.deepcopy(c)
        result_c=dry_stage(tmp_path/name,c,monkeypatch,tasks=plan['tasks'],invoke=stage.stage)
        for key in ('cache_plan_name','model','domain','filter_failed_knob','longseg_1017',
                    'seed','task_seed','runtime_source_pins','cache_control'):
            assert result_c.get(key)==expected.get(key),key
        assert stamps=={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in (source/'campaign.json',source/'EPISODE_PLAN.json')}
        assert 'timan108' not in result_c.get('execution_worker_nodes',{})
        mode='fixed' if name=='fa_co_f_f_oc1' else 'dynamic'
        assert (result_c.get('autoscale') or {}).get('mode','fixed')==mode
        if mode=='fixed':assert result_c['autoscale_disposition']=='kept_fixed'
        else:
            from exp.offline_search.rounds.r17.autoscale_1017.defaults import maxima
            assert result_c['autoscale_node_maxima']==maxima(c['domain'])
        result=dict(name=name,pipeline='R17',mode=mode,
            desired_maxima=result_c['execution_worker_nodes'],episodes=len(plan['tasks']),
            uid_sha256=hashlib.sha256('\n'.join(t['task_uid'] for t in plan['tasks']).encode()).hexdigest(),
            full_plan_bytes_unchanged=True,scientific_fields_unchanged=True,
            formal_campaign_metadata_unchanged=True)
    result.update(status='PASS',prepare_scripts=scripts,external_effects_stubbed=True,
        output_root=str(tmp_path),remote_calls=0,launches=0,policy_under_test='explicit_cpu_on',
        production_switch=(HERE/'ENABLED').read_text().strip())
    atomic(HERE/'new_chain_dry_run'/f'{name}.json',result)


def test_deferred_longseg_workflow_uses_shared_stage(tmp_path,monkeypatch):
    from exp.offline_search.rounds.r17.longseg_1017 import workflow
    from exp.offline_search.rounds.r17.filter_failed import stage
    root=tmp_path/'deferred'
    unit=dict(name='deferred',jobs=[dict(run=str(root))])
    monkeypatch.setattr(workflow,'build_levels',lambda name:unit)
    c=dict(run_name='deferred',domain='libero',worker_nodes=dict(timan1=150,timan107=100))
    native=stage.stage
    result=[]
    def prepare(path):result.append(dry_stage(path,c,monkeypatch,invoke=native))
    monkeypatch.setattr(stage,'stage',prepare)
    workflow.dispatch('deferred','prepare')
    assert result[0]['autoscale']['mode']=='dynamic'
    assert result[0]['autoscale_node_maxima']==dict(timan1=150,timan107=104)


def test_longseg_check_local_only_never_contacts_h100(tmp_path,monkeypatch):
    """The actual H100 check above is run serially by the command deliverable."""
    import inspect
    from exp.offline_search.rounds.r17.longseg_1017 import check
    signature=inspect.signature(check.check)
    assert signature.parameters['local_only'].default is False
    # Metadata-only negative case must fail before any remote fallback.
    from exp.r17_integration.ops import formal_eval
    monkeypatch.setattr(formal_eval,'remote',lambda *a,**kw:pytest.fail('remote fallback'))
    with pytest.raises(ValueError,match='No strict CPU loader evidence'):
        check.check(dict(name='fa_ls_fw_r_p_m',run=str(EVAL/'fa_ls_fw_r_p_m')),[],
                    local_only=True,output_dir=tmp_path)
