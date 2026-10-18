import json

import pytest

from exp.offline_search.rounds.r17.autoscale_1017.node import atomic
from exp.offline_search.rounds.r17.autoscale_1017.reset_zero import reset


def fixture(root):
    atomic(root/'campaign.json',dict(run_name=root.name,seed=7,cache_plan_name='original'))
    atomic(root/'EPISODE_PLAN.json',dict(tasks=[dict(task_uid='unchanged',orig_init_state_idx=9,seed=7)]))
    atomic(root/'RECEIPT.json',dict(status='FAIL',all_owned_closed=True))
    atomic(root/'LAUNCH.json',dict(pid=123))
    atomic(root/'PREPARED.json',dict(files={}))
    atomic(root/'assets/science.json',dict(model='original'))
    (root/'driver').mkdir();(root/'driver/journal.jsonl').write_text('')
    (root/'driver/rejected_worker_probes.json').write_text('original evidence')


def test_zero_reset_preserves_entire_evidence_and_exact_scientific_inputs(tmp_path):
    root=tmp_path/'run';backup=tmp_path/'failed';fixture(root)
    originals={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    result=reset(root,backup)
    assert result['accepted']==0 and result['launches']==0
    assert all((backup/n).read_bytes()==v for n,v in originals.items())
    for n in ('campaign.json','EPISODE_PLAN.json','assets/science.json'):
        assert (root/n).read_bytes()==originals[n]
    assert not (root/'LAUNCH.json').exists() and not (root/'PREPARED.json').exists()
    assert not (root/'RECEIPT.json').exists() and not (root/'driver').exists()
    assert json.loads((root/'EPISODE_PLAN.json').read_text())['tasks'][0]['task_uid']=='unchanged'


@pytest.mark.parametrize('cause',['accepted','not_closed','completed','continuation','existing_backup'])
def test_zero_reset_refuses_unsafe_or_nonzero_progress(tmp_path,cause):
    root=tmp_path/'run';backup=tmp_path/'failed';fixture(root)
    if cause=='accepted':(root/'driver/journal.jsonl').write_text('{"accepted":true}\n')
    elif cause=='not_closed':atomic(root/'RECEIPT.json',dict(status='FAIL',all_owned_closed=False))
    elif cause=='completed':atomic(root/'DONE.json',{})
    elif cause=='continuation':atomic(root/'campaign.json',dict(resume_from='/source'))
    else:backup.mkdir()
    plan=(root/'EPISODE_PLAN.json').read_bytes()
    with pytest.raises(ValueError):reset(root,backup)
    assert (root/'EPISODE_PLAN.json').read_bytes()==plan and (root/'LAUNCH.json').exists()


def test_cli_refuses_before_generation_fix_is_installed(tmp_path,monkeypatch):
    from exp.offline_search.rounds.r17.autoscale_1017.reset_zero import main
    root=tmp_path/'run';fixture(root)
    monkeypatch.setattr('exp.offline_search.rounds.r17.autoscale_1017.switch.enabled',lambda:True)
    monkeypatch.setattr('inspect.getsource',lambda obj:'old worker implementation')
    with pytest.raises(ValueError,match='fix is not installed'):
        main(['--run',str(root),'--backup',str(tmp_path/'failed'),'--reprepare'])
    assert (root/'LAUNCH.json').exists()


def test_cli_reprepares_same_plan_without_launching(tmp_path,monkeypatch):
    from exp.offline_search.rounds.r17.autoscale_1017.reset_zero import main
    from exp.offline_search.rounds.r17.autoscale_1017 import formal_trial
    root=tmp_path/'run';fixture(root);plan=(root/'EPISODE_PLAN.json').read_bytes();calls=[]
    def prepare(path,maxima,*,reprepare):
        assert path==root and reprepare and (root/'EPISODE_PLAN.json').read_bytes()==plan
        assert not (root/'LAUNCH.json').exists();calls.append(maxima)
        atomic(root/'PREPARED.json',dict(test_double=True))
    monkeypatch.setattr(formal_trial,'prepare',prepare)
    main(['--run',str(root),'--backup',str(tmp_path/'failed'),'--reprepare',
          '--worker-counts','timan1=332,timan107=440'])
    assert calls==[{'timan1':332,'timan107':440}]
    assert json.loads((root/'RESET_ZERO_RECEIPT.json').read_text())['status']=='RESET_PREPARED'


def test_failed_formal_campaign_frozen_planner_keeps_real_uids_inits_seeds(tmp_path):
    """Read the failed campaign, but run its real frozen CPU planner in tmp."""
    import copy
    from pathlib import Path
    from exp.offline_search.rounds.r17.autoscale_1017.defaults import apply
    from exp.offline_search.rounds.r17.autoscale_1017.compare import wait_io
    from exp.r17_integration.ops.driver_release import planned_tasks
    root=Path('/data/mw_finetune/eval/fa_ls_fw_r_p_m')
    wait_io()
    campaign=json.loads((root/'campaign.json').read_text())
    expected=json.loads((root/'EPISODE_PLAN.json').read_text())['tasks']
    stamps={n:((root/n).stat().st_size,(root/n).stat().st_mtime_ns)
            for n in ('campaign.json','EPISODE_PLAN.json')}
    from exp.offline_search.rounds.r17.autoscale_1017.formal_trial import configure
    candidate=configure(campaign,{'timan1':332,'timan107':440})
    assert candidate['autoscale_node_maxima']=={'timan1':332,'timan107':440}
    atomic(tmp_path/'campaign.json',candidate)
    atomic(tmp_path/'ORCHESTRATION.json',campaign['orchestration'])
    assert planned_tasks(candidate,tmp_path)==expected
    assert stamps=={n:((root/n).stat().st_size,(root/n).stat().st_mtime_ns) for n in stamps}



def test_reset_keeps_immutable_cache_inodes_but_copies_mutable_metadata(tmp_path):
    import os
    root=tmp_path/'run';backup=tmp_path/'failed';fixture(root)
    source=tmp_path/'sealed_source';source.mkdir()
    (source/'fit.bin').write_bytes(b'immutable fitted numeric array')
    cache=root/'assets/cache_inputs';cache.mkdir()
    os.link(source/'fit.bin',cache/'fit.bin')
    atomic(cache/'manifest.json',dict(files={'fit.bin':'sealed'}))
    reset(root,backup)
    assert (root/'assets/cache_inputs/fit.bin').samefile(source/'fit.bin')
    assert (backup/'assets/cache_inputs/fit.bin').samefile(source/'fit.bin')
    assert not (root/'assets/cache_inputs/manifest.json').samefile(backup/'assets/cache_inputs/manifest.json')
    assert not (root/'assets/science.json').samefile(backup/'assets/science.json')
    (root/'assets/science.json').write_text('fresh metadata')
    assert json.loads((backup/'assets/science.json').read_text())==dict(model='original')
