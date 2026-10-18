import json
from pathlib import Path
import pytest
from exp.offline_search.rounds.r17.autoscale_1017.node import atomic
from exp.offline_search.rounds.r17.autoscale_1017.reset_legacy_zero import reset_many


def roots(tmp):
    result=[]
    for name in ('s50','l50'):
        p=tmp/('fa_pi_f_u_o_'+name)
        atomic(p/'arms.json',[dict(arm='same_'+name,seed=7)])
        atomic(p/'config/science.json',dict(init=8))
        atomic(p/'LEGACY_AUTOSCALE.json',dict(mode='dynamic'))
        (p/'runs').mkdir();(p/'runs/chain.log').write_text('original evidence')
        result.append(p)
    return result


def test_backups_every_arm_and_preserves_science(tmp_path):
    rs=roots(tmp_path);before={p:{str(q.relative_to(p)):q.read_bytes() for q in p.rglob('*') if q.is_file()} for p in rs}
    rows=reset_many(rs,'.failed',owner_confirmed_closed_zero=True)
    for p,row in zip(rs,rows):
        backup=Path(row['backup'])
        assert all((backup/n).read_bytes()==v for n,v in before[p].items())
        assert (p/'arms.json').read_bytes()==before[p]['arms.json']
        assert not (p/'LEGACY_AUTOSCALE.json').exists() and not (p/'runs').exists()

@pytest.mark.parametrize('cause',['no_attestation','completed','accepted','backup'])
def test_refuses_all_before_renaming(tmp_path,cause):
    rs=roots(tmp_path)
    if cause=='completed':atomic(rs[1]/'state/arm.DONE',{})
    if cause=='accepted':atomic(rs[1]/'runs/arm/journal.jsonl',dict(task_uid='uid',status='failed',accepted=True))
    if cause=='backup':rs[1].with_name(rs[1].name+'.failed').mkdir()
    with pytest.raises(ValueError):reset_many(rs,'.failed',owner_confirmed_closed_zero=cause!='no_attestation')
    assert all((p/'LEGACY_AUTOSCALE.json').exists() for p in rs)
