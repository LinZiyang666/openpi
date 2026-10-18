import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from . import vla_full as v


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def test_candidates_preserve_metric_corrector_guard_and_caps():
    table=NS(rows=np.array([0,1,2]),Z=np.ones((3,2)),z2=np.ones(3),HD=None,h2=None,RS=None,rs2=None,
             V0=None,V1=None,Z0=None,n20=None,Wf=np.eye(2),head=np.ones(2))
    base=NS(tasks={0:table},lib_ep=np.array([1,1,2]),pca={'frozen':np.eye(2)})
    method=NS(inner=NS(base=base,C=np.ones((2,2))),knob_settings={'setting':3.25},heads={0:np.ones(2)})
    source=dict(method=method);filtered=v.candidates(source,[2]);v.unchanged(source,filtered)
    assert filtered['method'].inner.base.tasks[0].rows.tolist()==[2]
    assert filtered['method'].knob_settings is method.knob_settings


def test_historical_baselines_use_r14_only_at_size50_without_rerun_controls(tmp_path):
    root=tmp_path/'new';source=tmp_path/'r15';r14=tmp_path/'r14';trace=tmp_path/'trace'
    old=dict(y=np.zeros((10,50),int).tolist(),cost=np.ones((10,50)).tolist(),n=np.ones((10,50)).tolist())
    write(r14/'test_a.json',dict(episode_data={'pi05_l10_50':{'task_descent':old}},levels={'pi05_l10_50':{'0':'ir25'}}))
    write(source/'test_a.json',dict(episode_data={'pi05_l10_100':{'task_descent':old}},levels={'pi05_l10_100':{'0':'ir40'}}))
    write(source/'protocol.json',dict(input_sha256={}))
    (trace/'r11_knob_1').mkdir(parents=True);(trace/'r11_knob_1/eval500.json').write_text('[]')
    write(root/'CELLS.json',dict(cells=[dict(cell=f'pi05_l10_{n}',model='pi05',suite='libero_10',size=n,
                                          collected=n,kept=n-6,failed=6,empty_tasks=[]) for n in (50,100)]))
    def complete(cell):
        arm='filtered';path=v.root_for(cell,arm,'test')
        write(path/'descent_table.json',dict(levels={},uniform='off'))
        write(path/'arms.json',[dict(arm='test')]);write(path/'runs/test/summary.json',{})
        write(root/arm/'cap_tables.json',{})
        frozen=root/arm/'freeze/pi05'
        write(frozen/'levels.json',{f'pi05_l10_{n}':{} for n in (50,100)})
        write(frozen/'uniform.json',{f'pi05_l10_{n}':'off' for n in (50,100)})
        write(frozen/'FREEZE.json',dict(levels_sha256=v.sha(frozen/'levels.json'),uniform_sha256=v.sha(frozen/'uniform.json'),frozen_at_utc='2000-01-01T00:00:00+00:00'))
    def load(*args):
        assert v.aa.TRACE.arm=='filtered'
        y=np.ones((10,50),int)
        return {'task_descent':NS(y=y,cost=np.ones((10,50)),n=np.ones((10,50)),audit=dict(earliest_accepted_terminal_ts=1792100000))}
    with patch.object(v,'ROOT',root),patch.object(v,'SOURCE',source),patch.object(v,'R14_SOURCE',r14),patch.object(v,'TRACE',trace), \
         patch.object(v,'protocol',return_value={}),patch.object(v.aa,'load_a',side_effect=load):
        v.analyze()
        pending=json.loads((root/'RESULTS.json').read_text())['rows']
        assert all(r['status']=='PENDING' and r['unfiltered']['n']==500 and r['paired'] is None for r in pending)
        for n in (50,100):complete(f'pi05_l10_{n}')
        v.analyze()
        for row in json.loads((root/'RESULTS.json').read_text())['rows']:
            assert row['status']=='COMPLETE' and row['paired']['paired_n']==500
            assert row['paired']['difference']==1 and set(row['paired']['by_init'].values())=={1}
            historical=r14 if row['size']==50 else source
            assert row['unfiltered_source']==str(historical/'test_a.json')
            assert row['unfiltered_source_sha256']==v.sha(historical/'test_a.json')
            assert row['unfiltered_pipeline']==('R14' if row['size']==50 else 'R15')
            assert row['cross_pipeline_comparison']==(row['size']==50)
            assert row['comparison_pipeline']==('R14→R15' if row['size']==50 else 'R15→R15')
        assert not (root/'unfiltered').exists()


def test_layout_maps_test_loader_to_new_campaign_root(tmp_path):
    with patch.object(v,'ROOT',tmp_path):
        actual=v.Layout('filtered')/'r15_descent_test_groot_spatial_50'
    assert actual.name=='ffk1015_groot_libero_full_r15_spatial_50_filtered_test'


def test_freeze_uses_native_first_failure_even_when_lower_levels_recover(tmp_path):
    write(tmp_path/'selection.json',{})
    write(tmp_path/'filtered/cap_tables.json',{})
    def episode(success):
        ones=np.ones((10,30),int)
        return v.a.Episodes(ones*success,ones,ones,ones,ones.astype(float),{})
    def load(root,*args):
        assert isinstance(root,Path)
        if v.a.ARMS==('c10',):return {'c10':episode(1)}
        return {level:episode(int(level!='ir40')) for level in v.p.DEV_ORDER}
    with patch.object(v,'ROOT',tmp_path),patch.object(v.p,'REPO',tmp_path),patch.object(v,'arms',return_value=('pi05_l10_50',)), \
         patch.object(v.a,'load_cell',side_effect=load):
        v.freeze('pi05')
        levels,uniform,freeze=v.frozen_tables('filtered','pi05')
        assert set(levels['pi05_l10_50'].values())=={'ir45'}
        assert uniform['pi05_l10_50']=='ir45' and freeze['frozen_at_utc']
        assert not (tmp_path/'unfiltered').exists()


def test_unfiltered_campaign_catalog_is_not_available():
    assert v.ARMS==('filtered',)
    try:v.arms('unfiltered')
    except ValueError:pass
    else:raise AssertionError('Unfiltered knob controls must not be defined')
