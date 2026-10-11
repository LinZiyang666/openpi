import json
from pathlib import Path

import numpy as np
import pytest

from exp.r17_integration.debug.reader_profiles import CaptureReader
from exp.r17_integration.debug.validate import digest
from exp.r17_integration.search.fit import LibraryView, library_data, load_fitted_recipe
from exp.r17_integration.search.teacher_library import build_teacher_parent, fit_nested_libraries, RowIndexStore, calibrate_teacher_knobs
from exp.r17_integration.tests.search.library_fixture import synthetic_capture


def test_synthetic_collect_build500_nested_all_sizes_and_sealed_fits(tmp_path):
    source = tmp_path/'capture'
    b,episodes,audit,names = synthetic_capture(source)
    reader = CaptureReader(source,max_cached_blocks=2)
    first,values,keys = next(reader.decisions())
    assert set(values)=={'state','normalized_actions','image_0','image_1'}
    assert set(keys)==set(b['model']['feature_spec']['fields'])
    assert 'secret' not in first and 'diagnostics' not in first and first['capture_enabled'] is True
    assert values['image_0'].tobytes()==np.arange(48,dtype=np.uint8).reshape(4,4,3).tobytes()
    parent = build_teacher_parent([source],tmp_path/'bpool',b,episodes,audit)
    assert len(parent.rows)==1000 and len(parent.manifest['metadata']['episodes'])==500
    assert sum(not e['success'] for e in parent.manifest['metadata']['episodes'])==250
    assert parent.rows[0]['progress']==0 and parent.rows[1]['progress']==1
    assert parent.rows[0]['next_id']==parent.rows[1]['row_id']
    assert parent.rows[1]['prev_id']==parent.rows[0]['row_id'] and parent.rows[1]['next_id'] is None
    assert {r['library_episode_id'] for r in parent.rows}==set(range(500))
    products = fit_nested_libraries(parent,audit,task_names=names)
    assert set(products)=={'50','100','200','300','400','500'}
    previous = set()
    for size in (50,100,200,300,400,500):
        directory = parent.directory/'subsets'/('size_'+str(size))
        assert {p.name for p in directory.iterdir()}=={'row_indices.npy','VIEW.json'}
        view = RowIndexStore(directory,parent=parent)
        chosen = set(view.manifest['episode_ids'])
        assert len(chosen)==size and previous<=chosen
        assert np.array_equal(np.asarray(view.array('image_0')),parent.array('image_0')[view.indices])
        fit = load_fitted_recipe(parent.directory/'fits'/('size_'+str(size)),expected_binding=b)
        assert set(fit.episode)==chosen and len(fit.action)==size*2
        assert set(fit.task)==set(range(10))
        assert all(r['query_episode'] not in r['donor_episodes'] for r in fit.fit_evidence['loeo'])
        assert {e for f in fit.fit_evidence['distance_folds'] for e in f['train_episodes']+f['heldout_episodes']}==chosen
        receipt = json.loads((parent.directory/'fits'/('size_'+str(size))/'READY.json').read_text())['receipt']
        assert receipt['store_digest']==view.manifest['content_digest']
        assert receipt['serving_loeo_required_on_B']==(size==500)
        previous = chosen
    # Closed CPU finish is idempotent, and changed knobs cannot reuse a fit seal.
    assert fit_nested_libraries(parent,audit,task_names=names)==products
    pending = calibrate_teacher_knobs(parent,products,task_names=names)
    assert all(p['status']=='pending_cost_qualification' for p in pending.values())
    from exp.r17_integration.common.recipe import retrieval_model_identity
    costs = dict(schema='r17.ir_costs.v1',vision=.2,miss=.8,source_identity='synthetic-CPU-no-pricing-claim',
        sampler_identity=digest(b['sampler']),benchmark_protocol=b['benchmark_protocol'],
        retrieval_model_identity=retrieval_model_identity(b['model']))
    calibrated = calibrate_teacher_knobs(parent,{'50':products['50']},task_names=names,cost_profile=costs)
    receipt = json.loads(Path(calibrated['50']['receipt']).read_text())
    assert receipt['status']=='calibrated' and set(receipt['calibrations'])=={'0.25','0.32','0.4','0.45'}
    for value in receipt['calibrations'].values():
        assert value['library_only'] is True and value['N']==100
        assert value['store_digest']==RowIndexStore(parent.directory/'subsets/size_50',parent=parent).manifest['content_digest']
    with pytest.raises(ValueError,match='model/protocol'):
        calibrate_teacher_knobs(parent,{'50':products['50']},task_names=names,cost_profile=dict(costs,sampler_identity='different-grid'))
    with pytest.raises(ValueError,match='settings'):
        fit_nested_libraries(parent,audit,task_names=names,k=8)
    with pytest.raises(KeyError,match='allowlist'):
        LibraryView(parent).array('image_0')


def test_small_capture_and_l10_canonical_ids_do_not_remap_library_ids(tmp_path):
    b,episodes,audit,names = synthetic_capture(tmp_path/'capture',episodes=5,suite='libero_10')
    parent = build_teacher_parent([tmp_path/'capture'],tmp_path/'parent',b,episodes,audit,expected_episodes=5)
    data = library_data(parent,task_names=names)
    assert set(data.episode)==set(range(5)) and set(data.task_id)=={30}
    assert len(parent.rows)==10


@pytest.mark.parametrize('corruption',('A','wrong_id','wrong_state','error','retry','missing_episode','wrong_sampler','no_terminal','duplicate_coordinate'))
def test_bad_teacher_cohorts_cannot_publish_library(tmp_path,corruption):
    b,episodes,audit,names = synthetic_capture(tmp_path/'capture',episodes=5)
    if corruption=='A': episodes[0]['init_pool']='A'
    elif corruption=='wrong_id': episodes[0]['library_episode_id']=49
    elif corruption=='wrong_state': episodes[0]['init_state_sha256']='other'
    elif corruption=='error': episodes[0]['error']='lost controls'
    elif corruption=='retry': episodes[0]['attempt']=2
    elif corruption=='missing_episode': episodes.pop()
    elif corruption=='duplicate_coordinate': episodes[1]=dict(episodes[0],task_uid='another-uid')
    elif corruption=='wrong_sampler':
        # Strict capture binding rejects changing the teacher NFE after collect.
        b['sampler']['num_steps']=100
    else:
        path = tmp_path/'capture'/'execution.jsonl'
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        rows[1]['terminal']=False
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        # Seal the altered synthetic closed capture to test terminal semantics,
        # independently of the already covered checksum rejection.
        from exp.r17_integration.debug.validate import file_sha
        manifest = tmp_path/'capture'/'r17_capture.json'
        document = json.loads(manifest.read_text()); document['files']['execution.jsonl']=file_sha(path)
        manifest.write_text(json.dumps(document))
    with pytest.raises((ValueError,KeyError)):
        build_teacher_parent([tmp_path/'capture'],tmp_path/'parent',b,episodes,audit,expected_episodes=5)
    assert not (tmp_path/'parent'/'READY.json').exists()


def test_subset_tampering_and_forbidden_fit_reads(tmp_path):
    from exp.r17_integration.search.teacher_library import nested_views
    b,episodes,audit,names = synthetic_capture(tmp_path/'capture')
    parent = build_teacher_parent([tmp_path/'capture'],tmp_path/'parent',b,episodes,audit)
    view = nested_views(parent,audit,sizes=(50,))[50]
    expected = library_data(view,task_names=names)
    # The fitter's array source physically contains only selected indices.
    assert len(expected.state)==100 and set(expected.episode)==set(view.manifest['episode_ids'])
    with pytest.raises(KeyError): LibraryView(view).array('image_1')
    path = view.directory/'row_indices.npy'
    with path.open('wb') as f: np.save(f,np.array([0],np.int64),allow_pickle=False)
    with pytest.raises(ValueError,match='checksum'):
        RowIndexStore(view.directory,parent=parent)


def test_declared_float32_key_dtype_is_not_silently_coerced_at_fit(tmp_path):
    b,episodes,audit,names = synthetic_capture(tmp_path/'capture',episodes=2,bad_key_dtype=True)
    with pytest.raises(ValueError,match='retrieval field geometry'):
        build_teacher_parent([tmp_path/'capture'],tmp_path/'parent',b,episodes,audit,expected_episodes=2)
    assert not (tmp_path/'parent'/'READY.json').exists()
