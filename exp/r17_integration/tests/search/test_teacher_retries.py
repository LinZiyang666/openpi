import pytest

from exp.r17_integration.search.teacher_library import build_teacher_parent
from exp.r17_integration.tests.search.library_fixture import synthetic_capture

EGL = "FatalError('Offscreen framebuffer is not complete, error 0x8cdd')"


def prior(episode, **extra):
    return dict(task_uid=episode['task_uid'],attempt=1,success=False,error=EGL,n_steps=0,
                accepted=True,capture_decisions=0,reported_decisions=0,**extra)


def test_parent_accepts_proven_zero_decision_retry_and_seals_provenance(tmp_path):
    b,episodes,audit,_ = synthetic_capture(tmp_path/'capture',episodes=1,attempt=2)
    episodes[0]['prior_attempts']=[prior(episodes[0])]
    parent=build_teacher_parent([tmp_path/'capture'],tmp_path/'parent',b,episodes,audit,expected_episodes=1)
    assert len(parent.rows)==2 and all(r['attempt']==2 for r in parent.rows)
    assert parent.manifest['metadata']['pre_decision_infrastructure_attempts']==episodes[0]['prior_attempts']


@pytest.mark.parametrize('bad',['capture','step','outcome','missing','noncontiguous','unknown','reported','late_capture'])
def test_parent_cannot_hide_prior_actions_or_cherry_pick_failed_outcomes(tmp_path,bad):
    b,episodes,audit,_ = synthetic_capture(tmp_path/'retry',episodes=1,attempt=2)
    row=prior(episodes[0]); episodes[0]['prior_attempts']=[row]
    sources=[tmp_path/'retry']
    if bad in ('capture','late_capture'):
        synthetic_capture(tmp_path/'original',episodes=1,attempt=1)
        sources.append(tmp_path/'original')
    elif bad=='step': row['n_steps']=1
    elif bad=='outcome': row['error']=None
    elif bad=='missing': episodes[0]['prior_attempts']=[]
    elif bad=='noncontiguous': row['attempt']=2
    elif bad=='unknown': row['evidence_unknown']=True
    else: row['reported_decisions']=1
    with pytest.raises(ValueError):
        build_teacher_parent(sources,tmp_path/'parent',b,episodes,audit,expected_episodes=1)
    assert not (tmp_path/'parent/READY.json').exists()
