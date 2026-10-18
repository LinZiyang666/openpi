"""Recognize only the explicitly disabled-cache pure control in native audit.

All native NFE, binding, ACK, error, episode coverage and lifetime checks stay.
"""
SUFFIX='''
_native_empty_uids=empty_uids
def empty_uids(directory):
    import yaml
    root=Path(directory);c=json.loads((root/'campaign.json').read_text())
    kind=c.get('longseg_1017_control_kind')
    if kind not in ('pure','long'):return _native_empty_uids(directory)
    assert c.get('longseg_1017',{}).get('clock_unit')=='segment_decision'
    study=json.loads((root/'assets/CACHE_STUDY.json').read_text())
    for row in study['groups']:
        doc=yaml.safe_load((root/'assets'/row['file']).read_text())
        assert row['group'].endswith('_pure') and not doc['enabled'] and not doc['checkpoints']
        assert not doc['recipe']['enabled'] and not doc['recipe']['corrector'] and not doc['recipe']['guard']
        assert doc['adapter_binding']['commit_horizon']==(10 if kind=='long' else 5)
    name='FULL_EPISODE_PLAN.json' if c.get('cache_resume_selection') else 'EPISODE_PLAN.json'
    return {t['task_uid'] for t in json.loads((root/name).read_text())['tasks']}
'''
