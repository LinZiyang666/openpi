"""Attested 500 worker opt-in and fail-closed invalid/toggled contracts."""
import copy,json
from pathlib import Path
import pytest,yaml
from exp.offline_search.closed_loop import devset,worker_pool
from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import load_manifest
from .prepare import HERE,TRACE

@pytest.mark.parametrize('cell',('pi05_l10_500','pi05_spatial_500','groot_l10_500','groot_spatial_500'))
def test_actual_500_worker_contract_and_default_rejection(cell,tmp_path,monkeypatch):
    root=TRACE/('r15_descent_'+cell)
    row=json.loads((root/'arms.json').read_text())[0]
    contract=devset.arm_contract(root,row,load_manifest(row['manifest']))
    matrix=tmp_path/'matrix.yaml';matrix.write_text(yaml.safe_dump(dict(arms=[dict(arm=row['arm'])])))
    path=tmp_path/'contract.json'
    args=['--task-suite',row['suite'],'--arm-matrix',str(matrix),'--journal',str(tmp_path/'journal.jsonl')]
    monkeypatch.setenv('OSCL_INIT_POOL','B');monkeypatch.setenv('OSCL_POOL_CONTRACT',str(path))
    def check(value):
        path.write_text(json.dumps(value));monkeypatch.setenv('OSCL_POOL_CONTRACT_SHA256',devset.sha(path))
        return worker_pool.validate(args,load_manifest(row['manifest']))
    assert check(contract)[0]=='B'
    for change in (dict(leave_own_episode_out=False),dict(query_pool='A'),dict(exclusion='none'),
                   dict(leave_own_episode_out='true'),dict(fits_include_own_episode=False)):
        with pytest.raises(ValueError):check(dict(contract,**change))
    # Root claims alone cannot waive disjointness without the attested method.
    disabled=dict(row,r15_leave_own_episode_out=False)
    with pytest.raises(ValueError,match='DEV_LIBRARY_OVERLAP'):
        devset.arm_contract(root,disabled,load_manifest(row['manifest']))
    malformed=dict(row,r15_leave_own_episode_out='true')
    with pytest.raises(ValueError,match='boolean'):
        devset.arm_contract(root,malformed,load_manifest(row['manifest']))
