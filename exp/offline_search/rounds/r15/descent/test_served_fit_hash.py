"""Synthetic relocation proofs and byte-preservation of every other loader check."""
import ast
import copy
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from exp.offline_search.closed_loop.ops.h100.assets import relocated_fit
from . import analyze as a
from .test_analysis import dump, fake_b, jl, seal
from .verify_fit_hashes import decision_hashes


def relocated_b(tmp_path,cell):
    root,p,selection,caps=fake_b(tmp_path,cell)
    store=tmp_path/'store'
    for source in (tmp_path/'r15_descent_c10_pi05_l10',root):
        rows=json.loads((source/'arms.json').read_bytes())
        contracts=json.loads((source/'dev_contracts.json').read_bytes())
        for row in rows:
            row.update(intended_host='h100')
            row['plugin_args']+=['--os-root',str(store)]
            path=Path(row['plugin_args'][1])
            blob=dict(store=str(store/'library/synthetic'),run=str(source/'runs'/row['arm']),
                      external='/home/weiland/synthetic/library.pkl',unchanged=[1,2,3])
            original=pickle.dumps(blob,protocol=4);path.write_bytes(original)
            digest=a.sha(original);level=row['r14_descent_level']
            source_cell='pi05_l10_100' if level=='c10' else cell
            caps['cells'][source_cell]['fit_sha256_by_level'][level]=digest
            contracts[row['arm']]['fit_sha256']=digest
            dump(source/'runs'/row['arm']/'pool_contract.json',contracts[row['arm']])
            moved,changed=relocated_fit(copy.deepcopy(blob),source,store,Path('/custom/h100'))
            assert changed
            served=a.sha(pickle.dumps(moved,protocol=4))
            assert served!=digest
            for shard in (source/'runs'/row['arm']).glob('server_*/decisions_*.jsonl'):
                decisions=[json.loads(line) for line in shard.read_bytes().splitlines()]
                for d in decisions:
                    d['r14']['fit_sha256']=served
                jl(shard,decisions)
        dump(source/'arms.json',rows);dump(source/'dev_contracts.json',contracts)
        seal(source,'pi05_l10_100' if source!=root else cell,'B',rows)
        # As in real sync, only the last arm remains in the plan. Earlier arm
        # hashes must be recomputed using the actual nondefault remote base.
        dump(source/'h100_sync/plan.json',dict(run=str(source),
             remote_run='/custom/h100/runs/'+source.name,arms=[rows[-1]],files=[]))
    return root,p,selection,caps


@pytest.mark.parametrize('cell',('pi05_l10_100','pi05_l10_500'))
def test_loader_accepts_deterministic_relocation_and_records_branch(cell,tmp_path):
    root,p,selection,caps=relocated_b(tmp_path,cell)
    reader=a.Reader()
    arms=a.load_cell(root,cell,p,selection,caps,reader)
    assert all(ep.y.sum()==299 and ep.n.sum()==300 for ep in arms.values())
    for level,ep in arms.items():
        audit=ep.audit['fit_hash_verification']
        assert audit['relocated_sha256']!=audit['source_sha256']
        assert list(audit['observed_sha256_branches'].values())==['h100_relocated']
        assert audit['remote_run'].startswith('/custom/h100/runs/')
    reader.stable()


def test_loader_rejects_unrelated_hash_even_if_plan_claims_it(tmp_path):
    root,p,selection,caps=relocated_b(tmp_path,'pi05_l10_100')
    name=a.arm_name('pi05_l10_100','ir45')
    shard=next((root/'runs'/name).glob('server_*/decisions_*.jsonl'))
    rows=[json.loads(line) for line in shard.read_bytes().splitlines()]
    unrelated=a.sha(b'unrelated fit')
    for row in rows:
        row['r14']['fit_sha256']=unrelated
    jl(shard,rows)
    plan=json.loads((root/'h100_sync/plan.json').read_bytes())
    plan['files']=[dict(original=str(tmp_path/'pi05_l10_100_ir45.bin'),sha256=unrelated)]
    dump(root/'h100_sync/plan.json',plan)
    with pytest.raises(ValueError,match='neither frozen source nor deterministic H100 relocation'):
        a.load_cell(root,'pi05_l10_100',p,selection,caps,a.Reader())


def test_source_branch_and_invalid_remote_root(tmp_path):
    root,p,selection,caps=fake_b(tmp_path,'pi05_l10_100')
    arms=a.load_cell(root,'pi05_l10_100',p,selection,caps,a.Reader())
    assert all(list(ep.audit['fit_hash_verification']['observed_sha256_branches'].values())==['frozen_source']
               for ep in arms.values())
    root,p,selection,caps=relocated_b(tmp_path/'relocated','pi05_l10_100')
    path=tmp_path/'relocated/r15_descent_c10_pi05_l10/h100_sync/plan.json'
    plan=json.loads(path.read_bytes());plan['remote_run']='/custom/h100/runs/another_root'
    dump(path,plan)
    with pytest.raises(ValueError,match='Invalid H100 remote run path'):
        a.load_cell(root,'pi05_l10_100',p,selection,caps,a.Reader())


def test_unchanged_relocation_preserves_original_pickle_bytes(tmp_path):
    root=tmp_path/'r15_descent_pi05_l10_100';root.mkdir()
    path=tmp_path/'fit.pkl';data=pickle.dumps({'numbers':[1,2,3]},protocol=5);path.write_bytes(data)
    dump(root/'h100_sync/plan.json',dict(run=str(root),remote_run='/custom/h100/runs/'+root.name))
    row=dict(intended_host='h100',plugin_args=['--os-root',str(tmp_path/'store'),'--os-fit-artifact',str(path)])
    verifier=a.ServedFitVerifier(root,row,a.sha(data),a.Reader())
    assert verifier.relocated_hash()==a.sha(data)
    assert verifier.check(a.sha(data))=='frozen_source'


def test_fresh_relocation_hash_is_independent_of_previously_unpickled_artifacts(tmp_path):
    root=tmp_path/'r15_descent_pi05_l10_100';store=tmp_path/'store';base=Path('/custom/h100')
    blobs=[pickle.dumps(dict(path=str(root/'runs'),arrays=[np.ones((3,2),dtype=dtype),
        np.zeros((2,4),dtype=dtype)]),protocol=4) for dtype in ('float32','float64')]
    first=a.h100_relocated_sha256(blobs[0],root,store,base)
    assert first!=a.sha(blobs[0])
    a.h100_relocated_sha256(blobs[1],root,store,base)
    # Populate parent dtype/pickle state as well; it cannot influence children.
    for data in blobs:
        pickle.loads(data)
    assert a.h100_relocated_sha256(blobs[0],root,store,base)==first


def test_hash_only_shard_snapshot_defers_an_incomplete_running_tail(tmp_path):
    path=tmp_path/'decisions_0.jsonl';value=a.sha(b'frozen synthetic fit')
    rows=[dict(ev='dec',r14=dict(fit_sha256=value)),dict(ev='other')]
    tail=b'{"ev": "dec", "r14": '
    path.write_bytes(b''.join(json.dumps(r).encode()+b'\n' for r in rows)+tail)
    hashes,snapshot=decision_hashes(path)
    assert hashes=={value} and snapshot['decisions_with_fit_hash']==1
    assert snapshot['incomplete_tail_bytes']==len(tail)
    assert snapshot['snapshot_bytes']==path.stat().st_size


def test_only_served_hash_validation_and_audit_changed_in_loader():
    def loader(path):
        return next(n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='load_cell')
    before=loader(a.HERE/'analyze.before_served_hash_fix.py.txt')
    after=loader(a.HERE/'analyze.py')
    class Restore(ast.NodeTransformer):
        def visit_Assign(self,node):
            if any(isinstance(t,ast.Name) and t.id=='fit_verifier' for t in node.targets):
                return None
            return self.generic_visit(node)
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and \
                    isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='fit_verifier':
                return ast.parse("require(diag.get('fit_sha256')==fit_digest, 'Wrong served fit hash')").body[0]
            return self.generic_visit(node)
        def visit_Call(self,node):
            node.keywords=[k for k in node.keywords if k.arg!='fit_hash_verification']
            return self.generic_visit(node)
    restored=Restore().visit(after)
    assert ast.dump(restored,include_attributes=False)==ast.dump(before,include_attributes=False)
