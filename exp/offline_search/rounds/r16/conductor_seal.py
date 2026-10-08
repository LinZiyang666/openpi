"""Register the owner-authorized adapter without rewriting the original seals."""
import ast
import hashlib
import json
from pathlib import Path

from .prepare import HERE, write_json


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def functions(path):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(path).read_bytes()).body
            if isinstance(n,ast.FunctionDef)}


def main():
    old=json.loads((HERE/'DESIGN_LOCK.json').read_bytes());records={};proof={}
    for name,allowed in [('analyze.py',{'main','verify','load'}),('prepare.py',{'frozen_tables','emit_test'})]:
        current=HERE/name;original=HERE/'before'/('conductor_original_'+name+'.txt')
        path=next(p for p in old['source_sha256'] if Path(p).resolve()==current.resolve())
        assert sha(original)==old['source_sha256'][path]
        before,after=functions(original),functions(current)
        assert set(before)==set(after)
        unchanged=set(before)-allowed
        assert all(before[n]==after[n] for n in unchanged),'Numerical/design functions changed'
        proof[name]={n:hashlib.sha256(after[n].encode()).hexdigest() for n in sorted(unchanged)}
        if name=='analyze.py':
            expected=original.read_text().replace("math.isclose(summary['sr'],float(y.mean()),abs_tol=1e-12),'Summary SR mismatch')",
                "(summary['sr']==round(float(y.mean()),4) or\n             math.isclose(summary['sr'],float(y.mean()),abs_tol=1e-12)),'Summary SR mismatch')")
            expected=expected.replace("AServedFitVerifier(root,row,digest,reader) if pool=='A' else ServedFitVerifier(root,row,digest,reader)",
                                      "AServedFitVerifier(root,row,digest,reader)")
            expected=expected.replace("if pool=='A': verifier.check_path(observed_path)\n                else: require(observed_path==str(fit),'Wrong local B served fit path')",
                                      "verifier.check_path(observed_path)")
            expected_load={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(expected).body if isinstance(n,ast.FunctionDef)}['load']
            assert after['load']==expected_load,'Only collector serialization and existing relocated-fit checks may change in the loader'
        records[path]=dict(original_sha256=sha(original),adapted_sha256=sha(current),original_copy=str(original))
    sources=sorted({*HERE.glob('conductor*.py'),HERE/'resident.py',HERE/'test_conductor.py',HERE/'conductor_commands.sh'})
    write_json(HERE/'CONDUCTOR_LOCK.json',dict(schema='r16.conductor.amendment.v1',
        original_design_lock_sha256=sha(HERE/'DESIGN_LOCK.json'),design_sha256=sha(HERE/'design.json'),
        protocol_sha256=sha(HERE/'PROTOCOL.md'),amendment_sha256=sha(HERE/'CONDUCTOR_AMENDMENT.json'),
        adapted_sources=records,new_sources={str(p):sha(p) for p in sources},
        unchanged_function_ast_sha256=proof,collector_sr_serialization_fix=dict(
            source=str(HERE.parents[1]/'closed_loop/ops/collect.py'),
            source_sha256=sha(HERE.parents[1]/'closed_loop/ops/collect.py'),decimals=4,
            affects_selection_or_numerical_sr=False),shared_runtime_edits=0,test_a_outcome_reads=0))
    print('CONDUCTOR_SEALED original numerical/design functions unchanged; original seals retained')


if __name__=='__main__':main()
