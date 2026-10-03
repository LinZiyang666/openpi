"""Read every listed frozen input in full; bind numerical tables and protected files."""
import json
from pathlib import Path
import numpy as np
from exp.offline_search.rounds.r10.data import sha
HERE = Path(__file__).resolve().parent
R11 = HERE.parent

def main():
    paths = [R11/'BRIEF.md', R11/'devset/HANDBACK.md',
             R11.parents[1]/'closed_loop/devset.py']
    paths += [R11/'opus'/n for n in ('SPEC.md','IR_MODEL.md','HANDBACK.md','out/arm_grid.json',
              'ir_model.py','loeo_anchor_table.py','offline_curves.py','arm_grid.py')]
    paths += [R11/'astra'/n for n in ('SPEC.md','HANDBACK.md','COORDINATION.md','POST_FREEZE_NOTES.md',
              'freeze.json','arm_grid.csv','experiment.py','signals.py','analyze.py','controller.py','crosscheck.py','freeze.py')]
    paths += list((R11/'astra/data').glob('*/*'))
    paths += list((R11.parent/'r10/recipe').glob('*.py')) + [R11.parent/'r10/recipe/README.md']
    paths += list((R11/'opus/out/anchor').glob('*.npz'))
    report = []
    for p in sorted(set(paths)):
        if p.suffix == '.npz':
            with np.load(p, allow_pickle=False) as z:
                arrays = {k:np.array(z[k]) for k in z.files}
            details = {k:dict(shape=list(v.shape),dtype=str(v.dtype),
                       finite=bool(np.isfinite(v).all()) if v.dtype.kind in 'fci' else None)
                       for k,v in arrays.items()}
        else:
            content = p.read_text()
            details = dict(lines=len(content.splitlines()))
            if p.suffix == '.json':
                obj=json.loads(content)
                details['keys_or_rows'] = list(obj) if isinstance(obj,dict) else len(obj)
        report.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,details=details))
    output=HERE/'input_audit.json'
    if output.exists():
        assert json.loads(output.read_text())==report, 'protected input snapshot changed'
    else:
        output.write_text(json.dumps(report,indent=1)+'\n')
    print('INPUTS_READ',len(report),'bytes',sum(r['bytes'] for r in report))
    for p in (R11/'opus/out/arm_grid.json', R11/'astra/freeze.json'):
        obj=json.loads(p.read_text()); rows=obj['arms'] if isinstance(obj,dict) else obj
        print(p,len(rows))
        for r in rows:
            keys=('cell','method','target','target_ir','setting','dose','threshold','tie_probability','beta','eta')
            print({k:r[k] for k in keys if k in r})
    for p in sorted((R11/'astra/data').glob('*/predictor.json')):
        print(p, json.loads(p.read_text()))

if __name__=='__main__': main()
