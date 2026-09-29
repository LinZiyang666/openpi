"""Copy deployed B specs and transplant their fits without refitting a metric."""
import copy
import json
from pathlib import Path
import pickle

from exp.offline_search.closed_loop.plugin import load_method_class
from exp.offline_search.rounds.r06.p1_groot_commit.make_arms import source_rows
from exp.offline_search.rounds.r06.ideation_Q3.stall.fit_models import sha

HERE=Path(__file__).resolve().parent
OUT=Path('/tmp/q3_stall_fits/bmech')
RUNS=Path('/home/weiland/trace_runs/os_closed_loop')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[];notes=[]
    for (model,kind,cell),entry in sorted(source_rows().items()):
        if kind!='B' or not cell.endswith('_50'):continue
        source=entry['row'];row=copy.deepcopy(source)
        name=f'r6p5_bmech_{"p" if model=="pi05" else "g"}_{cell}'
        source_name=source['name']
        run='r05_q1' if model=='pi05' else 'r06_paper'
        source_path=RUNS/run/'fits'/(source_name+'.pkl')
        source_cls,_=load_method_class(source['method']);source_cls(**source['kwargs'])
        with source_path.open('rb') as f:blob=pickle.load(f)
        assert blob['spec']==source['method'] and blob['kwargs']==source['kwargs']
        row['name']=name
        row['method']='exp.offline_search.rounds.r06.ideation_Q3.stall.bmech:'+('BmechCommitJudge' if model=='pi05' else 'BmechGrootCommitJudge')
        args=row['plugin_args']
        if '--os-root' not in args:args[:0]=['--os-root','/home/weiland/trace_runs/offline_search_store']
        else:args[args.index('--os-root')+1]='/home/weiland/trace_runs/offline_search_store'
        if '--os-fit-artifact' not in args:args.extend(['--os-fit-artifact',f'<RUN>/fits/{name}.pkl'])
        else:args[args.index('--os-fit-artifact')+1]=f'<RUN>/fits/{name}.pkl'
        cls,_=load_method_class(row['method']);method=cls(**row['kwargs'])
        method.__dict__.update(blob['method'].__dict__)
        method._q3_mask_no_progress=True
        fitted={**blob,'method':method,'spec':row['method'],'kwargs':row['kwargs']}
        fitted['q3_provenance']=dict(source_fit=str(source_path),source_sha256=sha(source_path),
                                    operation='no-refit class transplant; only NP output verdict masked')
        path=OUT/(name+'.pkl')
        with path.open('xb') as f:pickle.dump(fitted,f,protocol=4)
        rows.append(row)
        notes.append(dict(name=name,source_spec=source,artifact=str(path),sha256=sha(path),
                          **fitted['q3_provenance']))
    assert len(rows)==4
    (HERE/'emit_arms_bmech.json').write_text(json.dumps(rows,indent=2)+'\n')
    (HERE/'bmech_artifacts.json').write_text(json.dumps(notes,indent=2)+'\n')
    print(json.dumps(dict(arms=len(rows),artifacts=[n['artifact'] for n in notes])))


if __name__=='__main__':main()
