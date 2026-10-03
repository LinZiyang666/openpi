"""Task descriptions from exactly one admitted init-0 P10 episode per task."""
import json
from .analyze import compact
from .safe import HERE, RUNS, admitted, json_identity, dump


def main():
    a=compact('groot','P10',['decision_id','seq'])
    out={}
    for t in range(10):
        keep=(a['task']==t)&(a['init']==0)&(a['seq']==0)
        assert keep.sum()==1
        ep=str(a['decision_id'][keep][0]).split(':')[0]
        assert '/' not in ep and '..' not in ep
        path=admitted(RUNS/'r08_main/runs/r8_groot_l10_P10/debug/client'/ep/'episode.json')
        raw=path.read_text()
        assert json_identity(raw)==(t,0)
        r=json.loads(raw)
        out[str(t)]={k:r[k] for k in ('task_text','task_id','init','task_uid','env_seed')}
        out[str(t)]['source']=str(path)
    dump(HERE/'results/task_metadata.json',out)


if __name__=='__main__':main()
