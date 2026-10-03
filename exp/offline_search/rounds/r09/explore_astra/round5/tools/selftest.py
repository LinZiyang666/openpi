"""Run the real CPU plugin selftest through an identity-first QueryCell adapter.

Never call the stock QueryCell constructor on a raw mixed-init store. Only two
admitted training episodes (task 0/1, init 0) are read, and only their first 24
rows. All generated captures live in the new run root. No GPU/sim/network.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from .safe import HERE, RUN, STORE, dump, episode_metadata, records
from .build import SAFE_STORE


def install_filtered_query_cell():
    from exp.offline_search.harness import store
    Original=store.QueryCell

    class FilteredQueryCell(Original):
        def __init__(self, root, cell):
            self.root=Path(root);self.cell=cell
            self.model,self.suite,self.arm=store.parse_cell(cell)
            self.lib_key=f'{self.model}_{self.suite}'
            self.dir=STORE/'queries'/cell
            eps=sorted(episode_metadata(self.dir/'episodes.json',pairs={(0,0),(1,0)}),key=lambda e:e['task_id'])
            assert len(eps)==2
            self.selected_rows=[];self.episodes=[];self._a={}
            for e in eps:
                assert 0<=int(e['init'])<20 and e['end']-e['start']>=24
                start=len(self.selected_rows)
                self.selected_rows.extend(range(e['start'],e['start']+24))
                self.episodes.append(dict(e,start=start,end=start+24,num_steps=24))
            self.selected_rows=np.asarray(self.selected_rows,np.int64)

        def _arr(self,name):
            if name not in self._a:
                if name=='ep':
                    self._a[name]=np.repeat(np.arange(2),24)
                else:
                    # np.load with mmap_mode reads the array header, not payload.
                    # Gather only rows whose episode identity was admitted above.
                    source=np.load(self.dir/f'{name}.npy',mmap_mode='r',allow_pickle=False)
                    self._a[name]=np.array(source[self.selected_rows])
            return self._a[name]
    store.QueryCell=FilteredQueryCell


def one(name,forced):
    from exp.offline_search.closed_loop import selftest
    rows={a['arm']:a for a in json.loads((RUN/'arms.json').read_text())};a=rows[name]
    kw=dict(a['kwargs'])
    suffix='forced' if forced else 'prod'
    out=RUN/'selftest'/f'{name}_{suffix}'
    if (out/'selftest_report.json').exists():
        raise ValueError('successful selftest output already exists; use cached batch evidence')
    if out.exists():
        # Preserve failed evidence and give the plugin a clean output directory.
        n=1
        while out.with_name(out.name+f'_failed{n}').exists():n+=1
        out.rename(out.with_name(out.name+f'_failed{n}'))
    argv=['--cell',a['cell'],'--yaml',a['yaml'],'--method',a['method'],'--root',str(SAFE_STORE),
          '--blind','--policy-tail','--policy-tail-blocks','1','--judge','guard_only','--no-shadow',
          '--out',str(out)]
    if forced:
        if name.endswith('_control'):
            # Exercise imported control's existing guard/grasp coexistence path.
            # Control production artifact/kwargs remain exactly unchanged.
            kw.update(max_calls=2,force_trigger_at=[2,6])
        else:
            kw.update(force_at=[2])
    else:
        argv+=['--fit-artifact',a['plugin_args'][a['plugin_args'].index('--os-fit-artifact')+1]]
    argv+=['--kwargs',json.dumps(kw)]
    install_filtered_query_cell()
    rc=selftest.main(argv)
    report=json.loads((out/'selftest_report.json').read_text())
    decs=[r for _,r in records(next(out.glob('decisions_*.jsonl')),lo=0,hi=20) if r.get('ev')=='dec']
    assert len(decs)==48 and report['PASS'] and rc==0
    assert not forced or report['miss']>0
    if forced and not name.endswith('_control'):
        starts=[r for r in decs if r.get('extras',{}).get('r9a5_start')]
        assert len(starts)==4, 'must exercise takeover after every connection/episode reset'
        assert all(r['step']==2 for r in starts)
        assert report['policy_tail']>0
    dump(out/'admission.json',dict(allowed_pairs=[[0,0],[1,0]],rows_per_episode=24,
         forced=forced,production_unchanged=True,new_trigger_verified=forced and not name.endswith('_control')))
    return rc


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm');ap.add_argument('--forced',action='store_true');a=ap.parse_args()
    if a.arm:return one(a.arm,a.forced)
    results=[]
    for row in json.loads((RUN/'arms.json').read_text()):
        for forced in [False,True]:
            name=row['arm'];suffix='forced' if forced else 'prod'
            log=RUN/'selftest'/f'{name}_{suffix}.log';log.parent.mkdir(parents=True,exist_ok=True)
            report=RUN/'selftest'/f'{name}_{suffix}'/'selftest_report.json'
            admission=report.with_name('admission.json')
            if report.exists() and admission.exists():
                info=json.loads(report.read_text())
                assert info['PASS']
                results.append(dict(arm=name,forced=forced,returncode=0,report=info))
                print(name,suffix,'PASS (existing)',flush=True)
                continue
            cmd=[sys.executable,'-m',__package__+'.selftest','--arm',name]+(['--forced'] if forced else [])
            with log.open('w') as f:
                p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=False)
            info=json.loads(report.read_text()) if report.exists() else {}
            results.append(dict(arm=name,forced=forced,returncode=p.returncode,report=info))
            print(name,suffix,'PASS' if p.returncode==0 else 'FAIL',flush=True)
            if p.returncode:
                raise SystemExit(f'CPU selftest failed; inspect {log}')
    dump(HERE/'results/plugin_selftests.json',results)


if __name__=='__main__':raise SystemExit(main())
