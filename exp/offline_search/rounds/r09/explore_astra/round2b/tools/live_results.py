"""Read ONLY coordinator's explicitly eval20..29 round2 root, never holdouts.

Identity regex precedes JSON payload parsing. Do not open arbitrary run roots.
Only complete 100-pair arms receive SR/paired estimates; partial counts are
status only. This is read-only and never touches running processes or configs.
"""
import json
import re
from datetime import datetime,timezone
from pathlib import Path
from scipy.stats import binomtest
from .data import HERE,dump

ROOT=Path('/home/weiland/trace_runs/os_closed_loop/r09_astra_round2_eval')


def admitted(line):
    match=re.search(r'"task_uid"\s*:\s*"([^"\\]+)"',line)
    if not match:return None
    try:task,init=map(int,match.group(1).rsplit(':',2)[-2:])
    except ValueError:return None
    if not (0<=task<10 and 20<=init<30):return None  # BEFORE outcome parsing
    return task,init


def main():
    manifest=json.loads((ROOT/'manifests/eval100.json').read_text())
    pairs={(x['task'],x['init']) for x in manifest['selected']}
    expected={(t,i) for t in range(10) for i in range(20,30)}
    if pairs!=expected:raise ValueError('Refuse non-eval100 manifest')
    results={};ledgers={}
    for spec in json.loads((ROOT/'arms.json').read_text()):
        name=spec['arm']
        if not name.startswith('r9r2_astra_'):raise ValueError('arm allowlist')
        path=ROOT/'runs'/name/'client/journal.jsonl'
        if not path.exists():continue
        acc={}
        with path.open() as f:
            for line in f:
                identity=admitted(line)
                if identity is None:continue
                row=json.loads(line)
                if row.get('accepted') and row.get('success') is not None and not row.get('error'):
                    value=bool(row['success'])
                    if identity in acc and value!=acc[identity]:raise ValueError('conflicting accepted result')
                    acc[identity]=value
        result=dict(n=len(acc),complete=set(acc)==expected)
        if result['complete']:result['sr']=sum(acc.values())/100;ledgers[name]=acc
        results[name]=result
    paired={}
    for name,ledger in ledgers.items():
        if name.endswith(('_cache','_P10')):continue
        base=name.rsplit('_',1)[0]+'_cache'
        if base not in ledgers:continue
        wins=sum(ledger[x] and not ledgers[base][x] for x in expected)
        losses=sum(not ledger[x] and ledgers[base][x] for x in expected)
        paired[name]=dict(cache=base,wins=wins,losses=losses,delta=(wins-losses)/100,
            mcnemar_exact_p=binomtest(wins,wins+losses,.5).pvalue if wins+losses else 1.)
    out=dict(snapshot_utc=datetime.now(timezone.utc).isoformat(),root=str(ROOT),
        allowed_inits=list(range(20,30)),arms=results,paired=paired,
        warning='Snapshot of coordinator run; absent/incomplete arms are not failures. No process/config touched.')
    dump(HERE/'results/round2_live_snapshot.json',out)
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
