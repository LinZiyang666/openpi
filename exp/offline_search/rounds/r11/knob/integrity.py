"""Final read-only source, arm, diagnostic and coordinator-command checks."""
import json
from pathlib import Path
import re
import subprocess
from .build import HERE,REPO,ROOTS
from .verification import audit_trace
from exp.offline_search.rounds.r10.data import sha,write_json

def main():
    arms=state_anchors=0
    expected={r['root']:r for r in json.loads((HERE/'roots.json').read_text())}
    fits={r['artifact']:r['sha256'] for r in json.loads((HERE/'calibration_audit.json').read_text())['records']}
    seen_fits=set()
    for root in ROOTS:
        rows=json.loads((root/'arms.json').read_text())
        for row in rows:
            assert row['full_model'] and row['cost_ledger']
            args=row['plugin_args']
            assert args[args.index('--os-policy-tail-blocks')+1]=='1'
            assert args[args.index('--os-judge')+1]=='guard_only'
            path=args[args.index('--os-fit-artifact')+1]
            if path not in seen_fits:
                assert sha(path)==fits[path];seen_fits.add(path)
            file=next((HERE/'replays'/row['arm']).glob('decisions_*.jsonl'))
            decs=[json.loads(l) for l in file.read_text().splitlines() if '"ev": "dec"' in l]
            assert len(decs)==360
            audit_trace(decs,row)
            arms+=1
            if row['r11_method'] in ('distance','disagreement','error_hybrid','adaptive_error_hybrid'):
                state_anchors+=sum(d['vision'] for d in decs)
        assert len(rows)==expected[str(root)]['arm_count']
        assert set(r['arm'] for r in rows)==set(expected[str(root)]['arms'])
    assert arms==114
    for p in (HERE/'calibration').glob('*.json'):
        r=json.loads(p.read_text())
        base=HERE.parents[1]/'r10/recipe/artifacts'/f'r10_recipe_{r["cell"]}.pkl'
        assert sha(base)==r['base_artifact_sha256']
    for p in HERE.glob('*.py'):compile(p.read_text(),str(p),'exec')
    checked=0
    for file in ('H100_SOURCES.sha256','ALL_SOURCES.sha256','DOCUMENTS.sha256'):
        for line in (HERE/file).read_text().splitlines():
            digest,rel=line.split('  ',1);assert sha(REPO/rel)==digest,rel;checked+=1
    blocks=0
    for name in ('README.md','HANDBACK.md'):
        for code in re.findall(r'```bash\n(.*?)```',(HERE/name).read_text(),re.S):
            # Syntax only. No coordinator command is executed.
            result=subprocess.run(['bash','-n'],input=code,text=True,capture_output=True)
            assert result.returncode==0,result.stderr;blocks+=1
    report=dict(PASS=True,arms=arms,diagnostic_state_anchors=state_anchors,checksummed_files=checked,
                coordinator_bash_blocks_syntax_checked=blocks,no_coordinator_commands_executed=True)
    write_json(HERE/'integrity.json',report);print('INTEGRITY_PASS',json.dumps(report))

if __name__=='__main__':main()
