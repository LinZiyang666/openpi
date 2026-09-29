"""Read-only audit of verified arms and locally collected spill evidence."""
import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path

from .repair_stream import accepted_rows, rows
from .stream_receiver import digest


def audit(run, hash_files=False):
    run=Path(run); reports=[]; failures=[]; spills=[]; totals=Counter()
    verified_paths=sorted(run.glob('runs/*/p3_stream_verified.json'))
    for vp in verified_paths:
        root=vp.parent; stamp=json.loads(vp.read_text()); accepted=accepted_rows(root)
        report=dict(arm=root.name,accepted_attempts=len(accepted),files=0,bytes=0,errors=[])
        if digest(root/'client/journal.jsonl')!=stamp['journal_sha256']:
            report['errors'].append('journal changed since verification')
        for key,journal in accepted.items():
            rec=json.loads((root/'p3_stream'/key/'complete.json').read_text())
            if (rec['uid'],rec['attempt'],rec['status'],rec['success']) != (journal['task_uid'],journal['attempt'],'complete',journal['success']):
                report['errors'].append(key+': receipt/journal mismatch')
            expected=hashlib.sha256(json.dumps(rec['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
            if rec['sha256']!=expected or rec['bytes']!=sum(v['bytes'] for v in rec['files'].values()):
                report['errors'].append(key+': aggregate mismatch')
            for name,info in rec['files'].items():
                path=root/'client_telemetry'/key/name
                if path.is_symlink() or path.stat().st_size!=info['bytes'] or (hash_files and digest(path)!=info['sha256']):
                    report['errors'].append(key+'/'+name+': content mismatch')
                report['files']+=1;report['bytes']+=info['bytes']
            if rec.get('sender_stats',{}).get('spill'):
                totals['accepted_spill_receipts']+=1
        if any(report[k]!=stamp[k] for k in ('accepted_attempts','files','bytes')):
            report['errors'].append('verification totals mismatch')
        for k in ('accepted_attempts','files','bytes'):totals[k]+=report[k]
        if report['errors']:failures.append(root.name)
        reports.append(report)
    for root in sorted((run/'runs').iterdir()):
        if not root.is_dir():continue
        # Do not recurse into repair_*/ copies (they are evidence, not live arms).
        markers=sorted((root/'client_spills').glob('*/stream.p3spill.json'))
        if markers or (root/'client_spills.tar').exists():
            spills.append(dict(arm=root.name,verified=(root/'p3_stream_verified.json').exists(),
                markers=[dict(path=str(p),**json.loads(p.read_text())) for p in markers],
                tar_bytes=(root/'client_spills.tar').stat().st_size if (root/'client_spills.tar').exists() else None))
    return dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run=str(run),
        verified_arms=len(reports),hash_files=hash_files,totals=dict(totals),failures=failures,
        arms=reports,spills=spills,remote_scan=False,live_data_modified=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--hash-files',action='store_true')
    a=ap.parse_args();result=audit(a.run_root,a.hash_files)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('arms','spills')}))
    if result['failures']:raise SystemExit(1)


if __name__=='__main__':main()
