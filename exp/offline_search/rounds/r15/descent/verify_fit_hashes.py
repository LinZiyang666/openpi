"""Hash-only B fit audit: no journals, summaries, SR, selection or test-A reads."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from .analyze import HERE, MODELS, Reader, ServedFitVerifier, require, sha, verify_protocol, write_outputs


def decision_hashes(path):
    """Scan only a snapshot of this shard; running GR00T may append later."""
    digest=hashlib.sha256();hashes=set();count=0;tail=0
    with path.open('rb') as stream:
        before=os.fstat(stream.fileno());remaining=before.st_size
        while remaining:
            line=stream.readline(remaining)
            require(line, f'Decision shard truncated: {path}')
            remaining-=len(line);digest.update(line)
            if not line.endswith(b'\n'):
                tail=len(line);break
            if not line.strip():
                continue
            row=json.loads(line)
            if row.get('ev')=='dec':
                value=row.get('r14',{}).get('fit_sha256')
                require(isinstance(value,str), f'Missing served fit hash: {path}')
                hashes.add(value);count+=1
        after=os.fstat(stream.fileno());current=path.stat()
        require(before.st_ino==after.st_ino==current.st_ino and
                after.st_size>=before.st_size and current.st_size>=before.st_size,
                f'Decision shard replaced/truncated: {path}')
    return hashes,dict(path=str(path),snapshot_bytes=before.st_size,sha256=digest.hexdigest(),
                       decisions_with_fit_hash=count,incomplete_tail_bytes=tail)


def verify(models=MODELS):
    require(set(os.sched_getaffinity(0)) <= set(range(2,10))|set(range(46,54)), 'Use taskset -c 2-9,46-53')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'Hide GPUs before hash verification')
    reader=Reader();p=verify_protocol(reader);caps=reader.json(HERE/'cap_tables.json')
    records=reader.json(HERE/'c10_roots.json')+reader.json(HERE/'dev_roots.json')
    evidence=[];pending=[]
    for rec in records:
        if rec['cell'].split('_')[0] not in models:
            continue
        root=Path(rec['root'])
        require(root.parent==Path(p['trace_parent']) and root.name.startswith('r15_descent_') and
                'test' not in root.name and rec['init_pool']=='B', 'Disallowed non-B root')
        sealed=reader.json(root/'ROOT.json')
        require(sealed==rec, 'Root index differs from seal')
        for name,digest in sealed['definitions_sha256'].items():
            path=(root/name).resolve()
            require(path.is_relative_to(root.resolve()) and sha(reader.read(path))==digest, 'Changed root definition')
        rows=reader.json(root/'arms.json');contracts=reader.json(root/'dev_contracts.json')
        require([r['arm'] for r in rows]==rec['arms'], 'Wrong B arm set')
        for row in rows:
            require(row['dev'] is True and row['init_pool']=='B', 'Non-B arm')
            level=row['r14_descent_level'];source=caps['cells'][rec['cell']]['fit_sha256_by_level'][level]
            fit=Path(row['plugin_args'][row['plugin_args'].index('--os-fit-artifact')+1])
            require(source==sealed['fit_sha256'][str(fit)]==contracts[row['arm']]['fit_sha256'] and
                    sha(reader.read(fit))==source, 'Changed frozen source fit')
            run=root/'runs'/row['arm'];contract=run/'pool_contract.json'
            if contract.is_file():
                require(reader.json(contract)==contracts[row['arm']], 'Changed run pool contract')
            shards=sorted(run.glob('server_*/decisions_*.jsonl'))
            observed=set();snapshots=[]
            for path in shards:
                hashes,snapshot=decision_hashes(path)
                observed.update(hashes);snapshots.append(snapshot)
            verifier=ServedFitVerifier(root,row,source,reader)
            for value in sorted(observed):
                verifier.check(value)
            if not observed:
                pending.append(row['arm'])
            evidence.append(dict(arm=row['arm'],cell=rec['cell'],model=row['model'],
                frozen_source_matches=True,served_hashes_match=bool(observed),
                status='verified_observed_hashes' if observed else 'source_verified_no_decisions_yet',
                **verifier.audit(),decision_shard_snapshots=snapshots))
        print('FIT_HASH_PASS',root.name,len(rows),'frozen source fits',flush=True)
    reader.stable()
    return dict(PASS=True,verified_at_utc=datetime.now(timezone.utc).isoformat(),models=list(models),
        arms=len(evidence),all_frozen_sources_match=True,all_observed_served_hashes_match=True,
        arms_with_verified_served_hashes=len(evidence)-len(pending),pending_observed_hashes=pending,
        all_arms_have_observed_hashes=not pending,no_sr_or_outcome_tables=True,
        journals_read=False,summaries_read=False,test_a_read=False,apply_run=False,
        inputs=reader.inputs,records=evidence)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    args=parser.parse_args()
    result=verify(args.models)
    write_outputs({'served_fit_hash_verification.json':result},HERE)
    print(json.dumps({key:result[key] for key in ('PASS','arms','arms_with_verified_served_hashes',
        'all_frozen_sources_match','all_observed_served_hashes_match','pending_observed_hashes')}))


if __name__=='__main__':
    main()
