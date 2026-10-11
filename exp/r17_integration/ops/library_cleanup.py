"""Exact-file cleanup of verified stages for a closed H100 library campaign.

The caller supplies local durable-receipt/archive SHA proofs and an explicit
file plan. This never clears capture shards, quota state, process receipts,
logs, another campaign or a live process. All files are validated before any
unlink; optional dry run is the default. No CUDA or process signals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from .build_client_bundle import sha256
from .formal_process import alive

TRANSFER_ROOT=Path('/tmp/r17_formal')
Q_ROOT=Path('/tmp/r17_Q_qualification')


def cleanup_verified_stages(request, *, apply=False):
    name=request['campaign']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,149}',name):
        raise ValueError('exact plain campaign required')
    server=TRANSFER_ROOT/name/'server'
    qroot=Q_ROOT/(name+'_shards')
    if server.resolve()!=server or qroot.resolve()!=qroot:
        raise ValueError('symlink stage parent')
    campaign=json.loads((server/'campaign.json').read_text())
    if campaign.get('library') is not True or campaign.get('server_node')!='h100' or campaign.get('run_name')!=name:
        raise ValueError('cleanup requires this exact H100 library campaign')
    processes=list(server.glob('*.process.json'))
    if not processes:
        raise ValueError('owned closed process receipts missing')
    for path in processes:
        if path.resolve()!=path:
            raise ValueError('symlink process receipt')
        receipt=json.loads(path.read_text())
        if receipt.get('status')!='CLOSED' or any(alive(receipt[key]) for key in ('child','supervisor') if key in receipt):
            raise ValueError('campaign has a live/unclosed owned process; no stage cleanup')
    proofs=set(request['verified_archive_sha256s'])
    if not proofs or any(not re.fullmatch(r'[0-9a-f]{64}',value) for value in proofs):
        raise ValueError('verified local archive SHA proofs required')
    validated=[]
    for entry in request['files']:
        path=Path(entry['path'])
        if path.resolve()!=path or not path.is_file():
            raise ValueError('nonregular/symlink stage file')
        if sha256(path)!=entry['sha256']:
            raise ValueError('stage file changed after cleanup inventory')
        if path.parent==server and path.name in ('runtime.tar','assets.tar'):
            if entry['sha256'] not in proofs:raise ValueError('deployment archive lacks local verified SHA')
        elif path.parent==server/'shards' and re.fullmatch(r'[0-9a-f]{64}\.tar',path.name):
            if entry['sha256'] not in proofs:raise ValueError('shard source lacks verified receipt')
        elif path.parent==qroot and re.fullmatch(r'workers_[0-9a-f]{64}_[0-9a-f]{32}\.(tar|stage\.json)',path.name):
            filename_sha=path.name.split('_')[1]
            if filename_sha not in proofs:raise ValueError('Q stage lacks local verified receipt')
            if path.suffix=='.tar':
                if entry['sha256']!=filename_sha:raise ValueError('Q archive SHA differs from name')
            else:
                marker=json.loads(path.read_text())
                archive=path.with_suffix('').with_suffix('.tar')
                if (marker.get('run_key')!=name+'_shards' or marker.get('kind')!='workers' or
                    marker.get('archive')!=str(archive) or marker.get('archive_sha256')!=filename_sha or
                    marker.get('stage_sha_verified') is not True):
                    raise ValueError('Q stage ownership proof differs')
                if marker.get('stage_removed') is True and archive.exists():
                    raise ValueError('removed Q stage unexpectedly exists')
                if archive.exists() and sha256(archive)!=filename_sha:
                    raise ValueError('unverified Q stage archive')
        else:
            raise ValueError('file is outside this exact campaign transfer plan')
        validated.append(dict(entry,bytes=path.stat().st_size))
    if len({entry['path'] for entry in validated})!=len(validated):
        raise ValueError('duplicate cleanup path')
    if apply:
        for entry in validated:Path(entry['path']).unlink()
    return dict(schema='r17.h100_stage_cleanup.v1',status='PASS',campaign=name,
        applied=apply,files=validated,removed_bytes=sum(e['bytes'] for e in validated) if apply else 0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    print(json.dumps(cleanup_verified_stages(json.loads(Path(args.request).read_text()),apply=args.apply)))


if __name__=='__main__':main()
