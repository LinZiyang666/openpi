"""CPU cleanup audit for exact sealed Q run copies; original receipts untouched."""
import inspect
import json
from pathlib import Path

from exp.r17_integration.ops import formal_eval as launch,worker_storage as storage
from exp.r17_integration.ops.formal_finalize import closed_proof
from exp.r17_integration.ops.formal_process import ticks,alive,write_json
from exp.r17_integration.ops.sync_back import install_snapshot


def reclaim(name,audit):
    directory=Path('/data/mw_finetune/eval')/name
    original=directory/'RECEIPT.json';receipt_pin=storage.file_sha(original)
    sealed=json.loads(original.read_text())
    if sealed.get('scientific_evidence_seal'):
        reference=sealed['scientific_evidence_seal'];witness=Path(reference['path'])
        if witness.parent!=directory or storage.file_sha(witness)!=reference['sha256']:
            pass
        sealed=json.loads(witness.read_text())
    if sealed.get('status')!='PASS' or not sealed.get('sealed_at') or not sealed.get('evidence_verified'):
        pass
    campaign=json.loads((directory/'campaign.json').read_text())
    item=json.loads((directory/'WORKER_DEPLOYMENT.json').read_text())['timan107']
    closure=closed_proof(sealed)['timan107'];value=sealed['transfers']['timan107']['remote']
    archive=directory/'timan107.tar'
    size=value.get('archive_size_bytes',archive.stat().st_size)
    if archive.stat().st_size!=size or storage.file_sha(archive)!=value['archive_sha256']:
        raise ValueError('preserved local archive changed')
    manifest=install_snapshot(archive,directory/'workers/timan107',expected_sha256=value['archive_sha256'])
    if manifest['partials'] or manifest['digest']!=value['manifest']['digest']:
        pass
    proof=dict(node='timan107',run=item['worker_run'],run_name=campaign['run_name'],
        owners=[closure[k] for k in ('child','supervisor') if closure.get(k)],
        source_archive=value.get('source_archive',item['worker_run']+'.tar'),sealed_status='PASS',
        sealed_at=sealed['sealed_at'],evidence_verified=True,archive_sha256=value['archive_sha256'],
        archive_size_bytes=size,manifest_digest=value['manifest']['digest'],
        campaign_manifest_sha256=value['manifest']['files']['manifest.json'])
    code=('import os,json,stat,subprocess,hashlib,tarfile,fcntl; from pathlib import Path\n'+
        'REMOTE_RUNS='+repr(storage.REMOTE_RUNS)+'; LOCAL_RUNS='+repr(storage.LOCAL_RUNS)+'\n'+
        inspect.getsource(ticks)+inspect.getsource(alive)+inspect.getsource(storage.file_sha)+
        inspect.getsource(storage.remove_closed_run)+'\nprint(json.dumps(remove_closed_run('+repr(proof)+')))')
    result=json.loads(launch.remote('timan107',[item['python'],'-c',code],seconds=300,replay_safe=True))
    if storage.file_sha(original)!=receipt_pin:
        raise ValueError('original receipt changed concurrently')
    write_json(audit/(name+'.json'),dict(proof=proof,result=result,original_receipt_sha256=receipt_pin,
        original_receipt_untouched=True,local_archive_preserved=str(archive)))
    print(json.dumps(dict(name=name,result=result)),flush=True)


if __name__=='__main__':
    import sys
    audit=Path('/data/mw_finetune/qualification/disk_recovery_20261010');audit.mkdir(exist_ok=True)
    for name in sys.argv[1:]:reclaim(name,audit)
