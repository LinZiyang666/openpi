"""CPU-only recovery of the first smoke's missing parent transfer manifest.

Preserves RECEIPT.json and emits RECOVERED_RECEIPT.json. Only a closed four-
episode Fast-WAM smoke with this exact transfer error is eligible; no processes
are started, stopped or renewed. Runtime archive pins bind the historic code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from exp.r17_integration.common.contracts import digest
from exp.r17_integration.ops import formal_eval as ops
from exp.r17_integration.ops.formal_process import write_json
from exp.r17_integration.ops.sync_back import install_snapshot
from exp.r17_integration.ops.tether_retry import retry_scope
from .formal_batching import audit_decisions


def archived_implementation(archive):
    prefix='exp/r17_integration/models/fastwam/'
    names=[prefix+name for name in ('batch_runtime.py','batcher.py','staged.py',
                                   'formal_batch_server.py','formal_batching.py')]
    names.append('src/openpi/serving/batching_core.py')
    with tarfile.open(archive) as tar:
        return digest({Path(name).name:hashlib.sha256(tar.extractfile(name).read()).hexdigest() for name in names})


def recover(directory):
    directory=Path(directory).resolve(strict=True)
    if directory.parent != ops.FORMAL_ROOT:
        raise ValueError('recovery requires a direct formal result directory')
    read=lambda name:json.loads((directory/name).read_text())
    campaign,original,summary=read('campaign.json'),read('RECEIPT.json'),read('SUMMARY.json')
    if (not campaign.get('smoke') or campaign.get('smoke_episodes')!=4 or
            campaign.get('model')!='fastwam' or campaign.get('server_node')!='h100' or
            not original.get('all_owned_closed') or original.get('status')!='PARTIAL' or
            'run manifest missing' not in original.get('server_transfer_error','') or
            summary.get('status')!='PASS' or summary.get('completed_episodes')!=4):
        raise ValueError('not a closed successful four-episode smoke with the known manifest-transfer failure')
    if archived_implementation(directory/'runtime.tar') != campaign['fastwam_batch_treatment']['batch_implementation_identity']:
        raise ValueError('historic batch implementation pin differs from archived runtime')
    expected=json.loads((directory/'assets/batching.json').read_text())
    if expected['fastwam_batch_treatment_identity']!=campaign['fastwam_batch_treatment_identity']:
        pass
    python=ops.host_python(campaign)
    env=['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+ops.pythonpath(campaign['runtime']),python]
    owners=[item['receipt']['child'] for item in original['owned_remote'] if item['node']=='h100']
    key=campaign['run_name']+'_servers_recovery'
    archive='/tmp/r17_formal/'+campaign['run_name']+'/server/recovered_closed_servers.tar'
    code=('from pathlib import Path; import json; '
        'from exp.r17_integration.ops.formal_process import alive,write_json; '
        'from exp.r17_integration.ops.sync_back import snapshot_run; '
        'from exp.r17_integration.qualification.archive_stage import stage_archive; '
        'assert all(not alive(p) for p in '+repr(owners)+'), "owned H100 processes still live"; '
        'p=Path('+repr(campaign['server_run'])+')/"manifest.json"; '
        'c=json.loads('+repr(json.dumps(campaign))+'); '
        'assert not p.exists(), "recovery is only for the missing manifest"; '
        'write_json(p,c); r=snapshot_run('+repr(campaign['server_run'])+','+repr(archive)+'); '
        'r.update(stage_archive('+repr(archive)+','+repr(key)+',"workers",r["archive_sha256"])); print(json.dumps(r))')
    with retry_scope(directory):
        value=json.loads(ops.remote('h100',[*env,'-c',code]))
        local_archive=directory/'recovered_servers.tar'
        ops.transfer(['tether','pull','h100:'+value['archive'],str(local_archive)])
        verified=install_snapshot(local_archive,directory/'servers',expected_sha256=value['archive_sha256'])
        code=('from exp.r17_integration.qualification.archive_stage import cleanup_stage; import json; '
              'print(json.dumps(cleanup_stage('+repr(value['archive'])+','+repr(key)+',"workers",'+repr(value['archive_sha256'])+')))')
        cleaned=json.loads(ops.remote('h100',[*env,'-c',code]))
    audit=audit_decisions(campaign,directory/'servers')
    evidence=ops.backend_evidence(directory)
    if len(evidence)<min(2,campaign['replicas']) or len(summary['accepted_worker_ids'])<2:
        pass
    for path in (directory/'servers').glob('replica_*/closed.json'):
        if json.loads(path.read_text()).get('status')!='CLOSED':
            raise ValueError('resident close receipt absent')
    result=dict(original,status='PASS',cpu_transfer_recovery=True,original_receipt=str(directory/'RECEIPT.json'),
        original_receipt_sha256=hashlib.sha256((directory/'RECEIPT.json').read_bytes()).hexdigest(),
        archived_implementation_verified=True,batching_audit=audit,backend_evidence=evidence,
        recovered_transfer=dict(remote=value,verified=verified,cleaned=cleaned),
        gpu_reexecuted=False,physical_episodes_reexecuted=0,
        recovered_error=original['server_transfer_error'])
    result.pop('server_transfer_error')
    write_json(directory/'RECOVERED_RECEIPT.json',result)
    print(json.dumps(dict(status='PASS',receipt=str(directory/'RECOVERED_RECEIPT.json'),batching_audit=audit)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',required=True)
    args=p.parse_args()
    recover(args.directory)


if __name__=='__main__':main()
