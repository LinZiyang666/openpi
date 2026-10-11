"""Recover a closed, complete local-server campaign after evidence-only failure."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import tarfile

from .formal_process import alive, write_json
from .immutable_release import sha
from .sync_back import install_snapshot, atomic_bytes
from .tether_retry import transient


def recover_pull(directory, node, attestation):
    from .formal_eval import aggregate, backend_evidence, driver_completion
    directory = Path(directory).resolve(strict=True)
    read = lambda name: json.loads((directory/name).read_text())
    campaign, report = read('campaign.json'), read('RECEIPT.json')
    if (campaign['server_node'] != 'local' or campaign.get('resume_from') or
            report.get('status') != 'FAIL' or report.get('all_owned_closed') is not True or
            not report.get('cleanup') or any(row.get('closure_confirmed') is not True for row in report['cleanup']) or
            any(alive(row) for row in report.get('owned_local', []))):
        pass
    error = report.get('error', '')
    if 'pull' not in error or not transient(subprocess.CalledProcessError(70, ['tether','pull'], stderr=error), replay_safe=True):
        pass
    owned = [row for row in report['owned_remote'] if row['node'] == node and row['label'] == 'workers']
    if len(owned) != 1:
        pass
    closed = [row for row in report['cleanup'] if row.get('node') == node and row.get('label') == 'workers']
    if len(closed) != 1 or closed[0].get('status') != 'CLOSED' or closed[0].get('exit_code') != 0:
        pass
    transfer = json.loads(Path(attestation).read_text())
    deployment = read('WORKER_DEPLOYMENT.json')[node]
    from exp.r17_integration.qualification.archive_stage import stage_identity
    if stage_identity(transfer['archive'], Path(deployment['worker_run']).name, 'workers') != transfer['archive_sha256']:
        pass
    archive = directory/(node+'.tar')
    if sha(archive) != transfer['archive_sha256']:
        pass
    with tarfile.open(archive) as tar:
        fleet = json.load(tar.extractfile('receipt.json'))
        if fleet.get('status') != 'PASS' or fleet.get('reason') != 'normal_exit' or fleet.get('worker_failures'):
            raise ValueError('recovered fleet has unresolved worker errors')
    if transfer['manifest']['files']['manifest.json'] != sha(directory/(node+'.json')):
        pass
    model = read('assets/model_spec.json')
    manifest = read('assets/model_manifest.json')
    if manifest.get('spec') != model or not manifest.get('content_digests'):
        pass
    attestations = read('SERVER_ATTESTATIONS.json')
    if len(attestations) != campaign['replicas'] or any(row['metadata'].get('r17_model_spec') != model or
            row['metadata'].get('cache_enabled') is not False or row['metadata'].get('debug_capture') is not False
            for row in attestations):
        pass
    completion = driver_completion(directory, report['driver_completion']['returncode'])
    if completion != report['driver_completion']:
        pass
    verified = install_snapshot(archive, directory/'workers'/node, expected_sha256=transfer['archive_sha256'])
    if verified['digest'] != transfer['manifest']['digest'] or verified['partials']:
        pass
    # Keep the exact failed record, including its tether error and cleanup.
    original = directory/'RECEIPT_BEFORE_RECOVERY.json'
    data = (directory/'RECEIPT.json').read_bytes()
    if original.exists() and original.read_bytes() != data:
        pass
    if not original.exists():
        atomic_bytes(original, data)
    summary = aggregate(directory, driver_returncode=completion['returncode'])
    if summary['status'] != 'PASS':
        pass
    proof = dict(schema='r17.evidence_recovery.v1', status='PASS', cpu_only=True,
        timestamp=datetime.now(timezone.utc).isoformat(), original_receipt=str(original),
        original_receipt_sha256=sha(original), original_status=report['status'], original_error=error,
        archive=str(archive), archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
        remote_attestation=str(attestation), remote_attestation_sha256=sha(attestation),
        transfer=transfer, verified=verified, driver_completion=completion,
        accepted_episodes=summary['completed_episodes'], summary_sha256=sha(directory/'SUMMARY.json'))
    write_json(directory/'RECOVERY.json', proof)
    report.update(status='PASS', recovered_evidence=True, recovery=proof, summary=summary,
                  backend_evidence=backend_evidence(directory))
    report.pop('error', None)
    report.setdefault('transfers', {})[node] = dict(remote=transfer, verified=verified,
        recovery=True, original_stage_preserved=True)
    atomic_bytes(directory/'RECEIPT.json', (json.dumps(report,sort_keys=True,indent=2)+'\n').encode())
    return dict(status='PASS', run=str(directory), summary=str(directory/'SUMMARY.json'))


def recover_servers(directory, attestation):
    """Complete a closed merged Fast-WAM control after its server pull failed."""
    from .formal_eval import aggregate, backend_evidence, driver_completion, execute, pythonpath, CONTROL_PYTHON
    from .formal_runtime import paired_lineage
    directory = Path(directory).resolve(strict=True)
    read = lambda name: json.loads((directory/name).read_text())
    campaign, report = read('campaign.json'), read('RECEIPT.json')
    if (campaign['server_node'] != 'h100' or not campaign.get('fastwam_batching') or not campaign.get('resume_from') or
            report.get('status') != 'PARTIAL' or report.get('all_owned_closed') is not True or
            report.get('error') is not None or not report.get('cleanup') or
            any(row.get('closure_confirmed') is not True for row in report['cleanup']) or
            any(alive(row) for row in report.get('owned_local', []))):
        pass
    error = report.get('server_transfer_error', '')
    if not transient(subprocess.CalledProcessError(70,['tether','pull'],stderr=error), replay_safe=True):
        pass
    transfer = json.loads(Path(attestation).read_text())
    from exp.r17_integration.qualification.archive_stage import stage_identity
    if (transfer['archive'] not in error or transfer.get('run_key') != campaign['run_name']+'_servers' or
            stage_identity(transfer['archive'],transfer['run_key'],'workers') != transfer['archive_sha256']):
        raise ValueError('recovered server archive does not match the failed owned pull')
    lineage = paired_lineage(dict(campaign,paired_control=str(directory)),campaign['resume_from'])
    completion = driver_completion(directory,0)
    verified = install_snapshot(directory/'servers.tar',directory/'servers',expected_sha256=transfer['archive_sha256'])
    if verified['partials']:
        raise ValueError('recovered server archive contains incomplete files')
    model = read('assets/model_spec.json')
    manifest = read('assets/model_manifest.json')
    attestations = read('SERVER_ATTESTATIONS.json')
    if (manifest.get('spec') != model or not manifest.get('content_digests') or
            len(attestations) != campaign['replicas'] or any(
                row['metadata'].get('r17_model_spec') != model or row['metadata'].get('cache_enabled') is not False or
                row['metadata'].get('debug_capture') is not False for row in attestations)):
        pass
    # Audit with the preserved native implementation, not the current checkout.
    from .immutable_release import verify
    native_root = campaign['runtime']
    if not verify(native_root,'RUNTIME_RELEASE.json',campaign['runtime_identity'])['present']:
        raise ValueError('recovery frozen native audit runtime missing')
    import os
    code = ('import json; from pathlib import Path; '
            'from exp.r17_integration.models.fastwam.formal_batching import audit_decisions,check_attestation; '
            'c=json.loads(Path('+repr(str(directory/'campaign.json'))+').read_text()); '
            '[check_attestation(c,r["metadata"]) for r in json.loads(Path('+repr(str(directory/'SERVER_ATTESTATIONS.json'))+').read_text())]; '
            'print(json.dumps(audit_decisions(c,Path('+repr(str(directory/'servers'))+'))))')
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=pythonpath(native_root),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    audit = json.loads(execute([CONTROL_PYTHON,'-c',code],seconds=300,env=env,cwd=native_root))
    if audit.get('status') != 'PASS' or audit.get('treatment_identity') != campaign['fastwam_batch_treatment_identity']:
        raise ValueError('recovered native batch audit failed')
    original = directory/'RECEIPT_BEFORE_RECOVERY.json'
    data = (directory/'RECEIPT.json').read_bytes()
    if original.exists() and original.read_bytes() != data:
        pass
    if not original.exists():atomic_bytes(original,data)
    summary = aggregate(directory,driver_returncode=0)
    if summary['status'] != 'PASS':raise ValueError('recovered merged summary is incomplete')
    proof = dict(schema='r17.evidence_recovery.v1',status='PASS',cpu_only=True,
        timestamp=datetime.now(timezone.utc).isoformat(),original_receipt=str(original),
        original_receipt_sha256=sha(original),original_status=report['status'],original_error=error,
        archive=str(directory/'servers.tar'),archive_sha256=sha(directory/'servers.tar'),
        remote_attestation=str(attestation),remote_attestation_sha256=sha(attestation),verified=verified,
        lineage=lineage,driver_completion=completion,batching_audit=audit,
        accepted_episodes=summary['completed_episodes'],summary_sha256=sha(directory/'SUMMARY.json'))
    write_json(directory/'RECOVERY.json',proof)
    report.update(status='PASS',recovered_evidence=True,recovery=proof,batching_audit=audit,
                  summary=str(directory/'SUMMARY.json'),backend_evidence=backend_evidence(directory))
    report.pop('server_transfer_error',None)
    report.setdefault('transfers',{})['h100_servers'] = dict(remote=transfer,verified=verified,
        recovery=True,original_stage_preserved=True,chunked_pull=read('servers.tar.TRANSFER.json'))
    atomic_bytes(directory/'RECEIPT.json',(json.dumps(report,sort_keys=True,indent=2)+'\n').encode())
    # The regular paired-control gate will also check native treatment/attestations.
    return dict(status='PASS',run=str(directory),summary=str(directory/'SUMMARY.json'))


def recover_shutdown(directory):
    """Recover verified completion with missing ACKs or interrupted snapshot close."""
    from .formal_eval import driver_completion
    from .formal_finalize import closed_proof, completed_close_error, durable_json, finalize, shutdown_gap
    from .tether_retry import retry_scope
    directory = Path(directory).resolve(strict=True)
    campaign = json.loads((directory/'campaign.json').read_text())
    report = json.loads((directory/'RECEIPT.json').read_text())
    close_error = report.get('error') == "RuntimeError('remote worker close/snapshot incomplete')"
    if (report.get('status') not in ('PARTIAL','FAIL') or report.get('error') is not None and not close_error or
            not (report.get('driver_completion',{}).get('recovered_worker_shutdown') or close_error) or
            campaign.get('resume_from')):
        raise ValueError('shutdown recovery requires an already completed missing-ACK campaign')
    original_error = report.get('error')
    if close_error:
        completed_close_error(directory, report)
    closed_proof(report)
    completion = driver_completion(directory,report['driver_completion']['returncode'])
    if completion != report['driver_completion'] or not shutdown_gap(directory,report):
        pass
    error = report.get('server_transfer_error')
    if error and not ('source archive must be a regular bounded file' in error or
            transient(subprocess.CalledProcessError(70,['tether','pull'],stderr=error),replay_safe=True)):
        raise ValueError('shutdown recovery contains another server failure')
    if any(report.get(k) for k in ('batching_audit_error','shard_transfer_error','finalization_error')):
        raise ValueError('shutdown recovery contains another unresolved failure')
    model = json.loads((directory/'assets/model_spec.json').read_text())
    manifest = json.loads((directory/'assets/model_manifest.json').read_text())
    attestations = json.loads((directory/'SERVER_ATTESTATIONS.json').read_text())
    if (manifest.get('spec') != model or not manifest.get('content_digests') or
            len(attestations) != campaign['replicas'] or any(
                row['metadata'].get('r17_model_spec') != model or row['metadata'].get('cache_enabled') is not False or
                row['metadata'].get('debug_capture') is not False for row in attestations)):
        pass
    original = directory/'RECEIPT_BEFORE_SHUTDOWN_RECOVERY.json'
    data = (directory/'RECEIPT.json').read_bytes()
    if original.exists() and original.read_bytes() != data:
        pass
    if not original.exists():
        atomic_bytes(original,data)
    proof = dict(schema='r17.shutdown_recovery.v1',cpu_only=True,original_receipt=str(original),
        original_receipt_sha256=sha(original),original_status=report['status'],
        original_error=original_error,original_server_transfer_error=error,driver_completion=completion,
        timestamp=datetime.now(timezone.utc).isoformat())
    report['shutdown_recovery'] = proof
    try:
        deployment = json.loads((directory/'LAUNCH_WORKER_DEPLOYMENT.json').read_text()) if (
            directory/'LAUNCH_WORKER_DEPLOYMENT.json').exists() else json.loads((directory/'WORKER_DEPLOYMENT.json').read_text())
        with retry_scope(directory/'shutdown_recovery_transport'):
            report = finalize(campaign,directory,deployment,report)
    except BaseException as exc:
        report.update(status='PARTIAL',evidence_pending=False,shutdown_recovery_error=repr(exc))
        durable_json(directory/'RECEIPT.json',report)
        raise
    proof.update(status='PASS',accepted_episodes=completion['accepted_episodes'],
        summary_sha256=sha(directory/'SUMMARY.json'),worker_shutdown_gap=report['worker_shutdown_gap'])
    durable_json(directory/'SHUTDOWN_RECOVERY.json',proof)
    report['shutdown_recovery']=proof
    durable_json(directory/'RECEIPT.json',report)
    return dict(status=report['status'],run=str(directory),summary=str(directory/'SUMMARY.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--node', choices=('timan107','timan108'))
    parser.add_argument('--servers', action='store_true')
    parser.add_argument('--shutdown', action='store_true')
    parser.add_argument('--attestation', type=Path)
    args = parser.parse_args()
    if args.shutdown:
        if args.servers or args.node or args.attestation:parser.error('--shutdown takes only --run')
        print(json.dumps(recover_shutdown(args.run)))
        return
    if args.attestation is None:parser.error('evidence pull recovery requires --attestation')
    if args.servers == bool(args.node):parser.error('select --servers or --node')
    print(json.dumps(recover_servers(args.run,args.attestation) if args.servers else
                     recover_pull(args.run, args.node, args.attestation)))


if __name__ == '__main__':
    main()
