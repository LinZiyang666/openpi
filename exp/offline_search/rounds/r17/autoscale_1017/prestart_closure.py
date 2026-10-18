"""Read-only process proof and owner-invoked closure of a pre-spawn MPS failure.

This narrow repair never stops processes or prepares/launches a campaign. It
keeps the original failed receipt and attaches fresh node evidence to the FAIL.
"""
import argparse
import fcntl
import hashlib
import inspect
import json
from pathlib import Path
import time

from .node import atomic


def probe(request, *, proc_root='/proc'):
    """Standalone read-only node probe; no GPU commands or environment reads."""
    import hashlib
    import json
    import os
    from pathlib import Path
    import time
    roots = [Path(p) for p in request['roots']]
    proc = Path(proc_root)
    identities, files, paths, live = [], [], [], []
    unreadable = dict(identity=0, cmdline=0, cwd=0)

    def extract(value, source):
        if isinstance(value, dict):
            if value.get('pid') and value.get('start_ticks'):
                identities.append(dict(source=source, **{k:value[k] for k in
                    ('pid', 'start_ticks', 'pgid') if k in value}))
            for v in value.values():extract(v, source)
        elif isinstance(value, list):
            for v in value:extract(v, source)

    for root in roots:
        if root.is_symlink():raise ValueError('symlinked campaign root')
        paths.append(dict(path=str(root), exists=root.exists()))
        if not root.exists():continue
        selected = set()
        for pattern in ('*.process.json', '*receipt*.json', '*RECEIPT*.json', '*owner*.json', '*OWNER*.json',
                        '*manifest*.json', '*MANIFEST*.json', 'MPS_READY.json'):
            selected.update(root.rglob(pattern))
        for path in sorted(selected):
            if path.is_symlink() or path.stat().st_size > 2**20:
                raise ValueError('inspect oversized/symlinked ownership evidence manually')
            body = path.read_bytes()
            value = json.loads(body)
            files.append(dict(path=str(path), sha256=hashlib.sha256(body).hexdigest(),
                              mtime=path.stat().st_mtime, status=value.get('status')))
            extract(value, str(path))
        # These small pathname facts distinguish wrapper deployment from MPS start.
        paths[-1]['markers'] = {n:(root/n).exists() for n in
            ('mps_control', 'mps', 'MPS_READY.json', 'driver/journal.jsonl', 'servers')}

    def stat(pid):
        try:
            f = (proc/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
            return dict(state=f[0], pgid=int(f[2]), start_ticks=f[19])
        except (FileNotFoundError, ProcessLookupError):return None

    groups = {int(e['pgid']) for e in identities if e.get('pgid')}
    for e in identities:
        s = stat(e['pid'])  # unreadable known identity must fail closed
        e['alive'] = bool(s and s['state'] != 'Z' and s['start_ticks'] == str(e['start_ticks']))
        if e['alive']:live.append(dict(pid=e['pid'], reason='receipt_identity'))
    for pid in request.get('launcher_pids', []):
        s = stat(pid)
        if s and s['state'] != 'Z':live.append(dict(pid=pid, reason='launcher_pid'))
    for directory in proc.glob('[0-9]*'):
        pid = int(directory.name)
        if pid == os.getpid():continue
        try:
            s = stat(pid)
            if not s or s['state'] == 'Z':continue
            if s['pgid'] in groups:
                live.append(dict(pid=pid, reason='receipt_process_group'))
            if directory.stat().st_uid != os.getuid():continue
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            unreadable['identity'] += 1;continue
        try:
            args = (directory/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            unreadable['cmdline'] += 1;args = []
        for i, arg in enumerate(args):
            if i and args[i-1] in (b'-c', b'-lc'):continue
            for root in roots:
                prefix = str(root).encode()
                if arg == prefix or arg.startswith(prefix+b'/') or b'"'+prefix+b'/' in arg:
                    live.append(dict(pid=pid, reason='campaign_argv'));break
        try:
            cwd = (directory/'cwd').resolve(strict=True)
            if any(cwd == root or root in cwd.parents for root in roots):
                live.append(dict(pid=pid, reason='campaign_cwd'))
        except (FileNotFoundError, ProcessLookupError, PermissionError):unreadable['cwd'] += 1
    return dict(status='PASS' if not live else 'FAIL', time=time.time(), request=request,
                paths=paths, files=files, identities=identities, live=live,
                unreadable_process_fields=unreadable, processes_signalled=0)


def validate(root, receipt, campaign):
    expected = '/tmp/r17_formal/'+root.name+'/server/mps_guard.process.json'
    if (receipt.get('status') not in ('FAIL', 'PARTIAL') or
        receipt.get('owned_local') != [] or receipt.get('owned_remote') != [] or
        receipt.get('backend_evidence') != {} or campaign.get('server_node') != 'h100' or
        campaign.get('library') or not campaign.get('mps') or
        'FileExistsError: '+expected not in receipt.get('error', '')):
        raise ValueError('requires the exact pre-spawn MPS receipt collision with no owned processes')
    if (root/'driver/journal.jsonl').exists() or (root/'DONE.json').exists():
        raise ValueError('not a pre-driver failure')
    if any(e.get('label') != 'mps_daemon' for e in receipt.get('cleanup', [])):
        raise ValueError('unexpected cleanup ownership')


def finalize(root, *, moved_stage, evidence_dir, remote=None):
    """Lock the failed local campaign, probe nodes, then update closure only."""
    from exp.r17_integration.ops.formal_eval import remote as default_remote
    root = Path(root).resolve();evidence_dir = Path(evidence_dir)
    moved_stage = Path(moved_stage)
    stage = Path('/tmp/r17_formal')/root.name
    if moved_stage.parent != stage.parent or not moved_stage.name.startswith(root.name+'.'):
        raise ValueError('moved stage must be an explicit sibling of the original stage')
    if root == evidence_dir or root in evidence_dir.resolve().parents:
        raise ValueError('audit output must be outside the campaign')
    remote = remote or default_remote
    with (root/'.prepared.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = (root/'RECEIPT.json').read_bytes()
        receipt = json.loads(source);campaign = json.loads((root/'campaign.json').read_text())
        validate(root, receipt, campaign)
        launch = json.loads((root/'LAUNCH.json').read_text())
        deployment = json.loads((root/'LAUNCH_WORKER_DEPLOYMENT.json').read_text())
        requests = {'h100':dict(roots=[str(root), str(stage), str(moved_stage)]),
                    'local':dict(roots=[str(root), str(stage)], launcher_pids=[launch['pid']])}
        for node, item in deployment.items():
            if node not in ('local', 'timan1', 'timan107'):
                raise ValueError('unexpected worker node; never probe timan108')
            requests.setdefault(node, dict(roots=[]))['roots'].append(item['stage'])
            if item.get('worker_run'):
                requests[node]['roots'].append(item['worker_run'])
        proof = dict(schema='r17.prestart_closure.v1', run=str(root), accepted=0,
                     original_receipt_sha256=hashlib.sha256(source).hexdigest(), nodes={})
        for node, request in requests.items():
            code = inspect.getsource(probe)+'\nimport json\nprint(json.dumps(probe('+repr(request)+')))'
            proof['nodes'][node] = (probe(request) if node == 'local' else
                json.loads(remote(node, ['/usr/bin/python3', '-c', code], seconds=60, replay_safe=True)))
        proof['all_owned_closed'] = all(p['status']=='PASS' for p in proof['nodes'].values())
        proof['time'] = time.time()
        evidence_dir.mkdir(parents=True, exist_ok=True)
        atomic(evidence_dir/'CLOSURE_PROOF.json', proof)
        if not proof['all_owned_closed']:raise ValueError('live ownership remains; receipt unchanged')
        if (root/'RECEIPT.json').read_bytes() != source:raise ValueError('receipt changed during proof')
        backup = root/('RECEIPT.before_prestart_closure.'+proof['original_receipt_sha256'][:12]+'.json')
        if backup.exists() and backup.read_bytes()!=source:raise ValueError('receipt backup collision')
        if not backup.exists():
            with backup.open('xb') as f:f.write(source)
        atomic(root/'PRESTART_CLOSURE.json', proof)
        receipt.update(status='FAIL', all_owned_closed=True,
                       closure_finalization=dict(evidence='PRESTART_CLOSURE.json',
                           original_receipt=backup.name, time=proof['time'],
                           reason='MPS spawn rejected before process creation; all node ownership absent'))
        for e in receipt['cleanup']:
            e.update(original_closure_confirmed=e.get('closure_confirmed'), closure_confirmed=True,
                     closure_evidence='PRESTART_CLOSURE.json')
        atomic(root/'RECEIPT.json', receipt)
        return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--moved-stage', type=Path, required=True)
    p.add_argument('--evidence-dir', type=Path, required=True)
    a = p.parse_args()
    proof = finalize(a.run, moved_stage=a.moved_stage, evidence_dir=a.evidence_dir)
    print(json.dumps(dict(all_owned_closed=proof['all_owned_closed'], accepted=proof['accepted'],
                         nodes={n:v['status'] for n,v in proof['nodes'].items()})))


if __name__ == '__main__':main()
