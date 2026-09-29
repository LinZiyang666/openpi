"""Hash-certified recovery of a spill colliding with an older driver incarnation.

prepare writes ONLY inside a fresh repair directory. install is an explicit
coordinator action on a quiescent arm; it preserves originals, takes the same
flock as the running receiver, and publishes files atomically. No network IO.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .stream_protocol import attempt_key, filename, receive
from .stream_receiver import Store, atomic_json, digest, safe_path, sync_dir, trace_identity


def rows(path):
    with Path(path).open() as f:
        for line in f:
            yield json.loads(line)


def accepted_rows(root):
    out = {}
    for row in rows(root / 'client/journal.jsonl'):
        if row.get('accepted') and row.get('status') in ('done', 'failed') and not row.get('error'):
            key = attempt_key(row['task_uid'], int(row.get('attempt', 1)))
            if key in out and out[key] != row:
                raise ValueError('ambiguous accepted journal row')
            out[key] = row
    return out


def inventory(directory):
    out = {}
    if directory.exists():
        for p in sorted(directory.iterdir()):
            if p.name == '.lock':
                continue
            if p.is_symlink() or not p.is_file():
                raise ValueError('unexpected path in attempt directory: ' + str(p))
            out[p.name] = dict(bytes=p.stat().st_size, sha256=digest(p))
    return out


def spill_events(path, run, arm):
    info = json.loads(path.with_name('stream.p3spill.json').read_text())
    identity = {k: info[k] for k in ('run', 'arm', 'uid', 'attempt', 'key')}
    if (info['run'], info['arm']) != (Path(run).name, arm) or info['key'] != attempt_key(info['uid'], info['attempt']):
        raise ValueError('spill identity mismatch')
    if path.stat().st_size != info['bytes'] or digest(path) != info['sha256']:
        raise ValueError('spill checksum mismatch')
    with path.open('rb') as f:
        while True:
            at = f.tell()
            event = receive(f.read)
            if event is None:
                break
            h, body = event
            if any(h.get(k) != v for k, v in identity.items()):
                raise ValueError('spill frame identity mismatch')
            yield at, h, body


def accepted_proof(root, candidate, journal):
    """Bind the reconstructed incarnation to the DRIVER run, not just uid/aN.

    The terminal timing dictionary is independently journaled by the driver
    under run_id. Every decision then joins via its driver's unique winner to
    the server connection. No outcome-only or timestamp-nearness heuristic.
    """
    uid, attempt = journal['task_uid'], int(journal.get('attempt', 1))
    trace = trace_identity(candidate / 'controls.jsonl', uid, attempt)
    if trace['status'] != 'complete' or trace['success'] != journal['success']:
        raise ValueError('reconstructed attempt is not the accepted outcome')
    run_id = journal.get('run_id')
    if not run_id:
        raise ValueError('collision recovery requires journal driver run_id')
    step_rows = [r for r in rows(root/'client/per_step.jsonl') if
                 (r.get('task_uid'), r.get('attempt'), r.get('run_id'), r.get('accepted')) == (uid, attempt, run_id, True)]
    timing = [r for r in step_rows if r.get('_kind') == 'client_timing']
    ends, decisions = [], {}
    for row in rows(candidate/'controls.jsonl'):
        if row['ev'] == 'rollout_end':
            ends.append(row)
        if row['ev'] == 'decision':
            i = row['decision_step']
            if i in decisions:
                raise ValueError('duplicate client decision')
            decisions[i] = row
    if len(ends) != 1 or len(timing) != 1 or not ends[0].get('timing'):
        raise ValueError('missing unique driver/client timing proof')
    if any(timing[0].get(k) != v for k, v in ends[0]['timing'].items()):
        raise ValueError('spill belongs to a different driver incarnation (timing)')
    steps = [r for r in step_rows if r.get('_kind') != 'client_timing']
    winners = {r['winner_id'] for r in steps}
    if len(winners) != len(steps) or len(steps) != len(decisions):
        raise ValueError('driver/client decision count mismatch')
    server, markers = {}, {}
    for path in sorted(root.glob('server_*/decisions_*.jsonl')):
        for r in rows(path):
            if r.get('ev') == 'p3_decision' and (r.get('uid'),r.get('attempt')) == (uid,attempt):
                mk=(r['tag'],r['conn'],r['step'])
                if mk in markers and markers[mk] != r:
                    raise ValueError('ambiguous profile decision')
                markers[mk]=r
            if r.get('ev') == 'dec' and r.get('winner') in winners:
                i = r['step']
                if i in server and server[i] != r:
                    raise ValueError('ambiguous server decision')
                server[i] = r
    if set(server) != set(decisions):
        raise ValueError('accepted driver winners missing from server log')
    connections = set()
    for i, c in decisions.items():
        s = server[i]
        if (s['uid'], s['attempt']) != (uid, attempt) or c['marker']['p3_step'] != i:
            raise ValueError('server/client decision identity mismatch')
        marker=markers[(s['tag'],s['conn'],i)]
        if bool(c['marker']['p3_anchor']) != bool(s['vision']) or any(
                c['marker'][k] != marker[k] for k in ('source','parent_anchor','commit_controls')):
            raise ValueError('server/client decision marker mismatch')
        connections.add((s['tag'], s['conn']))
    if len(connections) != 1:
        raise ValueError('accepted attempt spans multiple server connections')
    return dict(driver_run_id=run_id, timing_fields=len(ends[0]['timing']),
                decisions=len(decisions), server_connections=sorted(connections), **trace)


def reconstruct(run, arm, spill, output):
    root = Path(run)/'runs'/arm
    info = json.loads(spill.with_name('stream.p3spill.json').read_text())
    events = list(spill_events(spill, run, arm))
    finals = [h for _, h, _ in events if h['op'] == 'attempt']
    if len(finals) != 1 or events[-1][1]['op'] != 'attempt':
        raise ValueError('spill needs exactly one terminal file manifest')
    final = finals[0]
    if 'controls.jsonl' not in final['files']:
        raise ValueError('missing controls manifest')
    key = info['key']; live = root/'client_telemetry'/key
    output.mkdir(parents=True, exist_ok=False)
    for name in final['files']:
        filename(name)
        source = safe_path(live, name)
        if not source.exists():
            source = safe_path(live, '.'+name+'.p3part')
        if source.exists():
            shutil.copyfile(source, output/name)
        else:
            (output/name).touch()
    conflicts, counts, first_offsets = [], Counter(), {}
    finishes = {}
    for position, h, body in events:
        op = h['op']; counts[op] += 1
        if op == 'attempt':
            continue
        if op not in ('data', 'finish') or h['file'] not in final['files']:
            raise ValueError('unexpected spill operation/file')
        name = filename(h['file']); path = output/name
        if op == 'finish':
            expected = {k: h[k] for k in ('bytes', 'sha256')}
            if expected != final['files'][name]:
                raise ValueError('finish/attempt manifest disagreement')
            finishes[name] = expected
            continue
        off = h['offset']
        if type(off) is not int or not 0 <= off <= path.stat().st_size:
            raise ValueError('missing acknowledged prefix; exact recovery impossible')
        if off + len(body) > final['files'][name]['bytes']:
            raise ValueError('spill extends beyond certified size')
        first_offsets.setdefault(name, off)
        with path.open('r+b') as f:
            f.seek(off); old = f.read(len(body))
            different = next((i for i in range(min(len(old), len(body))) if old[i] != body[i]), None)
            if different is not None:
                conflicts.append(dict(file=name, frame_position=position, frame_offset=off,
                    frame_bytes=len(body), first_difference=off+different,
                    old=old[max(0,different-60):different+160].decode(errors='replace'),
                    new=body[max(0,different-60):different+160].decode(errors='replace')))
            f.seek(off); f.write(body)
    for name, expected in final['files'].items():
        path = output/name
        if path.stat().st_size < expected['bytes']:
            raise ValueError('missing certified suffix')
        # A shorter new incarnation may supersede a longer failed one. The
        # final full-file hash, not this truncation, certifies every byte.
        with path.open('r+b') as f:
            f.truncate(expected['bytes']); f.flush(); os.fsync(f.fileno())
        if digest(path) != expected['sha256']:
            raise ValueError('reconstruction SHA mismatch: ' + name)
    return dict(identity={k: info[k] for k in ('run','arm','uid','attempt','key')},
                final=final, spill_sha256=info['sha256'], spill_bytes=info['bytes'],
                frames=dict(counts), conflicts=conflicts, first_offsets=first_offsets)


def prepare(run, arm, directory, spill_directory=None):
    """Produce a complete verified arm view; never mutate its source tree."""
    from .stream_collect import verify_arm
    run = Path(run).resolve(); root = run/'runs'/arm
    Store(run).locations(dict(run=run.name, arm=arm, uid='validate', attempt=1, key=attempt_key('validate',1)))
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    staged = directory/run.name; target = staged/'runs'/arm
    target.mkdir(parents=True)
    shutil.copyfile(run/'arms.json', staged/'arms.json')
    shutil.copytree(root/'client', target/'client')
    shutil.copytree(root/'client_telemetry', target/'client_telemetry')
    shutil.copytree(root/'p3_stream', target/'p3_stream')
    accepted = accepted_rows(root)
    sources = dict(journal=digest(root/'client/journal.jsonl'), per_step=digest(root/'client/per_step.jsonl'))
    report = dict(schema='r6p3.stream_repair.v1', source_run=str(run), arm=arm,
                  staged_run=str(staged), sources=sources, attempts=[])
    for spill in sorted(Path(spill_directory or root/'client_spills').glob('*/stream.p3spill')):
        key = spill.parent.name
        if key not in accepted:
            raise ValueError('collision recovery does not infer authority for unaccepted spills')
        result = reconstruct(run, arm, spill, directory/'candidates'/key)
        if result['identity']['key'] != key:
            raise ValueError('spill directory/key mismatch')
        result['proof'] = accepted_proof(root, directory/'candidates'/key, accepted[key])
        result['original_data'] = inventory(root/'client_telemetry'/key)
        result['original_meta'] = inventory(root/'p3_stream'/key)
        result['candidate'] = str(directory/'candidates'/key)
        # Replace ONLY the isolated copied attempt. Keep original evidence in
        # this repair directory, including error receipts and unaccepted data.
        backup = directory/'originals'/key; backup.mkdir(parents=True)
        for name in ('client_telemetry','p3_stream'):
            src = target/name/key
            if src.exists():
                os.replace(src, backup/name)
        dest = target/'client_telemetry'/key
        shutil.copytree(directory/'candidates'/key, dest)
        store = Store(staged)
        for name, expected in result['final']['files'].items():
            store.apply(dict(**result['identity'], op='finish', file=name, **expected))
        store.apply(result['final'])
        report['attempts'].append(result)
    if not report['attempts']:
        raise ValueError('no finished spill to reconstruct')
    report['verified'] = verify_arm(staged, arm)
    # Rechecking source hashes closes accidental changes during the copy.
    if sources != dict(journal=digest(root/'client/journal.jsonl'), per_step=digest(root/'client/per_step.jsonl')):
        raise ValueError('live journal/per-step changed during repair')
    report['ready'] = True
    atomic_json(directory/'repair.json', report)
    return report


def install(directory):
    """Explicit coordinator step; no network, cleanup or DONE marker."""
    from .stream_collect import verify_arm
    directory = Path(directory).resolve()
    report = json.loads((directory/'repair.json').read_text())
    run, arm = Path(report['source_run']), report['arm']; root = run/'runs'/arm
    if not report.get('ready'):
        raise ValueError('repair has not passed prepare verification')
    if report['sources'] != dict(journal=digest(root/'client/journal.jsonl'), per_step=digest(root/'client/per_step.jsonl')):
        raise ValueError('source journal changed; re-prepare')
    verify_arm(Path(report['staged_run']), arm)
    for item in report['attempts']:
        key, dest, meta = Store(run).locations(item['identity'])
        meta.mkdir(parents=True, exist_ok=True)
        with (meta/'.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            current = inventory(dest)
            expected = item['final']['files']
            # Idempotent success or crash-resume of our own installation only.
            started = directory/('install_'+key+'.json')
            if not started.exists():
                if current != item['original_data'] or inventory(meta) != item['original_meta']:
                    raise ValueError('receiver attempt changed; refusing stale repair')
                atomic_json(started, dict(original_data=current, original_meta=inventory(meta)))
            else:
                for name, value in current.items():
                    if value != expected.get(name) and value != item['original_data'].get(name):
                        raise ValueError('unrecognized bytes since interrupted install')
            candidate = Path(item['candidate'])
            for name, value in expected.items():
                if digest(candidate/name) != value['sha256'] or (candidate/name).stat().st_size != value['bytes']:
                    raise ValueError('candidate modified')
                fd, tmp = tempfile.mkstemp(prefix='.'+name+'.repair_', dir=dest)
                with os.fdopen(fd,'wb') as out, (candidate/name).open('rb') as src:
                    shutil.copyfileobj(src,out); out.flush(); os.fsync(out.fileno())
                os.replace(tmp,dest/name)
            # Do not leave a previous incarnation's extra snapshots/partials
            # in the canonical tree. Originals are already preserved above.
            quarantine = directory/'superseded'/key; quarantine.mkdir(parents=True,exist_ok=True)
            for p in list(dest.iterdir()):
                if p.name not in expected:
                    os.replace(p,quarantine/p.name)
            source_meta = Path(report['staged_run'])/'runs'/arm/'p3_stream'/key
            for p in list(meta.iterdir()):
                if p.name != '.lock' and p.name not in {n+'.json' for n in expected} | {'complete.json'}:
                    os.replace(p,quarantine/('meta_'+p.name))
            # Complete receipt publishes last. Receiver shares this flock and
            # has no cached file contents/receipts requiring a restart.
            for name in sorted(expected):
                atomic_json(meta/(name+'.json'),json.loads((source_meta/(name+'.json')).read_text()))
            atomic_json(meta/'complete.json',json.loads((source_meta/'complete.json').read_text()))
            sync_dir(dest)
    verified = verify_arm(run,arm)
    atomic_json(directory/'installed.json',verified)
    return verified


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-root',type=Path); ap.add_argument('--arm')
    ap.add_argument('--prepare',type=Path); ap.add_argument('--install',type=Path)
    a=ap.parse_args()
    if a.prepare and a.run_root and a.arm and not a.install:
        result=prepare(a.run_root,a.arm,a.prepare)
        print(json.dumps(dict(ready=result['ready'],verified=result['verified'],repair=str(a.prepare/'repair.json'))))
    elif a.install and not a.prepare:
        print(json.dumps(install(a.install)))
    else:
        ap.error('choose --run-root RUN --arm ARM --prepare FRESH_DIR, or --install REPAIR_DIR')


if __name__=='__main__':
    main()
