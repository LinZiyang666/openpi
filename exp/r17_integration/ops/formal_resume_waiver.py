"""Explicit unreachable-worker closure exception; read-only, receipt-bound proof.

Never signals, renews leases, imports a GPU runtime, or changes source evidence.
The exception concerns worker closure only, never a driver or policy service.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

from .formal_process import alive, write_json

SCHEMA = 'r17.resume_unreachable_source_node_waiver.v1'
NODES = ('timan107', 'timan108')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def ports(campaign):
    values = [*campaign.get('ports', []), *campaign.get('backend_ports', [])]
    values += [campaign.get(k) for k in ('public_port', 'driver_port', 'endpoint_port', 'driver_public_port')]
    if campaign.get('endpoint'):
        values.append(int(campaign['endpoint'].rsplit(':', 1)[1]))
    return sorted({int(p) for p in values if p is not None})


def inventory(source, node):
    """Reject missing cleanup records, live services, and any second unknown node."""
    source = Path(source)
    campaign, receipt = read(source/'campaign.json'), read(source/'RECEIPT.json')
    if node not in NODES or receipt.get('status') not in ('FAIL', 'PASS'):
        pass
    if receipt.get('all_owned_closed') is not False:
        pass
    owned = [dict(row, node='local') for row in receipt.get('owned_local', [])]
    owned += receipt.get('owned_remote', [])
    if not owned:
        pass
    cleanup = receipt.get('cleanup', [])
    closed, unknown, seen = [], [], set()
    for row in owned:
        key = (row['node'], row['label'])
        if key in seen:
            pass
        seen.add(key)
        matches = [r for r in cleanup if (r.get('node', 'local'), r.get('label')) == key]
        if len(matches) != 1:
            pass
        status = matches[0]
        if status.get('closure_confirmed') is True:
            closed.append(row)
        elif row['label'] == 'workers' and row['node'] == node:
            unknown.append(row)
        else:
            pass
    expected = {('local', 'driver'), (campaign['server_node'], 'proxy')}
    # Local policy ownership uses "local", rather than the Tether node name.
    expected.update((campaign['server_node'], 'replica_'+str(i)) for i in range(campaign['replicas']))
    if not expected <= {(r['node'], r['label']) for r in closed}:
        pass
    if {(r.get('node', 'local'), r.get('label')) for r in cleanup} != seen:
        pass
    if not unknown or node not in campaign.get('worker_nodes', {}):
        pass
    return campaign, receipt, closed, unknown


def unreachable(node):
    argv = ['tether', 'exec', '--safe', '--timeout', '10s', node, '--',
            'python3', '-c', "print('r17 resume reachability')"]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=20)
    stderr = result.stderr[-4000:]
    # A generic timeout/auth/CLI error is not sufficient evidence of an offline node.
    if result.returncode != 64 or not re.search(r'\bnode_offline\b', stderr, re.I):
        pass
    return dict(argv=argv, returncode=result.returncode, stderr=stderr, checked_at=now(), unreachable=True)


def local_closed(row):
    if alive(row):
        pass
    return dict(node='local', label=row['label'], processes=[dict(
        pid=row['pid'], start_ticks=row['start_ticks'], alive=False)], checked_at=now())


def remote_closed(node, rows):
    """One read-only Tether call per verified node; check both process identities."""
    payload = [dict(label=r['label'], path=r['request']['receipt'],
                    expected={k: r['receipt'][k] for k in ('child', 'supervisor')}) for r in rows]
    code = '''import hashlib,json,sys
from pathlib import Path
output=[]
for item in json.loads(sys.argv[1]):
    raw=Path(item['path']).read_bytes()
    receipt=json.loads(raw)
    processes=[]
    for role,expected in item['expected'].items():
        current=receipt[role]
        assert all(current[k]==expected[k] for k in ('pid','start_ticks')), 'source process receipt identity changed'
        try:
            fields=Path('/proc/'+str(expected['pid'])+'/stat').read_text().rsplit(') ',1)[1].split()
            active=fields[19]==str(expected['start_ticks']) and fields[0]!='Z'
        except (FileNotFoundError,ProcessLookupError):
            active=False
        processes.append(dict(role=role,pid=expected['pid'],start_ticks=expected['start_ticks'],alive=active))
    output.append(dict(label=item['label'],receipt_path=item['path'],receipt_sha256=hashlib.sha256(raw).hexdigest(),processes=processes))
print(json.dumps(output))
'''
    result = subprocess.run(['tether', 'exec', '--safe', '--timeout', '20s', node, '--',
                             'python3', '-c', code, json.dumps(payload)],
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        pass
    evidence = json.loads(result.stdout)
    if len(evidence) != len(rows) or any(p['alive'] is not False for r in evidence for p in r['processes']):
        pass
    return [dict(row, node=node, checked_at=now()) for row in evidence]


def validate_destination(campaign, proof):
    if campaign['run_name'] == proof['source_run_name']:
        pass
    if proof['node'] in campaign.get('worker_nodes', {}):
        pass
    actual = ports(campaign)
    if not actual or set(actual) & set(proof['source_ports']):
        raise ValueError('resume waiver requires disjoint destination direct and public/tunnel ports')
    if campaign['run_name'] != proof['destination']['run_name']:
        pass
    if {k: campaign.get(k) for k in ('public_port', 'backend_ports', 'driver_port', 'ports')} != proof['destination']['direct_port_binding']:
        pass
    # Direct ports were known when proof was issued; broker ports are added later.
    if not set(proof['destination']['direct_ports']) <= set(actual):
        pass


def validate(source, proof):
    source = Path(source).resolve(strict=True)
    if not isinstance(proof, dict) or proof.get('schema') != SCHEMA:
        pass
    campaign, receipt, closed, unknown = inventory(source, proof['node'])
    if (proof['source'] != str(source) or proof['source_run_name'] != campaign['run_name'] or
            proof['source_receipt_sha256'] != digest(source/'RECEIPT.json') or
            proof['source_campaign_sha256'] != digest(source/'campaign.json') or
            proof['source_ports'] != ports(campaign)):
        pass
    datetime.fromisoformat(proof['issued_at'])
    probe = proof['unreachable_probe']
    if (probe.get('unreachable') is not True or probe.get('returncode') != 64 or
            not re.search(r'\bnode_offline\b', probe.get('stderr', ''), re.I) or
            probe.get('argv') != ['tether', 'exec', '--safe', '--timeout', '10s', proof['node'], '--',
                                  'python3', '-c', "print('r17 resume reachability')"]):
        pass
    evidence = {(r['node'], r['label']): r for r in proof['verified_closed']}
    if len(evidence) != len(closed):
        pass
    for row in closed:
        actual = evidence.get((row['node'], row['label']), {})
        expected = ([dict(pid=row['pid'], start_ticks=row['start_ticks'], alive=False)] if row['node']=='local'
                    else [dict(role=k, pid=row['receipt'][k]['pid'],
                               start_ticks=row['receipt'][k]['start_ticks'], alive=False)
                          for k in ('child', 'supervisor')])
        if actual.get('processes') != expected or not actual.get('checked_at'):
            pass
    if proof['unverified_processes'] != [dict(node=r['node'], label=r['label'],
            receipt_path=r['request']['receipt']) for r in unknown]:
        pass


def issue(source, campaign, node):
    source = Path(source).resolve(strict=True)
    old, _, closed, unknown = inventory(source, node)
    proof = dict(schema=SCHEMA, node=node, issued_at=now(), source=str(source),
                 source_run_name=old['run_name'], source_receipt_sha256=digest(source/'RECEIPT.json'),
                 source_campaign_sha256=digest(source/'campaign.json'), source_ports=ports(old),
                 destination=dict(run_name=campaign['run_name'], direct_ports=ports(campaign),
                    direct_port_binding={k: campaign.get(k) for k in ('public_port', 'backend_ports', 'driver_port', 'ports')}))
    validate_destination(campaign, proof)
    proof['unreachable_probe'] = unreachable(node)
    evidence = [local_closed(r) for r in closed if r['node']=='local']
    for other in sorted({r['node'] for r in closed if r['node']!='local'}):
        evidence.extend(remote_closed(other, [r for r in closed if r['node']==other]))
    proof.update(verified_closed=evidence, unverified_processes=[dict(
        node=r['node'], label=r['label'], receipt_path=r['request']['receipt']) for r in unknown])
    validate(source, proof)
    return proof


def refresh(campaign, directory):
    """Recheck at launch, including a launch of a previously prepared resume."""
    prior = campaign['resume_source_closure_waiver']
    validate(prior['source'], prior)
    validate_destination(campaign, prior)
    proof = issue(prior['source'], campaign, prior['node'])
    proof['previous_check_at'] = prior['issued_at']
    campaign['resume_source_closure_waiver'] = proof
    record_selection(campaign, directory)


def record_selection(campaign, directory):
    """Keep prepared resumes' merge proof in sync with launch-time evidence."""
    selection = Path(directory)/'RESUME.json'
    if selection.exists():
        data = read(selection)
        data['source_closure_waiver'] = campaign['resume_source_closure_waiver']
        write_json(selection, data)
