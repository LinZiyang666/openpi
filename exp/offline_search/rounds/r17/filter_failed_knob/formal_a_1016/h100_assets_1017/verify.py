"""Read-only proof of the actual H100 inputs, including both asset addresses."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import zlib

import yaml

from exp.offline_search.rounds.r17.filter_failed.stage import server_asset_destinations


def absolute_strings(value):
    if isinstance(value, dict):
        for item in value.values():yield from absolute_strings(item)
    elif isinstance(value, list):
        for item in value:yield from absolute_strings(item)
    elif isinstance(value, str) and value.startswith('/'):
        yield value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    c = json.loads((a.run/'campaign.json').read_text())
    assets = Path(c['local_assets'])
    files = {str(f.relative_to(assets)): hashlib.sha256(f.read_bytes()).hexdigest()
             for f in sorted(assets.rglob('*')) if f.is_file() and
             f.name not in ('.eval.lock', 'TRANSFER_SHA256.json')}
    paths = set(absolute_strings(json.loads((assets/'load.json').read_text())))
    for f in assets.rglob('*.yaml'):
        paths.update(absolute_strings(yaml.safe_load(f.read_text())))
    paths.update([c['runtime'], c['server_python'], c['server_campaign']])
    payload = dict(destinations=server_asset_destinations(c), files=files, paths=sorted(paths))
    packed = base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
    code = '''import base64,hashlib,json,zlib
from pathlib import Path
d=json.loads(zlib.decompress(base64.b64decode(PACKED)))
errors=[]; checked=0
for root in d['destinations']:
 for name,pin in d['files'].items():
  p=Path(root)/name
  if not p.is_file():errors.append('missing '+str(p));continue
  h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
  if h.hexdigest()!=pin:errors.append('different '+str(p))
  checked+=1
for p in d['paths']:
 if not Path(p).exists():errors.append('missing referenced input '+p)
print(json.dumps(dict(status='PASS' if not errors else 'FAIL',destinations=d['destinations'],
 files_per_tree=len(d['files']),hashes_checked=checked,referenced_paths=len(d['paths']),errors=errors)))
'''.replace('PACKED', repr(packed))
    result = subprocess.run(['tether', 'exec', '--safe', '--timeout', '120s', 'h100', '--',
                             '/usr/bin/python3', '-B', '-S', '-c', code],
                            capture_output=True, text=True, check=True, timeout=130)
    proof = dict(json.loads(result.stdout), run=str(a.run), cpu_only=True, remote_read_only=True)
    output = Path(__file__).parent/(a.run.name+'.json')
    output.write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof))
    if proof['status'] != 'PASS':raise SystemExit(1)


if __name__ == '__main__':main()
