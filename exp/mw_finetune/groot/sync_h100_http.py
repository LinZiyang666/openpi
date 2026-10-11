"""Direct-NAT fallback when the tether artifact broker stalls.

Serves only B's code and public dataset archive pieces on the assigned GR00T
port during transfer. Checks SHA256 on H100; preserves staging files on both
hosts. The server closes as soon as the remote transfer command returns.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import shutil
import subprocess
import tempfile
import threading


REMOTE = r'''
import hashlib, json, pathlib, shutil, subprocess, sys, urllib.request
base, root, repo = sys.argv[1:]
root = pathlib.Path(root); root.mkdir(parents=True, exist_ok=True)
manifest = json.load(urllib.request.urlopen(base + 'manifest.json', timeout=60))
for entry in manifest['files']:
    dest = root / entry['name']
    if dest.exists() and hashlib.file_digest(dest.open('rb'), 'sha256').hexdigest() == entry['sha256']:
        print('reuse', dest, flush=True); continue
    with urllib.request.urlopen(base + entry['name'], timeout=120) as src, dest.open('wb') as dst:
        shutil.copyfileobj(src, dst, 8*1024*1024)
    with dest.open('rb') as src:
        actual = hashlib.file_digest(src, 'sha256').hexdigest()
    if actual != entry['sha256']: raise RuntimeError('SHA256 mismatch: ' + str(dest))
    print('verified', dest, dest.stat().st_size, flush=True)
pathlib.Path(repo).mkdir(parents=True, exist_ok=True)
subprocess.run(['tar', '-xzf', str(root/'code.tar.gz'), '-C', repo], check=True)
data = [entry for entry in manifest['files'] if entry['name'].startswith('data.part-')]
if data:
    output = pathlib.Path('/data/mw_finetune/datasets'); output.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(['tar', '-xf', '-', '-C', str(output)], stdin=subprocess.PIPE)
    try:
        for entry in data:
            with (root/entry['name']).open('rb') as src:
                shutil.copyfileobj(src, proc.stdin, 8*1024*1024)
    finally:
        proc.stdin.close()
    if proc.wait(): raise RuntimeError('dataset extraction failed')
print('TRANSFER_COMPLETE; retained remote staging:', root, flush=True)
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=Path('/data/mw_finetune/datasets/metaworld_mt50_groot'))
    p.add_argument('--parts-dir', type=Path, help='reuse archive pieces retained by sync_h100.sh')
    p.add_argument('--code-only', action='store_true', help='refresh B code after full dataset transfer')
    p.add_argument('--port', type=int, default=23151)
    p.add_argument('--public-host', default='ziyanglin.com')
    p.add_argument('--remote-repo', default='/home/exouser/openpi')
    a = p.parse_args()
    if not a.code_only and not (a.dataset/'conversion/verification.json').exists():
        p.error('verify the full dataset first')
    repo = Path(__file__).resolve().parents[3]
    staging = Path(tempfile.mkdtemp(prefix='r17_groot_http.', dir='/tmp'))
    code = staging/'code.tar.gz'
    subprocess.run(['tar', '--exclude=__pycache__', '-czf', str(code), '-C', str(repo),
                    'exp/mw_finetune/groot'], check=True)
    if a.code_only:
        parts = []
    elif a.parts_dir:
        parts = sorted(a.parts_dir.glob('data.part-*'))
        if not parts: p.error('no archive pieces in --parts-dir')
    else:
        producer = subprocess.Popen(['tar', '-cf', '-', '-C', str(a.dataset.parent), a.dataset.name], stdout=subprocess.PIPE)
        subprocess.run(['split', '-b', '512M', '-', str(staging/'data.part-')], stdin=producer.stdout, check=True)
        producer.stdout.close()
        if producer.wait(): raise RuntimeError('dataset archive failed')
        parts = sorted(staging.glob('data.part-*'))
    files = [code, *parts]
    manifest = {'files': []}
    for path in files:
        with path.open('rb') as f: digest = hashlib.file_digest(f, 'sha256').hexdigest()
        manifest['files'].append({'name': path.name, 'bytes': path.stat().st_size, 'sha256': digest})
    manifest_path = staging/'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    token = secrets.token_hex(16)
    allowed = {f'/{token}/{path.name}': path for path in [*files, manifest_path]}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = allowed.get(self.path)
            if path is None: self.send_error(404); return
            self.send_response(200)
            self.send_header('Content-Length', str(path.stat().st_size))
            self.end_headers()
            with path.open('rb') as f: shutil.copyfileobj(f, self.wfile, 8*1024*1024)
        def log_message(self, *_): pass

    server = ThreadingHTTPServer(('0.0.0.0', a.port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://{a.public_host}:{a.port}/{token}/'
    remote_staging = '/tmp/' + staging.name
    print('Transfer staging retained:', staging, a.parts_dir or '', flush=True)
    try:
        # Encode multiline code so the remote launcher gets one compact argument.
        command = f"exec(__import__('base64').b64decode('{base64.b64encode(REMOTE.encode()).decode()}'))"
        subprocess.run(['tether', 'exec', '--cwd', '/tmp', '--timeout', '30m', '--safe', 'h100', 'python3', '-c', command,
                        base, remote_staging, a.remote_repo], check=True, timeout=1800)
    finally:
        server.shutdown()
        server.server_close()
    print('Remote code and dataset ready; run setup/verify before GPU work.', flush=True)


if __name__ == '__main__':
    main()
