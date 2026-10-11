"""Download, verify and install a fresh exact runtime into our own scratch paths."""
import hashlib, json, os, pathlib, tarfile, time, urllib.request
import sys
ready = json.loads(pathlib.Path(sys.argv[1]).read_text())
stage = pathlib.Path('/tmp/r17_t1_provision_20261010')
stage.mkdir(parents=True,exist_ok=True)
archive=stage/'runtime.tar'
started=time.monotonic()
if archive.exists(): raise RuntimeError('refuse to overwrite completed transfer')
h=hashlib.sha256()
with urllib.request.urlopen(ready['url'],timeout=120) as src, pathlib.Path(str(archive)+'.part').open('xb') as dst:
    while True:
        chunk=src.read(8*1024*1024)
        if not chunk: break
        dst.write(chunk); h.update(chunk)
if h.hexdigest()!=ready['sha256']: raise RuntimeError('transport SHA mismatch')
os.rename(str(archive)+'.part',archive)
with tarfile.open(archive) as tar:
    members=tar.getmembers()
    own='/scratch/zixuans8/'
    roots=['/scratch/zixuans8/openpi/.venv','/scratch/zixuans8/mw_r17/formal/dependencies','/scratch/zixuans8/mw_r17/formal/assets','/scratch/zixuans8/mw_r17/formal/payload_e9d40ea8b0f96d1cf1b4c6829ad2f6e358d72fe9d8c69e156328d3102a06304d','/scratch/zixuans8/mw_r17/timan1_provision']
    if pathlib.Path(roots[0]).exists(): raise RuntimeError('refuse to overwrite existing venv')
    for m in members:
        p=pathlib.Path('/'+m.name)
        if '..' in p.parts or not any(str(p)==r or str(p).startswith(r+'/') for r in roots): raise RuntimeError('unsafe archive path '+m.name)
        if not (m.isfile() or m.isdir() or m.issym()): raise RuntimeError('unexpected archive member')
        if m.issym():
            target=pathlib.Path(m.linkname) if m.linkname.startswith('/') else p.parent/m.linkname
            target=target.resolve()
            if not str(target).startswith(own) and not str(target).startswith('/home/zixuans8/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/'):
                raise RuntimeError('unsafe runtime symlink '+m.name)
    tar.extractall('/',members=members)
root=pathlib.Path('/scratch/zixuans8/mw_r17/timan1_provision')
manifest=json.loads((root/'SOURCE_MANIFEST.json').read_text())
for name,pin in manifest['files'].items():
    h=hashlib.sha256()
    with pathlib.Path(name).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    if h.hexdigest()!=pin: raise RuntimeError('installed SHA mismatch '+name)
for name,target in manifest['links'].items():
    if os.readlink(name)!=target: raise RuntimeError('installed link mismatch '+name)
base=manifest['base_python']
if hashlib.sha256(pathlib.Path(base['path']).read_bytes()).hexdigest()!=base['sha256']: raise RuntimeError('base Python differs')
receipt={'status':'PASS','archive_sha256':ready['sha256'],'archive_bytes':archive.stat().st_size,'verified_files':len(manifest['files']),'verified_links':len(manifest['links']),'base_python':base,'seconds':time.monotonic()-started}
(root/'INSTALL_RECEIPT.json').write_text(json.dumps(receipt,sort_keys=True)+'\n')
print(json.dumps(receipt),flush=True)
