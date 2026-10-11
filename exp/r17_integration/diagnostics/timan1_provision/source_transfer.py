"""One-shot, token-gated transfer of our exact simulator environment. No GPU use."""
import hashlib, http.server, json, os, pathlib, secrets, tarfile, time
ROOT = pathlib.Path('/scratch/zixuans8/mw_r17/timan1_provision')
ROOT.mkdir(parents=True, exist_ok=True)
DEP = 'libero311_c786215bc2061c66ed8851f6aa32e7199a9f2e364b12c47e61235f5a3e83a7c7'
ASSET = 'libero_3eb6e3e7ee3a03663dba84ccc6d6cf9dbb4ae21fb20558ca4097abc066be57dc'
PAYLOAD = 'payload_e9d40ea8b0f96d1cf1b4c6829ad2f6e358d72fe9d8c69e156328d3102a06304d'
paths = [pathlib.Path('/scratch/zixuans8/openpi/.venv'), pathlib.Path('/scratch/zixuans8/mw_r17/formal/dependencies')/DEP, pathlib.Path('/scratch/zixuans8/mw_r17/formal/assets')/ASSET, pathlib.Path('/scratch/zixuans8/mw_r17/formal')/PAYLOAD]
archive = ROOT/'runtime.tar'
if archive.exists():
    raise RuntimeError('refuse to overwrite prior transfer')
manifest = {'files':{}, 'links':{}, 'roots':[str(p) for p in paths]}
class HashReader:
    def __init__(self, f): self.f, self.h = f, hashlib.sha256()
    def read(self, size=-1):
        data = self.f.read(size); self.h.update(data); return data
started=time.monotonic()
with tarfile.open(str(archive)+'.part','w') as tar:
    for root in paths:
        for p in [root]+sorted(root.rglob('*')):
            if '__pycache__' in p.parts or p.suffix=='.pyc': continue
            name=str(p).lstrip('/')
            info=tar.gettarinfo(str(p), arcname=name)
            if p.is_symlink():
                manifest['links'][str(p)]=os.readlink(str(p)); tar.addfile(info)
            elif p.is_file():
                with p.open('rb') as f:
                    reader=HashReader(f); tar.addfile(info,reader)
                    manifest['files'][str(p)]=reader.h.hexdigest()
            elif p.is_dir(): tar.addfile(info)
            else: raise RuntimeError('unexpected nonregular runtime file '+str(p))
    base=pathlib.Path('/scratch/zixuans8/openpi/.venv/bin/python').resolve()
    manifest['base_python']={'path':str(base),'sha256':hashlib.sha256(base.read_bytes()).hexdigest()}
    (ROOT/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,sort_keys=True)+'\n')
    tar.add(str(ROOT/'SOURCE_MANIFEST.json'),arcname=str(ROOT/'SOURCE_MANIFEST.json').lstrip('/'))
os.rename(str(archive)+'.part',archive)
h=hashlib.sha256()
with archive.open('rb') as f:
    for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
token=secrets.token_hex(24)
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path!='/'+token+'/runtime.tar': self.send_error(404); return
        self.send_response(200); self.send_header('Content-Length',str(archive.stat().st_size)); self.end_headers()
        with archive.open('rb') as f:
            while True:
                data=f.read(1024*1024)
                if not data: break
                self.wfile.write(data)
    def log_message(self,*unused): pass
server=http.server.HTTPServer(('0.0.0.0',0),Handler)
ready={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':h.hexdigest(),'files':len(manifest['files']),'port':server.server_port,'url':'http://192.17.58.207:'+str(server.server_port)+'/'+token+'/runtime.tar','pid':os.getpid(),'build_seconds':time.monotonic()-started}
(ROOT/'TRANSFER_READY.json').write_text(json.dumps(ready)+'\n')
print(json.dumps({k:v for k,v in ready.items() if k!='url'}),flush=True)
server.timeout=2
until=time.monotonic()+1800
while time.monotonic()<until and not (ROOT/'STOP_TRANSFER').exists(): server.handle_request()
server.server_close()
