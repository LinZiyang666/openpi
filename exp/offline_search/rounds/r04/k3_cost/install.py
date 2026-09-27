"""Install K3-owned shared files atomically; preserve before images and audit times."""
import hashlib,json,os,pathlib,py_compile,time
HERE=pathlib.Path(__file__).resolve().parent
DEST=HERE.parents[2]/'closed_loop'
files={'stage_overrides.py':DEST/'stage_overrides.py','serve_pi05.py':DEST/'serve_pi05.py',
       'serve_groot.py':DEST/'serve_groot.py','start_server.sh':DEST/'ops/start_server.sh'}
log=[]
for name,target in files.items():
    source=HERE/'dev'/name
    if name.endswith('.py'):py_compile.compile(str(source),doraise=True)
    data=source.read_bytes()
    before=target.read_bytes() if target.exists() else None
    if before is not None:(HERE/'dev'/('before_'+name)).write_bytes(before)
    tmp=target.with_name('.'+target.name+'.k3.tmp')
    tmp.write_bytes(data)
    if target.exists():tmp.chmod(target.stat().st_mode)
    os.replace(tmp,target)
    log.append({'file':str(target),'installed_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'before_sha256':hashlib.sha256(before).hexdigest() if before else None,
                'after_sha256':hashlib.sha256(data).hexdigest()})
(HERE/'results/install.json').write_text(json.dumps(log,indent=2))
print(json.dumps(log,indent=2))
