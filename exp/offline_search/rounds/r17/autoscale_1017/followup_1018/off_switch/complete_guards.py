from pathlib import Path
import json,hashlib
b=Path('exp/offline_search/rounds/r17/autoscale_1017');d=b/'followup_1018/off_switch';root=Path.cwd()
def edit(n,fn):
 p=Path(n);bk=d/n;bk.parent.mkdir(parents=True,exist_ok=True)
 if not bk.exists():bk.write_bytes(p.read_bytes())
 s=p.read_text();o=fn(s);assert s!=o,n;p.write_text(o)
# Freeze native payload bytes in the deliverable, independent of temporary backups.
native=b/'native_sources';native.mkdir(exist_ok=True)
pins=json.loads((b/'PATCH_BASE.json').read_text());manifest={}
for name,row in pins.items():
 data=(Path('/home/weiland/.claude/jobs/35f2869b/tmp/autoscale_backup')/name.replace('/','__')).read_bytes()
 assert hashlib.sha256(data).hexdigest()==row['sha256']
 if name.endswith('/control.py'):
  text=data.decode();old='        s.bind(("0.0.0.0", port))'
  # Preserve the coordinator's transport fix also in newly sealed native bytes.
  current=(root/name).read_text();lines=current.splitlines();i=next(i for i,l in enumerate(lines) if 's.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)' in l)
  addition='\n'.join(lines[i-2:i+1])+'\n'
  if old in text:text=text.replace(old,addition+old,1)
  else:
   target=next(l for l in text.splitlines() if 's.bind(' in l);text=text.replace(target,addition+target,1)
  data=text.encode()
 (native/name.replace('/','__')).write_bytes(data);manifest[name]=hashlib.sha256(data).hexdigest()
(native/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
with (b/'switch.py').open('a') as out:out.write('''

def source(path):
    """Native bytes for NEW archives when OFF; never read or alter frozen runs."""
    path = Path(path)
    root = Path(__file__).resolve().parents[5]
    try:name = str(path.resolve().relative_to(root))
    except ValueError:return path.read_bytes()
    native = Path(__file__).with_name('native_sources') / name.replace('/', '__')
    if not enabled() and native.is_file():return native.read_bytes()
    return path.read_bytes()
''')
# Import is local to coordinator packaging: archived native builder is its pre-switch version.
for name in ['exp/r17_integration/ops/build_client_bundle.py','exp/r17_integration/qualification/deploy_cpu.py']:
 p=Path(name);data=p.read_bytes();(native/name.replace('/','__')).write_bytes(data);manifest[name]=hashlib.sha256(data).hexdigest()
(native/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
edit('exp/r17_integration/ops/build_client_bundle.py',lambda s:s.replace('        content[name] = source.read_bytes()','        from exp.offline_search.rounds.r17.autoscale_1017.switch import source as preparation_source\n        content[name] = preparation_source(source)',1))
edit('exp/r17_integration/qualification/deploy_cpu.py',lambda s:s.replace('                tar.add(path, arcname=str(path.relative_to(root)), recursive=False)','                from exp.offline_search.rounds.r17.autoscale_1017.switch import source as preparation_source\n                import io\n                data = preparation_source(path)\n                info = tar.gettarinfo(str(path), arcname=str(path.relative_to(root)))\n                info.size = len(data)\n                tar.addfile(info, io.BytesIO(data))',1))
edit('exp/r17_integration/ops/local_workers.py',lambda s:s.replace("            p=temp/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((Path(root)/name).read_bytes())","            from exp.offline_search.rounds.r17.autoscale_1017.switch import source as preparation_source\n            p=temp/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(preparation_source(Path(root)/name))",1))
edit('exp/r17_integration/ops/driver_release.py',lambda s:s.replace("            content.update({name:(checkout/name).read_bytes() for name in overlay})","            from exp.offline_search.rounds.r17.autoscale_1017.switch import source as preparation_source\n            content.update({name:preparation_source(checkout/name) for name in overlay})",1))
edit('exp/r17_integration/ops/formal_eval.py',lambda s:s.replace("        args.workers = physical_maximum(args.worker_node, 'libero' if args.model.endswith('_libero') else 'metaworld')","        from exp.offline_search.rounds.r17.autoscale_1017.switch import enabled\n        args.workers = physical_maximum(args.worker_node, 'libero' if args.model.endswith('_libero') else 'metaworld') if enabled() else 24",1))
# Queue definition uses original template resource counts when disabled.
edit('exp/offline_search/rounds/r17/filter_failed_knob/formal_a_queue.py',lambda s:s.replace("WORKERS = {(model, mode): maxima", "from exp.offline_search.rounds.r17.autoscale_1017.switch import enabled\nWORKERS = {(model, mode): maxima",1).replace("workers=WORKERS.get((model,mode),source['workers']),jobs=[]","workers=(WORKERS.get((model,mode),source['workers']) if enabled() else source['workers']),jobs=[]",1))
# Source closure for eventual enabled trials. OFF archives use native copies above.
edit(str(b/'bundle.py'),lambda s:s.replace("'defaults')","'defaults', 'switch')",1).replace("return [PREFIX + name + '.py' for name in MODULES] + [","return [PREFIX + name + '.py' for name in MODULES] + [PREFIX + 'ENABLED',",1))
print('All native source pins saved; OFF bundle source resolver installed')
