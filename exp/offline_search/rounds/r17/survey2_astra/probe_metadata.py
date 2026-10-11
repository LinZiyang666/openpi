import sys,json,pathlib,zipfile,io,pickle,urllib.request,struct,collections,math
from fetch_sources import O,get
class IgnoreClass:
 def __new__(cls,*a,**kw):return {'__class__':'flax.nnx.filterlib.Nothing'}
class SafeUnpickler(pickle.Unpickler):
 def find_class(self,module,name):
  if (module,name)==('flax.nnx.filterlib','Nothing'):return IgnoreClass
  raise pickle.UnpicklingError((module,name))
for name,base in [('libero','dp_libero_lang_v1/30000'),('metaworld','dp_metaworld_lang_v1/85000')]:
 p=O/'raw'/('brandon_'+name+'_metadata.pt');z=zipfile.ZipFile(p);b=z.read([n for n in z.namelist() if n.endswith('data.pkl')][0]);d=SafeUnpickler(io.BytesIO(b)).load();(O/'raw'/('brandon_'+name+'_config.json')).write_text(json.dumps(d,indent=2));print(name,d['config']['model'])
 url=f'https://huggingface.co/brandonyang/dp-{name}-checkpoints/resolve/main/{base}/model.safetensors'
 req=urllib.request.Request(url,headers={'Range':'bytes=0-131071'})
 with urllib.request.urlopen(req,timeout=45) as r:
  b=r.read(131072);print('HEADER',r.status,r.headers.get('Content-Range'),len(b))
 n=struct.unpack('<Q',b[:8])[0]
 if n>131064: print('HEADER TOO LONG',n);continue
 d=json.loads(b[8:8+n]);(O/'raw'/('brandon_'+name+'_safetensors_header.json')).write_text(json.dumps(d,indent=2))
 counts=collections.Counter()
 for k,v in d.items():
  if k=='__metadata__':continue
  counts['.'.join(k.split('.')[:2])]+=math.prod(v['shape'])
 print('PARAMS',dict(counts),'firstkeys',list(d)[:20])
# list ZIP central directory from 64KiB tail only, without tensor data
b=(O/'raw'/'baku_zip_tail.bin').read_bytes();size=2561746368;tail_start=size-len(b);i=b.rfind(b'PK\x05\x06');e=struct.unpack('<4s4H2LH',b[i:i+22]);j=e[6]-tail_start;entries=[]
while b[j:j+4]==b'PK\x01\x02':
 h=struct.unpack('<4s6H3L5H2L',b[j:j+46]);n,extra,comm=h[10:13];name=b[j+46:j+46+n].decode();entries.append({'path':name,'compressed':h[8],'size':h[9],'method':h[4],'offset':h[-1]});j+=46+n+extra+comm
(O/'raw'/'baku_zip_listing.json').write_text(json.dumps(entries,indent=2));print('BAKU ZIP',entries)
