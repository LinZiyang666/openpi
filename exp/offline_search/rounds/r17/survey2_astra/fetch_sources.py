import pathlib,json,urllib.request,urllib.parse,concurrent.futures,sys
O=pathlib.Path(__file__).parent

def get(url,path):
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'robot-policy-survey/1.0'})
  with urllib.request.urlopen(req,timeout=45) as r: b=r.read(6_000_001)
  if len(b)>6_000_000: return 'SKIP too large '+url
  path.parent.mkdir(exist_ok=True,parents=True);path.write_bytes(b)
  return b.decode(errors='replace')
 except Exception as e: return 'ERROR '+url+' '+str(e)
def hf(mid):
 d=O/'raw'/'hf'/mid.replace('/','__');s=get('https://huggingface.co/api/models/'+mid+'?blobs=true',d/'metadata.json')
 try:m=json.loads(s)
 except:return s
 files=[(x['rfilename'],x.get('size')) for x in m.get('siblings',[])]
 for f,z in files:
  if f.lower().endswith(('.md','.json','.yaml','.yml')) and (z is None or z<200000) and not 'tokenizer' in f:
   get('https://huggingface.co/'+mid+'/resolve/main/'+urllib.parse.quote(f),d/f)
 return json.dumps({'id':mid,'created':m.get('createdAt'),'gated':m.get('gated'),'license':m.get('cardData',{}).get('license'),'files':files})
def gh(repo):
 d=O/'code'/repo.replace('/','__');s=get('https://api.github.com/repos/'+repo+'/git/trees/HEAD?recursive=1',d/'tree.json')
 try:m=json.loads(s)
 except:return s
 fs=[x['path'] for x in m.get('tree',[]) if x['type']=='blob']
 for f in fs:
  if f in ['README.md','Instructions.md','LICENSE','LICENSE.txt']:
   get('https://raw.githubusercontent.com/'+repo+'/HEAD/'+f,d/f)
 return json.dumps({'repo':repo,'files':fs})
if __name__=='__main__':
 mode=sys.argv[1];xs=sys.argv[2:]
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
  for result in ex.map(hf if mode=='hf' else gh,xs):print(result)
