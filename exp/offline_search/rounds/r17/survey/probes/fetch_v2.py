import urllib.request,json,pathlib,hashlib,concurrent.futures,datetime,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]/'sources_v2'
def fetch(item):
 name,url=item
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'research-verification'}),timeout=35) as r:
   data=r.read(1800001)
   if len(data)>1800000: raise ValueError('exceeds 1.8 MB metadata cap')
  path=ROOT/(name+'.txt');path.write_bytes(data)
  return dict(name=name,url=url,path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 except Exception as e:return dict(name=name,url=url,error=str(e))
if __name__=='__main__':
 urls=json.loads(sys.stdin.read())
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as p:rows=list(p.map(fetch,urls.items()))
 mf=ROOT/'manifest.json';old=json.loads(mf.read_text()) if mf.exists() else [];mf.write_text(json.dumps(old+rows,indent=2))
 for row in rows: print(row['name'],row.get('bytes',row.get('error')))
