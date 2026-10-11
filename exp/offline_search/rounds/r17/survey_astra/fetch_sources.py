import concurrent.futures, hashlib, json, pathlib, sys, urllib.request, urllib.error, datetime
root=pathlib.Path(__file__).resolve().parent
jobs=json.loads(sys.stdin.read())
def fetch(job):
    name,url=job
    out=root/'sources'/name
    req=urllib.request.Request(url,headers={'User-Agent':'independent-policy-research/1.0'})
    try:
        with urllib.request.urlopen(req,timeout=35) as r:
            data=r.read(5_000_001)
            if len(data)>5_000_000: raise ValueError('metadata size limit')
            meta={'url':url,'final_url':r.url,'status':r.status,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        out.write_bytes(data)
    except Exception as e: meta={'url':url,'error':str(e)}
    out.with_name(out.name+'.meta.json').write_text(json.dumps(meta,indent=2))
    return name,meta
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    for name,meta in pool.map(fetch,jobs): print(name,json.dumps(meta))
