import json, urllib.request, urllib.parse, concurrent.futures, pathlib, collections, datetime
out=pathlib.Path(__file__).parent
queries=['metaworld','meta-world','meta_world','mt50','mt10','libero']
def fetch(q):
    url='https://huggingface.co/api/models?'+urllib.parse.urlencode({'search':q,'limit':1000,'full':'true'})
    pages=[]; models=[]
    while url:
        with urllib.request.urlopen(url,timeout=90) as r:
            data=json.load(r); links=r.headers.get('Link','')
        pages.append({'url':url,'count':len(data),'link':links}); models.extend(data)
        url=None
        for link in links.split(','):
            if 'rel="next"' in link: url=link.split('<')[1].split('>')[0]
    (out/'raw'/f'hf_search_{q}.json').write_text(json.dumps({'query':q,'pages':pages,'models':models},indent=2))
    return q,models
allq=dict(concurrent.futures.ThreadPoolExecutor(max_workers=6).map(fetch,queries))
allm={m['id']:m for a in allq.values() for m in a}
mw={m['id'] for q,a in allq.items() if q!='libero' for m in a}; li={m['id'] for m in allq['libero']}
authmw={x.split('/')[0] for x in mw}; authli={x.split('/')[0] for x in li}; common=authmw&authli
(out/'raw'/'enumeration_summary.json').write_text(json.dumps({'counts':{q:len(a) for q,a in allq.items()},'unique':len(allm),'common_authors':sorted(common),'metaworld_ids':sorted(mw),'common_author_ids':sorted(x for x in allm if x.split('/')[0] in common)},indent=2))
print('COUNTS', {q:len(a) for q,a in allq.items()}, 'unique',len(allm))
print('MW IDS\n'+'\n'.join(sorted(mw)))
print('COMMON AUTHORS',sorted(common))
print('COMMON IDS\n'+'\n'.join(sorted(x for x in allm if x.split('/')[0] in common)))
