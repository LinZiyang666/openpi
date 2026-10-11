import urllib.request,json,pathlib,hashlib,concurrent.futures,datetime
ROOT=pathlib.Path(__file__).resolve().parents[1]/'sources'
URLS={
 'xvla_readme':'https://raw.githubusercontent.com/2toinf/X-VLA/main/README.md',
 'oft_readme':'https://raw.githubusercontent.com/moojink/openvla-oft/main/README.md',
 'flower_readme':'https://raw.githubusercontent.com/intuitive-robots/flower_vla_calvin/main/README.md',
 'rc_docs':'https://raw.githubusercontent.com/robocasa/robocasa/main/docs/benchmarking/policy_learning_algorithms.md',
 'rc_leaderboard':'https://api.github.com/repos/robocasa-benchmark/leaderboard/git/trees/main?recursive=1',
 'rc_dp_tree':'https://api.github.com/repos/robocasa-benchmark/diffusion_policy/git/trees/main?recursive=1',
 'rc_hf':'https://huggingface.co/api/models/robocasa/robocasa365_checkpoints',
 'groot17_hf':'https://huggingface.co/api/models/nvidia/GR00T-N1.7-LIBERO',
 'groot16_hf':'https://huggingface.co/api/models/nvidia/GR00T-N1.6-3B',
 'xvla_hf':'https://huggingface.co/api/models/2toINF/X-VLA-Libero',
 'pi0_hf':'https://huggingface.co/api/models/lerobot/pi0_libero_base',
 'smolvla_hf':'https://huggingface.co/api/models/HuggingFaceVLA/smolvla_libero',
 'liberoplus_readme':'https://raw.githubusercontent.com/sylvestf/LIBERO-plus/main/README.md',
 'liberopro_readme':'https://raw.githubusercontent.com/Zxy-MLlab/LIBERO-PRO/main/README.md',
 'rc_paper':'https://robocasa.ai/assets/robocasa365_iclr26.pdf',
}
def fetch(item):
 name,url=item
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'research-check/1.0'})
  with urllib.request.urlopen(req,timeout=30) as r:
   data=r.read(1500001)
   if len(data)>1500000: return {'name':name,'url':url,'error':'too large, not saved'}
  suffix='.json' if '/api/' in url or 'api.github' in url else '.pdf' if url.endswith('.pdf') else '.txt'
  path=ROOT/(name+suffix);path.write_bytes(data)
  return {'name':name,'url':url,'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 except Exception as e: return {'name':name,'url':url,'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as p: rows=list(p.map(fetch,URLS.items()))
(ROOT/'manifest.json').write_text(json.dumps(rows,indent=2))
for r in rows:print(r['name'],r.get('bytes',r.get('error')))
