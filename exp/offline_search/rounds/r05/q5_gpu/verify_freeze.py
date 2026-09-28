from pathlib import Path
import ast,hashlib,json
B=Path(__file__).resolve().parent
sources={str(p.relative_to(B)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (B/'dev/plugin.py',B/'dev/gpu_retrieval.py')}
for name in sources:ast.parse((B/name).read_text())
(B/'results/frozen_sources.json').write_text(json.dumps(sources,indent=2)+'\n');print(sources)
