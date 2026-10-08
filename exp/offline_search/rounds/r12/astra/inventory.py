from .common import *
install()
import numpy as np
from exp.offline_search.harness.store import LibraryView

out = {'timestamp': now(), 'libraries': [], 'pca': [], 'dev': []}
for cell in ('pi05_l10', 'pi05_spatial', 'groot_l10', 'groot_spatial'):
    for name in ('current', 'bpool_cs' if cell.startswith('pi05') else 'bpool_all'):
        lib = LibraryView(STORE, cell, name)
        fields = {}
        for f in ('key_v0', 'key_v1', 'rs', 'action', 'episode', 'step', 'progress', 'success'):
            a = getattr(lib, f)
            fields[f] = {'shape': a.shape, 'dtype': str(a.dtype)}
        tok = {f: None if lib.tok(f) is None else {'shape': lib.tok(f).shape, 'dtype': str(lib.tok(f).dtype)}
               for f in ('v0', 'v1', 'img0', 'img1')}
        out['libraries'].append(dict(cell=cell, name=name, rows=lib.L, episodes=len(lib.episodes),
             arrays=fields, tokens=tok, token_rows=len(lib.tok_rows), metadata=lib.meta))
    for size in (50, 200, 500):
        C, lib = load(cell, size)
        out['pca'].append(dict(cell=cell, size=size, rows=len(C['ep']), failed_episodes=int(
            sum(not np.any(C['success'][C['ep'] == e]) for e in np.unique(C['ep']))), provenance=C['provenance']))
for root in sorted(DEV - {'r06_c_cal'}):
    p = RUNS / root
    files = list(p.glob('runs/*/*'))
    out['dev'].append(dict(root=root, files=[str(f.relative_to(p)) for f in files[:24]],
                         summary=json.loads((p / 'summary.json').read_text()) if (p / 'summary.json').exists() else None))
dump(HERE / 'inventory.json', out)
dump(HERE / 'inventory_reads.json', sorted(READS))
print(json.dumps({k: out[k] for k in ('timestamp', 'pca', 'dev')}, indent=2))
