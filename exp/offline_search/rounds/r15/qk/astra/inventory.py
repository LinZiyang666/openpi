"""Read-only inventory of explicitly permitted library and B-dev data."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
STORE = Path('/home/weiland/trace_runs/offline_search_store')
RUNS = Path('/home/weiland/trace_runs/os_closed_loop')

def main():
    out = {'libraries': {}, 'recordings': {}, 'trace_files': {}}
    for cell in ('pi05_l10', 'pi05_spatial', 'groot_l10', 'groot_spatial'):
        name = 'bpool_cs' if cell.startswith('pi05') else 'bpool_all'
        p = STORE / 'library' / cell / name
        entry = {'path': str(p), 'arrays': {}}
        for f in list(p.glob('*.npy')) + list((p/'tok').glob('*.npy')):
            a = np.load(f, mmap_mode='r', allow_pickle=False)
            entry['arrays'][str(f.relative_to(p))] = {'shape': list(a.shape), 'dtype': str(a.dtype), 'bytes': f.stat().st_size}
        for n in ('manifest.json', 'episodes.json', 'ids.json'):
            v = json.loads((p/n).read_text())
            entry[n] = (v if n == 'manifest.json' else {'type': type(v).__name__, 'len': len(v), 'sample': v[:2] if isinstance(v, list) else dict(list(v.items())[:2])})
        if (p/'tok/rows.npy').exists():
            r = np.load(p/'tok/rows.npy')
            entry['image_row_map'] = {'n': len(r), 'unique': len(np.unique(r)), 'min': int(r.min()), 'max': int(r.max()), 'identity': bool(np.array_equal(r,np.arange(len(r))))}
        out['libraries'][cell] = entry
    roots = sorted(RUNS.glob('r12_devlooks*')) + [RUNS/'r13_devrec_50', RUNS/'r13_devrec_pi05_spatial_50']
    for root in roots:
        counts = []
        for f in sorted(root.glob('runs/*/server_*/looks/manifest_*.json')):
            m = json.loads(f.read_text())
            # Manifests are B-only by construction; check before accessing shards.
            counts.append({'path': str(f), 'keys': list(m), 'options': m.get('options'), 'contract': m.get('contract'), 'shards': len(m.get('shards', [])), 'first_shard': m.get('shards', [])[:1]})
        out['recordings'][root.name] = counts
    # Filenames only. Do not open any unknown-pool trace shard.
    for p in sorted(Path('/home/weiland/trace_runs/dual_20260923/runs').glob('tr_groot_*')):
        files = list((p/'trace').rglob('*'))
        counts = {}
        for f in files:
            if f.is_file(): counts[f.suffix] = counts.get(f.suffix, 0) + 1
        out['trace_files'][p.name] = {'suffix_counts': counts, 'examples': [str(f.relative_to(p)) for f in files if f.is_file()][:8]}
    (HERE/'inventory.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({'libraries': {k: {'arrays':v['arrays'], 'image_row_map':v.get('image_row_map'), 'episodes':v['episodes.json']} for k,v in out['libraries'].items()}}, indent=2))
    for r, ms in out['recordings'].items():
        print(r, 'manifests',len(ms), 'shards',sum(m['shards'] for m in ms), 'options',str([m['options'] for m in ms[:4]]))
        if ms: print('sample', json.dumps(ms[0])[:1800])
    print('traces', json.dumps(out['trace_files']))

if __name__ == '__main__': main()
