"""Read-only H100 CPU comparison after pull_fw_reference.sh. No GPU imports/launches."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
RUNS = {
    '4090': ['4090_fastwam_concurrent_fw4090_reduced_20261018T140530Z',
             '4090_fastwam_concurrent_fw4090_reduced_20261018T141216Z'],
    'h100': ['h100_fastwam_concurrent_fw4090_reduced_20261018T123618Z',
             'h100_fastwam_concurrent_fw4090_reduced_20261018T124033Z'],
}


def remote_program():
    receipts = {}
    for card in RUNS:
        floor = json.loads((ROOT / f'evidence/fw_{card}_floor_summary_1018.json').read_text())
        for run in floor['audit']:
            receipts[run['run']] = {k: run[k] for k in ('run_sha256', 'batch_coverage')}
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in ('concurrent_compare.py', 'neighbor_boundary.py', 'artifacts.py')}
    stats = json.loads((ROOT / 'evidence/fw_normalization_stats_1018.json').read_text())
    bounds = next(iter(stats.values()))['tensors']
    code = '''import json, sys, hashlib
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, '/home/exouser/xgpu_1017/tools')
import numpy as np
import concurrent_compare as cc
'''
    code += f'expected_hashes={hashes!r}\nreceipts={receipts!r}\nruns={RUNS!r}\nbounds={bounds!r}\n'
    code += '''for name, expected in expected_hashes.items():
    assert hashlib.sha256((Path('/home/exouser/xgpu_1017/tools')/name).read_bytes()).hexdigest()==expected, name
paths={card:[Path('/home/exouser/xgpu_1017')/('reference_results' if card=='4090' else 'results')/name for name in names] for card,names in runs.items()}
for group in paths.values():
    for root in group:
        assert hashlib.sha256((root/'RUN.json').read_bytes()).hexdigest()==receipts[root.name]['run_sha256'], root
# Reuse verified native-ledger counts from the on-host floor audit, avoiding
# another bulk log read and avoiding transfer of 108 MB of native ledgers.
def audited_batch_coverage(root):
    return receipts[Path(root).name]['batch_coverage']
base_errors=cc.errors
amin=np.array(bounds['action.min'],dtype=np.float64)[:6]
amax=np.array(bounds['action.max'],dtype=np.float64)[:6]
gain=(amax-amin)/2
assert np.all(gain>0)
def fw_errors(a,b):
    result=base_errors(a,b)
    an=(np.asarray(a,dtype=np.float64)[...,:6]-amin)/gain-1
    bn=(np.asarray(b,dtype=np.float64)[...,:6]-amin)/gain-1
    normalized=base_errors(an,bn)
    result['normalized6_max_abs']=normalized['max_abs']
    result['normalized6_relative_l2']=normalized['relative_l2']
    return result
'''
    code += '\n' + (ROOT / 'summarize_concurrent.py').read_text() + '\n'
    code += '''with patch.object(cc,'batch_coverage',audited_batch_coverage),patch.object(cc,'errors',fw_errors):
    result=calculate(paths['4090'][0],paths['h100'][0],paths['4090'][1:],paths['h100'][1:])
result['scope']='4090 versus H100; two cold starts and four repetitions per GPU; read-only CPU analysis'
summary=summarize(result)
summary['batch_coverage_provenance']='reused on-host floor audit counts, matched by RUN.json SHA256; omitted ledgers not re-read'
summary['normalization_note']='six motion channels inverse affine only; sign gripper not invertible; returned chunks, not denoising/prefix tensors; raw budgets unchanged'
summary['source_hashes']=expected_hashes
summary['run_receipts']=receipts
summary['by_group_normalized6']={}
for group in summary['by_group']:
    rows=[r for r in result['rows'] if r['key'][0]==group]
    summary['by_group_normalized6'][group]={}
    for side,key in [('4090','concurrent_floor_left'),('h100','concurrent_floor_right'),('cross','cross')]:
        summary['by_group_normalized6'][group][side]={
            'max_abs':max(r[key]['normalized6_max_abs'] for r in rows),
            'relative_l2':max(r[key]['normalized6_relative_l2'] for r in rows),
            'over_historical_first5_abs':sum(r[key]['normalized6_max_abs']>.0078125+1e-6 for r in rows) if group=='formal_full_v1' else None,
            'historical_threshold_applicable':group=='formal_full_v1'}
# Derived per-query metrics/decisions are returned, never raw observations or arrays.
rows=[{k:v for k,v in row.items() if k!='scores'} for row in result['rows']]
print(json.dumps(dict(summary=summary,rows=rows)))
'''
    return code


def main():
    cmd = ['tether', 'exec', '--timeout', '10m', 'h100', '--', 'env',
           'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1', 'OPENBLAS_NUM_THREADS=1',
           '/home/exouser/openpi/.venv/bin/python', '-c', remote_program()]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (ROOT / 'evidence/fw_cross_analysis_1018.log').write_text(proc.stderr)
    if proc.returncode:
        raise SystemExit(f'CPU comparison failed rc={proc.returncode}; see evidence/fw_cross_analysis_1018.log')
    result = json.loads(proc.stdout)
    (ROOT / 'evidence/fw_4090_h100_full_1018.json').write_text(json.dumps(result, indent=2)+'\n')
    (ROOT / 'evidence/fw_4090_h100_summary_1018.json').write_text(json.dumps(result['summary'], indent=2)+'\n')
    print(json.dumps({k: result['summary'][k] for k in ('scope','requests','outside_budget','outside_floor','max_abs')}))


if __name__ == '__main__':
    main()
