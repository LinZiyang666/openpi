"""Final read-only evidence audit, writes only K9 audit.json. No GPU context."""
from common import *
import ast,collections,math
stamp=float((OUT/'FINAL_CODE_TIME.txt').read_text())
ledger=json.loads((OUT/'final_commands.json').read_text())
final=[r for r in ledger if r['exit']==0 and r['started']>=stamp]
partial='--partial' in sys.argv
assert 0<len(final)<=48,len(final)
if not partial:assert len(final)==48,len(final)
counts=collections.Counter();data=[]
for ent in final:
 j=ent['job'];kind=j['kind'];run=ent['run'];cell=j['cell'];precision=j['precision']
 name=f'bench_{cell}_{precision}_r{run}.json' if kind=='bench' else f'stage_{cell}_r{run}.json'
 r=json.loads((OUT/name).read_text());data.append(r)
 counts[(kind,cell,precision)]+=1
 for file in ('gpu_awm.py','benchmark.py','common.py','stage1_adapter.py','bench_stage1.py'):
  assert r['source_sha256'][file]==hashlib.sha256((OUT/file).read_bytes()).hexdigest(),(name,file)
 assert r['gpu']['peak_reserved_mib']<=7168
 assert min(s['free_mib'] for s in r['gpu']['samples'])>=4096
 assert max(s.get('own_mib',0) for s in r['gpu']['samples'])<=8192
 if kind=='bench':
  p=r['parity'];assert p['n']==(1037 if 'spatial' in cell else 2200)
  assert all(math.isfinite(v) for v in p['max_abs'].values())
  assert all(math.isfinite(v) for s in p['delta_stats'].values() for v in s.values())
  assert p['agreement']['top1_agree']/p['n']>=.999
  assert r['append']['addresses_unchanged'] and r['append']['appended_row_seen'] and r['append']['graph_eager_topk_equal']
  assert r['append']['graph_eager_action_max_abs']==0
  assert r['batch8_same_query_parity']['topk']
 else:
  assert r['pool_matches_cpu_keybuilder']
  for p in r['parity']:
   if p['kind']=='changed_image_task_replay':
    assert p['all_stage_fields_exact'] and p['graph_eager_action_max_abs']==0 and p['graph_eager_packet_max_abs']==0
assert max(counts.values())<=2,counts
if not partial:assert len(counts)==24 and set(counts.values())=={2},counts
for p in OUT.glob('*.py'):ast.parse(p.read_text())
result=dict(experiment_complete=len(final)==48,n_successful_final_processes=len(final),
            successful_cell_precision_groups=len(counts),processes_per_group=dict(min=min(counts.values()),max=max(counts.values())),
            strict_chunk_1e4_gate_pass=all(r['parity']['max_abs']['action']<=1e-4 for r in data if isinstance(r['parity'],dict)),
            top1_999_gate_pass=True,
            max_observed_own_mib=max(s.get('own_mib',0) for r in data for s in r['gpu']['samples']),
            peak_torch_reserved_mib=max(r['gpu']['peak_reserved_mib'] for r in data),
            minimum_observed_free_mib=min(s['free_mib'] for r in data for s in r['gpu']['samples']),
            source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.py')},
            failed_final_attempts=[e for e in ledger if e['exit'] and e['started']>=stamp])
dump(OUT/('audit_partial.json' if partial else 'audit.json'),result);print(json.dumps({k:v for k,v in result.items() if k not in ('source_hashes','failed_final_attempts')},indent=2))
