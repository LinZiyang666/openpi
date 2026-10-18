"""CPU-only compact audit of a concurrent comparison, retaining failure examples."""
from collections import Counter
import json
from pathlib import Path

def summarize_rows(rows, budget):
    """Keep the DP analysis API; never substitute cross error for either floor."""
    result={'requests':len(rows),'outside_floor':sum(not r['within_floor'] for r in rows)}
    for name,key in [('4090_floor','concurrent_floor_left'),('h100_floor','concurrent_floor_right'),('cross','cross')]:
        result[name]={metric+'_max':max(r[key][metric] for r in rows) for metric in rows[0][key]}
        result[name]['outside_absolute_budget']=sum(r[key]['max_abs']>budget['max_abs'] for r in rows)
        result[name]['outside_relative_budget']=sum(r[key]['relative_l2']>budget['relative_l2'] for r in rows)
        result[name]['outside_budget']=sum(r[key]['max_abs']>budget['max_abs'] or r[key]['relative_l2']>budget['relative_l2'] for r in rows)
    return result

def summarize(report):
    output={k:v for k,v in report.items() if k!='rows'}
    groups={}
    for name in sorted({r['key'][0] for r in report['rows']}):
        rows=[r for r in report['rows'] if r['key'][0]==name]
        heads=[r for r in rows if r['any_segment_head']]
        cell=dict(requests=len(rows),head_requests=len(heads),metrics={},floor={},decisions={})
        for side,key in [('4090','concurrent_floor_left'),('h100','concurrent_floor_right'),('cross','cross')]:
            cell['metrics'][side]={k:max(r[key][k] for r in rows) for k in rows[0][key]}
            cell['metrics'][side]['over_absolute_budget']=sum(r[key]['max_abs']>report['budget']['max_abs'] for r in rows)
            cell['metrics'][side]['over_relative_budget']=sum(r[key]['relative_l2']>report['budget']['relative_l2'] for r in rows)
            cell['metrics'][side]['over_either_budget']=sum(r[key]['max_abs']>report['budget']['max_abs'] or r[key]['relative_l2']>report['budget']['relative_l2'] for r in rows)
        cell['outside_floor']=sum(not r['within_floor'] for r in rows)
        cell['gripper_sign_disagree']=sum(not r['gripper_sign_agree'] for r in rows)
        for field in report['agreement']:
            observed=[r['decisions'][field] for r in heads if r['decisions'][field]['available']]
            cell['decisions'][field]=dict(observed_heads=len(observed),unknown_heads=len(heads)-len(observed),
                **{mode:sum(x[mode] for x in observed) for mode in ('left_agree','right_agree','cross_agree')})
        regions={}
        for row in heads:
            n=row['neighbour_candidate_audit'];region='unknown' if not n['qualified'] else 'near' if n['near_boundary'] else 'away'
            entry=regions.setdefault(region,dict(heads=0,max_action_error=0.,top1_flips=0,donor_order_flips=0))
            entry['heads']+=1;entry['max_action_error']=max(entry['max_action_error'],row['cross']['max_abs'])
            for field,label in [('top1_neighbour','top1_flips'),('donor_rows','donor_order_flips')]:
                d=row['decisions'][field];entry[label]+=d['available'] and not d['cross_agree']
        cell['neighbor_regions']=regions
        def compact(r):
            return {k:v for k,v in r.items() if k!='scores'}
        cell['worst_action']=compact(max(rows,key=lambda r:r['cross']['max_abs']))
        cell['decision_flip_examples']=[compact(r) for r in heads if any(d['available'] and not d['cross_agree'] for field,d in r['decisions'].items() if field in ('source','hit_type','top1_neighbour','donor_rows','corrector_applied','stall_call','stall_state','knob_call'))][:12]
        groups[name]=cell
    output['by_group']=groups
    return output

def run_audit(paths):
    from artifacts import sha
    from concurrent_compare import batch_coverage
    result=[]
    for path in map(Path,paths):
        run=json.loads((path/'RUN.json').read_text());numeric=[]
        for p in sorted(path.glob('servers/replica_*/NATIVE_NUMERICS.json')):
            d=json.loads(p.read_text())
            numeric.append(dict(replica=p.parent.name,sha256=sha(p),**{k:d.get(k) for k in ('torch_flags','packages','driver','cpu_model','python','environment')}))
        m=run['mps']
        result.append(dict(run=path.name,run_sha256=sha(path/'RUN.json'),status=run['status'],
            pin=run['pin']['identity'],runtime=run['pin']['runtime_identity'],bootstrap=run['pin']['bootstrap_sha256'],
            input_identity=run['input_identity'],harness=json.loads((path/'EXECUTION_HARNESS.json').read_text())['receipt']['identity'],
            clients=run['clients'],rounds=run['rounds'],deliveries=run['deliveries'],
            topology=run['campaign']['xgpu_probe'],numerics=run['campaign'].get('xgpu_numerics'),
            segment_clock_overlay=run['pin'].get('segment_clock_overlay'),
            cache_files=json.loads((path/'CACHE_FILES_VERIFICATION.json').read_text()),
            batch_coverage=batch_coverage(path),native_numerics=numeric,
            mps=dict(seconds=m['closed']-m['started'],daemon_closed=m['daemon_closed'],background_contexts_preserved=m['background_contexts_preserved'],background_contexts=m['background_contexts']),
            startup_memory_mib=run['memory']['owned_total_mib'],gpu=run['gpu']))
    return result

def calculate(left,right,left_repeats=(),right_repeats=()):
    import concurrent_compare as cc
    from unittest.mock import patch
    original=cc.load_repeated;loaded={}
    def cached(root,repeats=()):
        key=(str(root),tuple(map(str,repeats)))
        if key not in loaded:loaded[key]=original(root,repeats)
        return loaded[key]
    with patch.object(cc,'load_repeated',cached):
        return cc.compare(left,right,.001,.001,left_repeats,right_repeats)
