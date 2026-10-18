"""CPU comparison against the SAME-GPU CONCURRENT floor, never B1 drift."""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from collections import Counter
from artifacts import load_tree, sha, write_json
from neighbor_boundary import audit as neighbor_audit

DECISIONS = ('source', 'hit_type', 'top1_neighbour', 'donor_rows', 'stall_call', 'stall_state', 'knob_call',
             'corrector_applied', 'decision_index', 'env_step_index', 'segment_head',
             'decision_seed', 'actual_nfe', 'commit_horizon')
SCORES = ('score', 'cp1_score', 'raw_anchor_distance', 'corrector_ratio', 'stall_score',
          'knob_gap', 'nearest_distance', 'neighbour_gap','stall_delta_hat','stall_e90','stall_a10',
          'corrector_lower_margin','corrector_upper_margin','stall_confirmed_margin','stall_ambiguous_margin')
MARGINS=('neighbour_gap','corrector_lower_margin','corrector_upper_margin',
         'stall_confirmed_margin','stall_ambiguous_margin')


def metadata(value):
    hit=dict(value['response'].get('__hit_meta__',{}))
    donors=hit.get('donor_rows')
    # Frozen layers/recipe.py _retrieve sorts by (distance,row); row0 is top1.
    if donors:hit['top1_neighbour']=donors[0]
    if 'knob_call' not in hit and 'knob' in hit:hit['knob_call']=hit['knob']
    status=hit.get('stall_status') or {}
    if 'state' in status:hit['stall_state']=status['state']
    for name in ('delta_hat','e90','a10'):
        if name in status:hit['stall_'+name]=status[name]
    if all(isinstance(status.get(name),(int,float)) and np.isfinite(status[name])
           for name in ('delta_hat','e90','a10')):
        hit['stall_confirmed_margin']=status['delta_hat']+status['e90']-status['a10']
        hit['stall_ambiguous_margin']=status['delta_hat']-status['a10']
    return hit


def head(hit):
    if type(hit.get('segment_head')) is bool:return hit['segment_head']
    if hit.get('source') in ('policy','library'):return True
    if str(hit.get('source','')).endswith('_tail'):return False
    return None


def errors(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('invalid action tensors')
    d = np.abs(a-b)
    return dict(max_abs=float(d.max()), p99_abs=float(np.quantile(d, .99)),
        max_relative=float((d/np.maximum(np.abs(a), 1e-6)).max()),
        relative_l2=float(np.linalg.norm(d)/max(np.linalg.norm(a), 1e-6)))


def load_run(root):
    root = Path(root)
    run = json.loads((root/'RUN.json').read_text())
    if run['status'] != 'COMPLETE' or run['rounds'] < 2:
        raise ValueError('two COMPLETE concurrent rounds required')
    rounds = {}
    for directory in run['artifacts']:
        d = root/directory
        rows = json.loads((d/'ROWS.json').read_text())
        repeat = rounds.setdefault(rows['round'], {})
        for entry in rows['rows']:
            file = d/entry['file']
            if not file.resolve().is_relative_to(root.resolve()) or sha(file.with_suffix('.json')) != entry['sha256']:
                raise ValueError('changed/unsafe request record')
            key = tuple(entry['key'])
            if key in repeat:
                raise ValueError('duplicate logical request')
            repeat[key] = load_tree(file)
    if set(rounds) != set(range(run['rounds'])):
        raise ValueError('missing round')
    expected = set(next(iter(rounds.values())))
    if not expected or any(set(r) != expected for r in rounds.values()):
        raise ValueError('unmatched logical requests between repeats')
    acknowledgments=json.loads((root/'BUNDLE_ACKS.json').read_text())
    bundles=[x['group']['bundle_id'] for x in acknowledgments]
    if not bundles or len(bundles)!=len(set(bundles)):
        raise ValueError('missing/duplicate native bundle acknowledgments')
    complete={(bundle,i,j)for bundle in bundles
        for i in range(run['clients'])for j in range(run['deliveries'])}
    if expected!=complete:raise ValueError('incomplete concurrent request population')
    return run, rounds


def batch_coverage(root):
    result = {}
    for path in Path(root).glob('servers/replica_*/batching.jsonl'):
        sizes, full_members = {}, 0
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                if 'size' in r:
                    sizes[str(r['size'])] = sizes.get(str(r['size']), 0)+1
                if r.get('members'):
                    full_members += 1
        result[path.parent.name] = dict(sizes=sizes, batches_with_members=full_members)
    return result


def request_coverage(rounds):
    """Report visited branches and telemetry completeness, never infer missing ones."""
    report={}
    for repeat, rows in rounds.items():
        groups={}
        for key,value in rows.items():
            group=groups.setdefault(key[0],dict(requests=0,decisions={},head_decisions={},heads=0,batch_members_observed=0))
            group['requests']+=1
            hit=metadata(value)
            is_head=head(hit) is True
            group['heads']+=is_head
            for field in ('source','hit_type','corrector_applied','stall_call','stall_state','knob_call'):
                counts=group['decisions'].setdefault(field,Counter())
                counts[json.dumps(hit.get(field,'__MISSING__'),sort_keys=True)]+=1
                if is_head:
                    group['head_decisions'].setdefault(field,Counter())[json.dumps(hit.get(field,'__MISSING__'),sort_keys=True)]+=1
            stages=hit.get('batching',{}).get('stages',{}).values()
            if hit.get('xgpu_batch_composition') or any(s.get('members') for s in stages):
                group['batch_members_observed']+=1
        report[str(repeat)]=groups
    return report


def clock_audit(rounds):
    """Compare reported clocks with protocol ACK counts, including long tails.

    Stress ACKs simulate execution; this verifies protocol bookkeeping, not a
    physical trajectory. Missing fields or unknown heads never become a pass.
    """
    reports={}
    for repeat,rows in rounds.items():
        groups={};streams={}
        for key,value in sorted(rows.items()):
            group=groups.setdefault(key[0],dict(requests=0,env_observed=0,decision_observed=0,
                env_mismatches=0,decision_mismatches=0,examples=[]))
            stream=streams.setdefault(key[:2],dict(env_step=0,decision=-1,known=True))
            hit=metadata(value);is_head=head(hit)
            if is_head is None:stream['known']=False
            if is_head is True:stream['decision']+=1
            group['requests']+=1
            for field,expected,label,known in (
                ('env_step_index',stream['env_step'],'env',True),
                ('decision_index',stream['decision'],'decision',stream['known'])):
                if known and type(hit.get(field))is int:
                    group[label+'_observed']+=1
                    if hit[field]!=expected:
                        group[label+'_mismatches']+=1
                        if len(group['examples'])<10:group['examples'].append(dict(key=key,field=field,actual=hit[field],expected=expected))
            ack=value.get('execution')or{}
            if type(ack.get('executed_count'))is not int or ack['executed_count']<0 or ack.get('decision_seq')!=key[2]:
                raise ValueError('invalid recorded stress execution ACK')
            stream['env_step']+=ack['executed_count']
        reports[str(repeat)]=groups
    return reports


def load_repeated(root, repeats=()):
    """Combine independent cold starts on one device, preserving every round."""
    a, ar=load_run(root)
    paths=[str(Path(root).resolve())]
    for repeat in repeats:
        path=str(Path(repeat).resolve())
        if path in paths:raise ValueError('duplicate cold-start directory')
        b,br=load_run(repeat)
        for field in ('pin','input_identity','clients','deliveries','live_reference','harness_files'):
            if a.get(field)!=b.get(field):raise ValueError('cold-start mismatch: '+field)
        for field in ('replicas','memory_budget_gib','xgpu_numerics','xgpu_batch_telemetry'):
            if a['campaign'].get(field)!=b['campaign'].get(field):
                raise ValueError('cold-start serving settings differ: '+field)
        if not a.get('gpu',{}).get('uuid') or a['gpu']['uuid']!=b.get('gpu',{}).get('uuid'):
            raise ValueError('cold starts must be on the same recorded GPU UUID')
        expected=set(next(iter(ar.values())))
        if any(set(rows)!=expected for rows in br.values()):
            raise ValueError('cold-start logical requests differ')
        offset=len(ar)
        ar.update({offset+i:rows for i,rows in enumerate(br.values())})
        paths.append(path)
    return a,ar,paths


def cache_files_attestation(paths):
    identities=set();missing=[]
    for root in paths:
        path=Path(root)/'CACHE_FILES_VERIFICATION.json'
        if not path.exists():missing.append(str(root));continue
        receipt=json.loads(path.read_text());before=receipt.get('before',{});after=receipt.get('after',{})
        pin=json.loads((Path(root)/'RUN.json').read_text())['pin']['identity']
        if (receipt.get('status')!='PASS' or before.get('status')!='PASS' or
                after.get('status')!='PASS' or not before.get('identity') or before!=after or
                before.get('serving_pin')!=pin):
            raise ValueError('cache byte verification failed or differs before/after run')
        identities.add(before['identity'])
    if len(identities)>1:raise ValueError('cache file bytes differ between compared runs')
    return dict(qualified=not missing,identity=next(iter(identities),None),missing_runs=missing,
        scope='independent pre/post file verification; old runs without receipts remain unknown')


def compare(left, right, absolute_budget, relative_budget=.001, left_repeats=(),right_repeats=()):
    a, ar, left_starts = load_repeated(left,left_repeats)
    b, br, right_starts = load_repeated(right,right_repeats)
    cache_attestation=cache_files_attestation(sorted(set(left_starts+right_starts)))
    for field in ('input_identity',):
        if a[field] != b[field]:
            raise ValueError('different '+field)
    if a['pin']['identity'] != b['pin']['identity']:
        raise ValueError('different pinned model/runtime/cache assets')
    if a.get('live_reference')!=b.get('live_reference'):
        raise ValueError('different actual deployment reference')
    if a['clients'] != b['clients'] or a['deliveries'] != b['deliveries']:
        raise ValueError('different concurrent request plan')
    if bool(a['campaign'].get('xgpu_batch_telemetry'))!=bool(b['campaign'].get('xgpu_batch_telemetry')):
        raise ValueError('different batch telemetry timing arms')
    keys = set(next(iter(ar.values())))
    if any(set(r) != keys for r in br.values()):
        raise ValueError('cross-GPU logical requests differ')
    reports = []
    for key in sorted(keys):
        av, bv = [r[key] for r in ar.values()], [r[key] for r in br.values()]
        first_request = av[0]['request']
        if any(x['request'] != first_request for x in av+bv):
            raise ValueError('noise seed/request mismatch')
        floor_by_gpu = [[errors(x['response']['actions'], y['response']['actions'])
                         for x, y in itertools.combinations(group, 2)] for group in (av, bv)]
        floor = floor_by_gpu[0] + floor_by_gpu[1]
        cross = [errors(x['response']['actions'], y['response']['actions']) for x in av for y in bv]
        f = {k: max(v[k] for v in floor) for k in floor[0]}
        fl, fr = [{k: max(v[k] for v in group) for k in group[0]} for group in floor_by_gpu]
        e = {k: max(v[k] for v in cross) for k in cross[0]}
        decisions = {}
        for field in DECISIONS:
            values = [metadata(x).get(field, '__MISSING__') for x in av+bv]
            decisions[field] = dict(available=all(x is not None and x != '__MISSING__' for x in values),
                cross_agree=all(x == y for x in values[:len(av)] for y in values[len(av):]),
                left_agree=all(x == values[0] for x in values[:len(av)]),
                right_agree=all(x == values[len(av)] for x in values[len(av):]),
                same_gpu_agree=all(x == values[0] for x in values[:len(av)]) and
                               all(x == values[len(av)] for x in values[len(av):]))
        scores = {}
        for field in SCORES:
            vals = [metadata(x).get(field) for x in av+bv]
            if all(isinstance(v,(int,float)) and np.isfinite(v) for v in vals):
                fa = max(vals[:len(av)])-min(vals[:len(av)])
                fb = max(vals[len(av):])-min(vals[len(av):])
                ec = max(abs(x-y) for x in vals[:len(av)] for y in vals[len(av):])
                scores[field] = dict(values=vals, floor=max(fa,fb), cross_error=ec,
                    boundary_band=max(1e-6, 5*max(fa,fb), 2*ec), threshold_margin=None)
                if field in MARGINS:
                    scores[field]['threshold_margin']=vals
                    scores[field]['near_boundary']=any(abs(v)<=scores[field]['boundary_band'] for v in vals)
        sign_equal = all(np.array_equal(np.sign(x['response']['actions'][:,-1]),
                                        np.sign(y['response']['actions'][:,-1])) for x in av for y in bv)
        reports.append(dict(key=key, concurrent_floor=f, cross=e,
            concurrent_floor_left=fl, concurrent_floor_right=fr,
            neighbour_candidate_audit=neighbor_audit([metadata(x) for x in av],[metadata(x) for x in bv]),
            any_segment_head=any(head(metadata(x)) is True for x in av+bv),
            within_floor=e['max_abs'] <= max(1e-6,5*f['max_abs']) and e['relative_l2'] <= max(1e-6,5*f['relative_l2']),
            within_budget=e['max_abs'] <= absolute_budget and e['relative_l2'] <= relative_budget,
            gripper_sign_agree=sign_equal, decisions=decisions, scores=scores))
    agreement = {}
    head_agreement = {}
    for field in DECISIONS:
        observed = [r['decisions'][field] for r in reports if r['decisions'][field]['available']]
        agreement[field] = dict(observed=len(observed), total=len(reports),
            agreement=sum(r['cross_agree'] for r in observed)/len(observed) if observed else None,
            concurrent_floor_agreement=sum(r['same_gpu_agree'] for r in observed)/len(observed) if observed else None)
        at_heads=[r['decisions'][field] for r in reports if r['any_segment_head'] and r['decisions'][field]['available']]
        head_agreement[field]=dict(observed=len(at_heads),
            agreement=sum(r['cross_agree'] for r in at_heads)/len(at_heads) if at_heads else None)
    boundaries={}
    for field in MARGINS:
        strata={}
        for row in reports:
            if not row['any_segment_head']:continue
            score=row['scores'].get(field)
            region='unknown' if score is None else 'near_boundary' if score['near_boundary'] else 'observed_away'
            if field=='neighbour_gap':
                candidate=row['neighbour_candidate_audit']
                region='unknown' if not candidate['qualified'] else 'near_boundary' if candidate['near_boundary'] else 'observed_away'
            cell=strata.setdefault(region,dict(head_pairs=0,max_action_error=0.,decision_flips={}))
            cell['head_pairs']+=1
            cell['max_action_error']=max(cell['max_action_error'],row['cross']['max_abs'])
            for decision in ('source','top1_neighbour','stall_call','corrector_applied','knob_call'):
                d=row['decisions'][decision]
                if d['available'] and not d['cross_agree']:
                    cell['decision_flips'][decision]=cell['decision_flips'].get(decision,0)+1
        boundaries[field]=strata
    # Missing scalar margins and unvisited decisions are coverage gaps, not passes.
    return dict(schema='xgpu.concurrent.comparison.v1', status='MEASURED_NOT_CERTIFIED',
        left=str(left), right=str(right), requests=len(reports),
        budget=dict(max_abs=absolute_budget,relative_l2=relative_budget),
        max_abs=max(r['cross']['max_abs'] for r in reports),
        max_concurrent_floor=max(r['concurrent_floor']['max_abs'] for r in reports),
        outside_budget=sum(not r['within_budget'] for r in reports),
        outside_floor=sum(not r['within_floor'] for r in reports),
        agreement=agreement,head_agreement=head_agreement,boundary_strata=boundaries,
        boundary_caution='neighbour band matches distances by donor identity (sorted gaps can conceal rank swaps); corrector margins describe ramp boundaries, not Boolean applied; unknown support is never away',
        batch_coverage_left=batch_coverage(left),batch_coverage_right=batch_coverage(right),
        native_batch_cap=a['pin'].get('batch'),
        request_coverage_left=request_coverage(ar),request_coverage_right=request_coverage(br),
        clock_audit_left=clock_audit(ar),clock_audit_right=clock_audit(br),
        topology_left=a['campaign']['xgpu_probe'], topology_right=b['campaign']['xgpu_probe'],
        numerics_left=a['campaign'].get('xgpu_numerics',[]),numerics_right=b['campaign'].get('xgpu_numerics',[]),
        harness_identity=dict(left=a.get('harness_files'),right=b.get('harness_files'),
            equal=bool(a.get('harness_files')) and a.get('harness_files')==b.get('harness_files')),
        cold_starts=dict(left=left_starts,right=right_starts,
            floor_includes_independent_cold_starts=len(left_starts)>1 and len(right_starts)>1),
        cache_files_attestation=cache_attestation,
        independent_queries=None, SR_evidence=False,
        limitations=['real-frame stress sequence, not physical rollout',
                    'independent cold-start floor still required' if min(len(left_starts),len(right_starts))<2 else
                    'cold-start floor included; more independent inputs still required',
                    'missing branch/scalar telemetry cannot establish agreement',
                    'action outputs only; per-stage tensors remain a separate diagnostic'], rows=reports)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('left'); p.add_argument('right')
    p.add_argument('--absolute-budget', type=float, required=True)
    p.add_argument('--relative-budget', type=float, default=.001)
    p.add_argument('--left-repeat',action='append',default=[])
    p.add_argument('--right-repeat',action='append',default=[])
    p.add_argument('--output', required=True)
    a=p.parse_args()
    r=compare(a.left,a.right,a.absolute_budget,a.relative_budget,a.left_repeat,a.right_repeat)
    write_json(a.output,r)
    print(json.dumps({k:v for k,v in r.items() if k not in ('rows','agreement')}))


if __name__=='__main__':
    main()
