"""Summarize the saved E5 audits; no capture writes or model execution."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import time

import numpy as np

OUT = Path('/tmp/r8cb_E5_debug_architecture')
ROOT = Path('/home/weiland/trace_runs/os_closed_loop/r08_main')


def read_group(name):
    return {p.stem:json.loads(p.read_text()) for p in sorted((OUT/name).glob('*.json'))}


def quant(values):
    v=np.asarray(values,float)
    return dict(n=len(v),mean=float(v.mean()),p50=float(np.median(v)),p95=float(np.percentile(v,95)),max=float(v.max()),total=float(v.sum())) if len(v) else dict(n=0)


def main():
    inv, cap, val, quality = [read_group(k) for k in ('inventory','capacity','validate','quality')]
    cells=defaultdict(list)
    for name,r in inv.items():
        cells[(r['model'],r['suite'])].append(name)
    result=dict(time=time.time(),counts=Counter(),status_counts=defaultdict(Counter),cells={},variants={},
                anomalies=[],validation_counts=Counter(),writer={},sender={},reset_pairs={},quality_counts=Counter(),
                physical_numeric=defaultdict(Counter),physical_checks=defaultdict(Counter),snapshots=Counter(),
                controller_skipped=Counter(),augmentation=defaultdict(Counter),attestation=Counter())
    frozen=json.loads((OUT/'augmentation_at_start.json').read_text())['arms']
    result['augmentation_frozen_arms']=frozen
    result['augmentation_still_done']=[n for n in frozen if (ROOT/'state'/(n+'.AUG_DONE')).exists()]
    result['augmentation_withdrawn']=[n for n in frozen if not (ROOT/'state'/(n+'.AUG_DONE')).exists()]
    result['augmentation_current_done']=[p.name[:-len('.AUG_DONE')] for p in sorted((ROOT/'state').glob('*.AUG_DONE'))]
    result['augmentation_id_complete_audited_arms']=[
        n for n in frozen if n in quality and quality[n]['augmentation'] and all(
            v['rows'] == v['expected'] and v['duplicate_ids'] == 0 and v['missing'] == 0 and v['unexpected'] == 0
            for v in quality[n]['augmentation'].values())]
    writers=[]
    for name,r in inv.items():
        result['counts'].update({k:r[k] for k in ('accepted','unaccepted','snapshots')})
        result['counts'].update(r['counts'])
        result['counts']['controls'] += r['episode_stats']['n_controls']['total']
        result['counts']['settle_controls'] += r['episode_stats']['settle_controls']['total']
        for k,v in r['status_counts'].items():
            result['status_counts'][k].update(v)
        if not r['pair_set_exact'] or r.get('error') or r['read_issues']:
            result['anomalies'].append(dict(arm=name,pair_exact=r['pair_set_exact'],error=r.get('error'),read_issues=r['read_issues']))
        if name in cap:
            writers.extend(dict(arm=name,file=k,**v) for k,v in cap[name]['io_stats'].items())
        if name in val:
            result['validation_counts'][val[name]['capture_status']]+=1
        if name in quality:
            q=quality[name]
            result['quality_counts'].update(q['counts'])
            for k,v in q['physical_numeric'].items():
                result['physical_numeric'][k].update(v)
            result['snapshots'].update(q['snapshots'])
            result['controller_skipped'].update(q['controller_skipped'])
            for row in q['physical_checks']:
                result['physical_checks'][row['check']][row['status']]+=1
                if 'passed' in row:
                    result['physical_checks'][row['check']]['passed' if row['passed'] else 'failed']+=1
            for k,v in q['augmentation'].items():
                result['augmentation'][k].update({x:y for x,y in v.items() if isinstance(y,int)})
                result['augmentation'][k].update(v['finite'])
            for att in q['attestation']:
                result['attestation']['processes']+=1
                result['attestation']['matched_arrays']+=len(att['compared'])
                result['attestation']['uncaptured_arrays']+=len(att['missing'])
                result['attestation']['mismatched_arrays']+=len(att['mismatch'])
                result['attestation']['catalog_hash_matches']+=att['catalog_parquet_hash_ok']
    for cell,names in cells.items():
        reports=[cap[n] for n in names if n in cap]
        episodes=[e['bytes'] for r in reports for e in r['episodes']]
        categories=Counter()
        fields=Counter()
        for r in reports:
            categories.update(r['categories'])
            fields.update({k:v['compressed'] for k,v in r['fields'].items()})
        timings={}
        for key in ('observer_ms','queue_wait_ms','infer_ms','s1_ms','s23_ms','method_ms','pre_ms'):
            n=sum(inv[name]['timings'].get(key,{}).get('n',0) for name in names)
            total=sum(inv[name]['timings'].get(key,{}).get('total',0.) for name in names)
            timings[key]=dict(n=n,mean=total/n if n else None,total=total,
                arm_p95_range=[min(inv[name]['timings'][key]['p95'] for name in names if key in inv[name]['timings']),
                               max(inv[name]['timings'][key]['p95'] for name in names if key in inv[name]['timings'])] if n else None,
                max=max(inv[name]['timings'].get(key,{}).get('max',0.) for name in names))
        result['cells']['_'.join(cell)] = dict(arms=len(names),capacity_arms=len(reports),episodes=500*len(names),
            decisions=sum(inv[n]['counts']['decisions'] for n in names),
            controls=sum(inv[n]['episode_stats']['n_controls']['total'] for n in names),
            source_bytes=sum(r['source_bytes'] for r in reports),
            unaccepted_bytes=sum(r['unaccepted_bytes'] for r in reports),
            shared_metadata_bytes=sum(r['shared_metadata_bytes'] for r in reports),
            episode_bytes=quant(episodes),categories=dict(categories),fields=dict(fields),timings=timings)
    for key in ('accepted','written','raw_bytes','bytes','blocks','queue_wait_ms','serialization_ms','oversized_items'):
        result['writer'][key]=sum(w.get(key,0) for w in writers)
    result['writer'].update(processes=len(writers),drained=sum(w.get('drained') is True for w in writers),
        errors=sum(len(w.get('errors',[])) for w in writers),
        high_water_bytes=quant([w['queue_high_water_bytes'] for w in writers]),
        peak_limit_ratio=max(w['queue_high_water_bytes']/w['queue_limit_bytes'] for w in writers),
        accepted_written_mismatch=[w for w in writers if w['accepted']!=w['written']])
    all_senders=[s for c in cap.values() for s in c['sender_stats'].values()]
    senders=[]
    for name,c in cap.items():
        accepted_keys={e['episode_key'] for e in inv[name]['episodes']}
        senders.extend(s for k,s in c['sender_stats'].items() if k in accepted_keys)
    for key in ('frames_acked','queue_high_water','reconnects'):
        result['sender'][key]=quant([s[key] for s in senders if key in s])
    result['sender'].update(episodes=len(senders),all_receipted_attempts=len(all_senders),spill=sum(bool(s.get('spill')) for s in senders),
        reconnect_more_than_one=sum(s.get('reconnects',0)>1 for s in senders))
    # Outcome-free initial-state matching, including the locked split.
    for suite in ('l10','spatial'):
        groups=defaultdict(list)
        for name,r in inv.items():
            if r['suite'] == suite:
                for ep in r['episodes']:
                    groups[(ep['task_id'],ep['init'])].append((name,ep))
        result['reset_pairs'][suite]=dict(pairs=len(groups),arm_counts=sorted(set(len(v) for v in groups.values())),
            inconsistent_seed=sum(len(set(e['env_seed'] for _,e in v))!=1 for v in groups.values()),
            inconsistent_reset=sum(len(set(e['reset_sha'] for _,e in v))!=1 for v in groups.values()),
            inconsistent_initial=sum(len(set(e['initial_sha'] for _,e in v))!=1 for v in groups.values()),
            inconsistent_settle=sum(len(set(e['settle'] for _,e in v))!=1 for v in groups.values()))
    variant_groups=defaultdict(list)
    for r in inv.values():
        variant_groups[r['variant']].append(r)
    for key, rows in variant_groups.items():
        counts=Counter()
        for r in rows:
            counts.update(r['counts'])
        result['variants'][key]=dict(arms=len(rows),episodes=500*len(rows),counts=dict(counts))
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    profiles={}
    for path in sorted((OUT/'profiles').glob('*/*/*.json')):
        payload=json.loads(path.read_text())
        if 'arms' not in payload:
            continue
        profiles[str(path.relative_to(OUT/'profiles'))]={
            'tool':payload['tool'], 'runtime_seconds':payload['runtime_seconds'],
            'arms':{name:{k:v for k,v in r.items() if k not in ('tables','input_diagnostics')}
                    for name,r in payload['arms'].items()}}
    (OUT/'profile_summary.json').write_text(json.dumps(profiles,indent=2)+'\n')
    with (OUT/'arms.csv').open('w',newline='') as f:
        fields=('arm','model','suite','variant','library_size','accepted','successes','decisions','controls','looks','calls','IR','capture_status','bytes','aug_bytes')
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for name,r in inv.items():
            c=r['counts']
            writer.writerow(dict(**{k:r[k] for k in fields[:7]},decisions=c['decisions'],controls=r['episode_stats']['n_controls']['total'],
                looks=c['live_looks'],calls=c['live_policy_calls'],IR=c['owner_cost']/c['decisions'],
                capture_status=val.get(name,{}).get('capture_status'),bytes=cap.get(name,{}).get('source_bytes'),aug_bytes=cap.get(name,{}).get('categories',{}).get('aug',0)))
    print(json.dumps(dict(inventory=len(inv),capacity=len(cap),validation=len(val),quality=len(quality),validation_counts=result['validation_counts'])))


if __name__ == '__main__':
    main()
