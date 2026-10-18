"""Read-only serial formal campaign census and paired episode audit.

No per_step reads. Only PASS campaigns enter historical outcome comparisons.
UID mismatches are reported separately as descriptive comparisons, never as
same-plan GPU equivalence. Writes are confined to this study.
"""
import collections
import hashlib
import itertools
import json
from pathlib import Path
import time

from paired_sr import paired_interval

HERE = Path(__file__).resolve().parent
ROOT = Path('/data/mw_finetune/eval')


def load(p):
    return json.loads(p.read_text())


def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def science(t):
    e = t['extra']
    return [t.get(k) for k in ('experiment', 'task_id', 'orig_init_state_idx', 'episode_idx', 'phase')] + [
        {k: v for k, v in e.items() if k not in
         ('campaign', 'model_identity', 'layout_identity', 'yaml_identity', 'bundle_generation')}]


def main():
    census = []
    for p in sorted(ROOT.iterdir()):
        if not (p/'campaign.json').is_file():
            continue
        try:
            c = load(p/'campaign.json')
            rec = load(p/'RECEIPT.json') if (p/'RECEIPT.json').exists() else {}
            dr = load(p/'driver/DRIVER_RECEIPT.json') if (p/'driver/DRIVER_RECEIPT.json').exists() else {}
        except (ValueError, OSError):
            continue
        census.append(dict(name=p.name, model=c.get('model'), domain=c.get('domain'),
            node=c.get('server_node'), status=rec.get('status'), driver_status=dr.get('status'),
            all_done=dr.get('all_done'), smoke=c.get('smoke'), library=c.get('library'),
            runtime=c.get('runtime_identity'), plan_exists=(p/'EPISODE_PLAN.json').exists(),
            cache_study=bool(c.get('cache_study'))))
    accepted = [r for r in census if r['status']=='PASS' and r['plan_exists']
                and not r['smoke'] and not r['library'] and r['node'] in ('local','h100')]
    domains = {tuple((r['model'],r['domain'])) for r in accepted
               if any(s['model']==r['model'] and s['domain']==r['domain'] and s['node']!=r['node']
                      for s in accepted)}
    plans = {}; groups = collections.defaultdict(list); io=[]
    # No matching opposite GPU exists for pi05 MT50; avoid its two 160+MB plans.
    for r in accepted:
        if (r['model'],r['domain']) not in domains:
            continue
        p=ROOT/r['name']
        io.append(dict(path=str(p/'EPISODE_PLAN.json'), before=Path('/proc/pressure/io').read_text()))
        plan=load(p/'EPISODE_PLAN.json'); plans[r['name']]=plan
        by=collections.defaultdict(list)
        for t in plan['tasks']:
            by[(t['experiment'],t['yaml_id'])].append(t)
        for (suite,yaml_id),ts in by.items():
            ts.sort(key=lambda t: digest(science(t)))
            # Only scientific task/init/seed/step fields may align descriptive rows.
            key=(r['model'],suite,digest([science(t) for t in ts]))
            groups[key].append(dict(census=r,suite=suite,yaml_id=yaml_id,tasks=ts))
        time.sleep(.05)
    candidates=[]
    for entries in groups.values():
        for a,b in itertools.combinations(entries,2):
            if a['census']['node']==b['census']['node']:
                continue
            if a['census']['node']!='local':a,b=b,a
            # Cache comparisons additionally require identical bundle and YAML bytes.
            if a['census']['cache_study'] or b['census']['cache_study']:
                if a['yaml_id']!=b['yaml_id']:
                    continue
            candidates.append((a,b))
    # Freeze candidate definitions before reading outcomes.
    prereg=[dict(left=a['census']['name'],right=b['census']['name'],suite=a['suite'],
        pairs=len(a['tasks']),alignment='same scientific task/init/seed/steps; UID and model differences audited') for a,b in candidates]
    (HERE/'evidence/existing_pairs_candidates_1018.json').write_text(json.dumps(prereg,indent=2)+'\n')
    outcomes={};errors={}
    for name in sorted({e['census']['name'] for pair in candidates for e in pair}):
        p=ROOT/name/'driver/journal.jsonl';before=Path('/proc/pressure/io').read_text()
        rows={};bad=[];h=hashlib.sha256();initial=p.stat()
        with p.open('rb') as f:
            for line in f:
                h.update(line);r=json.loads(line)
                if r.get('accepted') is not True:continue
                uid=r['task_uid']
                if uid in rows:bad.append('duplicate accepted '+uid)
                if r.get('error'):bad.append('accepted infrastructure error '+uid)
                rows[uid]=r
        if p.stat().st_size!=initial.st_size or p.stat().st_mtime_ns!=initial.st_mtime_ns:
            bad.append('journal changed during read')
        outcomes[name]=rows;errors[name]=bad
        io.append(dict(path=str(p),bytes=initial.st_size,sha256=h.hexdigest(),before=before,
                       after=Path('/proc/pressure/io').read_text()))
    results=[]
    for a,b in candidates:
        an,bn=a['census']['name'],b['census']['name']
        at,bt=a['tasks'],b['tasks']; ar,br=outcomes[an],outcomes[bn]
        issues=errors[an]+errors[bn]
        if any(t['task_uid'] not in ar for t in at) or any(t['task_uid'] not in br for t in bt):
            issues.append('missing planned accepted outcome')
        row=dict(left=an,right=bn,suite=a['suite'],num_steps=at[0]['extra'].get('num_steps'),
            uid_equal=[t['task_uid'] for t in at]==[t['task_uid'] for t in bt],
            identity_differences={k:[at[0]['extra'].get(k),bt[0]['extra'].get(k)] for k in
                ('model_identity','layout_identity','yaml_identity')
                if at[0]['extra'].get(k)!=bt[0]['extra'].get(k)},errors=issues)
        row['strict_same_plan']=row['uid_equal'] and not row['identity_differences']
        row['runtime_equal']=a['census']['runtime']==b['census']['runtime']
        if not issues:
            av=[int(ar[t['task_uid']]['success']) for t in at]
            bv=[int(br[t['task_uid']]['success']) for t in bt]
            row.update(paired_interval(av,bv),left_success=sum(av),right_success=sum(bv))
            per_episode=[dict(left_uid=x['task_uid'],right_uid=y['task_uid'],task_id=x['task_id'],
                init=x['orig_init_state_idx'],left=aa,right=bb,difference=aa-bb)
                for x,y,aa,bb in zip(at,bt,av,bv)]
            row['per_episode_file']=f'evidence/paired_{an}_{bn}_{a["suite"]}_1018.json'
            (HERE/row['per_episode_file']).write_text(json.dumps(per_episode,indent=2)+'\n')
        results.append(row)
    result=dict(schema='xgpu.existing.formal.pairs.v1',created=time.time(),
        census=census,pass_non_smoke_campaigns=len(accepted),
        strict_same_plan_pairs=[r for r in results if r['strict_same_plan']],
        descriptive_pairs=results,io=io,per_step_read=False,
        limitations=['PASS gate excludes failed, pending, partial, library and smoke campaigns',
          'same science with different UIDs/model/runtime is descriptive and cannot isolate GPU architecture'])
    (HERE/'evidence/existing_formal_pairs_1018.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(strict_pairs=len(result['strict_same_plan_pairs']),results=results),indent=2))


if __name__=='__main__':main()
