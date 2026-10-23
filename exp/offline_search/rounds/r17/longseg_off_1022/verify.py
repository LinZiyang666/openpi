"""Read back each built long-segment-OFF definition (CPU, read only) and print what the driver will plan.

Counts, over EPISODE_PLAN.json, the per-task commit_horizon / miss_mode / yaml_identity values, and joins each
yaml_identity to its group YAML (fitted/stall digests, guard, periodic_gap). Writes VERIFY.json.

    python -m exp.offline_search.rounds.r17.longseg_off_1022.verify
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import subprocess

import yaml

from .build import HERE, INDEX, read
from .workflow import INDEX as QUEUE


def counts(path, pattern):
    out = subprocess.run(f"grep -o '{pattern}' {path} | sort | uniq -c", shell=True, check=True,
                         capture_output=True, text=True).stdout
    return {line.split(None, 1)[1]: int(line.split(None, 1)[0]) for line in out.strip().splitlines()}


def one(job):
    root = Path(job['run'])
    c, seal, plan = read(root/'campaign.json'), read(root/'assets/CACHE_STUDY.json'), root/'EPISODE_PLAN.json'
    groups = {g['sha256']: g for g in seal['groups']}
    yamls = {}
    for sha, g in groups.items():
        doc = yaml.safe_load((root/'assets'/g['file']).read_text())
        r = doc['recipe']
        yamls[sha] = dict(file=g['file'], suite=g['suite'], size=g['library_size'], group=g['group'],
                          commit_horizon=doc['adapter_binding']['commit_horizon'], fitted_digest=r['fitted_digest'],
                          stall_digest=r['stall_digest'], stall_path=r['stall_path'], miss_mode=r['miss_mode'],
                          guard=r['guard'], periodic_gap=r['periodic_gap'], corrector=r['corrector'],
                          k=r['k'], kref=r['kref'], early=r['early'], serve_loeo=r['serve_loeo'],
                          fit_in_selection=r['fitted_digest'] in c['filter_failed_knob']['selection'])
    commit = counts(plan, '"commit_horizon":[0-9]*')
    miss = counts(plan, '"miss_mode":"[a-z_]*"')
    ids = counts(plan, '"yaml_identity":"[0-9a-f]*"')
    per_yaml = {k.split('"')[3]: v for k, v in ids.items()}
    stalls = Counter()
    for sha, n in per_yaml.items():
        stalls[yamls[sha]['stall_digest']] += n
    return dict(name=job['name'], episodes=job['episodes'],
                server_node=c['server_node'], replicas=c['replicas'], memory_budget_gib=c['memory_budget_gib'],
                max_batch=(c.get('dp_batching') or c.get('fastwam_batching') or {}).get('max_batch', 1),
                mps=bool(c.get('mps')), endpoint=c['endpoint'], driver_port=c['driver_port'],
                server_python=c['server_python'], input_roots=c['longseg_1017_input_roots'],
                plan_commit_horizon=commit, plan_miss_mode=miss,
                plan_tasks_by_yaml=len(per_yaml), plan_tasks=sum(per_yaml.values()),
                plan_yaml_identities_all_in_study=set(per_yaml) <= set(yamls),
                plan_tasks_by_stall_digest=dict(stalls),
                selection_keys=sorted(c['filter_failed_knob']['selection']),
                yamls=sorted(yamls.values(), key=lambda y: (y['suite'], y['size'], y['group'])))


def main():
    jobs = read(INDEX)['jobs']
    if QUEUE.exists():
        names = {j['name'] for j in jobs}
        jobs += [j for u in read(QUEUE)['units'] for j in u['jobs'] if j['name'] not in names]
    result = [one(j) for j in jobs]
    (HERE/'VERIFY.json').write_text(json.dumps(result, indent=1)+'\n')
    for r in result:
        ys = r['yamls']
        print(r['name'], r['plan_tasks'], 'episodes', r['plan_commit_horizon'], r['plan_miss_mode'],
              'C5 YAMLs', sum(y['commit_horizon'] == 5 for y in ys), '/', len(ys),
              'guard', sorted({y['guard'] for y in ys}), 'stall digests', len({y['stall_digest'] for y in ys}),
              'fits in selection', all(y['fit_in_selection'] for y in ys), r['server_node'], 'R', r['replicas'],
              'B', r['max_batch'], r['memory_budget_gib'], 'GiB', r['endpoint'], flush=True)


if __name__ == '__main__':
    main()
