"""CPU-only, read-only source analysis for the frozen Cosmos K2 study.

Writes only a new analysis JSON beside this script. No campaign imports/actions.
"""
from pathlib import Path
import collections
import hashlib
import json

import numpy as np
from scipy.stats import beta, binomtest

OUT = Path(__file__).resolve().parent
BASE = Path('/data/mw_finetune/eval')
SOURCE = BASE / 'formal_cosmos_libero_official2000_20261010_100533_5f558380'
K5 = BASE / 'formal_cosmos_libero_official2000_20261010_130639_a8ca2043'
K2 = BASE / 'hs2_cosmos_h100_k2incl_c16_t107_20261010_115120'
SUITES = ('libero_spatial', 'libero_10')
SEED, BOOT, FAMILY = 20261010, 10000, 24
inputs = {}


def read(path):
    data = path.read_bytes()
    inputs[str(path)] = hashlib.sha256(data).hexdigest()
    return data.decode()


def obj(path):
    return json.loads(read(path))


def rows(path):
    return [json.loads(line) for line in read(path).splitlines() if line.strip()]


def cell(task):
    return (task['experiment'], task['extra']['task_name'], task['orig_init_state_idx'])


def load(run, expected_nfe):
    plan = obj(run / 'EPISODE_PLAN.json')
    tasks = {t['task_uid']: t for t in plan['tasks']}
    assert len(tasks) == plan['expected_episodes']
    journal = [r for r in rows(run / 'driver/journal.jsonl') if r.get('accepted')]
    assert all(type(r['success']) is bool and
               r['status'] == ('done' if r['success'] else 'failed') for r in journal)
    accepted = {r['task_uid']: r for r in journal}
    assert len(accepted) == len(journal) == len(tasks)
    attempts = [r for r in rows(run / 'driver/attempts.jsonl') if r.get('accepted')]
    attempts = {r['task_uid']: r for r in attempts}
    assert set(attempts) == set(accepted)
    result = {}
    by_uid = {}
    for uid, r in accepted.items():
        t = tasks[uid]
        assert uid.rsplit(':', 2)[-2:] == [str(t['task_id']), str(t['orig_init_state_idx'])]
        a = attempts[uid]
        assert a['success'] == r['success'] and a['error'] is None
        data = dict(key=cell(t), task_id=t['task_id'], uid=uid,
                    success=r['success'], attempt=r['attempt'], worker=a['worker_id'],
                    node='timan107' if '_timan107_' in a['worker_id'] else 'timan108',
                    steps=a['n_steps'], duration_s=a['duration_s'],
                    decisions={}, acks={}, timing=None, init_hashes=set())
        assert data['key'] not in result
        result[data['key']] = by_uid[uid] = data
    for row in rows(run / 'driver/per_step.jsonl'):
        if not row.get('accepted'):
            continue
        d = by_uid[row['task_uid']]
        assert row['attempt'] == d['attempt']
        kind = row.get('_kind', 'decision')
        if kind == 'client_timing':
            assert d['timing'] is None
            d['timing'] = row
            continue
        assert row['task_name'] == d['key'][1]
        assert row['original_init_idx'] == d['key'][2]
        assert row['commit_horizon'] == 16 and row['seed'] == 195
        assert row['success'] == d['success']
        d['init_hashes'].add(row['task_data_hash'])
        bucket = d['decisions'] if kind == 'decision' else d['acks']
        assert kind in ('decision', 'execution_ack')
        assert row['decision_seq'] not in bucket
        bucket[row['decision_seq']] = row
        if kind == 'decision':
            assert row['source'] == 'policy'
            assert row['actual_nfe'] == expected_nfe == row['stage_calls'][2]
    for d in result.values():
        assert len(d['init_hashes']) == 1
        assert set(d['decisions']) == set(d['acks']) == set(range(len(d['decisions'])))
        assert sum(r['executed_count'] for r in d['acks'].values()) == d['steps']
        assert d['timing']['infers'] == len(d['decisions'])
        assert d['timing']['steps'] == d['steps'] + 10  # official settle actions
    summary = obj(run / 'SUMMARY.json')
    assert summary['status'] == 'PASS' and summary['completed_episodes'] == len(result)
    for suite, vals in summary['suites'].items():
        selected = [d for k, d in result.items() if k[0] == suite]
        assert len(selected) == vals['completed']
        assert sum(d['success'] for d in selected) == vals['successes']
    assert sum(len(d['decisions']) for d in result.values()) == summary['decisions']
    return result, summary


def cp_lower(g, losses, n, family):
    a = .05 / (2 * family)
    lower_g = 0.0 if g == 0 else beta.ppf(a, g, n-g+1)
    upper_l = 1.0 if losses == n else beta.ppf(1-a, losses+1, n-losses)
    return float(lower_g-upper_l)


def stats(keys, full, reduced):
    keys = sorted(keys)
    n = len(keys)
    g = sum(not full[k]['success'] and reduced[k]['success'] for k in keys)
    l = sum(full[k]['success'] and not reduced[k]['success'] for k in keys)
    # Restart same PCG64 stream for each reported cohort; sorted task/init order.
    # Within each task, the same sampled indices select BOTH outcomes.
    rng = np.random.default_rng(SEED)
    samples = np.zeros(BOOT, dtype=np.int64)
    for task in sorted({k[1] for k in keys}):
        group = [k for k in keys if k[1] == task]
        delta = np.array([int(reduced[k]['success'])-int(full[k]['success']) for k in group])
        idx = rng.integers(len(group), size=(BOOT, len(group)))
        samples += delta[idx].sum(axis=1)
    samples = samples/n
    s = dict(n=n, k5_success=sum(full[k]['success'] for k in keys),
             k2_success=sum(reduced[k]['success'] for k in keys), g=g, losses=l,
             both_failure=sum(not full[k]['success'] and not reduced[k]['success'] for k in keys),
             diff=(g-l)/n, cp_lb_fwer95=cp_lower(g,l,n,FAMILY),
             bootstrap_lb_fwer95=float(np.quantile(samples,.05/FAMILY,method='linear')),
             cp_lb_pointwise95=cp_lower(g,l,n,1),
             bootstrap_lb_pointwise95=float(np.quantile(samples,.05,method='linear')),
             mcnemar_exact_two_sided_p=float(binomtest(g,g+l,.5).pvalue) if g+l else 1.)
    s['ni_pass'] = s['cp_lb_fwer95'] > -.02 and s['bootstrap_lb_fwer95'] > -.02
    return s


def cost(keys, data):
    ds = [data[k] for k in keys]
    dec = [r for d in ds for r in d['decisions'].values()]
    M, T = len(dec), sum(d['steps'] for d in ds)
    nfe = sum(r['actual_nfe'] for r in dec)
    server_ms = sum(r['server_timing']['infer_ms'] for r in dec)
    return dict(episodes=len(ds), M=M, T=T, nfe=nfe,
                nfe_per_decision=nfe/M, decisions_per_episode=M/len(ds),
                nfe_per_episode=nfe/len(ds), nfe_per_control=nfe/T,
                controls_per_episode=T/len(ds),
                server_host_ms_per_call=server_ms/M,
                server_host_ms_per_control=server_ms/T,
                server_host_ms_per_episode=server_ms/len(ds),
                mean_client_ms_per_call=sum(d['timing']['infer_ms'] for d in ds)/M,
                mean_episode_seconds=sum(d['duration_s'] for d in ds)/len(ds),
                pricing_eligible_counts=dict(collections.Counter(str(r['pricing_eligible']) for r in dec)),
                service_ms_present=sum(r.get('service_ms') is not None for r in dec),
                gpu_service_ms_present=sum(r.get('gpu_service_ms') is not None for r in dec),
                queue_wait_ms_present=sum(r.get('queue_wait_ms') is not None for r in dec))


def main():
    seal = obj(K5 / 'RECOVERED_RESULT_SEAL.json')
    verified = []
    # Verify all sealed files, including original journals. No mutations to inputs.
    for prefix, mapping in ((K5,seal['files']), (SOURCE,seal['original_source_pins'])):
        for name, digest in mapping.items():
            data = (prefix/name).read_bytes()
            actual = hashlib.sha256(data).hexdigest()
            assert actual == digest, (str(prefix/name), actual, digest)
            verified.append(str(prefix/name))
    full, full_summary = load(K5,5)
    reduced, reduced_summary = load(K2,1)
    keys = sorted(reduced)
    assert len(keys) == 1000 and set(keys) <= set(full)
    assert all(full[k]['init_hashes'] == reduced[k]['init_hashes'] for k in keys)
    assert all(full[k]['task_id'] == reduced[k]['task_id'] for k in keys)
    resume = rows(K5/'resume_segment/driver/journal.jsonl')
    resume_uids = {r['task_uid'] for r in resume if r.get('accepted')}
    assert len(resume_uids) == 274
    same = [k for k in keys if full[k]['node'] == 'timan107']
    assert {full[k]['uid'] for k in same} == resume_uids
    assert all(d['node'] == 'timan107' for d in reduced.values())
    for suite in SUITES:
        group = [k for k in keys if k[0] == suite]
        assert len(group) == 500
        bytask = collections.defaultdict(set)
        for k in group: bytask[k[1]].add(k[2])
        assert len(bytask) == 10 and all(v == set(range(50)) for v in bytask.values())
    scopes = {suite:[k for k in keys if k[0] == suite] for suite in SUITES}
    scopes.update(same_timan107=same,
                  remaining_726=[k for k in keys if k not in set(same)],
                  libero_10_remaining_226=[k for k in keys if k[0]=='libero_10' and k not in set(same)],
                  both_primary=keys)
    cohorts = {name:dict(sr=stats(ks,full,reduced), k5_cost=cost(ks,full),
                        k2_cost=cost(ks,reduced)) for name,ks in scopes.items()}
    per_task = []
    for task in sorted({k[1] for k in keys}):
        ks = [k for k in keys if k[1] == task]
        per_task.append(dict(suite=ks[0][0],task=task,task_id=full[ks[0]]['task_id'],
                             **stats(ks,full,reduced),
                             resumed_pairs=sum(k in set(same) for k in ks),
                             gain_inits=[k[2] for k in ks if not full[k]['success'] and reduced[k]['success']],
                             loss_inits=[k[2] for k in ks if full[k]['success'] and not reduced[k]['success']]))
    manifests = [obj(p/'assets/model_manifest.json') for p in (K5,K2)]
    a,b = [m['content_digests'] for m in manifests]
    changed = [k for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)]
    assert all(k.startswith('code:r17:') for k in changed), changed
    campaigns = [obj(p/'campaign.json') for p in (SOURCE,K5,K2)]
    server_evidence = []
    for p in (SOURCE,K5,K2):
        attestations = obj(p/'SERVER_ATTESTATIONS.json')
        evidence = []
        for att in attestations:
            metadata = att['metadata']
            parity = metadata['native_parity_self_check']
            assert metadata['batch_policy'] == 'B1 serialized per resident; stock proxy replicas'
            assert parity['gpu'] == 'NVIDIA H100 80GB HBM3' and parity['passed']
            evidence.append(dict(batch_policy=metadata['batch_policy'],gpu=parity['gpu'],
                                 torch=parity['torch'],cuda=parity['cuda']))
        server_evidence.append(evidence)
    read(OUT/'STUDY.md')
    output = dict(method=dict(seed=SEED,bootstrap_draws=BOOT,family=FAMILY,
                             margin=.02,bootstrap='PCG64; sorted task/init; task-stratified paired; linear quantile',
                             cp='CP_low(g,n,.05/(2*m))-CP_high(l,n,.05/(2*m))'),
                  verified_seal_files=len(verified), paired_init_hash_matches=len(keys),
                  runtime_content_changes=changed,
                  server_evidence=server_evidence,
                  worker_nodes=[c['worker_nodes'] for c in campaigns],
                  cohorts=cohorts,tasks=per_task,
                  paired_cells=[dict(suite=k[0],task=k[1],init=k[2],
                                     init_hash=next(iter(full[k]['init_hashes'])),
                                     k5_uid=full[k]['uid'],k2_uid=reduced[k]['uid'],
                                     k5_success=full[k]['success'],k2_success=reduced[k]['success'],
                                     k5_worker=full[k]['worker'],k2_worker=reduced[k]['worker']) for k in keys],
                  source_sha256=inputs)
    (OUT/'COSMOS_K2_ANALYSIS.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(cohorts=cohorts,drop_tasks=[t for t in per_task if t['suite']=='libero_10' and t['diff']<0],
                          verified_seal_files=len(verified)),indent=2))


if __name__ == '__main__':
    main()
