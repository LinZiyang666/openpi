"""R7 A2 data-quality audit: cost reconciliation and exception-episode audit, per arm.

For every complete arm (DONE marker + summary.json) of the chosen run:
  outcomes   accepted journal recount vs summary complete/success; server 'episode' rows are not used;
  decisions  strict accepted-incarnation stream (a2_common.load_arm) N/V/M vs the ledger-style dedupe vs
             summary.cost_ledger decisions/vision_decisions/misses; client per_step rows (hit_type mix) vs server;
             client_timing 'infers' vs server decisions per episode;
  owner IR   c1*V/N + (1-c1)*M/N (x5/L), pooled over decisions and as an equal-episode mean (the CU/CT budget
             basis), next to the ledger's cost-table IR; SW: sum(owner_cost)/N from per-decision telemetry,
             recomputed from camera counters (.152 full, .055198 wrist, .049890 per completion, .848 call) and the
             two-camera ledger convention .152*V/N;
  calls      calls by os_reason (62 lottery / 63 stall), forced ambiguous LOOKs (look_reason 8), policy-tail rows;
  exceptions client_timing termination_reason for every accepted (task_uid, run_id, attempt); any accepted
             'exception' is a residual; uncovered accepted records; success disagreement; EXC_PURGED /
             ARM_INCOMPLETE / ARM_FAILED / COLLECT_FAILED events from runs/chain.log; local purged_exceptions.jsonl.
Optional --refs: exception coverage of the A/B three-replicate reference journals of the audited cells.

    <A2 prefix> -m exp.offline_search.rounds.r07.analysis_scripts.a2_audit --run eval [--arms ...] [--refs]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from exp.offline_search.rounds.r07.analysis_scripts import a2_common as C


def owner_components(d, model):
    """Actual-work owner cost of one decision from counters (not from the owner_cost field)."""
    c1 = C.C1[model]
    comp = dict(full=0., wrist=0., completion=0., call=0.)
    if d.get('vision'):
        cam = d.get('camera_mode') or 'full'
        if cam == 'wrist_only':
            comp['wrist'] = C.WRIST
            comp['completion'] = C.COMPLETION * int(d.get('camera_completion_calls') or 0)
        elif cam == 'full':
            comp['full'] = c1
        else:
            comp['full'] = c1  # unknown camera string charged full (flagged below)
    if d.get('hit') is False:
        comp['call'] = 1 - c1
    return comp


def audit_arm(run_root, arm, events, purged_n):
    A = C.load_arm(run_root, arm)
    info, qc = A['info'], A['qc']
    model = info['model']
    c1 = C.C1[model]
    summ = json.loads((Path(run_root) / 'runs' / arm / 'summary.json').read_text())
    led = summ.get('cost_ledger') or {}
    L = float((led.get('l_per_request') or {}).get('mean') or 5.)
    eps = A['episodes']
    flags = []

    # ---- episode set / provenance (manifest-bound or plain chain)
    arm_dir = Path(run_root) / 'runs' / arm
    markers = [m.name for m in C.done_markers(run_root, arm)]
    pairs = sorted(C.pair_of(u) for u in A['accepted'])
    launch = arm_dir / 'client' / 'per_step.jsonl.launch.json'
    lj = json.loads(launch.read_text()) if launch.exists() else {}
    prov = dict(done_markers=markers, manifest_json=(arm_dir / 'manifest.json').exists(),
                selection_json=(arm_dir / 'selection.json').exists(), pairs=len(pairs),
                pairs_are_10x50=(pairs == [(t, i) for t in range(10) for i in range(50)]) if not info['profile'] else None,
                apool_rollup_sha256=(lj.get('apool') or {}).get('rollup_sha256'),
                trials_per_task=lj.get('trials_per_task'), init_map_sha256=lj.get('init_map_sha256'))
    if not info['profile'] and not prov['pairs_are_10x50']:
        flags.append('episode set is not the 10x50 test pairs')
    if not prov['manifest_json']:
        flags.append('NOTE no manifest.json/selection.json (plain-chain arm; DONE marker ' + ','.join(markers) + ')')

    # ---- outcomes
    n_acc = len(A['accepted'])
    s_acc = sum(bool(j['success']) for j in A['accepted'].values())
    if summ.get('complete') != n_acc or summ.get('success') != s_acc:
        flags.append(f"summary complete/success {summ.get('complete')}/{summ.get('success')} != journal {n_acc}/{s_acc}")
    if len(eps) != n_acc:
        flags.append(f'{n_acc - len(eps)} accepted episodes without a decision stream')

    # ---- decision counts
    N = V = M = 0
    cams, comps, reasons, looks, srcs = Counter(), Counter(), Counter(), Counter(), Counter()
    owner_sum = comp_sum = 0.
    owner_missing = owner_mismatch = 0
    ep_rows = []
    timing_ms = defaultdict(list)
    for e in eps:
        for d in e['decisions']:
            if d.get('vision'):
                if d.get('q_us') is not None:
                    timing_ms['vision_method_query_ms'].append(d['q_us'] / 1000)
            else:
                if d.get('infer_ms') is not None:
                    timing_ms['blind_server_ms'].append(d['infer_ms'])
                if d.get('blind_prepare_ms') is not None:
                    timing_ms['blind_prepare_ms'].append(d['blind_prepare_ms'])
    for e in eps:
        n = v = m = 0
        cost = 0.
        for d in e['decisions']:
            n += 1
            v += bool(d.get('vision'))
            m += d.get('hit') is False
            srcs[d.get('src')] += 1
            if d.get('vision'):
                cams[d.get('camera_mode') or 'full(implicit)'] += 1
                looks[d.get('look_reason')] += 1
            if d.get('camera_completion_calls'):
                comps['completion_calls'] += int(d['camera_completion_calls'])
            if d.get('hit') is False:
                reasons[C.ex(d, 'os_reason')] += 1
            comp = owner_components(d, model)
            cs = sum(comp.values())
            comp_sum += cs
            if d.get('owner_cost') is not None:
                owner_sum += float(d['owner_cost'])
                if abs(float(d['owner_cost']) - cs) > 1e-8:
                    owner_mismatch += 1
            elif 'camera_mode' in d:
                owner_missing += 1
            cost += (c1 * bool(d.get('vision')) + (1 - c1) * (d.get('hit') is False))
        N += n; V += v; M += m
        ep_rows.append(dict(uid=e['uid'], N=n, V=v, M=m, ir_twocam=cost / n * 5 / L if n else None))
    ledger_counts = dict(N=led.get('decisions'), V=led.get('vision_decisions'), M=led.get('misses'))
    strict = dict(N=N, V=V, M=M)
    if ledger_counts != strict:
        flags.append(f'summary ledger {ledger_counts} != strict stream {strict}')
    if A['ledger_style'] != strict:
        flags.append(f"ledger-style dedupe {A['ledger_style']} != strict stream {strict}")
    if qc.get('duplicate_conflicting') or qc.get('episodes_gapped') or qc.get('episodes_missing_stream'):
        flags.append('stream problems: ' + json.dumps({k: qc[k] for k in ('duplicate_conflicting', 'episodes_gapped',
                                                                            'episodes_missing_stream') if qc.get(k)}))

    # ---- client side
    timing, steps, present = C.client_timing(run_root, arm)
    acc_keys = {(u, j.get('run_id'), int(j.get('attempt', 1) or 1)): j for u, j in A['accepted'].items()}
    client_mix = Counter()
    for (u, rid, att, ht), k in steps.items():
        if (u, rid, att) in acc_keys:
            client_mix[ht] += k
    client_n = sum(client_mix.values())
    if present and client_n != N:
        flags.append(f'client per_step rows {client_n} != server decisions {N}')
    if present and client_mix.get('MISS', 0) != M:
        flags.append(f"client MISS {client_mix.get('MISS', 0)} != server misses {M}")
    infers_mismatch = 0
    by_uid = {e['uid']: e for e in eps}
    term, residual, uncovered, succ_mismatch, multi = Counter(), [], [], 0, 0
    for key, j in acc_keys.items():
        rows = timing.get(key, [])
        if not rows:
            uncovered.append(key)
            continue
        rs = {r.get('termination_reason') for r in rows}
        if len(rs) > 1:
            multi += 1
        r = rows[-1]
        term[r.get('termination_reason')] += 1
        if r.get('termination_reason') == 'exception':
            residual.append(key)
        if bool(r.get('success')) != bool(j['success']):
            succ_mismatch += 1
        e = by_uid.get(key[0])
        if e is not None and r.get('infers') is not None and int(r['infers']) != len(e['decisions']):
            infers_mismatch += 1
    exc_all = [k for k, rows in timing.items() if any(r.get('termination_reason') == 'exception' for r in rows)]
    purged_local = Path(run_root) / 'runs' / arm / 'client' / 'purged_exceptions.jsonl'
    purged_rows = list(C.jsonl(purged_local)) if purged_local.exists() else []
    if residual:
        flags.append(f'{len(residual)} accepted exception episodes survive')
    if present and uncovered:
        flags.append(f'{len(uncovered)} accepted records without client_timing')
    if succ_mismatch:
        flags.append(f'{succ_mismatch} client_timing success != journal')
    if infers_mismatch:
        flags.append(f'{infers_mismatch} episodes client infers != server decisions')

    # ---- owner IR
    pooled_twocam = (c1 * V + (1 - c1) * M) / N * 5 / L if N else None
    ledger_twocam = ((c1 * ledger_counts['V'] + (1 - c1) * ledger_counts['M']) / ledger_counts['N'] * 5 / L
                     if ledger_counts['N'] else None)
    ep_mean = float(np.mean([r['ir_twocam'] for r in ep_rows])) if ep_rows else None
    out = dict(arm=arm, info=info, status='complete', flags=flags, qc=qc, provenance=prov,
               outcomes=dict(accepted=n_acc, success=s_acc, sr=s_acc / n_acc if n_acc else None,
                             summary_complete=summ.get('complete'), summary_success=summ.get('success')),
               counts=dict(strict=strict, ledger_style=A['ledger_style'], summary_ledger=ledger_counts,
                           client_rows=client_n, client_hit_mix=dict(client_mix), sources=dict(srcs),
                           cameras=dict(cams), look_reasons={str(k): v for k, v in looks.items()},
                           call_reasons={str(k): v for k, v in reasons.items()}, **comps),
               ir=dict(owner_twocam_pooled=pooled_twocam, owner_twocam_from_summary_counts=ledger_twocam,
                       owner_twocam_episode_mean=ep_mean, ledger_table_ir=led.get('ir_per_five_controls'),
                       ledger_cost_source=led.get('cost_source'), L=L,
                       v=V / N if N else None, m=M / N if N else None,
                       decisions_per_episode=N / len(eps) if eps else None),
               exceptions=dict(per_step_present=present, accepted_keys=len(acc_keys), covered=len(acc_keys) - len(uncovered),
                               accepted_termination=dict(term), residual_accepted_exceptions=len(residual),
                               residual_keys=residual, exception_attempts_in_per_step=len(exc_all),
                               multi_reason_keys=multi, success_mismatch=succ_mismatch, infers_mismatch=infers_mismatch,
                               purged_local_rows=len(purged_rows),
                               chain_events=dict(events.get(arm, {})), chain_exc_purged=purged_n.get(arm, 0)),
               cpu_ms={k: dict(n=len(v), p50=float(np.percentile(v, 50)), p95=float(np.percentile(v, 95)),
                               mean=float(np.mean(v))) for k, v in timing_ms.items() if v},
               startups=A['startups'])
    if info['family'] == 'wrist' or cams.get('wrist_only'):
        out['ir']['sw'] = dict(owner_cost_sum_over_N=owner_sum / N * 5 / L if N else None,
                               counters_over_N=comp_sum / N * 5 / L if N else None,
                               two_camera_convention=pooled_twocam,
                               owner_cost_missing=owner_missing, owner_cost_mismatch=owner_mismatch,
                               wrist_share_of_looks=cams.get('wrist_only', 0) / V if V else None,
                               completions=comps.get('completion_calls', 0))
        if owner_mismatch or owner_missing:
            flags.append(f'SW owner_cost mismatch {owner_mismatch}, missing {owner_missing}')
    if info['family'] == 'calls':
        rho = float((summ.get('kwargs') or {}).get('rho') or .3)
        out['ir']['calls'] = dict(rho=rho, pooled_minus_rho=pooled_twocam - rho if pooled_twocam is not None else None,
                                  episode_mean_minus_rho=ep_mean - rho if ep_mean is not None else None,
                                  lottery_calls=reasons.get(62., 0), stall_calls=reasons.get(63., 0),
                                  forced_ambiguous_looks=looks.get(8, 0), policy_tail_rows=srcs.get('policy_tail', 0))
    return out, ep_rows


def audit_refs(specs):
    """Exception coverage of reference arms (journal + client_timing only)."""
    res = {}
    for spec in specs:
        run, arm = spec.split(':')
        root = C.ROOT / run
        acc = {}
        for j in C.jsonl(root / 'runs' / arm / 'client/journal.jsonl'):
            if j.get('accepted') and j.get('status') in ('done', 'failed') and not j.get('error'):
                acc[(j['task_uid'], j.get('run_id'), int(j.get('attempt', 1) or 1))] = j
        timing, _, present = C.client_timing(root, arm)
        term, residual, uncovered = Counter(), 0, 0
        for k in acc:
            rows = timing.get(k)
            if not rows:
                uncovered += 1
                continue
            t = rows[-1].get('termination_reason')
            term[t] += 1
            residual += t == 'exception'
        launch = root / 'runs' / arm / 'client' / 'per_step.jsonl.launch.json'
        lj = json.loads(launch.read_text()) if launch.exists() else {}
        res[spec] = dict(apool_rollup_sha256=(lj.get('apool') or {}).get('rollup_sha256'),
                         accepted=len(acc), per_step_present=present, covered=len(acc) - uncovered,
                         termination=dict(term), residual_exceptions=residual)
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', default='eval', help='eval | profile | <run root path>')
    p.add_argument('--arms', nargs='*')
    p.add_argument('--family', nargs='*')
    p.add_argument('--refs', action='store_true')
    p.add_argument('--out', type=Path, default=C.OUT)
    p.add_argument('--detail', type=Path, default=C.DETAIL)
    a = p.parse_args()
    run_root, arms, skipped = C.cli_arms(a)
    events, purged = C.chain_events(run_root)
    reports, eprows = [], []
    for arm in arms:
        r, er = audit_arm(run_root, arm, events, purged)
        reports.append(r)
        eprows += [dict(arm=arm, **x) for x in er]
        print(f"{arm:32s} N={r['counts']['strict']['N']:6d} V={r['counts']['strict']['V']:6d} M={r['counts']['strict']['M']:5d} "
              f"IR2cam={r['ir']['owner_twocam_pooled']:.4f} ledgerIR={r['ir']['ledger_table_ir'] or float('nan'):.4f} "
              f"term={r['exceptions']['accepted_termination']} flags={len(r['flags'])}", flush=True)
        for f in r['flags']:
            print('    FLAG', f)
    refs = {}
    if a.refs:
        specs = []
        from exp.offline_search.rounds.r06.analysis_scripts import common as R6
        for cell in sorted({r['info']['cell'] for r in reports}):
            info = C.parse_arm(f'r7_{cell}_SF1')
            A_, B_ = R6.ab_arms(info['model'], info['r6cell'])
            specs += A_ + B_
        refs = audit_refs(sorted(set(specs)))
        for k, v in refs.items():
            print('ref', k, v)
    tag = run_root.name + ('_subset' if (a.arms or a.family) else '')
    C.write_json(a.out / f'a2_audit_{tag}.json', dict(schema='r7.a2.audit.v1', run=str(run_root), audited=arms,
                                                      skipped=skipped, chain_events={str(k): dict(v) for k, v in events.items()},
                                                      reports=reports, references=refs))
    C.write_csv(a.detail / f'a2_audit_episodes_{tag}.csv.gz', eprows)
    print('skipped (not complete):', skipped)


if __name__ == '__main__':
    main()
