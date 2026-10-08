"""Fixed D5 test-A measurement only; no fit, selection, rollout or remote access.

Reads only the twelve lean arms and the seven prespecified pre-stop supplements.
Missing/incomplete arms stay pending. Updates only the marked A sections of the
two final reports, plus final_test_a_* artifacts. Use the CPU command in handback.
"""
from __future__ import annotations

import csv as csvlib
import datetime
from collections import defaultdict
from zoneinfo import ZoneInfo

from final_common import *

SUPPLEMENT = (
    'r13_groot_l10_pure_c5_rep0',
    'r13_groot_l10_200_distance_feedback_ir40_rep0',
    'r13_groot_l10_200_periodic_ir40_rep0',
    'r13_groot_l10_500_distance_feedback_ir25_rep0',
    'r13_groot_l10_500_periodic_ir25_rep0',
    'r13_groot_l10_500_distance_feedback_ir32_rep0',
    'r13_groot_l10_500_periodic_ir32_rep0',
)
CELLS = tuple(f'{m}_{s}_50' for m in ('groot', 'pi05') for s in ('l10', 'spatial'))
METHODS = ('distance_feedback', 'periodic', 'off')
LEAN = tuple(f'r13_{cell}_{method}' + ('_ir32' if method != 'off' else '') + '_rep0'
             for cell in CELLS for method in METHODS)
STOP = datetime.datetime(2026, 10, 4, 20, 38, tzinfo=ZoneInfo('America/Chicago')).timestamp()


def local_time(ts):
    return datetime.datetime.fromtimestamp(ts, ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M:%S %Z')


def measure(spec, supplementary=False):
    run = RUNS / ('r13_test_' + spec['batch']) / 'runs' / spec['arm']
    assert spec['arm'] in LEAN + SUPPLEMENT
    result = dict(arm=spec['arm'], cell=spec['cell'], model=spec['model'], method=spec['method'],
                  target=float(spec['target']) if spec['target'] else None, rep=int(spec['replicate']),
                  expected_episodes=int(spec['episodes']), planned_host=spec['host'], run=str(run),
                  role='supplementary' if supplementary else 'lean', status='pending')
    summary_file = run / 'summary.json'
    if not summary_file.exists():
        result['reason'] = 'No completed summary; partial attempts not analyzed.'
        return result, [], []
    s = read(summary_file)
    if s.get('complete') != result['expected_episodes']:
        result['reason'] = f"Summary has {s.get('complete')} / {result['expected_episodes']} completed episodes."
        return result, [], []
    needed = [run / 'client/journal.jsonl', run / 'manifest.json']
    if not all(p.exists() for p in needed):
        result['reason'] = 'Complete summary exists; journal/manifest not yet available locally.'
        return result, [], []
    assert s['arm'] == spec['arm'] and s['model'] == spec['model'] and s['init_pool'] == spec['init_pool'] == 'A'
    manifest = {tuple(v) for v in read(run / 'manifest.json')}
    journal = list(jsonlines(run / 'client/journal.jsonl'))
    accepted = [j for j in journal if j.get('accepted') and j.get('status') in ('done', 'failed') and not j.get('error')]
    accepted_keys = {(j['task_uid'], j['attempt']): j for j in accepted}
    assert len(accepted_keys) == len(accepted) == len(manifest) == result['expected_episodes']
    identities = [tuple(map(int, j['task_uid'].split(':')[-2:])) for j in accepted]
    assert len(set(identities)) == len(identities) and set(identities) == manifest
    assert all(j['init_pool'] == 'A' for j in accepted)
    assert sum(bool(j['success']) for j in accepted) == s['success']
    assert abs(s['sr'] - s['success']/s['complete']) < 1e-12
    start = min(j['ts'] - j['duration_s'] for j in accepted)
    end = max(j['ts'] for j in accepted)
    if supplementary:
        assert end <= STOP, 'Supplement finished after D5 cutoff'
    episodes = [dict(arm=spec['arm'], cell=spec['cell'], suite=spec['suite'], task=task, init=init,
                     method=spec['method'], success=int(j['success']), attempt=j['attempt'])
                for j, (task, init) in zip(accepted, identities)]
    led = s['cost_ledger']
    N, V, M = (led[k] for k in ('decisions', 'vision_decisions', 'misses'))
    cv, cm = COSTS[spec['model']]
    ir = (cv*V + cm*M)/N
    result.update(N=N, V=V, M=M, ir=ir, sr=s['success']/s['complete'], successes=s['success'],
                  episodes=len(accepted), rejected_or_incomplete_attempts=len(journal)-len(accepted),
                  start_cdt=local_time(start), end_cdt=local_time(end), start_ts=start, end_ts=end)
    if supplementary:
        # Supplementary outcomes/IR are descriptive only; never feed lean estimates.
        result.update(status='complete', cadence_status='not requested for supplementary arms',
                      validation='summary ledger + accepted journal + full init manifest')
        return result, episodes, []
    logs = sorted(run.glob('server_*/decisions*.jsonl'))
    if not logs:
        result['reason'] = 'Complete summary exists; decision logs not yet available locally.'
        return result, [], []
    decisions, terminals, startups = defaultdict(list), defaultdict(list), []
    excluded = 0
    for path in logs:
        for r in jsonlines(path):
            if r['ev'] == 'startup':
                startups.append(r)
            if r['ev'] not in ('dec', 'episode'):
                continue
            key = (r['uid'], r['attempt'])
            if key not in accepted_keys:
                excluded += int(r['ev'] == 'dec')
                continue
            if r['ev'] == 'dec':
                # Keep only fields needed for measurement, not action arrays.
                decisions[key].append({k: r[k] for k in ('step', 'hit', 'vision', 'task_id', 'init')})
            else:
                terminals[key].append(r)
    observed = [sum(len(v) for v in decisions.values()),
                sum(r['vision'] for v in decisions.values() for r in v),
                sum(not r['hit'] for v in decisions.values() for r in v)]
    if any(x < y for x, y in zip(observed, (N, V, M))):
        result['reason'] = f'Incomplete local decision-log sync: observed N/V/M={observed}; summary={[N,V,M]}.'
        return result, [], []
    assert observed == [N, V, M], (spec['arm'], observed, [N, V, M])
    assert decisions.keys() == accepted_keys.keys()
    if set(terminals) != set(accepted_keys):
        result['reason'] = 'Complete decisions exist; accepted episode terminal records not yet fully synced.'
        return result, [], []
    gaps, control_gaps, terminal_gaps, first_calls = [], [], [], []
    for key, rr in decisions.items():
        rr.sort(key=lambda r: r['step'])
        assert [r['step'] for r in rr] == list(range(len(rr))), key
        task, init = map(int, key[0].split(':')[-2:])
        assert all((r['task_id'], r['init']) == (task, init) for r in rr)
        look = 0
        last_look = last_step = None
        for r in rr:
            call = not r['hit']
            assert r['vision'] or not call
            if not r['vision']:
                continue
            if call:
                if last_look is not None:
                    gaps.append(look - last_look)
                    control_gaps.append(5*(r['step'] - last_step))
                else:
                    first_calls.append(5*r['step'])
                last_look, last_step = look, r['step']
            look += 1
        if last_look is not None:
            terminal_gaps.append(look - 1 - last_look)
        assert len(terminals[key]) == 1
        assert terminals[key][0]['success'] == accepted_keys[key]['success']
        assert terminals[key][0]['n_decisions'] == len(rr)
    result.update(status='complete', observed_hosts=sorted({r['host'] for r in startups}),
                  excluded_decisions=excluded, gap=distribution(gaps), control_gap=distribution(control_gaps),
                  startup_calls=len(first_calls), first_call_controls=distribution(first_calls),
                  terminal_censored_looks=distribution(terminal_gaps),
                  validation='accepted journal + init manifest + summary/decision N/V/M + contiguous episode steps')
    meta = dict(cell=spec['cell'], method=spec['method'], target=result['target'], rep=0, kind='any')
    cdf = cdf_rows(gaps, dict(meta, unit='real_looks')) + cdf_rows(control_gaps, dict(meta, unit='controls'))
    return result, episodes, cdf


def main():
    before = verify_frozen()
    with record(HERE / 'test_grid.csv').open() as f:
        specs = {r['arm']: r for r in csvlib.DictReader(f)}
    assert len(LEAN) == 12 and len(SUPPLEMENT) == 7
    assert set(LEAN).isdisjoint(SUPPLEMENT)
    assert set(LEAN + SUPPLEMENT) <= specs.keys()
    results, eps, cdfs = [], [], []
    for name in LEAN + SUPPLEMENT:
        r, e, c = measure(specs[name], name in SUPPLEMENT)
        results.append(r)
        if name in LEAN and r['status'] == 'complete':
            eps.extend(e)
            cdfs.extend(c)
    lean = [r for r in results if r['role'] == 'lean']
    supp = [r for r in results if r['role'] == 'supplementary']
    pairs, per_cell = [], []
    for cell in CELLS:
        arms = {r['method']: r for r in lean if r['cell'] == cell and r['status'] == 'complete'}
        row = dict(cell=cell, status='pending')
        if {'distance_feedback', 'periodic'} <= arms.keys():
            de = {(e['task'], e['init']): e for e in eps if e['cell'] == cell and e['method'] == 'distance_feedback'}
            pe = {(e['task'], e['init']): e for e in eps if e['cell'] == cell and e['method'] == 'periodic'}
            assert de.keys() == pe.keys()
            local = [dict(cell=cell, suite=cell.split('_')[1], task=t, init=i,
                          success_d=de[t, i]['success'], success_p=pe[t, i]['success']) for t, i in sorted(de)]
            pairs.extend(local)
            delta = arms['distance_feedback']['ir'] - arms['periodic']['ir']
            row.update(status='complete', delta_ir=delta, matched=abs(delta) <= .01, **paired_stats(local))
        per_cell.append(row)
    # Never pool whichever cells happen to finish first or exclude a failing budget.
    all_pairs_ready = all(r['status'] == 'complete' for r in per_cell)
    pooled = paired_stats(pairs) if all_pairs_ready else None
    complete = sum(r['status'] == 'complete' for r in lean)
    status = 'COMPLETE' if complete == len(LEAN) else 'PENDING'
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    arm_table = table(['Fixed cell', 'Rule', 'Status', 'IR', 'SR', 'Gap mean [p10/50/90]'], [
        [cell_label(r['cell']), r['method'], r['status'],
         f"{r['ir']:.4f}" if r['status'] == 'complete' else '—',
         pct(r['sr']) if r['status'] == 'complete' else '—', gap_text(r.get('gap'))] for r in lean])
    pair_table = table(['Cell', 'Pairs', 'D−P IR', 'Matched', 'D−P SR pp [95% CI]'], [
        [cell_label(r['cell']), r.get('pairs', 'pending'),
         f"{r['delta_ir']:+.4f}" if r['status'] == 'complete' else '—',
         ('yes' if r['matched'] else 'NO') if r['status'] == 'complete' else '—',
         pp_ci(r) if r['status'] == 'complete' else 'pending'] for r in per_cell])
    supplemental_table = table(['Prespecified supplementary arm', 'Episodes', 'IR', 'SR'], [
        [r['arm'], r.get('episodes', 'pending'), f"{r['ir']:.4f}" if r['status'] == 'complete' else '—',
         pct(r['sr']) if r['status'] == 'complete' else '—'] for r in supp])
    pooled_text = (f"Pooled fixed-set paired D−P: {pp_ci(pooled)} pp ({pooled['pairs']:,} episode pairs; "
                   f"{pooled['original_init_clusters']} suite/task/init clusters)." if pooled else
                   'Pooled fixed-set paired D−P SR: **PENDING** until both methods are complete in every fixed cell; no available-cell substitution.')
    if pooled:
        pooled_text += ' Nominal-dose contrast; a matched-budget interpretation requires every cell to satisfy |ΔIR|≤.01. Failed cells remain in the fixed pool and are flagged.'
    section = f"""**D5 lean confirmation: {status} ({complete}/{len(LEAN)} arms validated at {now}).** Fixed by rule before these outcomes: four size-50 cells, target .32, rep0, distance feedback / periodic / off. No tuning, selection, expansion, or stopping rule is inferred from A.

Scheduled by the owner: GR00T on the local 4090 during the night of October 4; π0.5 on H100 after approximately 07:30–08:00 CDT October 5. This schedule is not a claim of observed execution. Missing local summaries/logs remain pending; completed-arm host/timing evidence is in `final_test_a_numbers.json`.

{arm_table}

{pooled_text}

{pair_table}

IR uses the frozen owner cost coefficients and pooled accepted N/V/M, not current latency-derived summary IR. Gaps are between successive actual calls in real looks (guards included), with startup and terminal censoring separate; the figure CSV also has elapsed control gaps. SR pairs use the same suite/task/init; {BOOT:,} paired cluster resamples within suite/task, seed {SEED}, jointly carrying both models. The fixed-set pool weights cells equally because every complete arm uses the same full init count. A single rep0 cannot supply leave-one-block sensitivity or the superseded multi-block precision claim. Off is contextual and never used to choose a method.

**Pre-D5 supplementary measurements — all {len(SUPPLEMENT)} arms, descriptive only.** Their accepted episodes finished before the stop; nothing is selected, promoted into the lean set, or used for tuning. The previously unspecified seventh arm is GR00T L10-500 periodic .32 rep0. The aborted GR00T L10-50 periodic .25 arm is discarded and is never read by this script.

{supplemental_table}

Re-run `final_test_a.py` with the CPU command in `final/HANDBACK.md` after local summaries, journals, manifests and decision logs are available. It replaces only this marked section in ANALYSIS/REPORT, writes `final_test_a_numbers.json`, arm/pair tables and gap CDF CSVs, and refuses frozen-file writes. The B conclusion and all controller artifacts are unchanged.
"""
    # Avoid a stale owner/top-level sentence: status lives only in these markers.
    for name in ('ANALYSIS.md', 'REPORT.md'):
        replace_section(name, section)
    columns = ['arm', 'cell', 'method', 'role', 'status', 'reason', 'target', 'ir', 'sr', 'N', 'V', 'M',
               'episodes', 'planned_host', 'observed_hosts', 'start_cdt', 'end_cdt']
    csv('final_test_a_arms.csv', [{k: r.get(k) for k in columns} for r in results])
    if cdfs:
        csv('final_test_a_gap_cdf.csv', cdfs)
    else:
        write('final_test_a_gap_cdf.csv', 'cell,method,target,rep,kind,unit,gap,count,total,cdf\n')
    if pairs:
        csv('final_test_a_sr_pairs.csv', pairs)
    else:
        write('final_test_a_sr_pairs.csv', 'cell,suite,task,init,success_d,success_p\n')
    csv('final_test_a_sr_summary.csv', per_cell + ([dict(cell='pooled', status='complete', **pooled)] if pooled else []))
    result = dict(timestamp_utc=now, status=status, complete_lean_arms=complete, lean_expected=list(LEAN),
                  supplementary_expected=list(SUPPLEMENT), boundary='Test-A measurement only; fixed D5 arms; no selection/tuning',
                  arms=results, per_cell=per_cell, pooled=pooled,
                  frozen_before=before, frozen_after=verify_frozen(), input_files=INPUTS,
                  scripts={n: sha(HERE / n) for n in ('final_common.py', 'final_test_a.py')})
    dump('final_test_a_numbers.json', result)
    print(json.dumps(dict(status=status, completed_lean=complete, supplementary=len(supp), pooled=pooled)))


if __name__ == '__main__':
    main()
