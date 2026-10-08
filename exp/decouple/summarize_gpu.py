"""Summarize complete and interrupted GPU package runs; no GPU/remote access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from exp.decouple.package_config import REPO, write_json


def paired_ratios(state, model):
    results = state.get('completed', {})
    selection = state.get('selected', {}).get(model)
    if not selection:
        return []
    ratios = []
    for repeat in range(3):
        if repeat == 0:
            current = results.get(f'{model}/initial/{selection["current"]}')
            split = results.get(f'{model}/initial/{selection["split"]}')
        else:
            current = results.get(f'{model}/repeat{repeat}/current')
            split = results.get(f'{model}/repeat{repeat}/split')
        if current and split:
            if current['manifest_sha256'] != split['manifest_sha256']:
                raise ValueError('paired manifest mismatch')
            ratios.append(split['episodes_per_min'] / current['episodes_per_min'])
    return ratios


def summarize(state_path, out):
    state_path, out = Path(state_path), Path(out)
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(completed={})
    episodes = state.get('expected_episodes',300)
    smoke = state.get('mode')=='smoke'
    def number(value, digits=1):
        return f'{value:.{digits}f}' if value is not None else 'unavailable'
    lines = ['# CPU/GPU serving comparison', '',
             f"Package: `{state.get('package_id', 'not started')}`. Status: **{state.get('status', 'prepared; no GPU measurements')}**.", '',
             f'All measured cohorts contain {episodes} accepted B episodes. Ten disjoint warmup episodes, model startup, worker startup and worker teardown are excluded. '
             'B=1, k=1, 64 attested timan107 workers, frozen R16 spatial/100 task levels; common allocation budget 46,000 MiB.', '',
             '| Model / arm | Copies / fronts | Episodes/min | GPU % | Host CPU % | Serving CPU (cores) | Peak VRAM MiB | Decision p50 / p99 ms | Rollout failures / errors |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key, r in state.get('completed', {}).items():
        lines.append(f"| {key} | {r['model_processes']} / {r.get('fronts') or '—'} | {r['episodes_per_min']:.2f} | "
                     f"{number(r['gpu_util_mean_pct'])} | {number(r['system_cpu_mean_pct'])} | {number(r['cpu_proc_mean_pct']/100 if r['cpu_proc_mean_pct'] is not None else None,2)} | "
                     f"{number(r['gpu_memory_peak_mb'],0)} | {r['latency_p50_ms']:.2f} / {r['latency_p99_ms']:.2f} | "
                     f"{r.get('episode_failures',0)} / {r.get('episode_error_records',0)} |")
    if not state.get('completed'):
        lines += ['', 'No GPU benchmark has been run. No performance numbers have been inferred as results.']
    for key, row in state.get('completed',{}).items():
        if row.get('collection_warnings'):
            lines += ['', f"{key} latency scope: **{row['latency_scope']}**."]
            lines += [f"- {warning}" for warning in row['collection_warnings']]
            if row.get('gpu_memory_peak_lower_bound_mb') is not None:
                lines += [f"- Startup VRAM lower bound: {row['gpu_memory_peak_lower_bound_mb']:.2f} MiB; full-run peak unavailable."]
            if row.get('legacy_unattributed_relay'):
                relay = row['legacy_unattributed_relay']
                lines += [f"- Unattributed relay p50/p99: {relay['p50_ms']:.2f}/{relay['p99_ms']:.2f} ms ({relay['scope']})."]
    deadline = state.get('deadline')
    if deadline:
        lines += ['', '## Deadline', '',
                  f"Absolute deadline: `{deadline['wall']}`; effective end: `{deadline['effective_hard_end']}`. "
                  f"Active work stops {deadline['shutdown_reserve_s']} seconds earlier for owned-process cleanup."]
    if state.get('deadline_stop_reason'):
        lines += ['', f"`PKG_DEADLINE_STOP`: {state['deadline_stop_reason']}"]
    if state.get('skipped_deadline'):
        lines += ['', 'Arms skipped at admission (no processes or attempt roots started):', '']
        for key, item in state['skipped_deadline'].items():
            estimate = f"{item['estimated_s']:.0f}s" if item['estimated_s'] is not None else 'unavailable'
            lines.append(f"- `{key}`: estimate {estimate}, active time available {item['available_s']:.0f}s; {item['reason']}.")
    comparisons = {}
    lines += ['', '## Paired best current versus best split', '']
    for model in ('groot', 'pi05'):
        ratios = paired_ratios(state, model)
        if not ratios:
            lines.append(f'- {model}: pending.')
            continue
        comparisons[model] = dict(ratios=ratios, median=statistics.median(ratios),
                                  minimum=min(ratios), maximum=max(ratios),
                                  stdev=statistics.stdev(ratios) if len(ratios)>1 else None,
                                  selection=state['selected'][model])
        screened = sum(key.startswith(model + '/initial/') for key in state.get('completed', {}))
        planned = state.get('screening_arms',5)
        screening = f' Best completed screening arms ({screened}/{planned}; partial).' if screened < planned else ''
        lines.append(f"- {model}: split/current ratios {', '.join(f'{r:.3f}×' for r in ratios)}; "
                     f"median **{statistics.median(ratios):.3f}×**, range {min(ratios):.3f}–{max(ratios):.3f}× "
                     f"({len(ratios)}/{1 if smoke else 3} pairs). Selection: `{state['selected'][model]['current']}` versus `{state['selected'][model]['split']}`.{screening}")
    lines += ['', ('Smoke checks collection and launch wiring on 20 episodes per arm; it is not the full throughput experiment.' if smoke else
              'The first pair is the initial screening measurement and has selection bias; the two additional pairs run in alternating orders '
              '(split→current, then current→split). Spread is descriptive, not a confidence interval. Host CPU is whole-machine utilization; '
              'serving cores sum the owned model/front/router/relay process CPU percentages divided by 100. '
              'VRAM peak includes startup and warmup; utilization means use only the measurement window. '
              'Decision latency is relay ingress through response send, excluding the worker network. '
              'Rollout failures are completed episodes with success=false; errors are accepted error records and are never treated as rollout failures.'), '',
              '## Numerical gates', '']
    for name, gate in state.get('gates', {}).items():
        lines.append(f"- {name}: {'PASS (bit exact)' if gate.get('passed') else 'FAIL'}; `{gate.get('path', '')}`.")
    if state.get('failure'):
        lines += ['', '## Last failure', '', f"`{state['failure']}`"]
    lines += ['', '## Artifacts', '']
    for key, value in state.get('completed', {}).items():
        lines.append(f"- {key}: `{value.get('out', '')}` (metrics, samples, journal, latencies, server/worker logs, worker census and sealed group).")
    for attempt in state.get('failed_attempts', []):
        lines.append(f"- Incomplete attempt: `{attempt['out']}` — {attempt['error']}")
    out.mkdir(parents=True, exist_ok=True)
    (out / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    write_json(out / 'comparison.json', dict(package_id=state.get('package_id'), status=state.get('status'),
                                           deadline=state.get('deadline'), skipped_deadline=state.get('skipped_deadline', {}),
                                           arms=state.get('completed', {}), paired=comparisons))
    return out / 'SUMMARY.md'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, default=REPO / 'exp/decouple/results/gpu_comparison/state.json')
    parser.add_argument('--out', type=Path, default=REPO / 'exp/decouple/results/gpu_comparison')
    args = parser.parse_args()
    print(summarize(args.state, args.out))


if __name__ == '__main__':
    main()
