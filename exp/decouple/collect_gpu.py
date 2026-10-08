"""CPU-only collection and re-analysis of completed benchmark raw artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from exp.decouple.package_config import REPO, write_json


def jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def selected_latencies(rows, accepted, window):
    selected, missing = [], 0
    for row in rows:
        if not row.get('ok'):
            continue
        uid = row.get('uid')
        if uid is None:
            missing += 1
            continue
        if uid not in accepted or row.get('attempt') != accepted[uid].get('attempt', 1):
            continue
        if row.get('started_t') is not None and not (window['start'] <= row['started_t'] <= row['completed_t'] <= window['end']):
            continue
        selected.append(float(row['latency_ms']))
    return selected, missing


def collect(out, *, arm=None, allow_legacy=False):
    from exp.decouple.benchmark_gpu import accepted_cohort
    out = Path(out)
    arm = arm or json.loads((out/'arm.json').read_text())
    window = json.loads((out/'measurement.json').read_text())
    if not window.get('start') or not window.get('end') or window['end'] <= window['start']:
        raise ValueError('incomplete warmup-excluded measurement window')
    if window['manifest_sha256'] != arm['manifest_sha256']:
        raise ValueError('measurement manifest identity differs')
    records = jsonl(out/'journal.jsonl')
    accepted = accepted_cohort(records, set(window['measured_uids']))
    if set(accepted) != set(window['measured_uids']) or len(accepted) != arm['expected_episodes']:
        raise ValueError('accepted episodes differ from the complete sealed B manifest')
    done = json.loads((out/'DONE.json').read_text())
    if done['accepted'] != len(accepted) or done['manifest_sha256'] != arm['manifest_sha256']:
        raise ValueError('DONE receipt differs from the measured cohort')
    rows = jsonl(out/'latencies.jsonl')
    lat, missing_identity = selected_latencies(rows, accepted, window)
    scope = 'benchmark relay ingress-to-response-send; worker network excluded'
    warnings, legacy_relay = [], None
    if not lat and allow_legacy:
        # Old rows have neither identity nor timestamps; their order cannot
        # safely be joined across interleaved connections and retries.
        decisions = [row for path in sorted((out/'runs').rglob('decisions*.jsonl')) for row in jsonl(path)]
        lat = [float(row['infer_ms']) for row in decisions if row.get('ev')=='dec' and row.get('ok') and
               row.get('uid') in accepted and row.get('attempt')==accepted[row['uid']].get('attempt',1)]
        scope = 'resident policy after_infer timing; legacy recovery; excludes relay transport and queue'
        warnings.append('Legacy relay has no UID/attempt/timestamps; measured-cohort relay latency is unrecoverable. Policy timings are reported separately.')
        all_lat = [row['latency_ms'] for row in rows if row.get('ok')]
        if all_lat:
            legacy_relay = dict(count=len(all_lat), p50_ms=float(np.percentile(all_lat,50)),
                                p99_ms=float(np.percentile(all_lat,99)), scope='all relay rows; includes warmup and rejected attempts; not cohort-attributed')
    if not lat:
        raise ValueError(f'no accepted-cohort latencies; {missing_identity} successful relay rows lack UID/attempt')
    sample_path = out/'resource_samples.jsonl'
    samples = jsonl(sample_path) if sample_path.exists() else []
    measured = [row for row in samples if window['start'] <= row['t'] <= window['end']]
    if not measured and not allow_legacy:
        raise ValueError('no durable utilization samples inside the measurement window')
    mean = lambda key:float(np.mean([row[key] for row in measured])) if measured else None
    startup_path = out/'startup.json'
    startup = json.loads(startup_path.read_text()) if startup_path.exists() else {}
    fleet_path = out/'fleet.json'
    fleet = json.loads(fleet_path.read_text()) if fleet_path.exists() else {}
    peak = max([startup.get('gpu_memory_peak_mb',0), *[row['gpu_memory_mb'] for row in samples]]) if samples else None
    lower_bound = max((row['startup_peak_mb'] for row in fleet.get('admissions',[])), default=None)
    if not measured:
        warnings.append('Original sampler was memory-only and lost on collection failure; GPU/CPU means and full-run VRAM peak cannot be reconstructed.')
    elapsed = window['end']-window['start']
    result = dict(name=arm['name'], logical_arm=arm['logical_arm'], model=arm['model'],
                  model_processes=startup.get('model_processes',fleet.get('model_processes',arm.get('model_processes'))),
                  fronts=arm.get('fronts'), split=arm.get('split',False), out=str(out),
                  elapsed_s=elapsed, episodes_completed=len(accepted), episodes_per_min=60*len(accepted)/elapsed,
                  manifest_sha256=arm['manifest_sha256'], warmup_excluded=True, measurement_window=window,
                  episode_failures=sum(not row.get('success',False) for row in accepted.values()),
                  episode_error_records=sum(bool(row.get('error')) and row.get('accepted') is not False and
                                            row['task_uid'] in accepted for row in records),
                  latency_count=len(lat), latency_p50_ms=float(np.percentile(lat,50)),
                  latency_p99_ms=float(np.percentile(lat,99)), latency_scope=scope,
                  latency_missing_identity_rows=missing_identity, legacy_unattributed_relay=legacy_relay,
                  gpu_util_mean_pct=mean('gpu_util_pct'), cpu_proc_mean_pct=mean('cpu_proc_pct'),
                  system_cpu_mean_pct=mean('system_cpu_pct'), gpu_memory_peak_mb=peak,
                  gpu_memory_peak_lower_bound_mb=lower_bound, gpu_memory_budget_mb=arm['gpu_memory_budget_mb'],
                  within_budget=peak <= arm['gpu_memory_budget_mb'] if peak is not None else None,
                  gpu_name=startup.get('gpu_name'), collection_warnings=warnings,
                  resource_sample_count=len(measured), recovery=bool(warnings))
    if peak is not None and peak > arm['gpu_memory_budget_mb']:
        raise ValueError('arm exceeded common GPU memory budget')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm-root', type=Path, required=True, help='read-only raw arm root')
    parser.add_argument('--out', type=Path, default=REPO/'exp/decouple/results/gpu_reanalysis')
    parser.add_argument('--allow-legacy', action='store_true', help='recover policy latency and explicitly mark absent telemetry')
    args = parser.parse_args()
    if args.out.resolve() == args.arm_root.resolve() or not args.out.resolve().is_relative_to(REPO):
        parser.error('write re-analysis only into the worktree, separately from raw input')
    if (args.arm_root/'arm.json').exists():
        arm = None
    else:
        group = json.loads((args.arm_root/'group.json').read_text())
        # Compatibility with the single completed legacy current/1 arm.
        fleet = json.loads((args.arm_root/'fleet.json').read_text())
        if fleet['model_processes'] != 1:
            parser.error('legacy recovery without arm.json currently supports the one-copy control')
        arm = dict(name=args.arm_root.name, logical_arm='current_1', model=group['model'],
                   model_processes=1, fronts=None, split=False,
                   manifest_sha256=group['benchmark_manifest_sha256'],
                   expected_episodes=len(json.loads((args.arm_root/'measurement.json').read_text())['measured_uids']),
                   gpu_memory_budget_mb=fleet['budget_mb'])
    result = collect(args.arm_root, arm=arm, allow_legacy=args.allow_legacy)
    write_json(args.out/'metrics.json',result)
    key = f'{result["model"]}/initial/{result["logical_arm"]}'
    state = dict(package_id='reanalysis_'+args.arm_root.name, status='reanalysed partial',
                 expected_episodes=result['episodes_completed'],completed={key:result}, gates={}, selected={})
    write_json(args.out/'state.json',state)
    from exp.decouple.summarize_gpu import summarize
    summarize(args.out/'state.json',args.out)
    print(json.dumps({key:value for key,value in result.items() if key!='measurement_window'},indent=2))


if __name__ == '__main__':
    main()
