"""Append controller follow-up evidence without replacing the initial report."""
import csv
import hashlib
import json
from pathlib import Path

from .build_fast_handback import MODULE, PREFIX, table
from .verify_fast import CELLS, HERE, SCRATCH

OUT = SCRATCH/'controller_followup'
MARKER = '## 8. Controller path (follow-up)'


def read(name):
    return json.loads((OUT/name).read_text())


def main():
    parity = {c: read(f'parity_{c}.json') for c in CELLS}
    calibration = {c: read(f'calibration_{c}.json') for c in CELLS}
    timings = {c: {r['input']: r for r in read(f'timing_{c}.json')} for c in CELLS}
    tests = read('test_results_v2b.json')
    packaging = read('packaging_tests_v2b.json')
    cleanup = read('test_cleanup.json')
    replays = [read(f'replay_test_{c}_{case}.json') for c in ('pi05_l10_50', 'groot_l10_50')
               for case in ('A', 'floor', 'uniform', 'R', 'R_reverse', 'R45', 'stall', 'stall30')]
    assert all(r['differences'] == r['maximum_absolute_float_difference'] == 0 for r in parity.values())
    assert all(r['calibration_byte_equal'] and r['cost_replay_byte_equal'] for r in calibration.values())
    assert tests['status'] == packaging['status'] == 'PASS'
    assert all(r['status'] == 'PASS' for r in replays)
    assert cleanup['tests'] == 22 and cleanup['errors'] == cleanup['failures'] == 0
    assert cleanup['before'] == cleanup['after'] and cleanup['created_dirs_retained'] == 0
    counts = [[c, parity[c]['counts']['bval']['all_fresh_actual_recorded_anchors'],
               parity[c]['counts']['bval']['all_fresh_anchors'],
               parity[c]['counts']['bval']['C_cadence_anchors'],
               parity[c]['counts']['pilot']['pilot_anchors_anchors'], 0] for c in CELLS]
    totals = [sum(r[i] for r in counts) for i in range(1, 5)]
    compared = sum(totals[1:])
    assert totals == [1817, 3606, 1867, 3406] and compared == 8879
    assert all(timings[c][n]['anchors'] == parity[c]['counts']['bval']['C_cadence_anchors']
               for c in CELLS for n in ('raw', 'code'))
    early = sum(r['counts'][cohort]['early_codes'] for r in parity.values() for cohort in ('bval', 'pilot'))
    main_codes = sum(r['counts'][cohort]['main_codes'] for r in parity.values() for cohort in ('bval', 'pilot'))
    assert early == 320 and main_codes == 8559
    rows = [timings[c][n] for c in CELLS for n in ('raw', 'code')]
    with (HERE/'timing_controller_fast.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    samples = 0
    with (HERE/'timing_controller_fast_samples.csv').open('w', newline='') as f:
        writer = None
        for c in CELLS:
            for n in ('raw', 'code'):
                with (OUT/f'timing_{n}_{c}.csv').open() as source:
                    for r in csv.DictReader(source):
                        row = dict(cell=c, input=n, **r)
                        if writer is None:
                            writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader()
                        writer.writerow(row); samples += 1
    assert samples == 3734
    method_path = HERE.parents[1]/'ideation_Q1/method_c/methods.py'
    evidence = dict(parity=parity, calibration=calibration, timing=timings,
        tests=tests, packaging=packaging, replays=replays, test_cleanup=cleanup,
        compared_anchors=compared, early_codes=early, main_codes=main_codes,
        before_controller_source=str(OUT/'methods_before.py'),
        before_controller_sha256=hashlib.sha256((OUT/'methods_before.py').read_bytes()).hexdigest(),
        after_controller_sha256=hashlib.sha256(method_path.read_bytes()).hexdigest())
    (HERE/'verification_controller_fast.json').write_text(json.dumps(evidence, indent=2, allow_nan=False)+'\n')
    speedups = [timings[c]['raw']['incremental_controller_cpu_ms_median'] /
                timings[c]['code']['incremental_controller_cpu_ms_median'] for c in CELLS]
    def span(name, field, suffix=None):
        values = [timings[c][name][field] for c in CELLS if suffix is None or c.endswith(suffix)]
        return f'{min(values):.3f}–{max(values):.3f}'
    def timing_table(stem):
        keys = [f'{stem}_{clock}_ms_{stat}' for clock, stat in
                [('cpu', 'median'), ('cpu', 'p99'), ('wall', 'median'), ('wall', 'p99')]]
        return table(['Cell', 'Anchors / pass', 'CPU median ms raw → code', 'CPU p99 ms raw → code',
                      'Wall median ms raw → code', 'Wall p99 ms raw → code'],
            [[c, timings[c]['raw']['anchors'],
              *[f'{timings[c]["raw"][k]:.3f} → {timings[c]["code"][k]:.3f}' for k in keys]] for c in CELLS])
    lines = [MARKER, '',
        '**The deployed C consumer now supplies A\'s already computed metric code.** '
        f'All {compared:,} before/after controller anchor comparisons have zero differences '
        f'(maximum absolute float difference 0), including {early} early and {main_codes:,} main codes. '
        'All eight production calibration and cost-replay JSONs are byte-equal after another production rerun. '
        'The 22 unit tests, 16 existing plugin replays and existing result/packaging checks pass.', '',
        f'On real B-val inputs following C call/LOOK cadence, the **end-to-end incremental C/stall CPU '
        f'cost per fresh observation** falls from {span("raw", "incremental_controller_cpu_ms_median")} ms '
        f'to {span("code", "incremental_controller_cpu_ms_median")} ms across cells '
        f'({min(speedups):.2f}–{max(speedups):.2f}× median speedup). '
        'This includes tracker inference, status, code handoff and unchanged C decision bookkeeping; '
        'the original A work is excluded as described below. The float64 distance/DP/reference path is unchanged.', '',
        '**Consumer change.** Only `ideation_Q1/method_c/methods.py` changes outside `stall/`. '
        '`_query_with_metric_code` calls the original A `query`/`_dist` once and captures the local code '
        'at the matrix multiplication that consumes it. `_MetricCodeMatrix.__matmul__` forwards to the '
        'original ndarray with the same operand. `_metric_distance` uses shallow method/task facades '
        'so fitted matrices and the real task dictionary are unchanged; A query state updates remain '
        'on the real base. The temporary `_dist` hook is restored in `finally`, including instance '
        'overrides and failure paths. No projection or retrieval arithmetic is copied or recomputed.', '',
        'The tracker receives `{"metric_code": code, "metric": regime}` after successful A retrieval, '
        'with `early` exactly when `step == 0 and base.early`, otherwise `main`. Main capture uses `Z`; '
        'early capture uses `Z0`, or `A0` for the unstored early transform, before its code is transformed '
        'into main coordinates. Every fresh `query`, including an extra LOOK, uses this path and the '
        'same `step * block_controls` timestamp. No-stall stubs retain their original raw-query protocol. '
        'A failed query still sends the raw observation to preserve invalid-window clearing before '
        'propagating the failure. `stall_bridge.py`, public interfaces, controller kwargs, log fields, '
        'budget rules, artifacts, specs, arms and fitted models are unchanged.', '',
        '**Controller equivalence on real inputs.** The frozen before-consumer source is '
        '`/tmp/codex_stall_fast/controller_followup/methods_before.py`; both versions use the same '
        'current exact tracker and actual production controller/calibration arguments. Input vectors '
        'and executed chunks come from the tables\' `absolute_input_archive` NPZs. Recorded A history '
        'and preceding executed action are supplied; no Y field is read. All 80 B-val episodes are '
        'checked twice: all 3,606 recorded/diagnostic observations (covering every one of the 1,817 '
        'actual calibration anchors), then 1,867 fresh observations at C cadence, advancing one request '
        'after an extra LOOK and two otherwise. Pilot sampling checks every actual anchor in the first '
        'episode of each task and A/B cohort: 160 episodes / 3,406 anchors.', '',
        table(['Cell', 'Actual B-val anchors covered', 'All B-val observation probes', 'C-cadence probes',
               'Pilot anchors', 'Differences'], counts + [['TOTAL', *totals, 0]]),
        'At every probe, raw projection, recorded A code and code delivered through the actual new '
        'controller are equal by packed float bytes after the public encode boundary, with the same '
        'regime. The entire stall status matches, including all float diagnostics, reference windows, '
        'W/K/counts/span and state. All retrieval result arrays/scalars/extras and C internal/log fields '
        'match: Ehat, nominal p, p, coin, call, stall call, cooldown, scheduled extra LOOK, control/anchor '
        'bookkeeping and the policy-tail gate. These streams exercise 4,557 calls, 1,077 cooldown '
        'anchors and 337 scheduled LOOK flags. The unchanged plugin replay separately verifies 35 '
        'realized extra LOOKs and their next fresh observations on both models. No float tolerance '
        'is applied to before/after equality. Original §2 template/path/top-K verification remains valid.', '',
        '**Production calibration rerun.** The unchanged production fitter is invoked again with '
        'the same original bank, B-val tables, manifest, reset-attestation telemetry, c1, targets '
        'and original stall artifacts described in §3. Fresh outputs are under '
        '`/tmp/codex_stall_fast/controller_followup/calibration_verified/<cell>/`. The output '
        'allowlist alone is redirected in memory; existing-output refusal remains in effect. '
        'Full bytes of both JSONs equal production, including all λ/d solutions, floors, ceilings, '
        'feasibility and stall fingerprints; even timestamps and paths require no exclusions.', '',
        table(['Cell', 'Solutions', 'calibration.json byte-equal', 'cost_replay.json byte-equal',
               'Stall fingerprint'], [[c, calibration[c]['solution_count'], 'yes', 'yes', 'unchanged'] for c in CELLS]),
        '**Timing.** Both passes run in one CPU-only process, one BLAS/OpenMP thread, affinity '
        '22–25,66–69, with no concurrent analysis Python job. They preload the same raw B-val '
        'NPZ vectors outside timers and follow the same C cadence, 80 episodes / 1,867 observations '
        'per pass, including warmup and first-use caches. Timers are `process_time_ns` and '
        '`perf_counter_ns`; medians/p99 use pandas as in §5. The before-consumer passes raw q, '
        'the after-consumer passes A code; both use the same exact fast tracker.', '',
        'Observe-only timing (raw projection included before; code validation/copy included after):', '',
        timing_table('observe'),
        'End-to-end incremental controller cost over the original A work:', '',
        timing_table('incremental_controller'),
        'The latter is measured as full C query minus its nested base query, adding back the '
        'code-capture facade preparation and operand interception that run inside the base query. '
        'Thus it includes the entire code handoff, tracker/status, Ehat/coin/cooldown/LOOK and result/log '
        'bookkeeping, while subtracting only original A projection/retrieval/synthesis. Nested timers '
        'add overhead, so this is a conservative instrumented host estimate. Per-cell full C query '
        '(including A) and capture-overhead statistics are retained in the CSV and evidence JSON. '
        'The all-observation diagnostic timing pass is retained separately under scratch '
        '`diagnostic_timing/`; headline numbers use the C-cadence pass above.', '',
        f'Afterward, observe-only CPU medians are {span("code", "observe_cpu_ms_median", "_50")} ms '
        f'(50) and {span("code", "observe_cpu_ms_median", "_500")} ms (500). '
        'The original ≤0.1 / ≤0.3 ms targets remain missed. No precision, summation order, '
        'quantile or tie rule was relaxed. The remaining float64 norm/DP/reference work is described '
        'in §5. A\'s float32 retrieval distances are still not reused (§6). ', '',
        '**Tests and cleanup.** All 22 unit tests pass: 13 existing stall tests, seven exact '
        'tracker tests and two consumer-capture tests covering early/main, absent Z0, original '
        'operand dtype/bytes, task identity and hook restoration on failure. `test_stall.py` now '
        'uses `tempfile.mkdtemp(prefix="_test_model_")` with no hard-coded directory, and '
        '`self.addCleanup(shutil.rmtree, path)` removes exactly its own directory. The final combined '
        'test run proves the default `/tmp/_test_model_*` names are identical before and after: '
        '`_test_model_1gm0whs2` and `_test_model_o17uckyv` remain untouched. The pre-existing source '
        '`_test_model_5beh_ykn/` remains untouched.', '',
        'The existing method_c assertions run unchanged against new replay products: 16 cases on '
        'π0.5/GR00T L10-50, 6,208 decisions, 3,126 fresh anchors; 2,738 C anchors match the budget DAG. '
        'Served NPZ arrays (actions/source/vision/HIT) are byte-equal to the retained v2b products. '
        '`check_results` passes eight-cell 1,000-seed rate checks, 192 exhaustive budget values, '
        'loader gates and B-val exclusions. Its pre-existing independent DAG-versus-enumeration '
        'check reports max difference 3.197442310920451e-14; this is not a before/after comparison '
        'or a relaxed equivalence criterion. `check_packaging` passes 20 prefits, 32 validation '
        'specs and 44 C spec rows. Output paths alone are redirected to authorized scratch. '
        'For the existing replay RecordingTracker telemetry probe, a test-only dict exposes raw '
        'query attributes while retaining metric-code keys; production observe still takes the '
        'encoded path. Each replay uses a fresh interpreter because plugin installation is a '
        'singleton; the retained initial multi-replay attempt hit that guard after the first case.', '',
        'Commands (from the repository root, with the required prefix on every Python command):', '',
        '```bash',
        f'{PREFIX} -m unittest {MODULE}.test_stall {MODULE}.test_stall_fast {MODULE}.test_controller_fast -v',
        f'{PREFIX} -m {MODULE}.verify_controller_fast parity',
        'for cell in pi05_l10_50 groot_l10_50; do',
        '  for case in A floor uniform R R_reverse R45 stall stall30; do',
        f'    {PREFIX} -m {MODULE}.verify_controller_fast replays --cell "$cell" --case "$case"',
        '  done', 'done',
        f'{PREFIX} -m {MODULE}.verify_controller_fast checks',
        f'{PREFIX} -m {MODULE}.verify_controller_fast calibration',
        f'{PREFIX} -m {MODULE}.verify_controller_fast timing',
        f'{PREFIX} -m {MODULE}.build_fast_handback',
        f'{PREFIX} -m {MODULE}.build_controller_handback',
        '```', '',
        'The final unit suite was run through unittest\'s loader/runner to also snapshot temporary '
        'directory names; `unit_final.log` and `test_cleanup.json` retain results. All successful '
        'stages exit 0. Existing replay/calibration outputs are never overwritten; reruns need '
        'fresh scratch destinations. Counts apply to fixed recorded observations and histories, '
        'not new closed-loop trajectories. C validation logs still lack replayable visual inputs '
        '(§4); no additional validation-log agreement claim is made.', '',
        'Follow-up files changed: `ideation_Q1/method_c/methods.py`, `test_stall.py`, '
        '`build_fast_handback.py`, `HANDBACK_FAST.md`. New files: `test_controller_fast.py`, '
        '`verify_controller_fast.py`, `build_controller_handback.py`, `timing_controller_fast.csv`, '
        '`timing_controller_fast_samples.csv` (3,734 rows), `verification_controller_fast.json`. '
        'The two builders preserve §8 when regenerating the initial report. Other follow-up outputs '
        'and full source hashes are in `/tmp/codex_stall_fast/controller_followup/` and the evidence JSON.', ''
    ]
    report = HERE/'HANDBACK_FAST.md'
    initial = report.read_text().split(MARKER, 1)[0].rstrip()
    report.write_text(initial+'\n\n'+'\n'.join(lines))
    print(json.dumps(dict(compared_anchors=compared, exact_differences=0,
        calibration_byte_equal=8, tests=cleanup['tests'], replays=len(replays),
        timing_samples=samples, cpu_median_speedup_min=min(speedups), cpu_median_speedup_max=max(speedups))))


if __name__ == '__main__':
    main()
