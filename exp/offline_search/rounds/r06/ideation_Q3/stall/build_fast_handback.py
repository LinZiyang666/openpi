"""Render the fast-tracker report/CSV evidence from completed scratch checks."""
import csv
import hashlib
import json
from pathlib import Path

from .verify_fast import CELLS, HERE, SCRATCH

PREFIX = "taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python"
MODULE = 'exp.offline_search.rounds.r06.ideation_Q3.stall'


def read(name): return json.loads((SCRATCH/name).read_text())


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])+'\n'


def main():
    report = HERE/'HANDBACK_FAST.md'
    marker = '## 8. Controller path (follow-up)'
    previous = report.read_text() if report.exists() else ''
    followup = marker + previous.split(marker, 1)[1] if marker in previous else ''
    pilots = {c: read(f'pilot_{c}.json') for c in CELLS}
    calibration = {c: read(f'calibration_{c}.json') for c in CELLS}
    timings = {c: {r['implementation']: r for r in read(f'timing_{c}.json')} for c in CELLS}
    breakdowns = {c: read(f'breakdown_{c}.json') for c in CELLS}
    audit = read('metric_audit/metric_audit.json')
    artifacts = read('artifact_verification.json')
    logs = read('decision_log_check.json')
    assert sum(r['anchors'] for r in pilots.values()) == 20836
    assert all(r['differences'] == 0 for r in [*pilots.values(), *calibration.values()])
    assert sum(r['raw_code_checks'] for r in audit) == 1920
    rows = [timings[c][name] for c in CELLS for name in ('reference', 'fast')]
    with (HERE/'timing_fast.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    count = 0
    with (HERE/'timing_fast_samples.csv').open('w', newline='') as f:
        writer = None
        for c in CELLS:
            for name in ('reference', 'fast'):
                with (SCRATCH/f'timing_{name}_{c}.csv').open() as src:
                    reader = csv.DictReader(src)
                    for r in reader:
                        r = dict(cell=c, implementation=name, **r)
                        if writer is None:
                            writer = csv.DictWriter(f, fieldnames=list(r)); writer.writeheader()
                        writer.writerow(r); count += 1
    assert count == 41672
    evidence = dict(pilot=pilots, calibration=calibration, timing=timings, breakdown=breakdowns,
        metric_audit=audit, artifacts=artifacts, decision_logs=logs,
        frozen_source_sha256=hashlib.sha256((HERE/'stall_reference.py').read_bytes()).hexdigest())
    (HERE/'verification_fast.json').write_text(json.dumps(evidence, indent=2, allow_nan=False)+'\n')
    speedups = {c: timings[c]['reference']['cpu_median_ms']/timings[c]['fast']['cpu_median_ms'] for c in CELLS}
    fast50 = [timings[c]['fast']['cpu_median_ms'] for c in CELLS if c.endswith('_50')]
    fast500 = [timings[c]['fast']['cpu_median_ms'] for c in CELLS if c.endswith('_500')]
    distances = {c: breakdowns[c]['components']['distance']['median_ms'] for c in CELLS}
    lines = [
        '# Exact online stall optimization: verification and timing', '',
        '**All equivalence checks pass; performance targets are missed on this host.** '
        f'The eight-cell pilot replay compares 20,836 anchors / 960 episodes and '
        f'{sum(r["per_template_paths_and_distance_matrices"] for r in pilots.values()):,} per-template '
        'distance matrices and complete paths, with zero differences and maximum absolute float difference 0. '
        'All 1,920 raw-query audit checks pass. All eight production calibration JSONs and their cost-replay JSONs '
        'are exactly equal to production; 160 budget solutions and 5,052 cadence nodes are covered. '
        'Sixteen artifact copies load and verify with unchanged payload hashes, sidecars and content fingerprints.', '',
        f'Observe-only CPU medians are {min(fast50):.3f}–{max(fast50):.3f} ms for 50-episode cells and '
        f'{min(fast500):.3f}–{max(fast500):.3f} ms for 500-episode cells. '
        f'Median speedups are {min(speedups.values()):.2f}–{max(speedups.values()):.2f}×. '
        'None meets the requested ≤0.1 / ≤0.3 ms target. No numeric tolerance or tie-breaking relaxation was used.', '',
        '## 1. Implementation and files', '',
        '- `stall.py`: `_template_cache` derives per-task/per-regime contiguous float64 codes stacked over valid '
        'points, padded phases, valid-point masks and episode identities. `_distances` retains the original '
        'subtraction / `einsum("ij,ij->i")` / square-root arithmetic. `_estimate_distances` and '
        '`_monotone_alignment_batch` batch the backward suffix DP and greedy earliest-path recovery over all '
        'templates. Stable sorting preserves (mean, original template order). Quantile indices are the same '
        'inverse ECDF, including the lower middle at even K. `_reference_cache` and `calibrated_status` batch '
        'nearest L1 contexts over all reference episodes; padding is masked and first argmin preserves earliest ties.',
        '- `StallTracker.observe`: each tracker has a private bounded distance deque. The first full window '
        'computes its W+1 columns once; subsequent anchors compute only the newest observation. Invalid '
        'observations clear both deques. A distance overflow raises the same ValueError at the same first '
        'complete window as before, and clears the distance cache for correct recovery if the caller continues. '
        'Distance computation is deliberately deferred during warmup to preserve the original exception timing.',
        '- Repeated alignment validation and five estimate-quantile conversions/checks per anchor are gone. '
        'Public `encode`, `monotone_alignment`, timestamp validation and `status` copies retain their original '
        'behavior. Encoded input snapshots remain owned/read-only, so caller mutation cannot corrupt a window. '
        'The derived caches are ordinary private model attributes outside `_payload`, never inserted into '
        '`tasks`, metric, provenance or serialized artifacts. The model remains immutable by contract.',
        '- `stall_reference.py`: byte-for-byte snapshot of the original entire module, SHA256 '
        f'`{evidence["frozen_source_sha256"]}`. `_estimate_reference` / `calibrated_status_reference` in the new '
        'model expose it; its own model/tracker classes also retain the original full boundary and fit behavior.',
        '- `test_stall_fast.py`: seven permanent parity tests; `test_stall.py` retains all 13 existing assertions/tests, '
        'with its temporary-output setup using default-location `tempfile.mkdtemp(prefix="_test_model_")` '
        'and `self.addCleanup(shutil.rmtree, path)` for exactly that test-created directory (§8).',
        '- `verify_fast.py`: separate artifacts, pilot, timing, audit, calibration, logs and component-breakdown '
        'stages, reading real inputs without changing consumers. `build_fast_handback.py` renders this report, '
        '`timing_fast.csv`, `timing_fast_samples.csv` and `verification_fast.json` from scratch evidence.', '',
        'The initial tracker optimization changed no consumer file; the follow-up changes only '
        '`method_c/methods.py` to use the existing metric-code protocol (§8). Consumer interfaces, '
        '`StallModel.fit/load/save`, `_payload`, SCHEMA, projection, '
        'CDFs, serialized templates/calibration, and artifact overwrite refusal remain intact. No production '
        'artifact, arm, spec, run root, or fitted model was rewritten. No real bank was refitted.', '',
        '## 2. Exact equivalence', '',
        'The pilot joins/filter/order/timestamps are exactly those in `replay_pilot.py`: only A/B cohorts; '
        'strict decisions and anchors joined one-to-one by (arm, uid, attempt, step); control timestamp = '
        'cumulative preceding `actual_controls`. Inputs are the strict tables under '
        '`/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot/tables/<cell>/`. No Y column is read. '
        'Warmup anchors are compared and counted.', '',
        'At every anchor the complete status dictionary is compared recursively: state/reason, '
        '`delta_hat`, `phase_hat`, `distance_hat`, `spread_hat`, `e90`, `a10`, `reference_windows`, '
        '`window_span`, `W`, `K`, `reference_episodes`. Floats are compared by packed float64 bytes, '
        'including sign bits; integers/list order/dictionary keys match. At every active anchor, independently '
        'recomputed reference distances and paths are compared byte-for-byte for every template, and the '
        'selected template identities/top-K order match. All fields and all cells have zero differences.', '',
        table(['Cell', 'A anchors', 'B anchors', 'Total', 'Active', 'Template distance/path pairs', 'Differences'],
            [[c, pilots[c]['cohorts']['A'], pilots[c]['cohorts']['B'], pilots[c]['anchors'], pilots[c]['active'],
              pilots[c]['per_template_paths_and_distance_matrices'], 0] for c in CELLS]),
        'Synthetic parity covers 14 randomized complete fits (W=2 and W=3; dimensions 1, 2, 3, 7, 16, 17, 136 '
        'and different early dimensions), 630 observation events, integer costs/code ties plus noninteger queries, '
        'variable template/reference lengths, mixed early/main windows, exclusions, span scaling, invalid '
        'observations and warmup. Every synthetic fitted payload and fingerprint equals the frozen fit. '
        'There are also 120 batched DP cases checked against the frozen scalar DP and an exhaustive path oracle; '
        'an all-contexts-tied reference test; cache call-count / episode-isolation / caller-mutation tests; '
        'timestamp/type/shape/nonfinite rejection tests and overflowing-distance recovery; and real-bank '
        'tests on π0.5 L10-50/L10-500, tasks 0 and 8 (including real W=3). A temporary save test proves '
        'caches do not change serialized payload bytes.', '',
        'Artifact evidence covers both `/tmp/q3_stall_fits/<cell>/` and '
        '`/home/weiland/trace_runs/os_closed_loop/r06_c_cal/stall/<cell>/` (all 16). '
        'Both the new and frozen loaders verify each payload SHA256/content fingerprint; loaded payloads '
        'are byte/field equal; fingerprints are rechecked after all task caches are populated; files/sidecars '
        'are hashed again and unchanged. See `verification_fast.json` for full fingerprints/hashes.', '',
        '`audit_metrics.main()` ran verbatim against the new code, with only its output directory redirected '
        'to `/tmp/codex_stall_fast/metric_audit/`. Every cell has 240 raw-query checks: A/B saved metrics '
        'match, projected raw codes equal deployed A exactly in both regimes, and raw/code tracker statuses '
        'match. The original `metric_audit.json` was preserved.', '',
        '## 3. Production calibration rerun', '',
        'The unchanged `fit_calibration.fit` was rerun in production mode (not dry-run), using the original '
        'R bank, strict B-val tables, frozen `calibration_manifest_<cell>.json`, actual reset attestation '
        'client telemetry, original stall artifact, c1=0.152 (π0.5) / 0.148 (GR00T), targets 0.18/0.30/0.45 '
        'and stall cooldown. The inputs were traced from production `calibration.json`, manifests, '
        '`chain_calibration.sh` (read only), and `fit_*.log` / `read_v2_*.log`. Spatial production directories '
        'use `spatial`, while stall artifact/pilot cells use `sp`.', '',
        'Outputs are `/tmp/codex_stall_fast/calibration_verified/<cell>/`. The consumer output allowlist '
        'only accepts `method_c/` or `/tmp/q1_method_c_fits/`; a runtime-only `unittest.mock.patch` replaces '
        'that single output-path function with a check permitting exactly the authorized scratch destination. '
        'No fitting, validation, reset attestation, budget calculation, hashing, or overwrite check is patched. '
        'An initial comparator checked the returned Python dict against parsed JSON and encountered int task '
        'keys versus serialized string keys. The verifier now compares emitted JSON to emitted JSON; '
        'the first scratch result is retained separately, and the full successful rerun used fresh outputs.', '',
        'The **entire emitted calibration JSON** equals production, without discarding even timestamps/paths. '
        'This includes λ/d parameters, floors, ceilings, feasibility, coefficients, library budgets, every '
        'stall/no-stall and R/uniform solution, provenance/hash/reset fields and stall fingerprints. '
        'The entire emitted `cost_replay.json` also equals production (topology, compact stall states/diagnostics '
        'and cost inputs). Each cell covers 20 solutions; totals: 80 episodes, 1,817 labeled anchors, '
        '160 solutions and 5,052 cadence nodes.', '',
        table(['Cell', 'Anchors', 'No-stall / stall nodes', 'Solutions', 'Stall floor', 'Stall ceiling', 'JSON / replay differences'],
            [[c, calibration[c]['anchors'], f'{calibration[c]["cadence_nodes"]["no_stall"]} / {calibration[c]["cadence_nodes"]["stall"]}',
              calibration[c]['solution_count'],
              f'{json.loads(Path(calibration[c]["production"]).read_text())["solutions"]["stall"]["R"]["0.18"]["floor"]:.12g}',
              f'{json.loads(Path(calibration[c]["production"]).read_text())["solutions"]["stall"]["R"]["0.18"]["ceiling"]:.12g}',
              '0 / 0'] for c in CELLS]),
        'Full expanded input paths, targets and fingerprints are in `verification_fast.json` and scratch '
        '`calibration_<cell>.json`; rerun products include the original bank copied into the scratch output '
        'directory by the unchanged production fitter.', '',
        '## 4. Recorded C validation decision logs', '',
        f'Inspected all {logs["arms"]} recorded arms with a configured stall artifact: '
        f'{logs["decisions"]:,} decisions and {logs["fresh_anchors"]:,} fresh anchors under '
        '`/home/weiland/trace_runs/os_closed_loop/r06_c_validation/runs/`. Fresh records log '
        '`os_c_stall_state`, control index and scalar diagnostics, but none contains a metric code, '
        'raw visual key, or input-archive pointer. Retrieval rows/weights/scores and `robot_state` cannot '
        'recover the visual query code. Therefore logged stall codes cannot be independently replayed from '
        'these logs; no agreement count is claimed. Counts describe the read snapshot. Per-arm counts are '
        'in `verification_fast.json` / `/tmp/codex_stall_fast/decision_log_check.json`.', '',
        '## 5. Observe-only timing and remaining cost', '',
        'Methodology reproduces HANDBACK §4: actual pilot code lists/regimes, every fresh anchor including '
        'warmup, only `observe()` timed with `process_time_ns` and `perf_counter_ns`. CSV/JSON parse, joins, '
        'model loading, A retrieval/projection and the following `status()` copy are outside the interval. '
        'Lazy per-task preparation is included when first needed. Fresh models/trackers are used in two '
        'standalone full passes per cell (reference then fast); equivalence/path instrumentation runs '
        'separately. Timing had one analysis Python process active, one BLAS/OpenMP thread and affinity '
        '22–25,66–69. No hardware isolation or real-time guarantee is claimed. Percentiles use the same '
        'pandas median/quantile convention as `replay_pilot.py`; float computation itself uses inverse ECDF.', '',
        table(['Cell', 'CPU median ms ref → fast', 'CPU p99 ms ref → fast', 'Wall median ms ref → fast',
               'Wall p99 ms ref → fast', 'CPU median speedup'],
            [[c, *[f'{timings[c]["reference"][k]:.3f} → {timings[c]["fast"][k]:.3f}'
                   for k in ('cpu_median_ms', 'cpu_p99_ms', 'wall_median_ms', 'wall_p99_ms')],
              f'{speedups[c]:.2f}×'] for c in CELLS]),
        'All eight target medians are missed. The retained float64 norm and NumPy DP/reference-array '
        'operations remain material. Below are CPU component medians on a separate instrumented pilot '
        'pass (including first-use work); the method timers add overhead, and these medians are not '
        'additive. Distance call counts equal anchor counts: the initial complete window fills all columns, '
        'then exactly one new column is computed per fresh anchor.', '',
        table(['Cell', 'Encode ms', 'Newest-distance ms', 'DP + estimate ms', 'Calibrated status ms'],
            [[c, *[f'{breakdowns[c]["components"][k]["median_ms"]:.3f}'
                   for k in ('encode', 'distance', 'alignment_estimate', 'calibrated_status')]] for c in CELLS]),
        '`timing_fast.csv` contains 16 full-precision summary rows. `timing_fast_samples.csv` contains '
        '41,672 per-anchor samples (20,836 each implementation). Scratch `timing_<implementation>_<cell>.csv` '
        'and `breakdown_<cell>.json` retain the separate inputs/results. Existing `timing.csv` and original '
        'pilot status CSVs were preserved.', '',
        '## 6. Retrieval-distance reuse estimate (not implemented)', '',
        'Reusing A distances could at most remove the remaining new-observation distance computation, '
        f'roughly {min(distances.values()):.3f}–{max(distances.values()):.3f} ms per anchor in the component '
        'measurement, before row gathering, square roots and padding overhead. The per-cell upper-bound '
        'estimate is the newest-distance column above; it would not remove DP, reference lookup or raw-key '
        'projection. No retrieval-reuse path or experiment was implemented.', '',
        'It is not equivalent: A computes float32 squared-norm/dot distances; stall uses float64 subtraction '
        'and `einsum` Euclidean norms, without a floor or action-tail continuity term. The rerun metric '
        f'audit records a largest absolute distance discrepancy {max(r["float64_metric_vs_A_float32_distance_max_abs"] for r in audit):.6f}; '
        f'excluding the source episode the maximum is {max(r["other_episode_distance_max_abs"] for r in audit):.6f}. '
        'Such substitutions violate bitwise `distance_hat` and could alter alignment, context selection and '
        'state inequalities. Reusing the already computed metric code remains the exact supported boundary.', '',
        '## 7. Commands, results, and limitations', '',
        'All commands run from `/home/weiland/projects/openpi` (branch Ziyang). Every Python invocation '
        'used the prefix below, at most four Python processes at once; only one during timing/breakdown. '
        'No server, chain, worker, simulator, GPU, remote host, tmux session, port or unrelated process '
        'was started or touched. The initial optimization products are confined to this `stall/` directory '
        'and `/tmp/codex_stall_fast/`; §8 additionally changes the authorized consumer `methods.py`. '
        'Initial existing-test runs left new `_test_model_*` temporary directories directly under `/tmp`; '
        'they are retained. Follow-up tests create their own default-location temporary directory and '
        'remove exactly that directory through unittest cleanup. '
        'No existing file was deleted; the pre-existing untracked `_test_model_5beh_ykn/` directory remains untouched. '
        'Unrelated pre-existing repository changes remain untouched. `tests/review_tests/` was not read.', '',
        '```bash', f'P="{PREFIX}"',
        f'$P -m unittest {MODULE}.test_stall -v',
        f'$P -m unittest {MODULE}.test_stall_fast -v',
        f'$P -m {MODULE}.verify_fast artifacts',
        f'$P -m {MODULE}.verify_fast pilot',
        f'$P -m {MODULE}.verify_fast audit',
        f'$P -m {MODULE}.verify_fast calibration',
        f'$P -m {MODULE}.verify_fast logs',
        f'$P -m {MODULE}.verify_fast timing',
        f'$P -m {MODULE}.verify_fast breakdown',
        f'$P -m {MODULE}.build_fast_handback', '```', '',
        'The two unittest commands pass: 13 existing tests and 7 new parity tests. Artifacts/pilot/audit/'
        'calibration/log-inspection/timing/breakdown stages exit 0; result counts are above. '
        'Scratch logs preserve complete output. Production calibration continues refusing existing '
        'outputs; a second calibration invocation must use a fresh scratch destination (or compare '
        'the existing scratch JSON read-only). Other verification products can be regenerated.', '',
        'Limitations: latency targets remain unmet on this host; model caches use extra memory and assume '
        'the documented immutable model contract; NumPy/CPU floating-point portability beyond the tested '
        'environment is not asserted. Timing in §5 concerns metric-code input. The follow-up removes '
        'the consumer\'s duplicated raw projection and measures controller overhead in §8; the public '
        'raw-query projection remains available. Validation log replay lacks inputs. No closed-loop trajectory/SR claim follows '
        'from fixed-stream equivalence or unchanged recorded calibration; the original HANDBACK scientific '
        'limitations still apply.', '',
        'Files written: `stall.py`, `stall_reference.py`, `test_stall.py`, `test_stall_fast.py`, '
        '`verify_fast.py`, `build_fast_handback.py`, `HANDBACK_FAST.md`, `timing_fast.csv`, '
        '`timing_fast_samples.csv`, `verification_fast.json`; follow-up files are listed in §8. Other outputs are under '
        '`/tmp/codex_stall_fast/`.', ''
    ]
    # A shell variable string does not preserve quoted empty values on
    # expansion. Emit literal commands instead of an executable $P shorthand.
    prefix_line = lines.index(f'P="{PREFIX}"')
    lines.pop(prefix_line)
    lines = [line.replace('$P ', PREFIX+' ') if line.startswith('$P ') else line for line in lines]
    report.write_text('\n'.join(lines) + followup)
    print(json.dumps(dict(report=str(HERE/'HANDBACK_FAST.md'), timing_samples=count,
        anchors=20836, template_checks=427679, raw_checks=1920, budget_solutions=160,
        exact_differences=0, median_cpu_speedup_min=min(speedups.values()),
        median_cpu_speedup_max=max(speedups.values()), targets_met=0)), flush=True)


if __name__ == '__main__': main()
