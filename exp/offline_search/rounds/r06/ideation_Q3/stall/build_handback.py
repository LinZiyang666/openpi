"""Render the delivery from completed checks, without running an experiment."""
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=Path('/tmp/q3_stall_fits')
CELLS=[f'{m}_{s}_{n}' for m in ('pi05','groot') for s in ('l10','sp') for n in (50,500)]
PREFIX="taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python"
MODULE='exp.offline_search.rounds.r06.ideation_Q3.stall'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while data:=f.read(8<<20):h.update(data)
    return h.hexdigest()


def md(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'


def main():
    fit={c:json.loads((HERE/f'fit_{c}.json').read_text()) for c in CELLS}
    refit={c:json.loads((HERE/f'refit_{c}.json').read_text()) for c in CELLS}
    replay={c:json.loads((HERE/f'pilot_replay_{c}.json').read_text()) for c in CELLS}
    metric=json.loads((HERE/'metric_audit.json').read_text())
    specs=json.loads((HERE/'emit_arms_bmech.json').read_text())
    bfits=json.loads((HERE/'bmech_artifacts.json').read_text())
    bchecks=[json.loads((HERE/('bmech_replay_'+r['name']+'_Bmech.json')).read_text()) for r in specs]
    assert len(metric)==8 and len(specs)==4
    for c in CELLS:
        assert fit[c]['fingerprint']==refit[c]['fingerprint']==replay[c]['model_fingerprint']
        assert refit[c]['status']=='INDEPENDENT_REFIT_IDENTICAL'
        assert replay[c]['status_counts_match_previous_run']
        assert 'observe_cpu_p99_us' in replay[c]
    assert all(r['assertions']=='PASS' for r in bchecks)
    assert all(r['A_B_metric_bit_identical'] and r['saved_metric_matches_A'] and r['raw_and_code_status_identical'] for r in metric)
    states=['inactive','ok','slow_confirmed','slow_ambiguous']
    table=[]
    for c in CELLS:
        for r in replay[c]['summaries']:
            table.append(dict(cell=c,cohort=r['cohort'],scope=r['scope'],n=r['n'],**r['counts'],
                              **{k+'_rate':r['rates'][k] for k in states}))
    with (HERE/'status_rates.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    timing=[dict(cell=c,anchors=replay[c]['anchors'],
        cpu_median_ms=replay[c]['observe_cpu_median_us']/1000,cpu_p99_ms=replay[c]['observe_cpu_p99_us']/1000,
        wall_median_ms=replay[c]['observe_median_us']/1000,wall_p99_ms=replay[c]['observe_p99_us']/1000) for c in CELLS]
    with (HERE/'timing.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(timing[0]));w.writeheader();w.writerows(timing)
    commands=[]
    for c in CELLS:commands.append(f'{PREFIX} -m {MODULE}.fit_models --cell {c}')
    for c in CELLS:commands.append(f'{PREFIX} -m {MODULE}.fit_models --cell {c} --verify-reproducible')
    commands += [f'{PREFIX} -m unittest {MODULE}.test_stall -v',f'{PREFIX} -m {MODULE}.emit_bmech']
    commands += [f'{PREFIX} -m {MODULE}.replay_bmech --name {s["name"]}' for s in specs]
    commands += [f'{PREFIX} -m {MODULE}.audit_metrics']
    commands += [f'{PREFIX} -m {MODULE}.replay_pilot --cell {c}' for c in CELLS]
    commands += [f'{PREFIX} -m {MODULE}.build_handback']
    (HERE/'commands.txt').write_text('# Executed from /home/weiland/projects/openpi; at most four Python processes concurrently.\n'
        '# Initial fit/emission commands refuse existing artifacts. Refit verification is read-only on artifacts.\n'+'\n'.join(commands)+'\n')
    total_anchors=sum(r['anchors'] for r in replay.values())
    total_windows=sum(sum(t['windows'] for t in r['tasks'].values()) for r in fit.values())
    lines=['# P5 handback: C stall component and Bmech', '',
        '**Implemented, fitted, and checked.** `stall.py` exposes the exact `SELECTION.md §5` classes/methods. '
        'All eight library-only fits are under `/tmp/q3_stall_fits/<cell>/`; all eight independent refits match. '
        'Four Bmech specs and fitted artifacts are ready for the coordinator to copy. No server, worker, chain, '
        'simulator, remote host, or policy inference was started.', '',
        '**Scientific caveat from the descriptive check:** at B’s no-progress-flagged anchors, `slow_confirmed` '
        'occurs at 169/361 (46.81%) on π0.5 L10-50, 378/802 (47.13%) on GR00T L10-50, and 25/40 (62.50%) '
        'on GR00T Spatial-50. This does not demonstrate selective suppression of the harmful cell. Constants '
        'were not changed in response. These are status overlaps on B trajectories, not retained causal benefits '
        'or predicted C SR. C’s shared budget/cooldown and changed state visitation still require closed-loop validation.', '',
        '## 1. Integration and implemented recipe', '',
        'Read [INTERFACE.md](INTERFACE.md) for the metric/key protocol. The preferred input reuses the A metric code '
        'already computed at retrieval, avoiding another visual projection:', '',
        '```python\nfrom exp.offline_search.rounds.r06.ideation_Q3.stall.stall import StallModel, StallTracker\n'
        'model = StallModel.load("/path/to/copied/cell/stall.pkl")\ntracker = StallTracker(model, task_id)  # new at reset\n'
        'tracker.observe({"metric_code": code, "metric": regime}, actual_executed_control_index)\n'
        'diagnostics = tracker.status()\n```', '',
        '`observe` handles every fresh vision observation, including an extra LOOK. `status` returns cached diagnostics '
        'with state `inactive`, `ok`, `slow_confirmed`, or `slow_ambiguous`. It never requests inference, schedules a LOOK, '
        'samples a coin, or solves a budget. **C owns rho, its call priority/cooldown, and at most one ambiguous extra LOOK '
        'per W*L controls.** The previous Q3 step-5 q/B-budget rule is absent. B’s three other guards are not added to C '
        'by this component; they exist only in the separate Bmech control.', '',
        'Implemented steps 1–4:', '',
        '1. Read H/R from manifest `H`/`exec_steps`; require L=min(H,2R). Legacy library timestamps are '
        '`step*exec_steps`; a generic input may supply actual `control_index`. Phase uses the final **observed** library '
        'timestamp as phase 1, consistent with `step/(ep_len−1)` for complete regular episodes. '
        'T_med is the median final-observation duration/L for successful episodes of that task. W=max(2,ceil(.05*T_med)). '
        'Resample at L-control grid points using the latest observation at/before each point, remove repeated selections, '
        'and retain the final observation. Templates and LOEO windows use those sampled rows.',
        '2. Align W+1 query observations independently against each successful template with unrestricted nondecreasing '
        'indices (repeats and skips allowed), minimizing summed A-metric Euclidean distances. A backward suffix DP '
        'followed by earliest optimal indices gives the lexicographically earliest complete path. Template ties use '
        'episode identifier order. Use K=min(5,E−1) both online and in LOEO; multiply advances/spread by W*L/actual span. '
        'Every quantile, including the median, uses the specified inverse ECDF (the lower middle value at even n), '
        'not an interpolated median.',
        '3. On every eligible successful-library window, exclude its entire episode from templates. Keep known phase '
        'advance, estimated phase/distance/spread, and true-minus-estimated residual. Build episode-equal CDFs for '
        'distance/spread. At a current context, select one nearest L1 context per reference episode, with earliest '
        'window breaking ties; e90 is its residual q90 and a10 its true-advance q10. Require ≥4 contributing reference '
        'episodes. The fitted representation stays fixed; this is empirical conditional calibration, not a conformal theorem.',
        '4. `delta_hat+e90<a10` gives confirmed slow; `delta_hat<a10<=delta_hat+e90` gives ambiguous; otherwise ok. '
        'Unknown task, insufficient references, warmup, or invalid observation yields inactive. Invalid observations '
        'clear the window. Duplicate/decreasing control timestamps raise an error rather than creating artificial progress. '
        'No robot displacement, gripper, contact, suite, or task-name threshold is used.', '',
        'Library success metadata only selects successful demonstrations, as requested. No evaluation success label '
        'enters a fit, and no recorded calibration trajectory is required for this component. The B-val recordings and '
        'shared-budget solution belong to P4.', '',
        '## 2. Artifacts and reproducibility', '',
        f'The eight fits contain **{total_windows:,} LOEO calibration windows**. Each directory contains `stall.pkl`, '
        '`stall.json` (payload hash, model fingerprint, actual bank-file hashes, source fit hash, task metadata), '
        'and `stall.loeo.csv`. Copy the `.pkl` and `.json` together; the CSV is the inspectable numerical table. '
        'The artifacts are self-contained and require no source-bank path at load time. Load verifies both the serialized '
        'payload hash and deterministic content fingerprint. Only trusted artifacts should be loaded (pickle is not a sandbox).', '',
        md(['Cell','Successful E/task','W','LOEO windows','Artifact MiB','Fingerprint prefix'],[
            [c,f'{min(t["E"] for t in fit[c]["tasks"].values())}–{max(t["E"] for t in fit[c]["tasks"].values())}',
             '/'.join(map(str,sorted({t['W'] for t in fit[c]['tasks'].values()}))),
             sum(t['windows'] for t in fit[c]['tasks'].values()),f'{fit[c]["bytes"]/2**20:.2f}',fit[c]['fingerprint'][:16]] for c in CELLS]),
        'All 80 tasks meet the ≥4-reference rule. Only task 8 in the two L10-500 cells has W=3; all other tasks have W=2. '
        'Independent full refits, not merely load/save roundtrips, reproduce every fingerprint (which includes every LOEO '
        'record and reference array). See `fit_<cell>.json`, `refit_<cell>.json`, [source_manifest.json](source_manifest.json), '
        'and [delivery_manifest.json](delivery_manifest.json). Fit artifacts are intentionally outside the source tree.', '',
        '### Deployed metric verification and numerical convention', '',
        f'[metric_audit.json](metric_audit.json) verifies that A/B fitted metric arrays and projections are identical in '
        f'all eight cells, and the saved stall metric matches them. **{sum(r["raw_code_checks"] for r in metric):,} raw-query '
        'checks** across both regimes reproduce A’s query metric codes exactly and return the same tracker status as '
        'passing those codes directly. The early observation remains in the early coordinate system even inside a '
        'mixed early/main window; candidate template codes use that observation’s regime.', '',
        'Distances are evaluated as float64 Euclidean norms of these fixed metric codes. They are not bit-identical '
        'to A’s float32 squared-norm/dot-product evaluation (particularly near self matches), and action-tail continuity '
        'reranking is not included in the alignment metric. No PCA, learned metric, or kernel is refitted. '
        f'The recorded audit’s largest absolute distance difference is '
        f'{max(r["float64_metric_vs_A_float32_distance_max_abs"] for r in metric):.6f}; when excluding the query’s '
        f'entire source episode, the maximum is {max(r["other_episode_distance_max_abs"] for r in metric):.6f}, '
        f'maximum relative difference {100*max(r["other_episode_distance_max_relative"] for r in metric):.4f}%. '
        'This numerical convention was fixed before the pilot replays; the same arithmetic builds LOEO and serves online. '
        'Do not label the distance evaluation bit-identical to the old A implementation.', '',
        '## 3. Pilot status rates — descriptive / engineering only', '',
        f'Replayed all A/B cohorts: **960 episodes and {total_anchors:,} anchors**, 60 episodes per cohort/cell. '
        'Inputs are the official strict tables at `/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot/tables/<cell>/`. '
        'Timestamps are cumulative preceding `decisions.actual_controls`, joined to each anchor by '
        '(arm,uid,attempt,step); no nominal step count substitutes for actual execution. '
        'The replay reads no Y column. Every model was already fitted before these reads. Status counts were identical '
        'when the CPU timer was added. Warmup is included in the denominator; no interval or SR interpretation is claimed.', '',
        md(['Cell','Cohort','Anchors','Inactive %','Ok %','Confirmed %','Ambiguous %'],[
            [r['cell'],r['cohort'],r['n'],*[f'{100*r[k+"_rate"]:.2f}' for k in states]]
            for r in table if r['scope']=='all_anchors']),
        '### Overlap with B’s no-progress flags', '',
        md(['Cell','NP-flagged anchors','Confirmed count (%)','Ambiguous count (%)','Ok count','Inactive count'],[
            [r['cell'],r['n'],f'{r["slow_confirmed"]} ({100*r["slow_confirmed_rate"]:.2f}%)',
             f'{r["slow_ambiguous"]} ({100*r["slow_ambiguous_rate"]:.2f}%)',r['ok'],r['inactive']]
            for r in table if r['cohort']=='B' and r['scope']=='noprog_flagged']),
        'These include co-fired guard bits; they are not isolated NP-only calls. A-cohort overlaps use the profiler’s '
        'shadow B diagnostic and are exported separately. Source files: `pilot_status_<cell>.csv`, '
        '`pilot_replay_<cell>.json`, and [status_rates.csv](status_rates.csv). Status eligibility does not equal actual '
        'C calls or LOOKs after its shared budget, and this recorded replay cannot estimate C’s future trajectory/SR.', '',
        '## 4. Per-anchor CPU cost', '',
        'The main table times only `observe()` on the actual pilot code streams. CPU time uses `process_time_ns`; '
        'wall latency uses `perf_counter_ns`. It includes code validation/copy, alignment, reference lookup, and '
        'status construction. It excludes CSV/JSON parsing, model loading, A’s existing key projection/retrieval, '
        'and the cheap subsequent `status()` copy. Warmup anchors are included. At most four analysis processes '
        'ran concurrently, each with one BLAS/OpenMP thread and affinity 22–25,66–69; these are host measurements, '
        'not a real-time guarantee on another robot.', '',
        md(['Cell','Anchors','CPU median ms','CPU p99 ms','Wall median ms','Wall p99 ms'],[
            [r['cell'],r['anchors'],*[f'{r[k]:.3f}' for k in ('cpu_median_ms','cpu_p99_ms','wall_median_ms','wall_p99_ms')]] for r in timing]),
        'Full timing samples are in the status CSVs and [timing.csv](timing.csv). `metric_audit.json` also measures '
        'the raw-query adapter, including its additional projection, on 240 library queries per cell and both regimes. '
        'That synthetic warmup-heavy microbenchmark is explicitly separate from the pilot latency distribution. '
        'P4 should reuse the already available metric code.', '',
        '## 5. Bmech and replay identity', '',
        '[bmech.py](bmech.py) subclasses the deployed π0.5/GR00T B judges. The parent computes all diagnostics; '
        'only returned `os_flags`, `os_reason`, and `os_force_miss` are rederived with bit 8 masked. '
        'The raw diagnostic memo `_s["flag"]`, `_vision_progress`, `_noprog_span`, all other histories, '
        'cache action/score/confidence, blind LOOK veto, and policy-tail lifecycle are retained. '
        'It deliberately does **not** inherit the P2 ablation’s `blind_step` override. '
        '`mask_no_progress=False` is the exact B identity used in tests; emitted arms use its default True.', '',
        '[emit_arms_bmech.json](emit_arms_bmech.json) contains the four 50-library arms. Deployed B kwargs, '
        'client overrides and serving flags are copied; only the name, subclass, fit destination and explicit '
        '`--os-root /home/weiland/trace_runs/offline_search_store` are changed as needed. '
        'The fit is transplanted without any retraining. Copy the following files to each emitted `<RUN>/fits/<name>.pkl`:', '',
        md(['Name','Source artifact to copy'],[[r['name'],r['artifact']] for r in bfits]),
        'On 20 recorded trajectories per cell (ten tasks × inits 0/49), B, Bmech, and mask-disabled Bmech receive '
        'the **same history**, driven by Bmech decisions and same-observation recorded policy chunks. '
        'The unmasked result equals B bit-for-bit; the masked result differs only in the three verdict fields. '
        'All diagnostic memos, blind results, and policy tails are checked. Masked NP-only decisions actually exercise '
        'the still-active next-step LOOK veto in this replay.', '',
        md(['Cell arm','Decisions','Vision queries','NP-only MISSes masked','NP LOOK vetoes retained','Tail comparisons'],[
            [r['name'],r['decisions'],r['vision_queries'],r['masked_miss_decisions'],r['noprog_look_vetoes'],r['policy_tail_comparisons']]
            for r in bchecks]),
        f'Total: **{sum(r["decisions"] for r in bchecks):,} decisions**, '
        f'{sum(r["vision_queries"] for r in bchecks):,} exact result comparisons per judge, '
        f'{sum(r["masked_miss_decisions"] for r in bchecks):,} NP-only verdict changes, '
        f'{sum(r["noprog_look_vetoes"] for r in bchecks):,} retained NP LOOK vetoes; all checks pass. '
        'A preliminary B-driven-history coverage assertion failed because that history did not exercise NP LOOKs; '
        'the test driver was corrected to Bmech while preserving identical inputs across compared judges. '
        'No method/threshold was changed to obtain the pass. Different deployed controllers need not have identical '
        'future histories after a changed MISS. This test does not claim closed-loop SR parity.', '',
        '## 6. Tests, commands, and limitations', '',
        '**13 unit tests passed** via the command in [commands.txt](commands.txt): exhaustive nondecreasing-path '
        'oracle checks on 150 integer-cost matrices with ties, inverse-ECDF/episode weighting, successful-only LOEO, '
        'minimum reference support, model roundtrip/corruption/no-overwrite, mixed regimes, variable control span, '
        'warmup/missing data/timestamps, all status inequalities, every four-bit guard combination, and mask-disabled '
        'B identity. Non-default synthetic H=11, R=3 checks that the core does not assume LIBERO control lengths. '
        'All eight real-bank independent refits and the recorded Bmech/metric checks above also passed.', '',
        'All commands were run from `/home/weiland/projects/openpi`; the complete expanded list is '
        '[commands.txt](commands.txt). Initial fitting/emission refuse to overwrite artifacts. To reproduce an existing '
        'fit, use `--verify-reproducible` (recomputes fully and checks the fingerprint):', '',
        '```bash\n'+f'{PREFIX} -m {MODULE}.fit_models --cell pi05_l10_50 --verify-reproducible\n'
        +f'{PREFIX} -m {MODULE}.replay_pilot --cell pi05_l10_50\n'
        +f'{PREFIX} -m {MODULE}.replay_bmech --name r6p5_bmech_p_l10_50\n'+'```', '',
        'Only this `stall/` directory and `/tmp/q3_stall_fits/` were written. Source code and result hashes are in '
        '[delivery_manifest.json](delivery_manifest.json). The controller integration and budget replay are P4’s work; '
        'they were not executed here. The component neither establishes progress-calibration coverage on shifted cache '
        'states nor certifies beneficial calls. No LIBERO worker, closed-loop smoke, new benchmark, or robot validation '
        'was run. The overlap counterexample above must remain visible when interpreting C’s eventual results.', '']
    (HERE/'HANDBACK.md').write_text('\n'.join(lines)+'\n')
    manifest=dict(status='COMPLETE_CPU_ENGINEERING_DELIVERY',models=8,independent_refits_identical=8,
        descriptive_pilot_episodes=960,descriptive_pilot_anchors=total_anchors,loeo_windows=total_windows,
        unit_tests_observed_passed=13,metric_raw_query_checks=sum(r['raw_code_checks'] for r in metric),
        bmech_arms=4,bmech_replay_decisions=sum(r['decisions'] for r in bchecks),
        sources={p.name:sha(p) for p in sorted(HERE.glob('*.py'))},
        artifacts={c:{p.name:sha(p) for p in sorted((OUT/c).iterdir()) if p.is_file()} for c in CELLS},
        model_fingerprints={c:fit[c]['fingerprint'] for c in CELLS},
        bmech_artifacts={r['name']:r['sha256'] for r in bfits},
        results={p.name:sha(p) for p in sorted(HERE.glob('*.json')) if p.name!='delivery_manifest.json'},
        handback_sha256=sha(HERE/'HANDBACK.md'),no_closed_loop_experiment=True,no_test_outcome_fitting=True)
    (HERE/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('sources','artifacts','results','model_fingerprints','bmech_artifacts')}))


if __name__=='__main__':main()
