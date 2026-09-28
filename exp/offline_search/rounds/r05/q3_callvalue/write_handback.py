"""Render the Q3 hand-back strictly from verified analysis files and file stats."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
DATA=Path('/home/weiland/trace_runs/os_closed_loop/r04_k5')


def read(p):
    return json.loads(p.read_text())


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def measure(p):
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),
            'modified_utc':datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat()}


def ci(value, interval, digits=6):
    return f'{value:+.{digits}f} [{interval[0]:+.{digits}f}, {interval[1]:+.{digits}f}]'


def main():
    stamp=datetime.now(timezone.utc).isoformat()
    r={s:read(HERE/f'results/g{s}.json') for s in (500,50)}
    sensitivity=read(HERE/'results/task_sensitivity.json')
    verification=read(HERE/'verification.json')
    assert verification['status']=='PASS'
    assert all(not r[s][f]['deployment']['deployable'] for s in r for f in ('leaf','parent'))
    assert read(HERE/'arms_q3.json')==[]
    assert read(Path('/tmp/q3_empty_arms_check/arms.json'))==[]
    metadata={
        'completed_utc':stamp,'shared_files_installed':[],
        'q5_handback_exists_at_completion':(HERE.parent/'q5_gpu/HANDBACK.md').exists(),
        'existing_guard_fits':{str(s):measure(DATA/'fits'/('r3mx_p_l10_g500.pkl' if s==500 else 'r3mx_p_l10_g.pkl')) for s in r},
        'original_deployed_pkl':measure(Path('/home/weiland/trace_runs/dual_20260923/libs/libero_10/cp1_spatial_pool_16.pkl')),
        'new_serving_table_bytes':0,
        'emitter_output':measure(Path('/tmp/q3_empty_arms_check/arms.json')),
    }
    (HERE/'delivery_metadata.json').write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
    commands=[
        {'purpose':'initial raw accepted-attempt audit, original fitter, init/task held-out analyses',
         'command':'bash exp/offline_search/rounds/r05/q3_callvalue/run_analysis.sh > exp/offline_search/rounds/r05/q3_callvalue/analysis.log 2>&1'},
        {'purpose':'final analysis and rerun, independent verification, empty arm emitter',
         'command':'bash exp/offline_search/rounds/r05/q3_callvalue/run_final.sh'},
        {'purpose':'render verified results, measure source artifacts, produce hand-back and SHA256 inventory',
         'command':"taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 HIGHS_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r05/q3_callvalue/write_handback.py"},
    ]
    (HERE/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    lines=['# Q3 hand-back: causal landmark call value', '', f'Completed UTC: {stamp}.', '',
           '**Both g500 and g50: baseline CALL everywhere at eligible landmarks under the unchanged guard-only controller.** '
           'No fitted context or parent has a supported positive conservative saving, and every held-init and held-task '
           'fit returns zero suppression. No plugin feature, candidate, installer, or pilot arm is warranted. '
           '`arms_q3.json` is the valid empty emit_arms list `[]`.', '',
           'This completes the conditional deployment decision; it is not waiting for Q5. No shared file was edited '
           'or installed. The requested K2/K1/K4/K5/K6/K7/K10/Q2 plugin selftest matrices were **not run**, because '
           'no plugin implementation exists. Q5 ownership was respected. No live run, GPU, server, port, LIBERO worker, '
           'chain, git command, or review-test access occurred. All Python processes used CPUs 26–29,70–73, '
           'BLAS/OMP=1 and CUDA disabled; analysis commands ran sequentially.', '',
           '**Provenance: borrowed big-library information.** These outcome-derived fits use randomized rollout '
           'states and outcomes beyond the deployed action library, at both scales. They are not demo-only fits. '
           'No new closed-loop SR or IR is claimed.', '',
           '## Inputs and audit', '',
           f'Raw root: `{DATA}`. Arms: `r4k5_p_l10_g500_{{r1,r2}}` and `r4k5_p_l10_g50_{{r1,r2}}`. '
           'K5 `estimate.py:load_arm` reloaded accepted terminal attempts, checked continuity, served heads, '
           'ordinary guard verdicts, assignments, contexts, opportunities, treatment compliance and episode totals. '
           '`validate_pairs` retained both complementary replicas for every `(task_id, original init)`. '
           'Every task 0–9 and original init 0–49 is present at each scale.', '',
           '| Scale | Episodes / clusters | Decisions / MISS | Exposures | Exposure-discordant pairs | Duplicates |',
           '|---|---:|---:|---:|---:|---:|']
    for scale,a in r.items():
        d=a['audit'];lines.append(f"| g{scale} | {d['episodes']} / {d['init_clusters']} | {d['total_decisions']} / {d['total_misses']} | {d['exposures']} | {d['discordant_pair_exposure']} | {sum(x['duplicate_decisions'] for x in d['input_audit'])} |")
    lines += ['', 'There are zero unknown exposed contexts at both scales. Raw-derived episode records exactly '
              'match the coordinator’s `k5_g{scale}_episodes.json`. Every field returned by a fresh K5 estimate '
              '(including all parent/child effects, bootstrap intervals and audits) exactly matches '
              '`k5_g{scale}_estimate.json`. Source paths, byte sizes and SHA256 values are recorded in each '
              '`results/audited_g{scale}_episodes.json`; raw runs remained read-only.', '',
              'Both controller contracts are exactly guard-only MixedJudge/AWM, π0.5 l10, full stage 1, K=10 MISS, '
              'cap 0, judge burst 1, step0 judge, method guards=true and events=none. g500 base_kwargs is `{}`; '
              'g50 is `{"lib":"current","kref":5}`. No blindness or tail transfer was fitted.', '',
              '## Solver and held-out evaluation', '',
              '`cost_solver_reference.py` is a byte-for-byte snapshot of ideation B `cost_solver.py`; '
              '`causal_fit` and `ope` are called unchanged. SHA256: '
              f"`{sha(HERE/'cost_solver_reference.py')}`. `PROTOCOL.md` records the analysis rules and acknowledges "
              'the initial read-only null fit before the full raw-data audit. No support, tolerance, rho, '
              'Bonferroni multiplier or model family was tuned to obtain an override.', '',
              '- Support: ≥30 init clusters and ≥10 episode observations per treatment in a cell. '
              'The full-data leaf fit tests 26 observed cells at g500 and 27 at g50; all 48 prespecified cells '
              'are enumerated in the report, including unsupported unobserved cells.',
              '- Success-loss constraint: .01 absolute. Cost is `Cρ=.848 M+(.152−ρ) N`, with '
              '`ρ500=.23821622358554875` and `ρ50=.3234160220826887`. It includes all downstream calls '
              'and episode length; saving is CALL cost minus policy cost. C units are per episode, not IR.',
              '- Approximate cluster-normal bounds use `z=Φ⁻¹(1−.05/(2·observed_cells))`, separately per outcome. '
              'Full-data z is 3.101861833740095 at g500, 3.1130172634086897 at g50; the two-parent variant uses '
              '2.241402727604947. These are not exact small-sample or joint two-outcome/two-scale guarantees.',
              '- Five init-mod-5 folds: 400 training clusters, 100 held out per fold, both replicas kept together. '
              'Held-out IPW uses propensity .5 and pools 500 init-cluster contributions. Per-context reports '
              'give full-data probability, all five training-fold probabilities, and held-out cost/SR intervals.',
              '- Parent-only first/third-landmark fits map the exposed context to `{}`, retaining the assigned '
              'landmark, all clusters, outcomes and support/loss rules. These are reported separately from the '
              'primary leaf fit. Ten task folds use 450 training and 50 held-out clusters.', '',
              '| Scale / table | Sample-supported | Supported positive cost LCB | Nonzero p | Held-init saving [95%] | Held-init ΔSR [95%] |',
              '|---|---:|---:|---:|---|---|']
    for scale in r:
        for family in ('leaf','parent'):
            a=r[scale][family];d=a['deployment'];held=a['init_crossfit']['pooled']
            lines.append(f"| g{scale} / {family} | {d['sample_supported_cells']} / {len(a['table'])} | {d['positive_cost_lcb_supported_cells']} | 0 | {ci(held['mean']['cost_saving'],held['ci95_pointwise']['cost_saving'])} | {ci(held['mean']['SR_change'],held['ci95_pointwise']['SR_change'])} |")
    lines += ['', 'All 20 init-fold and all 40 task-fold fits have p=0 in every cell. Pooled task-held-out '
              'saving and ΔSR are likewise 0 [0,0] in all four scale/family combinations. These are structural '
              'zero contrasts because the learned policy equals baseline CALL; they do not establish that '
              'suppressing calls is safe or that an unobserved context has zero causal effect.', '',
              'The complete **96 leaf rows plus four parent rows** are in [results/CONTEXTS.md](results/CONTEXTS.md). '
              '`results/g500.json` and `results/g50.json` retain every full/fold fit, support counts, probability, '
              'Bonferroni raw effect interval, held-out per-cell interval, and per-init policy contribution. '
              'Raw cell/parent deltas are population contributions with denominator 500, exactly as in B; '
              'K5’s conditional-on-exposure effects are separately retained in `results/k5_g*_reproduced.json`.', '',
              'The closest supported g50 leaf is landmark 1, progress ≥.5, gripper closed, confidence ≤−.3, '
              'stall age 0–9: 41 clusters, CALL/CACHE 25/38. Its full-suppression population C saving is '
              '+.2995718773 with Bonferroni lower bound **−.0127865539**; p remains zero. '
              'No threshold was relaxed for this cell.', '',
              '| Scale / parent | Clusters | CALL / CACHE | Sample support | p | Full-suppression C saving [Bonf.] |',
              '|---|---:|---:|---|---:|---|']
    for scale in r:
        for c in r[scale]['parent']['table']:
            lines.append(f"| g{scale} / {c['landmark_class']} | {c['clusters']} | {c['n_call']} / {c['n_cache']} | yes | 0 | {ci(c['delta_Y_N_M_C_population'][3],c['CALL_minus_CACHE_population_ci95_bonferroni'][3])} |")
    lines += ['', '## Does the scale sign flip survive task holdout?', '',
              '**Not as a robust SR/cost sign reversal.** The g500 SR point estimate stays positive under '
              'all ten task exclusions (+.024444 to +.042222); g50 ranges from −.008889 to +.015556 '
              '(seven negative, two zero, one positive). Excluding task 0 changes g50 to +.015556. '
              'The g500 C effect at its own rho changes sign in three exclusions (range −.227868 to +.094226); '
              'g50 stays positive at its rho (+.179292 to +.608073). The MISS-count point-sign difference '
              'does survive all ten exclusions: g500 −.468889 to −.024444, g50 +.055556 to +.742222. '
              'Its uncertainty still includes zero.', '',
              '| ITT or scale contrast | ΔSR [init bootstrap 95%] | ΔSR [task t9 95%] |',
              '|---|---|---|']
    for name,a in [('g500 CALL−CACHE',sensitivity['scale_effects']['500']),('g50 CALL−CACHE',sensitivity['scale_effects']['50']),('g500−g50 treatment-effect contrast',sensitivity['paired_scale_contrast'])]:
        lines.append(f"| {name} | {ci(a['mean']['Y'],a['ci95_init_bootstrap']['Y'])} | {ci(a['mean']['Y'],a['ci95_task_t9']['Y'])} |")
    lines += ['', 'The scale SR contrast is positive in all ten nine-task remainders, but the held-out '
              'tasks themselves have five positive, three zero and two negative contrasts. Neither the paired '
              'init-bootstrap contrast nor the task-level t interval excludes zero. This supports treating '
              'the apparent reversal as unresolved heterogeneity, not a deployable sign rule.', '',
              'Cost contrasts use **the same rho for both scales** to avoid attributing a change in objective '
              'to library size. g500-minus-g50 ΔC is −.338442 [−1.267851,+.603059] at rho50 and '
              '−.469479 [−1.551337,+.612736] at rho500 (paired init bootstrap). Task t9 intervals also '
              'include zero. At rho50 the full g500 CALL−CACHE cost effect is itself positive +.069356, '
              'so the original own-rho cost sign flip is not invariant to the target rho.', '',
              'Sensitivity intervals use 2,000 original-init bootstrap draws, seed 20260927, with the same '
              'cluster weights across scales. The t9 analysis treats the ten task means as sampling units '
              'and is a sensitivity check with only ten tasks. Complete task and exclusion numbers are '
              'in [SENSITIVITY.md](SENSITIVITY.md) and `results/task_sensitivity.json`. All ten held-task '
              'policy fits at each scale/family still return baseline CALL.', '',
              '## Verification and exact commands', '',
              f"Final PASS: **{verification['counts']['identical_files']} deterministic artifacts byte-identical** across final and rerun directories; "
              f"**{verification['counts']['fits_checked']} fits, {verification['counts']['cells_checked']} cell calculations, "
              f"{verification['counts']['fold_OPE_checked']} held-out evaluations** independently checked. "
              'Six nonzero diagnostic policies exercise IPW sign, .5 propensity, replica averaging and clustered SE '
              'on real data. Four planted fixtures check beneficial-call suppression, the 30-versus-29 cluster '
              'boundary, the exact .01 SR-loss constraint, and rejection at nine CALL observations despite '
              '30 clusters. Planted outcomes are verification fixtures only. No final numerical check failed. '
              'The first hand-back renderer invocation had an unterminated string literal; it was corrected '
              'before publication and did not modify the fitted results.', '',
              '`run_final.sh` recomputed the fitter/held-out evaluation twice against the final files, '
              'rehashed every raw source, compared the deterministic results, ran `verify.py`, and invoked '
              'the unchanged arm emitter. The emitter produced `/tmp/q3_empty_arms_check/arms.json` = `[]`. '
              'Evidence: `verification.json`, `verification.log`, `final_analysis.log`, `rerun.log`, `emitter.log`.', '',
              'Commands actually run from `/home/weiland/projects/openpi`:', '', '```bash']
    lines += [c['command'] for c in commands]
    lines += ['```', '', 'The shell wrappers prefix every Python command with `taskset -c 26-29,70-73 env '
              "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 HIGHS_THREADS=1 "
              "CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python`. "
              '`run_analysis.sh` alone repeats raw ingestion. `run_final.sh` reuses the raw-audited episode '
              'exports only after checking every original input hash. It refits every fold and repeats the '
              'coordinator estimator. NumPy/SciPy versions and source hashes are in `results/source_code.json`.', '',
              '## Storage, files and deployment disposition', '',
              '| Existing artifact (measured; no new fit) | Bytes | SHA256 |', '|---|---:|---|']
    for name,a in [('g500 guard fit',metadata['existing_guard_fits']['500']),('g50 guard fit',metadata['existing_guard_fits']['50']),('Original pi05 l10 deployed library pkl',metadata['original_deployed_pkl'])]:
        lines.append(f"| {name} | {a['bytes']} | `{a['sha256']}` |")
    lines += ['', 'K5’s library sizes are 500 episodes / 29,472 entries and 50 episodes / 2,640 entries '
              '(K5 hand-back); the measured guard pickles above include representation/actions and auxiliary '
              'arrays. They are distinct from the original deployed library pickle. Added serving-table '
              'storage is **0 bytes**, since no table is deployed. Analysis JSON size is not a served fit size.', '',
              'Files added under `exp/offline_search/rounds/r05/q3_callvalue/`: `PROTOCOL.md`, '
              '`cost_solver_reference.py`, `analyze.py`, `verify.py`, `run_analysis.sh`, `run_final.sh`, '
              '`write_handback.py`, `arms_q3.json`, `commands.json`, `delivery_metadata.json`, '
              '`verification.json`, `SENSITIVITY.md`, `HANDBACK.md`, logs, and the `results/` and '
              '`rerun/` analyses. `inventory.json` records final byte sizes, modification UTC and SHA256 '
              'for every owned artifact except itself. Shared install times: none.', '',
              'Coordinator next step: **do not launch R5-d call-value arms at either scale**. No prefit, '
              'plugin flag, smoke run or randomized override control is required. With zero deployable '
              'scales, there are zero baseline/table/equal-override triples; `arms_q3.json=[]` deliberately '
              'avoids scheduling three identical zero-override policies. Q5 hand-back status at completion: '
              f"`{metadata['q5_handback_exists_at_completion']}`; it does not change the statistical decision. "
              'There is no pending candidate installation.', '',
              'K5 identifies one assigned first/third baseline MISS under this exact guard-only controller. '
              'It supplies no permission or evidence for every-decision suppression, a different library '
              'scale, K7/commit tails, GR00T, or spatial. Further data collection or a different acceptance '
              'rule would be a new study, not completion of a supported table from this one.', '']
    (HERE/'HANDBACK.md').write_text('\n'.join(lines))
    task_lines=['# Task-held-out sensitivity', '', '**Borrowed big-library information.** CALL minus CACHE ITT, '
                'both replicas retained. C500 and C50 use the corresponding common rho; the table below uses '
                'each scale’s own rho. `results/task_sensitivity.json` also has both common-rho effects and '
                'paired cross-scale intervals. The held-task policy OPE is zero [0,0] for every fitted family.', '']
    for group,title in [('by_task','Effect on each held-out task (50 clusters)'),('leave_one_task_out','Effect after excluding each task (450 clusters)')]:
        task_lines += [f'## {title}', '', '| Task held out | g500 ΔSR | g50 ΔSR | g500 ΔM | g50 ΔM | g500 ΔCρ500 | g50 ΔCρ50 | ΔSR contrast |', '|---:|---:|---:|---:|---:|---:|---:|---:|']
        for d in sensitivity[group]:
            vals=[d['g500']['Y'],d['g50']['Y'],d['g500']['M'],d['g50']['M'],d['g500']['C_g500'],d['g50']['C_g50'],d['g500_minus_g50']['Y']]
            task_lines.append('| '+str(d['task'])+' | '+' | '.join(f'{x:+.6f}' for x in vals)+' |')
        task_lines += ['']
    (HERE/'SENSITIVITY.md').write_text('\n'.join(task_lines)+'\n')
    inventory={str(p.relative_to(HERE)):measure(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='inventory.json'}
    (HERE/'inventory.json').write_text(json.dumps(inventory,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'completed_utc':stamp,'handback':str(HERE/'HANDBACK.md'),'sha256':sha(HERE/'HANDBACK.md'),
                      'deployable_scales':[],'owned_files':len(inventory)}))


if __name__=='__main__':main()
