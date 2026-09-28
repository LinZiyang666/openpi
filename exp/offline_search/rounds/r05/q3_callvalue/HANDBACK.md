# Q3 hand-back: causal landmark call value

Completed UTC: 2026-09-28T03:09:30.536623+00:00.

**Both g500 and g50: baseline CALL everywhere at eligible landmarks under the unchanged guard-only controller.** No fitted context or parent has a supported positive conservative saving, and every held-init and held-task fit returns zero suppression. No plugin feature, candidate, installer, or pilot arm is warranted. `arms_q3.json` is the valid empty emit_arms list `[]`.

This completes the conditional deployment decision; it is not waiting for Q5. No shared file was edited or installed. The requested K2/K1/K4/K5/K6/K7/K10/Q2 plugin selftest matrices were **not run**, because no plugin implementation exists. Q5 ownership was respected. No live run, GPU, server, port, LIBERO worker, chain, git command, or review-test access occurred. All Python processes used CPUs 26–29,70–73, BLAS/OMP=1 and CUDA disabled; analysis commands ran sequentially.

**Provenance: borrowed big-library information.** These outcome-derived fits use randomized rollout states and outcomes beyond the deployed action library, at both scales. They are not demo-only fits. No new closed-loop SR or IR is claimed.

## Inputs and audit

Raw root: `/home/weiland/trace_runs/os_closed_loop/r04_k5`. Arms: `r4k5_p_l10_g500_{r1,r2}` and `r4k5_p_l10_g50_{r1,r2}`. K5 `estimate.py:load_arm` reloaded accepted terminal attempts, checked continuity, served heads, ordinary guard verdicts, assignments, contexts, opportunities, treatment compliance and episode totals. `validate_pairs` retained both complementary replicas for every `(task_id, original init)`. Every task 0–9 and original init 0–49 is present at each scale.

| Scale | Episodes / clusters | Decisions / MISS | Exposures | Exposure-discordant pairs | Duplicates |
|---|---:|---:|---:|---:|---:|
| g500 | 1000 / 500 | 60225 / 6511 | 634 | 98 | 0 |
| g50 | 1000 / 500 | 65566 / 12860 | 818 | 68 | 0 |

There are zero unknown exposed contexts at both scales. Raw-derived episode records exactly match the coordinator’s `k5_g{scale}_episodes.json`. Every field returned by a fresh K5 estimate (including all parent/child effects, bootstrap intervals and audits) exactly matches `k5_g{scale}_estimate.json`. Source paths, byte sizes and SHA256 values are recorded in each `results/audited_g{scale}_episodes.json`; raw runs remained read-only.

Both controller contracts are exactly guard-only MixedJudge/AWM, π0.5 l10, full stage 1, K=10 MISS, cap 0, judge burst 1, step0 judge, method guards=true and events=none. g500 base_kwargs is `{}`; g50 is `{"lib":"current","kref":5}`. No blindness or tail transfer was fitted.

## Solver and held-out evaluation

`cost_solver_reference.py` is a byte-for-byte snapshot of ideation B `cost_solver.py`; `causal_fit` and `ope` are called unchanged. SHA256: `e24be42c36b244bcf025da921e0353646193c2e958dc753ac411828cc47babfb`. `PROTOCOL.md` records the analysis rules and acknowledges the initial read-only null fit before the full raw-data audit. No support, tolerance, rho, Bonferroni multiplier or model family was tuned to obtain an override.

- Support: ≥30 init clusters and ≥10 episode observations per treatment in a cell. The full-data leaf fit tests 26 observed cells at g500 and 27 at g50; all 48 prespecified cells are enumerated in the report, including unsupported unobserved cells.
- Success-loss constraint: .01 absolute. Cost is `Cρ=.848 M+(.152−ρ) N`, with `ρ500=.23821622358554875` and `ρ50=.3234160220826887`. It includes all downstream calls and episode length; saving is CALL cost minus policy cost. C units are per episode, not IR.
- Approximate cluster-normal bounds use `z=Φ⁻¹(1−.05/(2·observed_cells))`, separately per outcome. Full-data z is 3.101861833740095 at g500, 3.1130172634086897 at g50; the two-parent variant uses 2.241402727604947. These are not exact small-sample or joint two-outcome/two-scale guarantees.
- Five init-mod-5 folds: 400 training clusters, 100 held out per fold, both replicas kept together. Held-out IPW uses propensity .5 and pools 500 init-cluster contributions. Per-context reports give full-data probability, all five training-fold probabilities, and held-out cost/SR intervals.
- Parent-only first/third-landmark fits map the exposed context to `{}`, retaining the assigned landmark, all clusters, outcomes and support/loss rules. These are reported separately from the primary leaf fit. Ten task folds use 450 training and 50 held-out clusters.

| Scale / table | Sample-supported | Supported positive cost LCB | Nonzero p | Held-init saving [95%] | Held-init ΔSR [95%] |
|---|---:|---:|---:|---|---|
| g500 / leaf | 2 / 48 | 0 | 0 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| g500 / parent | 2 / 2 | 0 | 0 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| g50 / leaf | 7 / 48 | 0 | 0 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |
| g50 / parent | 2 / 2 | 0 | 0 | +0.000000 [+0.000000, +0.000000] | +0.000000 [+0.000000, +0.000000] |

All 20 init-fold and all 40 task-fold fits have p=0 in every cell. Pooled task-held-out saving and ΔSR are likewise 0 [0,0] in all four scale/family combinations. These are structural zero contrasts because the learned policy equals baseline CALL; they do not establish that suppressing calls is safe or that an unobserved context has zero causal effect.

The complete **96 leaf rows plus four parent rows** are in [results/CONTEXTS.md](results/CONTEXTS.md). `results/g500.json` and `results/g50.json` retain every full/fold fit, support counts, probability, Bonferroni raw effect interval, held-out per-cell interval, and per-init policy contribution. Raw cell/parent deltas are population contributions with denominator 500, exactly as in B; K5’s conditional-on-exposure effects are separately retained in `results/k5_g*_reproduced.json`.

The closest supported g50 leaf is landmark 1, progress ≥.5, gripper closed, confidence ≤−.3, stall age 0–9: 41 clusters, CALL/CACHE 25/38. Its full-suppression population C saving is +.2995718773 with Bonferroni lower bound **−.0127865539**; p remains zero. No threshold was relaxed for this cell.

| Scale / parent | Clusters | CALL / CACHE | Sample support | p | Full-suppression C saving [Bonf.] |
|---|---:|---:|---|---:|---|
| g500 / 1 | 217 | 198 / 195 | yes | 0 | -0.036732 [-0.564591, +0.491126] |
| g500 / 3 | 149 | 117 / 124 | yes | 0 | -0.027359 [-0.437968, +0.383250] |
| g50 / 1 | 229 | 219 / 223 | yes | 0 | +0.309092 [-0.393256, +1.011439] |
| g50 / 3 | 214 | 186 / 190 | yes | 0 | +0.222035 [-0.389828, +0.833898] |

## Does the scale sign flip survive task holdout?

**Not as a robust SR/cost sign reversal.** The g500 SR point estimate stays positive under all ten task exclusions (+.024444 to +.042222); g50 ranges from −.008889 to +.015556 (seven negative, two zero, one positive). Excluding task 0 changes g50 to +.015556. The g500 C effect at its own rho changes sign in three exclusions (range −.227868 to +.094226); g50 stays positive at its rho (+.179292 to +.608073). The MISS-count point-sign difference does survive all ten exclusions: g500 −.468889 to −.024444, g50 +.055556 to +.742222. Its uncertainty still includes zero.

| ITT or scale contrast | ΔSR [init bootstrap 95%] | ΔSR [task t9 95%] |
|---|---|---|
| g500 CALL−CACHE | +0.034000 [+0.004000, +0.066000] | +0.034000 [-0.000423, +0.068423] |
| g50 CALL−CACHE | -0.002000 [-0.040000, +0.032000] | -0.002000 [-0.045422, +0.041422] |
| g500−g50 treatment-effect contrast | +0.036000 [-0.010000, +0.084000] | +0.036000 [-0.016155, +0.088155] |

The scale SR contrast is positive in all ten nine-task remainders, but the held-out tasks themselves have five positive, three zero and two negative contrasts. Neither the paired init-bootstrap contrast nor the task-level t interval excludes zero. This supports treating the apparent reversal as unresolved heterogeneity, not a deployable sign rule.

Cost contrasts use **the same rho for both scales** to avoid attributing a change in objective to library size. g500-minus-g50 ΔC is −.338442 [−1.267851,+.603059] at rho50 and −.469479 [−1.551337,+.612736] at rho500 (paired init bootstrap). Task t9 intervals also include zero. At rho50 the full g500 CALL−CACHE cost effect is itself positive +.069356, so the original own-rho cost sign flip is not invariant to the target rho.

Sensitivity intervals use 2,000 original-init bootstrap draws, seed 20260927, with the same cluster weights across scales. The t9 analysis treats the ten task means as sampling units and is a sensitivity check with only ten tasks. Complete task and exclusion numbers are in [SENSITIVITY.md](SENSITIVITY.md) and `results/task_sensitivity.json`. All ten held-task policy fits at each scale/family still return baseline CALL.

## Verification and exact commands

Final PASS: **10 deterministic artifacts byte-identical** across final and rerun directories; **64 fits, 912 cell calculations, 60 held-out evaluations** independently checked. Six nonzero diagnostic policies exercise IPW sign, .5 propensity, replica averaging and clustered SE on real data. Four planted fixtures check beneficial-call suppression, the 30-versus-29 cluster boundary, the exact .01 SR-loss constraint, and rejection at nine CALL observations despite 30 clusters. Planted outcomes are verification fixtures only. No final numerical check failed. The first hand-back renderer invocation had an unterminated string literal; it was corrected before publication and did not modify the fitted results.

`run_final.sh` recomputed the fitter/held-out evaluation twice against the final files, rehashed every raw source, compared the deterministic results, ran `verify.py`, and invoked the unchanged arm emitter. The emitter produced `/tmp/q3_empty_arms_check/arms.json` = `[]`. Evidence: `verification.json`, `verification.log`, `final_analysis.log`, `rerun.log`, `emitter.log`.

Commands actually run from `/home/weiland/projects/openpi`:

```bash
bash exp/offline_search/rounds/r05/q3_callvalue/run_analysis.sh > exp/offline_search/rounds/r05/q3_callvalue/analysis.log 2>&1
bash exp/offline_search/rounds/r05/q3_callvalue/run_final.sh
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 HIGHS_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r05/q3_callvalue/write_handback.py
```

The shell wrappers prefix every Python command with `taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 HIGHS_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python`. `run_analysis.sh` alone repeats raw ingestion. `run_final.sh` reuses the raw-audited episode exports only after checking every original input hash. It refits every fold and repeats the coordinator estimator. NumPy/SciPy versions and source hashes are in `results/source_code.json`.

## Storage, files and deployment disposition

| Existing artifact (measured; no new fit) | Bytes | SHA256 |
|---|---:|---|
| g500 guard fit | 141226847 | `dfe51ea22257e877d4036933a8c7b60655e4ac0bf66ca948f4c9b76898cbf52f` |
| g50 guard fit | 32602497 | `a6fe33ee8dda8843e693880d224b0c7aff38ea48356692d58c97a34975ccfcd8` |
| Original pi05 l10 deployed library pkl | 1103155631 | `f13517ad907817a0739d5702b2d3244f85be8659e5a7f3eebfbc4424eb848ce8` |

K5’s library sizes are 500 episodes / 29,472 entries and 50 episodes / 2,640 entries (K5 hand-back); the measured guard pickles above include representation/actions and auxiliary arrays. They are distinct from the original deployed library pickle. Added serving-table storage is **0 bytes**, since no table is deployed. Analysis JSON size is not a served fit size.

Files added under `exp/offline_search/rounds/r05/q3_callvalue/`: `PROTOCOL.md`, `cost_solver_reference.py`, `analyze.py`, `verify.py`, `run_analysis.sh`, `run_final.sh`, `write_handback.py`, `arms_q3.json`, `commands.json`, `delivery_metadata.json`, `verification.json`, `SENSITIVITY.md`, `HANDBACK.md`, logs, and the `results/` and `rerun/` analyses. `inventory.json` records final byte sizes, modification UTC and SHA256 for every owned artifact except itself. Shared install times: none.

Coordinator next step: **do not launch R5-d call-value arms at either scale**. No prefit, plugin flag, smoke run or randomized override control is required. With zero deployable scales, there are zero baseline/table/equal-override triples; `arms_q3.json=[]` deliberately avoids scheduling three identical zero-override policies. Q5 hand-back status at completion: `False`; it does not change the statistical decision. There is no pending candidate installation.

K5 identifies one assigned first/third baseline MISS under this exact guard-only controller. It supplies no permission or evidence for every-decision suppression, a different library scale, K7/commit tails, GR00T, or spatial. Further data collection or a different acceptance rule would be a new study, not completion of a supported table from this one.
