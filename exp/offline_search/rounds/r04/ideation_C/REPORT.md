# R4 ideation C: store actions at finer time offsets; measure changes with stratified pilots

This report recommends **one exploratory control experiment and one evaluation change**. I investigated four control/representation ideas rather than filling the proposal quota: virtual control-step entries survive as a testable hypothesis; hard phase partitioning, local affine action correction, and successor-aware whitening do not presently justify closed-loop arms.

No new closed-loop experiment was run. All SRs below are existing runs, reproduced from accepted journal attempts. All new performance forecasts are explicitly hypotheses. No subagents, GPU, server, worker, port, tmux session, or git operation was used. The requested hot store was absent; diagnostics read `/home/weiland/trace_runs/offline_search_store`. Scripts and outputs are entirely in `exp/offline_search/rounds/r04/ideation_C/`. The largest output is 15,717,324 bytes, so no large-array exception was needed.

## 1. Measured facts driving the recommendations

### Sources, reproduction, and limits

Binding inputs were read in the requested order: R4 `FINDINGS.md`, R3/R2 `ANALYSIS.md`, protocol §§8–10, `IDEATION_BRIEF.md`, both READMEs, and the AWM/AWM3/MixedJudge/V6/V7/plugin/ops code. In particular, `awm.py:AWM._dist` uses the policy tail only immediately after a MISS; `AWM._mix` averages full chunks; `PluginSession.on_executed` records the actual executed chunk. I did not propose another handoff schedule, vision-free interval, cheaper encoder, reduced denoising, growing/pruning library, or GR00T mixed arm.

Reproduce each script from the repository root with this command prefix, replacing `SCRIPT` with the names in the table:

```bash
taskset -c 24-37,68-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/weiland/projects/openpi/.venv/bin/python exp/offline_search/rounds/r04/ideation_C/SCRIPT
```

| Script, in dependency order | Outputs and purpose |
|---|---|
| `library_diagnostic.py` | `library_*.json`, `codes_*.npz`, `phases_*.npz`: eight libraries, action-pair labels and AWM geometry; eight workers |
| `log_diagnostic.py` | `logs_*.json/npz`: eight R2 CL2 arms plus five R3 π0.5-l10 mixed arms; 6,500 accepted episodes; eight workers |
| `pilot_diagnostic.py` | `pilot_backtest.json`: 39 historical method comparisons, target-excluded allocation calibration, 3,000 sampling repetitions each |
| `affine_diagnostic.py` | `affine_*.json`: bounded local affine synthesis on saved AWM results, both inf/cache regimes and both library sizes; eight workers |
| `segment_diagnostic.py` | `segment_*.json`: temporal interpolation geometry and repeated gripper transitions in CL2 spells; eight workers |
| `future_crossfit.py` | `crossfit_*.json`: whole-episode five-fold metric-label holdout; eight workers |
| `summarize.py` | `evidence_summary.json`: consolidated counts, crossfit results, actual baseline pickle bytes |
| `make_pilot.py` | 24 nested `pilot_manifest_<model>_<suite>_<50|500>_n<100|200|250>.json` files; only writes manifests |

JSONL decisions are filtered by the journal's accepted attempt, then deduplicated by `(uid, step)`. Reproduced CL2 SRs are `.800/.630/.888/.552` at 50 episodes and `.954/.768/.966/.706` at 500, in π0.5-sp/π0.5-l10/GR00T-sp/GR00T-l10 order. The five reproduced mixed SRs are `.740/.864/.816/.872/.792` for `g/g500/awm_h70/awm500_h70/perk5`.

**Important precision limit:** action/gripper diagnostics in old server logs reconstruct the logged top 10 of 16 kernel members. These are exact counts of the reconstructed patterns, not newly observed physical events. Against saved full offline synthesis, all-outcome stale-regime mean reconstruction RMS is up to `.0299σ` at 50 episodes and `.0477σ` at 500 (`affine_*.json`, `recon_rms`). Near-zero gripper votes can change sign. A deciding pilot must log the actual served head.

### A. Many failed spells repeatedly execute a within-chunk gripper transition

Definition: a spell has at least three consecutive identical top-1 picks. Count an episode when at least one spell contains at least two decisions whose reconstructed executed five-control head changes gripper sign internally. This identifies replay of a transition, not simply a constant closed/open gripper. `segment_diagnostic.py` counts:

| Cell | 50 episodes: failures with pattern / all failures | 50: successes with pattern / all successes | 500 episodes: failures with pattern / all failures | 500: successes with pattern / all successes |
|---|---:|---:|---:|---:|
| π0.5-sp | 27/100 | 1/400 | 13/23 | 2/477 |
| π0.5-l10 | 92/185 | 30/315 | 66/116 | 36/384 |
| GR00T-sp | 6/56 | 5/444 | 6/17 | 1/483 |
| GR00T-l10 | 143/224 | 77/276 | 111/147 | 97/353 |

On l10 these are 49.73%/63.84% of failures at 50 episodes and 56.90%/75.51% at 500. Successful episodes also contain them; the pattern is neither a sufficient failure condition nor evidence that these failures can all be rescued. The selected row can be a symptom of an earlier mistake.

There are actual within-head gripper-transition rows in the libraries: π0.5-l10 **146/2,640** and **1,903/29,472**; GR00T-l10 **145/2,645** and **2,030/29,631**, at 50 and 500 episodes respectively. A library entry therefore frequently represents a timed event rather than a stationary control command. Repeating its first five controls can repeat the event.

### B. Intermediate library observations are closer to their trajectory segment than to either endpoint, but interpolation is imperfect

For every real non-boundary library row `r`, hide it geometrically and project its representation onto the chord from `prev(r)` to `next(r)`. Divide projection residual by distance to the nearer endpoint. This spans ten controls and tests a real observation five controls between the endpoints. It does **not** validate synthetic one-control images or predict SR.

| Cell | Median residual ratio, AWM code, 50 / 500 | Median ratio, raw PCA-128 vision, 50 / 500 | AWM ratio at gripper events, 50 / 500 |
|---|---:|---:|---:|
| π0.5-sp | .7636 / .6910 | .5714 / .5796 | .7866 / .7492 |
| π0.5-l10 | .7146 / .6802 | .5646 / .5853 | .7597 / .7666 |
| GR00T-sp | .8243 / .7972 | .7153 / .7242 | .8219 / .8274 |
| GR00T-l10 | .8129 / .7990 | .7251 / .7422 | .8395 / .8489 |

The interpolation assumption has better measured support on π0.5. GR00T and gripper-event neighborhoods are more curved/noisy in these representations. The projection is allowed to choose its position using the held-out observation, so its advantage over endpoints is a geometric opportunity, not an unbiased performance gain. Fixed midpoints are slightly worse: π0.5-l10 AWM median ratios `.7252/.6917`, GR00T-l10 `.8212/.8082`.

### C. Sampling the evaluation pool better gives a measurable variance reduction

`pilot_diagnostic.py` compares a ten-task balanced sample of 100 inits with a sample stratified by **task and the known reference arm's success/failure**. Allocation variances are estimated only from other historical arms; the target method is excluded. Estimation uses population weights, not the oversampled raw success rate. The reference is CL2 at the same library scale.

| Library scale | Held-out historical comparisons | Median stratified / balanced design variance | Range | Balanced sample equivalent to 100 stratified inits |
|---|---:|---:|---:|---:|
| 50 episodes | 24 | .629863 | .105658–.820993 | 142.10 |
| 500 episodes | 15 | .704723 | .365343–.955207 | 130.93 |

The equivalent sample calculation includes the finite 500-init population correction. These are **conditional sampling variances of already recorded outcomes**. They do not remove stochastic rollout variance, GPU nondeterminism, or drift between runs. Maximum absolute Monte Carlo bias across comparisons was `.000949` at 50 and `.001352` at 500.

Examples, all relative to CL2 at the indicated scale:

| Target | True full-pool ΔSR | SD of 100-init estimate: balanced → stratified | Wrong/nonpositive sign frequency: balanced → stratified |
|---|---:|---:|---:|
| π0.5-sp a05, 50; **borrowed big-library information** | +4.8 pp | 3.185 → 2.404 pp | 7.90% → 2.83% |
| π0.5-l10 a05, 50; **borrowed big-library information** | +4.4 pp | 3.982 → 3.340 pp | 15.37% → 10.70% |
| π0.5-l10 g500, 500 | +9.6 pp | 3.273 → 2.540 pp | .17% → 0% |
| GR00T-l10 CL3, 500 | +3.0 pp | 3.230 → 2.711 pp | 19.43% → 12.87% |

For the full `r3f_p_l10_a05` arm, the old five-task/first-20-init subset gives **−2.0 pp**, versus **+4.4 pp** over all 500. This is a recomputation on the full arm, not the separate R3 pilot rerun's −6 pp comparison. Broader task coverage and correct weighting address this sampling problem directly.

### D. Byte and inference accounting used throughout

AWM stores a 580-byte retrieval representation per real entry, excluding action payload. Valid full chunks cost 280 bytes for π0.5 (`10×7×4`) or 448 for GR00T (`16×7×4`); executed heads alone cost 140 bytes. Existing shipped pickles retain padded `H×32` actions and auxiliary arrays, so the actual sizes below are larger. Fixed AWM overhead is approximately 17.0 MB for the two PCA bases plus approximately 1.5 MB for task transforms (AWM code and R2 analysis).

All sizes here use decimal MB. Exact pickle bytes are in `evidence_summary.json`; no new fitted deployment artifact was produced.

| Cell | Real episodes/rows, current | Real episodes/rows, big | Actual AWM pickle MB, current / big | Deployed pkl MB |
|---|---:|---:|---:|---:|
| π0.5-sp | 49 / 1,018 | 500 / 10,909 | 21.342 / 46.993 | 431 |
| π0.5-l10 | 50 / 2,640 | 500 / 29,472 | 24.631 / 94.143 | 1,103 |
| GR00T-sp | 50 / 1,063 | 500 / 11,751 | 22.249 / 58.157 | 429 |
| GR00T-l10 | 50 / 2,645 | 500 / 29,631 | 26.673 / 117.304 | 1,068 |

For π0.5, `IR=.152+.848m`, where `m` is MISS share. Pure-cache proposals below keep `m=0`, so **IR=.152 at both scales**; they cannot beat the stage-1 floor under the current cost definition. They might change requests per episode, which must also be reported. GR00T pure-cache IR is `α_G=s1/(s1+s23)` and its change is zero. The inspected R2 pure-cache logs contain no policy-stage observations from which to estimate `s23`; the R3 smoke run root contains only π0.5 arms. The historical approximate 58/232 ms smoke split in R3 documentation suggests .20, but I did not independently verify those raw GR00T timings and do not present .20 as a new measurement.

## 2. Ranked proposals

### 1 — `control_step_library`: retrieve a time offset inside a demonstrated transition

**Pitch.** Keep vision at every decision and execute five controls as today, but allow retrieval to start one to four controls into a stored trajectory edge. Serve an aligned splice of recorded controls instead of repeatedly restarting the same event chunk.

**Status and mechanism.** Exploratory, with moderate evidence for the mechanism and low confidence in SR improvement. Facts A/B establish a timed-event replay pattern and partial geometric support for finer temporal indexing. They do not establish causality. The hypothesis is that quantizing retrieval to five-control starting positions introduces avoidable timing bias near grasps/releases. A finer index could prevent some spells from forming. A truly unchanged observation still produces an unchanged deterministic answer; this method cannot guarantee escape from an existing fixed point. It does not exclude trajectories, mask terminal rows, enforce gripper commitment, schedule MISSes, or omit vision.

**Precise Method algorithm.** Build on plain AWM with `kref=5` at current scale and `kref=8` at 500, fitted only on that scale's real library.

1. Keep the original real-row PCA, transforms, main codes `Z_r`, actions, and metadata. Add `next[r]` as int32 and `ell2[r]=||Z_next−Z_r||²` as float32. Only interpolate within the same episode across consecutive recorded decision steps; otherwise retain offset zero. Terminal rows remain ordinary offset-zero candidates. Do not refit the metric on synthetic rows.
2. Keep the original step-0 and fresh-after-MISS branches. On stale queries, compute the usual current-vision main-code distances `d_r²` to every real candidate of the task.
3. For every nonterminal parent row, the code at offset `k∈{0,1,2,3,4}` is `Z_(r,k)=(1−k/5)Z_r+(k/5)Z_next`. Its squared query distance is `(1−a)d_r²+a d_next²−a(1−a)ell2[r]`, where `a=k/5`.
4. Choose one offset per parent, with ties toward smaller `k`. This is an O(C) vector operation: for nonzero `ell2`, compute `a*=(d_r²−d_next²+ell2)/(2 ell2)` and round/clamp `5a*` to 0…4; use `k=0` for zero-length edges. Keeping one offset per parent avoids adding five near-duplicate votes to a kernel. Rank parents by their chosen virtual distance and retain AWM's 16-member kernel and bandwidth rule.
5. Offset zero uses the original action unchanged. For `k>0`, the executed head is the concatenation `action[r,k:5,:7]` then `action[next(r),:k,:7]`. This is a time splice of **actually executed library controls**, not interpolation of open/closed gripper commands. Fill the unexecuted remainder to the model horizon by continuing recorded five-control heads; when the library ends, hold its last stored control. Apply the ordinary AWM kernel mean to these aligned chunks. No special gripper-sign rule is added.
6. `Result.library` remains the real library; `topk` contains parent row IDs; `Result.action` contains the synthesized `(H,32)` chunk. Parent IDs are diagnostics rather than complete action provenance, so expose offsets, effective fractional library step, event-replay metrics, and actual action head. Reset behavior stays stateless. Use AWM's confidence formula with dispersion computed from aligned heads and recompute its LOEO scales; confidence does not gate the initial pure-cache test.

**Plugin/server changes.** Serving needs no change: the plugin already accepts `Result.action`. For reviewable evidence, change `PluginSession.after_infer` to log the actual executed valid head on **HITs as well as MISSes**, and make KPI reconstruction prefer that head. Log `offset_0…offset_9` as scalars because current JSONL extras discard arrays. KPI action-repeat spells, not unchanged parent IDs alone, become the primary mechanism metric. The offline harness `phase_err` still uses the parent's progress; record fractional progress separately and label this limitation. A later MixedJudge integration must compute dispersion/votes/progress from aligned candidate heads/offsets; wrapping the current static `Tables.HD` implementation unchanged would give inconsistent guard features. No mixed arm is proposed here.

**Predicted SR/IR at both scales.** These are planning hypotheses, not fitted estimates or confidence intervals. A 100-init pilot cannot verify them. The priority is l10 because the repeated-event pattern affects many failures and spatial is already near saturation at 500.

| Cell | Measured AWM SR, 50 / 500 | Working forecast ΔSR, 50 / 500 | IR forecast at both scales |
|---|---:|---:|---:|
| π0.5-sp | .800 / .954 | 0…+2 / 0…+1 pp | .152 |
| π0.5-l10 | .630 / .768 | +1…+4 / 0…+3 pp | .152 |
| GR00T-sp | .888 / .966 | 0…+1 / approximately 0 pp | `α_G`, unchanged |
| GR00T-l10 | .552 / .706 | 0…+4 / 0…+3 pp | `α_G`, unchanged |

Negative effects are plausible, particularly on GR00T given its chord residuals. The proposed ranges express the modest gain worth investigating, not a claim that SR cannot fall. Offline error/regret/bad-rate direction is unknown: a one-control gripper event shift may raise teacher action error while improving completion. Do not rank the method by it.

**Cost tier and bytes.** T1, inherited AWM fit plus O(136L) edge statistics; query adds O(C) offset selection and O(16H×7) splicing, with the same encoder/policy calls. Method latency is unmeasured. Use an implicit representation: **588 B/real entry**, unchanged action payload and fixed bases. The equivalent explicit dictionary would have 4,894/13,000/5,115/13,025 positions at current scale and 52,545/145,360/56,755/146,155 at 500, in the usual cell order; it need not be materialized. Estimated shipped size, baseline pickle plus 8L bytes and negligible configuration metadata:

| Cell | Estimated MB at 50 / 500 | Deployed MB |
|---|---:|---:|
| π0.5-sp | 21.350 / 47.080 | 431 |
| π0.5-l10 | 24.652 / 94.379 | 1,103 |
| GR00T-sp | 22.258 / 58.251 | 429 |
| GR00T-l10 | 26.694 / 117.541 | 1,068 |

**Four-layer attribution.** Hold the real library and metric fixed. `B=AWM`; `G=virtual-distance ranking, original unshifted actions`; `GS=same ranking as G, spliced actions`. `GS−G` isolates temporal **synthesis**; `G−B` measures **method/index geometry at fixed real library**. Virtual observations are synthetic library construction, not additional collected episodes, and must be disclosed as such. Compare each exact arm at 50 and 500 for the **real-library** effect. The **control/policy-call** effect is zero. No borrowed big-library information is used in either fit.

**Cheapest next diagnostic.** Before a server arm, replay saved query observations with exact 16-member outputs, measure nonzero-offset selection, action changes, and actual versus reconstructed gripper signs. Also remove alternating real rows and test fractional retrieval against their observed codes, using only retained endpoints. Kill the temporal-resolution premise if fewer than 10% of stale l10 decisions select a nonzero offset or if offset selection has no relation to the within-head event neighborhoods. This diagnostic checks whether the intervention actually occurs; its action error is not an SR forecast.

**Closed-loop pilot.** Start only with π0.5-l10, both real-library scales. Use B's existing 500-init outcomes and run new G and GS arms on all ten tasks × inits 0…9: **four new arms × 100 episodes = 400 episodes**, solely for collapse screening. Stop a scale for an SR loss of at least 10 pp or a new task collapse. If it survives unchanged, use the already frozen 200-init manifests from proposal 2 for G and GS: reuse only convenience-pilot outcomes that belong to the manifest, run the remaining selected inits, and omit nonselected convenience inits from that weighted estimate. Do not treat the entire convenience sample as random. Confirm a claimed winner on all 500 inits. GR00T-l10 transfer is the same B/G/GS comparison at **both** scales, conditional on the π0.5 result; no GR00T mixed run is involved. Spatial is a subsequent regression check at both scales.

**Kill criterion.** A reduction in repeated events without a positive weighted ΔSR is insufficient—the CL3 lesson applies. Do not promote unless the final 500-init comparison improves SR at fixed IR with paired uncertainty supporting the improvement and no material per-task collapse. If the apparent synthesis gain disappears in `GS−G`, attribute it to index geometry rather than action timing. A small inconclusive gain is a stop-for-now outcome, not proof of success.

**Variants.** Full offsets 0…4 versus coarse `{0,2,4}`; otherwise identical fitting/kernel. G is the mandatory causal ablation. Do not add masks, commitment, random recovery, or new MISS schedules to make a negative pilot look positive.

### 2 — `stratified_paired_pilot`: spend episodes where a method comparison has variance

**Pitch.** Replace the fixed five-task pilot with a randomized, weighted sample covering every task and both known reference outcomes. This is an evaluation proposal, so it has no direct SR/IR benefit to rank against a deployed controller; its measured value is more precise closed-loop comparisons per evaluated init.

**Hypothesis.** Method changes have very different variance among reference successes and failures. Treating them as one population wastes samples and a fixed trap-task subset changes the target distribution. Fact C measures this effect at both library scales without using target outcomes to set allocation.

**Algorithm.** Let `h=(task, reference_success)` index nonempty strata in the existing A-pool, of size `N_h`. Keep the reference at the same library scale. Pool the binary success labels of CL0/CL1/CL3 conditional on CL2 success to estimate `p_b=(successes+1)/(observations+2)`; exclude the target if it is one of those historical comparators. Set `s_h=sqrt(p_b(1−p_b))`. Begin with `min(2,N_h)` samples per stratum; allocate remaining slots greedily to maximize `(N_h s_h)^2/[n_h(n_h+1)]` subject to `n_h≤N_h`. Draw without replacement within each stratum.

For paired differences `D_i=Y_new,i−Y_ref,i`, estimate

`ΔSR_hat = Σ_h (N_h/500) mean_h(D)`

with design variance

`Σ_h (N_h/500)^2 (1−n_h/N_h) sample_var_h(D)/n_h`.

Estimate candidate SR with the same weights. Report weighted paired intervals; a raw McNemar test on the oversampled rows is not the full-pool test. At 500 inits the estimator becomes the ordinary paired comparison. Freeze the allocation and random seed before seeing the new method's outcomes. The delivered manifests use seed 20260927 and nested within-stratum permutations for 100/200/250 inits. Do not use them to tune Method.query or pass init/reference-success labels into the controller.

The allocation calibration uses data beyond a deployed action library and is labelled **borrowed big-library information—evaluation-only historical closed-loop outcomes**. In the 50-episode case the calibration arms themselves all use the 50-episode library; no 500-episode representation is borrowed. The term is retained to comply with the owner's beyond-library-fit disclosure rule.

**Method/plugin/server changes.** Method API, server policy, and HIT/MISS decisions are unchanged. Ops currently supports a Cartesian `OSCL_TASKS × OSCL_EPISODES` filter; that cannot express these manifests. Add a manifest option to `ops/remote/run_gtp_subset.py` that filters `_episodes` by exact `(task_id, episode_idx)` membership; propagate it through `run_arm.sh/chain.sh`, set EXPECT to the number of distinct selected pairs, and count completion against those pairs. Keep the original 50-init pool attestation unchanged. Use distinct stage identifiers or explicit manifest-aware resumption so an earlier DONE marker cannot skip the larger sample. Extend KPI output with stratum counts, inclusion probabilities, weighted SR/paired ΔSR, and variance. Raw unweighted SR remains descriptive only.

**Predicted SR/IR, both scales.** No change to deployed SR, `m`, or IR. Pure-cache π0.5 remains `.152`; GR00T remains `α_G`; a mixed arm retains its own measured MISS share (for example existing π0.5-l10 g50 `.323416` and g500 `.238216`). Working forecast: at 100 inits retain some of the measured **37.0% / 29.5% design-variance reduction** at 50/500 library episodes. Equivalent historical savings were approximately 42/31 balanced inits per 100 stratified inits. This is evaluation sample efficiency, not a reduction in the deployment IR. Numerical intervals must still acknowledge the R3 rerun noise floor.

**Cost tier/library bytes.** T1 closed-form allocation, evaluation-side only. No added deployed keys, actions, model, or fixed fit; baseline 580 B/entry and actual 21.342/24.631/22.249/26.673 MB at current scale, 46.993/94.143/58.157/117.304 MB at 500, versus deployed 431/1,103/429/1,068 MB. Manifests and historical outcomes are research assets, not part of the deployed library.

**Attribution.** Synthesis, method, real library, and control behavior are unchanged. This changes the estimator and experiment allocation. An apparent improvement in unweighted sampled SR is not a method gain.

**Cheapest diagnostic and pilot.** The retrospective diagnostic is already complete and target-excluded: all 39 variance ratios are below 1. Next, use the delivered 200-init manifests for a new candidate versus CL2 in π0.5-l10 at 50 and 500, then the same design in GR00T-l10 at both scales if testing a pure-cache transfer. Freeze sampling before results arrive; finish the remaining pool for candidates that merit confirmation. The subsequent full 500 episodes supply the decisive prospective check of the weighted pilot, without a separate simulator experiment solely for the sampling proposal.

**Kill criterion.** Reject the claimed sampling advantage if prospective across-method design variance/RMSE fails to improve by at least 15% at either scale or weighted intervals systematically miss the eventual full-pool results. A single 100-init pilot does not assess interval coverage. Do not use this as a reason to rank 1–2 pp l10 differences at 100 inits.

**Variants.** Mandatory simple control: random ten-task-balanced sampling at identical n. Optional stratification solely by task/reference outcome with equal conditional variance, avoiding historical calibration; compare its design variance before deployment. Keep randomization and population weights in every variant.

## 3. Rejected ideas, with measured reasons

### Hard subtask gates from cumulative gripper events

A simple event-count phase is cheap but not a reliable cross-episode subtask identifier. At 50 episodes, **31.68%/38.13%** of immediate-action-nearest training pairs on π0.5-l10/GR00T-l10 have different cumulative event counts; at 500 these are **35.84%/38.28%** (`library_diagnostic.py`, row-weighted `pair_phase_mismatch`). Even successful CL2 episodes have kernel majority-phase mass below .8 on **31.8%/53.3%** of decisions at 50 and **36.5%/39.0%** at 500 (`log_diagnostic.py`). Mean terminal event counts in the l10 libraries are π0.5 **4.06/5.864**, GR00T **4.30/7.018**, and the 500-episode p90 is 12 for both models; counts include retries/chatter, not only task stages.

A hard gate would suppress candidates during a large fraction of successful control. This rejects this specific phase construction, not an independently validated graph/HMM. It is not a new proposal for gripper commitment. There is no extra IR to recover its likely cost: π0.5 would remain .152; GR00T `α_G`; storage would add a tiny phase label to the same 50/500 libraries in §1D.

### Bounded local affine synthesis from end-effector position

I fitted a weighted local linear action correction from the selected rows' first three robot-state dimensions, using a 3×3 trace-ridge solve. Total negative kernel mass was capped at .10, continuous-action correction RMS at .15σ, and the gripper was unchanged. This is a synthesis experiment at fixed library/ranking, with no policy blend. It runs on saved AWM top-10 neighbors and both trace regimes.

On stale π0.5-l10 observations, reconstructed mean error **.5184→.5242** at 50 and **.4355→.4478** at 500; opposite teacher translation direction increases **10.70%→11.47%** and **7.79%→8.70%**. On GR00T-l10, error is **.5180→.5170** at 50 but **.4492→.4529** at 500; fresh error worsens **.3311→.3401** and **.2671→.2690**. Uncapped negative weight mass averages π0.5-l10 **.6265/.8393**, GR00T-l10 **.5759/.9811** on stale observations: the geometry requests large extrapolation, not a small local correction.

Offline error cannot veto a real closed-loop mechanism by itself. Here the extrapolation magnitude and worsening direction checks provide no convincing reason to pay for that pilot. Spatial has isolated small improvements, not a consistent four-cell story. Same stage cost and library bytes as §1D at both scales; potential CPU solve cost brings no evidenced benefit.

### Successor-aware whitening as the next expensive closed-loop experiment

The idea was to change AWM's positive-pair target from the immediate 35-dimensional normalized head `h_r` to `[h_r, sqrt(.5)h_next, .5h_nextnext]`; terminal rows repeat their own head. It never masks candidates and uses only its own library. This addresses a real label ambiguity: immediate-action pair RMS on π0.5-l10 is `.2969/.1654` at 50/500, but those pairs' next-head RMS is `.4569/.3765`; GR00T-l10 is `.3294/.1854` versus `.4927/.3861`.

However, most apparent benefit shrinks under **whole-episode holdout from metric fitting and candidates**. Fixed PCA still uses the deployed library's unsupervised basis. Held-out next-head prediction RMS from the retrieved kernel:

| Cell | 50: ordinary → future-label metric | 500: ordinary → future-label metric |
|---|---:|---:|
| π0.5-sp | .44819 → .43655 | .33522 → .33252 |
| π0.5-l10 | .38289 → .38326 | .31024 → .30903 |
| GR00T-sp | .45612 → .44846 | .37880 → .37743 |
| GR00T-l10 | .44994 → .44578 | .38227 → .38043 |

At 500 the reduction is only `.00120–.00270σ`; π0.5-l10 at 50 slightly worsens. The CPU diagnostic is useful, but this does not establish the dynamic generalization needed to prioritize another metric arm. Spatial-50 might still benefit; that remains unknown. No future labels would be required online, so bytes and IR would be identical to §1D at both scales. I am declining the closed-loop proposal on value-of-information grounds, not declaring a new settled negative from action error.

The strongest deliverable here is the sampling method and its concrete manifests. The control-step library is a bounded, falsifiable experiment in temporal representation, with an explicit geometry-only ablation and full accounting of the evidence still missing.
