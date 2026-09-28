**R5 ideation C — treat the library as a versioned model component**

Linked artifacts sit beside this report. Source paths starting `r02/`–`r05/` are shorthand for `exp/offline_search/rounds/`; `closed_loop/` denotes `exp/offline_search/closed_loop/`. Measurements were made with CPU affinity 10–13,54–57, single-thread BLAS, and CUDA disabled. No policy, simulator, server, remote worker, or GPU was started. Source data and existing code were not changed. The absent `/dev/shm/offline_search_store` was replaced by the read-only cold store. [RUN.md](RUN.md) gives reproduction commands; [TABLES.md](TABLES.md) contains the longer numeric tables.

I recommend **two proposals**, ranked by expected improvement of the closed-loop SR/IR frontier: **(1) successful-episode MISS growth, primarily for an initial 50-episode bank; (2) retain all rows but compile a compact, task-partitioned resident library**. This is a research-priority ranking, not an established ranking of SR. The second proposal has stronger engineering evidence but changes search cost, which the primary IR currently excludes. I do not recommend a hard usage coreset or a hard scene prior.

The main limitation is consequential: the population mixed-run logs do **not** contain the visual keys needed to reproduce insertion and subsequent retrieval. I measure actual MISS supply and simulate frozen-fit growth from recorded **full-inference** trajectories with complete inputs. I do not present that simulation as mixed-policy replay or as a prediction of success.

**1. Measured facts driving the proposals**

**Cost and reference behavior.** Binding source: `rounds/r05/FINDINGS.md`, under `exp/offline_search/`. Every decision represents five controls:

`IR = 0.152 × vision_decisions / decisions + 0.848 × policy_MISSes / decisions`.

Every MISS is charged at full inference. No K2 discount, free teacher, or free warm-up collection is assumed. Search time is reported separately. The denominator must use actual executed controls if request length changes. The primary coefficients are the owner's π0.5 basis; applying them to GR00T is a reference accounting convention, not a measured GR00T latency ratio.

The relevant π0.5 K7 anchor-tail measurements were independently reproduced from accepted logs by [audit_logs.py](audit_logs.py):

| Cell / initial bank | Successes / episodes | Decisions N | Vision V | MISS M | SR | IR |
| --- | --- | --- | --- | --- | --- | --- |
| libero_10 / 50 | 403/500 | 31,186 | 18,566 | 5,554 | .806 | .241512987 |
| libero_10 / 500 | 440/500 | 28,888 | 16,547 | 3,959 | .880 | .203280809 |
| spatial / 500 | 491/500 | 10,630 | 5,643 | 596 | .982 | .128235560 |

The same K7 phase-B2 controller scores .700@.267979316 at 50 and .862@.177620813 at 500 on libero_10. This supports investigating density; it does not establish that arbitrary additional rows reproduce a 500-library fit. The original AWM comparison also changes PCA/whitening and kref (5 versus 8). FINDINGS additionally reports l10 inference L=10 at .904@.5 versus L=5 near .85@1: chunk execution is a separate control effect and must stay fixed in a growth comparison. Below, growth keeps anchor-tail B1 and its full chunks.

**1a. What the logs actually support.** [log_snapshot.json](log_snapshot.json) captures byte limits at **2026-09-28 00:25:34.732343 UTC**. [log_audit.json](log_audit.json) covers 107 arms, 327 decision files, **2,147,966 raw decisions**, and **2,116,242 accepted, deduplicated, non-conflicting decisions**. Two files grew during reading; only complete lines inside their initial byte limits were admitted. Accepted episodes follow the journal's accepted/status/no-error rule. Entire UIDs with conflicting duplicate decisions were excluded: 306 in `r3mx_p_l10_ev_h70`, 60 in `r3mx_p_sp_awm_h50`. The reference arms above are unaffected.

Of **315,808 raw MISSes**, 315,314 have `a_exec`, but that JSON field is the **5×7 executed head**, not the full 10×7 or 16×7 chunk. Only 50,373 MISSes have `robot_state`. **Zero JSON decisions have `key_v0` or `key_v1`.** There are only two complete input NPZs, both `r03_smoke/r3s_p_sp_b0q50`: 40 decisions and 30 MISSes across two tasks. [input_archives.json](input_archives.json) records their fields and paths; [summary.json](summary.json) records counts. They cannot establish an across-episode learning curve within a task.

The source confirms this distinction: `closed_loop/plugin.py:after_infer` writes `a_exec[:5,:7]`; `on_executed` has the complete policy chunk in memory; `--os-log-inputs` separately enables full key/chunk archives. Joining a mixed MISS to an offline query by task/init/step would substitute a different closed-loop state and is invalid.

Most JSON `topk` fields contain ten members: **2,036,287 decisions** have length ten. R4 additionally supplies full `rows`/`weights` on 354,829 decisions. Accordingly, pooled log coverage below means **reported selected members**, not “all historical top-16.” It is already a useful lower bound on required row support. Exact-ish full-16 offline replay is measured separately.

**1b. Natural MISS growth is much slower than collecting full episodes.** [memory_yield.py](memory_yield.py), [focused_usage_yield.json](focused_usage_yield.json):

| Existing controller | All MISSes in 500 deployment episodes | MISSes in successful episodes | Successful MISSes / deployment episode | Failed-episode MISSes |
| --- | --- | --- | --- | --- |
| π0.5 l10 guard, 50 | 6,664 | 2,129 | 4.258 | 4,535 |
| π0.5 l10 guard, 500 | 2,983 | 1,228 | 2.456 | 1,755 |
| π0.5 l10 K7 tail, 50 | 5,554 | 2,146 | 4.292 | 3,408 |
| π0.5 l10 K7 tail, 500 | 3,959 | 2,287 | 4.574 | 1,672 |
| π0.5 spatial guard, 50 | 1,617 | 732 | 1.464 | 885 |
| π0.5 spatial K7 tail, 500 | 596 | 487 | .974 | 109 |

With successful-episode admission, 500 l10 deployments increase the initial bank from 2,640 to **at most 4,786 rows**, before deduplication; the 500-episode bank has **29,472 rows**. Filling that **26,832-row gap** at the observed 4.292 rows/episode would take **6,251.63 deployment episodes**. Counting every MISS instead gives 2,415.56 episodes. Neither number is a forecast of equal SR: the arriving rows are concentrated in difficult states, and their rate changes as the controller changes. For spatial, the 50-library guard's successful yield implies **6,756.15 episodes** to fill its 9,891-row gap. There is no observed mixed GR00T yield from which to estimate a corresponding horizon; pure-cache deployment produces zero policy MISS samples.

For l10 tail/50, the completion-ordered successful-MISS counts are **276, 423, 1,091, 2,146** after 50,100,250,500 completed episodes. This order is task-heavy and affected by parallel completion; it must not be interpreted as a balanced deployment learning curve. The policy-input gap prevents measuring the visual novelty of those samples. As a limited redundancy check, only **210/2,870 consecutive-MISS pairs** have both head sigma-RMS ≤.1 and state motion below the current-library tenth percentile; at 500 it is **74/1,175**. Even these are not safe visual duplicates. Repeated actions alone are insufficient for deduplication.

**1c. Frozen-fit growth proxy, at both scales and both models.** [library_study.py](library_study.py), eight `library_<cell>_<scale>.json` files:

* Source: actual recorded `queries/<cell>_inf` trajectories, in balanced `(init,task)` order, inits **0–24**, all ten tasks. All source decisions are policy calls, including failed episodes. Only successful episodes publish rows, after completion.
* Evaluation: separate inits **25–49**, both inference and cache streams. Density evaluation samples steps 1,6,11,…; no source episode contributes evaluation rows. Source and big-library file paths have zero overlap in every cell, though they use the same benchmark's initialization distribution.
* Representation: the saved, own-library AWM fit, frozen. Query/library codes use its 136-dimensional main metric. For the 50-start comparison, the 500-bank rows are projected through the **same frozen 50 fit**, so the distance units match. This reference does not refit or improve the deployed 50 model.
* Diagnostic admission removes a row only when its nearest admitted code is within the base-library LOEO tenth-percentile distance, its head RMS is ≤.1 sigma, and its state distance is below the base-library tenth percentile of consecutive motion. These are diagnostic thresholds, not tuned SR thresholds. Final proposal defaults to exact duplicate removal because the measured near-dedup saving is small.
* The metric is mean nearest distance `D1` and mean sixteenth-neighbor distance `D16`, on the **main/stale geometry**. It omits early/fresh selection, synthesis, guards, and policy-induced trajectory changes. “Gap closure” is `(D50 − Dgrown)/(D50 − Dbig_under_50_fit)`, not accuracy or success.

| Cell | Start | Paid policy calls at 250 source episodes | Admitted rows at 250 | 50-start D1 gap closed at 50 / 100 / 250 source episodes | 500-start D1 reduction at 250 |
| --- | --- | --- | --- | --- | --- |
| π0.5 spatial | 50 | 5,310 | 5,221 | 54.02% / 78.89% / 118.33% | — |
| π0.5 spatial | 500 | 5,310 | 5,166 | — | 3.2443% |
| π0.5 l10 | 50 | 14,849 | 10,384 | 26.24% / 38.08% / 62.01% | — |
| π0.5 l10 | 500 | 14,849 | 10,378 | — | 1.5927% |
| GR00T spatial | 50 | 5,638 | 5,058 | 43.84% / 66.76% / 106.64% | — |
| GR00T spatial | 500 | 5,638 | 5,029 | — | 2.7126% |
| GR00T l10 | 50 | 14,463 | 10,916 | 29.79% / 47.06% / 71.10% | — |
| GR00T l10 | 500 | 14,463 | 10,990 | — | 1.3572% |

These are the held-out **cache-stream** results; both streams and exact distances are in the JSONs. At 250 source episodes, 50-start D16 gap closures are respectively **135.0040%, 53.9384%, 101.2462%, 60.2464%** in π-sp, π-l10, G-sp, G-l10 order. Values exceeding 100% mean denser geometry than that particular big-bank reference under the small-bank fit, not superior control. Near-dedup rejects only **1/5,222, 97/10,481, 8/5,066, 219/11,135** successful source rows at 50; the 500-start rejection counts are **56,103,37,145** over those same denominators.

The l10 50-start proxy needed 10,384 admitted rows for 62% nearest-distance gap closure. Natural tail/50 produces only 2,146 successful MISSes in 500 deployments. Substituting the latter distribution for the former is unmeasured. **There is no defensible estimate of the number of deployments needed to match 500-library closed-loop behavior.** The measurable conclusion is that 50–100 ordinary deployments do not recreate a tenfold larger bank, while targeted growth may still repair a small number of consequential gaps.

Collection is expensive if forced: adding the measured **14,849 full-policy collection decisions** before another 500 episodes at the existing l10 tail/50 operating point gives a decision-weighted lifecycle IR of **.486169740**, even before hypothesizing growth benefit. The steady-state baseline is .241512987. Arithmetic is in [report_tables.py](report_tables.py). Natural MISS recording adds no inference calls; its existing calls still belong in lifecycle cost.

**1d. There is little evidence for a small behavior-preserving row coreset.** Pooled HIT usage from [audit_logs.py](audit_logs.py):

| Cell | 50: reported row union / all rows | 500: reported row union / all rows | 500: union plus two `next` successors |
| --- | --- | --- | --- |
| π0.5 spatial | 1,018/1,018 | 10,750/10,909 | 10,856 |
| π0.5 l10 | 2,640/2,640 | 29,425/29,472 | 29,465 |
| GR00T spatial | 1,063/1,063 | 11,179/11,751 | 11,418 |
| GR00T l10 | 2,645/2,645 | 28,813/29,631 | 29,247 |

Pooling many methods is intentionally conservative. Restricting to the target controller does not solve generalization: l10 K7-tail/500 uses **25,970/29,472** rows overall. A union learned from inits 0–24 contains 22,834 rows but misses at least one reported member on **5,327/12,691** held-out HIT decisions. Spatial tail/500 uses **8,233/10,909**; its training union misses members on **1,628/5,003** held-out HITs. At l10 tail/50 the overall union is all 2,640 rows; the 2,639-row training union misses members on **2/12,667** held-out HITs. These are support checks, not rerun SR.

The separate full-16 replay reuses `r04/ideation_A/anchors_*.npz` and its documented scalar checks: 640/640 top1 matches, 638/640 complete top16 matches, maximum valid-action difference .000298366 in the original batched replay. Train usage on both query streams at inits 0–24; retain the most frequent rows **within each task**; evaluate cache-stream inits 25–49:

| Cell | Scale | Full-16 query support retained with 50% of rows | With 90% of rows | Kernel weight retained with 90% |
| --- | --- | --- | --- | --- |
| π0.5 spatial | 50 | .074519 | .636686 | .948779 |
| π0.5 spatial | 500 | .220145 | .852327 | .989400 |
| π0.5 l10 | 50 | .150788 | .666897 | .959341 |
| π0.5 l10 | 500 | .185468 | .808867 | .978579 |
| GR00T spatial | 50 | .157396 | .599292 | .954871 |
| GR00T spatial | 500 | .158245 | .828450 | .987310 |
| GR00T l10 | 50 | .165859 | .710465 | .962908 |
| GR00T l10 | 500 | .185892 | .804874 | .979823 |

High retained weight does not establish equal success, especially around phase transitions. Worse, even preserving every selected row does not prove equal selection: `r02/g1_awm/awm.py:_dist` uses `d/median(d)` over **all task candidates** after a MISS, before adding continuity. Removing unused rows changes that median; removing state-nearest rows changes confidence; recomputing calibration changes the judge. A strict equivalent implementation must preserve these computations, not just the old top16.

“Trap row” is not an intrinsic bad-row label. Defining a trap association as at least three consecutive HITs with identical top1, the 50-bank trap-associated row counts are π-sp **260**, π-l10 **1,186**, G-sp **145**, G-l10 **1,235**; **every one** also appears as a selected member in a successful episode. At 500, the respective trap/also-successful-member counts are **197/183, 2,812/2,584, 192/124, 2,261/1,683**. This is association, not action-level causal credit. Original failed-library rows are also widely selected: **526/572, 6,622/6,656, 1,520/1,936, 7,022/7,592** respectively at 500. Neither failure association nor low usage justifies deleting a row.

**1e. Task routing is useful; a persistent hard scene prior is poorly supported.** AWM already searches only `tasks[q.task_id]`. The new scene diagnostic ranks that task's library episodes by the **step-zero early-metric distance to their first row**, retains `ceil(number_of_episodes/2)`, and asks whether it contains later full-16 selections. Only the episode's first observed vision is used to form the prior. Held-out cache results are episode averages:

| Cell | 50: full16 / top1 recall | 500: full16 / top1 recall |
| --- | --- | --- |
| π0.5 spatial | .000000 / .685018 | .004835 / .703481 |
| π0.5 l10 | .011997 / .700554 | .019543 / .625412 |
| GR00T spatial | .000167 / .664645 | .002506 / .695688 |
| GR00T l10 | .010591 / .711797 | .030788 / .658756 |

At 50 this retains approximately 59–60% of task rows because most tasks have five episodes and the rule keeps three; at 500 it retains 48.7–50.1%. Even that generous small-bank shortlist loses much of the later kernel. The prior can order prefetches; it should not exclude candidates or survive a task/episode reset as a retrieval restriction. No reliable detector of within-episode scene change is established here; ordinary bounded vision anchors remain necessary.

Cross-task evidence is from `r04/ideation_C/REPORT_2.md` and its scripts, not recomputed as an SR claim. On sampled l10 cache queries, other-task candidates win only **359/8,177 (50) and 192/8,177 (500)** for π0.5; **307/8,078 and 247/8,078** for GR00T. On π0.5/500 winning-donor cases, own-task versus donor action error is .6479 versus .8910; for GR00T it is .8297 versus 1.0213 at 50 and .5755 versus .8005 at 500. These action diagnostics are not an SR ranking, but give no reason to reopen unrestricted donors. No cross-task donor is included in the proposals.

**1f. Preserve rows; remove representation overhead first.** [memory_yield.py](memory_yield.py) and [memory.json](memory.json) count distinct saved AWM ndarrays. The compact column changes only full action storage from H×32 to **H×7**, retaining the tail and every other array. It is a byte census/design target, **not a saved compact implementation or measured GPU allocation**. MB is decimal.

| Cell | Scale | AWM pickle MB | Arrays with full valid actions MB | Deployed pkl MB |
| --- | --- | --- | --- | --- |
| π0.5 spatial | 50 / 500 | 21.341661 / 46.993114 | 20.316804 / 36.075280 | 430.792483 |
| π0.5 l10 | 50 / 500 | 24.631208 / 94.143229 | 21.984220 / 64.662300 | 1103.155631 |
| GR00T spatial | 50 / 500 | 22.249307 / 58.156565 | 20.541648 / 39.346128 | 429.351282 |
| GR00T l10 | 50 / 500 | 26.672701 / 117.303700 | 22.433720 / 69.885168 | 1068.314575 |

The two PCA means/bases alone cost **17,039,360 bytes**, independent of row count. Therefore halving a small bank cannot halve the deployed model component. Do not store only a 5-step head: that destroys anchor-tail's payload.

The relevant judge has additional arrays. [guard_memory.py](guard_memory.py), [guard_memory.json](guard_memory.json), measured on the actual K7 tail fits:

| Cell / scale | Existing pickle bytes | Existing ndarray bytes | With both action copies packed to H×7 |
| --- | --- | --- | --- |
| π0.5 l10 / 50 | 32,704,527 | 32,685,868 | 27,405,868 |
| π0.5 l10 / 500 | 142,348,562 | 142,321,580 | 83,377,580 |
| π0.5 spatial / 500 | 66,500,552 | 66,473,674 | 44,655,674 |

Both `.base.act` and `.C.act` are padded copies. Packing while retaining the existing remaining arrays yields **2,086 variable bytes per row**, plus roughly 21.899 MB fixed arrays. Thus a compact full-row bank is already a small fraction of the 431/1103/429/1068 MB deployed references. Server model weights, per-connection history, graph workspaces, serialization metadata, and growth archives are additional; these tables do not hide them inside “library bytes.”

K9's rectangular allocation uses `ceil64(max(1.25*max_task_rows,max_task_rows+64))` slots **for every task**. At π-l10/500 that is **59,520 slots for 29,472 rows**. Applying the same spare rule separately per task needs **37,184 slots**. At 50 the corresponding counts are **5,120 versus 3,648 for 2,640 rows**. GR-l10 is **57,600 versus 37,376** at 500 and **5,760 versus 3,648** at 50. Spatial counts, including the case of no saving at 50, are in TABLES.md. These are allocation arithmetic, not measured latency gains.

**Growth capacity is a separate requirement.** [capacity_growth.py](capacity_growth.py) replays successful-MISS arrival counts from actual K7-tail logs, before visual dedup. The current l10/50 uniform capacity 512 overflows at completed episode **51**; task-specific 25%/64-row spares overflow at **21**. Observed 500-episode additions per task are `[287,160,197,332,36,162,207,302,255,208]`. Rounded capacities sufficient for this particular trace are `[576,448,512,640,256,384,448,576,704,512]`; these are not a worst-case bound. Neither l10/500 nor spatial/500 exhausts its existing K9 capacity in the same replay. A 25% spare allocation is not a general online-growth design.

**Numerical and GPU evidence is inherited, not newly measured.** `r04/k9_gpu_retrieval/REPORT.md` reports same-stage-1-graph retrieval overhead approximately .36–.63 ms, compared with K8 CPU search 1.2–2.2 ms. At 67.52 ms full-call cost these add **.005331754–.009330569 per vision decision** versus **.017772512–.032582938**. They are separate from primary IR. K9 reduces key D2H traffic from 131,328 bytes to roughly 1.4 KB of output, but is not wired into serving.

K9's float32 top1 agreement is at least **99.903568%**; largest non-step-zero chunk difference is **2.39089e-5**. Early expanded squared distance is ill-conditioned: maximum full-chunk difference **.0124661475**; float64 still gives **.0124055743** against the deployed CPU reference. Its strict 1e-4 gate therefore fails. The 32/32 successful in-place append/address tests append duplicates and establish pointer stability, **not useful learning from real deployment**. `gpu_awm.py` currently asserts π0.5 and computes features, not the complete K7 blind controller.

**Information provenance.** Every starting PCA, whitening matrix, and judge artifact here is fitted on its own deployed 50 or 500 library. The growth simulation freezes them. The 500-candidate distance reference for a 50 start is an explicitly external diagnostic comparison, never injected into its fit. Usage-ranked coresets are fitted on traces beyond the deployed library: **borrowed big-library information**, at both scales. Any thresholds, capacity priors, or calibration chosen by fitting these historical outcomes must carry the same label. The growth proxy's source trajectories are also external to the initial bank; deploying its prebuilt grown bank without paying for/identifying acquisition would be **borrowed big-library information**. Real samples generated and charged during a future deployment become part of that deployment's enlarged bank; report its actual row count and provenance, not “still a 50-episode library.”

**2. Ranked proposals and decisive pilots**

**Proposal 1 — Successful-episode MISS growth with a frozen representation**

**Pitch and hypothesis.** Let existing full-cost policy calls supply independent action targets where a small bank needs help. Publish successful episodes' MISS samples between episodes; keep anchor-tail and the guard intact. Hypothesis: a modest number of well-placed samples lowers repeated interventions at 50 without needing a full 500-library replacement. Expected value at 500 is much smaller, consistent with the density proxy, but actual SR/IR is unknown until paired rollout. This is a T1 library update using the existing T1 fit, not an external learned model.

**Algorithm and interfaces, sufficiently specific to implement.**

1. Start with the exact own-library K7 `VisionConfirmedBlindMixedJudge` and `BlindAWM(serving="anchor_tail", budget=1, gates="budget_only")`, guard-only MISSes, full policy calls. Keep kref=5 for the initial 50 fit and 8 for 500 throughout the first growth arm. Every episode starts with vision. Do not enable sparse-bank phase continuation merely because the row count rises.
2. At the real policy execution callback, collect `(episode_uid, task_id, decision_step, key inputs or their frozen PCA projections, rs[:8], full policy H×7 chunk, fit_id)`. The insertion key is the observation that produced the MISS, not the following state. Preserve the pre-action query input until `on_executed`; blind HITs never fabricate keys. The full predicted chunk is a policy target; only its first five controls have necessarily been physically executed.
3. Keep samples provisional until `client_episode_end(success=True)`. Abort/disconnect/incomplete episodes publish nothing. Failed episodes go to a quarantine archive and are excluded from the first pilot's active growth, rather than declared intrinsically useless. Do not delete original failed-library rows. Default exact dedup requires matching task, frozen representation/state, and **full valid action**, preserving a deterministic earliest ID. Do not dedup by head alone or merge disagreeing actions. Near-dedup from §1c is an optional measured-size variant, not necessary initially.
4. Exclude successful cache segments from the default insertion source. Their actions are already generated by the bank; reinserting them changes neighbor multiplicity without adding an independent teacher. If later tried, restrict to actual vision HITs and keep a separate provenance arm. Never count such segments as free newly observed policy targets.
5. On successful completion compute `ep_len = number of decisions`, `progress = step/max(ep_len-1,1)`. Preserve the actual decision step. Link `next/prev` only when **both stored samples are consecutive decisions in that same episode**. A missing edge means *unknown continuation*, not *terminal*. Add explicit `terminal_known/is_terminal` and `next_valid` fields. Existing rows retain their old semantics; a sparse new row triggers a terminal guard only when its endpoint is known. Phase methods must request vision at missing edges. Anchor-tail may use the new row's full predicted tail without inventing a neighbor observation.
6. Prepare Z, early norms, normalized camera projections and heads using **frozen** PCA, task centers, whitening, sigma, scales, confidence maps, and guard thresholds. Append with monotonically increasing stable IDs; no in-episode PCA/whitening refit. New rows intentionally alter candidate distances and the full-pool median, so old confidence calibration may drift even though its coefficients are frozen. Log provenance share, predicted confidence, guard reasons, and realized V/M by bank version.
7. Bind an immutable bank version to each episode. Publish only completed episodes into the next version; active episodes keep their version. On CPU, extend the base task tables, action table, wrapper `C` metadata, and registered action lookup coherently. In a harness evaluation, construct the grown snapshot before `Method.fit`, register its actions/progress/task IDs using `ctx.register_library`, and return that library name and stable row IDs in `Result`. `Method.reset` remains per-episode; never make offline results depend silently on harness iteration order. A separate chronological acquisition driver builds snapshots.
8. The live plugin needs explicit collection and publication hooks around `on_executed`/`client_episode_end`, version-aware `PluginStorage` resolution, and per-connection version binding. Existing K6 shared arrays are immutable and cannot be resized casually. GPU append uses preallocated fields, validates the entire batch, waits for readers, copies every field, then publishes valid/count/version. An episode-specific valid-count cutoff must hide newer rows from older episodes. Reserve growth headroom from a declared memory budget; on exhaustion queue insertion or build/recapture a new epoch, never overwrite a live row.

The light acquisition record can retain 128 PCA floats (512 B), eight state floats (32 B), full valid actions (280 B π0.5 or 448 B GR00T), identifiers and outcomes. Keep raw keys only in an explicitly budgeted archive if future PCA refitting is wanted; two float32 raw keys alone cost **262,144 B per row**. Frozen projections support later metric/calibration work but cannot reproduce arbitrary future PCA bases.

**Fit/calibration variants.** Primary arm freezes all fit objects. A separate `growth_recal` arm may recompute candidate-dependent scales and LOEO confidence on the committed bank at an epoch boundary, still freezing PCA/whitening; compare it to the same grown snapshot with frozen calibration to isolate a method effect. New sparse samples cannot supply nonexistent consecutive transitions to threshold fitting. Use only real observed adjacency, otherwise retain base thresholds. A later PCA/metric refit needs an entirely rebuilt bank, corresponding recalibration, and a new graph version. It is not in-place append. No cutoff is selected by an SR sweep on evaluation inits.

**Working forecasts, not estimates from offline error.** These are intentionally modest, falsifiable assumptions after about 500 natural deployment episodes, with no added acquisition calls. The numeric IR ranges hold vision share fixed; measured changes in vision share add `.152 Δv`.

| Cell / start | SR forecast | MISS-share assumption | IR forecast |
| --- | --- | --- | --- |
| π0.5 l10 / 50, tail control | .816–.846 versus .806 | decrease 1–2 percentage points | .224553–.233033 versus .241513 |
| π0.5 l10 / 500, tail control | .870–.890 versus .880; may be no benefit | decrease 0–.5 points | .199041–.203281 |
| π0.5 spatial / 50, tail control | static SR +0–2 points; static tail SR is unmeasured | decrease 0–1 point | static IR minus 0–.00848; absolute value unknown |
| π0.5 spatial / 500, tail control | .982–.986 versus .982 | decrease 0–.25 points | .126116–.128236 |
| GR00T, either suite, 50 and 500, existing pure-cache controller | no change: no MISS supply | zero inserted policy samples | no change; all-vision AWM reference IR .152 |

For the known spatial/50 **B0 guard** reference .888@.266449, the same 0–1-point MISS reduction would give .257969–.266449; this is a different control baseline and is not an absolute forecast for unmeasured tail/50. A mixed GR00T growth controller requires its own judged-MISS baseline and calibration; no SR gain or natural-growth horizon is claimed for it. The GR00T full-policy proxy demonstrates potential geometric coverage only, at paid acquisition cost.

Downside includes lower SR, more guards, and higher IR from bad multiplicity or calibration. Forecast ranges are hypotheses, not confidence intervals. FINDINGS places the l10 single-run noise floor near 3 percentage points, so a 1-point apparent gain is not confirmation.

**Bytes and cost.** At both starting scales retain the complete bank. Under the measured compact K7 array layout, l10/50 plus its observed 2,146 successful MISS rows would occupy **31,882,424 bytes**, versus 27,405,868 initially and the deployed 1,103,155,631-byte pkl. L10/500 plus 2,287 rows would occupy **88,148,262 bytes**; spatial/500 plus 487 rows **45,671,556 bytes**, versus the 430,792,483-byte deployed spatial pkl. These are conditional storage projections at 2,086 B/row, excluding new provenance flags, headroom, graph workspaces and archives. GR00T stays at the full retained-bank sizes in §1f unless a separately paid acquisition experiment is authorized. Every natural or forced policy call remains charged in lifecycle IR.

**Cheapest diagnostic and kill criterion.** First record only actual MISS frozen projections, state and full chunks plus end outcomes during the coordinator's next ordinary reference run. Replay admission into frozen snapshots and inspect task coverage, duplicate multiplicity, missing-edge semantics, capacity and guard/confidence distributions on held-out episodes. The current report already establishes the data-format blocker and count budget; no additional full-model calls are necessary just for recording. Stop before rollout if version isolation or terminal/edge semantics fail. In the paired full pilot, reject for an SR loss exceeding 3 points, or for IR increasing by >.01 without a credible SR gain. Do not promote solely from a density gain; require reproducible frontier improvement, such as ≥.01 IR saving with a paired SR interval excluding a loss worse than 3 points.

**Closed-loop pilot.** The coordinator runs it; no workers were started here. Main arms are `r5c_p_l10_{50,500}_tail_static` and `r5c_p_l10_{50,500}_tail_growth_frozen`. Use two acquisition/evaluation folds: acquire under the static reference on all ten tasks × inits 0–24, freeze and evaluate 25–49; reverse halves for the second fold. Each evaluation init sees growth from the opposite half only. Match seeds, execution length, method, guard and bank version within each pair. Together the two evaluation folds cover **500 paired task/inits per arm per scale**. Charge both acquisition folds separately and report lifecycle as well as evaluation IR. This tests a 250-episode acquisition horizon; it must not be relabeled the 500-deployment forecast above.

A 100-pair collapse screen uses both folds' evaluation inits `{0,5,10,15,20,25,30,35,40,45}` over all ten tasks, after excluding the acquisition half for each particular frozen bank. Full 500-pair evaluation decides modest effects. If l10 supports growth, replicate the same arms on spatial at both scales, obtaining the missing static tail/50 baseline. Optional variants are frozen-vs-recalibrated snapshots and success-only-vs-quarantined recovery samples; each changes one component. GR00T first needs a matched static mixed baseline; its pure-cache arm is a zero-growth negative control.

**Layer attribution.** Synthesis unchanged; method fixed in the primary arm; library gains real policy samples; control/vision schedule unchanged except endogenous guard responses, whose policy calls count. Recalibration is explicitly a method-at-fixed-grown-library ablation. Paid collection is a control cost, not a library-only gain.

**Proposal 2 — Compile the complete task library, with compact full chunks and explicit versioned capacity**

**Pitch and mechanism.** Prune padding, duplicate payloads where identical, and rectangular allocation waste before pruning demonstrations. Use the known task ID to address its resident partition. Keep all candidate codes, heads, state distances, statistics and topology, so rare phases remain available. Capture retrieval alongside stage 1 once numerical acceptance is met. This targets the measured 1.2–2.2 ms search component and data transfer; it does not predict a primary-IR improvement by itself. Fit tier remains T1; the storage/compiler itself introduces no additional learned fit.

**Algorithm and required changes.**

1. `fit` loads or builds the same own-library fit. Store full valid H×7 actions in float32, preserving all ten π0.5 or sixteen GR00T steps. Keep the logical row IDs and all candidates. An internal gather/synthesis adapter expands to the Method API's `(H,32)` output at the boundary; padded dimensions never enter distances, calibration, or cost diagnostics. Validate the valid outputs and postprocessing on both models. If `.base.act` and `.C.act` can share one immutable packed array, prove their index/normalization identity before eliminating a copy; the byte targets above conservatively retain both judge copies.
2. Partition by the already available task ID. Use per-task fixed blocks or a small number of capacity buckets, with explicit offsets/counts and stable logical IDs. The current AWM already routes by task, so this is a storage/execution optimization, not a new semantic prior. At task/episode switches clear anchor, progress and history state and perform the required vision anchor. A scene prior may prefetch payloads but cannot mask candidates or change ranking. No donor task is added.
3. Preserve full-pool medians and state minima. Do not claim an exact coreset by retaining only top16 payloads while recomputing statistics on a smaller code bank. If a future hot-payload cache is used, keep full geometry and transparently fetch any missing action; count the latency. The small complete banks here give no measured reason to add this complexity initially.
4. In `closed_loop/plugin.py`, add an opt-in packed backend and GPU graph slot integration around stage-1 output. Bind graph input/output buffers and task/version selection explicitly. Stage-1 keys stay on device; transfer the chunk and compact verdict inputs once into pinned host output. Keep the current host HIT/MISS scheduler, complete K7 guards, blind budget, and action broadcast semantics. K9 is not a drop-in final judge. Maintain per-connection device vision history and dense state/guard counters; copying previous raw keys back through the host every decision would defeat its traffic saving. Keep those per-connection bytes separate from bank storage.
5. Keep float32 codes/actions and the existing float64 kernel/calibration arithmetic initially; disable TF32 as in K9. Avoid fp16/bf16 library quantization until independently studied. Define deterministic ties by ascending stable row ID, including the rank-16 boundary and newly appended rows. CPU pure AWM's old `argpartition` boundary behavior is not guaranteed identical; either preserve legacy behavior or introduce a separately identified CPU canonical-tie reference before comparing GPU. Never conflate that method change with storage packing.
6. First integration may route **step zero through the legacy CPU retrieval**, after its vision anchor, because K9's large mismatch is concentrated there. Account for its key copy and search cost. A later stable early-distance formulation must be compared against a matching CPU formulation and then the legacy controller; float64 alone is not an equivalence fix. For other decisions require replay of selected IDs, weights, valid actions and final verdicts, not top1 agreement alone. Empirical tolerances are not universal certificates.
7. For growth, freeze representation and existing calibration, use `copy_` into spare fields, publish validity last, and serialize against readers. Include version cutoffs for active episodes. New capacity or a new fit means a new graph epoch; preserve old buffers until their readers finish. If full capacity is reached, defer growth while continuing the current controller. Never evict a rare row as an allocation side effect.

**At 50 and 500.** The intended SR change is **zero**, and `Δv=Δm=0` gives **ΔIR=0** on the primary basis. For π0.5 tail controls, the reference points remain .806@.241513 (l10/50), .880@.203281 (l10/500), .982@.128236 (spatial/500); spatial/50's tail point remains unknown. For an all-vision AWM parity arm, R2 reference SR is π-sp **.800/.954**, π-l10 **.630/.768**, G-sp **.888/.966**, G-l10 **.552/.706** at 50/500, with owner-basis IR .152. Numerical drift could change these results; zero change is the engineering hypothesis, not an observed packed/GPU rollout result.

Once the GPU integration is numerically acceptable, inherited K8/K9 timings suggest **search-inclusive** savings of approximately `v × .00844` to `v × .02725` in full-call units, before any step-zero fallback. These bounds combine benchmark ranges rather than constitute a paired latency experiment. There is no GPU latency measurement here for a GR00T port: K9 currently accepts only π0.5. GR00T's memory targets are measured CPU array counts; its runtime savings remain unknown until the coordinator measures them.

**Bytes.** Full-row AWM compact targets at 50/500 are **20.316804/36.075280 MB π-sp**, **21.984220/64.662300 MB π-l10**, **20.541648/39.346128 MB G-sp**, **22.433720/69.885168 MB G-l10**, against the deployed references in §1f. K7's relevant targets are **27.405868/83.377580 MB** for l10 and **44.655674 MB** for spatial/500. A sparse-bank K7 spatial target must be measured once that exact fit exists. These sums exclude capacity padding and GPU workspaces; K9's existing float32 resident allocations are, for example, **34.132 MiB** l10/50 MixedJudge and **174.882 MiB** l10/500. New packed graph residency must be measured, not inferred to equal the CPU array sum.

**Cheapest diagnostic and kill criterion.** CPU-only packed-vs-original replay across both query streams, both models and both scales is the first implementation gate. Require identical selected IDs/verdicts and unchanged valid actions under the same arithmetic; isolate any changed reduction order. Then let the coordinator test graph replay/address invariants and final judge parity, particularly step zero, exact ties, invalid capacity slots, and an append during another connection's episode. Reject a semantic packing change or an unexpected controller-cost shift. Do not promote the existing unmodified K9 prototype as a strict replacement: its published early-action gate already fails. Reject the GPU integration if the actual end-to-end search-inclusive saving disappears after fallback and synchronization.

**Closed-loop pilot.** Compare `r5c_<p|g>_<sp|l10>_<50|500>_awm_cpu` against `_awm_packed` on all eight cells. After CPU parity, compare π0.5 `_awm_packed` versus `_awm_gpu` and the same K7-tail CPU/GPU pair at both scales; spatial/50 needs a static K7 baseline. The collapse screen is all ten tasks × inits `{0,5,10,15,20,25,30,35,40,45}`. Full validation uses all ten tasks × inits 0–49, **500 paired episodes**, fixed seeds and identical policy-call settings; repeat an unresolved l10 difference rather than interpreting a sub-3-point change. GR00T GPU arms wait for a validated H=16 port and a measured cost basis. Variants are packed CPU, packed GPU with legacy early path, and a separately named canonical-early-distance arm. They do not change synthesis, candidate rows, or policy-call pricing.

**Layer attribution.** Synthesis, method, library content and control remain fixed in the storage/compiler arm. Only representation/deployment cost changes. Canonical ties or stabilized early arithmetic constitute a separate method change, and must not inherit an “exact engineering change” label automatically.

**3. Rejected ideas**

**Hard frequency/“trap” pruning.** At 50 all rows are selected somewhere; at 500 the pooled reported union already covers 95.13–99.84%, before missing historical ranks 11–16 are considered. A half-sized usage coreset retains entire held-out kernels on only .074519–.165859 of queries at 50 and .158245–.220145 at 500. Trap-associated rows overwhelmingly overlap successful selections, and deleting even never-selected candidates can change fresh normalization. No equal-SR smaller row bank is established. Prefer complete-bank representation packing. If a frequency order is nevertheless deployed, label its fit **borrowed big-library information**.

**A hard episode-long scene shortlist, or object-name cross-task routing.** Keeping roughly half the task's library episodes chosen at its first vision gives at most .011997 full-kernel recall at 50 and .030788 at 500 in this diagnostic. Cross-task nearest candidates rarely win and often worsen the diagnostic action comparison. Keep task routing, reset on task/episode switches, and use scene priors only for prefetch ordering with full fallback. These measurements do not rule out every possible soft prior; they reject this inexpensive hard prior as a justified R5 default.

**Append every executed cache/MISS chunk immediately and continually refit.** L10 tail/50 supplies **3,408 failed-episode MISSes versus 2,146 successful ones**; at 500 it is **1,672 versus 2,287**. Immediate unfiltered insertion would heavily represent failed tails, while cache self-insertion adds no independent teacher. Missing visual keys and full chunks prevent retrospective safety assessment; 50-bank capacity fails after 51 completed episodes in the success-only count replay already. In-place refitting also invalidates K9's frozen-code contract. These facts do not prove every failed-episode sample harmful—existing failed-library rows are widely used—so quarantine and a separate recovery-sample experiment are appropriate. They do rule out treating indiscriminate online append/refit as a free, already-supported route from 50 to 500.

The deciding missing evidence is a versioned, full-input acquisition log from natural MISSes followed by held-out paired rollouts. Until then, growth is a focused hypothesis for the sparse library; complete-bank compact residency is the supported engineering direction; aggressive semantic pruning and persistent hard scene restriction lack support.
