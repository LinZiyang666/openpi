# Data and identification

All reported outcome comparisons use **inits 20–29 only**, 100 task/init pairs per arm. Existing detector/threshold fitting used A/CU inits 0–19 with init-group OOF selection; the fitted weights and thresholds are reused unchanged. No parameter was fit or tuned on the current evaluation outcomes. The choice of the single additional variant is exploratory screen-informed selection and is not an independent test of its benefit.

Inputs: accepted-attempt client journals and decision JSONL in exactly `r09_astra_round2b`, `r09_astra_round2b_g`, and `r09_fable_r3c`; the already discovery-filtered compact and 2b score NPZs for the two long-50 cells; frozen fit artifacts and demonstration-library arrays. No aggregate run summary, raw server stdout, arbitrary run-root search, locked-root path, or init 30–49 outcome/trajectory is used.

`tools/safe.py` extracts identity before JSON payload decoding. Only accepted journal attempts are joined to server traces; duplicate accepted decisions and gaps are errors. The NPZ reader accesses task/init first, rejects any archive containing an inadmissible identity, then selects 20–29 before loading payload members. Raw query metadata for CPU probes is framed object-by-object and admitted before deserializing; only task 0/1, init 0, first 24 rows are materialized. Original `QueryCell.__init__` is bypassed. Serving/planning metadata is separately filtered to 20–29 and stripped to identity fields.

Machine-readable results and pair-level evidence: `results/analysis.json`. Reproduction: `tools.analyze`; no fitting or launching. The analysis performs no task-wise parameter selection.

## Same-run paired accounting

Success is the accepted client terminal verdict. If L is actual fresh looks, C actual policy calls, and D actual five-control decisions, aggregate owner IR is `(a*L + (1-a)*C)/D`, with a=.152 for π0.5 and .148 for GR00T. It is a ratio of pooled counts, not a mean of episode ratios. Blind policy tails are not extra calls. Slightly-over-.50 P10 IR arises from final odd decisions.

| Model | Arm | SR | Measured IR | Policy calls | Wins/losses vs its own cache | Exact paired p |
|---|---|---:|---:|---:|---:|---:|
| π0.5 | cache | .690 | .076462 | 0 | — | — |
| π0.5 | burst1 | .710 | .088115 | 91 | +6/−4 | .753906 |
| π0.5 | burst3 | .770 | .102873 | 199 | +8/−0 | .007813 |
| π0.5 | latch | .800 | .114701 | 287 | +11/−0 | .000977 |
| π0.5 | random3 | .740 | .094793 | 141 | +9/−4 | .266846 |
| π0.5 | P10 | .890 | .503906 | 2,838 | +25/−5 | .000325 |
| GR00T | cache | .570 | .074268 | 0 | — | — |
| GR00T | burst1 | .610 | .082785 | 73 | +6/−2 | .289063 |
| GR00T | burst3 | .630 | .095613 | 179 | +7/−1 | .070313 |
| GR00T | latch | .640 | .116408 | 350 | +9/−2 | .065430 |
| GR00T | random3 | .640 | .091749 | 148 | +10/−3 | .092285 |
| GR00T | P10 | .840 | .503332 | 2,946 | +35/−8 | .000042 |

Paired p is exact two-sided McNemar/binomial on discordant pairs; it is descriptive and unadjusted across these exploratory old arms. `analysis.json` additionally records 20,000-draw paired init-cluster bootstrap intervals (the ten init IDs resampled jointly across fixed tasks). The cache reference is from the same run/fleet as its candidate, never the other fleet's cache.

## Measured takeover rescue, distinguished from total paired gain

Define a post-alert paired rescue as: same-run cache fails, this candidate actually emits an `r9b_start` alert, this candidate succeeds. The denominator is **all cache-failing pairs that actually alert in this candidate**, not all alerts or all cache failures. This is an observed conditional rescue proportion, not a randomized causal treatment effect; alerting itself is path-dependent.

| Model | Response | Alerted episodes | Alerted cache failures | Post-alert rescues / denominator | Unalerted paired wins | Calls per alerted episode, mean |
|---|---|---:|---:|---:|---:|---:|
| π0.5 | burst1 | 33 | 26 | 1/26 = 3.85% | 5 | 2.76 |
| π0.5 | burst3 | 28 | 24 | 1/24 = 4.17% | 7 | 7.11 |
| π0.5 | latch | 31 | 25 | 5/25 = 20.00% | 6 | 9.26 |
| π0.5 | random3 | 44 | 19 | 6/19 = 31.58% | 3 | 3.20 |
| GR00T | burst1 | 42 | 33 | 6/33 = 18.18% | 0 | 1.74 |
| GR00T | burst3 | 42 | 33 | 7/33 = 21.21% | 0 | 4.26 |
| GR00T | latch | 41 | 32 | 8/32 = 25.00% | 1 | 8.54 |
| GR00T | random3 | 44 | 28 | 8/28 = 28.57% | 2 | 3.36 |

For latch, π0.5 alerts in six cache-success pairs and harms none; GR00T alerts in nine cache-success pairs and harms two. Success after any alert is 11/31 and 15/41 respectively; that broader number includes cases the cache already solved. Relative to all cache failures, actual post-alert rescues are 5/31 and 8/43. Latch reaches its 12-call cap in 20/31 and 16/41 alerted episodes.

The π0.5 unalerted wins are material. Client overrides match between cache and latch, and initial robot states match in all 100 pairs, but the first served action head is bit-identical in only 55/100 pairs (GR00T 99/100). This establishes that paired rollouts are not exact common action prefixes; it does not identify the cause of divergence. Do not attribute the six unalerted wins to takeover. No adjustment that subtracts “noise wins” yields a causal estimate either.

Direct candidate comparisons reinforce the limitation:

| Comparison | π0.5 paired delta / +− / p | GR00T paired delta / +− / p |
|---|---|---|
| latch vs burst3 | +.03 / +5−2 / .453125 | +.01 / +5−4 / 1.0 |
| latch vs random3 | +.06 / +8−2 / .109375 | .00 / +5−5 / 1.0 |

Random3 was matched to the three-call training budget, **not to latch's realized closed-loop calls**. Latch costs 287 vs 141 π0.5 calls and 350 vs 148 GR00T calls here. The timing/response comparison is therefore not budget-matched evidence of detector superiority.

## Alert timing: factual closed loop versus old offline paths

| Model | Offline cache-path first alert median | Closed-loop latch first alert median | Offline / online alerted episodes | Both / online-only / offline-only | Offline replay / measured latch calls |
|---|---:|---:|---:|---:|---:|
| π0.5 | 58 decisions / 290 controls | 58 / 290 | 28 / 31 | 24 / 7 / 4 | 313 / 287 |
| GR00T | 58 / 290 | 62 / 310 | 43 / 41 | 33 / 8 / 10 | 419 / 350 |

Among pairs alerting in both, online minus offline first-alert median is zero for both models; its mean is +1.75 decisions for π0.5 and −1.70 for GR00T. Aggregate median movement and matched-pair median are different statistics. Burst1/burst3 timing and remaining-decision summaries are in the JSON.

Replaying each deployed 2b gate on **its own logged scores and fresh-look grid** reproduces every call and start bit exactly (maximum error zero for all six detector/response cells). Before first treatment, alternative response lengths share the same gate-entry rule on that particular factual prefix. After treatment, later scores and rollout lengths depend on the selected response, so continued factual-score replay is not counterfactual success or call-cost identification. The older offline compact cache episodes are separate rollouts; matching task/init alone does not make them exact online prefixes.

## What can be inferred inside the leading stack

The recorded controls reproduce π0.5 .920 @ .161636 (2,759 looks, 550 calls, 5,480 decisions) and GR00T .890 @ .182512 (2,881 looks, 724 calls, 5,716 decisions). Each contains the imported only-no-progress guard and the half-strength correction active in `CorrectedCacheJ.os_synth`.

The learned detector requires current camera projections and trailing camera displacements. These fresh keys/projections are absent from the standard r3c logs. Library top rows, normalized robot state, and five-control served heads do **not** reconstruct those missing features or the full proposed corrected chunk at policy decisions. No imputation, task-specific surrogate, or full-trajectory reuse is called exact. Corrector actions also change the detector input distribution; it retains old weights without recalibration.

The latch wrapper performs an extra CPU metric pass to recover the original detector's normalized retrieval-distance input, without another camera look. This CPU overhead is not represented by owner IR; end-to-end latency remains unmeasured.

Pace entry is reconstructible: at a fresh decision `step<=80`, test `step - library_step[top1] >= 12`. The π0.5 reconstructed start agrees with the logged escalation start on every triggered trace. For the extra bounded variant, the guard/corrector and executed action source remain equal until a takeover verdict actually changes a cache/policy decision; guard-overlapping calls can postpone or eliminate this first divergence.

| Model | Pace-alert episodes | Median first pace alert | Logged paths with >12 calls after trigger | First action divergence under pace12 | Control successes in divergent set | Unchanged-path successes | Conditional pathwise SR bounds |
|---|---:|---:|---:|---:|---:|---:|---:|
| π0.5 | 15 | 48 | 8 | 6 | 2 | 90 | [.90,.96] |
| GR00T | 22 | 45 | 11 | 21 | 11 | 78 | [.78,.99] |

For π0.5, the first twelve takeover calls match persistent escalation. Starting at fresh call 13, divergence occurs only where the frozen guard would permit cache. Two of eight long-takeover paths remain unchanged because the guard covers the later calls. For GR00T, divergence is the first guard-permitted cache decision within the twelve-call window after the pace trigger. A trigger entirely covered by guard calls produces no action change.

The bounds hold only on those factual paths under the same exogenous randomness and deterministic controller mapping. They do not estimate a new stochastic fleet run, latency-induced effects, or post-divergence trajectories. Numerical pace timing is earlier than old latch timing on different populations; no within-episode earlier-alert benefit is identified. Individual prefix rows are saved only to audit the aggregate calculation, never to route a method.

Cross-batch overlap gives little support for latch complementarity: the old π0.5 latch's eleven paired wins overlap two leading-stack failures, but **none of its five actual post-alert rescues does**. GR00T has zero overlap for either the nine paired wins or eight actual post-alert rescues. Different batches, policies and paths make this descriptive only; it does not rule out new rescues inside the stack.

## Integrity and limits

No new fit uses evaluation data. Guard/corrector code is imported, never copied or patched on disk. The original corrector's task-specific fitted heads remain as the explicitly prescribed frozen baseline; no added takeover branch or parameter depends on task identity/difficulty. Both model-level thresholds and all budgets are fixed before emission. Within each model all three arms share client overrides, library, policy-tail settings and evaluation manifest.

Nine unit tests and twelve CPU plugin probes establish local numerical/lifecycle correctness, not closed-loop efficacy. Forced new arms actually start at decision 2 after each reset. Control forced probes use the unchanged r3c debug hook and do not change production settings. The initial local latch probe exposed a nonexistent optional base-metric method; the adapter was corrected to the actual frozen base's `min(d)/median(d)` diagnostic, then the full new-arm probes passed and diagnostic parity was tested. Final sources are hashed after this correction.

The extra store adapter is necessary because the stock planner otherwise hashes raw multi-init episode metadata. Only parser-filtered identity maps are hashed and shipped here, under a run-specific remote namespace. No holdout files, server logs with unadmitted identities, remote reads, process controls, launches, git operations or outside-root writes are part of this package's work.
