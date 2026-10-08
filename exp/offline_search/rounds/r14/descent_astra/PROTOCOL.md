# Frozen per-task knob descent protocol

Frozen UTC: 2026-10-05T04:13:48.459717+00:00  
America/Chicago: 2026-10-04T23:13:48.459717-05:00

Registered before reading any descent-root outcome. Only handback, selection/cap metadata,
existing allocation files and source code were read. No root journals/summaries, test-A data,
run polling or launches. `protocol.json` is the machine-readable registration; hashes bind
the input design and library estimates. Never edit these frozen files.

## Primary decision and tolerances

Use **5 percentage points, paired point-estimate noninferiority** as the PRIMARY development
selection rule; also evaluate 0 and 2 pp without choosing among them after outcomes.
For each task use all 30 frozen paired B identities, reference its same-init pure C10 successes.
Start `c10`, then visit `ir45 → ir40 → ir32 → ir25 → off`. A level passes iff
`100*(candidate successes - pure successes) >= -tolerance_pp*n`. Stop at the **first failure**,
select its predecessor; if all pass select off. Equality at the boundary passes. Improvements
pass even for tolerance zero; this is no degradation, not a literal equality test.
Non-monotone recovery below a failed level is reported and ignored for selection; do not skip
levels, smooth SR, or choose the globally lowest passing level. C10 passes by construction.

One outcome changes task SR by 3.3333 pp. At 30 trials, allowed net losses for 0/2/5 pp
are [0, 0, 1]; hence 0 and 2 pp give identical task tables. Five pp permits
one net loss and is the only requested nonzero per-task tolerance resolved by this sample.
This recommendation balances development sensitivity and quantization; it is not evidence
of population noninferiority. Report gains and losses separately, including offsetting gains.

For UNIFORM apply the identical first-failure point rule at the cell's 300 same-init episodes
(equal task weights), using the cell C10 reference. Net-loss budgets are
[0, 6, 15]. This aggregate rule can hide task regressions; the task rule
cannot. Do not force the chosen uniform or per-task table to have identical measured SR.
Emit complete string task IDs 0..9 for every cell in `levels.json`, and exactly the same
cells mapping to one level in `uniform.json`. Allowed spellings:
`off, ir25, ir32, ir40, ir45, c10`. Primary files use 5 pp; `*_tol{0,2,5}pp.json` preserve all
alternates. These are inputs to the existing emitter, not authorization to run it or read A.

## Uncertainty and data discipline

Let gains be `(candidate=1,pure=0)`, losses `(candidate=0,pure=1)`. Report a conservative
paired 95% one-sided lower bound `CP_low(g,n,.025)-CP_high(l,n,.025)`, and a 95% two-sided
interval using four CP tails of .0125. A simultaneous lower band uses tail `.05/(2*200)`
for all 4 cells × 10 tasks × 5 lower levels. Cell bounds average ten task bounds, retaining
the fixed task mixture. A confidence sensitivity table uses the same first-failure rule with
the simultaneous lower band; it never replaces the registered primary selection.
Even zero discordances at n=30 gives a marginal lower bound of -11.57 pp and a
simultaneous lower bound of -25.89 pp. Conservative confidence selection can
therefore stay at C10; these data cannot usually certify small tolerances. No optional
stopping on partial data, sequential significance claims or post-hoc rule changes.

Require all four cells, eight arms per cell, exact selected B identities and 30 episodes/task.
Follow collector semantics: last accepted terminal error-free journal record per UID;
normal task failures count, execution errors do not. Only that accepted attempt's decisions
count; last record per UID/step in sorted shard/line order. Check UID pool, contracts,
frozen fits, served-subset exclusions, contiguous steps and summary counts. Abort on missing,
extra, malformed, unpaired or inconsistent data; no imputation or complete-case selection.
Snapshot/hash every read input and reject changes during analysis. No refitting. B identities
overlap prior development selections; this is development reuse, not held-out validation.

## IR savings and the library prediction

Use fixed owner cost IR `(cV*V+cM*M)/N`: π0.5 coefficients .152/.848, GR00T .148/.852.
Count accepted full-policy/full-vision decisions at five nominal controls; also retain the
original stage-ledger IR and check its counts. IR means normalized compute, not raw call rate.
Stitch selected whole task trajectories from the uniform-level arms; report selected SR,
IR, signed absolute and relative IR savings versus chosen uniform, and cost/episode.
This B-selected stitched estimate is optimistic and assumes task-local controller equivalence;
it is not an independently executed mixed-policy result. Negative savings remain negative.

Within each cell compare library `priority.z` to the selected ordinal level
`off=0, ir25=1, ir32=2, ir40=3, ir45=4, c10=5`: Spearman (average ties) and Kendall tau-b;
higher priority predicts higher needed inference. Undefined ranks with constant levels are
NA. For each frozen ir32/ir40 cap vector independently, map each finite K to the nearest
`log1p(K)` among the five finite uniform caps; ties choose higher inference. Finite caps
cannot predict off. Report exact and within-one agreement, absolute/signed ordinal errors
and confusion tables for every tolerance. No fitted or outcome-tuned mapping. These caps
are budget allocations, not calibrated tolerance boundaries; agreement is a diagnostic.
Priorities are library-derived, but cap bias used earlier B state-path calibration; do not
describe the full cap estimate as purely library-only. Summarize all cells and both targets,
not only favorable ones; there is no binary post-hoc 'matches' threshold.

Compare actual task_periodic with periodic at ir32/ir40 on identical inits: paired SR delta,
gains/losses and intervals, achieved IR and IR gap. Treat abs(IR gap) <= .01 as approximately
matched; always disclose the gap. Additionally compute an exactly IR-standardized expected
mixture of the two adjacent uniform periodic arms bracketing allocation's achieved IR,
sorting all four periodic arms by achieved IR (ties favor higher inference). Solve the
ratio-of-sums mixture weight in `protocol.json`; no extrapolation if no bracket. Same-init
fractional mixture outcomes permit a paired comparison but are not an observed deterministic
uniform policy. Report this distinction even if nominally matched; never relabel a nominal
target as measured IR or claim causal SR benefit from interpolation.

Use 5,000 within-task paired bootstrap draws, seed 20261005, for descriptive savings and
mixture-delta intervals, holding selected tables and mixture weights fixed. These conditional
intervals omit selection uncertainty and do not establish independent validation. With only
ten tasks, rank estimates and tied/discrete level agreements are fragile; no rank p-value
or out-of-sample prediction claim. A later independent evaluation is outside this analysis.

## Later execution

Write rerunnable, read-only B analysis under this directory. On explicit coordinator resume,
apply this exact registration and emit `DESCENT.md` with a plain Chinese owner section first
(no agent codenames), all task levels per cell, actual savings versus uniform under the same
SR rule, library agreement, then detailed audits. Do not run that analysis in the freeze turn.
No final result tables or selected emitter files are fabricated while outcomes are unavailable.
