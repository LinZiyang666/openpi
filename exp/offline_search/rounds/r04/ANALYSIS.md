# R4 analysis — look once / act several steps, cheaper vision, randomized call value (closed loop, 2026-09-27/28 CDT)

Analysis agent (fable), R4 of `logs/offline_search_exploration.log.md` (§8 protocol, §9 owner rulings 9–15, §10 ledger).
Cells: **π0.5-l10 / π0.5-spatial** (mixed and pure cache) and **GR00T-l10 / GR00T-spatial** (pure cache only) at the
**50-** and **500-episode** library. Every arm is 500 paired episodes (10 tasks × inits 0–49); every paired number is on the common (task, init)
set with the exact two-sided McNemar p on the discordant pairs and a 10,000-draw multinomial bootstrap 95 % interval of
ΔSR; Wilson 95 % CIs for single SRs. **Owner IR is recomputed from the ledger `v`, `m` in every `summary.json`
(IR = .152·v + .848·m for π0.5 K=10, ×5/L for L-step arms; wrist-only stage 1 priced by `cost_table_owner.json`,
ratio-transfer ASSUMPTION; GR00T .148·v + .852·m)** — the watcher/summary "IR" of R4 arms is the eager ledger and is
reported separately. Search cost (K8 controlled single-thread p50 per decision, owner denominator 67.5 ms) and the K9
in-graph increment are third and fourth columns, never mixed into the primary IR. Everything was computed read-only
with `taskset -c 26-29,70-73`, ≤ 8 processes, BLAS 1 thread; scripts and every intermediate table are in
`/home/weiland/.claude/jobs/a607dd74/tmp/analysis_r04/` (§12).

**Library scale (owner rule).** *50* = deployed library: 49 / 50 / 50 / 50 episodes, 1,018 / 2,640 / 1,063 / 2,645
entries (π0.5-sp / π0.5-l10 / GR00T-sp / GR00T-l10); *500* = 10,909 / 29,472 / 11,751 / 29,631 entries. Deployed pkl
431 / 1,103 / 429 / 1,068 MB (262 KB/entry). Fit pickles actually served in R4: 21–142 MB (§9).

**Status.** Complete (2026-09-28 follow-up): all π0.5 arms of `r04_frontier`, `r04_cost` (incl. both seeded second
runs `r4f_p_{l10,sp}_inf_s2001`), `r04_blind`, `r04_k7`, `r04_b4w`, `r04_csl`, `r04_rep`, `r04_k5`, and the 8 GR00T
pure-cache blind arms of `r04_gblind` ({ph2, tail2u} × {l10, sp} × {50, 500}; two chains wrote that root concurrently,
so `state/current` is meaningless — every arm has its `.DONE`, its per-arm `summary.json`, and 500 accepted journal
rows). No R5 arm is used for any conclusion; one (`r05_q2/r5q2_g_l10_500_G10`) is cited as forward context only in §11.

---

## 0. Headline verdicts

1. **The noise floor is 1.6 pp of SR per single 500-init l10 run, and it bites.** The stock guard-only 500-library
   arm (`g500`) run three times gives .864 / .850 / .832 (mean **.849**, sd .016); pairs of identical runs are
   discordant on 11–12 % of inits, one of the three pairings is "significant" at p = .044. Against the pooled
   three-run reference the **only R4 SR gains on l10-500 that clear the floor are the execution-length effect
   (pure inference L=10: +5.5 pp [+2.4, +8.7]) and, at the edge, anchor_tail (+3.1 pp [+0.0, +6.3])**. Every other
   R4 arm is within ±3 pp of stock in SR; their value is inference ratio, not success rate (§1).
2. **Execution length is the largest single R4 effect and it is a control effect, not a cache effect.** Executing the
   whole 10-step chunk per policy call lifts l10 pure inference from .851 (three L=5 runs) to .904 / .900 (two L=10
   runs; 52/24 and 57/31 discordant, p .002 / .007), with no change on spatial (4/7, 3/7, 7/7 against the three L=5 runs). The gain has a task
   fingerprint (t8 +10, t9 +5, t2/t3/t6 +4) that the anchor_tail arms reproduce (task-level correlation .87 for K7
   tail, .80 for K1 tail; .35 for phase_particles, .43 for kernel_clock): **anchor_tail's SR advantage over stock is
   mostly "execute the chunk you already have", not better retrieval.** The pure-cache spatial-500 tail arm makes the
   same point with zero policy calls: +2.8 pp over CL2 (21/7, p .0125) at 49 % less IR (§2, §4).
3. **Frontier (owner basis, primary).** l10-500: ph2c .778 @ .091 → wrist .820 @ .165 → **K7 phase B=2 .862 @ .178 →
   K7 anchor_tail .880 @ .203** → L=10 inference .904 @ .500 (stock g500 .849 @ .244 is dominated by both K7 points;
   periodic 12, noprog-4, the K1-guard arms and the two-clock arm are dominated). l10-50: ph2c .620 @ .098 → CL2 .630
   @ .152 → wrist .734 @ .236 → **K7 anchor_tail .806 @ .242** → perk3 .832 @ .429 → L=10 .904 @ .5 (phase B=2 at 50
   .700 is dominated by both tail and wrist). spatial-500: **pure-cache anchor_tail .982 @ .078** dominates every
   other cache point including K7 tail-with-policy (.982 @ .128), and is within the suite's resolution of pure
   inference (three runs .986 / .992 / .994, mean .991; −0.9 pp [−2.2, +0.4] against their per-init mean; 4/9, 3/9,
   7/9 discordant, p .27 / .15 / .80). spatial-50: wrist .924 @ .156 (SR unresolved, §5) → perk3 → V7 h.7 .980 @ .430.
   **GR00T (pure cache only)**: l10-500 tail2u .808 @ .050 → policy .870 @ 1.0 (CL2 .706 and ph2 .724 dominated);
   l10-50 tail2u .570 @ .050 (≈ CL2 .552, n.s.); sp-500 **ph2 .976 @ .078** (above the policy's .940, 29/11, p .006)
   with tail2u .946 @ .052 (−3.0 vs ph2, p .006); sp-50 CL2 .888 stays the SR point, tail2u .864 @ .051 the cost point. Adding K8 search cost (+.015–.03 for full-vision arms, +.012–.02 for blind arms) or
   switching to the eager basis changes no Pareto membership; the K9 in-graph increment adds ≤ .01 (§3).
4. **Blind stepping works only with the vision-confirmed stuck guard and only as a cost lever.** K1's dense guard fired
   "stuck" 1,952× (B=0) vs 699–867× in the three stock runs (2.5× the stock mean) and cost .020–.039 IR on every arm;
   K7 restored the stock rule bit-exactly (B=0 .830 @ .245 is a fourth stock replicate in everything but name). With K7:
   B=2 phase ≡ B=1 phase in SR (.862 / .862, 38/38) at −.048 IR, so **B=2 is the budget**; anchor_tail > kernel_clock >
   phase_particles at B=1 under the K1 guard (.878 > .868 > .838; tail − phase +4.0, p .040) — the proprioceptive phase
   correction that ideation A ranked first is the worst of the three serving rules in the loop. Phase B=2 collapses on
   the 50-episode library (.700; −10.6 pp vs tail, 44/97, p < 1e-4; t7 −17) while anchor_tail holds (.806, +6.6 over
   g50, p .0025): phase advances 16 members along ≈3 effective demonstrations per kernel at 50 episodes and stalls
   (vision-confirmed stuck 2,139× vs 1,079× for tail). Two-clock (blind + periodic MISS) is dominated everywhere.
   Pure-cache blind keeps SR (+1.0 / −1.0 / 0.0 pp, all n.s.) at 36–46 % less IR (§4).
5. **GR00T pure-cache blind reproduces the π0.5 pattern on l10 and inverts it on spatial.** With two blind blocks
   (15 of the 16 synthesized steps executed, 95–98 % of decisions inside 15-step chunks, v .34, IR .050 = one third of
   pure cache) GR00T-l10-500 goes from CL2 .706 to **.808** (93/42, p < 1e-4; +8.4 over phase B=2, 82/40, p .0002),
   closing 62 % of the gap to the policy (.870) with zero policy calls — the same task fingerprint as π0.5 (t8 +23, t6
   +16, t4 +13). At l10-50 it is flat (+1.8 vs CL2, p .47; task swings ±19 cancel), and on **spatial the 15-step tail
   hurts**: −2.0 vs CL2 (8/18, p .076) and −3.0 vs phase (6/21, p .006) at 500, −2.4 (n.s.) at 50, while phase B=2 at
   sp-500 (.976 @ .078) is the best GR00T-spatial point ever measured, 3.6 pp above the policy. Phase collapses at
   sp-50 (−5.4, p .004; t2/t3), as it did for π0.5-l10-50 (§4.6).
6. **Wrist-only keys: no resolved SR effect, −.073 to −.110 IR under the ratio-transfer assumption.** The four cells
   give +3.6 (sp50, p .027), 0.0 (sp500), −0.6 (l10-50), −2.9 vs pooled stock (l10-500, CI [−.061, +.004]); with four
   tests, Holm-adjusted p ≥ .08 for every cell, so the sp50 gain (t4 +11, t9 +10, t6 −7) is a multiple-comparison
   candidate until the queued repeat lands. Exact dummy_cached repricing (bit-exact, K3) moves every full-vision π0.5
   point left by .048·v with zero SR change: K7 tail .203 → .176, K7 phase .178 → .150, pure-cache tail .078 → .053 (§5).
7. **Control-step library is dead**: G .222 / .114 (−55 / −52 pp, p < 1e-4, 94–98 decisions/episode) because it ranks
   by an interpolated offset but serves the un-shifted head, so every decision re-executes already-executed controls;
   GS (aligned splice) is exactly CL2 (+0.4 / −0.2 pp). The temporal-resolution premise buys nothing (§6).
8. **Randomized CALL/CACHE (K5) says guard-forced calls are causally valuable at 500 (ITT ΔY +.034 [+.004, +.066],
   ΔN −1.6 requests, ΔC ≈ 0) and unmeasurable at 50 (ΔY −.002 [−.040, +.032], ΔM +.47).** Q3 found no context with a
   supported saving at either scale (arms_q3 = []): a learned "skip this call" gate has no support in this data (§7).
9. **Four layers.** Library (50 → 500, same method): +13.8 (CL2), +7.4 (K7 tail), +16.2 (K7 phase), +8.6 (wrist) pp on
   l10 — blind phase is the most library-hungry method, tail the least. Control at fixed library: CL2-500 → K7 tail
   +11.2 pp for ΔIR +.051 (22 pp per .1 IR, the most inference-efficient control point of any round), CL2-50 → K7 tail
   +17.6 for +.090. Execution length: +5.5 pp (policy, l10), +2.8 pp (cache, spatial). Cost implementation: dummy
   −.048·v exact; wrist −.073…−.110 assumed; K2 (appendix only) −.029…−.067 with SR indistinguishable (§8).
10. **Engineering.** K6 removed a server-wide lock (K1 arms 37–61 min → 9–14 min per arm, results unaffected); K8
   measured the CPU method at 1.2–2.2 ms per vision decision (PCA of two 32,768-d keys is 31–49 % of it) and the blind
   step at .22 (tail) / .47 (phase) ms; K9 showed GPU-resident retrieval captured with stage 1 adds .36–.63 ms (≥ 99.9 %
   top-1, step-0 chunk drift .0125 from ill-conditioned early distances); K10's policy tail was vetoed by the inherited
   no-progress span on 88 / 94 % of l10 MISSes and never ran in closed loop (§10).

---

## 1. Noise floor first

Three runs of the stock 500-library guard-only arm (`r3mx_p_l10_g500`, `r4rep_p_l10_g500_a`, `_b`; identical fits,
yaml, judge; only server seeds / scheduling differ):

| pair | ΔSR | discordant (S→F / F→S) | SE(Δ) | McNemar p | bootstrap 95 % |
|---|---|---|---|---|---|
| rep a − original | −.014 | 59 / 500 (26 / 33) | .0153 | .435 | [−.044, +.016] |
| rep b − original | −.032 | 56 / 500 (20 / 36) | .0149 | **.044** | [−.062, −.002] |
| rep b − rep a | −.018 | 59 / 500 (25 / 34) | .0153 | .298 | [−.048, +.012] |

SR .864 / .850 / .832: **mean .849, sd .016, SE of the mean .009**. The MISS share moves with it (.102 / .109 / .117;
5.97 / 6.44 / 7.04 MISS per episode). Pure inference L=5 (trace_dual .844, seeded 1001 .848, seeded 2001 .860; mean
.851, sd .008) is discordant on 82–90 inits per pairing (16–18 %, SE .018–.019), the two L=10 runs on 74 (14.8 %),
spatial pure inference (trace .986, seeded 1001 .992, seeded 2001 .994; mean .991, sd .004) on 5–11 inits per pairing
(1.0–2.2 %, SE .004–.007; s2001 − trace +.008, 7/3, p .34; s2001 − s1001 +.002, 3/2, p 1.0). Reading rules used throughout: (i) a single-run l10 ΔSR is a signal only
if |Δ| ≳ 3 pp *and* p < .05 against **each** stock run; (ii) the primary comparison for l10-500 is against the
per-init mean of the three stock runs with an init-bootstrap interval; (iii) spatial differences below 1 pp are
unresolvable (5–14 discordant pairs); the spatial reference used below is the per-init mean of the three runs (.991).

**Which R4 "wins" survive against .849 (l10, 500-episode library):**

| arm | SR | d vs pooled stock | init-bootstrap 95 % | vs .864 / .850 / .832 (p) | verdict |
|---|---|---|---|---|---|
| pure inference L=10 (seed 3001) | .904 | **+.055** | [+.024, +.087] | — | real |
| pure inference L=10 (unseeded) | .900 | **+.051** | [+.019, +.084] | — | real (replicates) |
| K7 anchor_tail B=1 | .880 | **+.031** | [+.000, +.063] | +.016 (.43) / +.030 (.11) / +.048 (.017) | at the edge; only significant vs the low run |
| K1 anchor_tail B=1 | .878 | +.029 | [−.003, +.061] | +.014 (.53) / +.028 (.15) / +.046 (.018) | same |
| awm500_h70 (R3, IR .44) | .872 | +.023 | [−.004, +.049] | +.008 / +.022 / +.040 (.022) | n.s. |
| K1 kernel_clock B=1 | .868 | +.019 | [−.009, +.047] | +.004 / +.018 / +.036 (.041) | n.s. |
| K7 phase B=2 | .862 | +.013 | [−.016, +.043] | −.002 / +.012 / +.030 (.12) | n.s. (= stock) |
| K7 phase B=1 | .862 | +.013 | [−.016, +.041] | −.002 / +.012 / +.030 (.09) | n.s. |
| periodic 8 | .850 | +.001 | [−.027, +.029] | | = stock |
| K1 phase B=2 | .850 | +.001 | [−.030, +.033] | | = stock |
| K1 B=0 (dense guard) | .842 | −.007 | [−.033, +.019] | | = stock |
| K7 B=0 | .830 | −.019 | [−.045, +.006] | −.034 (.043) / −.020 / −.002 | n.s.; a fourth stock replicate (bit-equal rule) |
| periodic 12 | .828 | −.021 | [−.048, +.006] | −.036 (.033) / −.022 / −.004 | n.s. |
| two-clock phase B=2 + k8 | .822 | −.027 | [−.059, +.007] | −.042 (.035) / −.028 / −.010 | n.s., dominated |
| wrist-only guard | .820 | −.029 | [−.061, +.004] | −.044 (.021) / −.030 / −.012 | borderline loss |
| guard noprog-4 | .808 | **−.041** | [−.067, −.015] | −.056 (.001) / −.042 (.013) / −.024 | real loss |

The ledger's earlier readings "K7 ph2g equals g500", "tail +3.1 pp" and "noprog-4 dead" stand; "K7 b0g 3.4 pp below
g500" and "wrist −4.4 pp" were comparisons against the high draw and shrink to −1.9 / −2.9 (n.s.) against the mean.
R3's own frontier point g500 (.864) was a +1.5 pp draw; its cost (.238) was also the lowest of the three (.244 / .251
for the replicates), so **the stock 500-library reference is .849 @ .244** from here on.

---

## 2. Execution-length effect (the R4 confound)

**Policy side.** `r4f_p_l10_inf_k10_L10` (seed 3001, client executes all 10 steps of every chunk, 28.1 decisions /
episode, IR .5 per five controls) vs the three L=5 runs:

| L=10 run − L=5 run | ΔSR | S→F / F→S | p | bootstrap 95 % |
|---|---|---|---|---|
| L10 (3001) − seeded 1001 | +.056 | 52 / 24 | .0018 | [+.022, +.090] |
| L10 (3001) − seeded 2001 | +.044 | 50 / 28 | .017 | [+.010, +.080] |
| L10 (3001) − trace_dual | +.060 | 54 / 24 | .0009 | [+.026, +.094] |
| L10 (3001) − K2 seeded 1101 | +.054 | 52 / 25 | .0028 | [+.020, +.088] |
| L10 unseeded (`r4b3_p_l10_50_inferL10`) − 1001 / 2001 / trace | +.052 / +.040 / +.056 | 57/31, 50/30, 58/30 | .007 / .033 / .004 | |
| the two L=10 runs | −.004 | 36 / 38 | .91 | [−.038, +.030] |
| spatial: L10 − seeded 1001 / 2001 / trace | −.006 / −.008 / .000 | 4/7, 3/7, 7/7 | .55 / .34 / 1.0 | [−.022, +.006] |

Per task (successes of 50, L=10 mean of 2 runs − L=5 mean of 3 runs): t0 +0.2, t1 −0.2, **t2 +3.8, t3 +3.5, t6 +3.7,
t8 +10.2, t9 +4.8**, t4 −2.5, t5 +1.7, t7 +0.5. The mechanism proposed by R5-A (re-planning every 5 steps cancels an
imminent short grasp/place sequence; 41 % of l10 planned tail gripper switches are absent from the next head vs 18 % on
spatial) is consistent with the task pattern and the spatial null; it is not identified by these runs.

**Cache side, separated by construction.** Only anchor_tail changes the executed chunk length (it serves steps 5–9 of
the anchor's own synthesized chunk; 85 % / 81 % / 94 % of decisions in the l10-500 / l10-50 / sp-500 K7 tail arms belong
to 10-step cache chunks, 97 % in the pure-cache spatial tail). phase_particles and kernel_clock re-synthesize a fresh
5-step head from advanced library rows at every blind decision (0 % 10-step chunks), and every policy chunk in every
R4 mixed arm is executed for 5 steps (the anchor is invalid after a MISS; K10's tail never ran). So the R4 evidence
splits as:

| contrast (same guard, same library, same inits) | what changes | ΔSR | S→F / F→S | p | ΔIR owner |
|---|---|---|---|---|---|
| K7 B=0 → K7 phase B=2 (l10-500) | look once, keep the anchor kernel, new 5-step heads | +.032 | 44 / 28 | .076 | −.067 (v 1 → .573, m .109 → .107) |
| K7 phase B=2 → K7 tail B=1 (l10-500) | serve the stored tail instead of re-synthesizing; +1.7 MISS/ep | +.018 | 39 / 30 | .34 | +.025 (m .107 → .137) |
| K7 B=0 → K7 tail (l10-500) | both | **+.050** | 54 / 29 | **.008** | −.042 |
| K7 phase B=2 → K7 tail (l10-50) | same at 50 | **+.106** | 97 / 44 | < 1e-4 | −.026 |
| K7 phase B=2 → K7 tail (sp-500) | same on spatial | +.014 | 14 / 7 | .19 | −.012 |
| **pure cache**: CL2-500 → tail1uc (sp-500, no policy) | look once + whole chunk, zero MISS | **+.028** | 21 / 7 | **.0125** | −.074 |
| pure cache: ph2c → tail1uc (sp-500, same v ≈ .52) | stored tail vs re-synthesis | **+.028** | 20 / 6 | **.0094** | −.005 |
| pure cache: CL2 → ph2c (l10-500 / l10-50 / sp-500) | look once, re-synthesis | +.010 / −.010 / .000 | 42/37, 55/60, 15/15 | .65 / .71 / 1.0 | −.061 / −.054 / −.069 |

Reading. (i) On spatial the whole "tail beats everything" result is obtained **without a single policy call**: the
cache's own chunk executed whole (+2.8 pp, p .01) at the same vision share as the phase arm, i.e. the cache benefits
from committing to a plan exactly as the policy does not (spatial L=10 null for the policy, +2.8 for the cache — the
cache's 5-step re-retrieval jitter is worse than the policy's re-planning jitter on spatial). (ii) On l10 the
execution-length component cannot be isolated from the policy-call component with R4 arms alone: tail1ug has 27 % more
policy calls per episode than K7 phase (7.9 vs 6.2) and K5 says each guard-forced call is worth ≈ +.03–.05 SR at 500.
The task fingerprint argues that most of tail's +3.1 pp over stock is execution length: tail − pooled stock per task
t8 +12.0, t9 +5.3, t7 +3.7, t6 +2.7, t4 −5.7 (corr .87 with the L=10 fingerprint), whereas the arm that buys more
policy calls at the same v (awm500_h70, 19.8 MISS/ep) gains on t9 / t6 / t4 and not t8. (iii) The regression is also
task-specific: **t4 (two-mug placement) loses 5–8 episodes in every tail arm at both scales and does not lose under
L=10 inference (−2.5)** — a cache-tail-specific failure (R5-D reports t4 / t6 as 37 / 97 and 24 / 60 of tail failures)
that no proposed guard reaches. (iv) The direct l10 test — pure-cache anchor_tail at both scales — is queued as
`r05_ptail` (R5 data, not read here): when it lands, CL2 → tail1uc on l10 gives the execution-length effect of the
π0.5 cache with zero policy calls; if it is ≈ +3 pp at 500, tail's entire margin over stock is execution length. The
GR00T analogue is already measured (§4.6): CL2 → tail2u on GR00T-l10-500 is +10.2 pp with zero policy calls.

---

## 3. SR-vs-IR frontier per cell

Owner basis primary (IR per five control steps; pure inference L=5 = 1.0, L=10 = .5). "chord" = straight line from
pure cache CL2 (same library) to the mean of the L=5 pure-inference runs at the arm's IR; "search" = owner IR +
(v·q_vision + (1−v)·q_blind)/67.5 ms from the K8 controlled table (method only, no logging / shadow); "K9" = the same
with the vision-side query replaced by the measured in-graph increment (.40 ms AWM-type, .60 ms MixedJudge-type; the
blind step stays on the CPU, it was not ported). Pareto flags computed over all points of the cell in that basis.

### 3.1 π0.5-l10, 500-episode library (pure cache .768 @ .152; pure inference .851 @ 1.0)

| arm | SR [Wilson] | v | m | MISS/ep | **IR owner** | above chord | Pareto (owner) | +search | +K9 | eager |
|---|---|---|---|---|---|---|---|---|---|---|
| pure-cache phase B=2 (`ph2c`) | .778 [.740, .812] | .597 | 0 | 0 | **.091** | +1.6 | frontier | .106 | .097 | .085 |
| CL2 (R2) | .768 | 1 | 0 | 0 | .152 | 0 | dominated by ph2c | .173 | .158 | .142 |
| CSL GS / G | .772 / .222 | 1 | 0 | 0 | .152 | +0.4 / −54.6 | dominated | .178 / .177 | | .142 |
| wrist-only guard | .820 [.784, .851] | 1 | .123 | 7.5 | **.165** (A: wrist ratio; .256 at full s1) | +5.1 | frontier | .189 | .174 | .162 |
| **K7 phase B=2** | .862 [.829, .889] | .573 | .107 | 6.2 | **.178** | +9.2 | **frontier** | .198 | .186 | .173 |
| **K7 anchor_tail B=1** | .880 [.849, .906] | .573 | .137 | 7.9 | **.203** | +10.7 | **frontier** | .222 | .210 | .199 |
| two-clock phase B=2 + periodic 8 | .822 | .642 | .125 | 7.6 | .204 | +4.9 | dom. (K7 ph2g, tail) | .220 | .210 | .198 |
| K1 phase B=2 (dense guard) | .850 | .586 | .144 | 8.6 | .211 | +7.6 | dom. | .233 | .219 | .207 |
| guard noprog-4 | .808 | 1 | .073 | 4.5 | .214 | +3.4 | dom. | .245 | .223 | .205 |
| periodic 12 | .828 | 1 | .075 | 4.5 | .216 | +5.4 | dom. | .237 | .222 | .206 |
| K1 anchor_tail B=1 | .878 | .585 | .158 | 9.1 | .223 | +10.3 | dom. (K7 tail) | .243 | .230 | .219 |
| K7 phase B=1 | .862 | .695 | .142 | 8.4 | .226 | +8.7 | dom. | .249 | .234 | .220 |
| stock g500 original / rep a / rep b | .864 / .850 / .832 | 1 | .102 / .109 / .117 | 6.0 / 6.4 / 7.0 | .238 / .244 / .251 | +8.8 / +7.3 / +5.4 | dom. (tails) | .269 / .275 / .282 | .247–.260 | .229–.242 |
| K7 B=0 | .830 | 1 | .109 | 6.6 | .245 | +5.3 | dom. | .276 | .253 | .236 |
| K1 kernel_clock B=1 | .868 | .698 | .169 | 9.9 | .250 | +9.0 | dom. (tails) | .274 | .258 | .244 |
| periodic 8 | .850 | 1 | .120 | 7.0 | .254 | +7.2 | dom. | .274 | .259 | .245 |
| K1 phase B=1 / K1 B=0 | .838 / .842 | .705 / 1 | .186 / .140 | 11.1 / 8.3 | .265 / .271 | +5.9 / +6.2 | dom. | .289 / .302 | .273 / .280 | .259 / .262 |
| awm500_h70 (R3, V7 h.7) | .872 | 1 | .341 | 19.8 | .441 | +7.6 | dom. (tails) | .472 | .450 | .435 |
| **pure inference L=10** (2 runs) | .904 / .900 | 1 | 1 | 28.1 | **.500** | — | **frontier** | .500 | .500 | .500 |
| pure inference L=5 (3 runs) | .844 / .848 / .860 | 1 | 1 | 59 | 1.000 | — | dominated by L=10 | | | |

K2 appendix (owner ruling 12, never on the frontier; SR indistinguishable from K10 in every pair: g500_k2 − g500
−.022 p .16 / −.008 / +.010 vs the replicates; perk5_500_k2 − per8 +.002, p 1.0; inference K2 − K10 +.002, p 1.0):
g500_k2 .842 @ .209 (eager .168), perk5_500_k2 .852 @ .247 (eager .186), inference K2 .850 @ .650 (eager .371).

### 3.2 π0.5-l10, 50-episode library (pure cache .630 @ .152; pure inference .851 @ 1.0)

| arm | SR | v | m | MISS/ep | **IR owner** | above chord | Pareto (owner) | +search | eager |
|---|---|---|---|---|---|---|---|---|---|
| pure-cache phase B=2 | .620 [.577, .661] | .642 | 0 | 0 | **.098** | +0.4 | frontier | .112 | .091 |
| CL2 (R2) / CSL GS / G | .630 / .628 / .114 | 1 | 0 | 0 | .152 | 0 / −0.2 / −51.6 | CL2 frontier | .169 / .171 | .142 |
| wrist-only guard | .734 [.694, .771] | 1 | .201 | 13.4 | **.236** (A; .323 full s1) | +8.2 | frontier | .255 | .234 |
| **K7 anchor_tail B=1** | .806 [.769, .838] | .595 | .178 | 11.1 | **.242** | **+15.3** | **frontier** | .258 | .237 |
| K7 phase B=2 | .700 [.658, .739] | .641 | .201 | 13.8 | .268 | +4.0 | dom. (tail, wrist) | .287 | .264 |
| two-clock phase B=2 + periodic 5 | .776 | .713 | .200 | 12.7 | .278 | +11.3 | dom. (tail) | .293 | .273 |
| periodic 6 | .764 | 1 | .161 | 10.4 | .288 | +9.8 | dom. | .306 | .280 |
| guard noprog-4 | .690 | 1 | .167 | 11.4 | .293 | +2.3 | dom. | .318 | .285 |
| periodic 5 (R3) | .792 | 1 | .192 | 11.9 | .315 | +12.0 | dom. (tail) | .332 | .307 |
| stock g50 (R3) | .740 | 1 | .202 | 13.3 | .323 | +6.5 | dom. | .348 | .315 |
| periodic 3 (R3) | .832 | 1 | .327 | 19.2 | .429 | +13.0 | frontier | .446 | .422 |
| V7 h.7 / h.5 (R3) | .816 / .868 | 1 | .343 / .532 | 20.9 / 31.2 | .443 / .603 | +11.0 / +12.1 | dom. | | |
| pure inference L=10 / L=5 | .904 / .851 | 1 | 1 | | .500 / 1.0 | | L=10 frontier | | |

K2 appendix: g50_k2 .732 @ .261 (−.008 vs g50, p .73), perk5_50_k2 .790 @ .248 (−.002 vs perk5, p 1.0).

### 3.3 π0.5-spatial, 500-episode library (pure cache .954 @ .152; pure inference .991 @ 1.0, mean of three runs)

| arm | SR [Wilson] | v | m | MISS/ep | **IR owner** | above chord | Pareto (owner) | +search | eager |
|---|---|---|---|---|---|---|---|---|---|
| **pure-cache anchor_tail (`tail1uc`, stage-1-only server)** | **.982** [.966, .991] | .514 | **0** | **0** | **.078** | +3.1 | **frontier** | .090 | .073 |
| pure-cache phase B=2 | .954 | .544 | 0 | 0 | .083 | +0.3 | dom. (tail1uc) | .097 | .077 |
| wrist-only guard | .974 | 1 | .069 | 1.5 | .117 (A) | +2.1 | dom. | .137 | .114 |
| K7 anchor_tail B=1 | .982 | .531 | .056 | 1.2 | .128 | +2.9 | dom. (tail1uc, same SR) | .144 | .123 |
| K7 phase B=2 | .968 | .545 | .068 | 1.5 | .140 | +1.4 | dom. | .158 | .135 |
| CL2 (R2) | .954 | 1 | 0 | 0 | .152 | 0 | dom. | .172 | .142 |
| periodic 12 / guard-only g500 | .976 / .974 | 1 | .061 / .064 | 1.3 / 1.4 | .204 / .206 | +2.0 / +1.8 | dom. | .224 / .237 | .195 / .197 |
| V7 h.7 on AWM-500 (R3) | .978 | 1 | .337 | 7.2 | .438 | +1.2 | dom. | .469 | .431 |
| pure inference L=10 / L=5 (trace, s1001, s2001) | .986 / .986, .992, .994 | 1 | 1 | 11 / 21.4 | .500 / 1.0 | | frontier | | |

K2 appendix: sp_g500_k2 .974 @ .185 (3/3 vs g500), inference K2 .990 @ .650.

### 3.4 π0.5-spatial, 50-episode library (pure cache .800 @ .152)

| arm | SR | v | m | MISS/ep | **IR owner** | above chord | Pareto (owner) | +search | eager |
|---|---|---|---|---|---|---|---|---|---|
| CL2 (R2) | .800 | 1 | 0 | 0 | .152 | 0 | frontier | .169 | .142 |
| wrist-only guard | .924 [.897, .944] | 1 | .112 | 2.6 | **.156** (A; .247 full s1) | +12.3 | frontier (SR unresolved, §5) | .175 | .153 |
| K1 phase B=2 + dense guard | .884 | .588 | .135 | 3.2 | .204 | +7.2 | dom. (wrist) | .222 | .199 |
| stock g50 (R3) | .888 | 1 | .135 | 3.2 | .266 | +6.2 | dom. (wrist) | .289 | .258 |
| periodic 3 / V7 h.7 / V7 h.5 (R3) | .938 / .980 / .986 | 1 | .317 / .328 / .534 | | .421 / .430 / .605 | | perk3, h.7 frontier | | |

No K7 arm exists at spatial-50 (the tail cell is queued in R5); the only blind point uses K1's dense guard.
K2 appendix: sp_g50_k2 .896 @ .219 (+.008 vs g50, p .62).

### 3.5 GR00T, pure cache only (owner basis IR = .148·v, m = 0; K8 search: BlindAWM 1.20–1.43 ms vision, .22 tail / .47 phase blind)

| cell | arm | SR [Wilson] | v | dec/ep | **IR owner** | Pareto (owner) | +search | eager | vs CL2 of the cell (S→F / F→S, p, boot 95 %) | vs policy (trace_dual) |
|---|---|---|---|---|---|---|---|---|---|---|
| l10-500 | CL2 (R2) | .706 [.665, .744] | 1 | 66.8 | .148 | dom. (ph2, tail2u) | .168 | .130 | — | −.164 (32/114, < 1e-4) |
| l10-500 | phase B=2 | .724 [.683, .761] | .641 | 66.2 | .095 | dom. (tail2u) | .111 | .083 | +.018 (51/42, .41, [−.020, +.054]) | −.146 (28/101) |
| l10-500 | **tail2u (two blind blocks)** | **.808** [.771, .840] | .339 | 60.2 | **.050** | **frontier** | .060 | .044 | **+.102** (93/42, < 1e-4, [+.058, +.146]) | **−.062** (43/74, .005) |
| l10-500 | policy (trace_dual) | .870 | 1 | — | 1.0 | frontier | | | | |
| l10-50 | CL2 (R2) | .552 [.508, .595] | 1 | 76.3 | .148 | dom. | .166 | .130 | — | −.318 |
| l10-50 | phase B=2 | .566 [.522, .609] | .672 | 76.1 | .099 | dom. (tail2u) | .114 | .087 | +.014 (73/66, .61, [−.032, +.060]) | |
| l10-50 | tail2u | .570 [.526, .613] | .338 | 73.0 | **.050** | frontier | .058 | .044 | +.018 (65/56, .47, [−.026, +.062]) | −.300 |
| sp-500 | CL2 (R2) | .966 [.946, .979] | 1 | 22.1 | .148 | dom. (ph2) | .167 | .130 | — | **+.026** (29/16, .073) |
| sp-500 | **phase B=2** | **.976** [.959, .986] | .529 | 21.9 | **.078** | **frontier** | .092 | .069 | +.010 (11/6, .33, [−.006, +.026]) | **+.036** (29/11, .006, [+.012, +.062]) |
| sp-500 | tail2u | .946 [.923, .963] | .350 | 22.3 | **.052** | frontier (cost only) | .061 | .045 | **−.020** (8/18, .076, [−.040, .000]); vs ph2 **−.030** (6/21, **.006**) | +.006 (26/23, .78) |
| sp-500 | policy (trace_dual) | .940 | 1 | — | 1.0 | dominated by all three | | | | |
| sp-50 | CL2 (R2) | .888 [.857, .913] | 1 | 24.6 | .148 | frontier (SR) | .165 | .130 | — | −.052 |
| sp-50 | phase B=2 | .834 [.799, .864] | .604 | 25.6 | .089 | dom. (tail2u) | .103 | .079 | **−.054** (28/55, **.004**, [−.090, −.018]); t3 −14, t2 −11 | |
| sp-50 | tail2u | .864 [.831, .891] | .346 | 25.1 | **.051** | frontier (cost) | .060 | .045 | −.024 (42/54, .26, [−.062, +.016]); vs ph2 +.030 (61/46, .18) | |
| sp-50 | policy (trace_dual) | .940 | 1 | — | 1.0 | frontier | | | | |

Reading. On GR00T-l10 the two-block tail is the whole story: at 500 it is the only cache point above the CL2 → policy
chord (+12 pp above it at IR .050) and it reaches .808 with **no policy at all**; phase B=2 is CL2 at 36 % less IR
(as on π0.5). On GR00T-spatial the ordering flips: phase B=2 is the best point (.976, above the policy), and the
two-block tail loses 2–3 pp at both scales — a 15-step blind commitment is too long for a suite whose episodes are
≈ 110 controls and whose placements are precise (its losses sit on t1 −7 / t0 −2 at 500). A one-block GR00T tail
(10 steps, the π0.5 configuration) was not run; it is the missing point between ph2 and tail2u. Library layer at fixed
method (50 → 500): CL2 +15.4 / +7.8, phase +15.8 / +14.2, **tail2u +23.8 / +8.2** (l10 / sp; all p < 1e-4) — on
GR00T-l10 the long tail is the most library-hungry rule (on π0.5-l10 the one-block tail was the least, +7.4).

### 3.6 Basis sensitivity

- **Search (K8, method only)**: +.017–.031 for full-vision arms (MixedJudge 1.65–2.10 ms), +.012–.022 for blind arms
  (their vision decisions cost 1.7–2.1 ms but 43–49 % of decisions cost .22–.48 ms), +.017–.021 for pure cache. The
  blind arms' cost advantage *grows* slightly with search included (K7 tail .222 vs stock .269–.282). No Pareto
  membership changes in any cell. Native CP1 shadow search (2.1 / 7.4 ms on spatial / l10) is not charged anywhere
  because `--os-no-shadow-native` was on for all R4 arms. GR00T blind arms: +.008–.016 (tail2u .050 → .060, ph2 .095
  → .111; CL2 .148 → .168), so the GR00T-l10 tail keeps a 2.8× cost margin over pure cache with search included.
- **K9 in-graph retrieval**: replacing the CPU vision query by the .36–.63 ms increment brings the search surcharge
  to +.006–.010; the blind step (.22–.48 ms CPU) then dominates the residual and would need its own GPU port.
- **Eager basis (K3 measured, no CUDA graph; s1/s2/s3 = .142/.072/.786)**: K10 arms move by −.005 to −.010 (the stage-1
  share is .142 instead of .152); K2 arms move a lot (g500_k2 .209 → .168, inference K2 .650 → .371) because stage 3 is
  79 % of the eager cost; wrist arms are essentially unchanged (.162 vs .165). Pareto sets are identical to the owner
  basis in all four π0.5 cells.

---

## 4. Blind stepping ("look once, act several steps")

### 4.1 The K1 dense guard and the K7 fix

| arm (l10-500) | SR | v | m | stuck fires (force:1) | no-progress (force:4) | terminal / overtime | MISS-after-MISS | IR |
|---|---|---|---|---|---|---|---|---|
| stock g500 ×3 | .864 / .850 / .832 | 1 | .102 / .109 / .117 | 699 / 809 / 867 | 1,899 / 2,042 / 2,264 | 250 / 135 … | .427 / .452 / .448 | .238 / .244 / .251 |
| K1 B=0 (dense stuck guard) | .842 | 1 | .140 | **1,952** (2.5× stock mean) | 1,785 | 257 / 150 | .530 | .271 |
| K7 B=0 (vision-confirmed) | .830 | 1 | .109 | 601 | 2,278 | 248 / 173 | .420 | .245 |
| K1 phase B=2 | .850 | .586 | .144 | 1,774 | 2,128 | 244 / 158 | .471 | .211 |
| K7 phase B=2 | .862 | .573 | .107 | **484** | 2,278 | 256 / 93 | .349 | .178 |
| K1 tail B=1 | .878 | .585 | .158 | 1,217 | 3,052 | 198 / 98 | .385 | .223 |
| K7 tail B=1 | .880 | .573 | .137 | 253 | 3,456 | 209 / 41 | .297 | .203 |

The K1 guard replaced stock's "state motion < library p10 **and** visual key cosine ≥ p95" by dense proprioceptive
motion alone, so it fires on every intentional hold; K7 requires the same visual confirmation at the two anchors that
bracket the gap (bit-equal to stock on 15,272 all-vision queries, K7 hand-back). Effect at equal serving: K7 − K1 SR
+1.2 (phase B=2, 38/32, p .55), +.2 (tail, 29/28, p 1.0), +2.4 (phase B=1, p .23), −1.2 (B=0, p .51); IR −.033 / −.020 /
−.039 / −.026. The guard is worth nothing in SR and .02–.04 in IR; every K1-guard blind arm is dominated by its K7 twin.

### 4.2 Serving rule, budget and clock

- **Budget**: K7 phase B=1 vs B=2: .862 vs .862 (38 / 38, p 1.0), IR .226 vs .178. Blind runs in K7 phase B=2 reach the
  budget 87 % of the time (5,786 of 6,659 runs are length 2; 873 are cut to 1 by a gate: gripper-ahead 472, low
  motion 219, near-terminal 97, residual 68). **B=2 is free in SR; nothing in R4 tests B=3.**
- **Serving rule at B=1 (K1 guard, l10-500)**: anchor_tail .878 @ .223 > kernel_clock .868 @ .250 > phase_particles
  .838 @ .265. Paired: tail − phase +4.0 (53/33, **p .040**), clock − phase +3.0 (41/26, p .086), tail − clock +1.0
  (39/34, p .64); MISS share .158 / .169 / .186. Ideation A's ranked-first mechanism (per-member proprioceptive phase
  correction, offline error −.022 to −.037 vs the clock) is the worst of the three in the loop; it also looks more
  often (v .705 vs .698 vs .585) because its residual / low-motion gates fire on its own heads.
- **Two clocks** (blind B=2 + global periodic MISS, no guards): l10-500 k=8 .822 @ .204 vs K7 phase B=2 .862 @ .178
  (33/53, **p .040**), vs periodic 8 alone .850 (35/49, p .16); l10-50 k=5 .776 @ .278 vs K7 tail-50 .806 @ .242
  (dominated), vs periodic 5 .792 (p .52), but **+7.6 over K7 phase-50** (90/52, p .0018). The periodic MISS rescues
  phase particles from their 50-library collapse (§4.4) yet stays below the guard-based tail arm at lower cost.
  Dead as a design; the K1 dense guard is not involved here (BlindAWM, no guard), so this is a clean negative.

### 4.3 Realized shares, look reasons, MISS anatomy (decision logs)

| arm | vision share v | HIT anchors followed by a blind run | mean blind-run length | look reasons on vision decisions (share) | blind / ep S \| F | MISS / ep S \| F | MISS share in failed eps |
|---|---|---|---|---|---|---|---|
| K7 phase B=2, l10-500 | .573 | 49 % (6,659 / 13,580) | 1.87 | budget .35, **gripper-ahead .26**, lifecycle .21, noprog-span .11, terminal .04, low-motion .03, residual .01 | 23.5 \| 33.8 | 3.4 \| 23.6 | .52 |
| K7 tail B=1, l10-500 | .573 | **98 %** (12,341 / 12,588) | 1.00 | budget .73, lifecycle .26, noprog-span .005 | 22.9 \| 37.4 | 5.2 \| 27.9 | .42 |
| K7 phase B=2, l10-50 | .641 | 45 % | 1.83 | budget .25, gripper .22, lifecycle .33, noprog-span .12, residual .02 | 22.9 \| 28.5 | 5.5 \| 33.1 | .72 |
| K7 tail B=1, l10-50 | .595 | 97 % | 1.00 | budget .67, lifecycle .32 | 23.3 \| 33.4 | 5.3 \| 35.1 | .61 |
| K7 phase B=2, sp-500 | .545 | 53 % | 1.80 | budget .37, gripper .29, lifecycle .14, terminal .12, noprog-span .06 | 9.9 \| 10.3 | 1.1 \| 12.7 | .28 |
| K7 tail B=1, sp-500 | .531 | 99 % | 1.00 | budget .85, lifecycle .14 | 9.9 \| 15.6 | 1.0 \| 12.1 | .18 |
| pure-cache tail, sp-500 | .514 | 94 % (rest = episode end) | 1.00 | budget .91, lifecycle .09 | 10.1 \| 22.0 | 0 | — |
| pure-cache phase B=2, l10-500 / l10-50 | .597 / .642 | 36 % / 31 % | 1.88 / 1.80 | budget .32 / .25, gripper .49 / .47, terminal .06 / .13, low-motion .09 / .09 | | 0 | — |
| GR00T phase B=2, l10-500 / l10-50 | .641 / .672 | 30 % / 27 % | 1.86 / 1.80 | budget .26 / .22, **gripper .63 / .62**, terminal .04 / .07, low-motion .04 / .05 | | 0 | — |
| GR00T tail2u, l10-500 / l10-50 | .339 / .338 | 99 % / 99 % | 1.97 / 1.97 (97.8 % / 98.0 % of decisions in 15-step chunks) | budget .95 / .96, lifecycle .05 / .04 | | 0 | — |
| GR00T phase B=2, sp-500 / sp-50 | .529 / .604 | 48 % / 36 % | 1.86 / 1.84 | budget .41 / .30, gripper .37 / .42, terminal .11 / .19 | | 0 | — |
| GR00T tail2u, sp-500 / sp-50 | .350 / .346 | 95 % / 97 % | 1.95 / 1.95 (95.0 % / 95.6 % in 15-step chunks) | budget .87 / .89, lifecycle .13 / .11 | | 0 | — |

Facts. (i) The gated phase rule spends a look on the gripper-ahead gate at 22–29 % of its vision decisions and only
half of its HIT anchors ever start a blind run; anchor_tail (budget-only gates) starts a run at 97–99 % of HIT anchors,
which is why both reach the same v ≈ .57 with very different blind-run structure. (ii) Failed episodes are the ones
that run blind more (tail: 37 vs 23 blind decisions per episode) and MISS far more (28 vs 5): the blind stretches do
not hide failures, the guards still catch them; but a failed tail episode contains 28 policy calls and still times out
at 104 decisions (all failures are cap timeouts, as in R3). (iii) MISS-after-MISS falls from .43–.45 (stock) to .30
(K7 tail): with the anchor invalid after a MISS, the next look re-accepts the cache more often. (iv) Per-task v is flat
(.53–.65) in every blind arm; the blind budget does not adapt to task difficulty; per-task m tracks the failure tasks
(t6 .21, t8 .16, t9 .16 in K7 tail-500). (v) On GR00T the gripper-ahead gate is the dominant look reason of the phase
rule (62–63 % of l10 vision decisions vs 22–26 % on π0.5): GR00T's 16-step chunks contain a gripper event far more
often, so only 27–30 % of its l10 anchors ever start a blind run and v stays at .64–.67; the two-block tail ignores
those gates (budget-only) and reaches v .34 with 97–99 % of anchors followed by a full two-block run.

### 4.4 Why phase_particles loses on the 50-episode library and anchor_tail holds

| l10-50 arm | SR | vs g50 (S→F/F→S, p) | per-task net vs g50 | stuck fires | MISS/ep | IR |
|---|---|---|---|---|---|---|
| K7 phase B=2 | .700 | −.040 (53/73, .090) | **t7 −17**, t4 −8, t0 −5, t8 +5, t9 +3 | 2,139 | 13.8 | .268 |
| K7 anchor_tail B=1 | .806 | **+.066** (73/40, **.0025**) | **t8 +19**, t2 +7, t6 +5, t9 +5, t1 +4, t4 −5 | 1,079 | 11.1 | .242 |
| tail − phase | | +.106 (97/44, < 1e-4) | t7 +15, t8 +14 | | | |
| phase 50 → 500 | .700 → .862 | +.162 (37/118, < 1e-4) | t4 +20, t7 +16, t0 +15 | | | |
| tail 50 → 500 | .806 → .880 | +.074 (39/76, .0007) | t0 +14, t4 +13 | | | |

Mechanism (supported by the logs, not proven): phase_particles advances each of the 16 kernel members along its own
demonstration and re-synthesizes a head; at 50 episodes a kernel draws on ≈ 3 effective episodes per task (ideation A:
3.1 at 50 vs 5.1 at 500), so the advanced members are 2–3 demonstrations extrapolated by one or two rows each, and
the re-synthesized head tracks the scene poorly — the vision-confirmed stuck guard then fires **twice as often** as
under tail (2,139 vs 1,079) and the arm spends more MISSes (13.8 vs 11.1 per episode) for a lower SR. anchor_tail
re-serves the chunk already synthesized at the anchor (the same mixture, no per-member advance), so its extrapolation
is one step of the *mixture*, and it is the only blind rule whose library sensitivity (+7.4 pp 50 → 500) is smaller
than the pure cache's (+13.8). The t7 collapse of phase-50 (28 vs 45 successes) is the single largest per-task loss of
any non-dead R4 arm. Verdict: **phase_particles is a 500-library method; anchor_tail is the scale-robust blind rule.**

### 4.5 Pure-cache blind vs R2 CL2, and the spatial-500 zero-policy result, checked hard

| pair | ΔSR | S→F / F→S | p | bootstrap 95 % | IR (owner) | IR change |
|---|---|---|---|---|---|---|
| ph2c − CL2, l10-500 | +.010 | 42 / 37 | .65 | [−.024, +.044] | .091 vs .152 | −40 % |
| ph2c − CL2, l10-50 | −.010 | 55 / 60 | .71 | [−.052, +.032] | .098 vs .152 | −36 % |
| ph2c − CL2, sp-500 | .000 | 15 / 15 | 1.0 | [−.022, +.022] | .083 vs .152 | −46 % |
| **tail1uc − CL2, sp-500** | **+.028** | 21 / 7 | **.0125** | [+.008, +.050] | .078 vs .152 | −49 % |
| tail1uc − ph2c, sp-500 | +.028 | 20 / 6 | .0094 | [+.008, +.048] | .078 vs .083 | |
| tail1uc − K7 tail-with-policy, sp-500 | .000 | 6 / 6 | 1.0 | [−.014, +.014] | .078 vs .128 | zero MISS vs m .056 |
| tail1uc − pure inference s1001 (.992) | −.010 | 4 / 9 | .27 | [−.024, +.004] | .078 vs 1.0 | |
| tail1uc − trace_dual (.986) / L=10 (.986) / K2 (.990) | −.004 / −.004 / −.008 | 7/9, 7/9, 5/9 | .80 / .80 / .42 | [−.020, +.012] | | |
| tail1uc − s2001 (.994) | −.012 | 3 / 9 | .146 | [−.026, +.002] | | |
| tail1uc − per-init mean of the three inference runs (.991) | −.009 | — | — | [−.022, +.004] (init bootstrap) | | |

Checks on `r4b3_p_sp_500_tail1uc`: server launched with `stage1_only=1`, `--stage2-device meta --stage3-device meta`
(policy stages never loaded; `server_*.log`, `launch_*.sh`); 10,593 accepted decisions, `src` ∈ {cache 5,449,
cache_blind 5,144}, **0 policy rows, 0 MISS in all 500 episodes**; blind runs after 94.4 % of anchors (the remaining
305 anchors are episode ends); per-task v .51–.53; 9 failed episodes: t9 ×5, t0 ×2, t3 ×2 (per task vs s1001: t9 −5,
t0 −1, t3 −1, t4 +1, t7 +1). So the claim "pure cache executed whole reaches the policy on spatial" holds at the
resolution the suite allows, with a consistently negative point estimate: −1.0 / −1.2 / −0.4 pp against the three
inference runs, −0.9 [−2.2, +0.4] against their per-init mean (K7 tail-with-policy: identical −0.9 [−2.2, +0.3]; K7
phase −2.3 [−3.9, −0.8] and stock sp-g500 −1.7 [−3.2, −0.3] are resolved deficits), and the deficit is concentrated on
one task (t9, 45/50 vs 50/50 in all three inference runs). Note the spatial policy itself leaves only 4–7 failures, so no 500-init spatial run can resolve
differences below ≈ 1 pp; "indistinguishable" is a statement about power as much as about the cache.

### 4.6 GR00T pure-cache blind: the execution-length pattern, compared with π0.5

GR00T's synthesized chunk is H=16, so `tail2u` (anchor_tail, budget 2, budget-only gates) executes the anchor head and
two stored tail blocks — 15 of 16 steps — before the next look; `ph2` is phase_particles B=2 with all gates. Both are
pure cache (0 policy rows, 500/500 zero-MISS episodes in all eight arms; `src` ∈ {cache, cache_blind}).

| cell | tail2u − phase B=2 (S→F / F→S, p, boot) | tail2u − CL2 | phase − CL2 | per task, tail2u − phase (of 50) | π0.5 counterpart (same contrast) |
|---|---|---|---|---|---|
| l10-500 | **+.084** (82/40, .0002, [+.042, +.126]) | **+.102** (93/42, < 1e-4) | +.018 (51/42, .41) | t8 +19, t4 +16, t6 +8, t1 +4; t7 −5 | K7 tail − phase +.018 (p .34), tail − B=0 +.050 (p .008); t8 +12 |
| l10-50 | +.004 (68/66, .93, [−.040, +.050]) | +.018 (65/56, .47) | +.014 (73/66, .61) | t4 +16, t1 +12, t8 +4; **t9 −19**, t0 −8 | K7 tail − phase **+.106** (p < 1e-4) |
| sp-500 | **−.030** (6/21, **.006**, [−.050, −.010]) | −.020 (8/18, .076) | +.010 (11/6, .33) | t1 −6, t7 −4, t0 −3; t6 +2 | pure-cache tail − phase **+.028** (p .009) |
| sp-50 | +.030 (61/46, .18, [−.010, +.072]) | −.024 (42/54, .26) | **−.054** (28/55, **.004**) | t3 +18, t2 +15; t7 −7, t8 −6 | (no π0.5 spatial-50 blind arm without the dense guard) |

Same as π0.5: (i) blind phase B=2 keeps pure-cache SR at both scales on l10 (+1.8 / +1.4, n.s.) at −33 to −36 % IR;
(ii) executing the stored tail beats re-synthesizing from advanced rows on l10-500, and by a larger margin than on
π0.5 (+8.4 vs +1.8) — consistent with a longer commitment (15 vs 10 steps) buying more on a suite whose policy also
gains from long chunks; (iii) the l10 task fingerprint is the same one (t8, t6, t4 up); (iv) phase collapses at the
small library on the suite where its kernel is sparsest (GR00T sp-50 −5.4, t2/t3 −11/−14; π0.5 l10-50 −4.0 / t7 −17).
Different from π0.5: (v) on spatial the 15-step tail is worse than both phase and CL2 (−3.0 / −2.0 at 500, −2.4 vs
CL2 at 50), whereas the π0.5 10-step tail was the best spatial point (+2.8 over CL2, +2.8 over phase); (vi) at l10-50
the long tail is SR-neutral on aggregate but moves individual tasks by ±16–19 (t4/t1 up, t9/t0 down): with ≈ 3
effective demonstrations per kernel a 15-step replay either matches the demonstration or runs it into the wrong
object for three blocks. Hypothesis (not tested): the 15-step length is the variable, not GR00T — R5-A measured GR00T's
second tail block disagreeing with the replanned head more than the first (gripper-element disagreement 3.1 % / 8.9 %
vs 2.1 % / 6.8 % on spatial / l10), and spatial placements are the precision-limited part of the episode. The missing
arm is a one-block GR00T tail (10 of 16 steps) at both suites; the R5 pure-policy GR00T L=10 control will say whether
the GR00T policy itself has the l10 execution-length effect.

Cost. Owner IR = .148·v: tail2u .050 (all four cells, v .34–.35), phase .078–.099 (v .53–.67), CL2 .148; eager ledger
.043–.046 / .069–.086 / .130; +search .058–.061 / .092–.114 / .165–.168. On GR00T-l10-500 the frontier is therefore
CL2 .706 @ .148 → tail2u .808 @ .050 → policy .870 @ 1.0, i.e. the cache point with **one third** of pure cache's cost
closes 62 % of the SR gap to the policy; on GR00T-sp-500 phase B=2 (.976 @ .078) is above the policy (.940) and above
CL2, so the GR00T-spatial product needs neither a policy nor a long tail.

---

## 5. Cheaper vision: wrist-only stage-1 key

| cell | wrist SR | two-camera guard SR (same lib) | ΔSR | S→F / F→S | p | bootstrap 95 % | per-task net | IR wrist (A) | IR two-cam | IR wrist at full s1 |
|---|---|---|---|---|---|---|---|---|---|---|
| sp-50 | .924 | .888 (R3 g50) | **+.036** | 39 / 21 | .027 | [+.006, +.066] | t4 +11, t9 +10, t1 +3, **t6 −7** | .156 | .266 | .247 |
| sp-500 | .974 | .974 (g500) | .000 | 9 / 9 | 1.0 | [−.016, +.016] | t9 −3 | .117 | .206 | .210 |
| l10-50 | .734 | .740 (g50) | −.006 | 59 / 62 | .86 | [−.048, +.036] | t8 −8, t2 +5 | .236 | .323 | .323 |
| l10-500 | .820 | .864 / .850 / .832; pooled .849 | −.044 / −.030 / −.012; **−.029 vs pooled** | 31/53, 36/51, 42/48 | .021 / .13 / .60 | pooled [−.061, +.004] | t8 −10, t3 −6, t7 −4, t9 +5 | .165 | .244 (pooled) | .256 |

Multiple comparisons: four cells were tested; the two nominal hits are sp-50 (p .027) and l10-500 vs the original
g500 only (p .021). Holm over four: .021×4 = .086, .027×3 = .082 — **neither survives at α = .05**, and the l10-500
loss is n.s. against both replicates and against the pooled reference. The sp-50 effect is concentrated in two tasks
(t4, t9) and reversed on t6 (28 vs 35), the layout-ambiguity task where the third-person camera is what disambiguates
(the 500-library CL2 gets .86 on t6; wrist-50 vs CL2-500 on t6 is −15). Hypothesis (unverified): the wrist key ignores
the object layout that confuses the sparse 50-episode retrieval on t4/t9 and helps there, and loses the layout signal
it needs on t6. The queued wrist-sp50 repeat (in `r05_ptail`) is the test; until then the sp-50 gain is a candidate.
Against pure cache, wrist-sp50 is +12.4 pp over CL2-50 (p < 1e-4) and −3.0 vs CL2-500 (18/33, p .049).

**Cost.** Wrist stage 1 measured 23.9 ms vs 65.8 ms full (K3, eager; MISS completion +21.6 ms) → owner-basis prices
.0552 per vision decision and +.0499 per MISS by ratio transfer (**ASSUMPTION**: eager ratios transferred to the
CUDA-graph basis; a graphed wrist tower was not measured). Under it wrist saves .073–.110 IR at v = 1 (−31 to −41 %).
K8: WristMixedJudge 1.27–1.58 ms per decision (one PCA instead of two) vs 1.65–2.10.

**Exact dummy_cached repricing** (bit-exact: 24/24 identical action chunks, K3; s1 45.0 vs 65.8 ms eager → owner s1
.1039, same ASSUMPTION of ratio transfer for the *price*, none for the actions). No arm was run because SR is provably
unchanged; the repriced owner IR is IR − .048·v:

| point | v | owner IR | dummy_cached IR |
|---|---|---|---|
| stock g500 (pooled) / g50 | 1 / 1 | .244 / .323 | .196 / .275 |
| K7 phase B=2, l10-500 / l10-50 | .573 / .641 | .178 / .268 | **.150** / .237 |
| K7 anchor_tail, l10-500 / l10-50 / sp-500 | .573 / .595 / .531 | .203 / .242 / .128 | **.176 / .213 / .103** |
| pure-cache tail, sp-500; CL2 (any cell) | .514 / 1 | .078 / .152 | **.053** / .104 |
| sp guard-only g500 / periodic 12 | 1 | .206 / .204 | .158 / .156 |
| pure inference L=5 / L=10 | 1 | 1.0 / .5 | .952 / .476 |

---

## 6. Control-step library (G / GS) vs CL2

| arm | SR | vs CL2 (S→F / F→S, p) | decisions / ep | per-task |
|---|---|---|---|---|
| G, l10-500 | .222 | −.546 (9 / 282, < 1e-4) | 94.0 (timeouts) | every task −10 … −45 |
| GS, l10-500 | .772 | +.004 (34 / 32, .90) | 63.5 | noise |
| G, l10-50 | .114 | −.516 (2 / 260, < 1e-4) | 98.3 | every task |
| GS, l10-50 | .628 | −.002 (46 / 47, 1.0) | 71.5 | noise |
| GS − G (500) | | +.550 (284 / 9) | | |

Why. G keeps AWM's real rows but ranks each parent by the best interpolated code at offset k ∈ {0…4} along the edge
to its successor; a parent chosen at offset k describes an observation k controls *earlier* than the robot's current
state, yet G serves the parent's original un-shifted 5-control head, so each decision re-executes k already-executed
controls; the robot lags, repeats timed events (the very pattern C wanted to remove) and times out (94–98 decisions
per episode vs 63–72). GS splices `action[r, k:5] ‖ action[next(r), :k]`, which realigns the head, and lands exactly on
CL2 at both scales (discordance 66 / 93, split evenly). K1's geometry check confirmed the intervention happened (all 248
sampled stale queries chose a non-zero offset). Conclusion: quantizing retrieval to 5-control boundaries is not a
source of the l10 failures; finer temporal indexing is dead, and it costs +.25–.37 ms per decision (K8).

---

## 7. Randomized CALL / CACHE (K5) and Q3

Design: guard-only MixedJudge, π0.5-l10, both libraries, two complementary replicates per scale; per (task, init) a
SHA-seeded coin picks the 1st or 3rd baseline MISS opportunity as the landmark and CALL vs CACHE at it; nothing else
changes; 100 % compliance; per-episode outcomes Y (success), N (requests), M (actual MISSes), C_ρ = .848·ΔM +
(.152 − ρ)·ΔN at the cell's operating point; Horvitz–Thompson ITT with (task, init)-cluster bootstrap (2,000 draws).

| scale | ITT ΔY [95 %] | ΔN | ΔM | ΔC at own ρ | landmark 1 ΔY | landmark 3 ΔY | exposed-1 ΔY | exposed-3 ΔY | exposure (L1 / L3) | discordant-exposure pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| g500 (ρ .238) | **+.034 [+.004, +.066]** | −1.58 [−2.88, −.16] | −.24 [−1.02, +.54] | −.065 [−.63, +.51] | **+.050 [+.004, +.096]** | +.019 [−.020, +.062] | +.076 [.000, +.161] | −.017 [−.153, +.125] | 78–87 % / 38–52 % | 98 |
| g50 (ρ .323) | −.002 [−.040, +.032] | −.04 [−1.76, +1.67] | +.47 [−.85, +1.75] | +.41 [−.45, +1.20] | −.008 [−.058, +.046] | +.004 [−.047, +.052] | −.027 [−.092, +.043] | −.016 [−.113, +.085] | 91–94 % / 69–74 % | 68 |

Arm-level SR: g500 r1 .826 / r2 .836 (CALL half .848, CACHE half .814 — the CALL half equals the pooled stock .849,
the CACHE half loses 3.4 pp); g50 r1 .768 / r2 .738 (CALL .752, CACHE .754). Reading: at 500 a guard-forced call is
worth +3.4 pp of success *and* shortens the episode by 1.6 requests, so it pays for itself (ΔC point estimate slightly
negative); the first landmark carries the value (+5.0), the third is unresolved. At 50 a single call changes nothing
measurable and adds ≈ 0.5 later MISSes. Q3 (unchanged `causal_fit`, ≥ 30 clusters and ≥ 10 per treatment per cell,
SR-loss cap .01, Bonferroni bounds, 5 init folds + 10 task folds): **zero supported cells with a positive
conservative saving at either scale** (closest: a g50 leaf, LCB −.013), all held-out fits return baseline CALL, and
the g500-minus-g50 SR contrast (+.036) has an interval [−.010, +.084] that includes zero, so even the scale reversal
is not established. Implication: every learned gate ("skip this MISS") that R3/R4 hoped for has no support; the
guard's MISSes are not a saving pool. Combined with §4 this means the remaining IR levers are vision (blind, wrist,
dummy) and execution length, not fewer policy calls. Label: outcome-derived, **borrowed big-library information**
(research use only).

---

## 8. Four-layer decomposition (+ cost implementation), closed loop, paired

| layer | π0.5-l10 | π0.5-spatial | source / label |
|---|---|---|---|
| synthesis (B0 top-1 → mean-5, 50 ep) | −1.2 [−6.0, +3.8] | +9.6 [+4.8, +14.4] | R2 CL0 → CL1 (R3 §3) |
| method at fixed library: AWM ranking (50 ep) | +20.2 | +3.6 | R2 CL1 → CL2 |
| method at fixed library: R3 borrowed prior α .5 | +4.4 (**borrowed**) | +4.8 (**borrowed**) | R3 |
| **method at fixed library, R4**: blind serving rule (K1 guard, B=1, l10-500): tail − phase | **+4.0** (53/33, p .040) | not run | K1 ph1g → tail1ug |
| method: guard (K7 − K1) at equal serving | +1.2 / +0.2 / +2.4 / −1.2 (all n.s.); IR −.02…−.04 | not run | §4.1 |
| method: wrist key (same lib, same guard) | −2.9 [−.061, +.004] (500) / −0.6 (50) | 0.0 (500) / +3.6 (50, Holm n.s.) | §5 |
| method: control-step G / GS | −55 / +0.4 (500), −52 / −0.2 (50) | not run | §6 |
| **library (50 → 500)**, same method | CL2 +13.8; **K7 tail +7.4** (39/76, p .0007); **K7 phase +16.2** (37/118); wrist +8.6 (41/84, p .0002); guard-only +12.4 (R3) | CL2 +15.4 (R2) | paired 50 vs 500 arms |
| **control at the 500 library**: CL2-500 → K7 tail | **+11.2** [+.076, +.148] for ΔIR **+.051** (22 pp / .1 IR) | +2.8 [+.008, +.050] for **−.074** (pure-cache tail: control *and* cheaper) | §3 |
| control at 500: CL2-500 → K7 phase B=2 | +9.4 [+.056, +.132] for +.026 (36 pp / .1 IR) | +1.4 (p .23) for −.012 | |
| control at 500: CL2-500 → stock guard-only | +8.1 (pooled) for +.092 (8.8 pp / .1 IR) | +2.0 (14/4, p .031) for +.054 | R3 g500 / R4 sp g500 |
| control at the 50 library: CL2-50 → K7 tail | **+17.6** [+.132, +.220] for +.090 (19.6 pp / .1 IR) | (no K7 spatial-50 arm) | |
| control at 50: CL2-50 → K7 phase B=2 / periodic 5 / g50 | +7.0 / +16.2 / +11.0 for +.116 / +.163 / +.171 | +8.4 (K1 phase, p < 1e-4) for +.052 | |
| **control: execution length** (policy L=5 → L=10) | **+5.5** [+.024, +.087] for −.50 IR | 0.0 / −0.6 (n.s.) | §2 |
| control: execution length in the cache (same v, zero policy) | not yet run (`r05_ptail`) | +2.8 (p .0125) | tail1uc − ph2c |
| **cost implementation**: dummy_cached (bit-exact) | −.048·v IR, ΔSR = 0 by construction | same | K3 parity |
| cost implementation: wrist stage 1 (ASSUMPTION ratio) | −.073 / −.087 IR at v = 1 | −.080 / −.110 | §5 |
| cost implementation: MISS K=2 (appendix, owner ruling 12) | −.029 (g500) / −.062 (g50) / −.067 (perk5); ΔSR −2.2 (p .16) / −0.8 / −0.2 | −.021 / −.047; ΔSR 0 / +0.8 | never on the frontier |
| **total, deployed 50-ep scale → best R4 point** | .440 (B0) → **.806 @ .242** (K7 tail) | .668 → .924 @ .156 (wrist, unresolved) or .980 @ .43 (R3) | |
| total, 500-ep scale → best R4 point | .768 (CL2) → **.880 @ .203** (K7 tail) or .862 @ .178 (K7 phase) | .954 → **.982 @ .078** (pure-cache tail) | |

GR00T (pure cache only, paired on the same inits):

| layer | GR00T-l10 | GR00T-spatial | source |
|---|---|---|---|
| method at fixed library: blind serving rule (tail2u − phase B=2, no policy) | **+8.4** (500, p .0002) / +0.4 (50, n.s.) | **−3.0** (500, p .006) / +3.0 (50, p .18) | §4.6 |
| library (50 → 500), same method | CL2 +15.4; phase +15.8 (124/45); **tail2u +23.8** (144/25) | CL2 +7.8; phase +14.2 (77/6); tail2u +8.2 (61/20) | all p < 1e-4 |
| control at the 500 library: CL2-500 → tail2u (zero policy) | **+10.2** [+.058, +.146] for ΔIR **−.098** | −2.0 [−.040, .000] for −.096 | §3.5 |
| control at 500: CL2-500 → phase B=2 | +1.8 (n.s.) for −.053 | +1.0 (n.s.) for −.070 (and +3.6 over the policy, p .006) | |
| control at the 50 library: CL2-50 → tail2u / phase | +1.8 / +1.4 (n.s.) for −.098 / −.049 | −2.4 (n.s.) / **−5.4** (p .004) for −.097 / −.059 | |
| total, 50-ep scale → best R4 point | .468 (B0) → .570 @ .050 (tail2u) or .552 @ .148 (CL2) | .736 → .888 @ .148 (CL2; tail2u .864 @ .051 if cost rules) | |
| total, 500-ep scale → best R4 point | .608 (B0) → **.808 @ .050** (tail2u) | .812 → **.976 @ .078** (phase B=2) | |

Every "borrowed" row is labelled; no R4 50-library arm borrows 500-library statistics (K1/K3/K7 hand-backs). The kref
change (5 at 50, 8 at 500) rides inside every library-layer row, as in R2/R3; B1 in R5 tests it prospectively.

---

## 9. Library scale and bytes for every R4 configuration

Fit pickles are the files actually loaded by the servers (`<run>/fits/*.pkl`, sizes read from disk); representation
bytes/entry from the arms' `bytes_per_entry` and the K1/K3/K7 hand-backs; action payload 280 B/row (π0.5) / 448 B
(GR00T) is *inside* the pickles (padded H×32), plus ≈ 17 MB fixed two-camera PCA-64 bases per suite.

| configuration (arms) | library | entries (sp / l10) | B / entry | fit pickle MB (sp / l10) | vs deployed 431 / 1,103 MB |
|---|---|---|---|---|---|
| AWM CL2 (R2; periodic arms reuse it) | 50 | 1,018 / 2,640 | 580 | 21.3 / 24.6 | 5 % / 2 % |
| AWM CL2 | 500 | 10,909 / 29,472 | 580 | 47.0 / 94.1 | 11 % / 9 % |
| MixedJudge guard-only (stock g50 / g500, K5, noprog-4) | 50 / 500 | same | 626 | 26.0 / 32.6 → 66.1 / 141.2 | 6 % / 3 % → 15 % / 13 % |
| K1 BlindAWM (pure-cache blind, two-clock) | 50 / 500 | same | 586 | 26.1 (l10) → 47.4 / 95.3 | |
| K1 BlindMixedJudge (dense guard) | 50 / 500 | same | 632 | 26.1 (sp) / 32.7 → 142.3 (l10) | |
| **K7 VisionConfirmedBlindMixedJudge** (all K7 arms; K10 same) | 50 / 500 | same | 632 | 26.1 / 32.7 → 66.5 / 142.3 | 6 % / 3 % → 15 % / 13 % |
| K3 wrist MixedJudge (guard-only wrist) | 50 / 500 | same | 370 (WristAWM 324) | 15.4 / 21.1 → 50.4 / 116.0 | 4 % / 2 % → 12 % / 11 % |
| control-step G / GS | 50 / 500 | l10 | 588 | 26.0 → 94.4 | |
| SeededInference (pure-inference arms; raw B0 keys, not a deployable representation) | — | | 262,272 | 267 / 692 | |
| GR00T BlindAWM (`r04_gblind`, all 8 arms; ph2 and tail2u share the same fit size) | 50 / 500 | 1,063 / 2,645 → 11,751 / 29,631 | 586 (summary `bytes_per_entry`) | 22.8 / 28.1 → 58.6 / 118.4 | vs 429 / 1,068: 5 % / 3 % → 14 % / 11 % |

Blind serving adds no library bytes (the tail is already in the stored chunk; phase needs 21 B/entry of topology that
K1 folded into the 586–632 B rows). K9's GPU-resident float32 copies would be 22–175 MB per cell (its table). Fit
walls: K7 500-library 42–44 s, 50-library 3.5–9.5 s (K7 hand-back).

---

## 10. Engineering findings

- **K6 serving concurrency (timing only).** Before the fix the R4 plugin held a runtime-wide lock across the whole
  decision including policy inference: K1 phase B=2 / B=0 arms ran 36.6 / 61.1 min per 500 episodes at 7.6–12.6
  decisions/s with server `infer_ms` p50 3.0–4.5 s; after the per-connection lock (installed 15:47 CDT) the same
  code path runs 9–14 min at 33–50 decisions/s and p50 225–450 ms, i.e. the stock arms' 8–11 min. K6's CPU fake-policy
  parity: 2,196 concurrent decisions field-identical to serialized execution, 5.1–6.5× speed-up. No result in this
  document depends on it (LIBERO is synchronous; SR/decision logs are unaffected).
- **K8 search latency** (single thread, warm, real logged queries; p50 ms per vision decision): AWM 1.17 / 1.41 (50 /
  500), R3 MixedJudge 1.65 / 2.10, K7 2.05–2.10 (l10-500), 1.66–1.69 (l10-50), 1.80 (sp-500); blind step phase .47,
  anchor_tail .22; WristMixedJudge 1.27–1.58; native CP1 2.1 (sp) / 7.4 (l10). PCA of the two 32,768-d pooled keys is
  .7–.8 ms (31–49 %), library-size independent; scaling the task block ×10 takes AWM 1.35 → 3.4 ms; one process is
  GIL-bound at ≈ 400–600 queries/s, per-decision latency growing linearly with connections (24 connections: 36–70 ms).
  Owner-basis surcharge: g500 .238 → .271 (K8's number; .269 with the p50 in §3), K7 tail .203 → .222, K7 phase
  .178 → .198, spatial guard .197 → .229.
- **K9 GPU-resident retrieval** (prototype module, not in serving): stage-1 manual CUDA graph ≈ 10.2 ms reproduces
  the owner's 10.26 ms basis (eager 65–68 ms); AWM / MixedJudge features captured in the same graph add .36–.40 /
  .58–.63 ms (separate graphs .30 / .56–.80 ms; **GPU eager 4.5–10 ms, worse than the CPU**); PCIe per decision 131 KB
  (two bf16 pooled keys D2H) → ≈ 1.4 KB (chunk + verdict). Parity vs the CPU method (float32): top-1 ≥ 99.90 %, top-16
  set 99.8–100 %, non-step-0 chunk Δ ≤ 2.4e-5, **step-0 chunk Δ up to .0125** (expanded-distance cancellation, condition
  numbers 2.5k–16k; float64 does not fix it because the fitted buffers are float32); in-place append keeps graph
  addresses (32/32). Not done: guard state machine, HIT/MISS verdict on device, blind/K7 on GPU, GR00T, serving
  integration, any closed-loop check. IR consequence: search surcharge ≈ .006–.009 instead of .02–.03 (§3.6).
- **K10 policy tail**: `--os-policy-tail` (π0.5, needs `--os-blind`) serves `policy_chunk[5:10]` after a MISS as a
  zero-cost blind decision, wire-identical to the L=10 client (400 controls checked), byte-identical logs without the
  flag; but `PolicyTailJudge` routes the tail through K1's `blind_step`, which refuses whenever `_noprog_span > 0` —
  true for 88 % (50) / 94 % (500) of l10 MISSes (R5-A) — and K10's own guard-only replays served 0 tails. The three K10
  arms were therefore not run; the lifecycle-gated `CommitJudge` (R5-Q1) is the actual test. Engineering complete,
  effect unmeasured.

---

## 11. R5 check and R6 proposals

**R5 selections vs the R4 evidence.**

| R5 item | R4 evidence | verdict |
|---|---|---|
| Q1 C10 CommitJudge (execute the policy's own steps 5–9 after a MISS, lifecycle gate only) + D1 GraspCheck | L=10 pure inference +5.5 pp is the largest and cleanest R4 effect and is replicated (.904 / .900); 88–94 % of MISSes are inside a no-progress span, so the K10 veto blocks exactly the cases that matter; 13.7 % of K7-tail decisions are 5-step policy chunks. D1's frozen-path reach is 73 / 24 episodes (50 / 500) with a +1.3 / +0.4 pp planning scenario — below the .016 SE of one run at 500. | **C10 supported** (expect the policy-side share of +5.5 pp; forecast +1–3 pp at IR ≈ .15–.18); **D1 is not resolvable by a single 500-init run at 500** — run it at 50 only, or with 2–3 replicates, or as a mechanism-enriched pilot on the 73 alarm inits. |
| Q2 GR00T CycleTail (10-step chunks, MISS every 4th anchor) + pure-policy L=10 GR00T | The pure-cache GR00T tail already gives l10-500 .808 @ .050 with zero policy calls (§4.6); on π0.5 the two-clock (blind + periodic) design was the worst blind arm (§4.2) and periodic k=8/12 is dominated by guard-only at 500; GR00T-sp-500 pure cache beats the policy (CL2 +2.6, p .07; phase B=2 +3.6, p .006), so scheduled policy calls can only hurt there. Forward context (one R5 arm, not used elsewhere): `r05_q2/r5q2_g_l10_500_G10` = .828 @ .186 (v .503, m .131), +2.0 pp over tail2u (50/40, p .34) at 3.7× its IR. | **Supported only for l10 and only against the zero-policy tail baseline**: G10's +2 pp is inside the noise floor for a 3.7× cost increase; the sp-500 cell is a regression check, not a target. A one-block (10-step) pure-cache GR00T tail is the control both Q2 arms lack. |
| Q3 call everywhere | K5: g500 ΔY +.034 [+.004, +.066]; Q3: no supported saving at either scale | **Settled, supported**; learned suppression gates are dead on this data. |
| Q4 library growth (grow250) + demo scaling curve (100/200/300 × refit/frozen50) | Library layer at fixed R4 method: +7.4 (tail) … +16.2 (phase) pp; blind phase is the most library-sensitive rule; a 500-end point with kref 8 vs 5 elsewhere confounds the curve; natural MISS growth is 4.3 successful rows per episode (R5-C). | **Supported as measurement**; report the curve with the tail controller (least library-sensitive) and phase (most), and read kref off B1 first. |
| Q5 GPU retrieval into serving (shadow) | K9: +.36–.63 ms in-graph, ≥ 99.9 % top-1, step-0 Δ .0125; PCIe 131 KB → 1.4 KB; GIL cap 400–600 q/s. | **Supported as engineering**; needs a declared tie / step-0 numerical policy and a closed-loop shadow agreement rate before serving; the blind step (.22–.48 ms) stays CPU until ported. |
| Q6 wrist + blind (wrist-confirmed stuck guard + anchor_tail) | Wrist: no resolved SR change in any cell (Holm), −.073…−.110 IR (assumed ratio); tail: +3.1 pp / −.041 IR at 500. Predicted stack under the assumption: l10-500 .0552·.573 + .898·.137 = **.155** (vs .203), sp-500 .080 (vs .128, ≈ the pure-cache tail's .078 with policy rescue kept), l10-50 .193, sp-50 unknown (no K7 tail at sp-50). | **Supported**; the only cell with SR risk is l10-500 (wrist −2.9 vs pooled); demand the pooled 3-run comparison. |
| B1 constrained offline solver (ridge / kref / state×3, one change per cell) | R5-B's own retrospective: phase-vs-tail (strong miss: offline favoured phase, loop +10.6 for tail at 50), noprog 3→4 miss, B1→B2 unranked; R4 adds: offline error ranks tail < phase, loop says the opposite at both scales. kref 5 vs 8 has never been isolated. | **Supported only as a falsification test** (expect null within ±3 pp); its one clean prospective contribution is kref at 500. |
| owner item 10 (solve hyperparameters offline) | Every R4 hyperparameter that mattered (B, serving rule, guard type, wrist) was ranked wrongly or not at all by offline error; the only offline quantity that predicted the loop was cost arithmetic (IR) and the LOEO thresholds that were *copied* from stock (K7). | Offline solving sets thresholds and costs; it does not rank controllers. Keep closed-loop pairs as the exam. |

**Settled by R4.** (i) Blind budget 2 with the vision-confirmed guard; anchor_tail as the serving rule (scale-robust),
phase_particles only at 500. (ii) The stock 500-library reference is .849 @ .244 (three runs), not .864 @ .238.
(iii) Execution length is worth +5.5 pp on l10 for the policy and +2.8 pp on spatial for the cache; every R4 SR gain
should be read net of it. (iv) The MISS side is not a saving pool (K5/Q3). (v) dummy_cached is a free −.048·v.

**Dead.** Dense (motion-only) stuck guard; noprog-4 at both scales; two-clock blind + periodic; phase_particles at 50;
per-member proprioceptive phase correction as a serving rule (worse than the plain clock); control-step library G / GS;
stage-2 prefix packing; learned call-suppression gates; periodic 8 / 12 at 500 (dominated by K7 blind arms); MISS
K=2 as a system component (owner ruling; SR-neutral anyway).

**Ranked ideas beyond R5.**
1. **Replicate the frontier points before building on them.** K7 anchor_tail at l10-500 is +3.1 pp [+0.0, +6.3] over
   stock on one run; run it twice more (as was done for g500) and adopt "three-run pooled" as the acceptance rule for
   any frontier point claimed within 3 pp of its neighbour. Cost: 2 × 11 min of server time.
2. **Isolate the cache's execution-length effect on l10** (pure-cache anchor_tail at 50 and 500, already queued as
   `r05_ptail`) and, if it explains ≥ 2 pp, treat "commit to the plan" as a control primitive: the natural next arm is
   *policy* commit (C10) + *cache* commit (tail) + no re-retrieval mid-plan, i.e. a 10-step clock for both sources.
3. **Stack the free cost layers on the K7 tail point**: dummy_cached (exact) + wrist (Q6) + K9 in-graph retrieval →
   projected l10-500 ≈ .880 @ .13–.16 owner IR with search included; verify wrist's l10-500 SR first (item 1 rule).
4. **Task-4 anatomy of the tail arms.** t4 loses 5–8 episodes in every anchor_tail arm at both scales and not under
   L=10 inference; D1 reaches none of them. Log full chunks + states (`--os-log-inputs`) on t4 inits for tail vs B=0 and
   find whether the stored tail's gripper timing or its translation is what fails; this is the only consistent
   tail-specific regression and it caps the point at .88.
5. **Spatial: drop the policy.** Pure-cache anchor_tail (.982 @ .078 → .053 with dummy) equals pure inference within
   the suite's resolution; the remaining question is t9 (5 of 9 failures). A t9-only diagnostic (per-init replay of the
   failed inits with the K7 tail-with-policy arm, which also fails 4 of them) is cheaper than any new controller.
6. **GR00T**: the sp-500 product is pure-cache phase B=2 at .976 @ .078 (above the policy) with no mixed controller;
   the l10 product is the two-block tail at .808 @ .050, 6 pp under the policy, and the first thing to test is the
   one-block (10-step) tail at both suites (the point that separates "commit" from "commit for 15 steps"), then guards
   re-thresholded on GR00T library statistics (K7 refuses GR00T for the gripper-sign reason) before any periodic MISS.
7. **Search cost**: with K9 the retrieval surcharge is < .01; the next largest non-IR cost is the per-decision JSON
   emit (.3–.4 ms) and, under load, the GIL (24 connections → 36–70 ms). A per-server process count or a C++/Rust
   query path matters more than any further numpy work.

---

## 12. Reproduction (read-only on the run roots; CPU 26-29,70-73; scratch `/home/weiland/.claude/jobs/a607dd74/tmp/analysis_r04/`)

> Scripts are preserved in `analysis_scripts/` (copied from the job scratch directory `tmp/analysis_r04/` at 05:1x CDT on 2026-09-28).


```bash
cd /home/weiland/projects/openpi; S=/home/weiland/.claude/jobs/a607dd74/tmp/analysis_r04
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
taskset -c 26-29,70-73 .venv/bin/python $S/an.py          # journals -> an.json, arms.md (SR, Wilson, v, m, owner/eager/search IR), pairs.md (215 paired tests), per_task.md
taskset -c 26-29,70-73 .venv/bin/python $S/blind_stats.py # decision logs (8 procs) -> blind_stats.json: src mix, look reasons, judge/force mix, MISS-after-MISS, per-task v/m, stage timings
taskset -c 26-29,70-73 .venv/bin/python $S/frontier.py    # -> frontier.md/json: Pareto flags in three bases, chord, pooled 3-run g500 test, noise floor
taskset -c 26-29,70-73 .venv/bin/python $S/blind_runs.py  # -> blind_runs.json: blind-run lengths, gate cuts, anchors followed by blind; dummy_cached repricing table
```

Inputs: `/home/weiland/trace_runs/os_closed_loop/{r02_g50,r02_g500,r03_mx,r04_frontier,r04_cost,r04_blind,r04_k7,
r04_b4w,r04_csl,r04_rep,r04_k5,r04_gblind}/{summary.json,runs/<arm>/client/journal.jsonl,runs/<arm>/server_*/
decisions_*.jsonl,state/*.DONE}`, `/home/weiland/trace_runs/dual_20260923/runs/tr_{pi05,groot}_{l10,sp}_inf/client/
journal.jsonl`, `rounds/r04/cost_table_owner.json`, `closed_loop/ops/cost_table.json`, `r04_k5/k5_g{50,500}_estimate.json`.
Completion rule = collect.py (accepted ∧ status ∈ {done, failed} ∧ no error); server rows matched to the accepted
attempt. All 88 arms (including `inf_sp_s2001` and the 8 `gb_*`) are listed in `an.py:ARMS`, `blind_stats.py`, `blind_runs.py`;
the tables were regenerated on 2026-09-28 after the last `.DONE` marker appeared.

**Caveats.** All R4 servers ran 2 per 4090 with 24–32 workers; stage times in the logs are load numbers (s1 p50
100–600 ms) and were not used for IR. The wrist / dummy prices are eager-to-graph ratio transfers (assumption, labelled).
K5 identifies one landmark per episode under the guard-only controller only. GR00T has no mixed R4 arm and no
one-block tail arm; its owner basis (.148 / .174 / .678) is historical. `r04_gblind` was written by two concurrent
chains (its `state/current` is stale; per-arm summaries, `.DONE` markers and 500 accepted journal rows per arm were
checked). The single R5 arm cited in §11 is forward context and enters no table or verdict.
