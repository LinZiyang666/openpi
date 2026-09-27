# R2 analysis — offline families and the pure-cache closed loop (2026-09-27 00:3x CDT; addendum 1 with the complete 50-episode group 01:0x CDT)

Analysis agent (fable), R2 of `logs/offline_search_exploration.log.md`. Cell order everywhere: **π0.5-sp / π0.5-l10 / GR00T-sp /
GR00T-l10**. err = executed-segment RMS `[:5,:7]` in σ units vs the teacher's `a_inf` (lower is better); **stale** = cache cells,
step ≥ 1 (14,121 / 39,627 / 13,320 / 39,169 decisions; every previous decision a HIT); **fresh** = inf cells, step ≥ 1; **step 0** =
500 decisions per cell. **gs** = gripper-sign-faithful err (executed gripper snapped to its sign; LIBERO applies `np.sign`). AURC =
risk–coverage area under the method's own confidence, computed within the regime. Closed loop = pure cache (every decision a
HIT, stage 1 runs, stage 2/3 never), A-pool 500 inits per arm, 4 single-replica servers on the 4090 + 64 timan107 workers,
run root `/home/weiland/trace_runs/os_closed_loop/r02_g50` (50-episode library) and `r02_g500` (500-episode library).

**Library scale (owner rule, stated once here and next to every conclusion below).** *Current* = the deployed library:
49 / 50 / 50 / 50 episodes, 1,018 / 2,640 / 1,063 / 2,645 entries, all successful; deployed pkl 431 / 1,103 / 429 / 1,068 MB
at 262 KB key per entry. *10×* = 500 episodes (`bpool_cs` π0.5, `bpool_all` GR00T): 10,909 / 29,472 / 11,751 / 29,631
entries, of which 13 / 64 / 44 / 73 episodes failed. Method entry sizes: B0 / M4 / M8x 262,176–262,272 B; AWM 580 B (r32
codes 288 B); V4 292 B; T2 544 B; G3 wrapper +46 B. Fit-pickle sizes actually shipped to the servers are in §4.

Everything below was computed read-only with `taskset -c 30-33,74-77`, ≤ 8 processes, BLAS 1 thread; scratch and the
regeneration scripts are in `/home/weiland/.claude/jobs/a607dd74/tmp/analysis_r02/` (§7). Closed-loop KPIs come from
`closed_loop/ops/kpi.py` (validated against ideation A / C in `rounds/r03/h4_kpi/r02_kpi.md`, 30/30 checks); offline paired
statistics from `profile.compare` (episode bootstrap) and `g4_t2/decompose.py` (episode bootstrap, 2,000 reps).

**Arms complete (16 of 32): the whole 50-episode group** (`r02_g50` CHAIN_DONE 00:51 CDT; addendum 1 folded GR00T-l10
CL2 / CL3 in). **Pending:** the whole 500-episode group (16 arms, ≈ 3 h, now scheduled ≈ 03:00 CDT after the R3 pilots). See §7;
addendum 2 will fill those rows with one regeneration command.

---

## 0. Headline verdicts

1. **The exam is passed at the deployed library scale by AWM, and the win is real on four of four cells.**
   Pure-cache SR at 49–50 episodes: π0.5-sp .668 → **.800** (+13.2 pp), π0.5-l10 .440 → **.630** (+19.0 pp), GR00T-sp .736 →
   **.888** (+15.2 pp), GR00T-l10 .468 → **.552** (+8.4 pp), each on the same 500 inits (paired CL0→CL2 F→S / S→F: 137 / 71,
   173 / 78, 128 / 52, exact McNemar p < 1e-4; GR00T-l10 CL1→CL2 97 / 54, p = .0006). Against the pure-inference reference
   (.986 / .844 / .940 / .870) the remaining gap is 18.6 / 21.4 / 5.2 / 31.8 pp — GR00T-l10 is now the weakest cell for the cache.
   The AWM library is 21–27 MB (fit pickle incl. 17 MB PCA bases) against 429–1,103 MB deployed.
2. **The three layers do not have the same weight in closed loop as offline.** Offline stale err at 50 episodes attributes
   −.054 / −.058 / −.074 / −.077 to synthesis (top-1 → mean-5) and only −.011 / −.026 / −.038 / −.055 to the AWM ranking.
   Closed loop: synthesis +9.6 [+4.8, +14.4] / **−1.2 [−6.0, +3.8]** / +11.6 [+7.0, +16.0] / **−0.2 [−5.2, +4.8]** pp; ranking
   +3.6 [−0.6, +8.0] / **+20.2 [+15.4, +25.2]** / +3.6 [+0.0, +7.2] / **+8.6 [+3.8, +13.4]** pp. Synthesis alone is worth nothing on l10 for
   **both** models (GR00T-l10 CL1 .466 vs CL0 .468, 85 S→F / 84 F→S) while it is the larger layer on spatial; the ranking layer
   decides l10. Offline mean err (−.05 … −.08 for synthesis in all four cells) cannot tell which (§3).
3. **The recovery wrapper (CL3 = V6 blend + escalating recovery around AWM, insurance OFF) is not a general lever**: −0.2
   [−3.0, +2.6] / +1.2 [−2.2, +4.6] / **−2.6 [−5.6, +0.4]** / **+5.4 [+1.2, +9.8]** pp. Pooled over the four cells 142 S→F vs
   161 F→S = +0.95 pp, McNemar p = .30. The one positive cell (GR00T-l10, 47 S→F / 74 F→S, p = .018) is above the rerun noise
   floor (a same-arm rerun flips ≈ 6 % of inits ⇒ ≈ 30 symmetric discordant pairs, sd(ΔSR) ≈ 1.1 pp; CL2→CL3 has 121 discordant
   pairs, asymmetric) but it is task-specific: t6 +26 pp (16 / 3, p = .004), t4 +16, t8 +14 against t9 −16 (2 / 10, p = .04) and
   t5 −8. Everywhere the wrapper breaks the identical-pick spells (failed-episode spell share .525 → .105 / .479 → .228 / .438 →
   .021 / .515 → .286) and raises gripper flips in failed episodes (2.9 → 4.9 / 4.6 → 9.0 / 1.4 → 3.9 / 3.1 → 5.5). Verdict
   unchanged: drop as a pure-cache mechanism; keep its detector as a MISS trigger (R3 H3); the GR00T-l10 exception says a
   *forced trajectory switch* can rescue some multi-object l10 traps and is worth one targeted R3 look (§6.5).
4. **Every closed-loop failure is a step-cap timeout and ≥ 95 % of failed episodes contain a spell**, on GR00T exactly as on
   π0.5. GR00T-spatial is π0.5-spatial with a higher ceiling and an even purer terminal-row trap (first spell of failed episodes
   T = .88 / .72 / .82 for CL0 / CL1 / CL2 vs .67 / .62 / .38 on π0.5). GR00T-l10 CL0–CL3 show the l10 pattern (first spell of failed
   episodes H .60 / .62 / .50 / .57, Z .21 / .16 / .16 / .21, P .12 / .08 / .14 / .13, G – / .13 / .20 / .09, T .07 / 0 / 0 / 0; SR
   .468 / .466 / .552 / .606) with the largest gripper vote splits of any arms (CL1 S .197 / F .399, CL2 .196 / .334) — the same
   mean-5-at-the-grasp mechanism as π0.5-l10.
5. **Mean SR hides task collapses with two different mechanisms.** π0.5-sp task 6: .72 → .50 → .28 (CL0 → CL1 → CL2); the
   offline signature is the gripper vote split of the 50-episode metric (severe split .163 vs .045 with the borrowed 500-episode
   fit) — R3 H1's borrowed prior targets exactly this. GR00T-sp task 8: .62 → .78 → **.56** (15 S→F vs CL1, p = .019); its first
   spells are 91 % terminal-row absorption, its vote split is .013, and the borrowed fit does not change it — only a terminal
   guard (H1 ④) can. GR00T-l10 task 9 is a third collapse (.54 → .34 → .18 for CL1 → CL2 → CL3; 16 S→F vs CL1, p = .05; 10 S→F
   vs CL2, p = .04) of the hub class (first spell H .67 / .63, no gripper split .12) that no R3 switch targets. R3 must keep the H1
   switches separate and read per-task SR, not the mean.
6. **The 10× library is worth about as much offline as the whole method layer** (stale −.078 / −.080 / −.079 / −.065 at fixed
   method; oracle −29 … −37 %), and historically +12.2 / +6.0 pp SR for B0-style top-1 (S3 → S6). Its closed-loop value for AWM
   is **pending** (500-episode group). The borrowed 500-episode *fit* with 50-episode candidates is worth −.020 … −.025 stale err
   and −.01 … −.02 AURC ("borrowed big-library information"), but it doubles the severe gripper split on π0.5-l10 task 6 (.23 →
   .43) and raises it on GR00T-l10 overall (.20 → .26) — the α = .5 hedge in R3 SELECTION is justified.
7. **Offline verdicts**: keep G1 AWM (kr5 at 50 ep, kr8 at 10×; r32 codes as the compressed variant; insurance, λ .01, lc1,
   state×3 dropped); keep G2 V4 only as the π0.5 backup / ablation (ties AWM on π0.5 at 50 ep, loses .03–.05 on GR00T) and V5 as
   the fresh-regime branch (equal or better than AWM's continuity branch); drop V6 (blend ≤ −.007 offline, recovery no SR); keep
   V7 (AURC −.06 stale / −.04…−.06 fresh vs the base, one scale across regimes) but it is measured only on the stand-in base;
   drop G4 T2 (MLKR variant beats AWM by ≤ .017 in 3/4 cells, plain MLP is worse everywhere; GPU fit).

---

## 1. Closed-loop table (the exam)

### 1.1 Arms (SR with Wilson 95 % CI; n = 500 complete episodes per arm, journal ↔ server success agreement 500/500, every client decision FULL_HIT, `exec_ok` 1.0)

| cell | library | arm | method | SR | Wilson 95 % | dec/ep S \| F (cap) | method q_us p50 / p95 | server infer ms p50 / p95 | native agree | pure-inf ref | trace_dual B0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| π0.5-sp | 50 ep (1,018) | oscl50_p_sp_cl0 | native B0 top-1 | **.668** | [.626, .708] | 22.0 \| 44 (44) | 4,949 / 13,558 | 166 / 241 | – | .986 | .668 |
| π0.5-sp | 50 ep | oscl50_p_sp_cl1 | M4 = B0 rank + mean-5 | **.764** | [.725, .799] | 21.3 \| 44 | 4,399 / 9,873 | 183 / 266 | 1.000 | | |
| π0.5-sp | 50 ep | oscl50_p_sp_cl2 | AWM joint, fit on the 50 ep, kr5, kernel-16 | **.800** | [.763, .833] | 20.8 \| 44 | 3,488 / 8,336 | 171 / 253 | .445 | | |
| π0.5-sp | 50 ep | oscl50_p_sp_cl3 | V6 (blend + recovery, ot_still) around CL2, insurance off | **.798** | [.761, .831] | 20.8 \| 44 | 4,847 / 10,420 | 162 / 244 | .397 | | |
| π0.5-l10 | 50 ep (2,640) | oscl50_p_l10_cl0 | native | **.440** | [.397, .484] | 50.7 \| 104 (104) | 10,612 / 31,736 | 194 / 275 | – | .844 | .452 |
| π0.5-l10 | 50 ep | oscl50_p_l10_cl1 | M4 | **.428** | [.385, .472] | 54.4 \| 104 | 13,406 / 36,636 | 277 / 379 | 1.000 | | |
| π0.5-l10 | 50 ep | oscl50_p_l10_cl2 | AWM cur kr5 | **.630** | [.587, .671] | 53.8 \| 104 | 4,487 / 10,266 | 209 / 288 | .383 | | |
| π0.5-l10 | 50 ep | oscl50_p_l10_cl3 | V6(AWM) | **.642** | [.599, .683] | 51.0 \| 104 | 8,361 / 18,234 | 226 / 321 | .338 | | |
| GR00T-sp | 50 ep (1,063) | oscl50_g_sp_cl0 | native | **.736** | [.696, .773] | 21.8 \| 44 | 2,812 / 3,789 | 577 / 889 | – | .940 | .736 |
| GR00T-sp | 50 ep | oscl50_g_sp_cl1 | M4 | **.852** | [.818, .880] | 22.0 \| 44 | 3,895 / 5,201 | 598 / 910 | 1.000 | | |
| GR00T-sp | 50 ep | oscl50_g_sp_cl2 | AWM cur kr5 | **.888** | [.857, .913] | 22.1 \| 44 | 2,733 / 3,604 | 602 / 928 | .469 | | |
| GR00T-sp | 50 ep | oscl50_g_sp_cl3 | V6(AWM) | **.862** | [.829, .889] | 21.8 \| 44 | 3,465 / 4,747 | 622 / 946 | .429 | | |
| GR00T-l10 | 50 ep (2,645) | oscl50_g_l10_cl0 | native | **.468** | [.425, .512] | 51.3 \| 104 | 4,753 / 7,273 | 856 / 1,036 | – | .870 | .468 |
| GR00T-l10 | 50 ep | oscl50_g_l10_cl1 | M4 | **.466** | [.423, .510] | 52.3 \| 104 | 8,702 / – | 950 / – | 1.000 | | |
| GR00T-l10 | 50 ep | oscl50_g_l10_cl2 | AWM cur kr5 | **.552** | [.508, .595] | 53.9 \| 104 | 2,752 / 3,743 | 842 / 1,013 | .354 | | |
| GR00T-l10 | 50 ep | oscl50_g_l10_cl3 | V6(AWM) | **.606** | [.563, .648] | 53.6 \| 104 | 3,493 / 4,841 | 885 / 1,159 | .326 | | |
| all four | 500 ep (10×) | oscl500_* (CL0 = M8x top-1, CL1 = M8x mean-5, CL2 = AWM big kr8, CL3 = V6(AWM big)) | | *pending* | | | | | | | |

Reading notes. (i) `method q_us` is the method's own `query()` on the server (numpy, 1 BLAS thread, 4 servers sharing 9 cores,
so 2–3× the offline single-thread numbers); AWM is *faster* than the native search at l10 (4.5 vs 10.6 ms p50) because it
scores 136-d codes instead of 2 × 32,768-d keys. (ii) `server infer ms` includes stage 1 and, for the plugin arms, the
**native shadow search** run for logging (CL1 l10 277 vs CL0 194 ms is the shadow, not M4) — use `q_us` for method cost.
(iii) native agreement = plugin top-1 == native B0 winner on the same live keys: AWM agrees with B0 on 45 / 38 / 47 % of
decisions (CL1 = 1.0 by construction); GR00T-l10 CL2 / CL3 .354 / .326. (iv) The native reruns reproduce trace_dual's pure-cache
SR to 0 / −1.2 / 0 / 0 pp; the R3 pilot rerun of π0.5-sp CL2 on 100 inits flipped 3 S→F / 3 F→S (≈ 6 % discordant, symmetric,
from GPU nondeterminism), i.e. sd(ΔSR) ≈ 1.1 pp and a 95 % rerun band of ± 2.2 pp at n = 500 — the floor every paired ΔSR
below has to clear, on top of its McNemar p.
(v) All failures are timeouts at the step cap (44 / 104 decisions); successful episodes take half the cap in every arm, so no
method is "slow but succeeds" and AWM is not faster on l10 (53.8 vs 50.7 dec/ep).

### 1.2 Paired comparisons on the same inits (exact two-sided McNemar on the discordant pairs; multinomial bootstrap 95 % of ΔSR, 10,000 reps)

| cell | comparison | SR ref → arm | ΔSR (pp) | S→F | F→S | McNemar p | bootstrap 95 % (pp) | layer |
|---|---|---|---|---|---|---|---|---|
| π0.5-sp | CL0 → CL1 | .668 → .764 | **+9.6** | 56 | 104 | .0002 | [+4.8, +14.4] | synthesis |
| π0.5-sp | CL1 → CL2 | .764 → .800 | +3.6 | 50 | 68 | .117 | [−0.6, +8.0] | method @ 50-ep library |
| π0.5-sp | CL2 → CL3 | .800 → .798 | −0.2 | 25 | 24 | 1.000 | [−3.0, +2.6] | recovery wrapper |
| π0.5-sp | CL0 → CL2 | .668 → .800 | **+13.2** | 71 | 137 | < 1e-4 | – | total (per-init tables) |
| π0.5-l10 | CL0 → CL1 | .440 → .428 | −1.2 | 83 | 77 | .693 | [−6.0, +3.8] | synthesis |
| π0.5-l10 | CL1 → CL2 | .428 → .630 | **+20.2** | 40 | 141 | < 1e-4 | [+15.4, +25.2] | method @ 50-ep library |
| π0.5-l10 | CL2 → CL3 | .630 → .642 | +1.2 | 33 | 39 | .556 | [−2.2, +4.6] | recovery wrapper |
| GR00T-sp | CL0 → CL1 | .736 → .852 | **+11.6** | 42 | 100 | < 1e-4 | [+7.0, +16.0] | synthesis |
| GR00T-sp | CL1 → CL2 | .852 → .888 | +3.6 | 35 | 53 | .069 | [+0.0, +7.2] | method @ 50-ep library |
| GR00T-sp | CL2 → CL3 | .888 → .862 | **−2.6** | 37 | 24 | .124 | [−5.6, +0.4] | recovery wrapper |
| GR00T-l10 | CL0 → CL1 | .468 → .466 | −0.2 | 85 | 84 | 1.000 | [−5.2, +4.8] | synthesis |
| GR00T-l10 | CL1 → CL2 | .466 → .552 | **+8.6** | 54 | 97 | .0006 | [+3.8, +13.4] | method @ 50-ep library |
| GR00T-l10 | CL2 → CL3 | .552 → .606 | **+5.4** | 47 | 74 | .018 | [+1.2, +9.8] | recovery wrapper (task-specific: t6 +26 p .004, t4 +16, t8 +14; t9 −16 p .04, t5 −8) |
| all four | recovery pooled | | +0.95 | 142 | 161 | .30 | – | recovery wrapper, 2,000 paired inits |
| all | 500 vs 50 per CL | | *pending* (library layer) | | | | | |

The two spatial method-layer gains (+3.6 pp each) are not individually significant at n = 500 (p .07–.12); the synthesis gains
on spatial, the ranking gains on both l10 cells, the GR00T-l10 recovery gain and the CL0 → CL2 totals are. Paired CI half-widths are ± 4–5 pp at 500 inits (± 9–10 pp
for a 100-episode pilot): R3 must decide on paired counts, not point SR.

### 1.3 Per-task SR (n = 50 inits per task)

| arm | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| π0.5-sp CL0 | .54 | .74 | .82 | .78 | .54 | .74 | .72 | .68 | .68 | .44 | t9 .44 |
| π0.5-sp CL1 | .88 | .88 | .68 | .72 | .78 | .88 | .50 | .80 | .84 | .68 | t6 .50 |
| π0.5-sp CL2 | .80 | .84 | .94 | .98 | .74 | .92 | **.28** | .96 | .90 | .64 | t6 .28 |
| π0.5-sp CL3 | .80 | .88 | .88 | .94 | .76 | 1.00 | .26 | .96 | .86 | .64 | t6 .26 |
| π0.5-l10 CL0 | .10 | .54 | .64 | .72 | .24 | .88 | .14 | .30 | .14 | .70 | t0 .10 |
| π0.5-l10 CL1 | .20 | .64 | .50 | .54 | .22 | .80 | .26 | .40 | .10 | .62 | t8 .10 |
| π0.5-l10 CL2 | .32 | .82 | .52 | .88 | .62 | 1.00 | .50 | .62 | .32 | .70 | t0 .32 |
| π0.5-l10 CL3 | .44 | .86 | .56 | 1.00 | .52 | 1.00 | .52 | .56 | .28 | .68 | t8 .28 |
| GR00T-sp CL0 | .46 | .94 | .84 | .74 | .76 | .80 | .82 | .70 | .62 | .68 | t0 .46 |
| GR00T-sp CL1 | .90 | .84 | .72 | .84 | .88 | .88 | .98 | .86 | .78 | .84 | t2 .72 |
| GR00T-sp CL2 | .94 | .98 | .76 | .90 | .92 | .92 | 1.00 | .90 | **.56** | 1.00 | t8 .56 |
| GR00T-sp CL3 | .90 | .94 | .72 | .84 | .96 | .90 | .96 | .92 | .50 | .98 | t8 .50 |
| GR00T-l10 CL0 | .08 | .60 | .78 | .86 | .44 | .70 | .30 | .24 | .24 | .44 | t0 .08 |
| GR00T-l10 CL1 | .08 | .50 | .78 | .92 | .24 | .80 | .38 | .28 | .14 | .54 | t0 .08 |
| GR00T-l10 CL2 | .20 | .64 | .94 | .86 | .34 | .98 | .38 | .42 | .42 | **.34** | t0 .20 |
| GR00T-l10 CL3 | .26 | .68 | .94 | .88 | .50 | .90 | .64 | .52 | .56 | **.18** | t9 .18 |

AWM wins 8 of 10 spatial tasks on π0.5 (t2/t3 +26 pp each vs CL1) and 9 of 10 on GR00T, and 9 of 10 l10 tasks on π0.5 (t4
+40, t3 +34 pp vs CL1), and 7 of 10 l10 tasks on GR00T (t8 +28, t5 +18, t2 +16 vs CL1), while collapsing on one task per cell (π0.5-sp t6, GR00T-sp
t8, GR00T-l10 t9 −20 vs CL1). Fixing π0.5 t6 back to CL0's .72 alone is worth +4.4 pp; GR00T t8 to CL1's .78 +2.2 pp; GR00T-l10
t9 to CL1's .54 +2.0 pp.

### 1.4 Spell / trap KPIs (spell = ≥ 3 consecutive identical served top-1; first-spell class of failed episodes T terminal row / G |gripper vote| < .5 / Z translation < .6 σ / P library pause row / H none)

| arm | SR | spells/ep S \| F | spell dec share S \| F | P(S \| 0 spells) (n) | P(S \| ≥ 2) (n) | failed eps with spell | first-spell T/G/Z/P/H (fail) | flips g0 S \| F | w_term_late F | vote split < .8 S \| F | ep_eff S \| F |
|---|---|---|---|---|---|---|---|---|---|---|---|
| π0.5-sp CL0 | .668 | .17 \| 2.25 | .045 \| .350 | .97 (295) | .07 (134) | .95 | .67/.00/.05/.09/.19 | 1.05 \| 1.96 | .72 | – | 1.0 |
| π0.5-sp CL1 | .764 | .14 \| 2.48 | .031 \| .512 | 1.00 (338) | .06 (93) | 1.00 | .62/.03/.02/.04/.29 | 1.00 \| 0.99 | .47 | .081 \| .174 | 2.8 \| 2.5 |
| π0.5-sp CL2 | .800 | .07 \| 2.75 | .016 \| .525 | 1.00 (373) | .04 (84) | 1.00 | .38/.14/.03/.17/.28 | 1.00 \| 2.93 | .40 | .038 \| .282 | 3.8 \| 3.6 |
| π0.5-sp CL3 | .798 | .06 \| 1.21 | .008 \| .105 | .90 (421) | .03 (39) | .57 | .00/.16/.12/.29/.43 | 1.01 \| 4.86 | .21 | .039 \| .256 | 3.8 \| 3.1 |
| π0.5-l10 CL0 | .440 | .62 \| 4.37 | .066 \| .390 | .95 (129) | .10 (273) | .98 | .14/.00/.09/.17/.61 | 2.76 \| 5.49 | .25 | – | 1.0 |
| π0.5-l10 CL1 | .428 | 1.31 \| 6.01 | .142 \| .499 | .99 (92) | .20 (353) | 1.00 | .07/.19/.08/.12/.54 | 2.79 \| 4.14 | .12 | .149 \| .333 | 2.4 \| 2.1 |
| π0.5-l10 CL2 | .630 | 1.17 \| 6.45 | .128 \| .479 | .99 (145) | .34 (271) | .99 | .07/.12/.11/.22/.47 | 2.85 \| 4.56 | .10 | .094 \| .286 | 3.1 \| 3.1 |
| π0.5-l10 CL3 | .642 | .49 \| 5.07 | .032 \| .228 | .97 (210) | .16 (192) | .96 | .00/.14/.06/.23/.56 | 2.91 \| 8.99 | .09 | .091 \| .267 | 3.2 \| 2.9 |
| GR00T-sp CL0 | .736 | .11 \| 2.06 | .020 \| .404 | .99 (340) | .08 (92) | .98 | **.88**/.00/.02/.01/.10 | 1.12 \| 2.26 | .84 | – | 1.0 |
| GR00T-sp CL1 | .852 | .10 \| 2.16 | .030 \| .495 | 1.00 (395) | .15 (60) | 1.00 | .72/.00/.01/.09/.18 | 1.00 \| 1.00 | .57 | .111 \| .176 | 3.5 \| 3.2 |
| GR00T-sp CL2 | .888 | .14 \| 2.48 | .028 \| .438 | 1.00 (402) | .26 (58) | 1.00 | .82/.04/.02/.02/.11 | 1.05 \| 1.43 | .60 | .071 \| .083 | 4.0 \| 4.1 |
| GR00T-sp CL3 | .862 | .08 \| 0.26 | .012 \| .021 | .88 (455) | .57 (7) | .22 | .00/.20/.00/.40/.40 | 1.05 \| 3.91 | .31 | .069 \| .110 | 4.0 \| 3.1 |
| GR00T-l10 CL0 | .468 | .84 \| 5.36 | .073 \| .423 | .98 (129) | .16 (298) | .99 | .07/.00/**.21**/.12/.60 | 3.00 \| 5.97 | .11 | – | 1.0 |
| GR00T-l10 CL1 | .466 | 1.45 \| 6.07 | .133 \| .539 | 1.00 (92) | .25 (346) | 1.00 | .00/.13/.16/.08/.62 | 2.64 \| 3.27 | .08 | **.197 \| .399** | 2.7 \| 2.3 |
| GR00T-l10 CL2 | .552 | 1.32 \| 7.02 | .107 \| .515 | 1.00 (112) | .29 (311) | 1.00 | .00/.20/.16/.14/.50 | 2.92 \| 3.10 | .06 | .196 \| .334 | 3.2 \| 3.0 |
| GR00T-l10 CL3 | .606 | 1.09 \| 6.35 | .073 \| .286 | .98 (141) | .33 (281) | .98 | .00/.09/.21/.13/.57 | 3.16 \| 5.49 | .06 | .187 \| .245 | 3.2 \| 2.8 |

---

## 2. Three-layer decomposition, closed loop and offline side by side

Offline deltas are **paired on the same decisions** (episode-bootstrap 95 % CI, `decompose.py`); the chain is B0 top-1 →
M4 (B0 ranking, mean-5) → AWM fitted on the 50 episodes (kr5 = the CL2 config) → AWM on the 10× library fitted on it
(kr5; the 500-episode CL2 arm actually uses kr8 — offline kr5 vs kr8 differ by ≤ .005 stale, 0 at step 0). The "borrowed
fit" row is `AWM(lib = current, fit_data = big)` at kr8 against `AWM cur fcur` kr8: **borrowed big-library information**
(500-episode PCA basis and per-task whitening, 50-episode candidates).

| layer | closed-loop ΔSR pp (50-ep library) | offline stale Δerr (50-ep unless stated) | offline stale ΔAURC | offline fresh Δerr | offline step-0 Δerr |
|---|---|---|---|---|---|
| **synthesis** M4 − B0 (same ranking, top-1 → mean-5) | **+9.6** [+4.8,+14.4] / **−1.2** [−6.0,+3.8] / **+11.6** [+7.0,+16.0] / **−0.2** [−5.2,+4.8] | −.054 [−.063,−.046] / −.058 [−.064,−.051] / −.074 [−.090,−.059] / −.077 [−.085,−.070] | +.008 / −.011 / −.025 / −.030 | −.057 / −.054 / −.067 / −.068 | −.031 / +.019 / −.039 / +.021 |
| **method** AWM cur kr5 − M4 (same 50-ep library, fit on it) | +3.6 [−0.6,+8.0] / **+20.2** [+15.4,+25.2] / +3.6 [+0.0,+7.2] / **+8.6** [+3.8,+13.4] | −.011 [−.017,−.004] / −.026 [−.034,−.019] / −.038 [−.047,−.030] / −.055 [−.064,−.047] | −.047 / −.046 / −.033 / −.036 | −.103 / −.097 / −.119 / −.139 | −.005 / −.044 / −.020 / −.052 |
| **borrowed fit** AWM cur fbig − cur fcur (kr8; labelled: borrowed big-library information) | not run closed loop (R3 H1 α = 1) | −.025 [−.034,−.017] / −.020 [−.026,−.013] / −.019 [−.024,−.015] / −.023 [−.030,−.017] | −.019 / −.009 / −.005 / −.008 | −.000 / −.001 / −.001 / −.001 | −.017 / −.023 / −.006 / −.000 |
| **library** AWM big kr5 − AWM cur kr5 (10× candidates *and* 10× fit) | **pending** (500-episode group); history B0-style S3 → S6: +12.2 / +6.0 / – / – | −.078 [−.088,−.068] / −.080 [−.088,−.072] / −.079 [−.087,−.070] / −.065 [−.074,−.057] (of which candidates alone, kr8: −.063 / −.059 / −.067 / −.045) | −.075 / −.062 / −.051 / −.052 | −.084 / −.044 / −.094 / −.066 | −.072 / −.089 / −.077 / −.048 |
| **recovery** V6(AWM) − AWM | −0.2 [−3.0,+2.6] / +1.2 [−2.2,+4.6] / **−2.6** [−5.6,+0.4] / **+5.4** [+1.2,+9.8]; pooled +0.95 (p .30) | not a verdict offline (recorded states do not respond); blend-only component ≤ −.007 (stand-in base) | | | |
| total B0 → AWM cur kr5 | **+13.2 / +19.0 / +15.2 / +8.4** | −.064 / −.084 / −.111 / −.133 | −.039 / −.057 / −.058 / −.066 | −.160 / −.151 / −.187 / −.207 | −.035 / −.024 / −.059 / −.030 |
| total B0 → AWM big kr5 | pending | −.143 / −.164 / −.191 / −.198 | −.114 / −.119 / −.109 / −.118 | −.244 / −.196 / −.280 / −.274 | −.107 / −.113 / −.136 / −.079 |

Absolute levels behind the table (stale err / stale AURC; gs in parentheses): B0 .654 .601 .629 .649 / .414 .394 .410 .443;
M4 .600 .543 .556 .571 (gs .611 .554 .565 .586) / .422 .383 .385 .413; AWM cur kr5 .590 .517 .518 .516 (gs .602 .530 .525 .534) /
.375 .337 .352 .377; AWM cur fbig kr8 .573 .494 .504 .491 / .370 .334 .355 .373; AWM big kr5 .511 .437 .439 .451 (gs .526 .450
.451 .464) / .300 .275 .301 .325; M8 (B0 formula over all 10× candidates, mean-5, 262 KB/entry, 50 ms/query) .536 .484 .460
.492 / .344 .328 .329 .351. Fresh err: B0 .500 .470 .531 .540; M4 .443 .416 .463 .472; AWM cur kr5 .340 .319 .344 .333; AWM big
.256 .275 .250 .267. Step 0 (inf): B0 .317 .268 .392 .248; M4 .287 .287 .353 .270; AWM cur kr5 .282 .244 .333 .218; AWM big .210
.154 .256 .169; oracle over the current library .212 .140 .233 .132.

What the side-by-side says:

* **Synthesis** is worth −.05 … −.08 stale err in every cell offline, but +9.6 / −1.2 / +11.6 pp closed loop. On l10 the mean of
  five chunks straddling a gripper transition hovers at grasp height (ideation A §A.5; CL1's l10 vote split .149 / .333 S / F
  is the largest of any arm, its spells/episode 4.0 vs B0 2.7). Offline err rewards that hedge; the loop punishes it. On spatial
  the same averaging removes the terminal-row absorption (w_term_late F .72 → .47 π0.5, .84 → .57 GR00T) and wins big.
* **Method at fixed 50-episode library** is the smallest offline layer (−.011 on π0.5-sp, CI touching zero) and the largest
  closed-loop layer on l10 (+20.2 / +8.6 pp); on GR00T-l10 the offline layer is the largest of the four (−.055) for the
  second-smallest SR gain. AWM's win on l10 is dynamic, not per-decision: fewer and shorter spells that end
  productively (A §A.3) — P(S | ≥ 2 spells) .34 vs .20 / .10, P(S | exactly 1) .93 vs .93 / .72. The offline stale trace cannot
  show this because its states are B0's.
* **Borrowed fit** is a pure metric-quality effect (fresh Δ = 0, because the fresh branch is continuity-dominated), and it lowers
  AURC too. Its per-task footprint is not uniform (§3.3): it fixes the π0.5-sp task-6 split and worsens the l10 splits.
* **Library** is the largest offline layer (≈ −.07 … −.08 stale, −.05 … −.09 AURC, −.05 … −.09 at step 0) and roughly matches the
  method layer's whole offline size. Closed loop it is unmeasured for AWM; the historical B0-style S3 → S6 gain (+12.2 / +6.0 pp)
  is the prior, and it says the 10× library helps spatial about twice as much as l10 — the opposite of AWM's method layer.

---

## 3. Offline ↔ closed-loop mapping

### 3.1 Cell level

| question | offline (stale err, 50-ep library) | closed loop (50-ep library) | agree? |
|---|---|---|---|
| does mean-5 beat top-1? | yes in 4/4 (−.05 … −.08) | yes on spatial (+9.6 / +11.6 pp), **no on l10 for either model** (−1.2 / −0.2 pp) | 2 of 4 |
| does AWM beat M4 at the same library? | yes in 4/4, smallest on π0.5-sp (−.011), largest on GR00T-l10 (−.055) | yes in 4/4 (+3.6 / +20.2 / +3.6 / +8.6 pp), largest on π0.5-l10 | direction yes, **magnitude order no**: π0.5-l10 has the 2nd-smallest offline Δ and the largest SR gain; GR00T-l10 has the largest offline Δ for the 2nd-smallest SR gain; GR00T-sp's offline Δ is 3× π0.5-sp's for the same +3.6 pp |
| is AWM ≥ B0 on every cell? | yes (−.064 … −.133) | yes in 4/4 (+13.2 / +19.0 / +15.2 / +8.4 pp) | yes |
| does recovery help? | unmeasurable (blend ≤ −.007) | no in 3/4 (−0.2 / +1.2 / −2.6 pp), yes on GR00T-l10 (+5.4, task-specific); pooled +0.95 pp, p .30 | – |
| which model gains more from vision-aware ranking? | GR00T (offline method layer −.038 / −.055 vs −.011 / −.026) | π0.5 on l10 (+20.2 vs +8.6 pp); spatial equal (+3.6 / +3.6) | no |
| AURC as a proxy? | AWM −.047 … −.036 vs M4; M4 vs B0 mixed (+.008 … −.030) | no confidence gating in pure cache; irrelevant to SR here | – |

Regime identity holds (cache cells: 100 % `prev_hit = True`; inf cells 100 % False; harness data checks), so the offline
stale cell is exactly the pure-cache regime — the disagreement is not a regime mismatch, it is that offline states are B0's
states while the traps are method-specific dynamic events at grasps / terminal rows.

### 3.2 Per task (Spearman over the 10 tasks of ΔSR against −Δerr, stale, 50-ep library; `task_map.py`)

| cell | synthesis layer (CL1 − CL0 vs M4 − B0) | method layer (CL2 − CL1 vs AWM − M4) | total (CL2 − CL0 vs AWM − B0) |
|---|---|---|---|
| π0.5-sp | +.32 | +.01 | +.33 |
| π0.5-l10 | +.72 | +.07 | +.39 |
| GR00T-sp | +.22 | −.10 | +.36 |
| GR00T-l10 | −.10 | **−.35** | −.33 |

The synthesis layer is weakly-to-moderately predictable per task from offline err in the three cells where the layer has
an effect (ideation A reported the same magnitudes with the sign convention Spearman(ΔSR, Δerr) = −.3 … −.7; that is
agreement, not disagreement); on GR00T-l10, where synthesis is worth −0.2 pp overall, the per-task correlation is −.10. The **method layer is not
predictable at all** (≈ 0 in three cells and −.35 on GR00T-l10, where the task with the largest offline gain, t9 err .784 →
.651, is the one AWM loses, .54 → .34): where AWM's ranking lowers err most is unrelated to — or opposite to — where it raises SR. This is
the quantitative basis for R3's decision to screen by closed-loop pilots rather than by offline err — supported.

### 3.3 The two collapse tasks — same symptom, different mechanism (task-restricted KPIs, n = 50 inits each)

| task | arm | SR | first-spell T/G/Z/P/H (fail) | spell share F | flips g0 F | w_term_late F | vote split < .8 S \| F | offline severe split \|v\| < .5, fit-cur kr5 / kr8 / **fit-big** | offline stale err fit-cur / fit-big |
|---|---|---|---|---|---|---|---|---|---|
| π0.5-sp t6 | CL0 | .72 | .00/.00/.00/.58/.42 | .19 | 9.29 | .56 | – | – | .633 (B0) |
| | CL1 | .50 | .12/.04/.08/.08/.68 | .56 | 1.08 | .37 | .116 \| .581 | – | .609 (M4) |
| | CL2 | **.28** | .08/**.17**/.00/.39/.36 | .54 | 5.03 | .27 | .043 \| .484 | **.163 / .163 / .045** | .610 / .595 |
| GR00T-sp t8 | CL0 | .62 | **1.00**/.00/.00/.00/.00 | .48 | 1.00 | .92 | – | – | .658 (B0) |
| | CL1 | .78 | .82/.00/.00/.00/.18 | .54 | 1.00 | .58 | .055 \| .050 | – | .621 (M4) |
| | CL2 | **.56** | **.91**/.00/.00/.00/.09 | .49 | 1.00 | .73 | .013 \| .013 | .016 / .019 / .026 | .599 / .597 |
| GR00T-sp t0 (for contrast: AWM's biggest win, .46 → .94) | CL0 / CL2 | .46 / .94 | 1.00 / .67 T | .35 / .28 | 4.44 / 1.67 | .74 / .17 | – / .110 \| .280 | .109 / .133 / .069 | 1.142 / .608 |

π0.5-sp task 6 is the under-determined-metric story (111 rows, 5 episodes, 136 dims): the 50-episode fit splits the gripper
vote on 16 % of stale decisions (≤ 12 % on every other spatial task) and the borrowed 500-episode fit brings it to 4.5 %, so
R3 H1's borrowed prior is the right lever there. GR00T-sp task 8 has **no gripper ambiguity at all** (.013–.026 under every
fit) and a 91 % terminal-row first spell with w_term_late .73: AWM parks on the last row of a library episode while the task
is not finished. The borrowed fit leaves its err unchanged (.599 → .597). The lever is the terminal guard (H1 ④), and it must
be evaluated on GR00T-sp task 8, not only on the π0.5 pilot tasks (R3's pilot set is π0.5-only).

### 3.4 Is GR00T's pattern π0.5's?

Yes on spatial, with three quantitative differences: (i) the ceiling is higher (.888 vs .800; pure inference .940 vs .986, so
the cache closes the gap to 5 pp on GR00T vs 19 pp on π0.5); (ii) the synthesis layer is larger (+11.6 vs +9.6 pp) and the
terminal-row trap dominates even more (T .88 / .72 / .82 vs .67 / .62 / .38; w_term_late F .84 / .57 / .60 vs .72 / .47 / .40);
(iii) AWM's vote split is smaller (F .083 vs .282) — the near-deterministic teacher gives cleaner gripper neighbourhoods — so
the G / P classes that appear for π0.5's synthesized arms barely appear on GR00T (.04 / .02). Recovery hurts GR00T-sp
(−2.6 pp, 37 S→F): after an exclusion switch the flips per failed episode go 1.4 → 3.9 and P(S | 1 spell) is unchanged. On
l10 only CL0 is finished: H .60 / Z .21 / P .12 / T .07, spells/failed ep 5.4, flips g0 F 6.0 — the l10 hub-and-pause pattern
of π0.5 plus a larger Z class (frozen with near-zero translation) than π0.5-l10's .09. GR00T-l10 CL1 (.466, −0.2 pp
[−5.2, +4.8], 85 S→F / 84 F→S) confirms that mean-5 synthesis alone is worth nothing on l10 for both models: it carries the
largest vote split of any arm (S .197 / F .399), spells/failed episode 6.1 vs 5.4, spell share in failed episodes .54 vs .42,
and a G class (.13) that the top-1 arm cannot have — the "mean-5 hovers at the grasp" mechanism is model-independent. AWM
(CL2 .552, +8.6 pp [+3.8, +13.4]) wins 7 of 10 tasks but loses t9 (−20 pp, hub class) and leaves the vote split high (S .196 /
F .334); the recovery wrapper (CL3 .606, +5.4 pp [+1.2, +9.8]) is the only positive recovery cell: it lifts t6 / t4 / t8 / t7
(+26 / +16 / +14 / +10 pp) and sinks t9 / t5 (−16 / −8 pp), halves the failed-episode spell share (.515 → .286) without removing
spells from failed episodes (.98 still have one), and raises failed-episode gripper flips 3.1 → 5.5 (t6: 10.8). Where it wins
the first spell is G / Z-dominated (t6 G .26–.28, t4 Z .52–.71); where it loses it is H-dominated (t9 H .63–.67) — a forced
trajectory switch helps out of a gripper-split or frozen trap and hurts in a hub, which is consistent with the null on
π0.5-l10 (+1.2) and the loss on the terminal-row cells (GR00T-sp −2.6).

Offline the GR00T cells behave like π0.5's with two exceptions already known from R1/R2: whitened vision ≈ whitened state on
GR00T (vision-only AWM at 50 ep +.017 [+.012, +.022] on GR00T-sp vs −.023 [−.028, −.018] on π0.5-l10 against joint), and
V4's plain cosine z-sum loses to AWM by .05 / .03 on GR00T while tying on π0.5 (§5).

### 3.5 Where the remaining offline error sits (breakdown, stale, AWM cur kr5 vs B0)

| slice | B0 err | AWM cur kr5 err | note |
|---|---|---|---|
| early / mid / late third | .410 .434 .417 .468 / .761 .619 .681 .684 / .770 .742 .780 .783 | .369 .376 .362 .409 / .681 .529 .548 .538 / .698 .638 .634 .594 | AWM's gain grows with the phase (late −.07 … −.19) |
| B0-successful vs B0-failed episodes | .418 .402 .429 .445 / .868 .676 .885 .730 | .386 .357 .377 .396 / .773 .577 .696 .563 | 55–75 % of the stale error mass is in episodes the deployed cache failed |
| gripper transition steps (7–18 % of decisions) | .947 .808 .864 .773, grip .41 .39 .38 .42 | .842 .703 .693 .638, grip .45 .39 .40 .41 | AWM lowers err but **not** the gripper mismatch at transitions (an information problem, R2 ideation B F5) |
| oracle over the 50-ep library | .395 .277 .361 .261 | regret .195 .240 .157 .255 | the 10× library lowers the oracle to .282 .190 .229 .179 |

---

## 4. Library scale for every row

| row (method, config) | library | episodes | entries | B/entry (retrieval) | keys total | fixed per suite | fit pickle shipped to the servers (MB) | vs deployed pkl 431 / 1,103 / 429 / 1,068 MB | ms/query offline (timing conc. 4) |
|---|---|---|---|---|---|---|---|---|---|
| CL0 B0 native (top-1) | current | 49 / 50 / 50 / 50 | 1,018 / 2,640 / 1,063 / 2,645 | 262,272 (π0.5) / 262,176 (GR00T) | 255 / 660 / 266 / 661 MB | – | (native pkl) | 1× | 1.5 |
| CL1 M4 (B0 rank + mean-5) | current | same | same | 262,272 / 262,176 | same | – | **269 / 697 / 281 / 700** | 0.6× | 3.5 |
| CL2 AWM joint kr5, fit on current | current | same | same | **580** (f32 136-d; f16 272) + action 280 / 448 B | 0.6 / 1.5 / 0.6 / 1.5 MB | PCA-64 ×2 cams 17.0 MB (f16 8.5) + per-task mean/std/W/W0 ≈ 1.5 MB | **21.3 / 24.6 / 22.2 / 26.7** | **5 % / 2 % / 5 % / 2.5 %** | 2.7–2.9 (0.7 ms is the two 32,768 → 64 projections; on the GPU that is µs) |
| AWM r32 codes (offline variant) | either | | | 288 | | + rank-32 W | | | 2.8 |
| AWM cur, borrowed fit (R3 H1 α = 1) | current candidates, **500-episode fit** | 49–50 candidates / 500 fit | same as CL2 | 580 | same | same (basis and W from the 500 ep) | ≈ CL2 (same arrays) | ≈ CL2 | 2.7 |
| CL3 V6(AWM cur kr5) | current | same | same | 580 + 46 (next, episode, step, progress, rs8) | | + 2.6 MB task-mean raw keys | **26.0 / 32.6 / 27.9 / 36.7** | 6 % / 3 % / 6.5 % / 3.4 % | + 1.3 (wrapper) |
| oscl500 CL0 / CL1 M8x (B0 formula over all 10× candidates; exact M8, 4–10× faster) | 10× | 500 | 10,909 / 29,472 / 11,751 / 29,631 | 262,176 | **2.86 / 7.73 / 3.08 / 7.77 GB** | – | (fitted at server start) | 6.6–7.3× **not deployable**; reference arm only | 50 (M8, conc. 4) |
| oscl500 CL2 AWM big kr8 | 10× | 500 | same | 580 | 6.0 / 16.3 / 6.5 / 16.4 MB codes (+ 3.1 / 8.3 / 5.3 / 13.3 MB `[H,7]` actions) | 18.5 MB | **47.0 / 94.1 / 58.2 / 117.3** (the pickle carries the full `(H,32)` f32 action table: 14 / 38 / 24 / 61 MB) | 11 % / 8.5 % / 13.6 % / 11 % | 2.9 |
| oscl500 CL3 V6(AWM big) | 10× | 500 | same | 626 | | | **66.3 / 141.7 / 87.7 / 187.9** (two copies of the action table) | 15 % / 13 % / 20 % / 18 % | 4.2 |
| V4 PCA-32 cosine z-sum, kernel-8 (offline only) | current / 10× | 50 / 500 | | **292** | 0.3 / 0.7 / 0.3 / 0.7 MB; 3.0 / 8.2 / 3.3 / 8.3 MB | PCA-32 basis 8.4 MB | – | | 1.1–1.4 |
| T2 MLP metric (offline only, GPU fit) | current / 10× | | | 544 | | + MLP 136→64 | – | | 2.8–4.7 |
| V7 confidence wrapper | inherits base | | | + 46 | | maps < 10 KB | – | | + 1.1 |

Costs that do not scale with entries: the 17 MB PCA basis dominates the 50-episode AWM pickle (codes are 0.6–1.5 MB), so
the deployable AWM library is 20–27 MB at either scale of *candidates*; the 10× pickles are larger only because they carry
the 10× action payloads in the stored `(H, 32)` f32 layout (a `[5, 7]` f32 payload would cut them to ≈ 26–47 MB, ideation A
§g). Fit time: AWM 0.1–0.4 s per cell at 50 ep (PCA cached), ≈ 30 s per suite at 10×; V7 adds 6–65 s of LOEO calibration.

---

## 5. Per-method verdicts for the offline families (stale err / stale AURC unless stated; 50-ep = the deployed scale, 10× = 500 episodes)

### G1 AWM (action-whitened Mahalanobis metric on [PCA-64 v0, PCA-64 v1, rs8], kernel-16) — **KEEP, the main line**

| variant | 50-ep library | 10× library | verdict |
|---|---|---|---|
| joint, kr5 (CL2-50 config) | .590 .517 .518 .516 / .375 .337 .352 .377; step 0 .282 .244 .333 .218; fresh .340 .319 .344 .333 | .511 .437 .439 .451 / .300 .275 .301 .325 | keep; kr5 at 50 ep (step 0 .04 better than kr8 on π0.5-sp / GR00T-sp, stale within .01) |
| joint, kr8 (CL2-500 config) | .598 .514 .524 .514 / .389 .343 .360 .381; step 0 .321 .275 .371 .225 (the early-step W0 fitted on ≤ 111 rows is worse than B0 at step 0 on π0.5-sp) | .509 .435 .437 .446 / .302 .276 .301 .325; step 0 .211 .154 .256 .168 | keep at 10× only |
| borrowed fit (cur candidates, 500-ep fit) | .573 .494 .504 .491 / .370 .334 .355 .373; step 0 .304 .252 .364 .225 | = big fbig | refine → R3 H1 (label "borrowed big-library information"); per-task side effects §3.3 / §6 |
| vision-only | .597 .491 .541 .509 (vs joint kr8: −.001 / −.023 / **+.017** / −.005; paired CIs exclude 0 on the middle two) | .501 .432 .444 .451 (−.008 / −.003 / +.007 / +.005); step 0 without the joint branch .276 / .201 on GR00T vs .256 / .168 | ablation row only; the state block is needed at step 0 on GR00T and on GR00T-sp stale |
| r32 codes (288 B) | – | .507 .438 .435 .448 (≤ .003 from full) | keep as the compressed variant |
| λ .01 | – | .506 .433 .433 .443 stale, but step 0 .220 .164 .274 .191 (+.01 … +.02) | drop (step 0 loss) |
| state ×3 (GR00T only) | – | .426 / .450 (−.011 / +.004) | drop (one cell, one hundredth) |
| insurance V3 (norm-preserving mean + gripper hysteresis) | .628 .539 .541 .550 (**+.03 … +.04**) | .540 .460 .459 .471 (+.03) | drop offline; never tested closed loop (CL3 had it off) — superseded by R3 H1's gripper commitment |
| lc1 (fresh λ_c = 1) | – | fresh .261 .278 .255 .269 vs .259 .275 .254 .269 | drop (no effect) |
| noearly (no step-0 metric) | – | step 0 .213 .161 .280 .229 vs .211 .154 .256 .168 | keep the early fit at 10× (−.02 / −.06 on GR00T); at 50 ep it is the weak point (see kr8 row) |

Deciding numbers: the only R2 method run closed loop; +13.2 / +19.0 / +15.2 pp over B0 at 49–50 episodes; beats M8 (the
R1 best, 262 KB/entry) on the 10× library in 8/8 cells (paired stale −.026 [−.032, −.021] / −.048 [−.054, −.043] / −.022
[−.029, −.017] / −.046 [−.055, −.038]; fresh −.07 / −.03 / −.11 / −.08) at 1/450 of the entry size. Weakness: the 136-d
per-task covariance from 74–125 rows per task at 50 episodes (task-6 split; the 50-ep early-step W0). Fresh-regime branch is
matched or slightly beaten by V5 (below).

### G2 cosine family — V4 **KEEP as π0.5 backup / ablation**, V5 **KEEP as the fresh branch**, M8x **reference only**

| variant | 50-ep | 10× | vs AWM |
|---|---|---|---|
| V4 p32 st1 kernel-8, align0 (main) | .585 .518 .572 .544 / .360 .337 .352 .385; step 0 .263 .205 .314 .187; fresh .427 .396 .458 .460 | .526 .460 .457 .469 / .311 .288 .303 .329; step 0 .211 .161 .256 .167 | paired vs AWM cur kr5 (cache cells incl. step 0): +.006 [−.001, +.012] / −.001 [−.008, +.007] / **+.052 [+.036, +.068]** / **+.027 [+.019, +.035]** — ties AWM on π0.5, loses on GR00T; AURC tie |
| st = per-model (.5 π0.5 / 3 GR00T) | .581 .511 .567 .537 | .524 .454 .442 .469 | best V4; still ≥ .015 behind AWM big on 3/4 |
| st 3 / st .5 (10×) | – | .532 .477 .442 .469 / .524 .454 .465 .473 | the state weight must be per model (R1's finding stands) |
| mean-5 instead of kernel-8 | – | .535 .468 .466 .478 (+.01) | kernel wins |
| PCA-64 | – | .522 .458 .453 .468 (≤ .004) | not worth 2× bytes |
| no step-0 alignment | – | step 0 .221 .167 .271 .200 (+.01 … +.03) | keep alignment |
| basis fitted on the 500 episodes (LcurFbig) | .586 .517 .567 .545 (≈ 0) | – | the PCA basis is not the borrowed information that matters (AWM's whitening is) |
| **V5** fresh fusion λ .5 / 1 (continuity + visual z-sum + state, all candidates) | fresh .324 .314 .330 .329 / AURC .225 .224 .235 .241 (λ .5) | fresh .253 .273 .246 .263 / .177 .191 .174 .190 (λ 1) | vs AWM's fresh branch: 50 ep −.016 / −.005 / −.014 / −.004; 10× −.003 each; AURC −.006 … −.019 at 50 ep. Keep for the mixed system's MISS branch (irrelevant to pure cache) |
| M8x (exact pruned M8) | – | = M8: .536 .484 .460 .492; 262 KB/entry | 500-group CL0 / CL1 reference only; 2.9–7.8 GB not deployable |

V4's virtue is that it has **no per-task covariance to fit** (a task-agnostic PCA basis + z-scores), so it cannot suffer the
task-6 under-determination — but it was never run closed loop and its offline tie with AWM on π0.5 says nothing about its
spells (§3). If R3 wants a control without borrowed information, a V4 (st = pm) pilot on the π0.5 trap tasks is cheap.

### G3 mechanisms — V6 **DROP** (blend and recovery), V7 **KEEP (re-measure on AWM)**

Stand-in base VZS (r01 M9c vision z-sum + kernel-8): current .579 .517 .567 .538, 10× .520 .457 .451 .467.

| | 50-ep | 10× | verdict |
|---|---|---|---|
| V6 blend only (next(served) forced into the kernel set) | .574 .514 .560 .533 (−.005 / −.003 / −.007 / −.005) | .517 .456 .449 .467 (−.003 / −.001 / −.002 / .000) | below B-P2's own kill line (≤ −.005 on 10×) in 4/4; drop |
| V6 full (blend + detector + escalating recovery, ot_still) | .615 .543 .576 .573 | .545 .479 .436 .480 | offline not a verdict; closed loop (around AWM) −0.2 / +1.2 / −2.6 / +5.4 pp, pooled +0.95 pp (142 / 161, p .30); the GR00T-l10 gain is task-specific (t6 +26, t9 −16); spells broken but episodes not escaped on the three other cells; GR00T-sp t8 −6 pp, π0.5-sp t6 −2 pp; drop as a general pure-cache mechanism |
| V6 detector flags (still ∧ vis-cos, stuck_n ≥ 2, terminal, overtime ∧ lag) | flagged err ≈ 2× unflagged (.91 / .69 / 1.07 / .69 vs .43 / .44 / .38 / .43) | .81 / .62 / .81 / .59 vs .39 … .43 | keep as MISS triggers (R3 H3): in the CL2 logs `still > 1.98` occurs on .152 / .225 of failed-episode decisions vs .001 / .084 in successes (π0.5 sp / l10), on 0 / 0 for GR00T-sp |
| V7 iso / bin10 predicted-err confidence | stale AURC .357 .341 .348 .383 vs base .422 .364 .384 .420; fresh .272 .250 .309 .313 vs .334 .302 .361 .368 | stale .309 .293 .299 .326 vs .373 .343 .355 .385; fresh .199 .173 .246 .223 vs .239 .204 .299 .274 | keep: −.03 … −.07 in 16/16 cell × regime, bin10 = iso ± .001, one predicted-err scale across regimes. **Not yet measured on AWM** (AWM's own conf AURC .375 .337 .352 .377 at 50 ep is already .05 below the stand-in's, so V7's margin over AWM will be smaller); the 50-ep calibration set has no failed episodes (current libraries are 100 % success) |

### G4 T2 trained metric — **DROP** (confirmed)

Kernel kref 5 reproduction of AWM inside the family matches AWM exactly (V8_t2_awm_* = AWM kr5 to 3 decimals). Against
that baseline on the 10× library: plain MLP metric .515 .456 .451 .490 (**+.004 / +.019 / +.012 / +.039 worse**; paired
+.006 [−.000, +.012] / +.021 [+.015, +.027] / +.014 [+.009, +.019] / +.043 [+.035, +.051]), linear re-fit .511 .466 .446 .505
(fails to reproduce the closed form on l10: +.03 / +.05), MLKR-loss MLP .504 .426 .422 .448 (−.007 / −.011 / −.017 / −.003:
below the ≥ .02 in ≥ 3/4 cells bar in 4/4). At 50 ep: MLP .577 .549 .536 .567 vs AWM .590 .517 .518 .516 (−.013 / +.032 /
+.018 / +.051). With the borrowed 500-episode fit the MLP is .550 .478 .484 .498 vs AWM-borrowed .573 .494 .504 .491
(−.023 / −.016 / −.020 / +.007) — the only configuration where training earns anything, and it is the borrowed one. Fresh
and step 0 are identical to AWM (they do not use the trained metric). GPU fit, 544 B/entry. Keep the two analysis scripts
(`sign_faithful.py`, `decompose.py`), which this report uses.

### Cross-method facts worth keeping

* Gripper-sign-faithful err is +.012 … +.017 above the mean err for every averaging method (M4 +.011 … +.015, AWM +.012 …
  +.018, V4 +.008 … +.016) and does not change any ranking; top-1 methods are unaffected.
* Every method's stale AURC at 50 ep is ≥ .34; the oracle's is .21 / .16 / .21 / .16 — confidence is far from solved at the
  deployed scale, and in pure cache it does not matter (no gating). It matters for R3 H2/H3.
* Duplicate variants: `AWM_joint_big_fbig` / `_lc1` / `_noearly` differ only in fresh / step 0 as designed; `AWM_vis_big_fbig`
  / `_s0joint` only at step 0; V7 iso / bin10 identical to ± .001. No accidental duplicates found.

---

## 6. Risks and what R3 should know

1. **Statistical floor of the exam.** 500 paired inits give ± 4–5 pp; the R3 pilot (100 inits, 5 trap tasks) gives ± 9–10 pp
   and, on the R2 arms restricted to that subset, orders CL1 > CL2 on spatial (.74 vs .70) — the opposite of the full arms
   (.764 vs .800). A pilot can detect a task collapse (t6 .75 → .40) or a trap-class shift; it cannot rank two methods 3 pp
   apart. R3's H1 switches should be judged on paired counts and on the per-task rows, and confirmed on 500 inits.
2. **R3 H1 (borrowed prior) is aimed at the right mechanism on π0.5-spatial and at the wrong one on GR00T-spatial.** The
   borrowed fit removes the task-6 vote split (.163 → .045) but does nothing for GR00T-sp task 8 (terminal-row absorption,
   split .013). Evaluate the terminal guard on GR00T-sp task 8 explicitly; R3's pilot tasks are π0.5's.
3. **The borrowed prior has an l10 side effect on both models**: severe split .232 → .426 on π0.5-l10 task 6 (C's number
   reproduced: .229 → .450 at kr8), .264 → .323 on task 0; on GR00T-l10 the overall severe split rises .201 → .263 (task 0
   .257 → .500, task 7 .258 → .374) even as err falls .514 → .491. The α = .5 hedge is necessary, and the pilot should log the
   per-task vote split, not only SR.
4. **The synthesis layer is not universally positive**: −1.2 pp on π0.5-l10 and −0.2 pp on GR00T-l10 (both with the largest
   vote splits of their groups), against +9.6 / +11.6 pp on spatial. Offline err said −.05 … −.08 in all four cells. Because
   this is exactly a case where offline err and the loop disagree, a "mode-consistent" or committed-gripper synthesis
   (offline-negative per C: +.025 … +.034 err) deserves a closed-loop pilot on the l10 trap tasks regardless of its offline cost.
5. **Recovery is not a general pure-cache lever** (pooled +0.95 pp over 2,000 paired inits, p .30; −2.6 pp on GR00T-sp with
   37 S→F) **but it is the only thing that moved GR00T-l10** (+5.4 pp [+1.2, +9.8], p .018, 121 discordant pairs against ≈ 30
   expected from rerun noise), by lifting G / Z-trap tasks (t6 +26, t4 +16) and sinking hub tasks (t9 −16). R3 correctly dropped
   further *general* recovery arms; the one targeted question left is whether a forced trajectory switch gated on the G / Z
   flags (not on identical picks) keeps the GR00T-l10 gain without the hub losses — cheaper as a MISS trigger than as a
   pure-cache rule. The detector's flags remain the best MISS triggers we have measured (flagged err 2× unflagged; `still` 15–22 % of
   failed-episode decisions vs ≤ 8 % in successes on π0.5) — but C showed raw low confidence is a *late* detector; the spell
   proxies here are also retrospective.
6. **Kref mismatch in the 500-vs-50 pairing.** CL2-50 is kr5, CL2-500 is kr8 (default kwargs). Offline the two differ by
   ≤ .005 stale and 0 at step 0 on the 10× library, so the library layer will be readable, but note it in the addendum.
7. **The 500-episode CL0 / CL1 arms are M8x (262 KB/entry, 2.9–7.8 GB)**: they measure the library layer for the B0 formula
   (a replication of S6 .810 / .516) and are not deployable candidates.
8. **Confidence calibration at the deployed scale has no failed episodes.** V7's LOEO pseudo-queries on the current libraries
   (100 % successful episodes) never see drifted states; the 10× libraries have 13–73 failed episodes. R3 H3 must calibrate on
   the closed-loop logs (held out by init) or borrow the 10× library — and label it.
9. **V3 insurance was never tested in the loop** (CL3 had it off, C's correction). Its offline cost (+.03 err) is real, but the
   closed-loop question "does hysteresis reduce chatter" is still open; H1's stateful gripper commitment is the successor.
10. **Latency reading.** Plugin arms' `infer_ms` includes the native shadow search; method cost is `q_us` (AWM 2.7–4.5 ms p50
    on a loaded server vs native 2.8–10.6 ms). Offline AWM spends 0.7 of its 1.1 ms single-thread on the two 32,768 → 64
    projections that belong on the GPU next to the pooled key.
11. **No arm has a replicate yet.** The native reruns match trace_dual to 0 / −1.2 / 0 / 0 pp; R3's CL2 rerun in the pilot is
    the first same-method replicate and should be reported as the noise floor of the exam.
12. **Library layer for AWM is the largest unknown**: offline it equals the whole method layer (≈ −.08 stale, −.05 … −.09 step 0);
    history says +12 / +6 pp for B0 top-1. If AWM-500 lands near .85 / .70 on π0.5, the owner's deployability question (bytes vs
    trajectories) becomes decisive: at 580 B/entry the 10× AWM library is 47–117 MB (26–47 MB with `[5, 7]` payloads) against
    431–1,103 MB deployed today.

Where the R2 data supports R3 SELECTION: pilot-based screening over offline err (§3.2); dropping recovery (§1.2); keeping
the vision-mandatory joint metric (vision-only loses on GR00T-sp at 50 ep); the α = .5 hedge (§6.3); logging per-task KPIs
(§1.3). Where it contradicts or extends it: the GR00T task-8 mechanism (terminal, not gripper) must be in the pilot design;
V4 (st = pm) is a legitimate no-borrowed-information control R3 does not list; V7 has to be re-measured on AWM before it is
used as the handoff confidence.

---

## 7. Pending arms and regeneration

**Pending after addendum 1 (01:0x CDT):** the 500-episode group only — 16 `oscl500_*` arms (≈ 3 h), now scheduled ≈ 03:00 CDT
after the R3 pilots (the 50-episode chain finished 00:51 CDT; its last two arms started with 13–16 GB already in use on the
4090 by another process, without incident).
The `CHAIN_STOPPED at oscl500_p_sp_cl0 / PORT_BUSY` line of 23:22 in `r02_g500/runs/chain.log` is the premature relay (no
arm ran; `oscl500_p_sp_cl0.ERROR` is that leftover). Rows / sections that will change with addendum 2: §1.1–1.4 the `oscl500_*` rows; §2 the library layer (500 vs 50 per CL,
paired on the same inits; note the kref 5 vs 8 mismatch of §6.6); §4 nothing (sizes already listed); §6.12.

Everything regenerates with (read-only on the run roots; CPU 30-33,74-77):

```bash
bash /home/weiland/.claude/jobs/a607dd74/tmp/analysis_r02/regen_cl.sh          # kpi.py per group/run for every state/<arm>.DONE, + 500-vs-50 pairs -> cl/*.json|md, cl/STATUS.txt
taskset -c 30-33,74-77 .venv/bin/python /home/weiland/.claude/jobs/a607dd74/tmp/analysis_r02/cl_table.py     # -> cl/CL_TABLES.md (arm / paired / per-task / spell tables)
taskset -c 30-33,74-77 .venv/bin/python /home/weiland/.claude/jobs/a607dd74/tmp/analysis_r02/task_map.py     # -> task_map.md (per-task offline<->loop, vote splits, latency)
taskset -c 30-33,74-77 .venv/bin/python /home/weiland/.claude/jobs/a607dd74/tmp/analysis_r02/offline_regimes.py --out .../offline --procs 4   # -> offline/regimes.csv, regimes_wide.md
```

Other scratch outputs used above: `decomp/{awm_kr8,awm_kr5_cl2,v4,t2_mlp}/*.csv` (three-layer offline decomposition with
CIs), `profile/cmp_*` (paired compare), `profile/bd_{b0,awm_cur_kr5}` (breakdown), `cl/task_{p_sp_t6,g_sp_t8,g_sp_t0}.md`
(task-restricted KPIs). Coverage was read from the existing `profile_cache/reports/coverage/{current,bpool_cs,bpool_all}`.
