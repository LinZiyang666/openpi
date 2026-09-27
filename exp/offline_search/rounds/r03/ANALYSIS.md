# R3 analysis — pure-cache trap prevention (H1) and the mixed HIT/MISS cache (H2/H3), closed loop (2026-09-27 09:27 CDT; addendum with the last three l10 arms 10:0x CDT)

Analysis agent (fable), R3 of `logs/offline_search_exploration.log.md`. Cell order: **π0.5-sp / π0.5-l10 / GR00T-sp /
GR00T-l10**. Closed loop = A-pool 500 inits (pilots: 5 trap tasks × ep_idx 0–19 = 100 inits), servers on the 4090, timan107
workers; run roots `/home/weiland/trace_runs/os_closed_loop/{r03_pilot,r03_full,r03_mx}` (+ `r02_g50`, `r02_g500`, and the
trace_dual pure-inference arms `dual_20260923/runs/tr_pi05_{sp,l10}_inf` for pairing). SR = journal success (collect.py
judge); every paired number is on the common (task, init) inits with exact two-sided McNemar p and a multinomial bootstrap
95 % interval of ΔSR (10,000 reps). KPIs come from `closed_loop/ops/kpi.py` (`--recon AWM3=awm:5`; validated on R2 in
`rounds/r03/h4_kpi/r02_kpi.md`); the mechanistic numbers from the server decision logs' extras. Everything was computed
read-only with `taskset -c 18-33,62-77`, ≤ 27 processes, BLAS 1 thread; scripts and every intermediate table are in
`/home/weiland/.claude/jobs/a607dd74/tmp/analysis_r03/` (§7).

**Library scale (owner rule; stated once here and next to each conclusion).** *Current* = the deployed library: 49 / 50 / 50 /
50 episodes, 1,018 / 2,640 / 1,063 / 2,645 entries; deployed pkl 431 / 1,103 / 429 / 1,068 MB (262 KB/entry). *10×* = 500
episodes (`bpool_cs` / `bpool_all`): 10,909 / 29,472 / 11,751 / 29,631 entries. AWM3 codes 580 B/entry (581 with the terminal
mask), MixedJudge 626 B/entry; fit pickles actually shipped: 20.9–26.8 MB (current), 63.0 / 134.7 MB (10×, wrapper), B0Current
254.6 / 660.4 MB (raw keys). Full table §4.

**Status.** Everything is complete (queue `oscl_queue2d` ended ≈ 10:1x CDT): all 13 pilot arms (6 skipped on purpose), all 6
full pure-cache arms, 17 mixed arms (`r3mx_p_{sp,l10}_{b0h70, awm_h70, ev_h70, g, perk3}`, `_awm_h50` and `_awm500_h70` on
both suites, and the three l10 controls `b0perk3`, `perk5`, `g500` that ran last on 2 servers); `_ev_h50` skipped on both suites.
§2.7 states what the last three arms settled.

---

## 0. Headline verdicts

1. **The mixed cache reaches the policy on both π0.5 suites, at 43–61 % of the policy's inference.** V7 + guards on AWM (50-episode
   library) at h ≈ .67: spatial **.980** (IR .43; vs pure inference .986: 9 S→F / 6 F→S, p .61) and at h ≈ .46 **.986** (IR .61;
   7 / 7, p 1.0); l10 **.816** (IR .44; −2.8 pp, p .21). With the 500-episode library at h ≈ .66: l10 **.872** (IR .44; +2.8 pp
   over the policy, 49 policy failures rescued / 35 lost, p .16), spatial .978; **guard-only on the 500-episode AWM reaches the same
   SR at half the inference: l10 .864 at IR .24 (formula; .29 measured), +2.0 pp over the policy (44 / 54, p .36) and −0.8 vs
   awm500_h70 (31 / 27, p .69)**. Paired against pure cache CL2 on the same inits the mixed arms are +18.0 / +18.6 pp (50 ep) and
   +2.4 (p .036) / **+10.4** (awm500_h70) / **+9.6** (g500; 12 / 60, p < 1e-4) pp (500 ep).
2. **The SR gain is not just more inference.** Every AWM-based mixed arm sits 6–13 pp *above* the straight time-share line between
   pure cache and pure inference at its own IR (§2.2): the policy is worth 5–7 pp of SR per 0.1 of inference ratio when it is
   inserted into the cache loop, against 2.2–2.5 pp per 0.1 IR for replacing the cache wholesale.
3. **Targeting beats periodic MISS on spatial; on l10 periodic MISS dominates targeting at both budgets.** At matched realized h ≈ .67,
   V7 + guards vs the periodic k = 3 control: spatial **+4.2 pp** (7 / 28, p .0005), l10 −1.6 pp (55 / 47, p .49); at the low budget
   (IR ≈ .32) periodic k = 5 beats guard-only on l10 by **+5.2 pp** (.792 vs .740; 45 / 71, p .020). On l10 the gain comes from "the
   policy is called every few decisions" (drift interruption) and the judge's timing adds nothing; the judge line's own kill
   condition (periodic ≥ targeted − 2 pp, B-P2) is met there at both budgets. Early events + burst add nothing (l10 +0.6, p .82)
   and cost −2.4 pp on spatial (p .023). **B0 is dead on both sides of the mixed loop** (`b0perk3` splits the 18.4 pp l10 gap): with
   the same periodic judge the B0 *selector* costs −11.8 pp vs AWM (.714 vs .832; 96 / 37, p < 1e-4), and with the same B0 selector
   the B0 fused-score *judge* costs a further −8.2 pp vs periodic (.632 vs .714; 54 / 95, p .001) — its MISSes come in blocks
   (MISS-after-MISS .90, mean MISS run 8.8) and start late.
4. **Guard-only is the cheap operating point, and on the 500-episode library it is the best l10 point of the round**: 50 ep IR .27 / .32
   → .888 / .740 (+8.8 / +11.0 pp over CL2, p < 1e-4); 500 ep l10 IR .24 → **.864** (+9.6 over CL2-500, +12.4 over the 50-ep guard arm,
   31 / 93), 8.8 pp above the time-share line and 11 pp of SR per 0.1 IR — the most inference-efficient point measured. 100 % of its
   MISSes are forced (no-progress .64, stuck .23, terminal∧closed .08 on the 500 base); it interrupts 81–91 % of successful episodes
   but with only 1.6 / 5.8 / 2.8 MISSes each.
5. **Pure cache, the borrowed prior α = .5 is real but small on π0.5 and nothing on GR00T** (500 inits, paired vs R2 CL2): +4.8
   [+1.6, +8.0] (21 / 45, p .004) / +4.4 [+0.6, +8.4] (40 / 62, p .037) / +2.2 [−1.0, +5.4] (p .21) / −0.8 [−5.0, +3.4] (p .78; GR00T-l10
   t8 .42 → .08). The non-borrowing ridge-1.0 twin gets **+4.0** [+1.2, +6.8] (p .009) on π0.5-sp, i.e. ≈ 80 % of the borrowed gain
   without big-library information; a05 vs rm1 +0.8 (p .68). Label: every `a05` arm = **borrowed big-library information**.
6. **All three "prevent the trap" gripper / terminal switches are dead, mechanistically**: symmetric gripper commitment .10 (the
   grasp vote sits at +.36 for ~13 decisions, the served sign stays OPEN, the robot hovers; 59 / 100 episodes never close);
   release-guard .01 on l10 (releases need |v| ≥ .6 twice; 91 / 100 episodes never release the object, all time out) and −3 pp on
   spatial; terminal-row masking .23 on π0.5-sp (rows masked on 92 % of decisions, the trap moves from T to pause / hub rows: first
   spell P .53 / H .40) and .60 on GR00T-sp even with the progress-only gate (t3 .95 → .30; the gate is open at 86–100 % of the
   failing spells, so the terminal absorption simply happens later). §1.3.
7. **The library layer still dwarfs every R3 switch.** From the best 50-episode R3 arm to the 500-episode AWM (same inits):
   +10.6 / +9.4 / +5.6 / +16.2 pp (all p ≤ .0002). On spatial the 500-episode *pure cache* (.954 at IR .152) dominates every
   50-episode mixed arm below IR ≈ .43; on l10 the 500-episode cache (.768) is what the mixed judge should be wrapped around
   (awm500_h70 .872 vs awm_h70 .816: +5.6 pp, p .005, at equal IR; g500 .864 vs g .740: +12.4 pp at IR .24 vs .32).
8. **Noise floor, quantified**: same-method reruns flip 6 % (π0.5-sp), 13 % (π0.5-l10), 2 % (GR00T-sp) of inits ⇒ sd(ΔSR) 2.4 / 3.6 /
   1.4 pp at n = 100 and 1.1 / 1.6 / 0.6 pp at n = 500. A 100-init l10 pilot cannot see anything below ± 7 pp; the l10 pilot
   ordered a05 at −6 pp (vs its own rerun) where the 500-init arm gives +4.4 — the pilot subset (5 trap tasks) is not where a05
   gains on l10. Pilots screen collapses (they caught three) and nothing finer.
9. **In the mixed loop the failure anatomy changes**: no identical-pick spells survive (the guards fire at stuck_n ≥ 2 / no-progress
   3, the periodic arm's max HIT run is 2), failures still time out at the cap, and in 50–80 % of the failed episodes the policy held
   the majority of the decisions (28 / 92 l10 awm_h70 failures end with ≥ 20 consecutive MISSes). The residual failures are
   policy failures from cache-perturbed states plus the policy's own (l10: 32 of awm_h70's 92 failed inits are inits the policy
   also fails).
10. **Latency / infra caveat**: the MX arms ran with 4, 3 or 2 full-model servers sharing one 4090 (plus another project's
    training), so `s1` 290–530 ms and `s23` 1.6–3.9 s per MISS are load numbers, not the single-server CUDA-graph reference
    (117 / 483 ms); the IR formula (0.152 + 0.848·MISS share) and the measured stage ratio agree within .01–.02 anyway because the
    loaded s1 : s23 ratio (≈ .15) happens to match the formula's. The policy is identical across server counts.

---

## 1. Pure-cache results (H1)

### 1.1 Noise floor of the exam (same method re-run on the pilot subset)

| cell | R2 CL2 → R3 CL2 re-run (100 inits) | discordant | sd(ΔSR) n = 100 | 95 % band n = 100 | sd n = 500 | 95 % band n = 500 | expected discordant pairs at n = 500 under pure noise |
|---|---|---|---|---|---|---|---|
| π0.5-sp (tasks 6,9,0,4,1) | .70 → .70 | 3 S→F / 3 F→S | 2.4 pp | ± 4.8 | 1.1 | ± 2.1 | 30 |
| π0.5-l10 (tasks 0,4,6,8,7) | .52 → .59 (+7.0, p .09) | 3 / 10 | 3.6 pp | ± 7.1 | 1.6 | ± 3.2 | 65 |
| GR00T-sp (tasks 8,2,3,7,5) | .77 → .75 | 2 / 0 | 1.4 pp | ± 2.8 | 0.6 | ± 1.2 | 10 |

The l10 rerun itself moved +7 pp (3 / 10, one-sided in favour of the rerun, p .09): GPU nondeterminism on l10 is not symmetric
noise at n = 100. Reading rule used below: a pilot ΔSR is a signal only outside ± 5 pp (spatial) / ± 7 pp (l10) *and* with a
lopsided S→F / F→S split; a 500-init ΔSR needs its McNemar p on top of clearing ± 2 / ± 3 pp.

### 1.2 Pilot table (100 inits per arm; paired vs R2 CL2 on the subset and vs the pilot's own CL2 re-run)

| cell | arm (all 50-ep library) | SR [Wilson] | vs R2 CL2: ΔSR, S→F / F→S, p | vs CL2 re-run: ΔSR, S→F / F→S, p | per-task SR (pilot tasks) | spells / failed ep | first-spell T/G/Z/P/H (fail) | flips g0 F | w_term_late F | vote split < .8 S \| F |
|---|---|---|---|---|---|---|---|---|---|---|
| π0.5-sp | R2 CL2 (subset) | .70 [.60, .78] | – | – | t6 .40 t9 .65 t0 .75 t4 .85 t1 .85 | 2.57 | .30/.20/.03/.13/.33 | 2.93 | .35 | .039 \| .336 |
| π0.5-sp | cl2ref (re-run) | .70 [.60, .78] | 0, 3 / 3, 1.0 | – | .40 .60 .80 .85 .85 | 2.87 | .40/.07/.00/.13/.40 | 3.13 | .33 | .049 \| .358 |
| π0.5-sp | a1 (borrowed, α = 1) | .74 [.65, .82] | +4, 6 / 10, .45 | +4, 8 / 12, .50 | .45 .75 .90 .80 .80 | 2.46 | .50/.00/.23/.19/.08 | 1.04 | .49 | .060 \| .247 |
| π0.5-sp | **a05 (borrowed, α = .5)** | **.79** [.70, .86] | **+9**, 5 / 14, .064 [+1, +18] | **+9**, 3 / 12, **.035** [+2, +17] | .50 .80 .90 .75 1.00 | 2.86 | .24/.10/.10/.38/.19 | 1.19 | .55 | .051 \| .338 |
| π0.5-sp | ridge1 (non-borrowing) | .75 [.66, .82] | +5, 5 / 10, .30 | +5, 6 / 11, .33 | .50 .60 .90 .90 .85 | 1.96 | .76/.04/.04/.00/.16 | 1.08 | .46 | .063 \| .269 |
| π0.5-sp | a05_gr (α .5 + release guard) | .76 [.67, .83] | +6, 8 / 14, .29 | +6, 8 / 14, .29; vs a05 −3 (6 / 3, .51) | .55 .75 .90 .70 .90 | 2.79 | .29/.12/.12/.25/.21 | 1.67 | .44 | .085 \| .380 |
| π0.5-sp | a1_gc (symmetric gripper commit) | **.10** [.06, .17] | −60, 63 / 3, < 1e-4 | −60, 63 / 3 | .30 .05 .15 .00 .00 | 4.16 | .01/**.54**/.19/.08/.18 | 4.96 | .09 | .180 \| .702 |
| π0.5-sp | a1_tg (terminal guard, gate both) | **.23** [.16, .32] | −47, 48 / 1, < 1e-4 | −47, 48 / 1 | .00 .40 .00 .75 .00 | 2.87 | .04/.00/.03/**.53**/.40 | 1.32 | .04 | .065 \| .136 |
| π0.5-l10 | R2 CL2 (subset) | .52 [.42, .62] | – | – | t0 .35 t4 .75 t6 .55 t8 .30 t7 .65 | 6.40 | .06/.17/.06/.27/.44 | 4.31 | .08 | .151 \| .277 |
| π0.5-l10 | cl2ref (re-run) | .59 [.49, .68] | +7, 3 / 10, .09 | – | .45 .75 .70 .30 .75 | 6.76 | .05/.12/.10/.24/.49 | 4.46 | .10 | .162 \| .295 |
| π0.5-l10 | a1 | .49 [.39, .59] | −3, 17 / 14, .72 | −10, 20 / 10, .099 | .55 .40 .55 .20 .75 | 6.65 | .04/.10/.14/.26/.46 | 4.06 | .08 | .180 \| .302 |
| π0.5-l10 | a05 | .53 [.43, .62] | +1, 8 / 9, 1.0 | −6, 13 / 7, .26 | .40 .65 .65 .20 .75 | 6.17 | .06/.09/.11/.28/.47 | 3.83 | .07 | .166 \| .254 |
| π0.5-l10 | a05_gr | **.01** [.00, .05] | −51, 51 / 0 | −58, 58 / 0 | .00 .05 .00 .00 .00 | 6.64 | .00/.34/.08/.13/.44 | 5.48 | .02 | .140 \| .552 |
| GR00T-sp | R2 CL2 (subset 8,2,3,7,5) | .77 [.68, .84] | – | – | t8 .50 t2 .50 t3 .95 t7 .85 t5 .95 | 2.48 | .83/.04/.00/.04/.09 | 1.13 | .61 | .067 \| .048 |
| GR00T-sp | cl2ref (re-run) | .75 [.66, .82] | −2, 2 / 0, .50 | – | .50 .50 .95 .85 .95 | 2.44 | .84/.04/.00/.04/.08 | 1.12 | .62 | .067 \| .047 |
| GR00T-sp | tgp (terminal guard, progress-only gate) | **.60** [.50, .69] | −17, 18 / 1, .0001 | −15, 18 / 3, .0015 | .35 .45 **.30** .90 1.00 | 2.95 | .78/.03/.00/.00/.20 | 1.00 | .67 | .062 \| .028 |

Skipped on purpose (markers in `r03_pilot/state/*.SKIPPED`): sp `a1_gc_tg`, l10 `a05_gc`, `a05_gc_tg`, GR00T `gc_tg` (after the
`a1_gc` collapse); l10 `a05_tg`, GR00T `tg` (after the `a1_tg` collapse). Reading: (i) on spatial only `a05` clears the ± 5 pp
noise band with a lopsided split (3 / 12 vs the rerun) — and its 500-init confirmation is +4.8 pp, i.e. the pilot overstated it by
half; `a1`, `ridge1`, `a05_gr` are inside the band. (ii) The borrowed priors lower the gripper-split class of failed spatial
episodes (G .20 → .00 / .10) and the failed-episode flips (2.9 → 1.0–1.2) as designed, but shift the spells to terminal (a1 T .50,
ridge1 T .76) or pause rows (a05 P .38) — the trap moves, it is not removed. (iii) On l10 nothing but the collapse is readable
(a1 −10 / a05 −6 vs the rerun, both inside ± 7 pp; a1's t4 .75 → .40 is the one lopsided cell, 0 / 7). (iv) The full 500-init a05
arms restricted to the pilot subset reproduce the pilots exactly (spatial .79 vs .79, 3 / 3; l10 .50 vs .53, 8 / 5): the pilot is a
faithful sample, it is simply too small on l10 and its 5 trap tasks are not representative of where a05 gains there (t9 / t2 / t7).

### 1.3 The collapses, mechanistically (decision-log extras `gvote / gheld / gflip / gdwell / term_masked / term_open / w_term`; `MECH.md`)

**Symmetric gripper commitment (`a1_gc`, .10).** The rule serves sign(v) only when |v| ≥ .8, otherwise the previously executed
sign; a change needs |v| ≥ .8 on two consecutive decisions and a 3-decision dwell. In the loop the grasp vote is not decisive:

| arm | SR | dec/ep | held decisions (gheld) | held with \|v\| < .8 | vote on held decisions: mean / \|v\| p50 / share ≥ .8 | first served close (gflip): step mean / p50 (episodes) | episodes never closing | decisions before the first close with \|v\| in [.2, .8) |
|---|---|---|---|---|---|---|---|---|
| a1_gc | .10 | 42.1 | **.395** | .380 | **+.36 / .32 / .038** | 28.1 / 34 (41 of 100) | **59** | 13.0 |
| a1 (same metric, no commit) | .74 | 27.5 | 0 | 0 | – | first vote sign change 10.2 / 10 (100) | 0 | 0.6 |

Per task (a1_gc): t1 SR .00 with held .55 and first close at 37.4; t4 .00 with **20 / 20 episodes never closing**; t9 .05, 19 / 20
never closing; t0 .15 (17 / 20); t6 .30 (3 / 20 never; held .29). Plain AWM executes the sign of the mean, so a +.3 vote closes at
step ≈ 10; the committed rule keeps the gripper open while the robot hovers at grasp height, the vote never reaches .8 because the
kernel members straddle the grasp row, and the episode times out at 44. The off-line replay (own history, recorded observations)
could not show this (held 5.5 %, 1.0 flip / episode) because the recorded scene does not respond to the hover. Not fixable by a
threshold: the kernel vote at a grasp is a mixture by construction (ep_eff 3–4 library episodes).

**Release guard (`a05_gr`: closes freely, a release needs |v| ≥ .6 twice + dwell 3).** l10 .01 (51 / 0 vs R2): held share .083,
vote on held decisions −.22 / |v| p50 .18 / share ≥ .6 **.041**; served flips 1.01 per episode; only **9 / 100 episodes ever
release** (a05: 93 / 100, first release at step 34), dec/ep 103.4 = the cap. The l10 tasks require a release mid-task (put the
object down / in the drawer); the release vote in this library is weak (|v| .18 median), so the guard holds the object for the rest
of the episode. On spatial success is judged before any release (1 / 100 episodes with two flips in either arm), so `gr` is
neutral there (.76 vs .79, 6 / 3, p .51) — it is dead as a general switch. Together with `gc`: **every attempt to rewrite the
served gripper sign in pure cache failed because the vote lag is a property of the library's neighbourhoods, and any confirmation
rule blocks one class of gripper events** (grasp for the symmetric rule, release for the asymmetric one).

**Terminal-row guard.** `a1_tg` (gate: step ≥ .9·median length AND gripper re-opened): rows were masked on **91.5 %** of decisions
(10 rows per masked decision), the gate opened on 8.5 %; `w_term` (kernel weight on last-2 rows) .024 vs .202 for `a1`. The
trap did not disappear, it relocated: first spells of failed episodes are P .53 / H .40 (a1: T .50, P .19, H .08), the top-1 of
failed episodes' last 10 decisions sits at library step 16.4 (a1: 21.3), i.e. the cache parks the robot on a mid-episode pause /
hub row instead of the terminal hold row, with the same identical-pick share (.48 vs .50). Spatial library episodes end with the
gripper closed (last-row sign +.73), so the "re-opened" clause never fires in a normal episode — the guard is on for the whole
episode. `tgp` on GR00T-sp (progress-only gate, opens at 61 % masked / 39 % open): −15 pp vs its rerun (18 / 3, p .0015);
t3 .95 → .30, t8 .50 → .35, t2 .50 → .45; the first spells of failed episodes are still T-class (.78 vs .84) and at those spells the
gate was already open (t2 .91, t3 .86, t8 1.00 open; w_term at the spell .85–.99): masking the terminal rows early merely delays
the absorption to the moment the gate opens, and on t3 it removes the rows the successful episodes needed. **Any terminal masking
is dead for pure cache**; the terminal∧closed-gripper state is a MISS trigger (guard 2), which is where it works (§2).

### 1.4 Full 500-episode pure-cache arms (winners of the pilot), paired vs R2 CL2 of the same cell

| cell | arm (50-ep library) | SR [Wilson] | vs R2 CL2: ΔSR, S→F / F→S, p, bootstrap 95 % | vs R2 CL3 | vs pure inference (trace_dual) | per-task SR (t0…t9) | first-spell T/G/Z/P/H (fail) | flips g0 F | vote split < .8 S \| F | w_term_late F |
|---|---|---|---|---|---|---|---|---|---|---|
| π0.5-sp | R2 CL2 AWM | .800 [.76, .83] | – | – | −18.6 (97 / 4) | .80 .84 .94 .98 .74 .92 **.28** .96 .90 .64 | .38/.14/.03/.17/.28 | 2.93 | .038 \| .282 | .40 |
| π0.5-sp | **a05** (borrowed) | **.848** [.81, .88] | **+4.8**, 21 / 45, .004, [+1.6, +8.0] | +5.0, p .003 | −13.8 (74 / 5) | .94 .98 .90 .98 .72 .92 **.44** .98 .88 .74 | .51/**.03**/.09/.24/.13 | 1.08 | .039 \| .248 | .53 |
| π0.5-sp | **rm1** (ridge 1.0, non-borrowing) | **.840** [.81, .87] | **+4.0**, 17 / 37, .009, [+1.2, +6.8] | | | .90 .84 .94 .98 .78 .90 .38 .98 .92 .78 | .59/.10/.05/.07/.19 | 1.05 | .056 \| .282 | .53 |
| π0.5-sp | a05 vs rm1 | | +0.8, 24 / 28, .68, [−2.0, +3.6] | | | | | | | |
| GR00T-sp | R2 CL2 | .888 [.86, .91] | – | | −5.2 (52 / 26) | .94 .98 .76 .90 .92 .92 1.00 .90 **.56** 1.00 | .82/.04/.02/.02/.11 | 1.43 | .071 \| .083 | .60 |
| GR00T-sp | a05 | .910 [.88, .93] | +2.2, 26 / 37, .21, [−1.0, +5.4] | | −3.0 (43 / 28, p .096) | .92 .96 .90 .86 1.00 .98 .96 .84 .72 .96 | .89/.00/.00/.07/.04 | 1.02 | .079 \| .105 | .67 |
| GR00T-sp | rm1 | .894 [.86, .92] | +0.6, 24 / 27, .78, [−2.2, +3.4] | | | .98 .96 .78 .92 .94 .90 .96 .86 .68 .96 | .83/.02/.04/.06/.06 | 1.15 | .080 \| .112 | .66 |
| π0.5-l10 | R2 CL2 | .630 [.59, .67] | – | | −21.4 (147 / 40) | .32 .82 .52 .88 .62 1.00 .50 .62 .32 .70 | .07/.12/.11/.22/.47 | 4.56 | .094 \| .286 | .10 |
| π0.5-l10 | **a05** | **.674** [.63, .71] | **+4.4**, 40 / 62, .037, [+0.6, +8.4] | +3.2, p .13 | −17.0 (129 / 44) | .42 .74 .68 .92 .58 .98 .54 .74 .26 .88 | .05/.13/.11/.22/.49 | 4.25 | .107 \| .244 | .07 |
| GR00T-l10 | R2 CL2 | .552 [.51, .60] | – | | −31.8 (187 / 28) | .20 .64 .94 .86 .34 .98 .38 .42 .42 .34 | .00/.20/.16/.14/.50 | 3.10 | .196 \| .334 | .06 |
| GR00T-l10 | a05 | .544 [.50, .59] | −0.8, 58 / 54, .78, [−5.0, +3.4] | −6.2, p .013 | −32.6 (184 / 21) | .22 .84 .94 .96 .24 1.00 .36 .42 **.08** .38 | .00/.21/.20/.17/.43 | 3.96 | .210 \| **.432** | .06 |

Per-task footprint of a05 (F→S / S→F): π0.5-sp t6 +16 (11 / 3), t0 +14 (8 / 1), t1 +14 (7 / 0), t9 +10 (11 / 6); rm1 t9 +14 (8 / 1),
t6 +10, t0 +10. π0.5-l10 t9 +18 (12 / 3), t2 +16 (10 / 2), t7 +12 (9 / 3), t0 +10 (10 / 5); losses t1 −8 (1 / 5), t8 −6. GR00T-sp
t8 +16 (12 / 4), t2 +14 (9 / 2) vs t7 −6, t3 −4. GR00T-l10 **t8 −34 (1 / 18)** against t1 +20 (11 / 1), t3 +10: the borrowed
metric raises GR00T-l10's failed-episode vote split to .432 (the highest of any arm) — the l10 side effect C measured offline
(task-6 severe split doubling) shows up as a task collapse in the loop. `a05` fixes half of the π0.5-sp task-6 collapse (.28 →
.44; the 10× library gives .86) and moves GR00T-sp t8 .56 → .72 without touching its terminal-row anatomy (T .89).

### 1.5 Verdict per switch (50-episode library; "borrowed" = fit statistics from the 500 episodes, candidates unchanged)

| switch | verdict | evidence |
|---|---|---|
| borrowed prior α = .5 (`a05`, **borrowed big-library information**) | **keep on π0.5** (+4.8 / +4.4 pp, both significant); **no on GR00T** (+2.2 n.s., −0.8 n.s. with a t8 collapse) | §1.4; pilot .79 sp; offline task-6 split .174 → .149 |
| borrowed prior α = 1 (`a1`) | drop | pilot +4 sp (inside noise), −10 l10 vs rerun (t4 .75 → .40); offline doubles the l10 task-6 split |
| ridge 1.0 (`rm1`, non-borrowing) | **keep as the π0.5-sp default when no big library may be used** (+4.0 pp, p .009; = 80 % of a05) | §1.4; a05 − rm1 +0.8, p .68; GR00T-sp +0.6 n.s. |
| symmetric gripper commitment (`gc`) | **dead** | .10; hover mechanism §1.3 |
| release guard (`gr`) | **dead** | l10 .01 (91 / 100 never release); spatial −3 n.s. |
| terminal guard, gate both (`tg`) | **dead** | .23; trap relocates to P / H rows |
| terminal guard, progress-only gate (`tgp`) | **dead** | GR00T-sp .60, −15 pp, t3 .95 → .30; absorption merely delayed |
| combinations (`gc_tg`, `a05_gc*`) | skipped, dead by their components | SKIPPED markers |

What R3's pure-cache line bought at the deployed scale: **+4.8 / +4.4 pp on π0.5, 0 on GR00T**, against the +15.4 / +13.8 / +7.8 /
+15.4 pp the 500-episode library gives (R2) and the +10.6 / +9.4 / +5.6 / +16.2 pp still left between `a05` and the 500-episode AWM.

---

## 2. Mixed HIT/MISS results (H2 infrastructure + H3 judge; π0.5 only, full model on every MISS)

Conventions: h = realized HIT share of all decisions (kpi.py, decisions matched to the accepted attempt); **IR = 0.152 + 0.848·(1 − h)**
(project definition); "IR measured" = (mean s1 + (1 − h)·mean s23|MISS) / (mean s1 + mean s23|MISS) from the logged stage
times. Realized h is 3–4 pp below the .7 target on both suites (the controller holds the (1 − h)-quantile over a 1000-decision
window per server; forced MISSes enter as −∞ and consume budget), .46–.47 for the .5 target.

### 2.1 Arm tables (n = 500 inits each; paired vs pure cache CL2 on the same inits, vs the periodic control, vs pure inference)

**π0.5-l10** (pure cache B0 .440 / AWM-50 .630 / AWM-500 .768; pure inference .844, 58.8 dec/ep)

| arm | judge | servers | h (all / after-HIT / after-MISS) | IR formula / measured | SR [Wilson] | vs CL2-50 ΔSR (S→F / F→S, p) | vs periodic k3 | vs pure inference | MISS/ep S \| F | MISS share of decisions in F \| S eps | share of all MISSes in F eps (F eps' decision share) | success eps interrupted | dec/ep S |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b0h70 | B0 fused score, quantile .7 | 4 | .660 / **.941 / .096** | .440 / .441 | .632 [.589, .673] | +0.2 (86 / 87, 1.0) | −20.0 (134 / 34, < 1e-4) | −21.2 (139 / 33) | 11.4 \| 47.7 | .46 \| .21 | .71 (.52) | .62 | 54.8 |
| awm_h70 | V7 + guards, quantile .7 | 4 | .657 / .788 / .384 | .443 / .439 | **.816** [.780, .848] | **+18.6** (29 / 122, < 1e-4) | −1.6 (55 / 47, .49) | −2.8 (60 / 46, .21) | 13.4 \| 54.1 | .52 \| .26 | .48 (.31) | .97 | 51.0 |
| ev_h70 | + events + burst 2 | 4 → 3 (resumed) | .637 / .845 / .251 (collect: .660) | .460 / .459 (.440) | .822 [.786, .853] | +19.2 (19 / 115) | −1.0 (52 / 47, .69) | −2.2 (60 / 49, .34) | 15.5 \| 54.5 | .52 \| .29 | .43 (.30) | 1.00 | 52.6 |
| perk3 | AWM selector, MISS every 3rd | 3 | .673 / .505 / 1.0 | .429 / .432 | **.832** [.797, .862] | +20.2 (32 / 133) | – | −1.2 (56 / 50, .63) | 16.2 \| 34.0 | .33 \| .33 | .30 (.30) | 1.00 | 49.7 |
| g | guard-only (forced MISS only) | 3 | .798 / .904 / .354 | **.323** / .329 | .740 [.700, .777] | +11.0 (25 / 80) | −9.2 (83 / 37, < 1e-4) | −10.4 (96 / 44) | 5.8 \| 34.9 | .34 \| .11 | .68 (.41) | .91 | 52.6 |
| awm_h50 | V7 + guards, quantile .5 | 3 | .468 / .604 / .330 | .603 / .600 | **.868** [.836, .895] | +23.8 (24 / 143) | +3.6 (vs awm_h70 +5.2, 31 / 57, .007) | +2.4 (40 / 52, .25) | 25.2 \| 71.1 | .68 \| .49 | .30 (.23) | 1.00 | 51.7 |
| awm500_h70 | V7 + guards, .7, **500-ep AWM** | 3 | .659 / .830 / .301 | .441 / .439 | **.872** [.840, .898] | +24.2 (27 / 148); **vs CL2-500 +10.4** (18 / 70, < 1e-4) | vs awm_h70 +5.6 (33 / 61, .005) | **+2.8** (35 / 49, .16) | 15.2 \| 51.5 | .49 \| .30 | .33 (.23) | .98 | 51.4 |
| b0perk3 | **B0 selector**, MISS every 3rd | 2 | .673 / .506 / 1.0 | .429 / .411 | .714 [.673, .752] | +8.4 (60 / 102, .001) | **−11.8** (96 / 37, < 1e-4); vs b0h70 +8.2 (54 / 95, .001) | −13.0 (110 / 45) | 16.5 \| 34.0 | .33 \| .33 | .45 (.45) | 1.00 | 50.3 |
| perk5 | AWM selector, MISS every 5th | 2 | .808 / .758 / 1.0 | **.315** / .325 | **.792** [.754, .825] | +16.2 (31 / 112) | vs g **+5.2** (45 / 71, .020) at IR .32 | −5.2 (70 / 44, .019) | 9.8 \| 20.0 | .19 \| .19 | .35 (.35) | 1.00 | 51.1 |
| g500 | guard-only, **500-ep AWM** | 2 | .898 / .934 / .548 | **.238** / .289 | **.864** [.831, .891] | +23.4 (28 / 145); **vs CL2-500 +9.6** (12 / 60, < 1e-4) | vs g +12.4 (31 / 93); vs awm500_h70 −0.8 (31 / 27, .69) | **+2.0** (44 / 54, .36) | 2.8 \| 25.8 | .25 \| .06 | .59 (.24) | .81 | 51.5 |

**π0.5-spatial** (pure cache B0 .668 / AWM-50 .800 / AWM-500 .954; pure inference .986, 21.6 dec/ep)

| arm | judge | servers | h (all / after-HIT / after-MISS) | IR formula / measured | SR [Wilson] | vs CL2-50 | vs periodic k3 | vs pure inference | MISS/ep S \| F | MISS share in F \| S eps | share of MISSes in F eps (dec share) | success eps interrupted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b0h70 | B0 fused score, .7 | 3 | .633 / **.927 / .074** | .463 / .467 | .894 [.864, .918] | +9.4 (36 / 83); vs CL0 +22.6 | −4.4 (50 / 28, .017) | −9.2 (52 / 6) | 6.9 \| 30.0 | .68 \| .30 | .34 (.18) | .77 |
| awm_h70 | V7 + guards, .7 | 3 | .672 / .737 / .487 | .430 / .423 | **.980** [.964, .989] | **+18.0** (2 / 92) | **+4.2** (7 / 28, .0005) | −0.6 (9 / 6, .61) | 6.8 \| 33.1 | .75 \| .31 | .09 (.04) | 1.00 |
| ev_h70 | + events + burst | 3 | .672 / .800 / .341 | .430 / .424 | .956 [.934, .971] | +15.6 (6 / 84) | +1.8 (20 / 29, .25); vs awm_h70 **−2.4** (18 / 6, .023) | −3.0 (22 / 7, .008) | 6.3 \| 31.6 | .72 \| .29 | .19 (.09) | .99 |
| perk3 | periodic k = 3 | 3 | .683 / .513 / 1.0 | .421 / .418 | .938 [.913, .956] | +13.8 (23 / 92) | – | −4.8 (30 / 6, .0001) | 6.6 \| 14.0 | .32 \| .32 | .12 (.12) | 1.00 |
| g | guard-only | 3 | .865 / .918 / .376 | **.266** / .265 | .888 [.857, .913] | +8.8 (6 / 50) | −5.0 (48 / 23, .004) | −9.8 (54 / 5) | 1.6 \| 15.8 | .36 \| .08 | .55 (.21) | .83 |
| awm_h50 | V7 + guards, .5 | 3 → 2 (resumed) | .459 / .461 / .423 | .611 / .587 | **.986** [.971, .993] | +18.6 (4 / 97) | | **0.0** (7 / 7, 1.0) | 11.4 \| 36.6 | .83 \| .53 | .04 (.03) | 1.00 |
| awm500_h70 | V7 + guards, .7, 500-ep AWM | 3 | .663 / .728 / .468 | .438 / .431 | .978 [.961, .988] | +17.8; **vs CL2-500 +2.4** (8 / 20, .036) | vs awm_h70 −0.2 (9 / 8, 1.0) | −0.8 (11 / 7, .48) | 6.7 \| 30.9 | .70 \| .32 | .09 (.05) | 1.00 |

Per-task (mixed vs CL2-50): the π0.5-sp task-6 collapse is gone in every AWM-judge arm (.28 → 1.00 / .98 / .96 / .70 (g) / .98
(awm500); 36 F→S / 0 S→F for awm_h70) and in b0h70 (.92) — the policy rescues the split-grasp trap the borrowed prior only half
fixed; the remaining spatial failures are t4 (.86 awm_h70; the policy itself is .98 there, 1 / 6 in awm_h70's favour). On l10 the
per-task minimum stays t8 (.36 awm_h70 / .56 awm_h50 / .50 awm500 / .60 g500 / .56 perk5; policy .36–.40 in the same inits: t8 is a
policy-hard task — g500 and perk5 win it from the policy 18 / 6 and 16 / 6). The B0-selector periodic arm loses to the AWM-selector
periodic arm on t0 / t4 / t7 / t8 / t9 (−30 / −30 / −28 / −22 / −18) and wins t2 (+14): the served HIT actions decide those tasks.

### 2.2 SR-vs-IR frontier (π0.5; endpoints = pure cache h = 1 at IR .152 and pure inference h = 0 at IR 1)

`FRONTIER.md`: "time-share line" = the SR a naive mixture would give by running the policy on a fraction of *episodes* (linear
between the arm's own pure-cache base and pure inference at the same IR); "marginal" = ΔSR over that base per 0.1 IR.

| suite | point | library | h | IR | SR | time-share line at this IR | **above the line (pp)** | marginal pp / 0.1 IR (pure inference: 2.5 l10 / 2.2 sp) |
|---|---|---|---|---|---|---|---|---|
| l10 | pure cache AWM CL2 | 50 | 1 | .152 | .630 | – | – | – |
| l10 | pure cache AWM CL2 | 500 | 1 | .152 | .768 | – | – | – |
| l10 | **guard-only `g500`, 500-ep base** | **500** | .898 | **.238** (.289 measured) | **.864** | .776 | +8.8 | **11.1** |
| l10 | periodic k5 | 50 | .808 | .315 | .792 | .671 | +12.1 | 9.9 |
| l10 | guard-only `g` | 50 | .798 | .323 | .740 | .673 | +6.7 | 6.4 |
| l10 | periodic k3 | 50 | .673 | .429 | .832 | .700 | +13.2 | 7.3 |
| l10 | B0 selector, periodic k3 (base CL0 .440) | 50 | .673 | .429 | .714 | .572 | +14.2 | 9.9 |
| l10 | B0 judge h.7 (base CL0 .440) | 50 | .660 | .440 | .632 | .577 | +5.5 | 6.7 |
| l10 | V7 + guards h.7 | 50 | .657 | .443 | .816 | .704 | +11.2 | 6.4 |
| l10 | **V7 + guards h.7, 500-ep base** | **500** | .659 | .441 | **.872** | .794 | +7.8 | 3.6 |
| l10 | + events / burst h.7 | 50 | .637 | .460 | .822 | .708 | +11.4 | 6.2 |
| l10 | V7 + guards h.5 | 50 | .468 | .603 | .868 | .744 | +12.4 | 5.3 |
| l10 | pure inference | – | 0 | 1 | .844 | – | – | – |
| sp | pure cache AWM CL2 | 50 / 500 | 1 | .152 | .800 / **.954** | – | – | – |
| sp | guard-only `g` | 50 | .865 | .266 | .888 | .825 | +6.3 | 7.7 |
| sp | periodic k3 | 50 | .683 | .421 | .938 | .859 | +7.9 | 5.1 |
| sp | **V7 + guards h.7** | 50 | .672 | .430 | **.980** | .861 | **+11.9** | 6.5 |
| sp | + events / burst h.7 | 50 | .672 | .430 | .956 | .861 | +9.5 | 5.6 |
| sp | V7 + guards h.7, 500-ep base | 500 | .663 | .438 | .978 | .965 | +1.3 | 0.8 |
| sp | B0 judge h.7 (base CL0 .668) | 50 | .633 | .463 | .894 | .785 | +10.9 | 7.3 |
| sp | V7 + guards h.5 | 50 | .459 | .611 | **.986** | .901 | +8.5 | 4.1 |
| sp | pure inference | – | 0 | 1 | .986 | – | – | – |

Frontier reading. **l10**: the 50-episode frontier is perk5 (.32, .792) → perk3 (.43, .832) → awm_h50 (.60, .868), with the targeted
arms (g .740 at .32, awm_h70 .816 at .44) below the periodic ones at both budgets; the 500-episode frontier is CL2-500 (.152, .768)
→ **g500 (.24, .864)** → awm500_h70 (.44, .872): g500 dominates every 50-episode point (it beats awm_h50 at 40 % of its IR) and is
statistically at the policy (+2.0, p .36), so on l10 the frontier's knee is at IR ≈ .24 on the 500-episode library, and the 16 pp of
inference ratio between g500 and awm500_h70 buy +0.8 pp (n.s.). **Spatial**: CL2-500 pure cache (.152, .954) beats
every 50-episode mixed point below IR .43 (g .888 at .27, perk3 .938 at .42) and is 2.6 pp below awm_h70 (.43, .980); the policy is
matched at IR .43 (h .7, −0.6 pp n.s.) and reached exactly at IR .61 (h .5). Every AWM-based point is 6–13 pp above the time-share
line: the policy is worth 5–7 pp per 0.1 IR inside the cache loop versus 2.2–2.5 pp per 0.1 IR when it replaces the cache — the
gain is an interaction (drift interrupted before the trap), not proportional inference. The one exception is the 500-episode
spatial point (+1.3 pp above the line; +2.4 pp, p .036 over CL2-500): at .954 there is little left for the policy to rescue.

### 2.3 Judge comparison at matched realized h ≈ .63–.68 (same selector unless noted)

| comparison (arm − reference) | π0.5-l10 ΔSR (S→F / F→S, p) | π0.5-sp ΔSR (S→F / F→S, p) | reading |
|---|---|---|---|
| V7 + guards − periodic k3 (AWM selector both) | −1.6 (55 / 47, .49) | **+4.2** (7 / 28, .0005) | targeting is worth 4 pp on spatial, nothing on l10 |
| + events + burst − V7 + guards | +0.6 (37 / 40, .82) | **−2.4** (18 / 6, .023) | events / burst: dead (skipped at h .5) |
| guard-only − V7 + guards (IR .32 / .27 vs .44 / .43) | −7.6 (68 / 30, .0002) | −9.2 (48 / 2, < 1e-4) | the quantile MISSes on top of the guards buy 8–9 pp for +.12–.16 IR |
| guard-only − periodic k3 | −9.2 (83 / 37) | −5.0 (48 / 23, .004) | at 2/3 of the periodic arm's IR |
| B0 judge + B0 selector − periodic k3 (AWM) | −20.0 (134 / 34) | −4.4 (50 / 28, .017) | split below on l10 |
| **selector**: B0 periodic k3 − AWM periodic k3 (same judge) | **−11.8** (96 / 37, < 1e-4) | not run | the served HIT actions carry 2/3 of the l10 gap |
| **judge**: B0 fused score h .7 − B0 periodic k3 (same selector) | **−8.2** (54 / 95, .001) | not run | the B0 score as a judge costs the other third |
| V7 + guards (AWM) − B0 judge (B0) | +18.4 (30 / 122) | +8.6 (7 / 50) | ≈ selector 11.8 + judge 8.2 − 1.6 |
| periodic k5 − guard-only (IR .315 vs .323) | **+5.2** (45 / 71, .020) | not run | periodic dominates targeting on l10 at the low budget too |
| guard-only 500-ep − V7 + guards h .7 500-ep (IR .24 vs .44) | −0.8 (31 / 27, .69) | not run | the quantile MISSes buy nothing on the 500 base |
| guard-only 500-ep − guard-only 50-ep (IR .24 vs .32) | **+12.4** (31 / 93, < 1e-4) | not run | library layer inside the guard-only loop |
| h .5 − h .7 (V7 + guards) | +5.2 (31 / 57, .007) for +.16 IR | +0.6 (7 / 10, .63) for +.18 IR | l10 still inference-limited at h .7; spatial saturated |
| 500-ep base − 50-ep base (V7 + guards h .7, equal IR) | **+5.6** (33 / 61, .005) | −0.2 (9 / 8, 1.0) | the library layer survives inside the mixed loop on l10 |
| h .5 (50 ep) − h .7 (500 ep) | −0.4 (40 / 38, .91) at +.16 IR | – | 450 extra library episodes ≈ 16 pp of inference ratio |

Is targeting worth anything over periodic MISS? On **spatial, yes**: +4.2 pp (p .0005) at equal IR, and the periodic arm is
significantly below the policy (−4.8, p .0001) while the targeted one is not (−0.6, p .61); the difference is where the MISSes
land — awm_h70 puts 75 % of the decisions of failed episodes on the policy versus 31 % in successful ones (perk3: 32 % / 32 % by
construction), and only 9 % of its MISSes fall in failed episodes because it has almost none. On **l10, no, at either budget**: perk3 .832 ≥ awm_h70
.816 ≥ ev .822 within noise, all three ≈ the policy (−1.2 … −2.8, n.s.), and at IR ≈ .32 periodic k5 .792 beats guard-only .740
(p .020). The l10 mechanism is drift interruption: any MISS every 2–3
decisions keeps the cache's states on the policy's manifold; the targeted judge spends its budget the same way (mean HIT run at a
MISS 1.8, 48 % of its MISSes inside MISS runs ≥ 5, after-MISS h .38), it just concentrates it in the failing episodes (52 % vs 26 %),
which on l10 does not convert into SR because those episodes then become policy failures (§2.4). The B0 judge is worse than
periodic on both suites and worse than *no judge* on l10 (.632 vs AWM pure cache .630): with a fused-score threshold the policy is
handed the episode in blocks (after-HIT h .94, after-MISS h .10, mean MISS run 8.8, MISS runs ≥ 5 hold 88 % of the MISSes) and
half its failed episodes end in ≥ 20 consecutive MISSes — the policy takes over late (first MISS at step 35 of 104, 34 % into the
episode; targeted judges: 12–20) and from a state it did not produce. `b0perk3` (B0 selector, periodic judge, .714) separates the
two: the B0 selector's HIT actions alone cost 11.8 pp against AWM's under the identical periodic schedule, and the B0 score as a
judge costs another 8.2 pp against the blind schedule. Neither half of B0 has a place in the mixed line. On the 500-episode base the
guard-only arm equals the quantile arm (.864 vs .872, p .69) at 55 % of its inference: with fresher neighbourhoods the after-MISS
re-acceptance is fast (MISS-after-MISS .45 vs .65 on the 50-ep guard arm, mean MISS run 1.75) and the stuck / no-progress guards are
sufficient — the l10 lever at 500 episodes is the guard set plus the library, not the confidence.

### 2.4 MISS timing, reason codes, interruptions, run structure (`RUNS_MX.md`, kpi mixed block)

| arm | forced share of MISSes | reason codes among forced (1 stuck / 2 terminal∧closed / 3 overtime / 4 no-progress / 5 disp / 6 grip / 7 burst) | first MISS step in failed eps (mean; position) | first MISS − first *proposal* spell (median; early share) | HIT run at MISS mean / max | mean MISS run | MISS-after-MISS | MISSes in runs ≥ 5 | failed eps with policy majority | failed eps ending in ≥ 20 MISSes |
|---|---|---|---|---|---|---|---|---|---|---|
| l10 b0h70 | 0 | – | 35.0 (.34) | +5; .42 | 1.3 / 103 | 8.8 | .90 | .88 | .47 | 96 / 184 |
| l10 awm_h70 | .49 | .42 / .08 / .07 / .43 | 19.4 (.19) | −8; .64 | 1.8 / 69 | 2.5 | .62 | .48 | .58 | 28 / 92 |
| l10 ev_h70 | .75 | .24 / .06 / .05 / .30 / .03 / .05 / .28 | 11.5 (.11) | −20; .84 | 1.8 / 57 | 3.7 | .75 | .56 | .27 | 42 / 89 |
| l10 perk3 | 0 | – | 2.0 | – | 2.0 / 2 | 1.0 | 0 | 0 | 0 | 0 / 84 |
| l10 g | 1.0 | .33 / .08 / .06 / .52 | 20.4 (.20) | −12; .65 | 3.4 / 58 | 2.7 | .65 | .48 | .07 | 13 / 130 |
| l10 g500 | 1.0 | .23 / .08 / .05 / .64 | 21.6 (.21) | −14; .82 | 6.4 / 60 | 1.75 | .45 | .23 | .03 | 2 / 68 |
| l10 perk5 | 0 | – | 4.0 | – | 4.0 / 4 | 1.0 | 0 | 0 | 0 | 0 / 104 |
| l10 b0perk3 | 0 | – | 2.0 | – | 2.0 / 2 | 1.0 | 0 | 0 | 0 | 0 / 143 |
| l10 awm_h50 | .29 | .36 / .09 / .06 / .49 | 12.8 (.12) | −24; .84 | 0.9 / 35 | 2.9 | .67 | .57 | .80 | 35 / 66 |
| l10 awm500_h70 | .29 | .20 / .09 / .04 / .66 | 20.3 (.20) | −14; .67 | 1.9 / 71 | 3.1 | .70 | .56 | .50 | 21 / 64 |
| sp b0h70 | 0 | – | 10.2 (.23) | −6; .85 | 1.0 / 27 | 7.9 | .93 | .87 | .91 | 41 / 53 |
| sp awm_h70 | .23 | .08 / **.57** / .02 / .33 | 7.9 (.18) | −17; 1.0 (n = 6) | 2.0 / 24 | 1.8 | .51 | .26 | 1.00 | 7 / 10 |
| sp ev_h70 | .47 | .06 / .26 / .03 / .22 / .08 / .06 / .29 | 8.5 (.19) | −14; 1.0 | 2.0 / 26 | 2.4 | .66 | .42 | 1.00 | 18 / 22 |
| sp g | 1.0 | .11 / .36 / .07 / .47 | 16.1 (.36) | −4; .80 | 5.3 / 28 | 1.9 | .62 | .32 | 0 | 0 / 56 |
| sp awm_h50 | .15 | .08 / .53 / .03 / .36 | 3.3 | −16; 1.0 | 0.8 / 12 | 2.1 | .58 | .35 | .82 | 6 / 11 |

Facts. (i) **Guards, not the confidence, are the early component**: in the quantile arms 23–49 % of MISSes are forced (l10 .49: stuck
.42 + no-progress .43; spatial .23: terminal∧closed .57 + no-progress .33), and the first MISS of a failed episode comes a median 8–24
decisions *before* the first would-be spell (64–84 % "early"), at 11–20 % of the episode. The B0 judge fires after the spell (+5,
42 % early). (ii) **Events + burst change the timing (first MISS 11.5 vs 19.4, 84 % early) without changing SR**: the events
themselves are 8 % of the forced MISSes, the burst continuation 28 % — the extra budget goes into longer MISS runs (3.7 vs 2.5). (iii)
**Interrupted successes**: every quantile arm at h ≤ .7 interrupts 97–100 % of its successful episodes (13–25 MISSes each on l10, 6–11
on spatial); guard-only 83 % / 91 % with 1.6 / 5.8 MISSes; B0 62 % / 77 %. "Leave the good episodes alone" is not achieved by any
judge at these budgets; the guard-only arm is the only one whose MISSes are concentrated (68 % / 55 % of its MISSes in failed
episodes, which hold 41 % / 21 % of decisions). (iv) **MISS runs are where the budget goes**: after a MISS the judges MISS again 51–75 %
of the time (the policy's fresh states score below τ and the no-progress memo keeps firing), 26–57 % of all MISSes sit in runs ≥ 5.
The periodic arms have no runs at all and match (k3) or beat (k5) the targeted arms on l10 at equal or lower IR; on the 500-ep
base the guard-only arm's MISS-after-MISS drops to .45 (mean MISS run 1.75, only 23 % of MISSes in runs ≥ 5) and only 2 of its 68
failures end in ≥ 20 straight MISSes — the fresher library returns control to the cache quickly. (v) **Spells are gone, failures are not**:
no identical-pick spell survives in any AWM-judge arm (max served run 1.7–2.0; b0h70 still has 1.8 spells per failed episode), yet
all failures are cap timeouts; in 50–80 % of the failed episodes the policy held the majority of decisions, and 7–35 of them end with
≥ 20 straight MISSes — the policy could not finish from the state the cache left it in (or fails on its own: on l10 32 of awm_h70's 92
failed inits and 35 of awm500_h70's 64 are inits the pure policy also fails; on spatial 6 / 10 and 7 / 11). The remaining lever is
therefore *earlier* handover in the failing episodes or a better return decision, not more MISSes.

### 2.5 Latency and infrastructure

| arm group | servers on the 4090 | s1 mean (ms) | s23 mean on MISS (ms) | server infer ms p50 | IR formula vs measured |
|---|---|---|---|---|---|
| l10 b0h70, awm_h70 | 4 (+ other project's training from 06:10) | 366 / 352 | 2010 / 2061 | 484 (awm) | .440 / .441; .443 / .439 |
| l10 ev_h70 (resumed), g, perk3, awm_h50, awm500 | 3 | 456 / 439 / 421 / 534 / 418 | 2574 / 2319 / 2274 / 3158 / 2377 | | within .004 |
| sp b0h70 … awm500 | 3 | 287–465 | 1616–2836 | | within .007 |
| sp awm_h50 (resumed) | 3 → 2 (32 workers each) | 433 | 3867 | | .611 / .587 |
| l10 b0perk3, perk5, g500 (queue 2d) | 2 (32 workers each) | 547 / 575 / 594 | 3822 / 2912 / 2258 | | .429 / .411; .315 / .325; **.238 / .289** |

The single-server CUDA-graph reference is s1 117 ms / s23 483 ms (H2 smoke); under 3–4 full-model servers plus a foreign training
job both stages are 3–8× slower and s23 scales with the MISS load (awm_h50 3.2–3.9 s). The IR formula is the project definition
and is what the frontier uses; the measured stage ratio agrees within .02 because the loaded s1 : s23 ratio (.15–.17) is close to the
formula's .152 / .848 split. The judge itself costs nothing measurable (`q_us` p50 8.8 ms for MixedJudge vs 4.5 ms for AWM on a loaded
server, both dominated by the native shadow search). **The g500 gap (IR .238 formula vs .289 measured) is latency composition,
nothing else**: the measured ratio is s1/(s1+s23) + (1 − s1/(s1+s23))·(1 − h) with the *loaded* stage means, and with 2 servers ×
32 workers the stage-1 share is 594 / (594 + 2258) = .208 instead of the CUDA-graph reference .152, so measured − formula =
(.208 − .152)·h = .056 × .898 = .050 exactly. Both IRs are linear in the MISS share, so episode composition (MISS-heavy failed
episodes, step-0 decisions — step 0 is always a HIT in the guard arms) cannot open a gap; the gap grows with h because a
high-h arm's cost is dominated by stage 1, which the shared GPU inflates most (g500 has the highest s1 of the round). The formula
is the project definition and is what the frontier uses; on a single CUDA-graph server g500's cost is IR .238. Bookkeeping notes: `r3mx_p_l10_ev_h70` was aborted at 06:10 and resumed
(7 server startups; kpi's server-side h .637 vs the client mix .653 and collect's .660 — the resumed episodes share attempt = 1 with
their aborted first run, so ± .02 on h / ± .015 on IR is the uncertainty of that arm); `r3mx_p_sp_awm_h50` was restarted after 3
episodes with 2 servers (5 startups; the client and server mixes agree).

### 2.6 Bookkeeping checks

Journal ↔ server success 500 / 500 in every arm; client FULL_HIT / MISS counts == server hit / !hit (`verdict_mix_matches_server`
true); `exec_ok` 1.0 on every HIT row; the executed policy chunk is logged on 100 % of MISS rows; realized-h per regime shows the
controller working (step 0 always judged: h_step0 = 1.0 in the quantile arms).

### 2.7 What the last three l10 arms settled (2 servers, queue 2d)

| arm | question | answer |
|---|---|---|
| `r3mx_p_l10_b0perk3` .714 | selector vs judge in the 18.4 pp awm_h70 − b0h70 gap | both: B0 selector −11.8 pp under the same periodic schedule (vs perk3 .832), B0 score judge −8.2 pp under the same selector (b0h70 .632); ≈ additive (11.8 + 8.2 − 1.6 = 18.4) |
| `r3mx_p_l10_perk5` .792 at IR .315 | does targeting start to pay at the guard-only budget? | no: periodic k5 beats guard-only (.740 at IR .323) by +5.2 (45 / 71, p .020); periodic dominates targeting on l10 at IR .32 and .43 |
| `r3mx_p_l10_g500` .864 at IR .238 | the cheap end of the 500-episode frontier | at the policy (+2.0, p .36), equal to awm500_h70 (−0.8, p .69) at 55 % of its inference, +9.6 over CL2-500 (p < 1e-4), +12.4 over the 50-ep guard arm; the best l10 point of the round |

---

## 3. Four-layer decomposition (closed loop, paired on the same inits; policy inference counted as cost)

| layer | π0.5-sp | π0.5-l10 | GR00T-sp | GR00T-l10 | source / label |
|---|---|---|---|---|---|
| **synthesis** (B0 top-1 → mean-5, 50 ep) | +9.6 [+4.8, +14.4] | −1.2 [−6.0, +3.8] | +11.6 [+7.0, +16.0] | −0.2 [−5.2, +4.8] | R2 CL0 → CL1 |
| **method at fixed library**: AWM ranking (50 ep) | +3.6 [−0.6, +8.0] | +20.2 [+15.4, +25.2] | +3.6 [+0.0, +7.2] | +8.6 [+3.8, +13.4] | R2 CL1 → CL2 |
| method at fixed library: R3 metric switch, non-borrowing (ridge 1.0) | **+4.0** [+1.2, +6.8] | not run | +0.6 [−2.2, +3.4] | not run | CL2 → rm1 |
| method at fixed library: R3 metric switch, **borrowed big-library information** (α = .5) | **+4.8** [+1.6, +8.0] | **+4.4** [+0.6, +8.4] | +2.2 [−1.0, +5.4] | −0.8 [−5.0, +3.4] | CL2 → a05 |
| **library** (50 → 500 episodes at AWM) | +15.4 [+11.8, +19.2] | +13.8 [+8.8, +18.8] | +7.8 [+4.8, +11.0] | +15.4 [+10.4, +20.2] | R2 CL2-50 → CL2-500 |
| library measured from the R3 arm (a05-50 → AWM-500) | +10.6 [+7.4, +14.0] | +9.4 [+4.6, +14.2] | +5.6 [+2.8, +8.4] | +16.2 [+11.6, +20.8] | a05 → CL2-500 |
| **control (handoff)** at the 50-ep library, V7 + guards h .7: ΔSR for ΔIR = +.28 | **+18.0** [+14.6, +21.6] (6.5 pp / 0.1 IR) | **+18.6** [+14.0, +23.2] (6.4) | not run | not run | CL2-50 → awm_h70 |
| control at the 50-ep library, periodic k3 (no judge): ΔIR +.27 | +13.8 (5.1) | +20.2 (7.3) | | | CL2-50 → perk3 |
| control at the 50-ep library, guard-only: ΔIR +.11 / +.17 | +8.8 (7.7) | +11.0 (6.4) | | | CL2-50 → g |
| control at the 500-ep library, V7 + guards h .7: ΔIR +.29 | +2.4 [+0.4, +4.4] (0.8) | **+10.4** [+6.8, +14.0] (3.6) | | | CL2-500 → awm500_h70 |
| control at the 500-ep library, guard-only: ΔIR +.086 | not run | **+9.6** [+6.4, +12.8] (**11.1**) | | | CL2-500 → g500 |
| control at the 50-ep library, periodic k5: ΔIR +.16 | not run | +16.2 [+11.8, +20.6] (9.9) | | | CL2-50 → perk5 |
| selector inside the mixed loop (AWM − B0 served HIT actions, periodic k3 judge) | not run | **+11.8** [+7.4, +16.2] | | | b0perk3 → perk3 |
| control at the 50-ep library, h .5: ΔIR +.45 / +.46 | +18.6 (4.1) | +23.8 (5.3) | | | CL2-50 → awm_h50 |
| replacing the cache by the policy (reference cost): ΔIR +.85 | +18.6 (2.2) | +21.4 (2.5) | +5.2 (0.6) | +31.8 (3.7) | CL2-50 → pure inference |
| **total, deployed scale → best point** | .668 → .980 (awm_h70, IR .43) or .954 (CL2-500, IR .152) | .440 → .864 (g500, IR .24) / .872 (awm500_h70, IR .44) or .768 (CL2-500, IR .152) | .736 → .966 (CL2-500) | .468 → .706 / .736 (CL2-500 / CL3-500) | |

Reading. The control layer is the largest single layer at the deployed 50-episode scale on both π0.5 suites (+18 pp), but it is
paid for in inference (IR .152 → .43); normalised by cost it is 2.5–3× more efficient than replacing the cache. At the 500-episode
library the control layer shrinks to +2.4 pp on spatial (saturated at .954) and stays +9.6–10.4 pp on l10, where it also lifts the
cache to the policy — and the guard-only version gets it for ΔIR .086 (11 pp per 0.1 IR, the best cost efficiency of the round). The library layer remains the largest *free* layer (IR unchanged) and, measured from the R3 metric, is still
2–3× the R3 switch on π0.5 and 3–20× on GR00T. Every "borrowed" row is labelled; the awm500 arms are library-layer arms (500-episode
candidates and fit), not borrowed ones.

---

## 4. Library scale for every configuration run in R3

| configuration (arms) | library | episodes / entries (π0.5-sp / π0.5-l10 / GR00T-sp / GR00T-l10) | B / entry | fit pickle shipped to the servers (MB) | vs deployed pkl 431 / 1,103 / 429 / 1,068 MB | note |
|---|---|---|---|---|---|---|
| AWM CL2 (R2; `cl2ref`, `perk3` base) | current | 49 / 50 / 50 / 50 — 1,018 / 2,640 / 1,063 / 2,645 | 580 (+ 280 action) | 20.4 / 23.5 / 21.2 / 25.4 (r02_g50 fits; `r3p_*_cl2ref` 20.9 / 24.8 / 21.7) | 4.8 % / 2.1 % / 4.9 % / 2.4 % | 17.0 MB PCA bases dominate |
| AWM3 `a05`, `a1`, `rm1`, `gr`, `gc`, `tg` (pilot + full) | current candidates; `a*` = **500-episode fit statistics (borrowed)** | same entries | 580 (581 with `tg`) | **20.9 / 24.8 / 21.8 / 26.8** (identical to ± 0.1 MB across all switches) | 4.9 % / 2.2 % / 5.1 % / 2.5 % | borrowing changes no bytes online |
| MixedJudge on AWM-50 (`awm_h70/h50`, `ev_*`, `g`) | current | same | 626 (AWM 580 + V7 46) | **24.8 (sp) / 31.1 (l10)** | 5.8 % / 2.8 % | + 2.6 MB task-mean keys; two action tables |
| MixedJudge on AWM-500 (`awm500_h70`, `g500`) | 10× | 500 — 10,909 / 29,472 | 626 | **63.0 / 134.7** | 14.6 % / 12.2 % | the full (H, 32) f32 action table twice (a [5, 7] payload would halve it) |
| AWM-500 pure cache (R2 CL2-500, the l10 base to beat) | 10× | same | 580 | 44.8 / 89.8 | 10.4 % / 8.1 % | |
| B0Current judge (`b0h70`, `b0perk3`) | current | same | 262,272 (raw keys) | **254.6 / 660.4** | 59 % / 60 % | the only mixed arm at deployed-format size |

Fixed per suite: 17.0 MB PCA-64 bases (2 cameras) + ≈ 0.15 MB per task (W, W0, A) for AWM3; the judge adds < 10 KB (calibration
maps, thresholds). Fit walls: AWM3 current 0.6–4.3 s (borrowed fits need ≈ 1.3 GB RAM transiently), MixedJudge 3–9 s current /
18.7 s logged on the l10 server (V7 LOEO calibration), 10× 31–63 s. Query cost on the loaded server: AWM3 3.0–4.7 ms p50, MixedJudge
8.8 ms (l10), B0Current 4.9–10.6 ms (`method_query_us_p50` in `TABLES.md`).

---

## 5. What R4 should do (ranked), and what is settled / dead

**Settled.**
- Pure cache at the deployed scale: AWM + borrowed prior α = .5 is the π0.5 selector (+4.8 / +4.4 pp); ridge 1.0 is the non-borrowing
  π0.5-sp fallback (+4.0); GR00T keeps plain AWM. No pure-cache switch touches the l10 gap (−17 / −33 pp to the policy).
- The mixed cache with V7 + guards is *the* deployable system: at h ≈ .7 (IR .43) it equals the policy on spatial (.980 / .978) and
  is within 3 pp on l10 with 50 episodes (.816) / above it with 500 (.872); at h ≈ .5 it equals the policy on spatial (.986) and
  beats it on l10 (.868, n.s.). Guard-only is the low-budget point (50 ep: IR .27 / .32 → .888 / .740; **500 ep l10: IR .24 → .864 = the
  policy, = the quantile arm at 55 % of its inference**).
- On l10 the judge's *timing* is worth nothing at 50 episodes: periodic k3 ≥ V7 + guards at IR .43 (−1.6 n.s.) and periodic k5 >
  guard-only at IR .32 (+5.2, p .02); on spatial targeting is worth +4.2 pp. The selector is worth 11.8 pp inside the mixed loop
  (AWM vs B0 under the same schedule).
- The library layer is decisive everywhere and survives inside the mixed loop on l10 (+5.6 pp at equal IR). The owner's ruling
  (bytes vs trajectories) is still the single most valuable decision: at 63–135 MB the 500-episode MixedJudge pickle is 12–15 % of
  the deployed pkl.

**Dead** (do not spend arms on): symmetric gripper commitment, release guard, terminal-row masking (both gates), early events +
burst, B0 as a judge (−8.2 vs periodic) and B0 as the selector of a mixed cache (−11.8 vs AWM), library-side recovery (R2), α = 1
borrowing on l10, GR00T borrowing, quantile-threshold MISSes on top of the guards at the 500-episode l10 base (+0.8 n.s. for +.20 IR).

**Ranked R4 list.**
1. **Close the l10 frontier on the 500-episode library between IR .152 and .24**, where the knee now is: g500 (.864 at .24) already
   equals the policy, so the open question is how much of its 8.6 pp of inference is needed. Arms (all on AWM-500, paired vs CL2-500,
   g500 and the policy): periodic k = 8 and k = 12 (IR ≈ .26 / .22 — the periodic control g500 never had; if k = 8 ≥ g500 − 2 pp the
   guard timing is also worth nothing at 500 episodes), guards with `noprog_n = 4` (halves the no-progress volume, the .64 driver of
   g500's MISSes), and guards + HIT-run cap 8 without a threshold. Spatial needs nothing above IR .2 (CL2-500 .954; g500-sp is a
   one-arm sanity check). Kill for any new point: not above the g500 ↔ CL2-500 line.
2. **Fix the return decision on the 50-episode base, if the deployed scale stays 50 episodes.** 51–75 % of after-MISS decisions are
   MISSes again there (g500: .45) and 26–57 % of the budget sits in MISS runs ≥ 5, while the periodic arms (MISS runs of exactly 1)
   match or beat the targeted arms at every l10 budget. Arms: (a) guards with a hard MISS-run cap of 1 (re-judge after one policy
   chunk), (b) no-progress memo reset on MISS, (c) guards + HIT-run cap 6–8 without the quantile threshold. Kill: no arm beats
   perk5 (.792 at IR .32) or perk3 (.832 at .43) on l10 at equal IR — otherwise ship the periodic schedule, which needs no judge at all.
3. **Selector vs judge is resolved** (b0perk3): both halves of B0 are dead in the mixed line (selector −11.8, judge −8.2). No further
   B0 arms; every mixed arm is built on AWM (borrowed prior on π0.5 at 50 episodes, plain AWM at 500).
4. **GR00T mixed arms** (none run in R3: GPU budget). GR00T-l10 is the largest remaining gap (CL2-500 .706 / CL3-500 .736 vs .870) and
   GR00T-sp is already at the policy with 500 episodes; run V7 + guards h .7 and guard-only on GR00T-l10 at both library scales,
   with the measured stage split for IR (GR00T s1 58 / s23 232 ms single-server). The stuck / overtime guards never fire on GR00T
   (still-proxy off scale, H3 replay), so the guard set must be re-thresholded on GR00T library statistics first.
5. **Per-task budget allocation on l10**: realized h ranges .49 (t0) … .80 (t7) under one τ; t8 stays the floor in every arm (.36–.56)
   and is policy-hard (.36–.40). A per-task τ (or per-task periodic k) from the pilot logs is a cheap arm; the metric is SR at equal
   total IR.
6. **Pilot protocol**: keep 100-init pilots for collapse screening only; anything that must be *ranked* on l10 needs ≥ 250 inits
   (± 4.5 pp) or the full 500. The 5-task l10 subset should be re-drawn to include the tasks where the method layer moves (t9, t2, t7)
   or replaced by a random 250-init sample.
7. **Cheap pure-cache follow-ups only if the library ruling is "trajectories"**: V4 (st = per-model) as a no-borrowing control on
   π0.5-sp; a05 with GR00T-l10 t8 excluded from the borrowed statistics (its collapse is a single-task side effect). Otherwise stop
   the pure-cache metric line: its whole R3 yield (+4–5 pp on π0.5) is a third of the library layer and a quarter of the control layer.

---

## 6. Caveats

- The pure-inference reference is one closed-loop sample (2026-09-23) of a stochastic policy; "equals the policy" means
  indistinguishable on 500 paired inits (± 2–3 pp), not better. The l10 arms that are nominally above it (+2.4 / +2.8) are n.s.
- MX arms ran under a shared, changing GPU load (4 → 3 → 2 servers, a foreign training job); latencies are load numbers, IR uses the
  project formula; the policy and the judge are identical across server counts. `ev_h70` (l10) and `awm_h50` (sp) were resumed from
  their journals; `ev_h70`'s h is .64–.66 depending on the attempt bookkeeping. The last three l10 arms (`b0perk3`, `perk5`, `g500`)
  ran on 2 servers × 32 workers: their stage times are the most inflated of the round (s1 547–594 ms) and g500's measured IR (.289)
  is above its formula IR (.238) for that reason alone (§2.5); SR is unaffected (the policy and judge are identical).
- Served-action KPIs (flips, vote split, classes) are reconstructed from the logged top-10 of 16 kernel members (≈ .03 σ RMS, ideation
  C); gripper signs and votes are robust to it. In mixed arms spells / flips are computed on HIT decisions only.
- "Proposal spell" timing in the mixed arms is counterfactual after the first MISS; the early-share numbers say when the judge fires
  relative to what the cache *would* have repeated, not relative to a physical failure.
- The pilot subsets differ per cell (π0.5-sp {6,9,0,4,1}, π0.5-l10 {0,4,6,8,7}, GR00T-sp {8,2,3,7,5}); pilot SRs are not comparable
  across cells, only within a cell against the same-subset R2 rows.

---

## 7. Reproduction (read-only on the run roots; CPU 18-33,62-77; scratch `/home/weiland/.claude/jobs/a607dd74/tmp/analysis_r03/`)

```bash
cd /home/weiland/projects/openpi; S=/home/weiland/.claude/jobs/a607dd74/tmp/analysis_r03
bash $S/regen_kpi.sh                                   # kpi.py per arm group + R2 reference rows -> kpi/*.json|md (≈ 40 s, 27 procs)
taskset -c 18-33,62-77 .venv/bin/python $S/pair.py --pairs $S/pairs_r03.txt --md $S/pair/pairs_r03.md --json $S/pair/pairs_r03.json
taskset -c 18-33,62-77 .venv/bin/python $S/r3_tables.py     # -> TABLES.md (pilot / full / mixed / frontier / per-task / paired)
taskset -c 18-33,62-77 .venv/bin/python $S/mech.py          # -> MECH.md (grip_commit / term_guard / release_guard evidence)
taskset -c 18-33,62-77 .venv/bin/python $S/runs_mx.py       # -> RUNS_MX.md (MISS/HIT run structure, policy-majority failures)
taskset -c 18-33,62-77 .venv/bin/python $S/frontier.py      # -> FRONTIER.md (above-time-share-line, marginal pp / 0.1 IR, noise floor)
```

All arms are complete; the scripts re-run unchanged (a missing journal would show as a `PENDING` row in `pair.py` / `TABLES.md`).
