# Round-2 screening results, live (inits 20-29 only; 100 pairs; h100 + timan107; paired vs same-topology controls)

Updated 2026-10-02 00:05 CDT. Controls on the same 100 pairs (run root r09_fable_grown, round 1): pure cache .740 @ .0765, pure policy .930 @ .504.
Analysis: `python -m exp.offline_search.rounds.r09.explore_fable.round2.tools.paired_r2 --ref grown:r9f_ctrlA_p_l10_50 <arms>`.
Chain 1 (tmux `r9f_r2`, coordinator-run) and batch 2 (tmux `r9f_r2b`, queued behind it) are monitored by the coordinator; I only read journals.

| arm (pi0.5 LIBERO-10-50) | SR | IR (calls/decisions) | vs cache [95% CI] | +/- | p | per task (arm / cache) |
|---|---|---|---|---|---|---|
| corr1pt: corrector FULL strength, per-task heads fitted on inits 0-19 | .630 | .0762 (0/6963) | -11.0 pp [-23, 0] | 14/25 | .11 | 0:.6/.4 1:.6/.9 2:.6/1.0 3:.8/1.0 4:.4/.4 5:1/1 6:.4/.5 7:.8/.8 8:.5/.7 9:.6/.7 |
| corr1single: FULL strength, one task-agnostic head | .540 | .0763 (0/7698) | -20.0 pp [-32, -8] | 12/32 | .004 | 0:.4/.4 1:.6/.9 2:.4/1.0 3:.7/1.0 4:.3/.4 5:.8/1.0 6:.5/.5 7:.5/.8 8:.7/.7 9:.5/.7 |
| gm_corr1pt: empty-grasp trigger (2-call burst, <=2 triggers) on top of the FULL corrector | **.790** | **.0835** (54/6467) | **+5.0 pp [-5, +14]** vs cache; **+16 pp vs its own base corr1pt** (.630) | 17/12 | .46 | 0:.9/.4 1:.9/.9 2:.8/1.0 3:.8/1.0 4:.7/.4 5:1/1 6:.4/.5 7:1.0/.8 8:.8/.7 9:.6/.7 |
| gm_corr0 (empty-grasp trigger on the plain cache) | running (started 00:02) | | | | | |
| GR00T L10-50 cache control (`r9f2_groot_l10_50_corr0`) | .560 | pending ledger | reference for the GR00T arms below | | | |
| GR00T L10-50 corrector FULL strength (per-task heads fitted on 0-19) | **.700** | pending ledger | **+14 pp vs its cache control .560** (opposite sign to pi0.5, where full strength cost -11 pp) | pending | pending | pending |
| GR00T L10-50 gm_corr1pt (inverted trigger, see caveat) = .690 ≈ its base corr1pt .700, as predicted; gm_corr0 (inverted trigger) = .600 vs cache .560, within noise of its base, as expected | | **INVALID as trigger tests**: these classes test `hist > 0` for 'closed', which is pi0.5's convention; GR00T's normalized gripper is +1 = open / -1 = close (verified on recorded apertures), so the GR00T trigger never fires and the arms equal their bases. Sign-corrected GR00T trigger arms (`GraspMissCallsSigned`) are prepared in r09_fable_r2d. Same caveat for batch-2 `r9f2_groot_l10_50_gm_corr05pt`. | | | |
| pi0.5 Spatial-50: corr0 / corr1pt / gm_corr1pt | queued | | | | | |
| batch 2 (tmux r9f_r2b, coordinator-queued): corr05pt, gm_corr05pt (both L10 cells), np_corr05pt, gm_corr05pt_pu15 | queued | | | | | |

Reading so far:
1. **Full-strength correction hurts closed loop on LIBERO-10-50** (-11 pp per-task heads, -20 pp single head) although it is the best variant offline (held-out gap -31%). Correction strength must stay at or below half; the single task-agnostic head is worse than per-task heads.
2. **The empty-grasp recovery call is a large effect.** Put on top of the harmful full-strength base it lifts .630 -> .790 (+16 pp) for 54 policy calls in 100 episodes (IR +.007 over the base, .0835 total): roughly one episode in four triggered once, and the 2-call burst rescued most of the tasks the base was losing (task 0 .6 -> .9, task 4 .4 -> .7, task 7 .8 -> 1.0, task 8 .5 -> .8). It is already +5 pp over the plain cache at +.007 IR, with the handicap of a base that is 11 pp below cache. The clean test is `gm_corr0` (trigger on the plain cache, running now) and batch-2 `gm_corr05pt` (trigger on the half-strength base).

**00:08 update.** `gm_corr0` = plain cache + empty-grasp recovery: .790 @ .0863 vs cache .740 @ .0765 (+5 pp, +8/-3 discordant pairs; the trigger touches few episodes and almost only flips them upward). Same SR as the trigger on the full corrector (gm_corr0 vs gm_corr1pt: 0 pp, 13/13). IR cost +.010 for 0.74 recovery calls per episode. This is the cheapest LIBERO-10-50 point above the cache so far (R8 uniform calls reach .79 only at IR .19). GR00T L10-50 and Spatial-50 arms are running next; batch 2 (half-strength bases, no-progress combination, uniform coin) follows.

**00:25 update (paired, GR00T L10-50, inits 20-29, ref = cache .560 @ .0742).** corr1pt (FULL-strength corrector, heads fitted on 0-19): **.700 @ .0744, +14 pp [+5, +23], +20/-6, p .009**; gm_corr1pt .690 (+13 pp [+4, +22]; 0 calls - inverted trigger never fired); gm_corr0 .600 (+4 pp [+1, +7], 0 calls; noise). The full-strength corrector is the first cheap GR00T LIBERO-10-50 gain in the program (R8 uniform calls need IR .30 for .80). Spatial-50 arms (cache / corr1pt / gm_corr1pt), batch 2 and r2c are next in the coordinator's queue; r2d (re-anchor + sign-corrected GR00T triggers) is prepared.
