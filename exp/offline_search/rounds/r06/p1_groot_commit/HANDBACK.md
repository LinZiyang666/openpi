# P1 hand-back: configuration B (Commit-Cache + committed policy rescue) on GR00T

Final check 2026-09-28 10:55 America/Chicago (15:55 UTC). Execution authority, under the standing
search-tuning charter (no status card, no G1/G2). Written only `exp/offline_search/rounds/r06/p1_groot_commit/**`,
`/home/weiland/trace_runs/os_closed_loop/r06_paper/{fits,prefit_logs}` and `/tmp/p1_*`. No shared file was edited
(no `src/`, `harness/`, `closed_loop/*`, `ops/*`, earlier rounds), so no plugin install / regression matrix was needed.
No GPU, server, port, LIBERO worker, chain, tmux, timan107, git or `tests/review_tests/` access. Every Python process:
`taskset -c 26-29,70-73`, OMP/OpenBLAS/MKL threads 1, `CUDA_VISIBLE_DEVICES=''`, repo `.venv/bin/python`, at most 8
processes at once.

## 0. Summary

- `GrootCommitJudge` (`judge.py`) = C10 `CommitJudge` + one model constant (closed-gripper sign). It fits on GR00T
  (thresholds from the GR00T library by the same code), and on pi0.5 it is C10 bit for bit.
- A ⊂ B: B with its MISS trigger off (`guards=False`) serves exactly A's chunks on all 216,474 recorded decisions (8 cells x 500 episodes) once
  A uses the wrapper's documented k-th-slot tie rule; the deployed A differs from B-off only at 4 exact float32
  distance ties (1 anchor + its blind tail in 4 of 8 cells), a pre-existing A-vs-C10 property present on pi0.5 too.
- Same logic as C10: P1 on pi0.5 == C10 on every non-timing log field and served chunk over 8 full replays
  (189,904 decisions, 8.76 M compared fields); fitted state equals the deployed C10 fits; on GR00T all-vision decisions, C10's own code on
  gripper-mirrored history == P1 (21,724 decisions, 0 mismatches).
- GR00T B replay selftests (8 full replays, 187,784 decisions): every MISS is guard-forced, every non-final MISS gets
  exactly one policy tail executing the MISS chunk's rows 5..9, never more than one vision-free decision in a row,
  K4 ledger vision/MISS counts equal the plugin's. Plugin selftests 4/4 and 8-connection concurrency parity 4/4 PASS.
- 4 prefit artifacts in `r06_paper/fits`, `arms_p1.json` (4 arms) and `arms_rep.json` (32 replicate rows) validated.

## 1. What B is, and how it is switched on for GR00T

B = R5 C10 (`rounds/r05/q1_commit/judge.py:CommitJudge`, `policy_tail_gate="lifecycle"`) = A plus a committed
policy rescue. A real vision anchor retrieves/synthesizes an AWM chunk; K1 `anchor_tail` budget 1 serves that chunk's
rows 5..9 on the next, vision-free decision (10 controls per cache source, no policy call). The only MISS source is
the K7 vision-confirmed guard set (MixedJudge guards 1 stuck, 2 terminal-row-with-closed-gripper, 3 overtime-and-lag,
4 no-progress; events none, burst 0) under `--os-judge guard_only`. A MISS executes the policy chunk's rows 0..4 and
the lifecycle policy tail executes its rows 5..9 on the next, vision-free decision (10 controls); the decision after
that is vision again. GR00T's H=16 rows 10..15 are never executed (`--os-policy-tail-blocks 1`).

| item | GR00T B (new) |
|---|---|
| method | `exp.offline_search.rounds.r06.p1_groot_commit.judge:GrootCommitJudge` |
| kwargs, 50-episode library | `{"base_kwargs":{"lib":"current","kref":5,"serving":"anchor_tail","budget":1,"gates":"budget_only"},"progress_guard":"noprog_span","events":"none","stuck_guard":"vision_confirmed","policy_tail_gate":"lifecycle","monitor":"off"}` |
| kwargs, 500-episode library | same with `"lib":"big","kref":8` (GR00T big = `bpool_all`, 500 episodes) |
| plugin flags | `--os-root /home/weiland/trace_runs/offline_search_store --os-blind --os-policy-tail --os-policy-tail-blocks 1 --os-judge guard_only --os-no-shadow-native --os-fit-artifact <RUN>/fits/<arm>.pkl` |
| arm fields | `full_model: true`, `cost_ledger: true`, `client_overrides: {"replan_steps": 5, "resize_size": 256}` |

The kwargs are byte-for-byte C10's (`r05_q1/r5q1_c10_p_*`): the A base kwargs (`r05_x` `tail1u`: lib/kref,
anchor_tail, budget 1, budget_only) plus C10's guard/lifecycle settings. The client equals A's effective one
(`run_arm.sh` already uses resize 256 and L5 for any GR00T yaml). No plugin change: Q2's installed
`--os-policy-tail-blocks` path is used unchanged.

## 2. Why K7 refused GR00T, and the fix

K7 (`rounds/r04/k7_guard/judge.py:35-40`) raises `SkipCell("K7 supports pi05 only; GR00T stock terminal semantics
differ")`. With an all-vision history K7 runs stock `MixedJudge.query` (`rounds/r03/h3_judge/judge.py:225,247`), whose
guard 2 fires on `term1 and gexec > 0`, `gexec = sign(prev_a_exec[4, 6])`: "closed" in LIBERO's positive-close
convention, which pi0.5's normalized actions use. GR00T's normalized gripper is openness (the wire adapter applies
`1 - 2x` then `sign`), so closed is `< 0`: stock on GR00T fires guard 2 on "terminal row while OPEN" and never on
"terminal row while still holding". The K1/K7 gap branch (`k1_blind/judge.py:111`, `k7_guard/judge.py:104`) already
has `closed = gexec < 0 if groot else gexec > 0`, so K7's two branches disagreed on GR00T.

Measured on the store libraries (`current`; dim 6 of the executed head; next-row aperture from `rs[6]-rs[7]`):

| library | step-0 share with g >= 0 | aperture change after g >= 0 | after g < 0 | closed means |
|---|---:|---:|---:|---|
| pi05 spatial | 0.000 | -0.166 | +0.005 | g >= 0 |
| pi05 l10 | 0.000 | -0.088 | +0.042 | g >= 0 |
| GR00T spatial | 1.000 | +0.007 | -0.149 | g < 0 |
| GR00T l10 | 1.000 | +0.042 | -0.084 | g < 0 |

(Episodes start open; closing commands shrink the aperture.) Every other guard input is sign free: stuck (state L2
motion + min-camera centred key cosine), overtime/lag, no-progress; the optional grip event compares like with like.
Action dims beyond `[:, :7]` never enter a guard; policy tails copy the whole chunk verbatim.

Fix (`judge.py`, ~60 lines of logic): `CLOSED_SIGN = {"pi05": +1, "groot": -1}`. `GrootCommitJudge(CommitJudge)`
overrides only
- `fit`: pi05 runs C10's fit unchanged; GR00T skips exactly K7's refusal (K7.fit adds nothing else; CommitJudge.fit
  adds only the refused optional monitor) and runs the same MixedJudge / V7 / K1 fit on the GR00T library;
- `query`: after the stock all-vision verdict, guard 2 is recomputed with the model sign and os_flags / os_reason /
  os_force_miss re-derived exactly as stock does (lowest firing bit), keeping the `_s["flag"]` memo consistent. For
  pi05 this is the identity. The gap branch, `blind_step`, and `policy_tail_step` (C10 lifecycle) are inherited.

Refused: `burst > 1` (re-derivation would need the pre-decision burst state that stock overwrites; C10 deploys
burst 0), `policy_tail_gate != "lifecycle"`, `monitor != "off"`, `stuck_guard != "vision_confirmed"`,
`progress_guard != "noprog_span"`, and (inherited from K10) non-`anchor_tail` or budget > 1.

Where the fix acts: in B's own schedule every anchor is followed by a vision-free decision, so the stock branch runs
only at each episode's step 0 (exactly 500 per 500-episode replay, section 4.3), where no guard can fire
(gexec = 0). In the B replays it therefore changed 0 decisions; B's GR00T terminal guard is decided by the gap branch,
which was already sign-correct. The fix matters in K7's all-vision regime (542 of 21,724 GR00T verdicts change,
section 4.2) and whenever a lifecycle look puts two vision decisions back to back early in an episode; it removes the
inconsistency that made K7 refuse GR00T.

Thresholds come from the same fit code on the GR00T library, exactly as K7/C10 derive them for pi0.5: m_thr = library
10th percentile of consecutive-row valid-state L2 motion, c_thr = 95th percentile of the consecutive-row min-camera
task-centred key cosine, V7's library leave-one-episode-out isotonic calibration (ncal 3000), per-task median episode
length, K1 per-task state scales and motion10. Fixed constants (stuck_thr 2, lag_thr 5, noprog_n 3, prog_eps 0.5) are
C10's. Values in section 6.

## 3. Nesting A ⊂ B and shared decision logic

"MISS trigger disabled" = the deployed B class and kwargs with `guards=False`, fitted fresh (the fit does not depend
on `guards`). That switches off the forced MISS and K1's no-progress blind veto (`LookReason(8, "noprog_span")`). In
the replays with guards on, that veto never produced a separate look: every vision look reason is 1 (budget) or 6
(lifecycle: first decision / after a policy tail), because with anchors two decisions apart the no-progress span jumps
straight to 2 and forces the MISS at the same anchor. B's vision schedule therefore equals A's (identical vision
counts in all 8 cells, e.g. 19,891 on GR00T l10), and B departs from A only by replacing a cache source with a
committed policy source (and AWM's post-MISS continuity regime afterwards, as in C10).

Nesting replay (all 500 recorded cache-cell episodes per cell; A = deployed A spec with its ORIGINAL fit artifact;
B-off fitted fresh; the tie-rule A = test-only `tie_rule.StableTieBlindAWM`, identical to A except
`np.argsort(dt, kind="stable")[:k]` in place of `argpartition` + sort, fitted fresh):

| model | cell | decisions | vision | blind | tie-rule A == B-off | deployed A != B-off (decisions / anchors) | max abs diff | tie audit (A-only -> B-only row @ equal distance) |
|---|---|---:|---:|---:|---|---|---:|---|
| pi05 | l10 50 | 40127 | 20118 | 20009 | bit-exact | 0 / 0 | 0 | - |
| pi05 | l10 500 | 40127 | 20118 | 20009 | bit-exact | 2 / 1 | 0.009855 | step 0: 25665 -> 25132 @ 8.74196 |
| pi05 | sp 50 | 14621 | 7400 | 7221 | bit-exact | 0 / 0 | 0 | - |
| pi05 | sp 500 | 14621 | 7400 | 7221 | bit-exact | 2 / 1 | 0.01548 | step 0: 2590 -> 2177 @ 9.12372 |
| GR00T | l10 50 | 39669 | 19891 | 19778 | bit-exact | 2 / 1 | 0.08842 | step 98: 261 -> 121 @ 30.4086 |
| GR00T | l10 500 | 39669 | 19891 | 19778 | bit-exact | 2 / 1 | 0.02185 | step 0: 4410 -> 4022 @ 11.159 |
| GR00T | sp 50 | 13820 | 7000 | 6820 | bit-exact | 0 / 0 | 0 | - |
| GR00T | sp 500 | 13820 | 7000 | 6820 | bit-exact | 0 / 0 | 0 | - |

Every served H x 32 float32 chunk, the vision mask, the source sequence and HIT flags are compared; B-off never MISSes.
Deployed A differs from B-off at exactly the decisions where it differs from the tie-rule A, each an anchor plus its
own blind tail, and `analyze.tie_audit` verifies each: the swapped 16th kernel member has a float32 distance exactly
equal to the other tied row's and to the 16th-smallest distance, and B keeps the lower row. Cause: AWM.query
(`g1_awm/awm.py:447`) uses `argpartition`, whose choice among rows tied across the k-th slot is arbitrary, while
the G3 wrapper under MixedJudge/K7/K10/C10/P1 (`g3_core.topk_pos`) takes the lowest row; the AWM comment at
`awm.py:572` assumes they agree. Pre-existing and identical on pi0.5 (A vs C10); left unchanged because changing it
would change the deployed A or C10.

Shared logic with pi0.5 C10:
1. P1 on pi0.5 == C10 through the installed plugin, all 500 episodes of cache AND inf replay streams, 4 cells:
   every non-timing decision-log field (method name / tag / timings excluded, winner tag normalized) and every served
   chunk equal.

| cell | replay | decisions | MISS | policy tails | compared row fields |
|---|---|---:|---:|---:|---:|
| l10 50 | cache | 40127 | 8191 | 8135 | 1,853,533 |
| l10 50 | inf | 29406 | 3793 | 3703 | 1,355,969 |
| l10 500 | cache | 40127 | 8487 | 8431 | 1,853,829 |
| l10 500 | inf | 29406 | 3175 | 3102 | 1,355,351 |
| sp 50 | cache | 14621 | 2440 | 2297 | 674,506 |
| sp 50 | inf | 10798 | 588 | 397 | 496,796 |
| sp 500 | cache | 14621 | 2449 | 2312 | 674,515 |
| sp 500 | inf | 10798 | 490 | 281 | 496,698 |

2. P1 fitted on pi0.5 equals the deployed r05_q1 C10 fits in every fitted field (V7 calibration, m_thr, c_thr, task
   key means, med_len, disp_thr, k, T; K1 blind_next/blind_rs/blind_event/blind_terminal/motion10/state scales), all 4
   cells; the pi0.5 thresholds reproduce K7's published values.
3. Mirror test (GR00T, 100 evenly spaced recorded cache episodes per fit, all-vision stale regime): C10's own code
   (the fitted P1 object re-classed to `CommitJudge`) fed the history with the executed gripper column negated
   (openness -> positive-close) returns P1's Result exactly (topk, scores, action, confidence bytes, every extras
   value, `_s` memos, progress span), except `gexec`, which is negated.

| fit | decisions | mismatches | terminal fires P1 | terminal fires with stock sign | force-MISS verdicts changed by the fix |
|---|---:|---:|---:|---:|---:|
| l10 50 | 8226 | 0 | 347 | 346 | 141 |
| l10 500 | 8226 | 0 | 180 | 149 | 150 |
| sp 50 | 2636 | 0 | 301 | 180 | 123 |
| sp 500 | 2636 | 0 | 231 | 110 | 128 |

Model-specific constants of the shared B logic:

| constant | pi05 B (C10, `CommitJudge`) | GR00T B (P1, `GrootCommitJudge`) |
|---|---|---|
| closed normalized gripper (guard 2) | `prev_a_exec[4,6] >= 0` | `prev_a_exec[4,6] < 0` |
| action horizon H | 10 | 16 (rows 10..15 never executed) |
| policy tail switch | `--os-policy-tail` (1 block implicit) | `--os-policy-tail --os-policy-tail-blocks 1` |
| MISS denoising steps (`miss_k`) | 10 | 8 |
| robot_state key | 32-d (valid 8) | 8-d |
| 500-episode library (`lib: big`) | `bpool_cs` | `bpool_all` |
| m_thr / c_thr / V7 calibration / med_len / motion10 | fitted on the pi05 library | fitted on the GR00T library (same code) |
| client | L5 | L5, resize 256 (same as GR00T A) |
| identical | kwargs; stuck_thr 2, lag_thr 5, noprog_n 3, prog_eps 0.5, burst 0, events none, ncal 3000, m_pct 10, c_pct 95, AWM k 16 / kref, anchor_tail budget 1 budget_only, lifecycle policy tail, guard_only | same |

## 4. Tests and results (final files; numbers from `results/*.json`)

Replay protocol (`replay.py`): the REAL installed plugin (`plugin.install`, per-connection CacheOrchestrator,
PluginStrategy/Judge/Storage, blind path, policy-tail bookkeeping, decision logs) is driven over ALL 500 recorded
episodes of a store query cell; the fake interceptor answers a MISS with the recorded full inference `a_inf[row]` and
the blind output transform is the identity, so each returned action is the normalized H x 32 chunk that was served.
Observations are the recorded ones (they do not respond to served actions): decision audits on fixed observation
streams, not rollouts. 48 replay jobs, 1,217,014 decisions.

### 4.1 Method-level tests (`method_tests.py`, `results/method_tests_*.json`) - all PASS
- refusals (9): burst 2, inherited gate, monitor, dense stuck guard, noprog_n, budget 2, phase_particles; unknown
  model skipped before fit; C10 `CommitJudge` still refuses GR00T (unchanged).
- terminal re-derivation units (8): bit added / removed / kept with other bits, guards off, non-terminal rows,
  confidence and extras key order unchanged, memo updated; nonzero burst state refused.
- GR00T policy-tail lifecycle (per fit, recorded inf episodes 0/123/250/499): 80 / 80 / 74 / 74 MISS tails served
  (l10 50 / l10 500 / sp 50 / sp 500), each the exact 16-row `policy_tail_chunk(miss_chunk, 5)` with rows 0..4 equal
  to the MISS chunk's rows 5..9 in all 32 columns, even with zero cache budget, all gates on and a positive span;
  one use only (second call -> lifecycle look); cache anchor stays invalid; 15 lifecycle vetoes per fit (step 0,
  task, episode, prev HIT, blind age, vision/hit histories, executed_steps 4 and 10, non-finite rs/raw_state,
  short/NaN chunk, empty history, reset).
- fits and mirror: section 3.

### 4.2 All-vision guard trigger counts (K7's B=0 regime, `allvision_rates.py`, every recorded episode)

Recorded QueryViews, recorded executed history, deployed B fits (GR00T P1 r06_paper; pi0.5 C10 r05_q1). Reason =
lowest firing bit. The pi0.5 stuck counts equal K7's published all-vision table exactly (e.g. l10 cache 50: 7,154).

| model | cell | lib | decisions | force MISS (share) | stuck | terminal-closed | overtime | no-progress | terminal rows: closed by model sign / opposite sign |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| GR00T | l10 cache | 50 | 39669 | 16107 (40.60%) | 6309 | 926 | 412 | 8460 | 2394: 1259 / 1135 |
| GR00T | l10 inf | 50 | 29065 | 5966 (20.53%) | 2007 | 286 | 123 | 3550 | 726: 321 / 405 |
| GR00T | l10 cache | 500 | 39669 | 14137 (35.64%) | 3683 | 492 | 246 | 9716 | 1270: 570 / 700 |
| GR00T | l10 inf | 500 | 29065 | 4572 (15.73%) | 952 | 193 | 58 | 3369 | 450: 193 / 257 |
| GR00T | spatial cache | 50 | 13820 | 3812 (27.58%) | 1766 | 910 | 125 | 1011 | 2668: 1742 / 926 |
| GR00T | spatial inf | 50 | 11338 | 1374 (12.12%) | 377 | 469 | 50 | 478 | 545: 487 / 58 |
| GR00T | spatial cache | 500 | 13820 | 3300 (23.88%) | 1324 | 872 | 95 | 1009 | 1980: 1322 / 658 |
| GR00T | spatial inf | 500 | 11338 | 1119 (9.87%) | 207 | 404 | 24 | 484 | 484: 422 / 62 |
| pi05 | l10 cache | 50 | 40127 | 14994 (37.37%) | 7154 | 1423 | 307 | 6110 | 3777: 2745 / 1032 |
| pi05 | l10 inf | 50 | 29406 | 5785 (19.67%) | 2137 | 327 | 283 | 3038 | 649: 333 / 316 |
| pi05 | l10 cache | 500 | 40127 | 14033 (34.97%) | 4859 | 972 | 249 | 7953 | 2126: 1109 / 1017 |
| pi05 | l10 inf | 500 | 29406 | 4193 (14.26%) | 1060 | 236 | 156 | 2741 | 672: 236 / 436 |
| pi05 | spatial cache | 50 | 14621 | 4792 (32.77%) | 1521 | 1377 | 189 | 1705 | 3569: 2176 / 1393 |
| pi05 | spatial inf | 50 | 10798 | 1025 (9.49%) | 76 | 496 | 28 | 425 | 554: 512 / 42 |
| pi05 | spatial cache | 500 | 14621 | 4675 (31.97%) | 1485 | 1500 | 103 | 1587 | 3652: 2216 / 1436 |
| pi05 | spatial inf | 500 | 10798 | 900 (8.33%) | 57 | 513 | 19 | 311 | 564: 524 / 40 |

On the GR00T spatial inf cell at 500, 422 of 484 terminal rows have a closed gripper under the GR00T sign; stock's
sign would instead fire on the other 62 (terminal while open).

### 4.3 GR00T B replay selftests (`analyze.py`, `results/replay_checks.json`) - all asserted, PASS

Per decision, exactly: first decision of each episode is vision; at most one vision-free decision in a row; a blind
decision follows a vision decision; every MISS row has `judge == "force:<os_reason>"`, `os_force_miss == 1`,
`src == "policy"`, non-null `miss_k` and `s23_ms`, and every vision row with a fired guard is a MISS (no MISS without a
guard, no ignored guard); logged `os_flags` equal an independent recomputation from the logged extras with the GR00T
sign; every non-final MISS is followed by exactly one `src == "policy_tail"` row (`vision=false`, `hit=true`) whose
served chunk is byte-equal to `policy_tail_chunk(MISS chunk, 5)`, then a vision row; every cache blind row serves the
anchor's rows 5..9 (dims 0..6; K1 zero-pads the rest); blind rows have null stage timings and `miss_k`; K4 `ledger()` on the rows reports vision and MISS
counts equal to the plugin's (vision == stage-1 calls); the policy-tail rows alone cost 0 (0 vision, 0 MISS).

| cell | replay | decisions | vision | cache blind | MISS | policy tails | MISS at episode end | v | m | MISS / vision | owner IR per 5 controls | MISS reasons |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| l10 50 | cache | 39669 | 19891 | 10604 | 9235 | 9174 | 61 | 0.5014 | 0.2328 | 0.4643 | 0.2736 | no-progress 6119, stuck 2580, terminal 488, overtime 48 |
| l10 50 | inf | 29065 | 14632 | 9979 | 4557 | 4454 | 103 | 0.5034 | 0.1568 | 0.3114 | 0.2095 | no-progress 3558, stuck 805, terminal 182, overtime 12 |
| l10 500 | cache | 39669 | 19891 | 10738 | 9094 | 9040 | 54 | 0.5014 | 0.2292 | 0.4572 | 0.2706 | no-progress 7467, stuck 1432, terminal 181, overtime 14 |
| l10 500 | inf | 29065 | 14632 | 10434 | 4089 | 3999 | 90 | 0.5034 | 0.1407 | 0.2795 | 0.1958 | no-progress 3621, stuck 307, terminal 161 |
| sp 50 | cache | 13820 | 7000 | 4961 | 1999 | 1859 | 140 | 0.5065 | 0.1446 | 0.2856 | 0.1996 | no-progress 825, stuck 772, terminal 369, overtime 33 |
| sp 50 | inf | 11338 | 5784 | 4891 | 824 | 663 | 161 | 0.5101 | 0.0727 | 0.1425 | 0.1392 | no-progress 424, terminal 247, stuck 151, overtime 2 |
| sp 500 | cache | 13820 | 7000 | 5043 | 1915 | 1777 | 138 | 0.5065 | 0.1386 | 0.2736 | 0.1945 | no-progress 1036, stuck 501, terminal 368, overtime 10 |
| sp 500 | inf | 11338 | 5784 | 4927 | 804 | 627 | 177 | 0.5101 | 0.0709 | 0.1390 | 0.1377 | no-progress 496, terminal 231, stuck 77 |

Max vision-free run 1 everywhere. Vision look reasons: only 1 (budget) and 6 (lifecycle). Stock (all-vision) branch
500 decisions per replay (step 0 only), gap branch the rest; terminal-bit changes by the sign fix in these replays:
0. Owner IR = (0.152·vision + 0.848·MISS) / decisions (all decisions are 5-control requests). The K4 ledger's own IR
uses the historical GR00T measured stage table (`cost_source` in the JSON), a different basis.

Same protocol, pi0.5 C10 (deployed r05_q1 fits), for like-for-like comparison:

| cell | replay | decisions | MISS | policy tails | MISS at end | m | MISS / vision | owner IR per 5 controls | MISS reasons |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| l10 50 | cache | 40127 | 8191 | 8135 | 56 | 0.2041 | 0.4071 | 0.2493 | no-progress 4457, stuck 3239, terminal 409, overtime 86 |
| l10 50 | inf | 29406 | 3793 | 3703 | 90 | 0.1290 | 0.2563 | 0.1859 | no-progress 2469, stuck 912, terminal 371, overtime 41 |
| l10 500 | cache | 40127 | 8487 | 8431 | 56 | 0.2115 | 0.4219 | 0.2556 | no-progress 6368, stuck 1907, terminal 198, overtime 14 |
| l10 500 | inf | 29406 | 3175 | 3102 | 73 | 0.1080 | 0.2145 | 0.1681 | no-progress 2660, stuck 396, terminal 115, overtime 4 |
| sp 50 | cache | 14621 | 2440 | 2297 | 143 | 0.1669 | 0.3297 | 0.2184 | no-progress 1101, stuck 703, terminal 550, overtime 86 |
| sp 50 | inf | 10798 | 588 | 397 | 191 | 0.0545 | 0.1064 | 0.1240 | no-progress 306, terminal 261, stuck 19, overtime 2 |
| sp 500 | cache | 14621 | 2449 | 2312 | 137 | 0.1675 | 0.3309 | 0.2190 | no-progress 1198, stuck 688, terminal 498, overtime 65 |
| sp 500 | inf | 10798 | 490 | 281 | 209 | 0.0454 | 0.0887 | 0.1163 | no-progress 215, terminal 259, stuck 16 |

Replay rates overstate live ones: the same C10 l10/500 fit has m = 0.2115 here vs 0.0997 in the live r05_q1 arm
(2,869 / 28,784). Only the comparison under one protocol is meaningful: GR00T B's MISS share per decision is 0.018-0.033
higher than pi0.5 C10's on the l10 cache/inf and spatial inf streams, and 0.022-0.029 lower on the spatial cache
streams; no-progress is the largest MISS reason in every GR00T cell.

### 4.4 Plugin selftests (`plugin_selftests.py`, installed `closed_loop/selftest.py --blind` + `verify_logs`) - 4/4 PASS

Deployed fits; 2 interleaved connections x 2 episodes x 12 decisions, `--os-log-inputs`, offline harness replay of
every logged input (online == offline), duplicate-id rejection. Policy tails equal the eligible MISSes and match the
shifted normalized AND wire chunks.

| arm | decisions | vision | blind | MISS | policy tails | stage-1 calls | duplicate rejections |
|---|---:|---:|---:|---:|---:|---:|---:|
| r6p1_c10_g_l10_50 | 48 | 24 | 24 | 4 | 4 | 24 | 4 |
| r6p1_c10_g_l10_500 | 48 | 24 | 24 | 4 | 4 | 24 | 4 |
| r6p1_c10_g_sp_50 | 48 | 24 | 24 | 0 | 0 | 24 | 4 |
| r6p1_c10_g_sp_500 | 48 | 24 | 24 | 0 | 0 | 24 | 4 |

### 4.5 Eight-connection concurrency parity (`concurrency_test.py`, `results/concurrency_summary.json`) - 4/4 PASS

Copy of Q1's driver with GR00T configs: 8 threaded connections with sleeping fake stage 1, then a fresh process
replays the observed reservation order serially; every action, verdict, history digest and decision row matches;
verify_logs passes; every non-final MISS got its tail.

| config | decisions | vision | blind | MISS | policy tails | peak concurrent stage 1 | serialized / threaded time |
|---|---:|---:|---:|---:|---:|---:|---:|
| r6p1_c10_g_l10_50 | 183 | 93 | 90 | 12 | 11 | 8 | 4.80 |
| r6p1_c10_g_l10_500 | 183 | 93 | 90 | 13 | 13 | 8 | 4.73 |
| r6p1_c10_g_sp_50 | 183 | 93 | 90 | 2 | 1 | 8 | 5.27 |
| r6p1_c10_g_sp_500 | 183 | 93 | 90 | 2 | 2 | 8 | 4.91 |

## 5. Files (all new, under `exp/offline_search/rounds/r06/p1_groot_commit/`)

| file | role |
|---|---|
| `judge.py` | `GrootCommitJudge` (deployed method) + `CLOSED_SIGN` |
| `__init__.py` | package marker for the dotted method spec |
| `arms_p1.json` | 4 GR00T B arms, emit_arms format, `<RUN>` placeholders |
| `arms_rep.json` | 32 replicate rows (`_rep2`, `_rep3`) of A and B, both models x 4 cells; provenance in `results/arms_rep_provenance.json` |
| `make_arms.py` | writes the two arm files, `prefit_commands.json`, `prefit.sh` |
| `prefit.sh`, `prefit_commands.json` | exact prefit commands (`RUN` defaults to `r06_paper`) |
| `replay.py` | fixed-observation replay of recorded store episodes through the real installed plugin + fake policy |
| `run_replays.py` | replay matrix (groups N nesting, P pi05 parity, G GR00T B, T tie-rule control) |
| `analyze.py` | nesting + tie audit, pi05 parity, GR00T B / pi05 C10 selftests and ledger |
| `tie_rule.py` | TEST-ONLY `StableTieBlindAWM` (A with the wrapper's k-th-slot tie rule); never deployed |
| `method_tests.py` | refusals, terminal re-derivation units, GR00T policy-tail lifecycle, pi05 fit equality, GR00T mirror |
| `allvision_rates.py` | all-vision guard trigger counts on every recorded episode, both models |
| `plugin_selftests.py` | installed `selftest --blind` + `verify_logs` on the 4 deployed GR00T B fits |
| `concurrency_test.py` | copy of Q1's 8-connection parity; diff = `configs()`, `--os-policy-tail-blocks 1`, default `--source installed` |
| `validate_arms.py` | emitter / parser / CacheConfig / fit-artifact metadata / verbatim-replica checks |
| `summarize.py` | renders the result tables |
| `results/*.json` | evidence: `replay_checks`, `allvision_rates`, `method_tests_*`, `plugin_selftests`, `concurrency_summary`, `arms_validation`, `arms_rep_provenance` |

SHA256 of the delivered files: `results/file_hashes.txt`.

## 6. Prefits (run; installed plugin `prefit_main`, CPU, protocol 4, default ncal 3000)

Exact per-arm commands in `prefit_commands.json`; `prefit.sh` ran all four in parallel (61 s wall). Example (l10 500):

```bash
cd /home/weiland/projects/openpi
RUN=/home/weiland/trace_runs/os_closed_loop/r06_paper
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.closed_loop.plugin --os-method exp.offline_search.rounds.r06.p1_groot_commit.judge:GrootCommitJudge --os-kwargs '{"base_kwargs":{"lib":"big","kref":8,"serving":"anchor_tail","budget":1,"gates":"budget_only"},"progress_guard":"noprog_span","events":"none","stuck_guard":"vision_confirmed","policy_tail_gate":"lifecycle","monitor":"off"}' --os-cell groot_l10_cache --os-root /home/weiland/trace_runs/offline_search_store --os-log-dir $RUN/prefit_logs/r6p1_c10_g_l10_500 --os-tag r6p1_c10_g_l10_500 --os-blind --os-policy-tail --os-policy-tail-blocks 1 --os-judge guard_only --os-no-shadow-native --os-fit-artifact $RUN/fits/r6p1_c10_g_l10_500.pkl > $RUN/fits/r6p1_c10_g_l10_500.prefit.log 2>&1
```
(50-library arms: `"lib":"current","kref":5`; spatial arms: `--os-cell groot_spatial_cache`.) Startup rows in
`$RUN/prefit_logs/<arm>/` record H 16, `policy_tail_blocks` 1, `miss_steps` 8, guard_only, 632 bytes/entry.

| artifact (`r06_paper/fits/`) | bytes | sha256 | fit_s | library (rows) | m_thr | c_thr |
|---|---:|---|---:|---|---:|---:|
| `r6p1_c10_g_l10_50.pkl` | 36,788,048 | `407836eb861d7eefc58b8ab288f5f95cedb13d677c2d0ef42741baf0a1295d1f` | 13.0 | current (2,645) | 0.0424967 | 0.9053388 |
| `r6p1_c10_g_l10_500.pkl` | 188,520,600 | `7f12910c032328b4b143b0017856436d0c8e002c2f6c020fd1efce80cc14fbd1` | 59.6 | bpool_all (29,631) | 0.0247794 | 0.9511454 |
| `r6p1_c10_g_sp_50.pkl` | 27,893,516 | `9bdfa36b2765c1d8365d60c3de31041cd93de2b208c092cf85cf2db8f816943f` | 4.3 | current (1,063) | 0.1658364 | 0.8307866 |
| `r6p1_c10_g_sp_500.pkl` | 87,999,478 | `b41dae39fb728d22f1b343de7bcd602a6e18fdf5ca598bfffa5ecbfeac837c24` | 24.9 | bpool_all (11,751) | 0.1190655 | 0.8691122 |

The GR00T B pickles are larger than GR00T A's (28.1 / 118.4 / 22.8 / 58.6 MB) for the same reason C10's are larger
than pi0.5 A's: V7's tables, calibration and per-task key means. Each 50-library fit uses only its own current library
(no borrowed 500-episode information). pi0.5 C10 thresholds for comparison: l10 0.0518229 / 0.9574828 (50),
0.0385605 / 0.9844978 (500); spatial 0.0931859 / 0.8892201 (50), 0.0938193 / 0.9056166 (500). GR00T V7 calibration
pseudo-queries: stale/fresh n = 2,595 / 3,000 / 1,013 / 3,000 and step-0 n = 50 / 500 / 50 / 500 for l10-50 /
l10-500 / sp-50 / sp-500 (no regime borrowed). Pickle bytes are not bit-reproducible across fits (the blob stores
`fit_s`); fitted state is (section 3, item 2).

## 7. Arm specs

`arms_p1.json` (literal `<RUN>` only in the fit paths): `r6p1_c10_g_l10_50`, `r6p1_c10_g_l10_500`,
`r6p1_c10_g_sp_50`, `r6p1_c10_g_sp_500`; model groot, method/kwargs/flags as in section 1, `full_model: true`,
`cost_ledger: true`, `client_overrides {"replan_steps": 5, "resize_size": 256}`; 50: `lib current, kref 5`;
500: `lib big, kref 8` (= the A arms `r05_x/r5x_g_*_tail1u`).

`arms_rep.json` (32 rows): `<source>_rep2` and `<source>_rep3` for

| model | config | l10 50 | l10 500 | sp 50 | sp 500 |
|---|---|---|---|---|---|
| pi05 | A | `r5t_p_l10_50_tail1uc` | `r5t_p_l10_500_tail1uc` | `r5t_p_sp_50_tail1uc` | `r4b3_p_sp_500_tail1uc` |
| pi05 | B | `r5q1_c10_p_l10_50` | `r5q1_c10_p_l10_500` | `r5q1_c10_p_sp_50` | `r5q1_c10_p_sp_500` |
| GR00T | A | `r5x_g_l10_50_tail1u` | `r5x_g_l10_500_tail1u` | `r5x_g_sp_50_tail1u` | `r5x_g_sp_500_tail1u` |
| GR00T | B | `r6p1_c10_g_l10_50` | `r6p1_c10_g_l10_500` | `r6p1_c10_g_sp_50` | `r6p1_c10_g_sp_500` |

Source rows come from `r05_ptail/arms_in.json`, `r04_blind/arms_in.json`, `r05_q1/arms_in.json`, `r05_x/arms_in.json`
and `arms_p1.json`; each replicate equals its source row except `name` (asserted). All 36 rows were emitted with the
unchanged `emit_arms` into the scratch root `/tmp/p1_arm_validation` (not the paper root), yamls parsed by
`load_cache_config`, plugin args by `plugin.parse_cli`, and every referenced fit unpickled and matched on spec /
kwargs / cell exactly as the plugin's `_load_and_fit` does (`results/arms_validation.json`). Referenced existing fits:

| fit | bytes | sha256 |
|---|---:|---|
| `r05_ptail/fits/r5t_p_l10_50_tail1uc.pkl` | 26,091,371 | `6dfc2f714e58d9c9b8e48a8c3c5ec7b5b80bad95d28df6041c44fc012de51208` |
| `r05_ptail/fits/r5t_p_l10_500_tail1uc.pkl` | 95,264,533 | `93d56ffe81d4874c988f1e16eb5a6a9850268853189866054525f6d137e94504` |
| `r05_ptail/fits/r5t_p_sp_50_tail1uc.pkl` | 21,909,613 | `54a10c4a986d492a49ebd109eed1ee2fd7710631be6effe4c2a182662487a89e` |
| `r04_blind/fits/r4b3_p_sp_500_tail1uc.pkl` | 47,409,015 | `9b219adb2f978736d18c200a46262e119d902a3a865cfeb2a69a309ef68082c0` |
| `r05_q1/fits/r5q1_c10_p_l10_50.pkl` | 32,704,614 | `16ddfec3dc8d3af95c1a39c1f483475746e3ee8194e0730ef38a7f043b77c69a` |
| `r05_q1/fits/r5q1_c10_p_l10_500.pkl` | 142,348,649 | `186f635e234a229f7bbbe1fb93b7ed7c2a9d43581d138294aa2e978e5957cb9d` |
| `r05_q1/fits/r5q1_c10_p_sp_50.pkl` | 26,076,609 | `6cadcc8f75a127601ccf8bca3520d46eedd766c80d6b10a351e2fd5195f2654f` |
| `r05_q1/fits/r5q1_c10_p_sp_500.pkl` | 66,500,639 | `3ef80beaf240680327e75c07c2f067fd7e79f50b656e71cae3af04c1ccfbc315` |
| `r05_x/fits/r5x_g_l10_50_tail1u.pkl` | 28,135,621 | `e7df33a2b71949be66b5d2633cb0df55b62dc03e2cb38038f417250d96397404` |
| `r05_x/fits/r5x_g_l10_500_tail1u.pkl` | 118,431,046 | `c7727c501730949437f2bad7e06b685f4645153b3fe4348f9148aa8f64e5b96a` |
| `r05_x/fits/r5x_g_sp_50_tail1u.pkl` | 22,842,018 | `d0e6d03f4c758ff32e5fda5d0accf01669de6efeb93a8eff6011902ef27d2cb9` |
| `r05_x/fits/r5x_g_sp_500_tail1u.pkl` | 58,604,462 | `aca68fde36b9a988955ddbb7ee313c476e321a99592ae48b01f3d3f69d08c8e8` |

(r05_q1 hashes equal Q1's hand-back.)

## 8. Coordinator next steps (not executed here)

A separate smoke root keeps DONE markers away from the paper run; fit paths still resolve to `r06_paper/fits`
(only fit paths use `<RUN>` in `arms_p1.json`).

```bash
cd /home/weiland/projects/openpi
RUN=/home/weiland/trace_runs/os_closed_loop/r06_paper
SMOKE=/home/weiland/trace_runs/os_closed_loop/r06_paper_smoke
D=exp/offline_search/rounds/r06/p1_groot_commit
sha256sum $RUN/fits/r6p1_c10_g_*.pkl          # compare with section 6
mkdir -p $SMOKE
sed "s|<RUN>|$RUN|g" $D/arms_p1.json > $SMOKE/arms_p1_resolved.json
taskset -c <coordinator cpus> env OPENBLAS_NUM_THREADS=1 .venv/bin/python -m exp.offline_search.closed_loop.ops.emit_arms \
    --run-root $SMOKE --spec $SMOKE/arms_p1_resolved.json
bash exp/offline_search/closed_loop/ops/sync_remote.sh $SMOKE r6p1_c10_g_sp_50 r6p1_c10_g_l10_500
unset OSCL_MANIFEST
PORTS=<free ports> SERVER_CPUS=<cpus> OSCL_TASKS=0,1 OSCL_EPISODES=0,1 WPS=2 \
    bash exp/offline_search/closed_loop/ops/chain.sh $SMOKE r6p1_c10_g_sp_50 r6p1_c10_g_l10_500
```

Audit the smoke decision logs (all hold on the CPU replays): startup row `policy_tail_blocks=1`, `H=16`,
`miss_steps=8`, `method` starting `P1__`; every MISS row `judge == "force:<r>"` with `extras.os_force_miss == 1`;
every non-final MISS followed by exactly one `src=policy_tail` row (`vision=false`, `hit=true`) and then a vision
row; never two vision-free decisions in a row; `stage1_calls` equals the vision count; client at 5 controls per
request. Then the paper run: resolve `<RUN>` to `$RUN`, emit `arms_p1.json` and `arms_rep.json` into `$RUN`, run like
the other A/B arms (full model for B, the 500 paired A-pool inits). Replicate rows keep their original absolute fit
paths; GR00T B replicates use `<RUN>/fits/r6p1_c10_g_*.pkl`, so `<RUN>` must be `r06_paper` (or copy the fits).
Servers import `judge.py` at start; nothing already running changes.

## 9. Caveats and what is not verified

- Unverified: real GR00T GPU inference, websocket/simulator integration, closed-loop SR and realized cost. All checks
  are CPU with recorded store observations and a fake interceptor (MISS = recorded `a_inf[row]`); replays use an
  identity output transform (Q2 verified the production GR00T output adapter and original-wire tail reuse; the plugin
  selftests here check the wire tail equals the shifted MISS wire chunk). Replay MISS/vision shares are decisions on
  fixed recorded streams and overstate live rates (C10 l10/500: 0.2115 replay vs 0.0997 live), so compare GR00T B with
  pi0.5 C10 only under the same protocol (section 4.3).
- A ⊂ B is exact up to the pre-existing k-th-slot tie rule (section 3): deployed A (raw AWM `argpartition`) and every
  MixedJudge-based arm (K7/K10/C10/P1: lowest row among exactly tied distances) pick different members at float32
  distance ties across the 16th slot: 1 anchor (+ its blind tail) in 4 of 8 cells over all 500 recorded episodes, the
  same on pi0.5 A vs C10. Left unchanged because both the deployed A and C10 would change.
- The sign fix changes no decision inside B's own schedule in these replays (the stock branch runs only at step 0);
  it corrects K7's all-vision regime and early back-to-back vision decisions and removes the reason for the refusal.
  Re-derivation supports `burst <= 1` only (refused otherwise).
- GR00T H=16: rows 10..15 of policy and cache chunks are never executed. Action dims 7..31 never enter a guard; policy
  tails copy all 32 columns verbatim (checked).
- `arms_rep.json` copies source rows verbatim, including original absolute fit paths (shared read-only files, hashes
  above) and, for `r4b3_p_sp_500_tail1uc`, the absence of `cost_ledger` (collect.py still computes the ledger because
  its rows carry `vision`).
- Concurrency timing ratios are fake-policy CPU numbers under shared load, informational only.

## 10. Exact commands run (repository root, in this order)

```bash
P="taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python"
D=exp/offline_search/rounds/r06/p1_groot_commit
$P $D/make_arms.py                                        # arms_p1.json, arms_rep.json, prefit_commands.json, prefit.sh
bash $D/prefit.sh                                         # 4 fits -> r06_paper/fits (refuses existing artifacts)
$P $D/run_replays.py --root /tmp/p1_runs/matrix --groups NPG      # 40 replay jobs, 7 workers
$P $D/method_tests.py refusals resign lifecycle ; $P $D/method_tests.py fits ; $P $D/method_tests.py mirror
$P $D/plugin_selftests.py --out /tmp/p1_plugin_selftests
$P $D/concurrency_test.py --out /tmp/p1_concurrency && cp /tmp/p1_concurrency/summary.json $D/results/concurrency_summary.json
$P $D/run_replays.py --root /tmp/p1_runs/matrix --groups T --parallel 4   # tie-rule control, 8 jobs
$P $D/validate_arms.py                                    # emits into /tmp/p1_arm_validation (must not exist)
$P $D/analyze.py --root /tmp/p1_runs/matrix               # -> results/replay_checks.json
$P $D/allvision_rates.py --parallel 6                     # -> results/allvision_rates.json
$P $D/summarize.py
```
Reruns need fresh output roots (`replay.py`, `plugin_selftests.py`, `validate_arms.py`, the concurrency workers and
`prefit.sh` refuse existing outputs). A first `analyze.py` run failed on the then exact-equality nesting assertion
(pi05 l10/500); that led to the tie audit and the T control above, and the final run passed. Raw replay logs
(6.4 GB) stay in `/tmp/p1_runs/matrix`.
