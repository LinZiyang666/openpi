# R3 family H3 — `MixedJudge`: judge signals for a mixed HIT/MISS cache

A wrapper around any G3-contract base selector (default AWM `{"lib": "current", "kref": 5}` = the CL2 selector; H1's
`AWM3` keeps the contract, see "Wrapping AWM3") that leaves the served action untouched and adds what the server
plugin (H2) needs to hand control to the policy before the cache walks into a deadlock:

* `Result.confidence` — V7 `DriftCalibratedConfidence` on the base (r02 `g3_recovery/wrappers.py`, subclassed):
  `-pred_err + 1e-6 * zsum`, library-LOEO isotonic calibration (deployable, no trace data). Higher = HIT side of the
  plugin's `threshold` / `quantile` controller.
* `Result.extras["os_force_miss"]` ∈ {0, 1} with `os_reason` ∈ {0 none, 1 stuck, 2 terminal-with-closed-gripper,
  3 overtime∧lag, 4 no-progress, 5 dispersion event, 6 ambiguous gripper change, 7 burst continuation}; a forced
  decision is a MISS in the plugin's `threshold` / `quantile` / `guard_only` modes.

Files: `judge.py` (the method), `batch.json` (6 offline variants, R2 format), `arms_mx.json` (10 mixed closed-loop arm
rows, `emit_arms.py` spec), `tools/` (`contract_check_mx.py`, `replay_cl.py`, `tables_ac.py`, `make_arms.py`,
`run_all.sh`, `post_chain.sh`). Arrays / run outputs (not in the repo):
`/home/weiland/trace_runs/offline_search_store/derived/r03/h3_judge/` (`runs/<name>/<cell>.npz|json`, `fits/*.pkl`,
`replay/<arm>.json` + `replay_cl.md`, `tables/tables_ac.{json,md}`, `cl/` selftest + verify_logs).

## Algorithms as implemented (`judge.py`)

Per decision, `MixedJudge.query(q)` first runs the parent `DriftCalibratedConfidence.query(q)` (blend off, recovery
off): `rows, S, aux = base.os_score_all(q)`, top-`synth_k` by `S`, kernel weights `exp(-(S_0 - S_i) / synth_T)`,
`action = base.os_synth(q, rows, w)` — bit-identical to `base.query()` (`tools/contract_check_mx.py`: `topk_agree 1.0`,
`synth_bitexact 1.0`, `synth_max_abs_diff 0.0` on π0.5-spatial-cache and GR00T-l10-inf, 12 episodes each). Then:

**Confidence** = the parent's V7 value (`os_conf_raw`), lowered by `ret_margin` in the return phase (below).

**Guards** (`guards=True`; A-P4 trap score components + B-F5, V6 detector with `ot_still`):

| reason | condition | inputs |
|---|---|---|
| 1 stuck | `stuck_n >= stuck_thr` (2); `stuck_n` = consecutive decisions with state motion `|rs_t - rs_{t-1}|[:8] < m_thr` (library 10th pct of per-decision motion) **and** min-camera task-centred cosine `cos(key_t - m_task, key_{t-1} - m_task) >= c_thr` (library 95th pct) | `q.rs`, `q.key_*` vs `q.hist_*` |
| 2 terminal | the base's top-1 is the last row of its library episode (`next < 0`) **and** the previously executed gripper is closed: `q.prev_a_exec[4, 6] >= 0` | top-1, `q.prev_a_exec` |
| 3 overtime | `step / median library ep_len(task) > 1` **and** `lag = step - mean library step of the top-5 > lag_thr` (5) **and** `stuck_n >= 1` | top-5, `stuck_n` |
| 4 no-progress | the top-1's library progress did not advance over the last `noprog_n` (3) decisions: every consecutive difference, in library steps `(prog_t - prog_{t-1}) * (ep_len(top1_t) - 1)`, is `<= prog_eps` (0.5 step) | top-1 progress memo |

**Early events** (`events` ⊆ {`disp`, `grip`}; C-P3):

| reason | condition |
|---|---|
| 5 dispersion | `disp >= disp_thr`; `disp` = mean pairwise RMS of the top-5 heads (`[:5, :7]`, σ units) = V7's `disp` feature; `disp_thr` = `disp_abs` (default **1.0 σ**) or, with `disp_q`, that quantile of the library LOEO pseudo-query dispersions |
| 6 gripper | the served action's step-0 gripper sign ≠ the previously executed sign `sign(q.prev_a_exec[4, 6])` **and** `|vote| < vote_thr` (.8), `vote = Σ_i wn_i sign(g_i[0])` over the **exact** kernel members (all 16 for AWM) with the base's normalized kernel weights |

**Burst / return** (`burst` = 2 in the C-P3 variant; 0 = off): a guard / event at step `s` forces MISS at `s` and at
`s+1 .. s+burst-1` (reason 7, `os_phase = 1`). Then the *return phase* (`os_phase = 2`): the confidence is reported
as `conf - ret_margin` (0.1 in pred_err units — the plugin's τ is unknown to the method, so the stricter return
threshold is implemented as a confidence penalty) until `ret_n` (1) HITs have been observed in `q.hist_hit` after the
burst, or `ret_hold` (4) decisions have elapsed. A new guard / event restarts the burst. "Ambiguity cleared" = no
event fires (an event is itself a forced MISS). In the pure-cache offline cells (`hist_hit` all 1) the return phase
lasts exactly one decision; in the inf cells (all 0) `ret_hold` decisions; in the mixed loop until the first HIT.

**Precedence / extras.** Every condition is evaluated; `os_flags` is the bitmask (bit r−1 ↔ reason r), `os_reason`
the lowest firing code, then 7 if only the burst holds. Extras, in this order (the plugin log keeps the first 24
scalars): `os_force_miss, os_reason, os_flags, os_phase, os_conf_raw, pred_err, zsum, regime, stuck_n, overtime, lag,
term1, gexec, gprop, vote, disp, dnn, vis, top1_prog, noprog_n, burst_left, motion, vself` (NaN-valued ones dropped).

**Mixed-mode correctness.** Everything that depends on what was executed comes from the QueryView: `gexec` from
`q.prev_a_exec`, `stuck_n` from `q.rs` / `q.key_*` vs `q.hist_*`, the burst / return state from `q.hist_hit`. The two
per-episode memos (`top-1 progress per step` for guard 4, `own force flag per step` for the burst window) are
functions of the earlier QueryViews only (`os_score_all` is stateless and the QueryView at step `s < t` is a prefix
of `q.hist_*`), i.e. equivalent to re-scoring the history; they are indexed by step and dropped / NaN-padded when the
step sequence has a gap, never assumed executed. The parent's anchor is `"self"` (only used by its unused
`terminal`/blend bookkeeping; guard 2 uses the current top-1 instead).

**Fit.** Parent fit (base fit → library tables → V6 thresholds `m_thr`, `c_thr`, `med_len` → V7 LOEO calibration:
≤ 3000 stale + 3000 fresh + step-0 pseudo-queries). The judge adds `disp_thr` and a quantile grid of the pseudo-query
dispersions (`disp_qgrid`, diagnostics). Nothing per entry: bytes/entry = base + 46 (V7 tables).

**Name**: `MXJ_g<0|1>_ev<0|D|G|DG>[_b<burst>_rm<ret_margin>[_rn<ret_n>][_rh<ret_hold>]][_da<disp_abs>|_dq<disp_q>]
[_vt<vote_thr>][_sn<stuck_thr>][_lg<lag_thr>][_np<noprog_n>][_pe<prog_eps>]__<base name>`.

**Wrapping AWM3** (H1): `{"base": "exp/offline_search/rounds/r03/h1_trap/awm3.py:AWM3", "base_kwargs": {...}}` —
see the smoke section for the check run when `awm3.py` landed.

## Variants (`batch.json`, all 8 cells)

| name | what | closed-loop arm |
|---|---|---|
| `MXJ_g0_ev0__AWM_joint_cur_fcur_kr5` | V7-only (B-P1 confidence, no flags) | — |
| `MXJ_g1_ev0__AWM_joint_cur_fcur_kr5` | V7 + guards 1–4 | MX-AWM-h70 / h50, **MX-G** (`guard_only`: same fit, the plugin ignores the confidence) |
| `MXJ_g1_evDG_b2_rm0p1__AWM_joint_cur_fcur_kr5` | V7 + guards + events (disp 1.0 σ, grip .8) + burst 2 (C-P3 full) | MX-EV-h70 / h50 |
| `MXJ_g0_evDG_b2_rm0p1__AWM_joint_cur_fcur_kr5` | events-only + burst 2 | — |
| `MXJ_g1_ev0__AWM_joint_big_fbig`, `MXJ_g1_evDG_b2_rm0p1__AWM_joint_big_fbig` | 10× library twins (library-scale statement) | — |

Offline err of every variant equals the base's (served action identical); only confidence / flags differ. MX-G needs
no separate offline run (identical outputs to `MXJ_g1_ev0`).

## Smoke / verification (all PASS)

| check | result |
|---|---|
| `harness.smoke`, full variant `MXJ_g1_evDG_b2_rm0p1__AWM_joint_cur_fcur_kr5`, 5 episodes each | π0.5-sp cache: 149 dec, err .6725 (B0 .7679), AURC .3725, fit 5.8 s, 2.80 ms/q · π0.5-l10 inf: 306 dec, err .3794 (B0 .6804), AURC .2594, fit 15.4 s, 2.05 ms/q · GR00T-l10 cache: 394 dec, err .6862 (B0 .762), AURC .4622, fit 15.5 s, 1.87 ms/q · GR00T-sp inf: 101 dec, err .3146 (B0 .4807), AURC .2487, fit 6.0 s, 3.03 ms/q (4 smokes in parallel; prev_hit / extras / leak / static checks PASS) |
| `harness.smoke`, `MixedJudge(AWM3 prior_alpha=1, grip_commit, term_guard)` = `MXJ_g1_evDG_b2_rm0p1__AWM3_joint_cur_fbig_kr5_gc_tg`, GR00T-l10 inf | 293 dec, err .3881 (B0 .6053), AURC .2665, fit 41.9 s (borrowed-prior fit), 3.07 ms/q, bytes/entry 627 |
| `tools/contract_check_mx.py` (served action / top-k / library vs the base; confidence vs plain V7) | AWM base, π0.5-sp cache 365 dec and GR00T-l10 inf 598 dec: topk 1.0, synth bit-exact 1.0, max diff 0.0, `os_conf_raw` == V7 1.0, return penalty exact. AWM3 base (prior 1 + grip_commit + term_guard), π0.5-sp cache 365 dec: same, all 1.0 / 0.0 |
| whole-cell identity with the base run (`AWM_joint_cur_fcur_kr5`, harness run, 8 cells, 189k decisions) | err / synth arrays identical in 7 cells; GR00T-l10 cache: 1 of 39,669 decisions differs (max \|Δa\| .0039, Δerr .0008): rows 121 / 261 tie exactly at the 16-member boundary (dt 30.40858268737793), AWM's own `argpartition` keeps row 261, the wrapper's stable order the lower row — base-side tie nondeterminism, top-10 identical |
| `closed_loop/selftest.py` pure-cache (real per-connection stack, recorded keys) + `verify_logs.py` (offline replay of the logged inputs, every decision) | π0.5-sp cache 351 dec / GR00T-l10 cache 887 dec: exec_ok 100 %, winner == log, online == offline **bit for bit** on topk / scores / conf / synth / extras (incl. the os_* flags and the burst state), executed == selected 351/351, 887/887; q_us p50 2.5 / 3.0 ms |
| `closed_loop/selftest.py --judge guard_only --replay-cell pi05_spatial_inf` (H2's mixed mode driven by THIS wrapper's flags) | 221 dec = 193 HIT / 28 MISS (force:2 9, :4 5, :5 2, :6 4, :7 8; sequences MM 15, MH 7), client verdict == log, exec_ok on every HIT / None on every MISS, MISS executed == policy chunk 28/28, `bookkeeping_ok`, verdict / run violations 0, offline mini-store equality 1.0 on topk / scores / conf / lib / synth / extras → the `hist_hit`-derived burst / return state is reproduced offline through HIT→MISS→HIT |

## (a) Offline harness, all 8 cells (`tools/run_all.sh` → `tools/tables_ac.py`; full table `derived/.../tables/tables_ac.md`)

Cells π0.5-sp / π0.5-l10 / GR00T-sp / GR00T-l10. err = base's (identical). stale = cache cells step ≥ 1, fresh = inf
cells step ≥ 1. Accepted err at target h uses B §F4's mixture (after-HIT states weighted h, after-MISS 1−h, forced
decisions = −∞ confidence, τ = the weighted (1−h)-quantile). "capped" = the forced share alone exceeds the 1−h MISS
budget, the fixed point is −∞ (everything unforced is accepted).

| variant | AURC stale / fresh / pooled(p=.5) | forced stale / fresh | err forced vs unforced (stale) | accepted err h=.5 (τ pred_err) | accepted err h=.7 (τ) |
|---|---|---|---|---|---|
| V7-only `MXJ_g0_ev0` | .365 .339 .343 .379 / .227 .226 .239 .252 / .289 .277 .285 .306 | 0 | – | .269 .259 .282 .292 (.341 .330 .358 .377) | .357 .341 .337 .368 (.516 .516 .444 .522) |
| V7+guards `MXJ_g1_ev0` | same AURC | .339 .378 .248 .408 / .100 .200 .089 .210 | .887/.437 .659/.430 .884/.397 .616/.448 | .273 .266 .283 .298 (.346 .354 .363 .406) | .374 .384 .339 .399 (.625, capped, .463, capped) |
| full `MXJ_g1_evDG_b2_rm0p1` | .365 .340 .344 .380 / .230 .232 .242 .260 / .291 .280 .287 .311 | .410 .490 .337 .566 / .169 .293 .148 .306 | .874/.392 .657/.382 .801/.374 .612/.392 | .278 .275 .290 .309 (.353 .428 .374 .504) | .363 .347 .346 .357 (capped, capped, .595, capped) |
| events-only `MXJ_g0_evDG_b2_rm0p1` | ≈ V7-only | .197 .200 .126 .252 / .044 .027 .036 .031 | .935/.505 .698/.471 .721/.488 .650/.471 | .270 .260 .283 .290 | .361 .347 .339 .374 |
| 10×: V7+guards `MXJ_g1_ev0__AWM_joint_big_fbig` | .303 .283 .296 .325 / .177 .201 .176 .204 / .231 .237 .224 .253 | .331 .354 .212 .359 / .087 .145 .069 .162 | .786/.372 .544/.375 .724/.360 .528/.400 | .215 .223 .219 .236 (.298 .309 .310 .339) | .299 .330 .277 .345 |
| 10×: full | .303 .285 .297 .327 / .179 .205 .178 .210 | .403 .477 .312 .529 / .121 .225 .114 .239 | .769/.334 .550/.330 .673/.330 .529/.353 | .217 .228 .223 .242 | .299 .301 .282 .308 |

The V7-only rows reproduce ideation B's F3 / F4 exactly (stale AURC .365/.339/.343/.379, τ(h=.5) .34/.33/.36/.38,
accepted err .27/.26/.28/.29). Fresh-regime forced decisions are also worse than the rest (V7+guards: .460/.390/.481/.403
vs .327/.301/.331/.314). The AURC differences between V7-only and the burst variants come only from the return-phase
penalty (−.1 on ≤ 5 % of decisions).

Forced share per reason (rate, err of those decisions), current library, stale regime:

| cell | stuck | terminal | overtime | noprog | disp (1.0 σ) | grip (\|v\|<.8) | burst |
|---|---|---|---|---|---|---|---|
| π0.5-sp | .108 (.83) | .098 (.87) | .013 (.92) | .121 (.95) | .030 (.91) | .007 (.74) | .034 (.74) |
| π0.5-l10 | .181 (.59) | .036 (.72) | .008 (.74) | .154 (.72) | .028 (.83) | .019 (.57) | .064 (.60) |
| GR00T-sp | .133 (.93) | .031 (.94) | .010 (.82) | .074 (.79) | .023 (.71) | .017 (.55) | .050 (.51) |
| GR00T-l10 | .161 (.60) | .027 (.65) | .011 (.67) | .209 (.62) | .044 (.76) | .031 (.55) | .084 (.54) |

(lowest firing code wins, so `disp` alone — events-only variant — fires on .135/.097/.055/.099 of stale decisions with err
.97/.76/.81/.75.) Fresh regime: stuck .007/.074/.035/.070, terminal .048/.011/.005/.013, noprog .041/.105/.045/.123
(err .37–.48 vs unforced .30–.33), disp ≤ .008, grip .009–.015, burst .04–.08. Step 0 is never forced (no executed
gripper, no history; disp at step 0 stayed below 1.0 σ everywhere).

**Cap.** The guard set (with no-progress) forces .34–.41 of stale decisions at the current library, so at h = .7 the
controller saturates on π0.5-l10 and GR00T-l10 (V7+guards) and on 3 of 4 cells for the full variant: realized after-HIT
acceptance .51–.62 instead of .7. At h = .5 every variant has budget left (threshold-MISS share .11–.31 stale). B's F5
guard set alone (stuck | terminal | overtime∧lag) is .224 on π0.5-l10; the extra is guard 4 (`noprog_n=3`, the knob —
n=4 halves its volume but loses the early warning, see (b)).

## (b) Shadow triggers replayed on the R2 closed-loop logs (`tools/replay_cl.py`; full tables `derived/.../replay/replay_cl.md`)

Protocol (ideation C): per arm, thresholds on inits 0–24 per task, everything measured on inits 25–49 (250 episodes).
Spell = run of ≥ 3 identical top-1 picks; **early** = first flag ≤ first spell start + 1; **lead** = (first spell start −
first flag) in decisions among failed episodes touched (positive = before the spell); "successful touched (at)" = share
of successful episodes with a flag (median position of the first flag as a fraction of the episode). Flag rates are
shadow flags on the fixed pure-cache trajectories (counterfactual after the first would-be MISS); `v7@r` / `awm_conf@r`
= confidence below its train-init r-quantile (r = .1 / .3 / .5 ≈ h .9 / .7 / .5).

**π0.5-spatial CL2** (`oscl50_p_sp_cl2`, SR .800; test 250 eps, 57 failed; failed episodes hold 38 % of decisions)

| variant | flag rate | flags in failed eps | failed early | failed ever | successful touched (at) | lead med | lead ≥ 0 |
|---|---|---|---|---|---|---|---|
| awm_conf@.1 (C's raw-conf gate) | .104 | 1.00 | .02 | .79 | .00 | −11 | .00 |
| v7@.1 | .122 | .99 | .02 | 1.00 | .01 (.92) | −10 | .00 |
| v7@.3 | .342 | .76 | .93 | 1.00 | .71 (.56) | +11 | .91 |
| guards (1–4) | .266 | .82 | .74 | 1.00 | .80 (1.00) | 0 | .72 |
| · stuck (proxy) | .053 | .98 | .02 | .54 | .01 | −8 | .00 |
| · terminal∧closed | .178 | .83 | .46 | .91 | .73 (1.00) | −1.5 | .50 |
| · overtime∧lag∧stuck | .066 | .98 | .00 | .82 | .01 | −11 | .00 |
| · no-progress | .224 | .90 | .56 | 1.00 | .25 (.62) | 0 | .51 |
| ev_disp (1.0 σ) | .128 | .83 | .56 | .89 | .28 (.40) | +2 | .59 |
| ev_grip | .043 | .79 | .35 | .56 | .15 (.43) | +3 | .59 |
| events | .140 | .81 | .67 | .95 | .35 (.42) | +4.5 | .67 |
| guards+events+burst (full) | .313 | .78 | .82 | 1.00 | .82 (.58) | +6 | .79 |
| v7+guards@.3 | .372 | .72 | .93 | 1.00 | .90 (.65) | +11 | .93 |
| v7+guards+events@.3 | .389 | .70 | .95 | 1.00 | .90 (.50) | +11 | .95 |

**π0.5-l10 CL2** (`oscl50_p_l10_cl2`, SR .630; test 250 eps, 100 failed; failed episodes hold 57 % of decisions)

| variant | flag rate | flags in failed eps | failed early | failed ever | successful touched (at) | lead med | lead ≥ 0 |
|---|---|---|---|---|---|---|---|
| awm_conf@.1 | .132 | 1.00 | .06 | .70 | .02 (.63) | −37 | .07 |
| v7@.1 | .123 | 1.00 | .02 | .85 | .01 (.97) | −48 | .00 |
| v7@.3 | .338 | .88 | .14 | 1.00 | .33 (.63) | −10 | .13 |
| v7@.5 | .530 | .74 | .52 | 1.00 | .97 (.46) | −1 | .49 |
| guards (1–4) | .378 | .81 | .68 | 1.00 | .89 (.48) | +2 | .61 |
| · stuck (proxy) | .151 | .78 | .02 | .85 | .21 (.57) | −11 | .01 |
| · terminal∧closed | .052 | .84 | .05 | .35 | .41 (.98) | −30 | .14 |
| · overtime∧lag∧stuck | .126 | .82 | .00 | .79 | .14 (.76) | −41 | .00 |
| · no-progress | .359 | .83 | .66 | 1.00 | .83 (.46) | +2 | .60 |
| ev_disp (1.0 σ) | .097 | .87 | .22 | .83 | .42 (.64) | −21.5 | .26 |
| ev_grip | .070 | .75 | .33 | .93 | .59 (.51) | −8 | .28 |
| events | .153 | .82 | .43 | .97 | .73 (.51) | −5 | .39 |
| guards+events+burst (full) | .423 | .78 | .75 | 1.00 | .98 (.42) | +5 | .67 |
| v7+guards@.3 | .443 | .82 | .69 | 1.00 | .92 (.48) | +2 | .62 |
| v7+guards+events@.3 | .469 | .79 | .75 | 1.00 | .98 (.42) | +5 | .67 |

**GR00T CL2** (`oscl50_g_sp_cl2` SR .888, 27 failed of 250 test, failed hold 19 % of decisions; `oscl50_g_l10_cl2` SR .552,
117 failed, 63 %): sp — v7@.1 early .30 / ever 1.0 / succ .05; guards .146 flags, early .67, succ .31 (stuck and overtime
never fire: the still-proxy is off-scale for GR00T); events .064, early .52, succ .43; full .183, early .78, succ .58.
l10 — v7@.1 early .00 / ever .79; v7@.3 early .11; guards .444 flags (noprog .438), early .66, succ .92 (at .26); events
.233 (disp .121, grip .125), early .58; full .528, early .78, ever 1.0, succ .98. **CL3 arms** (V6 recovery over AWM,
stuck_n exact there): π0.5-sp guards .242 / early .92 / succ .83 (stuck alone .051, early .66); π0.5-l10 guards .343 /
early .73; GR00T-sp guards .151 / early 1.00 — recovery breaks the identical-pick spells, so "early" is easier there.

Reading: (i) the confidence alone is a late detector at a 10 % budget (early .02–.06 on π0.5, as C found) and V7 does
not change that — V7 only finds more of the failing episodes eventually (ever 1.0 / .85 vs .79 / .70). (ii) The guards
are early because of no-progress (lead 0 / +2 = at the spell start) and, on spatial, the terminal∧closed clause (early
.46); stuck / overtime are late by construction (lead −8…−41). (iii) The events are the low-budget early signal: on
spatial 14 % of decisions reach 67 % of the failures early while touching 35 % of the successes; on l10 they are weaker
(early .43, successes .73). (iv) At a matched 30–38 % budget on l10 the guards beat the threshold by far (early .68 vs
.14, lead +2 vs −10); on spatial a 30 % confidence budget is already early (.93) because spatial failures are late
terminal traps and 30 % covers the second half of every failing episode. (v) The full variant reaches 82 % / 75 % of
the failures early at 31 % / 42 % of the decisions, but touches 82 % / 98 % of the successful episodes (first flag at
58 % / 42 % of the episode) — that is the inference cost the closed-loop SR-vs-IR curve has to pay for.

## (c) Threshold-init table (`--os-judge quantile:h:1000:tau0`; τ0 in confidence units = −pred_err; from (a))

| variant | cell | h | τ0 | realized acceptance stale / fresh | forced share stale / fresh | threshold-MISS share stale / fresh | accepted err / bad / grip_mis |
|---|---|---|---|---|---|---|---|
| V7+guards | π0.5-sp | .5 | −0.3463 | .35 / .65 | .339 / .100 | .314 / .248 | .273 / .340 / .012 |
| V7+guards | π0.5-sp | .7 | −0.6251 | .62 / .88 | .339 / .100 | .039 / .017 | .374 / .486 / .047 |
| V7+guards | π0.5-l10 | .5 | −0.3540 | .35 / .65 | .378 / .200 | .269 / .152 | .266 / .289 / .028 |
| V7+guards | π0.5-l10 | .7 | capped (arm starts at V7-only −0.5157) | .62 / .80 | .378 / .200 | 0 / 0 | .384 / .429 / .069 |
| V7+guards | GR00T-sp | .5 / .7 | −0.3631 / −0.4627 | .31/.69 · .65/.82 | .248 / .089 | .445/.218 · .104/.091 | .283 · .339 |
| V7+guards | GR00T-l10 | .5 / .7 | −0.4062 / capped | .33/.67 · .59/.79 | .408 / .210 | .262/.119 · 0/0 | .298 · .399 |
| full (EV) | π0.5-sp | .5 | −0.3532 | .40 / .60 | .410 / .169 | .190 / .230 | .278 / .356 / .012 |
| full (EV) | π0.5-sp | .7 | capped (starts at −0.5158) | .59 / .83 | .410 / .169 | 0 / 0 | .363 / .473 / .043 |
| full (EV) | π0.5-l10 | .5 | −0.4284 | .40 / .60 | .490 / .293 | .111 / .106 | .275 / .304 / .025 |
| full (EV) | π0.5-l10 | .7 | capped (starts at −0.5157) | .51 / .71 | .490 / .293 | 0 / 0 | .347 / .390 / .046 |
| full (EV) | GR00T-sp | .5 / .7 | −0.3744 / −0.5954 | .38/.62 · .64/.83 | .337 / .148 | .283/.232 · .019/.020 | .290 · .346 |
| full (EV) | GR00T-l10 | .5 / .7 | −0.5043 / capped | .37/.63 · .43/.69 | .566 / .306 | .064/.064 · 0/0 | .309 · .357 |
| V7-only (reference) | sp / l10 / g-sp / g-l10 | .5 | −.3415 / −.3299 / −.3577 / −.3770 | .33/.67 .31/.69 .28/.72 .24/.76 | 0 | all | .269 .259 .282 .292 |
| V7-only (reference) | sp / l10 / g-sp / g-l10 | .7 | −.5158 / −.5157 / −.4441 / −.5218 | .60/.93 .61/.91 .63/.86 .60/.94 | 0 | all | .357 .341 .337 .368 |
| 10× V7+guards | sp / l10 / g-sp / g-l10 | .5 | −.2983 / −.3090 / −.3103 / −.3387 | .34/.66 .36/.64 .30/.70 .29/.71 | .331 .354 .212 .359 / .087 .145 .069 .162 | | .215 .223 .219 .236 |

Predicted forced-MISS share of all decisions in a mixed loop at target h: `h · forced_stale + (1−h) · forced_fresh`
(pessimistic, F9: after-HIT states of the cache cells are ends of long HIT runs) — V7+guards π0.5-sp .22 (h=.5) / .27
(h=.7), π0.5-l10 .29 / .32; full π0.5-sp .29 / .34, π0.5-l10 .39 / .43. The plugin's controller is initialised from τ0
and holds the realized rate (B F8: offline τ overshoot by 10–15 pp in AWM's own loop is expected).

## Closed-loop arms (`arms_mx.json`, 10 rows; `tools/make_arms.py` regenerates from `tables/tables_ac.json`)

`r3mx_p_{sp,l10}_{awm_h70, ev_h70, g, awm_h50, ev_h50}` — π0.5 spatial / l10, 50-episode library, `"full_model": true`,
`plugin_args = ["--os-fit-artifact", "<RUN>/fits/<name>.pkl", "--os-judge", <spec>]` with `quantile:0.7:1000:<τ0>` /
`quantile:0.5:1000:<τ0>` (τ0 from (c); capped cells start at the V7-only τ0 and drift) or `guard_only`. MX-AWM = kwargs
`{guards: true, events: "none"}`, MX-EV = `{guards: true, events: "all", burst: 2, ret_margin: 0.1}`, MX-G = MX-AWM's
wrapper under `guard_only`. No `--os-judge-burst` (the wrapper owns the burst, reason 7). Prefit: `python -m
exp.offline_search.closed_loop.plugin --os-method judge.py:MixedJudge --os-kwargs <kwargs> --os-cell pi05_<suite>_cache
--os-log-dir <RUN>/fits --os-fit-artifact <RUN>/fits/<name>.pkl` with `OPENBLAS_NUM_THREADS=1`; walls / sizes below.

## Cost, library scale, three layers

* **ms/query** (harness timing pass, single thread, one process): base AWM 1.21–1.27 ms → MixedJudge 1.57–1.72 ms on the
  current library (+0.4 ms: V7 features, detector, judge logic ≈ µs); 10× 1.7–3.6 ms (l10). Plugin selftest q_us p50
  2.5 / 3.0 ms (π0.5-sp / GR00T-l10, real stack).
* **fit**: 3.3 s (sp) / 9.0–9.4 s (l10) current (base 0.1–0.2 s with the cached PCA; the rest is V7's LOEO calibration);
  prefit pickles (4 in parallel): 5.9 / 13.0 / 6.3 / 12.6 s current, 30.6 / 62.4 / 34.4 / 62.9 s 10×.
* **pickle** (`--os-fit-artifact`, π0.5-sp / π0.5-l10 / GR00T-sp / GR00T-l10): current **26.0 / 32.6 / 27.9 / 36.7 MB**,
  10× **66.1 / 141.2 / 87.6 / 187.4 MB**; deployed pkl 431 / 1103 / 429 / 1068 MB. (The pickle holds the base's action
  table twice — base + wrapper tables — plus σ-scaled heads; the judge itself adds < 10 KB.)
* **library scale**: entries current 1018 / 2640 / 1063 / 2645 (49–50 episodes), 10× 10909 / 29472 / 11751 / 29631
  (500); bytes/entry 626 = AWM 580 + V7 46 (AWM3 base: 627); per-entry total current 0.64 / 1.65 / 0.67 / 1.66 MB, 10×
  6.8 / 18.4 / 7.4 / 18.5 MB; fixed per suite 17.0 MB PCA bases + 2.6 MB task-mean keys + ~0.15 MB/task metric + < 10 KB
  V7 maps / thresholds. The judge adds no per-entry bytes.
* **three layers**: the judge serves the base's action, so synthesis effect 0 and method effect at fixed library 0 (err
  identical to `AWM_joint_cur_fcur_kr5`, the one tie decision above); library effect = AWM's (10× twin: stale AURC .303
  .283 .296 .325 vs .365 .339 .343 .379, accepted err at h=.5 .215–.236 vs .273–.298, forced share similar). No borrowed
  information in the default fit (`os_fit_library = current`); the 10× twin fits on the big library (candidates big).
  The fourth layer (control effect of the handoff) is what the arms measure.

## Caveats

* **Log replay is approximate where marked**: gripper vote / executed sign from the logged 10 of 16 kernel members;
  `stuck_n` in CL2 from AWM's `still` extra (> 1.98, C's proxy) — exact only in CL3; the V7 confidence lacks `vis`
  (never logged) and, in CL3, `dnn` (stale AURC cost .005 / .005–.02, corr(z) .99 / .93–.99, table in `tables_ac.md`);
  CL3 weights ignore the V6 blend override. disp, lag, overtime, terminal and no-progress are exact.
* Shadow flags on fixed trajectories: after the first would-be MISS everything is counterfactual; "early" saturates at
  high flag rates (compare at matched budgets); no rescue probability is implied.
* `disp_abs = 1.0 σ` was chosen by inspecting the library-quantile sweep on the CL2 **train** inits (0–24): the library
  LOEO quantile does not transfer (q.9 → 25–30 % of closed-loop decisions), 1.0 σ ≈ q.99 on π0.5 / q.995 on GR00T. A
  deployable constant, but tuned on the R2 logs' train half.
* The no-progress guard is the early component and the volume driver: 22 % / 36 % of CL2 decisions on π0.5 (44 % on
  GR00T-l10), touching 25 % / 83 % of the successful episodes (l10: first flag at 46 % of the episode). `noprog_n=4`
  cuts the volume to 17 % / 25 % but the early share falls to .25 / .12 (replay sweep in `replay_cl.md`). Because of it
  the guard set exceeds the 30 % MISS budget on l10 (cap above).
* The terminal∧closed guard fires on the last decision of most successful spatial episodes (73 % touched, first flag
  at 1.00 of the episode): harmless for SR, ≈ 1 forced MISS per successful episode.
* Per-episode memos (top-1 progress, own flags) are functions of the history (see "Mixed-mode correctness"); a
  recompute-from-`hist_*` self-check is not implemented, the verify_logs / mixed selftest equalities are the evidence.
* GR00T arms: only `oscl50_g_{sp,l10}_cl2` and `oscl50_g_sp_cl3` were complete at the time of the replay; rerun
  `tools/replay_cl.py` when the rest land (it picks up every arm with a `summary.json`).
