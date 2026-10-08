# R12 opus — PREDICTION (frozen 2026-10-03 12:15 CDT)

Written before I opened any R12 closed-loop result. At this time `os_closed_loop/r12_devlooks_*` roots exist (I saw
their names only); I have not read them and will not use them for anything in SPEC. Sources: `OFFLINE.md`,
`out/prediction_table.json`, `out/sim3_all.json`. "pp" = percentage points of SR. IR per decision slot, owner ledger.

## P0 — prices (REQUESTS 0, coordinator measurement on the pi0.5 serving path)
- Batched two-camera stage 1 (base + wrist in one embed call, dummy cached): **27–32 ms**, i.e. ≤ 1.25 × the
  wrist-only stage 1 (R7: 26.4 ms) with 80 % probability. Price ≈ **.066–.078** of a call (vs .152).
- GR00T stage 1 with one image ≥ 0.8 × with two images (both images already go through one `extract_feature` call).
- Consequence predicted: the Phase-0 rule adopts S and stops C on pi0.5; no GR00T look-half.

## P1 — S (batched camera stage 1), pi0.5, if built
| quantity | prediction |
|---|---|
| ΔIR per slot, every cell, knob off | **−.037 to −.043** (= (p_S − .152)·V/N, V/N ≈ .505) |
| ΔIR at knob P@.32 | same −.037 to −.043 (looks only; calls are looks) |
| ΔSR (true effect) | 0; observed per-cell differences within ±3 pp; pooled 5 cells within ±1 pp (80 %) |
| step-0 top-1 agreement with the per-camera path | ≥ .90 (bf16-level key changes, 0.15 %) |
| matched-SR total IR | equal to ΔIR (no SR to buy back) |

## P2 — C (wrist-first consistency cascade) if it is built anyway; pi0.5, knob off unless noted
Columns: one-camera share of non-step-0 looks (80 % interval), completion share, ΔIR on the owner table, ΔIR with a
full look priced .115 ("fair"), true ΔSR (pp; 80 % interval), matched-SR ΔIR (owner / fair; knob slope 2.9 pp per
.1 IR on 50-episode cells, 1 pp per .1 on denser cells).

| cell | τ | one-camera | completion | ΔIR owner | ΔIR fair | ΔSR | matched-SR ΔIR owner / fair |
|---|---|---|---|---|---|---|---|
| L10-50 | .150 | .42 [.30, .55] | .58 | −.031 [−.036, −.024] | −.012 | −0.1 [−0.6, +0.3] | −.027 / −.008 |
| Sp-50 | .150 | .45 [.32, .58] | .55 | −.029 [−.034, −.022] | −.011 | −0.3 [−0.8, +0.2] | −.020 / −.001 |
| L10-200 | .075 | .33 [.22, .45] | .67 | −.024 [−.029, −.018] | −.005 | −0.2 [−0.6, +0.2] | −.007 / +.012 |
| L10-500 | .047 | .16 [.08, .30] | .84 | −.021 [−.025, −.017] | −.002 | −0.1 [−0.3, +0.2] | −.014 / +.005 |
| Sp-500 | .047 | .15 [.08, .30] | .85 | −.023 [−.027, −.018] | −.004 | −0.1 [−0.4, +0.2] | −.009 / +.010 |
| L10-50, knob P@.32 | .150 | .32 [.20, .45] | .44 | −.025 [−.031, −.018] | −.011 | 0.0 [−0.5, +0.4] | −.026 / −.012 |
| Sp-50, knob P@.32 | .150 | .33 [.20, .46] | .39 | −.023 [−.029, −.016] | −.009 | −0.2 [−0.7, +0.3] | −.017 / −.004 |

- Guard rate change: within ±.02 of the same-batch baseline in every cell.
- Of the owner-table saving, 60–85 % comes from "wrist + completion (.115) < stock full look (.152)"; only
  −.004 to −.011 per slot is the one-camera looks themselves.
- Pooled over the five knob-off cells, observed paired ΔSR: −0.2 pp, 80 % interval [−1.2, +0.8].
- C retargeted to the baseline's IR at P@.32 (spends the saving on calls): ΔSR vs baseline P@.32 +0.5 pp
  (80 % [−1.0, +2.0]) on L10-50, +0.3 on Sp-50 — not distinguishable from 0 in one 500-episode run.
- Under S, C's additional saving: ≤ .005 per slot.

## P3 — H0 (wrist-only at every non-call look), pi0.5 knob off
| cell | one-camera share | ΔIR owner / fair | ΔSR (pp, 80 %) | matched-SR ΔIR owner / fair |
|---|---|---|---|---|
| L10-50 | .72 | −.032 / −.013 | −0.7 [−1.6, 0.0] | −.009 / +.010 |
| Sp-50 | .84 | −.041 / −.022 | −0.8 [−1.8, 0.0] | −.013 / +.006 |

If H0 loses less than 0.3 pp pooled, my offline→SR calibration (1 pp per .01 e10) is too pessimistic and more
aggressive look-half becomes worth testing; if it loses more than 1.6 pp, the cascade's predictions above are
optimistic by the same factor.

## P4 — GR00T (only if Phase 0 shows a one-camera price ≤ 0.6 × two-camera)
Retrieval cost of the cascade (dq): L10-50 +.0019, Sp-50 +.0018, L10-500 +.0005 → ΔSR −0.2 / −0.2 / −0.05 pp.
One-camera share .38 / .49 / .08; GR00T Spatial-50 guard rate drops by ~.03 under the wrist cache (fewer calls).

---

## Addendum A (2026-10-03 12:28 CDT) — after the B-pool dev recordings, before any test-A result

Read after the freeze above: `COORDINATOR.md` (prices) and the coordinator's dev look recordings
`r12_devlooks_50L/50S` (deployed five-layer system, knob off, two-camera serving, 100 B-pool dev episodes per arm;
L10-50 with the policy's shadow chunk at every look). Analysis: `tools/dev_validate.py`,
`out/dev_validate*.json`. Dev data is allowed for calibration (BRIEF §5); test A untouched.

**Checks of the frozen predictions**
- P0, GR00T: measured one-camera look .143–.150 vs full .148 (coordinator) — predicted "≥ 0.8 × two-camera": ✓.
  GR00T look-half is dead on the current stack.
- P0, pi0.5 batched two-camera stage 1: still unmeasured (only "wrist + completion ≈ .118" estimated).
- P2 one-camera share at the frozen τ(50) = .15, replayed on the recorded closed-loop looks: pi05 L10-50 **.55**
  (predicted .42 [.30, .55], at the edge), pi05 Sp-50 **.54** (.45 [.32, .58] ✓), groot Sp-50 **.60** (.49).
- My F replica reproduces the deployed top-1 on 100 % of regime-2 HIT looks.
- **Closed-loop retrieval cost at τ = .15 (pi05 L10-50, vs the shadow policy chunk): +.0035 ± .0009**, about 3× the
  frozen offline value (+.0012). Wrist-only at every look: +.0057 ± .0023 (offline +.0066, consistent). Cache-driven
  states sit close to the followed demonstrations, so the wrist view and the previous agentview agree more often and
  the test lets through more looks whose fresh agentview would have changed the answer.

**Revised tolerance (dev-calibrated, replaces τ for any C arm):** `τ'(N) = 0.10·√(50/N)` (50 → .10, 200 → .071,
500 → .032), i.e. the frozen schedule scaled by the factor that brings the closed-loop cost back to the ≤ .002 target
on the only cell with shadow chunks. Closed-loop dev numbers at τ' = .10: one-camera share .40 (pi05 L10-50), .40
(pi05 Sp-50), .40 (groot Sp-50); retrieval cost +.0014 ± .0006 (pi05 L10-50); τ = .075 → share .29, +.0010; τ = .05 →
.16, +.0000.

**Revised P2 (C with τ', pi0.5, knob off)** — one-camera share / ΔIR owner / ΔIR fair / true ΔSR:
- L10-50: .40 [.30, .50] / −.029 / −.010 / −0.15 pp [−0.6, +0.3]
- Sp-50: .40 [.30, .50] / −.029 / −.010 / −0.2 pp [−0.7, +0.3]
- L10-200, L10-500, Sp-500: share .25 / .10 / .10 (±.10), ΔIR owner −.025 / −.021 / −.021, fair −.006 / −.003 / −.003,
  ΔSR −0.1 pp [−0.5, +0.2] (no dense-cell shadow data yet; extrapolated).
- H0 unchanged (−0.6 to −0.7 pp on the 50-cells).
Everything else in P1–P4 stands.

---

## Section B (2026-10-03 13:11 CDT) — DIRECTIVE_2: compute-share prices, method M1 (SPEC §6)

The sections above stay as frozen. Written after reading DIRECTIVE_2 and the B-pool dev recordings (50L, 50S, 200,
400, knob), before any test-A result of any R12 method (none exists). Prices: one camera .076 (GR00T .074),
completion .076 (.074), full .152 (.148), call + .848 (.852). M1 constants τ = .10, m = 1.0 are dev-calibrated.
Proposal S (P0/P1 above) is withdrawn by the owner and is not predicted any further.

**Per cell, M1 vs the same-batch deployed system, knob off** (ΔIR per decision slot from the ledger; ΔSR = true
effect in pp; matched-SR ΔIR = ΔIR + (−ΔSR)/slope with the R11 periodic-knob slope of the cell; 80 % intervals):

| cell | one-camera share | ΔIR (vision / fewer calls) | ΔSR | matched-SR ΔIR | verdict |
|---|---|---|---|---|---|
| pi05 L10-50 | .23 [.13, .33] | −.011 [−.015, −.006] (−.009 / −.003) | −0.2 [−1.0, +0.3] | −.003 [−.010, +.010] | ≈ break-even |
| pi05 Sp-50 | .30 [.18, .40] | −.012 [−.016, −.007] | −0.2 [−0.8, +0.3] | −.008 [−.014, +.005] | small gain |
| pi05 Sp-200 | .45 [.30, .55] | −.018 [−.023, −.012] | −0.15 [−0.6, +0.2] | −.004 [−.015, +.040] | uncertain (flat knob) |
| pi05 L10-200 | .33 [.20, .45] | −.017 [−.022, −.010] | −0.5 [−1.2, 0.0] | +.020 [−.010, +.10] | loss |
| pi05 L10-400 | .33 [.20, .45] | −.018 [−.023, −.010] | −0.55 [−1.3, 0.0] | +.035 [0, +.12] | loss |
| groot L10-50 | .12 [.06, .20] | −.007 [−.010, −.004] | −0.3 [−1.0, +0.2] | +.004 [−.004, +.020] | loss/neutral |
| groot Sp-50 | .31 [.18, .40] | −.015 [−.020, −.009] | −0.4 [−1.2, +0.2] | +.02 [−.005, +.08] | loss |
| groot L10-400 | .16 [.08, .25] | −.010 [−.014, −.005] | −0.5 [−1.2, 0.0] | +.04 [0, +.12] | loss |
| either model, L10-50, knob .32 | ≤ .01 | −.002 (missed calls only) | ≈ 0 | ≈ 0 | nothing to save |

- GR00T: add ~3–4 % of one-camera looks with a different top-1 from the camera-alone encode (not in the dev replay).
- Pooled over the four Phase-B gain candidates (pi05 L10-50, pi05 Sp-50, pi05 Sp-200, groot Sp-50): ΔIR −.014,
  ΔSR −0.25 pp [−0.9, +0.4]; matched-SR ΔIR ≈ −.002, 80 % [−.012, +.015]. **I do not expect test A to show a
  saving distinguishable from zero.**
- Phase A (own closed loop): realised one-camera share within ±.10 of the table; call rate within −10 % of the
  baseline; first guard call on average < 1 decision later than baseline. If the own-loop guard instead behaves like
  the replay of wrist-only looks (calls missed ≥ 20 %), M1 will lose ≥ 1 pp and should be dropped.
- A method that never misses a call (m ≥ 2) keeps ≤ 7 % of looks one-camera: ΔIR −.002 to −.003, ΔSR ≈ 0.

---

## GUARD_PREDICTION (2026-10-03 16:14 CDT) — guard H2 and look-half M2 (GUARD.md §6–§7)

Earlier sections untouched. Written before any closed-loop run of H2 or M2 exists. θ_a, θ_b are dev-calibrated
per cell (labelled, `out/guard_m2.json`); the risk model is library-fitted. Compute-share prices.

**H2 replacing the no-progress guard, two-camera looks, knob off (vs same-batch deployed):**
- call rate within ±.03 of deployed on every cell; ΔIR within ±.015 per slot;
- share of calls in failing episodes higher by +.05 to +.15; first call in failing episodes on average ≤ 3 decisions
  later on L10 cells, not later on Spatial cells;
- ΔSR (true effect): +0.5 pp on 50-episode cells (80 % [−1.5, +2.5]); 0 [−1, +1] on 200/400/500 cells.

**M2 (H2 + look-half), knob off** — one-camera share / ΔIR per slot / ΔSR / matched-SR ΔIR:

| cell | share | ΔIR | ΔSR (pp, 80 %) | matched-SR ΔIR (central) |
|---|---|---|---|---|
| pi05 L10-50 | .35 [.25, .45] | −.013 [−.018, −.008] | +0.3 [−1.7, +2.3] | −.023 (−.013 at ΔSR 0) |
| pi05 Sp-50 | .35 [.25, .45] | −.013 [−.018, −.008] | +0.3 [−1.5, +2.0] | −.020 (−.013 at ΔSR 0) |
| pi05 L10-200 | .47 [.35, .55] | −.018 [−.023, −.012] | −0.1 [−1.2, +1.0] | −.008 (−.018 at ΔSR 0) |
| pi05 Sp-200 | .50 [.38, .60] | −.019 [−.024, −.013] | −0.1 [−1.0, +0.8] | −.009 (−.019 at ΔSR 0) |
| groot L10-50 | .23 [.15, .32] | −.008 [−.012, −.004] | +0.3 [−1.7, +2.3] | −.015 (−.008 at ΔSR 0) |
| groot Sp-50 | .37 [.25, .47] | −.014 [−.019, −.008] | +0.3 [−1.5, +2.0] | −.044 (−.014 at ΔSR 0) |
| groot L10-400/500 | .29 [.18, .40] | −.011 [−.016, −.006] | 0.0 [−1.2, +1.0] | −.011 (−.011 at ΔSR 0) |

**M2 with the periodic knob .32 (L10-50):** one-camera share .22 (π0.5) / .15 (GR00T); ΔIR −.009 / −.006 at ΔSR
0 [−2, +2] pp.

The retrieval part of ΔSR is −0.01 to −0.12 pp; the rest is the guard swap (sign uncertain; positive on targeting).
Matched-SR ΔIR = ΔIR − ΔSR/slope with the R11 periodic-knob slope of the cell (pp per unit IR: 29 pi05 L10-50, 43
pi05 Sp-50, 44 groot L10-50, about 10 for groot Sp-50 and the dense cells, where the knob buys little; that flat
slope is why a small SR change moves those cells a lot either way). At ΔSR = 0 the matched-SR saving equals the
vision saving. Pooled over the four 50-episode cells: ΔIR −.012; matched-SR ΔIR −.025 central, 80 % [−.06, +.03]
(the interval is the guard-swap uncertainty). The decisive risk is the open-loop
caveat: H2's calls were scored on the old guard's trajectories; if the pilot shows its first calls ≥ 3 decisions
later in failing episodes, expect ΔSR −1 to −3 pp instead and drop it.

## TEST_A_PREDICTION (frozen 2026-10-03 20:25 CDT; saved by the coordinator from the opus pilot analysis, full text in PILOT_ANALYSIS.md §5)


Before any R12 test-A result. A pool, 500 eps/arm, all arms of a cell on one host. Reference off levels (GR00T absolute numbers assume H100): R11 test A knob off pi05 L10-50 .844 @ .167; pi05 Sp-50 .922 @ .133; groot Sp-50 .892 @ .128; groot L10-50 .748 @ .209; periodic .32 .908 @ .306; R10 test A pi05 Sp-200 .994 @ .0985. Δ vs same-batch off arm; 80% intervals incl. run noise; Δ host-independent if hosts matched. If GR00T arms are served locally: groot Sp-50 SR −4 pp, IR +.014, calls +0.5; groot L10-50 IR +.005 ± .008, SR −2 ± 3 pp.

### pi05 L10-50
| arm | SR | IR | one-camera share | calls/ep | ΔIR | ΔSR (pp) |
|---|---|---|---|---|---|---|
| off | .840 [.815, .865] | .167 [.162, .172] | 0 | 6.3 [5.8, 6.9] | — | — |
| M1 | .840 [.805, .875] | .157 [.151, .163] | .235 [.20, .27] | 6.2 [5.6, 6.8] | −.010 [−.014, −.006] | 0.0 [−2.8, +2.8] |
| H2 | .835 [.80, .87] | .158 [.151, .165] | 0 | 5.65 [5.0, 6.3] | −.009 [−.015, −.003] | −0.5 [−3.3, +2.3] |
| M2 | .834 [.80, .87] | .145 [.138, .152] | .35 [.31, .39] | 5.7 [5.0, 6.4] | −.022 [−.029, −.015] | −0.6 [−3.4, +2.2] |

### pi05 Sp-50
| arm | SR | IR | one-camera share | calls/ep | ΔIR | ΔSR (pp) |
|---|---|---|---|---|---|---|
| off | .920 [.900, .940] | .133 [.129, .138] | 0 | 1.50 [1.30, 1.72] | — | — |
| M1 | .920 [.895, .945] | .122 [.117, .127] | .27 [.23, .31] | 1.47 [1.25, 1.70] | −.011 [−.015, −.007] | 0.0 [−1.9, +1.9] |
| H2 | .920 [.895, .945] | .126 [.120, .131] | 0 | 1.31 [1.12, 1.52] | −.007 [−.012, −.003] | 0.0 [−2.0, +2.0] |
| M2 | .919 [.892, .944] | .111 [.105, .117] | .38 [.33, .42] | 1.30 [1.10, 1.52] | −.022 [−.028, −.016] | −0.1 [−2.1, +1.9] |

### pi05 Sp-200 (H2/M2 only if their screen passes)
| arm | SR | IR | one-camera share | calls/ep | ΔIR | ΔSR (pp) |
|---|---|---|---|---|---|---|
| off | .990 [.980, .997] | .0985 [.095, .102] | 0 | .51 [.42, .62] | — | — |
| M1 | .988 [.975, .997] | .084 [.080, .088] | .41 [.36, .46] | .53 [.42, .66] | −.015 [−.019, −.011] | −0.2 [−1.3, +0.8] |
| H2 | .990 [.978, .997] | .0955 [.091, .100] | 0 | .43 [.33, .55] | −.003 [−.008, +.002] | 0.0 [−1.1, +1.0] |
| M2 | .989 [.976, .997] | .077 [.071, .083] | .50 [.42, .58] | .43 [.33, .55] | −.022 [−.028, −.015] | −0.1 [−1.2, +0.9] |

### groot Sp-50 (H100)
| arm | SR | IR | one-camera share | calls/ep | ΔIR | ΔSR (pp) |
|---|---|---|---|---|---|---|
| off | .890 [.870, .910] | .128 [.124, .133] | 0 | 1.48 [1.30, 1.68] | — | — |
| M1 | .887 [.86, .91] | .119 [.113, .126] | .27 [.23, .31] | 1.52 [1.28, 1.80] | −.009 [−.015, −.003] | −0.3 [−2.6, +2.0] |
| H2 | .893 [.87, .915] | .120 [.115, .126] | 0 | 1.27 [1.08, 1.48] | −.008 [−.012, −.004] | +0.3 [−1.9, +2.5] |
| M2 | .892 [.865, .915] | .107 [.101, .113] | .36 [.31, .40] | 1.28 [1.06, 1.50] | −.021 [−.027, −.015] | +0.2 [−2.1, +2.5] |

### groot L10-50 (H100)
| arm | SR | IR | one-camera share | calls/ep | ΔIR | ΔSR (pp) |
|---|---|---|---|---|---|---|
| off | .748 [.72, .775] | .209 [.203, .215] | 0 | 10.3 [9.7, 10.9] | — | — |
| M1 | .748 [.71, .785] | .202 [.195, .209] | .125 [.10, .15] | 10.0 [9.2, 10.9] | −.007 [−.012, −.002] | 0.0 [−3.0, +3.0] |
| H2 | .743 [.705, .78] | .204 [.196, .212] | 0 | 9.8 [8.9, 10.7] | −.005 [−.012, +.002] | −0.5 [−3.6, +2.6] |
| M2 | .742 [.705, .78] | .196 [.187, .205] | .225 [.19, .26] | 9.8 [8.9, 10.8] | −.013 [−.021, −.006] | −0.6 [−3.7, +2.5] |

### pi05 L10-50, periodic knob .32
| arm | SR | IR | one-camera share | calls/ep | ΔIR | ΔSR (pp) |
|---|---|---|---|---|---|---|
| off | .905 [.885, .925] | .306 [.301, .311] | 0 | 15.0 [14.5, 15.5] | — | — |
| H2 | .905 [.88, .93] | .303 [.298, .308] | 0 | 14.9 [14.3, 15.5] | −.003 [−.007, +.001] | 0.0 [−2.2, +2.2] |
| M2 | .904 [.88, .93] | .294 [.289, .299] | .225 [.19, .26] | 14.8 [14.2, 15.4] | −.012 [−.016, −.008] | −0.1 [−2.3, +2.1] |

### Pooled over the four knob-off 50-episode cells (equal weights, paired)
| arm | ΔIR | ΔSR (pp) |
|---|---|---|
| M1 | −.009 [−.012, −.006] | −0.1 [−1.4, +1.2] |
| H2 | −.007 [−.011, −.003] | −0.2 [−1.6, +1.2] |
| M2 | −.020 [−.024, −.015] | −0.3 [−1.7, +1.1] |

**Matched-SR ΔIR, central values** (R11 knob slopes 29 pp per unit IR pi05 L10-50, 43 pi05 Sp-50, 44 groot L10-50):

| arm | pi05 L10-50 | pi05 Sp-50 | groot L10-50 |
|---|---|---|---|
| M1 | −.010 | −.011 | −.007 |
| H2 | +.008 | −.007 | +.006 |
| M2 | −.001 | −.020 | +.001 |

80% intervals ≈ ±.10 (L10) and ±.05 (Sp-50); not determinable on flat-slope cells (groot Sp-50, pi05 Sp-200). **Test A can establish the raw IR saving with SR non-inferiority, not a matched-SR saving.**

**Mechanism checks (80%)**: H2/M2 first call of any kind, paired by init over all episodes, vs off: +8..+14 decisions (pi05 L10-50), +1..+4 (both Sp-50 cells, pi05 Sp-200), +2..+8 (groot L10-50), 0 ± 0.5 (periodic). H2/M2 calls per failing episode within ±10% of off; per winning episode −15% to −45%. M1 first call within ±1.5 decisions of off; M1 calls per episode type within ±10%. ≤ 1% of failing episodes without any call, every arm. One-camera share within ±.05 of the centers above.

**Expected outcome**: vision savings as predicted (M1 ≈ −.010, M2 ≈ −.021), SR flat within noise for all arms, no call-starvation signature.
