# R12 opus — a guard that does not need both cameras (owner task, 2026-10-03 afternoon)

Evidence from B-pool data only: the B library (LOEO replays) and the B-dev closed-loop recordings
(`r12_devlooks_50L/50S/200/400/knob`, deployed system, two-camera serving, 100 episodes per arm, knob arms 50).
No test A, no astra directory. Scripts in `tools/` (`guard_table.py`, `guard_eval.py`, `guard_design.py`,
`guard_lib.py`, `guard_theta.py`, `guard_final.py`, `guard_timing.py`, `guard_hybrid.py`, `guard_m2.py`); outputs in
`out/guard*.json`, `out/guard/*.npz`.

## 1. What the old guards did and why only one survived
- R6–R8 ran four guards: stuck (state motion below a hand threshold, confirmed by camera cosine), terminal (top-1 at an
  episode end with the gripper closed), overtime (lag behind the library's median length) and no-progress (retrieved
  top-1 progress does not advance). R8's 500-episode ablations: removing stuck, terminal or overtime alone changed SR
  by −1.4…+1.6 pp (all n.s.); removing no-progress cost −4.4 to −5.0 pp; "only no-progress" recovered the full guard
  set on the sparse/long cells. They were dropped because they added nothing measurable, and they were
  hand-thresholded.
- No-progress reads the two-camera retrieval; one camera reproduces only 64–72 % of its calls (OFFLINE §12.3).

## 2. How a guard is judged here (on its own merit, not by agreement with the old one)
On every recorded look (`guard_table.py`): the episode outcome, whether the pure policy succeeds from the same
B init (library rollout) — a failing episode with pure success is "rescuable" —, and on the L10-50 arms the
**divergence D** between the cache chunk and the policy's own shadow chunk at that state (what a call would change).
At the same call rate as the deployed guard, a better guard puts more calls in failing episodes, fewer in episodes
the cache would win anyway, on higher-D states, and not later in rescuable failures.

The deployed guard is useful by these measures: D at its calls .573 / .585 (π0.5 / GR00T L10-50) vs .389 / .462 at
non-calls; 58–62 % of its calls fall in failing episodes (33–40 % of looks).

## 3. Agentview-free signals (`guard_eval.py`; AUC for episode failure / high D)
- Wrist-only distance to the library (relative to the arm median): failure .67–.86, high D .67–.76 — the strongest.
- Elapsed step .72–.75 / .65–.70; lag of the wrist top-1 behind its own demonstration .53–.74.
- Wrist-only progress advance (one-camera stall): predicts the OLD guard (.74–.90) but failure only .49–.70.
- Robot motion (xyz speed over two intervals): failure .62–.73; rotation, gripper opening, gripper-command
  consistency, oscillation and the action-vs-state mismatch (library ridge map from commanded heads to realised
  displacement): ≤ .62 each. Pure robot-state stall signals are weak; the calls that matter are not physical stalls.

## 4. Risk model (library-fitted)
`guard_lib.py`: on the B library's LOEO replay rows (50-episode subsets, both suites, per model) a logistic model of
"the two-camera cache chunk is in the top quarter of divergence from the stored policy chunk", from seven
agentview-free features: wrist-only nearest distance / its library median; step/100; (step − library step of the
wrist top-1)/its episode length; mean xyz speed of the last two look intervals; xyz direction reversals over the last
three intervals; log(1 + action-vs-state mismatch / library median residual); clip(wrist-top-1 progress advance,
−5, 8). Library in-sample AUC .76 (π0.5) / .66 (GR00T); on closed-loop dev it transfers: high-D AUC .77 / .67,
failure AUC .66–.83. Coefficients (standardised, intercept first) in `out/guard_lib.json`; the wrist distance
dominates (π0.5 +.91, GR00T +.48).

## 5. Thresholds: library quantiles over-call in closed loop
Matching the library rate of the old guard (`guard_theta.py`) makes the risk guard call more in closed loop (+.03 to
+.10 of looks; the opposite of R11's state-knob shortfall), and one threshold per model does not transfer between
suites (Spatial needs a higher one). Thresholds are therefore **per cell and dev-calibrated** (labelled), set so the
call rate equals the deployed guard's B-dev rate (IR unchanged).

## 6. Frozen guard H2 ("stall or risk")
Evaluated at every look with step ≥ 4 (two looks of history):
`call ⇔ [wrist-top-1 advance ≤ .5 library steps AND risk ≥ θ_a] OR risk ≥ θ_b`.
θ_a = 25th percentile of the risk over the cell's eligible dev looks; θ_b matches the deployed guard's dev call rate.
Dev-calibrated values (`out/guard_m2.json`):

| cell | θ_a | θ_b |
|---|---|---|
| pi05 L10-50 / Sp-50 | .122 / .140 | .635 / .673 |
| pi05 L10-200 / Sp-200 | .126 / .146 | .472 / .755 |
| pi05 L10-400 | .128 | .532 |
| groot L10-50 / Sp-50 | .170 / .195 | .550 / .843 |
| groot L10-400 | .178 | .394 |
(other sizes: use the nearest recorded size of the same model and suite.)

Same call rate as the deployed guard, B-dev, knob off (`guard_m2.py`):

| arm | calls in failing eps old → H2 | calls per look in winning eps | D at calls | first call in rescuable failures (median step) |
|---|---|---|---|---|
| pi05 L10-50 | .58 → .64 | .147 → .127 | .573 → .670 | 12 → 16 |
| pi05 Sp-50 | .58 → .66 | .092 → .076 | — | 12 → 12 |
| pi05 L10-200 | .33 → .40 | .117 → .106 | — | 24 → 34 |
| pi05 Sp-200 | .46 → .54 | .037 → .031 | — | 10 → 10 |
| pi05 L10-400 | .17 → .28 | .110 → .095 | — | 15 → 15 |
| groot L10-50 | .62 → .68 | .204 → .172 | .585 → .617 | 12 → 15 |
| groot Sp-50 | .59 → .75 | .067 → .041 | — | 18 → 10 |
| groot L10-400 | .40 → .51 | .200 → .163 | — | 10 → 10 |
| knob .32, pi05 / groot L10-50 | .32 → .37 / .42 → .48 | .131 → .107 / .191 → .168 | — | 20 → 16 / 11 → 11 |

Better targeting in every arm. The risk-alone guard (no stall gate) targeted even more sharply but called later on 5 of 8 arms
(median first call at rescuable failures e.g. 18 vs 12 on pi05 L10-50, 47 vs 24 on pi05 L10-200); the stall gate keeps the timing close to the old guard
except on three L10 arms (+3 to +10 steps). Open-loop caveat: all of this is replayed on the old guard's
trajectories.

## 7. Look-half with this guard (M2)
Because H2 needs no agentview, the "previous look called → full look" and "suspicion → complete" rules of M1 are no
longer needed: a look is wrist-first unless H2 calls or the knob plans a call; it stays one-camera when the agentview
memory is ≤ 8 decisions old and |W − WL_lag| + grip ≤ .10; otherwise the agentview is completed and the two-camera
cache answer is served (H2's decision stands either way). Serve the wrist + remembered-agentview chunk with the
wrist-only corrector head.

| arm | one-camera share | vision ΔIR per slot | retrieval cost (shadow) |
|---|---|---|---|
| pi05 L10-50 / Sp-50 | .35 / .35 | −.013 / −.013 | +.0012 / — |
| pi05 L10-200 / Sp-200 / L10-400 | .47 / .50 / .48 | −.018 / −.019 / −.018 | — |
| groot L10-50 / Sp-50 / L10-400 | .23 / .37 / .29 | −.008 / −.014 / −.011 | +.0001 / — / — |
| knob .32, pi05 / groot L10-50 | .22 / .15 | −.009 / −.006 | — |

Compared with M1 (old guard protected by completions: .12–.33 share, missed calls 1.4–8.6 %), M2 has no missed
calls by construction, a larger one-camera share, and works with the knob on.

## 8. Dev pilot plan (B-pool dev inits, `closed_loop/devset.py`, look recording on, 100 episodes per arm)
Cells: pi05 L10-50, pi05 Sp-50, groot L10-50, groot Sp-50, knob off; plus pi05 L10-50 with the periodic knob .32.
Arms per cell: (a) deployed; (b) H2 replacing the no-progress guard, two-camera looks (does the guard alone keep or
raise SR at equal calls?); (c) M2 (H2 + look-half). 15 arms.
Read-outs: call rate vs deployed (target ±.03), first-call step in failing episodes (stop if ≥ 3 decisions later on
average), share of calls in failing episodes, one-camera share (vs §7 ±.10), IR from the ledger, SR (±7–9 pp noise
at n = 100: a collapse check only). Nothing is refit on the pilot; θ stay as frozen. If (b) and (c) pass, test A:
same cells, 500 episodes, deployed vs (b) vs (c), plus pi05 Sp-200 and L10-200 for (c).
