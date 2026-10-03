# Opus round 3 — data analysis

Inputs: my round-2 screen `r09_opus_escalation` (inits 20–29, 100 pairs/arm, timan108), the discovery catalog and
ledgers of round 2 (inits 0–29), the R8 debug collection (inits 0–19 where anything is fitted or chosen). Holdout
roots (`r09_holdout_*`, `r09_astra_holdout*`) are refused by every reader (prefix rule, unit-tested); my round-2
catalog ran at 00:33, before those roots existed. Commands: `round3/tools/reproduce.sh`.

```bash
P=(taskset -c 2-9,46-53 env OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
M=exp.offline_search.rounds.r09.explore_opus.round3.tools
"${P[@]}" -m $M.dissect      # per-episode screen dissection -> out/dissect.json, out/episodes_*.parquet
"${P[@]}" -m $M.evidence     # crosstab, proximity, futility, early, b_calls -> out/evidence.json
"${P[@]}" -m $M.action_map   # model<->wire and wire->eef maps, inits 0-19 -> out/action_map_{pi05,groot}.json
```

## 1. Reading a 100-pair screen: noise first

Two runs of the *same* controller disagree on ≈ 10 (π0.5 long-50) / ≈ 12 (GR00T long-50) of 100 pairs (round-2
replicate churn 30.6 / 37.0 per 300). Paired differences of ±5 pp on 100 pairs are therefore not interpretable by
themselves. I attribute them with a cross-tab on whether *each arm's own trajectory* crossed the frozen rule (lag ≥ 12
by decision 80; for the base arm the rule is replayed on its logged top-1 rows). Before the rule fires the two
controllers are identical, so flips in the off-diagonal cells are run-to-run divergence; only the both-crossed cell
compares takeover vs no takeover on comparable episodes.

| comparison (round-2 screen) | headline Δ | neither crossed | only base crossed | only escalation arm crossed | **both crossed** |
|---|---|---|---|---|---|
| π0.5 long-50 escalation vs cache | +7 pp | 55 pairs, 0/0 | 6, +3/0 | 6, 0/−5 | **33: .55 vs .27 (+9/−0)** |
| π0.5 long-50 window-24 vs cache | +7 | 58, 0/−1 | 11, +5/0 | 3, 0/−3 | **28: .46 vs .25 (+6/−0)** |
| π0.5 long-50 guard+escalation vs guard | +8 | 60, +4/0 | 16, +7/0 | 10, 0/−4 | **14: .21 vs .14 (+1/−0)** |
| GR00T long-50 escalation vs cache | +13 | 48, 0/0 | – | – | **52: .44 vs .19 (+15/−2)** |
| GR00T long-50 window-24 vs cache | +11 | 48, 0/0 | – | – | **52: .40 vs .19 (+13/−2)** |
| GR00T long-50 guard+escalation vs guard | −5 | 56, 0/−3 | 10, +5/0 | 8, 0/−5 | **26: .27 vs .35 (+3/−5)** |
| π0.5 long-500 escalation vs cache | 0 | 80, +2/−2 | 3, 0/0 | 6, 0/−2 | **11: .36 vs .18 (+3/−1)** |
| GR00T long-500 escalation vs cache | +1 | 84, 0/0 | – | – | **16: .38 vs .31 (+1/−0)** |

Findings:
- On the pure cache the takeover works where it fires: +28 pp (π0.5) and +25 pp (GR00T) on the episodes that fall
  behind pace. GR00T runs on one fleet are deterministic before the trigger (trigger status identical in both arms on
  all 100 pairs); π0.5 runs diverge (12/100 pairs differ in trigger status).
- **"Escalation adds on top of the guard for π0.5 and subtracts for GR00T" is noise.** Where both runs crossed the rule,
  escalation changes nothing (π0.5 .21 vs .14, GR00T .27 vs .35). The headline +8 / −5 come entirely from pairs where
  only one run fell behind pace. Mechanism: by the time the lag rule fires under the guard controller, the guard has
  already called the policy 4.7 (π0.5) / 5.0 (GR00T) times in that episode; what is still behind pace is
  policy-hard (recovery .14–.35 with or without a takeover).
- **On 500-demo libraries escalation is nearly a no-op** because only 11–17% of episodes fall behind pace and the
  takeover lifts them modestly (+18 / +7 pp on 11 / 16 episodes ⇒ ≤ +2 pp overall, below the ±10-pair noise). On
  π0.5 two noise-induced flips cancelled it. Triggered 500-demo failures are the policy-hard residue: pure policy
  rescues 75–83% of 500-demo always-fail pairs when it runs from the start.
- **The bounded window is as good as persistence** where it matters: π0.5 .80 vs .80, GR00T .69 vs .71, at IR .118
  vs .172 and .143 vs .220. Rescued episodes finish a median 11–20 decisions (q75 27–36) after the trigger.

Per-episode details (`out/dissect.json`): escalation rate π0.5 39% / GR00T 52% / 500-demo 16–17%; median escalation
240–260 controls; base self-recovery after the same rule .31 / .19 / .36 / .31.

## 2. Fixes tried offline and rejected

- **Futility exit** (stop the takeover when the policy makes no library progress within 8 decisions): library progress
  under the policy does not separate rescued from unrescued takeovers (median advance 10.5 vs 5.0 decisions on π0.5,
  15 vs 14 on GR00T); a "< 3 steps" rule stops 1–2 rescues for 1–7 failures. Not proposed.
- **Earlier pace trigger**: no lag / no-progress combination fires before a median 250 controls at ≤ 16% false alarms
  (round 2). Not proposed.
- **Predicting structural failures at the start** with replicate-denoised labels (always-fail vs always-succeed pairs):
  initial retrieval distance gives within-task AUROC .57–.67 (`evidence.early`). Confirms fable's round-1 negative with
  cleaner labels. Not proposed.
- **Vetoing the guard's on-pace calls**: 44% (π0.5) / 36% (GR00T) of the guard's calls happen while on pace, mostly in
  always-succeeding episodes (78–91% on-pace there) — but 35% of the calls in episodes it rescues from always-fail pairs
  are also on pace. The saving would be large but the risk to its rescues is unidentified. Not proposed (`evidence.b_calls`).

## 3. New lever: start the takeover from the episode's start pose ("homing")

**Observation.** A takeover rescues a troubled π0.5 episode far more often when the arm is near where it started:

| data | takeovers | rescue, arm nearer than median | rescue, farther | AUROC (failed are farther) |
|---|---|---|---|---|
| round-2 screen π0.5 long-50 escalation (inits 20–29) | 39 | .65 | .32 | .69 |
| round-2 screen π0.5 long-50 window-24 | 31 | .56 | .27 | .71 |
| round-2 screen π0.5 long-500 escalation | 17 | .67 | .12 | .77 |
| R8 π0.5 long-50 uniform calls, after the same rule (inits 0–19, metric eef) | 41 | .76 | .30 | .85 |
| R8 π0.5 long-50 stage-tilted calls | 53 | .70 | .46 | .68 |
| R8 π0.5 long-50 pure cache (no policy) | 80 | .30 | .30 | .52 |
| R8 π0.5 long-50 coin p .25 (few calls) | 60 | .37 | .23 | .51 |
| GR00T (screen and R8, all arms) | 16–88 | .27–.50 | .16–.45 | .45–.60 |

The dependence appears only when the π0.5 policy acts, and not for GR00T. At the trigger the arm is a median 0.25–0.29 m
from its start. Interpretation (a hypothesis, confounded with how badly the episode went): π0.5's policy recovers from
configurations near its training starts; a troubled cache episode leaves it far away (usually at the end of the replayed
demonstration).

**Mechanism built.** `round3/methods.py:HomingEscalation`: same trigger; then a scripted homing segment from the robot's
own state — gripper open, lift 5 cm above max(current, start) height, move across, descend to the start position,
rotation unchanged — re-planned at every fresh decision (10-control segments, the blind decision serves the tail),
ending at 3 cm or after 8 fresh decisions; then a 24-decision policy window (`_home_w24`) or the cache directly
(`_home_cache`). Command maps fitted on inits 0–19 (`out/action_map_*.json`): model→wire affine exact on dims 0–5
(residual ≤ 2e-7); wire→end-effector ≈ 1 cm per unit command per control (R² .41–.85); gripper open = model −1 (π0.5) /
+1 (GR00T), checked against the recorded wire commands.

**Checks.** 9 unit tests (waypoints, saturation, state machine incl. give-up / deadline / reset, real-artifact identity
before the trigger, open gripper, lift first, blind tail = homing segment, pickling, the screen cross-tab on a
synthetic run root) pass; CPU plugin selftest PASS for
all four frozen homing arms; with test-only twins (lag 1, and lag 1 + one homing decision) the plugin serves the homing
segments and, for π0.5, the homing → policy window → policy-tail path (2 MISS, 2 policy tails).

**Predictions (pre-registered, 100 pairs, same-run controls).** π0.5: `_home_w24` .78–.86 @ ≈ .13 vs `_esc_w24`
.76–.82 @ ≈ .12 (round 2: .80 @ .118); `_home_cache` .70–.77 @ ≈ .077 vs cache .70–.76. GR00T (no proximity effect):
`_home_w24` .64–.72 @ ≈ .15, at most equal to `_esc_w24`; `_home_cache` .55–.62. Time budget: trigger ≈ 260 controls,
homing ≤ 80 (expected ≈ 40–50), window 120 — first-object failures that need the whole task redone may not fit.

## 4. Disclosure

One analysis command (`round3/tools/evidence.py`, 31 s) was run with `taskset -c 10-13` by mistake — outside my
assigned CPUs 2-9,46-53. Every other command used my range. No other rule was affected.

## 5. Screen outcome (coordinator, round 4 message, inits 20–29, 100 pairs) — homing is negative

| arm | π0.5 long-50 | GR00T long-50 |
|---|---|---|
| pure cache | .710 | .570 |
| escalation window 24 (round-2 rule) | .760 | .650 |
| homing then policy window 24 | .730 | .590 |
| homing then cache | .640 | .550 |

Homing lowered success relative to the plain window (−3 / −6 pp) and homing alone hurt the cache (−7 / −2 pp). The
pre-registered predictions (π0.5 homing window .78–.86, homing alone .70–.77) were wrong. Reading: the observational
proximity effect was confounded with how badly the episode had gone, and the 40–80 controls spent homing cost more than
any in-distribution benefit. **The homing line is dropped.** The bounded escalation window is the surviving lever.
