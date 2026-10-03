# Opus round 2 — data analysis (evidence, numbers, commands)

All numbers use **discovery inits 0–29 only**. Every reader in `tools/` drops records with init ≥ 30 at parse time
(before decoding the rest of the record; `common.iter_jsonl_discovery`, `forensics.stream_csv`) and the two holdout
run roots are refused (`common.check_root`); unit tests assert both. Wherever a threshold or rule was *chosen*, it
was chosen on inits **0–19** ("FIT"); inits **20–29** ("EVAL") are reported separately and are the inits of the
requested closed-loop screen. Cells: π0.5 / GR00T × LIBERO-10 ("long") / Spatial × 50 / 500-demo library.

One disclosure: before the round-2 rules were fully loaded I read the existing R8 summary file
`rounds/r08/abl/RESULTS.md`, whose aggregates pool all 500 pairs. Nothing in this analysis uses those aggregates;
every number below is recomputed from inits 0–29 records.

Prefix for every command:
```bash
cd /home/weiland/projects/openpi
P=(taskset -c 2-9,46-53 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
M=exp.offline_search.rounds.r09.explore_opus.round2.tools
```
Full order: `bash exp/offline_search/rounds/r09/explore_opus/round2/tools/reproduce.sh` (≈ 3 min CPU).

---

## 0. Population

`$M.catalog` scans every run root's `arms.json` (R2–R9, holdout roots refused): **1,163 arms, 929 with discovery
data, 160,734 accepted discovery episodes** (catalog of 10-02 00:3x CDT; it includes the other researchers' new
100-pair round-2 runs, which the replicate families exclude because they require ≥ 290 discovery pairs). Arms are grouped into *controller families* by method class, fit
artifact and plugin flags (`replicates.family`):

| family | what it is | replicate runs per cell (300 discovery pairs each) |
|---|---|---|
| cache | frozen pure cache (BlindAWM prefit r5t/r5x tail1u[c]), no judge | 6–7 (R5, R6 ×2, R8 main, R8 abl, R9 astra, R9 fable) |
| policy10 | pure policy every 10 controls | 2 (R8 main, R9) |
| corrector | astra's half-strength residual corrector | 1–2 (R9 astra, R9 fable combo) |
| guards_B | the deployed guard controller (four guards) | 3–4 |
| only_no_progress | B with only the no-progress guard | 1 |
| uniform_calls / stage_tilted_calls | R8 calls at ρ ≈ .3 | 1 |

Replicates come from different rounds and hosts, so run-to-run churn below includes topology differences.
Standard-mode client logs give per-decision hit/miss/judge and (π0.5 only) the top-1 library row; for GR00T the
standard *server* decision logs carry uid, step and the top-k rows (`$M.serverledger`, admitted through the same
init filter on `uid`); they reproduce the R8 debug server rows exactly (top-1 and top-2 equal on all 21,251 decisions
of the R8 GR00T long-50 pure-cache arm).

## 1. Replicate-pooled outcome decomposition (`$M.replicates`, `out/replicates/`)

Per (task, init) pair, the pooled success probability p under the pure cache is estimated from all cache runs.
"Churn" = mean number of discordant pairs between two runs of the *same* controller.

| cell | cache runs | cache SR | cache-vs-cache discordant /300 | always fail (p = 0) | flips between runs | always succeed | pure policy SR | corrector SR (runs) | corrector-vs-cache discordant |
|---|---|---|---|---|---|---|---|---|---|
| π0.5 long-50 | 7 | .723 | 30.6 | .167 | .250 | .583 | .912 | .757 (2) | 75.7 (+42.9/−32.9) |
| π0.5 long-500 | 6 | .843 | 23.7 | .087 | .187 | .727 | .912 | .853 (1) | 45.2 (+24.2/−21.0) |
| π0.5 Spatial-50 | 7 | .820 | 17.4 | .133 | .143 | .723 | .992 | .930 (2) | 50.3 (+41.6/−8.7) |
| π0.5 Spatial-500 | 6 | .970 | 7.1 | .013 | .053 | .933 | .992 | .983 (1) | 8.0 (+6.0/−2.0) |
| GR00T long-50 | 7 | .620 | 37.0 | .270 | .237 | .493 | .865 | .642 (2) | 81.1 (+43.9/−37.2) |
| GR00T long-500 | 6 | .826 | 26.2 | .100 | .167 | .733 | .865 | .820 (1) | 49.8 (+24.0/−25.8) |
| GR00T Spatial-50 | 6 | .876 | 12.2 | .083 | .080 | .837 | .940 | .933 (1) | 41.2 (+29.2/−12.0) |
| GR00T Spatial-500 | 6 | .958 | 8.7 | .020 | .057 | .923 | .940 | .953 (1) | 18.7 (+8.7/−10.0) |

Readings:
- Two identical pure-cache runs on long tasks already disagree on 31 (π0.5) / 37 (GR00T) of 300 pairs. Any paired
  "+b/−c" on long tasks must be read against this; a single 300-pair run has ±3 pp of pure replicate noise.
- The corrector's discordance on long-50 (76–81 pairs) is 2–2.5× the null churn: it is **not** noise. It changes
  many outcomes in both directions.

### 1.1 Rescue vs break: every controller against the pooled cache labels (`out/anatomy/rescue_break.json`)

"Rescue" = success on pairs the cache *always* fails; "break" = failure on pairs the cache *always* solves.

| controller | π0.5 long-50 rescue / break | GR00T long-50 | π0.5 Spatial-50 | π0.5 long-500 |
|---|---|---|---|---|
| corrector (half strength) | **.46 / .114** | **.33 / .122** | **.71 / .028** | .42 / .055 |
| B guards | .49 / .041 | .44 / .103 | .38 / .007 | .44 / .024 |
| B with only no-progress | .46 / .034 | .44 / .122 | .30 / .005 | .50 / .014 |
| uniform calls ρ ≈ .3 | .76 / .080 | .63 / .081 | .80 / .055 | .54 / .023 |
| pure policy | .92 / .080 | .83 / .074 | .99 / .007 | .83 / .057 |
| placebo / perturbation arms (look every 5, follow lottery, look-less, wrist variants, shifted looks) | .10–.22 / .05–.21 | .17–.21 / .11–.18 | .08–.30 / .08–.12 | .04–.27 / .03–.19 |

Net corrector effect = rescues − breaks: π0.5 long-50 +23 − 20 ≈ +3 pp; GR00T long-50 +26 − 18 ≈ +2 pp; π0.5
Spatial-50 +28 − 6 ≈ +11 pp (pooled over the corrector's runs: +3.4 / +2.2 / +11.0 pp).

### 1.2 Why the corrector breaks long-task successes: fragility (`out/anatomy/fragility.json`)

Always-solved pairs are stratified by how many R8 perturbation arms (which change nothing systematic) break them:

| cell | share of always-solved pairs broken by ≥ 1 perturbation | corrector break rate: robust / low / mid / fragile pairs | pure policy break rate (same strata) |
|---|---|---|---|
| π0.5 long-50 (6 perturbation arms) | .46 | .047 / .105 / .345 / .125 (n 95/43/29/8) | .079 / .070 / .086 / .125 |
| π0.5 Spatial-50 (5) | .34 | .007 / .021 / .125 / .222 (n 144/48/16/9) | .003 / .010 / .031 / .000 |
| GR00T long-50 (4) | .33 | .071 / – / .212 / .278 (n 99/0/40/9) | .061 / – / .087 / .167 |

The corrector's breaks concentrate on pairs that *any* perturbation breaks (Spearman fragility→break .29 π0.5
long-50, .24 GR00T long-50, .27 π0.5 Spatial-50), and even on robust long-task pairs it breaks 5–7% vs 0.7% on
Spatial. The pure policy's own breaks are flat across strata (its failures are its own, not perturbation).

**Answer to "why does the corrector work on Spatial but not on LIBERO-10":** on long tasks (i) a much larger share
of the cache's successes sit on a knife edge, and the correction acts on them like a perturbation (break 11–12%,
the same rate as placebo perturbations), and (ii) it rescues fewer structural failures (46%/33% vs 71%; placebo
perturbations alone rescue 10–22% of them). The two effects cancel. On Spatial it rescues 71% and breaks 3%. Any
intervention applied to *every* episode on long tasks pays this breakage; selective interventions (guards: 3–4%,
oracle grasp windows: 3%) do not.

## 2. What a long-task cache failure looks like

### 2.1 Labels (R8 forensics, pure-cache arm, by pooled cache p) (`out/anatomy/failure_labels.json`)

| cell | always-fail pairs: grasp miss / misplace / undone / other | sub-goals done at the end (always-fail): 0 / 1 / 2 |
|---|---|---|
| π0.5 long-50 | 28 / 8 / 7 / 7 of 50 | 21 / 27 / 2 |
| GR00T long-50 | 50 / 12 / 4 / 15 of 81 | 57 / 24 / 0 |
| π0.5 Spatial-50 | 31 grasp miss, 6 never reached, 3 other of 40 | 40 / – / – |

π0.5 long-50 grasp-miss onsets are bimodal (first object ≈ 55 controls, second ≈ 195); GR00T's are mostly the
first object (median 64).

### 2.2 The cache follows the demonstration script blindly (`out/anatomy/script_backjumps.json`)

Top-1 retrieved library progress over the episode (π0.5, all cache runs):

| cell / pair bin / outcome | episodes | reached demo end (progress ≥ .9) | median control when reached | share of later looks at demo end |
|---|---|---|---|---|
| long-50 always-fail / failed | 350 | .91 | 230 | .21 |
| long-50 always-succeed / succeeded | 1225 | .91 | 220 | .10 |
| Spatial-50 always-fail / failed | 280 | .93 | 110 | .45 |
| Spatial-50 always-succeed / succeeded | 1519 | .98 | 90 | .12 |

A failing episode reaches the end of the demonstration it imitates **at the same time** as a successful one (≈ 230
controls of 520): after a missed grasp the cache keeps replaying the demo (carrying nothing, "placing" nothing) and
then idles at the demo's end until the time limit. It has no notion that the sub-goal failed.

Successful long-task cache episodes are not slower than the policy (median 253 controls for both; p90 383 vs 358);
pace of *successes* is not the problem.

Phase back-jumps (top-1 step falling ≥ 8 decisions between looks) and regressions (progress from ≥ .55 back to ≤
.35): 2.2 back-jumps / 29% regressions in always-fail long-50 failures vs 0.27 / 4% in successes, **but they occur
after the script is exhausted** — a consequence of idling at the end, not a cause. Phase-coherent retrieval (my O4)
is therefore not supported and was not built.

### 2.3 A task-agnostic trouble signal: pace lag (`$M.triggers`, `out/triggers/`)

lag = (decision index) − (library step of the top-1 neighbour). On-track episodes track their demo's pace (lag ≈ 0
throughout, median lag at decision 60 = 0 for always-succeed pairs); failing ones start lagging once the script runs
out. Frozen rule: **escalate at the first fresh decision with lag ≥ 12 and decision index ≤ 80** (chosen on FIT).

| cell (cache runs) | split | failures | detected | median trigger (controls) | detected by control 250 | successes | false alarms | cache self-recovery after the trigger |
|---|---|---|---|---|---|---|---|---|
| π0.5 long-50 (7) | FIT | 390 | .951 | 280 | .35 | 1010 | .149 | .29 |
| | EVAL | 191 | .984 | 270 | .39 | 509 | .138 | .27 |
| GR00T long-50 (7) | FIT | 522 | .987 | 190 | .74 | 878 | .153 | .21 |
| | EVAL | 277 | .993 | 190 | .69 | 423 | .135 | .17 |
| π0.5 long-500 (6) | FIT | 183 | .732 | 270 | .35 | 1017 | .040 | .23 |
| | EVAL | 100 | .850 | 250 | .51 | 500 | .032 | .16 |
| GR00T long-500 (6) | FIT | 233 | .901 | 250 | .50 | 967 | .063 | .23 |
| | EVAL | 80 | .825 | 170 | .67 | 520 | .056 | .31 |
| GR00T Spatial-50 (6) | FIT | 166 | 1.000 | 160 | – | 1034 | .050 | .24 |
| π0.5 Spatial-50 (7) | FIT | 244 | 1.000 | 170 | – | 1156 | .002 | .01 |

Alternatives evaluated on FIT (`triggers.DEFAULT_RULES`): weighted top-4 lag ≈ identical; the B guard's
no-progress span is earlier at small thresholds but non-specific (it fires in 89–98% of always-succeed episodes in
the real B runs, first at a median 60–90 controls); "demo-end reached k times" is late and misses half of long-task
failures; a pure time rule ("not done by decision 55") detects all failures but has 19–23% false alarms on
long-50 and 9–14% on long-500. A two-signal search (lag ≥ L and no-progress span ≥ n, FA ≤ 16%) found nothing
earlier than a median 250 controls on π0.5 long-50. The trigger trails the forensic failure onset by a median 71
(π0.5 long-50), 51 (GR00T long-50), 154 (π0.5 long-500), 52 (GR00T long-500) controls (`out/anatomy/onset.json`).

### 2.4 How much can the policy still do after the trigger? (`out/anatomy/recovery.json`, FIT inits)

Episodes of existing call arms that crossed the same trigger, success afterwards vs the share of policy calls in the
next 20 decisions:

| π0.5 long-50 arm | triggered | recovered | policy share after trigger |
|---|---|---|---|
| pure cache (7 runs) | 521 | .29 | 0 |
| independent coin p = .25 | 63 | .29 | .13 |
| oracle grasp windows | 53 | .32 | .03 |
| B guards / only no-progress | 236 / 53 | .50 / .49 | .31 / .30 |
| uniform calls / stage-tilted | 44 / 55 | .50 / .56 | .26 / .36 |

GR00T long-50: cache .21; coin .24; CU .37; CT .36; B .36; B only no-progress .31. Recovery falls with trigger time (π0.5 long-50: .46 before
control 200, .60 at 200–250, .48 at 250–300, .34 at 300–350, .21 at 350–400, 0 after 400 → deadline 80 decisions).
Recovered episodes finish a median 13 (π0.5) / 20 (GR00T) decisions after the trigger (q75 26 / 34).
A logistic fit of recovery on post-trigger call share across these arms (confounded by arm design; the within-coin
randomized slope is too noisy to use: −8 ± 4 for π0.5, +5 ± 3 for GR00T) extrapolates to .63 (π0.5) / .55 (GR00T)
at a full takeover (share .5). I use **r ∈ [.45, .65]** as the prior for the takeover's success probability.

## 3. Escalation simulator and pre-registered predictions (`$M.preregister`, `out/preregistration.json`)

`triggers.simulate` keeps each real cache episode's exact prefix (outcome, decisions, owner cost) up to the trigger
and replaces the suffix by a takeover: a policy call at every fresh decision (cost .152+.848, the policy tail
free), success with probability r (also for triggered episodes the cache would have recovered — conservative),
a rescued episode ending 20 decisions later, an unrescued one running to the time limit. Owner IR is pooled
cost / pooled decisions. Unit-tested accounting (`test_simulator_accounting`).

Frozen rule (lag ≥ 12, deadline 80); EVAL inits 20–29 (the FIT-inits values are in the JSON; they differ by up to
.05 where the base SR differs between the two init groups, e.g. GR00T long-500 base .805 FIT vs .870 EVAL):

| base | cell | base SR @ IR | r = .45 | r = .55 | r = .65 | 24-decision window, r×.7 |
|---|---|---|---|---|---|---|
| pure cache | π0.5 long-50 | .727 @ .076 | .793 @ .169 | .830 @ .162 | .867 @ .156 | .74–.80 @ .130 |
| pure cache | GR00T long-50 | .604 @ .074 | .736 @ .201 | .784 @ .193 | .831 @ .184 | .67–.74 @ .141 |
| pure cache | π0.5 long-500 | .833 @ .077 | .882 @ .126 | .899 @ .122 | .916 @ .118 | .86–.88 @ .104 |
| pure cache | GR00T long-500 | .867 @ .075 | .890 @ .125 | .905 @ .121 | .921 @ .117 | .87–.89 @ .101 |
| B only no-progress | π0.5 long-50 | .790 @ .185 (1 run) | .811 @ .209 | .839 @ .204 | .869 @ .199 | .77–.81 @ .173 |
| B only no-progress | GR00T long-50 | .750 @ .212 (1 run) | .797 @ .247 | .833 @ .241 | .871 @ .234 | .75–.80 @ .192 |
| corrector (round 1; fitted on inits 0–29, so **not** a valid 20–29 candidate) | π0.5 long-50 | .790 @ .076 (2 runs) | .852 @ .149 | .878 @ .143 | .904 @ .138 | – |

References recomputed on inits 0–29 from the per-decision ledgers (pooled over runs; owner IR):

| cell | pure cache | B (4 guards) | B only no-progress | uniform calls ρ .3 | pure policy |
|---|---|---|---|---|---|
| π0.5 long-50 | .723 @ .076 (7) | .840 @ .179 (4) | .840 @ .170 (1) | .880 @ .301 (1) | .912 @ .504 (2) |
| GR00T long-50 | .620 @ .074 (7) | .733 @ .220 (4) | .727 @ .215 (1) | .797 @ .303 (1) | .865 @ .504 (2) |
| π0.5 long-500 | .843 @ .077 (6) | .893 @ .156 (4) | .890 @ .153 (1) | .920 @ .187 (1) | .912 @ .504 (2) |
| GR00T long-500 | .826 @ .074 (6) | .870 @ .189 (4) | .883 @ .178 (1) | .830 @ .189 (1) | .865 @ .504 (2) |

The same simulator applied to the round-2 trajectories of the other researchers (inits 20–29, read-only):
fable's empty-grasp recovery `gm_corr0` (.790 @ .086) has 36% triggered episodes that already recover at .44, so
escalation on top predicts .79–.86 @ .16–.18; astra's round-2 shared-head corrector arms (.63 vs cache .72 on
π0.5 long-50) would be lifted to .75–.83 @ .17–.18.

**What the simulator cannot tell:** r itself (the single unknown), whether a takeover from a messy state behaves
like the policy from scratch, and second-order effects (the policy finishing faster or slower than 20 decisions).
Everything before the trigger is measured, not modelled — the pre-trigger trajectory of an escalation arm is the
pure-cache trajectory (verified in `test_escalate_calls_real_library`: identical actions and top-k until the
trigger).

## 4. Serving candidates and checks (`methods.py`, `tools/prepare_arms.py`)

- `EscalateCalls` (R8 `AnchorCalls` subclass; base = frozen pure-cache prefit): lag computed from the base anchor's
  top-1 row at each fresh decision; escalated → every fresh decision is a forced call (P10 pattern with CU's policy
  tail); optional `window`. `EscalateOnlyNP` / `EscalateOnlyNPGroot`: the frozen R8 only-no-progress artifacts
  plus the same rule as a forced MISS (os_reason 91).
- 14 unit tests (`tools/tests/test_round2.py`): rule-1 filter (a forbidden line is never decoded), forbidden
  roots, replicate statistics, signal arithmetic, simulator accounting, real-library identity before escalation /
  persistence / reset / deadline / clone isolation / pickling, the forced-MISS hook, the window, the B-artifact
  load, and the screen-analysis reader (client and server logs) on synthetic run roots.
- CPU plugin selftest (real per-connection stack, recorded store episodes, `--blind --policy-tail --judge
  guard_only`): all 8 frozen escalation arms PASS (`out/selftest_frozen_arms.log`); with test-only lag-1 artifacts
  the MISS + policy-tail path is exercised for π0.5/GR00T × cache/only-no-progress bases (PASS; 2–6 escalation
  MISSes each, persistent after the trigger, the next blind decision serves the policy tail).
- Local h100 dependency plan: 81 files, 16.3 GiB logical (`r09_opus_escalation/h100_sync/plan.json`).
