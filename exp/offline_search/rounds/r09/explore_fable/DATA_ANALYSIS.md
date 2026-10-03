# R9 explore_fable — data analysis (evidence, commands, numbers)

Explorer: fable (Claude). Written 2026-10-01/02 (CDT). All numbers come from the R8 debug collection
`/home/weiland/trace_runs/os_closed_loop/r08_main` (70 arms × 500 pairs, deferred policy shadows at every
decision) unless stated. **Every selection, threshold, fold or model fit uses discovery inits 0–29 only; inits
30–49 were never read by any analysis below.** Arm-level descriptive SR/IR of the R8 arms are quoted from the
R8 arm table (all 500 pairs) only where labelled "R8 table".

Conventions: owner IR = (.152·v + .848·m) for π0.5 and (.148·v + .852·m) for GR00T, pooled over decisions
(wrist look .0646). "Motion gap" = RMS over the executed 5 controls × 6 motion dims of (served − policy
shadow), in units of the library action std per dim ("σ"); "grip disagreement" = fraction of the 5 executed
controls whose gripper sign differs from the shadow's. Intervals are 95% task-stratified init bootstraps
(2,000 draws) unless noted. Reproduction prefix for every command:

```
cd /home/weiland/projects/openpi
P="taskset -c 22-29 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src .venv/bin/python"
T=exp.offline_search.rounds.r09.explore_fable.tools
```

Derived tables (parquet/npz, ~7 GB): `/home/weiland/trace_runs/offline_search_store/derived/r09_fable/`
(`episodes/`, `decisions/`, `arrays/`, `arrays_norm/`, one file per arm). Analysis outputs: `out/` next to this
file. Unit tests: `tools/tests/test_tools.py` (14 tests, `$P -m pytest -q <path>`).

---

## 0. Extraction (tool `extract.py`)

```
$P -m $T.extract --workers 12            # 70 arms, ~3 min; episodes/decisions/arrays per arm
python /tmp/fable_extract_norm.py        # state_norm arrays (kept in tools/extract_norm note below)
```

For every accepted episode: success, decisions, looks, calls, wrist looks, summed owner cost. For every
decision: vision/src/owner_cost/d1/conf/retrieval extras + client join. Arrays per decision: served 10×7 block,
policy shadow 10×7, PCA-64 keys (both cameras), wire state, the 16 retrieval rows and weights. Checks: 70/70 arms
extracted, 0 errors; shadows finite at 100% of decisions in the 8 pure-cache arms; keys finite at 100% of look
decisions and NaN at 100% of blind decisions; pooled SR/IR of every arm reproduce the R8 arm table to 4 decimals.

Pre-treatment check (`paired`-style): at decision 0 the wire state is bit-identical across arms for the same
(task, init) (max |Δ| = 0 in all 500 pairs of three cells); the policy shadow at decision 0 is identical across
arms to ≤ .02 (same seed, same observation); PCA keys differ by a numerical jitter for π0.5 (p50 .2 on 64-d keys
whose d1 is 16–38; GR00T p99 .07). So task identity and step-0 observations are legitimate pre-treatment
features for a paired mixture, and the per-(task, init) pairing across arms is exact.

---

## 1. Per-task heterogeneity of the cache → policy gap (tool `paired.py`)

```
$P -m $T.paired --n-boot 500            # out/paired/<cell>.json
```

Per-task success on discovery inits 0–29 (30 episodes per task), four sparse cells. Columns: pure cache A /
independent coin IP (.25) / uniform calls CU (ρ .30) / pure policy P10; last column P10 − A.

| cell | task 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| π0.5 L10-50 A→P10 | .33→.97 | .90→1.0 | .90→.83 | .97→.90 | .53→.93 | 1.0→1.0 | .53→.83 | .63→.97 | .63→.70 | .73→.93 |
| π0.5 Sp-50 A→P10 | .90→1.0 | .73→1.0 | .97→1.0 | .97→.97 | .73→.97 | .83→1.0 | **.37→1.0** | .97→1.0 | .90→1.0 | .63→.97 |
| GR00T L10-50 A→P10 | **.23→.77** | .77→.90 | 1.0→.97 | .93→1.0 | .57→.83 | .97→.97 | .60→.83 | .43→.93 | .63→.80 | **.23→.93** |
| GR00T L10-500 A→P10 | .57→.77 | .97→.90 | .97→.97 | 1.0→1.0 | .83→.83 | .97→.97 | .80→.83 | .80→.93 | **.47→.80** | .90→.93 |

Reading: in every sparse cell 3–5 tasks carry essentially the whole gap and 4–5 tasks have A ≥ P10 (zero
value of calls). The hard tasks are the same for both policies on LIBERO-10 (tasks 0, 4, 6, 7, 9 for 50-demo
libraries; 0 and 8 for 500-demo). Spatial-50 is dominated by one task (6: A .37 vs 1.0; it alone is 6 of the
16 pp cell gap for π0.5). Full per-task tables for all 8 cells and all variants: `out/paired/<cell>.json`
(`per_task_discovery`).

### 1.1 Dose-assignment simulator

A per-pair *dose assignment* picks, for every (task, init), the outcome of one existing arm (A, O5b, O5a,
IP, CU, CT, P10, …). Because all arms ran the same pairs with the same environment seed, the mixture's SR and
pooled owner IR are real paired closed-loop estimates of the mixed controller as long as the choice depends
only on pre-treatment information (task identity, a per-task score fitted on other inits). This is the
central tool of this report; `paired.mixture / summarize / task_frontier / cv_frontier`.

**In-sample oracle** (per-task Lagrangian over all call variants, fitted and scored on the same 300 pairs)
vs **3-fold cross-validated** allocation (folds = init mod 3; the per-task choice is fitted on 20 inits and
applied to the other 10, pooled). Hulls, SR @ IR; "gain" = SR minus the uniform-dose family (A, IP, CU, P10)
linearly interpolated at the same IR:

| cell | uniform arms (discovery) | CV hull (selected points) | gain vs uniform at same IR |
|---|---|---|---|
| π0.5 L10-50 | A .717@.076, IP .790@.187, CU .880@.301, P10 .907@.504 | .840@.125, .887@.268 | +5.6 pp, +3.3 pp |
| π0.5 Sp-50 | A .800@.077, IP .900@.183, CU .927@.304 | .897@.123, .943@.176 | +5.8, +5.0 |
| GR00T L10-50 | A .637@.074, IP .713@.182, CU .797@.303 | .740@.148, .843@.336 | +2.7, +3.1 |
| GR00T Sp-50 | A .880@.075, IP .890@.185, CU .930@.303 | .927@.162 | +3.8 |
| π0.5 L10-500 | A .857@.077, CU .920@.187 | .903@.218 | −0.6 |
| GR00T L10-500 | A .827@.074, CU .830@.189 | .853@.093 | +2.6 |
| π0.5 Sp-500 / GR00T Sp-500 | ≈ .97–.99 everywhere | — | ≤ +0.5 / 0 |

The in-sample oracle is far higher (e.g. π0.5 L10-50 .933@.292) and is **not** evidence: choosing among 7
variants per task on 30 inits overfits. The CV numbers are the honest ones for an *outcome-calibrated*
allocation (what ~20 recorded episodes per task would give).

---

## 2. Label-free per-task calibration signals (tools `shadow_gap.py`, `task_alloc.py`)

```
$P -m $T.shadow_gap                     # 8 pure-cache arms → out/shadow/{decisions,episodes,summary}_<arm>
$P -m $T.task_alloc                     # correlations, signal-driven CV fronts → out/task_alloc/task_alloc.json
$P -m $T.task_alloc --rules --n-boot 1000   # pre-declared rules with paired bootstrap → out/task_alloc/rules.json
```

Per-task signals computed from the pure-cache arm's own trajectories (no success labels): `gap_all` = mean
motion gap served vs shadow over all decisions, `grip_all` = mean gripper disagreement, `d1_mean_looks` = mean
retrieval distance at looks (needs no policy at all), `a_fail` = A's failure rate (needs outcomes).

Spearman correlation with the per-task P10 − A gap (10 tasks, discovery inits):

| cell | gap_all | grip_all | d1_mean_looks | a_fail |
|---|---|---|---|---|
| π0.5 L10-50 | +.41 | +.83 | **+.97** | +.86 |
| π0.5 L10-500 | +.34 | +.62 | +.86 | +.57 |
| π0.5 Sp-50 | +.85 | **+.98** | +.51 | +.99 |
| π0.5 Sp-500 | +.76 | +.56 | −.54 | +.86 |
| GR00T L10-50 | +.72 | +.77 | +.78 | +1.00 |
| GR00T L10-500 | +.32 | +.75 | +.70 | +.87 |
| GR00T Sp-50 | +.26 | +.45 | +.42 | +.80 |
| GR00T Sp-500 | +.22 | +.73 | −.28 | +.69 |

Fold-to-fold stability of the per-task ranking (Spearman between disjoint 10-init folds) is .85–.94 for
`gap_all` in all cells, .73–.93 for `grip_all` on sparse cells, and only .02–.88 for `a_fail` (outcomes on 10
episodes per task are too noisy). Ranking stability vs. calibration size (Spearman of an n-episode-per-task
subsample against the 30-init ranking, mean over 200 draws): `gap_all` n=5: .77–.93, n=10: .88–.97; `grip_all`
n=10: .66–.96; `a_fail` n=10: .69–.94 (nan where A never fails). **Ten calibration episodes per task with a
shadow policy evaluation suffice to rank tasks reliably on the sparse cells.**

### 2.1 Pre-declared rules, cross-validated, paired against an IR-matched uniform mixture

Rule: rank tasks by the signal fitted on the two other folds; the top-k tasks run the "high" variant (CU =
uniform calls ρ .30 with stall trigger; or P10), the others run pure cache (A). Reference = the uniform-dose
family at the *same* owner IR: a random fraction f of episodes runs the next-cheaper/dearer uniform arm
(A/IP, A/CU or CU/P10), evaluated on the same pairs (fractional expectation), so the gain is a paired
bootstrap of (allocation − uniform) per pair. Selected rows (full table: `out/task_alloc/rules.json`):

| cell | signal | rule | SR @ IR | gain vs IR-matched uniform [95% CI] |
|---|---|---|---|---|
| π0.5 L10-50 | gap_all | top3 CU / A | .833 @ .139 | **+7.4 [+3.8, +10.9]** |
| π0.5 L10-50 | gap_all | top4 CU / A | .843 @ .159 | +7.1 [+3.1, +10.9] |
| π0.5 L10-50 | d1_mean_looks | top5 CU / A | .867 @ .188 | +6.2 [+3.6, +8.8] |
| π0.5 L10-50 | grip_all | top4 P10 / A | .883 @ .237 | +4.5 [+1.1, +8.2] |
| π0.5 Sp-50 | grip_all / gap_all | top2 CU / A | .887 @ .124 | **+4.0 [+1.4, +6.6]** |
| π0.5 Sp-50 | grip_all | top5 CU / A | .917 @ .194 | +4.8 [+2.7, +7.0] |
| π0.5 Sp-50 | grip_all | top3 P10 / A | .923 @ .209 | +4.7 [+2.1, +7.3] |
| GR00T L10-50 | gap_all | top3 CU / A | .753 @ .147 | **+6.4 [+1.8, +11.0]** |
| GR00T L10-50 | gap_all | top3 P10 / A | .810 @ .200 | +8.1 [+4.6, +11.6] |
| GR00T L10-50 | gap_all | top5 CU / A | .767 @ .185 | +4.8 [+2.1, +7.4] |
| GR00T Sp-50 | gap_all | top5 CU / A | .923 @ .176 | +3.4 [−0.7, +7.4] |
| π0.5 L10-500 | grip_all | top5 CU / A | .913 @ .131 | +2.5 [+0.7, +4.3] |
| GR00T L10-500 | grip_all | top2 CU / A | .857 @ .108 | +2.9 [+0.2, +5.3] |
| GR00T L10-500 | grip_all | top4 P10 / A | .890 @ .278 | +4.2 [+0.4, +7.8] |
| π0.5 Sp-500 | gap_all | top5 CU / A | .990 @ .124 | +1.1 [+0.3, +2.0] |
| GR00T Sp-500 | any | any CU / A | .96 @ .10–.13 | ≈ 0 |

Negative/neutral: P10-on-hard-tasks over-spends on π0.5 L10-500 (−2.6 to −4.9 pp vs the matched mix, because
CU already reaches P10 there); `d1_mean_looks` is unstable on 500-demo and Spatial-500 cells. All 4 signals
agree on the sparse cells; the shadow gap is the most stable, the retrieval distance needs no policy at all.

What a cell gains in cost terms at the *same SR* (reading the CV hulls): π0.5 Sp-50 reaches .943 at IR .176
vs CU .927 at .304 (−42% IR); π0.5 L10-50 reaches .887 at .268 vs CU .880 at .301; GR00T L10-50 .843 at .336
vs CU .797 at .303 (SR +4.6 at +.03 IR). Dense Spatial-500 cells have nothing to gain (A ≈ P10 already).

---

## 3. What does *not* predict failure early (tool `shadow_gap.py`, negative result)

Per-episode AUROC for "this pure-cache episode fails" on discovery inits, from features available at or
near the start of the episode:

| cell | gap first 1 / 3 / 5 / 10 decisions | d1 at step 0 | −conf step 0 | gap over whole episode (post hoc) |
|---|---|---|---|---|
| π0.5 L10-50 | .41 / .40 / .44 / .43 | .58 | .44 | .89 |
| π0.5 L10-500 | .58 / .53 / .52 / .50 | .63 | .50 | .82 |
| π0.5 Sp-50 | .54 / .51 / .52 / .63 | .63 | .57 | .99 |
| GR00T L10-50 | .47 / .50 / .54 / .51 | .54 | .42 | .89 |
| GR00T L10-500 | .47 / .49 / .48 / .45 | .59 | .46 | .69 |
| GR00T Sp-50 | .51 / .47 / .55 / .58 | .62 | .51 | .89 |

The disagreement between the cache and the policy during the first 10–50 controls carries no information
about the episode's fate (AUROC ≈ .5); the retrieval distance at step 0 is weakly informative (.54–.75). The
whole-episode gap is post hoc (failing episodes run long and drift off-distribution). **A "probe call at the
start to decide the episode's budget" is dead**; so is any episode-level gate from step-0 retrieval
statistics. This agrees with R6's finding that predicted disagreement does not place calls better than
uniform, and explains it: disagreement and failure are nearly unrelated at the time a decision could be made.

Noise floors (independent policy draws on the 1/32 sample, pairwise): motion .19/.16/.19/.15 (π0.5 cells)
and .051/.043/.054/.053 (GR00T); gripper disagreement .06/.03/.04/.01 and .010/.007/.003/.003. The cache's
look-time gap to the policy is .45/.28/.48/.28 (π0.5) and .52/.35/.47/.33 (GR00T): 2.4× the floor for π0.5
sparse cells and **10×** for GR00T sparse cells. Gripper disagreement .13–.16 on sparse cells vs ≤ .06 floor.

---

## 4. Synthesis rule is not the lever (tool `synth_eval.py`, negative result)

```
$P -m $T.synth_eval --max-decisions 6000     # out/synth/<arm>.json ; log out/synth_all.log
```

The fitted method artifact reproduces the served chunk from the logged 16 rows (max |Δ| ≤ 1.4e-5) and the
logged kernel weights (≤ 1e-5), so alternative rules can be scored against the shadow at 3,000–6,000 look
decisions per pure-cache arm (discovery inits). Motion gap, deployed kernel (kref 5 on 50-demo / 8 on
500-demo libraries) vs alternatives:

| cell | served | top-1 | uniform-16 | kernel kref 2 | kref 12 | local-linear (3 code PCs) | best Δ |
|---|---|---|---|---|---|---|---|
| π0.5 L10-50 | .447 | .532 | .465 | .478 | .451 | .434 | −3% |
| π0.5 L10-500 | .284 | .356 | .297 | .307 | .286 | .275 | −3% |
| π0.5 Sp-50 | .516 | .605 | .566 | .546 | .534 | .506 | −2% |
| GR00T L10-50 | .512 | .624 | .520 | .557 | .510 | .506 | −1% |
| GR00T Sp-50 | .482 | .573 | .527 | .518 | .496 | .479 | −1% |

Gripper majority vote changes nothing (the kernel mean's sign is already the weighted majority). The deployed
kernel is within 1–3% of the best re-weighting; no re-weighting, width, k, or local linear correction of the
16 neighbours moves the gap. **The information needed to match the policy is not in the neighbourhood.**

---

## 5. The shadow labels are the lever: student and grown library (tools `student.py`, `grow_library.py`)

```
CUDA_VISIBLE_DEVICES=0 $P -m $T.student --epochs 40          # out/student/<cell>.json, log out/student_all.log
CUDA_VISIBLE_DEVICES=0 $P -m $T.student --curve              # out/student/curve_<cell>.json, log out/student_curve.log
$P -m $T.grow_library --cell pi05:l10:50 --arms A --out <run>/fits/r9f_grownA_p_l10_50.pkl     # and --arms all
```

Every look decision of every R8 arm is a (PCA keys, state) → policy-chunk pair labelled where the *cache*
went (DAgger-style data). Train on inits 0–19 (all arms of the cell pooled), validate on the pure-cache arm's
look decisions at inits 20–29 (cache-served decisions only, seq > 0). Motion gap / gripper disagreement to the
shadow:

| cell | n train | served (kernel) | student MLP (keys+state+task) | student + cache chunk input | student state-only | floor |
|---|---|---|---|---|---|---|
| π0.5 L10-50 | 71,345 | .412 / .104 | **.238 / .044** | .241 / .046 | .333 / .072 | .188 / .060 |
| π0.5 L10-500 | 35,619 | .283 / .066 | .273 / .032 | .252 / .037 | .352 / .066 | .162 / .030 |
| π0.5 Sp-50 | 18,648 | .518 / .183 | **.283 / .053** | .285 / .052 | .389 / .101 | .187 / .043 |
| π0.5 Sp-500 | 11,842 | .292 / .037 | .300 / .025 | .270 / .018 | .372 / .035 | .150 / .011 |
| GR00T L10-50 | 77,908 | .525 / .163 | **.282 / .044** | .287 / .049 | .325 / .052 | .051 / .010 |
| GR00T L10-500 | 38,659 | .330 / .079 | .312 / .043 | .293 / .043 | .348 / .050 | .043 / .007 |
| GR00T Sp-50 | 17,764 | .455 / .117 | **.326 / .024** | .329 / .023 | .356 / .024 | .054 / .003 |
| GR00T Sp-500 | 11,715 | .323 / .026 | .337 / .015 | .310 / .015 | .369 / .020 | .053 / .003 |

Paired bootstrap of (student − served) motion gap: π0.5 L10-50 −.174 [−.183, −.166]; π0.5 Sp-50 −.235
[−.254, −.217]; GR00T L10-50 −.243 [−.254, −.234]; GR00T Sp-50 −.129 [−.144, −.115]; 500-demo cells −.01 to
−.04 (student + cache) or ≈ 0 (student alone). A 3×512 MLP (0.6 M parameters, CPU inference < 1 ms) on the
*same inputs the cache uses* is 40–46% closer to the policy than kernel retrieval on every sparse cell, and
removes 60–80% of the gripper-sign disagreements. Even a state-only student (no camera) beats the vision cache
on sparse cells.

### 5.1 Data, not architecture: learning curves and kernel kNN on the shadow-labelled set

Same validation; training rows subsampled by episode. kNN = AWM kernel (k 16, kref 5) on standardized
[keys, state, task] features, i.e. "grow the library with the labelled rows, keep retrieval":

| cell | training rows (episodes) | student | kNN on labelled rows | served |
|---|---|---|---|---|
| π0.5 L10-50 | A arm only: 6,563 (200) | .354 | **.301** | .412 |
| | 17,428 (550) / 36,282 (1,100) / 71,345 (2,200) | .317 / .264 / .238 | .283 / .269 / .257 | |
| GR00T L10-50 | A arm only: 6,874 (200) | .378 | **.346** | .525 |
| | 19,405 / 38,376 / 77,908 | .331 / .302 / .282 | .325 / .309 / .298 | |
| π0.5 Sp-50 | A arm only: 2,401 (200) | .412 | **.319** | .518 |
| | 4,630 / 9,313 / 18,648 | .377 / .327 / .283 | .317 / .293 / .280 | |
| π0.5 L10-500 | A arm only: 5,617 (200) | .380 | .315 | .283 |
| | 9,029 / 17,749 / 35,619 | .368 / .307 / .273 | .308 / .281 / .271 | |

Two hundred pure-cache episodes with deferred policy labels (6–7k rows) already cut the sparse cells' gap by
27–38% with plain kernel retrieval; the MLP only wins at ≥ 1,000 episodes. The 500-demo library (≈ 26k
policy-rollout rows) is matched only at ≈ 18k labelled rows and beaten at 36k. **The lever is adding
shadow-labelled rows at the states the cache visits**, not a new function class.

### 5.2 Grown-library artifacts with the deployed metric (`grow_library.py`)

The frozen A fit (per-task metric, kernel, scales) is kept; the candidate set is extended with the labelled
rows (train inits 0–19) as a registered library `grown`; the deployed arm differs from A only by
`--os-fit-artifact`. Offline check on the A arm's held-out look decisions (inits 20–29), emulating the
method's own regime-2 synthesis in numpy (emulation reproduces the served chunk to ≤ 1.4e-5):

| cell | rows added (source) | served motion / grip | grown motion / grip | share of retrieved rows that are new |
|---|---|---|---|---|
| π0.5 L10-50 | 6,763 (A arm, 200 ep.) | .412 / .104 | **.285 / .059** | 72% |
| π0.5 L10-50 | 73,945 (all 14 arms) | .412 / .104 | **.254 / .058** | 97% |
| GR00T L10-50 | 7,074 (A arm) | .525 / .163 | **.336 / .088** | 72% |
| GR00T L10-50 | 80,108 (all arms) | .525 / .163 | **.293 / .059** | 97% |
| π0.5 Sp-50 | 2,601 (A arm) | .518 / .183 | **.326 / .077** | 72% |
| π0.5 Sp-50 | 20,848 (all arms) | .518 / .183 | **.289 / .072** | 95% |

Artifacts: `/home/weiland/trace_runs/os_closed_loop/r09_fable_grown/fits/r9f_grown{A,all}_{p_l10_50,g_l10_50,p_sp_50}.pkl`
(+ `.json` check files).

### 5.3 Closed-loop confirmation (π0.5 LIBERO-10-50, 100 held-out pairs, same topology)

Run root `/home/weiland/trace_runs/os_closed_loop/r09_fable_grown`, h100 servers (ports 23240–23243) + 48
timan107 workers, standard mode, manifest = tasks 0–9 × inits 20–29 (`manifest_inits20_29.json`; the grown
rows come from inits 0–19 only). Arms: `r9f_ctrlA_p_l10_50` (A), `r9f_grownA_p_l10_50`,
`r9f_grownall_p_l10_50`, `r9f_P10_p_l10` (pure policy). Analysis: `$P -m $T.confirm_grown`.

Chain `runs/chain.log`: 4 arms × 100 pairs, 2.5–3 min each, no retries, servers/workers stopped after each arm.

| arm | SR (100 pairs) | owner IR | paired vs A: Δ [95% CI], wins/losses, McNemar p |
|---|---|---|---|
| `r9f_ctrlA_p_l10_50` (pure cache A) | **.740** | .0765 | — |
| `r9f_grownA_p_l10_50` (+6,763 rows from 200 A episodes) | .700 | .0763 | −4.0 pp [−15, +6], 16/20, p = .62 |
| `r9f_grownall_p_l10_50` (+73,945 rows, all 14 arms) | .730 | .0764 | −1.0 pp [−11, +9], 17/18, p = 1.0 |
| `r9f_P10_p_l10` (pure policy) | .930 | .504 | +19.0 pp [+10, +27], 24/5, p = .0005 |

Same 100 pairs under the R8-main topology (weilandserver servers): A .73, IP .82, CU .86, CT .91, P10 .87 —
the same-topology A control (.74) agrees.

**Reading.** A 31–38% reduction of the offline action gap to the policy produced **no change in success**
(−1 pp ± 10 at identical IR). The per-task picture explains the null: the grown library helps exactly on the
tasks the original 5 demos handle badly and hurts on the tasks they handle well.

| task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| A (10 pairs each) | .4 | .9 | 1.0 | 1.0 | .4 | 1.0 | .5 | .8 | .7 | .7 |
| grown (A rows) | .8 | 1.0 | .5 | .7 | .7 | 1.0 | .5 | .8 | .6 | .4 |
| grown (all rows) | .8 | .9 | .5 | .6 | .8 | .9 | .5 | .9 | .5 | .9 |

Tasks 0, 4 (and 9 with all rows) gain +30–40 pp; tasks 2, 3 lose 40–50 pp (there the policy itself is weaker
than the original cache: P10 .83/.90 vs A .90/.97 on inits 0–29, and the grown rows mimic the policy). With 10
pairs per task these per-task numbers carry ±15 pp noise.

**Post-hoc mixture, same pairs, identical IR .076:** pick the grown library for the tasks ranked hardest by a
label-free score computed on inits 0–19 only (the grown rows' source), the original library otherwise
(`tools/confirm_grown.py` session, 3,000-draw task-stratified bootstrap of the paired difference vs A):

| ranking signal (inits 0–19) | hard set | library on hard set | mixture SR | Δ vs A [95% CI] |
|---|---|---|---|---|
| mean retrieval distance d1 | top-3 {0, 4, 7} | all-row grown | .83 | **+9 pp [+2, +16]** |
| mean retrieval distance d1 | top-4 {0, 4, 6, 7} | all-row grown | .83 | +9 [+2, +16] |
| gripper disagreement | top-4 {0, 4, 6, 7} | all-row grown | .83 | +9 [+2, +16] |
| shadow gap | top-3 {0, 7, 9} | all-row grown | .81 | +7 [+1, +13] |
| shadow gap | top-5 {0, 1, 3, 7, 9} | all-row grown | .77 | +3 [−4, +10] |
| d1 / gripper / gap, top-3..5 | — | A-row grown | .73–.82 | −1 to +8 (CIs include 0 except d1 top-3: +7 [0, +13]) |

This is a post-hoc selection among 18 (signal, k, library) variants on 100 pairs and is reported as a
**hypothesis** with its exact numbers, not as a result: "per-task library choice" (grown rows only where the
original demos are weak) at unchanged IR. The confirmation plan is in PROPOSALS.md.

Lesson for offline screening (again): a large, cross-validated reduction of the action gap to the policy
(−38% on held-out inits, with the deployed metric) did not move closed-loop success at all when applied
uniformly. Distance to the policy is a necessary-direction signal at best; task-level closed-loop outcomes
(§1–2) remain the only quantity that predicted closed-loop gains in this round.

---

## 6. What offline evidence can and cannot say here

- The dose simulator (§1–2) uses **real closed-loop outcomes**; its only approximations are single-run noise
  (±3 pp per arm) and that the realized per-task controller will be a per-task ρ inside one lottery controller
  rather than a switch between whole arms. Its cross-validated numbers are honest; the in-sample oracle is not.
- The shadow-gap evidence (§3–5) measures *distance to the policy*, which R8 showed can under-predict
  closed-loop effects. Its value is screening, with one supporting in-system anchor: across the 50 → 500 demo
  contrast the cache's gap falls .45 → .28 (π0.5 L10) while SR rises .72 → .85, and .52 → .35 while SR rises
  .61 → .81 (GR00T L10). The closed-loop test in §5.3 is the direct check of the grown-library mechanism.
- Early-episode disagreement does not predict failure (§3), so no per-episode gating proposal is made.
- Nothing here used inits 30–49.
