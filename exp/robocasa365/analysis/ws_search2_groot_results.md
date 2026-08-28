# ws_search2 — RoboCasa365 retrieval-weight search, round 2 (GR00T + pi0.5)

Round 1 searched a 132-cell field-weight matrix over a small cache library with
round-1 retrieval. Round 2 re-runs the same frozen matrix over the **full704**
library and adds **text-IVF** retrieval (prompt-bucket scoping in place of
task-key scoping). Because two things changed at once, the round-over-round
difference is a *joint* effect; a matched control arm splits it.

All numbers below are **pure retrieval**: `gate: always_search`, `judge:
always_hit`, `top_k 1`, `write_policy: never`. Every control step is answered
from the library, so `macro_sr` is the success rate of retrieval alone, not of
the teacher policy.

| arm | library | retrieval | scope | episodes |
|---|---|---|---|---|
| `ws1` (round 1) | small | `weighted_score_sum` + brute force | 132 cells | 104/cell |
| `ws2` main | **full704** | **`text_ivf`** | 132 cells | 104/cell |
| `ws2c` control | **full704** | `weighted_score_sum` + brute force | 12 manifest cells | 104/cell |
| `ws2e` densify | **full704** | **`text_ivf`** | 10 manifest cells | **416/cell** |
| `ws2` pi0.5 | pi0.5 full704 | **`text_ivf`** | 132 cells | 104/cell |

Completion is exact: every arm finalized with `n_err = 0` and `n_missing = 0`,
and every cell holds exactly its full episode count (`ws2` 132×104 = 13,728).

---

## 1. Joint effect over the full matrix (formal)

Evidence: `ws2_joint_full132.txt` (`compare`, no `--allow-partial`, so the tool
enforced full 132-cell coverage on both arms).

| | round 1 | round 2 | delta |
|---|---|---|---|
| mean `macro_sr` over 132 cells | 0.1579 | **0.2781** | **+0.1202 (+76 %)** |
| best single cell | 0.269 | 0.385 | — |
| worst single cell | 0.019 | 0.077 | — |

- **No cell got worse**: 131 improved, 1 unchanged, 0 regressed.
- **95 / 132 cells at p < 0.05**, 66 at p < 0.01 (paired sign-flip, 20 000 resamples).
- 82 cells now exceed round 1's *best* cell.

### 1.1 The optimum moved — this is the structural finding

Round 1's champion was `grid_vision_2@87_robot_state@12`: almost all weight on
`vision_2`. In round 2 that cell reaches only 0.279 and is no longer
competitive. The new leaders are **balanced mixtures with `robot_state` at
37–50 and only a moderate `vision_2`**:

| cell | ws1 | ws2 | delta | p |
|---|---|---|---|---|
| `grid3 v0@12 v2@37 rs@50` | 0.192 | **0.385** | +0.192 | 0.0000 |
| `grid v2@62 rs@37` | 0.221 | 0.375 | +0.154 | 0.0002 |
| `grid4 v0@12 v1@25 v2@12 rs@50` | 0.173 | 0.375 | +0.202 | 0.0000 |
| `grid3 v0@37 v2@12 rs@50` | 0.173 | 0.375 | +0.202 | 0.0002 |
| `grid4 v0@37 v1@12 v2@12 rs@37` | 0.173 | 0.365 | +0.192 | 0.0001 |

Single-field baselines: `iso_vision_2` 0.279, `iso_robot_state` 0.212,
`iso_vision_0` 0.096, `iso_vision_1` 0.077 — **every mixture beats every single
field**. Round 1's reading that `vision_0` / `vision_1` contribute nothing also
fails to survive: both roughly quintupled off a 0.019 floor. They remain the
weakest fields, but "no contribution" was an artifact of the small library.

---

## 2. Splitting the joint effect (formal, 12 manifest cells only)

Evidence: `ws2_factor_decomposition.txt`. The control arm holds the library
fixed at full704 and reverts only the retrieval shape, so
`lib = ws2c − ws1` and `ivf = ws2 − ws2c`.

| arm | mean `macro_sr` (12 cells) |
|---|---|
| `ws1` | 0.1899 |
| `ws2c` | 0.2516 |
| `ws2` | 0.2557 |

| factor | mean effect | share of joint | cells at p < 0.05 |
|---|---|---|---|
| library growth | **+0.0617** | **94 %** | 3 / 12 |
| text-IVF | **+0.0040** | **6 %** | **0 / 12** |

text-IVF's per-cell sign is 6 positive / 3 zero / 3 negative — indistinguishable
from zero at this sample size.

**Reading.** text-IVF's purpose is to keep retrieval cheap as the library grows,
not to raise SR; *not hurting SR* is the pass condition. The measured effect is
neutral-to-slightly-positive and not significant, so **text-IVF passes**, and
the SR gain is attributable to the larger library — which was the expected
mechanism, not a claim of this method.

⚠ **This decomposition is valid only for these 12 cells.** They were selected on
round-1 performance (top-8 plus the 4 `iso` cells), so they are round-1's strong
cells with less headroom: their joint effect is +0.066 against +0.120 over the
full matrix. Do not extrapolate the split to all 132 cells.

⚠ **No latency evidence was collected this round** (owner's call). The claim
"text-IVF makes retrieval cheaper" therefore has *no* experimental support here;
only "text-IVF does not cost SR" is supported. A paired latency measurement is
feasible from the existing setup (same library, same cells, same seeds, same
hardware) and would be needed before making the cost claim.

---

## 3. Measurement precision (densify arm)

Evidence: `ws2e_reproduce.txt`.

### 3.1 Replay noise floor = 3.0 %

`ws2e`'s first 104 episodes reuse `ws2`'s seeds, giving 1,040 matched pairs:

```
s→s 309    s→f 15    f→s 16    f→f 700
```

31 / 1040 outcomes flip (**3.0 %**), and the flips are symmetric (15 vs 16). So
identical (cell, task, seed) is not bit-deterministic, and **any per-cell
difference below ~3 pp is inside replay noise**. The +12.0 pp matrix-mean effect
is far above it.

### 3.2 Winner's curse ≈ 5 pp on the top cells

Re-running the 10 selected cells at 416 episodes each:

| cell | 104 ep | 416 ep | diff |
|---|---|---|---|
| `grid3 v0@12 v2@37 rs@50` (matrix leader) | 0.385 | **0.334** | −0.050 |
| `grid3 v0@12 v2@50 rs@37` | 0.356 | 0.332 | −0.024 |
| `grid4 v0@37 v1@25 v2@12 rs@25` | 0.356 | 0.329 | −0.026 |
| `grid3v v0@25 v1@37 v2@37` | 0.356 | 0.315 | −0.041 |
| `grid4 v0@37 v1@12 v2@12 rs@37` | 0.365 | 0.315 | −0.050 |
| `grid v2@62 rs@37` | 0.375 | 0.310 | −0.065 |
| `grid3 v0@37 v2@12 rs@50` | 0.375 | 0.308 | −0.067 |
| `grid4 v0@12 v1@25 v2@12 rs@50` | 0.375 | 0.298 | −0.077 |
| `iso_vision_1` (matrix floor) | 0.077 | **0.113** | +0.036 |
| `iso_vision_0` (matrix floor) | 0.096 | 0.099 | +0.002 |

All eight top cells fall; both bottom cells rise. Textbook selection bias: these
cells were chosen *because* they scored extreme on 104 episodes, and more data
shrinks them toward the truth.

**Consequences for how this must be reported.**

- The leader's true rate is **≈ 0.334**, not 0.385, and the top eight collapse
  into **0.298–0.334** — a band narrower than the noise floor times two. There
  is **no single best weight configuration**; there is a first tier of balanced
  mixtures.
- The same curse inflates round 1's 0.269 (also an extremum over 104 episodes).
  The **matrix-mean +12.0 pp comparison stands** (both rounds measured at equal
  precision), but **best-cell-vs-best-cell (0.385 vs 0.269) must not be used**.

---

## 4. Per-task effect (full matrix)

| task | ws1 | ws2 | delta |
|---|---|---|---|
| OpenCabinet | 0.055 | 0.416 | **+0.361** |
| CloseFridge | 0.475 | 0.737 | **+0.261** |
| SlideDishwasherRack | 0.119 | 0.372 | **+0.253** |
| OpenStandMixerHead | 0.455 | 0.683 | **+0.227** |
| OpenDrawer | 0.426 | 0.602 | +0.176 |
| TurnOnSinkFaucet | 0.208 | 0.313 | +0.105 |
| PickPlaceToasterToCounter | 0.145 | 0.228 | +0.083 |
| PickPlaceDrawerToCounter | 0.017 | 0.066 | +0.049 |
| CloseBlenderLid | 0.071 | 0.115 | +0.044 |
| PickPlaceCounterToStove | 0.012 | 0.031 | +0.019 |
| CoffeeSetupMug | 0.041 | 0.046 | +0.006 |
| PickPlaceSinkToCounter | 0.011 | 0.006 | **−0.006** |
| PickPlaceCounterToCabinet | 0.016 | 0.000 | **−0.016** |

The gains are almost entirely **contact-manipulation** tasks (open / close /
slide). The **pick-and-place family did not revive**: its best gain is +0.083,
three of five are ≤ +0.049, and two are negative with `CounterToCabinet` at an
absolute zero. A larger library plus prompt bucketing does not solve long-horizon
transport on this benchmark — a clean negative result, and the most likely place
to look next.

---

## 4b. pi0.5 arm — the same matrix on a second executor

Evidence: `ws2_pi05_matrix.txt`. The same frozen 132-cell matrix, same retrieval
recipe (`text_ivf`), over pi0.5's own full704 library. Completion is exact:
132/132 cells, every cell 104 episodes, `n_err = 0`, `n_missing = 0`.

⚠ **Round 1 never ran pi0.5** (groot_tp only), so pi0.5 has *no* round-over-round
delta. What it gives is a second, independent instance of the same search.

| | pi0.5 | GR00T |
|---|---|---|
| mean `macro_sr` over 132 cells | 0.1669 | 0.2781 |
| best cell | 0.298 | 0.385 |
| worst cell | 0.058 | 0.077 |

### 4b.1 The winning field is teacher-specific — and nearly inverted

Single-field baselines tell the story:

| cell | pi0.5 | GR00T |
|---|---|---|
| `iso_vision_1` | **0.231** ← pi0.5's best field | 0.077 ← GR00T's worst |
| `iso_vision_2` | 0.077 | **0.279** ← GR00T's best field |
| `iso_robot_state` | 0.125 | 0.212 |
| `iso_vision_0` | 0.067 | 0.096 |

pi0.5's top cells are `vision_1`-dominant (`grid v1@87 rs@12` 0.298,
`grid v1@87 v2@12` 0.269, `grid v1@62 rs@37` 0.260); GR00T's are
`vision_2` + `robot_state` balanced. pi0.5's *worst* cells are the
`vision_0`/`vision_2` mixtures that GR00T does well on
(`grid v0@50 v2@50`: pi0.5 0.058 vs GR00T 0.356).

**Spearman rank correlation between the two teachers' orderings of the same 132
configurations: ρ = +0.175** — essentially uncorrelated. GR00T scores higher in
126 / 132 cells (mean gap −0.111), but the *ranking* barely transfers.

⇒ **Retrieval field weights do not transfer across executors; they must be
searched per teacher.** This is the end-to-end confirmation of what the Phase-1
calibration only hinted at (pi0.5's `vision_2` separation J = 0.4356 vs GR00T's
0.3206 — different field-separability structure, hence per-teacher calibration).

⚠ The **absolute** level gap (pi0.5 lower everywhere) is *confounded*: the two
arms use different libraries (pi0.5 63,977 entries / 29.8 GB vs GR00T 50,795 /
20.5 GB), different models and different render resolutions. Do not read it as
"GR00T retrieves better than pi0.5" — it is a joint statement about
(teacher, library) pairs. The **rank** finding is not affected by this, because
it is computed *within* each arm.

### 4b.2 Per-task, pi0.5 vs GR00T

pi0.5 wins on `SlideDishwasherRack` (0.580 vs 0.372, **+0.208**) and marginally
on `PickPlaceSinkToCounter` (+0.043); GR00T wins everywhere else, most heavily on
`CloseFridge` (−0.488), `OpenStandMixerHead` (−0.372), `PickPlaceToasterToCounter`
(−0.221) and `OpenCabinet` (−0.215). Both teachers agree on the pick-and-place
verdict: the family sits near zero for both (pi0.5's best PickPlace is 0.048).

---

## 5. What can and cannot be claimed

**Supported**
- Growing the library raises pure-retrieval SR substantially and uniformly
  (+12.0 pp mean over 132 cells; no cell regressed). *(GR00T)*
- text-IVF is SR-neutral (+0.004, 0/12 cells significant) — it does not pay for
  its scoping with accuracy. *(GR00T, 12 manifest cells)*
- The optimal field mixture shifts from `vision_2`-dominant to
  `vision_2` + `robot_state` balanced once the library is large. *(GR00T)*
- Contact-manipulation tasks benefit; pick-and-place does not. *(both teachers)*
- **Field weights do not transfer across executors** — the two teachers' orderings
  of the same 132 configurations are essentially uncorrelated (ρ = +0.175), and
  their best single fields are inverted (`vision_1` for pi0.5, `vision_2` for
  GR00T). Per-teacher search is required, not optional.

**Not supported by this round**
- Any statement about retrieval latency or throughput (not measured).
- Any single "best" configuration (top eight are statistically tied at 416 ep).
- Extending the library-vs-IVF split beyond the 12 manifest cells.
- Best-cell-vs-best-cell round comparisons (winner's curse on both sides).
- Any round-over-round claim for pi0.5 (round 1 never ran it).
- "GR00T retrieves better than pi0.5" — the absolute gap is confounded by
  different libraries, models and render resolutions.

---

## 6. Evidence index

| file | contents |
|---|---|
| `ws2_joint_full132.txt` | formal 132-cell joint effect + per-task table |
| `ws2_factor_decomposition.txt` | formal 3-arm split over the 12 manifest cells |
| `ws2e_reproduce.txt` | matched-seed replay flips (noise floor) |
| `ws2_pi05_matrix.txt` | pi0.5 132-cell matrix + cross-teacher rank comparison |
| `ws2_s0a_bucket_variants.txt` | bucket map admission gate (111 buckets, 0 unresolved) |
| `ws2_s1_smoke.txt` | 2-cell smoke: 1,071 inferences all FULL_HIT, zero join gaps |
| `ws2_stage0_capacity.txt` | capacity calibration behind the 6-server × 6-fleet topology |

Run log, deviations and incident records: `logs/robocasa365_ws_search2_text_ivf_plan.log.md` §6b-1 … §6b-8.
