# R6 ideation G — Portability: one algorithm, its interface, and how to prove transfer

Ideation agent G (fable), R6 of `logs/offline_search_exploration.log.md`, 2026-09-28. Read-only research; no server, worker, chain, tmux, port, host or git access. One CPU census script was run (`/tmp/r6_G/lib_census.py`, `taskset -c 6-9,50-53`, BLAS 1 thread, CUDA hidden; output `/tmp/r6_G/lib_census.json`; numbers quoted in §1.3 and §4; script reproduced in §8). Every other number in this report is read from the cited file. Q1 (library quality), Q2 (budget) and Q3 (placement) are other agents' questions; this report only fixes the interfaces they plug into.

**Verdict in one paragraph.** Configurations A and B are already *mostly* calibrated from the library (PCA, whitening metric, kernel scales, stuck/visual thresholds, median episode length, V7 confidence map, blind motion scales): the non-portable part is small and concrete. It consists of (i) five protocol integers baked into code paths instead of read from the library manifest (2 cameras, 5 executed controls, H per model, 7 valid action dims with gripper at index 6, 8 valid state dims), (ii) one robot/policy sign convention (closed gripper) that P1 already turned into a declared constant, (iii) the task key taken from the instruction string, and (iv) a set of hand integers on the guards (stuck 2 decisions, lag 5 decisions, overtime 1.0, no-progress 3 transitions / 0.5 steps, percentile levels 10/95) whose *effective* false-alarm level on the library's own successful episodes is not one number but ranges from 0 % to 38 % across the eight LIBERO libraries (R5-B, `rounds/r05/ideation_B/REPORT.md:111-126`). The unified rule proposed in §2 keeps A's retrieval/synthesis and B's guard *statistics* unchanged, replaces every hand threshold by a leave-one-episode-out (LOEO) conformal quantile at one shared level α, keeps the commit rule as a function of (H, R), and exposes exactly three knobs: α, the commit block count B (default from H/R), and an optional hard call-rate cap ρ. What is provable is only false-alarm control on states exchangeable with the library's (finite-sample, distribution-free); every SR statement is empirical, and the closed-loop record of this line says offline proxies must not be trusted to rank SR (R4 §11, R5 §7). The transfer test that can settle "not LIBERO-specific" is RoboCasa365 on both policies with the existing W13 libraries (13 tasks × 50 successful trajectories, three cameras, 12 action dims, H = 50 for π0.5), with a *transplanted-constants* arm as the control that separates "the method transfers" from "the calibration transfers". MetaWorld MT50 (one camera, 4-d action, H = R = 5) is the boundary case where the cost lever disappears by construction and should only be run if the owner wants that boundary demonstrated.

---

## 1. Portability audit

### 1.1 Reading key

Status labels: **CAL** = already computed from the library at fit time (portable as is); **DERIV** = can be derived from the library or the policy metadata by a rule stated here; **DECL** = must be a declared interface input (manifest field or adapter function); **HARD** = a benchmark/model constant hard-coded in a code path that must become CAL/DERIV/DECL; **OFF** = present in the code but disabled in the deployed A/B (listed so nobody re-enables it without calibration). "Specific to": L = LIBERO, M = the policy model (π0.5 / GR00T), R = the robot/action convention, P = the serving protocol (decision every R controls, vision at step 0 and after a MISS).

### 1.2 Table

| # | Constant / assumption | Where (file:line) | Deployed value | Specific to | Status | Portable recipe |
|---|---|---|---|---|---|---|
| **Retrieval and synthesis (A)** | | | | | | |
| 1 | Two cameras (`key_v0`, `key_v1`) | `rounds/r02/g1_awm/awm.py:287` (`for f in ("v0","v1")`), `:300-304`; `r02/g3_recovery/g3_core.py:239-257`; `r04/k7_guard/judge.py:67-70`; `closed_loop/plugin.py` buffers `b_v0/b_v1` (`:818-819`); store builder `r0/build_library_store.py:6,256`; `r0/extract_queries.py:206-207` | 2 | L (RoboCasa 3, MetaWorld 1) | HARD | manifest `cameras` = C (the W13 RoboCasa manifests already list `vision_0..2`); every per-camera loop runs over C; PCA-64 per camera; stuck confirmation = min over C cameras |
| 2 | Key width 32,768 (4×4 pool × 2048) | `harness/dims.py:27` | 32768 | M | DECL | manifest `key_dim` (present); PCA is dimension-agnostic |
| 3 | PCA dims per camera | `awm.py:79` | 64 | hand | DECL (default) | keep 64; B1 showed offline sweeps do not rank SR, so do not "solve" it |
| 4 | Valid robot-state dims | `awm.py:80` (`RSV = 8`), `dims.py:25`, `dims.valid_state` used at `awm.py:311,486`, `k7:62-63`, `k1_blind/blind_awm.py:75` | 8 | L/R (RoboCasa π0.5 32-d with unknown valid count, GR00T 20-d; MetaWorld 4-d) | DERIV | valid mask = library dims with non-zero variance (π0.5 pads with exact zeros, `dims.py:10`); fall back to manifest `rs_valid_dims` (present) |
| 5 | Task key = instruction string | `plugin.py:796-803` (`rt.task_map.get(task)`), manifest `task_map` | 1 instruction per task | L (RoboCasa: 1–37 instruction variants per env task, GR00T h5 has no `prompt` attr; MetaWorld: 1 prompt per task) | DECL | adapter `task_key(episode_meta)` returning a *hierarchical* key (benchmark task id → instruction variant); retrieval restricted to the finest level with ≥ n_min library episodes (the whitening fit needs ≥ 2 episodes, `awm.py:326-328`) |
| 6 | Whitening metric (mean/std/W from 3 action-nearest other-episode pairs, ridge 0.1) | `awm.py:156-178,216` | nn 3, λ .1 | hand structure, library-fitted | CAL | closed form on the library; heads `[:R, :D_valid]` in σ units (σ from the library, `ctx.action_sigma`) |
| 7 | Kernel k and bandwidth kref | `awm.py:216`; `r03/h3_judge/judge.py:73` (`kref 5` at 50 episodes), `r04/k1_blind/blind_awm.py:54-55` (k must be 16) | k 16; kref 5 (50-ep) / 8 (500-ep) | hand, by library size | DECL (default) | keep k 16; kref 5 for ≤ 10 episodes/task else 8 is the current hand rule; B1 (`r05/ANALYSIS.md` §7) could not resolve 5 vs 8 at 500, so declare it, do not solve it |
| 8 | Early (step-0) fit rows: `step <= 2` | `awm.py:82` | 2 decisions | P (units of R) | DERIV | express as controls: rows with `step*R <= 2R` |
| 9 | Executed controls per decision and the stored tail `[5:10]` | `awm.py:308-310` (`act7[:, :EXEC_STEPS]`, `act7[:, 5:10]`), `:508` (`prev_a_exec[5:10]`); `dims.py:21` | 5 | P (LIBERO, RoboCasa and MetaWorld all use replan 5 today) | HARD → DECL | manifest `exec_steps` = R (present); slices become `[R:2R]` |
| 10 | Continuity after a MISS: λ_c and scale s_c | `awm.py:216,394,508-510` | λ_c .5; s_c = per-task median 1-NN tail→other-episode head | hand / CAL | CAL (s_c), DECL (λ_c) | unchanged |
| 11 | Confidence scales s_d, s_a, z-means/stds, zs_sd | `awm.py:393,419-427` | library pseudo-queries | CAL | CAL | unchanged (already LOEO: same-episode rows excluded, `:405`) |
| 12 | Insurance (norm_cap, hyst) | `awm.py:229-230,543-566` | off | — | OFF | — |
| 13 | Library names `BIG = {pi05: bpool_cs, groot: bpool_all}`, PCA cache path | `awm.py:75-77` | store-specific | store | DECL | `lib` kwarg names a manifest entry; PCA cache keyed by the library fingerprint (`:126-131`) |
| **Blind commit (A's second half)** | | | | | | |
| 14 | Horizon H per model | `dims.py:23`; `plugin.py:433,458` | 10 / 16 | M (RoboCasa π0.5 H = 50, GR00T 16; MetaWorld 5; `exp/step_diag/envs.py:69-80`) | HARD → DECL | manifest `H` (present) |
| 15 | Blind block = 5 controls, tail slicing, `executed_steps != 5` | `blind_awm.py:145,177,216-218`; `closed_loop/blind.py:56-59` (offsets 5/10); `plugin.py:1690-1704` (`cursor // 5`, `executed_steps != 5`) | 5 | P | HARD → DECL | R everywhere; policy-tail offset ∈ {R, 2R, …} |
| 16 | Blind budget B and its bound | `blind_awm.py:48` (0..4), `r04/k10_policy_tail/judge.py:24-27` (0/1), deployed B = 1 | 1 | measured on LIBERO (π0.5: H = 10 forces 1; GR00T: 1 on spatial, 2 on l10, `r04/ANALYSIS.md` §4.6, `r05` §2) | DECL (default) with DERIV bound | bound B ≤ ⌊H/R⌋ − 1 (exact, from the stored chunk); default B = 1; the arm in §4 tests B ∈ {1, 2} where H allows |
| 17 | Gripper channel index 6 in `lib.action[:, :5, 6]` | `blind_awm.py:76-77`; `dims.py:22`; `h3 judge.py:76,225` | 6 | R (7-d LIBERO action; RoboCasa has 12 valid dims) | DERIV | the action dim that is bimodal at ±1 in the library (gate G0 already checks bimodality, `harness/README.md` G0); fall back to manifest `gripper_dim` (present) |
| 18 | Terminal rows = last 2 rows | `blind_awm.py:78` | 2 decisions | P | DERIV | 2R controls |
| 19 | Blind state scale floor .05, motion percentile 10 | `blind_awm.py:82,86` | raw state units | L/R | OFF in A/B (`gates=budget_only` skips `gate_motion`; K7 replaced K1's dense stuck path) | if re-enabled: floor = a fraction of the library's global state std, never an absolute number |
| 20 | Gate masses .20/.20, residual .5, phase penalty .05, phase window +2 | `blind_awm.py:45,191-192,206-208` | hand | L | OFF in A/B (budget-only tail) | if re-enabled: LOEO quantiles at α; R5-B measured that the residual q95 fitted on inference paths fires on 11–15 % of cache-path windows (`ideation_B/REPORT.md:132`), i.e. drift, see §3 |
| 21 | Policy commit blocks | plugin `--os-policy-tail-blocks 1`; `q1_commit/judge.py:112-144` | 1 block = R controls | P | DECL | = B (same commitment for both sources) |
| **Guards (B's MISS trigger)** | | | | | | |
| 22 | Stuck thresholds m_thr, c_thr | `g3_recovery/wrappers.py:186-196` (percentiles of consecutive-row state motion and min-camera task-centred key cosine on the fit library) | p10 / p95 | percentile levels hand, thresholds CAL | CAL thresholds; levels → α (§2) | one level α replaces (m_pct, c_pct, stuck_thr) jointly via the run-length distribution `_F_still` (`wrappers.py:204-205`) |
| 23 | stuck_thr | `h3 judge.py:98`, `wrappers.py:381` | 2 consecutive still decisions | hand | DERIV | smallest n with P_lib(max still-run of a successful episode ≥ n) ≤ α |
| 24 | Overtime ratio > 1.0 and lag > 5 decisions, needs stuck ≥ 1 | `wrappers.py:342-343,377-380`; `h3 judge.py:249` | med_len per task CAL (`wrappers.py:197-200`); 1.0 and 5 hand | decision units | DERIV | overtime threshold = (1−α) quantile of ep_len/med_len over library episodes of the task; lag threshold = (1−α) quantile of LOEO pseudo-query lag (the `lag_pool` feature already computed in `_features`, `:344`) |
| 25 | No-progress: noprog_n 3, prog_eps .5 library steps | `h3 judge.py:98,237-240`; `r04/k1_blind/judge.py:15,76-86` (span variant) | hand | decision / library-step units | DERIV | run LOEO top-1 progress along library episodes; prog_eps = a low quantile of positive per-decision progress increments; n = smallest with P_lib(longest non-advancing run ≥ n) ≤ α |
| 26 | Terminal-with-closed-gripper; closed sign | `h3 judge.py:225,247` (`pa[4,6] >= 0`), `k1 judge.py:110-111`, `k7 judge.py:103-104`, `r06/p1_groot_commit/judge.py:39` (`CLOSED_SIGN`) | +1 π0.5 / −1 GR00T; index 4 = R−1, 6 = gripper | M×R | DECL (P1) | keep `CLOSED_SIGN` as a manifest field, **or** the sign-free variant (§1.3 (c)) |
| 27 | Events (dispersion 1.0 σ, vote .8), burst, return margin | `h3 judge.py:97-99` | off | — | OFF | — |
| 28 | V7 confidence: fixed feature signs, isotonic map, ncal 3000, MIN_CAL 30 | `wrappers.py:51-54,213-290` | LOEO on the fit library | hand structure, CAL map | CAL | portable; inactive under `guard_only` (no arm ever used it as a *level* test, see §2.5) |
| 29 | K7 model gate `ctx.model != "pi05"` | `k7 judge.py:38-39` | π0.5 only | M | removed by P1 | — |
| 30 | D1 aperture `state[:,6]-state[:,7]`, width .05, masses .5/.25/.75; C10 xyz monitor `rs[:, :3]`, q99 | `q1_commit/judge.py:18-30,33-58,61-88,138` | off | L/R state layout | OFF | if wanted: declare finger-state indices in the manifest; otherwise do not port |
| **Client, benchmark, cost** | | | | | | |
| 31 | Episode cap (controls) | `examples/libero/main.py:136-147` (spatial 220, l10 520); RoboCasa: registry horizon per task 450–1050 (`exp/robocasa365/episode_runner.py:263-266,552,589`); MetaWorld 160 policy steps (`exp/metaworld/tasks.py`) | benchmark | benchmark | DECL | the guard's overtime is library-relative (portable); the cap is the benchmark's and *defines SR* (every LIBERO failure is a cap timeout, `r04/ANALYSIS.md` §4.3 (ii)) |
| 32 | Pairing identity for paired tests | `plugin.py:804-810` (`orig_init_state_idx`, `task_uid`) | (task, init idx) | L init pool | DECL | RoboCasa: (task, seed, layout, style, pin) — same seed + same pinned scene ⇒ same initial state (memory `reference_robocasa365_benchmark_semantics`); MetaWorld: (task, idx into `MT1.train_tasks`) |
| 33 | Cost coefficients c1 (.152/.148), MISS steps k (10/8), controls per request L | `closed_loop/ops/cost_table.json`; FINDINGS | M | M | DECL | measured stage times per model and benchmark (`cost_ledger.stage_costs` already carries them) |
| 34 | Library admission = successful episodes only | store `success.npy`; libraries `current` are 100 % successful (§1.3) | benchmark success signal | benchmark | DECL | adapter `success(episode)` |
| 35 | Protocol: one vision anchor per R controls; vision at step 0 and after a MISS; blind decisions carry no keys | `plugin.py:1690-1704`; `closed_loop/README.md` "R4" | — | P | keep | protocol constants, not benchmark constants |

### 1.3 What the audit says

(a) **Most of A and B is already library-calibrated.** Rows 6, 10, 11, 22, 24 (med_len), 28 and the blind motion scales are fit from the library with no LIBERO number in them; P1's GR00T fit (`r06/p1_groot_commit/HANDBACK.md` §6) shows the same code producing model-appropriate thresholds (m_thr .0425 vs π0.5 .0518 on l10-50; c_thr .905 vs .957). The genuinely hard-coded set is rows 1, 4, 5, 9, 14, 15, 17 — all of which are *already present as fields in the library manifest* (`library/pi05_l10/current/manifest.json`: `H`, `Drs`, `act_valid_dims`, `exec_steps`, `rs_valid_dims`, `gripper_dim`, `task_map`, `key_dim`, `img_source_keys`) but the code reads `harness/dims.py` instead. Portability step 1 is a mechanical refactor: read those fields from the manifest and loop over C cameras. The W13 RoboCasa manifests (`/data/robocasa365_cache/cache_artifacts_w13/*_manifest.json`) already declare `vector_dims` with `vision_0..2`, `prompt_emb` and `robot_state` 32/20, so the input side exists.

(b) **The guards' hand integers are not one false-alarm level.** R5-B computed, for the deployed stuck rule (p10/p95, run ≥ 2), the share of *successful library episodes* it would alert: π0.5 spatial 0 / .023, π0.5 l10 .240 / .073, GR00T spatial .020 / .064, GR00T l10 .380 / .091 (50- / 500-episode libraries; `ideation_B/REPORT.md:111-120`). The same constants therefore run at an effective per-episode false-alarm level between 0 and .38 depending on library and suite — a 4× difference between library sizes on the same suite. Nothing in the constants was chosen to make that level equal across libraries; on a new benchmark it will be some other number. P1's all-vision replays make the same point at the decision level: forced-MISS share on recorded pure-inference streams is 20.5 % (GR00T l10-50) vs 8.3 % (π0.5 spatial-500) with identical constants (`p1_groot_commit/HANDBACK.md:203-220`). This is the concrete reason the guards need a *level*, not thresholds, to be portable (§2).

(c) **Gripper semantics.** P1 resolved the only sign-dependent guard with one declared constant per model (`CLOSED_SIGN`, `p1_groot_commit/judge.py:39`), and measured that the library itself reveals the convention (aperture change after g ≥ 0: −.166/−.088 on π0.5, +.007/+.042 on GR00T, `HANDBACK.md:61-66`) — but that measurement uses LIBERO's finger-state layout `rs[6]-rs[7]`, so it is not a portable *derivation*. Two portable options: declare `closed_sign` in the manifest (one bit), or replace guard 2 by a sign-free rule: "fire when top-1 is terminal and the executed gripper sign differs from the task's majority terminal gripper sign in the library" needs no convention (HYPOTHESIS; it changes which decisions fire; P1's table shows 422 of 484 GR00T spatial-inf-500 terminal rows are closed under the GR00T sign, so on that cell the two rules differ mainly on the 62 open-terminal rows). §4 keeps the declared-bit version for the transfer arms and lists the sign-free rule as an ablation.

(d) **Task key.** The instruction-string map is exact on LIBERO (10 strings ↔ 10 tasks). On RoboCasa the instruction is templated per object/placement (7 tasks single-instruction, OpenDrawer/OpenCabinet/SlideDishwasherRack 2 semantically different variants, PickPlace tasks 9–37 variants; `PickPlaceCounterToCabinet`: 37 unique prompts in 52 successful episodes — memory `reference-robocasa-build-l1s1-census`), and GR00T's collector stored no `prompt` attribute at all. A retrieval restricted to the instruction variant would leave 1–2 episodes per key; a retrieval restricted to the env task would mix left/right-drawer goals. Hence the hierarchical key with a library-derived fallback (row 5). This is the one place where "calibrated from the library" (the episode count per key) and "declared" (the hierarchy) must combine.

(e) **What is not an assumption but a precondition.** A's cost advantage needs H > R (blind blocks exist). MetaWorld's RLinf π0.5 has H = R = 5 (`exp/step_diag/analysis/step_vs_warmstart.md` §6.16: action_horizon 5), so on MetaWorld A degenerates to a vision decision every R controls (v = 1, IR = c1 ≈ .15 for π0.5) and the commit rule has nothing to commit. That is a stated precondition of the method, not a portability failure, and it is why MetaWorld is the boundary case in §4.

(f) **Library census (computed, `/tmp/r6_G/lib_census.json`).** `current` libraries: 5 episodes per task (π0.5 spatial 4–5), 100 % successful, median episode length per task 33–75 decisions (π0.5 l10), 16–25 (spatial), 35–81 (GR00T l10); 500-episode pools: 50 per task, success share .872 / .974 / .854 / .912 (π0.5 l10 / π0.5 spatial / GR00T l10 / GR00T spatial). These counts set the resolution of any per-task episode-level calibration: n = 5 exchangeable units per task at the deployed scale (§3).

### 1.4 Minimal interface a new benchmark / robot must supply

**Library manifest** (one JSON per library; ✓ = already exists in `offline_search.library.v1`): `model` ✓, `benchmark`, `H` ✓, `exec_steps` = R ✓, `act_valid_dims` ✓ (or a list of valid dims), `gripper_dim` ✓ (checked against bimodality), `closed_sign` (new; or the sign-free guard), `rs_valid_dims` ✓ (or derived by variance), `cameras` C and `key_dim` ✓ (`img_source_keys` ✓ names them), `task_map` ✓ generalised to a hierarchical task key, `action_sigma` (derived), `cost` {s1, s2, s3, k} per model.

**Benchmark adapter** (functions, no constants): `task_key(episode_meta) -> (coarse, fine)`; `success(episode) -> bool` (library admission and SR); `pairing_id(episode)` (paired tests); `episode_cap` (client); a key builder producing C pooled keys of width `key_dim` from stage 1 (the same `cp1_*` builders serve LIBERO and RoboCasa today; memory `reference_groot_stack_traps`: builder names must start with `cp1_`); the policy's normalized action space for stored chunks.

**Store layout** unchanged: `key_v<c>.npy` for c < C, `rs`, `action`, `task_id`, `episode`, `step`, `ep_len`, `progress`, `success`, `prev`, `next` (`r0/build_library_store.py:5-30`), plus `queries/<model>_<bench>_{inf,cache}` from trace-mode runs (`r0/extract_queries.py:10-17`: keys, rs, raw_state, `a_inf`, `a_hit`, `a_exec`). The builder needs the two-camera assumption removed (`:256,281,310`) and `raw_state` width read from the manifest (`extract_queries.py:209` asserts 8).

Everything else (metric, kernel, confidence, guard thresholds, commit bound) is computed from these inputs by the recipes in §2.4.

---

## 2. A unified formulation

### 2.1 Objects

- Library L: episodes e ∈ E, rows r with features x_r (C×PCA-64 + valid state), chunk a_r ∈ ℝ^{H×D}, state rs_r, step, ep_len, task key. All successful.
- Fit (per task key, as now): PCA per camera, whitening (mean, std, W), σ, kernel (k, kref), s_c, s_d, s_a, z-scales, V7 isotonic map, m_thr, c_thr, med_len. Unchanged from `awm.py` / `wrappers.py`.
- **LOEO calibration set**: every library row posed as a query against the *other episodes of its task* (already built: `awm.py:395-417` pseudo-queries; `wrappers.py:213-290` V7 calibration; `g3_core.py:344-369` `PseudoQuery`). For each pseudo-query we obtain the same statistics the online judge computes, and along each library episode the trajectory statistics (still-run lengths `wrappers.py:204-205`, lag, top-1 progress runs). Episodes are the exchangeable units; rows within an episode are not.

### 2.2 Statistics (each with a library reference distribution)

At a vision anchor t the method computes

- s^cov_t: coverage nonconformity = V7's predicted LOEO error ê_t (`wrappers.py:348-357`; monotone in the fixed-sign z-sum of −d_nn, −disp, −overtime, +vis, −stuck, −|lag|), i.e. the existing `Result.confidence` negated. (Ablation: the whitened nearest-other-episode distance d1 alone.)
- s^stuck_t: current vision-confirmed still-run length (K7's `confirmed_stuck`, `k7 judge.py:46-74`).
- s^lag_t: step − mean library step of the top-5 (`wrappers.py:342`).
- s^ot_t: step / med_len[task] (`wrappers.py:343`).
- s^prog_t: length of the current non-advancing run of top-1 library progress (K1's `noprog_span`, `k1 judge.py:76-86`).
- s^term_t: top-1 is terminal ∧ gripper condition (row 26).

Each s^j has a calibration sample {S^j_e : e ∈ E_task} = the per-episode maximum of the same statistic along the LOEO pseudo-trajectory of episode e (for s^cov: max over the episode's rows of the LOEO ê; for s^stuck: the episode's longest still run; etc.). Per-episode maxima make the calibration units exchangeable episodes and make the online test "is this the most anomalous the trajectory has been" compatible with sequential use (§3 G1 and its caveat).

### 2.3 Decision rule, commit rule, return rule, budget cap

**Conformal p-value** for test j at anchor t: p^j_t = (1 + #{e : S^j_e ≥ s^j_t}) / (n_j + 1), computed per task key when n_j ≥ n_min, else pooled over tasks after per-task standardisation of s^j (z-scoring by the task's LOEO mean/std, as V7 already does for its features).

**MISS rule**: call the policy at anchor t iff  min_j p^j_t ≤ α / T, where T is the number of tests in the family (Bonferroni, so the family-wise false-alarm level on library-like trajectories is ≤ α; if the owner prefers the per-test level, use α and report Tα). A test that cannot be calibrated (n_j < n_min at every level of the hierarchy) is dropped from the family, not run at a hand threshold.

**Commit rule** (both sources, unchanged from C10/P1 in structure): after a cache anchor, execute the anchor's own stored rows [R, (B+1)R) blindly (K1 `anchor_tail`); after a policy call, execute the policy chunk's rows [R, (B+1)R) blindly (C10 lifecycle tail). B = min(B_decl, ⌊H/R⌋ − 1), B_decl = 1 by default. Vision is mandatory at step 0, after every committed block, and after a MISS's tail.

**Return rule**: the anchor after a policy tail is judged by the same MISS rule; AWM's fresh regime uses the continuity term to the policy's executed tail (`awm.py:508-510`). No burst, no return margin (both OFF in the deployed B; P1 did not need them).

**Budget cap** (optional): if a hard rate ρ is required, run the plugin's existing running-quantile controller on top (`closed_loop/README.md` `--os-judge quantile:h:W`) with h = the share ρ implies (R5-B's inverse `m_target = (ρ − .152 v)/.848`, `ideation_B/REPORT.md:269-271`). The cap is a rate, the rule is a level; the cap only binds when the level rule over-spends.

**Library-quality coupling** (interface to Q1/Q2, not designed here): α is the single knob that Q2's budget rule sets from Q1's quality score and the SR target. With α fixed, the realized call share rises automatically with off-manifold states (drift) but *not* with within-library sparsity — see §2.6.

### 2.4 Constants and their calibration recipe

| Constant | Value / recipe | Source of calibration | Knob? |
|---|---|---|---|
| PCA dims, k, kref, λ, nn, λ_c | 64, 16, 5 (≤10 eps/task) / 8, .1, 3, .5 | declared defaults (B1: not SR-identifiable offline) | no |
| σ, whitening, s_c, s_d, s_a, z-scales, V7 map, m_thr, c_thr, med_len | as now | library (LOEO where applicable) | no |
| α | family-wise false-alarm level on library-like trajectories | user knob; Q2 maps quality/target → α | **yes** |
| per-test thresholds (stuck run, lag, overtime, no-progress run, coverage) | (1 − α/T) quantiles of per-episode LOEO maxima, per task key or pooled-standardised | library | no |
| n_min (episodes per key for a per-key calibration) | 10 (below it, pool); the deployed 50-episode libraries have 5 per task, so per-task calibration is *always pooled* at that scale | arithmetic (§3 G1) | no |
| B | min(B_decl, ⌊H/R⌋ − 1); B_decl = 1 | declared; bound from the library chunk | **yes** (default) |
| ρ | optional hard call-rate cap | user knob | optional |
| closed_sign / sign-free guard 2 | manifest bit, or the task-majority terminal sign from the library | declared / library | no |
| task key hierarchy, n_min for the key level | adapter; finest key with ≥ 2 episodes | declared + library | no |

Three knobs (α, B, ρ) against the deployed B's eleven hand numbers (stuck 2, lag 5, overtime 1.0, noprog 3, prog_eps .5, m_pct 10, c_pct 95, kref 5/8, budget 1, tail blocks 1, closed sign).

### 2.5 Implementation on the existing plugin (method side only)

- A new judge class wrapping `GrootCommitJudge` (so A ⊂ B ⊂ U nests, as P1 verified for A ⊂ B): in `fit()`, after the parent fit, build the per-test calibration tables from the LOEO pseudo-queries the parent already computes (V7's `_calibrate` loop touches every needed statistic; still-run lengths are `_F_still`; progress runs need one offline pass of top-1 progress along each library episode, which `_progress` (`k1 judge.py:76-86`) does online). Store per-task and pooled sorted arrays (a few KB per task).
- In `query()`: compute the statistics as now (all already in `extras`: `stuck_n`, `lag`, `overtime`, `noprog_span`, `pred_err`, `term1`, `gexec`), convert each to its p-value, set `extras["os_force_miss"] = 1` iff min p ≤ α/T, and `os_reason` to the arg-min test (codes 1–4 as now, 5 for coverage). Set confidence = min p so that `--os-judge threshold:<α/T>` reproduces the rule through the plugin's level mode — a mode that exists (`closed_loop/README.md:145`) but that no arm has ever used: the R3/R4/R5 arm files contain only `quantile` (12 arms), `periodic` (12) and `guard_only` (12) judges (counted from `r03_mx/arms.json`, `r04_frontier/arms.json`, `r05_q1/arms.json`).
- Nothing in the plugin, harness, client or `src/` changes for LIBERO. For RoboCasa, the changes of §1.4 (manifest-driven dims, C cameras, task-key adapter) touch `harness/dims.py` consumers, `r0/build_library_store.py`, `r0/extract_queries.py` and the plugin's buffer allocation; the RoboCasa server entry points (`exp/step_diag/serve_diag_pi05.py`, the GR00T RoboCasa server) must install the plugin the way `serve_pi05.py` / `serve_groot.py` do (`closed_loop/README.md` "How it plugs in": the wrap point `openpi.cache.config.build_per_connection_components` is shared).
- P1's replay tooling (`p1_groot_commit/replay.py`, `allvision_rates.py`) is the right CPU check for any new judge: its all-vision replay reports the realized force-MISS share on recorded inference and cache streams per cell, which under the unified rule should sit near α on inference streams and above it on cache streams by the drift excess (§3).

### 2.6 Why this covers "library strong → rely on cache; weak → insert calls", and where it does not by itself

- Reactive tests (stuck, lag, overtime, no-progress) fire on trajectory anomalies; anomalies are more frequent when the library is weak because the cache drives the robot off the library manifold more often. This self-adaptation is already visible in B: realized MISS share .123 vs .100 (C10 l10, 50 vs 500), .178 vs .137 (K7 tail), .201 vs .107 (K7 phase) (`r05/ANALYSIS.md` §3, `r04/ANALYSIS.md` §3.1–3.2). A level rule keeps that property.
- The coverage test has the *opposite* built-in bias: the LOEO reference distribution of a sparse library is wide (its held-out episodes are far from the rest), so a deployment query must be farther still to fall below α. Self-relative coverage measures "is this state as covered as the library covers itself", not absolute quality. Therefore α must depend on library quality through Q1/Q2, and the coverage test alone cannot be the whole answer. R5-B recorded the same fact from the other side: LOEO thresholds are stable under episode deletion but "a robust percentile can still be the wrong control threshold" (`ideation_B/REPORT.md:124`).
- An absolute reference exists only in the policy itself: the shadow-policy probe (being built) gives, on cache-driven states, the disagreement Δ between the served cache chunk and the policy's chunk; a second isotonic map z-sum → Δ, fitted on N recorded A trajectories, turns s^cov into an absolute predicted disagreement in σ units. The owner's charter allows "a small number of recorded trajectories" for calibration; this is the one place they are needed. Thresholding Δ against the policy's own resample dispersion (the store's `floor/`, median .13–.15 σ for π0.5 but .02–.06 σ for GR00T, `harness/README.md` metrics) is *not* portable as a fixed multiple, so the Δ threshold must come from the causal call-value curve (randomized injection, §4.3), not from a constant.

---

## 3. Guarantees versus empirical claims

**G1 (provable, finite-sample, distribution-free).** For a test statistic s^j_t at an anchor whose per-episode maximum is exchangeable with the n_j calibration maxima {S^j_e}, P(p^j_t ≤ α/T) ≤ α/T; with T tests, the family-wise false-alarm probability is ≤ α. This is the standard conformal/permutation bound; it needs no model of the policy or the environment, only that the calibration episodes were computed leave-one-episode-out (they are) and that deployment trajectories on which we *want* no alarm behave like library episodes. Resolution caveat: with n_j = 5 episodes per task at the 50-episode library, per-task p-values are multiples of 1/6, so any α < 1/6 per task is vacuous; pooling across tasks after standardisation gives n = 49–50 (α resolution ≈ .02); at the 500-episode pools n = 50 per task. Sequential caveat: using per-episode maxima makes the online statistic comparable to the calibration units, but repeated testing along one deployment episode still inflates the per-episode alarm probability relative to α/T unless the statistic is the running maximum; the honest statement is per-anchor.

**G2 (provable, trivial).** With the running-quantile cap, the realized call share over any window W of decisions is ≤ h + O(1/W) deterministically.

**G3 (design property, verified for A ⊂ B by P1).** With α → 0 the rule is A; with α → 1 it is "call at every anchor"; nesting holds bit for bit up to the k-th-slot tie rule P1 documented (`HANDBACK.md` §3).

**Not provable, empirical only.**
- That a call at a flagged anchor raises SR. K5 measured the causal value of a guard-forced call at +.034 [+.004, +.066] SR at 500 and −.002 [−.040, +.032] at 50 on π0.5 l10 (`r04/ANALYSIS.md` §7; `r05/q3_callvalue/HANDBACK.md`), i.e. positive at the scale where B ≈ A and unresolved at the scale where B ≫ A. Nothing in §2 changes that; the rule only chooses *where* the calls go.
- That false alarms on library-like states equal false alarms in deployment. Deployment states are cache-driven; R5-B measured residual quantiles fitted on inference paths firing at 11–15 % instead of 5 % on cache paths. The excess is the *drift term*; it is measurable online without labels (realized alarm share − α on episodes that succeed) and is itself a library-quality diagnostic (HYPOTHESIS for Q1).
- That ê (or any offline error) ranks SR across controllers. R5 B1's six prospective cells (−0.53 pp pooled, `r05/ANALYSIS.md` §7) and R4's tail-vs-phase reversal say it does not for *hyperparameters*; for *per-state* gating as a level rule it has never been tested (§2.5).
- That the calibrated guards at one α reproduce B's +5 to +12 pp at the 50-episode library. The current constants run at an effective level of .24–.38 on l10-50 successful demos; a level rule at α = .05 would fire *less* there. Whether the sparse-library gain survives is exactly what the transplanted-vs-calibrated arms in §4 test.

---

## 4. Transfer validation plan

### 4.1 What exists on the other benchmarks (read-only inventory)

| Benchmark | Policies | Observation / action contract | Closed-loop infrastructure | Libraries | Recorded trajectories |
|---|---|---|---|---|---|
| **RoboCasa365**, 13-task atomic subset, scene (1,1) pinned | π0.5 (`/data/ckpt/pi05_robocasa_pytorch`), GR00T-tp K=4 | π0.5: 3 cameras, robot_state 32-d, H = 50, 12 valid action dims; GR00T: 3 cameras, robot_state 20-d, H = 16, 12 valid dims; replan 5 (`exp/step_diag/envs.py:69-80`; W13 manifests `vector_dims`) | conductor + `exp/robocasa365/episode_runner.py`, servers `exp/step_diag/serve_diag_{pi05,groot}.py`; step_diag ran 9,760 formal episodes on it (`step_vs_warmstart.md` §1); pure-inference SR macro-13: π0.5 ≈ .55, GR00T ≈ .64 (`logs/session_handoff.md` appendix A §4) | native-format W13 pickles: 13 tasks × 50 successful trajectories per task per policy (`full`, 28.0 GB π0.5 / 19.0 GB GR00T) with nested tiers S1–S6 = 1, 2, 5, 10, 20, 50 trajectories per task (`/data/robocasa365_cache/cache_artifacts_w13/*_manifest.json`); build h5 corpus with per-step `vision_0..2` tokens, `robot_state`, `clean_action` (memory `reference-robocasa-build-l1s1-census`) | step_diag rows carry per-decision `top1_score`, `hit_type`, executed / full / k-step / warm chunks (`exp/step_diag/recorder.py`, `data/server/pi05/*/rows_*.jsonl`) but **no keys, no state, no 16-member retrieval** — not a query store |
| **MetaWorld MT50** | π0.5 RLinf SFT | 1 camera (corner2, 480² rotated), state `obs[:4]`, 4-d action, H = R = 5, ≤ 160 policy steps (`exp/metaworld/env.py`, `tasks.py`) | `exp.warm_reset.run` + `exp/metaworld/*`; 50 tasks × 20 episodes × 6 arms ran (`/data/wr_mw/formal/`) | none (owner ruling 2026-09-25: no library for that line) | full / plain / self arms, no cache keys |
| LIBERO (this line) | π0.5, GR00T | 2 cameras, 8-d state, 7 dims, H 10 / 16 | plugin + chain | store `library/*` (50 / 100 / 200 / 300 / 500 / grow250) | `queries/*_{inf,cache}` |

The RoboCasa W13 tiers give, for free, the analogue of LIBERO's 50-vs-500 axis: S3 (5 per task, 65 episodes) ≈ the deployed LIBERO scale, S6/full (50 per task, 650 episodes) ≈ the 500-episode pool. The scene is pinned, so paired episodes exist per (task, seed) inside scene (1,1); the cross-scene cells (1,7), (5,1), (5,7) of the owner's 2×2 design are the natural "library weak by construction" extension (the library comes from another kitchen).

### 4.2 The smallest convincing experiment

Claim to establish: *the same code, fitted only on the target library, with α calibrated as in §2, reaches on RoboCasa the same qualitative frontier as on LIBERO (A cheap and close to inference at the large library; B/U recover SR at the small library), and does so at least as well as LIBERO's hand constants transplanted.* Two policies, one benchmark, two library sizes, five configurations, all paired on (task, seed) within scene (1,1), 13 tasks × 50 seeds = 650 episodes per arm (or 13 × 25 = 325 for a first pass).

| Arm | Config | Library | Frozen from LIBERO | Recalibrated on RoboCasa | Paired comparison it answers |
|---|---|---|---|---|---|
| INF | pure inference L = R (and L = 2R if the client allows) | — | — | — | reference and noise floor (2 seeds of the same arm, as R4 §1 did) |
| A-S | commit-cache, B = 1 | S3 (5/task) | code, defaults (k, kref 5, PCA 64, λ) | PCA, metric, scales | A's cost/SR at the sparse library |
| A-L | same | full (50/task) | same (kref 8) | same | A at the dense library; gap to INF |
| A-L-B2 | commit-cache, B = 2 (π0.5 only; H = 50 allows up to 9) | full | | | is "B from H/R with default 1" right where H ≫ 2R (falsifier F4) |
| B-hand-L | P1 judge with LIBERO's constants (stuck 2, lag 5, overtime 1, noprog 3/.5, p10/p95) | full | **all hand constants** | thresholds m_thr/c_thr/med_len (library-fitted by construction) | the *transplant* control |
| U-L | §2 rule, α ∈ {.05, .10} (two arms), B = 1 | full | nothing but defaults | every threshold from the LOEO tables | calibrated vs transplanted (falsifier F1); U vs A (does calling help at the dense library); U vs INF |
| B-hand-S, U-S | same two at S3 | S3 | | | the sparse-library gain (does U keep B's +5–12 pp when the level is controlled) |

That is 10 arms for π0.5 (INF ×2 seeds, A-S, A-L, A-L-B2, B-hand-L, U-L ×2, B-hand-S, U-S) and 9 for GR00T. At 650 episodes: ≈ 12,350 episodes. Reduced first pass (325 episodes, π0.5 only, drop A-L-B2 and one α): 7 arms × 325 ≈ 2,300 episodes.

Acceptance criteria (declared before running; numbers are targets, not predictions): (i) U-L within a pre-declared non-inferiority margin of INF (δ = 3 pp on this benchmark's noise; SR ≈ .55–.65 makes one 650-episode arm's binomial sd ≈ 1.9 pp, larger than LIBERO spatial's) at IR ≤ .25 owner basis; (ii) U-S − A-S with a paired interval above zero; (iii) U-L ≥ B-hand-L and U-S ≥ B-hand-S in SR at no higher IR, or equal within noise (either outcome supports portability; only "B-hand ≫ U" refutes the calibration); (iv) a diff audit: the RoboCasa run uses zero benchmark-specific edits outside the manifest/adapter files of §1.4.

Predictions (HYPOTHESES, so the experiment can falsify them): A-L will sit further below INF on RoboCasa than on LIBERO (RoboCasa initial states are continuous poses with no init pool, so coverage per library trajectory is lower); the realized alarm share of B-hand on RoboCasa successful episodes will differ from LIBERO's by more than 2× (it already differs 4× between LIBERO library sizes); U at α = .05 will spend fewer calls than B-hand at S3 and the sparse-library gain will be smaller — the risk that motivates the α(quality) coupling.

Compute (memory `reference_robocasa365_benchmark_semantics`: π0.5 ≈ 92–97 s/episode single lane, GR00T-tp ≈ 46.5 s; 6–7 lanes per GPU host): π0.5 650-episode arm ≈ 17 lane-hours ≈ 2.5–3 wall-hours at 6 lanes; GR00T ≈ 8.4 lane-hours ≈ 1.5 wall-hours. Full design ≈ 10 × 3 + 9 × 1.5 ≈ 45 wall-hours on one server host plus timan workers; the reduced first pass ≈ 10 wall-hours. Plus the data collection below.

### 4.3 Data the unified algorithm needs, and which instrument collects it

The coordinator reports three instruments in construction: (I1) a shadow-policy probe logging the policy chunk next to the served cache chunk at every anchor; (I2) randomized MISS injection at every anchor with logged propensities; (I3) a feasibility study of simulator-state branch rollouts. Mapping of needs to instruments and run sizes:

| Need | Why the unified algorithm needs it | On LIBERO (calibrate U before transfer) | On RoboCasa (transfer) | Instrument |
|---|---|---|---|---|
| N1. Offline store for the target benchmark: library rows with C keys, state, chunks, episode structure; query trajectories from pure-inference and pure-cache runs | fit + LOEO calibration (§2.1); Q1's along-trajectory coverage on the target | exists | build from the W13 pickles + build h5 (bit-exact keys as `build_library_store.py` does for `current`): CPU, hours, ~30 GB read per policy; query trajectories: trace-mode (`--trace-out`, `logs/cache_trace_mode_plan.log.md`) pure-inference runs 13 × 50 seeds per policy = 650 episodes each (these double as INF) plus one A run | none new (trace mode + store builder with §1.4 edits) |
| N2. Policy chunk at every cache anchor on *cache-driven* states, with the served chunk and the retrieval statistics | the absolute recalibration of the coverage score (§2.6); Q1/Q3 surrogates; measuring drift per cell | A at both library sizes, both models, both suites: 8 arms × 500 = 4,000 episodes; server cost ≈ B's plus a full inference per anchor (half the decisions) ≈ 2–3× A's 10–14 min per arm | A-S and A-L per policy: 4 arms × 650 = 2,600 episodes at ≈ 1.5× the pure-inference cost | **I1** |
| N3. Per-anchor causal call value stratified by the p-values / statistics of §2.2 | choosing α (default and Q2's map) from data rather than from a proxy; testing whether low p predicts positive call value (falsifier F2) | A with injection at propensity π ≈ .10–.15 at every anchor, propensity logged: 8 cells × 2 replicates × 500 = 8,000 episodes (an injected arm is itself a valid "A + random calls" point; IPW recovers per-stratum effects; K5's one-landmark design gave [+.004, +.066] with 1,000 episodes at 500, so every-anchor injection at 1,000 episodes per cell is the minimum for stratified estimates) | A-S and A-L with injection, π0.5 first: 2 × 2 × 650 = 2,600 episodes | **I2** |
| N4. Counterfactual SR from the same simulator state (CALL vs CACHE branches at selected anchors) | validates the IPW estimates of N3 on a subsample; gives Q3's "value of a MISS here" without propensity variance | 10 tasks × 20 inits × ~5 anchors × 2 branches ≈ 2,000 partial rollouts per cell, one cell per model first | RoboCasa MuJoCo state restore is heavier (fixtures, RNG) — feasibility first | **I3** |
| N5. Realized false-alarm share of every test on (a) library LOEO pseudo-trajectories, (b) recorded inference trajectories, (c) recorded cache trajectories, (d) live U episodes that succeed | the drift ladder that says whether G1's level means anything in deployment; audit that transplanted constants and calibrated tables are compared at known effective levels | CPU replay only (P1's `allvision_rates.py` already produces (b) and (c) for the current constants) | same, after N1 | none new (extend P1's replay to emit per-test p-values) |
| N6. Pairing identity, scene/pin ids, instruction variant, success and cap-hit per episode | paired tests; hierarchical task key; SR definition | exists | must be carried in `__extra__` by the RoboCasa client (the conductor already stamps `task_uid`, `init_idx`, `env_seed`, `layout`, `style`, `pin_id` into step_diag rows) | none new |

Order: N1 → N2/N3 on LIBERO (choose the α default, verify F2) → N1 on RoboCasa → N2 on RoboCasa (recalibration) → the arms of §4.2 → N3 on RoboCasa only if U-L vs B-hand-L is unresolved.

### 4.4 MetaWorld as the boundary case (optional, needs an owner ruling)

Different robot (Sawyer), one camera, 4-d state, 4-d action, H = R. A library must be collected (50 tasks × 20 episodes of trace-mode pure inference ≈ 1,000 short episodes; ≤ 32 decisions each), against the standing ruling that MetaWorld gets no library. If run, it tests exactly two things: that the retrieval/guard code fits and calibrates on a 1-camera / 4-d contract without edits (interface claim), and that with B = 0 the rule's only lever is the call rate (IR floor c1 ≈ .15). It cannot show a cost advantage over pure inference beyond c1, which should be stated in advance so the null on cost is not read as a failure.

---

## 5. Data we want but do not have — requirements for one superset profiler

The owner wants one profiler that collects everything useful in a single closed-loop run rather than repeated single-purpose runs. The run executes A's control path perturbed only by randomized injection (so its outcomes are those of "A + π random calls", a legitimate arm), while computing and logging every shadow quantity. Requirements, each with granularity, the question it serves, how it is measured online, and cost:

| # | Quantity | Granularity | Answers | Online measurement | Cost |
|---|---|---|---|---|---|
| P1 | Policy chunk (full inference, H×D, normalized) at **every** decision — anchors *and* blind decisions — with its noise seed; never executed unless injected | per decision | Q1 (coverage → disagreement), Q3 (where the cache diverges), G (absolute recalibration §2.6; tail-vs-replan disagreement for the commit rule) | run stage 1+2+3 in shadow; the K10/K5 overlays already keep shadow computations off the control path | one full inference per decision: the run costs IR ≈ 1 + c1 regardless of what is served; ≈ 2–3× A's wall time |
| P2 | Stage-1 keys at blind decisions ("shadow vision") | per decision | makes the stuck/visual statistics computable at every decision; measures key-space drift during blind execution | run stage 1 on blind decisions; log the C pooled keys (or PCA-64 codes to save space) | stage-1 cost on ~half the decisions; 131 KB per decision raw keys (log the codes: 256 B) |
| P3 | Policy resample dispersion at a subsample of anchors (K = 4 fresh-noise samples) | per sampled decision (1/16 by the step_diag dense rule, `recorder.py:57-62`) | the policy's own noise floor on *deployment* states (the store's `floor/` is on library states) | K−1 extra stage-3 passes | ≈ +20 % of P1's stage-3 cost |
| P4 | Full retrieval diagnostics: top-16 rows, weights, whitened distances, the served chunk itself (not reconstructed), all §2.2 statistics and their p-values, the fired-test set, V7 features | per decision | Q1, Q3, G (audit of effective levels, N5) | already mostly in `extras`; add the served chunk (the R4 analysis had to reconstruct heads from the logged top-10, `closed_loop/README.md` KPI section) and the p-values | negligible (40 scalars + one chunk) |
| P5 | Randomized injection with logged propensity: at every anchor a Bernoulli coin with π_t (fixed, or stratified by the p-value bin so rare bins are oversampled), the assignment, and the executed source | per decision | Q2 (SR vs call share by IPW), Q3 (per-context call value), G (α default, F2) | the K5 overlay generalised from one landmark to every anchor; the coin from a SHA of (run, uid, step) as K5 does | the injected calls themselves (m ≈ π); no extra inference beyond P1 |
| P6 | Executed chunk and the resulting **control-resolution** state trajectory (raw state at every control, not only at decisions), plus gripper aperture / finger state and contact flags from the simulator, object poses where the env exposes them | per control step | Q3 (event-relative placement: grasp / release / contact), G (sign-free gripper semantics, portable event detection), the t4-style tail failures (`r04/ANALYSIS.md` §11 item 4) | client-side `per_step` rows (the conductor already writes per-step rows for RoboCasa); LIBERO needs the runner to emit sim state per control | cheap (a few hundred bytes per control) |
| P7 | Simulator state snapshot at every anchor (MuJoCo qpos/qvel + env RNG state + the episode identity) | per decision | branch rollouts later from any anchor (Q3's counterfactual value; validation of P5's IPW) | `env.sim.get_state()` on the client; store compressed | ≈ 2 KB per anchor → ~60 MB per 500-episode arm; branch rollouts are separate partial episodes |
| P8 | Task progress ground truth: sub-goal predicates from the simulator (LIBERO has per-predicate goal checks; RoboCasa exposes `_check_success` and object states) | per control step | Q1/Q3 (where along the *task* coverage breaks vs where along the *library episode* the guard thinks it is); evaluates the progress guard without library proxies | env predicate evaluation on the client | cheap where predicates exist; RoboCasa may only give the final success |
| P9 | Stage timings (s1, s2, s3, key build, search, judge) per decision | per decision | cost model declared per benchmark (row 33) | already logged for anchors; add blind and shadow paths | negligible |
| P10 | Episode record: success, cap hit, decisions, controls, hierarchical task key, pairing id (init idx / seed / layout / style / pin), library id and fit hash, α and tables' hash, injection propensity schedule | per episode | everything paired | `journal.jsonl` + `__extra__` | negligible |
| P11 | Library-side emissions at fit time: LOEO calibration tables per test (per task and pooled), derived interface fields (valid-dim mask, gripper dim, H, R, C), the *effective* α of any hand constants on the library's successful episodes, Q1's quality score, the policy noise floor on library states | per library | N5 audit; Q1; the α(quality) map | CPU at prefit | seconds to minutes |
| P12 | The same schema on the transfer benchmark, plus scene / pin / instruction-variant fields and the 3-camera keys | all of the above | transfer (§4) | same profiler through the RoboCasa server entry | RoboCasa episodes are 50–100× LIBERO's cost |

Run sizes for the profiler as a single experiment: LIBERO 8 cells × 500 episodes with π ≈ .10–.15 (one replicate gives P1–P4, P6–P10 for all anchors and P5 at ~1,500–2,000 injected anchors per cell; a second replicate doubles the causal sample) ≈ 4,000–8,000 episodes, ≈ 25–40 min per arm at the shadow cost, i.e. 4–10 hours; RoboCasa π0.5 A-S and A-L ≈ 1,300 episodes ≈ 45 lane-hours ≈ 8 wall-hours at 6 lanes; GR00T half that. P7 makes branch rollouts a later, separate, targeted experiment instead of a rerun.

---

## 6. What would falsify the unified approach, and ranked risks

**Falsifiers.**
- F1 (calibration is not the portable part). On RoboCasa, B-hand ≫ U at either library size with the same or lower IR. Then the hand integers carry information the LOEO tables do not, and "calibrate from the library" is the wrong recipe.
- F2 (no per-state signal). In the profiler data, the per-anchor call value (I2/I3) is flat in the p-value of every test in §2.2, at both library sizes, on both benchmarks. Then the level rule has no placement value and the method collapses to reactive guards + a rate cap; Q3's other signals would have to carry placement.
- F3 (drift dominates). The realized alarm share of U on *successful* cache-driven episodes exceeds α by a factor that differs by more than ~2× across cells with similar A-to-INF gaps. Then G1 says nothing useful about deployment and the level must be set per cell empirically, which is what the hand constants already are.
- F4 (commit rule). On RoboCasa π0.5 (H = 50), A-L-B2 ≫ A-L (B = 1) or A-L ≪ a B = 0 variant. Then B is not a function of (H, R) with default 1; a library-derived commit length (tail agreement along LOEO trajectories) is needed, and it must be tested rather than declared.
- F5 (the 50-library gain is a false-alarm artefact). U-S at α ≤ .10 loses B's sparse-library gain and only recovers it at α ≈ .3–.4, i.e. at the effective level the hand constants happen to have. Then the "one level" idea survives only with a steep α(quality) map, and the rule is not cheaper than B at the sparse scale.
- F6 (task key). On RoboCasa the hierarchical key at the finest level with ≥ 2 episodes leaves PickPlace tasks with 1–2 episodes per variant and the coarse level mixes goals; if A's per-task SR on those tasks is far below INF while the single-instruction tasks match LIBERO's pattern, the task-key interface, not the algorithm, is what fails to transfer.

**Ranked risks.**
1. Drift between library-like and cache-driven states (F3): measured 3× on residual quantiles (R5-B); the only mitigation inside the charter is the probe-based recalibration on a few recorded A trajectories (§2.6).
2. The sparse-library tension (F5): a controlled false-alarm level fires less exactly where B's calls buy the most (+5 to +12 pp at 50 episodes); α must rise with weakness, and Q2's map is empirical.
3. Calibration resolution at 5 episodes per task (§3 G1): per-task levels below 1/6 are vacuous; everything runs pooled at the deployed scale, which assumes tasks are exchangeable after standardisation (they are not: t4 loses under every tail arm, `r04/ANALYSIS.md` §2 (iii)).
4. RoboCasa contract differences beyond the manifest: H = 50 with 12 valid dims makes the whitening heads 60-d instead of 35-d and the stored tail 45 controls long; 3 cameras triple the PCA cost (K8: PCA is 31–49 % of the query time); instruction variants (F6); GR00T's 20-d state and missing `prompt` attribute.
5. Engineering: the plugin has only been installed on the LIBERO entry points; RoboCasa servers and the conductor client must carry `__extra__` identity, and trace-mode query extraction must accept C cameras and `raw_state` widths ≠ 8 (`extract_queries.py:209`).
6. Noise: RoboCasa SR ≈ .55–.65 gives ≈ 1.9 pp binomial sd per 650-episode arm and higher discordance than LIBERO spatial; 3 pp effects need replicates (R4/R5's acceptance rule).
7. History: every offline proxy in this line has failed to rank closed-loop SR (R4 §11, R5 §7). The unified rule is a proxy-driven gate; it must be scored only by paired closed-loop tests, never by its own LOEO numbers.
8. Owner rulings: MetaWorld has no library by ruling; cross-scene RoboCasa cells need admission-gate runs first (memory); K2-style MISS step reduction stays out of the system (ruling 12), so the cost basis of the transfer arms is full-inference MISS.
9. Cost/throughput on RoboCasa (≈ 95 s per π0.5 episode): the full design is ≈ 2 days of one server host; the reduced first pass is the realistic first step.
10. Guard 2's semantics: the declared `closed_sign` is one bit per (policy, robot); the sign-free replacement is a hypothesis with a different firing set (P1's 62 open-terminal rows on GR00T spatial-500) and must be an ablation, not a silent swap.

---

## 7. Summary of what is portable now, what must change, and what is only empirical

- Portable now (CAL): PCA, whitening metric, kernel scales, confidence scales, V7 map, stuck thresholds, median episode length, blind state scales, the terminal-row definition, the commit bound ⌊H/R⌋ − 1, the nesting A ⊂ B.
- Must change (HARD → manifest/adapter): 2 cameras, R = 5, H per model, 7 action dims / gripper 6, 8 state dims, the instruction-string task key, the library names in `awm.py`; plus one declared bit (`closed_sign`) or the sign-free guard.
- Must change (hand levels → one α): stuck 2, lag 5, overtime 1.0, noprog 3 / .5, p10 / p95 — replaced by LOEO quantiles at α with G1's false-alarm guarantee on library-like trajectories.
- Declared defaults that stay: PCA 64, k 16, kref 5/8, λ .1, nn 3, λ_c .5, B = 1.
- Only empirical: every SR consequence; the value of a call; the drift excess; the α(quality) map (Q1/Q2); the placement value of low p (Q3); the commit length beyond the bound.

---

## 8. Reproduction

Census (the only computation in this report):

```bash
cd /home/weiland/projects/openpi
mkdir -p /tmp/r6_G
cat > /tmp/r6_G/lib_census.py <<'EOF'
import json, numpy as np
root='/home/weiland/trace_runs/offline_search_store/library'
out={}
for lib in ['pi05_l10/current','pi05_l10/bpool_cs','pi05_spatial/current','pi05_spatial/bpool_cs','groot_l10/current','groot_l10/bpool_all','groot_spatial/current','groot_spatial/bpool_all']:
    d=f'{root}/{lib}'
    ep=np.load(f'{d}/episode.npy'); task=np.load(f'{d}/task_id.npy'); L=np.load(f'{d}/ep_len.npy'); succ=np.load(f'{d}/success.npy')
    n_ep_task={int(t): int(len(np.unique(ep[task==t]))) for t in np.unique(task)}
    med={int(t): float(np.median([int(L[ep==e][0]) for e in np.unique(ep[task==t])])) for t in np.unique(task)}
    ok=float(np.mean([bool(succ[ep==e][0]) for e in np.unique(ep)]))
    out[lib]=dict(rows=int(len(ep)),eps=int(len(np.unique(ep))),eps_per_task=[min(n_ep_task.values()),max(n_ep_task.values())],median_len_decisions=[min(med.values()),max(med.values())],success_frac=ok)
    print(lib, out[lib])
json.dump(out,open('/tmp/r6_G/lib_census.json','w'),indent=1)
EOF
taskset -c 6-9,50-53 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python /tmp/r6_G/lib_census.py
```

Output (2026-09-28): `pi05_l10/current` 2640 rows, 50 episodes, 5 per task, median length 33–75 decisions, success 1.0; `pi05_l10/bpool_cs` 29472 / 500 / 50 / 36–103.5 / .872; `pi05_spatial/current` 1018 / 49 / 4–5 / 16–25 / 1.0; `pi05_spatial/bpool_cs` 10909 / 500 / 50 / 16–25.5 / .974; `groot_l10/current` 2645 / 50 / 5 / 35–81 / 1.0; `groot_l10/bpool_all` 29631 / 500 / 50 / 37–92.5 / .854; `groot_spatial/current` 1063 / 50 / 5 / 16–25 / 1.0; `groot_spatial/bpool_all` 11751 / 500 / 50 / 17–26 / .912.

Judge-mode census of closed-loop arms (which modes were ever run): `r03_mx/arms.json` quantile 12, periodic 4, guard_only 3; `r04_frontier/arms.json` guard_only 3, periodic 8; `r05_q1/arms.json` guard_only 6; no `threshold:` arm anywhere (python one-liner over the `judge` / `plugin_args` fields of `arms.json`, run from the repo root with the same `taskset` prefix).

Files read (all read-only): `rounds/r06/FINDINGS.md`, `rounds/r05/ANALYSIS.md`, `rounds/r04/ANALYSIS.md`, `rounds/r02/g1_awm/awm.py`, `rounds/r02/g3_recovery/{wrappers,g3_core}.py`, `rounds/r03/h3_judge/judge.py`, `rounds/r04/k1_blind/{blind_awm,judge}.py`, `rounds/r04/k7_guard/judge.py`, `rounds/r04/k10_policy_tail/judge.py`, `rounds/r05/q1_commit/judge.py`, `rounds/r05/ideation_B/REPORT.md`, `rounds/r05/q3_callvalue/HANDBACK.md`, `rounds/r06/p1_groot_commit/{judge.py,HANDBACK.md}`, `rounds/r06/prompts/*`, `closed_loop/{README.md,blind.py,plugin.py (task map, blind lifecycle)}`, `harness/{README.md,dims.py}`, `r0/{build_library_store,extract_queries}.py` headers, `exp/step_diag/{envs.py,recorder.py,analysis/step_vs_warmstart.md}`, `exp/warm_reset/{envs,plan,run}.py` headers, `exp/metaworld/{env,tasks}.py` headers, `exp/robocasa365/episode_runner.py` (horizon), `examples/libero/main.py:136-147`, `logs/offline_search_exploration.log.md` §9 and R6 entries, `logs/session_handoff.md` appendix A, store manifests, W13 manifests, `r06_paper` summaries, and the memory notes on RoboCasa365 / GR00T conventions.
