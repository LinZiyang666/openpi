# R3 family H1 — pure cache, prevent trap entry: `AWM3` (AWM + borrowed-prior metric, ridge / pooled twins, gripper commitment, terminal-row guard)

`awm3.py: class AWM3(AWM)` (subclass of `rounds/r02/g1_awm/awm.py:AWM`; r02 untouched). Every switch is independent and
every non-default value is encoded in `name`. Defaults reproduce the CL2 selector: `AWM3()` = `AWM(lib="current", kref=5)`
**bit for bit** (top-k, scores, confidence, synthesized action, extras; `tools/equiv_check.py`, 330 / 278 / 336 decisions
on pi05_spatial_cache / groot_l10_inf / pi05_l10_inf), `AWM3(prior_alpha=1)` = `AWM(lib="current", fit_data="big", kref=5)`
bit for bit (330 / 150 decisions), `AWM3(lib="big")` = `AWM(lib="big", kref=5)` bit for bit (367 decisions, pi05_l10_cache).
The only intentional difference from AWM: exact distance ties in the top-k are broken by the lower row (the G3 wrappers'
`topk_pos` convention) instead of `argpartition`'s arbitrary pick — duplicate step-0 keys exist in the 10× library, and
plain `AWM(lib="big")` therefore fails `contract_check` on 1 / 246 decisions (`derived/r03/h1_trap/contract/c0_AWM_big_kr5_reference.json`);
AWM3 passes.

## Algorithms as implemented

Notation: AWM's per-task metric fits `S_w` = mean outer product of standardized feature differences over action-similar
pairs (each row's `nn = 3` nearest heads among OTHER episodes of the task), `M = (S_w + lam·tr(S_w)/d·I)^-1`, `W = chol(M)`,
codes `((x − mean)/std) @ W`, kernel-16 with `kref = 5`, features = [PCA-64(v0), PCA-64(v1), rs[:8]] (136-d).

1. **`prior_alpha` a ∈ [0, 1] — BORROWED BIG-LIBRARY INFORMATION** (A-P1 / C-P1). Candidates stay `current`. The basis is
   the big library's PCA-64 (r01 `bpool_cs` for π0.5 / `bpool_all` for GR00T), `mean/std` are the big rows' per-task
   statistics, and `S = (1−a)·S_w(current rows) + a·S_w(big rows)` with both covariances estimated in that same basis and
   same standardization (C's rule), then AWM's trace-scaled ridge and Cholesky. Same construction for the early (step-0)
   metric on the `step ≤ 2` rows of both libraries. a = 1 → the big fit alone (`…_cur_fbig_kr5`); a = 0.5 → `…_pa0p5`;
   a = 0 → the current rows' covariance in the borrowed basis/standardization ("basis-only borrowing", `…_pa0`).
   Confidence scales stay AWM's (candidate-library pseudo-queries with the blended metric).
2. **Non-borrowing twins**: `ridge_main` (`…_rm1`: main metric ridge 1.0, early metric keeps 0.1, current-only fit) and the
   offline-only ablation `shared_beta` (`…_sb0p5`: `S_task ← (1−β)·S_task + β·mean(other tasks' S_w)`, each in its own
   standardized coordinates as C's `metric_cv.py`; main metric only).
3. **`grip_commit`** (`…_gc`; A-P2, stateful part). `v = Σ wn_i·sign(g_i[step 0])` over the kernel members (`gvote`).
   Served sign = sign(v) if |v| ≥ `grip_thr` (.8) else the previously EXECUTED sign. A change vs the executed sign needs
   |v| ≥ thr on this AND the previous decision (`grip_confirm="mag"`, default = the brief's wording; `"sign"` (`gcs`)
   additionally requires the previous vote to point to the new sign) and no executed sign change within the last
   `grip_dwell` (3) decisions (`gdwell` = decisions since the executed sign last changed). The previously executed sign and
   the change history are read from `q.hist_a_exec` (dim 6 of steps 0..4 of every executed chunk; mixed mode: the chunk
   after a MISS is the policy's) — never from this method's own output; only the previous decision's vote (a proposal)
   is per-episode instance state (`reset()`). Step 0: the top-1 row's step-0 sign. Served magnitude **±1** (LIBERO
   applies `sign()`; the executed chunk then carries the sign exactly, so the history read-back is clean; for these
   variants err ≡ gripper-sign-faithful err). Translation/rotation = AWM's kernel mean over all members;
   `grip_class_mean` (`gcm`) = mean over the served class only (≥ 3 members) — variant only, C measured it worse and so
   do we (+.02 err).
4. **`term_guard`** (`…_tg`; A-P3). Candidate rows with `lib_step ≥ ep_len − term_rows` (2; `tg1` = 1) are dropped unless
   `q.step ≥ term_late (0.9) × median library episode length of the task` AND the executed gripper has re-opened after
   having been closed in this episode (from `q.hist_a_exec`; open sign = the library's step-0 gripper sign: −1 π0.5,
   +1 GR00T). `term_gate="late"` (`tgp`): the step condition alone opens the gate. If the mask would empty the candidate
   set it is not applied. The mask is applied inside `os_score_all` too, so V6/V7/MixedJudge wrappers see the same set.
5. **Combos** and the **10× family** `AWM3(lib="big")` (+ `grip_commit` + `term_guard`).

G3 base contract: `os_score_all` (admissible = masked rows; S, aux `awm_d / awm_c / vis_v0 / vis_v1` over them),
`os_synth` (kernel mean + AWM insurance + grip_commit, same per-episode state as `query()`), `os_confidence` (aux indexed
by the admissible rows). `contract_check.py` **PASS** (Passthrough == query bit for bit) for `{prior_alpha .5, gc, tg}` on
pi05_spatial_cache (330 dec) and groot_l10_cache (395), `{lib big, gc, tg}` on pi05_spatial_cache (246) and groot_l10_inf
(293), `{gc, gcs, gcm, tg1}` on groot_spatial_inf (194) — `derived/r03/h1_trap/contract/`. `V7dc_iso__AWM3…pa0p5_gc_tg`
(H3's judge base) smokes PASS (same err as the base, AURC .365 vs .370).

Extras (8 new keys): `gvote`, `w_term` (always: kernel weight on the last-2 rows of library episodes), `gheld` / `gflip`
/ `gdwell` (grip_commit), `term_masked` / `term_open` (term_guard), `prior_alpha` (prior variants).

## Files

```
h1_trap/
  awm3.py                 AWM3 (this README's algorithms)
  _make_batch.py          VARIANTS (single source) -> batch.json; prints the name table
  batch.json              22 variants x all 8 cells (R2 format)            <- offline full round (coordinator)
  arms_pilot.json         13 emit_arms rows (pi05 sp 7 + l10 6, names r3p_p_<sp|l10>_<short>, <RUN> placeholder)
  arms_full_template.json the same rows + 10x + 2 alternates for all 4 cells (r3f_…), coordinator picks
  tools/equiv_check.py    bit-for-bit A vs B through the harness in-process runner
  tools/subset_run.py     run.run_cell on --every n / --tasks t episode subsets (dev numbers)
  tools/kpi.py            H1 KPI table from npz + store GT (err, gs-err, vote split, w_term_late, transitions, flips)
  tools/diag.py           err by third, why grip_commit held at teacher transitions, flips per episode
derived/r03/h1_trap/ (not in the repo): sub5/ (every-5th-episode runs, 22 variants x 4 cells), task6/ (all task-6
  episodes, pi05 sp + l10 cache), kpi/ (sub5.md/json, task6.md/json, smoke_summary.md, diag_*.json, readme_tables.md),
  smoke/ (88 official smoke transcripts), contract/ (contract + equivalence reports), cl/ (selftest + verify_logs),
  fits/ (prefit pickles + logs), timing/
```

Variant table (`_make_batch.py`; names are AWM3's):

| tag | name | group |
|---|---|---|
| ref | `AWM3_joint_cur_fcur_kr5` | current (== CL2 AWM; regression check) |
| ridge1 / sb05 | `…_rm1` / `…_sb0p5` | non-borrowing twins (sb05 offline-only) |
| gc / tg / gc_tg / gcs_tg | `…_gc` / `…_tg` / `…_gc_tg` / `…_gcs_tg` | current |
| a1 / a05 / a0 | `AWM3_joint_cur_fbig_kr5` / `…_pa0p5` / `…_pa0` | borrowed |
| a1_gc / a1_tg / a1_gc_tg | `…_fbig_kr5_gc` / `_tg` / `_gc_tg` | borrowed |
| a05_gc / a05_tg / a05_gc_tg | `…_fbig_kr5_pa0p5_gc` / `_tg` / `_gc_tg` | borrowed |
| a1_gcm_tg / a1_gc_tg1 / a1_gct06_tg / a1_gc_tgp | `…_gcm_tg` / `…_gc_tg1` / `…_gct0p6_tg` / `…_gc_tgp` | borrowed variants |
| big / big_gc_tg | `AWM3_joint_big_fbig_kr5` / `…_gc_tg` | 10× |

## Smoke (official `harness.smoke`, 5 episodes, `--no-ref`): 88 / 88 PASS, 0 WARN, 0 FAIL
All 22 variants on pi05_spatial_{inf,cache} + groot_l10_{inf,cache}; leak check (reversed episodes) identical everywhere;
`prev_hit` consistent; `bytes_per_entry` 580 (581 with term_guard: 1 bool/entry). Table: `derived/r03/h1_trap/kpi/smoke_summary.md`.

## Development numbers (`derived/r03/h1_trap/kpi/`; the closed-loop pilot is the gate, not these)

Subset = every 5th episode (100 episodes / cell; 3013 / 2205 / 7894 / 5527 decisions); task 6 = all 50 episodes.

### A. err | gripper-sign-faithful err | grip_mis | AURC (cache = stale regime, inf = fresh regime)

| variant | p-sp_cache | p-sp_inf | g-l10_cache | g-l10_inf |
|---|---|---|---|---|
| cur_fcur_kr5 (CL2) | .587 / .597 / .210 / .374 | .354 / .356 / .033 / .256 | .492 / .503 / .161 / .364 | .331 / .335 / .064 / .252 |
| cur_fcur_kr5_rm1 | .578 / .590 / .207 / .360 | .353 / .355 / .034 / .256 | .497 / .514 / .182 / .368 | .332 / .335 / .062 / .252 |
| cur_fcur_kr5_sb0p5 | .590 / .601 / .205 / .358 | .353 / .355 / .033 / .256 | .496 / .513 / .181 / .366 | .331 / .335 / .063 / .251 |
| cur_fcur_kr5_gc | .598 / .598 / .208 / .381 | .360 / .360 / .041 / .258 | .526 / .526 / .202 / .378 | .333 / .333 / .061 / .251 |
| cur_fcur_kr5_tg | .615 / .624 / .212 / .389 | .377 / .378 / .037 / .265 | .491 / .503 / .160 / .365 | .333 / .337 / .064 / .252 |
| cur_fcur_kr5_gc_tg | .625 / .625 / .208 / .395 | .382 / .382 / .042 / .267 | .525 / .525 / .202 / .378 | .335 / .335 / .061 / .252 |
| cur_fbig_kr5 (a1) | .561 / .568 / .202 / .355 | .351 / .353 / .032 / .251 | .482 / .500 / .167 / .358 | .331 / .334 / .062 / .253 |
| cur_fbig_kr5_pa0p5 (a05) | **.557** / .568 / .203 / **.349** | .351 / .354 / .033 / .251 | **.474** / .490 / .159 / .357 | .331 / .334 / .063 / .251 |
| cur_fbig_kr5_pa0 | .584 / .596 / .211 / .371 | .354 / .356 / .033 / .256 | .495 / .508 / .168 / .366 | .331 / .335 / .064 / .250 |
| cur_fbig_kr5_gc | .571 / .571 / .209 / .360 | .358 / .358 / .041 / .253 | .518 / .518 / .201 / .372 | .331 / .331 / .056 / .252 |
| cur_fbig_kr5_tg | .598 / .605 / .202 / .372 | .375 / .376 / .037 / .259 | .483 / .501 / .167 / .359 | .333 / .336 / .062 / .253 |
| cur_fbig_kr5_gc_tg | .608 / .608 / .209 / .377 | .379 / .379 / .041 / .262 | .519 / .519 / .201 / .373 | .333 / .333 / .057 / .253 |
| cur_fbig_kr5_pa0p5_gc_tg | .597 / .597 / .209 / .370 | .380 / .380 / .041 / .262 | .512 / .512 / .201 / .373 | .335 / .335 / .061 / .252 |
| cur_fbig_kr5_gcm_tg | .629 / .629 / .209 / .382 | .386 / .386 / .041 / .263 | .522 / .522 / .201 / .371 | .333 / .333 / .057 / .253 |
| cur_fbig_kr5_gc_tg1 | .584 / .584 / .209 / .366 | .364 / .364 / .041 / .256 | .519 / .519 / .201 / .373 | .332 / .332 / .056 / .253 |
| cur_fbig_kr5_gc_tgp | .581 / .581 / .209 / .366 | .366 / .366 / .041 / .256 | .519 / .519 / .201 / .373 | .333 / .333 / .057 / .253 |
| big_fbig_kr5 (10×) | .505 / .520 / .199 / .294 | .269 / .271 / .028 / .195 | .443 / .457 / .148 / .315 | .258 / .259 / .060 / .204 |
| big_fbig_kr5_gc_tg | .545 / .545 / .208 / .313 | .284 / .284 / .039 / .201 | .483 / .483 / .198 / .335 | .260 / .260 / .062 / .204 |

(`gcs_tg` ≡ `gc_tg` on the cache cells, .336 vs .335 on g-l10_inf; `gct0p6_tg` ≡ `gc_tg` within .002; full tables in `kpi/sub5.md`.)

### B. Trap KPIs

| variant | cell | vote split \|v\|<.5 / <.8 | w_term_late | grip_mis @ teacher transitions (n) | served flips [teacher] | held | term_open |
|---|---|---|---|---|---|---|---|
| cur_fcur_kr5 | p-sp_cache | .056 / .116 | .380 | .553 (806) | .145 [.166] | | |
| cur_fcur_kr5_gc | p-sp_cache | .056 / .116 | .380 | .770 (806) | .004 [.166] | .141 | |
| cur_fcur_kr5_tg | p-sp_cache | .062 / .126 | .166 | .551 (806) | .153 [.166] | | .208 |
| cur_fbig_kr5 (a1) | p-sp_cache | .062 / .133 | .403 | .561 (806) | .126 [.166] | | |
| cur_fbig_kr5_pa0p5 | p-sp_cache | .058 / .118 | .430 | .553 (806) | .129 [.166] | | |
| cur_fbig_kr5_gc_tg | p-sp_cache | .066 / .142 | .199 | .770 (806) | .005 [.166] | .121 | .208 |
| cur_fbig_kr5_gc_tgp | p-sp_cache | .063 / .135 | .364 | .770 (806) | .005 [.166] | .121 | .380 |
| big_fbig_kr5 | p-sp_cache | .061 / .150 | .560 | .514 (806) | .134 [.166] | | |
| big_fbig_kr5_gc_tg | p-sp_cache | .063 / .145 | .316 | .770 (806) | .003 [.166] | .133 | .206 |
| cur_fcur_kr5 | p-sp_inf | .018 / .048 | .294 | .377 (167) | .021 [.016] | | |
| cur_fcur_kr5_gc | p-sp_inf | .018 / .048 | .294 | .526 (167) | .006 [.016] | .015 | |
| cur_fbig_kr5_gc_tg | p-sp_inf | .047 / .096 | .017 | .533 (167) | .004 [.016] | .024 | .057 |
| cur_fbig_kr5_gc_tgp | p-sp_inf | .033 / .072 | .190 | .533 (167) | .004 [.016] | .020 | .143 |
| cur_fcur_kr5 | g-l10_cache | .212 / .356 | .122 | .477 (1920) | .131 [.171] | | |
| cur_fcur_kr5_gc | g-l10_cache | .212 / .356 | .122 | .825 (1920) | .002 [.171] | .130 | |
| cur_fbig_kr5 (a1) | g-l10_cache | .260 / .403 | .111 | .433 (1920) | .155 [.171] | | |
| cur_fbig_kr5_pa0p5 | g-l10_cache | .252 / .392 | .113 | .436 (1920) | .141 [.171] | | |
| cur_fbig_kr5_gc_tg | g-l10_cache | .260 / .402 | .097 | .823 (1920) | .002 [.171] | .154 | .383 |
| cur_fcur_kr5 | g-l10_inf | .026 / .040 | .078 | .437 (538) | .037 [.052] | | |
| cur_fcur_kr5_gc | g-l10_inf | .026 / .040 | .078 | .543 (538) | .017 [.052] | .020 | |
| cur_fbig_kr5_gc_tg | g-l10_inf | .026 / .047 | .044 | .548 (538) | .011 [.052] | .023 | .136 |

### C. Task 6, all episodes: err | vote split |v|<.5 | <.8 (spatial: 1488 decisions; l10: 4791)

| variant | spatial task 6 | l10 task 6 |
|---|---|---|
| cur_fcur_kr5 (CL2) | .600 / **.174** / .267 | .620 / .224 / .437 |
| cur_fcur_kr5_rm1 | .581 / .108 / .310 | .561 / .255 / .499 |
| cur_fcur_kr5_sb0p5 | .574 / .121 / .297 | .545 / .264 / .473 |
| cur_fbig_kr5 (a1) | .566 / **.059** / .256 | .542 / **.403** / .560 |
| cur_fbig_kr5_pa0p5 (a05) | .550 / .149 / .266 | .532 / .329 / .510 |
| cur_fbig_kr5_pa0 | .594 / .150 / .253 | .587 / .263 / .507 |
| cur_fbig_kr5_tg / _gc_tg | .583 / .058 / .217 | .543 / .403 / .561 |
| cur_fbig_kr5_pa0p5_gc_tg | .580 / .142 / .224 | .550 / .329 / .511 |
| big_fbig_kr5 (10×) | .517 / .029 / .046 | .478 / .256 / .440 |

Reading (offline, so only the parts that do not depend on the executed history are trustworthy):
- **Borrowed prior**: a = 1 removes the spatial task-6 severe split (.174 → .059) and cuts spatial stale err by .026, but
  doubles the l10 task-6 severe split (.224 → .403; C's 22.9 → 45.0 %). a = .5 is the best offline on every cache cell
  (err −.030 / −.018 vs CL2, AURC −.025) yet only hedges task 6 (spatial .149, l10 .329). Basis-only borrowing (a = 0)
  changes little (−.003). Fresh-regime err is unchanged by any prior (continuity dominates).
- **Non-borrowing twins**: ridge 1.0 −.009 spatial / +.005 GR00T l10, does not fix the split (spatial task 6 .108/.310);
  shared_beta .5 +.004 / +.005 (C's finding reproduced); both leave l10 task 6 worse.
- **term_guard** costs +.028 / +.023 (sp cache / inf) and ~0 on GR00T l10; the cost sits in the late third (ref .725 →
  .783; mid .688 → .716; early unchanged) where B0's trace states are legitimately terminal. `w_term_late` .38 → .17.
  The progress-only gate (`tgp`) recovers most of it (.581 vs .608 with a = 1 + gc) — in π0.5 spatial the library
  episodes end with the gripper still CLOSED (last-row sign mean +.73 with open = −1), so the "re-opened" clause never
  opens the gate in a normal spatial episode (selftest: `term_open` 0 / 604 decisions); it opens only after a
  close→open chatter (offline .21 of B0's decisions = the failing episodes). Last-1 rows (`tg1`) halves the cost too.
- **grip_commit** cannot be evaluated on the cache trace: its state is B0's executed history (chatter), so it almost
  never flips there (.004 vs the teacher's .166) and its offline gripper numbers are identical across all priors. On the
  inf cells (clean policy history) it holds on 1.5–2.4 % of decisions; at teacher transitions the kernel vote itself
  already lags the teacher in ~75 % of the cases (vote sign = teacher's at only 24 % of the 33 spatial transitions; AWM's
  mean is wrong there too), gc adds a hold on 6–7 of 33 (83 % because |v| < .8 at that decision). With its OWN history
  (plugin selftest, 20 spatial episodes, recorded observations) it serves exactly **1.0 gripper flip / episode**, holds
  7.4 % of decisions, vote split .076 / .123. `grip_confirm="sign"` is far too conservative (.01 flips / episode on the
  inf cell) → "mag" stays the default. thr .6 ≈ thr .8 offline. Class-mean (`gcm`) +.02 err (C's result reproduced).
- **10×**: big .505 / .443 (R2 full-cell .501 / .433); +gc+tg .545 / .483.

## Closed-loop readiness

- `closed_loop/selftest.py` (pure cache, `AWM3(prior_alpha=.5, grip_commit, term_guard)`, pi05_spatial_cache, 20 episodes,
  yaml `r02_g50/config/oscl50_p_sp_cl2.yaml`): 604 decisions, `exec_ok` 604 / 604, `winner_matches_log`, online ==
  offline on the compared decisions 1.0 (topk / scores / conf / synth / extras), native shadow == recorded 1.0;
  `verify_logs.py` (offline replay of the LOGGED inputs = the method's own served chunks as history): **604 / 604
  decisions bit-identical** (topk, scores, conf, lib, synth, extras), executed == selected 604 / 604, q_us p50 2.6 ms in
  the plugin stack (incl. native shadow). `derived/r03/h1_trap/cl/`.
- **Prefit** (`python -m exp.offline_search.closed_loop.plugin --os-method … --os-fit-artifact …`, single-threaded BLAS):
  current π0.5 sp / l10: 0.6–1.9 s / 4.3 s, **21.9 / 26.0 MB** (ref, a1 and a05_gc_tg pickles differ by < 100 B: the
  borrowed fit stores the same 64-column bases + per-task W); GR00T sp / l10 22.8 / 28.1 MB. 10× (`lib=big` + gc + tg):
  2.8 / 11.5 s, **47.1 / 94.3 MB** π0.5, 58.2 / 117.5 MB GR00T. Prior fits need ~1.3 GB RAM while fitting (big-row
  features), the pickle does not.
- **ms/query** (`subset_run.py --timing`, 300-query single-thread pass, one process per physical core, 4 at once):
  current π0.5-l10 1.35 (CL2) / 1.41 (a05_gc_tg); GR00T-l10 2.49 / 1.94; 10× π0.5-l10 1.90 (big) / 1.79 (big_gc_tg);
  GR00T-l10 2.37 / 2.12 — all ≲ 2.5 ms (harness smoke at 8 concurrent processes: 3.4–4.9 ms).
- Arms: `arms_pilot.json` (spatial: cl2ref = `AWM(lib=current,kref=5)` re-run, a1, a05, ridge1, a1_gc, a1_tg, a1_gc_tg;
  l10: cl2ref, a1, a05, a05_gc, a05_tg, a05_gc_tg; names ≤ 19 chars), `arms_full_template.json` (rows for all 4 cells +
  10× + alternates `a1_gc_tgp`, `a1_gct06_tg`).

## Library scale (protocol §9)

| | π0.5 sp | π0.5 l10 | GR00T sp | GR00T l10 |
|---|---|---|---|---|
| current: episodes / entries | 49 / 1,018 | 50 / 2,640 | 50 / 1,063 | 50 / 2,645 |
| 10×: episodes / entries | 500 / 10,909 | 500 / 29,472 | 500 / 11,751 | 500 / 29,631 |
| bytes / entry (AWM3 codes, f32) | 580 (581 with term_guard) | same | same | same |
| fixed / suite | 17.0 MB PCA (2 × 64 cols) + ~0.15 MB per task (W, W0, A) | | | |
| fit pickle, current | 21.9 MB | 26.0 MB | 22.8 MB | 28.1 MB |
| fit pickle, 10× | 47.1 MB | 94.3 MB | 58.2 MB | 117.5 MB |
| deployed pkl (262 KB/entry) | 431 MB | 1,103 MB | 429 MB | 1,068 MB |

Borrowed variants (`prior_alpha`) are **50-episode online storage with 500-episode training information**: same entries,
same bytes/entry, same pickle size as the current fit; only the metric's statistics come from the big library.

## Three-layer statement (protocol §9)

- **Synthesis layer**: `grip_commit` (served gripper sign, ±1 on steps 0..4) and `grip_class_mean` change only how the
  selected set is turned into an action. Offline they are neutral to slightly negative on gs-err (+.001 sp, +.02 GR00T-l10
  cache where the history is B0's); their purpose (no chatter, one transition per grasp) is only visible with the method's
  own executed history (selftest: 1.0 flip / episode) → the pilot decides.
- **Method layer at fixed library** (same 50-episode candidates): `prior_alpha` (borrowed information: −.026 / −.030 sp
  stale, −.010 / −.018 GR00T-l10 stale for a = 1 / .5; spatial task-6 split .174 → .059 at a = 1), `ridge_main`
  (−.009 sp, non-borrowing), `shared_beta` (no gain), `term_guard` (candidate mask; +.02–.03 offline, concentrated late).
- **Library layer**: `lib="big"` (10×, 500 episodes): −.08 / −.05 stale err vs CL2, task-6 split .029; the 10× + gc + tg
  arm isolates the synthesis/method switches at the larger library (offline +.04 there).
- Every `fbig` variant is labelled **borrowed big-library information**; its non-borrowing twins are `ref`, `rm1`,
  `sb0p5`, `gc`, `tg`, `gc_tg`.

## Caveats

- Offline err is not the gate (brief). The offline trace's executed history is B0's, so `grip_commit`'s and
  `term_guard`'s history-dependent behaviour is measured on the wrong history there (see B); only the selftest replay
  (own history, recorded observations) shows the intended one-transition-per-grasp behaviour, and neither can show
  trajectory-level effects.
- The default `term_guard` gate (`both`) practically never opens in π0.5 spatial (library episodes end closed): terminal
  rows are masked for the whole normal episode. If the pilot shows late-phase stalls, run the `tgp` (progress-only)
  alternate; the arms in `arms_pilot.json` use the brief's specification.
- `grip_commit` adds one decision of gripper latency in ~20 % of clean transitions (|v| < .8 at the transition decision);
  the kernel vote itself lags the teacher at most transitions anyway. `grip_thr=.6` is in the batch / template.
- `prior_alpha=1` on l10 doubles the task-6 severe vote split (as C measured); `a = .5` is the hedge, `a = 1` the spatial
  fix — the pilot compares both per suite.
- AWM's `insure` / `hyst` cannot be combined with `grip_commit` (raises); `norm_cap` can.
- Inherited: plain `AWM(lib="big")` breaks step-0 exact ties arbitrarily (fails contract_check on 1 / 246 decisions);
  AWM3 breaks them by lower row, so `AWM3(lib="big")` and `AWM(lib="big")` differ on such tied decisions only (none
  observed in the 367-decision equivalence check on pi05_l10_cache, 1 in 246 on pi05_spatial_cache in the wrapper check).
- Static smoke scan: the class source mentions `ep_len` (the LIBRARY's episode lengths, `Lc.ep_len`, as g3_core does) —
  no query-side forbidden field is touched (smoke `static` PASS on all 88 runs).
