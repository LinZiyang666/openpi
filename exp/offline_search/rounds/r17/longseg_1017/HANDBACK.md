# Decision-head clocks: withdrawal and fresh reruns (2026-10-17)

**Owner decision-clock recalibration (2026-10-18):** 840 potential level definitions and 42 distinct C10 Q3 artifacts have been recalibrated. Level IR is M/N segment heads (absolute prediction tolerance .005); latency-weighted IR remains separate. C5 artifacts and independent queued units are unchanged. 2 tolerance exceptions are explicitly listed in [RECALIBRATION.md](RECALIBRATION.md). The coordinator alone reviews and writes `[recal] released`; this agent has not released or launched anything.

CPU-only handback. No new server, worker, driver, chain or GPU job was launched. Only recalibration-dependent, never-prepared gated definitions were regenerated under the owner ruling. Active/independent campaigns and chain scripts were not edited. The new namespace is `fa_ls_*`; `NAMES.json` maps withdrawn lanes to fresh definitions. Do not launch the withdrawn NAMEs or resume their old continuations.

The source model runtimes, worker archives, retrieval metric/whitening/PCA and correctors stay unchanged. Only C10 Q3 artifacts and periodic gaps are recalibrated under the owner ruling. New bootstraps pin the corrected server methods; new driver overlays preserve their decision metadata. Every new job requests timan1 **GPU 2** and timan107, with the original total W. The existing capacity allocator records any launch-time redistribution. timan108 is absent. No worker source/archive changes or worker rsync update is needed.

## Corrected execution and calibration

`ConnectionPolicy` counts segment heads separately from deliveries. A pending segment delivers all remaining tails; a tail extracts no new state and evaluates no retrieval/recipe/gate/knob/stall/corrector/model logic. Histories, semantic RNG index, corrector step, periodic schedule and Q3 schedules advance on heads. Frozen worker delivery IDs remain the ACK fence. `decision_index`, `delivery_index`, `env_step_index`, `segment_head`, and the actual/wire seeds survive frozen-worker transport in `server_timing`; new drivers promote the clock fields.

C10 Q3 is refitted on successful-library segment-head sequences using the original empirical LOEO q90 residual/q10 advance procedure. Its temporal units are exec_steps=commit_controls=1 decision, with the original window fraction, minimum windows, K and reference requirements. This removes the former quantum conversion. No new online FAR or conformal-coverage guarantee is claimed. All C5 artifacts remain unchanged. See RECALIBRATION.md for actual per-task parameters and achieved empirical metrics.

The same-step native-length pure result remains the SR target. The fresh `pure + long` controls measure the only admissible pure fallback inside the complete system. `RESULTS.json` stitches those C10 outcomes for tasks selecting pure, leaves them missing until measured, and reports whether their actual per-task tolerance holds. It never silently inserts C5 episodes into the C10 system. The native-helper stitch is retained under `native_length_fallback_diagnostic`, not as the final result.

## Audit and backup

433 definitions inventoried; 166 logs have empirical evidence. 38 have sampled midpoint cuts; four additional uninterrupted ch/chr lanes used delivery-count recipe clocks and are withdrawn for the decoupling. “0%” means no cuts observed in the stated sample, not a full-file proof.

The final sampler uses one reader, up to eight windows totaling 16 MiB / 16,000 lines per file, 4 MiB/s, five seconds between files, and `/proc/pressure/io` full avg10 <=20. Already completed bounded-prefix scans (up to 32 MiB) and full audits from before the I/O correction were reused and labeled; no full log was reread. `AUDIT.tsv` lists all affected and unaffected definitions and sample sizes.

| Withdrawn run | Sample rows | Sample MiB | Midpoint cut % | Reason |
|---|---:|---:|---:|---|
| `csdp_1504` | 5,508,696 | 13081.77 | 2.554 | midpoint cuts |
| `csfw_k3` | 2,268,788 | 5352.38 | 1.569 | midpoint cuts |
| `csfw_k3w4` | 425,535 | 1004.31 | 0.000 | delivery-count recipe clock on long ch/chr heads |
| `csfw_k3w5` | 934,961 | 2204.80 | 2.191 | midpoint cuts |
| `csfw_k3w6` | 2,213,699 | 5218.03 | 3.357 | midpoint cuts |
| `csgr_h5` | 1,204,434 | 3140.10 | 2.070 | midpoint cuts |
| `csgr_h6` | 4,235,196 | 11002.04 | 3.162 | midpoint cuts |
| `csl10d` | 183,067 | 687.98 | 0.709 | midpoint cuts |
| `csl10f` | 201,085 | 750.87 | 1.127 | midpoint cuts |
| `csmdf1010` | 2,538 | 9.26 | 1.075 | midpoint cuts |
| `csmf1010` | 2,698 | 9.65 | 1.286 | midpoint cuts |
| `csztd_r2_c383_r2_e5b0` | 596,546 | 2258.77 | 1.185 | midpoint cuts |
| `csztd_r3_l48` | 3,112,222 | 11866.18 | 3.409 | midpoint cuts |
| `dpc3464789869` | 213,232 | 773.67 | 0.000 | delivery-count recipe clock on long ch/chr heads |
| `dpc8713063796_r2_d1dd` | 208,412 | 818.99 | 6.643 | midpoint cuts |
| `dpmk1_fulx` | 4,058,746 | 9712.23 | 41.050 | midpoint cuts |
| `dpmk1_redx` | 4,036,454 | 9645.83 | 40.850 | midpoint cuts |
| `fa_dp_f_f_o` | 8,801 | 32.00 | 0.763 | midpoint cuts |
| `fa_dp_f_u_o` | 8,442 | 32.00 | 3.310 | midpoint cuts |
| `fa_dp_r_f_l` | 430,422 | 1753.94 | 48.243 | midpoint cuts |
| `fa_dp_r_u_l` | 8,609 | 32.00 | 18.686 | midpoint cuts |
| `fa_fw_f_f_o` | 532,072 | 2323.48 | 4.414 | midpoint cuts |
| `fa_fw_f_u_o` | 7,615 | 32.00 | 2.365 | midpoint cuts |
| `fa_fw_r_u_l` | 6,411 | 31.99 | 33.205 | midpoint cuts |
| `fastwamc5232306668` | 3,620 | 15.98 | 0.000 | delivery-count recipe clock on long ch/chr heads |
| `fastwamc5481985003` | 4,057 | 15.98 | 2.870 | midpoint cuts |
| `fastwamc6066547406` | 3,181 | 15.97 | 2.655 | midpoint cuts |
| `ffk1015_dp_libero_reduced_off` | 4,330 | 15.99 | 2.746 | midpoint cuts |
| `ffk1015_dp_mt50_reduced_descent` | 6,737 | 15.99 | 47.480 | midpoint cuts |
| `ffk1015_dp_mt50_reduced_descentc1` | 6,687 | 15.99 | 62.840 | midpoint cuts |
| `ffk1015_fastwam_libero_reduced_off` | 4,167 | 15.97 | 3.746 | midpoint cuts |
| `ffk1015_fastwam_mt50_reduced_descent` | 6,819 | 15.99 | 46.197 | midpoint cuts |
| `ffk1015_groot_mt50_full_descent` | 6,088 | 15.99 | 52.337 | midpoint cuts |
| `ffk1015_groot_mt50_reduced_descent` | 6,103 | 15.99 | 39.967 | midpoint cuts |
| `fsreddp2` | 4,272 | 15.98 | 7.183 | midpoint cuts |
| `fsredfw` | 4,206 | 15.97 | 5.038 | midpoint cuts |
| `fwmk1_fulx` | 6,733 | 15.99 | 36.038 | midpoint cuts |
| `fwmk1_redx` | 6,762 | 15.99 | 44.132 | midpoint cuts |
| `grmk1_ful` | 6,124 | 15.99 | 30.414 | midpoint cuts |
| `grmk1_red` | 6,123 | 15.99 | 41.222 | midpoint cuts |
| `hsdp_0725` | 428 | 1.01 | 0.000 | delivery-count recipe clock on long ch/chr heads |
| `knobsmokedp1355` | 788 | 2.92 | 26.906 | midpoint cuts |

Backup verification: **PASS; 911 rename-only entries; at least 314,396,566,371 bytes (292.805 GiB)**. Sizes of 133 directory subtrees are shallow lower bounds, avoiding a cold inode walk. Same-filesystem device/inode checks verified every move. The manifest and dry run are `/home/weiland/projects/_trash_longseg_20261017/MANIFEST.json` and `DRY_RUN.tsv`; `MOVE_JOURNAL.jsonl` and `VERIFICATION.json` record execution/verification. This covers local collected outcomes; no remote outcome directory was moved through tether.

Definitions, plans, archives, assets, libraries/fits/stall artifacts, pure references and ready/admission records were kept. Moved derived artifacts include the GR00T/DP/Fast-WAM MT50 descent outputs, affected full-system/filtered outputs, DP/Fast-WAM formal_A rows/off decisions, cache rows and website blobs. Unaffected JSON rows were retained. Invalid continuation registrations were withdrawn; Cosmos and pure-Cache registrations stay intact. `knobsmokedp1355` is **trash only, no replacement and no queue item**, by the owner ruling.

## Fresh definitions

60 built campaigns: 1,637,572 episodes. Eight deferred LIBERO level units add at most 240,000; total fresh rerun bound **1,877,572** before the off-cell skips. Of this, 19,000 are ten library-independent pure+long controls and 10,500 are six new native-length pure references. The historical resume cohorts contain no old outcomes; canonical task/init/seed/bundle/recipe/YAML signatures cover each full source lane with zero overlap (`source_cohorts` in `CAMPAIGNS.json`). Scheduler UID namespaces can change for a cohort; scientific identities and initial states do not.

**Fast-WAM MT50 cache variants:** `fa_ls_v02_s`: R3, sizes 250/500/1000, 60,000 episodes. `fa_ls_v02_b`: R2, sizes 1500/2000/2500, 60,000. This is the owner-selected RUN_CONFIGS memory-cap split, recorded in each campaign and FFK definition. `OWNER_SPLIT_PROOF.log` verifies the canonical source plans.

| NAME | Episodes | GPU | R/B | MPS | Requested workers | Stage |
|---|---:|---|---|---|---|---|
| `fa_ls_dp_r_u_m` | 90,000 | RTX 4090 | 4/16 | on | timan1 GPU2×60, timan107×60 | descent |
| `fa_ls_dp_r_f_m` | 90,000 | RTX 4090 | 4/16 | on | timan1 GPU2×60, timan107×60 | descent |
| `fa_ls_dp_r_p_m` | 2,500 | RTX 4090 | 4/16 | on | timan1 GPU2×60, timan107×60 | long |
| `fa_ls_dp_r_b_m` | 2,500 | RTX 4090 | 4/16 | on | timan1 GPU2×60, timan107×60 | pure |
| `fa_ls_dp_f_u_m` | 90,000 | RTX 4090 | 8/32 | on | timan1 GPU2×60, timan107×60 | descent |
| `fa_ls_dp_f_f_m` | 90,000 | RTX 4090 | 8/32 | on | timan1 GPU2×60, timan107×60 | descent |
| `fa_ls_dp_f_p_m` | 2,500 | RTX 4090 | 8/32 | on | timan1 GPU2×60, timan107×60 | long |
| `fa_ls_dp_f_b_m` | 2,500 | RTX 4090 | 8/32 | on | timan1 GPU2×60, timan107×60 | pure |
| `fa_ls_gr_r_u_m` | 90,000 | H100 | 11/1 | on | timan1 GPU2×20, timan107×20 | descent |
| `fa_ls_gr_r_f_m` | 90,000 | H100 | 11/1 | on | timan1 GPU2×20, timan107×20 | descent |
| `fa_ls_gr_r_p_m` | 2,500 | H100 | 11/1 | on | timan1 GPU2×20, timan107×20 | long |
| `fa_ls_gr_f_u_m` | 90,000 | H100 | 11/1 | on | timan1 GPU2×20, timan107×20 | descent |
| `fa_ls_gr_f_f_m` | 90,000 | H100 | 11/1 | on | timan1 GPU2×20, timan107×20 | descent |
| `fa_ls_gr_f_p_m` | 2,500 | H100 | 11/1 | on | timan1 GPU2×20, timan107×20 | long |
| `fa_ls_fw_r_u_m` | 90,000 | H100 | 2/16 | on | timan1 GPU2×60, timan107×60 | descent |
| `fa_ls_fw_r_f_m` | 90,000 | H100 | 2/16 | on | timan1 GPU2×60, timan107×60 | descent |
| `fa_ls_fw_r_p_m` | 2,500 | H100 | 2/16 | on | timan1 GPU2×60, timan107×60 | long |
| `fa_ls_fw_f_u_m` | 90,000 | H100 | 2/16 | on | timan1 GPU2×60, timan107×60 | descent |
| `fa_ls_fw_f_f_m` | 90,000 | H100 | 2/16 | on | timan1 GPU2×60, timan107×60 | descent |
| `fa_ls_fw_f_p_m` | 2,500 | H100 | 2/16 | on | timan1 GPU2×60, timan107×60 | long |
| `fa_ls_dp_r_u_o_l` | 6,000 | RTX 4090 | 4/8 | on | timan1 GPU2×24, timan107×24 | off |
| `fa_ls_dp_r_f_o_l` | 6,000 | RTX 4090 | 4/8 | on | timan1 GPU2×24, timan107×24 | off |
| `fa_ls_dp_f_u_o_l` | 6,000 | RTX 4090 | 4/8 | on | timan1 GPU2×24, timan107×24 | off |
| `fa_ls_dp_f_f_o_l` | 6,000 | RTX 4090 | 4/8 | on | timan1 GPU2×24, timan107×24 | off |
| `fa_ls_fw_r_u_o_l` | 6,000 | RTX 4090 | 2/16 | on | timan1 GPU2×35, timan107×35 | off |
| `fa_ls_fw_r_f_o_l` | 6,000 | RTX 4090 | 2/16 | on | timan1 GPU2×35, timan107×35 | off |
| `fa_ls_fw_f_u_o_l` | 6,000 | H100 | 2/16 | on | timan1 GPU2×42, timan107×43 | off |
| `fa_ls_fw_f_f_o_l` | 6,000 | H100 | 2/16 | on | timan1 GPU2×42, timan107×43 | off |
| `fa_ls_dp_r_p_l` | 1,000 | RTX 4090 | 4/8 | on | timan1 GPU2×24, timan107×24 | long |
| `fa_ls_dp_r_b_l` | 1,000 | RTX 4090 | 4/8 | on | timan1 GPU2×24, timan107×24 | pure |
| `fa_ls_dp_f_p_l` | 1,000 | RTX 4090 | 4/8 | on | timan1 GPU2×24, timan107×24 | long |
| `fa_ls_dp_f_b_l` | 1,000 | RTX 4090 | 4/8 | on | timan1 GPU2×24, timan107×24 | pure |
| `fa_ls_fw_r_p_l` | 1,000 | RTX 4090 | 2/16 | on | timan1 GPU2×35, timan107×35 | long |
| `fa_ls_fw_f_p_l` | 1,000 | H100 | 2/16 | on | timan1 GPU2×42, timan107×43 | long |
| `fa_ls_fw_f_b_l` | 1,000 | H100 | 2/16 | on | timan1 GPU2×42, timan107×43 | pure |
| `fa_ls_v01_0` | 120,000 | RTX 4090 | 4/8 | on | timan1 GPU2×25, timan107×25 | cache_variants |
| `fa_ls_v02_s` | 60,000 | H100 | 3/16 | on | timan1 GPU2×60, timan107×60 | cache_variants |
| `fa_ls_v02_b` | 60,000 | H100 | 2/16 | on | timan1 GPU2×60, timan107×60 | cache_variants |
| `fa_ls_v03_0` | 24,406 | H100 | 8/1 | on | timan1 GPU2×30, timan107×30 | cache_variants |
| `fa_ls_v03_1` | 95,594 | H100 | 8/1 | on | timan1 GPU2×20, timan107×20 | cache_variants |
| `fa_ls_v04_0` | 4,001 | RTX 4090 | 2/8 | on | timan1 GPU2×16, timan107×16 | cache_variants |
| `fa_ls_v04_1` | 513 | RTX 4090 | 2/8 | on | timan1 GPU2×16, timan107×16 | cache_variants |
| `fa_ls_v04_2` | 1,734 | RTX 4090 | 2/8 | on | timan1 GPU2×16, timan107×16 | cache_variants |
| `fa_ls_v04_3` | 1,752 | RTX 4090 | 4/8 | on | timan1 GPU2×16, timan107×17 | cache_variants |
| `fa_ls_v05_0` | 4,023 | H100 | 2/16 | on | timan1 GPU2×24, timan107×24 | cache_variants |
| `fa_ls_v05_1` | 288 | H100 | 2/16 | on | timan1 GPU2×24, timan107×24 | cache_variants |
| `fa_ls_v05_2` | 397 | H100 | 2/16 | on | timan1 GPU2×24, timan107×24 | cache_variants |
| `fa_ls_v05_3` | 347 | H100 | 2/16 | on | timan1 GPU2×24, timan107×24 | cache_variants |
| `fa_ls_v05_4` | 2,945 | H100 | 2/16 | on | timan1 GPU2×24, timan107×24 | cache_variants |
| `fa_ls_v06_0` | 32 | RTX 4090 | 2/8 | on | timan1 GPU2×16, timan107×16 | cache_variants |
| `fa_ls_v07_0` | 32 | H100 | 2/16 | on | timan1 GPU2×24, timan107×24 | cache_variants |
| `fa_ls_v08_1` | 40,000 | H100 | 2/16 | on | timan1 GPU2×42, timan107×43 | cache_variants |
| `fa_ls_v09_0` | 6,350 | RTX 4090 | 4/8 | on | timan1 GPU2×30, timan107×30 | cache_variants |
| `fa_ls_v09_1` | 33,650 | RTX 4090 | 4/8 | on | timan1 GPU2×24, timan107×24 | cache_variants |
| `fa_ls_v10_0` | 6,000 | RTX 4090 | 4/8 | on | timan1 GPU2×24, timan107×24 | cache_variants |
| `fa_ls_v11_0` | 6,000 | RTX 4090 | 1/8 | off | timan1 GPU2×11, timan107×11 | cache_variants |
| `fa_ls_v12_0` | 6,000 | RTX 4090 | 4/8 | on | timan1 GPU2×24, timan107×24 | cache_variants |
| `fa_ls_v13_0` | 6,000 | RTX 4090 | 1/8 | off | timan1 GPU2×11, timan107×11 | cache_variants |
| `fa_ls_v14_0` | 8 | RTX 4090 | 1/8 | on | timan1 GPU2×4, timan107×4 | cache_variants |
| `fa_ls_dp_r_b_v` | 2,500 | RTX 4090 | 4/8 | on | timan1 GPU2×25, timan107×25 | pure |

| Model | Benchmark | Mode | Fresh episode bound incl. controls/studies |
|---|---|---|---:|
| dp | libero | full | 122,032 |
| dp | libero | reduced | 86,000 |
| dp | mt50 | full | 185,008 |
| dp | mt50 | reduced | 307,500 |
| fastwam | libero | full | 122,032 |
| fastwam | libero | reduced | 85,000 |
| fastwam | mt50 | full | 182,500 |
| fastwam | mt50 | reduced | 302,500 |
| groot | mt50 | full | 182,500 |
| groot | mt50 | reduced | 302,500 |

## Same-GPU pure references and timing

`REFERENCES.json` inventories both GPUs for every affected model/benchmark/mode, with counts, R/B, descriptor equality and measured rates. `REFERENCE_SELECTION.json` is authoritative for the fresh main grids; `REFERENCE_ACTUAL_IDENTITY.json` also verifies actual ready.json model identities for all five reused references. The new native pure jobs are `fa_ls_dp_{r,f}_b_m` (2,500 each), `fa_ls_dp_{r,f}_b_l` (1,000 each), `fa_ls_fw_f_b_l` (1,000), and `fa_ls_dp_r_b_v` (2,500 for the migrated cache study). Each LIBERO 1,000 is 500 per suite. The pure+long jobs end `_p_m` / `_p_l` and use the same serving as their grids.

| Model / benchmark / mode | H100 existing native pure | RTX 4090 existing native pure |
|---|---|---|
| dp / libero / full | Missing; fresh cost 1000 episodes | `hs5dp_ctl_142812` (different sealed identity), `hs7dpctl_full` (different sealed identity) |
| dp / libero / reduced | Missing; fresh cost 1000 episodes | `hs5dp_ddim10s0` (different sealed identity), `hs7dpctl_ddim10` (different sealed identity) |
| dp / metaworld / full | `tdp7h1_full2` (identity matches) | `tdp7_0729` (different sealed identity) |
| dp / metaworld / reduced | `tdp7d10_0851` (different sealed identity) | Missing; fresh cost 2500 episodes |
| fastwam / libero / full | `hs6fw_ctl_143332` (different sealed identity), `hs6fw_res_152931` (different sealed identity) | Missing; fresh cost 1000 episodes |
| fastwam / libero / reduced | `hs6fw_k1` (different sealed identity) | `hs7fwk1_4090c` (identity matches) |
| fastwam / metaworld / full | `tfw7full_0944` (identity matches) | Missing; fresh cost 2500 episodes |
| fastwam / metaworld / reduced | `tfw7k3_1051` (identity matches) | Missing; fresh cost 2500 episodes |
| groot / metaworld / full | `tgr7h1_full` (identity matches) | `tgr7_0502` (different sealed identity) |
| groot / metaworld / reduced | `tgr7h1_k2` (identity matches) | `tgr7k2_1044` (different sealed identity) |

Pure references for the unaffected legacy VLA and Cosmos queue units are listed in `UNAFFECTED_REFERENCES.json`; their unchanged serving/ref selection is reused. Existing GR00T/Fast-WAM MT50 pure references have fewer replicas (R8 vs R11, R3 vs R2 respectively), as in the approved source pipeline; B and GPU match. Their published references are reused, with this provenance explicit.

`VARIANT_REFERENCES.json` separately inventories exact-descriptor references for the older cache studies. Their within-study corrector/long/stall comparisons remain paired; the analyzer reports no incompatible absolute pure comparison. The five missing historical LIBERO pure reference shapes cost **5,000 episodes** total at matching R/B (1,000 each, shared across repeated sizes/cohorts); these extra references are inventoried, not queued. Their absence does not affect fresh original-versus-filtered main-grid pairing. No historical pure result or differing-version latency receipt is silently substituted. Historical C10/C15 pure horizon controls are preserved by the keep-pure rule but are not reused as decoupled fallback controls.

The existing cost receipts are 4090 profiles; GR00T/Fast-WAM MT50 receipts were not valid exclusive profiles. `PROFILES.json` defines seven existing-profiler jobs, including the unaffected H100 Cosmos pipeline, shared by full/reduced modes whose actual model identities agree. Their standalone bundles contain exact load.json copies and recorded RGB/state; no experiment is restaged or modified for profiling. Profile outputs are accepted only with valid=true, matching actual loader model identity and matching GPU. Until timing is measured, latency-weighted IR is null. Level IR is the independently reported head-count M/N, matching the gap fitter. This measures costs only; it does not recalibrate gaps or Q3. Legacy R15/R16 keeps its native per-decision cost ledger; no new legacy timing qualification is claimed.

## Commands and scheduling

Run from `/home/weiland/projects/openpi`. Every fresh/deferred unit has the following exact pair (substitute a NAME from the tables/lists):

```bash
bash exp/offline_search/rounds/r17/longseg_1017/commands/prepare_<NAME>.sh
bash exp/offline_search/rounds/r17/longseg_1017/commands/launch_<NAME>.sh
```

`H100_COMMANDS.txt` / `4090_COMMANDS.txt` contain every expanded prepare/launch pair; `H100_NAMES.txt` / `4090_NAMES.txt` are chain-ready ordered NAME lists. Additive fa_ls wrappers also exist in `filter_failed_knob/commands`, where the current chain runner looks; no existing command or chain file was replaced. Do not rebuild the catalog or invoke `filter_failed.build` / `filter_failed_knob` global builders. Existing active runs `fa_co_r_u_l` and `ffk1015_pi05_mt50_reduced_descent` are preserved and omitted from the new remaining-work lists. The current old chains still mention withdrawn ffk DP/Fast-WAM jobs; replace their remaining items with these lists before those jobs are dispatched.

LIBERO OFF runs first in both modes/arms. Its levels prepare invokes the existing cell rule through `longseg_1017.workflow`: new OFF reaching the same-step pure SR skips the entire cell; only failing cells get the unchanged five-level grid. Build explicitly after the corresponding OFF and pure references finish:

```bash
.venv/bin/python -m exp.offline_search.rounds.r17.longseg_1017.workflow build-levels --name fa_ls_dp_r_u_l_l
.venv/bin/python -m exp.offline_search.rounds.r17.longseg_1017.workflow build-levels --name fa_ls_dp_r_f_l_l
.venv/bin/python -m exp.offline_search.rounds.r17.longseg_1017.workflow build-levels --name fa_ls_dp_f_u_l_l
.venv/bin/python -m exp.offline_search.rounds.r17.longseg_1017.workflow build-levels --name fa_ls_dp_f_f_l_l
.venv/bin/python -m exp.offline_search.rounds.r17.longseg_1017.workflow build-levels --name fa_ls_fw_r_u_l_l
.venv/bin/python -m exp.offline_search.rounds.r17.longseg_1017.workflow build-levels --name fa_ls_fw_r_f_l_l
.venv/bin/python -m exp.offline_search.rounds.r17.longseg_1017.workflow build-levels --name fa_ls_fw_f_u_l_l
.venv/bin/python -m exp.offline_search.rounds.r17.longseg_1017.workflow build-levels --name fa_ls_fw_f_f_l_l
```

Analysis and CPU checks:

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.longseg_1017.check
.venv/bin/python -m exp.offline_search.rounds.r17.longseg_1017.analyze --benchmark all
.venv/bin/python -m exp.offline_search.rounds.r17.longseg_1017.analyze --benchmark all --include-unaffected
.venv/bin/python -m exp.offline_search.rounds.r17.longseg_1017.cache_variants
```

The analyses and off classification use the single-reader I/O governor. New prepares use 4 MiB/s Python archive/hash I/O and public rsync, PSI pauses, and reuse installed immutable input seals. H100 stages the small serving metadata tree and any missing original input roots; it does not duplicate already installed fitted arrays. Transport only: new local servers use public 23130, driver 23139; H100 servers keep their original serving ports and use local driver 23199. `TRANSPORT.json` and each campaign record the changes; task UIDs and scientific metadata are unchanged.

Measure the seven timing units **after both experiment queues are closed**, on idle GPUs with MPS off, so another lane’s local renderers cannot invalidate exclusivity. Use the same prepare/launch pair with NAMEs in `H100_TIMING_NAMES.txt` / `4090_TIMING_NAMES.txt`. The profiler already enforces its own exclusivity/validity rules. Re-run analysis after profiles finish. `--include-unaffected` adds Cosmos and legacy VLA formal-A rows through the existing analyzer, writing its auxiliary output only under this package; it leaves the existing formal_A output intact. Pi0.5 MT50 remains with its unchanged existing analysis.

## Ordered queues and estimates

One lane per GPU; reduced steps first, OFF before dependent levels. New DP MetaWorld grids and its cache study are on 4090 with unchanged R/B/MPS/archive and fresh matching pure controls. The migrated cache study keeps its qualified R4/B8/24-GiB shape (`BALANCE_BUILD.log`, `BALANCE_CPU.log`). GR00T R11 and Fast-WAM R2 MT50 must stay on H100 under the unchanged resident-memory limits. Other cache studies preserve their source GPU/configuration, including the owner’s R3/R2 split. The source-attested local DP R4 admission presentation is reused only where the source preserved CACHE_CAPACITY_ADMISSION.json, exactly as in the standard ffk builder; actual R/B, sampler and all loader identity checks remain intact. The lists are a near-balanced feasible parallel plan, not a measured optimal makespan.

Rates are previous accepted-episode spans (not withdrawn SR). DP MT50 on its new GPU uses the measured 4090 native-pure rate as a conservative proxy; the new R8/B32 and R4/B16 grid throughputs are unmeasured. Corrected clocks, level mixtures, shared worker capacity and infrastructure losses can change these estimates. Queue hours exclude active-run remaining time, prepare/startup/cleanup and use all-cell level upper bounds. Legacy pure test-A spans supply the VLA estimates; historical source metadata supplies R17 estimates. `QUEUES.json` records every rate and source.

### H100: 1,225,047 episodes; approximately 138.5 GPU-hours

```text
fa_co_f_f_oc1
fa_ls_fw_r_p_m
fa_ls_gr_r_p_m
fa_ls_fw_r_f_m
fa_ls_fw_r_u_m
fa_ls_gr_r_f_m
fa_ls_gr_r_u_m
fa_ls_v02_s
fa_ls_v02_b
fa_ls_v03_0
fa_ls_v03_1
fa_ls_fw_f_b_l
fa_ls_fw_f_p_l
fa_ls_fw_f_p_m
fa_ls_gr_f_p_m
fa_ls_fw_f_f_o_l
fa_ls_fw_f_u_o_l
fa_ls_fw_f_f_l_l
fa_ls_fw_f_u_l_l
fa_ls_fw_f_f_m
fa_ls_fw_f_u_m
fa_ls_gr_f_f_m
fa_ls_gr_f_u_m
fa_ls_v05_0
fa_ls_v05_1
fa_ls_v05_2
fa_ls_v05_3
fa_ls_v05_4
fa_ls_v07_0
fa_ls_v08_1
fa_pi_f_u_o
fa_pi_f_f_o
fa_co_r_f_l
fa_co_f_u_l
fa_co_f_f_l
fa_pi_f_u_l
fa_pi_f_f_l
```

### RTX 4090: 863,540 episodes; approximately 142.1 GPU-hours

```text
fa_pi_r_f_l
fa_gr_r_u_l
fa_gr_r_f_l
fa_ls_dp_r_b_l
fa_ls_dp_r_b_m
fa_ls_dp_r_b_v
fa_ls_dp_r_p_l
fa_ls_dp_r_p_m
fa_ls_fw_r_p_l
fa_ls_dp_r_f_o_l
fa_ls_dp_r_u_o_l
fa_ls_fw_r_f_o_l
fa_ls_fw_r_u_o_l
fa_ls_dp_r_f_l_l
fa_ls_dp_r_u_l_l
fa_ls_fw_r_f_l_l
fa_ls_fw_r_u_l_l
fa_ls_dp_r_f_m
fa_ls_dp_r_u_m
fa_ls_v01_0
fa_ls_v10_0
fa_ls_v11_0
fa_ls_v12_0
fa_ls_v13_0
fa_ls_dp_f_b_l
fa_ls_dp_f_b_m
fa_ls_dp_f_p_l
fa_ls_dp_f_p_m
fa_ls_dp_f_f_o_l
fa_ls_dp_f_u_o_l
fa_ls_dp_f_f_l_l
fa_ls_dp_f_u_l_l
fa_ls_dp_f_f_m
fa_ls_dp_f_u_m
fa_ls_v04_0
fa_ls_v04_1
fa_ls_v04_2
fa_ls_v04_3
fa_ls_v06_0
fa_ls_v09_0
fa_ls_v09_1
fa_ls_v14_0
fa_gr_f_u_l
fa_gr_f_f_l
```

Timing after both queues: H100: 0.50 h proxy, RTX 4090: 0.35 h proxy.

## Initial CPU evidence (before recalibration)

**22 targeted tests passed; 60 frozen startup/planner checks passed; 8 deferred level-template checks passed.** CUDA was initialized in zero probes; the longest computed MPS pipe is 52 bytes (limit 85). `PROFILE_INPUT_CPU.json` also checks seven independent profiler bundles / eleven observation inputs and exact load.json bytes. `ANALYSIS_SMOKE.log` exercises all 84 expected affected grid cells; every unmeasured SR and paired difference remains null.

`CPU_TESTS.log` records the real fitted-recipe replay, forced periodic/stall calls, same-observation short/long clock invariant, exact tails, frozen-worker metadata, disabled-cache controls, staged-input reuse, transport disjointness and I/O archive roundtrip tests. No simulator/server/socket was started by these tests.

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m pytest -q \
  exp/r17_integration/tests/common/test_serving.py::test_forced_knob_or_stall_look_waits_for_committed_tail \
  exp/r17_integration/tests/ops/test_cache_study.py::test_sealed_fit_real_layout_cache_corrector_and_tail \
  exp/r17_integration/tests/ops/test_cache_study.py::test_original_q3_confirmed_ambiguous_and_one_fresh_anchor_cooldown \
  exp/offline_search/rounds/r17/longseg_1017/test_boundary.py
```

`CPU_PROOF.json` binds each new campaign’s exact descriptor bytes to the existing strict frozen-loader CPU proof, reruns the new pinned bootstrap in its actual sealed-version interpreter with CUDA disabled, checks native DP admission, unchanged input inodes and native planner/UID coverage, and checks actual GPU-hash MPS paths <=85 bytes. We reused immutable loader proofs rather than rehash/reload multi-GB weights/fits on the busy data disk; the receipt says so. `DEFERRED_CPU_PROOF.json` checks all potential level tasks and ties the deferred templates’ identical descriptors/runtime/interpreter to their checked OFF campaign. Actual selected cells are instantiated only after the new OFF results.

Broader pre-existing tests contained six unrelated model-binding failures in code with existing pass branches; they were not changed or used to weaken any identity check. Only the targeted required tests are claimed here.

## Recalibration evidence and current scheduling

The queue lists above are the original balancing proposal. The coordinator has since placed six independent cohorts before the existing recalibration gate; the live chain order is authoritative and was not edited here. See RECALIBRATION_DEPENDENCIES.json for every gated item. Deferred levels automatically consume the new input map when built after OFF.

Regenerated definitions preserve task UID/init/seed and serving topology; their new recipe/YAML identities remain in actual evidence. RECALIBRATION_APPLIED.json beside each changed campaign records the comparison. RECALIBRATION_CPU_PROOF.json and RECALIBRATION_DEFERRED_CPU_PROOF.json are the new startup/planner receipts; RECALIBRATION_TESTS.final.log is the final targeted test result.

```bash
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.longseg_1017.check --out exp/offline_search/rounds/r17/longseg_1017/RECALIBRATION_CPU_PROOF.json
```

Final recalibrated checks: **60/60 startup/planner, 8/8 potential level templates, 35 targeted tests passed**. CUDA initialization count 0; maximum MPS pipe 52/85 bytes. The level map has 838/840 entries within .005; the two original-library LIBERO spatial size-100 DP k10 entries (full/reduced) have a confirmed-Q3 call floor .1059288538. They are explicitly marked outside tolerance, with no silent target or stall-rule relaxation.

