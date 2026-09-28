Q4 complete, verified 2026-09-28T02:37:04.970944+00:00. Both required π0.5 libraries and all four CL2 artifacts are ready for coordinator-owned closed-loop evaluation. No existing method, plugin, or store library was edited. GR00T was optional and was not built; no required cell is missing.

**Published libraries and paid provenance.** Each cell is labeled **“50 + 250 policy episodes”**. Only successful episodes from the 250 acquired task/init pairs contribute rows; all acquisition policy calls, including failed episodes, remain paid data.

| Suite | Base episodes / rows | Acquisition episodes / successes | Paid policy calls | Appended rows | Final episodes / rows | Bytes per store copy |
|---|---:|---:|---:|---:|---:|---:|
| l10 | 50 / 2,640 | 250 / 208 | 14,849 | 10,481 | 258 / **13,121** | 3,459,476,787 |
| spatial | 49 / 1,018 | 250 / 248 | 5,310 | 5,222 | 297 / **6,240** | 1,645,435,050 |

The nominal static-50 spatial library actually contains **49 episodes**, all preserved. Exact duplicate removals were **zero** in both suites. The source acquisition rows total 14,849 for l10 and 5,310 for spatial; successful rows total 10,481 and 5,222. Acquisition is all ten tasks × inits 0–24; evaluation is all ten tasks × inits 25–49. Failed acquisition rows and all evaluation rows are excluded from append. Both source-to-500-library file-overlap counts are zero; this does not imply independent benchmark initialization distributions.

The new directories are:

- `/home/weiland/trace_runs/offline_search_store/library/pi05_l10/grow250/`
- `/dev/shm/offline_search_store/library/pi05_l10/grow250/`
- `/home/weiland/trace_runs/offline_search_store/library/pi05_spatial/grow250/`
- `/dev/shm/offline_search_store/library/pi05_spatial/grow250/`

Each directory is a standard `offline_search.library.v1` store library, with full float32 pooled keys, robot state, `(10,32)` policy chunks, the twelve standard row arrays, `ids.json`, `episodes.json`, and `manifest.json`. Added arrays are `next_valid`, `terminal_known`, `is_terminal`, `is_growth`, `source_query_row`, and `source_query_episode`. `provenance.json` enumerates all 250 acquisition episodes, source H5 paths, success flags, source row ranges, per-episode admissions, the source fit SHA256, and hashes/bytes of source store inputs. `checksums.json` covers all 22 other files. No token/image archive is copied; AWM needs only pooled keys, state and actions.

Base IDs, row order, metadata arrays, actions and keys are preserved bit for bit. New row IDs increase monotonically after the base prefix, in `(init, task_id, actual step)` order; string IDs are `growth:<lib_key>:<source_uid>:<step>`. Exact dedup compares task, frozen PCA64×2 representation, valid state8, and the **full 10×7 valid action**, with byte-equality confirmation on hash matches. An independent audit using frozen whitened codes/state gives exactly the same admission set and zero duplicates. Padding and head-only equality are excluded from admission decisions.

`ep_len` is the complete source episode decision count and `progress=step/max(ep_len-1,1)`. Edges link only adjacent actual decisions of the same stored episode. A missing edge is unknown continuation; `next==-1` never defines terminal. `terminal_known` records knowledge from the completed episode, and `is_terminal` is true only at the known endpoint. These complete snapshots have zero missing nonterminal edges; a deliberately sparse fixture verifies that omitted middle rows create gaps without terminal flags or cross-episode links.

Before the l10 write, `/dev/shm` had **60,114,132,992 bytes** available; before spatial it had **56,654,602,240**. After both copies it had **55,009,099,776**. Total new library bytes are **5,104,911,837 per store**, including JSON and checksums. Free space was checked again immediately before each RAM-store copy. Publication used new `.partial` directories and rename; existing destinations are refused.

**Method switches and fit provenance.** Method spec: `exp/offline_search/rounds/r05/q4_growth/method.py:GrowthAWM`. Exact kwargs are `{"library":"grow250","variant":"refit","kref":5}` or `{"library":"grow250","variant":"frozen","kref":5}`. The wrapper uses the existing AWM query/synthesis implementation and returns `library="grow250"`; the plugin discovers and resolves the stored library without registration or plugin changes.

- `refit`: AWM’s ordinary library fit on all grown rows, including randomized PCA, task centers, main and early whitening, candidate scales and LOEO confidence calibration. The action sigma remains from `current`, exactly as the existing AWM/harness convention does for any candidate library. The grown fingerprint cannot reuse the current PCA cache.
- `frozen`: loads the actual R2 static-50 CL2 artifact; preserves PCA, main/early maps, sigma, scales, confidence coefficients, and every old candidate array exactly; projects only new candidate codes through those saved maps. The old R2 artifacts predate cached visual auxiliary task centers, so these unused-by-pure-CL2 centers are reconstructed from **base rows only**. New heads/state/norms/full actions are appended. No growing-bank recalibration occurs.

Both new variants hold CL2 `k=16`, `kref=5`, joint features, early metric and `lam_c=.5` fixed, with insurance off. The R2 500 reference uses its deployed `kref=8`; its comparison is therefore not a candidate-count-only ablation. Frozen fitting is a CPU snapshot analogue of resident append; no GPU residency or latency claim is made.

**Offline results.** These are fixed-recorded-observation diagnostics on both `inf` and `cache` streams, all 250 held-out pairs per stream, sampled at steps 1,6,11,… exactly as C’s main/stale geometry study. Every outcome is included. Main/stale top16 kernel actions are compared to the recorded policy `a_inf` head using current-library sigma and only 5×7 valid action dimensions. Early/fresh behavior is covered by separate query parity checks, not by this table.

D1/D16 below use a **common frozen static-50 PCA/whitening frame for every candidate bank**. Identical grown contents therefore have identical density for frozen/refit. Action and successor losses use each variant’s actual fit and CL2 kernel; native own-fit distances are also saved in JSON, but should not be compared across separately whitened fits.

Successor RMS compares kernel-weighted observed library `rs[next]-rs` to the recorded query’s next-state displacement, normalized per task by current-library state std floored at .05. Missing/terminal edges contribute **no invented zero transition**: their weight is omitted and the observed-edge weights are renormalized. All four variants use the same complete-case query mask. “Edge mass” exposes the original kernel weight retained before renormalization. This is a conditional dynamics diagnostic, not a rollout predictor.

| Suite / stream | Variant | N action / successor | D1 | D16 | Action RMS | Successor RMS | Edge mass |
|---|---|---:|---:|---:|---:|---:|---:|
| l10 / inf | static50 | 2960 / 2917 | 12.104026 | 14.800682 | 0.383899 | 0.106412 | 0.976075 |
| l10 / inf | grown_frozen | 2960 / 2917 | 10.040574 | 13.176809 | 0.322190 | 0.091107 | 0.973652 |
| l10 / inf | grown_refit | 2960 / 2917 | 10.040574 | 13.176809 | 0.307294 | 0.086430 | 0.974829 |
| l10 / inf | library500 | 2960 / 2917 | 8.830404 | 11.851418 | 0.262884 | 0.071452 | 0.982390 |
| l10 / cache | static50 | 4110 / 4086 | 17.184276 | 19.478475 | 0.523548 | 0.167209 | 0.954498 |
| l10 / cache | grown_frozen | 4110 / 4086 | 15.727958 | 18.240226 | 0.488800 | 0.165738 | 0.945734 |
| l10 / cache | grown_refit | 4110 / 4086 | 15.727958 | 18.240226 | 0.472315 | 0.158913 | 0.933712 |
| l10 / cache | library500 | 4110 / 4086 | 14.839132 | 17.200168 | 0.437378 | 0.160665 | 0.966395 |
| spatial / inf | static50 | 1149 / 1100 | 14.979224 | 17.897769 | 0.428826 | 0.131243 | 0.946764 |
| spatial / inf | grown_frozen | 1149 / 1100 | 11.425285 | 15.739621 | 0.354243 | 0.106458 | 0.949498 |
| spatial / inf | grown_refit | 1149 / 1100 | 11.425285 | 15.739621 | 0.321230 | 0.096016 | 0.945026 |
| spatial / inf | library500 | 1149 / 1100 | 11.984702 | 16.132242 | 0.310081 | 0.090879 | 0.947263 |
| spatial / cache | static50 | 1523 / 1480 | 18.430499 | 21.098876 | 0.571204 | 0.165966 | 0.881539 |
| spatial / cache | grown_frozen | 1523 / 1480 | 16.542650 | 19.695702 | 0.540317 | 0.158583 | 0.875238 |
| spatial / cache | grown_refit | 1523 / 1480 | 16.542650 | 19.695702 | 0.497636 | 0.144528 | 0.796401 |
| spatial / cache | library500 | 1523 / 1480 | 16.835062 | 20.060553 | 0.490877 | 0.149665 | 0.811598 |

Both growth variants reduce action and successor RMS relative to static-50 in these four recorded streams. The 500 library retains lower action RMS in all four. These observations establish neither SR gains nor a closed-loop ordering. `results/offline_{l10,spatial}.json` includes own-fit distances, new-row neighbor shares and equal-episode means; the sixteen accompanying NPZs contain query rows, masks and individual losses for reanalysis.

**Arm specs and held-out manifest.** `arms_q4.json` is emit_arms input with literal `<RUN>` placeholders. `evaluation_pairs.json` is exactly `[[task_id, init], ...]`, all 10×25 evaluation pairs. K4’s loader/count tool accepted it with `EXPECT=250` and selection SHA256 `88992d8e649eceaccf952da1d008847d0cf558b9f8f812a86c17647e9bbca654`. The same manifest works for either suite.

Resolved copies are already installed at `/home/weiland/trace_runs/os_closed_loop/r05_growth/arms_q4_resolved.json`, `/home/weiland/trace_runs/os_closed_loop/r05_growth/evaluation_pairs.json`, and `/home/weiland/trace_runs/os_closed_loop/r05_growth/arms.json`; emitted server YAMLs and matrices are under `/home/weiland/trace_runs/os_closed_loop/r05_growth/config/`. All four have `full_model:false`, `server_env:{"STAGE1_ONLY":"1"}`, five controls/request, no `--os-judge` flag, no blind/policy-tail flags, and `--os-no-shadow-native`.

**Completed prefits.** All artifacts are under `/home/weiland/trace_runs/os_closed_loop/r05_growth/fits/`. Sizes are serialized fit artifacts, not raw library archives. For scale, the original deployed π0.5 cache PKLs are 1,103,155,631 bytes for l10 and 430,792,483 bytes for spatial. Each grown store copy additionally retains raw pooled keys so refitting remains reproducible.

| Arm / fit filename | Bytes | SHA256 |
|---|---:|---|
| `r5q4_p_l10_grow250_frozen.pkl` | 52,612,030 | `8353f5d7553ef595ea8f9101f99323e24080569545e3116d9d67583c135e75a4` |
| `r5q4_p_l10_grow250_refit.pkl` | 52,611,819 | `dde79bc6806424c86c74ca40015768265824dc5c453c12a8769754d71ea5d496` |
| `r5q4_p_sp_grow250_frozen.pkl` | 35,134,284 | `f8de2c2a80d6ca8ff3d48f5cec3da105deb2ba4569664800a9e591bcad48e670` |
| `r5q4_p_sp_grow250_refit.pkl` | 35,134,074 | `5b980f1d964e11ae27b4a4da3b858743f1a681b4dffe0b69ef14a1d6032aad46` |

The following exact commands were executed from `/home/weiland/projects/openpi`; they are also in `prefit_commands.sh`. The small prefit entry calls the standard plugin prefit API with git metadata reading disabled. Existing artifact paths are refused, so these are reproduction commands for an empty fit destination, not an instruction to overwrite the completed artifacts.

```bash
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.rounds.r05.q4_growth.prefit --os-root /dev/shm/offline_search_store --os-method exp/offline_search/rounds/r05/q4_growth/method.py:GrowthAWM --os-kwargs '{"library":"grow250","variant":"refit","kref":5}' --os-cell pi05_l10_cache --os-log-dir /home/weiland/trace_runs/os_closed_loop/r05_growth/prefit_logs/r5q4_p_l10_grow250_refit --os-tag r5q4_p_l10_grow250_refit --os-fit-artifact /home/weiland/trace_runs/os_closed_loop/r05_growth/fits/r5q4_p_l10_grow250_refit.pkl --os-no-shadow-native > /home/weiland/projects/openpi/exp/offline_search/rounds/r05/q4_growth/results/prefit_r5q4_p_l10_grow250_refit.log 2>&1
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.rounds.r05.q4_growth.prefit --os-root /dev/shm/offline_search_store --os-method exp/offline_search/rounds/r05/q4_growth/method.py:GrowthAWM --os-kwargs '{"library":"grow250","variant":"frozen","kref":5}' --os-cell pi05_l10_cache --os-log-dir /home/weiland/trace_runs/os_closed_loop/r05_growth/prefit_logs/r5q4_p_l10_grow250_frozen --os-tag r5q4_p_l10_grow250_frozen --os-fit-artifact /home/weiland/trace_runs/os_closed_loop/r05_growth/fits/r5q4_p_l10_grow250_frozen.pkl --os-no-shadow-native > /home/weiland/projects/openpi/exp/offline_search/rounds/r05/q4_growth/results/prefit_r5q4_p_l10_grow250_frozen.log 2>&1
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.rounds.r05.q4_growth.prefit --os-root /dev/shm/offline_search_store --os-method exp/offline_search/rounds/r05/q4_growth/method.py:GrowthAWM --os-kwargs '{"library":"grow250","variant":"refit","kref":5}' --os-cell pi05_spatial_cache --os-log-dir /home/weiland/trace_runs/os_closed_loop/r05_growth/prefit_logs/r5q4_p_sp_grow250_refit --os-tag r5q4_p_sp_grow250_refit --os-fit-artifact /home/weiland/trace_runs/os_closed_loop/r05_growth/fits/r5q4_p_sp_grow250_refit.pkl --os-no-shadow-native > /home/weiland/projects/openpi/exp/offline_search/rounds/r05/q4_growth/results/prefit_r5q4_p_sp_grow250_refit.log 2>&1
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.rounds.r05.q4_growth.prefit --os-root /dev/shm/offline_search_store --os-method exp/offline_search/rounds/r05/q4_growth/method.py:GrowthAWM --os-kwargs '{"library":"grow250","variant":"frozen","kref":5}' --os-cell pi05_spatial_cache --os-log-dir /home/weiland/trace_runs/os_closed_loop/r05_growth/prefit_logs/r5q4_p_sp_grow250_frozen --os-tag r5q4_p_sp_grow250_frozen --os-fit-artifact /home/weiland/trace_runs/os_closed_loop/r05_growth/fits/r5q4_p_sp_grow250_frozen.pkl --os-no-shadow-native > /home/weiland/projects/openpi/exp/offline_search/rounds/r05/q4_growth/results/prefit_r5q4_p_sp_grow250_frozen.log 2>&1
```

**Validation completed.** `results/validation.json` and the finishing rerun `results/validation_final.json` both passed all four store/suite combinations. Checks include the existing store loaders, existing G5 library-vs-deployed-export gate, existing plugin payload validator against independent base/query source adapters, and the library registration validator. All full keys/state/actions and base metadata prefixes were checked for exact equality; all published files were checksummed in both copies. G5’s key/state/action maximum differences were all 0.

All four prefits were loaded through `PluginRuntime` in fresh processes against **each store root**, independently fitted again, and queried across all ten tasks, init25/init49, both streams, and early/fresh/stale regimes. The **eight checks passed 2,560 query comparisons** (320 per check, including reversed episode order), with bit-identical fit arrays, topk, scores, actions, confidence and extras. Grown indices resolved through the plugin’s action tables. The frozen checks also verified all old candidate arrays and all saved fit/calibration fields unchanged. `results/check_*.json` contains individual hashes and counts. Semantic counterexamples cover full-tail differences, padding exclusion, task/state separation, and missing adjacency.

The build and analysis commands actually run used this mandatory prefix; repeated suite/variant invocations below correspond to the recorded per-command logs:

```bash
PY=(taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python)
Q4=exp/offline_search/rounds/r05/q4_growth
"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.build --suite l10 > "$Q4/results/build_l10.log" 2>&1
"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.build --suite spatial > "$Q4/results/build_spatial.log" 2>&1
"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.prepare > "$Q4/results/prepare.log" 2>&1
bash "$Q4/prefit_commands.sh"
"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.validate > "$Q4/results/validation.log" 2>&1
"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.check_fit --suite l10 --variant refit > "$Q4/results/check_l10_refit_shm.log" 2>&1
"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.check_fit --suite l10 --variant refit --root /home/weiland/trace_runs/offline_search_store > "$Q4/results/check_l10_refit_cold.log" 2>&1
"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.check_fit --suite l10 --variant frozen > "$Q4/results/check_l10_frozen_shm.log" 2>&1
"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.check_fit --suite l10 --variant frozen --root /home/weiland/trace_runs/offline_search_store > "$Q4/results/check_l10_frozen_cold.log" 2>&1
"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.offline --suite l10 > "$Q4/results/offline_l10.log" 2>&1
"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.check_fit --suite spatial --variant refit > "$Q4/results/check_spatial_refit_shm.log" 2>&1
"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.check_fit --suite spatial --variant refit --root /home/weiland/trace_runs/offline_search_store > "$Q4/results/check_spatial_refit_cold.log" 2>&1
"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.check_fit --suite spatial --variant frozen > "$Q4/results/check_spatial_frozen_shm.log" 2>&1
"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.check_fit --suite spatial --variant frozen --root /home/weiland/trace_runs/offline_search_store > "$Q4/results/check_spatial_frozen_cold.log" 2>&1
"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.offline --suite spatial > "$Q4/results/offline_spatial.log" 2>&1
"${PY[@]}" -m exp.offline_search.closed_loop.ops.remote.count --manifest-info "$Q4/evaluation_pairs.json" --model pi05 --suite libero_10 > "$Q4/results/manifest_validation.txt"
"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.semantic_checks > "$Q4/results/semantic_checks.log" 2>&1
"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.validate --out "$Q4/results/validation_final.json" > "$Q4/results/validation_final.log" 2>&1
bash -n "$Q4/prefit_commands.sh"
```

These commands used at most eight concurrent Python processes, each restricted to CPUs 26–29,70–73 with single-thread BLAS and CUDA disabled. No server, port, tmux session, LIBERO worker or running chain was touched.

**Coordinator next steps (not executed here).** Use the already emitted run files. Run a distinct smoke arm/run on held-out pairs, e.g. tasks 0 and 9 × inits 25 and 49, before the 250-pair manifest. Ensure `STAGE1_ONLY=1` at chain launch because the chain’s launch environment can override per-arm environment values. Use only coordinator-owned ports/resources and the existing remote sync/chain procedures. For full evaluation, unset unrelated `OSCL_MANIFEST` overrides so each arm’s manifest field is honored, or explicitly set it to `/home/weiland/trace_runs/os_closed_loop/r05_growth/evaluation_pairs.json`. Do not collect acquisition inits again.

Compare against `/home/weiland/trace_runs/os_closed_loop/r02_g50` arms `oscl50_p_l10_cl2` / `oscl50_p_sp_cl2` and `/home/weiland/trace_runs/os_closed_loop/r02_g500` arms `oscl500_p_l10_cl2` / `oscl500_p_sp_cl2`, restricting accepted records to exactly the same 250 manifest pairs. Report paired outcomes and actual executed controls; the two growth arms are pure cache with no judge. Charge acquisition separately: l10 14,849 and spatial 5,310 full-policy decisions. Do not describe the resulting banks as 50-episode libraries.

**Files and limits.** Added source files are `common.py`, `build.py`, `method.py`, `prefit.py`, `prepare.py`, `validate.py`, `check_fit.py`, `semantic_checks.py`, `offline.py`, and `write_handback.py`, plus specs, manifest, command script, this hand-back, and `results/`. `file_manifest.json` records their observed modification times, byte sizes and SHA256, as well as fit and store-manifest identities. All work is confined to `q4_growth/`, the four new store directories, and `/home/weiland/trace_runs/os_closed_loop/r05_growth/`. The published libraries are immutable; the builder refuses existing names.

No new closed-loop SR/IR, GPU serving, latency, simulator integration or causal rollout improvement has been measured. The offline study omits step0 and fresh-continuity selection by design; the parity checks exercise those code paths but do not test simulator outcomes. The growth snapshot uses paid full-inference data, not naturally accumulated mixed-policy MISSes. Fixed calibration can drift when candidate counts change. No hyperparameter or admission threshold was chosen on evaluation outcomes. GR00T remains an optional unbuilt extension.


**Demo-data scaling extension — completed 2026-09-28T03:05:15.928297+00:00.** All six requested demo libraries are published in both stores, both fit variants were evaluated at every size, and all eight requested anchor-tail prefits are ready. This section is separate from the paid policy-growth experiment above.

**Corpus and nesting.** `bpool_cs` is the existing export of the August 2026 cache-OFF B-pool H5 collection (500 episodes per suite, 50 per task). It contains both successful and unsuccessful collection episodes: 436/500 successful l10 and 487/500 spatial. Selection retains both outcomes and uses no evaluation/query arrays or labels. The source metadata and source-file lists identify this as B-pool collection; the recorded query streams and the coordinator’s evaluation are A-pool. Numeric B-pool init IDs 0–49 are not A-pool evaluation trajectories.

The deployed `current` episodes are **not contained** in `bpool_cs` in either suite: zero shared episode file paths, zero shared row IDs, and zero exact whole-episode key/state/action/step payload hashes. Current has 50 l10 episodes and 49 spatial episodes. Therefore **50 current → 100 demo changes collection as well as size**. Only **100 ⊂ 200 ⊂ 300 ⊂ 500** is a nested size series. `frozen50` transfers the deployed-50 representation onto the demo bank; it does not append demo data to current.

Deterministic selection rule: within each task, sort the lowercase hexadecimal SHA256 of UTF-8 `q4_demo_nested_v1|<suite>|<task_id>|<stem>` ascending, with stem as the tie-break. Take the first 10/20/30 episodes for demo100/200/300. Concatenate complete episodes in `(rank, task_id, actual step)` order. IDs preserve the original `bpool_cs` strings; demo100 is an exact row/ID/array prefix of demo200, and demo200 of demo300. The existing 500 bank retains its original task-major row order; its membership is the superset and stable string IDs map shared rows. `provenance.json` records all selected episode files, source episode indices, source hashes and the exact rule. No success filter or deduplication changes these quotas.

| Suite | Library | Episodes / per task | Successful episodes | Rows | Bytes per store copy |
|---|---|---:|---:|---:|---:|
| l10 | `demo100` | 100 / 10 | 91 | 5,764 | 1,519,560,251 |
| l10 | `demo200` | 200 / 20 | 177 | 11,718 | 3,089,193,292 |
| l10 | `demo300` | 300 / 30 | 266 | 17,612 | 4,643,009,671 |
| spatial | `demo100` | 100 / 10 | 96 | 2,257 | 595,068,390 |
| spatial | `demo200` | 200 / 20 | 194 | 4,440 | 1,170,616,049 |
| spatial | `demo300` | 300 / 30 | 292 | 6,577 | 1,734,037,553 |

All six names exist under **both** `/dev/shm/offline_search_store/library/pi05_<suite>/` and `/home/weiland/trace_runs/offline_search_store/library/pi05_<suite>/`. No `grow250`, `current`, `bpool_cs`, or other pre-existing store directory was changed. The new directories contain the standard full raw library arrays and metadata, plus source-row/episode maps and explicit `next_valid`, `terminal_known`, `is_terminal` flags. Complete source episodes preserve their real adjacency; missing edges are never inferred terminal. Tokens/images are omitted because these methods use pooled keys.

Total new demo-library storage is **12,751,485,206 bytes per store**. `/dev/shm` free space was **55,009,095,680 bytes** before l10 and **50,400,264,192** before spatial; construction overlapped, and every individual copy rechecked free space. After all six it was **42,257,305,600 bytes**. Per-copy free-space measurements are in `results/demo/build_*.json`. Copies were published via new `.partial` directories and rename, with refusal on existing names.

Reference bank row counts: current has **2,640 l10 / 1,018 spatial** rows; bpool_cs has **29,472 l10 / 10,909 spatial** rows. The nominal 50 spatial point retains the deployed 49-episode bank exactly.

**Fit and controller rules.** Base method `exp/offline_search/rounds/r05/q4_growth/demo_method.py:DemoAWM` accepts `library` in `current,demo100,demo200,demo300,bpool_cs`, `variant` in `refit,frozen50`, and `kref=5`. **kref is fixed at 5 at every size and for both variants**, with k=16 and the other CL2 defaults unchanged. The 500 refit uses its own existing PCA basis and freshly recomputed AWM metric/calibration at kref=5; it is not the historical R2 kref=8 calibration. The 50 point is the same saved deployed CL2 fit for both labels.

- `refit`: PCA, task centers, main/early whitening and confidence calibration use the selected candidate library, through the existing AWM fit recipe. As in AWM, action sigma is from current.
- `frozen50`: PCA, main/early maps, task centers, sigma, s_c/s_d and confidence coefficients remain from current. Missing historical auxiliary task centers are reconstructed from current only. Candidate rows use a rowwise GEMV projection, independent of bank size, row position and row order. Shared candidate codes are **bit-identical across 100→200→300→500**, verified for every task in both suites. No candidate-size recalibration occurs.

The arm method is `exp/offline_search/rounds/r05/q4_growth/demo_method.py:DemoBlindAWM`: it composes the **unchanged K1 `_BlindMixin` used by `BlindAWM`** with `DemoAWM`. Exact controller kwargs are `serving="anchor_tail", budget=1, gates="budget_only"`. A vision anchor supplies the kernel chunk; one following blind decision executes its steps 5–9, then the budget requires vision again. No judge or policy tail is enabled. Frozen50 also preserves current’s K1 state scale and motion calibration; those are diagnostics under budget_only.

**Offline curve.** All 500 A-pool episodes per stream, all ten tasks × inits 0–49, all outcomes; no subset manifest. Queries are steps 1,6,11,… as in C’s main/stale study. The action loss uses top16 kernel first-5×7 actions versus recorded a_inf in current sigma units. Successor loss uses weighted observed library state displacement versus the query’s observed next displacement, per-task current state std floored at .05; missing candidate edges are excluded and weights renormalized. All ten size/variant cells in a stream share one complete-case successor mask. The JSON/NPZ outputs include retained edge mass and equal-episode means.

D1/D16 below use the **same frozen-50 geometry** for both fit labels at each size; they measure content density. They are computed from explicit Euclidean differences to avoid expanded-distance cancellation. Action/successor losses use each variant’s own fitted representation. Native distances in distinct whitening units are retained separately in JSON. An extra `tail_action_rms` compares an anchor’s predicted steps 5–9 with the next recorded policy head; this is also a fixed-observation diagnostic.

| Suite / stream | Size | Queries / successor | Common D1 | Common D16 | Action refit | Action frozen50 | Successor refit | Successor frozen50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| l10 / inf | 50 | 5986 / 5896 | 12.247298 | 14.927021 | 0.388684 | 0.388684 | 0.107475 | 0.107475 |
| l10 / inf | 100 | 5986 / 5896 | 10.699273 | 14.604504 | 0.325695 | 0.350672 | 0.091954 | 0.102184 |
| l10 / inf | 200 | 5986 / 5896 | 9.993423 | 13.434253 | 0.298528 | 0.320102 | 0.083284 | 0.093673 |
| l10 / inf | 300 | 5986 / 5896 | 9.583640 | 12.803697 | 0.282927 | 0.302786 | 0.078390 | 0.086804 |
| l10 / inf | 500 | 5986 / 5896 | 8.951888 | 11.993798 | 0.263569 | 0.280609 | 0.072897 | 0.081030 |
| l10 / cache | 50 | 8127 / 8075 | 17.036063 | 19.356934 | 0.514571 | 0.514571 | 0.164523 | 0.164523 |
| l10 / cache | 100 | 8127 / 8075 | 16.579468 | 19.534394 | 0.488253 | 0.515520 | 0.170420 | 0.168988 |
| l10 / cache | 200 | 8127 / 8075 | 15.523712 | 18.174356 | 0.456427 | 0.485070 | 0.163498 | 0.170050 |
| l10 / cache | 300 | 8127 / 8075 | 15.131562 | 17.652303 | 0.438905 | 0.474389 | 0.165672 | 0.171354 |
| l10 / cache | 500 | 8127 / 8075 | 14.730723 | 17.110152 | 0.432119 | 0.463349 | 0.164562 | 0.171170 |
| spatial / inf | 50 | 2259 / 2166 | 14.828180 | 17.737495 | 0.417228 | 0.417228 | 0.128414 | 0.128414 |
| spatial / inf | 100 | 2259 / 2166 | 14.083043 | 19.075701 | 0.370321 | 0.429811 | 0.111674 | 0.133314 |
| spatial / inf | 200 | 2259 / 2166 | 13.078386 | 17.645478 | 0.332819 | 0.385068 | 0.098371 | 0.119155 |
| spatial / inf | 300 | 2259 / 2166 | 12.533878 | 16.913368 | 0.316531 | 0.360490 | 0.093410 | 0.111434 |
| spatial / inf | 500 | 2259 / 2166 | 11.849542 | 15.993640 | 0.297920 | 0.335740 | 0.087335 | 0.102500 |
| spatial / cache | 50 | 3042 / 2939 | 18.543039 | 21.210736 | 0.574925 | 0.574925 | 0.166252 | 0.166252 |
| spatial / cache | 100 | 3042 / 2939 | 18.626091 | 22.668307 | 0.555028 | 0.613379 | 0.166523 | 0.183175 |
| spatial / cache | 200 | 3042 / 2939 | 17.918225 | 21.501602 | 0.523726 | 0.586266 | 0.154607 | 0.175416 |
| spatial / cache | 300 | 3042 / 2939 | 17.527843 | 20.937657 | 0.516827 | 0.572554 | 0.156160 | 0.173049 |
| spatial / cache | 500 | 3042 / 2939 | 16.938335 | 20.170035 | 0.495608 | 0.552407 | 0.151424 | 0.169870 |

Across the nested 100→500 series, mean action loss decreases at each size for both variants, both suites and both query streams. Refit action loss is lower than frozen50 at every demo size. Inference-query successor loss also decreases throughout; cache-query successor loss is not consistently monotone. In particular, the 500 frozen50 cache successor loss remains above current in both suites. These are fixed-observation measurements, not success-rate predictions.

The complete measurements are `results/demo/curve_{l10,spatial}.json` and forty per-cell NPZs. Standalone figures are [loss curves](results/demo/demo_curve_losses.png) and [common-frame density curves](results/demo/demo_curve_density.png), also supplied as PDFs. The 50 point is deliberately disconnected from the nested demo series. Density monotonicity from 100 through 500 is checked per query; it does not require action or successor loss to improve monotonically. No offline metric is presented as SR or as a closed-loop ranking.

**Arms and artifacts.** `arms_q4_demo.json` contains six refit arms (100/200/300 × l10/spatial) and two frozen50 arms (200 × l10/spatial). `<RUN>` placeholders resolve to `/home/weiland/trace_runs/os_closed_loop/r05_demo_curve`. Already emitted: `arms_q4_demo_resolved.json`, `arms.json`, and `config/` in that run root. All eight are pure cache, `--os-blind`, no `--os-judge`, `full_model:false`, `STAGE1_ONLY=1`, and replan_steps=5. **There is no manifest field**: each arm targets all 500 evaluation pairs.

| Prefit filename (under `/home/weiland/trace_runs/os_closed_loop/r05_demo_curve/fits/`) | Bytes | SHA256 |
|---|---:|---|
| `r5q4d_p_l10_100_refit_tail1.pkl` | 34,145,321 | `3cd2b5320540afb2e3f5ea87e67d7edb4c5599e4376734d91c702bfc97e61722` |
| `r5q4d_p_l10_200_refit_tail1.pkl` | 49,494,769 | `41a2a847fc4954663981c9cfab9df74bf6cd4c9013a2cdf8babd520022e92970` |
| `r5q4d_p_l10_300_refit_tail1.pkl` | 64,689,528 | `04cc1b25ef39a65a7b55c3a0a911e9858dbc37f37f4650b9ab18f7c91968bcce` |
| `r5q4d_p_l10_200_frozen50_tail1.pkl` | 49,494,984 | `e95e1e9780aca841ced3cc2b9eef1862bb8afe0edeeff64860e57b3282e746d4` |
| `r5q4d_p_sp_100_refit_tail1.pkl` | 25,104,085 | `937ddeff5e7302ff22d43c79cd7d16f745137b4731e8afad8f10e4d6555063c2` |
| `r5q4d_p_sp_200_refit_tail1.pkl` | 30,732,026 | `0d79a8dbec35894c0ba872a52b8300c2dcd34d377c50724d847ea42f6a4f56bb` |
| `r5q4d_p_sp_300_refit_tail1.pkl` | 36,241,257 | `abc1ccb1a70df2af842d1ce2864b1c48c4d493f17096f186844009a677ebeead` |
| `r5q4d_p_sp_200_frozen50_tail1.pkl` | 30,732,276 | `b4abd5c090245e98dd526562b7a896c93685bb4b9c10b06a4b358b7a89f9ab4c` |

Exact prefit commands were executed from the repo root and are stored in `demo_prefit_l10.sh` and `demo_prefit_spatial.sh`. Every command has the required affinity/thread/CUDA prefix. Existing artifacts are refused; do not blindly rerun these commands against populated fit paths. The final frozen50 commands were rerun after preserving earlier numerical-check artifacts under `r05_demo_curve/superseded/`; only the eight paths in `fits/` are advertised.

```bash
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.rounds.r05.q4_growth.prefit --os-root /dev/shm/offline_search_store --os-method exp/offline_search/rounds/r05/q4_growth/demo_method.py:DemoBlindAWM --os-kwargs '{"library":"demo100","variant":"refit","kref":5,"serving":"anchor_tail","budget":1,"gates":"budget_only"}' --os-cell pi05_l10_cache --os-log-dir /home/weiland/trace_runs/os_closed_loop/r05_demo_curve/prefit_logs/r5q4d_p_l10_100_refit_tail1 --os-tag r5q4d_p_l10_100_refit_tail1 --os-no-shadow-native --os-blind --os-fit-artifact /home/weiland/trace_runs/os_closed_loop/r05_demo_curve/fits/r5q4d_p_l10_100_refit_tail1.pkl > /home/weiland/projects/openpi/exp/offline_search/rounds/r05/q4_growth/results/demo/prefit_r5q4d_p_l10_100_refit_tail1.log 2>&1
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.rounds.r05.q4_growth.prefit --os-root /dev/shm/offline_search_store --os-method exp/offline_search/rounds/r05/q4_growth/demo_method.py:DemoBlindAWM --os-kwargs '{"library":"demo200","variant":"refit","kref":5,"serving":"anchor_tail","budget":1,"gates":"budget_only"}' --os-cell pi05_l10_cache --os-log-dir /home/weiland/trace_runs/os_closed_loop/r05_demo_curve/prefit_logs/r5q4d_p_l10_200_refit_tail1 --os-tag r5q4d_p_l10_200_refit_tail1 --os-no-shadow-native --os-blind --os-fit-artifact /home/weiland/trace_runs/os_closed_loop/r05_demo_curve/fits/r5q4d_p_l10_200_refit_tail1.pkl > /home/weiland/projects/openpi/exp/offline_search/rounds/r05/q4_growth/results/demo/prefit_r5q4d_p_l10_200_refit_tail1.log 2>&1
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.rounds.r05.q4_growth.prefit --os-root /dev/shm/offline_search_store --os-method exp/offline_search/rounds/r05/q4_growth/demo_method.py:DemoBlindAWM --os-kwargs '{"library":"demo300","variant":"refit","kref":5,"serving":"anchor_tail","budget":1,"gates":"budget_only"}' --os-cell pi05_l10_cache --os-log-dir /home/weiland/trace_runs/os_closed_loop/r05_demo_curve/prefit_logs/r5q4d_p_l10_300_refit_tail1 --os-tag r5q4d_p_l10_300_refit_tail1 --os-no-shadow-native --os-blind --os-fit-artifact /home/weiland/trace_runs/os_closed_loop/r05_demo_curve/fits/r5q4d_p_l10_300_refit_tail1.pkl > /home/weiland/projects/openpi/exp/offline_search/rounds/r05/q4_growth/results/demo/prefit_r5q4d_p_l10_300_refit_tail1.log 2>&1
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.rounds.r05.q4_growth.prefit --os-root /dev/shm/offline_search_store --os-method exp/offline_search/rounds/r05/q4_growth/demo_method.py:DemoBlindAWM --os-kwargs '{"library":"demo200","variant":"frozen50","kref":5,"serving":"anchor_tail","budget":1,"gates":"budget_only"}' --os-cell pi05_l10_cache --os-log-dir /home/weiland/trace_runs/os_closed_loop/r05_demo_curve/prefit_logs/r5q4d_p_l10_200_frozen50_tail1 --os-tag r5q4d_p_l10_200_frozen50_tail1 --os-no-shadow-native --os-blind --os-fit-artifact /home/weiland/trace_runs/os_closed_loop/r05_demo_curve/fits/r5q4d_p_l10_200_frozen50_tail1.pkl > /home/weiland/projects/openpi/exp/offline_search/rounds/r05/q4_growth/results/demo/prefit_r5q4d_p_l10_200_frozen50_tail1.log 2>&1
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.rounds.r05.q4_growth.prefit --os-root /dev/shm/offline_search_store --os-method exp/offline_search/rounds/r05/q4_growth/demo_method.py:DemoBlindAWM --os-kwargs '{"library":"demo100","variant":"refit","kref":5,"serving":"anchor_tail","budget":1,"gates":"budget_only"}' --os-cell pi05_spatial_cache --os-log-dir /home/weiland/trace_runs/os_closed_loop/r05_demo_curve/prefit_logs/r5q4d_p_sp_100_refit_tail1 --os-tag r5q4d_p_sp_100_refit_tail1 --os-no-shadow-native --os-blind --os-fit-artifact /home/weiland/trace_runs/os_closed_loop/r05_demo_curve/fits/r5q4d_p_sp_100_refit_tail1.pkl > /home/weiland/projects/openpi/exp/offline_search/rounds/r05/q4_growth/results/demo/prefit_r5q4d_p_sp_100_refit_tail1.log 2>&1
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.rounds.r05.q4_growth.prefit --os-root /dev/shm/offline_search_store --os-method exp/offline_search/rounds/r05/q4_growth/demo_method.py:DemoBlindAWM --os-kwargs '{"library":"demo200","variant":"refit","kref":5,"serving":"anchor_tail","budget":1,"gates":"budget_only"}' --os-cell pi05_spatial_cache --os-log-dir /home/weiland/trace_runs/os_closed_loop/r05_demo_curve/prefit_logs/r5q4d_p_sp_200_refit_tail1 --os-tag r5q4d_p_sp_200_refit_tail1 --os-no-shadow-native --os-blind --os-fit-artifact /home/weiland/trace_runs/os_closed_loop/r05_demo_curve/fits/r5q4d_p_sp_200_refit_tail1.pkl > /home/weiland/projects/openpi/exp/offline_search/rounds/r05/q4_growth/results/demo/prefit_r5q4d_p_sp_200_refit_tail1.log 2>&1
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.rounds.r05.q4_growth.prefit --os-root /dev/shm/offline_search_store --os-method exp/offline_search/rounds/r05/q4_growth/demo_method.py:DemoBlindAWM --os-kwargs '{"library":"demo300","variant":"refit","kref":5,"serving":"anchor_tail","budget":1,"gates":"budget_only"}' --os-cell pi05_spatial_cache --os-log-dir /home/weiland/trace_runs/os_closed_loop/r05_demo_curve/prefit_logs/r5q4d_p_sp_300_refit_tail1 --os-tag r5q4d_p_sp_300_refit_tail1 --os-no-shadow-native --os-blind --os-fit-artifact /home/weiland/trace_runs/os_closed_loop/r05_demo_curve/fits/r5q4d_p_sp_300_refit_tail1.pkl > /home/weiland/projects/openpi/exp/offline_search/rounds/r05/q4_growth/results/demo/prefit_r5q4d_p_sp_300_refit_tail1.log 2>&1
taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.rounds.r05.q4_growth.prefit --os-root /dev/shm/offline_search_store --os-method exp/offline_search/rounds/r05/q4_growth/demo_method.py:DemoBlindAWM --os-kwargs '{"library":"demo200","variant":"frozen50","kref":5,"serving":"anchor_tail","budget":1,"gates":"budget_only"}' --os-cell pi05_spatial_cache --os-log-dir /home/weiland/trace_runs/os_closed_loop/r05_demo_curve/prefit_logs/r5q4d_p_sp_200_frozen50_tail1 --os-tag r5q4d_p_sp_200_frozen50_tail1 --os-no-shadow-native --os-blind --os-fit-artifact /home/weiland/trace_runs/os_closed_loop/r05_demo_curve/fits/r5q4d_p_sp_200_frozen50_tail1.pkl > /home/weiland/projects/openpi/exp/offline_search/rounds/r05/q4_growth/results/demo/prefit_r5q4d_p_sp_200_frozen50_tail1.log 2>&1
```

**Verification.** The initial and final store passes each validated all **12** suite/size/store copies: existing loaders, source-backed plugin payload validator, registration validator, current-library G5 gate, per-file hashes, complete source equality, IDs, progress and remapped adjacency. All six source subsets have zero missing nonterminal continuations. Both copies have identical checksum manifests.

The eight artifacts each passed a fresh-process existing **K1/plugin blind selftest** and its **independently fitted** logged replay: **384 decisions**, **192 vision**, **192 blind**, **zero MISSes**, and one broadcast per decision. Topk, scores, confidence, library, action, extras, look reason and dense history matched exactly. Each test uses two interleaved CPU orchestrators and four episode resets. Additional artifact load/query checks covered both stores, all ten tasks, both recorded streams and early/fresh/stale regimes: **960 queries**. Frozen calibration, shared-code identity, exact tail-head equality and lifecycle reset/budget/MISS/task-change checks passed.

A stricter intermediate test exposed float32 batching/row-order drift (2.15e-6 on an initial shared-code check, then 2.40e-5 at the reordered 500 bank). The final rowwise projection removes this drift rather than relaxing the shared-code equality check. Stale frozen artifacts temporarily failed replay against changed projection code; they were superseded, rebuilt and verified again. First-pass logs are preserved as diagnostics; the final `semantics.json`, `validation_final.json`, and `check_<arm>.json` are the acceptance evidence.

Other exact commands run (the loops expand the individual suite/size invocations; all paths are under owned output directories):

```bash
PY=(taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python)
Q4=exp/offline_search/rounds/r05/q4_growth
"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.demo_build --suite l10 > "$Q4/results/demo/build_l10.log" 2>&1
"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.demo_build --suite spatial > "$Q4/results/demo/build_spatial.log" 2>&1
"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.demo_prepare > "$Q4/results/demo/prepare.log" 2>&1
bash "$Q4/demo_prefit_l10.sh"
bash "$Q4/demo_prefit_spatial.sh"
"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.demo_validate > "$Q4/results/demo/validation.log" 2>&1
"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.demo_validate --out "$Q4/results/demo/validation_final.json" > "$Q4/results/demo/validation_final.log" 2>&1
for suite in l10 spatial; do
  "${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.demo_offline --suite "$suite" > "$Q4/results/demo/offline_${suite}.log" 2>&1
  for n in 100 200 300; do
    "${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.demo_check --suite "$suite" --size "$n" --variant refit --tag final > "$Q4/results/demo/check_${suite}_${n}_refit_final.log" 2>&1
  done
done
"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.demo_check --suite l10 --size 200 --variant frozen50 --tag final > "$Q4/results/demo/check_l10_200_frozen50_final.log" 2>&1
"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.demo_check --suite spatial --size 200 --variant frozen50 --tag rowwise > "$Q4/results/demo/check_spatial_200_frozen50_rowwise.log" 2>&1
"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.demo_semantics > "$Q4/results/demo/semantics.log" 2>&1
"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.demo_plot > "$Q4/results/demo/plot.log" 2>&1
"${PY[@]}" -m exp.offline_search.rounds.r05.q4_growth.demo_report
```

**Coordinator next steps and limits.** Use the emitted eight arms with the full 500-pair A-pool evaluation. Clear `OSCL_MANIFEST`, `OSCL_EPISODES` and `OSCL_TASKS` before a full run; do not reuse the previous policy-growth 250-pair manifest. Set `STAGE1_ONLY=1` in the launch environment, use coordinator-owned resources and the existing sync/chain procedure. Optional smoke recipe (not run): copy desired arms with `_smoke` names into a separate run root, retain the absolute active prefit paths, emit and sync; run with `OSCL_TASKS=0,9 OSCL_EPISODES=0 STAGE1_ONLY=1 WPS=1` and no manifest (two pairs per arm). Check initial vision, one blind decision per anchor, zero MISSes, and complete logs; keep smoke DONE markers separate from full-run arms. For a matched 500-library controller reference, hold kref=5; the historical K1 500 kref=8 arm has an additional kernel-setting difference.

No servers, ports, tmux sessions, simulator workers, remote island, GPU or closed-loop chains were used. The plugin tests use fake policy observations and CPU orchestrators. No live SR/IR or serving-latency improvement is claimed. Single fixed nested selection gives one curve, not uncertainty over alternative demo subsets. All demo collection episodes are paid source data, not policy-growth acquisitions or free online learning. The 50→100 corpus change remains a confound. All requested cells were built.

**Added files.** `demo_method.py`, `demo_build.py`, `demo_prepare.py`, `demo_validate.py`, `demo_offline.py`, `demo_check.py`, `demo_semantics.py`, `demo_plot.py`, `demo_report.py`, `arms_q4_demo.json`, the two `demo_prefit_*.sh` scripts, and `results/demo/`. `demo_file_manifest.json` records sizes, SHA256 and observed modification times for these, this extended hand-back, active run artifacts, and each new store manifest/checksum file. `results/demo/handback_original.md` preserves the prior hand-back exactly. No earlier method files or shared code were edited.
