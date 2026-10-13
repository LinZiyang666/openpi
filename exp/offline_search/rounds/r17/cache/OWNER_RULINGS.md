
## 2026-10-10 15:4x owner ruling (cache study)
Evaluation set = official test set A (official-50: 50 inits per task, 500 episodes per suite), libero_spatial + libero_10 only.
The owner's "全B集" was a slip; B pool is only the library source. Libraries: full B500 build at /data/r17_library/.

## Stagnation gate (停滞门) — definition to use (R6/R7)
The R6 stall module / R7 "calibrated stall trigger": when progress stalls under cached actions, force one real model call
(a MISS); "confirmed slow -> call, ambiguous -> look one more step" using Q3 progress reliability; thresholds calibrated on
the B validation recording (r06_c_cal); cooldown applies only after stall-triggered calls (R6-C-v2). R7 result: at equal
call budget the calibrated stall trigger gave +1.15 pp [+0.12, +2.23]. Sources: logs/offline_search_exploration.log.md
(around lines 1644-1696), logs/session_handoff.md (R7 decision 2, ~line 1053). In the 4 gated groups, gate-triggered calls
are the only model calls and count as cost.

## 2026-10-13 owner confirmation: stall gate = Q3 only; old guard removed
The five-layer design's stall gate is the R6/R7 Q3 calibrated stall trigger above (YAML `miss_mode: stall_only` + `stall_path`).
The old only-no-progress `NoProgressGuard` (R8 span arithmetic, n=3, eps=0.5) is not part of the design; it was removed from
the code (common/layers/guard.py raises; `recipe.guard: true` on an enabled recipe is rejected). Applies to LIBERO and MetaWorld.

## Coordinator decision 2026-10-10 15:5x (astra CACHE_STUDY open question 2)
Keep the batched server configs already used by the pure-inference baselines (Fast-WAM h100 2xB16/5ms+MPS, DP 4090
2xB8/5ms+MPS, Cosmos h100 2xB1) and reuse those baselines as controls (hs6fw_res_152931 merged, hs5dp_ctl_142812,
Cosmos merged K5 formal_cosmos_libero_official2000_20261010_130639_a8ca2043). Cache-served actions do not depend on
batching; only stall-gate MISS calls do, and they then match the control's server config. No extra controls (20,000 ep).
Evaluation set A confirmed by the owner (question 1 resolved).
