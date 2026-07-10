# TRACER Phase 4 — dual-retrieval margin report (libero_10)

- **Verdict**: PASS ✅
- config: `exp/zixuan_proposal/config/dual_retrieval_active_l10.yaml`
- artifact: `exp/common/data/cache_artifacts/libero_10/cp1_mean_pool_dual.pkl`
- entries: 11480 (D+ = 2640, D- = 8840, untagged = 0)
- sampled per pool: 60 D+ / 60 D-

## Gate checks — all 3 must pass (D+ queries = success states, genuinely NOT in D-; history reset per query)
1. **Coverage** (both pools non-empty, zero None): PASS
2. **Non-trivial margin** (load-bearing: D+ queries' s_neg non-zero + varies, and margin < s_pos for 60/60 of them): PASS
3. **Discrimination** (necessary-but-weak: mean s_neg D- queries 0.0518 > D+ queries 0.0483; D- self-match inflates it): PASS

## Signal distributions
- D+ queries (success states, NOT in D- -> genuine cross-pool s_neg):
    - s_pos  : {'n': 60, 'mean': 0.0571, 'median': 0.0567, 'min': 0.0516, 'max': 0.0645, 'std': 0.0035}
    - s_neg  : {'n': 60, 'mean': 0.0483, 'median': 0.0495, 'min': 0.0264, 'max': 0.0613, 'std': 0.0073}
    - margin : {'n': 60, 'mean': 0.0329, 'median': 0.0331, 'min': 0.0235, 'max': 0.0465, 'std': 0.0045}
- D- queries (failure states, self-match in D- inflates s_neg toward 1):
    - s_pos  : {'n': 60, 'mean': 0.0492, 'median': 0.0523, 'min': 0.0291, 'max': 0.0607, 'std': 0.0078}
    - s_neg  : {'n': 60, 'mean': 0.0518, 'median': 0.051, 'min': 0.0495, 'max': 0.0567, 'std': 0.0021}
    - margin : {'n': 60, 'mean': 0.0233, 'median': 0.0265, 'min': 0.0039, 'max': 0.0335, 'std': 0.0077}

## Caveat
A D- entry used as a query self-matches inside D- (s_neg -> ~1.0), so the D- row's s_neg is inflated. The honest non-degeneracy signal is the D+ queries' s_neg spread (they are not in D-). The margin = s_pos - lambda*s_neg is now data-driven, no longer the enable_dual=false degenerate margin = s_pos.

## Provenance — held-out init split (contamination fix)
D- was collected on the **held-out** init pool `exp/common/data/db_init/libero/libero_10` (50 states/task = full `.init` minus the `.pruned_init` evaluation set), matching the D+ library's split and **disjoint from the Phase 7 pruned_init eval set**. This supersedes the earlier D- run that was mistakenly collected on `pruned_init` (the eval set). The much larger D- (8840 steps from 85 failure episodes, vs the earlier run's 2520) reflects libero_10's higher natural failure rate on held-out inits (17% vs pruned's ~5%). See `phase4_dminus_provenance.md`.

## library_stats — D+-only (G2 R2 fix)
The libero_10 D+ source `cp1_mean_pool.pkl` carries **no** `library_stats` key. `build_dual_artifact` now computes D+-only stats via `LibraryStats.compute_from_entries` over the 2640 D+ entries alone (never left None) — otherwise the in_memory backend fallback-recomputes stats over all 11480 entries (D+ ∪ D-), folding the 8840 D- failure steps into Phase 5 u_t normalization. This report was regenerated after the fix; the backend loads D+-only stats from the artifact (no fallback warning).
