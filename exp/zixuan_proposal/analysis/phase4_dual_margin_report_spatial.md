# TRACER Phase 4 — dual-retrieval margin report (libero_spatial)

- **Verdict**: PASS ✅
- config: `exp/zixuan_proposal/config/dual_retrieval_active.yaml`
- artifact: `exp/common/data/cache_artifacts/libero_spatial/cp1_mean_pool_dual.pkl`
- entries: 1810 (D+ = 1018, D- = 792, untagged = 0)
- sampled per pool: 60 D+ / 60 D-

## Gate checks — all 3 must pass (D+ queries = success states, genuinely NOT in D-; history reset per query)
1. **Coverage** (both pools non-empty, zero None): PASS
2. **Non-trivial margin** (load-bearing: D+ queries' s_neg non-zero + varies, and margin < s_pos for 60/60 of them): PASS
3. **Discrimination** (necessary-but-weak: mean s_neg D- queries 0.0571 > D+ queries 0.0491; D- self-match inflates it): PASS

## Signal distributions
- D+ queries (success states, NOT in D- -> genuine cross-pool s_neg):
    - s_pos  : {'n': 60, 'mean': 0.0595, 'median': 0.0593, 'min': 0.0547, 'max': 0.0656, 'std': 0.0029}
    - s_neg  : {'n': 60, 'mean': 0.0491, 'median': 0.0482, 'min': 0.0339, 'max': 0.0641, 'std': 0.0061}
    - margin : {'n': 60, 'mean': 0.0349, 'median': 0.0352, 'min': 0.0266, 'max': 0.0438, 'std': 0.0045}
- D- queries (failure states, self-match in D- inflates s_neg toward 1):
    - s_pos  : {'n': 60, 'mean': 0.0488, 'median': 0.0485, 'min': 0.0362, 'max': 0.0615, 'std': 0.0065}
    - s_neg  : {'n': 60, 'mean': 0.0571, 'median': 0.0567, 'min': 0.0516, 'max': 0.0648, 'std': 0.0039}
    - margin : {'n': 60, 'mean': 0.0202, 'median': 0.0208, 'min': 0.0046, 'max': 0.0322, 'std': 0.0066}

## Caveat
A D- entry used as a query self-matches inside D- (s_neg -> ~1.0), so the D- row's s_neg is inflated. The honest non-degeneracy signal is the D+ queries' s_neg spread (they are not in D-). The margin = s_pos - lambda*s_neg is now data-driven, no longer the enable_dual=false degenerate margin = s_pos.

## Provenance — held-out init split (contamination fix)
D- was collected on the **held-out** init pool `exp/common/data/db_init/libero/libero_spatial` (50 states/task = full `.init` minus the `.pruned_init` evaluation set), matching the D+ library's split and **disjoint from the Phase 7 pruned_init eval set**. This supersedes the earlier D- run that was mistakenly collected on `pruned_init` (the eval set). See `phase4_dminus_provenance.md`.
