# R6 owner IR–SR frontier (completed full evaluations)

This is a retrospective observed frontier, not a guarantee about untested controllers or the global minimum IR. Pilot outcomes are excluded from these estimates. See `completion_plan.md` for placement-only pilot use.

## Accounting and inclusion

Census: 225 completed arm/reference records; 207 are eligible on the common 500-pair evaluation set. Every excluded summary is listed in `skipped.csv` (165 records). `frontier_points.csv` also retains completed library variants and uncertain-cost rows with `eligible=false`.

This frozen census contains 165 summary-ledger rows, 54 verified legacy raw-audit rows, four historical native-policy references, and two conflicting-cost rows. The exclusions are 147 smoke/pilot summaries, eight demo-bank variants, four grow250 holdouts, four AWM3 extra-bank-prior variants, and the two cost conflicts. Four DUAL references are explicitly added from outside os_closed_loop because PAPER_AB uses them for L=5.

Owner IR = `(c1*v + (1-c1)*m)*5/L`, c1=.152 (pi05), .148 (GR00T), with v=V/N and m=M/N. L=5 for normal requests; pure L=10 has v=m=1 but costs .5 per five controls. Blind and policy-tail requests contribute neither V nor M. Summary eager/measured IR is ignored. Per the current request, wrist and K2 arms receive the same owner coefficients; no historical hardware discount is applied.

For summaries without cost_ledger, reuse the earlier accepted raw-decision N/V/M audit in `../arms.json`, after verifying the current journal SHA and every recorded raw-log size/mtime. The two conflicting legacy R3 records remain visible but are excluded from the frontier. The actual served bank comes from `server.lib_mix`, verified against its manifest and episode.npy. Nominal 50/500 labels describe named banks, with actual source-episode counts in a separate column. demo100/200/300 and grow250 are separate banks; grow250 additionally evaluates only inits25–49. AWM3 prior_alpha=.5 uses 500-bank information with current-bank retrieval and is excluded from a strict 50-bank comparison.

All comparable individual runs have exactly tasks0–9 × inits0–49 and validated accepted outcomes. Single-run SR intervals are Wilson 95%. A/B three-run aggregates retain the same 500 init clusters in their task-stratified bootstrap. Individual run frontiers are shown as requested; selecting their best replicate is optimistic. The A/B aggregate table below prevents confusing a winning single replicate with the reproducible headline. Historical DUAL L=5 references have matching task/init identities but different harness/seed provenance.

## Noninferiority definition

Margin epsilon=.02 absolute SR; nominal one-sided alpha=.05. Exact McNemar p-values test equality, **not** noninferiority. `paired_mcnemar.csv` gives wins/losses and exact two-sided binomial McNemar p for every matched run/reference pair. For NI use the conservative exact paired risk-difference lower bound `BetaQuantile(alpha/2; wins,n-wins+1) - BetaQuantile(1-alpha/2; losses+1,n-losses)`, with the usual 0/1 endpoint conventions. The two component Clopper–Pearson bounds cover jointly by Bonferroni; matching makes wins/losses the paired discordant events. Declare NI only if this lower bound is greater than −.02. This can be less powerful than optimized paired score procedures; failure is inconclusive, not inferiority.

For the owner’s pi05 L=5 mean, combine the DUAL and two full-step r04_cost references. Bounds are averaged after dividing alpha among the constituent paired comparisons; never treat repeated init outcomes as independent. L=10 uses the owner’s r04_cost K10 reference (.904/.986), with the additional .900 l10 run still visible as another point. GR00T L=5 is DUAL, L=10 is r05_q2. A candidate identical to its reference has difference zero by identity. `REFERENCE/...` denotes the reference controller or declared reference mixture as an available baseline option at its known cost; it is not an extra observed run. A separate paired cluster-bootstrap interval is reported as a sensitivity, never mislabeled exact McNemar. These intervals use an independent-init/exchangeable-pair sampling model; the fixed, stratified benchmark is not a random sample of future robots or task families.

The nominal minimum is an exploratory selection over many arms. Also report simultaneous bounds with alpha=.05/444 across the 444 cell/arm × reference tests (shared policy references counted conservatively in both library cells). A nominally passing single run therefore does not establish a portable or repeat-robust optimum.

## Current gap and minimum observed costs

| Cell | Reference | Pure SR | B SR @ IR (3 runs) | B loss pp | Cheapest point reaching reference SR: IR | Cheapest nominal 2 pp NI: IR | Simultaneous NI: IR |
|---|---|---:|---|---:|---:|---:|---:|
| pi05_l10_50 | L=5 | 0.851 | 0.817 @ 0.181 | +3.40 | 0.500 | 0.500 | 1.000 |
| pi05_l10_50 | L=10 | 0.904 | 0.817 @ 0.181 | +8.73 | 0.500 | 0.500 | 0.500 |
| pi05_l10_500 | L=5 | 0.851 | 0.873 @ 0.159 | -2.27 | 0.156 | 0.197 | 1.000 |
| pi05_l10_500 | L=10 | 0.904 | 0.873 @ 0.159 | +3.07 | 0.197 | 0.500 | 0.500 |
| pi05_spatial_50 | L=5 | 0.991 | 0.910 @ 0.140 | +8.07 | 1.000 | 1.000 | 1.000 |
| pi05_spatial_50 | L=10 | 0.986 | 0.910 @ 0.140 | +7.60 | 0.500 | 0.500 | 0.500 |
| pi05_spatial_500 | L=5 | 0.991 | 0.982 @ 0.119 | +0.87 | 1.000 | 1.000 | 1.000 |
| pi05_spatial_500 | L=10 | 0.986 | 0.982 @ 0.119 | +0.40 | 0.118 | 0.123 | 0.500 |
| groot_l10_50 | L=5 | 0.870 | 0.719 @ 0.221 | +15.13 | 1.000 | 1.000 | 1.000 |
| groot_l10_50 | L=10 | 0.866 | 0.719 @ 0.221 | +14.73 | 0.500 | 0.500 | 0.500 |
| groot_l10_500 | L=5 | 0.870 | 0.867 @ 0.187 | +0.33 | 1.000 | 1.000 | 1.000 |
| groot_l10_500 | L=10 | 0.866 | 0.867 @ 0.187 | -0.07 | 0.188 | 0.500 | 0.500 |
| groot_spatial_50 | L=5 | 0.940 | 0.875 @ 0.147 | +6.47 | 1.000 | 1.000 | 1.000 |
| groot_spatial_50 | L=10 | 0.938 | 0.875 @ 0.147 | +6.27 | 0.500 | 0.500 | 0.500 |
| groot_spatial_500 | L=5 | 0.940 | 0.959 @ 0.127 | -1.87 | 0.052 | 0.076 | 1.000 |
| groot_spatial_500 | L=10 | 0.938 | 0.959 @ 0.127 | -2.07 | 0.052 | 0.076 | 0.500 |

Negative loss means SR is above the reference. Point crossing, being within 2 pp, and statistical NI are separate columns in `cell_gaps.csv`; none is an interpolation or an unobserved crossing.

## Pareto points by cell

Dominance uses exact point estimates: no other eligible point has at least this SR at no greater IR, with one strict improvement. Ties are retained. Pure policy points require no cache bank and are shared across the two library columns.

### pi05_l10_50

| Run / arm | Family | Library | SR [95%] | Owner IR |
|---|---|---|---|---:|
| `r06_abl/r6p2_identity_p_l10_50` | ablation_IdentityBlindAWM | current | 0.580 [0.536,0.622] | 0.07630 |
| `r06_paper/r5t_p_l10_50_tail1uc_rep3` | A_commit_cache | current | 0.726 [0.685,0.763] | 0.07639 |
| `r06_abl/r6p2_stuck_p_l10_50` | monitored_committed_rescue | current | 0.824 [0.788,0.855] | 0.17886 |
| `r05_q1/r5q1_c10_p_l10_50` | B_guard_committed_rescue | current | 0.830 [0.795,0.860] | 0.18073 |
| `r03_mx/r3mx_p_l10_perk3` | periodic_MISS | current | 0.832 [0.797,0.862] | 0.42912 |
| `r04_cost/r4f_p_l10_inf_k10_L10` | pure_inference_L10 | none | 0.904 [0.875,0.927] | 0.50000 |

L=5 reference 0.851: cheapest nominal NI point is `r04_cost/r4f_p_l10_inf_k10_L10` at IR 0.5, lower paired SR difference -0.0053783450549424044. Best observed cache/hybrid SR is 0.868 at IR 0.603 (`r03_mx/r3mx_p_l10_awm_h50`), loss -1.73 pp.

L=10 reference 0.904: cheapest nominal NI point is `REFERENCE/pi05_l10_L10` at IR 0.5, lower paired SR difference 0.0. Best observed cache/hybrid SR is 0.868 at IR 0.603 (`r03_mx/r3mx_p_l10_awm_h50`), loss +3.60 pp.

### pi05_l10_500

| Run / arm | Family | Library | SR [95%] | Owner IR |
|---|---|---|---|---:|
| `r06_abl/r6p2_identity_p_l10_500` | ablation_IdentityBlindAWM | bpool_cs | 0.734 [0.694,0.771] | 0.07639 |
| `r05_ptail/r5t_p_l10_500_tail1uc` | A_commit_cache | bpool_cs | 0.828 [0.792,0.859] | 0.07650 |
| `r06_paper/r5t_p_l10_500_tail1uc_rep3` | A_commit_cache | bpool_cs | 0.834 [0.799,0.864] | 0.07654 |
| `r06_paper/r5q1_c10_p_l10_500_rep2` | B_guard_committed_rescue | bpool_cs | 0.864 [0.831,0.891] | 0.15623 |
| `r06_paper/r5q1_c10_p_l10_500_rep3` | B_guard_committed_rescue | bpool_cs | 0.890 [0.860,0.915] | 0.15927 |
| `r05_b1/r5b_p_l10_500_hand` | B1_hand_K7_guard | bpool_cs | 0.906 [0.877,0.929] | 0.19653 |

L=5 reference 0.851: cheapest nominal NI point is `r05_b1/r5b_p_l10_500_hand` at IR 0.19653347555555556, lower paired SR difference -0.0067813818549177785. Best observed cache/hybrid SR is 0.906 at IR 0.197 (`r05_b1/r5b_p_l10_500_hand`), loss -5.53 pp.

L=10 reference 0.904: cheapest nominal NI point is `REFERENCE/pi05_l10_L10` at IR 0.5, lower paired SR difference 0.0. Best observed cache/hybrid SR is 0.906 at IR 0.197 (`r05_b1/r5b_p_l10_500_hand`), loss -0.20 pp.

### pi05_spatial_50

| Run / arm | Family | Library | SR [95%] | Owner IR |
|---|---|---|---|---:|
| `r06_abl/r6p2_identity_p_sp_50` | ablation_IdentityBlindAWM | current | 0.776 [0.737,0.810] | 0.07721 |
| `r06_abl/r6p2_direct_p_sp_50` | ablation_TokenPCAAWM | current | 0.826 [0.790,0.857] | 0.07742 |
| `r06_paper/r5t_p_sp_50_tail1uc_rep3` | A_commit_cache | current | 0.828 [0.792,0.859] | 0.07750 |
| `r06_paper/r5t_p_sp_50_tail1uc_rep2` | A_commit_cache | current | 0.844 [0.810,0.873] | 0.07753 |
| `r06_paper/r5q1_c10_p_sp_50_rep2` | B_guard_committed_rescue | current | 0.912 [0.884,0.934] | 0.13852 |
| `r05_ptail/r5t_p_sp_g50_wrist_rep` | wrist_guard_or_tail | current | 0.928 [0.902,0.948] | 0.24662 |
| `r03_mx/r3mx_p_sp_perk3` | periodic_MISS | current | 0.938 [0.913,0.956] | 0.42070 |
| `r03_mx/r3mx_p_sp_ev_h70` | confidence_quantile | current | 0.956 [0.934,0.971] | 0.42996 |
| `r03_mx/r3mx_p_sp_awm_h70` | confidence_quantile | current | 0.980 [0.964,0.989] | 0.43036 |
| `r04_cost/r4f_p_sp_inf_k10_L10` | pure_inference_L10 | none | 0.986 [0.971,0.993] | 0.50000 |
| `r04_cost/r4f_p_sp_inf_s2001` | pure_inference_L5 | none | 0.994 [0.983,0.998] | 1.00000 |

L=5 reference 0.991: cheapest nominal NI point is `REFERENCE/pi05_spatial_L5` at IR 1.0, lower paired SR difference 0.0. Best observed cache/hybrid SR is 0.980 at IR 0.430 (`r03_mx/r3mx_p_sp_awm_h70`), loss +1.07 pp.

L=10 reference 0.986: cheapest nominal NI point is `REFERENCE/pi05_spatial_L10` at IR 0.5, lower paired SR difference 0.0. Best observed cache/hybrid SR is 0.980 at IR 0.430 (`r03_mx/r3mx_p_sp_awm_h70`), loss +0.60 pp.

### pi05_spatial_500

| Run / arm | Family | Library | SR [95%] | Owner IR |
|---|---|---|---|---:|
| `r06_abl/r6p2_identity_p_sp_500` | ablation_IdentityBlindAWM | bpool_cs | 0.956 [0.934,0.971] | 0.07793 |
| `r06_paper/r4b3_p_sp_500_tail1uc_rep3` | A_commit_cache | bpool_cs | 0.972 [0.954,0.983] | 0.07807 |
| `r06_paper/r4b3_p_sp_500_tail1uc_rep2` | A_commit_cache | bpool_cs | 0.974 [0.956,0.985] | 0.07808 |
| `r04_blind/r4b3_p_sp_500_tail1uc` | A_commit_cache | bpool_cs | 0.982 [0.966,0.991] | 0.07819 |
| `r06_paper/r5q1_c10_p_sp_500_rep3` | B_guard_committed_rescue | bpool_cs | 0.986 [0.971,0.993] | 0.11842 |
| `r05_q6/r5q6_p_spatial_500_tail` | wrist_guard_or_tail | bpool_cs | 0.990 [0.977,0.996] | 0.12273 |
| `r04_cost/r4f_p_sp_inf_s2001` | pure_inference_L5 | none | 0.994 [0.983,0.998] | 1.00000 |

L=5 reference 0.991: cheapest nominal NI point is `REFERENCE/pi05_spatial_L5` at IR 1.0, lower paired SR difference 0.0. Best observed cache/hybrid SR is 0.990 at IR 0.123 (`r05_q6/r5q6_p_spatial_500_tail`), loss +0.07 pp.

L=10 reference 0.986: cheapest nominal NI point is `r05_q6/r5q6_p_spatial_500_tail` at IR 0.12273463499086275, lower paired SR difference -0.01753462699933521. Best observed cache/hybrid SR is 0.990 at IR 0.123 (`r05_q6/r5q6_p_spatial_500_tail`), loss -0.40 pp.

### groot_l10_50

| Run / arm | Family | Library | SR [95%] | Owner IR |
|---|---|---|---|---:|
| `r04_gblind/r4b3_g_l10_50_tail2u` | cache_longer_tail | current | 0.570 [0.526,0.613] | 0.05004 |
| `r06_paper/r5x_g_l10_50_tail1u_rep3` | A_commit_cache | current | 0.602 [0.558,0.644] | 0.07426 |
| `r06_abl/r6p2_direct_g_l10_50` | ablation_TokenPCAAWM | current | 0.604 [0.560,0.646] | 0.07427 |
| `r05_x/r5x_g_l10_50_tail1u` | A_commit_cache | current | 0.608 [0.565,0.650] | 0.07428 |
| `r06_paper/r5x_g_l10_50_tail1u_rep2` | A_commit_cache | current | 0.618 [0.575,0.660] | 0.07429 |
| `r05_q2/r5q2_g_l10_50_G10` | periodic_anchor_policy_tail_G10 | current | 0.718 [0.677,0.756] | 0.18503 |
| `r06_paper/r6p1_c10_g_l10_50_rep3` | B_guard_committed_rescue | current | 0.728 [0.687,0.765] | 0.21948 |
| `r05_q2/r5q2_g_l10_policy_L10` | pure_inference_L10 | none | 0.866 [0.833,0.893] | 0.50000 |
| `DUAL/tr_groot_l10_inf` | pure_inference_L5 | none | 0.870 [0.838,0.897] | 1.00000 |

L=5 reference 0.870: cheapest nominal NI point is `REFERENCE/groot_l10_L5` at IR 1.0, lower paired SR difference 0.0. Best observed cache/hybrid SR is 0.728 at IR 0.219 (`r06_paper/r6p1_c10_g_l10_50_rep3`), loss +14.20 pp.

L=10 reference 0.866: cheapest nominal NI point is `REFERENCE/groot_l10_L10` at IR 0.5, lower paired SR difference 0.0. Best observed cache/hybrid SR is 0.728 at IR 0.219 (`r06_paper/r6p1_c10_g_l10_50_rep3`), loss +13.80 pp.

### groot_l10_500

| Run / arm | Family | Library | SR [95%] | Owner IR |
|---|---|---|---|---:|
| `r04_gblind/r4b3_g_l10_500_tail2u` | cache_longer_tail | bpool_all | 0.808 [0.771,0.840] | 0.05018 |
| `r06_paper/r5x_g_l10_500_tail1u_rep3` | A_commit_cache | bpool_all | 0.828 [0.792,0.859] | 0.07449 |
| `r06_paper/r5x_g_l10_500_tail1u_rep2` | A_commit_cache | bpool_all | 0.832 [0.797,0.862] | 0.07449 |
| `r06_paper/r6p1_c10_g_l10_500` | B_guard_committed_rescue | bpool_all | 0.864 [0.831,0.891] | 0.18663 |
| `r06_paper/r6p1_c10_g_l10_500_rep2` | B_guard_committed_rescue | bpool_all | 0.868 [0.836,0.895] | 0.18752 |
| `DUAL/tr_groot_l10_inf` | pure_inference_L5 | none | 0.870 [0.838,0.897] | 1.00000 |

L=5 reference 0.870: cheapest nominal NI point is `REFERENCE/groot_l10_L5` at IR 1.0, lower paired SR difference 0.0. Best observed cache/hybrid SR is 0.868 at IR 0.188 (`r06_paper/r6p1_c10_g_l10_500_rep2`), loss +0.20 pp.

L=10 reference 0.866: cheapest nominal NI point is `REFERENCE/groot_l10_L10` at IR 0.5, lower paired SR difference 0.0. Best observed cache/hybrid SR is 0.868 at IR 0.188 (`r06_paper/r6p1_c10_g_l10_500_rep2`), loss -0.20 pp.

### groot_spatial_50

| Run / arm | Family | Library | SR [95%] | Owner IR |
|---|---|---|---|---:|
| `r04_gblind/r4b3_g_sp_50_tail2u` | cache_longer_tail | current | 0.864 [0.831,0.891] | 0.05127 |
| `r06_paper/r5x_g_sp_50_tail1u_rep2` | A_commit_cache | current | 0.866 [0.833,0.893] | 0.07526 |
| `r06_paper/r5x_g_sp_50_tail1u_rep3` | A_commit_cache | current | 0.868 [0.836,0.895] | 0.07529 |
| `r06_abl/r6p2_direct_g_sp_50` | ablation_TokenPCAAWM | current | 0.910 [0.882,0.932] | 0.07532 |
| `r05_q2/r5q2_g_spatial_50_G10` | periodic_anchor_policy_tail_G10 | current | 0.920 [0.893,0.941] | 0.19794 |
| `r05_q2/r5q2_g_spatial_policy_L10` | pure_inference_L10 | none | 0.938 [0.913,0.956] | 0.50000 |
| `DUAL/tr_groot_sp_inf` | pure_inference_L5 | none | 0.940 [0.916,0.958] | 1.00000 |

L=5 reference 0.940: cheapest nominal NI point is `REFERENCE/groot_spatial_L5` at IR 1.0, lower paired SR difference 0.0. Best observed cache/hybrid SR is 0.920 at IR 0.198 (`r05_q2/r5q2_g_spatial_50_G10`), loss +2.00 pp.

L=10 reference 0.938: cheapest nominal NI point is `REFERENCE/groot_spatial_L10` at IR 0.5, lower paired SR difference 0.0. Best observed cache/hybrid SR is 0.920 at IR 0.198 (`r05_q2/r5q2_g_spatial_50_G10`), loss +1.80 pp.

### groot_spatial_500

| Run / arm | Family | Library | SR [95%] | Owner IR |
|---|---|---|---|---:|
| `r04_gblind/r4b3_g_sp_500_tail2u` | cache_longer_tail | bpool_all | 0.946 [0.923,0.963] | 0.05175 |
| `r05_x/r5x_g_sp_500_tail1u` | A_commit_cache | bpool_all | 0.964 [0.944,0.977] | 0.07551 |
| `r04_gblind/r4b3_g_sp_500_ph2` | blind_cache_phase_particles | bpool_all | 0.976 [0.959,0.986] | 0.07834 |

L=5 reference 0.940: cheapest nominal NI point is `r05_x/r5x_g_sp_500_tail1u` at IR 0.07550910415149308, lower paired SR difference -0.013113965986885287. Best observed cache/hybrid SR is 0.976 at IR 0.078 (`r04_gblind/r4b3_g_sp_500_ph2`), loss -3.60 pp.

L=10 reference 0.938: cheapest nominal NI point is `r05_x/r5x_g_sp_500_tail1u` at IR 0.07550910415149308, lower paired SR difference -0.010649067269550498. Best observed cache/hybrid SR is 0.976 at IR 0.078 (`r04_gblind/r4b3_g_sp_500_ph2`), loss -3.80 pp.

## Three-replicate A/B check

| Cell | Arm | Replicate SRs | Mean SR [cluster 95%] | Owner IR |
|---|---|---|---|---:|
| pi05_l10_50 | A | 0.706, 0.710, 0.726 | 0.714 [0.683,0.746] | 0.07640 |
| pi05_l10_50 | B | 0.830, 0.794, 0.826 | 0.817 [0.793,0.841] | 0.18109 |
| pi05_l10_500 | A | 0.828, 0.820, 0.834 | 0.827 [0.800,0.855] | 0.07651 |
| pi05_l10_500 | B | 0.866, 0.864, 0.890 | 0.873 [0.850,0.896] | 0.15889 |
| pi05_spatial_50 | A | 0.838, 0.844, 0.828 | 0.837 [0.810,0.862] | 0.07752 |
| pi05_spatial_50 | B | 0.910, 0.912, 0.908 | 0.910 [0.890,0.929] | 0.14043 |
| pi05_spatial_500 | A | 0.982, 0.974, 0.972 | 0.976 [0.964,0.986] | 0.07811 |
| pi05_spatial_500 | B | 0.982, 0.978, 0.986 | 0.982 [0.973,0.990] | 0.11949 |
| groot_l10_50 | A | 0.608, 0.618, 0.602 | 0.609 [0.575,0.643] | 0.07428 |
| groot_l10_50 | B | 0.726, 0.702, 0.728 | 0.719 [0.690,0.747] | 0.22095 |
| groot_l10_500 | A | 0.830, 0.832, 0.828 | 0.830 [0.801,0.858] | 0.07449 |
| groot_l10_500 | B | 0.864, 0.868, 0.868 | 0.867 [0.843,0.889] | 0.18740 |
| groot_spatial_50 | A | 0.868, 0.866, 0.868 | 0.867 [0.838,0.895] | 0.07528 |
| groot_spatial_50 | B | 0.874, 0.874, 0.878 | 0.875 [0.848,0.901] | 0.14714 |
| groot_spatial_500 | A | 0.964, 0.964, 0.964 | 0.964 [0.948,0.978] | 0.07553 |
| groot_spatial_500 | B | 0.958, 0.960, 0.958 | 0.959 [0.941,0.974] | 0.12654 |

## Coverage by method family

| Family | Eligible runs |
|---|---:|
| AWM3_metric_variant | 2 |
| A_commit_cache | 24 |
| B1_hand_AWM | 2 |
| B1_hand_AWM3 | 1 |
| B1_hand_K7_guard | 3 |
| B1_solve_AWM | 2 |
| B1_solve_AWM3 | 1 |
| B1_solve_K7_guard | 3 |
| B_guard_committed_rescue | 24 |
| D1_grasp_guard | 2 |
| K7_vision_guard | 11 |
| ablation_IdentityBlindAWM | 8 |
| ablation_TokenPCAAWM | 4 |
| blind_cache_phase_particles | 7 |
| blind_guard_anchor_tail | 1 |
| blind_guard_kernel_clock | 1 |
| blind_guard_phase_particles | 4 |
| cache_longer_tail | 4 |
| confidence_quantile | 8 |
| control_step_library | 4 |
| five_step_cache_AWM | 8 |
| five_step_cache_B0BigLibConsFast | 8 |
| five_step_cache_B0TopkConsensus | 4 |
| five_step_cache_StuckRecovery | 8 |
| five_step_cache_native | 4 |
| guard_only | 12 |
| monitored_committed_rescue | 1 |
| periodic_MISS | 12 |
| periodic_anchor_policy_tail_G10 | 4 |
| pure_inference_L10 | 5 |
| pure_inference_L5 | 8 |
| pure_inference_L5_K2 | 2 |
| randomized_guard_landmark | 4 |
| retrieval_shadow | 2 |
| wrist_guard_or_tail | 9 |

## Reproduce / provenance

Sources: `logs/offline_search_exploration.log.md` §10; rounds r04/r05 ANALYSIS; `r06/PAPER_AB.md`; each summary, emitted arms.json, accepted journal, served-bank manifest; the earlier Q2 raw audit. `source_evidence.json` records hashes/configurations; `summary_paths.json` freezes the census; `outcomes.json` contains accepted per-init outcomes. No simulator, GPU inference, server, worker, chain, or external host was invoked.

```bash
taskset -c 18-21,62-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/offline_search/rounds/r06/ideation_Q2/frontier/build_frontier.py
```

Figure script and PNG/PDF are outside the repository: `/home/weiland/projects/openpi_ext/artifacts/frontier_r6/plot_frontier.py`. Pilot-only placement data and proposed configurations are separate artifacts, never frontier evidence.

```bash
taskset -c 18-21,62-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python /home/weiland/projects/openpi_ext/artifacts/frontier_r6/plot_frontier.py
taskset -c 18-21,62-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/offline_search/rounds/r06/ideation_Q2/frontier/validate_frontier.py
```

Validation output is `validation.json`; the figure was visually inspected. [Provisional completion plan](completion_plan.md), [existing-controller emit specs](emit_arms_existing.json), [PNG](/home/weiland/projects/openpi_ext/artifacts/frontier_r6/frontier_r6.png), [PDF](/home/weiland/projects/openpi_ext/artifacts/frontier_r6/frontier_r6.pdf).
