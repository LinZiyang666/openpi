# Offline light query-key prototypes

**Result: do not promote the current models.** A CPU-only implementation is practical, and the action-aware CNN is close in library donor-LOEO, but the unseen L10 recordings still show an accuracy gap. No closed-loop experiment of the new method was run. The offline gates and coordinator approval are prerequisites for the planned pilot.

Sources: rerunnable scripts in this directory; `analysis.json`, `metrics.csv`, `pi05_*/loeo_results.json`, per-model training/results files, and `verification.json`. No test-A root or query store was opened.

## Order, scope and metric

Following the owner addendum, the final assessment starts with every library row as a pseudoquery against other episodes of the same task, then checks existing B-dev. The initial interim B-dev results predated that delivered addendum; subsequent work added the full LOEO comparison and complete CPU training. The final representation is trained on the whole cell library; donor-LOEO does not hold the encoder training episode out.

All comparisons use the same selected action payload, 16-neighbor Gaussian kernel with the cell artifact's deployed kref (5 at size 50; 8 at size 200), early metric at step 0 and stale geometry thereafter. RMS5 is the R2 err definition: RMS over the first 5 controls and 7 valid action dimensions, divided by the selected library sigma. RMS10 uses the 10 controls committed by this system. No action padding is scored. B-dev targets are shadow-policy actions, or explicitly labeled policy MISS actions for the secondary Spatial analysis.

This isolates layer-1 key/synthesis geometry, before insurance, correction or guard decisions. On L10-50, the reference has all recorded top-16 sets identical and only one top-1 difference among 3,198 looks; Spatial-50 has identical top-1 and top-16. The one discrepancy is at step 0; replay through the runtime `_dist` reproduces the recorded top-1 (`reference_order_audit.json`). The prototype uses explicit-code Euclidean arithmetic, while runtime uses its quadratic-form arithmetic; no exact tie is claimed. At L10-200, the size-200 deployed fit is evaluated on the 73 truly unseen episodes of size-50 recorded trajectories; recorded size-50 neighbors are not misused as its reference.

## Methods actually fitted

- **Symmetric spatial RGB AWM:** each camera retains a 14×14 RGB mean grid; per-camera library PCA-64, state-8, and the deployed cross-episode action-pair AWM/early metric. Both sides use this same representation.
- **Asymmetric ridge distillation:** task-specific ridge from both spatial image grids plus state to the cell teacher PCA-128; library keys remain exact teacher AWM codes. Alpha .01/.1/1 were all reported, chosen only on B-dev.
- **Spatial CNN:** shared 3-layer camera trunk retaining 7×7 positions, state and task conditioning, PCA and metric distillation plus action auxiliary loss. Tested with and without a same-task, cross-episode action-pair hinge, in both symmetric and asymmetric retrieval. No pretrained weights. See SPEC for the exact 856,299-parameter recipe.
- **Local residual kernel:** per-task state-to-teacher affine prediction plus a spatial-image RBF residual; four length/ridge combinations. This was the nonlinear alternative to dense CNN regression.
- **Direct action query key:** ridge predicts normalized 10-control actions; exact library action chunks and state define keys. Only library synthesis is served, not the regressed action. Two regularizers × two state weights. This tests a more direct action objective and its phase ambiguity.
- **State-only AWM:** negative control using the same action-pair fitting rule.

The R8 CLIP global-vector failure (reported 3–30 pp losses and phase confusion in `logs/offline_search_exploration.log.md`, CLIP section) motivates retaining spatial positions. R12 REPORT documents why both cameras are retained, especially GR00T L10. Those prior reports are context, not newly opened A data.

## Library LOEO and unseen B-dev, side by side

LOEO and balanced B-dev columns use equal episode weight within task and equal task weight. The B-dev look RMS columns are separately shown because long/failing trajectories contribute more looks. Bootstrap intervals resample paired episodes within task (2,000 draws, seed 20261005) and are descriptive after development selection.

### pi05_l10_50

100 unseen B-dev episodes, 3198 looks, 3198 policy-labeled looks.

| Candidate | Library LOEO RMS10 balanced | B-dev RMS5 | B-dev RMS10 | B-dev RMS10 balanced | Balanced excess vs reference (95% interval) | Top-16 overlap |
|---|---:|---:|---:|---:|---|---:|
| deployed_key_geometry | 0.3583 | 0.4217 | 0.4396 | 0.4056 | +0.0% [+0.0, +0.0] | 1.000 |
| symmetric_rgb_spatial_awm | 0.4127 | 0.4967 | 0.5145 | 0.4781 | +17.9% [+16.4, +19.5] | 0.588 |
| asymmetric_ridge_0.01 | 0.3602 | 0.4899 | 0.5067 | 0.4672 | +15.2% [+13.5, +17.0] | 0.682 |
| asymmetric_ridge_0.1 | 0.3669 | 0.4763 | 0.4945 | 0.4569 | +12.6% [+11.2, +14.1] | 0.708 |
| asymmetric_ridge_1 | 0.3840 | 0.4784 | 0.4973 | 0.4623 | +14.0% [+12.5, +15.5] | 0.713 |
| state_only_awm | 0.4540 | 0.4875 | 0.4983 | 0.4615 | +13.8% [+11.7, +15.9] | 0.578 |
| asymmetric_cnn_action_cpu | 0.3597 | 0.4737 | 0.4915 | 0.4566 | +12.6% [+11.2, +14.0] | 0.704 |
| symmetric_cnn_action_cpu | 0.3565 | 0.4731 | 0.4913 | 0.4563 | +12.5% [+11.1, +13.9] | 0.696 |
| asymmetric_cnn_action | 0.3597 | 0.4737 | 0.4915 | 0.4567 | +12.6% [+11.3, +14.0] | 0.704 |
| symmetric_cnn_action | 0.3565 | 0.4731 | 0.4913 | 0.4563 | +12.5% [+11.0, +14.0] | 0.696 |
| asymmetric_cnn_distill | 0.3602 | 0.4757 | 0.4944 | 0.4596 | +13.3% [+12.0, +14.7] | 0.701 |
| symmetric_cnn_distill | 0.3580 | 0.4732 | 0.4922 | 0.4572 | +12.7% [+11.3, +14.1] | 0.693 |
| local_residual_krr_l0.25_a0.01 | 0.3584 | 0.5088 | 0.5223 | 0.4827 | +19.0% [+16.5, +21.6] | 0.681 |
| local_residual_krr_l1_a0.01 | 0.3591 | 0.4907 | 0.5051 | 0.4630 | +14.1% [+12.0, +16.3] | 0.711 |
| local_residual_krr_l0.25_a0.001 | 0.3583 | 0.5090 | 0.5225 | 0.4828 | +19.0% [+16.5, +21.5] | 0.680 |
| local_residual_krr_l1_a0.001 | 0.3583 | 0.4921 | 0.5066 | 0.4642 | +14.4% [+12.4, +16.7] | 0.707 |
| action_qk_a0.1_s0.25 | 0.3097 | 0.5545 | 0.5666 | 0.5254 | +29.5% [+27.1, +32.0] | 0.498 |
| action_qk_a0.1_s1 | 0.3434 | 0.5041 | 0.5159 | 0.4809 | +18.5% [+16.5, +20.7] | 0.581 |
| action_qk_a1_s0.25 | 0.3597 | 0.5290 | 0.5401 | 0.5054 | +24.6% [+22.2, +27.3] | 0.514 |
| action_qk_a1_s1 | 0.3768 | 0.4870 | 0.4984 | 0.4673 | +15.2% [+12.9, +17.4] | 0.588 |

**Primary CPU-trained candidate, by task.** RMS10 is episode-balanced; no task is removed.

| Task | Episodes | Reference RMS10 | Student RMS10 | Excess |
|---|---:|---:|---:|---:|
| 0 | 10 | 0.5502 | 0.6042 | +9.8% |
| 1 | 10 | 0.3960 | 0.4501 | +13.7% |
| 2 | 10 | 0.3445 | 0.3925 | +13.9% |
| 3 | 10 | 0.3881 | 0.4422 | +14.0% |
| 4 | 10 | 0.4675 | 0.4987 | +6.7% |
| 5 | 10 | 0.2713 | 0.2895 | +6.7% |
| 6 | 10 | 0.3920 | 0.4568 | +16.5% |
| 7 | 10 | 0.4853 | 0.5559 | +14.5% |
| 8 | 10 | 0.2947 | 0.3260 | +10.6% |
| 9 | 10 | 0.4666 | 0.5473 | +17.3% |

Phase proxies on all looks: top-1 agreement 0.313; backwards jumps greater than .10 donor progress 0.061 versus reference 0.042; any backwards movement 0.145 versus 0.101; mean donor-progress difference from teacher 0.059; >1/3 disagreement 0.059.

### pi05_l10_200

73 unseen B-dev episodes, 2289 looks, 2289 policy-labeled looks.

| Candidate | Library LOEO RMS10 balanced | B-dev RMS5 | B-dev RMS10 | B-dev RMS10 balanced | Balanced excess vs reference (95% interval) | Top-16 overlap |
|---|---:|---:|---:|---:|---|---:|
| deployed_key_geometry | 0.3083 | 0.3692 | 0.3851 | 0.3510 | +0.0% [+0.0, +0.0] | 1.000 |
| symmetric_rgb_spatial_awm | 0.3650 | 0.4412 | 0.4534 | 0.4167 | +18.7% [+16.4, +21.1] | 0.520 |
| asymmetric_ridge_0.01 | 0.3232 | 0.4203 | 0.4347 | 0.3969 | +13.1% [+11.4, +14.9] | 0.640 |
| asymmetric_ridge_0.1 | 0.3331 | 0.4251 | 0.4406 | 0.4035 | +15.0% [+13.1, +17.1] | 0.636 |
| asymmetric_ridge_1 | 0.3556 | 0.4380 | 0.4538 | 0.4203 | +19.7% [+17.3, +22.2] | 0.597 |
| state_only_awm | 0.3732 | 0.4302 | 0.4422 | 0.4088 | +16.5% [+14.0, +19.4] | 0.477 |
| asymmetric_cnn_action_cpu | 0.3170 | 0.4145 | 0.4290 | 0.3927 | +11.9% [+10.0, +13.9] | 0.650 |
| symmetric_cnn_action_cpu | 0.3134 | 0.4079 | 0.4217 | 0.3861 | +10.0% [+8.1, +11.8] | 0.630 |
| asymmetric_cnn_action | 0.3170 | 0.4147 | 0.4291 | 0.3928 | +11.9% [+10.0, +13.9] | 0.650 |
| symmetric_cnn_action | 0.3134 | 0.4079 | 0.4217 | 0.3861 | +10.0% [+8.1, +11.8] | 0.630 |
| asymmetric_cnn_distill | 0.3178 | 0.4130 | 0.4278 | 0.3918 | +11.6% [+9.8, +13.6] | 0.655 |
| symmetric_cnn_distill | 0.3141 | 0.4126 | 0.4260 | 0.3891 | +10.9% [+9.0, +12.7] | 0.630 |
| local_residual_krr_l0.25_a0.01 | 0.3087 | 0.4394 | 0.4563 | 0.4162 | +18.6% [+16.2, +20.9] | 0.608 |
| local_residual_krr_l1_a0.01 | 0.3100 | 0.4263 | 0.4416 | 0.4009 | +14.2% [+12.4, +16.2] | 0.648 |
| local_residual_krr_l0.25_a0.001 | 0.3083 | 0.4389 | 0.4559 | 0.4157 | +18.4% [+16.3, +20.7] | 0.608 |
| local_residual_krr_l1_a0.001 | 0.3084 | 0.4268 | 0.4419 | 0.4010 | +14.2% [+12.4, +16.2] | 0.647 |
| action_qk_a0.1_s0.25 | 0.3144 | 0.4792 | 0.4911 | 0.4485 | +27.8% [+25.5, +30.0] | 0.333 |
| action_qk_a0.1_s1 | 0.3156 | 0.4339 | 0.4477 | 0.4137 | +17.9% [+15.3, +20.5] | 0.435 |
| action_qk_a1_s0.25 | 0.3780 | 0.4883 | 0.4997 | 0.4656 | +32.7% [+29.0, +36.2] | 0.307 |
| action_qk_a1_s1 | 0.3581 | 0.4450 | 0.4586 | 0.4287 | +22.2% [+19.1, +25.3] | 0.420 |

**Primary CPU-trained candidate, by task.** RMS10 is episode-balanced; no task is removed.

| Task | Episodes | Reference RMS10 | Student RMS10 | Excess |
|---|---:|---:|---:|---:|
| 0 | 7 | 0.4704 | 0.5172 | +10.0% |
| 1 | 8 | 0.3308 | 0.3576 | +8.1% |
| 2 | 8 | 0.2886 | 0.3149 | +9.1% |
| 3 | 7 | 0.3249 | 0.3655 | +12.5% |
| 4 | 8 | 0.4083 | 0.4279 | +4.8% |
| 5 | 6 | 0.2206 | 0.2491 | +13.0% |
| 6 | 8 | 0.3417 | 0.4204 | +23.0% |
| 7 | 7 | 0.4596 | 0.4786 | +4.1% |
| 8 | 7 | 0.2650 | 0.2877 | +8.6% |
| 9 | 7 | 0.3998 | 0.4418 | +10.5% |

Phase proxies on all looks: top-1 agreement 0.311; backwards jumps greater than .10 donor progress 0.048 versus reference 0.037; any backwards movement 0.142 versus 0.109; mean donor-progress difference from teacher 0.040; >1/3 disagreement 0.024.

### pi05_spatial_50

100 unseen B-dev episodes, 1221 looks, 189 policy-labeled looks.

| Candidate | Library LOEO RMS10 balanced | B-dev RMS5 | B-dev RMS10 | B-dev RMS10 balanced | Balanced excess vs reference (95% interval) | Top-16 overlap |
|---|---:|---:|---:|---:|---|---:|
| deployed_key_geometry | 0.4136 | 0.8003 | 0.8074 | 0.5640 | +0.0% [+0.0, +0.0] | 1.000 |
| symmetric_rgb_spatial_awm | 0.4345 | 0.8784 | 0.8833 | 0.5899 | +4.6% [+1.1, +8.4] | 0.729 |
| asymmetric_ridge_0.01 | 0.4136 | 0.8983 | 0.8903 | 0.5960 | +5.7% [+2.3, +9.1] | 0.787 |
| asymmetric_ridge_0.1 | 0.4149 | 0.8977 | 0.8886 | 0.5869 | +4.1% [+0.6, +7.7] | 0.802 |
| asymmetric_ridge_1 | 0.4219 | 0.8932 | 0.8892 | 0.5798 | +2.8% [-1.7, +7.1] | 0.816 |
| state_only_awm | 0.4850 | 0.8337 | 0.8351 | 0.5824 | +3.3% [-1.4, +8.1] | 0.722 |

Spatial is secondary: its 189 policy labels are selected MISS states, not a random/all-look sample; 1,032 HIT looks lack policy targets. Phase and neighbor diagnostics use all looks, but the error table uses labeled MISS looks. This cannot clear the offline promotion gate.

## CPU feasibility and prior GPU work

| Cell | CPU completed epochs | CPU training seconds | Train rows / episodes |
|---|---:|---:|---|
| π0.5 L10-50 | 80 | 108.0 | 2925 / 50 |
| π0.5 L10-200 | 80 | 372.2 | 11437 / 200 |

These are full CPU fits from fresh initialization, not CPU inference of GPU-trained weights. The earlier four CNN GPU fits and two GPU latency jobs completed under the original authorization before the interrupt delivering the CPU-preference directive. Their logs record the fresh free-memory gate and a 3-GiB allocator cap, stricter than the allowed 4-GiB cap. Their CPU paths were feasible; they are not presented as CPU-infeasible exceptions. No GPU was used after the addendum was delivered. Future GPU paths now refuse to run without an explicit CPU-infeasibility reason.

## Interpretation and limitations

The action-aware symmetric student performs well enough on its training-library representation to make donor-LOEO look promising. The unseen B-dev excess remains substantially larger. This is consistent with representation overfitting/domain shift, but the experiments do not isolate which mechanism dominates. The symmetric/asymmetric comparison shows that code-space consistency helps somewhat; it does not solve the generalization gap.

This is a bounded prototype search, not evidence that a light key is impossible. In particular, no 112-pixel model, temporal student, stronger augmentation or additional library-only episode-cross-fitting study was completed. Those would be new versioned offline experiments. External pretraining is neither used nor proven necessary. No SR estimate is obtained by converting RMS error. Full guard/corrector refits and the real tower-bypass plugin remain implementation work, so none of these files is a production artifact.
