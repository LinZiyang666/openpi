# PROTOTYPE2 — image-driven CPU query keys

The fourth owner directive permits the best image-driven student to enter a B-dev pilot even if the original offline accuracy gates fail. GATES.md is byte-identical; this report keeps numerical results separate from pilot permission. Every candidate reported here uses its own symmetric light features, the same per-task action-supervised AWM and early fit, and the same 16-neighbor synthesis. State-only is an ineligible negative control. No online run or GPU use occurred in this round.

## Library-only experiment and selection

Two deterministic episode hold-outs per cell: one episode/task held out at size 50, four/task at size 200. Student weights, normalization, action sigma, positive pairs and fold AWM are fitted only on the remaining episodes. All recipes use the same splits and select among 40/80/120 epochs. The fixed full-cell teacher PCA coordinate system is reused as the distillation target basis; this is student episode hold-out, not a fully cross-fitted teacher representation. No B-dev image or label enters fitting or recipe/epoch selection.

The new recipes combine small translations/scales and color/brightness changes, per-task target normalization, weight decay .01 and dropout .1. `vision_aug` removes the state shortcut inside the CNN; state still enters the downstream AWM. `token_aug` adds weight .3 spatial-token supervision. `token_keypoint` replaces flattened maps with spatial-softmax expected coordinates. `state_drop_aug` keeps an internal state branch with whole-branch dropout .5. `r1_style` is a fresh original-style global-normalization/state-conditioned control, with no augmentation/dropout and weight decay .001, added before new-candidate B-dev assessment. It is not byte-identical to the old checkpoint: it uses the v2 module/input concatenation and a fresh seed.

Spatial targets use every selected row's full 256×2048 token map from both cameras, average pooled to 4×4, followed by a fixed signed orthonormal channel-group projection 2048→16. The resulting 512-dimensional target is auxiliary supervision, not a current-look input. The projection learns nothing from other rows. This is a bounded, low-cost spatial distillation experiment; it does not test every use of full-resolution token maps.

Implementation: shared 3→16 (5×5/stride2), 16→24 and 24→32 (3×3/stride2) SiLU convolutions retain a 7×7 map/camera. The flat models concatenate both maps, an 8-D task embedding and the optional normalized state, then use a 256-unit SiLU head to predict 128 teacher-PCA coordinates. Keypoint models use softmax temperature factor 5 and expected x/y coordinates instead. State is standardized on fit episodes and clipped to ±8. AdamW uses learning rate .001, batch 96 and gradient clip 5. Seeds are 2026100506/2026100507 for folds 0/1 and 2026100506 for fresh final fits. Loss is standardized PCA MSE + teacher-AWM-code MSE + .1 action smooth-L1 + .5 cross-episode action-pair hinge, plus the optional .3 spatial loss. Positives are among the three nearest same-task action heads from other episodes; negatives must be other-episode and action-distance worse by .05; hinge margin .1. Augmentation uses ±1 pixel translation, ±2.5% scale, contrast .9–1.1, channel gain .95–1.05 and brightness ±.05; no flips. Full implementation and exact normalizers are retained in the scripts/checkpoints.

Fresh final fits use the complete exact own-cell library after recipe/epoch selection. Their all-row donor-LOEO assessment precedes existing unseen B-dev. No history is needed by the selected interface; current images, state and task are all available online. Temporal/action/history candidates were not trained within this time box and remain untested options. The two-fold comparisons are development selection, not independent confirmation. The chosen recipe changes several regularizers together, so its gain cannot be attributed to state dropout alone.

## Hold-out selection

| Cell | Recipe | Best epoch | Mean two-fold RMS10 ratio | Fold ratios | Shuffle error rise | Zero-image error rise | Image audit |
|---|---|---:|---:|---|---:|---:|---|
| pi05_l10_50 | r1_style | 40 | 1.1120 | 1.0631, 1.1609 | 144.6% | 107.3% | True |
| pi05_l10_50 | state_drop_aug | 40 | 1.0863 | 1.0543, 1.1183 | 155.9% | 114.7% | True |
| pi05_l10_50 | token_aug | 80 | 1.0953 | 1.0712, 1.1194 | 155.3% | 106.7% | True |
| pi05_l10_50 | token_keypoint | 80 | 1.1142 | 1.0836, 1.1449 | 148.4% | 96.5% | True |
| pi05_l10_50 | vision_aug | 120 | 1.1004 | 1.0765, 1.1242 | 154.0% | 105.0% | True |
| pi05_l10_200 | r1_style | 40 | 1.0951 | 1.0919, 1.0984 | 231.7% | 156.1% | True |
| pi05_l10_200 | state_drop_aug | 120 | 1.0850 | 1.0764, 1.0935 | 243.5% | 150.3% | True |
| pi05_l10_200 | token_aug | 120 | 1.0882 | 1.0833, 1.0931 | 242.0% | 156.2% | True |
| pi05_l10_200 | token_keypoint | 120 | 1.1475 | 1.1451, 1.1500 | 221.1% | 131.5% | True |
| pi05_l10_200 | vision_aug | 120 | 1.0910 | 1.0867, 1.0952 | 241.3% | 158.5% | True |

All epoch/fold results and image audits are retained in `round2/<cell>/<recipe>/cv{0,1}_train.json`; selection timestamps and input hashes are in each cell's `selection.json`. These candidates were not ranked using B-dev. The frozen original checkpoint is assessed separately as an incumbent reference.

## Final symmetric comparison on existing B-dev

| Cell | Candidate | Library LOEO ratio | B-dev balanced ratio | RMS5 / RMS10, look-weighted | B-dev excess 95% interval | Top-16 overlap | Shuffle rise | Zero rise |
|---|---|---:|---:|---|---|---:|---:|---:|
| pi05_l10_50 | r1_action | 0.9950 | 1.1250 | 0.4731 / 0.4913 | [+11.2%, +13.9%] | 0.696 | 156.0% | 114.2% |
| pi05_l10_50 | state_drop_aug | 1.0146 | 1.0983 | 0.4615 / 0.4787 | [+8.3%, +11.4%] | 0.691 | 169.0% | 117.5% |
| pi05_l10_200 | r1_action | 1.0168 | 1.1000 | 0.4079 / 0.4217 | [+8.3%, +11.8%] | 0.630 | 203.5% | 128.3% |
| pi05_l10_200 | state_drop_aug | 1.0299 | 1.0802 | 0.3978 / 0.4118 | [+6.2%, +9.8%] | 0.643 | 217.5% | 131.9% |

Size 50 uses 100 unseen episodes / 3,198 looks; size 200 uses the 73 episodes / 2,289 looks outside its own subset. Both are existing size-50 controller trajectories, not new size-200 rollouts. The reference is controlled layer-1 deployed-key geometry, without insurance/guard/corrector effects. Equal-task/episode RMS10 is primary; intervals use 2,000 paired stratified episode draws, seed 20261005. Donor-LOEO fixes the full-library-trained representation and excludes the complete query episode only from donors.

The original student is image-sensitive too: closeness of its clean error to state-only did not establish blindness. Shuffling may create an image/state inconsistency, so a large ablation effect is necessary evidence of image use, not proof of improved clean accuracy or SR. The state-only negative control has exactly zero change under either image operation because its input is unchanged; it is rejected regardless of its error.

## Per-task final error and image-use audit

| Cell | Task | Reference RMS10 | Student RMS10 | Excess | Shuffle rise | Zero rise |
|---|---:|---:|---:|---:|---:|---:|
| pi05_l10_50 | 0 | 0.5502 | 0.5752 | +4.5% | 92.9% | 42.4% |
| pi05_l10_50 | 1 | 0.3960 | 0.4370 | +10.3% | 180.1% | 124.0% |
| pi05_l10_50 | 2 | 0.3445 | 0.3852 | +11.8% | 234.3% | 193.7% |
| pi05_l10_50 | 3 | 0.3881 | 0.4569 | +17.7% | 219.1% | 152.5% |
| pi05_l10_50 | 4 | 0.4675 | 0.4887 | +4.5% | 98.9% | 60.7% |
| pi05_l10_50 | 5 | 0.2713 | 0.2906 | +7.1% | 263.2% | 224.4% |
| pi05_l10_50 | 6 | 0.3920 | 0.4452 | +13.6% | 135.3% | 89.5% |
| pi05_l10_50 | 7 | 0.4853 | 0.5302 | +9.3% | 120.1% | 76.6% |
| pi05_l10_50 | 8 | 0.2947 | 0.3307 | +12.2% | 226.8% | 165.5% |
| pi05_l10_50 | 9 | 0.4666 | 0.5155 | +10.5% | 207.0% | 136.7% |
| pi05_l10_200 | 0 | 0.4704 | 0.4890 | +3.9% | 105.3% | 45.3% |
| pi05_l10_200 | 1 | 0.3308 | 0.3542 | +7.1% | 250.9% | 143.3% |
| pi05_l10_200 | 2 | 0.2886 | 0.3023 | +4.7% | 338.1% | 249.5% |
| pi05_l10_200 | 3 | 0.3249 | 0.3688 | +13.5% | 292.5% | 156.3% |
| pi05_l10_200 | 4 | 0.4083 | 0.4317 | +5.7% | 131.8% | 77.8% |
| pi05_l10_200 | 5 | 0.2206 | 0.2504 | +13.5% | 321.0% | 189.3% |
| pi05_l10_200 | 6 | 0.3417 | 0.4075 | +19.3% | 162.5% | 94.5% |
| pi05_l10_200 | 7 | 0.4596 | 0.4673 | +1.7% | 151.5% | 97.1% |
| pi05_l10_200 | 8 | 0.2650 | 0.2835 | +7.0% | 287.8% | 196.6% |
| pi05_l10_200 | 9 | 0.3998 | 0.4366 | +9.2% | 270.6% | 168.8% |

Blindness uses paired camera images shuffled within task with three fixed seeds, and both cameras set to raw black, while state/task/step and the library index remain fixed. The new owner requirement was operationalized before results as ≥5% balanced error rise for both mean shuffle and zero, plus ≥8/10 tasks worsening under shuffle. The numerical thresholds in GATES.md were not changed.

## Negative control and paired improvement

| Cell | State-only B-dev / reference | State-only shuffle / zero rise | Selected / state-only excess (95% interval) | Selected / original student excess (95% interval) |
|---|---:|---|---|---|
| pi05_l10_50 | 1.1378 | 0% / 0% — rejected | -3.47% [-5.30%, -1.68%] | -2.37% [-3.37%, -1.34%] |
| pi05_l10_200 | 1.1647 | 0% / 0% — rejected | -7.25% [-9.31%, -5.34%] | -1.80% [-2.77%, -0.81%] |

State-only uses the same per-task AWM/early metric and 16-NN synthesis on state alone; its frozen round-1 arrays are reused and hash-checked. Zero audit effect follows exactly from its image-independent function, so it is ineligible regardless of clean accuracy. Paired intervals use the same 2,000 equal-task episode resamples. Negative excess means the selected model improves that comparator; these reused development comparisons do not establish causal attribution to one regularizer.

## Tail, neighbor and phase diagnostics

| Cell | Scope | RMS5 | RMS10 p90 | Top-1 agreement | Top-16 overlap | Known library phase MAE |
|---|---|---:|---:|---:|---:|---:|
| pi05_l10_50 | library | 0.3365 | 0.6185 | 0.580 | 0.864 | 0.1009 |
| pi05_l10_50 | b_dev | 0.4615 | 0.8889 | 0.318 | 0.691 | unlabeled |
| pi05_l10_200 | library | 0.3027 | 0.5520 | 0.494 | 0.767 | 0.0576 |
| pi05_l10_200 | b_dev | 0.3978 | 0.7969 | 0.343 | 0.643 | unlabeled |

| Cell | Student / reference large backward rate | Teacher phase discrepancy >1/3 |
|---|---|---:|
| pi05_l10_50 | 0.0539 / 0.0423 | 0.0563 |
| pi05_l10_200 | 0.0469 / 0.0370 | 0.0227 |

Tail and phase summaries are look-weighted diagnostics. Known library phase uses the query row's stored progress and the top donor's progress; B-dev has only the stated teacher/backward proxies. Complete early/stale and per-task summaries remain in each assessment JSON.

## Unchanged gate application and local cost

| Cell | Selected recipe / epochs | Library gate | B-dev gate | Image audit | CPU median / p95 ms | Local cost gate |
|---|---|---|---|---|---|---|
| pi05_l10_50 | state_drop_aug / 40 | True | False | True | 1.028 / 1.202 | False |
| pi05_l10_200 | state_drop_aug / 120 | True | False | True | 0.958 / 1.127 | True |

| Cell | One-thread CPU IQR ms | Epsilon: graph basis | Epsilon: historical R12 eager sensitivity |
|---|---|---:|---:|
| pi05_l10_50 | 0.963–1.124 | 0.015229 | 0.002282 |
| pi05_l10_200 | 0.932–0.998 | 0.014188 | 0.002126 |

The R12 eager sensitivity uses its previously measured 68.484-ms π0.5 tower and the same .152 owner coefficient (sources in COST.md). It is not H100 deployment timing and cannot convert a conservative graph-budget failure into a pass.

Local timing uses one CPU thread, 400 timed raw-image-to-main-AWM-code calls across all ten tasks after warmup; four-thread timing is also saved. Raw two-camera bundles are prepared outside the timer; resize, float conversion, state/task tensor conversion, model inference and AWM projection are timed. Generic loader checks, camera-bundle construction and common neighbor synthesis are not timed. These are encoder-path measurements, not complete serving latency. H100 deployment-host timing and the complete five-layer runtime verification belong to sol/coordinator and are not asserted here. Failure of an old offline numerical gate is reported explicitly but no longer vetoes this owner-authorized pilot.

## Cost projection at the frozen size-50 epsilon

| R14 B-dev operating point | Current IR | Projected IR | Relative saving |
|---|---:|---:|---:|
| r14d_pi05_l10_50_periodic_ir25 | 0.2359 | 0.1956 | 17.1% |
| r14d_pi05_l10_50_off | 0.1689 | 0.1166 | 30.9% |
| fixed R14 task table (stitched) | 0.2264 | 0.1844 | 18.6% |

These use the already reconciled B-only counts in cost_counts.json and hold N/V/M fixed: IR_new=(epsilon*V+M)/N. They do not predict the extra policy calls or changed trajectory lengths caused by the new key. No size-200 count projection is borrowed from size 50; the matched pilot must measure its own cost. The task table is only a fixed-table projection, not an added pilot arm.

## Verdict

Continue with the bounded, paired B-dev pilot in PILOT.md using the frozen hold-out-selected image-driven artifacts. The exploratory search does not establish that the generalization gap is solved. Keep the original matched-SR and total-cost criteria for interpreting the pilot; do not treat image sensitivity, a library result or owner pilot permission as an accuracy pass. No further architecture changes should enter that pilot after artifact freeze.

Sources: round2 training/selection/assessment JSON, `round2/pilot/*/{manifest,latency}.json`, `round2/pilot_identity_baseline.json`, `round2_metrics.csv`, `gate_report.json`. All scripts and outputs are in the authorized astra/derived roots.
