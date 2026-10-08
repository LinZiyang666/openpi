这次每百次观察只有约 2–12 次真正只用一个相机，严格版本不到 1 次。主要原因是：观察前，示范库中支持“少看也安全”的权重不足；看完第一个相机后，两种估计给出的动作或进度又经常不一致，只好补看。早期预测过于乐观，后来修正的预测其实已接近这次实测。放宽动作限制能增加覆盖，但要达到四分之一，还需放宽安全检查，并出现漏调用和夹爪分歧。目前没有充分的开发集依据支持“成功率不变、至少四分之一只看一个相机”；不建议进入正式测试。

**Evidence and forecast correction.** These are the seven 100-episode B pilots, knob off. I verified shard hashes and B contracts, selected accepted journal attempts, excluded six abandoned-attempt shards, and reconciled all **14,336 looks** to the scored ledgers. No test-A data or new rollouts were used.

The early [forecast](PREDICTION.md) predicted 15–29% from library-atlas eligibility transported to dev trajectories. That was an optimistic forecast of the earlier gate, before requiring both current-observation witnesses. The final valid-root [replay report](REPORT.md) already predicted **6.07%, 6.55%, 4.96%, 2.13%** for π0.5 L10-50, π0.5 Spatial-50, GR00T Spatial-50 and GR00T L10-50; pilots gave **6.43%, 6.93%, 4.36%, 1.81%**. Thus most of the apparent collapse occurred when replacing the optimistic atlas rule with the safer consensus rule, before closed-loop evaluation. The final forecast explicitly withdrew the old Spatial-200 source; its early 29.2% is not a valid forecast of this pilot.

**Where coverage went.** `C` = `astra_consensus2` (motion threshold .10); `S` = `astra_strict` (.05). Counts below partition every scored look. “Lifecycle” is the recorded `lifecycle_or_due`; all periodic-due flags are false. “Atlas” is `atlas_or_state` before camera acquisition. Every actual completion has `completion_reason=consensus`; no accepted half subsequently needed policy completion.

| Cell / arm | Looks | One camera, count (%) | Full: lifecycle | Full: atlas/state | Wrist then completion |
|---|---:|---:|---:|---:|---:|
| pi05_l10_50 / C | 3,031 | 195 (6.43) | 1,016 | 1,025 | 795 |
| pi05_l10_50 / S | 3,063 | 26 (0.85) | 840 | 1,107 | 1,090 |
| pi05_spatial_50 / C | 1,256 | 87 (6.93) | 479 | 317 | 373 |
| pi05_spatial_50 / S | 1,243 | 8 (0.64) | 396 | 326 | 513 |
| pi05_spatial_200 / C | 1,088 | 133 (12.22) | 376 | 246 | 333 |
| groot_spatial_50 / C | 1,239 | 54 (4.36) | 405 | 386 | 394 |
| groot_l10_50 / C | 3,416 | 62 (1.81) | 1,397 | 1,453 | 504 |

Each lifecycle column contains 200 warm-up looks. The remaining counts split into previous positive no-progress span / refresh after accepted half: **621/195, 614/26, 194/85, 188/8, 51/125, 152/53, 1,135/62**, in table order. Half looks do not renew full-camera memory: the next look must refresh it.

Individual failed conditions below **overlap**; they are not an ordered attribution. Pre-look limits are safe donor mass ≥.75, displacement RMS ≤.2, successor support ≥.99. Completion limits are motion RMS ≤.10/.05, witness progress difference ≤.5 donor steps, identical gripper signs, and H advancement >.5 steps.

| Cell / arm | Safe mass | Displacement | Support | Motion | Progress | Gripper | No advance |
|---|---:|---:|---:|---:|---:|---:|---:|
| pi05_l10_50 / C | 857 | 240 | 142 | 652 | 498 | 54 | 94 |
| pi05_l10_50 / S | 898 | 271 | 153 | 1,078 | 558 | 59 | 98 |
| pi05_spatial_50 / C | 222 | 123 | 117 | 331 | 210 | 32 | 33 |
| pi05_spatial_50 / S | 213 | 131 | 118 | 513 | 227 | 34 | 31 |
| pi05_spatial_200 / C | 225 | 34 | 59 | 260 | 157 | 20 | 18 |
| groot_spatial_50 / C | 317 | 107 | 107 | 351 | 189 | 31 | 18 |
| groot_l10_50 / C | 1,330 | 369 | 175 | 431 | 321 | 27 | 86 |

**Safe-mass failure dominates pre-look rejection in every cell; motion disagreement dominates completion**, occurring in 78–89% of consensus completions and 99–100% of strict completions. Progress is the next largest completion veto. Only 17–43% of consensus looks even reach wrist-first acquisition; only 11–29% of those attempts survive. This explains the small camera saving without attributing it to call-count fluctuations.

**Can a dev-supported variant reach 25% at flat SR? No recommendation meets both conditions.** I recomputed proposals on the five matching `off` B recordings (10,268 looks), using the existing selected-library fits, and replayed full-memory refresh and donor-progress guards. Full-only replay exactly reproduces recorded spans/calls; 50 sampled pilot proposals reproduce witness RMS within 6×10⁻⁸, with identical top donors, progress and gripper checks. States/look slots stay recorded; after divergence these are diagnostics, not new trajectories or SR estimates. The off runs disable execution auditing, so this optimistic shadow assumes their recorded successful five-control blocks would be acknowledged. GR00T estimates also remain conditional on the recorded encoder keys.

The following percentages charge completion on baseline policy calls and retain support, displacement, progress and gripper checks. The first three columns keep safe mass .75; the last relaxes it to .50.

| Cell | Motion .10 | Motion .20 | No motion veto | Motion .20, mass .50 |
|---|---:|---:|---:|---:|
| pi05_l10_50 | 6.14 | 11.30 | 12.32 | 13.87 |
| pi05_spatial_50 | 6.14 | 12.52 | 14.94 | 13.49 |
| pi05_spatial_200 | 12.96 | 22.98 | 23.53 | 26.75 |
| groot_spatial_50 | 4.35 | 12.44 | 14.39 | 15.01 |
| groot_l10_50 | 1.77 | 4.52 | 5.29 | 6.38 |

Spatial-200 is the only plausible coverage lead. Merely rethresholding its observed pilot attempts at .20 suggests 25.28%, but freezes the old memory history; replay with refresh gives 22.98% on the off recordings. At .20/mass .75, fixed-call saving is **.00894**, with no missed/extra calls or gripper disagreements, but 16 donor-progress disagreements with full vision: promising for more B investigation, still below 25%. A finer dev screen reaches **25.28% at .20/mass .65** (275/1,088, fixed-call saving .00983), but introduces **one missed call, two gripper disagreements and 20 progress disagreements**. Mass .50 gives 26.75% with the same call/gripper issues. Removing both witnesses gives 30.33%, but two missed/six extra calls and four gripper disagreements. These observations do not prove an SR loss; they do prevent a defensible flat-SR recommendation. No SR is inferred from action error or the original episodes’ success labels.

**Verdict.** The implemented gates behaved close to the revised replay expectation: they purchased agreement by completing most partial looks, leaving only .00025–.00476 direct IR saving across the seven arms. The strict rule is especially unproductive. The coordinator’s screen-out is justified; apparent total-IR gains beyond the camera ledger are not evidence of a better method given roughly .005 deployed-replicate IR noise, and 100 episodes do not establish unchanged SR. Keep Spatial-200’s moderate relaxation as a development hypothesis only; no examined ≥25% variant has enough evidence to advance to test A.

Evidence: [gate counts and provenance](pilot_analysis.json), [counterfactual replay](pilot_replay_pi05_spatial_200.json), [finer Spatial-200 screen](pilot_replay_spatial200_followup.json), [numerical checks](pilot_replay_validation.json). Reproduction scripts: [pilot_analysis.py](pilot_analysis.py), [pilot_replay.py](pilot_replay.py). Run with `taskset -c 22-37,66-81 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python` followed by the script path; `pilot_replay.py --cached` reuses extracted dev proposals. All new artifacts remain in this directory.
