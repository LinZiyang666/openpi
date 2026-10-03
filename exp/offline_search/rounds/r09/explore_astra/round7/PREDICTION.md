给负责人的预测：两个候选都明显降低了离线动作误差；更宽的共享头基本补上了原先的拟合缺口。闭环仍可能受抓取时机和轨迹分布变化影响，不能据此宣称成功率已经追平。

Frozen before arm emission: **2026-10-02 11:41:27 UTC**.
Training-only selection freeze: **2026-10-02T11:37:48.282011+00:00** (`SELECTION.json`).
Round6 outcomes on 20–29 were used to diagnose the gap and design this round. Round7 candidate tuning uses only grouped CV on 0–19; 20–29 offline results were inspected only after selection. This is an iterative development screen on reused evaluation inits, not an untouched final test. No round7 closed-loop data exists.

## Frozen candidates and paired controls

GR00T only, LIBERO-10-50 and Spatial-50. Two candidates per cell and one shared same-batch per-task control per cell: six arms, 100 episodes each, tasks 0–9 × manifest inits 20–29. Both controls are byte-identical copies of the corresponding round6 control. No pi0.5 arm is emitted.

Both candidates: one cell-level shared ridge head with 3072 RFFs, alpha 100, seed 260602, 231 observation/action inputs; a residual table indexed only by the existing retrieved rows, lambda 1; fit on A/CU/IP inits 0–19. The 24 added features describe GR00T normalized closure (-1 closes), chunk motion and their interactions. No task id, one-hot, task selector, task head or task threshold is added. Only ten controls × six motion channels are changed.

- `capacity`: fixed strength 0.5.
- `confidence`: strength `0.5 + 0.25 / (1 + v/s)`, where `v` is weighted motion variance among retrieved demonstration chunks and `s` is its training-cell median. Variance uses first 10 controls, channels 0–5; it excludes gripper and padding. Strength is in [0.5, 0.75]. `s=0.011751592822234036` for long, `0.014458684080732365` for spatial. This measures retrieval agreement, not calibrated probability or absolute distance.

The heads/tables in these two variants are identical within each cell. All other stack fits, guards, policy calls, clients, standard-store paths and ten-control commitment semantics remain those of the same-batch control. GR00T keeps resize 256 and horizon 16; correction/commitment remains first 10.

## Subjective forecasts before emission

Differences are candidate minus the **new same-batch control**, not minus the historical round6 score.

| Cell | Variant | Predicted SR difference | Predicted owner IR difference | Illustrative SR if control repeats round6 |
|---|---|---:|---:|---:|
| GR00T LIBERO-10-50 | capacity | −0.02 | +0.005 | 0.84 |
| GR00T LIBERO-10-50 | confidence | −0.01 | +0.003 | 0.85 |
| GR00T Spatial-50 | capacity | −0.02 | +0.005 | 0.96 |
| GR00T Spatial-50 | confidence | −0.02 | +0.004 | 0.96 |

Plausible paired SR differences extend approximately ±0.06 around these points (clipped by the SR ceiling); these are judgment forecasts, not calibrated intervals. The confidence arm has greater teacher-error improvement but also greater action change, and only 46/100 Spatial A-path episodes improve over the control in offline MSE. There is no assumption that lower teacher MSE guarantees higher SR. Added CPU latency is not priced by owner stage IR.

## Evaluation rules fixed now

1. Score all 100 admitted pairs for every arm; use accepted completed attempts and verify contiguous server decisions. No task dropping, replacement control or manifest expansion.
2. Report paired SR delta, +/− discordant counts, 5000-draw init-cluster interval (all ten tasks retained), aggregate owner IR, looks, calls, decisions and episode counts. Retain per-task/phase diagnostics for explanation only; never feed them to the corrector.
3. Practical candidate screen: SR delta >= −0.02 and owner IR delta <= +0.005 **in each cell**. This finite screen is not a statistical proof of noninferiority. Prefer higher equal-cell SR; use lower IR to break a tie. If neither passes, keep the per-task control.
4. Define gap narrowing descriptively relative to round6's −0.06/−0.05, but use new same-batch controls for every acceptance decision. Do not silently retune after results. No launches or remote actions are part of this package.
