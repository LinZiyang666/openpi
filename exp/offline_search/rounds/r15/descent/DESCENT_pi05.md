# 给负责人的结果说明

冻结规则不变：每任务 30 个配对 B 初始状态，以纯 C10 为参照按 ir45、ir40、ir32、ir25、off 下降，首次失败即停止。主容差 5 个百分点；0 和 2 个百分点保留为备选。

500 集库在服务时排除当前 B 初始状态的自身库轨迹；拟合仍包含该轨迹，因此存在轻微乐观偏差。这里只使用开发集。

部分模型结果：pi05。每个所选单元的全部 B 臂与共享 C10 必须完整；未读取其他模型结果。跨单元汇总延后至完整分析。

PARTIAL reporting: the fixed 200-comparison family is retained within each library-size group; only the selected model cells are reported. Confidence sensitivity is reporting only and never changes selection.

## Tolerance 0 pp

| Cell | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | Uniform | Task SR % | Uniform SR % | Task IR | Uniform IR | IR saved pp | Leave own out |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pi05_l10_100 | ir32 | ir40 | ir40 | ir40 | c10 | c10 | ir45 | ir45 | c10 | ir25 | c10 | 92.333 | 90.333 | 0.415 | 0.504 | 8.893 | no |
| pi05_spatial_100 | c10 | ir45 | ir45 | off | off | c10 | off | ir45 | c10 | ir25 | c10 | 98.000 | 96.333 | 0.337 | 0.511 | 17.422 | no |
| pi05_l10_200 | off | off | ir40 | ir40 | c10 | ir45 | ir45 | ir40 | c10 | off | ir25 | 91.667 | 91.333 | 0.355 | 0.244 | -11.144 | no |
| pi05_spatial_200 | c10 | ir40 | ir32 | ir40 | off | c10 | off | c10 | ir45 | ir32 | c10 | 96.667 | 96.333 | 0.355 | 0.511 | 15.636 | no |
| pi05_l10_500 | off | off | c10 | off | c10 | c10 | ir45 | ir45 | c10 | c10 | c10 | 91.000 | 90.333 | 0.389 | 0.504 | 11.499 | yes |
| pi05_spatial_500 | c10 | ir32 | ir40 | c10 | off | off | ir32 | ir45 | c10 | off | c10 | 96.667 | 96.333 | 0.322 | 0.511 | 18.898 | yes |


## Tolerance 2 pp

| Cell | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | Uniform | Task SR % | Uniform SR % | Task IR | Uniform IR | IR saved pp | Leave own out |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pi05_l10_100 | ir32 | ir40 | ir40 | ir40 | c10 | c10 | ir45 | ir45 | c10 | ir25 | off | 92.333 | 88.667 | 0.415 | 0.155 | -25.984 | no |
| pi05_spatial_100 | c10 | ir45 | ir45 | off | off | c10 | off | ir45 | c10 | ir25 | ir40 | 98.000 | 95.333 | 0.337 | 0.395 | 5.733 | no |
| pi05_l10_200 | off | off | ir40 | ir40 | c10 | ir45 | ir45 | ir40 | c10 | off | off | 91.667 | 89.667 | 0.355 | 0.150 | -20.498 | no |
| pi05_spatial_200 | c10 | ir40 | ir32 | ir40 | off | c10 | off | c10 | ir45 | ir32 | ir25 | 96.667 | 94.667 | 0.355 | 0.253 | -10.156 | no |
| pi05_l10_500 | off | off | c10 | off | c10 | c10 | ir45 | ir45 | c10 | c10 | c10 | 91.000 | 90.333 | 0.389 | 0.504 | 11.499 | yes |
| pi05_spatial_500 | c10 | ir32 | ir40 | c10 | off | off | ir32 | ir45 | c10 | off | off | 96.667 | 94.667 | 0.322 | 0.108 | -21.479 | yes |


## Tolerance 5 pp

| Cell | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | Uniform | Task SR % | Uniform SR % | Task IR | Uniform IR | IR saved pp | Leave own out |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pi05_l10_100 | ir32 | off | ir40 | off | ir32 | c10 | ir32 | off | c10 | off | off | 90.000 | 88.667 | 0.298 | 0.155 | -14.269 | no |
| pi05_spatial_100 | ir45 | ir40 | off | off | off | c10 | off | ir40 | ir45 | off | off | 95.667 | 94.333 | 0.272 | 0.114 | -15.730 | no |
| pi05_l10_200 | off | off | off | off | ir32 | ir45 | off | ir32 | off | off | off | 92.000 | 89.667 | 0.201 | 0.150 | -5.054 | no |
| pi05_spatial_200 | ir32 | off | off | off | off | c10 | off | c10 | ir45 | off | off | 96.333 | 93.333 | 0.241 | 0.113 | -12.729 | no |
| pi05_l10_500 | off | off | c10 | off | c10 | c10 | off | ir32 | off | ir25 | off | 93.333 | 92.000 | 0.266 | 0.148 | -11.781 | yes |
| pi05_spatial_500 | ir40 | off | off | off | off | off | ir25 | off | ir45 | off | off | 95.667 | 94.667 | 0.179 | 0.108 | -7.179 | yes |


R14 point-estimate selection, paired CP intervals, conditional paired bootstrap (5,000, seed 20261005), and fixed owner IR coefficients are unchanged. The registered 200-comparison simultaneous band applies separately to each four-cell library-size group, not jointly across all 600 R15 comparisons. C10 is shared within model/suite. R14 task_periodic allocation arms are not repeated in R15, so allocation agreement and matched-allocation comparisons are not available. B-selected stitched estimates remain optimistic and are not independent mixed-policy measurements.
