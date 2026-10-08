# 给负责人的结果说明

冻结规则不变：每任务 30 个配对 B 初始状态，以纯 C10 为参照按 ir45、ir40、ir32、ir25、off 下降，首次失败即停止。主容差 5 个百分点；0 和 2 个百分点保留为备选。

500 集库在服务时排除当前 B 初始状态的自身库轨迹；拟合仍包含该轨迹，因此存在轻微乐观偏差。这里只使用开发集。

部分模型结果：groot。每个所选单元的全部 B 臂与共享 C10 必须完整；未读取其他模型结果。跨单元汇总延后至完整分析。

PARTIAL reporting: the fixed 200-comparison family is retained within each library-size group; only the selected model cells are reported. Confidence sensitivity is reporting only and never changes selection.

## Tolerance 0 pp

| Cell | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | Uniform | Task SR % | Uniform SR % | Task IR | Uniform IR | IR saved pp | Leave own out |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| groot_l10_100 | c10 | c10 | off | ir32 | off | ir32 | c10 | c10 | c10 | c10 | c10 | 90.333 | 87.667 | 0.411 | 0.504 | 9.284 | no |
| groot_spatial_100 | c10 | off | c10 | c10 | c10 | c10 | off | c10 | off | c10 | c10 | 92.333 | 89.667 | 0.391 | 0.510 | 11.852 | no |
| groot_l10_200 | ir45 | c10 | off | ir45 | off | off | c10 | c10 | ir40 | c10 | ir45 | 91.667 | 88.333 | 0.392 | 0.448 | 5.572 | no |
| groot_spatial_200 | ir45 | ir45 | off | c10 | off | c10 | off | c10 | off | c10 | c10 | 92.667 | 89.667 | 0.343 | 0.510 | 16.715 | no |
| groot_l10_500 | c10 | ir40 | off | ir25 | off | off | c10 | ir40 | ir45 | c10 | ir40 | 91.333 | 89.000 | 0.365 | 0.393 | 2.858 | yes |
| groot_spatial_500 | c10 | ir45 | c10 | off | ir32 | ir32 | ir40 | ir25 | off | c10 | c10 | 91.000 | 89.667 | 0.353 | 0.510 | 15.694 | yes |


## Tolerance 2 pp

| Cell | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | Uniform | Task SR % | Uniform SR % | Task IR | Uniform IR | IR saved pp | Leave own out |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| groot_l10_100 | c10 | c10 | off | ir32 | off | ir32 | c10 | c10 | c10 | c10 | ir32 | 90.333 | 88.000 | 0.411 | 0.317 | -9.393 | no |
| groot_spatial_100 | c10 | off | c10 | c10 | c10 | c10 | off | c10 | off | c10 | off | 92.333 | 90.000 | 0.391 | 0.126 | -26.535 | no |
| groot_l10_200 | ir45 | c10 | off | ir45 | off | off | c10 | c10 | ir40 | c10 | ir32 | 91.667 | 88.667 | 0.392 | 0.314 | -7.858 | no |
| groot_spatial_200 | ir45 | ir45 | off | c10 | off | c10 | off | c10 | off | c10 | ir45 | 92.667 | 88.667 | 0.343 | 0.452 | 10.884 | no |
| groot_l10_500 | c10 | ir40 | off | ir25 | off | off | c10 | ir40 | ir45 | c10 | off | 91.333 | 86.000 | 0.365 | 0.184 | -18.107 | yes |
| groot_spatial_500 | c10 | ir45 | c10 | off | ir32 | ir32 | ir40 | ir25 | off | c10 | off | 91.000 | 90.333 | 0.353 | 0.122 | -23.096 | yes |


## Tolerance 5 pp

| Cell | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | Uniform | Task SR % | Uniform SR % | Task IR | Uniform IR | IR saved pp | Leave own out |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| groot_l10_100 | ir45 | ir25 | off | off | off | ir32 | c10 | ir45 | ir32 | c10 | ir25 | 90.333 | 83.000 | 0.332 | 0.255 | -7.752 | no |
| groot_spatial_100 | ir40 | off | c10 | c10 | off | c10 | off | ir45 | off | c10 | off | 91.667 | 90.000 | 0.330 | 0.126 | -20.432 | no |
| groot_l10_200 | ir45 | off | off | off | off | off | c10 | c10 | ir32 | ir45 | off | 89.667 | 83.333 | 0.316 | 0.186 | -13.056 | no |
| groot_spatial_200 | ir45 | ir45 | off | c10 | off | ir45 | off | ir45 | off | c10 | off | 92.000 | 93.333 | 0.331 | 0.122 | -20.970 | no |
| groot_l10_500 | c10 | off | off | off | off | off | c10 | ir32 | off | c10 | off | 90.000 | 86.000 | 0.289 | 0.184 | -10.521 | yes |
| groot_spatial_500 | c10 | ir45 | ir45 | off | off | ir32 | off | ir25 | off | c10 | off | 90.667 | 90.333 | 0.295 | 0.122 | -17.251 | yes |


R14 point-estimate selection, paired CP intervals, conditional paired bootstrap (5,000, seed 20261005), and fixed owner IR coefficients are unchanged. The registered 200-comparison simultaneous band applies separately to each four-cell library-size group, not jointly across all 600 R15 comparisons. C10 is shared within model/suite. R14 task_periodic allocation arms are not repeated in R15, so allocation agreement and matched-allocation comparisons are not available. B-selected stitched estimates remain optimistic and are not independent mixed-policy measurements.
