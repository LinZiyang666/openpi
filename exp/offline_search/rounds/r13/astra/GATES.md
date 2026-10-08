# R13 B pilot 放行审计

GR00T 结论：l10_50：released；l10_200：released；spatial_50：released；l10_500：held。

GR00T Spatial 已通过内容放行门槛；L10 冻结整批含 size-500，transfer 前提未满足时整批 held。冻结本地队列先 L10 后 Spatial，因此不能跳过 held 的 L10、删 arm 或改顺序来声称现在整批可启动。后续 replicate block 仍须等上一完整 block 的两个 fleet 均完成。

规则解释：SPEC 把配对 IR 差 ≤.01 写为直接同预算标签的条件；停止/重设计段落明确针对预算或参与度失败。因此配对差单独失败会取消该点的同预算标签，不冒充控制器预算失败。Size-500 的前提写的是全部可用小库 cell，没有按模型豁免；目前 pi05 L10-50 尚未完成同机验证，两种模型的 size-500 均 held。

本次纳入 64 个完整 B arm、6400 条 episode；其余 0 个 arm 待完成。pi05 L10-50 运行主机改为本地 4090（用户说明是容量决定、未看结果）；即便数据补齐，也不等于冻结要求的 H100 同机验证。脚本可重跑补齐。

判定只用预算、距离参与度、机械及协议检查；成功标签仅进入重复方差/功效审计，没有按 SR 选方法、剂量或 cell。预算或参与度失败保留为 Q1 failure，交回另行版本化的 B-only 重设计；不在本验证集或 A 上调参。

审计时间：2026-10-04T23:00:55.454242+00:00。冻结文件校验 126 项，前后均 通过。

| Cell | Verdict | Budget failing arms | Engagement failing arms | Unmatched pairs |
| --- | --- | ---: | ---: | ---: |
| groot_l10_50 | released | 0 | 0 | 0 |
| groot_l10_200 | released | 0 | 0 | 1 |
| groot_spatial_50 | released | 0 | 0 | 0 |
| groot_l10_500 | held | 0 | 0 | 0 |
| pi05_l10_50 | held | 0 | 0 | 1 |
| pi05_l10_200 | released | 0 | 0 | 0 |
| pi05_spatial_50 | released | 0 | 0 | 0 |
| pi05_l10_500 | held | 0 | 0 | 0 |

Technical conventions: frozen owner cost = .148 V + .852 M (GR00T), .152 V + .848 M (pi05); IR = pooled cost / pooled N. Do not substitute summary latency or current cost-table IR. 95% percentile intervals use 5,000 paired episode/init-cluster resamples, seed 20261004, within each fixed task. Point estimates, not interval endpoints, determine the frozen ≤.02 and |D−P|≤.01 gates. Repeats are checked separately; no pooling rescues failures.

Calibration complexity across the frozen grid: feedback 49 scalars (two scale parameters/cell, one bias/cell/dose, one selected shared width); periodic 32 library-solver caps; static 8 B-only diagnostic thresholds. Exact B/library source paths are retained per cell in gates.json. No scalar is refit here.

| Cell | Method / target / rep | N / V / M | IR [95% CI] | Error | Budget |
| --- | --- | --- | --- | ---: | --- |
| groot_l10_50 | D / 0.25 / 0 | 6375/3209/1315 | 0.2502 [0.2415, 0.2586] | +0.0002 | pass |
| groot_l10_50 | D / 0.32 / 0 | 5930/2982/1714 | 0.3207 [0.3138, 0.3277] | +0.0007 | pass |
| groot_l10_50 | D / 0.32 / 1 | 5982/3014/1715 | 0.3188 [0.3134, 0.3245] | -0.0012 | pass |
| groot_l10_50 | D / 0.40 / 0 | 6480/3258/2492 | 0.4021 [0.3989, 0.4054] | +0.0021 | pass |
| groot_l10_50 | D / 0.45 / 0 | 5587/2816/2468 | 0.4510 [0.4496, 0.4525] | +0.0010 | pass |
| groot_l10_50 | S / 0.32 / 0 | 6343/3193/1778 | 0.3133 [0.2958, 0.3308] | -0.0067 | pass |
| groot_l10_50 | S / 0.40 / 0 | 5932/2988/2115 | 0.3783 [0.3622, 0.3930] | -0.0217 | FAIL |
| groot_l10_50 | P / 0.25 / 0 | 6606/3322/1418 | 0.2573 [0.2464, 0.2677] | +0.0073 | pass |
| groot_l10_50 | P / 0.32 / 0 | 6319/3180/1887 | 0.3289 [0.3228, 0.3350] | +0.0089 | pass |
| groot_l10_50 | P / 0.32 / 1 | 6121/3079/1815 | 0.3271 [0.3206, 0.3334] | +0.0071 | pass |
| groot_l10_50 | P / 0.40 / 0 | 6266/3150/2396 | 0.4002 [0.3946, 0.4059] | +0.0002 | pass |
| groot_l10_50 | P / 0.45 / 0 | 5847/2941/2562 | 0.4478 [0.4424, 0.4530] | -0.0022 | pass |
| groot_l10_200 | D / 0.25 / 0 | 5636/2841/1059 | 0.2347 [0.2268, 0.2425] | -0.0153 | pass |
| groot_l10_200 | D / 0.32 / 0 | 5689/2869/1574 | 0.3104 [0.3063, 0.3148] | -0.0096 | pass |
| groot_l10_200 | D / 0.40 / 0 | 5385/2714/2020 | 0.3942 [0.3921, 0.3965] | -0.0058 | pass |
| groot_l10_200 | D / 0.45 / 0 | 5834/2938/2544 | 0.4461 [0.4445, 0.4476] | -0.0039 | pass |
| groot_l10_200 | P / 0.25 / 0 | 5932/2992/1209 | 0.2483 [0.2396, 0.2571] | -0.0017 | pass |
| groot_l10_200 | P / 0.32 / 0 | 5460/2755/1535 | 0.3142 [0.3079, 0.3210] | -0.0058 | pass |
| groot_l10_200 | P / 0.40 / 0 | 5483/2764/2049 | 0.3930 [0.3875, 0.3985] | -0.0070 | pass |
| groot_l10_200 | P / 0.45 / 0 | 5610/2824/2452 | 0.4469 [0.4422, 0.4516] | -0.0031 | pass |
| groot_spatial_50 | D / 0.25 / 0 | 2452/1250/508 | 0.2520 [0.2450, 0.2588] | +0.0020 | pass |
| groot_spatial_50 | D / 0.32 / 0 | 2534/1289/743 | 0.3251 [0.3205, 0.3296] | +0.0051 | pass |
| groot_spatial_50 | D / 0.32 / 1 | 2533/1286/736 | 0.3227 [0.3170, 0.3286] | +0.0027 | pass |
| groot_spatial_50 | D / 0.40 / 0 | 2349/1201/911 | 0.4061 [0.4027, 0.4095] | +0.0061 | pass |
| groot_spatial_50 | D / 0.45 / 0 | 2302/1176/1028 | 0.4561 [0.4523, 0.4598] | +0.0061 | pass |
| groot_spatial_50 | S / 0.32 / 0 | 2307/1179/745 | 0.3508 [0.3298, 0.3716] | +0.0308 | FAIL |
| groot_spatial_50 | S / 0.40 / 0 | 2314/1183/963 | 0.4302 [0.4168, 0.4430] | +0.0302 | FAIL |
| groot_spatial_50 | P / 0.25 / 0 | 2498/1270/521 | 0.2529 [0.2445, 0.2610] | +0.0029 | pass |
| groot_spatial_50 | P / 0.32 / 0 | 2487/1269/735 | 0.3273 [0.3205, 0.3339] | +0.0073 | pass |
| groot_spatial_50 | P / 0.32 / 1 | 2485/1262/712 | 0.3193 [0.3137, 0.3247] | -0.0007 | pass |
| groot_spatial_50 | P / 0.40 / 0 | 2408/1227/921 | 0.4013 [0.3935, 0.4091] | +0.0013 | pass |
| groot_spatial_50 | P / 0.45 / 0 | 2498/1274/1110 | 0.4541 [0.4476, 0.4605] | +0.0041 | pass |
| pi05_l10_50 | D / 0.25 / 0 | 5776/2907/1156 | 0.2462 [0.2371, 0.2556] | -0.0038 | pass |
| pi05_l10_50 | D / 0.32 / 0 | 5664/2853/1636 | 0.3215 [0.3156, 0.3276] | +0.0015 | pass |
| pi05_l10_50 | D / 0.32 / 1 | 5822/2930/1680 | 0.3212 [0.3149, 0.3283] | +0.0012 | pass |
| pi05_l10_50 | D / 0.40 / 0 | 5612/2833/2136 | 0.3995 [0.3973, 0.4020] | -0.0005 | pass |
| pi05_l10_50 | D / 0.45 / 0 | 5923/2982/2621 | 0.4518 [0.4499, 0.4536] | +0.0018 | pass |
| pi05_l10_50 | S / 0.32 / 0 | 5704/2872/1712 | 0.3311 [0.3083, 0.3527] | +0.0111 | pass |
| pi05_l10_50 | S / 0.40 / 0 | 5340/2691/2129 | 0.4147 [0.3970, 0.4317] | +0.0147 | pass |
| pi05_l10_50 | P / 0.25 / 0 | 5888/2966/1120 | 0.2379 [0.2276, 0.2483] | -0.0121 | pass |
| pi05_l10_50 | P / 0.32 / 0 | 5676/2859/1556 | 0.3090 [0.3007, 0.3171] | -0.0110 | pass |
| pi05_l10_50 | P / 0.32 / 1 | 5641/2842/1561 | 0.3112 [0.3024, 0.3199] | -0.0088 | pass |
| pi05_l10_50 | P / 0.40 / 0 | 5637/2842/2083 | 0.3900 [0.3845, 0.3953] | -0.0100 | pass |
| pi05_l10_50 | P / 0.45 / 0 | 5536/2785/2410 | 0.4456 [0.4407, 0.4505] | -0.0044 | pass |
| pi05_l10_200 | D / 0.25 / 0 | 5469/2757/1132 | 0.2521 [0.2460, 0.2590] | +0.0021 | pass |
| pi05_l10_200 | D / 0.32 / 0 | 5663/2856/1632 | 0.3210 [0.3164, 0.3259] | +0.0010 | pass |
| pi05_l10_200 | D / 0.40 / 0 | 5792/2921/2217 | 0.4012 [0.3995, 0.4031] | +0.0012 | pass |
| pi05_l10_200 | D / 0.45 / 0 | 5478/2762/2418 | 0.4509 [0.4497, 0.4521] | +0.0009 | pass |
| pi05_l10_200 | P / 0.25 / 0 | 5747/2897/1166 | 0.2487 [0.2399, 0.2573] | -0.0013 | pass |
| pi05_l10_200 | P / 0.32 / 0 | 5552/2798/1573 | 0.3169 [0.3114, 0.3224] | -0.0031 | pass |
| pi05_l10_200 | P / 0.40 / 0 | 5448/2745/2042 | 0.3944 [0.3885, 0.4002] | -0.0056 | pass |
| pi05_l10_200 | P / 0.45 / 0 | 5399/2722/2382 | 0.4508 [0.4464, 0.4552] | +0.0008 | pass |
| pi05_spatial_50 | D / 0.25 / 0 | 2258/1153/448 | 0.2459 [0.2375, 0.2542] | -0.0041 | pass |
| pi05_spatial_50 | D / 0.32 / 0 | 2246/1148/626 | 0.3140 [0.3090, 0.3189] | -0.0060 | pass |
| pi05_spatial_50 | D / 0.32 / 1 | 2295/1171/641 | 0.3144 [0.3091, 0.3195] | -0.0056 | pass |
| pi05_spatial_50 | D / 0.40 / 0 | 2224/1137/840 | 0.3980 [0.3943, 0.4014] | -0.0020 | pass |
| pi05_spatial_50 | D / 0.45 / 0 | 2246/1144/997 | 0.4538 [0.4505, 0.4570] | +0.0038 | pass |
| pi05_spatial_50 | S / 0.32 / 0 | 2225/1137/828 | 0.3932 [0.3755, 0.4097] | +0.0732 | FAIL |
| pi05_spatial_50 | S / 0.40 / 0 | 2256/1150/997 | 0.4522 [0.4369, 0.4664] | +0.0522 | FAIL |
| pi05_spatial_50 | P / 0.25 / 0 | 2274/1161/454 | 0.2469 [0.2385, 0.2554] | -0.0031 | pass |
| pi05_spatial_50 | P / 0.32 / 0 | 2262/1156/652 | 0.3221 [0.3156, 0.3287] | +0.0021 | pass |
| pi05_spatial_50 | P / 0.32 / 1 | 2233/1145/644 | 0.3225 [0.3159, 0.3292] | +0.0025 | pass |
| pi05_spatial_50 | P / 0.40 / 0 | 2307/1177/892 | 0.4054 [0.3976, 0.4130] | +0.0054 | pass |
| pi05_spatial_50 | P / 0.45 / 0 | 2341/1196/1040 | 0.4544 [0.4477, 0.4609] | +0.0044 | pass |

D = distance_feedback; P = periodic; S = static-b diagnostic. Static failures remain calibration failures and do not select a replacement. Full per-arm counts, guard floors, short/long and guard-heavy operating checks, startup/later/terminal debt, raw-score saturation, hosts and hashes are in gates.json.

| Cell | Target / rep | D−P IR [95% paired CI] | Matched |
| --- | --- | --- | --- |
| groot_l10_50 | 0.25 / 0 | -0.0071 [-0.0191, +0.0050] | pass |
| groot_l10_50 | 0.32 / 0 | -0.0082 [-0.0163, -0.0002] | pass |
| groot_l10_50 | 0.32 / 1 | -0.0083 [-0.0162, -0.0001] | pass |
| groot_l10_50 | 0.40 / 0 | +0.0019 [-0.0043, +0.0082] | pass |
| groot_l10_50 | 0.45 / 0 | +0.0032 [-0.0021, +0.0085] | pass |
| groot_l10_200 | 0.25 / 0 | -0.0136 [-0.0220, -0.0060] | FAIL |
| groot_l10_200 | 0.32 / 0 | -0.0038 [-0.0112, +0.0035] | pass |
| groot_l10_200 | 0.40 / 0 | +0.0012 [-0.0045, +0.0067] | pass |
| groot_l10_200 | 0.45 / 0 | -0.0008 [-0.0056, +0.0039] | pass |
| groot_spatial_50 | 0.25 / 0 | -0.0010 [-0.0106, +0.0092] | pass |
| groot_spatial_50 | 0.32 / 0 | -0.0022 [-0.0097, +0.0058] | pass |
| groot_spatial_50 | 0.32 / 1 | +0.0034 [-0.0036, +0.0108] | pass |
| groot_spatial_50 | 0.40 / 0 | +0.0048 [-0.0038, +0.0136] | pass |
| groot_spatial_50 | 0.45 / 0 | +0.0020 [-0.0057, +0.0096] | pass |
| pi05_l10_50 | 0.25 / 0 | +0.0083 [-0.0034, +0.0196] | pass |
| pi05_l10_50 | 0.32 / 0 | +0.0125 [+0.0040, +0.0213] | FAIL |
| pi05_l10_50 | 0.32 / 1 | +0.0100 [+0.0010, +0.0186] | pass |
| pi05_l10_50 | 0.40 / 0 | +0.0095 [+0.0036, +0.0154] | pass |
| pi05_l10_50 | 0.45 / 0 | +0.0061 [+0.0008, +0.0113] | pass |
| pi05_l10_200 | 0.25 / 0 | +0.0035 [-0.0069, +0.0143] | pass |
| pi05_l10_200 | 0.32 / 0 | +0.0042 [-0.0034, +0.0118] | pass |
| pi05_l10_200 | 0.40 / 0 | +0.0068 [+0.0012, +0.0127] | pass |
| pi05_l10_200 | 0.45 / 0 | +0.0002 [-0.0045, +0.0049] | pass |
| pi05_spatial_50 | 0.25 / 0 | -0.0010 [-0.0089, +0.0065] | pass |
| pi05_spatial_50 | 0.32 / 0 | -0.0081 [-0.0146, -0.0017] | pass |
| pi05_spatial_50 | 0.32 / 1 | -0.0081 [-0.0155, -0.0004] | pass |
| pi05_spatial_50 | 0.40 / 0 | -0.0074 [-0.0156, +0.0006] | pass |
| pi05_spatial_50 | 0.45 / 0 | -0.0005 [-0.0076, +0.0067] | pass |

Engagement: denominators exclude guards and nonfinite fallback looks. Constant replay substitutes u=.5 and recomputes its own call ledger on the same actual look/guard sequence; change = symmetric difference / union of eligible selected positions. Shuffling is within episode (seed 20261004), descriptive only. These are fixed-state diagnostics, not counterfactual rollouts. “Near ceiling” is explicitly descriptive theta≥.95; high saturation is theta≥1.

| Cell | Target / rep | Eligible | Interior ≥.40 | Constant ≥.20 | Shuffle | theta <0 / ≥1 / ≥.95 | Nonfinite | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| groot_l10_50 | 0.25 / 0 | 2249 | 0.421 | 0.750 | 0.709 | 118/1184/1192 | 0 | pass |
| groot_l10_50 | 0.32 / 0 | 2141 | 0.630 | 0.639 | 0.506 | 444/348/375 | 0 | pass |
| groot_l10_50 | 0.32 / 1 | 2189 | 0.641 | 0.636 | 0.524 | 427/358/387 | 0 | pass |
| groot_l10_50 | 0.40 / 0 | 2193 | 0.597 | 0.503 | 0.398 | 708/175/188 | 0 | pass |
| groot_l10_50 | 0.45 / 0 | 2080 | 0.462 | 0.258 | 0.216 | 1098/21/25 | 0 | pass |
| groot_l10_200 | 0.25 / 0 | 2179 | 0.508 | 0.812 | 0.656 | 166/907/947 | 0 | pass |
| groot_l10_200 | 0.32 / 0 | 2169 | 0.621 | 0.724 | 0.532 | 503/320/347 | 0 | pass |
| groot_l10_200 | 0.40 / 0 | 2101 | 0.502 | 0.536 | 0.375 | 898/148/157 | 0 | pass |
| groot_l10_200 | 0.45 / 0 | 2208 | 0.470 | 0.234 | 0.238 | 1135/36/42 | 0 | pass |
| groot_spatial_50 | 0.25 / 0 | 1065 | 0.661 | 0.710 | 0.610 | 86/275/309 | 0 | pass |
| groot_spatial_50 | 0.32 / 0 | 1076 | 0.691 | 0.657 | 0.601 | 130/203/205 | 0 | pass |
| groot_spatial_50 | 0.32 / 1 | 1097 | 0.691 | 0.685 | 0.606 | 163/176/180 | 0 | pass |
| groot_spatial_50 | 0.40 / 0 | 1037 | 0.715 | 0.302 | 0.375 | 272/24/26 | 0 | pass |
| groot_spatial_50 | 0.45 / 0 | 1033 | 0.660 | 0.236 | 0.230 | 338/13/13 | 0 | pass |
| pi05_l10_50 | 0.25 / 0 | 2315 | 0.587 | 0.709 | 0.703 | 144/812/902 | 0 | pass |
| pi05_l10_50 | 0.32 / 0 | 2266 | 0.667 | 0.573 | 0.569 | 414/341/392 | 0 | pass |
| pi05_l10_50 | 0.32 / 1 | 2285 | 0.667 | 0.581 | 0.588 | 399/363/429 | 0 | pass |
| pi05_l10_50 | 0.40 / 0 | 2281 | 0.651 | 0.468 | 0.374 | 688/108/195 | 0 | pass |
| pi05_l10_50 | 0.45 / 0 | 2275 | 0.563 | 0.274 | 0.243 | 962/32/44 | 0 | pass |
| pi05_l10_200 | 0.25 / 0 | 2297 | 0.616 | 0.707 | 0.725 | 199/684/767 | 0 | pass |
| pi05_l10_200 | 0.32 / 0 | 2356 | 0.654 | 0.621 | 0.563 | 401/414/454 | 0 | pass |
| pi05_l10_200 | 0.40 / 0 | 2348 | 0.649 | 0.465 | 0.398 | 663/162/183 | 0 | pass |
| pi05_l10_200 | 0.45 / 0 | 2313 | 0.557 | 0.273 | 0.239 | 990/34/48 | 0 | pass |
| pi05_spatial_50 | 0.25 / 0 | 1025 | 0.666 | 0.664 | 0.606 | 65/277/321 | 0 | pass |
| pi05_spatial_50 | 0.32 / 0 | 1011 | 0.709 | 0.672 | 0.568 | 126/168/268 | 0 | pass |
| pi05_spatial_50 | 0.32 / 1 | 1042 | 0.685 | 0.656 | 0.571 | 207/121/221 | 0 | pass |
| pi05_spatial_50 | 0.40 / 0 | 1028 | 0.766 | 0.481 | 0.403 | 179/62/64 | 0 | pass |
| pi05_spatial_50 | 0.45 / 0 | 1009 | 0.671 | 0.245 | 0.204 | 322/10/18 | 0 | pass |

| Cell | Complete arms | Logged decisions | Mechanics violations | Guard / knob calls | Max no-call slots |
| --- | ---: | ---: | ---: | --- | ---: |
| groot_l10_50 | 12 | 73788 | 0 | 11128/12547 | 45 |
| groot_l10_200 | 8 | 45029 | 0 | 5432/9010 | 17 |
| groot_spatial_50 | 12 | 29167 | 0 | 2103/7530 | 27 |
| pi05_l10_50 | 12 | 68219 | 0 | 6993/14807 | 46 |
| pi05_l10_200 | 8 | 44548 | 0 | 3882/10680 | 19 |
| pi05_spatial_50 | 12 | 27167 | 0 | 1622/7437 | 25 |

Implementation evidence: functional pass=True. The old RELEASE digest is stale for 1 evidence file (mechanics.json); retained JUnit independently records 71 passing tests. Current runtime differs in 2 shared files only through default-off audit hooks: the audit reconstructs the SHA-bound released files from saved originals + shared_changes.diff and verifies AST equality with those disabled hooks removed. No R14 result is read. The full drift evidence is retained in gates.json.

Live checks cover accepted N/V/M prefixes, OR/reason arithmetic, one actual MISS charge, client/server/terminal reconciliation, tail transitions, identities, fit bindings, and 145390 look shard SHA/raw-distance checks. Existing CPU evidence supplies off equivalence, retry/reset/clone/order isolation and invalid-tail/early-end cases; live logs cannot directly count hidden internal guard invocations. All 20 size-500 fit hashes are checked against implementation records. Remote loaded code/checkpoint bytes cannot be freshly attested by this local audit; logged paths/hosts/seeds and fit/contract hashes are retained.

| Duplicate .32 cell | Paired discordance r0 / r1 | Within-task variance r0 / r1 | Covariance | rho [95% CI] | 4-rep cell MDE / infinite-rep floor (pp) |
| --- | --- | --- | ---: | --- | --- |
| groot_l10_50 | 0.190/0.220 | 0.1700/0.1844 | 0.0256 | 0.144 [-0.120, 0.397] | 3.16/2.00 |
| groot_spatial_50 | 0.150/0.160 | 0.1544/0.1667 | -0.0333 | -0.208 [-0.398, 0.010] | 2.51/0.00 |
| pi05_l10_50 | 0.160/0.150 | 0.1533/0.1656 | 0.0233 | 0.146 [-0.030, 0.318] | 3.00/1.91 |
| pi05_spatial_50 | 0.030/0.070 | 0.0322/0.0722 | -0.0111 | -0.230 [-0.552, 0.098] | 1.43/0.00 |

Power audit: planning_scenario_only. Two repeats estimate dependence imprecisely. Size-50/.32 does not identify covariance across unreplicated sizes or .40. Full-grid precision is not established. Process seeds are not independent per-episode streams; shadows use private noise but rep0 shadow overhead can alter concurrent request order. Missing model/suite cells are not reweighted away.

MDE values are scenarios at 500 inits/cell, alpha=.05 two-sided and 80% power. Raw task-conditional covariance is reported; the exchangeable floor is clipped to [0, mean variance] only for projections. Pilot mean SR differences are not used or ranked. Two .32 repeats cannot certify pooled .32/.40, all-size precision; no precision-based arm selection or extra independent-episode credit is taken.

| Frozen test-A batch root | Host | Verdict |
| --- | --- | --- |
| r13_test_groot_l10_rep0 | local_4090 | held |
| r13_test_groot_spatial_rep0 | local_4090 | released |
| r13_test_groot_l10_rep1 | local_4090 | held |
| r13_test_groot_spatial_rep1 | local_4090 | released |
| r13_test_groot_l10_rep2 | local_4090 | held |
| r13_test_groot_spatial_rep2 | local_4090 | released |
| r13_test_groot_l10_rep3 | local_4090 | held |
| r13_test_groot_spatial_rep3 | local_4090 | released |
| r13_test_pi05_l10_rep0 | h100 | held |
| r13_test_pi05_spatial_rep0 | h100 | released |
| r13_test_pi05_l10_rep1 | h100 | held |
| r13_test_pi05_spatial_rep1 | h100 | released |
| r13_test_pi05_l10_rep2 | h100 | held |
| r13_test_pi05_spatial_rep2 | h100 | released |
| r13_test_pi05_l10_rep3 | h100 | held |
| r13_test_pi05_spatial_rep3 | h100 | released |

Root verdict is the content-gate decision for the complete frozen batch, including controls. Scheduling is separate: the frozen queue places each L10 root before Spatial on its fleet. Released Spatial roots cannot bypass held L10 predecessors without changing that plan; blocked predecessors are enumerated in gates.json. Rep1–rep3 also wait for prior complete blocks on both fleets. No installation, rollout, remote command or GPU work was performed.

Reproduce (from repository root; only the three new deliverables are in astra):

```bash
taskset -c 10-21,54-65 env CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python exp/offline_search/rounds/r13/astra/evaluate_gates.py
```
