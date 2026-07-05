# Stage 2a — SR 增益分解 + H1/H2/H3 裁决

cache-run include_ws=False；口径见 plan §3.2。net/SR 单位 pp。

## spatial_fh75_ws10_A (libero_spatial / A)

**H1 run-length / H2 FH率 / H3 WS执行**

| cond | n_runs | mean | median | max | searched FH% | WS succ/fail |
|---|---|---|---|---|---|---|
| baseline | 770 | 12.14 | 11.0 | 29 | 69.9 | 1.00/3.51 |
| N1 | 749 | 12.47 | 11.0 | 29 | 80.8 | 0.92/3.43 |
| periodic(k7/n1) | 1635 | 5.32 | 7.0 | 7 | 79.6 | 0.84/2.56 |

**Δinf 分解 + 配对统计（vs baseline）**

| cond | d_inf(pool) | skip_conv | verdict_mix | ep_len_res | ΔSR pp | chi2 | exact_p | b/c |
|---|---|---|---|---|---|---|---|---|
| n1 | -0.004 | +0.091 | -0.095 | +0.001 | +3.0 | 3.32 | 0.0674 | 22/37 |
| periodic | -0.007 | +0.077 | -0.084 | +0.028 | +7.8 | 15.87 | 0.0001 | 26/65 |

N1 per-task 最低 ΔSR（排单任务异常）: t9 -4pp(n50), t3 +0pp(n50), t4 +0pp(n50)

## spatial_fh75_ws10_B (libero_spatial / B)

**H1 run-length / H2 FH率 / H3 WS执行**

| cond | n_runs | mean | median | max | searched FH% | WS succ/fail |
|---|---|---|---|---|---|---|
| baseline | 770 | 12.14 | 11.0 | 29 | 69.9 | 1.00/3.51 |
| N1 | 733 | 12.41 | 11.0 | 29 | 87.2 | 0.78/3.05 |
| periodic(k4/n1) | 2280 | 3.46 | 4.0 | 4 | 76.2 | 0.70/2.05 |

**Δinf 分解 + 配对统计（vs baseline）**

| cond | d_inf(pool) | skip_conv | verdict_mix | ep_len_res | ΔSR pp | chi2 | exact_p | b/c |
|---|---|---|---|---|---|---|---|---|
| n1 | +0.006 | +0.143 | -0.137 | +0.008 | +5.2 | 10.08 | 0.0013 | 18/44 |
| periodic | +0.081 | +0.130 | -0.049 | +0.021 | +6.4 | 9.61 | 0.0018 | 34/66 |

N1 per-task 最低 ΔSR（排单任务异常）: t9 -6pp(n50), t7 +0pp(n50), t3 +2pp(n50)

## l10_fh5_ws40_A (libero_10 / A)

**H1 run-length / H2 FH率 / H3 WS执行**

| cond | n_runs | mean | median | max | searched FH% | WS succ/fail |
|---|---|---|---|---|---|---|
| baseline | 884 | 10.26 | 9.0 | 57 | 27.8 | 22.02/23.65 |
| N1 | 885 | 10.18 | 9.0 | 57 | 35.1 | 21.30/21.84 |
| periodic(k4/n1) | 2132 | 3.18 | 4.0 | 4 | 27.3 | 18.86/22.10 |

**Δinf 分解 + 配对统计（vs baseline）**

| cond | d_inf(pool) | skip_conv | verdict_mix | ep_len_res | ΔSR pp | chi2 | exact_p | b/c |
|---|---|---|---|---|---|---|---|---|
| n1 | +0.006 | +0.077 | -0.072 | -0.004 | -0.6 | 0.05 | 0.8199 | 40/37 |
| periodic | +0.064 | +0.070 | -0.006 | +0.009 | +4.6 | 4.44 | 0.0346 | 43/66 |

N1 per-task 最低 ΔSR（排单任务异常）: t2 -6pp(n50), t9 -6pp(n50), t7 -4pp(n50)

## l10_fh5_ws40_B (libero_10 / B)

**H1 run-length / H2 FH率 / H3 WS执行**

| cond | n_runs | mean | median | max | searched FH% | WS succ/fail |
|---|---|---|---|---|---|---|
| baseline | 884 | 10.26 | 9.0 | 57 | 27.8 | 22.02/23.65 |
| N1 | 868 | 10.22 | 9.0 | 57 | 39.4 | 20.40/22.95 |
| periodic(k2/n1) | 2873 | 1.82 | 2.0 | 2 | 25.4 | 16.51/19.58 |

**Δinf 分解 + 配对统计（vs baseline）**

| cond | d_inf(pool) | skip_conv | verdict_mix | ep_len_res | ΔSR pp | chi2 | exact_p | b/c |
|---|---|---|---|---|---|---|---|---|
| n1 | +0.018 | +0.118 | -0.099 | -0.002 | -1.4 | 0.49 | 0.4828 | 40/33 |
| periodic | +0.123 | +0.119 | +0.004 | +0.018 | +4.8 | 5.29 | 0.0210 | 38/62 |

N1 per-task 最低 ΔSR（排单任务异常）: t4 -10pp(n50), t6 -6pp(n50), t2 -4pp(n50)
