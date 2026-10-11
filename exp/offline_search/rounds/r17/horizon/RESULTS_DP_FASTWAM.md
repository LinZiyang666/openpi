# DP / Fast-WAM：R17 horizon / 纯减步预注册分析

快照 UTC：2026-10-11T00:06:01.139090+00:00。仅分析有明确 PASS 收据且通过本脚本完整性审计的臂。
**当前已验收11臂、pending 0臂；两suite同时通过24项校正SR门：无；未校正下同时通过：Fast-WAM/C10。** 服务成本门尚未认证；cache的最终L与fallback理由见末表。
原始结果、STUDY和运行队列只读；未调用GPU、远程工具或campaign代码。

## 方法与入组

复跑：`.venv/bin/python exp/offline_search/rounds/r17/horizon/analyze_arms.py`（在仓库根执行）；仅重写本目录分析Markdown/JSON。`--self-test`运行CPU合成审计。

配对键为(suite, canonical task, official init index)，再逐局核验init字节hash、任务ID、seed7、同benchmark profile；每suite10任务×50初态。control先审计全部2000局，再只分析spatial/10。按最后accepted terminal及对应attempt取数据；decision/execution_ack/client_timing分别计数，重复uid/step取最后记录；不把retry/error、尾请求或ACK当作新模型调用。

保持STUDY的24比较family（含Cosmos，未完成时也不缩小比较族），a=.05/(2×24)：`CP_low(g,500,a)-CP_high(l,500,a)`；这是保守的配对gains/losses CP差界，不是声称另一个精确风险差检验。bootstrap为task内成对重采样10000次，PCG64 seed20261010，canonical task/init排序、所有共享控制臂使用相同索引，取.05/24下分位。两种下界都严格大于−2pp且两个suite都过才通过SR门。未校正单侧95%界仅作敏感性，不替代主判定；未通过表示证据不足，不等于已证明损失超过2pp。

原horizon首次失败即停的采用规则与cache研究“最长通过C，否则10”分别报告；不修改STUDY、不据未校正结果选C。step成本门用实际NFE/控制步，horizon用模型调用/控制步，二者还都需要可比服务时间/控制步下降。

来源：[预注册](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/STUDY.md:1)；[同方法Cosmos报告](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/COSMOS_K2.md:1)；[cache选择规则](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/cache/CACHE_STUDY.md:1)。

| 控制 | 验收/规模 | 实际配置 |
|---|---|---|
| DP [hs5dp_ctl_142812](/data/mw_finetune/eval/hs5dp_ctl_142812/RECEIPT.json:1) | PASS；2000审计、1000主集；recovered=True | local, R2/B8/5.0ms/MPS；timan107×32；group-commit |
| Fast-WAM [hs6fw_res_152931](/data/mw_finetune/eval/hs6fw_res_152931/RECEIPT.json:1) | PASS；2000审计、1000主集；recovered=True | h100, R2/B16/5.0ms/MPS；timan107×48；group-commit |

Fast控制使用merged resume完整2000局，不把首段与续段重复累计。当前控制/入组臂逐项匹配GPU节点、replicas、batch设置、MPS、worker资源、EGL、benchmark、原生content_digests、模型/归一化/feature/layout与resident执行模式。与STUDY首版的旧replica/worker配置有偏离，本次按owner指定的hs5/hs6匹配控制，不追改旧登记；相同batch cap不保证各次实际batch组成、外部负载完全相同。

## 已验收的配对结果

SR单位%；Δ及下界单位pp；未校正LB列依次为CP / bootstrap。表内判决只针对该suite的SR门。

| 模型 | 臂 | suite | n | control SR | arm SR | Δ | CP LB family | bootstrap LB family | 未校正LB CP / boot | 判决 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| DP | C10 | libero_spatial | 500 | 91.20 | 89.80 | -1.40 | -7.87 | -5.60 | -5.59 / -3.80 | 证据不足 |
| DP | C10 | libero_10 | 500 | 79.60 | 83.80 | +4.20 | -5.10 | -2.00 | -1.79 / 0.60 | 证据不足 |
| DP | C15 | libero_spatial | 500 | 91.20 | 86.40 | -4.80 | -11.94 | -9.40 | -9.43 / -7.40 | 证据不足 |
| DP | C15 | libero_10 | 500 | 79.60 | 84.00 | +4.40 | -4.93 | -1.80 | -1.61 / 0.80 | 证据不足 |
| DP | C16 | libero_spatial | 500 | 91.20 | 85.60 | -5.60 | -13.31 | -10.63 | -10.60 / -8.60 | 证据不足 |
| DP | C16 | libero_10 | 500 | 79.60 | 81.00 | +1.40 | -8.20 | -5.00 | -4.79 / -2.40 | 证据不足 |
| DP | DDIM10 | libero_spatial | 500 | 91.20 | 90.00 | -1.20 | -7.62 | -5.40 | -5.36 / -3.60 | 证据不足 |
| DP | DDIM10 | libero_10 | 500 | 79.60 | 76.40 | -3.20 | -12.63 | -9.60 | -9.29 / -7.00 | 证据不足 |
| DP | DDIM5 | libero_spatial | 500 | 91.20 | 90.60 | -0.60 | -6.87 | -4.63 | -4.66 / -3.00 | 证据不足 |
| DP | DDIM5 | libero_10 | 500 | 79.60 | 75.40 | -4.20 | -13.45 | -10.40 | -10.18 / -7.80 | 证据不足 |
| Fast-WAM | C10 | libero_spatial | 500 | 90.80 | 91.80 | +1.00 | -3.23 | -1.60 | -1.74 / -0.60 | 证据不足 |
| Fast-WAM | C10 | libero_10 | 500 | 91.60 | 94.20 | +2.60 | -2.81 | -0.60 | -0.89 / 0.60 | 证据不足 |
| Fast-WAM | C16 | libero_spatial | 500 | 90.80 | 92.60 | +1.80 | -2.57 | -0.80 | -1.02 / 0.20 | 证据不足 |
| Fast-WAM | C16 | libero_10 | 500 | 91.60 | 90.80 | -0.80 | -7.64 | -5.20 | -5.23 / -3.40 | 证据不足 |
| Fast-WAM | C20 | libero_spatial | 500 | 90.80 | 92.00 | +1.20 | -3.11 | -1.40 | -1.59 / -0.40 | 证据不足 |
| Fast-WAM | C20 | libero_10 | 500 | 91.60 | 91.60 | +0.00 | -6.86 | -4.40 | -4.43 / -2.60 | 证据不足 |
| Fast-WAM | C32 | libero_spatial | 500 | 90.80 | 90.00 | -0.80 | -6.42 | -4.40 | -4.44 / -2.80 | 证据不足 |
| Fast-WAM | C32 | libero_10 | 500 | 91.60 | 85.40 | -6.20 | -14.00 | -11.80 | -11.26 / -9.40 | 证据不足 |
| Fast-WAM | K2 | libero_spatial | 500 | 90.80 | 91.80 | +1.00 | -3.05 | -1.40 | -1.62 / -0.40 | 证据不足 |
| Fast-WAM | K2 | libero_10 | 500 | 91.60 | 89.80 | -1.80 | -7.80 | -5.63 | -5.69 / -4.00 | 证据不足 |
| Fast-WAM | K1 | libero_spatial | 500 | 90.80 | 90.40 | -0.40 | -5.20 | -3.40 | -3.51 / -2.20 | 证据不足 |
| Fast-WAM | K1 | libero_10 | 500 | 91.60 | 88.60 | -3.00 | -9.71 | -7.40 | -7.34 / -5.60 | 证据不足 |

| 模型/臂 | suite | gains / losses | McNemar双侧p（仅描述） | 未校正2pp门 |
|---|---|---:|---:|---|
| DP/C10 | libero_spatial | 24 / 31 | 0.418754 | 未通过 |
| DP/C10 | libero_10 | 73 / 52 | 0.0732143 | 通过 |
| DP/C15 | libero_spatial | 24 / 48 | 0.00630983 | 未通过 |
| DP/C15 | libero_10 | 74 / 52 | 0.0609383 | 通过 |
| DP/C16 | libero_spatial | 29 / 57 | 0.00335327 | 未通过 |
| DP/C16 | libero_10 | 71 / 64 | 0.605742 | 未通过 |
| DP/DDIM10 | libero_spatial | 24 / 30 | 0.496617 | 未通过 |
| DP/DDIM10 | libero_10 | 58 / 74 | 0.191499 | 未通过 |
| DP/DDIM5 | libero_spatial | 24 / 27 | 0.779768 | 未通过 |
| DP/DDIM5 | libero_10 | 53 / 74 | 0.0755306 | 未通过 |
| Fast-WAM/C10 | libero_spatial | 13 / 8 | 0.38331 | 通过 |
| Fast-WAM/C10 | libero_10 | 25 / 12 | 0.047031 | 通过 |
| Fast-WAM/C16 | libero_spatial | 16 / 7 | 0.0931396 | 通过 |
| Fast-WAM/C16 | libero_10 | 29 / 33 | 0.703537 | 未通过 |
| Fast-WAM/C20 | libero_spatial | 14 / 8 | 0.286279 | 通过 |
| Fast-WAM/C20 | libero_10 | 31 / 31 | 1 | 未通过 |
| Fast-WAM/C32 | libero_spatial | 18 / 22 | 0.635828 | 未通过 |
| Fast-WAM/C32 | libero_10 | 29 / 60 | 0.00133526 | 未通过 |
| Fast-WAM/K2 | libero_spatial | 12 / 7 | 0.359283 | 通过 |
| Fast-WAM/K2 | libero_10 | 19 / 28 | 0.24296 | 未通过 |
| Fast-WAM/K1 | libero_spatial | 13 / 15 | 0.850554 | 未通过 |
| Fast-WAM/K1 | libero_10 | 23 / 38 | 0.0721774 | 未通过 |

**500对的精度限制：** 对接近持平、存在不一致结局的臂，24重校正让通过2pp很困难，但不是数学上不可能。以下固定n=500示例说明，不能把所有未通过都归于SR下降。

| 假设g / l | family CP LB，pp | 未校正CP LB，pp |
|---|---:|---:|
| 0 / 0 | -1.36 | -0.74 |
| 0 / 1 | -1.82 | -1.11 |
| 0 / 2 | -2.22 | -1.44 |
| 5 / 5 | -3.09 | -1.99 |
| 10 / 10 | -4.15 | -2.68 |

bootstrap的.05/24分位约落在10000次中的第21个尾部样本，具有Monte Carlo离散误差；保持已登记draw数和seed，不为让某臂通过而增加样本或换随机种子。尤其不能将点估计已下降超过2pp的suite也解释成只有检验保守；点差与区间应一起判断。

## 成本：调用、NFE与观测时间分开

M=真正policy调用，Q=含policy_tail的全部请求，T=ACK确认的实际控制步（不含10步settle）。NFE/decision使用Q作分母，另给NFE/policy-call，避免把零NFE尾请求说成缩短了sampler。host总时包括尾请求；policy host/call只平均真正前向调用。

| 模型/臂 | suite | M/局 control→arm | NFE/policy-call | NFE/decision control→arm | M/T减少 | NFE/T减少 | host ms/T control→arm | host观测减少 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DP/C10 | libero_spatial | 23.012→11.758 | 100→100 | 100.000→51.060 | 48.93% | 48.93% | 123.170→45.613 | 62.97% |
| DP/C10 | libero_10 | 63.344→29.760 | 100→100 | 100.000→50.342 | 49.63% | 49.63% | 138.133→55.819 | 59.59% |
| DP/C15 | libero_spatial | 23.012→8.226 | 100→100 | 100.000→34.809 | 65.26% | 65.26% | 123.170→23.544 | 80.88% |
| DP/C15 | libero_10 | 63.344→19.536 | 100→100 | 100.000→33.872 | 66.12% | 66.12% | 138.133→28.604 | 79.29% |
| DP/C16 | libero_spatial | 23.012→7.742 | 100→100 | 100.000→26.476 | 67.48% | 67.48% | 123.170→22.173 | 82.00% |
| DP/C16 | libero_10 | 63.344→18.928 | 100→100 | 100.000→25.622 | 68.14% | 68.14% | 138.133→26.818 | 80.59% |
| DP/DDIM10 | libero_spatial | 23.012→23.356 | 100→10 | 100.000→10.000 | 0.01% | 90.00% | 123.170→32.125 | 73.92% |
| DP/DDIM10 | libero_10 | 63.344→65.102 | 100→10 | 100.000→10.000 | 0.06% | 90.01% | 138.133→41.378 | 70.04% |
| DP/DDIM5 | libero_spatial | 23.012→23.212 | 100→5 | 100.000→5.000 | 0.05% | 95.00% | 123.170→22.608 | 81.64% |
| DP/DDIM5 | libero_10 | 63.344→65.770 | 100→5 | 100.000→5.000 | 0.03% | 95.00% | 138.133→40.455 | 70.71% |
| Fast-WAM/C10 | libero_spatial | 23.770→11.948 | 10→10 | 10.000→5.096 | 49.01% | 49.01% | 363.699→160.151 | 55.97% |
| Fast-WAM/C10 | libero_10 | 56.418→27.450 | 10→10 | 10.000→5.045 | 49.54% | 49.54% | 369.465→161.956 | 56.16% |
| Fast-WAM/C16 | libero_spatial | 23.770→7.526 | 10→10 | 10.000→2.644 | 67.50% | 67.50% | 363.699→93.680 | 74.24% |
| Fast-WAM/C16 | libero_10 | 56.418→17.598 | 10→10 | 10.000→2.568 | 68.11% | 68.11% | 369.465→98.540 | 73.33% |
| Fast-WAM/C20 | libero_spatial | 23.770→6.176 | 10→10 | 10.000→2.646 | 73.50% | 73.50% | 363.699→71.429 | 80.36% |
| Fast-WAM/C20 | libero_10 | 56.418→14.068 | 10→10 | 10.000→2.565 | 74.35% | 74.35% | 369.465→74.920 | 79.72% |
| Fast-WAM/C32 | libero_spatial | 23.770→4.098 | 10→10 | 10.000→1.586 | 82.85% | 82.85% | 363.699→43.190 | 88.12% |
| Fast-WAM/C32 | libero_10 | 56.418→9.464 | 10→10 | 10.000→1.504 | 83.62% | 83.62% | 369.465→45.105 | 87.79% |
| Fast-WAM/K2 | libero_spatial | 23.770→23.614 | 10→2 | 10.000→2.000 | -0.07% | 79.99% | 363.699→204.462 | 43.78% |
| Fast-WAM/K2 | libero_10 | 56.418→57.526 | 10→2 | 10.000→2.000 | 0.06% | 80.01% | 369.465→230.837 | 37.52% |
| Fast-WAM/K1 | libero_spatial | 23.770→23.590 | 10→1 | 10.000→1.000 | -0.03% | 90.00% | 363.699→184.291 | 49.33% |
| Fast-WAM/K1 | libero_10 | 56.418→57.520 | 10→1 | 10.000→1.000 | 0.02% | 90.00% | 369.465→189.533 | 48.70% |

| 模型/臂 | suite | Q/局 control→arm | T/局 control→arm | policy host ms/call control→arm | service/queue/pricing有效行 control→arm（各） |
|---|---|---:|---:|---:|---|
| DP/C10 | libero_spatial | 23.012→23.028 | 113.198→113.254 | 605.88→434.85 | 0/0/0 of 11506 → 0/0/0 of 11514 |
| DP/C10 | libero_10 | 63.344→59.116 | 315.114→293.908 | 687.16→546.31 | 0/0/0 of 31672 → 0/0/0 of 29558 |
| DP/C15 | libero_spatial | 23.012→23.632 | 113.198→116.482 | 605.88→326.17 | 0/0/0 of 11506 → 0/0/0 of 11816 |
| DP/C15 | libero_10 | 63.344→57.676 | 315.114→286.812 | 687.16→411.65 | 0/0/0 of 31672 → 0/0/0 of 28838 |
| DP/C16 | libero_spatial | 23.012→29.242 | 113.198→117.096 | 605.88→324.57 | 0/0/0 of 11506 → 0/0/0 of 14621 |
| DP/C16 | libero_10 | 63.344→73.874 | 315.114→295.550 | 687.16→405.60 | 0/0/0 of 31672 → 0/0/0 of 36937 |
| DP/DDIM10 | libero_spatial | 23.012→23.356 | 113.198→114.896 | 605.88→158.03 | 0/0/0 of 11506 → 0/0/0 of 11678 |
| DP/DDIM10 | libero_10 | 63.344→65.102 | 315.114→324.046 | 687.16→205.96 | 0/0/0 of 31672 → 0/0/0 of 32551 |
| DP/DDIM5 | libero_spatial | 23.012→23.212 | 113.198→114.238 | 605.88→111.27 | 0/0/0 of 11506 → 0/0/0 of 11606 |
| DP/DDIM5 | libero_10 | 63.344→65.770 | 315.114→327.268 | 687.16→201.30 | 0/0/0 of 31672 → 0/0/0 of 32885 |
| Fast-WAM/C10 | libero_spatial | 23.770→23.448 | 117.048→115.394 | 1790.92→1543.33 | 0/0/0 of 11885 → 0/0/0 of 11724 |
| Fast-WAM/C10 | libero_10 | 56.418→54.414 | 280.196→270.154 | 1834.92→1590.56 | 0/0/0 of 28209 → 0/0/0 of 27207 |
| Fast-WAM/C16 | libero_spatial | 23.770→28.468 | 117.048→114.014 | 1790.92→1409.10 | 0/0/0 of 11885 → 0/0/0 of 14234 |
| Fast-WAM/C16 | libero_10 | 56.418→68.526 | 280.196→274.042 | 1834.92→1523.56 | 0/0/0 of 28209 → 0/0/0 of 34263 |
| Fast-WAM/C20 | libero_spatial | 23.770→23.340 | 117.048→114.748 | 1790.92→1318.57 | 0/0/0 of 11885 → 0/0/0 of 11670 |
| Fast-WAM/C20 | libero_10 | 56.418→54.856 | 280.196→272.370 | 1834.92→1441.33 | 0/0/0 of 28209 → 0/0/0 of 27428 |
| Fast-WAM/C32 | libero_spatial | 23.770→25.834 | 117.048→117.648 | 1790.92→1224.47 | 0/0/0 of 11885 → 0/0/0 of 12917 |
| Fast-WAM/C32 | libero_10 | 56.418→62.930 | 280.196→287.024 | 1834.92→1349.96 | 0/0/0 of 28209 → 0/0/0 of 31465 |
| Fast-WAM/K2 | libero_spatial | 23.770→23.614 | 117.048→116.204 | 1790.92→1006.15 | 0/0/0 of 11885 → 0/0/0 of 11807 |
| Fast-WAM/K2 | libero_10 | 56.418→57.526 | 280.196→285.868 | 1834.92→1147.11 | 0/0/0 of 28209 → 0/0/0 of 28763 |
| Fast-WAM/K1 | libero_spatial | 23.770→23.590 | 117.048→116.132 | 1790.92→907.25 | 0/0/0 of 11885 → 0/0/0 of 11795 |
| Fast-WAM/K1 | libero_10 | 56.418→57.520 | 280.196→285.720 | 1834.92→941.47 | 0/0/0 of 28209 → 0/0/0 of 28760 |

**服务成本门尚未认证。** 当前计时不能以queue-free service证据证明相同负载或无外部竞争；host wall差包含排队、batch组成、CPU/GPU竞争。上表实测host降幅是描述，不是净服务时间或CUDA kernel节省；只有调用/NFE减少可据逐行receipt确认。JSON另存共同成功pairs、每局时间与全量成本。控制driver耗时覆盖四suite且Fast含resume，候选仅两suite；不能直接相除宣称端到端加速。

DP减步应称DDPM100 vs 同ε模型trailing-DDIM10/5（eta0）；不是截短DDPM。Fast K2/K1仍为原生shifted-flow scheduler；脚本逐行核对K、sampler identity和实际NFE。

## 每臂结论、最终长段与验收状态

| 模型/臂 | 两suite SR门 | 成本/总体采用结论 | 数据证据 |
|---|---|---|---|
| DP/C10 | 未通过 | 未通过：SR证据不足；服务成本未定；未标可采用 | [收据](/data/mw_finetune/eval/hs5dp_c10/RECEIPT.json:1)、[SUMMARY](/data/mw_finetune/eval/hs5dp_c10/SUMMARY.json:1)、[plan](/data/mw_finetune/eval/hs5dp_c10/EPISODE_PLAN.json:1)、[step ledger](/data/mw_finetune/eval/hs5dp_c10/driver/per_step.jsonl:1) |
| DP/C15 | 未通过 | 未通过：SR证据不足；服务成本未定；未标可采用 | [收据](/data/mw_finetune/eval/hs5dp_c15s0/RECEIPT.json:1)、[SUMMARY](/data/mw_finetune/eval/hs5dp_c15s0/SUMMARY.json:1)、[plan](/data/mw_finetune/eval/hs5dp_c15s0/EPISODE_PLAN.json:1)、[step ledger](/data/mw_finetune/eval/hs5dp_c15s0/driver/per_step.jsonl:1) |
| DP/C16 | 未通过 | 未通过：SR证据不足；服务成本未定；未标可采用 | [收据](/data/mw_finetune/eval/hs5dp_c16s0/RECEIPT.json:1)、[SUMMARY](/data/mw_finetune/eval/hs5dp_c16s0/SUMMARY.json:1)、[plan](/data/mw_finetune/eval/hs5dp_c16s0/EPISODE_PLAN.json:1)、[step ledger](/data/mw_finetune/eval/hs5dp_c16s0/driver/per_step.jsonl:1) |
| DP/DDIM10 | 未通过 | 未通过：SR证据不足；服务成本未定；未标可采用 | [收据](/data/mw_finetune/eval/hs5dp_ddim10s0/RECEIPT.json:1)、[SUMMARY](/data/mw_finetune/eval/hs5dp_ddim10s0/SUMMARY.json:1)、[plan](/data/mw_finetune/eval/hs5dp_ddim10s0/EPISODE_PLAN.json:1)、[step ledger](/data/mw_finetune/eval/hs5dp_ddim10s0/driver/per_step.jsonl:1) |
| DP/DDIM5 | 未通过 | 未通过：SR证据不足；服务成本未定；未标可采用 | [收据](/data/mw_finetune/eval/hs5dp_ddim5s0/RECEIPT.json:1)、[SUMMARY](/data/mw_finetune/eval/hs5dp_ddim5s0/SUMMARY.json:1)、[plan](/data/mw_finetune/eval/hs5dp_ddim5s0/EPISODE_PLAN.json:1)、[step ledger](/data/mw_finetune/eval/hs5dp_ddim5s0/driver/per_step.jsonl:1) |
| Fast-WAM/C10 | 未通过 | 未通过：SR证据不足；服务成本未定；未标可采用 | [收据](/data/mw_finetune/eval/hs6fw_c10/RECEIPT.json:1)、[SUMMARY](/data/mw_finetune/eval/hs6fw_c10/SUMMARY.json:1)、[plan](/data/mw_finetune/eval/hs6fw_c10/EPISODE_PLAN.json:1)、[step ledger](/data/mw_finetune/eval/hs6fw_c10/driver/per_step.jsonl:1) |
| Fast-WAM/C16 | 未通过 | 未通过：SR证据不足；服务成本未定；未标可采用 | [收据](/data/mw_finetune/eval/hs6fw_c16/RECEIPT.json:1)、[SUMMARY](/data/mw_finetune/eval/hs6fw_c16/SUMMARY.json:1)、[plan](/data/mw_finetune/eval/hs6fw_c16/EPISODE_PLAN.json:1)、[step ledger](/data/mw_finetune/eval/hs6fw_c16/driver/per_step.jsonl:1) |
| Fast-WAM/C20 | 未通过 | 未通过：SR证据不足；服务成本未定；未标可采用 | [收据](/data/mw_finetune/eval/hs6fw_c20/RECEIPT.json:1)、[SUMMARY](/data/mw_finetune/eval/hs6fw_c20/SUMMARY.json:1)、[plan](/data/mw_finetune/eval/hs6fw_c20/EPISODE_PLAN.json:1)、[step ledger](/data/mw_finetune/eval/hs6fw_c20/driver/per_step.jsonl:1) |
| Fast-WAM/C32 | 未通过 | 未通过：SR证据不足；服务成本未定；未标可采用 | [收据](/data/mw_finetune/eval/hs6fw_c32s1/RECEIPT.json:1)、[SUMMARY](/data/mw_finetune/eval/hs6fw_c32s1/SUMMARY.json:1)、[plan](/data/mw_finetune/eval/hs6fw_c32s1/EPISODE_PLAN.json:1)、[step ledger](/data/mw_finetune/eval/hs6fw_c32s1/driver/per_step.jsonl:1) |
| Fast-WAM/K2 | 未通过 | 未通过：SR证据不足；服务成本未定；未标可采用 | [收据](/data/mw_finetune/eval/hs6fw_k2s1/RECEIPT.json:1)、[SUMMARY](/data/mw_finetune/eval/hs6fw_k2s1/SUMMARY.json:1)、[plan](/data/mw_finetune/eval/hs6fw_k2s1/EPISODE_PLAN.json:1)、[step ledger](/data/mw_finetune/eval/hs6fw_k2s1/driver/per_step.jsonl:1) |
| Fast-WAM/K1 | 未通过 | 未通过：SR证据不足；服务成本未定；未标可采用 | [收据](/data/mw_finetune/eval/hs6fw_k1/RECEIPT.json:1)、[SUMMARY](/data/mw_finetune/eval/hs6fw_k1/SUMMARY.json:1)、[plan](/data/mw_finetune/eval/hs6fw_k1/EPISODE_PLAN.json:1)、[step ledger](/data/mw_finetune/eval/hs6fw_k1/driver/per_step.jsonl:1) |

| 模型 | cache研究最终L | 来源 | 原horizon采用顺序 |
|---|---:|---|---|
| DP | 10 | 无已认证长C；按cache规则fallback10 | C10 SR未通过，保留C5 |
| Fast-WAM | 10 | 无已认证长C；按cache规则fallback10 | C10 SR未通过，保留C5 |

最终L只使用已验收臂的24项校正SR门；未校正界、partial结果和减步成绩都不参与。fallback10是owner规定的历史选择，**不是C10已获非劣认证**。全部预注册DP/Fast臂已验收，无pending。

1000 accepted或SUMMARY PASS不能替代PASS/明确恢复验收；未验收臂的SR和成本不进入本报告。s<n>按已封存重准备序号选择，绝不按SR挑retry；旧r<n>调试目录不在owner指定arm族内。

完整逐task计数、g/l初态、1000对身份、0/5pp敏感性、共同成功成本和输入SHA256见 [分析JSON](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/horizon/RESULTS_DP_FASTWAM.json:1)。本次复核164个输入文件在分析期间SHA未改变。
