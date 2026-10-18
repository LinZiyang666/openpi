# DP 双卡 TF32-off：两个冷启动/卡（2026-10-18）

**结论：TF32-off 明显降低动作误差，但没有恢复包含 cache 的 DP 跨 GPU 等价。** 4090↔H100 仍有 **51/2,304** 个逻辑请求超过预注册硬预算，全部来自 cache；**131/2,304** 超过逐请求并发 floor 门槛。4090 同卡 cache 本身也有19个请求超硬预算。缓存离散决策全部一致，不能据此忽略连续动作差。此次结果取代[仅 H100 第一冷启动的暂定判断](DP_TF32OFF_1018.md:1)，并不改变既定大显存排期。

证据：[双卡摘要](evidence/dp_both_tf32off_summary_20261018.json:1)、[16个 resident 的有效 flags 与失败样例审计](evidence/dp_both_tf32off_audit_20261018.json:1)、[native 基线](DP_4090_H100.md:1)。本次仅 CPU 分析，未启动 GPU 模型、未改正式 lane，sudo 命令为零。

## 1. 配对总体与身份

| GPU | 冷启动1 | 冷启动2 | 本地执行日志 |
|---|---|---|---|
| 4090 | `4090_dp_concurrent_longseg_recal1018_20261018T122706Z` | `4090_dp_concurrent_longseg_recal1018_20261018T123059Z` | [1](evidence/run_concurrent_4090_dp_tf32off_1.log:1)、[2](evidence/run_concurrent_4090_dp_tf32off_2.log:1) |
| H100 | `h100_dp_concurrent_longseg_recal1018_20261018T115441Z` | `h100_dp_concurrent_longseg_recal1018_20261018T122426Z` | [1](evidence/run_concurrent_h100_dp_tf32off_1.log:1)、[2](evidence/run_concurrent_h100_dp_tf32off_2.log:1) |

每次运行均 COMPLETE，R4/B8、48客户端、2轮、pure/cache 两组、每客户端每组每轮24次 delivery：4,608 delivery/冷启动，四次合计18,432。每卡合并4轮；每个逻辑请求计算卡内全部6对、跨卡全部16对。总体2,304个逻辑请求，pure/cache 各1,152；每轮cache含576个head、576个tail。输入为重复使用的三帧真实观察，不是2,304个独立样本或闭环episode。

以下 identity 四次完全匹配；原始请求及seed逐请求一致，cache依赖文件前后SHA校验全部通过：

| 身份 | SHA256 |
|---|---|
| serving/cache PIN | `59b1b483a694396cc035146a3c227cb14dd6bf37d87490fbdea2f34f282ecd39` |
| frozen runtime | `5927a753312bf5981cba81a2e389edea914bde0580523b0dc3c8ea8d58f1fa1e` |
| bootstrap / decision-step fix | `4e07fb3ecbfbc15000886cdaeb6cc16da2fbd346e045df22ad8ac77a1e5e4f5e` |
| 执行 harness | `8498998dbff134eea77988a1868eced5bab2ebdf786d6e910aedea98a2ca1096` |
| 录制输入 | `5361e28384e29ffc28b9e6417e2530538f6a38836c1584eaf43c3b3eed491997` |
| cache 文件清单 | `39fda40580296ea1aca75121e568540badd248b90b4bb19c8d6bc558abac0123` |

16份最终 `NUMERICS.json` 均确认 `matmul_tf32=False`、`cudnn_tf32=False`、`matmul_precision=highest`。相对各卡native，报告的torch flags唯一变化为 **cuDNN TF32 True→False**；package版本完全相同。cuDNN deterministic/benchmark及deterministic algorithms仍False，CUBLAS_WORKSPACE_CONFIG仍未设置，三个SDPA backend仍启用。flags不构成实际kernel dispatch证书。

MPS、真实socket/client、动态batch、原生CUDA graph及longseg ACK时钟均保留。16个resident原生graph自检全部PASS；每个resident实际访问B8，次数依运行依次为4090 `[60,57,54,66]`、`[60,55,52,51]`，H100 `[52,51,50,52]`、`[50,50,52,48]`；所有运行均访问B1…B8。18,432条delivery的decision/env-step时钟核对无不一致。这里测试的是正式服务实现加明确TF32消融的并发路径，不是B1 replay，也不表示在线正式lane已改用TF32-off。

## 2. 动作误差与固定预算

沿用预注册标准：wire action **max-abs≤0.000345707 且 relative-L2≤0.001**；另外逐请求两项误差均须≤`max(1e-6, 5×该请求两侧同卡floor较大值)`。不以全局最大floor替代逐请求floor，不因消融改善而改阈值。下表每格取所有请求、所有重复对中的最大值。

| 分组 | 4090同卡 max-abs | H100同卡 max-abs | 跨卡 max-abs | 跨卡最大 relative-L2 | 跨卡超硬预算 | 跨卡超floor |
|---|---:|---:|---:|---:|---:|---:|
| pure | 0.0000038147 | 0.0000035763 | 0.0000036955 | 0.0000019293 | 0/1,152 | 94/1,152 |
| cache n50/C10/knob-off | 0.0005117655 | 0.0001524687 | 0.0006102324 | 0.0003958878 | **51/1,152** | **37/1,152** |

| 同卡 floor | pure最大 relative-L2 | cache最大 relative-L2 | 超绝对硬预算 | 超相对硬预算 |
|---|---:|---:|---:|---:|
| 4090 | 0.0000017138 | 0.0002642525 | **19/2,304，均cache** | 0/2,304 |
| H100 | 0.0000016735 | 0.0001055529 | 0/2,304 | 0/2,304 |

跨卡51个预算失败均为max-abs失败，含35个head、16个tail；其中32个同时超floor。超floor的131个请求由129个绝对floor失败与57个相对floor失败的并集组成。Pure 组通过硬预算，但94个请求仍超严格floor，因此不能写成pure已完整等价。Pure 的floor失败请求最大误差仅0.0000022650；例如client21/delivery21两卡floor均约1.1921e-7，跨卡误差2.2650e-6、relative-L2 1.5677e-6，超过两项1e-6下限。这说明绝对误差很小与符合预注册floor是两个不同结论。

| 与同样两个冷启动/卡的native比较 | native | 两卡TF32-off |
|---|---:|---:|
| pure跨卡 max-abs | 0.0009070635 | 0.0000036955 |
| cache跨卡 max-abs | 0.0014218092 | 0.0006102324 |
| 4090同卡最大floor | 0.0010988712 | 0.0005117655 |
| H100同卡最大floor | 0.0013872385 | 0.0001524687 |
| 跨卡超硬预算 | 779/2,304 | **51/2,304** |
| 跨卡超逐请求floor | 7/2,304 | **131/2,304** |

超floor数量上升伴随同卡floor大幅收紧，不能解释为整体误差变大，也不能省略该失败项。最大cache误差仍是硬预算的约1.77倍。

## 3. Cache 决策和阈值

576个逻辑cache head的hit状态、top1、donor顺序、corrector是否应用、可观测stall/knob状态，在各卡4轮内部及跨卡全部一致（576/576）。2,304个动作chunk的gripper符号无跨卡翻转。Pure均MISS；cache均FULL_HIT，corrector在528个head应用、48个不应用；stall均inactive，knob均off。Pure的MISS不补足cache fallback MISS覆盖。

按同ID donor距离漂移定义的近邻band，本次8个head在band内、568个在观测band外，均无top1翻转；band外最大动作差仍为 **0.0006102324**。与native的48个near不能直接解释成输入减少，band宽度随本次漂移重新计算。所有576个head均离开已观测corrector ramp边界；stall margin缺失仍记unknown。

最大nearest-distance差为0.0003395081，corrector ratio差为0.0000209294。最大动作失败样例cache client6/delivery0的top1和donor顺序不变，top2 gap≥0.0073051，大于band0.0008583，但动作差0.0006102。因此剩余误差不限于近邻排序阈值，连续cache特征/距离/混合/校正路径值得逐段定位；现有action与scalar收据不能唯一归因某个算子、CPU BLAS、CUDA graph或GPU架构。

**层(b)仅可观测决策一致，完整覆盖未通过。** 尚缺cache MISS、active stall/periodic knob、足够独立查询以及实际内部noise/stage tensor的并发观测；层(c)SR及首次物理分歧未测。TF32消融对pure误差的强改善支持cuDNN低精度路径是重要因素，但关闭该项仍不足以恢复整个cache部署等价，也没有证明历史A100 SR下降的因果来源。

## 4. 资源、复现与排期

| 运行 | MPS生命周期秒 | 启动驻留MiB（非峰值） | 原有GPU context数 |
|---|---:|---:|---:|
| 4090 TF32-off 1 | 186.03 | 4,374 | 5 |
| 4090 TF32-off 2 | 186.66 | 4,374 | 5 |
| H100 TF32-off 1 | 164.12 | 5,916 | 3 |
| H100 TF32-off 2 | 144.40 | 5,916 | 1 |

四次均关闭本研究MPS daemon，原有context全部保留。8GiB仍是这些DP命令的空闲准入请求，不是峰值证明或硬隔离；原40GiB allocator cap未变。并行背景负载不同，不能从这些总时长估计TF32-off的纯性能成本。

本次只串行传输两份新增4090结果至H100 `reference_results/`，共 **208,913,094 bytes**，只读allowlist、`--bwlimit=16384`，删除文件0；[传输收据](evidence/transfer_both_tf32off_20261018.log:1)。传输前/后检查IO PSI，无并行大读。H100原始结果保留原位置，仅将CPU比较摘要和审计收据返回本地。比较明细位于H100 `/home/exouser/xgpu_1017/evidence/dp_both_tf32off_cross_20261018.json`，SHA256 **727947cc31609d949d6fb2fb404cf81c28bf3d18694ac2d02cc0f2657dfac2fd**。

CPU复现（GPU显存0，约1分钟；原始结果已就位）：

```bash
tether exec --timeout 5m h100 -- env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 /home/exouser/openpi/.venv/bin/python /home/exouser/xgpu_1017/tools/concurrent_compare.py /home/exouser/xgpu_1017/reference_results/4090_dp_concurrent_longseg_recal1018_20261018T122706Z /home/exouser/xgpu_1017/results/h100_dp_concurrent_longseg_recal1018_20261018T115441Z --left-repeat /home/exouser/xgpu_1017/reference_results/4090_dp_concurrent_longseg_recal1018_20261018T123059Z --right-repeat /home/exouser/xgpu_1017/results/h100_dp_concurrent_longseg_recal1018_20261018T122426Z --absolute-budget 0.000345707 --relative-budget 0.001 --output /home/exouser/xgpu_1017/evidence/dp_both_tf32off_cross_20261018.json
```

执行harness及比较算法未改；沿用已通过的85项CPU回归，另核对本次完整总体、hash、flags、budget及floor计数，见[交付验证](evidence/dp_both_tf32off_delivery_receipt_20261018.json:1)。两卡TF32-off四次已足够回答本轮问题，无须为本轮分析重复GPU命令。

**大显存计划原样保留：** Fast-WAM H100在`fa_pi_f_f_o`之后、4090在π0.5 MT50之后，原42GiB/次、预计8–15分钟、每卡两个冷启动；π0.5/GR00T/Cosmos与五模型闭环pilot仍按[COMMANDS](COMMANDS.md:1)及[闭环设计](CLOSEDLOOP_PILOT_1018.md:1)。A100此次未启动新实验。后续定位保留既定顺序：highest已生效，不重复算作新消融；固定实际使用的attention backend、关闭/固定compile与CUDA graph、deterministic。无实际attention调用的开关只能记空操作，不能宣称恢复设置。任何恢复结论仍须匹配正式并发配置、两次冷启动与原预算，并最终补闭环证据。
