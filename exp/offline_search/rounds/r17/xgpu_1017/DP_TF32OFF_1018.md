# H100 DP TF32-off：第一冷启动分析（2026-10-18）

**历史记录：两卡各两个TF32-off冷启动已完成。** 最新[双卡分析](DP_BOTH_TF32OFF_1018.md:1)显示跨卡51/2,304超硬预算、131/2,304超逐请求floor，尚未恢复等价；下文首轮数据保留用于审计，末尾待补轮次现已完成。

**关闭cuDNN TF32是有实测支持的稳定化候选，但尚未恢复跨卡等价。** H100本次两轮动作floor低于预注册硬预算；与4090 native比较仍有253/2,304个逻辑请求超预算，且两卡设置不同。还需要第二次H100冷启动，以及4090同设置的两个冷启动。

证据：[分析摘要](evidence/dp_tf32off_summary_20261018.json:1)、[执行日志](evidence/run_concurrent_h100_dp_tf32off_1.log:1)。原始结果位于H100 `/home/exouser/xgpu_1017/results/h100_dp_concurrent_longseg_recal1018_20261018T115441Z`；详细CPU比较报告为同机`evidence/dp_h100_native_vs_tf32off.json`与`dp_4090_native_vs_h100_tf32off.json`，SHA记录于摘要。本次只有读取结果及CPU分析，没有启动GPU模型。

## 设置与完整性

R4/B8、48客户端、2轮×2组、4,608 delivery，状态COMPLETE。PIN、runtime、bootstrap、harness、input、cache文件前后校验均与native相同；四个resident原生CUDA graph自检全部PASS。四份NUMERICS收据均显示`cudnn_tf32=False`，matmul TF32仍False、precision仍highest，其余报告的torch flags未变。此消融的有效变化是cuDNN TF32关闭，单独再设highest不会引入新对照。

四个resident均实际访问B8，未将该消融降为B1 staged replay；batch分布见[原生ledger摘要](evidence/dp_tf32off_batch_coverage.json:1)。报告的全部package版本与H100 native相同，四份flags差异均仅`cudnn_tf32`，见[交付核对](evidence/tf32off_pilot_delivery_receipt_1018.json:1)。

MPS生命周期164.12s，启动驻留5,916MiB（5.78GiB，非峰值）；已有3个后台GPU context，关闭时全部保留。此前H100 native为121.79/124.06s、无后台context，所以不能把本次耗时增加直接归因TF32-off。

## 同卡并发floor

| 比较总体 | max-abs | 最大relative-L2 | 超硬预算请求 |
|---|---:|---:|---:|
| H100 native冷启动1内部两轮 | 0.0013344288 | 0.0010821896 | 63 / 2,304 |
| H100 native冷启动2内部两轮 | 0.0013872385 | 0.0010501897 | 56 / 2,304 |
| H100 TF32-off内部两轮 | **0.0001524687** | **0.0001055529** | **0 / 2,304** |

TF32-off分组floor：pure max-abs **0.0000034571**、relative-L2 0.0000016735；cache max-abs **0.0001524687**、relative-L2 0.0001055529。按请求最坏误差的第99百分位为0.0000007749。等两轮对照仍观察到max-abs约8.75–9.10倍下降；不是仅拿native四轮与ablation两轮比较。但只有一次ablation冷启动，不能把此值当完整冷启动floor，也没有实际kernel dispatch或全部内部noise tensor的因果证据。

## 相对native输出的变化

以下native侧均含两个冷启动共4轮，TF32-off侧为1个冷启动2轮；每请求跨设置比较8对。

| 对照 | pure max-abs | cache max-abs | 超硬预算 / 2,304 | 超逐请求5×floor / 2,304 |
|---|---:|---:|---:|---:|
| H100 native → H100 TF32-off | 0.0004107952 | 0.0007355213 | 98 | 111 |
| 4090 native ↔ H100 TF32-off | 0.0007616282 | 0.0008635521 | 253 | 82 |
| 原4090 native ↔ H100 native | 0.0009070635 | 0.0014218092 | 779 | 7 |

硬预算仍为max-abs≤0.000345707且relative-L2≤0.001，floor门槛仍为每请求`max(1e-6,5×两侧同设置重复漂移较大值)`。两组新比较的relative-L2分别最大0.0006155125、0.0005366656，均通过该项；max-abs仍失败。跨卡绝对误差减小，但超floor请求从7增加到82，说明不能只报告改善的指标，更不能因floor缩小就放宽标准。TF32-off相对native会改变连续动作，不是字节等价替代。

两组对照的576个cache head，top1、donor顺序、corrector、可观测stall/knob均一致，gripper符号无翻转；原生ACK时钟一致。与4090对照有48个head处在观测近邻边界band，仍无top1翻转；band外动作差最大0.0002388954。stall仍inactive、knob-off、cache无MISS，覆盖缺口不变；SR未测。

## 后续操作

保留已排定Fast-WAM窗口。低显存窗口先补H100同命令第二冷启动；4090再跑两次同一TF32-off配置，然后比较双方相同设置与各自冷启动floor。每次请求8GiB空闲，预计5–10分钟、watchdog15分钟；只测到native/本次启动驻留，未有ablation峰值保证。精确命令见 [COMMANDS](COMMANDS.md)。如果共同TF32-off仍不通过，再按原顺序固定attention、关闭/固定compile与CUDA graph、deterministic；已是highest/matmul TF32-off的空操作不重复当作有效消融。
