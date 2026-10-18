# 五模型闭环 pilot：正常 formal prepare/launch，CPU 接线交付（2026-10-18）

**20 个独立 campaign 已生成并完成 CPU 核验；没有启动任何 GPU、worker、driver、chain，也没有远程 prepare。** 每个 campaign 为一个模型×一个 arm×一个 GPU，500 episodes；每模型每卡 pure+cache 共1,000。协调员在仓库根目录执行下列脚本。`prepare` 是正常 formal 的 CPU staging，`launch` 才启动；不得把 CPU fixture 的 PREPARED 当真实准备收据。

## 用法及每 campaign 命令

先做一次 H100 研究输入同步（已有输入会复用；公共 rsync，16MiB/s、单流、先查 IO PSI）：

```bash
cd /home/weiland/projects/openpi
bash exp/offline_search/rounds/r17/xgpu_1017/stage_h100.sh
```

所有 prepare/launch 均在 **weilandserver** 执行，H100 脚本通过正常 formal coordinator 调用 tether exec；不要把 H100 launcher 搬到 H100 执行。准备可以逐条提前完成；launch 每条只执行一次，放到对应 lane 边界。所有命令已保存为独立脚本，方便链直接引用。

```bash
cd /home/weiland/projects/openpi
P=exp/offline_search/rounds/r17/xgpu_1017/closedloop_launch_1018
```

### 4090 链

```bash
bash "$P/prepare_xp4_pi05_pure.sh" && bash "$P/launch_xp4_pi05_pure.sh"
bash "$P/prepare_xp4_pi05_cache.sh" && bash "$P/launch_xp4_pi05_cache.sh"
bash "$P/prepare_xp4_groot_pure.sh" && bash "$P/launch_xp4_groot_pure.sh"
bash "$P/prepare_xp4_groot_cache.sh" && bash "$P/launch_xp4_groot_cache.sh"
bash "$P/prepare_xp4_dp_pure.sh" && bash "$P/launch_xp4_dp_pure.sh"
bash "$P/prepare_xp4_dp_cache.sh" && bash "$P/launch_xp4_dp_cache.sh"
bash "$P/prepare_xp4_fastwam_pure.sh" && bash "$P/launch_xp4_fastwam_pure.sh"
bash "$P/prepare_xp4_fastwam_cache.sh" && bash "$P/launch_xp4_fastwam_cache.sh"
bash "$P/prepare_xp4_cosmos_pure.sh" && bash "$P/launch_xp4_cosmos_pure.sh"
bash "$P/prepare_xp4_cosmos_cache.sh" && bash "$P/launch_xp4_cosmos_cache.sh"
```

### H100 链

```bash
bash "$P/prepare_xph_pi05_pure.sh" && bash "$P/launch_xph_pi05_pure.sh"
bash "$P/prepare_xph_pi05_cache.sh" && bash "$P/launch_xph_pi05_cache.sh"
bash "$P/prepare_xph_groot_pure.sh" && bash "$P/launch_xph_groot_pure.sh"
bash "$P/prepare_xph_groot_cache.sh" && bash "$P/launch_xph_groot_cache.sh"
bash "$P/prepare_xph_dp_pure.sh" && bash "$P/launch_xph_dp_pure.sh"
bash "$P/prepare_xph_dp_cache.sh" && bash "$P/launch_xph_dp_cache.sh"
bash "$P/prepare_xph_fastwam_pure.sh" && bash "$P/launch_xph_fastwam_pure.sh"
bash "$P/prepare_xph_fastwam_cache.sh" && bash "$P/launch_xph_fastwam_cache.sh"
bash "$P/prepare_xph_cosmos_pure.sh" && bash "$P/launch_xph_cosmos_pure.sh"
bash "$P/prepare_xph_cosmos_cache.sh" && bash "$P/launch_xph_cosmos_cache.sh"
```

## 单 campaign 预算（500 episodes；pure、cache 各自占一行同等窗口）

表中小时是每个 arm 的服务器 GPU 窗口估计，含启动余量；不是实测吞吐。pure/cache 的实际速度可能不同；不能因先完成一部分 episode 就算完整。

| 模型 | 4090 每 campaign GPUh / 空闲GiB | H100 每 campaign GPUh / 空闲GiB | R/B；固定总worker |
|---|---:|---:|---|
| π0.5 | 0.5–1 / 44 | 0.5–1 / 44 | R4/B8；112 |
| GR00T | 0.5–1.3 / 42 | 0.3–0.8 / 76 | 4090 R6、H100 R11，B1；40 |
| DP | 0.25–0.75 / 8 | 0.25–0.75 / 8 | R4/B8、原生 CUDA graphs；48 |
| Fast-WAM | 0.75–1.5 / 42 | 0.75–1.5 / 42 | 实际4090 CPU-T5 profile R2/B16；12 |
| Cosmos | 0.75–1.5 / 47（条件） | 0.5–1 / 76 | 源R8/B1；44 |

两 arm 合计就是以上各项×2。四小时 campaign watchdog 与500-episode完成检查保留；本表不是强制时间中止阈值。GR4090是显式容量arm，不能把R6/R11比较当纯架构因果；Cosmos4090须先通过R8开环容量准入，失败不能自动降R。DP仍用既定 **native numerics** 闭环arm，TF32-off是另行记录的诊断，不静默混用。原估计依据与SR区间见 [CLOSEDLOOP_PILOT_1018.md](CLOSEDLOOP_PILOT_1018.md)。

## renderer 槽位已接线，运行时仍须有实际空闲资源

| 模型 | timan1 GPU2 worker | timan107 worker（设备0–7内原生分配） | renderer额外VRAM总请求 | timan107 RAM：pure / cache |
|---|---:|---:|---:|---:|
| π0.5 | 64 | 48 | GPU2 8.3125GiB；107合计48GiB | 96 / 192GiB |
| GR00T | 20 | 20 | GPU2 2.5977GiB；107合计20GiB | 40 / 80GiB |
| DP | 0 | 48 | 107合计33.61GiB | 96 / 192GiB |
| Fast-WAM | 0 | 12 | 107合计8.40GiB | 24 / 48GiB |
| Cosmos | 0 | 44 | 107合计33GiB | 132 / 176GiB |

GPU2使用原生UUID/PCI隔离授权（`filter_failed/timan1_gpu2.py:25`、`:35`、`:185`），MT50每worker 133MiB VRAM/512MiB RAM，故π另需32GiB RAM、GR另需10GiB。timan107数值沿用campaign请求；cache按原生fleet的至少4GiB RAM/worker保守预检。各节点还要保留20GiB RAM headroom及原生GPU headroom/已有reservation，不能把表内合计当唯一空闲要求。原生GPU2授权没有LIBERO pure，因此三个LIBERO模型的两个arm都只放timan107，避免改动guard或给两arm不同placement。**没有timan108或模型GPU上的本地renderer。**

两条链共享本研究 `pilot_renderer.lock`，在启动任何模型前串行占用本研究的renderer窗口；不锁住、不改动其他人的lane。取得锁后重新执行原生GPU/RAM admission；不足则在创建LAUNCH之前报错，保留诊断，可待资源释放后重试。正式fleet仍用原生GPU lease、worker recycle和cleanup。没有声称当前槽位已预留。

## 五项接线与证据

1. 复用队列入口的 `filter_failed.stage._stage` 主体、`formal_prestage.launch_prepared`、冻结runtime/workers档案、封存bootstrap、模型spec/cache YAML。`pilot_formal.context`仅在当前进程强制autoscale OFF，并跳过自适应placement；全局ENABLED文件不动。
2. 正常 `formal_eval.transfer → direct_transfer` 使用公共rsync（`exp/r17_integration/ops/direct_transfer.py:98`、`:121`），内部push/pull是方向参数，网络上没有tether push/pull或SSH。每次只暴露一个临时payload，固定16MiB/s、串行锁，原生关闭自己daemon；不改现有23177只读模块/ACL。H100新增assets只写研究目录，缺失canonical cache文件由进程内只读精确别名映射到已有同SHA私有输入，不修改YAML/fit或运行中目录。
3. native OFF driver已逐份生成且planner核验20/20。两卡各arm完整任务字段在去掉endpoint后完全一致。MT50固定num_trials=10；GR/DP cache的当前planner尊重longseg `task_namespace`，与旧任务表仅task_uid/yaml_id变化，保留 `TASK_NAMESPACE_MAP.json`；init、seed、其余字段逐个相等。实际执行列表是各目录 `EPISODE_PLAN.json`，原预注册PLAN保留。
4. trace按值注入原生fleet最终child factory；初次、recycle及GPU2隔离child均接入真实WebsocketClientPolicy。记录wire action、cache决策、观测state/RGB hash、server_timing、decision-step、terminal ACK执行前缀；写入本campaign worker_run/xgpu_trace，随正常worker证据回收。无RGB大payload；缺trace计unknown，不能伪装成模拟器全部隐状态一致。
5. P0…PJ短目录调用各自实际冻结mps.py，CPU逐份验证pipe≤85字节；prepare seal核对后才能launch。driver release只安装在研究目录；冻结共享runtime只读校验复用。测试覆盖真实stage主流程（仅远程传输/worker准备叶子stub）、实际fit loader、native worker argv及回收、两卡任务一致和资源拒绝；不宣称远程端到端已运行。

机器索引：[CAMPAIGNS.json](closedloop_launch_1018/CAMPAIGNS.json)。**98项CPU回归通过，最后13项pilot回归再次通过。** 不同冻结版本fixture会改变Python namespace，测试按模块使用独立CPU解释器；入口 `run_cpu_tests.py`。CPU收据：[20个driver/plan](evidence/pilot_driver_binding_1018.json)、[MPS路径](evidence/pilot_mps_CPU_1018.json)、[CPU回归](evidence/pilot_all_cpu_tests_1018.txt)、[最终pilot回归](evidence/pilot_cpu_tests_final_1018.txt)、[交付收据](evidence/pilot_delivery_1018.json)。本轮无sudo、无lane/chain编辑、无GPU启动。

## 剩余大显存并发轮次顺序

Fast-WAM既定边界不变。之后两卡均按 **π0.5 → GR00T → Cosmos**；每模型连续两个独立冷启动，再换模型。π每卡44GiB、合计30–50min；GR4090 42GiB/H100 76GiB、合计16–30min；Cosmos4090 47GiB/H100 76GiB、合计30–50min。H100后两项需接近整卡边界，不能塞进没有足够空闲的formal lane；不把两个大模型并行加载。具体6条并发命令保持 [COMMANDS.md §0.1](COMMANDS.md) 不变。
