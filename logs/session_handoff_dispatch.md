# Session Handoff — Dispatch Surface（Rev 1 执行期，2026-08-29 00:20）

> **A′ primary 两 suite 已跑完并分析：Gate 1 双双未过（`line_demoted` × 2），实验按 owner 指示暂停。**
> 结果、机制分解、出路选项与给 codex 的问题全在 [`dispatch_surface_rev1_aprime_result.md`](dispatch_surface_rev1_aprime_result.md)。
> 本地副本 `exp/dispatch_surface/data/aprime_rev1/<suite>_primary/`。weilandserver 的 policy server **已停**（跑分后未重启），重启命令见 §1。
>
> ~~**A′ 正式实验正在跑。**~~ 本文只写「现在什么状态 / 接下来做什么 / 会踩什么坑」。
> 设计推导与评审史不在这里，见 §0。**旧协议的负结果永久保留，不得被 Rev 1 的结果覆盖或冒充。**

## 0. 权威文档

| 文档 | 内容 |
|---|---|
| [`dispatch_surface_revised_protocol_draft.md`](dispatch_surface_revised_protocol_draft.md) | **Rev 1 权威正文**（§1–§9 由 Review Authority 改写）+ §10–§11 审查轨迹 + §12 G1 裁决 + §13 执行方复核 + §14 D0 记录 + §15 A1 规格空隙 + §16 G2 R1 的八条 blocking |
| [`dispatch_surface_open_questions.md`](dispatch_surface_open_questions.md) | 旧协议双双止损的问题文档（抬头有首要更正块）+ §7 codex 复核 |
| [`dispatch_surface_plan.log.md`](dispatch_surface_plan.log.md) | §11 执行日志（含 B1 重大更正、机制链条） |
| [`dispatch_surface_cost_axis_change.md`](dispatch_surface_cost_axis_change.md) | 成本轴口径（GPU inference ratio，非墙钟） |

## 1. 现在的拓扑（**照着做，别按老记忆**）

```
timan107 (client/worker)                      weilandserver (server)
/scratch/zixuans8/openpi_dispatch    ──►      :23150  ziyanglin.com 1:1 直连
8× GTX1080 做 EGL 渲染, --gpus 8              4090 48G, 4 replica, batch 32/25ms
conda -p /scratch/zixuans8/libero_sim         tmux srv, /tmp/srv.log
./precheck_t107.sh <suite> <layer>            HOME 必须 export /home/weiland
产物 /tmp/dsp_precheck/<suite>_<layer>/
artifact+配置 /tmp/dsp_shared/（两机同路径）
```

**几条硬事实**（都是踩出来的）：

- **worker 绝不能放 weilandserver**：MuJoCo 的 EGL 渲染在 GPU 上是 **G 类上下文**，32 个 worker 吃掉 **9.7 GB** 显存，把 48 GB 卡压到只剩 731 MB。`nvidia-smi --query-compute-apps` **只列 C 类**，用它查会得出「worker 没上 GPU」的错误结论——要看 `nvidia-smi` 完整输出的 G 行。
- **client 不需要 library pkl**：`load_cache_config` 对 `preload_path` **只校验非空、不校验存在**（只有 `surface_artifact_path` 有 `exists()`）。所以传给 client 的包是 **21 KB 不是 1.4 GB**。
- **conductor 只能本地 spawn worker**（`agent.py:_default_spawn` 用 `subprocess.Popen`），所以 runner 必须跑在 client 机器上；而 runner 会校验 yaml 里的 artifact 路径存在 ⇒ **artifact 必须在两机同一绝对路径**。timan107 无 root 建不了 `/data`，故用 `/tmp/dsp_shared`。
- **不能直接把 artifact 复制到新路径**：emitter 有内容绑定检查（`fit record does not bind this artifact path`），必须让 `fit_surface --out-dir` 直接输出到目标路径。
- **走直连不走 broker**：`ziyanglin.com` 的 **23100-23199 是 1:1 映射**（公网 :23150 → 本机 :23150），比 `tether expose` 少一跳。
- **tether push 的 allow_roots**：timan107 是 `[/home /tmp /srv]`，`/scratch` 推不进去，要先推 `/tmp` 再 `mv`。传输通道堵住（`too_many_in_flight`）时，小载荷可走 `tether exec` 的 stdout 用 base64 搬。

## 2. 已完成

| 步骤 | 结果 |
|---|---|
| G2 R1 的八条 blocking | ✅ 全修（B1 resume 时间戳 / B2 D0 绑定 / B3 A1 降级为诊断 / B4 冻结常数 / B5 内容绑定 / B6 双层 roster / B7 跨-suite finalizer / B8 verdict 措辞） |
| 两张标定表重建 | ✅ l10 9205 行、spatial 3364 行，各 150 ep、`{fit:50,cal:100}`；旧表存 `*.PRE_B1FIX.jsonl`（**只读锁定，勿删**） |
| **影响面验证** | ✅ `s`/`v`/`y10`/`k_eff`/`winner_id` **max\|diff\|=0**，只有 `y7` 变（中位 0.10，约 3%）——与 B1 的影响面预测完全一致 |
| 新表 D0 | ✅ 两 suite PASS，`check1 failures=0`，用**冻结的字面量 0.3** |
| fit → artifact | ✅ 两 suite 各 SV primary / SV minus / S0，全部 `empirical_no_certificate` |
| **冻结 δ\* 复算** | ✅ l10 **5.9096355438**、spatial **6.1298200607**，均在 1e-6 内 |
| emit 四个矩阵 | ✅ primary 6 臂 `always_search` / secondary 4 臂 `score_hysteresis` |
| launch dry validation | ✅ 四条链全过；timan107 跨节点也过 |
| 回归 | 3356 passed / 14 skipped；改动文件 ruff 全净 |

## 3. 已跑完 / 已暂停

l10 primary 1800/1800 → `line_demoted`（`d_c_p95=−0.0128`）；spatial primary 1800/1800 → `line_demoted`（`d_c_p95=+0.0865`）。
secondary 两层与 `finalize_cross_suite` **未跑**。cron 已删，Monitor 已结束。

server 重启命令（weilandserver，tmux `srv`）：
```
cd /data/openpi_dispatch && export HOME=/home/weiland && export PYTHONPATH=/data/openpi_dispatch/src:/data/openpi_dispatch && export OPENPI_SERVER_GPU_MEMORY_LOCK=0 && /home/weiland/openpi/.venv/bin/python scripts/serve_policy.py --replicas 4 --replica-spawn-batch 2 --port 23150 policy:checkpoint --policy.config pi05_libero --policy.dir /home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch 2>&1 | tee /tmp/srv.log
```
timan107 启动脚本已加 `export LIBERO_CONFIG_PATH=/home/zixuans8/.libero`（否则 LIBERO import 时 `input()` 卡死）。

吞吐实测（**这是本轮唯一可信的性能数据，之前的推断全错过**）：

| 配置 | worker | ep/min |
|---|---|---|
| weilandserver 本机闭环 | 8 | 9.6 |
| timan107 跨节点 | 16 | 12.6 |
| **timan107 跨节点** | **48** | **22.8** |

3 倍 worker 换 1.81 倍吞吐 ⇒ **server 未饱和**，但收益已递减；再提速要动 server 侧（加 replica / 加机器），
不是加 worker。**6000 episodes 从 10.4 h 压到 4.4 h。**
timan107 余量充足（load 8.95/48 核、每卡 2 GB/8.1 GB），瓶颈在 weilandserver 4 replica（util 100%）。

> 教训：`GPU util 100%` **不等于吞吐饱和**。今晚三次拿它当饱和证据都判错，最后靠实测推翻。

## 4. 没做 / 接下来做（**Rev 1 已归档；Rev 2 Phase 0 G1 APPROVED**）

codex 两轮裁决在 result 文档 §7、§9；执行方复核与计划在 §8。**Rev 2 协议草案 v0**：[`dispatch_surface_rev2_protocol_draft.md`](dispatch_surface_rev2_protocol_draft.md)。
Rev 1 finalizer 已跑：`suite_specific_only`。development AUC（l10）+0.0753，LB +0.030，LOTO 全正 ⇒ Decision Gate A-1 ✅。
codex §12 预 G1 裁决已全部采纳 → 协议 **v1**；Phase 0 代码计划 [`dispatch_surface_rev2_phase0_plan.log.md`](dispatch_surface_rev2_phase0_plan.log.md) 已于 2026-08-29 **G1 APPROVED Round 5**。下一步仅为按冻结计划进入 Code；代码仍须独立 G2，通过后才可由 owner 放行 rollout：0a anchor / 0b spatial p95,p97.5 / 0b′ l10 p85 / 0e S0 p80,p95 × 2 suite，合计 2700 ep。server 当前保持停止。

~~旧内容：~~

**l10 primary → analyze → spatial primary → l10 secondary → spatial secondary → `finalize_cross_suite`**

- primary 每层 1800 ep（6 臂），secondary 1200 ep（4 臂），合计 6000
- analyze：`exp.dispatch_surface.analysis.analyze_precheck --journal … --per-step … --arm-matrix … --fit-record … --launch-manifest … --split-manifest … --trials 30 --seed 20260827 --out …`
- 收口：`finalize_cross_suite --verdict <l10> --verdict <spatial>`，**四个 Gate 全过**才 `cross_suite_confirmed`，否则只能报 suite-specific
- 之后：data_authority 收编、`analysis/analysis.md`、同步 `docs/iclr/dispatch_defense_plan.md` §5

## 5. 等 codex 定夺

### 5.1 Gate 2 的 SR 分量是优越性检验，不是非劣性（**最重要**）

Rev 1 §7.3 冻结为 `q0.05(ΔSR) >= 0`。实测：真值相等时 `q0.05 = −1.645σ < 0`，σ 从 0.02 到 0.0002 **全部判败**。
即 **Rev 1 明确要它接纳的场景——「SR 相同、成本显著更低，真正的 Pareto dominance」——依然过不了**，与它诊断出的旧 Gate 2 缺陷同型。

真正的非劣性需要预注册裕度 `q0.05(ΔSR) >= −m`。**`m` 取多少是 owner/Review Authority 的决定，执行方不填。**
风险是实的：SV 省成本靠多用 FULL（离线 0.196→0.375），那恰是最可能压低 SR 的机制；而 S0 的 hitshare 和 accuracy 都不劣于 SV，**SV 全部的主张只落在成本一项**。

**但现在跑不浪费**：rollout 数据可复用，同一批 replicate 之后能用修订判据重新裁决，analyzer 重跑即可。已用测试把该行为钉住（`test_gate2_sr_component_is_superiority_not_non_inferiority`，注明「不要靠放宽门来修」）。

### 5.2 A1 止损的规格空隙（§15）

Rev 1 §4.1/§4.2 没继承旧协议的 A1 门，执行方据此在 empirical 分支只记录不 stop（否则两 suite 都产不出 artifact）。
A1 违反率强依赖网格粒度：(12,6) 0.393/0.282、(8,4) 0.212/0.173、(6,3) 0.185/0.111——每格中位仅 47 行时，无约束 0.95 分位实质是第 3 大值。**是否继承、是否允许因 A1 降档，待裁。**

### 5.3 D0 check1 的容差解释（§14.2）

§8.1 只写「从该 intermediate 用同一 current stage2 resume」，未规定 `start_t` 实参。执行方按冻结的**字面量 0.3** 门控（G2R1-B1 要求），并修 `run_stage3_from` 让它重放全循环的 float32 累加——现在 `check1 failures=0`。**该修改动了 `src/openpi/models_pytorch/pi0_pytorch.py`，属部署期语义变更，请复核。**

## 6. 未提交 / 未收编

- 本线所有改动**全部未 commit**（含 `d0_check.py`、`finalize_cross_suite.py`、schema v2、Rev 1 的 fit/emit/runner/analyzer 改动、`batching_coordinator` 默认值改 32/25）
- `/tmp/dsp_shared`（weilandserver 1.4 G / timan107 304 K）、timan107 的 `/scratch/zixuans8/openpi_dispatch`（41 M）实验后清理
- 132 GB 纯 teacher 语料仍无 data_authority 台账（`cache_size__*__all_s3.json` 记的是派生 pkl，不是它）

## 7. 勿误判

- `tests/exp/test_prebuilt_matrix_backend.py` 两个 bit-identical 断言是**既有失败**，验收要 `--ignore` 掉；用 `-x` 会停在那里导致后面根本没跑。
- `tests/review_tests/` 对执行方**密封**，不读不列不搜。
- 工作区有其他会话的 unstaged 改动（robocasa365 / rl_router / docs/iclr），勿动勿提交。
- **杀远端进程一律先 `ps` 取 PID 再 `kill <pid>`**：`pkill -f <模式>` / `pgrep -f <模式>` 会匹配到自己这条命令的 argv 把自己的 shell 杀掉。今晚踩三次，其中一次还让一个已被 owner 拒绝的运行偷偷跑了 7 分钟。
- **`tether exec` 本地返回 exit 0 ≠ 远端跑完**：长任务会因 10 分钟 chunk 超时提前返回。判断远端是否结束看 `pgrep` 或日志 mtime，别看本地退出码。
- **改完代码立刻推远端并核 sha256**：远端是独立克隆。今晚因此白跑两轮（`fit_surface` 不认 `--d0-record`、`d0_check` 缺 `D0_PROTOCOL`）。**符号级报错先怀疑同步，再怀疑代码。**
