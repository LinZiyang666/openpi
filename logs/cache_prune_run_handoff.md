# Cache Prune：conductor / concurrent server 运行交接

> 2026-09-11；Execution；Owner Ziyang Lin 已授权替换旧运行计划并直接实施、测试、commit/push。
> 本文是当前运行入口；`cache_prune_plan.log.md` §6.3/§9/§11 的单臂运行安排已废弃。剪枝算法和统计定义仍按原计划。
> 本会话交付运行代码和准备步骤，尚未启动模型服务、smoke 或正式 rollout。

## 1. 启动是否需要 warmup

**不需要 RIT warmup、风险拟合或归一化重标定。** 每个臂只需已经冻结的 YAML 和对应 PKL。检索权重、μ/σ、`always_search`、`always_hit`、d1 保持不变。Stage 1 仍要运行以提取 query key，Stage 2/3 为 meta，不调用动作模型填补 cache miss。

运行入口会做依赖与 init 检查、加载本组全部库、以及独立 smoke。它们是运行完整性检查，不产出标定数据，不改变参数。独立检索微基准有计时预热，也不属于 RIT warmup。

## 2. 实际使用的基础设施

- `run_concurrent.py`：本实验的分组、资源监测、失败留证和验收包装。
- `run_size_eval.PureCacheEvalStrategy`：每个 YAML 产生十任务 × 五十 init 的 eval stage；无 warmup stage、无 calibration barrier。
- `ConductorDriver` / `EpisodeScheduler` / `WorkerAgent`：现有中央队列、episode pull、attempt fence、worker 监督、journal 与 snapshot，不复制实现。
- `serve_policy.py` 默认 concurrent：每组若干独立端点，端点内多个 bundle；worker 依 `EpisodeTask.server` 连接并 `select_bundle(bundle_id)`。

worker 在完成一集后领取下一集；`eval_concurrency=2` 允许同一端点上前一 YAML 收尾时领取下一 YAML。默认 YAML 归属由 `assign_servers` 按 worker 容量分配，**不承诺任意端点间 work-stealing**。本入口冻结相同的分配结果并写出该组 matrix，实际 driver 使用同一算法和容量。

本版 launcher 在服务节点启动多个本机 concurrent 端点及本机 WorkerAgent；每个端点都是独立进程，不使用 `--replicas` router 广播复制全部库。它未提供跨机器 agent 启动/远端 PID 验收；本轮部署以 weilandserver 为单位，不把现有框架的跨机能力写成本入口已经实现的功能。

## 3. 分组及资源规则

`BackendPool` 以实际 preload 路径等指纹复用 backend，不逐出已载库。不同剪枝 PKL 是不同指纹，因此要约束**整组累计驻留库**，不能只依赖 `eval_concurrency`。

入口在任何 outcome 之前按冻结臂顺序、suite、内存预算生成 `schedule.json`：

1. 一组只含同一个 suite，worker 使用该 suite 的同一 A-pool。
2. 每端点最多 `arms_per_server` 个 YAML，默认 2；全部 distinct PKL 的 bytes 都计入估算。
3. 每端点估算 `model_budget_gib + resident_multiplier × Σ不同PKL字节`，再加全部 worker 的预算。默认 multiplier=3，仅为保守排程估算，不声称等于实测 RSS。
4. 超过整组预算或单进程 64 GiB 时缩小组；连一臂都放不下则停止。实际启动时还要满足 `min(配置预算, 0.75×MemTotal, MemAvailable−16 GiB)`。
5. 依次启动本组服务器、加载全部指定库并记录 `bundle_id` ack；库加载也受资源采样监测。全部 ready 后才启动 conductor/worker。
6. 运行中每 0.5 秒测量，2 秒、新峰值或进程集合变化时落盘。超预算则该组无效；只终止本组记录的 PID/start_time。
7. 整组结束后关闭这些专用 server，释放所有库，再开始下一组。不逐臂重启，不调用 `BackendPool.reset_for_tests`。

独立端点数、worker 数、设备映射和预算由运行会话在启动前按机器空闲容量填写并冻结。每 render GPU 总 worker 数不超过 15。示例不是性能最优宣称；无需为调整部署参数重新做算法标定。运行后改配置、代码、模型或设备环境需要新 run_id。

## 4. 已完成数据与还需准备的内容

权威离线根：weilandserver `/data/openpi/ablation_study/cache_prune/g2_r1_20260911`。

| suite | regime | 源轨迹/entries | 已有 |
|---|---|---:|---|
| libero_spatial | rit50 | 49 / 1,018 | source、scores、grid、P05、verification |
| libero_spatial | cs500_success | 439 / 9,329 | 同上 |
| libero_10 | rit50 | 50 / 2,640 | 同上 |
| libero_10 | cs500_success | 392 / 20,461 | 同上 |

每源子目录内：`source.json`、`scores/scores.json`、`grid_freeze.json`、`P05/artifact.json`、`verification.json`；每 suite 目录内已有 `eval_membership.json`。四份 P05 覆盖 33,448 query，max regret 全为 0。两 suite 各 500/500 common-unseen，seen 为空。

尚需完成其余 32 个剪枝子库、四个 P00 manifest/验证、完整 40 YAML 与 freeze。旧 P05 verification 在新增报告标签前生成，仍可作为已验证证据复用；不要改写其 seal。新报告用 `sanity_envelope` 标记宽松数值界，freeze 中的 1.0631978511810303e-4 实测包络只供漂移诊断，不作阈值。

## 5. 准备命令（运行会话执行）

先将提交同步到 weilandserver 的 `/home/weiland/openpi`。该目录此前只上传了离线验证代码，不能假设已包含本轮新入口；交接提交可用 `git log -1 --format=%H -- logs/cache_prune_run_handoff.md` 定位，再核对服务节点版本。使用服务节点 `.venv`，保留既有库与原始 H5。可在同节点另建工作目录，但源 manifest 里的文件、配置证据路径仍须可读且 SHA 一致。

```bash
cd /home/weiland/openpi
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
```

生成实际 inputs 并复用四份已验证 P05（只复制 manifest，不搬动或复制大 PKL）：

```python
import json
import shutil
from pathlib import Path

audit = Path('/data/openpi/ablation_study/cache_prune/g2_r1_20260911')
root = Path('/data/openpi/ablation_study/cache_prune/concurrent_v1')
root.mkdir(parents=True, exist_ok=True)
sources = []
for suite in ('libero_spatial', 'libero_10'):
    for regime in ('rit50', 'cs500_success'):
        base = audit / suite / regime
        sources.append({'source': str(base/'source.json'),
                        'scores': str(base/'scores/scores.json'),
                        'grid': str(base/'grid_freeze.json')})
        target = root/'cache_artifacts'/f'cache_prune_{suite}_{regime}_P05'
        target.mkdir(parents=True, exist_ok=True)
        for src, name in ((base/'P05/artifact.json', 'artifact.json'),
                          (base/'verification.json', 'verification.json')):
            dst = target/name
            if dst.exists():
                assert dst.read_bytes() == src.read_bytes()
            else:
                shutil.copyfile(src, dst)
inputs = {'sources': sources,
          'memberships': {s: str(audit/s/'eval_membership.json')
                          for s in ('libero_spatial', 'libero_10')}}
target = root/'inputs.json'
if target.exists():
    assert json.loads(target.read_text()) == inputs
else:
    target.write_text(json.dumps(inputs, indent=2))
```

```bash
prune_root=/data/openpi/ablation_study/cache_prune/concurrent_v1
.venv/bin/python -m exp.ablation_study.cache_prune.emit_prune_arms \
  --inputs "$prune_root/inputs.json" \
  --artifacts-root "$prune_root/cache_artifacts" --out "$prune_root/config"
```

emitter 会按冻结网格物理导出、逐臂验证并原子发布配置。四个 P00 指向各自原源 PKL。输出配置已存在时不可覆盖；恢复未完成的导出可复用通过身份门的 artifact，但不要覆盖不完整/冲突文件来强行继续。大型全量验证耗时较长，运行会话应留存 stdout/stderr 并检查完成状态。

## 6. 分组、smoke、正式评测

复制 `exp/ablation_study/cache_prune/config/concurrent_spec.example.json` 为 `$prune_root/concurrent_spec.json`，填写真实 checkpoint、解释器、conda 环境、端口、设备与预算。`server_python` 必须是运行入口的同一解释器。GPU budget 是本组全部相关 GPU 分配之和；render worker 与 server 都计入。

```bash
.venv/bin/python -m exp.ablation_study.cache_prune.run_concurrent \
  --freeze "$prune_root/config/freeze.json" --spec "$prune_root/concurrent_spec.json" \
  --run-dir "$prune_root/runs/run_001" --phase plan
```

`plan` 仅检查并冻结分组，不加载模型或运行 episode。检查 schedule 覆盖两 suite、两 regime、P00–P09 共 40 臂且每臂只出现一次；此时可以调整部署参数并换一个新 run 目录。正式运行保持同一 schedule。

获得实际启动指令后：

```bash
.venv/bin/python -m exp.ablation_study.cache_prune.run_concurrent \
  --freeze "$prune_root/config/freeze.json" --spec "$prune_root/concurrent_spec.json" \
  --run-dir "$prune_root/runs/run_001" --phase smoke

.venv/bin/python -m exp.ablation_study.cache_prune.run_concurrent \
  --freeze "$prune_root/config/freeze.json" --spec "$prune_root/concurrent_spec.json" \
  --run-dir "$prune_root/runs/run_001" --phase eval
```

smoke 是四源 × P00/P09 × 十任务各一个 init，共 80 集；正式 phase 在八臂 smoke 全通过前拒绝启动。正式每臂 500，共 20,000 集。两阶段的 group 和逐臂视图分开，smoke 不计入正式分母。

再次调用同一 phase 会跳过已完整验收的组；失败/中断组保留原始目录，以新 attempt 重新运行整组，不拼接不同 producer 的成功子集。conductor 在单次运行内仍使用原有 episode 重试与 attempt fencing。人工终止后先确认没有残留本组 PID，入口会拒绝仍有活进程的未完组。

## 7. 证据与分析

每组保留 `launch.json`、`preloads.json`、各 server log、资源 JSONL、原始混合 journal/per_step、原 runner launch、postflight 和 completion。`view_<wave>_<arm>/` 是原始字节按 YAML 分区的视图，保留所有 attempt，验收逐字节核对分区 SHA。不得手工修改视图来选择 episode。

逐臂继续验收：唯一 accepted producer/attempt、500 个配对 init、FULL_HIT=1、确实 searched、winner 属于对应保留库/任务、`steps−10` 与 replan=5 一致，以及正常失败必须达到 task timeout。未知 YAML/多余 episode、重复有效臂、库/模型/代码漂移均拒绝。

```bash
.venv/bin/python -m exp.ablation_study.cache_prune.analysis.analyze_prune \
  --freeze "$prune_root/config/freeze.json" --run-dir "$prune_root/runs/run_001" \
  --out "$prune_root/analysis/run_001.json"
```

原独立检索微基准和绘图入口继续有效，见原计划 §11.5。混合负载下的 `infer_ms` 含排队，标为 `concurrent_call`；共享 server RSS 标为 `shared_server_wave`，单库 RSS 留空。单个剪枝点的独立检索延迟仍以相同机器/线程的离线微基准比较；不将整组 RSS 或不同组负载的在线时延解释为某一库的独立内存/加速。

## 8. 本次交付状态

Owner 的运行替换明确覆盖原单臂流程及重复审批要求；不是自动 APPROVED 新代码。本次相关回归 **566 passed / 11 skipped**（含 cache_prune 全部 87 项），结果记录在原计划 §14。**裸全仓 Verify 在约 48% 的既有长时统计套件处主动停止，未完成，不声明全仓通过。** 主提交包含 cache_prune 实验、对应测试及文档索引；全仓检查发现的既有 CP2 测试夹具漏填 `schedule_id` 已单独修复提交为 `d016598`，`src/` 零改动。正式硬件可行性、吞吐和闭环效果由运行会话的真实 smoke/eval 给出。
