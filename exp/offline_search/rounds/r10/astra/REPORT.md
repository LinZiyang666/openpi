# 给 owner 的说明

已完成第三种修正器 GC_dist，共 24 个 arms，已经冻结并生成，尚未启动闭环评测。
全程没有读取任何旧闭环运行的客户端、流水、汇总、服务端日志或结果。只复用了允许的
R10 配置、缓存和修正头；训练、选规则和校准都来自 B 示范库。sol 的文件和已有 run root 未改动。

做法很简单：沿用 G 守卫和第二版 LOEO 修正头，离示范库近时最多修正一半，距离变远就逐渐收手。
距离达到校准标尺的两倍时不修正，第一步也不修正。每个模型、任务套件和库大小只增加一个距离标尺，
没有按具体任务另设参数，也没有增加训练样本。距离标尺通过整局留出、重新拟合训练局的距离度量得到，
没有拿含有查询局的度量来假装测量新的一局。

离线结果是小幅收益：24 组中 19 组比固定半强度的 LOEO 动作误差更低。各组等权平均，
LOEO 相对无修正降低 3.669%，GC_dist 降低 4.096%；GC_dist 相对 LOEO 平均降低约 0.424%。
GR00T 的 12 组全部改善。π0.5 最大两个库有取舍：500 局的长任务、空间任务分别比 LOEO 差约
0.85%、1.01%。这不是成功率，也不能承诺闭环会提高成功率。

规则是从预先列出的 18 个候选中，用同一批整局验证结果选出的。因此这些数字是调参验证结果，
不是额外独立测试。PCA 沿用当前 B 子库的固定基；每折的距离度量、动作尺度和修正头都重新拟合，
验证局不参与这些拟合。校准使用比最终部署更小的候选库，未做库大小外推，门限可能偏宽松。

预测在 2026-10-02T18:22:47.770823+00:00 写下并加哈希封存，早于两个根目录的 emitter 输出。
24 个真实 CPU plugin 自检全部通过；两个标准 control plan 均已生成。没有进行同步、链式运行、
服务端、评测 worker 或远程启动。

## Frozen rule

For step > 0, `r = min geometric distance to retrieved rows / scale` and
`strength = 0.5 * clip((2.0-r)/1.25, 0, 1)`. Step 0 and nonfinite distances use zero.
Strength is .5 below r=.75, .4 at r=1, .2 at r=1.5, and zero at r>=2.
The same rule serves all 24 arms. `calibration/*.json` stores one scalar per cell-size.
These units are held-out-episode distance units, not Opus's in-library LOEO ratio.

The implementation subclasses sol's SizeController and RecipeCorrectedBase, preserves
the exact GC_loeo fit composition, and multiplies only the motion residual. The first
10 steps and channels 0–5 are corrected. Gripper, remaining steps, G behavior,
candidate retrieval, kernel, library indexing, and corrected-anchor continuation
follow the existing implementation. G can fire differently on later trajectories.
Exceptional regime-1 looks use geometric distances of the actual retrieved members.
One additional `_dist` calculation is made for each non-step-0 correction; no key
encoder, new store array, task gate, online adaptation, or additional policy call is added.

## Offline comparison

Metric: normalized 10-step six-motion MSE, averaged within each episode and then
over episodes. Five outer folds exclude whole episodes from head/metric fit and
donors. Four inner folds calibrate each outer training set's distance reference.
All 252,218 selected B rows are represented; nested sizes reuse the same episodes.
Failures are retained. Final deployment reuses sol's 240 original task heads exactly;
only CV heads are refitted. No PAIR or drift-augmented head is used.

| Cell-size | LOEO vs no corrector | GC_dist vs no corrector |
|---|---:|---:|
| pi05_l10_50 | +0.07% | -1.00% |
| pi05_l10_100 | -4.39% | -4.40% |
| pi05_l10_200 | -5.76% | -5.80% |
| pi05_l10_300 | -5.94% | -6.18% |
| pi05_l10_400 | -5.89% | -6.05% |
| pi05_l10_500 | -7.21% | -6.42% |
| pi05_spatial_50 | -3.06% | -2.94% |
| pi05_spatial_100 | -2.72% | -3.43% |
| pi05_spatial_200 | -6.49% | -6.63% |
| pi05_spatial_300 | -7.47% | -7.19% |
| pi05_spatial_400 | -8.46% | -7.62% |
| pi05_spatial_500 | -9.36% | -8.44% |
| groot_l10_50 | +1.53% | -0.43% |
| groot_l10_100 | +0.06% | -1.69% |
| groot_l10_200 | -2.35% | -3.21% |
| groot_l10_300 | -2.49% | -3.49% |
| groot_l10_400 | -2.43% | -3.42% |
| groot_l10_500 | -3.43% | -3.64% |
| groot_spatial_50 | -1.19% | -1.53% |
| groot_spatial_100 | -1.15% | -1.59% |
| groot_spatial_200 | -1.35% | -2.33% |
| groot_spatial_300 | -2.54% | -3.27% |
| groot_spatial_400 | -2.90% | -3.85% |
| groot_spatial_500 | -3.17% | -3.78% |

[OFFLINE.md](OFFLINE.md) includes absolute MSE, conditional episode-bootstrap intervals,
and the pooled distance bins. [distance_bins.csv](distance_bins.csv) contains every
cell-size breakdown. In calibrated r∈[2,3), LOEO is +0.86% vs no corrector on average;
at r≥3 it is +11.23% (19 cell-sizes have support). GC_dist is exactly zero change in
both bins. These are equal-cell-size means, with sparse far-tail support.
The offline no-corrector comparator is G's synthesis without the head, not a simulated
closed-loop guard trajectory. See [PROTOCOL.md](PROTOCOL.md) for selection details.

## Emission and validation

- pi05: `/home/weiland/trace_runs/os_closed_loop/r10_corr3_pi05` — 12 arms.
- GR00T: `/home/weiland/trace_runs/os_closed_loop/r10_corr3_groot` — 12 arms.
- Both manifests are byte-copied from corr2 and checked as tasks 0–9 × official inits 0–49.
- Standard emitter supplies all YAMLs and matrices. Full-model, policy-tail, guard-only,
  cost-ledger and client settings are copied from the corresponding corr2 GC_loeo arm.
- Frozen G/head equality audit: all 24 fits, every existing inner attribute/array unchanged;
  only the base class plus `distance_scale` and `distance_rule` are added.
- CPU integration: 24 passing selftests, 1,152 decisions, 48 connections, 96 episodes,
  96 forced no-progress triggers, 576 serving-correction checks.
- Additional gate audit: 576 peak/interior/zero/early/gripper/anchor checks; all pass.
- Both control plans pass; all 24 relocated fits pass path/head checks.
- Four semantic tests pass. Vectorized offline vs single-query serving maximum sampled
  chunk difference is 0.0001953; serving plugin checks use exact correction bytes.
- 134 protected source/spec/subset/G/head/GC files have unchanged hashes.

Synthetic CPU-test telemetry is under `astra/test_runs/`; no test telemetry is read
from an os_closed_loop run. The filesystem audit hook restricts old closed-loop reads
to allowed arms/manifest/fit/config files and all file writes to astra or the two new roots.
Its initial selftest attempt blocked dill's `/dev/null` type probe before inference;
allowing that character sink fixed the harness. No frozen serving rule or fit changed.

Emission receipts: `{"r10_corr3_groot": "2026-10-02T18:23:06.574132+00:00", "r10_corr3_pi05": "2026-10-02T18:23:06.328341+00:00"}`.

## Deployment preparation

Extra h100 store bytes: **0**. New run/calibration destinations in the standard plans:
**1,030,759,546 bytes (0.960 GiB)**. The union of both complete dependency plans
is **25,414,418,693 bytes (23.669 GiB)**, including existing shared payload/store,
canonical G and head dependencies. The incremental count assumes those existing R10
dependencies are already available; no remote inventory was inspected.
The new serving sources add **3,740 bytes**, outside the asset plans.
No transfer has occurred. Exact files, hashes and reasons are in each root's
`h100_sync/plan.json` and [deployment.json](deployment.json).

| New h100 source | Bytes | SHA256 |
|---|---:|---|
| `exp/offline_search/rounds/r10/astra/__init__.py` | 68 | `c9b7d0f526d8321530931b86fc144fe49688062f9a6c29d8ee9e127e15af618e` |
| `exp/offline_search/rounds/r10/astra/method.py` | 3672 | `b0614331f55fe30307ec7cbd8c05e5523edcdd8858b97b5e02e209280d2ba794` |

Only these two new Python files are needed at serving time, in addition to the already
required sol R10 and imported R9/R8/R4 runtime sources. Offline/build/audit helpers are
listed separately in [source_files.json](source_files.json). All new Python hashes are
in [ALL_SOURCES.sha256](ALL_SOURCES.sha256); runtime hashes in [H100_SOURCES.sha256](H100_SOURCES.sha256).

Prepared at 2026-10-02T18:30:23.050866+00:00. [PREDICTION.md](PREDICTION.md) and [freeze.json](freeze.json) remain unchanged.
