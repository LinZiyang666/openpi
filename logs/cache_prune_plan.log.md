# Cache Prune：跨轨迹短剩余优先剪枝 — 实施计划

> Status: `G2 R2 APPROVED；Owner 并发替换已完成，相关回归通过；全仓 Verify 未完成（2026-09-11）`。
> Date: 2026-09-11；Authority: Execution；Level: **L2**。
> 实验：`exp/ablation_study/cache_prune/`；当前授权范围：按 Owner 裁定完成并发替换、测试、commit/push 与运行交接；本轮不启动正式 rollout。
> **当前运行方法见 [并发运行交接](cache_prune_run_handoff.md) 与 §13/§14；下文旧单臂运行步骤仅保留为历史记录。**
> 核对代码基线：`b852027`；工作区已有其他实验改动，本计划不接管这些改动。

## 0. 需求与本轮边界

用户明确要求：在原约 500-trajectory 级大库和刚才分析的约 50-trajectory 级小库上，分别测试 **10 个剪枝点，含不剪枝**；LIBERO Spatial、LIBERO-10 共 **2 × 2 × 10 = 40 组**。沿用 RIT 中间 search strategy 的模态参数，`gate: always_search`、`judge: always_hit`，做纯 cache 闭环实验。剪枝采用已讨论的跨轨迹短剩余优先规则。

本计划将两个容易混淆的口径明确分开：

- “500 级”是原采集规模名称，本轮大库采用最大 S6 的**成功子库：Spatial 439 条、L10 392 条**；“50 级”指本轮离线分析用的 RIT 底座，Spatial **49 条**、L10 **50 条**。不把 cache_size/S3 的另一份 50 条库替换进来。
- 本轮十点是**不剪枝 + 九个剪枝相似度阈值**。§4.2 预注册删点率覆盖目标与机械选点规则；C1 中对四份源库分别做不读取评测结果的离线扫点，再按规则冻结四套九阈值。横轴使用实际删点比例，不把目标比例当实测值。
- **Owner 2026-09-11 对 G1 B3 明确裁定 (a)：大库改 S6/success，Spatial 439 条、L10 392 条。** 两档源库均只含成功轨迹，P00 同样引用对应成功库；不再保留 all 臂，不增加额外 outcome 扫描，仍为 40 组。未剪枝指“不执行 entry 剪枝”，不把之前整条轨迹的成功筛选算进压缩率分母。
- 不新增模型、benchmark、随机剪枝对照、teacher 臂或 RIT 阈值扫描。36 组剪枝 + 4 组同条件重跑的原库基线，合计 40 组。

早先 [离线报告](../exp/rit_pareto/analysis/trajectory_prune_20260911/report.md) 对完整 RIT 系统提出过“prune → warmup 重标定 → eval”。**用户本次纯 cache 指令收窄了实验**：本轮无 verdict 阈值、无 RIT 风险拟合，连检索的模态归一化参数也固定。模型/索引的计时预热可以做，但不是重标定，不增加实验轴。

## 1. 科学问题与证据定位

主问题：在相同检索规则和执行方式下，删除被跨轨迹短剩余点直接覆盖的缓存条目，如何改变 **闭环 SR、库大小和检索成本**？这一取舍是否依赖源库规模？

次问题：剪枝后，成功 episode 的实际执行长度是否变化？“选中缓存点的原轨迹剩余长度”只是离线代理，不能直接替代执行时间。

这是经验压缩消融。完整呈现四条十点曲线及相对各自 P00 的配对变化，不预设剪枝一定提高 SR。仅这 40 组不能证明本算法优于随机删点或其他压缩算法，也不能证明完整 RIT 系统在相同 SR 下加速；这些不在本轮结论范围内。

B3 机制边界：按 owner 选择的成功库方案，失败轨迹终点不进入任何一档的候选集或代表集。r 表示“距原成功轨迹记录结束的后继决策数”，仍是离线进度代理，不承诺从新状态回放该动作必然成功或更快。

## 2. 四份输入的身份

| suite | regime / 臂内标签 | 源 PKL | 轨迹 / entry | 来源 |
|---|---|---|---:|---|
| libero_spatial | 小库 / `rit50` | `exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl` | 49 / 1,018 | RIT 正式底座，本轮离线分析源 |
| libero_10 | 小库 / `rit50` | `exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl` | 50 / 2,640 | RIT 正式底座，本轮离线分析源 |
| libero_spatial | 大库 / `cs500_success` | weilandserver: `/data/openpi/ablation_study/cache_size/artifacts/cache_size_libero_spatial_success_S6.pkl` | 439 / 9,329 | cache_size S6/success |
| libero_10 | 大库 / `cs500_success` | weilandserver: `/data/openpi/ablation_study/cache_size/artifacts/cache_size_libero_10_success_S6.pkl` | 392 / 20,461 | cache_size S6/success |

小库权威记录：[`weighted_sum__libero_spatial__cp1_spatial_pool_16.json`](../exp/data_authority/records/weighted_sum__libero_spatial__cp1_spatial_pool_16.json)、[`weighted_sum__libero_10__cp1_spatial_pool_16.json`](../exp/data_authority/records/weighted_sum__libero_10__cp1_spatial_pool_16.json)。本地文件 SHA256 分别为：

```text
Spatial 09cacb814017a40c3b612da45830db07e52b291b3ae37f8b017732a9f475d2e1
L10     de51731df08e93af2f34adb85c919b45ed53b244e92b3f15e1332b8655410593
```

服务器小库副本字节 SHA 与本地不同，登记记录中已有对应 SHA 和内容 digest，不用本地 SHA 错判服务器副本。**在服务节点从实际服务的那一份副本剪枝，P00 preload 直接引用同一文件**；全部 Pj 满足 `parent_sha(Pj) == preload_sha(P00)`，本地内容 digest 相同不能替代这条字节身份约束。落地时记录实际读取节点、resolved path、文件 SHA 和字节数，旧 digest 不是本次重新实测的替代品。L10 小库含早期 factors 富化；本实验 always_hit 不读取 factors。

大库 census 来自 `cache_size/config/entries_<suite>_success.json`，并由 2026-09-10 的 [成功子集长度统计](../exp/ablation_study/cache_size/analysis/trajectory_lengths_20260910/success_comparison_summary.json) 交叉核对；后者是 all-PKL 元数据按冻结成功名单筛选，不等于本次重新读取 success-PKL。**实施阶段第一步**在服务节点实测实际 success-PKL 的 SHA256、逐轨迹/任务 entry census、原链及 step 连续性，并核对 `lists_success/episodes_<suite>_S6.txt` 的完整 ID 集，再写 source manifest 后才能剪枝。

原 S6/all 中的 Spatial 11 条、L10 58 条失败轨迹本轮全部排除。大库 entry 分母分别固定为 9,329/20,461，小库为 1,018/2,640；不使用 all 库的 9,813/26,493 计算删点率。两档均来自成功轨迹，但采集时间、init 池和示范身份仍不同，不构成只改变规模的同源嵌套集合。主对照仍为各源库内部 P01–P09 对 P00，跨规模比较注明这一限制。

**成功身份验证**：大库按冻结成功名单与构建 manifest 逐轨迹核对，小库按其源 HDF5 的 success 标签及 init map 核对；源记录缺失或与库内显式 outcome 冲突即失败。老 entry 的 outcome 可能为 None，成功事实写在本实验 source manifest，不修改原 pickle 对象来补标签。全部来源验证通过后才能接受“成功库”命名；不能在数值网格冻结后再变 outcome 口径。

## 3. 检索配置与纯 cache 契约

模板选择现有 RIT+gate K3/IR60 YAML，只提取其**检索配置**；K3、IR60、原 gate 和原 judge 不成为本实验参数：

```text
exp/rit_pareto/data/k3_rith/libero_spatial/k3/rith/k3_sp_rith_ir60.yaml
exp/rit_pareto/data/k3_rith/libero_10/k3/rith/k3_l10_rith_ir60.yaml
```

两份 YAML 的快照也完整保存在离线分析 `trajectory_prune_20260911/<suite>/summary.json` 的 `config_contents`。进入 Code 后将检索模板固化在新实验 `config/`，记录原文件 SHA 和规范化检索配置 SHA，避免依赖另一实验未追踪的 `data/` 长期存在。

| 项目 | Spatial | LIBERO-10 |
|---|---|---|
| key_builder | `cp1_spatial_pool_16` | 同左 |
| vision_0 / vision_1 / robot_state 权重 | 0.0625 / 0.5 / 0.4375 | 0.5625 / 0.25 / 0.1875 |
| vision_0 zscore μ / σ | 0.977693693699334 / 0.00699373921570407 | 0.9739899923664463 / 0.0061831533438692935 |
| vision_1 zscore μ / σ | 0.9691840492031897 / 0.007853951498497307 | 0.9659078322399228 / 0.006527797454113087 |
| robot_state zscore μ / σ | −1.8439531429434792 / 1.0018373754826044 | −1.9584325681212513 / 0.7484941685797242 |

两个 suite 均采用：

- `weighted_score_sum_knn`、`top_k=1`、`task_scoped=true`、`trajectory_depth=1`、`step_filter=all`。
- 两路 vision 使用 cosine；state 使用 L2，保留模板 `to_similarity: {type: exp, tau: 1.0}`；最终打分调用正式 SearchStrategy/backend 的 per-field zscore–tanh 流程，不自行重写 L2 的方向或归一化顺序。
- `vision_2` 与 `prompt_emb` 禁用、权重 0。离线 query 也只传三个启用字段，防止 disabled field 通过 backend 默认权重混入。
- CP1 唯一启用；`gate: {type: always_search}`、`judge: {type: always_hit}`、`write_policy: {type: never}`；无 routing、sidecar、warm tiers、动态 normalizer、trajectory replay gate 或在线写入。
- 相同 suite 的两档库、十个点使用完全相同的上述配置，只有 preload 路径不同。正式 40 臂统一 `timer.enabled=false`；线上成本采用现成 client_timing 落盘数据，见 §6.4。服务端细分计时只可作为另行记录的诊断，不混入正式成本曲线。

YAML deep-diff 白名单只允许 preload 路径；臂身份、源库与剪枝信息写 matrix/manifest，不给 cache config 添加不受支持的键。

`AlwaysHitJudge` 在检索为空时会返回 MISS，故“配置 always_hit”不足以保证纯 cache：每个任务必须仍有可检索点，每个 accepted episode 的每个有效策略调用都必须 `FULL_HIT`，且 winner 位于该臂保留集。出现 MISS/WARM_START/跨任务 winner 即实验契约失败，不把 teacher 回退的结果当纯 cache SR。

## 4. 剪枝算法：固定原长度、跨轨迹、直接保留代表

记经 §2 成功身份验证的源库 entry 为 i，原成功轨迹长度为 L(i)，原 step 为 t(i)，固定剩余长度 `r(i)=L(i)-1-t(i)`。这里每个 step 是一次缓存决策；不将其乘以 5 当作精确环境步数。失败轨迹在源库验收时即被排除，不作为短剩余直接代表。

对每个任务独立执行，分数 `s(i,j)` 为 §3 **原库检索参数下**的融合分数。算法为：

1. 原库先校验每条轨迹的 step 序列为 `0..L-1`，prev/next 与源轨迹一致，entry ID 唯一，任务不混杂。固定原数组序号 `source_ordinal`。
2. 按 `(r(i), source_ordinal(i))` 升序访问。建立只增不减的保留集合 K。
3. 候选代表 j 必须同时满足：同任务、不同 trajectory、`j ∈ K`、`r(j) < r(i)`、`s(i,j) >= θ`。
4. 无候选则保留 i；有候选则仅删除 i，代表点按 `(-s(i,j), r(j), source_ordinal(j))` 确定，写直接见证 `(removed_id, retained_id, score, r_removed, r_retained)`。
5. 不递归删后继，不用已删点作为代表，不因删除重算 r，不删除剩余长度相等的点。所有原终点 r=0 必然保留。
6. 导出时恢复原 entry 顺序的子序列，避免因排序改变 runtime 并列 winner 的顺序。

P00 明确旁路剪枝并直接引用原 PKL，不用“足够高的阈值”近似不剪枝。每个阈值都从同一原库独立计算；不在上一档裁剪结果上继续剪。贪心代表集随阈值改变，不承诺相邻保留集嵌套、删点率单调或恰好命中特定比例。

### 4.1 分数计算与数值口径

复用 `WeightedScoreSumKnnStrategy → CacheStorage → InMemoryBackend`。离线枚举时扩大 `top_k` 以拿到同任务全部候选；评测仍为 top_k=1。按任务存储分数块，禁止为不同任务分配/计算无意义的全局稠密块。每个点都覆盖完整合法候选集，不用近似 NN 或 top-1 替代全部直接见证搜索。

保留 float32 的正式融合分数，删除比较为明确的 `>= θ`，不借 epsilon 扩大阈值。不同库形状的 L2 内核可能有浮点差异，既有 L10 诊断已出现约 1e-5 量级差异，因此不声称所有重新装载后的分数位相等。验证器以冻结 float32 分数块在保留列上的投影为主参照；按向量维度、float32 舍入与 σ 传播计算误差上界，float64 独立 oracle 只记录 max/p99 诊断。重载 top-1 必须等于实际 float32 全候选最大分数，且相对冻结参照的 regret 不超过该 query **实测**分数误差的两倍；不能用理论最坏上界替代这个 regret 门。阈值附近点按一个 float32 ULP 计数，删除比较仍不加 epsilon。具体模型与限制见 §12。

### 4.2 C1 必做：四源独立扫点，再机械冻结十点

G1 冻结的是下述选点算法，数值 θ 必须在 C1 实测保留量后得到。每个 `(suite, regime)` 独立选九阈值，不再把小库 θ 直接套给大库。所有库共用删点率覆盖目标：

| point | 目标删点率 |
|---|---:|
| P00 | 0%，直接使用原库 |
| P01 | 5% |
| P02 | 10% |
| P03 | 20% |
| P04 | 30% |
| P05 | 40% |
| P06 | 50% |
| P07 | 60% |
| P08 | 70% |
| P09 | 80% |

**outcome-blind 的含义**：选 θ 只读取已定成功源库、固定检索分数与剪枝保留量，不读取 eval journal、SR、成功执行长度、动作 RMSE 或“最好点”。Owner 已固定成功库口径；建库的 success 标签只用于 §2 来源验收，不是扫点器可以调优的轴。

候选与选择流程固定如下：

1. 从原 score blocks 计算每个非终点的 `m_i = max_j s(i,j)`，j 满足 §4 的同任务/跨轨迹/严格更短，但此处尚不要求 j 被贪心保留。无合法 j 的 i 不进入分位计算。m 只用于候选生成，实际删点率仍调用完整 §4 算法计算。
2. 初始 θ 集合为 m 的 65 个等间隔经验分位 `q=k/64, k=0..64`（numpy linear），合并固定锚点 `{0, 0.25, 0.5, 0.75, 0.85, 0.90, 0.95, 0.97, 0.98, 0.985, 0.987, 0.99, 0.995, 0.997, 0.998, 0.9985, 0.999, 1}`。转 float32、按数值去重；逐 θ 从原库独立剪枝，记录 `removed_count/N`、保留 ID digest、每任务保留量及见证摘要。
3. 做两轮确定性加密：按数值排序现有 θ，对每两个相邻 θ 取 float32 中点；若与端点相同则跳过；加入全部新中点并实测。最多约 329 个候选/源，分数矩阵只算一次。这里不对删点率作单调假设、不用二分反解、不依据 SR 挑加密区间。
4. 删除零剪枝候选（由 P00 独占）；相同保留 ID digest 仅保留最高 θ。相同删点数但不同保留集亦只保留最高 θ，保证所选横轴不同。按实际删点率升序排列，rate 相同的 tie 已消除。
5. 在有序候选中选择九个严格递增 rate 的点，依次对应上述九目标，最小化 `Σ_j abs(actual_rate_j − target_j)`（动态规划/等价精确解）。每点约束偏差 ≤5 个百分点，且 P01 实测删点率 >0、≤7.5%；目标未达不能报目标值。代价并列时优先实际删点率向量字典序较小者，再优先 θ 向量字典序较大者。
6. 无可行九点解、候选不足九个不同保留量或任一任务无保留点时，C1 报 `grid_not_representable`，附完整扫点表并停止发臂；不得用重复保留集、删减点数或悄悄放宽偏差消耗 rollout 预算。需要调整协议时返回计划修订，不能在看了 eval 结果后修补本次网格。
7. 保存每候选与每源完整扫点耗时、实际候选数、完成状态（中断不能写完成冻结），并将四源完整候选表 SHA、选中 θ 的十进制 round-trip 值/float32 bits、九个 keep digest、实际/目标删点率、检索/source digest、算法版本写 `grid_freeze.json`。四源全部通过才生成 **4×(1+9)=40** 臂，之后禁止换点；P00 无需 θ。

阈值仍是各 suite 正式融合分数，不是 cosine。跨规模比较采用近似对齐的**实测删点率**，同时展示实际偏差及绝对 retained entries。此次 G1 阶段不运行扫点程序或计算正式数值 θ；C1 的离线数据准备不包含任何闭环 rollout。

## 5. 剪枝 PKL、链与溯源

主方案为**物理导出保留 entry 的新 PKL**，使库字节数和索引规模可以实测。只保存候选 mask 而保留全库常驻内存不能作为物理压缩结果。

- 导出必须重新 `pickle.load` 服务节点的源文件，从**未经 backend/strategy 触碰的原始对象**按 ID 取子集；打分用的是独立副本，不从会回填 schedule/outcome、转 torch 的 runtime 对象反向导出。分阶段释放打分对象后再载原文件，避免同时常驻多份大数组。`LibraryStats` 在独立可转换副本上计算，不能污染待序列化源值。
- 顶层除 `entries`、`library_stats` 两个重建键外，所有原键（含未知扩展键）及其值原样复制；只新增 `cp1_d1_pure_cache_only: true`。源文件若已含该键或标为其他剪枝产物，拒绝将其当本实验完整原库。不得丢失 `model/schedule_id/pin_id/vector_dims/key_builder_type/checkpoint_id` 等来源信息。
- 保留 entry 的 ID、trajectory_id、step_idx、query_keys、action_chunk、intermediates、schedule、outcome、factors 值均保持源值；不得顺手删快照、压 dtype、截 action horizon 或重做 key。这些会另引压缩变量。
- `prev_ids/next_ids` 只保留**原本存在且两端都保留**的边；删除指向移除点的悬空边，不跨缺口连接、不拼接不同轨迹。原链、原 L/r 和源 ID 列表写 sidecar manifest，可从源 SHA 还原。
- 导出 PKL 标记为 `cp1_d1_pure_cache_only`（artifact 元数据及本实验 manifest），本实验启动器强制 §3 配置。该标记不是通用 backend 的强制兼容门：不得把这些片段库直接交给 d>1、follow_winner 或依赖链的判据使用。
- 原轨迹 step 不重编号；剪后不再具有完整 `0..L-1` 序列，L/r 只从源 manifest 读取。由于每条原终点都保留，轨迹 ID 个数不下降；压缩单位是 entry，不称“减少了示范轨迹数”。
- 使用 `LibraryStats.compute_from_entries` 对保留池重算 artifact 统计，不能留着全库计数。此操作不改变 §3 的固定检索归一化参数。factors 保留来源语义且本轮不消费，不宣称它们描述剪后拼接链。
- manifest 记录 source/output SHA256、字节数、source census、保留/删除 ID、每任务/轨迹 retained counts、原 L/r、直接见证、删边数、阈值、retrieval digest、实现版本及软件环境。浮点与数组按规定 dtype/shape/bytes 比较。
- 先写新目录临时文件，再原子发布；拒绝覆盖源文件、已有不同 digest 的产物或非空运行目录。输出根固定在新实验的 `data/cache_artifacts/` 对应大盘路径，原 PKL 只读。

离线验证必须检查：`kept ∪ removed = source`、两者无交集、每个 removed 恰有一个 retained 直接代表、所有代表满足 §4、每任务非空、全部源终点保留、保留 payload 数组等值、无悬空/新增边、装载后 entry 数正确。P00 的 source SHA 和数据内容不变。

## 6. 评测协议、配对与成本

### 6.1 矩阵与运行量

臂命名 `cache_prune_<suite>_<rit50|cs500_success>_<P00..P09>`。每 suite 一个 20-arm matrix，总 manifest 检查笛卡尔积**精确等于 40**，每个臂只能对应一个源 SHA、一个保留集、一个 YAML。拒绝 `cs500_all` 臂混入本协议。

沿用现有 LIBERO A-pool，每 task 50 个 init、10 tasks、**500 episodes/arm，共 20,000 episodes**。原库四个 P00 也同条件重跑，不能拿旧 RIT 阈值运行或旧 cache_size 结果代替。无新增 warmup rollout。技术 smoke 使用单独 run_id / 输出目录，不并入正式分母。

使用既有 π0.5 LIBERO teacher/checkpoint、norm stats、图像处理及 client `replan_steps=5`、`num_steps_wait=10`；启动记录实际模型、配置、LIBERO/robosuite 版本、节点、线程、worker 和随机种子设置。薄 runner 在真实 client 环境核对 `main.Args` 默认值与实际 worker 命令，确认没有其他覆盖，再把两个有效值及各 task 的 `max_steps` 写 launch manifest；当前 run_size_eval 未暴露 wait 覆盖参数，不虚构 CLI。有效值与上述冻结值不同即 preflight 失败。纯 cache 仍需 Stage 1 构 key；不能写成“整个模型完全不运行”。

### 6.2 init 来源与共同未见子集

冻结来源为 `exp/ablation_study/cache_size/config/apool_<suite>.yaml`，运行前重新读取实际 `.init` 文件核对 hash，拦截优先级更高的残留 `.pruned_init`。配对身份使用 `(suite, task_id, orig_init_state_idx)`，不是日志文件行号。

大库的 B-pool 与 A-pool 不相交已有记录；小 RIT 底座由老 LIBERO init 子集建库，**不能由此推导小库也与 A-pool 不相交**。实施前用 `exp/common/data/db/libero_cache/<suite>_init_map.json`、源 HDF5/初始化文件及实际状态数组核对两个来源。索引相同不等于跨池状态相同，必须绑定池 digest 与状态内容。

Owner 2026-09-11 对 G2 N2 裁定接受 L10 **集合层面证据**：紧凑 map 所声明的每任务 init 集合必须与实际 subset/full init 文件字节一致，覆盖完整子集且无重复，并与原 HDF5 逐任务数量/任务身份一致。manifest 的 `collection_rows` 使用 `identity_basis=declared_collection_init_set`，不填写不存在的逐轨迹绑定；此裁定仅用于 seen/unseen 集合，不授权按 episode 编号推断轨迹 init。

冻结 `eval_membership.json`，记录 A-pool 中相对两档**原始源库采集集合并集**的 seen/unseen 身份；成功筛选或剪枝不改变这个定义。主论文剪枝曲线使用两档共同未见子集，full 500 的曲线作为协议完整结果同时报告，seen 子集单列。2026-09-11 两 suite 的实际状态字节核验均得到 500/500 common-unseen（每任务 50 个），seen 为空；full 与主子集完全一致。旧 450 集推断已被实测否定。

40 臂仍各运行完整 500，不改变运行量；所有子集在看 eval outcome 前固定。任一任务共同未见子集为空或初始化来源无法证明时，相关“未见 init”主张不得发布，先修复来源证据；不能把缺失身份当 unseen。

### 6.3 runner 与数据完整性

新实验写薄包装 `run_prune_eval.py`：先校验本实验 manifest/保留集/检索模板，再调用现有 **`exp.ablation_study.cache_size.run_size_eval`** 的纯 cache 运行链，保持其 A-pool 和 FULL_HIT 门。通过结构化 argv 的 subprocess 调用，不复制 conductor 调度、snapshot、resume 代码。

不要误用 `exp.gate_threshold_pareto.run_gtp`：当前其 `JUDGE_TYPES` 只有 threshold/dispatch_surface，不接受 always_hit。若现有纯 cache runner 的性能不够，先使用已经支持的 server/worker 参数；本计划不顺手重构共用 runner。

正式运行前后均校验 launch manifest：40 臂身份、源/产物/YAML SHA、A-pool digest、retrieval digest、软件版本。resume 必须绑定同一 run fingerprint；任何输入变化创建新 run_id，不能往旧 journal 追加新实验。每臂使用相同任务/init 网格和完整 timeout 配置。

**CP1 服务节点绑定**：本轮 wire 的 `library_sha256` 为空是正常现状，不能声称从响应或 load_config ack 验证了库身份。每批在实际服务节点对唯一 preload 的 resolved path 实测 SHA256/bytes，并写 `node_preflight.json`；按 §9 启动专用新 server PID，启动配置直接使用该批 YAML。保存 server 完整日志和 bundle/yaml_id 装载事件，以 `Loaded <N> entries from <path>` 只出现一次、path 唯一、entry 数匹配验证单 fingerprint；重复 load_cache_config 命中 BackendPool 是合法行为，不按配置调用次数判错。结束后源节点重测 SHA/bytes 写 `node_postflight.json`，绑定 PID、process start、endpoint、arm、YAML SHA 和日志路径；本任务独占这些 server，不复用已有共享服务。缺节点侧证据、缺装载日志、路径/数量不符或前后 hash 变化均拒绝该批结果。

聚合复用 `exp.common.conductor_journal.load_accepted` 与 cache_size 的 FULL_HIT 检查，以 `(task_uid, accepted attempt)` 联结逐步证据：

- 每臂 500 个唯一身份，task 0..9 × init 0..49，配对键与 P00 完全相同；每个任务都进入分母。
- 同一 accepted attempt 的 step 不重复，日志从首个调用到末个调用完整；未知 winner、缺证据或结果/逐步身份冲突立即失败。旧 attempt 不能填补新 attempt 的证据。
- 每个 accepted episode 必须有且只有一条同 attempt 的 `_kind: client_timing` 行；其 `infers` 必须等于有效策略调用行数。当前 `main._run_episode` 的 step_idx 是扣除等待步后的环境决策位置，replan=5 时应为 `0,5,...,5×(infers−1)`，据此检查缺首行、缺中间行和缺尾行，不能只检查“至少一行且全部 FULL_HIT”。
- 普通任务失败 `success=false` 是有效 outcome，保留在 SR 分母；infra error、进程失败、未完成 episode 不冒充任务失败完成正式网格，也不静默删除。
- 对每个有效调用检查 FULL_HIT、winner 存在/同任务，并要求 `searched is True`；该字段已由 `__hit_meta__` 经 `_hit_row` 落盘，缺失或 false 均失败，不设置兼容兜底。

**提前退出/异常批次的补救**：一次批次出现缺集、无效提前退出或证据错误，标 `invalid` 并保留全部日志。修复运行条件后以同臂新 batch_id、全新 server 和空 journal 重跑全部 500 集，不对 terminal failed 执行原 journal resume。正式 run catalog 只可为该 arm 选择一个完整有效批次，整臂替换先前 invalid 批次；禁止逐 uid 挑取两批较好的 outcome。已有完整有效批次不得自动再跑或按 SR 替换。崩溃后也采用同一整批恢复规则，旧证据永不覆盖。

### 6.4 指标与统计

所有主比较为同 suite、同源库、Pj 对 P00，先完整性后统计：

| 指标 | 定义 / 报告方式 |
|---|---|
| SR | 每任务成功数/该预冻结子集任务集数，再对十任务等权平均；全 500 同时报告总体成功数 |
| 配对 ΔSR | 同一 init 的 success 差，按任务等权；全部 36 个非基线对比完整给出 |
| 库大小 | retained entries、删点率、PKL 实测 bytes、加载后 RSS；只删点百分比不代称字节压缩率 |
| 检索成本 | 固定查询 bank 上真实 top-1 SearchStrategy p50/p95、原始样本；同机线程/预热/计时方法一致 |
| 线上成本 | 同 accepted attempt 的 `client_timing.infer_ms/infers`：主聚合为 `Σinfer_ms/Σinfers`，另列每集调用数和 journal `duration_s`；包含网络/服务端排队，非纯服务端时延 |
| 完成长度 | `control_steps = client_timing.steps − launch.num_steps_wait`；Pj/P00 共同成功 init 上给配对差及共同成功样本数；与 `client_timing.infers` 分开 |
| 机制诊断 | winner 原成功轨迹 r、winner trajectory 切换、各 task 保留量与 SR；任何失败来源 winner 都是来源/完整性错误 |

`Journal.record` 根本不写 `n_steps`；`EpisodeResult.n_steps` 仅在内存中，本计划不以其为持久化数据源。`load_accepted` 选定 `(uid, attempt)` 后，原始 journal 只用于补取 `duration_s/error/run_id`；控制步数和调用次数从同 producer/attempt 的唯一 client_timing 行获取，不重新选择 outcome。

`client_timing.steps` 是成功执行的 `env.step` 次数，包含最前面的 wait dummy 步，也包含成功触发 done 的最后一步。完整正常 episode 直接扣除 manifest 的 W=10，不用 loop counter 的 t，也不使用 `5×infers` 近似。以下校验先于长度统计：

- steps/infers 为整数，steps≥W，infers>0；否则标记无效/提前异常，不能 clamp 负值到 0。
- `1≤control_steps≤max_steps`，`ceil(control_steps/replan_steps)==infers`；最后一块不足五步是合法成功，不填充到整块。
- LIBERO 此 evaluator 的正常未成功结束应跑至 max_steps；`success=false` 且 control_steps<max_steps 视为提前退出，查 worker 异常日志并将该批标为 incomplete，不能把被 broad except 捕获的异常当短失败/短成功。wait 未完成、infer 已返回但未执行第一步就异常等边界同样不得进入完成长度统计。
- 保存原始 steps/infers/W/replan_steps 和派生 control_steps，可逐条复算。episode_summary 的 `num_steps` 是策略调用数，禁止替代。

线上主指标所需原始数据路径为 `data/runs/<run_id>/<batch_id>/per_step.jsonl` 的 client_timing 行及同目录 journal；记录每集分子、分母和比值。只有 episode 汇总耗时，不能由此构造“逐调用 p95”；episode 平均耗时的分位数须明确标注。正式模板关闭 SystemTimer；不依赖未设置 `output_csv_dir` 的控制台 timer 汇总发布服务端分段成本。额外服务端计时若以后需要，另定持久化协议和运行，不进入本轮 40 臂主读数。

离线 latency bank：每份源库按 task 分层固定抽 100 个 query，seed=20260911，所有十点复用；先各点预热，再以固定种子随机化顺序跑 7 轮，记录每次真实 search。来源是库内 query 的检索微基准，不能用它推断未见状态 SR 或端到端提速。使用 CPU 4 线程与既有离线分析对齐；另记实际硬件，不把跨机器比值混在一条曲线。

SR/ΔSR 的 95% 区间采用两层配对 bootstrap：重采 task，再在所选 task 内重采 init；十个点共享每次重采样索引，10,000 次，seed=20260911。这是十任务上的描述性不确定性，不宣称四条扫描自动构成确认性显著性检验。主图展示全曲线；若事后挑最佳点，必须写 exploratory，不把点置信区间当选择校正后的保证。

完成长度只比较共同成功子集，并同时展示 SR、失败/新成功数；不能仅看过滤失败后的平均长度就宣称执行效率提高。smoke 核对 client_timing 的 wait/动作步口径；缺失长度数据视为观测链未交付并阻止正式分析，不能从 journal 虚构 n_steps 或用调用数乘 5 填充。

## 7. 文件、接口与集成范围

以下程序/配置/测试属于 G1 R2 批准的实现交付；数值网格和生产运行数据由所交付工具在对应数据节点生成。

| 路径 | 责任 |
|---|---|
| `exp/ablation_study/cache_prune/__init__.py` | 子实验包声明 |
| `.../common.py`、`prepare_membership.py` | manifest/文件指纹等共用辅助、源 init 状态与 A-pool 的共同未见身份绑定 |
| `.../prune_library.py` | 源库审计、任务分数块、确定性剪枝、新 PKL 与直接见证导出 |
| `.../verify_prune.py` | manifest/条目/链/终点/真实装载与检索验证 |
| `.../select_prune_grid.py` | C1 outcome-blind 扫点、两轮加密、唯一保留集筛选、九目标精确匹配与 grid_freeze |
| `.../emit_prune_arms.py` | 固定检索模板 + 十点 grid → 四十个 YAML、两份 suite matrix 与总 freeze manifest |
| `.../run_prune_eval.py` | 实验专属 preflight/resume/postflight + 调用既有纯 cache runner |
| `.../config/search_<suite>.yaml`、`grid.yaml` | 可追踪检索快照、九个目标删点率/候选算法/偏差门与 P00 旁路；数值 θ 从四源 grid_freeze 读取 |
| `.../config/inputs.example.json`、`launch_spec.example.json` | 四源输入与节点运行参数示例；生产路径在冻结前填写 |
| `.../config/arms/`、`matrix_<suite>.yaml` | 40 个正式 YAML 与 20+20 臂矩阵 |
| `.../data/` | source/grid_freeze/freeze/membership manifests、候选表、score blocks、keep IDs、直接见证；`runs/<run_id>/<batch_id>/` 保存节点 hash/加载证据、进程与内存记录、journal、per-step、latency 原始样本 |
| `.../data/cache_artifacts/` | 剪枝 PKL，原库 P00 只引用源文件 |
| `.../analysis/analyze_prune.py` | accepted ledger、配对 SR/长度、bootstrap 与机器可读摘要 |
| `.../analysis/plot_prune.py` | 四条压缩–SR 曲线、成本曲线、共同成功长度差与 per-task 图 |
| `.../analysis/benchmark_prune.py` | §6.4 已批准的固定查询 bank / CPU 四线程 / 七轮随机顺序真实检索微基准入口 |
| `tests/ablation_study/cache_prune/test_cache_prune_*.py` | §8 测试，文件 basename 全局唯一 |
| `tests/ablation_study/cache_prune/__init__.py` | 测试子包标记；G1 APPROVED 后与测试代码一起创建 |
| `exp/ablation_study/README.md` | 新子实验登记，阶段与分析入口 |
| `docs/experiments/artifact_layout.md`、`docs/README.md` | 既有 family 的子实验清单同步；不修改目录或架构规则 |
| `logs/cache_prune_plan.log.md`、`logs/README.md` | 本计划、评审记录与状态同步 |

主要 Python 接口计划（名字与参数语义在 G1 冻结）：

```python
audit_source(path, expected_census) -> SourceManifest
compute_task_scores(entries, retrieval_config, out_dir) -> ScoreManifest
select_retained(source_rows, task_scores, threshold: float | None) -> PruneSelection
select_grid(source_manifest, score_manifest, grid_spec) -> GridFreeze
export_pruned(source_path, selection, out_dir) -> ArtifactManifest
verify_pruned(source_manifest, artifact_manifest, retrieval_config) -> VerificationReport
emit_arms(source_manifests, artifact_manifests, templates, grid, out_dir) -> FreezeManifest
validate_run(freeze_manifest, suite, apool_record, run_dir) -> LaunchManifest
analyze_run(freeze_manifest, journals, per_step, membership) -> AnalysisResult
```

`SourceManifest` 明确原始序号/原长度/源任务；`PruneSelection` 明确保留与删除 ID 及直接见证；`ArtifactManifest` 明确真实输出 SHA/字节和 parent source；`GridFreeze` 包含 §4.2 的候选表和选点证据；`FreezeManifest` 绑定完整四十臂、模板、四源 grid、membership 和版本；`LaunchManifest` 包含 W/replan、批内唯一 arm、服务 PID/endpoint、节点 hash、RAM 预算和日志。数据对象采用本实验内的 typed dataclass/JSON schema；不扩展 `src/` 接口。

复用 `LibraryStats.compute_from_entries`、正式 SearchStrategy/storage/backend、cache_size 的 `load_apool_digest`/FULL_HIT 检查、common journal parser 与现有 runner。已有 `prune_feasibility.py` 仅为算法/诊断参照，不能 import 它的 argparse/main 或把输出的 keep_ids 不经来源核对直接上线。其留一轨迹分析不进入正式 40-arm 策略。

原 RIT/缓存大小实验文件、原 PKL、源 HDF5、共享 serving/model/检索代码均不修改；无 gitignore 改动。本轮不编辑论文正文或替换已有实验结论。

## 8. 验证策略与门

测试围绕会改变实验结论的边界，不只对照实现本身：

1. **剪枝规则与成功库边界**：人工小图和独立穷举候选 oracle；同轨迹、跨 task、相同剩余、分数恰等阈值、已删代表不能转递、分叉/断裂输入拒绝；证明每次删除有最终仍保留的直接见证，终点不会消失。把一条失败终点混入源库、成功名单缺条或 HDF5/显式标签冲突时必须在剪枝前拒绝；老 entry.outcome=None 但来源标签已验证成功时允许且不改原对象。
2. **确定性与基线**：同 manifest 重跑 ID/配对一致；保留源顺序；P00 原文件 SHA 不变；阈值各自从源库计算，不假设嵌套和固定删点率。
3. **序列化集成**：真实 CacheEntry 与实际 backend 装载；打分对象被 backend 回填/转 torch 后仍从独立原始 pickle 导出，payload/key 的类型/dtype/shape/bytes 和缺失属性状态不变；顶层已知/未知键保留、统计匹配、删边无悬空也无桥接、step 不重编号。父本 SHA 与 P00 服务文件不一致、误指源文件、输出冲突、错 SHA、少 task 均拒绝。
4. **检索等价性**：通过真实 factory/strategy 验证模板字段及 pruned top-1；与独立遍历候选参考对拍，分别处理高 margin 与近并列的数值边界，记录误差而非断言虚假的全量位等价。
5. **40-arm 配置与选点**：笛卡尔积精确匹配；top_k、depth、task_scope、gate/judge、write、权重及 μ/σ 任一被修改都被专属 preflight 拒绝；错误沿用 run_gtp 必须在测试暴露。选点算法与小规模穷举参考对拍，覆盖非单调 rate、相同 keep digest、同 rate 不同 mask、DP tie、候选不足及目标不可达；改变 eval outcome/RMSE 文件不得影响 θ，四源各有独立 grid digest。
6. **生产日志语义**：同臂 invalid 批次全量重跑、完整批次整臂替换且不能拼接挑 outcome；真实 journal schema、accepted attempt、重复/缺失 step、过期 attempt、未知 winner、漏 task、变更 source/APool 后 resume、infra error、partial journal、searched 缺失/false 均不能产生有效正式结果。CP1 wire 无 SHA 时必须有节点 hash/加载日志；错误 PID、错 path/count、postflight 变更、同批试图加载第二个库及超 RAM 预算均拒绝。
7. **长度与配对统计**：W=10、steps=23、infers=3、replan=5 得 control_steps=13，覆盖最后不足一块、done 在整块末步、wait 未完成、等待后/首次 env.step 前异常、未成功但提前 break、缺 client_timing、错误 attempt，禁止读取不存在的 journal n_steps。小型可手算 ledger 验证 SR/ΔSR/任务等权；同一 bootstrap 索引跨臂复用；共同成功为空记不可估，不填 0；改变 eval outcome 不能影响预冻结 seen/unseen membership。
8. **手工/集成 smoke**：四份源库 P00 与最激进 P09，各覆盖全部十任务至少一集；验证所有有效调用 FULL_HIT、worker 配置和产物加载内存。smoke 输出不进入正式评测；发现行为异常先修复，不用放宽纯 cache 门启动大跑。

实施期可运行针对性测试作为自查。G2 后按 `WORKING_AGREEMENT.md §2.7` / Execution §6 做正式 `uv run pytest` 并留日志；若扩大实现触及 staged inference 路径，须相应 API 测试和重新说明范围。本计划预计 `src/` 零变更。

## 9. 实施步骤、资源与风险

| 阶段 | 交付 / 完成条件 |
|---|---|
| 本次 Understand / Plan | 本计划、索引和 cache_prune 目录骨架；无剪枝库生成或 rollout |
| G1 | 独立 Review Authority 审计划；APPROVED 后完成正文整理再进入 Code |
| C1 来源/模板/网格 | 完成必要离线工具后，四源 census/hash、检索快照、eval membership；按 §4.2 outcome-blind 扫点并冻结四份九阈值，无 rollout |
| C2 离线工具 | 剪枝/导出/验证器与边界测试；按任务分块，物理 PKL 可由 manifest 重建 |
| C3 运行与分析工具 | 四十臂 emitter、薄 runner、统计和绘图；synthetic 集成及日志负例通过 |
| G2 / Verify | 独立代码评审与正式测试；不把本计划的 G1 请求当已获 rollout 放行 |
| 实验启动 | 后续授权执行时先四源 P00/P09 smoke，再运行冻结的 40×500；每臂过完整性门 |
| 分析 | 同一权威结果表生成全部图与报告，汇报完整网格及未见 init 子集 |

大盘根建议 weilandserver `/data/openpi/ablation_study/cache_prune/`，工作树 `data/` 对应软链在实施期创建。逐源串行构建，复用只读内存数组，不并行装入四份大库。分数块内存是 `4 × max_task_entries²` 字节；源 key/payload、backend 矩阵及复制开销另外计入峰值 RSS，不能仅用分数块估算全部内存。

最多 36 个新 PKL。以裁定后四份源文件总字节 ×9 为输出规模参考，再加 score/日志/临时输出余量；P00 不复制整库。不要在小磁盘工作树下生成几十 GB。资源规则如下：

1. **每批仅一个 arm**：`run_prune_eval --arms <one_arm>` 验证后只向 run_size_eval 传这一个 arm 和本批专用 `--servers <host:port>`，500 episodes 可由该进程的多个 worker 服务。一个 server 进程生命周期只允许一个 preload fingerprint；启动 cache 配置就是本批 YAML，不能先载其他默认库再换入。四十臂对应四十个单臂批次，重试沿用该臂身份并记录新的进程 generation。
2. **批间重启**：本批完成、节点 postflight 和日志落盘后，关闭本任务专用 server，确认 PID 退出才启动下一批；不能通过 load_config/换 bundle 假装释放内存，也不能在生产调用 `BackendPool.reset_for_tests`。不停止其他实验进程。
3. **预算**：每 server 进程峰值 RSS 上限 64 GiB，包含模型、原 PKL、加载暂存、索引与 worker 服务开销。每个节点预留至少 16 GiB，任务启动时固定 `B_host = max(0, min(0.75×MemTotal, MemAvailable_before_task−16 GiB))`；本任务所有同时存活进程（含 client workers）预算总和不得超过 B_host。其他任务变化使可用余量不足时暂停发新批；GPU 显存另由 smoke 实测约束，不以主存预算代替。
4. **首批预检与实测**：每源 P00 预加载在独立短生命周期进程测 peak RSS；每个候选批按至少 `max(1.25×对应源P00实测峰值, 2×该臂PKL字节)` 加 client 预算排程，并检查小于单进程/节点上限。正式批持续记录 RSS/worker 数/线程；触发上限先停止该批并记 infra failure，禁止删日志后当已跑完。
5. manifest 必有 batch_id、唯一 arm、argv、server/node/PID/start time、worker PID 集、节点 total/available/B_host、估算与实测 peak RSS、前后 SHA/bytes、server log 与 journal/per_step 路径；合并器验证四十臂全部覆盖，不能把重复启动当新增实验点。

跨节点或同节点同时运行多个独立单臂批只在上述预算内允许；未测之前不预估 GPU 小时或宣称删点同比加速。

| 风险 | 处置 |
|---|---|
| “500/50”名义值混淆、拿错 S3 | 四源路径/census/SHA 与 regime 显式绑定，表内报告实际轨迹数 |
| 大库失败尾部成为短剩余代表 | Owner 已裁定 S6/success；来源 ID/标签验收拒绝失败轨迹，P00 和所有删点分母同为成功库 |
| 换库同时换 normalizer/命中阈值 | 固定六组 μ/σ、模态权重及 always_hit；deep-diff 和 digest 阻止漂移 |
| 删点误变为新示范/断链回退 | d1-only、原 step/r、只删失效边不跨缺口；终点和每任务覆盖验证 |
| 小库 train/eval init 重合 | 逐源身份审计；冻结共同 unseen 子集并同时报告全 500 |
| 同轨迹或非保留见证造成虚假覆盖 | 独立 oracle、逐删除直接见证复核；离线不使用 eval outcome |
| 近阈值浮点差异 | 固定源分数输入、边界诊断、真实子库复核，禁止调容差“过门” |
| 只汇报最快/最好点 | 十点完整曲线、四基线重跑、选择性结论标 exploratory |
| 任务失败/进程失败/缺日志混为一谈 | accepted attempt 与逐步证据严格对齐，infra failure 阻止正式发布 |
| 长失败 episode 被丢掉后假装变快 | SR 与共同成功配对步数同时报告；实际环境步与调用数分开 |

## 10. 实施状态与 G2 交接

G1 R2 已由独立 Review Authority 于 2026-09-11 18:50 CDT 批准；三条非阻塞建议已并入 §4.2/§6.3/§8/§9。按 Execution §3.1 完成正文整理并删除全部 G1 Review Log。当前会话保持 Execution，只推进代码交付与 advisory 自查到 G2，不代为审批。

Owner 的不自行 git add 裁定继续有效，故 Post-G1 和 G2 的暂存步骤均由该裁定覆盖；本会话不暂存、不提交。正式 rollout 需后续授权，G2 后的 Verify 与本阶段自查分开记录。

### 10.1 首轮 G2 交付记录（历史，R1 修订见 §12）

**Plan conformance: code fully follows the approved plan.** 本声明限于 §7 批准的程序/配置/测试交付及缺证据即拒绝的实验门，不声称 C1 生产数据已齐备。§9 中四源正式 source/grid、36 个物理子库、八个真实 smoke 与 20,000 rollout 均未完成；L10 原轨迹 init 绑定的实际缺口见 §11.1，未通过放宽协议消除。

| 已批准责任 | 实现与自查证据 |
|---|---|
| §2 成功来源、四源身份 | `expected_source/audit_source` 校验名单/census、真实 HDF5 success/长度、CP1 原链、已登记小库 SHA；兼容 L10 紧凑 map 做成功证明，不伪造 init 绑定 |
| §3/§4 固定检索、直接见证 | 两份 RIT 检索快照与原配置 provenance；真实 backend/task score blocks、确定性短剩余剪枝；独立穷举 oracle 与同轨迹/等剩余/已删代表等边界 |
| §4.2 四源机械选点 | 65 分位+18 锚点+两轮加密、保留集/删点数去重、精确目标匹配与 tie；候选和总耗时落盘，无合法候选/目标不可达明确报告后停止 |
| §5 原始 pickle 物理导出 | P00 原路径/原字节；独立 raw reload 保留 payload/未知顶层键、只删失效原边；真实 LibraryStats/backend 与 float64 独立检索参考、近阈值数量诊断 |
| §6.1/§6.2 矩阵与 init 身份 | 四源×十点 emitter、两 suite matrix、冻结配置与 A-pool；HDF5/map/实际 init 字节检查；缺完整 init 绑定明确拒绝，旧 L10 数据没有被认证为共同未见 |
| §6.3/§9 单臂生命周期 | 专用 server、每用户每节点串行锁、PID/start_time 所有权、唯一 Loaded 事件、前后 source/model/code SHA、原 worker launch metadata；完整无效批次保留后整臂重跑，已完成有效批次不得再跑 |
| §9 资源预算 | P00 preload probe、server 64 GiB 上限与节点预算、VmHWM、实际 worker/thread、NVML 本任务分配；超限或遥测缺失拒绝；task/GPU 是采样峰值，见 §11.4 |
| §6.4 统计与图 | 公共 `analyze_run` 也校验整个 run catalog；accepted attempt/FULL_HIT/searched/winner/timing 严格对齐；steps−10、共同成功配对、两层共享 bootstrap、全曲线 8 个 PNG/PDF |
| §6.4 检索微基准 | 每源 100 query、十点、随机七轮、CPU 四线程；真实 7,000 次计时搜索与逐次原始记录校验，不把库内 query 当闭环 SR |
| §7/§8 交接 | 九个 CLI、两个 JSON 输入示例、§11 完整命令；模块/公共接口 docstring、格式/编译检查；测试文件独立于 reviewer 空间 |

实现全部位于新实验与对应测试目录。`src/`、共享 serving、原 RIT/cache_size 实现、原数据均未修改；同一工作树既有其他实验脏文件不属于本次改动。没有代码范围偏离、没有自行改变源库或评测协议。实际 source/grid 数值不由 synthetic 数据冒充。

### 10.2 首轮 Advisory 自查记录与限制（历史）

- 第一轮 48 项、第二轮 49 项通过；第二轮保留日志 `exp/ablation_study/cache_prune/data/advisory/code_selfcheck_20260911.log`，历史记录不覆盖。
- 最终 `.venv/bin/python -m pytest tests/ablation_study/cache_prune -q`：**73 passed / 1 upstream pynvml deprecation warning，44.64s**。包含四份 synthetic 源库 → 40 次物理导出/真实 backend 验证 → 40 YAML → 20,000 条合成 accepted episode → 配对统计/8 张图，以及 7,000 次真实检索微基准。
- runner 集成实际创建短生命 loopback listener/子进程，验证 probe/smoke/失败批次/新完整批次/有效结果拒绝重跑；11 类节点证据负例覆盖 PID、路径、重复装库、数量、RSS、GPU、worker pool、模型、代码和退出状态。真实模型、LIBERO/GPU 加载在此测试替换，不能用测试值证明生产显存或闭环可行。测试为允许本机 socket 使用了已获准的 sandbox escalation，没有启动生产服务。
- 真实只读小库审计：Spatial **49 trajectories / 1,018 entries**，L10 **50 / 2,640**，SHA 与 §2 本地字节一致；分别验证 52/53 份 provenance 文件身份。L10 task scene 前缀与紧凑 map 已覆盖；init 对应关系仍缺失，缺口明确拦截。大库实物本轮未读。
- Ruff `--select F`、Ruff format、Python compileall、九个 CLI `--help` 与 JSON 示例解析通过；最终代码/config/test 文件 SHA 清单保存在 advisory 目录。

最终日志：[`code_selfcheck_g2_20260911.log`](../exp/ablation_study/cache_prune/data/advisory/code_selfcheck_g2_20260911.log)；文件清单：[`code_inventory_g2_20260911.json`](../exp/ablation_study/cache_prune/data/advisory/code_inventory_g2_20260911.json)。这些是 Code 阶段自查，**不是** G2 后 §6 Verify，不替代独立审查或真实八组 smoke。

### 10.3 Gate 交接

首轮 G2 于 2026-09-11 20:26 CDT 返回 NEEDS REVISION；当前 §12 修订完成：79 项 advisory 测试通过，四源九点网格、四份 P05 全查询验证及两套真实 membership 齐备，现申请独立 G2 复审。Review 可直接读未跟踪的新文件；因 owner 禁止未经指示 `git add`，此变更集未暂存、未提交，也没有混入其他实验的既有改动。G2 Reviewer 已建立 Review Log；该段按原文保留，执行方不自审、不代发 APPROVED。

## 11. 数据节点复现入口（交接命令，尚未执行生产运行）

所有命令从实际服务节点的仓库根执行，使用该节点 `.venv/bin/python`；路径中的 `/data/openpi/ablation_study/cache_prune` 为新实验大盘目录。源 PKL、HDF5、原 init map 均只读。四源分别串行准备；发生错误即停，不能跳过失败的来源或 membership。以下命令是交接说明，不代表已经生成生产产物，也不构成 rollout 授权。

### 11.1 当前数据前置条件

2026-09-11 实读发现 L10 历史 init map 只有 task/prompt/subset_idx/orig_init_state_idx，没有 trajectory_id/h5_path；HDF5 没有 init 身份，且重试使 episode 编号跨 task 重复。同一事实已载于 `logs/tracer_phase6_projection_training.log.md` §B2 和 `exp/data_authority/records/dispatch_surface__libero_10__init_pools.json`。因此不能从文件序号、采集顺序或 step0 robot_state 编造轨迹与 init 的对应关系。

Owner 在 G2 R1 修订会话已明确接受集合层面证据。`audit_source` 仍按确切 HDF5 stem/task/num_steps/success 验证成功来源；`prepare_membership` 新增紧凑 map 分支，逐 init 验证 subset/full 状态字节、完整子集覆盖、H5 task census 和 source ID 覆盖，再冻结每任务采集集合。不填写逐轨迹 init 绑定。状态字节不等、索引重复/遗漏、任务或 H5 census 不符均拒绝。这一分支支持 L10 两档，实际数据验收结果记录于 §12；不得把“接受证据类型”当作“任意 map 自动通过”。

大库实际 PKL/原采集 H5 在 weilandserver；本轮正在该节点直接生成 source/grid/P05 验证证据，见 §12。A-pool YAML 含服务节点绝对路径，必须与真实 client 读取路径一致；迁移到其他节点时在新实验数据目录生成同内容池的新路径记录，并在冻结前重新验证，不改旧实验记录。旧 L10 map 未跟踪于 git，部署清单必须显式包括它及对应 H5/init 文件。

### 11.2 四源审计、打分与数值网格

先通过 11.1 的集合身份验收，再在同一服务节点依次执行。P00 直接使用这里的实际源路径，不能在另一副本上剪枝后换 P00。

```bash
set -euo pipefail
prune_root=/data/openpi/ablation_study/cache_prune
for suite in libero_spatial libero_10; do
  for regime in rit50 cs500_success; do
    if [ "$regime" = rit50 ]; then
      prune_source="exp/common/data/cache_artifacts/$suite/cp1_spatial_pool_16.pkl"
    else
      prune_source="/data/openpi/ablation_study/cache_size/artifacts/cache_size_${suite}_success_S6.pkl"
    fi
    .venv/bin/python -m exp.ablation_study.cache_prune.prune_library \
      --suite "$suite" --regime "$regime" --source "$prune_source" \
      --repository "$PWD" --out "$prune_root/$suite/$regime"
    .venv/bin/python -m exp.ablation_study.cache_prune.select_prune_grid \
      --source-manifest "$prune_root/$suite/$regime/source.json" \
      --score-manifest "$prune_root/$suite/$regime/scores/scores.json" \
      --out "$prune_root/$suite/$regime/grid"
  done
done
```

每个 `grid/grid_freeze.json` 必须 `status=frozen`；`grid_not_representable` 会保留候选/耗时记录但阻止发臂。输出采用新目录且不覆盖。中断 source/score 目录不能伪装完整产物复用；保留诊断并使用新的输出根，随后同步 inputs 路径。

### 11.3 membership 与 40 臂

```bash
for suite in libero_spatial libero_10; do
  .venv/bin/python -m exp.ablation_study.cache_prune.prepare_membership \
    --suite "$suite" --repository "$PWD" \
    --sources "$prune_root/$suite/rit50/source.json" "$prune_root/$suite/cs500_success/source.json" \
    --apool-record "exp/ablation_study/cache_size/config/apool_$suite.yaml" \
    --large-collection "/data/openpi/ablation_study/cache_size/collect_h5/$suite" \
    --difference-pool "exp/common/data/db_init/libero/$suite" \
    --out "$prune_root/$suite/eval_membership.json"
done
.venv/bin/python -m exp.ablation_study.cache_prune.emit_prune_arms \
  --inputs exp/ablation_study/cache_prune/config/inputs.example.json \
  --artifacts-root "$prune_root/cache_artifacts" --out "$prune_root/config"
```

`inputs.example.json` 给出四 source/score/grid 与两 membership 的完整参数格式，使用其他目录时复制到新实验数据目录并替换绝对路径。emitter 重新检查四源完整笛卡尔积、选点和 membership，逐臂导出与真实 backend 验证，最终原子发布 `config/arms/*.yaml`、两套 matrix、`config/freeze.json`。已有 artifact 仅在 manifest/指纹/验证证据全部一致时可复用；正式 config 输出不可覆盖。

### 11.4 probe、smoke 与正式批次（另获启动授权后）

复制 `config/launch_spec.example.json` 到 `$prune_root/launch_spec.json`，替换三个 `REPLACE` 字段。`server_python` 必须等于运行本程序的解释器绝对路径；checkpoint 必须是实际 π0.5 LIBERO 权重及 norm_stats 目录；conda_env 是该节点既有 LIBERO 环境名或绝对路径。示例中的 workers=4、client budget=16 GiB、GPU budget=40 GiB 只是填写示例，实际配置在首 probe 前按节点容量固定；GPU budget 是本任务所有 GPU 分配之和上限，不是设备容量声明。端口必须空闲；runner 只终止自己记录的 PID/start_time。禁止借用正在服务其他实验的 server。

以下 Python 调用与 CLI 相同，便于准确接续随机 batch_id，不猜测 probe 路径。统一 run 目录，先四个 P00 preload probe，再八个 P00/P09 smoke；run catalog 会区分三种模式，probe/smoke 不进入正式 500 集分母。

```python
from pathlib import Path
from exp.ablation_study.cache_prune.common import read_json
from exp.ablation_study.cache_prune.run_prune_eval import run_batch, accepted_batches

root = Path('/data/openpi/ablation_study/cache_prune')
freeze = read_json(root / 'config/freeze.json')
spec = read_json(root / 'launch_spec.json')
run = root / 'runs/run_001'
probes = {}
for arm in freeze['arms']:
    if arm['point'] == 'P00':
        result = run_batch(freeze, arm['arm'], spec, run, probe=True)
        assert result['status'] == 'valid', result
        probes[arm['source_digest']] = Path(result['node_postflight']['path']).with_name('completion.json')
for arm in freeze['arms']:
    if arm['point'] in ('P00', 'P09'):
        result = run_batch(freeze, arm['arm'], spec, run, smoke=True,
                           p00_probe=probes[arm['source_digest']])
        assert result['status'] == 'valid', result

# Execute only after all smoke batches have passed and rollout is authorized.
for arm in freeze['arms']:
    if arm['arm'] in accepted_batches(run, freeze):
        continue
    result = run_batch(freeze, arm['arm'], spec, run,
                       p00_probe=probes[arm['source_digest']])
    assert result['status'] == 'valid', result
```

单臂 CLI 示例：

```bash
.venv/bin/python -m exp.ablation_study.cache_prune.run_prune_eval \
  --freeze "$prune_root/config/freeze.json" \
  --arm cache_prune_libero_spatial_rit50_P00 \
  --launch-spec "$prune_root/launch_spec.json" \
  --run-dir "$prune_root/runs/run_001" --probe
```

smoke 增加 `--smoke --p00-probe <真实有效 completion.json>`；正式臂保留 `--p00-probe`、去掉两种模式 flag。任一无效/提前退出批次都保留完整证据，修复后使用同臂新 batch 跑全部 500 集；不编辑 accepted 结果、删失败 episode 或拼接多个批次。恢复时从有效 probe completion 中恢复对应 source 的路径，验证器会检查模型/节点/来源；已完成的正式有效臂不重跑。若代码、模型、worker 配置或冻结输入改变，使用新 run_id 并重新 probe/smoke，不混算。

每批保存 `node_preflight.json`、`launch.json`、`server.log`、`eval.log`、`resources.jsonl`、`journal.jsonl`、`per_step.jsonl` 及原 runner 的 `.launch.json`、`node_postflight.json`、`completion.json`。server peak RSS 包含 Linux VmHWM；task RSS 与 GPU 显存仍每 0.5 秒测量；日志每 2 秒、进程集合变化或新峰值时落一行，验证器流式读取，保留全部已观测峰值。不能声称采样捕获了任意短瞬态。预算/版本/线程/Loaded 唯一库事件与退出证据先过门，才接受逐集 FULL_HIT 和真实控制步数。

### 11.5 全曲线分析与检索微基准

```bash
.venv/bin/python -m exp.ablation_study.cache_prune.analysis.analyze_prune \
  --freeze "$prune_root/config/freeze.json" --run-dir "$prune_root/runs/run_001" \
  --out "$prune_root/analysis/run_001.json"
for suite in libero_spatial libero_10; do
  for regime in rit50 cs500_success; do
    .venv/bin/python -m exp.ablation_study.cache_prune.analysis.benchmark_prune \
      --freeze "$prune_root/config/freeze.json" --suite "$suite" --regime "$regime" \
      --out "$prune_root/analysis/latency/$suite/$regime"
  done
done
.venv/bin/python -m exp.ablation_study.cache_prune.analysis.plot_prune \
  --analysis "$prune_root/analysis/run_001.json" \
  --latency "$prune_root"/analysis/latency/*/*/latency.json \
  --out "$prune_root/analysis/figures_run_001"
```

`analyze_run` 公共 Python 入口与 CLI 都要求同一 run catalog 的 40 个有效完整批次；不能只给合成的裸 journal 绕过节点证据。每源微基准共 7,000 个计时样本（100 query × 10 points × 7 rounds），四份全部提供后才画微基准成本；省略 `--latency` 时不伪造缺失的搜索延迟。线上成本始终来自 client_timing 的总 infer_ms / 总 infers，完成长度采用 steps−10，主图和机器表同时保留 full/common-unseen/seen 的定义与共同成功样本量。

## 12. G2 R1 修订与真实数据证据

### 12.1 数值验证协议

删除固定 `5e-5` 门，主比较为每个 query 对全部保留 candidate 的 `abs(loaded_float32 − frozen_float32)`。原 mask 仍由冻结 score blocks 精确决定，未改剪枝阈值或选点。报告 `score_reference=frozen_float32_v1`；emitter 拒绝旧版 verification，不能复用旧 oracle 硬门报告。

误差预算采用串行归约的保守模型：`u=2^-24`，`n=D+8`，`gamma=n*u/(1−n*u)`；两次 cosine 计算的原始差上界取 `4*gamma/(1−gamma)`，两次 L2 计算取 `2*gamma/(1−gamma)*distance`，再经 zscore–tanh 的 `1/(2*sigma)` Lipschitz 系数与权重传播，并计入两条标量归一化路径的舍入项。假设有限正常 float32 运算及 tanh 绝对误差不超过 4u；不声称对任意硬件/非正常值都是严格数学认证。n*u 过大或非有限输入拒绝。

**理论界可能很保守**：本次 Spatial 的界约 0.285、L10 约 0.507；四源实际最大误差为 0 / 0 / 1.0632e-4 / 0（依次 Spatial 小/大、L10 小/大）。该理论界不是允许 top-1 出错的阈值；`winner.score == actual.max` 仍为精确硬门，`regret ≤ nextafter(2*measured_error,+∞)` 使用每 query 实测误差。因此实测误差为 0 时必须选冻结参照最大值；高 margin 选错不能用保守理论界通过。报告另给出 `max_top1_regret`、`near_tie_queries`。float64 oracle 的 max 与 **per-query 最大误差的 p99** 单列 `float64_diagnostic`，不参与接受判定，也不参与剪枝。

针对性测试使用 32,768 维、μ=0.978/σ=0.007，复现大于旧 5e-5 的 float64 差异而重载验证通过；故意偏置 runtime 分数和返回非最大 winner 均必须失败。float64 oracle 单独加诊断偏移不改变主验证结果。

### 12.2 运行与集合证据修订

- 全量 PKL/H5 内容哈希在 emit/freeze 和新 run 的首批执行；同 run 后续批仍校验完整冻结元数据，并定点重哈希本批 artifact/YAML/A-pool。节点前后实际 preload/model/code 指纹保留，源或运行身份变更拒绝恢复。批内不再重读全部 900 个无关采集 H5。
- Owner 已接受紧凑 map 的集合身份分支，见 §6.2/§11.1；不新增逐轨迹映射。合成完整流水线验证两种 schema 导出相同 seen/unseen 集合、且状态字节不等时拒绝。
- 空任务防御分支移入失败报告路径，返回完整候选表/耗时和 `grid_not_representable`。
- 资源测量保留 0.5 秒频率，写盘节流至 2 秒并在新峰值或进程集合变更时立即写；验证器流式遍历。client 冷启动 probe 超时延长至 600 秒，超时/非零退出保留 `client_probe.log` 和 `preflight_failure.json`（infra_failure），不启动 server。

### 12.3 复审证据（已完成）

真实验证脚本：`exp/ablation_study/cache_prune/data/advisory/run_real_g2r1.py`；在 weilandserver 的 `/home/weiland/openpi` 执行，输出根 `/data/openpi/ablation_study/cache_prune/g2_r1_20260911`。四源串行调用真实 `audit_source → compute_task_scores → select_grid → export_pruned(P05) → verify_pruned`，CPU/BLAS 四线程，不启动模型服务或 rollout。每源保留全部九点表和 P05 全 query 验证。P05 固定用于复现本次 bug，不依据评测结果挑点。

**四源全部通过，覆盖 33,448 个源 query；最大 top-1 regret 均为 0。**

| suite / regime | 全查询数 | P05 保留 | frozen→loaded 最大误差 | 最大 regret | near ties | float64 最大误差 / per-query-max p99 | 全流程秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| libero_spatial / rit50 | 1,018 | 614 | 0 | 0 | 0 | 8.03136e-05 / 6.8e-05 | 72.14 |
| libero_spatial / cs500_success | 9,329 | 5,603 | 0 | 0 | 0 | 0.000138799 / 0.000112025 | 740.13 |
| libero_10 / rit50 | 2,640 | 1,584 | 0.00010632 | 0 | 218 | 0.000179003 / 0.000152006 | 196.72 |
| libero_10 / cs500_success | 20,461 | 12,251 | 0 | 0 | 1 | 0.000226546 / 0.000178431 | 2704.42 |

近并列不是错误 winner 数。L10 小库的 218 个近并列 query 与大库的 1 个并列均保留在机器表；不声称全量位等价。以阈值一 ULP 为带的 legal/deleted pair 数依次为 Spatial 小库 0/0、大库 3/0，L10 小库 1/0、大库 44/7；全部严格冻结分数比较仍通过，未放宽剪枝阈值。全流程包含源审计、打分、选点、导出和验证，不是检索延迟基准。

**两套真实 membership 全量内容核验均通过**：Spatial 与 L10 各 500/500 common-unseen、每任务 50 个；两档原采集记录分别为 50/450 条（使用成功筛选前的原采集集合）。因此本批 full 与 common-unseen 一致，seen 子集为空，报告不可估计而非 SR=0。L10 50 条紧凑集合记录全部仅标记 `declared_collection_init_set`，没有 `trajectory_id`。服务节点核验耗时 Spatial 400.10 s、L10 1,387.31 s；回传后 seal、双源绑定与字节 SHA 再核通过。本地另有 L10 50/50 子集/全量状态相等的独立证据。

本轮 advisory 完整自查 **79 passed / 1 upstream pynvml warning，42.06s**，Ruff/9 CLI/公共接口 docstring 通过。新 preflight 测试首跑因 fixture 漏 manifest version 失败，补齐后全部通过，未放宽生产检查。被测 25 份代码/config/test 的 SHA 在实测完成后仍一致。远端 `verify_prune.py` 与本地仅 Ruff 折行不同，AST SHA 相同；具体字节/AST 指纹保留在自查日志。

复审入口：

- [四源报告及九点表](../exp/ablation_study/cache_prune/data/advisory/real_g2r1_20260911/report.md)、[原始 grid CSV](../exp/ablation_study/cache_prune/data/advisory/real_g2r1_20260911/grid_points.csv)、[回传证据 SHA 清单](../exp/ablation_study/cache_prune/data/advisory/real_g2r1_20260911/evidence_inventory.json)。同目录保留四套 `grid_freeze.json`/`verification.json`、两个 `eval_membership.json` 及两份 summary。大 PKL/score blocks/source/artifact manifests 保留在上述服务节点路径，不拉回工作树。
- [79 项自查日志](../exp/ablation_study/cache_prune/data/advisory/code_selfcheck_g2r1_20260911.log)、[25 文件代码指纹](../exp/ablation_study/cache_prune/data/advisory/code_inventory_g2r1_20260911.json)。
- [本轮 9 文件增量 patch](../exp/ablation_study/cache_prune/data/advisory/cache_prune_g2r1_incremental.patch)、[当前完整 scoped patch](../exp/ablation_study/cache_prune/data/advisory/cache_prune_g2r1_full.patch)。patch 为未暂存工作树快照，不接管其他实验。

**剩余阶段明确**：本轮完成四套真实网格、四份代表 P05 的物理子库及全查询验证、两套 membership。其余 32 个剪枝子库、完整 40 臂发放/验证、四源 preload probe、八个真实 smoke 和 20,000 ep 正式 rollout 尚未执行。这些 advisory 实证不替代 G2 后 Verify，也不构成自行批准 G2。

### 12.4 四源九点冻结表

全部按 §4.2 同一预注册规则各自选点，四源各 329 个候选；阈值保留可往返 float32 的精度。P00 不剪枝，见 §2 分母；表内是 entry 删点率，不是轨迹删减比例或磁盘压缩率。

**libero_spatial / rit50**：完整选点 10.31 s，最大目标偏差 0.3143 个百分点。

| 点 | 阈值 θ | 目标删点率 | 实际删点率 | 保留条目 |
|---|---:|---:|---:|---:|
| P01 | 0.98642092943191528 | 5% | 4.9116% | 968 |
| P02 | 0.9853365421295166 | 10% | 10.2161% | 914 |
| P03 | 0.98335790634155273 | 20% | 20.1375% | 813 |
| P04 | 0.9812852144241333 | 30% | 29.9607% | 713 |
| P05 | 0.97886109352111816 | 40% | 39.6857% | 614 |
| P06 | 0.97495996952056885 | 50% | 49.7053% | 512 |
| P07 | 0.96737754344940186 | 60% | 60.0196% | 407 |
| P08 | 0.94843864440917969 | 70% | 69.7446% | 308 |
| P09 | 0.83648133277893066 | 80% | 79.7642% | 206 |

**libero_spatial / cs500_success**：完整选点 133.39 s，最大目标偏差 0.2165 个百分点。

| 点 | 阈值 θ | 目标删点率 | 实际删点率 | 保留条目 |
|---|---:|---:|---:|---:|
| P01 | 0.98771166801452637 | 5% | 5.0059% | 8,862 |
| P02 | 0.98736381530761719 | 10% | 9.8724% | 8,408 |
| P03 | 0.98669981956481934 | 20% | 20.2165% | 7,443 |
| P04 | 0.98607069253921509 | 30% | 29.9603% | 6,534 |
| P05 | 0.98535263538360596 | 40% | 39.9400% | 5,603 |
| P06 | 0.98446494340896606 | 50% | 50.0697% | 4,658 |
| P07 | 0.98325538635253906 | 60% | 60.1029% | 3,722 |
| P08 | 0.98108410835266113 | 70% | 70.2005% | 2,780 |
| P09 | 0.9760785698890686 | 80% | 80.2122% | 1,846 |

**libero_10 / rit50**：完整选点 29.26 s，最大目标偏差 0.4167 个百分点。

| 点 | 阈值 θ | 目标删点率 | 实际删点率 | 保留条目 |
|---|---:|---:|---:|---:|
| P01 | 0.99849998950958252 | 5% | 4.9242% | 2,510 |
| P02 | 0.99837797880172729 | 10% | 9.8485% | 2,380 |
| P03 | 0.99809229373931885 | 20% | 20.0379% | 2,111 |
| P04 | 0.99778330326080322 | 30% | 30.0000% | 1,848 |
| P05 | 0.99720340967178345 | 40% | 40.0000% | 1,584 |
| P06 | 0.99625909328460693 | 50% | 50.0379% | 1,319 |
| P07 | 0.9945063591003418 | 60% | 59.8864% | 1,059 |
| P08 | 0.99083924293518066 | 70% | 70.4167% | 781 |
| P09 | 0.96605491638183594 | 80% | 79.8485% | 532 |

**libero_10 / cs500_success**：完整选点 386.60 s，最大目标偏差 0.2043 个百分点。

| 点 | 阈值 θ | 目标删点率 | 实际删点率 | 保留条目 |
|---|---:|---:|---:|---:|
| P01 | 0.99862128496170044 | 5% | 4.9509% | 19,448 |
| P02 | 0.99854981899261475 | 10% | 10.1510% | 18,384 |
| P03 | 0.99844348430633545 | 20% | 19.9306% | 16,383 |
| P04 | 0.99833881855010986 | 30% | 30.0474% | 14,313 |
| P05 | 0.99821764230728149 | 40% | 40.1251% | 12,251 |
| P06 | 0.99806523323059082 | 50% | 50.0415% | 10,222 |
| P07 | 0.99784153699874878 | 60% | 59.8993% | 8,205 |
| P08 | 0.99745845794677734 | 70% | 69.8011% | 6,179 |
| P09 | 0.99667012691497803 | 80% | 79.7957% | 4,134 |

## 13. Owner 并发运行裁定（覆盖旧运行约束）

2026-09-11，Owner Ziyang Lin 明确指示废弃过时运行计划，改用已有 conductor + concurrent server，免除本次替换的重复 G1/G2 确认。此前 commit/push 授权继续有效；不含本轮启动正式 rollout。§6.3、§9、§11 的单臂批次、单库进程、四源 probe 与逐臂重启安排退为历史实现；本轮以新的多 YAML 分组入口和运行交接计划为准。剪枝规则、四源身份、40 臂、每臂 500 集、固定检索参数、FULL_HIT 与 accepted-attempt 统计不变。

每组按内存预算容纳多个 YAML/PKL，使用既有 `PureCacheEvalStrategy` / `ConductorDriver` / `WorkerAgent` / concurrent server；组开始前加载该组全部库，episode 按 server 与 bundle_id 调度，组结束释放本组专用 server。`eval_concurrency` 仅限制活跃 YAML，不能代替累计驻留库预算。保留组级原始日志及可核验的逐臂视图，单组失败保留证据并整组重跑；共享 RSS 不归因到单个 PKL。无 warmup/标定 rollout。

## 14. 并发实现与 Verify / 提交记录

新增 `concurrent_plan.py` 与 `run_concurrent.py`，复用现有纯 cache ExperimentStrategy、ConductorDriver、WorkerAgent 和 concurrent server。冻结分组全量驻留预算、逐组专用多端点、组前 preload/bundle ack、组后清理、失败整组重跑、原始混合日志与逐臂无损视图均已实现。旧 runner 保留为历史数据读取/底层工具，其 `accepted_batches` 可识别并发 catalog；新正式入口见 [运行交接](cache_prune_run_handoff.md)。共享 RSS 与在线排队延迟标记见分析输出，独立检索微基准保持原口径。

新增 8 项测试通过（21.78s）：实际 conductor 尾部跨 YAML 领取、整组累计内存、重复/缺失/额外 PKL 拒绝、单/双端点的短生命本机进程与 sockets、失败清理、smoke→eval、原始多臂日志分区防篡改。模型/模拟器部分在测试替换，不能当作真实硬件 smoke。

全仓检查时发现两个测试环境/夹具问题并修正：进程集成测试固定模拟节点容量，避免小内存测试机触发生产的 16 GiB 预留门，实际进程 RSS 采样、预算断言和清理继续执行；对应 22 项通过（35.69s）。另一个是既有 `tests/actioncache_baseline/test_acb_interceptor_cp2.py` 的 warm-start 夹具漏填必需 `schedule_id`：原文件及相关生产代码均逐字节等于 `77c5ef9`，单独运行复现 1 failed / 6 passed；只补 `pi05_v1` 字段后原 7 项断言通过（8.30s），没有改动生产逻辑或放松断言。这一独立夹具修复已单独提交为 `d016598`。

**全仓 Verify 未完成，不声明 all-pass。** 在 `77c5ef9` 的隔离快照加本轮文件上执行裸 `uv run pytest`，共收集 5,327 项；最后一轮 cache_prune 的 87 项全部通过，随后运行到约 48% 的既有 `dispatch_surface/test_rev2_confirmation.py` 长时统计重采样时主动停止。该快照仍含修复前的 CP2 夹具，观察到的该项失败已由上段独立修复/复测处理；不推断尚未执行项目通过。首次收集的 editable 客户端路径冲突已通过显式 `PYTHONPATH` 修正，原环境未重装。完整/中止日志保留于 `data/advisory/verify_full_20260911.log`、`verify_full_r2_20260911.log`、`verify_full_r3_20260911.log`。

Owner 本轮明确指示“废弃当前过时的运行计划，改用我们实验的 conductor concurrentserver 基础设施，无需流程桎梏……无需再次询问”。据此，本次以实现及其基础设施的相关回归完成交付，不再让旧全仓流程门阻止已授权的 commit/push，也不自行签发新 G2。最终相关回归覆盖 cache_prune、cache_size、conductor、serving、backend 并发/冻结及修复后的 CP2 测试；结果及完整输出在下段登记。隔离快照不包含未提交的其他实验改动或 reviewer 专用测试。

最终相关回归：**566 passed、11 skipped、3 warnings / 127.70s**；11 个 skip 来自仓库原有 manual/env_dependent 默认策略，本轮未新增 skip。完整输出：[verify_relevant_20260911.log](../exp/ablation_study/cache_prune/data/advisory/verify_relevant_20260911.log)；快照清单：[verify_candidate_relevant_20260911.json](../exp/ablation_study/cache_prune/data/advisory/verify_candidate_relevant_20260911.json)。使用既有依赖环境（`UV_NO_SYNC=1`、`UV_OFFLINE=1`），`PYTHONPATH` 明确指向隔离快照的 src/openpi-client，OMP/MKL/OPENBLAS 各 4 线程。

```bash
uv run pytest tests/ablation_study/cache_prune tests/ablation_study/cache_size \
  tests/conductor tests/serving tests/actioncache_baseline/test_acb_interceptor_cp2.py \
  tests/cache/test_in_memory_backend_concurrent.py tests/cache/test_backend_frozen_autoguard.py
```

`ruff check` 与 `ruff format --check` 均通过（22 个实验/测试 Python 文件），`run_concurrent --help` 通过；提交前 `uv-lock` pre-commit 通过。主提交包含 35 个 cache_prune 实现/配置/测试/文档文件，`src/` 零改动。主交接版本用 `git log -1 --format=%H -- logs/cache_prune_run_handoff.md` 定位，独立夹具修复为 `d016598`；正式数据补全与真实 smoke/eval 尚未启动。

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-09-11 20:26 CDT

评审基线：工作树（untracked `exp/ablation_study/cache_prune/`、`tests/ablation_study/cache_prune/` + 四处索引/文档 diff），对照 polished 计划 §2–§9。通读全部 18 个 py/config 文件与 4 个测试模块；亲跑执行方套件 `73 passed / 41s`；`ruff check` + `ruff format --check` 全绿。独立探针（`tests/review_tests/`，不入索引）：R1 在**真实** `libero_spatial/cp1_spatial_pool_16.pkl` 上跑 `audit_source → compute_task_scores → select_grid → export_pruned(P05) → verify_pruned`；R2 真实 L10 小库经紧凑 map 审计；R3 用真实 `episode_runner._hit_row` + `Journal.record` 产出的行喂 `episode_ledger`。R1 的审计/打分/选点通过：49/1,018、329 候选 4.1 s 冻结、九点 4.9/10.2/20.1/30.0/39.7/49.7/60.0/69.7/79.8%，且与 `prune_feasibility.py` 扫点在全部 6 个重合阈值上保留数逐一相等（998/895/647/436/312/241）；R2、R3 通过。

- [Blocking] [Concern] `verify_prune.py:134` 固定 `numerical_bound = 5e-5` 在真实数据上不成立，`verify_pruned` 会拒绝每一个真实 spatial 产物，`emit_prune_arms.main` 因此无法冻结任何臂 — reasoning: 对真实 spatial P05 子库（614 entries）以 1,018 个源 query 复核，backend float32 分数与 float64 oracle 的每 query 最大误差**中位 4.6e-5、p99 6.8e-5、最大 8.0e-5，360/1,018 个 query 超过 5e-5**（`retrieval scores exceed fixed numeric bound: 5.789e-05`），而 top-1 与 oracle 的 regret 全部为 0（winner 无一错误）。根因是 32768 维 cosine 的 float32 归约误差经 zscore 被 1/σ≈143 放大，执行方测试只用 3 维向量、μ=0/σ=1 的合成库，从未触及该量级。计划 §4.1/§8.4 要求「记录误差而非断言虚假的全量位等价」「不能靠放宽分数误差掩盖明显 winner 错误」，而不是一个未经真实数据标定的魔数。修法：以冻结的 float32 score blocks（同一生产管线）作为装载后分数的主等价参照并按 float32 误差传播给出可解释的界，float64 oracle 降为记录性诊断（报告 max/p99），保留 `regret ≤ 2·error` 与 `top-1 == max` 作为硬门；并新增按真实维度/σ（如 4096+ 维、σ=0.007）构造的测试，且在四份真实源上跑通 `verify_pruned` 后再申请复审。
- [Non-blocking] [Concern] `validate_run`→`validate_freeze(rehash=True)`→`validate_membership(rehash=True)` 在**每个批次启动时**对全部 4 源 PKL、40 个产物 PKL 以及 membership evidence 里的 900 个采集 H5（S6/all 各 450 个，spatial 约 38 GB、L10 约 104 GB）逐字节重哈希 — reasoning: 单批约 150 GB 顺序读，weilandserver HDD 约 150 MB/s ⇒ 每批 15 分钟以上、40 批累计十余小时，且与本机 serving/HDD 串行化约束冲突（`reference_weilandserver_disks`）。计划 §9 的批次 preflight 只要求本批 preload 库与 A-pool 的实测 SHA。建议 `validate_run` 用 `rehash=False` + 本批 artifact/yaml/apool 的定点 `check_identity`（`run_batch` 已做），全量重哈希只在 emit/freeze 与首批各做一次。
- [Non-blocking] [Question] L10 小库 membership 被 `prepare_membership.py:130` 的完整 schema 要求整体拦下（§11.1），但 §6.2 的 seen/unseen 定义只需要**每任务采集 init 的集合**，不需要逐轨迹↔init 绑定 — reasoning: 我实测 L10 紧凑 map 的 50 条 `(task, subset_idx→orig_init_state_idx)` 与 `db_init/libero_cache/libero_10/*.init` ↔ `db_init/libero/libero_10/*.init` 的状态字节**全部一致（50/50）**，`audit_source` 又已证明 H5 逐任务数量与 map 相符；因此集合层面的 seen 集可由 map 声明 + 子集/全量状态字节等式得到，其证据强度与 spatial map（同样是采集脚本写下的声明）相当。请 owner 裁定：(a) 接受集合层面声明作为 L10 seen 集（`prepare_membership` 加紧凑 map 分支，`collection_rows` 不填逐轨迹绑定）；(b) 维持现状，L10 两档 20 臂在找回逐轨迹证据前不发臂。此为协议解释，不构成代码缺陷。
- [Non-blocking] [Concern] `select_prune_grid.py:158` 的「任一任务无保留点」`require` 位于 try 之外，触发时会以未捕获 ValueError 退出而不写 `grid_freeze.json`，与 §4.2 步骤 6「报 grid_not_representable 并附完整扫点表」不一致 — reasoning: 因每条轨迹终点 r=0 必保留，此分支在合法源上不可达，属防御性代码；建议要么并入候选过滤（DP 只在满足每任务非空的候选上选），要么把它移进 try 使报告落盘。
- [Non-blocking] [Suggestion] `run_prune_eval.py` 采样每 0.5 s 一行且每行重写全部 owned `processes`/`workers` 列表，L10 正式批（≈10 h）会产生 7 万+ 行、数百 MB 的 `resources.jsonl`，`validate_batch_evidence` 又整文件 `json.loads` 进内存 — reasoning: 建议 2–5 s 采样或只在集合变化时写进程列表；峰值语义（VmHWM/采样峰）不受影响。另 `_client_probe` 的 `timeout=120` 在冷 HDD 上 `conda run` 导入 libero/robosuite/torch 可能超时，建议放宽并把超时当 infra failure 记录。
- [Non-blocking] [Suggestion] 复审时请一并附上四份真实源 `verify_pruned` 的 `max_score_error/near_tie_queries` 与 `grid_freeze.json` 的九点表（R1 已证明 spatial rit50 可在数秒内完成，不涉及 rollout），作为 B1 修复的实证而非合成测试。

### G2 Round 1 — Executor — 修订完成，申请复审 — 2026-09-11 21:42 CDT

- Accepted — **B1 数值等价参照与真实维度覆盖**：`verify_prune.py:130/170` 改为冻结 float32 score blocks 的主比较，并按维度、σ、权重传播可解释的保守 float32 误差预算；float64 oracle 只报告 max / per-query-max p99。`top-1 == actual.max` 与基于每 query 实测误差的 `regret ≤ 2·error` 保留硬门，不能借理论界掩盖 winner 错误。`emit_prune_arms.py:146` 拒绝旧版验证报告。新增 32,768 维、σ=0.007 的数值回归与损坏 runtime/winner 负例；四份真实源固定 P05 的全查询验证均通过，数值与限制见 §12.1/§12.3。
- Accepted — **N1 每批全量重哈希开销**：`run_prune_eval.py:258/669` 将完整内容核验限制在 emit/freeze 和新 run 首批；后续批仍核验冻结元数据并重哈希本批实际 PKL/YAML/A-pool，保留模型/代码/节点前后身份门。新增首批与续批行为测试；未通过跳过本批实际输入检查来节省 I/O。
- Accepted — **N2 L10 集合层面身份裁决**：Owner 已明确接受集合证据并支持 L10 两档发臂。`prepare_membership.py:124/212` 增加紧凑 map 分支：核对每任务集合、HDF5 task census、子集/全量 init 状态字节以及成功源属于原采集集合；`identity_basis=declared_collection_init_set`，不填逐轨迹 init 绑定。真实两 suite membership 已完成内容核验及冻结；L10 不再因缺逐轨迹绑定而整体拦截。详见 §6.2/§11.1/§12.3。
- Accepted — **N3 空任务失败报告**：`select_prune_grid.py:160` 的每任务非空检查已移入 try，失败统一落完整扫点表、耗时和 `grid_not_representable`，新增触发该防御路径的测试。合法源的剪枝和机械选点规则未改变。
- Accepted — **N4 资源日志与冷启动 probe**：`run_prune_eval.py:368/669` 保留 0.5 秒测量，改为每 2 秒、新峰值或进程集合变化时写盘；验收流式消费 resources.jsonl。`_client_probe:147` 超时改为 600 秒，超时/非零退出保存 client_probe.log 与带版本的 preflight_failure.json（infra_failure），不启动 server；新增模拟 TimeoutExpired 的证据保留测试。既有中断批次恢复分支仍可整读旧资源文件，本次未将其改称流式。
- Accepted — **N5 四源真实复审材料**：§12.3 给出四源 P05 全查询的 max_score_error / max_top1_regret / near_tie_queries 及 float64 诊断，§12.4 给出四套完整九点阈值表。服务节点保留 source/scores/artifact/report；本地回传 grid/report/membership 并核对 seal、绑定与字节 SHA，完整路径和 inventory 见 §12.3。79 项 advisory 测试通过；没有把四份 P05 离线验证宣称为四十臂已发放、八组真实 smoke 已通过或正式 rollout 已完成。

### G2 Round 2 — Reviewer — APPROVED — 2026-09-11 21:50 CDT

评审基线：工作树；用 R1/R2 两版 `code_inventory_*.json` 逐文件 SHA 比对定位本轮改动（5 个模块 + 3 个测试模块修改、新增 `test_cache_prune_numerics.py`，清单与磁盘一致），并逐段读完 `cache_prune_g2r1_incremental.patch`。R1 全部 6 条均有对应 Executor 回应且实现与描述一致：B1（冻结 float32 blocks 为主参照、`winner.score == max` 精确门、`regret ≤ nextafter(2·实测误差)`、float64 降为诊断、emitter 拒旧版报告）、N1（首批全量重哈希、续批定点重哈希本批 PKL/YAML/A-pool）、N2（紧凑 map 集合分支：子集/全量状态字节等式 + H5 任务 census + 完整覆盖子集，`identity_basis=declared_collection_init_set`）、N3（空任务检查移入 try）、N4（写盘节流 2 s/新峰值/进程集变更，验收流式读取；probe 600 s 超时留证）、N5（四源真实报告）。R1 Reviewer 段逐字未动。

独立复核：执行方套件亲跑 **79 passed / 41 s**，`ruff check`/`format --check` 全绿；用修订后的 `verify_pruned` 复核 R1 被拒的**同一份**真实 spatial P05 产物 → `passed=True`、`max_score_error=0`、`max_top1_regret=0`、float64 诊断 8.03e-5/p99 6.8e-5，与执行方服务节点数字逐位一致；回传的四源 `grid_freeze.json`/`verification.json` 与两套 `eval_membership.json` 封印、spec/score 绑定、`validate_membership(rehash=False)` 全部通过；spatial rit50 九点阈值/保留数与我 R1 本地独立选点逐点相同（跨机确定性）。

- [Non-blocking] [Suggestion] `float32_error_budget` 在生产尺度上是空泛的（Spatial 0.285、L10 0.507，而四源实测 ≤1.07e-4）——它只能挡住数量级错误，接受判定实际落在精确 top-1 门、`regret ≤ 2·实测误差` 与记录的实测误差上，§12.1 已如实说明 — reasoning: 建议在 verification 报告与分析摘要里把该界标为 `sanity_envelope` 而非「acceptance bound」，并把四源实测 `max_score_error` 的跨源包络（当前 1.07e-4）作为**记录性**期望写入 freeze，以便未来重跑时发现漂移；不作为门。
- [Non-blocking] [Suggestion] §6.2 第 172 行仍保留「若核对证实小库原采集恰为 A-pool 每任务 5 个，主子集为 45/task、450；这是待身份核对的推断」——实测已否定该前提（两 suite 小库采集 init 均落在差集池，A-pool 500/500 共同未见，seen 子集为空，见 §12.3） — reasoning: 属 Post-G2 polish 的正文整理项，请改为陈述实测结论，避免读者误以为主子集是 450。
- [Non-blocking] [Suggestion] `run_batch` 内 `accepted_batches()` 对每个既有有效批次都重跑 `validate_completion → episode_ledger`（解析 500 集 journal/per_step），40 批时呈二次增长 — reasoning: 单批秒级、总量分钟级，可接受；若日后扩批，可把已验证批次的 `completion.digest` 缓存到 `run.json` 免重解析。

### G2 Round 2 — Executor — Owner 授权后续运行替换

- Accepted — **N1 数值界说明**：新 verification 增加 `float32_error_model_role=sanity_envelope`，新 freeze 记录四源 P05 实测误差及跨源包络，明确 `hard_gate=false`；不修改旧报告 seal 或数值判定门。
- Accepted — **N2 共同未见规模**：§6.2 已改为两 suite 均 500/500 的实测结论，seen 为空；删除旧 450 集假设。
- Accepted — **N3 重复解析成本**：本轮新并发 catalog 按完整 group 验证一次再返回各臂视图；跨调用仍重验证，后续扩大规模时再引入以 completion/content SHA 绑定的缓存，本轮不通过跳过证据节省时间。

Owner 已明确要求废弃旧运行方式并免除替换的重复流程询问；本条记录授权而非自行发布新 G2 verdict。原两轮 Reviewer 记录逐字保留。
