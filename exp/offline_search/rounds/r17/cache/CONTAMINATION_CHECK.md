# R17 cache study：A/B 数据污染核查

**结论：就“是否在缓存自己的初态上测试”而言，未发现污染。A1000 与六个 B500 库的精确重叠均为 0；库行、corrector 与 stall 的冻结输入均来自 B。** 距离筛查标记 spatial 11/500、libero_10 21/500 个 A 初态；这些是近邻提示，不是已证泄漏。Cosmos 尚无本次执行记录，结论覆盖其已封存输入和计划。

审计完成：2026-10-10T19:59:20.613047-05:00（CDT）。原始结果、配置、库和进程只读；无 GPU、campaign 操作或凭据访问。运行中日志按打开时文件长度截取完整行，保存 prefix SHA，以下不是完整实验完成声明。

## 1. 实际评测集合与加载链

OWNER_RULINGS 已明确：official test A、每任务50初态、每suite500；此前“全B集”是口误，B只用于建库。因此早版 CACHE_STUDY 的待确认 B 分支不代表本次正在执行的集合。[owner裁定](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/cache/OWNER_RULINGS.md:2)；[原条件分支](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/cache/CACHE_STUDY.md:103)。

| 模型/lane | 计划局数 / 唯一A初态 | 已检查的带身份记录 | 已观察A身份 spatial / l10 | 服务器bundle确认 |
|---|---:|---:|---:|---:|
| DP / csl10d | 8000 / 1000 | 179,066 | 500 / 1 | 16 |
| Fast-WAM / csl10f | 8000 / 1000 | 88,182 | 500 / 0 | 16 |
| Cosmos / csl10c | 4000 / 1000 | 0 | 0 / 0 | 0 |

每个实际group×suite计划均为500个唯一身份；DP/Fast各8组，Cosmos折叠为4个唯一组。所有计划均 `init_pool=A, partition=test, orig_init_state_idx=0..49`；已观察记录的 A标签、实际 `task_data_hash`、bundle/recipe/YAML身份零不一致。DP/Fast各16份load_cache_config ACK与封存配置一致；Cosmos无ACK/逐步日志，不能声称已运行验证。[DP plan](/data/mw_finetune/eval/csl10d/EPISODE_PLAN.json:1)；[Fast plan](/data/mw_finetune/eval/csl10f/EPISODE_PLAN.json:1)；[Cosmos plan](/data/mw_finetune/eval/csl10c/EPISODE_PLAN.json:1)；[DP ACK](/data/mw_finetune/eval/csl10d/STUDY_BUNDLE_ACKS.json:1)；[Fast ACK](/data/mw_finetune/eval/csl10f/STUDY_BUNDLE_ACKS.json:1)。

**实际数组，不只比较 A/B 标签。** formal worker按封存benchmark的canonical task表转为suite-local ID，走 `_build_episode_setup`、`_load_init_states`；未指定自定义池时，官方 `get_task_init_states` 加载 `<task>.pruned_init`，随后取原始index。审计用相同官方初态的本机副本，以CPU `torch.load` 保留dtype/shape/字节；未构造环境，也未用机器人8/9维policy state或settle后的状态替代完整simulator init。[worker入口](/home/weiland/projects/openpi/exp/r17_integration/conductor/worker_entry.py:96)；[按原index加载](/home/weiland/projects/openpi/examples/libero/worker_entry.py:77)；[加载分支](/home/weiland/projects/openpi/examples/libero/main.py:1211)；[官方pruned_init命名](/home/weiland/projects/openpi_ext/envs/libero_sim/lib/python3.8/site-packages/libero/libero/benchmark/__init__.py:76)；[官方加载函数](/home/weiland/projects/openpi_ext/envs/libero_sim/lib/python3.8/site-packages/libero/libero/benchmark/__init__.py:158)。

字节绑定方法直接调用 teacher capture 使用的 `array_identity`：`SHA256(raw.tobytes(order="C"))`，同时保留原dtype和shape；`task_data_identity`再把canonical task名一起封入hash。worker也在环境接收初态前计算同一身份。[数组hash函数](/home/weiland/projects/openpi/exp/r17_integration/ops/library_alignment.py:20)；[B teacher赋值](/home/weiland/projects/openpi/exp/r17_integration/ops/library_campaign.py:39)；[实际worker状态hash](/home/weiland/projects/openpi/exp/r17_integration/benchmarks/libero_bridge.py:188)。

本机 A 文件根为 `/home/weiland/projects/openpi_ext/envs/libero_sim/lib/python3.8/site-packages/libero/libero/init_files/`；B 文件根为 `/home/weiland/projects/openpi/exp/common/data/db_init/libero/`。三份已完成控制组各1000主集身份的实测task_data_hash全部等于重算A；当前cache日志也逐行相等，补足“本机副本是否为worker实际A字节”的证据。B原文件SHA又与R10历史pool record、R17 alignment及全部library episode SHA一致。[spatial alignment](/home/weiland/projects/openpi/exp/r17_integration/ops/samples/libero_library_alignment/libero_spatial.json:1)；[l10 alignment](/home/weiland/projects/openpi/exp/r17_integration/ops/samples/libero_library_alignment/libero_10.json:1)。

## 2. 精确重叠：六库全部为0

额外逐任务验证了 `B50 = 原始 .init 中100状态 − 官方 .pruned_init 中50状态`，20任务全部成立；不是仅相信旧报告“difference pool”。A1000字节hash全唯一；每个B库均500个唯一task/init和hash（每任务50）。三模型使用同一批B初态，但各自产生轨迹。下表“全A”不加task过滤，对全1000个A hash求交，仍为0。

| 模型 | suite | A / B episode | 同task重叠 / 全A重叠 | 核验B-only库行 | 近邻标记 | 原始库证据 |
|---|---|---:|---:|---:|---:|---|
| DP | libero_spatial | 500 / 500 | 0 / 0 | 12,028 | 11/500 | [READY](/data/r17_library/dp_libero_libero_spatial/bpool/READY.json:1) |
| DP | libero_10 | 500 / 500 | 0 / 0 | 31,546 | 21/500 | [READY](/data/r17_library/dp_libero_libero_10/bpool/READY.json:1) |
| Fast-WAM | libero_spatial | 500 / 500 | 0 / 0 | 12,460 | 11/500 | [READY](/data/r17_library/fastwam_libero_libero_spatial/bpool/READY.json:1) |
| Fast-WAM | libero_10 | 500 / 500 | 0 / 0 | 27,806 | 21/500 | [READY](/data/r17_library/fastwam_libero_libero_10/bpool/READY.json:1) |
| Cosmos | libero_spatial | 500 / 500 | 0 / 0 | 3,676 | 11/500 | [READY](/data/r17_library/cosmos_libero_libero_spatial/bpool/READY.json:1) |
| Cosmos | libero_10 | 500 / 500 | 0 / 0 | 8,408 | 21/500 | [READY](/data/r17_library/cosmos_libero_libero_10/bpool/READY.json:1) |

合计：3,000个库episode、95,924行，A污染episode/行均0；同一B初态在三模型重复计入，不能当3,000个独立B状态。每一库行均核验 `init_pool=B, split=library`，task_uid/attempt/task名/original-index/init SHA/task_data_hash与其episode及实际B文件一致；不因同为index0..49就把B与A混为一池。[构库身份检查](/home/weiland/projects/openpi/exp/r17_integration/search/teacher_library.py:25)；[capture的B约束](/home/weiland/projects/openpi/exp/r17_integration/debug/library_capture.py:23)。

## 3. 原始状态向量近重复检查

按任务独立计算未标准化float64欧氏距离：`dAB(i)=min_j ||A_i−B_j||₂`；A–A/B–B均排除自己，再取各初态的NN距离。阈值 `q01=quantile(BB_NN,0.01,method="linear")`，严格 `dAB<q01` 标记。A/B每任务各50；raw维数45–123，包含完整保存状态、物体位姿等，单位混合；不跨任务算距离、不去掉坐标、不做quaternion等价折叠。阈值是筛查规则，不是物理意义上的“同一个场景”证书。

下表中位数比较典型A–A、A–B、B–B距离；数字保留6位，判定用未四舍五入值。task ID沿formal全局编号，完整canonical名和每个A的近邻均在CSV。

| task / 简称 | 维数 | A–A NN中位数 | A–B NN中位数 | B–B NN中位数 | B–B q01 | 最小A–B | flags |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 黑碗在盘与小碟之间 | 92 | 0.048487 | 0.050858 | 0.054768 | 0.041385 | 0.040826 | 2 |
| 1 黑碗在小碟旁 | 92 | 0.052506 | 0.053542 | 0.049470 | 0.031161 | 0.037492 | 0 |
| 2 桌面中央黑碗 | 92 | 0.049855 | 0.050043 | 0.050012 | 0.027352 | 0.036136 | 0 |
| 3 饼干盒上黑碗 | 92 | 0.056332 | 0.055505 | 0.055982 | 0.026198 | 0.041176 | 0 |
| 4 上层抽屉黑碗 | 92 | 0.052295 | 0.055767 | 0.051479 | 0.038195 | 0.037367 | 1 |
| 5 小碟上黑碗 | 92 | 0.050377 | 0.051362 | 0.052631 | 0.042183 | 0.035952 | 5 |
| 6 饼干盒旁黑碗 | 92 | 0.053397 | 0.052898 | 0.053454 | 0.035076 | 0.038628 | 0 |
| 7 炉上黑碗 | 92 | 0.057150 | 0.055733 | 0.054900 | 0.039336 | 0.038710 | 1 |
| 8 盘旁黑碗 | 92 | 0.050354 | 0.050505 | 0.050564 | 0.040309 | 0.034349 | 2 |
| 9 柜上黑碗 | 92 | 0.057958 | 0.058831 | 0.056257 | 0.042379 | 0.044105 | 0 |
| 30 字母汤＋番茄酱入篮 | 123 | 0.074653 | 0.074441 | 0.074506 | 0.063746 | 0.061140 | 3 |
| 31 奶酪＋黄油入篮 | 123 | 0.074191 | 0.073292 | 0.072568 | 0.056504 | 0.052132 | 2 |
| 32 开炉并放摩卡壶 | 47 | 0.049811 | 0.048331 | 0.048351 | 0.023803 | 0.034226 | 0 |
| 33 黑碗入底抽屉并关闭 | 51 | 0.050028 | 0.047973 | 0.047280 | 0.033709 | 0.033254 | 1 |
| 34 两杯分别放两盘 | 84 | 0.066787 | 0.067159 | 0.064097 | 0.052158 | 0.049345 | 2 |
| 35 书放收纳盒后格 | 45 | 0.064116 | 0.064844 | 0.063771 | 0.044276 | 0.042398 | 1 |
| 36 白杯放盘＋布丁放右侧 | 71 | 0.059634 | 0.060787 | 0.064062 | 0.052737 | 0.042736 | 7 |
| 37 字母汤＋奶酪入篮 | 84 | 0.062027 | 0.060176 | 0.062338 | 0.042332 | 0.040188 | 1 |
| 38 两摩卡壶放炉上 | 47 | 0.069694 | 0.073273 | 0.075999 | 0.060765 | 0.041997 | 4 |
| 39 杯放微波炉并关闭 | 47 | 0.070459 | 0.067045 | 0.072803 | 0.036491 | 0.042856 | 0 |

**全部近邻标记（同一列表适用于三个模型）：** spatial 11/500=2.2%，l10 21/500=4.2%；去重32个A初态，按模型重复计数为96，不能说是96个不同初态。A→B编号均为其各自池的原始index。

| task | A init → 最近B init（距离） |
|---|---|
| 0 黑碗在盘与小碟之间 | 24→11 (0.041121)；48→9 (0.040826) |
| 4 上层抽屉黑碗 | 6→10 (0.037367) |
| 5 小碟上黑碗 | 5→35 (0.035952)；6→25 (0.037115)；10→49 (0.041379)；20→26 (0.039355)；43→3 (0.036552) |
| 7 炉上黑碗 | 11→3 (0.038710) |
| 8 盘旁黑碗 | 14→33 (0.034898)；31→11 (0.034349) |
| 30 字母汤＋番茄酱入篮 | 14→25 (0.061140)；37→2 (0.062984)；46→20 (0.062493) |
| 31 奶酪＋黄油入篮 | 36→35 (0.055921)；46→5 (0.052132) |
| 33 黑碗入底抽屉并关闭 | 40→36 (0.033254) |
| 34 两杯分别放两盘 | 41→43 (0.049471)；43→29 (0.049345) |
| 35 书放收纳盒后格 | 16→36 (0.042398) |
| 36 白杯放盘＋布丁放右侧 | 1→47 (0.042736)；2→20 (0.051167)；3→26 (0.043642)；12→38 (0.050629)；19→24 (0.042889)；38→2 (0.048401)；47→24 (0.048779) |
| 37 字母汤＋奶酪入篮 | 42→11 (0.040188) |
| 38 两摩卡壶放炉上 | 5→17 (0.060167)；6→8 (0.041997)；42→2 (0.053202)；48→45 (0.057115) |

20任务的A–B中位数与各自A–A/B–B中位数处于相近尺度。本次每任务50个B的最接近一对会在NN样本中出现两次，因此线性q01恰等于B–B最小NN；“第1百分位”不意味着flag率应为1%。32个flag的A/B字节均不同，最近距离为0.033254–0.062984，最大坐标差为0.014601–0.035415（raw混合单位），不是仅hash序列化或浮点末位不同。只能说存在分布内近邻，不能由该规则认定复制泄漏，也不能因零hash重叠宣称不存在任何语义近似。

## 4. Library、corrector、stall 来源链

逐一校验：原始parent READY与lane的study_inputs副本字节相同；size_500 VIEW/row_indices绑定同parent，选择覆盖全部库行且无重复；2,513个capture manifest来源引用SHA匹配；所有供fit读取的key/state/normalized_actions数组SHA匹配；10份fit.npz与READY绑定一致。正确器冻结的state/action/task/episode/step数组与所选B库逐元素相等，LOEO每个query/donor均属B且排除完整自身episode；五折train/heldout无交集、仅覆盖B500。不是只检查receipt里的“B-only”文字。

| 模型 | suite | C | corrector LOEO行 | stall校准窗口 | 参与校准的成功B episode | fit证据 |
|---|---|---:|---:|---:|---:|---|
| DP | libero_spatial | 5 | 12,028 | 4,056 | 436 | [READY](/data/mw_finetune/eval/csl10d/study_inputs/libero_spatial/C5/fit/READY.json:1) |
| DP | libero_spatial | 10 | 12,028 | 4,056 | 436 | [READY](/data/mw_finetune/eval/csl10d/study_inputs/libero_spatial/C10/fit/READY.json:1) |
| DP | libero_10 | 5 | 31,546 | 10,330 | 405 | [READY](/data/mw_finetune/eval/csl10d/study_inputs/libero_10/C5/fit/READY.json:1) |
| DP | libero_10 | 10 | 31,546 | 10,330 | 405 | [READY](/data/mw_finetune/eval/csl10d/study_inputs/libero_10/C10/fit/READY.json:1) |
| Fast-WAM | libero_spatial | 5 | 12,460 | 4,194 | 432 | [READY](/data/mw_finetune/eval/csl10f/study_inputs/libero_spatial/C5/fit/READY.json:1) |
| Fast-WAM | libero_spatial | 10 | 12,460 | 4,194 | 432 | [READY](/data/mw_finetune/eval/csl10f/study_inputs/libero_spatial/C10/fit/READY.json:1) |
| Fast-WAM | libero_10 | 5 | 27,806 | 11,350 | 462 | [READY](/data/mw_finetune/eval/csl10f/study_inputs/libero_10/C5/fit/READY.json:1) |
| Fast-WAM | libero_10 | 10 | 27,806 | 11,350 | 462 | [READY](/data/mw_finetune/eval/csl10f/study_inputs/libero_10/C10/fit/READY.json:1) |
| Cosmos | libero_spatial | 16 | 3,676 | 2,256 | 465 | [READY](/data/mw_finetune/eval/csl10c/study_inputs/libero_spatial/C16/fit/READY.json:1) |
| Cosmos | libero_10 | 16 | 8,408 | 6,757 | 479 | [READY](/data/mw_finetune/eval/csl10c/study_inputs/libero_10/C16/fit/READY.json:1) |

Corrector/PCA/metric输入被LibraryView限制为B的feature、state、action；test/dev分区会拒绝，成功标签不送入corrector。stall另用B的成功标签选轨迹：本次重算其bank_labels_sha256，与冻结的task/episode/step/controls/success完全相等；stall.pkl指纹、metric主码、模板/reference/calibration episode也逐一核验为本task的成功B轨迹，CSV窗口数与模型一致。[输入allowlist](/home/weiland/projects/openpi/exp/r17_integration/search/fit.py:11)；[corrector的LOEO](/home/weiland/projects/openpi/exp/r17_integration/common/layers/recipe.py:137)；[实际fit/stall构建](/home/weiland/projects/openpi/exp/r17_integration/ops/cache_study.py:180)。

实际配置使用48份封存YAML（Cosmos含别名），fitted_path/store_path/stall_path及digest都指向上述已核验副本；DP/Fast的ACK、plan及已观察per_step身份对应相同封存配置。运行路径为verified_fit加载不可变fit，不以A query重新训练。[冻结fit加载](/home/weiland/projects/openpi/exp/r17_integration/common/recipe.py:172)；[fit完整性检查](/home/weiland/projects/openpi/exp/r17_integration/search/cache_recipe.py:141)。

**需披露的协议差异：** OWNER_RULINGS写的是R6/Q3、B validation recording；实物全部写 `calibration_partition=B_library_successful_LOEO`、`independent_B_validation_recording=false`，实际在完整teacher B500的成功轨迹上做LOEO校准，沿用冻结B表征；并非另采独立B验证轨迹。这仍是B-only，未发现A污染，但不能把它表述为独立validation校准，也不是早版CACHE_STUDY的NoProgressGuard。阈值来源是Q3固定常数（error分位.90、advance分位.10等）和B残差分布，不是本次A上的SR调参。[owner gate定义](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/cache/OWNER_RULINGS.md:6)；[明确的校准范围](/home/weiland/projects/openpi/exp/r17_integration/ops/cache_study.py:210)；[Q3 provenance/固定常数](/home/weiland/projects/openpi/exp/offline_search/rounds/r06/ideation_Q3/stall/stall.py:196)。

## 5. 复核产物与结论边界

冻结输入共2,847个SHA，分析结束时再次一致；运行中逐步日志单独保存完整prefix的时间/长度/SHA，不要求整个增长文件静止。[审计JSON](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/cache/CONTAMINATION_CHECK.json:1)保存全部20任务/1000初态距离、六库/十fit结果与来源SHA；[逐A初态CSV](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/cache/CONTAMINATION_DISTANCES.csv:1)保存最近B编号、hash、距离及flag。

复跑（仓库根，只写上述分析JSON/CSV）：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m exp.offline_search.rounds.r17.cache.check_contamination`。[CPU审计脚本](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/cache/check_contamination.py:1)。Markdown是本次快照，不会自动追随运行进度。

此审计回答缓存初态和fit数据是否重用A；不审计基础模型预训练/示范数据。前轮A horizon选择也不能因此被称为独立测试复现，相关范围已在[选择复用说明](/home/weiland/projects/openpi/exp/offline_search/rounds/r17/cache/CACHE_STUDY.md:93)披露。

**一句话：当前是B库→A测试，未发现“用缓存自己的数据测试”的精确泄漏；保留32个近邻提示，并明确stall为B库LOEO校准、Cosmos执行证据尚待产生。**
