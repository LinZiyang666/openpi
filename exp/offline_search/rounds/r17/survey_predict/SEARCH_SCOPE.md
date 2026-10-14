# 本次证据范围与检索记录

本文件为本次新增审计说明，不是实验结果。

- 仅使用现行 R17 文档、模型接入与正式 campaign 证据；同一文档内引用历史实现的段落不作为结果来源。
- 文本检索覆盖 exp/r17_integration 与 exp/offline_search/rounds/r17；关键词包括 horizon、replan、commit、denoising、d_1、difference、deviation、predict、predictor、MSE、quality、proxy、LOEO、先验、预测、减步、长执行段。
- 检索后人工区分：实际实验结果、实现正确性/单元测试、计划但未执行、文献调研、历史结果转述。模型名称 Cosmos Predict 中的 predict、网络 proxy、计算成本预测都不是 SR 预测器。
- 重点核对 horizon/{STUDY,SUMMARY,RESULTS_DP_FASTWAM,COSMOS_K2}.md 的当前实验部分、cache/{CACHE_STUDY,CACHE_RESULTS,CONTAMINATION_CHECK}.md、knob/REPORT.md，以及 R17 handbacks/review/models/search/qualification/ops。
- 实际质量代理定位到 DP 的 review/repro_B、review/repro_B2、models/dp/quality.py 与 HANDBACK_P4.md；正文只引用修复后的当前 sampler 结果。未发现其他模型在现行记录中有预测 K/H_exec 安全边界、再对照闭环 SR 验证预测的实验。
- search/analyze.py 的 offline_action_scores/offline_sweep 有实现和合成单测，但未检出用其验证各模型安全 K/H_exec 的真实数据结果表。缓存 LOEO、停滞判别器、IR 回放各有其他目标，不能替代该验证。
- 扫描 /data/mw_finetune/eval 下 campaign/SUMMARY 的模型、domain、步数、commit、完成状态和 suite，选取完整正式 ladder 与对照；smoke、同设置早期小样本和未齐纯策略臂不池化为 ladder。
- MT50_AUDIT.md、LIBERO_AUDIT.md 是新增 journal 计数与身份核对；MT50_CACHE_AUDIT.md 是新增缓存 journal 描述统计。原始 accepted 成功与正常失败均保留，基础设施错误排除。后者严格按续跑 retained_task_uids 继承；未齐 cell 不报 SR。
- JSON 附件保存输入 SHA256、逐臂来源或纳入行号；缓存正在追加的 journal 按读入字节快照计算 SHA256，时间在 mt50_cache_audit.json 的 snapshot_utc。没有重新执行预注册分析，没有读取凭据或调用 GPU/环境/远程 campaign。

“未找到”是以上限定记录内的检索结论，不是理论不存在性证明，也不是对未留档工作的断言。
