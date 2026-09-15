# online_rit — 结果报告

> 状态：**尚无实验数据**（代码 §4 完成，待 G2 → Verify → M1 离线检验 → M2 闭环）。本文件是报告骨架；
> 每一节在对应阶段产物落盘后填入，数值来源在括号中注明（实测 / 回放 / 参考计价）。

## 1. M0 资产核验
- 库快照完整性（`library_check.json`）：待填
- 尺度与 mask（`update_scales.npz` meta）：待填
- 结点（`knots.json`）：待填

## 2. M1 离线信号检验（Q1）
- 逐档 Spearman / 控制 s 的偏相关 / 分层 AUROC（`signal_check.json`）：待填
- `d_self` 地板与 parity：待填
- 放行结论：待填

## 3. M1 估计器回放
- held-out 覆盖率（E 按档 × 分位段 × 支持类型）、finite-q 份额、pinball（`replay.json`）：待填
- FM-0 vs FM-1 覆盖缺口与切点轨迹：待填
- 冷启动首个有限切点：待填

## 4. 成本台账与 δ 寻址
- `fb_batch_ms`（batch 1/2/3）、warm@0.875（`data/cost/<hw_mode>/cost.json`）：待填
- 可达目标集合与各目标 δ（online / R′ 两条回放）：待填

## 5. M2 闭环
- 主图：SR 对计价 IR（含/不含反馈），R′ / F / O-init / O-cold-frozen；R 作背景：待填
- 执行违规率（单侧 ≤ α）、决策前尾率、遮蔽档、连续 warm 串长：待填
- S3b 换库轴、FM-0 消融、独立重复：待填
- 结论与负结果：待填
