# exp/rit_loto/analysis

RIT 离线标定（LOTO）线的报告与图。设计与裁定见 `logs/rit_loto_calibration_plan.log.md`；运行数值产物在 `exp/rit_loto/data/<suite>/`（默认不入库，weilandserver `/data/libero_cache/rit_loto/<suite>/` 留权威副本，报告里记 sha256）。

| 文件 | 内容 |
|---|---|
| `data_inventory.md` | 任务 0：建库语料、失败轨迹、字段、主实验日志字段、设备盘点（2026-09-13 实测） |
| `smoke_evidence.md` | G2 证据：执行者历史岛上冒烟 + R2 owner 授权修复后的本地回归、最终代码 sha256 与未运行范围（advisory） |
| `results.md` | 任务 1–4 结果（运行后写） |
| `<suite>/equivalence.png` | 任务 2：LOTO 曲线 vs 原 shadow 曲线（含分层、ψ(s)、参考带） |
| `libero_10/verify_overlay.png` | 任务 3：闭环 (s, D) 点叠 LOTO 曲线 |

渲染脚本（`plot_equivalence.py` / `plot_verify.py`）按 owner 规则不入库；它们只读 `data/<suite>/{fits,compare,bootstrap_band,verify}.json`。
