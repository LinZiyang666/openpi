# Stage 2c — 危险步离线 join（libero_spatial）

oracle 危险 = deviate_score ≥ 5.0（keybuilder `spatial16_w8_d4`）；gate 信号取自 `cp1_spatial_pool_16__grid3_vision_0@6_vision_1@50_robot_state@43__d1__fh40_ws40_quantile`。**跨配置 proxy（R2）+ 轨迹发散（R3）**，结论定性 suggestive。

join 3254 步，危险 199（6.1%）。

**全程 AUC → danger（信号已定向为 higher=more danger）**

| signal | AUC |
|---|---|
| neg_prev_score | 0.521 |
| prev_is_MISS | 0.502 |
| neg_cp1_score | 0.544 |
| step_idx | 0.521 |

**早期相位切片（前 50% 步，R3 对齐更可信）**： join 1665 步，危险率 6.3%

| signal | AUC |
|---|---|
| neg_prev_score | 0.565 |
| prev_is_MISS | 0.509 |
| neg_cp1_score | 0.606 |
| step_idx | 0.638 |

及格线：若某廉价信号 AUC 显著偏离 0.5 → N4 注入可做危险步靶向的证据；否则记录“危险步不可廉价预测”。