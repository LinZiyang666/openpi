# R17 formal A 逐格完整性对账（只读，2026-10-22 18:4x CDT）

## 口径

- **计划从哪来**：
  - `formal_a_1016/QUEUE.json`：只取仍然有效的 Cosmos 6 个单位和 π0.5 纯推理参照 2 个。DP/Fast-WAM 的 `fa_dp_*`/`fa_fw_*` 已被 `longseg_1017/NAMES.json` 撤回，改由 `FORMAL_QUEUE.json` 接替；π0.5/GR00T 的旧门单位已被 Q3 移植取代。
  - `longseg_1017/CAMPAIGNS.json`（60 个）+ `FORMAL_QUEUE.json`（8 个各档单位）。
  - `q3_port_1020/QUEUE.json`（18 个单位）。
  - `cache_study_1019/CAMPAIGNS.json`（`fa_cs_pi_r`、`fa_cs_gr_r`）。
- **R17 批次**：从末段沿 `CACHE_RESUME.json` 的 `source` 往回走（和 campaign 里 `cache_resume_selection.source` 一致）。逐行读每段的 `driver/journal.jsonl`，规则同 `analyze_cache.resume_segments` + `accepted_terminal`：
  - 只要 `accepted: true` 的行；
  - 丢掉 `FatalEpisodeError … retry budget exhausted` 的行；
  - 按 `task_uid` 去重，ts 晚的覆盖早的。
  - `status` 的 `done` 和 `failed` 都算干净局（`failed` 只表示任务没成功）。
  - 再用 grep 取首段 `EPISODE_PLAN.json` 里的全部 `task_uid`，算「缺 / 多」。表里的「干净局」都是 **缺 0、多 0** 的唯一局数，例外会单独写出。
- **R15（无回执）**：逐个读 `runs/<arm>/client/journal.jsonl`，并与 `runs/chain.log` 的 `ARM_DONE complete=… success=…` 对照，每 arm 应为 500。
- **R16 group（无回执）**：读顶层 `journal.jsonl`，按 `yaml_id` 对照 `group.json` 的 entries，每个 entry 应为 500，并核对 `DONE.json`。
- **全程只读**：单进程。没读 `per_step.jsonl`，没碰任何进程，也没用 tether/ssh。

## LIBERO

| 实验 | 计划局 | 干净接受局 | 最终链（旧→新） | 末段回执 | 判定 |
|---|---:|---:|---|---|---|
| 原库 · DP · LIBERO · 减步 · 旋钮关 | 6,000 | 6,000 | fa_ls_dp_r_u_o_l | PASS | 完整 |
| 成功过滤库 · DP · LIBERO · 减步 · 旋钮关 | 6,000 | 6,000 | fa_ls_dp_r_f_o_l | PASS | 完整 |
| 原库 · DP · LIBERO · 全步 · 旋钮关 | 6,000 | 6,000 | fa_ls_dp_f_u_o_l | PASS | 完整 |
| 成功过滤库 · DP · LIBERO · 全步 · 旋钮关 | 6,000 | 6,000 | fa_ls_dp_f_f_o_l | PASS | 完整 |
| 原库 · DP · LIBERO · 减步 · 各档 | 5,000 | 5,000 | fa_ls_dp_r_u_l_l | PASS | 完整 |
| 成功过滤库 · DP · LIBERO · 减步 · 各档 | 5,000 | 5,000 | fa_ls_dp_r_f_l_l | PASS | 完整 |
| 原库 · DP · LIBERO · 全步 · 各档 | 5,000 | 5,000 | fa_ls_dp_f_u_l_l | PASS | 完整 |
| 成功过滤库 · DP · LIBERO · 全步 · 各档 | 7,500 | 7,500 | fa_ls_dp_f_f_l_l | PASS | 完整 |
| DP · LIBERO · 减步 · 纯推理参照 | 1,000 | 1,000 | fa_ls_dp_r_b_l | PASS | 完整 |
| DP · LIBERO · 全步 · 纯推理参照 | 1,000 | 1,000 | fa_ls_dp_f_b_l | PASS | 完整 |
| DP · LIBERO · 减步 · 纯推理+长执行段对照 | 1,000 | 1,000 | fa_ls_dp_r_p_l | PASS | 完整 |
| DP · LIBERO · 全步 · 纯推理+长执行段对照 | 1,000 | 1,000 | fa_ls_dp_f_p_l | PASS | 完整 |
| 原库 · Fast-WAM · LIBERO · 减步 · 旋钮关 | 6,000 | 6,000 | fa_ls_fw_r_u_o_l | PASS | 完整 |
| 成功过滤库 · Fast-WAM · LIBERO · 减步 · 旋钮关 | 6,000 | 6,000 | fa_ls_fw_r_f_o_l | PASS | 完整 |
| 原库 · Fast-WAM · LIBERO · 全步 · 旋钮关 | 6,000 | 6,000 | fa_ls_fw_f_u_o_l | PASS | 完整 |
| 成功过滤库 · Fast-WAM · LIBERO · 全步 · 旋钮关 | 6,000 | 6,000 | fa_ls_fw_f_f_o_l → lc1 → lc2 → lc3 → lc4 → lc5 → lc7（lc6 只准备未启动） | PASS | 完整（链内 1,982 行重复，见注 1） |
| 原库 · Fast-WAM · LIBERO · 减步 · 各档 | 30,000 | 30,000 | fa_ls_fw_r_u_l_l(11,341) → lc1(18,659) | PASS | 完整 |
| 成功过滤库 · Fast-WAM · LIBERO · 减步 · 各档 | 25,000 | 25,000 | fa_ls_fw_r_f_l_l(16,168) → lc1(8,832) | PASS | 完整 |
| 原库 · Fast-WAM · LIBERO · 全步 · 各档 | 30,000 | 30,000 | fa_ls_fw_f_u_l_l | PASS | 完整 |
| 成功过滤库 · Fast-WAM · LIBERO · 全步 · 各档 | 30,000 | 30,000 | fa_ls_fw_f_f_l_l(12,545，丢 41 局 infra) → lc1(17,455) | PASS | 完整 |
| Fast-WAM · LIBERO · 全步 · 纯推理参照 | 1,000 | 1,000 | fa_ls_fw_f_b_l | PASS | 完整 |
| Fast-WAM · LIBERO · 减步 · 纯推理+长执行段对照 | 1,000 | 1,000 | fa_ls_fw_r_p_l | PASS | 完整 |
| Fast-WAM · LIBERO · 全步 · 纯推理+长执行段对照 | 1,000 | 1,000 | fa_ls_fw_f_p_l | PASS | 完整 |
| 原库 · Cosmos · LIBERO · 减步 · 旋钮关（formal A 复用） | 6,000 | 6,000 | ffk1015_cosmos_libero_reduced_off_u | PASS | 完整 |
| 成功过滤库 · Cosmos · LIBERO · 减步 · 旋钮关（复用） | 6,000 | 6,000 | ffk1015_cosmos_libero_reduced_off | PASS | 完整 |
| 原库 · Cosmos · LIBERO · 全步 · 旋钮关 | 6,000 | 6,000 | fa_co_f_u_o | FAIL（finalization：`No module named exp.r17_integration.ops.direct_transfer`） | 数据完整但回执未封 |
| 成功过滤库 · Cosmos · LIBERO · 全步 · 旋钮关 | 6,000 | 6,000 | fa_co_f_f_o(2,985) → oc1(3,015) | PASS | 完整 |
| 原库 · Cosmos · LIBERO · 减步 · 各档 | 30,000 | 30,000 | fa_co_r_u_l | PASS | 完整 |
| 成功过滤库 · Cosmos · LIBERO · 减步 · 各档 | 30,000 | 30,000 | fa_co_r_f_l → lc1 → … → lc8 | FAIL（native CPU evidence aggregate is incomplete） | 数据完整但回执未封（链内 4,191 行重复，见注 1） |
| 原库 · Cosmos · LIBERO · 全步 · 各档 | 30,000 | 30,000 | fa_co_f_u_l | PASS | 完整 |
| 成功过滤库 · Cosmos · LIBERO · 全步 · 各档 | 30,000 | 30,000 | fa_co_f_f_l(29,978) → lc1(22) | PASS | 完整 |
| 原库 · π0.5 · LIBERO · 全步 · Q3 旋钮关 | 4,000 | 4,000 | fa_pi_f_u_o_q_{l50…s500}（8 arm×500） | 无回执（R15） | 完整 |
| 成功过滤库 · π0.5 · LIBERO · 全步 · Q3 旋钮关 | 4,000 | 4,000 | fa_pi_f_f_o_q_*（8×500） | 无回执（R15） | 完整 |
| 原库 · GR00T · LIBERO · 全步 · Q3 旋钮关 | 4,000 | 4,000 | fa_gr_f_u_o_q_*（8×500） | 无回执（R15） | 完整 |
| 成功过滤库 · GR00T · LIBERO · 全步 · Q3 旋钮关 | 4,000 | 4,000 | fa_gr_f_f_o_q_*（8×500） | 无回执（R15） | 完整 |
| 原库 · π0.5 · LIBERO · 减步 · Q3 旋钮关 | 4,000 | 4,000 | fa_pi_r_u_o_q（8 entry） | DONE.json（R16） | 完整 |
| 成功过滤库 · π0.5 · LIBERO · 减步 · Q3 旋钮关 | 4,000 | 4,000 | fa_pi_r_f_o_q | DONE.json | 完整 |
| 原库 · GR00T · LIBERO · 减步 · Q3 旋钮关 | 4,000 | 4,000 | fa_gr_r_u_o_q | DONE.json | 完整 |
| 成功过滤库 · GR00T · LIBERO · 减步 · Q3 旋钮关 | 4,000 | 4,000 | fa_gr_r_f_o_q | DONE.json | 完整 |
| 原库 · π0.5 · LIBERO · 全步 · Q3 各档 | 16,000 | 16,000 | fa_pi_f_u_l_q_*（8 格×4 档=32 arm×500） | 无回执（R15） | 完整 |
| 成功过滤库 · π0.5 · LIBERO · 全步 · Q3 各档 | 16,000 | 16,000 | fa_pi_f_f_l_q_*（32×500） | 无回执（R15） | 完整 |
| 原库 · π0.5 · LIBERO · 减步 · Q3 各档 | 16,000 | 16,000 | fa_pi_r_u_l_q（32 entry） | DONE.json | 完整 |
| 成功过滤库 · π0.5 · LIBERO · 减步 · Q3 各档 | 16,000 | 16,000 | fa_pi_r_f_l_q（32 entry） | DONE.json | 完整 |
| 原库 · GR00T · LIBERO · 减步 · Q3 各档 | 10,000 | 10,000 | fa_gr_r_u_l_q（20 entry；中途 4,0 重启） | DONE.json | 完整 |
| 成功过滤库 · GR00T · LIBERO · 减步 · Q3 各档 | 10,000 | 10,000 | fa_gr_r_f_l_q（20 entry） | DONE.json | 完整 |
| 原库 · GR00T · LIBERO · 全步 · Q3 各档 | 12,000 | 12,000 | fa_gr_f_u_l_q_{l50,l100,l200,s50,s100,s500}（24×500） | 无回执（R15） | 完整 |
| 成功过滤库 · GR00T · LIBERO · 全步 · Q3 各档 | 12,000 | 12,000 | fa_gr_f_f_l_q_{l50,l100,l200,s50,s100,s200}（24×500） | 无回执（R15） | 完整 |
| 原库 · π0.5 · LIBERO · 减步 · 缓存部件四组（c/cr/ch/chr） | 16,000 | 16,000 | fa_cs_pi_r（32 entry） | DONE.json | 完整 |
| 原库 · GR00T · LIBERO · 减步 · 缓存部件四组 | 16,000 | 16,000 | fa_cs_gr_r | DONE.json | 完整 |
| 原库 · π0.5 · LIBERO · 减步 · 缓存部件带门四组 | 16,000 | 16,000 | fa_cs_pi_rg | DONE.json | 完整 |
| 原库 · GR00T · LIBERO · 减步 · 缓存部件带门四组 | 16,000 | 16,000 | fa_cs_gr_rg | DONE.json | 完整 |
| π0.5 · LIBERO · 全步 · 纯推理参照（H100，spatial / l10） | 1,000 | 1,000 | fa_pi_f_ps、fa_pi_f_p10 | 无回执（R15） | 完整 |
| 原库 · DP · LIBERO · 全步 · 缓存变体 v04_0 | 4,001 | 4,001 | fa_ls_v04_0_j3（pod） | PASS | 完整 |
| 原库 · DP · LIBERO · 全步 · 缓存变体 v04_1 | 513 | 513 | fa_ls_v04_1 | PASS | 完整 |
| 原库 · DP · LIBERO · 全步 · 缓存变体 v04_2 | 1,734 | 1,734 | fa_ls_v04_2 | FAIL（finalization：audit_batches 命令失败） | 数据完整但回执未封 |
| 原库 · DP · LIBERO · 全步 · 缓存变体 v04_3 | 1,752 | 1,752 | fa_ls_v04_3_j2（pod） | PASS | 完整 |
| 原库 · Fast-WAM · LIBERO · 全步 · 缓存变体 v05_0…v05_4 | 4,023 / 288 / 397 / 347 / 2,945 | 同左（各缺 0） | fa_ls_v05_0 … v05_4 | 全 PASS | 完整 |
| 原库 · DP · LIBERO · 全步 · 缓存变体 v06_0 | 32 | 32 | fa_ls_v06_0_j2（pod） | FAIL（证据整理在 pod 上 tether exec 失败） | 数据完整但回执未封 |
| 原库 · Fast-WAM · LIBERO · 全步 · 缓存变体 v07_0（旧 csmf1010） | 32 | **0** | fa_ls_v07_0（10-18 19:04 reset 后一直 PREPARED，从没启动） | 无 | **真缺口** |
| 原库 · Fast-WAM · LIBERO · 全步 · 缓存变体 v08_1 | 40,000 | 38,500（18:4x） | fa_ls_v08_1 → c1 → … → c6（都是 0 局 FAIL）→ c7 | c7 在跑（目前的 RECEIPT 是 12:40 写的 PARTIAL 占位） | 在跑 |
| 原库 · DP · LIBERO · 全步 · 缓存变体 v09_0 | 6,350 | 6,350 | fa_ls_v09_0_j4（pod） | FAIL（pod 证据 tether exec 失败） | 数据完整但回执未封 |
| 原库 · DP · LIBERO · 全步 · 缓存变体 v09_1 | 33,650 | 33,650 | fa_ls_v09_1_j4（pod） | PASS | 完整 |
| 成功过滤库 · DP · LIBERO · 减步 · 缓存变体 v10_0 | 6,000 | 6,000 | fa_ls_v10_0 | PASS | 完整 |
| 成功过滤库 · Fast-WAM · LIBERO · 减步 · 缓存变体 v11_0 | 6,000 | 6,000 | fa_ls_v11_0 | PASS | 完整 |
| 原库 · DP · LIBERO · 减步 · 缓存变体 v12_0 | 6,000 | 6,000 | fa_ls_v12_0 | PASS | 完整 |
| 原库 · Fast-WAM · LIBERO · 减步 · 缓存变体 v13_0 | 6,000 | 6,000 | fa_ls_v13_0 | PASS | 完整 |

- LIBERO 合计计划 661,564 局，干净 660,032 局。
- 差额 1,532 局：v07_0 缺 32 局；v08_1c7 还剩 1,500 局，正在跑。
- R15 共 146 个 arm，每个都是 500/500，与 `ARM_DONE` 完全一致。

## MetaWorld（MT50）

| 实验 | 计划局 | 干净接受局 | 最终链（旧→新） | 末段回执 | 判定 |
|---|---:|---:|---|---|---|
| 原库 · DP · MetaWorld · 减步 · 旋钮下降 | 90,000 | 90,000 | fa_ls_dp_r_u_m_j(42,553，PARTIAL) → jc1(47,447) | FAIL（native CPU evidence aggregate is incomplete） | 数据完整但回执未封 |
| 成功过滤库 · DP · MetaWorld · 减步 · 旋钮下降 | 90,000 | 90,000 | fa_ls_dp_r_f_m_j(15,123) → jc2(74,877)（jc1 没启动） | FAIL（同上） | 数据完整但回执未封 |
| 原库 · DP · MetaWorld · 全步 · 旋钮下降 | 90,000 | 90,000 | fa_ls_dp_f_u_m_j（pod） | PASS | 完整 |
| 成功过滤库 · DP · MetaWorld · 全步 · 旋钮下降 | 90,000 | 90,000 | fa_ls_dp_f_f_m_j（pod） | PASS | 完整 |
| DP · MetaWorld · 减步 · 纯推理参照（pod 同卡） | 2,500 | 2,500 | fa_ls_dp_r_b_m_j | FAIL（evidence aggregate） | 数据完整但回执未封 |
| DP · MetaWorld · 全步 · 纯推理参照（pod 同卡） | 2,500 | 2,500 | fa_ls_dp_f_b_m_j | PASS | 完整 |
| DP · MetaWorld · 减步 · 纯推理+长执行段（pod） | 2,500 | 2,500 | fa_ls_dp_r_p_m_j | PASS | 完整 |
| DP · MetaWorld · 全步 · 纯推理+长执行段（pod） | 2,500 | 2,500 | fa_ls_dp_f_p_m_j | PASS | 完整 |
| DP · MetaWorld · 减步 · 纯推理参照（缓存研究迁移用，4090） | 2,500 | 2,500 | fa_ls_dp_r_b_v | FAIL（evidence aggregate） | 数据完整但回执未封 |
| 原库 · GR00T · MetaWorld · 减步 · 旋钮下降 | 90,000 | 90,000 | fa_ls_gr_r_u_m | PASS | 完整 |
| 成功过滤库 · GR00T · MetaWorld · 减步 · 旋钮下降 | 90,000 | 90,000 | fa_ls_gr_r_f_m(34,810) → mc1(55,190) | FAIL（evidence aggregate） | 数据完整但回执未封 |
| 原库 · GR00T · MetaWorld · 全步 · 旋钮下降 | 90,000 | 90,000 | fa_ls_gr_f_u_m_j(1,307 干净 + 3,693 infra 已丢) → jc1(88,693)（pod） | PASS | 完整 |
| 成功过滤库 · GR00T · MetaWorld · 全步 · 旋钮下降 | 90,000 | 90,000 | fa_ls_gr_f_f_m_j（pod） | PASS | 完整 |
| GR00T · MetaWorld · 减步 · 纯推理+长执行段 | 2,500 | 2,500 | fa_ls_gr_r_p_m(939) → _c2(1,561) | FAIL（evidence aggregate） | 数据完整但回执未封 |
| GR00T · MetaWorld · 全步 · 纯推理+长执行段 | 2,500 | 2,500 | fa_ls_gr_f_p_m（H100） | FAIL（evidence aggregate） | 数据完整但回执未封（见注 3） |
| 原库 · Fast-WAM · MetaWorld · 减步 · 旋钮下降 | 90,000 | 90,000 | fa_ls_fw_r_u_m_j(23,715 + 66,285 infra 已丢) → jc2(66,285)（jc1 0 局） | PASS | 完整 |
| 成功过滤库 · Fast-WAM · MetaWorld · 减步 · 旋钮下降 | 90,000 | 90,000 | fa_ls_fw_r_f_m_j（pod 全新跑；H100 的 fa_ls_fw_r_f_m/mc1/mc2 不用） | PENDING_EVIDENCE，all_owned_closed=False | 数据完整但回执未封 |
| 原库 · Fast-WAM · MetaWorld · 全步 · 旋钮下降 | 90,000 | 90,000 | fa_ls_fw_f_u_m(86,145 + 3,855 EMFILE infra 已丢) → mc1(3,855) | PASS | 完整 |
| 成功过滤库 · Fast-WAM · MetaWorld · 全步 · 旋钮下降 | 90,000 | 90,000 | fa_ls_fw_f_f_m_j(7,505 + 82,495 infra 已丢) → jc2(82,495)（jc1 0 局） | PASS | 完整 |
| Fast-WAM · MetaWorld · 减步 · 纯推理+长执行段（pod 同卡） | 2,500 | 2,500 | fa_ls_fw_r_p_m_j | PASS | 完整 |
| Fast-WAM · MetaWorld · 全步 · 纯推理+长执行段（pod 同卡） | 2,500 | 2,500 | fa_ls_fw_f_p_m_j | PASS | 完整 |
| 原库 · DP · MetaWorld · 减步 · 缓存变体 v01_0 | 120,000 | 120,000 | fa_ls_v01_0(12,500) → c1(107,500) | PASS | 完整 |
| 原库 · Fast-WAM · MetaWorld · 减步 · 缓存变体 v02_s | 60,000 | 60,000 | fa_ls_v02_s(2,942) → c1(6,950) → c2(49,521) → c3(587) | FAIL（evidence aggregate） | 数据完整但回执未封 |
| 原库 · Fast-WAM · MetaWorld · 减步 · 缓存变体 v02_b | 60,000 | 60,000 | fa_ls_v02_b | PASS | 完整 |
| 原库 · GR00T · MetaWorld · 减步 · 缓存变体 v03_0 | 24,406 | 24,406 | fa_ls_v03_0 | PASS | 完整 |
| 原库 · GR00T · MetaWorld · 减步 · 缓存变体 v03_1 | 95,594 | 95,594 | fa_ls_v03_1(30,516) → c1(65,078) | PASS | 完整 |
| 原库 · DP · MetaWorld · 全步 · 缓存变体 v14_0 | 8 | 8 | fa_ls_v14_0 | FAIL（evidence aggregate） | 数据完整但回执未封 |
| π0.5 · MetaWorld · 旋钮下降（ffk1015_pi05_mt50_* / pimk1_*，formal A 之外，longseg 没撤回） | 4×90,000 | 只看了回执 | 见注 5 | 见注 5 | 不在计划（未逐局核实） |

- MetaWorld 计划合计 1,462,508 局，干净局 1,462,508。
- 12 个旋钮下降格每格都是 90,000，缺 0、多 0。

## 每个非 PASS 名字族的归类

### A. 数据完整但回执未封（15 个末段）

失败都发生在收尾环节，不是局数据问题。下面「证据聚合」指 `finalization_error = native CPU evidence aggregate is incomplete`。每个末段的 `driver_completion.accepted_episodes` 都等于计划，`error_attempts` = 0。

| 末段 | 失败原因 | 链总局数 |
|---|---|---|
| fa_co_f_u_o | `ModuleNotFoundError exp.r17_integration.ops.direct_transfer` | 6,000/6,000 |
| fa_co_r_f_lc8 | 证据聚合 | 30,000/30,000 |
| fa_ls_dp_r_u_m_jc1 | 证据聚合 | 90,000 |
| fa_ls_dp_r_f_m_jc2 | 证据聚合 | 90,000 |
| fa_ls_gr_r_f_mc1 | 证据聚合 | 90,000 |
| fa_ls_fw_r_f_m_j | PENDING_EVIDENCE，all_owned_closed=False（timan1 当时被他人压死），证据审计没起 | 90,000，0 错 |
| fa_ls_dp_r_b_m_j | 证据聚合 | 2,500 |
| fa_ls_dp_r_b_v | 证据聚合 | 2,500 |
| fa_ls_gr_r_p_m_c2 | 证据聚合 | 2,500 |
| fa_ls_gr_f_p_m | 证据聚合 | 2,500 |
| fa_ls_v02_s_c3 | 证据聚合 | 60,000 |
| fa_ls_v14_0 | 证据聚合 | 8 |
| fa_ls_v04_2 | audit_batches 命令失败 | 1,734 |
| fa_ls_v06_0_j2 | 证据整理器在 pod 上 tether exec 失败 | 32 |
| fa_ls_v09_0_j4 | 同上（handoff 也写过：snapshot 不到 run manifest） | 6,350 |

### B. 中间段非 PASS、但同链末段 PASS（续跑链，归为完整）

| 中间段 | 回执 / 原因 | 接手的末段 |
|---|---|---|
| fa_co_f_f_l | FAIL（driver incomplete，96 行未接受重试） | lc1 PASS |
| fa_co_f_f_o | PARTIAL（hard deadline） | oc1 PASS |
| fa_ls_fw_f_f_l_l | PARTIAL（OOM 后手停，41 局 infra） | lc1 PASS |
| fa_ls_fw_r_u_l_l | PARTIAL | lc1 PASS |
| fa_ls_fw_r_f_l_l | PARTIAL | lc1 PASS |
| fa_ls_fw_f_f_o_l、lc1–lc5 | PARTIAL | lc7 PASS；lc6 无 journal，只准备未启动 |
| fa_ls_fw_f_u_m | FAIL，3,855 局 EMFILE infra | mc1 PASS |
| fa_ls_fw_f_f_m_j | FAIL，82,495 局 pod 掉线 infra | jc2 PASS |
| fa_ls_fw_r_u_m_j | FAIL，66,285 局 pod 掉线 infra | jc2 PASS |
| fa_ls_fw_f_f_m_jc1、fa_ls_fw_r_u_m_jc1 | FAIL，0 局（源 assets 的 READY.json 不在），已作废 | jc2 |
| fa_ls_gr_f_u_m_j | PARTIAL，3,693 infra | jc1 PASS |
| fa_ls_v01_0 | FAIL | c1 PASS |
| fa_ls_v03_1 | PARTIAL，20 infra | c1 PASS |
| fa_ls_dp_r_u_m_j、fa_ls_dp_r_f_m_j、fa_ls_gr_r_f_m、fa_ls_gr_r_p_m、fa_co_r_f_l、lc1–lc7、fa_ls_v02_s、c1、c2 | PARTIAL/FAIL | 各自的末段已列在 A 类（数据完整） |

### C. 被取代 / 改名，数据不用或本来就没有

**同卡副本取代的非末段**（局数据完整，但配对用 pod 的 `_j` 副本）：

| 被取代 | 回执 | 取代者 |
|---|---|---|
| fa_ls_dp_r_b_m（4090） | FAIL 证据聚合，2,500 局完整 | _j |
| fa_ls_dp_f_b_m | FAIL，2,500 | fa_ls_dp_f_b_m_j PASS |
| fa_ls_dp_r_p_m | FAIL，2,500 | _j PASS |
| fa_ls_dp_f_p_m | FAIL，2,500 | _j PASS |
| fa_ls_fw_r_p_m（H100） | FAIL，2,500 | _j PASS |
| fa_ls_fw_f_p_m（H100） | FAIL，2,500 | _j PASS |

**H100 上作废的批次**：fa_ls_fw_r_f_m（PARTIAL 453）+ mc1（FAIL：replica_0 exited，1,537 干净 + 510 infra）+ mc2（没启动），由 pod 全新的 fa_ls_fw_r_f_m_j 取代。

**pod 副本的源名**：下列名字没有回执也没有 journal，真正的局在它们的 `_j`/`_j2`/`_j3`/`_j4` 里：

- fa_ls_dp_r_u_m、dp_r_f_m、dp_f_u_m、dp_f_f_m
- fa_ls_gr_f_u_m、gr_f_f_m
- fa_ls_fw_r_u_m、fw_f_f_m
- fa_ls_v04_0、v04_3、v06_0、v09_0、v09_1

**从没启动的续跑**：fa_ls_dp_r_f_m_jc1（74,877）、fa_ls_fw_r_f_mc2（88,010）。

**撤销的认领**：fa_ls_v08_1_j，pod 认领后撤销，无数据。

**longseg 撤回的旧 formal A DP/Fast-WAM**：

- 名单：fa_dp_f_f_o、fa_dp_f_u_o、fa_dp_r_f_l、fa_dp_r_u_l、fa_fw_f_f_o、fa_fw_f_u_o、fa_fw_r_u_l、fa_fw_r_u_lc1、fa_fw_r_f_l（fa_fw_r_f_l 不在 NAMES，但它从没启动，由 fa_ls_fw_r_f_l_l 接替）。
- 依据：`longseg_1017/NAMES.json` 的 old_to_new 指向 `FORMAL_QUEUE.json`；原因是决策头时钟问题，journal 已移到 `/home/weiland/projects/_trash_longseg_20261017`。
- 现状：目录里只剩定义，没有 RECEIPT 和 journal。

**被 Q3 移植取代的旧门 π0.5/GR00T formal A 单位**：

- 名单：fa_pi_r_u_c、fa_pi_r_f_l、fa_pi_r_u_l、fa_pi_f_*_o、fa_pi_f_*_l、fa_gr_r_u_c、fa_gr_r_*_l、fa_gr_f_*_o、fa_gr_f_*_l。
- 依据：`q3_port_1020/HANDBACK.md` 第 6.9 条「旧结论作废」；这些 eval 目录已不在。

**改短名前的长名目录**：ffk1015_*_formal_A_*（23 个），都只有定义，0 局。另有 ffk1015_{dp,fastwam}_libero_reduced_off 两个：被 longseg 撤回、没有回执，分别改由 v10_0、v11_0 重跑。

**备份 / 改名目录**（目录名带点后缀，共 29 个）：

- `.before_*`：fa_ls_v05_1/2 ×3、fa_ls_fw_r_p_m ×3、fa_ls_v07_0.before_localentry_1018。
- `.unused_*`：v04_0_j、v04_3_j、v06_0_j、v09_0_j、v09_1_j、dp_f_p_m_j、dp_f_u_o_l_j。
- `.failed_*`：v04_0_j2、v09_0_j2/j3、v09_1_j2/j3。
- `.incomplete_*`：fa_ls_fw_f_u_m、fa_ls_dp_r_f_l_l。
- `.cpu_validation_only_*`：v01_b、v01_s。
- 其他：`.six_infra_only_*`（fa_fw_r_u_lc1）、`.before_gpu2_counts_*`（fa_co_f_f_oc1）、`.retry_build_*`（fa_ls_fw_r_p_m）。

### D. 作废探针 / 非实验目录

- **fa_ls_t_62d4d702、7ab163cc、a2cd2e01、ca7540fd、d0fb8429**：DP longseg 的源模板（只有 campaign/runtime/workers.tar，0 局），是 fa_ls_dp_*_m*、v01_0、dp_r_b_v 的 FFK_DEFINITION `source`。
- **fa_ls_fw_f_b_l_j**：Fast-WAM LIBERO 上 pod 的探针，16 GiB 切片 OOM，FAIL，23 干净 / 52 infra。H100 上的 fa_ls_fw_f_b_l PASS 1,000 已覆盖这一格。
- **fa_ls_fw_f_p_l_j**：没启动。
- **fa_asv_r15_pi05_1018、fa_asv_r17_fw_1018**：10-18 autoscale 验证目录。

### E. 无回执但这是设计如此（R15/R16），逐 arm 已核完整

- R15：
  - fa_{pi,gr}_f_{u,f}_o_q_* 共 32 个目录；
  - fa_{pi,gr}_f_{u,f}_l_q_* 共 28 个目录；
  - fa_pi_f_ps、fa_pi_f_p10。
  - 都有 `state/CHAIN.DONE`，合计 146 arm × 500。
- R16：
  - fa_{pi,gr}_r_{u,f}_{o,l}_q 共 8 个；
  - fa_cs_{pi,gr}_{r,rg} 共 4 个。
  - 都有 DONE.json，每个 entry 500。

### F. 在跑

- fa_ls_v08_1（c7）。

### G. 真缺口

- fa_ls_v07_0。

## 真缺口

1. **原库 · Fast-WAM · LIBERO · 全步 · 缓存变体 v07_0（fa_ls_v07_0，旧名 csmf1010）：计划 32 局，干净 0 局。**
   - 目录在 10-18 19:04 vfix reset 成 PREPARED，之后没有 journal、没有 RECEIPT。
   - 它排在 h100-temp 链 tmp7–tmp14 的列表里。h100-temp 10-19 01:56 被外部结束后，余下批次移到 h100m20 时**漏掉了它**（handoff 第 783 行的名单里没有 v07_0），之后再没进任何链。
   - 对应的 DP 同类 v06_0 已跑（_j2，32 局）。

除此之外：

- **在跑**：fa_ls_v08_1c7（38,500/40,000）。
- **其余所有计划格**：干净接受局都等于计划局，逐 `task_uid` 缺 0、多 0。

## 注

1. **链内重复局**。去重后计数没问题，但同一局有两次结果，数字会受影响：
   - **fa_co_r_f 链**：4,191 行重复，3,575 个 uid 最终取了 lc3/lc4/lc5 的结果，23 个前后结果不一致。
   - **fa_ls_fw_f_f_o 链**：1,982 行重复，1,890 个 uid 取 lc4/lc5 的结果，115 个不一致。
   - 原因（推断，未核实）：lc2–lc5 的续跑计划只保留了直接上一段，把更早已接受的局又跑了一遍；lc6/lc7 的计划又把最早段算作保留。
   - `analyze_cache.accepted_terminal` 取 ts 最晚的那条，即 lc3–lc5 的结果。这和续跑计划声明的「保留」不一致，是否改口径请协调者定。
   - 其他链没有重复（每段 new == accepted）。
2. **库标签以 FFK_DEFINITION 为准**（`unfiltered=true` = 原库）：
   - v01_0、v02_s、v09_1 都是 `unfiltered=true`。handoff 里写成「成功过滤库」的几处（10-20 09:4x 的 v01_0、10-21 的 v02_s、v09_1_j4）与定义不符；v10_0、v11_0 才是成功过滤库。
   - Cosmos 复用的旋钮关：`ffk1015_cosmos_libero_reduced_off` = 成功过滤库，`_off_u` = 原库。
3. **GR00T · MetaWorld · 全步两批的配对（未核实是否需要）**：两批（gr_f_u_m_j/jc1、gr_f_f_m_j）在 pod H200 MIG 上跑，但 GR00T 全步的纯推理+长执行段对照 fa_ls_gr_f_p_m 和纯推理参照在 H100 上。DP、Fast-WAM 的 pod 批次都补了 pod 同卡参照，GR00T 没有。这不是计划缺口；是否需要同卡参照请协调者判断。
4. **progress.json 计数和 journal 不一致**：fa_gr_r_u_l_q、fa_gr_r_f_l_q、fa_cs_pi_r、fa_cs_gr_r 的 `progress.episodes_done` 比 journal 里 success=true 的数多（例：9,438 vs 8,763），done+failed 仍等于总数。推测是 runner 重启后把续上的局都记成 done（未核实）。journal 本身 10,000/16,000 唯一、全部 accepted、0 错，以 journal 为准。
5. **范围外（只看了回执）**：
   - 成功过滤库 · π0.5 · MetaWorld · 全步：ffk1015_pi05_mt50_full_descent_01–04 = 4,006 + 1,761 + 502 + 83,731 = 90,000。前三个 PASS，_04 是 PENDING_EVIDENCE。
   - 成功过滤库 · π0.5 · MetaWorld · 减步：ffk1015_pi05_mt50_reduced_descent，90,000，FAIL（证据聚合）。
   - 原库 · π0.5 · MetaWorld：全步 pimk1_ful → … → pimk1_fuld PASS（末段 83,731）；减步 pimk1_redc PASS 90,000。
   - 这些不在 formal A 的 fa_ 计划里，未逐局核实。
