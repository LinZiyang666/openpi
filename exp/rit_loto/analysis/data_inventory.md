# 任务 0：数据盘点（GR00T N1.5 × LIBERO，2026-09-13 实测）

## 1. 建库语料在盘，失败轨迹保留

W13 采集（2026-09-08，`exp/libero_groot/launch_collection.sh`）对 B 池每任务全部 50 个 init 各跑一集；库 S3 从成功集里按 seed 0 每任务抽 5 条（`<suite>_w13_manifest.json`）。位置 weilandserver `/archive/libero_cache/build_{spatial,libero10}_w13/<suite>/episode_<gid>_<ts>.h5`，`gid = task_id*50 + init_idx`。

| suite | 集数 | 成功 | 失败 | 决策总数 | 均/集 | 体量 | 库 S3 轨迹 | 库轨迹全在语料 | 库内含失败 |
|---|---|---|---|---|---|---|---|---|---|
| libero_spatial | 500 | 450 | 50 | 11,838 | 23.7 | 26 GB | 50（5×10） | 50/50 | 0 |
| libero_10 | 500 | 436 | 64 | 29,318 | 58.6 | 63 GB | 50（5×10） | 50/50 | 0 |

逐任务失败数（task 0..9）：spatial `12,5,2,1,2,2,2,9,9,6`；libero_10 `8,4,4,2,7,2,12,3,5,17`。

任务 1 查询集 = 库内 50 条（留一整条轨迹）+ 库外 450 条（成功 400 / 386，失败 50 / 64）。库外成功轨迹占多数，失败轨迹不为零；偏差在 `compare.json` 的 in_library × episode_success 四格行数表中读。

## 2. 每条轨迹存了什么

- 文件属性：`task`、`task_id`、`orig_init_state_idx`、`episode_id`、`success`、`num_steps`、`denoise_schedule_id=groot_n15_k8_v1`、`denoising_num_steps=8`、`timestamp`。
- 每个 `step_XXXX`（一次决策，每 5 个控制步）：`vision_0` / `vision_1` `[256,2048] f16`、`prompt_emb` `[n_tok,2048] f16`（54 / 47）、`robot_state` `[8] f32`、`noise_action_0` `[16,32] f32`（纯噪声起点）、`noise_action_1..7`（Euler 第 i 步消费的 x_t，t=i/8）、`clean_action` `[16,32] f32`（当时执行的教师 chunk）。
- 无原始图像。stage-1 序列由指令模板 + 存储切片重建（`exp/libero_groot/cp2_reconstruct.py`）；动作头续跑一侧的保真度由 `parity_gate.json` 自证。
- 库条目：`id="<stem>:<step>"`、`trajectory_id=stem`、`payload.action_chunk [16,32]`、`payload.intermediates {0.125..0.875}`、`denoising_num_steps=8`、`task_key=指令`、`query_keys` vision_0/1/2（32768）+ prompt_emb（2048）+ robot_state（8）。条目数 1,078 / 2,598。

## 3. 现行 shadow 标定（被替代对象）

每 suite 150 集，A 池 seed 20260901 抽 5 fit + 10 cal / 任务，教师驱动、`GrootRitShadow` 影子打标。行文件 `exp/libero_groot/data/rit/shadow/<suite>/shadow_rows.jsonl`（3,432 / 8,383 行），字段 `task, episode_id, step_idx, s, winner_id, y_full, y_rem2, y_rem4, episode_success`。参考 = 该步教师自己的完整推理 chunk；W 来自 `library_action_weights(S3.pkl)`；h_exec=5。`arm_record.json` 只存 knots 不存 q，原曲线由同一份 `fit_ladders` 重拟合并逐值对账。

## 4. 主实验（84 臂 × 500 集/suite）日志字段

`run_gtp` 的 `journal.jsonl` + `per_step.jsonl`，在 timan107/108 `/scratch/zixuans8/openpi_lg/exp/libero_groot/data/rit/eval/<suite>_{anchor,hg}/`。per_step 每决策一行：`yaml_id, task_id, subset_init_state_idx, orig_init_state_idx, episode_id, task_uid, phase, step_idx, hit_type, start_t, winner_id, cp1_score, checkpoint, score, library_sha256, searched, executor, router_outputs, factor_outputs, success, attempt, accepted, run_id`。没有观测、没有执行 chunk，因此任务 3 需要服务端逐决策日志（`--loto-log-out`）。

## 5. 设备

| 节点 | 角色 | 关键路径 |
|---|---|---|
| weilandserver | server + 离线计算（4090 49 GB） | 语料 `/archive/libero_cache/build_*_w13`；库 `/data/libero_cache/libraries_w13/<suite>/<suite>_w13_S3.pkl`；ckpt `/home/weiland/ckpt_n15_libero_{spatial,10}`；岛 venv `/home/weiland/gr00t_n15_venv/.venv/bin/python`；岛克隆 `/data/openpi_lg` |
| h100 | server（备用） | `/data/openpi_lg`、S3 两库、岛 venv；无 W13 语料 |
| timan107 / timan108 | client worker | `/scratch/zixuans8/openpi_lg`、`--conda-env /scratch/zixuans8/libero_sim` |
