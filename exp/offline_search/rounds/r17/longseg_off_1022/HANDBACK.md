# 长执行段关 · 成功过滤库 · 旋钮下降：交回（10-22）

只建定义，没有 prepare / launch，没起 GPU 进程，没碰在跑的进程和 chain 脚本，没 commit。
本机 CPU 全程单进程（`nice -n 10`），没读 per_step.jsonl。新 run 目录全部经 `nvme_dir` 建在
`/nvme/weiland/openpi/eval/<name>`，`/data/mw_finetune/eval/<name>` 只是软链。
tether 只做了一次只读 `exec`（查 H100 上 DP LIBERO 环境和 C5 输入是否在，见 §6.3），没有 expose/push/pull。

## 1. 新批次

全部是「成功过滤库」、长执行段关（每次决策执行 5 个控制，commit_horizon 5 = replan_steps 5）、服务端 H100。

| 批次 | 白话 | 局数 | 部署来源 | H100 常驻 | 端点 | 对应长段开批次 |
|---|---|---:|---|---|---|---|
| `fa_lo_gr_r_f_m` | 成功过滤库 · GR00T · MetaWorld MT50 · 减步(K2) · 旋钮逐档下降 | 90,000 | ffk1015_groot_mt50_reduced_descent | R5/B1/36 GiB/MPS | 149.165.153.233:23160 | fa_ls_gr_r_f_m |
| `fa_lo_gr_f_f_m` | 成功过滤库 · GR00T · MetaWorld MT50 · 全步(K4) · 旋钮逐档下降 | 90,000 | ffk1015_groot_mt50_full_descent | R5/B1/36 GiB/MPS | 149.165.153.233:23160 | fa_ls_gr_f_f_m |
| `fa_lo_dp_r_f_m` | 成功过滤库 · DP · MetaWorld MT50 · 减步(DDIM10) · 旋钮逐档下降 | 90,000 | ffk1015_dp_mt50_reduced_descent | R4/B16/24 GiB/MPS | 149.165.153.233:23180 | fa_ls_dp_r_f_m |
| `fa_lo_dp_f_f_m` | 成功过滤库 · DP · MetaWorld MT50 · 全步(DDPM100) · 旋钮逐档下降 | 90,000 | ffk1015_dp_mt50_full_descent | R8/B32/40 GiB/MPS | 149.165.153.233:23180 | fa_ls_dp_f_f_m |
| `fa_lo_fw_r_f_m` | 成功过滤库 · Fast-WAM · MetaWorld MT50 · 减步(K3) · 旋钮逐档下降 | 90,000 | ffk1015_fastwam_mt50_reduced_descent | R2/B16/76 GiB/MPS | 149.165.153.233:23170 | fa_ls_fw_r_f_m |
| `fa_lo_fw_f_f_m` | 成功过滤库 · Fast-WAM · MetaWorld MT50 · 全步(K10) · 旋钮逐档下降 | 90,000 | ffk1015_fastwam_mt50_full_descent | R2/B16/76 GiB/MPS | 149.165.153.233:23170 | fa_ls_fw_f_f_m |
| `fa_lo_dp_r_f_o_l` | 成功过滤库 · DP · LIBERO · 减步(DDIM10) · 旋钮关 | 5,000 | ffk1015_dp_libero_reduced_formal_A_filtered（4090→H100） | R4/B8/40 GiB/MPS | 149.165.153.233:23180 | fa_ls_dp_r_f_o_l |
| `fa_lo_dp_f_f_o_l` | 成功过滤库 · DP · LIBERO · 全步(DDPM100) · 旋钮关 | 5,000 | ffk1015_dp_libero_full_formal_A_filtered（4090→H100） | R4/B8/40 GiB/MPS | 149.165.153.233:23180 | fa_ls_dp_f_f_o_l |
| `fa_lo_fw_r_f_o_l` | 成功过滤库 · Fast-WAM · LIBERO · 减步(K1) · 旋钮关 | 5,000 | ffk1015_fastwam_libero_full_formal_A_filtered（H100 常驻，YAML 用减步） | R2/B16/70 GiB/MPS | 149.165.153.233:23420 | fa_ls_fw_r_f_o_l |
| `fa_lo_fw_f_f_o_l` | 成功过滤库 · Fast-WAM · LIBERO · 全步(K10) · 旋钮关 | 5,000 | ffk1015_fastwam_libero_full_formal_A_filtered | R2/B16/70 GiB/MPS | 149.165.153.233:23420 | fa_ls_fw_f_f_o_l |
| `fa_lo_dp_r_f_l_l` | 成功过滤库 · DP · LIBERO · 减步 · 旋钮各档（只调未达纯推理的格） | ≤25,000 | 旋钮关跑完后再建 | 同旋钮关 | | fa_ls_dp_r_f_l_l |
| `fa_lo_dp_f_f_l_l` | 成功过滤库 · DP · LIBERO · 全步 · 旋钮各档 | ≤25,000 | 同上 | 同上 | | fa_ls_dp_f_f_l_l |
| `fa_lo_fw_r_f_l_l` | 成功过滤库 · Fast-WAM · LIBERO · 减步 · 旋钮各档 | ≤25,000 | 同上 | 同上 | | fa_ls_fw_r_f_l_l |
| `fa_lo_fw_f_f_l_l` | 成功过滤库 · Fast-WAM · LIBERO · 全步 · 旋钮各档 | ≤25,000 | 同上 | 同上 | | fa_ls_fw_f_f_l_l |

- MT50：每批 6 个库大小（250/500/1000/1500/2000/2500）× 6 组（chrg_koff、chrg_k50/k40/k30/k20/k10）× 50 任务 × 50 初始状态（seed 7，init 0–49），与长段开批次同形。
- LIBERO 旋钮关：libero_spatial、libero_10 × 库大小 50/100/200/300/400 = 10 格 × 500 局。**库大小 500 两格没建**（没有 C5 fit，见 §6.1），所以是 5,000 局而不是 6,000。
- 各档：旋钮关跑完后按 `formal_a_queue.classify`（原样）对同步数、同卡纯推理逐格判定，只给 `tune` 的格建 k50..k10，每格 5 档 × 500 局；全部达标则 `SKIP_OFF_SUFFICIENT`。
- 合计：MT50 540,000 局；LIBERO 旋钮关 20,000 局；各档最多 100,000 局。
- worker 数只是标签（沿用长段开批次的 timan1/timan107 数），实际由中央调度器在 a100/timan1/timan107（加 H100 本机）上分配。

## 2. 每组 YAML 改了什么（其余一字不动）

对每个组，以长段开批次当前的 YAML（MT50、LIBERO 旋钮关）或其来源梯子 YAML（LIBERO 各档）为模板，只改：

- `adapter_binding.commit_horizon`: 10 → **5**
- `recipe.fitted_path / fitted_digest` → 同一库（store_digest 相同、同 suite/size）已有的 **C5 fit**
- `recipe.stall_path / stall_digest` → 该库的 **原始 C5 Q3 停滞门**（exec_steps 5，未重标时钟），即 C5 带门组（cg/crg）当年用的那份；**不是** `q3_decision1018`
- `recipe.miss_mode: stall_only`、`recipe.guard: false`（原值即如此，显式写死）
- `recipe.fit_identity`：重新计算（它对 commit_horizon 做哈希；不重算会留下 C10 的值）
- `recipe.periodic_gap`：MT50 与 LIBERO 旋钮关取长段开 YAML 同档同格的值，koff 为 None；LIBERO 各档取 `longseg_1017/RECALIBRATION.json` 里该格 C10 fit 的 `:filtered:<档>` 解——这正是长段开各档 YAML 里的值（已对 4 个 `fa_ls_*_l_l` 共 135 组逐一核对，全等）。没有跑 recalibrate.fit_all / solve。
- k / kref / early：C5 fit 自己的设置是 16 / 8 / true，与长段开完全相同，**没有任何一格需要改**。corrector、ordered_tasks、serve_loeo（false）、store_path/store_digest 不变。

逐组前后值在 `MAPPING.json`；读回结果在 `VERIFY.json`。

### 过滤库怎么实现（与你给的写法有一处不同，原因如下）

你要求用 `fit_keep_success_1015_norefit_*`（过滤后的 C5 fit）。但这个 fit 和原始 C5 停滞门在 R17 里**接不上**：
`cache_recipe.load_stall` 要求停滞门 READY 的 `fitted_digest` 等于 YAML 的 `fitted_digest`，并用同一个 fit 的度量去核对停滞门度量；
而原始 C5 停滞门绑定的是**原始 C5 fit**（例 GR00T n250：停滞门 `53e2bee8` 的 fitted_digest = `5374a82a`，过滤 fit 是 `85496225`）。
照字面写，每个副本加载 YAML 时都会报 `stall artifact B/fit/source binding mismatch`。

长段开的过滤批次本来就不是用过滤 fit 文件做的：它们用**原始 C10 fit**，再由常驻进程按 `campaign.filter_failed_knob.selection[fit_digest]` 只保留成功库局的候选行
（`filter_failed_knob/runtime.py`，停滞门仍用原始 fit 核对）。新批次用同一机制：YAML 指原始 C5 fit，selection 改成以 C5 fit digest 为键、值为长段开同一格的保留局。
已核对：每一格的保留局集合与对应 `fit_keep_success_1015_norefit_*` 的 `candidate_filter.kept_episodes` **完全相同**（全部 76 个 selection 键：6 个 MT50 批次各 6 个、4 个 LIBERO 旋钮关批次各 10 个），所以检索看到的候选集就是过滤 fit 的候选集。
这样「Q3 only + 原始 C5 停滞门 + 成功过滤库」三条同时成立。

## 3. 每格 fit / 停滞门映射

C5 fit 与停滞门的路径都取自该库 C5 带门组（cg/crg，stall_only）用过的目录：
MT50 `/data/mw_finetune/eval/{csgr_h5,csdp_1504,csfw_k3}/assets/cache_inputs/mt50/n<size>_C5/{fit,stall}`，
LIBERO `/data/mw_finetune/eval/{cszed,cszef}/assets/cache_inputs/<suite>/n<size>_C5/{fit,stall}`。
MT50 这些文件与 `/data/r17_library/studies_mt50/{gr_k2,dp_k10,fw_k3}/artifacts/n<size>/<model>_mw_mt50/C5/{fit,stall}` 的 digest 逐格相同（READY 里 C5 停滞门 exec_steps 5、commit_horizon 5、calibration_partition B_library_successful_LOEO）。
用 eval 下的路径是因为 H100 预备只同步 `.../cache_inputs` 根（`longseg_1017_input_roots`，要有 manifest.json），studies_mt50 目录不在其中。
store_digest 一列：长段开 C10 fit、原始 C5 fit、过滤 C5 fit、C5 停滞门四者的 store_digest 逐格相同；C5 fit 与长段开 C10 fit 的 teacher binding、task_names 也逐格相同。
减步与全步用同一套库与 fit（MT50、LIBERO 都如此），下表对两种步数都适用。digest 取前 12 位，全文在 `MAPPING.json` / 各 YAML。

**GR00T · MetaWorld MT50**（C5 带门组来自 `csgr_h5`）

| suite/size | store_digest | 长段开 C10 fit | 用的 C5 fit（原始） | 过滤 C5 fit（同保留局） | 保留局 | C5 Q3 停滞门 | 停滞门绑定的 fit |
|---|---|---|---|---|---:|---|---|
| mt50/250 | `9111cd81eb35` | `0768e7ca269d` | `5374a82a60c5` | `85496225e98e` | 186 | `53e2bee8cadf` | `5374a82a60c5` |
| mt50/500 | `6306f6d49374` | `9703c66765e6` | `2e769b87ac06` | `3608fa689c73` | 379 | `2368764672c4` | `2e769b87ac06` |
| mt50/1000 | `3c92aa068216` | `8e3af1def46d` | `e79c9ce35c46` | `e7de5a0b5b79` | 762 | `f6a79c9e7329` | `e79c9ce35c46` |
| mt50/1500 | `c7109cda07e8` | `22f39cd8b65e` | `c4d6c78de7f9` | `3acb12b090b0` | 1140 | `6b556e321ba0` | `c4d6c78de7f9` |
| mt50/2000 | `f73cebf1e69a` | `bae649fe5cd5` | `428ef6d98d89` | `f8804b81b43e` | 1516 | `a3672a5057c8` | `428ef6d98d89` |
| mt50/2500 | `67603c290218` | `6e216151d852` | `eace68371976` | `71b7eb9c35d5` | 1899 | `2fbd516107eb` | `eace68371976` |

**DP · MetaWorld MT50**（`csdp_1504`）

| suite/size | store_digest | 长段开 C10 fit | 用的 C5 fit（原始） | 过滤 C5 fit | 保留局 | C5 Q3 停滞门 | 停滞门绑定的 fit |
|---|---|---|---|---|---:|---|---|
| mt50/250 | `5a0fa63aa0b3` | `292668803ef1` | `a2a145aa25dc` | `5817bc9b7abc` | 153 | `6f1e4aca1b8f` | `a2a145aa25dc` |
| mt50/500 | `fbc684616e23` | `b66d3b4e3992` | `0e1cf894afc6` | `2a063af7caa2` | 312 | `85eeaddfa6a3` | `0e1cf894afc6` |
| mt50/1000 | `cfda9ddca7b5` | `14f604ccdcdd` | `3760441f9f29` | `07ff4d7d712a` | 623 | `474e82e55419` | `3760441f9f29` |
| mt50/1500 | `d722ae3829f9` | `aabf0a6c60c9` | `af11d80a1d26` | `3dccc9a5bddc` | 943 | `4a5baab6f624` | `af11d80a1d26` |
| mt50/2000 | `3a6ab14da0fc` | `0b39c5f7efc6` | `6f4d35003da4` | `21ed61377eb2` | 1262 | `620943596a57` | `6f4d35003da4` |
| mt50/2500 | `3618bf9f3d67` | `adca6f852342` | `45f52a080319` | `6d3c1d0e324b` | 1577 | `8f904e8cdc95` | `45f52a080319` |

**Fast-WAM · MetaWorld MT50**（`csfw_k3`）

| suite/size | store_digest | 长段开 C10 fit | 用的 C5 fit（原始） | 过滤 C5 fit | 保留局 | C5 Q3 停滞门 | 停滞门绑定的 fit |
|---|---|---|---|---|---:|---|---|
| mt50/250 | `57b582d5e180` | `47bc59b9b485` | `b83b1b74cb82` | `098c24dc6f60` | 144 | `b19b6160eec2` | `b83b1b74cb82` |
| mt50/500 | `48e99739df29` | `3a1f585129f2` | `ccbc9b5b5cfa` | `8bd25471db94` | 295 | `6bd6930b6cc8` | `ccbc9b5b5cfa` |
| mt50/1000 | `c7a49a0d81a3` | `0764b23772ac` | `50d5186c16a1` | `c87d8c228cef` | 590 | `3b20469f0114` | `50d5186c16a1` |
| mt50/1500 | `1e02211b71fa` | `b804669411ff` | `9a614dc8228c` | `ad044cf06db3` | 874 | `ba9565effbe0` | `9a614dc8228c` |
| mt50/2000 | `3c177e1da7d3` | `f517b631cf2e` | `fdb7fa785329` | `f72d0b99f93b` | 1159 | `cda2f95c5ef0` | `fdb7fa785329` |
| mt50/2500 | `bcae8e06c42e` | `9ab29962f3b6` | `091143e76795` | `5356cffac395` | 1457 | `4ed73731ca9a` | `091143e76795` |

**DP · LIBERO**（C5 带门组来自 `csztd_r3_l48`，文件在 `cszed`；过滤 C5 fit 即 ff1015_dp_li_14 等用的那些）

| suite/size | store_digest | 长段开 C10 fit | 用的 C5 fit（原始） | 过滤 C5 fit | 保留局 | C5 Q3 停滞门 | 停滞门绑定的 fit |
|---|---|---|---|---|---:|---|---|
| libero_10/50 | `d8c66a4d76d8` | `55f38996f90a` | `83a2667bb77f` | `143cb5ce7b87` | 42 | `b9dbd1f2ffef` | `83a2667bb77f` |
| libero_10/100 | `197e679aacf8` | `7dd17b61a9a1` | `9b693b2e4ff0` | `f703d21e4c13` | 82 | `fa758ecfe75e` | `9b693b2e4ff0` |
| libero_10/200 | `cd33b4fcc69b` | `1e5735fe3d13` | `e73b7e6408db` | `82153074f20e` | 164 | `3d9acbd26b7a` | `e73b7e6408db` |
| libero_10/300 | `c6d00a800c35` | `5cee112d23bb` | `eb95677eb075` | `8d8d45d3add6` | 245 | `fff567fb649e` | `eb95677eb075` |
| libero_10/400 | `42659200ad91` | `270b90d204b6` | `ca8080967974` | `6fc07e10f6a4` | 330 | `cd29be06c88c` | `ca8080967974` |
| libero_10/500 | `0ff3a35e1196` | `5956e15b3f21` | 无 C5 fit | — | — | — | — |
| libero_spatial/50 | `c528beeb976a` | `b8be064da386` | `45e0d1e21fb0` | `480c4dbb78d2` | 42 | `b906dcb45a0b` | `45e0d1e21fb0` |
| libero_spatial/100 | `c928be2659b2` | `a1556650590e` | `dfcdb8bddfe3` | `a4a4bfd854d7` | 84 | `1a4d19dc7185` | `dfcdb8bddfe3` |
| libero_spatial/200 | `7ba3cddbd1fc` | `c3497d728eb1` | `deffd3597a14` | `4fca399b4355` | 177 | `06dafc511534` | `deffd3597a14` |
| libero_spatial/300 | `2ec9b77c942d` | `04479ab496e0` | `1331eea2e050` | `399556b750d6` | 265 | `e28b03b3a486` | `1331eea2e050` |
| libero_spatial/400 | `9c47d5194d2d` | `7a274546e1e5` | `04dfc1a2f17d` | `b17e041e5ff2` | 349 | `ba7374ce0fb1` | `04dfc1a2f17d` |
| libero_spatial/500 | `44ea45d9a2e1` | `0c141049fd68` | 无 C5 fit | — | — | — | — |

**Fast-WAM · LIBERO**（C5 带门组来自 `fastwamc6066547406`，文件在 `cszef`；过滤 C5 fit 即 ff1015_fastwam_li_18 等用的那些）

| suite/size | store_digest | 长段开 C10 fit | 用的 C5 fit（原始） | 过滤 C5 fit | 保留局 | C5 Q3 停滞门 | 停滞门绑定的 fit |
|---|---|---|---|---|---:|---|---|
| libero_10/50 | `910c1b92cb3b` | `5478de03b6c5` | `1f03922e9640` | `02a4ab852914` | 49 | `343a33b12e18` | `1f03922e9640` |
| libero_10/100 | `c28ae9e4c309` | `676c417ce797` | `b6566c5db2b3` | `5f13626e1c38` | 95 | `1020c777f374` | `b6566c5db2b3` |
| libero_10/200 | `033a8f685b68` | `a6ca12ac3958` | `2ac7795d28ea` | `b2dcceab2dd0` | 188 | `3c05c8fa3613` | `2ac7795d28ea` |
| libero_10/300 | `a10de4618f85` | `92affdba2656` | `132d7b7586af` | `f43f7a37a0e5` | 279 | `70e5db31613d` | `132d7b7586af` |
| libero_10/400 | `7d9a95596d7c` | `3a30862d38bb` | `2fca3bcb11e0` | `de96ce94b3fe` | 371 | `db4e0e18e259` | `2fca3bcb11e0` |
| libero_10/500 | `b5876beb4ef0` | `a021fd5e9721` | 无 C5 fit | — | — | — | — |
| libero_spatial/50 | `5457c9b97e4f` | `722f9885de7e` | `7fa241ec3998` | `356a1bacf298` | 43 | `9aa88075d6af` | `7fa241ec3998` |
| libero_spatial/100 | `9dba70970b54` | `a757cc82699f` | `9f144725a365` | `c4a04fe63a05` | 85 | `16035c8dccd5` | `9f144725a365` |
| libero_spatial/200 | `748a23298bf7` | `472e7dfd680d` | `b99629ca61a2` | `0b1340107169` | 175 | `894e33e3319b` | `b99629ca61a2` |
| libero_spatial/300 | `51f6cdd152b8` | `ef98f04c0934` | `46d15c3f8859` | `6fb417fb26e9` | 259 | `0c9965b1a157` | `46d15c3f8859` |
| libero_spatial/400 | `d1ed512428e4` | `a2d349b6d012` | `baf7b6e61cfb` | `cabedb458cda` | 345 | `f1fbb757e0b7` | `baf7b6e61cfb` |
| libero_spatial/500 | `fee6908f004c` | `aa3440937097` | 无 C5 fit | — | — | — | — |

停滞门文件一个都没动：没有新建硬链接、没有 mv、没改元数据（建完后查过：这些 C5 stall 文件链接数仍为 1，ctime 仍是 10-20 迁移时的值）。新 run 的 YAML 直接指向上述原目录。

## 4. 部署（只换服务位置，科学定义不变）

- GR00T MT50：来源是 R11/76 GiB，按 owner 10-20 裁定改 **R5/36 GiB**（和 `fa_ls_gr_f_f_m` 的改法一样；B1，服务线程按 H100 R5 自动取 2）。
- DP MT50：长段开跑在 4090（`fa_ls_t_*` 模板）；新批次直接从 H100 原始来源 `ffk1015_dp_mt50_{reduced,full}_descent` 克隆，R/B/MPS/显存与 4090 版相同（R4/B16/24 GiB、R8/B32/40 GiB），env `/data/mw_finetune/dp/env/bin/python`（H100 上在）。
- DP LIBERO：来源是 4090 批次；`server_node` 改 h100、公共端口 23180（DP 端口段，后端 23181–23184），其余 R4/B8/5 ms/MPS/40 GiB 不变；把 CUDA 图自检帧 `wire_observations.npz` 复制进 run 的 assets（H100 上没有这份文件，assets 里的会先被读，随 assets 一起同步到 H100）。
- Fast-WAM LIBERO 减步：长段开跑在 4090（文本缓存配置）；新批次用 H100 的 Fast-WAM LIBERO 常驻（`ffk1015_fastwam_libero_full_formal_A_filtered`：R2/B16/70 GiB/MPS，无文本缓存），YAML 用减步（K1）那套；`cache_plan_name` 保持 `knobtestfw_red`。
- 驱动端口都是 23199（H100 lane）。没有任何定义用到 4090（`server_node` 全是 h100，调度器只会给 a100/timan1/timan107 和 H100 本机分 worker）。

## 5. 改动的现有文件

| 文件 | 改动 |
|---|---|
| `exp/offline_search/rounds/r17/longseg_1017/build.py` | 删掉 `clone()` 里 `fa_ls_` 前缀检查（`Use a fresh short fa_ls_ name`）；`clone()` 加可选参数 `edit`，在写入组 YAML 之后、收集输入根之前调用（新批次用它写 C5 YAML、换 selection、改部署字段）。不传时行为不变。 |
| `exp/offline_search/rounds/r17/longseg_1017/pod_copy.py` | `copy_job` 原来只接受 4090 的 DP 来源（`('local','dp')`），现在接受任何 DP 来源——新的 DP MT50 正式定义在 H100 上，否则没法做 `_j` 副本。只改了这一个判断和报错文字。 |

新文件：
- `exp/offline_search/rounds/r17/longseg_off_1022/`：`build.py`（MT50 构建 + 通用 C5 改写）、`workflow.py`（LIBERO 旋钮关→各档）、`verify.py`（只读回读）、`CAMPAIGNS.json`、`FORMAL_QUEUE.json`、`MAPPING.json`、`VERIFY.json`、`BUILD_*.log`、本文件。
- `exp/offline_search/rounds/r17/filter_failed_knob/commands/`：28 个脚本 `{prepare,launch}_fa_lo_{gr,dp,fw}_{r,f}_f_m.sh`、`{prepare,launch}_fa_lo_{dp,fw}_{r,f}_f_o_l.sh`、`{prepare,launch}_fa_lo_{dp,fw}_{r,f}_f_l_l.sh`。
- 10 个 run 目录 `/nvme/weiland/openpi/eval/fa_lo_*`（+ `/data/mw_finetune/eval` 软链）。
- 没有加任何新检查、哈希钉、源码锁、测试。

注意：同一时间另有一个任务在建 R15/R16 的 `fa_lo_{gr,pi}_*_q*` 批次和 `filter_failed_knob/longseg_off_1022/`（不是本目录）。名字不冲突，我没碰它们。

## 6. 命令（协调者执行；都在仓库根目录）

```bash
cd /home/weiland/projects/openpi
C=exp/offline_search/rounds/r17/filter_failed_knob/commands
# MetaWorld（H100）
bash $C/prepare_fa_lo_gr_r_f_m.sh && bash $C/launch_fa_lo_gr_r_f_m.sh
bash $C/prepare_fa_lo_gr_f_f_m.sh && bash $C/launch_fa_lo_gr_f_f_m.sh
bash $C/prepare_fa_lo_dp_r_f_m.sh && bash $C/launch_fa_lo_dp_r_f_m.sh
bash $C/prepare_fa_lo_dp_f_f_m.sh && bash $C/launch_fa_lo_dp_f_f_m.sh
bash $C/prepare_fa_lo_fw_r_f_m.sh && bash $C/launch_fa_lo_fw_r_f_m.sh
bash $C/prepare_fa_lo_fw_f_f_m.sh && bash $C/launch_fa_lo_fw_f_f_m.sh
# LIBERO 旋钮关（H100）
bash $C/prepare_fa_lo_dp_r_f_o_l.sh && bash $C/launch_fa_lo_dp_r_f_o_l.sh
bash $C/prepare_fa_lo_dp_f_f_o_l.sh && bash $C/launch_fa_lo_dp_f_f_o_l.sh
bash $C/prepare_fa_lo_fw_r_f_o_l.sh && bash $C/launch_fa_lo_fw_r_f_o_l.sh
bash $C/prepare_fa_lo_fw_f_f_o_l.sh && bash $C/launch_fa_lo_fw_f_f_o_l.sh
# LIBERO 各档：对应旋钮关跑完后再执行；prepare 先判格、只建要调的格，再 stage
bash $C/prepare_fa_lo_dp_r_f_l_l.sh && bash $C/launch_fa_lo_dp_r_f_l_l.sh
bash $C/prepare_fa_lo_dp_f_f_l_l.sh && bash $C/launch_fa_lo_dp_f_f_l_l.sh
bash $C/prepare_fa_lo_fw_r_f_l_l.sh && bash $C/launch_fa_lo_fw_r_f_l_l.sh
bash $C/prepare_fa_lo_fw_f_f_l_l.sh && bash $C/launch_fa_lo_fw_f_f_l_l.sh
```

prepare/launch 脚本与 `longseg_1017/build.commands` 写法相同（`filter_failed.stage --run` / `filter_failed_knob.launch --run`）；各档脚本调用
`python -m exp.offline_search.rounds.r17.longseg_off_1022.workflow {prepare,launch} --name <name>`（也可先单独 `build-levels --name <name>` 只判格建定义）。

**MetaWorld 的 pod 副本**（没有做；做法与现有 `_j` 相同，从未 prepare 的 H100 正式定义复制，`server_node` 仍是 h100，由 `R17_H100_NODE` 别名在 stage 时落到 pod）：

```bash
P="python -m exp.offline_search.rounds.r17.longseg_1017.pod_copy"
$P --source fa_lo_gr_f_f_m --name fa_lo_gr_f_f_m_j --pod jupyter-ziyang10    --driver-port 23191 --replicas 1   # pod 上 GR00T 一律 R1/14 GiB
$P --source fa_lo_dp_r_f_m --name fa_lo_dp_r_f_m_j --pod jupyter-ziyang10-02 --driver-port 23192                # DP 默认 R1/14 GiB，自动嵌 admission bridge
$P --source fa_lo_fw_r_f_m --name fa_lo_fw_r_f_m_j --pod jupyter-ziyang10-02 --driver-port 23192 --memory-gib 15.5 --max-batch 8  # 自动用 H100 文本表
bash exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_fa_lo_gr_f_f_m_j.sh   # pod_copy 写的包装，脚本里已 export 别名
```
同时在跑的副本各用不同驱动端口（23191/23192/23193…，不能用 H100 lane 的 23199）；pod 先按 RUN_CONFIGS「Jupyter pods」一节装环境（groot / dp / fastwam_nfs）和隧道。
`_j` 副本的科学定义（assets、YAML、EPISODE_PLAN）与 H100 正式定义相同，只换服务位置；Fast-WAM 在 pod 上 B8，按惯例需要 pod 同卡纯推理参照。

## 7. CPU 计划回读（`verify.py` → `VERIFY.json`）

| 批次 | 计划局数 | 每局 commit_horizon | 每局 miss_mode | C5 YAML | guard | 停滞门 digest 数 | fit 都在 selection |
|---|---:|---|---|---|---|---:|---|
| fa_lo_gr_r_f_m | 90,000 | 5 ×90,000 | stall_only ×90,000 | 36/36 | false | 6 | 是 |
| fa_lo_gr_f_f_m | 90,000 | 5 ×90,000 | stall_only ×90,000 | 36/36 | false | 6 | 是 |
| fa_lo_dp_r_f_m | 90,000 | 5 ×90,000 | stall_only ×90,000 | 36/36 | false | 6 | 是 |
| fa_lo_dp_f_f_m | 90,000 | 5 ×90,000 | stall_only ×90,000 | 36/36 | false | 6 | 是 |
| fa_lo_fw_r_f_m | 90,000 | 5 ×90,000 | stall_only ×90,000 | 36/36 | false | 6 | 是 |
| fa_lo_fw_f_f_m | 90,000 | 5 ×90,000 | stall_only ×90,000 | 36/36 | false | 6 | 是 |
| fa_lo_dp_r_f_o_l | 5,000 | 5 ×5,000 | stall_only ×5,000 | 10/10 | false | 10 | 是 |
| fa_lo_dp_f_f_o_l | 5,000 | 5 ×5,000 | stall_only ×5,000 | 10/10 | false | 10 | 是 |
| fa_lo_fw_r_f_o_l | 5,000 | 5 ×5,000 | stall_only ×5,000 | 10/10 | false | 10 | 是 |
| fa_lo_fw_f_f_o_l | 5,000 | 5 ×5,000 | stall_only ×5,000 | 10/10 | false | 10 | 是 |

另核对：每组 periodic_gap 与长段开同格同档相等（koff 全是 None）；没有任何 stall_path 指向 `q3_decision1018`；YAML 里的 fitted/stall digest 与 `STUDY.json`、`CACHE_STUDY.json` 一致。可复查的 grep：

```bash
cd /data/mw_finetune/eval
grep -o '"commit_horizon":[0-9]*' fa_lo_gr_r_f_m/EPISODE_PLAN.json | sort | uniq -c     # 90000 "commit_horizon":5
grep -o '"miss_mode":"[a-z_]*"'   fa_lo_gr_r_f_m/EPISODE_PLAN.json | sort | uniq -c     # 90000 "miss_mode":"stall_only"
grep -o '"fit_digest": "[0-9a-f]*"\|"stall_digest": "[0-9a-f]*"' fa_lo_gr_r_f_m/STUDY.json | sort | uniq -c
grep -h 'commit_horizon\|fitted_digest\|stall_digest\|miss_mode\|guard:' fa_lo_gr_r_f_m/assets/cache_yamls/lo_*.yaml | sort | uniq -c
```

## 8. 没解决 / 需要协调者决定

1. ~~LIBERO 库大小 500 没有 C5 fit~~（已解决，见 §9）：原文：**LIBERO 库大小 500 没有 C5 fit**：DP、Fast-WAM 两个 suite 的 500 格只有 C10 fit（knob 研究建的），eval 和 r17_library 下都没有 `n500_C5`，也没有对应 C5 停滞门。按「不重拟合」这 2 格没建，LIBERO 旋钮关每批 10 格 5,000 局；各档判定也只对这 10 格。
2. **DP LIBERO 在 H100 上缺服务端环境**：定义里 `server_python` 仍是 `/home/weiland/projects/openpi/.venv/bin/python`（与长段开和 pod 一致；`formal_eval.server_env` 只对这个路径不注入 python_deps312）。只读查过 H100：这个路径**不存在**；`/data/mw_finetune/openpi_venv_4090/venv/bin/python` 存在且链接能解析（pod 也是从这里拉的）。需要协调者在 H100 上把它放到这个路径（例如 `mkdir -p /home/weiland/projects/openpi && ln -s /data/mw_finetune/openpi_venv_4090/venv /home/weiland/projects/openpi/.venv`），否则两个 DP LIBERO 批次在副本启动时会失败。正式 R17 批次里 DP LIBERO 只在 4090 和 pod 上服务过，没在真 H100 上跑过。另外 H100 上没有 `cszed` 的 cache_inputs，prepare 会推一次（约 843 MB，4 MiB/s 限速约 3.5 分钟）；其余 C5 根（csgr_h5、csdp_1504、csfw_k3、cszef）H100 上已有。
3. ~~H100 同卡纯推理参照缺三条~~（已建，见 §9）：原文：**H100 同卡纯推理参照缺三条**：`dp:libero:reduced:H100`、`dp:libero:full:H100`、`fastwam:libero:reduced:H100` 在 `longseg_1017/REFERENCE_SELECTION.json` 里都没有（现有的是 4090 上的；Fast-WAM 已知不跨卡等价）。这三条线的各档 prepare 会停在「register the same-step H100 pure LIBERO run as …」。要么先跑并登记 H100 纯推理，要么协调者决定用哪条参照。`fastwam:libero:full:H100`（fa_ls_fw_f_b_l）已在，Fast-WAM 全步各档可直接走。MT50 分析时同理：`dp:mt50:reduced:H100` 也没登记（全步有 tdp7h1_full2）。
4. **Fast-WAM LIBERO 减步换到 H100 常驻**：H100 常驻无文本缓存（RUN_CONFIGS：测试线保持 H100 配置），K1 由 YAML 选。这条线以前的长段开跑在 4090 文本缓存配置上，两者不是同卡对比。
5. 过滤 fit 的写法与你给的不同（§2）：没有用 `fit_keep_success_1015_norefit_*` 文件本身，用原始 C5 fit + 相同保留局的候选行选择；原因是原始 C5 停滞门绑定原始 C5 fit，照字面写会在加载时报错。若你仍要过滤 fit 文件本身，就需要一份绑定过滤 fit 的 C5 停滞门（等于重拟合），这超出了「不重拟合」。

## 9. 协调者追加（10-22 晚）：补 500 格、补同卡 C5 纯推理参照

### 9.1 LIBERO 库大小 500 的 C5 fit 与 C5 Q3 停滞门

- 做法照抄 `exp/r17_integration/ops/cache_study.fit_inputs`（C5 时）：`fit_recipe(k=16, kref=8, early=True, corrector=True, guard=False,
  periodic_gap=None, compact_loeo=True, commit_horizon=5)`，同样的 receipt 字段；停滞门 `StallModel.fit`（H、exec_steps 取库布局，
  commit_controls=min(H, 2·exec_steps)=10）+ 同样的 `r17.cache_stall.v1` receipt。没有 IR/gap 标定。代码 `fit500.py`。
- **逐位复现核对**：用同一脚本重拟 DP libero_10 n50 C5，单线程（OMP=1）得到 fit `83a2667b…`、停滞门 `b9dbd1f2…`，与 cszed 里原有的完全一致；
  4 线程时只在浮点末位不同（distance_scale 11.983152 vs 11.983151），所以 500 格全部单线程拟合（`nice -n 19`，4 个进程）。
- 新目录（/nvme 实目录，/data 软链，带 manifest.json，prepare 时会推到 H100：DP 163 MB、Fast-WAM 118 MB）：

| 库 | suite | store_digest | C5 fit | C5 Q3 停滞门（exec 5，绑定该 fit） | 路径 |
|---|---|---|---|---|---|
| DP | libero_10 | `0ff3a35e1196` | `6451fd399d3c` | `53b67d9a5fe0` | /data/mw_finetune/eval/c5n500_dp_libero/assets/cache_inputs/libero_10/n500_C5 |
| DP | libero_spatial | `44ea45d9a2e1` | `34a5d2f4e7c6` | `58a781537258` | …/c5n500_dp_libero/…/libero_spatial/n500_C5 |
| Fast-WAM | libero_10 | `b5876beb4ef0` | `18df2ed82291` | `909f139869cf` | /data/mw_finetune/eval/c5n500_fastwam_libero/assets/cache_inputs/libero_10/n500_C5 |
| Fast-WAM | libero_spatial | `fee6908f004c` | `3408179f1b1f` | `b123fe68bc68` | …/c5n500_fastwam_libero/…/libero_spatial/n500_C5 |

  与长段开同格 C10 fit 比：teacher binding、store_digest、task_names 全等；k/kref/early/corrector 与其他尺寸相同。过滤沿用 §2 的机制（selection 键为新 C5 fit digest，保留局取长段开同格）。
- 四个旋钮关批次（未 prepare）已按 12 格重建：`fa_lo_{dp,fw}_{r,f}_f_o_l` 各 **6,000 局**，12/12 组 C5、stall_only、guard false、12 个停滞门、fit 全在 selection。
  旧的 10 格版本挪到 `/nvme/weiland/openpi/eval/fa_lo_*_f_o_l.superseded_10cells_*`（没删）。各档单元现在 12 格、最多 30,000 局。

### 9.2 同卡 C5 纯推理参照（照 fa_ls_*_b_*：recipe 关、commit 5、每 5 个控制调一次模型、50 init）

| 键（REFERENCES.json） | 参照批次 | 局数 | 卡/常驻 | 说明 |
|---|---|---:|---|---|
| dp:libero:reduced:H100 | `fa_lo_dp_r_b_l`（新） | 1,000 | H100 R4/B8/40 GiB/MPS | 同 fa_ls_dp_r_b_l 来源，部署同 fa_lo_dp_r_f_o_l |
| dp:libero:full:H100 | `fa_lo_dp_f_b_l`（新） | 1,000 | 同上 | |
| fastwam:libero:reduced:H100 | `fa_lo_fw_r_b_l`（新） | 1,000 | H100 R2/B16/70 GiB/MPS，K1 | 常驻同 fa_lo_fw_r_f_o_l |
| fastwam:libero:full:H100 | `fa_ls_fw_f_b_l`（已有，**是 C5**：commit 5、recipe 关、K10、H100 R2/B16，已跑完） | 1,000 | | 复用 |
| groot:mt50:{reduced,full}:H200 MIG 1g.18gb | `fa_lo_gr_{r,f}_b_m_j`（新） | 2,500 ×2 | jupyter-ziyang10，R2/14 GiB，K2/K4 | 与 fa_lo_gr_r_f_m_j 同配置 |
| fastwam:mt50:{reduced,full}:H200 MIG 1g.18gb | `fa_lo_fw_{r,f}_b_m_j`（新） | 2,500 ×2 | jupyter-ziyang10-02，R1/B16/14 GiB，H100 文本表，K3/K10 | 与 fa_lo_fw_r_f_m_j 同配置 |
| dp:mt50:{reduced,full}:H200 MIG 1g.18gb | `fa_ls_dp_{r,f}_b_m_j`（已有，**是 C5**：commit 5、recipe 关；R1 B16 / B32） | | | 复用 |

- pod 参照先建 H100 正式定义 `fa_lo_{gr,fw}_{r,f}_b_m`（GR00T R5），再用 `pod_copy` 默认参数做 `_j`（与协调者的两个 `_j` 相同），驱动端口 jupyter-ziyang10 23193、-02 23192。
- 登记：`longseg_off_1022/REFERENCES.json`（本研究唯一来源）；同样的键也只**新增**进 `longseg_1017/REFERENCE_SELECTION.json`（9 个新键，旧键一个没改）。
  各档判格 `workflow.build_levels` 现在只读 `REFERENCES.json`（`build.reference`），MT50 分析按 `build.card(run)` 取卡（pod 副本 → `H200 MIG 1g.18gb`，否则 H100）再查同一表。
- 若 MT50 旋钮关改在 H100 上跑：H100 的 C5 纯推理参照 `dp:mt50:reduced:H100` 仍没有；可用同法建（`fa_lo_gr/fw_*_b_m` 的 H100 版已经建好可直接跑）。

### 9.3 命令

```bash
cd /home/weiland/projects/openpi
C=exp/offline_search/rounds/r17/filter_failed_knob/commands
# LIBERO 旋钮关（已重建为 12 格，脚本同名）
bash $C/prepare_fa_lo_dp_r_f_o_l.sh && bash $C/launch_fa_lo_dp_r_f_o_l.sh
bash $C/prepare_fa_lo_dp_f_f_o_l.sh && bash $C/launch_fa_lo_dp_f_f_o_l.sh
bash $C/prepare_fa_lo_fw_r_f_o_l.sh && bash $C/launch_fa_lo_fw_r_f_o_l.sh
bash $C/prepare_fa_lo_fw_f_f_o_l.sh && bash $C/launch_fa_lo_fw_f_f_o_l.sh
# H100 LIBERO 纯推理参照（DP 两条需要 H100 上的 .venv 软链）
bash $C/prepare_fa_lo_dp_r_b_l.sh && bash $C/launch_fa_lo_dp_r_b_l.sh
bash $C/prepare_fa_lo_dp_f_b_l.sh && bash $C/launch_fa_lo_dp_f_b_l.sh
bash $C/prepare_fa_lo_fw_r_b_l.sh && bash $C/launch_fa_lo_fw_r_b_l.sh
# pod MT50 纯推理参照（包装脚本已 export R17_H100_NODE / R17_DRIVER_PORT）
bash $C/prepare_fa_lo_gr_r_b_m_j.sh && bash $C/launch_fa_lo_gr_r_b_m_j.sh   # jupyter-ziyang10, 23193
bash $C/prepare_fa_lo_gr_f_b_m_j.sh && bash $C/launch_fa_lo_gr_f_b_m_j.sh   # jupyter-ziyang10, 23193
bash $C/prepare_fa_lo_fw_r_b_m_j.sh && bash $C/launch_fa_lo_fw_r_b_m_j.sh   # jupyter-ziyang10-02, 23192
bash $C/prepare_fa_lo_fw_f_b_m_j.sh && bash $C/launch_fa_lo_fw_f_b_m_j.sh   # jupyter-ziyang10-02, 23192
# 各档（对应旋钮关和参照都跑完后）
bash $C/prepare_fa_lo_dp_r_f_l_l.sh && bash $C/launch_fa_lo_dp_r_f_l_l.sh   # 其余三条同名替换 dp_f / fw_r / fw_f
```
同一 pod 上的批次共用一个驱动端口，只能排队依次跑。没有碰在跑的 chain（h100lo1/h100lo3/pod1lo1/pod2lo1）和它们的目录，也没碰 `fa_lo_{fw,gr}_r_f_m_j`。
