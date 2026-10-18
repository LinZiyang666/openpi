# 协调员执行单：4090 / H100 / A100

当前首选：DP full DDPM100，真实3帧×3次、真实 fitted cache。先比较GPU算术和cache决策，之后才做闭环pilot。所有输出只写本工作目录或远端 `/home/exouser/xgpu_1017`。以下GPU命令**尚未执行**；4090/H100一律由协调员运行。A100已获本项诊断授权，但公共rsync认证尚未打通。

## 1. 第一条GPU命令（4090，协调员执行）

**空出约5GiB：PyTorch allocator上限4GiB，另留约1GiB CUDA上下文/非allocator余量；1–5分钟，内部300s watchdog，外层310s timeout。** 脚本不停止/恢复任何replica。协调员若临时减少replica，请自行恢复原lane；若余量不足，脚本失败退出，不自动挤占。

```bash
cd /home/weiland/projects/openpi
bash exp/offline_search/rounds/r17/xgpu_1017/run_dp_local.sh native
```

打印的输出路径形如 `.../results/4090_native_20261018TxxxxxxZ`。保留整个目录，含 `REPLAY.json` 与每行 JSON/NPZ。结果中必须 `status=COMPLETE`、9 rows、noise/denoise覆盖完整、首repeat每帧 capture_plain_parity.exact=true；OOM/timeout/INCOMPLETE 是未完成，不是差异为零。

等价性工具使用冻结 runtime `5927a753312bf5981cba81a2e389edea914bde0580523b0dc3c8ea8d58f1fa1e` 和 `/data/mw_finetune/eval/fa_dp_f_u_o/assets/cache_yamls/n50_sp_chrg_koff.yaml`。输入为 `inputs_dp3`，不是旧的 `inputs_tiny3*`。

## 2. 远端准备：先磁盘，再单份最小包

下面给出A100的完整公共rsync步骤；认证入口需要协调员提供现成可用方式或代执行。**已试 `exouser@149.165.152.232` 的 BatchMode SSH dry-run，返回 Permission denied。不要重试口令，不读取/变更key，不改sshd。** 不使用 tether push/pull/expose。`XGPU_RSYNC_SSH` 可由协调员替换为已授权的SSH调用；目标仍应是公共地址。

本地预备清单已生成：`prepared_dp_final/`，只它有效；旧 `prepared_dp` / `prepared_a100*` 不传。所有rsync串行执行，限速16MiB/s；先看本地 `/proc/pressure/io`，若短期压力明显上升，推迟传输，不并行读取 /data。传输前至少10GiB可用；本轮实测新A100可用47G。结果空间也保留在该单一目录。

```bash
cd /home/weiland/projects/openpi
cat /proc/pressure/io
tether exec --timeout 30s a100 -- df -h -l
tether exec --timeout 30s a100 -- mkdir -p /home/exouser/xgpu_1017/rootfs /home/exouser/xgpu_1017/tools /home/exouser/xgpu_1017/prepared /home/exouser/xgpu_1017/inputs_dp3 /home/exouser/xgpu_1017/results

XGPU_WORK=/home/weiland/projects/openpi/exp/offline_search/rounds/r17/xgpu_1017
XGPU_REMOTE=exouser@149.165.152.232
XGPU_RSYNC_SSH='ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=yes'

rsync -a --bwlimit=16384 -e "$XGPU_RSYNC_SSH" --files-from="$XGPU_WORK/prepared_dp_final/rootfs.files" / "$XGPU_REMOTE:/home/exouser/xgpu_1017/rootfs/"
rsync -a --bwlimit=16384 -e "$XGPU_RSYNC_SSH" "$XGPU_WORK/prepared_dp_final/" "$XGPU_REMOTE:/home/exouser/xgpu_1017/prepared/"
rsync -a --bwlimit=16384 -e "$XGPU_RSYNC_SSH" "$XGPU_WORK/inputs_dp3/" "$XGPU_REMOTE:/home/exouser/xgpu_1017/inputs_dp3/"
rsync -a --bwlimit=16384 -e "$XGPU_RSYNC_SSH" --include='*.py' --include='*.sh' --exclude='*' "$XGPU_WORK/" "$XGPU_REMOTE:/home/exouser/xgpu_1017/tools/"
```

Python匹配本地当前DP cache用的 **3.11.15 openpi env**；不要复制 nominal DP `.pth` 空壳，也不要用A100系统3.12去加载cp311轮子。无pip install、无下载cache，只复制本次需要的一份依赖：

```bash
rsync -a --bwlimit=16384 -e "$XGPU_RSYNC_SSH" --exclude='__pycache__' --exclude='*.pyc' /home/weiland/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/ "$XGPU_REMOTE:/home/exouser/xgpu_1017/python/"
tether exec --timeout 30s a100 -- /home/exouser/xgpu_1017/python/bin/python3.11 -m venv --without-pip /home/exouser/xgpu_1017/env
rsync -a --bwlimit=16384 -e "$XGPU_RSYNC_SSH" --files-from="$XGPU_WORK/prepared_dp_final/site_packages.files" /home/weiland/projects/openpi/.venv/lib/python3.11/site-packages/ "$XGPU_REMOTE:/home/exouser/xgpu_1017/env/lib/python3.11/site-packages/"
```

DP历史teacher validator有两条canonical绝对路径。仅在这两条路径**不存在**时，给它们指向本包中唯一副本的链接；如已存在，核实seal并直接使用，不能覆盖。先读权限和存在性：

```bash
tether exec --timeout 30s a100 -- python3 -c 'from pathlib import Path; print([(str(p),p.exists(),p.is_symlink()) for p in (Path("/data/mw_finetune/dp"),Path("/data/mw_finetune/dp/libero_policy_100000"),Path("/data/mw_finetune/dp/clip_text"))])'
```

仅当父目录不存在或需要root创建时，这条 **planned sudo** 才需要执行。执行后把准确命令、原因、退出码追加本目录审计；截至报告生成实际sudo=0条。

```bash
tether exec --timeout 30s a100 -- sudo install -d -o exouser -g exouser /data/mw_finetune/dp
tether exec --timeout 30s a100 -- ln -s /home/exouser/xgpu_1017/rootfs/data/mw_finetune/dp/libero_policy_100000 /data/mw_finetune/dp/libero_policy_100000
tether exec --timeout 30s a100 -- ln -s /home/exouser/xgpu_1017/rootfs/data/mw_finetune/dp/clip_text /data/mw_finetune/dp/clip_text
```

链接不能传给model loader的checkpoint字段（它拒绝symlink）；本包的load.json已使用rootfs中的真实目录。链接仅满足历史library validator；不复制第二份checkpoint。

先跑远端CPU依赖/fit检查，禁止CUDA：

```bash
tether exec --timeout 120s a100 -- env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /home/exouser/xgpu_1017/env/bin/python /home/exouser/xgpu_1017/tools/cpu_cache_check.py --campaign /home/exouser/xgpu_1017/prepared/job.json --cache-yaml /home/exouser/xgpu_1017/prepared/cache.yaml --output /home/exouser/xgpu_1017/results/cpu_cache_preflight.json
tether exec --timeout 30s a100 -- du -sh /home/exouser/xgpu_1017
tether exec --timeout 30s a100 -- df -h -l
```

如果缺动态依赖，补本地相同版本包的必要文件再重试，不自动升级。若缺系统库，先记录具体缺库错误再决定是否sudo安装，不能预装一批“可能用到”的包。`du`空间结果与sudo命令都需回填REPORT。

## 3. H100与A100短回放

H100由协调员准备同样 rootfs / prepared / tools / inputs_dp3 包，目标换为 `exouser@149.165.153.233`，目录仍 `/home/exouser/xgpu_1017`，执行远端命令的node换 `h100`。**跳过A100 Python/env步骤**，使用已有 `/home/exouser/openpi/.venv/bin/python`（已核对主要包版本）。H100 canonical `/data/mw_finetune/dp`已由既有部署使用，只核对现存内容，不改目录/链接或lane。若H100缺某文件，由协调员按file list补入本工作区；本代理不做H100写操作。

```bash
# 协调员；同样先预留约5GiB，不操作现有resident。
tether exec --timeout 360s h100 -- bash /home/exouser/xgpu_1017/tools/run_dp_remote.sh h100 native

# A100公共传输及CPU preflight通过后，本代理获准执行此单次诊断；<=5min。
tether exec --timeout 360s a100 -- bash /home/exouser/xgpu_1017/tools/run_dp_remote.sh a100 native
```

每个profile单独排期，watchdog同上；不能把整串profiles当作一次无上限GPU占用。只为下一步生成新的输出目录。协调员先做4090/H100 native，再A100，随后按差异选择：

```bash
bash exp/offline_search/rounds/r17/xgpu_1017/run_dp_local.sh tf32_off
tether exec --timeout 360s h100 -- bash /home/exouser/xgpu_1017/tools/run_dp_remote.sh h100 tf32_off
tether exec --timeout 360s a100 -- bash /home/exouser/xgpu_1017/tools/run_dp_remote.sh a100 tf32_off
```

后续每次把最后一个参数依序换成 `highest`、`eager`、`math`、`deterministic`、`full_reduction`。这是累积开关，必须检查加载后的flags；有效状态相同则记冗余，发现恢复再回退开关消融。`math`只强制PyTorch SDPA，不能直接约束vendor FA2/TE。A100单次最多900s的扩时需显式改变deadline；超过15min全部交协调员。

## 4. 结果回收与三对CPU比较

只用公共rsync、限速、串行回收本工作区的结果；这不是tether pull。H100回收由协调员执行。目录timestamp请填前述命令打印的实际目录，以下shell变量明确要求真实路径，避免比较错run。

```bash
rsync -a --bwlimit=16384 -e "$XGPU_RSYNC_SSH" exouser@149.165.153.233:/home/exouser/xgpu_1017/results/ "$XGPU_WORK/results/h100_received/"
rsync -a --bwlimit=16384 -e "$XGPU_RSYNC_SSH" exouser@149.165.152.232:/home/exouser/xgpu_1017/results/ "$XGPU_WORK/results/a100_received/"

: "${XGPU_4090_RESULT:?set actual COMPLETE 4090 replay directory}"
: "${XGPU_H100_RESULT:?set actual received COMPLETE H100 replay directory}"
: "${XGPU_A100_RESULT:?set actual received COMPLETE A100 replay directory}"
CUDA_VISIBLE_DEVICES='' .venv/bin/python "$XGPU_WORK/compare.py" --left "$XGPU_4090_RESULT" --right "$XGPU_H100_RESULT" --output "$XGPU_WORK/results/4090_vs_h100.json"
CUDA_VISIBLE_DEVICES='' .venv/bin/python "$XGPU_WORK/compare.py" --left "$XGPU_4090_RESULT" --right "$XGPU_A100_RESULT" --output "$XGPU_WORK/results/4090_vs_a100.json"
CUDA_VISIBLE_DEVICES='' .venv/bin/python "$XGPU_WORK/compare.py" --left "$XGPU_H100_RESULT" --right "$XGPU_A100_RESULT" --output "$XGPU_WORK/results/h100_vs_a100.json"
```

退出码0仅为SCREEN_PASS；2表示INVALID / 超预算 / 缓存分歧 / 未测cache。每个profile的同GPU前后对照也用相同命令。始终读取JSON里的noise floor、boundary coverage和`deployment_equivalence_established=false`；不能只看exit0。

## 5. 扩展至另四个模型

DP是第一条可直接执行、较小显存的(a)+(b)筛查。五个模型不能都承诺“几GiB、任意时刻插跑”：Fast-WAM一个resident约26–28GB，需lane边界。以下为单resident300s的**初始预算，不是实测峰值**：π20GiB，GR双suite24GiB（只加载单suite可另量），DP4GiB，FW36GiB，Cosmos10GiB；另留CUDA context/headroom。A100只在轮到某模型后传该模型必需资产，不能一次搬五套checkpoint/env；先df，已有文件不重复制。

统一入口从选定的sealed campaign.json读runtime identity，不自动选“最新”、不改campaign。本地FW示例（由协调员，先预留36GiB+余量）：

```bash
XGPU_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
.venv/bin/python "$XGPU_WORK/run_campaign.py" \
  --job /data/mw_finetune/eval/fa_fw_f_u_o/campaign.json \
  --python /data/mw_finetune/envs/fastwam/bin/python \
  --inputs "$XGPU_WORK/inputs_dp3" \
  --cache-yaml /data/mw_finetune/eval/fa_fw_f_u_o/assets/cache_yamls/n50_sp_chrg_koff.yaml \
  --output "$XGPU_WORK/results/4090_fastwam_native_$XGPU_STAMP" --memory-gib 36 --profile native
```

Cosmos已另行准备同一批原始真实帧、绑定其实际protocol的 `inputs_cosmos3`。此campaign的cache binding是 **K2（实际NFE1）/C16**；工具按cache binding取sampler，不能从ModelSpec默认K5推断当前MISS。完整命令（协调员，先预留10GiB+余量）：

```bash
XGPU_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
.venv/bin/python "$XGPU_WORK/run_campaign.py" \
  --job /data/mw_finetune/eval/fa_co_r_u_l/campaign.json \
  --python /home/weiland/projects/openpi_ext/lines/cosmos-policy/.venv/bin/python \
  --vendor-path /home/weiland/projects/openpi_ext/lines/cosmos-policy \
  --inputs "$XGPU_WORK/inputs_cosmos3" \
  --cache-yaml /data/mw_finetune/eval/fa_co_r_u_l/assets/cache_yamls/n50_sp_crg_koff.yaml \
  --output "$XGPU_WORK/results/4090_cosmos_native_$XGPU_STAMP" --memory-gib 10 --profile native
```

DP/FW示例cache实际C10；比较器按实际binding截取执行动作，绝不把ModelSpec默认C5当作C10。改变K/C要另建明确的binding并重新验证，不能只改命令标题。

π/GR本次未找到当前LIBERO sealed campaign，故不编造一条能加载其live lane的路径。协调员先选与对应lane一致的sealed job；下面是完整参数接口，要求显式传入已有文件，不准备/更改现有lane：

```bash
: "${XGPU_MODEL_JOB:?sealed campaign.json for pi05 or groot}"
: "${XGPU_MODEL_PYTHON:?matching serving interpreter}"
: "${XGPU_MODEL_CACHE:?fitted cache YAML from that campaign}"
: "${XGPU_MODEL_INPUTS:?real recorded inputs with matching benchmark protocol}"
: "${XGPU_MODEL_VENDOR:?matching vendor source directory}"
: "${XGPU_MODEL_MEMORY_GIB:?20 for pi05 or 24 for dual-suite groot initial admission}"
XGPU_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
.venv/bin/python "$XGPU_WORK/run_campaign.py" --job "$XGPU_MODEL_JOB" --python "$XGPU_MODEL_PYTHON" \
  --cache-yaml "$XGPU_MODEL_CACHE" --inputs "$XGPU_MODEL_INPUTS" --vendor-path "$XGPU_MODEL_VENDOR" \
  --output "$XGPU_WORK/results/model_native_$XGPU_STAMP" --memory-gib "$XGPU_MODEL_MEMORY_GIB" --profile native
```

H100/A100用复制后的相同sealed runtime与路径重定位描述；模型/规范化/recipe identity不可改变。现有prepare_transfer.py只实现DP最小包；**不声称另四模型已可一键迁移到A100**。先完成DP三卡对照，再根据首次分歧与磁盘预算准备下一模型，可显著缩小排障范围。

## 6. layer(c) pilot：协调员长任务

预注册：±2pp的paired SR 95%CI完整落入margin；exact McNemar及首分歧control step；同GPU至少20对重复。每suite100对（10tasks×init0..9），pure/cache各一固定recipe。任务量、估算GPU时间与power不足见REPORT §3；先做4090/H100，再A100。**现有formal_eval `--server-node`只支持local/h100，不能把A100伪装成h100或改设备注册来启动正式campaign。** A100长pilot须协调员安排隔离launcher，保持同样worker/renderer/seed，不能直接复用下方server selector。

当前 formal_eval 只允许结果目录为 `/data/mw_finetune/eval` 的直接子目录（`exp/r17_integration/ops/formal_eval.py:1821`），不能把 `--output` 指到本工作目录来绕过；本轮没有修改此限制或生成长campaign。layer(c)交付的是上述明确样本量/配对/资源设计，启动由协调员在新的正式campaign下安排。所需纯策略参数为 `--official-50 --eval-pool A --init-count 10 --suites libero_spatial --replicas 1 --max-batch 1 --denoising-steps 100 --commit-horizon 5`（DP示例），并用所选已完成baseline的 `--runtime-from` 锁定同一源码。cache pilot须另制同100init cohort的sealed study manifest，绑定matched pure control；现有cache-study是official-50全量，不能假设`--init-count`会裁小cache study。

协调员先审查episode表确为100条，再分配相同worker/renderer、全新端口和GPU时段。首20ep用实测耗时更新预算；目标生产B8/B16/MPS须独立matched-batch检验，当前B1不覆盖。

## 7. CPU复现与新增连续记录

本轮已运行30个CPU单测、真实fit/stall CPU集成，以及完整DP checkpoint **108.17s / 9 rows** 的CPU回放；输入、最终动作和缓存的重复对照均一致。CPU自比较只验证工具能正确读写/比较，不是独立跨GPU对照。

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s "$XGPU_WORK" -p test_xgpu.py -v
CUDA_VISIBLE_DEVICES='' .venv/bin/python "$XGPU_WORK/compare.py" \
  --left "$XGPU_WORK/results/cpu_dp_native_final" --right "$XGPU_WORK/results/cpu_dp_native_final" \
  --output "$XGPU_WORK/evidence/cpu_self_comparison_recheck.json"
```

当前per_step缺观测时不能恢复原episode。新增独立capture运行的外层接入方法（不修改共享serving.py）：

```python
from capture import Recorder
rec = Recorder(new_directory, provenance={"run": run_identity, "real": True}, limit=128)
# 在拥有真实observation / RequestContext / QueryView history的独立worker中，推理前：
rec.add(observation, request_dict, history=query_history,
        episode_start=is_first_query,
        executed_controls_since_last_query=actual_ack_count)
# 原来的推理和控制照常；记录完整序列后：
rec.close()
```

若独立worker已将观测存为本工具JSON/NPZ、per_step包含相对`observation_file`、完整`request`、`history`、`episode_start`和ACK：

```bash
: "${XGPU_CAPTURE_LOG:?new isolated run per_step.jsonl with observation_file}"
CUDA_VISIBLE_DEVICES='' .venv/bin/python "$XGPU_WORK/capture.py" --per-step "$XGPU_CAPTURE_LOG" \
  --limit 128 --max-lines 10000 --output "$XGPU_WORK/inputs_history_new"
```

首轮3帧只能做initial cache smoke。想验证stall，必须有真实连续history/ACK和足够覆盖；不能把3帧重复1000遍充作2995个独立查询。

## 8. 空间 / sudo收据与结束

本轮A100实际新增0 bytes、sudo0条、GPU0次，传输认证未解决。未来每次传输/回放后记录`df -h -l`、`du -sh /home/exouser/xgpu_1017`、模型/环境/结果各项字节和所有sudo准确命令。只准在对应测试结束后清理本任务亲自创建的文件；本执行单不包含删除命令，遵守本轮no-rm。长campaign和正式设备接入另由协调员处理。
