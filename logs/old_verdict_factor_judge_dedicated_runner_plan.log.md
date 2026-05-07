> Status: Plan
> Date: 2026-04-27
> Level: L3

# verdict_factor_judge B1+B2 — observability via `__hit_meta__` + warmup preload

## 0. TL;DR

Phase 1 idx 4 实测后 P0 `all_nan_fallback=warm_start@0.7` 已经把 dead loop 救活（commit `da097bc`，98.6% WARM_START → success matches Ceiling-W = 0.980 exact）。剩下两个独立问题：

1. **观测性零** — 每 yaml 实际 FULL_HIT/WARM_START/MISS 分布只在 server stdout，要 grep + tee 才能拿，事后没法补
2. **89.5% sentinel firing → composer 只跑 10.5% verdict** — factor signal 评估样本不足，paired McNemar 不够稳

本 plan 一并解决两个问题，分两段实施但**同一 plan 同一次 G1 评审**：

- **B1 — Observability**：inference response 加 `__hit_meta__` 字段，client `examples/libero/main.py` always-on 写 per-step JSONL。0 新 ws ctrl，judge.py 不动。~250 行。
- **B2 — Warmup preload**：每 Phase 1 yaml 自动 emit sibling `<eval_yaml_id>__warmup.yaml` 用现成 `AlwaysWarmStartJudge + DumpingJudge` 跑 W=2 trial/task 收集 in-distribution factor raw；client 通过 `fetch_dump{warmup_yaml_id}`（**allowlisted root + `.resolve()` 双解析，非任意 path**）拉回，再 `preload_normalizer_buffer{eval_yaml_id, buffer}` 推给 server-side WarmupPool；eval yaml load 时 `build_per_connection_components` (`cache/config.py`) 给所有 per-conn normalizer preload。3 ctrl + 1 字段扩展全 opt-in。~500 行。

总 ~750 行，Level L3（**wire 协议加 response field + 3 ctrl + cross-module 改 client SDK / server / cache config / runner，符合 WA §2.1 L3 cross-module 定义**，R1-G1R3 修正）。一并 G1 评审 + 一并实施 + 一并验证。

效果对比（vs 当前 commit `e1d2266`）：

| 指标 | 当前 | B1 后 | B2 后 |
|---|---|---|---|
| Per-step hit_type 持久化 | server stdout grep | **client JSONL always-on** | 同 B1 |
| Sentinel firing | 89.5% | 不影响 | **~0%** |
| Composer 实跑 verdict % | 10.5% | 不影响 | **~100%** |
| 新 ws ctrl 数 | n/a | 0 | 3 + 1 字段扩展 |
| `judge.py` 改动 | n/a | 0 行 | 0 行 |

## 1. 问题精确复盘

**dead loop 不是问题，已被 P0 解掉**。Phase 1 idx 4 实测 server-log + `phase1_debug_analyze.py`：

```
hit_type:    WARM_START 2109 (98.6%)  MISS 30 (1.4%)
sentinel:    all_nan_warm_fallback 1915 (89.5%)
composer:    跑了 224 / 2139 verdict (10.5%)，其中 86.6% 过阈值给 WARM_START
client end: success 98/100 = 0.98 = Ceiling-W exact
```

P0 fallback 把 89.5% 的 verdict 从 sentinel-MISS 救成 sentinel-WARM_START。剩下两个独立问题：(a) per-step hit_type 不持久化；(b) 89.5% sentinel 让 composer 只跑 10.5% verdict，factor signal 样本不够 paired McNemar。

## 2. 目标

- **B1 必达**：每 verdict 的 `(hit_type, start_t, winner_id, cp1_score)` + identity 持久化到 client-owned JSONL，事后任何 yaml 都可复查
- **B2 必达**：每 yaml 的 normalizer 在 eval 第一个 verdict 前已经预填 buffer，sentinel firing → ~0%，composer 跑全量
- **不在本 plan**：N-PRELOAD 用 Phase 0 calibration（plan §3.4 单独路径，与 sibling warmup yaml 是两条互补路径）；composer threshold 调参；通用 runner 改造去 batch

## 3. 架构

### 3.1 B1 架构

**0 新 ws ctrl**。inference response 加一个 optional 字段 `__hit_meta__`：

```
[client main.py worker]              [ws]                       [server]
  client.infer(obs)        ──────────►   WebsocketPolicyServer.handle_inference
                                              ↓
                                          conn_policy.infer()
                                              ↓
                                          InferenceInterceptor.infer()
                                              ↓ 在所有 return 路径上 attach
                                          result["__hit_meta__"] = { ... }
                            ◄──────────  result + __hit_meta__
  解出 __hit_meta__
  per_step_writer.write(row)   →  exp/.../per_step/<yaml_id>.jsonl
```

### 3.2 `__hit_meta__` 字段映射（R2-G1R2 修正）

`CheckResult` (`src/openpi/cache/orchestrator.py:56`) 真实字段是 `hit_type`, `payload`, `score`, `entry_id`, `query_keys`, `start_t`。**纠正映射**：

```python
# 在 InferenceInterceptor.infer() 所有 return 路径
result["__hit_meta__"] = {
    "hit_type": cp1_result.hit_type.name,        # "FULL_HIT" | "WARM_START" | "MISS"
    "start_t":  cp1_result.start_t,              # float | None (仅 WARM_START)
    "winner_id": cp1_result.entry_id,            # cp1_result.entry_id, str | None
    "cp1_score": cp1_result.score,               # cp1_result.score, float | None
}
```

JSONL row schema 保留 `winner_id` / `cp1_score` 名字（与 Phase 0 calibration JSONL schema 同名，方便 B2 join），值取 `entry_id` / `score`。

### 3.3 InferenceInterceptor 4 个 return 路径

`__hit_meta__` 必须在所有 return 路径 attach：

| 路径 | 在 interceptor.py 的位置 | hit_type 来源 |
|---|---|---|
| 1. CP1 FULL_HIT 早 return | `interceptor.py:507-518` | `cp1_result.hit_type=FULL_HIT, entry_id, score` |
| 2. MISS 路径（stage3 inference 完成后）| 同函数末尾 | `cp1_result.hit_type=MISS, entry_id, score` |
| 3. WARM_START 路径（stage3 用 cached intermediate）| 同函数末尾 | `cp1_result.hit_type=WARM_START, entry_id, score, start_t` |
| 4. CP1 cache check 失败 / disabled fallback | 函数初始 fallback | 占位 `{"hit_type": "MISS", "start_t": null, "winner_id": null, "cp1_score": null}` |

### 3.4 B2 架构 — 严格双 yaml_id 命名契约（R1-G1R4 修正）

**两个独立的 yaml_id 必须始终分清**：

| 名称 | 含义 | 由谁产生 |
|---|---|---|
| `eval_yaml_id` | 真正 eval 的 yaml stem，e.g. `clip_w7_d4_phase1_f_full_d_all_t_dual_07` | `phase1_spec.py` 写文件名时 |
| `warmup_yaml_id` | 严格派生：`f"{eval_yaml_id}__warmup"`，无可调字符串 | 同上，命名约定 enforcement |

**WarmupPool 用 `eval_yaml_id` key**（per-conn build 查找时也用 `bundle.yaml_id == eval_yaml_id`）；**warmup dump 文件路径**用 `warmup_yaml_id`：`/tmp/openpi_warmup/{warmup_yaml_id}.jsonl`。

**3 新 ws ctrl + 1 现有 ctrl 字段扩展**。每 yaml 跑 warmup → preload → eval 三段：

```
[client run_phase.py]                   [server]
for eval_yaml_id in phase_dir:
  warmup_yaml_id = f"{eval_yaml_id}__warmup"   # 严格派生

  1. ws.send(load_cache_config{
       yaml_content: <warmup_yaml>,
       yaml_id: warmup_yaml_id              ← 写进 CurrentCacheBundle.yaml_id
     })
                                       --> CurrentCacheBundle.yaml_id = warmup_yaml_id
                                       --> server 启动时已 mkdir /tmp/openpi_warmup mode 0o700
                                       --> warmup yaml 的 dump.deferred=True 触发 server 端
                                           rewrite (§3.5)：
                                             cache_config.checkpoints[cp1].judge.dump.path
                                               = f"/tmp/openpi_warmup/{warmup_yaml_id}.jsonl"
                                       --> bundle 加载 warmup yaml
  2. spawn main.py(num_workers=5,
                    num_trials_per_task=2,
                    task_ids=0..9)            ← 直接 subprocess.run，shim 不经 common runner
                                       --> 10 task × 2 trial = 20 ep × 21 verdict ≈ 420 verdict
                                       --> DumpingJudge 写 /tmp/openpi_warmup/<warmup_yaml_id>.jsonl
                                       --> AlwaysWarmStart 永远 WARM_START 0.7
  3. ws.send(fetch_dump{
       warmup_yaml_id: warmup_yaml_id        ← 注意字段名是 warmup_yaml_id 不是 yaml_id
     })
                                       --> server resolve + .resolve() 防符号链 (R4-G1R4):
                                             root = Path("/tmp/openpi_warmup").resolve()
                                             path = (root / f"{warmup_yaml_id}.jsonl").resolve()
                                             if not path.is_relative_to(root): reject
                                             return path.read_bytes()
  4. client 解析 dump → 按 key 聚合
     {"f1a_a_jerk": [v1,v2,...], ...}
     截到 window_size=50 per key
  5. ws.send(load_cache_config{
       yaml_content: <eval_yaml>,
       yaml_id: eval_yaml_id
     })
                                       --> CurrentCacheBundle.yaml_id = eval_yaml_id
                                       --> bundle 加载 eval yaml (无 dump block)
  6. ws.send(preload_normalizer_buffer{
       eval_yaml_id: eval_yaml_id,           ← 注意字段名是 eval_yaml_id
       buffer: <merged dict>
     })
                                       --> WarmupPool[eval_yaml_id] = buffer
  7. spawn main.py(num_workers=5,
                    num_trials_per_task=10,
                    task_ids=0..9)
                                       --> serve_policy.py 动态 bundle 分支调:
                                             build_per_connection_components(
                                               bundle.cache_config,
                                               bundle.shared_storage,
                                               yaml_id=bundle.yaml_id,        ← 新参数
                                               quiet=True,
                                             )
                                       --> config.py 内部:
                                             if yaml_id and (pool := WarmupPool.get(yaml_id)):
                                                 normalizer.preload_buffer(pool)
                                          composer 第 1 verdict 就工作
                                       <-- per inference 回 __hit_meta__   ← B1
     client.main.py 写 per-step JSONL  ← B1
  8. ws.send(unload_warmup_buffer{
       eval_yaml_id: eval_yaml_id            ← 用 eval_yaml_id；server 派生 warmup name
     })
                                       --> WarmupPool.pop(eval_yaml_id, None)
                                       --> 删除 /tmp/openpi_warmup/{eval_yaml_id}__warmup.jsonl
                                          (server 派生路径 + .resolve() 安全检)
next yaml ↻
```

**测试 `test_warmup_protocol.py` 必须断言**（R1-G1R4 验收）：
- fetch → preload → unload 的完整流程跑完后：
  - `WarmupPool` 内 `eval_yaml_id` key 已清
  - `/tmp/openpi_warmup/<warmup_yaml_id>.jsonl` 文件已删
  - `<eval_yaml_id>` 路径下的文件 / pool entry 都没被误动

**Warmup verdict 数核算**：W=2 默认 → 10 task × 2 trial = 20 warmup ep × ~21 verdict ≈ 420 verdict / yaml；5 worker 平均每 worker ~4 ep × 21 = 84 verdict；每 key 累 valid sample ~ 420 × (1 - 0.44) ≈ 235 / key，远超 window=50。

### 3.5 B2 sibling warmup yaml 结构 — 用 `dump.deferred=True` 避开 CI 校验失败（R3-G1R4 修正）

**问题（R3-G1R4 抓出）**：当前 `_validate_dump_static` (`config.py:553-558`) 强制 `dump.path` 非空且 parent 目录存在。`phase1_spec.py` 生成的 warmup yaml 走 `validate_cache_config` 时（CI / `phase1_debug_analyze.py` / 离线工具），`/tmp/openpi_warmup` 不一定存在 → 拒收。

**修法**：`DumpConfig` 加 `deferred: bool = False` 字段。`deferred=True` 表示"path 由 server 在 `load_cache_config` 时按命名约定填入"，validator 跳过 path 检查。

`DumpConfig` 改动（B2.3 spec）：

```python
# src/openpi/cache/config.py:147 dataclass DumpConfig
@dataclass
class DumpConfig:
    path: str = ""
    config_id: str = ""
    factors: list[FactorConfig] = field(default_factory=list)
    deferred: bool = False              # NEW (R3-G1R4)
```

`_validate_dump_static` (`config.py:545-558`) 改动：

```python
parent = os.path.dirname(dump.path) or "."
if dump.deferred:
    # Path will be filled by server-side load_cache_config handler from
    # warmup_dump_root + config_id. Skip path / parent-dir checks; still
    # validate config_id non-empty + factors registered (existing rules).
    pass
elif not dump.path:
    errors.append(f"{prefix}.judge.dump.path is required (non-empty string)")
elif not os.path.isdir(parent):
    errors.append(
        f"{prefix}.judge.dump.path parent directory does not exist: {parent!r}"
    )
```

Server-side `load_cache_config` ctrl handler（B2.3 spec，在 `websocket_policy_server.py`）在调 `validate_cache_config` 之前把 deferred 路径填好：

```python
# pseudo-code in load_cache_config ctrl handler
cfg = parse_yaml(yaml_content)
yaml_id = msg.get("yaml_id")
for cp in cfg.checkpoints.values():
    dump = cp.judge.dump
    if dump and dump.deferred:
        if not yaml_id:
            return error_ack("deferred dump requires yaml_id in load_cache_config")
        dump.path = str(WARMUP_DUMP_ROOT / f"{yaml_id}.jsonl")
        dump.deferred = False  # downstream validator passes
validate_cache_config(cfg)  # 现在 path 真实存在
```

Warmup yaml 实际产出（`phase1_spec.py` emit）：

```yaml
# <eval_yaml_id>__warmup.yaml
keys: { ... }                       # 同 eval sibling
key_builder: { type: <cfg_kb> }
checkpoints:
  cp1:
    judge:
      type: always_warm_start         # 现成 AlwaysWarmStartJudge (judge.py:148)
      start_t: 0.7
      dump:                           # 现成 DumpingJudge (judge.py:400)
        deferred: true                # ← server-side fill (R3-G1R4)
        # path: omitted (server fills)
        config_id: <eval_yaml_id>__warmup    # = warmup_yaml_id
        factors: [ <同 eval yaml 的 5 因子，含 W-MIX 5 窗口 × 4 desc + F2 K=5> ]
    search_strategy: { top_k: 5 ... }  # F2 K=5 撑 top_k
backend: { ... }                       # 同 eval sibling
```

测试要点（B2.3 + B2.5 验收）：
- `test_dump_config_deferred.py`：`validate_cache_config` 于 deferred=True + path="" 通过；deferred=False + path="" 仍拒
- `test_phase1_spec_warmup_yamls.py`：phase1_spec.py 生成的 24 个 warmup yaml 用 `validate_cache_config` 全过（**CI host 上 `/tmp/openpi_warmup` 不存在也通**）— 这是 R3-G1R4 关键验收
- `test_load_cache_config_deferred.py`：server 端 ctrl handler 把 deferred path 正确填成 `<warmup_dump_root>/<yaml_id>.jsonl`；缺 yaml_id 时返回 error ack

`AlwaysWarmStartJudge` (`judge.py:148`) + `DumpingJudge` (`judge.py:400`) 全部现成复用，**`judge.py` 0 改动**。

### 3.6 Wire ctrl 安全模型（R4-G1R2 + R4-G1R4 修正）

**`fetch_dump` 不接受任意路径，且 root + candidate 都先 `.resolve()` 防符号链**（R4-G1R4 修正）。Server 端：

```python
# websocket_policy_server.py 新 handler
WARMUP_DUMP_ROOT = Path("/tmp/openpi_warmup").resolve()  # 启动时 resolve 一次

def _handle_fetch_dump(self, payload):
    warmup_yaml_id = payload.get("warmup_yaml_id")        # 字段名严格匹配 §3.4
    if not warmup_yaml_id or "/" in warmup_yaml_id or ".." in warmup_yaml_id:
        return {"__ack__": "error", "msg": "invalid warmup_yaml_id"}
    # 双 .resolve() — root + candidate 都 follow symlinks 后再比，
    # 满足 §11 风险表 "Server 启动时 /tmp/openpi_warmup 已被恶意软链" 防护语义
    candidate = (WARMUP_DUMP_ROOT / f"{warmup_yaml_id}.jsonl").resolve()
    if not candidate.is_relative_to(WARMUP_DUMP_ROOT):
        return {"__ack__": "error", "msg": "path traversal rejected"}
    if not candidate.exists():
        return {"__ack__": "error", "msg": "dump not found"}
    return {"__ack__": "fetch_dump", "content": candidate.read_bytes()}
```

**`unload_warmup_buffer{eval_yaml_id}`**（R1-G1R4 修正：用 eval_yaml_id 不是 yaml_id）：server 派生 warmup name 删 allowlisted root 内文件 + 同样 .resolve() 检查。

```python
def _handle_unload_warmup_buffer(self, payload):
    eval_yaml_id = payload.get("eval_yaml_id")
    if not eval_yaml_id or "/" in eval_yaml_id or ".." in eval_yaml_id:
        return {"__ack__": "error", "msg": "invalid eval_yaml_id"}
    WARMUP_POOL.pop(eval_yaml_id, None)
    warmup_yaml_id = f"{eval_yaml_id}__warmup"
    candidate = (WARMUP_DUMP_ROOT / f"{warmup_yaml_id}.jsonl").resolve()
    if candidate.is_relative_to(WARMUP_DUMP_ROOT) and candidate.exists():
        candidate.unlink()
    return {"__ack__": "unload_warmup_buffer"}
```

`/tmp/openpi_warmup` server 启动时（`scripts/serve_policy.py` 的 main 入口）：

```python
# 启动 hook
WARMUP_DUMP_ROOT = Path(args.warmup_dump_root or "/tmp/openpi_warmup")
WARMUP_DUMP_ROOT.mkdir(mode=0o700, exist_ok=True)
# Owner / mode sanity check
st = WARMUP_DUMP_ROOT.stat()
if st.st_uid != os.getuid() or (st.st_mode & 0o077):
    raise RuntimeError(f"WARMUP_DUMP_ROOT {WARMUP_DUMP_ROOT} has unsafe owner/mode")
```

`--warmup-dump-root` CLI flag（B2.3 spec，R5-G1R4 修正：归 server stage 拥有，不是 client SDK）：在 `scripts/serve_policy.py` 加 `Args.warmup_dump_root: Optional[str] = None`，默认 `/tmp/openpi_warmup`。

## 4. 组件清单

### 4.1 新建（client）

| 文件 | 内容 | 行数 |
|---|---|---|
| `exp/verdict_factor_judge/per_step_log_writer.py` | per-step JSONL writer（详见 §7 lifetime 设计）| ~120 |
| `exp/verdict_factor_judge/run_phase.py` | **直接 subprocess.run main.py**（单一路径，不通过 common runner）：(1) `_send_cache_config` ws helper 切到 warmup yaml（payload 含 `yaml_id=warmup_yaml_id`）；(2) launch main.py warmup 跑 W=2 ep；(3) `fetch_dump{warmup_yaml_id}` 拉回；(4) `_send_cache_config` 切到 eval yaml（payload 含 `yaml_id=eval_yaml_id`） + `preload_normalizer_buffer{eval_yaml_id, buffer}`；(5) launch main.py eval 跑 100 ep；(6) `unload_warmup_buffer{eval_yaml_id}`；(7) per-yaml summary aggregate | ~350 |

### 4.2 新建（server）

| 文件 | 内容 | 行数 |
|---|---|---|
| `src/openpi/cache/warmup_pool.py` | yaml_id-key 共享 normalizer warmup 池；map `yaml_id → {key: list[float]}`；线程安全；带 LRU 上限 + 启动清空 | ~120 |
| `tests/cache/test_warmup_pool.py` | concurrent insert + read + clear + LRU eviction | ~100 |

### 4.3 修改（server，code-grounded R2-G1R4 + R3-G1R4 + R5-G1R4 修正）

| 文件 | 改动 | 阶段 |
|---|---|---|
| `src/openpi/cache/interceptor.py` | `InferenceInterceptor.infer()` 4 个 return 路径 attach `result["__hit_meta__"]`（详见 §3.3）| **B1** |
| `src/openpi/cache/config.py` | (a) `DumpConfig` (line 147) 加 `deferred: bool = False` 字段（R3-G1R4，详见 §3.5）；(b) `_validate_dump_static` (line 545-558) 在 `deferred=True` 时跳过 path / parent-dir 检查；(c) `build_per_connection_components` (line 1347) 签名加 `yaml_id: Optional[str] = None`（R2-G1R4，明确 keyword-only），内部检 `WarmupPool.get(yaml_id)` 调 `normalizer.preload_buffer(pool)`（无 pool 或无 yaml_id 时跳过）| B2 |
| `src/openpi/serving/websocket_policy_server.py` | (a) `CurrentCacheBundle` (line 69) 加 `yaml_id: Optional[str] = None` 字段（R6-G1R2）；(b) `load_cache_config` ctrl handler 解 payload `yaml_id` → 写入 bundle；如解析后的 `cfg` 含 `dump.deferred=True`，handler 把 dump.path 填成 `<warmup_dump_root>/<yaml_id>.jsonl` 后才调 `validate_cache_config`（R3-G1R4，详见 §3.5）；(c) 新 ctrl handler `_handle_fetch_dump` / `_handle_preload_normalizer_buffer` / `_handle_unload_warmup_buffer`（详见 §3.6 + §5.2）| B2 |
| `src/openpi/cache/components/factors/normalizers/__init__.py` | `PercentileRollingNormalizer.preload_buffer(values: dict[str, list[float]])` 方法（详见 §7.5）| B2 |
| `scripts/serve_policy.py` | (a) Args 加 `warmup_dump_root: Optional[str] = None`（默认 `/tmp/openpi_warmup`）；(b) main 入口 mkdir warmup root + 检 owner/mode；(c) **dynamic-bundle 分支 (line 261)** 调用从 `build_per_connection_components(bundle.cache_config, bundle.shared_storage, quiet=True)` 改成 `build_per_connection_components(bundle.cache_config, bundle.shared_storage, yaml_id=bundle.yaml_id, quiet=True)`；**静态 --cache_config 分支 (line 300)** 不传 yaml_id（默认 None，无 warmup pool 路径）| B2 |

`judge.py` / `CompositeJudge` / `AlwaysWarmStartJudge` / `DumpingJudge` 全部不动。

**测试 `test_serve_policy_yaml_id_propagation.py` 必须断言**（R2-G1R4 验收）：
- mock `WarmupPool` 注入一个 entry `{"f1a_a_jerk": [0.1] * 60}`
- mock 一个 `CurrentCacheBundle` with `yaml_id="test_eval_yaml"`
- 模拟新 worker connection → `build_per_connection_components(..., yaml_id="test_eval_yaml")` 调用
- 断言返回的 `judges["cp1"]._normalizer._buffers["f1a_a_jerk"]` 含 50 个值（buffer maxlen 截尾）
- 反向 case：yaml_id=None → buffer 空

### 4.4 修改（client SDK）— **R5-G1R3 简化**

| 文件 | 改动 |
|---|---|
| `packages/openpi-client/src/openpi_client/websocket_client_policy.py` | **B1 不需代码改动** — `infer()` 已经返回 unpacked dict，`__hit_meta__` 自动透传给 caller。**只加一个 compatibility test 锁住行为**（R5-G1R3）。**B2 加 ctrl helpers** `fetch_dump` / `preload_normalizer_buffer` / `unload_warmup_buffer` + `load_cache_config` 加 `yaml_id` kwarg | B1: 0 行 + test；B2: ~80 行 |

### 4.5 修改（client app）

| 文件 | 改动 | 阶段 |
|---|---|---|
| `examples/libero/main.py` | (B1) 接 `__hit_meta__`；新 CLI flag `--per-step-log-dir <dir>` + `--yaml-id <str>`（默认 `None` = no-op）；写 per-step JSONL 行（含 R4-G1R3 修正：用现有 `global_episode_id` 整数，不构造新字符串） | B1 |

### 4.6 修改（实验配置）

| 文件 | 改动 | 阶段 |
|---|---|---|
| `exp/verdict_factor_judge/phase1_spec.py` | 24 yaml 同时 emit sibling `<eval_yaml_id>__warmup.yaml`（48 yaml 总）；**warmup yaml 用 `dump.deferred=True` + 省略 `dump.path`**（R3-G1R4，让 `validate_cache_config` 不依赖 `/tmp/openpi_warmup` 存在；server-side `load_cache_config` ctrl handler 在 load 时按 `<warmup_dump_root>/<warmup_yaml_id>.jsonl` 命名约定填进去）；`dump.config_id` 一律 = `<warmup_yaml_id>` | B2 |
| `logs/verdict_factor_judge_phase0_phase1_run_commands.log.md` | runbook §2.2 改用 `run_phase.py` | B2 |

### 4.7 测试

| 文件 | 测试 case | 阶段 |
|---|---|---|
| `tests/cache/test_interceptor_hit_meta.py` | 4 个 return 路径都 attach `__hit_meta__`：FULL_HIT 早返、MISS 末返、WARM_START 末返、CP1 disabled fallback | B1 |
| `tests/exp/verdict_factor_judge/test_per_step_writer.py` | writer lifetime 跨多 ep 不丢行（§7）；行字段含 `subset_init_state_idx + orig_init_state_idx + episode_id + step_idx`；merge 后排序键确定 | B1 |
| `tests/serving/test_websocket_response_hit_meta.py` | response 含 `__hit_meta__` 时老 client 能正常解；不含时 main.py 不崩；SDK 0 改动 backward-compat smoke | B1 |
| `tests/exp/verdict_factor_judge/test_run_phase_shim.py` | shim runner 完整流程（warmup → preload → eval → unload）+ per-yaml summary 字段完整 | B2 |
| `tests/cache/components/factors/test_normalizers_preload.py` | `preload_buffer` 后 percentile 立即可算；deque maxlen 截尾 | B2 |
| `tests/cache/test_warmup_pool.py` | eval_yaml_id 隔离 + 并发 + LRU eviction | B2 |
| `tests/serving/test_warmup_protocol.py` | 3 个 ctrl roundtrip（fetch_dump / preload / unload）+ load_cache_config yaml_id 字段；**双 yaml_id 名称契约 e2e（R1-G1R4 必跑）**：从 fetch (`warmup_yaml_id`) → preload (`eval_yaml_id`) → unload (`eval_yaml_id`) 跑完后断言 `WarmupPool[eval_yaml_id]` 已清 + `<warmup_dump_root>/<warmup_yaml_id>.jsonl` 已删 + `<eval_yaml_id>` 路径与 pool 都未被误动；`fetch_dump{warmup_yaml_id: "../etc/passwd"}` 应被 reject（R4-G1R2 + R4-G1R4：含 `/` 或 `..` 均拒；并测一个**符号链接 traversal** case 验 `.resolve()` 防护生效） | B2 |
| `tests/cache/test_dump_config_deferred.py` | **R3-G1R4 必跑**：`validate_cache_config` 于 `dump.deferred=True + path=""` 通过；`deferred=False + path=""` 仍拒；`deferred=True + path="/some/real/path"` 也通过（path 给了就不 fall back）| B2 |
| `tests/exp/verdict_factor_judge/test_phase1_spec_warmup_yamls.py` | **R3-G1R4 关键验收**：phase1_spec.py 生成的 24 个 warmup yaml 用 `validate_cache_config` 全过；**测试 host 上 `/tmp/openpi_warmup` 不存在也通**（用 tmpdir + 显式确认 `/tmp/openpi_warmup` 不存在的 fixture）| B2 |
| `tests/serving/test_load_cache_config_deferred.py` | server 端 ctrl handler 把 `dump.deferred=True` + payload 含 `yaml_id` 的 yaml load 时正确填 `dump.path = <warmup_dump_root>/<yaml_id>.jsonl` 后才 validate；缺 yaml_id 时返回 error ack；非 deferred 走旧路径不动 | B2 |
| `tests/serving/test_serve_policy_yaml_id_propagation.py` | **R2-G1R4 必跑**：mock `WarmupPool` + `CurrentCacheBundle.yaml_id` → `build_per_connection_components(..., yaml_id=...)` 调用断言 `judges["cp1"]._normalizer._buffers` 含预填值；yaml_id=None 路径 buffer 空 | B2 |

## 5. Wire 协议增量

### 5.1 B1 — 1 个 response 字段（无 ctrl）

inference response（在 `InferenceInterceptor.infer()` attach，最终经 `WebsocketPolicyServer.handle_inference` 透传）：

```json
{
  "actions": [...],
  "__hit_meta__": { "hit_type": "...", "start_t": ..., "winner_id": "...", "cp1_score": ... }
}
```

老 client 不解析 `__hit_meta__` → 透明忽略，零破坏。

### 5.2 B2 — 3 个新 ctrl + 1 个 ctrl 字段扩展（R1-G1R4 双 yaml_id 字段名严格区分）

| ctrl name | direction | payload | response | 语义 |
|---|---|---|---|---|
| `load_cache_config` (extend) | client→server | `{yaml_content, yaml_id: str}` | `{__ack__: load_cache_config, version}` | **现有 ctrl 加 `yaml_id` 字段**。`yaml_id` 是当前 load 的 yaml 的 stem（warmup yaml 时 = `warmup_yaml_id`，eval yaml 时 = `eval_yaml_id`），server 写进 `CurrentCacheBundle.yaml_id`；如 cfg 含 `dump.deferred=True`，server 用此 yaml_id 填 dump.path 再 validate（R3-G1R4，§3.5）；老 client 不带 yaml_id 就 `bundle.yaml_id = None`，B2 路径全跳过（向后兼容）|
| `fetch_dump` | client→server | **`{warmup_yaml_id: str}`**（**字段名严格区分**，R1-G1R4）| `{__ack__: fetch_dump, content: bytes}` 或 `{__ack__: error, msg: str}` | server 端 .resolve() allowlist `(warmup_dump_root / f"{warmup_yaml_id}.jsonl").resolve()` ∈ root（**非任意 path**，R4-G1R2 + R4-G1R4）；含 `/` 或 `..` 的 warmup_yaml_id 立即 reject |
| `preload_normalizer_buffer` | client→server | **`{eval_yaml_id: str, buffer: dict[str, list[float]]}`**（**字段名严格区分**，R1-G1R4）| `{__ack__: preload_normalizer_buffer}` | server 把 buffer 塞 `WarmupPool[eval_yaml_id]` |
| `unload_warmup_buffer` | client→server | **`{eval_yaml_id: str}`**（**字段名严格区分**，R1-G1R4）| `{__ack__: unload_warmup_buffer}` | 清 `WarmupPool[eval_yaml_id]` + server 派生 `warmup_yaml_id = f"{eval_yaml_id}__warmup"` 后删 `<warmup_dump_root>/<warmup_yaml_id>.jsonl`（同样 .resolve() allowlist）|

所有新 ctrl optional，老 client 不发就不动；老 server 不识别返回 `{__ack__: ignored}`。**字段命名严格区分** `eval_yaml_id` / `warmup_yaml_id` —— `load_cache_config` 用通用 `yaml_id`（语义随 load 的是哪份 yaml 而定），其它 ctrl 字段名一律含明确前缀避免 R1-G1R4 类型混淆。

## 6. 数据流

见 §3.1（B1）+ §3.4（B2）architecture 图。

## 7. Per-step JSONL writer 设计（R7-G1R1 + R3-G1R2 + R4-G1R3）

### 7.1 输出 path

`exp/verdict_factor_judge/data/phase1/<cfg>/per_step/<yaml_id>.jsonl`

`yaml_id` 由 `run_phase.py` shim 从 yaml file stem 计算，通过 `--yaml-id` CLI flag 传给 main.py。

### 7.2 行 schema

```json
{
  "yaml_id":"clip_w7_d4_phase1_f_full_d_all_t_dual_07",
  "task_id":3,
  "subset_init_state_idx":7,
  "orig_init_state_idx":7,
  "episode_id":42,
  "step_idx":17,
  "phase":"eval",
  "hit_type":"WARM_START",
  "start_t":0.7,
  "winner_id":"episode_0019:42",
  "cp1_score":0.92
}
```

字段说明（**R4-G1R3 episode_id 改用现有 integer**）：
- `subset_init_state_idx`：filter 模式下 episode 在 subset 里的序号（main.py 已有变量）
- `orig_init_state_idx`：原 trajectory 中的 init state 序号；**join Phase 0 calibration 必须用这个**
- `episode_id`：**main.py 现有 `global_episode_id` 整数**（R4-G1R3 — 不构造新字符串）。warmup ep 和 eval ep 各自独立计数（warmup 跑完 main.py 退出，eval 重新计数），用 `phase` 字段区分
- `step_idx`：episode 内的物理 step（每 ep reset 为 0）
- `phase`：`"warmup"` / `"eval"`，分析阶段直接 filter

### 7.3 Writer lifetime + concurrent design

**Writer lifetime = main.py worker thread 整个生命周期**（不是 per-episode）。一个 worker 处理多个 episode，文件 handle 全程开着。

```python
# 在 main.py worker thread 启动时 (per_step_log_writer.py)
class PerStepWriter:
    def __init__(self, path: Path):
        self._fh = open(path, "a", buffering=1)  # line-buffered
    def write_row(self, row: dict) -> None:
        self._fh.write(json.dumps(row) + "\n")
        # buffering=1 自动 flush per line
    def close(self) -> None:
        self._fh.close()

# 5 worker 各自起 writer 写到独立 temp file
worker_writers = [
    PerStepWriter(Path(args.per_step_log_dir) / f"_worker_{wid}.jsonl")
    for wid in range(args.num_workers)
]

# 每 verdict 在 worker thread 内
worker_writers[wid].write_row({...})

# main.py 退出前 (atexit hook)
for w in worker_writers:
    w.close()

# Final merge — sort by (task_id, subset_init_state_idx, episode_id, step_idx)
merged = []
for wid in range(args.num_workers):
    merged.extend(json.loads(line) for line in open(worker_temp_paths[wid]))
merged.sort(key=lambda r: (r["task_id"], r["subset_init_state_idx"], r["episode_id"], r["step_idx"]))

with open(Path(args.per_step_log_dir) / f"{args.yaml_id}.jsonl", "w") as fh:
    for row in merged:
        fh.write(json.dumps(row) + "\n")

# Cleanup: 删 temp files
for p in worker_temp_paths:
    p.unlink()
```

**关键设计点**：
- 每 worker 自己 file handle = 0 lock 竞争
- `buffering=1` line-buffered = 每 row flush，崩溃时已写部分保留
- Merge 在 main.py 退出前一次性做（不是 per-episode），避免 reopen
- Sort key `(task_id, subset_init_state_idx, episode_id, step_idx)` 4-tuple：含 episode_id 后跨 ep `step_idx` reset 也唯一

### 7.4 Per-yaml summary（runner shim 写）

`exp/verdict_factor_judge/data/phase1/<cfg>/per_yaml_summary.jsonl`：

```json
{"yaml_id":"...","success_rate":0.98,"n_ep":100,"n_full_hit":120,"n_warm_start":1830,"n_miss":150,"warmup_verdicts":420,"eval_verdicts":2100}
```

直接喂 plan §6.1 列入的 `phase1_factor_ablation_summary.py`。

## 8. 向后兼容

- **`exp/common/run_cache_experiments.py` 完全不动** — 通用 runner 服务其它实验（warm_start / random_periodic_gate / phase1_libero_*）
- **现有 yaml schema 字段全部 optional** — 不设 `judge.dump`（warmup yaml 才有）时 B2 逻辑全跳过
- **inference response `__hit_meta__` optional** — 老 client 忽略
- **新 ws ctrl 全部 optional** — 老 client 不发，server 走旧路径
- **`AlwaysWarmStartJudge` / `DumpingJudge` / `CompositeJudge` / `judge.py` 全部不动** — 现有 tests 全过
- **`load_cache_config` 加 `yaml_id` 字段 backward-compat** — 老 client 不带，server 端 `bundle.yaml_id = None`（B2 preload 路径全跳过，不影响老行为；与 §5.2 / §11 语义一致）

## 9. 架构 doc 更新（R1-G1R3 必做，L3 要求）

`docs/architecture/cache_system.md` 加新 §5.13 「Wire-level observability + warmup preload protocol」：
- `__hit_meta__` response 字段定义（B1）
- 3 个新 ws ctrl 完整 schema（B2）
- `CurrentCacheBundle.yaml_id` 字段语义
- WarmupPool 生命周期（load_cache_config 写入 + per-conn build 读取 + unload 清空）
- Server-owned warmup dump root 安全模型（`/tmp/openpi_warmup` allowlist）
- 与现有 §5.6 SimilarityJudge purity contract / §5.12 Verdict Factor System 的关系

更新 `docs/README.md` 索引同步该 §。

## 10. 测试策略

### 10.1 单元（必跑）

见 §4.7 表。

### 10.2 集成（必跑）

- 现有 `test_websocket_policy_server` 全部：response 加字段 + 新 ctrl 不破现有 episode_start / prefill_trajectory / load_cache_config 测试
- 现有 `test_interceptor` 系列全部：4 个 return 路径行为不变（除多 attach `__hit_meta__`）
- 现有 `test_dumping_judge` 全部：sibling warmup yaml 复用 DumpingJudge，行为零变化
- 现有 LIBERO main.py 测试：新 CLI flag 默认无值时行为零变化

### 10.3 端到端（land 后云端跑）

云端跑 1 个 yaml（如 idx 4 F-FULL × T-DUAL_07）：

| 验证点 | 期望 |
|---|---|
| (B1) 100 ep 跑完后 `<yaml_id>.jsonl` 行数 | ≈ 2100（eval phase） |
| (B1) hit_type 字段分布 | 与 server stdout grep（OPENPI_CACHE_VERDICT_DEBUG=1 同时开作为对照）一致 |
| (B1) Sort 后行序确定 | task_id ↑, subset_init_state_idx ↑, episode_id ↑, step_idx ↑ |
| (B2) Warmup 阶段 stdout | 显示 warmup yaml load + 跑 ~420 verdict |
| (B2) Eval 阶段 server stdout sentinel firing | 接近 0 |
| (B2) Per-yaml summary success_rate | ∈ [Ceiling-A − 5pp, Ceiling-W + 2pp] |

## 11. 风险登记

| 风险 | 可能性 | 影响 | 缓解 |
|---|---|---|---|
| `__hit_meta__` 加在 ws response 破老 client | 低 | 高 | optional 字段，老 client 自动忽略；CI 加 backward-compat smoke test |
| InferenceInterceptor 改动遗漏某 return 路径 → __hit_meta__ 缺失 | 中 | 中 | `test_interceptor_hit_meta.py` 4 case 覆盖（FULL_HIT 早返/MISS 末返/WARM_START 末返/CP1 disabled fallback） |
| `cp1_result.entry_id` / `cp1_result.score` 字段名错 → AttributeError | 低 | 高 | 已 grep `storage_types.py` + `orchestrator.py:56` 确认字段名（R2-G1R2 修正）；单元测试覆盖 |
| Per-step JSONL 多线程写竞争 | 低 | 中 | 每 worker 自己临时 file，main.py exit 前 merge；不用全局 lock（§7.3）|
| Worker thread 崩溃丢失未 flush 的 row | 低 | 低 | `buffering=1` line-buffered，每行 flush；崩溃时已写部分保留 |
| Merge sort key 重复 | 低 | 中 | `(task_id, subset_init_state_idx, episode_id, step_idx)` 4-tuple，episode_id 是 main.py 整数 unique；测试断 sort 后无重复 key |
| `fetch_dump{warmup_yaml_id}` 路径穿越攻击 | 中 | 高 | warmup_yaml_id 含 `/` 或 `..` 立即 reject；root + candidate 都 `.resolve()` 后 `is_relative_to` 比对（R4-G1R2 + R4-G1R4，§3.6）；测试含 symlink traversal case |
| **eval_yaml_id / warmup_yaml_id 名称混用导致 unload 删错文件 / 清错 pool entry** | 中（之前 G1R3 plan 真有这 bug）| 高 | wire 协议字段名严格区分：fetch_dump 用 `warmup_yaml_id`，preload/unload 用 `eval_yaml_id`（R1-G1R4，§5.2）；`unload` 内部 server 自己派生 warmup name（不经客户端）；`test_warmup_protocol.py` e2e 必断言 unload 后只删了对应 warmup 文件 + 清了对应 pool entry |
| WarmupPool 跨 conn 内存泄漏 | 中 | 中 | yaml 跑完 client 显式 `unload_warmup_buffer`；server 加 LRU 上限（默认 100 个 eval_yaml_id）+ 启动清空 |
| Server 启动时 `/tmp/openpi_warmup` 已被恶意软链 | 低 | 中 | server 启动时 `os.makedirs(..., mode=0o700, exist_ok=True)` + 检 owner / mode；运行时所有路径解析用 `.resolve()` 跟随软链后再 `is_relative_to` 比对（R4-G1R4） |
| **生成的 warmup yaml 在 CI / `phase1_debug_analyze.py` 等离线场景 validate 失败**（`/tmp/openpi_warmup` 不存在）| 高（之前 G1R3 plan 真有这 bug）| 高 | `DumpConfig` 加 `deferred: bool` 字段，validator 在 `deferred=True` 时跳过 path / parent-dir 检查；server-side `load_cache_config` ctrl handler 在 validate 之前按 `<warmup_dump_root>/<yaml_id>.jsonl` 命名约定填 path（R3-G1R4，§3.5）；`test_phase1_spec_warmup_yamls.py` 在 `/tmp/openpi_warmup` 不存在的 fixture 下 validate 全 24 warmup yaml 通过 |
| **`build_per_connection_components` 签名变更影响其它实验** | 低 | 中 | 新参数 `yaml_id: Optional[str] = None` keyword-only + 默认 None（R2-G1R4）；现有 caller（serve_policy.py 静态分支 line 300、其它实验运行时）不传时与改前等价；动态分支 line 261 显式传 `bundle.yaml_id`；新签名走 `test_serve_policy_yaml_id_propagation.py` 验证；其它实验 unit / integration 全过 |
| Sibling warmup yaml 文件数 24 → 48 | 低 | 低 | phase1_spec.py 自动生成；config 目录大小翻倍但仍小 |
| 24 yaml × warmup 20 ep = 480 ep 额外开销 | 高 | 低 | 480 ep × ~10 s/verdict × ~21 verdict ÷ 5 worker ≈ 5.6 hour 额外。换来 sentinel 0% / composer 全量样本，回报大 |
| Warmup 跑挂了 | 中 | 中 | run_phase.py yaml-级 retry 1 次；retry 不行就跳过 preload，fallback 到 P0 all_nan_warm_fallback（仍能跑只是 sentinel firing 高） |
| `load_cache_config` 加 yaml_id 字段破老 server | 低 | 高 | server 端 `payload.get("yaml_id")` 缺时 `bundle.yaml_id = None`，B2 路径全跳过（preload 不发生），不影响老行为 |

## 12. Plan-level 决策点（G1 评审重点）

1. **Warmup `--num-trials-per-task` 选 1 还是 2**：默认 2（10 task × 2 trial = 20 warmup ep / yaml ≈ 420 verdict）。1 也行（210 verdict / yaml），数学余量更紧。
2. **`__hit_meta__` always-on vs opt-in**：默认 always-on（< 200 byte / response，开销可忽略）。alt: 加 ctrl `enable_hit_meta_stream` 开关（增加协议复杂度）。建议 always-on。
3. **Warmup 失败 fallback**：retry 1 次；不行就 server-side 不 preload，fallback 到 P0 all_nan_warm_fallback（仍能跑只是 sentinel firing 高）。yaml 级 fail，不阻塞下个 yaml。
4. **WarmupPool 是否持久化磁盘**：默认内存 only，server 重启丢；如需持久化加 `--warmup-cache-dir` flag。建议**不做**，rerun 一次 warmup 才 ~420 verdict 不贵。
5. **`run_phase.py` 与现有 `run_cache_experiments.py` 共存还是替代**：共存 — verdict_factor_judge 专用，其它实验照旧。
6. **B1 / B2 是否一起 G1 评审**：**一起评**（用户 explicit 要求；R7-G1R2 split 已撤销）。一起 G1 → 一起 land → 一起验证。
7. **Warmup dump root path**：`/tmp/openpi_warmup` 默认。如果生产用 `--warmup-dump-root` 覆盖（B2.3 server stage 加 CLI flag — R5-G1R4 修正：归 server stage 拥有，不是 client SDK）。

## 13. 阶段实施

| 阶段 | 内容 | 增量代码 |
|---|---|---|
| **B1.1 server interceptor** | `InferenceInterceptor.infer()` 4 个 return 路径 attach `__hit_meta__`；测试 `test_interceptor_hit_meta.py` 4 case；`test_websocket_response_hit_meta.py` SDK backward-compat | ~80 |
| **B1.2 client app** | `examples/libero/main.py` 新 CLI flag + 接 `__hit_meta__` 写 JSONL；新 module `per_step_log_writer.py`；测试 `test_per_step_writer.py` | ~170 |
| **B2.1 normalizer preload** | `PercentileRollingNormalizer.preload_buffer(values)` 方法；测试 `test_normalizers_preload.py` | ~50 |
| **B2.2 warmup pool** | `src/openpi/cache/warmup_pool.py` + `tests/cache/test_warmup_pool.py` | ~220 |
| **B2.3 server ctrl + bundle** | `CurrentCacheBundle.yaml_id` 字段；`DumpConfig.deferred` 字段 + validator 跳过分支（R3-G1R4）；`load_cache_config` 解 yaml_id + 填 deferred dump.path；3 新 ctrl handler + .resolve() allowlist（R4-G1R4）；`scripts/serve_policy.py` Args 加 `--warmup-dump-root` (R5-G1R4 归 server 拥有) + 启动 mkdir + dynamic-bundle 分支 (line 261) 调 `build_per_connection_components(..., yaml_id=bundle.yaml_id)` (R2-G1R4)；`build_per_connection_components` 签名改 (R2-G1R4) 加 keyword-only `yaml_id` + 内部 WarmupPool preload；测试 `test_warmup_protocol.py` (含 R1-G1R4 双 yaml_id e2e + R4-G1R4 symlink case) + `test_dump_config_deferred.py` (R3-G1R4) + `test_load_cache_config_deferred.py` (R3-G1R4) + `test_serve_policy_yaml_id_propagation.py` (R2-G1R4) | ~280 |
| **B2.4 client SDK** | ctrl helpers `fetch_dump{warmup_yaml_id}` / `preload_normalizer_buffer{eval_yaml_id, buffer}` / `unload_warmup_buffer{eval_yaml_id}` + `load_cache_config` 加 `yaml_id` kwarg（R1-G1R4 严格双 yaml_id 字段名）| ~80 |
| **B2.5 runner shim + sibling yaml** | `exp/verdict_factor_judge/run_phase.py` 直接 subprocess.run main.py + 编排 warmup (`warmup_yaml_id`) → preload (`eval_yaml_id`) → eval → unload (`eval_yaml_id`) ；phase1_spec.py 加 sibling warmup yaml emit（**用 `dump.deferred=True` + 省 path**, R3-G1R4）；测试 `test_run_phase_shim.py` + `test_phase1_spec_warmup_yamls.py` (R3-G1R4 关键验收：CI host 上 `/tmp/openpi_warmup` 不存在也 validate 全过) | ~370 |
| **B2.6 architecture doc** | `docs/architecture/cache_system.md` §5.13 + `docs/README.md` 索引（R1-G1R3 L3 必做）| ~100 |

总 ~1370 行（含测试 + R4 修正后扩量）。L3.

实施次序：B1.1 → B1.2 → B2.1 → B2.2 → B2.3 → B2.4 → B2.5 → B2.6 串行（每步独立单测可验）。全部 land 后 commit + push + 重跑 1 yaml 验 §10.3 端到端。

## 14. 不在本 plan

- N-PRELOAD 用 Phase 0 calibration JSONL（plan §3.4 单独路径）— 与 sibling warmup yaml 是两条互补路径
- composer threshold 调参（如 0.3 → 0.2）— plan §3.4b 范围
- 通用 runner 改造去 batch — 不在 verdict_factor_judge 范围

## Review Log

### G2 Round 1 — Reviewer — APPROVED — 2026-04-27 20:20 CDT

Checklist:
- Consistency with approved plan: PASS. B1 `__hit_meta__` is attached at the CP1 early-return and final return paths; B2 implements strict `warmup_yaml_id` / `eval_yaml_id` ctrl fields, deferred dump resolution, `CurrentCacheBundle.yaml_id` propagation, `WarmupPool`, per-connection normalizer preload, sibling warmup YAML generation, and the dedicated `run_phase.py` orchestration path.
- Test coverage and passing: PASS. Reviewer ran the new G2 target suite (85 tests) and adjacent regressions (159 tests); both passed. Coverage includes hit-meta wire propagation, deferred dump validation, symlink/path traversal rejection, double-yaml-id unload semantics, SDK ctrl helpers, runner shim sequencing, phase1 warmup YAML emission, and existing cache config / websocket server regressions.
- Docs and indexes updated: PASS with advisory. `docs/architecture/cache_system.md` now has §5.13 and `docs/README.md` points to it. The active runbook and logs index still describe the pre-B2 runner/status, but this is an operational documentation lag, not an implementation bug.
- No regressions: PASS. Existing `load_cache_config` callers remain backward-compatible (`yaml_id=None`), old clients can ignore `__hit_meta__`, static `--cache_config` branch does not preload, and the new warmup dump root is opt-in and guarded.

Review entries:
- [Non-blocking] Suggestion update `logs/verdict_factor_judge_phase0_phase1_run_commands.log.md` Phase 1 commands before a human operator uses the runbook — reasoning: lines 351-430 still instruct `run_cache_experiments.py` over 24 eval YAMLs and omit `--warmup-dump-root`, while the implemented B2 path requires `run_phase.py`, sibling `__warmup.yaml` files, per-step output paths, and an explicit server warmup dump root.
- [Non-blocking] Suggestion refresh `logs/README.md` entry for this plan before final archive/polish — reasoning: line 43 still says `Plan` / "待 §4 Code" and uses older wording like `fetch_dump{yaml_id}` and `WarmupPool (key=yaml_id)`, while the implemented protocol now uses strict `fetch_dump{warmup_yaml_id}` and `WarmupPool[eval_yaml_id]`.

G2 approved under the requested lenient standard: no blocking bug found.

### G2 Round 1 — Executor — 2026-04-27 20:45 CDT

- Accepted — **Suggestion #1 (`logs/verdict_factor_judge_phase0_phase1_run_commands.log.md` Phase 1 commands)**: rewrote §2.1 server commands to add `--warmup-dump-root /tmp/openpi_warmup_s{1,2,3}` (one isolated root per server, with rationale comment); rewrote §2.2 client commands to invoke `python -m exp.verdict_factor_judge.run_phase` instead of the generic `run_cache_experiments.py`, dropped now-unused `--episodes-per-run` / `--state-path` / `--resume` flags, added the prerequisite that `phase1_spec.main()` must run first to emit the 24 sibling `__warmup.yaml` files, added `--per-step-log-dir` + `--summary-out` paths matching the dedicated runner's output schema, and added a note on the 1 client/server constraint surviving into B2 (because `_current_bundle` is still a module-level global and `preload_normalizer_buffer` would race the same way).
- Accepted — **Suggestion #2 (`logs/README.md` line 43 entry)**: status flipped from `Plan (G1 APPROVED R5；Post-G1 polish 完成，待 §4 Code)` → `Implemented (G1 APPROVED R5；§4 Code 完成；G2 APPROVED R1，2 非阻塞 doc 建议已 accept)`. Wire-naming corrected throughout the entry: `fetch_dump{warmup_yaml_id}` (was `fetch_dump{yaml_id}`), `WarmupPool[eval_yaml_id]` (was `WarmupPool (key=yaml_id)`), `unload_warmup_buffer{eval_yaml_id}` 含 server 派生 warmup name 防 G1R3 conflation 注释。Added the new symbols that landed during §4 Code (`_fill_deferred_dump_paths`, `_run_episode infer_recorder`, `dump.deferred=true`, double `.resolve()` allowlist, drill-past-DumpingJudge preload), bumped line count `~1250 → ~1370`, replaced "7 测试 file" with "12 测试 file" + their actual coverage (`path traversal + symlink escape + 双 yaml_id e2e + 24 sibling warmup yaml validate / /tmp/openpi_warmup 不存在 case`), and recorded the G2 reviewer's 85 new + 159 regression PASS counts.

Both items are documentation-only; no source code or test changes were needed. The G2 APPROVED verdict stands.
