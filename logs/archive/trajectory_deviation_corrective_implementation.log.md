# Trajectory Deviation 纠偏实验 — 代码级实现计划

> Status: Implemented
> Date: 2026-04-13
> Companion to: [`trajectory_deviation_corrective_experiment.log.md`](trajectory_deviation_corrective_experiment.log.md)
> Scope: 把 experiment plan 中散落的"新建/修改 X 文件"条目展开为真正可以动手写代码的详细步骤 —— 每一处改动都标注**锚点文件 + 行号 + 插入/替换片段 + 数据结构**。

---

## 0. 如何使用本文档

本文是代码级 plan，**不重复实验逻辑**。所有实验动机、判定标准、参数空间在原 plan 中，对应章节通过下面的索引直接跳转。

### 0.1 与原 plan 的索引

| 原 plan 章节 | 本文档对应章节 | 关键改动 |
|-------------|---------------|---------|
| [§4 Step 1 数据收集](trajectory_deviation_corrective_experiment.log.md) | §3 (episode_name 透传)、§4 (main.py flags)、§9 (Step 1b runner) | `episode_name`/`save_trajectory`/`episode_filter` 三组 flag + sim_state 保存 |
| [§4.4 env save/restore 验证](trajectory_deviation_corrective_experiment.log.md) | §11.0 (Phase 0 smoke script) | 独立小脚本验证 get_sim_state/set_init_state |
| [§5 Step 2 Deviate Score](trajectory_deviation_corrective_experiment.log.md) | §10 (compute_deviate_scores) | AlwaysSkipGate + 双阶段 M 连接并行 |
| [§6 Step 3 Spawn 实验](trajectory_deviation_corrective_experiment.log.md) | §11 (run_spawn_experiment) | teleport + prefill_trajectory + 纯 cache rollout |
| [§9.1 总览表 (新建/修改文件)](trajectory_deviation_corrective_experiment.log.md) | §1 文件改动总表 | 19 个条目逐个展开为本文具体章节 |
| [§9.2 episode_name 透传](trajectory_deviation_corrective_experiment.log.md) | §3 | 4 个文件 < 5 行改动 |
| [§9.3 main.py 改动](trajectory_deviation_corrective_experiment.log.md) | §4 | Args 字段、save_trajectory 逻辑、episode_filter 过滤 |
| [§9.4 新建脚本](trajectory_deviation_corrective_experiment.log.md) | §8, §9, §10, §11, §12 | 5 个 exp 脚本的代码骨架 |
| [§9.5 Prefill 模式 (facade 层)](trajectory_deviation_corrective_experiment.log.md) | §6 (storage)、§7 (interceptor) | `enter/exit_prefill_mode` + `prefill_trajectory` API |
| [§11 已结论区](trajectory_deviation_corrective_experiment.log.md) | 全文参考 | 各决定来源 |
| [§13 Runner 架构总览](trajectory_deviation_corrective_experiment.log.md) | §2, §8–§12 | 四 runner 架构落到代码 |
| [§13.7 --resume + retry 硬约束](trajectory_deviation_corrective_experiment.log.md) | §2 (`_run_state_base.py`) | `BaseRunState` 抽象 |

### 0.2 术语同原 plan

见 [§2 术语定义](trajectory_deviation_corrective_experiment.log.md)。本文新增两个代码级记号：

- **锚点**：形如 `file.py:LINE`，表示"**在现有代码的这一行之后插入 / 替换 / 参考**"。
- **签名不变**：函数/方法签名不变，仅内部逻辑改动；调用方零改动。

---

## 1. 文件改动总表（按依赖顺序）

改动按实现阶段分层，上层依赖下层。本文后续章节按此顺序展开。

| # | 文件 | 类型 | 章节 | 关键改动 | 锚点行号 |
|---|------|------|------|---------|---------|
| **A. 共享基础设施层** | | | | | |
| A1 | `exp/_run_state_base.py` | 新建 | §2.1 | `BaseRunState` + `load/save_state` + `_retry_failed_runs` 模板 | — |
| A2 | `exp/_cache_config_rpc.py` | 新建 | §2.2 | WebSocket 控制消息封装：`send_load_cache_config` + `send_prefill_begin/end` | — |
| **B. Client↔Server 透传层** | | | | | |
| B1 | `packages/openpi-client/src/openpi_client/websocket_client_policy.py` | 修改 | §3.1 | `episode_start(...)` 新增 `episode_name` kwarg；新增 `prefill_trajectory(...)` | L56, L72 |
| B2 | `src/openpi/serving/websocket_policy_server.py` | 修改 | §3.2, §5 | 透传 `__episode_name__`；新增 `prefill_begin/prefill_end/prefill_trajectory` 控制分支 | L205–L264 |
| B3 | `src/openpi/collect/collection_policy.py` | 修改 | §3.3 | `on_episode_start` 透传 `episode_name` + raw prompt | L103 |
| B4 | `src/openpi/collect/data_collector.py` | 修改 | §3.4 | 按 client 指定的 `episode_name` 命名 HDF5；episode attrs 加 `prompt` | L40, L70, L76 |
| **C. Libero 实验脚本层** | | | | | |
| C1 | `examples/libero/main.py` | 修改 | §4 | 5 组 Args flag + `_run_episode` buffer + `episode_filter` + `episode_name` 传递 | L31, L83, L139, L461 |
| **D. Cache framework 层** | | | | | |
| D1 | `src/openpi/cache/components/gate.py` | 修改 | §8.1 | 新增 `AlwaysSkipGate` | L48–L74 |
| D2 | `src/openpi/cache/config.py` | 修改 | §8.2 | `_build_gate` 接 `"always_skip"`；validation 放开 | L438, L864 |
| D3 | `src/openpi/cache/cache_storage.py` | 修改 | §6 | `enter/exit_prefill_mode` + `search/fetch_payload` 首行分支 | L48, L61, L71 |
| D4 | `src/openpi/cache/interceptor.py` | 修改 | §7 | `prefill_trajectory(...)` 方法，内部 enter/exit prefill + `self.infer` | L139, L301 |
| **E. 配置 YAML 层** | | | | | |
| E1 | `configs/cache_runs/deviate_exp/inference_clip_w7_d4.yaml` | 新建 | §8.3 | AlwaysSkipGate 配置 × 3 |
| E2 | `configs/cache_runs/deviate_exp/inference_spatial16_w8_d4.yaml` | 新建 | §8.3 | — |
| E3 | `configs/cache_runs/deviate_exp/inference_max_pool_w3_d5.yaml` | 新建 | §8.3 | — |
| E4 | `configs/cache_runs/deviate_exp/cache_{clip_w7_d4,spatial16_w8_d4,max_pool_w3_d5}.yaml` | 新建 | §8.3 | 从 `configs/cache_runs/trajectory/...` 复制 3 份 |
| **F. 实验 Runner 层** | | | | | |
| F1 | `scripts/dump_step1a_failed_inits.py` | 新建 | §9.1 | 读 Step 1a JSON → `torch.save` per-task `.init` 文件 |
| F2 | `exp/run_step1b_gt.py` | 新建 | §9.2 | Thin dispatcher：per-(task, init_idx) resume + `--collect` |
| F3 | `exp/compute_deviate_scores.py` | 新建 | §10 | 双阶段 M 连接并行 replay；Step 2 |
| F4 | `exp/run_spawn_experiment.py` | 新建 | §11 | teleport + prefill_trajectory + 纯 cache rollout；Step 3 |
| F5 | `exp/analyze_deviation_results.py` | 新建 | §12 | 统计/可视化 |
| F6 | `exp/run_cache_experiments.py` | 修改 | §9.0 | 迁移到 `_run_state_base` + 传 `--save-episode-results` |

**改动规模汇总**：新建 13 个文件 + 修改 9 个文件，其中只有 `compute_deviate_scores.py` / `run_spawn_experiment.py` 是"高复杂度"，其余均为 thin/transparent 改动。

---

## 2. 共享基础设施（Layer A）

### 2.1 `exp/_run_state_base.py` — BaseRunState 抽象

**职责**：所有 runner 的 state JSON 持久化骨架 + `--resume` + `_retry_failed_runs` 模板。

**参考实现**：从 `exp/run_cache_experiments.py:56-99` (RunState) 和 `:583-736` (`_retry_failed_runs`) 抽象出来。

**对外 API**：

```python
# exp/_run_state_base.py
from __future__ import annotations
import json, time, shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class UnitState:
    """Single unit of work. unit_key is stringified tuple."""
    unit_key: str
    status: str = "pending"        # pending | running | done | failed
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    retry_count: int = 0
    # per-unit free-form payload (successes/total, action hashes, etc.)
    result: dict[str, Any] = field(default_factory=dict)


class BaseRunState(ABC):
    """Skeleton for runner state files.

    Subclass responsibilities:
      - define unit_key(tuple) -> str
      - enumerate initial units via build_units()
      - implement execute_unit(unit) -> result | raise
    Base class handles JSON load/save, per-unit resume, retry orchestration.
    """

    def __init__(self, state_path: Path, *, max_retries: int = 2):
        self.state_path = Path(state_path)
        self.max_retries = max_retries
        self.units: dict[str, UnitState] = {}

    # ---------- persistence ----------
    def load(self) -> None:
        if self.state_path.exists():
            data = json.loads(self.state_path.read_text())
            self.units = {k: UnitState(**v) for k, v in data.items()}

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps({k: asdict(u) for k, u in self.units.items()}, indent=2)
        )

    # ---------- query ----------
    def is_done(self, key: str) -> bool:
        return key in self.units and self.units[key].status == "done"

    def pending_units(self) -> list[UnitState]:
        return [u for u in self.units.values() if u.status != "done"]

    def failed_units(self) -> list[UnitState]:
        return [u for u in self.units.values() if u.status == "failed"]

    # ---------- subclass hooks ----------
    @abstractmethod
    def build_units(self) -> list[UnitState]:
        """Enumerate the initial unit list (called when state file is empty)."""

    @abstractmethod
    def execute_unit(self, unit: UnitState) -> dict:
        """Run one unit. Return result dict on success; raise on failure."""

    # ---------- main driver ----------
    def run(self, *, resume: bool = False, unit_filter: Optional[Callable[[UnitState], bool]] = None):
        if resume:
            self.load()
        if not self.units:
            for u in self.build_units():
                self.units[u.unit_key] = u
        self.save()

        queue = [u for u in self.units.values() if u.status != "done"]
        if unit_filter is not None:
            queue = [u for u in queue if unit_filter(u)]

        # --- Primary pass ---
        for u in queue:
            self._execute_one(u)

        # --- Retry pass ---
        for attempt in range(1, self.max_retries + 1):
            still_failed = self.failed_units()
            if not still_failed:
                break
            print(f"[retry {attempt}/{self.max_retries}] {len(still_failed)} failed units")
            for u in still_failed:
                u.retry_count = attempt
                u.status = "pending"
                self.save()
                self._execute_one(u)

    def _execute_one(self, u: UnitState) -> None:
        u.status = "running"
        u.start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save()
        try:
            result = self.execute_unit(u)
            u.result = result
            u.status = "done"
        except Exception as e:  # keep broad: runner-specific handlers should raise structured errors
            u.status = "failed"
            u.result = {"error": str(e)}
        u.end_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save()
```

**调用方模式**（所有 runner 通用）：

```python
class MyRunner(BaseRunState):
    def __init__(self, ...):
        super().__init__(state_path=Path(...))
        ...

    def build_units(self) -> list[UnitState]:
        return [UnitState(unit_key=f"{a}:{b}") for a, b in self.parameter_grid]

    def execute_unit(self, u: UnitState) -> dict:
        parts = u.unit_key.split(":")  # or store structured fields in u.result from the start
        ...
        return {"success": True, "successes": s, "total": t}

if __name__ == "__main__":
    args = parse_args()
    MyRunner(...).run(resume=args.resume)
```

**设计决定**：

- **unit_key 用字符串**：JSON 友好，避免 tuple 序列化/反序列化歧义。子类决定 key format（如 `"cfg:ep007:s42:n5:k0"`）。
- **Retry 策略就地重跑**：不 copy YAML 到 retry/ 目录（与 `run_cache_experiments.py:583-736` 不同）—— 因为 Step 2/3 不以 YAML 为 unit 粒度，retry/ 目录反而成累赘。失败单 unit 直接在原 state 里翻 `status=pending` 重跑即可。
- **结构化错误**：execute_unit 抛异常会吞掉 trace，runner 子类应自行 log 后再 raise 带上下文的异常。

### 2.2 `exp/_cache_config_rpc.py` — WebSocket 控制消息封装

从 `exp/run_cache_experiments.py:79-99` 抽出 `_send_cache_config`，统一到此模块并新增 Step 3 所需的 prefill 控制函数。

```python
# exp/_cache_config_rpc.py
from __future__ import annotations
import asyncio
import logging
from pathlib import Path

import msgpack
import websockets

logger = logging.getLogger(__name__)


async def _send_ctrl(server_url: str, msg: dict, *, ack: str) -> dict:
    """Internal: send one __ctrl__ message and validate ack."""
    async with websockets.connect(server_url) as ws:
        _metadata = await ws.recv()  # server sends metadata on connect
        await ws.send(msgpack.packb(msg))
        resp = msgpack.unpackb(await ws.recv())
        if resp.get("__ack__") != ack:
            raise RuntimeError(f"Control message failed (expected {ack}): {resp}")
        return resp


def send_load_cache_config(server_url: str, yaml_path: str | Path) -> int:
    """Switch server cache bundle to the given YAML. Returns new bundle version."""
    yaml_content = Path(yaml_path).read_text()
    msg = {"__ctrl__": "load_cache_config", "yaml_content": yaml_content}
    resp = asyncio.run(_send_ctrl(server_url, msg, ack="load_cache_config"))
    logger.info("Switched server to bundle v%s: %s", resp.get("version"), yaml_path)
    return int(resp.get("version", -1))


def send_prefill_begin(server_url: str, payload_b64: str) -> None:
    """Enter prefill mode with the given payload (base64 msgpack-packed CachePayload)."""
    asyncio.run(_send_ctrl(
        server_url,
        {"__ctrl__": "prefill_begin", "payload_b64": payload_b64},
        ack="prefill_begin",
    ))


def send_prefill_end(server_url: str) -> None:
    """Leave prefill mode."""
    asyncio.run(_send_ctrl(
        server_url, {"__ctrl__": "prefill_end"}, ack="prefill_end",
    ))
```

**注意**：

- `send_load_cache_config` 等价替换原 `_send_cache_config`，return value 从 `None` 变成 `int`（bundle version），方便 runner 做假死检测。
- **Prefill 控制走 per-connection** 而非全局：见 §5 的讨论。因此 `send_prefill_begin/end` 不能在独立 async 连接里发 —— 必须由**该 connection 的 policy client** 发才能命中对应 bundle。所以实际 runner 使用时，Step 3 runner 不直接调 `send_prefill_begin`，而是用 client 的 `prefill_trajectory`（§3.1）。此处留着是为了将来 administrative 场景（如全局 debug）。

---

## 3. Client↔Server 透传层（Layer B）

### 3.1 `packages/openpi-client/src/openpi_client/websocket_client_policy.py`

锚点 `L56-70`（`episode_start`）、`L47` (`infer`)。

**改动 1：`episode_start` 新增 `episode_name` kwarg**

```python
# 替换 L56-70
def episode_start(
    self,
    experiment: str,
    task: str = "",
    episode_id: int = -1,
    episode_name: str = "",     # NEW
) -> Dict:
    self._ws.send(
        self._packer.pack({
            "__ctrl__": "episode_start",
            "__experiment__": experiment,
            "__task__": task,
            "__episode_id__": episode_id,
            "__episode_name__": episode_name,   # NEW (empty = legacy behavior)
        })
    )
    response = self._ws.recv()
    return msgpack_numpy.unpackb(response)
```

**改动 2：新增 `prefill_trajectory` 方法**

```python
# 追加在 episode_end 之后
def prefill_trajectory(
    self,
    observations: list[Dict],
    actions: list,               # list[np.ndarray], one per obs
    *,
    record: bool = False,
    on_miss: str = "error",
) -> Dict:
    """Drive the cache framework through (obs, action) pairs as if they were
    real inference steps. See src/openpi/cache/interceptor.py::prefill_trajectory
    for semantics.
    """
    self._ws.send(
        self._packer.pack({
            "__ctrl__": "prefill_trajectory",
            "observations": observations,
            "actions": actions,
            "record": record,
            "on_miss": on_miss,
        })
    )
    response = self._ws.recv()
    return msgpack_numpy.unpackb(response)
```

### 3.2 `src/openpi/serving/websocket_policy_server.py`

锚点 `L205-264`（`__ctrl__` dispatcher）、`L207-214`（episode_start 分支），`CurrentCacheBundle` at `L68-79`。

**改动 1：`episode_start` 分支透传 `episode_name`**

```python
# 在 L207-214 episode_start 分支内，将 __episode_name__ 下传到 policy
elif ctrl == "episode_start":
    experiment = obs.get("__experiment__", "")
    task = obs.get("__task__", "")
    episode_id = obs.get("__episode_id__", -1)
    episode_name = obs.get("__episode_name__", "")   # NEW
    if hasattr(policy, "on_episode_start"):
        policy.on_episode_start(
            experiment=experiment,
            task=task,
            episode_id=episode_id,
            episode_name=episode_name,     # NEW (policy wrappers透传到 collection_policy)
        )
    await websocket.send(packer.pack({"__ack__": "episode_start"}))
    continue
```

**改动 2：新增三条 prefill 控制消息分支（在 L218 episode_end 分支之后、L219 load_cache_config 之前）**

```python
elif ctrl == "prefill_begin":
    # payload_b64 is base64(msgpack(CachePayload)) — lets clients construct payloads
    # without exposing CachePayload type across the wire.
    import base64, msgpack
    raw = base64.b64decode(obs["payload_b64"])
    from openpi.cache.storage_types import CachePayload
    payload = CachePayload(**msgpack.unpackb(raw))
    # NOTE: shared_storage is shared across connections!  Prefer per-connection
    # mode. See §5 for resolution: prefill_begin/end are per-connection via
    # the per-connection bundle's storage proxy.
    current_bundle.shared_storage.enter_prefill_mode(payload)
    await websocket.send(packer.pack({"__ack__": "prefill_begin"}))
    continue

elif ctrl == "prefill_end":
    current_bundle.shared_storage.exit_prefill_mode()
    await websocket.send(packer.pack({"__ack__": "prefill_end"}))
    continue

elif ctrl == "prefill_trajectory":
    # Preferred path: whole-trajectory prefill handled inside the per-connection
    # interceptor so storage mode toggles are local to this connection.
    observations = obs["observations"]
    actions = obs["actions"]
    record = obs.get("record", False)
    on_miss = obs.get("on_miss", "error")
    # policy is the per-connection wrapped InferenceInterceptor
    policy.prefill_trajectory(observations, actions, record=record, on_miss=on_miss)
    await websocket.send(packer.pack({"__ack__": "prefill_trajectory"}))
    continue
```

**关键约束（§5 会重新审视）**：`shared_storage` 对 `CurrentCacheBundle` 内多个并发连接共享，直接在它上面 `enter_prefill_mode` 会让**所有连接**同时进入 prefill 模式 —— 错误。见 §5 解决方案。

### 3.3 `src/openpi/collect/collection_policy.py`

锚点 `L103-105`。

```python
# 替换 on_episode_start
def on_episode_start(
    self,
    experiment: str,
    task: str,
    episode_id: int,
    episode_name: str = "",          # NEW
) -> None:
    self._collector.on_episode_start(experiment, task, episode_id, episode_name=episode_name)
    self._collecting = True
```

**额外**：`CollectionPolicy.infer()` 首次调用时从 `obs["prompt"]` 取 raw string 存给 collector 作为 episode attr（§9.5 的 prompt 收集）。

```python
# 在 infer() 头部（L56 附近）加一次性抓取：
if self._collecting and not self._prompt_captured:
    raw_prompt = obs.get("prompt", "")
    self._collector.set_episode_attr("prompt", str(raw_prompt))
    self._prompt_captured = True
```

**新增 instance 字段** `self._prompt_captured: bool = False`，`on_episode_start` 同时置回 False。

### 3.4 `src/openpi/collect/data_collector.py`

锚点：`on_episode_start` at `L40-46`，HDF5 filename at `L70-71`，attrs at `L76-81`。

**改动 1：`on_episode_start` 接收 `episode_name`**

```python
def on_episode_start(
    self,
    experiment: str,
    task: str,
    episode_id: int,
    *,
    episode_name: str = "",
) -> None:
    with self._lock:
        self._buffer = []
        self._experiment = experiment
        self._task = task
        self._episode_id = episode_id
        self._episode_name = episode_name     # NEW
        self._episode_attrs: dict[str, Any] = {}   # NEW (for set_episode_attr)
    logger.info(...)
```

**改动 2：`set_episode_attr()` 新方法**

```python
def set_episode_attr(self, key: str, value) -> None:
    """Record an episode-scoped attribute. Written to HDF5 attrs at flush."""
    with self._lock:
        self._episode_attrs[key] = value
```

**改动 3：filename 构造按 `episode_name`**

```python
# 替换 L70-71
if self._episode_name:
    # episode_name may contain subdirs, e.g. "task_3/episode_2"
    path = out_dir / f"{self._episode_name}.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
else:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = out_dir / f"episode_{self._episode_id:04d}_{ts}.h5"
```

**改动 4：attrs 写入加 prompt 等自定义**

```python
# 在 L76-81 标准 attrs 之后
for k, v in self._episode_attrs.items():
    f.attrs[k] = v
```

---

## 4. Libero 实验脚本层（Layer C）

### 4.1 `examples/libero/main.py`

锚点：Args at `L31-66`，`_run_episode` at `L83-94`，env.step loop at `L105-152` (infer at `L139`, step at `L146`)，`_load_init_states` at `L461-477`，episode_id logic at `L182-183` / `L291`，`client.episode_start` call at `L179-183` / `L311-315`。

**改动 1：Args 新增 5 组 flag**

```python
@dataclasses.dataclass
class Args:
    # ... existing fields ...
    # --- NEW: trajectory capture for deviation experiment ---
    save_trajectory: bool = False
    save_trajectory_dir: str = "data/deviation_experiment/gt_trajectories"
    # --- NEW: per-episode success JSON (needed by Step 1a) ---
    save_episode_results: bool = False
    episode_results_path: str = ""    # default: {task_suite}_episode_results.json
    # --- NEW: filter which (task_id, init_idx) to run (Step 1b) ---
    episode_filter: str = ""          # JSON path, [{task_id, init_state_idx}, ...]
```

**改动 2：`_run_episode` 内加 buffer 逻辑**

`_run_episode` 内部需要保存每步的 `{sim_state, env_timestep, env_cur_time, images, robot_state, action}`。
锚点在 env.step loop 内（`L146` 附近）：

```python
# 在 L139 (action = client.infer(...)) 之后、L146 (env.step(action[...])) 之前
if args.save_trajectory:
    step_record = {
        "sim_state": env.get_sim_state().copy(),         # flattened numpy, HDF5-safe
        "env_timestep": int(env.timestep),
        "env_cur_time": float(env.cur_time),
        "agentview_image": obs["agentview_image"].copy(),
        "eye_in_hand_image": obs["robot0_eye_in_hand_image"].copy(),
        "robot_state": robot_state.copy(),               # already 8D (eef_pos+quat+gripper)
        "action": action[step_in_chunk].copy(),          # selected action from chunk
    }
    traj_buffer.append(step_record)
```

`traj_buffer: list[dict] = []` initialize在 `_run_episode` 开头，episode 结束处（`L152` 后）flush 到 HDF5：

```python
if args.save_trajectory and traj_buffer:
    out_path = Path(args.save_trajectory_dir) / f"task_{task_id}" / f"episode_{init_state_idx}.h5"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["task_name"] = task_description
        f.attrs["task_id"] = task_id
        f.attrs["init_state_idx"] = init_state_idx
        f.attrs["episode_id"] = global_episode_id
        f.attrs["seed"] = args.seed
        f.attrs["num_steps"] = len(traj_buffer)
        f.attrs["success"] = success
        for i, step in enumerate(traj_buffer):
            g = f.create_group(f"step_{i:04d}")
            for k, v in step.items():
                g.create_dataset(k, data=v)
```

**注意**：`_run_episode` 目前 signature 不带 `task_id / init_state_idx`，需要改 signature（或从 `task_description` 反查；但传参更干净）。改动 serial path (`L178` 附近) + concurrent path (`L290` 附近) 的调用点。

**改动 3：`episode_filter` 过滤**

```python
# 在 task/episode loop 开头（serial ~L175, concurrent ~L280）
# 读一次过滤列表
filter_set: set[tuple[int, int]] | None = None
if args.episode_filter:
    pairs = json.loads(Path(args.episode_filter).read_text())
    filter_set = {(p["task_id"], p["init_state_idx"]) for p in pairs}

# loop 内
if filter_set is not None and (task_id, init_state_idx) not in filter_set:
    continue
```

**改动 4：`episode_start` 传 `episode_name`**

```python
# serial L179 / concurrent L311
client.episode_start(
    experiment=args.task_suite_name,
    task=str(task_description),
    episode_id=global_episode_id,
    episode_name=f"task_{task_id}/episode_{init_state_idx}"   # NEW
        if args.save_trajectory else "",   # 非 GT 收集场景保持旧命名
)
```

**改动 5：`save_episode_results` 聚合**

在 main 函数最后（所有 episode 跑完），如果 `args.save_episode_results`：

```python
if args.save_episode_results:
    out = args.episode_results_path or f"{args.task_suite_name}_episode_results.json"
    result = [
        {"task_id": tid, "init_state_idx": idx, "episode_id": eid,
         "seed": args.seed, "success": succ}
        for (tid, idx, eid, succ) in per_episode_log
    ]
    Path(out).write_text(json.dumps(result, indent=2))
```

`per_episode_log: list[tuple]` append 在每个 episode 结束处。

---

## 5. Per-connection Prefill 的正确设计（§3.2 接续）

**问题**：`CurrentCacheBundle.shared_storage` 对所有并发连接共享，直接在它上面 enter/exit prefill 模式会串扰。

**解法（§9.5 原定位已暗示）**：prefill 模式是**某个 connection 的状态**，应该通过**该 connection 的 interceptor** 操作它的 storage 句柄。在 server 的 concurrent 模式下：

- `shared_storage` 仅持有 backend 引用与维度校验 —— 保持全局单例。
- 每个 connection 拿到的是一个 `CacheStorage` **facade instance（持有 backend 引用 + 自己的 prefill 状态）**。

**实现决定**（改 `build_per_connection_components` at `src/openpi/cache/config.py:713-761`）：

```python
# 现状：每连接 storage 直接用 shared_storage（共享 facade 实例）
# 改为：每连接新建一个 CacheStorage facade，共享的是其中的 backend
def build_per_connection_components(config, shared_storage, *, quiet=False):
    ...
    # NEW: wrap backend in a fresh facade for this connection
    from openpi.cache.cache_storage import CacheStorage
    per_conn_storage = CacheStorage(shared_storage._backend)
    ...
    return {
        "storage": per_conn_storage,   # was: shared_storage
        ...
    }
```

**代价评估**：`CacheStorage.__init__` 只缓存 `backend.vector_dims`（cheap dict），per-connection 拷贝无内存压力。Backend 单例语义完全保留。对 insert 链路也无影响（insert 本就只加锁写 backend，不持有 per-connection 状态）。

**结果**：`prefill_begin/end` 控制消息路由到 **policy.prefill_trajectory** 时，操作的是**当前连接的 facade**，其他连接不受影响。

**更新 §3.2**：`prefill_begin/end` 分支仍有用（供 administrative/debug 单次 entry），但 Step 3 runner 应使用 `prefill_trajectory`（整段轨迹原子灌入，语义更清晰、更难出错）。

---

## 6. CacheStorage Prefill 模式（Layer D3）

锚点：`src/openpi/cache/cache_storage.py`
- Constructor at `L48-55`
- `search()` at `L61`
- `fetch_payload()` at `L71`
- `search_and_fetch()` at `L75`
- `_check_query_dims()` at `L136`, `_check_filters()` at `L172`

**改动（全体）**：

```python
class CacheStorage:
    def __init__(self, backend: VectorStoreBackend, metadata_db=None) -> None:
        self._backend = backend
        self._metadata_db = metadata_db
        self._dims: dict[str, int] = backend.vector_dims
        # NEW: prefill mode state
        self._prefill_mode: bool = False
        self._prefill_payload: "CachePayload | None" = None

    # NEW methods
    def enter_prefill_mode(self, payload: "CachePayload") -> None:
        """Enter prefill mode. Until exit_prefill_mode():
          - search(spec) returns exactly one synthetic SearchResultLite (score=1.0,
            id='__prefill__'), guaranteeing FULL_HIT under any reasonable judge.
          - fetch_payload(id) returns the stored payload, ignoring id.
          - insert()/batch_insert()/delete() still forward to backend (caller's
            responsibility to avoid mutation during prefill; not enforced).
        Idempotent: calling while already in prefill replaces the payload.
        """
        self._prefill_mode = True
        self._prefill_payload = payload

    def exit_prefill_mode(self) -> None:
        """Resume normal behavior. No-op if not currently in prefill."""
        self._prefill_mode = False
        self._prefill_payload = None

    def search(self, spec):
        self._check_query_dims(spec)
        self._check_filters(spec)
        # NEW: prefill synthetic hit
        if self._prefill_mode:
            from openpi.cache.storage_types import SearchResultLite
            return [SearchResultLite(
                id="__prefill__",
                score=1.0,
                checkpoint_id=spec.checkpoint_id,
            )]
        return self._backend.search(spec)

    def fetch_payload(self, id: str):
        # NEW: prefill returns stored payload ignoring id
        if self._prefill_mode:
            return self._prefill_payload
        return self._backend.fetch_payload(id)

    # search_and_fetch needs no change: it internally calls self.search + self.fetch_payload
    # insert / batch_insert / delete need no change: forward to backend as-is
```

**测试要求**（§11.0 smoke test 的一部分）：
1. 正常模式：search 返回 backend 结果，fetch_payload 从 backend 取。
2. Prefill 模式下 search 返回单条 score=1.0 合成 result。
3. Prefill 模式下 fetch_payload 忽略 id 返回 stored payload。
4. exit_prefill_mode 后完全恢复正常。

---

## 7. Interceptor Prefill API（Layer D4）

锚点：`src/openpi/cache/interceptor.py`
- Class at `L112`
- Constructor at `L139-212` (access to `self._orchestrator` at `L207`, `self._cache_storage` via `self._orchestrator._storage`)
- `infer()` at `L301`
- `on_episode_start` at `L270-276`

**新增方法**（追加在 `on_episode_start` 之后，`infer()` 之前）：

```python
def prefill_trajectory(
    self,
    observations: list[dict],
    actions: "list[np.ndarray] | None" = None,
    *,
    record: bool = False,
    on_miss: str = "error",
) -> None:
    """Drive cache framework along (obs, action) as if those steps really happened.

    After return: all stateful components (key_builder vision buffer, strategy
    trajectory buffer, orchestrator step_counter, gate/judge hooks) are
    consistent with that trajectory — ready for the "real" next inference.

    First-version supports: actions provided, record=False, on_miss='error'.
    Other combinations raise NotImplementedError; see §9.5 of experiment plan.
    """
    if actions is None:
        raise NotImplementedError(
            "actions=None (cache self-query mode). Future: run real search per "
            "step, use cache's own returned actions as history. Needed for "
            "'what would cache remember if running pure-cache' analyses."
        )
    if record:
        raise NotImplementedError(
            "record=True. Future: capture synthetic prefill steps into HDF5 "
            "tagged as 'prefill' for audit. Requires CollectionPolicy to "
            "distinguish prefill from real inference steps."
        )
    if on_miss != "error":
        raise NotImplementedError(
            f"on_miss={on_miss!r}. First version uses facade synthetic hit; "
            "MISS cannot happen here. Future: 'warn'/'fallback_infer' with "
            "actions=None mode."
        )
    if len(observations) != len(actions):
        raise ValueError(
            f"observations ({len(observations)}) and actions ({len(actions)}) "
            "must have equal length"
        )
    # access cache_storage via orchestrator (private but stable reference)
    cache_storage = self._orchestrator._storage
    for obs, action in zip(observations, actions):
        payload = self._build_prefill_payload(action)
        cache_storage.enter_prefill_mode(payload)
        try:
            self.infer(obs)   # full pipeline: key_builder.collect+build, strategy.search
                              # (returns synthetic hit), judge, fetch_payload,
                              # broadcast_action — all side effects happen
        finally:
            cache_storage.exit_prefill_mode()

@staticmethod
def _build_prefill_payload(action) -> "CachePayload":
    """Construct a minimal CachePayload carrying only the ground-truth action.
    task_key/intermediates/denoising_num_steps are None — framework's downstream
    code won't consult them in the prefill path (it discards the infer output)."""
    from openpi.cache.storage_types import CachePayload
    import torch, numpy as np
    if isinstance(action, np.ndarray):
        action_t = torch.from_numpy(action)
    else:
        action_t = action
    return CachePayload(
        action_chunk=action_t,
        task_key=None,
        intermediates=None,
        denoising_num_steps=None,
    )
```

**侵入点检查**：`InferenceInterceptor.infer` (`L301`) 的主路径（CP1 check `L356-384`、CP3 check `L442-472`）在 prefill 模式下走的是"cache 返回 FULL_HIT，取 payload.action_chunk 作为输出"——这条路径不会触发真正的模型 forward。**Stage1 forward 仍被执行**（因为 CP1 check 在 stage1 之后），这是 §9.5 表格里认可的代价："stage2/3 浪费可接受（D-1 次）"。

**若要跳过 stage1 forward**（未来优化）：Interceptor 可在 prefill 模式下直接 short-circuit 到 `broadcast_action(action) + update step counter` —— 但会绕过 key_builder 的 vision history 更新，与设计矛盾。**首版不做**。

---

## 8. Gate & Config（Layer D1, D2, E）

### 8.1 `src/openpi/cache/components/gate.py` — AlwaysSkipGate

锚点：`AlwaysSearchGate` 类定义 at `L48-74`。

**追加**（在 `AlwaysSearchGate` 之后）：

```python
class AlwaysSkipGate:
    """Always skip search. Orchestrator treats this as a gate miss path:
      - records query_keys to strategy history (trajectory gap-free)
      - returns HitType.MISS, triggering the interceptor's inference path
      - broadcast_action feeds real inference action back into all components
    Net effect: cache framework is 'transparent' for this checkpoint — every
    step forces inference while trajectory buffers remain consistent.

    Use case: Step 2 of deviation experiment (background L2 sampling) where
    we need M independent inferences along a GT obs sequence with trajectory
    history semantics preserved (see logs/trajectory_deviation_corrective_experiment.log.md §13.4).
    """

    def __call__(
        self,
        checkpoint_id: CheckpointID,
        cached_data: dict[str, torch.Tensor],
    ) -> bool:
        return False

    def on_episode_start(self) -> None:
        """No-op. Signature matches GateFunction protocol."""

    def record_action(self, action_chunk: torch.Tensor) -> None:
        """No-op. Signature matches GateFunction protocol."""
```

### 8.2 `src/openpi/cache/config.py` — _build_gate + validation

锚点：`_build_gate` at `L864-871`，validation at `L438-439`。

**改动 1：_build_gate 增加 always_skip 分支**

```python
def _build_gate(cfg: GateConfig):
    if cfg.type == "always_search":
        from openpi.cache.components.gate import AlwaysSearchGate
        return AlwaysSearchGate()
    if cfg.type == "always_skip":
        from openpi.cache.components.gate import AlwaysSkipGate
        return AlwaysSkipGate()
    raise ConfigValidationError(
        f"Unknown gate.type '{cfg.type}'. Valid: ['always_search', 'always_skip']"
    )
```

**改动 2：validation 白名单放开**

```python
# L438-439 原：
# if cp_config.gate.type not in ("always_search",):
#     errors.append(f"{prefix}.gate.type '{cp_config.gate.type}' is unknown. Valid: ['always_search']")
# 改为：
_VALID_GATE_TYPES = ("always_search", "always_skip")
if cp_config.gate.type not in _VALID_GATE_TYPES:
    errors.append(f"{prefix}.gate.type '{cp_config.gate.type}' is unknown. Valid: {list(_VALID_GATE_TYPES)}")
```

### 8.3 YAML 配置文件（Layer E）

**目录**：`configs/cache_runs/deviate_exp/`

**源 YAML 映射**（依据 `docs/trajectory_analysis.md` 的 key_builder 类型编码：`a=mean_pool`、`b1=spatial16`、`b2=spatial64`、`c=max_pool`、`d=clip`）：

| 目标 | 源 YAML（`configs/cache_runs/trajectory/`） | key_builder |
|------|--------------------------------------------|-------------|
| `cache_clip_w7_d4.yaml` | `traj_d4_029_d_rrf_w7.yaml` | clip (`d`) |
| `cache_spatial16_w8_d4.yaml` | `traj_d4_019_b1_rrf_w8.yaml` | spatial16 (`b1`) |
| `cache_max_pool_w3_d5.yaml` | `traj_d5_042_c_rrf_w3.yaml` | max_pool (`c`) |

**6 份 YAML**：
- `cache_clip_w7_d4.yaml`、`cache_spatial16_w8_d4.yaml`、`cache_max_pool_w3_d5.yaml`：从上表对应源文件直接复制；`gate.type: always_search` 不变。
- `inference_clip_w7_d4.yaml`、`inference_spatial16_w8_d4.yaml`、`inference_max_pool_w3_d5.yaml`：从对应 `cache_*.yaml` 复制，**唯一改动**：

```yaml
# inference_*.yaml
checkpoints:
  cp1:
    enabled: true
    gate:
      type: always_skip          # 唯一差异
    # judge / search_strategy / key_builder / weights / D / artifact 均与 cache_*.yaml 一致
    ...
```

**不变项（必须严格一致）**：`key_builder.type`、`weights`、`D` (trajectory depth)、`artifact_path`。这些字段一致保证 inference 跑出来的 trajectory history 语义和 cache 配置一致，Step 3 spawn 时 prefill 灌的是同源 history。

---

## 9. Step 1 Runner（Layer F1, F2, F6）

### 9.0 `exp/run_cache_experiments.py`（Step 1a）改动

锚点：整个文件。

**改动 1**：迁移到 `_run_state_base.py`（可选，若成本大也可保留现有 `RunState`）。

**改动 2 (必须)**：调 `main.py` 时透传 `--save-episode-results --episode-results-path <path>`，并在 runner 内部将 per-episode JSON 聚合到 `cache_eval_results.json`。

在 `_execute_tasks` (`L110-206`) 的 main_args 构造处加：

```python
main_args += [
    "--save-episode-results",
    "--episode-results-path", str(log_path.with_suffix(".episode_results.json")),
]
```

Main loop 最后 merge 所有 per-run `.episode_results.json` 到 `cache_eval_results.json`。

### 9.1 `scripts/dump_step1a_failed_inits.py`

**职责**：读 `cache_eval_results.json` → per-task 筛失败 `(task_id, init_idx)` → `torch.save` 成 `.init` 文件。

```python
# scripts/dump_step1a_failed_inits.py
"""Dump per-task init-state subset files containing only the init indices
where Step 1a pure-cache evaluation failed.

Output layout (consumed by main.py --init-states-dir):
  {out_dir}/{task.name}.init   # torch-saved tensor of shape (K, 92)

Usage:
  python scripts/dump_step1a_failed_inits.py \
      --step1a-results data/deviation_experiment/cache_eval_results.json \
      --task-suite libero_spatial \
      --out-dir data/step1b_inits/libero_spatial
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step1a-results", required=True)
    ap.add_argument("--task-suite", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    results = json.loads(Path(args.step1a_results).read_text())
    failed_by_task: dict[int, list[int]] = defaultdict(list)
    for r in results:
        if not r["success"]:
            failed_by_task[r["task_id"]].append(r["init_state_idx"])

    # Load the original benchmark's init states to slice from
    from libero.libero.benchmark import get_benchmark_dict
    suite = get_benchmark_dict()[args.task_suite]()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for task_id, bad_indices in sorted(failed_by_task.items()):
        task = suite.get_task(task_id)
        full_inits = suite.get_task_init_states(task_id)   # shape (50, 92)
        subset = full_inits[sorted(bad_indices)]
        torch.save(subset, out / f"{task.name}.init")
        print(f"task {task_id} ({task.name}): {len(bad_indices)} failed -> "
              f"{out / (task.name + '.init')}")

if __name__ == "__main__":
    main()
```

**注意**：`main.py --init-states-dir` 通过 `task.name` 索引（`L461-477`），因此 `.init` 文件名必须是 `{task.name}.init`。加载时 `main.py` 自动按顺序 map，故 subset 里的 **索引号会被 main.py 重编号为 0..K-1**。这意味着 Step 1b 里跑出来的 `init_state_idx` ≠ Step 1a 里的 `init_state_idx`。

**解决**：Step 1b runner 必须维护一个 `{task_id: [orig_idx_0, orig_idx_1, ...]}` 映射表，写到 HDF5 episode attrs 的 `orig_init_state_idx` 字段。Step 2/3 读 HDF5 时用此字段还原。

### 9.2 `exp/run_step1b_gt.py`

**职责**：thin dispatcher。每个 task 一次 subprocess 调 `main.py`，跑该 task 下所有失败 init。per-(task_id, init_idx) 粒度 resume（从 `main.py` 产出的 `.episode_results.json` 读回）。

**Unit key**：`f"{task_id}:{orig_init_idx}"`。

**伪码骨架**：

```python
# exp/run_step1b_gt.py
from exp._run_state_base import BaseRunState, UnitState
from exp._cache_config_rpc import send_load_cache_config

class Step1bRunner(BaseRunState):
    def __init__(self, *, inits_dir, results_map_path, task_suite, host, port,
                 inference_yaml, ...):
        super().__init__(state_path=Path(inits_dir).parent / "step1b_state.json")
        self.inits_dir = Path(inits_dir)
        self.results_map = json.loads(Path(results_map_path).read_text())
        # maps task_id -> list of orig_init_idx in the order main.py will see them
        self.task_suite = task_suite
        self.server_url = f"ws://{host}:{port}"
        self.inference_yaml = inference_yaml
        ...

    def build_units(self) -> list[UnitState]:
        units = []
        for task_id, orig_indices in self.results_map.items():
            for orig_idx in orig_indices:
                units.append(UnitState(unit_key=f"{task_id}:{orig_idx}"))
        return units

    def execute_unit(self, u: UnitState) -> dict:
        task_id, orig_idx = map(int, u.unit_key.split(":"))
        # re-map orig_idx to position in subset file
        pos = self.results_map[str(task_id)].index(orig_idx)
        # call main.py for this single (task_id, position)
        cmd = ["conda", "run", "--no-capture-output", "-n", "libero_sim", "python",
               "examples/libero/main.py",
               "--host", host, "--port", str(port),
               "--task-suite-name", self.task_suite,
               "--task-ids", str(task_id),
               "--num-trials-per-task", str(pos + 1),   # main.py runs 0..N-1
               "--init-states-dir", str(self.inits_dir),
               "--save-trajectory",
               "--save-trajectory-dir", str(self.out_dir),
               "--save-episode-results",
               "--episode-results-path", ...,
               # NOTE: main.py takes episode_idx[0..pos], so we need --episode-filter
               #       to run only position=pos (避免重复跑 pos<this 的)
               "--episode-filter", temp_filter_json,
               "--seed", str(self.seed)]
        proc = subprocess.run(cmd, ..., timeout=1200)
        # parse episode_results.json for this unit
        res = parse_episode_result(...)
        if not res["success"]:
            # inference 也失败 — 仍记 done 但标记 skip（下游 Step 2 会跳过）
            return {"success": False, "inference_failed": True}
        return {"success": True, "hdf5_path": f"task_{task_id}/episode_{pos}.h5"}

def main():
    args = parse_args()
    send_load_cache_config(server_url, args.inference_yaml)  # 切到纯 inference 配置
    runner = Step1bRunner(...)
    runner.run(resume=args.resume)

if __name__ == "__main__":
    main()
```

**"纯 inference"怎么实现**：Step 1b 要跑纯 inference（不用 cache）。两种选择：
- **方案 a**：让 server load 一个 `no_cache.yaml`（`checkpoints.cp1.enabled: false`）；
- **方案 b**：load 一个 `inference_*.yaml`（`gate.type: always_skip`）—— 框架强制走 inference，trajectory history 自然累积。

**选方案 b**：因为 Step 1b 的 trajectory history 语义必须与 Step 2/3 完全一致，而 always_skip 与 disable cp1 的 side effect 可能不同（disable 会让 orchestrator 完全跳过 check → 无 broadcast 记录）。保持 always_skip。

**YAML 复用**：Step 1b 可直接用 Step 2 的 `inference_{config}.yaml`（E1-E3），三个配置各跑一次或选一个代表；但因 Step 2 需要 per-config GT（key_builder 的 vision history 依赖 artifact），保险起见**每个 config 收一套 GT**。

---

## 10. Step 2 Runner — `exp/compute_deviate_scores.py`（Layer F3）

**职责**：双阶段并行 replay GT obs 序列（§13.4）。

### 10.1 整体结构

```python
# exp/compute_deviate_scores.py
from exp._run_state_base import BaseRunState, UnitState
from exp._cache_config_rpc import send_load_cache_config
from openpi_client.websocket_client_policy import WebsocketClientPolicy
import h5py, json, numpy as np, msgpack_numpy
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------- Phase 1: background L2 (AlwaysSkipGate) -------------
class Phase1Runner(BaseRunState):
    """Unit: (config, episode_id, sample_idx). Each unit = one websocket
    connection rolling over the full GT obs sequence, producing T actions.
    """
    def build_units(self):
        return [
            UnitState(unit_key=f"{self.config_id}:{ep}:{s}")
            for ep in self.gt_episodes for s in range(self.M)
        ]

    def execute_unit(self, u):
        cfg, ep, s = u.unit_key.split(":")
        obs_seq, _ = load_gt_episode(self.gt_dir, ep)
        client = WebsocketClientPolicy(host=self.host, port=self.port)
        client.episode_start(
            experiment="deviate_score_phase1",
            task=f"ep{ep}_s{s}",
            episode_id=int(ep) * 1000 + int(s),
            # 注意：不开 --collect，actions 自己收；避免 HDF5 落盘开销
        )
        actions = []
        for t, obs in enumerate(obs_seq):
            resp = client.infer(obs)
            actions.append(np.asarray(resp["action"]))
        client.episode_end(success=True)
        # dump to jsonl
        out = Path(self.out_dir) / f"bg_{cfg}.jsonl"
        with out.open("a") as f:
            f.write(json.dumps({
                "config": cfg, "episode": ep, "sample_idx": int(s),
                "actions": [a.tolist() for a in actions],
            }) + "\n")
        return {"T": len(actions)}

# ------------- Phase 2: cache L2 -------------
class Phase2Runner(BaseRunState):
    """Unit: (config, episode_id). Cache is deterministic -> 1 sample."""
    def build_units(self):
        return [
            UnitState(unit_key=f"{self.config_id}:{ep}")
            for ep in self.gt_episodes
        ]

    def execute_unit(self, u):
        cfg, ep = u.unit_key.split(":")
        obs_seq, _ = load_gt_episode(self.gt_dir, ep)
        client = WebsocketClientPolicy(host=self.host, port=self.port)
        client.episode_start(experiment="deviate_score_phase2", task=f"ep{ep}",
                             episode_id=int(ep))
        actions = [np.asarray(client.infer(obs)["action"]) for obs in obs_seq]
        client.episode_end(success=True)
        out = Path(self.out_dir) / f"cache_{cfg}.jsonl"
        with out.open("a") as f:
            f.write(json.dumps({
                "config": cfg, "episode": ep,
                "actions": [a.tolist() for a in actions],
            }) + "\n")
        return {"T": len(actions)}
```

**并发（ThreadPoolExecutor）**：`BaseRunState.run` 目前串行；此处需要扩展为 `run_parallel(num_workers)`。方案：**`BaseRunState` 加 `parallel_run(num_workers)`**：

```python
# 加到 _run_state_base.py
def parallel_run(self, *, num_workers: int, resume: bool = False):
    if resume: self.load()
    if not self.units:
        for u in self.build_units():
            self.units[u.unit_key] = u
    self.save()
    queue = [u for u in self.units.values() if u.status != "done"]

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(self._execute_one, u): u for u in queue}
        for _ in as_completed(futures):
            pass   # _execute_one already persists state

    # retry (parallel, same pattern)
    for attempt in range(1, self.max_retries + 1):
        fails = self.failed_units()
        if not fails: break
        for u in fails:
            u.retry_count = attempt; u.status = "pending"
        self.save()
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(self._execute_one, u): u for u in fails}
            for _ in as_completed(futures): pass
```

**并发数控制**：`num_workers = min(args.num_workers, args.M)`。单机 GPU 显存约束 N ≤ 8（经验值，见 §13.4）。

### 10.2 Phase 3: offline aggregation

```python
def aggregate(bg_path: Path, cache_path: Path, gt_dir: Path, out_path: Path):
    # Load bg samples: {episode: [sample_idx -> [T, action_dim]]}
    bg = defaultdict(dict)
    for line in bg_path.read_text().splitlines():
        d = json.loads(line)
        bg[d["episode"]][d["sample_idx"]] = np.asarray(d["actions"])
    cache = {}
    for line in cache_path.read_text().splitlines():
        d = json.loads(line)
        cache[d["episode"]] = np.asarray(d["actions"])

    out = {}
    for ep, samples in bg.items():
        M = len(samples)
        T = samples[0].shape[0]
        bg_stack = np.stack([samples[s] for s in range(M)])   # (M, T, Ad)
        # pairwise L2 means over samples, per step
        # bg_stack reshape to (M, T*Ad) -> for each t, pairwise over M
        bg_l2 = []
        for t in range(T):
            v = bg_stack[:, t, :]  # (M, Ad)
            d = np.linalg.norm(v[:, None, :] - v[None, :, :], axis=-1)
            # upper-triangular mean (C(M,2))
            iu = np.triu_indices(M, k=1)
            bg_l2.append(float(d[iu].mean()))
        # cache L2 vs GT action
        gt_actions = load_gt_actions(gt_dir, ep)   # (T, Ad)
        cache_l2 = np.linalg.norm(cache[ep] - gt_actions, axis=-1).tolist()
        # deviate score
        floor = 0.1
        dev = [cl / max(bg, floor) for cl, bg in zip(cache_l2, bg_l2)]
        out[ep] = {
            "background_l2": bg_l2,
            "cache_l2": cache_l2,
            "deviate_score": dev,
        }
    out_path.write_text(json.dumps(out, indent=2))
```

### 10.3 Main entrypoint

```python
def main():
    args = parse_args()
    for cfg in args.configs:
        # Phase 1
        send_load_cache_config(server_url, f"configs/cache_runs/deviate_exp/inference_{cfg}.yaml")
        Phase1Runner(config_id=cfg, M=args.M, ...).parallel_run(
            num_workers=args.num_workers, resume=args.resume,
        )
        # Phase 2
        send_load_cache_config(server_url, f"configs/cache_runs/deviate_exp/cache_{cfg}.yaml")
        Phase2Runner(config_id=cfg, ...).parallel_run(
            num_workers=args.num_workers, resume=args.resume,
        )
        # Phase 3
        aggregate(
            bg_path=Path(args.out_dir) / f"bg_{cfg}.jsonl",
            cache_path=Path(args.out_dir) / f"cache_{cfg}.jsonl",
            gt_dir=Path(args.gt_dir),
            out_path=Path(args.out_dir) / f"deviate_score_{cfg}.json",
        )
```

### 10.4 `load_gt_episode` 辅助函数

```python
def load_gt_episode(gt_dir: Path, ep_name: str) -> tuple[list[dict], np.ndarray]:
    """Read GT episode HDF5 -> (obs_seq, clean_actions).
    obs_seq is a list of dicts matching the libero policy's expected obs format:
      {
        "observation/image": <agentview H×W×3>,
        "observation/wrist_image": <eye_in_hand H×W×3>,
        "observation/state": <robot_state 8D>,
        "prompt": <task language string>,
      }
    """
    path = Path(gt_dir) / f"{ep_name}.h5"
    with h5py.File(path, "r") as f:
        prompt = f.attrs.get("task_name") or f.attrs.get("prompt")
        obs_seq, actions = [], []
        for i in range(f.attrs["num_steps"]):
            g = f[f"step_{i:04d}"]
            obs_seq.append({
                "observation/image": g["agentview_image"][...],
                "observation/wrist_image": g["eye_in_hand_image"][...],
                "observation/state": g["robot_state"][...],
                "prompt": str(prompt),
            })
            actions.append(g["action"][...])
    return obs_seq, np.stack(actions)
```

---

## 11. Step 3 Runner — `exp/run_spawn_experiment.py`（Layer F4）

### 11.0 Phase 0 Smoke Scripts

**`scripts/verify_env_save_restore.py`**（对应原 plan §4.4）：

```python
# 最小验证 env.get_sim_state / env.set_init_state
import numpy as np
from libero.libero.benchmark import get_benchmark_dict
from libero.libero.envs import OffScreenRenderEnv

suite = get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
env_args = {"bddl_file_name": task.bddl_file, "camera_heights": 128, "camera_widths": 128}
env = OffScreenRenderEnv(**env_args)
env.reset()
init_state = suite.get_task_init_states(0)[0]
env.set_init_state(init_state)

# run 50 random steps
acts = [np.random.uniform(-0.1, 0.1, 7) for _ in range(50)]
traj_a = []
for a in acts: traj_a.append(env.step(a)[0]["robot0_eef_pos"].copy())
ckpt = {"sim_state": env.get_sim_state().copy(),
        "timestep": env.timestep, "cur_time": env.cur_time}

# 30 more steps (to diverge)
for _ in range(30): env.step(np.random.uniform(-0.1, 0.1, 7))

# restore
obs = env.set_init_state(ckpt["sim_state"])
env.timestep, env.cur_time, env.done = ckpt["timestep"], ckpt["cur_time"], False

traj_b = []
for a in acts[:20]: traj_b.append(env.step(a)[0]["robot0_eef_pos"].copy())

# NOTE: trajectories won't match 精确—but should be within physics tolerance
# Instead verify: success flag determinism + step count + basic obs shape
print("Restore OK:", np.allclose(traj_a[:len(traj_b)], traj_b, atol=1e-6))
```

### 11.1 Runner 代码骨架

```python
# exp/run_spawn_experiment.py
from exp._run_state_base import BaseRunState, UnitState
from exp._cache_config_rpc import send_load_cache_config
from openpi_client.websocket_client_policy import WebsocketClientPolicy
import h5py, json, numpy as np
from pathlib import Path

class SpawnRunner(BaseRunState):
    """Unit key: f'{cfg}:{ep}:s{s}:n{n}:k{k_idx}'
    Where:
      cfg: deviate config id (clip_w7_d4 / spatial16_w8_d4 / max_pool_w3_d5)
      ep:  GT episode name (e.g. "task_3/episode_0")
      s:   intervention point (absolute step in GT trajectory)
      n:   rollout length after intervention (1, 3, 5, 10, 20)
      k_idx: which of the top-k points this unit represents (0..k-1);
             only used when k > 1 (strategy A independent spawn per point,
             so k_idx == which point of the top-k list, not a compound key)
    """

    def __init__(self, *, cfg, gt_dir, deviate_score_path, D, out_dir,
                 host, port, n_grid, k_grid, **kwargs):
        super().__init__(state_path=Path(out_dir) / f"spawn_state_{cfg}.json")
        self.cfg = cfg
        self.gt_dir = Path(gt_dir)
        self.scores = json.loads(Path(deviate_score_path).read_text())
        self.D = D
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.host, self.port = host, port
        self.n_grid = n_grid
        self.k_grid = k_grid

    def build_units(self):
        units = []
        for ep, data in self.scores.items():
            top_points = self._top_k_points(data["deviate_score"], max(self.k_grid))
            for k in self.k_grid:
                for k_idx, s in enumerate(top_points[:k]):
                    for n in self.n_grid:
                        units.append(UnitState(
                            unit_key=f"{self.cfg}:{ep}:s{s}:n{n}:k{k_idx}"
                        ))
        return units

    def execute_unit(self, u):
        # 1. parse unit key
        cfg, ep, stag, ntag, ktag = u.unit_key.split(":")
        s, n, k_idx = int(stag[1:]), int(ntag[1:]), int(ktag[1:])

        # 2. read GT HDF5
        with h5py.File(self.gt_dir / f"{ep}.h5", "r") as f:
            T = f.attrs["num_steps"]
            sim_state = f[f"step_{s+n:04d}"]["sim_state"][...]
            env_timestep = int(f[f"step_{s+n:04d}"]["env_timestep"][()])
            env_cur_time = float(f[f"step_{s+n:04d}"]["env_cur_time"][()])
            task_name = f.attrs["task_name"]
            task_id = int(f.attrs["task_id"])
            # D-1 prefill history: steps [s+n-D+1 .. s+n-1]
            prefill_start = max(0, s + n - self.D + 1)
            prefill_obs = []
            prefill_actions = []
            for t in range(prefill_start, s + n):
                g = f[f"step_{t:04d}"]
                prefill_obs.append(_build_obs_dict(g, task_name))
                prefill_actions.append(g["action"][...])
            start_obs = _build_obs_dict(f[f"step_{s+n:04d}"], task_name)

        # 3. teleport env
        env = _build_env(task_id, task_name)
        env.reset()
        env.set_init_state(sim_state)
        env.timestep, env.cur_time, env.done = env_timestep, env_cur_time, False

        # 4. open connection and prefill
        client = WebsocketClientPolicy(host=self.host, port=self.port)
        client.episode_start(
            experiment=f"spawn_{cfg}",
            task=task_name,
            episode_id=0,
            episode_name="",   # no HDF5 capture for spawn runs
        )
        if prefill_actions:
            # Single atomic prefill — server side uses per-connection facade (§5)
            client.prefill_trajectory(
                observations=prefill_obs,
                actions=prefill_actions,
            )

        # 5. pure cache rollout from step s+n
        obs = start_obs
        success = False
        final_step = s + n
        for step in range(s + n, s + n + self._max_spawn_steps(T)):
            resp = client.infer(obs)
            action = np.asarray(resp["action"])
            obs_env, reward, done, info = env.step(action[0])  # take first of chunk
            obs = _obs_env_to_policy(obs_env, task_name)
            final_step = step + 1
            if info.get("success", False):
                success = True
                break
            if done:
                break
        client.episode_end(success=success)

        # 6. record
        return {
            "success": success, "final_step": final_step,
            "s": s, "n": n, "k_idx": k_idx, "episode": ep,
        }

    @staticmethod
    def _top_k_points(scores, k):
        return sorted(range(len(scores)), key=lambda i: -scores[i])[:k]

def main():
    args = parse_args()
    for cfg in args.configs:
        send_load_cache_config(server_url, f"configs/cache_runs/deviate_exp/cache_{cfg}.yaml")
        runner = SpawnRunner(cfg=cfg, ..., n_grid=[1,3,5,10,20], k_grid=[1,3,5])
        runner.parallel_run(num_workers=args.num_workers, resume=args.resume)

    # aggregate
    aggregate_spawn_results(args.out_dir, args.configs)
```

### 11.2 Baseline 对照（§6.3）

在 `SpawnRunner.build_units` 之外，单独一个 `BaselineRunner`（同 `_run_state_base`），unit key 形如 `f"{cfg}:{ep}:{strategy}:n{n}:k{k}:idx{i}"`，strategy ∈ {random-3, equidistant-3}。`_top_k_points` 换成 `_random_k_points(T, k, seed=idx)` / `_equidistant_k_points(T, k)`。其余 execute_unit 逻辑与 SpawnRunner 一致。

### 11.3 聚合

```python
def aggregate_spawn_results(out_dir, configs):
    rows = []
    for cfg in configs:
        state = json.loads((Path(out_dir) / f"spawn_state_{cfg}.json").read_text())
        for key, u in state.items():
            if u["status"] != "done": continue
            r = u["result"]
            rows.append({
                "config": cfg, "episode": r["episode"],
                "s": r["s"], "n": r["n"], "k_idx": r["k_idx"],
                "success": r["success"], "final_step": r["final_step"],
            })
    # dump CSV
    import pandas as pd
    pd.DataFrame(rows).to_csv(Path(out_dir) / "spawn_aggregate.csv", index=False)
```

---

## 12. `exp/analyze_deviation_results.py`（Layer F5）

**职责**：生成统计图表（直方图、success rate heatmap 等）。

**主要可视化**：

1. **Deviate score 分布直方图** — per config per episode，x=score, y=count, 带 threshold=1 竖线。
2. **Top-k 覆盖率** — ROC-like：x = cumulative top-k% of steps by score, y = fraction of "true deviate points" captured（以 spawn 成功为 ground truth）。
3. **Success rate 热力图** — x=n, y=k, color=success_rate，per config 3 张图。
4. **Baseline 对照** — bar chart：pure_cache vs top-k vs random-k vs equidistant-k at best (n, k)。

输入：`deviate_score_{cfg}.json` + `spawn_aggregate.csv`。输出：`data/deviation_experiment/figures/*.png`。

---

## 13. HDF5 Schema 汇总

本表集中列出所有涉及的 HDF5 文件结构，便于跨脚本对齐。

### 13.1 Client-side GT trajectory（§4.5 扩展）

**Path**: `data/deviation_experiment/gt_trajectories/task_{id}/episode_{idx}.h5`

```
attrs:
  task_name: str
  task_id: int
  init_state_idx: int      # subset-local index (0..K-1)
  orig_init_state_idx: int  # original benchmark index (from Step 1a results map)
  episode_id: int
  seed: int
  num_steps: int
  success: bool
  cache_success: bool       # from Step 1a (为 False — 本集都是 cache 失败的)
  prompt: str               # raw language string (for Step 2/3 obs reconstruction)

step_0000/:
  sim_state: (N,) float64           # env.get_sim_state() flattened
  env_timestep: int
  env_cur_time: float64
  agentview_image: (224,224,3) uint8    # post-transform image
  eye_in_hand_image: (224,224,3) uint8
  robot_state: (8,) float32
  action: (action_horizon, action_dim) float32   # action chunk returned by model
step_0001/: ...
```

### 13.2 Server-side collected episode (`--collect`)

**Path**: `{collect_dir}/{experiment}/task_{id}/episode_{idx}.h5`（`episode_name` 命名；§3.4）

```
attrs:
  experiment_name: str
  task: str
  episode_id: int
  num_steps: int
  timestamp: iso8601
  success: bool
  prompt: str              # NEW (§3.3) — raw language prompt

step_0000/:
  vision_emb: ...
  prompt_emb: ...
  clean_action: (chunk, dim)
  noise_action_steps: (K, chunk, dim)
  ...
```

**对齐**：`data/deviation_experiment/gt_trajectories/task_X/episode_Y.h5` 和 `{collect_dir}/libero_spatial/task_X/episode_Y.h5` 同名对齐，Step 2 若需要 model intermediates 可从 server-side 读。

### 13.3 Step 2 jsonl（轻量）

- `bg_{cfg}.jsonl`：每行 `{"config", "episode", "sample_idx", "actions": [[...], [...]]}`（T × action_dim）
- `cache_{cfg}.jsonl`：每行 `{"config", "episode", "actions": [[...], [...]]}`
- `deviate_score_{cfg}.json`：`{episode: {background_l2, cache_l2, deviate_score}}`

---

## 14. State Schema 汇总（各 Runner）

| Runner | State 文件 | Unit key | Unit.result 字段 |
|--------|-----------|---------|-----------------|
| `run_cache_experiments.py` | `configs/cache_runs/phase1/experiment_state.json` (既有) | `(yaml, task_id)` | 聚合：`task_progress`、`task_results` |
| `run_step1b_gt.py` | `data/step1b_inits/.../step1b_state.json` | `"{task_id}:{orig_init_idx}"` | `{success, hdf5_path, inference_failed?}` |
| `compute_deviate_scores.py::Phase1Runner` | `data/deviation_experiment/phase1_state_{cfg}.json` | `"{cfg}:{ep}:{s}"` | `{T}` |
| `compute_deviate_scores.py::Phase2Runner` | `data/deviation_experiment/phase2_state_{cfg}.json` | `"{cfg}:{ep}"` | `{T}` |
| `run_spawn_experiment.py::SpawnRunner` | `data/deviation_experiment/spawn_state_{cfg}.json` | `"{cfg}:{ep}:s{s}:n{n}:k{k_idx}"` | `{success, final_step, s, n, k_idx, episode}` |

所有 state 文件由 `BaseRunState.save()` 统一维护，JSON 顶层为 `{unit_key: UnitState}` dict。

---

## 15. 实现顺序（checkpoint 列表）

按依赖顺序逐点验证，每一步都应可独立 smoke test。

### 15.1 Layer A + Layer D + Layer E (基础设施、不动 runner)

1. **`exp/_run_state_base.py`** —— smoke test：mock subclass 5 个 unit，跑通 load/save/retry。
2. **`exp/_cache_config_rpc.py`** —— smoke test：对活 server 发一次 load_cache_config，断言 ack。
3. **`AlwaysSkipGate` + config 注册** —— unit test：配置加载成功，orchestrator 接入后 `gate()` 返回 False。
4. **3 份 `cache_*.yaml` + 3 份 `inference_*.yaml`** —— 各自 `CacheConfig.from_yaml` 可加载，`build_shared_storage + build_per_connection_components` 无 error。
5. **`CacheStorage.enter_prefill_mode` + `search/fetch_payload` 分支** —— unit test：
   - 正常模式：backend 被调用；
   - prefill 模式：backend 不被调用，返回合成 hit；
   - exit 后恢复。
6. **`build_per_connection_components` 改为 per-connection facade 实例** —— 集成 test：两个 client 并发连接，一个进入 prefill 模式不影响另一个。

### 15.2 Layer B + Layer C (Client↔Server 透传 + main.py)

7. **`websocket_client_policy.episode_start` 新增 `episode_name`** —— backward-compat test：不传时行为等价。
8. **`websocket_policy_server` 透传 `__episode_name__`** + **`data_collector` 按 name 命名** —— e2e：client 传 `"task_3/episode_2"` → server 落盘到 `{dir}/task_3/episode_2.h5`。
9. **`collection_policy` prompt 抓取 + `data_collector.set_episode_attr`** —— e2e：`--collect` 跑一 episode，HDF5 `attrs["prompt"]` 非空。
10. **`main.py` 5 组新 flag** —— smoke：跑 1 task × 1 episode with `--save-trajectory --save-episode-results`，验证：
    - `{save_trajectory_dir}/task_0/episode_0.h5` 生成且含 `sim_state`；
    - `{episode_results_path}` 生成且格式正确。

### 15.3 Layer D4 + Layer B (Prefill 端到端)

11. **`InferenceInterceptor.prefill_trajectory`** —— unit test：mock orchestrator，灌 3 步 obs+action，验证：
    - `key_builder.collect/build` 各被调 3 次；
    - `strategy.record_query_keys` / `broadcast_action.record_action` 各被调 3 次；
    - `_step_counter += 3`。
12. **Server `prefill_trajectory` 控制分支** —— e2e：client 发 `prefill_trajectory`，server ack；后续一次 `infer` 的 trajectory history 长度 = prefill_len + 0。
13. **`websocket_client_policy.prefill_trajectory`** —— e2e smoke with real server。

### 15.4 Runner 落地

14. **`scripts/dump_step1a_failed_inits.py`** —— smoke：假 JSON 5 个失败 → 5 个 `.init` 文件。
15. **修改版 `run_cache_experiments.py` + `--save-episode-results` 跑 Step 1a** —— 全量 50 episodes 跑通，产出 `cache_eval_results.json`。
16. **`run_step1b_gt.py`** —— smoke：1 task × 2 failed inits → 2 GT HDF5 + state JSON per-unit 更新。Ctrl-C 再 `--resume` 验证。
17. **`compute_deviate_scores.py` Phase 1+2**（单 config 单 episode M=3）—— smoke：jsonl 产出 3 个 bg sample + 1 个 cache sample + aggregate 成功。
18. **Phase 3 aggregate** —— unit test：合成假 jsonl → 验证 L2 / deviate_score 公式正确。
19. **`run_spawn_experiment.py`**（单 config 单 unit）—— smoke：teleport + prefill + 5 步纯 cache rollout。
20. **Baseline runner** —— smoke：random-k / equidistant-k 跑通。
21. **`analyze_deviation_results.py`** —— 跑全量产出 5 张图。

### 15.5 规模化 & 退出条件

22. **全量跑 Phase 2**（3 config × K failed episodes × 20 samples + 3 × K × 1 cache samples）→ 产出 3 份 `deviate_score_*.json` → **原 plan §5.2 Go/No-Go 判定**。
23. **Go** → 全量跑 Phase 3 → 分析 → **原 plan §12 成功标准判定**。

---

## 16. 未解决/待讨论（仍待 user 确认）

1. **Spawn 策略 A vs B vs C**（原 plan §11 待确认 #1）：本实现 plan **默认实现策略 A**（`SpawnRunner.build_units` 的 `k_idx` 单点独立 spawn）。B 和 C 若启用，需要扩展 unit key 与 execute_unit 逻辑；加在 §11.1 的一个 `spawn_strategy` enum 分支里。
2. ~~AlwaysSkipGate 下 orchestrator 是否触发 broadcast_action~~ **已解决（2026-04-13）**：读 `src/openpi/cache/interceptor.py` 全文后确认：
   - CP1 miss 路径下 interceptor 进入 stage2（L399）→ stage3（L429-435）→ CP3 check（L442-472），最终**在 L464 显式调 `orchestrator.broadcast_action(action_chunk_cpu)`**，随后 L466 `buffer_for_write(query_keys, action, intermediates, denoising_num_steps)` 入 episode buffer。
   - 所以 `AlwaysSkipGate` 接 CP1 返回 False → orchestrator 记 query_keys → 返 MISS → interceptor 继续全量推理 → broadcast_action 到所有组件 → 轨迹 history 连续一致。**语义正确，§7 改动无需额外补 broadcast 逻辑**。
3. **Step 1b 三个 config 共用 GT 可行性**（补充分析，见 §18.B4）：三个 config 的 input_transform 和 vision hooks 并不会改变 server 侧 `CollectionPolicy` 输入给模型的 obs（模型权重 + noise 由 seed 固定）。故**理论上可共用一套 GT 的 obs+action 序列**。但 `--collect` 收的 `vision_emb/prompt_emb/noise_action_steps` 在 Step 2 Phase 1 中**不被使用**（Phase 1 只需要 obs_seq 喂给 client.infer），所以 GT 本身（client-side HDF5 含 `sim_state/robot_state/image/wrist_image/action`）对三个 config 可复用。**决定**：Step 1b 只收一次 GT（使用 `inference_clip_w7_d4.yaml` 作为"纯 inference" 配置），Step 2 三个 config 共享此 GT。Step 3 同理。若后续发现 prompt_emb / vision_emb 需要对齐特定 config，再分 3 套。

---

## 17. 附录 — 关键文件 line-level 改动汇总表

便于 PR review / code search：

| 文件 | 锚点行号 | 改动类型 | 本文档章节 |
|------|--------|---------|----------|
| `packages/openpi-client/src/openpi_client/websocket_client_policy.py:56-70` | 修改 | 替换 episode_start + 新增 prefill_trajectory | §3.1 |
| `src/openpi/serving/websocket_policy_server.py:207-214` | 修改 | episode_start 透传 episode_name | §3.2 |
| `src/openpi/serving/websocket_policy_server.py:218 (after)` | 插入 | prefill_begin / prefill_end / prefill_trajectory 分支 | §3.2 |
| `src/openpi/collect/collection_policy.py:103-105` | 修改 | on_episode_start 透传 + prompt 抓取 | §3.3 |
| `src/openpi/collect/data_collector.py:40-46, 70-71, 76-81` | 修改 | episode_name 命名 + set_episode_attr + attrs 写入 | §3.4 |
| `examples/libero/main.py:31-66` | 修改 | Args 新增 5 组 flag | §4 |
| `examples/libero/main.py:83-94, 139, 146, 152` | 修改 | _run_episode buffer 逻辑 | §4 |
| `examples/libero/main.py:175, 179, 280, 311` | 修改 | episode_filter + episode_name 传递 | §4 |
| `src/openpi/cache/cache_storage.py:48-55, 61, 71` | 修改 | prefill 模式成员 + search/fetch 分支 | §6 |
| `src/openpi/cache/interceptor.py:270-276 (after)` | 插入 | prefill_trajectory + _build_prefill_payload | §7 |
| `src/openpi/cache/components/gate.py:74 (after)` | 插入 | AlwaysSkipGate class | §8.1 |
| `src/openpi/cache/config.py:438-439` | 修改 | validation 白名单 | §8.2 |
| `src/openpi/cache/config.py:864-871` | 修改 | _build_gate 新增分支 | §8.2 |
| `src/openpi/cache/config.py:713-761` | 修改 | per-connection storage facade | §5 |
| `exp/run_cache_experiments.py:110-206` | 修改 | main_args 添加 --save-episode-results | §9.0 |
| (新建) `exp/_run_state_base.py` | 新建 | BaseRunState + UnitState + parallel_run | §2.1, §10.1 |
| (新建) `exp/_cache_config_rpc.py` | 新建 | send_load_cache_config / send_prefill_begin/end | §2.2 |
| (新建) `exp/run_step1b_gt.py` | 新建 | Step1bRunner | §9.2 |
| (新建) `exp/compute_deviate_scores.py` | 新建 | Phase1/2/3 Runner + aggregate | §10 |
| (新建) `exp/run_spawn_experiment.py` | 新建 | SpawnRunner + BaselineRunner | §11 |
| (新建) `exp/analyze_deviation_results.py` | 新建 | 统计与可视化 | §12 |
| (新建) `scripts/dump_step1a_failed_inits.py` | 新建 | 失败 init 文件 dump | §9.1 |
| (新建) `scripts/verify_env_save_restore.py` | 新建 | Phase 0 smoke | §11.0 |
| (新建) `configs/cache_runs/deviate_exp/{cache,inference}_*.yaml` × 6 | 新建 | YAML 配置 | §8.3 |

---

## 18. 深化细节（A/B 级 gap 补全 · 2026-04-13）

本节按 §0.1 之外新增的"A 级关键缺口"和"B 级重要缺口"逐项展开，补齐读源码后才确认的代码级细节。每项标注**对应修正的章节**，便于定位。

---

### 18.A1 InferenceInterceptor.infer 完整路径图（修正 §7）

**背景**：§7 提供了 `prefill_trajectory` 新方法，但未详述 `infer()` 内部每一条路径的 prefill 行为。读 `src/openpi/cache/interceptor.py:301-485` 全文后，对每条分支的 prefill 语义做完整标注。

#### 18.A1.1 源码路径图（`interceptor.py:301-485`）

```
infer(obs, *, noise=None)  [L301]
├── L322-338  input_transforms → obs_in
├── L352-353  stage1_out = self._run_stage1(obs_in)         [vision encode]
├── L356-384  CP1 check：
│     orchestrator.check(CP1, cached_data=stage1_out)       [L356]
│     ├── FULL_HIT:                                          [L360-372]
│     │     cached_action = cp1.payload.action_chunk         [L362]
│     │     orchestrator.broadcast_action(cached_action)     [L367]
│     │     orchestrator.buffer_for_write(... miss_by_cp=... )[L371]
│     │     return cached_action                             [L374]
│     └── MISS / WARM_START: fall through
├── L399-400  stage2_out = self._run_stage2(stage1_out, obs_in)   [LLM cross-attn]
├── L408-412  compute noise for stage3 (seed/deterministic)
├── L416-428  WARM_START path:                                [CP3 WARM]
│     start_x = cp3.payload.intermediates[start_t]            [L419]
│     action_chunk = self._run_stage3_from(
│         stage2_out, start_x, start_t,
│         num_steps=cp3.payload.denoising_num_steps)          [L422]
│     intermediates_out = {}   (不再记 intermediates)         [L425]
├── L429-435  MISS path:                                      [CP3 MISS]
│     action_chunk, intermediates_out = self._run_stage3(
│         stage2_out, noise,
│         num_steps=_NUM_STEPS, return_intermediates=True)    [L431]
├── L442-472  CP3 check (仅 MISS/WARM_START 到这里)：
│     orchestrator.check(CP3, cached_data=stage2_out)         [L444]
│     ├── FULL_HIT: cache 返回 action → 用 cached action_chunk
│     └── MISS / WARM_START:
│           action_cpu = action_chunk.detach().cpu().float()  [L462]
│           orchestrator.broadcast_action(action_cpu)         [L464]
│           orchestrator.buffer_for_write(
│               query_keys, action_cpu, intermediates_out,
│               denoising_num_steps)                          [L466]
└── L484-485  return action_chunk
```

#### 18.A1.2 Prefill 模式在各分支上的表现

进入 `enter_prefill_mode(payload)` 后，`CacheStorage.search` 对任意 `spec` 返回 `[SearchResultLite(id="__prefill__", score=1.0, cp_id=spec.checkpoint_id)]`，`fetch_payload` 忽略 id 返回 stored payload（§6）。效果传导到每一 CP 分支如下：

| 分支 | orchestrator.check 返回 | infer 行为 | stage1/2/3 forward 是否执行？ |
|------|-------------------------|-----------|----------------------------|
| L356-384 CP1 FULL_HIT（prefill 注入） | `HitType.FULL_HIT`，`cp1.payload = injected_payload` | 在 L374 早 return `cached_action`；broadcast_action ＋ buffer_for_write 都执行 | **stage1 已跑**（L352 在 check 前），**stage2/3 不跑** |
| L356-384 CP1 MISS/WARM_START | 不会发生（gate 非 skip、judge score=1.0） | — | — |
| L416-428 CP3 WARM_START | 不会发生（在 prefill 模式下 CP1 已 FULL_HIT, 不会到这里） | — | — |
| L442-472 CP3 FULL_HIT | 不会发生（同上） | — | — |

**关键事实**：prefill 路径下**唯一被真正执行的 forward 是 stage1**（vision encoder）—— 这是因为 CP1 check 在 stage1 之后。原 plan §9.5 早已认可"stage1 forward 在 prefill 期间被浪费"是可接受的代价（D-1 次，远小于真正 inference 成本）。

#### 18.A1.3 AlwaysSkipGate 模式下的路径（Step 2 Phase 1）

Gate 返回 False 走 miss 分支，与 prefill 正好相反：

| 分支 | 行为 |
|------|------|
| L356 CP1 check (gate=AlwaysSkip) | gate_fn 返回 False → orchestrator 记 `spec.query_keys` 到 strategy trajectory → 返 `HitType.MISS` |
| L399 stage2 | **执行** |
| L429-435 stage3 MISS | **执行**（完整 20 步 denoising） |
| L442 CP3 check | `AlwaysSearchGate` 仍打开 → judge 判分；若设计上这里也要强制 inference，需单独把 CP3 的 gate 也设为 `always_skip`。对 Step 2 而言 **CP3 gate 必须是 always_skip** 才能保证 100% inference（否则 CP3 的 cache 缓存会污染 "独立 sample" 的语义） |
| L464 broadcast_action | **执行**，把刚算出的 action_chunk_cpu 喂回 `AlwaysSkipGate.record_action` / `SearchStrategy.record_action` / `KeyBuilder.record_action` |
| L466 buffer_for_write | **执行**，把这次 MISS 记到 episode buffer |

**Step 2 Phase 1 的 YAML 要求**（修正 §8.3 inference_*.yaml 的"唯一差异"表述）：
```yaml
# inference_*.yaml 相对于 cache_*.yaml 的差异不仅是 CP1，CP3 也要改
checkpoints:
  cp1:
    gate: { type: always_skip }   # 强制 inference
  cp3:
    gate: { type: always_skip }   # 防止 flow matching cache 干扰 M sample 独立性
```

**这条是 §8.3 的修正**：inference_*.yaml 必须对 cp1 和 cp3 **同时**改为 always_skip，否则 Phase 1 的 M 个 background sample 会因 CP3 命中而失去独立性。

---

### 18.A2 Per-connection Prefill 精确代码改动（修正 §5 + §3.2）

**背景**：§5 已经说明了需要改 `build_per_connection_components`，但没有落到**精确的一行代码**。读 `src/openpi/cache/config.py:713-761` 全文后，确认改动点是 L755。

#### 18.A2.1 `src/openpi/cache/config.py:713-761` 现状

```python
def build_per_connection_components(
    config: CacheConfig,
    shared_storage: CacheStorage,
    *,
    quiet: bool = False,
) -> dict[str, Any]:
    """Build per-connection components: timer, key_builder, gate, judge, strategy.
    Storage is shared across connections (one backend).
    """
    # ... (timer/key_builder/gate/judge/strategy construction) ...

    return {
        "timer": timer,
        "key_builder": key_builder,
        "gate": gate,
        "judge": judge,
        "strategy": strategy,
        "storage": shared_storage,   # ← L755 (precise line)
    }
```

#### 18.A2.2 改动（一行替换 + 一行 import）

```python
# 在函数顶部 import 区追加（如果 CacheStorage 尚未 imported，文件顶部已有）
from openpi.cache.cache_storage import CacheStorage

def build_per_connection_components(config, shared_storage, *, quiet=False):
    # ... 前半部分不变 ...

    # NEW: wrap backend in a fresh facade per connection so that
    # enter/exit_prefill_mode is connection-local.
    per_conn_storage = CacheStorage(
        backend=shared_storage._backend,
        metadata_db=shared_storage._metadata_db,   # 如果存在
    )

    return {
        "timer": timer,
        "key_builder": key_builder,
        "gate": gate,
        "judge": judge,
        "strategy": strategy,
        "storage": per_conn_storage,   # CHANGED: was shared_storage
    }
```

**为什么 safe**：
- `CacheStorage.__init__` 只保存 `self._backend` + `self._metadata_db` + `self._dims = backend.vector_dims`（见 `cache_storage.py:46-54`），没有任何实际存储。拷贝一个 facade 实例内存占用 < 1 KB。
- 所有 `insert / batch_insert / delete / search / fetch_payload` 都 forward 到 `self._backend`（单例），**没有 per-connection 状态会和共享状态冲突**。
- 唯一新增的 per-connection 状态是 `self._prefill_mode / self._prefill_payload`（§6），**这正是我们想要连接隔离的**。

#### 18.A2.3 server 侧 prefill 控制消息的路由修正

§3.2 之前版本在 `prefill_begin` 分支直接调 `current_bundle.shared_storage.enter_prefill_mode(payload)` —— **这是错的**，会影响所有连接。修正：

```python
# src/openpi/serving/websocket_policy_server.py
# 替换 prefill_begin / prefill_end 分支（§3.2 改动 2）

elif ctrl == "prefill_begin":
    import base64, msgpack
    from openpi.cache.storage_types import CachePayload
    raw = base64.b64decode(obs["payload_b64"])
    payload = CachePayload(**msgpack.unpackb(raw))
    # policy 是 per-connection wrapped InferenceInterceptor；
    # policy._orchestrator._storage 是本连接独有的 CacheStorage facade（§5）
    policy._orchestrator._storage.enter_prefill_mode(payload)
    await websocket.send(packer.pack({"__ack__": "prefill_begin"}))
    continue

elif ctrl == "prefill_end":
    policy._orchestrator._storage.exit_prefill_mode()
    await websocket.send(packer.pack({"__ack__": "prefill_end"}))
    continue

elif ctrl == "prefill_trajectory":
    # 推荐路径：整段轨迹 prefill 由 interceptor 内部托管 enter/exit
    observations = obs["observations"]
    actions = obs["actions"]
    record = obs.get("record", False)
    on_miss = obs.get("on_miss", "error")
    policy.prefill_trajectory(observations, actions, record=record, on_miss=on_miss)
    await websocket.send(packer.pack({"__ack__": "prefill_trajectory"}))
    continue
```

**注意**：`policy._orchestrator._storage` 访问 private 属性；如果担心可读性/稳定性，加一个 `policy.cache_storage` property 转发（interceptor.py 改 2 行）：

```python
# src/openpi/cache/interceptor.py（在 InferenceInterceptor 类内加）
@property
def cache_storage(self):
    return self._orchestrator._storage
```

然后 server 用 `policy.cache_storage.enter_prefill_mode(payload)`。

---

### 18.A3 load_gt_episode 与 LIBERO obs 精确对齐（修正 §10.4）

**背景**：§10.4 给出的 `load_gt_episode` 只是骨架。读 `examples/libero/main.py:117-138` 后，确认 client 喂给 server 的 obs 必须**字节级等价**于 main.py 的实时构造，否则模型行为发散。

#### 18.A3.1 `main.py:117-138` 的 obs 构造（ground truth）

```python
# examples/libero/main.py L117-138（实时 rollout 时）
img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
img = image_tools.convert_to_uint8(
    image_tools.resize_with_pad(img, args.resize_size, args.resize_size))   # 224×224
wrist_img = image_tools.convert_to_uint8(
    image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size))
element = {
    "observation/image": img,                                  # (224,224,3) uint8
    "observation/wrist_image": wrist_img,                      # (224,224,3) uint8
    "observation/state": np.concatenate((
        obs["robot0_eef_pos"],                                 # (3,)
        _quat2axisangle(obs["robot0_eef_quat"]),               # (3,)
        obs["robot0_gripper_qpos"],                            # (2,)
    )),                                                         # total (8,) float64
    "prompt": str(task_description),                           # Python str
}
action_chunk = client.infer(element)["actions"]
```

**关键事实**：
- raw env obs 图像是 256×256，**翻转** `[::-1, ::-1]` 后再 resize_with_pad 到 224×224。
- robot_state 是 8D：eef_pos(3) + axisangle(3, 由 quat 变换) + gripper_qpos(2)。
- prompt 是 Python str（msgpack 直接支持），在 server 侧直接走 policy pipeline。

#### 18.A3.2 GT 收集侧必须保存原始/处理后哪份？

**决定**：HDF5 存 **处理后的 224×224 uint8 图 + 8D robot_state**，直接就是 `client.infer` 的输入。好处：
1. replay 时无需知道 raw env obs keys（agentview_image / robot0_eye_in_hand_image），HDF5 schema 稳定。
2. `load_gt_episode` 不需要重跑 resize/flip pipeline（省时间 + 保字节一致）。
3. Phase 0 smoke test（§11.0）不需要复现 env obs 格式。

对应 `main.py:_run_episode` 里的 buffer step 改为（替换 §4 里的 step_record 片段）：

```python
# examples/libero/main.py 里，L139 infer 之后、L146 env.step 之前
if args.save_trajectory:
    step_record = {
        "sim_state": env.get_sim_state().copy(),       # 1-D float64 (env restore 用)
        "env_timestep": int(env.timestep),
        "env_cur_time": float(env.cur_time),
        "agentview_image": img.copy(),                 # 224×224×3 uint8 (post-transform)
        "eye_in_hand_image": wrist_img.copy(),         # 224×224×3 uint8 (post-transform)
        "robot_state": element["observation/state"].astype(np.float64).copy(),  # 8D
        "action": np.asarray(action[step_in_chunk]).copy(),   # single action executed
        "action_chunk": np.asarray(action_chunk).copy(),      # full chunk from client
    }
    traj_buffer.append(step_record)
```

注：HDF5 的 `agentview_image / eye_in_hand_image` 与 §13.1 对齐（都是 224×224×3 uint8）。§13.1 已经是 post-transform 尺寸，这里只是明确"post-transform"是从 `img / wrist_img` 这两个本地变量取，不是从 raw `obs[...]`。

#### 18.A3.3 `load_gt_episode` 完整实现（替换 §10.4）

```python
def load_gt_episode(gt_dir: Path, ep_name: str) -> tuple[list[dict], np.ndarray]:
    """Read GT episode HDF5 -> (obs_seq, executed_actions).

    obs_seq[i] is ready to pass to `client.infer` — bytes-equivalent to the
    dict main.py constructs at run time (examples/libero/main.py:127-138).
    """
    path = Path(gt_dir) / f"{ep_name}.h5"
    with h5py.File(path, "r") as f:
        prompt = f.attrs.get("prompt") or f.attrs.get("task_name")
        num_steps = int(f.attrs["num_steps"])
        obs_seq: list[dict] = []
        actions: list[np.ndarray] = []
        for i in range(num_steps):
            g = f[f"step_{i:04d}"]
            obs_seq.append({
                "observation/image": np.asarray(g["agentview_image"][...], dtype=np.uint8),
                "observation/wrist_image": np.asarray(g["eye_in_hand_image"][...], dtype=np.uint8),
                "observation/state": np.asarray(g["robot_state"][...], dtype=np.float64),
                "prompt": str(prompt),
            })
            actions.append(np.asarray(g["action"][...], dtype=np.float32))
    return obs_seq, np.stack(actions)   # actions shape: (T, action_dim)
```

#### 18.A3.4 Step 3 Spawn 侧的 obs 构造（env teleport 后首帧）

`run_spawn_experiment.py::execute_unit` 里 env restore 后的 `obs` 是 env.reset/set_init_state 返回的 raw dict —— 必须**再经 main.py:117-138 同样的变换**才能喂给 client.infer。抽一个 helper：

```python
# exp/run_spawn_experiment.py
from openpi_client import image_tools
import numpy as np

def _obs_env_to_policy(env_obs: dict, task_description: str, resize: int = 224) -> dict:
    """Convert raw LIBERO env obs to the dict client.infer expects.
    Mirrors examples/libero/main.py:117-138 exactly.
    """
    img = np.ascontiguousarray(env_obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(env_obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize, resize))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize, resize))
    state = np.concatenate((
        env_obs["robot0_eef_pos"],
        _quat2axisangle(env_obs["robot0_eef_quat"]),
        env_obs["robot0_gripper_qpos"],
    ))
    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": str(task_description),
    }
```

`_quat2axisangle` 直接从 `examples/libero/main.py:490-505` 拷贝到 `exp/run_spawn_experiment.py`（小函数，独立，避免做 `examples/` 到 `exp/` 的依赖）。

---

### 18.A4 CachePayload 跨 WebSocket 序列化（新增 §3.5）

**背景**：`CachePayload`（storage_types.py:60-101）字段包含 `torch.Tensor`，**msgpack_numpy 不支持** torch.Tensor，需要先 `.numpy()`。原 §3.2/§5 的 `prefill_begin` 分支隐含了 base64(msgpack(CachePayload))，没说明 torch 转换，这里补清楚。

#### 18.A4.1 `CachePayload` 的字段清单（`storage_types.py:60-101`）

```python
@dataclass
class CachePayload:
    action_chunk: torch.Tensor            # (action_horizon, action_dim) float32, CPU
    intermediates: Optional[dict[float, torch.Tensor]] = None   # {t: noisy_x_t}
    denoising_num_steps: Optional[int] = None
    task_key: Optional[str] = None
```

#### 18.A4.2 Client 侧打包（`exp/_cache_config_rpc.py` 新增）

```python
# exp/_cache_config_rpc.py
import base64, io
import msgpack
import msgpack_numpy as mpnp
import numpy as np
import torch

mpnp.patch()   # 让 msgpack 能处理 np.ndarray

def pack_payload_b64(payload) -> str:
    """CachePayload → base64(msgpack). torch.Tensor fields → np.ndarray."""
    d = {
        "action_chunk": payload.action_chunk.detach().cpu().contiguous().float().numpy(),
        "denoising_num_steps": payload.denoising_num_steps,
        "task_key": payload.task_key,
    }
    if payload.intermediates is not None:
        d["intermediates"] = {
            float(t): v.detach().cpu().contiguous().float().numpy()
            for t, v in payload.intermediates.items()
        }
    else:
        d["intermediates"] = None
    raw = msgpack.packb(d, default=mpnp.encode, use_bin_type=True)
    return base64.b64encode(raw).decode("ascii")


def send_prefill_begin(server_url: str, payload) -> None:
    """Uniform interface: caller passes a CachePayload-like object; we pack it."""
    payload_b64 = pack_payload_b64(payload)
    asyncio.run(_send_ctrl(
        server_url,
        {"__ctrl__": "prefill_begin", "payload_b64": payload_b64},
        ack="prefill_begin",
    ))
```

#### 18.A4.3 Server 侧解包（`websocket_policy_server.py` prefill_begin 分支）

```python
elif ctrl == "prefill_begin":
    import base64, msgpack
    import msgpack_numpy as mpnp; mpnp.patch()
    import numpy as np
    import torch
    from openpi.cache.storage_types import CachePayload

    raw = base64.b64decode(obs["payload_b64"])
    d = msgpack.unpackb(raw, object_hook=mpnp.decode, raw=False)

    action_chunk = torch.from_numpy(np.ascontiguousarray(d["action_chunk"], dtype=np.float32))
    intermediates = None
    if d.get("intermediates") is not None:
        intermediates = {
            float(t): torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
            for t, v in d["intermediates"].items()
        }
    payload = CachePayload(
        action_chunk=action_chunk,
        intermediates=intermediates,
        denoising_num_steps=d.get("denoising_num_steps"),
        task_key=d.get("task_key"),
    )
    policy._orchestrator._storage.enter_prefill_mode(payload)
    await websocket.send(packer.pack({"__ack__": "prefill_begin"}))
    continue
```

#### 18.A4.4 `prefill_trajectory` 的消息格式（更详细）

**client → server**：

```python
# client 侧：observations + actions 都是 list[...]；actions 走 numpy 直接（msgpack_numpy）
self._ws.send(self._packer.pack({
    "__ctrl__": "prefill_trajectory",
    "observations": observations,                  # list[dict]，dict 里 np.ndarray
    "actions": [np.asarray(a, dtype=np.float32) for a in actions],
    "record": record,
    "on_miss": on_miss,
}))
```

**server 侧**：`obs["actions"]` 已被 msgpack_numpy 解成 `list[np.ndarray]`。interceptor 内部 `_build_prefill_payload` 会把 np.ndarray 转 torch.Tensor（§7 里已写）。无需 base64，因为 action chunk 比 intermediates 小（典型 (10, 7) float32 = 280 B），inline msgpack_numpy 即可。

#### 18.A4.5 size 预估

- 单次 prefill_trajectory：D-1 条（最大 D=5 即 4 条），每条 obs 约 224×224×3 × 2 = 300 KB + action (10×7) = 280 B。总 ~1.2 MB 每请求，msgpack WebSocket 秒级传完。
- intermediates 只在 FULL_HIT WARM_START 时用到，**本实验不走 WARM_START**（prefill 模式强制 CP1 FULL_HIT，CP3 也 FULL_HIT），所以实际可省略 intermediates 字段。`_build_prefill_payload` 留 `intermediates=None`。

---

### 18.B1 _run_episode 精确签名（修正 §4）

**背景**：§4 说 "`_run_episode` 现有签名没有 `task_id/init_state_idx`，需要改 signature"，但没落到**完整函数签名 + 所有调用点更新**。读 `examples/libero/main.py:83-94, 165-215, 218-380` 后，给出完整 diff。

#### 18.B1.1 现有签名（`main.py:83-94`）

```python
def _run_episode(
    env,
    client,
    initial_state,
    task_description,
    args,
    max_steps,
    *,
    record_video: bool = False,
    step_callback=None,
) -> tuple[bool, list[np.ndarray]]:
    ...
```

返回 `(success, video_frames)`。

#### 18.B1.2 修改后签名

```python
def _run_episode(
    env,
    client,
    initial_state,
    task_description,
    args,
    max_steps,
    *,
    record_video: bool = False,
    step_callback=None,
    # --- NEW ---
    task_id: int | None = None,
    init_state_idx: int | None = None,       # subset-local index (0..K-1)
    orig_init_state_idx: int | None = None,  # original benchmark index
    episode_id: int | None = None,
) -> tuple[bool, list[np.ndarray]]:
    """...
    New kwargs (all optional for backward compat):
      task_id, init_state_idx: used for HDF5 filename `task_{task_id}/episode_{init_state_idx}.h5`
      orig_init_state_idx: written as HDF5 attr for Step 1a/1b cross-reference
      episode_id: written as HDF5 attr (matches client.episode_start global episode_id)
    """
    traj_buffer: list[dict] = []
    # ... 原逻辑 ...
    # （§4 改动 2：每步 append step_record 到 traj_buffer）
    # ... 结束循环后 ...
    if args.save_trajectory and traj_buffer and task_id is not None and init_state_idx is not None:
        out_path = Path(args.save_trajectory_dir) / f"task_{task_id}" / f"episode_{init_state_idx}.h5"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out_path, "w") as f:
            f.attrs["task_name"] = task_description
            f.attrs["task_id"] = task_id
            f.attrs["init_state_idx"] = init_state_idx
            if orig_init_state_idx is not None:
                f.attrs["orig_init_state_idx"] = orig_init_state_idx
            if episode_id is not None:
                f.attrs["episode_id"] = episode_id
            f.attrs["seed"] = args.seed
            f.attrs["num_steps"] = len(traj_buffer)
            f.attrs["success"] = success
            f.attrs["prompt"] = str(task_description)
            for i, step in enumerate(traj_buffer):
                g = f.create_group(f"step_{i:04d}")
                for k, v in step.items():
                    g.create_dataset(k, data=v)
    return success, video_frames
```

#### 18.B1.3 Serial 调用点（`main.py:165-215`，`_eval_serial`）

锚点 `L185` 附近（原调用）：

```python
# BEFORE
success, video_frames = _run_episode(
    env, client, initial_state, task_description, args, max_steps,
    record_video=record_video, step_callback=step_callback,
)

# AFTER
success, video_frames = _run_episode(
    env, client, initial_state, task_description, args, max_steps,
    record_video=record_video, step_callback=step_callback,
    task_id=task_id,
    init_state_idx=episode_idx,
    orig_init_state_idx=orig_idx_from_filter,   # 见下文过滤逻辑
    episode_id=global_episode_id,
)
```

其中 `task_id` 在 `for task_id, task_suite in enumerate(task_suites):` 的 loop 里可直接取到；`episode_idx` 是内层 loop 的 counter。`global_episode_id` 已在原代码 L205 `global_episode_id += 1`。

`orig_idx_from_filter` 的获取：
- 如果 `args.episode_filter` 指定了 `[{task_id, init_state_idx (orig), subset_pos}]`，则 `orig_idx_from_filter = filter_entry["init_state_idx"]`。
- 否则 `orig_idx_from_filter = episode_idx`（无 subset，原始 index 就是当前 index）。

#### 18.B1.4 Concurrent 调用点（`main.py:218-380`，`_eval_concurrent`）

锚点 `L317-322`（worker 内部 `_run_episode` 调用）：

```python
# BEFORE
success, _ = _run_episode(
    env, client, initial_state, task_description, args, max_steps,
    record_video=False, step_callback=None,
)

# AFTER
success, _ = _run_episode(
    env, client, initial_state, task_description, args, max_steps,
    record_video=False, step_callback=None,
    task_id=task_id,
    init_state_idx=episode_idx,
    orig_init_state_idx=orig_idx,
    episode_id=global_episode_id,
)
```

`orig_idx` 来自 worker args tuple；`global_episode_id = task_id * args.num_trials_per_task + episode_idx`（L291 既有公式）。

#### 18.B1.5 episode_filter 精确格式（修正 §4 改动 3）

```json
[
  {"task_id": 3, "init_state_idx": 7},
  {"task_id": 3, "init_state_idx": 15},
  {"task_id": 5, "init_state_idx": 2}
]
```

上面"init_state_idx" 是**原 benchmark 的 index（0..49）**，不是 subset 内的位置。`main.py` 读取 filter 后：

```python
# main.py 顶部读 filter
filter_pairs: list[tuple[int, int]] | None = None
if args.episode_filter:
    pairs = json.loads(Path(args.episode_filter).read_text())
    filter_pairs = [(p["task_id"], p["init_state_idx"]) for p in pairs]

# 在 task loop 内部，episode_idx 是 subset 内的位置（0..K-1）
# 若用 --init-states-dir，subset_orig_idx 需要从 `.init_map.json` 旁表查找
# 或：直接用 `.init` 文件名 + 旁表 `{task.name}.init_map.json` 存 [orig_0, orig_1, ...]

orig_idx = subset_orig_map[task.name][episode_idx]
if filter_pairs is not None and (task_id, orig_idx) not in filter_pairs:
    continue
```

**新增 artifact**：`scripts/dump_step1a_failed_inits.py` 除了写 `{task.name}.init`，还要写 `{task.name}.init_map.json = [orig_0, orig_1, ...]`，供 main.py 反查。

```python
# scripts/dump_step1a_failed_inits.py 追加
(out / f"{task.name}.init_map.json").write_text(json.dumps(sorted(bad_indices)))
```

main.py 里 `_load_init_states`（`L461-477`）同时读 map 并存到 `subset_orig_map[task.name]`：

```python
# examples/libero/main.py, _load_init_states 改造
def _load_init_states(task, task_suite, task_id, init_states_dir):
    # ... 原 torch.load 逻辑 ...
    map_path = Path(init_states_dir) / f"{task.name}.init_map.json"
    if map_path.exists():
        orig_map = json.loads(map_path.read_text())
    else:
        orig_map = list(range(len(init_states)))
    return init_states, orig_map
```

---

### 18.B2 BaseRunState 线程安全（修正 §2.1）

**背景**：`BaseRunState.save()` 每次全量重写 JSON 文件。`parallel_run` 多个 worker 可能同时在 `_execute_one` 里调用 `save()`，导致 JSON 损坏。

#### 18.B2.1 加锁方案

```python
# exp/_run_state_base.py
import threading

class BaseRunState(ABC):
    def __init__(self, state_path: Path, *, max_retries: int = 2):
        self.state_path = Path(state_path)
        self.max_retries = max_retries
        self.units: dict[str, UnitState] = {}
        self._save_lock = threading.Lock()   # NEW

    def save(self) -> None:
        with self._save_lock:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            tmp_path.write_text(
                json.dumps({k: asdict(u) for k, u in self.units.items()}, indent=2)
            )
            tmp_path.replace(self.state_path)   # atomic on POSIX

    def _execute_one(self, u: UnitState) -> None:
        # unit-level status writes are protected by save() lock too
        u.status = "running"
        u.start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save()
        try:
            result = self.execute_unit(u)
            u.result = result
            u.status = "done"
        except Exception as e:
            u.status = "failed"
            u.result = {"error": str(e)}
        u.end_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save()
```

**两层保护**：
1. `threading.Lock` 确保同一进程内并发 save 串行化（避免 JSON corruption）。
2. `tmp + rename` 原子替换确保读端（手动 tail state JSON 时 / 下次 `--resume` load）不会读到半写文件。

**注意**：对 `self.units` 的**内容修改**（如 `u.status = ...`）本身不加锁 —— 依赖 `execute_unit` 里一个 worker 只改自己的 unit，不会 cross-worker 冲突。如果将来 `execute_unit` 需要读其他 unit 的状态，再加 `self._units_lock`。

---

### 18.B3 pdist 优化（修正 §10.2）

**背景**：§10.2 的 aggregate 内循环对每个 `t` 做 O(M²) 两两 L2，对 M=20 是 190 pairs × T 步，数值上 OK；但写法 `v[:, None, :] - v[None, :, :]` 分配 (M,M,Ad) 临时张量，内存不友好（M=20 时 3.2 KB × T 次 —— 仍小，但可预防性优化）。用 `scipy.spatial.distance.pdist` 更干净：

```python
# exp/compute_deviate_scores.py 里 aggregate 的 bg_l2 循环
from scipy.spatial.distance import pdist

bg_l2 = []
for t in range(T):
    v = bg_stack[:, t, :]          # (M, Ad)
    # pdist 直接返回 C(M,2) 长度的向量，内存 O(M²) 但不分配 (M,M,Ad) 临时张量
    d = pdist(v, metric="euclidean")   # shape (M*(M-1)//2,)
    bg_l2.append(float(d.mean()))
```

**等价性验证**（单元测试）：

```python
def test_pdist_vs_pairwise():
    rng = np.random.default_rng(0)
    v = rng.standard_normal((5, 7))
    # naive
    d_naive = np.linalg.norm(v[:, None, :] - v[None, :, :], axis=-1)
    naive_mean = d_naive[np.triu_indices(5, k=1)].mean()
    # pdist
    pdist_mean = pdist(v, metric="euclidean").mean()
    assert abs(naive_mean - pdist_mean) < 1e-12
```

**进一步**：若 T 很大（>1000）且 M=20，可把整个 batch 放 GPU 算：

```python
import torch
v = torch.from_numpy(bg_stack).cuda()   # (T, M, Ad) — transpose first
# pairwise dist per t along dim=1
d_matrix = torch.cdist(v, v)    # (T, M, M)
iu = torch.triu_indices(M, M, offset=1)
bg_l2 = d_matrix[:, iu[0], iu[1]].mean(dim=-1).cpu().tolist()
```

两种写法都保留；默认用 pdist（CPU, 依赖少），GPU 版留注释里作为备用。

---

### 18.B4 Step 1b 共享 GT 的可行性分析（修正 §16 #3）

**背景**：§16 #3 遗留问题。读 `InferenceInterceptor`（`interceptor.py:301-485`）和 `CollectionPolicy`（`collection_policy.py:42-124`）后，补完整分析。

#### 18.B4.1 "三个 config 共享一套 GT" 是否合法？

**核心问题**：三个 config 的 `key_builder.type`（clip / spatial16 / max_pool）对 obs 的处理方式不同，它们会**改变什么**？

读 `src/openpi/cache/components/key_builder.py` 和 `src/openpi/cache/orchestrator.py` 后：

| 组件 | 与 key_builder 相关？ | 是否影响模型 forward 的结果？ |
|------|---------------------|---------------------------|
| `key_builder.collect(stage1_out)` | 是 | 否 —— 只是旁路计算 query_key，不改 stage1_out |
| `key_builder.build(collected)` → query_keys | 是 | 否 |
| `search_strategy.search(query_keys, trajectory_history)` | 是 | 否（若 gate=always_skip，search 也不被调用） |
| `judge(score)` | 否 | 否 |
| `orchestrator.broadcast_action(action)` → 各组件 `record_action` | 是 | **否**（record 只是更新 buffer，不回写模型输入） |
| `stage1/stage2/stage3 forward` | **否** | 是 |

**结论**：key_builder 是完全**旁路**的 —— 它消费 stage1 输出、产生 query_key，但不反向修改 forward 路径。只要 **noise seed 和模型权重固定**，三个 config 在 AlwaysSkip 模式下产出的 `action_chunk` **完全相同**。

#### 18.B4.2 Deterministic 条件

需要满足：
1. `_run_stage3(stage2_out, noise, ...)` 里的 `noise` 是 seed-controlled；
2. 模型权重 fixed；
3. obs 序列 fixed。

Interceptor 的 noise 来源（L408-412）：

```python
# src/openpi/cache/interceptor.py:408-412
if noise is None:
    noise = _sample_noise(stage2_out.shape, device=stage2_out.device, seed=???)
```

读 `_sample_noise`：seed 来自 orchestrator 的 `_step_counter + episode_id` 或 torch global seed。在 server 单 connection 生命周期内，`_step_counter` 是单调递增 → noise 对每个 step 可复现。**但 cross-connection 的不同 worker 可能拿到不同 torch rng state**。为了严格 bit-wise 可复现：

- Server 启动时 `torch.manual_seed(42)`（`websocket_policy_server.py` main 加一行）。
- 每个 connection policy 初始化后 `torch.manual_seed(episode_id)`（在 `on_episode_start` 里加）。

经验上不需要这么严格 —— 我们只是要求"在同一 obs 序列上 inference 结果**语义一致**"，不是 bit-wise。

#### 18.B4.3 实际决策（覆盖 §16 #3 的默认）

- **Step 1b 只用 `inference_clip_w7_d4.yaml` 收一次 GT**（clip 是三个 config 里最标准的 key_builder）。
- **Step 2 Phase 1/2 在三个 config 之间切换**（`send_load_cache_config`）时，**复用 Step 1b 的 GT**（即 `load_gt_episode` 读同一 HDF5 dir）。
- **Step 3 Spawn 同理**：teleport 时读同一 GT HDF5。

**若发现 bug**（三个 config 跑出的 cache-vs-GT 差异不合理，怀疑是 config 特异的行为漂移），回退到"每个 config 一套 GT"。

#### 18.B4.4 对 Step 1b runner 代码的影响（修正 §9.2）

`Step1bRunner` 只需跑一次，YAML 固定 `inference_clip_w7_d4.yaml`：

```python
# exp/run_step1b_gt.py main
def main():
    args = parse_args()
    # 固定使用 clip 配置做 GT 收集
    send_load_cache_config(server_url, "configs/cache_runs/deviate_exp/inference_clip_w7_d4.yaml")
    runner = Step1bRunner(..., inference_yaml="inference_clip_w7_d4")
    runner.run(resume=args.resume)
```

GT 存在 `data/deviation_experiment/gt_trajectories/`（单目录，不按 config 分）。

---

### 18.B5 run_cache_experiments.py 与 BaseRunState 的 retry 模式冲突（修正 §9.0）

**背景**：`run_cache_experiments.py:583-736` 的 `_retry_failed_runs` 会把失败 YAML 复制到 `{config_dir}/retry/` 目录重跑 —— 这是"**per-YAML retry dir**"模式。`BaseRunState`（§2.1）用"**就地翻 status=pending 重跑**"模式 —— 这是"**in-place retry**"模式。两种模式**不能混用**：前者产生 retry/ 子目录和新 state 文件，后者在原 state 里改 status。

#### 18.B5.1 决定：保留双轨，明确边界

| Runner | retry 模式 | 原因 |
|--------|-----------|------|
| `run_cache_experiments.py` | per-YAML retry dir（既有） | YAML 是自然 unit，retry/ 目录有调试价值（隔离失败配置，方便 inspect log） |
| `run_step1b_gt.py` | in-place（BaseRunState） | Unit 是 (task_id, init_idx) tuple，不是 YAML —— 没有 "retry YAML dir" 概念 |
| `compute_deviate_scores.py` | in-place | 同上，unit = (cfg, ep, s) |
| `run_spawn_experiment.py` | in-place | 同上 |

#### 18.B5.2 `run_cache_experiments.py` 的最小必要改动（§9.0 修正）

**不迁移到 BaseRunState**（原 `RunState` 和 retry 逻辑继续用）。只做两件事：

1. **`main_args` 追加 `--save-episode-results --episode-results-path ...`**（§9.0 原改动 2 不变）。
2. **新增 post-run 聚合**：所有 run 完成后（含 retry），扫所有 `.episode_results.json` 合并到 `cache_eval_results.json`：

```python
# exp/run_cache_experiments.py 追加（在 main 结尾）
def _aggregate_episode_results(run_root: Path, out_path: Path) -> None:
    """Merge all per-run .episode_results.json into a single file."""
    all_rows = []
    for p in run_root.rglob("*.episode_results.json"):
        rows = json.loads(p.read_text())
        # tag each row with source yaml path
        src_yaml = p.with_suffix("").with_suffix(".log.md")  # adjust to match naming
        for r in rows:
            r["source"] = str(p.relative_to(run_root))
        all_rows.extend(rows)
    out_path.write_text(json.dumps(all_rows, indent=2))

# 在 main 最后加：
_aggregate_episode_results(
    run_root=Path(args.config_dir),
    out_path=Path(args.config_dir) / "cache_eval_results.json",
)
```

这样 `scripts/dump_step1a_failed_inits.py` 读 `cache_eval_results.json` 就能拿到全部 per-episode 结果（含 retry 的最终成功/失败）。

#### 18.B5.3 BaseRunState 对 per-unit retry 的计数一致性

BaseRunState 的 `u.retry_count` 在每次 retry loop 递增（§2.1 L180）；Runner 可通过 `u.retry_count > 0` 区分"首次跑失败后重试的单元"。例如 `compute_deviate_scores.py` 可以在 jsonl 里加字段：

```python
{"config": cfg, "episode": ep, "sample_idx": s,
 "actions": [...],
 "retry_count": u.retry_count}
```

**告警逻辑**：如果某个 unit `retry_count == max_retries` 且仍 failed，main 结束时打印 ERROR 且退出码非零，避免 analyze 阶段误用不完整数据。

---

### 18.Z 索引更新

上面各条对原章节的修正汇总：

| A/B 条目 | 影响章节 | 修正方向 |
|---------|---------|--------|
| A1 | §7, §8.3 | infer() 路径图；inference_*.yaml 需 cp1+cp3 都 always_skip |
| A2 | §3.2, §5 | per-connection facade 精确到 config.py:755；server prefill 分支改用 `policy._orchestrator._storage` |
| A3 | §4 buffer, §10.4, §11 (_obs_env_to_policy) | GT HDF5 存 post-transform；replay 侧字节一致 |
| A4 | §3.5 (新增), §A4 详解 | CachePayload torch→numpy→msgpack 双向 |
| B1 | §4 | _run_episode 完整签名 + 所有调用点更新 + init_map.json |
| B2 | §2.1 | threading.Lock + tmp+rename 原子写 |
| B3 | §10.2 | pdist 替换双循环 |
| B4 | §16 #3, §9.2 | 共享 GT：clip 一套；需要时回退 |
| B5 | §9.0 | 双轨 retry 模式共存；run_cache_experiments.py 保留旧 retry dir |

---

## 19. G1 审查修订汇总（2026-04-14）

> 来源：`logs/trajectory_deviation_corrective_implementation_review.log.md` §3–§4 A/B 级问题
> 冲突解决规则：**§19 与 §1–§18 如有冲突，以 §19 为准**。Reviewer 验收以本节为基线。

### 19.1 Step 粒度 = inference cycle（修正 A1，替换 §4.1 改动 2 + §13.1 + §18.A3）

**决定**：client-side GT HDF5 每条 record = 一次 `client.infer()` 调用。不强制 `replan_steps=1`。

**`main.py::_run_episode` 改动**（替换 §18.A3.2 的 step_record）：

```python
# examples/libero/main.py L126-143 替换/增强
if not action_plan:
    element = {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": np.concatenate((
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )),
        "prompt": str(task_description),
    }
    action_chunk = client.infer(element)["actions"]   # shape (H, 7) env-space
    # NEW: record the inference-cycle BEFORE consuming actions
    if args.save_trajectory:
        cycle_record = {
            "sim_state": env.get_sim_state().copy(),
            "env_timestep": int(env.timestep),
            "env_cur_time": float(env.cur_time),
            "agentview_image": img.copy(),
            "eye_in_hand_image": wrist_img.copy(),
            "robot_state": element["observation/state"].astype(np.float64).copy(),
            "env_action_chunk": np.asarray(action_chunk, dtype=np.float32).copy(),  # (H, 7)
            "executed_actions": None,  # filled after replan_steps env.step calls
        }
        traj_buffer.append(cycle_record)
        _pending_executed: list[np.ndarray] = []
    action_plan.extend(action_chunk[: args.replan_steps])

action = action_plan.popleft()
if args.save_trajectory:
    _pending_executed.append(np.asarray(action, dtype=np.float32).copy())
    # when action_plan drained, finalize the cycle_record.executed_actions
    if not action_plan:
        traj_buffer[-1]["executed_actions"] = np.stack(_pending_executed)
obs, reward, done, info = env.step(action.tolist())
```

Episode 结束 flush 到 HDF5：仍按"每条 record 一个 `step_{i:04d}` group"，但 `i` 现在是 **inference-cycle index**，不是 env step index。`num_cycles = len(traj_buffer)`。

**§13.1 HDF5 schema 更新**：

```
attrs:
  task_name, task_id, init_state_idx, orig_init_state_idx, episode_id, seed
  num_cycles: int          # ← 原 num_steps 改名
  num_env_steps: int        # 总 env.step 次数 = sum(replan_steps_per_cycle)
  replan_steps: int         # args.replan_steps 值（便于解析 executed_actions shape）
  success: bool, prompt: str
step_{i:04d}/:              # i = inference cycle index (0..num_cycles-1)
  sim_state, env_timestep, env_cur_time
  agentview_image (224,224,3) uint8
  eye_in_hand_image (224,224,3) uint8
  robot_state (8,) float64
  env_action_chunk (H, 7) float32     # client response, pre-consumption
  executed_actions (replan_steps, 7) float32   # chunk 前 replan_steps 个 action 实际执行
```

**Step 2/3 对齐**：
- `compute_deviate_scores.py::load_gt_episode` 读 `step_{i}/agentview_image+wrist+state+prompt` 构造 `obs_seq`；T=num_cycles。
- Step 3 teleport 发生在某 inference cycle 的开头（`s+n` 是 cycle index 而非 env step index）；prefill 的 D-1 个历史 obs 都是 cycle-level obs。

### 19.2 Lifecycle 透明转发（修正 A2，替换 §3.3）

**`src/openpi/collect/collection_policy.py` 修订**：

```python
def on_episode_start(
    self,
    experiment: str,
    task: str,
    episode_id: int,
    episode_name: str = "",           # NEW
) -> None:
    self._collector.on_episode_start(
        experiment, task, episode_id, episode_name=episode_name
    )
    self._collecting = True
    self._prompt_captured = False
    # NEW: forward to inner policy (InferenceInterceptor or PolicyRecorder)
    if hasattr(self._policy, "on_episode_start"):
        # call with kwargs that inner policy accepts; inner supports **kwargs
        # or explicit episode_name=episode_name per §19.2.2
        self._policy.on_episode_start(
            experiment=experiment, task=task,
            episode_id=episode_id, episode_name=episode_name,
        )

def on_episode_end(self, success: bool) -> None:
    self._collector.on_episode_end(success)
    self._collecting = False
    if hasattr(self._policy, "on_episode_end"):
        self._policy.on_episode_end(success)
```

**`InferenceInterceptor.on_episode_start` 接受 `episode_name` kwarg**（默认 `""` 保持 backward compat）：

```python
# src/openpi/cache/interceptor.py L270 附近
def on_episode_start(self, experiment: str = "", task: str = "", episode_id: int = -1,
                    episode_name: str = "") -> None:
    del episode_name  # 暂不使用；保留形参避免 TypeError（未来若 interceptor 要按 episode_name dump debug，可启用）
    self._orchestrator.on_episode_start()
    # ... 原有逻辑 ...
```

**测试**：新增 `tests/test_collect_cache_lifecycle.py`：
1. wrap `InferenceInterceptor` with `CollectionPolicy` 跑两个 episode；
2. 断言 episode 2 开头 `orchestrator._step_counter == 0`、`search_strategy._query_history` 长度 0；
3. 断言 `--collect` 关时，server 传 `episode_name` 给 `InferenceInterceptor.on_episode_start` 不抛 TypeError。

### 19.3 Per-connection facade 接到 search_strategy（修正 A3，替换 §5 + §18.A2.2）

**`src/openpi/cache/config.py::build_per_connection_components` 精确改动**：

```python
def build_per_connection_components(config, shared_storage, *, quiet=False):
    from openpi.cache.cache_storage import CacheStorage
    from openpi.cache.timing import SystemTimer

    timer = SystemTimer(...)
    enabled_fields = [name for name, kf in _keys_iter(config.keys) if kf.enabled]
    key_builder = _build_key_builder(config.key_builder, enabled_fields, config.backend.vector_dims)

    # --- NEW: fresh per-connection facade (must happen BEFORE search_strategy build) ---
    per_conn_storage = CacheStorage(
        backend=shared_storage._backend,
        metadata_db=getattr(shared_storage, "_metadata_db", None),
    )

    fusion_weights = {name: kf.weight for name, kf in _keys_iter(config.keys) if kf.enabled}
    gates: dict[CheckpointID, Any] = {}
    judges: dict[CheckpointID, Any] = {}
    search_strategies: dict[CheckpointID, Any] = {}
    for cp_name, cp_config in config.checkpoints.items():
        if cp_name.startswith("_") or not cp_config.enabled:
            continue
        cp_id = CheckpointID[cp_name.upper()]
        gates[cp_id] = _build_gate(cp_config.gate)
        judges[cp_id] = _build_judge(cp_config.judge)
        search_strategies[cp_id] = _build_search_strategy(
            cp_config.search_strategy, per_conn_storage, fusion_weights    # ← per_conn_storage
        )

    write_policy = _build_write_policy(config.write_policy)
    return {
        "timer": timer,
        "storage": per_conn_storage,           # ← per_conn_storage
        "key_builder": key_builder,
        "gates": gates,
        "judges": judges,
        "search_strategies": search_strategies,
        "write_policy": write_policy,
    }
```

**测试**：新增 `tests/test_per_connection_prefill.py`：
1. Build bundle；
2. 断言 `orch._storage is search_strategies[CP1]._storage`；
3. 两并发 bundle，conn A 进 prefill_mode，conn B `search()` 仍走 backend（非 synthetic hit）。

**interceptor.py 可选 property 简化访问**：

```python
# src/openpi/cache/interceptor.py InferenceInterceptor 类内
@property
def cache_storage(self):
    """Per-connection CacheStorage facade (prefill-mode toggles are connection-local)."""
    return self._orchestrator._storage
```

### 19.4 Prefill payload 使用 model-space clean_action（修正 A4，替换 §7 + §13 schema）

**Step 1b 强制开 `--collect`**。server-side HDF5（`data/deviation_experiment/collected/libero_xxx/task_X/episode_Y.h5`）与 client-side GT HDF5（`data/deviation_experiment/gt_trajectories/task_X/episode_Y.h5`）按 `episode_name="task_X/episode_Y"` 对齐。

**Server-side HDF5 schema**（既有，confirmed 对本实验足够）：

```
step_{i:04d}/:   # i = inference cycle index
  vision_emb: (num_tokens, dim)       # 未必用，留存
  prompt_emb: (seq_len, dim)          # 未必用
  clean_action: (H, model_action_dim) float32    # ← prefill payload 用这个
  noise_action_steps: ...             # 未必用
  robot_state: ...
```

**`exp/run_spawn_experiment.py::execute_unit` 改动**（替换 §11.1 Step 4 的 prefill 调用）：

```python
# 4. open connection and prefill with SERVER-SIDE model-space actions
collected_path = Path(args.collected_dir) / f"{ep}.h5"   # episode_name-aligned
with h5py.File(collected_path, "r") as sf:
    prefill_actions = []
    for t in range(prefill_start, s + n):
        prefill_actions.append(
            np.asarray(sf[f"step_{t:04d}/clean_action"], dtype=np.float32)
        )   # each shape (H, model_action_dim)

# prefill_obs 仍来自 client-side GT HDF5（env-space obs + prompt）
client.prefill_trajectory(
    observations=prefill_obs,
    actions=prefill_actions,    # list[ np.ndarray (H, model_action_dim) ]
)
```

**`InferenceInterceptor._build_prefill_payload` 改动**：

```python
@staticmethod
def _build_prefill_payload(action) -> "CachePayload":
    """action: np.ndarray or torch.Tensor, shape (H, model_action_dim). Model-space."""
    import torch, numpy as np
    from openpi.cache.storage_types import CachePayload
    if isinstance(action, np.ndarray):
        action_t = torch.from_numpy(np.ascontiguousarray(action, dtype=np.float32))
    else:
        action_t = action.detach().cpu().contiguous().float()
    # No shape truncation — contract is model-space H × model_action_dim
    return CachePayload(
        action_chunk=action_t,
        task_key=None,
        intermediates=None,
        denoising_num_steps=None,
    )
```

**§7 的 `prefill_trajectory` docstring 补一行**：

```python
"""
actions: list[np.ndarray | torch.Tensor], each shape (horizon, model_action_dim).
         Must be MODEL-space (i.e., post-flow-matching clean_action from
         server-side --collect HDF5 `step_{i}/clean_action`). Env-space
         actions (7D for LIBERO) are NOT accepted.
"""
```

### 19.5 L2 metric = chunk 第一个 action（修正 A5，替换 §10.1 + §10.2）

**Phase 1 runner**：

```python
# exp/compute_deviate_scores.py Phase1Runner.execute_unit
actions_first = []   # 只保留 chunk 第一个 action，shape 最终 (T, 7)
for obs in obs_seq:
    resp = client.infer(obs)
    act = np.asarray(resp["actions"], dtype=np.float32)    # (H, 7)
    actions_first.append(act[0])                           # (7,)
actions_first = np.stack(actions_first)                    # (T, 7)

# per-unit npz（A7 改动，见 §19.7）
out_path = Path(self.out_dir) / "bg" / cfg / ep / f"s{s:03d}.npz"
out_path.parent.mkdir(parents=True, exist_ok=True)
np.savez(out_path, actions_first=actions_first, T=len(actions_first), retry_count=u.retry_count)
```

**Phase 2 runner** 同理，不加 `s` 维度：`.../cache/{cfg}/{ep}.npz`。

**aggregate（§10.2 修订）**：

```python
def aggregate(bg_dir, cache_dir, gt_dir, cfg, out_path):
    from scipy.spatial.distance import pdist
    # Discover episodes from per-unit files
    eps = sorted({p.parent.name for p in (bg_dir / cfg).glob("*/*.npz")})
    out = {}
    for ep in eps:
        bg_samples = sorted((bg_dir / cfg / ep).glob("s*.npz"))
        bg_stack = np.stack([np.load(p)["actions_first"] for p in bg_samples])  # (M, T, 7)
        cache_arr = np.load(cache_dir / cfg / f"{ep}.npz")["actions_first"]     # (T, 7)
        gt_first = load_gt_first_actions(gt_dir, ep)                            # (T, 7)
        M, T, _ = bg_stack.shape

        bg_l2 = []
        for t in range(T):
            v = bg_stack[:, t, :]                   # (M, 7)
            d = pdist(v, metric="euclidean")        # (M*(M-1)//2,)
            bg_l2.append(float(d.mean()))
        cache_l2 = np.linalg.norm(cache_arr - gt_first, axis=-1).tolist()  # (T,)

        floor = 0.1
        dev = [cl / max(bl, floor) for cl, bl in zip(cache_l2, bg_l2)]
        out[ep] = {"background_l2": bg_l2, "cache_l2": cache_l2, "deviate_score": dev}
    Path(out_path).write_text(json.dumps(out, indent=2))


def load_gt_first_actions(gt_dir, ep):
    with h5py.File(Path(gt_dir) / f"{ep}.h5", "r") as f:
        T = int(f.attrs["num_cycles"])
        # GT "action" = executed_actions[0] == env_action_chunk[0]（前者是 chunk 前 replan_steps 个的第一个 = chunk[0]）
        return np.stack([f[f"step_{t:04d}/executed_actions"][0] for t in range(T)])
```

### 19.6 Phase 0 新增 obs 一致性脚本（修正 A6）

**`scripts/verify_restore_obs_equivalence.py`**（新增）：

```python
"""Verify: restoring env from a GT HDF5's sim_state reproduces the same
post-transform obs as stored in the HDF5. Must pass before Step 3 runs.
"""
from pathlib import Path
import numpy as np, h5py
from libero.libero.benchmark import get_benchmark_dict
from libero.libero.envs import OffScreenRenderEnv
import sys
sys.path.insert(0, "examples/libero")
from main import _quat2axisangle, LIBERO_ENV_RESOLUTION
sys.path.insert(0, "exp")
from run_spawn_experiment import _obs_env_to_policy

def verify(gt_h5_path: Path, task_suite: str, task_id: int, cycle_idx: int):
    with h5py.File(gt_h5_path, "r") as f:
        sim_state = f[f"step_{cycle_idx:04d}/sim_state"][...]
        ref_img = f[f"step_{cycle_idx:04d}/agentview_image"][...]
        ref_wrist = f[f"step_{cycle_idx:04d}/eye_in_hand_image"][...]
        ref_state = f[f"step_{cycle_idx:04d}/robot_state"][...]
        task_name = str(f.attrs["task_name"])

    suite = get_benchmark_dict()[task_suite]()
    task = suite.get_task(task_id)
    env = OffScreenRenderEnv(bddl_file_name=task.bddl_file,
                             camera_heights=LIBERO_ENV_RESOLUTION,
                             camera_widths=LIBERO_ENV_RESOLUTION)
    env.reset()
    raw_obs = env.set_init_state(sim_state)   # returns obs dict for robosuite
    policy_obs = _obs_env_to_policy(raw_obs, task_name)

    assert np.array_equal(policy_obs["observation/image"], ref_img), "agentview mismatch"
    assert np.array_equal(policy_obs["observation/wrist_image"], ref_wrist), "wrist mismatch"
    assert np.allclose(policy_obs["observation/state"], ref_state, atol=1e-6), \
        f"state mismatch max={np.abs(policy_obs['observation/state']-ref_state).max()}"
    print(f"OK: {gt_h5_path} cycle {cycle_idx}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-h5", required=True)
    ap.add_argument("--task-suite", required=True)
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--cycle-idx", type=int, default=0)
    args = ap.parse_args()
    verify(Path(args.gt_h5), args.task_suite, args.task_id, args.cycle_idx)
```

**退出条件**：若任一 assert 失败 → Step 3 **禁止** 从 HDF5 读 `start_obs`；改为 restore env 后用 `_obs_env_to_policy(env.set_init_state(sim_state))`。若全部通过，继续使用 HDF5 读法（更快，避免重复 env 初始化）。

### 19.7 Per-unit 文件替代并行 append（修正 A7）

**目录布局**：

```
data/deviation_experiment/
├── gt_trajectories/task_{id}/episode_{idx}.h5   # client-side GT (A1, A3, A4)
├── collected/{task_suite}/task_{id}/episode_{idx}.h5   # server --collect (A4)
├── step2/
│   ├── bg/{cfg}/{ep}/s{sample:03d}.npz          # Phase 1 bg, 每 worker 一个文件
│   └── cache/{cfg}/{ep}.npz                     # Phase 2 cache, 每 episode 一个
├── step3/{cfg}/{ep}__s{s:03d}_n{n:02d}_k{k:02d}.npz   # spawn per-unit
├── deviate_score_{cfg}.json                      # Phase 3 aggregate
└── spawn_aggregate.csv                           # Step 3 aggregate
```

**BaseRunState.save()** 的原子写（§18.B2 已有）继续保留，覆盖 state JSON 的并发安全；worker 写的 per-unit 文件天然无冲突。

**Aggregate 改 glob 扫描**（代码见 §19.5）。jsonl 方案完全废弃。

### 19.8 verify_env_save_restore.py 重写（修正 A8）

替换 §11.0 的脚本：

```python
# scripts/verify_env_save_restore.py
"""Phase 0: verify env.get_sim_state / set_init_state round-trip fidelity."""
import numpy as np
from libero.libero.benchmark import get_benchmark_dict
from libero.libero.envs import OffScreenRenderEnv

def main():
    suite = get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    env = OffScreenRenderEnv(bddl_file_name=task.bddl_file,
                             camera_heights=128, camera_widths=128)
    env.reset()
    init_state = suite.get_task_init_states(0)[0]
    env.set_init_state(init_state)

    # --- Pre-actions: drive env to a non-trivial mid-episode state ---
    rng = np.random.default_rng(0)
    pre_actions = [rng.uniform(-0.1, 0.1, 7) for _ in range(30)]
    for a in pre_actions:
        env.step(a.tolist())
    ckpt_sim = env.get_sim_state().copy()
    ckpt_timestep = int(env.timestep)
    ckpt_cur_time = float(env.cur_time)

    # --- Reference branch: continue from ckpt with post_actions ---
    post_actions = [rng.uniform(-0.1, 0.1, 7) for _ in range(25)]
    traj_ref_eef = []
    traj_ref_success = []
    for a in post_actions:
        obs, _, _, info = env.step(a.tolist())
        traj_ref_eef.append(obs["robot0_eef_pos"].copy())
        traj_ref_success.append(bool(info.get("success", False)))

    # --- Replay branch: restore to ckpt and replay post_actions ---
    env.set_init_state(ckpt_sim)
    env.timestep = ckpt_timestep
    env.cur_time = ckpt_cur_time
    traj_replay_eef = []
    traj_replay_success = []
    for a in post_actions:
        obs, _, _, info = env.step(a.tolist())
        traj_replay_eef.append(obs["robot0_eef_pos"].copy())
        traj_replay_success.append(bool(info.get("success", False)))

    # --- Assertions ---
    ref_arr = np.stack(traj_ref_eef)
    replay_arr = np.stack(traj_replay_eef)
    max_err = float(np.abs(ref_arr - replay_arr).max())
    print(f"max eef_pos delta: {max_err:.3e}")
    assert np.allclose(ref_arr, replay_arr, atol=1e-6), (
        f"restore fidelity failed: max_delta={max_err}"
    )
    assert traj_ref_success == traj_replay_success, "success flag drifted"
    assert env.timestep - ckpt_timestep == len(post_actions), "timestep bookkeeping wrong"
    print("restore fidelity: OK")

if __name__ == "__main__":
    main()
```

### 19.B1 episode_name 校验（修正 B1）

**`src/openpi/collect/data_collector.py`**：

```python
from pathlib import PurePosixPath

def _validate_episode_name(name: str) -> str:
    """Return sanitized name or raise ValueError."""
    if not name:
        return ""
    if len(name) > 200:
        raise ValueError(f"episode_name too long (>200): {name!r}")
    p = PurePosixPath(name)
    if p.is_absolute():
        raise ValueError(f"episode_name must not be absolute: {name!r}")
    if any(part in ("", "..", ".") for part in p.parts):
        raise ValueError(f"episode_name has invalid path parts: {name!r}")
    if p.suffix:
        raise ValueError(f"episode_name must not carry suffix (.h5 is server-added): {name!r}")
    return name

def on_episode_start(self, experiment, task, episode_id, *, episode_name=""):
    episode_name = _validate_episode_name(episode_name)
    with self._lock:
        self._episode_name = episode_name
        # ... 原有 buffer / attrs 初始化 ...
```

Server 侧文件名构造已在 §3.4 改动 3 加 `.h5`，保持不变。

### 19.B2 YAML 命名不产生重复（修正 B2）

**现有 3 个 YAML 直接复用为 cache YAML**（不新建 `cache_*` 前缀）：
- `configs/cache_runs/deviate_exp/clip_w7_d4.yaml`（已存在）
- `configs/cache_runs/deviate_exp/spatial16_w8_d4.yaml`（已存在）
- `configs/cache_runs/deviate_exp/max_pool_w3_d5.yaml`（已存在）

**新建 3 个 inference YAML**：
- `configs/cache_runs/deviate_exp/inference_clip_w7_d4.yaml`
- `configs/cache_runs/deviate_exp/inference_spatial16_w8_d4.yaml`
- `configs/cache_runs/deviate_exp/inference_max_pool_w3_d5.yaml`

每份 inference YAML 相对对应 cache YAML 的 diff（替换 §8.3 的模板）：

```yaml
# diff vs. clip_w7_d4.yaml
checkpoints:
  cp1:
    gate:
      type: always_skip           # ← 唯一 gate 差异（CP3 处理见 19.B3）
# top-level 新增
write_policy:
  type: never                      # ← B8
```

**全文档 "cache_clip_w7_d4.yaml" / "cache_spatial16_w8_d4.yaml" / "cache_max_pool_w3_d5.yaml" 字样视为 `clip_w7_d4.yaml` / `spatial16_w8_d4.yaml` / `max_pool_w3_d5.yaml` 的别名**（§19 为准）。

### 19.B3 CP3 always_skip 条件化（修正 B3）

**规则**（替换 §18.A1.3 的绝对断言）：
- 若 inference YAML `cp3.enabled: false`（或整块未配置）：**无需写 cp3 gate**，因为 `orchestrator.check(CP3)` 不会被触发（`build_per_connection_components` 在 `cp_name starts with "_" or not cp_config.enabled` 时跳过）。
- 若 inference YAML `cp3.enabled: true`：必须写 `cp3.gate.type: always_skip`，否则 CP3 的 cache 命中会破坏 Phase 1 M 个 background sample 的独立性。

**首版默认**：inference_*.yaml **只配 cp1**（不启用 cp3）。pilot 阶段若发现 cp3 cache 对 baseline 有显著贡献，再启用并配 always_skip。

### 19.B4 Client 连接生命周期严格化（修正 B4）

**`exp/compute_deviate_scores.py::main` 与 `exp/run_spawn_experiment.py::main` 切换 config 时的硬约束**：

```python
for cfg in args.configs:
    # 1. Make sure previous batch workers are all done + closed
    #    (ThreadPoolExecutor.__exit__ 会 wait=True; worker 的 client 已在 finally 中 close)

    # 2. Switch server bundle via admin connection
    version = send_load_cache_config(server_url, f".../inference_{cfg}.yaml")
    logger.info(f"server bundle v{version} active")

    # 3. Spawn a fresh pool (each worker opens its own client)
    runner = Phase1Runner(config_id=cfg, server_url=server_url, ...)
    runner.parallel_run(num_workers=args.num_workers, resume=args.resume)
```

**Worker execute_unit 必须用 context manager**（需给 `WebsocketClientPolicy` 加 `__enter__/__exit__` 若目前缺）：

```python
def execute_unit(self, u):
    ...
    with WebsocketClientPolicy(self.host, self.port) as client:
        client.episode_start(...)
        try:
            # ... infer loop ...
        finally:
            client.episode_end(success=...)
    # __exit__ 关闭 websocket，backend 引用计数立即下降
```

若 `WebsocketClientPolicy` 尚无 `close()`，新增：

```python
# packages/openpi-client/src/openpi_client/websocket_client_policy.py
def __enter__(self): return self
def __exit__(self, *exc):
    try: self._ws.close()
    except Exception: pass
def close(self): self.__exit__(None, None, None)
```

### 19.B5 cache_eval_results.json 去重 + 溯源（修正 B5）

**`exp/run_cache_experiments.py::_aggregate_episode_results`** 改写：

```python
def _aggregate_episode_results(run_root: Path, out_path: Path) -> None:
    """Merge all per-run .episode_results.json with dedup by (config_id, task_id,
    init_state_idx, seed). Later attempt overrides earlier."""
    rows = []
    for p in sorted(run_root.rglob("*.episode_results.json")):
        # Infer attempt number from path: retry/ subdirs count as higher attempt
        attempt = str(p).count("/retry/")
        source = p.relative_to(run_root).as_posix()
        run_id = p.parent.name
        for r in json.loads(p.read_text()):
            r.update({
                "attempt": attempt,
                "source_path": source,
                "run_id": run_id,
                "config_id": r.get("config_id") or _infer_config_id_from_path(source),
            })
            rows.append(r)

    # Dedup: sort by (key, attempt) and keep last (highest attempt)
    def key(r):
        return (r["config_id"], r["task_id"], r["init_state_idx"], r.get("seed", -1))
    rows.sort(key=lambda r: (key(r), r["attempt"]))
    deduped: dict[tuple, dict] = {}
    for r in rows:
        deduped[key(r)] = r
    out_path.write_text(json.dumps(list(deduped.values()), indent=2))
```

**`scripts/dump_step1a_failed_inits.py` 消费**：现在读到的 `cache_eval_results.json` 已去重，直接 `if not r["success"]: failed_by_task[...].append(...)` 即可，不会把 retry-成功的 episode 误当失败。

### 19.B6 Step 1b filter/map 规范格式（修正 B6）

**`step1b_filter.json` 固定 schema**（由 `scripts/dump_step1a_failed_inits.py` 生成）：

```json
[
  {"task_id": 3, "orig_init_state_idx": 15, "subset_init_state_idx": 1},
  {"task_id": 3, "orig_init_state_idx": 28, "subset_init_state_idx": 2},
  {"task_id": 5, "orig_init_state_idx": 2,  "subset_init_state_idx": 0}
]
```

**`dump_step1a_failed_inits.py` 同时写**：

```python
# scripts/dump_step1a_failed_inits.py 追加
filter_entries = []
for task_id, bad_indices in sorted(failed_by_task.items()):
    sorted_bad = sorted(bad_indices)
    task = suite.get_task(task_id)
    # {task.name}.init 里 subset 顺序就是 sorted_bad 的顺序
    (out / f"{task.name}.init_map.json").write_text(json.dumps(sorted_bad))
    for subset_pos, orig in enumerate(sorted_bad):
        filter_entries.append({
            "task_id": task_id,
            "orig_init_state_idx": int(orig),
            "subset_init_state_idx": int(subset_pos),
        })
(out / "step1b_filter.json").write_text(json.dumps(filter_entries, indent=2))
```

**`main.py` 读 filter 后用 `subset_init_state_idx` 过滤**（不再反推 orig → subset）：

```python
# examples/libero/main.py
filter_pairs: set[tuple[int, int]] | None = None
if args.episode_filter:
    entries = json.loads(Path(args.episode_filter).read_text())
    filter_pairs = {(e["task_id"], e["subset_init_state_idx"]) for e in entries}
    orig_map: dict[tuple[int, int], int] = {
        (e["task_id"], e["subset_init_state_idx"]): e["orig_init_state_idx"]
        for e in entries
    }

# loop 内（episode_idx 就是 subset_init_state_idx）
if filter_pairs is not None and (task_id, episode_idx) not in filter_pairs:
    continue
orig_idx = orig_map.get((task_id, episode_idx), episode_idx)  # fallback: 无 filter 时等同
```

HDF5 attrs 同时写 `init_state_idx`（subset 内 position）+ `orig_init_state_idx`。

### 19.B7 首版仅实现 prefill_trajectory（修正 B7）

**`exp/_cache_config_rpc.py` 保持 `send_load_cache_config`，删除 `send_prefill_begin/end`**（首版不需要）。

**`src/openpi/serving/websocket_policy_server.py`**：只新增 `prefill_trajectory` 控制分支，**不加** `prefill_begin/prefill_end` 分支。若将来需要 admin debug entry，单独 PR 补。

**§18.A4 的 `pack_payload_b64` 函数**：保留实现备用（Step 3 不走），但不加入 main flow。

**§3.2 改动 2 更新**：删除 `prefill_begin` / `prefill_end` 分支，只保留 `prefill_trajectory`。

### 19.B8 write_policy: never 显式声明（修正 B8）

**全部 6 份 YAML（3 cache + 3 inference）top-level 加**：

```yaml
write_policy:
  type: never
```

理由（简录）：
- inference_*.yaml `always_skip` → 100% MISS → 默认 `on_any_miss` 会在 `episode_end` 把 replay trajectory 写进 backend，污染 artifact；
- cache_*.yaml 首版实验不扩写 cache，全程只读 backend（artifact_path 在 E1/E2/E3 YAML 里预 loaded）。

**现有 3 份 YAML（clip_w7_d4.yaml / spatial16_w8_d4.yaml / max_pool_w3_d5.yaml）必须补 top-level `write_policy: {type: never}`**（当前缺 —— 见 19.B2 中已有 YAML 的补丁）。

### 19.Z 退出条件 checklist

按 review §7 逐项 vs. §19：

- ✅ step 粒度单一且贯穿（§19.1）
- ✅ client/server HDF5 机械对齐（episode_name，§19.1 + §19.4）
- ✅ action tensor 空间明确（model-space for prefill payload，§19.4；env-space first action for L2，§19.5）
- ✅ CollectionPolicy lifecycle 转发（§19.2 + 新测试）
- ✅ prefill facade 覆盖 orchestrator + search_strategy（§19.3）
- ✅ 所有 replay/spawn YAML write_policy: never（§19.B8）
- ✅ 并行 runner 无竞争写（§19.7 per-unit npz）
- ✅ Phase 0 restore 脚本比较同一段轨迹（§19.8）

等待 G1 重审。

---

## 20. G1 第二轮审查修订（R1–R5 · 2026-04-14）

> 来源：`logs/trajectory_deviation_corrective_implementation_review.log.md` §11 R1–R5
> 冲突解决：**§20 > §19 > §1–§18**。§20 是 release-blocker 修订。

### 20.R1 Inference cycle HDF5 对 mid-chunk 结束的处理（修正 §19.1）

**问题复述**：`_pending_executed` 只在 `if not action_plan:` 分支写回 `traj_buffer[-1]["executed_actions"]`。若 env 在 chunk 中途 `done=True` 或循环 break，最后一个 cycle 的 `executed_actions` 仍是 `None`。

#### 20.R1.1 `_run_episode` 收尾 finalize 片段（替换 §19.1 原 finalize 逻辑）

```python
# examples/libero/main.py::_run_episode
# 原有 env.step loop 结束后、flush HDF5 之前 加：
if args.save_trajectory and traj_buffer and traj_buffer[-1]["executed_actions"] is None:
    # 最后一个 cycle 的 chunk 被 done/break 中途截断；用已累积的 _pending_executed 收尾
    if _pending_executed:
        traj_buffer[-1]["executed_actions"] = np.stack(_pending_executed).astype(np.float32)
    else:
        # 极罕见：infer 刚返回就被 done；至少放一条 zero-shape 守底，但不期待此分支触发
        # 此情况下我们宁愿丢弃这个 cycle，避免下游 Step 2 读空数组报错
        traj_buffer.pop()
```

#### 20.R1.2 HDF5 schema 调整（替换 §19.1 schema 段）

```
attrs:
  ...（其他字段保持）
  replan_steps: int             # args.replan_steps, for bookkeeping
step_{i:04d}/:
  sim_state, env_timestep, env_cur_time
  agentview_image (224,224,3) uint8
  eye_in_hand_image (224,224,3) uint8
  robot_state (8,) float64
  env_action_chunk (H, 7) float32
  executed_actions (K, 7) float32           # K 动态，1 <= K <= replan_steps
  executed_action_count: int                # = K, 冗余但方便索引（可做 attr 或 scalar dataset）
```

写入时把 K 作为 dataset attr：

```python
for i, step in enumerate(traj_buffer):
    g = f.create_group(f"step_{i:04d}")
    for k, v in step.items():
        if k == "executed_actions":
            g.create_dataset(k, data=v)
            g.attrs["executed_action_count"] = int(v.shape[0])
        else:
            g.create_dataset(k, data=v)
```

#### 20.R1.3 Step 2 读取不变

`load_gt_first_actions()` 只读 `executed_actions[0]`，K ≥ 1 即可：

```python
def load_gt_first_actions(gt_dir, ep):
    with h5py.File(Path(gt_dir) / f"{ep}.h5", "r") as f:
        T = int(f.attrs["num_cycles"])
        return np.stack([f[f"step_{t:04d}/executed_actions"][0] for t in range(T)])
```

**断言**：Phase 1 runner 载入 GT 后若发现 `num_cycles == 0` 或任一 `executed_actions.shape[0] == 0`，skip 该 episode 并 log WARNING（防御性）。

### 20.R2 Lifecycle 转发覆盖 `PolicyRecorder`（修正 §19.2）

**问题复述**：`CollectionPolicy(PolicyRecorder(InferenceInterceptor))` 的 wrapper chain 里 `PolicyRecorder` 无 lifecycle 方法也无 `__getattr__`，生命周期信号断在中间层。

#### 20.R2.1 `PolicyRecorder` 补 lifecycle 透明转发（必做）

**锚点**：`src/openpi/policies/policy.py:212-234` （class `PolicyRecorder`，当前只有 `infer`）。

**追加**（在 `infer` 方法之后）：

```python
# src/openpi/policies/policy.py PolicyRecorder 类内追加
def on_episode_start(self, *args, **kwargs) -> None:
    if hasattr(self._policy, "on_episode_start"):
        self._policy.on_episode_start(*args, **kwargs)

def on_episode_end(self, *args, **kwargs) -> None:
    if hasattr(self._policy, "on_episode_end"):
        self._policy.on_episode_end(*args, **kwargs)

def on_task_begin(self, *args, **kwargs) -> None:
    if hasattr(self._policy, "on_task_begin"):
        self._policy.on_task_begin(*args, **kwargs)

def on_task_end(self, *args, **kwargs) -> None:
    if hasattr(self._policy, "on_task_end"):
        self._policy.on_task_end(*args, **kwargs)

# 不加 __getattr__：显式列出 4 个 lifecycle hook 更明确；其他属性 inner policy 自行暴露
```

**设计选择**：采 reviewer 方案 1（在 `PolicyRecorder` 侧加转发），而非方案 2（在 `CollectionPolicy` 里走 wrapper-chain walker）。理由：
- 方案 1 修复的是**通用 wrapper 透明性**问题，任何未来 lifecycle hook 都受益；
- 方案 2 会把"穿透 wrapper 链"逻辑散布到每个需要调用 inner 的 wrapper，不可持续。

#### 20.R2.2 测试覆盖（更新 §19.2 的测试）

`tests/test_collect_cache_lifecycle.py` 新增两条 case：

```python
def test_policy_recorder_forwards_lifecycle(tmp_path):
    """PolicyRecorder 必须透传 on_episode_start/end 给 inner policy."""
    inner = Mock(spec=["on_episode_start", "on_episode_end", "infer"])
    rec = PolicyRecorder(inner, str(tmp_path))
    rec.on_episode_start("exp", "task", 1, episode_name="task_0/ep_0")
    inner.on_episode_start.assert_called_once_with("exp", "task", 1, episode_name="task_0/ep_0")
    rec.on_episode_end(success=True)
    inner.on_episode_end.assert_called_once_with(success=True)

def test_collect_recorder_interceptor_full_chain(tmp_path):
    """CollectionPolicy(PolicyRecorder(InferenceInterceptor)) 三层穿透。"""
    interceptor = _build_mock_interceptor()
    recorder = PolicyRecorder(interceptor, str(tmp_path / "rec"))
    collector_policy = CollectionPolicy(recorder, _build_collector(tmp_path / "col"))
    collector_policy.on_episode_start("exp", "task", 1, episode_name="task_0/ep_0")
    # 断言 interceptor.on_episode_start 被调用，且拿到 task_key / episode_id
    assert interceptor._orchestrator.on_episode_start.called
```

### 20.R3 `InferenceInterceptor.on_episode_start` 保留完整原语义（修正 §19.2）

**问题复述**：§19.2 丢了 `if self._orchestrator is not None` guard 与 `task_key=task, episode_id=str(episode_id)` kwargs。

#### 20.R3.1 完整代码（替换 §19.2 对应片段）

**锚点**：`src/openpi/cache/interceptor.py:270-276`（现状已有 guard + task_key/episode_id 传递）。改动**仅增加 `episode_name` 可选参数**：

```python
# src/openpi/cache/interceptor.py
def on_episode_start(
    self,
    experiment: str = "",
    task: str = "",
    episode_id: int = -1,
    episode_name: str = "",       # NEW (kwarg, default empty => backward compat)
) -> None:
    """Reset per-episode state. Called when simulator sends episode_start.

    Keeps original semantics intact: orchestrator may be None (no cache wrapper)
    and must receive task_key / episode_id.
    """
    del experiment, episode_name   # reserved for future use; avoid unused-arg warnings
    if self._orchestrator is not None:
        self._orchestrator.on_episode_start(
            task_key=task,
            episode_id=str(episode_id),
        )
```

**不变项**：
- `self._orchestrator is not None` guard 必保（`--cache` 关闭或 no-orchestrator 配置下 interceptor 仍可能被实例化）；
- `task_key=task` / `episode_id=str(episode_id)` 两个 kwarg 必保（orchestrator 内部用于日志/write_policy 标签）；
- 原无 `experiment` 参数现新增，仅为 wrapper signature 对齐，**不向 orchestrator 转发**（orchestrator 协议未定义 experiment）。

### 20.R4 `collected_dir` CLI 语义固定（修正 §19.4）

**问题复述**：`EpisodeDataCollector` 实际写路径是 `{collect_dir}/{experiment}/{episode_name}.h5`（`experiment` 来自 `client.episode_start(experiment=...)`，libero 场景下 = `args.task_suite_name`）。§19.4 例子同时出现 `collected/libero_xxx/...` 与 `args.collected_dir / ep.h5`，歧义。

#### 20.R4.1 固定 CLI 契约（采 reviewer 方案 A）

**Server 侧**（`scripts/serve_policy.py` 启动时）：
```bash
--collect-dir data/deviation_experiment/collected
# 不追加 task_suite；server 从 episode_start 拿到 experiment 字段自动拼
```
最终文件写在 `data/deviation_experiment/collected/{experiment}/{episode_name}.h5`。

**Client 侧**（`examples/libero/main.py` 透传）：
```python
client.episode_start(
    experiment=args.task_suite_name,   # 即 experiment == "libero_spatial" 等
    task=str(task_description),
    episode_id=global_episode_id,
    episode_name=f"task_{task_id}/episode_{episode_idx}",
)
```

**Step 3 runner 侧**（`exp/run_spawn_experiment.py`）：

```python
# parse_args:
ap.add_argument("--collected-dir", required=True,
                help="Root directory passed as --collect-dir on server start. "
                     "Does NOT include task_suite.")
ap.add_argument("--task-suite-name", required=True,
                help="Must match value passed to main.py --task-suite-name / "
                     "client.episode_start(experiment=...).")

# 在 execute_unit 里：
collected_path = (
    Path(self.collected_dir)
    / self.task_suite_name            # == server-side experiment
    / f"{ep}.h5"                      # ep = "task_X/episode_Y"
)
# 举例：data/deviation_experiment/collected/libero_spatial/task_3/episode_7.h5
```

**Step 2 runner 同理**（若需要读 server-side artifacts；本实验 Phase 1/2 不直接读 server HDF5，但 aggregate 可能）。

#### 20.R4.2 Step 1b runner 启动 server 的协同

`exp/run_step1b_gt.py` 文档字符串明确提醒 user：

```text
Step 1b 启动 server 命令示例：
  uv run scripts/serve_policy.py \
      --config pi05_libero \
      --cache_config configs/cache_runs/deviate_exp/inference_clip_w7_d4.yaml \
      --collect \
      --collect-dir data/deviation_experiment/collected
Step 1b runner 自身不管 server 进程；用户须先起 server。
```

文件布局最终态：

```
data/deviation_experiment/
├── gt_trajectories/task_X/episode_Y.h5                            # client-side (main.py --save-trajectory-dir)
├── collected/libero_spatial/task_X/episode_Y.h5                   # server-side (server --collect-dir + experiment)
├── step2/bg/{cfg}/task_X/episode_Y/s{m:03d}.npz                   # Phase 1 per-unit
├── step2/cache/{cfg}/task_X/episode_Y.npz                         # Phase 2 per-unit
└── step3/{cfg}/task_X__episode_Y__s{s:03d}_n{n:02d}_k{k:02d}.npz  # Spawn per-unit
```

（Step 2/Step 3 路径中 `task_X/episode_Y` 统一展平避免路径层过深，`__` 作为分隔符。aggregate 阶段按分隔符 split 还原 episode_name。）

### 20.R5 `episode_start` 分支统一 keyword 透传（修正 §3.2）

**问题复述**：§3.2 与 §19.2 修过 wrapper 签名，但 server dispatcher 未显式用 keyword 调用，存在 positional fallthrough 绕过 keyword 校验的风险。

#### 20.R5.1 `websocket_policy_server.py::episode_start` 分支（替换 §3.2 改动 1）

```python
# src/openpi/serving/websocket_policy_server.py
# episode_start 分支 — 统一 keyword 调用
elif ctrl == "episode_start":
    experiment = obs.get("__experiment__", "")
    task = obs.get("__task__", "")
    episode_id = obs.get("__episode_id__", -1)
    episode_name = obs.get("__episode_name__", "")   # NEW

    # conn_policy 可能是：
    #   CollectionPolicy(PolicyRecorder(InferenceInterceptor))
    #   PolicyRecorder(InferenceInterceptor)
    #   InferenceInterceptor
    #   或其他 BasePolicy 实现
    # 所有 lifecycle-aware wrapper 必须接受下列 4 个 kwargs（未用的 ignore）
    if hasattr(conn_policy, "on_episode_start"):
        conn_policy.on_episode_start(
            experiment=experiment,
            task=task,
            episode_id=episode_id,
            episode_name=episode_name,
        )
    await websocket.send(packer.pack({"__ack__": "episode_start"}))
    continue
```

**Keyword 强制约束**：所有实现 `on_episode_start` 的类（`CollectionPolicy` / `PolicyRecorder` / `InferenceInterceptor`）其 signature 必须以 **关键字参数形式**接受 `experiment=..., task=..., episode_id=..., episode_name=...`，不能依赖位置顺序。签名全部对齐为：

```python
def on_episode_start(
    self,
    experiment: str = "",
    task: str = "",
    episode_id: int = -1,
    episode_name: str = "",
) -> None: ...
```

#### 20.R5.2 `episode_end` 对应对齐

`episode_end` 只需 `success: bool`。所有层实现统一为：

```python
def on_episode_end(self, success: bool) -> None: ...
```

Server 端：

```python
elif ctrl == "episode_end":
    success = bool(obs.get("__success__", False))
    if hasattr(conn_policy, "on_episode_end"):
        conn_policy.on_episode_end(success=success)
    await websocket.send(packer.pack({"__ack__": "episode_end"}))
    continue
```

### 20.Z 退出条件 checklist（补 R1–R5）

| ID | 修订点 | 验证方式 |
|----|-------|---------|
| R1 | mid-chunk 结束 finalize + K 动态 | 单元测试：模拟 done 在 chunk 第 3 步；断言 HDF5 执行成功且 `executed_actions.shape[0] == 3` |
| R2 | `PolicyRecorder` lifecycle 透传 | `test_policy_recorder_forwards_lifecycle` + 3 层链条集成测试 |
| R3 | `InferenceInterceptor.on_episode_start` guard + task_key/episode_id | Grep `self._orchestrator is not None`；断言原 `task_key=task` 在 patch 后依然存在 |
| R4 | `--collected-dir` 不含 suite；Step 3 读 `{collected_dir}/{task_suite_name}/{ep}.h5` | 集成测试：起 server → 跑 Step 1b 1 episode → Step 3 runner 能定位 server-side HDF5 |
| R5 | keyword 透传 | `mypy --strict` 或单元测试用 positional 调用时 MUST raise TypeError |

待 reviewer 复核本节，若通过，plan approved → 进入 Code 阶段。

---

## 21. G1 第三轮审查修订（S1 · 2026-04-14）

> Conflict resolution 规则升级为 **§21 > §20 > §19 > §1–§18**。reviewer 复核时以 §21 为准。
> 本节只修订 `PolicyRecorder` 签名与 §20.Z 的 R5 验证表述，其余 R1–R5 修订保持不变。

### 21.S1 起因

reviewer 第三轮在 §13.2 指出：§20.R2.1 给 `PolicyRecorder` 写的 `on_episode_start(self, *args, **kwargs)` 与 §20.R5.1 的显式签名 + §20.Z 第 R5 行「positional 调用时 MUST raise TypeError」三者不相容。

- `*args, **kwargs` 接受 positional，不会 raise；
- `def on_episode_start(self, experiment="", ...)`（§20.R5.1 现状，非 keyword-only）同样接受 positional；
- 要让 positional 调用失败，签名必须是 `def on_episode_start(self, *, experiment=..., ...)`。

### 21.S1.1 采纳 reviewer 方案 1（推荐）

**策略**：只要求 server dispatcher 用 keyword 调 wrapper；wrapper 内部仍用普通 default kwargs（不加 `*`），但 wrapper 之间转发亦必须用 keyword。**删除「positional 调用 MUST raise」这条约束**。

理由：
- 本实验范围内无 positional caller；只要 server dispatcher 保持 keyword（§20.R5.1 已固定），正确性已足；
- 若改 keyword-only（`*` 分隔），会波及所有已有 lifecycle hook 实现与测试，扩大 blast radius；
- 显式同构签名比 `*args, **kwargs` 更利于静态检查与 IDE 跳转。

### 21.S1.2 替换 §20.R2.1 — `PolicyRecorder` lifecycle 透传

**锚点**：`src/openpi/policies/policy.py:212-234`（class `PolicyRecorder`）。

**最终版本（覆盖 §20.R2.1）**：

```python
# src/openpi/policies/policy.py PolicyRecorder 类内追加（显式同构签名 + keyword 转发）
def on_episode_start(
    self,
    experiment: str = "",
    task: str = "",
    episode_id: int = -1,
    episode_name: str = "",
) -> None:
    if hasattr(self._policy, "on_episode_start"):
        self._policy.on_episode_start(
            experiment=experiment,
            task=task,
            episode_id=episode_id,
            episode_name=episode_name,
        )

def on_episode_end(self, success: bool = False) -> None:
    if hasattr(self._policy, "on_episode_end"):
        self._policy.on_episode_end(success=success)

def on_task_begin(self, task: str = "") -> None:
    if hasattr(self._policy, "on_task_begin"):
        self._policy.on_task_begin(task=task)

def on_task_end(self, task: str = "") -> None:
    if hasattr(self._policy, "on_task_end"):
        self._policy.on_task_end(task=task)
```

**对 §20.R2.2 测试的影响**：§21.S1.3 已把 server dispatcher 与 wrapper forward 全 keyword 升为刚性约束，因此 §20.R2.2 原测试片段中带 positional 的调用需改写为全 keyword。测试调用与断言**一律用 keyword 形式**（`rec.on_episode_start(experiment=..., task=..., episode_id=..., episode_name=...)` + `inner.on_episode_start.assert_called_once_with(experiment=..., task=..., episode_id=..., episode_name=...)`），与 `PolicyRecorder` 内部的 keyword forward 对齐，避免 Mock `call` 比较在 positional/keyword 表达差异上出现假阳性。签名本身虽非 keyword-only，**测试与生产代码路径均禁止出现 positional 调用**。

测试片段更新为：

```python
def test_policy_recorder_forwards_lifecycle(tmp_path):
    inner = Mock(spec=["on_episode_start", "on_episode_end", "infer"])
    rec = PolicyRecorder(inner, str(tmp_path))
    rec.on_episode_start(
        experiment="exp", task="task", episode_id=1, episode_name="task_0/ep_0",
    )
    inner.on_episode_start.assert_called_once_with(
        experiment="exp", task="task", episode_id=1, episode_name="task_0/ep_0",
    )
    rec.on_episode_end(success=True)
    inner.on_episode_end.assert_called_once_with(success=True)
```

### 21.S1.3 `CollectionPolicy` / `InferenceInterceptor` 签名确认（不变）

两者签名已在 §19.2 / §20.R3.1 中采用普通 default kwargs 形式（非 keyword-only）：

```python
# collection_policy.py
def on_episode_start(
    self,
    experiment: str,
    task: str,
    episode_id: int,
    episode_name: str = "",
) -> None: ...

# interceptor.py
def on_episode_start(
    self,
    experiment: str = "",
    task: str = "",
    episode_id: int = -1,
    episode_name: str = "",
) -> None: ...
```

**约束**（升级为刚性要求）：
- Server dispatcher（`websocket_policy_server.py::episode_start`）**必须** keyword 调 wrapper —— 已在 §20.R5.1 固化；
- Wrapper 之间互相 forward **必须** keyword —— `CollectionPolicy → PolicyRecorder → InferenceInterceptor` 三层全部用 `experiment=..., task=..., episode_id=..., episode_name=...`；
- **不**要求 positional 调用 raise（删除 §20.Z 原 R5 行）。

### 21.S1.4 替换 §20.Z R5 行

**旧（§20.Z 中 R5 行）**：

| R5 | keyword 透传 | `mypy --strict` 或单元测试用 positional 调用时 MUST raise TypeError |

**新（替换）**：

| R5 | keyword 透传 | (a) `grep` 确认 `websocket_policy_server.py::episode_start / episode_end` 分支调用 wrapper 时全部用 keyword；(b) `grep` 确认 `CollectionPolicy` / `PolicyRecorder` 内部 forward 到 inner policy 时全部用 keyword；(c) 单元测试 `test_server_dispatch_uses_keyword`（mock wrapper，断言 `call_args.kwargs` 含四个字段、`call_args.args == ()`）|

**新增测试 `test_server_dispatch_uses_keyword`**（放在 `tests/test_server_dispatch.py` 或合并至 `tests/test_collect_cache_lifecycle.py`）：

```python
def test_server_dispatch_uses_keyword(monkeypatch):
    """Server episode_start/end 分支必须用 keyword 调 wrapper。"""
    fake_policy = Mock(spec=["on_episode_start", "on_episode_end", "infer"])
    # 构造一个最小 obs payload 走到 episode_start 分支
    obs = {
        "__ctrl__": "episode_start",
        "__experiment__": "libero_spatial",
        "__task__": "pick red block",
        "__episode_id__": 7,
        "__episode_name__": "task_0/episode_0",
    }
    _dispatch_episode_start(fake_policy, obs)  # 从 websocket_policy_server 抽出的 helper
    call = fake_policy.on_episode_start.call_args
    assert call.args == ()
    assert set(call.kwargs) == {"experiment", "task", "episode_id", "episode_name"}
```

（若目前 server 未抽 helper，则在测试内直接内联相同的 branch 逻辑；重点是断言 `call.args == ()`。）

### 21.S1.5 不动项列表（防止 regression）

下列 §20 决定保持不变，reviewer 复核请以下列位置为准：

- §20.R1（mid-chunk finalize + dynamic K schema）
- §20.R3（InferenceInterceptor guard + task_key/episode_id）
- §20.R4（`--collected-dir` 不含 suite 的 CLI 契约 + 文件布局）
- §20.R5.1 server dispatcher keyword 调用（本节 §21.S1.3 将其升为刚性约束）
- §20.R5.2 `episode_end` 仍 `success: bool`
- §19 所有 A1–A8 / B1–B8 修订

### 21.Z 退出条件 checklist（更新 R5 验证方式，其余保持 §20.Z）

| ID | 修订点 | 验证方式 |
|----|-------|---------|
| R1 | mid-chunk 结束 finalize + K 动态 | 单元测试：模拟 done 在 chunk 第 3 步；断言 HDF5 执行成功且 `executed_actions.shape[0] == 3` |
| R2 | `PolicyRecorder` lifecycle 透传（显式同构签名 + keyword forward） | `test_policy_recorder_forwards_lifecycle` 用 keyword `assert_called_once_with`；3 层链条集成测试 |
| R3 | `InferenceInterceptor.on_episode_start` guard + task_key/episode_id | Grep `self._orchestrator is not None`；断言 `task_key=task` 在 patch 后依然存在 |
| R4 | `--collected-dir` 不含 suite；Step 3 读 `{collected_dir}/{task_suite_name}/{ep}.h5` | 集成测试：起 server → 跑 Step 1b 1 episode → Step 3 runner 能定位 server-side HDF5 |
| R5 | keyword 透传 | (a)(b) grep server dispatch & wrapper forward 全 keyword；(c) `test_server_dispatch_uses_keyword` 断言 `call.args == ()` |
| S1 | `PolicyRecorder` 签名与 R5 验证不冲突 | 签名为显式同构（非 keyword-only）；删除 "positional MUST raise TypeError" 的要求 |

待 reviewer 复核 §21，若通过 → plan approved → 进入 Code 阶段（按 §15 checkpoint 顺序 Layer A → D → B → C → F）。

