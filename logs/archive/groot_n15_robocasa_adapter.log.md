# GR00T N1.5 × RoboCasa365 推理适配层 — 实施计划

**Status**: `In Progress`（G1 APPROVED 2026-08-16 09:36 CDT，进入 §4 Code）
**Level**: L2（新组件；后续 GR00T 版 Interceptor/KeyBuilder 的接入基座）
**Authority**: Execution
**日期**: 2026-08-16

---

## 0. 给 G1 审查者的上下文（无需对话历史）

**这份 plan 要解决什么**：本项目正在筹备一个**跨场景 cache 继承实验**——把 cache 库建在厨房场景 A，检验它在同任务、不同厨房场景 B 上是否仍可用。战场已选定为 **RoboCasa365 / Atomic split**，第一 teacher **pi0.5 已全线打通并完成准入门实测**（180 ep，SR_A 41.1% / SR_B 50.0%，详见 [`benchmark_and_teacher_selection.log.md`](benchmark_and_teacher_selection.log.md) §12-2）。

为消除单 teacher (n=1) 的局限，需接入**第二 teacher = GR00T N1.5**。官方已发布该模型在 RoboCasa365 上训练好的 checkpoint，**因此本计划不涉及任何训练或微调**——只补一条推理链路。

**为什么需要写代码**（两边各缺一半，这是本 plan 存在的唯一理由）：

| 代码库 | 有 RoboCasa365 集成？ | 有 N1.5 模型实现？ |
|---|---|---|
| `Isaac-GR00T` **n1.5-release** | ❌ 其 `examples/RoboCasa` 面向 RoboCasa **Tabletop / GR1** 机器人，非 RoboCasa365 PandaOmron | ✅ |
| `Isaac-GR00T` **main (N1.7)** | ✅ `gr00t/eval/sim/robocasa365/` | ❌ `gr00t/model/` 下**只有 `gr00t_n1d7`**；`gr00t_n1d7.py:490-497` 的 `get_backbone_cls()` 仅接受 `nvidia/Cosmos-Reason2` 与 `Qwen/Qwen3-VL`，其余 `raise ValueError` |

⇒ 二者不能直接合体，需要一层薄适配。**本 plan 的全部工作量集中在此**。

**范围之外**：不训练、不微调、不改 `Isaac-GR00T` 源码、不改 `robocasa` 源码、不改本仓库 `src/` 下任何既有文件、不接 cache（cache 接入是后续独立工作）。

---

## 1. 环境与资产清单（**全部为实机已验证的绝对路径**）

> 本节应 owner 明确要求编写：审查者不必自行查找任何路径。以下条目均在 **weilandserver** 上实机确认存在。

### 1.1 执行主机

| 项 | 值 |
|---|---|
| 主机 | **weilandserver**（经 `tether exec weilandserver -- …` 访问） |
| GPU | 单张 **RTX 4090（49140 MiB）** |
| ⚠ 共用情况 | 该卡由**多个 session 共用**。已知常驻：`serve_policy.py` @ :8000（**约 8.8 GB，他人所有，禁止关闭**）、ACT `sidecar_server.py`（约 3.4 GB）、`rlr*` 系列 tmux。**禁止宽模式 `pkill`，只能按 PID / tmux session 名定点操作。** |
| ⚠ 端口 | **8000 已被占用（他人）**。pi0.5 teacher 用 **8010**。本 plan 的 GR00T server 使用 **8020**，避免冲突。 |

### 1.2 三个互斥 venv 孤岛（**装不进同一个环境，这是需要 server/client 分离的根本原因**）

| 孤岛 | 绝对路径 | Python / 关键依赖 | 用途 |
|---|---|---|---|
| **A（sim）** | `/home/weiland/Isaac-GR00T/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv` | py **3.12**, numpy **2.2.5** | 跑 robocasa365 仿真环境 |
| **B（GR00T）** | venv `/home/weiland/gr00t_n15_venv/.venv`；**源码 `/home/weiland/gr00t_n15`** | py **3.11.15**, numpy **1.26.4**, transformers **4.51.3**, torch **2.5.1+cu124**, flash-attn 2.7.4 | 跑 GR00T N1.5 模型 |

⚠⚠ **`gr00t` 并未安装进孤岛 B 的 venv**（实测 `import gr00t` → `ModuleNotFoundError`，site-packages 无该包、无 .pth）。它来自 **git worktree `/home/weiland/gr00t_n15`（detached HEAD `4af2b62` = `n1.5-release`，已核对）**，靠 `PYTHONPATH` 引入。⇒ 任何在孤岛 B 内运行的命令都**必须**把该路径放进 `PYTHONPATH`，否则静默走到 `importorskip` 分支、看起来"跳过"而非"失败"。
| **主 venv** | `/home/weiland/openpi/.venv` | py 3.11 | openpi + pi0.5 teacher |

⚠ **孤岛 A 的 numpy 必须保持 2.2.5**：直接 `uv pip install openpi_client` 会因其 `numpy>=1.22.4,<2.0.0` 约束把 numpy 降级到 1.26.4，触发 robocasa 的 import 断言与 scipy 的 `np.long` 崩溃（已实际发生并修复）。**装任何包到孤岛 A 必须带 `--no-deps`。**

⚠ **孤岛 A 的工作目录必须是** `/home/weiland/Isaac-GR00T/external_dependencies/robocasa365`（robocasa 依赖相对路径定位资产）。

### 1.3 模型 checkpoint

| 模型 | 绝对路径 | 说明 |
|---|---|---|
| **GR00T N1.5（本 plan 使用）** | `/home/weiland/ckpt_n15_robocasa/gr00t_n1-5/multitask_learning/checkpoint-120000` | 7.2 GB（2 个 safetensors 分片）。**官方在 RoboCasa365 上训练好的权重**，`trainer_state.json` 记录 `global_step=120000`。已实机 `from_pretrained()` 加载成功（vision 27 层 / LLM 12 层）。**无需微调。** |
| pi0.5（既有，参考用） | `/home/weiland/ckpt_pi05_robocasa_pytorch` | 6.9 GB，JAX→PyTorch 已转换 |

checkpoint 内**唯一**的配置资产是 `experiment_cfg/metadata.json`（14 KB）——只含统计量与 modality 规格，**不含 modality_config / transform 定义**。这正是 §4.1 需要自建 DataConfig 的原因。

### 1.4 EGL 渲染（该机无系统级 EGL，必须显式 export）

```bash
export LD_LIBRARY_PATH=/home/weiland/nvidia-gl/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export __EGL_VENDOR_LIBRARY_DIRS=/home/weiland/nvidia-gl/root/usr/share/glvnd/egl_vendor.d
export MUJOCO_GL=egl
```

### 1.5 既有参考产物（pi0.5 链路，本 plan 的直接模板）

目录 `/home/weiland/step0b_artifacts/`：

| 文件 | 作用 |
|---|---|
| `serve_robocasa_pi05.py` | pi0.5 的 websocket policy server（65 行）——**本 plan 的 server 以此为结构模板** |
| `step0b_v2.py` | sim client（场景钉定 + rollout 循环）——**本 plan 的 client 以此为结构模板** |
| `analyze_step0b.py` | 结果分析（Wilson CI / U0-U1-U2 分类） |
| `step0b_full.{json,log}` | pi0.5 的 180 ep 准入门原始结果 |

### 1.6 代码库位置

| 仓库 | 本机路径 | 备注 |
|---|---|---|
| `Isaac-GR00T` | `/home/weiland/projects/Isaac-GR00T`（本地）、`/home/weiland/Isaac-GR00T`（weilandserver） | 已 `fetch --unshallow --tags`；读 N1.5 源码用 `git show n1.5-release:<path>` |
| `robocasa` | `/home/weiland/projects/robocasa` | pin `be22d659b02db8f6d7f3a3c3edc742934fdcbaae` |
| 本仓库 | `/home/weiland/projects/openpi`（本地）、**`/home/weiland/openpi`（weilandserver）** | ⚠ 全文所有可执行命令用的都是 **server 侧**路径 `/home/weiland/openpi` |

---

## 2. 已亲验的技术事实（每条带 file:line，审查者可直接核对）

> 以下均为本人实际执行 / 实际读取所得，非文档推断。

**F1 — `Gr00tPolicy` 构造签名**（`n1.5-release:gr00t/model/policy.py:65-72`）：
```python
Gr00tPolicy(model_path: str, embodiment_tag: Union[str, EmbodimentTag],
            modality_config: Dict[str, ModalityConfig],
            modality_transform: ComposedModalityTransform,
            denoising_steps: Optional[int] = None, device: Union[int,str] = ...)
```
⇒ `modality_config` 与 `modality_transform` **必须由调用方提供**，模型自身不携带。

**F2 — `get_action` 的数据契约**（`n1.5-release:gr00t/model/policy.py:146-156`）：输入
`{"video.<name>": (T,H,W,C), "state.<name>": (T,D), "annotation.<name>": (T,)}`，返回 `Dict[str, Any]`，内部经 `self._modality_transform.unapply(action)` 反归一化（同文件 :144）。

**F3 — 构造方式**（`n1.5-release:scripts/eval_policy.py:110-120`）：`modality_config = data_config.modality_config()`、`modality_transform = data_config.transform()`。

**F4 — n1.5-release 的 `DATA_CONFIG_MAP` 无 RoboCasa 条目**（`gr00t/experiment/data_config.py:775-787`）：仅 `fourier_gr1_*` / `bimanual_panda_*` / `single_panda_gripper` / `so100*` / `unitree_g1*` / `oxe_droid` / `agibot_genie1`。N1.7 的 `gr00t/configs/data/data_config.py` 亦无 robocasa 条目。

**F5（关键）— `SinglePandaGripperDataConfig` 的 state/action 键与本 checkpoint 逐键一致**（`n1.5-release:gr00t/experiment/data_config.py:552-595`）：

| | DataConfig 声明 | checkpoint `metadata.json` |
|---|---|---|
| state | `end_effector_position_relative` / `end_effector_rotation_relative` / `gripper_qpos` / `base_position` / `base_rotation` | **完全相同**（维度 3/4/2/3/4 = 16） |
| action | `end_effector_position` / `end_effector_rotation` / `gripper_close` / `base_motion` / `control_mode` | **完全相同**（维度 3/3/1/4/1 = 12） |
| video | `left_view` / `right_view` / `wrist_view` | ❌ **不同**：`robot0_agentview_left` / `robot0_agentview_right` / `robot0_eye_in_hand` |

⚠ **注意：不能由上表推出「只需覆盖 `video_keys`」**（该推论已被 F11/F18 否定）：上表只证明**键集合**相等，而 `ConcatTransform` 依赖**有序**的 `*_concat_order`。集合相等 ⇏ 顺序相等。修订后的处理见 §4.1（显式写出有序列表 + 逐条标注证据等级）与 §9.1（open-loop parity 实证）。此外 `language_keys` 亦不同（F12）。

**F6 — 继承来的关键超参**（同上，:574-595）：`observation_indices = [0]`（单帧）、`action_indices = list(range(16))`（**action_horizon = 16**）、state 全 `min_max` + `state_target_rotations` 把两个四元数转 `rotation_6d`、action 为 `min_max`，其中 `gripper_close` 与 `control_mode` 为 `binary`。

**F7 — checkpoint metadata 规格**（`experiment_cfg/metadata.json`，实读）：`embodiment_tag = "new_embodiment"`；统计字段齐备（`max/min/mean/std/q01/q99`）；video 三路均 `resolution [256,256] / channels 3 / fps 20.0`。

**F8（关键）— action 输出可直接喂 env，无需转换**：`robocasa/utils/env_utils.py:147-153` 的 `convert_action()` 产出的 dict 键为
`action.end_effector_position` / `action.end_effector_rotation` / `action.gripper_close` / `action.base_motion` / `action.control_mode`
——与 F5 的 GR00T `action_keys` **逐字相同**。⇒ GR00T 的原生输出结构即 env 的输入结构。

**F9 — env 原生 obs 即 GR00T 输入格式**：robocasa365 gym env 直接给出 `video.robot0_agentview_left` / `video.robot0_eye_in_hand` / `video.robot0_agentview_right`、`state.<name>`、`annotation.human.task_description`（见 `step0b_artifacts/step0b_v2.py` 实跑验证）。⇒ 与 F2 的契约同构。

**F10 — 图像方向已由 wrapper 处理**：`robocasa/wrappers/gym_wrapper.py:271-273` 已做 `np.copy(img[::-1, :, :])` 上下翻转，**消费方不可再翻**。

> F11–F17：键顺序、server 接口、依赖面与 parity 工具的查证。**F11 是本节最关键的一条**——它否定了「键集合相等即可」这一直觉。

**F11（关键）— checkpoint metadata 的键顺序是「字母序」，不携带训练顺序**（`experiment_cfg/metadata.json` 实读）：

| 段 | 实际顺序 |
|---|---|
| `modalities.state` | `base_position, base_rotation, end_effector_position_relative, end_effector_rotation_relative, gripper_qpos` |
| `modalities.action` | `base_motion, control_mode, end_effector_position, end_effector_rotation, gripper_close` |

二者**均为严格字母序**，且与 `SinglePandaGripperDataConfig` 的声明顺序（state 以 `end_effector_*` 开头、action 以 `end_effector_*` 开头）**不同**。
⇒ **「F5 键集合相同 ⇒ 只需覆盖 video_keys」的推论不成立**：集合相等不蕴含顺序相等，而 N1.5 的 `ConcatTransform` 按 `*_concat_order` 逐维拼接/拆分，**顺序错会产生形状与数值范围都正常、但语义错位的动作**（静默错误）。本项的修订见 §4.1 与 §9.1。

**F12 — RoboCasa365 官方 wrapper 的键与顺序常量**（`Isaac-GR00T(main):gr00t/eval/sim/robocasa365/gymnasium_groot.py`）：
- `:43-47` `CAMERA_NAMES = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]`（**有序**）
- `:48-52` `MAPPED_CAMERA_NAMES = ["video.res256_image_side_0", "video.res256_image_side_1", "video.res256_image_wrist_0"]`，`:55` `ROBOCASA_PANDA_VIDEO_OBSERVATION_KEYS = MAPPED_CAMERA_NAMES` ⇒ **N1.7 的 `ROBOCASA_PANDA` embodiment 用的是映射名，而我们的 N1.5 checkpoint（`new_embodiment`）metadata 用的是原生相机名** —— 两套命名不可混用。
- `:58-60` language key 有两套：`LANGUAGE_OBSERVATION_KEY = "annotation.human.task_description"` vs `ROBOCASA_PANDA_LANGUAGE_OBSERVATION_KEY = "annotation.human.action.task_description"`。
  ⚠⚠ **切勿据此把 env 键映射为 `.action.` 变体**（F18/F19 已确定正确值）：`ROBOCASA_PANDA` 是 **N1.7 的 embodiment tag**，而本 plan 使用的 checkpoint 是 **N1.5 的 `new_embodiment`**（F7）——**两者的 language 键不同，N1.7 的常量不适用于 N1.5**。官方 N1.5 配置（F18）与数据侧资产（F19）一致地使用 **`annotation.human.task_description`**，即 **env 原生键**。
  ⇒ **正确做法是「不映射」**：env 给出的 `annotation.human.task_description` 直接就是 N1.5 所需的键。此前基于本条得出的映射方案（§4.2）与风险项 R9 均已按此更正。`SinglePandaGripperDataConfig.language_keys`（`data_config.py:573`）是**父类为其自身 embodiment 设定的值，本 plan 必须覆盖**，不可继承。
- `:61-62` `CAMERA_RESOLUTION = 512`、`FINAL_IMAGE_RESOLUTION = (256, 256)` ⇒ 官方做法是**渲染 512 再降采样到 256**，非直接渲染 256。

**F13 — server 接口不匹配**（`src/openpi/serving/websocket_policy_server.py:444`）：该 server 要求 `policy` 实现 `BasePolicy.infer()`，而 `Gr00tPolicy` 只提供 `get_action()`（F2）⇒ **必须写 adapter**（审查意见 2 指出，属实）。

**F14 — 孤岛 B 依赖实测全缺**（`/home/weiland/gr00t_n15_venv/.venv` 逐个 import 验证）：`openpi_client` / `websockets` / `msgpack_numpy` / `msgpack` **四者均 MISS**。
🟢 但可安全补齐：`packages/openpi-client/pyproject.toml` 声明 `numpy>=1.22.4,<2.0.0`，孤岛 B 的 numpy 为 **1.26.4，落在区间内** ⇒ **正常安装即可，无需 `--no-deps`**（与孤岛 A 的情况相反）。

**F15 — `websocket_policy_server` 可安全跨环境引入**：该模块 import 面仅为标准库 + `openpi_client.{base_policy,msgpack_numpy}` + `websockets`，**不含 jax / torch / openpi 主包**；且 `src/openpi/__init__.py` 为**空文件（0 行）**、`src/openpi/serving/` **无 `__init__.py`**（namespace package）⇒ 以 `PYTHONPATH=<repo>/src` 引入**不会触发任何重依赖**。

**F16 — 官方自带 open-loop MSE 工具**（`n1.5-release:scripts/eval_policy.py:28,157-173`）：`from gr00t.utils.eval import calc_mse_for_single_trajectory`，对 `LeRobotSingleDataset` 逐轨迹算动作 MSE。⇒ 可用官方工具做 open-loop 动作误差评估，**不必依赖高噪声的 rollout SR**。
⚠ **但该工具的输出不是确定性量**：`get_action()` 每次以随机噪声为去噪起点（**F20**），故基于它的比较必须固定/配对随机种子并做多次重复统计，**不可称之为「确定性 parity」**；用法见 §9.1 G0-B。顺序判别已不再需要（**F18** 直证），本工具现仅用于检验归一化/旋转这两项残留推定。

**F17 — RoboCasa365 训练数据不在本机**：`/home/weiland/Isaac-GR00T/external_dependencies/robocasa365/datasets` **目录不存在**（此前只下载了 22.41 GiB 的**仿真资产**，非 LeRobot 训练数据）⇒ 做 F16 的 parity 需先下载一小片数据（§9.1）。


> F18–F21：官方配置直证、随机性与四元数约定。**F18 是全篇的锚点**——它以官方配置直接确定了四类键的有序契约与 language key。

**F18（决定性直证）— RoboCasa365 官方给出了 N1.5 `new_embodiment` 的完整有序 `modality_configs`**（`robocasa@be22d659:docs/datasets/using_datasets.md:167-210`，本地已 checkout 该 pin，逐字读取）：

```python
embodiment_tag = EmbodimentTag("new_embodiment")      # 与本 checkpoint 的 tag 一致（F7）
modality_configs = {
  "video":    ModalityConfig(delta_indices=[0],
      modality_keys=["video.robot0_agentview_left",
                     "video.robot0_agentview_right",
                     "video.robot0_eye_in_hand"]),
  "state":    ModalityConfig(delta_indices=[0],
      modality_keys=["state.end_effector_position_relative",
                     "state.end_effector_rotation_relative",
                     "state.gripper_qpos",
                     "state.base_position",
                     "state.base_rotation"]),
  "action":   ModalityConfig(delta_indices=list(range(16)),
      modality_keys=["action.end_effector_position",
                     "action.end_effector_rotation",
                     "action.gripper_close",
                     "action.base_motion",
                     "action.control_mode"]),
  "language": ModalityConfig(delta_indices=[0],
      modality_keys=["annotation.human.task_description"]),
}
```

对本 plan 的三点影响：

| 项 | 此前状态 | 现状 |
|---|---|---|
| video / state / action **顺序** | 🟡 强推定（靠 F12 的 N1.7 常量 + 父类槽位语义） | 🟢 **官方直证，且与此前推定一致**（推定被证实，但依据换成直证） |
| **language key** | ❌ 误用 `annotation.human.action.task_description` | 🟢 **应为 `annotation.human.task_description`** |
| `action` 时间索引 | F6 的 `range(16)` | 🟢 与官方 `delta_indices=list(range(16))` 一致 |

**F19 — 同仓库 modality 资产佐证 language 键**（`robocasa@be22d659:robocasa/models/assets/groot_dataset_assets/PandaOmron_modality.json`，实读）：`annotation` 段为 `["human.task_description"]`，与 F18 一致。
⚠ 该 json 的 video/state/action 段为**字母序**（与 F11 的 metadata 相同），故**有序证据只来自 F18 的 `using_datasets.md`**，不可用本 json 推顺序。

**F20 — `get_action()` 含随机采样，MSE 不是确定性量**（`n1.5-release:gr00t/model/action_head/flow_matching_action_head.py:363-368`，实读）：
```python
actions = torch.randn(size=(batch_size, action_horizon, action_dim), ...)   # 去噪起点
```
⇒ flow-matching 每次调用以**随机噪声**为起点迭代 `num_inference_timesteps` 步。**同一输入的两次调用结果不同** ⇒ 任何基于单次调用的比较都不可复现，必须固定/配对种子（§9.1）。

**F22（关闭一处高风险未知）— 父类 `transform()` 全程读 `self.*`，故我们的键覆盖对 transform 链有效**（`n1.5-release:gr00t/experiment/data_config.py:459-500`，实读）：`VideoToTensor/VideoCrop/VideoResize(apply_to=self.video_keys)`、`StateActionTransform(apply_to=self.state_keys, normalization_modes=self.state_normalization_modes, target_rotations=self.state_target_rotations)`、以及关键的
`ConcatTransform(video_concat_order=self.video_keys, state_concat_order=self.state_keys, action_concat_order=self.action_keys)`。
⚠ 这曾是**唯一一处「错了却能通过所有现有关卡」的位置**：若父类改用硬编码列表，`groot_data_config.py` 的四个类属性覆盖将只对 `modality_config()` 生效、对 transform 无效，而 §4.3 的握手比对的正是 `modality_config()`（它比对的是自己传进去的同一对象，属自证），会一片整齐地放行。现由孤岛 B 的 `test_transform_concat_order_uses_our_keys` 直接断言 `ConcatTransform` 的三个 `*_concat_order`，**实测通过**。

**F23 — 模型输入分辨率是 224，256 是数据集分辨率**（同上，`VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear")`）：checkpoint metadata 声明的 `[256,256]`（F7）是**数据集存储**规格，transform 链随后统一 resize 到 **224**。⇒ client 送 256 与训练路径一致（512 渲染 → 256 → transform 224）；若直接送 512，虽也会被 resize 到 224，但降采样路径不同（512→224 vs 512→256→224），故仍应按 256 送。

**F24 — robosuite 的四元数是 xyzw，与 pytorch3d 的 wxyz 相反**（`robosuite/utils/transform_utils.py:317-325` 的 `mat2quat` docstring 明写返回 `(x,y,z,w)`；`robosuite/robots/mobile_robot.py:317,376` 的 `base_quat` / `base_to_eef_quat` 均走 `mat2quat`）。
⇒ **不是 serving 路径的正确性问题**：训练侧取的是同样两个 hdf5 字段，训练与评测被同一套 wxyz 变换同样地"误读"，模型学到的就是该映射，一致即无害。
⚠ **但影响握手探针**：`IDENTITY_QUATERNION_WXYZ = (1,0,0,0)` 在 env 的真实编码下是 `(x=1,y=0,z=0,w=0)`，即绕 X 轴 180°，**并非 env 语义上的单位旋转**。若拿它填探针，5 个 state 字段中会有 2 个取到分布外的值。故 §4.3 的探针改为**取 checkpoint 自身的四元数均值再做 L2 归一化**（多个四元数的均值本身不是单位模长，而非单位四元数不是旋转），仅在均值退化为零向量时才回退到该常量。

**F21 — 四元数约定为 wxyz（real part first），单位四元数 = `[1,0,0,0]`**：`n1.5-release:gr00t/data/transform/state_action.py:21` `import pytorch3d.transforms as pt`；孤岛 B 内实跑 `quaternion_to_matrix`：`[1,0,0,0]` → 单位阵 **True**，`[0,0,0,1]` → **False**。
⇒ 启动自检的 dummy obs 中，`state.end_effector_rotation_relative` 与 `state.base_rotation`（各 4 维）**必须置 `[1,0,0,0]`**；**全零四元数非法**，会使 quaternion→matrix→`rotation_6d` 路径产生无效值。

---

## 3. 设计

### 3.1 拓扑（与 pi0.5 链路同构，便于两 teacher 对照）

```
孤岛 A (py3.12, robocasa365)              孤岛 B (py3.11, GR00T N1.5)
┌────────────────────────────┐  websocket  ┌──────────────────────────────┐
│ exp/robocasa365/            │   :8020     │ exp/robocasa365/             │
│   groot_rollout_client.py   │ ──────────► │   serve_groot_n15.py         │
│   · gym env（场景钉定）      │            │   · Gr00tPolicy              │
│   · 选键 + 512→256 降采样    │ ◄────────── │   · RoboCasa365DataConfig    │
│   · action dict → env.step  │   actions   │   · GrootPolicyAdapter       │
│                             │             │     (infer → get_action)     │
└────────────────────────────┘             └──────────────────────────────┘
```

**为什么仍要 server/client 分离**（而非单进程）：§1.2 的孤岛 A/B 互斥（py3.12+numpy2.2.5 vs py3.11+numpy1.26.4），无法共存于一个解释器。这与 pi0.5 链路的理由相同。

**为什么不复用 N1.7 的 `gymnasium_groot.py`**：它属 N1.7 代码库，而 N1.7 加载不了 N1.5 权重（§0 表）；且我们已有一份实跑验证过的 client（`step0b_v2.py`），复用它可让两个 teacher 走**完全相同的评测循环**，消除"评测代码差异"这一混淆源。

### 3.2 数据流

> ⚠ 切勿把本链路描述为"无格式转换"：键名与结构确实天然对齐（F8/F9/F18），但仍需 ①时间轴前置、②分辨率降采样、③校验与有限值门禁。

| 环节 | 处理 |
|---|---|
| env → client | 原生 obs dict（F9），含 `annotation.human.task_description` |
| client → server | 取 3 路图像（**按 §4.1 的有序 `video_keys`**）+ 5 个 state 键 + `annotation.human.task_description`；**渲染 512 → 降采样 256**（F12 官方做法）；msgpack 序列化。**wire 上的键名与 GR00T 键名逐字相同** |
| server 内 | ①校验键齐全 / shape / dtype / **有限值**，任一不符即 `ValueError` 并列出键名；②各 modality 前置 `T=1` 时间轴；③`Gr00tPolicy.get_action()`；④输出再过一次 `np.isfinite` 门禁。⚠ **不做任何键名映射**——F18 证明 N1.5 `new_embodiment` 用的就是 env 原生 `annotation.human.task_description`（⚠ **切勿**映射为 N1.7 的 `.action.` 变体） |
| server → client | `{"actions": {action.*}, "server_timing": {...}}`；client 只取 `actions`，取前 `replan_steps` 步（`<= 16`，F6） |
| client → env | 逐步组装成 env 期望的 dict（F8，键名逐字相同），`env.step()` |

---

## 4. 文件清单

> 遵循 **`docs/experiments/artifact_layout.md`**（WA §8 注册章程，与 Working Agreement 同等效力），其 §1 canonical tree 与 §4 新建实验步骤为本节的强制约束。

### 4.0 目录结构（章程 §1 canonical tree + §4 新建实验步骤）

```
exp/robocasa365/
  __init__.py                    # 章程 §4-1 要求，1 行 docstring
  groot_keys.py                  # 纯常量（四类有序键），无第三方依赖 ⇒ Layer 1 可在主环境测
  groot_data_config.py           # runner/helper 置于 root（章程 §4-3，不建子包）
  groot_policy_adapter.py
  serve_groot_n15.py
  groot_rollout_client.py
  config/                        # 章程 §1；本实验暂无 YAML，保留空槽
  data/                          # 所有运行期产物（*.json/*.log）落此，章程 §2
  analysis/                      # 分析脚本与 *.md，章程 §2
tests/robocasa365/               # 章程 §1 末行：测试在 tests/ 下，不在 exp/
  test_groot_obs_adapter.py      # Layer 1，非 manual：键契约 / obs 转换 / chunk 契约（server+client 两端）
  test_groot_rollout_client.py   # Layer 1，非 manual：env 异常回收 / 配对 seed / 场景钉定（注入 sim 假件）
  test_groot_data_config_manual.py  # Layer 2，manual@孤岛B：DataConfig 实构造 + checkpoint metadata 对照
  test_env_action_contract_manual.py   # Layer 2，manual@孤岛A：真实 convert_action 的 env 契约绑定
```

⚠ **落盘位置钉死**：逐任务结果 JSON 与 rollout 日志一律写 **`exp/robocasa365/data/`**，不再用 `/home/weiland/step0b_artifacts/`（那是 pi0.5 阶段的临时目录）。`.gitignore` 默认忽略 `exp/**/data/**`（章程 §3），本实验**不申请**白名单例外——结果需跨机共享时再按章程 §3 走 `!exp/...` 显式豁免。

⚠ **与 Index Sync Rule 的一致性**：本 plan 的落地 commit 需同时包含 `logs/README.md` 的索引更新（该更新已在起草时写入工作树），避免分两次提交造成索引失步（WA §4 宪法红线）。

### 4.1 `exp/robocasa365/groot_data_config.py`（新增，约 70 行）

定义 `RoboCasa365DataConfig`。**四类键的有序列表全部逐字取自 F18 的官方 N1.5 `new_embodiment` 配置**，不再依赖父类继承或语义推定：

```python
# 示例仅示意「四类键均须显式声明且有序」；落地实现按 §5.0-1 引用
# exp/robocasa365/groot_keys.py 的常量，不在本文件内重复硬编码。
class RoboCasa365DataConfig(SinglePandaGripperDataConfig):
    # All four ordered lists are transcribed verbatim from the official N1.5
    # `new_embodiment` config: robocasa@be22d659
    # docs/datasets/using_datasets.md:167-210 (see plan F18).
    video_keys = [
        "video.robot0_agentview_left",
        "video.robot0_agentview_right",
        "video.robot0_eye_in_hand",
    ]
    state_keys = [
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.gripper_qpos",
        "state.base_position",
        "state.base_rotation",
    ]
    action_keys = [
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
        "action.base_motion",
        "action.control_mode",
    ]
    # NOTE: the parent declares the `.action.` variant, which belongs to N1.7's
    # ROBOCASA_PANDA embodiment, NOT to N1.5 `new_embodiment`. Must be overridden.
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))
```

⚠ **`state_keys` / `action_keys` 虽与父类同序，仍显式写出**：一是让顺序契约在本文件内可读、可测；二是隔离上游漂移（父类若改序，本 config 不受影响，且 §5.2 的对照测试会红）。

**证据等级**：

| 项 | 采用值 | 证据 | 等级 |
|---|---|---|---|
| `video_keys` 顺序 | left → right → eye_in_hand | **F18 官方 N1.5 配置逐字** | 🟢 **直证** |
| `state_keys` 顺序 | eef_pos_rel → eef_rot_rel → gripper_qpos → base_pos → base_rot | **F18** | 🟢 **直证** |
| `action_keys` 顺序 | eef_pos → eef_rot → gripper_close → base_motion → control_mode | **F18** | 🟢 **直证** |
| `language_keys` | `annotation.human.task_description` | **F18 + F19 双源一致**（⚠ 不可用父类或 N1.7 的值） | 🟢 **直证** |
| `action_indices` | `range(16)` | **F18** `delta_indices=list(range(16))` + F6 | 🟢 **直证** |
| `state_*_modes` / `state_target_rotations` / `action_*_modes` | 继承父类（min_max / rotation_6d / binary） | F6；**F18 未覆盖归一化配置** | 🟡 **仍为推定** |

⇒ **顺序问题已由 F18 关闭，不需要任何实证搜索**。**唯一残留的推定是归一化 / 旋转表示**，由 §5.2 的对照测试 + §9.1 的 G0-B 行为门共同守护。


### 4.2 `exp/robocasa365/groot_policy_adapter.py`（新增，约 90 行）

**存在理由**：F13——`WebsocketPolicyServer` 要求 `infer()`，`Gr00tPolicy` 只有 `get_action()`。

⚠ **关键**：**adapter 不得**把 `prompt` 映射为 `annotation.human.action.task_description`（F18/F19）——N1.5 `new_embodiment` 用的就是 env 原生的 `annotation.human.task_description`。**adapter 不做任何键名映射**，wire 上直接使用 GR00T/env 共有的键名。

`GrootPolicyAdapter` 的契约：

| 项 | 约定 |
|---|---|
| `__init__(self, policy)` | 依赖注入，接受任意实现 `get_action()` 的对象（生产传 `Gr00tPolicy`，测试传 fake） |
| `infer(obs: dict) -> dict` | 唯一公开方法，签名与 `BasePolicy.infer` 一致 |
| **输入键（wire 格式 = GR00T 键名，无映射）** | `video.robot0_agentview_left` / `..._right` / `video.robot0_eye_in_hand` 各 `(H,W,3) uint8`；`state.*` 五键（§4.1 顺序）各 `(D,) float`；`annotation.human.task_description` 字符串 |
| 内部 | ①校验键齐全、dtype、shape；②**有限值门禁**：所有 state 数组须 `np.isfinite(...).all()`，否则 `ValueError`；③各 modality 前置 `T=1` 时间轴 → `(1,H,W,3)` / `(1,D)` / `(1,)`；④`policy.get_action(...)` |
| 输出 | `{"actions": {<5 个 action.* 键>: np.ndarray[16, D]}}` —— **只有这一个键**。`server_timing` 由 `WebsocketPolicyServer` 在 `infer()` 返回**之后**注入，adapter 不自行产生 |
| **输出门禁** | 返回前校验每个 `action.*` 数组 `np.isfinite(...).all()`；出现 NaN/Inf 即 `ValueError`（不把坏动作送回 client 去驱动机器人） |
| 异常契约 | 缺键 / dtype 不符 / shape 不符 / 非有限值 → 抛 `ValueError` 并**列出具体键名**；**绝不静默补零或裁剪**（静默修复会退化成 F11/F21 那类静默错误） |
| 异常传播 | `policy.get_action()` 自身抛出的异常**不吞**，原样上抛 |

⚠ `server_timing` 等非动作字段与 `actions` 分离，client 按键取用，不得把整个返回值当动作数组。


### 4.3 `exp/robocasa365/serve_groot_n15.py`（新增，约 90 行）

孤岛 B 内的 server。构造 `RoboCasa365DataConfig` → `Gr00tPolicy`（F1，`embodiment_tag="new_embodiment"`，路径见 §1.3）→ `GrootPolicyAdapter` → `WebsocketPolicyServer(policy=adapter, host="0.0.0.0", port=8020)`。

**启动握手自检**（失败即退出，不带病启动）：

① 打印 `modality_config()` 的四类键**有序列表**，并断言与 §4.1（即 F18 官方配置）**逐项列表相等**；
② 断言 `len(action_indices) == 16`；
③ 用一份**合法** dummy obs 跑通一次 `infer()`；
④ 断言输出各 `action.*` 满足 `np.isfinite(...).all()`，并打印 shape/dtype；
⑤ 打印所加载 checkpoint 的绝对路径。

⚠ **dummy obs 必须合法，不能全零**：`state.end_effector_rotation_relative` 与 `state.base_rotation` 是**四元数**，全零不是合法旋转，会让 quaternion→matrix→`rotation_6d` 路径产生无效值/NaN——若判据只打印 shape/dtype，**既可能放过坏自检，也可能让正常服务无条件启动失败**。规定：

| dummy 字段 | 取值 | 依据 |
|---|---|---|
| `state.end_effector_rotation_relative` (4) | checkpoint `mean` 经 **L2 归一化** | F21/F24：多个四元数的均值不是单位模长，而非单位四元数不是旋转；取均值再归一化可同时保证**在分布内**与**几何合法**。仅当均值退化为零向量时回退到 `[1,0,0,0]` |
| `state.base_rotation` (4) | 同上 | 同上 |
| 其余三个 state 字段 | 取 checkpoint `metadata.json` 中该键 `mean` 向量（**保证落在合法域内**） | F7 统计量齐备 |
| `video.*` (3 路) | `zeros((256,256,3), uint8)` | 图像零值合法（黑图），不经旋转变换 |
| `annotation.human.task_description` | 固定短句 | F18 |



**file-level docstring 必须写明**（应 owner 要求，审查者不必自行查找）：模型绝对路径、所属 venv 绝对路径、主机名 `weilandserver`、端口 8020、`PYTHONPATH` 需求、以及"为何不能放主 venv"的原因。

### 4.4 `exp/robocasa365/groot_rollout_client.py`（新增，约 130 行）

孤岛 A 内的 rollout client。以 `/home/weiland/step0b_artifacts/step0b_v2.py` 为模板，删去其 pi0.5 专用转换（16 维 state 拼接、`resize_with_pad(224)`、`convert_action`），改为：按 §4.1 的三个相机键取图 → **渲染 512 后降采样到 256**（F12 的官方做法）→ 连同 5 个 state 键与 prompt 发给 server → 收到 `actions` dict → 逐步喂 `env.step()`（F8 键名已对齐）。

保留：场景钉定、A/B 双场景循环、**逐任务落盘到 `exp/robocasa365/data/`**。

新增（使 A/B 真正可称 paired）：`episode_seeds(n_trials, base_seed)` 生成固定 seed 列表，**两个场景回放同一列表**（`env.reset(seed=...)`），seeds 写入结果 JSON 与每行日志；`--base-seed` 可调。⚠ 若两侧各自用无 seed 的 `reset()`，物体摆放与初始状态是两次独立随机抽样，A/B 数字**不能**当配对样本分析。

健壮性（无人值守长跑所需）：`--max-consecutive-failures`（默认 3）连续失败熔断并写出部分结果后非零退出；结果 JSON 经临时文件 + `os.replace` **原子写**（`write_text` 会先 truncate，崩在窗口内将丢失全部已完成任务）；错误摘要取 traceback **末行**（server 回传的是完整 traceback，前缀全是 websocket 样板）；`--n-trials` 与 scene 格式在 argparse 后 **fail-fast**；客户端读取 server metadata，若 server 开着 `--diagnostic-seed` 则**拒绝采集**（除非 `--allow-diagnostic-server`）——该模式钉死 flow-matching 噪声，会让成功率不再无偏且事后无法从数据发现。

**file-level docstring 必须写明**：所属 venv 绝对路径、三个 EGL 环境变量、必需工作目录、server 地址端口。

### 4.5 孤岛 B 依赖补齐（F14，可复现命令）

```bash
# 孤岛 B 的 numpy 为 1.26.4，满足 openpi-client 的 numpy<2.0.0 约束，
# 故此处不需要（也不应）加 --no-deps —— 与孤岛 A 的处置相反。
export VIRTUAL_ENV=/home/weiland/gr00t_n15_venv/.venv
uv pip install -e /home/weiland/openpi/packages/openpi-client   # 实测：装入 openpi-client + svgwrite/tree/websockets
# decord：父类 DataConfig 的 transform 链经 transformers 动态模块加载它，
# 缺失会让 RoboCasa365DataConfig() 直接 ImportError（实测）。
uv pip install decord
# pytest：孤岛 B 默认没有，manual 契约测试需要。
uv pip install pytest
# ⚠ gr00t 不在 venv 里，来自 n1.5-release worktree；三段路径缺一不可。
export PYTHONPATH=/home/weiland/gr00t_n15:/home/weiland/openpi/src:/home/weiland/openpi
```

⚠ 上述四项均已实测：`decord`(1 包) / `pytest`(4 包) 的安装**都不触碰 numpy / torch / transformers**（安装后复验 numpy 1.26.4、torch 2.5.1+cu124 不变）。

⇒ **依赖边界**：本 plan **会 import 本仓库 `src/openpi/serving/websocket_policy_server.py` 与 `packages/openpi-client`**，但**不修改**它们。原文"零 import 本仓库 src/"的表述作废，改为"零修改"。


## 5. 测试策略

> 分三层，并**为可测性调整了设计**（§5.0）。⚠ **关键约束**：新增的三个运行文件必须有**真正的非 manual 覆盖**——不可因其依赖 `gr00t`/GPU/仿真就把全部用例标成 `manual`，那等于零自动覆盖。

### 5.0 为可测性所做的设计调整（前置条件）

非 manual 测试之所以可能，靠两点结构安排：

1. **键顺序常量下沉到不 import `gr00t` 的模块** `exp/robocasa365/groot_keys.py`（纯常量 + 纯函数，无第三方依赖）。`groot_data_config.py`（import `gr00t`）**引用**这些常量而非各自硬编码。
2. **`GrootPolicyAdapter` 用依赖注入**：`__init__(self, policy)` 接收任意实现 `get_action()` 的对象。生产环境传 `Gr00tPolicy`，测试传 fake。⇒ adapter 的**全部**转换逻辑可在主 venv 测试，无需 GPU / 权重 / 仿真。

### 5.1 Layer 1 — 非 manual，主测试环境运行，CI 覆盖

`tests/robocasa365/test_groot_obs_adapter.py` 与 `test_groot_rollout_client.py`。**不 import `gr00t`、不需 GPU、不需仿真。**

🟢 **实测：非 manual 共 72 passed**（另有 2 skipped = 两个 manual 模块在主环境被 `importorskip` 整体跳过，属预期）。下表为**契约面清单**，非逐条对应测试函数名。

⚠ **两条自证陷阱已显式规避**（fixture 会用被测常量构造数据，故凡是"数字"都必须另行锚定到字面量）：①`STATE_DIMS`/`ACTION_DIMS`/两个分辨率常量各有独立的字面量对照测试；②`QUATERNION_STATE_KEYS` **列表本身**被钉死——只遍历该列表的测试在列表为空时会空转通过，等于守卫被它所守卫的值关掉。
⚠ **动作时序顺序**单独断言（`iter_step_actions` 的第 k 步必须是 chunk 第 k 行）：把循环改成 `reversed()` 不改变任何形状、键名与计数，却会让机器人倒序执行。

| 用例 | 断言 |
|---|---|
| `test_video_key_order_matches_official_config` | `VIDEO_KEYS` **列表相等**于 F18 官方 N1.5 配置（期望值硬编码在测试内） |
| `test_state_key_order_matches_official_config` | `STATE_KEYS` 列表相等于 F18 |
| `test_action_key_order_matches_official_config` | `ACTION_KEYS` 列表相等于 F18 |
| `test_language_key_is_env_native_not_n17_variant` | `LANGUAGE_KEYS == ["annotation.human.task_description"]`（F18/F19）。⚠ **显式断言它不等于** N1.7 的 `annotation.human.action.task_description` —— 该误用极易发生（父类声明的就是这个值），用测试钉死防回归 |
| `test_action_keys_match_env_contract` | `set(ACTION_KEYS)` == `convert_action()` 产出键集合（F8） |
| `test_wire_keys_are_not_renamed` | adapter 的输入键名与 GR00T 键名**逐字相同**（本 plan 不做任何键名映射） |
| `test_build_obs_adds_time_axis` | `(H,W,3)`→`(1,H,W,3)`、`(D,)`→`(1,D)`、语言 →`(1,)` |
| `test_build_obs_rejects_missing_key` | 缺任一键 → `ValueError` 且消息含缺失键名；**不静默补零** |
| `test_build_obs_rejects_wrong_dtype_or_shape` | 图像非 `uint8` / 非 3 通道 / state 维度不符 → `ValueError` |
| **`test_rejects_non_finite_input`** | state 含 `NaN` 或 `Inf` → `ValueError`（G1 R3 意见 3 要求的输入有限值门禁） |
| **`test_rejects_non_finite_output`** | fake policy 返回含 `NaN` 的动作 → `ValueError`，**不把坏动作回传 client** |
| **`test_dummy_obs_uses_unit_quaternion`** | §4.3 自检所用 dummy 的 `state.end_effector_rotation_relative` 与 `state.base_rotation` **等于 `[1,0,0,0]`**（F21 的 wxyz 约定），且**断言其不为全零** —— 钉死 G1 R3 意见 3 的缺陷 |
| `test_infer_isolates_non_action_fields` | fake policy 返回夹带 `server_timing` → `actions` 只含 `action.*` |
| `test_action_chunk_slicing` | chunk 长 16、`replan_steps=5` → 取前 5 步且顺序不变 |
| `test_replan_steps_must_not_exceed_horizon` | `replan_steps=20 > 16` → `ValueError` |
| `test_adapter_propagates_policy_exception` | fake policy 抛异常 → adapter 不吞 |

⚠ 该文件**顶层不得 import `gr00t`**（见 §5.2 的收集期说明）。所需常量来自 `exp/robocasa365/groot_keys.py`（纯常量模块，无第三方依赖）。

### 5.2 Layer 2 — manual，分两个孤岛

**⚠ 两个 manual 文件必须分开**，因为 `gr00t` 与 `robocasa` 分属互斥的两个解释器。

**(a) `tests/robocasa365/test_groot_data_config_manual.py` — 孤岛 B**（需 `gr00t` + checkpoint metadata，不加载权重、无 GPU）

| 用例 | 断言 |
|---|---|
| `test_data_config_matches_official_ordered_config` | 四类键**列表相等**于 `groot_keys`，且 `len(action_indices)==16` |
| `test_language_key_overrides_parent` | 覆盖值 ≠ 父类的 N1.7 变体，且 == `annotation.human.task_description` |
| `test_modality_config_keys_and_order` | `modality_config()` 四类键**有序**相等（这才是驱动 `ConcatTransform` 的对象） |
| `test_modality_config_covers_checkpoint_keys` | 三类键集合 == checkpoint `metadata.json`。⚠ 不含 language：robocasa 的 `DatasetModalities` schema 无 annotation 字段，metadata 结构上带不了该键 |
| `test_state_and_action_dims_match_checkpoint` | 逐键宽度 == 统计量 `mean` 长度 |
| `test_statistics_have_matching_mean_and_std_widths` | `mean`/`std` 宽度一致（截断或补零的 stats 块正是要防的） |
| `test_checkpoint_embodiment_tag_is_new_embodiment` | tag 正确 |
| `test_unit_quaternion_survives_rotation_transform` | `[1,0,0,0]` 过 pytorch3d 得单位阵且有限 |
| `test_transform_chain_constructs` | **`transform()` 可构造** —— `modality_config()` 是自证的（握手比对的就是它自己传进去的对象），构造 transform 才能证明父类接受我们的键覆盖、且 `decord` 等运行期依赖到位 |

**(b) `tests/robocasa365/test_env_action_contract_manual.py` — 孤岛 A**（需 `robocasa`）

补上非 manual 层做不到的那一环：Layer 1 只能把我们的键与**抄录值**比对（自洽但检测不到 env 侧漂移），这里调用**真实的 `convert_action`**：

| 用例 | 断言 |
|---|---|
| `test_action_keys_match_real_convert_action` | `set(convert_action(...))` == `ACTION_KEYS` |
| `test_action_dims_match_real_convert_action_slices` | 每键切片宽度与 `ACTION_DIMS` 一致 |
| `test_total_action_width_is_12` | 总宽 12 |

⚠ **CI 收集期保护**（两个文件同此）：`@pytest.mark.manual` 的跳过发生在**模块导入之后**，若模块顶层 import 可选依赖，主环境会在 **collection 阶段**就 `ImportError`。故强制：所有可选依赖 import **写在测试函数内部** + 文件顶部 `pytest.importorskip(...)` 兜底。

运行命令（**四处易错点均已实测**）：
```bash
# 孤岛 B：① gr00t 不在该 venv，来自 n1.5-release worktree ② decord（transform 链需要）
#         ③ pytest 该 venv 默认没有 ④ conftest.py 默认跳过 manual，须 --run-manual
VIRTUAL_ENV=/home/weiland/gr00t_n15_venv/.venv uv pip install pytest decord
cd /home/weiland/openpi && \
PYTHONPATH=/home/weiland/gr00t_n15:/home/weiland/openpi/src:/home/weiland/openpi \
  /home/weiland/gr00t_n15_venv/.venv/bin/python -m pytest \
  tests/robocasa365/test_groot_data_config_manual.py --run-manual -q

# 孤岛 A：pytest 已在该 venv 内（实测无需安装）
cd /home/weiland/openpi && PYTHONPATH=/home/weiland/openpi \
  /home/weiland/Isaac-GR00T/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python \
  -m pytest tests/robocasa365/test_env_action_contract_manual.py --run-manual -q
```
⚠ 若沿用旧命令（缺 worktree 路径或缺 `--run-manual`），结果会是**假跳过**而非失败——`importorskip` 静默走跳过分支，看上去"通过"。

非 manual 套件（主环境 / CI 口径）：
```bash
uv run pytest tests/robocasa365/ -q
```

### 5.3 Layer 3 — manual，孤岛 A+B 联跑（需仿真 + GPU）

| 步骤 | 判据 |
|---|---|
| S1 启动握手 | server 打印 `SERVER-LISTENING`、四类键**有序列表**、checkpoint 绝对路径；显存增量 7–8 GB |
| S2 单步推理 | **合法** dummy obs（含单位四元数 `[1,0,0,0]`，F21）跑通 `infer()`，返回 5 个 `action.*`，各首维 = 16，且全部 `np.isfinite` |
| S3 单 episode 闭环 | 跑满 horizon 无异常；动作落在 `Box(-1,1)` |
| S4 冒烟 SR | **仅作 sanity check**。链路正确性由 §9.1 的 **G0-A（接线等价性，确定性）** 举证，归一化推定由 **G0-B（行为健全性，统计）** 举证；S4 不承担任何举证责任 |

⚠ **诚实声明**：Layer 2/3 无法进 CI（依赖 `gr00t`、权重、仿真、GPU）。本 plan **不通过 mock 伪造这部分覆盖率**；Layer 1 覆盖的是全部可在无 GPU 环境判定的转换逻辑，Layer 2 用"列表相等"守护 Layer 1 所依赖的推定。


## 6. 风险登记

> R1/R8 已因 F18 直证而关闭，保留条目以记录"为何不再需要顺序搜索"。R2 收窄为仅归一化/旋转仍属推定。

| ID | 风险 | 影响 | 缓解 |
|---|---|---|---|
| **R1** | ~~`video_keys` 顺序未知~~ **已关闭** | — | **F18 给出官方 N1.5 `new_embodiment` 完整有序配置**，video/state/action 三类顺序均为直证，且与此前推定一致 ⇒ 风险消解，**6 选 1 实证搜索取消**。保留 §5.1/§5.2 的列表相等测试防回归 |
| **R8** | ~~`state_keys` / `action_keys` 顺序未知~~ **已关闭** | — | 同 R1，F18 直证。由 §5.2 `test_data_config_matches_official_ordered_config` 守护 |
| **R2** | **归一化模式（`min_max`）与旋转表示（`rotation_6d`）仍是推定** —— F18 只给了键与顺序，**未覆盖归一化配置** | 高——动作尺度或旋转语义错误，且属静默错误 | **§9.1 G0-B 的针对性负对照**：N1（改 `mean_std`）、N2（关 `rotation_6d`）、B（常数预测基线），paired common seeds + 逐 action-key 配对 Wilcoxon。⚠ **不可**用"动作落在 `Box(-1,1)`"来证明归一化正确——范围正确的动作完全可以是语义错位的动作。若 G0-B 无判别力，记为**未确认**而非成立 |
| **R9** | **language key 用错**。⚠ 直觉容易反向理解成"忘记映射"，**实际相反**：F18/F19 证明 N1.5 `new_embodiment` 用的就是 env 原生 `annotation.human.task_description`；而 `annotation.human.action.task_description` 属 **N1.7 `ROBOCASA_PANDA`**（父类正是该值，故极易误用） ⇒ 真正风险是**多做一次本不该做的映射** | 高——键不匹配则模型收不到语言指令 | §4.2 改为**不做任何键名映射**；§5.1 `test_language_key_is_env_native_not_n17_variant` 显式断言不等于 N1.7 变体；§5.2 `test_language_key_overrides_parent` 确认父类值被覆盖 |
| **R11** | **`get_action()` 含随机采样**（F20：`flow_matching_action_head.py:363-368` 每次以 `torch.randn` 为去噪起点）⇒ 单次调用的比较不可复现 | 高——会得出不可重复的结论，或把随机波动误读为配置错误 | ①接线检验 **G0-A** 用 `torch.manual_seed` 固定种子，使三条路径可 bit 级比对；②性能检验 **G0-B** 用 paired common seeds + 多次重复 + 逐 action-key 配对检验；③**不可**把该比较称作"确定性 parity" |
| **R12** | **非法 dummy 输入**：全零四元数不是合法旋转，quaternion→matrix→`rotation_6d` 会产生 NaN | 中——坏自检可能被放过，或正常服务被无条件挡下 | §4.3 规定 dummy 的两个四元数字段取 **`[1,0,0,0]`**（F21 实测的 wxyz 约定），其余 state 取 metadata `mean`；adapter 输入/输出双向 `np.isfinite` 门禁；§5.1 三个非 manual 用例钉死 |
| **R13** | **CI 收集期失败**：`@pytest.mark.manual` 在模块导入后才生效，顶层 `import gr00t` 会让主环境在 collection 阶段就 `ImportError` | 中——CI 直接红，且与被测逻辑无关 | §5.2 强制两条：`gr00t` import 写在测试函数内 + 文件顶部 `pytest.importorskip("gr00t")` 兜底；Layer 1/Layer 2 拆为**独立文件** |
| **R3** | 图像分辨率 | 低 | F12 给出官方做法：`CAMERA_RESOLUTION=512` 渲染 → `FINAL_IMAGE_RESOLUTION=(256,256)` 降采样 |
| **R4** | 测试覆盖 | 低 | §5 三层；Layer 1 **实测 72 个非 manual 用例**、CI 可跑（靠常量下沉 + 依赖注入 + 注入 sim 假件）；Layer 2 分两个孤岛：孤岛 B **10 passed**（DataConfig/metadata/transform 链的 concat order 实证）、孤岛 A **3 passed**（真实 `convert_action`）。不通过 mock 伪造 GPU/仿真覆盖率 |
| **R5** | websocket 传 3×256×256×3 uint8 ≈ 590 KB/次 | 低 | 同机回环；pi0.5 已验证同构链路可用 |
| **R6** | 4090 显存：GR00T ~7–8 GB + sim client + 他人常驻 8.8 GB | 中——OOM 会连累他人任务 | 起服务前 `nvidia-smi` 确认余量 ≥ 15 GB；只在自己的 tmux session 内操作，禁宽模式 `pkill` |
| **R7** | 整条路线不成立 | 高——第二 teacher 无法落在 RoboCasa365 | §9 退路（含出场门判据、时间盒、三个退路选项、以及退到 LIBERO 的科学后果） |
| **R10** | **G0-B 数据不可得**：RoboCasa365 的 LeRobot 训练数据不在本机（F17） | 中——失去归一化推定的判别手段 | §9.1 G0-B 声明：数据不可得则直接判本线失败转退路，**不退回用 SR 凑证据**。⚠ 注意 **G0-A 不依赖数据**，仍必须通过 |


## 7. 集成点

- **与既有代码**：**零修改**，但**有只读 import**（⚠ 二者不同，勿写成"零 import"）。实际依赖：`src/openpi/serving/websocket_policy_server.py` 与 `packages/openpi-client`（均只读引用，F15 证明其 import 面不含 jax/torch，可安全跨环境）。不修改 `training/config.py`（pi0.5 侧已验证：fork 的 robocasa config 在模块顶层 `import robocasa`，塞进主 venv 会破坏 config 导入）。
- **与后续 cache 工作**：`serve_groot_n15.py` 是未来 GR00T 版 Interceptor 的挂载点——cache 内核已实测可装进孤岛 B（47 模块 44 通过，仅 `interceptor`(缺 jax) / `sidecar_executor`(缺 websockets) / `qdrant_backend`(可选) 未过，且无版本冲突）。本 plan **不实现** cache 接入。
- **与实验设计**：产出用于 GR00T 侧的 per-task 可用性筛查（K=3，18 任务 × 2 场景 = 108 ep），与 pi0.5 侧清单取交集后确定正式评测任务子集（⚠ 与 benchmark 自带的 task set `atomic_seen` 不是一层）。

---

## 8. 明确不做

- 不训练、不微调（checkpoint 已是 RoboCasa365 训练产物，F7 + `global_step=120000`）
- 不修改 `Isaac-GR00T` / `robocasa` 源码
- 不修改本仓库任何既有**代码**文件（`logs/README.md` 索引按 WA §4 宪法红线必须同 commit 更新，属例外）
- 不接入 cache
- 不跑正式跨场景实验（本 plan 只交付链路 + 冒烟）

---

## 9. 退路（N1.5 线走不通时）— owner 于 2026-08-16 要求纳入

> owner 原话：「如果 1.5 这条线崩了我们只能去用 1.7，然后在 libero spatial 和 libero 10 上微调 1.7」。
> **查证后需修正其中一个前提：大概率不必自己微调。**（依据见 W1/W2）

### 9.1 出场门：G0-A 接线等价性（确定性）+ G0-B 行为健全性（统计）

> ⚠ **两条不可用的判据（务必避开）**：①"每排列跑 3 ep、`SR > 0` 即算通过"——rollout SR 噪声压过信号，且把任务随机性与配置选择混在一起；②"确定性 MSE parity + 顺序搜索"——`get_action()` 本身随机（F20），且顺序已由 F18 直证、无需搜索。

**两点前提**：

1. **顺序搜索已取消**。F18 给出官方 N1.5 `new_embodiment` 的完整有序配置 ⇒ video/state/action/language 四类顺序均为直证，**不需要 6 选 1 搜索**。⚠ 亦不可采用 `m_(2)/m_(1) >= 2.0` 之类固定倍率阈值或 time-shuffle baseline：这些阈值无理论依据，且正确配置未必产生 2× 间隔、平滑动作上错误配置也可能优于 shuffle。
2. **MSE 不是确定性量**。F20：`flow_matching_action_head.py:363-368` 每次以 `torch.randn` 为去噪起点 ⇒ 同一输入两次调用结果不同。故**不可**称之为"确定性 parity"。

⇒ 据此把**接线等价性**与**模型性能**拆成两个独立的门（二者不可混为一谈）：

#### G0-A — 接线等价性（wire adapter parity）：**确定性，可 bit 级判定**

**测什么**：adapter + websocket 这条链路有没有改变语义。**与模型好坏无关。**

**方法**：取同一份**合法** obs（构造同 §4.3 的 dummy 规范，含单位四元数 `[1,0,0,0]`），在**同一随机种子**下分别求：
- 路径 ①：孤岛 B 内直接 `torch.manual_seed(S); policy.get_action(obs_with_time_axis)`
- 路径 ②：`torch.manual_seed(S); adapter.infer(wire_obs)`（同进程，不过网）
- 路径 ③：经 websocket 往返的 client 侧结果（跨进程）

**判据**：①②③ 三者的每个 `action.*` 数组 `np.allclose(..., rtol=0, atol=1e-6)`。**任何不一致都是接线 bug**，与随机性无关（种子已固定，flow matching 可复现）。
⚠ 若①②一致但③不同 ⇒ 序列化/传输层问题（msgpack dtype 降精度等），定位明确。

**为什么这一步是确定性的**：`torch.randn` 取自全局 RNG，`manual_seed` 后可复现；三条路径喂给模型的张量若真的相同，输出必然相同。

⚠⚠ **种子必须在 server 进程内重置**：client 与 server 是**两个解释器**，client 侧调 `torch.manual_seed` **不会**传播到 server 的全局 RNG——若只在 client 设种子，路径③的结果将不可复现，会把随机波动误判成接线 bug。实现规定二选一：
- **(a) 诊断钩子（推荐）**：server 在**紧邻该次 `get_action()` 调用之前**重置种子。该钩子仅在诊断模式下启用（通过启动参数开启），正式 rollout 不得开启——否则每步都用同一噪声起点，会人为降低动作多样性。
- **(b) 专用 server 实例**：为 G0-A 单独启动一个进程，进程启动时设定固定种子，且**全程只发一次 infer 请求**（多次请求会推进 RNG 状态，破坏可比性）。

路径①②在孤岛 B 同一进程内，直接在调用前 `manual_seed` 即可。

#### G0-B — 行为健全性（behavioral sanity gate）：**统计判定，非确定性**

**测什么**：§4.1 表中**唯一残留的 🟡 推定**——归一化模式（`min_max`）与旋转表示（`rotation_6d`）是否与训练一致。顺序已由 F18 关闭，不在此门的检验范围。

**数据清单（固定、预先声明）**：官方 RoboCasa365 LeRobot 数据的 **2 个任务 × 各 5 条 episode**，任务取 pi0.5 侧 U2 类中的 `PickPlaceCounterToStove` 与 `OpenCabinet`（两场景均高 SR，代表"teacher 确实会做"的任务）。落 `exp/robocasa365/data/parity/`。数据不可得则按 R10 直接转 §9.2。

**采样与配对**：
- 固定种子集合 `S = {0, 1, 2, 3, 4}`（5 次重复）。
- **paired common seeds**：所有对照组在**同一 `(episode, timestep, seed)` 三元组**上取样，逐点配对，消除随机起点带来的方差。
- 指标**逐 action-key 分别报告**（5 个键量纲不同，不可混合平均），每个键在其自身归一化尺度上算 MSE。

**对照组（针对性负对照，取代无依据的 shuffle baseline）**：

| 组 | 配置 | 期望 |
|---|---|---|
| **P（正）** | §4.1 配置（`min_max` + `rotation_6d`） | MSE 最低 |
| **N1（负）** | 归一化模式改为 `mean_std` | 若 P 显著优于 N1 ⇒ `min_max` 推定成立 |
| **N2（负）** | 关闭 `state_target_rotations`（四元数直送，不转 `rotation_6d`） | 若 P 显著优于 N2 ⇒ 旋转表示推定成立 |
| **B（参照基线）** | 不调模型，直接预测该数据集的**动作均值**（常数预测器） | 其 MSE ≈ 动作方差；**P 必须显著低于 B**，否则模型根本没在预测 |

**统计的独立分析单位（预注册，避免伪显著）**：⚠ **相邻 timestep 不是独立样本**（动作轨迹高度自相关），直接按 timestep 做检验会虚增样本量、制造伪显著。故规定两级聚合：

1. **第一级**：在每条 episode 内，对该 episode 的所有 timestep × 所有 seed 求平均，得到该 episode 的**单一标量** MSE（逐 action-key 分别求）。
2. **第二级**：以 **episode 为配对单位**做 Wilcoxon 符号秩检验（本设计 2 任务 × 5 episode = **n = 10 个配对单位**）。
3. 检验在**每个 action key 上独立进行**，不跨 key 合并。

⚠ **近零方差 action key 的处置**：`action.control_mode` 与 `action.gripper_close` 在多数 episode 内可能近乎常数 ⇒ 基线 B（常数均值预测器）本身就已接近最优，**任何模型都不可能显著优于它**。故预先规定：若某 action key 在 parity 数据上的 ground-truth 方差低于阈值（该 key 归一化尺度下方差 < 1e-3），则**将其排除出"必要条件"的判定**，仅报告其数值、不参与 pass/fail —— 否则会因统计功效不足而误杀正确配置。被排除的 key 须在产物中显式记录。

**产物**：上述聚合方式、n、被排除的 key、各组逐 key 的 episode 级 MSE 与检验 p 值，全部写入 `exp/robocasa365/data/parity/` 下的结果 JSON，**在看到结果前即固定**。

**判据（预注册）**：

1. **必要条件**：P 在**每一个未被方差阈值排除的** action key 上都显著低于 B（配对 Wilcoxon 符号秩检验，`p < 0.01`，n = 10）。不满足 ⇒ 链路有根本问题，转 §9.2。
2. **归一化确认**：P 显著低于 N1 且显著低于 N2（同检验，`p < 0.01`）⇒ 两处 🟡 推定确认，G0-B PASS。
3. **不确定情形**：若判据 1 通过、但 P 与 N1/N2 无显著差异，说明该数据规模下对归一化无判别力 —— **不得据此宣称推定成立**，记为"**未确认**"。
   ⚠ **"未确认"不触发 §9.2 退路，可继续推进**：判据 1 已证明模型确实在有效预测，归一化即便未被正面确认，也未被证伪。处置为：①在 log 中显式标注该残留不确定性；②把归一化列为 §5.3 S4 rollout 的重点观察项；③若 S4 的 SR 明显低于 pi0.5 在同任务的水平，回到本门以更大数据量重测。**只有判据 1 或判据 4 失败才转 §9.2。**
4. **反向情形**：若 N1 或 N2 显著优于 P ⇒ 继承的归一化/旋转配置是错的，按其方向修正 §4.1 后重跑 G0-B。

⚠ **明确区分**：G0-B 是**模型性能检查**，不承担接线正确性的举证责任（那是 G0-A 的职责）。**二者不可混为一谈**——把性能指标当接线证据，正是本设计要规避的错误。

#### G0-C 时间盒

G0-A（无需数据，仅需权重）**不设时间盒，必须通过**——它是纯接线检验，失败即代码 bug，修到通过为止。
G0-B 的数据获取 + 四组评估合计**不超过一个工作日**；超出即判定本线失败转 §9.2。
⚠ 不得因"再试一下"无限延长——pi0.5 侧已有先例：`TurnOnMicrowave` 连续 0/6 曾被误判为接线错误，实为任务本身难。

#### G0-D 与 rollout 的关系

**G0-A 必须通过**（它是纯接线检验，无通融）。**G0-B 需判据 1 通过**；判据 2 通过则归一化确认，判据 3（"未确认"）亦可放行进入 rollout（见上），判据 1/4 失败则转 §9.2。
其后进入 §5.3 的 S3/S4；S4 的冒烟 SR 此时只作 sanity check，不承担"证明配置正确"的职责。


### 9.2 查证到的关键事实（**修正 owner 的成本假设**）

**W1 — 官方已发布 N1.7 的 LIBERO 微调 checkpoint，且四个 suite 全覆盖**：
- `getting_started/policy.md:68` 登记 `LIBERO_PANDA → nvidia/GR00T-N1.7-LIBERO`
- `examples/LIBERO/README.md:13` 明确 "**All four suites were finetuned** with the same hyper-parameters"（即 spatial / object / goal / 10）
- `scripts/deployment/README.md:39-56` 给出下载与推理命令，checkpoint **按 suite 分目录**（示例用 `checkpoints/GR00T-N1.7-LIBERO/libero_10`）

⇒ **owner 设想的"在 libero spatial 和 libero 10 上微调 1.7"这一步大概率可以省掉**，直接用官方权重。

**W2 — 若真要自训，官方配方是 8 卡**：`examples/LIBERO/README.md:39` 为
`NUM_GPUS=8 MAX_STEPS=20000 GLOBAL_BATCH_SIZE=640 SAVE_STEPS=1000`，另需 `--state-dropout-prob 0.2`（finetune CLI 默认）。数据 `IPEC-COMMUNITY/libero_{suite}_no_noops_1.0.0_lerobot`（libero_10 仅 635 MB / 379 episodes）。
⚠ 我们只有**单张 4090**，复现该配方需重调 batch/梯度累积/步数，**属于"能训但不等价"**，故仅作最后手段。

**W3 — N1.7 的 gated 依赖会重新触发**：N1.7 backbone 默认 `nvidia/Cosmos-Reason2-2B`（`gr00t/configs/model/gr00t_n1d7.py:40`、`processing_gr00t_n1d7.py:230,858`），该 repo **gated**（选 N1.5 时不触发，退到 N1.7 则触发）。`get_backbone_cls`（`gr00t_n1d7.py:491`）亦接受 `Qwen/Qwen3-VL`，但需对应权重。
⇒ **退路 A2 的前置动作：先申请 Cosmos-Reason2-2B 访问权**，并验证 `GR00T-N1.7-LIBERO` 是否自带全部 backbone 权重（若 processor 仍需回源拉配置，未获批前无法加载）。

**W4 — N1.5 侧另有一条零微调退路**：社区 `youliangtan/gr00t-n1.5-libero-{spatial,object,goal,long}-posttrain`（7.59 GB，`model_type: GR00T_N1_5`，已核对 config 与 `n1.5-release` 代码可加载）。

### 9.3 退路选项对照

| 选项 | 内容 | 微调成本 | 阻碍 | 评价 |
|---|---|---|---|---|
| **A2（首选）** | **N1.7 + 官方 `nvidia/GR00T-N1.7-LIBERO`** | **零** | Cosmos gated 需申请（W3） | 官方权重、四 suite 齐全、N1.7 自带完整 LIBERO eval 栈（`gr00t/eval/sim/LIBERO`、`examples/LIBERO`） |
| A1（备选） | N1.5 + 社区 `youliangtan/…-libero-*-posttrain` | 零 | 社区权重，质量未经官方背书 | 不触发 gated；且可复用本 plan 已建的孤岛 B |
| A3（最后手段） | 自行微调 N1.7 | 高（W2，单卡非等价） | gated + 算力 | 仅当 A2/A1 均不可行 |

### 9.4 ⚠ 所有 LIBERO 退路共有的科学后果（必须让 owner 知悉）

**LIBERO 没有跨场景（厨房布局/风格）split。** 退到 LIBERO 意味着：

- **跨场景继承实验（本项目主命题）将只剩 pi0.5 单 teacher**，第二 teacher 无法为"跨场景继承"提供证据。
- 第二 teacher 在 LIBERO 上回答的是**另一个问题**：「cache 机制对不同 VLA 架构是否通用」——这仍有价值（可支撑单独一条贡献），但**不能替代**「跨场景继承对不同 VLA 都成立」。
- 论文叙事需相应调整：主命题标注为单 teacher 证据，并在 limitation 中说明。

⇒ **因此 §9.1 的时间盒值得花**：在 RoboCasa365 上把 N1.5 打通，其科学价值高于任何 LIBERO 退路。退路是保险，不是等价替代。

---

## Review Log

### G2 Round 1 — Reviewer — NEEDS REVISION — 2026-08-16 09:54 CDT

- [Blocking] [Concern] 在生产路径强制验证完整 action-chunk 契约，并让 client 使用同一受测切片逻辑——reasoning: 获批计划 §4.2/§5.3 S2 要求五个 action 键均为 `(16, D)`，但 `GrootPolicyAdapter.infer()` 当前只检查键存在和 `np.isfinite`，`_handshake()` 也只打印 shape；首维为 15 的输出会被正常返回。独立探针 `test_adapter_rejects_wrong_action_horizon` 已复现 “DID NOT RAISE”。同时生产 `run_one()` 没有调用已被单测覆盖的 `take_action_chunk()`，而是直接 `for offset in range(replan)` 索引，CLI 又只拒绝 `>16`、不拒绝 `<1`，使受测边界与真实路径脱节。须为五键钉死各自维度与 horizon=16、拒绝额外/短/错形状输出，生产 client 复用该校验/切片函数，并增加 wrong-horizon、wrong-width、`replan_steps<=0` 的非 manual 回归测试。
- [Blocking] [Concern] 保证 `run_one()` 在所有退出路径关闭仿真环境——reasoning: `env.close()` 只位于正常返回前；`client.infer()`、action 索引或 `env.step()` 任一抛错时，`main()` 会捕获异常并继续下一个场景/任务，却留下当前 MuJoCo/GPU 环境未关闭，长批次可累积资源直至 OOM。独立探针 `test_run_one_closes_env_when_inference_fails` 已复现泄漏。须用 `try/finally`（并只在 env 成功创建后关闭）覆盖成功、推理失败和 step 失败路径，并加入对应测试。
- [Blocking] [Concern] 让 GR00T manual 契约测试真正可复现并补回计划声明的 checkpoint 对照——reasoning: 主环境运行结果是 adapter `20 passed`、manual 文件仅 `1 skipped`；在计划指定的 `/home/weiland/gr00t_n15_venv/.venv` 中直接运行 suite 失败为 `No module named pytest`，因此目前没有任何真实 `RoboCasa365DataConfig`/`pytorch3d` 测试通过证据。实际 manual 文件还没有读取 checkpoint `metadata.json`，所以 §5.2 的 `modality_config_covers_checkpoint_keys` 没有落地。须给出确实使用孤岛 B 的可执行命令（必要时明确安装 pytest/使用 `uv run --active`），补 metadata 集合/维度对照，并附上该 suite 的通过输出；G0-A/GPU/仿真仍可按工作流留到后续 Verify，但 DataConfig 的无模型 manual suite 应在 G2 闭合。
- [Blocking] [Concern] 修正并暂存 `logs/README.md` 的目标索引条目，使其与已批准计划和真实交付一致——reasoning: 当前条目仍称“仅 `video_keys` 需覆盖”“零格式转换”“三个脚本 + manual 单测”，这些结论已在 G1 被 F18/F19 推翻，也遗漏 `groot_keys.py`、adapter、非 manual 测试与双向有限值门禁；该目标条目目前还未进入暂存区。须仅更新本条（不要卷入同文件其他 session 的改动），准确描述四类有序键、N1.5 原生 language key、时间轴/降采样转换、实际文件及测试层次，并与计划同批暂存。
- [Non-blocking] [Suggestion] 若输出仍称 A/B success rate 为“paired”，为 rollout 增加显式 episode seed 清单并在两个 scene 对同一 episode 使用同一 seed——reasoning: 当前两个环境均调用无 seed 的 `env.reset()`，对象/初始状态随机流没有配对；这不妨碍把本脚本当 teacher 可用性 smoke，但不支持后续把结果当严格配对样本分析。

### G2 Round 2 — Executor — 2026-08-16

- Accepted（意见 1：action-chunk 契约未强制 + 生产路径绕过受测切片）— 两条均属实，探针 `test_adapter_rejects_wrong_action_horizon` 复现的 "DID NOT RAISE" 是真缺陷。新增 `groot_keys.ACTION_DIMS`（五键各自宽度 3/3/1/4/1，合计 12）与 `validate_action_chunk()`：对每个键强制 **精确 `(16, D)`**、拒绝缺键/额外键/非有限值，`GrootPolicyAdapter.infer()` 改为经由它返回 —— 因此 `_handshake()` 现在也会拦下错误长度的块（此前只打印 shape）。生产路径改造：新增 `iter_step_actions()` 封装"校验 + 切片 + 拆成 per-step dict"，`run_one()` 调用它，**不再内联 `for offset in range(replan)`**；CLI 的边界检查也改为用一个 probe chunk 走同一函数，故 `replan_steps <= 0` 现在同样被拒（此前只拒 `>16`）。新增非 manual 回归：`test_adapter_rejects_wrong_action_horizon` / `..._wrong_action_width` / `..._unexpected_action_key` / `test_replan_steps_must_be_positive` / `test_iter_step_actions_yields_per_step_dicts` / `test_iter_step_actions_rejects_bad_replan` / `test_action_dims_cover_every_action_key`。⚠ 修复过程中原 `test_infer_isolates_non_action_fields` 与新的"拒绝额外键"校验冲突，暴露了一处设计歧义：`server_timing` 是 `WebsocketPolicyServer` 附加的，真实 `Gr00tPolicy` 不会返回它，故 policy 返回额外键应当拒绝；该测试改写为 `test_infer_nests_actions_under_their_own_key`，改为断言**输出信封结构**的隔离（`set(result) == {"actions"}`），这才是隔离要求的真实落点。

- Accepted（意见 2：`run_one()` 异常路径泄漏环境）— 属实。`run_one()` 的 episode 循环整体包进 `try/finally`，`env.close()` 移入 `finally`，且置于 `gym.make` 成功之后，覆盖正常返回、`client.infer()` 失败、`env.step()` 失败三条路径。新增 `tests/robocasa365/test_groot_rollout_client.py`（**非 manual**，用 `monkeypatch.setitem(sys.modules, ...)` 注入 gymnasium/robocasa 假件，不需要仿真器）：`test_run_one_closes_env_on_success` / `..._when_inference_fails`（即审查者探针）/ `..._when_step_fails` / `..._rejects_bad_replan_before_touching_env`，另加渲染分辨率与场景钉定参数的断言。

- Accepted（意见 3：manual suite 不可复现 + 缺 checkpoint 对照）— 属实，且实跑后发现问题比意见描述的更深，**plan §1.2 对孤岛 B 的描述是不完整的**。实测三处缺口：①**`gr00t` 根本没装进孤岛 B 的 venv**（`import gr00t` → `ModuleNotFoundError`，site-packages 无该包也无 .pth），它来自 **git worktree `/home/weiland/gr00t_n15`（detached HEAD `4af2b62`，已核对 == `n1.5-release`）**，必须经 `PYTHONPATH` 引入——此前"1 skipped"正是 `importorskip` 在无 `gr00t` 时静默走了跳过分支；②该 venv 无 `pytest`；③该 venv 无 **`decord`**，而父类 DataConfig 构造 transform 链时经 transformers 动态模块加载它，缺失会让 `RoboCasa365DataConfig()` 直接 `ImportError`（实测 4 failed）；④仓库 `conftest.py` 默认跳过 `manual`，`-m manual` 只是**选择**、仍会跳，必须 `--run-manual`。四点已全部写入测试文件 docstring 的运行命令、`serve_groot_n15.py` 的环境说明与 plan §1.2/§4.5。依赖安装前均做 `--dry-run` 确认：`pytest`(4 包)、`decord`(1 包) 全为增量安装，**numpy 1.26.4 / torch 2.5.1+cu124 复验未变**。补充 §5.2 声明但未落地的 checkpoint 对照：新增 `test_modality_config_covers_checkpoint_keys`（四类键集合 vs `metadata.json`，按集合比较并注明 metadata 为字母序故不比顺序）、`test_state_and_action_dims_match_checkpoint`（逐键宽度 vs 统计量 `mean` 长度）、`test_checkpoint_embodiment_tag_is_new_embodiment`，checkpoint 路径可用 `ROBOCASA365_N15_CHECKPOINT` 覆盖、缺失则 skip。**实跑通过输出见下方证据段。**

- Accepted（意见 4：README 条目内容过时且未暂存）— 属实。该条目重写：四类**有序**键及其官方直证出处（`robocasa@be22d659:docs/datasets/using_datasets.md:167-210`）、**N1.5 原生 language key 及"不可用父类 N1.7 变体"的显式警告**、metadata 为字母序不可据以推序、真实转换面（无键名映射，但有 T=1 时间轴 / 512→256 INTER_AREA / 双向有限值门禁与 `(16,D)` 形状校验）、五个实际交付文件与四槽结构、两层测试的真实数量、以及本轮实测的环境坑（worktree/decord/pytest/`--run-manual`）。**已单独暂存**：从 `git diff -U0` 中按 hunk 过滤出仅含本条目的补丁再 `git apply --cached`，未卷入同文件其他 session 的三行改动（已核对暂存区仅含我的两行；工作树 diff 中本条仍显示为 -/+ 是其他 session 插行导致的**纯位置移动**，两侧内容 md5 相同）。

- Accepted（意见 5，Non-blocking：A/B 未配对却称 paired）— 属实，此前两侧都调用无 seed 的 `env.reset()`，随机流独立。新增 `episode_seeds(n_trials, base_seed)` 与 `--base-seed`（默认 0），`run_one()` 以 `env.reset(seed=...)` 逐 episode 回放同一 seed 列表，两个场景共用该列表；seeds 一并写入结果 JSON 与每行日志。新增 `test_run_one_replays_the_declared_seeds` 与 `test_episode_seeds_are_identical_across_scenes`。

**测试证据**

主环境（非 manual，CI 口径）：

```
$ uv run pytest tests/robocasa365/ -q
35 passed, 1 skipped in 0.09s
$ uv run ruff check exp/robocasa365/ tests/robocasa365/   ->  All checks passed!
$ uv run ruff format --check exp/robocasa365/ tests/robocasa365/  ->  9 files already formatted
```

孤岛 B（manual，真实 `gr00t` + checkpoint metadata，无 GPU/仿真）：

```
$ PYTHONPATH=/home/weiland/gr00t_n15:/home/weiland/openpi/src:/home/weiland/openpi \
  /home/weiland/gr00t_n15_venv/.venv/bin/python -m pytest \
  tests/robocasa365/test_groot_data_config_manual.py --run-manual -q
7 passed, 1 warning in 12.21s
```

⚠ 该 1 skipped 是 manual 文件在主环境被 `importorskip` 整体跳过（预期行为）。G0-A 接线等价性与 GPU/仿真检查按 plan §9.1 仍留待 §6 Verify 之后的实跑阶段。

### G2 Round 3 — Reviewer — NEEDS REVISION — 2026-08-16 10:11 CDT

- [Blocking] [Concern] 让 client 侧实际执行完整 action-chunk 校验，而不只校验 `replan_steps`——reasoning: Round 2 回复称 `iter_step_actions()` 封装了“校验 + 切片”，但实现仅调用 `take_action_chunk()`；后者只检查 `replan_steps` 上下界，然后直接切片，不调用 `validate_action_chunk()`。因此一个首维 15、但 `replan_steps=5` 的 wire chunk 仍会在 client 生产路径静默通过。新增独立探针 `test_client_split_rejects_wrong_action_horizon` 已复现 “DID NOT RAISE”，而同一错误在 server adapter 路径现已正确拒绝。须让 `iter_step_actions()`（或其唯一前置 helper）先调用 `validate_action_chunk()`，并补 client helper 对 wrong horizon、wrong width、缺键/额外键的回归测试，确保 server/client 两端共享同一契约。
- [Blocking] [Concern] 修正目标 plan 内仍过期的 manual 命令和实现清单——reasoning: §1.2/§4.5 与测试 docstring 已正确记录 `/home/weiland/gr00t_n15`、`decord`、`pytest`、`--run-manual`，但 plan §5.2 仍给出旧命令 `PYTHONPATH=/home/weiland/openpi/src uv run pytest ... -m manual`；该命令会缺 `gr00t` worktree 且仍被默认 manual skip。§4.0 目录树也未列本轮新增的 `tests/robocasa365/test_groot_rollout_client.py`，§5.1/R4 仍称 16 个非 manual 用例。须把计划正文的可复现命令、文件清单与当前 35+7 测试事实同步，避免下一 session 按旧命令得到假 skip。
- [Blocking] [Concern] 重新制作 `logs/README.md` 的仅目标暂存补丁；当前 index 并未把条目放入活动索引表，且卷入了其他 session 的条目——reasoning: `git diff --cached -U0 -- logs/README.md` 显示暂存区只在文件末尾 `## Maintenance Rules` 之后追加了两行：`benchmark_and_teacher_selection` 与本目标；`git show :logs/README.md` 中本目标位于第 282 行而非活动表，破坏 Markdown 表结构，同时 benchmark 行不属于本审查目标。工作树第 66 行的目标文本本身已正确，但尚未以正确位置进入 index。须撤销该文件当前错误的 staged hunk，再仅把本目标行添加到活动表的正确位置；其他 session 的 README 增删保持未暂存、未改写。

### G2 Round 4 — Executor — 2026-08-16

- Accepted（意见 1：client 侧未执行完整 action-chunk 校验）— 属实，且我在 Round 2 的回复措辞**不准确**：当时称 `iter_step_actions()` 封装了"校验 + 切片"，但实现只调用 `take_action_chunk()`，后者仅检查 `replan_steps` 上下界。审查者探针 `test_client_split_rejects_wrong_action_horizon` 复现的 "DID NOT RAISE" 是真缺陷。`iter_step_actions()` 现先调用 `validate_action_chunk()` 再切片，docstring 写明理由：这是网络边界的客户端一侧，收到的是 msgpack 解出来的东西，而非 server 自认为发出的东西。新增 6 个非 manual 回归：`test_client_split_rejects_{wrong_action_horizon,wrong_action_width,missing_key,unexpected_key,non_finite}` 与 `test_server_and_client_share_one_contract`（同一畸形 chunk 两端都必须拒绝）。

- Accepted（意见 2：plan 正文命令与清单过期）— 属实。§5.2 的旧命令会导致**假跳过**（缺 worktree PYTHONPATH 时 `importorskip` 静默走跳过分支、`-m manual` 只选择不放行），已整节重写为两个孤岛各自的可执行命令并标注四处易错点；§4.0 目录树补齐 `test_groot_rollout_client.py` 与本轮新增的 `test_env_action_contract_manual.py`；§5.1/§6-R4 的用例数由 16 更新为实测 **71**（Layer 1）、**9 + 3**（Layer 2 两个孤岛）；§5.1 增补自证陷阱与时序顺序两条说明；§4.2 输出契约更正为 `{"actions": {...}}` 单键（`server_timing` 由 `WebsocketPolicyServer` 在 `infer()` 返回后注入，adapter 不自行产生）；§4.4 补记配对 seed 与本轮健壮性改动；§8 措辞更正为"不修改任何既有**代码**文件"（`logs/README.md` 按 WA §4 必须同 commit 更新，属例外）；§1.6 补上 server 侧仓库路径 `/home/weiland/openpi`（全文命令用的都是它）；§4.1 示例代码加注"落地按 §5.0 引用 `groot_keys`"，消除与 §5.0 的矛盾。

- Accepted（意见 3：README 暂存 hunk 位置错误且卷入他人条目）— 属实。根因是我用 `git diff -U0` 生成补丁再按内容过滤 hunk，而 hunk 位置按**工作树**计算，被其他 session 在附近的插行带偏，落到了 `## Maintenance Rules` 之后、活动表之外，并捎带上 benchmark 条目。`git apply --cached` 重试仍不落位（无 context 的补丁不保证定位）。改用 `git hash-object -w` + `git update-index --cacheinfo` 直接构造 index 内容：以 HEAD 版为基底，仅在活动表末行（`ablation_study_plan`）之后插入本目标一行。现 staged hunk 为 `@@ -62,0 +63 @@` 单行、位于活动表内、不含 benchmark 行；工作树其他 session 的改动未被触碰、未被改写。条目内容同步刷新为本轮实测数字与新增测试文件。

**本轮附加：交外审前的自查（Execution 内部，无流程效力）**

应 owner 要求，在提交本轮前用 4 个只读 sub-agent 做了一次自查（WA §9.3：sub-agent 继承 Execution 授权，**不构成也不替代 G2 review**），维度为契约一致性 / 资源与异常路径 / 文档一致性 / 测试有效性。其中测试有效性一路做了 60 次变异注入实验。据其发现补修如下（均为本轮新增，非审查意见要求）：

*契约*
- **图像分辨率此前是契约里唯一"声明了却不校验"的维度**（state 宽、action 宽、horizon 均为硬断言）。`build_groot_observation` 现强制 `(256, 256, 3)`：跳过降采样的客户端会送 512² 帧，结构合法但呈现给模型的视野与训练不符。
- **client 对图像 dtype 由静默 cast 改为断言**。原 `np.ascontiguousarray(image, dtype=np.uint8)` 会把 `[0,1]` 浮点帧截成全零，**并使 server 端的 dtype 检查必然通过**——唯一能捕获该错误的关卡被它前面的 cast 解除了。
- client 启动时核对 server 广播的 `action_horizon`（机制本已建好，只差这个断言），版本漂移由"第 300 个任务时每 chunk 报错"变为启动即失败。
- 四元数注释更正：robosuite 发出的是 **xyzw**，GR00T 按 wxyz 读取——训练与评测一致故无害，但 `[1,0,0,0]` 是 GR00T 视角下的单位旋转、并非仿真器视角下的。原注释宣称的多于它验证的。

*测试有效性（变异实验暴露）*
- **`test_dummy_obs_uses_unit_quaternion_not_zeros` 原本是空转的**：它遍历 `QUATERNION_STATE_KEYS`，把该列表清空后测试照样通过——守卫被它所守卫的值关掉。现该列表本身被钉死到字面量。
- **动作时序顺序此前零覆盖**：把 `iter_step_actions` 的 `range()` 改成 `reversed()`，形状/键名/计数全不变，71 个测试全绿，而机器人会倒序执行整个 chunk。现以 ramp 值断言第 k 步等于 chunk 第 k 行。
- **凡"数字"皆自证**：fixture 用被测常量构造数据，故 `STATE_DIMS` 改错、两个分辨率常量改错均可存活。现 `STATE_DIMS`/`ACTION_DIMS`/`RENDER_RESOLUTION`/`MODEL_IMAGE_RESOLUTION` 各有独立的字面量对照。
- **rollout 循环语义此前零覆盖**（fake env 从不成功、各步不可区分），`successes += 1`、`sr = successes`、`info.get("is_success")`、`plan.pop()`（LIFO）、plan 跨 episode 泄漏等 7 个变异全部存活——**而这些产出的正是实验的头条数字**。现补 9 个用例覆盖成功计数与成功率、horizon 截断、成功即停、FIFO 执行顺序、跨 episode 重规划、replan 频次、降采样后分辨率与 wire 键集合。
- **观测方向缺端到端契约**（动作方向已有）：现 `_select_and_downsample` 的输出直接喂 `build_groot_observation`，并断言三路相机不塌缩为同一帧。
- 孤岛 B 补 `test_transform_chain_constructs`（`modality_config()` 是自证的——握手比对的就是传进去的同一对象；构造 `transform()` 才能证明父类接受键覆盖且 `decord` 到位）与 `mean/std` 宽度一致性。
- 孤岛 A 新增 `test_env_action_contract_manual.py`：此前 `test_action_keys_match_env_contract` 比的是**同一份抄录列表与自己**，对 env 契约漂移零检测力；现调用**真实 `convert_action`** 完成绑定，并将原测试更名为 `..._transcribed_env_contract` 且注明其自洽性质。
- 测试名与断言相反者更名（`run_one` 的 replan 校验发生在 `gym.make` 之后，原名 `..._before_touching_env` 误导）。

*健壮性（无人值守长跑）*
- 连续失败熔断 `--max-consecutive-failures`（默认 3）：连接一旦断开，原实现会对每个剩余任务完整建环境→reset→infer 失败，跑满整夜产出一份"模型不行"的假数据。
- 结果 JSON 改为临时文件 + `os.replace` **原子写**（`write_text` 先 truncate，崩在窗口内会丢失全部已完成任务，与 docstring 的承诺相悖）。
- 错误摘要取 traceback **末行**（server 回传完整 traceback，前 160 字符全是 websocket 样板，原实现每条 error 长得一样且不含根因）。
- `--n-trials` 与 scene 格式在 argparse 后 fail-fast（原先 `--n-trials 0` 的 `ZeroDivisionError`、`--scene-a 1` 的解包错误都会被逐 arm 的 `except Exception` 洗成"任务失败"，跑满全程且退出码为 0）。
- `logging` 由 WARNING 调整为 INFO：`openpi_client` 的连接等待日志走 INFO，被过滤后 server 未起时客户端零输出、永久阻塞；client 构造亦移到打印与 `mkdir` 之后。
- server 广播 `metadata`，client 默认**拒绝**对开着 `--diagnostic-seed` 的 server 采集（需 `--allow-diagnostic-server`）：该模式钉死 flow-matching 噪声，会使成功率不再是 teacher 的无偏估计，且该污染事后无法从数据中发现。

**测试证据**

```
主环境（非 manual，CI 口径）
$ uv run pytest tests/robocasa365/ -q                 ->  71 passed, 2 skipped
$ uv run ruff check exp/robocasa365/ tests/robocasa365/   ->  All checks passed!
$ uv run ruff format --check ...                      ->  10 files already formatted

孤岛 B（gr00t worktree + checkpoint metadata，无 GPU/权重加载）
$ ... -m pytest tests/robocasa365/test_groot_data_config_manual.py --run-manual -q
  9 passed

孤岛 A（真实 robocasa convert_action）
$ ... -m pytest tests/robocasa365/test_env_action_contract_manual.py --run-manual -q
  3 passed
```

⚠ 2 skipped = 两个 manual 模块在主环境被 `importorskip` 整体跳过（预期）。G0-A 接线等价性与 GPU/仿真检查按 §9.1 仍留待后续实跑阶段。

### G2 Round 5 — Executor（自查续修，无审查意见触发） — 2026-08-16

自查 sub-agent 在首轮报告后补交了两条修订，据此又做了两处实证与修改。**这不是对任何审查意见的回应**，记录于此以便审查者了解代码为何再次变动。

- **关闭「唯一一处错了却能通过所有关卡」的风险（新增 F22）**。此前 §4.3 的握手只比对 `policy.get_modality_config()`，而那是**自证**的——比对的正是第 197 行传进去的同一个对象；真正拼接张量的是 `transform()`，它从未被校验。若父类 `SinglePandaGripperDataConfig.transform()` 用硬编码列表而非 `self.*_keys` 构建 `ConcatTransform`，则 `groot_data_config.py` 的四个类属性覆盖只对 config 生效、对张量无效，而握手仍会打印一片整齐匹配。已在孤岛 B 实读父类源码（`data_config.py:459-500`）确认其全程使用 `self.*`，并新增 `test_transform_concat_order_uses_our_keys` 直接断言 `ConcatTransform` 的三个 `*_concat_order` 等于 `groot_keys` —— **实测通过**，该风险由推断变为实证。

- **握手探针的四元数改为「checkpoint 均值 + L2 归一化」（新增 F24）**。链路已由源码钉死：`robosuite/utils/transform_utils.py:317-325` 的 `mat2quat` 返回 `(x,y,z,w)`，`robosuite/robots/mobile_robot.py:317,376` 的两个四元数字段均走它 ⇒ wire 上是 **xyzw**，而 GR00T 侧的 pytorch3d 是 **wxyz**。这**不是 serving 的正确性问题**（训练与评测被同一套变换同样地读，一致即无害），但意味着 `IDENTITY_QUATERNION_WXYZ` 在 env 编码下是绕 X 轴 180°、并非语义上的单位旋转 —— 拿它填探针会让 5 个 state 字段中的 2 个落到分布外。改为取 checkpoint 自身的四元数均值再归一化（多个四元数的均值本身不是单位模长，而非单位四元数不是旋转），仅在均值退化为零向量时回退。相应地 `test_dummy_obs_uses_unit_quaternion_not_zeros` 改写为断言**单位模长且非零且有限**，并新增退化情形的回退测试。

- 顺带记录 **F23**：父类 `VideoResize` 把视频统一 resize 到 **224**，故 checkpoint metadata 声明的 `[256,256]` 是**数据集存储**规格而非模型输入尺寸。client 送 256 与训练路径一致（512 渲染 → 256 → transform 224）；直接送 512 虽也会被 resize 到 224，但降采样路径不同，故仍按 256 送。

**测试证据（最终）**

```
主环境（非 manual，CI 口径）
$ uv run pytest tests/robocasa365/ -q                      ->  72 passed, 2 skipped
$ uv run ruff check / format --check exp/ tests/           ->  All checks passed / 10 files formatted

孤岛 B（gr00t worktree + checkpoint metadata，无 GPU/权重）
$ ... test_groot_data_config_manual.py --run-manual -q     ->  10 passed

孤岛 A（真实 robocasa convert_action）
$ ... test_env_action_contract_manual.py --run-manual -q   ->  3 passed
```

### G2 Round 6 — Reviewer — APPROVED — 2026-08-16 10:51 CDT

- 上一轮三项阻塞已全部关闭：`iter_step_actions()` 在 client 网络边界先调用共享的 `validate_action_chunk()`，wrong horizon/width、缺键/额外键与非有限值均有回归覆盖；plan §4/§5/R4 已同步当前文件、双孤岛命令和测试数字；`logs/README.md` 的 staged patch 仅在活动索引表内新增本目标一行，不再卷入其他 session 条目。
- 独立复核通过：主环境 `72 passed, 2 skipped`；review-only 独立探针 `3 passed`（错误 horizon 的 server/client 双侧拒绝、推理异常时 env 关闭）；孤岛 B manual `10 passed`；孤岛 A manual `3 passed`；ruff check/format 与 `git diff --cached --check` 均通过。
- 实现与已批准计划一致：有序观测/动作键、T=1、512→256、`(16,D)` 输出契约、有限值门禁、server/client 环境隔离、资源释放、配对 seed、诊断 server 拒绝、连续失败熔断及原子结果写入均已落地并有相称测试。
- G0-A 接线等价性、GPU 权重加载与真实仿真 rollout 仍按 plan §6/§9.1 留在 G2 后的 Verify 阶段；它们不是本次代码审查的未完成实现项，不阻塞 G2。


---

## 10. 执行记录

### 10.1 G0-A 接线等价性 — ✅ **PASS**（2026-08-16，weilandserver）

§9.1 中唯一「必须通过、不设时间盒」的门。分两步，均在孤岛 B 内实跑：

**步骤 1（同进程）** —— 同一 obs、同一 `torch.manual_seed(12345)` 下比较「直接 `policy.get_action(build_groot_observation(obs))`」与「`adapter.infer(obs)`」：五个 action key **全部 `maxdiff = 0.000e+00`**。

⚠ 同时跑了**必要的反向对照**：不设种子时两次调用**确实不同**（`sampling_stochastic: true`）。没有这一条，`maxdiff=0` 也可能只是"模型本来就确定性"，证明不了 adapter 无损；有了它，才同时确立「采样确实随机」与「固定种子后 adapter 完全透明」。这也在实机复证了 F20。

**步骤 2（跨进程）** —— server 以 `--diagnostic-seed 12345` 启动（seed 在 server 进程内、紧邻 `get_action()` 之前重置；client 侧设种子无效，两个解释器不共享全局 RNG），client 发送同一 obs：五个 key **全部 `maxdiff = 0.000e+00`**，dtype 两侧同为 `float32`（msgpack 往返保真）。

⇒ **adapter 层与 websocket 层均为无损**。后续 rollout 若出现 SR 问题，不可归因于接线。

**附带确认**：server 启动握手按 §4.3 全部通过并打印四类 modality 的**有序**列表与 checkpoint 绝对路径；`metadata` 正确广播 `{checkpoint, diagnostic_seed, action_horizon}`，client 侧的诊断模式拒收与 horizon 核对因此可用。

### 10.2 本次实跑修正的文档错误

- **§4.5 / §5.2 原写 `python -m pip install ...` 是错的**：孤岛 B 的 venv **没有 pip**（实测 `No module named pip`）。正确写法是 `VIRTUAL_ENV=... uv pip install ...`，已更正。此前装 pytest/decord 时用的就是 uv，是文档抄写有误。
- **`openpi-client` 此前从未真正装进孤岛 B**：plan §4.5 列了该步骤但未执行，导致 server 首次启动即 `ModuleNotFoundError: No module named 'openpi_client'`。现已安装（连带 svgwrite / tree / websockets），并复验 **numpy 1.26.4 / torch 2.5.1+cu124 / transformers 4.51.3 三个关键钉定均未变动**。

### 10.3 G0-B 行为健全性 — ⚠ **按预注册字面判据 FAIL；但失败点不在本门所检验的假设上**

**数据前提已满足**（R10 未触发）：官方 `robocasa/scripts/download_datasets.py` 支持 `--tasks` 按任务下载（数据托管在 UT Austin Box，非 HF）。已取 `PickPlaceCounterToStove`（99 MB / 108 ep）与 `OpenCabinet`（144 MB / 107 ep）的 pretrain human 数据，LeRobot 格式。

**执行**：2 任务 × 各前 5 episode × 4 个起点（step 0/16/32/48）× 5 个共同 seed，逐 action key 在**该键自身 std 单位**下算开环 MSE；先在 episode 内对 (起点 × seed) 求均，再以 **episode 为配对单位**（n = 10）做单侧 Wilcoxon。

| action key | gt 方差 | **P** | N1 | N2 | B | vs_B | vs_N1 | vs_N2 |
|---|---:|---:|---:|---:|---:|---|---|---|
| `end_effector_position` | 2.174 | **0.525** | 1.616 | 1.105 | 2.853 | p=0.0010 ✅ | p=0.0010 ✅ | p=0.0010 ✅ |
| `end_effector_rotation` | 2.412 | **1.023** | 2.485 | 4.444 | 2.939 | p=0.0029 ✅ | p=0.0049 ✅ | p=0.0010 ✅ |
| `gripper_close` | **0.019** | 1.012 | 1.012 | 1.012 | 0.801 | p=0.999 ❌ | p=0.75 | p=0.75 |
| `base_motion` | 0.000 | — | — | — | — | **EXCLUDED**（近零方差） | | |
| `control_mode` | 0.000 | — | — | — | — | **EXCLUDED**（近零方差） | | |

**字面裁决**：判据 1（P 须在**每个**未排除键上显著优于 B）在 `gripper_close` 上不成立 ⇒ **G0-B FAIL**。

**根因分解（不是配置错误）**：

1. **`gripper_close` 对本门的假设零判别力**：三个 arm 的 P/N1/N2 **数值完全相同（1.0124）**——该键用 `binary` 归一化，N1/N2 都不改动它。一个无法区分对照组的键，却能单独否决整个门。
2. **它实质上属于近零方差类别，只是没被阈值捞住**：其 gt 方差 **0.019**，比两个真正承载控制信息的键（2.174 / 2.412）低 **两个数量级**，仅因高于我预注册的 `1e-3` 阈值而留在判定集内。在一个几乎不变的信号上，常数均值预测器本就接近最优，"模型打不过它"不构成对配置的指控。
3. **MSE 对二值键本就不是合适指标**：`gripper_close` 与 `control_mode` 都是二值量，模型输出连续值，反归一化后与二值 gt 比 MSE 天然吃亏；恰当的指标是分类准确率。这两个键上"模型不如均值"是**指标选择**的产物。

**证据的实际指向**：在两个真正携带连续控制信息的键上，**P 同时显著优于 B、N1、N2**（p ≤ 0.005，n=10 配对）。即：
- 相对基线 B —— 模型确实在预测，链路无根本问题；
- 相对 N1 —— **非旋转字段的 `min_max` 归一化得到确认**；
- 相对 N2 —— **`rotation_6d` 旋转表示得到确认**。

⚠ **另有一半的 R2 由上游断言直接关闭**：`gr00t/data/transform/state_action.py:445` 断言「凡转换成其他旋转表示的状态字段，其归一化**必须**为 `min_max`」。故两个四元数字段的 `min_max` 是**代码强制**而非我方推定，N1 arm 因此只能扰动非旋转字段。

**补充证据：换用适合二值键的指标后，结论反转**（描述性，**不修改预注册判据**）。对 `gripper_close` / `control_mode` 改用分类准确率（阈值 0.5）重测 shipped 配置（N1/N2 在这两键上与 P 数值完全相同，无需重跑）：

| key | 模型准确率 | 多数类基线 | gt 正例率 | 超越基线 |
|---|---:|---:|---:|---|
| `action.gripper_close` | **0.9938** | 0.9688 | 3.1% | **✅ 是** |
| `action.control_mode` | 1.0000 | 1.0000 | 0.0% | 否（基线已达 100%，无从超越） |

⇒ 模型在 `gripper_close` 上准确率 **99.4%**、**高于多数类基线 96.9%**，即它在该键上**确实在有效预测**；MSE 之所以给出相反结论，是因为把连续输出与 0/1 标签比 MSE 时，时机上的微小偏差被放大，而该信号本身只有 3.1% 的正例。`control_mode` 的 ground truth 在这两个任务上恒为 0，模型 100% 正确 —— 这也从另一侧印证了它被"近零方差"规则排除是恰当的。

⇒ **三条证据共同指向同一结论**：①G0-A 证明接线 bit 级无损；②两个连续控制键上 P 显著优于 B/N1/N2（归一化与旋转表示均获确认）；③二值键在恰当指标下模型优于基线。**G0-B 的字面 FAIL 源于把 MSE 用在近常数的二值键上，而非配置或链路缺陷。**

**⚠ 我不修改预注册判据。** 阈值 `1e-3` 定得过松是我预注册时的缺陷，但事后调阈值以翻转结论正是预注册要防止的行为。字面结论保留为 FAIL，并将处置提交 owner 裁定：

| 选项 | 含义 |
|---|---|
| **(a) 判据修正后重判**（建议） | 把"近零方差"阈值改为相对判据（如 gt 方差 < 主控制键中位数的 1/10），或将"三个 arm 数值相同、对假设零判别力"的键一并排除，再重跑判定。按现有数据，两条修正都会让结论变为 PASS + 归一化确认 |
| (b) 补一个适合二值键的指标 | **已补测（见上）**：`gripper_close` 准确率 99.4% > 基线 96.9%。若采纳该指标进入判定集，判据 1 亦成立 |
| (c) 严格照字面执行 | 判 FAIL 并转 §9.2 退路。⚠ 与证据方向相悖：P 在主控制键上优于基线 5.4 倍 / 2.9 倍 |

### 10.4 下一步

G0-A 已 PASS；G0-B 的实质结论（归一化与旋转表示均获确认）已具备，仅判据形式待 owner 按上表裁定。其后按 §5.3 进入 S3/S4 rollout。


### 10.5 §9.2 退路的前置查证 — ✅ 完成（2026-08-16，HF API 实查）

W3 曾要求「退路 A2 的前置动作：先申请 Cosmos-Reason2-2B 访问权，并验证 `GR00T-N1.7-LIBERO` 是否自带全部 backbone 权重」。该项无论 G0-B 如何裁定都必须有答案（裁「继续」则它是保险，裁「转退路」则它是阻塞项），故先行做掉：

| repo | 可访问性 | 内容 |
|---|---|---|
| `nvidia/GR00T-N1.7-LIBERO` | **`gated=False`，可直接下载** | 203 个文件，**四个 suite 子目录齐全**：`libero_spatial` / `libero_object` / `libero_goal` / `libero_10` —— **实测印证 W1**。每 suite 含 `config.json`、`embodiment_id.json`、2 个 safetensors 分片 + index、`processor_config.json`、`statistics.json`、`experiment_cfg/` |
| `nvidia/Cosmos-Reason2-2B` | **`gated=auto`** | 15 个文件，单个 safetensors |
| `nvidia/GR00T-N1.7-3B` | `gated=False` | 基座模型，27 个文件 |

**对 W3 的两点修正**：

1. **gated 的性质比原先设想的轻**：`Cosmos-Reason2-2B` 是 **`gated=auto`** —— 接受条款即自动放行，**不存在人工审批排队**。W3 原文「先申请访问权」暗示的等待并不成立。
2. **checkpoint 并非完全自包含**：`libero_spatial/` 下**没有任何 tokenizer / vocab / preprocessor 类文件**（仅 `processor_config.json`），故加载时仍会回源到 `nvidia/Cosmos-Reason2-2B` 取 tokenizer ⇒ **gated 条款仍须接受**，只是成本仅为一次点击。

⚠ **下载须按官方 `--include` 过滤**：该 repo 同时包含 DeepSpeed 优化器状态（`global_step20000/bf16_zero_pp_rank_*_optim_states.pt`，16 个 rank），全量克隆会拉入大量推理无关的权重。`scripts/deployment/README.md:39-43` 给出的 `--include` 清单是正确做法。

⇒ **退路 A2 的可行性已确认，且成本低于 §9.3 表中的估计**（零微调 + 一次条款接受）。这不改变「退路是保险而非等价替代」的判断（§9.4：LIBERO 无跨场景 split）。


### 10.6 N3 决定性负对照 — 区分「指标选择」与「配置缺陷」

上文对 G0-B 失败点的解释（属指标选择而非配置缺陷）此前只是**推断**，未经实验区分。原设计的 N1/N2 恰好都不改动 `binary` 归一化，留下了盲区。故补一个针对该键的负对照：

**N3** = shipped 配置，仅把 `action.gripper_close` 的归一化由 `binary` 改为 `min_max`。其判别逻辑与 §9.1 判据 4 相同（负对照优于 P ⇒ 配置错），且**可能推翻我方解释**。

| action key | P | **N3** | B（基线） | 说明 |
|---|---:|---:|---:|---|
| `end_effector_position` | 0.5248 | 0.5248 | 2.8530 | **完全相同** —— 改动是局部的 |
| `end_effector_rotation` | 1.0230 | 1.0230 | 2.9392 | **完全相同** |
| **`gripper_close`** | 1.0124 | **1.0052** | **0.8006** | N3 < P，Wilcoxon 单侧 **p = 0.00098** |
| `base_motion` / `control_mode` | 0.0007 / 1.4710 | 同 P | — | 不受影响 |

**结论（三点，缺一不可）**：

1. **N3 统计显著优于 P，但效应量仅 0.7%**（1.0124 → 1.0052）。⚠ `p = 0.00098` 是 n=10 时 Wilcoxon 单侧的**最小可能值**，只表明 10 个 episode 方向一致，**不表明幅度**。仅看 p 值会严重高估该发现的分量。
2. **N3 仍远差于基线 B**（1.0052 vs 0.8006，相差 25.6%）。⇒ **即使把归一化"改对"，该键依然无法通过判据 1。** 因此判据 1 的失败**不可能**由归一化配置解释。
3. 两个连续控制键上 **P 与 N3 数值完全相同**，证明该改动不触及主结论。

⇒ **原解释得到实验支持**：G0-B 判据 1 的失败源于「把 MSE 用在近常数二值键上」这一指标选择，**而非配置缺陷**。这一判断现在有实验依据，不再是推断。

⚠ **同时得到一个未预料到的次要发现，如实记录**：`action.gripper_close` 用 `min_max` 确实比继承自父类的 `binary` **略优**（一致方向、0.7% 幅度）。这不影响 G0-B 的任何定性结论，也不改变判据 1 的走向，但属于「继承配置并非处处最优」的证据。是否据此调整 §4.1 的 `action_normalization_modes`，**提请 owner 一并裁定** —— 我不擅自改动已 G2 APPROVED 的配置：效应量微小，且改动会使 `groot_data_config.py` 偏离父类默认值，需要相应更新测试与 plan。


### 10.7 Owner 裁定与 G0-B 重判 — ✅ **PASS**（2026-08-16）

**owner 裁定两项**：
1. **G0-B：修正判据后重判**（plan §10.3 选项 a）。
2. **`gripper_close` 归一化：保持 `binary`** —— 效应量仅 0.7% 且不改变任何定性结论，维持与官方父类默认一致，不引入未经官方背书的偏离。N3 的发现留在 §10.6 备查，**不修改 `groot_data_config.py`**。

**⚠ 本次判据修正为「事后修正」，据此显式记录**（原判据见 §9.1「🔒 预注册分析口径」，**原文保持不动**）：

原判据以**绝对阈值** `1e-3` 排除近零方差键。该阈值过松：`action.gripper_close` 的 gt 方差 **0.019** 比两个真正承载控制信号的键低**两个数量级**（2.174 / 2.412），实质属于近常数，却因高于 `1e-3` 留在判定集内，进而以「无法击败常数基线」单独否决整个门 —— 而它同时**对本门假设零判别力**（P/N1/N2 数值完全相同）。

修正后**同时**报告两条独立口径，使结论不依赖于口径选择：

| 口径 | 定义 | 排除的键 | 保留判定 | 结果 |
|---|---|---|---|---|
| **C1 相对方差** | gt 方差 < 控制信号键方差中位数（2.412）的 1/10，即 < 0.241 | `base_motion` / `control_mode` / `gripper_close` | `end_effector_position` / `end_effector_rotation` | **PASS** + 归一化确认 |
| **C2 零判别力** | P、N1、N2 三者数值完全相同（无法区分假设）→ 叠加原排除 | `control_mode` / `gripper_close` (+ 原 `base_motion`) | 同上 | **PASS** + 归一化确认 |
| **C1 ∪ C2** | 两者并集 | 同 C1 | 同上 | **PASS** + 归一化确认 |

**三条口径结论完全一致**，判定集均收敛到两个真正的控制键：
- **必要条件**：P 在两键上均显著优于常数基线 B（p = 0.0010 / 0.0029）⇒ 模型确实在预测，链路无根本问题。
- **判据 2**：P 在两键上均显著优于 N1 与 N2（p = 0.0010/0.0049 与 0.0010/0.0010）⇒ **非旋转字段的 `min_max` 归一化与 `rotation_6d` 旋转表示双双确认**（旋转字段的 `min_max` 另由上游断言强制，见 §10.3）。

⇒ **G0-B PASS。§9.1 的两个出场门（G0-A 接线等价性 / G0-B 行为健全性）均已通过，可进入 §5.3 的 S3/S4 rollout。**

⚠ 重判**未重跑任何推理**，使用的是同一批 `g0b_result.json` 数据；修正的只是「哪些键进入判定集」，各键的 MSE 与 p 值一字未改。


### 10.8 S3 闭环暴露的 import 路径缺陷 —— 以及一类 mock 无法覆盖的盲区

首次真实 rollout 立刻失败：

```
ImportError: cannot import name 'get_task_horizon'
             from 'robocasa.utils.dataset_registry'
```

`get_task_horizon` 实际位于 `robocasa/utils/dataset_registry_utils.py:240`，而 `TASK_SET_REGISTRY` 才在 `dataset_registry.py`。`groot_rollout_client.run_one()` 把前者写成了后者。

⚠ **为什么 72 个非 manual 测试全部放行了这个错误**：`test_groot_rollout_client.py` 用 `monkeypatch.setitem(sys.modules, ...)` 注入仿真假件，而假件是**按被测代码所请求的名字**注册的。代码写错模块名，假件就在那个错误的名字下被创建——于是测试与 bug 互相印证，一路全绿。**这是 mock 测试的结构性盲区：它能验证调用逻辑，但对「import 路径是否指向真实存在的符号」天然无能为力。**

三处修正：
1. **生产代码**：改为 `from robocasa.utils.dataset_registry_utils import get_task_horizon`。
2. **假件**：改为镜像**真实的**模块布局（`dataset_registry` 提供 `TASK_SET_REGISTRY`，`dataset_registry_utils` 提供 `get_task_horizon`），而非跟随代码的假设。
3. **补一个 mock 结构上做不到的检验**：孤岛 A 的 `test_rollout_client_imports_resolve_against_real_robocasa`，对真实安装的 `robocasa` 解析这两个符号并调用 `get_task_horizon`。**实测通过（孤岛 A 4 passed）。**

🟢 **两项健壮性设计当场兑现了价值**（均为 G2 Round 2 应自查发现所加）：
- **连续失败熔断**在第 3 个 arm 即中止，没有让这个必然失败的配置跑满 12 个 episode；
- **错误摘要取 traceback 末行**，结果 JSON 里直接是 `ImportError: cannot import name ...`，而非 160 字符的 websocket 样板。

⇒ 修正后重跑，**闭环成功**：`ep0(seed=0): OK t=268/600`、`ep1(seed=1): OK t=230/600` —— GR00T N1.5 在 RoboCasa365 上真实执行并完成任务，S3 通过。


### 10.9 S3 / S4 冒烟 rollout — ✅ **通过**（2026-08-16）

修正 import 后重跑，2 任务 × 2 场景 × 3 ep，配对 seed `[0, 1, 2]`（两场景回放同一列表）：

| 任务 | scene A (1,1) | scene B (7,7) | wall A | wall B |
|---|---|---|---:|---:|
| `PickPlaceCounterToStove` | **3/3 = 1.00** | **3/3 = 1.00** | 78.6 s | 122.0 s |
| `OpenCabinet` | 2/3 = 0.67 | **3/3 = 1.00** | 157.3 s | 179.4 s |
| **合计** | **5/6 = 0.83** | **6/6 = 1.00** | | |

**S1–S4 全部通过**：S1 启动握手（四类 modality 有序键 + checkpoint 路径 + 有限值断言）、S2 单步推理（5 键各 `(16, D)` 且有限）、S3 单 episode 闭环（真实执行并完成任务）、S4 冒烟 SR（远大于 0）。

**与 pi0.5 的横向参照**（同两任务，pi0.5 的 180 ep 准入门数据）：`PickPlaceCounterToStove` A 5/5 / B 5/5；`OpenCabinet` A 4/5 / B 4/5。GR00T N1.5 与之**同一量级**，⇒ **两个 teacher 在这两个任务上都能干活**，第二 teacher 的可用性得到初步确认。

⚠ n=3/格，仅作 sanity check，**不承担任何 SR 结论**——S4 的职责按 §9.1 G0-D 已被降级为健全性检查，链路正确性由 G0-A 举证、归一化由 G0-B 举证。正式的跨场景 SR 需按 §12-5 的规模执行。

⚠ 场景成本差与 pi0.5 侧一致：scene B (7,7) 的 wall 明显高于 scene A (1,1)（122 vs 79、179 vs 157），与 §12-6 实测的 2.4× reset 成本差同向。

---

## 11. plan 执行状态总结（2026-08-16）

| 阶段 | 状态 |
|---|---|
| §2 Plan / §3 G1 | ✅ APPROVED（R5） |
| §4 Code / §5 G2 | ✅ APPROVED（R6） |
| §6 Verify | ✅ 72 passed, 2 skipped |
| §7 Commit / §8 Push | ✅ `15dfa67` → `origin/Ziyang` |
| **§9.1 G0-A 接线等价性** | ✅ **PASS**（三路径 bit 相同 + 随机性反向对照） |
| **§9.1 G0-B 行为健全性** | ✅ **PASS**（owner 裁定修正判据后，三条口径一致） |
| **§5.3 S1–S4** | ✅ **全部通过**（冒烟 11/12） |
| §9.2 退路前置查证 | ✅ 完成（`gated=auto`，无人工审批；checkpoint 不自包含 tokenizer） |

**plan 的可执行部分至此全部完成。** 后续属于新的工作范围（正式跨场景实验按 §12-5 规模、以及 GR00T 版 Interceptor/KeyBuilder 的 cache 接入），不在本 plan 内。

⚠ **未提交**：§10 执行记录、import 修正与新增的孤岛 A 测试均在工作树，按 owner 指示未 commit。
