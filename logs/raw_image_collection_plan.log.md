# 原始图片收集功能设计方案

**状态**: Plan  
**日期**: 2026-04-09  
**范围**: 仅 `--collect` 系统。Cache 框架零改动。

---

## 1. 需求分析

### 目标
在 `--collect` 数据收集过程中，保存每一步的**模型输入图片**（transform 后的 RGB uint8 像素，224×224），而非仅保存 embedding。

### 范围界定
- **保存的是什么**：`input_transform` 之后的 3 个 canonical image slot（`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`）中 `image_mask=True` 的图片。
- **不是什么**：不是环境原始分辨率图片，也不是环境中所有摄像头的图片。例如 ALOHA 的 `cam_low` 不进模型输入（见 `aloha_policy.py:49-76`），因此不会被保存。
- **适配逻辑**：通过 `image_mask` 过滤 padding 图。Libero 保存 2 张（`right_wrist_0_rgb` mask=False），ALOHA 保存 3 张。这不是"自动适配任意摄像头"，而是"自动过滤模型输入中的 padding slot"。

### 约束
- **仅改 `--collect` 系统**：`src/openpi/collect/` 下两个文件 + 一个共用工具函数
- **Cache 框架零改动**：KeyBuilder 当前不需要图片，不往 `cached_data` 塞数据。未来如需扩展，再通过 `**stage_outputs` kwargs 透传（一行代码的事，见附录 A）
- **低开销**：仅在 `--collect` 启用时执行图片提取

---

## 2. 改动方案

### 2.1 数据流

```
CollectionPolicy.infer(obs)
  │
  ├─ _extract_obs_fields(obs)  ←  一次 transform，同时提取：
  │   ├─ robot_state (float32)
  │   └─ input_images: {"base_0_rgb": (224,224,3) uint8, ...}
  │       仅包含 image_mask=True 的 slot
  │
  ├─ 注册 forward hooks → 调用 self._policy.infer(obs) → 收集 embeddings
  │
  └─ record_inference(InferenceEmbeddings)  ← 附带 input_images
        │
        └─ EpisodeDataCollector 缓存到 buffer
              │
              └─ on_episode_end() → HDF5 写入，包含 raw_images/ group
```

### 2.2 文件变更

#### `src/openpi/collect/data_collector.py`

**InferenceEmbeddings** 新增字段：

```python
@dataclass
class InferenceEmbeddings:
    vision_embs: list[np.ndarray]
    prompt_emb: np.ndarray
    robot_state: np.ndarray
    noise_action_steps: list[np.ndarray]
    clean_action: np.ndarray
    # ── 新增 ──
    input_images: dict[str, np.ndarray] | None = None
    # {"base_0_rgb": (224,224,3) uint8, "left_wrist_0_rgb": ...}
    # 仅包含 image_mask=True 的 slot，None 表示未启用图片收集
```

**HDF5 写入逻辑**新增（在 `on_episode_end()` 的 step 循环中）：

```python
if embs.input_images:
    img_grp = grp.create_group("input_images")
    for key, img in embs.input_images.items():
        img_grp.create_dataset(key, data=img, compression="lzf")
```

#### `src/openpi/collect/collection_policy.py`

**合并 transform 调用**，替换现有 `_extract_robot_state()`：

```python
def _extract_obs_fields(self, obs: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """一次 transform，同时提取 robot_state 和有效图片。
    
    Returns:
        robot_state: (state_dim,) float32
        input_images: {slot_name: (H,W,3) uint8} 仅 mask=True 的 slot
    """
    inputs = self._input_transform(jax.tree.map(lambda x: x, obs))
    
    robot_state = np.asarray(inputs["state"], dtype=np.float32).flatten()
    input_images = extract_valid_images(inputs)
    
    return robot_state, input_images
```

调用点修改：

```python
# 旧:
robot_state_np = self._extract_robot_state(obs)
# 新:
robot_state_np, input_images = self._extract_obs_fields(obs)
```

`_record()` 调用也对应传入 `input_images`。

#### `src/openpi/shared/image_utils.py`（新建）

```python
"""图片提取工具，从 input_transform 输出中过滤有效图片。"""

import numpy as np

# 模型固定的 3 个 image slot（见 model.py:40-42）
MODEL_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def extract_valid_images(
    inputs: dict,
    image_keys: tuple[str, ...] = MODEL_IMAGE_KEYS,
) -> dict[str, np.ndarray]:
    """从 input_transform 输出中提取 image_mask=True 的模型输入图片。
    
    注意：返回的是 transform 后的图片（224×224），不是环境原始分辨率。
    ALOHA 的 cam_low 等不进入模型输入的摄像头不会被提取。
    
    Args:
        inputs: input_transform 的输出，包含 "image" 和 "image_mask"
        image_keys: 要检查的 image slot
    
    Returns:
        {slot_name: (H, W, 3) uint8 ndarray} 仅有效 slot
    """
    images = inputs.get("image", {})
    masks = inputs.get("image_mask", {})
    result = {}
    for key in image_keys:
        if key in images and bool(masks.get(key, False)):
            img = np.asarray(images[key])
            if np.issubdtype(img.dtype, np.floating):
                img = (img * 255).astype(np.uint8)
            elif img.dtype != np.uint8:
                img = img.astype(np.uint8)
            result[key] = img
    return result
```

---

## 3. HDF5 Schema 变更

### 变更前（现有）

```
episode_XXXX.h5
├── attrs: experiment_name, task, episode_id, num_steps, timestamp, success
├── step_0000/
│   ├── vision_0          (float16, embedding)
│   ├── vision_1          (float16, embedding)
│   ├── vision_2          (float16, embedding)
│   ├── prompt_emb        (float16, embedding)
│   ├── robot_state       (float32)
│   ├── noise_action_1..N (float32)
│   └── clean_action      (float32)
└── step_0001/ ...
```

### 变更后

```
episode_XXXX.h5
├── attrs: experiment_name, task, episode_id, num_steps, timestamp, success
├── step_0000/
│   ├── vision_0          (float16, embedding)
│   ├── ...
│   ├── clean_action      (float32)
│   └── input_images/                    ← 新增 group
│       ├── base_0_rgb       (uint8, 224×224×3, lzf 压缩)
│       └── left_wrist_0_rgb (uint8, 224×224×3, lzf 压缩)
│       # right_wrist_0_rgb 不存在（Libero 下 mask=False）
└── step_0001/ ...
```

### 单步体积估算

| 数据 | 大小 |
|------|------|
| 现有 embeddings + actions | ~2.5 MB（float16 vision × 3 + prompt + state + actions） |
| 新增图片（Libero, 2 张） | ~295 KB raw，lzf 压缩后约 ~200 KB |
| 新增图片（ALOHA, 3 张） | ~442 KB raw，lzf 压缩后约 ~300 KB |

图片占总体积 < 15%，开销可接受。

---

## 4. 不动什么

| 组件 | 改动 | 说明 |
|------|------|------|
| `collection_policy.py` | ✅ 修改 | 合并 transform、提取图片 |
| `data_collector.py` | ✅ 修改 | 新增字段 + HDF5 写入 |
| `shared/image_utils.py` | ✅ 新建 | 共用工具函数 |
| Cache 框架全部 | ❌ 不动 | KeyBuilder / Orchestrator / Storage / Gate / Judge 零改动 |
| `serve_policy.py` | ❌ 不动 | `--collect` 已有，无需新 CLI 参数 |
| 模型代码 | ❌ 不动 | |
| 客户端代码 | ❌ 不动 | |

---

## 5. 实施步骤

### Phase 1: 工具函数
1. 创建 `src/openpi/shared/image_utils.py`，实现 `extract_valid_images()`

### Phase 2: 数据收集改动
1. `data_collector.py`：`InferenceEmbeddings` 新增 `input_images` 字段
2. `data_collector.py`：HDF5 写入逻辑新增 `input_images/` group
3. `collection_policy.py`：`_extract_robot_state()` → `_extract_obs_fields()`，合并 transform
4. `collection_policy.py`：`_record()` 传入 `input_images`

### Phase 3: 文档更新
1. 更新 `docs/data_collection_guide.md`：HDF5 schema、新增字段说明、体积估算
2. 更新 `logs/README.md` 索引

### Phase 4: 验证
1. 单元测试：`extract_valid_images()` 对 Libero（2 张）和 ALOHA（3 张）的 mask 过滤
2. 集成测试：`--collect` 写出的 HDF5 包含 `input_images/` group，key 数量正确
3. 回归测试：不开 `--collect` 时行为不变

---

## 附录 A：未来 Cache 框架扩展路径

如果未来 KeyBuilder 需要使用图片，扩展方式如下（当前不实施）：

```python
# interceptor.py — 多传一个 kwarg
input_images = extract_valid_images(inputs)
orchestrator.check(CP1, stage1=stage1, input_images=input_images)

# orchestrator.py — 零改动，**stage_outputs 自动透传
# key_builder.collect(CP1, stage1=stage1, input_images=input_images)

# 新 KeyBuilder 子类 — 按需读取
def collect(self, checkpoint_id, **stage_outputs):
    super().collect(checkpoint_id, **stage_outputs)
    if "input_images" in stage_outputs:
        self._cache["input_images"] = stage_outputs["input_images"]
```

**注意事项**（届时需要解决）：
- `cached_data` 当前类型契约是 `dict[str, torch.Tensor]`（GPU tensor）。numpy 图片混入会破坏语义。解决方案：要么单独加一个 `cached_images` 属性，要么将协议放宽为 `dict[str, torch.Tensor | np.ndarray]` 并在文档中标注。
- Gate/Judge 如果要读图片，需要明确处理 numpy 类型。
