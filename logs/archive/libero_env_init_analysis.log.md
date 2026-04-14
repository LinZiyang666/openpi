# LIBERO 环境初始化分析报告

## 当前 `main.py` 使用的初始化参数（仅3个）

在 `_get_libero_env`（第460-467行）中：

```python
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": resolution,
    "camera_widths": resolution,
}
env = OffScreenRenderEnv(**env_args)
```

其余全部使用默认值，然后通过 `set_init_state(initial_state)` 加载**预存的固定初始状态**。

## OffScreenRenderEnv 实际支持但未使用的参数

通过查看 `ControlEnv.__init__` 和 `BDDLBaseDomain.__init__`，有大量未暴露的选项：

| 类别 | 参数 | 默认值 | 作用 |
|------|------|--------|------|
| **机器人** | `robots` | `["Panda"]` | 机器人型号 |
| **控制** | `controller` | `"OSC_POSE"` | 控制方式 |
| | `control_freq` | `20` | 控制频率(Hz) |
| | `initialization_noise` | `None` | 机器人初始位姿噪声 |
| **场景** | `arena_type` | `"table"` | 场景类型(table/kitchen/floor/living_room等) |
| | `scene_xml` | 默认场景 | 自定义场景XML |
| | `scene_properties` | `{}` | 地板/墙壁风格等 |
| | `table_full_size` | `(1.0, 1.0, 0.05)` | 桌子尺寸 |
| | `workspace_offset` | `(0, 0, 0)` | 工作空间偏移 |
| **物体随机化** | `placement_initializer` | `None` | 自定义放置采样器 |
| | `object_property_initializers` | `None` | 物体属性采样器(开/关状态等) |
| **观测** | `use_object_obs` | `True` | 是否包含物体状态观测 |
| | `camera_depths` | `False` | 深度图 |
| | `camera_segmentations` | `None` | 语义分割 |
| **渲染** | `render_gpu_device_id` | `-1` | GPU渲染设备 |
| **仿真** | `horizon` | `1000` | 最大步数 |
| | `hard_reset` | `True` | 是否硬重置 |

## 初始状态文件详情

### 文件位置

初始状态存储在 libero 包内：
```
{libero_package}/libero/init_files/{suite_name}/{task_name}.pruned_init
```

实际路径：`/home/weiland/anaconda3/envs/libero_sim/lib/python3.8/site-packages/libero/libero/init_files/`

每个任务同时有 `.init`（原始完整版）和 `.pruned_init`（裁剪版）两个文件，代码使用的是 `.pruned_init`。

### 两种初始状态文件

LIBERO 官方为每个 task 提供了**两种**初始状态文件：

| 文件类型 | 后缀 | 说明 |
|----------|------|------|
| **完整版** | `.init` | 通过反复 `env.reset()` + `env.sim.get_state().flatten()` 采集的全部初始状态 |
| **裁剪版** | `.pruned_init` | 从 `.init` 中筛选出的子集，用于标准评估 |

`main.py` 当前使用的是 `.pruned_init`（在 `benchmark/__init__.py` 第76行硬编码）。

### 各 Suite 初始状态数量汇总

| Suite | Tasks 数 | `.init` 每 task 数量 | `.pruned_init` 每 task 数量 | 说明 |
|-------|---------|---------------------|---------------------------|------|
| libero_spatial | 10 | **100** | 50 | 完整版有100个，裁剪版50个 |
| libero_object | 10 | 50 | 50 | 完整版和裁剪版相同 |
| libero_goal | 10 | **无** | 50 | 没有 .init 文件 |
| libero_10 | 10 | **100** | 50 | 完整版有100个，裁剪版50个 |
| libero_90 | 90 | **100** | 50 | 完整版有100个，裁剪版50个 |

已验证：**所有有 `.init` 文件的 suite，其 `.pruned_init` 都是 `.init` 的严格子集**（逐行精确匹配）。

**关键发现**：libero_spatial、libero_10、libero_90 的 `.init` 文件实际上有 **100 个**初始状态可用，但 `main.py` 只使用了 `.pruned_init` 中的 50 个。如果要使用全部 100 个初始状态，需要修改 `benchmark/__init__.py` 中的 `init_states_file` 字段，将 `.pruned_init` 改为 `.init`。

### 各 Suite 初始状态详细列表（.pruned_init）

状态维度因场景复杂度不同而不同（反映场景中物体数量）。

#### libero_spatial（10 tasks，每 task 50 states，state_dim=92）

| Task | States | State Dim |
|------|--------|-----------|
| pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate | 50 | 92 |
| pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 50 | 92 |
| pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate | 50 | 92 |
| pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate | 50 | 92 |
| pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate | 50 | 92 |
| pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate | 50 | 92 |
| pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate | 50 | 92 |
| pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate | 50 | 92 |
| pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate | 50 | 92 |
| pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate | 50 | 92 |

#### libero_object（10 tasks，每 task 50 states，state_dim=110）

| Task | States | State Dim |
|------|--------|-----------|
| pick_up_the_alphabet_soup_and_place_it_in_the_basket | 50 | 110 |
| pick_up_the_bbq_sauce_and_place_it_in_the_basket | 50 | 110 |
| pick_up_the_butter_and_place_it_in_the_basket | 50 | 110 |
| pick_up_the_chocolate_pudding_and_place_it_in_the_basket | 50 | 110 |
| pick_up_the_cream_cheese_and_place_it_in_the_basket | 50 | 110 |
| pick_up_the_ketchup_and_place_it_in_the_basket | 50 | 110 |
| pick_up_the_milk_and_place_it_in_the_basket | 50 | 110 |
| pick_up_the_orange_juice_and_place_it_in_the_basket | 50 | 110 |
| pick_up_the_salad_dressing_and_place_it_in_the_basket | 50 | 110 |
| pick_up_the_tomato_sauce_and_place_it_in_the_basket | 50 | 110 |

#### libero_goal（10 tasks，每 task 50 states，state_dim=79）

| Task | States | State Dim |
|------|--------|-----------|
| open_the_middle_drawer_of_the_cabinet | 50 | 79 |
| open_the_top_drawer_and_put_the_bowl_inside | 50 | 79 |
| push_the_plate_to_the_front_of_the_stove | 50 | 79 |
| put_the_bowl_on_the_plate | 50 | 79 |
| put_the_bowl_on_the_stove | 50 | 79 |
| put_the_bowl_on_top_of_the_cabinet | 50 | 79 |
| put_the_cream_cheese_in_the_bowl | 50 | 79 |
| put_the_wine_bottle_on_the_rack | 50 | 79 |
| put_the_wine_bottle_on_top_of_the_cabinet | 50 | 79 |
| turn_on_the_stove | 50 | 79 |

#### libero_10（10 tasks，每 task 50 states，state_dim 不统一）

| Task | States | State Dim |
|------|--------|-----------|
| KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 50 | 47 |
| KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 50 | 51 |
| KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 50 | 47 |
| KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 50 | 47 |
| LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 50 | 84 |
| LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 50 | 123 |
| LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 50 | 123 |
| LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate | 50 | 84 |
| LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate | 50 | 71 |
| STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 50 | 45 |

#### libero_90（90 tasks，每 task 50 states，state_dim 按场景不同）

| 场景 | Tasks 数 | State Dim | 说明 |
|------|----------|-----------|------|
| KITCHEN_SCENE1 | 5 | 51 | 柜子+碗+盘 |
| KITCHEN_SCENE2 | 6 | 77 | 柜子+多碗+盘 |
| KITCHEN_SCENE3 | 4 | 47 | 炉灶+锅 |
| KITCHEN_SCENE4 | 6 | 51 | 柜子+碗+酒瓶 |
| KITCHEN_SCENE5 | 5 | 64 | 柜子+碗+盘+番茄酱 |
| KITCHEN_SCENE6 | 2 | 47 | 微波炉+杯子 |
| KITCHEN_SCENE7 | 3 | 47 | 微波炉+碗+盘 |
| KITCHEN_SCENE8 | 2 | 47 | 炉灶+摩卡壶 |
| KITCHEN_SCENE9 | 6 | 47 | 炉灶+锅+碗+柜架 |
| KITCHEN_SCENE10 | 6 | 77 | 柜子+碗+黄油+巧克力布丁 |
| LIVING_ROOM_SCENE1 | 4 | 84 | 篮子+食物 |
| LIVING_ROOM_SCENE2 | 5 | 123 | 篮子+更多食物 |
| LIVING_ROOM_SCENE3 | 5 | 97 | 托盘+食物 |
| LIVING_ROOM_SCENE4 | 5 | 84 | 托盘+碗+食物 |
| LIVING_ROOM_SCENE5 | 4 | 84 | 盘子+杯子 |
| LIVING_ROOM_SCENE6 | 4 | 71 | 盘子+杯子+布丁 |
| STUDY_SCENE1 | 4 | 45 | 书+收纳架 |
| STUDY_SCENE2 | 4 | 45 | 书+收纳架 |
| STUDY_SCENE3 | 5 | 58 | 书+收纳架+杯子 |
| STUDY_SCENE4 | 5 | 58 | 书+柜架 |

### 初始状态的局限性

1. **`main.py` 使用 `.pruned_init`（50个）**，但部分 suite 的 `.init` 文件有 **100 个**可用状态未被利用
2. **`num_trials_per_task=50`** 直接用 `initial_states[episode_idx]` 索引，如果设置超过预存数量会越界
3. **`env.seed(seed)` 虽然设了种子**，但因为后面直接 `set_init_state()` 覆盖了状态，seed 的随机化效果有限（注释也提到 seed 仍会影响物体位置，说明行为不完全确定）
4. **状态是 MuJoCo 扁平化向量**，直接设定所有 qpos/qvel，完全跳过环境自带的 placement_initializer 随机化逻辑
5. **如需更多初始状态**，可通过反复 `env.reset()` + `env.sim.get_state().flatten()` 自行生成，LIBERO 官方未提供专门的生成脚本

## 类继承结构

```
OffScreenRenderEnv (env_wrapper.py)
    └─ ControlEnv (env_wrapper.py)
        └─ BDDLBaseDomain (bddl_base_domain.py)
```

- `OffScreenRenderEnv.__init__` 仅强制 `has_renderer=False`, `has_offscreen_renderer=True`，其余委托给 `ControlEnv`
- `ControlEnv` 解析 BDDL 文件，通过 `TASK_MAPPING` 创建对应的任务环境
- `BDDLBaseDomain` 处理物体放置、属性初始化、场景加载等

## 关键方法

### `set_init_state` (env_wrapper.py)
```python
def set_init_state(self, init_state):
    return self.regenerate_obs_from_state(init_state)

def regenerate_obs_from_state(self, mujoco_state):
    self.set_state(mujoco_state)
    self.env.sim.forward()
    self.check_success()
    self._post_process()
    self._update_observables(force=True)
    return self.env._get_observations()
```
直接设置 MuJoCo 状态向量，跳过所有随机化逻辑。

### `reset` (env_wrapper.py)
```python
def reset(self):
    success = False
    while not success:
        try:
            ret = self.env.reset()
            success = True
        except RandomizationError:
            pass
    return ret
```
会触发 `_reset_internal`，执行物体放置采样和属性初始化。

### `_reset_internal` (bddl_base_domain.py)
处理物体放置和属性初始化：
- 通过 `placement_initializer` 采样物体位置
- 通过 `object_property_initializers` 采样物体属性（开/关状态等）
- 支持条件放置（物体在固定装置上、物体在物体上、物体在容器内）

## 可用的放置采样器

- **MultiRegionRandomSampler** — 在定义区域内随机放置，支持 x/y 范围、旋转、z 偏移
- **SiteRegionRandomSampler** — 在目标位点上放置
- **ObjectBasedSampler** — 在其他物体上放置
- **InSiteRegionRandomSampler** — 在容器内部放置

## 可用的属性采样器

- **OpenCloseSampler** — 随机化铰接物体的开/关状态
- **TurnOnOffSampler** — 随机化物体的开关状态

## 可用的场景类型 (arena_type)

- `"table"` — 桌面场景（默认）
- `"kitchen"` — 厨房桌面
- `"floor"` — 空地板
- `"coffee_table"` — 咖啡桌
- `"living_room"` — 客厅桌面
- `"study"` — 书房桌面

## 总结

`main.py` 本质上是一个**纯评估脚本**，使用预存的确定性初始状态做 benchmark 测试。如果需要更丰富的初始化（比如随机化物体位置、改变场景、调整机器人噪声），需要利用上述未使用的参数，或者不调用 `set_init_state` 而让环境自己通过 `placement_initializer` 随机采样。
