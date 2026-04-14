# LIBERO Environment Initialization Analysis Report

## Current `main.py` Initialization Parameters (Only 3)

In `_get_libero_env` (lines 460-467):

```python
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": resolution,
    "camera_widths": resolution,
}
env = OffScreenRenderEnv(**env_args)
```

All other parameters use defaults, then **pre-stored fixed initial states** are loaded via `set_init_state(initial_state)`.

## OffScreenRenderEnv Parameters Supported But Not Used

By examining `ControlEnv.__init__` and `BDDLBaseDomain.__init__`, there are numerous unexposed options:

| Category | Parameter | Default | Purpose |
|----------|-----------|---------|---------|
| **Robot** | `robots` | `["Panda"]` | Robot model |
| **Control** | `controller` | `"OSC_POSE"` | Control mode |
| | `control_freq` | `20` | Control frequency (Hz) |
| | `initialization_noise` | `None` | Robot initial pose noise |
| **Scene** | `arena_type` | `"table"` | Scene type (table/kitchen/floor/living_room, etc.) |
| | `scene_xml` | Default scene | Custom scene XML |
| | `scene_properties` | `{}` | Floor/wall styles, etc. |
| | `table_full_size` | `(1.0, 1.0, 0.05)` | Table dimensions |
| | `workspace_offset` | `(0, 0, 0)` | Workspace offset |
| **Object Randomization** | `placement_initializer` | `None` | Custom placement sampler |
| | `object_property_initializers` | `None` | Object property sampler (open/closed state, etc.) |
| **Observation** | `use_object_obs` | `True` | Whether to include object state observations |
| | `camera_depths` | `False` | Depth maps |
| | `camera_segmentations` | `None` | Semantic segmentation |
| **Rendering** | `render_gpu_device_id` | `-1` | GPU rendering device |
| **Simulation** | `horizon` | `1000` | Maximum steps |
| | `hard_reset` | `True` | Whether to hard reset |

## Initial State File Details

### File Location

Initial states are stored inside the libero package:
```
{libero_package}/libero/init_files/{suite_name}/{task_name}.pruned_init
```

Actual path: `/home/weiland/anaconda3/envs/libero_sim/lib/python3.8/site-packages/libero/libero/init_files/`

Each task has both an `.init` (original full version) and a `.pruned_init` (trimmed version) file; the code uses `.pruned_init`.

### Two Types of Initial State Files

LIBERO officially provides **two types** of initial state files for each task:

| File Type | Suffix | Description |
|-----------|--------|-------------|
| **Full version** | `.init` | All initial states collected via repeated `env.reset()` + `env.sim.get_state().flatten()` |
| **Trimmed version** | `.pruned_init` | Subset filtered from `.init`, used for standard evaluation |

`main.py` currently uses `.pruned_init` (hardcoded in `benchmark/__init__.py` line 76).

### Initial State Count Summary by Suite

| Suite | Tasks | `.init` per task | `.pruned_init` per task | Notes |
|-------|-------|------------------|-------------------------|-------|
| libero_spatial | 10 | **100** | 50 | Full version has 100, trimmed has 50 |
| libero_object | 10 | 50 | 50 | Full and trimmed versions are identical |
| libero_goal | 10 | **None** | 50 | No .init files |
| libero_10 | 10 | **100** | 50 | Full version has 100, trimmed has 50 |
| libero_90 | 90 | **100** | 50 | Full version has 100, trimmed has 50 |

Verified: **For all suites that have `.init` files, their `.pruned_init` is a strict subset of `.init`** (exact line-by-line match).

**Key finding**: libero_spatial, libero_10, and libero_90 `.init` files actually have **100** initial states available, but `main.py` only uses 50 from `.pruned_init`. To use all 100 initial states, the `init_states_file` field in `benchmark/__init__.py` needs to be changed from `.pruned_init` to `.init`.

### Detailed Initial State Lists by Suite (.pruned_init)

State dimensions vary by scene complexity (reflecting the number of objects in the scene).

#### libero_spatial (10 tasks, 50 states each, state_dim=92)

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

#### libero_object (10 tasks, 50 states each, state_dim=110)

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

#### libero_goal (10 tasks, 50 states each, state_dim=79)

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

#### libero_10 (10 tasks, 50 states each, non-uniform state_dim)

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

#### libero_90 (90 tasks, 50 states each, state_dim varies by scene)

| Scene | Tasks | State Dim | Description |
|-------|-------|-----------|-------------|
| KITCHEN_SCENE1 | 5 | 51 | Cabinet + bowl + plate |
| KITCHEN_SCENE2 | 6 | 77 | Cabinet + multiple bowls + plate |
| KITCHEN_SCENE3 | 4 | 47 | Stove + pot |
| KITCHEN_SCENE4 | 6 | 51 | Cabinet + bowl + wine bottle |
| KITCHEN_SCENE5 | 5 | 64 | Cabinet + bowl + plate + ketchup |
| KITCHEN_SCENE6 | 2 | 47 | Microwave + mug |
| KITCHEN_SCENE7 | 3 | 47 | Microwave + bowl + plate |
| KITCHEN_SCENE8 | 2 | 47 | Stove + moka pot |
| KITCHEN_SCENE9 | 6 | 47 | Stove + pot + bowl + shelf |
| KITCHEN_SCENE10 | 6 | 77 | Cabinet + bowl + butter + chocolate pudding |
| LIVING_ROOM_SCENE1 | 4 | 84 | Basket + food |
| LIVING_ROOM_SCENE2 | 5 | 123 | Basket + more food |
| LIVING_ROOM_SCENE3 | 5 | 97 | Tray + food |
| LIVING_ROOM_SCENE4 | 5 | 84 | Tray + bowl + food |
| LIVING_ROOM_SCENE5 | 4 | 84 | Plate + mug |
| LIVING_ROOM_SCENE6 | 4 | 71 | Plate + mug + pudding |
| STUDY_SCENE1 | 4 | 45 | Book + caddy |
| STUDY_SCENE2 | 4 | 45 | Book + caddy |
| STUDY_SCENE3 | 5 | 58 | Book + caddy + mug |
| STUDY_SCENE4 | 5 | 58 | Book + shelf |

### Limitations of Initial States

1. **`main.py` uses `.pruned_init` (50 states)**, but some suites have `.init` files with **100** available states that are not utilized
2. **`num_trials_per_task=50`** directly indexes `initial_states[episode_idx]`; setting it above the pre-stored count will cause an out-of-bounds error
3. **`env.seed(seed)` sets a seed**, but since `set_init_state()` overwrites the state afterwards, the seed's randomization effect is limited (comments also note that the seed still affects object positions, meaning behavior is not fully deterministic)
4. **States are flattened MuJoCo vectors** that directly set all qpos/qvel, completely bypassing the environment's built-in placement_initializer randomization logic
5. **To generate more initial states**, one can repeatedly call `env.reset()` + `env.sim.get_state().flatten()`; LIBERO does not provide a dedicated generation script

## Class Inheritance Structure

```
OffScreenRenderEnv (env_wrapper.py)
    └─ ControlEnv (env_wrapper.py)
        └─ BDDLBaseDomain (bddl_base_domain.py)
```

- `OffScreenRenderEnv.__init__` only forces `has_renderer=False`, `has_offscreen_renderer=True`; everything else is delegated to `ControlEnv`
- `ControlEnv` parses the BDDL file and creates the corresponding task environment via `TASK_MAPPING`
- `BDDLBaseDomain` handles object placement, property initialization, scene loading, etc.

## Key Methods

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
Directly sets the MuJoCo state vector, bypassing all randomization logic.

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
Triggers `_reset_internal`, which executes object placement sampling and property initialization.

### `_reset_internal` (bddl_base_domain.py)
Handles object placement and property initialization:
- Samples object positions via `placement_initializer`
- Samples object properties via `object_property_initializers` (open/closed state, etc.)
- Supports conditional placement (object on fixture, object on object, object in container)

## Available Placement Samplers

- **MultiRegionRandomSampler** — Random placement within defined regions, supports x/y range, rotation, z offset
- **SiteRegionRandomSampler** — Placement on target sites
- **ObjectBasedSampler** — Placement on other objects
- **InSiteRegionRandomSampler** — Placement inside containers

## Available Property Samplers

- **OpenCloseSampler** — Randomize articulated object open/closed state
- **TurnOnOffSampler** — Randomize object on/off state

## Available Scene Types (arena_type)

- `"table"` — Tabletop scene (default)
- `"kitchen"` — Kitchen counter
- `"floor"` — Empty floor
- `"coffee_table"` — Coffee table
- `"living_room"` — Living room table
- `"study"` — Study desk

## Summary

`main.py` is essentially a **pure evaluation script** that uses pre-stored deterministic initial states for benchmark testing. To achieve richer initialization (e.g., randomizing object positions, changing scenes, adjusting robot noise), one needs to leverage the unused parameters described above, or skip calling `set_init_state` and let the environment sample randomly via `placement_initializer` on its own.
