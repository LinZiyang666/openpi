# PickPlace 定物体重做 —— plan

> 状态:**In Progress**|阶段:Understand ✅ → Plan ✅ → **G1 ✅ APPROVED**(2026-09-05,4 轮 15 blocking 全闭合)→ **Code 🔄** → G2 ⬚ → Verify ⬚
> 授权:Execution|层级:**L3**(改外部依赖源码 + 新增 artifact 身份位 + 重采集 + 重建库 + 重评测 + 架构文档更新)
> 前序:`logs/robocasa365_ws_search2_text_ivf_plan.log.md`、结果 `exp/robocasa365/analysis/ws_search2_groot_results.md`

## 0. 一句话目标

把 **5 个 PickPlace 任务**场景里的**每一个物体槽**(主物 + 干扰物 + 容器 + 隐式容器,共 **13 个**)
钉死到**同一个具体实例**——⚠ **是一个物体(单个 mesh),不是一个类别**;
**位姿保持随机**,重新采集 → 重建 PickPlace-only 库 → 重跑 132 格权重矩阵,
看纯检索成功率能否从当前水平提升。

⚠ **口径**(owner 2026-09-05 明确):钉死粒度 = **实例级**。`obj_groups="plate"` 只锁到类别,
类别内仍有 5 个实例在 `kitchen_object_utils.py:469` 被 `rng.choice` 抽 —— **不满足要求**。
必须钉到 `.../objaverse/plate/plate_9/model.xml` 这一级。

**范围**:`PickPlaceCounterToCabinet`、`PickPlaceCounterToStove`、`PickPlaceDrawerToCounter`、
`PickPlaceSinkToCounter`、`PickPlaceToasterToCounter`。
**不在范围**:其余 8 个任务;`CoffeeSetupMug`(owner 2026-09-05 明确移出)。

---

## 1. 已核实事实

全部带 `文件:行号`;标 **实测** 者为本轮亲自运行/读取所得(四路侦察 + 我方复核)。

### 1.1 出发点:检索还原率

teacher SR 由采集 h5 的 `success` 属性实测(同一 l1s1 场景):

| | cache SR | teacher SR | **还原率** |
|---|---|---|---|
| PickPlace 5 个(GR00T) | 0.066 | 0.698 | **0.10** |
| PickPlace 5 个(pi0.5) | 0.020 | 0.670 | **0.03** |
| 其余 7 个(GR00T) | 0.463 | 0.638 | 0.84 |

**F1** teacher 对 PickPlace 并不弱(0.70),是检索只还原了 10%。

### 1.2 补丁面:robocasa 部署形态与改动点

**F2 ⚠ 补丁落点不是本地那份 robocasa。实测**:
`/home/weiland/projects/robocasa` **未被任何 venv 引用**(openpi venv `import robocasa` → `ModuleNotFoundError`;
全盘无指向它的 `.pth`/`.egg-link`),且 `robocasa/models/assets/objects/` **不存在**。
三台生产机实际使用的是各自的
`<HOME|/scratch/zixuans8>/Isaac-GR00T/external_dependencies/robocasa365`,
以 `__editable__.robocasa-1.0.1.pth` 装进岛 A 的 uv venv(python 3.12)。
**四份 checkout 的 HEAD 完全一致 = `be22d659b02db8f6d7f3a3c3edc742934fdcbaae`**,origin 同为
`github.com/robocasa/robocasa`,**全部处于 detached HEAD**。
`external_dependencies/robocasa`(submodule,指向 `squarefk/robocasa`)在三台上**都是空目录**,与本线无关。

**F3 ⚠ 三机工作树并非全干净。实测**:timan107 / timan1 各 1 条未跟踪(`robocasa/models/assets/README.md`,无害);
**weilandserver 有 3 条**:` M robocasa/scripts/bench_speed.py`(diff 仅 2 行,`camera_heights/widths` 84→128)、
`?? robocasa/models/assets/README.md`、`?? robocasa/scripts/bench_speed.py.orig`。不影响物体采样路径,
⚠ **裁定:这三条是他人改动,一律隔离保留,不删不改**(W1 的补丁只触碰 `kitchen.py`,与之无冲突面)。
等价性不靠清空工作树来证明,而靠 **apply 前后各一次 `git status --short` 对账** + 三个 SHA(W1)。

**F4** `Kitchen.__init__` = `kitchen.py:362-412`,**末尾没有 `**kwargs`**(`grep -n "def __init__" kitchen.py` 全文仅此一处)。
kwargs 直通链为 `gym.make` → `gym_wrapper.py:408-411` → `:144-152` → `:169-176` → `env_utils.create_env`(`:57`,`:79`,`:135`)
→ `robosuite.make`(`:138`)→ 任务类 → `Kitchen.__init__`。⇒ **走新 kwarg 路线必须改签名**,未知键直接 `TypeError`。

**F5 ⚠ 最小改动点只有一处。实测** `grep -rn _get_obj_cfgs`:全仓**唯一调用点**是
`kitchen.py:864`(`self.object_cfgs = self._get_obj_cfgs()`,在 `_create_objects` 的 `else` 分支内)。
在该行之后插入 override 即覆盖 5 个任务全部显式 slot,**且保留 `:875-972` 的隐式容器与 auxiliary 后处理**。
另一条零改动路线是 `set_ep_meta()` → `kitchen.py:854` 分支,但**该分支不建隐式容器**(`:854-862` 只做逐 cfg
`create_obj`),对 `PickPlaceCounterToStove` 反而更麻烦。**采用 `:864` 路线。**

**F6 ⚠ 只有 `cfg["info"]["mjcf_path"]` 这条路会跨机重定基**。`create_obj`(`env_utils.py:1403`)的
`"info" in cfg` 分支 `:1413-1427` 会 `new_path = os.path.join(robocasa.models.assets_root, "objects", mjcf_path.split("/objects/")[-1])`
(`:1424-1425`),再置 `obj_groups = new_path`、`exclude_obj_groups = None`。
**直接把绝对路径写进 `obj_groups` 不重定基**,跨机必然在 `kitchen_object_utils.py:381` `raise ValueError`。

**F7 ⚠ 精确 `.xml` 分支绕过的东西比原以为的多。实测**
`kitchen_object_utils.py:361-382` 分支体只做"反查类别 + 取 mjcf_kwargs",`else` 分支(`:383-473`)中被**完整跳过**的有:

| 跳过项 | 行号 | 后果(实测规模,`obj_registries=("objaverse","lightwheel")`) |
|---|---|---|
| `exclude_obj_groups` 归一化与过滤 | `:387-397` | `DrawerToCounter` 的 reamer/strainer/cheese_grater(`:1568`)与 `distr` 的 tool/utensil(`:1581`)排除**失效** |
| **7 个属性过滤**(graspable/washable/microwavable/cookable/fridgable/freezable/dishwashable) | `:420-433` | `CounterToStove.obj`(food+graspable+cookable)合法 **28/102** 类 ⇒ 74 类本应被拒;`SinkToCounter.obj` 合法 34 类;`CounterToCabinet.obj`(all+graspable)合法 **112/198** 类;`DrawerToCounter.obj` 合法 10 类 |
| **`split` 实例切分** | `:450-461` | `obj_instance_split="target"` **完全失效**,钉死一个 pretrain 实例零报错 |
| **RNG 消耗**(类别抽样 `:440`、registry 抽样 `:463-467`、实例抽样 `:469`) | — | ⚠ **钉死一个 slot 会位移后续所有 slot 与摆放采样的随机数序列** ⇒ 同 seed 下钉死前后位姿不可配对 |
| `rotate_upright` → `model_upright.xml` 替换 | `:470-471` | 见 F8 |

**兜底是 fail-loud**:路径不在 registry 中 → `:380-381 raise ValueError`,不会静默退化成随机抽样。
**未被跳过**(两分支共享 `:475-499`):`object_scale`、`groups_containing_sampled_obj`(由反查出的 `cat` 得出,
所以 F10 的隐式容器触发仍正常)、`info` 字典。⚠ `:495` 会把整条 xml 路径写进 `info["groups"]`;
`:497 "split": split` 写入一个**从未被应用过**的值(元数据说谎)。

**F8 ⚠ `rotate_upright` 只命中一个 slot,但形态确实不同。实测**:5 个任务中唯一 `rotate_upright=True` 的是
`PickPlaceToasterToCounter.obj`(`kitchen_pick_place.py:1035`)。全资产树 `model_upright.xml` 仅 **24 个**
(`lightwheel/sandwich_bread` 12 + `lightwheel/straw` 12),对 `model.xml` 共 3430 个。
两文件 `diff` 实测**内容不同**:upright 版给 visual/collision geom 加 90° 四元数,并把 `reg_bbox` 三轴尺寸轮换。
⇒ 传 `model.xml` 会让面包**平躺**而非竖插进烤面包机。**规避**:该 slot 直接钉 `model_upright.xml` 路径 ——
`:363` 的反查用 `dirname + "/model.xml"` 归一化,传 upright 路径**不会** `raise ValueError`。

**F9 `max_size` 是潜在雷不是活雷。实测** `grep -n max_size kitchen_pick_place.py` 仅 `:952`(属 `PickPlaceStoveToCounter`,
不在本轮 5 个任务内)⇒ **本轮 13 个 slot 全部没有 `max_size`**。但机制仍在:`sample_kitchen_object` 的
`while valid_object_sampled is False`(`:260-296`)精确分支照进,且分支**完全确定性、不消耗 rng**
⇒ 超限即**无限死循环**,不 raise、无重试上限(`kitchen.py:647` 的 50 次上限根本到不了)。
附带:`:282-288` 的 `ET.parse` + `reg_bbox` 查找是**无条件执行**的,钉死的 xml 若缺 `reg_bbox` geom 会 `AttributeError`。

**F10 ⚠ 隐式容器是每集必建,不是偶发**(修正前一稿的错误陈述)。**实测**
`OBJ_GROUPS["in_container"]`(`kitchen_objects.py:2986-2999`)含 117 个类别,`OBJ_GROUPS["food"]` 含 102 个,
**`food ⊆ in_container` 为 True**。`PickPlaceCounterToStove.obj` 的 `obj_groups` 默认 `"food"`(`:819`)
⇒ `kitchen.py:878-880` 的触发条件恒成立,`obj_container` **每集都建**。
其 cfg 构造在 `kitchen.py:881-886`:`name = cfg["name"] + "_container"`、
`obj_groups = cfg["placement"].get("try_to_place_in")`(→ `"container"` → `["plate"]`)、`placement = deepcopy(...)`,
**自身无任何属性约束、无 max_size、无 rotate_upright**;`:893-898` 会把 `try_to_place_in_kwargs` 的每个 k/v
**原样塞进 container_cfg**;建完后 `:908-914` 把原 obj 的 placement 改成 `size=(0.01,0.01)` + reference 到容器。
⇒ **钉死路线**:在 `obj` slot 的 `placement["try_to_place_in_kwargs"]` 里注入 `{"info": {"mjcf_path": ...}}`,
经 `:893-898` 落到 container_cfg 后走 F6 的重定基门。**不得**把路径直接写进 `try_to_place_in`(不重定基)。

### 1.3 5 个任务的 slot 全集(**冻结清单,共 13 个**)

文件 `robocasa/environments/kitchen/atomic/kitchen_pick_place.py`。未列出的 key 即为缺失;
`obj_groups` 缺失时由 `env_utils.py:1429` 兜底为 `"all"`。

| 任务 | slot | 行号 | obj_groups | exclude | 属性开关 | 其他 |
|---|---|---|---|---|---|---|
| CounterToCabinet(`:24`) | `obj` | `:78-94` | `self.obj_groups` → `"all"`(`:34`) | `self.exclude_obj_groups` → `None` | `graspable=True`(`:83`) | — |
| | `distr_counter` | `:97-111` | `"all"`(`:100`) | 缺失 | 无 | — |
| | `distr_cab` | `:112-123` | `"all"`(`:115`) | 缺失 | 无 | — |
| CounterToStove(`:811`) | `container` | `:854-865` | `("pan")` ⚠ **是字符串不是 tuple**(`:857` 无尾逗号) | 缺失 | 无 | `rotation=[(-3π/8,-π/4),(π/4,3π/8)]`(`:862`) |
| | `obj` | `:867-884` | `self.obj_groups` → `"food"`(`:819`) | `None` | `graspable`(`:872`)、`cookable`(`:873`) | `try_to_place_in="container"`(`:881`) |
| | **`obj_container`(隐式)** | `kitchen.py:881-886` | `"container"` → `["plate"]` | 无 | 无 | 每集必建(F10) |
| DrawerToCounter(`:1532`) | `obj` | `:1564-1576` | `("tool","utensil")`(`:1567`) | `("reamer","strainer","cheese_grater")`(`:1568`) | `graspable`(`:1569`) | ⚠ 不读 `self.obj_groups` |
| | `distr` | `:1578-1591` | **缺失** → `"all"` | `("tool","utensil")`(`:1581`) | 无 | — |
| SinkToCounter(`:365`) | `obj` | `:413-426` | `self.obj_groups` → `"food"`(`:373`) | `None` | `graspable`(`:418`)、`washable`(`:419`) | fixture=SINK ⇒ `env_utils.py:1444-1445` 另强制 `washable` |
| | `container` | `:427-441` | `"container"`(`:430`) | 缺失 | 无 | — |
| | `distr_counter` | `:444-459` | `"all"`(`:447`) | 缺失 | 无 | — |
| ToasterToCounter(`:994`) | `obj` | `:1031-1041` | `("sandwich_bread",)`(`:1034`) | 缺失 | 无 | **`rotate_upright=True`(`:1035`)**、`rotation=(0,0)`(`:1038`) |
| | `plate` | `:1042-1056` | `"plate"`(`:1045`) | 缺失 | `graspable=False`(`:1046`) | — |

**F11** `ToasterToCounter`(`:999-1000`)与 `DrawerToCounter`(`:1537-1538`)**接受 `obj_groups=` 但 cfg 里从不读**
⇒ 传进去静默无效。`PickPlace` 基类把 `obj_groups`/`exclude_obj_groups` 截留在 `self.` 上,`:18` 只转发 `*args, **kwargs`。

**F12** `obj_instance_split="target"` 在 **4 处硬编码**:`episode_runner.py:230`、`groot_rollout_client.py:228`、
`build_bucket_variants.py:94`、`baselines/pi05_step0b_client_ORIGINAL.py:37`。无任何 CLI/extra/run-plan 通路可改。

### 1.4 钉死后仍然在变的东西(owner 决定不锁,须在结果中标注)

**F13** 物体 xy/yaw(`env_utils.py:1262-1272`;`rotation` 缺省 **±π/4**,`:992`)。
**F14** 任务绑定到哪一个台面/柜子/抽屉(`kitchen.py:1709`、`:1745` 的 `rng.choice`)。
**F15** **accessory fixture 位姿**:layout001 中 `toaster [1.0,0.40]`、`coffee_machine [1.0,0.52]`、
`stand_mixer [0.90,0.25]`、`blender [0.5,0.25]` 每集重采(`kitchen.py:754-780`);
结构性 fixture(sink/fridge/stove/dishwasher/counter/cab_*)**无 placement 块,位姿固定**。
⇒ 本轮 5 个任务里**只有 `PickPlaceToasterToCounter` 的目的地会漂**。
**F16** 机器人底座 ±0.15m / ±0.05m(`kitchen.py:405-406` → `env_utils.py:1539`)。
**F17** `use_distractors` 是**死配置**(`kitchen.py:202/402/453`,全仓从未被读取);`clutter_mode` 默认 0 且
`create_env` 不暴露。fixture **不生成物体**(`grep MJCFObject( robocasa/models/fixtures/` 零命中)。

### 1.5 采集口径

**F18 ⚠ `run_collect.py` 全程没有"成功数"概念**:只派发静态 attempt 数(`--tasks A:N`),跑完即止。
"每任务 50 条成功"上一轮是**人工闭环**(跑一批 → 审计数成功 → 手算缺口 → 手写下一批)。

**F19 现成公式已在仓库内**:`verify_collection_artifacts.py:73-82`
`min_episodes_for_target(sr, *, target=20, confidence=0.90, cap=20000)` = 满足
`P(Binom(N, sr) ≥ target) ≥ confidence` 的最小 N(下尾用 `lgamma`,`:51-70`)。
docstring `:27-31` 明写用 **SR 点估计而非 Wilson 下界**("stacked conservatism explodes the budget;
deficits are covered by deterministic extension batches instead")。

**F20 上一轮实测产率与本轮首批派发量**(我方运行 `min_episodes_for_target(sr, target=50, confidence=0.90)`):

| task | groot_tp sr → N | pi05 sr → N |
|---|---|---|
| CounterToCabinet | 0.6279 → **89** | 0.6842 → **81** |
| CounterToStove | 0.7879 → **69** | 0.7727 → **70** |
| DrawerToCounter | 0.5618 → **100** | 0.5327 → **106** |
| SinkToCounter | 0.8281 → **65** | 0.9107 → **58** |
| ToasterToCounter | 0.6842 → **81** | 0.4496 → **126** |
| **合计** | **404** | **441** |

⚠ 这些 sr 来自**未钉死**分布,钉死后会变;首批数字是起点不是承诺,缺口由续批补(D6)。
容量核算:实测单集 ≈ 303 MB(groot)/ 427 MB(pi05)⇒ 全量 ≈ 310 GB;
weilandserver `/archive` **实测 12T 可用**(已用 1.3T)⇒ **无磁盘风险**。

**F21 seed 公式与分段。实测**:`episode_runner.py:383` `seed = int(extra["base_seed"]) + task.orig_init_state_idx`,
其中 `orig_init_state_idx = episode_idx`(`run_collect.py:164`)。
采集段 `base_seed=0`(17 份 run-plan 全一致,全局最大 `episode_idx` = 396),
评测段 `base_seed=1_000_000`(`run_ws_search2.py:449` 默认,上一轮 run-plan 实测同值)⇒ **不重叠,余量巨大**。
⚠ 但**没有任何代码断言** `base_seed + max_episode_idx < 1_000_000`。
⚠ `(task, episode_idx) → seed` **不是全局单射**(不同任务同 idx 得同 seed);
seed 不重叠性只能定义在 **per-task 粒度**。

**F22 续批机制**:`--episode-lo`(`run_collect.py:411`,格式 `TaskA:20,TaskB:15`)+ `--batch` 递增。
`episode_idx ∈ [lo, lo+n-1]` 闭区间(`:150-153`、`:277-278`)。
⚠ **历史瑕疵**:groot 的 b08/b10、b09/b11 是**同 range 换批号**,uid 集合逐一相同(实测 `set(b08.uids)==set(b10.uids)` True)
⇒ 同时喂给审计器会被 `merge_run_plans`(`verify_collection_artifacts.py:117-118`)`raise ValueError("batches must be disjoint")`。
**新算法禁止同 range 换批号**;真正的 resume 走 `write_run_plan:312` 的 byte-compatible 分支。

**F23 journal 无"每 uid 一条终态"不变量**(实测确认):`driver.py:382` 仅在 `terminal and evidence_safe` 时落账;
重试耗尽走 `scheduler.py:320-325`,`retriable=True` ⇒ `terminal=False` ⇒ **一行都不写**;
per-step flush 失败会主动扣住终态行(`driver.py:373-378`);陈旧 attempt 另写 `accepted:false` 行。
`verify_collection_artifacts.py:12-16` 的 docstring 逐字承认这点。准入判据 `:140-141` =
`accepted is True and success is True and error is None`。

**F24 `--target` 语义是"≥N"不是"恰好 N"**:审计层 `:252` `insufficient = {t: c for ... if c < target}` 进 `ok`(`:254`);
**manifest 层**才截成恰好 N —— `:290` `chosen = sorted(per_task[task_name])[:target]`,每条带
`task_uid / episode_idx / attempt / batch / path / sha256`(`:293-303`,sha 为全字节 `:293`)。
`ok=False` ⇒ `SystemExit(2)` 且**不写 manifest**(`:348-350`)。
⚠ 上一轮**没有产 target=50 的 manifest**,直接用全部 704 条成功建了 `full704` 库。

**F25 ⚠ 建库当前是目录枚举,不是 manifest**:真实脚本是 **`exp/common/build_in_memory_cache_artifact.py`**
(不在 `exp/robocasa365/` 下),`:1124` `resolve_h5_paths(data_dir, episode_list)`;
`:696-697` 默认 `sorted(Path(data_dir).rglob("*.h5"))`。
`docs/data_collection/guide.md:424-426` 声称"建库消费 manifest 而非目录枚举"—— **无实现代码**。
**好消息**:`--episode-list`(`:1408-1417`)已存在,吃**纯文本相对路径清单**,对绝对路径/`..` 逃逸/非 `.h5`/
不存在/重复**逐条 fail-fast**(`:699-724`);`--outcome-filter` 默认 `success`(`:1400-1406`)。
**缺的只有一段桥**:manifest JSON → episode-list,且**没有任何代码校验 manifest 里的 sha256**。

### 1.6 身份缺口(新旧数据静默混用的四个入口)

**F26** `build_run_plan` 的 `params`(`run_collect.py:263-285`)只含
`teacher/layout/style/base_seed/replan_steps/batch/tasks/collect_root`,
**不含任何 env 身份**(无 `obj_instance_split`、无 pin)。哈希 = canonical JSON(`sort_keys` + 紧凑分隔符)→ sha256(`:244-248`)。
唯一闸门在 `write_run_plan:306-311`(不符 `raise SystemExit`);
⚠ **conductor journal 本身完全不校验 plan_hash**(`driver.py:145-152`),绕过 `--role driver` 就没有这道闸。

**F27** artifact 元数据无 env 身份:`in_memory_backend.py:296-312` 只有
`key_builder_type/checkpoint_id/prompt_pool/projection/id_policy/model`;
builder 侧 stamp 清单 `build_in_memory_cache_artifact.py:1239-1248` 同样无 scene/layout/style/pin。
`load_artifact`(`:244-400`)**唯一硬校验是 `vector_dims`**(`:285-289`),注释 `:290-295` 自陈 "Recorded, not enforced"。

**F28** env 缓存键 `self._envs: dict[tuple[str,int,int]]`(`episode_runner.py:282`),
构造于 `:333` `key = (task_name, layout, style)` ⇒ **不含 pin**,同一 worker 换 pin 会静默复用旧 env。

**F29 ⚠ H5 元数据有 allowlist,新键会被静默丢弃**:`data_collector.py:51-54`
`_METADATA_ATTR_ALLOWLIST = ("task_id","init_state_idx","orig_init_state_idx","subset_init_state_idx","split")`;
过滤在 `:72-75`。而 `episode_runner.py:394-400` 实际发的是
`{task_uid, attempt, task_id, orig_init_state_idx, seed}` —— **`task_uid`/`attempt`/`seed` 三个键当前就在被静默丢弃**。
⇒ 新增 `pin_id` / `pin_task_id` / `realized_objects` 不改 allowlist 同样无声消失。

**F30 ✅ binding check 真实存在(不是注释里的愿望)**:
`build_shared_storage`(`config.py:2932-2945`)是 storage 构造的**唯一收口点**,`:2943-2944` 依次调用
`_check_text_ivf_artifact_binding`(`:3000`,`prompt_pool` 为 None 即"必须重建",`:3035-3041`)与
`_check_cp2_projection_binding`(`:2948`),不符抛 **`ConfigValidationError`**(`config.py:58`,继承 `ValueError`)。
运行时机有二:**server 启动期**(`serve_policy.py:597/:942`、`serve_groot_n15.py:227/:387`)与
**yaml 热切换**(`websocket_policy_server.py:795-797`,异常被 `:826-828` 捕获回 error ack,不杀进程)。
另有更贴近本需求的先例 `groot/load_guard.py:134-184` `validate_artifact_identity`。
⚠ **必须挂在 `build_shared_storage` 而非 `load_artifact`**:`BackendPool` 的 fingerprint
(`backend_pool.py:95-123`)= `(backend_type, preload_path, vector_dims, index_type, text_ivf_params)`,
**不含 pin**,每个 fingerprint 只 load 一次,挂 `load_artifact` 会漏检。

**F31 fail-fast 先例可复用**:`episode_runner.py:70`
`REQUIRED_EXTRA_KEYS = ("task_name","layout","style","teacher","base_seed","replan_steps")`,
缺键在 `:366-371` `raise ValueError`。`EpisodeTask.extra`(`conductor/task.py:102`)是
**wire-safe 的扩展点**(`protocol.py:49-54` 用 `dataclasses.asdict` / `EpisodeTask(**d)`,顶层加字段会破坏新旧混跑,
往 `extra` 塞键不会)。当前写入点 `run_collect.py:168-176` 共 7 键。

**F32 ⚠ artifact 错配当前表现为重试循环而非 fail-fast**:`run_ws_search2.py` 无任何 artifact CLI flag、
无 preflight(`:431-478` 只有 `--config-dir`/`--manifest`);pkl 路径烧在 emitter 常量里
(`emit_ws_search2_yamls.py:43-44`、`:50-51`),随 yaml 全文经 `:183-189` `ctl.load_cache_config` 下发。
server 侧抛 `ConfigValidationError` → error ack → `_ensure_bundle`(`:170-192`)把 stage 退回 `SETUP_PENDING` 重试。

**F43 ✅ realized provenance 是可读的(支撑正式集逐集审计)。实测**:
`kitchen.py:870-871` `model, info = EnvUtils.create_obj(self, cfg)` 后紧跟 `cfg["info"] = info`;
隐式容器同样写回(`:902-903`,且 `:901` 已 append 进 `all_obj_cfgs`);auxiliary 亦然(`:969-970`)。
最终 `:975 self.object_cfgs = all_obj_cfgs`。⇒ `_create_objects` 之后,**`env.object_cfgs` 里每个 cfg 的
`info["mjcf_path"]` 就是本集真正落地的实例**(含 `obj_container`),
`get_ep_meta()["object_cfgs"]`(`:1190`)是其 JSON 化副本。**这是「实际用了什么」的唯一权威读点。**
⚠ **但 `gym.make` 返回的是 Gym wrapper,不能直接取属性**。仓内已有先例:
`build_bucket_variants.py:156-160` 明确先 `inner = getattr(env, "unwrapped", env)` 再读 ——
依赖 wrapper 的隐式属性转发在 Gymnasium 版本间不稳定。⇒ D14 必须走同款 wrapper-safe 读法。

**F44 运行时管线的三个改动点(实测行号)**:
worker 命令行在 `run_collect.py:349-361`(`robocasa_spawn_fn` 拼 `python -m exp.robocasa365.worker_entry ...`,
`:362-364` 有 `--max-cached-envs` 的可选参数先例);
`_ensure_env` 在 `episode_runner.py:332-344`(键 `:333`,`gym.make` 调用 `:342`
`env = self._gym_make(task_name, layout, style, **self._adapter.env_kwargs())`)。

**F45 `expected_pin_id` 的落点**:`InMemoryConfig`(`config.py:474-477`)现有
`preload_path` / `index_type` / `text_ivf` 三字段 —— pin 期望值紧邻它约束的那个 artifact,
故新增 `expected_pin_id: Optional[str] = None`,由 `build_shared_storage` 读 `config.backend.in_memory.expected_pin_id`。

**F46 ⚠ 前一稿的配对键写错了**:teacher-only 地板臂只有**一个** cid,不可能与 132 个不同 cid 在
`(cid, task, idx)` 上相等。正确锚点是**所有 cid 共用的 `(task, episode_idx)`**;
`run_ws_search.py:100` 的 run_id 含 cid,但 journal 内每条记录的配对键是 `(task_name, episode_idx)`
(uid 形如 `<run_id>__<Task>:eval:<task_id>:<episode_idx>`)。

### 1.7 评测身份

**F33 132 格权威来源**:`emit_ws_search2_yamls.py:30` **import 复用** round-1 的
`exp/robocasa365/emit_ws_search_yamls.py:86-102` `weight_matrix()`;字段集 `:54`
`FIELDS = ["vision_0","vision_1","vision_2","robot_state"]`。
家族分解(**我方运行 `weight_matrix()` 实测**):`iso 4 + grid 42 + grid3 30 + grid3v 21 + grid4 35 = 132`。
cid 格式 `:81`。硬断言 `emit_ws_search2_yamls.py:163` `assert len(main_index) == 132`。
⚠ round-2 的 132+12 份 yaml、`index.json`、full704 Phase-1 标定 json **都不在仓库里**
(`config/ws_search2/` 只有 `selection_manifest.json`,且只含 `ws2c` 段)。

**F34 上一轮真实取值**(已核实,非默认值推测):scene `(1,1)`、每 task **8** trial、
`base_seed=1_000_000`、`replan_steps=5`、13 任务(`run_ws_search.py:66-82`);
⚠ 加密臂 ws2e 是 **32** trial 不是 8。来源:`logs/robocasa365_ws_search2_text_ivf_plan.log.md:71`(冻结的
`REQUIRED_EXTRA_KEYS` 六键)、`:462`/`:490` 实跑命令、`ws_search2_groot_results.md:16-20`(104 集/cell = 13×8)。

**F35 ✅ teacher-only 地板臂的 cache-off 机制无需新增 cache 代码**:`run_ws_search.py:246` `ctl_factory=lambda _server: _NoOpCtl()`
(注释 "static per-process config: nothing for ctl to do")⇒ **它根本不发 cache 配置**;
其 EpisodeTask 全部 `bundle_id="default"`(继承自 `run_collect.py:167`);
server 侧对 `"default"` 放行(`websocket_policy_server.py:829-853/:858-882`;
GR00T `serve_groot_n15.py:364-366`);`--cid` 只是标签,不校验 yaml(`run_ws_search.py:164` → `:100`);
`episode_runner.py:419-420` 对缺 `__hit_meta__` 不报错。
⇒ **配方**:server 起成不带 `--cache-config` 的 `--concurrent` 形态,driver 用
`run_ws_search.py --teacher <groot_tp|pi05> --server <host:port> --env-config <path> --cid teacher --run-prefix ws2t --tasks PickPlaceCounterToCabinet,PickPlaceCounterToStove,PickPlaceDrawerToCounter,PickPlaceSinkToCounter,PickPlaceToasterToCounter --episodes 50 --layout 1 --style 1 --base-seed 1000000 --replan-steps 5 --pinned-objects <path>`
⚠ `--episodes` 取 **50** 而非其默认 8(`:169`)—— D10 冻结 `idx 0..49`;
⚠ `--tasks` 也必须显式给出上述**恰好 5 项** —— `run_ws_search.py` 与 `run_ws_search2.py` 均默认
`DEFAULT_EVAL_TASKS` 的 13 项,省略会把范围外 8 个任务带入运行并破坏预算/清单完备性(D15);
⚠ **该 driver 必须同样接入 pin 载荷**,否则地板臂跑随机物体,分子分母来自不同环境分布,D10 的 estimand 直接失效。
产物与 132 格按 **`(task, episode_idx)`** 配对(⚠ 见 F46:**不是** `(cid, task, idx)`)。
**排除的三条替代路线**:`gate: always_skip` 会被 GR00T `load_guard.py:48` 的
`_BASE_ALLOWED_GATES = frozenset({"always_search"})` 拒(两 teacher 机制会不一致);
给 `run_ws_search2.py` 加相位需 15-20 行新代码;`groot_rollout_client.py` 的 `--base-seed` 默认 **0**(采集段),
seed 段不同,**不能**当配对地板臂。

**F36 exact-grid 两道独立门**:(a) 运行侧 `run_ws_search.py:103-157` `summarize_journal()` 与 run-plan uid 全集对账,
`:156` `complete = (n_err == 0 and n_missing == 0)`;`run_ws_search2.py:381-422` `finalize()` 逐 cell 写 summary。
(b) 分析侧 `analyze_ws2_vs_ws1.py:44-86` `require_full_matrix()` —— 逐臂检 cid **缺失与多余**(`:58-67`,superset 同样拒)、
检 grid 长度 `== tasks × episodes`(`:68-75`)、检各臂 `(task, idx)` 网格逐项相等(`:76-79`),不满足即 `SystemExit`;
`--allow-partial`(`:421`)才取交集且强制抬头改成 `PARTIAL (NOT A FORMAL RESULT)`(`:121-122`)。

**F37 ⚠ 每个 teacher 各有自己的库与标定,换库必须重标**:`emit_ws_search2_yamls.py:38-54` 的 `TEACHERS` 表是唯一权威
(groot_tp:`cp1_groot_spatial_pool_16` / stem `groot_tp_spatial_pool_16_full704` / knobs `{}`;
pi05:`cp1_spatial_pool_16` / stem `pi05_spatial_pool_16_full704` / knobs `{masked:True, span:True}`)。
`:144` `calib_entry = calib[spec["stem"]]` **按 stem 取该 teacher 那一段**,`:145-149` builder 不符即 `SystemExit`。
⇒ **新的 PickPlace-only 库必须用 `exp/common/calibrate_score_normalizers.py` 重新标定**,不能沿用 full704 的。
⚠ 两 teacher 库规模/渲染分辨率不同(pi05 63,977 entries/29.8GB vs GR00T 50,795/20.5GB)⇒ **绝对 SR 跨 teacher 不可比**。
⚠ **实测补充**:本机全盘 `find` 对 full704 标定文件**零命中**(仓内只有 round-1 的 n5 版
`config/ws_search/calibration_normalizers.json`)⇒ 本轮**无任何标定可继承**,W6 的重标是硬依赖不是可选项
(全库 LOEO,`--max-queries 300`,约 1h/库)。

**F38** prompt 取自 `info["cat"]`(`object_utils.py:713-729`)⇒ 钉死后 5 个任务各塌缩为 1 个 prompt
⇒ 桶数从 111 降到 **5**,`text_ivf` 在本子集上**等价于 `task_key` scoping**。

### 1.8 文档与索引义务

**F39 tracking policy**:`.gitignore:6` `exp/**/data/**` 默认全忽略,`:7` `!exp/**/data/` 只保目录桩;
`docs/experiments/artifact_layout.md:145-162` 规定白名单**必须写成 `.gitignore` 的 `!` 行**,
"**not by hand-adding via `git add -f`**"。⇒ **钉死清单不能放 `data/`,要放 `config/`**(tracked,如
`config/ws_search2/selection_manifest.json`)。文件种类规则见 `:123-132`(runner→exp 根;yaml→`config/`;
运行期 json/jsonl/h5→`data/`;pkl→`data/cache_artifacts/`;png/pdf/分析 md→`analysis/`;
未接入 pytest 的冒烟脚本→exp 根;接入 pytest 的→`tests/<exp>/`)。

**F40 ⚠ WA §4 红线已被触发**:`logs/robocasa365_pnp_pinned_objects_plan.log.md` **未登记进 `logs/README.md`**
(`grep` 零命中)。WA §4 原文:"after every doc creation / modification / move in `docs/` or `logs/`,
the corresponding README MUST be updated **in the same commit**. 'I'll update it later' is not acceptable."
格式范例 = `logs/README.md:87`(同族的 ws_search2 条目),三列 `| File | Status | Description |`,
归入 `### Cache System` 段。

**F41 docs 是双层索引**:`docs/README.md:27` "Each subdirectory has its own `README.md` index"
⇒ 改一份 `docs/<sub>/x.md` 要同时改 `docs/README.md` 的 Section Index 行**和** `docs/<sub>/README.md` 的行。
落点:`docs/data_collection/guide.md:351` `## RoboCasa365 teacher-library collection` 之下、
`### Driver + workers`(`:385`)之后、`### Audit + manifest`(`:406`)之前;
`docs/architecture/cache_system.md` §5.19 的 "Startup binding + guards"(`:1024`)旁边或其后新开 §5.20
(`_check_text_ivf_artifact_binding` 是可照抄的先例)。
**F42 WA §2.1** L3 的额外义务写在表格右列:`Verify + architecture doc update`。

---

## 2. 设计裁定

- **D1 范围**:5 个 PickPlace 任务、13 个 slot。`CoffeeSetupMug` 移出(owner 裁定)。
- **D2 钉死方式**:**实例级**,统一经 `cfg["info"]["mjcf_path"]`(F6 唯一重定基门),
  **不用**绝对路径 `obj_groups`(不重定基,F6),**更不用**类别名(只锁到池子,F7 的实例抽样仍在)。
  隐式 `obj_container` 经 `placement["try_to_place_in_kwargs"]["info"]` 注入(F10)。
  `ToasterToCounter.obj` 钉 **`model_upright.xml`**(F8)。
- **D3 自校验(选型期,补偿 F7 的旁路)**:选型脚本必须**重放随机分支的全部约束**——
  ① 原 cfg 的 `obj_groups` 展开;② `exclude_obj_groups` 扣除;③ 7 个属性开关过滤;
  ④ 按 registry 施加 split 规则(`split_th = max(len-5, ceil(len/2))`,取 `[split_th:]` 为 target);
  ⑤ `max_size` 校验(本轮 13 slot 全无,仍无条件断言);⑥ `rotate_upright` slot 断言 `model_upright.xml` 存在。
  选定实例必须落在 ①–④ 的合法集内,否则脚本 fail-fast。
  **选取规则(Code 期补冻,D3 原只冻结校验未冻结选法)**:候选按 normalized 路径排序后取首,
  但**同一任务内类别不得重复**。理由是实测出来的:纯字典序取首会让 `CounterToCabinet` 的
  `obj` / `distr_counter` / `distr_cab` 撞成同一个 `AluminumFoil004`;即便退一步只要求实例互异,
  三个 slot 仍会落在同一类别(`AluminumFoil004/005/006`)。而 prompt 由 `info["cat"]` 生成(F38),
  指令"把铝箔放进柜子"在含三张铝箔的场景里**指代不唯一** ⇒ teacher 无法判断该抓哪个,
  SR 会因与检索无关的原因塌掉,连带毁掉 F1 的分母。故互异约束下沉到**类别**层;
  跨任务复用同一类别无此歧义,不受限。
  **另一条同期实测补冻:摆放可行性**。`PickPlaceDrawerToCounter` 在真机上
  `RuntimeError: Ran _load_model() 50 times but could not initialize task!` ——
  摆放采样失败时 robocasa 会重试整个 `_load_model`(上限 50,`kitchen.py:647-650`),
  未钉死时每次**重抽一个不同物体**最终能摆下;**钉死后 50 次全是同一个摆不下的物体**,
  于是"重采即可恢复"变成"任务永远起不来"。D3 的六项重放**看不到**这一层
  (`max_size` 在这 13 个 slot 上全为空)。故选取规则再加两条:
  ① 候选按**体积接近中位数**排序(过大摆不下,过小难抓,两头都会以与检索无关的原因压低 teacher SR);
  ② 选完必须**真建环境并在验证 seed 上 reset 通过**才接受,失败则把**体积最大**的 slot 换下一个候选。
- **D4 位姿不锁**(owner 指令,F13-F16)。`ToasterToCounter` 的 toaster 漂移(F15)同样不锁,
  但**必须在结果里标注**它是 5 个任务中唯一目的地会动的。
  ⚠ 因 F7 的 RNG 位移,**钉死前后同 seed 的位姿不可配对**,报告不得做此比较。
- **D5 canonical pin identity(命名与载荷统一)**。**数据字段一律**只用以下三个名字(`pin_identity` 作为字段名不再使用;
  同名的**函数** `_check_pin_identity_binding` 是 D8 的绑定检查器,不是字段):
  - **载荷** `pinned_objects`:`{task_name: {slot_name: "objects/<...>/model.xml"}}`,
    路径一律是 `"objects/"` 起始的**相对路径**(F2 三机 assets_root 不同,绝对路径不可移植)。
  - **全局身份** `pin_id`(canonical JSON + sha256)
    —— 与 `compute_plan_hash`(`run_collect.py:244-248`)同风格。用于 artifact 绑定与 run-plan 身份。
  - **单任务身份** `pin_task_id`(同法哈希 `{task, slots}`)
    —— worker 只收到本任务那一片,用它做内容寻址自检。
  ⚠ **哈希不能凭自身建环境**:真正送达 env 的是 `pinned_objects` 载荷本身,`pin_id` 只是它的指纹。
  ⚠ **两个哈希必须域分隔**(Code 期实测补冻):若都直接哈希 `{task: slot_map}`,
  **单任务表下二者逐字节相同**,worker 校验切片就会顺带满足全局校验,两个身份的独立性失效。
  故各自带常量域前缀(`robocasa365/pin_table/v1` / `robocasa365/pin_task/v1`)。
  清单文件 `exp/robocasa365/config/pnp_pinned_objects.json` 形如
  `{"pin_id": "<sha256>", "pinned_objects": {...}}`,加载时**必须重算并断言** `pin_id` 自洽。
- **D6 采集完成算法**:首批 `N_task = min_episodes_for_target(sr_task, target=50, confidence=0.90)`(F19/F20);
  跑完调 `verify_collection_artifacts --target 50`,读 `insufficient`;非空则续批
  `--episode-lo <上批 hi+1> --batch <前批号+1> --tasks <task>:<缺口经同公式反算>`;
  循环至 `insufficient == {}` 且 `ok == True`。**禁止同 range 换批号**(F22)。
- **D7 建库输入**:`verify_collection_artifacts --target 50 --manifest-out` 产出的 manifest 为**唯一**输入;
  新增一段 manifest→episode-list 的桥并**逐条复核 sha256**(F25),把 manifest 的 `plan_hashes` 与 `pin_id`
  一并 stamp 进 artifact 元数据。
- **D8 强绑定挂点**:在 `config.py` 新增 `_check_pin_identity_binding(storage, config)`,
  在 `build_shared_storage:2943-2944` 之后调用(F30),覆盖启动期与每次热切换;不符抛 `ConfigValidationError`。
  **不挂 `load_artifact`**(F30 的 BackendPool 复用会漏检)。
- **D9 运行时载荷通路(冻结到签名一级)**。新 `--collect-root` / `--run-prefix` / `--data-dir`,外加:

  | 跳 | 位置 | 冻结内容 |
  |---|---|---|
  | ① CLI 加载 | **三个 driver 全部**加 `--pinned-objects <path>`:`run_collect.py:401-423`、`run_ws_search2.py:431-478`、**`run_ws_search.py:162-192`(teacher-only 地板臂的驱动,D10)**;后者同步改 `WsSearchStrategy` 构造(`:208`)、其 run-plan `params`、以及 `:270` 的 `robocasa_spawn_fn` 拼参。⚠ **路径必须在 driver 侧解析成绝对路径再转发**(`resolve_manifest_path`):worker 的 cwd 是外部 RoboCasa checkout,与 `REPO_ROOT` 不同,相对路径在 driver 上打得开、在 worker 里打不开;解析要早于角色分支(agent 角色不加载清单但仍要转发) | driver 读清单 → **重算 `pin_id` 并断言等于文件内声明值**,不等即 `SystemExit` |
  | ② run-plan | `build_run_plan` 的 `params`(`run_collect.py:263-285`)加 `pin_id` | 同参数不同 pin 被 `write_run_plan:306-311` 拒绝 resume(F26) |
  | ③ 派发 | `EpisodeTask.extra`(`run_collect.py:168-176`)加 **`pinned_objects`(本任务 slot map)+ `pin_task_id` + `pin_id`** | 走 `extra` 而非顶层字段,wire-safe(F31);载荷本身随任务下发,worker 无需读文件即可建环境 |
  | ④ worker 双重自检 | `worker_entry.py:57-83` 加 `--pinned-objects <path>`(由 `run_collect.py:349-361` 的 spawn 拼入,照 `--max-cached-envs` 的可选参数先例);`episode_runner.py:70` `REQUIRED_EXTRA_KEYS` 加三键 | (a) 由收到的 `extra["pinned_objects"]` **重算 `pin_task_id`** 并断言相等;(b) 由 worker 侧文件重算**全局 `pin_id`** 并断言等于 `extra["pin_id"]`,且其本任务切片与 `extra["pinned_objects"]` 全等。两者任一不符即 `raise ValueError`(照 `:366-371` 现有 fail-fast 形态)。⚠ **校验必须双向**:worker 有清单而 task 无 pin 键要拒;**task 带任一 pin 键而 worker 未加载清单更要拒** —— 后者是最危险的不对称输入(driver 钉死、worker 漏传 `--pinned-objects`),放行就会在合法身份下建随机物体场景 |
  | ⑤ 建环境 | `default_gym_make`(`episode_runner.py:221-233`)签名改为 `(task_name, layout, style, *, pinned_objects: dict[str,str] \| None = None, **kwargs)`,把它并入 `gym.make(...)` 的 kwargs;`_ensure_env:342` 相应传入 | 经 `gym_wrapper` → `create_env` → `robosuite.make` 抵达 `Kitchen.__init__(pinned_objects=...)`(W1 新增形参) |
  | ⑥ env 缓存键 | `episode_runner.py:333` 改为 `key = (task_name, layout, style, pin_task_id)` | 消除 F28 |
  | ⑦ H5 | `data_collector.py:51-54` allowlist 加 `pin_id` / `pin_task_id` / `realized_objects`(F29) | 否则新键静默丢弃 |
  | ⑧ artifact | builder stamp(`build_in_memory_cache_artifact.py:1239-1248`)加 `pin_id` + manifest `plan_hashes` | 消除 F27 |
  | ⑨ eval config | `InMemoryConfig`(`config.py:474-477`)加 `expected_pin_id`;emitter `emit_ws_search2_yamls.py` 的 `TEACHERS` 表随新 pkl 一并落 `backend.in_memory.expected_pin_id` | D8 的 `_check_pin_identity_binding` 读它与 artifact 元数据双向比对 |
- **D10 estimand(联合干预)**:本轮同时更换 eval 分布与 library,**只能**解释为
  "整套 pin 的联合干预效应"。报告**禁止**把改善归因到"库内实例对齐"单一因素。
  配套补 **teacher-only 地板臂**(F35 的 cache-off 配方;仍须实现 D9 的共享 pin plumbing),在同一批 eval seed 上给出 pinned teacher SR,
  以延续 F1 的"还原率"口径。
  **配对锚点 = `(task, episode_idx)`**(F46 修正):地板臂只跑**一个** cid,其 `(task, idx)` 网格被
  132 个 cid **共用**,不存在 cid 维度的配对。
  **预算**:每 teacher 5 任务 × `episode_idx 0..49` = **250 集**(相对 5,280 集的评测臂约 4.7%);
  其中 `idx 0..7` 与 132 格**逐格严格配对**(评测臂 `--episodes 8`),`idx 8..49` 只用于收紧
  pinned teacher SR 的点估计(40 集的 CI 太宽,不足以当 F1 的分母)。
  `base_seed=1_000_000` ⇒ seed 段 `1_000_000..1_000_049`,仍与采集段(F21)零重叠。
- **D11 库与标定**:PickPlace-only pkl × 2(每 teacher 一份),**各自重标 Phase-1 normalizer**(F37);
  检索作用域按桶/任务隔离 ⇒ 无需混入其他任务(**此为待验断言,W7 用测试证明**)。
- **D14 realized provenance(正式集逐集自证,不靠冒烟)**。读取走**冻结的 wrapper-safe helper**
  `realized_objects_of(env)`(置于 `exp/robocasa365/episode_runner.py`),实现照抄仓内先例
  `build_bucket_variants.py:156-160` 的 `inner = getattr(env, "unwrapped", env)`,
  **在 `env.reset()` 之后调用**(此时 `_create_objects` 已跑完),从 `inner.object_cfgs` 读出
  `{slot_name: normalized_relative_mjcf_path}`(F43,**含 `obj_container`**),
  经 `extra_metadata` 写入 H5 的 `realized_objects` attr(JSON 字符串)。
  `verify_collection_artifacts.py` 在 **admission 之前**逐集校验:
  ① 本任务 slot 集合与清单**全等**(缺一个/多一个都拒);② 每条路径与清单**逐字节全等**;
  ③ 由 realized 值重算的 `pin_task_id` 与 H5 声明的 `pin_task_id` 相等;
  ④ **H5 声明的全局 `pin_id` 与审计器自己加载的可信清单(`--pinned-objects`)重算值相等**
  —— 只比 task slice 不够:slice 正确而全局身份错误的 episode 仍会让 artifact provenance 与 H5 断裂。
  **只有四项全过的 episode 才进 `admitted` 与 manifest**。
  ⚠ 判据是 **realized 值**而非声明值 —— 声明正确但 override 未生效的 episode 必须被拒(S7)。
- **D15 评测任务 roster 强绑定**:权威 roster 是 D1 的**有序、恰好 5 项**
  `PickPlaceCounterToCabinet,PickPlaceCounterToStove,PickPlaceDrawerToCounter,PickPlaceSinkToCounter,PickPlaceToasterToCounter`。
  因 `run_ws_search.py` / `run_ws_search2.py` 的 `--tasks` 默认值均为 13 个 `DEFAULT_EVAL_TASKS`,
  W8 的 cache 臂与 teacher-only 臂命令都必须显式传该字符串;**传了 `--pinned-objects` 即触发硬门**
  `assert_pnp_eval_identity`:断言有序 roster 全等、每任务 trial 数一致且等于该臂冻结值、cell 数等于冻结值、
  以及乘出来的总量等于冻结总量。门作用在**不可变的全量 cell 集**上而非 `--only` 后的子集,
  这样断点续跑仍可行而实验形状不可变。这样主臂预算固定为
  `132 × 5 × 8 = 5,280` 集/teacher,teacher-only 固定为 `5 × 50 = 250` 集/teacher,
  且不会向 pin manifest 中不存在的范围外任务派发。

- **D12 指标**:5 任务族内 macro_sr。**与 13 任务口径不可混用**,新报告独立成文(`exp/robocasa365/analysis/`),
  不得改写 `ws_search2_groot_results.md`。
- **D13 采集口径**:每任务 **50 条成功**(沿用旧口径),双 teacher。

---

## 3. 工作单元

| # | 单元 | 交付物(精确路径) |
|---|---|---|
| **W1** | **robocasa 补丁(可恢复交付)**:补丁本体作为**受版本控制的文件**落 `exp/robocasa365/patches/robocasa_pnp_pinned_objects.patch`(tracked —— 不在 `data/` 下,不受 `.gitignore:6` 影响);内容 = 在 `be22d659` 之上改 ① `Kitchen.__init__`(`kitchen.py:362-412`)加 `pinned_objects=None` 形参 ② `kitchen.py:864` 后插 override,按 slot name 注入 `cfg["info"]["mjcf_path"]`,并对带 `try_to_place_in` 的 slot 注入 `placement["try_to_place_in_kwargs"]["info"]`(F10)。**契约:`pinned_objects=None` 时逐字节保持旧行为**(S2 证明)。⚠ **不清理他人改动**:在各机现有 checkout 上从 `be22d659` 建分支 `pnp-pinned-objects` 并**只提交本补丁显式列出的路径**;weilandserver 的 ` M bench_speed.py`、`?? bench_speed.py.orig`、`?? assets/README.md`(F3)**原样保留**,建分支与 apply 均不触碰(我方补丁只动 `kitchen.py`,无冲突面)。三机丢失后的**重建配方** = `clone robocasa @ be22d659` + `git apply <本仓库内的 patch 文件>` | patch 文件(tracked)+ 部署证据卡:每机记录 **baseline SHA(`be22d659`)/ patched commit SHA / `sha256sum` of patch 文件** 三者,外加 apply 前后各一次 `git status --short` 证明他人文件未被改动 |
| **W2** | **钉死清单选型**:实现 D3 的六项重放校验;为 13 个 slot 各选一个实例 | `exp/robocasa365/config/pnp_pinned_objects.json`(**tracked,放 `config/` 不放 `data/`**,F39)+ `exp/robocasa365/select_pinned_objects.py` |
| **W3** | **pin 载荷与身份串接(D9 九跳)**:三个 driver `run_collect.py` / `run_ws_search.py` / `run_ws_search2.py` 的 `--pinned-objects`、run-plan params、`extra` 三键与 spawn 拼参;`worker_entry.py` 的文件入口与双重自检;`episode_runner.py` 的 `REQUIRED_EXTRA_KEYS`、`default_gym_make` 签名、`_ensure_env` 缓存键及 `realized_objects_of`;`data_collector.py:51-54` allowlist;`config.py` 的 `InMemoryConfig.expected_pin_id`;`emit_ws_search2_yamls.py` 写入期望身份。并补 `base_seed + max_idx < 1_000_000` 断言(F21)及 D15 roster 断言 | 代码 + 单测 |
| **W4** | **采集 + realized provenance**:5 任务 × 50 成功 × 双 teacher,按 D6 算法;每集在 reset 后经 wrapper-safe `realized_objects_of(env)` 读取并把 `realized_objects` 写进 H5(D14/F43);`verify_collection_artifacts.py` 增加 D14 的**四项** admission 前校验,**只有通过的 episode 进 manifest** | h5 + journal + 审计报告(含逐集 realized 对账结果) |
| **W5** | **manifest→建库桥**:manifest JSON → episode-list + sha256 逐条复核;把 `pin_id` 与 `plan_hashes` stamp 进 `build_in_memory_cache_artifact.py:1239-1248` | 代码 + 测试 |
| **W6** | **建库 + 重标定 + 评测身份冻结**:PickPlace-only pkl × 2;`exp/common/calibrate_score_normalizers.py` 重标(F37,**硬依赖**,无标定可继承);随后用重标结果 emit 132 格 yaml 到 **`exp/robocasa365/config/ws_search2_pnp/<teacher>/main/`**(`<teacher>` ∈ {`groot_tp`, `pi05`});并产出**唯一冻结摘要**,repo-relative 路径统一写作
**`exp/robocasa365/config/ws_search2_pnp/index_digest.json`**(全文只此一种写法),schema:
`{"per_teacher": {<teacher>: {"cells": {<cid>: <sha256(yaml_text)>}, "digest": <该 teacher 132 项的 canonical 汇总 sha256>}}, "source_sha256": {"emit_ws_search_yamls.py": <sha>, "emit_ws_search2_yamls.py": <sha>}, "global_digest": <全表汇总 sha256>}`。
⚠ **必须按 teacher 分层**:两个 teacher 共用同一批 132 个 cid,但 yaml 文本因 builder / pkl / normalizer 不同而哈希不同,
扁平的 `{cid: sha}` 会让一边**静默覆盖**另一边,且只数 132 项的门照样放行。**pinned 库命名(Code 期定)**:pkl stem = `<teacher>_spatial_pool_16_<tag>`,`tag` 缺省 `pnp_pinned`,目录缺省 `/data/robocasa365_cache/cache_artifacts_pnp_pinned`,均可由 `--pinned-library-tag` / `--pinned-preload-dir` 覆盖;⚠ **标定 json 的 key 必须等于 pkl stem**(`calibrate_score_normalizers.py:213` 按 stem 建键),否则 emit 直接拒跑 —— 这条把「配置指向 A 库、读的却是 B 库的 normalizer」堵死。⚠ **排期纪律**:`index_digest.json` 含两份 emitter 源码的 `source_sha256`,所以**冻结必须晚于 emitter 代码定稿**;此后任何一次改动 emitter 都会让已冻结的 digest 失效,必须重 emit | pkl(`data/cache_artifacts/`)+ 标定 json(`config/`)+ **2×132 份 yaml 与 `index_digest.json`(均 tracked,`config/` 下不受 `.gitignore:6` 影响)** |
| **W7** | **绑定与隔离**:`_check_pin_identity_binding` + 挂进 `build_shared_storage`;作用域隔离测试(D11);桶数 == 5 断言(F38) | 代码 + 测试 |
| **W8** | **评测**:132 格 × 5 任务 × 8 trial × 双 teacher = 5,280 集/teacher;**外加 teacher-only 地板臂 250 集/teacher**(D10,`idx 0..49`,其中 `0..7` 与 132 格按 `(task, episode_idx)` 严格配对)。两个 driver 都须显式传 D15 的 `--tasks <exact-5-roster>`;生成 run-plan 后、派发前校验 roster 与预算恰好为 5,280 / 250。**开跑前 preflight**(`emit_ws_search2_yamls.verify_index_digest`,在 `resolve_cells` 之前、dispatch 之前调用;校验整棵树的**两个 teacher**而非仅当前 `--teacher`):**逐 teacher** 重算 132 份 yaml 的 SHA 与汇总 digest,并与 `exp/robocasa365/config/ws_search2_pnp/index_digest.json` 的对应 teacher 段全等比对(含 `source_sha256`),不等即拒跑 | journal + summary + `complete=132/132` + preflight 对账记录 |
| **W9** | **文档与索引同步(L3 义务,F40-F42)**:`docs/architecture/cache_system.md` 新增 §5.20(pin artifact identity/binding)、`docs/data_collection/guide.md` 在 `:385` 与 `:406` 之间新增 `### Pinned-object task variants`;**同批**更新 `docs/README.md`、`docs/architecture/README.md`、`docs/data_collection/README.md`、`logs/README.md`(含**补登记本 plan**) | doc diff |
| **W10** | **分析与报告**:族内 macro,含噪声地板与胜者诅咒的**重新陈述**;标注 D4/D10 的两条限制 | `exp/robocasa365/analysis/pnp_pinned_results.md` |
| **W11-opt** | **(不排期,owner 可裁)** matched control:旧 full704 库 + pinned eval,取代表性子集格 | 见 §4 说明 |

---

## 4. 风险

| 风险 | 依据 | 处置 |
|---|---|---|
| 钉死实例泄漏 pretrain / 违反属性约束 / 无视 exclude | F7 实测(74/86 类本应被拒) | D3 六项重放校验,W2 交付脚本 |
| 面包平躺(形态与随机分支不同) | F8 实测 diff | D2 钉 `model_upright.xml`,S1 断言 |
| 跨机路径不通 | F2/F6 实测 | D2 走 `info["mjcf_path"]` 重定基门;D5 用相对路径 |
| 无限死循环 | F9 | 本轮 13 slot 无 `max_size`;D3 仍无条件断言 |
| 三机补丁不等价 | F3 实测 | W1 以 **baseline / patched / patch 文件三个 SHA** 对账;他人脏文件**隔离保留**,apply 前后 `git status --short` 证明未被改动 |
| 传参静默无效 | F11 | W1 后加断言:钉死后每 slot 的 `mjcf_path` 必须等于清单值(S1) |
| 隐式 `obj_container` 漏钉 | F10(每集必建) | W1 覆盖 `try_to_place_in_kwargs` 路径;S1 逐 slot 对账含它 |
| 新键被 allowlist 静默丢弃 | F29 实测(`task_uid`/`attempt`/`seed` 当前就在丢) | W3 同批改 `data_collector.py:51-54` |
| 错库静默载入 | F27/F30 | D8 挂 `build_shared_storage`;**不挂 `load_artifact`** |
| 同参数不同 pin 被当 resume | F26 | D9 把 `pin_id` 进 run-plan `params` |
| 同 worker 换 pin 复用旧 env | F28 | D9 把 `pin_id` 进 env 缓存键 |
| 续批被审计器拒 | F22 实测(b08/b10) | D6 禁止同 range 换批号 |
| 采集 seed 泄漏进评测段 | F21(缺断言) | W3 补 `base_seed + max_idx < 1_000_000` 断言 |
| 沿用 full704 标定 ⇒ 打分尺度错 | F37 | W6 强制重标 |
| 归因错误(联合干预) | reviewer G1-R5 | D10:报告禁止单因素归因;W11-opt 由 owner 裁 |
| **载荷未送达 env**(只传哈希、实际仍随机) | reviewer G1R2-1 | D9 九跳把 `pinned_objects` 载荷本身下发;worker 两端重算校验;S1 逐 slot 对账 |
| **override 静默失效但元数据自洽** | reviewer G1R2-2 | D14 用 **realized 值**做 admission 判据;S7 专门测这种伪造 |
| **补丁不可恢复 / 覆盖他人改动** | reviewer G1R2-3 | patch 文件入仓(tracked);只提交显式路径,weilandserver 三条脏文件原样保留 |
| **`weight_matrix()` 源码后续变动静默改实验 / 双 teacher digest 互相覆盖** | reviewer G1R2-4、G1R3-4 | W6 的 `index_digest.json` 按 **teacher 分层**并含 `source_sha256`,W8 preflight 对账 |
| **评测 driver 默认 13 任务,越过 5-task 范围并少算预算** | `run_ws_search.py:168` / `run_ws_search2.py:445` | D15/W8 显式传 exact-5 roster,run-plan 派发前 fail-fast 对账 |

---

## 5. 门

- **S1 冒烟(升级)**:**5 个任务 × 各 2 个不同 seed**,逐 slot 断言
  ① **normalized relative `mjcf_path` 逐字节恒定**且与清单**全等**(含 `obj_container`);
  ② 位姿确实在变;③ prompt 恒定;④ `ToasterToCounter.obj` 命中 `model_upright.xml`。
- **S2 无 pin 回归**:`pinned_objects=None` 时行为与补丁前逐字节一致。
- **S3 负例**:unknown/missing slot、非 target 实例、原 group/属性不符、超 `max_size`、upright 文件错配
  —— 逐项必须 fail-fast(非静默、非死循环)。
- **S4 身份门**:同 worker 两个 pin_id 不复用 env;run-plan 同参数不同 pin **拒绝 resume**;
  旧/错 pin artifact **拒绝加载**;两个 teacher 的 PickPlace-only artifact 各自**为且仅为 5 桶**。
- **S5 采集门**:`verify_collection_artifacts --target 50` `ok=True` 且 `insufficient=={}`;
  manifest 每条 sha256 与建库输入逐条对账。
- **S6 收官**:5 任务 × 132 格全满,`complete = 132/132`,`n_err=0 / n_missing=0`;
  `analyze_*` **不得**使用 `--allow-partial`。
- **S7 realized provenance 拒收门(D14)**:构造"**H5 元数据声称正确、但实际物体不同**"的样例
  (直接篡改 realized 值 / 或用未打补丁的 env 采一集但写入正确的 `pin_id`),
  审计器**必须拒收**并将其排除出 manifest。此门证明的是 override **真的生效**,而不是我们**声称**它生效。
  另加两类用例:**(a) wrapper 读取**——在外层套一个 Gym wrapper 的 env 上调用 `realized_objects_of(env)`
  仍必须读到 realized 值(证明不依赖 wrapper 的隐式属性转发,F43);
  **(b) task slice 正确但全局 `pin_id` 错误**的 episode 必须被拒(D14-④)。
- **S8 评测身份与配对门**:① **逐 teacher** 校验 `index_digest.json`:该 teacher 段**恰好 132 个 cid**、
  与 `weight_matrix()` 的 cid 全集**无缺失也无多余**、每份 yaml 的 SHA 全等、`source_sha256` 全等;
  两个 teacher **各自独立通过**(不得因共用 cid 集合而互相顶替);
  ② 分析侧断言 teacher-only 臂与每个 cid 在 **`(task, episode_idx)`**(`idx 0..7`)上逐项可配对,
  且 teacher-only 只有一个 cid(F46 的错误不得复发);
  ③ cache 臂与 teacher-only 臂的 run-plan `tasks` 均须与 D15 的有序 5-task roster **全等**,
  总量分别恰好为 5,280 / 250 集/teacher,不得出现范围外任务。

---

## 7. Code 阶段真机证据

### S1 冒烟 —— PASS(timan107,2026-09-05)

补丁基线 `be22d659` + `exp/robocasa365/patches/robocasa_pnp_pinned_objects.patch`
(sha256 `191ee79e4541dd664c8072faf3bb54f04085f4d3e8d72a196188459e29ca7ee6`),
分支 `pnp-pinned-objects`;apply 前后 `git status --short` 均只有
` M robocasa/environments/kitchen/kitchen.py` 与他人原有的 `?? robocasa/models/assets/README.md`
—— **他人改动零触碰**。

`pin_id=4d13ac5effce76c0ec253c9cef7dc2ed25dcb7d9ed4bae596e8b08ca483edcaa`,seeds `[1000000, 1000001]`:

| task | slots | 场景状态摘要(两 seed) | prompt |
|---|---|---|---|
| CounterToCabinet | 3 | `e666cc16…` / `8b14ddc9…` | Pick the turmeric from the counter and place it in the cabinet. |
| CounterToStove | 3(含 `obj_container`) | `29bcd6ca…` / `3b48448d…` | Pick the corn from the plate and place it in the pan. |
| DrawerToCounter | 2 | `44fea406…` / `a91161a7…` | Pick the tongs from the drawer and place it on the counter. |
| SinkToCounter | 3 | `ccd2c7aa…` / `e49079c2…` | Pick the apple from the sink and place it on the plate located on the counter. |
| ToasterToCounter | 2 | `419dcb96…` / `25e5584a…` | Place the toasted item on a plate.(`obj` 命中 `model_upright.xml`) |

四项断言全过、`failures: []`:realized 与清单逐字节全等 / 场景状态每 seed 不同(位姿仍在变)/
prompt 恒定 / upright slot 正确。

### 单元测试(G2 Round 1 后)

`test_pinned_objects.py` 43 项(身份/realized/派发/worker 双向绑定/env 缓存身份/路径解析/D15 门)、
`test_pnp_audit_and_build.py` 40 项(审计器 realized 拒收与建库 manifest 绑定)、
`test_ws2_emitter.py` 32 项(原 15 + pinned 与 digest 17)。
`tests/robocasa365 + tests/cache + tests/collect` 合计 **2013 passed / 22 skipped,零回归**。
⚠ `tests/exp/test_prebuilt_matrix_backend.py` 有 2 个失败,实测在 `in_memory_backend.py` 的 **HEAD 版本**下同样复现,
属既有失败,与本线无关。

### G2 Round 1 期间自查发现、并已一并修掉的六处(不在 reviewer 的 6 条之内)

审计器与建库链上的**静默接受**面,由本轮补测试时反向暴露:

1. ⚠ **manifest 的 `pin_id` 与文件内容之间原本没有任何绑定** —— 逐条 sha256 只证明"字节没变",
   建库脚本从不读 episode 自己的 `pin_id` attr。**手改 manifest 顶层一个字段、一个 episode 都不动,
   就能给库盖上假身份**,而 serve 侧的 `_check_pin_identity_binding` 会照单全收。
   现改为建库时逐集比对 h5 的 `pin_id` attr(缺失或不符即 raise)。
2. `merge_run_plans` 不比较各 batch 的 `params.pin_id` —— 钉的与没钉的能静默 union 成一次审计。
3. run-plan 的 `params.pin_id` 非空却没传 `--pinned-objects` 时,审计器**全程跳过 pin 检查还照出 manifest**。
4. 传了表但**不是这次采集那张**时,原本表现为每一集一条 attr 不符;现改为 CLI 层一次性拒绝并点明。
5. `audit(pin_table=..., pin_id=None)` 从 Python API 直调会让每集以难读的消息失败 ⇒ 入口即拒。
6. `realized_objects` 是合法 JSON 但非 dict(如 `"3"`)会让 `set()` 抛 TypeError **冲出审计器**
   ⇒ 现记为一条 problem,其余 episode 照常判完。

### W1 部署证据卡(2026-09-06,三机全部就位)

patch 文件 `exp/robocasa365/patches/robocasa_pnp_pinned_objects.patch`
sha256 `191ee79e4541dd664c8072faf3bb54f04085f4d3e8d72a196188459e29ca7ee6`(三机 push 后逐台对账一致)。

| 机器 | robocasa 路径 | baseline | 分支 | apply 后 `git status --short` |
|---|---|---|---|---|
| timan107 | `/scratch/zixuans8/…/robocasa365` | `be22d659` | `pnp-pinned-objects` | ` M kitchen.py` + 他人原有 `?? assets/README.md` |
| weilandserver | `/home/weiland/…/robocasa365` | `be22d659` | `pnp-pinned-objects` | ` M kitchen.py` + **他人原有** ` M bench_speed.py`、`?? bench_speed.py.orig`、`?? assets/README.md` |
| timan1 | `/scratch/zixuans8/…/robocasa365` | `be22d659` | `pnp-pinned-objects` | ` M kitchen.py` + 他人原有 `?? assets/README.md` |

三台 `grep -c _pin_path kitchen.py` 均为 2。**他人改动零触碰**(weilandserver 的三条脏文件 apply 前后逐字相同)。
⚠ timan1 的 `/scratch` 是本地盘(`/dev/mapper/timan1_sys-root`),**与 timan107 不共享**,故两台各自部署。
