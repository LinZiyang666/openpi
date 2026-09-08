# PickPlace 定物体重做 —— 结果

> GR00T 臂（`groot_tp`），2026-09-06。cache 臂 `ws2` 132 格 × 5 任务 × 8 trial = 5,280 集，
> teacher 基线臂 `ws2t` 5 任务 × 50 trial = 250 集。两臂同一份钉死清单
> （`pin_id=4d13ac5e…`）、同一场景 `(layout,style)=(1,1)`、同一评测 seed 段
> `base_seed=1,000,000`，配对锚点 `(task, episode_idx)`。
> 完备性：cache 臂 `DONE complete=132/132`、`n_err=0`；地板臂 `n_err=0 n_missing=0`。
> **pi0.5 侧未跑**（owner 2026-09-06 裁定 GR00T 做完即停）。

---

## 1. 主结果

| 任务 | teacher 基线 | cache（132 格合计） | 落差 |
|---|---|---|---|
| PickPlaceSinkToCounter | 94.0%  47/50 | 4.26%  45/1056 | 22× |
| PickPlaceCounterToStove | 92.0%  46/50 | **0.00%**  0/1056 | — |
| PickPlaceToasterToCounter | 72.0%  36/50 | 5.59%  59/1056 | 13× |
| PickPlaceDrawerToCounter | 66.0%  33/50 | 12.88%  136/1056 | 5× |
| PickPlaceCounterToCabinet | 52.0%  26/50 | **0.19%**  2/1056 | 274× |
| **macro** | **75.2%** | **4.58%** | **16×** |

teacher 在评测段的 macro SR 是 **0.752**，与采集段实测的 0.650 同量级 ⇒ **落差不是因为评测段场景更难**。

---

## 2. ⚠ 本轮最重要的结论：132 格权重搜索没有信号

噪声地板与胜者诅咒**按本臂重新估**（不继承旧轮的 3.0pp / 5pp）。蒙特卡洛 4,000 次，
在"所有格子真实 SR 相同、等于实测均值 p=0.0458"的零假设下，模拟 132 格 × 5 任务 × 8 trial：

| 量 | 值 |
|---|---|
| 实测跨格 macro SR：mean / sd | 0.0458 / **0.0344** |
| 实测 min / median / max | 0.0000 / 0.0500 / **0.1250** |
| 实测最优格 | `grid_vision_2@75_robot_state@25` = 0.1250 |
| **纯噪声下 132 格最大值的期望** | **0.1502**（p95 = 0.2000） |
| 胜者诅咒（期望虚高） | **10.43 pp** |
| 噪声地板（单格 1 sd） | **3.44 pp** |

**实测最优格 0.1250 低于纯噪声该产生的 0.1502。** 跨格标准差 0.0344 也与二项噪声的预测
一致。⇒ **132 个权重配置之间的差异完全可由采样噪声解释，本轮搜索未产生任何可用的权重信号。**

⚠ 因此**不得**报告"最佳权重配置"。`grid_vision_2@75_robot_state@25` 只是 132 次抽样的最大值，
不是一个被证实更好的配置；在旧口径下它会被误读成"vision_2 主导"的证据。

---

## 3. 钉死本身是成功的（这不是失败的原因）

三层证据：

1. **覆盖面**：逐任务比对 robocasa 的 `_get_obj_cfgs` 槽位全集，13 个槽位（主物体 + 干扰物 +
   容器 + `CounterToStove` 由 `try_to_place_in` 隐式生成的 `obj_container`）**零遗漏**。
2. **真机冒烟**：5 任务 × 3 seed，`failures: []`，每个槽位的实际 mjcf 路径与清单逐字节一致，
   Toaster 的 `rotate_upright` 槽位正确落在 `model_upright.xml`。
3. **实跑生效**：5,280 集的 per-step 证据里 prompt **恰好 5 种**、一任务一种、零变异
   （prompt 取自物体类别，恒定即证明钉死持续生效）。

⚠ 钉死的是**物体身份**，**位姿仍随 seed 变化**（冒烟中 scene digest 逐 seed 不同，这是设计要求）。

---

## 4. 与 round-2（非钉死）的对照：钉死没有救回 PickPlace 族

| 任务 | round-2 非钉死 | 本轮钉死 | 变化 |
|---|---|---|---|
| ToasterToCounter | 0.228 | 0.056 | **↓↓** |
| DrawerToCounter | 0.066 | 0.129 | ↑ |
| CounterToStove | 0.031 | 0.000 | ↓ |
| SinkToCounter | 0.006 | 0.043 | ↑ |
| CounterToCabinet | 0.000 | 0.002 | ≈ |
| **五任务均值** | **0.066** | **0.046** | **↓** |

两升两降一平，**整体不升反降**。

这直接检验了 `robocasa365_seed_anatomy.md` 的因果结论 —— 那份报告把 PickPlace 族的死因归为
"物体类别每集重抽 + n=5 库只覆盖 19% 类别"，并把当时类别恒定的 ToasterToCounter（0.228）
当作支持该说法的反面对照。本轮把类别彻底钉死、并用每任务约 3,500 条的 PickPlace 专用库重做，
**没有救回来**，而且当年那个"对照组" ToasterToCounter 反而从 0.228 掉到 0.056。

⇒ **"类别重抽"至多是次要因素，不是 PickPlace 族的主因。**

---

## 5. 一个尚未定量的候选解释

渲染首帧对照（建库 seed 0 段 vs 评测 seed 1,000,000 段，同一任务、三路相机）显示：
表现最好的 DrawerToCounter 两段是**同一处工位**（同一台咖啡机、同一个水槽、同一个抽屉），
而归零的 CounterToCabinet 两段是厨房里**完全不同的两处**。

机制上说得通：任务绑定的是"**一个** CABINET 类型的固定装置"
（`register_fixture_ref("cab", dict(id=FixtureType.CABINET))`），layout 固定的是厨房本身，
具体绑到哪一个装置由该集的采样决定；而建库用 seed 0 段、评测用 seed 100 万段，两段不相交。

⚠ **这条尚未定量**。已写好逐帧探测（采集侧取 manifest 里真正进库的 50 个索引、评测侧取 8 个，
读固定装置名字与相机世界位姿），但为把 GPU 让给主跑而中止，只跑完 2/5 个任务。
**报告不据此下结论**，只记为最可能的后续方向。

---

## 6. 可以主张与不可以主张

**可以主张**
- teacher 在评测段 macro SR **0.752**，cache 臂 **0.0458**，**16 倍落差**，两臂逐集配对、完备性无缺口。
- 把 5 个 PickPlace 任务的全部 13 个物体槽位（含干扰物与隐式容器）钉死到单个 mesh 实例，
  **没有**把该族的纯 cache 检索救回来；相对 round-2 非钉死基线整体略降。
- 本轮 132 格权重搜索**无信号**：最优格低于纯噪声的期望最大值。

**不可以主张**
- ⚠ 不得报告"最佳权重配置"或据最优格谈字段重要性（§2）。
- ⚠ 不得跨 teacher 比较绝对 SR：两个 teacher 的库规模与渲染分辨率不同（本轮 pi0.5 未跑）。
- ⚠ 不得把结果归因于单一因素：钉死是**联合干预**（身份钉死 + PickPlace 专用库 + 重标定），
  本报告只能说这个组合没有奏效。
- ⚠ 不得沿用旧轮的噪声地板 3.0pp / 胜者诅咒 5pp —— 本臂的重估值是 **3.44 pp / 10.43 pp**。

---

## 7. 产物

| 内容 | 位置 |
|---|---|
| cache 臂（132 summary + 132 journal + per_step + run_plan） | `exp/robocasa365/data/ws_search2/groot_tp/` |
| teacher 基线臂 | `exp/robocasa365/data/ws_search/groot_tp/{journal,summary,run_plan}_ws2t-teacher__l1s1_groot_tp.*` |
| 库 | `/data/robocasa365_cache/cache_artifacts_pnp_pinned/groot_tp_spatial_pool_16_pnp_pinned.pkl`（17,499 条） |
| 钉死清单 | `exp/robocasa365/config/pnp_pinned_objects.json`（`pin_id=4d13ac5e…`） |
| 132 格 yaml + 冻结 digest | `exp/robocasa365/config/ws_search2_pnp/`（`global_digest=bc66962d…`） |
| 首帧对照 | 见 §5（探测未跑完，不入结论） |
