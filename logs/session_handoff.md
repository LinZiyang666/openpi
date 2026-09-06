# Session Handoff — PickPlace 定物体重做(**G1 APPROVED,尚未开工**)

> 覆盖 2026-09-05 会话。当前唯一在办事项就是这一件,**没有任何实验在跑,没有任何机器被占用**。
> 前一条线(ws_search2 检索权重搜索)已于 2026-08-28 全部完成并 push(`29c0359`),
> 本文档只保留它作为**基线数据**的部分,不复述其过程。

---

## 0. 接手第一步

```bash
cd /home/weiland/projects/openpi
cat logs/robocasa365_pnp_pinned_objects_plan.log.md    # ← 计划全文（G1 已过，Review Log 已按 §3.1 删除）
git status --short | grep -v rit_pareto
```

**当前阶段**:`Understand ✅ → Plan ✅ → G1 ✅ APPROVED → Code 🔄 → G2 ⬚ → Verify ⬚`(L3)
G1 走了 4 轮、15 条 blocking 全闭合。G1 Review Log 全文备份在
`<scratchpad>/plan_with_G1_review_log.md`(81KB),commit 前可取回。

---

## 1. Code 已落的东西

| 单元 | 产出 | 备注 |
|---|---|---|
| W1 | `exp/robocasa365/patches/robocasa_pnp_pinned_objects.patch` | 入仓可重建;已在 **timan107** 的 `pnp-pinned-objects` 分支落地,他人脏文件原样保留 |
| W2 | `select_pinned_objects.py` + `config/pnp_pinned_objects.json` | 真机产表,**13 slot**,每个任务都经真建环境验证 |
| W3 | 三 driver + `worker_entry` + `episode_runner` + allowlist + `InMemoryConfig.expected_pin_id` | 载荷九跳 |
| W5 | `build_in_memory_cache_artifact.py --manifest`(逐条 sha256)+ artifact 打 `pin_id` | |
| W7 | `config.py::_check_pin_identity_binding` 挂 `build_shared_storage` | |
| W9 | `cache_system.md` §5.20 + `guide.md` 新节 + 四份索引 | L3 义务 |
| 测试 | `tests/robocasa365/test_pinned_objects.py`(22 项) | 全量 1934 passed,零回归 |

**未做(按 plan 要在 G2 之后)**:W4 采集 / W6 建库+重标定 / W8 评测 / W10 报告。
**补丁只部署了 timan107**;weilandserver 与 timan1 尚未 apply。

---

## 2. Code 期实测暴露、并已补冻进 plan 的四件事

1. **选取规则不能只按字典序** —— 会让 `CounterToCabinet` 的目标物与两个干扰物撞成同一个网格;
   即便只要求实例互异,三者仍同类别,而 prompt 由 `info["cat"]` 生成 ⇒ 指令指代不唯一。
   互异必须下沉到**类别**层。
2. **钉死会把"摆不下就重采"变成死局** —— `DrawerToCounter` 真机报
   `Ran _load_model() 50 times but could not initialize task!`:50 次重试全是同一个物体。
   故候选按**体积接近中位数**排序,且选完必须**真建环境 reset 通过**才接受。
3. **两个身份哈希必须域分隔** —— 单任务表下 `pin_id` 与 `pin_task_id` 逐字节相同。
4. **路径形状要配合 robocasa 的重定基门** —— 它是 `split("/objects/")`,需要前导斜杠;
   表里存 `objects/...`,由补丁的 `_pin_path` 补斜杠。

---

## 7a0. G1 收口修正

1. **teacher-only 的 pin 入口漏了** —— 地板臂由 `run_ws_search.py` 驱动,而我只给另外两个 driver
   开了 `--pinned-objects`。地板臂跑随机物体 ⇒ 分子分母环境分布不同 ⇒ estimand 失效。
2. **`env.object_cfgs` 不能直接取** —— `gym.make` 返回 wrapper,须照 `build_bucket_variants.py:156-160`
   先 `getattr(env, "unwrapped", env)`。另外审计要连**全局 `pin_id`** 一起比,不能只比 task slice。
3. **正文自相矛盾** —— F3 与风险表还留着「先清理 weilandserver 脏文件」,与 W1 的保留策略打架。
4. **digest 会让双 teacher 互相覆盖** —— 两 teacher 共用 132 个 cid 但 yaml 哈希不同,
   扁平 `{cid: sha}` 必须改成按 teacher 分层。
5. **两个评测 driver 默认会跑 13 个任务** —— cache 臂与 teacher-only 臂都必须显式传 plan D15
   的有序 exact-5 roster,并在派发前对 run-plan 的任务集与 5,280/250 预算 fail-fast 对账。

## 7a. Round 2 抓到的两个**致命**缺口(接手者务必先看)

1. **只传哈希不传载荷** —— 前两稿的身份链只搬 `pin_id`,**slot map 从未抵达 `Kitchen`**,
   即使身份记录做全,跑的仍是随机物体。现已冻结九跳载荷通路(plan D9),`extra` 携带本任务 slot map,
   `default_gym_make` 加 `pinned_objects` 形参,driver/worker 两端各自重算哈希比对。
2. **声明 ≠ 生效** —— 期望值 + 文件 sha + 一次冒烟证明不了后续几百集真用了那个实例。
   现已冻结 realized provenance(plan D14):每集 reset 后经 wrapper-safe `realized_objects_of(env)` 读实际落地的 mjcf(`kitchen.py:870-871`
   写回,隐式容器 `:902-903` 同),写进 H5,审计器在 admission 前用**realized 值**判,不过就不进 manifest。

另外两条:补丁改为**入仓的 patch 文件**(可从 `be22d659` + patch 重建,不依赖任何机器的本地 checkout),
且**撤回了「先清理 weilandserver 脏文件」** —— 那三条是别人的改动,原样保留不碰。
teacher-only 的配对键我原先写错成 `(cid, task, idx)`,正确是 **`(task, episode_idx)`**,预算 250 集/teacher。

## 7b. G1 Round 2 之后新增的硬事实(细节见 plan §1)

- 最小改动点 = `kitchen.py:864`(全仓唯一 `_get_obj_cfgs()` 调用点),不必逐个改硬编码 slot
- 精确 XML 分支还跳过 **7 个属性过滤**(`CounterToStove.obj` 74/102 类本应被拒)与 **RNG 消耗** ⇒ 钉死前后同 seed 位姿**不可配对**
- `rotate_upright` 只命中 `ToasterToCounter.obj`;必须钉 `model_upright.xml`,否则面包平躺
- binding check **真实存在**(`config.py:2943-2944`),挂点必须是 `build_shared_storage` 而非 `load_artifact`
- `data_collector.py:51-54` 的 allowlist 当前就在静默丢 `task_uid/attempt/seed`
- 建库脚本真实路径 = `exp/common/build_in_memory_cache_artifact.py`,默认目录枚举;`--episode-list` 是现成的桥
- teacher-only 地板臂的 **cache-off 机制无需新增 cache 代码**(`run_ws_search.py` 的 `_NoOpCtl` + `bundle_id="default"`),但须实现共享 pin plumbing
- 换库必须**重标 Phase-1 normalizer**(`exp/common/calibrate_score_normalizers.py`)
- 本 plan 已补登记进 `logs/README.md`(WA §4 红线原本已触发)

## 8. 下一步

G1 已通过,按 plan §3 走:**W1**(robocasa 补丁,覆盖所有硬编码 slot + 隐式 `obj_container`)
→ **W2**(实例选型 + 校验 target split 与 max_size)→ **S1 冒烟**
(断言 `mjcf_path` 逐字节恒定、位姿在变、prompt 恒定)→ 拿到真机证据再往下。

**进入 Code 后仍须保持 plan 的 fail-fast 门与 owner 冻结范围。**
