# 第五轮（opus）报告：给"卡住保护"的大模型调用加两道闸门，降推理占比

> 本文件由协调者代存：opus 子代理的会话不允许写报告文件，全文由其交接消息转录，内容未改动。

## 给 owner 的白话版

**问的是什么**：目前最好的两套组合（"只保留卡住保护 + 半强度纠偏"，π0.5 另加"落后太多就交给大模型"）成功率很高（π0.5 .92、GR00T .89），但推理占比 .162 / .183，比纯缓存（约 .075）高出一倍多。多出来的几乎全是"卡住保护"触发的大模型调用。这一轮查两件事：哪些调用是白花的？能不能加不看任务的规则，砍掉 ≥25% 推理占比而成功率不掉？

**发现**（只用初始状态 0–19 定规则，20–29 只做描述和交叉核对）：
1. **"按节奏走着的卡顿"叫大模型是白花的。** 所谓卡住，大多数是检索在两条示范之间来回跳，看起来进度没动，机器人其实在正常动。机器人比示范节奏落后不超过 2 步时，纯缓存自己下一眼就恢复的概率（约 85–90%）和叫了大模型一样；落后 3 步以上，叫大模型才明显有用。π0.5 组合里 72% 的卡住调用、GR00T 里 43% 属于这种。
2. **"死循环"调用：一局里叫到 20 次以上的，几乎全是输局。** 这些局的缓存已经把示范放到了最后一帧，但任务没完成。之后检索进度再也不动，保护每隔一步就叫一次大模型（π0.5 则整段交给大模型），一直叫到时间耗尽也救不回来。8 局 / 11 局失败吃掉了全部调用的 37% / 44%。
3. **剩下的失败是什么**：π0.5 组合 8 局失败里有 7 局、GR00T 11 局里有 10 局，纯大模型自己都能做成。类型几乎一样：缓存"把剧本演完了但活没干完"（大约第 45 步就演到示范结尾），之后大模型接手也救不回来。一半是边缘局（同配置另一次跑就成功了），另一半两次都失败。要提成功率，得更早发现"剧本演完但没成功"，那属于接管规则那条线，不在本轮范围。

**两道闸门**（都不看任务编号，阈值只在 0–19 上定）：
- **闸门一"按节奏不叫"**：卡住保护要叫大模型时，如果机器人落后示范不超过 2 步，就不叫，继续让缓存走。卡顿若没自己恢复，下一眼落后更多了再叫。
- **闸门二"每局最多 20 次"**：一局里大模型调用满 20 次后，剩下的时间全交给缓存，不再叫。

**预测**（初始状态 20–29，发布前已写好并带时间戳）：
- **GR00T**：推理占比 .183 → .136（**降 26%**），成功率预计基本不变（模拟 +0.2 个百分点，交叉核对 +1.3）。达标。
- **π0.5**：推理占比 .162 → .126（**降 22%**），成功率预计降 2–3 个百分点。这在 100 局的噪声以内，但方向是降。
- **π0.5 为什么没达到 25%**：它的"落后太多就交给大模型"那部分，救回来的局本来就需要长时间接管，次数上限会误伤；另外 π0.5 按节奏的调用有一点"顺手把机器人摆正"的后续好处，砍掉会损失一点。要硬凑 25%（每局上限改 15），预计会掉 2–5 个百分点，所以没有发布那一档。

**已准备好、未启动**：新目录 `r09_opus_r5`，每个模型三组：原组合（同批对照）、加闸门一、两道闸门都加，各 100 局（初始状态 20–29）。CPU 自测全部通过：闸门全关时与原组合的 48 个决策逐个一致；强制触发时闸门按预期拦截。

## Technical summary
- **Lane:** IR reduction of the leading r3c stacks via gates on guard calls. Code: `round5/methods.py` (`GatedNpGraspEsc3`, `GatedNpGraspStackGroot3`), subclasses of fable's frozen r3c classes.
- **Q1:** no-progress calls at pace lag ≤ 2 do not end stalls more often than the cache alone (pooled fit recovery difference −2…+3 pp for lag −3…2; +8…+17 pp from lag 3). Calls after about 20 per episode are in lost episodes (fit: 13/13 π0.5 and 50/51 GR00T episodes with more than 20 calls failed). Failed episodes consume 37% / 44% of the stacks' calls. Caveat: π0.5 on-pace calls slightly reduce later off-pace stalls (51.3% vs 45.8%).
- **Q2:** gate P (L0 = 2) and gate C (C = 20), thresholds by pre-stated rules on inits 0–19. Predicted IR: π0.5 .162 → .135 → .126 (−22%); GR00T .183 → .146 → .136 (−26%). Predicted SR: GR00T about 0; π0.5 −0.6…−2.6 pp for gate P, a further −0.8…−1.5 pp for gate C.
- **Q3:** residual failures are "script exhaustion": the demo is played to its end (7/8 π0.5, 8/11 GR00T), retrieved progress freezes, the policy is in control at the end, and pure policy solves 7/8 and 10/11 of those pairs.
- **Arms:** `/home/weiland/trace_runs/os_closed_loop/r09_opus_r5`: `r9o5_{pi05,groot}_l10_50_{stack,stack_P2,stack_P2C20}`.
