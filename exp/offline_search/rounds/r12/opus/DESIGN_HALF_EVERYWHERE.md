# R12 opus: "look half everywhere", a five-layer system whose cache looks with one camera per task (design, 2026-10-04 ~02:00 CDT)

（子代理交回原文，由协调者代存。）

## 给 owner 的白话摘要

**做法：每个任务事先定好用哪路相机**
- 选"只看手腕"的任务：
  - 每次看图只编码手腕相机，检索只用手腕画面和机器人状态。
  - 检索的距离沿用现在两路的距离，只是去掉第三人称那一块，不重新训练。
- 选"两路都看"的任务：和现系统完全一样，一处不改。
- 第三人称单路不再考虑：B 池快筛里它五个格子都比两路差。

**每个任务怎么选相机**
- **数据**：只用 B 池开发集。测试集 A 不参与任何选择。
  - 今晚要跑的"选相机"数据：只开缓存，两路和手腕各跑一遍。每任务约 27–32 局（空间 200 集约 15–19 局）。
  - 再加上已有快筛的每任务 10 局。
- **难点**：每个任务只有约 40 局，单个任务的差值噪声约 ±10 个点。直接比，会被噪声骗。
- **办法**：
  - 先算每个任务"手腕减两路"的成功率差。
  - 再按噪声大小，把它往同一格子的平均值收缩。
  - 收缩后，只要不比两路差太多就选手腕，否则选两路。打平选两路。
- **"差太多"的界线**：等于手腕省下的开销用旋钮能买回的成功率。各格子在 0.5–2 个点之间，全部由 B 池数据算出，已冻结。
- **按格子整体选还是按任务选**：由数据自己决定。
  - 各任务差别不大的格子，所有任务会一起选。
  - 只有某个任务差别明显、证据也够时，它才单独选另一路。
- **示范库里的辅助信号**：示范库里有一个"手腕检索比两路差多少"的离线信号，和快筛的任务差值弱相关。只有它在选相机数据上统计显著时，才用来帮忙收缩。

**用今天已有的 B 数据（每任务 10 局）先试选一次**（只作参考，最终以今晚的数据为准）
- π0.5 空间 200 集、GR00T 空间 50 集：全部任务选手腕。
- π0.5 长任务 50 集：3 个任务选手腕。
- GR00T 长任务 50 集：6 个任务选手腕。这个格子各任务差别很大：有的任务手腕差 40–60 个点，有的反而好 20–30 个点。
- π0.5 空间 50 集：全部选两路。快筛里手腕平均差 4.6 个点。

**其他几层的调整**
- **守卫**
  - 手腕任务：用之前那个不需要第三人称画面的新守卫。旧守卫建立在两路检索上，换成手腕检索会漏叫。
  - 两路任务：保留原守卫。B 池小跑里，新守卫在长任务上第一次调用偏晚，所以不往两路任务上推。
- **修正器**
  - 手腕任务：换成一个专门训练的修正器。它只在示范库内用"留出一局"的方式训练，输入是手腕检索的结果。
  - 两路任务：保持原修正器。
- **旋钮**
  - 仍是定期调用。
  - 只把"看一次图的价格"按任务换成实际价格，手腕任务半价。
  - 间隔用原来那套示范库回放重新算。全部选两路时，算出的间隔与原来逐位相同（已验证）。
- **按任务分治**：每个任务用哪路相机，就是新增的按任务参数。旋钮仍是每格子一个数，不含按任务的参数。

**开销**
- 手腕任务上，不调用大模型的看图点价格减半。
- 一个格子全部选手腕时，每个决策点最多省 .026（GR00T 长任务）到 .037（π0.5 空间 200 集）。
- 作为对比，上一版"新守卫加部分只看手腕"的方案，视觉部分只省到 .008–.020。
- 实际能省多少，取决于有几个任务选了手腕。

**冻结预测**
- 只用 B 数据，并且把"今晚的数据会怎么选"这一层不确定性也算了进去。
- 测试集 A，五格合并：开销 −.022，80% 区间 [−.033, −.011]；成功率 +1.5 个点，80% 区间 [−0.3, +3.3]。
- 与上一版"新守卫加部分只看手腕"相比：合并开销持平（+.002）；成功率高约 2 个点，主要来自 GR00T 长任务，上一版在那里掉了 6 个点；π0.5 空间 50 集上可能省得比上一版少。
- 今晚选完相机后，按冻结的公式机械地更新逐格预测，不再加人为判断。

**风险**
1. 每任务约 40 局，仍可能选错。收缩能减少选错，但不能消除。
2. 从"只开缓存"推到"全系统"要打折，因为守卫会救回一部分失败。我用 B 数据估了这个折扣：大约保留 65%，区间按 35%–100% 算。
3. B 池起始状态已经全部用于选相机，所以 B 快筛只能和选相机用同一批起始状态。快筛的成功率因此可能略偏乐观，只作"是否崩溃、机制是否生效"的检查。真正的检验是测试集 A。
4. GR00T 单独编码一路相机时，与两路一起编码有约 2% 的数值偏差。选相机数据已经包含这个影响。

**哪些决定在看过测试集 A 后可能受了影响（如实说明）**
1. "按任务选相机"这个方向本身：起因是测试集 A 上 GR00T 两类任务方向相反。
2. 两路任务不用上一版的"部分只看手腕"：GR00T 长任务在测试集 A 上掉了 6 个点，加强了这个判断。B 池小跑只有不显著的同向迹象。
3. 用原距离、不重训：B 快筛合并后也是原距离略好；测试集 A 的结果与此一致。

其余部分全部只依据 B 数据或示范库：选相机的规则和界线、守卫分配、修正器、旋钮重算、预测。

**需要你裁定**
1. 两路任务保持原系统，还是用上一版方案：我的默认是保持原系统，更稳。代价是在"多数任务选两路"的格子里，比上一版少省约 .015–.025。
2. 实验顺序：今晚跑选相机数据；按冻结规则定相机；sol 实现，可与跑数据并行；B 池快筛 6–8 组，每组 100 局；测试集 A 最多 6 组，每组 500 局。没有选到手腕任务的格子不上测试集 A。

---

CPU only; nothing launched.

**Evidence base.** The B library (LOEO and held-out B episodes); B-dev closed-loop arms (the pure-cache camera screen, the R12 pilots and the R11 dev arms). Test-A results are cited as context only; §0.3 lists every decision that having seen A may have influenced.

**Tools and outputs.** Scripts in `rounds/r12/opus/tools/`: `design_bscreen`, `design_libsignal`, `design_corr`, `design_pertask_ledger`, `design_knob`, `camera_choice`, `design_predict`, `design_score_pilot`, `design_ledgers` (all `.py`). Outputs in `rounds/r12/opus/out/design/`.

## 0. Status, data and discipline

### 0.1 Evidence used

**Pure-cache camera ablation, B screen** (100 eps per arm, K=10 per task; roots `rounds/r12/ablate/pilots/`; per-task data `out/design/bscreen_pertask.json`):

| cell | both | wrist_sub | wrist_refit | agent_sub | agent_refit |
|---|---|---|---|---|---|
| pi05 L10-50 | .63 | .57 | .63 | .52 | .51 |
| pi05 Sp-50 | .87 | .81 | .82 | .76 | .74 |
| pi05 Sp-200 | .96 | .97 | .95 | .91 | .88 |
| GR00T Sp-50 | .79 | .82 | .72 | .73 | .76 |
| GR00T L10-50 | .54 | .49 | .46 | .43 | .38 |
| mean | .758 | .732 | .716 | .670 | .654 |

Per-task heterogeneity on the B screen (DerSimonian–Laird): GR00T L10-50 Q(9) 24.5 (p ≈ .004), τ̂ .25; pi05 L10-50 Q 13.6, τ̂ .135; Spatial cells Q ≤ 8.9, τ̂ 0.

**Pure-cache ablation, test A** (context only; 500 eps paired; roots `r12_ablate_test_<cell>`; re-scored 01:50 CDT):

| cell | both | wrist_sub | wrist_refit |
|---|---|---|---|
| GR00T L10-50 | .608 | .498 (−11.0 ± 4.8) | .486 |
| GR00T Sp-50 | .794 | .870 (+7.6 ± 4.3) | .854 |
| pi05 L10-50 | .636 | .606 (−3.0 ± 4.4) | .574 |
| pi05 Sp-50 | .832 | .836 (+0.4 ± 3.8) | .836 (+0.4) |
| pi05 Sp-200 | .966 | .958 (−0.8 ± 2.3) | .956 (−1.0 ± 2.2) |

Pure-cache vision cost per decision: two cameras .075–.077, one camera .037–.039.

**Full-system R12 test A** (context only; vs same-root off): M2 pi05 L10-50 −1.2 / −.0309; pi05 Sp-50 +3.8 / −.0341; pi05 Sp-200 +1.6 / −.0279; GR00T Sp-50 +0.2 / −.0236; GR00T L10-50 −6.2 / −.0048; pooled M2 −0.4 / −.0242, H2 −0.3 / −.0066, M1 −1.2 / −.0096 (ΔSR pp / ΔIR).

**B-dev pilots**: deployed replicates and M1/H2/M2 arms via `tools/design_score_pilot.py`; per-task ledger of the deployed system (decision share ω_t, non-call looks per decision u_t) in `out/design/pertask_ledger.json`.

**Library-only per-task signal** (`design_libsignal.py`): 25 held-out B episodes per task (not in the served subset) re-served by the frozen ablation artifacts; Δe10 = e10(wrist_sub) − e10(both). Cell means +.009 to +.023 (offline the wrist is worse everywhere). Within cells Δe10 correlates with the B-screen per-task Δ at r = −.25 over 50 tasks (one-sided permutation p = .054); strongest on GR00T L10 (r = −.55): tasks 6, 7, 8 are the only ones where the wrist is better offline, three of the four tasks where the wrist won on the screen.

### 0.2 Camera-choice data after tonight
- Camdev (`rounds/r12/ablate/camdev/HANDBACK.md`): both + wrist_sub, disjoint from the K=10 screen and served subsets; 27–32 inits per task (Sp-200 15–19); K=40 unavailable (B complement exhausted).
- Pooled with the screen: 37–42 inits per task per arm (25–29 on Sp-200). Per-task SE of a wrist−both difference ≈ .10 at SR ≈ .5, ≈ .03 at the ceiling.
- GR00T camdev on H100, as the screen was.

### 0.3 Decisions possibly influenced by having seen test A

| decision | B / library basis | A influence |
|---|---|---|
| per-task (not global) camera | B screen GR00T L10 heterogeneity, Q p = .004 | the direction flip on A started it |
| sub metric, no refit | B pooled sub .732 vs refit .716; no extra fit | A agrees (seen) |
| no third-person camera | B: agent_sub worse than both and wrist_sub in 5/5 cells (pooled −8.8 pp vs both) | none needed |
| both-mode = deployed, not the M2 cascade | B pilot M2 GR00T L10 −3.7 (n.s., host-confounded) | A GR00T L10 M1/M2 −6 strengthened it |
| both-mode guard = deployed | B pilot: H2 first call late on L10 (+11.5) | none |
| wrist-mode guard = H2 | GUARD.md dev replays + B pilot | A M2 context |
| tolerance C_cell, α, knob | library + B-dev only | none |
| predictions | B-only primary | A-informed variant shown separately (§9.4) |

No per-task test-A breakdown was computed for this design.

## 1. System overview (suggested class `R12HalfEverywhere`)

Each cell gets a frozen per-task table `camera[t] ∈ {wrist, both}`.

| layer | wrist task | both task |
|---|---|---|
| 1 cache | wrist camera only at every non-call look, step 0 included; retrieval on [wrist PCA64, state8] with the restricted principal block of the deployed per-task joint Mahalanobis ("sub"); AWM regimes, kernel, blind tail unchanged | deployed (R10 two-camera) |
| 2 guard | H2 (agentview-free; frozen coefficients, per-cell θ) | deployed only-no-progress rule on two-camera top-1 |
| 3 corrector | wrist-path LOEO head on sub synthesis + sub distance scale (GC_dist form) | deployed head |
| 4 knob | periodic (R11); one gap per cell, recalibrated with per-task look prices | same gap |
| 5 partition | camera[t] (new), plus heads/metric/scope per task | — |
| calls | a guard call completes the agentview (.076/.074); a knob-planned call encodes both cameras at stage 1 | unchanged |

If every task is `both`, the system is bit-identical to the deployed system (required test).

## 2. Layer 1: the one-camera cache (wrist tasks)

**Encoding.** Every look of a wrist task encodes only the wrist camera (step 0, the look after a policy call, lifecycle looks 6/7/8 included). π0.5 uses per-request `wrist_only` stage 1; GR00T the genuine single-image `extract_feature` (as `R12CacheAblate` and `lookhalf_cameras.GrootOverride.partial`). Exception: a knob-planned call (`periodic_due` before stage 1) gets a stock two-camera stage 1.

**Retrieval** (identical to the ablation `wrist_sub` arm measured by camdev): query `x = [B1T·key_v1 − muB1, rs8]`; distance `d = ‖(x − X_I)·Wf[I]‖`, I = (wrist PCA64, state8), `Wf` = deployed per-task `T.Wf` (step 0 `T.W0f[I]`), source `ablate/artifacts/<cell>/wrist_sub.pkl` field `camera_metrics[t]`. Selection/synthesis as deployed: top-16, kernel `_kernel_w(dk − dk0, kref)`, kernel mean of library chunks (`_mix`), anchor memo, budget-one blind tail. Regime 1 (look after a MISS) uses the deployed AWM continuity rule on these distances: `d ← d/median(d) + λ_c·c/T.s_c` with c the continuity distance of the executed policy tail `prev_a_exec[5:10,:7]/sig` to the library heads (`T.h2`, `T.HD`), mirroring `R12LookHalf._query_metric`.

**Why sub, not refit**: B pooled sub .732 vs refit .716 (sub better in 3/5 cells); no new fit; corrector and distance scale derive from the deployed metric; camdev measures only sub. Context: refit not better on A either.

**Third-person camera: not offered** (B: agent_sub worse than both and wrist_sub in every cell).

## 3. Layer 5: the per-task camera rule (B-dev only)

**Data**: journals of `both` and `wrist_sub` for the cell from the B screen (K=10) and camdev; accepted terminal episodes only; the code refuses test-A paths.

**Rule** (`tools/camera_choice.py`, frozen 2026-10-04 01:55 CDT):
1. Per task: `d_t = SR_wrist − SR_both`; `se_t² = p̃w(1−p̃w)/n_w + p̃b(1−p̃b)/n_b`, `p̃ = (s+1)/(n+2)` (unpaired, add-one, deliberately conservative).
2. Random effects per cell (DerSimonian–Laird): μ, τ².
3. Optional library covariate: within-cell-centred Δe10_t as meta-regression slope, used only if its WLS z ≤ −2; otherwise plain random effects.
4. Shrinkage: `D_t = m_t + B_t(d_t − m_t)`, `B_t = τ²/(τ² + se_t²)`, `m_t = μ` (or `a + b·Δe10_t`).
5. Choice: `camera_t = wrist` iff `D_t > −C_cell`; ties → both.
6. Tolerance `C_cell = max(.005, s_mix·κ·u/α)`: s_mix = B-measured mixing-line rate (library pure-policy SR − B-dev deployed SR)/(.5 − B-dev deployed IR) (lower-bound exchange rate; R11 periodic knob was above it); κ = .076 / .074; u = B-dev non-call looks per decision; α = .65.

| cell | lib. policy SR | B-dev deployed SR / IR | s_mix (pp per unit IR) | u | C_cell |
|---|---|---|---|---|---|
| pi05 L10-50 | .872 | .811 / .168 | 18.4 | .382 | .008 |
| pi05 Sp-50 | .974 | .886 / .146 | 24.9 | .415 | .012 |
| pi05 Sp-200 | .974 | .978 / .102 | ≤ 0 | .468 | .005 (floor) |
| GR00T Sp-50 | .912 | .852 / .140 | 16.7 | .435 | .008 |
| GR00T L10-50 | .854 | .698 / .217 | 55.2 | .340 | .021 |

7. Output `out/design/camera_choice_screen_camdev.json` (per task d, se, D, posterior sd, B, camera; per cell μ, τ, se_μ, covariate used); sol's builder binds its SHA256.
8. Granularity is data-driven (τ̂ = 0 → one choice per cell; strong heterogeneity → per-task).
9. Unscreened cells (other library sizes): all both.

**Dry run on the K=10 screen only** (provisional, `out/design/camera_choice_screen.json`): pi05 L10-50 μ −.048, τ .135 → wrist {3,6,7}; pi05 Sp-50 −.046, 0 → none; pi05 Sp-200 +.009, 0 → all; GR00T Sp-50 +.033, 0 → all; GR00T L10-50 −.050, .248 → {2,4,5,6,7,8} (meta-regression z −1.95, just short of the gate). Operating characteristic at camdev size (se ≈ .10, B ≈ .5, μ ≈ −.03): P(wrist) ≈ .14 / .46 / .66 for true Δ −10 / 0 / +5 pp.

## 4. Layer 2: guard

**Wrist tasks: H2 as frozen and implemented** (`R12Guard` h2): library-fitted logistic risk on seven agentview-free features (`out/guard_lib.json`); at looks with step ≥ 4 call ⇔ (W-top-1 advance ≤ .5 and risk ≥ θ_a) or risk ≥ θ_b; per-cell θ (GUARD.md §6): pi05 L10-50 .122/.635; pi05 Sp-50 .140/.673; pi05 Sp-200 .146/.755; GR00T Sp-50 .195/.843; GR00T L10-50 .170/.550. H2's features keep its own wrist retrieval (refit `opus_w`), computed at every look on CPU, independent of the served sub retrieval, so H2 stays byte-identical to the tested guard.

**Why not the deployed rule on the served wrist top-1** (B): it reproduces only 64–72 % of the two-camera guard's calls on cache-driven states and fires spuriously on 10–19 % of other looks (OFFLINE §12.3); R9's wrist loop with a wrist-retrieval guard called ~5 steps later and lost 9–15 pp on GR00T; at matched call rate H2 puts more calls in failing episodes and high-divergence states; in the B pilot H2 left no failing episode without a call.

**Both tasks: deployed only-no-progress guard** on the two-camera top-1 (H2's first call was late on L10, +11.5 decisions on pi05 L10-50).

H2's later first call now applies only to L10 tasks put in wrist mode (provisionally 3 of pi05 L10, 6 of GR00T L10); the 500-episode guard-swap estimate (−0.5 pp per cell if all tasks swap) scales with the wrist share. No union rule "H2 ∪ deployed rule on sub top-1". B-screen stop rule S4 watches for starvation. Knob-on cells: the knob calls at decision 2.

## 5. Layer 3: corrector

**Both tasks**: deployed head.

**Wrist tasks: wrist-path head per task** (R10 `loeo_heads` recipe with the sub retrieval): LOEO synthesis with the sub metric over every selected-library row (own episode excluded, top-16 kernel, early block at step 0; `lookhalf/build.py:loeo` with the sub metric). Target `(stored policy chunk − synthesis)[:10,:6]/sig_head`; features (153-d) `[P1 (64), rs8, synth[:10,:7]/sig_head (70), min(step,120)/120, one-hot task]`; 384 RFF (seed 0), ridge 100, equal episode weights (`r10.train.fit_head`, `anchor_weights`); blend .5 on [:10,:6]; never at step 0. Attenuation `factor = clip((2 − d_min/scale_sub)/1.25, 0, 1)` with d_min the sub distance of anchor rows (as `R10Recipe._correction`). `scale_sub`: R10 five-fold distance recipe (per fold refit the joint AWM metric on the training fold, take its restricted block, held-out non-step-0 nearest distances, episode-balanced median). The wrist path must never read `key_v0`. Fit heads for all 10 tasks per cell; the table picks which is served.

## 6. Layer 4: knob recalibration (library-only)

Mechanism unchanged (guard-aware jittered periodic gap cap, keyed coins `R11-gap-v1`, one setting per cell). Same frozen R11 LOEO anchor tables and guard flags (`r11/knob/calibration.tables`); only the IR model changes: non-call look costs `a_t` = .152/.148 (both) or .076/.074 (wrist); every call look costs 1.0. `IR(x) = Σ_ep [a_t·V_ep + (1 − a_t)·M_ep(x)] / Σ N_ep`; bisect x. Sketch `tools/design_knob.py`; all-both reproduces R11 pi05 L10-50 (.25 → 2.6125, .32 → 1.0657, .40 → .4386). Replay keeps deployed guard flags (H2 θ_b matched the deployed rate; periodic absorbs guard-rate errors; expect slight underspend).

| cell (target .32) | deployed gap | half-everywhere gap | calls per look (library) |
|---|---|---|---|
| pi05 L10-50, wrist {3,6,7} (provisional) | 1.0657 | 1.0164 | .569 → .579 |
| pi05 L10-50, all wrist | 1.0657 | 0.9002 | .569 → .605 |
| pi05 Sp-50, all wrist | 0.8707 | 0.7482 | .562 → .598 |
| GR00T Sp-50, all wrist | 0.8576 | 0.7402 | .562 → .597 |
| GR00T L10-50, wrist {2,4,5,6,7,8} | 1.0252 | 0.9493 | .573 → .595 |

Final gap from the decided table; near-ceiling pi05 Sp-200 keeps the knob off.

## 7. Cost accounting (owner compute-share prices)

| event (per decision of 5 controls) | π0.5 | GR00T |
|---|---|---|
| blind decision | 0 | 0 |
| non-call look, both task | .152 | .148 |
| non-call look, wrist task | .076 | .074 |
| guard call at a wrist-task look (wrist + completion + policy) | 1.000 | 1.000 |
| knob-planned call, or any call in a both task | 1.000 | 1.000 |

Camera saving `κ · Σ_{t∈W} ω_t·u_t`; all-wrist ceiling (A off ledgers; B in brackets): pi05 L10-50 .030 (.029); pi05 Sp-50 .034 (.032); pi05 Sp-200 .037 (.036); GR00T Sp-50 .033 (.032); GR00T L10-50 .026 (.025); periodic .32 .018. M2's realised camera saving on A: .014/.015/.020/.014/.008, periodic .008; pure cache .038. Call saving: guard swap on wrist tasks (B-pilot H2 effects −.014/−.010/−.004/−.007/−.002 pro rata); fewer/more guard calls when the wrist cache fails less/more (≈ −.1 to −.4 IR per unit ΔSR).

## 8. Implementation contract for sol

1. Method `R12HalfEverywhere(R11Knob)`; kwargs `library`, `episode_subset`, `camera_table` (path + SHA256), `knob_method ∈ {off, periodic}`, `target`; `allows_wrist_after_miss = True` plus a flag allowing every non-call look of a wrist task to be wrist-only (as `R12CacheAblate`).
2. `plan_cameras` before stage 1: both task → full; wrist task → `wrist_only` unless `periodic_due` → full. Execution-audit failures do not force a full look for wrist tasks (no camera memory); receipts as `R12CacheAblate`; still logged.
3. `query` dispatch by task mode: wrist → §2 retrieval → §5 head → H2; both → deployed R10/R11 stack. Guard state, anchor, blind tail, knob history shared bookkeeping.
4. H2 call on a wrist task: complete the agentview (`complete_session`, reason `h2_call`), then the policy call.
5. Ledger (`lookhalf_cost_ledger` schema): prices wrist .076/.074, full .152/.148, completion .076/.074, policy .848/.852; add `wrist_task_decisions`, `both_task_decisions`, `wrist_task_looks`, `wrist_task_two_camera_looks`.
6. Logs per look: task mode, plan, cameras encoded, completion reason; served top-1, d_min, factor; H2 risk, branch, W advance; deployed span (both tasks); knob due.
7. Build (library-only, `fit_path` / `library_boundary` audit): wrist heads and `scale_sub` per task; camera table bound by hash; knob gap from §6 with the table's prices; H2 artifacts byte-identical.
8. Required CPU tests: (a) all-both reproduces deployed decisions, served bytes, guard flags, ledger exactly (knob off and periodic .32); (b) all-wrist with guard/corrector/knob disabled reproduces `R12CacheAblate wrist_sub` served chunks and blind tails exactly; (c) H2 decisions on wrist tasks equal `R12Guard h2` on identical inputs; (d) knob builder with all-both prices reproduces R11 (2.6125 / 1.0657 / .4386 on pi05 L10-50); (e) engagement: zero two-camera non-call looks in wrist tasks, zero one-camera looks in both tasks, every call has both cameras, wrist path never reads `key_v0`; (f) LOEO heads exclude own episode, fit-read audit clean; (g) full final argv for local and H100 serving; (h) missing or mismatched camera table refused.
9. Roots: B `r12_half_<cell>` plus `r12_half_pi05_l10_50_periodic_ir32` on the K=10 selections (byte-identical), look recording on; A `r12_test_half_<cell>` from the R11 `eval500.json` templates, recording off.

## 9. Frozen predictions (2026-10-04 ~02:00 CDT, before camdev and any closed-loop run)

### 9.1 Model (B-only)
Wrist task full-system ΔSR = α·Δ_t(pure cache) + h; α ~ U(.35, 1.0) centred .65 (1 − B-dev share of pure-cache failures rescued by guard+corrector: .49/.12/.45/.30/.35, mean .34); h (PILOT_ANALYSIS §3) scaled by wrist share: −.5 pp L10, 0 pi05 Spatial, +.3 GR00T Spatial. ΔIR = −κ·Σ_W ω_t·u_t + (wrist decision share)·(B-pilot H2 call effect) − k·ΔSR, k ~ U(.1, .4). Run noise: A paired SE per cell 0.9–2.2 pp; B vs pooled B deployed replicates (H100 for GR00T). Camera-choice uncertainty: per-task Δ_t from the B-screen hierarchical posterior; camdev simulated at real sizes; frozen rule applied; mixed 50/50 with a skeptic model (Δ_t = μ). Tool `tools/design_predict.py`; samples `out/design/predict_sim.json`.

### 9.2 Test A (500 eps vs same-root R12 off; 80 % intervals)

| cell | expected wrist tasks [P(none)] | ΔIR | ΔSR (pp) | SR / IR predicted | vs M2 on A (ΔSR pp / ΔIR) |
|---|---|---|---|---|---|
| pi05 L10-50 | 3.0 [.34] | −.015 [−.041, +.002] | +0.9 [−2.2, +4.3] | .857 / .149 | +2.1 / +.016 |
| pi05 Sp-50 | 2.4 [.61] | −.011 [−.042, +.003] | +0.2 [−1.9, +2.3] | .930 / .120 | −3.6 / +.024 |
| pi05 Sp-200 | 6.9 [.29] | −.030 [−.053, +.001] | +1.1 [−1.6, +4.5] | .983 / .074 | −0.5 / −.002 |
| GR00T Sp-50 | 7.7 [.13] | −.037 [−.060, −.002] | +2.8 [−1.6, +8.0] | .914 / .091 | +2.6 / −.014 |
| GR00T L10-50 | 4.3 [.20] | −.018 [−.038, +.001] | +2.4 [−2.4, +7.9] | .784 / .189 | +8.6 / −.013 |
| pooled (equal weights) | | −.022 [−.033, −.011] | +1.5 [−0.3, +3.3] | | +1.9 / +.002 |

Knob cell pi05 L10-50 periodic .32 vs R12 periodic off (.890 @ .3084): SR .894 [.865, .925] (+0.4 [−2.6, +3.5]); realised IR .307 [.298, .316] (−.001 ± .008); vs M2 periodic (.918 / .2908) −2.4 pp / +.016. At the knob setting the camera saving is re-spent on calls.

Mechanism predictions (80 %): one-camera share of non-call looks in wrist tasks ≥ .97; H2 calls per look on wrist tasks 70–95 % of deployed; first call of any kind (paired) on wrist tasks +1 to +4 decisions later on Spatial, +4 to +12 on L10; calls per failing episode on wrist tasks within ±20 %; ≤ 1 % failing episodes without a call; knob realised IR within −.02 to +.005 of .32.

### 9.3 B screen (100 eps, K=10 inits, vs pooled B deployed; GR00T vs H100 replicates)

| cell | ΔIR | ΔSR (pp) | SR / IR predicted | B M2 arm (SR / IR) | vs B M2 (ΔSR pp / ΔIR) |
|---|---|---|---|---|---|
| pi05 L10-50 | −.015 [−.041, +.003] | +0.9 [−4.9, +6.9] | .820 / .153 | .80 / .146 | +2.0 / +.007 |
| pi05 Sp-50 | −.010 [−.042, +.005] | +0.2 [−4.3, +4.9] | .888 / .136 | .92 / .108 | −3.2 / +.028 |
| pi05 Sp-200 | −.030 [−.054, +.002] | +1.2 [−2.2, +5.1] | .990 / .071 | 1.00 / .072 | −1.0 / −.001 |
| GR00T Sp-50 | −.037 [−.061, −.003] | +2.8 [−3.7, +9.6] | .908 / .093 | .83 / .123 (local host) | host-confounded |
| GR00T L10-50 | −.018 [−.038, +.002] | +2.5 [−6.9, +12.0] | .725 / .197 | .66 / .208 | +6.5 / −.011 |

Knob arm: SR .935 [.88, .98] vs B deployed periodic .933 @ .302; IR .300 [.290, .315]. Caveat: no unused B inits remain, so the K=10 inits are part of the camera-choice data; B-screen SR is in-sample for the choice (expected 0 to +2 pp optimistic bias): a mechanism and collapse check, not an estimate.

### 9.4 After tonight: frozen conditional rule (mechanical)
Before the B screen run and log: (1) `camera_choice.py screen,camdev` → camera table; (2) `design_predict.py conditional out/design/camera_choice_screen_camdev.json` → per-cell ΔSR/ΔIR for A and B (same α, h, k, noise). These become the operative predictions. Screen-only dry-run example (ΔIR / ΔSR pp): pi05 L10-50 −.014 / +0.8; pi05 Sp-50 0 / 0; Sp-200 −.041 / +0.6; GR00T Sp-50 −.046 / +2.6; GR00T L10-50 −.024 / +3.1. A-informed variant (labelled, not used for decisions; prior mean = cell's A pure-cache Δ): pooled −.024 / +1.5; GR00T Sp-50 −.051 / +5.3; pi05 Sp-50 −.024 / +0.6; pi05 Sp-200 −.024 / +0.1; GR00T L10-50 −.010 / +1.0.

## 10. Experiment plan

1. Tonight: camdev (`rounds/r12/ablate/camdev/`; both + wrist_sub; 5 cells; GR00T on H100).
2. Decide cameras: `camera_choice.py screen,camdev`, then `design_predict.py conditional`; log both before any build depending on the table; no change to C, α or the rule after seeing camdev.
3. sol builds §8 in parallel with step 1; only the camera table and knob gap wait for step 2.
4. B screen, 100 eps per arm on the K=10 selections (GR00T on H100): `r12_half_{pi05_l10_50, pi05_spatial_50, pi05_spatial_200, groot_spatial_50, groot_l10_50}` (knob off) and `r12_half_pi05_l10_50_periodic_ir32` (periodic .32 recalibrated); plus same-batch H100 deployed off for each GR00T cell with wrist tasks. A cell with no wrist task is dropped.
   - Stop rules (frozen; can only stop a cell): S1 engagement (any two-camera non-call look in a wrist task, one-camera look in a both task, or call without both cameras → bug); S2 implementation (one-camera share of non-call looks in wrist tasks < .95, or camera part of ΔIR off the conditional prediction by > .008 → bug); S3 collapse (cell SR < pooled deployed B − 10 pp, or wrist-task SR < deployed on same tasks − 15 pp → drop from A); S4 starvation (wrist tasks: calls per failing episode < 70 % of deployed, or ≥ 2 failing episodes without a call → drop); S5 knob (realised IR outside .32 ± .02 → drop knob arm).
   - Read-outs split by task mode: first call of any kind (paired), calls per failing / winning episode.
5. Test A: 500 eps, one run each, one host type per cell (GR00T on H100): `r12_test_half_<cell>` for each cell that passed with ≥ 1 wrist task, plus `r12_test_half_pi05_l10_50_periodic_ir32`. Comparators: existing `r12_test_<cell>_{off, m2, h2}` and periodic ones (same A inits; 1–2 pp batch noise). Optional fresh same-batch GR00T off arms. Statistics: pairing by (task, init); per-cell paired ΔSR, ΔIR split camera/calls; wrist-task vs both-task breakdown; pooled stratified bootstrap with equal cell weights. Claims: (a) pooled ΔIR < 0 with pooled ΔSR 95 % lower bound > −1.5 pp; (b) vs M2: similar IR, better SR; (c) knob cell: realised IR on target, SR ≥ deployed periodic − 2 pp.

## 11. Deliberately not in this design
- M2 cascade on both-mode tasks (would recover ~.015–.025 IR where most tasks stay both, e.g. pi05 Sp-50, but was harmful where one camera is harmful; no per-task evidence; owner decision 1).
- H2 on both-mode tasks (saves .002–.014 IR on L10 but late first call).
- Refit wrist metric; third-person camera; re-encoding both cameras jointly at GR00T guard calls.
- Any per-task knob parameter (R11 rule).

## 12. Re-check of pending runs (02:00 CDT)
pi05 Sp-50 wrist_refit .836 (+0.4 ± 3.7 vs both .832); pi05 Sp-200 both .966, wrist_sub .958 (−0.8 ± 2.3), wrist_refit .956 (−1.0 ± 2.2), vision cost .078 → .039. Nothing in the design changes.

## Hand-back note
CPU only; no git, launches, pkill or rm -rf; `rounds/r12/astra/` not read. New files in `rounds/r12/opus/tools/` (`design_*.py`, `camera_choice.py` — frozen rule, `design_predict.py`) and `rounds/r12/opus/out/design/`. After camdev finishes:
1. `taskset -c 22-37,66-81 .venv/bin/python exp/offline_search/rounds/r12/opus/tools/camera_choice.py screen,camdev`
2. `.venv/bin/python exp/offline_search/rounds/r12/opus/tools/design_predict.py conditional exp/offline_search/rounds/r12/opus/out/design/camera_choice_screen_camdev.json`
