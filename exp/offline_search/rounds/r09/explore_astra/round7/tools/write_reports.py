"""Render the reviewable report from frozen evidence; never changes prediction or fits."""
import json
from .data import *


def main():
    diag=json.loads((HERE/'results/diagnosis.json').read_text())
    ev=json.loads((HERE/'results/evaluation.json').read_text())
    cv=json.loads((HERE/'SELECTION.json').read_text())
    extra=json.loads((HERE/'results/candidate_diagnostics.json').read_text())
    plan=json.loads((RUN/'h100_sync/plan.json').read_text())
    def label(c): return 'GR00T LIBERO-10-50' if 'l10' in c else 'GR00T Spatial-50'
    lines=['# Round7: diagnosis of the GR00T gap', '',
        'The evidence favors an underfit shared observation head plus a trajectory-distribution gap. It does not support a gripper-sign, 32-D padding, or chunk-length implementation error in round6. Missing row-table entries are ruled out in both GR00T cells; sparse or unrepresentative observation coverage remains possible. Larger shared heads improve held-out-init CV on exactly the same training corpus, so capacity is a directly supported contributor. The data do not isolate it as the sole cause of the closed-loop success deficit.', '',
        '## Admission, split and estimands', '',
        '`tools/data.py` rejects forbidden root prefixes before opening paths. JSONL records are lexed for top-level task/init or UID first; only admitted 20–29 records are deserialized. Conflicting/duplicate identities fail closed. Accepted terminal journal attempts are joined to server decisions, and every episode must have exactly contiguous step indices. All four GR00T round6 arms have 100 admitted complete episodes. No raw holdout root, init 30–49 trajectory, or pi0.5 outcome is used.', '',
        'Offline input is the existing A/CU/IP compact corpus under the standard store. Lazy task/init members are checked before payload access; an archive with any init outside 0–29 is rejected. Training and CV subsets use only 0–19. Evaluation uses 20–29. The same init is held out jointly across tasks and A/CU/IP (`init % 3`); this prevents repeated paths of the same initialization from crossing a fold. Training balances arm-episodes. Per-cell selection was frozen at '+cv['timestamp_utc']+' before candidate evaluation.', '',
        'Round6 20–29 outcomes were intentionally used to design round7. Reusing these inits makes round7 a development screen, not an independent final test. The older per-task controls used more trajectory variants: 80,108 long / 19,364 spatial anchors, versus 20,321 / 7,597 for both round6 and round7 shared heads. This is not a matched-data causal experiment on task identity.', '',
        'Headline offline MSE is averaged within episode, then over 100 episodes. Diagnostic task/phase/distance tables below average anchors in each stratum and therefore have different overall totals. MSE uses model-normalized motion channels 0–5 and first ten controls. Labels are policy shadow draws, not optimal actions or success. Live phases describe commanded gripper state, not verified physical grasp/contact.', '',
        '## Closed-loop reproduction', '',
        'Owner IR uses the historical GR00T stage shares, stage1 6.145862978883088 ms, stage2 7.191964512458071 ms, stage3 28.104 ms, full K=8: `IR = look_share*Nlook/N + (1-look_share)*Ncall/N`. This reproduces the supplied owner numbers; it is not the pi0.5 .152/.848 pricing and not the newer eager cost table.', '',
        '| Cell | Arm | Success / 100 | Decisions | Looks | Calls | Owner IR |', '|---|---|---:|---:|---:|---:|---:|']
    for cell,v in diag.items():
        for arm,a in v['live']['arms'].items():
            r=a['total'];lines.append(f"| {label(cell)} | {arm} | {a['success']} | {r['n']} | {r['look']} | {r['calls']} | {r['ir']:.6f} |")
    lines += ['', 'Long: +9/−15 discordant pairs, SR delta −0.06; paired init-cluster 95% interval [−0.15, +0.03]. Spatial: +2/−7, delta −0.05, interval [−0.10, 0.00]. Bootstrap: 5000 draws, ten init clusters, all ten tasks retained. These are small-sample descriptive intervals.', '',
        'All 846→972 long calls and 52→94 spatial calls carry `os_reason=4`, the unchanged no-progress guard. No new recovery trigger or call policy is present. Logs show larger state-nearest-neighbor distances (`dnn`: .1765→.2161 long, .1495→.1890 spatial) and larger progress lag (4.89→6.68, .44→.91). This is consistent with poorer subsequent trajectories, rather than a direct increase in the programmed call rate.', '',
        '| Cell | Paired outcome group | Episodes | Control decisions / calls | Task-free decisions / calls |', '|---|---|---:|---:|---:|']
    for cell,v in diag.items():
        for group,r in v['live']['paired'].items():
            lines.append(f"| {label(cell)} | {group} | {r['n']} | {r['control_decisions']} / {r['control_calls']} | {r['taskfree_decisions']} / {r['taskfree_calls']} |")
    lines += ['', 'The lost pairs generate +274 calls long and +50 spatial; gained pairs offset −201 and −14. Even pairs successful under both arms have +66 calls long and +6 spatial. In lost pairs, calls per decision rise from 12.25% to 24.74% long, and 5.95% to 19.48% spatial. Thus extra calls are not explained solely by failed episodes lasting longer. They still do not prove each preceding chunk is worse: the two trajectories diverge, success-conditioned groups are post-treatment, and round6 has no online policy-shadow labels.', '',
        'At fresh cache hits, subtracting the weighted retrieved library action from the logged served head gives mean correction RMS .02921→.03484 long and .03341→.04111 spatial. Task-free corrections are larger despite the same .5 strength. Within task-free trajectories, corrections preceding a call in the next four decisions average .03806 versus .03233 long, and .05815 versus .03935 spatial. These associations can reflect difficult states and feedback, not a causal dose-response.', '',
        '## Where the gap appears', '',
        'Task IDs here are reporting strata only. No reported task value is a fitting feature or runtime selector. SR entries are successes out of ten. Offline deltas are round6 task-free minus current per-task MSE on the identical A-path states.', '',
        '| Cell | Task | Control→task-free SR count | Control→task-free calls | Offline ΔMSE | Correction disagreement RMS |', '|---|---:|---|---|---:|---:|']
    for cell,v in diag.items():
        for t,r in v['offline']['by_task'].items():
            c=v['live']['arms']['control']['by_task'][t]; a=v['live']['arms']['taskfree']['by_task'][t]
            lines.append(f"| {label(cell)} | {t} | {c['success']}→{a['success']} | {c['calls']}→{a['calls']} | {r['delta']:+.6f} | {r['correction_disagreement_rms']:.5f} |")
    lines += ['', 'Long-task net losses concentrate in task 0 (−4) and task 8 (−3); task 9 and 2 lose one each, offset by tasks 3, 5 and 6 gaining one. The largest offline MSE deficits are tasks 9, 0 and 4. Task 8 has a smaller teacher-error deficit but a large success loss: mean teacher MSE is not a reliable task-success surrogate. Spatial loses two on task 9, one on 0/2/4/8 and gains one on task 1; task 0 has the largest offline deficit, but task 8 actually has lower task-free offline MSE.', '',
        'Phase labels use GR00T normalized negative gripper = close. Approach is before first close; close transitions mark the previous/current decision as grasp; open transitions mark previous/current/next as release; remaining closed/open decisions are carry/post. They are retrospective diagnostics only, never serving inputs. Offline labels use all recorded executed commands before selecting fresh-look anchors.', '',
        '| Cell | Phase | A anchors | Offline ΔMSE | Control→task-free live calls |', '|---|---|---:|---:|---|']
    for cell,v in diag.items():
        for ph,r in v['offline']['by_phase'].items():
            c=v['live']['arms']['control']['by_phase'][ph];a=v['live']['arms']['taskfree']['by_phase'][ph]
            lines.append(f"| {label(cell)} | {ph} | {r['n']} | {r['delta']:+.6f} | {c['calls']}→{a['calls']} |")
    lines += ['', 'Long task-free extra calls occur in approach (+31), carry (+11), grasp (+7), post (+57), release (+20); the largest offline deficit is post (+.002577), then carry (+.001995). Spatial extra calls are primarily post (+19), release (+15), carry (+11), while approach/grasp decline. Only five spatial A anchors are post, so that stratum is highly uncertain. Spatial release has better task-free teacher MSE despite worse live release outcomes; the trajectories and denominators differ.', '',
        '| Cell | Cached chunk contains a gripper sign transition | A anchors | Offline ΔMSE |', '|---|---|---:|---:|']
    for cell,v in diag.items():
        for event,r in v['offline']['by_grip_event'].items(): lines.append(f"| {label(cell)} | {event} | {r['n']} | {r['delta']:+.6f} |")
    lines += ['', 'A gripper event means any sign transition within the cached ten-control chunk. Long event/non-event deficits are +.001297/+.001792; Spatial is −.000506/+.000579. The gap is not confined to gripper transitions. Cache/policy shadow gripper-sign disagreement is 17.03% long and 9.96% spatial, but both correctors deliberately leave the same cached gripper untouched on the same state. This can limit either motion-only method without being a round6 sign bug.', '',
        '## Retrieval distance and row coverage', '',
        'Offline `d1` is the recorded retrieval metric, split by training-A median/p90 (long 10.0037/29.0289; spatial 13.0405/22.4821). Live `dnn` is the inherited guard’s nearest robot-state distance, not the same metric. Do not compare their numerical scales.', '',
        '| Cell | Offline d1 bin | A anchors | Control MSE | Round6 MSE | ΔMSE |', '|---|---|---:|---:|---:|---:|']
    for cell,v in diag.items():
        for b,r in v['offline']['by_distance'].items():
            lines.append(f"| {label(cell)} | {['≤train median','median–p90','>train p90'][int(b)]} | {r['n']} | {r['control_mse']:.6f} | {r['r6_mse']:.6f} | {r['delta']:+.6f} |")
    lines += ['', 'The far-distance deficit is clear in both cells. Live dnn bins share the control median/p90 thresholds within each cell. Long high-distance look counts increase 298→388 and calls 222→281; spatial increases 113→169 looks and 28→69 calls. Spatial high-distance states account for 41 of the 42 net extra calls. In long, the middle-distance bin also contributes +75 calls. This points to later-state drift and difficult retrieval neighborhoods; no absolute-distance threshold is introduced in either candidate.', '',
        '| Cell | Library rows | Rows with zero training mass | Training retrieval mass p10 / median | Evaluation weight on unsupported rows |', '|---|---:|---:|---|---:|']
    for cell,v in diag.items():
        r=v['offline']['coverage'];lines.append(f"| {label(cell)} | {r['rows']} | {r['zero_mass']} | {r['mass_quantiles'][1]:.3f} / {r['mass_quantiles'][2]:.3f} | {r['eval_weight_unsupported']:.1%} |")
    lines += ['', 'Weighted row support is broad: the A-evaluation p10 row-support average is 45.7 long / 57.5 spatial arm-episodes (maximum 60 = 20 inits × three paths per library task neighborhood). Those paths are not 60 independent inits. Some rows have low effective mass and local table quality can still vary. However, unsupported rows cannot explain the gap: their evaluation mass is exactly zero. Shared residual RMS is .1520/.1240 versus local-table .0252/.0172 long/spatial; the table contributes a much smaller correction. This motivates increasing shared capacity before expanding the row table.', '',
        '## Action-space and serving audit', '',
        '- GR00T libraries are `[2645,16,32]` and `[1063,16,32]`. Only channels 0–6 are physical LIBERO actions. Channels 7–31 are noise-like padding, not zeros (observed max absolute 4.797/4.577). Both old and new corrector inputs exclude these channels and their residual targets use only 0–5. New confidence also excludes padding. Tests perturb padding and unused steps without changing physical correction.',
        '- Normalized GR00T −1 means close, +1 means open. Both controls have the established GR00T stack sign. Reconstructed round6 fresh-cache served gripper differs from its retrieved gripper by <4.4e-7, compatible with arithmetic roundoff. The correction does not write channel 6.',
        '- H=16, but these arms commit only the first ten controls in two five-control decisions. Corrected motion is applied exactly once on the judge `os_synth` path, then stored as the blind-tail anchor. Steps 10–15 remain byte-identical; no full-16 training or unintended padding normalization occurs.',
        '- Round6 offline deficits exist in both halves: long first/second five ΔMSE +.001801/+.001638, spatial +.000355/+.000354. There is no evidence of a tail-specific off-by-five defect.',
        '- The new correction API takes observation keys, robot state, elapsed step, cached action, retrieval rows and weights. A poison-identity object raises on task/episode/init access and still passes. Row/library permutation with corresponding retrieval remapping gives identical predictions. Existing retrieval and guard bookkeeping are inherited unchanged; task-indexed models remain only in the requested controls.', '',
        '## Frozen alternatives and training evidence', '',
        'Six head recipes were compared, each with confidence gain 0/.1/.25: round6 768 RFF, wider 1536/3072 without added features, 1536/3072 with 24 GR00T action/gripper features, and the enriched 3072 head at alpha 30. All other alphas are 100. Each cell independently selects one fixed-.5 capacity recipe and one positive-confidence-gain recipe. Both cells select enriched 3072/alpha100; confidence gain .25. No 20–29 score enters this choice.', '',
        '| Recipe (same training data) | Long CV MSE/cache MSE | Spatial CV MSE/cache MSE |', '|---|---:|---:|']
    for name in ('r6:0.0','wide1536:0.0','wide3072:0.0','grip1536:0.0','grip3072:0.0','grip3072_a30:0.0','grip3072:0.25'):
        lines.append(f"| {name} | {cv['cells'][CELLS[0]]['scores'][name]:.6f} | {cv['cells'][CELLS[1]]['scores'][name]:.6f} |")
    lines += ['', 'Width alone improves CV by 4.89% long / 3.35% spatial relative to round6; enriched width improves 5.78% / 4.66%. Lower regularization worsens Spatial strongly, so merely fitting harder is insufficient. The capacity result is a same-corpus ablation supporting shared-head underfit. More diverse training paths could also help, but expanding to unvalidated caches was not necessary for this round.', '',
        'Both variants use the exact same shared head and row residual table per cell. Features are the prior 207 inputs plus closure fraction/first/last/event count (4), mean motion (6), second-half minus first-half motion (6), mean motion × closure fraction (6), and two gripper-state values × closure fraction (2). RFF seed is 260602. Row residual ridge lambda stays 1.', '',
        '`capacity` applies .5*(shared+local). `confidence` applies `[.5+.25/(1+v/s)]*(shared+local)`. The dispersion v is the original retrieval-weighted squared motion spread across ten controls × six channels. s is the training-cell median. This is an agreement proxy, not a calibrated confidence probability: mutually agreeing retrieved chunks can still be distant or wrong. Low agreement returns strength towards .5; it never turns the correction off. The head/table fitted values and s never use task IDs.', '',
        '## Frozen offline evaluation, 20–29', '',
        '| Cell | Method | A episode MSE | A relative to control | CU relative | IP relative | A episodes better than control |', '|---|---|---:|---:|---:|---:|---:|']
    for cell,vs in ev.items():
        for variant,r in vs.items():
            a=r['paths']['A']['candidate'];cu=r['paths']['CU']['candidate'];ip=r['paths']['IP']['candidate']
            lines.append(f"| {label(cell)} | {variant} | {a['mse']:.6f} | {a['relative_to_control']:+.2%} | {cu['relative_to_control']:+.2%} | {ip['relative_to_control']:+.2%} | {a['better_than_control']} / 100 |")
    lines += ['', 'Reference A MSE: current controls .014408 long / .012254 spatial; round6 task-free .015723 / .012548. Relative to round6, capacity reduces A MSE 7.62% / 4.69%, confidence 14.64% / 8.68%. Capacity therefore nearly closes the offline long gap and crosses the spatial mean; confidence beats control mean MSE in all six cell/path combinations.', '',
        '| Cell | Variant | Paired A ΔMSE | Init-cluster 95% interval |', '|---|---|---:|---|']
    for cell,vs in ev.items():
        for variant,r in vs.items():
            a=r['paths']['A']['candidate'];lo,hi=a['difference_ci95'];lines.append(f"| {label(cell)} | {variant} | {a['difference_to_control']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines += ['', 'These reuse round6’s ten evaluation init clusters and policy draws. Confidence Spatial improves the mean but only 46/100 individual episodes; there is meaningful heterogeneity. Candidate diagnostic task-average deltas improve for all ten Spatial tasks, which does not imply each episode improves. Its stronger correction may alter grasp geometry or future guard activation. These results justify the two frozen closed-loop screens, not a claim that the GR00T SR gap is already closed.', '',
        '## Runtime and reproducibility', '',
        'Serving prediction adds no new look/call. Single-thread CPU correction median is 1.24–1.31 ms, p95 1.35–1.55 ms (300 warmed synthetic-key queries, including projection; excluding retrieval/guard/GPU inference). Round6 measured roughly .80–.83 ms median on its fixture. This is local overhead, not H100 latency; owner IR does not price it. See `results/candidate_diagnostics.json` for tensor sizes and strength quantiles. Confidence median strength on evaluation A is .6226 long / .6290 spatial.', '',
        'Evidence files: `results/diagnosis.json`, per-cell CV/evaluation files, `results/candidate_diagnostics.json`, `results/plugin_selftests.json`, unit-test and standard-plan logs; `SELECTION.json`, `FROZEN.json`, `INTEGRITY.json`. Source pointers: `tools/data.py` admission/phase definitions, `tools/analyze.py` live/offline comparisons, `tools/learn.py` training-only selection, `tools/numeric.py` serving math, `tools/methods.py` exact corrector swap.']
    (HERE/'DATA_ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    report='''负责人说明：这轮已经备好两个只用于 GR00T 的无任务编号纠正器，并为两个测试单元各放入同批的原纠正器对照，共六个方案、六百个计划回合。预测先写入并冻结，随后才发包。没有启动评测、同步远端或改变现有进程。

主要发现是共享纠正头容量不足，训练轨迹种类较少也可能有影响；没有证据表明差距来自抓手正负号、32 维补齐通道或动作长度错误。两个单元的检索行都有训练数据，但这不等于所有状态都覆盖充分。多出来的模型调用全部由原来的“无进展”保护触发，主要集中在新方案失败而对照成功的回合；这支持轨迹质量变差的解释，但日志没有在线标准答案，不能把每次调用都解释为前一个动作错误。

第一方案扩大共享头，保持一半纠正强度。第二方案使用同一套拟合参数，在检索动作更一致时适度加大强度。离线长任务／Spatial 的动作误差相对对照分别为第一方案 +0.8%／−2.4%，第二方案 −6.8%／−6.5%。第二方案的 Spatial 平均误差较低，但只有 46/100 个回合的误差低于对照。闭环成功率是否补回来，仍需这次同批实验判断。

# Frozen package

| Cell | Same-batch per-task control | Capacity candidate | Confidence candidate |
|---|---|---|---|
| GR00T LIBERO-10-50 | `r9a7_groot_l10_50_control` | `r9a7_groot_l10_50_capacity` | `r9a7_groot_l10_50_confidence` |
| GR00T Spatial-50 | `r9a7_groot_spatial_50_control` | `r9a7_groot_spatial_50_capacity` | `r9a7_groot_spatial_50_confidence` |

One shared same-batch control per cell serves both candidate comparisons. Controls are byte-identical to round6. Both candidates use a 3072-RFF, alpha-100 shared head with 231 observation/action inputs and a retrieval-row residual table. Fixed strength .5 versus `.5+.25/(1+dispersion/train_median)`. No task id, one-hot, task head, task selector or task threshold is introduced. Both models fit only 0–19; all emitted manifests select exactly tasks 0–9 × inits 20–29.

Round6 loss diagnosis: long −6 pp, concentrated in tasks 0/8; Spatial −5 pp, largest on task 9. All extra calls are no-progress calls. Long offline deficit is greatest after release and during carry; spatial offline deficit is stronger during carry and in the farthest retrieval-distance bin. Gripper-event deficits do not explain the pattern on their own. See [DATA_ANALYSIS.md](DATA_ANALYSIS.md) for task/phase/distance tables and causal limits.

# Validation and handoff

- Fifteen unit tests passed: identity-first rejection, train/eval separation, task-input poison, row permutation, padding/tail invariance, strength bounds, exact control copies, unchanged non-corrector state, and judge correction/anchor behavior.
- Twelve standard-plugin CPU selftests passed: all six production artifacts plus forced-call fixtures; each covers two connections, two training episodes, 48 decisions. Forced fixtures exercise policy miss/tail paths; emitted arms retain `max_calls=0` and no forced trigger indices.
- The standard controller `plan` passed with six arms, PLAN_FILES files, PLAN_BYTES bytes. Standard store `/home/weiland/trace_runs/offline_search_store`; standard emitter and standard controller; no serving-store subset or control adapter.
- `H100_SOURCES.json` and `H100_SOURCES.sha256` record the 69-file experiment source import closure; `H100_NEW_SOURCES.sha256` lists the four new serving files. Static inventory and local hashes are complete. Remote hashes were not checked.
- Local correction latency: median 1.24–1.31 ms; p95 1.35–1.55 ms. No H100 timing or round7 closed-loop outcome is claimed.

Run root: `/home/weiland/trace_runs/os_closed_loop/r09_astra_r7/`.
Prediction: [PREDICTION.md](PREDICTION.md), frozen before emission. Source staging, checks and coordinator-only standard sync/chain commands: [HANDBACK.md](HANDBACK.md). Local integrity seal: `INTEGRITY.json`.

This is an iterative development screen on the reused 20–29 evaluation population. Forecasts remain conservative: capacity −2 pp in each cell; confidence −1 pp long / −2 pp spatial against the new same-batch controls. No claim of closed-loop parity is made before those runs.
'''
    report=report.replace('PLAN_FILES',str(len(plan['files']))).replace('PLAN_BYTES',f"{plan['bytes']:,}")
    (HERE/'REPORT.md').write_text(report)

if __name__=='__main__': main()
