"""Emit every REPORT number from executed analysis or declared forecast priors."""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import json
import numpy as np
from exp.offline_search.rounds.r11.astra.boundary import HERE, dump, install
from exp.offline_search.rounds.r11.astra.experiment import CELLS
from exp.offline_search.rounds.r11.astra.freeze import markdown_table as table


def main():
    install()
    freeze=json.loads((HERE/'freeze.json').read_text())
    analyses=[json.loads((HERE/'data'/f'{m}_{s}_{n}'/'analysis.json').read_text()) for m,s,n in CELLS]
    cross=json.loads((HERE/'crosscheck.json').read_text())
    supplement=json.loads((HERE/'supplement.json').read_text())
    check=json.loads((HERE/'selfcheck.json').read_text())
    all_curves=[];all_values=[];matched=[];adaptive=[]
    for a in analyses:
        all_curves.extend(dict(cell=a['cell'],**r) for r in a['curve'])
        all_values.extend(dict(cell=a['cell'],**r) for r in a['evidence'])
        target=.32 if a['size']==50 else .25
        uniform=next(c for c in a['calibrations'] if c['method']=='random_baseline' and c['target']==target)
        for method,beta in (('distance',1.),('predicted_error',1.),('predicted_error',.5),('disagreement',1.)):
            c=next(c for c in a['calibrations'] if c['method']==method and c['beta']==beta and c['target']==target)
            added=c['captured']-a['base']['captured']
            rnd_added=uniform['captured']-a['base']['captured']
            matched.append(dict(cell=a['cell'],method=method,beta=beta,target=target,
                additional_risk_lift=added/rnd_added,miss_per_look=c['miss_per_look'],
                random_miss_per_look=uniform['miss_per_look'],guard_fraction=a['base']['guard_per_look'],
                raw_signal_overlap=next(r['base_signal_overlap'] for r in a['overlaps'] if r['method']==method)))
        for d in a['adaptive']:
            vals=d['scenarios']
            adaptive.append(dict(cell=a['cell'],target=d['target'],feasible=d['feasible'],initial_dose=d['dose'],
                predicted=d['validation']['owner_ir_mean'],
                static_max_error=max(abs(s['static']['owner_ir_mean']-d['target']) for s in vals),
                adaptive_max_error=max(abs(s['adaptive']['owner_ir_mean']-d['target']) for s in vals)))
    for name,rows in (('offline_curves.csv',all_curves),('value_evidence.csv',all_values),
                      ('matched_ir_value.csv',matched),('adaptive_checks.csv',adaptive)):
        with (HERE/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    ev=[r for r in all_values if r['stride']==2 and r['dose']==.4]
    pred=[r for r in ev if r['method']=='predicted_error']
    dist=[r for r in ev if r['method']=='distance']
    disc=[r for r in ev if r['method']=='disagreement']
    grip_long=[r for r in ev if r['method']=='gripper_transition' and '_l10_' in r['cell']]
    robust=[r for r in supplement if r['method']=='predicted_error']
    hybrid=[r for r in matched if r['method']=='predicted_error' and r['beta']==.5]
    selected_adaptive=[r for r in adaptive if r['cell'].endswith('_50')]
    stats=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),cells=len(analyses),arms=freeze['arm_count'],
        rows=sum(a['rows'] for a in analyses),episode_memberships=sum(a['episodes'] for a in analyses),
        error_lift_range=[min(r['lift'] for r in pred),max(r['lift'] for r in pred)],
        distance_lift_range=[min(r['lift'] for r in dist),max(r['lift'] for r in dist)],
        disagreement_lift_range=[min(r['lift'] for r in disc),max(r['lift'] for r in disc)],
        long_grip_lift_range=[min(r['lift'] for r in grip_long),max(r['lift'] for r in grip_long)],
        error_rms_lift_range=[min(r['rms_lift'] for r in robust),max(r['rms_lift'] for r in robust)],
        within_episode_lift_range=[min(r['within_episode_lift'] for r in robust),max(r['within_episode_lift'] for r in robust)],
        added_hybrid_lift_range=[min(r['additional_risk_lift'] for r in hybrid),max(r['additional_risk_lift'] for r in hybrid)],
        max_shared_difference=max(r['max_shared_model_absolute_error'] for r in cross['comparisons']),
        max_static_mc_error=max(abs(r['simulation_error']) for r in cross['comparisons']),
        max_adaptive_validation_error=max(abs(r['predicted']-r['target']) for r in selected_adaptive),
        adaptive_stress_max_error=max(r['adaptive_max_error'] for r in selected_adaptive),
        static_stress_max_error=max(r['static_max_error'] for r in selected_adaptive),
        final_head_transfer_max_error=max(abs(r['final_head_on_OOF_features_ir']-r['dag_ir']) for r in cross['comparisons']),
        bootstrap_draws=2000,matched_miss_fraction=.4,checks=len(check['checks']),
        prediction_timestamp=freeze['timestamp_utc'])
    dump(HERE/'report_stats.json',stats)
    fmt_range=lambda x,d=2:f'{x[0]:.{d}f}–{x[1]:.{d}f}'
    value_rows=[]
    for a in analyses:
        row=[a['cell']]
        for method in ('distance','predicted_error','disagreement','gripper_transition'):
            r=next(x for x in ev if x['cell']==a['cell'] and x['method']==method)
            row.append(f"{100*r['capture']:.1f}% / {r['lift']:.2f}×")
        value_rows.append(row)
    curves=[]
    for a in analyses:
        for q in (.2,.4,.6):
            row=[a['cell'],f'{q:.2f}']
            for method,beta in (('distance',1.),('predicted_error',.5),('disagreement',1.)):
                r=next(c for c in a['curve'] if c['method']==method and c['beta']==beta and c['dose']==q)
                row.append(f"{r['miss_per_look']:.3f} / {r['owner_ir']:.3f}")
            curves.append(row)
    off=['# Offline library evidence and knob curves',
        'Generated by executed `write_docs.py`, using `experiment.py`, `analyze.py`, `supplement.py`, and `crosscheck.py`. '
        'All quantitative fitting and calibration inputs are selected B libraries or matching B-only R10 artifacts. '
        'No test-A closed-loop output, non-test recording or curated current library was used.',
        '## Data and validation boundary',
        table(['Cell','Library rows','Episodes','Failed episodes retained','Guard-only IR','No-guard IR','All-call IR'],[
            [a['cell'],a['rows'],a['episodes'],a['failed_library_episodes'],f"{a['base']['owner_ir']:.4f}",
             f"{a['no_guard']['owner_ir']:.4f}",f"{a['ceiling']['owner_ir']:.4f}"] for a in analyses]),
        f"There are {stats['rows']:,} row memberships and {stats['episode_memberships']:,} episode memberships across "
        f"{stats['cells']} nested cell-sizes; these are not counts of distinct episodes because the subsets overlap. "
        'Whole episodes, including all their nearby frames, are excluded from validation donors and metric fits. '
        'The inherited fixed selected-library PCA is disclosed. Dense libraries use five held-out episode groups; '
        'their donor banks are smaller than deployment. The error predictor has an additional strictly nested episode split.',
        '## Matched miss fraction: error captured',
        f"At {100*stats['matched_miss_fraction']:.0f}% mean miss fraction, a random selector captures "
        f"{100*stats['matched_miss_fraction']:.0f}% of the residual by expectation. Entries show captured corrected motion MSE / lift over random. "
        'Queries are even-step ten-control anchors, with equal episode mass. The target is the stored policy chunk minus '
        'the outer-held-out, distance-attenuated corrected cache chunk, normalized by training-library action scale.',
        table(['Cell','Distance','Predicted error','Neighbour disagreement','Donor gripper transition'],value_rows),
        f"The predictor's lift remains {fmt_range(stats['error_rms_lift_range'])}× when using RMS rather than squared error, "
        f"and {fmt_range(stats['within_episode_lift_range'])}× against a conditional-random baseline that preserves each "
        'episode’s own call fraction. Thus part of the signal is useful placement within an episode, rather than only '
        'allocating more calls to difficult episodes. `supplement.json` also includes a winsorized-tail check.',
        '## Does the learned score beat the cheap disagreement score?',
        table(['Cell','Extra corrected MSE captured, pp','Episode-bootstrap interval, pp'],[
            [a['cell'],f"{100*p['delta_capture']:+.2f}",f"[{100*p['low']:+.2f}, {100*p['high']:+.2f}]"]
            for a in analyses for p in a['predicted_vs_disagreement'] if p['stride']==2 and p['dose']==.4]),
        f"Intervals use {stats['bootstrap_draws']} episode resamples with the fitted selectors held fixed. "
        'They are exploratory, unadjusted for candidate comparison, and do not measure SR. '
        'The predictor’s advantage over disagreement is often small, particularly for GR00T; this motivates a limited '
        'cheap-signal comparator rather than a claim that learned placement is established.',
        '## Value after accounting for guard overlap',
        'The following entries compare **additional** corrected-error mass captured after removing the guard’s calls, '
        'against a uniform knob calibrated to the same total owner IR on the identical B sequences. '
        'This table uses pooled anchors, matching the pooled cost ledger. It is distinct from the episode-balanced table above.',
        table(['Cell','Target IR','Distance lift','Pure predicted-error lift','Half-random error lift','Disagreement lift'],[
            [a['cell'],'.32' if a['size']==50 else '.25']+
            [f"{next(x['additional_risk_lift'] for x in matched if x['cell']==a['cell'] and x['method']==m and x['beta']==b):.3f}"
             for m,b in (('distance',1.),('predicted_error',1.),('predicted_error',.5),('disagreement',1.))] for a in analyses]),
        '`matched_ir_value.csv` also records raw guard/signal overlap and both methods’ identical total miss fractions. '
        'No policy benefit is observed here: replacing the cache action by the recorded policy action makes its proxy error zero '
        'by construction, while the resulting state and success are unknown.',
        '## Dose → total miss fraction → owner IR',
        'Dose q is the high-score fraction in the pooled held-out all-row distribution. The half-random error mixture is '
        '`q/2 + threshold_indicator/2`; total miss fraction below is policy calls / real looks, **including the guard**. '
        'Each entry is total miss fraction / owner IR. Full curves, including gripper candidates, endpoint costs, '
        'thresholds, tie probabilities and overlap are in `offline_curves.csv`.',
        table(['Cell','Dose q','Distance: M/V / IR','Error hybrid: M/V / IR','Disagreement: M/V / IR'],curves),
        f"`crosscheck.py` reproduces opus’s shared cost/guard formula on identical astra input arrays to maximum "
        f"absolute difference {stats['max_shared_difference']:.3g}. The independent stochastic simulator differs from "
        f"the analytic static IR by at most {stats['max_static_mc_error']:.4f} in the checked runs. "
        'Normal reachable looks remain even: a stalled anchor already fires the mandatory guard before a cache-tail veto '
        'could create an extra look. The shared model’s cost logic is reused; its separately fitted retrieval tables are not mixed into this analysis.',
        '## Adaptive threshold: useful robustness, limited reachability',
        table(['Cell','Target','Initial dose','Held-out-seed IR','Feasible','Max static scenario error','Max adaptive scenario error'],[
            [r['cell'],f"{r['target']:.2f}",f"{r['initial_dose']:.4f}",f"{r['predicted']:.4f}",r['feasible'],
             f"{r['static_max_error']:.4f}",f"{r['adaptive_max_error']:.4f}"] for r in adaptive]),
        'The calibration and validation seed sets are independent random draws on the **same B state paths**, not independent '
        'environment episodes. The stress scenarios shift score ranks by ±.20 or substitute a deterministic persistent-stall '
        'pattern. They test cost feedback mechanics, not true distribution-shift coverage. Low-budget dense GR00T adaptation '
        'was infeasible even at the lowest initial dose and is dropped. The sparse higher-budget arms remain useful tests; '
        'short episodes still show transient overspend and mandatory guards impose a floor.',
        '## Selection and limits',
        'Drop standalone imminent-gripper and gripper-disagreement triggers: their long-task motion-error concentration '
        'is approximately random, even though Spatial has some gripper-related signal. Do not keep tuning them. '
        'Keep distance as the requested simple state baseline; retain a half-random learned-error hybrid and its feasible '
        'adaptive variant. Pure learned-error thresholds and disagreement hybrids remain offline baselines; the live grid '
        'contains only the small pure-disagreement ablation to assess whether the predictor earns its complexity.',
        'R6’s earlier disagreement placement did not improve success over uniform at matched spend. '
        'The present proxies include a newer corrector and show usable concentration, but are weak evidence for causal SR benefit. '
        'Threshold fitting, signal comparison and arm selection reuse the B library; the curves are calibration evidence, '
        'not independent generalization estimates. No task-indexed layer-4 state is introduced. '
        'Keep the knob off by default when extra spend is not requested; B-policy traces cannot prove a positive spend utility.',
        '## Reproduction',
        'Run the command prefix in `RUN.md`, then the modules in its recorded order. `freeze.py` intentionally refuses '
        'to overwrite its timestamped prediction payload. Input hashes, nested episode splits, code hashes and read inventories '
        'are included beside the data and in the final manifest.']
    (HERE/'OFFLINE.md').write_text('\n\n'.join(off)+'\n')
    report=['# 给负责人的结论',
        '建议保留“距离太远就调用策略”作为简单对照，把“预测动作误差，再混入随机调用”作为主要候选。'
        '自动调阈值只保留在稀疏示范库、较高推理预算的设置。手爪即将切换不再单独立项。',
        f"在同样调用 {100*stats['matched_miss_fraction']:.0f}% 的状态时，误差预测选中的动作偏差是随机调用的 "
        f"{fmt_range(stats['error_lift_range'])} 倍；邻居分歧是 {fmt_range(stats['disagreement_lift_range'])} 倍。"
        f"扣除守卫已经接管的状态后，带随机成分的主要方案在相同总推理开销下，额外覆盖的偏差仍有随机方案的 "
        f"{fmt_range(stats['added_hybrid_lift_range'])} 倍。这里量的是动作差异，不能当作成功率收益。", 
        '此前的探索已经出现过“离线看起来更会挑状态，实际成功率却没有超过随机调用”的情况。'
        '因此这次预测不预支选点带来的成功率优势，只预期增加推理可能帮助弱缓存。'
        '大库默认保持关闭；没有证据证明额外调用一定有益。',
        f"自动调阈值在所选设置的离线偏移情景中，把最大预算偏差从 {stats['static_stress_max_error']:.3f} "
        f"降到 {stats['adaptive_stress_max_error']:.3f}。这不是实机误差保证；守卫强制调用过多时，"
        '任何不关闭守卫的旋钮都不能守住较低目标。较大示范库上的低预算自动调节已停止推进。',
        f"交回 {stats['arms']} 个冻结候选，覆盖 {stats['cells']} 个模型、任务套件和示范库大小组合。"
        '全程只用示范库拟合和标定，没有读取官方测试闭环输出，没有启动模型或评测。'
        '实现说明、逐候选参数、预算曲线和预先写下的预测均已保存。',
        '## Technical evidence and handoff',
        f"Prediction freeze: `{stats['prediction_timestamp']}`. The grid contains "+
        ', '.join(f"{count} {method}" for method,count in freeze['method_counts'].items())+'. '+
        'Same-batch knob-off controls and opus’s random/schedule arms are outside this count.',
        table(['Cell','Distance lift','Error-prediction lift','Disagreement lift'],[
            [a['cell']]+[f"{next(r['lift'] for r in ev if r['cell']==a['cell'] and r['method']==m):.3f}"
                        for m in ('distance','predicted_error','disagreement')] for a in analyses]),
        f"Lifts above use equal episode mass and a {100*stats['matched_miss_fraction']:.0f}% miss fraction. "
        f"The learned score retains {fmt_range(stats['error_rms_lift_range'])}× lift under an RMS proxy and "
        f"{fmt_range(stats['within_episode_lift_range'])}× against an episode-conditional random baseline. "
        f"Standalone long-task gripper-transition lift is only {fmt_range(stats['long_grip_lift_range'])}×. "
        'The predictor’s improvement over cheap disagreement is often modest; `OFFLINE.md` reports episode-bootstrap intervals.',
        'The pooled error head is strictly outer-episode-held-out for value evaluation and has no task ID or task-specific coefficients. '
        'Distance calibration refits the lower-layer metric without held-out episodes. The fixed selected-library PCA convention '
        'is inherited from the frozen corrector; reduced fold bank sizes and final-head transfer remain limitations. '
        'Pre-corrector motion MSE trains the score; corrected motion MSE evaluates it. Failed library episodes remain included.',
        f"The shared owner-IR model agrees with the independent cadence implementation to {stats['max_shared_difference']:.3g}. "
        f"Static simulation error is at most {stats['max_static_mc_error']:.4f}; adaptive held-out-seed target error in the "
        f"selected arms is at most {stats['max_adaptive_validation_error']:.4f}. These are checks on fixed B trajectories, "
        'not claims of live calibration. Final-head score transfer shifts the checked static IR by at most '
        f"{stats['final_head_transfer_max_error']:.4f} on those feature tables; this is a training-distribution diagnostic.",
        f"{stats['checks']} groups of independent checks passed: quantile ties/endpoints, full path enumeration, feedback and guard handling, "
        'retry/reset behavior, ownership/read boundaries, and out-of-support static endpoint behavior. '
        'The endpoint clarification after numerical freeze affects no proposed static arm; it is recorded in `POST_FREEZE_NOTES.md`.',
        'Read [SPEC.md](SPEC.md) for exact serving and calibration semantics, [OFFLINE.md](OFFLINE.md) for curves and value evidence, '
        '[ARM_GRID.md](ARM_GRID.md) and [arm_grid.csv](arm_grid.csv) for parameters, [PREDICTION.md](PREDICTION.md) for prospective '
        'IR and SR forecasts, and [HANDBACK.md](HANDBACK.md) for sol’s integration work and open issues.',
        'Every number in this report is emitted by the executed `write_docs.py` from the recorded analysis payloads. '
        'All SR numbers in the separate prediction document are explicitly labeled subjective forecasts generated by `freeze.py`; '
        'historical aggregate markdown supplies context only and never supplies fits or thresholds.']
    (HERE/'REPORT.md').write_text('\n\n'.join(report)+'\n')
    print(json.dumps(stats,indent=2))


if __name__=='__main__':
    main()
