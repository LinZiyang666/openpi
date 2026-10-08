"""All numerical report tables and explicit forecast arithmetic, from own runs."""
from .common import *
install()
import numpy as np

def read(p):return json.loads((HERE/p).read_text())
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'
def f(x,d=4):return f'{x:.{d}f}'
def summaries(category):
    return {p.parent.name:json.loads(p.read_text()) for p in sorted((HERE/category).glob('*/summary.json'))}

retr=summaries('retrieval');corr=summaries('correction');dyn=summaries('correction_dynamic');hist=summaries('history');reps=summaries('representations')
atlas=read('atlas_guard_assessment.json');unguard=read('atlas_assessment.json');dev=read('dev_cost.json');check=read('validation.json')
assert len(retr)==len(corr)==len(dyn)==len(atlas['cells'])==12
index={f'{a["cell"]}_{a["size"]}':a for a in atlas['cells']}
rows=[]
for k,x in retr.items():
    vs={v['variant']:v for v in x['variants']}
    rows.append([k,x['rows'],*[f(vs[n]['motion_mse']) for n in ('full','wrist','agent','wrist_l2','stale2','memory2','transport2')]])
rawtable=table(['cell / library size','rows','full','wrist','third-person','wrist L2','stale v0','refit memory','transport'],rows)
rows=[]
for k,x in dyn.items():
    vs={v['variant']:v for v in x['variants']};old={v['variant']:v for v in corr[k]['variants']}
    rows.append([k,f(vs['full']['motion_mse']),f(old['wrist']['motion_mse']),f(old['memory2']['motion_mse']),
                 f(vs['wrist']['motion_mse']),f(vs['memory2']['motion_mse']),f(vs['full_delta']['motion_mse'])])
headtable=table(['cell / size','full','wrist','memory','wrist + displacement','memory + displacement','full + displacement'],rows)
rows=[]
for a in atlas['cells']:
    rows.append([f'{a["cell"]}_{a["size"]}',f(100*a['half_share'],1),f(100*a['corrected_relative_delta'],2),
                 f(a['corrected_delta']),f'[{f(a["corrected_delta_ci"][0])}, {f(a["corrected_delta_ci"][1])}]',
                 f(100*a['gripper_disagreement'],2),f(100*a['missed_stall'],2),f(100*a['extra_stall'],2)])
atlastable=table(['cell / size','half looks %','corrected MSE change %','MSE delta','episode bootstrap interval','grip disagreement %','missed stall %','extra stall %'],rows)
rows=[]
for k,x in reps.items():
    vs={v['variant']:v['motion_mse'] for v in x['variants']}
    rows.append([k,*[f(vs[n]) if n in vs else 'unavailable' for n in ('mean','pool2','pca128','max','fine8','rgb','rgb_plus64')]])
reptable=table(['cell / size','token mean','2x2 pool','PCA128','token max','8x8 sketch','RGB','RGB + PCA64'],rows)

# Cost ceiling tables use reconciled B-dev trajectories, never A outcomes.
costrows=[]
for a in dev['arms']:
    if a['model']!='pi05' or a['size']=='current' or a['method'] not in ('off','periodic'):continue
    if a['root']=='r11_dev_size50':continue # paired devknob-off is the size50 row here
    costrows.append([a['arm'],f(a['owner_ir']),f(a['v']),f(a['m']),f(a['all_half_ceiling_stock']),
                     f(a['half_noncall_ceiling_stock']),f(a['eligible_share_looks']),f(a['refreshed_eligible']/a['V'])])
costtable=table(['B dev arm','baseline owner IR','V/N','M/N','all-half upper saving','noncall-only upper saving','legacy eligible/look','valid-tail eligible/look'],costrows)

assumptions=dict(timestamp=now(),status='subjective forecast, not a fitted SR model',
    loss_pp_pi05_l10={50:.3,200:.2,500:.2},loss_pp_pi05_spatial={50:.2,200:.1,500:.1},
    loss_pp_groot_l10={50:.5,200:.3,500:.3},loss_pp_groot_spatial={50:.3,200:.2,500:.2},
    sr_delta_interval_pp=[-2.,1.],buyback_slopes_pp_per_ir=[25.,10.],
    missing_dev_v=.5,missing_dev_eligible_fraction_off=.4,missing_dev_eligible_fraction_periodic=.35,
    missing_dev_half_miss_fraction=.1,half_exposure_scale_uncertainty=[.5,1.25],
    hypothetical_groot_half=.074,hypothetical_groot_completion=.074,
    measured_pi05=dict(stock_full=.152,request_full=.1656,half=.0646,completion=.0502))
dump(HERE/'forecast_assumptions.json',assumptions)
forecast=[];frows=[]
for a in atlas['cells']:
    model,suite=a['cell'].split('_');n=a['size'];cv=.152 if model=='pi05' else .148
    for knob in ('off','periodic32'):
        method='off' if knob=='off' else 'periodic'
        choices=[d for d in dev['arms'] if d['model']==model and d['suite']==suite and d['size']==n and d['method']==method]
        choices.sort(key=lambda x:x['root']!='r11_devknob_50')
        d=choices[0] if choices else None
        v=d['v'] if d else .5
        m=d['m'] if d else ((.32-cv*v)/(1-cv) if knob=='periodic32' else .1)
        eligibility=d['refreshed_eligible']/d['V'] if d else (.4 if knob=='off' else .35)
        phmiss=d['refreshed_eligible_calls']/max(d['refreshed_eligible'],1) if d else .1
        h=min(a['half_share'],2*a['half_share']*eligibility)
        H=v*h;C=H*phmiss
        hf,co=(.0646,.0502) if model=='pi05' else (.074,.074)
        ff=.1656 if model=='pi05' else .148
        saving_request=(cv-ff)*v+(ff-hf)*H-co*C
        saving_stock=(cv-hf)*H-co*C
        loss=assumptions[f'loss_pp_{model}_{suite}'][n]
        record=dict(cell=a['cell'],size=n,knob=knob,baseline_dev_arm=d['arm'] if d else None,
            conditional_groot_price=model=='groot',v=v,m=m,eligible_share=eligibility,half_share=h,half_miss_fraction=phmiss,
            saving_current_path=saving_request,saving_stock_full=saving_stock,predicted_SR_delta_pp=-loss,
            SR_delta_range_pp=assumptions['sr_delta_interval_pp'],
            matched_SR_saving_current_at25=saving_request-loss/25,
            matched_SR_saving_stock_at25=saving_stock-loss/25,
            matched_SR_saving_current_at10=saving_request-loss/10,
            required_slope_current_pp_per_ir=loss/saving_request if saving_request>0 else None,
            all_half_ceiling=(cv-hf)*v-co*m,
            all_half_no_policy_ceiling=(cv-hf)*.5,
            alternating_no_policy_ceiling=(cv-hf)*.25)
        forecast.append(record)
        frows.append([f'{a["cell"]}_{n}',knob,f(100*h,1),f(saving_request),f(saving_stock),f(-loss,1),
                      f(record['matched_SR_saving_current_at25']),f(record['matched_SR_saving_stock_at25'])])
assert sum(x['baseline_dev_arm'] is not None for x in forecast)==12, 'expected B-dev baselines must actually join'
dump(HERE/'predictions.json',dict(timestamp=assumptions['timestamp'],assumptions=assumptions,cells=forecast))
forecasttable=table(['cell / size','knob','half/look %','IR saved current path','IR saved stock-full path','SR delta pp','matched saving current @25 pp/IR','matched saving stock @25 pp/IR'],frows)
ceilrows=[]
for cell in sorted(set((x['cell'],x['size']) for x in forecast)):
    off=next(x for x in forecast if (x['cell'],x['size'])==cell and x['knob']=='off')
    on=next(x for x in forecast if (x['cell'],x['size'])==cell and x['knob']=='periodic32')
    unit=.0874 if cell[0].startswith('pi05') else .074
    ceilrows.append([f'{cell[0]}_{cell[1]}',f(off['all_half_no_policy_ceiling']),f(off['alternating_no_policy_ceiling']),
                    f(off['all_half_ceiling']),f(on['all_half_ceiling']),
                    f(unit*min(off['v']/2,off['v']-off['m'])),f(unit*min(on['v']/2,on['v']-on['m']))])
ceiltable=table(['cell / size','no-call all-half','no-call alternating','all-half with guards','all-half periodic32','primary upper off','primary upper periodic32'],ceilrows)

prediction=f'''# Pre-pilot prediction

Frozen UTC timestamp: **{assumptions['timestamp']}**. This precedes any closed-loop result of this design. Only B library replay and old, explicitly allowed B-dev baselines were inspected. No changed-controller SR has been measured.

Positive saving means lower total IR. The central SR changes below are **subjective forecasts**, not outputs inferred from action MSE. Every central SR forecast has the broad plausibility interval **[{assumptions['sr_delta_interval_pp'][0]:g}, {assumptions['sr_delta_interval_pp'][1]:g}] pp**. These are not confidence intervals. The illustrative buyback slope is **25 pp per unit IR**; at **10 pp per unit IR** the buyback costs are larger (`predictions.json`). Existing B dev cannot identify reliable local slopes for every cell.

{forecasttable}

GR00T rows are **conditional arithmetic only**, assuming half/completion each cost {assumptions['hypothetical_groot_half']:.3f}; neither price nor an actual one-camera path is established. Do not call those deployable savings. Pi05 current-path columns use measured request-full/wrist/completion prices; stock-full columns require a separately measured dispatch improvement applied fairly to controls.

Exposure is the B-library safety-atlas half fraction, attenuated by valid-tail B-dev eligibility relative to its ideal alternating limit. Rows without matching B-dev observations use the explicit scenario constants in `forecast_assumptions.json`, not invented closed-loop measurements. Baseline half-MISS probabilities are transported only as a forecast; guard feedback can invalidate them. The stored SR loss priors do not vary with knob setting because no evidence supports a more precise interaction model.

My directional prediction: this is most likely a small raw-IR saving on pi05; **matched-SR savings on the existing request path are uncertain and several central scenarios lose**. Recovering stock-speed full dispatch improves the economics. GR00T is not ready for a cost claim. Small offline MSE changes do not justify predicting lossless SR with confidence.

The exact break-even condition is `IR_saved_raw > SR_loss / local_buyback_slope`. If the slope is zero or unavailable, goal (b) is not established. Goal (a) requires a paired closed-loop noninferiority result with uncertainty, not rounded equal SR.

Maximum pi05 savings without calls, at ideal look rate 0.5: all half {(.152-.0646)*.5:.4f}; this design's alternating limit {(.152-.0646)*.25:.4f} with stock-speed full looks, or {(.152-.1656)*.5+(.1656-.0646)*.25:.4f} with current request-full pricing. All-policy/all-half theoretical saving is {(.152-.0646-.0502)*.5:.4f}, but the primary deliberately chooses full on known periodic calls. Layer-4 calls therefore shrink the attainable cache-only opportunity.

Per-cell optimistic ceilings below retain baseline call rates and stock-speed full looks; they ignore warm-up, safety rejection and guard changes. “Primary upper” is `(full-half)*min(v/2, v-m)`, assuming every half avoids a policy call. B-dev rates are used where a matching baseline exists; the other rows are explicit scenarios as above. GR00T remains conditional on hypothetical prices.

{ceiltable}

No thresholds or cell choices may be revised from A outcomes. Fine settings and go/no-go decisions belong to the B pilot in EXPERIMENTS.md.
'''
(HERE/'PREDICTION.md').write_text(prediction)

offline=f'''# Offline evidence

Generated UTC: {now()}. All numerical tables were produced by this directory's executed code. Evidence is B-library or allowlisted B-dev only. These are action/retrieval/ledger diagnostics, not measured SR of look-half.

## Data and validation

Four model/suite cells, nested selected libraries of 50, 200 and 500 episodes. All failed library episodes remain. Five episode folds exclude held-out episodes from action sigma, metrics, donor sets, correction-head training and atlas-label fitting. The existing R10 PCA files are verified against selected-row hashes and remain fixed across folds; this is transductive unsupervised B-library representation, not fully held-out PCA. The alternative representations use the same convention. No test-A recordings, queries store, or other R12 explorer data were accessed.

Retrieval screens use even query steps at least two, modelling ten-control commits after hits; they do not simulate policy-induced state changes. Atlas screens begin at step four and discard the initial look pair; deployment exposure is consequently smaller. Equal episode weights prevent failed/long episodes dominating row means. State/action values and library successes are observations, not official-test labels.

Correction diagnostics refit heads inside each outer fold, with LOEO donor synthesis inside its training library. They use noninitial training rows and training-LOEO distance scales. Production R10 heads include their initial-row branch and five-fold distance calibration. Thus these are controlled integration diagnostics, not exact end-to-end R10 replay. The specification calls for the exact production calibration before serving.

Validation checked **{check['retrieval_rows_checked']:,}** donor-row cases for task and episode-fold exclusion, finite actions and normalized weights. Main full metric distances differ from the R10 equation by at most **{check['max_R10_distance_abs_difference']:.9g}**. **{check['planner_checks']}** pure planner checks passed. The planner arithmetic averaged **{check['planner_mean_cpu_seconds']*1e6:.1f} microseconds** in a local CPU loop; this is not end-to-end retrieval latency. CPU affinity and read audit are in validation.json. GPU was not used because its driver was unavailable.

Camera mapping was checked in library metadata and source: pi05 img0 is base_0_rgb and img1 is left_wrist_0_rgb; GR00T's LIBERO builder orders video.image then video.wrist_image. R7 replay explicitly poisoned key_v0 on wrist requests. The actual current pi05 deferred encoder uses a zero placeholder until completion, which is another reason availability must be tracked explicitly rather than inferred from finite arrays.

## Retrieval and representation screen

Values are episode-balanced normalized motion MSE over ten controls and six channels. Lower is better. “Stale v0” simply substitutes a two-slot-old key into the current full metric. “Refit memory” fits the correct delayed representation; “transport” adds a neighbour-predicted v0 innovation before full-metric retrieval.

{rawtable}

Unconditional wrist increases error in every screened cell. Pure third-person search is usually worse and its price is unmeasured. Naive stale-key substitution is especially poor; age-four substitution is worse still. A Schur-complement distance, minimizing the full metric over the missing view, nearly reproduces the independently refitted wrist metric. Both use the same action-pair covariance; marginalizing does not restore missing camera information. Euclidean wrist distance does not rescue it.

The useful direction is treating camera memory as a different representation and including observed displacement. Initial-only camera memory and longer refresh gaps were screened in history/*/summary.json; they did not justify dropping the short age contract. Raw state history expressed as a displacement was more useful than appending another absolute-state block. This is an empirical regularization effect, not additional physical information.

Alternative single-wrist features on size-50 libraries:

{reptable}

The 8x8 representation mean-pools tokens spatially after a fixed signed channel sketch; it retains finer spatial information than the original 4x4 keys. It helps the small GR00T spatial cell in this screen, but does not win consistently and has no measured one-camera deployment price. RGB features use downsampled wrist pixels; they fail to improve the main candidates. Larger PCA rank also does not consistently help. These representations are not selected for the primary.

## Correction and the fair full-information control

{headtable}

The half head must match its actual input. Copying old camera data into the original correction head is worse than fitting a matching head (`correction/*/summary.json`). Displacement also improves full vision: the final column is the necessary control for any claimed representational improvement. Memory-plus-displacement can approach or beat the old full cache on some cells while remaining worse than full vision with the same motion information. That is why the experiment grid includes both controls.

## Direct camera-safety segmentation

The primary atlas is trained from raw full/half action agreement, gripper agreement and progress-comparison agreement. At runtime the previous full donor mixture predicts safety two slots ahead; current state displacement checks whether the robot followed that prediction. No current image is used to choose the camera. Fixed label RMS .15, safe mass .75, support .99 and state residual .2 are B-screened choices, not test-A fits. `atlas_reference.py` contains the pure prelook function; SPEC.md gives the complete production recipe.

The following table applies the frozen atlas causally along held-out B policy trajectories, forcing a full look after each half. It substitutes the separately fitted corrected half action into the unchanged corrected-full branch. The bootstrap interval resamples whole episodes and is conditional on the fitted models; it is not an SR interval or a correction for selecting the proposal on B data.

This replay includes the camera-score and camera-age rules only. It does **not** simulate the runtime pending-no-progress veto, periodic-due veto, policy-tail lifecycle, new guard calls or changed trajectories. Thus the half fraction is an offline exposure opportunity, not a forecast of the complete serving state machine. Gripper disagreement averages sign mismatches over the ten controls, including zero mismatches on full looks; it is not the fraction of episodes harmed.

{atlastable}

“Missed stall” means the half top donor advances while the full top donor does not, relative to the previous full donor; “extra stall” is the converse. These are single-comparison diagnostics, not measured additional policy calls. Adding progress agreement to the atlas labels reduces many such errors versus action-only labels, but does not remove them. **The no-progress guard is the main unresolved system risk.**

The after-wrist gate in gates_dynamic.json was also explored. It has access to the current wrist proposal and can abstain based on corrected-action/progress disagreement, but needs a new transactional second-camera fallback. It remains secondary. Its measured feature gate must not be described as an existing plugin capability.

## Owner IR accounting and B dev limits

Let N be requested five-control slots, V all looks, H actual half looks, C half-to-policy completions, and M policy calls. At fixed trajectories/calls, pi05 current request-path savings against stock full vision are:

`S = (0.152-0.1656)*V/N + (0.1656-0.0646)*H/N - 0.0502*C/N`.

If full looks truly retain stock speed, `S = 0.0874*H/N - 0.0502*C/N`. Additional calls cost at least their policy computation; recompute the complete ledger rather than subtracting this fixed-trajectory formula after behavior changes. With no completions the current path needs half share above **{.0136/.101:.4f}** just to cover full-look dispatch overhead. Prices are supplied local-4090 proportional-latency measurements, not hardware-independent FLOPs.

The all-half ceiling on a baseline with calls is `.0874*v - .0502*m`. Restricting half to no-call looks gives the more operational ceiling `.0874*(v-m)`. These ceilings ignore warm-up and safety rejection and cannot be read as achievable controller savings.

{costtable}

All selected B-dev journals include both accepted successes and accepted failed episodes. Their N/V/M counts reconcile exactly with summaries. Owner-basis IR was recomputed; some stored stage-price IR fields use a different basis. Existing dev recordings lack query keys, so they support cost/opportunity bounds but cannot evaluate this camera policy on cache-driven states.

The legacy camera override discards many opportunities after a policy tail. A narrow, explicitly validated camera-memory path recovers some without altering the tail action lifecycle; blanket waiver of lifecycle errors is prohibited. Prediction scenarios use the new valid-tail column. No size-500 B-dev baseline exists in the allowlisted roots; its forecasts are stated scenarios. GR00T half/completion prices and a one-camera path remain unmeasured/unvalidated.

## Reproduction and scope

All commands run from repository root with:

```sh
taskset -c 10-21,54-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= .venv/bin/python
```

Append `-m exp.offline_search.rounds.r12.astra.<module>` for inventory, retrieval, correction, representations, history, atlas, dev_cost, assess_atlas, check, make_reports. Also run `correction --dynamic`, `gates`, `gates --dynamic`, and `assess_atlas --guard`. `representations` defaults to size 50. Run independent cell fits with their bounded process pools; all BLAS stays single-threaded inside the assigned CPU set. No server/worker/sync/test run is launched by any of these scripts.

`make_reports` regenerates numerical prose and predictions; it must not be rerun after receiving design closed-loop outcomes as if that were the original preregistration. PREDICTION.md is frozen by FINAL_MANIFEST.json. Curated current libraries were inventoried, not analysed with these candidate fits. Complete closed-loop requests, pilot/grid, unresolved questions and implementation work are in REQUESTS.md, EXPERIMENTS.md, SPEC.md and HANDBACK.md.

`fit_selected_atlas` builds the final selected-library safety tables and main metric references under final_atlas/. These are library-only ingredients for sol, not deployable full-system artifacts. [Forecast sensitivity figure](forecast.png) visualizes explicitly assumed buyback economics; it is not an experimental SR plot.
'''
(HERE/'OFFLINE.md').write_text(offline)

pi=[x for x in atlas['cells'] if x['cell'].startswith('pi05')]
report=f'''# 给负责人的说明

我建议先试一种有短期记忆的单相机检索：当前只编码腕部相机，保留最近一次完整观察中的外部相机信息，再加入这段时间机器人真实移动了多少。是否少看一个相机，由示范库里直接学出的“少看是否安全”来决定，不按抓取阶段划分。

离线结果有希望，但还不能说成功率不掉。直接把旧相机特征塞进原来的检索器，误差明显变大；重新训练适配这种输入的检索度量和动作修正器，效果好得多。在六个 π0.5 库大小与任务组合中，安全门选择了约 **{min(x['half_share'] for x in pi)*100:.1f}%–{max(x['half_share'] for x in pi)*100:.1f}%** 的可评估观察只用腕部相机。最终动作误差相对完整相机的变化为 **{min(x['corrected_relative_delta'] for x in pi)*100:+.2f}% 到 {max(x['corrected_relative_delta'] for x in pi)*100:+.2f}%**。这些数字来自示范库回放，不能换算成闭环成功率。

最需要小跑检查的是守卫。动作很接近时，检索到的示范进度仍可能不同，导致本该调用大模型时没有调用。把进度一致性加入安全标签后有所改善，风险仍在。早期的数据请求已写好；现有开发集日志没有相机特征，无法补出这个答案。

开销空间也不大：理想情况下每隔一个控制时隙观察一次，全部改成单相机最多省 **{(.152-.0646)*.5:.4f}** 的总推理比例；本方案要求单相机之后恢复完整观察，因此上限再减半。当前按请求切相机的完整观察路径还有额外开销。成功率只要稍微下降，补回它所需的策略调用就可能吃掉节省。因此我的预测是“原始视觉成本可能下降，匹配成功率后的总成本优势尚未确定”，而不是已经达标。

另一模型只有离线特征结果，尚没有验证过的真正单相机执行路径，也没有相应价格。不能把它列为已经省钱的方案。另外，加入机器人位移也能改善完整相机基线，实验必须把这个对照带上。

建议先做开发集配对小跑，核对守卫时机、真实相机成本和补调用成本；通过后再冻结官方测试格子。完整设计、预测和实验计划都已交接。

# Technical handback

Primary: pre-encoding, progress-aware safety atlas plus two-slot camera memory and observed displacement. Library-only task metric/head fits belong to layer 5; gate thresholds are task-agnostic. Prototype reference and fit/evidence code are included, but no deployable plugin is claimed.

Evidence covers the four cells at nested sizes 50/200/500, alternative token/image features on size 50, correction compatibility, fair full-plus-displacement controls, causal atlas replay, and accepted B-dev cost reconciliation. No R12 other-explorer data or A recordings were read. No launches, GPU work, or writes outside this directory occurred in the analysis.

Primary numerical evidence:

{atlastable}

OFFLINE.md documents the fixed-PCA and correction-calibration approximations. PREDICTION.md gives per-cell raw/conditional matched-SR forecasts with explicit assumptions; it does not manufacture closed-loop SR evidence. SPEC.md is implementation-ready for sol, EXPERIMENTS.md freezes the B-first workflow, and REQUESTS.md lists the missing recordings/cost measurements.
'''
(HERE/'REPORT.md').write_text(report)
dump(HERE/'report_generation.json',dict(timestamp=now(),tables=dict(raw=len(retr),correction=len(dyn),atlas=len(atlas['cells']),representations=len(reps),forecast=len(forecast)),read_sets=len(dev['arms'])))
print(json.dumps(dict(generated=['OFFLINE.md','PREDICTION.md','REPORT.md'],forecast_rows=len(forecast),timestamp=assumptions['timestamp'])))
