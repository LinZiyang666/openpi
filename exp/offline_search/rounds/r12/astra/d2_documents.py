"""Write D2 handback from executed evidence; append prediction exactly once."""
from .d2_screen import *

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'

def label(r):
    model,suite=r['cell'].split('_');return f"{'π0.5' if model=='pi05' else 'GR00T'} {suite}-{r['size']} / {'periodic .32' if r['periodic_setting'] is not None else 'off'}"

def main():
    E=json.loads((D2/'evidence.json').read_text());V=json.loads((D2/'validation.json').read_text());G=json.loads((D2/'groot_mismatch'/'summary.json').read_text());stamp=now()
    revision='--append-revision' in sys.argv
    pred_before=(HERE/'PREDICTION.md').read_bytes()
    assert pred_before.startswith((D2/'original_prediction.md').read_bytes())
    if pred_before!=(D2/'original_prediction.md').read_bytes():
        assert revision and (D2/'source_correction.json').exists(),'Existing prediction requires explicit append-only revision'
    if revision:
        (D2/'prediction_before_local_revision.md').write_bytes(pred_before)
    archive=D2/'prior_delivery';archive.mkdir(exist_ok=True)
    for name in ('REPORT','SPEC','OFFLINE','EXPERIMENTS','HANDBACK'):
        target=archive/f'{name}.md'
        if not target.exists():target.write_bytes((HERE/f'{name}.md').read_bytes())
    mainname='consensus2_r0.1';strictname='consensus2_r0.05'
    core=table(['cell / knob','N / looks / calls','baseline SR / IR','half looks','fixed-call IR saved','missed / extra calls','grip-disagreement looks'],[
        [label(r),f"{r['N']} / {r['V']} / {r['M']}",f"{r['baseline_sr']:.3f} / {r['baseline_ir']:.6f}",r['methods'][mainname]['H'],f"{r['methods'][mainname]['saving_fixed_calls']:.6f}",
         f"{r['methods'][mainname]['missed']} / {r['methods'][mainname]['extra']}",r['methods'][mainname]['gripper_disagree_half']] for r in E['cells']])
    old=table(['cell / knob','frozen atlas half looks','IR saved, new price','missed / extra calls','grip-disagreement looks'],[
        [label(r),r['methods']['frozen_atlas']['H'],f"{r['methods']['frozen_atlas']['saving_fixed_calls']:.6f}",f"{r['methods']['frozen_atlas']['missed']} / {r['methods']['frozen_atlas']['extra']}",r['methods']['frozen_atlas']['gripper_disagree_half']] for r in E['cells']])
    strict=table(['cell / knob','strict half looks','IR saved','missed / extra / grip'],[[label(r),r['methods'][strictname]['H'],f"{r['methods'][strictname]['saving_fixed_calls']:.6f}",
        f"{r['methods'][strictname]['missed']} / {r['methods'][strictname]['extra']} / {r['methods'][strictname]['gripper_disagree_half']}"] for r in E['cells']])
    costs=table(['cell / knob','all-half ceiling','alternating ceiling'],[[label(r),f"{r['allhalf_ceiling']:.6f}",f"{r['alternating_ceiling']:.6f}"] for r in E['cells']])
    uncertainty=table(['cell / knob','fixed-call saving 95% interval','saving charging extra calls, no missed-call credit','calibration H / missed / extra / grip','evaluation H / missed / extra / grip'],[
        [label(r),'['+', '.join(f'{x:.6f}' for x in r['methods'][mainname]['saving_ci']['ci95'])+']',f"{r['methods'][mainname]['extra_calls_only_penalized_saving']:.6f}",
         ' / '.join(str(r['methods'][mainname]['calibration'][k]) for k in ('H','missed','extra','gripper_disagree_half')),
         ' / '.join(str(r['methods'][mainname]['evaluation'][k]) for k in ('H','missed','extra','gripper_disagree_half'))] for r in E['cells']])
    numeric=[]
    for r in E['cells']:
        if 'numeric_stress' not in r:continue
        n=r['numeric_stress'];m=n['results'][0]['result']
        numeric.append([label(r),n['eligible'],n['top_changed'],m['H'],f"{m['saving_fixed_calls']:.6f}",f"{m['missed']} / {m['extra']} / {m['gripper_disagree_half']}"])
    numtable=table(['GR00T L10 cell','proposals stressed','top changed under stress','robust half looks','IR saved','missed / extra / grip'],numeric)
    paired=table(['model','common B pairs / tasks','off SR → knob SR','off IR → knob IR','SR change 95% interval, pp','chord pp/IR [95%]'],[
        [r['cell'],f"{r['paired']} / {r['tasks_covered']}",f"{r['off_sr']:.3f} → {r['knob_sr']:.3f}",f"{r['off_ir']:.6f} → {r['knob_ir']:.6f}",str([round(x,2) for x in r['sr_delta_ci95_pp']]),
         f"{r['chord_pp_per_ir']:.2f} [{r['chord_ci95'][0]:.2f}, {r['chord_ci95'][1]:.2f}]"] for r in E['paired_knob']])
    pi=[r['methods'][mainname]['saving_fixed_calls'] for r in E['cells'] if r['cell'].startswith('pi05')]
    report=f'''# 给负责人的说明

按新的算力计价，两种模型只看一个相机都能省下一半视觉开销。此前把 GR00T 因为单请求延迟而排除的判断作废。但这轮还没有证据证明“成功率不掉且总开销下降”，也没有证明额外调用能划算地补回成功率。

我重新检查了 {V['episodes']} 条 B 开发集轨迹、{V['looks']:,} 次观察，包含两种模型的小库、大库和刚完成的定期调用记录。它们都是原来的完整双相机控制器产生的轨迹；新方法只做离线影子判断，没有驱动机器人。旧方案的问题已经很清楚：动作看起来接近，并不保证检索进度相同，守卫会因此漏掉本来要做的调用。不能把漏调用带来的开销下降当成收益。

这里的统计只使用协调者当前认可的录制。空间任务采用本机重新录制的数据；被撤回的旧空间任务和旧中等库录制不再作为证据。预测文件保留旧段落，并追加了撤回说明和新的有效数据预测。

现在建议的小跑方案是：先看腕部相机，用最近一次完整观察和机器人位移估计另一相机；同时做一个只依赖当前腕部信息的独立检索。只有两个估计与拟执行动作、夹爪和进度都一致，才接受少看一半，否则补看另一相机并回到完整检索。这样直接检查当前观察，比只按上一次检索结果预判安全更可靠。所有检索度量和修正器仍只用所选 B 示范库拟合；新检查规则和阈值的选择明确属于冻结后的开发集调整。

在现有固定轨迹上，π0.5 的主要方案可省总 IR {min(pi):.6f}–{max(pi):.6f}。它大幅减少了守卫差异，但部分格子仍会漏调用。更严格的版本在这些记录中没有出现漏调用、额外调用或夹爪符号差异，代价是省下的量更小；这仍不能证明闭环成功率不掉。

GR00T 的问题现在是准确性和数值路径，而不是定价。我实际比较了 {G['count']} 组 B 开发图像重建后的联合编码与单相机编码。特征差异确实存在；进一步要求检索对这些差异稳定后，可接受的半观察更少。录下来的小图不能还原原始输入，所以这只是数值敏感性证据。补看另一相机后，分别编码得到的策略输入也不保证等于原来的联合编码，正式小跑必须把这一点核实。

定期调用记录虽然齐了，但和关闭调用的记录只重合 {E['paired_knob'][0]['paired']} 个初始状态，不能据此确定可靠的局部“多花多少开销能补回多少成功率”。因此建议先做严格配对的 B 小跑，优先检查 π0.5 的空间任务；GR00T 先通过真实单相机输入的数值检查。只有测到相同成功率下的总 IR 优势，才进入正式测试。

# Technical handback

Updated UTC: {stamp}. DIRECTIVE_2 supersedes the prior latency economics. Primary pilot: wrist-first two-witness completion gate (`consensus2_r0.1`); strict diagnostic: the same gate at `.05`. The frozen pre-look atlas remains a rejected baseline, not a deployment recommendation. Camera choice, broad missing-view ensembles, and numerical sensitivity were also evaluated.

The table is fixed-state shadow evidence. The SR column belongs to the **full baseline**. Missed/extra calls are propagated fixed-schedule diagnostics, not actual new-controller outcomes. GR00T rows use joint-path recording keys and require the numerical-path qualification described above.

{core}

The complete fit and acquisition contract is in [SPEC.md](SPEC.md), uncertainty and rejected alternatives in [OFFLINE.md](OFFLINE.md), new timestamped scenarios in [PREDICTION.md](PREDICTION.md), and the B-first controller plan in [EXPERIMENTS.md](EXPERIMENTS.md). Original delivery files are retained in `directive2/prior_delivery/`; the original prediction bytes remain untouched at the start of PREDICTION.md. No test-A evidence, other explorer's R12 files, launches, or external writes were used.
'''
    (HERE/'REPORT.md').write_text(report)
    spec='''# DIRECTIVE_2 implementation specification

Status: implementation-ready offline reference and B-pilot proposal, not a deployed plugin or established matched-SR saving. This document supersedes the old primary and its latency accounting. The old specification is preserved in `directive2/prior_delivery/SPEC.md`.

## Method and fit boundary

Use the unchanged full R10 recipe, only-no-progress guard, corrected ten-control commit, and off/periodic R11 knob. The primary is a **wrist-first completion gate with two independently constructed action/progress witnesses**. Every planned look still obtains current vision; there is no extra blind block. All PCA, task metrics, action/state normalization, correction heads and distance scales are fitted from exactly the selected B library, including failed episodes. Task-indexed fits belong to layer 5. Camera gate thresholds are global across tasks, models and sizes.

The reference is `d2_consensus.candidate` plus `d2_screen.gate_mask` and the state-machine equations in `d2_screen.replay_gate`. The replay is an analysis of fixed recorded states; its reads of the recorded full answer are fallback/evaluation inputs, not a substitute for actual acquisition in a serving implementation. A plugin adapter still has to implement the transaction below.

### Library-only ingredients

Keep the selected-library 64-coordinate PCA for each camera and the unchanged full metric/head. Let `p(i)` be the library predecessor exactly two five-control slots earlier in the same episode, clamped at the initial row. Define:

```
F(i) = [P0(i), P1(i), rs(i)[:8]]
H(i) = [P0(p(i)), P1(i), rs(i)[:8], rs(i)[:8] - rs(p(i))[:8]]
W(i) = [P1(i), rs(i)[:8], rs(i)[:8] - rs(p(i))[:8]]
```

Fit independent action-weighted Mahalanobis metrics for H and W per task, using the existing nearest-three other-episode action construction, first five controls/seven standardized action channels, ridge `0.1*trace(Sw)/d`, full-rank Cholesky, stable top-16 retrieval and the original kernel (kref 5 at size 50; 8 otherwise). Candidate payloads retain their current actions/progress and parent row mapping. The last real P0 is memory, not a fresh observation.

Fit H correction heads with selected-library LOEO synthesis, query steps at least two, residual targets on ten controls/six motion channels in selected-library sigma units. Features are H, standardized synthesized ten-control/seven-channel action, `min(step,120)/120`, and the task indicator. Use R10 episode-balanced weights, 384 RFF, seed zero, ridge alpha 100. Leave gripper/padding unchanged. Fit H distance scale with the five-episode-fold selected-library procedure, training-fold metric/sigma, and episode-balanced median noninitial held-out nearest distance. Correction is `.5 * clip((2-d/scale)/1.25,0,1)` times the residual in original action units. `dev_validation.fit_half` implements this scale and head completion; its artifacts now exist for every recorded cell.

Fit W heads identically, with W inputs. W is only a disagreement sensor, never the served action. Its scale is the selected-library training-LOEO median, as implemented in `d2_consensus.fit_wrist`. This sensor-specific calibration is intentional; it is not a claim of production five-fold W correction parity. Changing it requires a fresh B screen.

Retain the original library safety atlas: LOEO raw F/H synthesis at i and raw F at p(i); require standardized motion RMS ≤.15, gripper-sign identity on every committed control, and agreement of the progress-advancement Boolean. Progress advancement is `(p_now-p_previous)*max(ep_len_now-1,1) > .5`. Early rows before step four are unsafe. Save exact same-episode two-slot successors and selected-library state standard deviation, floored at `1e-6`. `final_atlas/*/table.npz` contains these tables and main metric references. Fixed selected-library PCA is transductive across library folds; no B-dev observations enter these fits.

### Before encoding

Maintain connection-private memory of the most recent **genuine full** look: both projected keys, state, step, actual full donor rows/weights, task/episode identity and accepted execution metadata. Reset on identity changes, discontinuity, failed execution, retry ambiguity or missing fields. Do not renew this memory on an accepted half look.

Force full on step <4, memory age other than exactly two slots, a positive previous no-progress span, a known periodic due call, or invalid lifecycle. The narrow valid-policy-tail exception requires the prior full policy look exactly two slots earlier, one successfully executed policy-tail block, both exact five-control execution acknowledgments, no pending span, and unchanged identity. Do not waive generic lifecycle errors.

Advance saved full donors by two library rows. Require successor support mass ≥.99, safe successor mass ≥.75, and RMS error between measured state displacement and weighted expected donor displacement ≤.2 selected-library state-SD units. Missing successors contribute zero support; do not renormalize them into safety. Rejections here choose full immediately. These checks consume no current image features.

For periodic calls, compute the cap from accepted history **before acquisition**: last actual call step, count of look anchors since that call, the saved R11 setting, and `uniform('R11-gap-v1',0,task,init,last_call,'cap')`. Use floor plus the fractional Bernoulli cap. Do not use the recorded fired flag, which is suppressed when the guard already calls. Peeking must not consume or mutate the scheduler. Distance-threshold knobs remain outside this first pilot because full/H distances need separate calibration maps.

### After encoding the wrist, before any commit

1. Construct H from the saved genuine P0, current wrist, current state and measured displacement. Retrieve/correct the proposed half action. Its top row is the proposed guard-progress source.
2. Retrieve/correct W from current wrist, state and displacement. This is the first witness.
3. For H donors `r_k` and weights `w_k`, estimate the missing-view innovation:

```
delta0 = sum_k w_k * (P0(r_k) - P0(p(r_k)))
P0_est = P0_saved + delta0
```

Retrieve with the unchanged full metric on `[P0_est,P1_now,rs_now]` and apply the unchanged full head to these explicitly imputed features. This is the second witness. It is an estimate used only for acceptance; it does not become a true full observation or overwrite the camera memory.
4. Require both corrected witnesses to be within standardized motion RMS `.1` of the proposed corrected H action and to match its gripper sign on every committed control. The strict alternative uses `.05`; everything else stays identical.
5. Require each witness top donor's normalized progress to be within `.5` donor steps of the H top donor: `abs(p_H-p_W)*max(len_H-1,len_W-1,1) <= .5`. Require H to advance more than `.5` donor steps relative to the saved full top donor. Do not substitute episode success, true environment progress or the hidden current view.
6. If any requirement fails, acquire the missing current camera, discard provisional proposals without mutating guard/knob/anchor state, and run ordinary full retrieval/correction once. If all pass, commit H once, then run the ordinary guard and knob once from that chosen donor history. If a policy call is required, complete current vision before the policy and charge completion once.

Keep the selected corrected action as the authoritative cache anchor and preserve its committed tail. Accepted half does not refresh full-camera memory, so the next look is full. On a half-to-policy completion, invalidate donor-planning memory unless a separate stateless genuine-full retrieval has actually been performed; never install H donors as full donors. Preserve mode/availability bits separately from all-or-nothing `has_vision`, and preserve idempotence on duplicate requests.

The analysis computes extra individual-donor innovation witnesses to screen wider ensembles. The selected method uses only W and the weighted-mean innovation witness. Computing only those witnesses is an implementation optimization and earns no additional owner IR saving.

### GR00T numerical path

The same compute prices and candidate method apply to GR00T. Its recorded joint-path key is not automatically a genuine single-camera key. `d2_groot_mismatch.py` uses the specified workspace Python interpreter and the installed deployed GR00T torch/FA2 packages together, loads only the selected checkpoint vision tower/projector, and compares batch-two versus batch-one on identical reconstructed B pixels under bf16 autocast. This measures numerical sensitivity, not original-pixel re-encoding of the library.

An optional numerical-stability filter is implemented by `d2_stress.py`: take the eight largest projected perturbation norms from the measured B-dev sample, use both signs, and require the two-witness gate, H top donor and gripper signs to stay unchanged under all perturbations, with corrected-H motion RMS change ≤.025. This perturbation set and tolerance are **post-freeze B-dev calibration**. They are not a certified bound, and the supplied set is for the L10 checkpoint only.

Before real GR00T serving, obtain exact original/preprocessed B pixels and either re-encode selected-library single-view features or validate a robust representation on true single-camera keys. Validate the fallback/completion prefix too: two separately encoded views need not equal a joint encoding. A full-information separate-camera control must use the same fit/numerical path and the same total compute-share price. Do not silently replay the original full result after a numerically different completion or charge a redundant joint re-encode as if it were free. The required evidence is in REQUESTS.md; no numerical-parity or policy-parity claim is made here.

### Accounting

Let N be requested five-control slots; F full looks; H accepted half looks; C accepted half looks subsequently completed; M policy calls. A rejected partial look plus completion counts in F, not H. `V=F+H`.

```
pi05 work = .152*F + .076*H + .076*C + .848*M
groot work = .148*F + .074*H + .074*C + .852*M
IR = work / N
```

At fixed full-controller trajectories and calls, saving is `c*(H-C)/N`, where c is .076/.074. A policy call always costs the full vision plus policy remainder. No request-dispatch, launch, batching or call-reordering latency enters these equations. Report CPU planning/search time separately.

### Selection status and verification

The two-witness structure and selection of `.1`/`.05` are **post-freeze B-dev-informed design/calibration**, despite all fitted metric/head/table artifacts being library-only. The calibration/evaluation init split in `screen.json` is a development diagnostic, not untouched confirmation: the earlier atlas examination saw these recordings. No task-specific acceptance thresholds were selected. Wider ensembles, tighter atlas thresholds, agentview-first, and donor-mass camera choice are measured alternatives and are not selected for the initial pilot.

`d2_validate.py` checks source manifests/B exclusion, forced-full action and guard parity, periodic cap peeking, legal two-slot memory, hidden-current-view poisoning, unchanged gripper, and additive completion accounting. This does not test a plugin adapter. Before serving, sol must also test actual camera dispatch, full fallback equivalence, exact GR00T prefixes, action padding, tail persistence, independent connections, retries, one guard update and one completion charge.
'''
    (HERE/'SPEC.md').write_text(spec)
    shadow=table(['cell / knob','proposal MSE delta vs full [95% episode bootstrap]'],[[label(r),f"{r['methods'][mainname]['shadow_mse_delta_ci']['mean']:.8f} [{r['methods'][mainname]['shadow_mse_delta_ci']['ci95'][0]:.8f}, {r['methods'][mainname]['shadow_mse_delta_ci']['ci95'][1]:.8f}]"] for r in E['cells'] if 'shadow_mse_delta_ci' in r['methods'][mainname]])
    choice=[]
    for r in E['cells']:
        if 'camera_choice' in r:
            for a in r['camera_choice']['results']:
                if a['config']['name']==mainname:
                    m=a['result'];choice.append([label(r),a['mode'],m['H'],f"{m['saving_fixed_calls']:.6f}",f"{m['missed']} / {m['extra']} / {m['gripper_disagree_half']}"])
    parity=table(['cell / knob','rank mismatches','max served-HIT action difference'],[[label(r),q['full_recording_parity']['row_mismatches'],f"{q['full_recording_parity']['max_served_hit_action_delta']:.9g}"] for r,q in zip(E['cells'],V['checks'])])
    offline=f'''# DIRECTIVE_2 offline evidence

Generated UTC: {stamp}. All numbers below come from executed scripts in this directory. This is post-freeze development evidence, entirely B-library/B-dev. No changed-controller SR was measured. Earlier library/token/representation/correction evidence remains in [the original OFFLINE](directive2/prior_delivery/OFFLINE.md); its latency-priced economics are superseded here.

## Data and baseline reconstruction

Read with the official `look_recording.read_looks` hash/B-contract checks: `r12_devlooks_50L`, `r12_devlooks_400`, now-complete `r12_devlooks_knob`, and the completed local replacement roots represented below. There are {V['episodes']} accepted episodes and {V['looks']:,} actual looks across these cells. Failed episodes are retained. Each recording contract excludes the served B subset; no queries store or A output is opened.

**Source correction:** the coordinator's 13:0x note, discovered during final handback review, retired `r12_devlooks_50S` and `r12_devlooks_200`. All current tables exclude them. Their earlier screens are superseded historical artifacts only. `r12_devlooks_50S_local` is the replacement; `r12_devlooks_200_local` is included only if complete and validated. Method/thresholds were not changed after this source correction. The 18:22 prediction's Spatial/200 rows were explicitly withdrawn in an append-only correction; their replacements below are post-freeze validation, not a silent rewrite. Current authority is the `VALID_ROOTS` allowlist; it blocks new source reads from the retired roots.

`dev_validation.py` reconstructs the deployed full ranking, kernel weights, correction and guard inputs; source counts reconcile to accepted journals and summaries. The current numerical replay uses the **actual recorded full donor rows and weights** for memory, preserving source behavior when tiny ranking differences occur. Independently reconstructed full-path parity is:

{parity}

Forced-full replay exactly reproduces stored full proposal actions, no-progress spans, calls, and look timing. Periodic due status is independently reconstructed from pre-look histories and the actual keyed cap, including guard-suppressed due events. The {V['poisoned_missing_camera_checks']} poisoning checks reproduce complete half proposals without any current missing-camera feature. `directive2/validation.json` records checks, source hashes and CPU affinity.

## Why the frozen atlas is not enough

Repricing changes the economics, not the guard errors. At the new prices, the original pre-look atlas gives:

{old}

These missed calls can make a naive replay ledger look cheaper. That is not evidence of preserved SR. Later full looks can also inherit a changed progress reference from an earlier half decision. Small action error alone was therefore an inadequate acceptance rule.

## Current-observation acceptance and explicit abstention

The revised method uses a current-wrist-only witness and a full-metric witness with a library-transported estimate of the missing view. It requires corrected action, gripper and progress agreement before committing. Rejection pays the other half of vision and restores a full-information decision; this is additive under DIRECTIVE_2.

All retrieval metrics, safety rows and heads are selected-library-only. Structural selection and the `.1`/`.05` comparison occurred **after the frozen atlas had been evaluated on B dev**, so they are expressly post-freeze dev calibration. The raw candidate artifacts separate legal proposal inputs from hidden-view evaluation. Full/half correction heads and the five-fold H scale are in `dev_validation/fits/`; wrist-only sensor fits are in `directive2/fits/`.

{core}

Saving holds the baseline denominator, look schedule and calls fixed and charges completion on baseline calls. The propagated missed/extra counts hold the recorded states/look slots fixed while recomputing donor progress and periodic histories. Once actions or timing diverge, these are counterfactual diagnostics, **not realizable changed-controller episodes or SR estimates**. Recorded execution metadata is likewise conditional on the full baseline; a serving adapter must use its own accepted history.

The stricter `.05` alternative has the following exposure:

{strict}

Zero observed call/gripper discrepancies in this small exposure does not certify lossless SR. Even motion-only changes can alter later states.

## Uncertainty, split and policy shadows

The following intervals bootstrap whole episodes within task, with fits/gate fixed. They do not account for development selection and are not SR intervals. The split takes the first half of sorted B init IDs within each task as calibration, the remainder as evaluation. Earlier atlas work had already inspected these recordings, so neither half is described as pristine confirmation. Periodic recording groups are smaller.

{uncertainty}

The extra-call column subtracts policy computation for observed additional calls while assigning **no saving to missed calls**. It still cannot predict actual total IR after trajectories change.

Only L10-50 off has a policy shadow at every look. The proposal motion MSE difference over ten controls/six standardized motion channels is:

{shadow}

These differences are close to zero; the single stochastic shadow sample is not optimal-action truth. Its MSE cannot be converted into a predicted success rate. No policy-shadow error is invented for the other roots.

## Additional method screens

`d2_screen.py` measures {len(configs())} configurations per cell: frozen/strict pre-look gates, wrist-only witness, two witnesses, more individual-donor missing-view hypotheses, exact-top agreement, reduced atlas mass, and tighter action tolerances. Broad ensembles largely eliminate acceptance; relaxing the old atlas mass adds exposure but brings back guard/gripper errors. Detailed rows are in each `screen.json`, not silently discarded.

External-camera-only retrieval and causal camera choice were separately fitted for L10-50/400 in both models. The alternative saves the last genuine wrist instead of the external view, fits matching metrics/heads and a five-fold scale, and chooses its first camera solely from the two safety masses on the previous full donor mixture. It never chooses after inspecting both current views. At the same `.1` gate:

{table(['cell / knob','camera method','half looks','IR saved','missed / extra / grip'],choice)}

The small coverage gains do not resolve the guard risk, so wrist-first remains the pilot proposal. The alternate fits and all outcomes are retained in `alternate.pkl`, `camera_choice.json` and the candidate NPZ files.

## GR00T numerical mismatch

Vision-only offline computation used the required interpreter, bounded allocation, and the deployed GR00T torch/FA2 dependency stack. No model server or rollout was launched. Peak reserved allocation was {G['peak_reserved_mib']:.0f} MiB. The first attempts exposed missing/incompatible FA2 dependencies in the workspace stack; the final run used the matching installed packages together without modifying either environment. Logs retain the failures and successful run.

On {G['count']} paired reconstructed B observations, maximum token delta was {G['raw_summary']['token_max']['max']:.8f}; mean token RMS delta {G['raw_summary']['token_rms']['mean']:.8f}; mean pooled-key RMS delta {G['raw_summary']['key_rms']['mean']:.8f}. Projected wrist delta RMS was {G['projection']['50']['rms']:.8f} at size 50 and {G['projection']['400']['rms']:.8f} at size 400. These are own executed measurements, not copied latency-prep values.

The images were upsampled from 56px and deterministically cropped/resized. This isolates batch-one versus batch-two numerical differences on identical inputs, but cannot reconstruct original input detail. The selected GR00T libraries lack original RGB. We therefore did **not** re-encode the actual library or claim these are exact single-camera keys for the recorded states.

The eight largest measured projected-norm perturbations, both signs, were applied to legal current-wrist inputs. Requiring the candidate and acceptance to remain stable gives:

{numtable}

This filter is post-freeze dev numerical calibration, not a certified perturbation bound. It shows the risk reaches retrieval and eligibility; it is insufficient for a deployment claim. The spatial checkpoint has no corresponding numerical fit in this exploration. A genuine single-camera library/query study remains required there too. Completed separate-camera policy prefixes also require validation; the current replay assumes the original full fallback and cannot establish its actual numerical equivalence.

## Knob evidence and compute-share accounting

The full-controller off/periodic roots do not share all their B initial states. Comparing only their actual intersection gives:

{paired}

This overlap is too small and incomplete across tasks to identify a reliable local buyback slope. Unpaired aggregate SR differences between the roots are not a paired knob response. The forecast uses explicit sensitivity slopes, not a fitted slope from these sparse intersections.

With c=.076/.074, work is `2c*F+c*H+c*C+(1-2c)*M`; a rejected half acquisition plus completion is one full look. At fixed states/calls, saving is `c*(H-C)/N`. A policy call costs one in either camera plan. There is no request-overhead penalty or latency credit. With look rate .5 and no calls, all-half ceilings are {E['ideal_no_call_ceilings']['pi05']['allhalf']:.6f}/{E['ideal_no_call_ceilings']['groot']['allhalf']:.6f}; alternating ceilings are {E['ideal_no_call_ceilings']['pi05']['alternating']:.6f}/{E['ideal_no_call_ceilings']['groot']['alternating']:.6f} for π0.5/GR00T. The observed-cell ceilings below ignore safety, warmup and future behavior:

{costs}

## Reproduction and provenance

Run from repository root with `taskset -c 10-21,54-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m exp.offline_search.rounds.r12.astra.<module>`. Use `CUDA_VISIBLE_DEVICES=` for CPU scripts. No script starts a server, worker, evaluation or sync. The common audit hook rejects foreign writes, the other R12 explorer directory, non-library store paths and nonallowlisted closed-loop roots.

Order: `dev_validation extract`/`fit` for the relevant root; `d2_consensus --roots ...`; `d2_screen`; `d2_camera_choice`; `d2_groot_mismatch`; `d2_stress`; `d2_validate`; `d2_evidence`. The GPU script only loads the vision tower/projector and caps tensors below the user limit. Inspect `nvidia-smi` before running it. `d2_documents` writes reports and appends a prediction once; do not rerun it as an original preregistration after receiving changed-controller outcomes. All raw fits, NPZ arrays, JSON summaries, read audits and logs stay in this directory.

Size 400 replaces the earlier size-500 dev ambition because size 500 has no nonserved B init in the existing pool. Earlier size-500 library replay remains historical evidence, not a new closed-loop validation. No new per-cell SR estimate is fabricated for unrecorded cells.
'''
    (HERE/'OFFLINE.md').write_text(offline)
    predrows=[];strictrows=[]
    bykey={(r['root'],r['cell'],r['size']):r for r in E['cells']}
    for f in E['forecasts']:
        r=bykey[(f['root'],f['cell'],f['size'])]
        row=[label(r),f"{f['raw_ir_saving']:.6f}",f"{f['sr_delta_central_scenario_pp']:+.2f}",
            f"{f['matched_ir_saving_at_25pp_per_ir']:+.6f}",f"{f['matched_ir_saving_at_10pp_per_ir']:+.6f}",f"{f['break_even_loss_at_25pp_per_ir']:.4f}"]
        (predrows if f['method']==mainname else strictrows).append(row)
    headers=['cell / knob','raw IR saved','SR Δ scenario, pp','matched saving @25 pp/IR','matched saving @10 pp/IR','max buyable loss @25, pp']
    prediction=f'''

---

## DIRECTIVE_2 valid-root forecast — {stamp}

This is a **new post-freeze B-dev-calibrated forecast**, before any changed-controller outcome of the revised design. The original {V['frozen_prediction_bytes']}-byte prefix remains unchanged (SHA256 `{V['frozen_prediction_sha256']}`), as do every earlier forecast/correction section. This section uses only the coordinator's currently accepted roots, including completed `_local` replacements; it supersedes the 18:22 source-dependent forecast. Retired non-local Spatial-50/200 results are excluded. The gate parameters are unchanged. Pending replacement cells get no new forecast. Earlier latency prices and GR00T cost exclusion are superseded. Both models are in scope at exactly half the full vision compute share; request latency tricks earn no savings.

Selected pilot rule: wrist-first two-witness completion gate at `.1`, with `.05` as the strict diagnostic. Metric/head/table fits are selected-library-only; the structure/threshold selection used post-freeze B dev. Raw savings transport the measured fixed-state camera acceptance to the observed full-controller schedule and charge completion on baseline calls. They are conditional forecasts, not measured IR of a new controller. The measured full-controller SRs in OFFLINE.md are not look-half SR.

I cannot infer an honest measured SR change from these recordings. To make the forecast falsifiable, the central **subjective SR-loss scenario** is −.10 pp for π0.5 and −.25 pp for GR00T at `.1`; use the broad subjective range [−1,+.5] pp, not a confidence interval. For `.05`, the central scenario is −.02 pp with range [−.5,+.25] pp. These are explicit assumptions generated by `d2_evidence.py`, never regressions from action MSE. A lossless outcome remains the goal, and then the matched-SR saving equals the raw saving.

For assumed positive local slope b in pp per unit IR and loss L in pp, matched saving is `S-L/b`. The table reports b=25 and b=10 sensitivity scenarios. Sparse paired knob evidence does not estimate these slopes, and no positive entry is a demonstrated matched-SR result.

{table(headers,predrows)}

Strict alternative:

{table(headers,strictrows)}

GR00T's primary table is conditional on the recorded joint-path representation. The numerical-stability stress filter lowers the L10 camera opportunity further:

{numtable}

That filter does not repair the missing original pixels or certify separate-camera policy prefixes. For a real single-camera GR00T deployment, raw saving, SR change and matched-SR total IR remain **unidentified** until the exact numerical path passes B validation. Spatial GR00T is likewise conditional. This is an accuracy qualification, not the superseded claim that GR00T has no half-look compute saving.

The all-half and alternating upper bounds under current baseline calls are:

{costs}

My directional expectation is that the stricter method is most plausible for goal (a), with very small raw savings; goal (b) is fragile because even a small SR loss can exceed the buyback budgets above. The `.1` method is a B-pilot candidate, not a claim of success. Do not spend missing guard calls as savings. The existing knob roots overlap on only {E['paired_knob'][0]['paired']} B pairs per model, so no reliable local buyback curve is available. Pending cells and size 500 get no invented forecast. Freeze any further changes in a new timestamped section before changed-controller or test-A outcomes; never alter the earlier sections.
'''
    pp=HERE/'PREDICTION.md'
    assert pp.read_bytes()==pred_before,'Prediction changed during generation; refuse concurrent mutation'
    with pp.open('ab') as f:f.write(prediction.encode())
    experiments=f'''# DIRECTIVE_2 experiments: B pilot before A

Frozen proposal timestamp UTC: {stamp}. No launches were performed or are authorized for this explorer. These are concrete handoff arms for the coordinator/sol. All parameter selection to date is post-freeze B development, explicitly labeled. Preserve the original prediction and the new timestamped section.

## Adapter and exact-input shadow pilot

Implement the SPEC transaction and verify actual one-camera acquisition, independent sessions/reset, exact execution-age eligibility, corrected tail/padding, hidden-view poisoning, idempotent retries, full fallback before one guard update, periodic pre-look due peeking, and one completion charge. Force-full behavior must match the appropriate full-information control. All ledger prices are compute shares: π0.5 `.152/.076/.076/.848`; GR00T `.148/.074/.074/.852` for full/half/completion/remainder. Latency measurements are technical reporting only.

For GR00T, first record exact B encoder input pixels, joint and single-camera keys, and matched-noise policy chunks from joint and separately completed prefixes. Refit selected-library single-view keys or validate robust matching; give the full-information control the same numerical path. The supplied L10 numerical stress filter is provisional B-dev calibration on reconstructed images. It does not pass the exact-input requirement. Do not count an extra joint re-encoding after a rejected half as a free completion. See REQUESTS.md.

## Paired B controller pilot

Initial candidate cells: both models L10-50 and L10-400; π0.5 Spatial-50/200 and L10-200; GR00T Spatial-50 after its own exact-input validation. Use the same task/init/seed pairs across **all** arms, outside the served B subset. At size 400, use the existing nonserved B pairs; do not substitute test A or pretend a size-500 disjoint pool exists. The completed off/knob recordings overlap on only {E['paired_knob'][0]['paired']} pairs per model and are insufficient for the buyback comparison; the pilot must fix that pairing.

Per cell, run full baseline, two-witness `.1`, two-witness `.05`, and the earlier full-vision-plus-displacement control. Start knob off. Then compare full and both half variants at the same frozen periodic setting; the existing L10-50 settings are {next(r['periodic_setting'] for r in E['cells'] if r['root']=='r12_devlooks_knob' and r['cell']=='pi05_l10'):.4f} (π0.5) and {next(r['periodic_setting'] for r in E['cells'] if r['root']=='r12_devlooks_knob' and r['cell']=='groot_l10'):.4f} (GR00T). They are settings calibrated to the old `.32` target, not guarantees of observed IR. Preserve them for the same-setting comparison.

For buyback, let g be the full baseline's selected-library periodic setting at `.32`. Add candidate settings `.9*g` and `.8*g` on B; smaller caps mean more policy calls. If B fails to bracket the target SR, choose additional settings on B and append a prediction before running them. Do not select from A. Determine an SR target inside both measured B curves and freeze it before A; if none exists, report matched-SR effect unidentified and stop that cell.

Use all available ten-task, nonserved B pairs with paired policy noise where supported; keep an independent noise repeat for stochastic-policy sensitivity. A small pilot can reject poor methods but cannot certify tiny SR losses. Report paired SR differences and intervals without turning rounded SR equality into losslessness. Report IR from complete work/slot ledgers, all full/half/completion counts, guard and knob calls, first-call times, no-progress spans, extra/missed shadow events, state/atlas scores, gripper discrepancies, mode-specific distances, camera memory age, and actual policy-prefix validation.

Prioritize π0.5 spatial for the first behavior-changing pilot because the current `.1` shadow screen has no missed guard calls or gripper disagreements there. This is a **post-freeze dev-informed priority**, not an A-selected winner. GR00T remains a real research/pilot arm after the numerical input requirement; its pricing is no longer a reason to exclude it.

## Predeclared decision and A grid

Goal (a) needs supported lossless/noninferior SR with lower total IR; a finite-sample zero-loss point estimate is insufficient. Do not invent a noninferiority margin for the owner. Goal (b) needs lower **measured total IR at matched SR**, including new calls, completion, changed episode lengths and guard feedback. A positive offline action-MSE result or fewer guard calls is insufficient. Failed or unbracketed cells do not advance.

After B, freeze the exact selected method, camera/numerical path, fits/source hashes, target SR, periodic settings, all paired init/seed lists and analysis before receiving any A outcomes. Candidate A grid is the model/suite/size cells listed above, filtered only by B go/no-go; pending replacement cells first require accepted B recordings. Each included cell gets the appropriate full baseline, one B-selected half variant, and full-plus-displacement control, with knob off and the frozen settings needed to bracket the target. Include the second threshold only if it was independently preselected on B; do not create an A tuning menu. Each included arm uses the full official ten-task × fifty-init A set through the coordinator. This explorer reads no A files and launches none of these runs.

Estimate matched-SR IR by mixing frozen policies across episodes. SR is linear in mixture probability p, but pooled IR is `((1-p)*work0+p*work1)/((1-p)*slots0+p*slots1)`. Do not interpolate reported IR ratios directly or extrapolate beyond the frozen SR bracket. Bootstrap paired initial states within task carrying whole-episode work, slots and success together. Report same-setting and matched-SR contrasts, uncertainty, and noise-repeat sensitivity. No threshold revision or failed-arm retest based on A is allowed.
'''
    (HERE/'EXPERIMENTS.md').write_text(experiments)
    handback=f'''# DIRECTIVE_2 handback

Updated UTC: {stamp}. All requested offline work and documents are delivered. No claim is made that the owner SR/IR goal is already achieved; no changed-controller run was launched.

Read [REPORT.md](REPORT.md) first (plain Chinese owner section), then [SPEC.md](SPEC.md), [OFFLINE.md](OFFLINE.md), the appended [PREDICTION.md](PREDICTION.md) section and [EXPERIMENTS.md](EXPERIMENTS.md). The old prediction prefix is unchanged: {V['frozen_prediction_bytes']} bytes, SHA256 `{V['frozen_prediction_sha256']}`. Other original reports are archived in `directive2/prior_delivery/`.

The cost ruling is fully applied: π0.5 one camera/completion `.076/.076`; GR00T `.074/.074`. No request latency optimization is a method or credited saving. The frozen atlas has substantial guard divergence on B cache-driven states and is not recommended unchanged.

The new pilot proposal is the two-witness wrist-first completion gate at `.1`; strict diagnostic `.05`. All metric/head/table fits use the selected B library. The structure/threshold selection is post-freeze dev calibration. Broader ensembles, external-camera-first and causal camera choice were evaluated and retained as alternatives, not silently omitted.

Validated source coverage: {V['episodes']} B episodes, {V['looks']:,} looks from the currently accepted roots listed in OFFLINE.md. Retired non-local Spatial/200 roots are excluded; their 18:22 forecast rows were withdrawn by an append-only correction. Completed local replacements use the same frozen gate parameters. Forced-full guard/action parity, causal periodic peeking, source contracts, memory age, hidden-view poisoning and cost identities passed. The actual knob/full intersection is only {E['paired_knob'][0]['paired']} initial states per model; a reliable local buyback slope remains unavailable.

GR00T received an actual bounded vision-tower experiment ({G['count']} reconstructed B observations, peak reserved {G['peak_reserved_mib']:.0f} MiB) and retrieval-level numerical perturbation tests. Exact original-pixel single-camera library/query compatibility and separately completed policy-prefix equivalence remain unproven. This limits deployment claims, not the model's inclusion or its compute savings. REQUESTS.md states the concrete missing input/path validation.

## Files for implementation and audit

- `d2_consensus.py`: library-fitted wrist witness and pure missing-view-transport proposal.
- `d2_screen.py`: legal gate conditions, causal guard/periodic fixed-state replay and all screened configurations.
- `dev_validation.py`, `dev_validation/fits/*/half.pkl`: official-reader extraction, deployed-full reconstruction, half heads and exact five-fold H distance scale.
- `final_atlas/*/table.npz`, `directive2/fits/*/wrist.pkl`: selected-library safety maps, metric references and wrist witness fits.
- `d2_camera_choice.py`, `alternate.pkl`: separately fitted external-view/memory alternative and pre-encoding choice screen.
- `d2_groot_mismatch.py`, `d2_stress.py`, `directive2/groot_mismatch/`: measured batch-shape mismatch and explicitly calibrated numerical stability filter.
- `directive2/*/*/screen.json`, `camera_choice.json`, `numeric_stress.json`: complete per-cell outcomes and raw NPZ arrays.
- `d2_validate.py`, `directive2/validation.json`: substantive parity/causality/data/cost checks.
- `d2_evidence.py`, `directive2/evidence.json`: reproducible tables, bootstrap diagnostics, actual paired knob overlap and forecast assumptions.

No deployable plugin is claimed. Sol's remaining implementation work is the acquisition transaction, mode-specific history/tail plumbing and exact GR00T numerical-path validation, followed by the paired B controller pilot. Test A is conditional on a B-qualified, newly frozen grid. Size 500 receives no invented disjoint-B validation.

All writes stayed in this directory; CPU affinity was 10–21,54–65 with single-threaded BLAS. The required workspace interpreter was used. No git, forbidden deletion/process command, server/worker/sync/evaluation launch, test-A output, query store or other R12 explorer file was used. GPU work is complete and did not approach the user cap. Remaining uncertainty is explicitly recorded rather than represented as a completed SR claim.
'''
    (HERE/'HANDBACK.md').write_text(handback)
    dump(D2/'document_generation.json',dict(timestamp=stamp,evidence_sha256=sha(D2/'evidence.json'),validation_sha256=sha(D2/'validation.json'),
        files={f'{n}.md':sha(HERE/f'{n}.md') for n in ('REPORT','SPEC','OFFLINE','PREDICTION','EXPERIMENTS','HANDBACK')},
        old_prediction_sha256=V['frozen_prediction_sha256'],old_prediction_bytes=V['frozen_prediction_bytes']))
    print('D2 documents generated; earlier prediction prefix preserved',stamp,flush=True)

if __name__=='__main__':main()
