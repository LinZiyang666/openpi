"""Render round8 diagnosis from the admitted, split-specific local results."""
import json
import numpy as np
from .safe import HERE,owned
from .diagnostics import read


def load(name):return json.loads((HERE/'results'/name).read_text())
def pct(x):return f'{100*x:.1f}%'
def n(x):return '—' if x is None or (isinstance(x,float) and not np.isfinite(x)) else str(int(x))
def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+[
        '| '+' | '.join(map(str,row))+' |' for row in rows]
def put(name,lines):owned(HERE/name).write_text('\n'.join(lines)+'\n')


def main():
    s=load('summary.json');o=load('corrector_summary.json');a=load('diagnostics.json');tasks=load('task_metadata.json')
    p=read('pairs');c=read('calls');d=read('decisions')
    report='''负责人说明：这轮只交诊断，不发新方案。GR00T 的主要问题更像是少数轨迹进入反复开合抓手、短暂恢复后又停住的状态，而不是保护条件满足后没有及时调用模型。也没有证据支持在抓取、搬运或释放阶段关闭现有纠正器。协调方正在测试的升级调用方案，本轮没有重复制作。

在允许查看的 0–29 上，GR00T 配方成功 250/300，纯策略成功 268/300；配方失败而纯策略成功的有 41 回合，反向获益有 23 回合。这个差距主要来自 0–19：只看用于评估的 20–29，配方是 88/100，纯策略是 89/100，损失／获益为 8／7，不能据此认定评估集上有稳定的六个百分点差距。π0.5 在同样的 0–29 上是 272/300 对 272/300；20–29 是 89/100 对 87/100。两边来自不同批次，差值还包含随机运行差异。

GR00T 的 41 个损失回合，首次保护调用中位数在第 10 个决策，每个决策是五个控制步；27 回合在首次闭合抓手之前已经调用。所有满足无进展条件的时点都执行了调用。每个损失回合都在纯策略完成对应任务的时间之前调用至少两次，因此“普遍第一次调用太晚”不是最有力的解释；日志仍无法确定物体何时真正抓错或掉落。

首次调用后的下一次观察，GR00T 与 π0.5 的进展恢复率在 20–29 都约 92%。算入所有重复调用后，分别为 63% 与 70%；逐回合等权后是 82% 与 83%。这说明反复陷入困难状态的少数回合拉低了每次调用的表面效率。GR00T 损失回合中，70% 的调用落在释放附近或释放后的张开阶段。π0.5 另有持续升级调用，会改变留下来接受普通保护调用的状态，不能将这些数字解释成两种模型的纯粹恢复能力差异。

现有纠正器在另外三种已记录轨迹的 20–29 上，将 GR00T 运动动作误差降低 41.2%，五类抓手指令阶段都改善。预先规定的四个停用条件均使误差上升 6.2%–45.5%，因此没有足够依据冻结观察触发的新臂。离线标签不能证明纠正器绝不伤害闭环；配方自己的日志缺少同状态策略标签及物体真值，这是剩余的关键不确定性。

## Evidence and decision

- [DATA_ANALYSIS.md](DATA_ANALYSIS.md) contains split-specific counts, task lists, timing/recovery definitions, phase/distance comparisons, and all 41 GR00T lost episodes.
- [Lost evaluation timelines](results/lost_eval_timelines.svg) show actual gripper commands and policy calls for all eight evaluation losses. [lost_episodes.csv](results/lost_episodes.csv) includes both models; [episode_events.json](results/episode_events.json) includes every admitted episode's event and call sequence.
- No missed no-progress verdicts: GR00T 2,647/2,647 eligible fresh decisions call policy. The unmodified corrector does not change the gripper; reconstructed fresh-cache gripper error is below 4e-7 for both models.
- GR00T's 0–29 net losses are concentrated in task 9 (microwave mug, −10), task 7 (soup + cheese basket, −7), task 0 (soup + sauce basket, −4), and task 6 (mug + pudding, −3). Task 8 gains four. Task IDs are descriptive reporting only; no task-indexed rule was fitted or added.
- A concrete counterexample to a distance-only explanation is task 3/init 23: 33 calls, mean distance .090, correction RMS .0229, and final retrieved progress .976, yet failure. Retrieved demo progress can look nearly complete while the environment remains unsolved.
- 18 safety/statistical tests pass. Numerical artifact comparison verifies 602 arrays and all shared serving state across the source and deployed recipe pickles. Their bytes differ because eight optional pace-wrist metadata/attribute fields are absent in each earlier deployed pickle; stack behavior and correction parameters agree.

## Scope

No arms frozen; no new run root, PREDICTION.md, emitter package, sync, launch, or process change. All new files are inside this round8 directory. Fit cuts use only 0–19; evaluation uses 20–29; no prohibited roots or 30–49 payloads are admitted. Existing per-task recipe heads are audited unchanged. This is a development diagnosis on the reused evaluation population, not a new independent confirmation. [HANDBACK.md](HANDBACK.md) gives reproduction and coordinator guidance.
'''
    put('REPORT.md',report.splitlines())
    lines=['# Data analysis — R9 astra round 8','',
        '## Population, sources, and units','',
        'The only full-run outcome sources are `r09_recipe_full_g/r9eq_groot_l10_50`, `r09_recipe_full_p/r9eq_pi05_l10_50`, and `r08_main/{r8_groot_l10_P10,r8_pi05_l10_P10}`. Each contributes exactly tasks 0–9 × inits 0–29. `safe.py` lexes top-level task/init and UID scalars before decoding any outcome payload; nested/escaped identities cannot admit a row, and duplicate/conflicting identities fail closed. Forbidden root prefixes are checked before and after resolving paths. Only accepted, error-free terminal attempts are joined, and decision sequences must be contiguous with no conflicting duplicates. No full-500 summary, full-population dataframe, or mixed-log hash is used.','',
        'P10 actions come from existing compact debug NPZs. Only task/init members are opened first; any archive containing an inadmissible identity is rejected before action/state members are accessed. Admitted decision IDs are checked against the accepted attempt. Corrector auditing uses the similarly admitted R8 A (plain cache), CU (uniform randomized calls), and IP (independent randomized calls) compact trajectories. No raw mixed NPZ action member is loaded. Library actions and the frozen recipe artifacts are existing deployment inputs. Task descriptions come from exactly one admitted init-0 P10 episode per task.','',
        'Fit/descriptive training split = 0–19, evaluation = 20–29. Pooled 0–29 tables are descriptive, never threshold fitting. Quantile cuts use 0–19 only. Bootstrap intervals resample init clusters, preserving tasks; repeated use of this development population limits confirmatory interpretation. All scripts run under CPU affinity 10–21,54–65 and one BLAS thread. No serving logic, task selector, or policy-call escalation is introduced.','',
        'A decision executes five simulator controls; P10 and a guard MISS commit ten controls through one policy-tail decision. Step indices below are zero-based decisions, not seconds. P10 is this matched ten-control pure-policy baseline, not a different replanning-horizon pure-policy run. GR00T client resize is 256 in both arm specs; π0.5 P10 explicitly sets 224 while recipe uses its client default. Both specs set replan_steps=5. This is a historical, cross-batch comparison without common policy random draws. π0.5 recipe additionally uses persistent pace-lag escalation; ordinary guard-call comparisons exclude reason 91 but remain affected by that selection.','',
        'Source vs full-run recipe pickle bytes are not equal. `tools.provenance` checks the numerical/object state: 301 arrays per model and every shared scalar agree. Only the optional `stage_fit`, `wrist_fit`, `follow_kwargs`, `pace_lag` fields and their fit-info entries differ by absence in the earlier deployed stack artifact. [PROVENANCE.json](PROVENANCE.json) records hashes for artifacts and stat/identity provenance for admitted inputs.','',
        '## Success and paired loss population','']
    rows=[]
    for model in ('groot','pi05'):
        for split in ('fit','eval','all'):
            z=s[model]['splits'][split];pairs=z['paired'];lo,hi=z['delta_ci95']
            rows.append([model,split,f"{z['recipe_success']}/{z['n']}",f"{z['pure_success']}/{z['n']}",pairs.get('lost',0),pairs.get('gained',0),f'{100*(z["recipe_success"]-z["pure_success"])/z["n"]:+.1f}',f'[{100*lo:+.1f}, {100*hi:+.1f}]'])
    lines+=table(['Model','Split','Recipe','P10','Lost','Gained','Delta pp','Init-cluster 95% interval pp'],rows)
    lines+=['','Lost = recipe fails and P10 succeeds. Gained = the reverse. These pairs identify trajectories to examine; they do not establish that the corrector caused the failures. GR00T fit delta is −8.5 pp, evaluation delta −1 pp. The full allowed pool delta is −6 pp, but it should not be presented as the evaluation estimate.','',
        '### Task inventory (descriptive only)','']
    rows=[]
    for t in range(10):
        g=s['groot']['splits']['all']['by_task'][t];pi=s['pi05']['splits']['all']['by_task'][t]
        rows.append([t,tasks[str(t)]['task_text'],f"{g['recipe']}/{g['pure']}",','.join(map(str,g['lost'])) or '—',f"{pi['recipe']}/{pi['pure']}",','.join(map(str,pi['lost'])) or '—'])
    lines+=table(['Task','Instruction','G recipe/P10 wins','G lost inits','π recipe/P10 wins','π lost inits'],rows)
    lines+=['','Tasks 7 and 9 account for 20/41 gross GR00T losses and −17 of the pooled net −18 wins; other tasks partly offset one another. This concentration motivates the forensic description, not a task-based intervention. Evaluation GR00T losses are exactly `(0,24), (3,23), (4,20), (4,24), (6,26), (7,20), (9,22), (9,25)`.','',
        '## Guard timing and response','',
        'The deployed threshold is `noprog_n=3`, implemented as at least two nonadvancing decision intervals (`noprog_span`/`noprog_n >= 2`). It measures retrieved demonstration progress at real vision anchors; it is not object motion, contact, or task completion. Blind policy tails are not fresh progress observations.','']
    rows=[]
    for model in ('groot','pi05'):
        for split in ('fit','eval','all'):
            z=s[model]['splits'][split];g=z['arms']['recipe'];a0=a[model][split]['timing']['lost']
            rows.append([model,split,g['guard_calls'],g['escalation_calls'],z['guard_admission']['missed'],n(z['lost']['first_call_median']),f"{a0['before_close']}/{a0['n']}",f"{a0['before_release']}/{a0['n']}",f"{a0['call_before_far']}/{a0['has_far']}"])
    lines+=table(['Model','Split','Guard calls','Escalation calls','Eligible missed','Lost first call median','Before first close','Before first open after close','Call no later than first far observation'],rows)
    lines+=['','All 41 GR00T losses received policy calls before the paired P10 episode ended: mean 10.88, minimum two. First call median 10, first far-distance observation median 42; 38/39 losses that ever cross the fit q90 distance already called by that crossing. On evaluation these figures are 9.25 calls before P10 end, minimum two, first call 10, first far 50, and 7/7. No-progress condition-to-call lag is zero at observed fresh decisions. This rules out missed or delayed decision-level verdicts on these recorded observations and weakens a universal late-first-call explanation. It does **not** rule out a guard that detects semantic mistakes too late: an initial grasp could be wrong before retrieved progress stalls. There is no object-error onset label in these logs.','',
        '### Recovery proxies, with censoring and call weighting made explicit','',
        '“Clear” means the next real vision decision after the ten-control call has no-progress <2. This is nearly the same as advancing more than half a demo decision; these are not independent evidence. End-of-episode calls without a next observation are censored, not counted as either rescued or failed. “Repeat4” means another actual call in the next four decisions and excludes calls without four observed future decisions. π0.5 reason-91 calls are excluded from the guard denominator, although a subsequent repeat can be an escalation call.','']
    rows=[]
    for model in ('groot','pi05'):
        for split in ('fit','eval','all'):
            rec=a[model][split]['recovery'];g=s[model]['splits'][split]['call_stats']['guard']
            rows.append([model,split,f"{g['observed_next']}/{g['n']}",pct(rec['guard']['call_weighted']),pct(rec['guard']['episode_weighted']),pct(rec['first_guard']['call_weighted']),pct(g['repeat4'])])
    lines+=table(['Model','Split','Calls with next / total','Clear, per call','Clear, equal episode','Clear, first call','Repeat4'],rows)
    lines+=['','On evaluation, first-call rates are 89/97 for GR00T and 85/93 for π0.5: essentially equal. Equal-episode rates are 81.5% vs 82.8%; pooled-call rates are 63.1% vs 70.1%. Thus recurrent difficult episodes, not uniformly poor first rescues, explain much of the apparent per-call gap. GR00T lost-evaluation calls clear only 33.2% (223 observed nexts), compared with 82.5% in both-win episodes (359 nexts). These outcome-conditioned summaries are retrospective and selection-biased.','',
        'A descriptive adjustment uses only pre-call commanded state (approach/carry/post), within-model fit-distance bins, and fixed decision bins 0–19/20–39/40–59/60+. Shared cells receive the smaller of the two model call counts as weight. Evaluation standardized clear rates are 65.8% vs 71.1%, covering 688/715 GR00T and 381/381 π0.5 observed guard calls. The remaining difference is not a causal model-quality estimate: no randomization, different failure histories, coarse matching, and π0.5 escalation selection remain.','',
        '### Phase and retrieval distance','',
        'GR00T normalized negative gripper means close; π0.5 nonnegative means close. Majority of each executed five-control head supplies a command state. Approach precedes the first close; close transitions mark prior/current decisions as grasp; opening transitions mark prior/current/next as release; other closed/open decisions are carry/post. These phase labels use neighboring executed commands and are retrospective only. “Post” can include transit to a second object or drawer operation; “carry” does not prove an object is held. Adjustment above separately uses only past commands.','']
    rows=[]
    for phase in ('approach','grasp','carry','release','post'):
        gg=s['groot']['splits']['eval']['call_by_phase'][phase];pp=s['pi05']['splits']['eval']['call_by_phase'][phase]
        rows.append([phase,gg['n'],pct(gg['next_np_clear']),pct(gg['repeat4']),pp['n'],pct(pp['next_np_clear']),pct(pp['repeat4'])])
    lines+=table(['Command phase','G calls','G clear','G repeat4','π calls','π clear','π repeat4'],rows)
    lines+=['','Of the 1,170 calls in the 41 GR00T lost episodes, 608 are post and 214 release: 70.3% together. Lost-evaluation episodes make 6.88 close transitions on average versus 2.13 in paired successful P10 episodes; even after duration normalization, close-transition rates are 6.61 vs 3.96 per 100 decisions. This is evidence of repeated command cycles, not a count of failed physical grasps.','',
        'Recipe `dnn` is compared only within the recipe metric; debug-shadow `d1` is a different distance scale and is never equated to it. Fit fresh-decision median/q90 dnn cuts are GR00T .11634/.54807 and π0.5 .12664/.39634.','']
    rows=[]
    for model in ('groot','pi05'):
        for b,label in [('0','≤ fit median'),('1','median–q90'),('2','> fit q90')]:
            z=s[model]['splits']['eval']['call_by_distance'][b]
            rows.append([model,label,z['n'],pct(z['next_np_clear']),pct(z['repeat4']),n(z['step_median'])])
    lines+=table(['Model','Within-model distance','Eval guard calls','Clear','Repeat4','Median call step'],rows)
    lines+=['','High-distance guard calls are difficult for both models, with evaluation clear rates about 27% and 26%. GR00T has more such calls in absolute count (122 versus 86), but a smaller fraction of its ordinary guard calls (16.2% versus 21.7%). The full call mix and escalation prevent a simple intrinsic-quality conclusion. Among all lost GR00T episodes, mean fresh dnn is .481 versus .110 for both-win episodes; observed mean correction RMS is .0475 versus .0266. These are episode means, and larger corrections can be a consequence of the bad trajectory.','',
        '## Corrector audit and rejected observation gates','',
        'At a fresh recipe cache hit, reconstruct the weighted first five library controls from logged rows and weights, then subtract from served_head. RMS over six motion channels measures the **applied half-strength** correction. Blind tails are excluded from this reconstruction; policy calls are excluded because their served head is policy output. Gripper reconstruction maximum error is 2.66e−7 for GR00T and 3.94e−7 for π0.5: no additional gripper correction or sign flip.','',
        'GR00T lost-vs-both-win correction RMS before the first call is .0245 vs .0229 in 0–29, but after that call is .0541 vs .0281. Evaluation is similar (.0257/.0237 before; .0525/.0284 after). The large overall correction difference is mostly on the subsequent divergent paths. This association neither proves correction harm nor proves harmlessness.','',
        'The direct error check evaluates the frozen deployed half-strength head on fresh R8 A/CU/IP observations and their same-state policy-shadow motion labels, retaining the full committed ten controls. There is no refitting. Heads take their original task one-hot/per-task dispatch because these are the inherited recipe being audited; none of the diagnostic gates uses a task selector. Cache-chunk reconstruction against the deployed library must agree to <1e−4. Input PCA keys, normalized state, action/sigma, and capped step match the serving feature layout.','']
    rows=[]
    for model in ('groot','pi05'):
        for split in ('fit','eval'):
            z=o[model]['splits'][split]['total']
            rows.append([model,split,z['n'],z['episodes'],f"{z['cache']:.6f}",f"{z['recipe']:.6f}",pct(z['relative']),pct(z['better_fraction'])])
    lines+=table(['Model','Split','Anchors','Arm-episodes','Cache MSE','Recipe MSE','Relative change','Episodes improved'],rows)
    lines+=['','MSE averages within each episode, then equally over arm-episodes; the 300 evaluation arm-episodes are three variants × 100 pairs, not 300 independent initial states. Bootstrap gate intervals keep init clusters together. Training scores overlap the original head training data and are descriptive; evaluation trajectories use only 20–29. Shadow labels come from different paths than the R9 recipe and are imitation targets, not oracle optimal actions.','']
    rows=[]
    for phase in ('approach','grasp','carry','release','post'):
        g=o['groot']['splits']['eval']['by_phase'][phase];pi=o['pi05']['splits']['eval']['by_phase'][phase]
        rows.append([phase,pct(g['relative']),pct(pi['relative']),pct(g['grip_disagree']),pct(pi['grip_disagree'])])
    lines+=table(['Phase','G motion MSE change','π motion MSE change','G cache/policy grip disagreement','π cache/policy grip disagreement'],rows)
    lines+=['','Every phase improves on both models. Both the first five controls and five-control tail improve in aggregate. GR00T cache/policy gripper disagreement is larger (overall 10.2% vs 7.0%; release 31.0% vs 18.0%), although the inherited corrector leaves that channel alone. This suggests residual grasp/release timing or retrieval errors worth future observation-level study; it is not evidence for changing signs or a justified gripper-correction arm. Policy-shadow action sampling also contributes to disagreement.','',
        '[SCREEN_PROTOCOL.json](SCREEN_PROTOCOL.json) records four zero-correction diagnostics before their evaluation. Correction-RMS and shadow-distance q90 thresholds are fitted on pooled A/CU/IP inits 0–19 only: GR00T .09877/24.80203, π0.5 .10404/26.87250. The other conditions are sign-based and require no fitted threshold. These are ablations of the motion correction only; nothing changes gripper output or guard timing.','']
    rows=[]
    for gate in ('crossing','open','large_correction','far_retrieval'):
        gt=o['groot']['splits']['fit']['gates'][gate];ge=o['groot']['splits']['eval']['gates'][gate];pe=o['pi05']['splits']['eval']['gates'][gate]
        rows.append([gate,pct(gt['relative']),pct(ge['relative']),f"[{ge['ci95'][0]:.6f}, {ge['ci95'][1]:.6f}]",pct(pe['relative'])])
    lines+=table(['Stop correction when…','G fit MSE vs recipe','G eval MSE vs recipe','G eval absolute ΔMSE 95% interval','π eval MSE vs recipe'],rows)
    lines+=['','`crossing`: proposal changes gripper sign inside the committed ten controls; `open`: majority of first five proposed commands are open; `large_correction`: RMS above fit q90; `far_retrieval`: shadow d1 above fit q90. Positive change is worse. All four are worse in every evaluation variant A, CU, and IP, for both models. Thus no candidate meets the predeclared requirement even on training, and no evaluation-favored gate is selected.','',
        '## Individual GR00T lost episodes','',
        'All failures below last 104 decisions. `F` = fit init 0–19, `E` = evaluation init 20–29. `C/O` lists the first majority-close and subsequent open decision in recipe and P10; these are commanded events. `cyc R/P` counts recipe/P10 close transitions. `dnn` and correction are means; full maxima, both opening/closing counts, and exact event/call sequences are in the CSV/JSON artifacts. Correction units are normalized action RMS, not meters.','']
    rows=[]
    for r in p[(p.model=='groot')&(p.group=='lost')].itertuples():
        rows.append([f'{r.task}/{r.init}', 'F' if r.init<20 else 'E',
            f'{n(r.first_call_recipe)} {r.first_call_phase_recipe}',n(r.calls_recipe),
            f'{n(r.first_close_recipe)}/{n(r.first_release_recipe)} ; {n(r.first_close_P10)}/{n(r.first_release_P10)}',
            f'{n(r.close_events_recipe)}/{n(r.close_events_P10)}',f'{r.dnn_mean_recipe:.3f}',f'{r.correction_mean_recipe:.4f}',
            f'{r.final_progress_recipe:.3f}',n(r.decisions_P10)])
    lines+=table(['task/init','split','first call / phase','calls','C/O recipe ; P10','cyc R/P','dnn','corr','last demo progress','P10 end'],rows)
    lines+=['','Three instructive evaluation patterns:','',
        '- **Task 7/init 20 (soup + cheese basket):** call at 10 precedes close at 19; recipe opens at 20, while P10 closes at 11 and opens at 22. Nine recipe close/open cycles versus two P10, 29 calls, dnn .711 (max 1.546), correction .0768 (max .2037). This is repeated command cycling on a divergent trajectory, not simply an absent early call.','- **Task 9/init 22 (microwave mug):** first close is similar, 19 vs 18; recipe opens at 25 vs P10 31. It makes 29 calls, eight close transitions versus one, dnn .483, and finishes near demo progress .987 without solving. Early opening and later cycling are factual command differences; the logs cannot establish where the mug was.','- **Task 3/init 23 (bowl + drawer):** recipe/P10 close at 15/16 and open at 28/29, with one cycle each. First call at 28 and 33 total calls, despite low dnn .090, small correction .0229, and final demo progress .976. This is a low-distance, near-terminal failure that a blanket far-distance gate would miss.','',
        '[Eight evaluation timelines](results/lost_eval_timelines.svg) visualize every evaluation loss, not cherry-picked examples. Blue episode-end markers make clear how many recipe calls precede the P10 completion time.','',
        '## What the evidence answers','',
        '1. **Too late?** No dispatch defect or universal late-first-call pattern. The no-progress detector can still be semantically late or misled by retrieved progress; actual grasp-error onset is unobserved.','2. **Less effective per call?** Observed GR00T guard calls clear less often on the next observation and repeat more often within four decisions. First calls are comparable, and episode weighting greatly reduces the gap. Difficult-state composition and the π0.5 escalation policy remain major confounders. No marginal causal success gain per call is identified.','3. **Corrector harms a GR00T phase?** No phase-wide harm appears in the held-out fixed-state shadow audit; every tested observation-keyed removal worsens imitation error. Large corrections on failed recipe paths are mostly later, which cannot distinguish correction feedback from a response to bad states. A recipe-path policy-shadow or randomized correction ablation would be needed for that causal claim.','4. **Deployable fix?** None supported by this evidence. No arms frozen and no escalation duplication. Existing task-indexed recipe heads are not replaced or extended.','',
        'The strongest remaining diagnosis is a small subset of recurring, sometimes near-terminal semantic failures under sparse policy intervention, with extra gripper-event disagreement on GR00T. The present measurements do not prove which observation-only intervention would recover them.','']
    put('DATA_ANALYSIS.md',lines)
    handback='''# Handback — R9 astra round 8

Diagnosis only. **Zero arms frozen.** No `/home/weiland/trace_runs/os_closed_loop/r09_astra_r8/` run package was created, no standard-emitter mutation was needed, and no sync or evaluation launch was performed. PREDICTION.md is intentionally absent because there is no proposed arm. The coordinator's GR00T escalation experiment remains separate.

Read [REPORT.md](REPORT.md) first (plain-Chinese owner explanation), then [DATA_ANALYSIS.md](DATA_ANALYSIS.md). The main conclusion is recurrent difficult-state recovery, not a missed no-progress call or demonstrated phase-wide corrector damage. On evaluation the success gap is only −1 pp (88 vs 89 wins); the −6 pp pooled gap is driven by 0–19. Guard first-call recovery is about 92% for both models, while repeated calls and gripper cycles accumulate in GR00T losses.

## Reviewable evidence

- `results/lost_episodes.csv`: all 41 GR00T and 22 π0.5 lost pairs, with timing, events, distance, correction, and pure-policy duration.
- `results/episode_events.json`: exact event indices and all calls for the 600 admitted recipe episodes.
- `results/lost_eval_timelines.svg`: all eight GR00T evaluation losses; gripper commands and real calls, plus paired pure completion time.
- `results/{decisions,episodes,pairs,calls,calls_annotated,corrector}.parquet`: admitted analysis tables.
- `results/{summary,diagnostics,corrector_summary}.json`: complete split-specific summaries, denominators, and ablations.
- `SCREEN_PROTOCOL.json`: four observation-only diagnostic gates, recorded before their evaluation. Every gate fails to improve training and evaluation; no new arm selected.
- `PROVENANCE.json`, `results/artifact_compare_*.json`: input lineage and source/deployed recipe comparison. All 602 arrays and shared scalar serving state agree; earlier deployed pickles omit eight later optional pace-wrist fields each.
- `results/tests.log`: 18 passing admission, split, phase-sign, episode-weighting, recovery-censoring, join and actual-output checks.

## Reproduce locally, analysis only

From `/home/weiland/projects/openpi`:

```bash
taskset -c 10-21,54-65 bash exp/offline_search/rounds/r09/explore_astra/round8/tools/reproduce.sh
```

The script sets `PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH=.:src`, and OMP/OpenBLAS/MKL/NumExpr threads to one. It runs only offline parsers, numerical audits, plots and tests. Every output and plotting cache stays in round8. It never calls a controller, emitter, serving API, remote synchronization, git, or process-control utility.

Read-only inputs are the four named allowed outcome arms (only parsed inits 0–29), eight existing all-0–29 debug compact archives (A/CU/IP/P10 for both models), library/model artifacts, and ten admitted init-0 episode metadata files. Root-prefix checks occur before opening files; compact identity members are checked before payloads. Full-population summary files and prohibited run roots are unnecessary. Thresholds are fitted on 0–19 only. The existing recipe task heads are replayed unchanged for diagnosis; no new task table, task gate, or task threshold exists.

## Coordinator implication

Use the listed failures to examine the separately planned GR00T escalation experiment, while retaining its same-batch R9Recipe control. Report first-call and repeated-call recovery separately, alongside success and calls; an improvement in demo-progress clearing alone is insufficient. For a later non-escalation investigation, the missing evidence is same-state policy shadow and object/contact evidence on recipe trajectories, or a predeclared randomized corrector ablation. That is a proposed evidence requirement, not an authorized or emitted experiment in this handback.

Do not turn the task loss ranking, retrospective phase labels, or the 20–29 results into serving selectors. Cross-batch variation, π0.5 escalation, sparse causal labels, and reuse of the development evaluation set limit the diagnosis. No new deployment success claim is made.
'''
    put('HANDBACK.md',handback.splitlines())


if __name__=='__main__':main()
