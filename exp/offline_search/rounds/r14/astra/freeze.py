"""Render the numerical R14 design contract and hash the frozen payload."""
from common import *
fence()
import numpy as np
from partition import VARIANTS,ANCHORS,FOLDS
from allocation import DOSES,GAIN,PRIOR_EPISODES

STAMP=now()
N_A=500
REPS_B=2
BOOTSTRAPS=10000
ROLLOUT_CLIENTS=16
GRID_SEED=20261014
BUDGET_BAND=.02
PAIR_BAND=.01
ENGAGEMENT=.05
TASK_SPREAD=.02
NI_MARGIN=.02
variants={k:dict(scope=s,metric=m,head=h,continuity='pool' if k in ('scope_only','all_pool') else 'task',
                         guard='pool' if k in ('scope_only','all_pool') else 'task')
          for k,(s,m,h) in VARIANTS.items()}
variants['continuity']=dict(scope='task',metric='task',head='task',continuity='pool',guard='task')
rows=[]
for cell in CELLS:
    host='local_4090' if cell.startswith('groot') else 'h100'
    for name in variants:
        rows.append(dict(arm=f'r14_{cell}_A_{name}_r0',part='A',cell=cell,variant=name,target='',replicate=0,
            rollout_seed=20261401,coin_seed=0,episodes=N_A,host=host,pair_group=f'{cell}_A_r0',**variants[name]))
    for target in DOSES:
        for rep in range(REPS_B):
            for method in ('periodic','task_periodic'):
                rows.append(dict(arm=f'r14_{cell}_B_{method}_{round(target*100)}_r{rep}',part='B',cell=cell,
                    variant=method,target=target,replicate=rep,rollout_seed=20261401+rep,coin_seed=0,
                    episodes=N_A,host=host,pair_group=f'{cell}_B_{target}_r{rep}',**variants['base']))
rng=np.random.default_rng(GRID_SEED);groups=list(dict.fromkeys(r['pair_group'] for r in rows));rng.shuffle(groups)
ordered=[]
for group in groups:
    group_rows=[r for r in rows if r['pair_group']==group];rng.shuffle(group_rows)
    ordered.extend(group_rows)
for i,r in enumerate(ordered):r['order']=i
csvout(HERE/'test_grid.csv',ordered)
fits=read(HERE/'allocation_fits.json');part=read(HERE/'partition_summary.json');dev=read(HERE/'dev_partition_summary.json')
cont=read(HERE/'continuity_summary.json');check=read(HERE/'selfcheck.json');controller=read(HERE/'controller_check.json')
al=read(HERE/'allocation_summary.json');split=read(HERE/'allocation_split_checks.json');ranks=read(HERE/'task_rank_agreement.json')
pilot=list(csv.DictReader((HERE/'pilot_grid.csv').open()))
pilot_n=sum(int(r['episodes']) for r in pilot)
assert all(not r['missing'] and len(r['pairs'])==50 for r in read(HERE/'pilot_selection.json')['cells'])

# Runtime scenario derives exclusively from allowed B episode-duration summaries.
rates={};runtime_sources=[]
for a in read(RUNS/'r11_devknob_50/arms.json'):
    if a.get('r11_method','off') not in ('off','periodic'):continue
    p=RUNS/'r11_devknob_50/runs'/a['arm']/'summary.json';s=read(p)
    rate=s['episode_duration_s'];key=(a['model'],a['suite_short'],a.get('r11_method','off'))
    rates[key]=(float(rate['p50']),float(rate['p95']))
    runtime_sources.append(dict(path=str(p),sha256=sha(p),p50=rate['p50'],p95=rate['p95']))
runtime=[]
for model in COSTS:
    low=high=0.;n=0
    for r in rows:
        if not r['cell'].startswith(model):continue
        suite=r['cell'].split('_')[1];method='off' if r['part']=='A' else 'periodic'
        lo,hi=rates[(model,suite,method)]
        # Planning uplift for higher call dose and pooled CPU work, not measurement.
        uplift=(max(1.,float(r['target'])/.32) if r['part']=='B' else (1.25 if r['scope']=='pool' else 1.))
        low+=r['episodes']*lo*uplift;high+=r['episodes']*hi*uplift;n+=r['episodes']
    runtime.append(dict(model=model,episodes=n,clients=ROLLOUT_CLIENTS,
                        lower_days=low/ROLLOUT_CLIENTS/86400,upper_days=high/ROLLOUT_CLIENTS/86400))
dump(HERE/'runtime_estimate.json',dict(scenarios=runtime,sources=runtime_sources,
    assumption='episode p50/p95 / 16 concurrent clients, B size50 timing; target/call and pooled-work planning uplifts; not a host throughput benchmark'))
plan=dict(timestamp=STAMP,cells=CELLS,doses=DOSES,variants=variants,A_replicates=1,B_replicates=REPS_B,
    episodes_per_test_arm=N_A,test_arms=len(rows),test_episodes=sum(r['episodes'] for r in rows),
    pilot_arms=len(pilot),pilot_episodes=pilot_n,grid_seed=GRID_SEED,bootstrap_replicates=BOOTSTRAPS,
    budget_band=BUDGET_BAND,pair_band=PAIR_BAND,placement_gate=ENGAGEMENT,task_rate_spread_gate=TASK_SPREAD,
    pooling_SR_noninferiority_margin=NI_MARGIN,allocation_gain=GAIN,allocation_prior_episodes=PRIOR_EPISODES,
    no_test_data_access=True,R13_validation_identities_unread=True)
dump(HERE/'design.json',plan)

def pval(cell,variant,field='mse_corrected'):
    return next(r[field] for r in part if r['cell']==cell and r['variant']==variant)
def change(cell,v):return 100*(pval(cell,v)/pval(cell,'base')-1)
pt=table(['cell','baseline loss','scope pooled','metric pooled','head pooled','scope+metric pooled','scope only task','all pooled'],[
    [c,f"{pval(c,'base'):.4f}"]+[f'{change(c,v):+.1f}%' for v in ('scope','metric','head','scope_metric','scope_only','all_pool')] for c in CELLS])
retr=table(['cell','foreign top1: scope pooled','foreign top1: scope+metric pooled','S×M loss interaction'],[
    [c,f"{100*(1-pval(c,'scope','top1_same_task')):.2f}%",f"{100*(1-pval(c,'scope_metric','top1_same_task')):.2f}%",
     f"{change(c,'scope_metric')-change(c,'scope')-change(c,'metric'):+.2f}%"] for c in CELLS])
ct=table(['cell','fresh top1 changed','raw action loss change'],[
    [r['cell'],f"{100*r['top1_changed']:.2f}%",f"{100*(r['pool_mse']/r['task_mse']-1):+.2f}%"] for r in cont])
dt=[]
for c in CELLS:
    rr=[r for r in dev if r['cell']==c]
    if not rr:continue
    base=next(r for r in rr if r['variant']=='base')
    for r in rr:
        if r['variant'] not in ('base','scope','metric','head','all_pool'):continue
        dt.append([c,r['variant'],'unlabelled' if r['mse_corrected'] is None else f"{r['mse_corrected']:.4f}",
                   'unlabelled' if r['frozen_mse'] is None else f"{r['frozen_mse']:.4f}",f"{100*r['guard_disagreement']:.2f}%"])
devtable=table(['cell','proposal','shadow motion loss','actual deployed loss','guard disagreement on fixed states'],dt)
alloc_table=table(['cell','dose','budget-band checks','paired-budget checks','worst |ΔIR|','min changed placement'],[
    [r['cell'],r['target'],f"{r['allocation_budget_pass']}/{r['checks']}",f"{r['paired_pass']}/{r['checks']}",
     f"{r['max_abs_delta']:.4f}",f"{r['min_engagement']:.3f}"] for r in al])
fit_table=table(['cell','dose','reference K','shared bias','task K min','task K max'],[
    [f['cell'],f['target'],f"{f['reference_K']:.4f}",f"{f['bias']:.6f}",f"{min(f['K']):.6f}",f"{max(f['K']):.6f}"] for f in fits])
sumchecks=sum(r['checks'] for r in al)
splitpass=sum(abs(r['allocation_ir']-r['target'])<=BUDGET_BAND for r in split)
splitmatch=sum(abs(r['delta_ir'])<=PAIR_BAND for r in split)
maxhit=max(read(p)['replay_hit_max_abs_error'] or 0 for p in (HERE/'data').glob('r12*.json'))
runtime_table=table(['host model','test episodes','runtime scenario days'],[
    [r['model'],r['episodes'],f"{r['lower_days']:.2f}–{r['upper_days']:.2f}"] for r in runtime])

offline=f'''# R14 offline partition evidence

Generated {STAMP}. All numerical tables are recomputed by scripts in this directory. No new variant has run closed loop. Action imitation loss is not SR, and fixed-state replay is not an environment simulation.

## What is actually task-indexed

The source audit covers `r10/recipe/recipe.py`, `fitting.py`, `build.py`, AWM, the inherited guard/blind/policy-tail implementations, `r11/knob/recipe.py`, and the serving plugin. `deployment_audit.json` hashes each actual recipe artifact.

* Retrieval scope restricts donor rows by query task. Both normal and early Mahalanobis transforms, including centering and standardization, are fitted per task.
* There is a separate task-specific continuity scale `T.s_c` in the fresh-after-MISS branch. It can affect actions when the normal policy tail falls back to a fresh look. `T.s_d` and projected visual centering affect confidence/diagnostics. With `guard_only`, no recovery or blending, those confidence changes cannot alter acceptance.
* The motion corrector has task-specific heads. Its task one-hot is constant within each head. The proposed pooled head removes that input to prevent an implicit task-conditioned correction model.
* Guard dictionaries `M0/M1` contain task-centred raw-camera means and `med_len` contains task median episode lengths. Motion/cosine thresholds themselves are shared. The deployed recipe disables stuck, terminal and overtime, leaving only no-progress, whose threshold is shared and whose inputs are retrieved-row progress and episode length. The row's progress/length are metadata, not learned task tables. Pooling guard dictionaries cannot change the active verdict under the deployed settings.
* Blind state scales and motion percentiles are task-indexed but their gates are inactive with budget-only anchor tails. Task identity checks on episode/anchor lifecycle remain necessary in every arm.
* Both camera PCAs and action sigma are fitted once per selected cell library. GC_dist uses one distance scale per cell/variant, and kref is cell-wide. Nothing is pooled across policies, suites or library sizes.

## Whole-episode library diagnostic

`partition.py` fits metrics on all training-fold rows and leaves every such row available as a donor. It evaluates {FOLDS} outer episode folds, with at most {ANCHORS} deterministic evenly spaced anchors per episode for head fitting and evaluation. Head weights preserve the represented full-row mass. An inner episode holdout fits the gate scale without exposing the outer validation episodes. All failed library episodes remain. Rows and labels never come from A.

This is an economical diagnostic, not the production fitter: head input moments and ridge sufficient statistics use anchor quadrature; gate scale uses one inner holdout rather than the production complete fold schedule; early distances use explicit float32 codes rather than the deployment affine representation. The selected-library PCA is fixed across folds and thus sees heldout covariates, as in the prior design. `SPEC.md` requires full-row production fitting and the original distance calibration. The earlier train-LOEO scale experiment was superseded because its in-metric-fit query distances suppressed correction; its log is retained, but none of its tables enters this freeze.

Loss below is episode-balanced mean squared motion error against stored policy chunks, scaled by training action sigma. Signed percentages compare corrected loss with the same diagnostic baseline; positive is worse. Gripper errors, raw loss, unattenuated half-strength correction, per-episode records and fold manifests are in the CSV/JSON payload.

{pt}

Corrector benefit is `mse_raw - mse_corrected`; head pooling is assessed at identical retrieval and gate scale. It is small relative to changing retrieval geometry in most cells. Library observations suggest retaining task geometry for the long-horizon suite, while pooling can help the small spatial libraries. This is not yet a closed-loop recommendation.

{retr}

“Foreign top1” measures task-identity agreement, not a ground-truth semantic retrieval score. Scope pooling often leaves task retrieval almost unchanged because the query's task metric already separates other tasks. The interaction is `(SM - S - M + base) / base`; it is an action-loss interaction. `scope_only` explicitly asks whether retaining only the retrieval boundary suffices after pooling learned geometry and correction.

## The additional continuity scale and guard null

`continuity.py` forces the fresh branch on library states, excludes the complete query episode from donors, and pools the tail-to-head continuity reference across tasks. This stress test conditions on the full fitted representation; it is not held-out metric performance and is not a forecast of fallback frequency. These are uncorrected action errors, isolating the scale's retrieval effect.

{ct}

The guard-null test changes all inactive task guard/blind tables and their shared thresholds much more strongly than ordinary pooling, while exercising dense, gapped and stationary histories. It preserves actions, donor IDs and active force/reason/flag fields on {sum(r['proposals'] for r in check['guard_null'])} proposals, including {sum(r['active_guard_proposals'] for r in check['guard_null'])} active no-progress decisions. This accompanies the source-level dependency argument; real plugin lifecycle checks are still sol's responsibility. No GPU arm is allocated to guard-only pooling. Continuity pooling is different: it has a real fresh-branch action effect and gets its own arm.

## Recorded B cache states

`dev_extract.py` is adapted from the frozen earlier extractor with its outputs and read fence redirected here. It reconciles accepted attempts, B contracts, journal outcomes, complete decision ledgers and shard hashes. Reconstructing the actual corrected cache on recorded HITs gives maximum absolute action error {maxhit:.3g}. `dev_partition.py` scores all recorded real looks with the same economical head/scale approximations described above.

{devtable}

The diagnostic baseline does not exactly reproduce the deployed full-row head; the actual deployed shadow loss is shown to make that gap visible. Compare pooled diagnostic proposals with their diagnostic baseline, not with the deployed column as if only one switch had changed. The tiny baseline guard disagreement also reflects prototype numerical differences. Guard comparisons hold the recorded state/look sequence fixed and omit feedback through actions and blind-veto timing; they are not IR predictions.

Shadow policy labels are available here only in the two small long-horizon cells. Spatial and middle-size recorded states still support retrieval/guard diagnostics but not a policy-action error claim. The requested shadow recordings and queued pilot fill that gap after the reservation is released. `task_descriptives.csv` reports old per-task off/periodic SR and IR without fitting to them; those methods spend different budgets.

## Task budget allocation

Priority uses only cross-fitted library cache/policy motion gaps, averaged within episode before log compression, then averaged per task and shrunk by the number of episodes. The two available task-rank checks against independent B shadow states are {', '.join(r['cell']+'='+format(r['spearman_library_dev'],'.3f') for r in ranks)}. This supports limited transfer of ranking, not a causal claim that an extra call will improve success. Neither SR, task success labels nor test outcomes set a priority.

The final shared intercept uses all eligible B-dev off/periodic paths to match reference spend; the final replay is therefore development evidence. An earlier identity-holdout fit/check is retained separately in `allocation_split_checks.json`: budget band passes {splitpass}/{len(split)}, and paired matching passes {splitmatch}/{len(split)}. Some sparse splits omit an entire task. These failures motivated retaining the final full-cohort refit as development calibration, not relabeling those splits as passed validation. Gain and shrinkage were not increased to rescue a weak result. Library identity-holdout checks fit both priorities and the intercept on the calibration identities; their input error tables themselves come from whole-episode cross-fitting on the selected library.

{alloc_table}

There are {sumchecks} rows in this combined table: full B-development replay where available plus a separate library identity-holdout diagnostic per cell/dose. In particular, the small pi05 spatial library has a heldout-library budget/matching warning. The full-cohort fits do not erase it. `allocation_checks.json` retains each source and both realized IR values.

`allocation_robustness.json` bootstraps task episodes with the scarce-sample cap while keeping the final bias fixed. It measures fragility, not an independent validation interval. `allocation_tasks.csv` retains every task priority/cap; no task effect is selected from SR. Missing tasks in a diagnostic split have null task rates, not estimated zero-call rates. Full fit tasks all have library evidence. Task-level changes in knob-call fraction are recorded relative to the periodic reference on the same states, so existing task guard/rate differences do not masquerade as allocation engagement.

At the large library size there is no B complement, so both utility and budget calibration use only the verified heldout-library tables. State distribution, episode length and guard-rate shifts remain unvalidated. Endpoints are explicitly tested: cap zero calls at every real look, a cap larger than an episode adds no calls, and mandatory guards always survive. No cost debt or learned state crosses episodes.

## Reproduction and limits

Run only the commands in `RUN.md`. Provenance files list every data read; the fence blocks test roots, query stores, reserved validation files, subprocesses and out-of-directory writes. No GPU, policy query, remote command or rollout was used. `controller_check.json` records {controller['reference_proposals']} proposals and {controller['idempotent_retries']} identical retries against actual R11 periodic query logic with its lower-layer query stubbed, including extra looks. This checks schedule arithmetic, not the production plugin.

The production full-row pooled fits, real serving parity, full B pilot and all test SR/IR effects remain future work. This deliverable freezes how to perform them, not their outcome.
'''
(HERE/'OFFLINE.md').write_text(offline)

spec=f'''# R14 implementation and experiment freeze

Frozen {STAMP}, before any closed-loop result of these new variants. The numerical payload is `design.json`, `allocation_fits.json`, `pilot_grid.csv`, `pilot_selection.json` and `test_grid.csv`. Exact floating-point values come from JSON, not rounded tables. This directory is an offline reference; sol owns the production adapter and fitting implementation.

## Part A switches and task identity

Expose independent `scope`, `metric`, `head`, `continuity`, and `guard` switches, each `task|pool`, on top of R10Recipe. Defaults must reproduce the original artifact/actions/decisions. Pool only within the same model, suite and nested B library. The input episode subset, cameras, action horizon, fixed hyperparameters and policy remain unchanged. Task ID is known from the original LIBERO instruction and must never be inferred from a retrieved donor.

{table(['arm','scope','metric','head','continuity','guard'],[[k]+[v[s] for s in ('scope','metric','head','continuity','guard')] for k,v in variants.items()])}

`guard=pool` by itself is an exact behavioral null under the active recipe and is checked on CPU, not a separate test arm. `scope_only` means only the retrieval boundary remains task-specific among the learned behavioral components. `all_pool` shares all of them. Both still use task identity for correct instruction delivery, logging, episode/anchor lifecycle checks and inherited periodic coins if a knob is separately enabled. Part A runs with layer four off. No pooled lower layers are combined with task allocation in the primary grid.

1. **Scope.** `task` retains only donors of the query task. `pool` exposes all rows of the selected library. With task metrics, transform every candidate using the query task's mean/std/main/early transform. Never compare candidates embedded by different task metrics. The median distance, neighbor ordering and kernel reference use the actual allowed candidate set. Use the inherited deterministic row-ID tie policy, donor-to-parent mapping, full action synthesis, corrector-anchor mapping and invalid-tail checks. Pooling scope alone still uses the query's task-specific head and continuity scale.
2. **Metric.** `pool` fits one joint geometry on every selected row: standardize joint PCA/state features; choose the inherited action-similar nearest rows from other episodes, now allowing other tasks; fit the original regularized within-pair covariance and full-rank Cholesky metric. Fit the early metric independently on the inherited early-step subset. Apply one shared transform to every task. The diagonal standardization and early affine-code representation are part of this switch. This does not pool `s_c`; continuity is a separate switch. `partition.py:metric_fit` is a bounded-memory reference to the same main fit equation, and its small exact-equation test is in `selfcheck.json`.
3. **Head.** For each library query, exclude the entire query episode from donor retrieval, use the selected scope/metric/continuity, and synthesize the uncorrected cache proposal with the inherited step-zero/fresh/stale rules. Default fitting uses the original R10 training query convention, with later anchors in the stale regime after a valid policy tail. Build the original PCA/state/cache-action/time features. Task heads preserve the existing task one-hot feature. The pooled head drops those task columns entirely and fits one R10 weighted ridge/RFF head on all tasks. Preserve the original random-feature count/seed, regularizer, action scaling, episode-balanced row weights, blend, unchanged gripper and correction horizon. Use every training row for production, not the economical anchor sample. Do not fit a separate task intercept, scale, embedding or calibrator in the pooled head.
4. **Continuity.** `task` retains each `T.s_c`. `pool` computes the inherited median of nearest normalized tail-to-head action distances across the whole library, excluding each row's own episode, with the inherited epsilon, and aliases that scalar for every query task. This acts only in the fresh branch. It must not alter ordinary stale/early geometry or periodic coins. `continuity.py` supplies its exact frozen library scalar per cell in `continuity_summary.json`; the builder must recompute and check it, not depend on explorer code at serving time.
5. **Guard and inactive auxiliaries.** Pool raw-camera centres over all selected rows, recompute the shared consecutive-frame cosine percentile with those centres, and replace each task median episode length by the median over all selected episodes. Motion threshold is already shared. Alias global versions of inactive blind state scales/motion percentiles and confidence-only state-distance/visual-centering references when producing the fully pooled artifact. Preserve no-progress constants, per-row normalized progress, each donor episode's length, the progress memo, and task/episode identity checks. Do not pool row progress into a global time axis. Disabled stuck/terminal/overtime bits, no events, no burst, no recovery/blending, budget-only cache tails, and lifecycle-only policy tails are invariant. Diagnostics/confidence may change; actions and active verdicts must not change for a guard-only toggle.

**Shared fitted components and descendants.** Reuse the same fresh selected-library PCA for every arm of a cell, once per camera. Action sigma and kref remain cell-wide and byte-identical to the baseline. When scope or metric changes, regenerate its LOEO training cache proposals, refit heads with the same head-sharing choice, and recalibrate the single GC_dist scale with the original complete episode-fold schedule using that scope/metric. This is a structural component ablation: dependent fitted values are regenerated, not left mismatched to their geometry. A head-only or continuity-only switch keeps the same ordinary GC_dist scale. Preserve the original attenuation equation, no step-zero correction and motion-only correction. Confidence fits may be regenerated as descendants but must remain excluded from `guard_only` acceptance. Record every source SHA, subset row SHA, scope and fit fold in the artifact manifest.

Inherited numerical constants are fixed: camera projection dimension 64 each; metric action neighbors 3, covariance ridge factor .1 and full rank; early metric rows have step at most 2; retrieval has 16 neighbors; kref is 5 for the size-50 cells and 8 for the other cells; ridge heads have 384 random Fourier features, alpha 100 and seed 0. The pooled head has 207 input features after dropping the 10 task columns, versus 217 for a task head. Predict the first 10 controls in 6 motion channels; keep the gripper untouched. The effective correction multiplier is `.5 * clip((2 - distance/scale)/1.25, 0, 1)` and is zero at step zero. At serving time distance is the minimum geometric distance among the actual proposal anchor donors, preserving the original fresh-branch nuance; do not substitute normalized kernel scores.

For production GC_dist calibration, assign folds by position in each task's frozen nested episode list exactly as R10 fitting does. Each distance query excludes its heldout episode fold from metric training and donors; pool eligible tasks only according to the switch. Keep the fixed selected-library PCA, fit training-only action sigma, retrieve using the early/main rule, omit step-zero distances from the episode-balanced median and fit one scalar. This is the full calibration, unlike the reduced inner split in the exploratory head diagnostic. At serving time both head and gate must reuse the cached proposal's actual donor rows, including foreign-task donors under pooled scope. Do not silently rerun task-restricted distance lookup inside the corrector.

## Part B: task_periodic on the unchanged recipe

Baseline is the existing R11 periodic controller and its original library-derived cap K. `allocation_fits.json` stores the exact recomputed reference K for each cell/dose. New task-periodic changes only which fixed K the periodic controller uses. No distance feedback, extra model, extra look, per-state action-gap predictor, shared mutable bank or cross-episode adaptation is introduced.

From the SHA-verified R11 whole-episode-held-out library error/guard tables, compute one priority per task:

```
eligible = real library anchor and not mandatory guard
episode_value = log1p(mean(eligible corrected motion MSE / action_sigma**2))
r[t] = mean(episode_value for task t)
n[t] = number of task episodes with at least one eligible finite error
mu = mean(r over represented tasks)
s = max(std(r over represented tasks), 0.1)
z[t] = n[t]/(n[t] + prior_episodes) * clip((r[t] - mu)/s, -2, 2)
z[t] = 0 when a task has no finite evidence
K[t] = max(expm1(log1p(K_reference) + bias - gain*z[t]), 0)
```

The table already uses normalized corrected motion error; do not normalize it twice. `allocation.py:libpack` squares the inherited RMS risk once to obtain MSE. Freeze gain `{GAIN}` and prior episode count `{PRIOR_EPISODES:g}` globally. Standardize before shrinkage so equal small sample sizes do not cancel the shrinkage. There is no task-SR input or success gate. All final priorities use the complete cross-fitted library; size-full-library uses the identical rule.

Calibrate only one shared `bias` per cell/dose. Use complete, accepted off/periodic episodes from the explicit B roots listed in each fit. Each fixed path includes actual real-look steps, observed mandatory guards and terminal decision count. Replay exact inherited keyed coins with actual original task/init IDs. Seek the smallest worst absolute difference between task-periodic and the cell-wide periodic IR over these paths. IR is always `sum(cv*V + cm*M)/sum(N)`, never a mean of task or episode IRs; `(cv,cm)` is `(.152,.848)` for pi05 and `(.148,.852)` for GR00T. `allocation.py:solve` bisects bias in `[-8,8]` for 32 iterations. Move the lower bound to the midpoint when the midpoint of the largest/smallest signed IR differences is positive, otherwise move the upper bound. Among visited candidates choose `(maximum absolute difference, absolute bias, signed bias)` lexicographically. Discrete keyed schedules make this a numerical search, not a claim of an exact smooth optimum. Use all allowed development episodes for the final refit. Keep sparse identity-holdout failures separate. If no B path exists, use verified library episodes and certified original B init identities. No fit or scalar update may use A, a pilot outcome, or the reserved R13 identities before release.

{fit_table}

At each real anchor call the lower-layer query once, preserve its cache proposal and guard memo, then take `K[query.task_id]`. Extract only accepted histories with existing `history(q)`. Compute `cap=floor(K)+I[uniform('R11-gap-v1',0,task,init,last_call_step,'cap') < frac(K)]`. Count real-look HITs since the last accepted policy call; call iff guard OR `run >= cap`. A mandatory guard resets the gap exactly like any call and keeps its original reason. A knob-only MISS gets a new collision-audited reason and the ordinary policy chunk/tail lifecycle. Do not change the existing coin seed, use a new coin key, index by donor task, charge a second call, or treat a proposal as acceptance.

The CPU reference is `controller.py`. Same-step retries return the same proposal; reset clears per-episode memo, decreasing steps and malformed histories fail through the ordinary contract fallback. Do not persist state across task, episode, connection, host or replicate. `off` delegates directly to the unchanged recipe; zero gain and zero bias reproduce cell-wide periodic exactly. A task cap of zero means always-call at real looks, not pure C5; long finite caps may yield no knob calls in a short episode while leaving guards intact. Do not clip away mandatory calls or silently normalize episode budgets at termination. A changed success rate changes episode lengths, so a fixed offline cap vector cannot guarantee aggregate live IR without the pilot gate.

Log the original accepted N/V/M, task ID, target, reference K, task K, cap, coin, guard/knob flags, priority, gain, bias, fit SHA and fresh/stale branch. Include actual CPU retrieval/correction latency, parent donor task IDs, corrector norm/gate, active guard reason, and lifecycle fallback counts. Diagnostics must survive the plugin scalar-field limit; use a versioned side record if necessary. Shadow-enabled B arms retain the corrected discarded cache proposal on misses. Never add shadow calls in A.

## B pilot and release gates

Run `pilot_grid.csv`: {len(pilot)} arms / {pilot_n} episode runs, paired existing B identities from `pilot_selection.json`. Each cell uses the specified same pairs across all methods/replicates. Small cells exercise every A switch and both B doses plus the lower-dose repeat; middle cells exercise baseline/all-pool and both B doses. The list is disjoint from each actual served subset and uses only previously consumed B identities, including cross-model exclusion verification for the middle GR00T cell. These are development implementation checks, not untouched validation. No new identity is selected from or against the reserved R13 list. All pilot runs wait behind the coordinator's R13 scheduling decision.

Before any live run, require default off equivalence, CPU guard-null equivalence, complete parent-row support for pooled donors, query-task metric arithmetic, episode/donor exclusion in fits, step-zero behavior, both fresh and stale branches, accepted-history reconciliation, same-step idempotence, independent reset/clone behavior, deterministic reordering of independent connections, and exact policy-tail/invalid-tail/early-termination handling in the real plugin. The offline reference tests do not replace these checks. The continuity arm must engage in an injected fresh fallback and be identical in nominal stale/early fixtures.

For each B arm require absolute realized total IR error at most `{BUDGET_BAND}`; each paired task-periodic/reference comparison must additionally differ by at most `{PAIR_BAND}`. Report episode-cluster bootstrap intervals, mandatory-guard floor and actual terminal cost. Do not pass a failing cell by averaging across tasks/cells/doses. Guards take precedence when an arm is infeasible. The target band applies to the entire paired cell arm, not separately to tasks, whose rates are intentionally unequal.

Allocation engagement requires both (a) at least `{ENGAGEMENT}` symmetric-difference/union change in non-guard call positions against the unchanged periodic reference replayed on the allocation arm's own recorded states, and (b) at least `{TASK_SPREAD}` range across tasks in **allocation-minus-reference** knob-call fraction on those same states. This removes rate differences that existed before allocation. Count every task with observations, flag missing tasks, and report actual task N/V/M descriptively. For A scope/metric/head, require structural activation in CPU fixtures and report live donor/head changes; no minimum SR or minimum action difference is a release criterion. Natural non-engagement of a continuity switch is a valid null result, with fallback counts reported.

Mechanics failures permit implementation repair with frozen mathematics. Budget or allocation-engagement failures stop release of that affected primary cell/dose and are retained as failures; there is no hidden bias refit on pilot data, gain search, arm substitution or A-driven adjustment. Hand back for a separately versioned B-only redesign if necessary. A-only ablations are not blocked by a B allocation failure. Size-full-library can proceed only after the controller passes the smaller long-horizon cells for that model and its own fit passes mechanics; label it library-only transfer. The coordinator may elect to run a failed frozen arm as an explicit failure arm, never as validated matched-budget evidence.

## Economical frozen test grid

`test_grid.csv` contains {len(rows)} arms / {sum(r['episodes'] for r in rows):,} episode runs: {sum(r['part']=='A' for r in rows)} A arms and {sum(r['part']=='B' for r in rows)} B arms. All cells are {', '.join(CELLS)}. Part A uses one replicate of each structural arm with layer four off. Part B uses the original lower layers, the two periodic methods, doses {list(DOSES)}, and {REPS_B} replicates. Every arm runs the full unchanged test manifest of {N_A} episodes; the explorer has not opened that manifest or any A data. Coordinator tooling binds the ordinary A contracts later.

Every GR00T arm and control is pinned to the local 4090; every pi05 arm and control is pinned to H100. Queue a cell if its host is unavailable. No cell/contrast migrates across hosts. Frozen group/arm order uses seed `{GRID_SEED}`; paired B methods are adjacent. Replicate rollout seeds differ; the inherited coin seed stays fixed. Verify that rollout seeds actually affect simulator/policy randomness; repeated metadata alone is not independent replication. Concurrent episodes are a throughput assumption, not an independence assumption. No additional pure-policy control is required for these contrasts; the A baseline is the newly paired recipe/off control, and B has its own newly paired periodic control. No previous A results are borrowed.

{runtime_table}

These runtime ranges are scenarios from allowed B episode-duration medians/upper quantiles, divided by {ROLLOUT_CLIENTS} concurrent clients, with explicit target/pooled-work uplifts in `freeze.py`. They do not measure H100 throughput, include fit time, guarantee scaling, or include queueing behind R13. With both hosts available, elapsed test time is the larger host duration; half the assumed concurrency doubles it. Pilot time, setup, failures and host capacity loss are additional. This grid saves a guard-null arm, omits the unused scope+head corner, limits A to one replicate, and does not repeat pure controls. It does not support a complete interaction factorial or precise per-cell SR claims.

## Frozen analysis

Always report per-cell SR and aggregate realized IR, their paired differences and per-task descriptions. For A, estimate scope/base, metric/base, head/base and continuity/base main contrasts, scope–metric difference-in-differences, scope-only/base and all-pool/base. The factorial interaction is conditional on task heads; head interactions beyond the observed combinations are unidentified. Guard pooling is a structural null, not an empirical claim with a made-up zero-error interval. A's natural guard/IR response belongs to the treatment effect and must be reported, not normalized away.

For B, estimate task-periodic minus cell-periodic SR at each frozen dose with equal total spend. A direct contrast is labelled matched only after both target and pair gates pass. If a live pair fails, retain its SR/IR results as unequal-budget results; do not call the SR difference a task-allocation benefit. Do not interpolate or extrapolate two noisy dose points to repair a failed budget gate. The intended full-grid matched estimand is unestablished if required cells lack valid matches; do not silently average only successful cells.

Use equal weight to model×suite, then equal weight to represented library sizes within that group, then doses and B replicates. Cluster paired bootstrap resampling by original suite/task/init across models, sizes, doses and repeats, stratified by suite/task, with `{BOOTSTRAPS}` resamples. Use the same resampled identities for paired arms. Report replicate-block sensitivity for B; A has no estimate of between-run variability. Per-task effects and dose interactions are descriptive. Do not count every repeated episode run as independent. Apply Holm correction to the four pooled A main contrasts; other contrasts are prespecified descriptive estimates with intervals. A pooling claim of practical equivalence requires a lower SR-difference bound above `-{NI_MARGIN}` and no material IR increase, not merely a nonsignificant p-value. Insufficient precision means inconclusive. No SR-based stopping, dose selection, best replicate, new arm or task-specific fit follows A.
'''
(HERE/'SPEC.md').write_text(spec)

prediction=f'''# R14 predictions before new closed-loop results

Frozen {STAMP}. No new pooled-component or task-allocation closed-loop outcome was read. These are qualitative forecasts grounded in the executed B-only code; numerical SR gains are deliberately not invented from imitation loss.

* **Metric and scope.** Task geometry should matter more than the explicit donor boundary in the long-horizon cells. With the query task metric retained, pooling scope should usually have a small effect. Removing both can expose cross-task donors and hurt more than removing scope alone. Spatial small libraries are the clearest exception: their library action losses favor pooled geometry, so a universal “everything must remain per task” conclusion is unlikely.
* **Corrector.** Pooled heads should have a smaller average effect than pooled geometry. Sparse-library pooling may reduce variance, but the current library diagnostics do not demonstrate a uniform advantage. The economical diagnostic head differs from the full production head; its disagreement with actual B shadow loss lowers confidence in precise SR ordering.
* **Guard.** Guard-only pooling will preserve actions and decisions under the unchanged active recipe. Any observed effect is an implementation/lifecycle violation, not a learned advantage. Continuity pooling can change fresh-fallback donors but should have little average effect if valid policy tails retain the usual stale regime.
* **Size.** Larger libraries do not necessarily remove the need for task geometry. The scope boundary can become less relevant while a task transform still shapes useful neighbors. Pooling all learned behavior can therefore remain worse even with abundant donors.
* **Allocation.** A modest pooled success gain is plausible at the lower dose if library error ranking identifies tasks where policy calls help. A null or adverse result is also plausible: high action gap can mark unrecoverable states or a weak policy. The higher dose leaves less useful allocation freedom. I do not predict a reliable improvement in every cell or task.
* **Budget.** Full-cohort replay is encouraging but is fitted development evidence. Sparse identity splits, especially small pi05 cells, exposed cost mismatch. Live episode-length/guard shifts may fail the frozen gates. Large-library behavior is transfer-only. A failed budget comparison cannot support a success claim at equal spend.

The numerical expectations already observed offline are in `OFFLINE.md`, `partition_summary.json`, `continuity_summary.json`, `allocation_summary.json` and `allocation_split_checks.json`; those are measurements, not predictions of SR. `allocation_tasks.csv` freezes who receives larger/smaller caps before any new outcome. Predictions will not be rewritten after a pilot or A result.
'''
(HERE/'PREDICTION.md').write_text(prediction)

(HERE/'RUN.md').write_text('''# CPU reproduction

Run from the repository root. Every script and output is in this directory. Prefix every Python command with:

```bash
taskset -c 10-21,54-65 env PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= .venv/bin/python
```

Pass these script paths in dependency order, with the common directory prefix `exp/offline_search/rounds/r14/astra/`:

1. `inventory.py`
2. `early_grid.py`, then `pilot_selection.py`
3. `dev_extract.py`
4. `partition.py` (threaded CPU), `dev_partition.py`, `continuity.py`
5. `allocation.py`, then `task_study.py`
6. `selfcheck.py`, `controller_check.py`, then `summarize.py`
7. `freeze.py`, then `audit.py`

The final freeze records timestamps and hashes, so rerendering it is an explicit new freeze. Do not run the inherited production CLIs: they own other directories. No command here launches an environment, contacts a remote machine or uses a GPU. The blocked-metric implementation bounds working memory; all source arrays and fit provenance come from B-library artifacts or the explicit B-development root allowlist.
''')

(HERE/'STATUS.md').write_text(f'''# R14 offline design status

Offline deliverables completed and frozen {STAMP}: `OFFLINE.md`, `SPEC.md`, `PREDICTION.md`, early `REQUESTS.md`, executable CPU studies, exact pilot identities/grids and the economical test grid. No new variant was run closed loop. No test-A data or reserved R13 validation identity file was opened. No GPU, remote action or rollout was used.

Final pilot: {len(pilot)} arms / {pilot_n} episode runs. Final test plan: {len(rows)} arms / {sum(r['episodes'] for r in rows):,} episode runs. The early pilot request was expanded to isolate the additional continuity scale discovered in the code audit; the current CSV is authoritative.

Important limitations: production full-row pooled fits and real-plugin parity remain for sol. Shadow policy labels currently support only the two small long-horizon cells. All final B budget refits use existing development data; earlier sparse-split failures remain visible. Size-full-library lacks a B complement. Runtime is a B-timing scenario, not a measured H100 benchmark.

Open questions for the coordinator: when the R13 pilot releases its reservation; when already requested spatial/middle-size shadows become available; whether the planned host/client capacity is available; whether the rollout seeds affect actual policy/simulator randomness. No answer is needed to review or implement this frozen design. Do not refit silently after these answers or results arrive.
''')

(HERE/'HANDBACK.md').write_text(f'''# R14 design handback

Deliverables through the offline design phase are complete. Start with `SPEC.md`; exact parameters are in `allocation_fits.json`, switches in `design.json`, and arm/identity contracts in `pilot_grid.csv`, `pilot_selection.json` and `test_grid.csv`. `PREDICTION.md` is frozen before new closed-loop outcomes.

The main code finding is that task-indexed guard dictionaries do not control the deployed active no-progress verdict. They are a CPU equivalence check, saving a GPU arm. The extra continuity scale is action-relevant only in the fresh branch and is isolated explicitly. Task geometry, donor scope and correction heads remain distinct switches; foreign donors must be scored in the query's coordinate system.

For sol: implement the full-row pooled fits and lower-layer switches, use the original periodic controller with the frozen task caps, and perform real-plugin lifecycle/row-mapping/coin equivalence checks. The CPU reference compared {controller['reference_proposals']} proposals to the actual periodic query logic, but is not a serving adapter. Keep all GR00T arms on the local 4090 and all pi05 arms on H100.

For the coordinator: `REQUESTS.md` queues {len(pilot)} B pilot arms / {pilot_n} episodes after the R13 scheduling decision. It launches nothing. The frozen A grid has {len(rows)} arms / {sum(r['episodes'] for r in rows):,} episodes. Budget or engagement failures stop the affected matched-budget claim; retain them and request a separately versioned B-only redesign. No A-based fitting, dose change, host migration or best-replicate selection is authorized by this design.

This is a design handback, not the later results handback. `ANALYSIS.md` and the owner-first Chinese `REPORT.md` are intentionally deferred until the coordinator resumes this work with completed runs. The outstanding data/capacity questions are in `STATUS.md`.
''')

# Bind all numerical payload and own source; final audit is deliberately separate.
names=['SPEC.md','PREDICTION.md','OFFLINE.md','REQUESTS.md','design.json','allocation_fits.json','allocation_tasks.csv',
       'pilot_grid.csv','pilot_selection.json','test_grid.csv','partition_summary.json','dev_partition_summary.json',
       'continuity_summary.json','allocation_checks.json','allocation_split_checks.json','allocation_robustness.json',
       'selfcheck.json','controller_check.json','runtime_estimate.json']
names+=sorted(p.name for p in HERE.glob('*.py'))
dump(HERE/'freeze.json',dict(timestamp=STAMP,no_new_variant_closed_loop_results=True,
    files={name:sha(HERE/name) for name in names},affinity=sorted(os.sched_getaffinity(0)),
    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),plan=plan))
provenance('freeze')
print(json.dumps(dict(test_arms=len(rows),test_episodes=sum(r['episodes'] for r in rows),pilot_arms=len(pilot),pilot_episodes=pilot_n,runtime=runtime),indent=2))
