"""One-shot pre-outcome registration. Refuses to replace any frozen file."""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
R14 = HERE.parent


def main():
    names = ('PROTOCOL.md', 'protocol.json', 'PROTOCOL.sha256')
    if any((HERE / name).exists() for name in names):
        raise SystemExit('Protocol already exists; never overwrite a frozen file.')
    now = datetime.now(timezone.utc)
    sources = ['descent/HANDBACK.md', 'descent/selection.json', 'descent/cap_tables.json',
               'descent/prepare.py', 'descent/recipe.py', 'astra/allocation_fits.json',
               'astra/allocation_tasks.csv']
    p = dict(
        schema='r14.descent_astra.protocol.v1', frozen_at_utc=now.isoformat(),
        frozen_at_chicago=now.astimezone(ZoneInfo('America/Chicago')).isoformat(),
        pre_outcome=True, outcome_reads_before_freeze=0,
        cells=['pi05_l10_50', 'pi05_spatial_50', 'groot_l10_50', 'groot_spatial_50'],
        trace_parent='/home/weiland/trace_runs/os_closed_loop', root_prefix='r14_descent_',
        pool='B', per_task=30, tasks=list(range(10)), episodes_per_arm=300,
        descent_order=['c10', 'ir45', 'ir40', 'ir32', 'ir25', 'off'],
        comparison_arms=['task_ir32', 'task_ir40'],
        tolerances_pp=[0, 2, 5], primary_tolerance_pp=5,
        selection_rule='point_estimate_first_failure',
        pass_formula='100 * (candidate_successes - pure_successes) >= -tolerance_pp * n',
        reference='same-init pure_c10 within task; equal-task pooled cell reference for uniform',
        nonmonotone='stop at first failing lower level; no skipping/re-entry, smoothing, or global-min search',
        equality='zero tolerance means no net SR loss; improvements pass; literal equality is not the rule',
        incomplete='fail closed: all four cells and all eight arms must have exactly the frozen B identities',
        retries='last accepted terminal error-free journal record per exact UID; count normal failure; decision logs only for its attempt',
        duplicate_decisions='sorted shard filename / line order, last per UID and step, matching collector; validate contiguous steps and summary ledger',
        confidence=dict(alpha=.05, family_comparisons=200,
            paired_lower='CP_lower(gains,n,alpha/2) - CP_upper(losses,n,alpha/2)',
            paired_two_sided='[CP_lower(g,n,alpha/4)-CP_upper(l,n,alpha/4), CP_upper(g,n,alpha/4)-CP_lower(l,n,alpha/4)]',
            simultaneous_lower='replace alpha with alpha/200, across 4 cells x 10 tasks x 5 non-reference levels',
            cell_bounds='average ten task bounds (stratified fixed task mix); never treat task-mixed Bernoulli trials as iid',
            sensitivity='same first-failure descent using simultaneous lower >= -tolerance; diagnostic only, no primary override',
            assumptions='independent representative paired inits within each task; no independence required between arms or tasks for union bound'),
        library=dict(priority_field='priority.z', expected_direction='higher priority -> higher ordinal descent level',
            rank_metrics=['Spearman with average ties', 'Kendall tau-b'],
            ordinal={'off':0, 'ir25':1, 'ir32':2, 'ir40':3, 'ir45':4, 'c10':5},
            cap_mapping='nearest log1p(K) among c10/ir45/ir40/ir32/ir25; ties choose higher inference; never infer off from finite K',
            cap_targets=['ir32','ir40'], level_metrics=['exact_fraction','within_one_fraction','mean_absolute_ordinal_error','mean_signed_ordinal_error','confusion'],
            interpretation='descriptive per cell, target, tolerance; no fitted mapping or pooled-only success verdict; allocation is budget conditioned, not a calibrated SR boundary predictor',
            provenance='priorities use library states; cap bias used previous B state-path calibration; do not call the full cap fit library-only'),
        cost=dict(coefficients={'pi05':[.152,.848], 'groot':[.148,.852]},
            formula='IR=(c_vision*sum(V)+c_policy*sum(M))/sum(N), fixed owner normalized cost per five nominal controls',
            validation='full vision mode, full policy steps, request length 5; reconcile N,V,M to summary cost_ledger; also report original ledger IR',
            savings='uniform_IR - stitched_task_IR; relative=(uniform_IR-stitched_task_IR)/uniform_IR; also total normalized cost per episode',
            stitched='select each task entire trajectories from its chosen uniform arm; B-selected descriptive estimate, not independent mixed-controller execution'),
        matched_ir=dict(nominal_targets=['ir32','ir40'], max_abs_ir_gap=.01,
            nominal='same target, same inits: report actual IR gap, paired SR delta and intervals; call approximately matched only if abs(gap)<=.01',
            exact='secondary expected episode-mixture of adjacent uniform periodic arms bracketing achieved allocation IR, sorted by measured IR, tie higher inference; use only ir25/ir32/ir40/ir45; no extrapolation',
            high_weight='(target_IR*N_low-C_low)/((C_high-C_low)-target_IR*(N_high-N_low))',
            paired_mixture='y_mix=(1-w)*y_low+w*y_high for each same init; fractional expected outcomes, not observed deterministic uniform policy'),
        bootstrap=dict(replicates=5000, seed=20261005, unit='init, resample 30 within each of ten tasks, paired across arms',
            interval='percentile 2.5/97.5; fixed selected levels and fixed mixture weights; descriptive conditional intervals only'),
        output=dict(primary=['levels.json','uniform.json'],
            alternates=['levels_tol0pp.json','uniform_tol0pp.json','levels_tol2pp.json','uniform_tol2pp.json','levels_tol5pp.json','uniform_tol5pp.json'],
            formats={'levels':'{cell: {"0": level, ..., "9": level}}', 'uniform':'{cell: level}'},
            report='DESCENT.md: plain Chinese owner section first, no agent codenames; all cells/task levels, actual IR savings, library agreement, then audit details',
            extras=['analysis.json','task_levels.csv','level_stats.csv','library_agreement.csv','matched_ir.csv','provenance.json']),
        input_sha256={str((R14 / s).relative_to(REPO)):hashlib.sha256((R14/s).read_bytes()).hexdigest() for s in sources},
    )
    budgets = {n: {t: n*t//100 for t in p['tolerances_pp']} for n in (30,300)}
    marginal = 100 * (1 - (.05/2)**(1/30))
    simultaneous = 100 * (1 - (.05/(2*200))**(1/30))
    md = f'''# Frozen per-task knob descent protocol

Frozen UTC: {p['frozen_at_utc']}  
America/Chicago: {p['frozen_at_chicago']}

Registered before reading any descent-root outcome. Only handback, selection/cap metadata,
existing allocation files and source code were read. No root journals/summaries, test-A data,
run polling or launches. `protocol.json` is the machine-readable registration; hashes bind
the input design and library estimates. Never edit these frozen files.

## Primary decision and tolerances

Use **5 percentage points, paired point-estimate noninferiority** as the PRIMARY development
selection rule; also evaluate 0 and 2 pp without choosing among them after outcomes.
For each task use all 30 frozen paired B identities, reference its same-init pure C10 successes.
Start `c10`, then visit `ir45 → ir40 → ir32 → ir25 → off`. A level passes iff
`100*(candidate successes - pure successes) >= -tolerance_pp*n`. Stop at the **first failure**,
select its predecessor; if all pass select off. Equality at the boundary passes. Improvements
pass even for tolerance zero; this is no degradation, not a literal equality test.
Non-monotone recovery below a failed level is reported and ignored for selection; do not skip
levels, smooth SR, or choose the globally lowest passing level. C10 passes by construction.

One outcome changes task SR by {100/30:.4f} pp. At 30 trials, allowed net losses for 0/2/5 pp
are {list(budgets[30].values())}; hence 0 and 2 pp give identical task tables. Five pp permits
one net loss and is the only requested nonzero per-task tolerance resolved by this sample.
This recommendation balances development sensitivity and quantization; it is not evidence
of population noninferiority. Report gains and losses separately, including offsetting gains.

For UNIFORM apply the identical first-failure point rule at the cell's 300 same-init episodes
(equal task weights), using the cell C10 reference. Net-loss budgets are
{list(budgets[300].values())}. This aggregate rule can hide task regressions; the task rule
cannot. Do not force the chosen uniform or per-task table to have identical measured SR.
Emit complete string task IDs 0..9 for every cell in `levels.json`, and exactly the same
cells mapping to one level in `uniform.json`. Allowed spellings:
`off, ir25, ir32, ir40, ir45, c10`. Primary files use 5 pp; `*_tol{{0,2,5}}pp.json` preserve all
alternates. These are inputs to the existing emitter, not authorization to run it or read A.

## Uncertainty and data discipline

Let gains be `(candidate=1,pure=0)`, losses `(candidate=0,pure=1)`. Report a conservative
paired 95% one-sided lower bound `CP_low(g,n,.025)-CP_high(l,n,.025)`, and a 95% two-sided
interval using four CP tails of .0125. A simultaneous lower band uses tail `.05/(2*200)`
for all 4 cells × 10 tasks × 5 lower levels. Cell bounds average ten task bounds, retaining
the fixed task mixture. A confidence sensitivity table uses the same first-failure rule with
the simultaneous lower band; it never replaces the registered primary selection.
Even zero discordances at n=30 gives a marginal lower bound of -{marginal:.2f} pp and a
simultaneous lower bound of -{simultaneous:.2f} pp. Conservative confidence selection can
therefore stay at C10; these data cannot usually certify small tolerances. No optional
stopping on partial data, sequential significance claims or post-hoc rule changes.

Require all four cells, eight arms per cell, exact selected B identities and 30 episodes/task.
Follow collector semantics: last accepted terminal error-free journal record per UID;
normal task failures count, execution errors do not. Only that accepted attempt's decisions
count; last record per UID/step in sorted shard/line order. Check UID pool, contracts,
frozen fits, served-subset exclusions, contiguous steps and summary counts. Abort on missing,
extra, malformed, unpaired or inconsistent data; no imputation or complete-case selection.
Snapshot/hash every read input and reject changes during analysis. No refitting. B identities
overlap prior development selections; this is development reuse, not held-out validation.

## IR savings and the library prediction

Use fixed owner cost IR `(cV*V+cM*M)/N`: π0.5 coefficients .152/.848, GR00T .148/.852.
Count accepted full-policy/full-vision decisions at five nominal controls; also retain the
original stage-ledger IR and check its counts. IR means normalized compute, not raw call rate.
Stitch selected whole task trajectories from the uniform-level arms; report selected SR,
IR, signed absolute and relative IR savings versus chosen uniform, and cost/episode.
This B-selected stitched estimate is optimistic and assumes task-local controller equivalence;
it is not an independently executed mixed-policy result. Negative savings remain negative.

Within each cell compare library `priority.z` to the selected ordinal level
`off=0, ir25=1, ir32=2, ir40=3, ir45=4, c10=5`: Spearman (average ties) and Kendall tau-b;
higher priority predicts higher needed inference. Undefined ranks with constant levels are
NA. For each frozen ir32/ir40 cap vector independently, map each finite K to the nearest
`log1p(K)` among the five finite uniform caps; ties choose higher inference. Finite caps
cannot predict off. Report exact and within-one agreement, absolute/signed ordinal errors
and confusion tables for every tolerance. No fitted or outcome-tuned mapping. These caps
are budget allocations, not calibrated tolerance boundaries; agreement is a diagnostic.
Priorities are library-derived, but cap bias used earlier B state-path calibration; do not
describe the full cap estimate as purely library-only. Summarize all cells and both targets,
not only favorable ones; there is no binary post-hoc 'matches' threshold.

Compare actual task_periodic with periodic at ir32/ir40 on identical inits: paired SR delta,
gains/losses and intervals, achieved IR and IR gap. Treat abs(IR gap) <= .01 as approximately
matched; always disclose the gap. Additionally compute an exactly IR-standardized expected
mixture of the two adjacent uniform periodic arms bracketing allocation's achieved IR,
sorting all four periodic arms by achieved IR (ties favor higher inference). Solve the
ratio-of-sums mixture weight in `protocol.json`; no extrapolation if no bracket. Same-init
fractional mixture outcomes permit a paired comparison but are not an observed deterministic
uniform policy. Report this distinction even if nominally matched; never relabel a nominal
target as measured IR or claim causal SR benefit from interpolation.

Use 5,000 within-task paired bootstrap draws, seed 20261005, for descriptive savings and
mixture-delta intervals, holding selected tables and mixture weights fixed. These conditional
intervals omit selection uncertainty and do not establish independent validation. With only
ten tasks, rank estimates and tied/discrete level agreements are fragile; no rank p-value
or out-of-sample prediction claim. A later independent evaluation is outside this analysis.

## Later execution

Write rerunnable, read-only B analysis under this directory. On explicit coordinator resume,
apply this exact registration and emit `DESCENT.md` with a plain Chinese owner section first
(no agent codenames), all task levels per cell, actual savings versus uniform under the same
SR rule, library agreement, then detailed audits. Do not run that analysis in the freeze turn.
No final result tables or selected emitter files are fabricated while outcomes are unavailable.
'''
    for name, data in [('protocol.json',json.dumps(p,indent=2,ensure_ascii=False,allow_nan=False)+'\n'), ('PROTOCOL.md',md)]:
        with (HERE/name).open('x') as f:
            f.write(data)
    seal = ''.join(f'{hashlib.sha256((HERE/name).read_bytes()).hexdigest()}  {name}\n' for name in names[:2])
    with (HERE/'PROTOCOL.sha256').open('x') as f:
        f.write(seal)
    print(json.dumps({'frozen_at_utc':p['frozen_at_utc'],'frozen_at_chicago':p['frozen_at_chicago'], 'net_loss_budgets':budgets, 'outcome_reads':0}))


if __name__ == '__main__':
    main()
