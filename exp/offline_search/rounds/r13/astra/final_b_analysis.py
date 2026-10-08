"""Rebuild final B-only paper support. Run using the CPU command in HANDBACK.

Does not open test-A. Preserves the separately generated test-A marker section.
"""
from __future__ import annotations

import datetime
from collections import Counter

from final_common import *


def main():
    before = verify_frozen()
    gates = read(HERE / 'gates.json')
    mech = read(HERE / 'mechanism_numbers.json')
    validation = read(HERE / 'mechanism_validation.json')
    looks = pd.read_pickle(record(HERE / 'mechanism_replayed.pkl'))
    assert set(looks.source) <= {'pilot', 'devrec'}
    assert all(r.startswith(('r13_pilot_', 'r13_devrec_')) for r in looks.root.unique())
    a = looks[looks.source == 'pilot'].copy()
    arms = gates['arms']
    assert all(x['complete'] for x in arms)
    counts = Counter(x['method'] for x in arms)
    by_arm = {x['arm']: x for x in arms}
    # Reconcile the existing audited mechanism cache with the gate ledger.
    for name, z in a.groupby('arm'):
        t = by_arm[name]['totals']
        assert [len(z), int(z.call.sum()), z.ep.nunique()] == [t['V'], t['M'], t['episodes']]
        assert int(z.groupby('ep').N.first().sum()) == t['N']
    assert a.arm.nunique() == len(arms)

    budgets = []
    for x in arms:
        budgets.append(dict(cell=x['cell'], method=x['method'], target=x['target'], rep=x['replicate'],
                            arm=x['arm'], **x['totals'], budget_pass=x['budget_pass'],
                            host=','.join(x['observed_server_hosts'])))
    csv('final_b_budget.csv', budgets)
    budget_groups = []
    for method in counts:
        z = [r for r in budgets if r['method'] == method]
        budget_groups.append(dict(method=method, arms=len(z), passes=sum(r['budget_pass'] for r in z),
                                  min_error=min(r['error'] for r in z), max_error=max(r['error'] for r in z)))
    static = [r for r in budgets if r['method'] == 'distance_static_b']
    assert sum(not r['budget_pass'] for r in static) == 5

    pairs = []
    for p in gates['pairs']:
        d, q = (by_arm[p[k]] for k in ('distance_arm', 'periodic_arm'))
        de = {(e['task'], e['init']): e for e in d['episodes']}
        pe = {(e['task'], e['init']): e for e in q['episodes']}
        assert de.keys() == pe.keys()
        for task, init in sorted(de):
            pairs.append(dict(cell=p['cell'], suite=p['cell'].split('_')[1], task=task, init=init,
                              target=p['target'], rep=p['replicate'], matched=p['matched_budget_pass'],
                              success_d=de[task, init]['success_for_variance_only'],
                              success_p=pe[task, init]['success_for_variance_only']))
    assert len(pairs) == 2800
    csv('final_b_sr_pairs.csv', pairs)
    sr = {'all_nominal_pairs': paired_stats(pairs),
          'matched_only': paired_stats([r for r in pairs if r['matched']]),
          'per_cell': [], 'rep0_only': paired_stats([r for r in pairs if r['rep'] == 0])}
    for c in sorted({r['cell'] for r in pairs}):
        z = [r for r in pairs if r['cell'] == c]
        sr['per_cell'].append(dict(cell=c, matched_pairs=sum(r['matched'] for r in z), **paired_stats(z)))
    # Only four size-50 cells have both replicate blocks, at .32.
    sr['duplicate32_by_block'] = [dict(rep=rep, **paired_stats([
        r for r in pairs if r['cell'].endswith('_50') and r['target'] == .32 and r['rep'] == rep]))
        for rep in (0, 1)]
    csv('final_b_sr_summary.csv', [dict(group='all nominal', **sr['all_nominal_pairs']),
        dict(group='matched only', **sr['matched_only'])] +
        [dict(group=r['cell'], **r) for r in sr['per_cell']])

    cadence, cdfs = [], []
    for (cell, target, rep, method), z in a.groupby(['cell', 'target', 'rep', 'method'], sort=True):
        meta = dict(cell=cell, target=target, rep=int(rep), method=method)
        for kind, selector, column in [('any', z.call, 'gap'), ('knob_since_any', z.knob, 'gap'),
                                       ('knob_since_knob', z.knob, 'gap_knob')]:
            gap = z.loc[selector, column].dropna()
            controls_col = 'gap_knob_controls' if column == 'gap_knob' else 'gap_controls'
            gc = z.loc[selector, controls_col].dropna()
            row = dict(meta, kind=kind, **distribution(gap))
            cadence.append(row)
            cdfs += cdf_rows(gap, dict(meta, kind=kind, unit='real_looks'))
            cdfs += cdf_rows(gc, dict(meta, kind=kind, unit='controls'))
    csv('final_b_gap_summary.csv', cadence)
    csv('final_b_gap_cdf.csv', cdfs)
    ix = {(r['cell'], r['target'], r['rep'], r['method'], r['kind']): r for r in cadence}
    def gap(c, t, rep, method):
        return gap_text(ix.get((c, t, rep, method, 'any')))
    ctrows = []
    for c, t, rep in sorted({(r['cell'], r['target'], r['rep']) for r in cadence}):
        ctrows.append([cell_label(c), f'{t:.2f}', rep] +
                      [gap(c, t, rep, m) for m in ('distance_feedback', 'periodic', 'distance_static_b')])
    cadence_table = table(['Cell', 'Target', 'Rep', 'Feedback gap', 'Periodic gap', 'Static gap'], ctrows)
    paper = []
    for c in sorted({r['cell'] for r in cadence if r['cell'].endswith('_50')}):
        row = dict(cell=c, target=.32, rep=0)
        for short, m in [('D', 'distance_feedback'), ('P', 'periodic'), ('S', 'distance_static_b')]:
            b = next(r for r in budgets if r['cell'] == c and r['target'] == .32 and r['rep'] == 0 and r['method'] == m)
            row[f'{short}_ir'] = b['ir']
            row[f'{short}_gap_mean'] = ix[c, .32, 0, m, 'any']['mean']
            for q in ['p10', 'p50', 'p90']:
                row[f'{short}_gap_{q}'] = ix[c, .32, 0, m, 'any'][q]
        paper.append(row)
    csv('final_paper_table.csv', paper)
    paper_table = table(['Cell (.32, rep0)', 'IR D / P / S', 'D gap', 'P gap', 'S gap'], [
        [cell_label(r['cell']), ' / '.join(f'{r[f"{m}_ir"]:.3f}' for m in 'DPS')] +
        [gap(r['cell'], .32, 0, m) for m in ('distance_feedback', 'periodic', 'distance_static_b')]
        for r in paper])

    # Exact B rep0 .32 cadence and same-state placement headline.
    primary32 = a[(a.target == .32) & (a.rep == 0) & a.method.isin(['distance_feedback', 'periodic'])]
    pooled_gap32 = {m: distribution(z.loc[z.call, 'gap'].dropna()) for m, z in primary32.groupby('method')}
    eligible = primary32[~primary32.guard]
    union = int((eligible.replay_d | eligible.replay_p).sum())
    diff = int((eligible.replay_d ^ eligible.replay_p).sum())
    assert union == mech['headline']['replay32']['union']
    assert diff == mech['headline']['replay32']['symmetric_difference']
    predictors = mech['predictors']['models']
    growth = [r for r in mech['distance_growth'] if r['source'] == 'pilot' and r['method'] == 'distance_feedback']
    shadow = mech['conditional_shadow']['pooled']['risk10']
    h = mech['headline']['feedback32']
    unmatched = [r for r in gates['pairs'] if not r['matched_budget_pass']]
    result = dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  boundary='B only; test-A is computed by final_test_a.py separately',
                  method_arm_counts=dict(counts), budget_groups=budget_groups,
                  calibration=gates['calibration_complexity'], sr=sr, unmatched=unmatched,
                  pooled_gap32=pooled_gap32, placement32=dict(union=union, different=diff, fraction=diff/union),
                  predictors=predictors, growth=growth, shadow=mech['conditional_shadow'],
                  prior_mechanism_validation=validation, frozen_before=before)

    bgtable = table(['Method', 'Passed / arms', 'IR error range'], [
        [r['method'], f"{r['passes']}/{r['arms']}", f"[{r['min_error']:+.4f}, {r['max_error']:+.4f}]"] for r in budget_groups])
    statictable = table(['Cell', 'Target', 'Static IR', 'Error', 'Budget'], [
        [cell_label(r['cell']), f"{r['target']:.2f}", f"{r['ir']:.4f}", f"{r['error']:+.4f}", 'pass' if r['budget_pass'] else 'FAIL'] for r in static])
    srtable = table(['B cell', 'Pairs (matched)', 'D SR', 'P SR', 'D−P pp [95% cluster CI]'], [
        [cell_label(r['cell']), f"{r['pairs']} ({r['matched_pairs']})", pct(r['sr_d']), pct(r['sr_p']), pp_ci(r)] for r in sr['per_cell']])
    predtable = table(['Predictor', 'AUC', 'Accuracy'], [
        [m, f"{r['auc']:.3f}", pct(r['accuracy'])] for m, r in predictors.items()])
    shadowtable = table(['Cell', 'D-only error', 'P-only error', 'D−P [95% CI]'], [
        [cell_label(r['cell']), f"{r['distance_only_standardized_mean']:.3f}", f"{r['periodic_only_standardized_mean']:.3f}",
         f"{r['distance_minus_periodic']:+.4f} [{r['bootstrap_95ci'][0]:+.4f}, {r['bootstrap_95ci'][1]:+.4f}]"]
        for r in mech['conditional_shadow']['by_cell']])
    nominal, matched = sr['all_nominal_pairs'], sr['matched_only']
    mismatch_text = '; '.join(f"{cell_label(r['cell'])} target {r['target']:.2f} rep{r['replicate']}: ΔIR={r['distance_minus_periodic_ir']:+.4f}" for r in unmatched)
    mechanism_text = f"""Budget feedback makes distance-threshold calls converge to approximately the periodic call cadence; distance still advances or delays calls within that cadence. At .32 rep0, pooled complete call gaps are {gap_text(pooled_gap32['distance_feedback'])} (D) and {gap_text(pooled_gap32['periodic'])} (P). The full distributions differ, so “converge” describes their operating cadence, not equality of distributions or exact call times. Static S has no budget feedback and is not a matched-cost comparator when its budget gate fails.

The predecision threshold is `bias + (cv*V + cm*M − target*N)/(cm*width)`. Waiting replenishes budget and lowers the threshold even at constant distance. Identity-held-out probes give:

{predtable}

The folds group cell/task/init across doses and repeats. Gap+ledger explains much of the schedule; distance adds placement information conditional on that ledger. The last predictor nearly reconstructs the rule, not its benefit. Within-episode gap categories explain only {100*min(r['raw_within_episode_gap_R2'] for r in growth):.2f}–{100*max(r['raw_within_episode_gap_R2'] for r in growth):.2f}% of raw-distance variation; median within-episode Spearman correlations across D cells range from {min(r['episode_spearman']['p50'] for r in growth):.3f} to {max(r['episode_spearman']['p50'] for r in growth):.3f}. Thus distance is not a simple increasing clock since the last call. This is descriptive: triggering truncates long gaps, and neither the weak trend nor its sign is causal.

On the same recorded .32 rep0 states (both behavior paths, six cells), {diff}/{union} = {pct(diff/union)} of the union of non-guard selected positions are selected by only one rule. Each replay keeps its own ledger/call history, with recorded states/guards fixed. This approximately two-thirds difference is local placement, not different task stages or counterfactual SR. D also makes {pct(h['knob_shorter_than_periodic_min']/h['knob_postfirst_count'])} of post-startup knob calls sooner than P's minimum gap and leaves {h['cache_looks_beyond_periodic_max']} eligible cache looks at/beyond P's maximum.

Shadow action error at D-only versus P-only picks, conditional on cell × recorded behavior × task × actual gap, is {shadow['distance_only_standardized_mean']:.3f} versus {shadow['periodic_only_standardized_mean']:.3f}: Δ={shadow['distance_minus_periodic']:.4f} [{shadow['bootstrap_95ci'][0]:.4f}, {shadow['bootstrap_95ci'][1]:.4f}], or {100*(shadow['ratio']-1):.1f}% higher at D-only picks. There are {shadow['common_support_n']} common-support looks out of {shadow['n']} disagreement looks. The label is frozen-sigma-normalized motion MSE over the first ten controls, not action truth or causal call value; uncertainty uses {shadow['bootstrap_replicates']} physical-init cluster resamples. Spatial cells carry the positive signal; GR00T L10 reverses direction.

{shadowtable}"""
    q2text = f"""Across all {nominal['pairs']:,} paired B episode-runs, D SR={pct(nominal['sr_d'])}, P SR={pct(nominal['sr_p'])}; pooled D−P={pp_ci(nominal)} percentage points. This is the complete nominal-dose descriptive pool, including the two unmatched cost points. There are {nominal['original_init_clusters']} distinct suite/task/init clusters, not {nominal['pairs']:,} independent initial states.

Restricting by the frozen, outcome-blind `|ΔIR|≤.01` budget label gives {matched['pairs']:,} pairs and D−P={pp_ci(matched)} pp. Unmatched points remain visible: {mismatch_text}. Both intervals include zero; the data do not distinguish SR at approximately matched budgets. This is not a formal equivalence/noninferiority result, and exact equal-IR performance is not identified by a nominal-dose comparison.

{srtable}

Intervals: {BOOT:,} paired original suite/task/init cluster resamples within fixed suite/task, seed {SEED}; carry all models, sizes, doses and repeats together, recomputing the equally weighted episode-pair mean. Per-cell results are descriptive, with no subgroup selection. No ordinary binomial/McNemar test is used on repeated init observations. Rep0-only pooled sensitivity: {pp_ci(sr['rep0_only'])} pp. The shared size-50/.32 subset gives rep0 {pp_ci(sr['duplicate32_by_block'][0])} and rep1 {pp_ci(sr['duplicate32_by_block'][1])} pp. These intervals are conditional on the recorded blocks; two repeats cannot establish future-block variance. The larger frozen A interpolation/precision plan was superseded by D5; no four- or two-block power claim is carried into the single-block lean confirmation."""
    calibration_text = """D needs a distance scale (two scalars per cell), a bias per cell/dose, a selected shared feedback width, and the running cost ledger. In the full frozen design this is 49 scalars. Its calibration used B trajectories where available and library-held-out transfer for GR00T L10-200 and size-500 (which lacks B pilot validation). P needs one cap per cell/dose, 32 across the full frozen design, from the existing whole-episode-held-out library ScheduleReplay solver; it needs no distance signal or online budget ledger. S uses eight B-calibrated thresholds on the same calibration paths as D, without feedback. These are configuration counts, not completed-arm counts; P is cheaper in calibration/state requirements, with no new latency benchmark claimed."""
    existing = {}
    for name in ('ANALYSIS.md', 'REPORT.md'):
        path = HERE / name
        if path.exists() and '<!-- TEST_A_MEASUREMENT_BEGIN -->' in path.read_text():
            existing[name] = path.read_text().split('<!-- TEST_A_MEASUREMENT_BEGIN -->')[1].split('<!-- TEST_A_MEASUREMENT_END -->')[0].strip()
    pending = 'Test-A measurement: PENDING. Run `final_test_a.py` to inventory the fixed lean set and supplementary arms. No test-A tuning or selection is permitted.'
    pi_l10 = [r for r in budgets if r['method'] == 'periodic' and r['cell'] == 'pi05_l10_50']
    analysis = f"""# R13 final analysis

The paper conclusion is: **with budget (cost) feedback, distance-threshold calls converge to almost the same cadence as periodic calls; distance mainly reorders calls within the budget-set cadence, with no detectable pooled matched-budget SR advantage. The simpler periodic schedule suffices in these measured settings.** B supplies the developmental evidence; the separate test-A measurement status is recorded below.

D5 (2026-10-04 20:38 CDT) ends the large grid and supersedes earlier scheduling/release language in PLAN, GATES and MECHANISM. This report is measurement and synthesis, not a design round. D=distance_feedback, P=periodic, S=distance_static_b without feedback. All preceding-test evidence below is B only.

**Q1: budget delivery and calibration**

The audited file contains {len(arms)} completed arms: {counts['distance_feedback']} D, {counts['periodic']} P and {counts['distance_static_b']} S. **The request's “32 primary B D arms” is a count mismatch: the actual completed D denominator is 28, all passing.** There are 32 periodic cap configurations in the full frozen grid, not 32 completed D arms. Target delivery uses `|IR−target|≤.02`, and paired matching separately uses `|IR_D−IR_P|≤.01`.

{bgtable}

IR is pooled owner cost / pooled decision slots: GR00T `(.148 V+.852 M)/N`, π0.5 `(.152 V+.848 M)/N`; V counts real looks and M all policy calls including mandatory guards. It is not mean episode IR or wall-clock latency. S misses {sum(not r['budget_pass'] for r in static)}/{len(static)} targets, both under and over; B recalibration alone does not stabilize closed-loop spending.

{statictable}

π0.5 P slightly underspends in L10-50 (signed IR errors {min(r['error'] for r in pi_l10):+.4f} to {max(r['error'] for r in pi_l10):+.4f}); Spatial is near target or slightly over. The direction is not uniform across all π0.5 cells. Exact errors and host records are in `final_b_budget.csv`. The mandatory-guard rate can shift a static cap's realized cost; D corrects the ledger online.

{calibration_text}

Pilot hosts: all GR00T and π0.5 L10-50 ran on local 4090; π0.5 L10-200 and Spatial-50 ran on H100. D2 authorizes the local π0.5 pilot; the obsolete GATES host hold is not a current verdict. All evaluated primary arms pass target delivery and D engagement; mechanics remain as audited in gates.json. The distinct paired cost misses are retained under Q2.

**Headline: cadence convergence, with local placement differences**

{mechanism_text}

Complete between-call gaps below use real observations: mean [p10/median/p90], call-weighted, startup excluded; terminal incomplete intervals are right censored and excluded. All calls include guards. On these B traces the look spacing is ten controls; the CSV separately supplies directly recorded control gaps. This table preserves every observed cell/dose/rep, including the no-feedback diagnostic where run; a dash is unmeasured, never imputed.

{cadence_table}

Paper table (the fixed size-50/.32 rep0 subset):

{paper_table}

Figure source: `final_b_gap_cdf.csv`, exact ECDFs at every integer gap with denominator and count, faceted by cell/target/rep/method; select kind=`any`, unit=`real_looks` for the table's distribution. `knob_since_any` and `knob_since_knob` expose the guard contribution. `final_b_gap_summary.csv` includes SD and maxima. S comparisons at failed budgets diagnose loss of cost control; they do not isolate a causal cadence effect of feedback.

**Q2: paired B success at comparable budget**

{q2text}

**Test-A: fixed measurement only**

<!-- TEST_A_MEASUREMENT_BEGIN -->

{existing.get('ANALYSIS.md', pending)}

<!-- TEST_A_MEASUREMENT_END -->

**Reproducibility and limits**

`final_b_analysis.py` derives all B tables and numbers from `gates.json`, `mechanism_numbers.json` and the B-only mechanism cache; it reconciles per-arm looks/calls/episodes/N to the gate ledger. Existing mechanism probes and shadows are reused as audited, without refitting. `final_b_numbers.json` records inputs, code hashes, cluster intervals and the frozen-file check. `final/HANDBACK.md` gives exact CPU commands. B pilot outcomes support a developmental comparison; test-A supplies a separate measurement confirmation as its fixed arms complete. No A result enters calibration, dose/cell choice, or the B estimates.
"""
    report = f"""# 给负责人的结论

加入成本反馈后，距离阈值调用会收敛到与周期调用几乎相同的节奏。距离主要是在预算决定的节奏内提前或推迟调用；在实际预算接近时，合并成功率没有可辨认的改善，因此现有证据支持采用更省事的周期方案。开发集的 {nominal['pairs']:,} 个配对结果中，距离方案平均高 {100*nominal['difference']:.2f} 个百分点，置信区间跨过零；这不等于证明两者完全等效。最终测试进度见文末，只作测量确认。

**Technical evidence (B only)**

Budget delivery: {counts['distance_feedback']}/{counts['distance_feedback']} observed feedback arms pass the ±.02 IR gate; static B recalibration misses {sum(not r['budget_pass'] for r in static)}/{len(static)}, in both directions. All {counts['periodic']} periodic arms pass; π0.5 L10 mildly underspends, while Spatial does not show uniform underspend. The requested denominator of 32 is not present in gates.json: there are {len(arms)} B arms, comprising {counts['distance_feedback']} D + {counts['periodic']} P + {counts['distance_static_b']} S.

{calibration_text}

Compact paper table: all-call gap mean [p10/median/p90] in real looks, excluding startup and censored terminal intervals; D=feedback, P=periodic, S=static threshold without feedback. IR uses the frozen pooled owner ledger. S is not matched cost where its gate fails.

{paper_table}

Across .32 rep0 B cells, D/P pooled mean gaps are {pooled_gap32['distance_feedback']['mean']:.2f}/{pooled_gap32['periodic']['mean']:.2f} looks. Held-out call prediction accuracy is {pct(predictors['gap']['accuracy'])} from gap, {pct(predictors['gap_ledger']['accuracy'])} with the ledger, and {pct(predictors['gap_ledger_distance_residual']['accuracy'])} with residual distance added. Within-episode gap explains only {100*min(r['raw_within_episode_gap_R2'] for r in growth):.2f}–{100*max(r['raw_within_episode_gap_R2'] for r in growth):.2f}% of distance variance. Distance is not a simple time-since-call clock. Yet {pct(diff/union)} of the same-state selected-position union differs at .32: similar cadence allows different exact placements. Shadow error at D-only/P-only picks is {shadow['distance_only_standardized_mean']:.3f}/{shadow['periodic_only_standardized_mean']:.3f}, adjusted difference {shadow['distance_minus_periodic']:.4f} [{shadow['bootstrap_95ci'][0]:.4f}, {shadow['bootstrap_95ci'][1]:.4f}]; heterogeneous across cells and not evidence of causal SR benefit.

Paired SR: all {nominal['pairs']:,} nominal-dose pairs yield D−P {pp_ci(nominal)} pp; the {matched['pairs']:,} pairs passing the frozen paired-cost label yield {pp_ci(matched)} pp. Both are statistically indistinguishable at this precision, not an equivalence proof. All-pair pooling retains {mismatch_text}. Intervals use {BOOT:,} suite/task-stratified original-init cluster resamples, preserving repeated models/sizes/doses/blocks. Per-cell descriptive results:

{srtable}

Pilot host exception: π0.5 L10-50 ran on local 4090 under D2; its other B cells used H100, and GR00T used local 4090. D5 supersedes the large-grid plan. `ANALYSIS.md` gives every B cell/dose/rep gap distribution and sensitivity; `final_paper_table.csv` and `final_b_gap_cdf.csv` are ready for tables/figures. “Periodic suffices” is the practical conclusion at measured precision, with cheaper calibration/state requirements, not a newly measured runtime advantage.

**Test-A measurement — no tuning or selection**

<!-- TEST_A_MEASUREMENT_BEGIN -->

{existing.get('REPORT.md', pending)}

<!-- TEST_A_MEASUREMENT_END -->
"""
    write('ANALYSIS.md', analysis)
    write('REPORT.md', report)
    result['frozen_after'] = verify_frozen()
    result['input_files'] = INPUTS
    result['scripts'] = {n: sha(HERE / n) for n in ('final_common.py', 'final_b_analysis.py')}
    dump('final_b_numbers.json', result)
    print(json.dumps(clean(dict(counts=counts, sr=sr, pooled_gap32=pooled_gap32, frozen=result['frozen_after'])), indent=2))


if __name__ == '__main__':
    main()
