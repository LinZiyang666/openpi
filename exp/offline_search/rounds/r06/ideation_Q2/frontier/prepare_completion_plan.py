"""Provisional post-pilot frontier placement; specifications only, never launches."""
import ast
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
Q2 = HERE.parent
sys.path.insert(0, str(Q2))
import pilot_q2 as q2

RUNS = Path('/home/weiland/trace_runs/os_closed_loop')
PILOT = RUNS/'r06_p3_pilot/tables'
REPO = Q2.parents[4]


def dump(name, value):
    (HERE/name).write_text(json.dumps(q2.clean(value), indent=2, allow_nan=False)+'\n')


def pilot_placement():
    paths_file = HERE/'pilot_placement_paths.json'
    if paths_file.exists():
        paths = [Path(p) for p in json.loads(paths_file.read_text())]
    else:
        paths = sorted(PILOT.glob('*/episodes.csv'))
        dump('pilot_placement_paths.json', [str(p) for p in paths])
    rows, sources = [], []
    for path in paths:
        raw = list(csv.DictReader(path.open()))
        cell = path.parent.name.replace('_sp_', '_spatial_')
        model = cell.split('_')[0]
        c1 = q2.C1[model]
        groups = {}
        for r in raw:
            cohort = r['arm'].rsplit('_r', 1)[0].split(path.parent.name+'_')[-1]
            groups.setdefault(cohort, []).append(r)
        complete = len(raw) == 540 and len(groups) == 9 and all(len(g) == 60 for g in groups.values())
        if not complete:
            sources.append(dict(path=str(path), used=False, reason='not a complete 540-episode cell table'))
            continue
        sources.append(dict(path=str(path), used=True, sha256=q2.sha(path), episodes=len(raw), scope='PLACEMENT ONLY'))
        for c in ['A', 'dose125', 'dose25', 'dose50', 'P10', 'B']:
            g = groups[c]
            N, V, M = (sum(int(r[k]) for r in g) for k in ['decisions', 'anchors', 'misses'])
            rows.append(dict(cell=cell, cohort=c, n=len(g), init_clusters=len({(r['task_id'], r['init']) for r in g}),
                SR=sum(int(r['Y']) for r in g)/len(g), owner_IR=(c1*V+(1-c1)*M)/N,
                scope='PLACEMENT ONLY; not part of completed-run frontier'))
    q2.write_csv(HERE/'pilot_placement_only.csv', rows)
    dump('pilot_placement_provenance.json', sources)
    return rows


def base_ab_ids(cell):
    model, suite, size = cell.split('_')
    sh = 'sp' if suite == 'spatial' else 'l10'
    a = f'r05_x/r5x_g_{sh}_{size}_tail1u' if model == 'groot' else f'r05_ptail/r5t_p_{sh}_{size}_tail1uc'
    if cell == 'pi05_spatial_500':
        a = 'r04_blind/r4b3_p_sp_500_tail1uc'
    b = f'r06_paper/r6p1_c10_g_{sh}_{size}' if model == 'groot' else f'r05_q1/r5q1_c10_p_{sh}_{size}'
    return a, b


def emit_copy(source, name, evidence):
    old = evidence[source]['spec']
    spec = {k: copy.deepcopy(old[k]) for k in ['model', 'method', 'kwargs', 'plugin_args', 'cost_ledger', 'full_model', 'client_overrides', 'replan_steps', 'yaml_patch'] if k in old}
    spec.update(name=name, suite='spatial' if old['suite'] == 'libero_spatial' else 'l10', mode='plugin',
                full_model=True, cost_ledger=True, manifest='<RUN>/manifests/eval500.json')
    args = spec.setdefault('plugin_args', [])
    if '--os-fit-artifact' in args:
        args[args.index('--os-fit-artifact')+1] = f'<RUN>/fits/{name}.pkl'
    else:
        args += ['--os-fit-artifact', f'<RUN>/fits/{name}.pkl']
    if '--os-no-shadow-native' not in args:
        args.append('--os-no-shadow-native')
    spec.setdefault('client_overrides', {'replan_steps': 5})
    return spec


def verify_class(spec):
    module, cls = spec['method'].split(':')
    path = REPO/module if module.endswith('.py') else REPO/(module.replace('.', '/')+'.py')
    tree = ast.parse(path.read_text())
    if cls not in {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}:
        raise ValueError('class definition missing: '+spec['method'])
    if not spec['full_model'] or not spec['cost_ledger'] or '<RUN>' not in ' '.join(spec['plugin_args']):
        raise ValueError('bad deploy spec')
    return dict(class_source=str(path), sha256=q2.sha(path), validation='AST presence and emitter schema; no import, fit, emit, or rollout')


def main():
    data = json.loads((HERE/'frontier_data.json').read_text())
    evidence = json.loads((HERE/'source_evidence.json').read_text())
    pilot = pilot_placement()
    quality, qsource = q2.quality_rows()
    plans, ready, needs, verification, weights = [], [], [], [], []
    # Cost targets are deliberately provisional and will be frozen after all pilot cells arrive.
    designs = {
        'pi05_l10_50': ('B', [.25, .5, .75], [.35, .45], 2, 'cap4'),
        'pi05_l10_500': ('B', [.125, .25], [.18, .24], 2, 'K7_repeat'),
        'pi05_spatial_50': ('B', [.25, .5, .75], [.35, .45], 2, 'R3_quantile_repeat'),
        'pi05_spatial_500': ('A', [.03125, .0625], [.085, .105], 2, 'A15'),
        'groot_l10_50': ('B', [.25, .5, .75], [.35, .45], 2, 'cycle2'),
        'groot_l10_500': ('A', [.125, .25], [.13, .18], 2, 'cycle8'),
        'groot_spatial_50': ('B', [.125, .25], [.12, .20], 2, 'cycle3'),
        'groot_spatial_500': ('A', [], [.065, .09], 3, 'dense15'),
    }
    for cell, (baseline, doses, rhos, blocks, existing) in designs.items():
        a_id, b_id = base_ab_ids(cell)
        a = next(x for x in data['pooled_AB'] if x['cell'] == cell and x['label'] == 'A')
        b = next(x for x in data['pooled_AB'] if x['cell'] == cell and x['label'] == 'B')
        ref = next(x for x in data['gaps'] if x['cell'] == cell and x['reference_L'] == 10)
        p = [r for r in pilot if r['cell'] == cell]
        if p:
            p10 = next(r for r in p if r['cohort'] == 'P10')['SR']
            meets = [r['cohort'] for r in p if r['cohort'] in ['A', 'dose125', 'dose25', 'dose50'] and r['SR'] >= p10-.02]
            placement = 'Pilot within 2 pp at '+(', '.join(meets) if meets else 'no sub-P10 fixed dose')
            if cell == 'groot_spatial_50':
                placement += '; pilot optimistic relative to the full 500-init data, retain low and moderate budgets'
        else:
            placement = 'Pilot unavailable at this frozen snapshot; provisional positions from completed-run gap'
        for d in doses:
            name = 'r6q2_'+cell+f'_{baseline}_dose'+str(d).replace('.', 'p')
            base_ir = b['owner_IR'] if baseline == 'B' else a['owner_IR']
            predicted = base_ir+(.5-base_ir)*d
            config = dict(name=name, cell=cell, baseline=baseline, source_arm=b_id if baseline == 'B' else a_id,
                baseline_spec=evidence[b_id if baseline == 'B' else a_id]['spec']['method'],
                baseline_kwargs=evidence[b_id if baseline == 'B' else a_id]['spec']['kwargs'],
                deployment_adapter='NEEDS_CODE: shadow-free guard-preserving Bernoulli anchor overlay',
                dose=d, pre_guard=False, cache_commit_controls=10, policy_commit_controls=10,
                mandatory_B_guards=(baseline == 'B'), randomization_domain='Q2-extra-call-v1',
                full_model=True, shadow=False, resamples=0, episodes=500, manifest='<RUN>/manifests/eval500.json')
            needs.append(config)
            plans.append(dict(cell=cell, name=name, policy=f'{baseline} + extra anchor dose {d:g}', n=500, code_status='needs new deployment adapter',
                predicted_IR=predicted, predicted_IR_scope='constant-cadence forecast only; measured B/A cost interpolated to .5',
                source=config['source_arm'], comparison=('versus B and both policy references; isolate added calls' if baseline == 'B' else 'versus A, B and both policy references; isolate added calls'),
                placement=placement, hypothesis='More calls may close the remaining SR gap; no numerical SR prediction is identified'))
        qcell = cell.replace('_spatial_', '_sp_')
        qcopy = copy.deepcopy(quality)
        if blocks == 3:
            # Fifteen-control commitment: recompute anchor opportunities from actual library episode lengths.
            libpath = Path(qcopy[qcell, 0]['provenance'].split(';')[0])
            ts = np.load(libpath/'task_id.npy', mmap_mode='r')
            eps = np.load(libpath/'episode.npy', mmap_mode='r')
            for t in range(10):
                _, counts = np.unique(eps[ts == t], return_counts=True)
                qcopy[qcell, t]['a'] = float(np.mean(np.ceil(counts/3)))
        for rho in rhos:
            name = 'r6q2_'+cell+'_risk_rho'+str(rho).replace('.', 'p')
            rule = q2.allocation(qcopy, qcell, rho, False)
            if rule['clamped']:
                raise ValueError('planned target IR infeasible in library: '+name)
            config = dict(name=name, cell=cell, baseline='A', deployment_adapter='NEEDS_CODE: per-task episode-dose lottery without shadow',
                source_arm=a_id, baseline_spec=evidence[a_id]['spec']['method'], baseline_kwargs=evidence[a_id]['spec']['kwargs'],
                rho=rho, commitment_blocks=blocks, cache_commit_controls=5*blocks, policy_commit_controls=5*blocks,
                dose_knots=[0, .125, .25, .5, 1], task_mixture_weights=rule['weights'],
                risk='frozen Q2 state_LOEO fallback; replace only with pre-frozen Q1 score and provenance',
                quality_source=qsource, full_model=True, shadow=False, resamples=0, episodes=500,
                randomization_domain='Q2-episode-dose-v1', manifest='<RUN>/manifests/eval500.json')
            needs.append(config)
            for tr in rule['tasks']:
                weights.append(dict(cell=cell, arm=name, rho=rho, commitment_blocks=blocks, **tr, dose_weights=rule['weights'][tr['task']]))
            plans.append(dict(cell=cell, name=name, policy=f'risk lottery rho={rho:g}, {5*blocks} controls', n=500,
                code_status='needs new deployment adapter', predicted_IR=rule['predicted_IR'], predicted_IR_scope='library forecast; no SR guarantee',
                source=a_id, comparison='versus B, nearest uniform/periodic budget, and both pure references; allocation versus cost',
                placement=placement, hypothesis='Risk allocation may improve SR at a matched budget; no positive gain assumed'))
        existing_specs = []
        if existing == 'cap4':
            source = b_id
            name = 'r6q2_'+cell+'_B_cap4'
            spec = emit_copy(source, name, evidence)
            spec['plugin_args'] += ['--os-judge-cap', '4']
            existing_specs.append((source, spec, 'B + cap of four consecutive HIT requests', 'guard_only retains B guards; cap can force an early vision anchor; not a fixed anchor dose'))
        elif existing == 'K7_repeat':
            source = 'r05_b1/r5b_p_l10_500_hand'
            spec = emit_copy(source, 'r6q2_'+cell+'_K7_confirmation', evidence)
            existing_specs.append((source, spec, 'K7 tail confirmation', 'confirm the observed .906 @ .197 point; do not assume that winning replicate is its true SR'))
        elif existing == 'R3_quantile_repeat':
            source = 'r03_mx/r3mx_p_sp_awm_h70'
            spec = emit_copy(source, 'r6q2_'+cell+'_R3_h70_confirmation', evidence)
            existing_specs.append((source, spec, 'R3 confidence quantile h=.70 confirmation', 'existing .980 point is close to .986 L10; .70 targets HIT share, not SR'))
        elif existing == 'A15':
            spec = emit_copy(a_id, 'r6q2_'+cell+'_A15', evidence)
            spec['kwargs']['budget'] = 2
            existing_specs.append((a_id, spec, 'A cached commitment 15 controls', 'test below the cheap A10 frontier; no MISS and no policy-tail inference'))
        elif existing.startswith('cycle'):
            k = int(existing[5:])
            model, suite, size = cell.split('_')
            source = f'r05_q2/r5q2_g_{suite}_{size}_G10'
            spec = emit_copy(source, 'r6q2_'+cell+f'_CycleTail_k{k}', evidence)
            spec['kwargs']['cycle_k'] = k
            existing_specs.append((source, spec, f'CycleTail every {k} anchors, 10 controls', 'existing GR00T A-periodic controller; no B guards; includes a first-anchor policy call'))
        elif existing == 'dense15':
            for source, suffix, title in [
                ('r04_gblind/r4b3_g_sp_500_tail2u', 'A15_confirmation', 'A15 confirmation'),
                ('r04_gblind/r4b3_g_sp_500_ph2', 'phase_confirmation', 'phase-particle pure-cache confirmation')]:
                spec = emit_copy(source, 'r6q2_'+cell+'_'+suffix, evidence)
                existing_specs.append((source, spec, title, 'validate the low-IR region already above the policy point estimate'))
            source = 'r05_q2/r5q2_g_spatial_500_G10'
            spec = emit_copy(source, 'r6q2_'+cell+'_CycleTail15_k8', evidence)
            spec['kwargs'].update(cycle_k=8, tail_blocks=2)
            args = spec['plugin_args']
            args[args.index('--os-policy-tail-blocks')+1] = '2'
            existing_specs.append((source, spec, 'CycleTail every 8 anchors, 15 controls', 'compare low-dose committed policy rescue against A15; native L15 SR is unmeasured'))
        for source, spec, title, note in existing_specs:
            check = verify_class(spec)
            ready.append(spec)
            verification.append(dict(name=spec['name'], source_arm=source, **check))
            plans.append(dict(cell=cell, name=spec['name'], policy=title, n=500, code_status='existing code; emit spec supplied',
                predicted_IR=None, predicted_IR_scope='measure realized v/m; no unverified runtime forecast', source=source,
                comparison='versus source method, B and both policy references', placement=placement, hypothesis=note))
    counts = {cell: sum(p['cell'] == cell for p in plans) for cell in designs}
    assert max(counts.values()) <= 6 and len(plans) <= 48
    dump('completion_plan.json', dict(status='PROVISIONAL; finalize only after all 8 pilot cells are available',
        arms=plans, per_cell=counts, total_arms=len(plans), total_episodes=500*len(plans), existing_code=len(ready), needs_code=len(needs)))
    dump('emit_arms_existing.json', ready)
    dump('needs_code_configs.json', needs)
    dump('emit_specs_static_validation.json', verification)
    dump('eval500_manifest.json', dict(selected=[dict(task=t, init=i) for t in range(10) for i in range(50)]))
    q2.write_csv(HERE/'completion_plan.csv', plans)
    q2.write_csv(HERE/'completion_risk_allocations.csv', weights)
    write_report(plans, counts, ready, needs, pilot)
    print(json.dumps(dict(arms=len(plans), episodes=500*len(plans), existing=len(ready), needs_code=len(needs), per_cell=counts,
                          pilot_cells=sorted({r['cell'] for r in pilot}))))


def write_report(plans, counts, ready, needs, pilot):
    lines = ['# Provisional frontier-completion plan', '',
        '**Not scheduled or executed. Finalize after all eight pilot cells arrive.** This frozen plan contains '
        f'{len(plans)} configurations × 500 episodes = {500*len(plans):,} episodes; at most {max(counts.values())} configurations per cell. '
        f'{len(ready)} use existing code and have emit_arms input specs; {len(needs)} need a small deployment adapter. '
        'No profiler or controller code is built here. Existing full-run results remain the only frontier evidence.', '',
        '## Placement evidence, not frontier evidence', '',
        'Three complete profiler cell tables were available at the single frozen read; the path list and hashes are in `pilot_placement_paths.json` / `pilot_placement_provenance.json`. '
        'The other five cells use only existing 500-episode gaps for provisional placement. There is no polling. These 60 episodes/cohort have only 20 distinct task/init clusters; a favorable point is not a final success claim.', '',
        '| Cell | Dose cohort | Episodes / init clusters | SR | Owner IR |', '|---|---|---|---:|---:|']
    for r in pilot:
        lines.append(f'| {r["cell"]} | {r["cohort"]} | {r["n"]} / {r["init_clusters"]} | {r["SR"]:.3f} | {r["owner_IR"]:.3f} |')
    lines += ['', 'For pi05 l10-50 and spatial-50, no measured sub-P10 fixed dose is within 2 pp of that pilot’s P10 point; rho=.35/.45 targets the missing high-budget interval. '
        'GR00T spatial-50 is much more optimistic in the pilot than the complete 500-init record; retain both low and moderate budgets rather than concluding that A is already sufficient. '
        'For the dense libraries, look below/around B, not only above it: existing cache methods already approach or exceed pure-policy SR. '
        'No SR is extrapolated from these pilot point estimates. Placement costs use the request-denominated owner ledger to match this frontier; '
        'P3 P10 can be slightly above .5 because episode termination interrupts the two-request commitment. It is not silently substituted for native L=10.', '', '## Per-cell configurations', '']
    for cell in counts:
        lines += ['', f'### {cell} — {counts[cell]} × 500 episodes', '', '| Policy | Code | Cost placement | Paired comparison / hypothesis |', '|---|---|---|---|']
        for r in [r for r in plans if r['cell'] == cell]:
            ir = 'measure' if r['predicted_IR'] is None else f'{r["predicted_IR"]:.3f} forecast'
            lines.append(f'| {r["policy"]} | {r["code_status"]} | {ir} | {r["comparison"]}. {r["hypothesis"]} |')
    lines += ['', '## Exact serving recipes and code availability', '',
        '**B plus fixed additional dose d (new adapter):** at each genuine vision anchor, compute B’s original proposal/guard. '
        'If B mandates policy, call it. Otherwise call iff a private hash uniform variate is below d. Preserve B’s current cache/return hooks, 10-control policy lifecycle tail, retrieval state invalidation, and early looks. '
        'Do not inspect a shadow action to choose the source. Hash `(manifest SHA, task, original init, replicate, anchor index, "Q2-extra-call-v1")` as compact UTF-8 JSON, take the first 64 SHA256 bits big-endian divided by 2^64. '
        'Use a separate policy-noise RNG stream. This is an extra-call probability on B-cache opportunities, not total m; guard state feedback makes the cost forecast approximate. '
        'A+d uses the identical Bernoulli mechanism with A and no mandatory guards. Fixed d is an experimental placement parameter, not the eventual owner interface.', '',
        '**Risk target rho (new adapter):** use `../PREREG.md`’s library-only weights and 80-iteration budget solve, with the exact per-task dose-lottery weights emitted in `completion_risk_allocations.csv` / `needs_code_configs.json`. '
        'At episode start sample one of {0,.125,.25,.5,1} from those weights using the frozen `Q2-episode-dose-v1` hash domain; hold that Bernoulli anchor policy for the episode. '
        'No validation-success or LIBERO task-name threshold enters the allocation. The default is 10-control commitment. The GR00T spatial-500 low-cost extension explicitly uses 15 controls (H=16 permits three R=5 blocks), '
        'and its library anchor opportunities are recomputed as mean ceil(episode_decisions/3), not borrowed from the 10-control estimate. This extension is a new proposed policy version and has no P3-identifiable 15-control SR. '
        'The owner knob stays rho; alpha is an internal guard calibration parameter, not another user control. The fallback library risk has not demonstrated a portable SR guarantee. '
        'Q1 may replace it only before outcome inspection for the new arm, with a verified library/metric mapping; a score chosen using pilot outcomes needs independent final evaluation.', '',
        '**Existing code:** B is `q1_commit.judge:CommitJudge` (pi05) or `p1_groot_commit.judge:GrootCommitJudge` (GR00T). '
        '`--os-judge guard_only --os-judge-cap 4` retains guards and adds a consecutive-HIT cap; cap counts request slots including blind tails, can force early vision, and is not “every four anchors.” '
        'GR00T `q2_groot.judge:CycleTail(cycle_k=k, tail_blocks=1 or 2)` is an A-periodic controller with no B guards; it is not B+periodic rescue. '
        'Other supplied existing specs copy the tested R3 quantile / K7 configurations or change `BlindAWM` cached-tail budget from 1 to 2. '
        'Exact kwargs, client overrides, full_model, cost_ledger, and plugin arguments are in `emit_arms_existing.json`.', '',
        '**Two shortcuts are unavailable:** `--os-judge periodic:k` ignores force-MISS guard flags and uses a shared server decision clock in blind mode (`closed_loop/plugin.py:853,873`); it does not implement B+d. '
        'P3 v2 `blind_shadow=False, resample_p=0` still computes shadow policy at every vision anchor (`v2_engine.py:79`). '
        '`enabled=False` bypasses its profiling/randomization path. Thus “P3 fixed dose with no shadow” is not an existing deployable configuration, and no such emit spec is claimed here.', '',
        '## Pairing, validation, and finalization', '',
        'Each proposed arm evaluates tasks0–9 × inits0–49 once, using `eval500_manifest.json`, identical environment seed/reset and matching policy-noise seed where meaningful. '
        'Pair comparisons by task/init and preserve any repeated blocks as clusters. Existing B and policy runs are retrospective references; pairing IDs alone does not make historical harness/seed differences disappear. '
        'If strict contemporaneous baseline replication is required, substitute B/P10 replays for lower-priority proposed configurations within the six-per-cell cap rather than silently enlarging the campaign. '
        'Use 2 pp SR NI against **both separately reported L=5/L=10 references**, exact McNemar equality diagnostics, and a valid paired difference bound. '
        '500 episodes may not certify a 2 pp loss when discordance is high; retain “inconclusive” and report the point frontier plus intervals. The supplied plan does not promise a crossing or universal transfer.', '',
        'When all pilot cells are available, finalize once: retain the existing near-crossing-family confirmation in each cell; use pilot fixed-dose knots only to bracket the P10−.02 crossing; '
        'place two rho values within the bracket (at one-third and two-thirds of the bracket in measured IR, rounded to the nearest .01). '
        'If no sub-P10 dose reaches it, use the [.5-dose,P10] interval; if A already reaches it, use the [A,.125-dose] interval but retain an intermediate historical-gap check when pilot and full-run evidence disagree. '
        'Choose up to three extra-B doses whose constant-cadence forecast lies closest to those budgets and their midpoint, from {1/16,1/8,1/4,1/2,3/4}; deduplicate. '
        'Dense-library 15-control candidates remain a separately labeled low-cost extension. Freeze the final file/hash before its 500-episode outcomes; do not turn the pilot’s best seed/task into the final success estimate.', '',
        '## Artifacts and reproduction', '',
        '`emit_arms_existing.json` is the input-spec list for existing classes only. `<RUN>` placeholders must be replaced by the coordinator; no emit_arms invocation was made. '
        'Copy `eval500_manifest.json` to `<RUN>/manifests/eval500.json` in a future authorized experiment. `needs_code_configs.json` is a design specification, not executable emit input. '
        '`emit_specs_static_validation.json` records AST class checks and source hashes; runtime/fitting/rollout validation is unperformed.', '', '```bash',
        'taskset -c 18-21,62-65 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=\'\' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/offline_search/rounds/r06/ideation_Q2/frontier/prepare_completion_plan.py', '```']
    (HERE/'completion_plan.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
