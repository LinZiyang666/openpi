"""Batches for the new subset seeds: pi0.5 / GR00T LIBERO, success-filtered library, sizes 50/100/200, servers on the H100.

Variants: full step (R15, servers restart per arm) / reduced K1 (R16 residents) x long execution segment on (as
q3_port_1020: --os-blind --os-policy-tail --os-policy-tail-blocks 1) / off (as longseg_off_1022: without those three).
Everything else as there: Q3 --os-judge stall_only, R15 TaskDescentKnob, rollout seed, eval500 A inits, K, replicas,
topologies, ports. Fits, Q3 and periods are each seed's own (fit.py; on and off share them, owner 10-22).
  R15  fa_x<d>_<pi|gr>_f_<on|lo>_<o|l|k1..k4>: one unit per model x seed x segment, one root per cell (_l50 .. _s200),
       SERVER_HOST=h100 (longseg_off_1022.R15_ENV); a model's R15 units share ports: one at a time
  R16  fa_x34_<pi|gr>_r_<o|l|k1..k4>: one resident group per model holding every seed x size x suite x on/off entry
       (24 at knob off; each entry carries its own fit and plugin arguments); residents on the H100
       (longseg_off_1022.runner), inputs synced by prepare
<d> is the seed's last digit (20261003 -> x3). Arm names carry seed and segment; R16 rows also carry subset_seed and
long_segment, and the analysis keys on both.

  build                                   knob-off units: QUEUE.json, commands/{prepare,launch}_fa_x*.sh
  levels --model M --mode full|reduced    after knob off: per seed x segment, off vs the same-card same-step pure
                                          reference (formal_a_queue.classify); cells below it get ir45/ir40/ir32/ir25
                                          with the seed's periods (R15: a unit per seed x segment; R16: one group)
  rounds --model M --mode full|reduced    step 2 as descent rounds instead of levels (owner 10-24): units k1..k4
                                          (WAITING; R15 per seed x segment, R16 one group). Round r runs LEVELS[r-1]
                                          only for the tasks of each tuned cell that passed every earlier level (at
                                          most two fewer successes than the pure reference on the same 50 inits); a
                                          cell without such tasks is dropped, an empty round is SKIP_NO_TASKS. The
                                          frozen levels and final SR/IR equal the full sweep's (formal_a_analysis.choose).
                                          R15 arms run a task-subset manifest (arms.json `tasks`), R16 entries
                                          task-subset pairs.
  repair --name N                         a finished unit (knob off, round or repair) whose accepted episodes include
                                          failures without a single decision row (a worker failure, never served; the
                                          loaders read them as missing): unit <N>r reruns exactly those pairs, each in
                                          a copy of its arm/entry (same cell, level, fit, plugin and resident settings).
                                          evaluate, build_round and results merge a repair into the unit it repairs.
  dispatch prepare|launch --name N        coordinator entry points (the commands/ scripts); on a WAITING round it first
                                          defines the round from the finished rounds 1..r-1 (ValueError until then)
References (H100): segment off = longseg_off_1022 fa_lo_*_p5_q (C5); segment on = pi0.5 full the formal-A H100 pure A
(fa_pi_f_p10 / fa_pi_f_ps, formal_a_queue.references), GR00T full and both reduced longseg_off_1022's
fa_lo_{gr_f,pi_r,gr_r}_p_q (C10, built 10-22, not run yet: run them before levels).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import subprocess
from unittest.mock import patch

from exp.r17_integration.ops.immutable_release import nvme_dir
from exp.offline_search.rounds.r17.filter_failed_knob.q3_port_1020 import build as q3
from exp.offline_search.rounds.r17.filter_failed_knob.longseg_off_1022 import build as lo
from . import fit as F
from .fit import read, dump, LEVELS

HERE = Path(__file__).resolve().parent
INDEX = HERE/'QUEUE.json'
COMMANDS = q3.COMMANDS
MODULE = 'exp.offline_search.rounds.r17.seeds_1023.vla'
DEFS = F.ROOT/'definitions'
SEGMENTS = ('on', 'off')
SEG = {'on': 'on', 'off': 'lo'}
M = {'pi05': 'pi', 'groot': 'gr'}
STAGES = {'off': 'o', 'levels': 'l', **{f'k{r}': f'k{r}' for r in range(1, len(LEVELS)+1)}}  # k<r>: descent round r
ORDER = tuple(LEVELS)
ORDER_ALL = ('off', *ORDER)
PAIRS = q3.PAIRS
BOTH = 'x' + ''.join(str(s)[-1] for s in F.SEEDS)  # x34: the R16 groups hold both seeds


def tag(seed):
    return 'x' + str(seed)[-1]


def r15_name(model, seed, segment, stage):
    return f'fa_{tag(seed)}_{M[model]}_f_{SEG[segment]}_{STAGES[stage]}'


def r16_name(model, stage):
    return f'fa_{BOTH}_{M[model]}_r_{STAGES[stage]}'


def round_name(model, mode, r, seed=None, segment=None):
    return r15_name(model, seed, segment, f'k{r}') if mode == 'full' else r16_name(model, f'k{r}')


def segment_args(segment, args):
    """The q3_port_1020 plugin arguments (long segment on), or longseg_off_1022's (the three flags removed)."""
    return args if segment == 'on' else lo.segment_off(args)


def cell_name(model, row):
    return f'{model}_{"l10" if row["suite"] == "libero_10" else "spatial"}_{row["size"]}'


# ------------------------------------------------------------------------------------- R15 (full step)
def r15_unit(model, seed, segment, stage, cells, levels, tasks=None, *, name=None, pairs=None):
    """tasks {cell: task indices}: only those tasks (a descent round), as a manifest of their 50 inits each and
    `tasks` in arms.json (r16.analyze.load leaves the others unscheduled); default all ten. pairs {cell: {level:
    [[task, init], ...]}}: only those levels and pairs (a repair; `pairs` in arms.json). name: default r15_name."""
    from exp.offline_search.rounds.r15.descent import prepare as p
    d, unit = F.seed_dir(seed), name or r15_name(model, seed, segment, stage)
    tops = q3.topologies(model, 'filtered'); jobs = []
    for cell in cells:
        root = q3.EVAL/f'{unit}_{q3.short(cell)}'; q3.fresh(root); nvme_dir(root, exist_ok=True)  # real dir on /nvme
        if (root/'arms.json').exists():
            (root/'arms.json').unlink()
        planned = sorted(tasks[cell]) if tasks else list(range(10))
        if pairs:
            manifests = {level: root/f'eval_{level}.json' for level in pairs[cell]}
            for level, path in manifests.items():
                p.write_json(path, sorted(map(list, pairs[cell][level])))
        elif tasks:
            manifests = dict.fromkeys(levels, root/'eval_tasks.json')
            p.write_json(root/'eval_tasks.json', [pr for pr in read(lo.MANIFEST) if pr[0] in planned])
        else:
            manifests = dict.fromkeys(levels, root/'eval500.json'); (root/'eval500.json').write_bytes(lo.MANIFEST.read_bytes())
        specs = []
        for level, manifest in manifests.items():
            spec = q3.r15_spec(cell, f'{root.name}_{level}', str(F.fit_path(d, cell, level)), F.knob_kwargs(d, cell, level), manifest)
            spec['plugin_args'] = segment_args(segment, spec['plugin_args'])
            specs.append(spec)
        p.emit(root, cell, specs, list(manifests), pool='A', final_root=root)
        if tasks or pairs:
            rows = [dict(row, **({'pairs': read(row['manifest'])} if pairs else {'tasks': planned}), episodes=len(read(row['manifest'])))
                    for row in read(root/'arms.json')]
            p.write_json(root/'arms.json', rows)
            p.write_json(root/'ROOT.json', dict(p.seal(root, final_root=root, pool='A', cell=cell), episodes_per_arm=
                                                {r['arm']: r['episodes'] for r in rows} if pairs else 50*len(planned)))
        top = copy.deepcopy(tops[q3.short(cell)])
        top['env'].update(lo.R15_ENV[model]); top.update(gpu='H100', replicas=len(top['env']['PORTS'].split(',')))
        jobs.append(dict(name=root.name, run=str(root), model=model, mode='full', arm='filtered', pipeline='R15', stage=stage,
                         cell=cell, subset_seed=seed, long_segment=segment, episodes=sum(len(read(m)) for m in manifests.values()),
                         levels=list(manifests), **({'tasks': planned} if tasks else {}), **({'pairs': pairs[cell]} if pairs else {}),
                         gpu='H100', replicas=top['replicas'], workers='central scheduler: a100/timan107/timan1', topology=top,
                         server_tree='/data/oscl_h100/openpi (h100)'))
    return dict(name=unit, model=model, mode='full', arm='filtered', stage=stage, pipeline='R15', subset_seed=seed,
                long_segment=segment, gpu='H100', replicas=jobs[0]['replicas'], episodes=sum(j['episodes'] for j in jobs),
                jobs=jobs, status='BUILT')


# --------------------------------------------------------------------------------- R16 (reduced, K1)
def r16_entry(group, arm, src, fit, *, seed, segment, cell, level, tasks=None, pairs=None):
    """tasks: plan only those tasks' pairs (a descent round), pairs: only those (a repair); the manifest stays the 500
    A pairs."""
    d = DEFS/group/arm; d.mkdir(parents=True, exist_ok=True)
    (d/'config.yaml').write_bytes(Path(src['yaml']).read_bytes())
    (d/'eval500.json').write_bytes(Path(src['manifest']).read_bytes())
    digest = q3.sha(fit)
    dump(d/'ROOT.json', dict(definitions_sha256={n: q3.sha(d/n) for n in ('config.yaml', 'eval500.json')}, fit_sha256={str(fit): digest},
                             init_pool='A', recording=False, source_arm=src['arm'], seed_split_1023=True))
    row = copy.deepcopy(src)
    for k in ('matrix', 'remote_yaml', 'remote_matrix', 'yaml_patch'):
        row.pop(k, None)
    pairs = sorted(map(list, pairs)) if pairs else [pr for pr in PAIRS if tasks is None or pr[0] in tasks]
    row.update(arm=arm, method=F.KNOB_SPEC, kwargs=F.knob_kwargs(F.seed_dir(seed), cell, level),
               plugin_args=segment_args(segment, q3.knob_args(src, str(fit))), yaml=str(d/'config.yaml'),
               yaml_sha256=q3.sha(d/'config.yaml'), manifest=str(d/'eval500.json'), judge='stall_only', full_model=True,
               episodes=len(pairs), stall_gate='q3', q3_artifact=str(F.q3.q3_path(cell, F.seed_dir(seed))), source_off_arm=src['arm'],
               r14_descent_level='task_descent', subset_seed=seed, long_segment=segment, intended_host='h100')
    return dict(row=row, source_root=str(d), source_seal_sha256=q3.sha(d/'ROOT.json'), fit_sha256=digest,
                yaml_sha256=q3.sha(d/'config.yaml'), pairs=copy.deepcopy(pairs), dispatch_pairs=copy.deepcopy(pairs),
                formal_grid_level=level, source_off_arm=src['arm'])


def r16_group(model, stage, cells_by, levels, tasks=None, *, name=None, pairs=None):
    """One resident group: cells_by {(seed, segment): [cells]}, every cell at every level; tasks {(seed, segment,
    cell): task indices} plans only those (a descent round); pairs {(seed, segment, cell, level): [[task, init], ...]}
    only those entries and pairs (a repair). name: default r16_name."""
    from exp.offline_search.rounds.r16 import conductor as rc
    name = name or r16_name(model, stage)
    source, offs = q3.source_rows(model)
    entries = []
    for (seed, segment), cells in sorted(cells_by.items()):
        for cell in cells:
            for level in levels:
                if pairs and (seed, segment, cell, level) not in pairs:
                    continue
                arm = f'{name}_{tag(seed)}_{SEG[segment]}_{q3.short(cell)}_{level}'
                entries.append(r16_entry(name, arm, offs[cell]['row'], F.fit_path(F.seed_dir(seed), cell, level),
                                         seed=seed, segment=segment, cell=cell, level=level,
                                         tasks=tasks[seed, segment, cell] if tasks else None,
                                         pairs=pairs[seed, segment, cell, level] if pairs else None))
    root = q3.EVAL/name; q3.fresh(root)
    group = {k: v for k, v in source.items() if k not in ('entries', 'definitions_sha256', 'group_sha256')}
    group.update(root=str(root), entries=entries, phase='test', references=None, formal_test_A=True, k1ref=False,
                 original_episode_uids=False, yaml_id_is_original_arm=False, intended_server_host='h100',
                 seed_split_1023=dict(stage=stage, levels=list(levels),
                                      cells={f'{tag(s)}_{SEG[g]}': c for (s, g), c in sorted(cells_by.items())},
                                      **({'tasks': {f'{tag(s)}_{SEG[g]}_{q3.short(c)}': t for (s, g, c), t in sorted(tasks.items())}}
                                         if tasks else {})))
    files = sorted(p for p in (DEFS/name).rglob('*') if p.is_file())
    group['definitions_sha256'] = {**{str(p): q3.sha(p) for p in files},
                                   **{e['row']['plugin_args'][e['row']['plugin_args'].index('--os-fit-artifact')+1]: e['fit_sha256'] for e in entries}}
    group['group_sha256'] = hashlib.sha256(rc.canonical(group)).hexdigest()
    nvme_dir(root, exist_ok=True)  # real dir on /nvme; the group root keeps its /data address
    dump(root/'group.json', group)
    dump(root/'ROOT.json', dict(group_sha256=group['group_sha256'], definitions_sha256=group['definitions_sha256']))
    suites = {e['row']['suite'] for e in entries}
    split = None if model == 'pi05' else ('3,1' if suites == {'libero_10', 'libero_spatial'} else
                                         '4,0' if suites == {'libero_10'} else '0,4')
    job = dict(name=name, run=str(root), group=str(root/'group.json'), model=model, pipeline='R16',
               episodes=sum(len(e['dispatch_pairs']) for e in entries), port=lo.R16[model]['port'], ports=lo.R16[model]['ports'],
               groot_suite_servers=split, replicas=4 if model == 'groot' else 3, gpu='H100',
               server_tree='/data/oscl_h100/openpi_r16_q3_server (h100)')
    return dict(name=name, model=model, mode='reduced', arm='filtered', stage=stage, pipeline='R16', gpu='H100',
                replicas=job['replicas'], episodes=job['episodes'], jobs=[job], status='BUILT')


def sync(group_path):
    """prepare (R16): longseg_off_1022's H100 sync (code, links, inputs) with ROOT as one more /home alias there."""
    from exp.offline_search.rounds.r17.filter_failed_knob.longseg_off_1022 import h100
    native = h100.inputs
    def inputs(group):
        with patch.object(h100, 'FITS', F.ROOT), patch.object(h100, 'DEFS_ROOT', F.ROOT):
            items = native(group)
        return [(source, rels, 'lo_1022/r17_seed_split/' if source == F.ROOT else dest) for source, rels, dest in items]
    with patch.object(h100, 'inputs', inputs), patch.object(h100, 'LINKS', dict(h100.LINKS, **{str(F.ROOT): h100.LO + '/r17_seed_split'})):
        h100.sync(group_path)


# ---------------------------------------------------------------------------------- index, commands
def put(units):
    index = read(INDEX) if INDEX.exists() else dict(units=[])
    names = {u['name'] for u in units}
    index['units'] = [u for u in index['units'] if u['name'] not in names] + units
    dump(INDEX, index)
    for u in units:
        commands(u)


def describe(u):
    model = {'pi05': 'pi0.5', 'groot': 'GR00T'}[u['model']]
    steps = 'full step' if u['mode'] == 'full' else 'reduced K1'
    what = {'off': 'knob off', 'levels': 'levels ir45/40/32/25'}.get(u['stage']) or (
        f'descent round {u.get("round")}: {u.get("level")} for the tasks that passed every earlier level' if u['stage'] == 'round'
        else f'repair of {u.get("repairs")}: its accepted episodes without a decision row, again')
    episodes = 'episodes set when defined (prepare)' if u['episodes'] is None else f'{u["episodes"]} episodes'
    if u['pipeline'] == 'R15':
        lib, seg = f'subset seed {u["subset_seed"]}', f'long segment {u["long_segment"]}'
        server = 'H100 servers (/data/oscl_h100/openpi)'
    else:
        lib, seg = 'subset seeds ' + ' and '.join(map(str, F.SEEDS)), 'long segment on and off'
        server = 'H100 residents (/data/oscl_h100/openpi_r16_q3_server)'
    return [f'New subset seeds (seeds_1023): success-filtered library ({lib}) . {model} . LIBERO . {steps} . {seg} . {what} . {episodes}',
            f'{server}; {u["replicas"]} {model} replicas; Q3 --os-judge stall_only. See seeds_1023/vla/build.py.']


def commands(u):
    if u['pipeline'] == 'R15' or not u['jobs']:  # R16 too while a round is undefined: dispatch defines it first
        for target in [u['name']] + [j['name'] for j in u['jobs']]:
            for action, prefix in (('prepare', q3.PREFIX), ('launch', q3.LAUNCH_PREFIX if u['pipeline'] == 'R15' else q3.PREFIX)):
                q3.script(COMMANDS/f'{action}_{target}.sh', [lo.WORKERS] * (u['pipeline'] == 'R15') +
                          [f'{prefix} -m {MODULE}.build dispatch {action} --name {target}'], describe(u))
    else:
        j = u['jobs'][0]
        q3.script(COMMANDS/f'prepare_{u["name"]}.sh', [f'{q3.PREFIX} -m {MODULE}.build dispatch prepare --name {u["name"]}'], describe(u))
        split = f' --groot-suite-servers {j["groot_suite_servers"]}' if j['groot_suite_servers'] else ''
        q3.script(COMMANDS/f'launch_{u["name"]}.sh', [f'{q3.LAUNCH_PREFIX} -m {lo.MODULE}.runner run --group {j["group"]} '
                  f'--agent-island /scratch/zixuans8/openpi_r16_conductor_v1 --port {j["port"]} --ports {j["ports"]}{split} '
                  '--no-local-workers'], describe(u))


def level_scripts():
    for model in F.MODELS:
        for mode in ('full', 'reduced'):
            target = f'fa_{BOTH}_{M[model]}_{mode[0]}'
            q3.script(COMMANDS/f'build_levels_{target}.sh', [f'{q3.PREFIX} -m {MODULE}.build levels --model {model} --mode {mode}'],
                      [f'New subset seeds: define the {mode}-step levels ir45/40/32/25 of {model} for the cells whose knob-off cohort',
                       'is below the same-card same-step pure reference. Run after those knob-off units/group. CPU only.'])


def lookup(target):
    for u in read(INDEX)['units']:
        if u['name'] == target:
            return u, u['jobs']
        for j in u['jobs']:
            if j['name'] == target:
                return u, [j]
    raise KeyError(target)


def dispatch(action, target):
    u, jobs = lookup(target)
    fresh = u['stage'] == 'round' and u['status'] == 'WAITING'
    if fresh:
        jobs = build_round(target)['jobs']
    elif u['stage'] == 'round' and not jobs:
        print('SKIP_NO_TASKS', target, flush=True)
    for j in jobs:
        if j['pipeline'] == 'R15':
            q3.r15_run(j, action)
        elif action == 'prepare' or fresh:  # a group defined by this launch is synced first
            sync(j['group'])
        if j['pipeline'] == 'R16' and action == 'launch':
            subprocess.run(['bash', str(COMMANDS/f'launch_{j["name"]}.sh')], check=True)


# ------------------------------------------------------------------------------------------ analysis
def references(model, mode, segment):
    """Same card (H100), same steps: ({suite: episodes}, arms). Off: longseg_off_1022's C5 pure. On: pi0.5 full the
    formal-A H100 pure A, the rest longseg_off_1022's C10 pure fa_lo_*_p_q. ValueError while one is incomplete."""
    if segment == 'off':
        return lo.references(model, mode)
    if (model, mode) == ('pi05', 'full'):
        from exp.offline_search.rounds.r17.filter_failed_knob import formal_a_queue as q
        return q.references(model, mode), ['formal_a_1016/PURE_A_OVERRIDES.json pi05:full']
    from exp.offline_search.rounds.r16 import analyze as a
    from exp.offline_search.rounds.r17.filter_failed_knob.formal_a_analysis import legacy_data
    unit = next(u for u in read(lo.INDEX)['units'] if (u['model'], u['mode'], u['stage'], u.get('segment')) == (model, mode, 'pure', 'c10'))
    out = {}
    for j in unit['jobs']:
        root, arm = Path(j['run']), j['arm']
        if not (root/'runs'/arm/'summary.json').exists():
            raise ValueError(f'H100 C10 pure reference {root.name} ({arm}) is not complete: run {unit["name"]} first')
        row = next(r for r in read(root/'arms.json') if r['arm'] == arm)
        ep = a.load(read(root/'ROOT.json'), arm, PAIRS, 'A', 1 if mode == 'reduced' else 8, a.Reader())
        out[row['suite']] = legacy_data(ep, row['suite'])
    return out, [f'{Path(j["run"]).name}/{j["arm"]}' for j in unit['jobs']]


def load(job):
    """{(suite, size, seed, segment): {level: episodes}} of a job (q3_port_1020's R15/R16 loaders, which lift the
    removed-judge checks); a round arm holds its planned tasks only, complete arms/groups only."""
    from exp.offline_search.rounds.r17.filter_failed_knob.q3_port_1020 import analysis as q3a
    if job['pipeline'] == 'R15':
        return q3a.load_job(job, lambda row: (row['suite'], row['r10_size'], job['subset_seed'], job['long_segment']))
    return q3a.load_job(job, lambda row: (row['suite'], row['r10_size'], row['subset_seed'], row['long_segment']))


def line(name):
    """The jobs of unit `name` and of its repairs."""
    return [j for u in read(INDEX)['units'] if name in (u['name'], u.get('repair_of')) for j in u['jobs']]


def decisionless(job):
    """{arm: [[task, init], ...]} of a finished job: accepted failures without a single decision (no outcome)."""
    from exp.offline_search.rounds.r17.filter_failed_knob.q3_port_1020 import analysis as q3a
    a, root = q3a.a, Path(job['run'])
    if job['pipeline'] == 'R15':
        rows = read(root/'arms.json')
        if not all((root/'runs'/r['arm']/'summary.json').exists() for r in rows):
            raise ValueError(f'{root.name} is not finished')
        eps = {r['arm']: a.load(read(root/'ROOT.json'), r['arm'], PAIRS, 'A', 10 if job['model'] == 'pi05' else 8, a.Reader())
               for r in rows}
    else:
        if not (root/'DONE.json').exists():
            raise ValueError(f'{root.name} is not finished')
        group = read(root/'group.json')
        eps = {e['row']['arm']: q3a.ca.load_entry(group, dict(e, pairs=PAIRS), a.Reader()) for e in group['entries']}
    return {arm: ep.audit['decisionless'] for arm, ep in eps.items() if 'decisionless' in ep.audit}


def evaluate(model, mode):
    """Knob off vs pure for every seed x segment: {'x3_on': decision, ...} (formal_a_queue.classify per suite x size)."""
    from exp.offline_search.rounds.r17.filter_failed_knob import formal_a_queue as q
    data = {}
    for u in read(INDEX)['units']:
        if (u['model'], u['mode'], u['stage']) != (model, mode, 'off'):
            continue
        for job in line(u['name']):
            for k, by_level in load(job).items():
                data.setdefault(k, {}).update(by_level.get('off', {}))
    out = {}
    for segment in SEGMENTS:
        try:
            pure, used = references(model, mode, segment)
        except ValueError as e:
            pure, used = {}, [f'pending: {e}']
        for seed in F.SEEDS:
            rows = [dict(suite=s, size=n, **q.classify(data.get((s, n, seed, segment), {}), pure.get(s, {})))
                    for s, n in q.cells(model, F.SIZES)]
            out[f'{tag(seed)}_{SEG[segment]}'] = dict(model=model, mode=mode, subset_seed=seed, long_segment=segment, rows=rows,
                                                     reference_arms=used, periods=str(F.seed_dir(seed)/'PERIODS.json'),
                                                     rule='tune iff complete off successes < same-card same-step pure successes')
    return out


def survivors(arms, ref, levels):
    """The tasks of a cell that pass every one of `levels` in order (complete, at most two fewer successes than the
    same-step pure reference `ref` on the same 50 inits), and the (level, task, n) still incomplete on the way."""
    names, missing = sorted({k[1] for k in ref}, key=int), []
    for level in levels:
        kept = []
        for task in names:
            inits = [k for k in ref if k[1] == task]
            got = [arms.get(level, {}).get(k) for k in inits]
            if len(inits) != 50 or None in got:
                missing.append((level, task, sum(g is not None for g in got)))
            elif sum(ref[k]['success'] for k in inits) - sum(g['success'] for g in got) <= 2:
                kept.append(task)
        names = kept
    return names, missing


# ------------------------------------------------------------------------------------------- actions
def build():
    missing = [str(F.fit_path(F.seed_dir(s), c, 'off')) for s in F.SEEDS for c in F.cells() if not F.fit_path(F.seed_dir(s), c, 'off').exists()]
    if missing:
        raise FileNotFoundError(f'knob-off fits missing (run fit.py prepare first): {missing}')
    units = []
    for model in F.MODELS:
        cells = sorted(F.cells(model), key=q3.cell_order)
        units += [r15_unit(model, seed, segment, 'off', cells, ('off',)) for seed in F.SEEDS for segment in SEGMENTS]
        units.append(r16_group(model, 'off', {(seed, segment): cells for seed in F.SEEDS for segment in SEGMENTS}, ('off',)))
    put(units)
    level_scripts()
    for u in units:
        print('DEFINED', u['name'], u['episodes'], u['pipeline'], u['gpu'], flush=True)


def levels(model, mode):
    if any((u['model'], u['mode'], u['stage']) == (model, mode, 'round') for u in read(INDEX)['units']):
        raise ValueError(f'{model} {mode}: step 2 is registered as descent rounds (rounds); no full levels sweep')
    decisions = evaluate(model, mode)
    path = HERE/'off_decisions'/f'{model}_{mode}.json'; dump(path, decisions)
    built = {u['name'] for u in read(INDEX)['units']}
    keep, pending = {}, []
    for seed in F.SEEDS:
        for segment in SEGMENTS:
            rows = decisions[f'{tag(seed)}_{SEG[segment]}']['rows']
            if any(r['status'] == 'pending' for r in rows):
                pending.append(f'{tag(seed)}_{SEG[segment]}: {[(r["suite"], r["size"], r["n"], r["pure_n"]) for r in rows if r["status"] == "pending"]}')
                continue
            keep[(seed, segment)] = sorted((cell_name(model, r) for r in rows if r['status'] == 'tune'), key=q3.cell_order)
    if mode == 'reduced' and pending:
        raise ValueError(f'wait for the complete knob-off group and pure references; pending {pending}')
    for seed in F.SEEDS:  # the seed's periods; empty tasks are refused by fit.py
        cells = sorted({c for (s, _), cs in keep.items() if s == seed for c in cs}, key=q3.cell_order)
        if cells:
            F.knob_fits(F.seed_dir(seed), cells, tuple(LEVELS))
    units = []
    if mode == 'full':
        for (seed, segment), cells in keep.items():
            name = r15_name(model, seed, segment, 'levels')
            if name in built or not cells:
                print('SKIP', name, 'already defined' if name in built else 'off sufficient', flush=True); continue
            units.append(dict(r15_unit(model, seed, segment, 'levels', cells, tuple(LEVELS)), off_decision=str(path), selected_cells=cells))
    elif r16_name(model, 'levels') in built or not any(keep.values()):
        print('SKIP', r16_name(model, 'levels'), 'already defined' if r16_name(model, 'levels') in built else 'off sufficient', flush=True)
    else:
        cells_by = {k: v for k, v in keep.items() if v}
        units.append(dict(r16_group(model, 'levels', cells_by, tuple(LEVELS)), off_decision=str(path),
                          selected_cells={f'{tag(s)}_{SEG[g]}': c for (s, g), c in cells_by.items()}))
    put(units)
    for p in pending:
        print('PENDING', p, flush=True)
    for u in units:
        print('DEFINED', u['name'], u['episodes'], flush=True)


def rounds(model, mode):
    """Register step 2 as descent rounds k1..k4 (WAITING; defined by dispatch, build_round)."""
    units = read(INDEX)['units']
    if any((u['model'], u['mode'], u['stage']) == (model, mode, 'levels') for u in units):
        raise ValueError(f'{model} {mode}: a full levels sweep is already defined')
    built = {u['name'] for u in units}
    keys = [(s, g) for s in F.SEEDS for g in SEGMENTS] if mode == 'full' else [(None, None)]
    new = []
    for r, level in enumerate(ORDER, 1):
        for seed, segment in keys:
            name = round_name(model, mode, r, seed, segment)
            if name in built:
                continue
            off, _ = lookup(r15_name(model, seed, segment, 'off') if mode == 'full' else r16_name(model, 'off'))
            new.append(dict(name=name, model=model, mode=mode, arm='filtered', stage='round', round=r, level=level,
                            pipeline=off['pipeline'], gpu='H100', replicas=off['replicas'], episodes=None, jobs=[], status='WAITING',
                            **({'subset_seed': seed, 'long_segment': segment} if mode == 'full' else {})))
    put(new)
    for u in new:
        print('ROUND', u['name'], u['level'], flush=True)


def build_round(name):
    """Define a WAITING round: LEVELS[r-1] for the tasks of each tuned cell (evaluate) that passed every earlier
    level in the finished rounds 1..r-1 of the same line (R15: seed x segment; R16: the group). ValueError while the
    knob-off cells, the references or an earlier round are incomplete."""
    u, _ = lookup(name)
    if u['status'] != 'WAITING':
        return u
    model, mode, r = u['model'], u['mode'], u['round']
    keys = [(u['subset_seed'], u['long_segment'])] if mode == 'full' else [(s, g) for s in F.SEEDS for g in SEGMENTS]
    priors = [lookup(round_name(model, mode, earlier, *keys[0]))[0] for earlier in range(1, r)]
    for prior in priors:
        if prior['status'] not in ('BUILT', 'SKIP_NO_TASKS'):
            raise ValueError(f'{name}: round {prior["round"]} ({prior["name"]}) is {prior["status"]}')
    decisions = evaluate(model, mode)
    path = HERE/'off_decisions'/f'{model}_{mode}.json'; dump(path, decisions)
    rows = {k: decisions[f'{tag(k[0])}_{SEG[k[1]]}']['rows'] for k in keys}
    pending = [(tag(s), g, x['suite'], x['size'], x['n'], x['pure_n']) for (s, g), xs in rows.items() for x in xs if x['status'] == 'pending']
    if pending:
        raise ValueError(f'{name}: wait for the complete knob-off cells and pure references: {pending}')
    measured = {}
    for prior in priors:
        for job in line(prior['name']):
            for k, by_level in load(job).items():
                for level, eps in by_level.items():
                    measured.setdefault(k, {}).setdefault(level, {}).update(eps)
    alive, missing = {}, []
    for seed, segment in keys:
        pure, _ = references(model, mode, segment)
        for x in rows[seed, segment]:
            if x['status'] == 'tune':
                names, gaps = survivors(measured.get((x['suite'], x['size'], seed, segment), {}), pure[x['suite']], ORDER[:r-1])
                missing += [(tag(seed), segment, cell_name(model, x), *g) for g in gaps]
                if names:
                    alive[seed, segment, cell_name(model, x)] = [int(t) for t in names]
    if missing:
        raise ValueError(f'{name}: earlier rounds incomplete {missing[:5]} (+{max(0, len(missing)-5)})')
    level, cells = u['level'], {}
    for seed, segment, cell in alive:
        cells.setdefault((seed, segment), []).append(cell)
    cells = {k: sorted(v, key=q3.cell_order) for k, v in sorted(cells.items())}
    for seed in sorted({s for s, _ in cells}):  # the seed's periods; empty tasks are refused by fit.py
        F.knob_fits(F.seed_dir(seed), sorted({c for (s, _), cs in cells.items() if s == seed for c in cs}, key=q3.cell_order), (level,))
    if not alive:
        built = dict(jobs=[], episodes=0, status='SKIP_NO_TASKS')
    elif mode == 'full':
        built = r15_unit(model, *keys[0], f'k{r}', cells[keys[0]], (level,), {c: alive[(*keys[0], c)] for c in cells[keys[0]]})
    else:
        built = r16_group(model, f'k{r}', cells, (level,), alive)
    u.update({k: built[k] for k in ('jobs', 'episodes', 'status')}, off_decision=str(path),
             tuned_cells={f'{tag(s)}_{SEG[g]}': [cell_name(model, x) for x in xs if x['status'] == 'tune'] for (s, g), xs in rows.items()},
             alive={f'{tag(s)}_{SEG[g]}_{q3.short(c)}': t for (s, g, c), t in sorted(alive.items())})
    put([u])
    print('DEFINED' if alive else 'SKIP_NO_TASKS', name, u['episodes'], flush=True)
    return u


def repair(name):
    """Define <name>r: the decision-less episodes of the finished unit `name`, exactly those pairs, each in a copy of
    its arm (R15) or entry (R16) checked against the source; nothing if there are none."""
    u, _ = lookup(name)
    rname, units = name + 'r', read(INDEX)['units']
    if any(x['name'] == rname for x in units):
        raise ValueError(f'{name}: already repaired by {rname} (repair that one if it needs it)')
    found = {j['name']: decisionless(j) for j in u['jobs']}
    if not any(found.values()):
        print('NO_DECISIONLESS', name, flush=True); return
    same = ('method', 'kwargs', 'plugin_args', 'judge', 'suite', 'r10_size', 'yaml_sha256', 'subset_seed', 'long_segment',
            'q3_artifact', 'r14_descent_level', 'client_overrides', 'replan_steps', 'full_model')
    if u['pipeline'] == 'R16':
        source = {e['row']['arm']: e for e in read(Path(u['jobs'][0]['group']))['entries']}
        key = lambda e: (e['row']['subset_seed'], e['row']['long_segment'], e['row']['r16_cell'].removesuffix('_k1'), e['formal_grid_level'])
        pairs = {key(source[arm]): v for arm, v in found[u['jobs'][0]['name']].items()}
        cells_by = {}
        for seed, segment, cell, _ in pairs:
            cells_by.setdefault((seed, segment), set()).add(cell)
        built = r16_group(u['model'], 'repair', {k: sorted(v, key=q3.cell_order) for k, v in cells_by.items()},
                          sorted({k[3] for k in pairs}, key=ORDER_ALL.index), name=rname, pairs=pairs)
        old = {key(e): e for e in source.values()}
        for e in read(Path(built['jobs'][0]['group']))['entries']:
            o = old[key(e)]
            assert {k: e['row'].get(k) for k in same} == {k: o['row'].get(k) for k in same} and e['fit_sha256'] == o['fit_sha256'], e['row']['arm']
    else:
        pairs, old = {}, {}
        for j in u['jobs']:
            rows = {r['arm']: r for r in read(Path(j['run'])/'arms.json')}
            for arm, v in found[j['name']].items():
                pairs.setdefault(j['cell'], {})[rows[arm]['r14_descent_level']] = v
                old[j['cell'], rows[arm]['r14_descent_level']] = rows[arm], read(Path(j['run'])/'ROOT.json')['fit_sha256']
        built = r15_unit(u['model'], u['subset_seed'], u['long_segment'], 'repair', sorted(pairs, key=q3.cell_order), None,
                         name=rname, pairs=pairs)
        for j in built['jobs']:
            fits = read(Path(j['run'])/'ROOT.json')['fit_sha256']
            for r in read(Path(j['run'])/'arms.json'):
                o, ofits = old[j['cell'], r['r14_descent_level']]
                fit = r['plugin_args'][r['plugin_args'].index('--os-fit-artifact')+1]
                assert {k: r.get(k) for k in same} == {k: o.get(k) for k in same} and fits[fit] == ofits[fit], r['arm']
    rep = dict(built, stage='repair', repair_of=u.get('repair_of', name), repairs=name,
               decisionless={f'{j}/{arm}': v for j, arms in found.items() for arm, v in arms.items()},
               **{k: u[k] for k in ('round', 'level', 'subset_seed', 'long_segment') if k in u})
    put([rep])
    print('DEFINED', rname, rep['episodes'], 'repairs', name, flush=True)
    return rep


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=('build', 'levels', 'rounds', 'repair', 'dispatch'))
    p.add_argument('dispatch_action', nargs='?', choices=('prepare', 'launch'))
    p.add_argument('--name'); p.add_argument('--model', choices=F.MODELS); p.add_argument('--mode', choices=('full', 'reduced'))
    a = p.parse_args()
    if a.action == 'build':
        build()
    elif a.action == 'levels':
        levels(a.model, a.mode)
    elif a.action == 'rounds':
        rounds(a.model, a.mode)
    elif a.action == 'repair':
        repair(a.name)
    else:
        dispatch(a.dispatch_action, a.name)


if __name__ == '__main__':
    main()
