"""Long execution segment OFF (10-22): pi0.5 / GR00T LIBERO knob units, success-filtered library only.

Each unit is its q3_port_1020 counterpart (same fit files, Q3 stall gate --os-judge stall_only, PERIODS.json caps,
R15 TaskDescentKnob, episode subsets, eval500 inits, rollout seed, K, replicas, recording off) without
--os-blind --os-policy-tail --os-policy-tail-blocks 1: every decision, cache hit or policy MISS, executes 5
controls and the next query is the next decision (segment boundary); nothing looks inside a segment.

Model servers on the H100 only:
  full step (R15)    SERVER_HOST=h100, the H100 copy /data/oscl_h100/openpi (pi0.5 as in 1020; GR00T moved off the 4090)
  reduced K1 (R16)   residents on the H100 from /data/oscl_h100/openpi_r16_q3_server (runner.py, h100.py)
Workers: the central scheduler on a100/timan107/timan1 (counts in the definitions are labels).

  build     fa_lo_{pi,gr}_f_f_o_q (8 cells x 500, R15) and fa_lo_{pi,gr}_r_f_o_q (8 x 500 in one R16 group)
  refs      H100 (same-card) pure-inference references fa_lo_gr_f_p_q, fa_lo_pi_r_p_q, fa_lo_gr_r_p_q: the formal-A
            pure A arms (one 500-episode root per suite) with only the server machine moved to the H100
  levels    after a complete OFF unit: per cell OFF vs the same-card same-step pure A (H100 only: pi0.5 full
            fa_pi_f_p10/fa_pi_f_ps, the rest the refs above; refused until that reference is complete);
            cells below it get ir45/ir40/ir32/ir25 with the 1020 periods (owner: no recalibration)
  prepare / dispatch   coordinator entry points (commands/prepare_fa_lo_*.sh, launch_fa_lo_*.sh)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

from ..q3_port_1020 import build as q3
from ..q3_port_1020.fit import CELLS, LEVELS, read, dump, short, REPO, KNOB
from exp.r17_integration.ops.immutable_release import nvme_dir

HERE = Path(__file__).resolve().parent
OUT = Path('/nvme/weiland/openpi/r17_lo_1022')
ALIAS = Path('/home/weiland/projects/openpi_ext/r17_lo_1022')  # -> OUT here; the same address on the H100 (h100.LINKS)
DEFS = ALIAS/'definitions'
INDEX = HERE/'QUEUE.json'
COMMANDS = KNOB/'commands'
MODULE = 'exp.offline_search.rounds.r17.filter_failed_knob.longseg_off_1022'
ARM = 'filtered'
M = {'pi05': 'pi', 'groot': 'gr'}
STAGES = {'off': 'o', 'levels': 'l'}
TRACE = Path('/home/weiland/trace_runs/os_closed_loop')
MANIFEST = Path('/home/weiland/trace_runs/os_closed_loop/r11_knob_1/eval500.json')  # the 1020 R15 eval500 template
# R15 on the H100 (node.py ports 23200..23299): pi0.5 keeps its 1020 topology; GR00T 5 replicas x 7.2 GB (owner 10-20).
R15_ENV = {'pi05': dict(SERVER_HOST='h100', WORKER_HOST='a100'),
           'groot': dict(SERVER_HOST='h100', WORKER_HOST='a100', PORTS='23233,23234,23235,23236,23237',
                         LOCAL_SERVER_COUNT='5', NEED_MB='7200')}
# R16: residents on the H100 (ports there), drivers on weilandserver (ziyanglin.com; 23159 is the cluster simulation's).
R16 = {'pi05': dict(port=23157, ports='23150,23151,23152'), 'groot': dict(port=23158, ports='23153,23154,23155,23156')}
# H100 pure references (R15 ports 23200..23299, clear of the fa_lo_* units); SYNC_PORT 23194 (fa_lo_* use 23195/23196).
PURE_PORTS = {('groot', 'full'): 23250, ('pi05', 'reduced'): 23254, ('groot', 'reduced'): 23257, ('pi05', 'full'): 23261}
PURE_NEED_MB = {'groot': '7200'}  # owner 10-20 for GR00T; pi0.5 keeps its source reference's start budget
WORKERS = 'export WORKER_HOST=${WORKER_HOST:-a100} WORKER_HOSTS=${WORKER_HOSTS:-a100,timan107,timan1}'


def name(model, mode, stage):
    return f'fa_lo_{M[model]}_{mode[0]}_f_{STAGES[stage]}_q'


def segment_off(args):
    """The 1020 plugin arguments without the long execution segment; the Q3 judge and fit stay."""
    out, i = [], 0
    while i < len(args):
        if args[i] in ('--os-blind', '--os-policy-tail'):
            i += 1
        elif args[i] == '--os-policy-tail-blocks':
            i += 2
        else:
            out.append(args[i]); i += 1
    assert out[out.index('--os-judge')+1] == 'stall_only'
    return out


def fits(cells, levels):
    """The q3_port_1020 success-filtered fit files; a missing one is built there with its knob_blob (one process)."""
    out = {}
    for cell in cells:
        missing = [(ARM, l) for l in levels if not q3.fit_file(ARM, cell, l).exists()]
        if missing:
            q3.build_cell_fits((cell, missing))
        out[cell] = {(ARM, l): (str(q3.fit_file(ARM, cell, l)), q3.sha(q3.fit_file(ARM, cell, l))) for l in levels}
    return out


# ------------------------------------------------------------------- H100 pure-inference references
def pure_sources(model, mode):
    """The formal-A pure A arm per suite (formal_a_analysis.vla_references): {suite_short: (root, arm)}."""
    from .. import vla_full as vf
    if (model, mode) == ('pi05', 'full'):  # the H100 formal-A pure A (fa_pi_f_p10 / fa_pi_f_ps)
        over = read(KNOB/'formal_a_1016'/'PURE_A_OVERRIDES.json')
        return {s: (Path(over[f'pi05:full:{suite}']['root']), over[f'pi05:full:{suite}']['arm'])
                for s, suite in (('l10', 'libero_10'), ('spatial', 'libero_spatial'))}
    if mode == 'full':
        refs = read(vf.SOURCE/'c10_proof.json')['test_references']
        return {s: (Path(refs[f'{model}_{s}']['root']), refs[f'{model}_{s}']['arm']) for s in ('l10', 'spatial')}
    design = read(REPO/'exp/offline_search/rounds/r16/design.json')
    rec = {r['cell']: r for r in design['phase1']}
    return {s: (Path(rec[f'{model}_{s}_50_k1']['root']), rec[f'{model}_{s}_50_k1']['arms'][0]) for s in ('l10', 'spatial')}


def pure_fit(cell):
    """Pure inference without the removed gate: the 1020 Q3 TaskDescentKnob with every task at c10 (cap 0, a knob
    call at every decision; --os-judge stall_only forwards it as a MISS). The formal-A pure fits (R13Knob pure_c10)
    pickle the removed noprog TriggerCommitJudge and need --os-judge guard_only; both are refused now."""
    path = ALIAS/'fits'/'pure'/f'{cell}_c10.pkl'
    if not path.exists():
        q3.write_fit(path, q3.knob_blob(cell, 'original', 'c10'))
    return str(path)


def pure_unit(model, mode, segment='c5'):
    """segment c5 (owner 10-22, the references in use): no --os-blind/--os-policy-tail, a model call every 5 controls;
    c10: the formal-A configuration as is (fa_lo_*_p_q, built first, unused)."""
    from exp.offline_search.rounds.r15.descent import prepare as p
    unit = f'fa_lo_{M[model]}_{mode[0]}_{"p5" if segment == "c5" else "p"}_q'
    base = q3.topologies(model, ARM)['l50']
    jobs = []
    for suite, (source, arm) in pure_sources(model, mode).items():
        cell = f'{model}_{suite}_50'
        root = q3.EVAL/f'{unit}_{"l10" if suite == "l10" else "s"}'; q3.fresh(root); nvme_dir(root, exist_ok=True)
        if (root/'arms.json').exists():
            (root/'arms.json').unlink()
        manifest = root/'eval500.json'
        manifest.write_bytes((source/'eval500.json').read_bytes())
        spec = copy.deepcopy(next(s for s in read(source/'arms_in.json') if s['name'] == arm))
        args = spec['plugin_args']
        args[args.index('--os-judge')+1] = 'stall_only'
        args[args.index('--os-fit-artifact')+1] = pure_fit(cell)
        if segment == 'c5':
            args = segment_off(args)
        # yaml_patch (miss num_steps K) and server_env (GROOT_DENOISING_STEPS) stay as in the source spec.
        spec.update(name=f'{root.name}_{segment}', manifest=str(manifest), method=q3.KNOB_SPEC,
                    kwargs=q3.knob_kwargs(cell, 'c10'), plugin_args=args)
        p.emit(root, cell, [spec], ['c10'], pool='A', final_root=root)
        launch = read(source/'runs'/arm/'h100_launch.json')
        n = len(launch['servers'])
        first = PURE_PORTS[(model, mode)]
        top = copy.deepcopy(base)
        top['env'].update(R15_ENV[model])
        top['env'].update(PORTS=','.join(str(first+i) for i in range(n)), LOCAL_SERVER_COUNT=str(n), SYNC_PORT='23194',
                          NEED_MB=PURE_NEED_MB.get(model, str(launch['servers'][0]['need_mb'])))
        top.update(gpu='H100', replicas=n, source_root=str(source), source_arm=arm)
        top.pop('source_receipt_sha256', None)
        jobs.append(dict(name=root.name, run=str(root), arm=spec['name'], model=model, mode=mode, arm_kind='pure',
                         pipeline='R15', stage='pure', segment=segment, cell=cell, suite=suite, episodes=500, levels=['c10'], gpu='H100',
                         replicas=n, workers='central scheduler: a100/timan107/timan1', topology=top,
                         reference_source=dict(root=str(source), arm=arm, server_host=launch.get('server_host', 'h100')),
                         server_tree='/data/oscl_h100/openpi (h100)'))
    return dict(name=unit, model=model, mode=mode, arm='pure', stage='pure', segment=segment, pipeline='R15', gpu='H100',
                replicas=jobs[0]['replicas'], episodes=sum(j['episodes'] for j in jobs), jobs=jobs, status='BUILT')


def refs():
    alias()
    units = [pure_unit(m, mode) for m, mode in (('pi05', 'full'), ('groot', 'full'), ('pi05', 'reduced'), ('groot', 'reduced'))]
    put(units)
    index = read(INDEX)  # the C10 copies stay on disk, out of use (owner 10-22: C5 references)
    for u in index['units']:
        if u['stage'] == 'pure' and u.get('segment', 'c10') == 'c10':
            u.update(segment='c10', status='UNUSED', unused='owner 10-22: levels select against the C5 references fa_lo_*_p5_q')
    dump(INDEX, index)
    for u in units:
        print('DEFINED', u['name'], u['episodes'], u['pipeline'], u['gpu'], [j['name'] for j in u['jobs']], flush=True)


def references(model, mode):
    """The H100 same-step C5 pure inference per suite (fa_lo_*_p5_q); refused while any is incomplete. Never the 4090 or C10 ones."""
    from exp.offline_search.rounds.r16 import analyze as a
    from ..formal_a_analysis import legacy_data
    unit = next(u for u in read(INDEX)['units'] if (u['model'], u['mode'], u['stage'], u.get('segment')) == (model, mode, 'pure', 'c5'))
    arms = [(Path(j['run']), j['arm']) for j in unit['jobs']]
    k = 1 if mode == 'reduced' else 10 if model == 'pi05' else 8
    out = {}
    for root, arm in arms:
        if not (root/'runs'/arm/'summary.json').exists():
            raise ValueError(f'H100 pure reference {root.name} ({arm}) is not complete; levels are built against it only')
        row = next(r for r in read(root/'arms.json') if r['arm'] == arm)
        ep = a.load(read(root/'ROOT.json'), arm, [(t, i) for t in range(10) for i in range(50)], 'A', k, a.Reader())
        out[row['suite']] = legacy_data(ep, row['suite'])
    return out, [f'{root.name}/{arm}' for root, arm in arms]


# ------------------------------------------------------------------------------------- R15 (full step)
def r15_unit(model, stage, cells, levels, fit):
    from exp.offline_search.rounds.r15.descent import prepare as p
    unit = name(model, 'full', stage)
    tops = q3.topologies(model, ARM)
    jobs = []
    for cell in cells:
        root = q3.EVAL/f'{unit}_{short(cell)}'; q3.fresh(root); nvme_dir(root, exist_ok=True)  # real dir on /nvme
        if (root/'arms.json').exists():
            (root/'arms.json').unlink()
        manifest = root/'eval500.json'
        manifest.write_bytes(MANIFEST.read_bytes())
        specs = []
        for level in levels:
            spec = q3.r15_spec(cell, f'{root.name}_{level}', fit[cell][(ARM, level)][0], q3.knob_kwargs(cell, level), manifest)
            spec['plugin_args'] = segment_off(spec['plugin_args'])
            specs.append(spec)
        p.emit(root, cell, specs, list(levels), pool='A', final_root=root)
        top = copy.deepcopy(tops[short(cell)])
        top['env'].update(R15_ENV[model]); top.update(gpu='H100', replicas=len(top['env']['PORTS'].split(',')))
        jobs.append(dict(name=root.name, run=str(root), model=model, mode='full', arm=ARM, pipeline='R15', stage=stage,
                         cell=cell, episodes=500*len(levels), levels=list(levels), gpu='H100', replicas=top['replicas'],
                         workers='central scheduler: a100/timan107/timan1', topology=top, long_segment='off',
                         server_tree='/data/oscl_h100/openpi (h100)'))
    return dict(name=unit, model=model, mode='full', arm=ARM, stage=stage, pipeline='R15', gpu='H100', long_segment='off',
                replicas=jobs[0]['replicas'], episodes=sum(j['episodes'] for j in jobs), jobs=jobs, status='BUILT')


# --------------------------------------------------------------------------------- R16 (reduced, K1)
def r16_entry(group_name, arm, src, fit, *, kwargs, args, level):
    d = DEFS/group_name/arm; d.mkdir(parents=True, exist_ok=True)
    (d/'config.yaml').write_bytes(Path(src['yaml']).read_bytes())
    (d/'eval500.json').write_bytes(Path(src['manifest']).read_bytes())
    path, digest = fit
    dump(d/'ROOT.json', dict(definitions_sha256={n: q3.sha(d/n) for n in ('config.yaml', 'eval500.json')}, fit_sha256={path: digest},
                             init_pool='A', recording=False, source_arm=src['arm'], longseg_off_1022=True))
    row = copy.deepcopy(src)
    for k in ('matrix', 'remote_yaml', 'remote_matrix', 'yaml_patch'):
        row.pop(k, None)
    cell = row['r16_cell'].removesuffix('_k1')
    row.update(arm=arm, method=q3.KNOB_SPEC, kwargs=kwargs, plugin_args=args, yaml=str(d/'config.yaml'),
               yaml_sha256=q3.sha(d/'config.yaml'), manifest=str(d/'eval500.json'), judge='stall_only', full_model=True,
               episodes=500, stall_gate='q3', q3_artifact=str(q3.q3_path(cell)), source_off_arm=src['arm'],
               r14_descent_level='task_descent', long_segment='off', intended_host='h100')
    return dict(row=row, source_root=str(d), source_seal_sha256=q3.sha(d/'ROOT.json'), fit_sha256=digest,
                yaml_sha256=q3.sha(d/'config.yaml'), pairs=copy.deepcopy(q3.PAIRS), dispatch_pairs=copy.deepcopy(q3.PAIRS),
                formal_grid_level=level, source_off_arm=src['arm'])


def r16_unit(model, stage, cells, levels, fit):
    from exp.offline_search.rounds.r16 import conductor as rc
    group_name = name(model, 'reduced', stage)
    source, offs = q3.source_rows(model)
    entries = []
    for cell in cells:
        src = offs[cell]['row']
        for level in levels:
            entries.append(r16_entry(group_name, f'{group_name}_{short(cell)}_{level}', src, fit[cell][(ARM, level)],
                                     kwargs=q3.knob_kwargs(cell, level), args=segment_off(q3.knob_args(src, fit[cell][(ARM, level)][0])),
                                     level=level))
    root = q3.EVAL/group_name; q3.fresh(root)
    group = {k: v for k, v in source.items() if k not in ('entries', 'definitions_sha256', 'group_sha256')}
    group.update(root=str(root), entries=entries, phase='test', references=None, formal_test_A=True, k1ref=False,
                 original_episode_uids=False, yaml_id_is_original_arm=False, intended_server_host='h100',
                 longseg_off_1022=dict(arm=ARM, stage=stage, levels=list(levels), cells=list(cells), long_segment='off'))
    files = sorted(p for p in (DEFS/group_name).rglob('*') if p.is_file())
    group['definitions_sha256'] = {**{str(p): q3.sha(p) for p in files},
                                   **{e['row']['plugin_args'][e['row']['plugin_args'].index('--os-fit-artifact')+1]: e['fit_sha256'] for e in entries}}
    group['group_sha256'] = hashlib.sha256(rc.canonical(group)).hexdigest()
    nvme_dir(root, exist_ok=True)
    dump(root/'group.json', group)
    dump(root/'ROOT.json', dict(group_sha256=group['group_sha256'], definitions_sha256=group['definitions_sha256']))
    suites = {e['row']['suite'] for e in entries}
    replicas = 4 if model == 'groot' else 3
    split = None if model == 'pi05' else ('3,1' if suites == {'libero_10', 'libero_spatial'} else
                                         '4,0' if suites == {'libero_10'} else '0,4')
    job = dict(name=group_name, run=str(root), group=str(root/'group.json'), model=model, pipeline='R16',
               episodes=sum(len(e['dispatch_pairs']) for e in entries), port=R16[model]['port'], ports=R16[model]['ports'],
               groot_suite_servers=split, replicas=replicas, gpu='H100', long_segment='off',
               server_tree='/data/oscl_h100/openpi_r16_q3_server (h100)')
    return dict(name=group_name, model=model, mode='reduced', arm=ARM, stage=stage, pipeline='R16', gpu='H100',
                long_segment='off', replicas=replicas, episodes=job['episodes'], jobs=[job], status='BUILT')


# ---------------------------------------------------------------------------------- index, commands
def put(units):
    index = read(INDEX) if INDEX.exists() else dict(units=[])
    names = {u['name'] for u in units}
    index['units'] = [u for u in index['units'] if u['name'] not in names] + units
    dump(INDEX, index)
    for u in units:
        commands(u)


def describe(u):
    what = {'off': 'knob off', 'levels': 'levels ir45/40/32/25',
            'pure': ('H100 pure-inference reference C5 (a model call every 5 controls, no long segment)' if u.get('segment') == 'c5'
                     else 'H100 pure-inference reference C10 (formal-A configuration; UNUSED)')}[u['stage']]
    steps = 'full step' if u['mode'] == 'full' else 'reduced K1'
    model = {'pi05': 'pi0.5', 'groot': 'GR00T'}[u['model']]
    server = ('H100 servers (/data/oscl_h100/openpi)' if u['pipeline'] == 'R15'
              else 'H100 residents (/data/oscl_h100/openpi_r16_q3_server)')
    if u['stage'] == 'pure':
        return [f'Same-card reference for longseg_off_1022: no library (pure inference) . {model} . LIBERO . {steps} . {what} . {u["episodes"]} episodes',
                f'{server}; {u["replicas"]} {model} replicas; source arms {[j["reference_source"]["arm"] for j in u["jobs"]]}. See longseg_off_1022/HANDBACK.md.']
    return [f'Long segment OFF (longseg_off_1022): success-filtered library . {model} . LIBERO . {steps} . {what} . {u["episodes"]} episodes',
            f'{server}; {u["replicas"]} {model} replicas; Q3 --os-judge stall_only; no --os-blind/--os-policy-tail. See longseg_off_1022/HANDBACK.md.']


def commands(u):
    if u['pipeline'] == 'R15':
        for target in [u['name']] + [j['name'] for j in u['jobs']]:
            for action, prefix in (('prepare', q3.PREFIX), ('launch', q3.LAUNCH_PREFIX)):
                q3.script(COMMANDS/f'{action}_{target}.sh', [WORKERS, f'{prefix} -m {MODULE}.build dispatch {action} --name {target}'],
                          describe(u))
    else:
        j = u['jobs'][0]
        q3.script(COMMANDS/f'prepare_{u["name"]}.sh', [f'{q3.PREFIX} -m {MODULE}.build prepare --name {u["name"]}'], describe(u))
        split = f' --groot-suite-servers {j["groot_suite_servers"]}' if j['groot_suite_servers'] else ''
        q3.script(COMMANDS/f'launch_{u["name"]}.sh', [f'{q3.LAUNCH_PREFIX} -m {MODULE}.runner run --group {j["group"]} '
                  f'--agent-island /scratch/zixuans8/openpi_r16_conductor_v1 --port {j["port"]} --ports {j["ports"]}{split} '
                  '--no-local-workers'], describe(u))


def level_scripts():
    for model in ('pi05', 'groot'):
        for mode in ('full', 'reduced'):
            target = name(model, mode, 'levels')
            q3.script(COMMANDS/f'build_levels_{target}.sh',
                      [f'{q3.PREFIX} -m {MODULE}.build levels --model {model} --mode {mode}'],
                      [f'Long segment OFF: define {target} (ir45/40/32/25 for the cells whose OFF cohort is below the same-step pure A).',
                       f'Run after {name(model, mode, "off")} is complete; then prepare_/launch_{target}.sh. CPU only.'])


def lookup(target):
    for u in read(INDEX)['units']:
        if u['name'] == target:
            return u, u['jobs']
        for j in u['jobs']:
            if j['name'] == target:
                return u, [j]
    raise KeyError(target)


def prepare(target):
    from . import h100
    _, jobs = lookup(target)
    for j in jobs:
        h100.sync(j['group'])


def dispatch(action, target):
    _, jobs = lookup(target)
    for j in jobs:
        if j['pipeline'] == 'R15':
            q3.r15_run(j, action)
        elif action == 'prepare':
            prepare(j['name'])
        else:
            subprocess.run(['bash', str(COMMANDS/f'launch_{j["name"]}.sh')], check=True)


# ------------------------------------------------------------------------------------------- actions
def alias():
    OUT.mkdir(parents=True, exist_ok=True)
    if not ALIAS.is_symlink():
        ALIAS.symlink_to(OUT, target_is_directory=True)


def build():
    alias()
    units = []
    for model in ('pi05', 'groot'):
        cells = q3.model_cells(model)
        fit = fits(cells, ('off',))
        units += [r15_unit(model, 'off', cells, ('off',), fit), r16_unit(model, 'off', cells, ('off',), fit)]
    put(units)
    level_scripts()
    for u in units:
        print('DEFINED', u['name'], u['episodes'], u['pipeline'], u['gpu'], flush=True)


def evaluate(model, mode):
    """OFF (q3_port_1020's R15/R16 loaders) vs the H100 same-card same-step pure A; refused while that is incomplete."""
    from ..q3_port_1020 import analysis as q3a
    from .. import formal_a_queue as q
    pure, used = references(model, mode)
    unit = next(u for u in read(INDEX)['units'] if (u['model'], u['mode'], u['stage']) == (model, mode, 'off'))
    data = {}
    for job in unit['jobs']:
        for cell, by_level in q3a.load_job(job).items():
            data.setdefault(cell, {}).update(by_level.get('off', {}))
    rows = [dict(suite=s, size=n, **q.classify(data.get((s, n), {}), pure.get(s, {}))) for s, n in q.cells(model)]
    return dict(model=model, mode=mode, arm=ARM, unit=unit['name'], rows=rows, long_segment='off', reference_arms=used,
                reference=f'H100 same-card same-step pure inference C5 ({dict(pi05="full K10", groot="full K8")[model] if mode == "full" else "reduced K1"}; '
                          'a model call every 5 controls, no long segment)',
                rule='tune iff complete off successes < pure successes', periods='q3_port_1020 PERIODS.json (no recalibration)')


def levels(model, mode):
    decision = evaluate(model, mode)
    path = HERE/'off_decisions'/f'{model}_{mode}_{ARM}.json'; dump(path, decision)
    pending = [r for r in decision['rows'] if r['status'] == 'pending']
    if pending:
        raise ValueError(f'wait for a complete OFF unit; pending {[(r["suite"], r["size"], r["n"]) for r in pending]}')
    keep = sorted((f'{model}_{"l10" if r["suite"] == "libero_10" else "spatial"}_{r["size"]}' for r in decision['rows']
                   if r['status'] == 'tune'), key=q3.cell_order)
    if not keep:
        print('SKIP_OFF_SUFFICIENT', model, mode, flush=True); return
    alias()
    fit = fits(keep, tuple(LEVELS))
    u = (r15_unit if mode == 'full' else r16_unit)(model, 'levels', keep, tuple(LEVELS), fit)
    u.update(off_decision=str(path), selected_cells=keep)
    put([u]); print('DEFINED', u['name'], u['episodes'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=('build', 'refs', 'levels', 'prepare', 'dispatch'))
    p.add_argument('dispatch_action', nargs='?', choices=('prepare', 'launch'))
    p.add_argument('--name'); p.add_argument('--model', choices=('pi05', 'groot'))
    p.add_argument('--mode', choices=('full', 'reduced'))
    a = p.parse_args()
    if a.action == 'build':
        build()
    elif a.action == 'refs':
        refs()
    elif a.action == 'levels':
        levels(a.model, a.mode)
    elif a.action == 'prepare':
        prepare(a.name)
    else:
        dispatch(a.dispatch_action, a.name)


if __name__ == '__main__':
    main()
