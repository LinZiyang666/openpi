"""Q3 port (10-20): fits and batch definitions for pi0.5/GR00T LIBERO with the Q3 stall gate.

  build    knob-off full (R15, 4 units x 8 cells x 500), knob-off reduced (R16, 4 groups x 8 cells x 500),
           cache-study gated groups cg/crg/chg/chrg (R16, 2 groups x 8 cells x 4 x 500); commands/ scripts
  levels   after the new knob-off results: per-cell off-vs-pure decision, then level batches for tune cells
  prepare / dispatch   coordinator entry points (see commands/prepare_*_q*.sh, launch_*_q*.sh)

Every fit is the frozen no-gate cache of the cache study (..._cr / ..._c fit.pkl) with R10Recipe.inner =
Q3StallInner(base, C(task only), Q3); knob batches add the unchanged R15 TaskDescentKnob layer (all ten
tasks at one level; caps 0 for off, Q3-resolved periods otherwise). Success-filtered fits are the existing
candidate-row filter (vla.candidates) of the same object. Plugin: --os-judge stall_only. Servers run the
Q3 server tree (runner.py / legacy.py); fits and definitions live under /nvme; new run roots are real
directories under /nvme/weiland/openpi/eval with their /data/mw_finetune/eval address left as a symlink.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import pickle
import shlex
import subprocess
from types import SimpleNamespace

import numpy as np

from .fit import HERE, KNOB, REPO, OUT, CELLS, LEVELS, read, dump, short, load_cs, library, stall_model, q3_path, cs_fit
from exp.r17_integration.ops.immutable_release import nvme_dir

EVAL = Path('/data/mw_finetune/eval')
TREE = Path('/home/weiland/projects/openpi_r16_q3_server')
FA = KNOB/'formal_a_1016'
# Fit paths go through a /home alias of /nvme: H100 sync mirrors only /home/weiland and /data inputs.
FITS, DEFS = Path('/home/weiland/projects/openpi_ext/r17_q3_port_1020/fits'), OUT/'definitions'
INDEX = HERE/'QUEUE.json'
COMMANDS = KNOB/'commands'
MODULE = 'exp.offline_search.rounds.r17.filter_failed_knob.q3_port_1020'
PREFIX = 'env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python'
LAUNCH_PREFIX = 'env -u CUDA_VISIBLE_DEVICES -u JAX_PLATFORMS OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python'
STORE = '/home/weiland/trace_runs/offline_search_store'
SEED = 2026101401
KNOB_SPEC = 'exp.offline_search.rounds.r15.descent.recipe:TaskDescentKnob'
R10_SPEC = 'exp.offline_search.rounds.r10.recipe.recipe:R10Recipe'
M = {'pi05': 'pi', 'groot': 'gr'}
A = {'original': 'u', 'filtered': 'f'}
GROUPS = {'cg': (False, False), 'crg': (True, False), 'chg': (False, True), 'chrg': (True, True)}  # (corrector, C10)
PAIRS = [[t, i] for t in range(10) for i in range(50)]
PORTS = {'pi05': '23150,23151,23152', 'groot': '23150,23151,23152,23153'}   # R16 residents: GR00T 2 per suite = 4
GROOT_R15 = dict(PORTS='23100,23101,23102,23103,23104', LOCAL_SERVER_COUNT='5', NEED_MB='7200')  # owner 10-20


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def cell_order(cell):
    _, suite, size = cell.split('_')
    return (suite != 'l10', int(size))


def model_cells(model):
    return sorted((c for c in CELLS if c.startswith(model + '_')), key=cell_order)


def kept(cell):
    return next(r['kept_episode_ids'] for r in read(FA/'LIBRARIES.json')['cells'] if r['cell'] == cell)


def caps_for(cell, level):
    # off and c10 (pure inference: a knob call at every decision) have zero cap placeholders (TaskDescentKnob.set_table)
    return [0.] * 10 if level in ('off', 'c10') else [float(read(OUT/'PERIODS.json')[cell][level])] * 10


# ---------------------------------------------------------------------------------------------- fits
def q3_inner(cell, source):
    from exp.offline_search.rounds.r06.ideation_Q1.method_c.stall_only import Q3StallInner
    lib, _ = library(cell, source)
    return Q3StallInner(source['method'].inner, SimpleNamespace(task=np.asarray(lib['task_id'])), stall_model(cell))


def knob_blob(cell, arm, level, source=None, inner=None):
    from exp.offline_search.rounds.r15.descent.recipe import TaskDescentKnob
    from ..vla import candidates
    source = source or load_cs(cell, True)
    r = source['method']; inner = inner or q3_inner(cell, source)
    levels, caps = {str(t): level for t in range(10)}, caps_for(cell, level)
    kwargs = read_kwargs(cell)
    m = TaskDescentKnob(kwargs['library'], kwargs['episode_subset'], task_levels=levels, task_caps=caps)
    m.knob_model = cell.split('_')[0]
    m.inner, m.row_subset = inner, r.row_subset
    m.size = sum(len(v) for v in kwargs['episode_subset'].values())
    m.fit_info = dict(r.fit_info, guard='Q3 stall gate (R6 Q3 successful-library LOEO); plugin --os-judge stall_only',
                      q3_port_1020=dict(cell=cell, arm=arm, level=level, caps=caps, q3=str(q3_path(cell)),
                                        base_fit=str(cs_fit(cell, True))))
    m.r14_fit = dict(cell=cell, variant='task_descent_q3', levels=levels, K=caps, q3=str(q3_path(cell)))
    m.name = f'Q3Knob_{level}__{inner.name}'
    blob = dict(method=m, registered=source['registered'], spec=KNOB_SPEC, kwargs=dict(kwargs, task_levels=levels, task_caps=caps),
                cell=source['cell'], fit_s=source.get('fit_s'))
    return candidates(blob, kept(cell)) if arm == 'filtered' else blob


def cs_blob(cell, corrector):
    from exp.offline_search.rounds.r10.recipe.recipe import R10Recipe
    source = load_cs(cell, corrector); r = source['method']
    kwargs = read_kwargs(cell)
    m = R10Recipe(**kwargs)
    m.inner, m.row_subset = q3_inner(cell, source), r.row_subset
    m.uses_nonlibrary_action = corrector
    m.name = ('Q3CS_corrected__' if corrector else 'Q3CS_cache__') + r.inner.name
    m.fit_info = dict(r.fit_info, q3_port_1020=dict(cell=cell, corrector=corrector, q3=str(q3_path(cell)),
                                                    base_fit=str(cs_fit(cell, corrector)), judge='stall_only'))
    return dict(method=m, registered=source['registered'], spec=R10_SPEC, kwargs=kwargs, cell=source['cell'],
                fit_s=source.get('fit_s'))


def fit_file(kind, cell, tag):
    return FITS/kind/f'{cell}_{tag}.pkl'


def write_fit(path, blob):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp'); tmp.write_bytes(pickle.dumps(blob, protocol=4)); tmp.replace(path)
    return str(path), sha(path)


def build_cell_fits(job):
    cell, jobs = job
    out = {}
    source = load_cs(cell, True); inner = q3_inner(cell, source)
    for kind, tag in jobs:
        path = fit_file(kind, cell, tag)
        if path.exists():
            out[(kind, tag)] = str(path), sha(path); continue
        if kind in ('original', 'filtered'):
            blob = knob_blob(cell, kind, tag, source, inner)
        else:
            blob = cs_blob(cell, tag == 'cr')
        out[(kind, tag)] = write_fit(path, blob)
    print('FITS', cell, len(out), flush=True)
    return cell, out


def build_fits(requests):
    """requests: {cell: [(kind, tag)]}; kind original/filtered (tag = level) or cs (tag c/cr)."""
    with ProcessPoolExecutor(16) as pool:
        return dict(pool.map(build_cell_fits, sorted(requests.items())))


# ------------------------------------------------------------------------------------- R15 (full step)
def topologies(model, arm):
    unit = next(u for u in read(FA/'QUEUE.json')['units'] if u['name'] == f'fa_{M[model]}_f_{A[arm]}_o')
    out = {}
    for j in unit['jobs']:
        top = copy.deepcopy(j['topology'])
        if model == 'groot':
            top['env'].update(GROOT_R15); top['replicas'] = 5
        out[j['name'].rsplit('_', 1)[1]] = top
    return out


def r15_spec(cell, name, fit, kwargs, manifest):
    model, suite, _ = cell.split('_')
    return dict(name=name, model=model, suite=suite, mode='plugin', full_model=True, cost_ledger=True,
                client_overrides=dict(replan_steps=5, **({'resize_size': 256} if model == 'groot' else {})),
                manifest=str(manifest), method=KNOB_SPEC, kwargs=kwargs,
                plugin_args=['--os-root', STORE, '--os-no-shadow-native', '--os-tokens', 'off',
                             '--os-blind', '--os-policy-tail', '--os-policy-tail-blocks', '1',
                             '--os-judge', 'stall_only', '--os-fit-artifact', fit,
                             '--os-rollout-seed', str(SEED), '--os-r14-audit'])


def fresh(root):
    if any(Path(root).glob('runs/*/h100_launch.json')) or any((Path(root)/n).exists() for n in ('launch.json', 'journal.jsonl', 'DONE.json')):
        raise ValueError(f'Never rebuild a launched run: {root}')


STAGES = {'off': 'o', 'levels': 'l'}


def r15_unit(model, arm, stage, cells, levels, fits):
    from exp.offline_search.rounds.r15.descent import prepare as p
    unit = f'fa_{M[model]}_f_{A[arm]}_{STAGES[stage]}_q'
    tops = topologies(model, arm); jobs = []
    for cell in cells:
        root = EVAL/f'{unit}_{short(cell)}'; fresh(root); nvme_dir(root, exist_ok=True)  # real dir on /nvme
        if (root/'arms.json').exists():
            (root/'arms.json').unlink()
        manifest = root/'eval500.json'
        manifest.write_bytes(Path('/home/weiland/trace_runs/os_closed_loop/r11_knob_1/eval500.json').read_bytes())
        specs = []
        for level in levels:
            path, _ = fits[cell][(arm, level)]
            specs.append(r15_spec(cell, f'{root.name}_{level}', path, knob_kwargs(cell, level), manifest))
        p.emit(root, cell, specs, list(levels), pool='A', final_root=root)
        top = tops[short(cell)]
        jobs.append(dict(name=root.name, run=str(root), model=model, mode='full', arm=arm, pipeline='R15', stage=stage,
                         cell=cell, episodes=500*len(levels), levels=list(levels), gpu=top['gpu'], replicas=top['replicas'],
                         workers=top['workers'], topology=top, server_tree=str(TREE) if model == 'groot' else '/data/oscl_h100/openpi (h100)'))
    return dict(name=unit, model=model, mode='full', arm=arm, stage=stage, pipeline='R15', gpu=jobs[0]['gpu'],
                replicas=jobs[0]['replicas'], episodes=sum(j['episodes'] for j in jobs), jobs=jobs, status='BUILT')


def knob_kwargs(cell, level):
    source = read_kwargs(cell)
    return dict(source, task_levels={str(t): level for t in range(10)}, task_caps=caps_for(cell, level))


_KW = {}
def read_kwargs(cell):
    if cell not in _KW:
        group = read(FA/'runs'/f'ffk1015_{cell.split("_")[0]}_libero_reduced_formal_A_original/group.json')
        row = next(e['row'] for e in group['entries'] if e['row']['r16_cell'] == cell + '_k1')
        _KW[cell] = dict(library=row['kwargs']['library'], episode_subset=row['kwargs']['episode_subset'])
    return _KW[cell]


# --------------------------------------------------------------------------------- R16 (reduced, K1)
def source_rows(model):
    group = read(FA/'runs'/f'ffk1015_{model}_libero_reduced_formal_A_original/group.json')
    offs = {e['row']['r16_cell'].removesuffix('_k1'): e for e in group['entries'] if e['row']['arm'].endswith('_off')}
    assert len(offs) == 8, sorted(offs)
    return group, offs


def r16_entry(name, arm, src, fit, *, method, kwargs, args, level, extra):
    d = DEFS/name/arm; d.mkdir(parents=True, exist_ok=True)
    (d/'config.yaml').write_bytes(Path(src['yaml']).read_bytes())
    (d/'eval500.json').write_bytes(Path(src['manifest']).read_bytes())
    path, digest = fit
    dump(d/'ROOT.json', dict(definitions_sha256={n: sha(d/n) for n in ('config.yaml', 'eval500.json')}, fit_sha256={path: digest},
                             init_pool='A', recording=False, source_arm=src['arm'], q3_port_1020=True))
    row = copy.deepcopy(src)
    for k in ('matrix', 'remote_yaml', 'remote_matrix', 'yaml_patch'):
        row.pop(k, None)
    row.update(arm=arm, method=method, kwargs=kwargs, plugin_args=args, yaml=str(d/'config.yaml'), yaml_sha256=sha(d/'config.yaml'),
               manifest=str(d/'eval500.json'), judge='stall_only', full_model=True, episodes=500, stall_gate='q3',
               q3_artifact=str(q3_path(row['r16_cell'].removesuffix('_k1'))), source_off_arm=src['arm'], **extra)
    return dict(row=row, source_root=str(d), source_seal_sha256=sha(d/'ROOT.json'), fit_sha256=digest,
                yaml_sha256=sha(d/'config.yaml'), pairs=copy.deepcopy(PAIRS), dispatch_pairs=copy.deepcopy(PAIRS),
                formal_grid_level=level, source_off_arm=src['arm'])


def r16_group(name, model, source, entries, info):
    from exp.offline_search.rounds.r16 import conductor as rc
    root = EVAL/name; fresh(root)
    group = {k: v for k, v in source.items() if k not in ('entries', 'definitions_sha256', 'group_sha256')}
    group.update(root=str(root), entries=entries, phase='test', references=None, formal_test_A=True, k1ref=False,
                 original_episode_uids=False, yaml_id_is_original_arm=False, q3_port_1020=info)
    files = sorted(p for p in (DEFS/name).rglob('*') if p.is_file())
    group['definitions_sha256'] = {**{str(p): sha(p) for p in files},
                                   **{e['row']['plugin_args'][e['row']['plugin_args'].index('--os-fit-artifact')+1]: e['fit_sha256'] for e in entries}}
    group['group_sha256'] = hashlib.sha256(rc.canonical(group)).hexdigest()
    nvme_dir(root, exist_ok=True)  # real dir on /nvme; group root keeps its /data address
    dump(root/'group.json', group)
    dump(root/'ROOT.json', dict(group_sha256=group['group_sha256'], definitions_sha256=group['definitions_sha256']))
    return dict(name=name, run=str(root), group=str(root/'group.json'), model=model, pipeline='R16',
                episodes=sum(len(e['dispatch_pairs']) for e in entries), ports=PORTS[model],
                replicas=4 if model == 'groot' else 3, gpu='RTX 4090', server_tree=str(TREE))


def knob_args(src, fit):
    args = src['plugin_args'][:src['plugin_args'].index('--os-blind')]   # --os-root .. --os-tokens off
    return args + ['--os-blind', '--os-policy-tail', '--os-policy-tail-blocks', '1', '--os-judge', 'stall_only',
                   '--os-fit-artifact', fit, '--os-rollout-seed', str(SEED), '--os-r14-audit']


def r16_knob_unit(model, arm, stage, cells, levels, fits):
    name = f'fa_{M[model]}_r_{A[arm]}_{STAGES[stage]}_q'
    source, offs = source_rows(model); entries = []
    for cell in cells:
        for level in levels:
            fit = fits[cell][(arm, level)]
            entries.append(r16_entry(name, f'{name}_{short(cell)}_{level}', offs[cell]['row'], fit, method=KNOB_SPEC,
                                     kwargs=knob_kwargs(cell, level), args=knob_args(offs[cell]['row'], fit[0]), level=level,
                                     extra=dict(r14_descent_level='task_descent')))
    job = r16_group(name, model, source, entries, dict(arm=arm, stage=stage, levels=list(levels), cells=list(cells)))
    return dict(name=name, model=model, mode='reduced', arm=arm, stage=stage, pipeline='R16', gpu='RTX 4090',
                replicas=job['replicas'], episodes=job['episodes'], jobs=[job], status='BUILT')


def cs_unit(model, fits):
    name = f'fa_cs_{M[model]}_rg'
    source, offs = source_rows(model); entries = []
    for cell in model_cells(model):
        src = offs[cell]['row']
        for group, (corrector, long) in GROUPS.items():
            fit = fits[cell][('cs', 'cr' if corrector else 'c')]
            args = src['plugin_args'][:src['plugin_args'].index('--os-blind')]
            args += (['--os-blind', '--os-policy-tail', '--os-policy-tail-blocks', '1'] if long else [])
            args += ['--os-judge', 'stall_only', '--os-fit-artifact', fit[0], '--os-rollout-seed', str(SEED), '--os-r14-audit']
            kwargs = dict(read_kwargs(cell))
            entries.append(r16_entry(name, f'{name}_{short(cell)}_{group}', src, fit, method=R10_SPEC, kwargs=kwargs, args=args,
                                     level='cs_' + group, extra=dict(r14_descent_level='cs_' + group, cache_study_group=group,
                                     corrector=corrector, commit_controls=10 if long else 5, model_calls_allowed='q3_stall_only')))
            entries[-1]['cache_study_group'] = group
    job = r16_group(name, model, source, entries, dict(groups=list(GROUPS), cost='the Q3-triggered calls are the only model calls'))
    return dict(name=name, model=model, mode='reduced', arm='original', stage='cache_gated', pipeline='R16', gpu='RTX 4090',
                replicas=job['replicas'], episodes=job['episodes'], jobs=[job], status='BUILT')


# ---------------------------------------------------------------------------------- index, commands
def put(units):
    index = read(INDEX) if INDEX.exists() else dict(units=[])
    names = {u['name'] for u in units}
    index['units'] = [u for u in index['units'] if u['name'] not in names] + units
    dump(INDEX, index)
    for u in units:
        commands(u)


def script(path, lines, comment):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/usr/bin/env bash\n' + ''.join(f'# {c}\n' for c in comment) + 'set -euo pipefail\ncd ' +
                    shlex.quote(str(REPO)) + '\n' + '\n'.join(lines) + '\n')
    path.chmod(0o755)


def describe(u):
    lib = 'success-filtered library' if u['arm'] == 'filtered' else 'original library'
    what = {'off': 'knob off', 'levels': 'levels ir45/40/32/25', 'cache_gated': 'cache-study gated groups cg/crg/chg/chrg'}[u['stage']]
    steps = 'full step' if u['mode'] == 'full' else 'reduced K1'
    model = {'pi05': 'pi0.5', 'groot': 'GR00T'}[u['model']]
    server = ('H100 servers (/data/oscl_h100/openpi)' if u['model'] == 'pi05' and u['mode'] == 'full'
              else f'4090 servers from {TREE}')
    return [f'Q3 stall gate (q3_port_1020): {lib} . {model} . LIBERO . {steps} . {what} . {u["episodes"]} episodes',
            f'{server}; {u["replicas"]} {model} replicas. See q3_port_1020/HANDBACK.md.']


def commands(u):
    if u['pipeline'] == 'R15':
        for name in [u['name']] + [j['name'] for j in u['jobs']]:
            for action, prefix in (('prepare', PREFIX), ('launch', LAUNCH_PREFIX)):
                script(COMMANDS/f'{action}_{name}.sh', [f'{prefix} -m {MODULE}.build dispatch {action} --name {name}'], describe(u))
    else:
        j = u['jobs'][0]
        script(COMMANDS/f'prepare_{u["name"]}.sh', [f'{PREFIX} -m {MODULE}.build prepare --name {u["name"]}'], describe(u))
        script(COMMANDS/f'launch_{u["name"]}.sh', [f'{LAUNCH_PREFIX} -m {MODULE}.runner run --group {j["group"]} '
               f'--agent-island /scratch/zixuans8/openpi_r16_conductor_v1 --port 23159 --ports {j["ports"]}'], describe(u))


def lookup(name):
    for u in read(INDEX)['units']:
        if u['name'] == name:
            return u, u['jobs']
        for j in u['jobs']:
            if j['name'] == name:
                return u, [j]
    raise KeyError(name)


def r15_run(job, action):
    from .. import vla_workers as vw
    native = vw.r15_command
    def command(job, action, host):
        argv, env = native(job, action, host)
        argv[argv.index('exp.offline_search.rounds.r17.filter_failed.legacy')] = MODULE + '.legacy'
        return argv, env
    vw.r15_command = command
    try:
        vw.r15(job, action)
    finally:
        vw.r15_command = native


def prepare(name):
    u, jobs = lookup(name)
    for j in jobs:
        if (Path(j['run'])/'DONE.json').exists():
            raise ValueError(f'{j["name"]} is already DONE')
    print('PREPARED', name, flush=True)


def dispatch(action, name):
    u, jobs = lookup(name)
    for j in jobs:
        if j['pipeline'] == 'R15':
            r15_run(j, action)
        elif action == 'prepare':
            prepare(j['name'])
        else:
            subprocess.run(['bash', str(COMMANDS/f'launch_{j["name"]}.sh')], check=True)


# ------------------------------------------------------------------------------------------- actions
def build():
    requests = {c: [(a, 'off') for a in ('original', 'filtered')] + [('cs', 'c'), ('cs', 'cr')] for c in CELLS}
    fits = build_fits(requests)
    units = []
    for model in ('pi05', 'groot'):
        cells = model_cells(model)
        for arm in ('original', 'filtered'):
            units.append(r15_unit(model, arm, 'off', cells, ('off',), fits))
            units.append(r16_knob_unit(model, arm, 'off', cells, ('off',), fits))
        units.append(cs_unit(model, fits))
    put(units)
    level_scripts()
    for u in units:
        print('DEFINED', u['name'], u['episodes'], u['pipeline'], u['gpu'], flush=True)


def level_scripts():
    for model in ('pi05', 'groot'):
        for mode in ('full', 'reduced'):
            for arm in ('original', 'filtered'):
                name = f'fa_{M[model]}_{mode[0]}_{A[arm]}_l_q'
                script(COMMANDS/f'build_levels_{name}.sh',
                       [f'{PREFIX} -m {MODULE}.build levels --model {model} --mode {mode} --arm {arm}'],
                       [f'Q3 port: define {name} (levels ir45/40/32/25 for the cells whose new knob-off cohort is below pure A).',
                        f'Run after fa_{M[model]}_{mode[0]}_{A[arm]}_o_q is complete; then prepare_/launch_{name}.sh. CPU only.'])


def levels(model, mode, arm):
    from .analysis import evaluate
    decision = evaluate(model, mode, arm)
    path = HERE/'off_decisions'/f'{model}_{mode}_{arm}.json'; dump(path, decision)
    pending = [r for r in decision['rows'] if r['status'] == 'pending']
    if pending:
        raise ValueError(f'wait for complete off and pure A cohorts; pending {[(r["suite"], r["size"], r["n"]) for r in pending]}')
    keep = [f'{model}_{"l10" if r["suite"] == "libero_10" else "spatial"}_{r["size"]}' for r in decision['rows'] if r['status'] == 'tune']
    if not keep:
        print('SKIP_OFF_SUFFICIENT', model, mode, arm, flush=True); return
    fits = build_fits({c: [(arm, l) for l in LEVELS] for c in keep})
    keep = sorted(keep, key=cell_order)
    u = (r15_unit if mode == 'full' else r16_knob_unit)(model, arm, 'levels', keep, tuple(LEVELS), fits)
    u.update(off_decision=str(path), selected_cells=keep)
    put([u]); print('DEFINED', u['name'], u['episodes'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('build', 'levels', 'prepare', 'dispatch'))
    p.add_argument('dispatch_action', nargs='?', choices=('prepare', 'launch'))
    p.add_argument('--name'); p.add_argument('--model', choices=('pi05', 'groot'))
    p.add_argument('--mode', choices=('full', 'reduced')); p.add_argument('--arm', choices=('original', 'filtered'))
    a = p.parse_args()
    if a.action == 'build':
        build()
    elif a.action == 'levels':
        levels(a.model, a.mode, a.arm)
    elif a.action == 'prepare':
        prepare(a.name)
    else:
        dispatch(a.dispatch_action, a.name)


if __name__ == '__main__':
    main()
