"""Eight-group cache component study (task #135): original library, pi0.5/GR00T, LIBERO, R16 conductor.

`build` and `plan` only write definitions and a CPU plan. `prepare`/`launch` are the
coordinator's chain items (see commands/prepare_fa_cs_*_r.sh, launch_fa_cs_*_r.sh).

Each cell's fit is the current formal-A reduced original knob-off fit, unchanged
(same library rows, PCA, metric, corrector heads). Only the serving wrapper changes:
the old R8 no-progress judge is dropped, so no group can call the model.
  c   C5  cache only            cr  C5  cache + corrector
  ch  C10 cache only            chr C10 cache + corrector
C10 = five controls plus one blind anchor tail (--os-blind), exactly as R10 A / R15 AC_dist.
The four stall-gated groups need the Q3 trigger, which the R16 stack does not have;
they are not built here (HANDBACK.md).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
from pathlib import Path
from unittest.mock import patch

from ..common import read, dump, sha

HERE = Path(__file__).resolve().parent
KNOB = HERE.parent
REPO = KNOB.parents[4]
EVAL = Path('/data/mw_finetune/eval')
SOURCE = KNOB/'formal_a_1016/runs/ffk1015_{model}_libero_reduced_formal_A_original/group.json'
CAMPAIGNS = HERE/'CAMPAIGNS.json'
NAMES = {'pi05': 'fa_cs_pi_r', 'groot': 'fa_cs_gr_r'}
# group -> (corrector, long horizon C10)
GROUPS = {'c': (False, False), 'cr': (True, False), 'ch': (False, True), 'chr': (True, True)}
NOT_BUILT = ('cg', 'crg', 'chg', 'chrg')
SPEC = 'exp.offline_search.rounds.r10.recipe.recipe:R10Recipe'
PAIRS = [[t, i] for t in range(10) for i in range(50)]
SERVER_CHECKOUT = '/home/weiland/projects/openpi_r16_k1ref_server'
ISLAND = '/scratch/zixuans8/openpi_r16_conductor_v1'
PORTS = {'pi05': '23150,23151,23152', 'groot': '23150,23151,23152,23153'}
DRIVER_PORT = 23159


def short(cell):
    _, suite, size, _ = cell.split('_')
    return ('l' if suite == 'l10' else 's') + size


def arg(args, flag):
    return args[args.index(flag)+1]


def fresh(root):
    for n in ('launch.json', 'journal.jsonl', 'DONE.json', 'LEGACY_AUTOSCALE.json', 'WORKER_HOST_SELECTION.json'):
        if (root/n).exists():
            raise ValueError(f'Never rebuild a prepared/launched run: {root/n}')


def fit_blob(source_row, corrector):
    """Same fitted arrays; R10Recipe wrapper without the R8 judge (no MISS path)."""
    from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM
    from exp.offline_search.rounds.r10.recipe.recipe import R10Recipe, _CorrectedCache
    path = Path(arg(source_row['plugin_args'], '--os-fit-artifact'))
    blob = pickle.loads(path.read_bytes())
    knob = blob['method']
    base = knob.inner.base
    if type(base) is not _CorrectedCache or any(knob.knob_settings['K']):
        raise ValueError(f'{path}: expected the knob-off R16 corrected cache')
    if not corrector:
        base.__class__ = BlindAWM   # drops only the residual head application
    base.invalidate_anchor()
    kwargs = dict(library=source_row['kwargs']['library'], episode_subset=source_row['kwargs']['episode_subset'])
    method = R10Recipe(**kwargs)
    method.inner, method.row_subset = base, knob.row_subset
    method.uses_nonlibrary_action = corrector
    method.name = ('CS1019_corrected__' if corrector else 'CS1019_cache__') + base.name
    method.fit_info = dict(knob.fit_info, cache_study_1019=dict(source_fit=str(path), source_fit_sha256=sha(path),
                           corrector=corrector, judge=None, refit=False))
    return dict(method=method, registered=blob['registered'], spec=SPEC, kwargs=kwargs, cell=blob['cell'],
                fit_s=blob.get('fit_s'), source_fit=str(path), source_fit_sha256=sha(path))


def build_model(model):
    from exp.offline_search.rounds.r16 import conductor as rc, analyze as a
    name = NAMES[model]; root = EVAL/name; defs = HERE/'runs'/name/'definitions'
    fresh(root)
    source = rc.read_group(Path(str(SOURCE).format(model=model)))
    offs = {e['row']['r16_cell']: e for e in source['entries'] if e['row']['arm'].endswith('_off')}
    if len(offs) != 8:
        raise ValueError(f'{model}: expected 8 knob-off cells, got {sorted(offs)}')
    entries = []
    for cell in sorted(offs, key=lambda c: (c.split('_')[1] != 'l10', int(c.split('_')[2]))):
        src = offs[cell]['row']
        if read(src['manifest']) != PAIRS:
            raise ValueError(f'{cell}: manifest is not the official A 10x50 template')
        for corrector in (False, True):
            d = defs/f'{name}_{short(cell)}_{"cr" if corrector else "c"}'
            d.mkdir(parents=True, exist_ok=True)
            (d/'fit.pkl').write_bytes(pickle.dumps(fit_blob(src, corrector), protocol=4))
            (d/'config.yaml').write_bytes(Path(src['yaml']).read_bytes())
            (d/'eval500.json').write_bytes(Path(src['manifest']).read_bytes())
            dump(d/'ROOT.json', dict(definitions_sha256={n: sha(d/n) for n in ('fit.pkl', 'config.yaml', 'eval500.json')},
                                     fit_sha256={str(d/'fit.pkl'): sha(d/'fit.pkl')}, init_pool='A', recording=False,
                                     source_arm=src['arm'], source_fit_sha256=offs[cell]['fit_sha256']))
        for group, (corrector, long) in GROUPS.items():
            d = defs/f'{name}_{short(cell)}_{"cr" if corrector else "c"}'
            args = src['plugin_args'][:src['plugin_args'].index('--os-blind')]   # --os-root .. --os-tokens off
            args += (['--os-blind'] if long else []) + ['--os-fit-artifact', str(d/'fit.pkl'),
                     '--os-rollout-seed', arg(src['plugin_args'], '--os-rollout-seed'), '--os-r14-audit']
            row = copy.deepcopy(src)
            for k in ('matrix', 'remote_yaml', 'remote_matrix', 'yaml_patch'):
                row.pop(k, None)
            row.update(arm=f'{name}_{short(cell)}_{group}', method=SPEC,
                       kwargs=dict(library=src['kwargs']['library'], episode_subset=src['kwargs']['episode_subset']),
                       plugin_args=args, yaml=str(d/'config.yaml'), yaml_sha256=sha(d/'config.yaml'),
                       manifest=str(d/'eval500.json'), judge=None, full_model=False, episodes=500,
                       r14_descent_level='cs_'+group, cache_study_group=group, corrector=corrector,
                       commit_controls=10 if long else 5, stall_gate=False, model_calls_allowed=False,
                       source_off_arm=src['arm'])
            entries.append(dict(row=row, source_root=str(d), source_seal_sha256=sha(d/'ROOT.json'),
                                fit_sha256=sha(d/'fit.pkl'), yaml_sha256=sha(d/'config.yaml'),
                                pairs=copy.deepcopy(PAIRS), dispatch_pairs=copy.deepcopy(PAIRS),
                                cache_study_group=group, source_off_arm=src['arm']))
    group = {k: v for k, v in source.items() if k not in ('entries', 'definitions_sha256', 'group_sha256')}
    group.update(root=str(root), entries=entries, phase='test', references=None, formal_test_A=True,
                 original_episode_uids=False, yaml_id_is_original_arm=False, cache_study_1019=dict(
                     groups=list(GROUPS), not_built=list(NOT_BUILT), source_group=str(Path(str(SOURCE).format(model=model))),
                     source_group_sha256=source['group_sha256']))
    group['definitions_sha256'] = {str(p): sha(p) for p in sorted(defs.rglob('*')) if p.is_file()}
    group['group_sha256'] = a.sha(rc.canonical(group))
    root.mkdir(parents=True, exist_ok=True)
    dump(root/'group.json', group)
    dump(root/'ROOT.json', dict(group_sha256=group['group_sha256'], definitions_sha256=group['definitions_sha256']))
    return dict(name=name, run=str(root), model=model, mode='reduced', arm='original', stage='cache_study',
                pipeline='R16', gpu='RTX 4090', replicas=3 if model == 'pi05' else 4, batch=1,
                workers={'timan107': 64}, episodes=len(entries)*500, groups=list(GROUPS), not_built=list(NOT_BUILT),
                k=group['k'], source=str(Path(str(SOURCE).format(model=model)).parent))


def build():
    jobs = [build_model(m) for m in NAMES]
    dump(CAMPAIGNS, dict(jobs=jobs, episodes=sum(j['episodes'] for j in jobs)))
    for j in jobs:print('DEFINED', j['name'], j['episodes'], flush=True)


def job(name):
    return next(j for j in read(CAMPAIGNS)['jobs'] if j['name'] == name)


def plan():
    """CPU only: native group seal, native planner, per-arm switches, fit equality, one plugin load per C."""
    from exp.offline_search.rounds.r16 import conductor as rc
    from exp.offline_search.closed_loop import plugin
    from openpi.conductor.task import ServerEndpoint
    from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM
    from exp.offline_search.rounds.r10.recipe.recipe import _CorrectedCache
    out = dict(jobs=[], total_episodes=0)
    uids = set()
    for j in read(CAMPAIGNS)['jobs']:
        group = rc.read_group(Path(j['run'])/'group.json')
        graph = rc.R16Strategy(group).plan({}, {'cpu': ServerEndpoint('localhost', 23150)})
        planned = [e for s in graph.stages.values() for e in s.episodes]
        assert len(planned) == j['episodes'] == len({e.task_uid for e in planned})
        assert not uids & {e.task_uid for e in planned}; uids |= {e.task_uid for e in planned}
        rows, cells = [], {}
        for e in group['entries']:
            r = e['row']; g = r['cache_study_group']; corrector, long = GROUPS[g]; args = r['plugin_args']
            assert sorted(map(tuple, e['pairs'])) == sorted(map(tuple, PAIRS)) and r['init_pool'] == 'A'
            assert ('--os-blind' in args) == long and '--os-judge' not in args and '--os-policy-tail' not in args
            blob = pickle.loads(Path(arg(args, '--os-fit-artifact')).read_bytes())
            base = blob['method'].inner
            assert type(base) is (_CorrectedCache if corrector else BlindAWM) and blob['spec'] == r['method']
            assert blob['kwargs'] == r['kwargs'] and blob['cell'] == r['cell']
            source = pickle.loads(Path(blob['source_fit']).read_bytes())['method'].inner.base
            source.invalidate_anchor()
            assert pickle.dumps(vars(source), protocol=4) == pickle.dumps(vars(base), protocol=4), 'fitted arrays changed'
            cells.setdefault(r['r16_cell'], []).append(g)
            rows.append(dict(arm=r['arm'], cell=r['r16_cell'], group=g, corrector=corrector,
                             commit_controls=10 if long else 5, base_class=type(base).__name__,
                             blind='--os-blind' in args, judge=None, episodes=len(e['pairs']),
                             library=r['library'], fit_sha256=e['fit_sha256']))
        assert all(sorted(v) == sorted(GROUPS) for v in cells.values()) and len(cells) == 8
        loads = []
        for e in group['entries']:
            r = e['row']
            if (short(r['r16_cell']), r['cache_study_group']) not in (('l50', 'c'), ('s500', 'chr')):continue
            opts, rest = plugin.parse_cli(['--os-method', r['method'], '--os-cell', r['cell'], *r['plugin_args'],
                                           '--os-kwargs', json.dumps(r['kwargs']), '--os-log-dir', str(HERE/'cpu'/r['arm'])])
            assert not rest
            opts.os_miss_steps = group['k']
            runtime = plugin.PluginRuntime(opts, group['model']); assert runtime.method is not None
            loads.append(dict(arm=r['arm'], method=runtime.method_name, blind=runtime.blind,
                              judge=runtime.judge, library_rows=runtime.L)); runtime.flush_all()
        out['jobs'].append(dict(name=j['name'], model=j['model'], k=group['k'], episodes=len(planned),
                                unique_task_uids=len({e.task_uid for e in planned}), stages=len(graph.stages),
                                group_sha256=group['group_sha256'], arms=rows, cpu_plugin_loads=loads))
        out['total_episodes'] += len(planned)
    out['unique_task_uids_all'] = len(uids)
    dump(HERE/'PLAN.json', out)
    print('PLAN', out['total_episodes'], 'episodes', len(uids), 'unique UIDs', flush=True)


def prepare(name):
    """Coordinator chain item: stage the dynamic-worker release on timan107 (tether exec + direct rsync)."""
    from .. import vla_workers as vw
    j = job(name)
    if (Path(j['run'])/'DONE.json').exists():raise ValueError(f'{name} is already DONE')
    # Repeating prepare after an interrupted launch is allowed: the group resumes from its journal.
    vw.prepare_r16_job(j, 'timan107')


def serve():
    """Launch body, run from the R16 server checkout; argv as the native R16 launch."""
    from exp.offline_search.rounds.r17.autoscale_1017.switch import enabled
    group = Path(sys.argv[sys.argv.index('--group')+1]); j = job(group.parent.name)
    if enabled():
        # Same re-exec as legacy_release.coordinator, but booting this module so the
        # job comes from CAMPAIGNS.json instead of formal_a_1016/QUEUE.json.
        record = read(group.parent/'LEGACY_AUTOSCALE.json')
        if os.environ.get('OPENPI_AUTOSCALE_COORDINATOR') != record['digest']:
            env = dict(os.environ, OPENPI_AUTOSCALE_COORDINATOR=record['digest'], OPENPI_AUTOSCALE_REMOTE=record['remote'])
            local = record['local']
            env['PYTHONPATH'] = local+':'+local+'/src:'+env.get('PYTHONPATH', '')+':'+str(REPO)
            boot = ('import sys;sys.path[:0]='+repr([local, local+'/src'])+';from '+__name__+' import serve;serve()')
            os.execve(sys.executable, [sys.executable, '-c', boot, *sys.argv[1:]], env)
    from .. import vla_workers as vw
    with patch.object(vw, 'lookup', lambda name: j):
        vw.run_r16_selected()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('build', 'plan', 'prepare'))
    p.add_argument('--name')
    args = p.parse_args()
    if args.action == 'build':build()
    elif args.action == 'plan':plan()
    else:prepare(args.name)


if __name__ == '__main__':main()
