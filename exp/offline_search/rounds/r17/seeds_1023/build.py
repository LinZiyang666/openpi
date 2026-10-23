"""New-seed success-filtered knob batches: definitions only, no prepare/launch/GPU.

One campaign per model x benchmark x steps K x card holds every seed x library size x suite x segment of a
stage (resident reuse, owner 10-23: one model load, YAMLs registered per group). Each group YAML is the old
seed's YAML of the same lane with only the library inputs changed (as longseg_off_1022.build.make_edit):
store/fit/stall paths and digests, teacher_sampler_identity, commit_horizon (segment), periodic_gap (None for
the knob-off stage, the cell's level gap for levels), fit k/kref/early from the fit, fit_identity recomputed;
miss_mode stall_only, guard false, serve_loeo false.

Names carry seed and segment so nothing collides inside a campaign or across seeds: group
<base>_<tag>c<segment>_<level> (e.g. chrg_x3c10_koff; tag = inputs.tag), YAML/bundle sd_<suite>_<size>_<group>;
analysis keys follow the group (knob.common.knob_cell), evidence reads the group's labels. The success filter is
the resident's candidate-row selection keyed by fit digest (campaign selection, and the selection file beside
each fit for filter_failed_knob.runtime).

Deployment is the old seed's for the same lane: DP/Fast-WAM LIBERO longseg_off_1022.workflow.deployment
(H100), DP/Fast-WAM/GR00T MT50 the H100 originals of fa_ls_*_m (GR00T R5), Fast-WAM with the H100 prompt
encoding table, Cosmos fa_co_<m>_f_l (H100), pi0.5 MT50 ffk1015_pi05_mt50_* (4090). Cosmos and pi0.5 keep
their native serving path (no longseg runtime patch; only the longseg port routing). A levels batch takes the
serving fields and the frozen bootstrap of its knob-off batch (resident reuse across the two steps).
"""
from __future__ import annotations

from contextlib import ExitStack
import copy
from pathlib import Path
import shutil
from unittest.mock import patch

import yaml

from exp.offline_search.rounds.r17.longseg_1017 import build as lb, ports
from exp.offline_search.rounds.r17.longseg_1017.io_budget import dump
from exp.offline_search.rounds.r17.longseg_off_1022 import build as lob
from exp.offline_search.rounds.r17.filter_failed_knob.common import read
from . import inputs as ip

HERE = Path(__file__).resolve().parent
EVAL = ip.EVAL
INDEX = HERE/'CAMPAIGNS.json'
TOK = {'dp': 'dp', 'fastwam': 'fw', 'groot': 'gr', 'cosmos': 'co', 'pi05': 'pi'}
# RUN_CONFIGS (10-21): H100 Fast-WAM serves without umT5-xxl, prompt encodings from this table (pods: pod_copy's).
FW_TEXT_TABLE = '/home/exouser/fw_text_tables/table_h100.pt'
_route = ports.route
# Serving fields a levels batch takes from its knob-off definition (as the coordinator may have tuned it), so a
# kept resident can serve both (resident_attach.SIGNATURE); ports follow the lane (ports.route, stage's alias).
SERVING = ('replicas', 'memory_budget_gib', 'server_threads', 'mps', 'dp_batching', 'stage_batching', 'fastwam_batching',
           'fastwam_text_table', 'server_python', 'ffk_dp_capacity_bridge', 'cache_worker_orchestration')


def name(bundle, mode, stage):
    """fa_sd_<model>_<r|f>_<l|m>_<o|k>: knob-off (o) or levels (k) batch of one lane."""
    return f'fa_sd_{TOK[ip.model(bundle)]}_{mode[0]}_{"m" if ip.benchmark(bundle) == "mt50" else "l"}_{"o" if stage == "off" else "k"}'


def text_table(deploy):
    def edit(root, c):
        if deploy:
            deploy(root, c)
        c['fastwam_text_table'] = FW_TEXT_TABLE
    return edit


def deployment(bundle, mode):
    """(clone source, deployment edit, worker labels, YAML template run, long-segment serving) of a lane."""
    model, tok = ip.model(bundle), TOK[ip.model(bundle)]
    if bundle in ('dp_libero', 'fastwam_libero'):
        from exp.offline_search.rounds.r17.longseg_off_1022.workflow import deployment as h100
        on = lob.on_job(f'fa_ls_{tok}_{mode[0]}_f_o_l')
        src, deploy = h100(model, mode, read(EVAL/on['name']/'campaign.json'))
        return Path(src), text_table(deploy) if model == 'fastwam' else deploy, on['workers'], EVAL/on['name'], True
    if bundle in ('dp_mw', 'fastwam_mw', 'groot_mw'):
        on = lob.on_job(f'fa_ls_{tok}_{mode[0]}_f_m')
        deploy = {'groot': lob.groot_r5, 'fastwam': text_table(None)}.get(model)
        return Path(on.get('original_source') or on['source']), deploy, on['workers'], EVAL/on['name'], True
    run = EVAL/(f'fa_co_{mode[0]}_f_l' if bundle == 'cosmos_libero' else
                {'reduced': 'ffk1015_pi05_mt50_reduced_descent', 'full': 'ffk1015_pi05_mt50_full_descent_04'}[mode])
    return run, None, read(run/'FFK_DEFINITION.json')['workers'], run, False


def plain_route(c, root, workers=None):
    """longseg_1017.ports.route without marking the campaign long-segment."""
    c['longseg_1017'] = {}
    _route(c, root, workers)
    c['seeds_1023_transport_rebind'] = c.pop('longseg_1017')['transport_rebind']
    return c


def plain_pin(root, c):
    from exp.offline_search.rounds.r17.filter_failed_knob.build import freeze_bootstrap
    freeze_bootstrap(root, c)


def make_edit(bundle, mode, stage, cells, levels, src, deploy, templates, like=None):
    """clone() hook: one group YAML per (seed, suite, size, segment, level); selection by fit digest."""
    def edit(root, c, seal):
        from exp.offline_search.rounds.r17.filter_failed.build import register_metadata
        from exp.offline_search.rounds.r17.filter_failed_knob.common import sha
        from exp.r17_integration.common.config import config_from_dict
        from exp.r17_integration.common.contracts import digest
        from exp.r17_integration.common.recipe import bundle_identity, recipe_identity, fit_identity
        if deploy:
            deploy(root, c)
        if like:
            prior = read(Path(like)/'campaign.json')
            c.update({k: copy.deepcopy(prior[k]) for k in SERVING if k in prior})
        template = {}
        for g in read(templates/'assets/CACHE_STUDY.json')['groups']:
            template.setdefault(g['suite'], (g, templates/'assets'/g['file']))
        made = {seed: ip.products(bundle, seed) for seed in sorted({cell[0] for cell in cells})}
        groups, selection, summaries, fits = [], {}, {}, {}
        for seed, suite, size, segment in cells:
            p = made[seed][suite, size]
            art = p['native'] if p['native']['commit'] == segment else p['c5']
            if art['commit'] != segment:
                raise ValueError(f'{bundle} seed {seed} {suite}/{size}: no C{segment} inputs')
            base = ip.inputs(bundle, seed)
            if (seed, art['fit']) not in fits:
                fits[seed, art['fit']] = read(base/art['fit']/'READY.json')['receipt']['config']
            row0, path = template[suite]
            for level in levels:
                gap = None if level == 'koff' else art['gaps'][level]
                doc = yaml.safe_load(path.read_text())
                r = doc['recipe']
                doc['adapter_binding']['commit_horizon'] = segment
                r.update(store_path=str(base/p['view']), store_digest=p['store_digest'],
                         teacher_sampler_identity=p['teacher_sampler_identity'],
                         fitted_path=str(base/art['fit']), fitted_digest=art['fit_digest'],
                         stall_path=str(base/art['stall']), stall_digest=art['stall_digest'],
                         miss_mode='stall_only', guard=False, periodic_gap=gap, serve_loeo=False,
                         **{k: fits[seed, art['fit']][k] for k in ('k', 'kref', 'early')})
                register_metadata(Path(c['filter_failed_source']), doc)
                r['fit_identity'] = None  # the template's; validate_recipe compares a present one
                r['fit_identity'] = fit_identity(config_from_dict(doc))
                config = config_from_dict(doc)
                identity = bundle_identity(config)
                group = f'{row0["group"].split("_")[0]}_{ip.tag(seed)}c{segment}_{level}'
                stem = f'{suite}_{size}_{group}'
                row = {k: v for k, v in copy.deepcopy(row0).items()
                       if k not in ('tasks', 'decision_ir_calibration', 'task_namespace', 'longseg_off', 'template')}
                row.update(group=group, suite=suite, library_size=size, library=str(ip.library(bundle, suite, seed)),
                           seed=seed, segment=segment, labels=ip.labels(bundle, suite, seed),
                           settings=dict(row0['settings'], commit=segment), view_rel=p['view'],
                           fit=str(base/art['fit']), fit_digest=art['fit_digest'], fit_rel=art['fit'],
                           stall=str(base/art['stall']), stall_digest=art['stall_digest'], stall_rel=art['stall'],
                           periodic_gap=gap, serve_loeo=False, indices=list(range(50)), num_trials_per_task=50,
                           bundle_identity=identity, recipe_identity=recipe_identity(config),
                           bundle_id=f'sd_{stem}_{identity[:12]}', file=f'cache_yamls/sd_{stem}.yaml')
                (root/'assets'/row['file']).write_text(yaml.safe_dump(doc, sort_keys=False))
                row['sha256'] = sha(root/'assets'/row['file'])
                groups.append(row)
            selection[art['fit_digest']] = p['summary']['kept_episode_ids']
            summaries[f'{suite}/{size}/{ip.tag(seed)}'] = dict(
                {k: p['summary'][k] for k in ('collected', 'kept', 'failed', 'empty_tasks')}, labels=ip.labels(bundle, suite, seed))
        seal['groups'] = groups
        manifest_path = root/'assets/cache_inputs/manifest.json'
        manifest = read(manifest_path)
        manifest['groups'] = copy.deepcopy(groups)
        manifest['digest'] = digest({k: v for k, v in manifest.items() if k != 'digest'})
        dump(manifest_path, manifest)
        c['filter_failed_knob'] = dict(method='frozen_fit_candidate_rows_only', refit=False, selection=selection,
                                       cells=summaries, source=str(src), arm='filtered')
        c['seeds_1023'] = dict(bundle=bundle, mode=mode, stage=stage, levels=list(levels), seeds=sorted(made),
                               segments=sorted({cell[3] for cell in cells}),
                               inputs={str(s): str(ip.inputs(bundle, s)) for s in made}, templates=str(templates))
    return edit


def define(bundle, mode, stage, cells, levels, like=None):
    """Build (or return the existing definition of) one batch. cells: (seed, suite, size, segment).

    like: the lane's knob-off definition; a levels batch takes its serving fields and frozen bootstrap."""
    src, deploy, workers, templates, long = deployment(bundle, mode)
    n = name(bundle, mode, stage)
    edit = make_edit(bundle, mode, stage, cells, levels, src, deploy, templates, like)
    with ExitStack() as stack:
        stack.enter_context(patch.object(lb, 'COMMANDS', lob.COMMANDS))
        if not long:  # Cosmos, pi0.5: no longseg runtime patch or staging path
            stack.enter_context(patch.object(lb, 'pin', plain_pin))
            stack.enter_context(patch.object(ports, 'route', plain_route))
        job = lb.clone(src, n, workers=workers, groups=[], mode=mode, stage=None, edit=edit)
    if like:
        root, prior = Path(job['run']), read(Path(like)/'campaign.json')
        c = read(root/'campaign.json')
        shutil.copyfile(Path(like)/'assets/ffk_bootstrap.py', root/'assets/ffk_bootstrap.py')
        c['ffk_bootstrap_sha256'] = prior['ffk_bootstrap_sha256']
        if c.get('longseg_1017'):
            c['longseg_1017']['bootstrap_sha256'] = prior['ffk_bootstrap_sha256']
        dump(root/'campaign.json', c)
        job['bootstrap_from'] = str(like)
    job.update(stage=stage, arm='filtered', bundle=bundle, levels=list(levels), longseg_runtime=long,
               cells=[list(cell) for cell in cells])
    dump(Path(job['run'])/'FFK_DEFINITION.json', job)
    catalog = read(INDEX) if INDEX.exists() else dict(jobs=[])
    catalog['jobs'] = [j for j in catalog['jobs'] if j['name'] != n]+[job]
    dump(INDEX, catalog)
    print('DEFINED', n, job['episodes'], job['gpu'], flush=True)
    return job
