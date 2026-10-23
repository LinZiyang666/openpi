"""Long segment OFF (commit_horizon 5 = replan_steps 5), success-filtered library: R17 knob-descent builds.

Build only: no prepare, launch, GPU or remote call. Each campaign is its long-segment ON campaign
(fa_ls_*, read only) with exactly these YAML changes per group:
  - adapter_binding.commit_horizon 10 -> 5;
  - recipe.fitted_path/fitted_digest -> the library's existing C5 fit, and recipe.stall_path/stall_digest
    -> its C5 Q3 artifact (exec_steps 5): the pair the C5 gated cache-study groups (cg/crg) served;
  - recipe.fit_identity recomputed (it hashes commit_horizon); miss_mode stall_only, guard false.
periodic_gap is the ON campaign's value for the same level/suite/size (koff: None); no recalibration.
corrector, early, k, kref, ordered_tasks, serve_loeo and store are unchanged (k/kref/early would follow
the C5 fit's own settings if they differed; they do not).

The filtered library uses the ON campaigns' own mechanism: the resident takes the original fit and keeps
only the successful library episodes' candidate rows (filter_failed_knob.runtime.install, selection keyed
by fit digest). The kept episodes are the ON campaign's, which equal the candidate_filter of
fit_keep_success_1015_norefit_*. The original C5 Q3 artifact is bound to the original C5 fit
(cache_recipe.load_stall compares its fitted_digest and attests its metric with that fit).

    python -m exp.offline_search.rounds.r17.longseg_off_1022.build mt50
    python -m exp.offline_search.rounds.r17.longseg_off_1022.workflow initialize   # LIBERO OFF + levels units
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
from unittest.mock import patch

import yaml

from exp.offline_search.rounds.r17.longseg_1017 import build as lb
from exp.offline_search.rounds.r17.longseg_1017.audit import EVAL, HERE as LS
from exp.offline_search.rounds.r17.longseg_1017.io_budget import dump
from exp.offline_search.rounds.r17.filter_failed_knob.common import read, HERE as KNOB

HERE = Path(__file__).resolve().parent
COMMANDS = KNOB/'commands'
INDEX = HERE/'CAMPAIGNS.json'
MAPPING = HERE/'MAPPING.json'
TOK = {'groot': 'gr', 'dp': 'dp', 'fastwam': 'fw'}
# Cache studies whose C5 gated groups (cg/crg, stall_only) served each library's C5 fit and Q3 stall.
C5_STUDIES = {('groot', 'mt50'): 'csgr_h5', ('dp', 'mt50'): 'csdp_1504', ('fastwam', 'mt50'): 'csfw_k3',
              ('dp', 'libero'): 'csztd_r3_l48', ('fastwam', 'libero'): 'fastwamc6066547406'}
MT50_STUDIES = {'groot': ('gr_k2', 'groot_mw_mt50'), 'dp': ('dp_k10', 'dp_mw_mt50'), 'fastwam': ('fw_k3', 'fastwam_mw_mt50')}
# H100 Fast-WAM LIBERO deployment (R2/B16/MPS, 70 GiB) for the reduced lane that ran on the 4090.
H100_FW_LIBERO = EVAL/'ffk1015_fastwam_libero_full_formal_A_filtered'
# DP LIBERO CUDA-graph self-check frames; the H100 has none (RUN_CONFIGS pitfall). assets/ is read first.
DP_PROBE_FRAMES = Path('/data/mw_finetune/qualification/assets/dp_libero_gpu_fix1_v1/wire_observations.npz')


def name(on):
    return 'fa_lo_'+on[len('fa_ls_'):]


def on_job(on):
    return next(j for j in read(LS/'CAMPAIGNS.json')['jobs'] if j['name'] == on)


_refs = {}


def c5_refs(model, benchmark):
    """store_digest -> the C5 fit/stall pair of that library, as its C5 gated cache-study groups used it."""
    key = model, benchmark
    if key in _refs:
        return _refs[key]
    study = EVAL/C5_STUDIES[key]
    result = {}
    for g in read(study/'assets/CACHE_STUDY.json')['groups']:
        doc = yaml.safe_load((study/'assets'/g['file']).read_text())
        r, b = doc['recipe'], doc['adapter_binding']
        if b['commit_horizon'] != 5 or r.get('miss_mode') != 'stall_only':
            continue
        row = result.setdefault(r['store_digest'], dict(
            suite=g['suite'], size=g['library_size'], study=study.name, fit=r['fitted_path'],
            fit_digest=r['fitted_digest'], stall=r['stall_path'], stall_digest=r['stall_digest'], groups=[]))
        row['groups'].append(g['group'])
        if r['corrector']:
            row['c5_gated_fit_identity'] = r['fit_identity']
    if benchmark == 'libero':
        # Size 500 had no C5 fit: fit500.py made them with cache_study.fit_inputs' procedure.
        from .fit500 import ROOTS
        new = ROOTS[model+'_libero']
        for fit in sorted((new/'assets/cache_inputs').glob('*/n*_C5/fit')):
            ready = read(fit/'READY.json')
            result.setdefault(ready['receipt']['store_digest'], dict(
                suite=fit.parent.parent.name, size=ready['receipt']['library_size'], study=new.name, fit=str(fit),
                fit_digest=ready['digest'], stall=str(fit.parent/'stall'),
                stall_digest=read(fit.parent/'stall/READY.json')['digest'], groups=['fit500 (cache_study.fit_inputs)']))
    for row in result.values():
        fit, stall = read(Path(row['fit'])/'READY.json'), read(Path(row['stall'])/'READY.json')
        params = read(Path(row['stall'])/'stall.json')
        row.update(fit_receipt=dict(fit['receipt']['config'], corrector=fit['receipt']['corrector'],
                                    store_digest=fit['receipt']['store_digest']),
                   stall_receipt=dict(fitted_digest=stall['fitted_digest'], store_digest=stall['store_digest'],
                                      commit_horizon=stall.get('commit_horizon'), exec_steps=params['exec_steps'],
                                      commit_controls=params['commit_controls']))
        if benchmark == 'mt50':
            # The same artifacts under /data/r17_library/studies_mt50 (C5/fit, C5/stall, filtered fit).
            base = Path('/data/r17_library/studies_mt50/{}/artifacts/n{}/{}/C5'.format(
                *MT50_STUDIES[model][:1], row['size'], MT50_STUDIES[model][1]))
            filtered = next(base.glob('fit_keep_success_1015_norefit_*'), None)
            row['library_copy'] = dict(fit=str(base/'fit'), fit_digest=read(base/'fit/READY.json')['digest'],
                                       stall=str(base/'stall'), stall_digest=read(base/'stall/READY.json')['digest'])
            if filtered is not None:
                f = read(filtered/'READY.json')
                row['filtered_fit'] = dict(path=str(filtered), digest=f['digest'],
                                           store_digest=f['receipt']['store_digest'],
                                           kept=sorted(f['candidate_filter']['kept_episodes']))
        else:
            filtered = None
            for run in sorted(EVAL.glob('ff1015_*_li_*')):
                path = run/'assets/cache_inputs'/row['suite']/f'n{row["size"]}_C5/fit'
                if (path/'READY.json').exists():
                    f = read(path/'READY.json')
                    if (f.get('candidate_filter') or {}).get('original_fit_digest') == row['fit_digest']:
                        filtered = dict(path=str(path), digest=f['digest'], store_digest=f['receipt']['store_digest'],
                                        kept=sorted(f['candidate_filter']['kept_episodes']))
                        break
            if filtered:
                row['filtered_fit'] = filtered
    _refs[key] = result
    return result


def make_edit(model, benchmark, items, selection_on, deploy=None, record=None):
    """clone() hook: write the C5 group YAMLs/rows, re-key the candidate selection, set deployment fields.

    items: (row, template YAML path, periodic_gap). selection_on: the ON campaign's selection
    (fit digest -> kept library episodes)."""
    def edit(root, c, seal):
        from exp.offline_search.rounds.r17.filter_failed.build import register_metadata
        from exp.r17_integration.common.config import config_from_dict
        from exp.r17_integration.common.contracts import digest
        from exp.r17_integration.common.recipe import bundle_identity, recipe_identity, fit_identity
        refs = c5_refs(model, benchmark)
        if deploy:
            deploy(root, c)
        groups, selection, rows, built = [], {}, [], set()
        for row, template, gap in items:
            doc = yaml.safe_load(Path(template).read_text())
            r = doc['recipe']
            ref = refs[r['store_digest']]
            before = dict(fit=r['fitted_digest'], stall=r.get('stall_digest'), commit=doc['adapter_binding']['commit_horizon'],
                          gap=r.get('periodic_gap'), fit_identity=r.get('fit_identity'))
            doc['adapter_binding']['commit_horizon'] = 5
            r.update(fitted_path=ref['fit'], fitted_digest=ref['fit_digest'], stall_path=ref['stall'],
                     stall_digest=ref['stall_digest'], miss_mode='stall_only', guard=False, periodic_gap=gap)
            fit_settings = {k: ref['fit_receipt'][k] for k in ('k', 'kref', 'early')}
            settings_changed = {k: [r[k], v] for k, v in fit_settings.items() if r[k] != v}
            r.update(fit_settings)
            register_metadata(Path(c['filter_failed_source']), doc)
            r['fit_identity'] = fit_identity(config_from_dict(doc))
            config = config_from_dict(doc)
            identity = bundle_identity(config)
            new = copy.deepcopy(row)
            for key in ('decision_ir_calibration', 'task_namespace'):
                new.pop(key, None)
            inputs = ref['fit'].split('/cache_inputs/')[0]+'/cache_inputs'
            stem = f'{row["suite"]}_{row["library_size"]}_{row["group"]}'
            new.update(settings=dict(row['settings'], commit=5), fit=ref['fit'], fit_digest=ref['fit_digest'],
                       fit_rel=os.path.relpath(ref['fit'], inputs), stall=ref['stall'], stall_digest=ref['stall_digest'],
                       stall_rel=os.path.relpath(ref['stall'], inputs), periodic_gap=gap,
                       bundle_identity=identity, recipe_identity=recipe_identity(config),
                       bundle_id='lo_'+stem+'_'+identity[:12], file='cache_yamls/lo_'+stem+'.yaml',
                       longseg_off=dict(template=str(template), on_fit_digest=before['fit']))
            path = root/'assets'/new['file']
            path.write_text(yaml.safe_dump(doc, sort_keys=False))
            from exp.offline_search.rounds.r17.filter_failed_knob.common import sha
            new['sha256'] = sha(path)
            kept = selection_on[before['fit']]
            selection[ref['fit_digest']] = kept
            groups.append(new)
            built.add((row['suite'], row['library_size']))
            rows.append(dict(suite=row['suite'], size=row['library_size'], group=row['group'], before=before,
                             after=dict(fit=ref['fit_digest'], stall=ref['stall_digest'], commit=5, gap=gap,
                                        fit_identity=r['fit_identity'], k=r['k'], kref=r['kref'], early=r['early'],
                                        miss_mode=r['miss_mode'], guard=r['guard'], serve_loeo=r['serve_loeo'],
                                        corrector=r['corrector']),
                             settings_changed=settings_changed, kept_episodes=len(kept),
                             file=new['file']))
        seal['groups'] = groups
        manifest_path = root/'assets/cache_inputs/manifest.json'
        manifest = read(manifest_path)
        manifest['groups'] = copy.deepcopy(groups)
        manifest['digest'] = digest({k: v for k, v in manifest.items() if k != 'digest'})
        dump(manifest_path, manifest)
        old = c.get('filter_failed_knob', {})
        c['filter_failed_knob'] = dict(method='frozen_fit_candidate_rows_only', refit=False, selection=selection,
            cells={k: v for k, v in (old.get('cells') or {}).items()
                   if tuple((k.split('/')[0], int(k.split('/')[1]))) in built},
            source=old.get('source'), arm='filtered')
        c['longseg_off_1022'] = dict(commit_horizon=5, replan_steps=5, periodic_gap='ON campaign value; no recalibration',
            fit='original C5 fit + resident candidate-row selection (kept episodes = ON = fit_keep_success_1015_norefit_*)',
            stall='C5 Q3 (exec_steps 5) of the C5 gated cache-study groups of the same library',
            c5_studies=sorted({refs[g]['study'] for g in refs}))
        if record is not None:
            record.extend(rows)
    return edit


def groot_r5(root, c):
    """Owner 10-20: GR00T at most 4-5 replicas; the R11 source becomes R5/36 GiB (as fa_ls_gr_f_f_m)."""
    c.update(replicas=5, memory_budget_gib=36.0,
             groot_replicas_owner_1020='owner 10-20: never 11 GR00T replicas, 4-5 are enough (R11 ran the H100 at 5% GPU)')
    if isinstance(c.get('cache_worker_orchestration'), dict) and 'replicas' in c['cache_worker_orchestration']:
        c['cache_worker_orchestration']['replicas'] = 5


def dp_libero_h100(root, c):
    """The ON DP LIBERO lanes ran on the 4090; same R4/B8/5 ms/MPS/40 GiB resident on the H100 instead."""
    before = {k: c.get(k) for k in ('server_node', 'public_port', 'server_python')}
    c.update(server_node='h100', public_port=23180)
    for key in ('local_worker_exception', 'execution_host', 'execution_host_rebind'):
        c.pop(key, None)
    shutil.copyfile(DP_PROBE_FRAMES, root/'assets/wire_observations.npz')
    c['longseg_off_1022_rehost'] = dict(before=before, after={k: c.get(k) for k in before},
        probe_frames=str(DP_PROBE_FRAMES), reason='4090 not used; LIBERO serves on the H100')


def fw_reduced_h100(on_campaign):
    def deploy(root, c):
        c['cache_plan_name'] = on_campaign['cache_plan_name']
        c['longseg_off_1022_rehost'] = dict(deployment=str(H100_FW_LIBERO), yamls='fa_ls_fw_r_f_o_l (K1)',
            reason='4090 not used; H100 Fast-WAM LIBERO resident (R2/B16/MPS, 70 GiB, no text cache) serves K1 YAMLs')
    return deploy


def pure_edit(templates, deploy=None, plan_name=None):
    """longseg_1017.build.variant(kind='pure') over template rows of another source (their K), then deployment.

    One row per suite (the largest library size, as variant keeps), recipe disabled, commit 5, 50 inits."""
    def edit(root, c, seal):
        from dataclasses import asdict
        from exp.offline_search.rounds.r17.filter_failed.build import register_metadata
        from exp.offline_search.rounds.r17.filter_failed_knob.common import sha
        from exp.r17_integration.common.config import config_from_dict, RecipeConfig
        from exp.r17_integration.common.recipe import bundle_identity, recipe_identity
        if deploy:
            deploy(root, c)
        candidates = {}
        for row, path in sorted(templates, key=lambda t: t[0]['library_size']):
            candidates[row['suite']] = row, path
        result = []
        for old, path in candidates.values():
            row, doc = copy.deepcopy(old), yaml.safe_load(Path(path).read_text())
            base = row['group'].split('_')[0]
            row.update(group=base+'_pure', periodic_gap=0., settings=dict(corrector=False, gated=False, commit=5, alias=None))
            doc.update(enabled=False, checkpoints={}, recipe=asdict(RecipeConfig(corrector=False)))
            doc['adapter_binding']['commit_horizon'] = 5
            row.update(indices=list(range(50)), serve_loeo=False, num_trials_per_task=50)
            row.pop('tasks', None)
            register_metadata(Path(c['filter_failed_source']), doc)
            cfg = config_from_dict(doc)
            identity = bundle_identity(cfg)
            stem = 'ls_'+row['suite']+'_'+str(row['library_size'])+'_'+row['group']
            row.update(bundle_identity=identity, recipe_identity=recipe_identity(cfg), bundle_id=stem+'_'+identity[:12],
                       file='cache_yamls/'+stem+'.yaml', template=str(path))
            (root/'assets'/row['file']).write_text(yaml.safe_dump(doc, sort_keys=False))
            row['sha256'] = sha(root/'assets'/row['file'])
            result.append(row)
        seal['groups'] = result
        c['filter_failed_knob'] = dict(method='frozen_fit_candidate_rows_only', refit=False, selection={}, cells={}, arm='unfiltered')
        c['longseg_1017_control_kind'] = 'pure'
        if plan_name:
            c['cache_plan_name'] = plan_name
    return edit


REFERENCES = HERE/'REFERENCES.json'
POD = 'H200 MIG 1g.18gb'  # jupyter-ziyang10 and jupyter-ziyang10-02: the same MIG slice hardware
GROOT_POD, FW_POD = ('jupyter-ziyang10', 23193), ('jupyter-ziyang10-02', 23192)


def job(name, src, *, model, benchmark, mode, workers, kind=None, edit=None, groups=None):
    with patch.object(lb, 'COMMANDS', COMMANDS):
        j = lb.clone(src, name, workers=workers, groups=groups, mode=mode, stage=None, kind=kind, edit=edit)
    j.update(stage='pure', arm='pure_reference', commit_horizon=5)
    dump(Path(j['run'])/'FFK_DEFINITION.json', j)
    catalog = read(INDEX)
    catalog['jobs'] = [x for x in catalog['jobs'] if x['name'] != name]+[j]
    dump(INDEX, catalog)
    return j


def dp_mt50_h100():
    """H100 C5 pure references for DP MT50 (the knob-off batches run on an H100: the H100 or h100-sean),
    from the same H100 originals as fa_lo_dp_*_f_m, as pure_refs builds fa_lo_{gr,fw}_*_b_m."""
    registry = read(REFERENCES)
    for mode in ('reduced', 'full'):
        on, n = on_job(f'fa_ls_dp_{mode[0]}_f_m'), f'fa_lo_dp_{mode[0]}_b_m'
        job(n, on.get('original_source') or on['source'], model='dp', benchmark='mt50', mode=mode,
            workers=on['workers'], kind='pure')
        registry['references'][f'dp:mt50:{mode}:H100'] = dict(run=str(EVAL/n), new=True, c5_pure=True)
    dump(REFERENCES, registry)
    return registry


def pure_refs():
    """Same-card plain C5 pure references (model called every 5 controls), mirroring fa_ls_*_b_*."""
    from exp.offline_search.rounds.r17.longseg_1017 import pod_copy
    refs = {}
    # H100 LIBERO: DP reduced/full from the same sources as fa_ls_dp_{r,f}_b_l, served like fa_lo_dp_*_f_o_l.
    for mode in ('reduced', 'full'):
        n = f'fa_lo_dp_{mode[0]}_b_l'
        on = on_job(f'fa_ls_dp_{mode[0]}_b_l')
        job(n, on['source'], model='dp', benchmark='libero', mode=mode, workers=on['workers'], kind='pure', edit=lambda root, c, seal: dp_libero_h100(root, c))
        refs[f'dp:libero:{mode}:H100'] = n
    # H100 LIBERO Fast-WAM reduced: the fa_lo_fw_r_f_o_l resident, K1 YAMLs of its reduced source.
    src = Path(on_job('fa_ls_fw_r_f_o_l')['source'])
    templates = [(g, src/'assets'/g['file']) for g in read(src/'assets/CACHE_STUDY.json')['groups']]
    job('fa_lo_fw_r_b_l', H100_FW_LIBERO, model='fastwam', benchmark='libero', mode='reduced',
        workers=on_job('fa_ls_fw_r_f_o_l')['workers'], groups=[g for g, _ in templates],
        edit=pure_edit(templates, plan_name=read(src/'campaign.json')['cache_plan_name']))
    refs['fastwam:libero:reduced:H100'] = 'fa_lo_fw_r_b_l'
    refs['fastwam:libero:full:H100'] = 'fa_ls_fw_f_b_l'  # existing, C5 pure (commit 5, recipe disabled), H100
    # Pods: canonical H100 C5 pure definitions of the MT50 lane sources, then pod copies (pod_copy defaults,
    # as the coordinator's fa_lo_{fw,gr}_r_f_m_j: GR00T R2, Fast-WAM R1, 14 GiB, no MPS).
    for model, tok, (pod, port) in (('groot', 'gr', GROOT_POD), ('fastwam', 'fw', FW_POD)):
        for mode in ('reduced', 'full'):
            on = on_job(f'fa_ls_{tok}_{mode[0]}_f_m')
            n = f'fa_lo_{tok}_{mode[0]}_b_m'
            job(n, on.get('original_source') or on['source'], model=model, benchmark='mt50', mode=mode,
                workers=on['workers'], kind='pure',
                edit=(lambda root, c, seal: groot_r5(root, c)) if model == 'groot' else None)
            if not (EVAL/(n+'_j')).exists():
                pod_copy.copy_job(n, n+'_j', pod, port)
            refs[f'{model}:mt50:{mode}:{POD}'] = n+'_j'
    for mode in ('reduced', 'full'):
        refs[f'dp:mt50:{mode}:{POD}'] = f'fa_ls_dp_{mode[0]}_b_m_j'  # existing pod C5 pure (commit 5, recipe disabled)
    registry = dict(references={k: dict(run=str(EVAL/v), new=v.startswith('fa_lo_'), c5_pure=True) for k, v in refs.items()},
                    rule='long-segment-OFF selection: same model/benchmark/steps, same card (H100 or pod MIG slice), C5 pure')
    dump(REFERENCES, registry)
    # Also in the long-segment registry (new keys only; no existing key changes).
    path = LS/'REFERENCE_SELECTION.json'
    selection = read(path)
    for key, value in registry['references'].items():
        selection['references'].setdefault(key, dict(run=value['run'], new=value['new'], longseg_off_1022=True))
    dump(path, selection)
    return registry


def card(run):
    c = read(Path(run)/'campaign.json')
    return POD if c.get('pod_copy') else 'H100'


def reference(model, benchmark, mode, gpu):
    """Pure outcomes by suite from this study's registry (same reader as longseg_1017.analyze.reference)."""
    from exp.offline_search.rounds.r17.longseg_1017.analyze import lane_data
    root = Path(read(REFERENCES)['references'][f'{model}:{benchmark}:{mode}:{gpu}']['run'])
    if not (root/'RECEIPT.json').exists():
        return {}
    result = {}
    for (label, suite, size), arms in lane_data(root).items():
        result.setdefault(suite, {}).update(arms.get('pure', {}))
    return result


def build(new, src, items, *, model, benchmark, mode, stage, workers, selection_on, deploy=None, on=None):
    record = []
    with patch.object(lb, 'COMMANDS', COMMANDS):
        job = lb.clone(src, new, workers=workers, groups=[row for row, _, _ in items], mode=mode, stage=None,
                       edit=make_edit(model, benchmark, items, selection_on, deploy, record))
    job.update(stage=stage, arm='filtered', longseg='off', commit_horizon=5, on_campaign=on,
               cells=sorted({(r['suite'], r['library_size']) for r, _, _ in items}))
    dump(Path(job['run'])/'FFK_DEFINITION.json', job)
    if record:
        mapping = read(MAPPING) if MAPPING.exists() else {}
        mapping[new] = record
        dump(MAPPING, mapping)
    catalog = read(INDEX) if INDEX.exists() else dict(jobs=[])
    catalog['jobs'] = [j for j in catalog['jobs'] if j['name'] != new]+[job]
    dump(INDEX, catalog)
    return job


def mt50():
    jobs = []
    for model, tok in TOK.items():
        for mode in ('reduced', 'full'):
            on = f'fa_ls_{tok}_{mode[0]}_f_m'
            j, run = on_job(on), EVAL/on
            src = j.get('original_source') or j['source']  # DP: the H100 original, not the 4090 template
            items = [(g, run/'assets'/g['file'], yaml.safe_load((run/'assets'/g['file']).read_text())['recipe']['periodic_gap'])
                     for g in read(run/'assets/CACHE_STUDY.json')['groups']]
            selection_on = read(run/'campaign.json')['filter_failed_knob']['selection']
            jobs.append(build(name(on), src, items, model=model, benchmark='mt50', mode=mode, stage='descent',
                              workers=j['workers'], selection_on=selection_on,
                              deploy=groot_r5 if model == 'groot' else None, on=on))
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('mt50', 'pure', 'dp_mt50_h100'))
    a = p.parse_args()
    if a.action == 'mt50':
        mt50()
    elif a.action == 'dp_mt50_h100':
        dp_mt50_h100()
    else:
        pure_refs()


if __name__ == '__main__':
    main()
