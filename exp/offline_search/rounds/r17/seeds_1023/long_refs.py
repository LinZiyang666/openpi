"""Same-card C10 pure-inference fallbacks for the new-seed R17 long-segment-on lines (build only; never prepare/launch).

Each new batch is its long-segment ON study's pure reference (longseg_1017 fa_ls_*_p_*: recipe disabled, commit_horizon
10): the same CACHE_STUDY rows and YAML bytes, EPISODE_PLAN task UIDs and episode count. Only where it is served changes:
  H100 (sean lane 2: the chain exports R17_H100_NODE/HOST/DRIVER_PORT/PUBLIC_PORT; the wrappers carry no env), cloned
  like longseg_off_1022's fa_lo_*_b_* pure references (longseg_1017.build.clone), the rows/YAMLs copied from the source:
    fa_sd_dp_{r,f}_l_p  <- fa_ls_dp_{r,f}_p_l  4090 -> H100 from the same source with dp_libero_h100 (fa_lo_dp_*_b_l)
    fa_sd_dp_{r,f}_m_p  <- fa_ls_dp_{r,f}_p_m  4090 -> H100 from the H100 original (dp_mt50_h100, fa_lo_dp_*_b_m)
    fa_sd_fw_r_l_p      <- fa_ls_fw_r_p_l      4090 -> the H100 Fast-WAM LIBERO resident (fa_lo_fw_r_b_l), R1 / 20 GiB /
                                               text table, so it fits beside sean's Fast-WAM LIBERO full R4 (4 x 14.2 GB)
  pods: fa_sd_gr_{r,f}_m_p_j <- fa_ls_gr_{r,f}_p_m  longseg_1017.pod_copy (GR00T R2 / 14 GiB, the source definition: all
        2,500 episodes again); reduced on jupyter-ziyang10-02 (driver 23192), full on jupyter-ziyang10 (driver 23193).
LONG_REFERENCES.json registers the C10 fallback of every R17 long-segment-on line (model:benchmark:mode:card); make_index
reads long_reference only from it.

    python -m exp.offline_search.rounds.r17.seeds_1023.long_refs build      # the 7 batches (skips existing ones)
    python -m exp.offline_search.rounds.r17.seeds_1023.long_refs register   # LONG_REFERENCES.json, cards checked
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from unittest.mock import patch

import yaml

from exp.offline_search.rounds.r17.filter_failed_knob.common import read, sha
from exp.offline_search.rounds.r17.longseg_1017 import build as lb, pod_copy
from exp.offline_search.rounds.r17.longseg_1017.io_budget import dump
from exp.offline_search.rounds.r17.longseg_off_1022 import build as lob
from .build import FW_TEXT_TABLE, HERE
from .inputs import EVAL

LONG_REFERENCES = HERE/'LONG_REFERENCES.json'
H100 = {  # new batch: (source pure-long run, clone source, deployment)
    'fa_sd_dp_r_l_p': ('fa_ls_dp_r_p_l', 'ffk1015_dp_libero_reduced_formal_A_original', 'dp_libero'),
    'fa_sd_dp_f_l_p': ('fa_ls_dp_f_p_l', 'ffk1015_dp_libero_full_formal_A_original', 'dp_libero'),
    'fa_sd_dp_r_m_p': ('fa_ls_dp_r_p_m', 'ffk1015_dp_mt50_reduced_descent', None),
    'fa_sd_dp_f_m_p': ('fa_ls_dp_f_p_m', 'ffk1015_dp_mt50_full_descent', None),
    'fa_sd_fw_r_l_p': ('fa_ls_fw_r_p_l', lob.H100_FW_LIBERO.name, 'fastwam_r1'),
}
PODS = {'fa_sd_gr_r_m_p_j': ('fa_ls_gr_r_p_m', 'jupyter-ziyang10-02', 23192),
        'fa_sd_gr_f_m_p_j': ('fa_ls_gr_f_p_m', 'jupyter-ziyang10', 23193)}
# Existing same-card C10 pure references, and the new ones above, by model:benchmark:mode:card.
REGISTRY = {
    'dp:libero:reduced:H100': 'fa_sd_dp_r_l_p', 'dp:libero:full:H100': 'fa_sd_dp_f_l_p',
    'dp:mt50:reduced:H100': 'fa_sd_dp_r_m_p', 'dp:mt50:full:H100': 'fa_sd_dp_f_m_p',
    'fastwam:libero:reduced:H100': 'fa_sd_fw_r_l_p', 'fastwam:libero:full:H100': 'fa_ls_fw_f_p_l',
    'fastwam:mt50:reduced:H100': 'fa_ls_fw_r_p_m', 'fastwam:mt50:full:H100': 'fa_ls_fw_f_p_m',
    f'groot:mt50:reduced:{lob.POD}': 'fa_sd_gr_r_m_p_j', f'groot:mt50:full:{lob.POD}': 'fa_sd_gr_f_m_p_j',
}
SERVING = ('server_node', 'execution_host', 'public_port', 'replicas', 'replica_startup_width', 'memory_budget_gib',
           'mps', 'server_threads', 'server_python', 'fastwam_text_table', 'dp_batching', 'fastwam_batching')


def dp_libero(root, c):
    wire = root/'assets/wire_observations.npz'
    if wire.exists():
        wire.unlink()  # dp_libero_h100 copies the probe frames; never write through a link into the clone source
    lob.dp_libero_h100(root, c)


def fastwam_r1(root, c):
    """The H100 Fast-WAM LIBERO resident at R1 / 20 GiB with the prompt-encoding table (no umT5-xxl, ~14 GB)."""
    c.update(replicas=1, replica_startup_width=1, memory_budget_gib=20.0, fastwam_text_table=FW_TEXT_TABLE, server_threads=4)
    if isinstance(c.get('cache_worker_orchestration'), dict) and 'replicas' in c['cache_worker_orchestration']:
        c['cache_worker_orchestration']['replicas'] = 1


def same_definition(source, deploy):
    """clone() edit: the source run's pure rows (passed as groups) with their YAML bytes; then the deployment."""
    def edit(root, c, seal):
        from exp.offline_search.rounds.r17.filter_failed.build import register_metadata
        src = read(source/'campaign.json')
        for row in seal['groups']:
            dst = root/'assets'/row['file']
            if dst.exists():
                dst.unlink()
            shutil.copyfile(source/'assets'/row['file'], dst)
            if sha(dst) != row['sha256']:
                raise ValueError(f'{source.name}: {row["file"]} bytes differ from its CACHE_STUDY row')
            register_metadata(Path(c['filter_failed_source']), yaml.safe_load(dst.read_text()))
        c['filter_failed_knob'] = dict(method='frozen_fit_candidate_rows_only', refit=False, selection={}, cells={},
                                       arm='unfiltered')
        c['longseg_1017_control_kind'] = src['longseg_1017_control_kind']
        c['cache_plan_name'] = src['cache_plan_name']
        if deploy:
            deploy(root, c)
        c['seeds_1023_long_reference'] = dict(definition_of=str(source), serving_before={k: src.get(k) for k in SERVING},
                                              serving_after={k: c.get(k) for k in SERVING},
                                              reason='same-card C10 pure fallback for the new-seed long-segment-on lines')
    return edit


def build_h100(name):
    source_name, clone_source, deploy = H100[name]
    source, on = EVAL/source_name, lob.on_job(source_name)
    groups = read(source/'assets/CACHE_STUDY.json')['groups']
    deploy = {'dp_libero': dp_libero, 'fastwam_r1': fastwam_r1}.get(deploy)
    with patch.object(lb, 'COMMANDS', lob.COMMANDS):  # filter_failed_knob/commands/{prepare,launch}_<name>.sh
        job = lb.clone(EVAL/clone_source, name, workers=on['workers'], groups=groups, mode=on['mode'], stage=None,
                       edit=same_definition(source, deploy))
    job.update(stage='long', arm='pure_reference', commit_horizon=10, long_reference_of=source_name)
    dump(Path(job['run'])/'FFK_DEFINITION.json', job)


def build(names):
    for name in names:
        if (EVAL/name).exists():
            print('EXISTS', name, flush=True)
        elif name in PODS:
            pod_copy.copy_job(PODS[name][0], name, *PODS[name][1:])
        else:
            build_h100(name)


def register():
    """LONG_REFERENCES.json; the card of every run (campaign server_node / pod_copy) is checked against its key."""
    from .make_index import card
    references = {}
    for key, name in REGISTRY.items():
        run = EVAL/name
        if card(run) != key.rsplit(':', 1)[1]:
            raise ValueError(f'{key}: {name} is on {card(run)}')
        groups = sorted({g['group'] for g in read(run/'assets/CACHE_STUDY.json')['groups'] if g['group'].endswith('_pure')})
        references[key] = dict(run=str(run), groups=groups, new=name.startswith('fa_sd_'),
                               definition_of=read(run/'FFK_DEFINITION.json').get('long_reference_of') or
                               read(run/'FFK_DEFINITION.json').get('pod_copy_of'))
    dump(LONG_REFERENCES, dict(references=references, rule='R17 long segment on: the C10 pure-inference fallback '
                               '(recipe disabled, commit_horizon 10), same model / benchmark / steps / card as the line'))
    print(f'Wrote {LONG_REFERENCES}: {len(references)} references', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=('build', 'register'))
    p.add_argument('--name', action='append', help='default: all seven')
    a = p.parse_args()
    if a.action == 'build':
        build(a.name or [*H100, *PODS])
    else:
        register()


if __name__ == '__main__':
    main()
