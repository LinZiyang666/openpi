"""Prepare audited B-val selections and private launch scripts; launch nothing."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
import yaml

from . import common as C

REPO=C.HERE.parents[4]
Q1=C.HERE.parents[1]/'r06/ideation_Q1/method_c'


def state_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x,dtype=np.float64).tobytes()).hexdigest()


def manifests(out):
    selections={};audit={}
    for suite,full in [('l10','libero_10'),('spatial','libero_spatial')]:
        split_path=REPO/f'exp/ablation_study/config/common/split_{full}.yaml'
        split=yaml.safe_load(split_path.read_text())
        prior=json.loads((Q1/f'calibration_manifest_pi05_{suite}_50.json').read_text())
        exclude={r['task']:r['init'] for r in prior['selected']}
        pool=REPO/f'exp/common/data/db_init/libero/{full}'
        official=pool.with_name(full+'_apool')
        if list(pool.glob('*.pruned_init')):raise ValueError('B pool shadowed by test states')
        record=yaml.safe_load((Q1/f'bval_pool_{suite}.yaml').read_text())
        if C.sha(split_path)!=record['split_sha256']:raise ValueError('frozen split changed')
        for stem,digest in record['per_task_digests'].items():
            if C.sha(pool/(stem+'.init'))!=digest:raise ValueError('B pool content changed')
        selected=[];task_audit=[]
        for key,value in sorted(split.items(),key=lambda kv:int(kv[0].split('_')[-1])):
            task=int(key.split('_')[-1]);stem=value['init_stem']
            indices=sorted(set(value['val'])-{exclude[task]})[:2]
            if len(indices)!=2:raise ValueError('fewer than two unused frozen B-val states')
            if set(indices)&set(value['protected_in_train']):raise ValueError('historical cache overlap')
            b=np.asarray(torch.load(pool/(stem+'.init'),weights_only=False))
            test=np.asarray(torch.load(official/(stem+'.init'),weights_only=False))
            bs={state_hash(x) for x in b};ts={state_hash(x) for x in test}
            if bs&ts:raise ValueError('B/test pools have identical state contents')
            for init in indices:selected.append(dict(task=task,init=init,state_sha256=state_hash(b[init])))
            task_audit.append(dict(task=task,selected=indices,excluded_r6=exclude[task],
                                   B_pool_count=len(b),test_pool_count=len(test),identical_states=0))
        record.update(role='NONTEST_BVAL',note='R7 PROFILE only; B state pool, not official test; legacy --apool-record flag.')
        target=out/f'bval_pool_{suite}.yaml';target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(yaml.safe_dump(record,sort_keys=False))
        for model in ['pi05','groot']:
            mf=dict(role='NONTEST_BVAL',purpose='R7_PROFILE',model=model,suite=suite,
                pool_sha256=record['rollup_sha256'],pool_record=f'<RUN>/manifests/bval_pool_{suite}.yaml',
                pool_directory=record['apool_dir'],split_sha256=record['split_sha256'],
                prior_calibration_manifest=str(Q1/f'calibration_manifest_pi05_{suite}_50.json'),
                prior_calibration_manifest_sha256=C.sha(Q1/f'calibration_manifest_pi05_{suite}_50.json'),
                selection_rule='lowest two frozen B-val indices/task excluding the R6 lowest-index calibration state; no outcomes consulted',
                selected=selected)
            C.write(out/f'{model}_{suite}_bval20.json',mf);selections[model,suite]=mf
        audit[suite]=dict(tasks=task_audit,selected_pairs=len(selected),record_sha256=C.sha(target),
            dense_library_note='These B states were library acquisition states in dense banks. Ordinary deployment bank retained; PROFILE is plumbing/allocation screening, not an independent library-generalization test.')
    C.write(out/'manifest_audit.json',audit)
    return selections


def launch_scripts(out):
    source=C.HERE.parents[1]/'r06/p3_profiling/chain_p3.sh'
    text=source.read_text()
    needle='bash $ISL/run_arm_v2.sh'
    if text.count(needle)!=1:raise ValueError('P3 chain launch site changed')
    text=text.replace(needle,'bash $ISL/run_arm_r7_bval.sh')
    text=text.replace('taskset -c 6-9,50-53 env','taskset -c 10-13,54-57 env')
    marker='RUN=${1:?run-root}; shift\n'
    guard='''# R7 PROFILE requires the existing fenced streaming client delivery.
if [ -f "$RUN/state/P3_STREAM_PORT" ]; then export P3_STREAM_PORT=$(cat "$RUN/state/P3_STREAM_PORT"); fi
: "${P3_STREAM_PORT:?coordinator must provision the run-specific P3 stream receiver}"
'''
    if text.count(marker)!=1:raise ValueError('chain startup changed')
    text=text.replace(marker,marker+guard)
    # Resolve ops helpers using their original location after copying the chain.
    out.mkdir(parents=True,exist_ok=True)
    (out/'chain_profile.sh').write_text(text)
    C.write(out/'launch_source.json',dict(source=str(source),source_sha256=C.sha(source),
        generated_sha256=C.sha(out/'chain_profile.sh'),executed=False,
        changes=['ordinary plugin arms; only remote launcher changed to B-val wrapper','helper CPU affinity','stream required before any launch']))
    wrapper='''#!/usr/bin/env bash
# Coordinator-only; ordinary plugin arms with audited B initial states.
set -euo pipefail
suite=${1:?suite}; arm=${2:?arm}
case "$arm" in r7_*_profile) ;; *) echo 'R7 PROFILE arm required' >&2; exit 2;; esac
case "$suite" in libero_10|l10) short=l10; full=libero_10;; libero_spatial|spatial|sp) short=spatial; full=libero_spatial;; *) exit 2;; esac
island=/scratch/zixuans8/openpi_trace/os_cl
record="$island/r7_profile_bval/bval_pool_${short}.yaml"
pool="/scratch/zixuans8/openpi_trace/exp/common/data/db_init/libero/$full"
test -f "$record" && test -d "$pool"
for f in "$pool"/*.pruned_init; do
  if [ -e "$f" ]; then echo "B pool shadowed by $f" >&2; exit 2; fi
done
: "${OSCL_MANIFEST:?exact R7 B-val manifest required}"
# Last flags override the stock launcher's A defaults, retaining digest checks.
exec bash "$island/run_arm_v2.sh" "$@" --apool-record "$record" --apool-dir "$pool"
'''
    (out/'run_arm_bval.sh').write_text(wrapper)


def arm_specs(out,call_specs=None):
    from exp.offline_search.rounds.r07.c1_follow.prefit import kwargs_for,VARIANTS
    from exp.offline_search.rounds.r07.c2_wrist.specs import make_specs
    profile=[];evaluation=[]
    for cell,source in C.sources().items():
        model,suite,_=cell.split('_')
        for variant,(cls,*_) in VARIANTS.items():
            name=f'r7_{cell}_{variant}'
            row=dict(name=name,model=model,suite=suite,mode='plugin',cost_ledger=True,
                method=f'exp.offline_search.rounds.r07.c1_follow.methods:{cls.__name__}',kwargs=kwargs_for(source,variant),
                client_overrides=dict(replan_steps=5,resize_size=224 if model=='pi05' else 256),
                plugin_args=['--os-root',str(C.STORE),'--os-no-shadow-native','--os-blind','--os-log-r4','--os-fit-artifact',f'<RUN>/fits/{name}.pkl'])
            if variant!='SF2':evaluation.append(row)
            profile.append(dict(row))
        # A reference on the same B starts is necessary for paired realized cost.
        name=f'r7_{cell}_A'
        profile.append(dict(name=name,model=model,suite=suite,mode='plugin',cost_ledger=True,
            method=source['method'],kwargs=source['kwargs'],
            client_overrides=dict(replan_steps=5,resize_size=224 if model=='pi05' else 256),
            plugin_args=['--os-root',str(C.STORE),'--os-no-shadow-native','--os-blind','--os-log-r4','--os-fit-artifact',f'<RUN>/fits/{name}.pkl']))
    wrist=make_specs()
    evaluation.extend(wrist)
    profile.extend(copy.deepcopy(r) for r in wrist if r['name'].startswith('r7_sw_'))
    # A literal spec from C3, not guessed method kwargs or calibration paths.
    pending=['CU(.30) ×4 sparse','CT(.30) ×4 sparse','SA(.30) ×4 sparse']
    if call_specs:
        calls=json.loads(call_specs.read_text())
        evaluation.extend(calls);profile.extend(copy.deepcopy(calls));pending=['SA(.30) ×4 sparse']
    for r in profile:
        old=r['name'];r['name']=old+'_profile'
        r['plugin_args']=[v.replace(f'/fits/{old}.pkl',f'/fits/{r["name"]}.pkl') for v in r['plugin_args']]
        if '--os-log-inputs' not in r['plugin_args']:r['plugin_args'].append('--os-log-inputs')
        if '--os-log-r4' not in r['plugin_args']:r['plugin_args'].append('--os-log-r4')
        r['manifest']=f'<RUN>/manifests/{r["model"]}_{r["suite"]}_bval20.json'
    C.write(out/'arms_profile.json',profile);C.write(out/'arms_eval500.json',evaluation)
    C.write(out/'pending_variants.json',dict(pending=pending,why='SA is explicitly composed by C1 after PROFILE analysis and needs its own cadence/camera budget calibration; never emit fabricated method/calibration kwargs. C3 rows imported from its literal delivered specs when available.'))
    C.write(out/'schedule_v2.json',[dict(arm=r['name'],client_env=dict(P3_ENV_SEED='1703',P3_SNAPSHOT_EVERY='1',P3_SNAPSHOT_P='0')) for r in profile])
    from .render_profile import recipes
    recipes(out)
    return profile,evaluation


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=C.HERE/'prepared')
    p.add_argument('--call-specs',type=Path);a=p.parse_args()
    if not any(r in a.out.resolve().parents for r in (C.HERE,C.SCRATCH)):raise ValueError('C4 output only')
    manifests(a.out/'manifests');launch_scripts(a.out)
    profile,evaluation=arm_specs(a.out,a.call_specs)
    print(json.dumps(dict(profile_arms=len(profile),eval_arms=len(evaluation),episodes_per_arm=20,executed=False)))


if __name__=='__main__':main()
