"""Render round-2 evidence and apply unchanged gates, with owner pilot override."""
import csv
import json
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from exp.offline_search.rounds.r15.qk.astra.prepare import HERE,DERIVED,dump,sha

GATES_SHA='188d3cc1d3beb807174c3bff84696542be72f00bcb66ac0cdc66cbc913a2d937'


def read(path):return json.loads(path.read_text())


def main():
    assert sha(HERE/'GATES.md')==GATES_SHA
    root=HERE/'round2';identity=read(root/'pilot_identity_baseline.json')
    report={'round':2,'created_utc':datetime.now(timezone.utc).isoformat(),'gates_sha256':GATES_SHA,
            'numeric_thresholds_changed':False,'owner_authorizes_b_dev_pilot_despite_offline_failure':True,
            'authority':'OWNER_ADDENDUM.md fourth directive and coordinator round-2 interrupt',
            'astra_launches_online':False,'cells':{}}
    prose=['# PROTOTYPE2 — image-driven CPU query keys', '',
        'The fourth owner directive permits the best image-driven student to enter a B-dev pilot even if the original offline accuracy gates fail. GATES.md is byte-identical; this report keeps numerical results separate from pilot permission. Every candidate reported here uses its own symmetric light features, the same per-task action-supervised AWM and early fit, and the same 16-neighbor synthesis. State-only is an ineligible negative control. No online run or GPU use occurred in this round.', '',
        '## Library-only experiment and selection', '',
        'Two deterministic episode hold-outs per cell: one episode/task held out at size 50, four/task at size 200. Student weights, normalization, action sigma, positive pairs and fold AWM are fitted only on the remaining episodes. All recipes use the same splits and select among 40/80/120 epochs. The fixed full-cell teacher PCA coordinate system is reused as the distillation target basis; this is student episode hold-out, not a fully cross-fitted teacher representation. No B-dev image or label enters fitting or recipe/epoch selection.', '',
        'The new recipes combine small translations/scales and color/brightness changes, per-task target normalization, weight decay .01 and dropout .1. `vision_aug` removes the state shortcut inside the CNN; state still enters the downstream AWM. `token_aug` adds weight .3 spatial-token supervision. `token_keypoint` replaces flattened maps with spatial-softmax expected coordinates. `state_drop_aug` keeps an internal state branch with whole-branch dropout .5. `r1_style` is a fresh original-style global-normalization/state-conditioned control, with no augmentation/dropout and weight decay .001, added before new-candidate B-dev assessment. It is not byte-identical to the old checkpoint: it uses the v2 module/input concatenation and a fresh seed.', '',
        'Spatial targets use every selected row\'s full 256×2048 token map from both cameras, average pooled to 4×4, followed by a fixed signed orthonormal channel-group projection 2048→16. The resulting 512-dimensional target is auxiliary supervision, not a current-look input. The projection learns nothing from other rows. This is a bounded, low-cost spatial distillation experiment; it does not test every use of full-resolution token maps.', '',
        'Implementation: shared 3→16 (5×5/stride2), 16→24 and 24→32 (3×3/stride2) SiLU convolutions retain a 7×7 map/camera. The flat models concatenate both maps, an 8-D task embedding and the optional normalized state, then use a 256-unit SiLU head to predict 128 teacher-PCA coordinates. Keypoint models use softmax temperature factor 5 and expected x/y coordinates instead. State is standardized on fit episodes and clipped to ±8. AdamW uses learning rate .001, batch 96 and gradient clip 5. Seeds are 2026100506/2026100507 for folds 0/1 and 2026100506 for fresh final fits. Loss is standardized PCA MSE + teacher-AWM-code MSE + .1 action smooth-L1 + .5 cross-episode action-pair hinge, plus the optional .3 spatial loss. Positives are among the three nearest same-task action heads from other episodes; negatives must be other-episode and action-distance worse by .05; hinge margin .1. Augmentation uses ±1 pixel translation, ±2.5% scale, contrast .9–1.1, channel gain .95–1.05 and brightness ±.05; no flips. Full implementation and exact normalizers are retained in the scripts/checkpoints.', '',
        'Fresh final fits use the complete exact own-cell library after recipe/epoch selection. Their all-row donor-LOEO assessment precedes existing unseen B-dev. No history is needed by the selected interface; current images, state and task are all available online. Temporal/action/history candidates were not trained within this time box and remain untested options. The two-fold comparisons are development selection, not independent confirmation. The chosen recipe changes several regularizers together, so its gain cannot be attributed to state dropout alone.', '',
        '## Hold-out selection', '',
        '| Cell | Recipe | Best epoch | Mean two-fold RMS10 ratio | Fold ratios | Shuffle error rise | Zero-image error rise | Image audit |',
        '|---|---|---:|---:|---|---:|---:|---|']
    csvrows=[];selections={};assessments={};manifests={}
    for size in (50,200):
        cell=f'pi05_l10_{size}';selection=read(root/cell/'selection.json');selections[cell]=selection
        names=sorted(set(x['name'] for x in selection['all_candidates']))
        for name in names:
            best=min((x for x in selection['all_candidates'] if x['name']==name),key=lambda x:x['mean_fold_ratio'])
            prose.append(f"| {cell} | {name} | {best['epoch']} | {best['mean_fold_ratio']:.4f} | {best['fold_ratios'][0]:.4f}, {best['fold_ratios'][1]:.4f} | {100*best['shuffle_rise_mean']:.1f}% | {100*best['zero_rise_mean']:.1f}% | {best['blindness_pass_both_folds']} |")
            csvrows.append({'cell':cell,'name':name,'scope':'library_episode_holdout','epochs':best['epoch'],'balanced_ratio':best['mean_fold_ratio'],
                'shuffle_rise':best['shuffle_rise_mean'],'zero_rise':best['zero_rise_mean'],'blindness_pass':best['blindness_pass_both_folds']})
        chosen=selection['selected']['name'];a=read(root/cell/chosen/'assessment.json');assessments[cell]=a
        manifest=read(root/'pilot'/cell/'manifest.json');manifests[cell]=manifest
        assert manifest['name']==chosen and manifest['blindness_pass']
        l,b,blind=a['library'],a['b_dev'],a['blindness'];lat=read(root/'pilot'/cell/'latency.json')['timings'][0]
        lib_checks={'balanced_err10_ratio_le_1p05':l['balanced_ratio']<=1.05,
                    'every_task_ratio_le_1p15':max(v['ratio'] for v in l['balanced_by_task'].values())<=1.15}
        b_checks={'all_look_policy_labels':True,'balanced_err10_ratio_le_1p05':b['balanced_ratio']<=1.05,
                  'upper95_excess_le_0p10':b['balanced_relative_excess_ci95'][1]<=.10,
                  'every_task_ratio_le_1p15':max(v['ratio'] for v in b['balanced_by_task'].values())<=1.15,
                  'top16_overlap_ge_p75':b['top16_overlap']>=.75,
                  'large_backwards_delta_le_p02':b['phase_large_backwards']-b['reference_phase_large_backwards']<=.02,
                  'phase_wrong_third_le_p05':b['phase_wrong_third_vs_teacher']<=.05}
        cost_checks={'local_median_le_1p026ms':lat['median_ms']<=1.026,'local_p95_le_2p052ms':lat['p95_ms']<=2.052,
                     'timed_samples_ge_400':lat['n']>=400}
        report['cells'][cell]={'method':chosen,'epochs':selection['selected']['epoch'],'library':{'ratio':l['balanced_ratio'],'checks':lib_checks,'pass':all(lib_checks.values())},
            'b_dev':{'ratio':b['balanced_ratio'],'upper95_excess':b['balanced_relative_excess_ci95'][1],'checks':b_checks,'pass':all(b_checks.values())},
            'blindness':{'shuffle_relative_rise':blind['shuffle_mean_relative_rise'],'zero_relative_rise':blind['zero']['relative_rise'],
                        'tasks_worse_under_shuffle':blind['tasks_worse_under_shuffle'],'pass':blind['pass']},
            'local_cpu_cost':{'median_ms':lat['median_ms'],'q1_ms':lat['q1_ms'],'q3_ms':lat['q3_ms'],'p95_ms':lat['p95_ms'],
                'epsilon_graph':.152*lat['median_ms']/10.26,'epsilon_r12_eager_sensitivity':.152*lat['median_ms']/68.484,
                'timing_scope':'prepared two-camera raw RGB bundle through same main AWM code; excludes generic loader and common synthesis',
                'checks':cost_checks,'pass':all(cost_checks.values())},
            'deployment_host_cost_verified':False,'full_five_layer_plugin_verified_by_astra':False,
            'offline_pass':False,'pilot_candidate_eligible':blind['pass'],
            'pilot_artifact_manifest':str(root/'pilot'/cell/'manifest.json')}
    prose += ['', 'All epoch/fold results and image audits are retained in `round2/<cell>/<recipe>/cv{0,1}_train.json`; selection timestamps and input hashes are in each cell\'s `selection.json`. These candidates were not ranked using B-dev. The frozen original checkpoint is assessed separately as an incumbent reference.', '',
        '## Final symmetric comparison on existing B-dev', '',
        '| Cell | Candidate | Library LOEO ratio | B-dev balanced ratio | RMS5 / RMS10, look-weighted | B-dev excess 95% interval | Top-16 overlap | Shuffle rise | Zero rise |',
        '|---|---|---:|---:|---|---|---:|---:|---:|']
    for cell,selected in assessments.items():
        for a in (read(root/cell/'r1_action/assessment.json'),selected):
            b=a['b_dev'];bl=a['blindness'];ci=b['balanced_relative_excess_ci95']
            prose.append(f"| {cell} | {a['name']} | {a['library']['balanced_ratio']:.4f} | {b['balanced_ratio']:.4f} | {b['err5']:.4f} / {b['err10']:.4f} | [{100*ci[0]:+.1f}%, {100*ci[1]:+.1f}%] | {b['top16_overlap']:.3f} | {100*bl['shuffle_mean_relative_rise']:.1f}% | {100*bl['zero']['relative_rise']:.1f}% |")
            csvrows.append({'cell':cell,'name':a['name'],'scope':'unseen_bdev','epochs':selections[cell]['selected']['epoch'] if a is selected else 80,
                'balanced_ratio':b['balanced_ratio'],'shuffle_rise':bl['shuffle_mean_relative_rise'],'zero_rise':bl['zero']['relative_rise'],'blindness_pass':bl['pass']})
    prose += ['', 'Size 50 uses 100 unseen episodes / 3,198 looks; size 200 uses the 73 episodes / 2,289 looks outside its own subset. Both are existing size-50 controller trajectories, not new size-200 rollouts. The reference is controlled layer-1 deployed-key geometry, without insurance/guard/corrector effects. Equal-task/episode RMS10 is primary; intervals use 2,000 paired stratified episode draws, seed 20261005. Donor-LOEO fixes the full-library-trained representation and excludes the complete query episode only from donors.', '',
        'The original student is image-sensitive too: closeness of its clean error to state-only did not establish blindness. Shuffling may create an image/state inconsistency, so a large ablation effect is necessary evidence of image use, not proof of improved clean accuracy or SR. The state-only negative control has exactly zero change under either image operation because its input is unchanged; it is rejected regardless of its error.', '',
        '## Per-task final error and image-use audit', '',
        '| Cell | Task | Reference RMS10 | Student RMS10 | Excess | Shuffle rise | Zero rise |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for cell,a in assessments.items():
        for t,v in a['b_dev']['balanced_by_task'].items():
            z=a['blindness']['zero']['balanced_by_task'][t]['err10']/v['err10']-1
            prose.append(f"| {cell} | {t} | {v['reference_err10']:.4f} | {v['err10']:.4f} | {100*(v['ratio']-1):+.1f}% | {100*a['blindness']['shuffle_per_task_relative_rise'][t]:.1f}% | {100*z:.1f}% |")
    prose += ['', 'Blindness uses paired camera images shuffled within task with three fixed seeds, and both cameras set to raw black, while state/task/step and the library index remain fixed. The new owner requirement was operationalized before results as ≥5% balanced error rise for both mean shuffle and zero, plus ≥8/10 tasks worsening under shuffle. The numerical thresholds in GATES.md were not changed.', '',
        '## Negative control and paired improvement', '',
        '| Cell | State-only B-dev / reference | State-only shuffle / zero rise | Selected / state-only excess (95% interval) | Selected / original student excess (95% interval) |',
        '|---|---:|---|---|---|']
    controls={};dp=DERIVED/'dev_pi05_l10_r12_devlooks_50L'
    with np.load(dp/'dev.npz') as d:uid_all=d['uid'];task_all=d['task_id']
    frozen=read(HERE/'FINAL_MANIFEST.json')['files']
    for size in (50,200):
        cell=f'pi05_l10_{size}';name=selections[cell]['selected']['name'];mask=np.load(dp/f'unseen_{size}.npy')
        uid=uid_all[mask];task=task_all[mask]
        state_file=HERE/cell/'eval_state_only_awm.npz';assert sha(state_file)==frozen[str(state_file)]['sha256']
        paths=[root/cell/name/'bdev_eval.npz',state_file,root/cell/'r1_action/bdev_eval.npz',HERE/cell/'eval_reference.npz']
        errors=[]
        for path in paths:
            with np.load(path) as a:errors.append(a['err10'])
        means=[];draws=[];rng=np.random.default_rng(20261005)
        for t in range(10):
            groups=[uid==u for u in np.unique(uid[task==t])]
            v=np.array([[e[g].mean() for e in errors] for g in groups]);means.append(v.mean(0))
            draws.append(v[rng.integers(len(v),size=(2000,len(v)))].mean(1))
        m=np.mean(means,axis=0);b=np.mean(draws,axis=0);comparisons={}
        for key,i in (('state_only',1),('incumbent',2)):
            comparisons[key]={'relative_excess':float(m[0]/m[i]-1),'ci95':np.quantile(b[:,0]/b[:,i]-1,[.025,.975]).tolist()}
        c={'state_only_ratio':float(m[1]/m[3]),'shuffle_rise':0.,'zero_rise':0.,'blindness_pass':False,
           'audit_basis':'exact invariance: state-only AWM has no image input; states/metric/library held fixed',
           'comparisons':comparisons,'state_only_source':str(state_file),'state_only_source_sha256':sha(state_file)}
        controls[cell]=c
        def fmt(x):return f"{100*x['relative_excess']:+.2f}% [{100*x['ci95'][0]:+.2f}%, {100*x['ci95'][1]:+.2f}%]"
        prose.append(f"| {cell} | {c['state_only_ratio']:.4f} | 0% / 0% — rejected | {fmt(comparisons['state_only'])} | {fmt(comparisons['incumbent'])} |")
        csvrows.append({'cell':cell,'name':'state_only_awm','scope':'ineligible_negative_control_bdev','epochs':'',
            'balanced_ratio':c['state_only_ratio'],'shuffle_rise':0.,'zero_rise':0.,'blindness_pass':False})
    dump(root/'control_comparison.json',controls)
    prose += ['', 'State-only uses the same per-task AWM/early metric and 16-NN synthesis on state alone; its frozen round-1 arrays are reused and hash-checked. Zero audit effect follows exactly from its image-independent function, so it is ineligible regardless of clean accuracy. Paired intervals use the same 2,000 equal-task episode resamples. Negative excess means the selected model improves that comparator; these reused development comparisons do not establish causal attribution to one regularizer.', '',
        '## Tail, neighbor and phase diagnostics', '',
        '| Cell | Scope | RMS5 | RMS10 p90 | Top-1 agreement | Top-16 overlap | Known library phase MAE |',
        '|---|---|---:|---:|---:|---:|---:|']
    diagnostics={}
    for cell,a in assessments.items():
        name=selections[cell]['selected']['name']
        with np.load(DERIVED/cell/'train.npz') as tr, np.load(root/cell/name/'loeo_eval.npz') as ev:
            phase_mae=float(np.mean(np.abs(tr['progress']-ev['progress'])))
        diagnostics[cell]={'library_known_phase_mae':phase_mae}
        for scope in ('library','b_dev'):
            v=a[scope];phase=f'{phase_mae:.4f}' if scope=='library' else 'unlabeled'
            prose.append(f"| {cell} | {scope} | {v['err5']:.4f} | {v['err10_p90']:.4f} | {v['top1_agreement']:.3f} | {v['top16_overlap']:.3f} | {phase} |")
    prose += ['', '| Cell | Student / reference large backward rate | Teacher phase discrepancy >1/3 |',
        '|---|---|---:|']
    for cell,a in assessments.items():
        b=a['b_dev'];prose.append(f"| {cell} | {b['phase_large_backwards']:.4f} / {b['reference_phase_large_backwards']:.4f} | {b['phase_wrong_third_vs_teacher']:.4f} |")
    dump(root/'diagnostics.json',diagnostics)
    prose += ['', 'Tail and phase summaries are look-weighted diagnostics. Known library phase uses the query row\'s stored progress and the top donor\'s progress; B-dev has only the stated teacher/backward proxies. Complete early/stale and per-task summaries remain in each assessment JSON.', '',
        '## Unchanged gate application and local cost', '',
        '| Cell | Selected recipe / epochs | Library gate | B-dev gate | Image audit | CPU median / p95 ms | Local cost gate |',
        '|---|---|---|---|---|---|---|']
    for cell,c in report['cells'].items():
        lat=c['local_cpu_cost'];prose.append(f"| {cell} | {c['method']} / {c['epochs']} | {c['library']['pass']} | {c['b_dev']['pass']} | {c['blindness']['pass']} | {lat['median_ms']:.3f} / {lat['p95_ms']:.3f} | {lat['pass']} |")
    prose += ['', '| Cell | One-thread CPU IQR ms | Epsilon: graph basis | Epsilon: historical R12 eager sensitivity |',
        '|---|---|---:|---:|']
    for cell,c in report['cells'].items():
        lat=c['local_cpu_cost'];prose.append(f"| {cell} | {lat['q1_ms']:.3f}–{lat['q3_ms']:.3f} | {lat['epsilon_graph']:.6f} | {lat['epsilon_r12_eager_sensitivity']:.6f} |")
    prose += ['', 'The R12 eager sensitivity uses its previously measured 68.484-ms π0.5 tower and the same .152 owner coefficient (sources in COST.md). It is not H100 deployment timing and cannot convert a conservative graph-budget failure into a pass.']
    prose += ['', 'Local timing uses one CPU thread, 400 timed raw-image-to-main-AWM-code calls across all ten tasks after warmup; four-thread timing is also saved. Raw two-camera bundles are prepared outside the timer; resize, float conversion, state/task tensor conversion, model inference and AWM projection are timed. Generic loader checks, camera-bundle construction and common neighbor synthesis are not timed. These are encoder-path measurements, not complete serving latency. H100 deployment-host timing and the complete five-layer runtime verification belong to sol/coordinator and are not asserted here. Failure of an old offline numerical gate is reported explicitly but no longer vetoes this owner-authorized pilot.', '',
        '## Cost projection at the frozen size-50 epsilon', '',
        '| R14 B-dev operating point | Current IR | Projected IR | Relative saving |',
        '|---|---:|---:|---:|']
    counts=read(HERE/'cost_counts.json');epsilon=report['cells']['pi05_l10_50']['local_cpu_cost']['epsilon_graph'];projections=[]
    chosen_counts=[]
    for a in counts['arms']:
        if a['arm'] in ('r14d_pi05_l10_50_off','r14d_pi05_l10_50_periodic_ir25'):
            chosen_counts.append((a['arm'],a['totals']))
    stitch=counts['r14_selected_table_stitched']['pi05_l10_50']['tasks']
    chosen_counts.append(('fixed R14 task table (stitched)',{k:sum(t[k] for t in stitch.values()) for k in ('N','V','M')}))
    for name,c in chosen_counts:
        old=(.152*c['V']+.848*c['M'])/c['N'];new=(epsilon*c['V']+c['M'])/c['N'];saving=1-new/old
        projections.append({'name':name,'counts':c,'epsilon':epsilon,'current_ir':old,'projected_ir':new,'relative_saving':saving})
        prose.append(f'| {name} | {old:.4f} | {new:.4f} | {100*saving:.1f}% |')
    dump(root/'cost_projection.json',{'source':str(HERE/'cost_counts.json'),'source_sha256':sha(HERE/'cost_counts.json'),'projections':projections})
    prose += ['', 'These use the already reconciled B-only counts in cost_counts.json and hold N/V/M fixed: IR_new=(epsilon*V+M)/N. They do not predict the extra policy calls or changed trajectory lengths caused by the new key. No size-200 count projection is borrowed from size 50; the matched pilot must measure its own cost. The task table is only a fixed-table projection, not an added pilot arm.', '',
        '## Verdict', '',
        'Continue with the bounded, paired B-dev pilot in PILOT.md using the frozen hold-out-selected image-driven artifacts. The exploratory search does not establish that the generalization gap is solved. Keep the original matched-SR and total-cost criteria for interpreting the pilot; do not treat image sensitivity, a library result or owner pilot permission as an accuracy pass. No further architecture changes should enter that pilot after artifact freeze.', '',
        'Sources: round2 training/selection/assessment JSON, `round2/pilot/*/{manifest,latency}.json`, `round2/pilot_identity_baseline.json`, `round2_metrics.csv`, `gate_report.json`. All scripts and outputs are in the authorized astra/derived roots.']
    (HERE/'PROTOTYPE2.md').write_text('\n'.join(prose)+'\n')
    with (HERE/'round2_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    report['offline_pass']=all(x['offline_pass'] for x in report['cells'].values())
    report['pilot_candidates_eligible']=all(x['pilot_candidate_eligible'] for x in report['cells'].values())
    dump(HERE/'gate_report.json',report);dump(root/'gate_report.json',report)
    (HERE/'OFFLINE_GATES.md').write_text('# Round-2 gate application\n\nThe numerical gates in GATES.md are unchanged (SHA256 '+GATES_SHA+'). Their old offline-veto permission language is superseded by owner directive 4. Pilot eligibility requires the image-use and own-library rules; it is separate from an offline accuracy/cost pass.\n\n'+ '\n'.join(prose[prose.index('| Cell | Selected recipe / epochs | Library gate | B-dev gate | Image audit | CPU median / p95 ms | Local cost gate |'):prose.index('## Verdict')-1])+'\n\nExact checks: `gate_report.json`. Pilot design and artifacts: `PILOT.md`.\n')
    print(json.dumps({k:{'candidate':v['method'],'bdev_ratio':v['b_dev']['ratio'],'blindness':v['blindness']['pass'],'cost':v['local_cpu_cost']['pass']} for k,v in report['cells'].items()}))


if __name__=='__main__':main()
