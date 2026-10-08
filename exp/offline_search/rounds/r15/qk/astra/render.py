"""Render evidence documents from the saved offline metrics and cost counts."""
import json
from pathlib import Path
from exp.offline_search.rounds.r15.qk.astra.prepare import HERE,REPO,dump
from exp.offline_search.rounds.r15.qk.astra.cost_counts import project

def f(x,n=4):return 'NA' if x is None else f'{x:.{n}f}'

def prototype():
    data=json.loads((HERE/'analysis.json').read_text());lines=[
    '# Offline light query-key prototypes', '',
    '**Result: do not promote the current models.** A CPU-only implementation is practical, and the action-aware CNN is close in library donor-LOEO, but the unseen L10 recordings still show an accuracy gap. No closed-loop experiment of the new method was run. The offline gates and coordinator approval are prerequisites for the planned pilot.', '',
    'Sources: rerunnable scripts in this directory; `analysis.json`, `metrics.csv`, `pi05_*/loeo_results.json`, per-model training/results files, and `verification.json`. No test-A root or query store was opened.', '',
    '## Order, scope and metric', '',
    'Following the owner addendum, the final assessment starts with every library row as a pseudoquery against other episodes of the same task, then checks existing B-dev. The initial interim B-dev results predated that delivered addendum; subsequent work added the full LOEO comparison and complete CPU training. The final representation is trained on the whole cell library; donor-LOEO does not hold the encoder training episode out.', '',
    'All comparisons use the same selected action payload, 16-neighbor Gaussian kernel with the cell artifact\'s deployed kref (5 at size 50; 8 at size 200), early metric at step 0 and stale geometry thereafter. RMS5 is the R2 err definition: RMS over the first 5 controls and 7 valid action dimensions, divided by the selected library sigma. RMS10 uses the 10 controls committed by this system. No action padding is scored. B-dev targets are shadow-policy actions, or explicitly labeled policy MISS actions for the secondary Spatial analysis.', '',
    'This isolates layer-1 key/synthesis geometry, before insurance, correction or guard decisions. On L10-50, the reference has all recorded top-16 sets identical and only one top-1 difference among 3,198 looks; Spatial-50 has identical top-1 and top-16. The one discrepancy is at step 0; replay through the runtime `_dist` reproduces the recorded top-1 (`reference_order_audit.json`). The prototype uses explicit-code Euclidean arithmetic, while runtime uses its quadratic-form arithmetic; no exact tie is claimed. At L10-200, the size-200 deployed fit is evaluated on the 73 truly unseen episodes of size-50 recorded trajectories; recorded size-50 neighbors are not misused as its reference.', '',
    '## Methods actually fitted', '',
    '- **Symmetric spatial RGB AWM:** each camera retains a 14×14 RGB mean grid; per-camera library PCA-64, state-8, and the deployed cross-episode action-pair AWM/early metric. Both sides use this same representation.',
    '- **Asymmetric ridge distillation:** task-specific ridge from both spatial image grids plus state to the cell teacher PCA-128; library keys remain exact teacher AWM codes. Alpha .01/.1/1 were all reported, chosen only on B-dev.',
    '- **Spatial CNN:** shared 3-layer camera trunk retaining 7×7 positions, state and task conditioning, PCA and metric distillation plus action auxiliary loss. Tested with and without a same-task, cross-episode action-pair hinge, in both symmetric and asymmetric retrieval. No pretrained weights. See SPEC for the exact 856,299-parameter recipe.',
    '- **Local residual kernel:** per-task state-to-teacher affine prediction plus a spatial-image RBF residual; four length/ridge combinations. This was the nonlinear alternative to dense CNN regression.',
    '- **Direct action query key:** ridge predicts normalized 10-control actions; exact library action chunks and state define keys. Only library synthesis is served, not the regressed action. Two regularizers × two state weights. This tests a more direct action objective and its phase ambiguity.',
    '- **State-only AWM:** negative control using the same action-pair fitting rule.', '',
    'The R8 CLIP global-vector failure (reported 3–30 pp losses and phase confusion in `logs/offline_search_exploration.log.md`, CLIP section) motivates retaining spatial positions. R12 REPORT documents why both cameras are retained, especially GR00T L10. Those prior reports are context, not newly opened A data.', '',
    '## Library LOEO and unseen B-dev, side by side', '',
    'LOEO and balanced B-dev columns use equal episode weight within task and equal task weight. The B-dev look RMS columns are separately shown because long/failing trajectories contribute more looks. Bootstrap intervals resample paired episodes within task (2,000 draws, seed 20261005) and are descriptive after development selection.', '']
    for cell,d in data.items():
        loeo={m['name']:m for m in json.loads((HERE/cell/'loeo_results.json').read_text())['methods']}
        lines += [f'### {cell}', '',f"{d['episodes']} unseen B-dev episodes, {d['looks']} looks, {d['labelled_looks']} policy-labeled looks.", '',
        '| Candidate | Library LOEO RMS10 balanced | B-dev RMS5 | B-dev RMS10 | B-dev RMS10 balanced | Balanced excess vs reference (95% interval) | Top-16 overlap |',
        '|---|---:|---:|---:|---:|---|---:|']
        for name,m in d['methods'].items():
            lo=loeo.get(name,{});ci=m['balanced_relative_excess_ci95']
            lines.append(f"| {name} | {f(lo.get('episode_task_balanced_err10'))} | {f(m['err5'])} | {f(m['err10'])} | {f(m['episode_task_balanced_err10'])} | {m['balanced_relative_excess']*100:+.1f}% [{ci[0]*100:+.1f}, {ci[1]*100:+.1f}] | {f(m['top16_overlap'],3)} |")
        if 'l10' in cell:
            m=d['methods']['symmetric_cnn_action_cpu'];r=d['methods']['deployed_key_geometry']
            lines += ['', '**Primary CPU-trained candidate, by task.** RMS10 is episode-balanced; no task is removed.', '', '| Task | Episodes | Reference RMS10 | Student RMS10 | Excess |', '|---|---:|---:|---:|---:|']
            for t,v in m['balanced_by_task'].items():lines.append(f"| {t} | {v['episodes']} | {f(v['reference_err10'])} | {f(v['err10'])} | {(v['err10']/v['reference_err10']-1)*100:+.1f}% |")
            lines += ['',f"Phase proxies on all looks: top-1 agreement {m['top1_agreement']:.3f}; backwards jumps greater than .10 donor progress {m['phase_large_backwards']:.3f} versus reference {r['phase_large_backwards']:.3f}; any backwards movement {m['phase_any_backwards']:.3f} versus {r['phase_any_backwards']:.3f}; mean donor-progress difference from teacher {m['phase_teacher_mae']:.3f}; >1/3 disagreement {m['phase_wrong_third_vs_teacher']:.3f}.", '']
        else:lines += ['', 'Spatial is secondary: its 189 policy labels are selected MISS states, not a random/all-look sample; 1,032 HIT looks lack policy targets. Phase and neighbor diagnostics use all looks, but the error table uses labeled MISS looks. This cannot clear the offline promotion gate.', '']
    lines += ['## CPU feasibility and prior GPU work', '', '| Cell | CPU completed epochs | CPU training seconds | Train rows / episodes |', '|---|---:|---:|---|']
    for size in (50,200):
        d=json.loads((HERE/f'pi05_l10_{size}/cnn_action_cpu_train.json').read_text());last=d['trace'][-1]
        lines.append(f"| π0.5 L10-{size} | {last['epoch']} | {last['seconds']:.1f} | {d['train_rows']} / {d['train_episodes']} |")
    lines += ['', 'These are full CPU fits from fresh initialization, not CPU inference of GPU-trained weights. The earlier four CNN GPU fits and two GPU latency jobs completed under the original authorization before the interrupt delivering the CPU-preference directive. Their logs record the fresh free-memory gate and a 3-GiB allocator cap, stricter than the allowed 4-GiB cap. Their CPU paths were feasible; they are not presented as CPU-infeasible exceptions. No GPU was used after the addendum was delivered. Future GPU paths now refuse to run without an explicit CPU-infeasibility reason.', '',
    '## Interpretation and limitations', '',
    'The action-aware symmetric student performs well enough on its training-library representation to make donor-LOEO look promising. The unseen B-dev excess remains substantially larger. This is consistent with representation overfitting/domain shift, but the experiments do not isolate which mechanism dominates. The symmetric/asymmetric comparison shows that code-space consistency helps somewhat; it does not solve the generalization gap.', '',
    'This is a bounded prototype search, not evidence that a light key is impossible. In particular, no 112-pixel model, temporal student, stronger augmentation or additional library-only episode-cross-fitting study was completed. Those would be new versioned offline experiments. External pretraining is neither used nor proven necessary. No SR estimate is obtained by converting RMS error. Full guard/corrector refits and the real tower-bypass plugin remain implementation work, so none of these files is a production artifact.']
    (HERE/'PROTOTYPE.md').write_text('\n'.join(lines)+'\n')

def cost():
    counts=json.loads((HERE/'cost_counts.json').read_text());prices=json.loads((REPO/'exp/offline_search/rounds/r12/prep/price_table.json').read_text());primary={}
    lines=['# Light-key cost and IR projections', '',
    'Measured locally, batch one and both cameras. **CPU is feasible and preferred.** Primary final measurements use the CPU-trained action-aware student. Prior GPU measurements are retained as historical comparisons; no further GPU benchmark is needed or authorized by convenience.', '',
    'Sources: `latency_cpu_{50,200}.json`, `latency_cuda_{50,200}.json` (historical), `latency_cpu_extra_{50,200}.json`, `cost_counts.json/.csv`, and cited existing policy benchmarks. Each variant has 20 warmups per task and 100 timed iterations for each of tasks 0/2/5/8 (400 samples). Input is two actual selected-library 224² uint8 camera images; timing includes nearest resize, conversion, encoder, task metric, and the usable code on CPU. Historical GPU timing additionally includes host/device and code-return transfers. Common neighbor search/synthesis is excluded from both key costs. These are shared-host measurements, not an exclusive-GPU study.', '',
    '## Per-look latency', '', '| Cell size | Candidate | Backend / CPU threads | Median ms | IQR ms | p95 ms |', '|---|---|---|---:|---|---:|']
    for size in (50,200):
        for dev in ('cpu','cuda'):
            d=json.loads((HERE/f'latency_{dev}_{size}.json').read_text())
            for r in d['results']:
                if r['cpu_threads']!=1:continue
                s=r['end_to_end'];lines.append(f"| {size} | {r['name']} | {dev} / 1 | {s['median_ms']:.3f} | {s['q1_ms']:.3f}–{s['q3_ms']:.3f} | {s['p95_ms']:.3f} |")
        extra=json.loads((HERE/f'latency_cpu_extra_{size}.json').read_text())
        for r in extra['results']:
            s=r['end_to_end'];lines.append(f"| {size} | {r['name']} | CPU {r['mode']} / {r['cpu_threads']} | {s['median_ms']:.3f} | {s['q1_ms']:.3f}–{s['q3_ms']:.3f} | {s['p95_ms']:.3f} |")
            if r['name']=='symmetric_cnn_action_cpu' and r['mode']=='torchscript' and r['cpu_threads']==1:primary[size]=r
    lines += ['', 'CPU export parity is checked against eager outputs on the timed inputs (`max_abs_export_parity` in the JSON). This timing trace is not a complete deployment export; all-task/dynamic-input parity remains part of the sol handoff. The historical CUDA CNN measurements use the same architecture with distillation weights; no new GPU run was made for the final CPU-trained variant. GR00T has no trained light model: corresponding cost conversions below are architecture proxies, not GR00T accuracy or deployability evidence.', '',
    '## Policy references and epsilon', '',
    '| Existing reference | Tower/stage-1 ms | Full-policy ms | Provenance / qualification |', '|---|---:|---:|---|',
    '| π0.5 CUDA-graph model timing | 10.26 | 67.48 | `exp/ablation_study/latency_bench/analysis/analysis.md`; owner basis, historical 4090 |',
    '| π0.5 production eager cost table | 65.783 | 463.394 | `closed_loop/ops/cost_table.json`; sum of recorded stages |',
    '| GR00T production eager cost table | 32.530 | 250.131 | Same cost table; full K8 policy |',
    f"| π0.5 R12 measured stock tower | {prices['pi05']['normalization_full_ms']:.3f} | {prices['pi05']['normalization_full_ms']+prices['pi05']['timings']['policy_remainder']['median_ms']:.3f} | R12 price table; full column is tower + remainder medians, not an end-to-end median |",
    f"| GR00T L10 R12 tower | {prices['groot_l10']['normalization_full_ms']:.3f} | {prices['groot_l10']['normalization_full_ms']+prices['groot_l10']['timings']['policy_remainder']['median_ms']:.3f} | R12 price table; sum of tower/remainder medians |",
    f"| GR00T Spatial R12 tower | {prices['groot_spatial']['normalization_full_ms']:.3f} | {prices['groot_spatial']['normalization_full_ms']+prices['groot_spatial']['timings']['policy_remainder']['median_ms']:.3f} | R12 price table; sum of tower/remainder medians |", '',
    'The fixed owner split is retained: π0.5 .152/.848, GR00T .148/.852. For a stated tower timing `t_v`, epsilon = `c_v * light_ms / t_v`. Do not refit the owner split to unrelated wall-time stage shares. The primary conservative π0.5 comparison uses the historical 10.26-ms tower; the same-production-stack R12 conversion is shown separately. CPU latency expressed in these units is a cost proxy, not a measurement of GPU utilization saved.', '',
    '| Size | Primary CPU TorchScript ms, one thread | ε π0.5 graph basis | ε π0.5 R12 eager basis | ε GR00T L10 R12 proxy | ε GR00T Spatial R12 proxy |', '|---|---:|---:|---:|---:|---:|']
    for size,r in primary.items():
        ms=r['end_to_end']['median_ms'];lines.append(f"| {size} | {ms:.4f} | {.152*ms/10.26:.5f} | {.152*ms/prices['pi05']['normalization_full_ms']:.5f} | {.148*ms/prices['groot_l10']['normalization_full_ms']:.5f} | {.148*ms/prices['groot_spatial']['normalization_full_ms']:.5f} |")
    lines += ['', 'The graph-basis median budget is 1.026 ms (epsilon≤.0152); an eager-only pass must not conceal a miss against that conservative sensitivity. Deployment-host H100 timing is not measured here. No policy was loaded for a new latency benchmark: existing authoritative measurements suffice and a full GPU model would exceed this task’s 4-GiB prototype limit.', '',
    '## Correct counterfactual ledger', '',
    '`IR_old = (c_v V + c_m M)/N` and `IR_new = (epsilon V + M)/N`, because a light-key MISS still executes the complete policy including the tower. Thus `saved = [c_v(V−M) − epsilon V]/N`. The benefit comes from cache-only looks. At all-policy looks a student is overhead; a separately implemented forced-policy bypass can avoid that overhead, but cannot save the policy tower.', '',
    'All following projections hold observed N/V/M fixed. Actual accuracy changes may alter M, V, episode length and success. `cost_counts.py` reconstructed accepted attempts and contiguous decisions and reconciled all 52 B arms with summary ledgers. No new-method outcome was read. Full arm/task counts are in `cost_counts.csv`; hashes of every source read are in `cost_counts.json`.', '',
    '## Uniform operating points at the target epsilon = c_v/10', '', '| Round | Cell | Knob | N | V | M | Current IR | Projected IR | Relative saving |', '|---|---|---|---:|---:|---:|---:|---:|---:|']
    measured=[]
    for a in counts['arms']:
        if a['kwargs']['method'] not in ('off','periodic'):continue
        p=a['at_gate_epsilon'];root=Path(a['root']).name;roundname='R14' if root.startswith('r14') else 'R13';cell=f"{a['model']}_{a['suite']}_50";level='off' if a['kwargs']['method']=='off' else f"ir{round(100*a['kwargs']['target'])}"
        lines.append(f"| {roundname} | {cell} | {level} | {p['N']} | {p['V']} | {p['M']} | {p['current_ir']:.4f} | {p['new_ir']:.4f} | {p['relative_saved']*100:.1f}% |")
        if a['model']=='pi05':measured.append({'arm':a['arm'],'epsilon':primary[50]['epsilon_pi05_graph'],**project(a['totals'],.152,primary[50]['epsilon_pi05_graph'])})
    lines += ['', '## R14 frozen selected task tables', '',
    'These are B-selected whole-task stitches of existing uniform arms, not independently executed mixed-policy results. They describe economic headroom at the existing table, not a new descent. The target-epsilon and zero-cost ceilings are shown together.', '',
    '| Cell | Current IR | IR at ε=c_v/10 | Saving | Maximum saving at ε=0 | Extra calls/100 looks before all savings disappear |', '|---|---:|---:|---:|---:|---:|']
    for cell,d in counts['r14_selected_table_stitched'].items():
        p=d['at_gate_epsilon'];z=d['zero_key_lower_bound'];lines.append(f"| {cell} | {p['current_ir']:.4f} | {p['new_ir']:.4f} | {p['relative_saved']*100:.1f}% | {z['relative_saved']*100:.1f}% | {p['max_extra_calls_per_100_looks']:.2f} |")
    lines += ['', '| Cell | Task | Fixed level | N | V | M | Current IR | Projected IR at cost gate |', '|---|---:|---|---:|---:|---:|---:|---:|']
    for cell,d in counts['r14_selected_table_stitched'].items():
        for t,p in d['tasks'].items():lines.append(f"| {cell} | {t} | {p['level']} | {p['N']} | {p['V']} | {p['M']} | {p['current_ir']:.4f} | {p['new_ir']:.4f} |")
    lines += ['', '## Projections using measured primary CPU epsilon', '', '| R14 π0.5 cell | Off: current → projected | ir25: current → projected | Selected table: current → projected |', '|---|---|---|---|']
    for suite in ('l10','spatial'):
        cell=f'pi05_{suite}_50';parts=[]
        for method,target in [('off',None),('periodic',.25)]:
            a=next(a for a in counts['arms'] if Path(a['root']).name==f'r14_descent_{cell}' and a['kwargs']['method']==method and a['kwargs'].get('target')==target)
            p=project(a['totals'],.152,primary[50]['epsilon_pi05_graph']);parts.append(f"{p['current_ir']:.4f} → {p['new_ir']:.4f} ({p['relative_saved']*100:.1f}%)")
        d=counts['r14_selected_table_stitched'][cell]['at_gate_epsilon'];p=project(d,.152,primary[50]['epsilon_pi05_graph']);parts.append(f"{p['current_ir']:.4f} → {p['new_ir']:.4f} ({p['relative_saved']*100:.1f}%)")
        lines.append('| '+cell+' | '+' | '.join(parts)+' |')
    lines += ['', 'Spatial’s last table is an architecture-cost projection only: its trained model and all-look accuracy gate are not established. The R13/R14 ir40–ir45 regimes have little removable cache-only tower work. GR00T L10’s chosen table cannot deliver a large total reduction at matched calls even with a free key. The economical initial target is π0.5 off/ir25, subject to the still-failing accuracy gate.']
    (HERE/'COST.md').write_text('\n'.join(lines)+'\n');dump(HERE/'cost_measured_projections.json',{'primary_cpu':primary,'pi05_arms_at_measured_l10_50_epsilon':measured,'scope':'fixed-count cost projections, not outcomes'})

if __name__=='__main__':prototype();cost()
