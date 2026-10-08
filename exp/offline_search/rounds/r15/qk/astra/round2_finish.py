"""Verify frozen student handoff, render final pilot/handback, content-freeze round2."""
import ast
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo
import numpy as np
import torch
from exp.offline_search.rounds.r15.qk.astra.prepare import HERE,DERIVED,REPO,load_cell,sha,dump
from exp.offline_search.rounds.r15.qk.astra.round2_report import GATES_SHA
from exp.offline_search.rounds.r15.qk.impl.encoder import FrozenEncoder


def read(path):return json.loads(path.read_text())


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert sha(HERE/'GATES.md')==GATES_SHA
    root=HERE/'round2';checks={'created_utc':datetime.now(timezone.utc).isoformat(),'gates_sha256':GATES_SHA,
                            'cpu_only':True,'torch_version':str(torch.__version__),'numpy_version':np.__version__,
                            'cpu_affinity':sorted(os.sched_getaffinity(0)),'cells':{},'syntax':[]}
    old_manifest=read(HERE/'FINAL_MANIFEST.json');old_checks=read(HERE/'verification.json')
    checks['sol_encoder_sha256']=sha(HERE.parent/'impl/encoder.py')
    dp=DERIVED/'dev_pi05_l10_r12_devlooks_50L'
    checks['inherited_data_evidence']={}
    for source in (HERE/'verification.json',HERE/'reference_order_audit.json',dp/'dev.npz',dp/'dev_meta.json',dp/'unseen_50.npy',dp/'unseen_200.npy'):
        digest=sha(source);assert digest==old_manifest['files'][str(source)]['sha256']
        checks['inherited_data_evidence'][str(source)]=digest
    for f in sorted(HERE.glob('round2_*.py')):ast.parse(f.read_text());checks['syntax'].append(f.name)
    gate=read(HERE/'gate_report.json');identity=read(root/'pilot_identity_baseline.json')
    pilot=['# R15-B round-2 frozen pilot handoff', '',
        'Owner directive 4 authorizes this B-dev pilot even when the existing offline gates fail. GATES.md stays byte-identical; its numerical SR/accuracy/cost standards are unchanged. This document supersedes the old offline-veto language and 100-episode-first pilot in the round-1 SPEC/REQUESTS. Astra has launched nothing. Sol completes the same-library runtime refits and integration verification; the coordinator executes the authorized pilot.', '',
        '## Chosen artifacts', '',
        '| Cell | Recipe | Epochs | Library hold-out ratio | B-dev ratio | Shuffle / zero error rise | One-thread CPU key ms |',
        '|---|---|---:|---:|---:|---|---:|']
    paths=[];selected={}
    for size in (50,200):
        cell=f'pi05_l10_{size}';out=root/'pilot'/cell;manifest=read(out/'manifest.json');spec=read(out/'encoder_config.json')
        selection=read(root/cell/'selection.json');name=selection['selected']['name'];selected[cell]=(selection,manifest,spec)
        assert name==manifest['name'] and manifest['controlled_cpu_latency']
        assert selection['gates_sha256']==GATES_SHA
        assert spec['completed_epochs']==spec['declared_epochs']==selection['selected']['epoch']
        assert spec['kwargs']['output_index'] is None and spec['library_only']
        assert spec['library_features_reference']==str(out/'library_features.npz')
        assert spec['library_features_reference_sha256']==sha(out/'library_features.npz')
        for filename,digest in manifest['files'].items():assert sha(out/filename)==digest
        for file,digest in selection['input_sha256'].items():assert sha(Path(file))==digest
        train=read(root/cell/name/'final_train.json');assessment=read(root/cell/name/'assessment.json')
        assert train['cpu_only'] and not train['heldout_episodes']
        assert [x['epoch'] for x in train['trace']]==list(range(1,spec['completed_epochs']+1))
        assert assessment['blindness']['pass'] and not assessment['training_used_bdev']
        m,base,lib,rows,provenance=load_cell('l10',size);tr=np.load(DERIVED/cell/'train.npz')
        train_digest=sha(DERIVED/cell/'train.npz')
        assert train_digest==train['training_source_sha256']==old_checks['cells'][cell]['train_npz_sha256']
        assert np.array_equal(np.load(lib/'tok/rows.npy'),np.arange(len(np.load(lib/'episode.npy'))))
        assert np.array_equal(tr['rows'],rows)
        expected=set(sum(m.episode_subset.values(),[]));assert set(train['fit_episodes'])==expected
        assert sha(Path(provenance['subset']))==spec['training_provenance']['subset_sha256']
        with np.load(out/'library_features.npz') as a:
            assert np.array_equal(a['rows'],rows);features=a['features']
        assert features.shape==(len(rows),128) and np.isfinite(features).all()
        with np.load(root/cell/name/'loeo_eval.npz') as a:
            assert not np.any(tr['episode'][a['top']]==tr['episode'][:,None])
            assert np.isfinite(a['action']).all()
        with np.load(root/cell/name/'bdev_eval.npz') as a:assert np.isfinite(a['action']).all()
        encoder=FrozenEncoder(spec);deltas=[]
        ims=[np.load(lib/f'tok/img{c}.npy',mmap_mode='r') for c in range(2)]
        for task in range(10):
            ids=np.flatnonzero(tr['task_id']==task)
            for i in ids[[0,len(ids)//2,-1]]:
                look=SimpleNamespace(img0=np.array(ims[0][rows[i]]),img1=np.array(ims[1][rows[i]]),rs=tr['rs'][i],task_id=task)
                actual=encoder.encode(look).ravel();assert np.allclose(actual,features[i],rtol=1e-5,atol=1e-4)
                deltas.append(float(np.max(np.abs(actual-features[i]))))
        checks['cells'][cell]={'own_library_episodes':size,'rows':len(rows),'epochs':spec['completed_epochs'],
            'actual_sol_loader_samples':len(deltas),'actual_sol_loader_max_abs':max(deltas),'own_episode_donors':0,
            'manifest_hashes_pass':True,'selection_inputs_unchanged':True,'blindness_pass':True,
            'training_source_sha256':train_digest,'inherited_data_alignment_unchanged':True,'finite_synthesized_chunks':True}
        g=gate['cells'][cell];bl=g['blindness'];cv=selection['selected']['mean_fold_ratio']
        pilot.append(f"| {cell} | {name} | {spec['completed_epochs']} | {cv:.4f} | {g['b_dev']['ratio']:.4f} | {100*bl['shuffle_relative_rise']:.1f}% / {100*bl['zero_relative_rise']:.1f}% | {g['local_cpu_cost']['median_ms']:.3f} |")
        paths += ['', f'### {cell}', '', f'Artifact directory: `{out}`. Manifest SHA256: `{sha(out/"manifest.json")}`.', '',
                  '| File | SHA256 |','|---|---|']
        paths += [f'| `{file}` | `{digest}` |' for file,digest in manifest['files'].items()]
    checks['passed']=True;dump(root/'verification.json',checks)
    gate['data_and_encoder_audit']={'pass':True,'evidence':str(root/'verification.json'),
        'teacher_reference_replay':'inherited unchanged round-1 reference-order audit; L10-50 top16 sets 100%',
        'scope':'data identities, same-cell training, unseen B masks, finite chunks, donor exclusion, actual encoder loader; complete runtime plugin separate'}
    dump(HERE/'gate_report.json',gate);dump(root/'gate_report.json',gate)
    pilot += ['', 'Recipe and epoch were selected using two within-library episode hold-outs, before the selected model\'s B-dev evaluation. The original checkpoint is a separately audited incumbent. No test-A data, GPU, current-look teacher token, or B-dev training label was used. All candidates have the same symmetric AWM comparison form. See PROTOTYPE2.md and gate_report.json for failures as well as improvements.']+paths
    pilot += ['', '## Stable artifact format: r15_qk_student_v2', '',
        '`student.ts` is a frozen CPU TorchScript module with `forward(images, state, task_id) -> features`. Images are float32 `[N,2,3,56,56]` in [0,1], agentview then wrist, from raw 224² uint8 RGB via `image[::4,::4]`, CHW and `/255`. State is float32 `[N,8]`, the existing valid normalized `rs[:8]`; student-specific normalization is internal. Task is int64 `[N]`, 0..9. Output is a single float32 `[N,128]` Tensor. No history input or new state channel is required by these selected models.', '',
        '`library_features.npz` binds float32 `features[L,128]` to int64 parent `rows[L]` in exact selected-library order. Split the 128 features into two 64-value blocks only for existing accessors; they are jointly produced light features, not original teacher tokens or independent camera embeddings. Append the unchanged valid state and refit the same AWM/main/early/confidence and R10 corrector from this cell only. Do not apply the teacher PCA again. `metric.pkl` is the exploratory per-task main/early affine metric; sol\'s complete layer-1–3 refit is the runtime artifact.', '',
        '`encoder_config.json` is directly compatible with `exp.offline_search.rounds.r15.qk.impl.build --size SIZE --encoder-config PATH`. Factory is `impl.encoder:CNNStudent`; **`kwargs.output_index` is null**, because the module returns a Tensor. `declared_epochs` matches the selected recipe rather than assuming 80. The config binds model, training record, exact subset and cost coefficient; `library_features_reference` also enables sol\'s all-row feature-parity check after encoding. `student.pt` is the portable config/state_dict checkpoint. All-task perturbation checks and all-library export parity are in the manifest; `round2/verification.json` additionally checks sol\'s actual FrozenEncoder loader on 30 raw-image examples/cell.', '',
        'Local encoder configs use one Torch thread and the authorized CPU affinity 10–21,54–65. When staging H100, sol/coordinator must bind paths to the frozen copies, record that host\'s available CPU allocation, and recompute its serving-config hash if paths or operational settings change. Student weight hashes stay fixed. Deployment-host timing is still required; local epsilon is the measured one-thread raw-image-to-AWM-code cost normalized by the owner π0.5 10.26-ms graph tower and .152 coefficient. Initial export timings are retained separately; final timing controls both Torch and NumPy BLAS threads and uses 400 distinct own-library image pairs.', '',
        '## Eight arms, 300 paired B episodes per arm', '',
        f'Use `{identity["selection"]}`, SHA256 `{identity["selection_sha256"]}`, field `model_suites.pi05_l10.pairs`. It has 30 identities/task, 300 total, verified disjoint from both served subsets. Rollout replicate seed: {identity["rollout_replicate_seed"]}; knob coin seed: 0. This is development reuse, not independent validation.', '',
        '| Cell | Current off | New key off | Current ir25 | New key ir25 | Matched ir25 cap |',
        '|---|---:|---:|---:|---:|---:|']
    for cell,c in identity['cells'].items():pilot.append(f"| {cell} | 300 | 300 | 300 | 300 | {c['ir25_cap']:.4f} |")
    pilot += ['', 'Both sizes are included directly, per the coordinator\'s revised directive. Total design: 2,400 arm-episodes. Preserve actual fixed caps, only-no-progress logic, same-library LOEO correction, commit/replan/policy-tail semantics, and paired host/seeds. Keep the baseline π0.5 H100 stack, five-control replan and full policy call. Do not reprice ir25 to consume the saved budget, introduce a new task table, or add teacher refresh probes.', '',
        '## Baseline reuse from R15 descent', '',
        'L10-200 current off and ir25 match the R15 descent selection, seed, library, fixed caps and intended H100 host:', '',
        '- `/home/weiland/trace_runs/os_closed_loop/r15_descent_pi05_l10_200/runs/r15d_pi05_l10_200_off`',
        '- `/home/weiland/trace_runs/os_closed_loop/r15_descent_pi05_l10_200/runs/r15d_pi05_l10_200_periodic_ir25`', '',
        'At the metadata audit, neither had a local completion summary or run manifest. These are eligible planned baselines, not completed evidence. Reuse once all 300 accepted paired outcomes exist and host/runtime/fit/cap identity is verified; no duplicate collection is needed if those conditions hold. There is no R15 size-50 descent root, so collect both size-50 current-system baselines on the exact same list. A size-100/200 baseline cannot substitute for size 50. With the two size-200 baselines reused, the pilot needs 1,800 additional arm-episodes: 1,200 student and 600 current-system size-50. Source hashes and identity checks are in `round2/pilot_identity_baseline.json`; no baseline success outcomes were inspected for this selection.', '',
        '## Runtime and interpretation', '',
        'Sol must complete its generic tower-skipping integration and the same-library layers 1–3 refits using these encoder configs. Verify zero tower on a light HIT, one complete policy/tower on MISS, neither encoder on legal blind continuation, single commit, exact parent rows, reset/receipt semantics and default teacher-path parity. Separate Q=student calls, T=tower calls and M=full-policy calls; the specified path has T=M and IR=(epsilon*Q+M)/N. Logical look history remains separate from tower work.', '',
        'Report the original numerical offline gate results without relabeling failures. The owner override permits the pilot, not a claim of accuracy equivalence. For the 300-episode matched comparison retain the frozen cell SR delta ≥−2 pp, each task ≥−5 pp, conservative paired cell lower95 ≥−5 pp, and the original large-saving targets (off ≥25%, ir25 ≥15%). Report actual IR, task/episode paired gains/losses, errors, executed controls and cost per episode. The fixed-table saving remains a separately labeled projection/comparison, not an extra pilot arm. No new A work is authorized here.', '',
        'Recommendation: continue with this bounded pilot, then decide from matched SR and real cost. The image audits pass, and the selected size-50 regularization improves its unseen error, but the offline gap remains. Any further architecture/epoch/knob change must be separately versioned before its outcomes are read.']
    (HERE/'PILOT.md').write_text('\n'.join(pilot)+'\n')
    hand=['# Round-2 handback — pilot candidates frozen', '',
        'Read [PILOT.md](PILOT.md) for exact artifacts, loader configs, eight arms and baseline reuse. [PROTOTYPE2.md](PROTOTYPE2.md) gives all hold-out comparisons, image audits and the unchanged numerical gate report. Recommendation: **continue with the owner-authorized paired B-dev pilot**; no online run or GPU use was launched by astra.', '']
    for cell,g in gate['cells'].items():
        hand.append(f"- **{cell}: {g['method']}, {g['epochs']} epochs.** B-dev balanced error is {100*(g['b_dev']['ratio']-1):+.2f}% versus the deployed key. Shuffling images raises error {100*g['blindness']['shuffle_relative_rise']:.1f}%; black images raise it {100*g['blindness']['zero_relative_rise']:.1f}%. Controlled CPU key median {g['local_cpu_cost']['median_ms']:.3f} ms. Offline B-dev gate: {'PASS' if g['b_dev']['pass'] else 'FAIL'}; image audit: PASS.")
    hand += ['', 'Selection used two within-library episode hold-outs over five recipes and 40/80/120 epochs, before selected-candidate B-dev evaluation. Every comparison uses the same symmetric AWM/early metric and 16-NN kernel. State-only remains rejected. The original student also passes the image audit, so its proximity to a state-only clean score was not evidence that it ignored images.', '',
        'GATES.md is unchanged (SHA256 `'+GATES_SHA+'`). Its old offline-veto/100-episode-first authorization is superseded by owner directive 4 and the coordinator\'s 300-episode plan. Numerical failures remain visible in [gate_report.json](gate_report.json). Sol\'s full runtime verification and H100 timing remain separate from these verified CPU encoder artifacts.', '',
        'Use the R15 list outside size 200 for both cells, current/new × off/ir25 × 300 episodes. Size-200 current baselines can reuse the corresponding R15 descent arms after completion/equivalence checks; no R15 size-50 baseline exists. Fixed ir25 caps are 2.6125 (50) and 2.0 (200). No other rollout or A work is requested.', '',
        'Stable model format, Tensor-output loader setting (`output_index: null`), selected epoch counts, manifest hashes and host CPU-binding notes are in PILOT.md. [round2/verification.json](round2/verification.json) records the actual sol-loader checks and artifact/data boundaries. [ROUND2_MANIFEST.json](ROUND2_MANIFEST.json) freezes this handoff. Round 1 remains preserved in `round2/round1_snapshot/` and the original final manifest.', '',
        'Open issue: the remaining unseen-state error may still reduce closed-loop SR or cause extra policy calls. Image sensitivity and faster key extraction do not settle that question; the bounded matched pilot is the next discriminating test.']
    (HERE/'HANDBACK.md').write_text('\n'.join(hand)+'\n')
    freeze_files=[*root.rglob('*'),*HERE.glob('round2_*.py')]+[HERE/f for f in ('PROTOTYPE2.md','PILOT.md','HANDBACK.md','OFFLINE_GATES.md','gate_report.json','round2_metrics.csv','GATES.md')]
    freeze_files += list((DERIVED/'round2').rglob('*'))
    files={str(p):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(set(freeze_files)) if p.is_file()}
    manifest={'frozen_utc':datetime.now(timezone.utc).isoformat(),'frozen_chicago':datetime.now(ZoneInfo('America/Chicago')).isoformat(),
        'scope':'round2 selected CPU student artifacts and pilot design, before their closed-loop outcomes',
        'gates_sha256':GATES_SHA,'owner_addendum_sha256':sha(HERE.parent/'OWNER_ADDENDUM.md'),
        'round1_manifest_sha256':sha(HERE/'FINAL_MANIFEST.json'),'files':files}
    dump(HERE/'ROUND2_MANIFEST.json',manifest);digest=sha(HERE/'ROUND2_MANIFEST.json')
    (HERE/'ROUND2_MANIFEST.sha256').write_text(digest+'  ROUND2_MANIFEST.json\n')
    print(json.dumps({'verified':checks['cells'],'frozen_files':len(files),'manifest_sha256':digest,'frozen_utc':manifest['frozen_utc']}))


if __name__=='__main__':main()
