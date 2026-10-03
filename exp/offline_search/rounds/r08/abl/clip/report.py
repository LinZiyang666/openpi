"""Render CLIP.md and the handback from recorded verification evidence."""
import hashlib
import json
from pathlib import Path
import re

from .make_arms import HERE,RUN,build
from .prepare import DERIVED,write_json


def load(name): return json.loads((HERE/'results'/name).read_text())


def main():
    rows,prov=build()
    audit=load('source_audit.json');validation=load('validation.json');fits=load('fits.json')
    latency=load('latency.json');emit=load('emission.json');arms=load('arm_validation.json')
    proofs=[load(n) for n in ('local_encoder_proof.json','h100_pi05_encoder_proof.json','h100_groot_encoder_proof.json')]
    ready=[r for r in validation if r['status']=='ready']
    assert len(ready)==4 and all(r['PASS'] for r in ready) and all(r['PASS'] for r in proofs)
    assert arms['PASS'] and emit['PASS'] and latency['PASS']
    testline=re.search(r'7 passed[^\n]*',(HERE/'results/pytest.log').read_text()).group(0)
    raw=max(r['parity']['online_library_embedding_max_abs_diff'] for r in ready)
    proj=max(r['parity']['online_library_pca64_max_abs_diff'] for r in ready)
    summary={'status':'pi05_ready_groot_blocked_missing_exact_images','ready_arms':emit['ready_arms'],
             'blocked_arms':emit['blocked_arms'],'unit_tests':testline,'library_rows_encoded':sum(r['rows'] for r in audit if r['status']=='ready'),
             'stock_metric_refit_checks':sum(r['fit_checks']['stock_metric_refit_checks'] for r in ready),
             'stock_query_payloads_bit_equal':sum(r['parity']['stock_query_payloads_bit_equal'] for r in ready),
             'blind_tail_payloads_bit_equal':sum(r['parity']['blind_tail_payloads_bit_equal'] for r in ready),
             'online_library_embedding_max_abs_diff':raw,'online_library_pca64_max_abs_diff':proj,
             'reference_encoder_max_abs_diff':[r['embedding_max_abs_diff'] for r in proofs],
             'prior_other_arms_preserved':emit['prior_other_arms_preserved'],
             'prior_other_config_files_byte_equal':emit['prior_other_config_files_byte_equal'],
             'configs_and_parser_pass':arms['configs_and_parser_pass'],'A_replicate_references':arms['A_replicate_references'],
             'official_pairs_each':arms['official_pairs_each'],'h100_gpu_encoder_or_server_smoke':'not run',
             'closed_loop_experiments_started':0}
    write_json(HERE/'results/verification_summary.json',summary)
    write_json(RUN/'clip/verification_summary.json',summary)

    lines=['# CLIP retrieval-key ablation — r08_abl', '',
           'Four pi05 arms are ready with fresh fit artifacts. All eight definitions are emitted; the four GR00T arms are blocked because the original libraries and sampled build HDF5s contain no images. Do not schedule those four arms. The shared encoder itself works in both h100 serving venvs without installing packages.', '',
           '## Exact A specification', '',
           '`r06/p2_ablations/make_arms.py:sources()` supplies the eight kind-A rows. Only `name`, `method`, and the existing `--os-fit-artifact` value change. `arms_clip.json` is separate from the concurrent guard-ablation `arms_in.json`; all original kwargs, flag order, absent fields and cost-ledger choices are retained.', '',
           'A is `BlindAWM` with joint features, fit_data=same, full-rank codes, early step-0 metric, k=16, nn=3, lam=.1, state_scale=1, lam_c=.5, no insurance, serving=anchor_tail, budget=1 and gates=budget_only. kref=5 for current libraries and 8 for big libraries. LOOK synthesizes the full cached chunk; the next blind decision executes its second five-control block. This preserves A’s ten-control commit.', '',
           '| Arm | Library | Library rows / demos | kref | Fit |', '|---|---|---:|---:|---|']
    for row,check in zip(rows,audit):
        lines.append(f"| `{row['name']}` | `{Path(check['library']).name}` | {check['rows']} / {check['episodes']} | {row['kwargs']['kref']} | {check['status']} |")
    lines += ['', 'The nominal Spatial-50 pi05 source has 49 demos and 1,018 rows; it is retained exactly. Scale labels are the source A labels.', '',
              '## Encoder and preprocessing proof', '',
              'The model is OpenAI CLIP ViT-B/32 (768-wide, 12 blocks, 12 attention heads, 32-pixel patches, 512-D image output). The cached timm `vit_base_patch32_clip_224.openai` file is an open_clip-format safetensors checkpoint. Its SHA256 is `e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31`; every Encoder load verifies this exact hash and strictly loads every visual tensor.', '',
              'OpenAI weights require QuickGELU (`x*sigmoid(1.702*x)`). The reference explicitly uses `force_quick_gelu=True`; open_clip’s untrained ViT-B-32 ordinary GELU default is inappropriate for these weights. `encoder.py` implements the image tower in plain torch, avoiding open_clip in GR00T. All three runtimes use this same file. Float32 preprocessing and weights are explicit; ambient autocast is disabled during encoding, TF32 is disabled for its CUDA operations, and caller backend settings are restored afterward.', '',
              'CLIP preprocessing is PIL bicubic shortest-edge resize to 224, round-offset center crop to 224, RGB conversion, float32 /255 and OpenAI mean/std normalization, then L2-normalized image embedding. Camera order is agentview/base_0_rgb first, wrist/left_wrist_0_rgb second. Each camera has its own centered PCA-64 fitted using A’s `rsvd_k128_o32_p3_seed0_keep64` recipe; the original eight state values are appended.', '',
              'Pi05 library frames are the full-row-aligned `tok/img0.npy` and `tok/img1.npy` uint8 224×224 RGB arrays. Twenty-four sampled camera-frame checks match source `input_images/base_0_rgb` and `input_images/left_wrist_0_rgb` bit-for-bit. Forty unique recorded server camera frames (init 0 across ten tasks per suite) match raw wire images to A’s post-transform inputs with max abs difference 0. Thus the pi05 wire-to-library pipeline is already 224×224; the CLIP resize/crop is identity before normalization. Evidence: `results/source_audit.json`.', '',
              '| Comparison (same six images) | Preprocess max abs diff | Embedding max abs diff |', '|---|---:|---:|']
    for label,result in zip(('Local plain torch vs open_clip','h100 pi05 venv vs local open_clip','h100 GR00T venv vs local open_clip'),proofs):
        lines.append(f"| {label} | {result['preprocess_max_abs_diff']:.3g} | {result['embedding_max_abs_diff']:.8g} |")
    lines += ['', 'The six-image proof uses four actual library frames plus 256×256 and rectangular synthetic images. Local comparison is bit-equal; both h100 comparisons are below 1e-5. Remote comparisons used CPU with torch 2.7.1+cu126 (pi05) and 2.5.1+cu124 (GR00T). h100 CUDA encoder parity, actual policy-server loading and network serving are unverified.', '',
              'The h100 weights are at `/home/exouser/r08abl_clip_sc/open_clip_model.safetensors`, with an identical-hash cache link at `~/.cache/huggingface/hub/models--timm--vit_base_patch32_clip_224.openai/snapshots/r08abl_clip_sc/open_clip_model.safetensors`. The runner’s HOME=/home/exouser resolves this automatically; no arm env/kwargs changes are needed. Disk observation after copying: /data 92 GiB free, root 26 GiB free. The existing `/home/exouser/openpi` checkout and both venvs were not modified.', '',
              '## Fit and online behavior', '',
              '`ClipAWM` binds the unchanged `AWM.fit` code to private globals substituting only the PCA source. This follows the R6 TokenPCAAWM precedent and does not mutate the AWM module. Stock `fit_metric`, per-task z-scores, action-supervised cross-demo neighbours, full and early Mahalanobis metrics, confidence calibration, top-16 weights, synthesis and blind-tail behavior remain inherited. The cached state/action/blind arrays and nonvisual settings are bit-equal to each source A artifact; all 80 full/early task metric refits are bit-equal.', '',
              f'The online path is `PluginSession.image -> OnlineQueryView.img0/img1 -> shared Encoder -> fitted PCA GEMV -> inherited AWM metric/query`. Across 29 sampled library rows per ready cell (116 rows, 232 embeddings), max embedding difference is {raw:.8g} and max projected key difference is {proj:.8g}. All 116 full query payload comparisons to stock BlindAWM on identical CLIP features and all 116 blind-tail comparisons pass bit-for-bit. This exercises real plugin accessors in process; it is not a running websocket/policy-server smoke.', '',
              'The encoder is a process singleton; fitted pickles contain only numerical fitted state, never the GPU model or lock. Connection clones share immutable fit arrays and get independent episode history. LOOK extras insert `os_clip_encode_ms` first, so the decision-log scalar cap retains it. The field measures both cameras’ preprocessing, transfer, inference, GPU completion and D2H. Initial model loading is excluded. Blind decisions do not encode or add this field.', '',
              '## Discovery retrieval sanity', '',
              'Candidate leave-one-demo-out excludes every row of the query demonstration, uses <=5 evenly spaced rows per demo, and queries only recorded init indices 0–29. There is no success filter. Deployed PCA and supervised metrics remain frozen from the complete A library; this is descriptive retrieval sanity, not refitted cross-validation. Neighbour agreement is CLIP vs A top-1 and top-16 overlap; action error is sigma-normalized RMS against the library chunk, on the full executed ten controls.', '',
              '| Cell | Demos / queries | Top-1 agreement | Top-16 overlap | CLIP RMS10 | A RMS10 |', '|---|---:|---:|---:|---:|---:|']
    for r in ready:
        t=r['retrieval']
        if t['status']=='ready':
            lines.append(f"| `{r['arm']}` | {t['demos']} / {t['queries']} | {t['mean_top1_agree']:.4f} | {t['mean_top16_overlap']:.4f} | {t['mean_clip_sigma_rms10']:.6f} | {t['mean_A_sigma_rms10']:.6f} |")
    q=ready[0]['discovery_query_sanity']
    lines += ['', f"LIBERO-10 current has unknown init indices for all 50 demos, so its discovery-only LODO is unverified. Separate logged inference-query sanity uses {q['queries']} query rows, inits {q['inits']}, and unchanged candidates: top-1 agreement {q['mean_top1_agree']:.4f}, top-16 overlap {q['mean_top16_overlap']:.4f}, CLIP/A RMS10 {q['mean_clip_sigma_rms10']:.6f}/{q['mean_A_sigma_rms10']:.6f}. It is explicitly not LODO. Inits 30–49 are not queried for either retrieval analysis. Full row identities and five-control error results are in `results/validation.json`.", '',
              'The CLIP retrieval key gives higher offline action error in both 500-demo cells. These results do not establish a closed-loop success rate; the coordinator must run the authorized real evaluation separately.', '',
              '## Local GPU LOOK latency', '',
              'RTX 4090 shared with the coordinator; 5 warmups and 50 measured in-process LOOKs per ready cell. Median/p95 timings include two cameras; full method adds PCA, metric, synthesis and anchor recording, but excludes policy stage 1, network, simulator and initial model loading. Long tails are observations on a shared GPU, not isolated H100 latency estimates.', '',
              '| Cell | CLIP encode median / p95 ms | Full method LOOK median / p95 ms |', '|---|---:|---:|']
    for r in latency['cells']:
        c,f=r['two_camera_clip_encode'],r['full_method_look']
        lines.append(f"| `{r['arm']}` | {c['median_ms']:.3f} / {c['p95_ms']:.3f} | {f['median_ms']:.3f} / {f['p95_ms']:.3f} |")
    lines += ['', 'Owner IR remains pi05 `.152*look + .848*call`, GR00T `.148*look + .852*call`; measured CLIP time is separate. A’s original cost_ledger presence/absence is preserved.', '',
              'Before each local GPU job the guard required >=8,000 MiB free and set an allocator cap of 3 GiB. Encoding used batches of four frames (eight camera images) and one job at a time under `.gpu_job.lock`. Encoding peak reserved GPU memory was 425,721,856 bytes; latest latency peak was 404,750,336 bytes. Actual checks and progress are in `results/encode.log`, derived `encoded.json` markers and `results/latency.log`. GPU use was limited to library encoding and LOOK latency; no policy/simulator jobs were launched.', '',
              '## Emission, pairing and remaining limits', '',
              'The common emitter already merges old arms by name. Its read/merge/config/write is now protected by `.emit_arms.lock`, and arms.json is replaced atomically; CLIP `emit.py` additionally verifies every pre-existing other row and config hash. Eight CLIP rows were added while retaining 24 other rows and 48 config files byte-for-byte, producing 32 total rows. The common emitter change is covered by a concurrent-emission unit test. `arms_in.json` owned by the guard task is untouched.', '',
              'All eight rows parse/configure in standard mode, without debug/oracle/input capture. The official manifest contains exactly 500 pairs (tasks 0–9 × inits 0–49). All 24 A replicate references have exactly that accepted terminal pair set; comparisons are separately paired to A replicate 1/2/3 in each cell. See `results/arm_validation.json` and run `clip/pairing.json`.', '',
              'Run `clip/ready_arms.json` lists only the four runnable pi05 arms; `clip/blocked_arms.json` lists the GR00T definitions. GR00T image arrays are absent and the sampled source step groups contain only clean_action, prompt_emb, robot_state, vision_0 and vision_1. No RGB reconstruction or substitute rollout was used. Recover the exact original per-frame camera inputs before building GR00T keys; its 256-wire/0.95-crop/224-resize model pipeline also requires validation against those recovered inputs. There is no GR00T fit or retrieval result.', '',
              'Refresh the h100 runner’s isolated source snapshot before selecting CLIP, so it includes the new module files. Use its explicit four-arm selection and the full official manifest. The snapshot runner can reuse the verified HOME cache weights; derived embeddings/PCA arrays are not required for serving a prefit artifact. No real closed-loop arm, remote server, LIBERO worker or running foreign process was started or stopped by this task. The temporary rsync daemon owned by this task on 23197 was stopped; receivers on 23198/23199 were left alone.', '']
    (HERE/'CLIP.md').write_text('\n'.join(lines))

    cpu='taskset -c 18-25,62-69 env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python'
    gpu='taskset -c 18-25,62-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python'
    module='exp.offline_search.rounds.r08.abl.clip'
    doc=['# HANDBACK S-C', '',
         '**Four pi05 arms are ready; four GR00T definitions are emitted but blocked by missing exact source images.** No packages were installed, and no real closed-loop experiments were run. The encoder is verified in both h100 serving venvs; GR00T availability is not the blocker.', '',
         'Design, preprocessing, retrieval tables and latency: [CLIP.md](CLIP.md). Complete evidence: `results/`; machine-readable status: `results/verification_summary.json` and run `clip/verification_summary.json`.', '',
         '## Files and artifacts', '',
         '- New code in this directory: encoder.py, method.py, make_arms.py, prepare.py, proof.py, remote_proof.sh, validate.py, latency.py, emit.py, report.py and __init__.py.',
         '- Separate input spec: arms_clip.json; provenance and exact A replicate references: results/provenance.json.',
         '- New tests: `tests/exp/offline_search/rounds/r08/abl/clip/test_clip.py`.',
         '- Existing file changed: `exp/offline_search/closed_loop/ops/emit_arms.py` (merge lock plus atomic arms.json replacement).',
         f'- Derived arrays only under `{DERIVED}`; original stores are read-only. Embeddings are float32 [rows,512] per camera; mean/basis/proj are stored under each camera directory.',
         f'- Fits under `{RUN}/fits/`; four artifacts listed below. Serving pickles include numerical PCA/metric/state/action data and exclude the image encoder.',
         '- Run config/<arm>.yaml and config/matrix_<arm>.yaml emitted for all eight definitions. Run clip/ready_arms.json, blocked_arms.json and pairing.json isolate readiness and references.',
         '- h100 task-owned proof directory: `/home/exouser/r08abl_clip_sc/`, plus the task-owned CLIP cache link described in CLIP.md. Existing checkout and venvs untouched.', '',
         '| Ready fit | Bytes | SHA256 |', '|---|---:|---|']
    for f in fits:
        if f['status']=='ready': doc.append(f"| `{Path(f['artifact']).name}` | {f['bytes']} | `{f['sha256']}` |")
    doc += ['', '## Reproduce preparation and verification', '', 'From `/home/weiland/projects/openpi`, these are the executed command forms. CUDA is hidden for CPU jobs; only --encode and latency use the local GPU. The source audit blocks GR00T without fabricating keys.', '',
            '```bash', f'{cpu} -m {module}.make_arms', '# Output: wrote 8 exact A-derived CLIP rows (name/method/fit only)',
            f'{cpu} -m {module}.prepare --audit', '# Output: pi05 rows 2640,29472,1018,10909 ready; GR00T rows 2645,29631,1063,11751 blocked (no exact images).',
            f'{gpu} -m {module}.prepare --encode', '# Output: gpu_guard free_MiB=12641 cap=3221225472; 4 pi05 encoded events; 4 skip_blocked events.',
            f'{cpu} -m {module}.prepare --fit', '# Output: 4 ready artifacts; remaining 4 blocked. Fit durations: 1.043,27.445,0.427,4.229 seconds.',
            f'{cpu} -m pytest tests/exp/offline_search/rounds/r08/abl/clip/ -q', '# Output: '+testline,
            f'{cpu} -m {module}.proof --reference --inputs {HERE}/results/proof_inputs.npz --output {HERE}/results/local_encoder_proof.json', '# Output: PASS=true; preprocess_max_abs_diff=0; embedding_max_abs_diff=0; bit_equal=true; images=6.',
            'tether exec --timeout 2m h100 -- bash /home/exouser/r08abl_clip_sc/remote_proof.sh', '# Output: both PASS=true; preprocess_max_abs_diff=0; embedding_max_abs_diff=2.682209014892578e-07; images=6.',
            f'{cpu} -m {module}.validate', f'# Output: 4 PASS=true ready cells; 80 metric refits; 116 identical query payloads; 116 identical tails; embedding diff <= {raw:.8g}; PCA diff <= {proj:.8g}.',
            '# Output: candidate LODO: 1500 L10-500,130 Spatial-50,1500 Spatial-500 queries; only init 0-29.',
            '# Output: L10-50 LODO blocked (unknown init metadata); 90 logged discovery queries on inits 0,10,20 separately reported.',
            f'{gpu} -m {module}.latency', '# Output: PASS=true; free_MiB=12635; cap=3221225472; peak reserved=404750336; 50 repetitions/cell. Timing table in CLIP.md.',
            f'{cpu} -m {module}.emit', '# Output: PASS=true; 8 CLIP arms; 24 prior other arms retained; 48 other config files byte-equal; total=32; ready=4; blocked=4.',
            f'{cpu} -m {module}.validate --arms', '# Output: PASS=true; source_rows_exact=8; configs_and_parser_pass=8; official_pairs_each=500; A_replicate_references=24; ready_fit_metadata_pass=4.',
            f'{cpu} -m {module}.report', '# Output: verification_summary.json reproduced below.', '```', '',
            'Commands were redirected to the correspondingly named `results/*.log` files. Source-frame checks are in source_audit.json; fitted-state/query/LODO results are in validation.json; exact artifact metadata/hashes in fits.json; emission retains the full foreign rows/config hashes in emission.json; pairing/journal identity checks are in arm_validation.json.', '',
            '```json',json.dumps(summary,indent=1),'```', '',
            'Remote source/weights verification commands and observed output:', '', '```bash',
            'sha256sum exp/offline_search/rounds/r08/abl/clip/encoder.py',
            '# 905aba7d363d2a9fae13f6eb87346e7c6aa41e75f7473aec7515668525b1bfdb',
            'tether exec --timeout 1m h100 -- bash -lc \'export HOME=/home/exouser; sha256sum /home/exouser/r08abl_clip_sc/encoder.py; cat /home/exouser/r08abl_clip_sc/pi05_code.sha /home/exouser/r08abl_clip_sc/groot_code.sha\'',
            '# All three lines: 905aba7d363d2a9fae13f6eb87346e7c6aa41e75f7473aec7515668525b1bfdb',
            'tether exec --timeout 1m h100 -- bash -lc \'export HOME=/home/exouser; sha256sum /home/exouser/.cache/huggingface/hub/models--timm--vit_base_patch32_clip_224.openai/snapshots/r08abl_clip_sc/open_clip_model.safetensors; df -h /data /\'',
            '# Weight SHA256 e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31; /data 92G free; root 26G free.', '```', '',
            'Weights were copied by an h100 rsync-daemon pull from a task-owned temporary sender on 23197. The initial symlink source failed with rsync exit 23; a regular task-owned copy fixed it, and the retry succeeded. The exact sender PID 3547111 was checked against rsync_sc.conf and stopped. Evidence: remote_transfer.log, remote_transfer_retry.log, h100_cache.log and h100_code_identity.log. Tether push/pull overwrote only this task’s own files with --force. Remote scripts use flock and resume by encoder source hash, tolerating duplicated h100 exec.', '',
            '## Coordinator follow-through and limits', '',
            'Use only the four names in run `clip/ready_arms.json` for h100 runner sync/chain. Refresh its isolated code snapshot first so all new Python files are present. The weights resolve through HOME=/home/exouser; fits keep weilandserver paths for the runner to rewrite. Retain 500 official pairs from `manifests/eval500.json`, standard mode, and the runner’s <=32-worker/four-GPU cap. These instructions are for the coordinator; this task did not launch the chain.', '',
            'Pair each CLIP arm episode-by-episode against its three A references listed in run `clip/pairing.json`. A/B journals were read only. CLIP is one new 500-pair arm per cell, not three newly run replicates. The owner IR formula remains unchanged; report extras.os_clip_encode_ms separately.', '',
            'GR00T blocker: the original stored libraries lack tok/img0.npy and tok/img1.npy; the sampled source HDF5s have only tokens/state/actions/prompt. The exact original RGB frames are required. No GR00T keys, PCA, metric fit, retrieval analysis, CUDA-serving parity or closed-loop test is claimed. With recovered source images, validate the GR00T 256-wire/crop/resize pipeline before encoding. The same verified plain-torch encoder is already available in its serving venv.', '',
            'Pi05 LIBERO-10 current’s 50 demonstration init indices are unknown in episodes.json and source HDF5 attributes; discovery-only LODO cannot be certified. Its separate 90 logged discovery queries are descriptive and are not relabeled as LODO. The other three LODO results retain the deployed complete-library fit, excluding the query demo only from retrieval candidates.', '',
            'Online key parity exercises real plugin accessors in process, not a websocket/policy-server smoke. h100 GPU latency and a real server load are unverified. Local latency was measured while the coordinator shared the GPU; p95 tails reach ~190 ms in two cells. No SR or causal improvement claim follows from offline action error.', '',
            'The combined arms.json now contains 32 rows. Any guard-task validator that assumes global length=24 needs to select its own 24 arm names when rerun; this task left that agent’s source files unchanged. The merge lock serializes callers loading the updated common emitter. No git state-changing commands were used and no protected research-line files were edited.', '']
    (HERE/'HANDBACK_SC.md').write_text('\n'.join(doc))
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
