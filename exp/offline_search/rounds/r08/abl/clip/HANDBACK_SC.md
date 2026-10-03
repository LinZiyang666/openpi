# HANDBACK S-C

**Four pi05 arms are ready; four GR00T definitions are emitted but blocked by missing exact source images.** No packages were installed, and no real closed-loop experiments were run. The encoder is verified in both h100 serving venvs; GR00T availability is not the blocker.

Design, preprocessing, retrieval tables and latency: [CLIP.md](CLIP.md). Complete evidence: `results/`; machine-readable status: `results/verification_summary.json` and run `clip/verification_summary.json`.

## Files and artifacts

- New code in this directory: encoder.py, method.py, make_arms.py, prepare.py, proof.py, remote_proof.sh, validate.py, latency.py, emit.py, report.py and __init__.py.
- Separate input spec: arms_clip.json; provenance and exact A replicate references: results/provenance.json.
- New tests: `tests/exp/offline_search/rounds/r08/abl/clip/test_clip.py`.
- Existing file changed: `exp/offline_search/closed_loop/ops/emit_arms.py` (merge lock plus atomic arms.json replacement).
- Derived arrays only under `/home/weiland/trace_runs/offline_search_store/derived/clip_vitb32`; original stores are read-only. Embeddings are float32 [rows,512] per camera; mean/basis/proj are stored under each camera directory.
- Fits under `/home/weiland/trace_runs/os_closed_loop/r08_abl/fits/`; four artifacts listed below. Serving pickles include numerical PCA/metric/state/action data and exclude the image encoder.
- Run config/<arm>.yaml and config/matrix_<arm>.yaml emitted for all eight definitions. Run clip/ready_arms.json, blocked_arms.json and pairing.json isolate readiness and references.
- h100 task-owned proof directory: `/home/exouser/r08abl_clip_sc/`, plus the task-owned CLIP cache link described in CLIP.md. Existing checkout and venvs untouched.

| Ready fit | Bytes | SHA256 |
|---|---:|---|
| `r8abl_clip_p_l10_50.pkl` | 9318256 | `468b4a935e75263ef62e31e7ccd17095ab1f1350a3649f55ad193ea6caa190ee` |
| `r8abl_clip_p_l10_500.pkl` | 78491445 | `840ee0106f7817be2a5cd255adcff200de8070e36e8a8be9bc0c654fa1c22be9` |
| `r8abl_clip_p_sp_50.pkl` | 5136498 | `0a9702a7c172ac79d48bfeb4831bb272783f31d22ea7f2c642b227b1b0832be4` |
| `r8abl_clip_p_sp_500.pkl` | 30635927 | `81dfc03baf5fbf8926c3360fcdb48e87669ab23dca5eb2923de1c503c4187a47` |

## Reproduce preparation and verification

From `/home/weiland/projects/openpi`, these are the executed command forms. CUDA is hidden for CPU jobs; only --encode and latency use the local GPU. The source audit blocks GR00T without fabricating keys.

```bash
taskset -c 18-25,62-69 env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r08.abl.clip.make_arms
# Output: wrote 8 exact A-derived CLIP rows (name/method/fit only)
taskset -c 18-25,62-69 env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r08.abl.clip.prepare --audit
# Output: pi05 rows 2640,29472,1018,10909 ready; GR00T rows 2645,29631,1063,11751 blocked (no exact images).
taskset -c 18-25,62-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r08.abl.clip.prepare --encode
# Output: gpu_guard free_MiB=12641 cap=3221225472; 4 pi05 encoded events; 4 skip_blocked events.
taskset -c 18-25,62-69 env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r08.abl.clip.prepare --fit
# Output: 4 ready artifacts; remaining 4 blocked. Fit durations: 1.043,27.445,0.427,4.229 seconds.
taskset -c 18-25,62-69 env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m pytest tests/exp/offline_search/rounds/r08/abl/clip/ -q
# Output: 7 passed, 1 warning in 16.69s
taskset -c 18-25,62-69 env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r08.abl.clip.proof --reference --inputs /home/weiland/projects/openpi/exp/offline_search/rounds/r08/abl/clip/results/proof_inputs.npz --output /home/weiland/projects/openpi/exp/offline_search/rounds/r08/abl/clip/results/local_encoder_proof.json
# Output: PASS=true; preprocess_max_abs_diff=0; embedding_max_abs_diff=0; bit_equal=true; images=6.
tether exec --timeout 2m h100 -- bash /home/exouser/r08abl_clip_sc/remote_proof.sh
# Output: both PASS=true; preprocess_max_abs_diff=0; embedding_max_abs_diff=2.682209014892578e-07; images=6.
taskset -c 18-25,62-69 env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r08.abl.clip.validate
# Output: 4 PASS=true ready cells; 80 metric refits; 116 identical query payloads; 116 identical tails; embedding diff <= 2.2873282e-06; PCA diff <= 3.4347177e-06.
# Output: candidate LODO: 1500 L10-500,130 Spatial-50,1500 Spatial-500 queries; only init 0-29.
# Output: L10-50 LODO blocked (unknown init metadata); 90 logged discovery queries on inits 0,10,20 separately reported.
taskset -c 18-25,62-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r08.abl.clip.latency
# Output: PASS=true; free_MiB=12635; cap=3221225472; peak reserved=404750336; 50 repetitions/cell. Timing table in CLIP.md.
taskset -c 18-25,62-69 env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r08.abl.clip.emit
# Output: PASS=true; 8 CLIP arms; 24 prior other arms retained; 48 other config files byte-equal; total=32; ready=4; blocked=4.
taskset -c 18-25,62-69 env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r08.abl.clip.validate --arms
# Output: PASS=true; source_rows_exact=8; configs_and_parser_pass=8; official_pairs_each=500; A_replicate_references=24; ready_fit_metadata_pass=4.
taskset -c 18-25,62-69 env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r08.abl.clip.report
# Output: verification_summary.json reproduced below.
```

Commands were redirected to the correspondingly named `results/*.log` files. Source-frame checks are in source_audit.json; fitted-state/query/LODO results are in validation.json; exact artifact metadata/hashes in fits.json; emission retains the full foreign rows/config hashes in emission.json; pairing/journal identity checks are in arm_validation.json.

```json
{
 "status": "pi05_ready_groot_blocked_missing_exact_images",
 "ready_arms": [
  "r8abl_clip_p_l10_50",
  "r8abl_clip_p_l10_500",
  "r8abl_clip_p_sp_50",
  "r8abl_clip_p_sp_500"
 ],
 "blocked_arms": [
  "r8abl_clip_g_l10_50",
  "r8abl_clip_g_l10_500",
  "r8abl_clip_g_sp_50",
  "r8abl_clip_g_sp_500"
 ],
 "unit_tests": "7 passed, 1 warning in 16.69s",
 "library_rows_encoded": 44039,
 "stock_metric_refit_checks": 80,
 "stock_query_payloads_bit_equal": 116,
 "blind_tail_payloads_bit_equal": 116,
 "online_library_embedding_max_abs_diff": 2.2873282432556152e-06,
 "online_library_pca64_max_abs_diff": 3.4347176551818848e-06,
 "reference_encoder_max_abs_diff": [
  0.0,
  2.682209014892578e-07,
  2.682209014892578e-07
 ],
 "prior_other_arms_preserved": 24,
 "prior_other_config_files_byte_equal": 48,
 "configs_and_parser_pass": 8,
 "A_replicate_references": 24,
 "official_pairs_each": 500,
 "h100_gpu_encoder_or_server_smoke": "not run",
 "closed_loop_experiments_started": 0
}
```

Remote source/weights verification commands and observed output:

```bash
sha256sum exp/offline_search/rounds/r08/abl/clip/encoder.py
# 905aba7d363d2a9fae13f6eb87346e7c6aa41e75f7473aec7515668525b1bfdb
tether exec --timeout 1m h100 -- bash -lc 'export HOME=/home/exouser; sha256sum /home/exouser/r08abl_clip_sc/encoder.py; cat /home/exouser/r08abl_clip_sc/pi05_code.sha /home/exouser/r08abl_clip_sc/groot_code.sha'
# All three lines: 905aba7d363d2a9fae13f6eb87346e7c6aa41e75f7473aec7515668525b1bfdb
tether exec --timeout 1m h100 -- bash -lc 'export HOME=/home/exouser; sha256sum /home/exouser/.cache/huggingface/hub/models--timm--vit_base_patch32_clip_224.openai/snapshots/r08abl_clip_sc/open_clip_model.safetensors; df -h /data /'
# Weight SHA256 e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31; /data 92G free; root 26G free.
```

Weights were copied by an h100 rsync-daemon pull from a task-owned temporary sender on 23197. The initial symlink source failed with rsync exit 23; a regular task-owned copy fixed it, and the retry succeeded. The exact sender PID 3547111 was checked against rsync_sc.conf and stopped. Evidence: remote_transfer.log, remote_transfer_retry.log, h100_cache.log and h100_code_identity.log. Tether push/pull overwrote only this task’s own files with --force. Remote scripts use flock and resume by encoder source hash, tolerating duplicated h100 exec.

## Coordinator follow-through and limits

Use only the four names in run `clip/ready_arms.json` for h100 runner sync/chain. Refresh its isolated code snapshot first so all new Python files are present. The weights resolve through HOME=/home/exouser; fits keep weilandserver paths for the runner to rewrite. Retain 500 official pairs from `manifests/eval500.json`, standard mode, and the runner’s <=32-worker/four-GPU cap. These instructions are for the coordinator; this task did not launch the chain.

Pair each CLIP arm episode-by-episode against its three A references listed in run `clip/pairing.json`. A/B journals were read only. CLIP is one new 500-pair arm per cell, not three newly run replicates. The owner IR formula remains unchanged; report extras.os_clip_encode_ms separately.

GR00T blocker: the original stored libraries lack tok/img0.npy and tok/img1.npy; the sampled source HDF5s have only tokens/state/actions/prompt. The exact original RGB frames are required. No GR00T keys, PCA, metric fit, retrieval analysis, CUDA-serving parity or closed-loop test is claimed. With recovered source images, validate the GR00T 256-wire/crop/resize pipeline before encoding. The same verified plain-torch encoder is already available in its serving venv.

Pi05 LIBERO-10 current’s 50 demonstration init indices are unknown in episodes.json and source HDF5 attributes; discovery-only LODO cannot be certified. Its separate 90 logged discovery queries are descriptive and are not relabeled as LODO. The other three LODO results retain the deployed complete-library fit, excluding the query demo only from retrieval candidates.

Online key parity exercises real plugin accessors in process, not a websocket/policy-server smoke. h100 GPU latency and a real server load are unverified. Local latency was measured while the coordinator shared the GPU; p95 tails reach ~190 ms in two cells. No SR or causal improvement claim follows from offline action error.

The combined arms.json now contains 32 rows. Any guard-task validator that assumes global length=24 needs to select its own 24 arm names when rerun; this task left that agent’s source files unchanged. The merge lock serializes callers loading the updated common emitter. No git state-changing commands were used and no protected research-line files were edited.
