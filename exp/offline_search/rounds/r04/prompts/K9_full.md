<task>
You are R4 agent K9: make cache retrieval part of the model on the GPU and measure it. Build a GPU-resident
retrieval module that is numerically equivalent (to a stated tolerance) to the deployed CPU methods, capture it in a
CUDA graph together with π0.5's stage 1 where possible, and measure latency. Do not change any running code path.
Measured facts (K8, rounds/r04/k8_search_latency/REPORT.md — read it fully first):
- single-thread CPU method cost per vision decision: AWM 1.2 ms (50-library) / 1.4 ms (500), R3 guard-only MixedJudge
  1.6 / 2.1 ms, R4 BlindMixedJudge 1.8 / 2.1 ms; blind phase step 0.47 ms, anchor_tail 0.22 ms; plugin log emission
  0.3–0.4 ms; deployed native CP1 search 2.1 ms (spatial) / 7.4 ms (l10).
- the largest component is the PCA projection: two fixed 64×32768 float32 bases read on every decision (≈0.7–0.8 ms,
  31–49% of the query), memory-bandwidth bound and library-size independent. The pooled 32768-d camera keys are
  produced on the GPU by stage 1 and copied to the host (`.cpu()`, forcing a sync) only to be projected on the CPU.
- scoring/normalization/median/top-k grow with the per-task block (π0.5 l10 500-library ≈ 2.9k rows per task).
- one server process tops out at ≈400–600 queries/s under concurrency (GIL), so per-decision wall time grows linearly
  with connections.
- in the owner cost basis (π0.5 CUDA-graph stages s1/s2/s3 = 10.26/27.69/29.57 ms, full 67.5 ms), the method-only
  search adds ≈.02–.03 to IR (e.g. l10 500 guard-only .238 → .271).
Deliverables (rounds/r04/k9_gpu_retrieval/):
1. `gpu_awm.py`: a torch module holding the fitted AWM state as device buffers (per-camera PCA bases and means,
   per-task whitening, per-task candidate blocks padded to a fixed capacity with a validity mask — preallocate spare
   capacity so rows can later be appended in place with `copy_` without changing addresses — action chunks, library
   step/episode/progress arrays), and a `forward(pooled_key_v0, pooled_key_v1, rs, task_id, regime inputs)` that
   reproduces AWM's projection, whitened distance, regime handling (step 0 / after MISS continuity term / fresh),
   median normalization, top-k (k=16, deterministic tie order as the CPU code), kernel weights with the kref reference,
   synthesis of the full chunk, and the confidence/extras used by the guard-only MixedJudge (d1, disp5, dst, and the
   task-centred camera cosine for the stuck guard). Keep everything fixed-shape so it can be captured in a CUDA graph;
   task selection by indexing into a stacked padded tensor. Load it from the existing fit pickles (e.g.
   /home/weiland/trace_runs/os_closed_loop/r03_mx/fits/r3mx_p_l10_g500.pkl and r3mx_p_l10_g.pkl; r02_g50/r02_g500 CL2
   fits for pure AWM; spatial counterparts).
2. Parity vs the deployed CPU method on real logged query inputs (the harness query sets in the store): top-1 and top-16
   set/order agreement, max |Δaction| on the synthesized chunk, confidence/extras differences, at both library scales
   and both suites for π0.5 (GR00T optional). Report every mismatch class; explain float64-vs-float32 effects; offer a
   float64 or a tie-stabilized variant if agreement is not ≥99.9% on top-1.
3. Latency on the 4090: (a) the module alone, eager vs CUDA-graph captured, batch 1 (and batch 8 if cheap); (b) the
   end-to-end "stage 1 → key pooling → retrieval → synthesis" with only the final chunk + verdict inputs copied to the
   host, vs today's path (stage 1 → pooled key D2H → CPU method), measured with CUDA events and wall clock; (c) if
   π0.5 stage 1 can be CUDA-graph captured in this repository (look for the existing stage-1 compile/graph options,
   e.g. `--compile-stage1`, the latency_bench line, the K3 stage adapter), capture stage 1 + retrieval in one graph and
   report the increment over stage 1 alone. State the bytes crossing PCIe per decision in each design.
4. A short design note: what of the guard-only MixedJudge / R4 blind judge / K7 vision-confirmed guard (per-episode
   history: stuck counts, progress memos, quantile controller) can be expressed as GPU state tensors in the same graph,
   what needs one host sync (the HIT/MISS branch before stage 2/3; CUDA 12.4+ conditional graph nodes as an option),
   and a plan for integrating into serving (not implemented here).
</task>

<context>
Repository /home/weiland/projects/openpi (branch Ziyang). Round 4 of an action-cache exploration for VLA policies on
LIBERO (π0.5 and GR00T N1.5). Read: rounds/r04/k8_search_latency/REPORT.md, rounds/r02/g1_awm/awm.py (the AWM math),
rounds/r03/h3_judge/judge.py (guard-only MixedJudge), rounds/r04/k1_blind/*.py and rounds/r04/k7_guard/judge.py (blind
and vision-confirmed guards), rounds/r04/k3_cost/HANDBACK.md (stage adapter and stage timing on the 4090),
closed_loop/serve_pi05.py, closed_loop/plugin.py (how pooled keys reach the method), exp/offline_search/harness/README.md.
Closed-loop servers of the coordinator run on this GPU and machine right now (ports 23150–23169): never touch them.
Store: /dev/shm/offline_search_store or /home/weiland/trace_runs/offline_search_store.
Python: /home/weiland/projects/openpi/.venv/bin/python from the repo root; π0.5 checkpoint as used by
closed_loop/ops/start_server.sh / K3 (read, do not modify).
</context>

<hard_constraints>
- Your CPU range: 26-29,70-73. Prefix every python command with `taskset -c 26-29,70-73`, BLAS/OMP threads 1, never
  more processes than threads. CPUs 38-43 and 82-87 belong to another project: never use them.
- GPU: the 4090 is shared with the coordinator's closed-loop servers and another project's training. Before every GPU
  step check `nvidia-smi`; start only if ≥ 12 GB are free; use at most 8 GB yourself; stop immediately on any OOM or if
  free memory drops below 4 GB; release the GPU after each measurement. Report the concurrent GPU load during timing
  (other servers are active, so label timings as measured under load and also report the minimum observed).
- Write only under rounds/r04/k9_gpu_retrieval/ and your own /tmp/k9_* scratch. Do not modify src/, harness/,
  profile/, closed_loop/, or any other rounds directory. No servers on ports, no timan107, no LIBERO workers, no chains,
  no git, no `rm -rf`, never `pkill -f` (kill only your own PIDs). Do not read tests/review_tests/.
</hard_constraints>

<completeness_contract>
Cover π0.5 l10 and spatial at both library scales for the pure AWM and the guard-only MixedJudge feature set. Say
explicitly what was not done and why.
</completeness_contract>

<verification_loop>
Re-run parity and latency after your final code change; report run-to-run variation (≥2 fresh processes).
</verification_loop>

<grounding_rules>
Report only what you ran and observed; exact commands, numbers, file paths. Label hypotheses.
</grounding_rules>

<structured_output_contract>
Write rounds/r04/k9_gpu_retrieval/REPORT.md (tables first: parity, latency, bytes; then design note; then caveats) with
scripts and raw JSON beside it, then print a one-paragraph summary as your final message.
</structured_output_contract>
