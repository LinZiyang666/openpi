<task>
You are R4 agent K8: measure the latency of the cache-search step (retrieval, synthesis, judge, blind step) at the
50- and 500-episode libraries, and relate it to the policy's stage costs. The owner asked for this explicitly.
Deliver rounds/r04/k8_search_latency/REPORT.md with:
1. Logged in-server evidence (read-only): summarize the per-decision fields `q_us` (method query: retrieval +
   synthesis + method-side judge features), `search_us` (q_us + native shadow search + verdict bookkeeping),
   `native_us` (the deployed cp1 storage-backend search run in shadow), `blind_prepare_ms`/`blind_output_ms` (blind
   decisions) across the closed-loop run roots under /home/weiland/trace_runs/os_closed_loop/ (r02_g50, r02_g500,
   r03_mx, r03_full, r04_*), per model × suite × library scale × method family, p50/p90/p99. Separate SERIALIZED
   measurements (R4-mode arms that ran before 15:47 CDT 2026-09-27, when the plugin still serialized all connections;
   e.g. r04_blind r4b3_p_l10_500_ph2g / b0g, r04_cost r4b2_p_l10_g500_k2, r04_k5_smoke, r04_bsmoke) from CONCURRENT
   ones (legacy arms: many connections share one Python process, so GIL contention inflates wall-clock). The
   coordinator's first cut: serialized π0.5 l10 MixedJudge q_us p50 ≈ 3.1 ms (50-library) vs 3.75 ms (500-library);
   concurrent ≈ 10.2 ms vs 19.4 ms; pure-cache AWM concurrent ≈ 4.5 vs 4.9 ms (l10), GR00T l10 ≈ 2.8 vs 3.0 ms;
   blind decisions ≈ 0.95 ms. Verify or correct these.
2. A controlled microbenchmark (the main deliverable): in one Python process pinned to your CPUs, replay real logged
   query inputs (the offline harness query sets in the store, /dev/shm/offline_search_store or
   /home/weiland/trace_runs/offline_search_store, and the methods' own fit pickles) through each deployed method's
   `query()` / `blind_step()` exactly as the plugin calls them, single-threaded, warm, with ≥ 1,000 timed queries per
   configuration, and report p50/p90/p99 and a component breakdown: key/feature construction from the stage-1 outputs
   (token pooling, PCA projection, whitening/state features), candidate scoring over the per-task library block,
   top-k selection, kernel synthesis, judge/V7 features and guards, blind phase-particle step, and the plugin's own
   per-decision overhead (history bookkeeping, logging serialization) if measurable. Configurations: π0.5 and GR00T ×
   spatial and libero_10 × 50- and 500-episode libraries × methods {AWM CL2 pure cache, R3 MixedJudge guard-only,
   R4 BlindMixedJudge (vision decision and blind decision), BlindAWM, K3 WristAWM / wrist judge,
   ControlStepLibrary G/GS} where the fits exist (fit pickles under /home/weiland/trace_runs/os_closed_loop/*/fits/;
   refit on your CPUs into your own scratch directory only when necessary, with the exact method/kwargs/cell).
   Include the deployed native cp1 search (the `native_us` path) for reference.
3. Scaling: how latency grows with library size at fixed task (entries per task for 50 vs 500 libraries, and a
   synthetic scale-up, e.g. tiling the 500-episode block to 1k/2k/5k episodes, to extrapolate), and under concurrency
   (1, 4, 8, 16, 24, 32 threads in one process issuing queries; use rounds/r04/k6_concurrency/concurrency_test.py as a
   model) to quantify GIL contention.
4. Put the numbers in the cost frame: retrieval per decision as a fraction of the owner-basis stage costs (π0.5
   CUDA-graph s1/s2/s3 = 10.26/27.69/29.57 ms, i.e. .152/.410/.438 of a full 67.5 ms inference) and of the eager
   basis in closed_loop/ops/cost_table.json; show what the current frontier points' IR become if the search cost is
   added per decision (owner-basis examples: π0.5 l10 500-library guard-only SR .864 @ IR .238 with vision share 1 and
   MISS share .101; blind B=2 SR .850 @ IR .211 with vision share .586, MISS share .144; spatial 500-library guard-only
   SR .974 @ IR .197). State which components dominate and which are cheap to remove (e.g. vectorized numpy, torch on
   GPU, precomputed per-task matrices), as hypotheses with the measured headroom, not as implementations.
</task>

<context>
Repository /home/weiland/projects/openpi (branch Ziyang). Round 4 of an action-cache exploration for VLA policies
(π0.5, GR00T N1.5) on LIBERO. A decision runs the policy's stage 1 (vision encoder), then retrieval over a per-task
library (keys: PCA-64 of each camera's tokens + robot state, per-task whitening), kernel synthesis of the top-16, and a
judge; a MISS also runs stages 2/3. Library scales: 50 and 500 demonstration episodes per suite (π0.5 l10: 2,640 and
29,472 entries). Read: exp/offline_search/harness/README.md, closed_loop/README.md, rounds/r04/CODING_BRIEF.md,
rounds/r04/k1_blind/HANDBACK.md, rounds/r04/k3_cost/HANDBACK.md (stage timing method and cost_table.json),
rounds/r04/k6_concurrency/HANDBACK.md, and the method code you benchmark (rounds/r02/g1_awm/awm.py,
rounds/r03/h3_judge/judge.py, rounds/r04/k1_blind/*.py, the K3 wrist code).
Closed-loop chains and servers are running on this machine on other CPUs; other CPUs are busy, so pin strictly and
report the load you observed during measurement (os.getloadavg and per-core idle on your cores).
</context>

<hard_constraints>
- Your CPU range: 34-37,78-81. Prefix every python/numpy command with `taskset -c 34-37,78-81` and set
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; never more processes than threads in your range; for
  the single-threaded timings use one core only and say which. CPUs 38-43 and 82-87 belong to another project: never
  use them.
- GPU: none (CUDA_VISIBLE_DEVICES=''). Any GPU-retrieval idea stays a hypothesis.
- Write only under rounds/r04/k8_search_latency/ and your own scratch directory under /tmp/k8_*. Do not modify any
  existing file (src/, harness/, profile/, closed_loop/, rounds/r01..r04 other directories).
- No git. No `rm -rf`. Never `pkill -f`; kill only processes you started, by PID. No servers, ports, timan107, LIBERO
  workers or chains. Do not read tests/review_tests/.
</hard_constraints>

<completeness_contract>
Cover every configuration in item 2 whose fit exists, both library scales, both models and suites; say explicitly
which could not be measured and why.
</completeness_contract>

<verification_loop>
Repeat each microbenchmark at least twice (fresh process) and report run-to-run variation; check that the benchmarked
query path returns the same top-k/actions as the logged decisions for a sample (so you timed the real path).
</verification_loop>

<grounding_rules>
Report only what you ran and observed; give exact commands, numbers and file paths. Label hypotheses.
</grounding_rules>

<structured_output_contract>
Write rounds/r04/k8_search_latency/REPORT.md (tables first, then method, then caveats) plus the scripts and raw JSON
results in that directory, then print a one-paragraph summary as your final message.
</structured_output_contract>
