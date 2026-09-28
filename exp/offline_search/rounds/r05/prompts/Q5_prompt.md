<task>
You are R5 coding agent Q5: wire the GPU-resident retrieval (R4 K9 prototype) into serving as an opt-in path, first in
shadow mode, so the coordinator can measure live agreement and latency in closed loop without changing any action.
The owner's view: a library that fits on the GPU can be compiled into the model (CUDA graph) so retrieval costs almost
nothing; K9 (rounds/r04/k9_gpu_retrieval/REPORT.md, gpu_awm.py) measured ≈ .36–.63 ms incremental inside stage 1's
graph, top-1 agreement ≥ 99.9 % (step-0 early distances ill-conditioned), PCIe 131 KB → 1.4 KB per decision.
1. Serving integration (π0.5 first): an opt-in flag (e.g. `--os-gpu-retrieval shadow|serve`) that builds K9's module
   from the arm's fitted method (AWM / guard-only MixedJudge feature set; K7 / blind methods where the prototype
   covers them, otherwise refuse clearly), keeps the pooled camera keys on the GPU right after stage 1 (no host copy on
   the GPU path), and runs retrieval + synthesis on the device. `shadow`: the CPU method still decides and serves;
   log per decision the GPU top-1 / top-16 / chunk / confidence agreement and GPU-path latency (CUDA events), plus the
   CPU path latency. `serve`: the GPU result replaces the CPU retrieval (guards / verdict stay on CPU, fed by the GPU
   features) — implement it but the coordinator will only enable it after shadow agreement is acceptable. Capture the
   retrieval in a CUDA graph (fixed shapes; per-connection static buffers; no host sync until the final small copy) and,
   if the stage-1 graph path exists in the served stack, report whether stage 1 + retrieval can share one graph.
   Honour K6's per-connection concurrency contract (many connections, one GPU).
2. Tests: CPU-only fake-stack tests where possible; GPU tests with a STAGE1_ONLY or full π0.5 model only if `nvidia-smi`
   shows ≥ 14 GB free, using ≤ 6 GB yourself, ports only in 23180–23189, and releasing the GPU after each test. Prove:
   flag absent ⇒ byte-identical behaviour and logs (re-run all existing selftest matrices K2, K1, K4, K5, K6, K7, K10 and
   Q2's if installed); shadow ⇒ identical served actions to the CPU path; agreement statistics on replayed real queries.
3. Arm specs rounds/r05/q5_gpu/arms_q5.json: shadow mode on π0.5 l10 at 50 and 500 for (a) pure-cache AWM CL2 and
   (b) K7 anchor_tail (if supported; otherwise say why), with the exact flags; prefits into /tmp/q5_fits/ if new fits
   are needed (reuse existing fits when method/kwargs/cell match exactly).
</task>

<context>
Repository /home/weiland/projects/openpi (branch Ziyang). Read first: exp/offline_search/rounds/r05/CODING_BRIEF.md
(binding rules), rounds/r04/k9_gpu_retrieval/REPORT.md and code, rounds/r04/k8_search_latency/REPORT.md,
rounds/r04/k6_concurrency/HANDBACK.md, rounds/r04/k10_policy_tail/HANDBACK.md, closed_loop/plugin.py,
closed_loop/serve_pi05.py, src/openpi/cache/interceptor.py (read only).
PLUGIN OWNERSHIP: agent Q2 currently owns closed_loop/plugin.py and friends (GR00T policy tail). Develop your change in
rounds/r05/q5_gpu/dev/. Do NOT install into closed_loop/ until rounds/r05/q2_groot/HANDBACK.md exists (Q2 finished);
then rebase your change onto the INSTALLED plugin, re-run every test, and install each shared file in ONE atomic step
(temp file in the same directory + mv). If Q2 has not finished when you are otherwise done, finish everything else,
leave a ready-to-install candidate plus an install script that checks the preimage hash, and say so in the hand-back.
The coordinator runs closed-loop chains on this code right now; codex never runs chains or LIBERO workers.
</context>

<hard_constraints>
- Your CPU range: 34-37,78-81. Prefix every python command with `taskset -c 34-37,78-81`, BLAS/OMP threads 1, never more
  processes than threads. CPUs 38-43 and 82-87 belong to another project: never use them.
- GPU: only as stated in the task (≥ 14 GB free before starting, ≤ 6 GB yours, stop on OOM, ports 23180–23189 only).
- Write only under rounds/r05/q5_gpu/ and /tmp/q5_*, plus the atomic plugin install described above. Do not modify
  src/, harness/, profile/, ops/*, other rounds directories. No git, no `rm -rf`, never `pkill -f`, no timan107, no
  LIBERO workers, no chains, never touch ports 23150–23169 or others' processes. Do not read tests/review_tests/.
</hard_constraints>

<completeness_contract>
Resolve fully: integration, tests, agreement + latency evidence, arm specs, hand-back; state what remains unimplemented.
</completeness_contract>

<verification_loop>
Re-run all checks against the final installed (or ready-to-install) files and report final numbers.
</verification_loop>

<grounding_rules>
Report only what you ran and observed; exact commands, numbers, file paths. Label hypotheses.
</grounding_rules>

<structured_output_contract>
Write exp/offline_search/rounds/r05/q5_gpu/HANDBACK.md, then print a one-paragraph summary.
</structured_output_contract>
