<task>
R15 track A: the π0.5 B-dev descent arms finish about 16:40 CDT; GR00T's finish about 18:30. To avoid idling the H100, add a per-model partial application to exp/offline_search/rounds/r15/descent/analyze.py and a matching per-model test-A emission, so the coordinator can freeze the π0.5 level tables and launch π0.5 test A while GR00T is still running.
</task>
<key_points>
1. New option, e.g. `analyze apply --models pi05` (and later `--models groot`), applying the frozen R14 rules unchanged to only those model's cells: per-task first-failure 5 pp primary selection, 0/2 pp alternates, uniform level by the same rule, using that model×suite shared C10 reference. Per-cell selection depends only on that cell's arms and its C10 reference; prove that in code/tests (selection for a cell is byte-identical whether computed in a partial or a full run).
2. Write partial outputs to distinct files (e.g. levels_pi05.json, uniform_pi05.json, DESCENT_pi05.md / analysis_pi05.json) so a later full run cannot be confused with them; the later full apply must reproduce the same π0.5 tables byte-for-byte (add a check). Cross-cell items (simultaneous confidence bands across a size group, pooled summaries) are reporting only; in a partial run label them partial or defer them to the full run. They never change selection.
3. Ensure `prepare emit-test-a --levels levels_pi05.json --uniform uniform_pi05.json --out <dir>` works for a subset of cells (only π0.5 cells), plus verify-test-a / plan-test-a / install-test-a on that subset, and later a GR00T subset into a separate out dir. Queue lines: π0.5 @h100.
4. Do not change any decision rule, threshold or protocol text (PROTOCOL.md stays byte-identical to R14's frozen text). Do not run apply on real outcomes yourself; test with synthetic data only. Do not read test-A data.
</key_points>
<constraints>
CPU only: taskset -c 2-9,46-53 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python. Closed-loop chains are running on both fleets: no servers, GPU, tether/remote ops, sync/chain, git, pkill, rm -rf. Write only under exp/offline_search/rounds/r15/descent/. pytest with -p no:cacheprovider and scratch outside the repo. Be quick: needed by about 16:40 CDT.
</constraints>
<done>
Append a short section to exp/offline_search/rounds/r15/descent/HANDBACK.md: the exact commands for the π0.5 partial apply, emit/verify/plan/install of the π0.5 test-A subset, and the later GR00T / full runs, plus the equivalence checks you added.
</done>
