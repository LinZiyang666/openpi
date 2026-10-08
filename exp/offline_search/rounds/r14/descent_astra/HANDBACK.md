# R14 final descent handback

Complete: [REPORT.md](REPORT.md) leads with the short Chinese owner conclusion and frozen task table; [TEST_A.md](TEST_A.md) has full confirmation statistics; [test_a.json](test_a.json) includes episode arrays and input hashes. No test-A selection or tuning; frozen protocol, B outputs, levels and uniforms are unchanged.

Confirmation audited all 6,000 accepted episodes. Task-descent SR 91.00% vs pure 93.20%; paired Δ -2.20 pp, CI [-3.65, -0.75]. There are 11/40 task point violations. Pooled IR 0.316519 saves 37.44% vs pure but costs 12.73% more than uniform. B pure-relative SR advantage optimism: 2.78 pp. Do not advertise an every-task guarantee or matched-IR superiority.

Rerun from the repository root (only analysis and report writes):

```bash
taskset -c 10-21,54-65 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python exp/offline_search/rounds/r14/descent_astra/analyze_test_a.py
```

A schema adaptations are documented in TEST_A.md. The script verifies frozen B seals, A manifests/definitions/fits, paired accepted attempts, all decision ledgers and input stability before writing. All analysis writes stay in this directory. No git, experiment launch, remote command or GPU use. Recommend a separately versioned B-only development confirmation before any deployment claim; never repair the table from A.
