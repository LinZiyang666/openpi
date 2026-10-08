# R13 final handback

Completed the D5 paper-support synthesis. Read [REPORT](../REPORT.md) for the short Chinese owner conclusion and technical evidence; [ANALYSIS](../ANALYSIS.md) retains all cell/dose/repeat cadence summaries and inference conventions.

The supported claim is that cost feedback sets a cadence close to periodic calling; distance changes placement within it, without a detectable pooled SR improvement at comparable budgets. B pooled D−P is +0.86 [-0.61, +2.29] pp across 2,800 pairs; the frozen matched-cost subset gives +0.85 [-0.73, +2.35] pp. Per-cell results remain descriptive, not selected winners.

Count correction: gates.json has 28 feedback + 28 periodic + 8 static arms, not 32 feedback arms. All observed feedback arms pass target delivery; static misses 5/8. Two paired budget mismatches remain labeled. The original GATES host/release wording and large-grid plan are superseded by coordinator D1–D5.

Paper assets: [compact table](../final_paper_table.csv), [B gap ECDF](../final_b_gap_cdf.csv), [gap summaries](../final_b_gap_summary.csv), [B budget ledger](../final_b_budget.csv), [B paired outcomes](../final_b_sr_pairs.csv), [computed B numbers and provenance](../final_b_numbers.json). For ECDF panels, facet cell/target/rep and method; `kind=any`, `unit=real_looks` matches the paper table. No-feedback static comparisons retain their actual unequal budgets.

Test-A is **PENDING**, with 0/12 fixed lean arms locally validated at the report snapshot. The separately marked A section lists every pre-D5 supplementary arm, including the formerly unspecified GR00T L10-500 periodic .32 arm. The aborted .25 size-50 periodic arm is discarded. No extra data collection, tuning, dose choice, or method/cell selection is authorized by these reports.

Run from `/home/weiland/projects/openpi`:

```bash
taskset -c 10-21,54-65 env CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python exp/offline_search/rounds/r13/astra/final_b_analysis.py
taskset -c 10-21,54-65 env CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python exp/offline_search/rounds/r13/astra/final_test_a.py
taskset -c 10-21,54-65 env CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python exp/offline_search/rounds/r13/astra/final_validate.py
```

For tomorrow's measurement update, only the second command is needed after summaries/journals/manifests/decision logs appear locally under the four `r13_test_{groot,pi05}_{l10,spatial}_rep0/runs/` roots. It reads the fixed twelve arms, flags incomplete local sync, computes pooled owner IR and complete call gaps, and fills paired SR per cell plus the fixed pooled contrast only when all paired cells are complete. Off is contextual. It replaces only the TEST_A_MEASUREMENT markers in both reports; all seven supplements are always listed descriptively. `final_test_a_numbers.json` and CSVs retain sources, hosts, times and budget-match labels. Do not use test-A to alter anything executable. Re-run the third command only if validation or this handback snapshot needs refreshing.

Validation passed: exact B cache/gate ledger reconciliation, ECDF count/mean/CDF checks, bootstrap invariance to duplicate init observations, and completed decision-log parsing on the two explicitly allowed supplementary arms. Results and hashes are in [final_validation.json](../final_validation.json). No GPU, launches, remote commands, git operations or writes outside this directory were used.

The original `../HANDBACK.md` is frozen in `../MANIFEST.sha256`, so this new `final/HANDBACK.md` is the final handback. All 126 manifest entries still match. No frozen file was edited.
