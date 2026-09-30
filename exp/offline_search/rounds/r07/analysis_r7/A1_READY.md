**Historical Phase 1 readiness note — superseded by [the Phase 2 analysis](../ANALYSIS.md).** The coordinator's pre-CT clarification in SELECTION §8 resolves the cost-match scope as per-cell. The final scripts and outputs use that binding reading and accept both completion-marker formats.

**A1 Phase 1 ready — 2026-09-30, CDT.** At this phase-1 snapshot, no method conclusions or `ANALYSIS.md` sections had been written.

Implemented [a1_common.py](../analysis_scripts/a1_common.py), [a1_analyze.py](../analysis_scripts/a1_analyze.py), and [a1_validate.py](../analysis_scripts/a1_validate.py). They produce all five §5 rule evaluations, per-cell SR @ owner IR tables, paired bootstrap intervals (10,000 draws; seed 20260930), exact McNemar and conservative CP bounds, SW actual-camera pricing plus both alternative price reports, R6 frontier placement/crossings, and accepted-attempt exception audits. Phase 1 withholds acceptance verdicts; incomplete arms are explicitly `missing`.

Validation passed:

- 17 regression checks, including acceptance boundaries, replicate clustering, missing arms, camera completion, reused connection counters, exception joins, and complete/missing frontier matrices.
- All 44 profile arms: 880 episodes / 37,999 decisions reproduce the existing report's outcomes, counts, and owner costs.
- All 84 historical references match R6 outcomes/costs; all eight frozen frontiers and L10 crossing costs reproduce. Four R6 U30/C30 arms additionally pass raw-decision reconciliation.
- The fixed **03:42:05 CDT** evaluation inventory contains three completed CU30 arms: π0.5 L10-50, π0.5 Spatial-50, GR00T Spatial-50. All pass; 25 arms remain `missing` in this snapshot. Across the 84 references and these three arms, all 43,500 accepted records have termination coverage and zero surviving exceptions. Profile audit also passes.

Outputs are the `a1_*.json` and `a1_*.csv` files here; [a1_snapshot.json](a1_snapshot.json) records inventory, conventions, source/script hashes, and pending rules. Debug reruns reused that finite inventory; no experiment was polled or awaited.

**Interpretation retained for Phase 2:** §5.4 does not explicitly specify pooled versus per-cell ΔIR. Both readings are implemented; differing verdicts are marked `scope-dependent`. Aggregate IR and equal-episode IR remain separate. Historical wrist points retain R6's frozen pricing. Applied-control IR remains unavailable.

After the Phase 2 follow-up, run from the repository root (omit `--inventory-from` to collect the completed matrix), then write `ANALYSIS.md` §§1–5:

```bash
taskset -c 22-25,66-69 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python exp/offline_search/rounds/r07/analysis_scripts/a1_analyze.py --phase final
```

Validation can be repeated with the same prefix and `a1_validate.py`. Phase 1 stops here; remaining evaluation data and the explicit Phase 2 follow-up are pending.
