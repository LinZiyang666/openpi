# Stage 4a Phase A — F5 lockstep re-check under N4 injection

Gating pre-requisite for the N2 FollowWinnerGate L3 build (roadmap Stage 4a entry condition #2). Offline, 0 GPU, on localized Stage 3a N4 live data.

F5 statistic = same-episode persistence % over strictly adjacent FULL_HIT step pairs within an episode (winner_id trajectory prefix), plus the Delta winner_step distribution (+1 = lockstep). Mirrors `gate_structure_analysis.py` block [5]. Injected skips break the FH run, so this is the *within-FH-run* persistence in the injected regime.

## Suite: spatial

- **Stage-0 baseline** (always-search): same-episode **94.4%**, Delta+1 **95.0%**, Delta0 3.8%, n_pairs=23915

| L | same-episode % | Delta+1 % | Delta0 % | n_pairs | vs baseline | verdict |
|---|---|---|---|---|---|---|
| L6 | 92.3 | 93.9 | 4.3 | 6869 | -2.1pp | **GO** |
| L8 | 92.6 | 94.6 | 4.2 | 7267 | -1.8pp | **GO** |
| L12 | 93.9 | 94.4 | 4.2 | 7882 | -0.5pp | **GO** |

## Suite: l10

- **Stage-0 baseline** (always-search): same-episode **94.3%**, Delta+1 **80.9%**, Delta0 14.4%, n_pairs=70570

| L | same-episode % | Delta+1 % | Delta0 % | n_pairs | vs baseline | verdict |
|---|---|---|---|---|---|---|
| L6 | 98.2 | 92.6 | 4.4 | 6139 | +3.9pp | **GO** |
| L8 | 98.0 | 94.1 | 3.7 | 6745 | +3.7pp | **GO** |
| L12 | 98.0 | 94.2 | 3.8 | 7541 | +3.7pp | **GO** |

## Go/No-Go

Per-run verdicts: ['GO', 'GO', 'GO', 'GO', 'GO', 'GO']

**Overall: GO** — thresholds: GO if same% >= baseline-10pp and Delta+1 majority; NO-GO if same% < 85% or Delta+1 not majority; else GRAY (owner review).

## Notes & interpretation

- **Methodology self-check**: the recomputed Stage-0 baselines reproduce the roadmap F5 range (same-episode 93-98%, Delta+1 75-97%, appendix A), so the pair engine is faithful to the original F5.
- **Baseline pooling caveat**: each suite baseline pools all always-search configs in `gate_rows.jsonl` (not keybuilder-matched to the specific N4 run). The GO margins (spatial within ~2pp; l10 above baseline) are large enough that keybuilder-matching would not flip any verdict.
- **Why l10 rises above baseline under injection**: l10 is the oscillating suite (high Delta0 dense-replan). N4 injection breaks up the long/oscillating cache-execution runs, so the surviving contiguous FH sub-runs are the cleaner lockstep segments -- within-run persistence and Delta+1 both go *up*, Delta0 drops. Injection concentrates, not destroys, the lockstep opportunity.
- **C8/C11 scope**: this validates only the *structural opportunity* (lockstep still present offline). N2's blind replay runs *without* verdict supervision and changes the execution flow, so its SR effect is offline-unmeasurable and MUST be live-validated (Phase C).
