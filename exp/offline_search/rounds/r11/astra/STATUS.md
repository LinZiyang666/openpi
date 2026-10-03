# Ready for handback

The final deliverables are complete: `SPEC.md`, `OFFLINE.md`, `ARM_GRID.md`, `PREDICTION.md`, `REPORT.md`, and `HANDBACK.md`. `final_audit.json` records the passing artifact audit. Use `freeze.json` for exact arm settings and `COORDINATION.md` for missing matched-budget random controls in the current combined grid.

The notes below describe the completed analysis inputs and shared-model interface.

All inputs are B libraries and R10 B-only artifacts; audit refuses every `os_closed_loop` path and every replay directory.

`data/<cell>/signals.npz` has parent `row`, episode, step, task, outer fold, whole-episode-held-out geometric distance, corrected motion error, neighbour disagreement, gripper signals, and a strictly outer-held-out pooled error score. `top` is a **subset-local** donor row; convert through the corresponding R10 subset rows before joining another table.

`experiment.py` retains R10's fixed selected-library PCA convention. Each outer fold excludes complete query episodes from the metric, donors and corrector. The pooled error model is trained on inner whole-episode metric holdouts that also exclude the outer query episodes. No task identity or task-specific parameters enter that model. Its training label is pre-corrector motion error; its evaluation target is post-corrector motion error.

`signals.py:dag` is the independently checked IR model: exact expectation on fixed B states with the span guard, hit-tail veto, policy tail, episode endings, and overlap. State is previous look lag plus previous span capped at the guard threshold. It agrees with opus's shared model on identical input arrays; ordinary reachable looks remain on the regular anchor cadence. It does not attempt to infer changed trajectories.

Final evidence: error prediction and neighbour disagreement concentrate motion error; gripper transition alone does not help on long tasks and is dropped. Distance remains as the requested simple baseline. A half-random error hybrid and feasible higher-target adaptive variants are proposed. The combined grid, matched random controls and same-batch knob-off controls fit the brief's budget; see the generated coordination table.

Use final SPEC/HANDBACK for integration; these exploratory files are not production adapters.
