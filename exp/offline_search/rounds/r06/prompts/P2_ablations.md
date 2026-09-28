<task>
You are coding agent P2 of the offline_search exploration line (action cache for VLA policies π0.5 / GR00T N1.5 on
LIBERO) in /home/weiland/projects/openpi (branch Ziyang). The paper uses two nested configurations:
- A (commit-cache): vision anchor → retrieve/synthesize a chunk → execute 10 control steps (anchor + one blind decision),
  no policy calls. Specs: π0.5 `/home/weiland/trace_runs/os_closed_loop/r05_ptail/arms_in.json` (`r5t_p_*_tail1uc`)
  and `r04_blind/arms_in.json` (`r4b3_p_sp_500_tail1uc`); GR00T `r05_x/arms_in.json` (`r5x_g_*_tail1u`). Method:
  `exp/offline_search/rounds/r04/k1_blind/blind_awm.py:BlindAWM` on top of `rounds/r02/g1_awm/awm.py:AWM`.
- B = A + committed policy rescue: MISS only when one of four guards fires (stuck with vision confirmation, terminal
  row with closed gripper, overtime-and-lag, no-progress), the policy chunk also executed for 10 steps. π0.5:
  `rounds/r05/q1_commit/judge.py:CommitJudge` (specs `r05_q1/arms_in.json`, `r5q1_c10_p_*`); GR00T:
  `rounds/r06/p1_groot_commit/judge.py:GrootCommitJudge` (read its HANDBACK.md; specs `rounds/r06/p1_groot_commit/
  arms_p1.json`). The guard code is `rounds/r03/h3_judge/judge.py:MixedJudge` + `rounds/r04/k7_guard/judge.py` +
  `rounds/r04/k1_blind/judge.py`.

Build two paper ablations in a new directory `exp/offline_search/rounds/r06/p2_ablations/` (you own it):
1. **Metric ablation of A**: identical to A (same PCA-64 per camera + 8 state features, same per-task z-scoring, same
   16-neighbour kernel synthesis, same kref, same early step-0 branch structure, same anchor_tail budget 1) except
   the learned per-task metric M_τ is replaced by the identity, i.e. Euclidean distance on the z-scored 136-d features
   (and likewise for the early metric). Prefer existing kwargs if they reproduce this exactly (check whether
   `codes`/`lam` can do it; a very large `lam` only approximates it — say which you used and prove the ranking equals
   Euclidean-on-z-scored top-16 on recorded queries). Otherwise a thin subclass in your directory. Both models × {l10,
   spatial} × {50, 500} = 8 arms.
2. **Trigger leave-one-out for B**: four B variants, each with exactly one of the four guards disabled (stuck,
   terminal, overtime, no-progress), everything else identical to B. Note overtime depends on the stuck counter —
   disabling "stuck" must not silently disable overtime; document the exact semantics you chose. Both models ×
   {l10, spatial} × 50-episode library = 16 arms (the cells where B matters). Prove on recorded replays that each
   variant equals B except where the disabled guard was the only firing reason, and report per variant how many MISS
   decisions it removes on replay.
Also a "B with all four guards disabled == A" check for the new code paths if you touch the judge.

Deliverables: arm specs `arms_metric.json` (8) and `arms_trigger_loo.json` (16) in emit_arms format with `<RUN>`
placeholders (copy all other fields from the existing A / B specs exactly), prefits into
`/home/weiland/trace_runs/os_closed_loop/r06_abl/fits/` (sizes, sha256), replay tests with numbers, and
`HANDBACK.md` (files, switches, tests, specs, prefit commands, smoke recipe, caveats). Final message: short summary.
</task>

<hard_constraints>
- CPU only: prefix every python command with `taskset -c 26-29,70-73`, at most 8 processes, OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1, CUDA_VISIBLE_DEVICES=''. Python: `.venv/bin/python` from the repo root.
  CPUs 38-43 and 82-87 belong to another project: never use them.
- Do not edit `src/`, `harness/`, `closed_loop/*`, `ops/*` or earlier rounds' files; subclass or wrap. If a shared
  plugin change were unavoidable, stop and say so in the hand-back instead of making it.
- No git. No `rm -rf`. Never `pkill -f`. Never start servers, LIBERO workers, chains, or touch tmux sessions, ports
  23100-23199 or timan107 — closed-loop experiments are running on this machine right now; the coordinator runs all
  experiments. Do not read `tests/review_tests/`.
- Do not stop to ask questions: when something is ambiguous pick the most reasonable reading, state it in HANDBACK.md,
  and continue.
</hard_constraints>

<grounding_rules>
Report only what you ran and observed, with commands and file paths; label anything unverified.
</grounding_rules>
