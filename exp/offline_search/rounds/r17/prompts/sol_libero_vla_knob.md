# sol follow-up: per-task IR knob for π0.5 and GR00T on LIBERO (CPU preparation only)

Same thread, after `HANDBACK_LIBERO_VLA.md`. The owner reruns π0.5/GR00T's reduced-step per-task IR knob on LIBERO
with the R17 rules that DP/Fast-WAM/Cosmos already use (`exp/offline_search/rounds/r17/knob/REPORT.md` §1,
`descent.py`, `test.py`, `openpi_ext/scratch/r17_ops/knob/prepare_dev.sh`):
- Full system: Cache + Long horizon (C10) + Corrector + Q3 Stagnation Guard, plus the Periodic Refresh Gate
  (levels pure, k50, k40, k30, k20, k10, koff). Every model call runs reduced steps **k = 1**.
- Reference = **k = 1 pure inference** on the same inits (not full-step pure inference). Dev: pool B inits 0–29,
  retrieval excluding the query's own episode, 30 episodes per task per level, 5-point tolerance, stop at the first
  failure. Test: official A, 50 per task; ≥5 successes below the reference → pure level, 3–4 below → one level up,
  rerun only those tasks, repeat until every task is ≤2 below.
- Deliver: the knob tools accept `pi05_libero` / `groot_libero` (dev and test campaign preparation, descent and test
  analysis), the k = 1 pure-inference dev and test baselines as formal commands, and the exact GPU command order.
  Same rules as before: CPU only, no gates, no tether push/pull/expose, no commit, append to `HANDBACK_LIBERO_VLA.md`.
