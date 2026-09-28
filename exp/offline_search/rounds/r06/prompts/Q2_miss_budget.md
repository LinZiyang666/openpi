<task>
Your question (Q2): **how many MISS (policy calls) should we spend, as a function of library quality and the SR target
(ideally pure-inference SR), and how should the budget be split across tasks / episodes?**
- From all existing closed-loop arms, build SR-vs-MISS-share (and SR-vs-IR) curves per cell and library size: periodic
  MISS arms (`r04_frontier` per6/8/12, `r03_mx` perk3/perk5), guard-only (`r03_mx` g / g500, `r04_rep`), confidence-quantile
  arms (`r03_mx` *_h50 / *_h70), B (`r05_q1`), GR00T periodic B (`r05_q2` G10), A (m = 0), pure inference (m = 1),
  and the randomized CALL/CACHE data (`r04_k5`, causal). Quantify diminishing returns: marginal SR per unit m.
- Model the required budget: e.g. the SR gap of A to pure inference as a function of a library-quality measure (a
  simple LOEO coverage proxy is fine for this; another agent designs the full quality score), and the m needed to close
  a fraction of that gap. Per-task budgets (some tasks need many calls, some none) — measure the per-task SR gaps and
  per-task MISS counts already realised.
- Propose a budget rule that is calibrated from the library (portable), with a single user-facing knob (for example a
  target SR loss ε or a target IR), and predict IR / SR per cell.
Your directory: `exp/offline_search/rounds/r06/ideation_Q2/`. CPU range: 14-17,58-61.
</task>
