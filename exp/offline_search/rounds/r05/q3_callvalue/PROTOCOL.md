# Q3 analysis protocol

The primary fit is the byte-for-byte copy `cost_solver_reference.py` of ideation B's
`cost_solver.py`. Its `causal_fit` and `ope` functions are called unchanged. The initial
read-only inspection of the coordinator's exports found zero full-data and init-fold
suppression probabilities at both scales. No threshold is tuned in response.

- Audit raw accepted attempts with K5 `load_arm` and `validate_pairs`; require all ten
  tasks and original inits 0..49 in each complementary replicate. Compare the raw-derived
  episodes and recomputed K5 estimates with the coordinator's exports.
- Separate g50 and g500, with the exact K5 baseline rho values, support >=30 init
  clusters and >=10 observations per treatment, loss tolerance .01, and the fitter's
  approximate cluster-normal Bonferroni bounds. The fitter corrects each outcome over
  observed leaves; it does not jointly correct both outcomes or the two library scales.
- Five folds are `original_init % 5`. Fit only on the other four folds. Compute held-out
  IPW contributions per init cluster (both replicates retained) and pool those contributions.
  Report SR change as policy minus CALL and cost saving as CALL minus policy, in
  C_rho units per episode, not IR. Per-cell effects are population contributions, with
  zeros for other cells, exactly as in the original fitter.
- Enumerate all 48 prespecified leaves in reports, including unobserved ones with
  unsupported/default CALL. Retain unknown contexts only for parent/ITT diagnostics.
  Report full-data support and probability, five training-fold probabilities, and pooled
  held-out SR/cost intervals. Use pointwise 95% and Bonferroni across the 48 reported
  leaves (separately for each outcome). All-zero policy contrasts have structural [0,0]
  intervals; they do not establish that suppression is harmless.
- Report the predeclared parent-only first/third-landmark variant separately, using the
  same fitter with all exposed contexts mapped to `{}`. It has two Bonferroni-tested
  cells and the same support/loss rules; it is not a post-hoc merging of favorable leaves.
- Task sensitivity: fit on nine tasks, evaluate the tenth, for all ten tasks and both
  table families. Also tabulate raw ITT for each held-out task and each nine-task
  remainder. Report the number/range of leave-one-task-out sign changes. Estimate paired
  scale contrasts by the same original-init bootstrap weights and also a Student-t
  interval over ten task means (df=9). Include both common rho values to distinguish
  a scale interaction from changing the cost target. These are sensitivity analyses,
  not an extra selection sweep or a claim of ten-task asymptotic precision.
- A deployable primary table must have a nonzero full-data supported leaf, pooled
  held-out cost saving lower bound >0, and held-out SR lower bound >=-.01. Parent
  tables are separately reported with the same gate. No deployment or arms when the
  gate fails. A null table does not require a serving plugin or randomized control.
- Re-run final fits and held-out evaluations, compare deterministic artifacts byte for
  byte, and independently verify IPW and clustered SE arithmetic. All trained tables
  and rollout-derived analysis are **borrowed big-library information**.

No servers, GPUs, simulators, git, or shared-file edits. All Python commands use the
user's CPU range 26-29,70-73 and single-thread BLAS/OMP, with CUDA disabled.
