# R14: Layer 5 "per-task partition" (沿任务分治)

Coordinator brief, 2026-10-04 CDT. It goes to the explorer (astra); the coding agent is sol.

**Status:** this is the offline design phase. Closed-loop runs come after the R13 test grid, which occupies the GPUs for the next 2–3 days. Small B-dev requests can be interleaved.

**Formal design (owner ruling; supersedes the B-dev per-task allocation in Part B and R14's B-pool descent).** No development set for IR knob tuning on any benchmark: the cache library is built from the development pool (LIBERO B pool, MetaWorld seed 8), so tuning on it means tuning on the library's own data. The per-task IR knob is descended directly on the test set: LIBERO uses the whole test set A (500 episodes per suite = 10 tasks × 50 initial states, indices 0–49); MetaWorld MT50 uses seed 7 (50 tasks × 50 initial states). For every task, measure the levels on the test set from the most expensive level downwards, 50 episodes per task per level; a level passes when it has at most 2 fewer successes than same-step-count pure inference on the same GPU and the same initial states (integer comparison); the first failing level freezes the previous level; the reference is always same-step-count pure inference. Library cells whose complete system with the knob off (stall gate only) already reaches the same-step pure-inference success rate are not tuned and use that knob-off result. R14's dev-selected LIBERO knob results are withdrawn and will be re-run under this design.

## 1. The owner's goal
Owner, 10-04: "之后把沿任务分治也做了" — after R13, do layer 5 as well.

The owner left the scope to the coordinator, who set it as two parts:
- **Part A, value of the existing partition.** The deployed layers 1–3 are partitioned by task. Measure what each partitioned component is worth by replacing it with a pooled, all-task version, one component at a time and in combination. The answer should say which components must stay per task, which can be pooled, and why.
- **Part B, extend the partition to layer 4.** Today layer 4 has no task-indexed parameters by rule. Test whether **allocating the layer-4 call budget by task** gives higher SR than the cell-wide knob at the **same total realized IR**. The idea is to call more on tasks where calls help and less where the cache is already as good as the policy. Allocation may come only from library or B-dev data.

You may propose other per-task ideas, such as per-task guard thresholds or partition granularity finer or coarser than the task. Justify them and keep them clearly separate from A and B.

## 2. The deployed system (`rounds/r10/recipe/`, `R10Recipe` + `rounds/r11/knob/`, `R11Knob`)
**What is per task today** (verify every item in the code, `recipe.py` / `fitting.py` / `build.py` and the plugin):
- **retrieval scope:** same-task demos only;
- **metric:** a per-task Mahalanobis metric;
- **corrector:** ten task heads (LOEO motion corrector);
- **guard:** whatever the guard tables index by task;
- **shared components:** the PCA, the distance scale (GC_dist) and kref may be global or per cell. Find out.

**Layer 4:**
- R11 periodic, the reference: one gap cap K per cell;
- R13 `distance_feedback`, specified in `rounds/r13/astra/SPEC.md` and still being validated.

For Part B, build on **periodic** as the base cell-wide knob. If R13 later favours distance, the coordinator will ask for an adaptation.

**Task identity** is known at serving time: LIBERO gives the language instruction, and the plugin receives `task_id`. A pooled variant may still use the task id where the pooled design needs it (for example, for logging). Say exactly what each variant uses.

## 3. Questions
- **A1.** For each partitioned component (scope, metric, corrector heads, guard tables, and anything else you find), replace it with a pooled version fitted on the same library. How much SR and IR does it cost or gain in each cell?
  - Use offline evidence first: retrieval accuracy, action error against the policy, and corrector benefit on held-out library episodes and B-dev cache states.
  - Then design a closed-loop ablation grid.
- **A2.** Is there an interaction? For example, a pooled scope might be harmless with a per-task metric but harmful with a pooled metric. Is "per-task scope only" enough?
- **A3.** At which library sizes does partitioning matter? Small libraries (50) have few demos per task, so pooling might help there; large libraries (500) might not need the partition.
- **B1.** A per-task allocation rule for the layer-4 budget:
  - Inputs may only be library or B-dev estimates, for example per-task cache-vs-policy action gap, per-task held-out cache failure, or per-task B-dev knob-off SR.
  - The rule should be simple, with few scalars. Shrinkage toward the cell-wide setting is welcome because per-task estimates are noisy (5 demos per task at size 50).
  - The **total** realized IR must match the cell-wide knob's target within ±.02, and the allocation should not change the total spend.
- **B2.** At matched total IR, how much SR does B1 add over cell-wide periodic in each cell and dose? Where does it help? Report per-task effects descriptively.
- **B3.** Robustness:
  - noisy per-task estimates (B-dev sample sizes);
  - tasks where the allocation pushes the knob to 0 or to always-call;
  - behaviour at size 500, where there is no B-dev complement.

## 4. Data rules (same as R13)
**Allowed:**
- B-pool libraries and R10 nested subsets;
- B-dev closed-loop roots: `r11_dev_*`, `r11_devknob_50`, `r12_devlooks_*`, `r13_devrec_50` and `r13_devrec_pi05_spatial_50` (these two are being recorded now; see `rounds/r13/DATA.md`), and later `r13_pilot_*`.
  - The R13 validation identities in `rounds/r13/astra/validation_selection.json` are reserved for the R13 pilot. Do not use them to fit anything in R14 until that pilot has run; after that they become ordinary B-dev data.
- `rounds/r13/astra/` (frozen R13 design) may be read.

**Forbidden for fitting, calibration, selection or design:**
- anything on test set A: `offline_search_store/queries/*`, every test-A root of R1–R13, and every R13 test result when it appears;
- you may read REPORT / ANALYSIS aggregates only.

**Per-task B-dev data is scarce:** about 45 dev inits per task at size 50 and about 30 at size 200. If you need more closed-loop B-dev recordings (per-task knob-off SR, per-task pooled-variant pilots, shadow labels), write `REQUESTS.md` early with exact arms, cells and episodes. Note that GPU time is mostly taken by R13 for the next 2–3 days.

## 5. Deliverables (`exp/offline_search/rounds/r14/astra/`)
1. **`OFFLINE.md` + code:** the per-component partition analysis and the per-task allocation analysis on library and B-dev data.
2. **`SPEC.md`:** implementation-ready variants for sol:
   - pooled-component switches on top of R10Recipe;
   - the per-task-allocation knob on top of R11 periodic;
   - exact fitting procedures;
   - the dev pilot and its pass/stop rules (budget ±.02, engagement);
   - the frozen test-A grid: arms, cells, doses, replicates, host pinning, with GR00T cells kept on one host. Keep the grid economical, and state its size and estimated run time.
3. **`PREDICTION.md`**, timestamped before any closed-loop result of the new variants.
4. **`REQUESTS.md`** (early, optional).
5. **Later**, when the coordinator resumes you:
   - `ANALYSIS.md`;
   - `REPORT.md`, with the owner section first in plain Chinese and no codenames;
   - `HANDBACK.md`.

## 6. Operating rules
- Write only in `exp/offline_search/rounds/r14/astra/`.
- No launches, no remote commands, no git, no `rm -rf`, no `pkill -f`.
- **CPU:** `taskset -c 10-21,54-65`, `.venv/bin/python`, `PYTHONPATH=.:src`, single-threaded BLAS unless you deliberately parallelize inside your set.
- **GPU:** the local 4090 is currently lent to another project and then serves R13. Do not use it unless the coordinator says so.
- Every number in your reports must come from code you ran in your directory. Write `STATUS.md` notes as you go.
