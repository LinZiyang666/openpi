# Ideation brief (static part) — offline retrieval-method exploration

Read with: the protocol `logs/offline_search_exploration.log.md` (authoritative), `exp/offline_search/harness/README.md`
(API, metrics, valid dims), `exp/offline_search/profile/README.md` (diagnostic tools), and the round-specific
findings file `exp/offline_search/rounds/rNN/FINDINGS.md` given in your prompt.

## The task being optimized
A VLA policy (π0.5 or GR00T N1.5) controls a robot in LIBERO. At each decision (every 5 control steps) the system may
skip the expensive policy call and instead **reuse an action chunk stored in a library** of past rollouts. Offline we
judge a method by how close the action it returns is to what the policy itself would have produced at that
observation (`a_inf`, recorded for every decision), and by whether its confidence knows when to trust itself
(risk–coverage). Online, a threshold on the confidence decides HIT (reuse) vs MISS (run the policy).

## Hard constraints
- CPU-only evaluation. Cost tiers: T0 training-free, T1 closed-form statistics (seconds), T2 training (GPU allowed,
  shorter is better). The system's selling point is low cost and fast deployment: prefer T0/T1; T2 must earn it.
- No external models (CLIP, DINO, …) — the cache exists to save compute. Lowest priority, avoid.
- The returned action must come from library actions (select one, or combine several); never call the policy.
- Online-available inputs only at query time: current pooled keys / tokens (tok subsample) / images (tok subsample,
  π0.5 has images, GR00T images only on the query side), robot state, task id, step index within the episode,
  this episode's history (earlier keys, executed actions). Not available: episode length, success, a_inf, a_hit.
- The library may be built freely (entries, fields, size): `current` (online libraries), `bpool_all` (all build
  episodes incl. failures; GR00T 500/suite), `bpool_cs` (π0.5 500/suite, Aug batch), or anything derived.
- ⛔ Valid dims: actions (H,32) only dims 0..6 are real (dim 6 = gripper ±1); GR00T dims 7..31 are large noise-like
  padding. Only the first 5 steps of a chunk are executed. robot_state: π0.5 only dims 0..7 valid (rest 0), GR00T 8.
  Use `harness/dims.py`.

## Settled negative results — do not re-propose
History depth d>1 mixed into the score (score smoothing); percentile normalization; RRF as the main fusion;
InfoNCE projection heads trained on same-distribution data; choosing fusion weights by offline action-L2 argmin;
refining weight grids at 100-episode resolution; linearly blending cached and policy actions in action space.
(Why: see `logs/session_handoff.md` appendix and the protocol §3.3/§7.) Variants that change the mechanism (e.g.
trajectory *tracking* or HMM filtering instead of score smoothing) are allowed.

## What a proposal must contain (output format)
For each proposal (≤ 4 per agent):
1. **Name** (snake_case) and one-line pitch.
2. **Hypothesis**: what is wrong/limiting today and why this fixes it (mechanism, not vibes).
3. **Algorithm**: precise enough to implement: representation, similarity, fusion/selection, confidence, library
   construction, any fitted parameters and on what data (library only unless justified).
4. **Predicted effect** on the scoreboard (err mean/median, regret, AURC, bad rate, inf vs cache) and on cost
   (bytes/entry, ms/query, fit time), with the reasoning.
5. **Cost tier** (T0/T1/T2) and deployment implications.
6. **Kill criterion**: the concrete result that would falsify it.
7. **Cheapest diagnostic first**: which profile tool / quick measurement would tell us early whether it can work.
8. **Variants** worth sweeping (few, principled).

## ⛔ CPU budget (owner rule, mandatory)
The machine is shared by several agents running at once. You get a fixed logical-CPU range in your prompt
(physical cores + their hyperthread siblings; topology: CPUs 0–43 are physical cores, 44–87 their siblings).
- Prefix EVERY command that runs Python/numpy work with `taskset -c <your range>`.
- Never run more processes/workers than the number of logical CPUs in your range; OMP/MKL/OPENBLAS threads = 1 per process
  (a single one-off BLAS-heavy process may use threads = your range size instead).
- If you started something that exceeds the budget, stop it by PID (never `pkill -f`) and relaunch within budget.
- Full-scale runs belong to the coordinator (it reserves CPUs with batch `--cpus`).

## ⛔ Library scale in every conclusion (owner rule)
Every result or comparison must state the library it used: episodes, entries, key bytes/entry, action bytes/entry,
total library volume (+ fixed overhead such as PCA bases), next to the current deployed library (pkl on disk:
π0.5 431 MB spatial / 1103 MB l10, GR00T 429 / 1068 MB; 262 KB key per entry). Until the owner settles whether the
10× library is deployable, report every conclusion both at current library size (~50 episodes) and on the 10× library.
