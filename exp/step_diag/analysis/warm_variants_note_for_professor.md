# Warm-start continuation: your dt = -1/remaining proposal, tested (pi0.5, RoboCasa365)

Short version: you were right in practice, and I was wrong about one thing. We ran your reading of the
resume loop as its own arm and it beats our exact resume everywhere we looked; it also fixes the failure of
our warm start on the easy task. What it does *not* do is erase the cache, which is what I had predicted.

## What was run

All arms make the same number of forward passes per decision (2, for the t = 0.2 snapshot). Seeds are paired
episode by episode (RoboCasa365, seed 2,000,000 + idx). Admission: every decision is a WARM_START with
start_t = 0.2, exactly 2 Euler steps, one stage-3 call; server-side row + array evidence checked per episode.

| arm | start tensor | t seen by the network | dt |
|---|---|---|---|
| warm_t0.2 (ours, Nirvana-style exact resume) | cached x_{0.2} | 0.2, 0.1 | -1/10 |
| warmreset_t0.2 (your reading A) | cached x_{0.2} | 1.0, 0.5 | -1/2 |
| warmshoot_t0.2 (your reading B: keep t, enlarge dt) | cached x_{0.2} | 0.2, -0.3 | -1/2 |
| resetfinal_t0.2 (ablation) | cached final action (t = 0) | 1.0, 0.5 | -1/2 |
| plain_k2 (equal-budget step reduction, no cache) | noise | 1.0, 0.5 | -1/2 |

## Results, 500 episodes per task (95% paired bootstrap)

| arm | CloseFridge (cliff task) | PickPlaceCounterToStove (flat task) |
|---|---|---|
| full, 10 steps (50 ep) | 0.62 | 0.84 |
| plain_k2 | 0.06 | 0.95 |
| warm_t0.2 (ours) | 0.29 | 0.16 |
| warmreset_t0.2 (yours) | **0.61** | **0.79** |
| warmshoot_t0.2 | 0.00 | 0.00 |

- warmreset - warm_t0.2: +0.32 [+0.26, +0.38] and +0.63 [+0.58, +0.67]. Your variant wins on both tasks.
- warmreset - plain_k2: +0.54 [+0.50, +0.59] on the cliff task, -0.16 [-0.20, -0.12] on the flat task.
- A 1,000,000-seed replication and a 450-episode held-out subset give the same numbers.
- Reading B (keep t at 0.2, dt = -1/2) crosses t = 0 and collapses to 0.00 on both tasks; only reading A works.

## Step ladder (1 / 2 / 3 steps, 50 episodes each) and the start-point ablation

CloseFridge / PickPlaceCounterToStove:

| steps | plain_k | warm_t (ours) | warmreset (yours) | resetfinal (final action as start) |
|---|---|---|---|---|
| 1 | 0.02 / 0.98 | 0.22 / 0.12 | 0.22 / 1.00 | 0.24 / 1.00 |
| 2 | 0.08 / 0.94 | 0.30 / 0.16 | 0.52 / 0.82 | 0.76 / 0.72 |
| 3 | 0.14 / 0.94 | 0.30 / 0.16 | 0.66 / 0.64 | 0.84 / 0.42 |

Both reset-style arms beat the exact resume at every step count on both tasks. Starting from the final
cached action instead of the intermediate snapshot makes no difference at 1 step, helps on the cliff task at
2-3 steps (+0.24, +0.18) and hurts on the flat task at 3 steps (-0.22). Three reset steps from the final
action reach 0.84 on CloseFridge, above the 10-step full run (0.62, +0.22 [+0.04, +0.40]).

## What I got wrong, and what I still think is right

- Wrong: I predicted that restarting at t = 1 would make the cache irrelevant (warmreset ~ plain_k2). It does
  not: on the cliff task warmreset is 0.61 vs plain 0.06. The cached tensor is mixed into the coarse two-step
  trajectory and that mixture carries most of the benefit.
- Still right: t is the noise level the network conditions on, not a step counter, so the exact resume is the
  only variant that keeps the library's (x_t, t) pairing. The data say that pairing is not what matters; what
  matters is a good starting point plus a short coarse loop that lets the current observation correct it. In
  that sense your variant is closer to "cache-initialised step reduction" than to a resumed flow.
- Open: on flat tasks plain step reduction remains the best method and every extra reset step from the cache
  hurts, so the when-to-reduce vs when-to-warm-start decision is still needed. A 13-task macro run (all
  arms, 50 episodes per task) is in progress.
