# sol task: per-task IR knob on MetaWorld MT50 (CPU preparation only)

Repo `/home/weiland/projects/openpi`. The owner wants the LIBERO per-task IR knob experiment
(`exp/offline_search/rounds/r17/knob/REPORT.md` §1) repeated on MetaWorld MT50 for **π0.5, GR00T, DP and Fast-WAM**,
**full-step and reduced-step**, at **every library size**, with **one knob per task (50 tasks)**. GPU runs come later
and are launched by the coordinator; your job is to make every step runnable on CPU and hand back exact commands.

## Read first
- `exp/r17_integration/ops/RUN_CONFIGS.md` (MetaWorld sections, "MT50 cache studies", Pitfalls).
- LIBERO knob tooling, which you extend (do not copy): `ops/cache_knob.py` (build/calibrate/commands, `serve_loeo`,
  `periodic_gap`, levels), `rounds/r17/knob/{common,descent,test}.py`, `openpi_ext/scratch/r17_ops/knob/prepare_dev.sh`.
- MT50 cache study: `ops/cache_study.py` (`generate --sizes 250 500 1000 1500 2000 2500 --miss-steps K`), the studies
  `/data/r17_library/studies_mt50/{dp_k10,fw_k3,gr_k2,pi_k2}/STUDY.json`, libraries
  `/data/r17_library/{dp,fastwam,pi05,groot}_mw_mt50/bpool` (subsets/fits `size_{5,10,20,30,40,50}` per task; library =
  seed 8 idx 0–49, `library_episode_id` = task×50+idx), MT50 IR stage costs `exp/r17_integration/profile/results_mt50/`.

## Design (fixed by the owner; same as LIBERO)
- Base = the full system: Cache + Long horizon (C10) + Corrector + Q3 Stagnation Guard (`chrg`; π0.5 has no C10, so
  `crg`), plus the Periodic Refresh Gate. Levels: `pure`, `k50`, `k40`, `k30`, `k20`, `k10`, `koff` (periodic off,
  stall guard only). Every library size 250, 500, 1000, 1500, 2000, 2500 (5..50 per task).
- Steps. Full-step: every model call (MISS, stall, periodic, pure) uses the native sampler: DP DDPM100, Fast-WAM K10,
  π0.5 K10, GR00T K4. Reduced-step: DP DDIM10 (eta 0), Fast-WAM K3, π0.5 K2, GR00T K2 (the MT50 defaults). GR00T's
  endpoint K is fixed at startup from the study YAMLs.
- Periodic gaps: calibrate each level per model × size × step mode on the selected library with LOEO replay, the
  same method the LIBERO DP/Fast-WAM knob used (target fraction of model calls).
- **No separate dev set (owner, 10-13 22:4x).** Seed 8 (2,500 episodes) is only the cache library. The descent runs
  directly on seed 7 idx 0–49 (50 per task per level, every size): run k50, k40, k30, k20, k10, koff for all 50 tasks
  in one campaign; the pure level is the existing same-step pure baseline of the same GPU type: DP full `<new H100 run>`
  (the existing `tdp7_0729` is on the 4090), DP reduced `tdp7d10_0851`, Fast-WAM `tfw7full_0944` / `tfw7k3_1051`,
  GR00T `tgr7h_full` / `tgr7h_k2` (H100, queued), π0.5 `tpi7_0536` / `tpi7k2_1135` (4090). Per task walk down
  pure → k50 → … → koff; a level passes with at most 2 fewer successes (of 50) than pure on the same inits; the first
  failing level freezes the previous one. The frozen level's 50 episodes are the final result; no separate test round.
- No-knob rule: a size whose `koff` cell (no periodic refresh) already reaches the same-step pure SR on TEST uses
  that cell directly. Report MT50 SR as the equal-weight mean of the four difficulty groups first, then the per-task
  mean; report formal IR from the MT50 stage costs.
- Placement (RUN_CONFIGS cache-study topologies): DP, Fast-WAM, GR00T on H100; π0.5 on the 4090. Workers timan107 and
  timan1 (GPU 3 only, ≤40 in total across lanes).

## Deliver
1. `cache_knob` (dev/test/calibrate/commands) accepts `dp_mw`, `fastwam_mw`, `pi05_mw`, `groot_mw` with
   `--miss-steps full|reduced`, all six sizes, seed-8 DEV with LOEO and seed-7 TEST plans, partial retests.
2. `descent.py` / `test.py` analyse MT50 lanes (50 tasks, group mean + per-task mean, the references above).
3. Formal command for the DP full-step H100 seed-7 pure baseline.
4. CPU stub tests following the LIBERO knob tests. Handback `exp/r17_integration/handbacks/HANDBACK_MT50_KNOB.md`:
   files changed, what the tests cover, episode counts per model and mode, and the exact GPU command order.

## Rules
- Only this. No new validation layers, gates, fingerprint or admission checks (owner: a gate that blocks a normal run
  is a bug). CPU only: start no GPU process, server, worker or campaign. No `tether push/pull/expose`, no credentials,
  no git commit/push, no `pkill`, no `rm -rf`; do not touch running campaigns under `/data/mw_finetune/eval/` or
  `/data/r17_frozen/`.
