# sol task: π0.5 and GR00T on LIBERO in the R17 cache study (CPU preparation only)

Repo: `/home/weiland/projects/openpi` (weilandserver). The owner wants the same LIBERO cache-variant study that DP,
Fast-WAM and Cosmos already have (`exp/offline_search/rounds/r17/cache/CACHE_STUDY.md`, groups c cr ch chr cg crg chg
chrg, library sizes 50/100/200/300/400/500 per suite, official A 500 episodes per suite for libero_spatial and
libero_10) for **π0.5 and GR00T**. The GPU runs come later; your job is to make every step runnable and hand back the
exact commands.

## Read first
- `exp/r17_integration/ops/RUN_CONFIGS.md` (launch rules, ports, worker rules, Pitfalls) and `ops/formal_eval.py`,
  `ops/library_campaign.py`, `ops/cache_study.py` (`LIBERO_MODELS`, `CONTROLS`, `generate`, `--miss-steps`).
- The MetaWorld ports you are extending: `exp/r17_integration/models/pi05/{adapter,provider,serve}.py` and
  `models/groot/{manifest,loader,provider,serve,key_builder}.py`. Both are MetaWorld-only today (metaworld_sawyer,
  one 480² camera, 4-D state/action).
- How DP/Fast-WAM/Cosmos do LIBERO in R17 (`models/dp`, `models/fastwam`, `benchmarks/libero_bridge.py`), and the
  previous π0.5/GR00T LIBERO path (R10/R14/R16: `exp/offline_search/rounds/r16/resident.py`, `RUN_CONFIGS.md` sections
  "π0.5 x LIBERO" and "GR00T x LIBERO", `src/openpi/cache/groot/key_builder.py`, the `cp1_*_pool_16` key builders) to
  find the LIBERO checkpoints, norm stats, cameras and key definition those runs used.

## Deliver
1. `pi05_libero` and `groot_libero` in the R17 formal path, by parameterising the existing π0.5/GR00T packages by
   benchmark (not by copying them): the LIBERO checkpoints those earlier rounds used, two cameras (agentview + wrist),
   8-D state, 7-D action, LIBERO norm stats, the same retrieval key as before (each view's 256 image tokens 4×4
   spatially pooled to 16×2048 = 32768, plus state). Native samplers: π0.5 K10, GR00T K4; reduced K as on MT50.
   Staged adapters with the same stage split as their MT50 versions.
2. `formal_eval --model pi05_libero|groot_libero`: pure-inference official-50 baselines (both suites, 2,000 episodes
   with `--official-50` like the other LIBERO models; spatial + libero_10 only is fine) and library capture.
3. `library_campaign`: B-pool libraries, 500 episodes per suite, full-step teacher, no images, the same layout as
   `/data/r17_library/dp_libero_*`; nested subsets 50..500, fits for C5 and C10, Q3 stall artifacts.
4. `cache_study`: both models in `LIBERO_MODELS`, all eight groups, long commit **C10** for both (π0.5 predicts 10
   actions; GR00T 16, and no NI horizon study exists, so the R16 historical C10 applies), MISS at the native K
   (π0.5 K10, GR00T K4, as the other LIBERO models use their native K). GR00T's endpoint K is fixed at startup
   (`live_k`): keep that working for LIBERO.
5. CPU tests with stubs for the new paths; reuse the existing MT50 tests' patterns.

## Rules
- Only this. No new validation layers, gates, identity/fingerprint checks or admission logic (owner rule: keep it
  simple; a gate that blocks a normal run is a bug).
- **CPU only.** Do not start any GPU process, server, worker fleet or campaign: both GPUs are running experiments.
- No `tether push/pull/expose`; files move by rsync over the machines' public IPs (`direct_transfer.py`). No credentials.
- No git commit/push, no `pkill`, no `rm -rf`; do not touch running processes, `/archive`, `/data/i018`,
  `/home/weiland/idea-sprint`, or any directory of a running campaign under `/data/mw_finetune/eval/cs*`.
- Handback: `exp/r17_integration/handbacks/HANDBACK_LIBERO_VLA.md`, short: what changed (files), what the CPU tests
  cover, the checkpoint/norm paths used and where they came from, and the exact GPU commands in order (library
  capture → finish → pure baselines → cache-study prepare) with a suggested topology from RUN_CONFIGS (π0.5 on the
  4090 R5 MPS B1 pinned to NUMA node 1; GR00T on H100). Note anything you could not verify on CPU.
