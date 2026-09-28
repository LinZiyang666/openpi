# R5 input: what R2–R4 established (2026-09-27 19:3x CDT; R4 closed loop still finishing)

Full reports: `rounds/r02/ANALYSIS.md`, `rounds/r03/ANALYSIS.md`, R4 prior input `rounds/r04/FINDINGS.md`, R4 ledger
`logs/offline_search_exploration.log.md` §10 (every R4 decision and number, in Chinese), R4 hand-backs
`rounds/r04/k{1..10}_*/HANDBACK.md`, `rounds/r04/k8_search_latency/REPORT.md`, `rounds/r04/k9_gpu_retrieval/REPORT.md`.
`rounds/r04/ANALYSIS.md` will be written in parallel with R5 ideation (noise floor, paired tests). Closed loop = LIBERO,
500 paired inits per arm (10 tasks × 50). Run-to-run noise on l10: ≈13 % discordant episodes ⇒ SE of a paired SR
difference ≈ .016; differences below ~3 pp are not resolved by single runs (two stock g500 replicates are running).

## Cost model (owner basis, primary; π0.5 CUDA-graph stages 10.26 / 27.69 / 29.57 ms, shares .152 / .410 / .438)
Per decision (5 control steps): vision decision .152, MISS adds .848, a vision-free ("blind") decision costs 0.
IR = .152·v + .848·m (v = vision share, m = MISS share). Arms that execute L controls per request are normalized per 5
controls. Eager serving costs (K3, no CUDA graph) are a separate basis (`closed_loop/ops/cost_table.json`), never mixed.
Search cost is NOT in IR yet: K8 measured the CPU method at 1.2–2.2 ms per vision decision (≈ .02–.03 IR in the owner
basis); K9 showed that a GPU-resident retrieval captured in the same CUDA graph as stage 1 adds only ≈ .36–.63 ms
(stage 1 graph ≈ 10.2 ms), i.e. ≈ .006–.009 IR, and cuts PCIe traffic per decision from 131 KB to ≈ 1.4 KB.

## R4 closed-loop results (owner basis; SR @ IR)
π0.5 libero_10, 500-episode library:
- stock guard-only MixedJudge g500 .864 @ .238 (likely a high draw: several comparable arms land .83–.85).
- periodic 8 .850 @ .245, periodic 12 .828 @ .206 (on the chord), guard noprog-4 .808 @ .214 (dead).
- blind "look once, act several steps" with K1's dense stuck guard: phase_particles B=2 .850 @ .211, B=1 .838 @ .265,
  B=0 adapter control .842 @ .271 — K1's guard dropped the visual-stillness confirmation and fired stuck ≈2.8×.
- K7 vision-confirmed stuck guard (B=0 bit-identical to stock): B=0 .830 @ .245; **phase B=2 .862 @ .178**; B=1
  .862 @ .226; **anchor_tail (serve steps 5–9 of the vision decision's own synthesized chunk, budget 1) .880 @ .203**
  (K1-guard anchor_tail .878 @ .223 — two runs agree).
π0.5 libero_10, 50-episode library:
- g50 .740 @ .323; periodic 6 .764 @ .280; periodic 5 .792 @ .315; noprog-4 .690 @ .285.
- K7 phase B=2 **.700 @ .268** (loses: per-neighbour continuation is inaccurate in a sparse library);
  **K7 anchor_tail .806 @ .242** (new 50-library frontier point).
π0.5 libero_spatial:
- 500: guard-only .974 @ .197; periodic 12 .976 @ .195; K7 phase B=2 .968 @ .140; **K7 anchor_tail .982 @ .128**
  (pure inference .986–.992 at IR 1).
Pure inference references: l10 L=5 .844 / .848 / .850 (three seeds/runs), spatial L=5 .986 / .990 / .992;
**l10 L=10 (execute the whole 10-step chunk per call) .904 @ IR .5** — execution length alone is worth ≈ +5–6 pp on l10,
so part of anchor_tail's SR likely comes from executing whole chunks, not from caching. K10 (in progress) adds an
opt-in "policy tail": after a MISS, the next decision may execute steps 5–9 of the policy's own chunk without vision.
MISS step reduction (K2 denoising steps instead of 10): SR indistinguishable from K10 everywhere (e.g. l10 pure
inference .850 vs .848, g500 .842, g50 .732, perk5 .790) — **owner ruling: K2 is an orthogonal drop-in, NOT part of the
system; price every MISS at full inference**.
Pending at the time of writing: pure-cache blind arms (l10 / spatial, both scales), control-step library G/GS, GR00T
pure-cache blind (phase / anchor_tail, both suites and scales), wrist-only stage-1 key (guard-only, both suites and
scales), randomized CALL/CACHE identification (K5: 4 arms, l10, both scales), stock g500 replicates, K10 policy tail.

## Engineering facts (R4)
- R4 serving (blind / R4 logs) used a server-wide lock until 15:47 (K6 fixed it; per-decision latency 2.9 s → normal).
- A server process is GIL-bound at ≈400–600 queries/s; per-decision wall time grows linearly with connections.
- The largest CPU retrieval component is PCA of two 32,768-d pooled keys (≈0.7–0.8 ms), library-size independent.
- K9 GPU prototype: float32 top-1 agreement ≥99.90 %; non-step-0 chunk differences ≤2.4e-5; step-0 early distance is
  ill-conditioned (expanded-distance cancellation), largest chunk difference .0125; in-place library append keeps graph
  addresses (library growth possible without re-capture).

## Dead or settled (do not re-propose)
Everything dead in `rounds/r04/FINDINGS.md`; plus: relaxing the no-progress guard (noprog 4) at both scales; blind
stepping with a motion-only stuck guard; per-neighbour phase continuation in the 50-episode library; periodic MISS
beating guard-only at 500; stage-2 prefix packing (bf16 action drift .014, rejected); MISS step reduction as a system
component (owner ruling); generic "stop after N failed calls", unrestricted cross-task retrieval, token alarms (C2).

## Owner rulings in force (protocol §9)
- Codex agents only. The coordinator advances R4 and R5 alone, no pause points (item 13).
- 500-episode library deployable; every conclusion also at 50 episodes; library bytes vs deployed pkl (π0.5 431 /
  1103 MB, GR00T 429 / 1068 MB); label borrowed big-library information.
- Cheaper vision keys allowed; "look once, act several steps" allowed with vision anchors and bounded blind stretches.
- No LIBERO workers on weilandserver (timan107 only). System throughput studies are out of scope, except the pinpoint
  search-latency measurement (item 11).
- **Owner R5 direction (item 10): how should hyperparameters be set? Can they be solved offline — from the h5 demos at
  library-build time or from logged closed-loop data — by leave-one-out (LOTO / LOEO) or closed form, the way the old
  implementation calibrated RIT thresholds by leave-one-task-out and solved LDA fusion weights in closed form, instead of
  hand-tuning or closed-loop sweeps?**
- Owner remarks: retrieval was never optimized and data moves around; a library that fits on the GPU could be compiled
  into the model (CUDA graph) — K9 confirmed the latency side.
