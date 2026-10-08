# R15: (A) per-task descent at library sizes 100 / 200 / 500, (B) a light image query-key model that skips the vision tower

Coordinator brief, 2026-10-05 CDT. Track A is built by sol and analysed under the frozen R14 protocol. Track B is explored and designed by astra, then built by sol.

## 1. The owner's requests (10-05)
- **Track A:** "下一轮我们补上100集和200和500集沿任务分治的实验": repeat the R14 per-task knob descent (layer 5) at library sizes 100, 200 and 500. R14 did size 50 only.
- **Track B (from the professor):** we already learn the Mahalanobis weights of the retrieval metric from data, but the key still needs the VLA vision tower at every look. Can a learned, very light query-key model work **directly on the images**, so a look never runs the vision tower? Train it the same way as the metric (action-supervised, per task, from the library), only with images as input. A better training method is also allowed.
  - **Hard data rule (owner):** train only on that cell's build-cache library. The 50-episode library may only use those 50 episodes; the 100-episode library only those 100; and so on.
  - **Worth it only if** it is much cheaper than the current agent and still accurate.

## 2. The deployed system (five layers)
1. **Cache:** R4 BlindAWM retrieval synthesis. Key = per-camera PCA-64 of the pooled vision-tower tokens (both cameras) + robot state; per-task action-supervised Mahalanobis metric (`rounds/r02/g1_awm/awm.py` docstring: S_w over action-similar cross-episode pairs, M = (S_w + λI)^-1, W = chol(M)); 16-NN kernel synthesis; commits 10 controls per decision.
2. **Only-no-progress guard.**
3. **In-library LOEO motion corrector**, strength attenuated by the retrieval distance ratio r.
4. **IR knob:** periodic gap cap (R13 settled periodic over distance).
5. **Per-task partition:** at deployment, per-task knob descent (R14).

Code: `rounds/r10/recipe/` (R10Recipe, layers 1–3), `rounds/r11/knob/` (R11Knob), `rounds/r14/descent/` (descent roots, TaskDescentKnob, test-A emitter), `rounds/r14/descent_astra/` (frozen protocol + analysis), `closed_loop/{plugin.py, blind.py, look_recording.py}`.

**Owner IR:** IR = (c_v·V + c_m·M) / N with π0.5 c_v = .152, c_m = .848; GR00T c_v = .148, c_m = .852. V = looks that run the vision tower, M = policy calls. Pure inference ≈ .50, pure cache ≈ .076.

## 3. Data discipline (all rounds)
- Fit, train, calibrate and select only on B-pool data: the cell's library (for track B: training data = that library only) and B-dev roots (B inits outside the served subset).
- Test A (official 500 inits) is measure-only, run once on frozen configurations. Never read test-A data (R1–R14 test roots, `offline_search_store/queries/*`) for design.
- Library subsets are nested: `rounds/r10/subsets/<model>_<suite>_{50,100,200,500}.json` (seed 20261002). Size 500 = the whole B pool, so no B init is unseen at 500.

## 4. Track A: descent at 100 / 200 / 500
- **Cells:** {π0.5, GR00T} × {L10, spatial} × {100, 200, 500} = 12 cells.
- **Protocol:** R14's frozen `descent_astra/PROTOCOL.md` verbatim (30 B inits per task, primary tolerance 5 pp, first-failure stop, order c10 → ir45 → ir40 → ir32 → ir25 → off; uniform level by the same rule on the cell's 300 episodes). Coordinator decisions in `COORDINATOR_DECISIONS.md`.
- **Fits:** each cell's own library only (R10Recipe layers 1–3 + R11 periodic caps). Fit any missing size-100 artifacts with the unchanged recipes.
- **Dev inits:** per model × suite, the B inits outside that model's 200-subset (30 per task), so they are unseen at 100 and 200. At 500 the same inits are served with **leave-own-episode-out** (the query init's own library episode is excluded from every neighbour lookup; fits unchanged, disclosed).
- **Arms:** off, periodic ir25 / ir32 / ir40 / ir45 per cell; pure C10 once per model × suite if it is provably library-independent (else per cell).
- **Test A:** per cell, uniform and per-task arms × 500 episodes; reuse R14's test-A pure C10 arms as the reference if library-independent.
- **Hosts:** GR00T on the local 4090 (GR00T has a 4090-vs-H100 host effect; R14's GR00T references ran on the 4090), π0.5 on h100.

## 5. Track B: light image query-key model
Questions, in order:
1. **Data:** π0.5 libraries store both camera images per row (`library/<m>_<s>/bpool_cs/tok/img{0,1}.npy`, 224², uint8, aligned with `tok/rows.npy`). GR00T libraries store tokens but no RGB. Find GR00T images (source traces `trace_runs/dual_20260923/runs/tr_groot_*`) or a faithful way to regenerate them (e.g. deterministic replay of the library episodes, verified row by row against stored robot state), or say it cannot be done.
2. **Method:** a light encoder from raw images (+ free robot state) to a retrieval code, trained per cell from that library only, with the AWM supervision (action-similar cross-episode pairs per task) or a better objective. Consider at least: (a) symmetric light encoder for library and query; (b) asymmetric "query-key": library keys may be computed offline by anything (including the vision tower), the online query encoder must be light, e.g. distilled into the library's existing AWM code space; (c) anything better. Prior evidence to respect: a frozen CLIP global key with our metric lost 3–30 pp because one global vector cannot tell phases apart within a task (R8 ablation); single-camera keys hurt GR00T L10.
3. **Cost:** measure the light encoder's per-look latency (batch 1, both cameras, same GPU as the policy; also CPU) against the vision tower and the full policy call, and convert to an IR coefficient ε. With a light key, a policy call still runs the vision tower, so IR_new = (ε·V + (c_v + c_m)·M) / N.
4. **Accuracy:** offline first (retrieval quality on B-dev looks against the policy's own actions, neighbour agreement with the current key), then closed loop on B-dev, then a frozen test-A confirmation.
5. **Worth-it gates (the professor's condition, to be frozen by astra before any closed-loop outcome):** per-look cost far below the vision tower (target ε ≤ c_v / 10), and SR at matched knob settings not meaningfully worse than the current system, with a large total IR reduction.

All other layers (guard, corrector, knob, descent) consume the new code unchanged where possible; if a layer needs a refit on the new code, refit it from the same library with its unchanged recipe.

## 6. Roles
- **sol:** track A roots now; track B implementation after astra's design.
- **astra:** track B exploration, design, frozen gates, later analysis. May run CPU prototypes, and small GPU prototypes on the local 4090 under a VRAM cap once the coordinator allows.
- **Coordinator:** all runs, GPU negotiation with the 沙堆实验 session, decisions, the report.
- No agent launches servers or chains, touches remote hosts, or waits on runs.
