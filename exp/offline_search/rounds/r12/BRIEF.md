# R12 — "Look half" (看一半): retrieve the cache with one camera

Coordinator brief, 2026-10-03 CDT. It is given to two independent explorers.

**Independence rule (owner):** you must NOT read anything under the other explorer's directory (`exp/offline_search/rounds/r12/astra/` or `exp/offline_search/rounds/r12/opus/`). You must not try to learn their results in any way. Work only from this brief, the shared sources below and your own analysis.

## 1. The owner's goal (faithful paraphrase)
"Look half" means a cache decision that encodes and searches with only ONE of the two cameras. That saves vision cost but lowers retrieval precision.

The goal is to mix look-half "sand" into the cache so that either:
- **(a)** look-half is SR-lossless while IR goes down; or
- **(b)** whatever SR look-half loses can be bought back by the layer-4 knob (extra policy calls) more efficiently than the IR it saved. The final criterion is **lower total IR at matched SR** for the whole system, compared with the same system without look-half.

The owner's own ideas (prompts, not limits — **do not let them constrain you**; you are free to try any direction you believe in):
- Can we tell when look-half is *safe*? For example, a segmentation of the episode by **look-half safety**. This is not the kinematic stages used before; it is a segmentation by how safe it is to look with one camera.
- Does look-half need its own retrained Mahalanobis metric, or other single-view metrics?
- Is there a better search for look-half: other distances, new data processing or representations, using the visual tokens or images directly, …?

## 2. The deployed system you build on (five layers)
1. **Cache:** R4 BlindAWM retrieval synthesis with a 10-control commit and kref 5 or 8. Features are per-view PCA keys of both cameras plus robot state, with a per-task Mahalanobis metric.
2. **Only-no-progress guard:** calls the policy when retrieved-demo progress stalls.
3. **In-library LOEO motion corrector,** with distance-attenuated strength.
4. **IR knob** to close the SR gap between pure cache and pure inference. Two methods are formally in the design:
   - guard-aware **periodic** calls;
   - **distance-threshold** calls (they underspend in closed loop because closed-loop states sit nearer the library than whole-episode-out calibration predicts).
5. **Per-task partition:** per-task retrieval scope, metric and corrector heads, already inside layers 1–3.

Look-half is a **cache-layer** (layer 1) change. Prefer task-agnostic mechanisms. If something needs task-indexed parameters, say so explicitly: it then belongs to layer 5.

**Code:**
- `exp/offline_search/rounds/r10/recipe/` (`R10Recipe`, layers 1–3+5, builder in its README);
- `exp/offline_search/rounds/r11/knob/` (`R11Knob`, layer 4);
- the closed-loop plugin in `exp/offline_search/closed_loop/{plugin.py, blind.py}`;
- the cost ledger in `plugin.py` (look cost ~L1134).

**Reports:**
- `rounds/r10/REPORT.md`, `rounds/r11/REPORT.md`;
- `rounds/r11/opus/ANALYSIS.md` and `rounds/r11/astra/ANALYSIS.md`. These are R11 deliverables and may be read by both of you.

## 3. Costs (owner IR)
- **π0.5:** IR = .152·v + .848·m on the owner basis. Per look, full two-camera vision = .152 of a policy call.
  - **Measured look prices on the per-request path** (R7, local RTX 4090, batch 1; `rounds/r07/ANALYSIS.md` Addendum A):
    - full look .1656;
    - **wrist-only look .0646**;
    - "missing-camera completion" .0502 (if a half look is followed by a policy call, the other camera must be encoded).
  - The R7 ledger used the assumed prices .055198 / .049890.
  - The third-person-only price was **not** measured.
- **GR00T:** basis .148. Half-look prices were **not** measured.
- If your design depends on a price that was not measured, say so. The coordinator can measure it on the local 4090.
- The vision share of IR is small: a look every other slot costs about .076 IR. State clearly how much IR look-half can save at best in each cell, with and without layer-4 calls.

## 4. Prior work on looking with fewer cameras (read; do not assume it is the answer)
- **R7 "stage wrist" (SW)**, π0.5 only: wrist-only looks in interior (easy) stages.
  - SR change −0.05 pp [−1.40, +1.28].
  - The IR saving was small at measured prices: .0764 → .0678.
  - Sources: `rounds/r07/ANALYSIS.md` (SW sections, Addendum A); code in `rounds/r07/c2_wrist/`, `rounds/r07/c1_follow/`.
- **R9 look saving on 500-demo libraries** (fable round 8 PaceWrist + stage-gated follow; `rounds/r09/REPORT.md` and `rounds/r09/explore_fable/round8/`):
  - π0.5 lost SR: L10-500 .810 vs .842 cache; Sp-500 −2.4 pp.
  - GR00T skipping looks / wrist-only lost 9–15 pp. Its first guard call came about 5 steps later, and blind blocks hurt.
- **Library views:** `key_v0` is the third-person (agentview) camera and `key_v1` is the wrist camera. The R7 wrist path NaN-ed v0. Verify this yourself.

## 5. Data and the hard data rules
**Allowed, and the main material — the B-pool libraries** (non-test): `/home/weiland/trace_runs/offline_search_store/library/<model>_<suite>/{bpool_cs (π0.5 500 eps), bpool_all (GR00T 500 eps), current (curated 50)}`, plus R10 nested subsets via `rounds/r10/data.py` `SubsetLibrary`. Per row they hold:

| array | content |
|---|---|
| `key_v0` / `key_v1` | 32768-d per view |
| `rs` | robot state |
| `action` | the policy's chunk |
| `progress`, `step`, `episode`, `success` | per-row metadata |
| `tok/v0`, `tok/v1` | visual tokens, 256×2048 fp16 per view |
| `tok/img0`, `tok/img1` | raw 224×224 RGB per view |

Library rows are pure-policy rollouts on B-pool inits, including failed episodes.

**Also allowed:**
- R6 B-val recordings (`/home/weiland/trace_runs/os_closed_loop/r06_c_cal/`);
- R11 dev-set roots on B-pool inits outside the served subsets: `r11_dev_current`, `r11_dev_size50`, `r11_dev_size200`, `r11_devknob_50` (decision logs, journals, summaries).

**FORBIDDEN for fitting, calibration, selection or design evidence:** anything recorded on the official test set A. That includes:
- `/home/weiland/trace_runs/offline_search_store/queries/*`: these are test-A closed-loop recordings from the trace_dual line, inits 0–49 of the official set;
- every test-A closed-loop root under `os_closed_loop/` (R1–R11 test arms: journals, decision logs, summaries).

You may read REPORT / ANALYSIS markdown, which holds aggregates only.

**Closed-loop data on non-test inits.** Library replays cannot see how states drift when the cache drives the robot (R11's main lesson). If you need closed-loop recordings — for example, per-look keys of both cameras on cache-driven trajectories, or shadow policy chunks — write a request to `REQUESTS.md` in your dir early: what to record, which cells and sizes, how many episodes. The coordinator can record them on **B-pool dev inits** (the non-test dev-set machinery) and will tell you where they land. You cannot launch anything yourself.

## 6. Deliverables (in your own dir)
1. `OFFLINE.md` + code. Your analyses and evidence, library-only or B-pool dev only.
2. `SPEC.md`. Implementation-ready design(s) for sol: exact algorithm, what is fitted, and how — library-only.
3. `PREDICTION.md`, timestamped before any closed-loop result of your design. Predicted IR saving, SR change and the net "matched-SR total IR" effect per cell.
4. A proposed experiment plan: a dev pilot first (B-pool), then a frozen test-A grid. Include matched-SR comparisons with the layer-4 knob if you go for goal (b).
5. `REPORT.md` (owner section in plain Chinese with no codenames first, then technical) and `HANDBACK.md`.
6. `REQUESTS.md` (optional, early): closed-loop dev recordings you need.

## 7. Operating rules
- **Writes:** only in your own dir. Never read the other explorer's R12 dir.
- **No launches:** no h100 / timan / local server, chain, worker or sync.
- **Prohibited commands:** no git, no `rm -rf`, no `pkill -f`.
- **GPU:** this machine's RTX 4090 may be used for offline feature computation (encoders, distances on tokens / images). Keep it bounded (< 20 GB VRAM) and check `nvidia-smi` first; no closed-loop serving.
- **Python:** `.venv/bin/python`, `PYTHONPATH=.:src`, single-threaded BLAS unless you deliberately parallelize within your CPU set. CPU sets:
  - astra: `taskset -c 10-21,54-65`;
  - opus: `taskset -c 2-9,46-53`;
  - 38–43 / 82–87 belong to another project; do not use them.
- **Time:** take the time you need to explore well (several hours is fine). Hand back with frozen proposals when you are confident. If a direction looks useless, say so and move on.
