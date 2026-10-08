# R12 directive 2 — owner ruling and new direction (2026-10-03 12:5x CDT)

This applies to both explorers. The independence rule still holds: never read the other explorer's directory.

## 1. Owner ruling on costs
Deployment means **centralized serving where the GPU is already saturated**. Any request-level latency trick only works when a single request leaves the GPU idle. Examples:
- putting several images into one encoder call;
- caching or skipping launch overhead;
- re-ordering encoder calls.

These are at most **technical optimizations**. They are **not methods**, and their "savings" do not count.

**Owner, verbatim: "必须是方法而不是代码层面上的效率优化"** — it must be a *method*, not a code-level efficiency optimization. Out of scope, as proposals and as claimed savings:
- batching or merging encoder calls;
- caching or skipping redundant encodes, including a masked placeholder slot;
- kernel or launch tricks, pipelining, re-ordering;
- precision changes;
- any other implementation speed-up of the same computation.

A valid look-half method changes *what is computed or decided*: which camera is encoded, when, and how retrieval, synthesis and correction use it. Its saving comes from not encoding a camera, priced by compute share.

**Price look-half by compute share.** Encoding one camera costs half of the full two-camera vision price; completing the other camera costs the other half.

| model | full look | one camera | completion of the other camera |
|---|---|---|---|
| π0.5 | .152 | .076 | .076 |
| GR00T | .148 | .074 | .074 |

- A policy call still costs the full price: vision + .848 for π0.5, vision + .852 for GR00T.
- The batch-1 latency table in `rounds/r12/prep/HANDBACK.md`, and R7's .0646 / .0502, are **not** the pricing basis any more.
- Consequence: **GR00T look-half now has the same relative saving as π0.5.** Both models are in scope.

**GR00T caveat (from sol's prep).** The deployed GR00T path encodes both cameras together. A camera encoded alone is not bit-identical to its slice of the joint encoding (max embedding delta about 1–4). The library keys come from the joint path. A GR00T one-camera method must handle or measure this mismatch, for example with a single-camera re-encode of the library or a robust metric.

## 2. New direction (mandatory)
Research **look-half as a method in the cache layer**. Each method decides when a cache decision can be made with one camera, and how to retrieve, synthesize and correct well with one camera. The goal is unchanged:
- **(a)** look-half is SR-lossless with lower IR; or
- **(b)** its SR loss is bought back by the layer-4 knob more cheaply than the IR saved, giving lower total IR at matched SR.

Re-evaluate everything under the compute-share prices above.

Explore freely. Possible threads, which do not limit you:
- safety segmentation or gating learned from the library and the dev recordings;
- camera memory, stale-view reuse, motion or displacement cues;
- single-view metrics and representations;
- different camera choices per situation;
- interaction with the guard (layer 2), the corrector (layer 3) and the knob (layer 4).

## 3. Use the closed-loop B-dev recordings now
All are under `/home/weiland/trace_runs/os_closed_loop/`. Read them with `exp.offline_search.closed_loop.look_recording.read_looks`. Every look records both views' keys, robot state, ids, step, top-k rows, weights and distances, served chunks and guard state.

| root | cells | extra fields | status |
|---|---|---|---|
| `r12_devlooks_50L` | π0.5 + GR00T L10-50 | tokens, 56-px images, shadow policy chunk | complete |
| `r12_devlooks_400` | π0.5 + GR00T L10-400 | 56-px images | complete |
| `r12_devlooks_knob` | π0.5 + GR00T L10-50, periodic knob .32 | — | recording now |

Spatial cells and size 200 are pending.

The R11 lesson applies: closed-loop states differ from library replay. Validate on these recordings, and label any post-freeze calibration as such.

## 4. Deliverables (update in your own dir)
- `SPEC.md`: an implementation-ready method or methods, library-fitted, plus any dev-calibrated parameters labelled as such.
- `PREDICTION.md`: a new **timestamped** section with per-cell IR saving, SR change and the matched-SR total-IR effect under the new prices. Leave the earlier frozen sections untouched.
- `OFFLINE.md`: evidence, including the closed-loop dev validation.
- `EXPERIMENTS.md` (or a `SPEC` section): a dev pilot first, then a test-A grid.
- `REPORT.md`: owner section first, in plain Chinese with no codenames.
- `HANDBACK.md`.

Same data rules and operating rules as `BRIEF.md` §5–§7: B-pool only, no test-A data, no launches, no git, no `rm -rf`, no `pkill -f`, writes only in your dir. The local GPU is shared with recording servers, so check `nvidia-smi` and stay ≤ 10 GB.
