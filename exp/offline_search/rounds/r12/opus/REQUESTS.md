# R12 opus — requests to the coordinator

First written 2026-10-03 ~11:45 CDT; revised 12:10 CDT (Request 0 added after the tower microbenchmark). All requests
are for the local 4090 or **B-pool dev inits only** (the R11 dev-set machinery, `closed_loop/devset.py`). Nothing
here touches test set A. I cannot launch anything myself.

## Request 0 (highest priority, decides whether look-half is worth building): stage-1 latency with BATCHED cameras

**Why.** pi0.5 stage 1 embeds the cameras one at a time (`PI0Pytorch.embed_prefix` loops over the three image slots,
including the always-masked third slot). The R7 Addendum A prices (full .152, wrist .0646, completion .0502) are
therefore per-camera sequential costs. My tower-only microbenchmark on the idle local 4090 (HF SigLIP-so400m +
projector rebuilt from the pi05_libero checkpoint, bf16 eager, batch-1 request; `out/bench_tower.json`):

| images in one tower call | 1 | 2 | 3 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| median ms | 23.0 | 23.5 | 22.9 | 25.3 | 25.0 | 34.3 |

Sequential calls: 2 cameras 46.8 ms, 3 cameras 69.0 ms (the stock pattern; R7 measured 62.1 ms for stock stage 1).
The tower is launch-bound at batch 1, so **two cameras in one call cost about the same as one camera.** If that holds
on the serving path, a full two-camera look costs about what a wrist-only look costs (~.065 instead of .152), every
policy call's stage 1 gets equally cheaper, and look-half saves essentially nothing on pi0.5. Keys from a batched
two-camera call differ from per-camera keys by 0.15 % (relative L2), the same size as the difference between my
recomputed keys and the stored library keys (`out/batch_parity.json`).

**What to measure (pi0.5, per-request path, batch-1 request, same convention as R7 Addendum A):**
1. stock stage 1 (three sequential image embeds);
2. stage 1 with the two real cameras embedded in ONE call and the masked third slot taken from the cached dummy
   embedding (R4's `Pi05Override._dummy` already caches it);
3. wrist-only stage 1 and the completion (to confirm R7's 26.4 / 20.5 ms on today's stack);
4. **GR00T**: `GrootStagedRunner.run_stage1` already calls `extract_feature` once on both images
   (`src/openpi/cache/groot/staged.py`), so a one-camera look probably costs the same as a two-camera look. Please
   time stage 1 with 2 images vs 1 image. If they are within ~10 %, GR00T look-half has no price to save.

## Request 1 (only if look-half is pursued after Request 0): closed-loop keys on cache-driven trajectories

**Why.** My closed-loop check used the R6 B-val recordings (old R6 cache, 10 episodes per cell). On the 500-episode
cells, whose recording library equals my evaluation library, the one-camera agreement statistics on closed-loop
states match library replay (OFFLINE §8). A recording of the deployed five-layer system would confirm the same for
the 50-episode cells.

**What to run.** The deployed R10Recipe (layers 1-3+5), unchanged two-camera serving, `--os-log-inputs`, knob off:
pi05 l10 50 (100 episodes), pi05 spatial 50 (50), pi05 l10 500 (50). Optional: the R11 periodic knob at .32 on
pi05 l10 50 (50 episodes). The dev pilot of the SPEC arms (SPEC §4) can carry `--os-log-inputs` instead.

## Request 2 (prices still unmeasured, only needed for the look-half IR claim)
- pi0.5 third-person-only look and its completion (I do not use it; wrist is the better single camera on pi0.5).
- GR00T one-camera look and completion (see Request 0 item 4).

## Status (12:30 CDT)
- Request 1 was recorded by the coordinator (`r12_devlooks_*`). I read `50L` (pi0.5 arm) and `50S` only after
  PREDICTION.md was frozen; results in OFFLINE §11 and PREDICTION Addendum A. The 200 / 400 / knob roots were still
  running and were not used.
- Request 2: measured (`prep/HANDBACK.md`). GR00T one-camera ≈ full, as predicted.
- **Request 0 is still the open, decisive item**: pi0.5 stage 1 with base + wrist embedded in one call (dummy cached),
  timed on an idle GPU. An untested convenience draft is `tools/bench_stage1.py` (model-level, eager, R7 observation
  convention; it was not run because the GPU was busy with the recording servers).

## Status after DIRECTIVE_2 (13:12 CDT)
- Request 0 (batched stage-1 price) is withdrawn: the owner ruled batching/launch tricks are not methods.
- Read after DIRECTIVE_2: the 400 and knob roots (and 200 / 50S). All used only as B-pool dev evidence (OFFLINE §12).
- New request (optional): the Phase-A own-loop dev pilot of SPEC §7 with look recording on (it is the only way to
  measure the guard in the one-camera system's own closed loop).

## Guard task (16:15 CDT)
- Dev pilot request: GUARD.md §8 (deployed / H2 guard swap / M2, 4 cells knob off + pi05 L10-50 knob .32, 100
  episodes per arm, look recording on). This supersedes the M1 pilot of SPEC §7.
