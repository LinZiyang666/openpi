# R4 coding brief (common to the four codex coding agents K1–K4)

Read first: `rounds/r04/SELECTION.md` (the approved list; its table is in Chinese — this brief and your task prompt
give you everything in English), `rounds/r04/FINDINGS.md`, the ideation reports `rounds/r04/ideation_A/REPORT.md`
(look once, act several steps — §3 is the server specification), `ideation_B/REPORT.md` (cost engine),
`ideation_C/REPORT.md` and `REPORT_2.md`, `rounds/r03/CODING_BRIEF.md` (method rules still in force), 
`exp/offline_search/closed_loop/README.md`, `exp/offline_search/harness/README.md`.

## File ownership (parallel agents: never edit a file you do not own)
| owner | files |
|---|---|
| K1 | `exp/offline_search/rounds/r04/k1_blind/**` (new) |
| K2 | `closed_loop/plugin.py`, `closed_loop/selftest.py`, `closed_loop/verify_logs.py`, `closed_loop/replay_client.py`, `closed_loop/probe.py`, new `closed_loop/blind.py` (or similar), `rounds/r04/k2_serving/**` |
| K3 | `closed_loop/ops/start_server.sh`, `closed_loop/serve_pi05.py`, `closed_loop/serve_groot.py`, new `closed_loop/stage_overrides.py`, new `closed_loop/ops/cost_table.json`, `rounds/r04/k3_cost/**` |
| K4 | `closed_loop/ops/emit_arms.py`, `ops/chain.sh`, `ops/pilot.sh`, `ops/collect.py`, `ops/kpi.py`, `ops/remote/*`, `rounds/r04/k4_eval/**` |
Everyone may append a section at the END of `closed_loop/README.md` (one atomic write: re-read, write temp, rename).
Nobody edits `src/`, `harness/`, `profile/`, `rounds/r01..r03/` (subclass or wrap instead).

## Live-system rule
The coordinator runs closed-loop arms on this same code while you work (servers import `closed_loop/plugin.py` and
friends at start; `chain.sh`, `start_server.sh`, `collect.py`, `kpi.py` are re-read per arm). Therefore: develop in a
copy (e.g. `rounds/r04/<your dir>/dev/`), keep every existing behaviour byte-identical when your new flags/fields are
absent, prove it (existing selftests / a before-after diff), and install each final file in ONE atomic step (write a
temp file in the same directory, then `mv`). Never leave a shared file half-edited. Remote scripts (`ops/remote/*`) are
pushed to timan107 only by the coordinator.

## Method <-> plugin contract for vision-free ("blind") decisions (K1 implements the method side, K2 the plugin side)
- A method that supports blind decisions implements `blind_step(bq) -> BlindResult | LookReason` (import both types
  from `exp.offline_search.closed_loop.blind`; K2 creates that module first thing, with these dataclasses:
  `LookReason(code: int, name: str)`; `BlindResult(action: np.ndarray (H,32) float32 normalized like library actions,
  rows: np.ndarray int64, weights: np.ndarray float32, library: str, extras: dict[str, float])`).
- `bq` (BlindQueryView) exposes only vision-free fields: `step`, `task_id`, `episode`, `rs` (bit-identical to the
  robot_state key the key builder would produce for this decision — K2's responsibility), `raw_state`, `prev_hit`,
  `prev_a_exec`, `hist_a_exec`, `hist_hit`, `hist_rs`, `hist_has_vision`, `blind_age` (consecutive blind decisions
  before this one). No keys, tokens or images.
- The method remembers its anchor (all 16 rows, weights, per-member phase) from its own last vision `query()`; per-episode
  state is reset in `reset()`; an anchor is invalid after a MISS (`bq.prev_hit is False`) and at the episode's first
  decision. Look-reason codes: 1 budget, 2 gripper event ahead, 3 near terminal, 4 low motion, 5 displacement residual,
  6 lifecycle (first decision / after MISS / invalid anchor), 7 periodic MISS due, 8 other.
- Plugin order per decision: (1) if a periodic MISS is due by the GLOBAL decision counter -> vision path; (2) else call
  `method.blind_step(bq)`; LookReason -> normal vision path (stage 1, retrieval, judge; the judge may MISS); BlindResult
  -> serve it without entering the interceptor (no stage 1/2/3), commit history once, broadcast once. A blind decision
  is never a MISS.
- Guards under blindness: stock MixedJudge assumes a visual observation every decision. K1 provides a gap-aware guard
  (`noprog_span`: progress measured only on vision anchors, accumulated over elapsed decisions) and uses dense
  proprioception for motion; K2 must pass `hist_has_vision` so the method can tell gaps apart.

## Decision-log fields (K2 emits, K4 consumes; add to every `ev: dec` row)
`vision` (bool: stage 1 ran), `src` ("cache" | "cache_blind" | "policy"), `hit` (bool), `blind_age` (int),
`look_reason` (int or null), `miss_k` (denoising steps used on a MISS, null on HIT), `s1_ms`, `s23_ms` (null when not
run), `served_head` (the executed [:5,:7] block as a list, on EVERY decision, HIT or MISS), plus existing fields.
Startup row adds `stage1_mode` (K3: e.g. "full", "dummy_cached", "wrist_only") and `miss_steps` (K3).

## Cost ledger (K4 implements in collect.py / kpi.py; constants from `closed_loop/ops/cost_table.json` written by K3,
falling back to the owner constants .152/.410/.438 for π0.5)
Per five-control-step decision: vision decision costs s1 (full or the measured cheaper stage-1 variant), blind decision
costs 0, MISS additionally costs s2 + s3·K/10 (π0.5; GR00T from its own table). IR = total / (N_decisions · full cost).
Arms that execute L control steps per request report IR per five control steps (scale by 5/L) and controls/episode.
Always report realized vision share v, MISS share m, and the ledger inputs.

## Rules
- Vision anchor mandatory in every episode's control loop; blind stretches bounded (owner ruling). No external models.
- Valid dims: actions [:, :7], executed steps [:5]; π0.5 robot_state [:8]; GR00T per its own convention (gripper sign
  reversed: see ideation A §3.4).
- Every conclusion at the 50- and the 500-episode library, with library bytes vs deployed pkl (π0.5 431/1103 MB, GR00T
  429/1068 MB); label borrowed big-library information.
- CPU budget: prefix every python command with `taskset -c <your range>`, BLAS threads 1, processes ≤ threads in range.
  Never CPUs 38-43, 82-87.
- GPU (only agents whose prompt allows it): the 4090 is shared with another project's training (priority) and with the
  coordinator's closed-loop servers. Check `nvidia-smi` free memory before each GPU step; use at most the amount your
  prompt allows; stop at once on any OOM; release the GPU after each measurement. Prefer STAGE1_ONLY=1 servers (≈2.2 GB)
  whenever the test does not need stage 2/3. Ports: only free ports in 23180–23189 (scan `ss -ltnH` first); never touch
  ports 23150–23159 or any process/tmux session you did not start.
- No timan107, no LIBERO workers, no closed-loop chains (the coordinator runs them). No git. No `rm -rf`. Never
  `pkill -f`; kill only processes you started, by PID. Do not read `tests/review_tests/`.

## Hand-back (final message and `rounds/r04/<your dir>/HANDBACK.md`)
Files changed/added (with install times), how each new feature is switched on (exact flags / arm-spec fields), tests run
and their results (numbers), what the coordinator must do next (e.g. push remote scripts, prefit commands), caveats.
