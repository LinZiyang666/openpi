# Resident cache-study lanes (CPU preparation, 2026-10-10)

Each model uses one resident fleet behind the stock proxy for the complete
study. The conductor changes the YAML per episode with suite/group barriers;
replicas are not restarted between groups. Evaluation is official A, 500
episodes per group per suite, spatial and LIBERO-10 only. B500 is exclusively
the sealed library/fit source. DP and Fast-WAM each execute8 groups/8,000 cells;
Cosmos executes4 distinct groups/4,000 cells and reports the four long aliases.

| Lane/model | Residents | Driver | timan107 workers | RAM/VRAM MiB | Recycling | Startup width |
|---|---|---|---|---|---|---|
|4090 / DP|R2/B8/5ms/MPS|GroupCommit|32|2048/576|4|1 (owner's serialized load instruction)|
|H100 / Fast-WAM|R2/B16/5ms/MPS|GroupCommit|48|2048/768|4|2|
|H100 / Cosmos, after Fast-WAM|R2/B1/MPSoff|legacy|8|3072/768|4|1|

All workers restart at most twice after faults. Fast-WAM uses the baseline's
explicit EGL0..7 list; DP/Cosmos preserve the baseline's automatic/null list.
Fast48+DP32 reserves54GiB across timan107's eight GPUs. After Fast closes,
Cosmos8+DP32 reserves24GiB. Fresh launch rechecks RAM and VRAM admission;
clock time alone does not authorize dispatch. No local4090 flock wraps H100.
Q's launcher owns GPU exclusivity, private MPS control guard and cleanup.

Controls are pinned read-only: DP `hs5dp_ctl_142812`; Fast-WAM's complete
`hs6fw_res_152931` uses native runtime `hs6fw_ctl_143332`; Cosmos uses merged K5
`formal_cosmos_libero_official2000_20261010_130639_a8ca2043`. Its outer receipt
still has the historical rc2 FAIL, while the sealed CPU merge, summary and
current driver-completion audit prove2,000 accepted non-error outcomes. The
cache control proof records that exact recovery rather than changing the
original receipt or claiming it was PASS. Changed control/native pins reject.

Ordinary frozen controls predate cache admission. The explicit cache runtime
overlay inherits every existing non-R17 byte from the control, adds the three
read-only original Q3 modules, and overlays the reviewed R17 framework/model
admission hooks. Model-owned source/content validators still certify native
producer equivalence at CPU sealing and live YAML publication. Worker native
simulator/client bytes are inherited exactly; only P6 ops observers change.
`CACHE_CONTROL.json`, `CACHE_RUNTIME_OVERLAY.json`, `CACHE_WORKER_OVERLAY.json`
and immutable model/worker/orchestration seals make the boundary reviewable.

Smoke (explicit GPU commands for the coordinator; not executed by P6):

```bash
bash exp/r17_integration/ops/cache_study/smoke.sh dp --run --run-name csmd1010
bash exp/r17_integration/ops/cache_study/smoke.sh fastwam --run --run-name csmf1010
bash exp/r17_integration/ops/cache_study/smoke.sh cosmos --run --run-name csmc1010
```

Each smoke executes32 episodes:2/group/suite, including Cosmos alias switches.
Inspect `SUMMARY.json` for hits/tails, zero ordinary MISS, native K100/K10/K5
stall calls, correction/gate activity, final server ACKs and resident generation
stability. No observed gate/corrector activity means unexercised; it is not a
fabricated qualification. Budget10–20/10–20/10–15min respectively including
cold load; these cache rates have not been measured on GPU.

Full prepared-lane dispatch, after the coordinator accepts each model smoke:

```bash
.venv/bin/python -u -m exp.r17_integration.ops.cache_lane --lane 4090 --directory /data/mw_finetune/eval/csl10dp --long-dp 10 --long-fast 10 --run
.venv/bin/python -u -m exp.r17_integration.ops.cache_lane --lane h100 --directory /data/mw_finetune/eval/csl10h --long-dp 10 --long-fast 10 --run
```

The H100 lane runs Fast-WAM then Cosmos. These commands reuse CPU-prepared
short names `csl10d/f/c`; they do not build a new server for every YAML.
GPU-first asynchronous finalization releases replicas/proxy/MPS/lock before
closed CPU evidence pulls. The next model may start while the previous evidence
seals; the queue stops on late failure and waits for every final scientific
seal before reporting PASS. Q's rc2 recovery requires all planned non-error
terminal outcomes, resolved retries and confirmed owned worker closure.

L is a single scalar per model: `--long-dp` and `--long-fast` on the lane CLI;
Cosmos is fixed16. Current horizon results leave fallback10 provisional,
without an NI certification. A changed L needs a new CPU prestage directory,
which fits C-specific heads/stall assets on B only; no C10-prefix reuse:

```bash
L_DP=10 L_FAST=10  # coordinator selection after reviewing RESULTS_DP_FASTWAM.md
.venv/bin/python -u -m exp.r17_integration.ops.cache_lane --lane 4090 --directory /data/mw_finetune/eval/csnextdp --prefix csnext --long-dp "$L_DP" --long-fast "$L_FAST" --prepare-only
.venv/bin/python -u -m exp.r17_integration.ops.cache_lane --lane h100 --directory /data/mw_finetune/eval/csnexth --prefix csnext --long-dp "$L_DP" --long-fast "$L_FAST" --prepare-only
# Repeat each command with --run instead of --prepare-only after smoke/admission.
```

L mismatch at dispatch refuses rather than mutating sealed recipes. `--print`
prints preparation argv without launching; preparation never constructs a
CUDA model, daemon, server or worker fleet. Running campaigns are untouched.

Repeat the same full-run command to continue a naturally closed failure. The
queue retains complete accepted successful AND failed outcomes only after
checking attempt, YAML/native binding, contiguous decision rows, execution
cursors and final server ACK. Torn/missing/error episodes are excluded with
`CACHE_RESUME_QUARANTINE.json` evidence and dispatched into a fresh segment;
the original is read-only. Original runtime, workers, YAML bytes, RNG/init
cells and task UIDs stay pinned. Resident lifetime is checked per segment;
a recovery deliberately starts a new fleet, while group hot-switches within
each segment do not. Unclosed fleets or pending evidence refuse continuation.
Zero-acceptance startup faults use Q's guarded immutable retry (at most2).

Planning ranges, not measured cache forecasts: DP180–300min; Fast-WAM180–330;
Cosmos120–260; H100 lane300–590 plus cold load/evidence sealing. The native
control durations were55.66/64.40/123.22min for2,000 cells; scaling by4/4/2
gives223/258/246min reference work. Cache key extraction, rendering, retrieval,
longer failed episodes and gate frequency change throughput; refine these
estimates from the32-episode smokes. Each full campaign has a24h hard bound.
