# S → R: continuous performance recorder

Recorder started 2026-10-10 18:09 UTC; current PID **1965274**. Stop only this recorder:
`kill -TERM 1965274`. Launcher: `exp/r17_integration/ops/perf_recorder.sh`.
Implementation: `exp/r17_integration/ops/perf_recorder.py`.
Output: `/data/mw_finetune/perf/2026-10-10/perf.jsonl`, rolling `summary.md` every five minutes;
launch log `/data/mw_finetune/perf/recorder.log`, PID file `/data/mw_finetune/perf/recorder.pid`.

Read-only on all campaigns; no model/GPU imports, campaign signals, configuration changes,
or workload launches. One batched H100 exec per sample, one persistent timan107
read-only node collector, and timan108 in the spare exec slot (five-minute offline
backoff). At most two tether invocations per sample/30 seconds, without retries.
GPU telemetry uses nvidia-smi only. Five CPU-only correctness tests passed (including restart-rate exclusion and matched client/server timing).

Initial inspection: both HS3 campaigns have live local drivers despite initial
`RECEIPT.json: PARTIAL` (that receipt is written at startup). H100 had two Fast-WAM
residents, 99% utilization, 255 W, about 53.7 GiB reported memory at 18:04 UTC.
Only an instantaneous observation; the >=20-minute result will follow below.
The current batched per-step logs have `queue_wait_ms: null` and
`queue_wait_status: unsupported`; batched `service_seconds` includes stage waiting.
Batch composition ledgers lack explicit enqueue/flush reasons. Recorder preserves
these limits: queue/compute remain unknown; under-cap flushes are expiry proxies.

Recorder-only restart at 18:11 UTC added per-campaign worker CPU/RSS/census and fixed rolling-table formatting. Prior samples remain in the same JSONL; campaign processes were untouched.

Current recorder PID **1965274** (recorder-only restart 18:16 UTC): now also records local driver thread wait channels/I/O counters and worker fleet failure/recycle counts. Stop `kill -TERM 1965274`. Initial inspection found both HS3 driver threads in `jbd2_log_wait_commit`; the source holds the scheduler lock across one fsync per per-step row. No campaign changed.

## Provisional finding, 18:19 UTC (9.1-minute span; 20-minute result pending)

The unexpected bottleneck is **local driver durability/dispatch**, before adding
workers or changing server caps. Both HS3 drivers were caught in
`jbd2_log_wait_commit` in **5/5** instrumented snapshots each. Source:
`ops/formal_driver.py:128` holds `self._scheduler._lock` while
`super().handle_result` flushes per-step rows; `append` flushes+fsyncs **each row**
(`formal_driver.py:114`). This blocks both result handling and new pulls on that
campaign. Local I/O PSI is around 25–33%, while worker-node I/O PSI is near zero.

Matched local ledger timing (completion time minus assignment time minus worker
`duration_s`, so no remote clock synchronization assumption):

| Campaign | report/transport/durability delay mean / p50 s | completed result → next assignment mean / p50 s |
|---|---:|---:|
| Fast-WAM HS3 | 18.9 / 12.2 | 18.6 / 15.2 |
| DP HS3 | 22.6 / 19.1 | 28.8 / 26.1 |
| Cosmos resume | 2.7 / 2.2 | 3.3 / 1.1 |

This is evidence of control-plane starvation; it is not pure fsync timing.
Client inference means minus per-step server WebSocket inference means are
computed on the **same completed episodes** by the offline analyzer, avoiding
biased subtraction of unrelated live-window samples. Intermediate artifact:
`/data/mw_finetune/perf/2026-10-10/headroom.json` (will be refreshed at >=20 min).
The current recorder PID is **1854373**, stop `kill -TERM 1965274`.

## External campaign change, observed 18:20 UTC

Both original HS3 receipts now have `all_owned_closed: true`, PARTIAL and
`TimeoutError('campaign stopped or hard deadline exceeded')`. S issued **no
campaign signals**. Their useful recorder intervals total about7.5 minutes.
`hs4_fast_b8w5_h100_k10_c5_control_20261010_131810` appeared automatically:
H100,2 replicas,B8/5ms,timan107 x48,2048MiB cooperative worker reservation.
Cosmos resume remains live. The final report will state observed durations and
will not imply20 minutes of HS3 data after an externally closed campaign.

## Final headroom evidence: 21 usable minutes, 2026-10-10 18:31 UTC

Continuous recorder remains alive as PID **1965274**. Stop only it with `kill -TERM 1965274`. The final recorder-only reload preserves the rolling history and fixes ended-campaign rate denominators. No campaign launch, signal, reconfiguration, GPU computation, or forbidden-node access was performed by S. Five CPU-only tests passed; no recorder errors.

Evidence window: 2026-10-10T18:09:47.113058+00:00 → 2026-10-10T18:31:21.283613+00:00; span **21.57 min**, valid sample intervals **21.00 min**, excluding every recorder process's bootstrap. HS3 campaigns were closed externally, leaving **7.5 usable minutes each**; HS4 has12 min and Cosmos21 min. Rates use each campaign's active observed interval.

Raw: `/data/mw_finetune/perf/2026-10-10/perf.jsonl`. Rolling: `summary.md` in the same directory. Fixed evidence snapshot: `headroom.json` in that directory. Offline reproduction (no probes):

```bash
python3 exp/r17_integration/ops/perf_recorder.py --analyze /data/mw_finetune/perf/2026-10-10/perf.jsonl --analysis-output /data/mw_finetune/perf/2026-10-10/headroom.json
```

| Campaign | usable min | accepted last | ep/min | calls/s | GPU util % / W | S3 B p50/p90 / cap | queue p50 ms | mean client = server + protocol ms | primary bottleneck |
|---|---:|---:|---:|---:|---|---|---:|---|---|
| HS3 Fast-WAM,32 workers (closed) | 7.5 | 455 | 18.40 | 7.46 | 99.7 / 436.9 | 3/8 / 8 | N/A | 2035 = 1973 + 62 | Driver fsync/dispatch; S3 queue+compute |
| HS3 DP,24 workers (closed) | 7.5 | 410 | 17.33 | 6.58 | 52.7 / 154.5 | 1/5 / 16 | N/A | 678 = 540 + 138 | Driver fsync/dispatch; S3 queue+compute |
| Cosmos baseline resume,8 workers | 21.0 | 233 | 10.48 | 2.98 | 98.0 / 448.6 | B1 serial / replica | 776 | 1457 = 1401 + 56 | Resident B1 queue + inference/sim |
| HS4 Fast-WAM,48 workers (external replacement) | 12.0 | 253 | 21.08 | 8.12 | 98.8 / 458.9 | 5/8 / 8 | N/A | 3023 = 2959 + 64 | Driver fsync/dispatch; S3 queue+compute |

H100 utilization/power are **shared whole-board** observations; Fast-WAM and Cosmos watts cannot be attributed separately. The H100 is GPU-busy (~98–100%), around437–459W /700W, at1980MHz in active samples; SM occupancy/kernel compute saturation is unmeasured. DP is clearly below board saturation:53% util,154W /450W, mostly2775MHz.

Latency means use the same accepted completed-episode cohort, joining by(task_uid,attempt). The protocol residual includes transport, serialization, proxy, and execution acknowledgment; it does not isolate tether RTT. Fast-WAM/DP per-step queue and GPU service fields are null/unsupported. Their stage3 means are1333ms and413ms (HS4:2193ms), **queue + compute + synchronization**, so an exclusive compute/queue split cannot be recovered read-only. Cosmos serving logs directly measure queue p50/p90 **776/1477ms**, mean792ms, with service mean613ms: queueing is56% of mean server time.

Batch statistics above are batch-weighted, not request-weighted. HS3 Fast-WAM S3 mean3.70,81.1% under cap; request-weighted p50/p90=6/8. DP mean2.19,98.2% under cap; request-weighted=3/16. HS4 mean4.77,57.0% under cap; requests mostly reachB8. Composition ledgers have no explicit flush reason/enqueue times: under-cap flushes are an expiry **proxy**, with possible compatibility-bucket splitting, not exact timeout counts.

Episode wall inference / sim-render / setup-other shares: HS3 Fast-WAM79.3/12.7/7.7%, DP55.1/27.9/16.4%, Cosmos62.2/25.6/11.3%, HS4 Fast-WAM79.6/11.7/8.4%; image conversion contributes the remaining<1%. Setup-other is the duration residual inside the runner; result persistence and subsequent dispatch gaps sit **outside** these percentages. All observed accepted-attempt retries/errors=0/0; fleet receipts show planned episode-boundary recycles, no worker failures.

Worker node timan107:48 CPUs, mean25.6% CPU, p90~50% (brief higher spikes); MemAvailable minimum82.0GiB, mean109GiB. CPU/memory/I/O PSI mostly near zero on the worker node; renderer GPU mean utilization2.1–3.6%, p90≤13%, peak recorded memory4.3GiB of8GiB each. There is no sustained worker-node CPU/RAM/renderer saturation evidence. Simulator/reset cost remains material, particularly DP/Cosmos. timan108 stayed offline; the recorder retains the node_offline evidence and retries every5min. Local driver-host I/O PSI avg10 was typically25–33% during the HS3 overlap, with low CPU (~2.7%).

### Driver starvation evidence

| Campaign | report/persistence delay mean / p50 s | result→next assignment mean / p50 s | snapshots with fsync wait / instrumented |
|---|---:|---:|---:|
| HS3 Fast-WAM,32 workers (closed) | 18.92 / 12.23 | 18.64 / 15.19 | 5/5 |
| HS3 DP,24 workers (closed) | 22.58 / 19.08 | 28.77 / 26.11 | 5/5 |
| Cosmos baseline resume,8 workers | 2.53 / 2.19 | 3.15 / 0.28 | 12/32 |
| HS4 Fast-WAM,48 workers (external replacement) | 20.61 / 17.54 | 22.24 / 21.19 | 14/24 |

Report delay = durable attempt timestamp − assignment timestamp − worker duration; both wall timestamps are local, so remote clock skew is not involved. It includes result transmission, scheduler-lock wait and durability work. Both HS3 drivers were in`jbd2_log_wait_commit` in5/5 recorded snapshots. `formal_driver.py` holds the scheduler lock around `super().handle_result`; its per-step writer opens/appends/flushes/fsyncs **every row**. Worker pulls need that lock. This couples durable evidence logging to GPU arrivals and explains underfilled batches despite nominal worker concurrency.

### Ranked changes (estimates; none applied)

1. **Batch per-episode durable evidence writes and shorten the scheduler-lock critical section.** Expected **DP1.45–2.2×, Fast-WAM1.2–1.55×, Cosmos1.05–1.12×** episode throughput. Model: remove50–90% of observed report+next-dispatch delay from episode cycles; ceilings are further bounded by GPU/simulator capacity. Keep evidence durable before terminal journal acceptance and preserve attempt fencing. The delay is not all fsync, so this is an estimated range, not a measured improvement.

2. **Qualify Fast-WAM action-head CUDA graphs / torch.compile for fixed batch signatures.** Expected **~1.1–1.3× episode throughput** if launch/service cost falls materially; low confidence until measured. Existing H100 native profile assigns62.9% of B1 wall to the K10 head; a hypothetical2× head improvement gives1.46× native server service, before queue/task/control-plane effects. Source: `models/fastwam/BATCHING.md` H100 qualification; event spans include host launch gaps.

3. **DP two replicas with appropriately smaller batch caps, after fixing arrivals; evaluate Cosmos extra replica only with VRAM/shared-GPU qualification.** DP expected **~1.05–1.15× episodes**: existing frame replay18calls/s (2×B4) versus15calls/s (1×B16), a1.20× server gain, diluted by sim/setup. Cosmos potential **1.0–1.15×**, unmeasured; queue is large but it shares a busy H100. Another Fast-WAM resident needs~26GiB, exceeding current free H100 memory (~14–17GiB). Do not infer a2× gain from doubling replicas.

4. **Recover effective worker concurrency first; then consider more workers or moving workers once timan108 returns.** Additional benefit beyond the above is plausibly **0–15%**. DP batches are under cap, but nominal24 workers spend large amounts of time outside episodes. External32→48 Fast-WAM observation raised calls7.46→8.12/s (~9%) while client mean latency rose2.03→3.02s (~49%); episode rate18.4→21.1/min. Task mix, startup and simultaneous DP closure differ, so this is not a controlled speedup. Worker-node saturation does not justify moving workers now; retain measured memory admission/recycle constraints.

5. **Improve transport/proxy path only after the larger stalls.** Expected **0–5% DP episode gain**, smaller Fast-WAM/Cosmos gain. Completely eliminating DP’s138ms protocol residual is at most1.26× call latency; Fast-WAM/Cosmos residuals56–64ms are only2–4% of call time. Residuals include ack/serialization and are not pure tunnel latency.

6. **Increase batch cap/window last.** Expected **0–5%**, possible regression. Fast-WAM p90 already reachesB8 (HS4 requests mostlyB8); a larger cap cannot fix scheduler starvation. Existing local B8→B16 replay gained only4.6% server throughput with72% higher mean latency (`models/fastwam/BATCHING.md`). DP’s1/5 batch p50/p90 under B16 points to arrival gaps rather than a too-small cap.

Cosmos resume accepted233 new episodes at this snapshot, plus1726 retained =1959/2000. Accepted successes and policy-quality equivalence were not inferred from throughput. All suggestions remain unapplied.

## S: targeted HS4 DP worker/server wait breakdown, 2026-10-10 14:21 CDT

Campaign: `hs4_dp_b16w5_4090_k100_c5_control_20261010_133915`, local 4090, one B16/5ms replica, 32 timan107 workers. **The present inference bottleneck is the serialized stage-3 host/sampler pipeline, while driver durability stalls waste another third of worker time. The 4090 is not GPU saturated.** The relative importance changes with episode length: the short episodes around the coordinator's 14:06 probe were much more driver limited.

Recorder PID1965274 was replaced by **2223906**, running detached with `--no-h100 --no-timan108`. Stop only the recorder with `kill -TERM 2223906`. It remains at `exp/r17_integration/ops/perf_recorder.py`; launch with `exp/r17_integration/ops/perf_recorder.sh --no-h100 --no-timan108`. Its node set is now only local/timan107. H100 probing is disabled for P3's exclusive sweep. The persistent timan107 collector starts once; subsequent healthy samples make zero tether calls. Targeted timan107 execs started at19:12:35 and19:13:21 UTC,46s apart, and over30s after the recorder collector started. The first targeted exec exited on a Python3.8 `Path.readlink` incompatibility; the corrected seven-minute stream completed naturally. No campaign process was signalled or changed.

### Evidence and method

Primary aligned window **14:13:22–14:19:34 CDT (19:13:22–19:19:34 UTC),372s**:372 one-second snapshots of all32 worker slots,371 one-second local snapshots of server PID2020942, proxy2022835 and driver2022909,145 accepted completed episodes,121 closed worker cycles,996 stage-3 batches. The complete targeted remote capture lasted420s. Scripts: `ops/dp_wait_probe.py` (read-only stdout stream) and `ops/dp_wait_analysis.py` (offline analysis). Artifacts under `/data/mw_finetune/perf/2026-10-10/`: `dp_workers_probe.jsonl`, `dp_local_probe.jsonl`, `dp_wait_probe_window.json`, `dp_wait_1400_1410.json`, `dp_socket_states.txt`, `dp_socket_summary.json`. Continuous raw evidence remains `perf.jsonl`.

Nonblocking py-spy attachment to the local driver/server was denied; `/proc/<pid>/stack` and syscall access were also denied. Both hosts have Yama ptrace_scope1; that policy was not changed. Worker main-thread `wait_woken` corresponds to blocking driver TCP receive in the stock WorkerLoop; `futex_wait_queue_me` with an established policy socket corresponds to the synchronous WebSocket response/control condition. Background WebSocket receive threads are excluded from this classification. Socket identities are established from process FD inodes and `/proc/net/tcp`, not just CPU percentages. Main-thread running/other waits include simulation, rendering, episode preparation and serialization.

| DP worker component | % of all32 worker slots' wall time | Basis |
|---|---:|---|
| Waiting for policy response / server controls | **49.0%** | Main WebSocket condition waits; includes control acknowledgments, not just inference |
| Waiting for driver dispatch/result processing | **32.8%** | Main blocking driver TCP receive |
| Simulation/rendering/image handling | **approximately10.5%** | Completed-cycle client env+image timers, reconciled with the sampled running/other bucket |
| Startup/attestation/recycle | **approximately3.2%** |2.34% without established driver TCP +0.84% absent slots; absent slots include up to5s census lag |
| Episode setup/serialization/other | **approximately4.5%** | Remainder of the15.0% running/other bucket after the simulation estimate |

The last three rows are reconciled estimates, not exact stack-derived allocation. A separate, exact completed-cycle accounting gives48.61% inside timed policy infer,10.48% sim/image,5.73% runner setup/control/other and35.17% outside the runner (driver + recycling). Thus two independent methods put active-runner time near65–67%, with approximately one third lost between episodes. Startup is not the dominant loss.

| DP server component | % of server wall | Scope |
|---|---:|---|
| Stage3 batch host execution in flight | **approximately93%** | Reconstructed host spans; includes Python launches, scheduler arithmetic, bookkeeping and GPU waits |
| No stage3 batch in flight | **approximately7%** | Arrival gaps and batch assembly/wait window |
| At least one policy request resident | **98.26%** | Direct union of serving end-to-end request intervals; includes queue waits |
| Stage1 / stage2 batch host execution | **approximately41.7% /3.7%** | Overlaps stage3; do not add these percentages |

The ledgers lack true enqueue/dispatch timestamps and per-step queue/service fields are explicitly null/unsupported. Reconstruction joins each completed per-step stage duration to its batch receipt: infer each enqueue as receipt end minus stage host duration, then apply the previous same-stage batch end and the sealed5ms assembly window. All996 stage3 batches in this window have matched member timings. Receipt publication/thread wakeup skew prevents exact compute-exclusive accounting; interpret93% as roughly90–95% host batch activity, not CUDA kernel duty. Stage3 idle gaps p50/p90 **5/120ms**; GPU utilization is **mean40.9%, p50/p90 35/92%,164W mean** versus450W power limit. There were33 zero-util samples,28 coinciding with driver filesystem commit wait. A6s zero-SM probe is an idle interval, not proof the whole campaign has no inference work.

Current throughput is **23.39 accepted episodes/min and14.73 policy calls/s**. Stage3 batch-weighted fill **p50/p90=3/16, mean5.50, cap16**;88.0% of batches are under cap. This is not the same fill pattern as earlier short episodes: the14:00–14:10 window had p50/p90=1/~7, mean2.72,97.1% under cap and9.39calls/s. Larger caps alone cannot fix either pattern.

### Latency, driver and node evidence

Matched completed-episode mean client call **1039ms =867ms server WebSocket inference +172ms transport/serialization/proxy/protocol residual**. Server stage host means: prepare16ms, stage1 119ms, stage2 16ms, stage3 **653ms**, to-wire17ms; remaining wrapper/host time is about46ms. Reconstructed stage3 queue wait **p50/p90 approximately256/503ms**, mean279ms; stage3 batch execution mean348ms across batches. Those means have different request/batch weighting and cannot be subtracted as an exact kernel split. Queueing and host sampler execution both matter; transport is approximately17% of call latency, not its majority.

**DP's live data path is not the H100 tether tunnel.** Workers connect to the public local-proxy endpoint on23180 and driver23189; the proxy connects to the replica over127.0.0.1:23181. TCP RTT to timan107: policy p50 about17.8ms, driver about18.1ms; proxy/backend minimum RTT p50 0.016ms. The172ms residual is much larger than one RTT and includes image serialization/transfer, proxy scheduling and protocol work. Policy socket queues are p50/p90 zero, with occasional large image bursts; there is no sustained policy transmit backlog evidence.

Driver threads spend **88.4%** of local snapshots in `jbd2_log_wait_commit`; an open `per_step.jsonl` FD co-occurs, and **87.3%** of samples have unread result data queued on driver sockets. Mean/p50 result-durability delay **13.34/13.65s**, accepted-result→next-dispatch gap **15.23/13.70s**. The frozen runtime `formal_runtime_ffaf8938.../exp/r17_integration/ops/formal_driver.py` lines114–117 fsync every append; its `steps` writer fsyncs every per-step row, while `handle_result` lines128–142 holds the scheduler lock across the writer and journal. Pending worker pulls share that lock. This is a concrete dispatcher bottleneck, not simulator CPU starvation.

Around the coordinator probe, **14:00–14:10**,157 closed cycles spent only35.25% in the episode runner and **64.75% outside it**. Result durability and next-assignment delays averaged31.53s and34.04s.15/20 recorder snapshots had driver filesystem commit waits. That window ran17.6episodes/min,9.39calls/s and mean48.45%4090 utilization/154W. Its extreme low-worker-CPU intervals are consistent with driver persistence/dispatch starvation. The later longer episodes amortize that overhead and push the single stage3 pipeline near its throughput ceiling.

timan107 whole-node CPU during the targeted window averaged about19% of48 cores, MemAvailable never below131GiB in recorder samples, and renderer GPUs remained lightly utilized. DP worker per-process CPU median was about15% of one core. The hottest local server thread (TID2022642, inferred stage3 from creation order/CPU pattern) used about63% of one core during the complete local capture; stage1 about20%, proxy8%. Existing denoiser CUDA graphs are already enabled. The source `models/dp/staged.py` still drives **100 Python sampler iterations**, including scheduler updates and private per-request posterior RNG work, outside the denoiser graph. High host batch residency with much lower GPU duty is evidence of host/launch/synchronization gaps; exact exclusive kernel duration cannot be recovered with the available read-only counters. No accepted episode errors/retries were observed.

### Ranked headroom for the current workload (estimates; no changes applied)

1. **Optimize/capture the entire DDPM100 sampler path, including scheduler updates and launch bookkeeping, while preserving request-private RNG semantics.** Estimated **1.15–1.35× episode throughput**, low confidence until qualified. Existing graphs cover the denoiser only; stage3 host activity~93% versus GPU duty41% leaves substantial host/launch gaps. If response wall falls30–50%, the~49% policy-wait share gives an ideal1.17–1.32× cycle improvement before changed queueing. Do not simply enable graphs again or change K/quality.

2. **Two independent local DP replicas, with B4–B16 caps, and evaluate MPS only if needed.** Estimated **1.10–1.20× episodes**; strongest existing measured server evidence. The saved4-second,12-client frame replay (`models/dp/samples/dynamic_batch_table.md` and CSV) measured **15calls/s/152W for1×B16 versus18calls/s/214.5W for2×B4** (also18calls/s for2×B16). This is a short microbenchmark, not a formal campaign result. It demonstrates overlapping independent host pipelines raises power/work; MPS itself was not measured. Expected gains are not2×.

3. **One durable per-episode per-step append/fsync and a shorter scheduler-lock critical section, retaining evidence-before-terminal-journal ordering and attempt fencing.** Current single-replica episode estimate **1.05–1.15×**, potentially **1.3–1.6× in the earlier short-episode/starved regime**, and useful before adding concurrency. Eliminating80% of sampled33% driver wait would give a1.36× worker-cycle ceiling, but the present14.73calls/s is already near the15calls/s short replay, so server capacity limits the realized gain. This also removes bursts/idle periods and should improve consistency of4090 utilization.

4. **Reduce image/protocol overhead and serialized execution acknowledgments; avoid extra routing/tunnel layers.** Estimated **1.00–1.10× episodes**, not a primary fix. Client-minus-server residual172ms is17% of call latency; zeroing it would reduce~49% inference-wall share by only~8%, an ideal~1.09× cycle gain. Some runner-other wall includes control acknowledgments. TCP itself is~18ms RTT and DP already uses direct local-proxy transport.

5. **More nominal workers, moving workers, larger cap/window:** estimated **0–5% current gain**, possible regressions.32 workers already keep requests resident98% and stage3 busy~93%; worker-node CPU/RAM/rendering have headroom but are not the current limiter. Existing B16/B24/B32 replay all measured15calls/s. More workers without driver/server changes mainly increase queue waits; a longer assembly window adds latency. Current underfilled median batches coexist with p90 at cap, so tuning must distinguish arrival gaps from a larger-cap need.

Numbers are not additive; changing the driver can expose more server queueing, and adding replicas changes batch sizes. Six CPU-only recorder tests passed, including an assertion that DP-only mode never contacts H100/108. Diagnostic streams have exited; recorder2223906 remains running. Campaign configuration, processes and GPU workloads were left untouched.

Offline reproduction of the aligned primary window:

```bash
python3 -m exp.r17_integration.ops.dp_wait_analysis --campaign /data/mw_finetune/eval/hs4_dp_b16w5_4090_k100_c5_control_20261010_133915 --start 1791659602 --end 1791659974 --output /data/mw_finetune/perf/2026-10-10/dp_wait_probe_window.json
```

