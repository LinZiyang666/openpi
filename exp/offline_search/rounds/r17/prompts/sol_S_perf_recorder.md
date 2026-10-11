<task>
Owner: record performance continuously while campaigns run; he is sure there is throughput headroom left. Build the
recorder, run it on the live campaigns now, and find the headroom.

1. exp/r17_integration/ops/perf_recorder.py (+ a tiny launcher script): read-only, runs alongside any formal
   campaigns, samples every 30 s into /data/mw_finetune/perf/<date>/perf.jsonl:
   - server GPUs: local 4090 (nvidia-smi: util, SM clock, power, memory) and h100 via ONE batched tether exec per
     sample (util, power, memory, per-process); keep total tether calls <= 2 per 30 s;
   - worker node timan107 (and timan108 when online): load, CPU %, MemAvailable, per-GPU memory/util, worker count;
   - per campaign (any /data/mw_finetune/eval/hs3_* or resume run not terminal): accepted episodes, episodes/min,
     policy calls/s, per-call latency and queue wait (per_step.jsonl), batch fill distribution and wait-window
     expiries from the batch ledgers, worker idle/sim/inference split if derivable, retries/errors.
   Write a rolling summary (summary.md, refreshed every 5 min) with the bottleneck per campaign.
2. Start it now (nohup, no GPU) and keep it running; give the coordinator the PID and how to stop it.
3. After >= 20 min of data: quantify headroom per campaign — is the server GPU saturated (util/power), is batch fill
   below cap (need more concurrent workers), are workers waiting on the server or on simulation/rendering, is the
   worker node CPU/RAM/renderer-bound, is per-call latency dominated by queue wait, transport (tether tunnel) or
   compute? Rank concrete changes by expected speedup (e.g. more workers, replicas, batch cap/window, CUDA graphs /
   torch.compile for Fast-WAM, moving workers, transport path) with the evidence. Do not apply them.
Running now: hs3_fast_b8w5_h100_k10_c5_control_* (h100, 2 replicas, B8/5ms, timan107 x32),
hs3_dp_b16w5_4090_k100_c5_control_* (4090, 1 replica, B16/5ms, timan107 x24), Cosmos baseline resume (h100,
2 replicas, timan107 x8). sol R (thread 01a126f9-3aaa-75c1-993e-e4fe8b770fcc) is writing ops/RUN_CONFIGS.md from the
same data; append your findings to exp/r17_integration/REQUESTS/S_to_R.md so it can cite them.
</task>
<compact_output_contract>
End with: recorder path/PID/stop command, a table per campaign (episodes/min, GPU util/W, batch fill p50/p90 vs cap,
queue wait p50, latency split, bottleneck), and the ranked headroom list with expected speedup and evidence.
</compact_output_contract>
<action_safety>
Read-only on campaigns: do not launch, stop, signal or reconfigure any campaign; no GPU work of your own. No pkill -f
bare patterns, no rm -rf, no /data/i018, /dev/shm/i018, codex credentials. timan1 forbidden.
</action_safety>
