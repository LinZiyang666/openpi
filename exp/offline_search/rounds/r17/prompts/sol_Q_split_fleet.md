<task>
Urgent, the H100 is idle. The Fast-WAM cache continuation fastwamc1456717485 (H100 lane, csl10h, log
/home/weiland/projects/openpi_ext/scratch/r17_ops/cache_lane_h100_r3.log) FAILED: timan107 kernel OOM-killed workers (dmesg:
python anon-rss 3.0-3.4 GB each) -> worker_restart_resource_timeout. Cause: the cache lanes reserve 2048 MiB/worker but cache
workers use ~3.1-3.5 GiB; DP continuation dpc3464789869 (32 workers, 4090 lane, running now — do not touch) + Fast-WAM 48 workers
both on timan107 (220 GiB) exceeded RAM. timan107 disk is 98% (5.8 GB free; P6 is adding a disk guard/cleanup now).
Do:
1. Add timan1 as a worker node from exp/r17_integration/REQUESTS/T1_to_Q.md (GPU 3 only via T1's device-namespace guard;
   max 16 workers, 6 GiB RAM / 1 GiB VRAM each, recycle 4).
2. Allow a cache continuation/resume to redistribute the SAME total worker count across nodes (timan107/local/timan1) while
   keeping every scientific setting pinned; record per-worker node/device. (Observations are byte-identical across nodes per
   your local smoke and T1's 672-array comparison.) Correct the cache-lane per-worker RAM reservation to the measured ~3.5 GiB
   so admission prevents OOM.
3. Relaunch the H100 lane yourself (cache_lane --lane h100 --directory /data/mw_finetune/eval/csl10h --run, resume-aware) with
   the Fast-WAM continuation fleet split to keep 48 workers without overloading timan107 next to DP's 32 — e.g.
   timan107=16, local=16, timan1=16 (check local admission next to the 4090 DP server, local CPU 88 cores) — and make the
   following Cosmos campaign use a similarly safe split. Keep it running; report run names and progress. Do not touch the 4090
   lane.
4. Write the new pitfall (cache workers need ~3.5 GiB; never pack two 48+32 fleets on timan107) to REQUESTS/Q_to_R.md.
</task>
<compact_output_contract>
End with: files/tests, the fleet split used, resumed Fast-WAM run name + accepted count, Cosmos plan, timan107 RAM headroom.
</compact_output_contract>
<action_safety>
timan1: GPU 3 only (T1 guard). Do not stop/signal the DP continuation or other users' processes. No pkill -f bare patterns, no
rm -rf, no /data/i018, /dev/shm/i018, codex credentials.
</action_safety>
