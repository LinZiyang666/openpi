<task>
Coordinator, 2026-10-17 01:2x. Small fix in exp/offline_search/rounds/r17/filter_failed/worker_resilience.py (your code).

Evidence: /data/mw_finetune/eval/ffk1015_cosmos_libero_reduced_off_u/WORKER_NODE_STATE.json (launch 01:09) measured timan108 capacity 0 with reason `nvidia-smi -i 0000:01:00.0 -q -x timed out after 4 seconds` (inside the `tether exec --safe --timeout 10s timan108 -- /usr/bin/python3` probe). timan108 was completely idle at the time (all four GPUs 0 MiB, none of our worker processes, load ~7). So an idle node was dropped from the campaign; its share went to timan1/timan107/local. The previous campaign ff1015_cosmos_li_04 (launch 23:44) also got only 7 timan108 workers although stage measured 70 — probably the same timeout or a partial probe; please check its WORKER_NODE_STATE.json / logs and confirm.

Fix with the smallest change:
- Capacity probe: give nvidia-smi a realistic timeout (idle GPUs without persistence mode can take tens of seconds on first query; use ~30 s per card query or one combined query) and the tether call a matching bound (e.g. 90 s for this read-only probe only).
- A probe that times out or errors is "unknown", not capacity 0: retry once; only after a second failure drop the node (and log the reason).
- Keep the short bounds you chose for spawn/control calls.
Add/adjust the targeted test. Do not change running runs or queue scripts; new prepares/launches pick it up.
</task>

<rules>
- CPU only locally; tether exec read-only checks only; never push/pull/expose; no workers/servers/GPU/chains; no pkill, rm, git commits; never touch credentials.
</rules>

<output>
Final message: root cause for both campaigns, the change, and the test result.
</output>
