<task>
Time-critical (deadline ~22:15 CDT, when Fast-WAM continuation fastwamc5481985003 on the H100 lane csl10h finishes): the next
campaign on that lane, the size-500 Cosmos cache campaign (prepared as /data/mw_finetune/eval/cosmosp6223294666 per your last
report, execution_nodes timan1=8, 8 workers), is far too small: 4000 episodes on 8 workers would take ~5 h. Owner rule: more
devices = more workers; local workers belong to the local 4090 lane; the H100 lane uses external nodes: timan107 ~45 + timan1 ~40
(GPU 3 only). Note timan107 also hosts the DP continuation dpc3464789869 (32 workers) until it finishes (~1.5 h), so admission
must account for it: start Cosmos with what fits now (e.g. timan107 ~13-16 + timan1 ~40) and, if the lane design allows, let the
fleet grow on timan107 when the DP continuation releases it — otherwise pick the largest safe static split.
Do: re-prepare that Cosmos campaign as a fresh campaign (it has not dispatched) with the larger split, keep every scientific
setting identical (YAMLs, library size 500, server config: h100 2 replicas B1; raise replicas only if the cache server CPU would
saturate with ~55-85 workers — cache actions do not depend on replicas; record it), make sure the running lane process will
dispatch the new prepared run (not the old 8-worker one) without interrupting Fast-WAM, and report. Also check the Cosmos server's
cache-lookup CPU at that worker count is not the new bottleneck.
</task>
<compact_output_contract>
End with: new Cosmos run name, fleet split, replicas, how the lane picks it up, expected duration.
</compact_output_contract>
<action_safety>
Do not interrupt fastwamc5481985003 or the 4090 lane. timan1 GPU 3 only. No pkill -f bare patterns, no rm -rf, no /data/i018,
/dev/shm/i018, codex credentials.
</action_safety>
