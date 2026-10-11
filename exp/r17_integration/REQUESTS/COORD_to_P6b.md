
## 2026-10-10 20:5x coordinator error — please re-run
While stopping a failed H100 cache lane, the coordinator mistakenly sent SIGTERM to your CPU prepare-only process
`cache_lane --lane h100 --directory /data/mw_finetune/eval/cszeh --prefix csze ...` (PIDs 463102/463103). It was CPU-only
(CUDA_VISIBLE_DEVICES=''), no GPU/campaign was involved. Please re-run that preparation (idempotently / fresh directory) and
check nothing half-written remains. Also note (Q 20:47): cache workers need ~3.5 GiB RAM each; Q is adding timan1 (GPU 3 only)
and node-split fleets (--worker-counts), so size-grid lanes should use split fleets (timan107/local/timan1) rather than packing
48+32 workers on timan107; timan107 disk is at 98% (P6 adding cleanup/guard).

## 2026-10-10 21:1x coordinator: fleet sizes for the size grid (supersedes the 32/48 split in your current prompt)
Owner: more devices must mean MORE workers (not the same 80 redistributed), and local workers serve the local 4090 lane first
(no extra transport; the 4090 lane may also add external workers). Measured capacity: timan107 ~45, local ~60, timan1 (GPU 3) ~40.
- 4090 lane (DP): local ~60 workers (localhost), plus timan107 only if the DP server keeps up; raise DP replicas to 3-4 so the
  cache lookup server CPU stays < ~70% per replica.
- H100 lane (Fast-WAM then Cosmos): timan107 ~45 + timan1 ~40, no local workers.
Respect RAM/renderer admission (cache workers ~3.5 GiB; timan1 6 GiB reservation) and the disk guard.
