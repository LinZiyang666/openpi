<task>
Coordinator, 2026-10-17 09:3x. The short-name formal_A H100 R17 campaigns cannot launch:
`fa_fw_f_f_o` (filtered Fast-WAM LIBERO full knob-off, H100) prepared fine, started 09:30:11, ended 09:32:22 rc=1 receipt FAIL (all_owned_closed=true) with
`Server rejected 'load_cache_config': broadcast failed on 2/2 replicas: [Errno 2] No such file or directory: '/data/mw_finetune/eval/fa_fw_f_f_o/assets/full.yaml'`.
The file exists locally. On H100, earlier working ffk campaigns have a copy under H100's own `/data/mw_finetune/eval/<run>/assets/` (e.g. ffk1015_cosmos_libero_reduced_off), but the fa_* runs only have `/tmp/r17_formal/<run>` — the H100-side /data asset copy is never made for them (probably lost in the short-name rename/rebuild). Every H100 R17 fa_* campaign (Fast-WAM full o/l, Cosmos full o/l, Cosmos reduced l, both arms) will fail the same way. The 4090-side fa_* campaigns serve locally and run fine (fa_dp_f_u_o is running). π0.5 R15 H100 items (fa_pi_f_p10/ps) ran fine.

Also earlier today H100's root disk filled (/tmp/r17_formal 24 GB) and fa_fw_f_u_o's prepare failed on rsync "No space left on device"; I moved closed runs' staging to H100 /data/r17_formal_moved (/ is now 60%) and a guard moves ended runs' staging when / > 80%.

Fix so that every H100 fa_* campaign gets exactly what the earlier working H100 campaigns had on H100 (the same /data asset copy and anything else the H100 server reads), through the existing staging path (public-network rsync only, rate limited), then:
- reset fa_fw_f_f_o (failed launch) and fa_fw_f_u_o (failed prepare) with the existing rename-only reset;
- verify read-only on H100, for one fa_* campaign after a CPU-side staging dry run or the real prepare, that every path the server will read exists; a short CPU test for the path mapping;
- tell me which fa_* NAMEs are now ready to queue on the H100, in the order of your previous list (knob-off stages before level stages).
</task>

<rules>
- CPU only locally; tether exec read-only checks only (plus the existing staging rsync if you run a real prepare); never push/pull/expose; never start or kill servers, workers, GPUs or chains; no pkill, rm, git commits; never touch credentials.
- Do not modify running runs or the running chains (ffk-h100z, ffk-4090z, h100_disk_guard).
</rules>

<output>
Final message: root cause, the change, the reset/verification results, and the ordered H100 NAME list.
</output>
