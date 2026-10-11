<task>
Minimal fix in `exp/r17_integration/ops/formal_eval.py` `prepare_workers` (around lines 164–176). The LIBERO worker env sets `LIBERO_CONFIG_PATH='/scratch/zixuans8/openpi_trace_h100/os_cl/libero'` (valid on timan107) and overrides it to `/home/zixuans8/.libero` only when `campaign['model']=='cosmos'` (your timan108 special case). On timan108 the legacy path is an EMPTY directory.
Result: the Fast-WAM LIBERO library smoke on timan108 failed in the worker validation probe with LIBERO's interactive "Do you want to specify a custom path for the dataset folder? (Y/N)" → EOFError (log: `/home/weiland/projects/openpi_ext/scratch/r17_ops/lib_fw_smoke_libero_spatial.log`).

Make the config selection depend on the worker node, not the model: timan108 → `/home/zixuans8/.libero` (existing, read-only); timan107 → unchanged legacy path. Also handle `--extra-workers` mixing both nodes, if that path builds per-node envs. Verify read-only on timan108 that `/home/zixuans8/.libero/config.yaml` points at assets/init files the libero311_v1 runtime can import for spatial and libero_10 (the worker import/validation probe passes).
Minimal, merge-friendly edit; P6 is concurrently editing other parts of the ops code. Add a CPU test.
No GPU work. Do not touch running processes: DP libero_10 on the 4090/timan107, and P6's work.
</task>
<compact_output_contract>Final message: 2 lines (change, probe/test result).</compact_output_contract>
