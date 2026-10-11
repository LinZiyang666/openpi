<task>
Review note on your LIBERO library pipeline: H100 `/data` has only about 90 GB free (91% used). Your own estimate for Fast-WAM × libero_10 is about 86 GB of capture/transfer/build working space on the capture host. That will not fit safely on H100.

Change the H100-side behavior so the capture host never accumulates more than a bounded amount, e.g. ≤ 20 GB:
- stream closed capture shards back to weilandserver during the campaign (SHA-checked, using the existing /tmp-stage transfer) and delete each H100 copy only after verified receipt;
- build the parent only on weilandserver (`/data/r17_library`, 2.4 TB free);
- add a free-space admission check on H100 before start and during the run, failing closed.
Keep the DP local path unchanged. CPU tests only. Do not touch running campaigns or GPUs.
</task>
<compact_output_contract>Append "## H100 bounded staging" to HANDBACK_P6.md. Final message: 3 lines.</compact_output_contract>
