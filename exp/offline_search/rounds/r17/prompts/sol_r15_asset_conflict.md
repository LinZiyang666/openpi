<task>
H100 R15 legacy prepare fails for the pi05 LIBERO full-step formal_A levels campaigns (fa_pi_f_u_l, likely fa_pi_f_f_l too):
RuntimeError: EXISTING_ASSET_CONFLICT mirror/home/weiland/projects/openpi/exp/offline_search/rounds/r11/knob/artifacts/r15_pi05_l10_100_ir25.pkl
(log: openpi_ext/scratch/r17_ops/ff/fa_pi_f_u_l.prep.log; sync of /data/mw_finetune/eval/fa_pi_f_u_l_l100, plan /data/mw_finetune/eval/fa_pi_f_u_l_l100/h100_sync/plan.json).
Facts I checked: the local original (sha 1263ca85..., 10-05 10:09) is unchanged; the plan expects the campaign-generated rewrite sha 4c3acf89... (35,631,780 B) at the shared H100 mirror path /data/oscl_h100/mirror/home/weiland/projects/openpi/exp/offline_search/rounds/r11/knob/artifacts/r15_pi05_l10_100_ir25.pkl, but the H100 holds a third version sha 6b3ab93f... (35,631,769 B, written 10-05 20:13 by an older campaign). So campaign-specific generated assets share one mirror path.

Do, CPU + read-only remote checks (no launches, no GPU):
1. Confirm the cause and list every conflicting rel path for all remaining R15 queue items (fa_pi_f_u_l, fa_pi_f_f_l and any other R15 item in the H100 chain), with expected vs remote sha.
2. Fix it so campaign-specific generated assets never collide (e.g. content-addressed or campaign-scoped mirror paths), without touching assets that a closed or running campaign's recorded plan points to (closed runs' evidence must stay reproducible). If the fastest safe fix is to move a stale remote copy aside (mv to a dated backup name, never delete) because no running/queued campaign uses it, say so and give the exact commands.
3. Give me the commands to re-prepare fa_pi_f_u_l and fa_pi_f_f_l. The H100 is otherwise idle until this or the R17 fix lands, so be fast.
</task>
<compact_output_contract>Reply in Chinese, short: cause, fix, exact commands, what you verified.</compact_output_contract>
