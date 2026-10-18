<task>
Owner question, 2026-10-17 22:0x: our worker fleets keep taking machines down. Is the way we run workers risky? Can timan108 be repaired? Investigate. Read-only unless stated.

Known incidents (verify each one and find any others):
- timan107: whole machine down 10-14 ~00:21 (ping unreachable, dirty journal, root reboot 12:37). See `logs/session_handoff.md` around lines 664 and 694–716.
- timan108: abnormal reboots or offline periods on 10-05, 10-10 to 10-14, 10-16 09:11 (uptime 0), and OFFLINE since 10-17 05:43 (16 h now).
- timan1: tether agent lost 10-16 07:36 for about 7 h.
- The device notes say timan108 once froze its NVIDIA kernel module at about 45 EGL workers.

For each incident, from our run records (`/data/mw_finetune/eval/*/WORKER_NODE_STATE.json`, `LAUNCH_WORKER_DEPLOYMENT.json`, `TETHER_CALLS.jsonl`, journals' last accepted timestamps, `logs/session_handoff.md`, `exp/r17_integration/ops/RUN_CONFIGS.md`, `/home/weiland/projects/dist_experiment_control/docs/devices.md`, `dynamical_system/BINDING.md`), establish:
- what we ran on that node at the time: worker count, EGL contexts per GPU, worker RAM/VRAM reservations, whether workers were recycled (`--worker-max-episodes`), legacy R15/R16 single-host fleets;
- disk usage, if known;
- the last evidence of life.

On the nodes that are online now (timan107, timan1 via tether exec, read-only), collect whatever we are permitted to read: `uptime`, `last -x`, `journalctl --list-boots` and `journalctl -b -1` / `-k` (if permitted), `dmesg` (if permitted), `/var/log/kern.log` (if readable), `nvidia-smi -q` (Xid / retired pages), `free -g`, `df -h`, our process RSS/VRAM now. Look for OOM kills, NVIDIA Xid, EGL/driver errors, NFS hangs (timan107 had a stalled mount today at 18:19), and filesystem errors.

timan108 repair check:
- From timan107 or timan1 (tether exec), test whether timan108 is reachable on the internal network (ping, ssh port, `ssh -o BatchMode=yes` if keys already work).
- If the host is up but only our tether agent died, report the exact way to restart our agent with its existing on-node configuration. Do not read or print any credential, token or auth file. Do not restart anything yourself.
- If it is down or unreachable, say who must act (owner/administrator) and which logs they should read after boot.

Conclude, with evidence and without speculation presented as fact:
- probable causes ranked, each with its evidence;
- the risk of our current worker practice;
- concrete safe limits and practices per machine: max EGL workers per GPU, recycling interval, RAM/VRAM headroom, disk headroom, pre-launch checks;
- what the coordinator should change. Mark anything unproven as such.
</task>

<rules>
- Read-only everywhere except your report file. tether exec only; never push, pull or expose. No kill, no restart, no rm, no sudo attempts beyond reading what is already permitted. Never touch credentials or auth files.
- Do not touch running experiments.
</rules>

<compact_output_contract>
Write `/home/weiland/projects/openpi/exp/offline_search/rounds/r17/ops_incidents/WORKER_MACHINE_CRASHES.md` in Chinese, with tables and file:line or command citations. Final message: a short summary of the timan108 repair verdict and the top recommendations.
</compact_output_contract>
