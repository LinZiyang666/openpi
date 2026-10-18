Coordinator answers to your two open questions (I interrupted your turn only to deliver these; continue the rebuild from where you were).

1. knobsmokedp1355 (old LIBERO dev-pool smoke run, pool B): keep it in the trash backup only. Do NOT build a replacement and do NOT put it in any GPU rerun queue. The formal design has no dev set.

2. Fast-WAM MT50 cache-variant lane: use the split that matches the original serving config: fresh R3 jobs for library sizes 250–1000 and R2 jobs for 1500–2500 (R3 exceeds the memory cap there, per RUN_CONFIGS.md). Record the split in the campaign definitions and in your handback.

Everything else in your current task stands: rebuild every withdrawn run with the fixed decision-step code (plus one "model on every decision + long segment" control per model × step mode), then give me a balanced two-queue plan (H100 / 4090) with prepare/launch commands. Do not launch anything on the GPUs yourself. Keep the /data I/O discipline (single reader, sampled, check /proc/pressure/io before bulk reads).
