<update>
Two corrections to your brief, continue the same task with these:
1. GPUs: both the local 4090 and the H100 are already free and exclusively ours (I018 cleared them at 23:4x). Short GPU smoke checks (≤ 15 min total) may start now.
2. H100 disk: the owner approved deleting our unneeded data. I am deleting /data/xwam (except /data/xwam/wan22_5b) on the H100 now (~165 GB will be freed; watch `df -h /data`). After it completes you may use up to ~120 GB under a new `/data/mw_finetune/` on the H100. Still never touch /dev/shm/i018, /data/idea_sprint or other dirs; ask me before deleting anything else.
3. Reuse: `Wan-AI/Wan2.2-TI2V-5B` is already on disk — H100 `/data/xwam/wan22_5b/` and weilandserver `/data/xwam/wan22_5b/` (contains Wan2.2_VAE.pth, DiT safetensors, configs; check whether the umt5-xxl text encoder Fast-WAM needs is present). Prefer pointing Fast-WAM at it instead of re-downloading.
4. Agents: only sol/astra/opus may be used; you need none.
</update>
