"""Current-frame-only frozen-video loss; no future VAE, no video head/loss."""
import torch
from torch.nn import functional as F


def frozen_video_loss(policy, sample):
    model = policy.model
    image = sample["video"][:, :, :1].to(device=model.device, dtype=model.torch_dtype)
    latent = model._encode_video_latents(image)
    context, mask = sample["context"], sample["context_mask"]
    context = context.to(device=model.device, dtype=model.torch_dtype)
    mask = mask.to(device=model.device, dtype=torch.bool)
    context, mask = model._append_proprio_to_context(context, mask, sample["proprio"][:, 0])
    action = sample["action"].to(device=model.device, dtype=model.torch_dtype)
    noise = torch.randn_like(action)
    t = model.train_action_scheduler.sample_training_t(action.shape[0], device=model.device, dtype=action.dtype)
    noisy = model.train_action_scheduler.add_noise(action, noise, t)
    target = model.train_action_scheduler.training_target(action, noise, t)
    video_pre = model.video_expert.pre_dit(x=latent, timestep=torch.zeros_like(t),
        context=context, context_mask=mask, action=None, fuse_vae_embedding_in_latents=True)
    action_pre = model.action_expert.pre_dit(action_tokens=noisy, timestep=t, context=context, context_mask=mask)
    attention = model._build_mot_attention_mask(video_pre["tokens"].shape[1], action_pre["tokens"].shape[1],
        int(video_pre["meta"]["tokens_per_frame"]), latent.device)
    tokens = model.mot(embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
        attention_mask=attention, freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
        context_all={"video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
            "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]}},
        t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]})
    prediction = model.action_expert.post_dit(tokens["action"], action_pre)
    per_token = F.mse_loss(prediction.float(), target.float(), reduction="none").mean(dim=-1)
    valid = (~sample["action_is_pad"]).to(per_token)
    per_sample = (per_token * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)
    weight = model.train_action_scheduler.training_weight(t).to(per_sample)
    loss = (per_sample * weight).mean()
    return loss, {"loss_action": float(loss.detach()), "loss_video": 0.0}
