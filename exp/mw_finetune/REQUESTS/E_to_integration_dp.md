# E → integration architect: importable stateless DP-T

Entry: `exp.mw_finetune.dp.policy.load_policy(checkpoint, device, **kw)`.
`infer(obs)` returns physical float32(16,4) for MetaWorld; `domain="libero"`
accepts the existing LIBERO wire and returns(16,7). `reset()` does nothing.
Only current images/state/text and a fresh private generator from `obs.seed`
(seedless LIBERO defaults to0) affect output; serving holds no request cache.

Full100-step epsilon DDPM is the default. `Policy.model` is the reconstructed
torch module; `model.condition(inputs)` computes one observation token;
`model.scheduler(steps)` returns a fresh diffusers DDPM scheduler, and
`model.sample(inputs,generator,steps)` denoises16 normalized actions then
undoes the internal affine. The wrapper also undoes external quantiles.
Both original checkpoints strictly load475 tensors. Do not bypass either
normalizer or reuse the private generator across requests when wrapping.

Native A registration is requested in `E_to_A_dp.md` (23154/23254).
The source-camera ML45 weights do not transfer usefully to A's single corner2
image in bounded sanity; the prepared fallback is scratch MT50 DP training.
LIBERO weights are mechanically runnable and have a low source-frame denoising
error, but the first multi-suite smoke is0/4. Four-suite fine-tuning tooling
is under `dp/libero/`. See `dp/RECIPE.md` and `dp/HANDBACK_E.md` for evidence.
