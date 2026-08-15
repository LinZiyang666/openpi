"""API-level student trainer for one model (Phase 1, lerobot venv).

Why this exists (post-G2 deviation, evidence in the plan log): the approved
route shelled out to ``lerobot.scripts.train``, but lerobot 0.3.3 cannot
consume pre-chunked labels through that CLI. Our datasets store the O5
contract label — the full teacher ``env_action_chunk`` [10, 7] — as a
per-frame feature named ``actions``. The stock pipeline hard-reads
``batch["action"]`` (singular) and, for a feature named exactly ``action``,
resolve_delta_timestamps() stacks chunk_size timesteps of it, which for a
pre-chunked value yields [10, 10, 7]. Either name therefore breaks: ``actions``
never reaches the policy, ``action`` gets double-chunked. This entry loads the
dataset with NO delta machinery and adapts each batch in-process:
``batch["action"] = batch.pop("actions")`` (already [B, 10, 7]) plus an
all-False ``action_is_pad`` — every label is a complete real chunk. The
training objective is byte-identical to O5; only the transport differs.

Checkpoint layout keeps the 0.3.3 contract the selector/sidecar depend on:
``<out>/checkpoints/<step:06d>/pretrained_model`` with a ``last`` symlink
(``select_student_checkpoint.resolve_pretrained_dir``).

SmolVLA: the base checkpoint's weights are loaded EXCEPT its normalization
buffers — normalize/unnormalize must come from OUR dataset stats, not the
base's training stats (silently keeping base stats would corrupt both inputs
and targets). chunk_size=10 re-uses per-token weights; length-dependent
buffers that fail to match are freshly initialised (strict=False, logged).

Coupling map:
  CONSUMED BY: train_smolvla.py / train_act.py (subprocess per model)
  DEPENDS ON:  lerobot 0.3.3 (pinned venv), select_student_checkpoint layout
"""

from __future__ import annotations

import argparse
import logging
import pathlib

logger = logging.getLogger(__name__)


def adapt_batch(batch: dict, chunk: int) -> dict:
    """Rename pre-chunked labels to the policy contract + full-valid pad mask."""
    import torch

    actions = batch.pop("actions")
    if actions.ndim != 3 or actions.shape[1] != chunk:
        raise SystemExit(f"actions shape {tuple(actions.shape)} != [B, {chunk}, D]")
    batch["action"] = actions
    batch["action_is_pad"] = torch.zeros(
        actions.shape[0], chunk, dtype=torch.bool, device=actions.device
    )
    return batch


def policy_features(ds_meta, chunk: int):
    """Split dataset features into policy input/output features (action = 7-dim).

    Manual mapping (not dataset_to_policy_features): the builder's image
    features carry no ``names`` metadata, and getitem returns images as CHW
    float tensors — declare VISUAL shapes channels-first accordingly.
    """
    from lerobot.configs.types import FeatureType, PolicyFeature

    feats = ds_meta.features
    if "actions" not in feats:
        raise SystemExit(f"dataset lacks the pre-chunked 'actions' feature: {list(feats)}")
    chunk_shape = tuple(feats["actions"]["shape"])
    if len(chunk_shape) != 2 or chunk_shape[0] != chunk:
        raise SystemExit(f"'actions' shape {chunk_shape} != ({chunk}, D)")
    inputs = {}
    for key, ft in feats.items():
        if ft["dtype"] in ("image", "video"):
            h, w, c = ft["shape"]
            inputs[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(c, h, w))
        elif key == "observation.state":
            inputs[key] = PolicyFeature(type=FeatureType.STATE, shape=tuple(ft["shape"]))
    outputs = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(chunk_shape[1],))}
    return inputs, outputs


def remap_stats(stats: dict) -> dict:
    """Dataset stats keyed 'actions' -> policy normalization key 'action'.

    The dataset computes per-chunk-position stats with shape (10, 7); the
    policy's normalize/unnormalize buffers are built from the ACTION feature
    shape (7,), and from_pretrained() rebuilds them as (7,) at load time —
    per-position buffers would train fine but be unloadable at serving. Pool
    the positions (equal counts) into position-invariant (7,) stats: pooled
    mean = mean of position means; pooled variance via the law of total
    variance; min/max = elementwise envelope over positions.
    """
    import numpy as np

    out = dict(stats)
    if "actions" not in out:
        return out
    a = out.pop("actions")
    mean_p = np.asarray(a["mean"], dtype=np.float64)   # (chunk, D)
    std_p = np.asarray(a["std"], dtype=np.float64)
    if mean_p.ndim == 2:
        mean = mean_p.mean(axis=0)
        var = (std_p**2 + mean_p**2).mean(axis=0) - mean**2
        pooled = {
            "mean": mean.astype(np.float32),
            "std": np.sqrt(np.maximum(var, 0.0)).astype(np.float32),
            "min": np.asarray(a["min"]).min(axis=0),
            "max": np.asarray(a["max"]).max(axis=0),
        }
        if "count" in a:
            pooled["count"] = a["count"]
        out["action"] = pooled
    else:
        out["action"] = a
    return out


def build_policy(student: str, chunk: int, ds_meta, base_checkpoint: str | None, device: str):
    inputs, outputs = policy_features(ds_meta, chunk)
    stats = remap_stats(ds_meta.stats)
    if student == "act":
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy

        cfg = ACTConfig(
            chunk_size=chunk, n_action_steps=chunk,
            input_features=inputs, output_features=outputs,
        )
        policy = ACTPolicy(cfg, dataset_stats=stats)
    elif student == "smolvla":
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        if not base_checkpoint:
            raise SystemExit("smolvla requires a base checkpoint (recipe base_checkpoint)")
        cfg = SmolVLAConfig(
            chunk_size=chunk, n_action_steps=chunk,
            input_features=inputs, output_features=outputs,
        )
        base = SmolVLAPolicy.from_pretrained(base_checkpoint)
        # Base weights minus its normalization buffers (ours come from stats).
        sd = {k: v for k, v in base.state_dict().items()
              if not k.startswith(("normalize_", "unnormalize_"))}
        del base
        policy = SmolVLAPolicy(cfg, dataset_stats=stats)
        missing, unexpected = policy.load_state_dict(sd, strict=False)
        logger.info("smolvla base load: %d missing, %d unexpected (re-initialised)",
                    len(missing), len(unexpected))
    else:
        raise SystemExit(f"unknown student {student!r}")
    return policy.to(device)


def save_checkpoint(policy, out_dir: pathlib.Path, step: int) -> pathlib.Path:
    """0.3.3 layout: checkpoints/<step:06d>/pretrained_model (+ 'last' symlink)."""
    ck = out_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
    ck.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ck)
    last = out_dir / "checkpoints" / "last"
    if last.is_symlink() or last.exists():
        last.unlink()
    last.symlink_to(ck.parent.name)
    return ck


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True, choices=["act", "smolvla"])
    parser.add_argument("--dataset", required=True, help="lerobot dataset root")
    parser.add_argument("--out", required=True, help="checkpoint output dir")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp-bf16", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=7)
    # O4 second knob (data downsampling): train on only the first N frames.
    # The per-task ACT datasets store one lerobot-episode per task with frames
    # ordered by source episode, so a frame prefix == an episode prefix.
    parser.add_argument("--frames-limit", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    torch.manual_seed(args.seed)
    ds = LeRobotDataset(repo_id=str(args.dataset), root=args.dataset)  # no delta machinery
    policy = build_policy(args.student, args.chunk_size, ds.meta, args.base_checkpoint, args.device)
    policy.train()

    train_ds = ds
    if args.frames_limit and args.frames_limit < len(ds):
        train_ds = torch.utils.data.Subset(ds, list(range(args.frames_limit)))
        logger.info("frames-limit: training on first %d/%d frames", args.frames_limit, len(ds))
    dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        drop_last=True, pin_memory=(args.device == "cuda"),
    )
    optim = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=policy.config.optimizer_lr,
        weight_decay=policy.config.optimizer_weight_decay,
    )

    out_dir = pathlib.Path(args.out)
    step, it = 0, iter(dl)
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        batch = {k: (v.to(args.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        batch = adapt_batch(batch, args.chunk_size)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp_bf16):
            loss, _ = policy.forward(batch)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        optim.step()
        step += 1
        if step % args.log_every == 0 or step == 1:
            logger.info("step %d/%d loss %.5f", step, args.steps, loss.item())
        if step % args.save_every == 0:
            ck = save_checkpoint(policy, out_dir, step)
            logger.info("checkpoint saved: %s", ck)
    ck = save_checkpoint(policy, out_dir, step)
    logger.info("final checkpoint: %s", ck)
    print("TRAIN-STUDENT-DONE", args.student, args.out)


if __name__ == "__main__":
    main()
