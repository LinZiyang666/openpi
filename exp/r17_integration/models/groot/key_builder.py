"""Benchmark camera/state geometry over the stock mask-driven CP1 builder."""
from __future__ import annotations

import torch
from openpi.cache.groot.key_builder import GrootCP1SpatialPool16KeyBuilder


def validate_stage1(stage1, *, hidden_width=None, state_dim=4):
    state, mask = stage1.state, stage1.state_mask
    if state.shape != (1, 1, 64) or mask.shape != state.shape or mask.dtype != torch.bool:
        raise RuntimeError("native state/state_mask must be [1,1,64] bool mask")
    expected = torch.zeros_like(mask)
    expected[..., :state_dim] = True
    if not torch.equal(mask, expected):
        raise RuntimeError(f"native state_mask must select first{state_dim} channels only")
    if not torch.isfinite(state[..., :state_dim]).all():
        raise RuntimeError("MetaWorld state is not finite")
    embodiment = stage1.action_inputs["embodiment_id"]
    if embodiment.numel() != 1 or int(embodiment.reshape(-1)[0]) != 31:
        raise RuntimeError("MetaWorld embodiment must be metaworld_sawyer/slot31")
    embeds, image_mask = stage1.input_embeds, stage1.image_token_mask
    if embeds.ndim != 3 or embeds.shape[0] != 1 or image_mask.shape != embeds.shape[:2] or image_mask.dtype != torch.bool:
        raise RuntimeError("wrong image-token mask geometry")
    if hidden_width is not None and embeds.shape[-1] != hidden_width:
        raise RuntimeError("GR00T hidden width differs from feature spec")


class GrootMetaWorldCP1SpatialPool16KeyBuilder(GrootCP1SpatialPool16KeyBuilder):
    vision_fields = ("vision_0",)

    def __init__(self, enabled_fields=None, *, hidden_width=2048, benchmark="metaworld"):
        self.vision_fields = ("vision_0", "vision_1") if benchmark == "libero" else ("vision_0",)
        self.state_dim = 8 if benchmark == "libero" else 4
        enabled_fields = [*self.vision_fields, "robot_state"] if enabled_fields is None else enabled_fields
        if set(enabled_fields) - {*self.vision_fields, "robot_state", "prompt_emb"}:
            raise ValueError("MetaWorld has no vision_1/vision_2 retrieval field")
        super().__init__(enabled_fields)
        self.hidden_width = hidden_width

    def collect(self, checkpoint_id, **stage_outputs):
        if "stage1" in stage_outputs:
            validate_stage1(stage_outputs["stage1"], hidden_width=self.hidden_width, state_dim=self.state_dim)
        super().collect(checkpoint_id, **stage_outputs)


def make_cp2_builder(*, token_len, feature_dim, state_feat_dim, seed, d=500, p=.01):
    return GrootMetaWorldCP2KeyBuilder(token_len=token_len, feature_dim=feature_dim,
                                      state_feat_dim=state_feat_dim, seed=seed, d=d, p=p)


from openpi.cache.groot.cp2_key_builder import GrootCP2TernaryKeyBuilder


class GrootMetaWorldCP2KeyBuilder(GrootCP2TernaryKeyBuilder):
    """Optional encoded-conditioning hook; explicit MW geometry, never LIBERO defaults."""
    def __init__(self, *, token_len, feature_dim, state_feat_dim, seed, d=500, p=.01):
        super().__init__(seed, d=d, p=p, token_len=token_len, feature_dim=feature_dim, state_feat_dim=state_feat_dim)

    def projection_meta(self):
        meta = super().projection_meta()
        meta["layout"].update(kind="groot_mw_encoded_v1", real_cameras=["video.image"],
                              active_state_indices=[0, 1, 2, 3], embodiment_id=31)
        return meta
