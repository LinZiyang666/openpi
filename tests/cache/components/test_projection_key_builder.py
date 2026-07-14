"""Tests for the M1 outcome-compatible projection key builder (Phase 2).

Covers the identity non-regression anchor (no weights -> value-equal to the
inner pool builder), the weighted projection path (shape/dim self-consistency,
robot_state passthrough), the offline fit mechanism (InfoNCE on synthetic
data), ProjectionParams save/load, and the load-time shape validation.
"""

import pytest
import torch

from openpi.cache.components.key_builder import (
    CP1MaxPoolKeyBuilder,
    CP1MeanPoolKeyBuilder,
    CP1SpatialPool4KeyBuilder,
    CP1SpatialPool16KeyBuilder,
)
from openpi.cache.components.projection_key_builder import (
    FieldTrainingBatch,
    ProjectionHead,
    ProjectionKeyBuilder,
    ProjectionParams,
    infonce_loss,
    proj_infonce_loss,
    validate_projection_params,
)
from openpi.cache.types import (
    PROMPT_EMB,
    ROBOT_STATE,
    VISION_0,
    VISION_1,
    VISION_2,
    CheckpointID,
)

EMB_DIM = 2048
STATE_DIM = 32
NUM_PROMPT_TOKENS = 20

ALL_FIELDS = [VISION_0, VISION_1, VISION_2, PROMPT_EMB, ROBOT_STATE]


class _FakeStage1:
    def __init__(self, prefix_embs, state):
        self.prefix_embs = prefix_embs
        self.state = state


def _make_fake_stage1() -> _FakeStage1:
    prefix_len = 256 * 3 + NUM_PROMPT_TOKENS
    prefix_embs = torch.randn(1, prefix_len, EMB_DIM)
    state = torch.randn(1, STATE_DIM)
    return _FakeStage1(prefix_embs, state)


# Inner pool builder types under test, with their vision output dim.
_INNER_BUILDERS = {
    "cp1_mean_pool": (CP1MeanPoolKeyBuilder, EMB_DIM),
    "cp1_max_pool": (CP1MaxPoolKeyBuilder, EMB_DIM),
    "cp1_spatial_pool_16": (CP1SpatialPool16KeyBuilder, 16 * EMB_DIM),
    "cp1_spatial_pool_4": (CP1SpatialPool4KeyBuilder, 4 * EMB_DIM),
}


# ---------------------------------------------------------------------------
# Test 1 — identity non-regression golden
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("inner_name", sorted(_INNER_BUILDERS))
@pytest.mark.parametrize("cp", [CheckpointID.CP1, CheckpointID.CP3])
def test_identity_matches_inner_pool(inner_name, cp):
    cls, _ = _INNER_BUILDERS[inner_name]
    stage1 = _make_fake_stage1()

    standalone = cls(enabled_fields=ALL_FIELDS)
    standalone.collect(cp, stage1=stage1)
    expected = standalone.build(cp)

    wrapped = ProjectionKeyBuilder(cls(enabled_fields=ALL_FIELDS), params=None)
    wrapped.collect(cp, stage1=stage1)
    got = wrapped.build(cp)

    assert set(got) == set(expected)
    for field in expected:
        assert torch.equal(got[field], expected[field]), field


# ---------------------------------------------------------------------------
# Test 2 & 3 — weighted projection applies; output dim == head out_dim
# ---------------------------------------------------------------------------
def test_weighted_projection_applies_and_dims_match():
    out_dim = 64
    heads = {
        VISION_0: ProjectionHead(torch.randn(out_dim, EMB_DIM)),
        VISION_1: ProjectionHead(torch.randn(out_dim, EMB_DIM)),
        VISION_2: ProjectionHead(torch.randn(out_dim, EMB_DIM)),
        PROMPT_EMB: ProjectionHead(torch.randn(out_dim, EMB_DIM)),
    }
    builder = ProjectionKeyBuilder(
        CP1MeanPoolKeyBuilder(enabled_fields=ALL_FIELDS), ProjectionParams(heads)
    )
    builder.collect(CheckpointID.CP1, stage1=_make_fake_stage1())
    keys = builder.build(CheckpointID.CP1)

    for field in (VISION_0, VISION_1, VISION_2, PROMPT_EMB):
        assert keys[field].shape == (out_dim,), field
        assert keys[field].dtype == torch.float32
        assert keys[field].device.type == "cpu"
        assert keys[field].is_contiguous()
    # robot_state is never projected; keeps its raw state dim.
    assert keys[ROBOT_STATE].shape == (STATE_DIM,)


def test_projection_matches_manual_matmul():
    out_dim = 8
    weight = torch.randn(out_dim, EMB_DIM)
    bias = torch.randn(out_dim)
    params = ProjectionParams({VISION_0: ProjectionHead(weight, bias)})
    inner = CP1MeanPoolKeyBuilder(enabled_fields=[VISION_0, ROBOT_STATE])
    stage1 = _make_fake_stage1()

    ref = CP1MeanPoolKeyBuilder(enabled_fields=[VISION_0, ROBOT_STATE])
    ref.collect(CheckpointID.CP1, stage1=stage1)
    raw = ref.build(CheckpointID.CP1)[VISION_0]

    builder = ProjectionKeyBuilder(inner, params)
    builder.collect(CheckpointID.CP1, stage1=stage1)
    got = builder.build(CheckpointID.CP1)[VISION_0]

    assert torch.allclose(got, raw @ weight.t() + bias, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 4 — robot_state is never projected, even with a stray head
# ---------------------------------------------------------------------------
def test_robot_state_never_projected():
    # A head for robot_state must be ignored by build() (it is an L2 field).
    params = ProjectionParams({ROBOT_STATE: ProjectionHead(torch.randn(8, STATE_DIM))})
    inner = CP1MeanPoolKeyBuilder(enabled_fields=ALL_FIELDS)
    stage1 = _make_fake_stage1()

    ref = CP1MeanPoolKeyBuilder(enabled_fields=ALL_FIELDS)
    ref.collect(CheckpointID.CP1, stage1=stage1)
    expected_state = ref.build(CheckpointID.CP1)[ROBOT_STATE]

    builder = ProjectionKeyBuilder(inner, params)
    builder.collect(CheckpointID.CP1, stage1=stage1)
    got = builder.build(CheckpointID.CP1)

    assert torch.equal(got[ROBOT_STATE], expected_state)
    assert got[ROBOT_STATE].shape == (STATE_DIM,)


# ---------------------------------------------------------------------------
# Test 5 — offline fit mechanism lowers the InfoNCE loss (synthetic data)
# ---------------------------------------------------------------------------
def test_fit_lowers_infonce_loss_and_shapes():
    torch.manual_seed(0)
    in_dim, out_dim = 8, 4
    u = torch.zeros(in_dim)
    u[0] = 3.0
    v = torch.zeros(in_dim)
    v[1] = 3.0
    group_a = u + 0.1 * torch.randn(16, in_dim)
    group_b = v + 0.1 * torch.randn(16, in_dim)
    features = torch.cat([group_a, group_b], dim=0)
    labels = torch.tensor([0] * 16 + [1] * 16)
    batch = FieldTrainingBatch(features=features, group_labels=labels)

    # Baseline loss with a random (unfit) projection of the same shape.
    baseline = torch.randn(out_dim, in_dim) * (in_dim ** -0.5)
    loss_before = infonce_loss(features @ baseline.t(), labels, 0.07)

    params = ProjectionKeyBuilder.fit(
        {VISION_0: batch}, out_dim=out_dim, epochs=300, lr=0.05
    )
    head = params.head(VISION_0)
    assert head.weight.shape == (out_dim, in_dim)
    loss_after = infonce_loss(features @ head.weight.t(), labels, 0.07)

    assert loss_after < loss_before


def test_fit_rejects_non_projectable_field():
    batch = FieldTrainingBatch(
        features=torch.randn(4, STATE_DIM), group_labels=torch.tensor([0, 0, 1, 1])
    )
    with pytest.raises(ValueError, match="non-projectable"):
        ProjectionKeyBuilder.fit({ROBOT_STATE: batch}, out_dim=4)


# ---------------------------------------------------------------------------
# Test 5b — masked Eq-15 loss (TRACER §B): fit path, dispatch, gray-zone, invariants
# ---------------------------------------------------------------------------
def _masks_from_groups(labels: torch.Tensor):
    """Positives = same group off-diagonal; negatives = every cross-group pair."""
    same = labels.view(-1, 1) == labels.view(1, -1)
    eye = torch.eye(labels.shape[0], dtype=torch.bool)
    return (same & ~eye), (~same)


def test_fit_masked_loss_lowers_via_auto_and_explicit():
    in_dim, out_dim = 8, 4
    u = torch.zeros(in_dim)
    u[0] = 3.0
    v = torch.zeros(in_dim)
    v[1] = 3.0
    for loss_arg in ("auto", "masked"):
        torch.manual_seed(0)
        features = torch.cat(
            [u + 0.1 * torch.randn(16, in_dim), v + 0.1 * torch.randn(16, in_dim)], dim=0
        )
        labels = torch.tensor([0] * 16 + [1] * 16)
        pos, neg = _masks_from_groups(labels)
        batch = FieldTrainingBatch(features=features, pos_mask=pos, neg_mask=neg)
        baseline = torch.randn(out_dim, in_dim) * (in_dim ** -0.5)
        before = proj_infonce_loss(features @ baseline.t(), pos, neg, 0.07)
        params = ProjectionKeyBuilder.fit(
            {VISION_0: batch}, out_dim=out_dim, epochs=300, lr=0.05, loss=loss_arg
        )
        after = proj_infonce_loss(features @ params.head(VISION_0).weight.t(), pos, neg, 0.07)
        assert after < before, loss_arg


def test_auto_dispatch_group_labels_is_backward_compatible():
    # A group_labels-only batch under the default loss="auto" must be byte-identical
    # to the legacy "group" path: same init seed -> same fitted weights.
    feats = torch.randn(12, 8)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    torch.manual_seed(7)
    p_auto = ProjectionKeyBuilder.fit(
        {VISION_0: FieldTrainingBatch(features=feats, group_labels=labels)}, out_dim=4, epochs=10
    )
    torch.manual_seed(7)
    p_group = ProjectionKeyBuilder.fit(
        {VISION_0: FieldTrainingBatch(features=feats, group_labels=labels)},
        out_dim=4,
        epochs=10,
        loss="group",
    )
    assert torch.allclose(p_auto.head(VISION_0).weight, p_group.head(VISION_0).weight)


@pytest.mark.parametrize("loss", ["auto", "masked", "group"])
def test_both_signal_families_rejected_for_every_selector(loss):
    # masks + group labels together is ambiguous for auto AND the explicit selectors.
    feats = torch.randn(4, 8)
    labels = torch.tensor([0, 0, 1, 1])
    pos, neg = _masks_from_groups(labels)
    with pytest.raises(ValueError, match="(?i)(both|ambiguous)"):
        ProjectionKeyBuilder.fit(
            {VISION_0: FieldTrainingBatch(features=feats, group_labels=labels, pos_mask=pos, neg_mask=neg)},
            out_dim=4,
            loss=loss,
        )


def test_partial_mask_rejected_not_silently_grouped():
    # Only pos_mask given (no neg) + group labels -> malformed, not a silent group fallback.
    feats = torch.randn(4, 8)
    labels = torch.tensor([0, 0, 1, 1])
    pos, _ = _masks_from_groups(labels)
    with pytest.raises(ValueError, match="partial mask"):
        ProjectionKeyBuilder.fit(
            {VISION_0: FieldTrainingBatch(features=feats, group_labels=labels, pos_mask=pos)}, out_dim=4
        )


def test_mask_device_mismatch_rejected():
    if not torch.cuda.is_available():
        pytest.skip("needs a second device to test mask/feature device mismatch")
    feats = torch.randn(4, 8, device="cuda")
    labels = torch.tensor([0, 0, 1, 1])
    pos, neg = _masks_from_groups(labels)  # CPU masks
    with pytest.raises(ValueError, match="device"):
        ProjectionKeyBuilder.fit(
            {VISION_0: FieldTrainingBatch(features=feats, pos_mask=pos, neg_mask=neg)}, out_dim=4, loss="masked"
        )


def test_loss_dispatch_errors():
    feats = torch.randn(4, 8)
    labels = torch.tensor([0, 0, 1, 1])
    pos, neg = _masks_from_groups(labels)
    with pytest.raises(ValueError, match="neither"):
        ProjectionKeyBuilder.fit({VISION_0: FieldTrainingBatch(features=feats)}, out_dim=4)
    with pytest.raises(ValueError, match="requires pos_mask"):
        ProjectionKeyBuilder.fit(
            {VISION_0: FieldTrainingBatch(features=feats, group_labels=labels)}, out_dim=4, loss="masked"
        )
    with pytest.raises(ValueError, match="requires group_labels"):
        ProjectionKeyBuilder.fit(
            {VISION_0: FieldTrainingBatch(features=feats, pos_mask=pos, neg_mask=neg)}, out_dim=4, loss="group"
        )


def test_proj_infonce_gray_zone_rows_excluded():
    # Operational proof of Eq-15 vs the supervised proxy: rows that are in NEITHER
    # mask (gray zone) must receive zero gradient from proj_infonce_loss, whereas
    # the all-off-diagonal-negative infonce_loss touches them.
    torch.manual_seed(1)
    n, d = 8, 5
    z = torch.randn(n, d, requires_grad=True)
    pos = torch.zeros(n, n, dtype=torch.bool)
    neg = torch.zeros(n, n, dtype=torch.bool)
    pos[0, 1] = pos[1, 0] = True  # anchor 0 positive = {1}
    neg[0, 2] = neg[2, 0] = True  # anchor 0 negative = {2}; rows 3..7 are gray zone
    grad_masked = torch.autograd.grad(proj_infonce_loss(z, pos, neg, 0.07), z)[0]
    assert torch.allclose(grad_masked[3:], torch.zeros_like(grad_masked[3:]))

    z2 = z.detach().clone().requires_grad_(True)
    labels = torch.tensor([0, 0, 1, 2, 3, 4, 5, 6])
    grad_group = torch.autograd.grad(infonce_loss(z2, labels, 0.07), z2)[0]
    assert not torch.allclose(grad_group[3:], torch.zeros_like(grad_group[3:]))


def test_validate_masks_invariants():
    feats = torch.randn(4, 8)
    pos = torch.zeros(4, 4, dtype=torch.bool)
    pos[0, 1] = pos[1, 0] = True
    neg = torch.zeros(4, 4, dtype=torch.bool)
    neg[0, 2] = neg[2, 0] = neg[0, 3] = neg[3, 0] = True

    def _fit(p, ng):
        ProjectionKeyBuilder.fit(
            {VISION_0: FieldTrainingBatch(features=feats, pos_mask=p, neg_mask=ng)}, out_dim=4, loss="masked"
        )

    asym = pos.clone()
    asym[2, 3] = True  # no mirror -> not symmetric
    with pytest.raises(ValueError, match="symmetric"):
        _fit(asym, neg)
    diag = pos.clone()
    diag[0, 0] = True
    with pytest.raises(ValueError, match="diagonal"):
        _fit(diag, neg)
    overlap = neg.clone()
    overlap[0, 1] = overlap[1, 0] = True  # pair (0,1) is also a positive
    with pytest.raises(ValueError, match="disjoint"):
        _fit(pos, overlap)
    with pytest.raises(ValueError, match="bool"):
        _fit(pos.float(), neg)


# ---------------------------------------------------------------------------
# Test 6 — ProjectionParams save/load round trip
# ---------------------------------------------------------------------------
def test_params_save_load_roundtrip(tmp_path):
    params = ProjectionParams(
        {
            VISION_0: ProjectionHead(torch.randn(16, EMB_DIM), torch.randn(16)),
            PROMPT_EMB: ProjectionHead(torch.randn(16, EMB_DIM), None),
        }
    )
    path = tmp_path / "proj.pt"
    params.save(path)
    loaded = ProjectionParams.load(path)

    assert set(loaded.heads) == {VISION_0, PROMPT_EMB}
    assert torch.allclose(loaded.head(VISION_0).weight, params.head(VISION_0).weight)
    assert torch.allclose(loaded.head(VISION_0).bias, params.head(VISION_0).bias)
    assert loaded.head(PROMPT_EMB).bias is None


# ---------------------------------------------------------------------------
# Test 7 — validate_projection_params fails loud on shape / field errors
# ---------------------------------------------------------------------------
def _valid_dims():
    # cp1_mean_pool inner: vision/prompt in=2048, projected out=64; state raw 32.
    return {VISION_0: 64, PROMPT_EMB: 64, ROBOT_STATE: STATE_DIM}


def test_validate_accepts_well_formed():
    params = ProjectionParams(
        {
            VISION_0: ProjectionHead(torch.randn(64, EMB_DIM)),
            PROMPT_EMB: ProjectionHead(torch.randn(64, EMB_DIM), torch.randn(64)),
        }
    )
    validate_projection_params(params, "cp1_mean_pool", _valid_dims())  # no raise


def test_validate_none_is_noop():
    validate_projection_params(None, "cp1_mean_pool", _valid_dims())


def test_validate_rejects_bad_out_dim():
    params = ProjectionParams({VISION_0: ProjectionHead(torch.randn(32, EMB_DIM))})
    with pytest.raises(ValueError, match="projection weight"):
        validate_projection_params(params, "cp1_mean_pool", _valid_dims())


def test_validate_rejects_bad_in_dim():
    params = ProjectionParams({VISION_0: ProjectionHead(torch.randn(64, 999))})
    with pytest.raises(ValueError, match="projection weight"):
        validate_projection_params(params, "cp1_mean_pool", _valid_dims())


def test_validate_rejects_bad_bias():
    params = ProjectionParams(
        {VISION_0: ProjectionHead(torch.randn(64, EMB_DIM), torch.randn(32))}
    )
    with pytest.raises(ValueError, match="projection bias"):
        validate_projection_params(params, "cp1_mean_pool", _valid_dims())


def test_validate_rejects_robot_state_head():
    params = ProjectionParams({ROBOT_STATE: ProjectionHead(torch.randn(64, STATE_DIM))})
    with pytest.raises(ValueError, match="non-projectable"):
        validate_projection_params(params, "cp1_mean_pool", _valid_dims())


def test_validate_rejects_passthrough_dim_mismatch():
    # robot_state has no head (passthrough) but declared dim != inner dim.
    dims = {VISION_0: 64, ROBOT_STATE: 99}
    params = ProjectionParams({VISION_0: ProjectionHead(torch.randn(64, EMB_DIM))})
    with pytest.raises(ValueError, match="passthrough"):
        validate_projection_params(params, "cp1_mean_pool", dims)
